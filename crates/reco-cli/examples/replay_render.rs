//! OEV visual A/B render harness (test only, not a production path).
//!
//! Renders the real OEV broadcast (two-camera stitch -> trackers -> field
//! panner -> lookahead smoothing -> encode) through `reco_io::StitchJob`,
//! with ONE substitution: the YOLO detector is replaced by a replay detector
//! that returns, frame by frame and camera by camera, the exact detections a
//! previous GPU run recorded in `events.jsonl` (`detections_raw`).
//!
//! Why: re-running YOLO26m on CPU would take hours and would not reproduce
//! the GPU detections. Replaying them makes the tracker the only difference
//! between two renders built from two video-stitcher refs.
//!
//! Recorded `detections_raw` are the post-ROI-wrapper output, so the ROI
//! wrapper is NOT applied again (its ball test shifts cy by +0.40 and is not
//! idempotent). Mapping to panorama uses the local calibration (`-c`).
//!
//! Wiring mirrors `reco_autocam::setup_autocam`, `TrackingMode::Field`, frame
//! stride 1: `BallTracker::new(32).with_max_jump_rad(0.8)
//! .with_max_coast_frames(20)`, `ClassProvider::new(0)` players,
//! `FieldPanner::with_config(fps, cfg)` (stride-1 rebase is the identity),
//! detection interval 1.
//!
//! Usage:
//! ```text
//! replay_render --left L.mp4 --right R.mp4 -c match.json --replay events.jsonl \
//!   -o out.mp4 --events out_events.jsonl [--panner-preset v4_micro_damping] \
//!   [--panner-config overlay.json] [--lookahead 1.5] [--max-frames N] \
//!   [--width 1920 --height 1080]
//! ```

use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use reco_core::detect::detector::{
    CameraId, Detection, DetectorError, DetectorFrame, UnifiedDetector,
};
use reco_core::detect::director::MappedDetection;

/// Per-camera queues of recorded detections, one entry per analysis frame.
struct ReplayDetector {
    left: VecDeque<Vec<Detection>>,
    right: VecDeque<Vec<Detection>>,
    served: Arc<AtomicU64>,
    overrun: Arc<AtomicU64>,
}

impl UnifiedDetector for ReplayDetector {
    fn name(&self) -> &'static str {
        "replay"
    }

    fn detect(
        &mut self,
        camera: CameraId,
        _frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        let queue = match camera {
            CameraId::Left => &mut self.left,
            CameraId::Right => &mut self.right,
        };
        match queue.pop_front() {
            Some(d) => {
                if camera == CameraId::Left {
                    self.served.fetch_add(1, Ordering::Relaxed);
                }
                Ok(d)
            }
            None => {
                self.overrun.fetch_add(1, Ordering::Relaxed);
                Ok(Vec::new())
            }
        }
    }
}

fn load_replay(path: &str) -> anyhow::Result<(VecDeque<Vec<Detection>>, VecDeque<Vec<Detection>>)> {
    let file = std::fs::File::open(path)?;
    let mut left = VecDeque::new();
    let mut right = VecDeque::new();
    let mut expected = 0u64;
    for line in BufReader::new(file).lines() {
        let line = line?;
        if !line.contains("\"detections_raw\"") {
            continue;
        }
        let ev: serde_json::Value = serde_json::from_str(&line)?;
        if ev["kind"] != "detections_raw" {
            continue;
        }
        let index = ev["frame_index"].as_u64().unwrap_or(u64::MAX);
        let detections: Vec<MappedDetection> = serde_json::from_value(ev["detections"].clone())?;
        anyhow::ensure!(index == expected, "detections_raw out of order at {index}");
        expected += 1;
        let mut l = Vec::new();
        let mut r = Vec::new();
        for m in detections {
            let d = Detection {
                camera: m.camera,
                class_id: m.class_id,
                confidence: m.confidence,
                center_x: m.camera_center.0,
                center_y: m.camera_center.1,
                width: m.camera_size.0,
                height: m.camera_size.1,
            };
            match m.camera {
                CameraId::Left => l.push(d),
                CameraId::Right => r.push(d),
            }
        }
        left.push_back(l);
        right.push_back(r);
    }
    Ok((left, right))
}

struct Args {
    left: String,
    right: String,
    calibration: String,
    replay: String,
    output: String,
    events: Option<String>,
    preset: String,
    panner_config: Option<String>,
    lookahead: f64,
    max_frames: Option<u64>,
    width: u32,
    height: u32,
}

fn parse_args() -> anyhow::Result<Args> {
    let mut a = Args {
        left: String::new(),
        right: String::new(),
        calibration: String::new(),
        replay: String::new(),
        output: String::new(),
        events: None,
        preset: "v4_micro_damping".into(),
        panner_config: None,
        lookahead: 1.5,
        max_frames: None,
        width: 1920,
        height: 1080,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut v = || it.next().ok_or_else(|| anyhow::anyhow!("missing value for {flag}"));
        match flag.as_str() {
            "--left" => a.left = v()?,
            "--right" => a.right = v()?,
            "-c" | "--calibration" => a.calibration = v()?,
            "--replay" => a.replay = v()?,
            "-o" | "--output" => a.output = v()?,
            "--events" => a.events = Some(v()?),
            "--panner-preset" => a.preset = v()?,
            "--panner-config" => a.panner_config = Some(v()?),
            "--lookahead" => a.lookahead = v()?.parse()?,
            "--max-frames" => a.max_frames = Some(v()?.parse()?),
            "--width" => a.width = v()?.parse()?,
            "--height" => a.height = v()?.parse()?,
            other => anyhow::bail!("unknown flag {other}"),
        }
    }
    anyhow::ensure!(
        !a.left.is_empty() && !a.right.is_empty() && !a.calibration.is_empty(),
        "--left, --right and -c are required"
    );
    anyhow::ensure!(!a.replay.is_empty() && !a.output.is_empty(), "--replay and -o are required");
    Ok(a)
}

fn panner_config(
    preset: &str,
    overlay: Option<&str>,
) -> anyhow::Result<reco_autocam::panners::FieldPannerConfig> {
    use reco_autocam::panners::FieldPannerConfig;
    let base = FieldPannerConfig::from_preset_name(preset)
        .ok_or_else(|| anyhow::anyhow!("unknown panner preset {preset}"))?;
    let Some(path) = overlay else {
        return Ok(base);
    };
    let mut v = serde_json::to_value(&base)?;
    let over: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    if let (Some(bm), serde_json::Value::Object(om)) = (v.as_object_mut(), over) {
        bm.extend(om);
    }
    Ok(serde_json::from_value(v)?)
}

fn main() -> anyhow::Result<()> {
    env_logger_init();
    let args = parse_args()?;
    let (left, right) = load_replay(&args.replay)?;
    let recorded_frames = left.len() as u64;
    println!("replay: {recorded_frames} recorded analysis frames from {}", args.replay);
    let served = Arc::new(AtomicU64::new(0));
    let overrun = Arc::new(AtomicU64::new(0));
    let detector = ReplayDetector {
        left,
        right,
        served: Arc::clone(&served),
        overrun: Arc::clone(&overrun),
    };
    let cfg = panner_config(&args.preset, args.panner_config.as_deref())?;
    println!(
        "panner: preset {} overlay {:?} -> {}",
        args.preset,
        args.panner_config,
        serde_json::to_string(&cfg)?
    );

    let cal = reco_core::calibration::MatchCalibration::from_file(std::path::Path::new(
        &args.calibration,
    ))?;
    let mut job = reco_io::StitchJob::with_calibration(
        reco_io::stitch_job::InputPath::Single(PathBuf::from(&args.left)),
        reco_io::stitch_job::InputPath::Single(PathBuf::from(&args.right)),
        cal,
        &args.output,
    )
    .codec(reco_io::output::Codec::H264)
    .quality(reco_io::output::Quality::Balanced)
    .resolution(args.width, args.height)
    .blend_width(0.15)
    .force_cpu_decode()
    .lookahead(args.lookahead)
    .on_progress(|p: &reco_core::session::types::FrameProgress| {
        if p.frames_completed % 300 == 0 {
            let fps = p.frames_completed as f64 / p.elapsed.as_secs_f64().max(1e-9);
            eprintln!("progress: {} frames, {:.2} fps", p.frames_completed, fps);
        }
    });
    if let Some(ref e) = args.events {
        job = job.events(e);
    }
    if let Some(n) = args.max_frames {
        job = job.max_frames(n);
    }
    let mut detector = Some(detector);
    job = job.on_session(move |session, source| {
        let fps = source.info().fps as f32;
        session.set_frame_stride(1);
        session.set_detector(Box::new(detector.take().expect("single session")));
        session.set_detection_interval(1);
        session.set_ball_tracker(Box::new(
            reco_autocam::trackers::BallTracker::new(32)
                .with_max_jump_rad(0.8)
                .with_max_coast_frames(reco_autocam::trackers::ball::DEFAULT_COAST_FRAMES),
        ));
        session.set_player_tracker(Box::new(reco_autocam::trackers::ClassProvider::new(0)));
        session.set_panner(Box::new(reco_autocam::panners::FieldPanner::with_config(
            fps, cfg,
        )));
        println!("session: field tracking wired (fps {fps:.4}), replay detector attached");
    });

    let interrupted = AtomicBool::new(false);
    let result = job.run(&interrupted)?;
    println!(
        "done: {} frames in {:.1}s ({:.2} fps) -> {}; replay frames served {} / {}, overrun calls {}",
        result.frames_processed,
        result.elapsed.as_secs_f64(),
        result.fps(),
        args.output,
        served.load(Ordering::Relaxed),
        recorded_frames,
        overrun.load(Ordering::Relaxed)
    );
    Ok(())
}

fn env_logger_init() {
    struct Stderr;
    impl log::Log for Stderr {
        fn enabled(&self, m: &log::Metadata) -> bool {
            m.level() <= log::Level::Info
        }
        fn log(&self, r: &log::Record) {
            let msg = r.args().to_string();
            // Keep the console readable: tracker transitions + warnings.
            if r.level() <= log::Level::Warn
                || msg.contains("dormant static site")
                || msg.contains("reactivated static site")
                || msg.contains("Analysis clock")
                || msg.contains("Lookahead")
            {
                eprintln!("[{}] {msg}", r.level());
            }
        }
        fn flush(&self) {}
    }
    static L: Stderr = Stderr;
    let _ = log::set_logger(&L);
    log::set_max_level(log::LevelFilter::Info);
}
