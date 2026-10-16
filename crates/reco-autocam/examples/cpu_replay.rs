//! CPU replay of recorded OEV pipeline events through the real [`BallTracker`].
//!
//! Reads an `events.jsonl` produced by the stitch session's pipeline event
//! sink, and for every recorded frame feeds the recorded `detections_raw`
//! (already mapped through the detector wrappers, including ROI filtering -
//! this is the post-wrapper tracker input, NOT a pre-ROI detector cache) and
//! the recorded player tracker output (`world_state.players`) into a freshly
//! constructed `BallTracker` built exactly as `reco-autocam` field mode builds
//! it (`max_jump_rad = 0.8`, `coast = DEFAULT_COAST_FRAMES` at stride 1).
//!
//! No tracker rule is reimplemented here. The harness only:
//! - deserializes recorded inputs,
//! - calls `observe_world` then `update` in the same order as
//!   `reco_core::detect::panner::dispatch_detect_only`,
//! - captures the tracker's own `log` output to observe dormant-site and
//!   release transitions,
//! - compares tracker-stage output against the recorded post-bridge
//!   `world_state.ball` with bridge-aware rules.
//!
//! Clock modes (`--clock`):
//! - `historical`: the recorded `world_state.timestamp_ms`, i.e. exactly the
//!   value the tracker received during the original run (faithful replay).
//! - `media`: `frame_index * fps_den * 1000 / fps_num` (source video time),
//!   a TEST COMPARISON ONLY. It does not change production time handling.
//!
//! Replay wall time is never used as tracker input; it is only reported as
//! processing-time profiling.
//!
//! Usage:
//! ```text
//! cargo run --release -p reco-autocam --no-default-features --example cpu_replay -- \
//!     --events run.jsonl --clock historical --out report.json [--trace trace.csv]
//!     [--fps-num 60000 --fps-den 1001] [--throttle-us 0] [--label name]
//! ```

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::sync::Mutex;
use std::time::Instant;

use reco_autocam::trackers::BallTracker;
use reco_core::detect::director::MappedDetection;
use reco_core::detect::tracker::{TrackState, TrackedEntity, Tracker, WorldState};

/// Field-mode configuration as built in `reco_autocam::lib` (TrackingMode::Field,
/// stride 1). Kept explicit and printed in every report.
const BALL_CLASS_ID: u16 = 32;
const FIELD_MAX_JUMP_RAD: f32 = 0.8;
const COAST_FRAMES_STRIDE1: u32 = reco_autocam::trackers::ball::DEFAULT_COAST_FRAMES;

/// Known decoy (spare ball) location from the September investigation.
const DECOY_YAW: f32 = 0.967;
const DECOY_PITCH: f32 = 0.0075;
const DECOY_RADIUS_RAD: f32 = 0.03;
/// Numeric roundoff tolerance for position parity.
const POS_TOL_RAD: f32 = 1e-6;

// ---------------------------------------------------------------- log capture

/// One tracker log line, tagged with the replay frame it occurred on.
#[derive(Debug, Clone)]
struct Captured {
    frame: u64,
    message: String,
}

static CAPTURE: Mutex<Vec<Captured>> = Mutex::new(Vec::new());
static CURRENT_FRAME: Mutex<u64> = Mutex::new(0);

struct CaptureLogger;

impl log::Log for CaptureLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record) {
        let message = record.args().to_string();
        if message.starts_with("BallTracker:") || message.starts_with("DIAG_RELEASE_SITE") {
            let frame = *CURRENT_FRAME.lock().expect("frame lock");
            CAPTURE
                .lock()
                .expect("capture lock")
                .push(Captured { frame, message });
        }
    }
    fn flush(&self) {}
}

static LOGGER: CaptureLogger = CaptureLogger;

// ---------------------------------------------------------------- input

#[derive(serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Event {
    FrameStart {
        frame_index: u64,
    },
    DetectionsRaw {
        frame_index: u64,
        detections: Vec<MappedDetection>,
    },
    WorldState {
        frame_index: u64,
        timestamp_ms: f64,
        players: Vec<TrackedEntity>,
        ball: Option<TrackedEntity>,
    },
    #[serde(other)]
    Other,
}

#[derive(Default)]
struct RecordedFrame {
    detections: Option<Vec<MappedDetection>>,
    world: Option<(f64, Vec<TrackedEntity>, Option<TrackedEntity>)>,
    has_frame_start: bool,
}

struct Args {
    events: String,
    clock: String,
    out: String,
    trace: Option<String>,
    fps_num: u64,
    fps_den: u64,
    throttle_us: u64,
    label: String,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        events: String::new(),
        clock: "historical".into(),
        out: "replay_report.json".into(),
        trace: None,
        fps_num: 60000,
        fps_den: 1001,
        throttle_us: 0,
        label: String::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("missing value for {flag}"));
        match flag.as_str() {
            "--events" => args.events = value()?,
            "--clock" => args.clock = value()?,
            "--out" => args.out = value()?,
            "--trace" => args.trace = Some(value()?),
            "--fps-num" => args.fps_num = value()?.parse().map_err(|e| format!("{e}"))?,
            "--fps-den" => args.fps_den = value()?.parse().map_err(|e| format!("{e}"))?,
            "--throttle-us" => args.throttle_us = value()?.parse().map_err(|e| format!("{e}"))?,
            "--label" => args.label = value()?,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if args.events.is_empty() {
        return Err("--events is required".into());
    }
    if args.clock != "historical" && args.clock != "media" {
        return Err("--clock must be historical or media".into());
    }
    if args.fps_num == 0 || args.fps_den == 0 {
        return Err("fps must be non-zero".into());
    }
    Ok(args)
}

fn load(path: &str) -> Result<Vec<RecordedFrame>, String> {
    let file = File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let mut frames: Vec<RecordedFrame> = Vec::new();
    let mut last_index: Option<u64> = None;
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("read line {line_no}: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let event: Event =
            serde_json::from_str(&line).map_err(|e| format!("parse line {line_no}: {e}"))?;
        let index = match &event {
            Event::FrameStart { frame_index }
            | Event::DetectionsRaw { frame_index, .. }
            | Event::WorldState { frame_index, .. } => *frame_index,
            Event::Other => continue,
        };
        if let Some(prev) = last_index {
            if index < prev {
                return Err(format!(
                    "frame ordering violated at line {line_no}: {index} after {prev}"
                ));
            }
        }
        last_index = Some(index);
        let slot = index as usize;
        if frames.len() <= slot {
            frames.resize_with(slot + 1, RecordedFrame::default);
        }
        let frame = &mut frames[slot];
        match event {
            Event::FrameStart { .. } => frame.has_frame_start = true,
            Event::DetectionsRaw { detections, .. } => {
                if frame.detections.replace(detections).is_some() {
                    return Err(format!("duplicate detections_raw for frame {index}"));
                }
            }
            Event::WorldState {
                timestamp_ms,
                players,
                ball,
                ..
            } => {
                if frame.world.replace((timestamp_ms, players, ball)).is_some() {
                    return Err(format!("duplicate world_state for frame {index}"));
                }
            }
            Event::Other => {}
        }
    }
    for (i, f) in frames.iter().enumerate() {
        if !f.has_frame_start || f.detections.is_none() || f.world.is_none() {
            return Err(format!("frame {i} incomplete"));
        }
    }
    Ok(frames)
}

// ---------------------------------------------------------------- analysis

fn state_name(ball: Option<&TrackedEntity>) -> &'static str {
    match ball.map(|b| b.state) {
        None => "None",
        Some(TrackState::Tracking) => "Tracking",
        Some(TrackState::Coasting) => "Coasting",
        Some(TrackState::Lost) => "Lost",
        Some(TrackState::Bridged) => "Bridged",
    }
}

fn same_pos(a: &TrackedEntity, b: &TrackedEntity) -> bool {
    (a.yaw - b.yaw).abs() <= POS_TOL_RAD && (a.pitch - b.pitch).abs() <= POS_TOL_RAD
}

/// Bridge-aware tracker-stage parity. Backward bridging only rewrites a
/// tracker `None` or `Coasting` into `Bridged`, so those pairings are
/// consistent; anything else must match exactly.
fn parity_class(replay: Option<&TrackedEntity>, recorded: Option<&TrackedEntity>) -> &'static str {
    match (replay, recorded) {
        (None, None) => "match",
        (None, Some(r)) if r.state == TrackState::Bridged => "match_bridged",
        (Some(p), Some(r)) if p.state == TrackState::Coasting && r.state == TrackState::Bridged => {
            "match_bridged"
        }
        (Some(p), Some(r)) if p.state == r.state => {
            if same_pos(p, r) {
                "match"
            } else {
                "mismatch_position"
            }
        }
        _ => "mismatch_state",
    }
}

fn near_decoy(e: &TrackedEntity) -> bool {
    let dy = e.yaw - DECOY_YAW;
    let dp = e.pitch - DECOY_PITCH;
    (dy * dy + dp * dp).sqrt() <= DECOY_RADIUS_RAD
}

/// Episodes of consecutive frames satisfying a predicate, in frames.
fn episodes(flags: &[bool]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, &f) in flags.iter().enumerate() {
        match (f, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                out.push((s, i - s));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push((s, flags.len() - s));
    }
    out
}

fn episode_summary(flags: &[bool], frame_s: f64) -> serde_json::Value {
    let eps = episodes(flags);
    let total: usize = eps.iter().map(|e| e.1).sum();
    let longest = eps.iter().map(|e| e.1).max().unwrap_or(0);
    serde_json::json!({
        "episodes": eps.len(),
        "frames": total,
        "total_s": total as f64 * frame_s,
        "longest_s": longest as f64 * frame_s,
        "first_frames": eps.iter().take(5).map(|e| e.0).collect::<Vec<_>>(),
    })
}

/// FNV-1a over the decision trace for cheap determinism comparison.
fn fnv1a(hash: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *hash ^= *b as u64;
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn peak_rss_kb() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
}

/// Parse `yaw=… pitch=…` and the trailing `(reason)` from a tracker log line.
fn parse_site(message: &str) -> (Option<f32>, Option<f32>, Option<String>) {
    let grab = |key: &str| {
        message
            .split_whitespace()
            .find_map(|tok| tok.strip_prefix(key))
            .and_then(|v| v.parse::<f32>().ok())
    };
    let reason = message
        .rfind('(')
        .zip(message.rfind(')'))
        .map(|(a, b)| message[a + 1..b].to_string());
    (grab("yaw="), grab("pitch="), reason)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("cpu_replay: {e}");
            std::process::exit(2);
        }
    };
    log::set_logger(&LOGGER).expect("logger already set");
    log::set_max_level(log::LevelFilter::Info);

    let load_t0 = Instant::now();
    let frames = match load(&args.events) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("cpu_replay: invalid input: {e}");
            std::process::exit(1);
        }
    };
    let load_s = load_t0.elapsed().as_secs_f64();

    let mut tracker = BallTracker::new(BALL_CLASS_ID)
        .with_max_jump_rad(FIELD_MAX_JUMP_RAD)
        .with_max_coast_frames(COAST_FRAMES_STRIDE1);

    let frame_ms = 1000.0 * args.fps_den as f64 / args.fps_num as f64;
    let mut trace_out = args
        .trace
        .as_ref()
        .map(|p| BufWriter::new(File::create(p).expect("create trace")));
    if let Some(w) = trace_out.as_mut() {
        writeln!(w, "frame,input_ms,replay_state,replay_yaw,replay_pitch,replay_cam,recorded_state,recorded_yaw,recorded_pitch,recorded_cam,parity").ok();
    }

    let mut replay_states: BTreeMap<&str, usize> = BTreeMap::new();
    let mut recorded_states: BTreeMap<&str, usize> = BTreeMap::new();
    let mut parity: BTreeMap<&str, usize> = BTreeMap::new();
    let mut first_mismatch: Option<serde_json::Value> = None;
    let mut ball_candidates = 0usize;
    let mut frames_with_candidate = 0usize;
    let mut decoy_tracker_tracking = Vec::with_capacity(frames.len());
    let mut decoy_recorded_tracking = Vec::with_capacity(frames.len());
    let mut decoy_recorded_any = Vec::with_capacity(frames.len());
    let mut decoy_recorded_bridged = Vec::with_capacity(frames.len());
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut prev_input_ms = f64::NEG_INFINITY;
    let mut nonmonotonic_inputs = 0usize;

    let replay_t0 = Instant::now();
    for (i, frame) in frames.iter().enumerate() {
        *CURRENT_FRAME.lock().expect("frame lock") = i as u64;
        let detections = frame.detections.as_ref().expect("validated");
        let (recorded_ms, players, recorded_ball) = frame.world.as_ref().expect("validated");
        let input_ms = if args.clock == "historical" {
            *recorded_ms
        } else {
            i as f64 * frame_ms
        };
        if input_ms <= prev_input_ms {
            nonmonotonic_inputs += 1;
        }
        prev_input_ms = input_ms;

        let n = detections
            .iter()
            .filter(|d| d.class_id == BALL_CLASS_ID && d.position.is_some())
            .count();
        ball_candidates += n;
        frames_with_candidate += usize::from(n > 0);

        // Same order as dispatch_detect_only: players first (recorded player
        // tracker output), then ball observe_world + update.
        let world = WorldState {
            ball: None,
            players: players.clone(),
        };
        tracker.observe_world(&world);
        let out = tracker.update(detections, input_ms);
        let replay_ball = out.first();

        *replay_states.entry(state_name(replay_ball)).or_default() += 1;
        *recorded_states
            .entry(state_name(recorded_ball.as_ref()))
            .or_default() += 1;
        let p = parity_class(replay_ball, recorded_ball.as_ref());
        *parity.entry(p).or_default() += 1;
        if p.starts_with("mismatch") && first_mismatch.is_none() {
            first_mismatch = Some(serde_json::json!({
                "frame": i,
                "class": p,
                "replay": replay_ball.map(|b| (state_name(Some(b)), b.yaw, b.pitch)),
                "recorded": recorded_ball.as_ref().map(|b| (state_name(Some(b)), b.yaw, b.pitch)),
            }));
        }

        decoy_tracker_tracking
            .push(replay_ball.is_some_and(|b| b.state == TrackState::Tracking && near_decoy(b)));
        decoy_recorded_tracking.push(
            recorded_ball
                .as_ref()
                .is_some_and(|b| b.state == TrackState::Tracking && near_decoy(b)),
        );
        decoy_recorded_any.push(recorded_ball.as_ref().is_some_and(near_decoy));
        decoy_recorded_bridged.push(
            recorded_ball
                .as_ref()
                .is_some_and(|b| b.state == TrackState::Bridged && near_decoy(b)),
        );

        let state = state_name(replay_ball);
        fnv1a(&mut hash, state.as_bytes());
        if let Some(b) = replay_ball {
            fnv1a(&mut hash, &b.yaw.to_bits().to_le_bytes());
            fnv1a(&mut hash, &b.pitch.to_bits().to_le_bytes());
        }

        if let Some(w) = trace_out.as_mut() {
            let fmt = |b: Option<&TrackedEntity>| match b {
                Some(b) => format!(
                    "{},{:.7},{:.7},{:?}",
                    state_name(Some(b)),
                    b.yaw,
                    b.pitch,
                    b.origin
                ),
                None => "None,,,".to_string(),
            };
            writeln!(
                w,
                "{i},{input_ms:.6},{},{},{p}",
                fmt(replay_ball),
                fmt(recorded_ball.as_ref())
            )
            .ok();
        }
        if args.throttle_us > 0 {
            std::thread::sleep(std::time::Duration::from_micros(args.throttle_us));
        }
    }
    let replay_s = replay_t0.elapsed().as_secs_f64();

    // Site transitions from the tracker's own log output.
    let captured = CAPTURE.lock().expect("capture lock").clone();
    let mut transitions = Vec::new();
    let mut dormant_events = 0usize;
    let mut releases_actual = 0usize;
    let mut releases_attempted = 0usize;
    let mut release_reasons: BTreeMap<String, usize> = BTreeMap::new();
    let mut decoy_release_attempts = 0usize;
    let mut decoy_release_reasons: BTreeMap<String, usize> = BTreeMap::new();
    let mut acquisitions = 0usize;
    let mut losses = 0usize;
    for c in &captured {
        let m = &c.message;
        let (yaw, pitch, reason) = parse_site(m);
        let at_decoy = yaw.zip(pitch).is_some_and(|(y, p)| {
            ((y - DECOY_YAW).powi(2) + (p - DECOY_PITCH).powi(2)).sqrt() <= DECOY_RADIUS_RAD
        });
        let kind = if m.contains("dormant static site") {
            dormant_events += 1;
            "dormant"
        } else if m.contains("reactivated static site") {
            releases_actual += 1;
            releases_attempted += 1;
            "release_actual"
        } else if m.starts_with("DIAG_RELEASE_SITE") {
            releases_attempted += 1;
            if m.contains("would_release=true") {
                releases_actual += 1;
            }
            "release_attempt_log_only"
        } else if m.contains("acquired") {
            acquisitions += 1;
            continue;
        } else if m.contains("track lost") {
            losses += 1;
            continue;
        } else {
            continue;
        };
        if kind.starts_with("release") {
            let r = reason.clone().unwrap_or_default();
            *release_reasons.entry(r.clone()).or_default() += 1;
            if at_decoy {
                decoy_release_attempts += 1;
                *decoy_release_reasons.entry(r).or_default() += 1;
            }
        }
        transitions.push(serde_json::json!({
            "frame": c.frame, "kind": kind, "yaw": yaw, "pitch": pitch,
            "reason": reason, "at_decoy": at_decoy,
        }));
    }
    // First dormancy at the decoy, and fresh decoy Tracking accepts after it.
    let first_decoy_dormant = transitions
        .iter()
        .find(|t| t["kind"] == "dormant" && t["at_decoy"] == true)
        .and_then(|t| t["frame"].as_u64());
    let decoy_tracking_after_first_dormancy = first_decoy_dormant.map(|f| {
        decoy_tracker_tracking
            .iter()
            .skip(f as usize + 1)
            .filter(|&&b| b)
            .count()
    });

    let frame_s = frame_ms / 1000.0;
    let total = frames.len();
    let mismatches: usize = parity
        .iter()
        .filter(|(k, _)| k.starts_with("mismatch"))
        .map(|(_, v)| *v)
        .sum();
    let report = serde_json::json!({
        "label": args.label,
        "events": args.events,
        "clock": args.clock,
        "frames": total,
        "config": {
            "ball_class_id": BALL_CLASS_ID,
            "max_jump_rad": FIELD_MAX_JUMP_RAD,
            "max_coast_frames": COAST_FRAMES_STRIDE1,
            "player_anchor_rad": reco_autocam::trackers::ball::DEFAULT_PLAYER_ANCHOR_RAD,
            "reacquire_grace_ms": reco_autocam::trackers::ball::DEFAULT_REACQUIRE_GRACE_MS,
            "reacquire_initial_weight": reco_autocam::trackers::ball::DEFAULT_REACQUIRE_INITIAL_WEIGHT,
            "player_source": "recorded world_state.players (ClassProvider person=0 output)",
            "fps": format!("{}/{}", args.fps_num, args.fps_den),
        },
        "input": {
            "ball_candidates_mapped": ball_candidates,
            "frames_with_candidate": frames_with_candidate,
            "nonmonotonic_input_timestamps": nonmonotonic_inputs,
            "note": "detections_raw is post-wrapper (ROI-filtered) mapped tracker input, not pre-ROI detector output",
        },
        "tracker_stage_states": replay_states,
        "recorded_post_bridge_states": recorded_states,
        "parity_vs_recorded": {
            "counts": parity,
            "mismatches": mismatches,
            "exact": mismatches == 0,
            "only_meaningful_for_clock": "historical",
            "first_mismatch": first_mismatch,
            "rule": "tracker None/Coasting may appear as recorded Bridged; all else must match state and position within 1e-6 rad",
        },
        "decoy_location_proxy": {
            "center": [DECOY_YAW, DECOY_PITCH],
            "radius_rad": DECOY_RADIUS_RAD,
            "note": "location proxy only, not identity truth",
            "tracker_stage_tracking": episode_summary(&decoy_tracker_tracking, frame_s),
            "recorded_tracking": episode_summary(&decoy_recorded_tracking, frame_s),
            "recorded_bridged": episode_summary(&decoy_recorded_bridged, frame_s),
            "recorded_any_state": episode_summary(&decoy_recorded_any, frame_s),
            "first_decoy_dormant_frame": first_decoy_dormant,
            "tracker_tracking_frames_after_first_decoy_dormancy": decoy_tracking_after_first_dormancy,
        },
        "sites": {
            "dormant_transitions": dormant_events,
            "release_calls_attempted": releases_attempted,
            "releases_actual": releases_actual,
            "release_reasons": release_reasons,
            "decoy_release_calls": decoy_release_attempts,
            "decoy_release_reasons": decoy_release_reasons,
            "acquisitions": acquisitions,
            "full_losses": losses,
            "transitions": transitions,
        },
        "decision_trace_fnv1a": format!("{hash:016x}"),
        "profiling": {
            "load_s": load_s,
            "replay_s": replay_s,
            "throttle_us": args.throttle_us,
            "peak_rss_kb": peak_rss_kb(),
            "cpus": std::thread::available_parallelism().map(|n| n.get()).ok(),
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
        },
    });
    let mut f = File::create(&args.out).expect("create report");
    serde_json::to_writer_pretty(&mut f, &report).expect("write report");
    println!(
        "{} clock={} frames={} parity_mismatches={} hash={:016x} decoy_tracking_eps={} releases_attempted={} actual={} replay_s={:.2}",
        args.label,
        args.clock,
        total,
        mismatches,
        hash,
        report["decoy_location_proxy"]["tracker_stage_tracking"]["episodes"],
        releases_attempted,
        releases_actual,
        replay_s
    );
}
