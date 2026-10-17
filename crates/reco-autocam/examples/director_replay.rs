//! CPU replay of a director arm over recorded OEV pipeline events (OEV-006).
//!
//! Reads an `events.jsonl` from a stitch run (`--events`), rebuilds the
//! per-frame world states, and runs a `FieldPanner` through
//! [`reco_core::session::replay_dense_lookahead`] - the CPU mirror of the
//! live stride-1 lookahead loop (bridging, lookahead window, centered
//! smoothing; parity asserted by reco-core `offline_director_matches_live_*`).
//! Output is a per-frame camera trace the OEV-006 scorer consumes, so arms
//! can be compared on identical detections without rendering.
//!
//! World sources (`--worlds`):
//! - `recorded`: the run's recorded `world_state` (post-bridge). Re-bridging
//!   is a no-op on already-bridged frames, but the lookahead window then
//!   holds post-bridge futures where the live loop held pre-bridge ones, so
//!   the replayed poses approximate (not equal) the recorded `pan_decision`.
//!   Use it to sanity-check a run against its own recording. Verified
//!   17 Oct 2026: run 34236809938 (sample_02, 10,775 frames, preset
//!   `v4_micro_damping`, lookahead 1.5 s) replays to the recorded
//!   `pan_decision` with zero difference in yaw, pitch and fov on every
//!   frame (pose hash `e46afd981ecd8989`).
//! - `tracker`: re-run this branch's `BallTracker` (field-mode construction,
//!   production-media clock) on the recorded `detections_raw` + recorded
//!   players. This is the arm-A path for new detections. Note that ROI
//!   filtering happened before `detections_raw` was recorded: an ROI change
//!   needs new GPU detections, not a replay.
//!
//! Panner: `--panner-preset` base + `--panner-config` JSON overlay, merged
//! exactly as `reco stitch` does. `--ball-weight 0` gives the B0 ablation
//! (player-only framing) without touching panner code.
//!
//! Usage:
//! ```text
//! cargo run --release -p reco-autocam --no-default-features --example director_replay -- \
//!     --events run.jsonl --trace trace.csv --report report.json \
//!     [--worlds recorded|tracker] [--active-ball] [--panner-preset broadcast]
//!     [--panner-config overlay.json] [--ball-weight W] [--lookahead-secs 1.5]
//!     [--fps-num 60000 --fps-den 1001] [--label name]
//! ```

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

use reco_autocam::filter_mapped_field_mode;
use reco_autocam::panners::{FieldPanner, FieldPannerConfig};
use reco_autocam::panners::{HybridConfig, HybridDirector};
use reco_autocam::trackers::players::{PlayerTracker, PlayerTrackerConfig};
use reco_autocam::trackers::{BallTracker, ClassProvider};
use reco_core::calibration::{CameraParams, MatchCalibration, PlaneLayout};
use reco_core::detect::director::{MappedDetection, ViewportPosition};
use reco_core::detect::tracker::{TrackState, TrackedEntity, Tracker, WorldState};
use reco_core::session::{OfflineFrame, replay_dense_lookahead};

const BALL_CLASS_ID: u16 = 32;
const PERSON_CLASS_ID: u16 = 0;
const FIELD_MAX_JUMP_RAD: f32 = 0.8;
/// A1/C ball gate (frozen for the OEV-006 first pass).
const BALL_GATE_PLAYER_RAD: f32 = 0.35;
const BALL_GATE_TRACK_RAD: f32 = 0.30;

#[derive(serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Event {
    DetectionsRaw {
        frame_index: u64,
        detections: Vec<MappedDetection>,
    },
    WorldState {
        frame_index: u64,
        players: Vec<TrackedEntity>,
        ball: Option<TrackedEntity>,
    },
    PanDecision {
        frame_index: u64,
        pose: ViewportPosition,
    },
    #[serde(other)]
    Other,
}

/// Deferred write of one event into its frame slot.
type Apply = Box<dyn FnOnce(&mut Recorded)>;

#[derive(Default)]
struct Recorded {
    detections: Option<Vec<MappedDetection>>,
    players: Option<Vec<TrackedEntity>>,
    ball: Option<TrackedEntity>,
    pan: Option<ViewportPosition>,
}

struct Args {
    events: String,
    trace: String,
    report: String,
    worlds: String,
    active_ball: bool,
    preset: Option<String>,
    config: Option<String>,
    ball_weight: Option<f32>,
    lookahead_secs: f64,
    roi: Option<String>,
    ball_roi_margin: f64,
    players: String,
    ownership_roi: Option<String>,
    ball_gate: bool,
    director: String,
    fps_num: u64,
    fps_den: u64,
    label: String,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        events: String::new(),
        trace: String::new(),
        report: String::new(),
        worlds: "tracker".into(),
        active_ball: false,
        preset: Some("broadcast".into()),
        config: None,
        ball_weight: None,
        lookahead_secs: 1.5,
        roi: None,
        ball_roi_margin: 0.40,
        players: "stateless".into(),
        ownership_roi: None,
        ball_gate: false,
        director: "field".into(),
        fps_num: 60000,
        fps_den: 1001,
        label: String::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--events" => a.events = value()?,
            "--trace" => a.trace = value()?,
            "--report" => a.report = value()?,
            "--worlds" => a.worlds = value()?,
            "--active-ball" => a.active_ball = true,
            "--panner-preset" => a.preset = Some(value()?),
            "--no-preset" => a.preset = None,
            "--panner-config" => a.config = Some(value()?),
            "--ball-weight" => {
                a.ball_weight = Some(
                    value()?
                        .parse()
                        .map_err(|e| format!("--ball-weight: {e}"))?,
                )
            }
            "--lookahead-secs" => {
                a.lookahead_secs = value()?
                    .parse()
                    .map_err(|e| format!("--lookahead-secs: {e}"))?
            }
            "--roi" => a.roi = Some(value()?),
            "--ball-roi-margin" => {
                a.ball_roi_margin = value()?
                    .parse()
                    .map_err(|e| format!("--ball-roi-margin: {e}"))?
            }
            "--players" => a.players = value()?,
            "--ownership-roi" => a.ownership_roi = Some(value()?),
            "--ball-gate" => a.ball_gate = true,
            "--director" => a.director = value()?,
            "--fps-num" => a.fps_num = value()?.parse().map_err(|e| format!("--fps-num: {e}"))?,
            "--fps-den" => a.fps_den = value()?.parse().map_err(|e| format!("--fps-den: {e}"))?,
            "--label" => a.label = value()?,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if a.events.is_empty() || a.trace.is_empty() || a.report.is_empty() {
        return Err("--events, --trace and --report are required".into());
    }
    if a.worlds != "recorded" && a.worlds != "tracker" {
        return Err("--worlds must be recorded or tracker".into());
    }
    if !["stateless", "tracked", "owned"].contains(&a.players.as_str()) {
        return Err("--players must be stateless, tracked or owned".into());
    }
    if !["field", "hybrid"].contains(&a.director.as_str()) {
        return Err("--director must be field or hybrid".into());
    }
    if (a.players == "owned" || a.ball_gate) && a.ownership_roi.is_none() {
        return Err("--players owned and --ball-gate need --ownership-roi".into());
    }
    if a.worlds == "recorded" && (a.players != "stateless" || a.ball_gate) {
        return Err("--players/--ball-gate need --worlds tracker".into());
    }
    if a.fps_num == 0 || a.fps_den == 0 || a.lookahead_secs <= 0.0 {
        return Err("fps and lookahead must be positive".into());
    }
    Ok(a)
}

/// Preset base + JSON overlay, merged the same way `reco stitch` does.
fn panner_config(a: &Args) -> Result<FieldPannerConfig, String> {
    let base = match &a.preset {
        Some(name) => Some(
            FieldPannerConfig::from_preset_name(name)
                .ok_or_else(|| format!("unknown preset {name}"))?,
        ),
        None => None,
    };
    let mut cfg = match &a.config {
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
            let over: serde_json::Value =
                serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
            match base {
                Some(b) => {
                    let mut v = serde_json::to_value(&b).map_err(|e| e.to_string())?;
                    if let (Some(bm), serde_json::Value::Object(om)) = (v.as_object_mut(), over) {
                        bm.extend(om);
                    }
                    serde_json::from_value(v).map_err(|e| format!("{path}: {e}"))?
                }
                None => serde_json::from_value(over).map_err(|e| format!("{path}: {e}"))?,
            }
        }
        None => base.unwrap_or_default(),
    };
    if let Some(w) = a.ball_weight {
        cfg.ball_weight = w;
    }
    Ok(cfg)
}

fn load(path: &str) -> Result<Vec<Recorded>, String> {
    let file = File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let mut frames: Vec<Recorded> = Vec::new();
    for (n, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("line {n}: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let ev: Event = serde_json::from_str(&line).map_err(|e| format!("line {n}: {e}"))?;
        let (idx, apply): (u64, Apply) = match ev {
            Event::DetectionsRaw {
                frame_index,
                detections,
            } => (
                frame_index,
                Box::new(move |r| r.detections = Some(detections)),
            ),
            Event::WorldState {
                frame_index,
                players,
                ball,
            } => (
                frame_index,
                Box::new(move |r| {
                    r.players = Some(players);
                    r.ball = ball;
                }),
            ),
            Event::PanDecision { frame_index, pose } => {
                (frame_index, Box::new(move |r| r.pan = Some(pose)))
            }
            Event::Other => continue,
        };
        let slot = idx as usize;
        if frames.len() <= slot {
            frames.resize_with(slot + 1, Recorded::default);
        }
        apply(&mut frames[slot]);
    }
    for (i, f) in frames.iter().enumerate() {
        if f.players.is_none() {
            return Err(format!("frame {i}: no world_state"));
        }
    }
    Ok(frames)
}

/// Records the hybrid director's state per decision (decisions run in
/// frame order, so entry `i` belongs to frame `i`).
struct StateLogging {
    inner: HybridDirector,
    log: std::sync::Arc<std::sync::Mutex<Vec<&'static str>>>,
}

impl reco_core::detect::panner::Panner for StateLogging {
    fn decide(
        &mut self,
        world: &WorldState,
        ctx: &reco_core::detect::panner::PanContext<'_>,
    ) -> ViewportPosition {
        self.decide_with_lookahead(world, &[], ctx)
    }

    fn decide_with_lookahead(
        &mut self,
        world: &WorldState,
        future: &[WorldState],
        ctx: &reco_core::detect::panner::PanContext<'_>,
    ) -> ViewportPosition {
        let pose = self.inner.decide_with_lookahead(world, future, ctx);
        self.log
            .lock()
            .expect("state log")
            .push(self.inner.state().name());
        pose
    }
}

/// The FieldPanner never reads calibration; the replay still needs one for
/// `PanContext`. A fixed placeholder keeps the trace independent of rigs.
fn placeholder_calibration() -> MatchCalibration {
    let cam = CameraParams {
        width: 3840,
        height: 2160,
        fx: 1800.0,
        fy: 1800.0,
        cx: 1920.0,
        cy: 1080.0,
        d: [0.0; 4],
    };
    MatchCalibration {
        left: cam.clone(),
        right: cam,
        layout: PlaneLayout {
            camera_axis_offset: 0.24,
            intersect: 0.54,
            x_ty: 0.0,
            x_rz: 0.0,
            z_rx: 0.0,
            x_rx: 0.0,
            z_rz: 0.0,
        },
        rig_tilt: 0.0,
        rig_roll: 0.0,
        sync_offset: 0,
        field_roi: None,
        lens_correction_amount: 1.0,
        blend_width: 0.05,
    }
}

fn state_name(b: Option<&TrackedEntity>) -> &'static str {
    match b.map(|b| b.state) {
        None => "none",
        Some(TrackState::Tracking) => "tracking",
        Some(TrackState::Coasting) => "coasting",
        Some(TrackState::Bridged) => "bridged",
        Some(TrackState::Lost) => "lost",
    }
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("director_replay: {e}");
            std::process::exit(2);
        }
    };
    let cfg = match panner_config(&args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("director_replay: panner config: {e}");
            std::process::exit(2);
        }
    };
    let recorded = match load(&args.events) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("director_replay: {e}");
            std::process::exit(1);
        }
    };
    let rate = (args.fps_num, args.fps_den);
    let fps = args.fps_num as f64 / args.fps_den as f64;
    let media_ms = |i: u64| -> f64 {
        let ns = i as u128 * rate.1 as u128 * 1_000_000_000u128 / rate.0 as u128;
        std::time::Duration::from_nanos(ns as u64).as_secs_f64() * 1000.0
    };

    let mut tracker = BallTracker::new(BALL_CLASS_ID)
        .with_max_jump_rad(FIELD_MAX_JUMP_RAD)
        .with_max_coast_frames(reco_autocam::trackers::ball::DEFAULT_COAST_FRAMES);
    if args.active_ball {
        tracker = tracker.with_active_ball_selection(
            reco_autocam::trackers::active_ball::ActiveBallConfig::sample02(PERSON_CLASS_ID),
        );
    }

    let load_roi = |p: &String| -> reco_core::calibration::FieldRoi {
        let text = std::fs::read_to_string(p).unwrap_or_else(|e| {
            eprintln!("director_replay: roi {p}: {e}");
            std::process::exit(2);
        });
        serde_json::from_str(&text).unwrap_or_else(|e| {
            eprintln!("director_replay: roi {p}: {e}");
            std::process::exit(2);
        })
    };
    let roi = args.roi.as_ref().map(load_roi);
    let ownership_roi = args.ownership_roi.as_ref().map(load_roi);
    let need_tracks = args.players != "stateless" || args.ball_gate;
    let mut player_tracker = need_tracks.then(|| {
        PlayerTracker::new(
            PlayerTrackerConfig::oev_default(PERSON_CLASS_ID, fps as f32),
            if args.players == "tracked" && !args.ball_gate {
                None
            } else {
                ownership_roi.clone()
            },
        )
    });
    let mut last_ball: Option<(f32, f32)> = None;
    let mut gated_out = 0usize;
    if roi.is_some() && args.worlds == "recorded" {
        eprintln!(
            "director_replay: --roi needs --worlds tracker (recorded worlds are already filtered)"
        );
        std::process::exit(2);
    }
    let mut players_provider = ClassProvider::new(PERSON_CLASS_ID);
    let mut player_mismatch = 0usize;

    let mut frames = Vec::with_capacity(recorded.len());
    for (i, r) in recorded.iter().enumerate() {
        let t = media_ms(i as u64);
        let (ball, players) = if args.worlds == "recorded" {
            (r.ball, r.players.clone().expect("validated"))
        } else {
            let Some(raw) = r.detections.as_ref() else {
                eprintln!(
                    "director_replay: frame {i} has no detections_raw (needed for --worlds tracker)"
                );
                std::process::exit(1);
            };
            let dets = match &roi {
                Some(roi) => filter_mapped_field_mode(
                    raw,
                    roi,
                    PERSON_CLASS_ID,
                    BALL_CLASS_ID,
                    args.ball_roi_margin,
                ),
                None => raw.clone(),
            };
            // Players exactly as field mode builds them (stateless provider).
            let stateless = players_provider.update(&dets, t);
            if roi.is_none()
                && r.players
                    .as_ref()
                    .is_some_and(|rec| format!("{rec:?}") != format!("{stateless:?}"))
            {
                player_mismatch += 1;
            }
            let tracks = player_tracker.as_mut().map(|pt| pt.update(&dets));
            let players: Vec<TrackedEntity> = match (args.players.as_str(), &tracks) {
                ("tracked", Some(ts)) => ts.iter().map(|p| p.entity).collect(),
                ("owned", Some(ts)) => ts.iter().filter(|p| p.owned).map(|p| p.entity).collect(),
                _ => stateless,
            };
            // A1/C: accept a ball candidate only near an owned player or
            // near the last tracked ball (or when nothing is owned yet).
            let dets = if args.ball_gate {
                let owned: Vec<(f32, f32)> = tracks
                    .as_ref()
                    .map(|ts| {
                        ts.iter()
                            .filter(|p| p.owned)
                            .map(|p| (p.entity.yaw, p.entity.pitch))
                            .collect()
                    })
                    .unwrap_or_default();
                let before = dets.len();
                let kept: Vec<MappedDetection> = dets
                    .into_iter()
                    .filter(|d| {
                        if d.class_id != BALL_CLASS_ID || owned.is_empty() {
                            return true;
                        }
                        let Some(p) = d.position else { return false };
                        let near = |q: (f32, f32), r: f32| {
                            ((p.yaw - q.0).powi(2) + (p.pitch - q.1).powi(2)).sqrt() <= r
                        };
                        owned.iter().any(|&q| near(q, BALL_GATE_PLAYER_RAD))
                            || last_ball.is_some_and(|q| near(q, BALL_GATE_TRACK_RAD))
                    })
                    .collect();
                gated_out += before - kept.len();
                kept
            } else {
                dets
            };
            tracker.observe_world(&WorldState {
                ball: None,
                players: players.clone(),
            });
            let ball = tracker.update(&dets, t).into_iter().next();
            if let Some(b) = ball.filter(|b| b.state == TrackState::Tracking) {
                last_ball = Some((b.yaw, b.pitch));
            }
            (ball, players)
        };
        frames.push(OfflineFrame {
            source_frame_index: i as u64,
            elapsed_ms: t,
            world: WorldState { ball, players },
        });
    }

    let lookahead = (args.lookahead_secs * fps).round() as usize;
    let field = FieldPanner::with_config(fps as f32, cfg.clone());
    let state_log = std::sync::Arc::new(std::sync::Mutex::new(Vec::<&'static str>::new()));
    let mut panner: Box<dyn reco_core::detect::panner::Panner> = if args.director == "hybrid" {
        Box::new(StateLogging {
            inner: HybridDirector::new(field, HybridConfig::oev006(fps as f32)),
            log: std::sync::Arc::clone(&state_log),
        })
    } else {
        Box::new(field)
    };
    let cal = placeholder_calibration();
    let poses = replay_dense_lookahead(
        &frames,
        panner.as_mut(),
        lookahead,
        fps,
        &cal,
        ViewportPosition::default(),
    );

    let mut out = BufWriter::new(File::create(&args.trace).expect("create trace"));
    writeln!(
        out,
        "frame,t_s,ball_state,ball_yaw,ball_pitch,n_players,raw_yaw,raw_pitch,raw_fov,yaw,pitch,fov,rec_yaw,rec_pitch,rec_fov,director_state"
    )
    .ok();
    let states: Vec<&'static str> = state_log.lock().expect("state log").clone();
    let mut state_counts: BTreeMap<&str, usize> = BTreeMap::new();
    for st in &states {
        *state_counts.entry(st).or_default() += 1;
    }
    let mut ball_states: BTreeMap<&str, usize> = BTreeMap::new();
    let mut max_dyaw = 0.0f32;
    let mut sum_dyaw = 0.0f64;
    let mut compared = 0usize;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for p in &poses {
        let i = p.source_frame_index as usize;
        let b = p.world.ball.as_ref();
        *ball_states.entry(state_name(b)).or_default() += 1;
        let rec = recorded[i].pan;
        if let Some(r) = rec {
            let d = (p.smoothed.yaw - r.yaw).abs();
            max_dyaw = max_dyaw.max(d);
            sum_dyaw += d as f64;
            compared += 1;
        }
        for v in [
            p.smoothed.yaw,
            p.smoothed.pitch,
            p.smoothed.fov_degrees.unwrap_or(f32::NAN),
        ] {
            hash = (hash ^ v.to_bits() as u64).wrapping_mul(0x0100_0000_01b3);
        }
        let f = |x: Option<f32>| x.map_or(String::new(), |v| v.to_string());
        writeln!(
            out,
            "{i},{:.6},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            media_ms(i as u64) / 1000.0,
            state_name(b),
            f(b.map(|b| b.yaw)),
            f(b.map(|b| b.pitch)),
            p.world.players.len(),
            p.raw.yaw,
            p.raw.pitch,
            f(p.raw.fov_degrees),
            p.smoothed.yaw,
            p.smoothed.pitch,
            f(p.smoothed.fov_degrees),
            f(rec.map(|r| r.yaw)),
            f(rec.map(|r| r.pitch)),
            f(rec.and_then(|r| r.fov_degrees)),
            states.get(i).copied().unwrap_or("field"),
        )
        .ok();
    }
    out.flush().ok();

    let report = serde_json::json!({
        "label": args.label,
        "events": args.events,
        "worlds": args.worlds,
        "active_ball": args.active_ball,
        "panner_preset": args.preset,
        "panner_config_file": args.config,
        "panner_config": cfg,
        "lookahead_frames": lookahead,
        "fps": [args.fps_num, args.fps_den],
        "frames": poses.len(),
        "roi_file": args.roi,
        "ball_roi_margin": args.ball_roi_margin,
        "recomputed_players_differ_from_recorded_frames": player_mismatch,
        "ball_states": ball_states,
        "director": args.director,
        "director_states": state_counts,
        "players_mode": args.players,
        "ownership_roi_file": args.ownership_roi,
        "ball_gate": args.ball_gate,
        "ball_candidates_gated_out": gated_out,
        "vs_recorded_pan_decision": {
            "frames_compared": compared,
            "max_abs_yaw_rad": max_dyaw,
            "mean_abs_yaw_rad": if compared > 0 { sum_dyaw / compared as f64 } else { 0.0 },
        },
        "pose_hash": format!("{hash:016x}"),
    });
    std::fs::write(
        &args.report,
        serde_json::to_string_pretty(&report).expect("report serializes"),
    )
    .expect("write report");
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("report serializes")
    );
}
