//! OEV-001 historical parity on a bounded real-data fixture.
//!
//! Fixture: frames 0-1299 of sample_02 run 34236809938 (video-stitcher
//! `fa3c79e`, ROI-relax experiment), see `tests/fixtures/PROVENANCE.md`.
//! It holds the recorded post-wrapper `detections_raw` (ball class only;
//! other classes are ignored by `BallTracker` on every path), the recorded
//! player tracker output and the recorded post-bridge `world_state.ball`,
//! with the recorded tracker input timestamps (historical clock).
//!
//! The real tracker is replayed from frame 0. The fixture was recorded with
//! the pre-OEV-003 static-site logic, so the replay must reproduce every
//! recorded decision up to the first dormant-site transition of the current
//! tracker: OEV-003 changes behaviour only through the static-site gate.
//! Backward bridging only rewrites a tracker `None` or `Coasting` into
//! `Bridged`; everything else must match exactly. (Exact full-fixture parity
//! of the pre-OEV-003 tracker is asserted on `audit/cpu-ball-replay-01`.)

use std::cell::RefCell;
use std::io::{BufRead, BufReader};

use reco_autocam::trackers::BallTracker;
use reco_core::detect::director::MappedDetection;
use reco_core::detect::tracker::{TrackState, TrackedEntity, Tracker, WorldState};

thread_local! {
    static DORMANT_AT: RefCell<Option<usize>> = const { RefCell::new(None) };
    static FRAME: RefCell<usize> = const { RefCell::new(0) };
}

struct DormancyLogger;

impl log::Log for DormancyLogger {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Info
    }
    fn log(&self, r: &log::Record) {
        if r.args().to_string().contains("dormant static site") {
            let frame = FRAME.with(|f| *f.borrow());
            DORMANT_AT.with(|d| {
                let mut d = d.borrow_mut();
                if d.is_none() {
                    *d = Some(frame);
                }
            });
        }
    }
    fn flush(&self) {}
}

static LOGGER: DormancyLogger = DormancyLogger;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/sample02_run34236809938_f0000-1299.jsonl.gz"
);

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
}

struct Frame {
    detections: Vec<MappedDetection>,
    timestamp_ms: f64,
    players: Vec<TrackedEntity>,
    ball: Option<TrackedEntity>,
}

fn load() -> Vec<Frame> {
    let file = std::fs::File::open(FIXTURE).expect("fixture present");
    let reader = BufReader::new(flate2::read::GzDecoder::new(file));
    let mut frames = Vec::new();
    let mut pending: Option<(u64, Vec<MappedDetection>)> = None;
    let mut expected_index = 0u64;
    for line in reader.lines() {
        let line = line.expect("read");
        match serde_json::from_str::<Event>(&line).expect("parse") {
            Event::FrameStart { frame_index } => assert_eq!(frame_index, expected_index),
            Event::DetectionsRaw {
                frame_index,
                detections,
            } => {
                assert_eq!(frame_index, expected_index);
                pending = Some((frame_index, detections));
            }
            Event::WorldState {
                frame_index,
                timestamp_ms,
                players,
                ball,
            } => {
                let (i, detections) = pending.take().expect("detections before world");
                assert_eq!(i, frame_index);
                frames.push(Frame {
                    detections,
                    timestamp_ms,
                    players,
                    ball,
                });
                expected_index += 1;
            }
        }
    }
    frames
}

fn field_tracker() -> BallTracker {
    // reco-autocam TrackingMode::Field at frame stride 1.
    BallTracker::new(32)
        .with_max_jump_rad(0.8)
        .with_max_coast_frames(reco_autocam::trackers::ball::DEFAULT_COAST_FRAMES)
}

fn replay(frames: &[Frame], scale_clock: f64) -> Vec<Option<TrackedEntity>> {
    let mut t = field_tracker();
    frames
        .iter()
        .map(|f| {
            t.observe_world(&WorldState {
                ball: None,
                players: f.players.clone(),
            });
            t.update(&f.detections, f.timestamp_ms * scale_clock)
                .into_iter()
                .next()
        })
        .collect()
}

#[test]
fn fixture_is_complete_and_ordered() {
    let frames = load();
    assert_eq!(frames.len(), 1300);
    assert!(
        frames
            .windows(2)
            .all(|w| w[1].timestamp_ms > w[0].timestamp_ms)
    );
}

#[test]
fn historical_parity_until_first_dormant_site() {
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Info);
    DORMANT_AT.with(|d| *d.borrow_mut() = None);
    let frames = load();
    let mut t = field_tracker();
    let mut out = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        FRAME.with(|fr| *fr.borrow_mut() = i);
        t.observe_world(&WorldState {
            ball: None,
            players: f.players.clone(),
        });
        out.push(t.update(&f.detections, f.timestamp_ms).into_iter().next());
    }
    let dormant_at = DORMANT_AT
        .with(|d| *d.borrow())
        .expect("fixture should reach a dormant site on the historical clock");
    let mut checked = 0;
    for (i, (replay, recorded)) in out
        .iter()
        .zip(frames.iter().map(|f| f.ball))
        .enumerate()
        .take(dormant_at + 1)
    {
        match (replay, recorded) {
            (None, None) => {}
            (None, Some(r)) | (Some(_), Some(r)) if r.state == TrackState::Bridged => {
                assert!(
                    replay.is_none_or(|p| p.state == TrackState::Coasting),
                    "frame {i}: Bridged over a non-Coasting tracker output"
                );
            }
            (Some(p), Some(r)) => {
                assert_eq!(p.state, r.state, "frame {i}: state");
                assert!((p.yaw - r.yaw).abs() <= 1e-6, "frame {i}: yaw");
                assert!((p.pitch - r.pitch).abs() <= 1e-6, "frame {i}: pitch");
                assert_eq!(p.confidence, r.confidence, "frame {i}: confidence");
            }
            (p, r) => panic!("frame {i}: replay {p:?} vs recorded {r:?}"),
        }
        checked += 1;
    }
    assert!(checked > 90, "too few frames checked: {checked}");
}

#[test]
fn replay_is_deterministic() {
    let frames = load();
    let key = |v: &[Option<TrackedEntity>]| {
        v.iter()
            .map(|b| b.map(|b| (b.state, b.yaw.to_bits(), b.pitch.to_bits())))
            .collect::<Vec<_>>()
    };
    let a = key(&replay(&frames, 1.0));
    assert_eq!(a, key(&replay(&frames, 1.0)));
    assert_eq!(a, key(&replay(&frames, 1.0)));
}

/// Documents current behaviour: the tracker's decisions depend on the
/// absolute clock scale it is given (static-site and grace thresholds are in
/// milliseconds). Replaying the recorded wall-clock inputs on source-video
/// time (~13x compressed here) changes decisions.
#[test]
fn decisions_change_under_media_clock() {
    let frames = load();
    let historical = replay(&frames, 1.0);
    let media: Vec<_> = {
        let mut t = field_tracker();
        frames
            .iter()
            .enumerate()
            .map(|(i, f)| {
                t.observe_world(&WorldState {
                    ball: None,
                    players: f.players.clone(),
                });
                t.update(&f.detections, i as f64 * 1001.0 / 60.0)
                    .into_iter()
                    .next()
            })
            .collect()
    };
    let diffs = historical
        .iter()
        .zip(&media)
        .filter(|(a, b)| {
            a.map(|a| (a.state, a.yaw.to_bits())) != b.map(|b| (b.state, b.yaw.to_bits()))
        })
        .count();
    assert!(diffs > 0, "expected clock sensitivity on this fixture");
}
