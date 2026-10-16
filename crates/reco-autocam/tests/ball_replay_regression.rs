//! OEV-001 causal regression fixtures for the real [`BallTracker`].
//!
//! These tests drive the unmodified tracker with small synthetic
//! detection streams and assert its EXISTING behaviour, including
//! behaviour that is known to be undesirable (marked "documents current
//! behaviour"). They exist so that later tickets can change a rule and see
//! exactly which causal case moved. Nothing here fixes or reimplements a
//! tracker rule; site transitions are observed through the tracker's own
//! `log` output.
//!
//! All streams are deterministic: fixed timestamps, fixed detections, no
//! wall-clock input.

use std::cell::RefCell;

use reco_autocam::trackers::BallTracker;
use reco_core::detect::detector::CameraId;
use reco_core::detect::director::{MappedDetection, ViewportPosition};
use reco_core::detect::tracker::{TrackState, TrackedEntity, Tracker, WorldState};

// ------------------------------------------------------------ log capture

thread_local! {
    static LINES: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

struct ThreadLogger;

impl log::Log for ThreadLogger {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Info
    }
    fn log(&self, r: &log::Record) {
        let line = r.args().to_string();
        LINES.with(|l| l.borrow_mut().push(line));
    }
    fn flush(&self) {}
}

static LOGGER: ThreadLogger = ThreadLogger;

fn init_capture() {
    // Every test calls this; only the first set_logger in the process wins,
    // later calls are harmless no-ops. Capture is per test thread.
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Info);
    LINES.with(|l| l.borrow_mut().clear());
}

fn captured(needle: &str) -> usize {
    LINES.with(|l| l.borrow().iter().filter(|s| s.contains(needle)).count())
}

// ------------------------------------------------------------ fixtures

const BALL: u16 = 32;
const STEP_MS: f64 = 100.0;

fn ball(cam: CameraId, yaw: f32, pitch: f32, conf: f32) -> MappedDetection {
    MappedDetection {
        camera: cam,
        class_id: BALL,
        confidence: conf,
        camera_center: (0.5, 0.5),
        camera_size: (0.01, 0.01),
        position: Some(ViewportPosition {
            yaw,
            pitch,
            fov_degrees: None,
        }),
    }
}

fn player(id: u64, yaw: f32, pitch: f32) -> TrackedEntity {
    TrackedEntity {
        id,
        class_id: 0,
        yaw,
        pitch,
        confidence: 0.9,
        state: TrackState::Tracking,
        age_frames: 1,
        origin: CameraId::Left,
    }
}

/// Field-mode tracker as built by reco-autocam, but with a short coast so
/// fixtures stay small. Coast length is a builder option, not a rule change.
fn tracker() -> BallTracker {
    BallTracker::new(BALL)
        .with_max_jump_rad(0.8)
        .with_max_coast_frames(2)
}

/// One frame: players observed first, then ball update (dispatch order).
fn step(
    t: &mut BallTracker,
    players: &[TrackedEntity],
    dets: &[MappedDetection],
    ms: f64,
) -> Option<TrackedEntity> {
    t.observe_world(&WorldState {
        ball: None,
        players: players.to_vec(),
    });
    t.update(dets, ms).into_iter().next()
}

const SPARE: (f32, f32) = (1.0, 0.0);

/// Players: one stationary next to the spare ball (so the player-anchor gate
/// admits it) and two moving players around yaw 0 (they dominate the
/// velocity-weighted centroid, contradicting the spare site).
fn players_at(frame: u64) -> Vec<TrackedEntity> {
    let wobble = 0.02 * ((frame % 4) as f32);
    vec![
        player(1, SPARE.0, SPARE.1 + 0.05),
        player(2, -0.05 + wobble, 0.0),
        player(3, 0.05 - wobble, 0.02),
    ]
}

/// Players during play-only phases: the bystander next to the spare has
/// walked away, so only the moving players near yaw 0 remain.
fn play_players(frame: u64) -> Vec<TrackedEntity> {
    players_at(frame).into_iter().skip(1).collect()
}

/// Builds a dormant spare-ball site: alternating 3 s phases of "only the
/// stationary spare is visible" and "only the real ball near play is
/// visible", repeated `cycles` times. The bystander player is present only
/// during spare phases (see `bystander_near_spare_triggers_pickup_release`
/// for what happens when it stays). Returns (tracker, next_frame).
fn dormant_spare_site(cycles: u64) -> (BallTracker, u64) {
    build_spare_site(cycles, false)
}

fn build_spare_site(cycles: u64, bystander_stays: bool) -> (BallTracker, u64) {
    let mut t = tracker();
    let mut f = 0u64;
    for _ in 0..cycles {
        for _ in 0..30 {
            step(
                &mut t,
                &players_at(f),
                &[ball(CameraId::Left, SPARE.0, SPARE.1, 0.5)],
                f as f64 * STEP_MS,
            );
            f += 1;
        }
        // 1 s with no ball candidates, longer than the 750 ms outgoing-start
        // window, so the play phase cannot itself release the site (see
        // `unrelated_ball_right_after_spare_sighting_releases_site`).
        for _ in 0..10 {
            let players = if bystander_stays {
                players_at(f)
            } else {
                play_players(f)
            };
            step(&mut t, &players, &[], f as f64 * STEP_MS);
            f += 1;
        }
        for k in 0..30 {
            let yaw = -0.1 + 0.01 * k as f32;
            let players = if bystander_stays {
                players_at(f)
            } else {
                play_players(f)
            };
            step(
                &mut t,
                &players,
                &[ball(CameraId::Left, yaw, 0.0, 0.8)],
                f as f64 * STEP_MS,
            );
            f += 1;
        }
    }
    (t, f)
}

// ------------------------------------------------------------ cases

/// Stationary spare ball, repeatedly selected while play is elsewhere,
/// becomes a dormant site and is then rejected.
#[test]
fn stationary_spare_ball_becomes_dormant_and_is_rejected() {
    init_capture();
    let (mut t, f) = dormant_spare_site(5);
    assert!(
        captured("dormant static site") >= 1,
        "site never went dormant"
    );
    // Only the spare is visible now: the tracker must not accept it.
    let mut accepted = 0;
    for k in 0..30 {
        let out = step(
            &mut t,
            &players_at(f + k),
            &[ball(CameraId::Left, SPARE.0, SPARE.1, 0.9)],
            (f + k) as f64 * STEP_MS,
        );
        if out.is_some_and(|b| b.state == TrackState::Tracking) {
            accepted += 1;
        }
    }
    assert_eq!(accepted, 0, "dormant spare was re-accepted");
}

/// Legitimate stationary dead ball (single long static episode with play
/// around it) stays trackable: one episode is never enough for dormancy.
#[test]
fn legitimate_dead_ball_single_episode_is_not_excluded() {
    init_capture();
    let mut t = tracker();
    let players = [player(1, 0.3, 0.0), player(2, 0.35, 0.02)];
    let mut tracking = 0;
    for f in 0..100u64 {
        let out = step(
            &mut t,
            &players,
            &[ball(CameraId::Left, 0.3, 0.0, 0.7)],
            f as f64 * STEP_MS,
        );
        tracking += usize::from(out.is_some_and(|b| b.state == TrackState::Tracking));
    }
    assert_eq!(tracking, 100);
    assert_eq!(captured("dormant static site"), 0);
}

/// Genuine ball disappears briefly (within coast) and returns: Coasting
/// then Tracking again, held position while coasting.
#[test]
fn genuine_ball_disappears_and_returns_within_coast() {
    init_capture();
    let mut t = tracker();
    let p = [player(1, 0.0, 0.0)];
    step(&mut t, &p, &[ball(CameraId::Left, 0.0, 0.0, 0.8)], 0.0);
    let c = step(&mut t, &p, &[], 100.0).expect("coast");
    assert_eq!(c.state, TrackState::Coasting);
    assert_eq!((c.yaw, c.pitch), (0.0, 0.0));
    let back = step(&mut t, &p, &[ball(CameraId::Left, 0.02, 0.0, 0.8)], 200.0).expect("back");
    assert_eq!(back.state, TrackState::Tracking);
}

/// Genuine ball lost fully, then returns far away after grace: acquired by
/// confidence (no max-jump gate after full loss).
#[test]
fn genuine_ball_lost_then_reacquired_far_away() {
    init_capture();
    let mut t = tracker();
    let p = [player(1, 0.0, 0.0), player(2, 0.6, 0.0)];
    step(&mut t, &p, &[ball(CameraId::Left, 0.0, 0.0, 0.8)], 0.0);
    step(&mut t, &p, &[], 100.0);
    step(&mut t, &p, &[], 200.0);
    let lost = step(&mut t, &p, &[], 300.0).expect("lost transition");
    assert_eq!(lost.state, TrackState::Lost);
    assert!(step(&mut t, &p, &[], 400.0).is_none());
    let far = step(&mut t, &p, &[ball(CameraId::Left, 0.6, 0.0, 0.5)], 1_500.0).expect("reacq");
    assert_eq!(far.state, TrackState::Tracking);
    assert_eq!(far.yaw, 0.6);
}

/// Real ball kicked away from a dormant site with no player escorting it:
/// coherent motion starting at the site releases it via
/// "persistent outgoing motion".
#[test]
fn real_ball_kicked_away_from_dormant_site_releases_it() {
    init_capture();
    let (mut t, f) = dormant_spare_site(5);
    assert!(captured("dormant static site") >= 1);
    let mut frame = f;
    step(
        &mut t,
        &play_players(frame),
        &[ball(CameraId::Left, SPARE.0, SPARE.1, 0.6)],
        frame as f64 * STEP_MS,
    );
    frame += 1;
    for k in 1..=10 {
        step(
            &mut t,
            &play_players(frame),
            &[ball(CameraId::Left, SPARE.0 - 0.03 * k as f32, 0.0, 0.6)],
            frame as f64 * STEP_MS,
        );
        frame += 1;
    }
    assert!(
        captured("(persistent outgoing motion)") >= 1,
        "kicked ball did not release the site"
    );
}

/// Real ball dribbled away by a player starting at the site: the site is
/// released. Documents current behaviour: the pickup-occlusion path fires
/// before outgoing motion accumulates its 3 detections / 750 ms.
#[test]
fn real_ball_dribbled_away_from_dormant_site_releases_it() {
    init_capture();
    let (mut t, f) = dormant_spare_site(5);
    let mut frame = f;
    step(
        &mut t,
        &players_at(frame),
        &[ball(CameraId::Left, SPARE.0, SPARE.1, 0.6)],
        frame as f64 * STEP_MS,
    );
    frame += 1;
    for k in 1..=10 {
        let yaw = SPARE.0 - 0.03 * k as f32;
        step(
            &mut t,
            &[player(1, yaw, 0.0)],
            &[ball(CameraId::Left, yaw, 0.0, 0.6)],
            frame as f64 * STEP_MS,
        );
        frame += 1;
    }
    assert!(captured("reactivated static site") >= 1);
    assert!(captured("(pickup occlusion)") >= 1);
}

/// Documents current behaviour (known weakness, OEV-001 hypothesis): an
/// UNRELATED ball far from the site that appears within 750 ms of the last
/// spare sighting and moves coherently releases the dormant site, because
/// "outgoing" candidates need only lie outside 2 * radius - they need not
/// originate at the site or be the same object.
#[test]
fn unrelated_ball_right_after_spare_sighting_releases_site() {
    init_capture();
    let (mut t, f) = dormant_spare_site(5);
    assert!(captured("dormant static site") >= 1);
    assert_eq!(captured("reactivated static site"), 0);
    let mut frame = f;
    // Spare seen at the site (bystander present).
    step(
        &mut t,
        &players_at(frame),
        &[ball(CameraId::Left, SPARE.0, SPARE.1, 0.4)],
        frame as f64 * STEP_MS,
    );
    frame += 1;
    // Next frames: only an unrelated ball 1.1 rad away, drifting slowly.
    for k in 0..10 {
        step(
            &mut t,
            &play_players(frame),
            &[ball(CameraId::Left, -0.1 + 0.01 * k as f32, 0.0, 0.8)],
            frame as f64 * STEP_MS,
        );
        frame += 1;
    }
    assert!(
        captured("(persistent outgoing motion)") >= 1,
        "documents current behaviour: unrelated distant motion released the site"
    );
}

/// Contrast case: the same unrelated ball appearing more than 750 ms after
/// the last spare sighting does NOT release the site.
#[test]
fn unrelated_ball_after_gap_does_not_release_site() {
    init_capture();
    let (mut t, f) = dormant_spare_site(5);
    let mut frame = f;
    step(
        &mut t,
        &players_at(frame),
        &[ball(CameraId::Left, SPARE.0, SPARE.1, 0.4)],
        frame as f64 * STEP_MS,
    );
    frame += 10; // 1 s gap
    for k in 0..10 {
        step(
            &mut t,
            &play_players(frame),
            &[ball(CameraId::Left, -0.1 + 0.01 * k as f32, 0.0, 0.8)],
            frame as f64 * STEP_MS,
        );
        frame += 1;
    }
    assert_eq!(captured("reactivated static site"), 0);
}

/// Pickup/occlusion path: ball disappears at the site with a player near,
/// then reappears moving with that player -> release "pickup occlusion".
#[test]
fn pickup_occlusion_releases_dormant_site() {
    init_capture();
    let (mut t, f) = dormant_spare_site(5);
    let mut frame = f;
    step(
        &mut t,
        &players_at(frame),
        &[ball(CameraId::Left, SPARE.0, SPARE.1, 0.6)],
        frame as f64 * STEP_MS,
    );
    frame += 1;
    // Player at the site, ball hidden.
    step(
        &mut t,
        &[player(1, SPARE.0, SPARE.1)],
        &[],
        frame as f64 * STEP_MS,
    );
    frame += 1;
    for k in 1..=3 {
        let yaw = SPARE.0 - 0.05 * k as f32;
        step(
            &mut t,
            &[player(1, yaw, 0.0)],
            &[ball(CameraId::Left, yaw, 0.0, 0.6)],
            frame as f64 * STEP_MS,
        );
        frame += 1;
    }
    assert!(
        captured("reactivated static site") >= 1,
        "pickup did not release the site"
    );
}

/// Competing detections while tracking: closer candidate wins over a more
/// confident far one (nearest-to-last scoring).
#[test]
fn competing_detections_nearest_wins_while_tracking() {
    init_capture();
    let mut t = tracker();
    let p = [player(1, 0.0, 0.0), player(2, 0.4, 0.0)];
    step(&mut t, &p, &[ball(CameraId::Left, 0.0, 0.0, 0.8)], 0.0);
    let out = step(
        &mut t,
        &p,
        &[
            ball(CameraId::Left, 0.4, 0.0, 0.95),
            ball(CameraId::Left, 0.02, 0.0, 0.3),
        ],
        100.0,
    )
    .expect("tracking");
    assert_eq!(out.yaw, 0.02);
}

/// Confidence competition after full Lost and expired grace: the highest
/// confidence candidate wins regardless of location (documents current
/// behaviour - this is how a confident decoy can be acquired).
#[test]
fn confidence_competition_after_full_lost_picks_highest_confidence() {
    init_capture();
    let mut t = tracker();
    let p = [player(1, 0.0, 0.0), player(2, 0.5, 0.0)];
    step(&mut t, &p, &[ball(CameraId::Left, 0.0, 0.0, 0.8)], 0.0);
    for k in 1..=4 {
        step(&mut t, &p, &[], k as f64 * 100.0);
    }
    let out = step(
        &mut t,
        &p,
        &[
            ball(CameraId::Left, 0.01, 0.0, 0.4),
            ball(CameraId::Left, 0.5, 0.0, 0.6),
        ],
        2_000.0,
    )
    .expect("acquire");
    assert_eq!(out.yaw, 0.5, "highest confidence acquired after grace");
}

/// Stale evidence: static accepts separated by more than the 750 ms episode
/// gap never accumulate into one qualifying episode, so no dormancy.
#[test]
fn stale_evidence_does_not_form_static_episodes() {
    init_capture();
    let mut t = tracker().with_max_coast_frames(1000);
    let players = players_at(0);
    for k in 0..60u64 {
        // One spare sighting every 800 ms (gap > 750 ms) for 48 s.
        step(
            &mut t,
            &players,
            &[ball(CameraId::Left, SPARE.0, SPARE.1, 0.6)],
            k as f64 * 800.0,
        );
    }
    assert_eq!(captured("dormant static site"), 0);
}

/// Cross-camera overlap: the same ball reported by both cameras at nearly
/// the same panorama position keeps one continuous track with no jump.
#[test]
fn cross_camera_overlap_keeps_single_continuous_track() {
    init_capture();
    let mut t = tracker();
    let p = [player(1, 0.0, 0.0)];
    let mut prev: Option<f32> = None;
    for f in 0..20u64 {
        let yaw = 0.01 * f as f32;
        let out = step(
            &mut t,
            &p,
            &[
                ball(CameraId::Left, yaw, 0.0, 0.7),
                ball(CameraId::Right, yaw + 0.004, 0.001, 0.75),
            ],
            f as f64 * STEP_MS,
        )
        .expect("tracking");
        assert_eq!(out.state, TrackState::Tracking);
        if let Some(p) = prev {
            assert!((out.yaw - p).abs() < 0.02, "jump at frame {f}");
        }
        prev = Some(out.yaw);
    }
}

/// Replay-rate invariance: identical inputs and timestamps produce
/// identical outputs regardless of execution speed.
#[test]
fn replay_rate_invariance() {
    init_capture();
    let run = |sleep_us: u64| {
        let mut t = tracker();
        let mut out = Vec::new();
        for f in 0..300u64 {
            let dets = if f % 7 == 0 {
                vec![]
            } else {
                vec![ball(CameraId::Left, (f as f32 * 0.003).sin(), 0.0, 0.6)]
            };
            let b = step(&mut t, &players_at(f), &dets, f as f64 * 16.683);
            out.push(b.map(|b| (b.state, b.yaw.to_bits(), b.pitch.to_bits())));
            if sleep_us > 0 {
                std::thread::sleep(std::time::Duration::from_micros(sleep_us));
            }
        }
        out
    };
    assert_eq!(run(0), run(50));
}

/// Clock sensitivity (documents current behaviour): the same spare-ball
/// stream yields a dormant site on a stretched clock but not when the
/// timestamps are compressed ~13x, as observed between the recorded wall
/// clock and source video time on sample_02.
#[test]
fn static_site_dormancy_depends_on_clock_scale() {
    let run = |scale: f64| {
        init_capture();
        let mut t = tracker();
        let mut f = 0u64;
        for _ in 0..5 {
            for _ in 0..30 {
                step(
                    &mut t,
                    &players_at(f),
                    &[ball(CameraId::Left, SPARE.0, SPARE.1, 0.5)],
                    f as f64 * STEP_MS * scale,
                );
                f += 1;
            }
            for k in 0..30 {
                step(
                    &mut t,
                    &players_at(f),
                    &[ball(CameraId::Left, -0.1 + 0.01 * k as f32, 0.0, 0.8)],
                    f as f64 * STEP_MS * scale,
                );
                f += 1;
            }
        }
        captured("dormant static site")
    };
    assert!(run(1.0) >= 1, "baseline stream should go dormant");
    assert_eq!(run(1.0 / 13.0), 0, "compressed clock should not go dormant");
}

/// Documents current behaviour (known weakness): a stationary bystander
/// player standing next to the spare, plus the real ball moving near other
/// players elsewhere, satisfies the "pickup occlusion" release although
/// nobody picked the spare up.
#[test]
fn bystander_near_spare_triggers_pickup_release() {
    init_capture();
    let _ = build_spare_site(5, true);
    assert!(captured("dormant static site") >= 1);
    assert!(
        captured("(pickup occlusion)") >= 1,
        "documents current behaviour: bystander + distant ball released the site"
    );
}
