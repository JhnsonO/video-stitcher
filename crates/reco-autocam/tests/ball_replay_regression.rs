//! OEV-001 causal regression fixtures for the real [`BallTracker`].
//!
//! These tests drive the real tracker with small synthetic detection
//! streams. OEV-001 recorded the pre-OEV-003 behaviour; OEV-003 (candidate
//! based static sites, associated + confirmed release, probation) flipped
//! the two known weaknesses (unrelated motion and bystanders releasing a
//! site) and replaced the clock-scale dependency case with frame-rate
//! invariance. Site transitions are observed through the tracker's own
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
const STEP_MS: f64 = 50.0;

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

/// Two moving players near yaw 0 (where play is).
fn play_players(frame: u64) -> Vec<TrackedEntity> {
    let wobble = 0.02 * ((frame % 4) as f32);
    vec![
        player(2, -0.05 + wobble, 0.0),
        player(3, 0.05 - wobble, 0.02),
    ]
}

/// Play players plus a bystander standing 0.03 rad from the spare.
fn players_with_bystander(frame: u64) -> Vec<TrackedEntity> {
    let mut p = play_players(frame);
    p.push(player(1, SPARE.0, SPARE.1 + 0.03));
    p
}

/// The real match ball wandering around near play (yaw -0.2..0.2).
fn real_ball_at(frame: u64) -> (f32, f32) {
    let t = frame as f32 * STEP_MS as f32 / 1000.0;
    (0.2 * (0.7 * t).sin(), 0.02 * (1.3 * t).cos())
}

/// Builds a dormant spare-ball site the way it happens in a match: the
/// spare is visible at its spot while the real ball is played elsewhere.
/// Returns (tracker, next_frame).
fn dormant_spare_site_with(step_ms: f64, seconds: f64, bystander: bool) -> (BallTracker, u64) {
    let mut t = tracker();
    let frames = (seconds * 1000.0 / step_ms) as u64;
    for f in 0..frames {
        let ms = f as f64 * step_ms;
        let tt = ms as f32 / 1000.0;
        let real = (0.2 * (0.7 * tt).sin(), 0.02 * (1.3 * tt).cos());
        let players = if bystander {
            players_with_bystander(f)
        } else {
            play_players(f)
        };
        step(
            &mut t,
            &players,
            &[
                ball(CameraId::Left, SPARE.0, SPARE.1, 0.4),
                ball(CameraId::Left, real.0, real.1, 0.8),
            ],
            ms,
        );
    }
    (t, frames)
}

fn dormant_spare_site() -> (BallTracker, u64) {
    dormant_spare_site_with(STEP_MS, 25.0, false)
}

/// Continue with only the real ball in play for `frames` frames.
fn play_on(t: &mut BallTracker, from: u64, frames: u64) -> u64 {
    for f in from..from + frames {
        let r = real_ball_at(f);
        step(
            t,
            &play_players(f),
            &[ball(CameraId::Left, r.0, r.1, 0.8)],
            f as f64 * STEP_MS,
        );
    }
    from + frames
}

// ------------------------------------------------------------ cases

/// Stationary spare visible while the real ball is played elsewhere
/// becomes a dormant site and is then rejected even when it is the only
/// candidate.
#[test]
fn stationary_spare_ball_becomes_dormant_and_is_rejected() {
    init_capture();
    let (mut t, f) = dormant_spare_site();
    assert!(
        captured("dormant static site") >= 1,
        "site never went dormant"
    );
    let mut accepted = 0;
    for k in 0..40 {
        let out = step(
            &mut t,
            &play_players(f + k),
            &[ball(CameraId::Left, SPARE.0, SPARE.1, 0.9)],
            (f + k) as f64 * STEP_MS,
        );
        accepted += usize::from(out.is_some_and(|b| b.state == TrackState::Tracking));
    }
    assert_eq!(accepted, 0, "dormant spare was re-accepted");
    assert_eq!(captured("reactivated static site"), 0);
}

/// Legitimate dead ball: the match ball rests for 30 s (e.g. waiting for a
/// goal kick) with nothing else moving. Never excluded, tracked throughout,
/// and still tracked when kicked back into play.
#[test]
fn legitimate_dead_ball_rest_then_kick_is_never_excluded() {
    init_capture();
    let mut t = tracker();
    let players = [player(1, 0.3, 0.05), player(2, 0.5, 0.0)];
    let mut tracking = 0;
    let frames = 600u64;
    for f in 0..frames {
        let out = step(
            &mut t,
            &players,
            &[ball(CameraId::Left, 0.3, 0.0, 0.7)],
            f as f64 * STEP_MS,
        );
        tracking += usize::from(out.is_some_and(|b| b.state == TrackState::Tracking));
    }
    assert_eq!(tracking, frames as usize);
    for k in 1..=20u64 {
        let yaw = 0.3 + 0.02 * k as f32;
        let out = step(
            &mut t,
            &players,
            &[ball(CameraId::Left, yaw, 0.0, 0.7)],
            (frames + k) as f64 * STEP_MS,
        )
        .expect("tracked");
        assert_eq!(out.state, TrackState::Tracking, "kick frame {k}");
    }
    assert_eq!(captured("dormant static site"), 0);
}

/// Genuine ball disappears briefly (within coast) and returns.
#[test]
fn genuine_ball_disappears_and_returns_within_coast() {
    init_capture();
    let mut t = tracker();
    let p = [player(1, 0.0, 0.0)];
    step(&mut t, &p, &[ball(CameraId::Left, 0.0, 0.0, 0.8)], 0.0);
    let c = step(&mut t, &p, &[], 50.0).expect("coast");
    assert_eq!(c.state, TrackState::Coasting);
    assert_eq!((c.yaw, c.pitch), (0.0, 0.0));
    let back = step(&mut t, &p, &[ball(CameraId::Left, 0.02, 0.0, 0.8)], 100.0).expect("back");
    assert_eq!(back.state, TrackState::Tracking);
}

/// Genuine ball lost fully, then returns far away after grace.
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

/// Real ball at the dormant spot kicked away; a chasing player 0.1 rad
/// behind (outside contact range, so this is the outgoing-motion path).
/// Tracked as soon as it leaves the zone; site released once it stays empty.
#[test]
fn real_ball_kicked_away_from_dormant_site_releases_it() {
    init_capture();
    let (mut t, f) = dormant_spare_site();
    assert!(captured("dormant static site") >= 1);
    let mut frame = f;
    let mut tracked_moving = 0;
    for k in 0..60u64 {
        let yaw = SPARE.0 - 0.015 * k as f32;
        let mut players = play_players(frame);
        players.push(player(4, yaw + 0.1, 0.0));
        let out = step(
            &mut t,
            &players,
            &[ball(CameraId::Left, yaw, 0.0, 0.8)],
            frame as f64 * STEP_MS,
        );
        if k >= 2 && out.is_some_and(|b| b.state == TrackState::Tracking) {
            tracked_moving += 1;
        }
        frame += 1;
    }
    assert!(
        tracked_moving >= 50,
        "moving ball tracked {tracked_moving}/58"
    );
    assert!(
        captured("(persistent outgoing motion)") >= 1,
        "kicked ball did not release the site"
    );
}

/// Ball dribbled away by the player standing at the site: released.
#[test]
fn real_ball_dribbled_away_from_dormant_site_releases_it() {
    init_capture();
    let (mut t, f) = dormant_spare_site();
    let mut frame = f;
    for k in 0..60u64 {
        let yaw = SPARE.0 - 0.01 * k as f32;
        let mut players = play_players(frame);
        players.push(player(1, yaw, 0.01));
        step(
            &mut t,
            &players,
            &[ball(CameraId::Left, yaw, 0.0, 0.8)],
            frame as f64 * STEP_MS,
        );
        frame += 1;
    }
    assert!(captured("reactivated static site") >= 1);
}

/// Player picks the ball up at the site (ball hidden), then carries it away.
#[test]
fn pickup_occlusion_releases_dormant_site() {
    init_capture();
    let (mut t, f) = dormant_spare_site();
    let mut frame = f;
    let mut players = play_players(frame);
    players.push(player(1, SPARE.0, SPARE.1 + 0.01));
    step(
        &mut t,
        &players,
        &[ball(CameraId::Left, SPARE.0, SPARE.1, 0.6)],
        frame as f64 * STEP_MS,
    );
    frame += 1;
    for _ in 0..6 {
        let mut players = play_players(frame);
        players.push(player(1, SPARE.0, SPARE.1 + 0.01));
        step(&mut t, &players, &[], frame as f64 * STEP_MS);
        frame += 1;
    }
    for k in 1..=60u64 {
        let yaw = SPARE.0 - 0.004 * k as f32;
        let mut players = play_players(frame);
        players.push(player(1, yaw, 0.01));
        step(
            &mut t,
            &players,
            &[ball(CameraId::Left, yaw, 0.0, 0.6)],
            frame as f64 * STEP_MS,
        );
        frame += 1;
    }
    assert!(
        captured("reactivated static site") >= 1,
        "pickup did not release"
    );
}

/// Flipped from OEV-001: an unrelated ball far from the site appearing
/// right after the last spare sighting no longer releases the site.
#[test]
fn unrelated_ball_right_after_spare_sighting_does_not_release_site() {
    init_capture();
    let (mut t, f) = dormant_spare_site();
    assert!(captured("dormant static site") >= 1);
    let end = play_on(&mut t, f, 120);
    assert_eq!(captured("reactivated static site"), 0);
    // The spare is still excluded afterwards.
    let out = step(
        &mut t,
        &play_players(end),
        &[ball(CameraId::Left, SPARE.0, SPARE.1, 0.95)],
        end as f64 * STEP_MS,
    );
    assert!(out.is_none_or(|b| b.state != TrackState::Tracking));
}

/// Flipped from OEV-001: a bystander standing next to the spare while the
/// real ball moves near other players does not trigger a pickup release.
#[test]
fn bystander_near_spare_does_not_release_site() {
    init_capture();
    let (mut t, f) = dormant_spare_site_with(STEP_MS, 25.0, true);
    assert!(captured("dormant static site") >= 1);
    // Spare momentarily undetected; bystander stays; real ball keeps playing.
    for k in 0..120u64 {
        let frame = f + k;
        let r = real_ball_at(frame);
        let mut dets = vec![ball(CameraId::Left, r.0, r.1, 0.8)];
        if k % 40 >= 20 {
            dets.push(ball(CameraId::Left, SPARE.0, SPARE.1, 0.4));
        }
        step(
            &mut t,
            &players_with_bystander(frame),
            &dets,
            frame as f64 * STEP_MS,
        );
    }
    assert_eq!(captured("reactivated static site"), 0);
}

/// A second stationary detection 16 mrad from a dormant spare (a
/// neighbouring detection of the same object) is excluded too, while an
/// area 5 deg away is not.
#[test]
fn neighbouring_static_detection_joins_exclusion() {
    init_capture();
    let (mut t, f) = dormant_spare_site();
    let neighbour = (SPARE.0 - 0.016, SPARE.1);
    let mut frame = f;
    for _ in 0..60 {
        let r = real_ball_at(frame);
        step(
            &mut t,
            &play_players(frame),
            &[
                ball(CameraId::Left, SPARE.0, SPARE.1, 0.4),
                ball(CameraId::Left, neighbour.0, neighbour.1, 0.4),
                ball(CameraId::Left, r.0, r.1, 0.8),
            ],
            frame as f64 * STEP_MS,
        );
        frame += 1;
    }
    assert!(
        captured("neighbour of dormant site") >= 1,
        "neighbour not excluded"
    );
    // Only the neighbour visible: rejected.
    let out = step(
        &mut t,
        &play_players(frame),
        &[ball(CameraId::Left, neighbour.0, neighbour.1, 0.9)],
        frame as f64 * STEP_MS,
    );
    assert!(out.is_none_or(|b| b.state != TrackState::Tracking));
    // A ball 5 deg away from the spare (with a player on it) is accepted.
    let far = (SPARE.0 - 0.0873, SPARE.1);
    let mut accepted = false;
    for k in 1..=40u64 {
        let mut players = play_players(frame + k);
        players.push(player(4, far.0, far.1 + 0.02));
        let out = step(
            &mut t,
            &players,
            &[ball(CameraId::Left, far.0, far.1, 0.9)],
            (frame + k) as f64 * STEP_MS,
        );
        accepted |= out.is_some_and(|b| b.state == TrackState::Tracking);
    }
    assert!(accepted, "area 5 deg from the spare was excluded");
}

/// Probation: a site released by a (spurious) departure that settles again
/// within the probation window re-dormants immediately, without relearning
/// the 20 s of evidence.
#[test]
fn released_site_that_resettles_redormants_during_probation() {
    init_capture();
    let (mut t, f) = dormant_spare_site();
    let mut frame = f;
    // Kick-like departure, site stays empty for 3 s.
    for k in 0..60u64 {
        let yaw = SPARE.0 - 0.015 * k as f32;
        step(
            &mut t,
            &play_players(frame),
            &[ball(CameraId::Left, yaw, 0.0, 0.8)],
            frame as f64 * STEP_MS,
        );
        frame += 1;
    }
    assert!(captured("reactivated static site") >= 1);
    // The spare is back at the spot (e.g. returned by a ball boy) while play
    // continues: re-dormant within a couple of seconds.
    let before = captured("re-static during probation");
    for _ in 0..60u64 {
        let r = real_ball_at(frame);
        step(
            &mut t,
            &play_players(frame),
            &[
                ball(CameraId::Left, SPARE.0, SPARE.1, 0.4),
                ball(CameraId::Left, r.0, r.1, 0.8),
            ],
            frame as f64 * STEP_MS,
        );
        frame += 1;
    }
    assert_eq!(captured("re-static during probation"), before + 1);
}

/// Competing detections while tracking: closer candidate wins.
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

/// Confidence competition after full Lost and expired grace: highest
/// confidence wins (unchanged by OEV-003; a known lever for later work).
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
    assert_eq!(out.yaw, 0.5);
}

/// Stale evidence: sightings 800 ms apart (gap > 750 ms) never form a static
/// window, so no dormancy.
#[test]
fn stale_evidence_does_not_form_static_windows() {
    init_capture();
    let mut t = tracker().with_max_coast_frames(1000);
    for k in 0..60u64 {
        let r = real_ball_at(k * 16);
        step(
            &mut t,
            &play_players(k),
            &[
                ball(CameraId::Left, SPARE.0, SPARE.1, 0.6),
                ball(CameraId::Left, r.0, r.1, 0.8),
            ],
            k as f64 * 800.0,
        );
    }
    assert_eq!(captured("dormant static site"), 0);
}

/// Cross-camera overlap keeps one continuous track with no jump.
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

/// Replay-rate invariance: identical inputs and timestamps give identical
/// outputs regardless of execution speed.
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
                vec![
                    ball(CameraId::Left, (f as f32 * 0.003).sin(), 0.0, 0.6),
                    ball(CameraId::Left, SPARE.0, SPARE.1, 0.4),
                ]
            };
            let b = step(&mut t, &play_players(f), &dets, f as f64 * 16.683);
            out.push(b.map(|b| (b.state, b.yaw.to_bits(), b.pitch.to_bits())));
            if sleep_us > 0 {
                std::thread::sleep(std::time::Duration::from_micros(sleep_us));
            }
        }
        out
    };
    assert_eq!(run(0), run(50));
}

/// Replaces OEV-001's clock-scale dependency case: the same 25 s match
/// situation sampled at 30, 60 and 120 fps goes dormant at the same video
/// time (within one static window) and is not released.
#[test]
fn spare_dormancy_is_frame_rate_invariant() {
    let mut first_dormant_ms = Vec::new();
    for fps in [30.0f64, 60.0, 120.0] {
        init_capture();
        let step_ms = 1000.0 / fps;
        let mut t = tracker();
        let mut at = None;
        let frames = (25_000.0 / step_ms) as u64;
        for f in 0..frames {
            let ms = f as f64 * step_ms;
            let tt = ms as f32 / 1000.0;
            let real = (0.2 * (0.7 * tt).sin(), 0.02 * (1.3 * tt).cos());
            step(
                &mut t,
                &play_players(f),
                &[
                    ball(CameraId::Left, SPARE.0, SPARE.1, 0.4),
                    ball(CameraId::Left, real.0, real.1, 0.8),
                ],
                ms,
            );
            if at.is_none() && captured("dormant static site") > 0 {
                at = Some(ms);
            }
        }
        assert_eq!(
            captured("reactivated static site"),
            0,
            "{fps} fps: released"
        );
        first_dormant_ms.push(at.unwrap_or_else(|| panic!("{fps} fps: no dormancy")));
    }
    let min = first_dormant_ms.iter().cloned().fold(f64::MAX, f64::min);
    let max = first_dormant_ms.iter().cloned().fold(f64::MIN, f64::max);
    assert!(
        max - min <= 1_600.0,
        "dormancy time differs across frame rates: {first_dormant_ms:?}"
    );
}
