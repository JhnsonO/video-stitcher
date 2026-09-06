//! Singleton ball tracker composing player-anchor
//! + nearest-to-last selection + [`Coaster`].
//!
//! Port of the Python POC at `/tmp/reco-ai-eval/build_tracker_video.py`
//! (see `build_trajectory`) into the `Tracker` contract from
//! [`reco_core::detect::tracker`].
//!
//! # Filter chain
//!
//! Each frame's detections pass through:
//!
//! 1. **Class filter** — only the tracker's `class_id` survives.
//! 2. **Position required** — detections whose
//!    [`MappedDetection::position`] is `None` (failed panorama
//!    projection) are dropped.
//! 3. **Player anchor** (optional) — if player anchors have been
//!    supplied via [`BallTracker::set_players`] and non-empty, a
//!    detection must be within `player_anchor_max_rad` of at least
//!    one player in panorama yaw/pitch space to survive. When no
//!    players have been supplied (no player provider attached, or a
//!    ball-only model), the filter is a no-op.
//! 4. **Nearest-to-last with max-jump** — among survivors, pick the
//!    one whose panorama position is closest to the last accepted
//!    tracked position, provided the jump is below `max_jump_rad`.
//!    On reacquisition after a coast, when a velocity estimate from
//!    consecutive genuine detections is available, ranking is instead
//!    performed against the time-projected position while the same
//!    static last-position `max_jump_rad` safety envelope remains in force.
//!    Cross-camera yaw/pitch is meaningful because the
//!    projection already unifies the coordinate frame, so same-cam
//!    vs cross-cam are scored identically (unlike the Python POC
//!    which worked in pixels and had to special-case cross-cam).
//! 5. **Coaster** — if no candidate survived this frame, hold the
//!    last known position for up to `max_coast_frames` frames, then
//!    transition to `Lost`.
//!
//! # Logging
//!
//! Following reco's explicit-decision principle, every state change
//! emits a log line at `info!` (acquisitions, losses) or `debug!`
//! (per-frame transitions). Rejection reasons for individual
//! detections log at `trace!` to keep the normal path quiet.
//!
//! [`MappedDetection::position`]: reco_core::detect::director::MappedDetection::position

use reco_core::detect::detector::CameraId;
use reco_core::detect::director::MappedDetection;
use reco_core::detect::tracker::{TrackState, TrackedEntity, Tracker};

use crate::trackers::filters::{CoastStatus, Coaster};

/// Default angular gate on jumps between frames (radians).
///
/// ~20° — a ball can legitimately cross a significant chunk of the
/// panorama in one detection interval when a long pass is in flight,
/// especially at typical 5-frame tracker sample cadence. Tighter
/// gates (Python POC used ~500 px in camera frame) caused frequent
/// false losses during fast plays.
pub const DEFAULT_MAX_JUMP_RAD: f32 = 0.35;

/// Default coast budget in sample-frames (tracker calls). At the
/// POC's every-5-source-frames cadence on 30 fps footage, 20
/// sample-frames ≈ 3.3 seconds of held position — matches what the
/// POC's 4-minute DJI+GoPro evaluations settled on.
pub const DEFAULT_COAST_FRAMES: u32 = 20;

/// Default player-anchor radius in radians (~11°). Equivalent to
/// the POC's 250-500 px pixel threshold on a 3840-wide frame.
pub const DEFAULT_PLAYER_ANCHOR_RAD: f32 = 0.20;

/// Singleton ball tracker emitting at most one
/// [`TrackedEntity`] per frame.
///
/// Internal state is the last accepted measurement, a coaster, and
/// the optional current-frame player anchors. Construct with
/// [`BallTracker::new`], optionally tune via the `with_*` builders,
/// and hand to the session as a `Box<dyn Tracker>`.
pub struct BallTracker {
    class_id: u16,
    coaster: Coaster,
    last: Option<LastKnown>,
    /// Velocity estimated only from consecutive fresh Tracking detections,
    /// in panorama radians per millisecond. Never updated from coast/bridge
    /// output and cleared after a coast reacquisition or full loss.
    last_velocity: Option<(f32, f32)>,
    max_jump_rad: f32,
    player_anchor_max_rad: f32,
    /// Current-frame player anchors in panorama yaw/pitch. Populated each
    /// frame by [`observe_world`](Tracker::observe_world) (the session
    /// calls it after the player tracker runs). Empty when no player
    /// tracker is registered → the player-anchor filter is a no-op.
    current_players: Vec<(f32, f32)>,
    /// Persistent age counter; singleton ball so `id` is always 0
    /// but `age_frames` ticks every frame we're actively tracking.
    age_frames: u64,
}

#[derive(Debug, Clone, Copy)]
struct LastKnown {
    yaw: f32,
    pitch: f32,
    origin: CameraId,
    timestamp_ms: f64,
}

impl BallTracker {
    /// Build a new ball tracker tracking the given `class_id` with
    /// default parameters.
    pub fn new(class_id: u16) -> Self {
        Self {
            class_id,
            coaster: Coaster::new(DEFAULT_COAST_FRAMES),
            last: None,
            last_velocity: None,
            max_jump_rad: DEFAULT_MAX_JUMP_RAD,
            player_anchor_max_rad: DEFAULT_PLAYER_ANCHOR_RAD,
            current_players: Vec::new(),
            age_frames: 0,
        }
    }

    /// Override the per-frame max jump gate (radians).
    ///
    /// Detections whose panorama yaw/pitch is further than this from
    /// the last accepted position are rejected. Cross-camera and
    /// same-camera candidates use the same gate because the
    /// underlying [`ViewportPosition`](reco_core::detect::director::ViewportPosition)
    /// yaw/pitch coordinate system is camera-agnostic.
    pub fn with_max_jump_rad(mut self, rad: f32) -> Self {
        self.max_jump_rad = rad.max(0.0);
        self
    }

    /// Override the coast budget.
    pub fn with_max_coast_frames(mut self, n: u32) -> Self {
        self.coaster = Coaster::new(n);
        self
    }

    /// Override the player-anchor radius (radians). Set to a large
    /// value (e.g. `f32::INFINITY`) to effectively disable while
    /// keeping the code path active.
    pub fn with_player_anchor_rad(mut self, rad: f32) -> Self {
        self.player_anchor_max_rad = rad.max(0.0);
        self
    }

    /// Supply the current frame's player anchors in panorama yaw/pitch.
    ///
    /// Called each frame by [`observe_world`](Tracker::observe_world) from
    /// the session's `WorldState`, after the player tracker has produced
    /// its output for this frame. Each call replaces the previous anchors -
    /// there is no accumulation. When no anchors are supplied, the
    /// player-anchor filter short-circuits to "accept" (no rejection).
    pub fn set_players(&mut self, players: &[TrackedEntity]) {
        self.current_players.clear();
        self.current_players
            .extend(players.iter().map(|p| (p.yaw, p.pitch)));
    }

    /// Score a candidate detection against the last known position.
    ///
    /// Lower = better. Returns `None` when the jump exceeds
    /// `max_jump_rad`. With no prior position, scoring is pure
    /// negative-confidence (highest-confidence detection wins).
    fn score(&self, det: &MappedDetection) -> Option<f32> {
        let pos = det.position?;
        match self.last {
            None => Some(-det.confidence),
            Some(last) => {
                let dy = pos.yaw - last.yaw;
                let dp = pos.pitch - last.pitch;
                let dist = (dy * dy + dp * dp).sqrt();
                if dist > self.max_jump_rad {
                    None
                } else {
                    // Balance proximity and confidence; the 0.1-rad
                    // weight on confidence picks the sharper detection
                    // when two candidates are within a pixel or two.
                    Some(dist - 0.1 * det.confidence)
                }
            }
        }
    }

    /// Score a coast-time reacquisition candidate against a time-projected
    /// position while preserving the existing static last-position jump gate.
    ///
    /// This is only called when a velocity estimate exists. A candidate can
    /// rank highly because it matches the projected trajectory, but it still
    /// cannot escape the same `max_jump_rad` envelope around the last genuine
    /// accepted observation that normal tracking uses.
    fn score_predicted(&self, det: &MappedDetection, timestamp_ms: f64) -> Option<f32> {
        let pos = det.position?;
        let last = self.last?;
        let (vy, vp) = self.last_velocity?;

        let static_dy = pos.yaw - last.yaw;
        let static_dp = pos.pitch - last.pitch;
        let static_dist = (static_dy * static_dy + static_dp * static_dp).sqrt();
        if static_dist > self.max_jump_rad {
            return None;
        }

        let elapsed_ms = (timestamp_ms - last.timestamp_ms).max(0.0) as f32;
        let predicted_yaw = last.yaw + vy * elapsed_ms;
        let predicted_pitch = last.pitch + vp * elapsed_ms;
        let dy = pos.yaw - predicted_yaw;
        let dp = pos.pitch - predicted_pitch;
        let predicted_dist = (dy * dy + dp * dp).sqrt();

        Some(predicted_dist - 0.1 * det.confidence)
    }

    /// Decide whether this detection survives the player-anchor gate.
    fn passes_player_anchor(&self, pos_yaw: f32, pos_pitch: f32) -> bool {
        if self.current_players.is_empty() {
            return true;
        }
        self.current_players.iter().any(|(py, pp)| {
            let dy = pos_yaw - *py;
            let dp = pos_pitch - *pp;
            (dy * dy + dp * dp).sqrt() <= self.player_anchor_max_rad
        })
    }
}

impl Tracker for BallTracker {
    fn update(&mut self, detections: &[MappedDetection], timestamp_ms: f64) -> Vec<TrackedEntity> {
        // Step 1-4: filter candidates down to survivors.
        let mut survivors: Vec<&MappedDetection> = Vec::with_capacity(detections.len());
        for det in detections {
            if det.class_id != self.class_id {
                continue;
            }
            let Some(pos) = det.position else {
                log::trace!(
                    "BallTracker: drop — projection failed (class={} conf={:.2})",
                    det.class_id,
                    det.confidence
                );
                continue;
            };
            if !self.passes_player_anchor(pos.yaw, pos.pitch) {
                log::trace!(
                    "BallTracker: drop off-player — yaw={:.3} pitch={:.3} nearest player > {:.3}rad",
                    pos.yaw,
                    pos.pitch,
                    self.player_anchor_max_rad
                );
                continue;
            }
            survivors.push(det);
        }

        // Step 5: normal tracking stays nearest-to-static-last. Only a
        // reacquisition after one or more coast frames may rank against the
        // velocity-projected position, and only when velocity is available.
        let was_coasting = self.coaster.frames_coasting() > 0;
        let use_predicted = was_coasting && self.last_velocity.is_some();
        let best: Option<&MappedDetection> = survivors
            .iter()
            .filter_map(|d| {
                let score = if use_predicted {
                    self.score_predicted(d, timestamp_ms)
                } else {
                    self.score(d)
                };
                score.map(|s| (s, *d))
            })
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, d)| d);

        // Step 6: lifecycle.
        if let Some(det) = best {
            let pos = det.position.expect("score() guarantees Some");
            let was_new_track = self.last.is_none();
            let previous_last = self.last;

            // Velocity is learned only from consecutive genuine accepted
            // detections. A coast-time reacquisition may use the pre-gap
            // velocity to rank candidates, but that velocity is discarded as
            // soon as the reacquisition is accepted so it cannot leak across
            // the observation gap.
            if was_coasting {
                self.last_velocity = None;
            } else if let Some(last) = previous_last {
                let dt_ms = timestamp_ms - last.timestamp_ms;
                self.last_velocity = if dt_ms > 0.0 {
                    let dt = dt_ms as f32;
                    Some(((pos.yaw - last.yaw) / dt, (pos.pitch - last.pitch) / dt))
                } else {
                    None
                };
            }

            self.coaster.accept_fresh();
            self.last = Some(LastKnown {
                yaw: pos.yaw,
                pitch: pos.pitch,
                origin: det.camera,
                timestamp_ms,
            });
            self.age_frames = self.age_frames.saturating_add(1);

            if was_new_track {
                log::info!(
                    "BallTracker: acquired yaw={:.3} pitch={:.3} cam={:?} conf={:.2}",
                    pos.yaw,
                    pos.pitch,
                    det.camera,
                    det.confidence
                );
            } else if was_coasting {
                log::debug!(
                    "BallTracker: reacquired after coast — yaw={:.3} pitch={:.3} cam={:?} conf={:.2}",
                    pos.yaw,
                    pos.pitch,
                    det.camera,
                    det.confidence
                );
            }

            return vec![TrackedEntity {
                id: 0,
                class_id: self.class_id,
                yaw: pos.yaw,
                pitch: pos.pitch,
                confidence: det.confidence,
                state: TrackState::Tracking,
                age_frames: self.age_frames,
                origin: det.camera,
            }];
        }

        // No fresh detection accepted this frame.
        match self.coaster.step_without_fresh() {
            CoastStatus::Coasting => {
                if let Some(last) = self.last {
                    log::trace!(
                        "BallTracker: coasting — held yaw={:.3} pitch={:.3} ({} frames)",
                        last.yaw,
                        last.pitch,
                        self.coaster.frames_coasting()
                    );
                    self.age_frames = self.age_frames.saturating_add(1);
                    vec![TrackedEntity {
                        id: 0,
                        class_id: self.class_id,
                        yaw: last.yaw,
                        pitch: last.pitch,
                        confidence: 0.0,
                        state: TrackState::Coasting,
                        age_frames: self.age_frames,
                        origin: last.origin,
                    }]
                } else {
                    // Coaster said Coasting but we have no last — only
                    // possible with a concurrent bug. Fail-soft to Lost.
                    log::warn!(
                        "BallTracker: coaster returned Coasting with no last — emitting Lost"
                    );
                    vec![]
                }
            }
            CoastStatus::Lost => {
                self.last_velocity = None;
                if let Some(last) = self.last.take() {
                    log::info!(
                        "BallTracker: track lost after {} coast frames (last yaw={:.3} pitch={:.3})",
                        self.coaster.frames_coasting(),
                        last.yaw,
                        last.pitch
                    );
                    // Age resets on full loss so the next acquisition
                    // starts a fresh count.
                    self.age_frames = 0;
                    vec![TrackedEntity {
                        id: 0,
                        class_id: self.class_id,
                        yaw: last.yaw,
                        pitch: last.pitch,
                        confidence: 0.0,
                        state: TrackState::Lost,
                        age_frames: 0,
                        origin: last.origin,
                    }]
                } else {
                    vec![]
                }
            }
            CoastStatus::Tracking => unreachable!("step_without_fresh never returns Tracking"),
        }
    }

    fn class_id(&self) -> u16 {
        self.class_id
    }

    /// Snapshot the current frame's players into the player-anchor
    /// filter. The session runs the player tracker before the ball
    /// tracker and hands the ball tracker a [`WorldState`] whose
    /// `players` field is already populated for this frame.
    ///
    /// A player tracker that is not registered leaves `world.players`
    /// empty, which `set_players` accepts — the downstream anchor
    /// filter short-circuits to "accept" when no players are known,
    /// preserving the Phase 2c behavior.
    ///
    /// [`WorldState`]: reco_core::detect::tracker::WorldState
    fn observe_world(&mut self, world: &reco_core::detect::tracker::WorldState) {
        self.set_players(&world.players);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::detect::director::ViewportPosition;

    fn det(camera: CameraId, yaw: f32, pitch: f32, conf: f32, cx: f32, cy: f32) -> MappedDetection {
        MappedDetection {
            camera,
            class_id: 0,
            confidence: conf,
            camera_center: (cx, cy),
            camera_size: (0.05, 0.05),
            position: Some(ViewportPosition {
                yaw,
                pitch,
                fov_degrees: None,
            }),
        }
    }

    #[test]
    fn empty_detections_produce_nothing() {
        let mut t = BallTracker::new(0);
        let out = t.update(&[], 0.0);
        assert!(out.is_empty());
    }

    #[test]
    fn first_detection_emits_tracking() {
        let mut t = BallTracker::new(0);
        let d = det(CameraId::Left, 0.2, 0.1, 0.8, 0.5, 0.5);
        let out = t.update(&[d], 0.0);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].state, TrackState::Tracking);
        assert_eq!(out[0].yaw, 0.2);
        assert_eq!(out[0].pitch, 0.1);
        assert_eq!(out[0].origin, CameraId::Left);
    }

    #[test]
    fn non_matching_class_id_ignored() {
        let mut t = BallTracker::new(32); // sports ball
        let d = det(CameraId::Left, 0.2, 0.1, 0.8, 0.5, 0.5);
        // d has class_id=0, tracker wants 32 — should be ignored.
        let out = t.update(&[d], 0.0);
        assert!(out.is_empty());
    }

    #[test]
    fn missing_position_ignored() {
        let mut t = BallTracker::new(0);
        let mut d = det(CameraId::Left, 0.2, 0.1, 0.8, 0.5, 0.5);
        d.position = None;
        let out = t.update(&[d], 0.0);
        assert!(out.is_empty());
    }

    #[test]
    fn coast_then_reacquire() {
        let mut t = BallTracker::new(0).with_max_coast_frames(3);
        // Frame 1: acquire.
        let d1 = det(CameraId::Left, 0.2, 0.1, 0.8, 0.5, 0.5);
        let out1 = t.update(&[d1], 0.0);
        assert_eq!(out1[0].state, TrackState::Tracking);
        // Frames 2-3: no detection, coasting.
        let out2 = t.update(&[], 16.6);
        assert_eq!(out2[0].state, TrackState::Coasting);
        let out3 = t.update(&[], 33.3);
        assert_eq!(out3[0].state, TrackState::Coasting);
        // Frame 4: reacquire — state back to Tracking.
        let d4 = det(CameraId::Left, 0.21, 0.11, 0.7, 0.51, 0.51);
        let out4 = t.update(&[d4], 50.0);
        assert_eq!(out4[0].state, TrackState::Tracking);
    }

    #[test]
    fn coast_then_lost() {
        let mut t = BallTracker::new(0).with_max_coast_frames(2);
        let d = det(CameraId::Left, 0.2, 0.1, 0.8, 0.5, 0.5);
        t.update(&[d], 0.0);
        t.update(&[], 16.6); // coast 1
        t.update(&[], 33.3); // coast 2
        let out = t.update(&[], 50.0);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].state, TrackState::Lost);
        // One more — already lost, nothing to emit.
        let out2 = t.update(&[], 66.6);
        assert!(out2.is_empty());
    }

    #[test]
    fn max_jump_rejects_implausible_detection() {
        let mut t = BallTracker::new(0).with_max_jump_rad(0.1);
        // Acquire at yaw=0.
        let d1 = det(CameraId::Left, 0.0, 0.0, 0.9, 0.5, 0.5);
        t.update(&[d1], 0.0);
        // Big jump to yaw=1.0 — exceeds 0.1 gate.
        let d2 = det(CameraId::Left, 1.0, 0.0, 0.9, 0.5, 0.5);
        let out = t.update(&[d2], 16.6);
        // No fresh accepted — tracker coasts on the last known.
        assert_eq!(out[0].state, TrackState::Coasting);
        assert_eq!(out[0].yaw, 0.0);
    }

    #[test]
    fn cross_camera_handoff_tracks() {
        let mut t = BallTracker::new(0).with_max_jump_rad(0.3);
        // Acquire on left at yaw=0.15.
        let d1 = det(CameraId::Left, 0.15, 0.0, 0.8, 0.9, 0.5);
        let out1 = t.update(&[d1], 0.0);
        assert_eq!(out1[0].origin, CameraId::Left);
        // Next frame: right camera reports the same ball at close yaw.
        // Even though pixel coords are totally different (ball now at
        // left edge of right frame), the panorama yaw distance (0.05)
        // is within max_jump — tracker must switch cameras.
        let d2 = det(CameraId::Right, 0.20, 0.0, 0.75, 0.05, 0.5);
        let out2 = t.update(&[d2], 16.6);
        assert_eq!(out2[0].state, TrackState::Tracking);
        assert_eq!(out2[0].origin, CameraId::Right);
    }

    #[test]
    fn player_anchor_rejects_far_ball_when_players_present() {
        let mut t = BallTracker::new(0).with_player_anchor_rad(0.1);
        // Inject one player at yaw=1.0.
        let player = TrackedEntity {
            id: 1,
            class_id: 0,
            yaw: 1.0,
            pitch: 0.0,
            confidence: 0.9,
            state: TrackState::Tracking,
            age_frames: 5,
            origin: CameraId::Right,
        };
        t.set_players(&[player]);
        // Ball at yaw=0.2 is 0.8 rad from player — rejected.
        let d = det(CameraId::Left, 0.2, 0.0, 0.9, 0.5, 0.5);
        let out = t.update(&[d], 0.0);
        assert!(out.is_empty());
    }

    #[test]
    fn player_anchor_no_op_when_no_players_set() {
        let mut t = BallTracker::new(0).with_player_anchor_rad(0.1);
        // No set_players() call — filter should not reject.
        let d = det(CameraId::Left, 0.2, 0.0, 0.9, 0.5, 0.5);
        let out = t.update(&[d], 0.0);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].state, TrackState::Tracking);
    }

    #[test]
    fn class_id_accessor() {
        let t = BallTracker::new(32);
        assert_eq!(t.class_id(), 32);
    }

    #[test]
    fn observe_world_populates_player_anchors() {
        use reco_core::detect::tracker::WorldState;
        let mut t = BallTracker::new(0).with_player_anchor_rad(0.1);
        let player = TrackedEntity {
            id: 7,
            class_id: 0,
            yaw: 1.0,
            pitch: 0.0,
            confidence: 0.9,
            state: TrackState::Tracking,
            age_frames: 3,
            origin: CameraId::Right,
        };
        let world = WorldState {
            ball: None,
            players: vec![player],
        };
        t.observe_world(&world);
        // A ball far from the (only) player must be rejected by the
        // anchor filter — proves observe_world propagated players.
        let d = det(CameraId::Left, 0.2, 0.0, 0.9, 0.5, 0.5);
        let out = t.update(&[d], 0.0);
        assert!(out.is_empty(), "observe_world did not populate anchors");
    }

    #[test]
    fn observe_world_empty_players_leaves_filter_as_noop() {
        use reco_core::detect::tracker::WorldState;
        let mut t = BallTracker::new(0).with_player_anchor_rad(0.1);
        // Pre-seed with anchors, then observe an empty world: set_players
        // replaces (does not accumulate), so the filter becomes a no-op.
        let player = TrackedEntity {
            id: 1,
            class_id: 0,
            yaw: 1.0,
            pitch: 0.0,
            confidence: 0.9,
            state: TrackState::Tracking,
            age_frames: 1,
            origin: CameraId::Right,
        };
        t.set_players(&[player]);
        t.observe_world(&WorldState::default());
        let d = det(CameraId::Left, 0.2, 0.0, 0.9, 0.5, 0.5);
        let out = t.update(&[d], 0.0);
        assert_eq!(out.len(), 1, "empty world should reset anchors");
    }

    #[test]
    fn prefers_closer_candidate_over_higher_confidence() {
        let mut t = BallTracker::new(0).with_max_jump_rad(1.0);
        // Acquire at yaw=0.0.
        let d0 = det(CameraId::Left, 0.0, 0.0, 0.9, 0.5, 0.5);
        t.update(&[d0], 0.0);
        // Two candidates: one high-conf far (0.4 rad), one low-conf close (0.05 rad).
        // Score balances proximity against confidence (0.1-rad weight);
        // the close candidate wins because proximity dominates.
        let far = det(CameraId::Left, 0.40, 0.0, 0.95, 0.5, 0.5);
        let near = det(CameraId::Left, 0.05, 0.0, 0.55, 0.5, 0.5);
        let out = t.update(&[far, near], 16.6);
        assert_eq!(out.len(), 1);
        assert!(
            (out[0].yaw - 0.05).abs() < 1e-6,
            "expected near, got {}",
            out[0].yaw
        );
    }

    #[test]
    fn coast_reacquire_prefers_velocity_consistent_ball_over_stationary_spare() {
        let mut t = BallTracker::new(0).with_max_jump_rad(0.35);
        // Two genuine consecutive observations establish +0.001 rad/ms yaw velocity.
        t.update(&[det(CameraId::Left, 0.0, 0.0, 0.9, 0.5, 0.5)], 0.0);
        t.update(&[det(CameraId::Left, 0.1, 0.0, 0.9, 0.5, 0.5)], 100.0);

        // One missed observation starts a coast; last genuine yaw remains 0.1.
        let coast = t.update(&[], 200.0);
        assert_eq!(coast[0].state, TrackState::Coasting);

        // At 300 ms the projected yaw is 0.3. The spare is much closer to
        // static last=0.1, but the moving real ball should win on trajectory.
        let spare = det(CameraId::Left, 0.11, 0.0, 0.95, 0.5, 0.5);
        let real = det(CameraId::Left, 0.30, 0.0, 0.70, 0.5, 0.5);
        let out = t.update(&[spare, real], 300.0);
        assert_eq!(out[0].state, TrackState::Tracking);
        assert!((out[0].yaw - 0.30).abs() < 1e-6, "expected moving ball");
    }

    #[test]
    fn normal_tracking_still_scores_against_static_last() {
        let mut t = BallTracker::new(0).with_max_jump_rad(1.0);
        t.update(&[det(CameraId::Left, 0.0, 0.0, 0.9, 0.5, 0.5)], 0.0);
        t.update(&[det(CameraId::Left, 0.1, 0.0, 0.9, 0.5, 0.5)], 100.0);

        // Velocity exists, but with no coast the existing static-last scoring
        // path must remain active: 0.15 is closer to last=0.1 than 0.30.
        let near_static = det(CameraId::Left, 0.15, 0.0, 0.60, 0.5, 0.5);
        let along_velocity = det(CameraId::Left, 0.30, 0.0, 0.95, 0.5, 0.5);
        let out = t.update(&[near_static, along_velocity], 200.0);
        assert!((out[0].yaw - 0.15).abs() < 1e-6, "normal path changed");
    }

    #[test]
    fn coast_reacquire_without_velocity_falls_back_to_static_last() {
        let mut t = BallTracker::new(0).with_max_jump_rad(1.0);
        // Only one genuine detection: no velocity can be established.
        t.update(&[det(CameraId::Left, 0.2, 0.0, 0.9, 0.5, 0.5)], 0.0);
        t.update(&[], 100.0);

        let near = det(CameraId::Left, 0.21, 0.0, 0.55, 0.5, 0.5);
        let far = det(CameraId::Left, 0.40, 0.0, 0.95, 0.5, 0.5);
        let out = t.update(&[near, far], 200.0);
        assert!((out[0].yaw - 0.21).abs() < 1e-6, "fallback path changed");
    }

    #[test]
    fn predicted_reacquire_still_obeys_static_max_jump_envelope() {
        let mut t = BallTracker::new(0).with_max_jump_rad(0.1);
        t.update(&[det(CameraId::Left, 0.0, 0.0, 0.9, 0.5, 0.5)], 0.0);
        // Establish a deliberately high velocity: +0.005 rad/ms.
        t.update(&[det(CameraId::Left, 0.05, 0.0, 0.9, 0.5, 0.5)], 10.0);
        t.update(&[], 20.0);

        // Prediction at 100ms is yaw=0.5, but that is 0.45rad from the
        // last real observation and must be rejected by the unchanged gate.
        let near_prediction = det(CameraId::Left, 0.50, 0.0, 0.99, 0.5, 0.5);
        let out = t.update(&[near_prediction], 100.0);
        assert_eq!(out[0].state, TrackState::Coasting);
        assert!((out[0].yaw - 0.05).abs() < 1e-6);
    }

    #[test]
    fn full_loss_clears_velocity_before_fresh_acquisition() {
        let mut t = BallTracker::new(0)
            .with_max_coast_frames(1)
            .with_max_jump_rad(0.1);
        t.update(&[det(CameraId::Left, 0.0, 0.0, 0.9, 0.5, 0.5)], 0.0);
        t.update(&[det(CameraId::Left, 0.05, 0.0, 0.9, 0.5, 0.5)], 10.0);
        let coast = t.update(&[], 20.0);
        assert_eq!(coast[0].state, TrackState::Coasting);
        let lost = t.update(&[], 30.0);
        assert_eq!(lost[0].state, TrackState::Lost);

        // Far beyond the old static gate, but after full loss this must be a
        // clean first acquisition with no stale velocity or last-position bias.
        let fresh = t.update(&[det(CameraId::Right, 1.2, 0.0, 0.8, 0.5, 0.5)], 40.0);
        assert_eq!(fresh[0].state, TrackState::Tracking);
        assert!((fresh[0].yaw - 1.2).abs() < 1e-6);
    }
}
