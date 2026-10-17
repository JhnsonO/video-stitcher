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
//! 3. **Dormant static-site exclusion** — sites that repeatedly produce
//!    qualified static episodes while contradicted by play elsewhere are
//!    rejected until coherent reactivation evidence shows that the physical
//!    ball at the site has moved back into play.
//! 4. **Player anchor** (optional) — if player anchors have been
//!    supplied via [`BallTracker::set_players`] and non-empty, a
//!    detection must be within `player_anchor_max_rad` of at least
//!    one player in panorama yaw/pitch space to survive. When no
//!    players have been supplied (no player provider attached, or a
//!    ball-only model), the filter is a no-op.
//! 5. **Nearest-to-last with max-jump** — among survivors, pick the
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
//! 6. **Post-loss grace** — after a full `Lost` transition, retain the
//!    last genuine position/velocity briefly and apply a decaying soft
//!    trajectory preference during fresh acquisition. The preference is
//!    never a gate and expires to the same pure-confidence acquisition
//!    used when no prior position exists.
//! 7. **Coaster** — if no candidate survived this frame, hold the
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

use crate::trackers::active_ball::{ActiveBallConfig, ActiveBallSelector, Verdict};
use crate::trackers::filters::{CoastStatus, Coaster};
use crate::trackers::static_sites::DormantStaticSites;

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

/// Experimental post-loss trajectory-memory grace period.
///
/// 0.75s is a review/tuning default, not a fixed production constant.
pub const DEFAULT_REACQUIRE_GRACE_MS: f64 = 750.0;

/// Experimental initial trajectory contribution during post-loss acquisition.
///
/// 0.4 is a review/tuning default; the contribution decays to zero over
/// [`DEFAULT_REACQUIRE_GRACE_MS`].
pub const DEFAULT_REACQUIRE_INITIAL_WEIGHT: f32 = 0.40;

/// Distance-score penalty (radians) applied to a candidate with active-ball
/// weight 0 during normal tracking; scales linearly with `1 - weight`.
pub const ACTIVE_BALL_PENALTY_RAD: f32 = 0.35;

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
    /// output. It is retained across full loss only for the short reacquisition
    /// grace window and discarded once that memory is consumed or expires.
    last_velocity: Option<(f32, f32)>,
    /// Timestamp of the first full Lost transition while retained trajectory
    /// memory is active. `None` means normal tracking/acquisition semantics.
    lost_at_ms: Option<f64>,
    reacquire_grace_ms: f64,
    reacquire_initial_weight: f32,
    max_jump_rad: f32,
    player_anchor_max_rad: f32,
    /// Current-frame player anchors in panorama yaw/pitch. Populated each
    /// frame by [`observe_world`](Tracker::observe_world) (the session
    /// calls it after the player tracker runs). Empty when no player
    /// tracker is registered → the player-anchor filter is a no-op.
    current_players: Vec<(f32, f32)>,
    dormant_sites: DormantStaticSites,
    /// Optional active-ball selection (OEV-004). `None` = disabled.
    active: Option<ActiveBallSelector>,
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
            lost_at_ms: None,
            reacquire_grace_ms: DEFAULT_REACQUIRE_GRACE_MS,
            reacquire_initial_weight: DEFAULT_REACQUIRE_INITIAL_WEIGHT,
            max_jump_rad: DEFAULT_MAX_JUMP_RAD,
            player_anchor_max_rad: DEFAULT_PLAYER_ANCHOR_RAD,
            current_players: Vec::new(),
            dormant_sites: DormantStaticSites::default(),
            active: None,
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

    /// Override the post-loss trajectory-memory grace period.
    pub fn with_reacquire_grace_ms(mut self, grace_ms: f64) -> Self {
        self.reacquire_grace_ms = grace_ms.max(0.0);
        self
    }

    /// Override the initial post-loss trajectory contribution.
    pub fn with_reacquire_initial_weight(mut self, weight: f32) -> Self {
        self.reacquire_initial_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Override the player-anchor radius (radians). Set to a large
    /// value (e.g. `f32::INFINITY`) to effectively disable while
    /// keeping the code path active.
    pub fn with_player_anchor_rad(mut self, rad: f32) -> Self {
        self.player_anchor_max_rad = rad.max(0.0);
        self
    }

    /// Enable active-ball selection (OEV-004 experiment): body-attachment
    /// down-weighting and a size prior, both soft except near-certain body
    /// attachment. Needs person detections in the `update` input.
    pub fn with_active_ball_selection(mut self, cfg: ActiveBallConfig) -> Self {
        self.active = Some(ActiveBallSelector::new(cfg));
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
        self.dormant_sites.set_players(players);
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

    /// Return normalized trajectory proximity in [0, 1] for post-loss memory.
    ///
    /// `max_jump_rad` is reused only as a normalization radius here, never as
    /// a rejection gate: candidates outside the radius receive proximity 0
    /// and remain eligible on confidence.
    fn trajectory_proximity(&self, det: &MappedDetection, elapsed_ms: f64) -> Option<f32> {
        let pos = det.position?;
        let last = self.last?;
        let (vy, vp) = self.last_velocity?;
        let elapsed = elapsed_ms.max(0.0) as f32;
        let predicted_yaw = last.yaw + vy * elapsed;
        let predicted_pitch = last.pitch + vp * elapsed;
        let dy = pos.yaw - predicted_yaw;
        let dp = pos.pitch - predicted_pitch;
        let dist = (dy * dy + dp * dp).sqrt();

        if self.max_jump_rad <= f32::EPSILON {
            return Some(if dist <= f32::EPSILON { 1.0 } else { 0.0 });
        }

        Some((1.0 - dist / self.max_jump_rad).clamp(0.0, 1.0))
    }

    /// Higher-is-better soft acquisition score while post-loss memory exists.
    ///
    /// Linear decay is deliberate: it reaches exactly zero at the grace
    /// boundary, making `t >= T` unambiguously identical to pure confidence
    /// with no exponential tail. Missing trajectory memory also falls back to
    /// pure confidence.
    fn lost_reacquire_score(&self, det: &MappedDetection, timestamp_ms: f64) -> f32 {
        let Some(lost_at_ms) = self.lost_at_ms else {
            return det.confidence;
        };
        if self.reacquire_grace_ms <= 0.0 {
            return det.confidence;
        }

        let elapsed_ms = (timestamp_ms - lost_at_ms).max(0.0);
        if elapsed_ms >= self.reacquire_grace_ms {
            return det.confidence;
        }

        let Some(proximity) = self.trajectory_proximity(det, elapsed_ms) else {
            return det.confidence;
        };
        let remaining = 1.0 - (elapsed_ms / self.reacquire_grace_ms) as f32;
        let weight = self.reacquire_initial_weight * remaining;
        det.confidence * (1.0 - weight) + proximity * weight
    }

    /// Clear retained post-loss trajectory memory once its grace window ends.
    fn expire_lost_memory(&mut self, timestamp_ms: f64) {
        let Some(lost_at_ms) = self.lost_at_ms else {
            return;
        };
        let elapsed_ms = (timestamp_ms - lost_at_ms).max(0.0);
        if self.reacquire_grace_ms <= 0.0 || elapsed_ms >= self.reacquire_grace_ms {
            self.last = None;
            self.last_velocity = None;
            self.lost_at_ms = None;
            log::debug!("BallTracker: post-loss trajectory grace expired");
        }
    }

    /// Fold an active-ball weight into a lower-is-better score. With no prior
    /// position the score is `-confidence`, so it is scaled; otherwise a
    /// distance penalty proportional to `1 - weight` is added. Weight 1.0
    /// leaves the score bit-identical.
    fn apply_active_weight(&self, score: f32, weight: f32) -> f32 {
        if weight >= 1.0 {
            return score;
        }
        if self.last.is_none() {
            score * weight
        } else {
            score + (1.0 - weight) * ACTIVE_BALL_PENALTY_RAD
        }
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
        self.expire_lost_memory(timestamp_ms);
        self.dormant_sites
            .observe_frame(detections, self.class_id, timestamp_ms);

        if let Some(a) = self.active.as_mut() {
            a.observe(detections, self.class_id, timestamp_ms);
        }

        // Steps 1-4: filter candidates down to survivors. Dormant sites are a
        // hard gate at the same reject-before-scoring tier as player anchors.
        // Each survivor carries its active-ball weight (1.0 when disabled).
        let mut survivors: Vec<(&MappedDetection, f32)> = Vec::with_capacity(detections.len());
        for (index, det) in detections.iter().enumerate() {
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
            if self.dormant_sites.rejects((pos.yaw, pos.pitch)) {
                log::trace!(
                    "BallTracker: drop dormant-site — yaw={:.3} pitch={:.3}",
                    pos.yaw,
                    pos.pitch
                );
                continue;
            }
            let weight = match self.active.as_ref().map(|a| a.verdict(detections, index)) {
                None => 1.0,
                Some(Verdict::Weight(w)) => w,
                Some(Verdict::Reject) => {
                    log::trace!(
                        "BallTracker: drop body-attached — yaw={:.3} pitch={:.3}",
                        pos.yaw,
                        pos.pitch
                    );
                    continue;
                }
            };
            survivors.push((det, weight));
        }

        // Step 5: normal tracking stays nearest-to-static-last. Only a
        // reacquisition after one or more coast frames may rank against the
        // velocity-projected position, and only when velocity is available.
        // A full-Lost reacquisition is separate: retained trajectory memory is
        // a soft higher-is-better blend with no max-jump rejection.
        let was_coasting = self.coaster.frames_coasting() > 0;
        let was_lost = self.lost_at_ms.is_some();
        let best: Option<&MappedDetection> = if was_lost {
            survivors
                .iter()
                .map(|&(d, w)| (self.lost_reacquire_score(d, timestamp_ms) * w, d))
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(_, d)| d)
        } else {
            let use_predicted = was_coasting && self.last_velocity.is_some();
            survivors
                .iter()
                .filter_map(|&(d, w)| {
                    let score = if use_predicted {
                        self.score_predicted(d, timestamp_ms)
                    } else {
                        self.score(d)
                    };
                    score.map(|s| (self.apply_active_weight(s, w), d))
                })
                .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(_, d)| d)
        };

        // Step 6: lifecycle.
        if let Some(det) = best {
            let pos = det.position.expect("scored candidates guarantee Some");
            let was_new_track = was_lost || self.last.is_none();
            let previous_last = self.last;

            // Velocity is learned only from consecutive genuine accepted
            // detections. A coast-time reacquisition may use the pre-gap
            // velocity to rank candidates, but that velocity is discarded as
            // soon as the reacquisition is accepted so it cannot leak across
            // the observation gap. Full-Lost memory is likewise consumed on
            // the first accepted fresh acquisition.
            if was_lost {
                self.last_velocity = None;
                self.lost_at_ms = None;
            } else if was_coasting {
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
                // Emit the Lost transition once, then retain position/velocity
                // only as soft acquisition memory until the grace expires.
                if self.lost_at_ms.is_some() {
                    return vec![];
                }
                if let Some(last) = self.last {
                    self.lost_at_ms = Some(timestamp_ms);
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
    fn full_loss_retains_trajectory_memory_without_reemitting_lost() {
        let mut t = BallTracker::new(0).with_max_coast_frames(1);
        t.update(&[det(CameraId::Left, 0.0, 0.0, 0.9, 0.5, 0.5)], 0.0);
        t.update(&[det(CameraId::Left, 0.05, 0.0, 0.9, 0.5, 0.5)], 10.0);
        let coast = t.update(&[], 20.0);
        assert_eq!(coast[0].state, TrackState::Coasting);
        let lost = t.update(&[], 30.0);
        assert_eq!(lost[0].state, TrackState::Lost);
        assert!(t.last.is_some());
        assert!(t.last_velocity.is_some());
        assert_eq!(t.lost_at_ms, Some(30.0));

        let repeated = t.update(&[], 40.0);
        assert!(repeated.is_empty());
        assert!(t.last.is_some());
        assert!(t.last_velocity.is_some());
    }

    #[test]
    fn lost_grace_yields_to_much_higher_confidence_mid_window() {
        let mut t = BallTracker::new(0)
            .with_max_coast_frames(0)
            .with_max_jump_rad(0.35);
        t.update(&[det(CameraId::Left, 0.0, 0.0, 0.9, 0.5, 0.5)], 0.0);
        t.update(&[det(CameraId::Left, 0.1, 0.0, 0.9, 0.5, 0.5)], 100.0);
        let lost = t.update(&[], 200.0);
        assert_eq!(lost[0].state, TrackState::Lost);

        // At 300ms into the 750ms grace, linear decay gives w=0.24.
        // Trajectory candidate score = 0.3*0.76 + 1.0*0.24 = 0.468.
        // A zero-proximity alternative needs confidence >0.616 to win;
        // 0.95 is intentionally well beyond that threshold.
        let trajectory = det(CameraId::Left, 0.40, 0.0, 0.30, 0.5, 0.5);
        let alternative = det(CameraId::Right, 1.40, 0.0, 0.95, 0.5, 0.5);
        let out = t.update(&[trajectory, alternative], 500.0);
        assert_eq!(out[0].state, TrackState::Tracking);
        assert!(
            (out[0].yaw - 1.40).abs() < 1e-6,
            "confidence gap should win"
        );
    }

    #[test]
    fn lost_grace_expires_to_pure_confidence_at_boundary() {
        let mut inside = BallTracker::new(0)
            .with_max_coast_frames(0)
            .with_max_jump_rad(0.35);
        inside.update(&[det(CameraId::Left, 0.0, 0.0, 0.9, 0.5, 0.5)], 0.0);
        inside.update(&[det(CameraId::Left, 0.1, 0.0, 0.9, 0.5, 0.5)], 100.0);
        inside.update(&[], 200.0);
        let trajectory = det(CameraId::Left, 0.20, 0.0, 0.40, 0.5, 0.5);
        let higher_conf = det(CameraId::Right, 1.20, 0.0, 0.60, 0.5, 0.5);
        let inside_out = inside.update(&[trajectory, higher_conf], 300.0);
        assert!(
            (inside_out[0].yaw - 0.20).abs() < 1e-6,
            "trajectory boost should still apply before T"
        );

        let mut expired = BallTracker::new(0)
            .with_max_coast_frames(0)
            .with_max_jump_rad(0.35);
        expired.update(&[det(CameraId::Left, 0.0, 0.0, 0.9, 0.5, 0.5)], 0.0);
        expired.update(&[det(CameraId::Left, 0.1, 0.0, 0.9, 0.5, 0.5)], 100.0);
        expired.update(&[], 200.0);
        let trajectory = det(CameraId::Left, 0.85, 0.0, 0.40, 0.5, 0.5);
        let higher_conf = det(CameraId::Right, 1.20, 0.0, 0.60, 0.5, 0.5);
        let expired_out = expired.update(&[trajectory, higher_conf], 950.0);
        assert!(
            (expired_out[0].yaw - 1.20).abs() < 1e-6,
            "t >= T must be pure confidence"
        );
    }

    #[test]
    fn lost_grace_does_not_block_far_new_possession() {
        let mut t = BallTracker::new(0)
            .with_max_coast_frames(0)
            .with_max_jump_rad(0.05);
        t.update(&[det(CameraId::Left, 0.0, 0.0, 0.9, 0.5, 0.5)], 0.0);
        t.update(&[det(CameraId::Left, 0.05, 0.0, 0.9, 0.5, 0.5)], 100.0);
        let lost = t.update(&[], 200.0);
        assert_eq!(lost[0].state, TrackState::Lost);

        // Far outside the 0.05rad normalization radius, so proximity is 0.
        // It must remain eligible and reacquire rather than being hard-gated.
        let new_possession = det(CameraId::Right, 1.50, 0.0, 0.80, 0.5, 0.5);
        let out = t.update(&[new_possession], 300.0);
        assert_eq!(out[0].state, TrackState::Tracking);
        assert!((out[0].yaw - 1.50).abs() < 1e-6);
    }

    /// 2 s of a stationary candidate at (0, 0) at 20 Hz, optionally with a
    /// second candidate moving steadily elsewhere (play contradicting it).
    fn play_static_episode(t: &mut BallTracker, start_ms: f64, contradiction: bool) {
        for index in 0..40 {
            let timestamp_ms = start_ms + index as f64 * 50.0;
            let site = det(CameraId::Left, 0.0, 0.0, 0.95, 0.5, 0.5);
            if contradiction {
                let elsewhere = det(
                    CameraId::Right,
                    0.30 + index as f32 * 0.003,
                    0.0,
                    0.40,
                    0.5,
                    0.5,
                );
                t.update(&[site, elsewhere], timestamp_ms);
            } else {
                t.update(&[site], timestamp_ms);
            }
        }
        t.update(&[], start_ms + 2_500.0);
    }

    fn seed_dormant_site(t: &mut BallTracker, center: (f32, f32), last_seen_ms: f64) {
        t.dormant_sites.seed_dormant(center, last_seen_ms);
    }

    fn tracked_player(id: u64, yaw: f32, pitch: f32) -> TrackedEntity {
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

    #[test]
    fn recurring_contradicted_static_site_becomes_dormant_and_rejects() {
        let mut t = BallTracker::new(0).with_max_coast_frames(0);
        play_static_episode(&mut t, 0.0, true);
        play_static_episode(&mut t, 10_000.0, true);
        play_static_episode(&mut t, 25_000.0, true);

        assert_eq!(t.dormant_sites.dormant_count(), 1);

        let rejected = t.update(&[det(CameraId::Left, 0.0, 0.0, 0.99, 0.5, 0.5)], 28_000.0);
        assert!(
            rejected.is_empty() || rejected[0].state != TrackState::Tracking,
            "dormant-site candidate was accepted"
        );
    }

    #[test]
    fn single_long_static_episode_does_not_become_dormant() {
        let mut t = BallTracker::new(0);
        play_static_episode(&mut t, 0.0, true);

        assert_eq!(t.dormant_sites.dormant_count(), 0);
    }

    #[test]
    fn recurring_static_site_without_contradiction_does_not_become_dormant() {
        let mut t = BallTracker::new(0);
        play_static_episode(&mut t, 0.0, false);
        play_static_episode(&mut t, 10_000.0, false);
        play_static_episode(&mut t, 25_000.0, false);

        assert_eq!(t.dormant_sites.dormant_count(), 0);
    }

    #[test]
    fn dormant_site_releases_after_persistent_outgoing_motion() {
        let mut t = BallTracker::new(0);
        seed_dormant_site(&mut t, (0.0, 0.0), 0.0);

        t.update(&[det(CameraId::Left, 0.0, 0.0, 0.9, 0.5, 0.5)], 0.0);
        t.update(&[det(CameraId::Left, 0.01, 0.0, 0.9, 0.5, 0.5)], 100.0);
        t.update(&[det(CameraId::Left, 0.03, 0.0, 0.9, 0.5, 0.5)], 400.0);
        t.update(&[det(CameraId::Left, 0.05, 0.0, 0.9, 0.5, 0.5)], 800.0);
        let moved = t.update(&[det(CameraId::Left, 0.06, 0.0, 0.9, 0.5, 0.5)], 900.0);
        // The moved ball is tracked immediately (it left the exclusion
        // zone); the site itself is released once it stayed empty.
        assert_eq!(moved[0].state, TrackState::Tracking);
        assert!((moved[0].yaw - 0.06).abs() < 1e-6);
        t.update(&[det(CameraId::Left, 0.08, 0.0, 0.9, 0.5, 0.5)], 1_900.0);
        t.update(&[det(CameraId::Left, 0.10, 0.0, 0.9, 0.5, 0.5)], 3_000.0);
        assert_eq!(t.dormant_sites.dormant_count(), 0);
        // And a ball returning to the old site is accepted again.
        let back = t.update(&[det(CameraId::Left, 0.0, 0.0, 0.9, 0.5, 0.5)], 3_100.0);
        assert_eq!(back[0].state, TrackState::Tracking);
    }

    #[test]
    fn dormant_site_releases_after_pickup_occlusion() {
        let mut t = BallTracker::new(0);
        seed_dormant_site(&mut t, (0.0, 0.0), 0.0);

        t.set_players(&[tracked_player(1, 0.0, 0.0)]);
        t.update(&[det(CameraId::Left, 0.0, 0.0, 0.9, 0.5, 0.5)], 0.0);
        t.set_players(&[tracked_player(1, 0.0, 0.0)]);
        t.update(&[], 100.0);
        t.set_players(&[tracked_player(1, 0.02, 0.0)]);
        t.update(&[det(CameraId::Left, 0.02, 0.0, 0.9, 0.5, 0.5)], 500.0);
        t.set_players(&[tracked_player(1, 0.03, 0.0)]);
        let carried = t.update(&[det(CameraId::Left, 0.03, 0.0, 0.9, 0.5, 0.5)], 800.0);
        assert_eq!(carried[0].state, TrackState::Tracking);
        t.set_players(&[tracked_player(1, 0.05, 0.0)]);
        t.update(&[det(CameraId::Left, 0.05, 0.0, 0.9, 0.5, 0.5)], 2_900.0);
        assert_eq!(t.dormant_sites.dormant_count(), 0);
    }

    #[test]
    fn dormant_site_is_not_released_while_still_observed() {
        let mut t = BallTracker::new(0);
        seed_dormant_site(&mut t, (0.0, 0.0), 0.0);
        // The spare stays visible at the site while another ball moves
        // outward from right next to it: no release.
        for i in 0..20 {
            let ms = i as f64 * 50.0;
            t.update(
                &[
                    det(CameraId::Left, 0.0, 0.0, 0.5, 0.5, 0.5),
                    det(CameraId::Left, 0.012 + 0.004 * i as f32, 0.0, 0.9, 0.5, 0.5),
                ],
                ms,
            );
        }
        assert_eq!(t.dormant_sites.dormant_count(), 1);
    }

    #[test]
    fn all_candidates_rejected_uses_existing_lost_path() {
        let mut t = BallTracker::new(0).with_max_coast_frames(0);
        t.update(&[det(CameraId::Left, 0.0, 0.0, 0.9, 0.5, 0.5)], 0.0);
        seed_dormant_site(&mut t, (0.0, 0.0), 0.0);

        let lost = t.update(&[det(CameraId::Left, 0.0, 0.0, 0.99, 0.5, 0.5)], 100.0);
        assert_eq!(lost.len(), 1);
        assert_eq!(lost[0].state, TrackState::Lost);
        assert!(t.update(&[], 200.0).is_empty());
    }

    // ---------------------------------------------------------------- OEV-004
    // Active-ball selection fixtures. Camera coords double as panorama coords
    // (yaw = cx, pitch = -cy) so geometry is easy to read.

    const BALL: u16 = 32;
    const PERSON: u16 = 0;
    const DT: f64 = 1000.0 / 59.94;

    fn expected_size(cy: f32) -> f32 {
        let m = crate::trackers::active_ball::SizePrior::SAMPLE02_FIT.left;
        (m.slope * cy + m.intercept).exp()
    }

    fn cand(class_id: u16, cx: f32, cy: f32, w: f32, h: f32, conf: f32) -> MappedDetection {
        MappedDetection {
            camera: CameraId::Left,
            class_id,
            confidence: conf,
            camera_center: (cx, cy),
            camera_size: (w, h),
            position: Some(ViewportPosition {
                yaw: cx,
                pitch: -cy,
                fov_degrees: None,
            }),
        }
    }

    /// Plausible ball at (cx, cy).
    fn ball_at(cx: f32, cy: f32) -> MappedDetection {
        let s = expected_size(cy);
        cand(BALL, cx, cy, s, s, 0.6)
    }

    fn person_at(cx: f32, cy: f32) -> MappedDetection {
        cand(PERSON, cx, cy, 0.05, 0.20, 0.9)
    }

    /// Point at relative (u, v) inside the person box centred at (px, py).
    fn on_body(px: f32, py: f32, u: f32, v: f32) -> (f32, f32) {
        (px + u * 0.05, py - 0.10 + v * 0.20)
    }

    fn field_tracker(active: bool) -> BallTracker {
        let t = BallTracker::new(BALL)
            .with_max_jump_rad(0.8)
            .with_max_coast_frames(DEFAULT_COAST_FRAMES);
        if active {
            t.with_active_ball_selection(crate::trackers::active_ball::ActiveBallConfig::sample02(
                PERSON,
            ))
        } else {
            t
        }
    }

    /// Knee patch rigidly attached to a walking player for 2 s while the real
    /// ball rolls nearby. Without selection the tracker stays on the knee; with
    /// it the tracker moves to the real ball.
    fn knee_scene(active: bool) -> (usize, usize) {
        let mut t = field_tracker(active);
        let (mut on_ball, mut on_knee) = (0, 0);
        for i in 0..120 {
            let ts = i as f64 * DT;
            let px = 0.30 + 0.0015 * i as f32;
            let (kx, ky) = on_body(px, 0.70, 0.05, 0.65);
            let knee = cand(BALL, kx, ky, 0.006, 0.006, 0.8);
            let ball = ball_at(0.55 + 0.001 * i as f32, 0.80);
            let mut dets = vec![person_at(px, 0.70), knee];
            // The real ball only appears after the knee has been tracked.
            if i >= 10 {
                dets.push(ball);
            }
            let out = t.update(&dets, ts);
            if i >= 60 {
                match out.first() {
                    Some(e)
                        if e.state == TrackState::Tracking
                            && (e.yaw - ball.camera_center.0).abs() < 1e-6 =>
                    {
                        on_ball += 1
                    }
                    Some(e) if e.state == TrackState::Tracking && (e.yaw - kx).abs() < 1e-6 => {
                        on_knee += 1
                    }
                    _ => {}
                }
            }
        }
        (on_ball, on_knee)
    }

    #[test]
    fn knee_patch_without_selection_stays_on_knee() {
        let (on_ball, on_knee) = knee_scene(false);
        assert_eq!(on_ball, 0);
        assert_eq!(on_knee, 60);
    }

    #[test]
    fn knee_patch_with_selection_switches_to_real_ball() {
        let (on_ball, on_knee) = knee_scene(true);
        assert_eq!(on_knee, 0);
        assert_eq!(on_ball, 60);
    }

    /// Ball controlled at the chest for ~0.6 s by a moving player, then drops
    /// and rolls away. It may be down-weighted while held but is never dropped,
    /// and stays tracked after separation.
    #[test]
    fn held_ball_is_never_rejected_and_tracks_after_separation() {
        let mut t = field_tracker(true);
        let mut ts = 0.0;
        for i in 0..36 {
            ts = i as f64 * DT;
            let px = 0.30 + 0.002 * i as f32;
            let (bx, by) = on_body(px, 0.70, 0.0, 0.35);
            let s = expected_size(by);
            let out = t.update(&[person_at(px, 0.70), cand(BALL, bx, by, s, s, 0.6)], ts);
            assert_eq!(out[0].state, TrackState::Tracking, "held frame {i}");
        }
        for k in 1..40 {
            let px = 0.372;
            let bx = px + 0.004 * k as f32;
            let by = (0.70 - 0.10 + 0.35 * 0.20 + 0.01 * k as f32).min(0.95);
            let out = t.update(&[person_at(px, 0.70), ball_at(bx, by)], ts + k as f64 * DT);
            assert_eq!(out[0].state, TrackState::Tracking, "separated frame {k}");
            assert!((out[0].yaw - bx).abs() < 1e-6);
        }
    }

    /// Dribble: ball at the feet moving with the player, then rolling ahead.
    #[test]
    fn dribble_is_never_rejected() {
        let mut t = field_tracker(true);
        for i in 0..180 {
            let px = 0.30 + 0.002 * i as f32;
            let ahead = if i < 120 {
                0.0
            } else {
                0.004 * (i - 120) as f32
            };
            let (bx, by) = on_body(px, 0.70, 0.2, 0.93);
            let out = t.update(
                &[person_at(px, 0.70), ball_at(bx + ahead, by)],
                i as f64 * DT,
            );
            assert_eq!(out[0].state, TrackState::Tracking, "frame {i}");
            assert!((out[0].yaw - (bx + ahead)).abs() < 1e-6);
        }
    }

    /// Boot swinging at the feet next to a tracked real ball: the real ball is
    /// chosen every frame.
    #[test]
    fn boot_swing_does_not_steal_real_ball() {
        let mut t = field_tracker(true);
        for i in 0..120 {
            let px = 0.30 + 0.002 * i as f32;
            let swing = 0.3 * (i as f32 * 0.4).sin();
            let (bx, by) = on_body(px, 0.70, swing, 0.95);
            let boot = cand(BALL, bx, by, 0.012, 0.008, 0.7);
            let ball = ball_at(0.40, 0.82);
            let out = t.update(&[person_at(px, 0.70), boot, ball], i as f64 * DT);
            assert_eq!(out[0].state, TrackState::Tracking);
            assert!((out[0].yaw - 0.40).abs() < 1e-6, "frame {i}");
        }
    }

    /// Boot only, no ball anywhere: the tracker keeps running (tracks, coasts
    /// or loses) without panicking.
    #[test]
    fn boot_only_runs_without_crash() {
        let mut t = field_tracker(true);
        for i in 0..300 {
            let px = 0.30 + 0.002 * i as f32;
            let (bx, by) = on_body(px, 0.70, 0.3 * (i as f32 * 0.4).sin(), 0.93);
            let dets = if i % 7 == 0 {
                vec![person_at(px, 0.70)]
            } else {
                vec![person_at(px, 0.70), cand(BALL, bx, by, 0.012, 0.008, 0.7)]
            };
            let out = t.update(&dets, i as f64 * DT);
            assert!(out.len() <= 1);
        }
    }

    /// Size prior: a much-too-small candidate loses to a plausible one, both on
    /// fresh acquisition (where it has higher confidence) and while tracking
    /// (where it is slightly closer).
    #[test]
    fn size_prior_prefers_plausible_candidate() {
        let tiny = |cx: f32, conf: f32| {
            let s = expected_size(0.5) * 0.25;
            cand(BALL, cx, 0.5, s, s, conf)
        };
        let plausible = |cx: f32, conf: f32| {
            let s = expected_size(0.5);
            cand(BALL, cx, 0.5, s, s, conf)
        };

        let mut t = field_tracker(true);
        let out = t.update(&[tiny(0.30, 0.9), plausible(0.40, 0.6)], 0.0);
        assert!((out[0].yaw - 0.40).abs() < 1e-6, "acquisition");

        let out = t.update(&[tiny(0.41, 0.6), plausible(0.43, 0.6)], DT);
        assert!((out[0].yaw - 0.43).abs() < 1e-6, "tracking");

        // Without selection the same frames pick by confidence / proximity.
        let mut base = field_tracker(false);
        let out = base.update(&[tiny(0.30, 0.9), plausible(0.40, 0.6)], 0.0);
        assert!((out[0].yaw - 0.30).abs() < 1e-6);
    }

    /// Ball-only input (no person detections, plausible sizes): selection is a
    /// no-op and the output is identical to the disabled tracker.
    #[test]
    fn selection_without_persons_matches_disabled_tracker() {
        let mut a = field_tracker(true);
        let mut b = field_tracker(false);
        for i in 0..200 {
            let x = 0.3 + 0.002 * i as f32;
            let dets = if i % 11 == 0 {
                vec![]
            } else {
                vec![ball_at(x, 0.6)]
            };
            let oa = a.update(&dets, i as f64 * DT);
            let ob = b.update(&dets, i as f64 * DT);
            assert_eq!(oa.len(), ob.len());
            for (x, y) in oa.iter().zip(&ob) {
                assert_eq!(x.state, y.state);
                assert_eq!(x.yaw.to_bits(), y.yaw.to_bits());
                assert_eq!(x.pitch.to_bits(), y.pitch.to_bits());
            }
        }
    }
}
