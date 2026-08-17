//! Provider-neutral, opt-in ball recovery fallback for offline processing.
//!
//! The normal detector -> tracker -> coast/backward-bridge path remains the
//! authority. This module is only consulted after that deterministic path has
//! failed to produce a ball for a sustained interval. A reasoner may choose
//! one of the supplied candidates (or `none`), but its answer is only a
//! hypothesis: deterministic validation must accept it before the session is
//! allowed to mutate `WorldState`.

use std::collections::BTreeMap;

use super::detector::CameraId;
use super::director::MappedDetection;
use super::tracker::{TrackedEntity, WorldState};

/// A trusted ball observation used as temporal context.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct BallRecoveryAnchor {
    pub frame_index: u64,
    pub yaw: f32,
    pub pitch: f32,
}

/// Panorama-space bounds used by the deterministic validation gate.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct BallRecoveryBounds {
    pub yaw_min: f32,
    pub yaw_max: f32,
    pub pitch_min: f32,
    pub pitch_max: f32,
}

/// One candidate the reasoner is allowed to select.
///
/// `stationary_observations` is intentionally part of the provider-neutral
/// contract. The current mapped-detection candidate generator leaves it at
/// zero; future high-resolution/tiled candidate generators can populate it
/// from temporal evidence without changing the validation API.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct BallRecoveryCandidate {
    pub id: u32,
    pub frame_index: u64,
    pub yaw: f32,
    pub pitch: f32,
    pub detector_confidence: f32,
    pub camera: CameraId,
    pub camera_center: (f32, f32),
    pub camera_size: (f32, f32),
    pub stationary_observations: u32,
}

/// Relation of a selected context frame to the uncertain frame.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BallRecoveryFrameRelation {
    Past,
    Current,
    Future,
}

/// Temporal context available to a reasoner.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BallRecoveryFrameContext {
    pub frame_index: u64,
    pub relation: BallRecoveryFrameRelation,
    pub trusted_ball: Option<BallRecoveryAnchor>,
    pub players: Vec<TrackedEntity>,
}

/// Visual material a backend may resolve at source resolution.
///
/// v1 deliberately stores references rather than image bytes so the core
/// contract remains independent of CPU/GPU frame residency. A real API or
/// OEV-owned backend can resolve these references from the offline source;
/// the deterministic replay backend ignores them.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BallRecoveryVisualRef {
    FullFrame {
        frame_index: u64,
    },
    CandidateCrop {
        frame_index: u64,
        candidate_id: u32,
        camera: CameraId,
        /// Request the backend to prefer original/high-resolution pixels.
        high_resolution: bool,
    },
}

/// Complete provider-neutral request presented to the reasoner.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BallRecoveryRequest {
    pub current_frame_index: u64,
    pub fps: f64,
    pub uncertain_frames: u64,
    pub previous_trusted: Option<BallRecoveryAnchor>,
    pub future_trusted: Option<BallRecoveryAnchor>,
    pub bounds: Option<BallRecoveryBounds>,
    pub context_frames: Vec<BallRecoveryFrameContext>,
    pub visuals: Vec<BallRecoveryVisualRef>,
    pub candidates: Vec<BallRecoveryCandidate>,
}

/// Constrained answer returned by a reasoner: choose a supplied candidate or
/// explicitly choose none. Arbitrary coordinates are not part of the API.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct BallRecoveryHypothesis {
    pub candidate_id: Option<u32>,
    pub confidence: f32,
    #[serde(default)]
    pub rationale: Option<String>,
}

impl BallRecoveryHypothesis {
    pub fn none() -> Self {
        Self {
            candidate_id: None,
            confidence: 0.0,
            rationale: None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BallRecoveryReasonerError {
    #[error("ball recovery backend failed: {0}")]
    Backend(String),
}

/// Backend abstraction for VLM-assisted recovery.
pub trait BallRecoveryReasoner: Send {
    fn name(&self) -> &str;
    fn reason(
        &mut self,
        request: &BallRecoveryRequest,
    ) -> Result<BallRecoveryHypothesis, BallRecoveryReasonerError>;
}

/// One deterministic replay decision used by fixtures/tests.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BallRecoveryReplayDecision {
    pub frame_index: u64,
    pub candidate_id: Option<u32>,
    pub confidence: f32,
    #[serde(default)]
    pub rationale: Option<String>,
}

/// Credential-free backend keyed by source frame index.
#[derive(Debug, Default)]
pub struct ReplayBallRecoveryReasoner {
    decisions: BTreeMap<u64, BallRecoveryHypothesis>,
}

impl ReplayBallRecoveryReasoner {
    pub fn new(decisions: impl IntoIterator<Item = BallRecoveryReplayDecision>) -> Self {
        let decisions = decisions
            .into_iter()
            .map(|d| {
                (
                    d.frame_index,
                    BallRecoveryHypothesis {
                        candidate_id: d.candidate_id,
                        confidence: d.confidence,
                        rationale: d.rationale,
                    },
                )
            })
            .collect();
        Self { decisions }
    }

    pub fn from_json_str(json: &str) -> Result<Self, serde_json::Error> {
        let decisions: Vec<BallRecoveryReplayDecision> = serde_json::from_str(json)?;
        Ok(Self::new(decisions))
    }
}

impl BallRecoveryReasoner for ReplayBallRecoveryReasoner {
    fn name(&self) -> &str {
        "replay"
    }

    fn reason(
        &mut self,
        request: &BallRecoveryRequest,
    ) -> Result<BallRecoveryHypothesis, BallRecoveryReasonerError> {
        Ok(self
            .decisions
            .get(&request.current_frame_index)
            .cloned()
            .unwrap_or_else(BallRecoveryHypothesis::none))
    }
}

/// Trigger/validation knobs for the opt-in recovery path.
#[derive(Debug, Clone)]
pub struct BallRecoveryConfig {
    /// Number of consecutive post-bridge `None` frames before first call.
    pub min_uncertain_frames: u64,
    /// Minimum spacing between repeated calls during one unresolved gap.
    pub retry_interval_frames: u64,
    pub min_reasoner_confidence: f32,
    pub max_speed_deg_per_s: f32,
    pub max_interpolation_error_deg: f32,
    pub stationary_reject_observations: u32,
}

impl Default for BallRecoveryConfig {
    fn default() -> Self {
        Self {
            // This threshold is applied only after tracker coasting and
            // backward bridging have already failed, so it is not a
            // single-frame-miss trigger.
            min_uncertain_frames: 6,
            retry_interval_frames: 12,
            min_reasoner_confidence: 0.65,
            max_speed_deg_per_s: 178.0,
            max_interpolation_error_deg: 20.0,
            stationary_reject_observations: 8,
        }
    }
}

/// Deterministic validation result recorded for every invocation.
#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct BallRecoveryValidation {
    pub accepted: bool,
    pub candidate_id: Option<u32>,
    pub reason: String,
}

/// Result of one engine evaluation.
#[derive(Debug, Clone)]
pub enum BallRecoveryDecision {
    NotTriggered,
    Invoked {
        backend: String,
        request: BallRecoveryRequest,
        hypothesis: BallRecoveryHypothesis,
        validation: BallRecoveryValidation,
        accepted_candidate: Option<BallRecoveryCandidate>,
    },
}

/// Stateful trigger + reasoner + deterministic validation gate.
pub struct BallRecoveryEngine {
    reasoner: Box<dyn BallRecoveryReasoner>,
    config: BallRecoveryConfig,
    uncertain_frames: u64,
    last_invocation_frame: Option<u64>,
}

impl BallRecoveryEngine {
    pub fn new(reasoner: Box<dyn BallRecoveryReasoner>, config: BallRecoveryConfig) -> Self {
        Self {
            reasoner,
            config,
            uncertain_frames: 0,
            last_invocation_frame: None,
        }
    }

    pub fn backend_name(&self) -> &str {
        self.reasoner.name()
    }

    /// Consider one post-bridge world state. Only `world.ball == None`
    /// advances uncertainty; Tracking, Coasting and Bridged all suppress the
    /// reasoner because deterministic logic still has a usable state.
    pub fn consider(
        &mut self,
        world: &WorldState,
        mut request: BallRecoveryRequest,
    ) -> BallRecoveryDecision {
        if world.ball.is_some() {
            self.uncertain_frames = 0;
            self.last_invocation_frame = None;
            return BallRecoveryDecision::NotTriggered;
        }

        self.uncertain_frames = self.uncertain_frames.saturating_add(1);
        request.uncertain_frames = self.uncertain_frames;

        if self.uncertain_frames < self.config.min_uncertain_frames {
            return BallRecoveryDecision::NotTriggered;
        }

        if let Some(last) = self.last_invocation_frame {
            let spacing = request.current_frame_index.saturating_sub(last);
            if spacing < self.config.retry_interval_frames.max(1) {
                return BallRecoveryDecision::NotTriggered;
            }
        }
        self.last_invocation_frame = Some(request.current_frame_index);

        let backend = self.reasoner.name().to_string();
        let hypothesis = match self.reasoner.reason(&request) {
            Ok(h) => h,
            Err(error) => {
                return BallRecoveryDecision::Invoked {
                    backend,
                    request,
                    hypothesis: BallRecoveryHypothesis::none(),
                    validation: BallRecoveryValidation {
                        accepted: false,
                        candidate_id: None,
                        reason: error.to_string(),
                    },
                    accepted_candidate: None,
                };
            }
        };

        let (validation, accepted_candidate) = self.validate(&request, &hypothesis);
        BallRecoveryDecision::Invoked {
            backend,
            request,
            hypothesis,
            validation,
            accepted_candidate,
        }
    }

    fn validate(
        &self,
        request: &BallRecoveryRequest,
        hypothesis: &BallRecoveryHypothesis,
    ) -> (BallRecoveryValidation, Option<BallRecoveryCandidate>) {
        let Some(candidate_id) = hypothesis.candidate_id else {
            return (rejected(None, "reasoner selected none"), None);
        };

        let Some(candidate) = request
            .candidates
            .iter()
            .find(|candidate| candidate.id == candidate_id)
            .cloned()
        else {
            return (
                rejected(Some(candidate_id), "reasoner selected an unknown candidate"),
                None,
            );
        };

        if !hypothesis.confidence.is_finite()
            || hypothesis.confidence < self.config.min_reasoner_confidence
        {
            return (
                rejected(Some(candidate_id), "reasoner confidence below threshold"),
                None,
            );
        }

        if let Some(bounds) = request.bounds
            && (candidate.yaw < bounds.yaw_min
                || candidate.yaw > bounds.yaw_max
                || candidate.pitch < bounds.pitch_min
                || candidate.pitch > bounds.pitch_max)
        {
            return (
                rejected(Some(candidate_id), "candidate outside panorama bounds"),
                None,
            );
        }

        if self.config.stationary_reject_observations > 0
            && candidate.stationary_observations >= self.config.stationary_reject_observations
        {
            return (
                rejected(
                    Some(candidate_id),
                    "stationary/spare-ball candidate rejected",
                ),
                None,
            );
        }

        if let Some(previous) = request.previous_trusted
            && !speed_is_plausible(
                previous,
                BallRecoveryAnchor {
                    frame_index: candidate.frame_index,
                    yaw: candidate.yaw,
                    pitch: candidate.pitch,
                },
                request.fps,
                self.config.max_speed_deg_per_s,
            )
        {
            return (
                rejected(
                    Some(candidate_id),
                    "trajectory from previous trusted ball is implausible",
                ),
                None,
            );
        }

        if let Some(future) = request.future_trusted
            && !speed_is_plausible(
                BallRecoveryAnchor {
                    frame_index: candidate.frame_index,
                    yaw: candidate.yaw,
                    pitch: candidate.pitch,
                },
                future,
                request.fps,
                self.config.max_speed_deg_per_s,
            )
        {
            return (
                rejected(
                    Some(candidate_id),
                    "trajectory to future trusted ball is implausible",
                ),
                None,
            );
        }

        if let (Some(previous), Some(future)) = (request.previous_trusted, request.future_trusted)
            && future.frame_index > previous.frame_index
            && candidate.frame_index >= previous.frame_index
            && candidate.frame_index <= future.frame_index
        {
            let fraction = (candidate.frame_index - previous.frame_index) as f32
                / (future.frame_index - previous.frame_index) as f32;
            let expected_yaw = previous.yaw + (future.yaw - previous.yaw) * fraction;
            let expected_pitch = previous.pitch + (future.pitch - previous.pitch) * fraction;
            let error_deg =
                angular_distance_deg(candidate.yaw, candidate.pitch, expected_yaw, expected_pitch);
            if error_deg > self.config.max_interpolation_error_deg {
                return (
                    rejected(
                        Some(candidate_id),
                        "candidate inconsistent with bidirectional trajectory",
                    ),
                    None,
                );
            }
        }

        (
            BallRecoveryValidation {
                accepted: true,
                candidate_id: Some(candidate_id),
                reason: "accepted".to_string(),
            },
            Some(candidate),
        )
    }
}

fn rejected(candidate_id: Option<u32>, reason: &str) -> BallRecoveryValidation {
    BallRecoveryValidation {
        accepted: false,
        candidate_id,
        reason: reason.to_string(),
    }
}

fn angular_distance_deg(yaw_a: f32, pitch_a: f32, yaw_b: f32, pitch_b: f32) -> f32 {
    let dyaw = (yaw_b - yaw_a + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
        - std::f32::consts::PI;
    let dpitch = pitch_b - pitch_a;
    dyaw.hypot(dpitch).to_degrees()
}

fn speed_is_plausible(
    from: BallRecoveryAnchor,
    to: BallRecoveryAnchor,
    fps: f64,
    max_speed_deg_per_s: f32,
) -> bool {
    if fps <= 0.0 || to.frame_index <= from.frame_index {
        return false;
    }
    let seconds = (to.frame_index - from.frame_index) as f64 / fps;
    let distance = angular_distance_deg(from.yaw, from.pitch, to.yaw, to.pitch) as f64;
    distance / seconds <= max_speed_deg_per_s as f64
}

/// Pure candidate generation from existing mapped detections. This function
/// never reads or mutates tracker state; callers may later replace/augment
/// this source with tiled/high-resolution candidate generation.
pub fn generate_ball_candidates(
    frame_index: u64,
    detections: &[MappedDetection],
    ball_class_id: u16,
) -> Vec<BallRecoveryCandidate> {
    detections
        .iter()
        .filter(|detection| detection.class_id == ball_class_id)
        .filter_map(|detection| {
            detection.position.map(|position| BallRecoveryCandidate {
                id: 0,
                frame_index,
                yaw: position.yaw,
                pitch: position.pitch,
                detector_confidence: detection.confidence,
                camera: detection.camera,
                camera_center: detection.camera_center,
                camera_size: detection.camera_size,
                stationary_observations: 0,
            })
        })
        .enumerate()
        .map(|(index, mut candidate)| {
            candidate.id = index as u32;
            candidate
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect::director::ViewportPosition;

    fn candidate(id: u32, frame_index: u64, yaw: f32) -> BallRecoveryCandidate {
        BallRecoveryCandidate {
            id,
            frame_index,
            yaw,
            pitch: 0.0,
            detector_confidence: 0.2,
            camera: CameraId::Left,
            camera_center: (0.5, 0.5),
            camera_size: (0.01, 0.01),
            stationary_observations: 0,
        }
    }

    fn request(frame_index: u64, candidates: Vec<BallRecoveryCandidate>) -> BallRecoveryRequest {
        BallRecoveryRequest {
            current_frame_index: frame_index,
            fps: 60.0,
            uncertain_frames: 0,
            previous_trusted: Some(BallRecoveryAnchor {
                frame_index: frame_index - 6,
                yaw: 0.0,
                pitch: 0.0,
            }),
            future_trusted: Some(BallRecoveryAnchor {
                frame_index: frame_index + 6,
                yaw: 0.12,
                pitch: 0.0,
            }),
            bounds: Some(BallRecoveryBounds {
                yaw_min: -1.0,
                yaw_max: 1.0,
                pitch_min: -0.5,
                pitch_max: 0.5,
            }),
            context_frames: Vec::new(),
            visuals: Vec::new(),
            candidates,
        }
    }

    #[test]
    fn mapped_candidate_generation_is_class_filtered_and_unmapped_safe() {
        let detections = vec![
            MappedDetection {
                camera: CameraId::Left,
                class_id: 0,
                confidence: 0.3,
                camera_center: (0.4, 0.5),
                camera_size: (0.02, 0.02),
                position: Some(ViewportPosition {
                    yaw: 0.1,
                    pitch: 0.0,
                    fov_degrees: None,
                }),
            },
            MappedDetection {
                camera: CameraId::Right,
                class_id: 1,
                confidence: 0.9,
                camera_center: (0.5, 0.5),
                camera_size: (0.2, 0.4),
                position: Some(ViewportPosition::default()),
            },
            MappedDetection {
                camera: CameraId::Right,
                class_id: 0,
                confidence: 0.4,
                camera_center: (0.6, 0.5),
                camera_size: (0.02, 0.02),
                position: None,
            },
        ];
        let candidates = generate_ball_candidates(99, &detections, 0);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].id, 0);
        assert_eq!(candidates[0].frame_index, 99);
    }

    #[test]
    fn single_frame_miss_does_not_invoke_reasoner() {
        let reasoner = ReplayBallRecoveryReasoner::new([BallRecoveryReplayDecision {
            frame_index: 10,
            candidate_id: Some(0),
            confidence: 0.99,
            rationale: None,
        }]);
        let mut engine = BallRecoveryEngine::new(Box::new(reasoner), BallRecoveryConfig::default());
        let world = WorldState::default();
        assert!(matches!(
            engine.consider(&world, request(10, vec![candidate(0, 10, 0.06)])),
            BallRecoveryDecision::NotTriggered
        ));
    }

    #[test]
    fn replay_backend_can_accept_only_after_trigger_and_validation() {
        let reasoner = ReplayBallRecoveryReasoner::new([BallRecoveryReplayDecision {
            frame_index: 15,
            candidate_id: Some(0),
            confidence: 0.95,
            rationale: Some("candidate lies on the bidirectional path".into()),
        }]);
        let mut config = BallRecoveryConfig::default();
        config.min_uncertain_frames = 2;
        let mut engine = BallRecoveryEngine::new(Box::new(reasoner), config);
        let world = WorldState::default();

        assert!(matches!(
            engine.consider(&world, request(14, vec![candidate(0, 14, 0.05)])),
            BallRecoveryDecision::NotTriggered
        ));
        match engine.consider(&world, request(15, vec![candidate(0, 15, 0.06)])) {
            BallRecoveryDecision::Invoked {
                validation,
                accepted_candidate,
                ..
            } => {
                assert!(validation.accepted);
                assert_eq!(accepted_candidate.unwrap().id, 0);
            }
            BallRecoveryDecision::NotTriggered => panic!("expected recovery invocation"),
        }
    }

    #[test]
    fn validator_rejects_stationary_spare_ball() {
        let reasoner = ReplayBallRecoveryReasoner::new([BallRecoveryReplayDecision {
            frame_index: 30,
            candidate_id: Some(0),
            confidence: 0.99,
            rationale: None,
        }]);
        let mut config = BallRecoveryConfig::default();
        config.min_uncertain_frames = 1;
        let mut engine = BallRecoveryEngine::new(Box::new(reasoner), config);
        let mut spare = candidate(0, 30, 0.06);
        spare.stationary_observations = 8;

        match engine.consider(&WorldState::default(), request(30, vec![spare])) {
            BallRecoveryDecision::Invoked { validation, .. } => {
                assert!(!validation.accepted);
                assert!(validation.reason.contains("stationary"));
            }
            BallRecoveryDecision::NotTriggered => panic!("expected recovery invocation"),
        }
    }

    #[test]
    fn validator_rejects_candidate_inconsistent_with_future_observation() {
        let reasoner = ReplayBallRecoveryReasoner::new([BallRecoveryReplayDecision {
            frame_index: 50,
            candidate_id: Some(0),
            confidence: 0.99,
            rationale: None,
        }]);
        let mut config = BallRecoveryConfig::default();
        config.min_uncertain_frames = 1;
        let mut engine = BallRecoveryEngine::new(Box::new(reasoner), config);
        let implausible = candidate(0, 50, 0.9);

        match engine.consider(&WorldState::default(), request(50, vec![implausible])) {
            BallRecoveryDecision::Invoked { validation, .. } => {
                assert!(!validation.accepted);
            }
            BallRecoveryDecision::NotTriggered => panic!("expected recovery invocation"),
        }
    }
}
