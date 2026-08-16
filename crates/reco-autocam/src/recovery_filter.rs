//! Trusted validation boundary for CUDA high-resolution ball recovery.
//!
//! The CUDA detector owns candidate generation because it has the native NV12
//! frame and GPU inference machinery. This decorator owns the sports-domain
//! decision about whether a candidate is trusted enough to mutate recovery
//! state or reach the tracker.

use reco_core::calibration::FieldRoi;
use reco_core::detect::detector::{CameraId, Detection, DetectorError, DetectorFrame, UnifiedDetector};
use reco_core::projection::point_in_polygon;
use reco_detect::OrtGpuDetector;

const PROVISIONAL_MAX_PREDICTION_DISTANCE: f32 = 0.12;
const PROVISIONAL_LINK_DISTANCE: f32 = 0.06;
const PROVISIONAL_MIN_MOTION: f32 = 0.0015;
const PROVISIONAL_CONFIRMATIONS: u8 = 2;
const STATIONARY_OUTSIDE_LIMIT: u8 = 2;
const CROSS_CAMERA_CONFIDENCE_MARGIN: f32 = 0.10;
const CROSS_CAMERA_MAX_AGE_CALLS: u64 = 2;

#[derive(Debug, Default, Clone, Copy)]
struct ProvisionalState {
    last: Option<(f32, f32)>,
    confirmations: u8,
    stationary: u8,
}

#[derive(Debug, Default, Clone, Copy)]
struct CrossCameraEvidence {
    confidence: f32,
    call_index: u64,
    strong: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BallDecision {
    StrongInside,
    PromotedOutside,
    RejectedNoTrajectory,
    RejectedTrajectoryGap,
    RejectedStationary,
    RejectedCrossCamera,
}

fn camera_index(camera: CameraId) -> usize {
    match camera {
        CameraId::Left => 0,
        CameraId::Right => 1,
    }
}

fn other_camera_index(camera: CameraId) -> usize {
    1 - camera_index(camera)
}

fn polygon_for<'a>(roi: &'a FieldRoi, camera: CameraId) -> &'a [[f64; 2]] {
    match camera {
        CameraId::Left => &roi.left,
        CameraId::Right => &roi.right,
    }
}

fn ball_inside_roi(detection: &Detection, roi: Option<&FieldRoi>) -> bool {
    let Some(roi) = roi else {
        return true;
    };
    let polygon = polygon_for(roi, detection.camera);
    polygon.len() < 3
        || point_in_polygon(
            [detection.center_x as f64, detection.center_y as f64],
            polygon,
        )
}

fn non_ball_inside_roi(
    detection: &Detection,
    roi: Option<&FieldRoi>,
    person_class_id: u16,
) -> bool {
    let Some(roi) = roi else {
        return true;
    };
    let polygon = polygon_for(roi, detection.camera);
    if polygon.len() < 3 {
        return true;
    }

    let cx = detection.center_x as f64;
    let cy = detection.center_y as f64;
    if detection.class_id == person_class_id {
        let half_h = detection.height as f64 * 0.5;
        let quarter_h = detection.height as f64 * 0.25;
        point_in_polygon([cx, cy + half_h], polygon)
            && point_in_polygon([cx, cy + quarter_h], polygon)
    } else {
        point_in_polygon([cx, cy], polygon)
    }
}

fn distance(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

fn choose_candidate<'a>(
    candidates: impl Iterator<Item = &'a Detection>,
    predicted: Option<(f32, f32)>,
) -> Option<Detection> {
    candidates
        .min_by(|a, b| {
            let score = |d: &&Detection| {
                if let Some(predicted) = predicted {
                    distance((d.center_x, d.center_y), predicted)
                } else {
                    1.0 - d.confidence
                }
            };
            score(a)
                .total_cmp(&score(b))
                .then_with(|| b.confidence.total_cmp(&a.confidence))
        })
        .copied()
}

/// Decorator used only for the opt-in CUDA high-resolution ball recovery path.
/// Candidate generation stays in `OrtGpuDetector`; trusted state mutation only
/// happens after this layer accepts a candidate.
pub(crate) struct ValidatedBallRecoveryDetector {
    inner: OrtGpuDetector,
    roi: Option<FieldRoi>,
    ball_class_id: u16,
    person_class_id: u16,
    provisional: [ProvisionalState; 2],
    cross_camera: [CrossCameraEvidence; 2],
    call_index: u64,
}

impl ValidatedBallRecoveryDetector {
    pub(crate) fn new(
        inner: OrtGpuDetector,
        roi: Option<FieldRoi>,
        ball_class_id: u16,
        person_class_id: u16,
    ) -> Self {
        Self {
            inner,
            roi,
            ball_class_id,
            person_class_id,
            provisional: [ProvisionalState::default(); 2],
            cross_camera: [CrossCameraEvidence::default(); 2],
            call_index: 0,
        }
    }

    fn validate_ball_candidates(
        &mut self,
        camera: CameraId,
        candidates: &[Detection],
    ) -> (Option<Detection>, BallDecision) {
        let predicted = self.inner.recovery_prediction(camera);
        let inside = choose_candidate(
            candidates
                .iter()
                .filter(|d| d.class_id == self.ball_class_id && ball_inside_roi(d, self.roi.as_ref())),
            predicted,
        );

        let idx = camera_index(camera);
        if let Some(candidate) = inside {
            self.provisional[idx] = ProvisionalState::default();
            self.cross_camera[idx] = CrossCameraEvidence {
                confidence: candidate.confidence,
                call_index: self.call_index,
                strong: true,
            };
            return (Some(candidate), BallDecision::StrongInside);
        }

        let outside = choose_candidate(
            candidates
                .iter()
                .filter(|d| d.class_id == self.ball_class_id && !ball_inside_roi(d, self.roi.as_ref())),
            predicted,
        );
        let Some(candidate) = outside else {
            return (None, BallDecision::RejectedNoTrajectory);
        };
        let Some(predicted) = predicted else {
            self.provisional[idx] = ProvisionalState::default();
            return (None, BallDecision::RejectedNoTrajectory);
        };

        if distance((candidate.center_x, candidate.center_y), predicted)
            > PROVISIONAL_MAX_PREDICTION_DISTANCE
        {
            self.provisional[idx] = ProvisionalState::default();
            return (None, BallDecision::RejectedTrajectoryGap);
        }

        let other = self.cross_camera[other_camera_index(camera)];
        if other.strong
            && self.call_index.saturating_sub(other.call_index) <= CROSS_CAMERA_MAX_AGE_CALLS
            && other.confidence > candidate.confidence + CROSS_CAMERA_CONFIDENCE_MARGIN
        {
            return (None, BallDecision::RejectedCrossCamera);
        }

        let state = &mut self.provisional[idx];
        match state.last {
            Some(last) if distance(last, (candidate.center_x, candidate.center_y)) <= PROVISIONAL_LINK_DISTANCE => {
                state.confirmations = state.confirmations.saturating_add(1);
                if distance(last, (candidate.center_x, candidate.center_y)) <= PROVISIONAL_MIN_MOTION {
                    state.stationary = state.stationary.saturating_add(1);
                } else {
                    state.stationary = 0;
                }
            }
            _ => {
                state.confirmations = 1;
                state.stationary = 0;
            }
        }
        state.last = Some((candidate.center_x, candidate.center_y));

        if state.stationary >= STATIONARY_OUTSIDE_LIMIT {
            *state = ProvisionalState::default();
            return (None, BallDecision::RejectedStationary);
        }
        if state.confirmations < PROVISIONAL_CONFIRMATIONS {
            return (None, BallDecision::RejectedNoTrajectory);
        }

        self.cross_camera[idx] = CrossCameraEvidence {
            confidence: candidate.confidence,
            call_index: self.call_index,
            strong: false,
        };
        (Some(candidate), BallDecision::PromotedOutside)
    }

    fn filter_non_balls(&self, detections: &[Detection]) -> Vec<Detection> {
        detections
            .iter()
            .copied()
            .filter(|d| d.class_id != self.ball_class_id)
            .filter(|d| non_ball_inside_roi(d, self.roi.as_ref(), self.person_class_id))
            .collect()
    }

    fn log_decision(&self, camera: CameraId, decision: BallDecision, candidate: Option<Detection>) {
        match candidate {
            Some(candidate) => log::info!(
                "BALL_CANDIDATE_DECISION camera={camera} decision={decision:?} confidence={:.3} center={:.4},{:.4}",
                candidate.confidence,
                candidate.center_x,
                candidate.center_y,
            ),
            None => log::debug!(
                "BALL_CANDIDATE_DECISION camera={camera} decision={decision:?}"
            ),
        }
    }
}

impl UnifiedDetector for ValidatedBallRecoveryDetector {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn detect(
        &mut self,
        camera: CameraId,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        self.call_index = self.call_index.saturating_add(1);

        let candidates = self.inner.detect(camera, frame)?;
        let mut output = self.filter_non_balls(&candidates);
        let (mut accepted_ball, mut decision) = self.validate_ball_candidates(camera, &candidates);

        // A normal full-frame ball can be a false positive outside the field.
        // If it was rejected before the CUDA detector had a chance to run its
        // recovery search, explicitly ask for local/tiled recovery candidates
        // from the same native frame and validate those through the same gate.
        if accepted_ball.is_none() && !self.inner.recovery_was_attempted(camera) {
            let recovery_candidates = self.inner.force_recovery_candidates(camera, frame)?;
            let validated = self.validate_ball_candidates(camera, &recovery_candidates);
            accepted_ball = validated.0;
            decision = validated.1;
        }

        self.log_decision(camera, decision, accepted_ball);
        if let Some(ball) = accepted_ball {
            self.inner.commit_ball_recovery(camera, &[ball]);
            output.push(ball);
        } else {
            self.inner.reject_ball_recovery(camera);
        }

        Ok(output)
    }

    fn class_names(&self) -> Option<&[String]> {
        self.inner.class_names()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ball(camera: CameraId, x: f32, y: f32, confidence: f32) -> Detection {
        Detection {
            camera,
            class_id: 32,
            confidence,
            center_x: x,
            center_y: y,
            width: 0.01,
            height: 0.01,
        }
    }

    #[test]
    fn center_roi_test_allows_airborne_candidate_to_be_classified_outside() {
        let roi = FieldRoi {
            left: vec![[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
            right: vec![],
        };
        assert!(ball_inside_roi(&ball(CameraId::Left, 0.5, 0.5, 0.8), Some(&roi)));
        assert!(!ball_inside_roi(&ball(CameraId::Left, 0.5, 0.1, 0.8), Some(&roi)));
    }

    #[test]
    fn candidate_choice_prefers_trajectory_over_confidence() {
        let near = ball(CameraId::Left, 0.51, 0.50, 0.55);
        let far = ball(CameraId::Left, 0.90, 0.90, 0.99);
        let candidates = [far, near];
        let chosen = choose_candidate(candidates.iter(), Some((0.50, 0.50))).unwrap();
        assert!((chosen.center_x - near.center_x).abs() < f32::EPSILON);
    }

    #[test]
    fn non_ball_player_anchor_keeps_original_bottom_policy() {
        let roi = FieldRoi {
            left: vec![[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
            right: vec![],
        };
        let player = Detection {
            camera: CameraId::Left,
            class_id: 0,
            confidence: 0.9,
            center_x: 0.5,
            center_y: 0.7,
            width: 0.1,
            height: 0.3,
        };
        assert!(!non_ball_inside_roi(&player, Some(&roi), 0));
    }
}
