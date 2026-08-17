use reco_core::detect::ball_recovery::{
    BallRecoveryAnchor, BallRecoveryCandidate, BallRecoveryConfig, BallRecoveryDecision,
    BallRecoveryEngine, BallRecoveryFrameContext, BallRecoveryFrameRelation, BallRecoveryRequest,
    BallRecoveryVisualRef, ReplayBallRecoveryReasoner,
};
use reco_core::detect::detector::CameraId;
use reco_core::detect::tracker::WorldState;

fn candidate(id: u32, yaw: f32) -> BallRecoveryCandidate {
    BallRecoveryCandidate {
        id,
        frame_index: 120,
        yaw,
        pitch: 0.0,
        detector_confidence: 0.25,
        camera: CameraId::Left,
        camera_center: (0.5, 0.5),
        camera_size: (0.01, 0.01),
        stationary_observations: 0,
    }
}

#[test]
fn replay_fixture_drives_constrained_reasoner_through_validation_gate() {
    let fixture = include_str!("fixtures/ball_recovery_replay.json");
    let reasoner = ReplayBallRecoveryReasoner::from_json_str(fixture).unwrap();
    let mut config = BallRecoveryConfig::default();
    config.min_uncertain_frames = 1;
    let mut engine = BallRecoveryEngine::new(Box::new(reasoner), config);

    let previous = BallRecoveryAnchor {
        frame_index: 114,
        yaw: 0.0,
        pitch: 0.0,
    };
    let future = BallRecoveryAnchor {
        frame_index: 126,
        yaw: 0.12,
        pitch: 0.0,
    };
    let request = BallRecoveryRequest {
        current_frame_index: 120,
        fps: 60.0,
        uncertain_frames: 0,
        previous_trusted: Some(previous),
        future_trusted: Some(future),
        bounds: None,
        context_frames: vec![
            BallRecoveryFrameContext {
                frame_index: 114,
                relation: BallRecoveryFrameRelation::Past,
                trusted_ball: Some(previous),
                players: Vec::new(),
            },
            BallRecoveryFrameContext {
                frame_index: 120,
                relation: BallRecoveryFrameRelation::Current,
                trusted_ball: None,
                players: Vec::new(),
            },
            BallRecoveryFrameContext {
                frame_index: 126,
                relation: BallRecoveryFrameRelation::Future,
                trusted_ball: Some(future),
                players: Vec::new(),
            },
        ],
        visuals: vec![
            BallRecoveryVisualRef::FullFrame { frame_index: 114 },
            BallRecoveryVisualRef::FullFrame { frame_index: 120 },
            BallRecoveryVisualRef::FullFrame { frame_index: 126 },
            BallRecoveryVisualRef::CandidateCrop {
                frame_index: 120,
                candidate_id: 1,
                camera: CameraId::Left,
                high_resolution: true,
            },
        ],
        candidates: vec![candidate(0, 0.5), candidate(1, 0.06)],
    };

    match engine.consider(&WorldState::default(), request) {
        BallRecoveryDecision::Invoked {
            hypothesis,
            validation,
            accepted_candidate,
            ..
        } => {
            assert_eq!(hypothesis.candidate_id, Some(1));
            assert!(validation.accepted);
            assert_eq!(accepted_candidate.unwrap().id, 1);
        }
        BallRecoveryDecision::NotTriggered => panic!("expected replay invocation"),
    }
}
