//! Session wiring for the opt-in ball recovery fallback.

use super::ball_recovery::{BallRecoveryConfig, BallRecoveryEngine, BallRecoveryReasoner};
use crate::session::StitchSession;

impl StitchSession {
    /// Explicitly enable ball recovery for this session.
    ///
    /// Production behavior is unchanged unless a caller invokes this method.
    pub fn set_ball_recovery_reasoner(
        &mut self,
        reasoner: Box<dyn BallRecoveryReasoner>,
        config: BallRecoveryConfig,
    ) {
        log::info!(
            "StitchSession: opt-in ball recovery attached (backend={})",
            reasoner.name()
        );
        self.ball_recovery = Some(BallRecoveryEngine::new(reasoner, config));
    }

    /// Disable ball recovery and discard its uncertainty/retry state.
    pub fn clear_ball_recovery_reasoner(&mut self) {
        self.ball_recovery = None;
    }
}
