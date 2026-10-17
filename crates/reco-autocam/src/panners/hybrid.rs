//! Hybrid four-state director (OEV-006 arm C). Experimental, opt-in.
//!
//! A state machine that decides *what* the camera should frame and hands
//! a shaped [`WorldState`] to an inner [`FieldPanner`], which keeps doing
//! *how* the camera moves (dynamics, FOV, lead, containment). Expects
//! `world.players` to be the target pitch's owned players with persistent
//! ids (see `trackers::players`), and `world.ball` from the ball tracker.
//!
//! States:
//!
//! - **BallTracked** - the match ball is tracked (`Tracking` or
//!   `Bridged`). Frame the ball with the owned players near it. Restricting
//!   players to the ball's neighbourhood is what lets the camera follow a
//!   long pass or a counter instead of staying with the pack (the field
//!   panner otherwise ignores a ball far from the player cluster).
//! - **BallMissing** - the ball was lost less than `missing_max_s` ago.
//!   Frame a virtual ball extrapolated from the last ball velocity with
//!   decay, plus the owned players near it.
//! - **Uncertain** - lost for longer, or never seen. No ball; frame every
//!   owned player (a wider shot sized by their spread).
//! - **StoppageRestart** - owned players nearly still for
//!   `stoppage_after_s` and no moving ball: hold a stable shot on the
//!   point the camera was framing when play stopped.
//!
//! A state must be held for `min_dwell_s` before another non-ball state
//! can replace it; a regained ball switches to BallTracked immediately.

use std::collections::HashMap;

use reco_core::detect::director::ViewportPosition;
use reco_core::detect::panner::{PanContext, Panner};
use reco_core::detect::tracker::{TrackState, TrackedEntity, WorldState};

use super::field::FieldPanner;

/// Director state of [`HybridDirector`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirectorState {
    /// Match ball tracked.
    BallTracked,
    /// Ball briefly missing; following its extrapolation.
    BallMissing,
    /// Ball unknown; framing the owned players.
    Uncertain,
    /// Play stopped; holding a stable shot.
    StoppageRestart,
}

impl DirectorState {
    /// Short name for traces.
    pub fn name(self) -> &'static str {
        match self {
            DirectorState::BallTracked => "ball_tracked",
            DirectorState::BallMissing => "ball_missing",
            DirectorState::Uncertain => "uncertain",
            DirectorState::StoppageRestart => "stoppage_restart",
        }
    }
}

/// Tuning for [`HybridDirector`].
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Frame rate (frames per second).
    pub fps: f32,
    /// Players within this radius (rad) of the (virtual) ball are framed
    /// with it.
    pub ball_neighbourhood_rad: f32,
    /// Maximum time in BallMissing before Uncertain (s).
    pub missing_max_s: f32,
    /// Per-frame decay of the extrapolated ball velocity.
    pub missing_velocity_decay: f32,
    /// Minimum time a non-ball state is held (s).
    pub min_dwell_s: f32,
    /// Mean owned-player speed below which play counts as stopped (rad/s).
    pub stoppage_speed_rad_s: f32,
    /// Time below that speed before StoppageRestart (s).
    pub stoppage_after_s: f32,
    /// Ball speed below which the ball counts as static (rad/s).
    pub static_ball_rad_s: f32,
}

impl HybridConfig {
    /// Starting values for 9v9 on a small pitch at about 60 fps. These are
    /// frozen for the OEV-006 first pass; do not tune on held-out footage.
    pub fn oev006(fps: f32) -> Self {
        Self {
            fps,
            ball_neighbourhood_rad: 0.30,
            missing_max_s: 2.0,
            missing_velocity_decay: 0.97,
            min_dwell_s: 0.5,
            stoppage_speed_rad_s: 0.02,
            stoppage_after_s: 1.5,
            static_ball_rad_s: 0.02,
        }
    }
}

/// Hybrid director; see the module docs.
pub struct HybridDirector {
    inner: FieldPanner,
    cfg: HybridConfig,
    state: DirectorState,
    state_frames: u32,
    last_ball: Option<(f32, f32)>,
    ball_vel: (f32, f32),
    ball_speed: f32,
    missing_frames: u32,
    virtual_vel: (f32, f32),
    virtual_pos: Option<(f32, f32)>,
    prev_players: HashMap<u64, (f32, f32)>,
    slow_frames: u32,
    hold: Option<(f32, f32)>,
    last_pose: Option<ViewportPosition>,
}

fn near(players: &[TrackedEntity], p: (f32, f32), r: f32) -> Vec<TrackedEntity> {
    players
        .iter()
        .filter(|q| ((q.yaw - p.0).powi(2) + (q.pitch - p.1).powi(2)).sqrt() <= r)
        .copied()
        .collect()
}

fn virtual_ball(p: (f32, f32), confidence: f32, state: TrackState) -> TrackedEntity {
    TrackedEntity {
        id: u64::MAX,
        class_id: u16::MAX,
        yaw: p.0,
        pitch: p.1,
        confidence,
        state,
        age_frames: 0,
        origin: reco_core::detect::detector::CameraId::Left,
    }
}

impl HybridDirector {
    /// Wrap `inner` (normally the arm's FieldPanner preset).
    pub fn new(inner: FieldPanner, cfg: HybridConfig) -> Self {
        Self {
            inner,
            cfg,
            state: DirectorState::Uncertain,
            state_frames: 0,
            last_ball: None,
            ball_vel: (0.0, 0.0),
            ball_speed: 0.0,
            missing_frames: 0,
            virtual_vel: (0.0, 0.0),
            virtual_pos: None,
            prev_players: HashMap::new(),
            slow_frames: 0,
            hold: None,
            last_pose: None,
        }
    }

    /// Current state (for traces).
    pub fn state(&self) -> DirectorState {
        self.state
    }

    fn frames(&self, s: f32) -> u32 {
        (s * self.cfg.fps).round().max(1.0) as u32
    }

    fn observe(&mut self, world: &WorldState) -> bool {
        // Ball kinematics from genuine/bridged positions.
        let ball_good = world
            .ball
            .is_some_and(|b| matches!(b.state, TrackState::Tracking | TrackState::Bridged));
        if ball_good {
            let b = world.ball.expect("checked");
            if let Some((y, p)) = self.last_ball {
                let v = (b.yaw - y, b.pitch - p);
                self.ball_vel = (
                    0.7 * self.ball_vel.0 + 0.3 * v.0,
                    0.7 * self.ball_vel.1 + 0.3 * v.1,
                );
            }
            self.last_ball = Some((b.yaw, b.pitch));
            self.ball_speed =
                (self.ball_vel.0.powi(2) + self.ball_vel.1.powi(2)).sqrt() * self.cfg.fps;
            self.missing_frames = 0;
            self.virtual_pos = None;
        } else {
            self.missing_frames = self.missing_frames.saturating_add(1);
        }

        // Owned-player motion (persistent ids).
        let mut speeds = Vec::new();
        let mut now = HashMap::with_capacity(world.players.len());
        for p in &world.players {
            if let Some(&(y, q)) = self.prev_players.get(&p.id) {
                speeds.push(((p.yaw - y).powi(2) + (p.pitch - q).powi(2)).sqrt() * self.cfg.fps);
            }
            now.insert(p.id, (p.yaw, p.pitch));
        }
        self.prev_players = now;
        let mean_speed = if speeds.is_empty() {
            f32::INFINITY
        } else {
            speeds.iter().sum::<f32>() / speeds.len() as f32
        };
        if mean_speed < self.cfg.stoppage_speed_rad_s {
            self.slow_frames = self.slow_frames.saturating_add(1);
        } else {
            self.slow_frames = 0;
        }
        ball_good
    }

    fn next_state(&self, ball_good: bool) -> DirectorState {
        let stopped = self.slow_frames >= self.frames(self.cfg.stoppage_after_s)
            && (!ball_good || self.ball_speed < self.cfg.static_ball_rad_s);
        if stopped {
            DirectorState::StoppageRestart
        } else if ball_good {
            DirectorState::BallTracked
        } else if self.last_ball.is_some()
            && self.missing_frames <= self.frames(self.cfg.missing_max_s)
        {
            DirectorState::BallMissing
        } else {
            DirectorState::Uncertain
        }
    }

    fn shape(&mut self, world: &WorldState) -> WorldState {
        let r = self.cfg.ball_neighbourhood_rad;
        match self.state {
            DirectorState::BallTracked => {
                let b = world.ball.expect("ball_tracked has a ball");
                WorldState {
                    ball: Some(b),
                    players: near(&world.players, (b.yaw, b.pitch), r),
                }
            }
            DirectorState::BallMissing => {
                let start = self.virtual_pos.or(self.last_ball).unwrap_or((0.0, 0.0));
                if self.virtual_pos.is_none() {
                    self.virtual_vel = self.ball_vel;
                }
                self.virtual_vel.0 *= self.cfg.missing_velocity_decay;
                self.virtual_vel.1 *= self.cfg.missing_velocity_decay;
                let pos = (start.0 + self.virtual_vel.0, start.1 + self.virtual_vel.1);
                self.virtual_pos = Some(pos);
                let conf =
                    1.0 - self.missing_frames as f32 / self.frames(self.cfg.missing_max_s) as f32;
                WorldState {
                    ball: Some(virtual_ball(
                        pos,
                        conf.clamp(0.05, 1.0),
                        TrackState::Bridged,
                    )),
                    players: near(&world.players, pos, r),
                }
            }
            DirectorState::Uncertain => WorldState {
                ball: None,
                players: world.players.clone(),
            },
            DirectorState::StoppageRestart => {
                let hold = *self.hold.get_or_insert_with(|| {
                    self.last_pose
                        .map(|p| (p.yaw, p.pitch))
                        .or(self.last_ball)
                        .unwrap_or((0.0, 0.0))
                });
                WorldState {
                    ball: Some(virtual_ball(hold, 1.0, TrackState::Tracking)),
                    players: near(&world.players, hold, r),
                }
            }
        }
    }
}

impl Panner for HybridDirector {
    fn decide(&mut self, world: &WorldState, ctx: &PanContext<'_>) -> ViewportPosition {
        self.decide_with_lookahead(world, &[], ctx)
    }

    fn decide_with_lookahead(
        &mut self,
        world: &WorldState,
        future: &[WorldState],
        ctx: &PanContext<'_>,
    ) -> ViewportPosition {
        let ball_good = self.observe(world);
        let wanted = self.next_state(ball_good);
        let can_switch = wanted == DirectorState::BallTracked
            || self.state == DirectorState::BallTracked
            || self.state_frames >= self.frames(self.cfg.min_dwell_s);
        if wanted != self.state && can_switch {
            if self.state == DirectorState::StoppageRestart {
                self.hold = None;
            }
            self.state = wanted;
            self.state_frames = 0;
        } else {
            self.state_frames = self.state_frames.saturating_add(1);
        }
        let shaped = self.shape(world);
        // Shape the lookahead window the same way (ball neighbourhood when
        // a future ball exists), so the lead is computed on comparable
        // inputs.
        let r = self.cfg.ball_neighbourhood_rad;
        let shaped_future: Vec<WorldState> = future
            .iter()
            .map(|w| match w.ball {
                Some(b) if matches!(b.state, TrackState::Tracking | TrackState::Bridged) => {
                    WorldState {
                        ball: Some(b),
                        players: near(&w.players, (b.yaw, b.pitch), r),
                    }
                }
                _ => WorldState {
                    ball: shaped
                        .ball
                        .filter(|_| self.state != DirectorState::Uncertain),
                    players: shaped.players.clone(),
                },
            })
            .collect();
        let pose = self
            .inner
            .decide_with_lookahead(&shaped, &shaped_future, ctx);
        self.last_pose = Some(pose);
        pose
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panners::FieldPannerConfig;
    use reco_core::detect::detector::CameraId;

    fn ent(id: u64, yaw: f32, pitch: f32, state: TrackState) -> TrackedEntity {
        TrackedEntity {
            id,
            class_id: 0,
            yaw,
            pitch,
            confidence: 0.9,
            state,
            age_frames: 10,
            origin: CameraId::Left,
        }
    }

    fn cal() -> reco_core::calibration::MatchCalibration {
        use reco_core::calibration::{CameraParams, MatchCalibration, PlaneLayout};
        let cam = CameraParams {
            width: 1920,
            height: 1080,
            fx: 900.0,
            fy: 900.0,
            cx: 960.0,
            cy: 540.0,
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

    fn director() -> HybridDirector {
        let fp = FieldPanner::with_config(
            60.0,
            FieldPannerConfig::from_preset_name("v4_micro_damping").unwrap(),
        );
        HybridDirector::new(fp, HybridConfig::oev006(60.0))
    }

    fn ctx(cal: &reco_core::calibration::MatchCalibration, f: u64) -> PanContext<'_> {
        PanContext {
            frame_index: f,
            timestamp_ms: f as f64 * 16.68,
            previous_position: ViewportPosition::default(),
            calibration: cal,
        }
    }

    /// Pack of moving players around yaw 0.0 (moving slightly so play is live).
    fn pack(f: u64) -> Vec<TrackedEntity> {
        let wobble = 0.01 * ((f as f32) * 0.2).sin();
        (0..8)
            .map(|i| {
                ent(
                    i + 1,
                    -0.15 + 0.04 * i as f32 + wobble,
                    0.0,
                    TrackState::Tracking,
                )
            })
            .collect()
    }

    /// Frames until yaw first exceeds `target` (None if never), for a
    /// long pass from the pack at 0.0 to `dest` over 1 s from frame 60.
    fn reach(panner: &mut dyn Panner, striker: bool, dest: f32, target: f32) -> (Option<u64>, f32) {
        let cal = cal();
        let mut first = None;
        let mut last = 0.0;
        for f in 0..300 {
            let by = if f < 60 {
                0.0
            } else {
                (dest * (f - 60) as f32 / 60.0).min(dest)
            };
            let mut players = pack(f);
            if striker {
                players.push(ent(99, dest, 0.0, TrackState::Tracking));
            }
            let w = WorldState {
                ball: Some(ent(500, by, 0.0, TrackState::Tracking)),
                players,
            };
            last = panner.decide(&w, &ctx(&cal, f)).yaw;
            if first.is_none() && last > target {
                first = Some(f);
            }
        }
        (first, last)
    }

    #[test]
    fn long_pass_hybrid_reaches_receiver_no_later_than_field_panner() {
        let preset = || FieldPannerConfig::from_preset_name("v4_micro_damping").unwrap();
        for striker in [true, false] {
            let mut plain = FieldPanner::with_config(60.0, preset());
            let mut hybrid = director();
            let (tp, yp) = reach(&mut plain, striker, 0.8, 0.5);
            let (th, yh) = reach(&mut hybrid, striker, 0.8, 0.5);
            eprintln!(
                "striker={striker}: plain first>0.5 at {tp:?} end {yp:.3}; hybrid {th:?} end {yh:.3}"
            );
            let th = th.expect("hybrid must reach the receiver area");
            assert!(
                tp.is_none_or(|tp| th <= tp),
                "hybrid slower than field panner"
            );
            assert!(yh > 0.55, "hybrid should settle near the ball, got {yh}");
        }
    }

    #[test]
    fn states_progress_missing_then_uncertain_then_back() {
        let cal = cal();
        let mut d = director();
        let mut seen = Vec::new();
        for f in 0..400u64 {
            let ball = if !(60..350).contains(&f) {
                Some(ent(500, 0.002 * f as f32, 0.0, TrackState::Tracking))
            } else {
                None
            };
            d.decide(
                &WorldState {
                    ball,
                    players: pack(f),
                },
                &ctx(&cal, f),
            );
            seen.push(d.state());
        }
        assert_eq!(seen[59], DirectorState::BallTracked);
        assert_eq!(seen[100], DirectorState::BallMissing);
        assert_eq!(seen[300], DirectorState::Uncertain);
        assert_eq!(
            seen[351],
            DirectorState::BallTracked,
            "regained ball switches immediately"
        );
    }

    #[test]
    fn stoppage_holds_a_stable_shot() {
        let cal = cal();
        let mut d = director();
        let mut yaws = Vec::new();
        for f in 0..400u64 {
            let players: Vec<_> = (0..8)
                .map(|i| ent(i + 1, -0.15 + 0.04 * i as f32, 0.0, TrackState::Tracking))
                .collect();
            let ball = Some(ent(500, 0.05, 0.0, TrackState::Tracking));
            let p = d.decide(&WorldState { ball, players }, &ctx(&cal, f));
            if f > 250 {
                yaws.push(p.yaw);
            }
        }
        assert_eq!(d.state(), DirectorState::StoppageRestart);
        let spread = yaws.iter().cloned().fold(f32::MIN, f32::max)
            - yaws.iter().cloned().fold(f32::MAX, f32::min);
        assert!(spread < 0.01, "held shot moved {spread} rad");
    }
}
