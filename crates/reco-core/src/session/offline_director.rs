//! Offline (CPU-only) replay of the dense lookahead director loop.
//!
//! [`replay_dense_lookahead`] feeds a recorded or replayed sequence of
//! per-frame [`WorldState`]s through any [`Panner`] using the same
//! schedule as `StitchSession::run`'s buffered lookahead path at
//! `frame_stride == 1`:
//!
//! - frame buffer of capacity `n + 1`, pre-filled with `n` frames;
//! - the panner runs `post_smooth_half = max(n / 2, 1)` frames ahead of
//!   rendering;
//! - each popped frame is backward-bridged with the production
//!   [`bridge_ball_world`](super::run_loop::bridge_ball_world) against the
//!   *unbridged* future frames still in the buffer;
//! - the panner sees exactly those future frames as its lookahead window;
//! - rendered poses are the production
//!   [`centered_smooth`](super::run_loop::centered_smooth) of past +
//!   current + ahead raw poses;
//! - `PanContext::previous_position` follows the production interleaving
//!   (the last raw pose, overwritten by each rendered smoothed pose).
//!
//! Nothing here touches the GPU, so director experiments (OEV-006) can be
//! scored on recorded detections without rendering. Parity with the real
//! loop is asserted by the session tests (`offline_director_matches_live_*`).

use crate::calibration::MatchCalibration;
use crate::detect::director::ViewportPosition;
use crate::detect::panner::{PanContext, Panner};
use crate::detect::tracker::WorldState;

/// One input frame: the tracker-stage world state *before* backward
/// bridging, plus the clock values the live loop would have attached.
#[derive(Debug, Clone)]
pub struct OfflineFrame {
    /// Source frame index (0-based, dense).
    pub source_frame_index: u64,
    /// Analysis clock (ms) the live loop passes as `PanContext::timestamp_ms`.
    pub elapsed_ms: f64,
    /// Tracker output for this frame, pre-bridge.
    pub world: WorldState,
}

/// One rendered output frame.
#[derive(Debug, Clone)]
pub struct OfflinePose {
    /// Source frame index this pose was rendered for.
    pub source_frame_index: u64,
    /// The world state after backward bridging (what the panner saw).
    pub world: WorldState,
    /// The panner's raw decision for this frame.
    pub raw: ViewportPosition,
    /// The centered-smoothed pose the renderer would receive
    /// (equal to the recorded `PanDecision` / `PosePresented` pose).
    pub smoothed: ViewportPosition,
}

/// Replay `frames` through `panner` exactly as the dense (stride 1)
/// lookahead loop would, returning one [`OfflinePose`] per input frame in
/// render order.
///
/// `lookahead_frames` must be > 0 (the immediate, no-lookahead loop has no
/// smoothing and no bridging; call `Panner::decide` directly for that).
/// `source_fps` is the source frame rate used by the bridge speed gate.
/// `initial_previous` is the session's starting `previous_panner_pose`.
pub fn replay_dense_lookahead(
    frames: &[OfflineFrame],
    panner: &mut dyn Panner,
    lookahead_frames: usize,
    source_fps: f64,
    calibration: &MatchCalibration,
    initial_previous: ViewportPosition,
) -> Vec<OfflinePose> {
    assert!(
        lookahead_frames > 0,
        "replay_dense_lookahead requires lookahead > 0"
    );
    let n = lookahead_frames;
    let capacity = n + 1;
    let half = (n / 2).max(1);
    let total = frames.len();
    let bridge_fps = source_fps.max(1.0);

    // Buffer = frames[popped..produced).
    let mut produced = 0usize;
    let mut popped = 0usize;
    let mut previous = initial_previous;
    let mut anchor: Option<(u64, f32, f32)> = None;
    let mut queue: std::collections::VecDeque<(usize, WorldState, ViewportPosition)> =
        std::collections::VecDeque::new();
    let mut past: std::collections::VecDeque<ViewportPosition> = std::collections::VecDeque::new();
    let mut out = Vec::with_capacity(total);

    let buf_len = |produced: usize, popped: usize| produced - popped;
    let produce = |produced: &mut usize| -> bool {
        if *produced < total {
            *produced += 1;
            true
        } else {
            false
        }
    };
    let mut run_once =
        |produced: usize,
         popped: &mut usize,
         previous: &mut ViewportPosition,
         anchor: &mut Option<(u64, f32, f32)>,
         queue: &mut std::collections::VecDeque<(usize, WorldState, ViewportPosition)>| {
            if *popped >= produced {
                return;
            }
            let i = *popped;
            *popped += 1;
            let future: Vec<WorldState> = frames[*popped..produced]
                .iter()
                .map(|f| f.world.clone())
                .collect();
            let mut world = frames[i].world.clone();
            super::run_loop::bridge_ball_world(
                frames[i].source_frame_index,
                &mut world,
                &future,
                1,
                anchor,
                bridge_fps,
            );
            let ctx = PanContext {
                frame_index: frames[i].source_frame_index,
                timestamp_ms: frames[i].elapsed_ms,
                previous_position: *previous,
                calibration,
            };
            let pose = panner.decide_with_lookahead(&world, &future, &ctx);
            *previous = pose;
            queue.push_back((i, world, pose));
        };

    let render = |queue: &mut std::collections::VecDeque<(usize, WorldState, ViewportPosition)>,
                  past: &mut std::collections::VecDeque<ViewportPosition>,
                  previous: &mut ViewportPosition,
                  out: &mut Vec<OfflinePose>| {
        let (i, world, raw) = queue.pop_front().expect("non-empty queue");
        let smoothed = super::run_loop::centered_smooth(
            raw,
            queue.iter().take(half).map(|(_, _, p)| *p),
            past.iter().copied(),
        );
        past.push_back(raw);
        if past.len() > half {
            past.pop_front();
        }
        *previous = smoothed;
        out.push(OfflinePose {
            source_frame_index: frames[i].source_frame_index,
            world,
            raw,
            smoothed,
        });
    };

    // Pre-fill.
    for _ in 0..n {
        if !produce(&mut produced) {
            break;
        }
    }
    let mut eof = buf_len(produced, popped) < n;

    // Warm-up: panner runs `half` frames ahead.
    for _ in 0..half {
        if buf_len(produced, popped) == 0 {
            break;
        }
        if !eof && buf_len(produced, popped) < capacity && !produce(&mut produced) {
            eof = true;
        }
        run_once(
            produced,
            &mut popped,
            &mut previous,
            &mut anchor,
            &mut queue,
        );
    }

    // Steady state.
    loop {
        if !eof && buf_len(produced, popped) < capacity && !produce(&mut produced) {
            eof = true;
        }
        if buf_len(produced, popped) > 0 {
            run_once(
                produced,
                &mut popped,
                &mut previous,
                &mut anchor,
                &mut queue,
            );
        }
        if queue.len() > half || (eof && !queue.is_empty()) {
            render(&mut queue, &mut past, &mut previous, &mut out);
        } else if eof && buf_len(produced, popped) == 0 && queue.is_empty() {
            break;
        }
    }

    // Drain (unreachable with an unlimited frame budget, kept for parity).
    while !queue.is_empty() {
        render(&mut queue, &mut past, &mut previous, &mut out);
    }
    out
}
