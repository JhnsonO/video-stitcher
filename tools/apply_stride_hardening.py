from __future__ import annotations

import re
from pathlib import Path


def read(path: str) -> str:
    return Path(path).read_text()


def write(path: str, text: str) -> None:
    Path(path).write_text(text)


def replace(path: str, old: str, new: str, *, count: int = 1) -> None:
    text = read(path)
    actual = text.count(old)
    if actual < count:
        raise RuntimeError(f"{path}: expected at least {count} copies, found {actual}: {old[:100]!r}")
    text = text.replace(old, new, count)
    write(path, text)


def sub(path: str, pattern: str, repl: str, *, count: int = 1) -> None:
    text = read(path)
    text2, n = re.subn(pattern, repl, text, count=count, flags=re.S)
    if n != count:
        raise RuntimeError(f"{path}: regex expected {count} replacements, got {n}: {pattern[:120]!r}")
    write(path, text2)


# ---------------------------------------------------------------------------
# reco-autocam: production/configured stride timing. The workflow restores
# this file from main before this script runs, so every edit below is against
# the clean production baseline rather than the old test-only env experiment.
# ---------------------------------------------------------------------------
AUTOCAM = "crates/reco-autocam/src/lib.rs"
replace(
    AUTOCAM,
    "use std::io;\nuse std::path::Path;\n",
    "use std::io;\nuse std::path::Path;\n\n"
    "/// Highest sparse-analysis stride currently validated for production use.\n"
    "pub const MAX_FRAME_STRIDE: u64 = 4;\n\n"
    "/// Rebase an EMA alpha from one source-frame step to `stride` source-frame\n"
    "/// steps while preserving its continuous-time response.\n"
    "fn stride_alpha(alpha: f32, stride: u64) -> f32 {\n"
    "    if stride <= 1 {\n"
    "        return alpha;\n"
    "    }\n"
    "    1.0 - (1.0 - alpha).powf(stride as f32)\n"
    "}\n\n"
    "/// Rebase panner values expressed per decision frame. Geometric values\n"
    "/// stay unchanged; max pan velocity is handled by constructing the panner\n"
    "/// with the effective analysis FPS.\n"
    "fn rebase_panner_config_for_stride(\n"
    "    mut config: crate::panners::FieldPannerConfig,\n"
    "    stride: u64,\n"
    ") -> crate::panners::FieldPannerConfig {\n"
    "    if stride <= 1 {\n"
    "        return config;\n"
    "    }\n"
    "    config.cluster_alpha = stride_alpha(config.cluster_alpha, stride);\n"
    "    config.fov_alpha = stride_alpha(config.fov_alpha, stride);\n"
    "    config.velocity_alpha = stride_alpha(config.velocity_alpha, stride);\n"
    "    config.lead_alpha = stride_alpha(config.lead_alpha, stride);\n"
    "    config.ball_presence_attack = stride_alpha(config.ball_presence_attack, stride);\n"
    "    config.ball_presence_decay = config.ball_presence_decay.powf(stride as f32);\n"
    "    config\n"
    "}\n\n"
    "fn coast_frames_for_stride(stride: u64) -> u32 {\n"
    "    let stride = stride.max(1);\n"
    "    let base = crate::trackers::ball::DEFAULT_COAST_FRAMES as u64;\n"
    "    base.div_ceil(stride).max(1) as u32\n"
    "}\n",
)
replace(
    AUTOCAM,
    "    /// Run detection every N frames (default: 1).\n    pub detection_interval: u64,\n",
    "    /// Run detection every N analysis frames (default: 1).\n"
    "    pub detection_interval: u64,\n"
    "    /// Analyze every Nth source frame while rendering every source frame.\n"
    "    /// 1 preserves the original full-rate analysis path.\n"
    "    pub frame_stride: u64,\n",
)
replace(
    AUTOCAM,
    "            detection_interval: 1,\n            field_roi: None,\n",
    "            detection_interval: 1,\n            frame_stride: 1,\n            field_roi: None,\n",
)
replace(
    AUTOCAM,
    "    /// Set the field ROI for detection filtering.\n",
    "    /// Analyze every Nth source frame while retaining full-rate rendering.\n"
    "    /// Values are clamped to the currently validated 1..=4 range.\n"
    "    pub fn with_frame_stride(mut self, stride: u64) -> Self {\n"
    "        self.frame_stride = stride.clamp(1, MAX_FRAME_STRIDE);\n"
    "        self\n"
    "    }\n\n"
    "    /// Set the field ROI for detection filtering.\n",
)
replace(
    AUTOCAM,
    "    let field_roi = config.field_roi.as_ref();\n    let is_10bit = config.is_10bit;\n\n    let mut detection_active = false;\n",
    "    let field_roi = config.field_roi.as_ref();\n"
    "    let is_10bit = config.is_10bit;\n"
    "    let frame_stride = config.frame_stride.clamp(1, MAX_FRAME_STRIDE);\n"
    "    let panner_fps = fps / frame_stride as f32;\n"
    "    let coast_frames = coast_frames_for_stride(frame_stride);\n"
    "    if frame_stride > 1 {\n"
    "        log::info!(\n"
    "            \"Autocam sparse analysis: source_fps={fps:.3}, stride={frame_stride}, \\\n"
    "             analysis_fps={panner_fps:.3}, ball_coast_frames={coast_frames}\"\n"
    "        );\n"
    "    }\n\n"
    "    let mut detection_active = false;\n",
)
replace(
    AUTOCAM,
    "                let ball_tracker =\n                    crate::trackers::BallTracker::new(ball_id).with_max_jump_rad(0.8);\n",
    "                let ball_tracker = crate::trackers::BallTracker::new(ball_id)\n"
    "                    .with_max_jump_rad(0.8)\n"
    "                    .with_max_coast_frames(coast_frames);\n",
)
replace(
    AUTOCAM,
    "                log::info!(\n                    \"FieldPanner: framing={:?}, confidence_weighted={}, lock_pitch={}\",\n",
    "                let fp_config = rebase_panner_config_for_stride(fp_config, frame_stride);\n"
    "                log::info!(\n                    \"FieldPanner: framing={:?}, confidence_weighted={}, lock_pitch={}\",\n",
)
replace(
    AUTOCAM,
    "                let field_panner = crate::panners::FieldPanner::with_config(fps, fp_config);\n",
    "                let field_panner =\n                    crate::panners::FieldPanner::with_config(panner_fps, fp_config);\n",
)
replace(
    AUTOCAM,
    "                let ball_tracker =\n                    crate::trackers::BallTracker::new(ball_id).with_max_jump_rad(0.5);\n",
    "                let ball_tracker = crate::trackers::BallTracker::new(ball_id)\n"
    "                    .with_max_jump_rad(0.5)\n"
    "                    .with_max_coast_frames(coast_frames);\n",
)
replace(
    AUTOCAM,
    "                fp_config.ball_weight = 1.0;\n                let panner = crate::panners::FieldPanner::with_config(fps, fp_config);\n",
    "                fp_config.ball_weight = 1.0;\n"
    "                let fp_config = rebase_panner_config_for_stride(fp_config, frame_stride);\n"
    "                let panner = crate::panners::FieldPanner::with_config(panner_fps, fp_config);\n",
)
replace(
    AUTOCAM,
    "    #[test]\n    fn model_path_validation_rejects_empty_and_missing_paths() {",
    "    #[test]\n"
    "    fn stride_one_panner_rebase_is_identity() {\n"
    "        let config = crate::panners::FieldPannerConfig::broadcast();\n"
    "        assert_eq!(rebase_panner_config_for_stride(config.clone(), 1), config);\n"
    "        assert_eq!(\n"
    "            coast_frames_for_stride(1),\n"
    "            crate::trackers::ball::DEFAULT_COAST_FRAMES\n"
    "        );\n"
    "    }\n\n"
    "    #[test]\n"
    "    fn stride_rebase_preserves_per_source_time_constants() {\n"
    "        let mut config = crate::panners::FieldPannerConfig::broadcast();\n"
    "        config.cluster_alpha = 0.10;\n"
    "        config.fov_alpha = 0.20;\n"
    "        config.velocity_alpha = 0.30;\n"
    "        config.lead_alpha = 0.40;\n"
    "        config.ball_presence_attack = 0.15;\n"
    "        config.ball_presence_decay = 0.90;\n"
    "        let rebased = rebase_panner_config_for_stride(config, 2);\n"
    "        assert!((rebased.cluster_alpha - 0.19).abs() < 1e-6);\n"
    "        assert!((rebased.fov_alpha - 0.36).abs() < 1e-6);\n"
    "        assert!((rebased.velocity_alpha - 0.51).abs() < 1e-6);\n"
    "        assert!((rebased.lead_alpha - 0.64).abs() < 1e-6);\n"
    "        assert!((rebased.ball_presence_attack - 0.2775).abs() < 1e-6);\n"
    "        assert!((rebased.ball_presence_decay - 0.81).abs() < 1e-6);\n"
    "        assert_eq!(coast_frames_for_stride(2), 10);\n"
    "        assert_eq!(coast_frames_for_stride(3), 7);\n"
    "        assert_eq!(coast_frames_for_stride(4), 5);\n"
    "    }\n\n"
    "    #[test]\n    fn model_path_validation_rejects_empty_and_missing_paths() {",
)

# ---------------------------------------------------------------------------
# reco-core session configuration.
# ---------------------------------------------------------------------------
SESSION_MOD = "crates/reco-core/src/session/mod.rs"
replace(
    SESSION_MOD,
    "    /// Number of lookahead frames (0 = disabled).\n    pub(crate) lookahead_frames: usize,\n",
    "    /// Number of lookahead frames (0 = disabled).\n"
    "    pub(crate) lookahead_frames: usize,\n"
    "    /// Analyze every Nth source frame while still rendering every frame.\n"
    "    /// Values above 1 are used by the buffered lookahead loop, which\n"
    "    /// interpolates camera poses between sparse analysis decisions.\n"
    "    pub(crate) frame_stride: u64,\n",
)
replace(
    SESSION_MOD,
    "            lookahead_frames: 0,\n            frame_count: 0,\n",
    "            lookahead_frames: 0,\n            frame_stride: 1,\n            frame_count: 0,\n",
)

WIRING = "crates/reco-core/src/session/wiring.rs"
replace(
    WIRING,
    "    /// Attach a stacked-video replay recorder.\n",
    "    /// Set sparse analysis cadence while retaining full-rate rendering.\n"
    "    ///\n"
    "    /// `1` is the original behavior. Values above 1 require buffered\n"
    "    /// lookahead so the loop can interpolate full-rate camera poses between\n"
    "    /// analysis decisions. Production-validated values are currently 1..=4.\n"
    "    pub fn set_frame_stride(&mut self, stride: u64) {\n"
    "        self.frame_stride = stride.clamp(1, 4);\n"
    "    }\n\n"
    "    /// Attach a stacked-video replay recorder.\n",
)

# ---------------------------------------------------------------------------
# Buffered-frame metadata and sparse lookahead filtering.
# ---------------------------------------------------------------------------
FRAME_BUFFER = "crates/reco-core/src/session/frame_buffer.rs"
replace(
    FRAME_BUFFER,
    "pub(crate) struct BufferedFrame {\n    pub frame: StereoFrame,\n",
    "pub(crate) struct BufferedFrame {\n"
    "    pub frame: StereoFrame,\n"
    "    /// Original source-frame index within this run.\n"
    "    pub source_frame_index: u64,\n"
    "    /// True when this frame ran the detector/tracker analysis chain.\n"
    "    pub analysis_frame: bool,\n",
)
replace(
    FRAME_BUFFER,
    "    pub fn future_world_states(&self) -> Vec<WorldState> {\n        self.frames.iter().map(|f| f.world_state.clone()).collect()\n    }\n",
    "    pub fn future_world_states(&self) -> Vec<WorldState> {\n"
    "        self.frames.iter().map(|f| f.world_state.clone()).collect()\n"
    "    }\n\n"
    "    /// Future states at the sparse analysis cadence. Skipped render-only\n"
    "    /// frames deliberately do not duplicate the last tracker state in the\n"
    "    /// panner's lookahead window.\n"
    "    pub fn future_analysis_world_states(&self) -> Vec<WorldState> {\n"
    "        self.frames\n"
    "            .iter()\n"
    "            .filter(|f| f.analysis_frame)\n"
    "            .map(|f| f.world_state.clone())\n"
    "            .collect()\n"
    "    }\n",
)
replace(
    FRAME_BUFFER,
    "        BufferedFrame {\n            frame: StereoFrame::Nv12",
    "        BufferedFrame {\n"
    "            frame: StereoFrame::Nv12",
)
# Insert metadata just after the Nv12 frame value closes, using the stable
# world_state marker as the insertion point.
replace(
    FRAME_BUFFER,
    "            }),\n            world_state: WorldState {\n",
    "            }),\n"
    "            source_frame_index: 0,\n"
    "            analysis_frame: true,\n"
    "            world_state: WorldState {\n",
)
replace(
    FRAME_BUFFER,
    "    #[test]\n    #[should_panic(expected = \"push called on full buffer\")]\n",
    "    #[test]\n"
    "    fn sparse_future_states_exclude_render_only_duplicates() {\n"
    "        let mut buf = FrameBuffer::new(4);\n"
    "        let mut a = make_frame(0.1);\n"
    "        a.analysis_frame = true;\n"
    "        let mut b = make_frame(0.1);\n"
    "        b.analysis_frame = false;\n"
    "        let mut c = make_frame(0.3);\n"
    "        c.analysis_frame = true;\n"
    "        buf.push(a);\n"
    "        buf.push(b);\n"
    "        buf.push(c);\n"
    "        let futures = buf.future_analysis_world_states();\n"
    "        assert_eq!(futures.len(), 2);\n"
    "        assert!((futures[0].ball.as_ref().unwrap().yaw - 0.1).abs() < 1e-6);\n"
    "        assert!((futures[1].ball.as_ref().unwrap().yaw - 0.3).abs() < 1e-6);\n"
    "    }\n\n"
    "    #[test]\n    #[should_panic(expected = \"push called on full buffer\")]\n",
)

# ---------------------------------------------------------------------------
# Detection/tracker cadence: source index is kept for staging, analysis index
# is what stateful detector interval/tracker logic sees.
# ---------------------------------------------------------------------------
DISPATCH = "crates/reco-core/src/session/detection_dispatch.rs"
replace(
    DISPATCH,
    "        elapsed: std::time::Duration,\n        produce_index: u64,\n    ) -> Result<crate::detect::tracker::WorldState, SessionError> {\n        let should_detect = self.detection.should_detect(produce_index);\n",
    "        elapsed: std::time::Duration,\n"
    "        source_index: u64,\n"
    "        analysis_index: u64,\n"
    "    ) -> Result<crate::detect::tracker::WorldState, SessionError> {\n"
    "        let should_detect = self.detection.should_detect(analysis_index);\n",
)
replace(
    DISPATCH,
    "                        let left_slot = (produce_index as usize * 2) % pool.n_slots();\n                        let right_slot = (produce_index as usize * 2 + 1) % pool.n_slots();\n",
    "                        let left_slot = (source_index as usize * 2) % pool.n_slots();\n"
    "                        let right_slot = (source_index as usize * 2 + 1) % pool.n_slots();\n",
)
replace(
    DISPATCH,
    "                frame_index: produce_index,\n                timestamp_ms,\n                caller: \"lookahead_produce\",\n            },\n        );\n\n        Ok(world)\n",
    "                frame_index: analysis_index,\n"
    "                timestamp_ms,\n"
    "                caller: \"lookahead_produce\",\n"
    "            },\n"
    "        );\n\n"
    "        self.last_world_state = world.clone();\n"
    "        Ok(world)\n",
)

# ---------------------------------------------------------------------------
# Full-rate render loop + sparse AI decisions + pose interpolation.
# ---------------------------------------------------------------------------
RUN_LOOP = "crates/reco-core/src/session/run_loop.rs"
replace(
    RUN_LOOP,
    "use crate::source::FrameSource;\n\n",
    "use crate::source::FrameSource;\n\n"
    "fn interpolate_pose(\n"
    "    from: crate::detect::director::ViewportPosition,\n"
    "    to: crate::detect::director::ViewportPosition,\n"
    "    t: f32,\n"
    ") -> crate::detect::director::ViewportPosition {\n"
    "    let t = t.clamp(0.0, 1.0);\n"
    "    let yaw_delta = (to.yaw - from.yaw + std::f32::consts::PI)\n"
    "        .rem_euclid(std::f32::consts::TAU)\n"
    "        - std::f32::consts::PI;\n"
    "    let fov_degrees = match (from.fov_degrees, to.fov_degrees) {\n"
    "        (Some(a), Some(b)) => Some(a + (b - a) * t),\n"
    "        (Some(a), None) => Some(a),\n"
    "        (None, Some(b)) => Some(b),\n"
    "        (None, None) => None,\n"
    "    };\n"
    "    crate::detect::director::ViewportPosition {\n"
    "        yaw: from.yaw + yaw_delta * t,\n"
    "        pitch: from.pitch + (to.pitch - from.pitch) * t,\n"
    "        fov_degrees,\n"
    "    }\n"
    "}\n\n"
    "fn queue_sparse_segment(\n"
    "    pose_queue: &mut std::collections::VecDeque<(\n"
    "        super::frame_buffer::BufferedFrame,\n"
    "        crate::detect::director::ViewportPosition,\n"
    "    )>,\n"
    "    anchor: (\n"
    "        super::frame_buffer::BufferedFrame,\n"
    "        crate::detect::director::ViewportPosition,\n"
    "    ),\n"
    "    between: &mut std::collections::VecDeque<super::frame_buffer::BufferedFrame>,\n"
    "    next_pose: Option<crate::detect::director::ViewportPosition>,\n"
    ") {\n"
    "    let (anchor_frame, anchor_pose) = anchor;\n"
    "    pose_queue.push_back((anchor_frame, anchor_pose));\n"
    "    let denominator = (between.len() + 1) as f32;\n"
    "    for (offset, frame) in between.drain(..).enumerate() {\n"
    "        let pose = next_pose.map_or(anchor_pose, |next| {\n"
    "            interpolate_pose(anchor_pose, next, (offset + 1) as f32 / denominator)\n"
    "        });\n"
    "        pose_queue.push_back((frame, pose));\n"
    "    }\n"
    "}\n\n",
)
replace(
    RUN_LOOP,
    "        self.configure_from_source(source);\n\n        let result = if self.lookahead_frames > 0 {\n",
    "        self.configure_from_source(source);\n\n"
    "        if self.frame_stride > 1 && self.lookahead_frames == 0 && self.panner.is_some() {\n"
    "            return Err(SessionError::Config(\n"
    "                \"--frame-stride > 1 requires lookahead so full-rate camera poses can be interpolated\"\n"
    "                    .to_string(),\n"
    "            ));\n"
    "        }\n"
    "        if self.frame_stride > 1 {\n"
    "            let fps = source.info().fps.max(1.0);\n"
    "            log::info!(\n"
    "                \"Frame stride: render {:.2} fps, analyze every {} frames ({:.2} decisions/s)\",\n"
    "                fps,\n"
    "                self.frame_stride,\n"
    "                fps / self.frame_stride as f64\n"
    "            );\n"
    "        }\n\n"
    "        let result = if self.lookahead_frames > 0 {\n",
)
# Replace produce_one body section from decode elapsed through buffer push.
sub(
    RUN_LOOP,
    r"            let decode_time = frame_t0\.elapsed\(\);\n            let elapsed = start\.elapsed\(\);\n\n            // Stage GPU frames to persistent slots BEFORE detection\.(.*?)            buffer\.push\(BufferedFrame \{\n                frame,\n                world_state,\n                detections,\n                elapsed_ms: elapsed\.as_secs_f64\(\) \* 1000\.0,\n                decode_time,\n                upload_time,\n                vram_slot,\n            \}\);",
    "            let decode_time = frame_t0.elapsed();\n"
    "            let wall_elapsed = start.elapsed();\n"
    "            let source_index = *produce_count;\n"
    "            let frame_stride = session.frame_stride.max(1);\n"
    "            let analysis_frame = source_index.is_multiple_of(frame_stride);\n"
    "            let analysis_elapsed = if frame_stride > 1 {\n"
    "                std::time::Duration::from_secs_f64(\n"
    "                    source_index as f64 / source.info().fps.max(1.0),\n"
    "                )\n"
    "            } else {\n"
    "                wall_elapsed\n"
    "            };\n\n"
    "            // Stage every source frame so rendering remains full-rate. Only\n"
    "            // the sparse analysis frames enter detector/tracker state.\n"
    "            let upload_t0 = std::time::Instant::now();\n"
    "            let vram_slot = session.copy_to_vram_pool(&frame, source_index)?;\n"
    "            let upload_time = if vram_slot.is_some() {\n"
    "                upload_t0.elapsed()\n"
    "            } else {\n"
    "                std::time::Duration::ZERO\n"
    "            };\n\n"
    "            let (world_state, detections) = if analysis_frame {\n"
    "                session.current_vram_slot = vram_slot;\n"
    "                let analysis_index = source_index / frame_stride;\n"
    "                let detection_result = session.detect_and_track_only(\n"
    "                    &frame,\n"
    "                    analysis_elapsed,\n"
    "                    source_index,\n"
    "                    analysis_index,\n"
    "                );\n"
    "                session.current_vram_slot = None;\n"
    "                let world_state = match detection_result {\n"
    "                    Ok(world_state) => world_state,\n"
    "                    Err(error) => {\n"
    "                        if let (Some(slot), Some(pool)) = (vram_slot, session.vram_pool.as_mut()) {\n"
    "                            pool.release(slot);\n"
    "                        }\n"
    "                        session.release_gpu_decode_slot(&frame);\n"
    "                        return Err(error);\n"
    "                    }\n"
    "                };\n"
    "                (world_state, session.detection.last_detections.clone())\n"
    "            } else {\n"
    "                (\n"
    "                    session.last_world_state.clone(),\n"
    "                    session.detection.last_detections.clone(),\n"
    "                )\n"
    "            };\n\n"
    "            // CUDA detection has now finished reading an analysis frame. A\n"
    "            // render-only frame never exposes the shared decode slot to AI.\n"
    "            session.release_gpu_decode_slot(&frame);\n\n"
    "            buffer.push(BufferedFrame {\n"
    "                frame,\n"
    "                source_frame_index: source_index,\n"
    "                analysis_frame,\n"
    "                world_state,\n"
    "                detections,\n"
    "                elapsed_ms: analysis_elapsed.as_secs_f64() * 1000.0,\n"
    "                decode_time,\n"
    "                upload_time,\n"
    "                vram_slot,\n"
    "            });",
)
# Replace pose queue + panner helper setup through the closure end.
sub(
    RUN_LOOP,
    r"        let mut pose_queue: std::collections::VecDeque<\(\n            BufferedFrame,\n            crate::detect::director::ViewportPosition,\n        \)> = std::collections::VecDeque::new\(\);\n        let mut past_poses: std::collections::VecDeque<crate::detect::director::ViewportPosition> =\n            std::collections::VecDeque::new\(\);\n        let mut panner_frame_idx: u64 = 0;\n\n        // Helper: run the panner on the oldest buffered frame, push\n        // the \(frame, pose\) pair into the pose queue\.\n        let run_panner_once = \|session: &mut StitchSession,(.*?)                true\n            \} else \{\n                false\n            \}\n        \};",
    "        let mut pose_queue: std::collections::VecDeque<(\n"
    "            BufferedFrame,\n"
    "            crate::detect::director::ViewportPosition,\n"
    "        )> = std::collections::VecDeque::new();\n"
    "        let mut past_poses: std::collections::VecDeque<crate::detect::director::ViewportPosition> =\n"
    "            std::collections::VecDeque::new();\n"
    "        let mut sparse_anchor: Option<(\n"
    "            BufferedFrame,\n"
    "            crate::detect::director::ViewportPosition,\n"
    "        )> = None;\n"
    "        let mut sparse_between: std::collections::VecDeque<BufferedFrame> =\n"
    "            std::collections::VecDeque::new();\n"
    "        let mut sparse_finalized = false;\n\n"
    "        // Run the panner only on analysis frames. Render-only frames are\n"
    "        // queued between adjacent analysis anchors and receive an\n"
    "        // interpolated pose once the next anchor is known.\n"
    "        let run_panner_once = |session: &mut StitchSession,\n"
    "                               buffer: &mut FrameBuffer,\n"
    "                               pose_queue: &mut std::collections::VecDeque<(\n"
    "            BufferedFrame,\n"
    "            crate::detect::director::ViewportPosition,\n"
    "        )>,\n"
    "                               sparse_anchor: &mut Option<(\n"
    "            BufferedFrame,\n"
    "            crate::detect::director::ViewportPosition,\n"
    "        )>,\n"
    "                               sparse_between: &mut std::collections::VecDeque<BufferedFrame>| {\n"
    "            if let Some(frame) = buffer.pop() {\n"
    "                if session.frame_stride <= 1 {\n"
    "                    session.lookahead_world_states = buffer.future_world_states();\n"
    "                    let pose = if let Some(panner) = session.panner.as_mut() {\n"
    "                        let pan_ctx = crate::detect::panner::PanContext {\n"
    "                            frame_index: frame.source_frame_index,\n"
    "                            timestamp_ms: frame.elapsed_ms,\n"
    "                            previous_position: session.previous_panner_pose,\n"
    "                            calibration: session.core.pipeline().calibration(),\n"
    "                        };\n"
    "                        let p = panner.decide_with_lookahead(\n"
    "                            &frame.world_state,\n"
    "                            &session.lookahead_world_states,\n"
    "                            &pan_ctx,\n"
    "                        );\n"
    "                        session.previous_panner_pose = p;\n"
    "                        p\n"
    "                    } else {\n"
    "                        session.previous_panner_pose\n"
    "                    };\n"
    "                    pose_queue.push_back((frame, pose));\n"
    "                } else if frame.analysis_frame {\n"
    "                    session.lookahead_world_states = buffer.future_analysis_world_states();\n"
    "                    let analysis_index = frame.source_frame_index / session.frame_stride;\n"
    "                    let pose = if let Some(panner) = session.panner.as_mut() {\n"
    "                        let pan_ctx = crate::detect::panner::PanContext {\n"
    "                            frame_index: analysis_index,\n"
    "                            timestamp_ms: frame.elapsed_ms,\n"
    "                            previous_position: session.previous_panner_pose,\n"
    "                            calibration: session.core.pipeline().calibration(),\n"
    "                        };\n"
    "                        let p = panner.decide_with_lookahead(\n"
    "                            &frame.world_state,\n"
    "                            &session.lookahead_world_states,\n"
    "                            &pan_ctx,\n"
    "                        );\n"
    "                        session.previous_panner_pose = p;\n"
    "                        p\n"
    "                    } else {\n"
    "                        session.previous_panner_pose\n"
    "                    };\n"
    "                    if let Some(anchor) = sparse_anchor.take() {\n"
    "                        queue_sparse_segment(pose_queue, anchor, sparse_between, Some(pose));\n"
    "                    }\n"
    "                    *sparse_anchor = Some((frame, pose));\n"
    "                } else {\n"
    "                    sparse_between.push_back(frame);\n"
    "                }\n"
    "                true\n"
    "            } else {\n"
    "                false\n"
    "            }\n"
    "        };",
)
# Update the two run_panner_once call sites.
replace(
    RUN_LOOP,
    "            run_panner_once(self, &mut buffer, &mut pose_queue, &mut panner_frame_idx);\n",
    "            run_panner_once(\n"
    "                self,\n"
    "                &mut buffer,\n"
    "                &mut pose_queue,\n"
    "                &mut sparse_anchor,\n"
    "                &mut sparse_between,\n"
    "            );\n",
)
replace(
    RUN_LOOP,
    "                run_panner_once(self, &mut buffer, &mut pose_queue, &mut panner_frame_idx);\n",
    "                run_panner_once(\n"
    "                    self,\n"
    "                    &mut buffer,\n"
    "                    &mut pose_queue,\n"
    "                    &mut sparse_anchor,\n"
    "                    &mut sparse_between,\n"
    "                );\n",
)
# Finalize the last sparse segment as soon as all source frames have entered
# the pose stage, before the existing render/drain conditions can see an empty
# pose_queue and terminate.
replace(
    RUN_LOOP,
    "            // Render: consume from pose queue when we have enough context\n",
    "            if eof && buffer.is_empty() && !sparse_finalized {\n"
    "                if self.frame_stride > 1 {\n"
    "                    if let Some(anchor) = sparse_anchor.take() {\n"
    "                        queue_sparse_segment(\n"
    "                            &mut pose_queue,\n"
    "                            anchor,\n"
    "                            &mut sparse_between,\n"
    "                            None,\n"
    "                        );\n"
    "                    }\n"
    "                }\n"
    "                sparse_finalized = true;\n"
    "            }\n\n"
    "            // Render: consume from pose queue when we have enough context\n",
)
# Unit tests for interpolation math.
write(
    RUN_LOOP,
    read(RUN_LOOP)
    + "\n#[cfg(test)]\nmod frame_stride_tests {\n"
    "    use super::*;\n\n"
    "    #[test]\n"
    "    fn sparse_pose_interpolation_hits_midpoint() {\n"
    "        let a = crate::detect::director::ViewportPosition {\n"
    "            yaw: 0.0,\n"
    "            pitch: 0.0,\n"
    "            fov_degrees: Some(30.0),\n"
    "        };\n"
    "        let b = crate::detect::director::ViewportPosition {\n"
    "            yaw: 0.3,\n"
    "            pitch: 0.12,\n"
    "            fov_degrees: Some(42.0),\n"
    "        };\n"
    "        let mid = interpolate_pose(a, b, 0.5);\n"
    "        assert!((mid.yaw - 0.15).abs() < 1e-6);\n"
    "        assert!((mid.pitch - 0.06).abs() < 1e-6);\n"
    "        assert!((mid.fov_degrees.unwrap() - 36.0).abs() < 1e-6);\n"
    "    }\n\n"
    "    #[test]\n"
    "    fn sparse_pose_interpolation_uses_short_yaw_path() {\n"
    "        let a = crate::detect::director::ViewportPosition {\n"
    "            yaw: 3.10,\n"
    "            pitch: 0.0,\n"
    "            fov_degrees: None,\n"
    "        };\n"
    "        let b = crate::detect::director::ViewportPosition {\n"
    "            yaw: -3.10,\n"
    "            pitch: 0.0,\n"
    "            fov_degrees: None,\n"
    "        };\n"
    "        let mid = interpolate_pose(a, b, 0.5);\n"
    "        assert!(mid.yaw.abs() > 3.0);\n"
    "    }\n"
    "}\n",
)

# ---------------------------------------------------------------------------
# CLI: public --frame-stride control, default 1 for backwards compatibility.
# OEV production orchestration will opt into 3 after validation.
# ---------------------------------------------------------------------------
CLI_MAIN = "crates/reco-cli/src/main.rs"
replace(
    CLI_MAIN,
    "        #[arg(long, default_value_t = 1)]\n        detection_interval: u64,\n\n        /// Lookahead buffer in seconds.",
    "        #[arg(long, default_value_t = 1)]\n"
    "        detection_interval: u64,\n\n"
    "        /// Analyze every Nth source frame while still rendering every frame.\n"
    "        /// Validated range: 1-4. Stride 3 is the OEV production candidate.\n"
    "        /// Values above 1 require lookahead so camera poses can be smoothly\n"
    "        /// interpolated between sparse AI decisions.\n"
    "        #[arg(long, default_value_t = 1)]\n"
    "        frame_stride: u64,\n\n"
    "        /// Lookahead buffer in seconds.",
)
replace(
    CLI_MAIN,
    "            detection_interval,\n            lookahead,\n",
    "            detection_interval,\n            frame_stride,\n            lookahead,\n",
)
replace(
    CLI_MAIN,
    "                detection_interval,\n                lookahead,\n",
    "                detection_interval,\n                frame_stride,\n                lookahead,\n",
)

STITCH = "crates/reco-cli/src/stitch.rs"
replace(
    STITCH,
    "    pub detection_interval: u64,\n    pub lookahead: f64,\n",
    "    pub detection_interval: u64,\n    pub frame_stride: u64,\n    pub lookahead: f64,\n",
)
replace(
    STITCH,
    "    anyhow::ensure!(\n        matches!(args.projection, \"l-shape\" | \"cylindrical-stereo\"),\n",
    "    anyhow::ensure!(\n"
    "        (1..=reco_autocam::MAX_FRAME_STRIDE).contains(&args.frame_stride),\n"
    "        \"--frame-stride must be between 1 and {}, got {}\",\n"
    "        reco_autocam::MAX_FRAME_STRIDE,\n"
    "        args.frame_stride,\n"
    "    );\n"
    "    if args.frame_stride > 1 {\n"
    "        anyhow::ensure!(\n"
    "            args.model_path.is_some() && args.tracking_mode != \"sweep\",\n"
    "            \"--frame-stride > 1 requires AI tracking (--model, non-sweep)\"\n"
    "        );\n"
    "        anyhow::ensure!(\n"
    "            args.lookahead > 0.0,\n"
    "            \"--frame-stride > 1 requires --lookahead > 0 for full-rate pose interpolation\"\n"
    "        );\n"
    "    }\n"
    "    anyhow::ensure!(\n        matches!(args.projection, \"l-shape\" | \"cylindrical-stereo\"),\n",
)
replace(
    STITCH,
    "        let interval = args.detection_interval;\n        let mode_str = args.tracking_mode.to_owned();\n",
    "        let interval = args.detection_interval;\n"
    "        let frame_stride = args.frame_stride;\n"
    "        let mode_str = args.tracking_mode.to_owned();\n",
)
replace(
    STITCH,
    "            let mut autocam_config = reco_autocam::AutocamConfig::new(&model_path)\n                .with_tracking_mode(mode)\n                .with_detection_interval(interval)\n                .with_10bit(is_10bit);\n",
    "            session.set_frame_stride(frame_stride);\n"
    "            let mut autocam_config = reco_autocam::AutocamConfig::new(&model_path)\n"
    "                .with_tracking_mode(mode)\n"
    "                .with_detection_interval(interval)\n"
    "                .with_frame_stride(frame_stride)\n"
    "                .with_10bit(is_10bit);\n",
)
replace(
    STITCH,
    "                Ok(true) => println!(\"Autocam: tracking enabled (model: {model_path})\"),\n",
    "                Ok(true) => {\n"
    "                    println!(\"Autocam: tracking enabled (model: {model_path})\");\n"
    "                    if frame_stride > 1 {\n"
    "                        println!(\n"
    "                            \"Frame stride: analyze 1/{frame_stride}, render every source frame\"\n"
    "                        );\n"
    "                    }\n"
    "                }\n",
)

print("production frame-stride hardening patch applied")
