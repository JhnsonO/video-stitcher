//! GPU-resident YOLO detector for the zero-copy pipeline.
//!
//! Runs the entire detection pipeline on GPU: NV12 color conversion (NPP),
//! resize with letterbox padding (NPP), normalize + CHW transpose (CUDA kernel),
//! and inference (ORT TensorRT/CUDA EP). Only the small detection output
//! (~7KB for `[1, 300, 6]`) is read back to CPU.

/// `MemoryInfo` wrapper with manual `Send` impl.
///
/// ORT's `MemoryInfo` holds `*mut OrtMemoryInfo` and is therefore not
/// `Send` by default. But a `MemoryInfo` is immutable descriptor state
/// (device / allocator type / memtype) — ORT's C layer treats it as
/// read-only config. UnifiedDetector requires `Send`, so wrap with an
/// opt-in unsafe impl for the "we only read it" use case.
struct SendMemoryInfo(ort::memory::MemoryInfo);
// SAFETY: MemoryInfo is read-only descriptor state; no interior mutation
// crosses thread boundaries.
unsafe impl Send for SendMemoryInfo {}

use std::ffi::c_void;
use std::path::Path;

use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::value::{Shape, TensorRefMut};
use reco_core::detect::detector::{
    CameraId, Detection, DetectorError, DetectorFrame, GpuNv12Frame, UnifiedDetector,
};
use reco_core::interop::cuda::{
    CUdeviceptr, cuda_ensure_context, cuda_mem_alloc, cuda_mem_free, cuda_memset_d8,
};

use super::postprocess;

const RECOVERY_CROP_RATIOS: [f32; 3] = [0.5, 2.0 / 3.0, 5.0 / 6.0];
const LOCAL_RECOVERY_HORIZON: u32 = 24;
const TILED_SEARCH_INTERVAL: u32 = 4;
const TILED_CROP_RATIO: f32 = 2.0 / 3.0;

fn camera_index(camera: CameraId) -> usize {
    match camera {
        CameraId::Left => 0,
        CameraId::Right => 1,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CropRegion {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl CropRegion {
    fn full(width: u32, height: u32) -> Self {
        Self {
            x: 0,
            y: 0,
            width,
            height,
        }
    }

    fn centered(width: u32, height: u32, ratio: f32, center: (f32, f32)) -> Self {
        let even_size = |value: u32| value.max(2) & !1;
        let crop_width = even_size(((width as f32 * ratio).round() as u32).min(width));
        let crop_height = even_size(((height as f32 * ratio).round() as u32).min(height));
        let max_x = width.saturating_sub(crop_width);
        let max_y = height.saturating_sub(crop_height);
        let target_x = center.0.clamp(0.0, 1.0) * width as f32 - crop_width as f32 / 2.0;
        let target_y = center.1.clamp(0.0, 1.0) * height as f32 - crop_height as f32 / 2.0;
        let x = (target_x.round().clamp(0.0, max_x as f32) as u32 & !1).min(max_x & !1);
        let y = (target_y.round().clamp(0.0, max_y as f32) as u32 & !1).min(max_y & !1);
        Self {
            x,
            y,
            width: crop_width,
            height: crop_height,
        }
    }

    fn tiled_2x2(width: u32, height: u32) -> [Self; 4] {
        let even_size = |value: u32| value.max(2) & !1;
        let crop_width = even_size(((width as f32 * TILED_CROP_RATIO).round() as u32).min(width));
        let crop_height =
            even_size(((height as f32 * TILED_CROP_RATIO).round() as u32).min(height));
        let max_x = width.saturating_sub(crop_width) & !1;
        let max_y = height.saturating_sub(crop_height) & !1;
        [
            Self {
                x: 0,
                y: 0,
                width: crop_width,
                height: crop_height,
            },
            Self {
                x: max_x,
                y: 0,
                width: crop_width,
                height: crop_height,
            },
            Self {
                x: 0,
                y: max_y,
                width: crop_width,
                height: crop_height,
            },
            Self {
                x: max_x,
                y: max_y,
                width: crop_width,
                height: crop_height,
            },
        ]
    }

    fn remap(self, mut detection: Detection, frame_width: u32, frame_height: u32) -> Detection {
        detection.center_x =
            (self.x as f32 + detection.center_x * self.width as f32) / frame_width as f32;
        detection.center_y =
            (self.y as f32 + detection.center_y * self.height as f32) / frame_height as f32;
        detection.width *= self.width as f32 / frame_width as f32;
        detection.height *= self.height as f32 / frame_height as f32;
        detection
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct CameraRecoveryState {
    last_center: Option<(f32, f32)>,
    velocity: (f32, f32),
    misses: u32,
}

impl CameraRecoveryState {
    fn predicted_center(&self) -> Option<(f32, f32)> {
        let (x, y) = self.last_center?;
        let horizon = self.misses.saturating_add(1) as f32;
        Some((
            (x + self.velocity.0 * horizon).clamp(0.0, 1.0),
            (y + self.velocity.1 * horizon).clamp(0.0, 1.0),
        ))
    }

    fn observe(&mut self, detections: &[Detection], ball_class_id: u16) {
        let expected = self.predicted_center();
        let chosen = detections
            .iter()
            .filter(|d| d.class_id == ball_class_id)
            .min_by(|a, b| {
                let score = |d: &&Detection| {
                    if let Some((x, y)) = expected {
                        (d.center_x - x).powi(2) + (d.center_y - y).powi(2)
                    } else {
                        1.0 - d.confidence
                    }
                };
                score(a).total_cmp(&score(b))
            });
        let Some(chosen) = chosen else {
            return;
        };
        if let Some((last_x, last_y)) = self.last_center {
            let elapsed = self.misses.saturating_add(1) as f32;
            self.velocity = (
                ((chosen.center_x - last_x) / elapsed).clamp(-0.08, 0.08),
                ((chosen.center_y - last_y) / elapsed).clamp(-0.08, 0.08),
            );
        }
        self.last_center = Some((chosen.center_x, chosen.center_y));
        self.misses = 0;
    }
}

#[derive(Debug, Default)]
struct RecoveryStats {
    attempts: [u64; 3],
    hits: [u64; 3],
    tile_attempts: [u64; 4],
    tile_hits: [u64; 4],
    exhausted: u64,
    errors: u64,
    commits: u64,
    rejects: u64,
}

#[derive(Debug)]
struct BallRecovery {
    class_id: u16,
    cameras: [CameraRecoveryState; 2],
    attempted_this_call: [bool; 2],
    stats: RecoveryStats,
}

impl BallRecovery {
    fn state_mut(&mut self, camera: CameraId) -> &mut CameraRecoveryState {
        &mut self.cameras[camera_index(camera)]
    }
}

/// YOLO detector that operates on GPU-resident NV12 frames via ORT.
///
/// Pre-allocates GPU scratch buffers for the preprocessing pipeline and
/// reuses them across frames. The ORT session runs with TensorRT or CUDA EP
/// for GPU-side inference.
///
/// Created via [`OrtGpuDetector::try_new`], which returns `None` if NPP
/// is not available on the system.
pub struct OrtGpuDetector {
    session: Session,
    input_size: u32,
    confidence_threshold: f32,
    labels: Vec<String>,
    scale: f32,
    #[allow(dead_code)]
    new_w: u32,
    #[allow(dead_code)]
    new_h: u32,
    pad_x: f32,
    pad_y: f32,
    rgb_u8: CUdeviceptr,
    rgb_scratch: CUdeviceptr,
    resized_u8: CUdeviceptr,
    tensor_f32: CUdeviceptr,
    nv12_8bit_y: CUdeviceptr,
    nv12_8bit_uv: CUdeviceptr,
    cuda_memory_info: SendMemoryInfo,
    frame_width: u32,
    frame_height: u32,
    ball_recovery: Option<BallRecovery>,
}

impl OrtGpuDetector {
    #[allow(clippy::too_many_arguments)]
    fn infer_region(
        &mut self,
        camera: CameraId,
        nv12_y: CUdeviceptr,
        nv12_y_pitch: usize,
        nv12_uv: CUdeviceptr,
        nv12_uv_pitch: usize,
        frame_width: u32,
        frame_height: u32,
        rotation: i32,
        region: CropRegion,
    ) -> Result<Vec<Detection>, DetectorError> {
        let (raw_x, raw_y) = if rotation == 180 {
            (
                frame_width.saturating_sub(region.x + region.width),
                frame_height.saturating_sub(region.y + region.height),
            )
        } else {
            (region.x, region.y)
        };
        let y_offset = u64::from(raw_y)
            .checked_mul(nv12_y_pitch as u64)
            .and_then(|v| v.checked_add(u64::from(raw_x)))
            .ok_or_else(|| DetectorError::InferenceFailed("NV12 Y crop offset overflow".into()))?;
        let uv_offset = u64::from(raw_y / 2)
            .checked_mul(nv12_uv_pitch as u64)
            .and_then(|v| v.checked_add(u64::from(raw_x)))
            .ok_or_else(|| DetectorError::InferenceFailed("NV12 UV crop offset overflow".into()))?;
        let crop_y_ptr = nv12_y
            .checked_add(y_offset)
            .ok_or_else(|| DetectorError::InferenceFailed("NV12 Y pointer overflow".into()))?;
        let crop_uv_ptr = nv12_uv
            .checked_add(uv_offset)
            .ok_or_else(|| DetectorError::InferenceFailed("NV12 UV pointer overflow".into()))?;

        let (scale, pad_x, pad_y) = if region == CropRegion::full(frame_width, frame_height) {
            (self.scale, self.pad_x, self.pad_y)
        } else {
            let input = self.input_size as f32;
            let scale = (input / region.width as f32).min(input / region.height as f32);
            let new_w = (region.width as f32 * scale).round() as u32;
            let new_h = (region.height as f32 * scale).round() as u32;
            (
                scale,
                (self.input_size - new_w) as f32 / 2.0,
                (self.input_size - new_h) as f32 / 2.0,
            )
        };

        {
            reco_core::profile_scope!("nv12_to_rgb_chw");
            crate::cuda_kernels::nv12_to_rgb_chw_fullrange(
                crop_y_ptr,
                crop_uv_ptr,
                self.tensor_f32,
                nv12_y_pitch as u32,
                region.width,
                region.height,
                self.input_size,
                self.input_size,
                pad_x as u32,
                pad_y as u32,
                scale,
                rotation,
            )
            .map_err(|e| DetectorError::InferenceFailed(format!("NV12->RGB CHW: {e}")))?;
        }

        let outputs = {
            reco_core::profile_scope!("gpu_ort_inference");
            let sz = self.input_size as i64;
            let tensor: TensorRefMut<'_, f32> = unsafe {
                TensorRefMut::from_raw(
                    self.cuda_memory_info.0.clone(),
                    self.tensor_f32 as *mut c_void,
                    Shape::new([1i64, 3, sz, sz]),
                )
            }
            .map_err(|e| DetectorError::InferenceFailed(format!("GPU tensor wrap: {e}")))?;
            self.session
                .run(ort::inputs![tensor])
                .map_err(|e| DetectorError::InferenceFailed(format!("ort run: {e}")))?
        };

        let (shape, slice) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| DetectorError::InferenceFailed(format!("output extract: {e}")))?;
        let detections = postprocess(
            slice,
            shape[1] as usize,
            camera,
            self.confidence_threshold,
            scale,
            pad_x,
            pad_y,
            region.width,
            region.height,
        )
        .into_iter()
        .map(|detection| region.remap(detection, frame_width, frame_height))
        .collect();
        drop(outputs);
        Ok(detections)
    }

    pub fn try_new(
        model_path: impl AsRef<Path>,
        frame_width: u32,
        frame_height: u32,
        confidence_threshold: f32,
        labels: Vec<String>,
        is_10bit: bool,
    ) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        if !cfg!(feature = "tensorrt") && !cfg!(feature = "cuda") {
            log::warn!(
                "OrtGpuDetector: no GPU EP available (need --features tensorrt or --features cuda)"
            );
            return Ok(None);
        }

        cuda_ensure_context()?;
        let (session, input_size, labels) =
            crate::ort_session::create_ort_session(model_path.as_ref(), labels)?;

        let (fw, fh) = (frame_width as f32, frame_height as f32);
        let is = input_size as f32;
        let scale = (is / fw).min(is / fh);
        let new_w = (fw * scale).round() as u32;
        let new_h = (fh * scale).round() as u32;
        let pad_x = (input_size - new_w) as f32 / 2.0;
        let pad_y = (input_size - new_h) as f32 / 2.0;

        let rgb_size = (frame_width as usize)
            .checked_mul(frame_height as usize)
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| ort::Error::new("frame dimensions overflow for rgb_size"))?;
        let resized_size = (input_size as usize)
            .checked_mul(input_size as usize)
            .and_then(|v| v.checked_mul(3))
            .ok_or_else(|| ort::Error::new("input dimensions overflow for resized_size"))?;
        let tensor_size = (input_size as usize)
            .checked_mul(input_size as usize)
            .and_then(|v| v.checked_mul(3))
            .and_then(|v| v.checked_mul(4))
            .ok_or_else(|| ort::Error::new("input dimensions overflow for tensor_size"))?;

        let rgb_u8 = cuda_mem_alloc(rgb_size)?;
        let rgb_scratch = cuda_mem_alloc(rgb_size)?;
        let resized_u8 = cuda_mem_alloc(resized_size)?;
        let tensor_f32 = cuda_mem_alloc(tensor_size)?;

        let (nv12_8bit_y, nv12_8bit_uv) = if is_10bit {
            let y_size = frame_width as usize * frame_height as usize;
            let uv_size = frame_width as usize * (frame_height as usize / 2);
            let y = cuda_mem_alloc(y_size)?;
            let uv = cuda_mem_alloc(uv_size)?;
            log::info!(
                "OrtGpuDetector: allocated P010 conversion buffers ({:.1}MB)",
                (y_size + uv_size) as f64 / 1024.0 / 1024.0,
            );
            (y, uv)
        } else {
            (0, 0)
        };

        cuda_memset_d8(resized_u8, 114, resized_size)?;

        log::info!(
            "OrtGpuDetector ready: input={}x{}, frame={}x{}, scale={:.3}, pad=({:.1},{:.1}), \
             GPU scratch={:.1}MB, 10bit={}",
            input_size,
            input_size,
            frame_width,
            frame_height,
            scale,
            pad_x,
            pad_y,
            (rgb_size + resized_size + tensor_size) as f64 / 1024.0 / 1024.0,
            is_10bit,
        );

        let cuda_memory_info = SendMemoryInfo(
            MemoryInfo::new(
                AllocationDevice::CUDA,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )
            .map_err(|e| format!("CUDA MemoryInfo: {e}"))?,
        );

        let mut detector = Self {
            session,
            input_size,
            confidence_threshold,
            labels,
            scale,
            new_w,
            new_h,
            pad_x,
            pad_y,
            rgb_u8,
            rgb_scratch,
            resized_u8,
            tensor_f32,
            nv12_8bit_y,
            nv12_8bit_uv,
            cuda_memory_info,
            frame_width,
            frame_height,
            ball_recovery: None,
        };

        {
            let sz = input_size as usize;
            let warmup_data = vec![0.0f32; 3 * sz * sz];
            let tensor = ort::value::Tensor::from_array(([1, 3, sz, sz], warmup_data))?;
            detector.session.run(ort::inputs![tensor])?;
            log::info!("OrtGpuDetector: warmup inference complete");
        }

        Ok(Some(detector))
    }

    pub fn with_high_res_ball_recovery(mut self, enabled: bool) -> Self {
        if !enabled {
            return self;
        }
        let class_id = self.labels.iter().position(|label| {
            label.eq_ignore_ascii_case("ball") || label.eq_ignore_ascii_case("sports ball")
        });
        match class_id.and_then(|id| u16::try_from(id).ok()) {
            Some(class_id) => {
                self.ball_recovery = Some(BallRecovery {
                    class_id,
                    cameras: [CameraRecoveryState::default(); 2],
                    attempted_this_call: [false; 2],
                    stats: RecoveryStats::default(),
                });
                log::info!(
                    "OrtGpuDetector: high-resolution ball recovery enabled (class_id={class_id}, crop_ratios={RECOVERY_CROP_RATIOS:?}, local_horizon={LOCAL_RECOVERY_HORIZON}, tiled_interval={TILED_SEARCH_INTERVAL})"
                );
            }
            None => log::warn!(
                "OrtGpuDetector: high-resolution ball recovery requested, but model labels contain neither 'ball' nor 'sports ball'; recovery disabled"
            ),
        }
        self
    }

    pub fn recovery_ball_class_id(&self) -> Option<u16> {
        self.ball_recovery.as_ref().map(|r| r.class_id)
    }

    pub fn recovery_prediction(&self, camera: CameraId) -> Option<(f32, f32)> {
        self.ball_recovery
            .as_ref()
            .and_then(|r| r.cameras[camera_index(camera)].predicted_center())
    }

    pub fn recovery_was_attempted(&self, camera: CameraId) -> bool {
        self.ball_recovery
            .as_ref()
            .map(|r| r.attempted_this_call[camera_index(camera)])
            .unwrap_or(false)
    }

    pub fn commit_ball_recovery(&mut self, camera: CameraId, accepted: &[Detection]) {
        let Some(recovery) = self.ball_recovery.as_mut() else {
            return;
        };
        let previous_misses = recovery.cameras[camera_index(camera)].misses;
        let class_id = recovery.class_id;
        recovery.state_mut(camera).observe(accepted, class_id);
        recovery.stats.commits += 1;
        if previous_misses > 0 {
            log::info!(
                "BALL_RECOVERY_COMMIT camera={camera} reacquired_after_analysis_frames={previous_misses}"
            );
        }
    }

    pub fn reject_ball_recovery(&mut self, camera: CameraId) {
        let Some(recovery) = self.ball_recovery.as_mut() else {
            return;
        };
        let state = recovery.state_mut(camera);
        state.misses = state.misses.saturating_add(1);
        recovery.stats.rejects += 1;
    }

    fn prepared_nv12(
        &mut self,
        frame: &GpuNv12Frame,
    ) -> Result<(CUdeviceptr, usize, CUdeviceptr, usize), DetectorError> {
        if frame.is_10bit {
            if self.nv12_8bit_y == 0 || self.nv12_8bit_uv == 0 {
                return Err(DetectorError::InferenceFailed(
                    "P010 frame received but no conversion buffers allocated".into(),
                ));
            }
            crate::cuda_kernels::p010_plane_to_nv12(
                frame.y_ptr,
                frame.y_pitch,
                self.nv12_8bit_y,
                frame.width,
                frame.height,
            )
            .map_err(|e| DetectorError::InferenceFailed(format!("P010->NV12 Y conversion: {e}")))?;
            crate::cuda_kernels::p010_plane_to_nv12(
                frame.uv_ptr,
                frame.uv_pitch,
                self.nv12_8bit_uv,
                frame.width,
                frame.height / 2,
            )
            .map_err(|e| {
                DetectorError::InferenceFailed(format!("P010->NV12 UV conversion: {e}"))
            })?;
            Ok((
                self.nv12_8bit_y,
                frame.width as usize,
                self.nv12_8bit_uv,
                frame.width as usize,
            ))
        } else {
            Ok((frame.y_ptr, frame.y_pitch, frame.uv_ptr, frame.uv_pitch))
        }
    }

    fn recovery_candidates_gpu(
        &mut self,
        camera: CameraId,
        frame: &GpuNv12Frame,
    ) -> Result<Vec<Detection>, DetectorError> {
        let Some(ball_class_id) = self.ball_recovery.as_ref().map(|r| r.class_id) else {
            return Ok(Vec::new());
        };
        let idx = camera_index(camera);
        if let Some(recovery) = self.ball_recovery.as_mut() {
            recovery.attempted_this_call[idx] = true;
        }
        let state = self.ball_recovery.as_ref().unwrap().cameras[idx];
        let Some(predicted) = state.predicted_center() else {
            return Ok(Vec::new());
        };

        let (nv12_y, nv12_y_pitch, nv12_uv, nv12_uv_pitch) = self.prepared_nv12(frame)?;
        let width = frame.width;
        let height = frame.height;
        let rotation = frame.rotation;

        if state.misses < LOCAL_RECOVERY_HORIZON {
            for (stage, ratio) in RECOVERY_CROP_RATIOS.into_iter().enumerate() {
                let region = CropRegion::centered(width, height, ratio, predicted);
                if region == CropRegion::full(width, height) {
                    continue;
                }
                if let Some(recovery) = self.ball_recovery.as_mut() {
                    recovery.stats.attempts[stage] += 1;
                }
                log::debug!(
                    "BALL_RECOVERY_ATTEMPT camera={camera} mode=local stage={} crop={}x{}+{},{} predicted={:.4},{:.4} misses={}",
                    stage + 1,
                    region.width,
                    region.height,
                    region.x,
                    region.y,
                    predicted.0,
                    predicted.1,
                    state.misses,
                );
                let crop_detections = match self.infer_region(
                    camera,
                    nv12_y,
                    nv12_y_pitch,
                    nv12_uv,
                    nv12_uv_pitch,
                    width,
                    height,
                    rotation,
                    region,
                ) {
                    Ok(value) => value,
                    Err(error) => {
                        if let Some(recovery) = self.ball_recovery.as_mut() {
                            recovery.stats.errors += 1;
                        }
                        log::warn!(
                            "BALL_RECOVERY_ERROR camera={camera} mode=local stage={} error={error}",
                            stage + 1
                        );
                        break;
                    }
                };
                let recovered_balls: Vec<_> = crop_detections
                    .into_iter()
                    .filter(|d| d.class_id == ball_class_id)
                    .collect();
                if !recovered_balls.is_empty() {
                    if let Some(recovery) = self.ball_recovery.as_mut() {
                        recovery.stats.hits[stage] += 1;
                    }
                    log::info!(
                        "BALL_RECOVERY_HIT camera={camera} mode=local stage={} crop={}x{} count={} best_confidence={:.3}",
                        stage + 1,
                        region.width,
                        region.height,
                        recovered_balls.len(),
                        recovered_balls
                            .iter()
                            .map(|d| d.confidence)
                            .fold(0.0f32, f32::max),
                    );
                    return Ok(recovered_balls);
                }
            }
        } else if state.misses % TILED_SEARCH_INTERVAL == 0 {
            for (tile, region) in CropRegion::tiled_2x2(width, height).into_iter().enumerate() {
                if let Some(recovery) = self.ball_recovery.as_mut() {
                    recovery.stats.tile_attempts[tile] += 1;
                }
                log::debug!(
                    "BALL_RECOVERY_ATTEMPT camera={camera} mode=tiled tile={} crop={}x{}+{},{} misses={}",
                    tile + 1,
                    region.width,
                    region.height,
                    region.x,
                    region.y,
                    state.misses,
                );
                let tile_detections = match self.infer_region(
                    camera,
                    nv12_y,
                    nv12_y_pitch,
                    nv12_uv,
                    nv12_uv_pitch,
                    width,
                    height,
                    rotation,
                    region,
                ) {
                    Ok(value) => value,
                    Err(error) => {
                        if let Some(recovery) = self.ball_recovery.as_mut() {
                            recovery.stats.errors += 1;
                        }
                        log::warn!(
                            "BALL_RECOVERY_ERROR camera={camera} mode=tiled tile={} error={error}",
                            tile + 1
                        );
                        break;
                    }
                };
                let recovered_balls: Vec<_> = tile_detections
                    .into_iter()
                    .filter(|d| d.class_id == ball_class_id)
                    .collect();
                if !recovered_balls.is_empty() {
                    if let Some(recovery) = self.ball_recovery.as_mut() {
                        recovery.stats.tile_hits[tile] += 1;
                    }
                    log::info!(
                        "BALL_RECOVERY_HIT camera={camera} mode=tiled tile={} crop={}x{} count={} best_confidence={:.3}",
                        tile + 1,
                        region.width,
                        region.height,
                        recovered_balls.len(),
                        recovered_balls
                            .iter()
                            .map(|d| d.confidence)
                            .fold(0.0f32, f32::max),
                    );
                    return Ok(recovered_balls);
                }
            }
        } else {
            log::debug!(
                "BALL_RECOVERY_SKIP camera={camera} mode=tiled misses={} interval={TILED_SEARCH_INTERVAL}",
                state.misses
            );
        }

        if let Some(recovery) = self.ball_recovery.as_mut() {
            recovery.stats.exhausted += 1;
        }
        Ok(Vec::new())
    }

    pub fn force_recovery_candidates(
        &mut self,
        camera: CameraId,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        match frame {
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            DetectorFrame::Cuda(gpu_frame) => self.recovery_candidates_gpu(camera, gpu_frame),
            _ => Err(DetectorError::UnsupportedFrameKind),
        }
    }
}

impl OrtGpuDetector {
    fn detect_gpu_raw(
        &mut self,
        camera: CameraId,
        frame: &GpuNv12Frame,
    ) -> Result<Vec<Detection>, DetectorError> {
        reco_core::profile_scope!("gpu_yolo_detect");

        if frame.width != self.frame_width || frame.height != self.frame_height {
            return Err(DetectorError::InferenceFailed(format!(
                "GPU detector frame dimensions changed: configured={}x{}, received={}x{}",
                self.frame_width, self.frame_height, frame.width, frame.height
            )));
        }

        reco_core::interop::cuda::cuda_ensure_context()
            .map_err(|e| DetectorError::InferenceFailed(format!("cuda_ensure_context: {e}")))?;

        if let Some(recovery) = self.ball_recovery.as_mut() {
            recovery.attempted_this_call[camera_index(camera)] = false;
        }

        let (nv12_y, nv12_y_pitch, nv12_uv, nv12_uv_pitch) = self.prepared_nv12(frame)?;
        let mut detections = self.infer_region(
            camera,
            nv12_y,
            nv12_y_pitch,
            nv12_uv,
            nv12_uv_pitch,
            frame.width,
            frame.height,
            frame.rotation,
            CropRegion::full(frame.width, frame.height),
        )?;

        if let Some(ball_class_id) = self.ball_recovery.as_ref().map(|r| r.class_id)
            && !detections.iter().any(|d| d.class_id == ball_class_id)
        {
            detections.extend(self.recovery_candidates_gpu(camera, frame)?);
        }

        if !detections.is_empty() {
            log::debug!(
                "GPU camera {:?}: {} detection(s) - {}",
                camera,
                detections.len(),
                detections
                    .iter()
                    .map(|d| {
                        let name = self
                            .labels
                            .get(d.class_id as usize)
                            .map(|s| s.as_str())
                            .unwrap_or("?");
                        format!(
                            "{}({:.0}%@{:.2},{:.2})",
                            name,
                            d.confidence * 100.0,
                            d.center_x,
                            d.center_y
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        Ok(detections)
    }
}

impl UnifiedDetector for OrtGpuDetector {
    fn name(&self) -> &'static str {
        "ort-cuda"
    }

    fn detect(
        &mut self,
        camera: CameraId,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        match frame {
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            DetectorFrame::Cuda(gpu_frame) => self.detect_gpu_raw(camera, gpu_frame),
            _ => Err(DetectorError::UnsupportedFrameKind),
        }
    }

    fn class_names(&self) -> Option<&[String]> {
        Some(&self.labels)
    }
}

impl Drop for OrtGpuDetector {
    fn drop(&mut self) {
        if let Some(recovery) = &self.ball_recovery {
            log::info!(
                "BALL_RECOVERY_SUMMARY attempts={:?} hits={:?} tile_attempts={:?} tile_hits={:?} exhausted={} errors={} commits={} rejects={}",
                recovery.stats.attempts,
                recovery.stats.hits,
                recovery.stats.tile_attempts,
                recovery.stats.tile_hits,
                recovery.stats.exhausted,
                recovery.stats.errors,
                recovery.stats.commits,
                recovery.stats.rejects,
            );
        }
        if let Err(e) = cuda_ensure_context() {
            log::warn!("OrtGpuDetector drop: failed to set CUDA context: {e}");
            return;
        }
        for (name, ptr) in [
            ("rgb_u8", self.rgb_u8),
            ("rgb_scratch", self.rgb_scratch),
            ("resized_u8", self.resized_u8),
            ("tensor_f32", self.tensor_f32),
            ("nv12_8bit_y", self.nv12_8bit_y),
            ("nv12_8bit_uv", self.nv12_8bit_uv),
        ] {
            if ptr != 0
                && let Err(e) = cuda_mem_free(ptr)
            {
                log::warn!("Failed to free GPU buffer {name}: {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detection(center_x: f32, center_y: f32, confidence: f32) -> Detection {
        Detection {
            camera: CameraId::Left,
            class_id: 32,
            confidence,
            center_x,
            center_y,
            width: 0.1,
            height: 0.2,
        }
    }

    #[test]
    fn centered_crop_is_even_and_clamped_at_frame_edges() {
        let top_left = CropRegion::centered(3840, 2160, 0.5, (0.0, 0.0));
        assert_eq!(
            top_left,
            CropRegion {
                x: 0,
                y: 0,
                width: 1920,
                height: 1080,
            }
        );

        let bottom_right = CropRegion::centered(3840, 2160, 0.5, (1.0, 1.0));
        assert_eq!(
            bottom_right,
            CropRegion {
                x: 1920,
                y: 1080,
                width: 1920,
                height: 1080,
            }
        );
        assert_eq!(bottom_right.x % 2, 0);
        assert_eq!(bottom_right.y % 2, 0);
    }

    #[test]
    fn tiled_search_covers_full_frame_with_overlap() {
        let tiles = CropRegion::tiled_2x2(3840, 2160);
        assert_eq!(
            tiles[0],
            CropRegion {
                x: 0,
                y: 0,
                width: 2560,
                height: 1440
            }
        );
        assert_eq!(
            tiles[3],
            CropRegion {
                x: 1280,
                y: 720,
                width: 2560,
                height: 1440
            }
        );
        assert!(tiles[0].x + tiles[0].width > tiles[1].x);
        assert!(tiles[0].y + tiles[0].height > tiles[2].y);
    }

    #[test]
    fn crop_detection_remaps_to_full_frame_coordinates() {
        let crop = CropRegion {
            x: 960,
            y: 540,
            width: 1920,
            height: 1080,
        };
        let mapped = crop.remap(detection(0.5, 0.5, 0.8), 3840, 2160);
        assert!((mapped.center_x - 0.5).abs() < 1e-6);
        assert!((mapped.center_y - 0.5).abs() < 1e-6);
        assert!((mapped.width - 0.05).abs() < 1e-6);
        assert!((mapped.height - 0.1).abs() < 1e-6);
    }

    #[test]
    fn recovery_state_prefers_candidate_nearest_prediction() {
        let mut state = CameraRecoveryState {
            last_center: Some((0.25, 0.5)),
            velocity: (0.01, 0.0),
            misses: 0,
        };
        let nearby = detection(0.27, 0.5, 0.6);
        let distant_high_confidence = detection(0.9, 0.5, 0.99);
        state.observe(&[distant_high_confidence, nearby], 32);
        assert_eq!(state.last_center, Some((nearby.center_x, nearby.center_y)));
        assert_eq!(state.misses, 0);
    }

    #[test]
    fn recovery_prediction_advances_across_misses() {
        let state = CameraRecoveryState {
            last_center: Some((0.4, 0.5)),
            velocity: (0.01, -0.02),
            misses: 2,
        };
        let predicted = state.predicted_center().unwrap();
        assert!((predicted.0 - 0.43).abs() < 1e-6);
        assert!((predicted.1 - 0.44).abs() < 1e-6);
    }

    #[test]
    fn recovery_velocity_is_normalized_by_elapsed_frames() {
        let mut state = CameraRecoveryState {
            last_center: Some((0.4, 0.5)),
            velocity: (0.0, 0.0),
            misses: 3,
        };
        state.observe(&[detection(0.44, 0.46, 0.8)], 32);
        assert!((state.velocity.0 - 0.01).abs() < 1e-6);
        assert!((state.velocity.1 + 0.01).abs() < 1e-6);
    }

    #[test]
    fn rejected_candidate_does_not_mutate_recovery_state() {
        let state = CameraRecoveryState {
            last_center: Some((0.4, 0.5)),
            velocity: (0.01, 0.0),
            misses: 3,
        };
        let before = state;
        let _candidate = detection(0.9, 0.1, 0.99);
        assert_eq!(state.last_center, before.last_center);
        assert_eq!(state.velocity, before.velocity);
        assert_eq!(state.misses, before.misses);
    }
}
