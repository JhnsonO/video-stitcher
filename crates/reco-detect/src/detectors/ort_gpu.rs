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
const MAX_RECOVERY_MISSES: u32 = 24;

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
    exhausted: u64,
    errors: u64,
}

#[derive(Debug)]
struct BallRecovery {
    class_id: u16,
    cameras: [CameraRecoveryState; 2],
    stats: RecoveryStats,
}

impl BallRecovery {
    fn state_mut(&mut self, camera: CameraId) -> &mut CameraRecoveryState {
        &mut self.cameras[match camera {
            CameraId::Left => 0,
            CameraId::Right => 1,
        }]
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
    // Pre-computed letterbox parameters (constant for fixed frame dimensions).
    scale: f32,
    #[allow(dead_code)]
    new_w: u32,
    #[allow(dead_code)]
    new_h: u32,
    pad_x: f32,
    pad_y: f32,
    // Pre-allocated GPU scratch buffers.
    rgb_u8: CUdeviceptr,
    /// Separate destination for the 180-degree mirror step. NPP's
    /// `nppiMirror_8u_C3R` with `NPPI_AXIS_BOTH` is *not* safe in-place
    /// (the top half gets overwritten before the bottom half is read),
    /// so a distinct scratch is required. Same size as `rgb_u8`.
    rgb_scratch: CUdeviceptr,
    resized_u8: CUdeviceptr,
    tensor_f32: CUdeviceptr,
    // P010 (10-bit NV12) conversion scratch buffers.
    // Allocated only when the source produces P010 frames.
    // Y plane: width * height bytes, UV plane: width * height/2 bytes.
    nv12_8bit_y: CUdeviceptr,
    nv12_8bit_uv: CUdeviceptr,
    // Cached CUDA device MemoryInfo. Constant for the detector's
    // lifetime; constructing one per inference showed up on the
    // per-frame alloc audit (plan M7 item 5).
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
        // Crop coordinates are expressed in the detector's oriented image
        // space. For a 180-degree source, translate the crop back into raw
        // plane coordinates before offsetting the CUDA pointers; the kernel
        // then applies its normal 180-degree mapping within that crop.
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

    /// Try to create a GPU YOLO detector.
    ///
    /// Returns `Ok(None)` if NPP libraries are not available (e.g. on systems
    /// without NVIDIA GPU or without CUDA toolkit). Returns `Err` for real
    /// failures like missing model file or ORT initialization errors.
    ///
    /// `frame_width`/`frame_height` are the raw camera frame dimensions
    /// (e.g. 3840x2160 for 4K). These must match what the decode pipeline
    /// produces. Letterbox parameters are pre-computed from these dimensions.
    ///
    /// When `is_10bit` is true, additional scratch buffers are allocated for
    /// converting P010 (10-bit NV12) frames to 8-bit before NPP color
    /// conversion.
    pub fn try_new(
        model_path: impl AsRef<Path>,
        frame_width: u32,
        frame_height: u32,
        confidence_threshold: f32,
        labels: Vec<String>,
        is_10bit: bool,
    ) -> Result<Option<Self>, Box<dyn std::error::Error>> {
        // GPU detection needs a CUDA-capable EP (TensorRT or CUDA) to
        // process device pointers. Without one, ORT falls back to CPU EP
        // which segfaults on CUDA memory.
        if !cfg!(feature = "tensorrt") && !cfg!(feature = "cuda") {
            log::warn!(
                "OrtGpuDetector: no GPU EP available (need --features tensorrt or --features cuda)"
            );
            return Ok(None);
        }

        cuda_ensure_context()?;

        let (session, input_size, labels) =
            crate::ort_session::create_ort_session(model_path.as_ref(), labels)?;

        // Pre-compute letterbox parameters.
        let (fw, fh) = (frame_width as f32, frame_height as f32);
        let is = input_size as f32;
        let scale = (is / fw).min(is / fh);
        let new_w = (fw * scale).round() as u32;
        let new_h = (fh * scale).round() as u32;
        let pad_x = (input_size - new_w) as f32 / 2.0;
        let pad_y = (input_size - new_h) as f32 / 2.0;

        // Allocate GPU scratch buffers (checked arithmetic to prevent overflow).
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

        // Allocate P010 conversion scratch buffers if needed.
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

        // Fill resized buffer with grey (114) for letterbox padding.
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

        // Warmup: force TRT EP to eagerly build the engine and initialize
        // CUDA resources. Without this, the first real inference triggers
        // lazy init which can conflict with NVDEC decode thread contexts.
        {
            let sz = input_size as usize;
            let warmup_data = vec![0.0f32; 3 * sz * sz];
            let tensor = ort::value::Tensor::from_array(([1, 3, sz, sz], warmup_data))?;
            detector.session.run(ort::inputs![tensor])?;
            log::info!("OrtGpuDetector: warmup inference complete");
        }

        Ok(Some(detector))
    }

    /// Enable native-resolution crop retries when the full-frame pass has no
    /// ball detection. The ball class is resolved from ONNX metadata; if the
    /// model has no `ball`/`sports ball` label, recovery remains disabled.
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
                    stats: RecoveryStats::default(),
                });
                log::info!(
                    "OrtGpuDetector: high-resolution ball recovery enabled (class_id={class_id}, crop_ratios={RECOVERY_CROP_RATIOS:?}, max_misses={MAX_RECOVERY_MISSES})"
                );
            }
            None => log::warn!(
                "OrtGpuDetector: high-resolution ball recovery requested, but model labels contain neither 'ball' nor 'sports ball'; recovery disabled"
            ),
        }
        self
    }
}

impl OrtGpuDetector {
    /// Core inference path shared by the legacy [`GpuDetector`] impl
    /// and the new [`UnifiedDetector`] impl. Returns a typed
    /// [`DetectorError`] so unified-trait consumers can distinguish
    /// "no CUDA context" from "inference failed"; the legacy impl
    /// collapses the error to a log + empty vector for backward
    /// compatibility.
    ///
    /// Each CUDA / NPP / ORT step that previously logged and returned
    /// an empty vec now returns
    /// `Err(DetectorError::InferenceFailed(msg))` preserving the
    /// original error text verbatim.
    fn detect_gpu_raw(
        &mut self,
        camera: CameraId,
        frame: &GpuNv12Frame,
    ) -> Result<Vec<Detection>, DetectorError> {
        let GpuNv12Frame {
            y_ptr,
            uv_ptr,
            y_pitch,
            uv_pitch,
            width,
            height,
            rotation,
            is_10bit,
        } = *frame;
        reco_core::profile_scope!("gpu_yolo_detect");

        if width != self.frame_width || height != self.frame_height {
            return Err(DetectorError::InferenceFailed(format!(
                "GPU detector frame dimensions changed: configured={}x{}, received={}x{}",
                self.frame_width, self.frame_height, width, height
            )));
        }

        // Ensure a CUDA context is current on this thread. The zero-copy
        // frame loop may not have one after NVDEC decode pushes/pops its
        // own context.
        reco_core::interop::cuda::cuda_ensure_context()
            .map_err(|e| DetectorError::InferenceFailed(format!("cuda_ensure_context: {e}")))?;

        // Step 0: Convert P010 (10-bit) to 8-bit NV12 if needed.
        // NPP's NV12->RGB expects 8-bit samples, so we must down-convert
        // first by extracting the high byte of each u16 sample.
        let (nv12_y, nv12_y_pitch, nv12_uv, nv12_uv_pitch) = if is_10bit {
            reco_core::profile_scope!("p010_to_nv12");
            if self.nv12_8bit_y == 0 || self.nv12_8bit_uv == 0 {
                return Err(DetectorError::InferenceFailed(
                    "P010 frame received but no conversion buffers allocated".into(),
                ));
            }
            // Convert Y plane: width * height samples.
            crate::cuda_kernels::p010_plane_to_nv12(
                y_ptr,
                y_pitch,
                self.nv12_8bit_y,
                width,
                height,
            )
            .map_err(|e| DetectorError::InferenceFailed(format!("P010->NV12 Y conversion: {e}")))?;
            // Convert UV plane: width * (height/2) samples.
            // UV plane has width/2 pixel pairs, each 2 u16 values = width u16 samples per row.
            crate::cuda_kernels::p010_plane_to_nv12(
                uv_ptr,
                uv_pitch,
                self.nv12_8bit_uv,
                width,
                height / 2,
            )
            .map_err(|e| {
                DetectorError::InferenceFailed(format!("P010->NV12 UV conversion: {e}"))
            })?;
            // The 8-bit buffers are tightly packed (no pitch padding).
            (
                self.nv12_8bit_y,
                width as usize,
                self.nv12_8bit_uv,
                width as usize,
            )
        } else {
            (y_ptr, y_pitch, uv_ptr, uv_pitch)
        };

        let mut detections = self.infer_region(
            camera,
            nv12_y,
            nv12_y_pitch,
            nv12_uv,
            nv12_uv_pitch,
            width,
            height,
            rotation,
            CropRegion::full(width, height),
        )?;

        if let Some(ball_class_id) = self.ball_recovery.as_ref().map(|r| r.class_id) {
            if detections.iter().any(|d| d.class_id == ball_class_id) {
                if let Some(recovery) = self.ball_recovery.as_mut() {
                    recovery
                        .state_mut(camera)
                        .observe(&detections, ball_class_id);
                }
            } else {
                let predicted = self.ball_recovery.as_ref().and_then(|recovery| {
                    let state = recovery.cameras[match camera {
                        CameraId::Left => 0,
                        CameraId::Right => 1,
                    }];
                    (state.misses < MAX_RECOVERY_MISSES)
                        .then(|| state.predicted_center())
                        .flatten()
                });

                if let Some(predicted) = predicted {
                    let mut recovered = false;
                    for (stage, ratio) in RECOVERY_CROP_RATIOS.into_iter().enumerate() {
                        let region = CropRegion::centered(width, height, ratio, predicted);
                        if region == CropRegion::full(width, height) {
                            continue;
                        }
                        if let Some(recovery) = self.ball_recovery.as_mut() {
                            recovery.stats.attempts[stage] += 1;
                        }
                        log::debug!(
                            "BALL_RECOVERY_ATTEMPT camera={camera} stage={} crop={}x{}+{},{} predicted={:.4},{:.4}",
                            stage + 1,
                            region.width,
                            region.height,
                            region.x,
                            region.y,
                            predicted.0,
                            predicted.1,
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
                                    "BALL_RECOVERY_ERROR camera={camera} stage={} error={error}",
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
                                recovery
                                    .state_mut(camera)
                                    .observe(&recovered_balls, ball_class_id);
                            }
                            log::info!(
                                "BALL_RECOVERY_HIT camera={camera} stage={} crop={}x{} count={} best_confidence={:.3}",
                                stage + 1,
                                region.width,
                                region.height,
                                recovered_balls.len(),
                                recovered_balls
                                    .iter()
                                    .map(|d| d.confidence)
                                    .fold(0.0f32, f32::max),
                            );
                            detections.extend(recovered_balls);
                            recovered = true;
                            break;
                        }
                    }
                    if !recovered && let Some(recovery) = self.ball_recovery.as_mut() {
                        let state = recovery.state_mut(camera);
                        state.misses = state.misses.saturating_add(1);
                        recovery.stats.exhausted += 1;
                    }
                }
            }
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
        // CUDA-residency backend: accept `Cuda(GpuNv12Frame)` and
        // route everything else to `UnsupportedFrameKind` so the
        // dispatcher can fall back to a CPU backend for `Cpu(_)`.
        // The wildcard arm keeps this stable against future
        // `#[non_exhaustive]` additions to `DetectorFrame`.
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
                "BALL_RECOVERY_SUMMARY attempts={:?} hits={:?} exhausted={} errors={}",
                recovery.stats.attempts,
                recovery.stats.hits,
                recovery.stats.exhausted,
                recovery.stats.errors,
            );
        }
        // Ensure a CUDA context is current before freeing GPU memory.
        // Drop may run on a different thread than the one that allocated.
        if let Err(e) = cuda_ensure_context() {
            log::warn!("OrtGpuDetector drop: failed to set CUDA context: {e}");
            return;
        }
        // Free GPU scratch buffers. Log errors but don't panic in Drop.
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
}
