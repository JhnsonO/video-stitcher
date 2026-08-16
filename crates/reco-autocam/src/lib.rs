//! Automatic camera control for reco.
//!
//! The intelligence layer. [`trackers`] turn noisy per-frame detections
//! into a clean [`WorldState`](reco_core::detect::tracker::WorldState) with
//! stable identities and lifecycle flags; [`panners`] turn that world
//! state into a virtual-camera [`ViewportPosition`](reco_core::detect::director::ViewportPosition).
//! Detector backends live in [`reco_detect`] and are re-exported at
//! crate root for convenience but are not owned here.

#![forbid(unsafe_code)]

pub mod panners;
#[cfg(all(feature = "ort", target_os = "linux"))]
mod recovery_filter;
mod roi_filter;
pub mod trackers;
mod tracking_mode;

#[cfg(feature = "ort")]
pub use reco_detect::CpuYoloDetector;
#[cfg(all(feature = "ort", target_os = "macos"))]
pub use reco_detect::MetalYoloDetector;
#[cfg(all(feature = "ort", any(target_os = "linux", target_os = "windows")))]
pub use reco_detect::OrtGpuDetector;
#[cfg(feature = "tensorrt-native")]
pub use reco_detect::TrtGpuDetector;

pub use roi_filter::{RoiAnchor, RoiFilteredDetector};
pub mod wgpu_detector;
pub use wgpu_detector::WgpuPreprocessingDetector;
pub use tracking_mode::TrackingMode;

use std::io;
use std::path::Path;

/// Highest sparse-analysis stride currently validated for production use.
pub const MAX_FRAME_STRIDE: u64 = 4;

fn stride_alpha(alpha: f32, stride: u64) -> f32 {
    if stride <= 1 {
        return alpha;
    }
    1.0 - (1.0 - alpha).powf(stride as f32)
}

fn rebase_panner_config_for_stride(
    mut config: crate::panners::FieldPannerConfig,
    stride: u64,
) -> crate::panners::FieldPannerConfig {
    if stride <= 1 {
        return config;
    }
    config.cluster_alpha = stride_alpha(config.cluster_alpha, stride);
    config.fov_alpha = stride_alpha(config.fov_alpha, stride);
    config.velocity_alpha = stride_alpha(config.velocity_alpha, stride);
    config.lead_alpha = stride_alpha(config.lead_alpha, stride);
    config.ball_presence_attack = stride_alpha(config.ball_presence_attack, stride);
    config.ball_presence_decay = config.ball_presence_decay.powf(stride as f32);
    config
}

fn coast_frames_for_stride(stride: u64) -> u32 {
    let stride = stride.max(1);
    let base = crate::trackers::ball::DEFAULT_COAST_FRAMES as u64;
    base.div_ceil(stride).max(1) as u32
}

pub fn validate_model_path(path: &Path) -> io::Result<()> {
    if path.as_os_str().is_empty() || !path.try_exists()? {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("AI model path does not exist: {}", path.display()),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct AutocamConfig {
    pub model_path: std::path::PathBuf,
    pub tracking_mode: TrackingMode,
    pub detection_interval: u64,
    pub frame_stride: u64,
    pub field_roi: Option<reco_core::calibration::FieldRoi>,
    pub is_10bit: bool,
    pub field_panner_config: Option<crate::panners::FieldPannerConfig>,
    pub confidence_threshold: Option<f32>,
    pub high_res_ball_recovery: bool,
}

impl AutocamConfig {
    pub fn new(model_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            tracking_mode: TrackingMode::Field,
            detection_interval: 1,
            frame_stride: 1,
            field_roi: None,
            is_10bit: false,
            field_panner_config: None,
            confidence_threshold: None,
            high_res_ball_recovery: false,
        }
    }

    pub fn with_tracking_mode(mut self, mode: TrackingMode) -> Self {
        self.tracking_mode = mode;
        self
    }

    pub fn with_detection_interval(mut self, interval: u64) -> Self {
        self.detection_interval = interval;
        self
    }

    pub fn with_frame_stride(mut self, stride: u64) -> Self {
        self.frame_stride = stride.clamp(1, MAX_FRAME_STRIDE);
        self
    }

    pub fn with_field_roi(mut self, roi: reco_core::calibration::FieldRoi) -> Self {
        self.field_roi = Some(roi);
        self
    }

    pub fn with_10bit(mut self, is_10bit: bool) -> Self {
        self.is_10bit = is_10bit;
        self
    }

    pub fn with_high_res_ball_recovery(mut self, enabled: bool) -> Self {
        self.high_res_ball_recovery = enabled;
        self
    }
}

#[cfg_attr(
    not(any(feature = "ort", feature = "tensorrt-native", feature = "ncnn")),
    allow(unused_variables, unreachable_code)
)]
pub fn setup_autocam(
    target: &mut impl reco_core::detect::DetectionTarget,
    config: &AutocamConfig,
    fps: f32,
    source_is_gpu_resident: bool,
) -> Result<bool, Box<dyn std::error::Error>> {
    if config.tracking_mode != TrackingMode::Sweep {
        validate_model_path(&config.model_path)?;
    }

    #[cfg(not(any(feature = "ort", feature = "tensorrt-native", feature = "ncnn")))]
    {
        let _ = (target, config, fps);
        log::warn!(
            "Autocam: no detector backend compiled in (enable ort, tensorrt-native, or ncnn). \
             Session will run without AI camera control."
        );
        return Ok(false);
    }

    let (input_width, input_height) = target.pipeline().source_info();
    let use_zero_copy = source_is_gpu_resident;
    let model_path = config.model_path.to_str().unwrap_or("");
    let detection_interval = config.detection_interval;
    let tracking_mode = config.tracking_mode;
    let field_roi = config.field_roi.as_ref();
    let is_10bit = config.is_10bit;
    let frame_stride = config.frame_stride.clamp(1, MAX_FRAME_STRIDE);
    let panner_fps = fps / frame_stride as f32;
    let coast_frames = coast_frames_for_stride(frame_stride);
    if frame_stride > 1 {
        log::info!(
            "Autocam sparse analysis: source_fps={fps:.3}, stride={frame_stride}, \
             analysis_fps={panner_fps:.3}, ball_coast_frames={coast_frames}"
        );
    }

    let mut detection_active = false;
    let is_onnx = model_path.ends_with(".onnx");
    #[cfg(feature = "ort")]
    let class_names = if is_onnx {
        match reco_detect::create_ort_session(Path::new(model_path), Vec::new()) {
            Ok((_, _, names)) => names,
            Err(e) => {
                log::warn!("Could not read model labels: {e}, using COCO defaults");
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };
    #[cfg(not(feature = "ort"))]
    let class_names: Vec<String> = {
        if is_onnx {
            log::warn!(
                "Autocam: ort feature disabled; can't parse ONNX class names from {model_path}. \
                 Using COCO defaults. For .engine models, place a <name>.labels sidecar."
            );
        }
        Vec::new()
    };

    let effective_roi = field_roi
        .filter(|roi| roi.left.len() >= 3 || roi.right.len() >= 3)
        .cloned();
    if effective_roi.is_some() {
        log::info!("Autocam: field ROI filtering enabled");
    }

    let person_id_for_roi = resolve_or(&class_names, &["person"], 0);
    let ball_id_for_recovery = resolve_or(&class_names, &["ball", "sports ball"], 32);

    let wrap_with_roi = |inner: Box<dyn reco_core::detect::detector::UnifiedDetector>,
                         roi: reco_core::calibration::FieldRoi|
     -> Box<dyn reco_core::detect::detector::UnifiedDetector> {
        Box::new(
            RoiFilteredDetector::new(inner, roi)
                .with_class_anchor(person_id_for_roi, RoiAnchor::Bottom),
        )
    };

    #[cfg(feature = "tensorrt-native")]
    if use_zero_copy && model_path.ends_with(".engine") {
        let labels_path = std::path::Path::new(model_path).with_extension("labels");
        let trt_labels = reco_detect::read_labels_file(&labels_path);
        if !trt_labels.is_empty() {
            log::info!(
                "Autocam: loaded {} class labels from {}",
                trt_labels.len(),
                labels_path.display()
            );
        }
        match reco_detect::TrtGpuDetector::try_new(
            model_path,
            input_width,
            input_height,
            config.confidence_threshold.unwrap_or(0.10),
            trt_labels,
            is_10bit,
        ) {
            Ok(Some(trt_det)) => {
                let detector: Box<dyn reco_core::detect::detector::UnifiedDetector> =
                    if let Some(roi) = effective_roi.clone() {
                        wrap_with_roi(Box::new(trt_det), roi)
                    } else {
                        Box::new(trt_det)
                    };
                target.set_detector(detector);
                detection_active = true;
                log::info!("Autocam: native TensorRT tracking enabled (engine: {model_path})");
            }
            Ok(None) => log::warn!("Autocam: NPP not available, TRT detection disabled"),
            Err(e) => log::warn!("Autocam: TRT detector init failed ({e})"),
        }
    }

    #[cfg(all(feature = "ort", target_os = "linux"))]
    if !detection_active && use_zero_copy {
        match OrtGpuDetector::try_new(
            model_path,
            input_width,
            input_height,
            config.confidence_threshold.unwrap_or(0.10),
            Vec::new(),
            is_10bit,
        ) {
            Ok(Some(gpu_det)) => {
                let gpu_det = gpu_det.with_high_res_ball_recovery(config.high_res_ball_recovery);
                let detector: Box<dyn reco_core::detect::detector::UnifiedDetector> =
                    if config.high_res_ball_recovery && gpu_det.recovery_ball_class_id().is_some() {
                        Box::new(recovery_filter::ValidatedBallRecoveryDetector::new(
                            gpu_det,
                            effective_roi.clone(),
                            ball_id_for_recovery,
                            person_id_for_roi,
                        ))
                    } else if let Some(roi) = effective_roi.clone() {
                        wrap_with_roi(Box::new(gpu_det), roi)
                    } else {
                        Box::new(gpu_det)
                    };
                target.set_detector(detector);
                detection_active = true;
                log::info!("Autocam: GPU YOLO ball tracking enabled (model: {model_path})");
            }
            Ok(None) => {
                log::warn!("Autocam: NPP not available, ball tracking disabled in zero-copy mode");
            }
            Err(e) => {
                log::warn!("Autocam: GPU detector init failed ({e}), ball tracking disabled");
            }
        }
    }

    #[cfg(all(feature = "ort", target_os = "macos"))]
    if use_zero_copy {
        match MetalYoloDetector::try_new(
            model_path,
            target.gpu(),
            input_width,
            input_height,
            config.confidence_threshold.unwrap_or(0.10),
            Vec::new(),
        ) {
            Ok(metal_det) => {
                let detector: Box<dyn reco_core::detect::detector::UnifiedDetector> =
                    if let Some(roi) = effective_roi.clone() {
                        wrap_with_roi(Box::new(metal_det), roi)
                    } else {
                        Box::new(metal_det)
                    };
                target.set_detector(detector);
                detection_active = true;
                log::info!("Autocam: Metal YOLO ball tracking enabled (model: {model_path})");
            }
            Err(e) => {
                log::warn!("Autocam: Metal detector init failed ({e}), ball tracking disabled");
            }
        }
    }

    #[cfg(feature = "ncnn")]
    if !detection_active && std::path::Path::new(model_path).is_dir() {
        match reco_detect::NcnnYoloDetector::new(
            model_path,
            640,
            input_width,
            input_height,
            config.confidence_threshold.unwrap_or(0.25),
            Vec::new(),
        ) {
            Ok(ncnn_det) => {
                let detector: Box<dyn reco_core::detect::detector::UnifiedDetector> =
                    if let Some(roi) = effective_roi.clone() {
                        wrap_with_roi(Box::new(ncnn_det), roi)
                    } else {
                        Box::new(ncnn_det)
                    };
                target.set_detector(detector);
                detection_active = true;
                log::info!("Autocam: NCNN YOLO tracking enabled (model: {model_path})");
            }
            Err(e) => {
                log::warn!("Autocam: NCNN detector init failed ({e}), trying ORT fallback");
            }
        }
    }

    #[cfg(feature = "ort")]
    if !detection_active && use_zero_copy {
        let gpu = target.gpu();
        let yolo = CpuYoloDetector::with_config(
            model_path,
            config.confidence_threshold.unwrap_or(0.10),
            Vec::new(),
        )?;
        let input_size = yolo.input_size();
        let wrapper = WgpuPreprocessingDetector::new(
            Box::new(yolo),
            gpu.device().clone(),
            gpu.queue().clone(),
            input_size,
            input_width,
            input_height,
        );
        let detector: Box<dyn reco_core::detect::detector::UnifiedDetector> =
            if let Some(roi) = effective_roi.clone() {
                wrap_with_roi(Box::new(wrapper), roi)
            } else {
                Box::new(wrapper)
            };
        target.set_detector(detector);
        detection_active = true;
        log::info!("Autocam: wgpu preprocessing + DirectML tracking enabled (model: {model_path})");
    }

    #[cfg(feature = "ort")]
    if !detection_active && !use_zero_copy {
        let conf = config.confidence_threshold.unwrap_or(0.10);
        let yolo = CpuYoloDetector::with_config(model_path, conf, Vec::new())?;
        let detector: Box<dyn reco_core::detect::detector::UnifiedDetector> =
            if let Some(roi) = effective_roi {
                wrap_with_roi(Box::new(yolo), roi)
            } else {
                Box::new(yolo)
            };
        target.set_detector(detector);
        detection_active = true;
        log::info!("Autocam: YOLO ball tracking enabled (model: {model_path})");
    }

    #[cfg(not(feature = "ort"))]
    if !detection_active {
        log::warn!(
            "Autocam: no detector attached. Build has `ort` disabled; only `.engine` \
             (tensorrt-native) and NCNN `_ncnn_model` directories are supported. \
             Received model_path={model_path}"
        );
    }

    if tracking_mode == TrackingMode::Sweep {
        log::info!("Tracking mode: sweep (debug, no AI)");
        let panner =
            Box::new(crate::panners::SweepPanner::new(0.8, 10.0).with_zoom(30.0, 90.0, 7.0));
        target.set_panner(panner);
        return Ok(true);
    }

    if detection_active {
        if detection_interval > 1 {
            target.set_detection_interval(detection_interval);
            log::info!("Detection interval: every {detection_interval} frames");
        }

        let ball_id = resolve_or(&class_names, &["ball", "sports ball"], 32);
        let person = resolve_class_id(&class_names, &["person"]);
        match person {
            Some(p) => log::info!(
                "Resolved class ids from {} model labels: ball={ball_id}, person={p}",
                class_names.len()
            ),
            None => log::info!(
                "Resolved class ids from {} model labels: ball={ball_id}, person=absent",
                class_names.len()
            ),
        }

        match tracking_mode {
            TrackingMode::Field => {
                let ball_tracker = crate::trackers::BallTracker::new(ball_id)
                    .with_max_jump_rad(0.8)
                    .with_max_coast_frames(coast_frames);
                target.set_ball_tracker(Box::new(ball_tracker));

                match person {
                    Some(person_id) => {
                        target.set_player_tracker(Box::new(crate::trackers::ClassProvider::new(
                            person_id,
                        )));
                        log::info!(
                            "Tracking mode: field with player cluster \
                             (player_class={person_id}, ball_class={ball_id})"
                        );
                    }
                    None => log::info!(
                        "Tracking mode: field, but the model names no player class - the panner \
                         will follow the ball alone (ball_class={ball_id})"
                    ),
                }

                let fp_config = config.field_panner_config.clone().unwrap_or(
                    crate::panners::FieldPannerConfig {
                        ball_weight: 0.20,
                        ..Default::default()
                    },
                );
                let fp_config = rebase_panner_config_for_stride(fp_config, frame_stride);
                log::info!(
                    "FieldPanner: framing={:?}, confidence_weighted={}, lock_pitch={}",
                    fp_config.framing,
                    fp_config.confidence_weighted,
                    fp_config.lock_pitch,
                );
                let field_panner = crate::panners::FieldPanner::with_config(panner_fps, fp_config);
                target.set_panner(Box::new(field_panner));
            }
            TrackingMode::Ball => {
                let ball_tracker = crate::trackers::BallTracker::new(ball_id)
                    .with_max_jump_rad(0.5)
                    .with_max_coast_frames(coast_frames);
                target.set_ball_tracker(Box::new(ball_tracker));

                let mut fp_config = config.field_panner_config.clone().unwrap_or_default();
                fp_config.ball_weight = 1.0;
                let fp_config = rebase_panner_config_for_stride(fp_config, frame_stride);
                let panner = crate::panners::FieldPanner::with_config(panner_fps, fp_config);

                log::info!(
                    "Tracking mode: ball-only (BallTracker + FieldPanner, \
                     ball_class={ball_id})"
                );
                target.set_panner(Box::new(panner));
            }
            TrackingMode::Sweep => unreachable!("handled before detection block"),
        }
    }

    Ok(detection_active)
}

fn resolve_class_id(class_names: &[String], candidates: &[&str]) -> Option<u16> {
    candidates.iter().find_map(|candidate| {
        class_names
            .iter()
            .position(|name| name.eq_ignore_ascii_case(candidate))
            .map(|idx| idx as u16)
    })
}

fn resolve_or(class_names: &[String], candidates: &[&str], default_id: u16) -> u16 {
    resolve_class_id(class_names, candidates).unwrap_or_else(|| {
        log::warn!(
            "Class '{}' not found in model labels; using COCO default id {default_id}",
            candidates[0]
        );
        default_id
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_path_validation_accepts_files_and_directories() {
        let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        assert!(validate_model_path(crate_dir).is_ok());
        assert!(validate_model_path(&crate_dir.join("Cargo.toml")).is_ok());
    }

    #[test]
    fn stride_one_panner_rebase_is_identity() {
        let config = crate::panners::FieldPannerConfig::broadcast();
        assert_eq!(rebase_panner_config_for_stride(config.clone(), 1), config);
        assert_eq!(
            coast_frames_for_stride(1),
            crate::trackers::ball::DEFAULT_COAST_FRAMES
        );
    }

    #[test]
    fn stride_rebase_preserves_per_source_time_constants() {
        let mut config = crate::panners::FieldPannerConfig::broadcast();
        config.cluster_alpha = 0.10;
        config.fov_alpha = 0.20;
        config.velocity_alpha = 0.30;
        config.lead_alpha = 0.40;
        config.ball_presence_attack = 0.15;
        config.ball_presence_decay = 0.90;
        let rebased = rebase_panner_config_for_stride(config, 2);
        assert!((rebased.cluster_alpha - 0.19).abs() < 1e-6);
        assert!((rebased.fov_alpha - 0.36).abs() < 1e-6);
        assert!((rebased.velocity_alpha - 0.51).abs() < 1e-6);
        assert!((rebased.lead_alpha - 0.64).abs() < 1e-6);
        assert!((rebased.ball_presence_attack - 0.2775).abs() < 1e-6);
        assert!((rebased.ball_presence_decay - 0.81).abs() < 1e-6);
        assert_eq!(coast_frames_for_stride(2), 10);
        assert_eq!(coast_frames_for_stride(3), 7);
        assert_eq!(coast_frames_for_stride(4), 5);
    }

    #[test]
    fn model_path_validation_rejects_empty_and_missing_paths() {
        let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let missing = crate_dir.join("model-that-does-not-exist.onnx");
        assert_eq!(
            validate_model_path(&missing).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
        assert_eq!(
            validate_model_path(Path::new("")).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
    }
}
