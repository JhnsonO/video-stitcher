//! ROI-filtered detector decorator.
//!
//! Wraps any [`UnifiedDetector`] to drop detections whose camera-space
//! position falls outside a playing-field polygon. This moves
//! sports-domain filtering out of `reco-core` (which stays domain-
//! agnostic) into `reco-autocam`.
//!
//! Per-class anchor policy (Step 7c):
//!
//! - [`RoiAnchor::Center`] (default) tests the bounding-box center.
//!   Appropriate for balls and any "the detection is a point-shaped
//!   target" class.
//! - [`RoiAnchor::Bottom`] tests the feet (bottom center) AND the
//!   75th-percentile height. Both must be inside the polygon. Meant
//!   for upright classes (players, refs) so a coach leaning in from
//!   the sideline whose feet are just inside the boundary but whose
//!   body is mostly outside gets rejected.
//! - [`RoiAnchor::Top`] is the upside-down mirror of `Bottom`.
//!
//! One decorator covers every residency (CPU / CUDA / Metal) because
//! [`UnifiedDetector`] dispatches internally on the
//! [`DetectorFrame`] variant the backend accepts.

use std::collections::HashMap;

use reco_core::calibration::FieldRoi;
use reco_core::detect::detector::{
    CameraId, Detection, DetectorError, DetectorFrame, UnifiedDetector,
};
use reco_core::projection::point_in_polygon;

/// Where on a detection's bounding box the ROI test samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoiAnchor {
    /// Sample only `(center_x, center_y)`. Default, appropriate for
    /// balls and other point-shaped targets.
    #[default]
    Center,
    /// Sample the feet `(center_x, center_y + height/2)` AND the
    /// 75th-percentile height `(center_x, center_y + height/4)`.
    /// Both must be inside the polygon. Meant for upright classes.
    Bottom,
    /// Mirror of [`Bottom`](RoiAnchor::Bottom) for upside-down
    /// captures or ceiling-mounted cameras.
    Top,
}

impl RoiAnchor {
    /// Return `true` if the detection passes the anchor test against
    /// `polygon`. A polygon with fewer than 3 vertices is treated as
    /// "no filter" by the caller; this method does not re-check that
    /// case.
    fn passes(self, d: &Detection, polygon: &[[f64; 2]]) -> bool {
        let cx = d.center_x as f64;
        let cy = d.center_y as f64;
        let half_h = (d.height as f64) * 0.5;
        let quarter_h = (d.height as f64) * 0.25;
        match self {
            RoiAnchor::Center => point_in_polygon([cx, cy], polygon),
            RoiAnchor::Bottom => {
                point_in_polygon([cx, cy + half_h], polygon)
                    && point_in_polygon([cx, cy + quarter_h], polygon)
            }
            RoiAnchor::Top => {
                point_in_polygon([cx, cy - half_h], polygon)
                    && point_in_polygon([cx, cy - quarter_h], polygon)
            }
        }
    }
}

/// Filter detections by field ROI polygon using a per-class anchor
/// policy. Classes without an explicit entry use `default_anchor`.
fn filter_by_roi(
    detections: Vec<Detection>,
    roi: &FieldRoi,
    class_anchors: &HashMap<u16, RoiAnchor>,
    default_anchor: RoiAnchor,
) -> Vec<Detection> {
    detections
        .into_iter()
        .filter(|d| {
            let polygon = match d.camera {
                CameraId::Left => &roi.left,
                CameraId::Right => &roi.right,
            };
            if polygon.len() < 3 {
                return true;
            }
            let anchor = class_anchors
                .get(&d.class_id)
                .copied()
                .unwrap_or(default_anchor);
            anchor.passes(d, polygon)
        })
        .collect()
}

/// Ball ROI test with a vertical allowance ABOVE the polygon.
///
/// Image y grows downward, so "a ball above the pitch line" means the
/// field polygon lies BELOW the ball. The ball passes if the polygon
/// contains the ball centre itself, or any point up to `margin` straight
/// below it (sampled every 0.02 of frame height, endpoints included).
/// The earlier single test at `cy + margin` accepted lofted balls but
/// also rejected every ball in the lower part of the frame (the near
/// side of the pitch), because `cy + margin` fell below the polygon's
/// bottom edge. This version accepts a strict superset of that test.
fn ball_passes_roi(cx: f64, cy: f64, margin: f64, polygon: &[[f64; 2]]) -> bool {
    if point_in_polygon([cx, cy], polygon) {
        return true;
    }
    if margin <= 0.0 {
        return false;
    }
    const STEP: f64 = 0.02;
    let n = (margin / STEP).ceil() as usize;
    (1..=n).any(|i| {
        let dy = (i as f64 * STEP).min(margin);
        point_in_polygon([cx, cy + dy], polygon)
    })
}

/// Apply the production field-mode ROI policy to detections that were
/// recorded WITHOUT an ROI (OEV-006 CPU replay).
///
/// Mirrors [`RoiFilteredDetector`] as `setup_autocam` builds it for field
/// mode: `person_class` uses [`RoiAnchor::Bottom`], every other non-ball
/// class [`RoiAnchor::Center`], and `ball_class` the vertical-allowance
/// test with `ball_vertical_margin`. Production filters once per camera
/// call, emitting the kept non-ball detections followed by the kept ball
/// detections; consecutive same-camera runs in `detections` are treated
/// as one call so the output order matches the live tracker input. The
/// test runs on each detection's recorded camera-space box
/// (`camera_center`, `camera_size`), the exact values the live decorator
/// saw.
pub fn filter_mapped_field_mode(
    detections: &[reco_core::detect::director::MappedDetection],
    roi: &FieldRoi,
    person_class: u16,
    ball_class: u16,
    ball_vertical_margin: f64,
) -> Vec<reco_core::detect::director::MappedDetection> {
    let mut anchors = HashMap::new();
    anchors.insert(person_class, RoiAnchor::Bottom);
    let mut out = Vec::with_capacity(detections.len());
    let mut start = 0;
    while start < detections.len() {
        let camera = detections[start].camera;
        let mut end = start;
        while end < detections.len() && detections[end].camera == camera {
            end += 1;
        }
        let group = &detections[start..end];
        let polygon: &[[f64; 2]] = match camera {
            CameraId::Left => &roi.left,
            CameraId::Right => &roi.right,
        };
        let as_detection = |m: &reco_core::detect::director::MappedDetection| Detection {
            camera: m.camera,
            class_id: m.class_id,
            confidence: m.confidence,
            center_x: m.camera_center.0,
            center_y: m.camera_center.1,
            width: m.camera_size.0,
            height: m.camera_size.1,
        };
        for m in group.iter().filter(|m| m.class_id != ball_class) {
            let keep = polygon.len() < 3 || {
                let anchor = anchors.get(&m.class_id).copied().unwrap_or_default();
                anchor.passes(&as_detection(m), polygon)
            };
            if keep {
                out.push(*m);
            }
        }
        for m in group.iter().filter(|m| m.class_id == ball_class) {
            let keep = polygon.len() < 3
                || ball_passes_roi(
                    m.camera_center.0 as f64,
                    m.camera_center.1 as f64,
                    ball_vertical_margin,
                    polygon,
                );
            if keep {
                out.push(*m);
            }
        }
        start = end;
    }
    out
}

/// An [`UnifiedDetector`] decorator that filters output detections by
/// field ROI polygon using a per-class anchor policy.
///
/// Wraps an inner detector and drops detections outside the playing
/// field after each `detect()` call. Works for every residency the
/// inner detector supports (CPU / CUDA / Metal / future variants)
/// because the filter runs on the post-inference `Vec<Detection>`.
///
/// Default anchor is [`RoiAnchor::Center`] (good for balls and
/// point-shaped targets); call
/// [`with_class_anchor`](Self::with_class_anchor) to override per
/// class. Common pattern:
///
/// ```rust,ignore
/// RoiFilteredDetector::new(inner, roi)
///     .with_class_anchor(person_class_id, RoiAnchor::Bottom)
/// ```
///
/// `name()` forwards to the inner detector's name so telemetry
/// still identifies the underlying backend; the ROI wrap is invisible
/// to log consumers.
pub struct RoiFilteredDetector {
    inner: Box<dyn UnifiedDetector>,
    roi: FieldRoi,
    class_anchors: HashMap<u16, RoiAnchor>,
    default_anchor: RoiAnchor,
    // Ball-ROI generous vertical margin (OEV hard-case 134-139s fix,
    // ported from ffa-automations commit 916529a61f -- originally a
    // diagnostic-only build-time patch, never committed to this repo).
    // Every other class's filtering path above is untouched;
    // ball_class_id stays None unless with_ball_roi_diagnostics is
    // called, so default behaviour is identical to production.
    ball_class_id: Option<u16>,
    ball_vertical_margin: f64,
    roi_diag_frame_left: u64,
    roi_diag_frame_right: u64,
    roi_diag_fps: f32,
    roi_diag_frame_stride: u32,
}

impl RoiFilteredDetector {
    /// Create a new ROI-filtered detector wrapping `inner`. All
    /// classes default to [`RoiAnchor::Center`] until
    /// [`with_class_anchor`](Self::with_class_anchor) overrides them.
    pub fn new(inner: Box<dyn UnifiedDetector>, roi: FieldRoi) -> Self {
        Self {
            inner,
            roi,
            class_anchors: HashMap::new(),
            default_anchor: RoiAnchor::Center,
            ball_class_id: None,
            ball_vertical_margin: 0.0,
            roi_diag_frame_left: 0,
            roi_diag_frame_right: 0,
            roi_diag_fps: 60.0,
            roi_diag_frame_stride: 1,
        }
    }

    /// Override the anchor for a specific class id. Chainable.
    pub fn with_class_anchor(mut self, class_id: u16, anchor: RoiAnchor) -> Self {
        self.class_anchors.insert(class_id, anchor);
        self
    }

    /// Override the default anchor for classes without an explicit
    /// [`with_class_anchor`](Self::with_class_anchor) entry. Chainable.
    pub fn with_default_anchor(mut self, anchor: RoiAnchor) -> Self {
        self.default_anchor = anchor;
        self
    }

    /// Gives `ball_class_id` a generous vertical allowance above the field
    /// polygon's top edge before the ROI test, so a lofted ball whose
    /// center is genuinely above the pitch line is not rejected purely for
    /// being airborne. No other class is affected: `filter_by_roi` below
    /// still runs unmodified for every class except this one. `fps` /
    /// `frame_stride` are used only to derive an approximate timestamp for
    /// window-scoped `eprintln!` diagnostics, matching the same cadence
    /// formula used elsewhere in the autocam pipeline. Chainable.
    ///
    /// Ported from the OEV hard-case 134-139s ball-ROI fix
    /// (ffa-automations commit 916529a61f); `vertical_margin=0.40` is the
    /// value that fix used, not re-derived here.
    pub fn with_ball_roi_diagnostics(
        mut self,
        ball_class_id: u16,
        vertical_margin: f64,
        fps: f32,
        frame_stride: u32,
    ) -> Self {
        self.ball_class_id = Some(ball_class_id);
        self.ball_vertical_margin = vertical_margin;
        self.roi_diag_fps = fps;
        self.roi_diag_frame_stride = frame_stride;
        self
    }
}

impl UnifiedDetector for RoiFilteredDetector {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn detect(
        &mut self,
        camera: CameraId,
        frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        let detections = self.inner.detect(camera, frame)?;

        // No-op (ball_dets always empty, kept == filter_by_roi(detections,
        // ...)) unless with_ball_roi_diagnostics was called.
        let frame_index = match camera {
            CameraId::Left => {
                let i = self.roi_diag_frame_left;
                self.roi_diag_frame_left += 1;
                i
            }
            CameraId::Right => {
                let i = self.roi_diag_frame_right;
                self.roi_diag_frame_right += 1;
                i
            }
        };
        let timestamp_ms = frame_index as f64 * 1000.0 * self.roi_diag_frame_stride as f64
            / self.roi_diag_fps as f64;
        let diag_window = (133000.0..=140000.0).contains(&timestamp_ms);

        let (ball_dets, other_dets): (Vec<Detection>, Vec<Detection>) = detections
            .into_iter()
            .partition(|d| self.ball_class_id == Some(d.class_id));

        let mut kept = filter_by_roi(
            other_dets,
            &self.roi,
            &self.class_anchors,
            self.default_anchor,
        );

        let polygon: &[[f64; 2]] = match camera {
            CameraId::Left => &self.roi.left,
            CameraId::Right => &self.roi.right,
        };

        if polygon.len() < 3 {
            kept.extend(ball_dets);
        } else {
            for d in ball_dets {
                let cx = d.center_x as f64;
                let cy = d.center_y as f64;
                let adj_cy = cy + self.ball_vertical_margin;
                let roi_pass = ball_passes_roi(cx, cy, self.ball_vertical_margin, polygon);
                if diag_window {
                    eprintln!(
                        "OEV_ROI_DIAG cam={:?} t_ms={:.3} cx={:.4} cy={:.4} adj_cy={:.4} roi_pass={}",
                        camera, timestamp_ms, cx, cy, adj_cy, roi_pass
                    );
                }
                if roi_pass {
                    kept.push(d);
                }
            }
        }

        Ok(kept)
    }

    fn class_names(&self) -> Option<&[String]> {
        self.inner.class_names()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::calibration::FieldRoi;
    use reco_core::detect::detector::{CameraId, Detection};

    fn make_detection(camera: CameraId, cx: f32, cy: f32, w: f32, h: f32) -> Detection {
        make_detection_class(camera, 0, cx, cy, w, h)
    }

    fn make_detection_class(
        camera: CameraId,
        class_id: u16,
        cx: f32,
        cy: f32,
        w: f32,
        h: f32,
    ) -> Detection {
        Detection {
            camera,
            class_id,
            confidence: 0.9,
            center_x: cx,
            center_y: cy,
            width: w,
            height: h,
        }
    }

    fn full_roi() -> FieldRoi {
        FieldRoi {
            left: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            right: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        }
    }

    fn small_roi() -> FieldRoi {
        FieldRoi {
            left: vec![[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
            right: vec![[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
        }
    }

    fn filter(dets: Vec<Detection>, roi: &FieldRoi) -> Vec<Detection> {
        filter_by_roi(dets, roi, &HashMap::new(), RoiAnchor::Center)
    }

    /// Inner detector returning a fixed per-camera script.
    struct Scripted {
        left: Vec<Detection>,
        right: Vec<Detection>,
    }

    impl UnifiedDetector for Scripted {
        fn name(&self) -> &'static str {
            "scripted"
        }

        fn detect(
            &mut self,
            camera: CameraId,
            _frame: &DetectorFrame<'_>,
        ) -> Result<Vec<Detection>, DetectorError> {
            Ok(match camera {
                CameraId::Left => self.left.clone(),
                CameraId::Right => self.right.clone(),
            })
        }
    }

    fn mapped(d: &Detection) -> reco_core::detect::director::MappedDetection {
        reco_core::detect::director::MappedDetection {
            camera: d.camera,
            class_id: d.class_id,
            confidence: d.confidence,
            camera_center: (d.center_x, d.center_y),
            camera_size: (d.width, d.height),
            position: None,
        }
    }

    /// `filter_mapped_field_mode` on unfiltered recorded detections must
    /// equal what the live `RoiFilteredDetector` (field-mode config) would
    /// have passed to the tracker, in the same order.
    #[test]
    fn mapped_field_mode_matches_live_decorator() {
        let roi = FieldRoi {
            left: vec![[0.1, 0.45], [0.95, 0.38], [1.0, 1.0], [0.1, 1.0]],
            right: vec![[0.0, 0.40], [0.8, 0.48], [0.75, 1.0], [0.0, 1.0]],
        };
        let mut grid = Vec::new();
        for (i, cls) in [0u16, 32, 0, 5, 32, 0, 32].iter().enumerate() {
            for k in 0..9 {
                let cx = 0.05 + 0.11 * k as f32;
                let cy = 0.1 + 0.12 * i as f32;
                grid.push((*cls, cx, cy, 0.02 + 0.01 * k as f32, 0.05 + 0.02 * i as f32));
            }
        }
        let side = |cam: CameraId| -> Vec<Detection> {
            grid.iter()
                .map(|&(c, x, y, w, h)| make_detection_class(cam, c, x, y, w, h))
                .collect()
        };
        let (left, right) = (side(CameraId::Left), side(CameraId::Right));
        let mut live = RoiFilteredDetector::new(
            Box::new(Scripted {
                left: left.clone(),
                right: right.clone(),
            }),
            roi.clone(),
        )
        .with_class_anchor(0, RoiAnchor::Bottom)
        .with_ball_roi_diagnostics(32, 0.40, 60.0, 1);
        let frame = DetectorFrame::PreprocessedChw {
            data: &[],
            input_size: 0,
            src_width: 0,
            src_height: 0,
        };
        let mut want: Vec<_> = live
            .detect(CameraId::Left, &frame)
            .unwrap()
            .iter()
            .map(mapped)
            .collect();
        want.extend(
            live.detect(CameraId::Right, &frame)
                .unwrap()
                .iter()
                .map(mapped),
        );
        let recorded: Vec<_> = left.iter().chain(&right).map(mapped).collect();
        let got = filter_mapped_field_mode(&recorded, &roi, 0, 32, 0.40);
        assert!(
            want.len() > 10 && want.len() < recorded.len(),
            "test must keep some and drop some"
        );
        assert_eq!(format!("{got:?}"), format!("{want:?}"));
    }

    #[test]
    fn detection_inside_roi_passes() {
        let roi = full_roi();
        let det = make_detection(CameraId::Left, 0.5, 0.4, 0.1, 0.2);
        assert_eq!(filter(vec![det], &roi).len(), 1);
    }

    #[test]
    fn detection_outside_roi_filtered() {
        let roi = small_roi();
        let det = make_detection(CameraId::Left, 0.05, 0.05, 0.1, 0.2);
        assert!(filter(vec![det], &roi).is_empty());
    }

    #[test]
    fn empty_polygon_passes_all() {
        let roi = FieldRoi {
            left: vec![],
            right: vec![],
        };
        let det = make_detection(CameraId::Left, 0.5, 0.5, 0.1, 0.2);
        assert_eq!(filter(vec![det], &roi).len(), 1);
    }

    #[test]
    fn degenerate_polygon_passes_all() {
        let roi = FieldRoi {
            left: vec![[0.0, 0.0], [1.0, 1.0]],
            right: vec![],
        };
        let det = make_detection(CameraId::Left, 0.5, 0.5, 0.1, 0.2);
        assert_eq!(filter(vec![det], &roi).len(), 1);
    }

    #[test]
    fn camera_id_selects_correct_polygon() {
        let roi = FieldRoi {
            left: vec![[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]],
            right: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        };
        let det_left = make_detection(CameraId::Left, 0.05, 0.05, 0.1, 0.2);
        let det_right = make_detection(CameraId::Right, 0.05, 0.05, 0.1, 0.2);
        assert!(filter(vec![det_left], &roi).is_empty());
        assert_eq!(filter(vec![det_right], &roi).len(), 1);
    }

    #[test]
    fn mixed_detections_filter_correctly() {
        let roi = small_roi();
        let inside = make_detection(CameraId::Left, 0.5, 0.4, 0.1, 0.2);
        let outside = make_detection(CameraId::Left, 0.05, 0.05, 0.1, 0.2);
        let filtered = filter(vec![inside, outside], &roi);
        assert_eq!(filtered.len(), 1);
        assert!((filtered[0].center_x - 0.5).abs() < f32::EPSILON);
    }

    // ── Step 7c: per-class anchor policy ─────────────────────────

    #[test]
    fn center_anchor_passes_ball_whose_feet_fall_outside() {
        // Small ROI [0.2, 0.8]^2. Detection centered in-bounds at
        // (0.5, 0.7), height 0.3 -> feet at y=0.85 outside, center
        // at y=0.7 inside. A ball class with Center anchor passes.
        let roi = small_roi();
        let ball = make_detection_class(CameraId::Left, 0, 0.5, 0.7, 0.1, 0.3);
        assert_eq!(filter(vec![ball], &roi).len(), 1);
    }

    #[test]
    fn bottom_anchor_rejects_player_whose_feet_fall_outside() {
        // Same geometry as above but as a Player (class=1) with
        // Bottom anchor: feet at y=0.85 outside -> rejected.
        let roi = small_roi();
        let player = make_detection_class(CameraId::Left, 1, 0.5, 0.7, 0.1, 0.3);
        let mut anchors = HashMap::new();
        anchors.insert(1u16, RoiAnchor::Bottom);
        let filtered = filter_by_roi(vec![player], &roi, &anchors, RoiAnchor::Center);
        assert!(filtered.is_empty());
    }

    #[test]
    fn mixed_classes_respect_per_class_policy() {
        // Same frame: a ball and a player at identical bboxes. With
        // Center default + Bottom for players, ball survives and
        // player gets dropped.
        let roi = small_roi();
        let ball = make_detection_class(CameraId::Left, 0, 0.5, 0.7, 0.1, 0.3);
        let player = make_detection_class(CameraId::Left, 1, 0.5, 0.7, 0.1, 0.3);
        let mut anchors = HashMap::new();
        anchors.insert(1u16, RoiAnchor::Bottom);
        let filtered = filter_by_roi(vec![ball, player], &roi, &anchors, RoiAnchor::Center);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].class_id, 0, "only the ball should survive");
    }

    #[test]
    fn with_class_anchor_builder_stores_mapping() {
        // Locks the public-API shape: builder method + lookup via
        // detect() path work together.
        let roi = small_roi();
        let det = make_detection_class(CameraId::Left, 7, 0.5, 0.7, 0.1, 0.3);
        let mut anchors = HashMap::new();
        anchors.insert(7u16, RoiAnchor::Bottom);
        assert!(filter_by_roi(vec![det], &roi, &anchors, RoiAnchor::Center).is_empty());
    }

    // ── ball vertical margin: allowance is above the polygon, not a
    //    downward shift that rejects near-side balls ──────────────────

    #[test]
    fn ball_margin_keeps_near_side_ball_inside_polygon() {
        // Polygon spans y in [0.4, 1.0]. A ball at y=0.7 is inside; the
        // old `cy + 0.40` test put it at 1.1 (outside) and dropped it.
        let poly = vec![[0.0, 0.4], [1.0, 0.4], [1.0, 1.0], [0.0, 1.0]];
        assert!(ball_passes_roi(0.5, 0.7, 0.40, &poly));
        assert!(!point_in_polygon([0.5, 0.7 + 0.40], &poly));
    }

    #[test]
    fn ball_margin_still_accepts_lofted_ball_above_top_edge() {
        let poly = vec![[0.0, 0.4], [1.0, 0.4], [1.0, 1.0], [0.0, 1.0]];
        // 0.25 above the top edge: within the 0.40 allowance.
        assert!(ball_passes_roi(0.5, 0.15, 0.40, &poly));
        // 0.45 above: beyond it.
        assert!(!ball_passes_roi(0.5, -0.05, 0.40, &poly));
    }

    #[test]
    fn ball_margin_rejects_ball_outside_sideways_or_below() {
        let poly = vec![[0.2, 0.4], [0.8, 0.4], [0.8, 0.9], [0.2, 0.9]];
        assert!(!ball_passes_roi(0.05, 0.6, 0.40, &poly)); // left of polygon
        assert!(!ball_passes_roi(0.5, 0.95, 0.40, &poly)); // below bottom edge
    }

    #[test]
    fn ball_margin_is_superset_of_old_test() {
        let poly = vec![[0.1, 0.45], [0.9, 0.38], [1.0, 1.0], [0.1, 1.0]];
        for xi in 0..=20 {
            for yi in 0..=20 {
                let (x, y) = (xi as f64 / 20.0, yi as f64 / 20.0);
                if point_in_polygon([x, y + 0.40], &poly) {
                    assert!(ball_passes_roi(x, y, 0.40, &poly), "lost ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn ball_margin_zero_equals_plain_center_test() {
        let poly = vec![[0.2, 0.4], [0.8, 0.4], [0.8, 0.9], [0.2, 0.9]];
        assert!(ball_passes_roi(0.5, 0.5, 0.0, &poly));
        assert!(!ball_passes_roi(0.5, 0.2, 0.0, &poly));
    }
}
