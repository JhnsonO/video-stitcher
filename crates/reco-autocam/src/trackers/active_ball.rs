//! Active-ball selection weights (OEV-004 steps 1-2, experiment).
//!
//! Two independent soft signals that down-weight ball candidates which are
//! very likely not the match ball:
//!
//! 1. **Body attachment.** A candidate whose centre sits inside a
//!    same-camera person box, in the body zone (not the feet), and keeps a
//!    near-constant offset inside that box for at least
//!    [`ActiveBallConfig::attach_min_ms`] while the person moves, is treated as
//!    part of the body (knee patch, shorts logo, hand). Any change in offset
//!    beyond the band breaks the attachment immediately, so a ball that is
//!    kicked, dropped or dribbled away leaves this state on its first
//!    separating frame. The feet zone is excluded so dribbles are never
//!    attached.
//! 2. **Size prior.** Expected ball box size against image row per camera
//!    (fixed camera, `log(size) = slope * cy + intercept`). Candidates much
//!    smaller than expected are down-weighted. Larger-than-expected boxes are
//!    not penalised (a ball in flight sits higher in the image than its ground
//!    point and looks large for its row).
//!
//! Attachment is inferred from positions: detections carry no identities.
//! The only hard reject is near-certain attachment (attached and tiny, or
//! attached for [`ActiveBallConfig::attach_reject_ms`]). Everything else is a
//! weight in `(0, 1]` that the tracker folds into its existing ranking.
//! With no person detections and no size prior, every weight is 1.0 and the
//! tracker's behaviour is unchanged.

use reco_core::detect::detector::CameraId;
use reco_core::detect::director::MappedDetection;

/// Per-camera expected-size model: `ln(sqrt(w*h)) = slope * cy + intercept`,
/// with residual standard deviation `sd` (normalised camera coordinates).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SizeModel {
    /// Slope of log size against normalised image row.
    pub slope: f32,
    /// Intercept of log size.
    pub intercept: f32,
    /// Residual standard deviation of log size.
    pub sd: f32,
}

impl SizeModel {
    /// Standardised residual of a box against the model (negative = smaller
    /// than expected).
    pub fn z(&self, cy: f32, size: (f32, f32)) -> Option<f32> {
        let s = (size.0.max(0.0) * size.1.max(0.0)).sqrt();
        if s <= 0.0 || self.sd <= 0.0 {
            return None;
        }
        Some((s.ln() - (self.slope * cy + self.intercept)) / self.sd)
    }
}

/// Size prior for a two-camera rig.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SizePrior {
    /// Model for the left camera.
    pub left: SizeModel,
    /// Model for the right camera.
    pub right: SizeModel,
}

impl SizePrior {
    /// Fit from OEV `sample_02` run 39039272579 (St Margaret's rig): confident
    /// (>= 0.5) ball detections outside every person box. Site-specific: one
    /// clip, one pitch. Refit before trusting on other rigs.
    pub const SAMPLE02_FIT: SizePrior = SizePrior {
        left: SizeModel {
            slope: 2.904,
            intercept: -6.295,
            sd: 0.196,
        },
        right: SizeModel {
            slope: 3.891,
            intercept: -6.732,
            sd: 0.158,
        },
    };

    fn model(&self, cam: CameraId) -> &SizeModel {
        match cam {
            CameraId::Left => &self.left,
            CameraId::Right => &self.right,
        }
    }
}

/// Tunables for active-ball selection. Defaults are review values, not
/// production constants.
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveBallConfig {
    /// Detection class of persons (COCO 0).
    pub person_class_id: u16,
    /// Body zone in relative box height (0 = head, 1 = feet). Candidates
    /// outside it are never attached.
    pub body_zone: (f32, f32),
    /// Max spread of relative v while attached.
    pub band_v: f32,
    /// Max spread of relative u while attached.
    pub band_u: f32,
    /// Max gap between observations of one attachment track.
    pub max_gap_ms: f64,
    /// Minimum stable duration before a candidate counts as attached.
    pub attach_min_ms: f64,
    /// Attached for this long: hard reject.
    pub attach_reject_ms: f64,
    /// Minimum person movement, in person heights, while attached.
    pub min_person_move: f32,
    /// Weight of an attached candidate (before hard-reject rules).
    pub attached_weight: f32,
    /// Size prior; `None` disables step 2.
    pub size_prior: Option<SizePrior>,
    /// Size residual (z) at which the penalty starts.
    pub size_z_soft: f32,
    /// Size residual at or below which an attached candidate is hard-rejected.
    pub size_z_reject_attached: f32,
    /// Floor of the size weight.
    pub size_min_weight: f32,
}

impl ActiveBallConfig {
    /// Review defaults with the `sample_02` size fit.
    pub fn sample02(person_class_id: u16) -> Self {
        Self {
            person_class_id,
            body_zone: (0.12, 0.88),
            band_v: 0.10,
            band_u: 0.40,
            max_gap_ms: 250.0,
            attach_min_ms: 300.0,
            attach_reject_ms: 1000.0,
            min_person_move: 0.10,
            attached_weight: 0.15,
            size_prior: Some(SizePrior::SAMPLE02_FIT),
            size_z_soft: -2.0,
            size_z_reject_attached: -2.5,
            size_min_weight: 0.10,
        }
    }
}

/// Verdict for one candidate this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Verdict {
    /// Candidate keeps the given weight in `(0, 1]`.
    Weight(f32),
    /// Near-certain body attachment: drop the candidate.
    Reject,
}

#[derive(Debug, Clone)]
struct AttachTrack {
    cam: CameraId,
    u_min: f32,
    u_max: f32,
    v_min: f32,
    v_max: f32,
    start_person: (f32, f32),
    last_person: (f32, f32),
    person_h: f32,
    max_move: f32,
    t_start: f64,
    t_last: f64,
}

/// Stateful scorer; call [`observe`](Self::observe) once per frame before
/// [`verdict`](Self::verdict) for that frame's candidates.
#[derive(Debug, Clone)]
pub struct ActiveBallSelector {
    cfg: ActiveBallConfig,
    tracks: Vec<AttachTrack>,
    /// Per candidate index of the last observed frame: track index if the
    /// candidate is in the body zone.
    frame_links: Vec<Option<usize>>,
    frame_ts: f64,
}

struct Containment {
    u: f32,
    v: f32,
    person: (f32, f32),
    person_h: f32,
}

fn containing_person(
    det: &MappedDetection,
    detections: &[MappedDetection],
    person_class_id: u16,
) -> Option<Containment> {
    let (cx, cy) = det.camera_center;
    detections
        .iter()
        .filter(|p| p.class_id == person_class_id && p.camera == det.camera)
        .filter(|p| {
            let (px, py) = p.camera_center;
            let (pw, ph) = p.camera_size;
            pw > 0.0 && ph > 0.0 && (cx - px).abs() <= pw / 2.0 && (cy - py).abs() <= ph / 2.0
        })
        // Smallest containing box: the nearest body when boxes overlap.
        .min_by(|a, b| {
            a.camera_size
                .1
                .partial_cmp(&b.camera_size.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|p| {
            let (px, py) = p.camera_center;
            let (pw, ph) = p.camera_size;
            Containment {
                u: (cx - px) / pw,
                v: (cy - (py - ph / 2.0)) / ph,
                person: (px, py),
                person_h: ph,
            }
        })
}

impl ActiveBallSelector {
    /// New selector with the given configuration.
    pub fn new(cfg: ActiveBallConfig) -> Self {
        Self {
            cfg,
            tracks: Vec::new(),
            frame_links: Vec::new(),
            frame_ts: f64::NEG_INFINITY,
        }
    }

    /// Configuration in use.
    pub fn config(&self) -> &ActiveBallConfig {
        &self.cfg
    }

    /// Update attachment tracks from this frame's detections (all classes).
    /// `ball_class_id` selects the candidates.
    pub fn observe(&mut self, detections: &[MappedDetection], ball_class_id: u16, ts_ms: f64) {
        self.frame_ts = ts_ms;
        self.frame_links.clear();
        self.frame_links.resize(detections.len(), None);
        let max_gap = self.cfg.max_gap_ms;
        self.tracks.retain(|t| ts_ms - t.t_last <= max_gap);

        let mut claimed = vec![false; self.tracks.len()];
        let mut new_tracks = Vec::new();
        for (i, det) in detections.iter().enumerate() {
            if det.class_id != ball_class_id || det.position.is_none() {
                continue;
            }
            let Some(c) = containing_person(det, detections, self.cfg.person_class_id) else {
                continue;
            };
            if c.v <= self.cfg.body_zone.0 || c.v >= self.cfg.body_zone.1 {
                continue;
            }
            // Continue the closest compatible track: same camera, same body
            // (person centre moved less than one person height... scaled by the
            // gap), offset still inside the band.
            // Prefer the oldest compatible track so a duplicate detection of
            // the same feature cannot restart its attachment clock.
            let mut best: Option<(usize, f64, f32)> = None;
            for (ti, t) in self.tracks.iter().enumerate() {
                if claimed[ti] || t.cam != det.camera || t.t_last >= ts_ms {
                    continue;
                }
                let dpx = c.person.0 - t.last_person.0;
                let dpy = c.person.1 - t.last_person.1;
                let pd = (dpx * dpx + dpy * dpy).sqrt() / t.person_h.max(1e-6);
                if pd > 0.5 {
                    continue;
                }
                let u_spread = t.u_max.max(c.u) - t.u_min.min(c.u);
                let v_spread = t.v_max.max(c.v) - t.v_min.min(c.v);
                if u_spread > self.cfg.band_u || v_spread > self.cfg.band_v {
                    continue;
                }
                let cost = pd + v_spread;
                let better = best.is_none_or(|(_, start, b)| {
                    t.t_start < start || (t.t_start == start && cost < b)
                });
                if better {
                    best = Some((ti, t.t_start, cost));
                }
            }
            match best {
                Some((ti, _, _)) => {
                    claimed[ti] = true;
                    let t = &mut self.tracks[ti];
                    t.u_min = t.u_min.min(c.u);
                    t.u_max = t.u_max.max(c.u);
                    t.v_min = t.v_min.min(c.v);
                    t.v_max = t.v_max.max(c.v);
                    t.last_person = c.person;
                    t.person_h = c.person_h;
                    let dx = c.person.0 - t.start_person.0;
                    let dy = c.person.1 - t.start_person.1;
                    t.max_move = t
                        .max_move
                        .max((dx * dx + dy * dy).sqrt() / c.person_h.max(1e-6));
                    t.t_last = ts_ms;
                    self.frame_links[i] = Some(ti);
                }
                None => {
                    new_tracks.push((
                        i,
                        AttachTrack {
                            cam: det.camera,
                            u_min: c.u,
                            u_max: c.u,
                            v_min: c.v,
                            v_max: c.v,
                            start_person: c.person,
                            last_person: c.person,
                            person_h: c.person_h,
                            max_move: 0.0,
                            t_start: ts_ms,
                            t_last: ts_ms,
                        },
                    ));
                }
            }
        }
        // Tracks not continued this frame by an in-band candidate end now: the
        // candidate either left the body zone, separated, or vanished. A gap
        // without any body-zone candidate keeps the track (detector misses).
        for (i, t) in new_tracks {
            self.frame_links[i] = Some(self.tracks.len());
            self.tracks.push(t);
        }
    }

    /// Verdict for `detections[index]`, valid for the frame passed to the most
    /// recent [`observe`](Self::observe).
    pub fn verdict(&self, detections: &[MappedDetection], index: usize) -> Verdict {
        let det = &detections[index];
        let z = self
            .cfg
            .size_prior
            .as_ref()
            .and_then(|p| p.model(det.camera).z(det.camera_center.1, det.camera_size));

        // Stable offset inside one body for at least `attach_min_ms`.
        let stable = self
            .frame_links
            .get(index)
            .copied()
            .flatten()
            .and_then(|ti| self.tracks.get(ti))
            .map(|t| (t.t_last - t.t_start, t.max_move))
            .filter(|&(ms, _)| ms >= self.cfg.attach_min_ms);
        // Stable and tiny for its image row: near-certainly part of the body,
        // whether or not the person moves.
        if stable.is_some() && z.is_some_and(|z| z <= self.cfg.size_z_reject_attached) {
            return Verdict::Reject;
        }
        // Stable while the person moves: attached (soft), hard reject once long.
        let attached_ms = stable
            .filter(|&(_, mv)| mv >= self.cfg.min_person_move)
            .map(|(ms, _)| ms);

        if attached_ms.is_some_and(|ms| ms >= self.cfg.attach_reject_ms) {
            return Verdict::Reject;
        }

        let mut w = 1.0f32;
        if attached_ms.is_some() {
            w *= self.cfg.attached_weight;
        }
        if let Some(z) = z.filter(|&z| z < self.cfg.size_z_soft) {
            let d = z - self.cfg.size_z_soft;
            w *= (-0.5 * d * d).exp().max(self.cfg.size_min_weight);
        }
        Verdict::Weight(w.clamp(f32::MIN_POSITIVE, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::detect::director::ViewportPosition;

    pub(crate) fn mk(class_id: u16, cx: f32, cy: f32, w: f32, h: f32) -> MappedDetection {
        MappedDetection {
            camera: CameraId::Left,
            class_id,
            confidence: 0.6,
            camera_center: (cx, cy),
            camera_size: (w, h),
            position: Some(ViewportPosition {
                yaw: cx,
                pitch: -cy,
                fov_degrees: None,
            }),
        }
    }

    fn cfg_no_size() -> ActiveBallConfig {
        ActiveBallConfig {
            size_prior: None,
            ..ActiveBallConfig::sample02(0)
        }
    }

    #[test]
    fn no_persons_is_noop() {
        let mut s = ActiveBallSelector::new(cfg_no_size());
        for i in 0..60 {
            let d = [mk(32, 0.5, 0.5, 0.01, 0.01)];
            s.observe(&d, 32, i as f64 * 16.7);
            assert_eq!(s.verdict(&d, 0), Verdict::Weight(1.0));
        }
    }

    #[test]
    fn knee_patch_on_moving_player_attaches_then_rejects() {
        let mut s = ActiveBallSelector::new(cfg_no_size());
        let mut last = Verdict::Weight(1.0);
        let mut first_attached = None;
        for i in 0..90 {
            let t = i as f64 * 16.7;
            let px = 0.3 + 0.002 * i as f32; // walking
            let d = [
                mk(0, px, 0.7, 0.05, 0.2),
                mk(32, px, 0.6 + 0.2 * 0.15, 0.005, 0.005),
            ];
            s.observe(&d, 32, t);
            last = s.verdict(&d, 1);
            if first_attached.is_none() && last != Verdict::Weight(1.0) {
                first_attached = Some(t);
            }
        }
        assert!(first_attached.unwrap() >= 300.0);
        assert_eq!(last, Verdict::Reject);
    }

    #[test]
    fn stationary_player_does_not_attach() {
        let mut s = ActiveBallSelector::new(cfg_no_size());
        for i in 0..90 {
            let d = [mk(0, 0.3, 0.7, 0.05, 0.2), mk(32, 0.3, 0.73, 0.005, 0.005)];
            s.observe(&d, 32, i as f64 * 16.7);
            assert_eq!(s.verdict(&d, 1), Verdict::Weight(1.0));
        }
    }

    #[test]
    fn feet_zone_never_attaches() {
        let mut s = ActiveBallSelector::new(cfg_no_size());
        for i in 0..120 {
            let px = 0.3 + 0.003 * i as f32;
            // ball at the feet moving with the player (dribble)
            let d = [mk(0, px, 0.7, 0.05, 0.2), mk(32, px, 0.795, 0.01, 0.01)];
            s.observe(&d, 32, i as f64 * 16.7);
            assert_eq!(s.verdict(&d, 1), Verdict::Weight(1.0));
        }
    }

    #[test]
    fn separation_breaks_attachment_immediately() {
        let mut s = ActiveBallSelector::new(cfg_no_size());
        let mut t = 0.0;
        for i in 0..40 {
            t = i as f64 * 16.7;
            let px = 0.3 + 0.002 * i as f32;
            let d = [mk(0, px, 0.7, 0.05, 0.2), mk(32, px, 0.63, 0.01, 0.01)];
            s.observe(&d, 32, t);
        }
        // Held at the chest for ~0.65 s: attached (soft), not rejected.
        let held = [
            mk(0, 0.378, 0.7, 0.05, 0.2),
            mk(32, 0.378, 0.63, 0.01, 0.01),
        ];
        assert!(matches!(s.verdict(&held, 1), Verdict::Weight(w) if w < 1.0));
        // Ball drops from the chest and rolls away: never rejected, and back to
        // full weight as soon as its offset leaves the band.
        for k in 1..20 {
            let tt = t + k as f64 * 16.7;
            let px = 0.38;
            let d = [
                mk(0, px, 0.7, 0.05, 0.2),
                mk(
                    32,
                    px + 0.004 * k as f32,
                    0.636 + 0.02 * k as f32,
                    0.01,
                    0.01,
                ),
            ];
            s.observe(&d, 32, tt);
            let v = s.verdict(&d, 1);
            assert_ne!(v, Verdict::Reject, "frame {k}");
            if k >= 6 {
                assert_eq!(v, Verdict::Weight(1.0), "frame {k}");
            }
        }
    }

    #[test]
    fn size_prior_penalises_only_small_boxes() {
        let m = SizePrior::SAMPLE02_FIT.left;
        let cy = 0.5;
        let expected = (m.slope * cy + m.intercept).exp();
        let mut s = ActiveBallSelector::new(ActiveBallConfig::sample02(0));
        let d = [
            mk(32, 0.2, cy, expected, expected),
            mk(32, 0.4, cy, expected * 0.3, expected * 0.3),
            mk(32, 0.6, cy, expected * 3.0, expected * 3.0),
        ];
        s.observe(&d, 32, 0.0);
        assert_eq!(s.verdict(&d, 0), Verdict::Weight(1.0));
        assert_eq!(s.verdict(&d, 2), Verdict::Weight(1.0));
        match s.verdict(&d, 1) {
            Verdict::Weight(w) => assert!(w < 0.2, "w={w}"),
            v => panic!("{v:?}"),
        }
    }
}
