//! Persistent player tracker with sticky target-pitch ownership (OEV-006).
//!
//! Experimental, opt-in; production field mode still uses the stateless
//! [`ClassProvider`](super::ClassProvider). Used by the OEV-006 B1/B2/A1/C
//! arms through the CPU director replay.
//!
//! Tracking is ByteTrack-style in panorama space:
//!
//! 1. Cross-camera de-duplication: person detections closer than
//!    `dedup_rad` (the seam overlap sees the same player twice) collapse to
//!    the most confident one.
//! 2. Each track predicts its next position with a constant-velocity
//!    alpha-beta filter.
//! 3. High-confidence detections are matched first, low-confidence ones
//!    (`low_conf..high_conf`) only to tracks left unmatched, greedily by
//!    distance inside a size-scaled gate.
//! 4. Tracks are confirmed after `confirm_hits` matches and dropped after
//!    `max_missed` frames without one; unmatched tracks coast on their
//!    velocity.
//!
//! Ownership: a track's foot point is tested against the venue field ROI
//! (the same per-camera polygon and `Bottom` anchor production uses). A
//! track becomes *owned* after `own_after_s` of accumulated inside time
//! and is only disowned after `disown_after_s` of continuous outside time,
//! so a player stepping over the line keeps ownership while a neighbouring
//! match's players never gain it.

use reco_core::calibration::FieldRoi;
use reco_core::detect::detector::CameraId;
use reco_core::detect::director::MappedDetection;
use reco_core::detect::tracker::{TrackState, TrackedEntity};
use reco_core::projection::point_in_polygon;

/// Tuning for [`PlayerTracker`].
#[derive(Debug, Clone)]
pub struct PlayerTrackerConfig {
    /// Detection class to track.
    pub class_id: u16,
    /// Detections at or above this confidence are matched in stage 1.
    pub high_conf: f32,
    /// Detections in `[low_conf, high_conf)` are only used in stage 2.
    pub low_conf: f32,
    /// Radius (rad) below which two detections are the same player.
    pub dedup_rad: f32,
    /// Approximate vertical field of view of one camera (rad), used to turn
    /// a box's normalized height into an angular size for gating.
    pub camera_vfov_rad: f32,
    /// Gate = clamp(gate_scale * angular_height, gate_min, gate_max).
    pub gate_scale: f32,
    /// Lower bound of the association gate (rad).
    pub gate_min: f32,
    /// Upper bound of the association gate (rad).
    pub gate_max: f32,
    /// Matches needed before a track is reported.
    pub confirm_hits: u32,
    /// Frames without a match before a track is dropped.
    pub max_missed: u32,
    /// Position smoothing factor of the alpha-beta filter.
    pub alpha: f32,
    /// Velocity smoothing factor of the alpha-beta filter.
    pub beta: f32,
    /// Accumulated inside-ROI time needed to own a track (s).
    pub own_after_s: f32,
    /// Continuous outside-ROI time needed to disown a track (s).
    pub disown_after_s: f32,
    /// Frame rate (frames per second) used to convert times to frames.
    pub fps: f32,
}

impl PlayerTrackerConfig {
    /// Defaults for 4K GoPro Wide panoramas at about 60 fps.
    pub fn oev_default(class_id: u16, fps: f32) -> Self {
        Self {
            class_id,
            high_conf: 0.5,
            low_conf: 0.15,
            dedup_rad: 0.02,
            camera_vfov_rad: 1.2,
            gate_scale: 0.6,
            gate_min: 0.015,
            gate_max: 0.08,
            confirm_hits: 3,
            max_missed: (fps * 1.0) as u32,
            alpha: 0.6,
            beta: 0.1,
            own_after_s: 1.5,
            disown_after_s: 5.0,
            fps,
        }
    }
}

#[derive(Debug, Clone)]
struct Track {
    id: u64,
    yaw: f32,
    pitch: f32,
    vyaw: f32,
    vpitch: f32,
    hits: u32,
    missed: u32,
    age: u64,
    confidence: f32,
    camera: CameraId,
    /// Last matched camera-space box (center, size) for the ROI test.
    last_box: Option<((f32, f32), (f32, f32))>,
    inside_frames: u32,
    outside_run: u32,
    owned: bool,
}

/// One tracked player as reported by [`PlayerTracker::update`].
#[derive(Debug, Clone, Copy)]
pub struct PlayerTrack {
    /// The entity in [`WorldState`](reco_core::detect::tracker::WorldState) form.
    pub entity: TrackedEntity,
    /// Whether the track currently owns the target pitch.
    pub owned: bool,
    /// Velocity estimate (rad/frame) in yaw and pitch.
    pub velocity: (f32, f32),
}

/// Stateful multi-player tracker; see the module docs.
#[derive(Debug, Clone)]
pub struct PlayerTracker {
    cfg: PlayerTrackerConfig,
    roi: Option<FieldRoi>,
    tracks: Vec<Track>,
    next_id: u64,
}

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

impl PlayerTracker {
    /// Build a tracker; `roi` enables ownership (without it every confirmed
    /// track is reported as owned).
    pub fn new(cfg: PlayerTrackerConfig, roi: Option<FieldRoi>) -> Self {
        Self {
            cfg,
            roi,
            tracks: Vec::new(),
            next_id: 1,
        }
    }

    fn gate(&self, d: &MappedDetection) -> f32 {
        (self.cfg.gate_scale * d.camera_size.1 * self.cfg.camera_vfov_rad)
            .clamp(self.cfg.gate_min, self.cfg.gate_max)
    }

    fn feet_inside(&self, camera: CameraId, center: (f32, f32), size: (f32, f32)) -> Option<bool> {
        let roi = self.roi.as_ref()?;
        let poly: &[[f64; 2]] = match camera {
            CameraId::Left => &roi.left,
            CameraId::Right => &roi.right,
        };
        if poly.len() < 3 {
            return Some(true);
        }
        let (cx, cy) = (center.0 as f64, center.1 as f64);
        let h = size.1 as f64;
        Some(
            point_in_polygon([cx, cy + h * 0.5], poly)
                && point_in_polygon([cx, cy + h * 0.25], poly),
        )
    }

    /// Feed one frame of detections; returns every confirmed track.
    pub fn update(&mut self, detections: &[MappedDetection]) -> Vec<PlayerTrack> {
        // 1. Candidates in panorama space, de-duplicated across cameras.
        let mut cands: Vec<&MappedDetection> = detections
            .iter()
            .filter(|d| d.class_id == self.cfg.class_id && d.position.is_some())
            .filter(|d| d.confidence >= self.cfg.low_conf)
            .collect();
        cands.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        let mut kept: Vec<&MappedDetection> = Vec::with_capacity(cands.len());
        for d in cands {
            let p = d.position.expect("filtered");
            if kept.iter().all(|k| {
                let q = k.position.expect("filtered");
                dist((p.yaw, p.pitch), (q.yaw, q.pitch)) > self.cfg.dedup_rad
            }) {
                kept.push(d);
            }
        }

        // 2. Predict.
        for t in &mut self.tracks {
            t.yaw += t.vyaw;
            t.pitch += t.vpitch;
            t.age += 1;
        }

        // 3. Two-stage greedy association.
        let mut det_used = vec![false; kept.len()];
        let mut trk_used = vec![false; self.tracks.len()];
        for stage in 0..2 {
            let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
            for (di, d) in kept.iter().enumerate() {
                if det_used[di] {
                    continue;
                }
                let high = d.confidence >= self.cfg.high_conf;
                if (stage == 0) != high {
                    continue;
                }
                let p = d.position.expect("filtered");
                let g = self.gate(d);
                for (ti, t) in self.tracks.iter().enumerate() {
                    if trk_used[ti] {
                        continue;
                    }
                    let c = dist((p.yaw, p.pitch), (t.yaw, t.pitch));
                    if c <= g {
                        pairs.push((c, di, ti));
                    }
                }
            }
            pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
            for (_, di, ti) in pairs {
                if det_used[di] || trk_used[ti] {
                    continue;
                }
                det_used[di] = true;
                trk_used[ti] = true;
                let d = kept[di];
                let p = d.position.expect("filtered");
                let t = &mut self.tracks[ti];
                let (ry, rp) = (p.yaw - t.yaw, p.pitch - t.pitch);
                t.yaw += self.cfg.alpha * ry;
                t.pitch += self.cfg.alpha * rp;
                t.vyaw += self.cfg.beta * ry;
                t.vpitch += self.cfg.beta * rp;
                t.hits += 1;
                t.missed = 0;
                t.confidence = d.confidence;
                t.camera = d.camera;
                t.last_box = Some((d.camera_center, d.camera_size));
            }
        }

        // 4. Unmatched tracks coast; stale ones are dropped.
        for (ti, t) in self.tracks.iter_mut().enumerate() {
            if !trk_used[ti] {
                t.missed += 1;
                t.vyaw *= 0.9;
                t.vpitch *= 0.9;
            }
        }
        let max_missed = self.cfg.max_missed;
        self.tracks.retain(|t| t.missed <= max_missed);

        // 5. New tracks from unmatched high-confidence detections.
        for (di, d) in kept.iter().enumerate() {
            if det_used[di] || d.confidence < self.cfg.high_conf {
                continue;
            }
            let p = d.position.expect("filtered");
            self.tracks.push(Track {
                id: self.next_id,
                yaw: p.yaw,
                pitch: p.pitch,
                vyaw: 0.0,
                vpitch: 0.0,
                hits: 1,
                missed: 0,
                age: 0,
                confidence: d.confidence,
                camera: d.camera,
                last_box: Some((d.camera_center, d.camera_size)),
                inside_frames: 0,
                outside_run: 0,
                owned: false,
            });
            self.next_id += 1;
        }

        // 6. Ownership (only measured on frames with a fresh match).
        let own_frames = (self.cfg.own_after_s * self.cfg.fps).round() as u32;
        let disown_frames = (self.cfg.disown_after_s * self.cfg.fps).round() as u32;
        let has_roi = self.roi.is_some();
        for i in 0..self.tracks.len() {
            let t = &self.tracks[i];
            if t.missed > 0 {
                continue;
            }
            let inside = match t.last_box {
                Some((c, s)) => self.feet_inside(t.camera, c, s),
                None => None,
            };
            let t = &mut self.tracks[i];
            match inside {
                None => t.owned = !has_roi || t.owned,
                Some(true) => {
                    t.inside_frames += 1;
                    t.outside_run = 0;
                    if t.inside_frames >= own_frames {
                        t.owned = true;
                    }
                }
                Some(false) => {
                    t.outside_run += 1;
                    if t.outside_run >= disown_frames {
                        t.owned = false;
                        t.inside_frames = 0;
                    }
                }
            }
        }
        if !has_roi {
            for t in &mut self.tracks {
                t.owned = true;
            }
        }

        self.tracks
            .iter()
            .filter(|t| t.hits >= self.cfg.confirm_hits)
            .map(|t| PlayerTrack {
                entity: TrackedEntity {
                    id: t.id,
                    class_id: self.cfg.class_id,
                    yaw: t.yaw,
                    pitch: t.pitch,
                    confidence: if t.missed == 0 { t.confidence } else { 0.0 },
                    state: if t.missed == 0 {
                        TrackState::Tracking
                    } else {
                        TrackState::Coasting
                    },
                    age_frames: t.age,
                    origin: t.camera,
                },
                owned: t.owned,
                velocity: (t.vyaw, t.vpitch),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reco_core::detect::director::ViewportPosition;

    fn det(cam: CameraId, yaw: f32, pitch: f32, conf: f32, cy: f32) -> MappedDetection {
        MappedDetection {
            camera: cam,
            class_id: 0,
            confidence: conf,
            camera_center: (0.5, cy),
            camera_size: (0.02, 0.08),
            position: Some(ViewportPosition {
                yaw,
                pitch,
                fov_degrees: None,
            }),
        }
    }

    fn cfg() -> PlayerTrackerConfig {
        PlayerTrackerConfig::oev_default(0, 60.0)
    }

    #[test]
    fn keeps_ids_through_motion_and_crossing() {
        let mut t = PlayerTracker::new(cfg(), None);
        let mut ids_a = Vec::new();
        let mut ids_b = Vec::new();
        for f in 0..120 {
            let x = -0.2 + 0.004 * f as f32;
            let out = t.update(&[
                det(CameraId::Left, x, 0.0, 0.9, 0.6),
                det(CameraId::Left, 0.2 - 0.004 * f as f32, 0.03, 0.9, 0.6),
            ]);
            if f >= 5 {
                let a = out
                    .iter()
                    .min_by(|p, q| {
                        (p.entity.yaw - x)
                            .abs()
                            .total_cmp(&(q.entity.yaw - x).abs())
                    })
                    .unwrap();
                ids_a.push(a.entity.id);
                let b = out.iter().find(|p| p.entity.id != a.entity.id).unwrap();
                ids_b.push(b.entity.id);
            }
        }
        // Players separated in pitch keep their identity through the crossing.
        assert!(ids_a.iter().all(|&i| i == ids_a[0]), "{ids_a:?}");
        assert!(ids_b.iter().all(|&i| i == ids_b[0]));
        assert_ne!(ids_a[0], ids_b[0]);
    }

    #[test]
    fn dedups_seam_double_detection_and_coasts_short_gaps() {
        let mut t = PlayerTracker::new(cfg(), None);
        let mut id = None;
        for f in 0..90 {
            let mut dets = vec![
                det(CameraId::Left, 0.0, 0.0, 0.9, 0.6),
                det(CameraId::Right, 0.005, 0.0, 0.8, 0.6),
            ];
            if (40..50).contains(&f) {
                dets.clear(); // 10-frame occlusion
            }
            let out = t.update(&dets);
            if f > 3 {
                assert_eq!(out.len(), 1, "frame {f}: {out:?}");
                let this = out[0].entity.id;
                assert_eq!(*id.get_or_insert(this), this, "id changed at {f}");
            }
        }
    }

    #[test]
    fn ownership_is_sticky_and_neighbours_never_own() {
        let roi = FieldRoi {
            left: vec![[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]],
            right: vec![[0.0, 0.5], [1.0, 0.5], [1.0, 1.0], [0.0, 1.0]],
        };
        let mut t = PlayerTracker::new(cfg(), Some(roi));
        let mut owned_ours = Vec::new();
        let mut owned_neighbour = Vec::new();
        for f in 0..600 {
            // Ours: inside, then steps out (feet above the line) for 2 s.
            let ours_cy = if (200..320).contains(&f) { 0.3 } else { 0.7 };
            let out = t.update(&[
                det(CameraId::Left, -0.3, 0.0, 0.9, ours_cy),
                det(CameraId::Left, 0.5, 0.1, 0.9, 0.2),
            ]);
            for p in &out {
                if p.entity.yaw < 0.0 {
                    owned_ours.push((f, p.owned));
                } else {
                    owned_neighbour.push(p.owned);
                }
            }
        }
        assert!(
            owned_ours.iter().filter(|(f, _)| *f > 100).all(|(_, o)| *o),
            "ours lost ownership while briefly outside"
        );
        assert!(
            owned_ours.iter().any(|(f, o)| *f < 80 && !*o),
            "ownership should need dwell time"
        );
        assert!(
            owned_neighbour.iter().all(|o| !o),
            "neighbour must never own"
        );
    }
}
