//! Candidate-based dormant static-site exclusion for [`BallTracker`].
//!
//! A stationary spare football is detected near-continuously by YOLO while
//! the real ball is in play elsewhere. This module learns such locations
//! from **all** ball-class candidates (not only the ones the tracker
//! accepted), decides when a location is a dormant spare-ball site, and
//! decides when a dormant site may be released because its ball has
//! genuinely moved.
//!
//! All rules are expressed in video time and positions, never in samples or
//! per-sample velocity, so decisions are independent of frame rate and of
//! how often the detector runs.
//!
//! # Stages
//!
//! 1. **Static clusters.** Each candidate is associated with the nearest
//!    cluster within [`ASSOC_RAD`]. A cluster records a *static window* each
//!    time [`STATIC_WINDOW_MS`] of observations (at least
//!    [`STATIC_MIN_DETECTIONS`], no gap above [`OBS_GAP_MS`]) stay within a
//!    robust jitter radius around their medoid with no net drift.
//! 2. **Contradiction.** A window is contradicted when, during it, another
//!    ball candidate elsewhere (at least [`CONTRADICTION_MIN_RAD`] away) was
//!    *moving coherently* for at least [`MOVING_MIN_MS`]. A lone dead ball has
//!    no such evidence.
//! 3. **Dormancy.** At least [`DORMANT_MIN_WINDOWS`] static windows spanning
//!    at least [`DORMANT_EVIDENCE_SPAN_MS`], one contradicted, make the
//!    cluster dormant. A static cluster within [`FAMILY_RAD`] of an existing
//!    dormant site (a neighbouring detection of the same stationary ball)
//!    needs a single contradicted window.
//! 4. **Associated release.** A dormant site is released only when a moving
//!    tracklet that started at the site travels outward while the site itself
//!    stops being observed ("persistent outgoing motion"), or the player who
//!    stood at the site when the ball vanished carries a moving ball away
//!    ("pickup occlusion"). Unrelated motion elsewhere and bystanders never
//!    release a site.
//! 5. **Probation.** A released site keeps its evidence for
//!    [`PROBATION_MS`]; if it goes static again within that time it becomes
//!    dormant again immediately.

use reco_core::detect::director::MappedDetection;
use reco_core::detect::tracker::TrackedEntity;

const DEG_TO_RAD: f32 = std::f32::consts::PI / 180.0;
/// Minimum robust site radius (detection jitter floor).
pub(crate) const SITE_MIN_RADIUS_RAD: f32 = 0.35 * DEG_TO_RAD;
/// Maximum robust site radius; also the hard cap on any exclusion zone.
pub(crate) const SITE_MAX_RADIUS_RAD: f32 = 1.0 * DEG_TO_RAD;
/// Candidate-to-cluster association radius. Smaller than the gap between
/// distinct stationary objects observed on sample_02 (~16 mrad).
pub(crate) const ASSOC_RAD: f32 = 0.7 * DEG_TO_RAD;
/// Video time a cluster must stay put to record one static window.
pub(crate) const STATIC_WINDOW_MS: f64 = 1_500.0;
/// Minimum observations inside one static window.
pub(crate) const STATIC_MIN_DETECTIONS: usize = 6;
/// Largest observation gap tolerated inside a static window.
pub(crate) const OBS_GAP_MS: f64 = 750.0;
/// Static windows required (spanning `DORMANT_EVIDENCE_SPAN_MS`).
pub(crate) const DORMANT_MIN_WINDOWS: usize = 3;
/// Video time the static evidence must span before dormancy.
pub(crate) const DORMANT_EVIDENCE_SPAN_MS: f64 = 20_000.0;
/// Minimum distance for "play elsewhere".
pub(crate) const CONTRADICTION_MIN_RAD: f32 = 0.15;
/// A tracklet is moving once it spans this long and displaced `MOVE_MIN_RAD`.
pub(crate) const MOVING_MIN_MS: f64 = 750.0;
/// Net displacement that distinguishes motion from detection jitter.
pub(crate) const MOVE_MIN_RAD: f32 = 0.02;
/// Upper bound on plausible ball angular speed for tracklet continuity
/// (same 178 deg/s bound the backward bridge uses).
pub(crate) const MAX_BALL_SPEED_RAD_PER_MS: f32 = 178.0 * DEG_TO_RAD / 1000.0;
/// Jitter allowance added to the speed-scaled continuity gate.
pub(crate) const TRACKLET_JITTER_RAD: f32 = 0.005;
/// Absolute cap on the continuity gate.
pub(crate) const TRACKLET_GATE_MAX_RAD: f32 = 0.35;
/// Player-to-ball contact distance for the pickup release path.
pub(crate) const PLAYER_CONTACT_RAD: f32 = 0.05;
/// Neighbour radius around a dormant site for single-window dormancy.
pub(crate) const FAMILY_RAD: f32 = 2.0 * DEG_TO_RAD;
/// A dormant site unseen this long counts as vacated.
pub(crate) const VACANT_MS: f64 = 300.0;
/// A departure must begin within this long after the site was last seen.
pub(crate) const DEPART_START_MS: f64 = 750.0;
/// Outgoing release: minimum span of the departing tracklet.
pub(crate) const REACTIVATION_MIN_MS: f64 = 750.0;
/// Outgoing release: minimum observations of the departing tracklet.
pub(crate) const REACTIVATION_MIN_DETECTIONS: usize = 3;
/// Pickup release window after the site was vacated.
pub(crate) const PICKUP_OCCLUSION_MAX_MS: f64 = 2_000.0;
/// A release is confirmed only once the site has stayed empty this long
/// after the departure evidence (longer than the spare's longest observed
/// detection dropout on sample_02, ~2.1 s at the 99th+ percentile is rare).
/// A genuinely moved ball has already left the (<= 1 deg) exclusion zone, so
/// this delay does not hold back tracking of it.
pub(crate) const RELEASE_CONFIRM_MS: f64 = 2_000.0;
/// Evidence kept after a release; a re-static site re-dormants inside it.
pub(crate) const PROBATION_MS: f64 = 10_000.0;
/// Non-dormant clusters with no window are dropped after this long unseen.
const CLUSTER_TTL_MS: f64 = 1_000.0;
/// Static-window evidence is kept this long (must exceed the dormancy span).
const EVIDENCE_TTL_MS: f64 = 3.0 * DORMANT_EVIDENCE_SPAN_MS;
/// Tracklets are dropped after this long without an observation.
const TRACKLET_TTL_MS: f64 = 750.0;

#[derive(Debug, Clone, Copy)]
pub(crate) struct TimedPoint {
    pub yaw: f32,
    pub pitch: f32,
    pub timestamp_ms: f64,
}

fn distance(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dy = a.0 - b.0;
    let dp = a.1 - b.1;
    (dy * dy + dp * dp).sqrt()
}

fn percentile(mut values: Vec<f32>, p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((values.len() - 1) as f32 * p)
        .round()
        .clamp(0.0, (values.len() - 1) as f32) as usize;
    values[index]
}

fn medoid(points: &[TimedPoint]) -> (f32, f32) {
    points
        .iter()
        .map(|c| {
            let total: f32 = points
                .iter()
                .map(|p| distance((c.yaw, c.pitch), (p.yaw, p.pitch)))
                .sum();
            (total, (c.yaw, c.pitch))
        })
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, p)| p)
        .unwrap_or((0.0, 0.0))
}

/// Robust jitter radius: 3 x scaled MAD of distances to the medoid
/// (same estimator as the September static-site filter).
fn robust_radius(points: &[TimedPoint], center: (f32, f32)) -> f32 {
    let distances: Vec<f32> = points
        .iter()
        .map(|p| distance((p.yaw, p.pitch), center))
        .collect();
    let median = percentile(distances.clone(), 0.5);
    let deviations = distances.into_iter().map(|d| (d - median).abs()).collect();
    (3.0 * 1.4826 * percentile(deviations, 0.5)).clamp(SITE_MIN_RADIUS_RAD, SITE_MAX_RADIUS_RAD)
}

fn mean(points: &[TimedPoint]) -> (f32, f32) {
    let n = points.len().max(1) as f32;
    (
        points.iter().map(|p| p.yaw).sum::<f32>() / n,
        points.iter().map(|p| p.pitch).sum::<f32>() / n,
    )
}

/// Static test on one window: spread within a jitter radius and no net
/// drift between the first and last quarter of the window. Uses positions
/// over time only, so it is independent of the sample rate.
fn static_fit(points: &[TimedPoint]) -> Option<((f32, f32), f32)> {
    if points.len() < STATIC_MIN_DETECTIONS {
        return None;
    }
    let center = medoid(points);
    let radius = robust_radius(points, center);
    let distances: Vec<f32> = points
        .iter()
        .map(|p| distance((p.yaw, p.pitch), center))
        .collect();
    if percentile(distances, 0.95) > radius {
        return None;
    }
    let quarter = (points.len() / 4).max(1);
    let head = mean(&points[..quarter]);
    let tail = mean(&points[points.len() - quarter..]);
    if distance(head, tail) > radius {
        return None;
    }
    Some((center, radius))
}

#[derive(Debug, Clone, Copy)]
struct StaticWindow {
    start_ms: f64,
    end_ms: f64,
    contradicted: bool,
}

/// Video time of recent positions kept per tracklet.
const TRACKLET_HISTORY_MS: f64 = 3_000.0;

/// Short-lived association of ball candidates across frames, with a short
/// recent history so motion is judged on the last moments only.
#[derive(Debug, Clone)]
struct Tracklet {
    history: std::collections::VecDeque<TimedPoint>,
}

impl Tracklet {
    fn last(&self) -> TimedPoint {
        *self.history.back().expect("tracklet never empty")
    }

    fn push(&mut self, point: TimedPoint) {
        self.history.push_back(point);
        while self
            .history
            .front()
            .is_some_and(|p| point.timestamp_ms - p.timestamp_ms > TRACKLET_HISTORY_MS)
        {
            self.history.pop_front();
        }
    }

    /// Moving: displaced at least `MOVE_MIN_RAD` over the last
    /// `MOVING_MIN_MS` (or more) of history.
    fn moving(&self) -> bool {
        let last = self.last();
        self.history
            .iter()
            .rev()
            .find(|p| last.timestamp_ms - p.timestamp_ms >= MOVING_MIN_MS)
            .is_some_and(|p| distance((p.yaw, p.pitch), (last.yaw, last.pitch)) >= MOVE_MIN_RAD)
    }

    /// Latest history point inside `radius` of `center`, if any.
    fn last_inside(&self, center: (f32, f32), radius: f32) -> Option<(usize, TimedPoint)> {
        self.history
            .iter()
            .enumerate()
            .rev()
            .find(|(_, p)| distance((p.yaw, p.pitch), center) <= radius)
            .map(|(i, p)| (i, *p))
    }
}

#[derive(Debug)]
struct Cluster {
    id: u64,
    center: (f32, f32),
    radius: f32,
    window: Vec<TimedPoint>,
    window_contradicted: bool,
    windows: Vec<StaticWindow>,
    last_seen_ms: f64,
    dormant: bool,
    /// Release time while on probation.
    probation_since_ms: Option<f64>,
    /// Player standing at the site when it was last observed.
    last_near_player: Option<u64>,
    /// Departure evidence awaiting confirmation: (time, reason).
    pending_release: Option<(f64, &'static str)>,
}

impl Cluster {
    fn new(id: u64, point: TimedPoint) -> Self {
        Self {
            id,
            center: (point.yaw, point.pitch),
            radius: SITE_MIN_RADIUS_RAD,
            window: vec![point],
            window_contradicted: false,
            windows: Vec::new(),
            last_seen_ms: point.timestamp_ms,
            dormant: false,
            probation_since_ms: None,
            last_near_player: None,
            pending_release: None,
        }
    }

    fn gate(&self) -> f32 {
        if self.dormant { self.radius } else { ASSOC_RAD }
    }
}

/// Dormant static-site state for one [`BallTracker`](super::BallTracker).
#[derive(Debug, Default)]
pub(crate) struct DormantStaticSites {
    clusters: Vec<Cluster>,
    tracklets: Vec<Tracklet>,
    players: Vec<TrackedEntity>,
    next_id: u64,
}

impl DormantStaticSites {
    pub(crate) fn set_players(&mut self, players: &[TrackedEntity]) {
        self.players.clear();
        self.players.extend_from_slice(players);
    }

    fn next_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    /// Observe one frame of candidates. Called once per tracker update,
    /// before candidate filtering.
    pub(crate) fn observe_frame(
        &mut self,
        detections: &[MappedDetection],
        class_id: u16,
        timestamp_ms: f64,
    ) {
        let candidates: Vec<TimedPoint> = detections
            .iter()
            .filter(|d| d.class_id == class_id)
            .filter_map(|d| d.position)
            .map(|p| TimedPoint {
                yaw: p.yaw,
                pitch: p.pitch,
                timestamp_ms,
            })
            .collect();

        self.update_tracklets(&candidates, timestamp_ms);
        let moving: Vec<(f32, f32)> = self
            .tracklets
            .iter()
            .filter(|t| t.last().timestamp_ms == timestamp_ms && t.moving())
            .map(|t| (t.last().yaw, t.last().pitch))
            .collect();

        for point in &candidates {
            self.observe_candidate(*point);
        }
        let dormant_centers: Vec<(f32, f32)> = self
            .clusters
            .iter()
            .filter(|c| c.dormant)
            .map(|c| c.center)
            .collect();
        for cluster in &mut self.clusters {
            if moving
                .iter()
                .any(|m| distance(*m, cluster.center) >= CONTRADICTION_MIN_RAD)
            {
                cluster.window_contradicted = true;
            }
            Self::evaluate_window(cluster, timestamp_ms, &dormant_centers);
        }
        self.update_releases(timestamp_ms);
        self.expire(timestamp_ms);
    }

    /// Speed-plausible continuity gate for an association `dt_ms` apart.
    fn gate(dt_ms: f64) -> f32 {
        (MAX_BALL_SPEED_RAD_PER_MS * dt_ms.max(0.0) as f32 + TRACKLET_JITTER_RAD)
            .min(TRACKLET_GATE_MAX_RAD)
    }

    /// Greedy nearest association of this frame's candidates to tracklets.
    fn update_tracklets(&mut self, candidates: &[TimedPoint], timestamp_ms: f64) {
        let mut taken = vec![false; self.tracklets.len()];
        let mut new = Vec::new();
        for point in candidates {
            let best = self
                .tracklets
                .iter()
                .enumerate()
                .filter(|(i, t)| {
                    !taken[*i]
                        && timestamp_ms - t.last().timestamp_ms <= TRACKLET_TTL_MS
                        && distance((t.last().yaw, t.last().pitch), (point.yaw, point.pitch))
                            <= Self::gate(timestamp_ms - t.last().timestamp_ms)
                })
                .min_by(|a, b| {
                    let da = distance((a.1.last().yaw, a.1.last().pitch), (point.yaw, point.pitch));
                    let db = distance((b.1.last().yaw, b.1.last().pitch), (point.yaw, point.pitch));
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i);
            match best {
                Some(i) => {
                    taken[i] = true;
                    self.tracklets[i].push(*point);
                }
                None => new.push(*point),
            }
        }
        for point in new {
            let mut history = std::collections::VecDeque::new();
            history.push_back(point);
            self.tracklets.push(Tracklet { history });
        }
        self.tracklets
            .retain(|t| timestamp_ms - t.last().timestamp_ms <= TRACKLET_TTL_MS);
    }

    fn observe_candidate(&mut self, point: TimedPoint) {
        let p = (point.yaw, point.pitch);
        let index = self
            .clusters
            .iter()
            .enumerate()
            .filter(|(_, c)| distance(c.center, p) <= c.gate())
            .min_by(|a, b| {
                distance(a.1.center, p)
                    .partial_cmp(&distance(b.1.center, p))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);
        let near_player = self
            .players
            .iter()
            .filter(|pl| distance((pl.yaw, pl.pitch), p) <= PLAYER_CONTACT_RAD)
            .min_by(|a, b| {
                distance((a.yaw, a.pitch), p)
                    .partial_cmp(&distance((b.yaw, b.pitch), p))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|pl| pl.id);
        match index {
            Some(i) => {
                let c = &mut self.clusters[i];
                if point.timestamp_ms - c.last_seen_ms > OBS_GAP_MS {
                    c.window.clear();
                    c.window_contradicted = false;
                }
                c.window.push(point);
                c.last_seen_ms = point.timestamp_ms;
                c.last_near_player = near_player;
            }
            None => {
                let id = self.next_id();
                self.clusters.push(Cluster::new(id, point));
            }
        }
    }

    /// Record a static window when the cluster has stayed put long enough;
    /// otherwise slide the window forward.
    fn evaluate_window(cluster: &mut Cluster, timestamp_ms: f64, dormant_centers: &[(f32, f32)]) {
        let Some(first) = cluster.window.first().copied() else {
            return;
        };
        if cluster.window.last().map(|p| p.timestamp_ms) != Some(timestamp_ms) {
            return;
        }
        if timestamp_ms - first.timestamp_ms < STATIC_WINDOW_MS {
            return;
        }
        match static_fit(&cluster.window) {
            Some((center, radius)) => {
                cluster.center = center;
                cluster.radius = radius;
                cluster.windows.push(StaticWindow {
                    start_ms: first.timestamp_ms,
                    end_ms: timestamp_ms,
                    contradicted: cluster.window_contradicted,
                });
                let contradicted = cluster.window_contradicted;
                cluster.window.clear();
                cluster.window_contradicted = false;
                Self::refresh_dormancy(cluster, timestamp_ms, contradicted, dormant_centers);
            }
            None => {
                // Not static over this span: drop the older half and keep
                // looking (the object may settle later).
                let cut = timestamp_ms - STATIC_WINDOW_MS / 2.0;
                cluster.window.retain(|p| p.timestamp_ms >= cut);
                if let Some(latest) = cluster.window.last() {
                    cluster.center = (latest.yaw, latest.pitch);
                }
            }
        }
    }

    fn refresh_dormancy(
        cluster: &mut Cluster,
        timestamp_ms: f64,
        window_contradicted: bool,
        dormant_centers: &[(f32, f32)],
    ) {
        if cluster.dormant {
            return;
        }
        // Probation: a released site that settles again is the same spare.
        if let Some(since) = cluster.probation_since_ms {
            if timestamp_ms - since <= PROBATION_MS {
                cluster.dormant = true;
                cluster.probation_since_ms = None;
                log::info!(
                    "BallTracker: dormant static site yaw={:.3} pitch={:.3} radius={:.4}rad site_id={} (re-static during probation)",
                    cluster.center.0,
                    cluster.center.1,
                    cluster.radius,
                    cluster.id
                );
                return;
            }
        }
        let windows = &cluster.windows;
        let span = windows
            .last()
            .zip(windows.first())
            .map_or(0.0, |(l, f)| l.end_ms - f.start_ms);
        let contradicted = windows.iter().filter(|w| w.contradicted).count();
        let recurring = windows.len() >= DORMANT_MIN_WINDOWS
            && span >= DORMANT_EVIDENCE_SPAN_MS
            && contradicted > 0;
        let neighbour = window_contradicted
            && dormant_centers.iter().any(|c| {
                let d = distance(*c, cluster.center);
                d > f32::EPSILON && d <= FAMILY_RAD
            });
        if recurring || neighbour {
            cluster.dormant = true;
            log::info!(
                "BallTracker: dormant static site yaw={:.3} pitch={:.3} radius={:.4}rad site_id={} windows={} contradicted={}{}",
                cluster.center.0,
                cluster.center.1,
                cluster.radius,
                cluster.id,
                windows.len(),
                contradicted,
                if recurring {
                    ""
                } else {
                    " (neighbour of dormant site)"
                }
            );
        }
    }

    /// Release a dormant site only on evidence that its own ball moved away.
    fn update_releases(&mut self, timestamp_ms: f64) {
        for c in self.clusters.iter_mut().filter(|c| c.dormant) {
            // Pending release: confirm once the site stayed empty long
            // enough; cancel if the site was observed again (spare still
            // there, the "departure" was another object or a dropout).
            if let Some((since, reason)) = c.pending_release {
                if c.last_seen_ms > since {
                    log::debug!(
                        "BallTracker: release cancelled site_id={} ({reason}): site observed again",
                        c.id
                    );
                    c.pending_release = None;
                } else if timestamp_ms - since >= RELEASE_CONFIRM_MS {
                    log::info!(
                        "BallTracker: reactivated static site yaw={:.3} pitch={:.3} site_id={} ({reason})",
                        c.center.0,
                        c.center.1,
                        c.id
                    );
                    c.dormant = false;
                    c.pending_release = None;
                    c.probation_since_ms = Some(timestamp_ms);
                    c.window.clear();
                    c.window_contradicted = false;
                    continue;
                } else {
                    continue;
                }
            }
            let vacant = timestamp_ms - c.last_seen_ms >= VACANT_MS;
            if !vacant {
                continue;
            }
            let start_rad = 3.0 * c.radius;
            // Departure: a live tracklet that was at the site around the last
            // sighting and has since travelled outward. The site itself must
            // have stayed empty since (checked via `vacant` above, and the
            // tracklet's own in-site point must be its last one there).
            let departure = self
                .tracklets
                .iter()
                .filter(|t| t.last().timestamp_ms == timestamp_ms)
                .filter_map(|t| {
                    let (index, at_site) = t.last_inside(c.center, start_rad)?;
                    if (at_site.timestamp_ms - c.last_seen_ms).abs() > DEPART_START_MS {
                        return None;
                    }
                    let after: Vec<TimedPoint> = t.history.iter().skip(index).copied().collect();
                    Some((t, at_site, after))
                })
                .max_by(|a, b| {
                    let da = distance((a.0.last().yaw, a.0.last().pitch), c.center);
                    let db = distance((b.0.last().yaw, b.0.last().pitch), c.center);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });
            let Some((tracklet, at_site, after)) = departure else {
                continue;
            };
            let last = tracklet.last();
            let start_d = distance((at_site.yaw, at_site.pitch), c.center);
            let now_d = distance((last.yaw, last.pitch), c.center);
            let outward = now_d >= (4.0 * c.radius).max(start_d + 2.0 * c.radius);

            let outgoing = outward
                && after.len() >= REACTIVATION_MIN_DETECTIONS
                && last.timestamp_ms - at_site.timestamp_ms >= REACTIVATION_MIN_MS;
            let pickup = outward
                && after.len() >= 2
                && timestamp_ms - c.last_seen_ms <= PICKUP_OCCLUSION_MAX_MS
                && c.last_near_player.is_some_and(|id| {
                    self.players.iter().any(|pl| {
                        pl.id == id
                            && distance((pl.yaw, pl.pitch), (last.yaw, last.pitch))
                                <= PLAYER_CONTACT_RAD
                    })
                });
            let reason = if outgoing {
                "persistent outgoing motion"
            } else if pickup {
                "pickup occlusion"
            } else {
                continue;
            };
            log::debug!("BallTracker: release pending site_id={} ({reason})", c.id);
            c.pending_release = Some((timestamp_ms, reason));
        }
    }

    fn expire(&mut self, timestamp_ms: f64) {
        self.clusters.retain_mut(|c| {
            if c.dormant {
                return true;
            }
            if c.probation_since_ms
                .is_some_and(|since| timestamp_ms - since > PROBATION_MS)
            {
                // Probation over without re-settling: forget the evidence.
                c.probation_since_ms = None;
                c.windows.clear();
            }
            if c.probation_since_ms.is_some() {
                return true;
            }
            c.windows
                .retain(|w| timestamp_ms - w.end_ms <= EVIDENCE_TTL_MS);
            !c.windows.is_empty() || timestamp_ms - c.last_seen_ms <= CLUSTER_TTL_MS
        });
    }

    /// Whether a candidate position lies inside a dormant site.
    pub(crate) fn rejects(&self, point: (f32, f32)) -> bool {
        self.clusters
            .iter()
            .any(|c| c.dormant && distance(point, c.center) <= c.radius)
    }

    #[cfg(test)]
    pub(crate) fn dormant_count(&self) -> usize {
        self.clusters.iter().filter(|c| c.dormant).count()
    }

    #[cfg(test)]
    pub(crate) fn seed_dormant(&mut self, center: (f32, f32), last_seen_ms: f64) {
        let id = self.next_id();
        let mut c = Cluster::new(
            id,
            TimedPoint {
                yaw: center.0,
                pitch: center.1,
                timestamp_ms: last_seen_ms,
            },
        );
        c.dormant = true;
        c.window.clear();
        self.clusters.push(c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pts(n: usize, step_ms: f64, f: impl Fn(usize) -> (f32, f32)) -> Vec<TimedPoint> {
        (0..n)
            .map(|i| {
                let (yaw, pitch) = f(i);
                TimedPoint {
                    yaw,
                    pitch,
                    timestamp_ms: i as f64 * step_ms,
                }
            })
            .collect()
    }

    fn jitter(i: usize) -> (f32, f32) {
        // Deterministic +-0.3 mrad jitter.
        let s = ((i * 7919) % 13) as f32 / 12.0 - 0.5;
        let c = ((i * 104_729) % 11) as f32 / 10.0 - 0.5;
        (1.0 + 0.0006 * s, 0.0006 * c)
    }

    #[test]
    fn static_fit_is_rate_invariant() {
        for step in [33.4, 16.7, 8.3] {
            let n = (1_600.0 / step) as usize;
            assert!(static_fit(&pts(n, step, jitter)).is_some(), "step {step}");
        }
    }

    #[test]
    fn static_fit_rejects_slow_drift_at_any_rate() {
        for step in [33.4, 16.7, 8.3] {
            let n = (1_600.0 / step) as usize;
            let drift = |i: usize| (1.0 + 0.02 * i as f32 / n as f32, 0.0);
            assert!(static_fit(&pts(n, step, drift)).is_none(), "step {step}");
        }
    }
}
