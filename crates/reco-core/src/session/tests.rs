//! Integration tests for [`StitchSession`] frame loop.
//!
//! GPU-dependent tests are marked `#[ignore]` so they can be skipped in CI
//! environments without a GPU. Run them explicitly with:
//! ```bash
//! cargo test -p reco-core -- --ignored
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::StitchSession;
use super::types::*;
use crate::calibration::{CameraParams, MatchCalibration, PlaneLayout};
use crate::detect::detector::{CameraId, Detection, DetectorError, DetectorFrame, UnifiedDetector};
use crate::detect::director::{MappedDetection, ViewportPosition};
use crate::detect::panner::{PanContext, Panner};
use crate::detect::tracker::{TrackState, TrackedEntity, Tracker, WorldState};
use crate::encoder::{EncodeError, Encoder, OutputFrame};
use crate::render::viewport::ViewportConfig;
use crate::source::{FramePair, FrameSource, SourceError, SourceInfo, StereoFrame, YuvData};

// ─── Helpers ───────────────────────────────────────────────────────────

/// Small test dimensions to keep GPU allocations minimal.
const W: u32 = 64;
const H: u32 = 64;

/// Create a minimal valid calibration for 64x64 frames.
fn test_calibration() -> MatchCalibration {
    let cam = CameraParams {
        width: W,
        height: H,
        fx: 32.0,
        fy: 32.0,
        cx: 32.0,
        cy: 32.0,
        d: [0.0; 4],
    };
    MatchCalibration {
        left: cam.clone(),
        right: cam,
        layout: PlaneLayout {
            camera_axis_offset: 0.25,
            intersect: 0.5,
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

/// Create a valid YUV420P stereo frame pair of solid gray.
fn solid_frame() -> StereoFrame {
    let y_size = (W * H) as usize;
    let uv_size = ((W / 2) * (H / 2)) as usize;
    let yuv = YuvData {
        y: vec![128u8; y_size],
        u: vec![128u8; uv_size],
        v: vec![128u8; uv_size],
    };
    StereoFrame::Yuv420p(FramePair {
        left: yuv.clone(),
        right: yuv,
    })
}

// ─── MockSource ────────────────────────────────────────────────────────

/// Mock frame source that returns N solid-color YUV420P frame pairs.
struct MockSource {
    remaining: u64,
}

impl MockSource {
    fn new(frame_count: u64) -> Self {
        Self {
            remaining: frame_count,
        }
    }
}

impl FrameSource for MockSource {
    fn info(&self) -> SourceInfo {
        SourceInfo {
            width: W,
            height: H,
            fps: 30.0,
            fps_rational: Some((30, 1)),
            total_frames: None,
        }
    }

    fn next_frame(&mut self) -> Result<Option<StereoFrame>, SourceError> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        Ok(Some(solid_frame()))
    }
}

// ─── MockDetector ──────────────────────────────────────────────────────

/// Mock detector that returns canned detections and counts calls.
struct MockDetector {
    detections: Vec<Detection>,
    call_count: Arc<AtomicU64>,
}

impl MockDetector {
    fn new(detections: Vec<Detection>, call_count: Arc<AtomicU64>) -> Self {
        Self {
            detections,
            call_count,
        }
    }
}

impl UnifiedDetector for MockDetector {
    fn name(&self) -> &'static str {
        "mock"
    }

    fn detect(
        &mut self,
        _camera: CameraId,
        _frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        Ok(self.detections.clone())
    }
}

// ─── MockTracker / MockPanner ──────────────────────────────────────────

/// Mock tracker that records how many detections it received each
/// frame and returns a single "ball" entity.
struct MockTracker {
    log: Arc<Mutex<Vec<TrackerSnapshot>>>,
    class_id: u16,
}

/// Snapshot of what a tracker saw on a single `update()` call.
#[derive(Debug)]
struct TrackerSnapshot {
    frame_index: u64,
    detection_count: usize,
}

impl MockTracker {
    fn new(class_id: u16, log: Arc<Mutex<Vec<TrackerSnapshot>>>) -> Self {
        Self { class_id, log }
    }
}

impl Tracker for MockTracker {
    fn update(&mut self, detections: &[MappedDetection], _timestamp_ms: f64) -> Vec<TrackedEntity> {
        // Session bumps frame_count after dispatch, so `log.len()` is
        // the per-call frame_index we see.
        let frame_index = self.log.lock().unwrap().len() as u64;
        self.log.lock().unwrap().push(TrackerSnapshot {
            frame_index,
            detection_count: detections.len(),
        });
        vec![TrackedEntity {
            id: 0,
            class_id: self.class_id,
            yaw: 0.0,
            pitch: 0.0,
            confidence: 1.0,
            state: TrackState::Tracking,
            age_frames: 1,
            origin: CameraId::Left,
        }]
    }

    fn class_id(&self) -> u16 {
        self.class_id
    }
}

/// Mock panner that returns a fixed position regardless of the world.
struct MockPanner {
    position: ViewportPosition,
}

impl Panner for MockPanner {
    fn decide(&mut self, _world: &WorldState, _ctx: &PanContext<'_>) -> ViewportPosition {
        self.position
    }
}

// ─── MockEncoder ───────────────────────────────────────────────────────

/// Mock encoder that records submitted frames (counts and dimensions).
struct MockEncoder {
    submitted: Arc<AtomicU64>,
}

impl MockEncoder {
    fn new(counter: Arc<AtomicU64>) -> Self {
        Self { submitted: counter }
    }
}

impl Encoder for MockEncoder {
    fn submit(&mut self, frame: OutputFrame<'_>) -> Result<(), EncodeError> {
        assert!(frame.width > 0, "frame width must be positive");
        assert!(frame.height > 0, "frame height must be positive");
        assert!(!frame.data.is_empty(), "frame data must not be empty");
        self.submitted.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn finish(&mut self) -> Result<(), EncodeError> {
        Ok(())
    }
}

// ─── NaN Panner ────────────────────────────────────────────────────────

/// Panner that returns NaN yaw/pitch to test coverage clamping resilience.
struct NanPanner;

impl Panner for NanPanner {
    fn decide(&mut self, _world: &WorldState, _ctx: &PanContext<'_>) -> ViewportPosition {
        ViewportPosition {
            yaw: f32::NAN,
            pitch: f32::NAN,
            fov_degrees: None,
        }
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

/// Build a session using the builder pattern with test defaults.
/// Trackers and panners are attached post-build (they have their own
/// setters and aren't part of the builder API).
fn build_test_session(
    encoder: Option<Box<dyn Encoder + Send>>,
    detector: Option<Box<dyn UnifiedDetector>>,
    detection_interval: u64,
) -> Result<StitchSession, SessionError> {
    let mut builder = StitchSession::builder()
        .calibration(test_calibration())
        .input_dimensions(W, H)
        .viewport(ViewportConfig {
            width: 64,
            height: 64,
            blend_width: 0.15,
            fov_degrees: 75.0,
            rig_tilt: 0.0,
            rig_roll: 0.0,
            ..ViewportConfig::default()
        })
        .detection_interval(detection_interval);

    if let Some(enc) = encoder {
        builder = builder.encoder(enc, 2);
    }
    if let Some(det) = detector {
        builder = builder.detector(det);
    }

    builder.build()
}

#[test]
#[ignore] // requires GPU
fn basic_frame_loop() {
    let submitted = Arc::new(AtomicU64::new(0));
    let encoder = MockEncoder::new(Arc::clone(&submitted));

    let mut session = build_test_session(Some(Box::new(encoder)), None, 1).expect("session build");

    let mut source = MockSource::new(10);
    let interrupted = AtomicBool::new(false);

    let processed = session
        .run(&mut source, u64::MAX, &interrupted, None)
        .expect("run");

    session.finish().expect("finish");

    assert_eq!(processed, 10, "should process all 10 frames");

    // The async encoder receives frames with a 1-frame pipeline delay
    // (NV12 double-buffering), so we expect at least 9 frames submitted.
    let count = submitted.load(Ordering::Relaxed);
    assert!(
        (9..=10).contains(&count),
        "encoder should receive 9-10 frames, got {count}"
    );
}

#[test]
#[ignore] // requires GPU
fn tracker_receives_detections() {
    let call_count = Arc::new(AtomicU64::new(0));
    let canned = vec![
        Detection {
            camera: CameraId::Left,
            class_id: 0,
            confidence: 0.9,
            center_x: 0.5,
            center_y: 0.5,
            width: 0.1,
            height: 0.1,
        },
        Detection {
            camera: CameraId::Right,
            class_id: 1,
            confidence: 0.8,
            center_x: 0.3,
            center_y: 0.7,
            width: 0.05,
            height: 0.05,
        },
    ];

    let detector = MockDetector::new(canned, Arc::clone(&call_count));
    let tracker_log = Arc::new(Mutex::new(Vec::new()));
    let tracker = MockTracker::new(0, Arc::clone(&tracker_log));

    let submitted = Arc::new(AtomicU64::new(0));
    let encoder = MockEncoder::new(Arc::clone(&submitted));

    let mut session = build_test_session(Some(Box::new(encoder)), Some(Box::new(detector)), 1)
        .expect("session build");
    session.set_ball_tracker(Box::new(tracker));
    session.set_panner(Box::new(MockPanner {
        position: ViewportPosition::default(),
    }));

    let mut source = MockSource::new(3);
    let interrupted = AtomicBool::new(false);

    let processed = session
        .run(&mut source, u64::MAX, &interrupted, None)
        .expect("run");

    session.finish().expect("finish");

    assert_eq!(processed, 3);

    let log = tracker_log.lock().unwrap();
    assert_eq!(log.len(), 3, "tracker should be updated once per frame");

    // The detector returns 2 detections per detect() call, and the session
    // calls detect() twice per frame (once for left, once for right camera).
    // So each frame produces 4 mapped detections the tracker observes.
    for (i, snapshot) in log.iter().enumerate() {
        assert_eq!(snapshot.frame_index, i as u64);
        assert!(
            snapshot.detection_count > 0,
            "frame {i}: tracker should receive non-empty detections, got {}",
            snapshot.detection_count,
        );
    }
}

#[test]
#[ignore] // requires GPU
fn detection_interval_respected() {
    let call_count = Arc::new(AtomicU64::new(0));
    let canned = vec![Detection {
        camera: CameraId::Left,
        class_id: 0,
        confidence: 0.9,
        center_x: 0.5,
        center_y: 0.5,
        width: 0.1,
        height: 0.1,
    }];

    let detector = MockDetector::new(canned, Arc::clone(&call_count));

    let submitted = Arc::new(AtomicU64::new(0));
    let encoder = MockEncoder::new(Arc::clone(&submitted));

    let mut session = build_test_session(Some(Box::new(encoder)), Some(Box::new(detector)), 3)
        .expect("session build");

    let mut source = MockSource::new(10);
    let interrupted = AtomicBool::new(false);

    let processed = session
        .run(&mut source, u64::MAX, &interrupted, None)
        .expect("run");

    session.finish().expect("finish");

    assert_eq!(processed, 10);

    // With interval=3 and 10 frames (indices 0..9), detection runs at
    // frames 0, 3, 6, 9 = 4 times. Each detection call triggers detect()
    // twice (left + right camera), so total calls = 4 * 2 = 8.
    let calls = call_count.load(Ordering::Relaxed);
    // Allow some tolerance: 6..=10 calls covers 3-5 detection frames x 2 cameras.
    assert!(
        (6..=10).contains(&calls),
        "with interval=3 over 10 frames, expected 6-10 detector calls, got {calls}"
    );
}

#[test]
#[ignore] // requires GPU
fn nan_panner_does_not_crash() {
    let submitted = Arc::new(AtomicU64::new(0));
    let encoder = MockEncoder::new(Arc::clone(&submitted));

    let mut session = build_test_session(Some(Box::new(encoder)), None, 1).expect("session build");
    session.set_panner(Box::new(NanPanner));

    let mut source = MockSource::new(5);
    let interrupted = AtomicBool::new(false);

    // This should not panic. Coverage clamping handles NaN gracefully.
    let processed = session
        .run(&mut source, u64::MAX, &interrupted, None)
        .expect("run should succeed even with NaN panner output");

    session.finish().expect("finish");

    assert_eq!(processed, 5, "all 5 frames should be processed");
}

// ─── compute_frame_limit tests (no GPU required) ──────────────────────

#[test]
fn compute_frame_limit_both_none() {
    assert_eq!(compute_frame_limit(None, None, 30.0), u64::MAX);
}

#[test]
fn compute_frame_limit_negative_duration() {
    // Negative duration should be treated as "no duration limit".
    assert_eq!(compute_frame_limit(Some(-5.0), None, 30.0), u64::MAX);
}

#[test]
fn compute_frame_limit_zero_fps_uses_fallback() {
    // Zero fps should use the 30.0 fallback.
    let result = compute_frame_limit(Some(10.0), None, 0.0);
    // 10.0 * 30.0 = 300
    assert_eq!(result, 300);
}

#[test]
fn compute_frame_limit_both_set_min_wins() {
    // duration=2s at 30fps = 60 frames, max_frames=40 -> min(60,40) = 40
    assert_eq!(compute_frame_limit(Some(2.0), Some(40), 30.0), 40);

    // duration=1s at 30fps = 30 frames, max_frames=100 -> min(30,100) = 30
    assert_eq!(compute_frame_limit(Some(1.0), Some(100), 30.0), 30);
}

#[test]
fn compute_frame_limit_duration_only() {
    // 5s at 60fps = 300 frames
    assert_eq!(compute_frame_limit(Some(5.0), None, 60.0), 300);
}

#[test]
fn compute_frame_limit_max_frames_only() {
    assert_eq!(compute_frame_limit(None, Some(42), 30.0), 42);
}

#[test]
fn compute_frame_limit_negative_fps_uses_fallback() {
    // Negative fps should also trigger the 30.0 fallback.
    let result = compute_frame_limit(Some(10.0), None, -1.0);
    assert_eq!(result, 300);
}

// ─── Analysis clock (OEV media-time fix) ───────────────────────────────
//
// These drive the real `StitchSession::run` loop (buffered lookahead and
// immediate paths) and record the `timestamp_ms` the tracker receives.
// File sources report an exact `fps_rational`; the tracker must see the
// recording's timeline (`source_index * den / num`), independent of how
// fast frames are processed. Live sources (no `fps_rational`) keep the
// wall clock.

/// Frame source with configurable timing metadata.
struct TimedSource {
    remaining: u64,
    fps: f64,
    fps_rational: Option<(i32, i32)>,
}

impl FrameSource for TimedSource {
    fn info(&self) -> SourceInfo {
        SourceInfo {
            width: W,
            height: H,
            fps: self.fps,
            fps_rational: self.fps_rational,
            total_frames: None,
        }
    }

    fn next_frame(&mut self) -> Result<Option<StereoFrame>, SourceError> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        Ok(Some(solid_frame()))
    }
}

/// Detector that sleeps per call to emulate slow processing.
struct SlowDetector {
    delay: std::time::Duration,
}

impl UnifiedDetector for SlowDetector {
    fn name(&self) -> &'static str {
        "slow-mock"
    }

    fn detect(
        &mut self,
        _camera: CameraId,
        _frame: &DetectorFrame<'_>,
    ) -> Result<Vec<Detection>, DetectorError> {
        std::thread::sleep(self.delay);
        Ok(vec![Detection {
            camera: CameraId::Left,
            class_id: 0,
            confidence: 0.9,
            center_x: 0.5,
            center_y: 0.5,
            width: 0.1,
            height: 0.1,
        }])
    }
}

/// Tracker that records every `timestamp_ms` it is given.
struct TimestampTracker {
    log: Arc<Mutex<Vec<f64>>>,
}

impl Tracker for TimestampTracker {
    fn update(&mut self, _detections: &[MappedDetection], timestamp_ms: f64) -> Vec<TrackedEntity> {
        self.log.lock().unwrap().push(timestamp_ms);
        vec![]
    }

    fn class_id(&self) -> u16 {
        0
    }
}

/// (event kind, frame_index, timestamp_ms).
type TimedEvent = (&'static str, u64, f64);

/// Event sink collecting timed events.
struct TimestampSink {
    log: Arc<Mutex<Vec<TimedEvent>>>,
}

impl crate::detect::pipeline_event::PipelineEventSink for TimestampSink {
    fn emit(&mut self, event: crate::detect::pipeline_event::PipelineEvent) {
        use crate::detect::pipeline_event::PipelineEvent as E;
        let entry = match event {
            E::FrameStart {
                frame_index,
                timestamp_ms,
            } => ("frame_start", frame_index, timestamp_ms),
            E::WorldState {
                frame_index,
                timestamp_ms,
                ..
            } => ("world_state", frame_index, timestamp_ms),
            _ => return,
        };
        self.log.lock().unwrap().push(entry);
    }
}

struct ClockRun {
    tracker_ms: Vec<f64>,
    events: Vec<TimedEvent>,
}

fn run_clock_session(
    frames: u64,
    fps: f64,
    fps_rational: Option<(i32, i32)>,
    lookahead: usize,
    stride: u64,
    detect_delay_ms: u64,
) -> ClockRun {
    let detector = SlowDetector {
        delay: std::time::Duration::from_millis(detect_delay_ms),
    };
    let mut session = build_test_session(None, Some(Box::new(detector)), 1).expect("session build");
    let tracker_log = Arc::new(Mutex::new(Vec::new()));
    session.set_ball_tracker(Box::new(TimestampTracker {
        log: Arc::clone(&tracker_log),
    }));
    session.set_panner(Box::new(MockPanner {
        position: ViewportPosition::default(),
    }));
    let events = Arc::new(Mutex::new(Vec::new()));
    session.set_event_sink(Box::new(TimestampSink {
        log: Arc::clone(&events),
    }));
    session.set_lookahead(lookahead);
    session.set_frame_stride(stride);
    let mut source = TimedSource {
        remaining: frames,
        fps,
        fps_rational,
    };
    let interrupted = AtomicBool::new(false);
    session
        .run(&mut source, u64::MAX, &interrupted, None)
        .expect("run");
    session.finish().expect("finish");
    let tracker_ms = tracker_log.lock().unwrap().clone();
    let events = events.lock().unwrap().clone();
    ClockRun { tracker_ms, events }
}

fn media_ms(index: u64, num: u64, den: u64) -> f64 {
    index as f64 * den as f64 * 1000.0 / num as f64
}

fn assert_media_times(got: &[f64], indices: &[u64], num: u64, den: u64) {
    assert_eq!(got.len(), indices.len(), "tracker call count: {got:?}");
    for (ms, &i) in got.iter().zip(indices) {
        let want = media_ms(i, num, den);
        assert!(
            (ms - want).abs() < 1e-6,
            "source frame {i}: tracker got {ms} ms, media time is {want} ms"
        );
    }
}

#[test]
#[ignore] // requires GPU
fn offline_stride1_lookahead_tracker_uses_media_time() {
    let run = run_clock_session(12, 60000.0 / 1001.0, Some((60000, 1001)), 4, 1, 5);
    let indices: Vec<u64> = (0..12).collect();
    assert_media_times(&run.tracker_ms, &indices, 60000, 1001);
    // Emitted world_state timestamps carry the same media timeline.
    let ws: Vec<f64> = run
        .events
        .iter()
        .filter(|e| e.0 == "world_state")
        .map(|e| e.2)
        .collect();
    assert_media_times(&ws, &indices, 60000, 1001);
}

#[test]
#[ignore] // requires GPU
fn offline_stride3_lookahead_tracker_uses_media_time() {
    let run = run_clock_session(13, 60000.0 / 1001.0, Some((60000, 1001)), 6, 3, 2);
    assert_media_times(&run.tracker_ms, &[0, 3, 6, 9, 12], 60000, 1001);
}

#[test]
#[ignore] // requires GPU
fn offline_immediate_tracker_uses_media_time() {
    let run = run_clock_session(8, 30000.0 / 1001.0, Some((30000, 1001)), 0, 1, 5);
    let indices: Vec<u64> = (0..8).collect();
    assert_media_times(&run.tracker_ms, &indices, 30000, 1001);
}

#[test]
#[ignore] // requires GPU
fn offline_tracker_time_independent_of_processing_speed() {
    for lookahead in [0usize, 4] {
        let fast = run_clock_session(10, 25.0, Some((25, 1)), lookahead, 1, 0);
        let slow = run_clock_session(10, 25.0, Some((25, 1)), lookahead, 1, 30);
        assert_eq!(
            fast.tracker_ms, slow.tracker_ms,
            "lookahead={lookahead}: tracker timeline changed with processing speed"
        );
    }
}

#[test]
#[ignore] // requires GPU
fn offline_tracker_time_is_strictly_monotonic() {
    for (lookahead, stride) in [(0usize, 1u64), (4, 1), (6, 2), (8, 4)] {
        let run = run_clock_session(17, 50.0, Some((50, 1)), lookahead, stride, 0);
        assert!(!run.tracker_ms.is_empty());
        assert!(
            run.tracker_ms.windows(2).all(|w| w[1] > w[0]),
            "lookahead={lookahead} stride={stride}: {:?}",
            run.tracker_ms
        );
    }
}

#[test]
#[ignore] // requires GPU
fn frame_start_events_keep_wall_clock_for_profiling() {
    // 1 fps media: frame 9 is at 9000 ms of media time. Processing 10 tiny
    // frames takes far less wall time, so frame_start must stay well below.
    let run = run_clock_session(10, 1.0, Some((1, 1)), 4, 1, 0);
    let starts: Vec<f64> = run
        .events
        .iter()
        .filter(|e| e.0 == "frame_start")
        .map(|e| e.2)
        .collect();
    assert_eq!(starts.len(), 10);
    assert!(starts.windows(2).all(|w| w[1] >= w[0]));
    assert!(
        *starts.last().unwrap() < 9000.0,
        "frame_start should be processing wall time, got {starts:?}"
    );
    assert_media_times(&run.tracker_ms, &(0..10).collect::<Vec<_>>(), 1, 1);
}

#[test]
#[ignore] // requires GPU
fn live_source_without_rational_fps_keeps_wall_clock() {
    // No fps_rational => live semantics preserved: tracker time is wall
    // clock since session start, far below the 1 fps nominal timeline.
    for lookahead in [0usize, 4] {
        let run = run_clock_session(10, 1.0, None, lookahead, 1, 0);
        assert_eq!(run.tracker_ms.len(), 10);
        assert!(run.tracker_ms.windows(2).all(|w| w[1] >= w[0]));
        assert!(
            *run.tracker_ms.last().unwrap() < 9000.0,
            "lookahead={lookahead}: live source should use wall time, got {:?}",
            run.tracker_ms
        );
    }
}

// ─── Offline director replay parity (OEV-006) ──────────────────────────
//
// `offline_director::replay_dense_lookahead` must reproduce the live
// stride-1 lookahead loop exactly: same bridged worlds, same lookahead
// windows, same `previous_position` interleaving, same smoothed poses.

/// Deterministic scripted ball: Tracking runs separated by None and
/// Coasting gaps so the backward bridge is exercised, with one fast jump
/// that the bridge speed gate must reject.
fn scripted_ball(i: u64) -> Option<TrackedEntity> {
    let state = match i % 11 {
        3 | 4 => return None,
        7 => TrackState::Coasting,
        _ => TrackState::Tracking,
    };
    let jump = if i >= 30 { 1.2 } else { 0.0 };
    Some(TrackedEntity {
        id: 1,
        class_id: 32,
        yaw: -0.4 + 0.013 * i as f32 + jump,
        pitch: 0.02 + 0.001 * (i % 5) as f32,
        confidence: 0.8,
        state,
        age_frames: i,
        origin: CameraId::Left,
    })
}

/// Ball tracker that replays [`scripted_ball`] by call order.
struct ScriptedBallTracker {
    calls: u64,
}

impl Tracker for ScriptedBallTracker {
    fn update(
        &mut self,
        _detections: &[MappedDetection],
        _timestamp_ms: f64,
    ) -> Vec<TrackedEntity> {
        let out = scripted_ball(self.calls).into_iter().collect();
        self.calls += 1;
        out
    }

    fn class_id(&self) -> u16 {
        32
    }
}

/// Panner whose output depends on the current world, the whole lookahead
/// window and `previous_position`, so any divergence in any of the three
/// shows up in the pose.
struct ProbePanner;

impl Panner for ProbePanner {
    fn decide(&mut self, world: &WorldState, ctx: &PanContext<'_>) -> ViewportPosition {
        self.decide_with_lookahead(world, &[], ctx)
    }

    fn decide_with_lookahead(
        &mut self,
        world: &WorldState,
        future: &[WorldState],
        ctx: &PanContext<'_>,
    ) -> ViewportPosition {
        let now = world.ball.map_or(0.0, |b| b.yaw);
        let mut fsum = 0.0f32;
        let mut fcount = 0u32;
        for (k, ws) in future.iter().enumerate() {
            if let Some(b) = ws.ball {
                fsum += b.yaw * (k as f32 + 1.0);
                fcount += 1;
            }
        }
        let state_code = match world.ball.map(|b| b.state) {
            None => 0.0,
            Some(TrackState::Tracking) => 0.001,
            Some(TrackState::Coasting) => 0.002,
            Some(TrackState::Bridged) => 0.003,
            Some(TrackState::Lost) => 0.004,
        };
        ViewportPosition {
            yaw: now + 0.01 * fsum + 0.1 * ctx.previous_position.yaw + state_code,
            pitch: world.ball.map_or(0.0, |b| b.pitch) + 0.0001 * fcount as f32,
            fov_degrees: Some(40.0 + future.len() as f32 * 0.25 + ctx.previous_position.pitch),
        }
    }
}

/// (bridged ball as recorded, rendered pose) per rendered frame.
type DirectorLog = Arc<Mutex<Vec<(Option<TrackedEntity>, ViewportPosition)>>>;

/// Event sink keeping the WorldState + PanDecision stream in order.
struct DirectorSink {
    log: DirectorLog,
    pending: Option<Option<TrackedEntity>>,
}

impl crate::detect::pipeline_event::PipelineEventSink for DirectorSink {
    fn emit(&mut self, event: crate::detect::pipeline_event::PipelineEvent) {
        use crate::detect::pipeline_event::PipelineEvent as E;
        match event {
            E::WorldState { ball, .. } => self.pending = Some(ball),
            E::PanDecision { pose, .. } => {
                let ball = self
                    .pending
                    .take()
                    .expect("world_state before pan_decision");
                self.log.lock().unwrap().push((ball, pose));
            }
            _ => {}
        }
    }
}

fn assert_offline_matches_live(frames: u64, lookahead: usize) {
    let fps_rational = (60000, 1001);
    let fps = 60000.0 / 1001.0;
    let detector = SlowDetector {
        delay: std::time::Duration::from_millis(0),
    };
    let mut session = build_test_session(None, Some(Box::new(detector)), 1).expect("session build");
    session.set_ball_tracker(Box::new(ScriptedBallTracker { calls: 0 }));
    session.set_panner(Box::new(ProbePanner));
    let log = Arc::new(Mutex::new(Vec::new()));
    session.set_event_sink(Box::new(DirectorSink {
        log: Arc::clone(&log),
        pending: None,
    }));
    session.set_lookahead(lookahead);
    let mut source = TimedSource {
        remaining: frames,
        fps,
        fps_rational: Some(fps_rational),
    };
    let interrupted = AtomicBool::new(false);
    session
        .run(&mut source, u64::MAX, &interrupted, None)
        .expect("run");
    session.finish().expect("finish");
    let live = log.lock().unwrap().clone();

    let offline_frames: Vec<super::OfflineFrame> = (0..frames)
        .map(|i| super::OfflineFrame {
            source_frame_index: i,
            elapsed_ms: super::run_loop::media_elapsed(i, (60000, 1001)).as_secs_f64() * 1000.0,
            world: WorldState {
                ball: scripted_ball(i),
                players: vec![],
            },
        })
        .collect();
    let cal = test_calibration();
    let offline = super::replay_dense_lookahead(
        &offline_frames,
        &mut ProbePanner,
        lookahead,
        fps,
        &cal,
        ViewportPosition::default(),
    );

    assert_eq!(live.len(), frames as usize, "live rendered frame count");
    assert_eq!(offline.len(), live.len(), "offline frame count");
    for (k, ((live_ball, live_pose), off)) in live.iter().zip(&offline).enumerate() {
        assert_eq!(off.source_frame_index, k as u64, "render order");
        assert_eq!(
            format!("{live_ball:?}"),
            format!("{:?}", off.world.ball),
            "bridged ball differs at frame {k}"
        );
        assert_eq!(
            (
                live_pose.yaw.to_bits(),
                live_pose.pitch.to_bits(),
                live_pose.fov_degrees.map(f32::to_bits)
            ),
            (
                off.smoothed.yaw.to_bits(),
                off.smoothed.pitch.to_bits(),
                off.smoothed.fov_degrees.map(f32::to_bits)
            ),
            "smoothed pose differs at frame {k}: live {live_pose:?} offline {:?}",
            off.smoothed
        );
    }
    // The script must actually exercise bridging, or the test proves less.
    assert!(
        frames < 12
            || offline
                .iter()
                .any(|p| p.world.ball.is_some_and(|b| b.state == TrackState::Bridged)),
        "script produced no bridged frames"
    );
}

#[test]
#[ignore] // requires GPU (runs on lavapipe)
fn offline_director_matches_live_lookahead_6() {
    assert_offline_matches_live(48, 6);
}

#[test]
#[ignore] // requires GPU (runs on lavapipe)
fn offline_director_matches_live_lookahead_90() {
    // Harness setting: --lookahead 1.5 at 59.94 fps -> 90 frames.
    assert_offline_matches_live(240, 90);
}

#[test]
#[ignore] // requires GPU (runs on lavapipe)
fn offline_director_matches_live_lookahead_1_and_short_clip() {
    assert_offline_matches_live(30, 1);
    assert_offline_matches_live(4, 6);
}
