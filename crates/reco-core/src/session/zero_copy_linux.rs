//! Linux CUDA/Vulkan zero-copy types.
//!
//! Contains [`SharedTextureSet`], the bundle of double-buffered shared
//! buffers, external semaphores, CUDA pointers, and slot channels used by
//! the GPU zero-copy decode path.

use crate::interop::vulkan::{SharedBuffer, SharedSemaphore};
use crate::interop::zero_copy::GpuBufInfo;

/// Bundled shared buffers, synchronization, CUDA metadata, and slot channels
/// for the Linux CUDA/Vulkan zero-copy path.
///
/// Constructed by the source (e.g. `SmartFileSource`), consumed by
/// `StitchSession::setup_gpu_source` + `run`. The caller must pass
/// `left_buf` / `right_buf` and the slot-free receivers to the decode
/// thread spawner, then pass this struct (minus the receivers) to the
/// session.
pub struct SharedTextureSet {
    /// The 8 shared buffers: [left_y_0, left_uv_0, left_y_1, left_uv_1,
    /// right_y_0, right_uv_0, right_y_1, right_uv_1].
    /// Must be dropped after decode threads are joined.
    pub buffers: [SharedBuffer; 8],
    /// Four binary semaphores: [left_0, left_1, right_0, right_1]. One
    /// semaphore covers both planes in its decode slot.
    pub semaphores: [SharedSemaphore; 4],
    /// CUDA buffer info for left camera decode thread.
    pub left_buf: GpuBufInfo,
    /// CUDA buffer info for right camera decode thread.
    pub right_buf: GpuBufInfo,
    /// Slot-free sender for left camera (decode backpressure).
    pub left_slot_free_tx: std::sync::mpsc::SyncSender<u8>,
    /// Slot-free sender for right camera (decode backpressure).
    pub right_slot_free_tx: std::sync::mpsc::SyncSender<u8>,
    /// Slot-free receiver for left camera. Taken by decode thread spawner.
    pub left_slot_free_rx: Option<std::sync::mpsc::Receiver<u8>>,
    /// Slot-free receiver for right camera. Taken by decode thread spawner.
    pub right_slot_free_rx: Option<std::sync::mpsc::Receiver<u8>>,
}
