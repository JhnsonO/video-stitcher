//! Linux CUDA/Vulkan zero-copy types.
//!
//! Contains [`SharedTextureSet`], the bundle of double-buffered shared
//! textures + CUDA pointers + slot channels used by the GPU zero-copy
//! decode path. Platform-specific session methods that operate on these
//! textures live in `run_loop.rs` (`setup_gpu_source`, `step_gpu_with_bufs`).

use crate::interop::vulkan::{SharedBuffer, SharedTexture};
use crate::interop::zero_copy::GpuBufInfo;

/// Bundled shared textures, CUDA buffer info, slot channels, and bind
/// groups for the Linux CUDA/Vulkan zero-copy path.
///
/// Constructed by the source (e.g. `SmartFileSource`), consumed by
/// `StitchSession::setup_gpu_source` + `run`. The caller must pass
/// `left_buf` / `right_buf` and the slot-free receivers to the decode
/// thread spawner, then pass this struct (minus the receivers) to the
/// session.
pub struct SharedTextureSet {
    /// The 8 shared textures: [left_y_0, left_uv_0, left_y_1, left_uv_1,
    /// right_y_0, right_uv_0, right_y_1, right_uv_1].
    /// Must be dropped after decode threads are joined.
    pub textures: [SharedTexture; 8],
    /// The 8 CUDA-shared buffers matching `textures`' plane/slot order.
    /// Decode writes target these buffers; Vulkan copies them into normal
    /// VramPool textures before rendering.
    pub buffers: [SharedBuffer; 8],
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
    /// Pre-built bind groups for the shared textures.
    /// `None` when the source creates textures without pipeline access
    /// (e.g. `SmartFileSource`). The session creates them lazily at
    /// the start of `run()`.
    pub bind_groups: Option<crate::render::pipeline::GpuSourceBindGroups>,
    /// The 4 exportable `VkSemaphore`s for the diagnostic CUDA->Vulkan
    /// sync (Ticket 2), one per double-buffer slot: `[left_slot_0,
    /// left_slot_1, right_slot_0, right_slot_1]` (Y+UV share one
    /// semaphore per slot). CUDA holds the imported/signaling side
    /// (`GpuBufInfo::sem_cuda`); the session waits on these via
    /// `add_wait_semaphore` in `copy_to_vram_pool_platform` before
    /// reading the shared textures. Owned here (not `SharedTexture`)
    /// since a semaphore, unlike a texture, isn't tied to one plane.
    pub vk_semaphores: [ash::vk::Semaphore; 4],
}
