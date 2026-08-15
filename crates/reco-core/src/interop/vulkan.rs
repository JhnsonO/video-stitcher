//! Vulkan side of CUDA/Vulkan interop.
//!
//! Imports CUDA-exported shared memory into Vulkan, then wraps the
//! resulting `VkImage` into a [`wgpu::Texture`] via the HAL escape hatch.
//!
//! The flow:
//! 1. CUDA allocates shareable memory and exports a POSIX fd
//! 2. This module creates a `VkImage` with `VK_KHR_external_memory_fd`
//! 3. Imports the fd as the backing memory for the image
//! 4. Wraps the `VkImage` into `wgpu::Texture` via `create_texture_from_hal`
//!
//! ## References
//! - [Gyroflow](https://github.com/gyroflow/gyroflow) for the general CUDA/Vulkan interop approach
//! - `VK_KHR_external_memory_fd` specification
//! - wgpu HAL interop API (`texture_from_raw`, `create_texture_from_hal`)

use crate::gpu::GpuContext;
use crate::interop::cuda::{CudaInteropError, CudaSharedMemory};

/// A wgpu texture backed by CUDA shared memory.
///
/// Owns both the wgpu texture and the underlying CUDA allocation.
/// When dropped, the wgpu texture is released first, then the CUDA memory.
pub struct SharedTexture {
    /// The wgpu texture, usable in bind groups and render passes.
    pub texture: wgpu::Texture,
    /// The CUDA device pointer to the shared memory.
    /// Used for `cuMemcpy2D` from NVDEC output to this texture.
    pub cuda_ptr: crate::interop::cuda::CUdeviceptr,
    /// Pitch (row stride in bytes) of the Vulkan image.
    /// May differ from `width * bpp` due to alignment requirements.
    pub pitch: usize,
    /// Keep the shared memory alive (dropped after texture).
    _shared_mem: CudaSharedMemory,
}

/// A wgpu buffer backed by CUDA VMM memory imported through OPAQUE_FD.
///
/// CUDA writes decoded plane bytes through [`cuda_ptr`](Self::cuda_ptr).
/// Vulkan/wgpu reads the same allocation as a copy source and transfers it
/// into an ordinary texture; the buffer is never mapped by wgpu.
pub struct SharedBuffer {
    /// The imported buffer, usable as a wgpu `COPY_SRC`.
    pub buffer: wgpu::Buffer,
    /// CUDA device pointer for the decode thread's `cuMemcpy2D` destination.
    pub cuda_ptr: crate::interop::cuda::CUdeviceptr,
    /// Row stride in bytes. Always aligned to wgpu's buffer-copy requirement.
    pub pitch: usize,
    /// Logical plane size in bytes (`pitch * height`).
    pub size: usize,
    /// Keep the CUDA VMM allocation alive until after the wgpu handle drops.
    _shared_mem: CudaSharedMemory,
}

/// Create a CUDA-VMM allocation shared with Vulkan as a `VkBuffer` and wrap
/// it as a wgpu copy source.
///
/// This deliberately follows the byte-exact EXP7 control: CUDA creates and
/// exports the allocation, Vulkan imports the OPAQUE_FD into a buffer using
/// the buffer's exact `memRequirements.size`, and wgpu owns the imported
/// `VkBuffer`/`VkDeviceMemory` lifetime. `vkGetMemoryFdPropertiesKHR` is not
/// valid for OPAQUE_FD and must not be added here.
#[cfg(target_os = "linux")]
pub fn create_shared_buffer(
    gpu: &GpuContext,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
) -> Result<SharedBuffer, CudaInteropError> {
    use ash::vk;
    use wgpu::hal::api::Vulkan;

    let bytes_per_texel = format_bytes_per_pixel(format);
    let row_bytes = width as usize * bytes_per_texel;
    let pitch = row_bytes.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let size = pitch * height as usize;

    let shared_mem = crate::interop::cuda::allocate_shared_memory(size)?;
    let cuda_ptr = shared_mem.device_ptr;
    let fd = shared_mem.shared_handle;

    let (vk_buffer, device_memory, memory_size) = unsafe {
        let hal_device_guard = gpu
            .device
            .as_hal::<Vulkan>()
            .ok_or(CudaInteropError::NotVulkan)?;
        let hal_device = &*hal_device_guard;
        let raw_device = hal_device.raw_device();
        let physical_device = hal_device.raw_physical_device();
        let instance = hal_device.shared_instance().raw_instance();

        let mut external_info = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external_info);
        let vk_buffer = match raw_device.create_buffer(&buffer_info, None) {
            Ok(buffer) => buffer,
            Err(e) => {
                libc::close(fd);
                return Err(CudaInteropError::VulkanError(format!(
                    "vkCreateBuffer ({label}): {e:?}"
                )));
            }
        };

        let mem_reqs = raw_device.get_buffer_memory_requirements(vk_buffer);
        let mem_props = instance.get_physical_device_memory_properties(physical_device);
        let pick_memory_type = |required: vk::MemoryPropertyFlags| {
            (0..mem_props.memory_type_count).find(|&i| {
                (mem_reqs.memory_type_bits & (1 << i)) != 0
                    && mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(required)
            })
        };
        let Some(memory_type_index) = pick_memory_type(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .or_else(|| pick_memory_type(vk::MemoryPropertyFlags::empty()))
        else {
            raw_device.destroy_buffer(vk_buffer, None);
            libc::close(fd);
            return Err(CudaInteropError::VulkanError(format!(
                "no compatible memory type for imported buffer {label}"
            )));
        };

        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD)
            .fd(fd);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_reqs.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut import_info);
        let device_memory = match raw_device.allocate_memory(&alloc_info, None) {
            Ok(memory) => memory,
            Err(e) => {
                // Vulkan consumes the imported fd only when allocation succeeds.
                raw_device.destroy_buffer(vk_buffer, None);
                libc::close(fd);
                return Err(CudaInteropError::VulkanError(format!(
                    "vkAllocateMemory ({label} OPAQUE_FD): {e:?}"
                )));
            }
        };

        if let Err(e) = raw_device.bind_buffer_memory(vk_buffer, device_memory, 0) {
            raw_device.destroy_buffer(vk_buffer, None);
            raw_device.free_memory(device_memory, None);
            return Err(CudaInteropError::VulkanError(format!(
                "vkBindBufferMemory ({label}): {e:?}"
            )));
        }

        log::info!(
            "CUDA/Vulkan shared buffer {label}: {width}x{height} {format:?}, \
             row_bytes={row_bytes}, pitch={pitch}, logical_size={size}, \
             cuda_alloc_size={}, vk_mem_reqs_size={}, vk_alignment={}, \
             allocationSize_used={}, memory_type_index={memory_type_index}",
            shared_mem.alloc_size,
            mem_reqs.size,
            mem_reqs.alignment,
            mem_reqs.size,
        );

        (vk_buffer, device_memory, mem_reqs.size)
    };

    let wgpu_buffer = unsafe {
        let hal_device_guard = gpu
            .device
            .as_hal::<Vulkan>()
            .ok_or(CudaInteropError::NotVulkan)?;
        let hal_buffer =
            wgpu::hal::vulkan::Buffer::from_raw_managed(vk_buffer, device_memory, 0, memory_size);
        drop(hal_device_guard);
        gpu.device.create_buffer_from_hal::<Vulkan>(
            hal_buffer,
            &wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            },
        )
    };

    Ok(SharedBuffer {
        buffer: wgpu_buffer,
        cuda_ptr,
        pitch,
        size,
        _shared_mem: shared_mem,
    })
}

/// Create a wgpu texture backed by CUDA shared memory.
///
/// This is the main entry point for zero-copy interop. The returned texture
/// can be used in wgpu bind groups just like any other texture, but its memory
/// is shared with CUDA — writes via `cuMemcpy2D` are visible to wgpu.
///
/// # Arguments
/// - `gpu`: the wgpu GPU context (must be Vulkan backend)
/// - `width`, `height`: texture dimensions in pixels
/// - `format`: wgpu texture format (e.g. `R8Unorm` for Y/U/V planes)
///
/// # Errors
/// - `NotVulkan` if the wgpu backend is not Vulkan
/// - `CudaError` if shared memory allocation fails
/// - `VulkanError` if Vulkan image creation or memory import fails
#[cfg(target_os = "linux")]
pub fn create_shared_texture(
    gpu: &GpuContext,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> Result<SharedTexture, CudaInteropError> {
    use ash::vk;
    use wgpu::hal::api::Vulkan;

    // ZC_EXP7 (diagnostic-only, avenue 3): run the VkBuffer alias control
    // exactly once, before any decode work, on the first shared-texture
    // creation. Self-contained -- allocates, writes, reads and frees its
    // own resources; depends on nothing but `gpu`.
    zc_exp7_run_buffer_control_once(gpu);

    let bpp = format_bytes_per_pixel(format);

    // Allocate row-aligned: Vulkan may require specific row pitch alignment.
    // Start with a generous pitch (aligned to 256 bytes, common GPU requirement).
    let row_bytes = width as usize * bpp;
    let pitch = (row_bytes + 255) & !255; // align to 256
    let alloc_size = pitch * height as usize;

    let shared_mem = crate::interop::cuda::allocate_shared_memory(alloc_size)?;
    let cuda_ptr = shared_mem.device_ptr;
    let fd = shared_mem.shared_handle;

    // Access the raw Vulkan device through wgpu's HAL
    let (vk_image, device_memory, actual_pitch) = unsafe {
        let hal_device_guard = gpu
            .device
            .as_hal::<Vulkan>()
            .ok_or(CudaInteropError::NotVulkan)?;
        let hal_device = &*hal_device_guard;
        let raw_device = hal_device.raw_device();
        let physical_device = hal_device.raw_physical_device();
        let vk_format = wgpu_format_to_vk(format);

        // --- ZC_EXP7 (diagnostic-only, avenue 3): image capability query ---
        // Asks the driver whether THIS EXACT external image configuration is
        // importable from OPAQUE_FD, and whether it demands a dedicated
        // allocation -- rather than inferring it from vkAllocateMemory
        // returning VK_SUCCESS, which run 31854089581 showed is not evidence
        // of real page sharing. Read-only, and fully gated: with ZC_EXP7
        // unset there is no behavioural change whatsoever.
        if zc_exp7_enabled() {
            let instance = hal_device.shared_instance().raw_instance();
            let api_version = hal_device.shared_instance().instance_api_version();
            if zc_exp7_vulkan_11_available(api_version) {
                zc_exp7_image_caps(
                    instance,
                    physical_device,
                    vk_format,
                    vk::ImageTiling::LINEAR,
                    vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::SAMPLED,
                    vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD,
                    width,
                    height,
                    format,
                );
            }
        }
        // --- end ZC_EXP7 image capability query ---

        // Create VkImage with external memory support
        let mut external_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::LINEAR)
            .usage(
                vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::PREINITIALIZED)
            .push_next(&mut external_info);

        let vk_image = raw_device
            .create_image(&image_info, None)
            .map_err(|e| CudaInteropError::VulkanError(format!("vkCreateImage: {e:?}")))?;

        // Get memory requirements
        let mem_reqs = raw_device.get_image_memory_requirements(vk_image);

        // --- ZC_EXP7 (diagnostic-only): requirements2 + dedicated reqs ---
        // Additive. The v1 call above is deliberately left in place so the
        // existing ZC_DIAG line stays byte-comparable with prior runs.
        // Reports requiresDedicatedAllocation/prefersDedicatedAllocation --
        // never queried anywhere in this codebase before -- and measures
        // shared_mem.alloc_size against mem_reqs2.size for EXACT equality.
        // The existing ZC_DIAG only ever asserted `>=`; NVIDIA's own
        // simpleVulkanMMAP passes memRequirements.size verbatim as
        // allocationSize, so the delta has never actually been measured.
        if zc_exp7_enabled()
            && zc_exp7_vulkan_11_available(hal_device.shared_instance().instance_api_version())
        {
            zc_exp7_mem_reqs2(raw_device, vk_image, shared_mem.alloc_size, format);
        }
        // --- end ZC_EXP7 requirements2 ---

        // Get actual row pitch from the image layout
        let subresource = vk::ImageSubresource {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            array_layer: 0,
        };
        let layout = raw_device.get_image_subresource_layout(vk_image, subresource);
        let actual_pitch = layout.row_pitch as usize;

        // --- ZC_DIAG: diagnostic-only, no behavioral change ---
        // Narrowed hypothesis check (per Johnson, 14 Aug 2026): pitch already
        // confirmed matching on real footage (run 31557269688 logged
        // actual_pitch=3840 for both Y/UV, same as the guessed pitch here).
        // The only open question is whether shared_mem.alloc_size (after CUDA
        // VMM granularity rounding) actually satisfies Vulkan's mem_reqs.size.
        let mem_reqs_size = mem_reqs.size as usize;
        let mem_reqs_alignment = mem_reqs.alignment as usize;
        let actual_pitch_x_height = actual_pitch * height as usize;
        let pass_vs_mem_reqs = shared_mem.alloc_size >= mem_reqs_size;
        let pass_vs_pitch_x_height = shared_mem.alloc_size >= actual_pitch_x_height;
        log::info!(
            "ZC_DIAG: {}x{} {:?} bpp={} row_bytes={} guessed_pitch={} requested_alloc_size={} shared_mem_alloc_size={} vk_actual_pitch={} vk_mem_reqs_size={} vk_mem_reqs_alignment={} actual_pitch_x_height={} PASS_vs_mem_reqs={} PASS_vs_pitch_x_height={} vk_subresource_offset={} vk_subresource_size={}",
            width,
            height,
            format,
            bpp,
            row_bytes,
            pitch,
            alloc_size,
            shared_mem.alloc_size,
            actual_pitch,
            mem_reqs_size,
            mem_reqs_alignment,
            actual_pitch_x_height,
            if pass_vs_mem_reqs { "PASS" } else { "FAIL" },
            if pass_vs_pitch_x_height {
                "PASS"
            } else {
                "FAIL"
            },
            layout.offset,
            layout.size,
        );
        // --- end ZC_DIAG ---

        // Find a DEVICE_LOCAL memory type
        let mem_props = {
            let instance = hal_device.shared_instance().raw_instance();
            instance.get_physical_device_memory_properties(physical_device)
        };

        // Prefer DEVICE_LOCAL, but fall back to any supported memory type.
        // On unified-memory GPUs (Jetson/Tegra) the driver may not flag
        // imported-fd-compatible types as DEVICE_LOCAL.
        let memory_type_index = (0..mem_props.memory_type_count)
            .find(|&i| {
                let type_bits = 1 << i;
                let is_supported = (mem_reqs.memory_type_bits & type_bits) != 0;
                let props = mem_props.memory_types[i as usize].property_flags;
                is_supported && props.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .or_else(|| {
                log::warn!(
                    "No DEVICE_LOCAL memory type for imported image, \
                         falling back to any supported type (unified memory GPU?)"
                );
                (0..mem_props.memory_type_count)
                    .find(|&i| (mem_reqs.memory_type_bits & (1 << i)) != 0)
            })
            .ok_or_else(|| {
                CudaInteropError::VulkanError("no compatible memory type for imported image".into())
            })?;

        // Import the CUDA fd as Vulkan memory
        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD)
            .fd(fd);

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(shared_mem.alloc_size as u64)
            .memory_type_index(memory_type_index)
            .push_next(&mut import_info);

        let device_memory = raw_device.allocate_memory(&alloc_info, None).map_err(|e| {
            CudaInteropError::VulkanError(format!("vkAllocateMemory (import fd): {e:?}"))
        })?;

        // Bind the imported memory to the image
        raw_device
            .bind_image_memory(vk_image, device_memory, 0)
            .map_err(|e| CudaInteropError::VulkanError(format!("vkBindImageMemory: {e:?}")))?;

        // --- ZC_EXP5 (diagnostic): one-time PREINITIALIZED -> ---
        // --- TRANSFER_SRC_OPTIMAL layout transition ---
        //
        // The image is created with initial_layout = PREINITIALIZED
        // (above). Left alone, wgpu's create_texture_from_hal previously
        // hard-coded TextureUses::UNINITIALIZED for every imported HAL
        // texture -- and per the Vulkan spec, a transition FROM
        // UNDEFINED/PREINITIALIZED is legally allowed to discard existing
        // contents on first use. That's the leading hypothesis for why
        // Vulkan's own readback of this texture has shown all-zero bytes
        // even when CUDA's own view of the same shared memory is correct
        // (see docs/ai-project-state.md, "zero-copy NV12 corruption").
        //
        // This performs the transition ourselves, once, at texture
        // creation time -- before any CUDA write has happened and before
        // wgpu ever sees the image -- so that when create_texture_from_hal
        // is told (a few lines below, outside this block) that the
        // texture's initial_state is COPY_SRC, that claim is actually
        // true of the real VkImageLayout, not just of wgpu's tracker.
        //
        // This does NOT synchronize against CUDA's later per-frame
        // writes to the same shared memory -- that's a separate, already
        // scoped concern (Ticket 2's external-semaphore design). This is
        // strictly a one-time "stop wgpu from treating fresh memory as
        // discardable" fix, submitted on the exact queue wgpu itself
        // uses (via Queue::as_hal, mirroring the proven pattern already
        // used for add_wait_semaphore below in this same file) so there
        // is no cross-queue-family hazard.
        {
            let (raw_queue, queue_family_index) = gpu
                .queue
                .as_hal::<Vulkan>()
                .map(|hal_queue| (hal_queue.as_raw(), hal_queue.family_index()))
                .ok_or(CudaInteropError::NotVulkan)?;

            let pool_info =
                vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index);
            let cmd_pool = raw_device
                .create_command_pool(&pool_info, None)
                .map_err(|e| {
                    CudaInteropError::VulkanError(format!(
                        "vkCreateCommandPool (ZC_EXP5 transition): {e:?}"
                    ))
                })?;

            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(cmd_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let cmd_buf = raw_device
                .allocate_command_buffers(&alloc_info)
                .map_err(|e| {
                    CudaInteropError::VulkanError(format!(
                        "vkAllocateCommandBuffers (ZC_EXP5 transition): {e:?}"
                    ))
                })?[0];

            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            raw_device
                .begin_command_buffer(cmd_buf, &begin_info)
                .map_err(|e| {
                    CudaInteropError::VulkanError(format!(
                        "vkBeginCommandBuffer (ZC_EXP5 transition): {e:?}"
                    ))
                })?;

            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::PREINITIALIZED)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(vk_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                // There is no real prior GPU-visible write to synchronize
                // against yet (the image is freshly allocated/bound, CUDA
                // hasn't written anything through it), so HOST_WRITE is
                // the conventional (if imperfect) srcAccessMask for a
                // transition away from PREINITIALIZED -- see Vulkan spec
                // 11.4, "Layout Transitions".
                .src_access_mask(vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);

            raw_device.cmd_pipeline_barrier(
                cmd_buf,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier),
            );

            raw_device.end_command_buffer(cmd_buf).map_err(|e| {
                CudaInteropError::VulkanError(format!(
                    "vkEndCommandBuffer (ZC_EXP5 transition): {e:?}"
                ))
            })?;

            let fence = raw_device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(|e| {
                    CudaInteropError::VulkanError(format!(
                        "vkCreateFence (ZC_EXP5 transition): {e:?}"
                    ))
                })?;

            let submit_info =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd_buf));
            raw_device
                .queue_submit(raw_queue, std::slice::from_ref(&submit_info), fence)
                .map_err(|e| {
                    CudaInteropError::VulkanError(format!(
                        "vkQueueSubmit (ZC_EXP5 transition): {e:?}"
                    ))
                })?;

            raw_device
                .wait_for_fences(&[fence], true, u64::MAX)
                .map_err(|e| {
                    CudaInteropError::VulkanError(format!(
                        "vkWaitForFences (ZC_EXP5 transition): {e:?}"
                    ))
                })?;

            // Destroying the pool implicitly frees the command buffer
            // allocated from it -- no separate free_command_buffers call.
            raw_device.destroy_fence(fence, None);
            raw_device.destroy_command_pool(cmd_pool, None);

            log::info!(
                "ZC_EXP5: transitioned imported VkImage PREINITIALIZED -> \
                 TRANSFER_SRC_OPTIMAL (one-time, before any CUDA write)"
            );
        }

        log::info!(
            "Vulkan image created: {}x{} {:?}, pitch={}, imported fd={}",
            width,
            height,
            format,
            actual_pitch,
            fd
        );

        (vk_image, device_memory, actual_pitch)
    };

    // Wrap the VkImage into a wgpu texture via HAL.
    //
    // We provide a drop_callback that destroys the VkImage and frees the
    // imported VkDeviceMemory. This way wgpu won't try to manage the memory
    // through its own allocator, and cleanup happens when the texture is dropped.
    // Build the drop callback outside the unsafe block to avoid nested-unsafe warning.
    // This closure is called by wgpu when the texture is dropped.
    let device_for_drop = gpu.device.clone();
    let drop_image = vk_image;
    let drop_memory = device_memory;
    let drop_callback: Box<dyn FnOnce() + Send + Sync> = Box::new(move || {
        // SAFETY: these Vulkan resources are no longer referenced after
        // the wgpu texture is dropped.
        unsafe {
            if let Some(hal_device) = device_for_drop.as_hal::<Vulkan>() {
                let raw = hal_device.raw_device();
                raw.destroy_image(drop_image, None);
                raw.free_memory(drop_memory, None);
            }
        }
    });

    // SAFETY: we've created a valid VkImage backed by imported CUDA memory,
    // and the drop_callback will clean up both when the texture is released.
    let wgpu_texture = unsafe {
        let hal_device_guard = gpu
            .device
            .as_hal::<Vulkan>()
            .ok_or(CudaInteropError::NotVulkan)?;

        let hal_texture = hal_device_guard.texture_from_raw(
            vk_image,
            &wgpu::hal::TextureDescriptor {
                label: Some("cuda_shared"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_DST,
                memory_flags: wgpu::hal::MemoryFlags::empty(),
                view_formats: vec![],
            },
            Some(drop_callback),
            wgpu::hal::vulkan::TextureMemory::External,
        );

        // Drop the HAL device guard before calling create_texture_from_hal
        drop(hal_device_guard);

        gpu.device.create_texture_from_hal::<Vulkan>(
            hal_texture,
            &wgpu::TextureDescriptor {
                label: Some("cuda_shared"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            },
            // ZC_EXP5 (diagnostic): tell wgpu the true state of this
            // imported texture instead of letting it assume UNINITIALIZED
            // (which, per Vulkan spec, permits wgpu to discard the
            // contents CUDA already wrote on first use -- see the
            // one-time PREINITIALIZED -> TRANSFER_SRC_OPTIMAL transition
            // performed above, immediately before this call). COPY_SRC
            // because the first real operation on this texture is always
            // a copy-out (zc_exp4_readback_texture's copy_texture_to_buffer,
            // then VramPool::copy_from_textures' copy_texture_to_texture)
            // -- confirmed by direct inspection of both call sites, this
            // texture is never sampled or copy-dst'd before being read.
            wgpu::TextureUses::COPY_SRC,
        )
    };

    // ZC_EXP7: optional abort-before-decode. `ZC_EXP7_ABORT_AFTER=<n>` exits
    // cleanly once n shared textures have been probed, so the diagnostic
    // never spends decode/render/encode time. n=2 covers one camera's Y+UV
    // pair plus the once-only buffer control -- everything this experiment
    // needs. Unset (the default) = no behavioural change.
    zc_exp7_maybe_abort();

    Ok(SharedTexture {
        texture: wgpu_texture,
        cuda_ptr,
        pitch: actual_pitch,
        _shared_mem: shared_mem,
    })
}

// ── ZC_EXP7 (diagnostic-only, avenue 3) ─────────────────────────────
//
// Avenue 2 (run 31854089581) established that the synchronized Vulkan
// IMAGE path failed to observe a CUDA-written sentinel that a CUDA-side
// readback confirmed byte-exact at the same offsets. That is NOT the same
// as proving CUDA and Vulkan are definitively on different physical pages
// -- an image-specific external-memory requirement or layout/semantics
// mismatch would produce the identical symptom. Distinguishing those two
// explanations is exactly what EXP7 exists to do. Every API call in the
// import path returns VK_SUCCESS, so success codes are known not to be
// evidence here.
//
// This block adds three things, all read-only or self-contained:
//   A. Does the driver actually declare this exact external IMAGE config
//      importable from OPAQUE_FD, and does it require a dedicated alloc?
//   B. Modern requirements2 path incl. VkMemoryDedicatedRequirements.
//   C. A runtime VkBuffer control mirroring NVIDIA's simpleVulkanMMAP --
//      the only official NVIDIA implementation of the CUDA-VMM-export ->
//      Vulkan-import direction, which imports into a VkBuffer, never a
//      VkImage, and passes memRequirements.size as allocationSize.
//
// Reference: NVIDIA/cuda-samples @ v11.8,
// Samples/5_Domain_Specific/simpleVulkanMMAP/VulkanBaseApp.cpp,
// importExternalBuffer(), lines ~1624-1676.
//
// Classification is pre-registered in docs/ai-project-state.md; do not
// re-derive it after seeing the numbers.

#[cfg(target_os = "linux")]
static ZC_EXP7_BUFFER_CONTROL_RAN: std::sync::Once = std::sync::Once::new();

#[cfg(target_os = "linux")]
static ZC_EXP7_TEXTURES_PROBED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// ZC_EXP7 master gate. With `ZC_EXP7` unset or != "1", every probe in this
/// block is skipped and behaviour is byte-identical to the base commit.
#[cfg(target_os = "linux")]
fn zc_exp7_enabled() -> bool {
    std::env::var("ZC_EXP7").as_deref() == Ok("1")
}

/// ZC_EXP7: `vkGetPhysicalDeviceImageFormatProperties2`,
/// `vkGetImageMemoryRequirements2` and
/// `vkGetPhysicalDeviceExternalBufferProperties` are Vulkan 1.1 CORE, not
/// extensions. On a 1.0 instance their function pointers are null and
/// calling them segfaults rather than returning an error, so the instance
/// API version is checked and logged before any of them is invoked.
#[cfg(target_os = "linux")]
fn zc_exp7_vulkan_11_available(api_version: u32) -> bool {
    use std::sync::atomic::{AtomicBool, Ordering};
    static LOGGED: AtomicBool = AtomicBool::new(false);

    let ok = api_version >= ash::vk::API_VERSION_1_1;
    if !LOGGED.swap(true, Ordering::SeqCst) {
        log::info!(
            "ZC_EXP7_API: instance_api_version={}.{}.{} (raw={api_version}) \
             vulkan_1_1_core_available={ok}{}",
            ash::vk::api_version_major(api_version),
            ash::vk::api_version_minor(api_version),
            ash::vk::api_version_patch(api_version),
            if ok {
                ""
            } else {
                " -- SKIPPING all ZC_EXP7 1.1-core queries; this run yields NO \
                 capability evidence and must not be classified"
            }
        );
    }
    ok
}

/// ZC_EXP7 step A: query external-memory capabilities for this exact image
/// configuration. Never fatal -- `VK_ERROR_FORMAT_NOT_SUPPORTED` is itself
/// a first-class result for this experiment, not an error to unwrap on.
#[cfg(target_os = "linux")]
#[allow(clippy::too_many_arguments)]
unsafe fn zc_exp7_image_caps(
    instance: &ash::Instance,
    physical_device: ash::vk::PhysicalDevice,
    vk_format: ash::vk::Format,
    tiling: ash::vk::ImageTiling,
    usage: ash::vk::ImageUsageFlags,
    handle_type: ash::vk::ExternalMemoryHandleTypeFlags,
    width: u32,
    height: u32,
    wgpu_format: wgpu::TextureFormat,
) {
    use ash::vk;

    let mut external_info =
        vk::PhysicalDeviceExternalImageFormatInfo::default().handle_type(handle_type);

    let format_info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(vk_format)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(tiling)
        .usage(usage)
        .flags(vk::ImageCreateFlags::empty())
        .push_next(&mut external_info);

    let mut external_props = vk::ExternalImageFormatProperties::default();
    let (query, image_format_properties) = {
        let mut props = vk::ImageFormatProperties2::default().push_next(&mut external_props);
        let query = unsafe {
            instance.get_physical_device_image_format_properties2(
                physical_device,
                &format_info,
                &mut props,
            )
        };
        (query, props.image_format_properties)
    };

    match query {
        Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => {
            log::info!(
                "ZC_EXP7_IMG: wgpu_format={wgpu_format:?} vk_format={vk_format:?} \
                 tiling={tiling:?} usage={usage:?} extent={width}x{height} \
                 requested_handle_type={handle_type:?} \
                 query_result=FORMAT_NOT_SUPPORTED \
                 -- driver declares this external image configuration \
                 unsupported; this is a RESULT, not a harness failure"
            );
            return;
        }
        Err(e) => {
            log::info!(
                "ZC_EXP7_IMG: wgpu_format={wgpu_format:?} vk_format={vk_format:?} \
                 tiling={tiling:?} usage={usage:?} extent={width}x{height} \
                 requested_handle_type={handle_type:?} query_result=ERROR({e:?})"
            );
            return;
        }
        Ok(()) => {}
    }

    let features = external_props
        .external_memory_properties
        .external_memory_features;
    let exportable = features.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE);
    let importable = features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE);
    let dedicated_only = features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY);
    let compatible = external_props
        .external_memory_properties
        .compatible_handle_types;
    let export_from_imported = external_props
        .external_memory_properties
        .export_from_imported_handle_types;
    let opaque_fd_compatible = compatible.contains(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

    log::info!(
        "ZC_EXP7_IMG: wgpu_format={wgpu_format:?} vk_format={vk_format:?} tiling={tiling:?} \
         usage={usage:?} extent={width}x{height} requested_handle_type={handle_type:?} \
         query_result=SUCCESS externalMemoryFeatures_raw={:#x} EXPORTABLE={exportable} \
         IMPORTABLE={importable} DEDICATED_ONLY={dedicated_only} \
         compatibleHandleTypes={compatible:?} opaque_fd_in_compatible={opaque_fd_compatible} \
         exportFromImportedHandleTypes={export_from_imported:?} maxExtent={}x{}x{} \
         maxMipLevels={} maxArrayLayers={}",
        features.as_raw(),
        image_format_properties.max_extent.width,
        image_format_properties.max_extent.height,
        image_format_properties.max_extent.depth,
        image_format_properties.max_mip_levels,
        image_format_properties.max_array_layers,
    );
}

/// ZC_EXP7 step B: `vkGetImageMemoryRequirements2` +
/// `VkMemoryDedicatedRequirements`. Additive -- the v1 call at the real call
/// site is left untouched so the existing ZC_DIAG line stays comparable.
#[cfg(target_os = "linux")]
unsafe fn zc_exp7_mem_reqs2(
    raw_device: &ash::Device,
    vk_image: ash::vk::Image,
    cuda_alloc_size: usize,
    wgpu_format: wgpu::TextureFormat,
) {
    use ash::vk;

    let info = vk::ImageMemoryRequirementsInfo2::default().image(vk_image);
    let mut dedicated = vk::MemoryDedicatedRequirements::default();
    let memory_requirements = {
        let mut reqs2 = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        unsafe {
            raw_device.get_image_memory_requirements2(&info, &mut reqs2);
        }
        reqs2.memory_requirements
    };

    let size = memory_requirements.size as usize;
    let size_exact_match = cuda_alloc_size == size;

    log::info!(
        "ZC_EXP7_REQ2: wgpu_format={wgpu_format:?} \
         requiresDedicatedAllocation={} prefersDedicatedAllocation={} \
         size={size} alignment={} memoryTypeBits={:#x} \
         cuda_alloc_size={cuda_alloc_size} size_exact_match={size_exact_match}",
        dedicated.requires_dedicated_allocation != 0,
        dedicated.prefers_dedicated_allocation != 0,
        memory_requirements.alignment,
        memory_requirements.memory_type_bits,
    );
}

/// ZC_EXP7: exit cleanly once `ZC_EXP7_ABORT_AFTER=<n>` shared textures have
/// been probed, so the diagnostic never reaches decode/render/encode.
/// Unset = no behavioural change.
#[cfg(target_os = "linux")]
fn zc_exp7_maybe_abort() {
    use std::sync::atomic::Ordering;

    let Ok(raw) = std::env::var("ZC_EXP7_ABORT_AFTER") else {
        return;
    };
    if !zc_exp7_enabled() {
        log::warn!(
            "ZC_EXP7_ABORT_AFTER is set but ZC_EXP7 != 1 -- ignoring, since no \
             EXP7 evidence is being collected"
        );
        return;
    }
    let Ok(limit) = raw.trim().parse::<usize>() else {
        log::warn!("ZC_EXP7_ABORT_AFTER={raw:?} is not a number; ignoring");
        return;
    };

    let probed = ZC_EXP7_TEXTURES_PROBED.fetch_add(1, Ordering::SeqCst) + 1;
    if probed < limit {
        return;
    }

    log::info!(
        "ZC_EXP7_COMPLETE: probed {probed} shared texture(s) (limit={limit}); \
         exiting before decode. All ZC_EXP7_IMG / ZC_EXP7_REQ2 / \
         ZC_EXP7_BUF_* evidence above is complete."
    );
    // Flush the logger before exit -- process::exit does not run destructors.
    log::logger().flush();
    std::process::exit(0);
}

/// ZC_EXP7 step C: run the VkBuffer alias control exactly once.
#[cfg(target_os = "linux")]
fn zc_exp7_run_buffer_control_once(gpu: &GpuContext) {
    if !zc_exp7_enabled() {
        return;
    }
    ZC_EXP7_BUFFER_CONTROL_RAN.call_once(|| match zc_exp7_buffer_alias_control(gpu) {
        Ok(()) => {}
        Err(e) => {
            // A harness failure is NOT a falsified hypothesis. Say so
            // explicitly so the result cannot be misread as "buffer FAIL".
            log::error!(
                "ZC_EXP7_BUF_ALIAS_RESULT=HARNESS_ERROR ({e:?}) -- the control \
                 itself failed to run to completion. This is NOT evidence \
                 about buffer aliasing; fix the harness and rerun before \
                 interpreting anything."
            );
        }
    });
}

/// ZC_EXP7 step C: CUDA-VMM-export -> Vulkan-VkBuffer-import sentinel alias
/// control, mirroring NVIDIA's simpleVulkanMMAP `importExternalBuffer()`.
///
/// Uses the SAME `allocate_shared_memory()` as production (same `cuMemCreate`
/// + `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` + `cuMemExportToShareableHandle`)
/// and the SAME sentinel generator as the avenue-2 image test
/// (`(j as u8) ^ 0xC3`), so the two runs are directly comparable. The only
/// deliberate variable is buffer-vs-image (and, following NVIDIA,
/// `memRequirements.size` as `allocationSize`).
///
/// Self-contained: allocates, writes, reads and frees everything it touches.
#[cfg(target_os = "linux")]
fn zc_exp7_buffer_alias_control(gpu: &GpuContext) -> Result<(), CudaInteropError> {
    use ash::vk;
    use wgpu::hal::api::Vulkan;

    const CONTROL_SIZE: usize = 1024 * 1024;
    const SENTINEL_LEN: usize = 1024;
    const OFFSETS: [usize; 3] = [0, 524_288, 1_047_552];

    // Same generator as the avenue-2 image sentinel (run 31854089581).
    let sentinel: Vec<u8> = (0..SENTINEL_LEN).map(|j| (j as u8) ^ 0xC3).collect();

    crate::interop::cuda::cuda_ensure_context()?;
    let shared_mem = crate::interop::cuda::allocate_shared_memory(CONTROL_SIZE)?;
    let cuda_ptr = shared_mem.device_ptr;
    let fd = shared_mem.shared_handle;

    log::info!(
        "ZC_EXP7_BUF: allocated via the production allocate_shared_memory() path: \
         requested={CONTROL_SIZE} cuda_alloc_size={} device_ptr=0x{cuda_ptr:x} fd={fd}",
        shared_mem.alloc_size,
    );

    unsafe {
        let hal_device_guard = gpu
            .device
            .as_hal::<Vulkan>()
            .ok_or(CudaInteropError::NotVulkan)?;
        let hal_device = &*hal_device_guard;
        let raw_device = hal_device.raw_device();
        let physical_device = hal_device.raw_physical_device();
        let instance = hal_device.shared_instance().raw_instance();

        let buffer_usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;

        if !zc_exp7_vulkan_11_available(hal_device.shared_instance().instance_api_version()) {
            log::warn!(
                "ZC_EXP7_BUF_ALIAS_RESULT=SKIPPED_NO_VK11 -- \
                 vkGetPhysicalDeviceExternalBufferProperties is Vulkan 1.1 core and \
                 is unavailable on this instance. No aliasing evidence collected."
            );
            libc::close(fd);
            return Ok(());
        }

        // --- Buffer capability query (the buffer-side twin of step A) ---
        // Gate: a DEDICATED_ONLY result means a plain (non-dedicated)
        // allocation is not a legal way to back this buffer, so a byte
        // mismatch afterwards would say nothing about aliasing. Classify
        // and stop instead of producing a fake FAIL.
        {
            let mut ext_buf_info = vk::PhysicalDeviceExternalBufferInfo::default()
                .usage(buffer_usage)
                .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
            ext_buf_info.flags = vk::BufferCreateFlags::empty();

            let mut ext_buf_props = vk::ExternalBufferProperties::default();
            instance.get_physical_device_external_buffer_properties(
                physical_device,
                &ext_buf_info,
                &mut ext_buf_props,
            );

            let features = ext_buf_props
                .external_memory_properties
                .external_memory_features;
            let importable = features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE);
            let dedicated_only = features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY);
            let compatible = ext_buf_props
                .external_memory_properties
                .compatible_handle_types;
            let opaque_fd_compatible =
                compatible.contains(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

            log::info!(
                "ZC_EXP7_BUF_CAPS: usage={buffer_usage:?} \
                 requested_handle_type=OPAQUE_FD externalMemoryFeatures_raw={:#x} \
                 EXPORTABLE={} IMPORTABLE={importable} DEDICATED_ONLY={dedicated_only} \
                 compatibleHandleTypes={compatible:?} \
                 opaque_fd_in_compatible={opaque_fd_compatible} \
                 exportFromImportedHandleTypes={:?}",
                features.as_raw(),
                features.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE),
                ext_buf_props
                    .external_memory_properties
                    .export_from_imported_handle_types,
            );

            if dedicated_only || !importable || !opaque_fd_compatible {
                log::warn!(
                    "ZC_EXP7_BUF_ALIAS_RESULT=CONFIG_UNSUPPORTED \
                     (IMPORTABLE={importable} DEDICATED_ONLY={dedicated_only} \
                     opaque_fd_in_compatible={opaque_fd_compatible}) -- a plain \
                     non-dedicated import is not a legal backing for this buffer, \
                     so the control is NOT run. This is a configuration result, \
                     NOT a buffer aliasing failure, and must not be classified as \
                     'buffer FAIL'. Next step is a VkMemoryDedicatedAllocateInfo \
                     variant of this control, not a production import change."
                );
                // Nothing Vulkan-side has been created yet, and no import
                // took ownership of the fd, so close it here to avoid a leak.
                libc::close(fd);
                return Ok(());
            }
        }

        // --- Create + import the shared buffer (NVIDIA's path) ---
        let mut external_buffer_info = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

        let buffer_info = vk::BufferCreateInfo::default()
            .size(CONTROL_SIZE as u64)
            .usage(buffer_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external_buffer_info);

        let shared_buffer = raw_device
            .create_buffer(&buffer_info, None)
            .map_err(|e| CudaInteropError::VulkanError(format!("vkCreateBuffer (EXP7): {e:?}")))?;

        let buf_reqs = raw_device.get_buffer_memory_requirements(shared_buffer);
        let mem_props = instance.get_physical_device_memory_properties(physical_device);

        let pick_memory_type = |type_bits: u32, required: vk::MemoryPropertyFlags| {
            (0..mem_props.memory_type_count).find(|&i| {
                (type_bits & (1 << i)) != 0
                    && mem_props.memory_types[i as usize]
                        .property_flags
                        .contains(required)
            })
        };

        // OPAQUE_FD note (Khronos VUID-00674 / issue #1783):
        // `vkGetMemoryFdPropertiesKHR` must NOT be called with OPAQUE_FD,
        // so there is no fd-side memoryTypeBits query available for this
        // CUDA-created payload. For OPAQUE_FD imported from outside Vulkan,
        // the external-memory-fd spec does not provide a second runtime
        // memory-type/size oracle equivalent to the DMA_BUF path.
        // Therefore this control follows NVIDIA's simpleVulkanMMAP pattern:
        // choose from the VkBuffer's own `memRequirements.memoryTypeBits`,
        // prefer DEVICE_LOCAL, then fall back to any permitted type.
        // The runtime sentinel comparison is the actual aliasing evidence.
        // If this buffer control passes while the imported image path still
        // fails, that points toward image-specific external-memory semantics
        // rather than a generic CUDA-VMM -> Vulkan OPAQUE_FD failure.
        // Do not reintroduce vkGetMemoryFdPropertiesKHR for OPAQUE_FD here.
        // Capability/dedicated-allocation queries above remain the queryable
        // evidence for whether this buffer configuration is importable.
        //
        let memory_type_index = pick_memory_type(
            buf_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .or_else(|| pick_memory_type(buf_reqs.memory_type_bits, vk::MemoryPropertyFlags::empty()))
        .ok_or_else(|| {
            CudaInteropError::VulkanError(
                "EXP7: no compatible memory type for imported buffer".into(),
            )
        })?;

        // NVIDIA passes memRequirements.size verbatim -- not the CUDA
        // granularity-rounded size, which is what the image path uses.
        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD)
            .fd(fd);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(buf_reqs.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut import_info);

        let shared_memory = raw_device.allocate_memory(&alloc_info, None).map_err(|e| {
            CudaInteropError::VulkanError(format!("vkAllocateMemory (EXP7 import fd): {e:?}"))
        })?;

        raw_device
            .bind_buffer_memory(shared_buffer, shared_memory, 0)
            .map_err(|e| {
                CudaInteropError::VulkanError(format!("vkBindBufferMemory (EXP7): {e:?}"))
            })?;

        log::info!(
            "ZC_EXP7_BUF: imported OPAQUE_FD into VkBuffer -- vk_mem_reqs_size={} \
             alignment={} cuda_alloc_size={} allocationSize_used={} \
             size_exact_match={} memory_type_index={memory_type_index}",
            buf_reqs.size,
            buf_reqs.alignment,
            shared_mem.alloc_size,
            buf_reqs.size,
            shared_mem.alloc_size as u64 == buf_reqs.size,
        );

        // --- CUDA->Vulkan sync: diagnostic external semaphore ---
        // Without this the copy submit races the CUDA writes and a byte
        // mismatch would be inconclusive. Created here (not earlier) so the
        // capability gates above can bail without leaking semaphore state.
        let (vk_semaphore, sem_fd) = create_export_semaphore(gpu)?;
        // Takes ownership of sem_fd on success; closes it itself on failure.
        let cuda_semaphore = crate::interop::cuda::cuda_import_external_semaphore(sem_fd)?;

        // --- CUDA writes the sentinel at the three offsets ---
        for off in OFFSETS {
            crate::interop::cuda::cuda_memcpy_htod_2d(
                cuda_ptr + off as u64,
                SENTINEL_LEN,
                sentinel.as_ptr(),
                SENTINEL_LEN,
                SENTINEL_LEN,
                1,
            )?;
        }
        // Ordered after the writes have *completed*, not merely been
        // submitted, so the semaphore signal cannot precede the data.
        crate::interop::cuda::cuda_synchronize()?;
        crate::interop::cuda::cuda_signal_external_semaphore(cuda_semaphore)?;
        log::info!(
            "ZC_EXP7_BUF_SYNC: sentinel writes complete, CUDA signalled the \
             imported external semaphore; the shared->staging copy submit below \
             waits on it at TRANSFER"
        );

        // --- CUDA-side ground truth (control for the control) ---
        // If this fails the harness is broken, not the hypothesis.
        let mut cuda_nonzero = [0usize; OFFSETS.len()];
        let mut cuda_exact = [false; OFFSETS.len()];
        for (i, off) in OFFSETS.iter().enumerate() {
            let mut host = vec![0u8; SENTINEL_LEN];
            crate::interop::cuda::cuda_memcpy_dtoh(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                cuda_ptr + *off as u64,
                SENTINEL_LEN,
            )?;
            cuda_nonzero[i] = host.iter().filter(|b| **b != 0).count();
            cuda_exact[i] = host == sentinel;
        }

        // --- Vulkan-side read: copy shared buffer -> host-visible staging ---
        let staging_info = vk::BufferCreateInfo::default()
            .size(CONTROL_SIZE as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_buffer = raw_device.create_buffer(&staging_info, None).map_err(|e| {
            CudaInteropError::VulkanError(format!("vkCreateBuffer (EXP7 staging): {e:?}"))
        })?;
        let staging_reqs = raw_device.get_buffer_memory_requirements(staging_buffer);
        let staging_type = pick_memory_type(
            staging_reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or_else(|| {
            CudaInteropError::VulkanError("EXP7: no HOST_VISIBLE|HOST_COHERENT memory type".into())
        })?;
        let staging_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(staging_reqs.size)
            .memory_type_index(staging_type);
        let staging_memory = raw_device
            .allocate_memory(&staging_alloc, None)
            .map_err(|e| {
                CudaInteropError::VulkanError(format!("vkAllocateMemory (EXP7 staging): {e:?}"))
            })?;
        raw_device
            .bind_buffer_memory(staging_buffer, staging_memory, 0)
            .map_err(|e| {
                CudaInteropError::VulkanError(format!("vkBindBufferMemory (EXP7 staging): {e:?}"))
            })?;

        let (raw_queue, queue_family_index) = gpu
            .queue
            .as_hal::<Vulkan>()
            .map(|hal_queue| (hal_queue.as_raw(), hal_queue.family_index()))
            .ok_or(CudaInteropError::NotVulkan)?;

        let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index);
        let cmd_pool = raw_device
            .create_command_pool(&pool_info, None)
            .map_err(|e| {
                CudaInteropError::VulkanError(format!("vkCreateCommandPool (EXP7): {e:?}"))
            })?;
        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_buf = raw_device
            .allocate_command_buffers(&cmd_alloc)
            .map_err(|e| {
                CudaInteropError::VulkanError(format!("vkAllocateCommandBuffers (EXP7): {e:?}"))
            })?[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        raw_device
            .begin_command_buffer(cmd_buf, &begin_info)
            .map_err(|e| {
                CudaInteropError::VulkanError(format!("vkBeginCommandBuffer (EXP7): {e:?}"))
            })?;
        let region = vk::BufferCopy::default()
            .src_offset(0)
            .dst_offset(0)
            .size(CONTROL_SIZE as u64);
        raw_device.cmd_copy_buffer(
            cmd_buf,
            shared_buffer,
            staging_buffer,
            std::slice::from_ref(&region),
        );
        raw_device.end_command_buffer(cmd_buf).map_err(|e| {
            CudaInteropError::VulkanError(format!("vkEndCommandBuffer (EXP7): {e:?}"))
        })?;

        let fence = raw_device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(|e| CudaInteropError::VulkanError(format!("vkCreateFence (EXP7): {e:?}")))?;
        // THE synchronized submit: waits on the CUDA-signalled semaphore at
        // TRANSFER before the shared->staging copy executes. PASS/FAIL may
        // only be classified from bytes read after this completes.
        let wait_semaphores = [vk_semaphore];
        let wait_stages = [vk::PipelineStageFlags::TRANSFER];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(std::slice::from_ref(&cmd_buf));
        raw_device
            .queue_submit(raw_queue, std::slice::from_ref(&submit_info), fence)
            .map_err(|e| CudaInteropError::VulkanError(format!("vkQueueSubmit (EXP7): {e:?}")))?;
        raw_device
            .wait_for_fences(&[fence], true, u64::MAX)
            .map_err(|e| CudaInteropError::VulkanError(format!("vkWaitForFences (EXP7): {e:?}")))?;

        let mapped = raw_device
            .map_memory(
                staging_memory,
                0,
                CONTROL_SIZE as u64,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|e| CudaInteropError::VulkanError(format!("vkMapMemory (EXP7): {e:?}")))?;
        let host_view = std::slice::from_raw_parts(mapped as *const u8, CONTROL_SIZE);

        let mut all_pass = true;
        for (i, off) in OFFSETS.iter().enumerate() {
            let window = &host_view[*off..*off + SENTINEL_LEN];
            let vk_nonzero = window.iter().filter(|b| **b != 0).count();
            let byte_exact = window == sentinel.as_slice();
            if !byte_exact {
                all_pass = false;
            }
            log::info!(
                "ZC_EXP7_BUF_ALIAS: offset={off} expected_nonzero={SENTINEL_LEN} \
                 cuda_nonzero={} cuda_byte_exact={} vk_nonzero={vk_nonzero} \
                 byte_exact={byte_exact}",
                cuda_nonzero[i],
                cuda_exact[i],
            );
        }

        let cuda_ground_truth_ok = cuda_exact.iter().all(|v| *v);
        raw_device.unmap_memory(staging_memory);

        if !cuda_ground_truth_ok {
            log::error!(
                "ZC_EXP7_BUF_ALIAS_RESULT=HARNESS_ERROR -- the CUDA-side ground-truth \
                 readback did not return the sentinel, so the intended mechanism was \
                 never exercised. This says NOTHING about buffer aliasing."
            );
        } else {
            log::info!(
                "ZC_EXP7_BUF_ALIAS_RESULT={}",
                if all_pass { "PASS" } else { "FAIL" }
            );
        }

        // --- Teardown ---
        // The wait above consumed the binary semaphore's signal, so it is
        // unsignalled and has no pending operations; safe to destroy. The
        // CUDA handle is destroyed separately from the VkSemaphore, and the
        // exported fd was consumed by the import.
        if let Err(e) = crate::interop::cuda::cuda_destroy_external_semaphore(cuda_semaphore) {
            log::warn!("ZC_EXP7: cuDestroyExternalSemaphore failed: {e:?}");
        }
        raw_device.destroy_semaphore(vk_semaphore, None);
        raw_device.destroy_fence(fence, None);
        raw_device.destroy_command_pool(cmd_pool, None);
        raw_device.destroy_buffer(staging_buffer, None);
        raw_device.free_memory(staging_memory, None);
        raw_device.destroy_buffer(shared_buffer, None);
        // Frees the imported memory and, with it, the fd Vulkan took
        // ownership of at import time.
        raw_device.free_memory(shared_memory, None);
    }

    drop(shared_mem);
    Ok(())
}

/// Bytes per pixel for the texture formats we use.
fn format_bytes_per_pixel(format: wgpu::TextureFormat) -> usize {
    match format {
        wgpu::TextureFormat::R8Unorm => 1,
        wgpu::TextureFormat::Rg8Unorm => 2,
        wgpu::TextureFormat::R16Unorm => 2,
        wgpu::TextureFormat::Rg16Unorm => 4,
        wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => 4,
        _ => panic!("unsupported format for CUDA interop: {format:?}"),
    }
}

/// Map wgpu texture format to Vulkan format.
fn wgpu_format_to_vk(format: wgpu::TextureFormat) -> ash::vk::Format {
    match format {
        wgpu::TextureFormat::R8Unorm => ash::vk::Format::R8_UNORM,
        wgpu::TextureFormat::Rg8Unorm => ash::vk::Format::R8G8_UNORM,
        wgpu::TextureFormat::R16Unorm => ash::vk::Format::R16_UNORM,
        wgpu::TextureFormat::Rg16Unorm => ash::vk::Format::R16G16_UNORM,
        wgpu::TextureFormat::Rgba8Unorm => ash::vk::Format::R8G8B8A8_UNORM,
        wgpu::TextureFormat::Rgba8UnormSrgb => ash::vk::Format::R8G8B8A8_SRGB,
        _ => panic!("unsupported format for Vulkan interop: {format:?}"),
    }
}

/// NV12 plane identifier for [`create_nv12_shared_texture`].
#[derive(Debug, Clone, Copy)]
pub enum Nv12Plane {
    /// Luminance plane (full resolution, `R8Unorm` or `R16Unorm` for 10-bit).
    Y,
    /// Chrominance plane (half resolution in each dimension, `Rg8Unorm` or `Rg16Unorm` for 10-bit).
    Uv,
}

/// Create a shared texture sized and formatted for an NV12 plane.
///
/// This is a convenience wrapper around [`create_shared_texture`] that
/// infers the wgpu format and dimensions from the plane type and pixel
/// format. The texture formats are determined by
/// `GpuPixelFormat::y_format()` and `GpuPixelFormat::uv_format()`.
///
/// Unorm normalization maps both 8-bit and 16-bit values to `[0.0, 1.0]`
/// in the shader, so the fragment shader works unchanged across formats.
#[cfg(target_os = "linux")]
pub fn create_nv12_shared_texture(
    gpu: &GpuContext,
    width: u32,
    height: u32,
    plane: Nv12Plane,
    pixel_format: crate::render::renderer::GpuPixelFormat,
) -> Result<SharedTexture, CudaInteropError> {
    match plane {
        Nv12Plane::Y => create_shared_texture(gpu, width, height, pixel_format.y_format()),
        Nv12Plane::Uv => {
            create_shared_texture(gpu, width / 2, height / 2, pixel_format.uv_format())
        }
    }
}

/// Create a shared buffer sized for one NV12/P010 plane.
#[cfg(target_os = "linux")]
pub fn create_nv12_shared_buffer(
    gpu: &GpuContext,
    width: u32,
    height: u32,
    plane: Nv12Plane,
    pixel_format: crate::render::renderer::GpuPixelFormat,
    label: &str,
) -> Result<SharedBuffer, CudaInteropError> {
    match plane {
        Nv12Plane::Y => create_shared_buffer(gpu, width, height, pixel_format.y_format(), label),
        Nv12Plane::Uv => {
            create_shared_buffer(gpu, width / 2, height / 2, pixel_format.uv_format(), label)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_shared_texture() {
        if !crate::interop::cuda::is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let gpu = match pollster::block_on(GpuContext::new()) {
            Ok(g) => g,
            Err(e) => {
                println!("Skipping: no GPU: {e}");
                return;
            }
        };

        // Only works on Vulkan backend
        if gpu.adapter_info.backend != wgpu::Backend::Vulkan {
            println!(
                "Skipping: not Vulkan backend ({:?})",
                gpu.adapter_info.backend
            );
            return;
        }

        // Use a small texture - this test verifies the interop pipeline works,
        // not that it handles production sizes. Smaller textures avoid OOM on
        // memory-constrained devices (e.g. Jetson with shared CPU/GPU RAM).
        let tex = create_shared_texture(&gpu, 256, 256, wgpu::TextureFormat::R8Unorm)
            .expect("should create shared texture");

        println!(
            "Shared texture created: {}x{}, cuda_ptr=0x{:x}, pitch={}",
            256, 256, tex.cuda_ptr, tex.pitch
        );

        // Verify the texture is usable by creating a view
        let _view = tex
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        println!("Texture view created successfully");
    }

    /// Proves that Vulkan takes ownership of the fd on import.
    ///
    /// After `vkAllocateMemory` with `VkImportMemoryFdInfoKHR`, the fd should
    /// no longer be valid. Calling `close()` on it should fail with EBADF,
    /// proving that the driver consumed it. This is what the Vulkan spec
    /// requires, and it means any code that calls `close(fd)` after import
    /// (like Gyroflow does) is performing an invalid operation.
    #[test]
    fn test_fd_ownership_transfers_to_vulkan() {
        if !crate::interop::cuda::is_cuda_available() {
            println!("Skipping: CUDA not available");
            return;
        }

        let gpu = match pollster::block_on(GpuContext::new()) {
            Ok(g) => g,
            Err(e) => {
                println!("Skipping: no GPU: {e}");
                return;
            }
        };

        if gpu.adapter_info.backend != wgpu::Backend::Vulkan {
            println!("Skipping: not Vulkan ({:?})", gpu.adapter_info.backend);
            return;
        }

        // Step 1: Allocate CUDA shared memory and grab the fd (small size for Jetson compat)
        let shared_mem =
            crate::interop::cuda::allocate_shared_memory(256 * 256).expect("alloc shared mem");
        let fd = shared_mem.shared_handle;
        println!("CUDA exported fd = {fd}");

        // Verify the fd is valid before Vulkan import
        // fstat() on a valid fd returns 0, on invalid fd returns -1 with EBADF
        let valid_before = unsafe { libc::fcntl(fd, libc::F_GETFD) };
        assert!(
            valid_before >= 0,
            "fd {fd} should be valid before import, got errno {}",
            std::io::Error::last_os_error()
        );
        println!("Before Vulkan import: fd {fd} is valid (fcntl returned {valid_before})");

        // Step 2: Import into Vulkan (this should consume the fd)
        let tex = create_shared_texture(&gpu, 256, 256, wgpu::TextureFormat::R8Unorm)
            .expect("create shared texture");
        let imported_fd = tex._shared_mem.shared_handle;

        // Step 3: Try to use the fd after Vulkan import
        let valid_after = unsafe { libc::fcntl(imported_fd, libc::F_GETFD) };
        let errno_after = std::io::Error::last_os_error();

        println!(
            "After Vulkan import: fcntl(fd={imported_fd}) returned {valid_after}, errno = {errno_after}"
        );

        if valid_after < 0 {
            println!(
                "CONFIRMED: fd {imported_fd} is invalid after Vulkan import (EBADF). \
                 Vulkan took ownership. Calling close() on it would be a spec violation."
            );
        } else {
            println!(
                "UNEXPECTED: fd {imported_fd} is still valid after Vulkan import. \
                 Driver may have dup()'d internally (not spec-compliant behavior). \
                 close() would still be a spec violation per VK_KHR_external_memory_fd."
            );
        }

        // Step 4: If the fd IS still valid, try closing it and see what happens
        // to the texture (this would be the Gyroflow pattern)
        if valid_after >= 0 {
            println!("Attempting close(fd={imported_fd}) like Gyroflow does...");
            let close_ret = unsafe { libc::close(imported_fd) };
            let close_errno = std::io::Error::last_os_error();
            println!("close() returned {close_ret}, errno = {close_errno}");

            // The texture should still work because Vulkan imported the memory
            // (the fd is just a handle, the dmabuf reference is held by Vulkan)
            let _view = tex
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            println!("Texture still usable after close(fd) - dmabuf ref held by Vulkan");
        }
    }
}

/// Create a binary Vulkan semaphore exportable as a POSIX fd, for the
/// diagnostic CUDA->Vulkan cross-API sync (Ticket 2). CUDA imports the
/// returned fd via `cuImportExternalSemaphore` and signals it after
/// `cuCtxSynchronize()` confirms a frame's `cuMemcpy2D` writes are
/// complete; Vulkan waits on the returned `VkSemaphore` (via the
/// existing `add_wait_semaphore`, see the compile-only probe below)
/// before reading the same shared memory.
///
/// Requires `VK_KHR_external_semaphore_fd` enabled on the device --
/// added to `JhnsonO/wgpu`'s optional-extension list (version-gated:
/// `VK_KHR_external_semaphore` is only separately requested on a
/// Vulkan 1.0 device, since it's core as of 1.1) alongside the
/// existing `VK_KHR_external_memory_fd` request this file already
/// relies on for the shared textures above.
///
/// Checks the physical device's advertised extensions directly before
/// touching the KHR loader or its function pointers, so an
/// unsupported device fails with a clear error here rather than
/// risking a call through an unresolved (null) `vkGetSemaphoreFdKHR`
/// if the extension wasn't actually enabled at device creation.
///
/// Per the extension spec, each successful `vkGetSemaphoreFdKHR` call
/// transfers ownership of a fresh fd to the caller; the caller must
/// ensure it's eventually consumed. On the CUDA side specifically:
/// `cuImportExternalSemaphore` takes ownership of the fd only on a
/// *successful* import -- if that import fails, the fd is still ours
/// to close (see `cuda_import_external_semaphore`'s caller contract).
#[cfg(target_os = "linux")]
pub fn create_export_semaphore(
    gpu: &GpuContext,
) -> Result<(ash::vk::Semaphore, std::os::raw::c_int), CudaInteropError> {
    use ash::vk;
    use wgpu::hal::api::Vulkan;

    unsafe {
        let hal_device_guard = gpu
            .device
            .as_hal::<Vulkan>()
            .ok_or(CudaInteropError::NotVulkan)?;
        let hal_device = &*hal_device_guard;
        let raw_device = hal_device.raw_device();
        let physical_device = hal_device.raw_physical_device();
        let raw_instance = hal_device.shared_instance().raw_instance();

        // Fail clearly here rather than risk dereferencing an
        // unavailable extension function pointer below: confirm the
        // physical device actually advertises the fd-export extension
        // (equivalent to the check the wgpu-hal patch above uses to
        // decide whether to enable it at device creation).
        let supported_extensions = raw_instance
            .enumerate_device_extension_properties(physical_device)
            .map_err(|e| {
                CudaInteropError::VulkanError(format!(
                    "vkEnumerateDeviceExtensionProperties: {e:?}"
                ))
            })?;
        let has_semaphore_fd = supported_extensions
            .iter()
            .any(|ep| ep.extension_name_as_c_str() == Ok(ash::khr::external_semaphore_fd::NAME));
        if !has_semaphore_fd {
            return Err(CudaInteropError::VulkanError(
                "VK_KHR_external_semaphore_fd not supported by this device -- \
                 diagnostic CUDA<->Vulkan semaphore sync unavailable"
                    .into(),
            ));
        }

        let mut export_info = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
        let sem_info = vk::SemaphoreCreateInfo::default().push_next(&mut export_info);
        let semaphore = raw_device.create_semaphore(&sem_info, None).map_err(|e| {
            CudaInteropError::VulkanError(format!("vkCreateSemaphore (export): {e:?}"))
        })?;

        let ext_semaphore_fd =
            ash::khr::external_semaphore_fd::Device::new(raw_instance, raw_device);
        let get_fd_info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
        // ash 0.38's `khr::external_semaphore_fd::Device` drops the `_khr`
        // suffix on its idiomatic wrapper methods (confirmed against the
        // actual pinned version, ash-0.38.0+1.3.281, via docs.rs -- the
        // real method is `get_semaphore_fd`, not `get_semaphore_fd_khr`;
        // the first real compile-proof dispatch caught this as E0599).
        let fd = ext_semaphore_fd
            .get_semaphore_fd(&get_fd_info)
            .map_err(|e| CudaInteropError::VulkanError(format!("vkGetSemaphoreFdKHR: {e:?}")))?;

        Ok((semaphore, fd))
    }
}

/// Compile-only probe (ticket 1b): proves `wgpu::Queue::as_hal::<Vulkan>()`
/// reaches the patched `wgpu-hal` and that `add_wait_semaphore` type-checks
/// through Reco's existing HAL access pattern. Intentionally never called --
/// no semaphore is created or waited on. Backing implementation lives in
/// `JhnsonO/wgpu@62e15ce1` (backport of upstream #9461 onto v28.0.1).
#[allow(dead_code)]
fn _compile_only_probe_add_wait_semaphore_typechecks(gpu: &GpuContext) {
    use wgpu::hal::api::Vulkan;

    // SAFETY: compile-only probe, never called. `as_hal` requires the
    // caller not to destroy the returned resource while in use by the
    // GPU; no resource is destroyed here, and no semaphore is created
    // or waited on -- this exists purely to type-check
    // `add_wait_semaphore` against the patched wgpu-hal.
    unsafe {
        if let Some(hal_queue) = gpu.queue.as_hal::<Vulkan>() {
            hal_queue.add_wait_semaphore(
                ash::vk::Semaphore::null(),
                None,
                ash::vk::PipelineStageFlags::TOP_OF_PIPE,
            );
        }
    }
}
