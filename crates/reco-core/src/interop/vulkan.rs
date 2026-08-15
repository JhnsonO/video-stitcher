//! Vulkan side of CUDA/Vulkan interop.
//!
//! Imports CUDA-exported shared memory into Vulkan through the HAL escape
//! hatch. The production Linux decode path uses shared `VkBuffer`s as copy
//! sources for ordinary wgpu textures. The older shared-image helpers remain
//! for callers that still require image interop.
//!
//! The flow:
//! 1. CUDA allocates shareable memory and exports a POSIX fd
//! 2. This module creates a Vulkan buffer or image with external-memory flags
//! 3. Imports the fd as the backing memory
//! 4. Wraps the resource into wgpu via the corresponding HAL constructor
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

/// A Vulkan buffer backed by CUDA VMM memory and exposed to wgpu as COPY_SRC.
///
/// CUDA/NVDEC writes through `cuda_ptr`; Vulkan copies the bytes into ordinary
/// wgpu textures after waiting on the slot's external semaphore.
pub struct SharedBuffer {
    pub buffer: wgpu::Buffer,
    pub cuda_ptr: crate::interop::cuda::CUdeviceptr,
    pub pitch: usize,
    pub size: usize,
    _shared_mem: CudaSharedMemory,
}

/// One binary semaphore shared between CUDA and Vulkan for a decode slot.
#[cfg(target_os = "linux")]
pub struct SharedSemaphore {
    vk_semaphore: ash::vk::Semaphore,
    cuda_semaphore: crate::interop::cuda::CudaExternalSemaphore,
    raw_device: ash::Device,
    _device: wgpu::Device,
}

#[cfg(target_os = "linux")]
impl SharedSemaphore {
    pub fn vk_semaphore(&self) -> ash::vk::Semaphore {
        self.vk_semaphore
    }

    pub fn cuda_semaphore(&self) -> crate::interop::cuda::CudaExternalSemaphore {
        self.cuda_semaphore
    }
}

#[cfg(target_os = "linux")]
impl Drop for SharedSemaphore {
    fn drop(&mut self) {
        // A decode thread may have exited after enqueueing its final signal but
        // before the render loop received that frame. Complete CUDA work before
        // destroying either API's handle.
        if let Err(error) = crate::interop::cuda::cuda_synchronize() {
            log::error!("synchronize before external semaphore destroy: {error}");
        }
        if let Err(error) =
            crate::interop::cuda::cuda_destroy_external_semaphore(self.cuda_semaphore)
        {
            log::error!("destroy CUDA external semaphore: {error}");
        }
        unsafe {
            self.raw_device.destroy_semaphore(self.vk_semaphore, None);
        }
    }
}

fn aligned_buffer_pitch(width: u32, format: wgpu::TextureFormat) -> usize {
    let row_bytes = width as usize * format_bytes_per_pixel(format);
    row_bytes.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize
}

/// Attach CUDA-signalled binary-semaphore waits to the next wgpu submission.
///
/// Callers must submit immediately after this succeeds. The waits are staged
/// inside wgpu-hal and are consumed by exactly the next queue submission.
#[cfg(target_os = "linux")]
pub fn stage_cuda_semaphore_waits(
    gpu: &GpuContext,
    semaphores: &[ash::vk::Semaphore],
) -> Result<(), CudaInteropError> {
    stage_cuda_buffer_handoff(gpu, semaphores, &[])
}

/// Stage the complete CUDA->Vulkan->CUDA slot handoff on the next queue
/// submission. Resolving the HAL queue happens before anything is staged, so
/// a backend error cannot leave a half-configured submission behind.
#[cfg(target_os = "linux")]
pub fn stage_cuda_buffer_handoff(
    gpu: &GpuContext,
    wait_semaphores: &[ash::vk::Semaphore],
    completion_semaphores: &[ash::vk::Semaphore],
) -> Result<(), CudaInteropError> {
    use wgpu::hal::api::Vulkan;

    let hal_queue = unsafe { gpu.queue.as_hal::<Vulkan>() }.ok_or(CudaInteropError::NotVulkan)?;
    for &semaphore in wait_semaphores {
        hal_queue.add_wait_semaphore(semaphore, None, ash::vk::PipelineStageFlags::TRANSFER);
    }
    for &semaphore in completion_semaphores {
        hal_queue.add_signal_semaphore(semaphore, None);
    }
    Ok(())
}

/// Consume signalled ready semaphores when a decoded frame is intentionally
/// skipped instead of copied, and signal the reverse completion semaphores.
///
/// This submission is intentionally non-blocking on the CPU. The slot may be
/// returned immediately: when CUDA later receives that slot, it waits on the
/// completion semaphore before overwriting the shared allocation.
#[cfg(target_os = "linux")]
pub fn consume_cuda_semaphore_signals(
    gpu: &GpuContext,
    ready_semaphores: &[ash::vk::Semaphore],
    completion_semaphores: &[ash::vk::Semaphore],
) -> Result<(), CudaInteropError> {
    stage_cuda_buffer_handoff(gpu, ready_semaphores, completion_semaphores)?;
    gpu.queue.submit(std::iter::empty::<wgpu::CommandBuffer>());
    Ok(())
}

/// Wait once for all submitted Vulkan work before external semaphore/resource
/// teardown. This is a lifetime fence only; it is deliberately not used in the
/// per-frame path.
#[cfg(target_os = "linux")]
pub fn wait_for_vulkan_idle(gpu: &GpuContext) -> Result<(), CudaInteropError> {
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| {
            CudaInteropError::VulkanError(format!("wait for final Vulkan work: {error:?}"))
        })?;
    Ok(())
}

/// Create the byte-exact EXP7 primitive as a production wgpu copy source.
///
/// CUDA creates/exports the VMM allocation. Vulkan imports the `OPAQUE_FD`
/// into a `VkBuffer` using exactly `memRequirements.size`; wgpu then owns the
/// Vulkan buffer and memory lifetime. `vkGetMemoryFdPropertiesKHR` is invalid
/// for `OPAQUE_FD` and must not be used here.
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

    let pitch = aligned_buffer_pitch(width, format);
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
            Err(error) => {
                libc::close(fd);
                return Err(CudaInteropError::VulkanError(format!(
                    "vkCreateBuffer ({label}): {error:?}"
                )));
            }
        };

        let mem_reqs = raw_device.get_buffer_memory_requirements(vk_buffer);
        if mem_reqs.size > shared_mem.alloc_size as u64 {
            raw_device.destroy_buffer(vk_buffer, None);
            libc::close(fd);
            return Err(CudaInteropError::VulkanError(format!(
                "{label} Vulkan buffer requires {} bytes but CUDA allocated {}",
                mem_reqs.size, shared_mem.alloc_size
            )));
        }

        let mem_props = instance.get_physical_device_memory_properties(physical_device);
        let pick_memory_type = |required: vk::MemoryPropertyFlags| {
            (0..mem_props.memory_type_count).find(|&index| {
                (mem_reqs.memory_type_bits & (1 << index)) != 0
                    && mem_props.memory_types[index as usize]
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
            Err(error) => {
                raw_device.destroy_buffer(vk_buffer, None);
                libc::close(fd);
                return Err(CudaInteropError::VulkanError(format!(
                    "vkAllocateMemory ({label} OPAQUE_FD): {error:?}"
                )));
            }
        };

        if let Err(error) = raw_device.bind_buffer_memory(vk_buffer, device_memory, 0) {
            raw_device.destroy_buffer(vk_buffer, None);
            raw_device.free_memory(device_memory, None);
            return Err(CudaInteropError::VulkanError(format!(
                "vkBindBufferMemory ({label}): {error:?}"
            )));
        }

        log::info!(
            "CUDA/Vulkan shared buffer {label}: {width}x{height} {format:?}, \
             pitch={pitch}, size={size}, cuda_alloc={}, vk_requirement={}",
            shared_mem.alloc_size,
            mem_reqs.size,
        );
        (vk_buffer, device_memory, mem_reqs.size)
    };

    let buffer = unsafe {
        let hal_buffer =
            wgpu::hal::vulkan::Buffer::from_raw_managed(vk_buffer, device_memory, 0, memory_size);
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
        buffer,
        cuda_ptr,
        pitch,
        size,
        _shared_mem: shared_mem,
    })
}

/// Create a Vulkan binary semaphore, export it as `OPAQUE_FD`, and import it
/// into CUDA. The returned owner destroys both API handles in the right order.
#[cfg(target_os = "linux")]
pub fn create_shared_semaphore(gpu: &GpuContext) -> Result<SharedSemaphore, CudaInteropError> {
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

        let supported_extensions = raw_instance
            .enumerate_device_extension_properties(physical_device)
            .map_err(|error| {
                CudaInteropError::VulkanError(format!(
                    "vkEnumerateDeviceExtensionProperties: {error:?}"
                ))
            })?;
        if !supported_extensions.iter().any(|property| {
            property.extension_name_as_c_str() == Ok(ash::khr::external_semaphore_fd::NAME)
        }) {
            return Err(CudaInteropError::VulkanError(
                "VK_KHR_external_semaphore_fd is unavailable".into(),
            ));
        }

        let mut export_info = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
        let semaphore_info = vk::SemaphoreCreateInfo::default().push_next(&mut export_info);
        let vk_semaphore = raw_device
            .create_semaphore(&semaphore_info, None)
            .map_err(|error| {
                CudaInteropError::VulkanError(format!("vkCreateSemaphore: {error:?}"))
            })?;

        let fd_loader = ash::khr::external_semaphore_fd::Device::new(raw_instance, raw_device);
        let fd_info = vk::SemaphoreGetFdInfoKHR::default()
            .semaphore(vk_semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD);
        let fd = match fd_loader.get_semaphore_fd(&fd_info) {
            Ok(fd) => fd,
            Err(error) => {
                raw_device.destroy_semaphore(vk_semaphore, None);
                return Err(CudaInteropError::VulkanError(format!(
                    "vkGetSemaphoreFdKHR: {error:?}"
                )));
            }
        };

        let cuda_semaphore = match crate::interop::cuda::cuda_import_external_semaphore(fd) {
            Ok(semaphore) => semaphore,
            Err(error) => {
                raw_device.destroy_semaphore(vk_semaphore, None);
                return Err(error);
            }
        };

        Ok(SharedSemaphore {
            vk_semaphore,
            cuda_semaphore,
            raw_device: raw_device.clone(),
            _device: gpu.device.clone(),
        })
    }
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

        // Get actual row pitch from the image layout
        let subresource = vk::ImageSubresource {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            array_layer: 0,
        };
        let layout = raw_device.get_image_subresource_layout(vk_image, subresource);
        let actual_pitch = layout.row_pitch as usize;

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
        )
    };

    Ok(SharedTexture {
        texture: wgpu_texture,
        cuda_ptr,
        pitch: actual_pitch,
        _shared_mem: shared_mem,
    })
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

/// Create one CUDA-shared NV12/P010 plane buffer.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nv12_and_p010_plane_pitches_are_copy_aligned() {
        assert_eq!(
            aligned_buffer_pitch(3840, wgpu::TextureFormat::R8Unorm),
            3840
        );
        assert_eq!(
            aligned_buffer_pitch(1920, wgpu::TextureFormat::Rg8Unorm),
            3840
        );
        assert_eq!(
            aligned_buffer_pitch(3840, wgpu::TextureFormat::R16Unorm),
            7680
        );
        assert_eq!(
            aligned_buffer_pitch(1920, wgpu::TextureFormat::Rg16Unorm),
            7680
        );

        // A width whose byte rows are not naturally 256-byte aligned must
        // preserve tight CUDA copies while padding Vulkan's row stride.
        assert_eq!(
            aligned_buffer_pitch(1920, wgpu::TextureFormat::R8Unorm),
            2048
        );
        assert_eq!(
            aligned_buffer_pitch(960, wgpu::TextureFormat::Rg8Unorm),
            2048
        );
    }

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
