from pathlib import Path


def replace(path: str, old: str, new: str, count: int = 1) -> None:
    p = Path(path)
    text = p.read_text()
    actual = text.count(old)
    if actual != count:
        raise SystemExit(
            f"{path}: expected {count} occurrence(s), found {actual}\n--- needle ---\n{old}"
        )
    p.write_text(text.replace(old, new, count))


# Keep the wgpu PollType dependency inside reco-core, which already owns wgpu.
# reco-io asks for a Vulkan-idle lifetime fence through this narrow helper.
replace(
    "crates/reco-core/src/interop/vulkan.rs",
    """pub fn consume_cuda_semaphore_signals(
    gpu: &GpuContext,
    ready_semaphores: &[ash::vk::Semaphore],
    completion_semaphores: &[ash::vk::Semaphore],
) -> Result<(), CudaInteropError> {
    stage_cuda_buffer_handoff(gpu, ready_semaphores, completion_semaphores)?;
    gpu.queue.submit(std::iter::empty::<wgpu::CommandBuffer>());
    Ok(())
}
""",
    """pub fn consume_cuda_semaphore_signals(
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
""",
)

replace(
    "crates/reco-io/src/smart_source.rs",
    """            if let Err(error) = state
                .gpu
                .device()
                .poll(wgpu::PollType::wait_indefinitely())
            {
                log::warn!("Linux zero-copy teardown GPU wait failed: {error:?}");
            }
""",
    """            if let Err(error) =
                reco_core::interop::vulkan::wait_for_vulkan_idle(&state.gpu)
            {
                log::warn!("Linux zero-copy teardown GPU wait failed: {error}");
            }
""",
)
