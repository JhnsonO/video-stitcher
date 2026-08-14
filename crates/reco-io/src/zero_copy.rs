//! Decode thread spawning for zero-copy GPU paths.
//!
//! These functions spawn FFmpeg decode threads that write directly to
//! GPU-shared memory (CUDA/Vulkan on Linux, VideoToolbox/Metal on macOS).
//! The types they consume and produce are defined in [`reco_core::interop::zero_copy`].
//!
//! This module lives in `reco-io` (not `reco-core`) because it needs
//! `VideoDecoder` from the FFmpeg backend. `reco-core` orchestrates
//! the frame loop; `reco-io` handles the decode threads.

/// Spawn a single-video GPU decode thread that writes NV12 frames directly
/// to CUDA/Vulkan shared textures via `cuMemcpy2D`.
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub fn spawn_single_decoder_gpu(
    input: crate::stitch_job::InputPath,
    label: &'static str,
    buf: reco_core::interop::zero_copy::GpuBufInfo,
    slot_free_rx: std::sync::mpsc::Receiver<u8>,
    skip_frames: u64,
    shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> (std::sync::mpsc::Receiver<u8>, std::thread::JoinHandle<()>) {
    use crate::ffmpeg::decoder::VideoDecoder;

    let (tx, rx) = std::sync::mpsc::sync_channel::<u8>(1);

    let handle = std::thread::Builder::new()
        .name(format!("decode_{label}_gpu"))
        .spawn(move || {
            let mut dec = match VideoDecoder::open_input(&input) {
                Ok(d) => {
                    log::info!(
                        "{label} GPU decoder: {} ({}x{})",
                        d.backend(),
                        d.width(),
                        d.height()
                    );
                    d
                }
                Err(e) => {
                    log::error!("Failed to open {label} video: {e}");
                    return;
                }
            };

            // Skip frames for temporal sync (decode and discard, no GPU write).
            for i in 0..skip_frames {
                match dec.next_frame() {
                    Ok(Some(_)) => {}
                    Ok(None) => {
                        log::error!("{label}: EOF after skipping {i}/{skip_frames} frames");
                        return;
                    }
                    Err(e) => {
                        log::error!("{label} skip decode error: {e}");
                        return;
                    }
                }
            }
            if skip_frames > 0 {
                log::info!("{label}: skipped {skip_frames} frames for sync offset");
            }

            let mut frame_count: u64 = 0;
            loop {
                if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
                    break;
                }
                let slot = match slot_free_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                    Ok(s) => s,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                };
                match dec.next_frame_gpu() {
                    Ok(Some(frame)) => {
                        let s = slot as usize;

                        if let Err(e) = reco_core::interop::cuda::cuda_ensure_context() {
                            log::error!("{label} cuda_ensure_context: {e}");
                            break;
                        }

                        // Byte width for CUDA copy: width * bytes_per_sample.
                        // For UV, half-width but 2 components cancels out.
                        let bps = buf.pixel_format.bytes_per_sample();
                        let y_width_bytes = buf.width as usize * bps;
                        let uv_width_bytes = buf.width as usize * bps;

                        // Copy Y plane: NVDEC -> shared texture
                        if let Err(e) = reco_core::interop::cuda::cuda_2d_copy(
                            buf.y_ptr[s],
                            buf.y_pitch[s],
                            frame.y_ptr,
                            frame.y_pitch,
                            y_width_bytes,
                            buf.height as usize,
                        ) {
                            log::error!("{label} cuMemcpy2D Y: {e}");
                            break;
                        }

                        // Copy UV plane: NVDEC -> shared texture
                        if let Err(e) = reco_core::interop::cuda::cuda_2d_copy(
                            buf.uv_ptr[s],
                            buf.uv_pitch[s],
                            frame.uv_ptr,
                            frame.uv_pitch,
                            uv_width_bytes,
                            buf.height as usize / 2,
                        ) {
                            log::error!("{label} cuMemcpy2D UV: {e}");
                            break;
                        }

                        if let Err(e) = reco_core::interop::cuda::cuda_synchronize() {
                            log::error!("{label} cuCtxSynchronize: {e}");
                            break;
                        }

                        // ZC_SEM (diagnostic, Ticket 2): signal this
                        // slot's CUDA->Vulkan semaphore now that
                        // cuCtxSynchronize() above has confirmed both
                        // cuMemcpy2D writes for this frame are actually
                        // complete (not merely submitted). The render
                        // side waits on the matching VkSemaphore
                        // (session::frame_processing::
                        // copy_to_vram_pool_platform) before reading
                        // this slot's shared textures via Vulkan.
                        if let Err(e) =
                            reco_core::interop::cuda::cuda_signal_external_semaphore(buf.sem_cuda[s])
                        {
                            log::error!("{label} cuSignalExternalSemaphoresAsync: {e}");
                            break;
                        }

                        // DIAGNOSTIC (temporary, env-gated, no effect unless
                        // RECO_DEBUG_DUMP_FRAME=1): read back the Y plane we
                        // just wrote via cuMemcpy2D and report its actual
                        // byte content. This isolates whether NVDEC->shared
                        // texture writes ever contain real pixel data,
                        // independent of anything wgpu/Vulkan does with that
                        // memory afterward. Only fires once, on this decode
                        // thread's first successfully-copied frame.
                        if frame_count == 0 && std::env::var("RECO_DEBUG_DUMP_FRAME").is_ok() {
                            let w = buf.width as usize;
                            let h = buf.height as usize;
                            let mut host_buf = vec![0u8; w * h];
                            match reco_core::interop::cuda::cuda_2d_copy_dtoh(
                                host_buf.as_mut_ptr() as *mut std::ffi::c_void,
                                w,
                                buf.y_ptr[s],
                                buf.y_pitch[s],
                                w,
                                h,
                            ) {
                                Ok(()) => {
                                    let min = *host_buf.iter().min().unwrap_or(&0);
                                    let max = *host_buf.iter().max().unwrap_or(&0);
                                    let sum: u64 = host_buf.iter().map(|&b| b as u64).sum();
                                    let mean = sum as f64 / host_buf.len() as f64;
                                    let nonzero = host_buf.iter().filter(|&&b| b != 0).count();
                                    log::warn!(
                                        "{label} DIAG frame0 Y-plane readback: {w}x{h}, min={min} max={max} mean={mean:.2} nonzero_bytes={nonzero}/{}",
                                        host_buf.len()
                                    );
                                    if let Some(dir) = std::env::var_os("RECO_DEBUG_DUMP_DIR") {
                                        let path = std::path::Path::new(&dir)
                                            .join(format!("{label}_frame0_y_{w}x{h}.raw"));
                                        if let Err(e) = std::fs::write(&path, &host_buf) {
                                            log::error!("{label} DIAG dump write failed: {e}");
                                        } else {
                                            log::warn!("{label} DIAG dumped raw Y plane to {}", path.display());
                                        }
                                    }
                                }
                                Err(e) => {
                                    log::error!("{label} DIAG readback failed: {e}");
                                }
                            }

                            // UV plane: same diagnostic, same trigger. NV12 UV
                            // is interleaved U/V, full width in bytes, half
                            // height. This is upstream of the cycle-3
                            // device.poll(Wait) fix (which lives in
                            // render_gpu_resident(), after this decode thread
                            // has already returned the slot) -- so this
                            // readback tells us whether the CUDA-side UV copy
                            // itself is valid, independent of that fix.
                            let uv_w = buf.width as usize;
                            let uv_h = buf.height as usize / 2;
                            let mut uv_host_buf = vec![0u8; uv_w * uv_h];
                            match reco_core::interop::cuda::cuda_2d_copy_dtoh(
                                uv_host_buf.as_mut_ptr() as *mut std::ffi::c_void,
                                uv_w,
                                buf.uv_ptr[s],
                                buf.uv_pitch[s],
                                uv_w,
                                uv_h,
                            ) {
                                Ok(()) => {
                                    let min = *uv_host_buf.iter().min().unwrap_or(&0);
                                    let max = *uv_host_buf.iter().max().unwrap_or(&0);
                                    let sum: u64 = uv_host_buf.iter().map(|&b| b as u64).sum();
                                    let mean = sum as f64 / uv_host_buf.len() as f64;
                                    let nonzero = uv_host_buf.iter().filter(|&&b| b != 0).count();
                                    log::warn!(
                                        "{label} DIAG frame0 UV-plane readback: {uv_w}x{uv_h}, min={min} max={max} mean={mean:.2} nonzero_bytes={nonzero}/{}",
                                        uv_host_buf.len()
                                    );
                                    if let Some(dir) = std::env::var_os("RECO_DEBUG_DUMP_DIR") {
                                        let path = std::path::Path::new(&dir)
                                            .join(format!("{label}_frame0_uv_{uv_w}x{uv_h}.raw"));
                                        if let Err(e) = std::fs::write(&path, &uv_host_buf) {
                                            log::error!("{label} DIAG UV dump write failed: {e}");
                                        } else {
                                            log::warn!("{label} DIAG dumped raw UV plane to {}", path.display());
                                        }
                                    }
                                }
                                Err(e) => {
                                    log::error!("{label} DIAG UV readback failed: {e}");
                                }
                            }
                        }
                        frame_count += 1;

                        if tx.send(slot).is_err() {
                            break;
                        }
                    }
                    Ok(None) => {
                        log::error!("{label}: next_frame_gpu returned None (non-CUDA?)");
                        break;
                    }
                    Err(e) => {
                        log::error!("{label} decode error: {e}");
                        break;
                    }
                }
            }
        })
        .expect("spawn GPU decode thread");

    (rx, handle)
}

/// Spawn parallel GPU decode threads and a pairing thread.
///
/// `sync_offset` applies temporal alignment: positive skips right frames,
/// negative skips left frames (see [`FfmpegFileSource::open_with_offset`](crate::adapters::FfmpegFileSource::open_with_offset)).
///
/// Returns [`GpuDecodeHandles`](reco_core::interop::zero_copy::GpuDecodeHandles) containing the paired frame signal
/// receiver and join handles for graceful shutdown.
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub fn spawn_decode_threads_gpu(
    left_input: crate::stitch_job::InputPath,
    right_input: crate::stitch_job::InputPath,
    left_buf: reco_core::interop::zero_copy::GpuBufInfo,
    right_buf: reco_core::interop::zero_copy::GpuBufInfo,
    left_slot_free_rx: std::sync::mpsc::Receiver<u8>,
    right_slot_free_rx: std::sync::mpsc::Receiver<u8>,
    sync_offset: i64,
    shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
) -> reco_core::interop::zero_copy::GpuDecodeHandles {
    use reco_core::interop::zero_copy::{GpuDecodeHandles, GpuFrameSignal};

    // Compute per-decoder skip counts from the sync offset.
    let (left_skip, right_skip) = if sync_offset > 0 {
        (0, sync_offset as u64)
    } else {
        (sync_offset.unsigned_abs(), 0)
    };

    let (left_rx, left_handle) = spawn_single_decoder_gpu(
        left_input,
        "left",
        left_buf,
        left_slot_free_rx,
        left_skip,
        shutdown.clone(),
    );
    let (right_rx, right_handle) = spawn_single_decoder_gpu(
        right_input,
        "right",
        right_buf,
        right_slot_free_rx,
        right_skip,
        shutdown,
    );

    let (tx, rx) = std::sync::mpsc::sync_channel::<GpuFrameSignal>(1);

    let pair_handle = std::thread::Builder::new()
        .name("decode_pair_gpu".into())
        .spawn(move || {
            while let (Ok(left_slot), Ok(right_slot)) = (left_rx.recv(), right_rx.recv()) {
                if tx
                    .send(GpuFrameSignal {
                        left_slot,
                        right_slot,
                    })
                    .is_err()
                {
                    break;
                }
            }
        })
        .expect("spawn GPU pairing thread");

    GpuDecodeHandles {
        frame_rx: rx,
        join_handles: vec![left_handle, right_handle, pair_handle],
    }
}

/// Spawn a VideoToolbox decode thread that sends retained CVPixelBuffers.
#[cfg(target_os = "macos")]
pub fn spawn_vt_decode_thread(
    input: crate::stitch_job::InputPath,
    label: &'static str,
) -> std::sync::mpsc::Receiver<reco_core::interop::metal::RetainedCVPixelBuffer> {
    use crate::ffmpeg::decoder::VideoDecoder;
    use reco_core::interop::metal::RetainedCVPixelBuffer;

    let (tx, rx) = std::sync::mpsc::sync_channel::<RetainedCVPixelBuffer>(4);

    std::thread::Builder::new()
        .name(format!("vt_decode_{label}"))
        .spawn(move || {
            let mut dec = match VideoDecoder::open_input(&input) {
                Ok(d) => d,
                Err(e) => {
                    log::error!("Failed to open {label} video: {e}");
                    return;
                }
            };
            log::info!("VT decode thread {label}: backend={}", dec.backend());

            loop {
                match dec.next_frame_vt() {
                    Ok(Some(vt)) => {
                        let retained = unsafe { RetainedCVPixelBuffer::retain(vt.cv_pixel_buffer) };
                        if tx.send(retained).is_err() {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        log::error!("{label} VT decode error: {e}");
                        break;
                    }
                }
            }
        })
        .expect("spawn VT decode thread");

    rx
}

/// Spawn paired VideoToolbox decode threads and return the pair receiver.
///
/// `sync_offset` applies temporal alignment: positive skips right frames,
/// negative skips left frames.
///
/// Spawns two VT decode threads (left + right) and a pairing thread
/// that zips frames into [`VtFramePair`]s.
#[cfg(target_os = "macos")]
pub fn spawn_vt_decode_pair(
    left: &crate::stitch_job::InputPath,
    right: &crate::stitch_job::InputPath,
    sync_offset: i64,
) -> std::sync::mpsc::Receiver<reco_core::interop::zero_copy::VtFramePair> {
    use reco_core::interop::zero_copy::VtFramePair;

    let left_rx = spawn_vt_decode_thread(left.clone(), "left");
    let right_rx = spawn_vt_decode_thread(right.clone(), "right");

    let (pair_tx, pair_rx) = std::sync::mpsc::sync_channel::<VtFramePair>(4);
    std::thread::Builder::new()
        .name("vt_pair".into())
        .spawn(move || {
            // Apply sync offset.
            if sync_offset > 0 {
                for _ in 0..sync_offset {
                    if right_rx.recv().is_err() {
                        return;
                    }
                }
                log::info!("VT sync offset: skipped {sync_offset} right frames");
            } else if sync_offset < 0 {
                let skip = sync_offset.unsigned_abs();
                for _ in 0..skip {
                    if left_rx.recv().is_err() {
                        return;
                    }
                }
                log::info!("VT sync offset: skipped {skip} left frames");
            }

            while let (Ok(left), Ok(right)) = (left_rx.recv(), right_rx.recv()) {
                if pair_tx.send(VtFramePair { left, right }).is_err() {
                    break;
                }
            }
        })
        .expect("spawn VT pairing thread");

    pair_rx
}

/// Spawn a D3D11VA decode thread using a shared hw device.
#[cfg(target_os = "windows")]
fn spawn_d3d11_decode_thread_shared(
    input: crate::stitch_job::InputPath,
    label: &'static str,
    shared_device: crate::ffmpeg::decoder::SharedHwDevice,
) -> std::sync::mpsc::Receiver<crate::ffmpeg::decoder::D3d11Frame> {
    use crate::ffmpeg::decoder::{D3d11Frame, VideoDecoder};

    let (tx, rx) = std::sync::mpsc::sync_channel::<D3d11Frame>(4);

    std::thread::Builder::new()
        .name(format!("d3d11_decode_{label}"))
        .spawn(move || {
            let mut dec = match VideoDecoder::open_input_with_shared_device(&input, &shared_device)
            {
                Ok(d) => d,
                Err(e) => {
                    log::error!("Failed to open {label} video: {e}");
                    return;
                }
            };
            log::info!("D3D11VA decode thread {label}: backend={}", dec.backend());

            loop {
                match dec.next_frame_d3d11() {
                    Ok(Some(frame)) => {
                        if tx.send(frame).is_err() {
                            break;
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        log::error!("{label} D3D11VA decode error: {e}");
                        break;
                    }
                }
            }
        })
        .expect("spawn D3D11VA decode thread");

    rx
}

/// Spawn paired D3D11VA decode threads and return the pair receiver.
///
/// Creates a single shared D3D11VA device so both decoders produce textures
/// on the same `ID3D11Device`. This is required for `CopySubresourceRegion`
/// in the staging pool.
///
/// `sync_offset` applies temporal alignment: positive skips right frames,
/// negative skips left frames.
#[cfg(target_os = "windows")]
pub fn spawn_d3d11_decode_pair(
    left: &crate::stitch_job::InputPath,
    right: &crate::stitch_job::InputPath,
    sync_offset: i64,
) -> std::sync::mpsc::Receiver<(
    crate::ffmpeg::decoder::D3d11Frame,
    crate::ffmpeg::decoder::D3d11Frame,
)> {
    use crate::ffmpeg::decoder::D3d11Frame;

    let shared_device = crate::ffmpeg::decoder::create_shared_hw_device()
        .expect("D3D11VA hw device creation failed");

    let left_rx = spawn_d3d11_decode_thread_shared(left.clone(), "left", shared_device.new_ref());
    let right_rx =
        spawn_d3d11_decode_thread_shared(right.clone(), "right", shared_device.new_ref());

    let (pair_tx, pair_rx) = std::sync::mpsc::sync_channel::<(D3d11Frame, D3d11Frame)>(4);
    std::thread::Builder::new()
        .name("d3d11_pair".into())
        .spawn(move || {
            if sync_offset > 0 {
                for _ in 0..sync_offset {
                    if right_rx.recv().is_err() {
                        return;
                    }
                }
                log::info!("D3D11VA sync offset: skipped {sync_offset} right frames");
            } else if sync_offset < 0 {
                let skip = sync_offset.unsigned_abs();
                for _ in 0..skip {
                    if left_rx.recv().is_err() {
                        return;
                    }
                }
                log::info!("D3D11VA sync offset: skipped {skip} left frames");
            }

            while let (Ok(left), Ok(right)) = (left_rx.recv(), right_rx.recv()) {
                if pair_tx.send((left, right)).is_err() {
                    break;
                }
            }
        })
        .expect("spawn D3D11VA pairing thread");

    pair_rx
}
