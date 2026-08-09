//! Full-screen cylindrical stereo renderer.
//!
//! This is intentionally separate from the perspective renderer so selecting
//! the existing L-shape path creates and executes exactly the same resources.

use bytemuck::{Pod, Zeroable};
use nalgebra::Matrix4;

use super::renderer::InputFormat;
use super::scene::SceneGeometry;
use crate::calibration::{CameraParams, MatchCalibration};
use crate::gpu::GpuContext;
use crate::projection::CylindricalStereoProjectionConfig;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CameraUniforms {
    inv_model: [[f32; 4]; 4],
    intrinsics: [f32; 4],
    dist: [f32; 4],
    color_scale: [f32; 4],
    // xyz = transfer offset; w = physical plane aspect ratio.
    color_offset: [f32; 4],
    flags: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    left: CameraUniforms,
    right: CameraUniforms,
    camera_position: [f32; 4],
    projection: [f32; 4],
}

/// GPU pipeline for yaw-linear cylindrical stereo compositing.
pub(crate) struct CylindricalRenderer {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform_buffer: wgpu::Buffer,
    config: CylindricalStereoProjectionConfig,
    input_format: InputFormat,
    flip_180: [bool; 2],
    is_full_range: bool,
}

impl CylindricalRenderer {
    pub(crate) fn new(
        gpu: &GpuContext,
        output_format: wgpu::TextureFormat,
        input_format: InputFormat,
        config: CylindricalStereoProjectionConfig,
    ) -> Self {
        let device = &gpu.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cylindrical_stereo"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/cylindrical_stereo.wgsl").into(),
            ),
        });
        let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let mut entries: Vec<_> = (0..6).map(texture_entry).collect();
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 6,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 7,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cylindrical_stereo_layout"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cylindrical_stereo_pipeline_layout"),
            bind_group_layouts: &[&layout],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("cylindrical_stereo_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_cylindrical_stereo"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("cylindrical_stereo_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cylindrical_stereo_uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            layout,
            sampler,
            uniform_buffer,
            config,
            input_format,
            flip_180: [false; 2],
            is_full_range: false,
        }
    }

    pub(crate) fn set_flip_180(&mut self, left: bool, right: bool) {
        self.flip_180 = [left, right];
    }
    pub(crate) fn set_full_range(&mut self, value: bool) {
        self.is_full_range = value;
    }

    pub(crate) fn render(
        &self,
        gpu: &GpuContext,
        scene: &SceneGeometry,
        calibration: &MatchCalibration,
        views: [(wgpu::TextureView, wgpu::TextureView, wgpu::TextureView); 2],
        target: &wgpu::Texture,
    ) -> wgpu::CommandBuffer {
        let inv = inverse_models(scene);
        let uniforms = Uniforms {
            left: camera_uniforms(
                &inv.0,
                &calibration.left,
                scene.plane_aspect,
                self.input_format,
                self.flip_180[0],
                self.is_full_range,
            ),
            right: camera_uniforms(
                &inv.1,
                &calibration.right,
                scene.plane_aspect,
                self.input_format,
                self.flip_180[1],
                self.is_full_range,
            ),
            camera_position: [
                scene.camera_position[0],
                scene.camera_position[1],
                scene.camera_position[2],
                0.0,
            ],
            projection: [
                self.config.yaw_span_rad,
                self.config.yaw_center_rad,
                (self.config.vertical_fov_rad * 0.5).tan(),
                self.config.blend_width,
            ],
        };
        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cylindrical_stereo_bind_group"),
            layout: &self.layout,
            entries: &[
                texture_binding(0, &views[0].0),
                texture_binding(1, &views[0].1),
                texture_binding(2, &views[0].2),
                texture_binding(3, &views[1].0),
                texture_binding(4, &views[1].1),
                texture_binding(5, &views[1].2),
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cylindrical_stereo_encoder"),
            });
        let target_view = target.create_view(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("cylindrical_stereo_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.finish()
    }
}

fn texture_binding<'a>(binding: u32, view: &'a wgpu::TextureView) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}
fn camera_uniforms(
    inv: &Matrix4<f32>,
    c: &CameraParams,
    aspect: f32,
    format: InputFormat,
    flip: bool,
    full: bool,
) -> CameraUniforms {
    let (w, h) = (c.width as f32, c.height as f32);
    CameraUniforms {
        inv_model: matrix_columns(inv),
        intrinsics: [
            c.fx as f32 / w,
            c.fy as f32 / h,
            c.cx as f32 / w,
            c.cy as f32 / h,
        ],
        dist: c.d.map(|v| v as f32),
        color_scale: [1.0, 1.0, 1.0, 0.0],
        color_offset: [0.0, 0.0, 0.0, aspect],
        flags: [
            0,
            match format {
                InputFormat::Yuv420p => 0,
                InputFormat::Nv12 => 1,
                InputFormat::Bgra => 2,
            },
            flip as u32,
            full as u32,
        ],
    }
}
fn matrix_columns(m: &Matrix4<f32>) -> [[f32; 4]; 4] {
    let s = m.as_slice();
    [
        [s[0], s[1], s[2], s[3]],
        [s[4], s[5], s[6], s[7]],
        [s[8], s[9], s[10], s[11]],
        [s[12], s[13], s[14], s[15]],
    ]
}
/// Compute inverse calibrated plane transforms on the CPU.
pub fn inverse_models(scene: &SceneGeometry) -> (Matrix4<f32>, Matrix4<f32>) {
    (
        scene
            .model_matrix_left()
            .try_inverse()
            .expect("left model is invertible"),
        scene
            .model_matrix_right()
            .try_inverse()
            .expect("right model is invertible"),
    )
}
/// Convert output UV to shader yaw/pitch.
pub fn output_uv_to_yaw_pitch(uv: [f32; 2], c: CylindricalStereoProjectionConfig) -> [f32; 2] {
    [
        c.yaw_center_rad + (uv[0] - 0.5) * c.yaw_span_rad,
        ((0.5 - uv[1]) * 2.0 * (c.vertical_fov_rad * 0.5).tan()).atan(),
    ]
}
/// Convert shader yaw/pitch to output UV.
pub fn yaw_pitch_to_output_uv(v: [f32; 2], c: CylindricalStereoProjectionConfig) -> [f32; 2] {
    [
        0.5 + (v[0] - c.yaw_center_rad) / c.yaw_span_rad,
        0.5 - v[1].tan() / (2.0 * (c.vertical_fov_rad * 0.5).tan()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::PlaneLayout;
    use nalgebra::{Point3, Vector3};

    #[test]
    #[ignore = "requires a GPU adapter"]
    fn cylindrical_pipeline_compiles() {
        let gpu = GpuContext::new_blocking().expect("test GPU");
        let _renderer = CylindricalRenderer::new(
            &gpu,
            wgpu::TextureFormat::Rgba8Unorm,
            InputFormat::Yuv420p,
            CylindricalStereoProjectionConfig::default(),
        );
    }
    #[test]
    fn output_angle_roundtrip_across_grid() {
        let c = Default::default();
        for y in 0..=16 {
            for x in 0..=32 {
                let uv = [x as f32 / 32.0, y as f32 / 16.0];
                let b = yaw_pitch_to_output_uv(output_uv_to_yaw_pitch(uv, c), c);
                assert!((uv[0] - b[0]).abs() < 1e-5 && (uv[1] - b[1]).abs() < 1e-5);
            }
        }
    }
    #[test]
    fn rays_intersect_both_planes_at_original_local_point() {
        let l = PlaneLayout {
            camera_axis_offset: 0.25,
            intersect: 0.5,
            x_ty: 0.01,
            x_rz: 0.02,
            z_rx: -0.01,
            x_rx: 0.01,
            z_rz: -0.02,
        };
        let s = SceneGeometry::from_layout_with_aspect(&l, 16.0 / 9.0);
        let o = Point3::from(s.camera_position);
        let local = Point3::new(0.1, -0.08, 0.0);
        let inv = inverse_models(&s);
        for (m, i) in [s.model_matrix_left(), s.model_matrix_right()]
            .into_iter()
            .zip([inv.0, inv.1])
        {
            let d: Vector3<f32> = (m.transform_point(&local) - o).normalize();
            let lo = i.transform_point(&o);
            let ld = i.transform_vector(&d);
            assert!((lo + ld * (-lo.z / ld.z) - local).norm() < 1e-5);
        }
    }
}
