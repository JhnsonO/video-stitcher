//! GPU data contract and mathematical helpers for cylindrical stereo rendering.
//!
//! This renderer is deliberately parallel to the perspective renderer: it owns a
//! full-screen pipeline and never changes the established L-shape implementation.

use bytemuck::{Pod, Zeroable};
use nalgebra::Matrix4;

use super::scene::SceneGeometry;
use crate::projection::CylindricalStereoProjectionConfig;

/// Uniform data consumed by `cylindrical_stereo.wgsl` for one camera.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct CylindricalCameraUniforms {
    inv_model: [[f32; 4]; 4],
    intrinsics: [f32; 4],
    dist: [f32; 4],
    color_scale: [f32; 4],
    color_offset: [f32; 4],
    flags: [u32; 4],
}
/// Build the CPU-computed inverse plane transforms. Keeping inversion here
/// avoids expensive and numerically fragile per-fragment matrix inversion.
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
/// Convert an output UV to the yaw and pitch sampled by the shader.
pub fn output_uv_to_yaw_pitch(uv: [f32; 2], config: CylindricalStereoProjectionConfig) -> [f32; 2] {
    [
        config.yaw_center_rad + (uv[0] - 0.5) * config.yaw_span_rad,
        ((0.5 - uv[1]) * 2.0 * (config.vertical_fov_rad * 0.5).tan()).atan(),
    ]
}
/// Convert yaw and pitch back to output UV.
pub fn yaw_pitch_to_output_uv(
    value: [f32; 2],
    config: CylindricalStereoProjectionConfig,
) -> [f32; 2] {
    [
        0.5 + (value[0] - config.yaw_center_rad) / config.yaw_span_rad,
        0.5 - value[1].tan() / (2.0 * (config.vertical_fov_rad * 0.5).tan()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::PlaneLayout;
    use nalgebra::{Point3, Vector3};

    #[test]
    fn output_angle_roundtrip_across_grid() {
        let config = CylindricalStereoProjectionConfig::default();
        for y in 0..=16 {
            for x in 0..=32 {
                let uv = [x as f32 / 32.0, y as f32 / 16.0];
                let back = yaw_pitch_to_output_uv(output_uv_to_yaw_pitch(uv, config), config);
                assert!((uv[0] - back[0]).abs() < 1e-5);
                assert!((uv[1] - back[1]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn rays_intersect_both_planes_at_original_local_point() {
        let layout = PlaneLayout {
            camera_axis_offset: 0.25,
            intersect: 0.5,
            x_ty: 0.01,
            x_rz: 0.02,
            z_rx: -0.01,
            x_rx: 0.01,
            z_rz: -0.02,
        };
        let scene = SceneGeometry::from_layout_with_aspect(&layout, 16.0 / 9.0);
        let origin = Point3::from(scene.camera_position);
        let local = Point3::new(0.1, -0.08, 0.0);
        let models = [scene.model_matrix_left(), scene.model_matrix_right()];
        let inverses = inverse_models(&scene);
        for (model, inverse) in models.into_iter().zip([inverses.0, inverses.1]) {
            let world = model.transform_point(&local);
            let direction: Vector3<f32> = (world - origin).normalize();
            let o = inverse.transform_point(&origin);
            let d = inverse.transform_vector(&direction);
            let hit = o + d * (-o.z / d.z);
            assert!((hit - local).norm() < 1e-5, "hit={hit:?}");
        }
    }
}
