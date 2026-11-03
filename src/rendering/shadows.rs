use mikage::glam::{
    Mat4, Vec3,
    camera::rh::{proj::directx, view},
};

use crate::simulation::constants::SimulationConstants;

/// Fit the entire container and its possible oscillation in the light volume.
pub fn shadow_matrix(constants: &SimulationConstants, direction: Vec3) -> Mat4 {
    let direction = direction.normalize_or(Vec3::Y);
    // A vertical light needs an up vector that is not parallel to its direction.
    let up = if direction.dot(Vec3::Y).abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let padding = constants
        .particle
        .large_radius
        .max(constants.particle.small_radius)
        + 0.01;
    let extents = constants.container.half_extents
        + Vec3::Y * constants.oscillation.amplitude.abs()
        + Vec3::splat(padding);
    let radius = extents.length().max(0.01);
    let center = constants.container.base_position;
    directx::orthographic(-radius, radius, -radius, radius, 0.001, radius * 4.0)
        * view::look_at_mat4(center + direction * (radius * 2.0), center, up)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadow_volume_contains_container_and_oscillation_for_vertical_and_oblique_lights() {
        let mut constants = SimulationConstants::default();
        constants.container.base_position = Vec3::new(1.0, 2.0, 3.0);
        let bounds =
            constants.container.half_extents + Vec3::Y * constants.oscillation.amplitude.abs();
        for direction in [Vec3::Y, -Vec3::Y, Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0)] {
            let matrix = shadow_matrix(&constants, direction);
            assert!(matrix.is_finite());
            for x in [-1.0, 1.0] {
                for y in [-1.0, 1.0] {
                    for z in [-1.0, 1.0] {
                        let corner =
                            constants.container.base_position + bounds * Vec3::new(x, y, z);
                        let clip = matrix.project_point3(corner);
                        assert!(clip.x.abs() < 1.0 && clip.y.abs() < 1.0);
                        assert!((0.0..=1.0).contains(&clip.z));
                    }
                }
            }
        }
    }
}
