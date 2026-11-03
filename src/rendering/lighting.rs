use mikage::SceneLighting;
use mikage::glam::{
    Mat4, Vec3,
    camera::rh::{proj::directx, view},
};

use crate::simulation::constants::{LightSettings, SimulationConstants};

/// Convert the legacy configuration units to independent, unitless light gains.
pub fn scene_lighting(light: &LightSettings) -> SceneLighting {
    // Retain the existing configuration units: 10500 corresponds to unit gain.
    // This is a fixed scale: changing one light no longer changes the other.
    const LIGHT_UNITS_PER_GAIN: f32 = 10500.0;
    SceneLighting {
        direction: light.directional_position - light.directional_target,
        ambient_intensity: light.ambient_brightness / LIGHT_UNITS_PER_GAIN,
        directional_intensity: light.directional_illuminance / LIGHT_UNITS_PER_GAIN,
    }
}

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
    use mikage::glam::Vec3;

    #[test]
    fn legacy_light_units_preserve_gain_and_direction() {
        let light = LightSettings {
            ambient_brightness: 500.0,
            directional_illuminance: 10000.0,
            directional_position: Vec3::new(3.0, 5.0, 7.0),
            directional_target: Vec3::new(3.0, 3.0, 7.0),
            ..Default::default()
        };
        let lighting = scene_lighting(&light);
        assert_eq!(lighting.direction, Vec3::new(0.0, 2.0, 0.0));
        assert_eq!(lighting.ambient_intensity, 500.0 / 10500.0);
        assert_eq!(lighting.directional_intensity, 10000.0 / 10500.0);
    }

    #[test]
    fn lights_can_be_disabled_independently_or_together() {
        for (ambient, directional) in [(500.0, 0.0), (0.0, 10000.0), (0.0, 0.0)] {
            let light = LightSettings {
                ambient_brightness: ambient,
                directional_illuminance: directional,
                ..Default::default()
            };
            let lighting = scene_lighting(&light);
            assert_eq!(lighting.ambient_intensity, ambient / 10500.0);
            assert_eq!(lighting.directional_intensity, directional / 10500.0);
        }
    }

    #[test]
    fn light_intensities_scale_independently() {
        let mut light = LightSettings::default();
        let original = scene_lighting(&light);
        light.ambient_brightness *= 2.0;
        let ambient_changed = scene_lighting(&light);
        assert_eq!(
            ambient_changed.ambient_intensity,
            original.ambient_intensity * 2.0
        );
        assert_eq!(
            ambient_changed.directional_intensity,
            original.directional_intensity
        );

        light.directional_illuminance *= 2.0;
        let both_changed = scene_lighting(&light);
        assert_eq!(
            both_changed.ambient_intensity,
            original.ambient_intensity * 2.0
        );
        assert_eq!(
            both_changed.directional_intensity,
            original.directional_intensity * 2.0
        );
    }

    #[test]
    fn large_light_values_remain_finite() {
        let light = LightSettings {
            ambient_brightness: f32::MAX,
            directional_illuminance: f32::MAX,
            ..Default::default()
        };
        let lighting = scene_lighting(&light);
        assert!(lighting.ambient_intensity.is_finite());
        assert!(lighting.directional_intensity.is_finite());
        assert!(lighting.ambient_intensity > 1.0);
        assert!(lighting.directional_intensity > 1.0);
    }

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
