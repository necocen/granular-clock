use mikage::{Camera, SceneUniform};

use crate::simulation::constants::LightSettings;

/// Map the configured light weights to mikage's normalized Lambert lighting.
pub fn scene_uniform(camera: &impl Camera, aspect: f32, light: &LightSettings) -> SceneUniform {
    // Sum in f64 so even large, finite configuration values stay finite.
    let total = light.ambient_brightness as f64 + light.directional_illuminance as f64;
    let ambient = if total > 0.0 {
        (light.ambient_brightness as f64 / total) as f32
    } else {
        0.0
    };
    let mut uniform = SceneUniform::with_light(
        camera.view_projection_matrix(aspect),
        camera.position(),
        light.directional_position - light.directional_target,
        ambient,
    );
    if light.directional_illuminance == 0.0 {
        uniform.light_dir = [0.0; 4];
    }
    uniform
}

#[cfg(test)]
mod tests {
    use super::*;
    use mikage::{OrbitCamera, glam::Vec3};

    #[test]
    fn configured_direction_and_relative_brightness_are_used() {
        let light = LightSettings {
            ambient_brightness: 500.0,
            directional_illuminance: 10000.0,
            directional_position: Vec3::new(3.0, 5.0, 7.0),
            directional_target: Vec3::new(3.0, 3.0, 7.0),
            ..Default::default()
        };
        let mut camera = OrbitCamera::default();
        let first = scene_uniform(&camera, 1.5, &light);
        assert_eq!(first.light_dir, [0.0, 1.0, 0.0, 0.0]);
        assert!((first.ambient[0] - 500.0 / 10500.0).abs() < 1e-6);

        camera.yaw += 0.5;
        let moved = scene_uniform(&camera, 1.5, &light);
        assert_ne!(first.view_proj, moved.view_proj);
        assert_eq!(first.light_dir, moved.light_dir);
        assert_eq!(first.ambient, moved.ambient);
    }

    #[test]
    fn lights_can_be_disabled_independently_or_together() {
        for (ambient, directional, expected_ambient) in
            [(500.0, 0.0, 1.0), (0.0, 10000.0, 0.0), (0.0, 0.0, 0.0)]
        {
            let light = LightSettings {
                ambient_brightness: ambient,
                directional_illuminance: directional,
                ..Default::default()
            };
            let uniform = scene_uniform(&OrbitCamera::default(), 1.0, &light);
            assert_eq!(uniform.ambient[0], expected_ambient);
            assert_eq!(uniform.light_dir == [0.0; 4], directional == 0.0);
        }
    }

    #[test]
    fn large_light_weights_do_not_overflow() {
        let light = LightSettings {
            ambient_brightness: f32::MAX,
            directional_illuminance: f32::MAX,
            ..Default::default()
        };
        let uniform = scene_uniform(&OrbitCamera::default(), 1.0, &light);
        assert_eq!(uniform.ambient[0], 0.5);
    }
}
