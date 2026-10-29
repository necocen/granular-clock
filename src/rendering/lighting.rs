use mikage::SceneLighting;

use crate::simulation::constants::LightSettings;

/// Convert the legacy configuration units to independent, unitless light gains.
pub fn scene_lighting(light: &LightSettings) -> SceneLighting {
    // Preserve the previous default gains (500 / 10500 and 10000 / 10500).
    // This is a fixed scale: changing one light no longer changes the other.
    const LIGHT_UNITS_PER_GAIN: f32 = 10500.0;
    SceneLighting {
        direction: light.directional_position - light.directional_target,
        ambient_intensity: light.ambient_brightness / LIGHT_UNITS_PER_GAIN,
        directional_intensity: light.directional_illuminance / LIGHT_UNITS_PER_GAIN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mikage::glam::Vec3;

    #[test]
    fn configured_direction_and_default_brightness_are_preserved() {
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
}
