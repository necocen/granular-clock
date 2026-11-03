use mikage::glam::Vec3;
use mikage::{SceneLightColors, SceneLighting};

/// Camera initial settings (startup-only)
#[derive(Clone, Copy, Debug)]
pub struct CameraSettings {
    /// Camera world position
    pub position: Vec3,
    /// Look-at target point
    pub target: Vec3,
}

impl Default for CameraSettings {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.65, 0.5, 0.65),
            target: Vec3::ZERO,
        }
    }
}

/// Scene light initial settings (startup-only)
#[derive(Clone, Copy, Debug)]
pub struct LightSettings {
    pub lighting: SceneLighting,
    pub colors: SceneLightColors,
    pub shadows_enabled: bool,
    pub shadow_map_resolution: u32,
    /// Render and composite in linear HDR before display tone mapping.
    pub hdr_enabled: bool,
    /// Display exposure in stops (gain = 2^EV), applied before tone mapping.
    pub exposure_ev: f32,
}

impl Default for LightSettings {
    fn default() -> Self {
        Self {
            lighting: SceneLighting {
                direction: Vec3::new(-2.0, 4.0, 3.0),
                ambient_intensity: 0.4,
                directional_intensity: 0.8,
            },
            colors: SceneLightColors::default(),
            shadows_enabled: true,
            shadow_map_resolution: 1024,
            hdr_enabled: true,
            exposure_ev: 0.5,
        }
    }
}
