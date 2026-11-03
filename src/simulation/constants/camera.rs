use mikage::SceneLighting;
use mikage::glam::Vec3;

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
    pub shadows_enabled: bool,
}

impl Default for LightSettings {
    fn default() -> Self {
        Self {
            lighting: SceneLighting {
                direction: Vec3::new(-2.0, 4.0, 3.0),
                ambient_intensity: 0.4,
                directional_intensity: 0.8,
            },
            shadows_enabled: true,
        }
    }
}
