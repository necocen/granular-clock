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
    /// Ambient light weight, relative to directional_illuminance.
    pub ambient_brightness: f32,
    /// Directional light weight, relative to ambient_brightness.
    pub directional_illuminance: f32,
    /// Directional light position
    pub directional_position: Vec3,
    /// Directional light target point
    pub directional_target: Vec3,
    /// Retained for config compatibility; mikage does not support shadows.
    pub shadows_enabled: bool,
}

impl Default for LightSettings {
    fn default() -> Self {
        Self {
            ambient_brightness: 500.0,
            directional_illuminance: 10000.0,
            directional_position: Vec3::new(1.0, 2.0, 1.0),
            directional_target: Vec3::ZERO,
            shadows_enabled: false,
        }
    }
}
