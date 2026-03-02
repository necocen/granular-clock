use bevy::prelude::*;

/// Camera initial settings (startup-only)
#[derive(Resource, Clone, Copy, Debug)]
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
            target: Vec3::new(0.0, 0.075, 0.0),
        }
    }
}

/// Scene light initial settings (startup-only)
#[derive(Resource, Clone, Copy, Debug)]
pub struct LightSettings {
    /// Ambient light brightness
    pub ambient_brightness: f32,
    /// Directional light illuminance
    pub directional_illuminance: f32,
    /// Directional light position
    pub directional_position: Vec3,
    /// Directional light target point
    pub directional_target: Vec3,
    /// Whether directional shadows are enabled
    pub shadows_enabled: bool,
}

impl Default for LightSettings {
    fn default() -> Self {
        Self {
            ambient_brightness: 500.0,
            directional_illuminance: 10000.0,
            directional_position: Vec3::new(1.0, 2.0, 1.0),
            directional_target: Vec3::ZERO,
            shadows_enabled: true,
        }
    }
}
