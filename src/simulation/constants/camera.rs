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
    /// Ambient light intensity in legacy config units (10500 = unit gain).
    pub ambient_brightness: f32,
    /// Directional light intensity in legacy config units (10500 = unit gain).
    pub directional_illuminance: f32,
    /// Linear RGB colors, independent of light intensity.
    pub ambient_color: Vec3,
    pub directional_color: Vec3,
    /// Directional light position
    pub directional_position: Vec3,
    /// Directional light target point
    pub directional_target: Vec3,
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
            ambient_brightness: 500.0,
            directional_illuminance: 10000.0,
            ambient_color: Vec3::ONE,
            directional_color: Vec3::ONE,
            directional_position: Vec3::new(1.0, 2.0, 1.0),
            directional_target: Vec3::ZERO,
            shadows_enabled: true,
            shadow_map_resolution: 1024,
            hdr_enabled: true,
            exposure_ev: 0.0,
        }
    }
}
