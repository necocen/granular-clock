use bevy::prelude::*;
use bevy::render::view::NoIndirectDrawing;
use bevy_egui::PrimaryEguiContext;
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

use crate::simulation::constants::{CameraSettings, LightSettings};

/// メインカメラのマーカー
#[derive(Component)]
pub struct MainCamera;

/// PanOrbitCameraPluginを返す
pub fn camera_plugin() -> PanOrbitCameraPlugin {
    PanOrbitCameraPlugin
}

/// カメラをセットアップ
pub fn setup_camera(
    mut commands: Commands,
    camera: Res<CameraSettings>,
    light: Res<LightSettings>,
) {
    commands.spawn((
        MainCamera,
        Camera3d::default(),
        // Attach Egui explicitly to avoid relying on auto-created primary context selection.
        PrimaryEguiContext,
        PanOrbitCamera::default(),
        Transform::from_translation(camera.position).looking_at(camera.target, Vec3::Y),
        NoIndirectDrawing,
    ));

    // 環境光
    commands.spawn(AmbientLight {
        color: Color::WHITE,
        brightness: light.ambient_brightness,
        affects_lightmapped_meshes: true,
    });

    // ディレクショナルライト
    commands.spawn((
        DirectionalLight {
            illuminance: light.directional_illuminance,
            shadows_enabled: light.shadows_enabled,
            ..default()
        },
        Transform::from_translation(light.directional_position)
            .looking_at(light.directional_target, Vec3::Y),
    ));
}
