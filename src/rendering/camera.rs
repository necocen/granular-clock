use bevy::camera_controller::free_camera::{FreeCamera, FreeCameraPlugin};
use bevy::prelude::*;
use bevy::render::view::NoIndirectDrawing;

/// メインカメラのマーカー
#[derive(Component)]
pub struct MainCamera;

/// FreeCameraPluginを返す
pub fn camera_plugin() -> FreeCameraPlugin {
    FreeCameraPlugin
}

/// カメラをセットアップ
pub fn setup_camera(mut commands: Commands) {
    commands.spawn((
        MainCamera,
        Camera3d::default(),
        FreeCamera::default(),
        Transform::from_xyz(0.65, 0.5, 0.65).looking_at(Vec3::new(0.0, 0.075, 0.0), Vec3::Y),
        NoIndirectDrawing,
    ));

    // 環境光
    commands.spawn(AmbientLight {
        color: Color::WHITE,
        brightness: 500.0,
        affects_lightmapped_meshes: true,
    });

    // ディレクショナルライト
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(1.0, 2.0, 1.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}
