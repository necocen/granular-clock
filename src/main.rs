mod analysis;
mod debug;
mod physics;
mod rendering;
mod simulation;
mod ui;

use bevy::prelude::*;

use analysis::{update_distribution, CurrentDistribution, DistributionHistory};
// use debug::debug_particles; // 必要時のみ有効化
use physics::{
    ContactHistory, MaterialProperties, PhysicsConstants, SpatialHashGrid, WallProperties,
};
use rendering::{
    camera_plugin, setup_camera, setup_rendering, spawn_particles, sync_transforms,
    update_container_transforms, SimulationConfig,
};
use simulation::{
    run_physics_substeps, Container, OscillationParams, SimulationSettings, SimulationTime,
};
use ui::{
    handle_amplitude_buttons, handle_control_buttons, handle_frequency_buttons,
    handle_oscillation_toggle, handle_reset, setup_bevy_ui_controls,
    setup_distribution_graph, update_button_colors, update_distribution_display,
    update_graph_lines, SimulationState,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Granular Clock".into(),
                resolution: (1280u32, 720u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(camera_plugin())
        // リソース
        .insert_resource(SimulationConfig::default())
        .insert_resource(Container::default())
        .insert_resource(OscillationParams::default())
        .insert_resource(PhysicsConstants::default())
        .insert_resource(MaterialProperties::default())
        .insert_resource(WallProperties::default())
        .insert_resource(SpatialHashGrid::new(0.05, 4096))
        .insert_resource(ContactHistory::default())
        .insert_resource(DistributionHistory::default())
        .insert_resource(CurrentDistribution::default())
        .insert_resource(SimulationState::default())
        .insert_resource(SimulationTime::default())
        .insert_resource(SimulationSettings::default())
        // スタートアップシステム（カメラはPreStartupで先に生成）
        .add_systems(PreStartup, setup_camera)
        .add_systems(Startup, setup_rendering)
        .add_systems(Startup, spawn_particles.after(setup_rendering))
        // 物理サブステップ（Updateで実行、内部でサブステップを回す）
        .add_systems(Update, run_physics_substeps)
        // 更新システム
        .add_systems(Update, sync_transforms)
        .add_systems(Update, update_container_transforms)
        .add_systems(Update, update_distribution)
        .add_systems(Update, handle_reset)
        // デバッグ出力は必要時のみ有効化
        // .add_systems(Update, debug_particles)
        // bevy_uiベースのコントロールパネル
        .add_systems(Startup, setup_bevy_ui_controls)
        .add_systems(Startup, setup_distribution_graph)
        .add_systems(Update, handle_oscillation_toggle)
        .add_systems(Update, handle_amplitude_buttons)
        .add_systems(Update, handle_frequency_buttons)
        .add_systems(Update, handle_control_buttons)
        .add_systems(Update, update_button_colors)
        .add_systems(Update, update_distribution_display)
        .add_systems(Update, update_graph_lines)
        .run();
}
