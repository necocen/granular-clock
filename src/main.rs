mod analysis;
mod debug;
mod gpu;
mod physics;
mod rendering;
mod simulation;
mod ui;

use bevy::prelude::*;

use analysis::{update_distribution, CurrentDistribution, DistributionHistory};
use gpu::{apply_gpu_results, GpuInstanceWriterPlugin, GpuPhysicsPlugin};
// use debug::debug_particles; // 必要時のみ有効化
use physics::{
    init_spatial_hash_grid, ContactHistory, MaterialProperties, ParticleStore, PhysicsConstants,
    WallProperties,
};
use rendering::{
    camera_plugin, is_cpu_backend, is_gpu_backend, setup_camera, setup_rendering, spawn_particles,
    update_container_transforms, GpuInstancingPlugin, InstanceCpuWriterPlugin,
    RenderExtractResourcesPlugin,
};
use simulation::{
    run_physics_substeps, Container, OscillationParams, PhysicsBackend, SimulationConfig,
    SimulationSettings, SimulationState, SimulationTime,
};
use ui::{
    handle_amplitude_buttons, handle_control_buttons, handle_frequency_buttons,
    handle_oscillation_toggle, handle_physics_backend_toggle, handle_reset, setup_bevy_ui_controls,
    setup_distribution_graph, update_button_colors, update_distribution_display,
    update_fps_display, update_graph_lines, update_simulation_time_display,
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
        // Shared (CPU/GPU 共通): Main→Render 抽出
        .add_plugins(RenderExtractResourcesPlugin)
        // GPU physics / GPU instance write
        .add_plugins(GpuPhysicsPlugin)
        .add_plugins(GpuInstanceWriterPlugin)
        // CPU instance write
        .add_plugins(InstanceCpuWriterPlugin)
        // Shared particle draw path
        .add_plugins(GpuInstancingPlugin)
        // リソース
        .insert_resource(SimulationConfig::default())
        .insert_resource(Container::default())
        .insert_resource(OscillationParams::default())
        .insert_resource(PhysicsConstants::default())
        .insert_resource(MaterialProperties::default())
        .insert_resource(WallProperties::default())
        .insert_resource(PhysicsBackend::default())
        .insert_resource(ContactHistory::default())
        .insert_resource(DistributionHistory::default())
        .insert_resource(CurrentDistribution::default())
        .insert_resource(SimulationState::default())
        .insert_resource(SimulationTime::default())
        .insert_resource(SimulationSettings::default())
        .insert_resource(ParticleStore::default())
        // スタートアップシステム（カメラはPreStartupで先に生成）
        .add_systems(PreStartup, setup_camera)
        .add_systems(PreStartup, init_spatial_hash_grid)
        .add_systems(Startup, setup_rendering)
        .add_systems(Startup, spawn_particles.after(setup_rendering))
        // 物理サブステップ（CPU物理）
        .add_systems(Update, run_physics_substeps.run_if(is_cpu_backend))
        // GPU 物理結果の適用
        .add_systems(Update, apply_gpu_results.run_if(is_gpu_backend))
        .add_systems(Update, update_container_transforms)
        .add_systems(Update, update_distribution)
        .add_systems(Update, handle_reset)
        // デバッグ出力は必要時のみ有効化
        // .add_systems(Update, debug_particles)
        // bevy_uiベースのコントロールパネル
        .add_systems(Startup, setup_bevy_ui_controls)
        .add_systems(Startup, setup_distribution_graph)
        .add_systems(Update, handle_oscillation_toggle)
        .add_systems(Update, handle_physics_backend_toggle)
        .add_systems(Update, handle_amplitude_buttons)
        .add_systems(Update, handle_frequency_buttons)
        .add_systems(Update, handle_control_buttons)
        .add_systems(Update, update_button_colors)
        .add_systems(Update, update_distribution_display)
        .add_systems(Update, update_graph_lines)
        .add_systems(Update, update_simulation_time_display)
        .add_systems(Update, update_fps_display)
        .run();
}
