mod analysis;
mod debug;
mod physics;
mod rendering;
mod simulation;
mod ui;

use std::path::PathBuf;

#[cfg(target_family = "wasm")]
use bevy::asset::{AssetMetaCheck, AssetPlugin};
use bevy::prelude::*;
#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::wasm_bindgen;

use analysis::{update_distribution, CurrentDistribution, DistributionHistory};
// use debug::debug_particles; // 必要時のみ有効化
use physics::{
    cpu::{run_physics_substeps, InstanceCpuWriterPlugin},
    gpu::{apply_gpu_results, GpuInstanceWriterPlugin, GpuPhysicsPlugin},
    init_spatial_hash_grid, ContactHistory, ParticleStore,
};
use rendering::{
    camera_plugin, is_cpu_backend, is_gpu_backend, setup_camera, setup_rendering, spawn_particles,
    update_container_transforms, GpuInstancingPlugin, RenderExtractResourcesPlugin,
};
use simulation::{
    config_toml::resolve_startup_config, constants::PhysicsBackend, state::SimulationState,
};
use ui::UiPlugin;

#[cfg(target_family = "wasm")]
pub use wasm_bindgen_rayon::init_thread_pool;

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub fn run() {
    run_with_config_path(None);
}

pub fn run_with_config_path(config_path: Option<PathBuf>) {
    let loaded = resolve_startup_config(config_path.as_deref());

    let primary_window = Window {
        title: "Granular Clock".into(),
        resolution: (1280u32, 960u32).into(),
        #[cfg(target_family = "wasm")]
        fit_canvas_to_parent: true,
        ..default()
    };

    let default_plugins = DefaultPlugins.set(WindowPlugin {
        primary_window: Some(primary_window),
        ..default()
    });

    #[cfg(target_family = "wasm")]
    let default_plugins = default_plugins.set(AssetPlugin {
        meta_check: AssetMetaCheck::Never,
        ..default()
    });

    let mut app = App::new();
    app.add_plugins(default_plugins)
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
        // UI
        .add_plugins(UiPlugin)
        // リソース
        .insert_resource(loaded.simulation)
        .insert_resource(loaded.ui_ranges)
        .insert_resource(PhysicsBackend::default())
        .insert_resource(ContactHistory::default())
        .insert_resource(DistributionHistory::default())
        .insert_resource(CurrentDistribution::default())
        .insert_resource(SimulationState::default())
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
        // デバッグ出力は必要時のみ有効化
        // .add_systems(Update, debug_particles)
        ;

    for warning in loaded.warnings {
        warn!("{warning}");
    }

    app.run();
}
