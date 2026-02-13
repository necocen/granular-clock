use bevy::{
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_graph::RenderGraph,
        render_resource::{CommandEncoderDescriptor, MapMode},
        renderer::{RenderDevice, RenderQueue},
        Render, RenderApp,
    },
};
use std::sync::Arc;

use crate::physics::{
    MaterialProperties, ParticleSize, ParticleStore, PhysicsConstants, WallProperties,
};
use crate::simulation::{
    advance_oscillation, ContainerParams, OscillationParams, PhysicsBackend, SimulationSettings,
    SimulationState, SimulationTimeParams,
};

use super::{
    buffers::{GpuPhysicsBuffers, ParticleGpu, SimulationParams},
    node::{GpuPhysicsLabel, GpuPhysicsNode},
    pipeline::GpuPhysicsPipelines,
    readback::{GpuReadbackBuffer, ReadbackSettings, ReadbackStaging},
};
use crate::physics::GridSettings;

/// GPU 物理が有効かどうかのフラグ
#[allow(dead_code)]
#[derive(Resource, Clone, Default)]
pub struct GpuPhysicsEnabled(pub bool);

/// Main World で管理する GPU 物理用データ
#[derive(Resource, Clone, Default)]
pub struct GpuParticleData {
    /// 粒子データ（CPU側、Arc で clone を O(1) に）
    pub particles: Arc<Vec<ParticleGpu>>,
    /// シミュレーションパラメータ
    pub params: SimulationParams,
    /// 初回データ転送済みか
    pub initialized: bool,
    /// データ世代（抽出のたびにインクリメント）
    pub generation: u64,
    /// 一時停止中か（Render World に伝搬するためここに持つ）
    pub paused: bool,
    /// 1フレームあたりのサブステップ数（Render World に伝搬）
    pub substeps: u32,
    /// 最後に確認した ParticleStore の世代番号
    pub last_store_generation: u64,
}

impl ExtractResource for GpuParticleData {
    type Source = GpuParticleData;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone() // Arc clone は O(1)
    }
}

/// コンテナパラメータの抽出用リソース
#[derive(Resource, Clone, Default)]
pub struct ExtractedContainerParams {
    pub container_offset: f32,
    pub base_position_y: f32,
    pub oscillation_enabled: bool,
    pub oscillation_amplitude: f32,
    pub oscillation_frequency: f32,
    pub oscillation_phase_start: f32,
}

impl ExtractResource for ExtractedContainerParams {
    type Source = ExtractedContainerParams;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

// GpuReadbackBuffer を Render World に抽出
impl ExtractResource for GpuReadbackBuffer {
    type Source = GpuReadbackBuffer;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

/// GPU 物理プラグイン
pub struct GpuPhysicsPlugin;

impl Plugin for GpuPhysicsPlugin {
    fn build(&self, app: &mut App) {
        // 読み戻しバッファを Main World に追加
        let readback_buffer = GpuReadbackBuffer::default();

        app.insert_resource(GpuParticleData::default())
            .insert_resource(GpuPhysicsEnabled(true))
            .insert_resource(ExtractedContainerParams::default())
            .insert_resource(GridSettings::default())
            .insert_resource(ReadbackSettings::default())
            .insert_resource(readback_buffer)
            .add_plugins(ExtractResourcePlugin::<GpuParticleData>::default())
            .add_plugins(ExtractResourcePlugin::<ExtractedContainerParams>::default())
            .add_plugins(ExtractResourcePlugin::<GpuReadbackBuffer>::default())
            .add_systems(
                Update,
                update_oscillation_for_gpu
                    .run_if(|backend: Res<PhysicsBackend>| *backend == PhysicsBackend::Gpu),
            )
            .add_systems(PostUpdate, extract_particle_data)
            .add_systems(PostUpdate, update_container_params)
            .add_systems(Update, update_simulation_time);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.add_systems(
            Render,
            (
                init_pipelines,
                prepare_gpu_buffers,
                update_params_only,
                copy_to_staging,
                process_readback,
            )
                .chain(),
        );

        // レンダーグラフにノードを追加
        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(GpuPhysicsLabel, GpuPhysicsNode::new());
    }
}

/// パイプラインを初期化（一度だけ）
fn init_pipelines(
    mut commands: Commands,
    pipelines: Option<Res<GpuPhysicsPipelines>>,
    pipeline_cache: Res<bevy::render::render_resource::PipelineCache>,
    asset_server: Res<AssetServer>,
) {
    // 既に初期化済みならスキップ
    if pipelines.is_some() {
        return;
    }

    let new_pipelines = GpuPhysicsPipelines::create(&pipeline_cache, &asset_server);
    commands.insert_resource(new_pipelines);
}

/// コンテナのオフセットを更新
fn update_container_params(
    container_params: Res<ContainerParams>,
    osc_params: Res<OscillationParams>,
    sim_state: Res<SimulationState>,
    mut params: ResMut<ExtractedContainerParams>,
) {
    // デフォルト（フレーム基準）オフセット。GPUノード側でサブステップごとに上書きする。
    params.container_offset = container_params.base_position.y + sim_state.container_offset;
    params.base_position_y = container_params.base_position.y;
    params.oscillation_enabled = osc_params.enabled;
    params.oscillation_amplitude = osc_params.amplitude;
    params.oscillation_frequency = osc_params.frequency;
    params.oscillation_phase_start = sim_state.oscillation_frame_start_phase;
}

/// GPU モード用: CPU と同じ位相更新ロジックで振動を進める。
fn update_oscillation_for_gpu(
    mut sim_state: ResMut<SimulationState>,
    params: Res<OscillationParams>,
    time_params: Res<SimulationTimeParams>,
    settings: Res<SimulationSettings>,
) {
    if sim_state.paused {
        return;
    }

    sim_state.oscillation_frame_start_phase = sim_state.oscillation_phase;
    for _ in 0..settings.substeps_per_frame {
        advance_oscillation(&mut sim_state, &params, time_params.dt);
    }
}

/// Main World の粒子データを GpuParticleData に抽出（初回または粒子数変更時）
#[allow(clippy::too_many_arguments)]
fn extract_particle_data(
    store: Res<ParticleStore>,
    container_params: Res<ContainerParams>,
    sim_state: Res<SimulationState>,
    time_params: Res<SimulationTimeParams>,
    material: Res<MaterialProperties>,
    wall_props: Res<WallProperties>,
    physics: Res<PhysicsConstants>,
    grid_settings: Res<GridSettings>,
    backend: Res<PhysicsBackend>,
    mut gpu_data: ResMut<GpuParticleData>,
) {
    // ParticleStore の世代が変わったかチェック（spawn/clear でのみインクリメントされる）
    // GPU readback による書き戻しでは generation は変わらないので、巻き戻りが起きない
    let store_generation_changed = store.generation != gpu_data.last_store_generation;
    let backend_switched_to_gpu = backend.is_changed() && *backend == PhysicsBackend::Gpu;
    if gpu_data.initialized && !store_generation_changed && !backend_switched_to_gpu {
        return;
    }

    // 粒子データを更新
    let mut particles = Vec::with_capacity(store.len());
    for p in &store.particles {
        particles.push(ParticleGpu {
            pos: p.position.into(),
            radius: p.radius,
            vel: p.velocity.into(),
            mass_inv: 1.0 / p.mass,
            omega: p.angular_velocity.into(),
            inertia_inv: 1.0 / p.inertia,
            size_flag: match p.size {
                ParticleSize::Large => 1,
                ParticleSize::Small => 0,
            },
            _pad: [0; 3],
        });
    }

    info!(
        "extract_particle_data: extracted {} particles",
        particles.len(),
    );

    if particles.is_empty() {
        return;
    }

    // パラメータを更新（各種リソースから値を取得）
    let world_half = [
        container_params.half_extents.x,
        container_params.half_extents.y,
        container_params.half_extents.z,
    ];
    gpu_data.params = SimulationParams {
        dt: time_params.dt,
        gravity: physics.gravity.y, // Vec3のY成分を使用
        cell_size: grid_settings.cell_size,
        grid_dim: grid_settings.compute_grid_dim(world_half),
        world_half,
        num_particles: particles.len() as u32,
        youngs_modulus: material.youngs_modulus,
        poisson_ratio: material.poisson_ratio,
        restitution: material.restitution,
        friction: material.friction,
        container_offset: container_params.base_position.y + sim_state.container_offset,
        divider_height: container_params.divider_height,
        container_half_x: container_params.half_extents.x,
        container_half_y: container_params.half_extents.y,
        container_half_z: container_params.half_extents.z,
        divider_thickness: container_params.divider_thickness,
        rolling_friction: material.rolling_friction,
        wall_restitution: wall_props.restitution,
        wall_friction: wall_props.friction,
        wall_damping: wall_props.damping,
        wall_stiffness: wall_props.stiffness,
        _pad_end: 0.0,
    };

    gpu_data.particles = Arc::new(particles);
    gpu_data.initialized = true;
    gpu_data.generation += 1;
    gpu_data.last_store_generation = store.generation;
}

/// シミュレーション時間を更新（GPU物理実行時）
fn update_simulation_time(
    mut gpu_data: ResMut<GpuParticleData>,
    mut sim_state: ResMut<SimulationState>,
    time_params: Res<SimulationTimeParams>,
    settings: Res<SimulationSettings>,
    backend: Res<PhysicsBackend>,
) {
    // paused 状態と substeps を GpuParticleData に同期（Render World に伝搬）
    gpu_data.paused = sim_state.paused;
    gpu_data.substeps = settings.substeps_per_frame;

    // CPU モードでは run_physics_substeps が時間を進めるので、ここでは何もしない
    if *backend != PhysicsBackend::Gpu {
        return;
    }

    // 一時停止中は時間を進めない
    if sim_state.paused {
        return;
    }

    // GPU物理が初期化済みの場合、サブステップ数分だけシミュレーション時間を進める
    if gpu_data.initialized {
        for _ in 0..settings.substeps_per_frame {
            sim_state.step_time(time_params.dt);
        }
    }
}

/// GPU バッファを準備・更新（再確保と、世代変更時の粒子データアップロード）
/// フレームごとの params 更新は `update_params_only` が担当する。
fn prepare_gpu_buffers(
    gpu_data: Res<GpuParticleData>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut buffers: Option<ResMut<GpuPhysicsBuffers>>,
    staging: Option<Res<ReadbackStaging>>,
    mut commands: Commands,
) {
    let num_particles = gpu_data.params.num_particles;
    if num_particles == 0 {
        return;
    }

    // bitonic sort の非2冪問題を避けるため、容量は常に 2 のべき乗で確保する
    let min_capacity = num_particles.max(2048);
    let desired_capacity = min_capacity
        .checked_next_power_of_two()
        .unwrap_or(min_capacity);

    // バッファがなければ作成
    if buffers.is_none() {
        let grid_size = gpu_data.params.grid_dim;
        let mut new_buffers = GpuPhysicsBuffers::new(&render_device, desired_capacity, grid_size);
        // 容量はバッファサイズだが、実際の粒子数は0（まだアップロードしていない）
        new_buffers.num_particles = 0;
        commands.insert_resource(new_buffers);

        // ステージングバッファも作成
        if staging.is_none() {
            let new_staging = ReadbackStaging::new(&render_device, desired_capacity);
            commands.insert_resource(new_staging);
        }
        return;
    }

    let buffers = buffers.as_mut().unwrap();

    // 粒子数が容量を超えたら再確保
    if num_particles > buffers.capacity {
        let grid_size = gpu_data.params.grid_dim;
        let mut new_buffers = GpuPhysicsBuffers::new(&render_device, desired_capacity, grid_size);
        new_buffers.num_particles = 0;
        commands.insert_resource(new_buffers);

        let needs_new_staging = staging
            .map(|s| s.size < std::mem::size_of::<ParticleGpu>() as u64 * desired_capacity as u64)
            .unwrap_or(true);
        if needs_new_staging {
            let new_staging = ReadbackStaging::new(&render_device, desired_capacity);
            commands.insert_resource(new_staging);
        }
        return;
    }

    // 世代が変わった場合（初回またはReset後）に粒子データを GPU に転送
    // A/B 固定スロット運用のため、両方へ同一データを書き込む。
    if !gpu_data.particles.is_empty() && buffers.last_uploaded_generation != gpu_data.generation {
        let particle_bytes = bytemuck::cast_slice(&gpu_data.particles);
        render_queue.write_buffer(&buffers.particles_a, 0, particle_bytes);
        render_queue.write_buffer(&buffers.particles_b, 0, particle_bytes);
        buffers.num_particles = num_particles;
        buffers.last_uploaded_generation = gpu_data.generation;
    }
}

/// 共通パラメータをフレーム単位で更新する。
/// サブステップごとの container_offset 上書きは node.run 側が担当する。
fn update_params_only(
    gpu_data: Res<GpuParticleData>,
    container_params: Res<ExtractedContainerParams>,
    render_queue: Res<RenderQueue>,
    buffers: Option<Res<GpuPhysicsBuffers>>,
) {
    let Some(buffers) = buffers else {
        return;
    };

    if buffers.num_particles == 0 {
        return;
    }

    // フレーム基準の container_offset を反映（サブステップ内で node が上書き）。
    let mut params = gpu_data.params;
    params.container_offset = container_params.container_offset;

    let params_bytes = bytemuck::bytes_of(&params);
    render_queue.write_buffer(&buffers.params, 0, params_bytes);
}

/// GPU 出力をステージングバッファにコピー（N フレームに1回）
fn copy_to_staging(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    buffers: Option<Res<GpuPhysicsBuffers>>,
    mut staging: Option<ResMut<ReadbackStaging>>,
    settings: Option<Res<ReadbackSettings>>,
) {
    let Some(buffers) = buffers else {
        return;
    };
    let Some(ref mut staging) = staging else {
        return;
    };

    if buffers.num_particles == 0 || staging.mapping_requested {
        return;
    }

    // フレームカウンタをインクリメントし、interval に達したかチェック
    let interval = settings.map(|s| s.interval).unwrap_or(1);
    staging.frame_counter += 1;
    if staging.frame_counter < interval {
        return; // まだ readback のタイミングではない
    }
    staging.frame_counter = 0; // カウンタをリセット

    let particle_size = std::mem::size_of::<ParticleGpu>() as u64;
    let copy_size = particle_size * buffers.num_particles as u64;

    if copy_size > staging.size {
        return;
    }

    // コピーコマンドを発行
    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("readback_copy_encoder"),
    });

    encoder.copy_buffer_to_buffer(buffers.latest_particles(), 0, &staging.buffer, 0, copy_size);

    render_queue.submit([encoder.finish()]);

    // マッピングをリクエスト
    staging.num_particles = buffers.num_particles;
    staging.mapping_requested = true;
    staging
        .mapping_complete
        .store(false, std::sync::atomic::Ordering::SeqCst);

    let mapping_complete = staging.mapping_complete.clone();
    let buffer_slice = staging.buffer.slice(0..copy_size);
    buffer_slice.map_async(MapMode::Read, move |result| {
        // マッピング完了時のコールバック
        if result.is_ok() {
            mapping_complete.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    });
}

/// 読み戻し処理（前フレームのマッピング結果を読み取り）
fn process_readback(
    mut staging: Option<ResMut<ReadbackStaging>>,
    readback_buffer: Option<Res<GpuReadbackBuffer>>,
) {
    let Some(ref mut staging) = staging else {
        return;
    };
    let Some(readback_buffer) = readback_buffer else {
        return;
    };

    if !staging.mapping_requested {
        return;
    }

    // マッピングが完了したかチェック
    if !staging
        .mapping_complete
        .load(std::sync::atomic::Ordering::SeqCst)
    {
        return; // まだ完了していない - 次フレームで再試行
    }

    let particle_size = std::mem::size_of::<ParticleGpu>();
    let num_particles = staging.num_particles as usize;
    let byte_len = particle_size * num_particles;

    // マッピング完了済みなので安全に読み取り可能
    let buffer_slice = staging.buffer.slice(0..byte_len as u64);
    let data = buffer_slice.get_mapped_range();

    // データを読み取り
    if data.len() >= byte_len {
        let particles: &[ParticleGpu] = bytemuck::cast_slice(&data[..byte_len]);

        // 共有バッファに書き込み
        if let Ok(mut guard) = readback_buffer.data.write() {
            guard.clear();
            guard.extend_from_slice(particles);
        }

        // フレームカウンタをインクリメント
        if let Ok(mut frame) = readback_buffer.frame.write() {
            *frame += 1;
        }
    }

    drop(data);
    staging.buffer.unmap();
    staging.mapping_requested = false;
    staging
        .mapping_complete
        .store(false, std::sync::atomic::Ordering::SeqCst);
}
