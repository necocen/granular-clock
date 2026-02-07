use bevy::{prelude::*, render::render_resource::*};

/// コンピュートパイプラインのリソース
#[derive(Resource)]
pub struct GpuPhysicsPipelines {
    /// バインドグループレイアウト記述子（パイプラインキャッシュから実際のレイアウトを取得するため）
    pub bind_group_layout_desc: BindGroupLayoutDescriptor,
    /// ハッシュキー生成パイプライン
    pub hash_keys_pipeline: CachedComputePipelineId,
    /// Bitonic sort パイプライン（複数ステップ）
    pub bitonic_sort_pipeline: CachedComputePipelineId,
    /// セル範囲構築パイプライン
    pub cell_ranges_pipeline: CachedComputePipelineId,
    /// 衝突検出・力計算パイプライン
    pub collision_pipeline: CachedComputePipelineId,
    /// 積分パイプライン
    pub integrate_pipeline: CachedComputePipelineId,
    /// 共有型定義シェーダー（#import 解決用にハンドルを保持）
    pub _physics_types_shader: Handle<Shader>,
}

impl GpuPhysicsPipelines {
    pub fn create(pipeline_cache: &PipelineCache, asset_server: &AssetServer) -> Self {
        // バインドグループレイアウト記述子（manual entries）
        let bind_group_layout_desc = BindGroupLayoutDescriptor::new(
            "gpu_physics_bind_group_layout",
            &[
                // 0: params (uniform)
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: particles_in (storage, read)
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: particles_out (storage, read_write)
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: keys (storage, read_write)
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: particle_ids (storage, read_write)
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: cell_ranges (storage, read_write)
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6: forces (storage, read_write)
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 7: torques (storage, read_write)
                BindGroupLayoutEntry {
                    binding: 7,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        );

        // 共有型定義シェーダーをロード（#import 解決に必要）
        let physics_types_shader: Handle<Shader> = asset_server.load("shaders/physics_types.wgsl");

        // シェーダーをロード
        let hash_keys_shader = asset_server.load("shaders/hash_keys.wgsl");
        let bitonic_sort_shader = asset_server.load("shaders/bitonic_sort.wgsl");
        let cell_ranges_shader = asset_server.load("shaders/cell_ranges.wgsl");
        let collision_shader = asset_server.load("shaders/collision.wgsl");
        let integrate_shader = asset_server.load("shaders/integrate.wgsl");

        let hash_keys_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("hash_keys_pipeline".into()),
            layout: vec![bind_group_layout_desc.clone()],
            shader: hash_keys_shader,
            shader_defs: vec![],
            entry_point: Some("build_keys".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: true,
        });

        let bitonic_sort_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("bitonic_sort_pipeline".into()),
                layout: vec![bind_group_layout_desc.clone()],
                shader: bitonic_sort_shader,
                shader_defs: vec![],
                entry_point: Some("bitonic_sort_step".into()),
                push_constant_ranges: vec![PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    range: 0..8,
                }],
                zero_initialize_workgroup_memory: true,
            });

        let cell_ranges_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("cell_ranges_pipeline".into()),
                layout: vec![bind_group_layout_desc.clone()],
                shader: cell_ranges_shader,
                shader_defs: vec![],
                entry_point: Some("build_cell_ranges".into()),
                push_constant_ranges: vec![],
                zero_initialize_workgroup_memory: true,
            });

        let collision_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("collision_pipeline".into()),
            layout: vec![bind_group_layout_desc.clone()],
            shader: collision_shader,
            shader_defs: vec![],
            entry_point: Some("collision_response".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: true,
        });

        let integrate_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("integrate_pipeline".into()),
            layout: vec![bind_group_layout_desc.clone()],
            shader: integrate_shader,
            shader_defs: vec![],
            entry_point: Some("integrate".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: true,
        });

        Self {
            bind_group_layout_desc,
            hash_keys_pipeline,
            bitonic_sort_pipeline,
            cell_ranges_pipeline,
            collision_pipeline,
            integrate_pipeline,
            _physics_types_shader: physics_types_shader,
        }
    }
}
