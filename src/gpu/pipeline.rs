use bevy::{prelude::*, render::render_resource::*};

/// コンピュートパイプラインのリソース
#[derive(Resource)]
pub struct GpuPhysicsPipelines {
    /// ハッシュキー生成パスのバインドグループレイアウト
    pub hash_bind_group_layout_desc: BindGroupLayoutDescriptor,
    /// Bitonic ソートパスのバインドグループレイアウト
    pub bitonic_bind_group_layout_desc: BindGroupLayoutDescriptor,
    /// セル範囲構築パスのバインドグループレイアウト
    pub cell_ranges_bind_group_layout_desc: BindGroupLayoutDescriptor,
    /// 衝突検出・接触力計算パスのバインドグループレイアウト
    pub collision_bind_group_layout_desc: BindGroupLayoutDescriptor,
    /// 積分パスのバインドグループレイアウト
    pub integrate_bind_group_layout_desc: BindGroupLayoutDescriptor,
    /// ハッシュキー生成パイプライン
    pub hash_keys_pipeline: CachedComputePipelineId,
    /// Bitonic sort パイプライン（複数ステップ）
    pub bitonic_sort_pipeline: CachedComputePipelineId,
    /// セル範囲構築パイプライン
    pub cell_ranges_pipeline: CachedComputePipelineId,
    /// 衝突検出・力計算パイプライン
    pub collision_pipeline: CachedComputePipelineId,
    /// 積分パイプライン（Verlet 前半）
    pub integrate_first_half_pipeline: CachedComputePipelineId,
    /// 積分パイプライン（Verlet 後半）
    pub integrate_second_half_pipeline: CachedComputePipelineId,
    /// 共有型定義シェーダー（#import 解決用にハンドルを保持）
    pub _physics_types_shader: Handle<Shader>,
}

impl GpuPhysicsPipelines {
    pub fn create(pipeline_cache: &PipelineCache, asset_server: &AssetServer) -> Self {
        // Pass 1: Hash keys
        let hash_bind_group_layout_desc = BindGroupLayoutDescriptor::new(
            "gpu_physics_hash_bind_group_layout",
            &[
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
            ],
        );

        // Pass 2: Bitonic sort
        let bitonic_bind_group_layout_desc = BindGroupLayoutDescriptor::new(
            "gpu_physics_bitonic_bind_group_layout",
            &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
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

        // Pass 3: Cell ranges
        let cell_ranges_bind_group_layout_desc = BindGroupLayoutDescriptor::new(
            "gpu_physics_cell_ranges_bind_group_layout",
            &[
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
            ],
        );

        // Pass 4: Collision
        let collision_bind_group_layout_desc = BindGroupLayoutDescriptor::new(
            "gpu_physics_collision_bind_group_layout",
            &[
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
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
            ],
        );

        // Pass 5: Integrate
        let integrate_bind_group_layout_desc = BindGroupLayoutDescriptor::new(
            "gpu_physics_integrate_bind_group_layout",
            &[
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
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
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
            layout: vec![hash_bind_group_layout_desc.clone()],
            shader: hash_keys_shader,
            shader_defs: vec![],
            entry_point: Some("build_keys".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: true,
        });

        let bitonic_sort_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("bitonic_sort_pipeline".into()),
                layout: vec![bitonic_bind_group_layout_desc.clone()],
                shader: bitonic_sort_shader,
                shader_defs: vec![],
                entry_point: Some("bitonic_sort_step".into()),
                push_constant_ranges: vec![PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    range: 0..12,
                }],
                zero_initialize_workgroup_memory: true,
            });

        let cell_ranges_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("cell_ranges_pipeline".into()),
                layout: vec![cell_ranges_bind_group_layout_desc.clone()],
                shader: cell_ranges_shader,
                shader_defs: vec![],
                entry_point: Some("build_cell_ranges".into()),
                push_constant_ranges: vec![],
                zero_initialize_workgroup_memory: true,
            });

        let collision_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("collision_pipeline".into()),
            layout: vec![collision_bind_group_layout_desc.clone()],
            shader: collision_shader,
            shader_defs: vec![],
            entry_point: Some("collision_response".into()),
            push_constant_ranges: vec![],
            zero_initialize_workgroup_memory: true,
        });

        let integrate_first_half_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("integrate_first_half_pipeline".into()),
                layout: vec![integrate_bind_group_layout_desc.clone()],
                shader: integrate_shader.clone(),
                shader_defs: vec![],
                entry_point: Some("integrate_first_half".into()),
                push_constant_ranges: vec![],
                zero_initialize_workgroup_memory: true,
            });

        let integrate_second_half_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("integrate_second_half_pipeline".into()),
                layout: vec![integrate_bind_group_layout_desc.clone()],
                shader: integrate_shader,
                shader_defs: vec![],
                entry_point: Some("integrate_second_half".into()),
                push_constant_ranges: vec![],
                zero_initialize_workgroup_memory: true,
            });

        Self {
            hash_bind_group_layout_desc,
            bitonic_bind_group_layout_desc,
            cell_ranges_bind_group_layout_desc,
            collision_bind_group_layout_desc,
            integrate_bind_group_layout_desc,
            hash_keys_pipeline,
            bitonic_sort_pipeline,
            cell_ranges_pipeline,
            collision_pipeline,
            integrate_first_half_pipeline,
            integrate_second_half_pipeline,
            _physics_types_shader: physics_types_shader,
        }
    }
}
