use crate::rendering::shaders::physics_shader_processor;
use std::num::NonZeroU64;

use super::buffers::SortParamsGpu;

/// コンピュートパイプラインのリソース（raw wgpu）
pub struct GpuPhysicsPipelines {
    /// 共通パラメータ（uniform）のバインドグループレイアウト
    pub params_bind_group_layout: wgpu::BindGroupLayout,
    /// 粒子入出力（in/out）のバインドグループレイアウト
    pub particles_bind_group_layout: wgpu::BindGroupLayout,
    /// 空間探索（keys / ids / cell_ranges）のバインドグループレイアウト
    pub spatial_bind_group_layout: wgpu::BindGroupLayout,
    /// Bitonic sort パラメータ（uniform）のバインドグループレイアウト
    pub sort_params_bind_group_layout: wgpu::BindGroupLayout,
    /// 接触力・トルク（forces / torques）のバインドグループレイアウト
    pub contact_bind_group_layout: wgpu::BindGroupLayout,
    /// ハッシュキー生成パイプライン
    pub hash_keys_pipeline: wgpu::ComputePipeline,
    /// Bitonic sort パイプライン
    pub bitonic_sort_pipeline: wgpu::ComputePipeline,
    /// セル範囲構築パイプライン
    pub cell_ranges_pipeline: wgpu::ComputePipeline,
    /// 衝突検出・力計算パイプライン
    pub collision_pipeline: wgpu::ComputePipeline,
    /// 積分パイプライン（Verlet 前半）
    pub integrate_first_half_pipeline: wgpu::ComputePipeline,
    /// 積分パイプライン（Verlet 後半）
    pub integrate_second_half_pipeline: wgpu::ComputePipeline,
}

impl GpuPhysicsPipelines {
    pub fn create(device: &wgpu::Device) -> Self {
        // --- Bind group layouts ---

        // Group 0: params (uniform)
        let params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_physics_params_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        // Group 1: particles (in/out)
        let particles_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_physics_particles_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Group 2: spatial search buffers (keys / ids / cell_ranges)
        let spatial_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_physics_spatial_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Group 3: contact force buffers (forces / torques)
        let contact_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_physics_contact_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Group 4: bitonic sort params (uniform, dynamic offset)
        let sort_params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_physics_sort_params_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: Some(
                            NonZeroU64::new(std::mem::size_of::<SortParamsGpu>() as u64)
                                .expect("SortParamsGpu size must be non-zero"),
                        ),
                    },
                    count: None,
                }],
            });

        // --- Shader modules ---

        let sp = physics_shader_processor();

        let hash_keys_source = sp
            .resolve(include_str!("../../../assets/shaders/hash_keys.wgsl"))
            .expect("failed to resolve hash_keys shader");
        let hash_keys_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hash_keys_shader"),
            source: wgpu::ShaderSource::Wgsl(hash_keys_source.into()),
        });

        let bitonic_sort_source = sp
            .resolve(include_str!("../../../assets/shaders/bitonic_sort.wgsl"))
            .expect("failed to resolve bitonic_sort shader");
        let bitonic_sort_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bitonic_sort_shader"),
            source: wgpu::ShaderSource::Wgsl(bitonic_sort_source.into()),
        });

        let cell_ranges_source = sp
            .resolve(include_str!("../../../assets/shaders/cell_ranges.wgsl"))
            .expect("failed to resolve cell_ranges shader");
        let cell_ranges_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cell_ranges_shader"),
            source: wgpu::ShaderSource::Wgsl(cell_ranges_source.into()),
        });

        let collision_source = sp
            .resolve(include_str!("../../../assets/shaders/collision.wgsl"))
            .expect("failed to resolve collision shader");
        let collision_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("collision_shader"),
            source: wgpu::ShaderSource::Wgsl(collision_source.into()),
        });

        let integrate_source = sp
            .resolve(include_str!("../../../assets/shaders/integrate.wgsl"))
            .expect("failed to resolve integrate shader");
        let integrate_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("integrate_shader"),
            source: wgpu::ShaderSource::Wgsl(integrate_source.into()),
        });

        // --- Compute pipelines ---

        // hash_keys: group0=params, group1=particles(read-only), group2=spatial
        let hash_keys_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hash_keys_pipeline_layout"),
                bind_group_layouts: &[
                    &params_bind_group_layout,
                    &particles_bind_group_layout,
                    &spatial_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let hash_keys_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hash_keys_pipeline"),
            layout: Some(&hash_keys_pipeline_layout),
            module: &hash_keys_module,
            entry_point: Some("build_keys"),
            compilation_options: Default::default(),
            cache: None,
        });

        // bitonic_sort: group0=spatial, group1=sort_params
        let bitonic_sort_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bitonic_sort_pipeline_layout"),
                bind_group_layouts: &[&spatial_bind_group_layout, &sort_params_bind_group_layout],
                push_constant_ranges: &[],
            });
        let bitonic_sort_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("bitonic_sort_pipeline"),
                layout: Some(&bitonic_sort_pipeline_layout),
                module: &bitonic_sort_module,
                entry_point: Some("bitonic_sort_step"),
                compilation_options: Default::default(),
                cache: None,
            });

        // cell_ranges: group0=params, group1=spatial
        let cell_ranges_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("cell_ranges_pipeline_layout"),
                bind_group_layouts: &[&params_bind_group_layout, &spatial_bind_group_layout],
                push_constant_ranges: &[],
            });
        let cell_ranges_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("cell_ranges_pipeline"),
                layout: Some(&cell_ranges_pipeline_layout),
                module: &cell_ranges_module,
                entry_point: Some("build_cell_ranges"),
                compilation_options: Default::default(),
                cache: None,
            });

        // collision: group0=params, group1=particles(read-only), group2=spatial, group3=contact
        let collision_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("collision_pipeline_layout"),
                bind_group_layouts: &[
                    &params_bind_group_layout,
                    &particles_bind_group_layout,
                    &spatial_bind_group_layout,
                    &contact_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let collision_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("collision_pipeline"),
            layout: Some(&collision_pipeline_layout),
            module: &collision_module,
            entry_point: Some("collision_response"),
            compilation_options: Default::default(),
            cache: None,
        });

        // integrate_first_half: group0=params, group1=particles(in/out), group2=contact
        let integrate_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("integrate_pipeline_layout"),
                bind_group_layouts: &[
                    &params_bind_group_layout,
                    &particles_bind_group_layout,
                    &contact_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let integrate_first_half_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("integrate_first_half_pipeline"),
                layout: Some(&integrate_pipeline_layout),
                module: &integrate_module,
                entry_point: Some("integrate_first_half"),
                compilation_options: Default::default(),
                cache: None,
            });

        // integrate_second_half uses same layout
        let integrate_second_half_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("integrate_second_half_pipeline_layout"),
                bind_group_layouts: &[
                    &params_bind_group_layout,
                    &particles_bind_group_layout,
                    &contact_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
        let integrate_second_half_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("integrate_second_half_pipeline"),
                layout: Some(&integrate_second_half_pipeline_layout),
                module: &integrate_module,
                entry_point: Some("integrate_second_half"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            params_bind_group_layout,
            particles_bind_group_layout,
            spatial_bind_group_layout,
            sort_params_bind_group_layout,
            contact_bind_group_layout,
            hash_keys_pipeline,
            bitonic_sort_pipeline,
            cell_ranges_pipeline,
            collision_pipeline,
            integrate_first_half_pipeline,
            integrate_second_half_pipeline,
        }
    }
}
