use crate::rendering::shaders::physics_shader_processor;
use mikage::{create_compute_pipeline, storage_buffer_entry, uniform_buffer_entry, wgpu};
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

        let cs = wgpu::ShaderStages::COMPUTE;

        // Group 0: params (uniform)
        let params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_physics_params_bind_group_layout"),
                entries: &[uniform_buffer_entry(0, cs)],
            });

        // Group 1: particles (in/out)
        let particles_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_physics_particles_bind_group_layout"),
                entries: &[
                    storage_buffer_entry(0, cs, true),
                    storage_buffer_entry(1, cs, false),
                ],
            });

        // Group 2: spatial search buffers (keys / ids / cell_ranges)
        let spatial_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_physics_spatial_bind_group_layout"),
                entries: &[
                    storage_buffer_entry(0, cs, false),
                    storage_buffer_entry(1, cs, false),
                    storage_buffer_entry(2, cs, false),
                ],
            });

        // Group 3: contact force buffers (forces / torques)
        let contact_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_physics_contact_bind_group_layout"),
                entries: &[
                    storage_buffer_entry(0, cs, false),
                    storage_buffer_entry(1, cs, false),
                ],
            });

        // Group 4: bitonic sort params (uniform, dynamic offset) — raw wgpu
        // (needs has_dynamic_offset + min_binding_size)
        let sort_params_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("gpu_physics_sort_params_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: cs,
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

        // --- Compute pipelines ---

        let sp = physics_shader_processor();
        let resolve = |src| sp.resolve(src).expect("failed to resolve shader");

        let params_particles_spatial = &[
            &params_bind_group_layout,
            &particles_bind_group_layout,
            &spatial_bind_group_layout,
        ];
        let params_particles_contact = &[
            &params_bind_group_layout,
            &particles_bind_group_layout,
            &contact_bind_group_layout,
        ];

        let hash_keys_pipeline = create_compute_pipeline(
            device,
            "hash_keys",
            &resolve(include_str!("../../../assets/shaders/hash_keys.wgsl")),
            params_particles_spatial,
            "build_keys",
        );

        let bitonic_sort_pipeline = create_compute_pipeline(
            device,
            "bitonic_sort",
            &resolve(include_str!("../../../assets/shaders/bitonic_sort.wgsl")),
            &[&spatial_bind_group_layout, &sort_params_bind_group_layout],
            "bitonic_sort_step",
        );

        let cell_ranges_pipeline = create_compute_pipeline(
            device,
            "cell_ranges",
            &resolve(include_str!("../../../assets/shaders/cell_ranges.wgsl")),
            &[&params_bind_group_layout, &spatial_bind_group_layout],
            "build_cell_ranges",
        );

        let collision_pipeline = create_compute_pipeline(
            device,
            "collision",
            &resolve(include_str!("../../../assets/shaders/collision.wgsl")),
            &[
                &params_bind_group_layout,
                &particles_bind_group_layout,
                &spatial_bind_group_layout,
                &contact_bind_group_layout,
            ],
            "collision_response",
        );

        let integrate_source = resolve(include_str!("../../../assets/shaders/integrate.wgsl"));

        let integrate_first_half_pipeline = create_compute_pipeline(
            device,
            "integrate_first_half",
            &integrate_source,
            params_particles_contact,
            "integrate_first_half",
        );

        let integrate_second_half_pipeline = create_compute_pipeline(
            device,
            "integrate_second_half",
            &integrate_source,
            params_particles_contact,
            "integrate_second_half",
        );

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
