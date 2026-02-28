use crate::simulation::constants::{advance_oscillation_phase, oscillation_displacement};
use bevy::{
    prelude::*,
    render::{
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
    },
};
use std::num::NonZeroU64;

use super::{
    buffers::{GpuPhysicsBuffers, SimulationParams, SortParamsGpu},
    pipeline::GpuPhysicsPipelines,
    plugin::{ExtractedContainerParams, GpuParticleData},
};

/// 物理計算ノードのラベル
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct GpuPhysicsLabel;

/// GPU 物理計算を実行するレンダーグラフノード
///
/// 1フレームあたり複数サブステップを実行する。
/// バッファスロット A/B を固定で使い分ける。
/// bind group は用途単位（params / particles / spatial / contact）で再利用する。
struct GpuPhysicsBindGroups {
    params: BindGroup,
    particles: [BindGroup; 2],
    spatial: BindGroup,
    sort_params: BindGroup,
    contact: BindGroup,
}

#[derive(Default)]
pub struct GpuPhysicsNode {
    /// 用途別バインドグループ一式
    bind_groups: Option<GpuPhysicsBindGroups>,
    /// 1フレームあたりのサブステップ数
    substeps: u32,
    /// 現在キャッシュしている sort_count
    cached_sort_count: u32,
    /// 現在キャッシュしている sort params の stride
    cached_sort_stride: u32,
    /// bitonic sort のステージ数
    sort_stage_count: usize,
    /// 動的オフセット用にパック済みの sort params
    sort_params_packed: Vec<u8>,
    /// sort params の再アップロードが必要か
    sort_params_dirty: bool,
}

impl GpuPhysicsNode {
    pub fn new() -> Self {
        Self::default()
    }
}

fn build_bitonic_sort_params_payload(sort_count: u32, stride: u32) -> (Vec<u8>, usize) {
    let mut stages: Vec<SortParamsGpu> = Vec::new();
    let mut k = 2u32;
    while k <= sort_count {
        let mut j = k / 2;
        while j > 0 {
            stages.push(SortParamsGpu {
                j,
                k,
                n: sort_count,
                _pad: 0,
            });
            j /= 2;
        }
        k *= 2;
    }

    let stage_count = stages.len();
    if stage_count == 0 {
        return (Vec::new(), 0);
    }

    let stride = stride as usize;
    let mut packed = vec![0u8; stage_count * stride];
    for (idx, params) in stages.iter().enumerate() {
        let start = idx * stride;
        let end = start + std::mem::size_of::<SortParamsGpu>();
        packed[start..end].copy_from_slice(bytemuck::bytes_of(params));
    }

    (packed, stage_count)
}

impl Node for GpuPhysicsNode {
    fn update(&mut self, world: &mut World) {
        // GpuParticleData からサブステップ数と一時停止状態を取得
        if let Some(gpu_data) = world.get_resource::<GpuParticleData>() {
            self.substeps = gpu_data.substeps;
            if gpu_data.paused {
                self.bind_groups = None;
                return;
            }
        }

        // パイプラインとバッファが準備できているか確認（immutable access）
        let Some(buffers) = world.get_resource::<GpuPhysicsBuffers>() else {
            return;
        };
        let Some(pipelines) = world.get_resource::<GpuPhysicsPipelines>() else {
            return;
        };

        let buffers_changed = world.is_resource_changed::<GpuPhysicsBuffers>();
        let pipelines_changed = world.is_resource_changed::<GpuPhysicsPipelines>();

        // sort params は粒子数（= sort_count）か stride が変わったときに再生成。
        let sort_count = buffers.num_particles.next_power_of_two();
        if sort_count != self.cached_sort_count
            || buffers.sort_params_stride != self.cached_sort_stride
        {
            self.cached_sort_count = sort_count;
            self.cached_sort_stride = buffers.sort_params_stride;
            let (packed, stage_count) =
                build_bitonic_sort_params_payload(sort_count, buffers.sort_params_stride);
            self.sort_params_packed = packed;
            self.sort_stage_count = stage_count;
            self.sort_params_dirty = true;
        }

        if buffers_changed {
            self.sort_params_dirty = true;
        }

        if self.sort_stage_count as u32 <= buffers.sort_params_capacity_passes {
            if self.sort_params_dirty && !self.sort_params_packed.is_empty() {
                let render_queue = world.resource::<RenderQueue>();
                render_queue.write_buffer(&buffers.sort_params, 0, &self.sort_params_packed);
            }
        } else {
            // 容量不足時は run 側でもスキップする。
            self.sort_stage_count = 0;
            self.sort_params_packed.clear();
        }
        self.sort_params_dirty = false;

        let needs_rebuild = self.bind_groups.is_none() || buffers_changed || pipelines_changed;
        if !needs_rebuild {
            return;
        }

        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // パイプラインキャッシュから用途別バインドグループレイアウトを取得
        let params_layout =
            pipeline_cache.get_bind_group_layout(&pipelines.params_bind_group_layout_desc);
        let particles_layout =
            pipeline_cache.get_bind_group_layout(&pipelines.particles_bind_group_layout_desc);
        let spatial_layout =
            pipeline_cache.get_bind_group_layout(&pipelines.spatial_bind_group_layout_desc);
        let sort_params_layout =
            pipeline_cache.get_bind_group_layout(&pipelines.sort_params_bind_group_layout_desc);
        let contact_layout =
            pipeline_cache.get_bind_group_layout(&pipelines.contact_bind_group_layout_desc);

        // Params
        let params = render_device.create_bind_group(
            Some("gpu_physics_params_bind_group"),
            &params_layout,
            &BindGroupEntries::sequential((buffers.params.as_entire_binding(),)),
        );

        // Particles (Forward/Reverse)
        let particles_forward = render_device.create_bind_group(
            Some("gpu_physics_particles_bind_group_forward"),
            &particles_layout,
            &BindGroupEntries::sequential((
                buffers.forward_in().as_entire_binding(),
                buffers.forward_out().as_entire_binding(),
            )),
        );
        let particles_reverse = render_device.create_bind_group(
            Some("gpu_physics_particles_bind_group_reverse"),
            &particles_layout,
            &BindGroupEntries::sequential((
                buffers.reverse_in().as_entire_binding(),
                buffers.reverse_out().as_entire_binding(),
            )),
        );

        // Spatial (keys / ids / cell_ranges)
        let spatial = render_device.create_bind_group(
            Some("gpu_physics_spatial_bind_group"),
            &spatial_layout,
            &BindGroupEntries::sequential((
                buffers.keys.as_entire_binding(),
                buffers.particle_ids.as_entire_binding(),
                buffers.cell_ranges.as_entire_binding(),
            )),
        );

        let sort_params = render_device.create_bind_group(
            Some("gpu_physics_sort_params_bind_group"),
            &sort_params_layout,
            &BindGroupEntries::single(BufferBinding {
                buffer: &buffers.sort_params,
                offset: 0,
                size: Some(
                    NonZeroU64::new(std::mem::size_of::<SortParamsGpu>() as u64)
                        .expect("SortParamsGpu size must be non-zero"),
                ),
            }),
        );

        // Contact (forces / torques)
        let contact = render_device.create_bind_group(
            Some("gpu_physics_contact_bind_group"),
            &contact_layout,
            &BindGroupEntries::sequential((
                buffers.forces.as_entire_binding(),
                buffers.torques.as_entire_binding(),
            )),
        );

        self.bind_groups = Some(GpuPhysicsBindGroups {
            params,
            particles: [particles_forward, particles_reverse],
            spatial,
            sort_params,
            contact,
        });
    }

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(bind_groups) = &self.bind_groups else {
            return Ok(());
        };
        let Some(buffers) = world.get_resource::<GpuPhysicsBuffers>() else {
            return Ok(());
        };
        let Some(pipelines) = world.get_resource::<GpuPhysicsPipelines>() else {
            return Ok(());
        };
        let pipeline_cache = world.resource::<PipelineCache>();
        let render_device = world.resource::<RenderDevice>();
        let gpu_data = world.get_resource::<GpuParticleData>();
        let container_params = world.get_resource::<ExtractedContainerParams>();

        // パイプラインが準備できているか確認
        let Some(hash_keys_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.hash_keys_pipeline)
        else {
            return Ok(());
        };
        let Some(bitonic_sort_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.bitonic_sort_pipeline)
        else {
            return Ok(());
        };
        let Some(cell_ranges_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.cell_ranges_pipeline)
        else {
            return Ok(());
        };
        let Some(collision_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.collision_pipeline)
        else {
            return Ok(());
        };
        let Some(integrate_first_half_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.integrate_first_half_pipeline)
        else {
            return Ok(());
        };
        let Some(integrate_second_half_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.integrate_second_half_pipeline)
        else {
            return Ok(());
        };

        let num_particles = buffers.num_particles;
        if num_particles == 0 {
            return Ok(());
        }

        let workgroups_particles_64 = num_particles.div_ceil(64);
        let sort_count = num_particles.next_power_of_two();
        if sort_count > buffers.capacity {
            return Ok(());
        }
        let workgroups_sort_64 = sort_count.div_ceil(64);
        let workgroups_sort_256 = sort_count.div_ceil(256);

        let encoder = render_context.command_encoder();
        // `copy_buffer_to_buffer` に使う upload buffer の寿命をこの関数末尾まで保持する。
        let mut param_upload_buffers: Vec<Buffer> = Vec::with_capacity(self.substeps as usize);

        let clear_neighbor_and_contact_buffers = |encoder: &mut CommandEncoder| {
            // CPU 実装の clear_forces/build_spatial_grid に合わせて
            // half-step ごとにグリッド/力バッファをクリアする。
            encoder.clear_buffer(&buffers.cell_ranges, 0, None);
            encoder.clear_buffer(&buffers.forces, 0, None);
            encoder.clear_buffer(&buffers.torques, 0, None);
        };

        let run_neighbor_search =
            |encoder: &mut CommandEncoder, particles_bind_group: &BindGroup| {
                // Pass 1: Hash Keys
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("hash_keys_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(hash_keys_pipeline);
                    pass.set_bind_group(0, &bind_groups.params, &[]);
                    pass.set_bind_group(1, particles_bind_group, &[]);
                    pass.set_bind_group(2, &bind_groups.spatial, &[]);
                    pass.dispatch_workgroups(workgroups_sort_64, 1, 1);
                }

                // Pass 2: Bitonic Sort
                {
                    if self.sort_stage_count == 0 {
                        // 1粒子など、ソート不要ケース
                    } else if self.sort_stage_count as u32 > buffers.sort_params_capacity_passes {
                        return;
                    } else {
                        // 1pass 内で全ステージを dispatch して、Pass生成オーバーヘッドを削減する。
                        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                            label: Some("bitonic_sort_pass"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(bitonic_sort_pipeline);
                        pass.set_bind_group(0, &bind_groups.spatial, &[]);
                        for stage_idx in 0..self.sort_stage_count {
                            let dynamic_offset = stage_idx as u32 * buffers.sort_params_stride;
                            pass.set_bind_group(1, &bind_groups.sort_params, &[dynamic_offset]);
                            pass.dispatch_workgroups(workgroups_sort_256, 1, 1);
                        }
                    }
                }

                // Pass 3: Cell Ranges
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("cell_ranges_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(cell_ranges_pipeline);
                    pass.set_bind_group(0, &bind_groups.params, &[]);
                    pass.set_bind_group(1, &bind_groups.spatial, &[]);
                    pass.dispatch_workgroups(workgroups_particles_64, 1, 1);
                }
            };

        let run_collision = |encoder: &mut CommandEncoder, particles_bind_group: &BindGroup| {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("collision_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(collision_pipeline);
            pass.set_bind_group(0, &bind_groups.params, &[]);
            pass.set_bind_group(1, particles_bind_group, &[]);
            pass.set_bind_group(2, &bind_groups.spatial, &[]);
            pass.set_bind_group(3, &bind_groups.contact, &[]);
            pass.dispatch_workgroups(workgroups_particles_64, 1, 1);
        };

        let run_integrate = |encoder: &mut CommandEncoder,
                             particles_bind_group: &BindGroup,
                             integrate_pipeline: &ComputePipeline| {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("integrate_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(integrate_pipeline);
            pass.set_bind_group(0, &bind_groups.params, &[]);
            pass.set_bind_group(1, particles_bind_group, &[]);
            pass.set_bind_group(2, &bind_groups.contact, &[]);
            pass.dispatch_workgroups(workgroups_particles_64, 1, 1);
        };

        let dispatch_half_step =
            |encoder: &mut CommandEncoder,
             particles_bind_group: &BindGroup,
             integrate_pipeline: &ComputePipeline| {
                // CPU と同じ順序:
                // 近傍探索 -> 衝突/接触力 -> 積分
                clear_neighbor_and_contact_buffers(encoder);
                run_neighbor_search(encoder, particles_bind_group);
                run_collision(encoder, particles_bind_group);
                run_integrate(encoder, particles_bind_group, integrate_pipeline);
            };

        let mut substep_phase = container_params.map(|params| params.oscillation_phase_start);

        // 1サブステップ:
        // - 前半: hash/sort/cell + collision + integrate_first_half
        // - 後半: hash/sort/cell + collision + integrate_second_half
        for _ in 0..self.substeps {
            // plugin::update_params_only がフレーム基準の共通 params を更新し、
            // ここではサブステップ単位の container_offset のみ上書きする。
            if let (Some(gpu_data), Some(container_params)) = (gpu_data, container_params) {
                let mut runtime_params = gpu_data.params;
                // 振動無効時でも現在の当たり判定位置を維持するため、
                // まずフレーム基準オフセットを反映する。
                runtime_params.container_offset = container_params.container_offset;
                let mut phase = substep_phase.unwrap_or(container_params.oscillation_phase_start);
                if container_params.oscillation_enabled {
                    advance_oscillation_phase(
                        &mut phase,
                        container_params.oscillation_frequency,
                        runtime_params.dt,
                    );
                    runtime_params.container_offset = container_params.base_position_y
                        + oscillation_displacement(container_params.oscillation_amplitude, phase);
                }
                substep_phase = Some(phase);
                // queue.write_buffer を使うと同一エンコード内での更新境界が曖昧になるため、
                // サブステップごとに明示的なコピーコマンドを記録する。
                let upload = render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("gpu_physics_substep_params_upload"),
                    contents: bytemuck::bytes_of(&runtime_params),
                    usage: BufferUsages::COPY_SRC,
                });
                encoder.copy_buffer_to_buffer(
                    &upload,
                    0,
                    &buffers.params,
                    0,
                    std::mem::size_of::<SimulationParams>() as u64,
                );
                param_upload_buffers.push(upload);
            }

            // 前半（A -> B）
            dispatch_half_step(
                encoder,
                &bind_groups.particles[0],
                integrate_first_half_pipeline,
            );

            // 後半（B -> A）。サブステップ完了時の最新状態は常に A 側。
            dispatch_half_step(
                encoder,
                &bind_groups.particles[1],
                integrate_second_half_pipeline,
            );
        }

        Ok(())
    }
}
