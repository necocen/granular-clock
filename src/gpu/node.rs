use bevy::{
    prelude::*,
    render::{
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
    },
};

use super::{
    buffers::GpuPhysicsBuffers,
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
/// パスごとに最小 binding のバインドグループを保持する。
struct GpuPhysicsBindGroups {
    hash: [BindGroup; 2],
    bitonic: BindGroup,
    cell_ranges: BindGroup,
    collision: [BindGroup; 2],
    integrate: [BindGroup; 2],
}

#[derive(Default)]
pub struct GpuPhysicsNode {
    /// パス別バインドグループ一式
    bind_groups: Option<GpuPhysicsBindGroups>,
    /// 1フレームあたりのサブステップ数
    substeps: u32,
}

impl GpuPhysicsNode {
    pub fn new() -> Self {
        Self::default()
    }

    fn container_offset_for_substep(
        container_params: &ExtractedContainerParams,
        substeps: u32,
        substep_index: u32,
        dt: f32,
    ) -> f32 {
        if !container_params.oscillation_enabled {
            return container_params.base_position_y;
        }

        let steps = substeps.max(1) as f64;
        let dt64 = dt as f64;
        let start_time = container_params.sim_elapsed - steps * dt64;
        let t = start_time + (substep_index as f64 + 1.0) * dt64;
        let phase = 2.0 * std::f64::consts::PI * container_params.oscillation_frequency as f64 * t;

        container_params.base_position_y
            + container_params.oscillation_amplitude * phase.sin() as f32
    }
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
        let render_device = world.resource::<RenderDevice>();
        let pipeline_cache = world.resource::<PipelineCache>();

        // パイプラインキャッシュからパス別バインドグループレイアウトを取得
        let hash_layout =
            pipeline_cache.get_bind_group_layout(&pipelines.hash_bind_group_layout_desc);
        let bitonic_layout =
            pipeline_cache.get_bind_group_layout(&pipelines.bitonic_bind_group_layout_desc);
        let cell_ranges_layout =
            pipeline_cache.get_bind_group_layout(&pipelines.cell_ranges_bind_group_layout_desc);
        let collision_layout =
            pipeline_cache.get_bind_group_layout(&pipelines.collision_bind_group_layout_desc);
        let integrate_layout =
            pipeline_cache.get_bind_group_layout(&pipelines.integrate_bind_group_layout_desc);

        // Hash (A -> keys / B -> keys)
        let hash_forward = render_device.create_bind_group(
            Some("gpu_physics_hash_bind_group_forward"),
            &hash_layout,
            &BindGroupEntries::sequential((
                buffers.params.as_entire_binding(),
                buffers.forward_in().as_entire_binding(),
                buffers.keys.as_entire_binding(),
                buffers.particle_ids.as_entire_binding(),
            )),
        );
        let hash_reverse = render_device.create_bind_group(
            Some("gpu_physics_hash_bind_group_reverse"),
            &hash_layout,
            &BindGroupEntries::sequential((
                buffers.params.as_entire_binding(),
                buffers.reverse_in().as_entire_binding(),
                buffers.keys.as_entire_binding(),
                buffers.particle_ids.as_entire_binding(),
            )),
        );

        // Bitonic (keys / ids)
        let bitonic = render_device.create_bind_group(
            Some("gpu_physics_bitonic_bind_group"),
            &bitonic_layout,
            &BindGroupEntries::sequential((
                buffers.keys.as_entire_binding(),
                buffers.particle_ids.as_entire_binding(),
            )),
        );

        // Cell ranges (params / keys / ranges)
        let cell_ranges = render_device.create_bind_group(
            Some("gpu_physics_cell_ranges_bind_group"),
            &cell_ranges_layout,
            &BindGroupEntries::sequential((
                buffers.params.as_entire_binding(),
                buffers.keys.as_entire_binding(),
                buffers.cell_ranges.as_entire_binding(),
            )),
        );

        // Collision (in / keys / ids / ranges / forces / torques)
        let collision_forward = render_device.create_bind_group(
            Some("gpu_physics_collision_bind_group_forward"),
            &collision_layout,
            &BindGroupEntries::sequential((
                buffers.params.as_entire_binding(),
                buffers.forward_in().as_entire_binding(),
                buffers.keys.as_entire_binding(),
                buffers.particle_ids.as_entire_binding(),
                buffers.cell_ranges.as_entire_binding(),
                buffers.forces.as_entire_binding(),
                buffers.torques.as_entire_binding(),
            )),
        );
        let collision_reverse = render_device.create_bind_group(
            Some("gpu_physics_collision_bind_group_reverse"),
            &collision_layout,
            &BindGroupEntries::sequential((
                buffers.params.as_entire_binding(),
                buffers.reverse_in().as_entire_binding(),
                buffers.keys.as_entire_binding(),
                buffers.particle_ids.as_entire_binding(),
                buffers.cell_ranges.as_entire_binding(),
                buffers.forces.as_entire_binding(),
                buffers.torques.as_entire_binding(),
            )),
        );

        // Integrate (in / out / forces / torques)
        let integrate_forward = render_device.create_bind_group(
            Some("gpu_physics_integrate_bind_group_forward"),
            &integrate_layout,
            &BindGroupEntries::sequential((
                buffers.params.as_entire_binding(),
                buffers.forward_in().as_entire_binding(),
                buffers.forward_out().as_entire_binding(),
                buffers.forces.as_entire_binding(),
                buffers.torques.as_entire_binding(),
            )),
        );
        let integrate_reverse = render_device.create_bind_group(
            Some("gpu_physics_integrate_bind_group_reverse"),
            &integrate_layout,
            &BindGroupEntries::sequential((
                buffers.params.as_entire_binding(),
                buffers.reverse_in().as_entire_binding(),
                buffers.reverse_out().as_entire_binding(),
                buffers.forces.as_entire_binding(),
                buffers.torques.as_entire_binding(),
            )),
        );

        self.bind_groups = Some(GpuPhysicsBindGroups {
            hash: [hash_forward, hash_reverse],
            bitonic,
            cell_ranges,
            collision: [collision_forward, collision_reverse],
            integrate: [integrate_forward, integrate_reverse],
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
        let render_queue = world.resource::<RenderQueue>();
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

        let run_neighbor_search =
            |encoder: &mut CommandEncoder,
             hash_bind_group: &BindGroup,
             bitonic_bind_group: &BindGroup,
             cell_ranges_bind_group: &BindGroup| {
                // CPU 実装の clear_forces/build_spatial_grid に合わせて
                // half-step ごとにグリッド/力バッファを再計算する。
                encoder.clear_buffer(&buffers.cell_ranges, 0, None);
                encoder.clear_buffer(&buffers.forces, 0, None);
                encoder.clear_buffer(&buffers.torques, 0, None);

                // Pass 1: Hash Keys
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("hash_keys_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(hash_keys_pipeline);
                    pass.set_bind_group(0, hash_bind_group, &[]);
                    pass.dispatch_workgroups(workgroups_sort_64, 1, 1);
                }

                // Pass 2: Bitonic Sort
                {
                    let mut k = 2u32;
                    while k <= sort_count {
                        let mut j = k / 2;
                        while j > 0 {
                            let push_constants = [j, k, sort_count];
                            let push_constant_bytes = bytemuck::bytes_of(&push_constants);

                            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                                label: Some("bitonic_sort_pass"),
                                timestamp_writes: None,
                            });
                            pass.set_pipeline(bitonic_sort_pipeline);
                            pass.set_bind_group(0, bitonic_bind_group, &[]);
                            pass.set_push_constants(0, push_constant_bytes);
                            pass.dispatch_workgroups(workgroups_sort_256, 1, 1);
                            j /= 2;
                        }
                        k *= 2;
                    }
                }

                // Pass 3: Cell Ranges
                {
                    let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                        label: Some("cell_ranges_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(cell_ranges_pipeline);
                    pass.set_bind_group(0, cell_ranges_bind_group, &[]);
                    pass.dispatch_workgroups(workgroups_particles_64, 1, 1);
                }
            };

        let run_collision = |encoder: &mut CommandEncoder, bind_group: &BindGroup| {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("collision_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(collision_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups_particles_64, 1, 1);
        };

        let run_integrate = |encoder: &mut CommandEncoder,
                             bind_group: &BindGroup,
                             integrate_pipeline: &ComputePipeline| {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("integrate_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(integrate_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups_particles_64, 1, 1);
        };

        let dispatch_half_step =
            |encoder: &mut CommandEncoder,
             hash_bind_group: &BindGroup,
             collision_bind_group: &BindGroup,
             integrate_bind_group: &BindGroup,
             integrate_pipeline: &ComputePipeline| {
                // CPU と同じ順序:
                // 近傍探索 -> 衝突/接触力 -> 積分
                run_neighbor_search(
                    encoder,
                    hash_bind_group,
                    &bind_groups.bitonic,
                    &bind_groups.cell_ranges,
                );
                run_collision(encoder, collision_bind_group);
                run_integrate(encoder, integrate_bind_group, integrate_pipeline);
            };

        // 1サブステップ:
        // - 前半: hash/sort/cell + collision + integrate_first_half
        // - 後半: hash/sort/cell + collision + integrate_second_half
        for substep_index in 0..self.substeps {
            // plugin::update_params_only がフレーム基準の共通 params を更新し、
            // ここではサブステップ単位の container_offset のみ上書きする。
            if let (Some(gpu_data), Some(container_params)) = (gpu_data, container_params) {
                let mut runtime_params = gpu_data.params;
                runtime_params.container_offset = Self::container_offset_for_substep(
                    container_params,
                    self.substeps,
                    substep_index,
                    runtime_params.dt,
                );
                render_queue.write_buffer(&buffers.params, 0, bytemuck::bytes_of(&runtime_params));
            }

            // 前半（A -> B）
            dispatch_half_step(
                encoder,
                &bind_groups.hash[0],
                &bind_groups.collision[0],
                &bind_groups.integrate[0],
                integrate_first_half_pipeline,
            );

            // 後半（B -> A）。サブステップ完了時の最新状態は常に A 側。
            dispatch_half_step(
                encoder,
                &bind_groups.hash[1],
                &bind_groups.collision[1],
                &bind_groups.integrate[1],
                integrate_second_half_pipeline,
            );
        }

        Ok(())
    }
}
