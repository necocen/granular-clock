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
/// ping-pong バッファの入出力方向を交互に切り替えるため、
/// 2つのバインドグループ（forward/reverse）を保持する。
#[derive(Default)]
pub struct GpuPhysicsNode {
    /// バインドグループ: [forward, reverse]
    /// forward: particles_in → particles_out
    /// reverse: particles_out → particles_in
    bind_groups: Option<[BindGroup; 2]>,
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

        container_params.base_position_y + container_params.oscillation_amplitude * phase.sin() as f32
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

        // まずバッファを swap（mutable access が必要）
        {
            let Some(mut buffers) = world.get_resource_mut::<GpuPhysicsBuffers>() else {
                return;
            };
            // 前フレームの結果を今フレームの入力として使う
            buffers.swap();
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

        // パイプラインキャッシュからバインドグループレイアウトを取得
        let bind_group_layout =
            pipeline_cache.get_bind_group_layout(&pipelines.bind_group_layout_desc);

        // Forward バインドグループ: current_input → current_output
        let forward = render_device.create_bind_group(
            Some("gpu_physics_bind_group_forward"),
            &bind_group_layout,
            &BindGroupEntries::sequential((
                buffers.params.as_entire_binding(),
                buffers.current_input().as_entire_binding(),
                buffers.current_output().as_entire_binding(),
                buffers.keys.as_entire_binding(),
                buffers.particle_ids.as_entire_binding(),
                buffers.cell_ranges.as_entire_binding(),
                buffers.forces.as_entire_binding(),
                buffers.torques.as_entire_binding(),
            )),
        );

        // Reverse バインドグループ: current_output → current_input
        let reverse = render_device.create_bind_group(
            Some("gpu_physics_bind_group_reverse"),
            &bind_group_layout,
            &BindGroupEntries::sequential((
                buffers.params.as_entire_binding(),
                buffers.current_output().as_entire_binding(),
                buffers.current_input().as_entire_binding(),
                buffers.keys.as_entire_binding(),
                buffers.particle_ids.as_entire_binding(),
                buffers.cell_ranges.as_entire_binding(),
                buffers.forces.as_entire_binding(),
                buffers.torques.as_entire_binding(),
            )),
        );

        self.bind_groups = Some([forward, reverse]);
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

        let run_neighbor_search = |encoder: &mut CommandEncoder, bind_group: &BindGroup| {
            // 前半開始時にグリッド/力バッファをクリア
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
                pass.set_bind_group(0, bind_group, &[]);
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
                        pass.set_bind_group(0, bind_group, &[]);
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
                pass.set_bind_group(0, bind_group, &[]);
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

        // 1サブステップ:
        // - 前半: hash/sort/cell + collision + integrate_first_half
        // - 後半: hash/sort/cell + collision + integrate_second_half
        for substep_index in 0..self.substeps {
            // CPU 実装に合わせてサブステップごとに振動オフセットを更新する。
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

            // 前半（in -> out）
            run_neighbor_search(encoder, &bind_groups[0]);
            run_collision(encoder, &bind_groups[0]);
            run_integrate(encoder, &bind_groups[0], integrate_first_half_pipeline);

            // 後半（out -> in）
            run_neighbor_search(encoder, &bind_groups[1]);
            run_collision(encoder, &bind_groups[1]);
            run_integrate(encoder, &bind_groups[1], integrate_second_half_pipeline);
        }

        // 2 half-step 実行後、最終結果は current_input() 側にある。
        // copy_to_staging / instancing は current_output() を読むため、常にコピーする。
        if self.substeps > 0 {
            let copy_size =
                std::mem::size_of::<super::buffers::ParticleGpu>() as u64 * num_particles as u64;
            encoder.copy_buffer_to_buffer(
                buffers.current_input(),
                0,
                buffers.current_output(),
                0,
                copy_size,
            );
        }

        Ok(())
    }
}
