use crate::simulation::{advance_oscillation_phase, oscillation_displacement};
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
/// bind group は用途単位（params / particles / spatial / contact）で再利用する。
struct GpuPhysicsBindGroups {
    params: BindGroup,
    particles: [BindGroup; 2],
    spatial: BindGroup,
    contact: BindGroup,
}

#[derive(Default)]
pub struct GpuPhysicsNode {
    /// 用途別バインドグループ一式
    bind_groups: Option<GpuPhysicsBindGroups>,
    /// 1フレームあたりのサブステップ数
    substeps: u32,
}

impl GpuPhysicsNode {
    pub fn new() -> Self {
        Self::default()
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

        // パイプラインキャッシュから用途別バインドグループレイアウトを取得
        let params_layout =
            pipeline_cache.get_bind_group_layout(&pipelines.params_bind_group_layout_desc);
        let particles_layout =
            pipeline_cache.get_bind_group_layout(&pipelines.particles_bind_group_layout_desc);
        let spatial_layout =
            pipeline_cache.get_bind_group_layout(&pipelines.spatial_bind_group_layout_desc);
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
                            pass.set_bind_group(0, &bind_groups.spatial, &[]);
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
                let mut phase = substep_phase.unwrap_or(container_params.oscillation_phase_start);
                if container_params.oscillation_enabled {
                    advance_oscillation_phase(
                        &mut phase,
                        container_params.oscillation_frequency,
                        runtime_params.dt,
                    );
                }
                runtime_params.container_offset = container_params.base_position_y
                    + oscillation_displacement(
                        container_params.oscillation_enabled,
                        container_params.oscillation_amplitude,
                        phase,
                    );
                substep_phase = Some(phase);
                render_queue.write_buffer(&buffers.params, 0, bytemuck::bytes_of(&runtime_params));
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
