use bevy::{
    prelude::*,
    render::{
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
    },
};

use super::{
    buffers::{GpuPhysicsBuffers, ParticleGpu},
    pipeline::GpuPhysicsPipelines,
    plugin::GpuParticleData,
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
        let Some(integrate_pipeline) =
            pipeline_cache.get_compute_pipeline(pipelines.integrate_pipeline)
        else {
            return Ok(());
        };

        let num_particles = buffers.num_particles;
        if num_particles == 0 {
            return Ok(());
        }

        let workgroups_64 = num_particles.div_ceil(64);
        let workgroups_256 = num_particles.div_ceil(256);

        let encoder = render_context.command_encoder();

        // 2のべき乗に切り上げ（bitonic sort 用、ループ外で計算）
        let n = num_particles.next_power_of_two();

        // サブステップをループ実行
        // 偶数ステップ (0, 2, 4, ...) は forward (in→out)
        // 奇数ステップ (1, 3, 5, ...) は reverse (out→in)
        for step in 0..self.substeps {
            let bind_group = &bind_groups[step as usize % 2];

            // バッファをクリア（各サブステップで必要）
            encoder.clear_buffer(&buffers.cell_ranges, 0, None);
            encoder.clear_buffer(&buffers.forces, 0, None);
            encoder.clear_buffer(&buffers.torques, 0, None);

            // Pass 1: Hash Keys - 各粒子のセルキーを計算
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("hash_keys_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(hash_keys_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(workgroups_64, 1, 1);
            }

            // Pass 2: Bitonic Sort - (key, particle_id) ペアをソート
            {
                let mut k = 2u32;
                while k <= n {
                    let mut j = k / 2;
                    while j > 0 {
                        let push_constants = [j, k];
                        let push_constant_bytes = bytemuck::bytes_of(&push_constants);

                        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                            label: Some("bitonic_sort_pass"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(bitonic_sort_pipeline);
                        pass.set_bind_group(0, bind_group, &[]);
                        pass.set_push_constants(0, push_constant_bytes);
                        pass.dispatch_workgroups(workgroups_256, 1, 1);
                        j /= 2;
                    }
                    k *= 2;
                }
            }

            // Pass 3: Cell Ranges - 各セルの開始/終了インデックスを構築
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("cell_ranges_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(cell_ranges_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(workgroups_64, 1, 1);
            }

            // Pass 4: Collision - 衝突検出と接触力計算
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("collision_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(collision_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(workgroups_64, 1, 1);
            }

            // Pass 5: Integrate - 速度・位置の更新
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some("integrate_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(integrate_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(workgroups_64, 1, 1);
            }
        }

        // サブステップ数が偶数の場合、最終結果が current_input() に残るので
        // current_output() にコピーして copy_to_staging との整合性を保つ
        if self.substeps > 0 && self.substeps.is_multiple_of(2) {
            let copy_size = std::mem::size_of::<ParticleGpu>() as u64 * num_particles as u64;
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
