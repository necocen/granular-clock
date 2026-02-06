use bevy::{
    prelude::*,
    render::{
        render_graph::{Node, NodeRunError, RenderGraphContext, RenderLabel},
        render_resource::*,
        renderer::{RenderContext, RenderDevice},
    },
};

use super::{buffers::GpuPhysicsBuffers, pipeline::GpuPhysicsPipelines, plugin::GpuParticleData};

/// 物理計算ノードのラベル
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct GpuPhysicsLabel;

/// GPU 物理計算を実行するレンダーグラフノード
pub struct GpuPhysicsNode {
    /// バインドグループ（フレームごとに更新）
    bind_group: Option<BindGroup>,
}

impl Default for GpuPhysicsNode {
    fn default() -> Self {
        Self { bind_group: None }
    }
}

impl GpuPhysicsNode {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Node for GpuPhysicsNode {
    fn update(&mut self, world: &mut World) {
        // 一時停止中はスワップも計算もスキップ
        if let Some(gpu_data) = world.get_resource::<GpuParticleData>() {
            if gpu_data.paused {
                self.bind_group = None;
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

        // バインドグループを作成（BindGroupEntries::sequential を使用）
        self.bind_group = Some(render_device.create_bind_group(
            Some("gpu_physics_bind_group"),
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
        ));
    }

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let Some(bind_group) = &self.bind_group else {
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

        let workgroups_64 = (num_particles + 63) / 64;
        let workgroups_256 = (num_particles + 255) / 256;

        let encoder = render_context.command_encoder();

        // cell_ranges バッファをクリア（前フレームのゴミデータを除去）
        encoder.clear_buffer(&buffers.cell_ranges, 0, None);

        // forces バッファもクリア
        encoder.clear_buffer(&buffers.forces, 0, None);

        // torques バッファもクリア
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
        // 完全なbitonic sortには log2(n) * (log2(n) + 1) / 2 パスが必要
        {
            // 2のべき乗に切り上げ
            let n = num_particles.next_power_of_two();

            // Bitonic sort の全パス
            let mut k = 2u32;
            while k <= n {
                let mut j = k / 2;
                while j > 0 {
                    // Push constants: [step_size, pass_offset]
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

        Ok(())
    }
}
