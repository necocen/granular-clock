//! GPU physics state: combines buffer management, compute dispatch, and instance writing.

use std::num::NonZeroU64;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use super::buffers::{GpuPhysicsBuffers, ParticleGpu, SimulationParams, SortParamsGpu};
use super::pipeline::GpuPhysicsPipelines;
use crate::rendering::shaders::physics_shader_processor;
use crate::rendering::{LARGE_PARTICLE_COLOR, SMALL_PARTICLE_COLOR};
use crate::simulation::constants::SimulationConstants;
use bytemuck::{Pod, Zeroable};
use mikage::{create_compute_pipeline, glam, storage_buffer_entry, uniform_buffer_entry, wgpu};

/// Instance writer パラメータ
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceParams {
    num_particles: u32,
    num_large: u32,
    _pad: [u32; 2],
    large_color: [f32; 4],
    small_color: [f32; 4],
}

impl Default for InstanceParams {
    fn default() -> Self {
        Self {
            num_particles: 0,
            num_large: 0,
            _pad: [0; 2],
            large_color: LARGE_PARTICLE_COLOR,
            small_color: SMALL_PARTICLE_COLOR,
        }
    }
}

/// Bind group の一式
struct GpuPhysicsBindGroups {
    params: wgpu::BindGroup,
    particles: [wgpu::BindGroup; 2], // [forward, reverse]
    spatial: wgpu::BindGroup,
    sort_params: wgpu::BindGroup,
    contact: wgpu::BindGroup,
}

/// Readback 用ステージングバッファ
struct ReadbackStaging {
    buffer: wgpu::Buffer,
    size: u64,
    /// copy_buffer_to_buffer がエンコード済み（submit 待ち or map 待ち）
    copy_pending: bool,
    /// map_async が発行済み（完了待ち）
    map_requested: bool,
    mapping_complete: Arc<AtomicBool>,
    num_particles: u32,
    frame_counter: u32,
}

/// GPU 物理のステート。バッファ・パイプライン・bind group を一つにまとめる。
pub struct GpuPhysicsState {
    buffers: GpuPhysicsBuffers,
    pipelines: GpuPhysicsPipelines,
    bind_groups: Option<GpuPhysicsBindGroups>,

    // Instance writer
    instance_writer_pipeline: wgpu::ComputePipeline,
    instance_writer_bind_group_layout: wgpu::BindGroupLayout,
    instance_writer_bind_group: Option<wgpu::BindGroup>,
    instance_writer_params_buffer: wgpu::Buffer,

    // Sort params
    sort_params_packed: Vec<u8>,
    sort_stage_count: usize,
    cached_sort_count: u32,
    cached_sort_stride: u32,
    sort_params_dirty: bool,

    // Substeps
    substeps: u32,

    // Readback
    staging: Option<ReadbackStaging>,
    /// 読み戻したデータ（共有）
    readback_data: Arc<std::sync::RwLock<Vec<ParticleGpu>>>,
    readback_frame: Arc<std::sync::RwLock<u64>>,
    readback_interval: u32,
    last_readback_frame: u64,

    /// bind group 再構築が必要か
    needs_bind_group_rebuild: bool,
}

impl GpuPhysicsState {
    pub fn new(
        device: &wgpu::Device,
        constants: &SimulationConstants,
        instance_buffer: &wgpu::Buffer,
    ) -> Self {
        let particle = &constants.particle;
        let container = &constants.container;
        let grid_settings = constants.grid;

        let total_particles = particle.num_large + particle.num_small;
        let min_capacity = total_particles.max(2048);
        let desired_capacity = min_capacity
            .checked_next_power_of_two()
            .unwrap_or(min_capacity);

        let world_half = [
            container.half_extents.x,
            container.half_extents.y,
            container.half_extents.z,
        ];
        let grid_dim = grid_settings.compute_grid_dim(world_half);

        let buffers = GpuPhysicsBuffers::new(device, desired_capacity, grid_dim);
        let pipelines = GpuPhysicsPipelines::create(device);

        // Instance writer pipeline
        let cs = wgpu::ShaderStages::COMPUTE;
        let instance_writer_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("particle_to_instance_layout"),
                entries: &[
                    storage_buffer_entry(0, cs, true),
                    storage_buffer_entry(1, cs, false),
                    uniform_buffer_entry(2, cs),
                ],
            });

        let instance_source = physics_shader_processor()
            .resolve(include_str!(
                "../../../assets/shaders/particle_to_instance.wgsl"
            ))
            .expect("failed to resolve particle_to_instance shader");
        let instance_writer_pipeline = create_compute_pipeline(
            device,
            "particle_to_instance",
            &instance_source,
            &[&instance_writer_bind_group_layout],
            "main",
        );

        let instance_writer_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instance_compute_params"),
            size: std::mem::size_of::<InstanceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create initial instance writer bind group
        let instance_writer_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle_to_instance_bind_group"),
            layout: &instance_writer_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.latest_particles().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: instance_writer_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Readback staging
        let staging_buffer = GpuPhysicsBuffers::create_staging_buffer(device, desired_capacity);
        let staging_size = std::mem::size_of::<ParticleGpu>() as u64 * desired_capacity as u64;

        Self {
            buffers,
            pipelines,
            bind_groups: None,
            instance_writer_pipeline,
            instance_writer_bind_group_layout,
            instance_writer_bind_group: Some(instance_writer_bind_group),
            instance_writer_params_buffer,
            sort_params_packed: Vec::new(),
            sort_stage_count: 0,
            cached_sort_count: 0,
            cached_sort_stride: 0,
            sort_params_dirty: false,
            substeps: constants.settings.substeps_per_frame,
            staging: Some(ReadbackStaging {
                buffer: staging_buffer,
                size: staging_size,
                copy_pending: false,
                map_requested: false,
                mapping_complete: Arc::new(AtomicBool::new(false)),
                num_particles: 0,
                frame_counter: 0,
            }),
            readback_data: Arc::new(std::sync::RwLock::new(Vec::new())),
            readback_frame: Arc::new(std::sync::RwLock::new(0)),
            readback_interval: 4,
            last_readback_frame: 0,
            needs_bind_group_rebuild: true,
        }
    }

    /// 粒子データを GPU にアップロード
    pub fn upload_particles(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particles: &[ParticleGpu],
        instance_buffer: &wgpu::Buffer,
    ) {
        if particles.is_empty() {
            return;
        }

        let num_particles = particles.len() as u32;

        // 容量不足なら再確保
        if num_particles > self.buffers.capacity {
            let min_capacity = num_particles.max(2048);
            let desired_capacity = min_capacity
                .checked_next_power_of_two()
                .unwrap_or(min_capacity);

            // grid_dim は現状のパラメータから取得（再利用）
            // Use a reasonable default grid dim; will be overwritten by update_params
            let grid_dim = 32u32;
            self.buffers = GpuPhysicsBuffers::new(device, desired_capacity, grid_dim);

            // Re-create staging buffer
            let staging_buffer = GpuPhysicsBuffers::create_staging_buffer(device, desired_capacity);
            let staging_size = std::mem::size_of::<ParticleGpu>() as u64 * desired_capacity as u64;
            self.staging = Some(ReadbackStaging {
                buffer: staging_buffer,
                size: staging_size,
                copy_pending: false,
                map_requested: false,
                mapping_complete: Arc::new(AtomicBool::new(false)),
                num_particles: 0,
                frame_counter: 0,
            });

            self.needs_bind_group_rebuild = true;
        }

        // A/B 固定スロット運用のため、両方へ同一データを書き込む
        let particle_bytes = bytemuck::cast_slice(particles);
        queue.write_buffer(&self.buffers.particles_a, 0, particle_bytes);
        queue.write_buffer(&self.buffers.particles_b, 0, particle_bytes);
        self.buffers.num_particles = num_particles;
        self.needs_bind_group_rebuild = true;

        // Instance writer bind group も再構築
        self.instance_writer_bind_group =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("particle_to_instance_bind_group"),
                layout: &self.instance_writer_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffers.latest_particles().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: instance_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.instance_writer_params_buffer.as_entire_binding(),
                    },
                ],
            }));
    }

    /// シミュレーションパラメータを更新
    pub fn update_params(
        &mut self,
        queue: &wgpu::Queue,
        constants: &SimulationConstants,
        sim_state: &crate::simulation::state::SimulationState,
    ) {
        let container_params = &constants.container;
        let time_params = constants.time;
        let material = constants.material;
        let wall_props = constants.wall;
        let physics = constants.physics;
        let grid_settings = constants.grid;

        let world_half = [
            container_params.half_extents.x,
            container_params.half_extents.y,
            container_params.half_extents.z,
        ];

        let params = SimulationParams {
            dt: time_params.dt,
            gravity: physics.gravity.y,
            cell_size: grid_settings.cell_size,
            grid_dim: grid_settings.compute_grid_dim(world_half),
            world_half,
            num_particles: self.buffers.num_particles,
            youngs_modulus: material.youngs_modulus,
            poisson_ratio: material.poisson_ratio,
            restitution: material.restitution,
            friction: material.friction,
            container_offset: container_params.base_position.y + sim_state.container_offset,
            divider_height: container_params.divider_height,
            container_half_x: container_params.half_extents.x,
            container_half_y: container_params.half_extents.y,
            container_half_z: container_params.half_extents.z,
            divider_thickness: container_params.divider_thickness,
            rolling_friction: material.rolling_friction,
            wall_restitution: wall_props.restitution,
            wall_friction: wall_props.friction,
            wall_damping: wall_props.damping,
            wall_stiffness: wall_props.stiffness,
            _pad_end: 0.0,
        };

        let params_bytes = bytemuck::bytes_of(&params);
        queue.write_buffer(&self.buffers.params, 0, params_bytes);

        self.substeps = constants.settings.substeps_per_frame;
    }

    /// Bind group を再構築
    fn rebuild_bind_groups(&mut self, device: &wgpu::Device) {
        let buffers = &self.buffers;
        let pipelines = &self.pipelines;

        // Sort params の更新チェック
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

        // Params
        let params = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_physics_params_bind_group"),
            layout: &pipelines.params_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffers.params.as_entire_binding(),
            }],
        });

        // Particles (Forward: A→B, Reverse: B→A)
        let particles_forward = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_physics_particles_bind_group_forward"),
            layout: &pipelines.particles_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.forward_in().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.forward_out().as_entire_binding(),
                },
            ],
        });
        let particles_reverse = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_physics_particles_bind_group_reverse"),
            layout: &pipelines.particles_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.reverse_in().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.reverse_out().as_entire_binding(),
                },
            ],
        });

        // Spatial (keys / ids / cell_ranges)
        let spatial = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_physics_spatial_bind_group"),
            layout: &pipelines.spatial_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.particle_ids.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers.cell_ranges.as_entire_binding(),
                },
            ],
        });

        // Sort params (dynamic offset)
        let sort_params = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_physics_sort_params_bind_group"),
            layout: &pipelines.sort_params_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffers.sort_params,
                    offset: 0,
                    size: Some(
                        NonZeroU64::new(std::mem::size_of::<SortParamsGpu>() as u64)
                            .expect("SortParamsGpu size must be non-zero"),
                    ),
                }),
            }],
        });

        // Contact (forces / torques)
        let contact = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_physics_contact_bind_group"),
            layout: &pipelines.contact_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers.forces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.torques.as_entire_binding(),
                },
            ],
        });

        self.bind_groups = Some(GpuPhysicsBindGroups {
            params,
            particles: [particles_forward, particles_reverse],
            spatial,
            sort_params,
            contact,
        });

        self.needs_bind_group_rebuild = false;
    }

    /// sort params が dirty なら書き込む
    fn upload_sort_params(&mut self, queue: &wgpu::Queue) {
        if self.sort_stage_count as u32 <= self.buffers.sort_params_capacity_passes {
            if self.sort_params_dirty && !self.sort_params_packed.is_empty() {
                queue.write_buffer(&self.buffers.sort_params, 0, &self.sort_params_packed);
            }
        } else {
            self.sort_stage_count = 0;
            self.sort_params_packed.clear();
        }
        self.sort_params_dirty = false;
    }

    /// Compute pass をエンコード（物理サブステップ + instance writer）
    pub fn encode_compute(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        constants: &SimulationConstants,
        sim_state: &crate::simulation::state::SimulationState,
    ) {
        if self.buffers.num_particles == 0 {
            return;
        }

        // Bind group の再構築が必要なら行う
        if self.needs_bind_group_rebuild || self.bind_groups.is_none() {
            self.rebuild_bind_groups(device);
            self.sort_params_dirty = true;
        }

        // Sort params のアップロード
        self.upload_sort_params(queue);

        let Some(bind_groups) = &self.bind_groups else {
            return;
        };

        let num_particles = self.buffers.num_particles;
        let workgroups_particles_64 = num_particles.div_ceil(64);
        let sort_count = num_particles.next_power_of_two();
        if sort_count > self.buffers.capacity {
            return;
        }
        let workgroups_sort_64 = sort_count.div_ceil(64);
        let workgroups_sort_256 = sort_count.div_ceil(256);
        let params_size = std::mem::size_of::<SimulationParams>() as u64;

        // サブステップごとの params を一括で作成
        let substep_params = {
            use crate::simulation::constants::{
                advance_oscillation_phase, oscillation_displacement,
            };

            let container_params = &constants.container;
            let osc_params = &constants.oscillation;
            let time_params = constants.time;
            let material = constants.material;
            let wall_props = constants.wall;
            let physics = constants.physics;
            let grid_settings = constants.grid;

            let world_half = [
                container_params.half_extents.x,
                container_params.half_extents.y,
                container_params.half_extents.z,
            ];
            let grid_dim = grid_settings.compute_grid_dim(world_half);

            let base_params = SimulationParams {
                dt: time_params.dt,
                gravity: physics.gravity.y,
                cell_size: grid_settings.cell_size,
                grid_dim,
                world_half,
                num_particles,
                youngs_modulus: material.youngs_modulus,
                poisson_ratio: material.poisson_ratio,
                restitution: material.restitution,
                friction: material.friction,
                container_offset: container_params.base_position.y + sim_state.container_offset,
                divider_height: container_params.divider_height,
                container_half_x: container_params.half_extents.x,
                container_half_y: container_params.half_extents.y,
                container_half_z: container_params.half_extents.z,
                divider_thickness: container_params.divider_thickness,
                rolling_friction: material.rolling_friction,
                wall_restitution: wall_props.restitution,
                wall_friction: wall_props.friction,
                wall_damping: wall_props.damping,
                wall_stiffness: wall_props.stiffness,
                _pad_end: 0.0,
            };

            let mut phase = sim_state.oscillation_frame_start_phase;
            let mut params_list = Vec::with_capacity(self.substeps as usize);
            for _ in 0..self.substeps {
                let mut runtime_params = base_params;
                if osc_params.enabled {
                    advance_oscillation_phase(&mut phase, osc_params.frequency, runtime_params.dt);
                    runtime_params.container_offset = container_params.base_position.y
                        + oscillation_displacement(osc_params.amplitude, phase);
                }
                params_list.push(runtime_params);
            }
            params_list
        };

        let params_upload = if !substep_params.is_empty() {
            Some(GpuPhysicsBuffers::create_params_upload_buffer(
                device,
                &substep_params,
            ))
        } else {
            None
        };

        // Encode substeps
        for substep_idx in 0..self.substeps {
            // params 上書き
            if let Some(upload) = params_upload.as_ref() {
                encoder.copy_buffer_to_buffer(
                    upload,
                    substep_idx as u64 * params_size,
                    &self.buffers.params,
                    0,
                    params_size,
                );
            }

            // 前半（A -> B）
            self.encode_half_step(
                encoder,
                bind_groups,
                &bind_groups.particles[0],
                &self.pipelines.integrate_first_half_pipeline,
                workgroups_particles_64,
                workgroups_sort_64,
                workgroups_sort_256,
            );

            // 後半（B -> A）
            self.encode_half_step(
                encoder,
                bind_groups,
                &bind_groups.particles[1],
                &self.pipelines.integrate_second_half_pipeline,
                workgroups_particles_64,
                workgroups_sort_64,
                workgroups_sort_256,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_half_step(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_groups: &GpuPhysicsBindGroups,
        particles_bind_group: &wgpu::BindGroup,
        integrate_pipeline: &wgpu::ComputePipeline,
        workgroups_particles_64: u32,
        workgroups_sort_64: u32,
        workgroups_sort_256: u32,
    ) {
        // Clear buffers
        encoder.clear_buffer(&self.buffers.cell_ranges, 0, None);
        encoder.clear_buffer(&self.buffers.forces, 0, None);
        encoder.clear_buffer(&self.buffers.torques, 0, None);

        // Hash keys
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("hash_keys_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.hash_keys_pipeline);
            pass.set_bind_group(0, &bind_groups.params, &[]);
            pass.set_bind_group(1, particles_bind_group, &[]);
            pass.set_bind_group(2, &bind_groups.spatial, &[]);
            pass.dispatch_workgroups(workgroups_sort_64, 1, 1);
        }

        // Bitonic sort
        if self.sort_stage_count > 0
            && (self.sort_stage_count as u32) <= self.buffers.sort_params_capacity_passes
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("bitonic_sort_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.bitonic_sort_pipeline);
            pass.set_bind_group(0, &bind_groups.spatial, &[]);
            for stage_idx in 0..self.sort_stage_count {
                let dynamic_offset = stage_idx as u32 * self.buffers.sort_params_stride;
                pass.set_bind_group(1, &bind_groups.sort_params, &[dynamic_offset]);
                pass.dispatch_workgroups(workgroups_sort_256, 1, 1);
            }
        }

        // Cell ranges
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cell_ranges_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.cell_ranges_pipeline);
            pass.set_bind_group(0, &bind_groups.params, &[]);
            pass.set_bind_group(1, &bind_groups.spatial, &[]);
            pass.dispatch_workgroups(workgroups_particles_64, 1, 1);
        }

        // Collision
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("collision_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.collision_pipeline);
            pass.set_bind_group(0, &bind_groups.params, &[]);
            pass.set_bind_group(1, particles_bind_group, &[]);
            pass.set_bind_group(2, &bind_groups.spatial, &[]);
            pass.set_bind_group(3, &bind_groups.contact, &[]);
            pass.dispatch_workgroups(workgroups_particles_64, 1, 1);
        }

        // Integrate
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("integrate_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(integrate_pipeline);
            pass.set_bind_group(0, &bind_groups.params, &[]);
            pass.set_bind_group(1, particles_bind_group, &[]);
            pass.set_bind_group(2, &bind_groups.contact, &[]);
            pass.dispatch_workgroups(workgroups_particles_64, 1, 1);
        }
    }

    /// particle_to_instance compute pass をエンコード
    pub fn encode_particle_to_instance(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        constants: &SimulationConstants,
    ) {
        let Some(bind_group) = &self.instance_writer_bind_group else {
            return;
        };

        let particle = &constants.particle;
        let total = particle.num_large + particle.num_small;
        if total == 0 || self.buffers.num_particles < total {
            return;
        }

        let workgroups = total.div_ceil(64);

        let params = InstanceParams {
            num_particles: total,
            num_large: particle.num_large,
            _pad: [0; 2],
            large_color: LARGE_PARTICLE_COLOR,
            small_color: SMALL_PARTICLE_COLOR,
        };
        queue.write_buffer(
            &self.instance_writer_params_buffer,
            0,
            bytemuck::bytes_of(&params),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu_instance_write_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.instance_writer_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Readback: particles_a をステージングバッファにコピー（N フレームに1回）。
    ///
    /// copy_buffer_to_buffer のみエンコードする。
    /// map_async は submit 後に `process_readback` で発行する。
    pub fn encode_readback(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let Some(ref mut staging) = self.staging else {
            return;
        };

        // 前回の copy/map がまだ完了していなければ skip
        if self.buffers.num_particles == 0 || staging.copy_pending || staging.map_requested {
            return;
        }

        staging.frame_counter += 1;
        if staging.frame_counter < self.readback_interval {
            return;
        }
        staging.frame_counter = 0;

        let particle_size = std::mem::size_of::<ParticleGpu>() as u64;
        let copy_size = particle_size * self.buffers.num_particles as u64;

        if copy_size > staging.size {
            return;
        }

        encoder.copy_buffer_to_buffer(
            self.buffers.latest_particles(),
            0,
            &staging.buffer,
            0,
            copy_size,
        );

        staging.num_particles = self.buffers.num_particles;
        staging.copy_pending = true;
    }

    /// Submit 後に呼ぶ。copy 完了後に map_async を発行し、マッピング完了時にデータを読み取る。
    pub fn process_readback(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Poll);

        let Some(ref mut staging) = self.staging else {
            return;
        };

        // ステート: copy_pending=true, map_requested=false → map_async を発行
        if staging.copy_pending && !staging.map_requested {
            staging.copy_pending = false;
            staging.map_requested = true;
            staging.mapping_complete.store(false, Ordering::SeqCst);

            let particle_size = std::mem::size_of::<ParticleGpu>() as u64;
            let copy_size = particle_size * staging.num_particles as u64;
            let mapping_complete = staging.mapping_complete.clone();
            let buffer_slice = staging.buffer.slice(0..copy_size);
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    mapping_complete.store(true, Ordering::SeqCst);
                }
            });
            return;
        }

        // ステート: map_requested=true → 完了チェック
        if !staging.map_requested {
            return;
        }

        if !staging.mapping_complete.load(Ordering::SeqCst) {
            return;
        }

        // マッピング完了: データ読み取り
        let particle_size = std::mem::size_of::<ParticleGpu>();
        let num_particles = staging.num_particles as usize;
        let byte_len = particle_size * num_particles;

        let buffer_slice = staging.buffer.slice(0..byte_len as u64);
        let data = buffer_slice
            .get_mapped_range()
            .expect("completed readback buffer should be mapped");

        if data.len() >= byte_len {
            let particles: &[ParticleGpu] = bytemuck::cast_slice(&data[..byte_len]);

            if let Ok(mut guard) = self.readback_data.write() {
                guard.clear();
                guard.extend_from_slice(particles);
            }

            if let Ok(mut frame) = self.readback_frame.write() {
                *frame += 1;
            }
        }

        drop(data);
        staging.buffer.unmap();
        staging.map_requested = false;
        staging.mapping_complete.store(false, Ordering::SeqCst);
    }

    /// 読み戻しデータから ParticleStore を更新
    pub fn apply_readback_to_store(&mut self, store: &mut crate::physics::ParticleStore) {
        let current_frame = {
            let guard = self.readback_frame.read().unwrap();
            *guard
        };

        if current_frame == self.last_readback_frame || current_frame == 0 {
            return;
        }
        self.last_readback_frame = current_frame;

        let guard = self.readback_data.read().unwrap();
        if guard.is_empty() {
            return;
        }

        let num = guard.len().min(store.particles.len());
        for i in 0..num {
            let gp = &guard[i];
            let p = &mut store.particles[i];
            p.position = glam::Vec3::new(gp.pos[0], gp.pos[1], gp.pos[2]);
            p.velocity = glam::Vec3::new(gp.vel[0], gp.vel[1], gp.vel[2]);
            p.angular_velocity = glam::Vec3::new(gp.omega[0], gp.omega[1], gp.omega[2]);
        }
    }

    /// Instance writer bind group を再構築（instance buffer が変わった場合）
    pub fn rebuild_instance_bind_group(
        &mut self,
        device: &wgpu::Device,
        instance_buffer: &wgpu::Buffer,
    ) {
        self.instance_writer_bind_group =
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("particle_to_instance_bind_group"),
                layout: &self.instance_writer_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffers.latest_particles().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: instance_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.instance_writer_params_buffer.as_entire_binding(),
                    },
                ],
            }));
    }

    /// 粒子数
    pub fn num_particles(&self) -> u32 {
        self.buffers.num_particles
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
