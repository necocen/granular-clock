//! GPU-only instance writer.
//! Converts latest GPU particle buffer into the shared instance buffer.

use bevy::{
    prelude::*,
    render::{
        Render, RenderApp, RenderSystems,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        graph::CameraDriverLabel,
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext, RenderLabel},
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderQueue},
    },
};
use bytemuck::{Pod, Zeroable};

use crate::{
    rendering::{
        InstanceBuffer, LARGE_PARTICLE_COLOR, ParticleBatchMarker, RenderInstanceBufferResource,
        SMALL_PARTICLE_COLOR, is_gpu_backend, normalized_instance_capacity,
    },
    simulation::constants::{PhysicsBackend, SimulationConstants},
};

use super::{
    buffers::GpuPhysicsBuffers,
    node::GpuPhysicsLabel,
    shaders::{
        PARTICLE_TO_INSTANCE_SHADER_HANDLE, PHYSICS_TYPES_SHADER_HANDLE, load_gpu_internal_shaders,
    },
};

impl ExtractResource for SimulationConstants {
    type Source = SimulationConstants;

    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

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

#[derive(Resource)]
struct GpuInstanceWriterResources {
    params_buffer: Buffer,
    compute_pipeline: CachedComputePipelineId,
    /// 共有型定義シェーダー（#import 解決用にハンドルを保持）
    _physics_types_shader: Handle<Shader>,
}

fn instance_compute_bind_group_layout_desc() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "particle_to_instance_layout",
        &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
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
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    )
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
pub struct GpuInstanceWriteLabel;

#[derive(Default)]
pub struct GpuInstanceWriteNode {
    bind_group: Option<BindGroup>,
    workgroups: u32,
    params: InstanceParams,
}

impl GpuInstanceWriteNode {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Node for GpuInstanceWriteNode {
    fn update(&mut self, world: &mut World) {
        self.workgroups = 0;
        self.params = InstanceParams::default();

        let Some(backend) = world.get_resource::<PhysicsBackend>() else {
            return;
        };
        if *backend != PhysicsBackend::Gpu {
            return;
        }

        let Some(constants) = world.get_resource::<SimulationConstants>() else {
            return;
        };
        let particle = &constants.particle;
        let total = particle.num_large + particle.num_small;
        if total == 0 {
            return;
        }

        let Some(physics_buffers) = world.get_resource::<GpuPhysicsBuffers>() else {
            return;
        };
        if physics_buffers.num_particles < total {
            return;
        }

        let Some(instance_buffer) = world.get_resource::<RenderInstanceBufferResource>() else {
            return;
        };
        if instance_buffer.len == 0 {
            return;
        }

        let Some(resources) = world.get_resource::<GpuInstanceWriterResources>() else {
            return;
        };
        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_compute_pipeline(resources.compute_pipeline) else {
            return;
        };

        let needs_rebuild = self.bind_group.is_none()
            || world.is_resource_changed::<GpuPhysicsBuffers>()
            || world.is_resource_changed::<RenderInstanceBufferResource>()
            || world.is_resource_changed::<GpuInstanceWriterResources>();
        if needs_rebuild {
            let render_device = world.resource::<RenderDevice>();
            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let bind_group_layout: BindGroupLayout = bind_group_layout.clone().into();
            let bind_group = render_device.create_bind_group(
                Some("particle_to_instance_bind_group"),
                &bind_group_layout,
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: physics_buffers.latest_particles().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: instance_buffer.buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: resources.params_buffer.as_entire_binding(),
                    },
                ],
            );
            self.bind_group = Some(bind_group);
        }

        self.workgroups = total.div_ceil(64);
        self.params = InstanceParams {
            num_particles: total,
            num_large: particle.num_large,
            _pad: [0; 2],
            large_color: LARGE_PARTICLE_COLOR,
            small_color: SMALL_PARTICLE_COLOR,
        };
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
        if self.workgroups == 0 {
            return Ok(());
        }

        let Some(resources) = world.get_resource::<GpuInstanceWriterResources>() else {
            return Ok(());
        };

        let pipeline_cache = world.resource::<PipelineCache>();
        let Some(pipeline) = pipeline_cache.get_compute_pipeline(resources.compute_pipeline) else {
            return Ok(());
        };

        let render_queue = world.resource::<RenderQueue>();
        render_queue.write_buffer(
            &resources.params_buffer,
            0,
            bytemuck::bytes_of(&self.params),
        );

        let encoder = render_context.command_encoder();
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("gpu_instance_write_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(self.workgroups, 1, 1);

        Ok(())
    }
}

pub struct GpuInstanceWriterPlugin;

impl Plugin for GpuInstanceWriterPlugin {
    fn build(&self, app: &mut App) {
        load_gpu_internal_shaders(app);

        app.add_plugins(ExtractResourcePlugin::<SimulationConstants>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.add_systems(
            Render,
            (init_instance_writer_resources, prepare_gpu_instance_buffer)
                .chain()
                .in_set(RenderSystems::PrepareResources)
                .run_if(is_gpu_backend),
        );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(GpuInstanceWriteLabel, GpuInstanceWriteNode::new());
        render_graph.add_node_edge(GpuPhysicsLabel, GpuInstanceWriteLabel);
        render_graph.add_node_edge(GpuInstanceWriteLabel, CameraDriverLabel);
    }
}

fn init_instance_writer_resources(
    mut commands: Commands,
    resources: Option<Res<GpuInstanceWriterResources>>,
    render_device: Res<RenderDevice>,
    pipeline_cache: Res<PipelineCache>,
) {
    if resources.is_some() {
        return;
    }

    let params_buffer = render_device.create_buffer(&BufferDescriptor {
        label: Some("instance_compute_params"),
        size: std::mem::size_of::<InstanceParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let physics_types_shader = PHYSICS_TYPES_SHADER_HANDLE.clone();
    let shader = PARTICLE_TO_INSTANCE_SHADER_HANDLE.clone();
    let compute_pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
        label: Some("particle_to_instance_pipeline".into()),
        layout: vec![instance_compute_bind_group_layout_desc()],
        shader,
        shader_defs: vec![],
        entry_point: Some("main".into()),
        push_constant_ranges: vec![],
        zero_initialize_workgroup_memory: true,
    });

    commands.insert_resource(GpuInstanceWriterResources {
        params_buffer,
        compute_pipeline,
        _physics_types_shader: physics_types_shader,
    });
}

fn prepare_gpu_instance_buffer(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    constants: Res<SimulationConstants>,
    instance_buffer: Option<ResMut<RenderInstanceBufferResource>>,
    batch_query: Query<Entity, With<ParticleBatchMarker>>,
) {
    let particle = &constants.particle;
    let instance_len = particle.num_large + particle.num_small;

    if instance_len > 0 {
        let required_capacity = normalized_instance_capacity(instance_len);
        let needs_realloc = instance_buffer
            .as_ref()
            .map(|res| res.capacity < required_capacity)
            .unwrap_or(true);

        if needs_realloc {
            commands.insert_resource(RenderInstanceBufferResource::new(
                &render_device,
                required_capacity,
            ));
            return;
        }
    }

    let Some(mut instance_buffer) = instance_buffer else {
        return;
    };

    instance_buffer.len = instance_len;

    for entity in &batch_query {
        commands.entity(entity).insert(InstanceBuffer {
            buffer: instance_buffer.buffer.clone(),
            len: instance_buffer.len,
        });
    }
}
