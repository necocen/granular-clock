//! Instanced Rendering for particles (both CPU and GPU physics modes)
//!
//! Bevy の SpecializedMeshPipeline パターンを使って、
//! パーティクルをインスタンス描画する。
//! - GPU モード: compute shader で物理バッファ → インスタンスバッファ変換
//! - CPU モード: ECS の Position から CPU でインスタンスデータを構築してアップロード

use bevy::{
    camera::visibility::NoFrustumCulling,
    core_pipeline::core_3d::Transparent3d,
    ecs::system::{lifetimeless::*, SystemParamItem},
    mesh::VertexBufferLayout,
    pbr::{
        MeshPipeline, MeshPipelineKey, RenderMeshInstances, SetMeshBindGroup,
        SetMeshViewBindGroup, SetMeshViewBindingArrayBindGroup,
    },
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        mesh::{allocator::MeshAllocator, RenderMesh, RenderMeshBufferInfo},
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
            RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::*,
        renderer::{RenderDevice, RenderQueue},
        view::ExtractedView,
        Render, RenderApp, RenderStartup, RenderSystems,
    },
};
use bytemuck::{Pod, Zeroable};

use crate::gpu::GpuPhysicsBuffers;
use crate::physics::{ParticleProperties, ParticleSize, Position};
use crate::rendering::{ParticleMeshes, SimulationConfig};
use crate::simulation::PhysicsBackend;

// ──────────────────── Data types ────────────────────

/// Instance data layout: position + scale + color (32 bytes)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct InstanceData {
    /// xyz = world position, w = radius (uniform scale)
    pub pos_scale: [f32; 4],
    /// RGBA color
    pub color: [f32; 4],
}

/// Params for the particle-to-instance compute shader (GPU mode only)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct InstanceParams {
    pub num_particles: u32,
    pub num_large: u32,
    pub _pad: [u32; 2],
    pub large_color: [f32; 4],
    pub small_color: [f32; 4],
}

// ──────────────────── Main World components/resources ────────────────────

/// Marker component for the particle batch proxy entity
#[derive(Component, Clone)]
pub struct ParticleBatchMarker;

impl ExtractComponent for ParticleBatchMarker {
    type QueryData = &'static ParticleBatchMarker;
    type QueryFilter = ();
    type Out = Self;

    fn extract_component(
        _item: bevy::ecs::query::QueryItem<'_, '_, Self::QueryData>,
    ) -> Option<Self> {
        Some(ParticleBatchMarker)
    }
}

/// Extracted particle counts (Main World → Render World)
#[derive(Resource, Clone, Default)]
pub struct ExtractedParticleCounts {
    pub num_large: u32,
    pub num_small: u32,
    pub large_color: [f32; 4],
    pub small_color: [f32; 4],
}

impl ExtractResource for ExtractedParticleCounts {
    type Source = ExtractedParticleCounts;
    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

/// CPU mode instance data (Main World → Render World)
#[derive(Resource, Clone, Default)]
pub struct CpuInstanceData(pub Vec<InstanceData>);

impl ExtractResource for CpuInstanceData {
    type Source = CpuInstanceData;
    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

/// Physics backend extracted to Render World
impl ExtractResource for PhysicsBackend {
    type Source = PhysicsBackend;
    fn extract_resource(source: &Self::Source) -> Self {
        *source
    }
}

// ──────────────────── Render World components/resources ────────────────────

/// Instance buffer component attached to batch proxy entity in Render World
#[derive(Component)]
pub struct InstanceBuffer {
    pub buffer: Buffer,
    pub len: u32,
}

/// Persistent resources for particle instancing (Render World)
#[derive(Resource)]
pub struct ParticleInstanceResources {
    /// Combined instance buffer (all particles)
    pub instance_buffer: Buffer,
    /// Params uniform buffer (GPU mode compute shader)
    pub params_buffer: Buffer,
    /// Compute pipeline for particle-to-instance conversion (GPU mode)
    pub compute_pipeline: CachedComputePipelineId,
    /// Current capacity
    pub capacity: u32,
}

/// Custom pipeline extending MeshPipeline for instanced particle rendering
#[derive(Resource)]
pub struct ParticleInstancePipeline {
    pub shader: Handle<Shader>,
    pub mesh_pipeline: MeshPipeline,
}

impl SpecializedMeshPipeline for ParticleInstancePipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &bevy::mesh::MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut desc = self.mesh_pipeline.specialize(key, layout)?;
        desc.vertex.shader = self.shader.clone();
        desc.fragment.as_mut().unwrap().shader = self.shader.clone();

        // パーティクルは不透明: デプス書き込み有効、ブレンド無し
        // Transparent3d フェーズで描画されるが、デプスバッファに書き込むことで
        // 後から描画される仕切り等の透明オブジェクトとの前後関係を正しく処理する
        if let Some(ref mut depth_stencil) = desc.depth_stencil {
            depth_stencil.depth_write_enabled = true;
        }
        if let Some(ref mut frag) = desc.fragment {
            for target in &mut frag.targets {
                if let Some(ref mut state) = target {
                    state.blend = None;
                }
            }
        }

        // Add per-instance vertex buffer layout
        desc.vertex.buffers.push(VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceData>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: vec![
                // @location(3): pos_scale (vec4<f32>)
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 3,
                },
                // @location(4): color (vec4<f32>)
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: VertexFormat::Float32x4.size(),
                    shader_location: 4,
                },
            ],
        });

        Ok(desc)
    }
}

// ──────────────────── Draw commands ────────────────────

type DrawParticleInstanced = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshViewBindingArrayBindGroup<1>,
    SetMeshBindGroup<2>,
    DrawMeshInstanced,
);

struct DrawMeshInstanced;

impl<P: PhaseItem> RenderCommand<P> for DrawMeshInstanced {
    type Param = (
        SRes<RenderAssets<RenderMesh>>,
        SRes<RenderMeshInstances>,
        SRes<MeshAllocator>,
    );
    type ViewQuery = ();
    type ItemQuery = Read<InstanceBuffer>;

    fn render<'w>(
        item: &P,
        _view: (),
        instance_buffer: Option<&'w InstanceBuffer>,
        (meshes, mesh_instances, allocator): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(inst) = instance_buffer else {
            return RenderCommandResult::Skip;
        };
        if inst.len == 0 {
            return RenderCommandResult::Skip;
        }

        let meshes = meshes.into_inner();
        let mesh_instances = mesh_instances.into_inner();
        let allocator = allocator.into_inner();

        let Some(mesh_instance) = mesh_instances.render_mesh_queue_data(item.main_entity())
        else {
            return RenderCommandResult::Skip;
        };
        let Some(gpu_mesh) = meshes.get(mesh_instance.mesh_asset_id) else {
            return RenderCommandResult::Skip;
        };

        let Some(vertex_slice) = allocator.mesh_vertex_slice(&mesh_instance.mesh_asset_id) else {
            return RenderCommandResult::Skip;
        };

        pass.set_vertex_buffer(0, vertex_slice.buffer.slice(..));
        pass.set_vertex_buffer(1, inst.buffer.slice(..));

        match &gpu_mesh.buffer_info {
            RenderMeshBufferInfo::Indexed {
                index_format,
                count,
            } => {
                let Some(index_slice) =
                    allocator.mesh_index_slice(&mesh_instance.mesh_asset_id)
                else {
                    return RenderCommandResult::Skip;
                };
                pass.set_index_buffer(index_slice.buffer.slice(..), *index_format);
                pass.draw_indexed(
                    index_slice.range.start..(index_slice.range.start + count),
                    vertex_slice.range.start as i32,
                    0..inst.len,
                );
            }
            RenderMeshBufferInfo::NonIndexed => {
                pass.draw(vertex_slice.range.clone(), 0..inst.len);
            }
        }

        RenderCommandResult::Success
    }
}

// ──────────────────── Plugin ────────────────────

pub struct GpuInstancingPlugin;

impl Plugin for GpuInstancingPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ExtractedParticleCounts::default())
            .insert_resource(CpuInstanceData::default())
            .add_plugins(ExtractResourcePlugin::<ExtractedParticleCounts>::default())
            .add_plugins(ExtractResourcePlugin::<CpuInstanceData>::default())
            .add_plugins(ExtractResourcePlugin::<PhysicsBackend>::default())
            .add_plugins(ExtractComponentPlugin::<ParticleBatchMarker>::default())
            .add_systems(PostUpdate, sync_particle_counts)
            .add_systems(PostUpdate, build_cpu_instance_data)
            .add_systems(
                Startup,
                spawn_particle_batch.after(crate::rendering::setup_rendering),
            );

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedMeshPipelines<ParticleInstancePipeline>>()
            .add_render_command::<Transparent3d, DrawParticleInstanced>()
            .add_systems(RenderStartup, init_particle_render_pipeline)
            .add_systems(
                Render,
                (
                    prepare_particle_instances.in_set(RenderSystems::PrepareResources),
                    queue_particle_instances.in_set(RenderSystems::QueueMeshes),
                ),
            );
    }
}

// ──────────────────── Main World systems ────────────────────

/// Sync particle counts from SimulationConfig to extracted resource
fn sync_particle_counts(
    config: Res<SimulationConfig>,
    mut counts: ResMut<ExtractedParticleCounts>,
) {
    counts.num_large = config.num_large;
    counts.num_small = config.num_small;
    counts.large_color = [0.8, 0.2, 0.2, 1.0]; // 赤
    counts.small_color = [0.2, 0.2, 0.8, 1.0]; // 青
}

/// Build instance data from ECS (CPU mode)
fn build_cpu_instance_data(
    backend: Res<PhysicsBackend>,
    particles: Query<(&Position, &ParticleProperties, &ParticleSize)>,
    mut cpu_data: ResMut<CpuInstanceData>,
) {
    if *backend != PhysicsBackend::Cpu {
        cpu_data.0.clear();
        return;
    }

    cpu_data.0.clear();
    for (pos, props, size) in &particles {
        let color = match size {
            ParticleSize::Large => [0.8, 0.2, 0.2, 1.0],
            ParticleSize::Small => [0.2, 0.2, 0.8, 1.0],
        };
        cpu_data.0.push(InstanceData {
            pos_scale: [pos.0.x, pos.0.y, pos.0.z, props.radius],
            color,
        });
    }
}

/// Spawn the particle batch proxy entity for instanced rendering
fn spawn_particle_batch(mut commands: Commands, meshes: Res<ParticleMeshes>) {
    commands.spawn((
        Mesh3d(meshes.sphere.clone()),
        Transform::IDENTITY,
        Visibility::default(),
        ParticleBatchMarker,
        NoFrustumCulling,
        // No MeshMaterial3d → Bevy's standard mesh rendering won't draw this
    ));
}

/// Hide individual particle entities (instancing handles all rendering)
pub fn hide_particle_entities(
    mut particles: Query<
        &mut Visibility,
        (
            Added<Position>,
            Without<ParticleBatchMarker>,
        ),
    >,
) {
    for mut v in &mut particles {
        *v = Visibility::Hidden;
    }
}

// ──────────────────── Render World systems ────────────────────

/// Initialize the instanced particle render pipeline (one-time)
fn init_particle_render_pipeline(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mesh_pipeline: Res<MeshPipeline>,
) {
    commands.insert_resource(ParticleInstancePipeline {
        shader: asset_server.load("shaders/particle_instancing.wgsl"),
        mesh_pipeline: mesh_pipeline.clone(),
    });
}

/// Compute bind group layout for particle-to-instance shader
fn instance_compute_bind_group_layout_desc() -> BindGroupLayoutDescriptor {
    BindGroupLayoutDescriptor::new(
        "particle_to_instance_layout",
        &[
            // binding 0: particles (storage, read)
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
            // binding 1: instances (storage, read_write)
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
            // binding 2: params (uniform)
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

/// Prepare instance buffer: GPU mode uses compute shader, CPU mode uploads from ECS
fn prepare_particle_instances(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    asset_server: Res<AssetServer>,
    counts: Res<ExtractedParticleCounts>,
    backend: Option<Res<PhysicsBackend>>,
    physics_buffers: Option<Res<GpuPhysicsBuffers>>,
    cpu_data: Option<Res<CpuInstanceData>>,
    mut resources: Option<ResMut<ParticleInstanceResources>>,
    batch_query: Query<Entity, With<ParticleBatchMarker>>,
) {
    let total = counts.num_large + counts.num_small;
    if total == 0 {
        return;
    }

    let is_gpu = backend
        .as_ref()
        .map(|b| **b == PhysicsBackend::Gpu)
        .unwrap_or(true);

    // Create resources if they don't exist
    if resources.is_none() {
        let capacity = total.max(2048);
        let instance_size = std::mem::size_of::<InstanceData>() as u64;

        let instance_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("particle_instance_buffer"),
            size: instance_size * capacity as u64,
            usage: BufferUsages::VERTEX | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("instance_compute_params"),
            size: std::mem::size_of::<InstanceParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = asset_server.load("shaders/particle_to_instance.wgsl");
        let compute_pipeline =
            pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("particle_to_instance_pipeline".into()),
                layout: vec![instance_compute_bind_group_layout_desc()],
                shader,
                shader_defs: vec![],
                entry_point: Some("main".into()),
                push_constant_ranges: vec![],
                zero_initialize_workgroup_memory: true,
            });

        commands.insert_resource(ParticleInstanceResources {
            instance_buffer,
            params_buffer,
            compute_pipeline,
            capacity,
        });
        return; // Resources will be available next frame
    }

    let res = resources.as_mut().unwrap();

    let instance_len;

    if is_gpu {
        // ── GPU mode: compute shader converts physics buffer → instance buffer ──
        let Some(physics_buffers) = physics_buffers else {
            return;
        };
        if physics_buffers.num_particles == 0 {
            return;
        }

        // Update params uniform
        let params = InstanceParams {
            num_particles: total,
            num_large: counts.num_large,
            _pad: [0; 2],
            large_color: counts.large_color,
            small_color: counts.small_color,
        };
        render_queue.write_buffer(&res.params_buffer, 0, bytemuck::bytes_of(&params));

        // Get compute pipeline (might not be compiled yet)
        let Some(pipeline) = pipeline_cache.get_compute_pipeline(res.compute_pipeline) else {
            return;
        };

        // Create bind group fresh each frame (physics buffers swap each frame)
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group_layout: BindGroupLayout = bind_group_layout.clone().into();

        let bind_group = render_device.create_bind_group(
            Some("particle_to_instance_bind_group"),
            &bind_group_layout,
            &[
                BindGroupEntry {
                    binding: 0,
                    resource: physics_buffers.current_output().as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: res.instance_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: res.params_buffer.as_entire_binding(),
                },
            ],
        );

        // Run compute shader
        let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("particle_to_instance_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("particle_to_instance_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (total + 63) / 64;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        render_queue.submit([encoder.finish()]);
        instance_len = total;
    } else {
        // ── CPU mode: upload instance data directly ──
        let Some(ref cpu_data) = cpu_data else {
            return;
        };
        if cpu_data.0.is_empty() {
            return;
        }

        instance_len = cpu_data.0.len() as u32;
        let bytes = bytemuck::cast_slice(&cpu_data.0);
        render_queue.write_buffer(&res.instance_buffer, 0, bytes);
    }

    // Attach InstanceBuffer component to batch proxy entity in render world

    for entity in &batch_query {
        commands.entity(entity).insert(InstanceBuffer {
            buffer: res.instance_buffer.clone(),
            len: instance_len,
        });
    }
}

/// Queue particle batch entities into Transparent3d render phase
fn queue_particle_instances(
    draw_funcs: Res<DrawFunctions<Transparent3d>>,
    custom_pipeline: Option<Res<ParticleInstancePipeline>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<ParticleInstancePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<RenderMesh>>,
    mesh_instances: Res<RenderMeshInstances>,
    batch_query: Query<
        (Entity, &bevy::render::sync_world::MainEntity),
        With<ParticleBatchMarker>,
    >,
    mut phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    views: Query<(&ExtractedView, &Msaa)>,
) {
    let Some(ref custom_pipeline) = custom_pipeline else {
        return;
    };

    let draw_function = draw_funcs.read().id::<DrawParticleInstanced>();

    for (view, msaa) in &views {
        let Some(phase) = phases.get_mut(&view.retained_view_entity) else {
            continue;
        };

        let base_key = MeshPipelineKey::from_msaa_samples(msaa.samples())
            | MeshPipelineKey::from_hdr(view.hdr);

        let rangefinder = view.rangefinder3d();

        for (render_entity, main_entity) in &batch_query {
            let Some(mesh_instance) = mesh_instances.render_mesh_queue_data(*main_entity) else {
                continue;
            };
            let Some(mesh) = meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };

            let key =
                base_key | MeshPipelineKey::from_primitive_topology(mesh.primitive_topology());

            let Ok(pipeline_id) =
                pipelines.specialize(&pipeline_cache, custom_pipeline, key, &mesh.layout)
            else {
                continue;
            };

            // Sort distance を最小にして、他の透明オブジェクト（仕切り等）より先に描画。
            // Bevy の Transparent3d は距離の昇順でソートする（小さい値が先に描画）。
            // パーティクルはデプスバッファに書き込むので、後から描画される
            // 仕切りが正しくデプステストされ、粒子を適切に遮蔽できる。
            phase.add(Transparent3d {
                entity: (render_entity, *main_entity),
                pipeline: pipeline_id,
                draw_function,
                distance: f32::NEG_INFINITY,
                batch_range: 0..1,
                extra_index: PhaseItemExtraIndex::None,
                indexed: true,
            });
        }
    }
}
