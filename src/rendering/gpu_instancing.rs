//! Instanced Rendering for particles.
//! This plugin is rendering-only: it queues and draws from an already prepared instance buffer.

use bevy::{
    camera::visibility::NoFrustumCulling,
    core_pipeline::core_3d::Transparent3d,
    ecs::system::{SystemParamItem, lifetimeless::*},
    mesh::VertexBufferLayout,
    pbr::{
        MeshPipeline, MeshPipelineKey, RenderMeshInstances, SetMeshBindGroup, SetMeshViewBindGroup,
        SetMeshViewBindingArrayBindGroup,
    },
    prelude::*,
    render::{
        Render, RenderApp, RenderStartup, RenderSystems,
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        mesh::{RenderMesh, RenderMeshBufferInfo, allocator::MeshAllocator},
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
            RenderCommandResult, SetItemPipeline, TrackedRenderPass, ViewSortedRenderPhases,
        },
        render_resource::*,
        view::ExtractedView,
    },
};

use crate::rendering::{InstanceBuffer, InstanceData, ParticleMeshes};

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
        if let Some(ref mut depth_stencil) = desc.depth_stencil {
            depth_stencil.depth_write_enabled = true;
        }
        if let Some(ref mut frag) = desc.fragment {
            for ref mut state in frag.targets.iter_mut().flatten() {
                state.blend = None;
            }
        }

        // Add per-instance vertex buffer layout
        desc.vertex.buffers.push(VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceData>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: vec![
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 3,
                },
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

        let Some(mesh_instance) = mesh_instances.render_mesh_queue_data(item.main_entity()) else {
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
                let Some(index_slice) = allocator.mesh_index_slice(&mesh_instance.mesh_asset_id)
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

pub struct GpuInstancingPlugin;

impl Plugin for GpuInstancingPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractComponentPlugin::<ParticleBatchMarker>::default())
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
                queue_particle_instances.in_set(RenderSystems::QueueMeshes),
            );
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

/// Queue particle batch entities into Transparent3d render phase
#[allow(clippy::too_many_arguments)]
fn queue_particle_instances(
    draw_funcs: Res<DrawFunctions<Transparent3d>>,
    custom_pipeline: Option<Res<ParticleInstancePipeline>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<ParticleInstancePipeline>>,
    pipeline_cache: Res<PipelineCache>,
    meshes: Res<RenderAssets<RenderMesh>>,
    mesh_instances: Res<RenderMeshInstances>,
    batch_query: Query<(Entity, &bevy::render::sync_world::MainEntity), With<ParticleBatchMarker>>,
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
