//! CPU-only instance writer.
//! Builds `InstanceData` in Main World and uploads it in Render World.

use bevy::{
    prelude::*,
    render::{
        extract_resource::ExtractResourcePlugin,
        renderer::{RenderDevice, RenderQueue},
        Render, RenderApp, RenderSystems,
    },
};

use crate::physics::{ParticleSize, ParticleStore};

use super::{
    is_cpu_backend, normalized_instance_capacity, CpuInstanceData, InstanceBuffer, InstanceData,
    ParticleBatchMarker, RenderInstanceBufferResource, LARGE_PARTICLE_COLOR, SMALL_PARTICLE_COLOR,
};

pub struct InstanceCpuWriterPlugin;

impl Plugin for InstanceCpuWriterPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(CpuInstanceData::default())
            .add_plugins(ExtractResourcePlugin::<CpuInstanceData>::default())
            .add_systems(PostUpdate, build_cpu_instance_data.run_if(is_cpu_backend));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.add_systems(
            Render,
            upload_cpu_instances
                .in_set(RenderSystems::PrepareResources)
                .run_if(is_cpu_backend),
        );
    }
}

/// Build instance data from ParticleStore (CPU mode)
fn build_cpu_instance_data(store: Res<ParticleStore>, mut cpu_data: ResMut<CpuInstanceData>) {
    cpu_data.0.clear();
    for p in &store.particles {
        let color = match p.size {
            ParticleSize::Large => LARGE_PARTICLE_COLOR,
            ParticleSize::Small => SMALL_PARTICLE_COLOR,
        };
        cpu_data.0.push(InstanceData {
            pos_scale: [p.position.x, p.position.y, p.position.z, p.radius],
            color,
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn upload_cpu_instances(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    cpu_data: Option<Res<CpuInstanceData>>,
    instance_buffer: Option<ResMut<RenderInstanceBufferResource>>,
    batch_query: Query<Entity, With<ParticleBatchMarker>>,
) {
    let Some(cpu_data) = cpu_data else {
        return;
    };

    let instance_len = cpu_data.0.len() as u32;

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

    if instance_len > 0 {
        let bytes = bytemuck::cast_slice(&cpu_data.0);
        render_queue.write_buffer(&instance_buffer.buffer, 0, bytes);
    }

    for entity in &batch_query {
        commands.entity(entity).insert(InstanceBuffer {
            buffer: instance_buffer.buffer.clone(),
            len: instance_buffer.len,
        });
    }
}
