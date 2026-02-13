use bevy::{
    prelude::*,
    render::{
        extract_resource::ExtractResource,
        render_resource::{Buffer, BufferDescriptor, BufferUsages},
        renderer::RenderDevice,
    },
};
use bytemuck::{Pod, Zeroable};

/// Instance data layout: position + scale + color (32 bytes)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct InstanceData {
    /// xyz = world position, w = radius (uniform scale)
    pub pos_scale: [f32; 4],
    /// RGBA color
    pub color: [f32; 4],
}

/// Shared particle colors used by CPU/GPU instance writers.
pub const LARGE_PARTICLE_COLOR: [f32; 4] = [0.8, 0.2, 0.2, 1.0];
pub const SMALL_PARTICLE_COLOR: [f32; 4] = [0.2, 0.2, 0.8, 1.0];

/// Instance buffer component attached to batch proxy entity in Render World
#[derive(Component)]
pub struct InstanceBuffer {
    pub buffer: Buffer,
    pub len: u32,
}

/// Shared instance buffer resource used by CPU/GPU writers
#[derive(Resource)]
pub struct RenderInstanceBufferResource {
    pub buffer: Buffer,
    pub capacity: u32,
    pub len: u32,
}

impl RenderInstanceBufferResource {
    pub fn new(render_device: &RenderDevice, capacity: u32) -> Self {
        let capacity = normalized_instance_capacity(capacity);
        let instance_size = std::mem::size_of::<InstanceData>() as u64;

        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("particle_instance_buffer"),
            size: instance_size * capacity as u64,
            usage: BufferUsages::VERTEX | BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            capacity,
            len: 0,
        }
    }
}

pub fn normalized_instance_capacity(required: u32) -> u32 {
    let min_capacity = required.max(2048);
    min_capacity
        .checked_next_power_of_two()
        .unwrap_or(min_capacity)
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
