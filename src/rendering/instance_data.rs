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

pub fn normalized_instance_capacity(required: u32) -> u32 {
    let min_capacity = required.max(2048);
    min_capacity
        .checked_next_power_of_two()
        .unwrap_or(min_capacity)
}
