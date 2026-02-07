// 空間ハッシュキー生成シェーダー

#import granular_clock::physics_types::{Particle, Params}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particles_out: array<Particle>;
@group(0) @binding(3) var<storage, read_write> keys: array<u32>;
@group(0) @binding(4) var<storage, read_write> particle_ids: array<u32>;
@group(0) @binding(5) var<storage, read_write> cell_ranges: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read_write> forces: array<vec4<f32>>;

fn hash_cell(cell: vec3<u32>) -> u32 {
    let dim = params.grid_dim;
    return (cell.z * dim + cell.y) * dim + cell.x;
}

@compute @workgroup_size(64)
fn build_keys(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= params.num_particles) {
        return;
    }

    let p = particles_in[id];

    // 位置からセル座標を計算
    let world_half = vec3<f32>(params.container_half_x, params.container_half_y, params.container_half_z);
    let normalized_pos = (p.pos + world_half) / params.cell_size;
    let cell = vec3<u32>(
        clamp(u32(floor(normalized_pos.x)), 0u, params.grid_dim - 1u),
        clamp(u32(floor(normalized_pos.y)), 0u, params.grid_dim - 1u),
        clamp(u32(floor(normalized_pos.z)), 0u, params.grid_dim - 1u)
    );

    let key = hash_cell(cell);

    keys[id] = key;
    particle_ids[id] = id;
}
