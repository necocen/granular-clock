// 空間ハッシュキー生成シェーダー

struct Particle {
    pos: vec3<f32>,
    radius: f32,
    vel: vec3<f32>,
    mass_inv: f32,
    omega: vec3<f32>,
    inertia_inv: f32,
    size_flag: u32,
    _pad: array<u32, 3>,
}

struct Params {
    dt: f32,
    gravity: f32,
    cell_size: f32,
    grid_dim: u32,
    world_half: vec3<f32>,
    num_particles: u32,
    youngs_modulus: f32,
    poisson_ratio: f32,
    restitution: f32,
    friction: f32,
    container_offset: f32,
    divider_height: f32,
    container_half_x: f32,
    container_half_y: f32,
    container_half_z: f32,
    divider_thickness: f32,
    rolling_friction: f32,
    _pad: f32,
}

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
