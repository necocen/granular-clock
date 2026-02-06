// Bitonic Sort シェーダー
// (key, particle_id) ペアをソート

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

struct SortParams {
    step_size: u32,
    pass_offset: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particles_out: array<Particle>;
@group(0) @binding(3) var<storage, read_write> keys: array<u32>;
@group(0) @binding(4) var<storage, read_write> particle_ids: array<u32>;
@group(0) @binding(5) var<storage, read_write> cell_ranges: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read_write> forces: array<vec4<f32>>;

var<push_constant> sort_params: SortParams;

@compute @workgroup_size(256)
fn bitonic_sort_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= params.num_particles) {
        return;
    }

    let step_size = sort_params.step_size;
    let pass_offset = sort_params.pass_offset;

    // Bitonic sort のステップ
    let block_size = step_size * 2u;
    let block_id = id / block_size;
    let local_id = id % block_size;

    // 比較・交換
    let ascending = (block_id % 2u) == 0u;

    let half = step_size;
    if (local_id < half) {
        let partner = id + half;
        if (partner < params.num_particles) {
            let key_a = keys[id];
            let key_b = keys[partner];
            let id_a = particle_ids[id];
            let id_b = particle_ids[partner];

            let should_swap = (ascending && key_a > key_b) || (!ascending && key_a < key_b);

            if (should_swap) {
                keys[id] = key_b;
                keys[partner] = key_a;
                particle_ids[id] = id_b;
                particle_ids[partner] = id_a;
            }
        }
    }
}
