// Bitonic Sort シェーダー
// (key, particle_id) ペアをソート

#import granular_clock::physics_types::{Particle, Params}

struct SortParams {
    j: u32,
    k: u32,
    n: u32,
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
    if (id >= sort_params.n) {
        return;
    }

    // 標準 bitonic compare-exchange:
    // partner = id XOR j, direction は (id & k) で決まる
    let partner = id ^ sort_params.j;
    if (partner >= sort_params.n || partner <= id) {
        return;
    }

    let key_a = keys[id];
    let key_b = keys[partner];
    let id_a = particle_ids[id];
    let id_b = particle_ids[partner];

    let ascending = (id & sort_params.k) == 0u;
    let should_swap = (ascending && key_a > key_b) || (!ascending && key_a < key_b);

    if (should_swap) {
        keys[id] = key_b;
        keys[partner] = key_a;
        particle_ids[id] = id_b;
        particle_ids[partner] = id_a;
    }
}
