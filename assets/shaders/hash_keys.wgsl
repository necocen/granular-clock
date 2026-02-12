// 空間ハッシュキー生成シェーダー

#import granular_clock::physics_types::{Particle, Params}

@group(0) @binding(0) var<uniform> params: Params;
@group(1) @binding(0) var<storage, read> particles_in: array<Particle>;
@group(2) @binding(0) var<storage, read_write> keys: array<u32>;
@group(2) @binding(1) var<storage, read_write> particle_ids: array<u32>;

fn hash_cell(cell: vec3<i32>) -> u32 {
    let dim = i32(params.grid_dim);
    let half = dim / 2;
    let shifted = cell + vec3<i32>(half, half, half);
    let c = vec3<i32>(
        clamp(shifted.x, 0, dim - 1),
        clamp(shifted.y, 0, dim - 1),
        clamp(shifted.z, 0, dim - 1),
    );
    return u32((c.z * dim + c.y) * dim + c.x);
}

@compute @workgroup_size(64)
fn build_keys(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id < params.num_particles) {
        let p = particles_in[id];

        // CPU 実装と同じく原点基準でセル化し、インデックス化時のみ中心シフトする
        let cell = vec3<i32>(floor(p.pos / params.cell_size));

        let key = hash_cell(cell);
        keys[id] = key;
        particle_ids[id] = id;
    } else {
        // 2 のべき乗パディング領域は番兵値で埋める（ソート時に末尾へ送る）
        keys[id] = 0xffffffffu;
        particle_ids[id] = 0xffffffffu;
    }
}
