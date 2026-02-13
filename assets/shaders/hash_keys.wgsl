// 空間ハッシュキー生成シェーダー

#import granular_clock::physics_types::{Particle, Params, grid_cell_from_pos, grid_hash_cell}

@group(0) @binding(0) var<uniform> params: Params;
@group(1) @binding(0) var<storage, read> particles_in: array<Particle>;
@group(2) @binding(0) var<storage, read_write> keys: array<u32>;
@group(2) @binding(1) var<storage, read_write> particle_ids: array<u32>;

@compute @workgroup_size(64)
fn build_keys(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id < params.num_particles) {
        let p = particles_in[id];

        // CPU 実装と同じく原点基準でセル化し、インデックス化時のみ中心シフトする
        let cell = grid_cell_from_pos(p.pos, params.cell_size);
        let key = grid_hash_cell(cell, params.grid_dim);
        keys[id] = key;
        particle_ids[id] = id;
    } else {
        // 2 のべき乗パディング領域は番兵値で埋める（ソート時に末尾へ送る）
        keys[id] = 0xffffffffu;
        particle_ids[id] = 0xffffffffu;
    }
}
