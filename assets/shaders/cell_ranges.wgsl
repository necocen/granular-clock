// セル範囲構築シェーダー
// ソート済みの keys 配列から各セルの開始・終了インデックスを計算

#import granular_clock::physics_types::{Particle, Params}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particles_out: array<Particle>;
@group(0) @binding(3) var<storage, read_write> keys: array<u32>;
@group(0) @binding(4) var<storage, read_write> particle_ids: array<u32>;
@group(0) @binding(5) var<storage, read_write> cell_ranges: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read_write> forces: array<vec4<f32>>;

@compute @workgroup_size(64)
fn build_cell_ranges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= params.num_particles) {
        return;
    }

    let key = keys[id];

    // 最初の粒子または前の粒子と異なるキーの場合、セルの開始
    if (id == 0u) {
        cell_ranges[key].x = 0u;
    } else {
        let prev_key = keys[id - 1u];
        if (key != prev_key) {
            // 前のセルの終了
            cell_ranges[prev_key].y = id;
            // 新しいセルの開始
            cell_ranges[key].x = id;
        }
    }

    // 最後の粒子の場合、現在のセルの終了をマーク
    if (id == params.num_particles - 1u) {
        cell_ranges[key].y = params.num_particles;
    }
}
