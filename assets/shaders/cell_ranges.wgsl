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

    let num_cells = params.grid_dim * params.grid_dim * params.grid_dim;
    let key = keys[id];

    // 各セルの開始インデックス thread のみが start/end を一括で書く。
    // x/y を別 thread から更新すると競合する可能性があるため避ける。
    let is_cell_start = id == 0u || key != keys[id - 1u];
    if (!is_cell_start || key >= num_cells) {
        return;
    }

    var end = id + 1u;
    while (end < params.num_particles && keys[end] == key) {
        end += 1u;
    }

    cell_ranges[key] = vec2<u32>(id, end);
}
