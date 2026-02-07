// Particle to Instance conversion shader
// 粒子バッファからインスタンス描画用のデータを生成
//
// 粒子は spawn 順で格納:
// - [0, num_large) = 大粒子
// - [num_large, num_particles) = 小粒子

#import granular_clock::physics_types::Particle

struct InstanceData {
    pos_scale: vec4<f32>,  // xyz = world position, w = radius (uniform scale)
    color: vec4<f32>,      // RGBA
}

struct Params {
    num_particles: u32,
    num_large: u32,
    // 8 bytes padding (WGSL auto-aligns vec4 to 16 bytes)
    large_color: vec4<f32>,
    small_color: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> instances: array<InstanceData>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;

    if (id >= params.num_particles) {
        return;
    }

    let p = particles[id];

    var inst: InstanceData;
    inst.pos_scale = vec4<f32>(p.pos, p.radius);

    if (id < params.num_large) {
        inst.color = params.large_color;
    } else {
        inst.color = params.small_color;
    }

    instances[id] = inst;
}
