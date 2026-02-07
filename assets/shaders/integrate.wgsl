// 積分シェーダー (Velocity Verlet)

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
@group(0) @binding(7) var<storage, read_write> torques: array<vec4<f32>>;

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= params.num_particles) {
        return;
    }

    var p = particles_in[id];
    let f = forces[id].xyz;
    let t = torques[id].xyz;

    // NaN ガード: 入力が NaN なら前の状態をそのまま出力
    if (any(p.pos != p.pos) || any(p.vel != p.vel) || any(p.omega != p.omega) || any(f != f) || any(t != t)) {
        particles_out[id] = p;
        return;
    }

    // 加速度
    let gravity_force = vec3<f32>(0.0, params.gravity / p.mass_inv, 0.0);
    let a = (f + gravity_force) * p.mass_inv;

    // 角加速度
    let alpha = t * p.inertia_inv;

    // Velocity Verlet 積分
    p.vel += a * params.dt;
    p.pos += p.vel * params.dt;

    // 角速度の積分
    p.omega += alpha * params.dt;

    // 速度クランプ（数値不安定の防止）
    let max_vel = 10.0; // m/s
    let vel_mag = length(p.vel);
    if (vel_mag > max_vel) {
        p.vel = p.vel * (max_vel / vel_mag);
    }

    // 角速度クランプ
    let max_omega = 100.0; // rad/s
    let omega_mag = length(p.omega);
    if (omega_mag > max_omega) {
        p.omega = p.omega * (max_omega / omega_mag);
    }

    // 壁との衝突（簡易版）
    let floor_y = -params.container_half_y + params.container_offset;
    let ceiling_y = params.container_half_y + params.container_offset;

    // 床
    if (p.pos.y - p.radius < floor_y) {
        p.pos.y = floor_y + p.radius;
        p.vel.y = -p.vel.y * params.restitution;
    }

    // 天井
    if (p.pos.y + p.radius > ceiling_y) {
        p.pos.y = ceiling_y - p.radius;
        p.vel.y = -p.vel.y * params.restitution;
    }

    // 左右の壁
    if (p.pos.x - p.radius < -params.container_half_x) {
        p.pos.x = -params.container_half_x + p.radius;
        p.vel.x = -p.vel.x * params.restitution;
    }
    if (p.pos.x + p.radius > params.container_half_x) {
        p.pos.x = params.container_half_x - p.radius;
        p.vel.x = -p.vel.x * params.restitution;
    }

    // 前後の壁
    if (p.pos.z - p.radius < -params.container_half_z) {
        p.pos.z = -params.container_half_z + p.radius;
        p.vel.z = -p.vel.z * params.restitution;
    }
    if (p.pos.z + p.radius > params.container_half_z) {
        p.pos.z = params.container_half_z - p.radius;
        p.vel.z = -p.vel.z * params.restitution;
    }

    // 仕切り（中央のX=0付近）— 位置ベースの押し出し（CPU版と同等）
    let divider_top = floor_y + params.divider_height;
    let half_thickness = params.divider_thickness * 0.5;

    if (p.pos.y - p.radius < divider_top) {
        let particle_left = p.pos.x - p.radius;
        let particle_right = p.pos.x + p.radius;

        // 粒子が仕切りと重なっているかチェック
        if (particle_left < half_thickness && particle_right > -half_thickness) {
            if (p.pos.x >= 0.0) {
                // 右側に押し出す
                let overlap = half_thickness - particle_left;
                if (overlap > 0.0) {
                    p.pos.x += overlap;
                    p.vel.x = abs(p.vel.x) * params.restitution;
                }
            } else {
                // 左側に押し出す
                let overlap = particle_right - (-half_thickness);
                if (overlap > 0.0) {
                    p.pos.x -= overlap;
                    p.vel.x = -abs(p.vel.x) * params.restitution;
                }
            }
        }
    }

    particles_out[id] = p;
}
