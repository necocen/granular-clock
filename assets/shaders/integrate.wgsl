// 積分シェーダー (Velocity Verlet)

#import granular_clock::physics_types::{Particle, Params}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particles_out: array<Particle>;
@group(0) @binding(3) var<storage, read_write> keys: array<u32>;
@group(0) @binding(4) var<storage, read_write> particle_ids: array<u32>;
@group(0) @binding(5) var<storage, read_write> cell_ranges: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read_write> forces: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> torques: array<vec4<f32>>;

fn clamp_to_container(p: ptr<function, Particle>) {
    let box_min = vec3<f32>(
        -params.container_half_x,
        -params.container_half_y + params.container_offset,
        -params.container_half_z,
    );
    let box_max = vec3<f32>(
        params.container_half_x,
        params.container_half_y + params.container_offset,
        params.container_half_z,
    );

    // X軸
    let x_min = box_min.x + (*p).radius;
    let x_max = box_max.x - (*p).radius;
    if ((*p).pos.x < x_min) {
        (*p).pos.x = x_min;
        (*p).vel.x = max((*p).vel.x, 0.0);
    } else if ((*p).pos.x > x_max) {
        (*p).pos.x = x_max;
        (*p).vel.x = min((*p).vel.x, 0.0);
    }

    // Y軸
    let y_min = box_min.y + (*p).radius;
    let y_max = box_max.y - (*p).radius;
    if ((*p).pos.y < y_min) {
        (*p).pos.y = y_min;
        (*p).vel.y = max((*p).vel.y, 0.0);
    } else if ((*p).pos.y > y_max) {
        (*p).pos.y = y_max;
        (*p).vel.y = min((*p).vel.y, 0.0);
    }

    // Z軸
    let z_min = box_min.z + (*p).radius;
    let z_max = box_max.z - (*p).radius;
    if ((*p).pos.z < z_min) {
        (*p).pos.z = z_min;
        (*p).vel.z = max((*p).vel.z, 0.0);
    } else if ((*p).pos.z > z_max) {
        (*p).pos.z = z_max;
        (*p).vel.z = min((*p).vel.z, 0.0);
    }
}

fn clamp_velocity(p: ptr<function, Particle>) {
    // CPU 側と同じ制限値
    let max_vel = 10.0;
    let max_omega = 100.0;

    let vel_mag = length((*p).vel);
    if (vel_mag > max_vel) {
        (*p).vel *= max_vel / vel_mag;
    }

    let omega_mag = length((*p).omega);
    if (omega_mag > max_omega) {
        (*p).omega *= max_omega / omega_mag;
    }
}

@compute @workgroup_size(64)
fn integrate_first_half(@builtin(global_invocation_id) gid: vec3<u32>) {
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

    let a = f * p.mass_inv + vec3<f32>(0.0, params.gravity, 0.0);
    let alpha = t * p.inertia_inv;

    // Velocity Verlet 前半:
    // 速度を半更新し、位置を全更新
    p.vel += 0.5 * a * params.dt;
    p.omega += 0.5 * alpha * params.dt;
    p.pos += p.vel * params.dt;

    clamp_to_container(&p);
    clamp_velocity(&p);

    particles_out[id] = p;
}

@compute @workgroup_size(64)
fn integrate_second_half(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= params.num_particles) {
        return;
    }

    var p = particles_in[id];
    let f = forces[id].xyz;
    let t = torques[id].xyz;

    if (any(p.pos != p.pos) || any(p.vel != p.vel) || any(p.omega != p.omega) || any(f != f) || any(t != t)) {
        particles_out[id] = p;
        return;
    }

    let a = f * p.mass_inv + vec3<f32>(0.0, params.gravity, 0.0);
    let alpha = t * p.inertia_inv;

    // Velocity Verlet 後半:
    // 新しい力で速度を半更新
    p.vel += 0.5 * a * params.dt;
    p.omega += 0.5 * alpha * params.dt;

    clamp_to_container(&p);
    clamp_velocity(&p);

    particles_out[id] = p;
}
