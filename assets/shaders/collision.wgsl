// 衝突検出・接触力計算シェーダー

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

/// 接触力とトルクの結果
struct ContactResult {
    force: vec3<f32>,
    torque: vec3<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> particles_in: array<Particle>;
@group(0) @binding(2) var<storage, read_write> particles_out: array<Particle>;
@group(0) @binding(3) var<storage, read_write> keys: array<u32>;
@group(0) @binding(4) var<storage, read_write> particle_ids: array<u32>;
@group(0) @binding(5) var<storage, read_write> cell_ranges: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read_write> forces: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> torques: array<vec4<f32>>;

fn hash_cell(cell: vec3<i32>) -> u32 {
    let dim = i32(params.grid_dim);
    let c = vec3<i32>(
        clamp(cell.x, 0, dim - 1),
        clamp(cell.y, 0, dim - 1),
        clamp(cell.z, 0, dim - 1)
    );
    return u32((c.z * dim + c.y) * dim + c.x);
}

fn get_cell(pos: vec3<f32>) -> vec3<i32> {
    let world_half = vec3<f32>(params.container_half_x, params.container_half_y, params.container_half_z);
    let normalized_pos = (pos + world_half) / params.cell_size;
    return vec3<i32>(floor(normalized_pos));
}

fn compute_contact(p: Particle, q: Particle) -> ContactResult {
    var result: ContactResult;
    result.force = vec3<f32>(0.0);
    result.torque = vec3<f32>(0.0);

    let delta = p.pos - q.pos;
    let dist_sq = dot(delta, delta);
    let r_sum = p.radius + q.radius;

    if (dist_sq >= r_sum * r_sum || dist_sq < 1e-10) {
        return result;
    }

    let dist = sqrt(dist_sq);
    var overlap = r_sum - dist;

    // sqrt の浮動小数点精度により overlap が負になるケースをガード
    if (overlap <= 0.0) {
        return result;
    }

    let n = delta / dist;

    // 有効半径
    let r_eff = (p.radius * q.radius) / (p.radius + q.radius);

    // オーバーラップを制限して数値安定性を確保（CPU 版と同じ）
    let max_overlap = r_eff * 0.2;
    overlap = min(overlap, max_overlap);

    // 有効ヤング率
    let e_eff = params.youngs_modulus / (2.0 * (1.0 - params.poisson_ratio * params.poisson_ratio));

    // 法線剛性 (Hertz): k_n = (4/3) * e_eff * sqrt(r_eff * overlap)
    let k_n = (4.0 / 3.0) * e_eff * sqrt(r_eff) * sqrt(overlap);

    // 法線弾性力: f_n_elastic = k_n * overlap
    let f_n_elastic = k_n * overlap;

    // 相対速度
    let rel_vel = p.vel - q.vel;
    let v_n = dot(rel_vel, n);

    // 法線方向の減衰（接近時のみ）
    var f_n_damping = 0.0;
    if (v_n < 0.0) {
        let m_eff = 1.0 / (p.mass_inv + q.mass_inv);
        let beta = -log(params.restitution) / sqrt(3.14159265 * 3.14159265 + log(params.restitution) * log(params.restitution));
        f_n_damping = 2.0 * beta * sqrt(k_n * m_eff) * (-v_n);
    }

    // 法線力合計（引力にならないようクランプ）
    let f_n_total = max(f_n_elastic + f_n_damping, 0.0);

    // 接触点での相対速度（角速度を考慮）— CPU版 contact.rs:168 と同等
    let v_contact = rel_vel - v_n * n
        - cross(p.omega, p.radius * n)
        + cross(q.omega, q.radius * n);

    // 接線速度
    let v_t = v_contact - dot(v_contact, n) * n;
    let v_t_mag = length(v_t);

    // 接線力（正則化クーロン摩擦）
    // 摩擦限界は弾性力のみ基準（CPU版と同じ。減衰力を含めるとエネルギー増加の原因になる）
    var f_t = vec3<f32>(0.0);
    if (v_t_mag > 1e-10) {
        let v_t_dir = v_t / v_t_mag;
        let f_t_max = params.friction * f_n_elastic;
        let v_char = 0.1; // 特性速度
        let viscous_coeff = f_t_max / v_char;
        let f_t_mag_clamped = min(viscous_coeff * v_t_mag, f_t_max);
        f_t = -f_t_mag_clamped * v_t_dir;
    }

    // 摩擦トルク — CPU版 contact.rs:214
    var torque = cross(p.radius * n, f_t);

    // 転がり抵抗 — CPU版 contact.rs:195-210
    let omega_rel = p.omega - q.omega;
    let omega_roll = omega_rel - dot(omega_rel, n) * n;
    let omega_roll_mag = length(omega_roll);
    let t_r_max = params.rolling_friction * f_n_elastic * r_eff;

    if (omega_roll_mag > 1e-10) {
        let omega_roll_dir = omega_roll / omega_roll_mag;
        let omega_char = 10.0; // rad/s
        let viscous_coeff = t_r_max / omega_char;
        let t_viscous = viscous_coeff * omega_roll_mag;
        let t_r = min(t_viscous, t_r_max);
        torque -= t_r * omega_roll_dir;
    }

    result.force = f_n_total * n + f_t;
    result.torque = torque;
    return result;
}

@compute @workgroup_size(64)
fn collision_response(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if (id >= params.num_particles) {
        return;
    }

    let p = particles_in[id];

    // NaN ガード: 入力が NaN なら力をゼロに
    if (any(p.pos != p.pos) || any(p.vel != p.vel)) {
        forces[id] = vec4<f32>(0.0);
        return;
    }

    var total_force = vec3<f32>(0.0);
    var total_torque = vec3<f32>(0.0);

    let cell = get_cell(p.pos);
    let dim = i32(params.grid_dim);

    // 27セル近傍探索
    for (var dz: i32 = -1; dz <= 1; dz++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dx: i32 = -1; dx <= 1; dx++) {
                let nc = cell + vec3<i32>(dx, dy, dz);

                // 範囲外チェック
                if (nc.x < 0 || nc.x >= dim || nc.y < 0 || nc.y >= dim || nc.z < 0 || nc.z >= dim) {
                    continue;
                }

                let cell_key = hash_cell(nc);
                let range = cell_ranges[cell_key];

                for (var j: u32 = range.x; j < range.y; j++) {
                    let other_id = particle_ids[j];
                    if (other_id == id) {
                        continue;
                    }

                    let q = particles_in[other_id];
                    let contact = compute_contact(p, q);
                    total_force += contact.force;
                    total_torque += contact.torque;
                }
            }
        }
    }

    // NaN ガード: 結果が NaN なら力・トルクをゼロに
    if (any(total_force != total_force)) {
        forces[id] = vec4<f32>(0.0);
    } else {
        forces[id] = vec4<f32>(total_force, 0.0);
    }

    if (any(total_torque != total_torque)) {
        torques[id] = vec4<f32>(0.0);
    } else {
        torques[id] = vec4<f32>(total_torque, 0.0);
    }
}
