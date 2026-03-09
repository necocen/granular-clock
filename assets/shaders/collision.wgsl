// 衝突検出・接触力計算シェーダー

#import granular_clock::physics_types::{Particle, Params, grid_cell_from_pos, grid_hash_cell, grid_cell_in_bounds}

/// 接触力とトルクの結果
struct ContactResult {
    force: vec3<f32>,
    torque: vec3<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(1) @binding(0) var<storage, read> particles_in: array<Particle>;
@group(2) @binding(0) var<storage, read_write> keys: array<u32>;
@group(2) @binding(1) var<storage, read_write> particle_ids: array<u32>;
@group(2) @binding(2) var<storage, read_write> cell_ranges: array<vec2<u32>>;
@group(3) @binding(0) var<storage, read_write> forces: array<vec4<f32>>;
@group(3) @binding(1) var<storage, read_write> torques: array<vec4<f32>>;

fn compute_wall_normal_force(overlap_in: f32, v_n: f32, radius: f32, mass_inv: f32) -> f32 {
    if (overlap_in <= 0.0) {
        return 0.0;
    }

    let overlap = min(overlap_in, radius);
    let stiffness = params.wall_stiffness;
    let mass = 1.0 / mass_inv;

    let f_spring = stiffness * overlap;
    let ln_e = log(max(params.wall_restitution, 0.01));
    let damping_ratio = -ln_e / sqrt(3.14159265 * 3.14159265 + ln_e * ln_e);
    let c_crit = 2.0 * sqrt(stiffness * mass);
    let damping_coeff = damping_ratio * c_crit;

    var f_damp = 0.0;
    if (v_n < 0.0) {
        f_damp = -damping_coeff * v_n;
    }

    return max(f_spring + f_damp, 0.0);
}

// 仕切り中心で法線方向が決められない場合のタイブレーク（CPU実装と同じ式）。
fn divider_tiebreak_sign(pos: vec3<f32>) -> f32 {
    let qy = i32(floor(pos.y * 1024.0));
    let qz = i32(floor(pos.z * 1024.0));
    let seed = (bitcast<u32>(qy) * 0x9E3779B1u)
        ^ (bitcast<u32>(qz) * 0x85EBCA6Bu)
        ^ 0xC2B2AE35u;
    return select(-1.0, 1.0, (seed & 1u) == 1u);
}

fn compute_wall_contact(p: Particle) -> ContactResult {
    var result: ContactResult;
    result.force = vec3<f32>(0.0);
    result.torque = vec3<f32>(0.0);

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

    // 床
    let floor_y = box_min.y;
    let floor_overlap = floor_y + p.radius - p.pos.y;
    let contact_threshold = 0.001;
    let is_floor_contact = floor_overlap > -contact_threshold;
    let mass = 1.0 / p.mass_inv;

    if (is_floor_contact) {
        let effective_overlap = max(floor_overlap, 0.0);
        let gravity_mag = 9.81;
        var floor_normal_force = 0.0;

        if (effective_overlap > 0.0) {
            let f_n = compute_wall_normal_force(effective_overlap, p.vel.y, p.radius, p.mass_inv);
            floor_normal_force += f_n;
            result.force.y += f_n;
        }

        if (p.vel.y < 0.0 && p.vel.y > -0.01) {
            let damping_force = -p.vel.y * mass * 100.0;
            floor_normal_force += damping_force;
            result.force.y += damping_force;
        }

        let support_eps = 1e-6;
        if (floor_overlap >= -support_eps && p.vel.y <= 0.0) {
            let support_force = max(mass * gravity_mag - floor_normal_force, 0.0);
            if (support_force > 0.0) {
                floor_normal_force += support_force;
                result.force.y += support_force;
            }
        }

        var normal_force = floor_normal_force;
        if (floor_overlap >= -support_eps) {
            normal_force = max(normal_force, mass * gravity_mag);
        }

        let contact_point = vec3<f32>(0.0, -p.radius, 0.0);
        let v_t = vec3<f32>(p.vel.x, 0.0, p.vel.z) + cross(p.omega, contact_point);
        let v_t_mag = length(v_t);
        if (v_t_mag > 1e-8) {
            let f_t_max = params.wall_friction * normal_force;
            let viscous_friction = v_t_mag * params.wall_damping * 2.0;
            let stick_friction = f_t_max * 0.1;
            let f_t = min(f_t_max, viscous_friction + stick_friction);
            let friction_force = -f_t * (v_t / v_t_mag);
            result.force += friction_force;
            result.torque += cross(contact_point, friction_force);
        }
    }

    // 天井
    let ceiling_overlap = p.pos.y + p.radius - box_max.y;
    if (ceiling_overlap > 0.0) {
        let f_n = compute_wall_normal_force(ceiling_overlap, -p.vel.y, p.radius, p.mass_inv);
        result.force.y -= f_n;
    }

    // 左壁 (-X)
    let left_overlap = box_min.x + p.radius - p.pos.x;
    if (left_overlap > 0.0) {
        let f_n = compute_wall_normal_force(left_overlap, p.vel.x, p.radius, p.mass_inv);
        result.force.x += f_n;
    }

    // 右壁 (+X)
    let right_overlap = p.pos.x + p.radius - box_max.x;
    if (right_overlap > 0.0) {
        let f_n = compute_wall_normal_force(right_overlap, -p.vel.x, p.radius, p.mass_inv);
        result.force.x -= f_n;
    }

    // 前壁 (-Z)
    let front_overlap = box_min.z + p.radius - p.pos.z;
    if (front_overlap > 0.0) {
        let f_n = compute_wall_normal_force(front_overlap, p.vel.z, p.radius, p.mass_inv);
        result.force.z += f_n;
    }

    // 後壁 (+Z)
    let back_overlap = p.pos.z + p.radius - box_max.z;
    if (back_overlap > 0.0) {
        let f_n = compute_wall_normal_force(back_overlap, -p.vel.z, p.radius, p.mass_inv);
        result.force.z -= f_n;
    }

    // 仕切りとの接触（壁よりソフトな制約）
    let divider_top = floor_y + params.divider_height;
    if (p.pos.y - p.radius < divider_top) {
        let half_thickness = params.divider_thickness * 0.5;
        let dist_from_center = abs(p.pos.x);
        let dist_to_face = dist_from_center - half_thickness;
        let raw_overlap = p.radius - dist_to_face;

        if (raw_overlap > 0.0) {
            // 深いめり込みでも強い押し出しになりすぎないよう制限
            let max_overlap = p.radius * 0.25;
            let overlap = min(raw_overlap, max_overlap);

            // x=0 付近では速度方向を優先して法線を決め、左右対称性を保つ
            var sign = 1.0;
            if (dist_from_center > 1e-8) {
                sign = select(-1.0, 1.0, p.pos.x > 0.0);
            } else if (abs(p.vel.x) > 1e-8) {
                sign = select(1.0, -1.0, p.vel.x > 0.0);
            } else {
                sign = divider_tiebreak_sign(p.pos);
            }

            let v_n = sign * p.vel.x;
            let divider_force_scale = 0.35;
            let f_n = divider_force_scale
                * compute_wall_normal_force(overlap, v_n, p.radius, p.mass_inv);
            result.force.x += sign * f_n;

            // 接線摩擦（Y-Z平面）も弱める
            if (f_n > 0.0) {
                let v_t = vec3<f32>(0.0, p.vel.y, p.vel.z);
                let v_t_mag = length(v_t);
                if (v_t_mag > 1e-6) {
                    let f_t_max = params.wall_friction * 0.5 * f_n;
                    let f_t = min(f_t_max, v_t_mag * params.wall_damping * 0.25);
                    result.force -= f_t * (v_t / v_t_mag);
                }
            }
        }
    }

    if (any(result.force != result.force)) {
        result.force = vec3<f32>(0.0);
    }
    if (any(result.torque != result.torque)) {
        result.torque = vec3<f32>(0.0);
    }

    return result;
}

fn compute_contact(p: Particle, q: Particle) -> ContactResult {
    var result: ContactResult;
    result.force = vec3<f32>(0.0);
    result.torque = vec3<f32>(0.0);

    let delta = p.pos - q.pos;
    let dist_sq = dot(delta, delta);
    let r_sum = p.radius + q.radius;

    if (dist_sq >= r_sum * r_sum) {
        return result;
    }

    let dist = sqrt(dist_sq);
    if (dist < 1e-10) {
        return result;
    }

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
        let restitution = max(params.restitution, 0.01);
        let log_e = log(restitution);
        let beta = -log_e / sqrt(3.14159265 * 3.14159265 + log_e * log_e);
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

    // CPU 実装と同様に接触力・トルクをクランプして数値暴走を防ぐ
    let max_accel = 1000.0;
    let max_force = max_accel / p.mass_inv;
    let force_mag = length(result.force);
    if (force_mag > max_force && force_mag > 1e-10) {
        result.force *= max_force / force_mag;
    }

    let max_torque = max_force * p.radius;
    let torque_mag = length(result.torque);
    if (torque_mag > max_torque && torque_mag > 1e-10) {
        result.torque *= max_torque / torque_mag;
    }

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

    let cell = grid_cell_from_pos(p.pos, params.cell_size);

    // 27セル近傍探索
    for (var dz: i32 = -1; dz <= 1; dz++) {
        for (var dy: i32 = -1; dy <= 1; dy++) {
            for (var dx: i32 = -1; dx <= 1; dx++) {
                let nc = cell + vec3<i32>(dx, dy, dz);

                // 範囲外チェック
                if (!grid_cell_in_bounds(nc, params.grid_dim)) {
                    continue;
                }

                let cell_key = grid_hash_cell(nc, params.grid_dim);
                let range = cell_ranges[cell_key];
                let start = min(range.x, params.num_particles);
                let end = min(range.y, params.num_particles);
                if (end <= start) {
                    continue;
                }

                for (var j: u32 = start; j < end; j++) {
                    // セル範囲が壊れた場合でも他セル粒子を誤加算しないようキー一致を確認
                    if (keys[j] != cell_key) {
                        continue;
                    }
                    let other_id = particle_ids[j];
                    if (other_id >= params.num_particles || other_id == id) {
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

    // CPU 実装に合わせて、粒子間力の加算後に壁接触力を加算する。
    let wall = compute_wall_contact(p);
    total_force += wall.force;
    total_torque += wall.torque;

    // 非有限値ガード（isFinite の実装差を避けるため NaN/巨大値で判定）
    let invalid_force = any(total_force != total_force) || any(abs(total_force) > vec3<f32>(1e10));
    if (invalid_force) {
        forces[id] = vec4<f32>(0.0);
    } else {
        forces[id] = vec4<f32>(total_force, 0.0);
    }

    let invalid_torque =
        any(total_torque != total_torque) || any(abs(total_torque) > vec3<f32>(1e10));
    if (invalid_torque) {
        torques[id] = vec4<f32>(0.0);
    } else {
        torques[id] = vec4<f32>(total_torque, 0.0);
    }
}
