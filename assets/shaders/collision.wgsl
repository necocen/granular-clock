// 衝突検出・接触力計算シェーダー

#import granular_clock::physics_types

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

fn wall_surface(normal: vec3<f32>, overlap: f32, p: Particle) -> ContactResult {
    var result: ContactResult;
    result.force = vec3<f32>(0.0); result.torque = vec3<f32>(0.0);
    if (overlap <= 0.0) { return result; }
    let velocity = p.vel - vec3<f32>(0.0, params.container_velocity, 0.0);
    let log_e = log(max(params.wall_restitution, 0.01));
    let ratio = -log_e / sqrt(3.14159265 * 3.14159265 + log_e * log_e);
    let damping = 2.0 * ratio * sqrt(params.wall_stiffness / p.mass_inv);
    let fn_mag = max(params.wall_stiffness * overlap - damping * dot(velocity, normal), 0.0);
    let arm = -p.radius * normal;
    let surface_velocity = velocity + cross(p.omega, arm);
    let tangent = surface_velocity - dot(surface_velocity, normal) * normal;
    let speed = length(tangent);
    var friction = vec3<f32>(0.0);
    if (speed > 1e-8) { friction = -tangent / speed * min(params.wall_friction * fn_mag, speed * params.wall_damping); }
    result.force = fn_mag * normal + friction;
    result.torque = cross(arm, friction);
    return result;
}

fn compute_wall_contact(p: Particle) -> ContactResult {
    let lo = vec3<f32>(-params.container_half_x, -params.container_half_y + params.container_offset, -params.container_half_z);
    let hi = vec3<f32>(params.container_half_x, params.container_half_y + params.container_offset, params.container_half_z);
    let normals = array<vec3<f32>,6>(vec3<f32>(1.0,0.0,0.0), vec3<f32>(-1.0,0.0,0.0), vec3<f32>(0.0,1.0,0.0),vec3<f32>(0.0,-1.0,0.0),vec3<f32>(0.0,0.0,1.0),vec3<f32>(0.0,0.0,-1.0));
    let overlaps = array<f32,6>(lo.x+p.radius-p.pos.x,p.pos.x+p.radius-hi.x,lo.y+p.radius-p.pos.y,p.pos.y+p.radius-hi.y,lo.z+p.radius-p.pos.z,p.pos.z+p.radius-hi.z);
    var result: ContactResult;
    result.force=vec3<f32>(0.0); result.torque=vec3<f32>(0.0);
    for (var i=0u;i<6u;i++) { let f=wall_surface(normals[i],overlaps[i],p); result.force+=f.force;result.torque+=f.torque; }
    if (params.divider_height <= 0.0) { return result; }
    let dmin=vec3<f32>(-params.divider_thickness*0.5,lo.y,lo.z);
    let dmax=vec3<f32>(params.divider_thickness*0.5,lo.y+params.divider_height,hi.z);
    let delta=p.pos-clamp(p.pos,dmin,dmax);
    let distance=length(delta);
    var normal=vec3<f32>(0.0);
    var overlap=0.0;
    if (distance>1e-8) { normal=delta/distance; overlap=p.radius-distance; }
    else {
        let distances=array<f32,6>(p.pos.x-dmin.x,dmax.x-p.pos.x,p.pos.y-dmin.y,dmax.y-p.pos.y,p.pos.z-dmin.z,dmax.z-p.pos.z);
        var nearest=select(0u,1u,p.pos.x>=0.0);
        for(var i=0u;i<6u;i++) { if(distances[i]<distances[nearest]) { nearest=i; } }
        normal=-normals[nearest]; overlap=p.radius+distances[nearest];
    }
    let f=wall_surface(normal,overlap,p);result.force+=f.force;result.torque+=f.torque;
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
    let k_n = (4.0 / 3.0) * e_eff * sqrt(r_eff * overlap);

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

    // n は q -> p。CPU と同じ接触点の腕で相対速度とトルクを計算する。
    let arm_p = -p.radius * n;
    let arm_q = q.radius * n;
    let v_contact = rel_vel + cross(p.omega, arm_p) - cross(q.omega, arm_q);

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

    // 転がり抵抗
    let omega_rel = p.omega - q.omega;
    let omega_roll = omega_rel - dot(omega_rel, n) * n;
    let omega_roll_mag = length(omega_roll);
    let t_r_max = params.rolling_friction * f_n_elastic * r_eff;

    var rolling_torque = vec3<f32>(0.0);
    if (omega_roll_mag > 1e-10) {
        let omega_roll_dir = omega_roll / omega_roll_mag;
        let omega_char = 10.0; // rad/s
        let viscous_coeff = t_r_max / omega_char;
        let t_viscous = viscous_coeff * omega_roll_mag;
        let t_r = min(t_viscous, t_r_max);
        rolling_torque = -t_r * omega_roll_dir;
    }

    let force = f_n_total * n + f_t;
    let torque_p = cross(arm_p, f_t) + rolling_torque;
    let torque_q = cross(arm_q, -f_t) - rolling_torque;

    // ペア共通の倍率で力・トルクを縮小する。相手側のスレッドでも
    // 同じ倍率になり、大小粒子間でも作用反作用と摩擦の散逸を保つ。
    let max_accel = 1000.0;
    let max_force = max_accel / max(p.mass_inv, q.mass_inv);
    let max_torque_p = (max_accel / p.mass_inv) * p.radius;
    let max_torque_q = (max_accel / q.mass_inv) * q.radius;
    var scale = 1.0;
    let force_mag = length(force);
    if (force_mag > max_force) {
        scale = max_force / force_mag;
    }
    let torque_p_mag = length(torque_p);
    if (torque_p_mag > max_torque_p) {
        scale = min(scale, max_torque_p / torque_p_mag);
    }
    let torque_q_mag = length(torque_q);
    if (torque_q_mag > max_torque_q) {
        scale = min(scale, max_torque_q / torque_q_mag);
    }

    result.force = force * scale;
    result.torque = torque_p * scale;
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
