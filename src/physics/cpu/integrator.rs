use bevy::prelude::*;

/// Velocity Verlet積分の前半ステップ
/// 速度を半分更新し、位置を全更新
#[allow(clippy::too_many_arguments)]
pub fn integrate_first_half(
    pos: &mut Vec3,
    vel: &mut Vec3,
    omega: &mut Vec3,
    force: Vec3,
    torque: Vec3,
    mass: f32,
    inertia: f32,
    gravity: Vec3,
    dt: f32,
) {
    // 加速度
    let accel = force / mass + gravity;
    let alpha = torque / inertia;

    // 速度の半更新
    *vel += 0.5 * accel * dt;
    *omega += 0.5 * alpha * dt;

    // 位置の全更新
    *pos += *vel * dt;
}

/// Velocity Verlet積分の後半ステップ
/// 新しい力で速度を再度半更新
#[allow(clippy::too_many_arguments)]
pub fn integrate_second_half(
    vel: &mut Vec3,
    omega: &mut Vec3,
    force: Vec3,
    torque: Vec3,
    mass: f32,
    inertia: f32,
    gravity: Vec3,
    dt: f32,
) {
    // 加速度
    let accel = force / mass + gravity;
    let alpha = torque / inertia;

    // 速度の半更新
    *vel += 0.5 * accel * dt;
    *omega += 0.5 * alpha * dt;
}

/// 粒子の位置をコンテナ内にクランプ（ハード制約）
pub fn clamp_to_container(
    pos: &mut Vec3,
    vel: &mut Vec3,
    radius: f32,
    box_min: Vec3,
    box_max: Vec3,
) {
    // X軸
    let x_min = box_min.x + radius;
    let x_max = box_max.x - radius;
    if pos.x < x_min {
        pos.x = x_min;
        vel.x = vel.x.max(0.0); // 壁に向かう速度をゼロに
    } else if pos.x > x_max {
        pos.x = x_max;
        vel.x = vel.x.min(0.0);
    }

    // Y軸
    let y_min = box_min.y + radius;
    let y_max = box_max.y - radius;
    if pos.y < y_min {
        pos.y = y_min;
        vel.y = vel.y.max(0.0);
    } else if pos.y > y_max {
        pos.y = y_max;
        vel.y = vel.y.min(0.0);
    }

    // Z軸
    let z_min = box_min.z + radius;
    let z_max = box_max.z - radius;
    if pos.z < z_min {
        pos.z = z_min;
        vel.z = vel.z.max(0.0);
    } else if pos.z > z_max {
        pos.z = z_max;
        vel.z = vel.z.min(0.0);
    }
}

/// 速度をクランプ（数値安定性のため）
pub fn clamp_velocity(vel: &mut Vec3, omega: &mut Vec3, max_linear: f32, max_angular: f32) {
    if vel.length() > max_linear {
        *vel = vel.normalize() * max_linear;
    }
    if omega.length() > max_angular {
        *omega = omega.normalize() * max_angular;
    }
}
