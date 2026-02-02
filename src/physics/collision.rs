use bevy::prelude::*;

use crate::simulation::Container;

/// 壁との接触力計算結果
#[derive(Debug, Clone, Copy, Default)]
pub struct WallContactForce {
    pub force: Vec3,
    pub torque: Vec3,
}

/// 壁との衝突パラメータ
#[derive(Resource, Clone, Copy)]
pub struct WallProperties {
    /// 壁の剛性
    pub stiffness: f32,
    /// 壁の減衰係数
    pub damping: f32,
    /// 壁との摩擦係数
    pub friction: f32,
}

impl Default for WallProperties {
    fn default() -> Self {
        Self {
            stiffness: 5e4,   // 剛性を下げて数値安定性を確保
            damping: 200.0,   // 減衰を下げる（500→200）
            friction: 0.3,    // 摩擦係数を下げる（0.5→0.3）
        }
    }
}

/// 粒子と壁との接触力を計算
pub fn compute_wall_contact_force(
    pos: Vec3,
    vel: Vec3,
    omega: Vec3,
    radius: f32,
    mass: f32,
    container: &Container,
    wall_props: &WallProperties,
) -> WallContactForce {
    let mut force = Vec3::ZERO;
    let mut torque = Vec3::ZERO;

    // コンテナの現在位置（振動を考慮）
    let box_offset = Vec3::Y * container.current_offset;
    let box_min = container.base_position - container.half_extents + box_offset;
    let box_max = container.base_position + container.half_extents + box_offset;

    // 床との接触（最も重要）
    let floor_y = box_min.y;
    let floor_overlap = floor_y + radius - pos.y;
    if floor_overlap > 0.0 {
        let overlap = floor_overlap.min(radius); // オーバーラップを制限
        let f_n = wall_props.stiffness * overlap;
        // 減衰: 床に向かう速度（負のvel.y）に対してのみ抵抗
        let f_d = -wall_props.damping * vel.y.min(0.0);
        force.y += (f_n + f_d).max(0.0);

        // 接線摩擦（Coulomb摩擦モデル）
        let contact_point = Vec3::new(0.0, -radius, 0.0);
        let v_t = Vec3::new(vel.x, 0.0, vel.z) + omega.cross(contact_point);
        if v_t.length() > 1e-6 {
            // 最大摩擦力 = μ * N
            let f_t_max = wall_props.friction * f_n;
            // 粘性減衰項も加味するが、最大摩擦力で制限
            let f_t = f_t_max.min(v_t.length() * wall_props.damping * 0.3);
            let friction_force = -f_t * v_t.normalize();
            force += friction_force;
            torque += contact_point.cross(friction_force);
        }
    }

    // 天井との接触
    let ceiling_y = box_max.y;
    let ceiling_overlap = pos.y + radius - ceiling_y;
    if ceiling_overlap > 0.0 {
        let overlap = ceiling_overlap.min(radius);
        let f_n = wall_props.stiffness * overlap;
        let v_n = vel.y;
        let f_d = -wall_props.damping * v_n;
        force.y -= (f_n - f_d).max(0.0);
    }

    // 左壁 (-X)
    let left_overlap = box_min.x + radius - pos.x;
    if left_overlap > 0.0 {
        let overlap = left_overlap.min(radius);
        let f_n = wall_props.stiffness * overlap;
        let v_n = vel.x;
        let f_d = -wall_props.damping * v_n;
        force.x += (f_n + f_d).max(0.0);
    }

    // 右壁 (+X)
    let right_overlap = pos.x + radius - box_max.x;
    if right_overlap > 0.0 {
        let overlap = right_overlap.min(radius);
        let f_n = wall_props.stiffness * overlap;
        let v_n = vel.x;
        let f_d = -wall_props.damping * v_n;
        force.x -= (f_n - f_d).max(0.0);
    }

    // 前壁 (-Z)
    let front_overlap = box_min.z + radius - pos.z;
    if front_overlap > 0.0 {
        let overlap = front_overlap.min(radius);
        let f_n = wall_props.stiffness * overlap;
        let v_n = vel.z;
        let f_d = -wall_props.damping * v_n;
        force.z += (f_n + f_d).max(0.0);
    }

    // 後壁 (+Z)
    let back_overlap = pos.z + radius - box_max.z;
    if back_overlap > 0.0 {
        let overlap = back_overlap.min(radius);
        let f_n = wall_props.stiffness * overlap;
        let v_n = vel.z;
        let f_d = -wall_props.damping * v_n;
        force.z -= (f_n - f_d).max(0.0);
    }

    // 仕切り壁との接触
    let divider_top = floor_y + container.divider_height;

    // 仕切りより下にいる場合のみ接触チェック
    if pos.y - radius < divider_top {
        let half_thickness = container.divider_thickness / 2.0;

        // 仕切りの右側（+X側から接近）
        if pos.x > 0.0 && pos.x - radius < half_thickness {
            let overlap = half_thickness - (pos.x - radius);
            if overlap > 0.0 {
                let overlap = overlap.min(radius);
                let f_n = wall_props.stiffness * overlap;
                // 減衰: 仕切りに向かう速度（負のvel.x）に対して抵抗
                let f_d = -wall_props.damping * vel.x.min(0.0);
                force.x += (f_n + f_d).max(0.0);

                // 接線摩擦（Y-Z平面）
                let v_t = Vec3::new(0.0, vel.y, vel.z);
                if v_t.length() > 1e-6 {
                    let f_t = (wall_props.friction * f_n).min(v_t.length() * wall_props.damping * 0.5);
                    force -= f_t * v_t.normalize();
                }
            }
        }
        // 仕切りの左側（-X側から接近）
        else if pos.x < 0.0 && pos.x + radius > -half_thickness {
            let overlap = (pos.x + radius) - (-half_thickness);
            if overlap > 0.0 {
                let overlap = overlap.min(radius);
                let f_n = wall_props.stiffness * overlap;
                // 減衰: 仕切りに向かう速度（正のvel.x）に対して抵抗
                let f_d = wall_props.damping * vel.x.max(0.0);
                force.x -= (f_n + f_d).max(0.0);

                // 接線摩擦（Y-Z平面）
                let v_t = Vec3::new(0.0, vel.y, vel.z);
                if v_t.length() > 1e-6 {
                    let f_t = (wall_props.friction * f_n).min(v_t.length() * wall_props.damping * 0.5);
                    force -= f_t * v_t.normalize();
                }
            }
        }
    }

    // 壁の力は粒子間よりも強くする必要がある（壁から飛び出さないように）
    let max_accel = 5000.0; // m/s² - 壁は500g程度まで許容
    let max_force = max_accel * mass;

    if force.is_nan() || !force.is_finite() {
        force = Vec3::ZERO;
    } else {
        force = force.clamp_length_max(max_force);
    }
    if torque.is_nan() || !torque.is_finite() {
        torque = Vec3::ZERO;
    } else {
        torque = torque.clamp_length_max(max_force * radius);
    }

    WallContactForce { force, torque }
}
