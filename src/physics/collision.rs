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
    /// 壁の剛性（ペナルティ法用）
    pub stiffness: f32,
    /// 壁の減衰係数
    pub damping: f32,
    /// 壁との摩擦係数
    pub friction: f32,
    /// 反発係数（0=完全非弾性、1=完全弾性）
    pub restitution: f32,
}

impl Default for WallProperties {
    fn default() -> Self {
        Self {
            stiffness: 10000.0, // ペナルティ剛性（高めに設定して貫通を減らす）
            damping: 20.0,      // 減衰係数（参考値、実際は質量から計算）
            friction: 0.4,      // 摩擦係数
            restitution: 0.6,   // 反発係数（低めに設定）
        }
    }
}

/// 壁との接触力を計算するヘルパー関数
/// overlap: 食い込み量（正=接触中）
/// v_n: 法線速度（負=接近、正=離反）
/// mass: 粒子の質量
/// 戻り値: 法線方向の力（正=押し出し方向）
fn compute_wall_normal_force(
    overlap: f32,
    v_n: f32,
    radius: f32,
    mass: f32,
    wall_props: &WallProperties,
) -> f32 {
    if overlap <= 0.0 {
        return 0.0;
    }

    let overlap = overlap.min(radius);

    // バネ力（常に押し出し方向）
    let f_spring = wall_props.stiffness * overlap;

    // 質量に基づいた減衰係数を計算
    // 臨界減衰: c_crit = 2 * sqrt(k * m)
    // 反発係数からの減衰比: ζ = -ln(e) / sqrt(π² + ln²(e))
    let ln_e = wall_props.restitution.max(0.01).ln();
    let damping_ratio = -ln_e / (std::f32::consts::PI * std::f32::consts::PI + ln_e * ln_e).sqrt();
    let c_crit = 2.0 * (wall_props.stiffness * mass).sqrt();
    let damping_coeff = damping_ratio * c_crit;

    // 非対称減衰：接近中のみ減衰を適用
    // 接近中（v_n < 0）: 減衰力でエネルギーを吸収
    // 離反中（v_n > 0）: 減衰なし、バネ力のみで押し出す
    let f_damp = if v_n < 0.0 {
        // 接近中: -damping * v_n は正（v_nが負なので）
        -damping_coeff * v_n
    } else {
        // 離反中: 減衰なし（正しい反発係数を実現するため）
        0.0
    };

    // 合計力（常に正＝押し出し方向）
    (f_spring + f_damp).max(0.0)
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

    // 接触閾値：clamping後でも床との接触として扱う（1mm以内）
    let contact_threshold = 0.001;
    let is_floor_contact = floor_overlap > -contact_threshold;

    if is_floor_contact {
        // 法線力の計算（実際のoverlapがある場合のみ）
        let effective_overlap = floor_overlap.max(0.0);
        if effective_overlap > 0.0 {
            let v_n = vel.y;
            let f_n = compute_wall_normal_force(effective_overlap, v_n, radius, mass, wall_props);
            force.y += f_n;
        }

        // 床に乗っている場合の垂直方向減衰
        // Velocity Verlet + クランプによる数値的なY方向振動を抑制
        if vel.y < 0.0 && vel.y > -0.01 {
            // 微小な下向き速度を急速に減衰（床との接触を維持）
            let damping_force = -vel.y * mass * 100.0; // 強い減衰
            force.y += damping_force;
        }

        // 接線摩擦（Coulomb摩擦モデル）
        // 床に乗っている場合、垂直抗力は少なくとも重力と釣り合う
        let gravity_mag = 9.81;
        let normal_force = if effective_overlap > 0.0 {
            let f_n = compute_wall_normal_force(effective_overlap, vel.y, radius, mass, wall_props);
            // 重力を支えるのに十分な垂直抗力を使用
            f_n.max(mass * gravity_mag)
        } else {
            // overlap=0（床にちょうど乗っている）場合、垂直抗力=mg
            mass * gravity_mag
        };

        let contact_point = Vec3::new(0.0, -radius, 0.0);
        let v_t = Vec3::new(vel.x, 0.0, vel.z) + omega.cross(contact_point);
        let v_t_mag = v_t.length();
        if v_t_mag > 1e-8 {
            let f_t_max = wall_props.friction * normal_force;
            // 粘性摩擦 + スティック摩擦（低速での停止を助ける）
            let viscous_friction = v_t_mag * wall_props.damping * 2.0;
            let stick_friction = f_t_max * 0.1; // 小さな静止摩擦
            let f_t = f_t_max.min(viscous_friction + stick_friction);
            let friction_force = -f_t * v_t.normalize();
            force += friction_force;
            torque += contact_point.cross(friction_force);
        }
    }

    // 天井との接触
    let ceiling_y = box_max.y;
    let ceiling_overlap = pos.y + radius - ceiling_y;
    if ceiling_overlap > 0.0 {
        // 法線速度（負=接近=上向き、正=離反=下向き）
        let v_n = -vel.y;
        let f_n = compute_wall_normal_force(ceiling_overlap, v_n, radius, mass, wall_props);
        force.y -= f_n;
    }

    // 左壁 (-X)
    let left_overlap = box_min.x + radius - pos.x;
    if left_overlap > 0.0 {
        // 法線速度（負=接近=左向き、正=離反=右向き）
        let v_n = vel.x;
        let f_n = compute_wall_normal_force(left_overlap, v_n, radius, mass, wall_props);
        force.x += f_n;
    }

    // 右壁 (+X)
    let right_overlap = pos.x + radius - box_max.x;
    if right_overlap > 0.0 {
        // 法線速度（負=接近=右向き、正=離反=左向き）
        let v_n = -vel.x;
        let f_n = compute_wall_normal_force(right_overlap, v_n, radius, mass, wall_props);
        force.x -= f_n;
    }

    // 前壁 (-Z)
    let front_overlap = box_min.z + radius - pos.z;
    if front_overlap > 0.0 {
        // 法線速度（負=接近=前向き、正=離反=後ろ向き）
        let v_n = vel.z;
        let f_n = compute_wall_normal_force(front_overlap, v_n, radius, mass, wall_props);
        force.z += f_n;
    }

    // 後壁 (+Z)
    let back_overlap = pos.z + radius - box_max.z;
    if back_overlap > 0.0 {
        // 法線速度（負=接近=後ろ向き、正=離反=前向き）
        let v_n = -vel.z;
        let f_n = compute_wall_normal_force(back_overlap, v_n, radius, mass, wall_props);
        force.z -= f_n;
    }

    // 仕切り壁との接触
    let divider_top = floor_y + container.divider_height;

    // 仕切りより下にいる場合のみ接触チェック
    if pos.y - radius < divider_top {
        let half_thickness = container.divider_thickness / 2.0;

        // 粒子の左端と右端のX座標
        let particle_left = pos.x - radius;
        let particle_right = pos.x + radius;

        // 仕切りの範囲: -half_thickness から +half_thickness
        // 粒子が仕切りと重なっているかチェック
        if particle_left < half_thickness && particle_right > -half_thickness {
            // 仕切りと重なっている
            // どちら側から押し出すかを決定（粒子中心の位置に基づく）
            if pos.x >= 0.0 {
                // 右側に押し出す
                let overlap = half_thickness - particle_left;
                if overlap > 0.0 {
                    // 法線速度（負=接近=左向き、正=離反=右向き）
                    let v_n = vel.x;
                    let f_n = compute_wall_normal_force(overlap, v_n, radius, mass, wall_props);
                    force.x += f_n;

                    // 接線摩擦（Y-Z平面）
                    if f_n > 0.0 {
                        let v_t = Vec3::new(0.0, vel.y, vel.z);
                        if v_t.length() > 1e-6 {
                            let f_t = (wall_props.friction * f_n)
                                .min(v_t.length() * wall_props.damping * 0.5);
                            force -= f_t * v_t.normalize();
                        }
                    }
                }
            } else {
                // 左側に押し出す
                let overlap = particle_right - (-half_thickness);
                if overlap > 0.0 {
                    // 法線速度（負=接近=右向き、正=離反=左向き）
                    let v_n = -vel.x;
                    let f_n = compute_wall_normal_force(overlap, v_n, radius, mass, wall_props);
                    force.x -= f_n;

                    // 接線摩擦（Y-Z平面）
                    if f_n > 0.0 {
                        let v_t = Vec3::new(0.0, vel.y, vel.z);
                        if v_t.length() > 1e-6 {
                            let f_t = (wall_props.friction * f_n)
                                .min(v_t.length() * wall_props.damping * 0.5);
                            force -= f_t * v_t.normalize();
                        }
                    }
                }
            }
        }
    }

    // NaN/Infチェックのみ（加速度制限は速度クランプに任せる）
    // 力のクランプはエネルギー非対称性の原因となるため削除
    if force.is_nan() || !force.is_finite() {
        force = Vec3::ZERO;
    }
    if torque.is_nan() || !torque.is_finite() {
        torque = Vec3::ZERO;
    }

    WallContactForce { force, torque }
}
