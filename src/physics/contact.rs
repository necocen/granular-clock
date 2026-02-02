use bevy::prelude::*;
use std::collections::HashMap;
use std::f32::consts::PI;

/// 接触履歴（Mindlin接線力の計算に必要）
#[derive(Default, Clone)]
pub struct ContactState {
    /// 接線方向の累積変位
    pub tangential_displacement: Vec3,
    /// 転がり方向の累積変位
    pub rolling_displacement: Vec3,
    /// 最後の法線方向
    pub last_normal: Vec3,
    /// 接触が有効かどうか
    pub active: bool,
}

/// 接触履歴を管理するリソース
#[derive(Resource, Default)]
pub struct ContactHistory {
    /// キー: (min(entity_a, entity_b), max(entity_a, entity_b))
    pub contacts: HashMap<(Entity, Entity), ContactState>,
}

impl ContactHistory {
    /// 接触ペアのキーを生成（順序を正規化）
    pub fn key(e1: Entity, e2: Entity) -> (Entity, Entity) {
        if e1 < e2 {
            (e1, e2)
        } else {
            (e2, e1)
        }
    }

    /// 接触状態を取得または作成
    pub fn get_or_create(&mut self, e1: Entity, e2: Entity) -> &mut ContactState {
        let key = Self::key(e1, e2);
        self.contacts.entry(key).or_default()
    }

    /// 非アクティブな接触を削除
    pub fn cleanup(&mut self) {
        self.contacts.retain(|_, state| state.active);
        // 全ての接触をリセット
        for state in self.contacts.values_mut() {
            state.active = false;
        }
    }
}

/// 材料パラメータ
#[derive(Resource, Clone, Copy)]
pub struct MaterialProperties {
    /// ヤング率 (Pa)
    pub youngs_modulus: f32,
    /// ポアソン比
    pub poisson_ratio: f32,
    /// 反発係数
    pub restitution: f32,
    /// 摩擦係数
    pub friction: f32,
    /// 転がり摩擦係数
    pub rolling_friction: f32,
}

impl Default for MaterialProperties {
    fn default() -> Self {
        Self {
            youngs_modulus: 1e5, // 数値安定性のため大幅に下げる
            poisson_ratio: 0.25,
            restitution: 0.5,
            friction: 0.5,
            rolling_friction: 0.1,
        }
    }
}

/// 接触力の計算結果
#[derive(Debug, Clone, Copy)]
pub struct ContactForce {
    pub force: Vec3,
    pub torque: Vec3,
}

impl Default for ContactForce {
    fn default() -> Self {
        Self {
            force: Vec3::ZERO,
            torque: Vec3::ZERO,
        }
    }
}

/// Hertz-Mindlin接触モデルで粒子間の力を計算
#[allow(clippy::too_many_arguments)]
pub fn compute_particle_contact_force(
    pos_i: Vec3,
    vel_i: Vec3,
    omega_i: Vec3,
    radius_i: f32,
    mass_i: f32,
    pos_j: Vec3,
    vel_j: Vec3,
    omega_j: Vec3,
    radius_j: f32,
    mass_j: f32,
    material: &MaterialProperties,
    contact_state: &mut ContactState,
    dt: f32,
) -> (ContactForce, ContactForce) {
    let delta_pos = pos_i - pos_j;
    let dist = delta_pos.length();
    let overlap = (radius_i + radius_j) - dist;

    // 接触していない場合
    if overlap <= 0.0 || dist < 1e-10 {
        return (ContactForce::default(), ContactForce::default());
    }

    contact_state.active = true;

    // 法線ベクトル (jからiへ)
    let n = delta_pos / dist;

    // 有効パラメータ
    let r_eff = (radius_i * radius_j) / (radius_i + radius_j);

    // オーバーラップを制限して数値安定性を確保
    let max_overlap = r_eff * 0.5;
    let overlap = overlap.min(max_overlap);
    let m_eff = (mass_i * mass_j) / (mass_i + mass_j);
    let e_eff =
        material.youngs_modulus / (2.0 * (1.0 - material.poisson_ratio * material.poisson_ratio));

    // 法線剛性 (Hertz)
    let k_n = (4.0 / 3.0) * e_eff * (r_eff * overlap).sqrt();

    // 法線弾性力
    let f_n_elastic = k_n * overlap;

    // 法線減衰係数
    let ln_e = material.restitution.max(0.01).ln();
    let gamma_n = -2.0 * ln_e * (k_n * m_eff).sqrt() / (PI * PI + ln_e * ln_e).sqrt();

    // 相対速度
    let v_rel = vel_i - vel_j;
    let v_n = v_rel.dot(n);
    let f_n_damping = -gamma_n * v_n;

    // 法線力（引力にならないようにクランプ）
    let f_n_total = (f_n_elastic + f_n_damping).max(0.0);
    let f_n_vec = f_n_total * n;

    // 接触点での相対速度（回転を考慮）
    let v_t = v_rel - v_n * n - omega_i.cross(radius_i * n) + omega_j.cross(radius_j * n);

    // 接線変位の更新
    contact_state.tangential_displacement += v_t * dt;
    // 接触面内に射影
    contact_state.tangential_displacement -=
        contact_state.tangential_displacement.dot(n) * n;
    contact_state.last_normal = n;

    // 接線剛性
    let g_eff = material.youngs_modulus
        / (4.0 * (2.0 - material.poisson_ratio) * (1.0 + material.poisson_ratio));
    let k_t = 8.0 * g_eff * (r_eff * overlap).sqrt();

    // 接線力（Coulomb摩擦でクランプ）
    let f_t_elastic = -k_t * contact_state.tangential_displacement;
    let f_t_max = material.friction * f_n_elastic;

    let f_t_vec = if f_t_elastic.length() > f_t_max {
        let direction = f_t_elastic.normalize_or_zero();
        // すべりが発生：変位をリセット
        contact_state.tangential_displacement = -direction * f_t_max / k_t;
        -direction * f_t_max
    } else {
        f_t_elastic
    };

    // 転がり抵抗
    let omega_rel = omega_i - omega_j;
    let omega_roll = omega_rel - omega_rel.dot(n) * n;
    contact_state.rolling_displacement += omega_roll * dt * r_eff;

    let k_r = k_n * r_eff * r_eff;
    let t_r_elastic = -k_r * contact_state.rolling_displacement;
    let t_r_max = material.rolling_friction * f_n_elastic * r_eff;

    let t_r_vec = if t_r_elastic.length() > t_r_max {
        let direction = t_r_elastic.normalize_or_zero();
        contact_state.rolling_displacement = -direction * t_r_max / k_r;
        -direction * t_r_max
    } else {
        t_r_elastic
    };

    // 粒子iに作用する力とトルク
    let force_i = f_n_vec + f_t_vec;
    let torque_i = (radius_i * n).cross(f_t_vec) + t_r_vec;

    // 粒子jに作用する力とトルク（作用反作用）
    let force_j = -force_i;
    let torque_j = (radius_j * (-n)).cross(-f_t_vec) - t_r_vec;

    // 最大加速度を制限（100g程度）して数値安定性を確保
    let max_accel = 1000.0; // m/s²
    let max_force_i = max_accel * mass_i;
    let max_force_j = max_accel * mass_j;

    let clamp_force = |v: Vec3, max: f32| -> Vec3 {
        if v.is_nan() || !v.is_finite() {
            Vec3::ZERO
        } else {
            v.clamp_length_max(max)
        }
    };

    (
        ContactForce {
            force: clamp_force(force_i, max_force_i),
            torque: clamp_force(torque_i, max_force_i * radius_i),
        },
        ContactForce {
            force: clamp_force(force_j, max_force_j),
            torque: clamp_force(torque_j, max_force_j * radius_j),
        },
    )
}
