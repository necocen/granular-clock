use bevy::prelude::*;
use std::collections::HashMap;
use std::f32::consts::PI;

use crate::simulation::constants::MaterialProperties;

/// 接触履歴
#[derive(Default, Clone, Copy)]
pub struct ContactState {
    /// 最後の法線方向
    pub last_normal: Vec3,
    /// 接触が有効かどうか
    pub active: bool,
}

/// 接触履歴を管理するリソース
#[derive(Resource, Default)]
pub struct ContactHistory {
    /// キー: (min(index_a, index_b), max(index_a, index_b))
    pub contacts: HashMap<(usize, usize), ContactState>,
}

impl ContactHistory {
    /// 接触ペアのキーを生成（順序を正規化）
    pub fn key(i: usize, j: usize) -> (usize, usize) {
        if i < j {
            (i, j)
        } else {
            (j, i)
        }
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
    _dt: f32,
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

    // オーバーラップを制限して数値安定性を確保（貫通を減らすため厳しく制限）
    let max_overlap = r_eff * 0.2;
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

    // 非対称減衰：接近中（v_n < 0）のみ減衰を適用
    // 離反中（v_n > 0）は減衰なし - バネ力のみで押し出す
    // これにより、反発係数が正しく実現され、エネルギーが増加しない
    let f_n_damping = if v_n < 0.0 {
        // 接近中: -gamma * v_n は正（v_nが負なので）
        -gamma_n * v_n
    } else {
        // 離反中: 減衰なし
        0.0
    };

    // 法線力（引力にならないようにクランプ、通常は不要だが安全のため）
    let f_n_total = (f_n_elastic + f_n_damping).max(0.0);
    let f_n_vec = f_n_total * n;

    // 接触点での相対速度（回転を考慮）
    let v_t = v_rel - v_n * n - omega_i.cross(radius_i * n) + omega_j.cross(radius_j * n);

    contact_state.last_normal = n;

    // 正規化Coulomb摩擦モデル（エネルギー保存のため）
    // バネベースのMindlinモデルは k_t が overlap に依存するため
    // overlap が変化するとエネルギーが保存されない問題があった
    // 代わりに純粋な粘性摩擦 + Coulomb限界を使用
    let f_t_max = material.friction * f_n_elastic;
    let v_t_mag = v_t.length();

    let f_t_vec = if v_t_mag > 1e-10 {
        let v_t_dir = v_t / v_t_mag;
        // 正規化速度（特性速度でスケール）
        // 低速域では粘性摩擦、高速域ではCoulomb摩擦
        // v_char を大きくして粘性摩擦を緩やかにする
        let v_char = 0.1; // 特性速度 10cm/s
        let viscous_coeff = f_t_max / v_char;
        let f_viscous = viscous_coeff * v_t_mag;
        // Coulomb限界でクランプ
        let f_t = f_viscous.min(f_t_max);
        -f_t * v_t_dir
    } else {
        Vec3::ZERO
    };

    // 転がり抵抗（同様に粘性モデルに変更）
    let omega_rel = omega_i - omega_j;
    let omega_roll = omega_rel - omega_rel.dot(n) * n;
    let omega_roll_mag = omega_roll.length();

    let t_r_max = material.rolling_friction * f_n_elastic * r_eff;
    let t_r_vec = if omega_roll_mag > 1e-10 {
        let omega_roll_dir = omega_roll / omega_roll_mag;
        // 特性角速度でスケール（大きくして緩やかに）
        let omega_char = 10.0; // rad/s
        let viscous_coeff = t_r_max / omega_char;
        let t_viscous = viscous_coeff * omega_roll_mag;
        let t_r = t_viscous.min(t_r_max);
        -t_r * omega_roll_dir
    } else {
        Vec3::ZERO
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
