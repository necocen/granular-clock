use bevy::prelude::*;

use super::{config::SimulationConfig, container::ContainerParams};

/// 粒子間接触の材料パラメータ（CPU/GPU 共通）
#[derive(Resource, Clone, Copy)]
pub struct MaterialProperties {
    /// 粒子密度 (kg/m^3)
    pub density: f32,
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
            density: 5000.0,
            youngs_modulus: 1e7, // 10MPa　かため
            poisson_ratio: 0.25,
            restitution: 0.6, // 反発係数（中程度）
            friction: 0.3,
            rolling_friction: 0.1,
        }
    }
}

/// 壁との接触パラメータ（CPU/GPU 共通）
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
            friction: 0.6,      // 摩擦係数
            restitution: 0.5,   // 反発係数（低めに設定）
        }
    }
}

/// シミュレーションの物理定数（CPU/GPU 共通）
#[derive(Resource, Clone, Copy)]
pub struct PhysicsConstants {
    /// 重力加速度
    pub gravity: Vec3,
}

impl Default for PhysicsConstants {
    fn default() -> Self {
        Self {
            gravity: Vec3::new(0.0, -9.81, 0.0),
        }
    }
}

/// 空間グリッド設定（CPU/GPU 共通）
#[derive(Resource, Clone, Copy, Debug)]
pub struct GridSettings {
    /// セルサイズ（最大粒子直径以上が推奨）
    pub cell_size: f32,
    /// ハッシュテーブルサイズ（CPU用、2のべき乗推奨）
    pub table_size: usize,
}

impl Default for GridSettings {
    fn default() -> Self {
        Self::derive_from_scene(&SimulationConfig::default(), &ContainerParams::default())
    }
}

impl GridSettings {
    /// 粒子径とコンテナ寸法からグリッド設定を自動導出
    pub fn derive_from_scene(particle: &SimulationConfig, container: &ContainerParams) -> Self {
        let max_radius = particle.large_radius.max(particle.small_radius).max(1e-6);
        let min_cell_size = (2.0 * max_radius * 1.05).max(1e-4);

        let max_extent = container
            .half_extents
            .x
            .max(container.half_extents.y)
            .max(container.half_extents.z);
        let world_size = (2.0 * max_extent * 1.5).max(min_cell_size * 8.0);

        // grid_dim 上限(64)を満たすようにセルサイズ下限も加味する
        let cell_size = min_cell_size.max(world_size / 64.0);
        let dim = ((world_size / cell_size).ceil() as usize).clamp(8, 64);
        let approx_cells = dim * dim * dim;

        // CPU側 HashMap 初期容量。粒子数とセル数の中間程度を狙い、2冪に正規化。
        let num_particles = (particle.num_large as usize)
            .saturating_add(particle.num_small as usize)
            .max(1);
        let target_capacity = num_particles
            .saturating_mul(2)
            .max(approx_cells / 2)
            .max(1024);
        let table_size = target_capacity
            .checked_next_power_of_two()
            .unwrap_or(4096)
            .clamp(1024, 1 << 20);

        Self {
            cell_size,
            table_size,
        }
    }

    /// ワールドサイズからグリッド次元を計算（GPU用）
    /// world_half: 各軸の半分のサイズ（例: container.half_extents）
    pub fn compute_grid_dim(&self, world_half: [f32; 3]) -> u32 {
        // 最大軸のサイズを基準にグリッド次元を計算
        // 2 * world_half でワールド全体をカバー + マージン
        let max_extent = world_half[0].max(world_half[1]).max(world_half[2]);
        let world_size = 2.0 * max_extent * 1.5; // 50%マージン
        let dim = (world_size / self.cell_size).ceil() as u32;
        // 最小8、最大64に制限（メモリ効率のため）
        dim.clamp(8, 64)
    }
}
