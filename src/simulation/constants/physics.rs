use bevy::prelude::*;

/// 粒子間接触の材料パラメータ（CPU/GPU 共通）
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
        Self {
            cell_size: 0.03, // 大粒子直径 0.02 より大きい
            table_size: 4096,
        }
    }
}

impl GridSettings {
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
