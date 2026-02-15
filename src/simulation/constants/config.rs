use bevy::prelude::*;

/// シミュレーションの設定
#[derive(Resource, Clone)]
pub struct SimulationConfig {
    /// 大粒子の半径
    pub large_radius: f32,
    /// 小粒子の半径
    pub small_radius: f32,
    /// 大粒子の数
    pub num_large: u32,
    /// 小粒子の数
    pub num_small: u32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            large_radius: 0.01,  // 10 mm
            small_radius: 0.006, // 6 mm
            num_large: 250,
            num_small: 1500,
        }
    }
}
