use bevy::prelude::*;

/// シミュレーションの設定
#[derive(Resource, Clone)]
pub struct SimulationConfig {
    /// 大粒子の半径
    pub large_radius: f32,
    /// 小粒子の半径
    pub small_radius: f32,
    /// 粒子密度
    pub density: f32,
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
            density: 5000.0,     // kg/m^3
            num_large: 250,
            num_small: 1500,
        }
    }
}

/// シミュレーションの状態
#[derive(Resource, Default)]
pub struct SimulationState {
    /// 一時停止中かどうか
    pub paused: bool,
    /// リセットが要求されているか
    pub reset_requested: bool,
    /// 現在の振動オフセット
    pub container_offset: f32,
    /// 現在の振動位相 (rad)
    pub oscillation_phase: f32,
    /// 現在フレームのサブステップ開始時位相 (rad)
    pub oscillation_frame_start_phase: f32,
    /// 経過したシミュレーション時間（秒）
    pub elapsed: f64,
}

impl SimulationState {
    /// シミュレーション時間を 1 ステップ進める
    pub fn step_time(&mut self, dt: f32) {
        self.elapsed += dt as f64;
    }

    /// シミュレーション時間をリセットする
    pub fn reset_time(&mut self) {
        self.elapsed = 0.0;
    }
}
