use bevy::prelude::*;

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
