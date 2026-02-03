use bevy::prelude::*;

/// シミュレーション時間（リアルタイムとは独立）
#[derive(Resource)]
pub struct SimulationTime {
    /// 経過したシミュレーション時間（秒）
    pub elapsed: f64,
    /// 固定タイムステップ（秒）
    pub dt: f32,
}

impl Default for SimulationTime {
    fn default() -> Self {
        Self {
            elapsed: 0.0,
            dt: 1.0 / 5000.0, // 5000Hz相当の細かいタイムステップ
        }
    }
}

impl SimulationTime {
    /// 1ステップ進める
    pub fn step(&mut self) {
        self.elapsed += self.dt as f64;
    }

    /// リセット
    pub fn reset(&mut self) {
        self.elapsed = 0.0;
    }
}

/// シミュレーション設定
#[derive(Resource)]
pub struct SimulationSettings {
    /// 1フレームあたりのサブステップ数
    pub substeps_per_frame: u32,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            substeps_per_frame: 8, // デフォルトで8サブステップ/フレーム
        }
    }
}
