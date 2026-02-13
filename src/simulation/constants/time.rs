use bevy::prelude::*;

/// 物理計算バックエンド
#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PhysicsBackend {
    /// CPU（rayon並列化）
    Cpu,
    /// GPU（WebGPU compute shader）
    #[default]
    Gpu,
}

/// シミュレーション時間の不変パラメータ
#[derive(Resource, Clone, Copy)]
pub struct SimulationTimeParams {
    /// 固定タイムステップ（秒）
    pub dt: f32,
}

impl Default for SimulationTimeParams {
    fn default() -> Self {
        Self {
            dt: 1.0 / 2500.0, // 2500Hz相当の細かいタイムステップ
        }
    }
}

/// シミュレーション設定
#[derive(Resource, Clone, Copy)]
pub struct SimulationSettings {
    /// 1フレームあたりのサブステップ数
    pub substeps_per_frame: u32,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        Self {
            substeps_per_frame: 3, // 3サブステップ/フレーム
        }
    }
}
