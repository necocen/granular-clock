/// 物理計算バックエンド
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PhysicsBackend {
    /// CPU
    Cpu,
    /// GPU（WebGPU compute shader）
    #[default]
    Gpu,
}

/// シミュレーション時間の不変パラメータ
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
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
