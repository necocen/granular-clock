use bevy::prelude::*;
use std::f32::consts::PI;

use super::Container;

/// 振動パラメータ
#[derive(Resource, Clone, Copy)]
pub struct OscillationParams {
    /// 振幅 (m)
    pub amplitude: f32,
    /// 周波数 (Hz)
    pub frequency: f32,
    /// 現在の位相 (rad)
    pub phase: f32,
    /// 振動が有効かどうか
    pub enabled: bool,
}

impl Default for OscillationParams {
    fn default() -> Self {
        Self {
            amplitude: 0.03, // 30 mm（非常に穏やかな振動）
            frequency: 5.0,  // 5 Hz（低周波）
            phase: 0.0,
            enabled: true,
        }
    }
}

impl OscillationParams {
    /// 最大加速度を計算 (m/s²)
    #[allow(dead_code)]
    pub fn max_acceleration(&self) -> f32 {
        use std::f32::consts::PI;
        self.amplitude * (2.0 * PI * self.frequency).powi(2)
    }
}

/// GPU モード用: 振動を更新するシステム
/// CPU モードでは `systems::update_oscillation` がサブステップ内で呼ばれるため不要
pub fn update_oscillation_for_gpu(
    mut container: ResMut<Container>,
    params: Res<OscillationParams>,
    sim_time: Res<super::SimulationTime>,
    backend: Res<super::PhysicsBackend>,
) {
    if *backend != super::PhysicsBackend::Gpu {
        return;
    }
    if !params.enabled {
        container.current_offset = 0.0;
        return;
    }

    let t = sim_time.elapsed as f32;
    container.current_offset = params.amplitude * (2.0 * PI * params.frequency * t).sin();
}
