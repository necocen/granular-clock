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
            amplitude: 0.025, // 25 mm（非常に穏やかな振動）
            frequency: 5.0,   // 5 Hz（低周波）
            phase: 0.0,
            enabled: true,
        }
    }
}

impl OscillationParams {
    /// 最大加速度を計算 (m/s²)
    pub fn max_acceleration(&self) -> f32 {
        use std::f32::consts::PI;
        self.amplitude * (2.0 * PI * self.frequency).powi(2)
    }
}

/// 振動を更新するシステム
pub fn update_oscillation(
    mut container: ResMut<Container>,
    mut params: ResMut<OscillationParams>,
    sim_time: Res<super::SimulationTime>,
) {
    if !params.enabled {
        container.current_offset = 0.0;
        return;
    }

    // 位相を更新
    params.phase += params.frequency * 2.0 * PI * sim_time.dt;
    if params.phase > 2.0 * PI {
        params.phase -= 2.0 * PI;
    }

    // 正弦波で振動
    container.current_offset = params.amplitude * params.phase.sin();
}
