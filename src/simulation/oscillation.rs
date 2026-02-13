use bevy::prelude::*;
use std::f32::consts::PI;

use super::SimulationState;

const TWO_PI: f32 = 2.0 * PI;

/// 振動パラメータ
#[derive(Resource, Clone, Copy)]
pub struct OscillationParams {
    /// 振幅 (m)
    pub amplitude: f32,
    /// 周波数 (Hz)
    pub frequency: f32,
    /// 振動が有効かどうか
    pub enabled: bool,
}

impl Default for OscillationParams {
    fn default() -> Self {
        Self {
            amplitude: 0.03, // 30 mm（非常に穏やかな振動）
            frequency: 5.0,  // 5 Hz（低周波）
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

/// 1 サブステップあたりの位相増分を返す。
pub fn oscillation_phase_step(frequency: f32, dt: f32) -> f32 {
    frequency * TWO_PI * dt
}

/// 位相を 1 サブステップ進める（CPU/GPU 共通）。
pub fn advance_oscillation_phase(phase: &mut f32, frequency: f32, dt: f32) {
    *phase += oscillation_phase_step(frequency, dt);
    if *phase > TWO_PI {
        *phase -= TWO_PI;
    }
}

/// 位相から振動変位（base からの相対オフセット）を返す。
pub fn oscillation_displacement(amplitude: f32, phase: f32) -> f32 {
    amplitude * phase.sin()
}

/// 振動位相を 1 ステップ進め、コンテナオフセットを更新する（CPU/GPU 共通）。
pub fn advance_oscillation(sim_state: &mut SimulationState, params: &OscillationParams, dt: f32) {
    // 振動無効時は「現在位置で停止」させる（オフセットを維持する）。
    if !params.enabled {
        return;
    }

    advance_oscillation_phase(&mut sim_state.oscillation_phase, params.frequency, dt);
    sim_state.container_offset =
        oscillation_displacement(params.amplitude, sim_state.oscillation_phase);
}
