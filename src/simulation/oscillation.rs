use bevy::prelude::*;
use std::f32::consts::PI;

use super::Container;

const TWO_PI: f32 = 2.0 * PI;

/// 振動パラメータ
#[derive(Resource, Clone, Copy)]
pub struct OscillationParams {
    /// 振幅 (m)
    pub amplitude: f32,
    /// 周波数 (Hz)
    pub frequency: f32,
    /// 現在の位相 (rad)
    pub phase: f32,
    /// 現在フレームのサブステップ開始時位相 (rad)
    pub frame_start_phase: f32,
    /// 振動が有効かどうか
    pub enabled: bool,
}

impl Default for OscillationParams {
    fn default() -> Self {
        Self {
            amplitude: 0.03, // 30 mm（非常に穏やかな振動）
            frequency: 5.0,  // 5 Hz（低周波）
            phase: 0.0,
            frame_start_phase: 0.0,
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
pub fn oscillation_displacement(enabled: bool, amplitude: f32, phase: f32) -> f32 {
    if !enabled {
        return 0.0;
    }
    amplitude * phase.sin()
}

/// 振動位相を 1 ステップ進め、コンテナオフセットを更新する（CPU/GPU 共通）。
pub fn advance_oscillation(container: &mut Container, params: &mut OscillationParams, dt: f32) {
    if params.enabled {
        advance_oscillation_phase(&mut params.phase, params.frequency, dt);
    }
    container.current_offset =
        oscillation_displacement(params.enabled, params.amplitude, params.phase);
}
