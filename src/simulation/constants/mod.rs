use bevy::prelude::*;

pub mod config;
pub mod container;
pub mod oscillation;
pub mod physics;
pub mod time;

pub use config::SimulationConfig;
pub use container::ContainerParams;
pub use oscillation::{
    advance_oscillation, advance_oscillation_phase, oscillation_displacement, OscillationParams,
};
pub use physics::{GridSettings, MaterialProperties, PhysicsConstants, WallProperties};
pub use time::{PhysicsBackend, SimulationSettings, SimulationTimeParams};

/// UI スライダーのレンジ定義
#[derive(Clone, Copy, Debug)]
pub struct UiSliderRange {
    pub min: f32,
    pub max: f32,
    pub step: f32,
}

/// UI で変更可能な値のレンジ設定
#[derive(Resource, Clone, Copy, Debug)]
pub struct UiControlRanges {
    pub oscillation_amplitude: UiSliderRange,
    pub oscillation_frequency: UiSliderRange,
}

impl Default for UiControlRanges {
    fn default() -> Self {
        Self {
            oscillation_amplitude: UiSliderRange {
                min: 0.001,
                max: 0.1,
                step: 0.001,
            },
            oscillation_frequency: UiSliderRange {
                min: 1.0,
                max: 20.0,
                step: 1.0,
            },
        }
    }
}

/// シミュレーションで使う不変パラメータを集約した定数リソース。
///
/// これを Main World の単一リソースとして管理し、
/// CPU/GPU/Rendering/UI から参照する。
#[derive(Resource, Clone, Default)]
pub struct SimulationConstants {
    pub config: SimulationConfig,
    pub container: ContainerParams,
    pub oscillation: OscillationParams,
    pub time: SimulationTimeParams,
    pub settings: SimulationSettings,
    pub physics: PhysicsConstants,
    pub material: MaterialProperties,
    pub wall: WallProperties,
    pub grid: GridSettings,
}
