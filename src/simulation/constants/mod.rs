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

/// シミュレーションで使う不変パラメータを集約した定数リソース。
///
/// これを Main World の単一リソースとして管理し、
/// CPU/GPU/Rendering/UI から参照する。
#[derive(Resource, Clone)]
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

impl Default for SimulationConstants {
    fn default() -> Self {
        Self {
            config: SimulationConfig::default(),
            container: ContainerParams::default(),
            oscillation: OscillationParams::default(),
            time: SimulationTimeParams::default(),
            settings: SimulationSettings::default(),
            physics: PhysicsConstants::default(),
            material: MaterialProperties::default(),
            wall: WallProperties::default(),
            grid: GridSettings::default(),
        }
    }
}
