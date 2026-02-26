use bevy::prelude::*;

pub mod config;
pub mod container;
pub mod oscillation;
pub mod physics;
pub mod time;

pub use config::SimulationConfig;
pub use container::ContainerParams;
pub use oscillation::{
    OscillationParams, advance_oscillation, advance_oscillation_phase, oscillation_displacement,
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
    pub divider_height: UiSliderRange,
    pub particle_restitution: UiSliderRange,
    pub particle_friction: UiSliderRange,
    pub wall_restitution: UiSliderRange,
    pub wall_friction: UiSliderRange,
}

impl Default for UiControlRanges {
    fn default() -> Self {
        let container_height = ContainerParams::default().half_extents.y * 2.0;
        let divider_min = 0.03_f32;
        let divider_max = (container_height - 0.03).max(divider_min + 0.001);

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
            divider_height: UiSliderRange {
                min: divider_min,
                max: divider_max,
                step: 0.001,
            },
            particle_restitution: UiSliderRange {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
            particle_friction: UiSliderRange {
                min: 0.0,
                max: 2.0,
                step: 0.01,
            },
            wall_restitution: UiSliderRange {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
            wall_friction: UiSliderRange {
                min: 0.0,
                max: 2.0,
                step: 0.01,
            },
        }
    }
}

/// シミュレーションで使う不変パラメータを集約した定数リソース。
///
/// これを Main World の単一リソースとして管理し、
/// CPU/GPU/Rendering/UI から参照する。
#[derive(Resource, Clone)]
pub struct SimulationConstants {
    pub particle: SimulationConfig,
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
        Self::new(
            SimulationConfig::default(),
            ContainerParams::default(),
            OscillationParams::default(),
            SimulationTimeParams::default(),
            SimulationSettings::default(),
            PhysicsConstants::default(),
            MaterialProperties::default(),
            WallProperties::default(),
        )
    }
}

impl SimulationConstants {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        particle: SimulationConfig,
        container: ContainerParams,
        oscillation: OscillationParams,
        time: SimulationTimeParams,
        settings: SimulationSettings,
        physics: PhysicsConstants,
        material: MaterialProperties,
        wall: WallProperties,
    ) -> Self {
        let grid = GridSettings::derive_from_scene(&particle, &container);

        Self {
            particle,
            container,
            oscillation,
            time,
            settings,
            physics,
            material,
            wall,
            grid,
        }
    }

    pub fn set_particle(&mut self, particle: SimulationConfig) {
        self.particle = particle;
        self.refresh_grid_settings();
    }

    pub fn set_container(&mut self, container: ContainerParams) {
        self.container = container;
        self.refresh_grid_settings();
    }

    pub fn refresh_grid_settings(&mut self) {
        self.grid = GridSettings::derive_from_scene(&self.particle, &self.container);
    }
}
