pub mod config;
pub mod container;
pub mod oscillation;
pub mod time;

/// CPU/GPU 共通で参照するシミュレーション要素。
pub mod common {
    pub use super::config::{SimulationConfig, SimulationState};
    pub use super::container::ContainerParams;
    pub use super::oscillation::{
        advance_oscillation, advance_oscillation_phase, oscillation_displacement, OscillationParams,
    };
    pub use super::time::{PhysicsBackend, SimulationSettings, SimulationTimeParams};
}

// 既存参照との互換性のため、主要シンボルは root でも再公開する。
pub use common::{
    advance_oscillation, advance_oscillation_phase, oscillation_displacement, ContainerParams,
    OscillationParams, PhysicsBackend, SimulationConfig, SimulationSettings, SimulationState,
    SimulationTimeParams,
};
