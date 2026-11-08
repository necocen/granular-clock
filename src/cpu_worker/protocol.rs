use serde::{Deserialize, Serialize};

use crate::physics::ParticleStore;
#[cfg(any(target_family = "wasm", test))]
use crate::simulation::constants::SimulationConstants;
use crate::simulation::state::SimulationState;

/// The loader sends this only after WASM has installed its message handler.
#[cfg(target_family = "wasm")]
pub const WORKER_READY: &str = "granular-clock-cpu-v1";
pub const MAX_STEPS_PER_REQUEST: u32 = 4;

#[derive(Serialize, Deserialize)]
#[cfg(any(target_family = "wasm", test))]
pub enum WorkerRequest {
    Initialize {
        constants: SimulationConstants,
        particles: ParticleStore,
        state: SimulationState,
    },
    Advance {
        constants: SimulationConstants,
        steps: u32,
    },
}

#[derive(Serialize, Deserialize)]
#[cfg(any(target_family = "wasm", test))]
pub enum WorkerResponse {
    Snapshot(CpuSnapshot),
    Error(String),
}

/// Particle data and the moving container always belong to the same step.
#[derive(Clone, Serialize, Deserialize)]
pub struct CpuSnapshot {
    pub particles: ParticleStore,
    pub state: SimulationState,
    pub completed_steps: u32,
}

impl CpuSnapshot {
    /// UI commands may have changed while this snapshot was being computed.
    pub fn apply_to(self, particles: &mut ParticleStore, state: &mut SimulationState) {
        let paused = state.paused;
        let reset_requested = state.reset_requested;
        *particles = self.particles;
        *state = self.state;
        state.paused = paused;
        state.reset_requested = reset_requested;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::ParticleSize;
    use mikage::glam::Vec3;

    #[test]
    fn initialization_round_trip_preserves_runtime_parameters() {
        let mut constants = SimulationConstants::default();
        constants.oscillation.frequency = 23.0;
        constants.container.divider_height = 0.0875;
        constants.material.friction = 0.509;
        let request = WorkerRequest::Initialize {
            constants,
            particles: ParticleStore::default(),
            state: SimulationState {
                elapsed: 1.23,
                ..Default::default()
            },
        };
        let encoded = postcard::to_allocvec(&request).unwrap();
        let WorkerRequest::Initialize {
            constants, state, ..
        } = postcard::from_bytes(&encoded).unwrap()
        else {
            panic!("expected initialization");
        };
        assert_eq!(constants.oscillation.frequency, 23.0);
        assert_eq!(constants.container.divider_height, 0.0875);
        assert_eq!(constants.material.friction, 0.509);
        assert_eq!(state.elapsed, 1.23);
    }

    #[test]
    fn snapshot_round_trip_preserves_dynamics_and_pending_ui_commands() {
        let mut particles = ParticleStore::default();
        particles.spawn(Vec3::new(0.1, 0.2, 0.3), 0.01, 2500.0, ParticleSize::Large);
        let particle = &mut particles.particles[0];
        particle.velocity = Vec3::new(-0.3, 0.4, 0.5);
        particle.angular_velocity = Vec3::new(1.0, 2.0, 3.0);
        particle.force = Vec3::new(0.01, 0.02, 0.03);
        particle.torque = Vec3::new(-0.01, -0.02, -0.03);
        particles.generation = 47;
        let snapshot = CpuSnapshot {
            particles,
            state: SimulationState {
                elapsed: 12.3456789,
                oscillation_phase: 2.3,
                oscillation_frame_start_phase: 2.2,
                container_offset: 0.012,
                container_velocity: -0.71,
                ..Default::default()
            },
            completed_steps: 3,
        };
        let expected = postcard::to_allocvec(&snapshot).unwrap();
        let encoded = postcard::to_allocvec(&WorkerResponse::Snapshot(snapshot)).unwrap();
        let WorkerResponse::Snapshot(decoded) = postcard::from_bytes(&encoded).unwrap() else {
            panic!("expected snapshot");
        };
        assert_eq!(postcard::to_allocvec(&decoded).unwrap(), expected);

        let mut display_particles = ParticleStore::default();
        let mut display_state = SimulationState {
            paused: true,
            reset_requested: true,
            ..Default::default()
        };
        decoded.apply_to(&mut display_particles, &mut display_state);
        assert!(display_state.paused);
        assert!(display_state.reset_requested);
        assert_eq!(display_state.elapsed, 12.3456789);
        assert_eq!(display_state.oscillation_phase, 2.3);
        assert_eq!(display_state.oscillation_frame_start_phase, 2.2);
        assert_eq!(display_state.container_offset, 0.012);
        assert_eq!(display_state.container_velocity, -0.71);
        assert_eq!(display_particles.generation, 47);
        assert_eq!(
            display_particles.particles[0].velocity,
            Vec3::new(-0.3, 0.4, 0.5)
        );
        assert_eq!(
            display_particles.particles[0].angular_velocity,
            Vec3::new(1.0, 2.0, 3.0)
        );
    }
}
