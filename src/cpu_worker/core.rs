use std::time::Duration;
use web_time::Instant;

use super::protocol::{CpuSnapshot, MAX_STEPS_PER_REQUEST};
use crate::physics::{ContactHistory, ParticleStore, SpatialHashGrid, run_physics_step_plain};
use crate::simulation::{constants::SimulationConstants, state::SimulationState};

const WORK_BUDGET: Duration = Duration::from_millis(8);

/// Owned by the worker, including contact history between requests.
pub struct CpuWorkerSimulation {
    constants: SimulationConstants,
    particles: ParticleStore,
    state: SimulationState,
    contact_history: ContactHistory,
    grid: SpatialHashGrid,
}

impl CpuWorkerSimulation {
    pub fn new(
        constants: SimulationConstants,
        particles: ParticleStore,
        state: SimulationState,
    ) -> Self {
        let grid = SpatialHashGrid::new(constants.grid.cell_size, constants.grid.table_size);
        Self {
            constants,
            particles,
            state,
            contact_history: ContactHistory::default(),
            grid,
        }
    }

    pub fn snapshot(&self, completed_steps: u32) -> CpuSnapshot {
        CpuSnapshot {
            particles: self.particles.clone(),
            state: self.state.clone(),
            completed_steps,
        }
    }

    pub fn advance(&mut self, constants: SimulationConstants, steps: u32) -> CpuSnapshot {
        if self.constants.grid.cell_size != constants.grid.cell_size
            || self.constants.grid.table_size != constants.grid.table_size
        {
            self.grid = SpatialHashGrid::new(constants.grid.cell_size, constants.grid.table_size);
        }
        self.constants = constants;
        let started = Instant::now();
        let mut completed_steps = 0;
        for _ in 0..steps.min(MAX_STEPS_PER_REQUEST) {
            // The main thread sends Advance only while running. A paused
            // Initialize must also be able to resume on the first request.
            self.state.paused = false;
            run_physics_step_plain(
                &self.constants,
                &mut self.particles,
                &mut self.contact_history,
                &mut self.state,
                &mut self.grid,
            );
            completed_steps += 1;
            // Never interrupt a physical step, but return to the worker event
            // loop between steps so controls cannot queue behind a long batch.
            if started.elapsed() >= WORK_BUDGET {
                break;
            }
        }
        self.snapshot(completed_steps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::ParticleSize;
    use mikage::glam::Vec3;

    fn fixture() -> (SimulationConstants, ParticleStore, SimulationState) {
        let mut constants = SimulationConstants::default();
        constants.time.dt = 0.0004;
        constants.physics.gravity = Vec3::ZERO;
        let mut particles = ParticleStore::default();
        particles.spawn(
            Vec3::new(0.045, 0.1, 0.0),
            0.01,
            2500.0,
            ParticleSize::Large,
        );
        particles.spawn(
            Vec3::new(0.0645, 0.1, 0.0),
            0.01,
            2500.0,
            ParticleSize::Large,
        );
        particles.particles[0].velocity = Vec3::new(0.01, 0.02, 0.0);
        particles.particles[1].angular_velocity = Vec3::new(0.0, 0.0, 0.25);
        let state = SimulationState {
            elapsed: 7.25,
            oscillation_phase: 1.5,
            oscillation_frame_start_phase: 1.4,
            container_offset: 0.025,
            container_velocity: 0.1,
            ..Default::default()
        };
        (constants, particles, state)
    }

    #[test]
    fn worker_steps_match_direct_cpu_including_contact_history_and_parameter_edits() {
        let (mut constants, mut particles, mut state) = fixture();
        let mut worker =
            CpuWorkerSimulation::new(constants.clone(), particles.clone(), state.clone());
        let mut history = ContactHistory::default();
        let mut grid = SpatialHashGrid::new(constants.grid.cell_size, constants.grid.table_size);
        for step in 0..8 {
            if step == 3 {
                constants.oscillation.frequency = 7.5;
                constants.wall.friction = 0.41;
                constants.material.restitution = 0.83;
            }
            if step == 5 {
                constants.grid.cell_size *= 1.5;
                constants.grid.table_size /= 2;
                grid = SpatialHashGrid::new(constants.grid.cell_size, constants.grid.table_size);
            }
            run_physics_step_plain(
                &constants,
                &mut particles,
                &mut history,
                &mut state,
                &mut grid,
            );
            let actual = worker.advance(constants.clone(), 1);
            assert_eq!(actual.completed_steps, 1);
            assert_eq!(
                postcard::to_allocvec(&actual.particles).unwrap(),
                postcard::to_allocvec(&particles).unwrap()
            );
            assert_eq!(
                postcard::to_allocvec(&actual.state).unwrap(),
                postcard::to_allocvec(&state).unwrap()
            );
            assert_eq!(
                worker.contact_history.contacts.len(),
                history.contacts.len()
            );
            for (pair, contact) in &history.contacts {
                let actual = &worker.contact_history.contacts[pair];
                assert_eq!(actual.last_normal, contact.last_normal);
                assert_eq!(actual.active, contact.active);
            }
        }
    }

    #[test]
    fn initialization_keeps_paused_state_and_new_session_drops_old_contacts() {
        let (constants, particles, mut state) = fixture();
        state.paused = true;
        let mut worker =
            CpuWorkerSimulation::new(constants.clone(), particles.clone(), state.clone());
        let initial = worker.snapshot(0);
        assert!(initial.state.paused);
        assert_eq!(initial.state.elapsed, state.elapsed);
        assert_eq!(
            postcard::to_allocvec(&initial.particles).unwrap(),
            postcard::to_allocvec(&particles).unwrap()
        );
        let stepped = worker.advance(constants.clone(), 1);
        assert!(!stepped.state.paused);
        assert_eq!(
            stepped.state.elapsed,
            state.elapsed + f64::from(constants.time.dt)
        );
        assert!(!worker.contact_history.contacts.is_empty());

        let replacement = ParticleStore::default();
        let reset_worker =
            CpuWorkerSimulation::new(constants, replacement, SimulationState::default());
        assert!(reset_worker.contact_history.contacts.is_empty());
        assert!(reset_worker.snapshot(0).particles.particles.is_empty());
        assert_eq!(reset_worker.snapshot(0).state.elapsed, 0.0);
    }

    #[test]
    fn request_budget_cannot_run_more_than_four_steps() {
        let (constants, particles, state) = fixture();
        let initial_elapsed = state.elapsed;
        let mut worker = CpuWorkerSimulation::new(constants.clone(), particles, state);
        assert_eq!(worker.advance(constants.clone(), 0).completed_steps, 0);
        assert_eq!(worker.snapshot(0).state.elapsed, initial_elapsed);
        let result = worker.advance(constants.clone(), u32::MAX);
        assert!((1..=MAX_STEPS_PER_REQUEST).contains(&result.completed_steps));
        assert_eq!(
            result.state.elapsed,
            initial_elapsed + f64::from(constants.time.dt) * f64::from(result.completed_steps)
        );
    }
}
