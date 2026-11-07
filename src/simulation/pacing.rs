use std::time::Duration;

const MAX_CATCH_UP_TICKS: u32 = 4;

/// One timer tick is one physical step, independently of redraws.
/// The runner accounts for wall time and discards excessive catch-up work.
pub fn simulation_policy(physics_dt: f32) -> mikage::SimulationPolicy {
    let dt = Duration::from_secs_f64(f64::from(physics_dt)).max(Duration::from_nanos(1));
    mikage::SimulationPolicy::Fixed {
        dt,
        max_ticks_per_wake: MAX_CATCH_UP_TICKS as usize,
        max_backlog: dt * MAX_CATCH_UP_TICKS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timer_tick_matches_each_configured_physics_step() {
        for physics_dt in [0.0002_f32, 0.0004, 0.001] {
            let mikage::SimulationPolicy::Fixed {
                dt,
                max_ticks_per_wake,
                max_backlog,
            } = simulation_policy(physics_dt)
            else {
                panic!("physics must be scheduled independently of redraws");
            };
            assert!((dt.as_secs_f64() - f64::from(physics_dt)).abs() <= 0.5e-9);
            assert_eq!(max_ticks_per_wake, 4);
            assert_eq!(max_backlog, dt * 4);
        }
    }
}
