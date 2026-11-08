use std::time::Duration;

use super::protocol::MAX_STEPS_PER_REQUEST;

/// Wall-time credit, independent of rendering and of mikage's tick counter.
/// A slow worker can accumulate at most one request, never an unbounded backlog.
#[derive(Default)]
pub(super) struct WorkerPacing {
    credit: Duration,
}

impl WorkerPacing {
    pub fn account(&mut self, elapsed: Duration, dt: Duration, running: bool) {
        self.credit = if running {
            self.credit
                .saturating_add(elapsed)
                .min(dt * MAX_STEPS_PER_REQUEST)
        } else {
            Duration::ZERO
        };
    }

    pub fn take_steps(&mut self, dt: Duration) -> u32 {
        let steps = (self.credit.as_nanos() / dt.as_nanos().max(1))
            .min(u128::from(MAX_STEPS_PER_REQUEST)) as u32;
        self.credit = self.credit.saturating_sub(dt * steps);
        steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_replies_cannot_advance_faster_than_wall_time() {
        let dt = Duration::from_micros(400);
        let mut pacing = WorkerPacing::default();
        let mut steps = 0;
        for _ in 0..1000 {
            pacing.account(Duration::from_micros(100), dt, true);
            steps += pacing.take_steps(dt);
        }
        assert_eq!(steps, 250);
    }

    #[test]
    fn slow_worker_and_background_tab_only_leave_one_batch_of_credit() {
        let dt = Duration::from_micros(400);
        let mut pacing = WorkerPacing::default();
        pacing.account(Duration::from_secs(60), dt, true);
        assert_eq!(pacing.take_steps(dt), MAX_STEPS_PER_REQUEST);
        assert_eq!(pacing.take_steps(dt), 0);
    }

    #[test]
    fn pause_and_loading_discard_credit() {
        let dt = Duration::from_micros(400);
        let mut pacing = WorkerPacing::default();
        pacing.account(dt * 4, dt, true);
        pacing.account(Duration::from_secs(60), dt, false);
        pacing.account(dt, dt, true);
        assert_eq!(pacing.take_steps(dt), 1);
    }
}
