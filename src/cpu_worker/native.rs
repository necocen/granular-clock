use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
    thread::{self, JoinHandle},
    time::Duration,
};

use parking_lot::{Condvar, Mutex};
use web_time::Instant;

use super::{core::CpuWorkerSimulation, pacing::WorkerPacing, protocol::CpuSnapshot};
use crate::{
    physics::ParticleStore,
    simulation::{constants::SimulationConstants, state::SimulationState},
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Empty,
    Initializing,
    Idle,
    Advancing,
}

struct InitialState {
    constants: SimulationConstants,
    particles: ParticleStore,
    state: SimulationState,
}

struct WorkerParameters {
    constants: SimulationConstants,
    paused: bool,
    dt: Duration,
}

struct Mailbox {
    epoch: u64,
    phase: Phase,
    initial: Option<InitialState>,
    parameters: Option<WorkerParameters>,
    requests_enabled: bool,
    snapshot: Option<CpuSnapshot>,
    error: Option<String>,
    pacing: WorkerPacing,
    last_account: Instant,
    redraw: Option<mikage::RedrawRequest>,
    shutdown: bool,
}

impl Default for Mailbox {
    fn default() -> Self {
        Self {
            epoch: 0,
            phase: Phase::Empty,
            initial: None,
            parameters: None,
            requests_enabled: false,
            snapshot: None,
            error: None,
            pacing: WorkerPacing::default(),
            last_account: Instant::now(),
            redraw: None,
            shutdown: false,
        }
    }
}

impl Mailbox {
    fn account_at(&mut self, now: Instant) {
        let elapsed = now.saturating_duration_since(self.last_account);
        self.last_account = now;
        let dt = self
            .parameters
            .as_ref()
            .map_or(Duration::from_nanos(1), |parameters| parameters.dt);
        let running = self.requests_enabled
            && self
                .parameters
                .as_ref()
                .is_some_and(|parameters| !parameters.paused)
            && matches!(self.phase, Phase::Idle | Phase::Advancing);
        self.pacing.account(elapsed, dt, running);
    }

    fn synchronize(&mut self, parameters: WorkerParameters, now: Instant) {
        // Account with the old controls first. A pause/resume pair is observed
        // here even if both happen during one long physical step.
        self.account_at(now);
        self.parameters = Some(parameters);
        self.requests_enabled = true;
        self.account_at(now);
    }

    fn stop_requests(&mut self) {
        self.requests_enabled = false;
        self.account_at(Instant::now());
    }

    fn is_advancing(&self) -> bool {
        matches!(self.phase, Phase::Initializing | Phase::Advancing)
            || self.snapshot.is_some()
            || self.error.is_some()
    }
}

#[derive(Default)]
struct Shared {
    mailbox: Mutex<Mailbox>,
    changed: Condvar,
}

enum Work {
    Initialize(InitialState),
    Advance {
        constants: SimulationConstants,
        steps: u32,
    },
}

#[cfg(test)]
#[derive(Clone, Default)]
struct TestHooks {
    before_advance: Option<Arc<dyn Fn(u64, u32) + Send + Sync>>,
    on_exit: Option<Arc<dyn Fn() + Send + Sync>>,
}

fn worker_loop(shared: Arc<Shared>, #[cfg(test)] hooks: TestHooks) {
    let mut simulation = None::<(u64, CpuWorkerSimulation)>;
    loop {
        let (epoch, work) = {
            let mut mailbox = shared.mailbox.lock();
            loop {
                if mailbox.shutdown {
                    drop(mailbox);
                    #[cfg(test)]
                    if let Some(on_exit) = &hooks.on_exit {
                        on_exit();
                    }
                    return;
                }
                if let Some(initial) = mailbox.initial.take() {
                    break (mailbox.epoch, Work::Initialize(initial));
                }

                mailbox.account_at(Instant::now());
                let ready = mailbox.requests_enabled
                    && mailbox.error.is_none()
                    && mailbox.phase == Phase::Idle
                    && mailbox
                        .parameters
                        .as_ref()
                        .is_some_and(|parameters| !parameters.paused)
                    && simulation
                        .as_ref()
                        .is_some_and(|(epoch, _)| *epoch == mailbox.epoch);
                if !ready {
                    shared.changed.wait(&mut mailbox);
                    continue;
                }

                let parameters = mailbox.parameters.as_ref().expect("ready parameters");
                let dt = parameters.dt;
                let steps = mailbox.pacing.take_steps(dt);
                if steps == 0 {
                    // Condvar also wakes immediately for pause/reset/settings.
                    shared.changed.wait_for(&mut mailbox, dt);
                    continue;
                }
                let constants = mailbox
                    .parameters
                    .as_ref()
                    .expect("ready parameters")
                    .constants
                    .clone();
                mailbox.phase = Phase::Advancing;
                break (mailbox.epoch, Work::Advance { constants, steps });
            }
        };

        // Both the solver and snapshot cloning stay outside the mailbox lock.
        // Catch per request so Reset can recover without starting another thread.
        let result = catch_unwind(AssertUnwindSafe(|| match work {
            Work::Initialize(initial) => {
                let engine =
                    CpuWorkerSimulation::new(initial.constants, initial.particles, initial.state);
                let snapshot = engine.snapshot(0);
                simulation = Some((epoch, engine));
                snapshot
            }
            Work::Advance { constants, steps } => {
                #[cfg(test)]
                if let Some(before_advance) = &hooks.before_advance {
                    before_advance(epoch, steps);
                }
                simulation
                    .as_mut()
                    .expect("ready simulation")
                    .1
                    .advance(constants, steps)
            }
        }))
        .map_err(panic_message);
        if result.is_err() {
            simulation = None;
        }

        let redraw = {
            let mut mailbox = shared.mailbox.lock();
            if mailbox.shutdown || mailbox.epoch != epoch {
                // A completed old step cannot overwrite a reset or GPU state.
                None
            } else {
                mailbox.account_at(Instant::now());
                mailbox.phase = Phase::Idle;
                match result {
                    Ok(snapshot) => mailbox.snapshot = Some(snapshot),
                    Err(error) => {
                        mailbox.error = Some(error);
                        mailbox.stop_requests();
                    }
                }
                mailbox.redraw.clone()
            }
        };
        shared.changed.notify_all();
        if let Some(redraw) = redraw {
            redraw.request_redraw();
        }
    }
}

#[cfg(target_os = "macos")]
fn current_thread_qos() -> std::ffi::c_uint {
    use std::ffi::c_uint;

    const QOS_CLASS_UNSPECIFIED: c_uint = 0;
    const QOS_CLASS_DEFAULT: c_uint = 0x15;
    unsafe extern "C" {
        fn qos_class_self() -> c_uint;
    }

    // SAFETY: This macOS query reads only the calling thread's requested class
    // and takes no pointers. UNSPECIFIED cannot be passed to the setter.
    match unsafe { qos_class_self() } {
        QOS_CLASS_UNSPECIFIED => QOS_CLASS_DEFAULT,
        class => class,
    }
}

#[cfg(target_os = "macos")]
fn configure_thread_qos(class: std::ffi::c_uint) {
    use std::ffi::{c_int, c_uint};

    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(class: c_uint, relative_priority: c_int) -> c_int;
    }

    // SAFETY: This macOS API changes only the calling thread. The class came
    // from qos_class_self (or DEFAULT), zero is a valid relative priority, and
    // the call takes no pointers or ownership.
    let error = unsafe { pthread_set_qos_class_self_np(class, 0) };
    if error != 0 {
        tracing::warn!(
            "Could not set CPU worker QoS: {}",
            std::io::Error::from_raw_os_error(error)
        );
    }
}

fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .unwrap_or("unknown panic");
    format!("CPU worker panicked: {message}")
}

/// A persistent thread with one configuration, one initial state and one result.
/// Reset invalidates the current epoch immediately; it never joins a busy solver.
#[derive(Default)]
pub struct CpuWorkerController {
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
    initialized: bool,
    error: Option<String>,
    #[cfg(test)]
    hooks: TestHooks,
}

impl CpuWorkerController {
    pub fn set_redraw_request(&mut self, request: mikage::RedrawRequest) {
        self.shared.mailbox.lock().redraw = Some(request);
    }

    pub fn reset(&mut self) {
        {
            let mut mailbox = self.shared.mailbox.lock();
            mailbox.epoch = mailbox.epoch.wrapping_add(1);
            mailbox.phase = Phase::Empty;
            mailbox.initial = None;
            mailbox.parameters = None;
            mailbox.snapshot = None;
            mailbox.error = None;
            mailbox.stop_requests();
        }
        self.initialized = false;
        self.error = None;
        self.shared.changed.notify_one();
    }

    /// Also remains true for an unread final result. A reply racing with a
    /// poll followed by a GPU handoff must be consumed before ownership moves.
    pub fn is_advancing(&self) -> bool {
        self.shared.mailbox.lock().is_advancing()
    }

    pub fn stop_requests(&mut self) {
        self.shared.mailbox.lock().stop_requests();
        self.shared.changed.notify_one();
    }

    pub fn error_message(&self) -> Option<&'static str> {
        self.error
            .as_ref()
            .map(|_| "CPU calculation stopped. Reset to retry, or switch to GPU.")
    }

    pub fn poll(&mut self, particles: &mut ParticleStore, state: &mut SimulationState) {
        let (snapshot, error) = {
            let mut mailbox = self.shared.mailbox.lock();
            (mailbox.snapshot.take(), mailbox.error.take())
        };
        if let Some(snapshot) = snapshot {
            snapshot.apply_to(particles, state);
        }
        if let Some(error) = error {
            self.fail(error, state);
        }
    }

    pub fn advance(
        &mut self,
        constants: &SimulationConstants,
        particles: &ParticleStore,
        state: &mut SimulationState,
    ) {
        if self.error.is_some() {
            state.paused = true;
            return;
        }
        let dt = match Duration::try_from_secs_f64(f64::from(constants.time.dt)) {
            Ok(dt) => dt.max(Duration::from_nanos(1)),
            Err(error) => {
                self.fail(format!("Invalid CPU time step: {error}"), state);
                return;
            }
        };
        if let Err(error) = self.ensure_thread() {
            self.fail(error, state);
            return;
        }
        let initial = (!self.initialized).then(|| InitialState {
            constants: constants.clone(),
            particles: particles.clone(),
            state: state.clone(),
        });
        let error = {
            let mut mailbox = self.shared.mailbox.lock();
            if let Some(error) = mailbox.error.take() {
                Some(error)
            } else {
                if let Some(initial) = initial {
                    mailbox.initial = Some(initial);
                    mailbox.phase = Phase::Initializing;
                    self.initialized = true;
                }
                mailbox.synchronize(
                    WorkerParameters {
                        constants: constants.clone(),
                        paused: state.paused,
                        dt,
                    },
                    Instant::now(),
                );
                None
            }
        };
        if let Some(error) = error {
            self.fail(error, state);
        } else {
            self.shared.changed.notify_one();
        }
    }

    fn ensure_thread(&mut self) -> Result<(), String> {
        if self.thread.is_some() {
            return Ok(());
        }
        let shared = self.shared.clone();
        #[cfg(test)]
        let hooks = self.hooks.clone();
        // std::thread defaults to a different macOS QoS class. Preserve the
        // caller's scheduling class when moving its physics work off-thread.
        #[cfg(target_os = "macos")]
        let qos = current_thread_qos();
        let task = move || {
            #[cfg(target_os = "macos")]
            configure_thread_qos(qos);
            #[cfg(test)]
            worker_loop(shared, hooks);
            #[cfg(not(test))]
            worker_loop(shared);
        };
        self.thread = Some(
            thread::Builder::new()
                .name("Granular Clock CPU".into())
                .spawn(task)
                .map_err(|error| format!("Could not start CPU worker: {error}"))?,
        );
        Ok(())
    }

    fn fail(&mut self, error: String, state: &mut SimulationState) {
        tracing::error!("CPU worker failed: {error}");
        self.error = Some(error);
        state.paused = true;
        let redraw = {
            let mut mailbox = self.shared.mailbox.lock();
            mailbox.stop_requests();
            mailbox.redraw.clone()
        };
        self.shared.changed.notify_one();
        if let Some(redraw) = redraw {
            redraw.request_redraw();
        }
    }
}

impl Drop for CpuWorkerController {
    fn drop(&mut self) {
        {
            let mut mailbox = self.shared.mailbox.lock();
            mailbox.shutdown = true;
            mailbox.initial = None;
            mailbox.stop_requests();
        }
        self.shared.changed.notify_one();
        if let Some(thread) = self.thread.take()
            && thread.is_finished()
        {
            let _ = thread.join();
        }
        // A running step is not interruptible. Detaching lets the UI close
        // immediately; the thread sees shutdown after this one bounded batch.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::{self, Receiver, Sender};

    const TEST_TIMEOUT: Duration = Duration::from_secs(5);

    enum StepOutcome {
        Continue,
        Panic,
    }

    /// Blocks inside a physical request, after the mailbox lock is released.
    /// Test cleanup releases the gate even when an assertion fails.
    struct StepGate {
        entered: Receiver<(u64, u32)>,
        release: Sender<StepOutcome>,
        exited: Receiver<()>,
    }

    impl StepGate {
        fn entered(&self) -> (u64, u32) {
            self.entered.recv_timeout(TEST_TIMEOUT).unwrap()
        }

        fn release(&self) {
            self.release.send(StepOutcome::Continue).unwrap();
        }
    }

    impl Drop for StepGate {
        fn drop(&mut self) {
            let _ = self.release.send(StepOutcome::Continue);
        }
    }

    fn controlled_worker() -> (CpuWorkerController, StepGate) {
        let (entered_tx, entered) = mpsc::channel();
        let (release, release_rx) = mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let (exited_tx, exited) = mpsc::channel();
        let mut controller = CpuWorkerController::default();
        controller.hooks = TestHooks {
            before_advance: Some(Arc::new(move |epoch, steps| {
                let _ = entered_tx.send((epoch, steps));
                if let Ok(StepOutcome::Panic) = release_rx.lock().recv() {
                    panic!("injected physical step failure");
                }
            })),
            on_exit: Some(Arc::new(move || {
                let _ = exited_tx.send(());
            })),
        };
        (
            controller,
            StepGate {
                entered,
                release,
                exited,
            },
        )
    }

    fn fixture() -> (SimulationConstants, ParticleStore, SimulationState) {
        let mut constants = SimulationConstants::default();
        constants.grid.table_size = 64;
        (
            constants,
            ParticleStore::default(),
            SimulationState::default(),
        )
    }

    fn wait_for(controller: &CpuWorkerController, predicate: impl Fn(&Mailbox) -> bool) {
        let deadline = Instant::now() + TEST_TIMEOUT;
        let mut mailbox = controller.shared.mailbox.lock();
        while !predicate(&mailbox) {
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "worker did not reach expected state");
            controller.shared.changed.wait_for(&mut mailbox, remaining);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn worker_inherits_the_callers_requested_qos() {
        unsafe extern "C" {
            fn qos_class_self() -> std::ffi::c_uint;
        }

        // Dedicated callers avoid changing other tests' scheduling. Cover both
        // DEFAULT and the interactive window class, so a fixed class or removal
        // of propagation cannot accidentally satisfy the test.
        for caller_qos in [0x15, 0x21] {
            thread::spawn(move || {
                configure_thread_qos(caller_qos);
                assert_eq!(current_thread_qos(), caller_qos);
                let (qos_tx, qos_rx) = mpsc::channel();
                let mut controller = CpuWorkerController::default();
                controller.hooks.before_advance = Some(Arc::new(move |_, _| {
                    // SAFETY: This query reads only the current worker's class and
                    // takes no pointers. Read the actual value without normalizing it.
                    let worker_qos = unsafe { qos_class_self() };
                    let _ = qos_tx.send(worker_qos);
                }));
                let (constants, particles, mut state) = fixture();
                controller.advance(&constants, &particles, &mut state);
                let worker_qos = qos_rx.recv_timeout(TEST_TIMEOUT);
                controller.stop_requests();
                controller.reset();
                drop(controller);
                assert_eq!(worker_qos.unwrap(), caller_qos);
            })
            .join()
            .unwrap();
        }
    }

    #[test]
    fn pause_and_resume_discard_stopped_wall_time() {
        let start = Instant::now();
        let dt = Duration::from_millis(10);
        let parameters = |paused| WorkerParameters {
            constants: SimulationConstants::default(),
            paused,
            dt,
        };
        let mut mailbox = Mailbox {
            phase: Phase::Idle,
            last_account: start,
            ..Default::default()
        };
        mailbox.synchronize(parameters(false), start);
        mailbox.account_at(start + Duration::from_millis(5));
        assert_eq!(mailbox.pacing.take_steps(dt), 0);
        mailbox.synchronize(parameters(true), start + Duration::from_millis(5));

        let resumed = start + Duration::from_secs(60);
        mailbox.synchronize(parameters(false), resumed);
        mailbox.account_at(resumed + Duration::from_millis(9));
        assert_eq!(mailbox.pacing.take_steps(dt), 0);
        mailbox.account_at(resumed + dt);
        assert_eq!(mailbox.pacing.take_steps(dt), 1);

        mailbox.stop_requests();
        let restarted = resumed + Duration::from_secs(60);
        mailbox.synchronize(parameters(false), restarted);
        assert_eq!(mailbox.pacing.take_steps(dt), 0);
        mailbox.account_at(restarted + dt);
        assert_eq!(mailbox.pacing.take_steps(dt), 1);
    }

    #[test]
    fn completed_snapshot_blocks_handoff_until_polled() {
        let (mut controller, gate) = controlled_worker();
        let (constants, mut particles, mut state) = fixture();
        controller.advance(&constants, &particles, &mut state);
        gate.entered();
        controller.stop_requests();
        controller.poll(&mut particles, &mut state);
        assert_eq!(state.elapsed, 0.0);
        assert!(controller.is_advancing());

        // Finish exactly between the main thread's poll and busy check.
        gate.release();
        wait_for(&controller, |mailbox| mailbox.phase == Phase::Idle);
        assert!(controller.is_advancing(), "final snapshot is still unread");
        controller.poll(&mut particles, &mut state);
        assert!(state.elapsed > 0.0);
        assert!(!controller.is_advancing());
    }

    #[test]
    fn worker_progresses_without_polling_and_keeps_latest_snapshot() {
        let (mut controller, gate) = controlled_worker();
        let (constants, mut particles, mut state) = fixture();
        controller.advance(&constants, &particles, &mut state);
        gate.entered();
        gate.release();
        gate.entered();
        let first = controller
            .shared
            .mailbox
            .lock()
            .snapshot
            .as_ref()
            .unwrap()
            .state
            .elapsed;
        gate.release();
        gate.entered();
        controller.poll(&mut particles, &mut state);
        assert!(state.elapsed > first);
        assert!(controller.shared.mailbox.lock().snapshot.is_none());
        controller.stop_requests();
        gate.release();
        wait_for(&controller, |mailbox| mailbox.phase == Phase::Idle);
    }

    #[test]
    fn reset_while_step_is_blocked_is_nonblocking_and_reuses_thread() {
        let (mut controller, gate) = controlled_worker();
        let (constants, mut particles, mut state) = fixture();
        controller.advance(&constants, &particles, &mut state);
        let (old_epoch, _) = gate.entered();
        let worker_id = controller.thread.as_ref().unwrap().thread().id();

        let (returned_tx, returned) = mpsc::channel();
        let operations = thread::spawn(move || {
            controller.stop_requests();
            controller.poll(&mut particles, &mut state);
            assert!(controller.is_advancing());
            for generation in 0_u32..20 {
                controller.reset();
                state.paused = true;
                state.elapsed = 100.0 + f64::from(generation);
                particles.generation = generation.into();
                controller.advance(&constants, &particles, &mut state);
            }
            assert!(returned_tx.send((controller, particles, state)).is_ok());
        });
        let outcome = returned.recv_timeout(TEST_TIMEOUT);
        if outcome.is_err() {
            gate.release();
        }
        let (mut controller, mut particles, mut state) =
            outcome.expect("UI operations waited for the blocked solver");
        operations.join().unwrap();
        assert_eq!(controller.thread.as_ref().unwrap().thread().id(), worker_id);
        assert_ne!(controller.shared.mailbox.lock().epoch, old_epoch);
        gate.release();
        wait_for(&controller, |mailbox| {
            mailbox.phase == Phase::Idle && mailbox.snapshot.is_some()
        });
        controller.poll(&mut particles, &mut state);
        assert_eq!(state.elapsed, 119.0);
        assert_eq!(particles.generation, 19);
        assert!(state.paused);
        assert!(!controller.is_advancing());
        assert!(
            gate.entered
                .recv_timeout(Duration::from_millis(20))
                .is_err()
        );
    }

    #[test]
    fn latest_snapshot_survives_following_panic_and_reset_recovers() {
        let (mut controller, gate) = controlled_worker();
        let (constants, mut particles, mut state) = fixture();
        controller.advance(&constants, &particles, &mut state);
        gate.entered();
        let worker_id = controller.thread.as_ref().unwrap().thread().id();
        controller.poll(&mut particles, &mut state);
        gate.release();
        gate.entered();
        let last_elapsed = controller
            .shared
            .mailbox
            .lock()
            .snapshot
            .as_ref()
            .unwrap()
            .state
            .elapsed;
        gate.release.send(StepOutcome::Panic).unwrap();
        wait_for(&controller, |mailbox| mailbox.error.is_some());
        assert!(controller.is_advancing());
        controller.poll(&mut particles, &mut state);
        assert!(state.paused);
        assert!(controller.error_message().is_some());
        assert_eq!(state.elapsed, last_elapsed);
        assert!(!controller.is_advancing());

        controller.reset();
        state.elapsed = 1000.0;
        controller.advance(&constants, &particles, &mut state);
        wait_for(&controller, |mailbox| mailbox.snapshot.is_some());
        controller.poll(&mut particles, &mut state);
        assert_eq!(state.elapsed, 1000.0);
        assert!(controller.error_message().is_none());
        assert_eq!(controller.thread.as_ref().unwrap().thread().id(), worker_id);
    }

    #[test]
    fn drop_does_not_join_a_blocked_step_and_thread_exits_afterward() {
        let (mut controller, gate) = controlled_worker();
        let (constants, particles, mut state) = fixture();
        controller.advance(&constants, &particles, &mut state);
        gate.entered();
        let (dropped_tx, dropped) = mpsc::channel();
        let closing = thread::spawn(move || {
            drop(controller);
            dropped_tx.send(()).unwrap();
        });
        let outcome = dropped.recv_timeout(TEST_TIMEOUT);
        if outcome.is_err() {
            gate.release();
        }
        outcome.expect("controller Drop waited for the blocked solver");
        closing.join().unwrap();
        assert!(gate.exited.try_recv().is_err());
        gate.release();
        gate.exited.recv_timeout(TEST_TIMEOUT).unwrap();
    }
}
