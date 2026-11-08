use std::{cell::RefCell, rc::Rc, time::Duration};

use js_sys::{Array, Uint8Array};
use wasm_bindgen::{JsCast, JsValue, closure::Closure};
use web_sys::{ErrorEvent, MessageEvent, Worker, WorkerOptions, WorkerType};
use web_time::Instant;

use super::{
    pacing::WorkerPacing,
    protocol::{CpuSnapshot, WORKER_READY, WorkerRequest, WorkerResponse},
};
use crate::{
    physics::ParticleStore,
    simulation::{constants::SimulationConstants, state::SimulationState},
};

enum Incoming {
    Ready,
    Reply(WorkerResponse),
    Failed(String),
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Loading,
    Initializing,
    Idle,
    Advancing,
}

/// One worker, one outstanding request, and one completed snapshot at most.
/// Dropping a session cancels it, including replies from before a reset.
struct CpuWorker {
    session: Rc<RefCell<WorkerSession>>,
}

struct WorkerParameters {
    constants: SimulationConstants,
    paused: bool,
}

struct WorkerSession {
    worker: Worker,
    phase: Phase,
    initial: Option<Vec<u8>>,
    parameters: WorkerParameters,
    requests_enabled: bool,
    snapshot: Option<CpuSnapshot>,
    error: Option<String>,
    pacing: WorkerPacing,
    last_account: Instant,
    redraw: Option<mikage::RedrawRequest>,
    timer: Option<i32>,
    // Each callback holds only a Weak reference back to this session.
    on_timer: Option<Closure<dyn FnMut()>>,
    _on_message: Option<Closure<dyn FnMut(MessageEvent)>>,
    _on_error: Option<Closure<dyn FnMut(ErrorEvent)>>,
}

impl CpuWorker {
    pub fn new(
        constants: &SimulationConstants,
        particles: &ParticleStore,
        state: &SimulationState,
        redraw: Option<mikage::RedrawRequest>,
    ) -> Result<Self, String> {
        let initial = postcard::to_allocvec(&WorkerRequest::Initialize {
            constants: constants.clone(),
            particles: particles.clone(),
            state: state.clone(),
        })
        .map_err(|e| e.to_string())?;
        let options = WorkerOptions::new();
        options.set_type(WorkerType::Module);
        options.set_name("Granular Clock CPU");
        let worker =
            Worker::new_with_options("./cpu-worker_loader.js", &options).map_err(js_error)?;
        let session = Rc::new(RefCell::new(WorkerSession {
            worker,
            phase: Phase::Loading,
            initial: Some(initial),
            parameters: WorkerParameters {
                constants: constants.clone(),
                paused: state.paused,
            },
            requests_enabled: true,
            snapshot: None,
            error: None,
            pacing: WorkerPacing::default(),
            last_account: Instant::now(),
            redraw,
            timer: None,
            on_timer: None,
            _on_message: None,
            _on_error: None,
        }));

        let messages = Rc::downgrade(&session);
        let on_message = Closure::wrap(Box::new(move |event: MessageEvent| {
            let Some(session) = messages.upgrade() else {
                return;
            };
            let data = event.data();
            let incoming = if let Some(version) = data.as_string() {
                if version == WORKER_READY {
                    Incoming::Ready
                } else {
                    Incoming::Failed("CPU worker version mismatch; reload the page".into())
                }
            } else if data.is_instance_of::<js_sys::ArrayBuffer>() {
                match postcard::from_bytes(&Uint8Array::new(&data).to_vec()) {
                    Ok(reply) => Incoming::Reply(reply),
                    Err(error) => Incoming::Failed(format!("Invalid CPU worker reply: {error}")),
                }
            } else {
                Incoming::Failed("Unexpected CPU worker message".into())
            };
            let redraw = {
                let mut session = session.borrow_mut();
                if session.error.is_none()
                    && let Err(error) = session.receive(incoming)
                {
                    session.fail(error);
                }
                session.redraw.clone()
            };
            if let Some(request) = redraw {
                request.request_redraw();
            }
        }) as Box<dyn FnMut(MessageEvent)>);

        let errors = Rc::downgrade(&session);
        let on_error = Closure::wrap(Box::new(move |event: ErrorEvent| {
            let Some(session) = errors.upgrade() else {
                return;
            };
            let redraw = {
                let mut session = session.borrow_mut();
                session.fail(event.message());
                session.redraw.clone()
            };
            if let Some(request) = redraw {
                request.request_redraw();
            }
        }) as Box<dyn FnMut(ErrorEvent)>);

        let timers = Rc::downgrade(&session);
        let on_timer = Closure::wrap(Box::new(move || {
            let Some(session) = timers.upgrade() else {
                return;
            };
            let redraw = {
                let mut session = session.borrow_mut();
                session.timer = None;
                if let Err(error) = session.drive() {
                    session.fail(error);
                    session.redraw.clone()
                } else {
                    None
                }
            };
            if let Some(request) = redraw {
                request.request_redraw();
            }
        }) as Box<dyn FnMut()>);

        {
            let mut session = session.borrow_mut();
            session
                .worker
                .set_onmessage(Some(on_message.as_ref().unchecked_ref()));
            session
                .worker
                .set_onerror(Some(on_error.as_ref().unchecked_ref()));
            session._on_message = Some(on_message);
            session._on_error = Some(on_error);
            session.on_timer = Some(on_timer);
        }

        Ok(Self { session })
    }

    /// Rendering consumes the newest completed state without driving the worker.
    pub fn poll(&mut self) -> Result<Option<CpuSnapshot>, String> {
        let mut session = self.session.borrow_mut();
        if let Some(error) = session.error.take() {
            Err(error)
        } else {
            Ok(session.snapshot.take())
        }
    }

    /// A GPU handoff must let the outstanding computation finish before taking ownership.
    pub fn is_advancing(&self) -> bool {
        self.session.borrow().phase == Phase::Advancing
    }

    pub fn synchronize(
        &mut self,
        constants: &SimulationConstants,
        paused: bool,
    ) -> Result<(), String> {
        let mut session = self.session.borrow_mut();
        // Account with the previous parameters first, so resuming cannot add
        // time spent paused or stopped to the next request.
        session.account_elapsed();
        session.parameters = WorkerParameters {
            constants: constants.clone(),
            paused,
        };
        session.requests_enabled = true;
        session.drive()
    }

    pub fn stop_requests(&mut self) {
        self.session.borrow_mut().stop_requests();
    }

    pub fn set_redraw_request(&mut self, request: mikage::RedrawRequest) {
        self.session.borrow_mut().redraw = Some(request);
    }
}

impl WorkerSession {
    /// Initialization and subsequent requests advance from replies, not render frames.
    fn receive(&mut self, incoming: Incoming) -> Result<(), String> {
        self.account_elapsed();
        match incoming {
            Incoming::Ready if self.phase == Phase::Loading => {
                let initial = self.initial.take().expect("initial payload before Ready");
                self.post(&initial)?;
                self.phase = Phase::Initializing;
            }
            Incoming::Reply(WorkerResponse::Snapshot(snapshot))
                if matches!(self.phase, Phase::Initializing | Phase::Advancing) =>
            {
                self.phase = Phase::Idle;
                self.snapshot = Some(snapshot);
            }
            Incoming::Reply(WorkerResponse::Error(message)) | Incoming::Failed(message) => {
                return Err(message);
            }
            _ => return Err("Unexpected CPU worker response order".into()),
        }
        self.drive()
    }

    fn dt(&self) -> Duration {
        Duration::from_secs_f64(f64::from(self.parameters.constants.time.dt))
            .max(Duration::from_nanos(1))
    }

    fn account_elapsed(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_account);
        self.last_account = now;
        self.pacing.account(
            elapsed,
            self.dt(),
            self.requests_enabled
                && !self.parameters.paused
                && matches!(self.phase, Phase::Idle | Phase::Advancing),
        );
    }

    fn drive(&mut self) -> Result<(), String> {
        if let Some(error) = &self.error {
            return Err(error.clone());
        }
        self.account_elapsed();
        if !self.requests_enabled || self.parameters.paused {
            self.cancel_timer();
            return Ok(());
        }
        if self.phase != Phase::Idle {
            return Ok(());
        }
        let dt = self.dt();
        let steps = self.pacing.take_steps(dt);
        if steps == 0 {
            self.schedule_retry(dt)?;
            return Ok(());
        }
        self.cancel_timer();
        let data = postcard::to_allocvec(&WorkerRequest::Advance {
            constants: self.parameters.constants.clone(),
            steps,
        })
        .map_err(|e| e.to_string())?;
        self.post(&data)?;
        self.phase = Phase::Advancing;
        Ok(())
    }

    fn schedule_retry(&mut self, dt: Duration) -> Result<(), String> {
        if self.timer.is_some() {
            return Ok(());
        }
        let window = web_sys::window().ok_or("CPU worker requires a browser window")?;
        // A positive delay avoids spinning when replies consume less than one
        // physics step of wall time. Keep retries short when dt is large.
        let delay_ms = dt.as_nanos().div_ceil(1_000_000).clamp(1, 16) as i32;
        self.timer = Some(
            window
                .set_timeout_with_callback_and_timeout_and_arguments_0(
                    self.on_timer
                        .as_ref()
                        .expect("timer callback is installed before running")
                        .as_ref()
                        .unchecked_ref(),
                    delay_ms,
                )
                .map_err(js_error)?,
        );
        Ok(())
    }

    fn cancel_timer(&mut self) {
        if let Some(timer) = self.timer.take()
            && let Some(window) = web_sys::window()
        {
            window.clear_timeout_with_handle(timer);
        }
    }

    fn stop_requests(&mut self) {
        self.requests_enabled = false;
        self.cancel_timer();
        self.account_elapsed();
    }

    fn fail(&mut self, error: String) {
        self.stop_requests();
        self.error.get_or_insert(error);
    }

    fn post(&self, bytes: &[u8]) -> Result<(), String> {
        // Transfer a JS-owned buffer; WASM memory itself is never detached or shared.
        let buffer = Uint8Array::from(bytes).buffer();
        self.worker
            .post_message_with_transfer(&buffer, &Array::of1(&buffer))
            .map_err(js_error)
    }
}

impl Drop for CpuWorker {
    fn drop(&mut self) {
        let mut session = self.session.borrow_mut();
        session.stop_requests();
        session.worker.set_onmessage(None);
        session.worker.set_onerror(None);
        session.worker.terminate();
    }
}

fn js_error(value: JsValue) -> String {
    value.as_string().unwrap_or_else(|| format!("{value:?}"))
}

/// Browser-side lifecycle. A reset owns a new session, so no old reply can
/// overwrite newly spawned particles or a subsequent GPU simulation.
#[derive(Default)]
pub struct CpuWorkerController {
    worker: Option<CpuWorker>,
    error: Option<String>,
    redraw: Option<mikage::RedrawRequest>,
}

impl CpuWorkerController {
    pub fn set_redraw_request(&mut self, request: mikage::RedrawRequest) {
        if let Some(worker) = &mut self.worker {
            worker.set_redraw_request(request.clone());
        }
        self.redraw = Some(request);
    }

    pub fn reset(&mut self) {
        self.worker = None;
        self.error = None;
    }

    pub fn is_advancing(&self) -> bool {
        self.worker.as_ref().is_some_and(CpuWorker::is_advancing)
    }

    /// Stop issuing work while preserving the result of an outstanding request.
    pub fn stop_requests(&mut self) {
        if let Some(worker) = &mut self.worker {
            worker.stop_requests();
        }
    }

    pub fn error_message(&self) -> Option<&'static str> {
        self.error
            .as_ref()
            .map(|_| "CPU calculation stopped. Reset to retry, or switch to GPU.")
    }

    pub fn poll(&mut self, particles: &mut ParticleStore, state: &mut SimulationState) {
        let Some(worker) = &mut self.worker else {
            return;
        };
        match worker.poll() {
            Ok(Some(snapshot)) => snapshot.apply_to(particles, state),
            Ok(None) => {}
            Err(error) => self.fail(error, state),
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
        if self.worker.is_none() {
            match CpuWorker::new(constants, particles, state, self.redraw.clone()) {
                Ok(worker) => self.worker = Some(worker),
                Err(error) => {
                    self.fail(error, state);
                    return;
                }
            }
        }
        if let Err(error) = self
            .worker
            .as_mut()
            .unwrap()
            .synchronize(constants, state.paused)
        {
            self.fail(error, state);
        }
    }

    fn fail(&mut self, error: String, state: &mut SimulationState) {
        tracing::error!("CPU worker failed: {error}");
        self.worker = None;
        self.error = Some(error);
        state.paused = true;
        // GUI runs before polling during render preparation. With automatic
        // ticks paused, explicitly draw again to show the error and paused UI.
        if let Some(request) = &self.redraw {
            request.request_redraw();
        }
    }
}
