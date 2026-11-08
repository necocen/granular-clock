use std::{cell::RefCell, rc::Rc};

use js_sys::{Array, Uint8Array};
use wasm_bindgen::{JsCast, JsValue, closure::Closure};
use web_sys::{DedicatedWorkerGlobalScope, MessageEvent};

use super::{
    core::CpuWorkerSimulation,
    protocol::{WORKER_READY, WorkerRequest, WorkerResponse},
};

/// Entry point of the separate CPU-only WASM binary. No window or GPU is initialized.
pub fn serve() {
    let scope: DedicatedWorkerGlobalScope = js_sys::global().unchecked_into();
    let simulation = Rc::new(RefCell::new(None::<CpuWorkerSimulation>));
    let reply_scope = scope.clone();
    let on_message = Closure::wrap(Box::new(move |event: MessageEvent| {
        let bytes = Uint8Array::new(&event.data()).to_vec();
        let request = postcard::from_bytes::<WorkerRequest>(&bytes);
        let response = match request {
            Ok(WorkerRequest::Initialize {
                constants,
                particles,
                state,
            }) => {
                let engine = CpuWorkerSimulation::new(constants, particles, state);
                let snapshot = engine.snapshot(0);
                *simulation.borrow_mut() = Some(engine);
                WorkerResponse::Snapshot(snapshot)
            }
            Ok(WorkerRequest::Advance { constants, steps }) => {
                if let Some(engine) = simulation.borrow_mut().as_mut() {
                    WorkerResponse::Snapshot(engine.advance(constants, steps))
                } else {
                    WorkerResponse::Error("CPU worker is not initialized".into())
                }
            }
            Err(error) => WorkerResponse::Error(format!("Invalid CPU worker request: {error}")),
        };
        let bytes = postcard::to_allocvec(&response).expect("serializable CPU snapshot");
        let buffer = Uint8Array::from(bytes.as_slice()).buffer();
        reply_scope
            .post_message_with_transfer(&buffer, &Array::of1(&buffer))
            .expect("post CPU snapshot");
    }) as Box<dyn FnMut(MessageEvent)>);
    scope.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    // The handler lives for the worker's lifetime; terminating the worker frees it.
    on_message.forget();
    scope
        .post_message(&JsValue::from_str(WORKER_READY))
        .expect("CPU worker ready");
}
