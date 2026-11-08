//! CPU physics offloaded to a worker. The renderer only reads completed snapshots.

mod core;
mod pacing;
mod protocol;

#[cfg(target_family = "wasm")]
mod client;
#[cfg(not(target_family = "wasm"))]
mod native;
#[cfg(target_family = "wasm")]
mod server;

#[cfg(target_family = "wasm")]
pub use client::CpuWorkerController;
#[cfg(not(target_family = "wasm"))]
pub use native::CpuWorkerController;
#[cfg(target_family = "wasm")]
pub use server::serve;
