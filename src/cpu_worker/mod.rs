//! CPU physics offloaded to a browser worker. The renderer only reads completed snapshots.

#[cfg(any(target_family = "wasm", test))]
mod core;
#[cfg(any(target_family = "wasm", test))]
mod pacing;
#[cfg(any(target_family = "wasm", test))]
mod protocol;

#[cfg(target_family = "wasm")]
mod client;
#[cfg(target_family = "wasm")]
mod server;

#[cfg(target_family = "wasm")]
pub use client::CpuWorkerController;
#[cfg(target_family = "wasm")]
pub use server::serve;
