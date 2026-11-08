#[cfg(target_family = "wasm")]
fn main() {
    granular_clock::cpu_worker::serve();
}

#[cfg(not(target_family = "wasm"))]
fn main() {}
