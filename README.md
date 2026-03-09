# Granular Clock Simulator

![Demo](./img/demo.gif "Particle distribution changing due to vibration")

A simulator for the granular clock phenomenon [1]. When a mixture of two particle sizes is placed in a box with a low partition and subjected to vibration, the particle distribution on each side of the partition periodically reverses.

This program uses physics simulation to model the particle dynamics and visualizes the results with [Bevy](https://bevyengine.org/).

**Live demo (runs in your browser):** https://granular.necocen.info/  
**Supported browsers:** Chrome, Firefox, and Safari with WebGPU support

## Getting Started

Requires Rust (2024 edition or newer).

### Running Locally

```sh
cargo run --release
```

Simulation parameters can be configured in `simulation.toml`. Once the binary is built, you can also run it directly with a different config file:

```sh
/path/to/granular-clock --config ./simulation.toml
```

### Running in the Browser

Install Web support tooling:

```sh
rustup target add wasm32-unknown-unknown
cargo install trunk
```

Then run:

```sh
trunk serve
```

This will open the simulation in your browser. Note that the build is quite slow due to WASM size optimization.

When running in the browser, custom `simulation.toml` files cannot be passed at runtime.

## Controls

### Camera

- `Left mouse button drag`: orbit
- `Right mouse button drag`: pan
- `Mouse wheel`: zoom

### UI (top-right panel)

- Switch physics backend: `GPU` / `CPU`
- Adjust `Substeps / Frame`
- Toggle oscillation and adjust amplitude/frequency
- Adjust divider height
- Adjust particle/wall contact parameters
- `Pause` / `Resume`, `Reset`

## References

1. R. Lambiotte, J.M. Salazar, L. Brenig — [From particle segregation to the granular clock](https://doi.org/10.1016/j.physleta.2005.06.006), Physics Letters A, 343, 2005, 224
