# Granular Clock Simulator

![Demo](./img/demo.gif "Particle distribution changing due to vibration")

A simulator for the granular clock phenomenon [1]. When a mixture of two particle sizes is placed in a box with a low partition and subjected to vibration, the particle distribution on each side of the partition periodically reverses.

This program uses physics simulation to model the particle dynamics and visualizes the results with [mikage](https://github.com/necocen/mikage).

**Live demo (runs in your browser):** https://granular.necocen.info/  
**Supported browsers:** Chrome, Firefox, and Safari with WebGPU support

## Getting Started

Requires Rust 1.95 or newer.

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

### Lighting

`[ui.light]` controls particle and container lighting at startup. The vector
toward the light is `directional_position - directional_target`. The legacy settings
`ambient_brightness` and `directional_illuminance` are independent intensities:
each value is divided by the fixed scale `10500` to obtain mikage's unitless linear
gain. These are not physical lux values. The defaults `500` and `10000` preserve
the previous particle lighting; changing one intensity no longer changes the other.
Scaling both by the same factor now scales brightness. Zero disables each light;
setting both to zero renders particles and container surfaces black. Gains above
one are supported, but output above the display range is clipped without tone mapping.

The transparent floor, divider, and walls opt into mikage's lit shading while
retaining their alpha blending. Shadows are unsupported;
`shadows_enabled = true` produces a configuration warning and is treated as false.

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
