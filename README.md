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

`[ui.light]` controls particle and container lighting at startup. `direction`
points toward the light and is normalized by mikage. `ambient_intensity` and
`directional_intensity` are independent, unitless linear gains, passed directly
to mikage. The defaults `0.4` and `0.8` keep shadowed faces visible while
oblique front lighting preserves the shape of each particle. Changing one
intensity does not change the other.
Scaling both by the same factor now scales brightness. Zero disables each light;
setting both to zero renders particles and container surfaces black.
Unknown keys in `[ui.light]` are rejected.

`ambient_color` and `directional_color` are independent linear RGB triples, defaulting
to white `[1.0, 1.0, 1.0]`. Convert sRGB/hex colors to linear RGB before using them.
Components must be finite and nonnegative; values above one are allowed.

`shadows_enabled = true` enables directional shadows from opaque particles onto
other particles and the transparent floor, divider, and walls. The transparent
container receives shadows but does not cast them. `shadow_map_resolution`
defaults to 1024 (valid configuration range: 1–8192, capped by device limits).
The light volume fits the container and oscillation amplitude automatically.
Unsupported devices use unshadowed lighting and report a warning.

`hdr_enabled = true` renders opaque and transparent geometry into a linear
`Rgba16Float` target, then applies `exposure_ev` and Reinhard tone mapping before
the GUI. Exposure defaults to +0.5 EV. Zero EV is unity; +1 doubles radiance and
−1 halves it (range: −32 to +32).
This preserves highlight differences above one instead of clipping them at the
scene pass. The mapper handles display sRGB encoding once; the GUI is unaffected
by exposure. HDR and shadows are enabled by default. Set `hdr_enabled = false`
for direct rendering with clipping and no exposure adjustment. Devices without
the required HDR target support use direct rendering and report a warning.

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
