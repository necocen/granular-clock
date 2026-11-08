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

Lengths in `simulation.toml` are in **millimeters**: particle radii, box size,
divider height/thickness, oscillation amplitude, camera position/target, and
length slider ranges/steps. The UI also displays and edits lengths in mm.
For example, a 10 mm radius is `large_radius = 10.0` and a 400 × 700 × 100 mm
box is `size = [400.0, 700.0, 100.0]`. Older meter-based configs need these
values multiplied by 1000. Density remains kg/m³, gravity m/s², Young's modulus
Pa, wall stiffness N/m, damping N·s/m, frequency Hz, and time s. Internal physics
calculations continue to use meters.

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

`[ui.light]` has three startup settings: `ambient_intensity` (default `0.4`),
`directional_intensity` (default `0.8`), and `shadows_enabled` (default `true`).
The intensities are independent, unitless linear gains passed directly to mikage;
zero disables a contribution. Output above the display range is clipped.
Unknown keys in this section are rejected.

The simulation uses fixed white lights from an oblique front direction and a
1024² shadow map. The light volume fits the container and oscillation automatically.
Transparent container pieces receive shadows but do not cast them. Devices without
shadow support use unshadowed lighting and report a warning.

## Controls

### Camera

- `Left mouse button drag`: orbit
- `Right mouse button drag`: pan
- `Mouse wheel`: zoom

### UI (top-right panel)

- Switch physics backend: `GPU` / `CPU`
- Toggle oscillation and adjust amplitude/frequency
- Adjust divider height
- Adjust particle/wall contact parameters
- `Pause` / `Resume`, `Reset`

Physics ticks use a fixed clock, while mikage 0.9's VSync presentation policy
updates the display independently. On native builds, surface acquisition runs
on a worker so physics can continue while waiting for the next frame.
Simulation time is limited to real time; slower hardware runs below 1x.
GPU mode performs one physics step of duration `simulation.step.dt` per fixed tick.
Catch-up work is bounded to four ticks
(1.6 ms with the default `dt = 0.0004`), so a stall or pause cannot build an
unlimited backlog. There is no adjustable substep cap.

CPU physics runs in a dedicated native thread or browser Web Worker. Rendering
uses the latest completed snapshot without waiting for computation. A single
worker advances at most four fixed steps per batch, with an 8 ms work budget
checked between steps. It schedules subsequent work independently of rendering
and keeps at most one completed snapshot. mikage's automatic ticks are suspended
in CPU mode to avoid empty GPU submissions, and resumed for GPU mode.
Pause stops new work; Reset discards the previous simulation session without
waiting for its computation. Switching back to GPU finishes outstanding CPU
work and transfers its latest state. Worker failure pauses the simulation and
offers Reset or GPU mode as recovery options. Native headless callers retain
one CPU step per explicit tick for deterministic batch execution.

Particle-to-instance conversion and display readback run during rendering,
instead of every physics tick. Switching from GPU to CPU waits for a fresh
asynchronous snapshot, including when no display frames are being drawn.
Redraw requests coalesce until a frame is available, without a fixed frame-rate
cap or a fixed number of physics steps per frame.

`Speed` shows recent simulation seconds per real second, for example `0.30x`.
FPS and steps/s are measured from elapsed time and actual progress, including
frames where the simulation does not advance.

## References

1. R. Lambiotte, J.M. Salazar, L. Brenig — [From particle segregation to the granular clock](https://doi.org/10.1016/j.physleta.2005.06.006), Physics Letters A, 343, 2005, 224
