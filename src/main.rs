mod analysis;
mod physics;
mod rendering;
mod simulation;
mod ui;

#[cfg(not(target_family = "wasm"))]
use clap::Parser;
use std::path::PathBuf;

use analysis::{CurrentDistribution, DistributionHistory, update_distribution_plain};
use physics::{
    ContactHistory, ParticleSize, ParticleStore,
    cpu::{SpatialHashGrid, run_physics_substeps_plain},
    gpu::{GpuPhysicsState, ParticleGpu},
};
use rendering::{ContainerRenderer, LARGE_PARTICLE_COLOR, SMALL_PARTICLE_COLOR};
use simulation::{
    config_toml::{LoadedConfig, resolve_startup_config},
    constants::{PhysicsBackend, SimulationConstants, UiControlRanges, advance_oscillation},
    state::SimulationState,
};
use ui::{PerfDisplayState, draw_control_panel, draw_distribution_panel};

use mikage::winit::dpi::PhysicalSize;
use mikage::{InstanceData, InstanceRenderer, InstanceRendererConfig, wgpu};

#[cfg(not(target_family = "wasm"))]
#[derive(Debug, Parser)]
#[command(name = "granular-clock")]
struct CliArgs {
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,
}

struct RenderResources {
    instance_renderer: InstanceRenderer,
    scene: mikage::SceneBinding,
    depth_view: wgpu::TextureView,
    container: ContainerRenderer,
}

struct GranularClockApp {
    constants: SimulationConstants,
    ui_ranges: UiControlRanges,
    sim_state: SimulationState,
    particle_store: ParticleStore,
    contact_history: ContactHistory,
    spatial_hash: SpatialHashGrid,
    backend: PhysicsBackend,
    prev_backend: PhysicsBackend,
    distribution: CurrentDistribution,
    distribution_history: DistributionHistory,
    dt: f32,
    perf_display: PerfDisplayState,
    // GPU rendering (Option because created in init())
    render: Option<RenderResources>,
    // GPU physics (Option because created in init())
    gpu_physics: Option<GpuPhysicsState>,
    /// GPU 初回アップロード済みフラグ
    gpu_particles_uploaded: bool,
    /// ParticleStore の世代追跡（GPU アップロード用）
    last_gpu_store_generation: u64,
}

impl GranularClockApp {
    fn new(loaded: LoadedConfig) -> Self {
        let constants = loaded.simulation;
        let grid_settings = constants.grid;
        let spatial_hash = SpatialHashGrid::new(grid_settings.cell_size, grid_settings.table_size);

        Self {
            constants,
            ui_ranges: loaded.ui_ranges,
            sim_state: SimulationState::default(),
            particle_store: ParticleStore::default(),
            contact_history: ContactHistory::default(),
            spatial_hash,
            backend: PhysicsBackend::Gpu,
            prev_backend: PhysicsBackend::Gpu,
            distribution: CurrentDistribution::default(),
            distribution_history: DistributionHistory::default(),
            dt: 0.0,
            perf_display: PerfDisplayState::default(),
            render: None,
            gpu_physics: None,
            gpu_particles_uploaded: false,
            last_gpu_store_generation: 0,
        }
    }

    /// Spawn particles into the ParticleStore (replicates spawn_particles from materials.rs)
    fn spawn_particles(&mut self) {
        use rand::Rng;
        let mut rng = rand::rng();

        let particle = &self.constants.particle;
        let container = &self.constants.container;
        let material = &self.constants.material;

        let spawn_area_x = container.half_extents.x - particle.large_radius;
        let spawn_area_z = container.half_extents.z - particle.large_radius;
        let base_y = container.base_position.y - container.half_extents.y;

        // Spawn large particles
        for _ in 0..particle.num_large {
            let x = rng.random_range(-spawn_area_x..spawn_area_x);
            let z = rng.random_range(-spawn_area_z..spawn_area_z);
            let y = base_y + particle.large_radius + rng.random_range(0.0..0.2);

            self.particle_store.spawn(
                glam::Vec3::new(x, y, z),
                particle.large_radius,
                material.density,
                ParticleSize::Large,
            );
        }

        // Spawn small particles
        for _ in 0..particle.num_small {
            let x = rng.random_range(-spawn_area_x..spawn_area_x);
            let z = rng.random_range(-spawn_area_z..spawn_area_z);
            let y = base_y + particle.small_radius + rng.random_range(0.0..0.2);

            self.particle_store.spawn(
                glam::Vec3::new(x, y, z),
                particle.small_radius,
                material.density,
                ParticleSize::Small,
            );
        }
    }

    /// Build instance data from particles (replicates CPU instance writer logic)
    fn build_instance_data(&self) -> Vec<InstanceData> {
        self.particle_store
            .particles
            .iter()
            .map(|p| {
                let color = match p.size {
                    ParticleSize::Large => LARGE_PARTICLE_COLOR,
                    ParticleSize::Small => SMALL_PARTICLE_COLOR,
                };
                InstanceData {
                    pos_scale: [p.position.x, p.position.y, p.position.z, p.radius],
                    color,
                }
            })
            .collect()
    }

    /// Convert ParticleStore to GPU format
    fn particles_to_gpu(&self) -> Vec<ParticleGpu> {
        self.particle_store
            .particles
            .iter()
            .map(|p| ParticleGpu {
                pos: p.position.into(),
                radius: p.radius,
                vel: p.velocity.into(),
                mass_inv: 1.0 / p.mass,
                omega: p.angular_velocity.into(),
                inertia_inv: 1.0 / p.inertia,
                size_flag: match p.size {
                    ParticleSize::Large => 1,
                    ParticleSize::Small => 0,
                },
                _pad: [0; 3],
            })
            .collect()
    }

    /// Handle reset request: clear particles and respawn.
    fn handle_reset(&mut self) {
        if !self.sim_state.reset_requested {
            return;
        }
        self.sim_state.reset_requested = false;
        self.sim_state.reset_time();
        self.distribution_history.clear();
        self.contact_history = ContactHistory::default();
        self.particle_store.clear();
        self.spawn_particles();
        // GPU particles will be re-uploaded in next update
        self.gpu_particles_uploaded = false;
    }

    /// Upload current particles to GPU physics state
    fn upload_particles_to_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let gpu_particles = self.particles_to_gpu();
        if gpu_particles.is_empty() {
            return;
        }

        let generation = self.particle_store.generation;

        let (Some(render), Some(gpu)) = (self.render.as_mut(), self.gpu_physics.as_mut()) else {
            return;
        };

        gpu.upload_particles(
            device,
            queue,
            &gpu_particles,
            generation,
            render.instance_renderer.instance_buffer(),
        );
        self.last_gpu_store_generation = generation;
        self.gpu_particles_uploaded = true;
    }
}

impl mikage::App for GranularClockApp {
    fn init(&mut self, ctx: &mikage::GpuContext, size: PhysicalSize<u32>) {
        let device = &ctx.device;

        let scene = mikage::SceneBinding::new(device);

        // Create instance renderer with IcoSphere mesh
        let sphere = mikage::IcoSphereMesh::generate(2);
        let mut instance_renderer = InstanceRenderer::new(
            device,
            ctx.render_format(),
            scene.layout(),
            &sphere.positions,
            &sphere.normals,
            &sphere.indices,
            InstanceRendererConfig::default_3d(),
        );

        // Pre-allocate instance buffer for expected particle count
        let total_particles = self.constants.particle.num_large + self.constants.particle.num_small;
        instance_renderer.ensure_capacity(device, total_particles);

        // Create depth texture
        let (_depth_texture, depth_view) =
            mikage::create_depth_texture(device, size, mikage::DEPTH_FORMAT);

        // Create container renderer (shares the scene bind group layout)
        let container = ContainerRenderer::new(device, ctx.render_format(), scene.layout());

        // Create GPU physics state
        let gpu_physics =
            GpuPhysicsState::new(device, &self.constants, instance_renderer.instance_buffer());

        self.render = Some(RenderResources {
            instance_renderer,
            scene,
            depth_view,
            container,
        });

        self.gpu_physics = Some(gpu_physics);

        // Spawn particles
        self.spawn_particles();
    }

    fn update(&mut self, ctx: &mut mikage::UpdateContext) {
        // Store dt for UI
        self.dt = ctx.dt;

        // Handle reset request (from UI)
        self.handle_reset();

        // Detect backend switch
        let backend_switched = self.backend != self.prev_backend;
        if backend_switched {
            if self.backend == PhysicsBackend::Gpu {
                // CPU -> GPU: upload current particle state
                self.gpu_particles_uploaded = false;
            }
            self.prev_backend = self.backend;
        }

        match self.backend {
            PhysicsBackend::Cpu => {
                // Run CPU physics substeps (unless paused)
                run_physics_substeps_plain(
                    &self.constants,
                    &mut self.particle_store,
                    &mut self.contact_history,
                    &mut self.sim_state,
                    &mut self.spatial_hash,
                );

                // Build instance data from particles and upload
                let instances = self.build_instance_data();

                if let Some(render) = self.render.as_mut() {
                    let reallocated = render
                        .instance_renderer
                        .ensure_capacity(&ctx.gpu.device, instances.len() as u32);
                    if reallocated && let Some(gpu) = self.gpu_physics.as_mut() {
                        gpu.rebuild_instance_bind_group(
                            &ctx.gpu.device,
                            render.instance_renderer.instance_buffer(),
                        );
                    }
                    render.instance_renderer.update_instances(
                        &ctx.gpu.device,
                        &ctx.gpu.queue,
                        &instances,
                    );
                }
            }
            PhysicsBackend::Gpu => {
                // Check if we need to upload particles (initial or after reset)
                let store_gen_changed =
                    self.particle_store.generation != self.last_gpu_store_generation;
                if !self.gpu_particles_uploaded || store_gen_changed {
                    self.upload_particles_to_gpu(&ctx.gpu.device, &ctx.gpu.queue);
                }

                // Process readback from previous frame
                if let Some(gpu) = self.gpu_physics.as_mut() {
                    gpu.process_readback(&ctx.gpu.device);
                    gpu.apply_readback_to_store(&mut self.particle_store);
                }

                if !self.sim_state.paused {
                    // Update GPU params
                    if let Some(gpu) = self.gpu_physics.as_mut() {
                        gpu.update_params(&ctx.gpu.queue, &self.constants, &self.sim_state);
                    }

                    // Advance simulation time (GPU computes physics, we just track time)
                    for _ in 0..self.constants.settings.substeps_per_frame {
                        self.sim_state.step_time(self.constants.time.dt);
                    }

                    // Advance oscillation phase (same logic as CPU side)
                    self.sim_state.oscillation_frame_start_phase = self.sim_state.oscillation_phase;
                    for _ in 0..self.constants.settings.substeps_per_frame {
                        advance_oscillation(
                            &mut self.sim_state,
                            &self.constants.oscillation,
                            self.constants.time.dt,
                        );
                    }
                }

                // Set instance count for rendering (GPU compute writes instances directly)
                if let Some(render) = self.render.as_mut() {
                    let total =
                        self.constants.particle.num_large + self.constants.particle.num_small;
                    render.instance_renderer.set_instance_count(total);
                }
            }
        }

        // Update view uniform from camera
        if let Some(render) = self.render.as_mut() {
            let aspect = if ctx.window_size.height > 0 {
                ctx.window_size.width as f32 / ctx.window_size.height as f32
            } else {
                1.0
            };
            render
                .scene
                .update_from_camera(&ctx.gpu.queue, &*ctx.camera, aspect);

            // Update container transforms
            render
                .container
                .update_transforms(&ctx.gpu.queue, &self.constants, &self.sim_state);
        }

        // Update distribution
        update_distribution_plain(
            &self.particle_store,
            &self.constants,
            &mut self.distribution,
            &mut self.distribution_history,
            &self.sim_state,
        );
    }

    fn compute(&mut self, ctx: &mut mikage::ComputeContext) {
        if self.backend != PhysicsBackend::Gpu {
            return;
        }

        if self.sim_state.paused {
            return;
        }

        let Some(gpu) = self.gpu_physics.as_mut() else {
            return;
        };

        if gpu.num_particles() == 0 {
            return;
        }

        // Encode physics compute passes
        gpu.encode_compute(
            ctx.device,
            ctx.queue,
            ctx.encoder,
            &self.constants,
            &self.sim_state,
        );

        // Encode particle_to_instance compute pass
        gpu.encode_particle_to_instance(ctx.queue, ctx.encoder, &self.constants);

        // Encode readback (copies to staging buffer every N frames)
        gpu.encode_readback(ctx.encoder);
    }

    fn render(&mut self, ctx: &mut mikage::RenderContext) {
        let Some(render) = self.render.as_ref() else {
            return;
        };

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main_render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: ctx.surface_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        // #1a1a2e in linear space
                        r: 0.0088,
                        g: 0.0088,
                        b: 0.025,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &render.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Draw particles (opaque, instanced)
        pass.set_bind_group(0, render.scene.bind_group(), &[]);
        render.instance_renderer.render(&mut pass);

        // Draw container (opaque pieces first, then transparent walls)
        pass.set_bind_group(0, render.scene.bind_group(), &[]);
        render.container.render(&mut pass);
    }

    fn gui(&mut self, egui_ctx: &egui::Context) {
        draw_control_panel(
            egui_ctx,
            self.dt,
            &mut self.perf_display,
            &mut self.constants,
            &self.ui_ranges,
            &mut self.backend,
            &mut self.sim_state,
        );

        draw_distribution_panel(egui_ctx, &self.distribution, &self.distribution_history);
    }

    fn resize(&mut self, ctx: &mikage::GpuContext, new_size: PhysicalSize<u32>) {
        if let Some(render) = self.render.as_mut() {
            let (_depth_texture, depth_view) =
                mikage::create_depth_texture(&ctx.device, new_size, mikage::DEPTH_FORMAT);
            render.depth_view = depth_view;
        }
    }
}

#[cfg(not(target_family = "wasm"))]
fn main() {
    let cli = CliArgs::parse();
    run_with_config_path(cli.config);
}

#[cfg(target_family = "wasm")]
fn main() {
    run_with_config_path(None);
}

fn run_with_config_path(config_path: Option<PathBuf>) {
    let loaded = resolve_startup_config(config_path.as_deref());

    for warning in &loaded.warnings {
        tracing::warn!("{warning}");
    }

    // Create OrbitCamera from CameraSettings
    let mut camera = mikage::OrbitCamera::default();
    // Convert position + target to orbit camera params
    let cam_pos = loaded.camera.position;
    let cam_target = loaded.camera.target;
    let diff = cam_pos - cam_target;
    camera.target = cam_target;
    camera.distance = diff.length();
    // Compute yaw and pitch from the offset vector
    camera.yaw = diff.x.atan2(diff.z);
    camera.pitch = (diff.y / diff.length()).asin();

    mikage::run(
        GranularClockApp::new(loaded),
        mikage::RunConfig {
            title: "Granular Clock".to_string(),
            width: 1440,
            height: 960,
            camera: Box::new(camera),
            ..Default::default()
        },
    );
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::CliArgs;
    use clap::Parser;
    use std::path::PathBuf;

    #[test]
    fn parse_cli_args_without_config() {
        let args = CliArgs::try_parse_from(["granular-clock"]).expect("parse args");
        assert_eq!(args.config, None);
    }

    #[test]
    fn parse_cli_args_with_config_path() {
        let args = CliArgs::try_parse_from(["granular-clock", "--config", "a.toml"])
            .expect("parse args with config");
        assert_eq!(args.config, Some(PathBuf::from("a.toml")));
    }

    #[test]
    fn parse_cli_args_with_missing_config_path_errors() {
        let result = CliArgs::try_parse_from(["granular-clock", "--config"]);
        assert!(result.is_err());
    }

    #[test]
    fn parse_cli_args_ignores_positionals() {
        let result = CliArgs::try_parse_from(["granular-clock", "unexpected"]);
        assert_eq!(
            result
                .expect_err("unexpected positional should be rejected")
                .kind(),
            clap::error::ErrorKind::UnknownArgument
        );
    }
}
