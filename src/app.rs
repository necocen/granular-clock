use mikage::winit::dpi::PhysicalSize;
use mikage::{InstanceData, InstanceRenderer, InstanceRendererConfig, egui, glam, wgpu};

use crate::analysis::{CurrentDistribution, DistributionHistory, update_distribution_plain};
use crate::physics::{
    ContactHistory, ParticleSize, ParticleStore,
    cpu::{SpatialHashGrid, run_physics_substeps_plain},
    gpu::{GpuPhysicsState, ParticleGpu},
};
use crate::rendering::{
    ContainerRenderer, LARGE_PARTICLE_COLOR, SMALL_PARTICLE_COLOR, scene_uniform,
};
use crate::simulation::{
    config_toml::LoadedConfig,
    constants::{
        LightSettings, PhysicsBackend, SimulationConstants, UiControlRanges, advance_oscillation,
    },
    state::SimulationState,
};
use crate::ui::{PerfDisplayState, draw_control_panel, draw_distribution_panel};

struct RenderResources {
    instance_renderer: InstanceRenderer,
    scene: mikage::SceneBinding,
    depth_view: wgpu::TextureView,
    target_config: mikage::RenderTargetConfig,
    container: ContainerRenderer,
}

pub struct GranularClockApp {
    constants: SimulationConstants,
    ui_ranges: UiControlRanges,
    light: LightSettings,
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
    render: RenderResources,
    gpu_physics: GpuPhysicsState,
    /// GPU 初回アップロード済みフラグ
    gpu_particles_uploaded: bool,
    /// ParticleStore の世代追跡（GPU アップロード用）
    last_gpu_store_generation: u64,
}

impl GranularClockApp {
    pub fn new(
        loaded: LoadedConfig,
        ctx: &mikage::GpuContext,
        target: mikage::RenderTargetConfig,
        size: PhysicalSize<u32>,
    ) -> Self {
        let constants = loaded.simulation;
        let grid_settings = constants.grid;
        let spatial_hash = SpatialHashGrid::new(grid_settings.cell_size, grid_settings.table_size);

        let device = &ctx.device;

        // Create scene binding
        let scene = mikage::SceneBinding::new(device);

        // Create instance renderer with IcoSphere mesh
        let sphere = mikage::IcoSphereMesh::generate(2);
        let mut instance_renderer = InstanceRenderer::new(
            ctx,
            target,
            scene.layout(),
            &sphere.positions,
            &sphere.normals,
            &sphere.indices,
            InstanceRendererConfig {
                storage_binding: true,
                ..InstanceRendererConfig::default_3d()
            },
        );

        // Pre-allocate instance buffer for expected particle count
        let total_particles = constants.particle.num_large + constants.particle.num_small;
        instance_renderer.ensure_capacity(device, total_particles);

        // Create depth texture
        let (_depth_texture, depth_view) = mikage::create_depth_texture(ctx, size, target);

        // Create container renderer (shares the scene bind group layout)
        let container = ContainerRenderer::new(ctx, target, scene.layout());

        // Create GPU physics state
        let gpu_physics =
            GpuPhysicsState::new(device, &constants, instance_renderer.instance_buffer());

        let render = RenderResources {
            instance_renderer,
            scene,
            depth_view,
            target_config: target,
            container,
        };

        let mut app = Self {
            constants,
            ui_ranges: loaded.ui_ranges,
            light: loaded.light,
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
            render,
            gpu_physics,
            gpu_particles_uploaded: false,
            last_gpu_store_generation: 0,
        };

        // Spawn particles
        app.spawn_particles();

        app
    }

    /// Spawn particles into the ParticleStore
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

    /// Build instance data from particles
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
        // GPU particles will be re-uploaded in the next tick.
        self.gpu_particles_uploaded = false;
    }

    /// Upload current particles to GPU physics state
    fn upload_particles_to_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let gpu_particles = self.particles_to_gpu();
        if gpu_particles.is_empty() {
            return;
        }

        let generation = self.particle_store.generation;

        self.gpu_physics.upload_particles(
            device,
            queue,
            &gpu_particles,
            self.render.instance_renderer.instance_buffer(),
        );
        self.last_gpu_store_generation = generation;
        self.gpu_particles_uploaded = true;
    }
}

impl mikage::App for GranularClockApp {
    type Camera = mikage::OrbitCamera;

    fn tick(&mut self, ctx: &mut mikage::TickContext) {
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
            }
            PhysicsBackend::Gpu => {
                let total = self.constants.particle.num_large + self.constants.particle.num_small;
                let instance_buffer = self
                    .render
                    .instance_renderer
                    .prepare_compute(&ctx.gpu.device, total);
                if instance_buffer.reallocated {
                    self.gpu_physics
                        .rebuild_instance_bind_group(&ctx.gpu.device, instance_buffer.buffer);
                }

                // Check if we need to upload particles (initial or after reset)
                let store_gen_changed =
                    self.particle_store.generation != self.last_gpu_store_generation;
                if !self.gpu_particles_uploaded || store_gen_changed {
                    self.upload_particles_to_gpu(&ctx.gpu.device, &ctx.gpu.queue);
                }

                // Process readback from previous frame
                self.gpu_physics.process_readback(&ctx.gpu.device);
                self.gpu_physics
                    .apply_readback_to_store(&mut self.particle_store);

                if !self.sim_state.paused {
                    // Update GPU params
                    self.gpu_physics.update_params(
                        &ctx.gpu.queue,
                        &self.constants,
                        &self.sim_state,
                    );

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

                    self.gpu_physics.encode_compute(
                        &ctx.gpu.device,
                        &ctx.gpu.queue,
                        ctx.encoder,
                        &self.constants,
                        &self.sim_state,
                    );
                }

                if self.gpu_physics.num_particles() > 0 {
                    // Keep instance data current even when resetting while paused.
                    self.gpu_physics.encode_particle_to_instance(
                        &ctx.gpu.queue,
                        ctx.encoder,
                        &self.constants,
                    );
                    self.gpu_physics.encode_readback(ctx.encoder);
                }
            }
        }

        update_distribution_plain(
            &self.particle_store,
            &self.constants,
            &mut self.distribution,
            &mut self.distribution_history,
            &self.sim_state,
        );
    }

    fn prepare_render(&mut self, ctx: &mut mikage::RenderUpdateContext<mikage::OrbitCamera>) {
        self.dt = ctx.dt;

        if self.backend == PhysicsBackend::Cpu {
            let instances = self.build_instance_data();
            let reallocated = self
                .render
                .instance_renderer
                .ensure_capacity(&ctx.gpu.device, instances.len() as u32);
            if reallocated {
                self.gpu_physics.rebuild_instance_bind_group(
                    &ctx.gpu.device,
                    self.render.instance_renderer.instance_buffer(),
                );
            }
            self.render
                .instance_renderer
                .update_instances(ctx.gpu, &instances);
        }

        // Update view uniform from camera
        let aspect = if ctx.target_size.height > 0 {
            ctx.target_size.width as f32 / ctx.target_size.height as f32
        } else {
            1.0
        };
        let uniform = scene_uniform(ctx.camera, aspect, &self.light);
        self.render.scene.update(&ctx.gpu.queue, &uniform);

        // Update container transforms
        self.render
            .container
            .update_transforms(ctx.gpu, &self.constants, &self.sim_state);
    }

    fn render(&mut self, ctx: &mut mikage::RenderContext<mikage::OrbitCamera>) {
        // Render
        let color_attachment = ctx.color_attachment(wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color {
                // #1a1a2e in linear space
                r: 0.0088,
                g: 0.0088,
                b: 0.025,
                a: 1.0,
            }),
            store: wgpu::StoreOp::Store,
        });
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main_render_pass"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.render.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        // Draw particles (opaque, instanced)
        pass.set_bind_group(0, self.render.scene.bind_group(), &[]);
        self.render.instance_renderer.render(&mut pass);

        // Draw container (opaque pieces first, then transparent walls)
        pass.set_bind_group(0, self.render.scene.bind_group(), &[]);
        self.render.container.render(&mut pass);
    }

    fn gui(&mut self, ui: &mut egui::Ui) {
        let egui_ctx = ui.ctx();
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
        let (_depth_texture, depth_view) =
            mikage::create_depth_texture(ctx, new_size, self.render.target_config);
        self.render.depth_view = depth_view;
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    #[ignore = "requires native GPU adapter/device and renders real frames"]
    fn configured_lighting_changes_rendered_pixels_and_survives_camera_updates() {
        let mut loaded = crate::simulation::config_toml::load_embedded_config();
        loaded.simulation.particle.num_large = 1;
        loaded.simulation.particle.num_small = 0;
        loaded.light.ambient_brightness = 1.0;
        loaded.light.directional_illuminance = 0.0;
        let mut camera = mikage::OrbitCamera::default();
        camera.distance = 1.0;
        let mut harness = pollster::block_on(mikage::HeadlessHarness::new(
            crate::gpu_descriptor(),
            mikage::RenderTargetConfig::default(),
            PhysicalSize::new(320, 240),
            camera,
            move |gpu, target, size| GranularClockApp::new(loaded, gpu, target, size),
        ))
        .expect("initialize headless app");
        harness.runtime.app.backend = PhysicsBackend::Cpu;
        harness.runtime.app.sim_state.paused = true;
        let particle = &mut harness.runtime.app.particle_store.particles[0];
        particle.position = glam::Vec3::ZERO;
        particle.radius = 0.08;

        let render_center_red = |harness: &mut mikage::HeadlessHarness<GranularClockApp>| {
            let frame = harness.render_once().unwrap();
            harness.runtime.wait_for(&frame).unwrap();
            let image = harness.capture_named("scene").unwrap().data.unwrap();
            let mut red = 0.0;
            for y in 116..124 {
                for x in 156..164 {
                    red += image[(y * 320 + x) * 4] as f64;
                }
            }
            red / 64.0
        };

        let ambient_only = render_center_red(&mut harness);
        assert!(ambient_only > 210.0, "ambient-only red: {ambient_only}");

        harness.runtime.app.light.ambient_brightness = 0.0;
        let dark = render_center_red(&mut harness);
        assert!(ambient_only > dark + 80.0);

        harness.runtime.camera.yaw += 0.4;
        let moved_dark = render_center_red(&mut harness);
        assert!(ambient_only > moved_dark + 80.0);

        harness.runtime.app.light.directional_illuminance = 10000.0;
        harness.runtime.app.light.directional_target = glam::Vec3::ZERO;
        harness.runtime.app.light.directional_position = glam::Vec3::Z;
        let front_light = render_center_red(&mut harness);
        harness.runtime.app.light.directional_position = -glam::Vec3::Z;
        let back_light = render_center_red(&mut harness);
        assert!(front_light > back_light + 80.0);

        harness.runtime.shutdown();
    }

    #[test]
    #[ignore = "requires native GPU adapter/device and renders real frames"]
    fn lifecycle_runs_both_backends_without_ticking_during_render() {
        let mut loaded = crate::simulation::config_toml::load_embedded_config();
        loaded.simulation.particle.num_large = 2;
        loaded.simulation.particle.num_small = 2;
        loaded.simulation.settings.substeps_per_frame = 2;
        let elapsed_per_tick = loaded.simulation.time.dt as f64 * 2.0;
        let mut harness = pollster::block_on(mikage::HeadlessHarness::new(
            crate::gpu_descriptor(),
            mikage::RenderTargetConfig::default(),
            PhysicalSize::new(320, 240),
            mikage::OrbitCamera::default(),
            move |gpu, target, size| GranularClockApp::new(loaded, gpu, target, size),
        ))
        .expect("initialize headless app");
        let dt = Duration::from_secs_f32(1.0 / 60.0);

        let tick = harness.advance_ticks(1, dt).unwrap().unwrap();
        harness.runtime.wait_for(&tick).unwrap();
        assert_eq!(harness.runtime.app.sim_state.elapsed, elapsed_per_tick);

        let frame = harness.render_once().unwrap();
        harness.runtime.wait_for(&frame).unwrap();
        let image = harness.capture_named("scene").unwrap().data.unwrap();
        assert_eq!(image.len(), 320 * 240 * 4);
        assert!(image.chunks_exact(4).any(|pixel| pixel != &image[..4]));
        assert_eq!(harness.runtime.app.sim_state.elapsed, elapsed_per_tick);

        harness.runtime.app.sim_state.paused = true;
        let tick = harness.advance_ticks(1, dt).unwrap().unwrap();
        harness.runtime.wait_for(&tick).unwrap();
        assert_eq!(harness.runtime.app.sim_state.elapsed, elapsed_per_tick);

        let generation = harness.runtime.app.particle_store.generation;
        harness.runtime.app.sim_state.reset_requested = true;
        let tick = harness.advance_ticks(1, dt).unwrap().unwrap();
        harness.runtime.wait_for(&tick).unwrap();
        assert_eq!(harness.runtime.app.sim_state.elapsed, 0.0);
        assert!(harness.runtime.app.particle_store.generation > generation);
        let frame = harness.render_once().unwrap();
        harness.runtime.wait_for(&frame).unwrap();

        harness.runtime.app.backend = PhysicsBackend::Cpu;
        harness.runtime.app.sim_state.paused = false;
        let tick = harness.advance_ticks(1, dt).unwrap().unwrap();
        harness.runtime.wait_for(&tick).unwrap();
        assert_eq!(harness.runtime.app.sim_state.elapsed, elapsed_per_tick);
        let frame = harness.render_once().unwrap();
        harness.runtime.wait_for(&frame).unwrap();
        assert_eq!(harness.runtime.app.sim_state.elapsed, elapsed_per_tick);

        harness.runtime.app.backend = PhysicsBackend::Gpu;
        let tick = harness.advance_ticks(1, dt).unwrap().unwrap();
        harness.runtime.wait_for(&tick).unwrap();
        assert_eq!(
            harness.runtime.app.sim_state.elapsed,
            elapsed_per_tick * 2.0
        );
        let frame = harness.render_once().unwrap();
        harness.runtime.wait_for(&frame).unwrap();
        harness.runtime.shutdown();
    }
}
