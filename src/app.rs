use mikage::winit::dpi::PhysicalSize;
use mikage::{InstanceData, InstanceRenderer, InstanceRendererConfig, egui, glam, wgpu};

use crate::analysis::{CurrentDistribution, DistributionHistory, update_distribution_plain};
use crate::physics::{
    ContactHistory, ParticleSize, ParticleStore,
    cpu::{SpatialHashGrid, run_physics_substeps_plain},
    gpu::{GpuPhysicsState, ParticleGpu},
};
use crate::rendering::{
    ContainerRenderer, LARGE_PARTICLE_COLOR, SMALL_PARTICLE_COLOR, SceneOutput, scene_lighting,
    shadow_matrix,
};
use crate::simulation::{
    config_toml::LoadedConfig,
    constants::{PhysicsBackend, SimulationConstants, UiControlRanges, advance_oscillation},
    state::SimulationState,
};
use crate::ui::{PerfDisplayState, draw_control_panel, draw_distribution_panel};

struct RenderResources {
    instance_renderer: InstanceRenderer,
    scene: mikage::ColoredSceneBinding,
    shadow: mikage::DirectionalShadow,
    light_direction: glam::Vec3,
    output: SceneOutput,
    target_config: mikage::RenderTargetConfig,
    container: ContainerRenderer,
}

pub struct GranularClockApp {
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
        let scene = mikage::ColoredSceneBinding::new(device);
        let lighting = scene_lighting(&loaded.light);
        scene.update_lighting(&ctx.queue, lighting);
        scene.update_light_colors(
            &ctx.queue,
            mikage::SceneLightColors {
                ambient: loaded.light.ambient_color,
                directional: loaded.light.directional_color,
            },
        );
        let output = SceneOutput::new(
            ctx,
            target,
            size,
            loaded.light.hdr_enabled,
            loaded.light.exposure_ev,
        );
        let scene_target = output.config(target);
        let resolution = loaded
            .light
            .shadow_map_resolution
            .min(device.limits().max_texture_dimension_2d);
        if resolution != loaded.light.shadow_map_resolution {
            tracing::warn!("Shadow map resolution limited to {resolution} by the GPU");
        }
        let shadow = mikage::DirectionalShadow::new(
            ctx,
            mikage::DirectionalShadowConfig {
                resolution,
                light_view_projection: shadow_matrix(&constants, lighting.direction),
                enabled: loaded.light.shadows_enabled,
                ..Default::default()
            },
        )
        .expect("valid directional shadow configuration");
        if loaded.light.shadows_enabled && !shadow.is_supported() {
            tracing::warn!(
                "Directional shadows are unavailable on this GPU; using unshadowed lighting"
            );
        }

        // Create instance renderer with IcoSphere mesh
        let sphere = mikage::IcoSphereMesh::generate(2);
        let mut instance_renderer = InstanceRenderer::with_light_colors_and_shadows(
            ctx,
            scene_target,
            &scene,
            &shadow,
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

        // Create container renderer (shares the scene bind group layout)
        let container = ContainerRenderer::new(ctx, scene_target, &scene, &shadow);

        // Create GPU physics state
        let gpu_physics =
            GpuPhysicsState::new(device, &constants, instance_renderer.instance_buffer());

        let render = RenderResources {
            instance_renderer,
            scene,
            shadow,
            light_direction: lighting.direction,
            output,
            target_config: target,
            container,
        };

        let mut app = Self {
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
        self.render
            .scene
            .update_camera(&ctx.gpu.queue, ctx.camera, aspect);

        // Update container transforms
        self.render
            .container
            .update_transforms(ctx.gpu, &self.constants, &self.sim_state);
        self.render
            .shadow
            .update_matrix(
                &ctx.gpu.queue,
                shadow_matrix(&self.constants, self.render.light_direction),
            )
            .expect("finite shadow volume");
    }

    fn render(&mut self, ctx: &mut mikage::RenderContext<mikage::OrbitCamera>) {
        if let Some(mut pass) = self.render.shadow.begin_pass(ctx.encoder) {
            self.render.container.render_shadow(&mut pass);
            self.render.instance_renderer.render_shadow(&mut pass);
        }
        // Composite opaque particles and transparent container surfaces before mapping.
        let scene_target = self.render.output.target(ctx.target);
        let color_attachment = wgpu::RenderPassColorAttachment {
            view: scene_target.view,
            resolve_target: scene_target.resolve_target,
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
        };
        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main_render_pass"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: scene_target.depth_view.expect("scene depth attachment"),
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
        drop(pass);
        self.render.output.finish(ctx.encoder, ctx.target);
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
        self.render
            .output
            .resize(ctx, new_size, self.render.target_config);
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;
    use std::time::Duration;

    fn capture_scene(harness: &mut mikage::HeadlessHarness<GranularClockApp>) -> Vec<u8> {
        let frame = harness.render_once().unwrap();
        harness.runtime.wait_for(&frame).unwrap();
        harness.capture_named("scene").unwrap().data.unwrap()
    }

    fn update_lighting(
        harness: &mikage::HeadlessHarness<GranularClockApp>,
        lighting: mikage::SceneLighting,
    ) {
        harness
            .runtime
            .app
            .render
            .scene
            .update_lighting(&harness.runtime.gpu.queue, lighting);
    }

    fn centered_particle_harness(
        mut loaded: LoadedConfig,
        sample_count: u32,
    ) -> mikage::HeadlessHarness<GranularClockApp> {
        loaded.simulation.particle.num_large = 1;
        loaded.simulation.particle.num_small = 0;
        let mut camera = mikage::OrbitCamera::default();
        camera.distance = 1.0;
        let mut harness = pollster::block_on(mikage::HeadlessHarness::new(
            crate::gpu_descriptor(),
            mikage::RenderTargetConfig {
                sample_count,
                ..Default::default()
            },
            PhysicalSize::new(320, 240),
            camera,
            move |gpu, target, size| GranularClockApp::new(loaded, gpu, target, size),
        ))
        .expect("initialize lighting test app");
        harness.runtime.app.backend = PhysicsBackend::Cpu;
        harness.runtime.app.sim_state.paused = true;
        let particle = &mut harness.runtime.app.particle_store.particles[0];
        particle.position = glam::Vec3::ZERO;
        particle.radius = 0.08;
        harness
    }

    fn center_rgb(image: &[u8]) -> [f64; 3] {
        let mut rgb = [0.0; 3];
        for y in 116..124 {
            for x in 156..164 {
                for channel in 0..3 {
                    rgb[channel] += image[(y * 320 + x) * 4 + channel] as f64 / 64.0;
                }
            }
        }
        rgb
    }

    #[test]
    #[ignore = "requires native GPU adapter/device and renders real frames"]
    fn configured_light_colors_survive_camera_and_gain_changes_in_hdr() {
        let mut loaded = crate::simulation::config_toml::load_embedded_config();
        loaded.light.ambient_brightness = 10500.0;
        loaded.light.directional_illuminance = 0.0;
        loaded.light.ambient_color = glam::Vec3::Z;
        loaded.light.directional_color = glam::Vec3::Y;
        loaded.light.shadows_enabled = false;
        let mut harness = centered_particle_harness(loaded, 4);
        let blue = center_rgb(&capture_scene(&mut harness));
        assert!(
            blue[2] > 80.0 && blue[0] < 10.0 && blue[1] < 10.0,
            "{blue:?}"
        );
        harness.runtime.camera.yaw += 0.4;
        let moved_blue = center_rgb(&capture_scene(&mut harness));
        assert!(moved_blue[2] > 80.0 && moved_blue[0] < 10.0 && moved_blue[1] < 10.0);
        update_lighting(
            &harness,
            mikage::SceneLighting {
                direction: glam::Vec3::Z,
                ambient_intensity: 0.0,
                directional_intensity: 1.0,
            },
        );
        let green = center_rgb(&capture_scene(&mut harness));
        assert!(
            green[1] > 60.0 && green[0] < 10.0 && green[2] < 10.0,
            "{green:?}"
        );
        assert_eq!(harness.runtime.progress().encoded_ticks, 0);
        harness.runtime.shutdown();
    }

    #[test]
    #[ignore = "requires native GPU adapter/device and renders real frames"]
    fn hdr_exposure_preserves_highlights_and_resizes_with_msaa_without_ticks() {
        let mut loaded = crate::simulation::config_toml::load_embedded_config();
        loaded.light.ambient_brightness = 42000.0; // Independent linear gain 4.
        loaded.light.directional_illuminance = 0.0;
        loaded.light.exposure_ev = -1.0;
        loaded.light.shadows_enabled = false;
        let mut harness = centered_particle_harness(loaded, 4);
        let dim = center_rgb(&capture_scene(&mut harness))[0];
        let SceneOutput::Hdr { mapper, .. } = &harness.runtime.app.render.output else {
            panic!("native test GPU must support HDR");
        };
        mapper.update_exposure(&harness.runtime.gpu.queue, 0.0);
        let bright = center_rgb(&capture_scene(&mut harness))[0];
        update_lighting(
            &harness,
            mikage::SceneLighting {
                ambient_intensity: 8.0,
                directional_intensity: 0.0,
                ..Default::default()
            },
        );
        let brighter = center_rgb(&capture_scene(&mut harness))[0];
        assert!(
            bright > dim + 10.0 && brighter > bright + 5.0 && brighter < 254.0,
            "HDR output: {dim}, {bright}, {brighter}"
        );

        let new_size = PhysicalSize::new(192, 128);
        let destination = mikage::OffscreenTarget::new(
            &harness.runtime.gpu,
            new_size,
            harness.render_target_config(),
        )
        .unwrap();
        // Zero-size suspension retains valid attachments; restore at a new size.
        mikage::App::resize(
            &mut harness.runtime.app,
            &harness.runtime.gpu,
            PhysicalSize::new(0, 0),
        );
        mikage::App::resize(&mut harness.runtime.app, &harness.runtime.gpu, new_size);
        let token = harness
            .runtime
            .render(mikage::RenderTarget {
                view: destination.color_view(),
                resolve_target: destination.resolve_target(),
                depth_view: Some(destination.depth_view()),
                size: new_size,
                config: destination.render_target_config(),
            })
            .unwrap();
        harness.runtime.wait_for(&token).unwrap();
        let SceneOutput::Hdr { target, .. } = &harness.runtime.app.render.output else {
            unreachable!()
        };
        assert_eq!(target.size(), new_size);
        assert_eq!(harness.runtime.progress().encoded_ticks, 0);
        harness.runtime.shutdown();
    }

    #[test]
    #[ignore = "requires native GPU adapter/device and renders real frames"]
    fn gpu_written_particles_cast_shadows_on_container_and_preserve_ambient() {
        let mut loaded = crate::simulation::config_toml::load_embedded_config();
        loaded.light.ambient_brightness = 2100.0;
        loaded.light.directional_illuminance = 10500.0;
        loaded.light.directional_position = glam::Vec3::Y;
        loaded.light.directional_target = glam::Vec3::ZERO;
        let mut harness = centered_particle_harness(loaded, 4);
        harness.runtime.app.backend = PhysicsBackend::Gpu;
        harness.runtime.camera.target.y = -0.2;
        harness.runtime.camera.pitch = 0.65;
        harness.runtime.camera.yaw = 0.4;
        let particle = &mut harness.runtime.app.particle_store.particles[0];
        particle.position = glam::Vec3::new(0.10, -0.10, 0.0);
        particle.radius = 0.04;
        let tick = harness
            .advance_ticks(1, Duration::from_secs_f32(1.0 / 60.0))
            .unwrap()
            .unwrap();
        harness.runtime.wait_for(&tick).unwrap();
        let ticks = harness.runtime.progress().encoded_ticks;
        assert!(harness.runtime.app.render.shadow.is_supported());
        let shadowed = capture_scene(&mut harness);
        harness
            .runtime
            .app
            .render
            .shadow
            .set_enabled(&harness.runtime.gpu.queue, false);
        let unshadowed = capture_scene(&mut harness);
        let darker_pixels = shadowed
            .chunks_exact(4)
            .zip(unshadowed.chunks_exact(4))
            .filter(|(shadowed, plain)| plain[0] > shadowed[0].saturating_add(5))
            .count();
        assert!(darker_pixels > 30, "shadowed pixels: {darker_pixels}");
        // Check the receiver below the particle, away from the sphere's screen footprint.
        let container = &harness.runtime.app.constants.container;
        let floor_point = container.base_position
            + glam::Vec3::new(0.10, -container.half_extents.y + 0.0025, 0.0);
        let clip = mikage::Camera::view_projection_matrix(&harness.runtime.camera, 320.0 / 240.0)
            .project_point3(floor_point);
        let x = ((clip.x * 0.5 + 0.5) * 320.0) as usize;
        let y = ((0.5 - clip.y * 0.5) * 240.0) as usize;
        let red_at_floor = |image: &[u8]| {
            let mut sum = 0.0;
            for row in y - 1..=y + 1 {
                for column in x - 1..=x + 1 {
                    sum += image[(row * 320 + column) * 4] as f64 / 9.0;
                }
            }
            sum
        };
        let floor_shadow = red_at_floor(&shadowed);
        let floor_plain = red_at_floor(&unshadowed);
        assert!(
            floor_plain > floor_shadow + 10.0,
            "floor red with/without shadow: {floor_shadow}/{floor_plain}"
        );

        update_lighting(
            &harness,
            mikage::SceneLighting {
                ambient_intensity: 1.0,
                directional_intensity: 0.0,
                ..Default::default()
            },
        );
        let ambient = capture_scene(&mut harness);
        harness
            .runtime
            .app
            .render
            .shadow
            .set_enabled(&harness.runtime.gpu.queue, true);
        assert!(
            capture_scene(&mut harness) == ambient,
            "shadows must not attenuate ambient light"
        );
        assert_eq!(harness.runtime.progress().encoded_ticks, ticks);
        assert_eq!(harness.runtime.app.sim_state.elapsed, 0.0);
        harness.runtime.shutdown();
    }

    #[test]
    #[ignore = "requires native GPU adapter/device and renders real frames"]
    fn configured_lighting_changes_rendered_pixels_and_survives_camera_updates() {
        let mut loaded = crate::simulation::config_toml::load_embedded_config();
        loaded.light.hdr_enabled = false;
        loaded.light.shadows_enabled = false;
        loaded.simulation.particle.num_large = 1;
        loaded.simulation.particle.num_small = 0;
        loaded.light.ambient_brightness = 10500.0;
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
            let image = capture_scene(harness);
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

        let mut lighting = mikage::SceneLighting {
            direction: glam::Vec3::Z,
            ambient_intensity: 0.0,
            directional_intensity: 0.0,
        };
        update_lighting(&harness, lighting);
        let dark = render_center_red(&mut harness);
        assert!(ambient_only > dark + 80.0);

        harness.runtime.camera.yaw += 0.4;
        let moved_dark = render_center_red(&mut harness);
        assert!(ambient_only > moved_dark + 80.0);

        lighting.directional_intensity = 0.3;
        update_lighting(&harness, lighting);
        let dim_front_light = render_center_red(&mut harness);
        lighting.directional_intensity = 0.9;
        update_lighting(&harness, lighting);
        let front_light = render_center_red(&mut harness);
        assert!(front_light > dim_front_light + 30.0);

        lighting.direction = -glam::Vec3::Z;
        update_lighting(&harness, lighting);
        let back_light = render_center_red(&mut harness);
        assert!(front_light > back_light + 80.0);

        harness.runtime.shutdown();
    }

    #[test]
    #[ignore = "requires native GPU adapter/device and renders real frames"]
    fn transparent_container_responds_to_light_intensity() {
        let mut loaded = crate::simulation::config_toml::load_embedded_config();
        loaded.light.hdr_enabled = false;
        loaded.light.shadows_enabled = false;
        loaded.simulation.particle.num_large = 1;
        loaded.simulation.particle.num_small = 0;
        loaded.light.ambient_brightness = 0.0;
        loaded.light.directional_illuminance = 0.0;
        let mut camera = mikage::OrbitCamera::default();
        camera.distance = 1.0;
        camera.yaw = 0.4;
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
        // Keep the allocated instance buffer, but render only the container.
        harness.runtime.app.particle_store.clear();

        let dark = capture_scene(&mut harness);
        let mut lighting = mikage::SceneLighting {
            ambient_intensity: 0.25,
            directional_intensity: 0.0,
            ..Default::default()
        };
        update_lighting(&harness, lighting);
        let dim = capture_scene(&mut harness);
        lighting.ambient_intensity = 1.0;
        update_lighting(&harness, lighting);
        let bright = capture_scene(&mut harness);

        let brighter_pixels = dark
            .chunks_exact(4)
            .zip(dim.chunks_exact(4))
            .zip(bright.chunks_exact(4))
            .filter(|((dark, dim), bright)| {
                dim[0] > dark[0].saturating_add(5) && bright[0] > dim[0].saturating_add(10)
            })
            .count();
        assert!(
            brighter_pixels > 500,
            "lit container pixels: {brighter_pixels}"
        );

        // The camera update in prepare_render must not restore a default light.
        update_lighting(
            &harness,
            mikage::SceneLighting {
                ambient_intensity: 0.0,
                directional_intensity: 0.0,
                ..lighting
            },
        );
        assert!(
            capture_scene(&mut harness) == dark,
            "disabling both lights should restore the dark container"
        );
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
