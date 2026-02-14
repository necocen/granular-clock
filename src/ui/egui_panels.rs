use bevy::camera_controller::free_camera::FreeCameraState;
use bevy::prelude::*;
use bevy_egui::{
    egui, input::EguiWantsInput, EguiContexts, EguiGlobalSettings, EguiPlugin, EguiPostUpdateSet,
    EguiPrimaryContextPass,
};
use egui_plot::{HLine, Legend, Line, Plot};

use crate::analysis::{CurrentDistribution, DistributionHistory};
use crate::physics::{ParticleSize, ParticleStore};
use crate::rendering::MainCamera;
use crate::simulation::{
    constants::{PhysicsBackend, SimulationConstants},
    state::SimulationState,
};

const MAX_GRAPH_POINTS: usize = 600;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(EguiGlobalSettings {
            // Primary context is explicitly attached to MainCamera.
            auto_create_primary_context: false,
            ..default()
        })
        .add_plugins(EguiPlugin::default())
        .add_systems(
            EguiPrimaryContextPass,
            (draw_control_panel_egui, draw_distribution_panel_egui),
        )
        .add_systems(Update, apply_reset_request)
        .add_systems(
            PostUpdate,
            sync_free_camera_input_lock.after(EguiPostUpdateSet::ProcessOutput),
        );
    }
}

fn draw_control_panel_egui(
    mut contexts: EguiContexts,
    time: Res<Time>,
    mut constants: ResMut<SimulationConstants>,
    mut backend: ResMut<PhysicsBackend>,
    mut sim_state: ResMut<SimulationState>,
    mut history: ResMut<DistributionHistory>,
) -> Result {
    let ctx = contexts.ctx_mut()?;

    egui::Window::new("Simulation Settings")
        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-10.0, 10.0))
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(format!("Time: {:.1} s", sim_state.elapsed));

            let dt = time.delta_secs();
            if dt > 0.0 {
                let fps = 1.0 / dt;
                let sim_steps = fps * constants.settings.substeps_per_frame as f32;
                ui.label(format!("FPS: {:.0}  Sim: {:.0} steps/s", fps, sim_steps));
            } else {
                ui.label("FPS: --  Sim: -- steps/s");
            }

            ui.separator();

            ui.checkbox(&mut constants.oscillation.enabled, "Oscillation");

            ui.horizontal(|ui| {
                ui.label("Physics:");
                // Avoid marking PhysicsBackend changed every frame.
                // (GPU path uses backend.is_changed() to detect actual backend switches.)
                let mut selected_backend = *backend;
                ui.radio_value(&mut selected_backend, PhysicsBackend::Gpu, "GPU");
                ui.radio_value(&mut selected_backend, PhysicsBackend::Cpu, "CPU");
                if selected_backend != *backend {
                    *backend = selected_backend;
                }
            });

            ui.separator();

            ui.add(
                egui::Slider::new(&mut constants.oscillation.amplitude, 0.001..=0.1)
                    .text("Amplitude [m]")
                    .step_by(0.001)
                    .fixed_decimals(3),
            );

            ui.add(
                egui::Slider::new(&mut constants.oscillation.frequency, 1.0..=20.0)
                    .text("Frequency [Hz]")
                    .step_by(1.0)
                    .fixed_decimals(1),
            );

            ui.separator();

            let pause_label = if sim_state.paused { "Resume" } else { "Pause" };
            if ui.button(pause_label).clicked() {
                sim_state.paused = !sim_state.paused;
            }

            if ui.button("Reset").clicked() {
                sim_state.reset_requested = true;
                history.clear();
            }

            ui.separator();
            ui.label("WASD: move, Mouse: look");
        });

    Ok(())
}

fn draw_distribution_panel_egui(
    mut contexts: EguiContexts,
    current: Res<CurrentDistribution>,
    history: Res<DistributionHistory>,
) -> Result {
    let ctx = contexts.ctx_mut()?;

    let large_total = current.total_large();
    let small_total = current.total_small();
    let large_ratio = current.left_large_ratio() as f32;
    let small_ratio = current.left_small_ratio() as f32;

    egui::Window::new("Particle Distribution")
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 10.0))
        .default_width(420.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(format!(
                "Large: L {} / R {} ({:.1}%)",
                current.left_large,
                current.right_large,
                large_ratio * 100.0
            ));
            ui.add(
                egui::ProgressBar::new(large_ratio)
                    .fill(egui::Color32::from_rgb(204, 77, 77))
                    .show_percentage(),
            );

            ui.label(format!(
                "Small: L {} / R {} ({:.1}%)",
                current.left_small,
                current.right_small,
                small_ratio * 100.0
            ));
            ui.add(
                egui::ProgressBar::new(small_ratio)
                    .fill(egui::Color32::from_rgb(77, 77, 204))
                    .show_percentage(),
            );

            ui.separator();
            ui.label(format!(
                "Total: large={}, small={}",
                large_total, small_total
            ));
            ui.label("History (Left ratio over time)");

            let start_idx = history.timestamps.len().saturating_sub(MAX_GRAPH_POINTS);
            let large_points: Vec<[f64; 2]> = history
                .timestamps
                .iter()
                .zip(history.left_large_ratio.iter())
                .skip(start_idx)
                .map(|(t, r)| [*t, *r])
                .collect();
            let small_points: Vec<[f64; 2]> = history
                .timestamps
                .iter()
                .zip(history.left_small_ratio.iter())
                .skip(start_idx)
                .map(|(t, r)| [*t, *r])
                .collect();

            Plot::new("distribution_history_plot")
                .height(150.0)
                .legend(Legend::default())
                .allow_zoom(false)
                .allow_drag(false)
                .allow_scroll(false)
                .include_y(0.0)
                .include_y(1.0)
                .show(ui, |plot_ui| {
                    plot_ui.hline(HLine::new("50%", 0.5).color(egui::Color32::GRAY));

                    if !large_points.is_empty() {
                        plot_ui.line(
                            Line::new("Large", large_points)
                                .color(egui::Color32::from_rgb(230, 102, 102)),
                        );
                    }
                    if !small_points.is_empty() {
                        plot_ui.line(
                            Line::new("Small", small_points)
                                .color(egui::Color32::from_rgb(102, 102, 230)),
                        );
                    }
                });
        });

    Ok(())
}

fn apply_reset_request(
    mut sim_state: ResMut<SimulationState>,
    constants: Res<SimulationConstants>,
    mut store: ResMut<ParticleStore>,
) {
    if !sim_state.reset_requested {
        return;
    }

    sim_state.reset_requested = false;
    sim_state.reset_time();

    store.clear();

    use rand::Rng;
    let mut rng = rand::rng();
    let config = &constants.config;
    let container = &constants.container;

    let spawn_area_x = container.half_extents.x - config.large_radius;
    let spawn_area_z = container.half_extents.z - config.large_radius;
    let base_y = container.base_position.y - container.half_extents.y;

    for _ in 0..config.num_large {
        let x = rng.random_range(-spawn_area_x..spawn_area_x);
        let z = rng.random_range(-spawn_area_z..spawn_area_z);
        let y = base_y + config.large_radius + rng.random_range(0.0..0.2);

        store.spawn(
            Vec3::new(x, y, z),
            config.large_radius,
            config.density,
            ParticleSize::Large,
        );
    }

    for _ in 0..config.num_small {
        let x = rng.random_range(-spawn_area_x..spawn_area_x);
        let z = rng.random_range(-spawn_area_z..spawn_area_z);
        let y = base_y + config.small_radius + rng.random_range(0.0..0.2);

        store.spawn(
            Vec3::new(x, y, z),
            config.small_radius,
            config.density,
            ParticleSize::Small,
        );
    }
}

fn sync_free_camera_input_lock(
    egui_wants_input: Res<EguiWantsInput>,
    mut camera_states: Query<&mut FreeCameraState, With<MainCamera>>,
) {
    let enabled = !egui_wants_input.wants_any_input();
    for mut state in &mut camera_states {
        state.enabled = enabled;
    }
}
