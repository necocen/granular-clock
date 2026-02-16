use bevy::prelude::*;
use bevy_egui::{
    EguiContexts, EguiGlobalSettings, EguiPlugin, EguiPostUpdateSet, EguiPrimaryContextPass, egui,
    input::EguiWantsInput,
};
use bevy_panorbit_camera::PanOrbitCamera;
use egui_plot::{Corner, HLine, Legend, Line, Plot};

use crate::analysis::{CurrentDistribution, DistributionHistory};
use crate::physics::{ParticleSize, ParticleStore};
use crate::rendering::MainCamera;
use crate::simulation::{
    constants::{PhysicsBackend, SimulationConstants, UiControlRanges},
    state::SimulationState,
};

const MAX_GRAPH_POINTS: usize = 600;
const CONTROL_WINDOW_WIDTH: f32 = 360.0;
const DISTRIBUTION_WINDOW_WIDTH: f32 = 440.0;

const COLOR_PANEL_BG: egui::Color32 = egui::Color32::from_rgb(18, 22, 30);
const COLOR_PANEL_BORDER: egui::Color32 = egui::Color32::from_rgb(60, 68, 84);
const COLOR_SECTION_BG: egui::Color32 = egui::Color32::from_rgb(30, 36, 48);
const COLOR_GPU: egui::Color32 = egui::Color32::from_rgb(106, 185, 255);
const COLOR_CPU: egui::Color32 = egui::Color32::from_rgb(255, 176, 96);
const COLOR_ACCENT: egui::Color32 = egui::Color32::from_rgb(128, 210, 255);
const COLOR_RESET: egui::Color32 = egui::Color32::from_rgb(178, 72, 72);

fn section_frame() -> egui::Frame {
    egui::Frame {
        fill: COLOR_SECTION_BG,
        stroke: egui::Stroke::new(1.0, COLOR_PANEL_BORDER),
        corner_radius: egui::CornerRadius::same(8),
        inner_margin: egui::Margin::symmetric(10, 8),
        ..Default::default()
    }
}

fn full_width_section(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    let full_width = ui.available_width();
    ui.allocate_ui_with_layout(
        egui::vec2(full_width, 0.0),
        egui::Layout::top_down(egui::Align::Min),
        |ui| {
            section_frame().show(ui, |ui| {
                ui.set_width(ui.available_width());
                add_contents(ui);
            });
        },
    );
}

fn status_chip(ui: &mut egui::Ui, text: &str, color: egui::Color32) {
    egui::Frame {
        fill: color.gamma_multiply(0.18),
        stroke: egui::Stroke::new(1.0, color),
        corner_radius: egui::CornerRadius::same(6),
        inner_margin: egui::Margin::symmetric(8, 4),
        ..Default::default()
    }
    .show(ui, |ui| {
        ui.label(egui::RichText::new(text).color(color).strong());
    });
}

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
            sync_orbit_camera_input_lock.after(EguiPostUpdateSet::ProcessOutput),
        );
    }
}

fn draw_control_panel_egui(
    mut contexts: EguiContexts,
    time: Res<Time>,
    mut constants: ResMut<SimulationConstants>,
    ui_ranges: Res<UiControlRanges>,
    mut backend: ResMut<PhysicsBackend>,
    mut sim_state: ResMut<SimulationState>,
) -> Result {
    let ctx = contexts.ctx_mut()?;

    egui::Window::new(egui::RichText::new("Simulation Settings").strong())
        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-10.0, 10.0))
        .default_width(CONTROL_WINDOW_WIDTH)
        .collapsible(false)
        .resizable(false)
        .frame(egui::Frame {
            fill: COLOR_PANEL_BG,
            stroke: egui::Stroke::new(1.0, COLOR_PANEL_BORDER),
            corner_radius: egui::CornerRadius::same(10),
            inner_margin: egui::Margin::same(12),
            ..Default::default()
        })
        .show(ctx, |ui| {
            ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);

            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new(format!("Time: {:.1}s", sim_state.elapsed))
                            .color(egui::Color32::WHITE)
                            .strong(),
                    );
                    let dt = time.delta_secs();
                    if dt > 0.0 {
                        let fps = 1.0 / dt;
                        let sim_steps = fps * constants.settings.substeps_per_frame as f32;
                        ui.label(
                            egui::RichText::new(format!("{fps:.0} FPS  |  {sim_steps:.0} steps/s"))
                                .color(egui::Color32::from_gray(190)),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new("FPS: --  |  steps/s: --")
                                .color(egui::Color32::from_gray(160)),
                        );
                    }
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let (mode_label, mode_color) = match *backend {
                        PhysicsBackend::Gpu => ("GPU Mode", COLOR_GPU),
                        PhysicsBackend::Cpu => ("CPU Mode", COLOR_CPU),
                    };
                    status_chip(ui, mode_label, mode_color);
                });
            });

            full_width_section(ui, |ui| {
                ui.label(
                    egui::RichText::new("Physics Backend")
                        .color(COLOR_ACCENT)
                        .strong(),
                );
                ui.add_space(2.0);
                ui.horizontal(|ui| {
                    ui.label("Physics:");
                    // Avoid marking PhysicsBackend changed every frame.
                    // (GPU path uses backend.is_changed() to detect actual backend switches.)
                    let mut selected_backend = *backend;
                    ui.radio_value(
                        &mut selected_backend,
                        PhysicsBackend::Gpu,
                        egui::RichText::new("GPU").color(COLOR_GPU),
                    );
                    ui.radio_value(
                        &mut selected_backend,
                        PhysicsBackend::Cpu,
                        egui::RichText::new("CPU").color(COLOR_CPU),
                    );
                    if selected_backend != *backend {
                        *backend = selected_backend;
                    }
                });
            });

            full_width_section(ui, |ui| {
                ui.label(
                    egui::RichText::new("Oscillation")
                        .color(COLOR_ACCENT)
                        .strong(),
                );
                ui.add_space(2.0);
                ui.checkbox(&mut constants.oscillation.enabled, "Enabled");
                constants.oscillation.amplitude = constants.oscillation.amplitude.clamp(
                    ui_ranges.oscillation_amplitude.min,
                    ui_ranges.oscillation_amplitude.max,
                );
                ui.add(
                    egui::Slider::new(
                        &mut constants.oscillation.amplitude,
                        ui_ranges.oscillation_amplitude.min..=ui_ranges.oscillation_amplitude.max,
                    )
                    .text("Amplitude [m]")
                    .step_by(ui_ranges.oscillation_amplitude.step as f64)
                    .fixed_decimals(3),
                );
                constants.oscillation.frequency = constants.oscillation.frequency.clamp(
                    ui_ranges.oscillation_frequency.min,
                    ui_ranges.oscillation_frequency.max,
                );
                ui.add(
                    egui::Slider::new(
                        &mut constants.oscillation.frequency,
                        ui_ranges.oscillation_frequency.min..=ui_ranges.oscillation_frequency.max,
                    )
                    .text("Frequency [Hz]")
                    .step_by(ui_ranges.oscillation_frequency.step as f64)
                    .fixed_decimals(1),
                );
            });

            full_width_section(ui, |ui| {
                ui.label(egui::RichText::new("Controls").color(COLOR_ACCENT).strong());
                let pause_label = if sim_state.paused { "Resume" } else { "Pause" };
                if ui
                    .add_sized(
                        [ui.available_width(), 28.0],
                        egui::Button::new(egui::RichText::new(pause_label).strong())
                            .fill(egui::Color32::from_rgb(62, 86, 145)),
                    )
                    .clicked()
                {
                    sim_state.paused = !sim_state.paused;
                }

                if ui
                    .add_sized(
                        [ui.available_width(), 28.0],
                        egui::Button::new(egui::RichText::new("Reset").strong()).fill(COLOR_RESET),
                    )
                    .clicked()
                {
                    sim_state.reset_requested = true;
                }
            });

            ui.separator();
            ui.small(
                egui::RichText::new("LMB: orbit, RMB: pan, Wheel: zoom")
                    .color(egui::Color32::from_gray(165)),
            );
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

    egui::Window::new(egui::RichText::new("Particle Distribution").strong())
        .anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 10.0))
        .default_width(DISTRIBUTION_WINDOW_WIDTH)
        .collapsible(false)
        .resizable(false)
        .frame(egui::Frame {
            fill: COLOR_PANEL_BG,
            stroke: egui::Stroke::new(1.0, COLOR_PANEL_BORDER),
            corner_radius: egui::CornerRadius::same(10),
            inner_margin: egui::Margin::same(12),
            ..Default::default()
        })
        .show(ctx, |ui| {
            ui.spacing_mut().item_spacing = egui::vec2(8.0, 8.0);

            full_width_section(ui, |ui| {
                ui.label(
                    egui::RichText::new("Current Distribution")
                        .color(COLOR_ACCENT)
                        .strong(),
                );
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
                ui.small(
                    egui::RichText::new(format!("Total: large={large_total}, small={small_total}"))
                        .color(egui::Color32::from_gray(175)),
                );
            });

            full_width_section(ui, |ui| {
                ui.label(
                    egui::RichText::new("History (Left ratio over time)")
                        .color(COLOR_ACCENT)
                        .strong(),
                );

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
                    .legend(Legend::default().position(Corner::LeftTop))
                    .allow_zoom(false)
                    .allow_drag([true, false])
                    .allow_scroll(false)
                    .include_y(0.0)
                    .include_y(1.0)
                    .show(ui, |plot_ui| {
                        // Drag で過去範囲を見ている間は保持し、カーソルが離れたら
                        // X 軸の auto-bounds に戻して最新データへ追従させる。
                        let response = plot_ui.response();
                        if !response.hovered() && !response.dragged() {
                            plot_ui.set_auto_bounds([true, true]);
                        }

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
        });

    Ok(())
}

fn apply_reset_request(
    mut sim_state: ResMut<SimulationState>,
    constants: Res<SimulationConstants>,
    mut store: ResMut<ParticleStore>,
    mut history: ResMut<DistributionHistory>,
) {
    if !sim_state.reset_requested {
        return;
    }

    sim_state.reset_requested = false;
    sim_state.reset_time();
    history.clear();

    store.clear();

    use rand::Rng;
    let mut rng = rand::rng();
    let particle = &constants.particle;
    let container = &constants.container;
    let material = &constants.material;

    let spawn_area_x = container.half_extents.x - particle.large_radius;
    let spawn_area_z = container.half_extents.z - particle.large_radius;
    let base_y = container.base_position.y - container.half_extents.y;

    for _ in 0..particle.num_large {
        let x = rng.random_range(-spawn_area_x..spawn_area_x);
        let z = rng.random_range(-spawn_area_z..spawn_area_z);
        let y = base_y + particle.large_radius + rng.random_range(0.0..0.2);

        store.spawn(
            Vec3::new(x, y, z),
            particle.large_radius,
            material.density,
            ParticleSize::Large,
        );
    }

    for _ in 0..particle.num_small {
        let x = rng.random_range(-spawn_area_x..spawn_area_x);
        let z = rng.random_range(-spawn_area_z..spawn_area_z);
        let y = base_y + particle.small_radius + rng.random_range(0.0..0.2);

        store.spawn(
            Vec3::new(x, y, z),
            particle.small_radius,
            material.density,
            ParticleSize::Small,
        );
    }
}

fn sync_orbit_camera_input_lock(
    egui_wants_input: Res<EguiWantsInput>,
    mut cameras: Query<&mut PanOrbitCamera, With<MainCamera>>,
) {
    let enabled = !egui_wants_input.wants_any_input();
    for mut camera in &mut cameras {
        camera.enabled = enabled;
    }
}
