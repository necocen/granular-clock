use egui_plot::{Corner, HLine, Legend, Line, Plot};
use mikage::egui;

use crate::analysis::{CurrentDistribution, DistributionHistory};
use crate::simulation::{
    constants::{PhysicsBackend, SimulationConstants, UiControlRanges},
    state::SimulationState,
};

const CONTROL_WINDOW_WIDTH: f32 = 360.0;
const DISTRIBUTION_WINDOW_WIDTH: f32 = 440.0;
const MOBILE_BREAKPOINT_WIDTH: f32 = 820.0;
const MOBILE_BREAKPOINT_HEIGHT: f32 = 700.0;
const MOBILE_CONTROL_WINDOW_MAX_WIDTH: f32 = 300.0;
const MOBILE_DISTRIBUTION_WINDOW_MAX_WIDTH: f32 = 320.0;
const MOBILE_PLOT_HEIGHT: f32 = 120.0;
const DESKTOP_PLOT_HEIGHT: f32 = 150.0;

const COLOR_PANEL_BG: egui::Color32 = egui::Color32::from_rgb(18, 22, 30);
const COLOR_PANEL_BORDER: egui::Color32 = egui::Color32::from_rgb(60, 68, 84);
const COLOR_SECTION_BG: egui::Color32 = egui::Color32::from_rgb(30, 36, 48);
const COLOR_GPU: egui::Color32 = egui::Color32::from_rgb(106, 185, 255);
const COLOR_CPU: egui::Color32 = egui::Color32::from_rgb(255, 176, 96);
const COLOR_ACCENT: egui::Color32 = egui::Color32::from_rgb(128, 210, 255);
const COLOR_RESET: egui::Color32 = egui::Color32::from_rgb(178, 72, 72);
const PERF_SMOOTHING_TAU_SEC: f64 = 0.6;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PerfDisplayState {
    pub smoothed_fps: Option<f64>,
    pub smoothed_steps_per_sec: Option<f64>,
    pub smoothed_speed: Option<f64>,
    previous_sim_elapsed: Option<f64>,
}

impl PerfDisplayState {
    /// Record once per rendered frame using uncapped wall-clock time.
    /// GUI layout/render passes only read these metrics.
    pub fn record_frame(&mut self, wall_dt: f64, sim_elapsed: f64, physics_dt: f64, paused: bool) {
        let valid_wall_dt = wall_dt.is_finite() && wall_dt > 0.0;
        let alpha = if valid_wall_dt {
            -(-wall_dt / PERF_SMOOTHING_TAU_SEC).exp_m1()
        } else {
            0.0
        };
        if valid_wall_dt {
            smooth_metric(&mut self.smoothed_fps, 1.0 / wall_dt, alpha);
        } else {
            self.smoothed_fps = None;
        }

        let previous = self.previous_sim_elapsed;
        self.previous_sim_elapsed =
            (sim_elapsed.is_finite() && sim_elapsed >= 0.0).then_some(sim_elapsed);
        let reset = previous
            .zip(self.previous_sim_elapsed)
            .is_some_and(|(previous, current)| current < previous);
        if paused || reset {
            self.smoothed_speed = Some(0.0);
            self.smoothed_steps_per_sec = Some(0.0);
            return;
        }

        let Some((previous, current)) = previous.zip(self.previous_sim_elapsed) else {
            self.smoothed_speed = None;
            self.smoothed_steps_per_sec = None;
            return;
        };
        if !valid_wall_dt {
            self.smoothed_speed = None;
            self.smoothed_steps_per_sec = None;
            return;
        }

        let speed = (current - previous) / wall_dt;
        smooth_metric(&mut self.smoothed_speed, speed, alpha);
        if physics_dt.is_finite() && physics_dt > 0.0 {
            smooth_metric(&mut self.smoothed_steps_per_sec, speed / physics_dt, alpha);
        } else {
            self.smoothed_steps_per_sec = None;
        }
    }
}

fn smooth_metric(smoothed: &mut Option<f64>, sample: f64, alpha: f64) {
    *smoothed = if sample.is_finite() && sample >= 0.0 {
        Some(smoothed.map_or(sample, |previous| previous + (sample - previous) * alpha))
    } else {
        None
    };
}

#[derive(Clone, Copy)]
struct UiLayoutProfile {
    mobile: bool,
    control_width: f32,
    distribution_width: f32,
    plot_height: f32,
    item_spacing: egui::Vec2,
    control_offset: egui::Vec2,
    distribution_offset: egui::Vec2,
    button_height: f32,
    panel_margin: i8,
}

fn ui_layout_profile(ctx: &egui::Context) -> UiLayoutProfile {
    let viewport = ctx.content_rect();
    let width = viewport.width();
    let height = viewport.height();
    let mobile = width <= MOBILE_BREAKPOINT_WIDTH
        || height <= MOBILE_BREAKPOINT_HEIGHT
        || (width < 960.0 && height > width);

    if mobile {
        UiLayoutProfile {
            mobile: true,
            control_width: (width - 20.0).clamp(220.0, MOBILE_CONTROL_WINDOW_MAX_WIDTH),
            distribution_width: (width - 20.0).clamp(240.0, MOBILE_DISTRIBUTION_WINDOW_MAX_WIDTH),
            plot_height: MOBILE_PLOT_HEIGHT,
            item_spacing: egui::vec2(6.0, 6.0),
            control_offset: egui::vec2(-8.0, 8.0),
            distribution_offset: egui::vec2(8.0, -8.0),
            button_height: 24.0,
            panel_margin: 8,
        }
    } else {
        UiLayoutProfile {
            mobile: false,
            control_width: CONTROL_WINDOW_WIDTH,
            distribution_width: DISTRIBUTION_WINDOW_WIDTH,
            plot_height: DESKTOP_PLOT_HEIGHT,
            item_spacing: egui::vec2(8.0, 8.0),
            control_offset: egui::vec2(-10.0, 10.0),
            distribution_offset: egui::vec2(10.0, 10.0),
            button_height: 28.0,
            panel_margin: 12,
        }
    }
}

fn settings_section(
    ui: &mut egui::Ui,
    title: &str,
    default_open: bool,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    full_width_section(ui, |ui| {
        egui::CollapsingHeader::new(egui::RichText::new(title).color(COLOR_ACCENT).strong())
            .default_open(default_open)
            .show(ui, |ui| {
                ui.add_space(2.0);
                add_contents(ui);
            });
    });
}

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

fn length_slider(
    ui: &mut egui::Ui,
    value_m: &mut f32,
    range_m: std::ops::RangeInclusive<f32>,
    step_m: f32,
    label: &str,
) -> egui::Response {
    let mut value_mm = f64::from(*value_m) * 1000.0;
    let response = ui.add(
        egui::Slider::new(
            &mut value_mm,
            f64::from(*range_m.start()) * 1000.0..=f64::from(*range_m.end()) * 1000.0,
        )
        .text(format!("{label} [mm]"))
        .step_by(f64::from(step_m) * 1000.0)
        .fixed_decimals(3)
        // Displaying an existing value must not snap it to the slider's step grid.
        .clamping(egui::SliderClamping::Edits),
    );
    if response.changed() {
        *value_m = (value_mm / 1000.0) as f32;
    }
    response
}

/// Draw the Simulation Settings control panel.
pub fn draw_control_panel(
    ctx: &egui::Context,
    perf_display: &PerfDisplayState,
    constants: &mut SimulationConstants,
    ui_ranges: &UiControlRanges,
    backend: &mut PhysicsBackend,
    sim_state: &mut SimulationState,
    cpu_error: Option<&str>,
) {
    let layout = ui_layout_profile(ctx);

    egui::Window::new(egui::RichText::new("Simulation Settings").strong())
        .anchor(egui::Align2::RIGHT_TOP, layout.control_offset)
        .default_width(layout.control_width)
        .min_width(layout.control_width)
        .max_width(layout.control_width)
        .collapsible(true)
        .default_open(!layout.mobile)
        .resizable(false)
        .frame(egui::Frame {
            fill: COLOR_PANEL_BG,
            stroke: egui::Stroke::new(1.0, COLOR_PANEL_BORDER),
            corner_radius: egui::CornerRadius::same(10),
            inner_margin: egui::Margin::same(layout.panel_margin),
            ..Default::default()
        })
        .show(ctx, |ui| {
            ui.spacing_mut().item_spacing = layout.item_spacing;
            if let Some(message) = cpu_error {
                ui.colored_label(egui::Color32::LIGHT_RED, message);
            }
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label(
                        egui::RichText::new(format!("Time: {:.1}s", sim_state.elapsed))
                            .color(egui::Color32::WHITE)
                            .strong(),
                    );
                    let speed_text = perf_display
                        .smoothed_speed
                        .map_or_else(|| "--".to_owned(), |speed| format!("{speed:.2}x"));
                    ui.label(
                        egui::RichText::new(format!("Speed: {speed_text}"))
                            .color(egui::Color32::from_gray(190))
                            .monospace(),
                    );
                    let fps_text = perf_display
                        .smoothed_fps
                        .map_or_else(|| " --".to_owned(), |fps| format!("{fps:>3.0}"));
                    let steps_text = perf_display
                        .smoothed_steps_per_sec
                        .map_or_else(|| "  --".to_owned(), |steps| format!("{steps:>4.0}"));
                    ui.label(
                        egui::RichText::new(format!("{fps_text} FPS  |  {steps_text} steps/s"))
                            .color(egui::Color32::from_gray(190))
                            .monospace(),
                    );
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let (mode_label, mode_color) = match *backend {
                        PhysicsBackend::Gpu => ("GPU Mode", COLOR_GPU),
                        PhysicsBackend::Cpu => ("CPU Mode", COLOR_CPU),
                    };
                    status_chip(ui, mode_label, mode_color);
                });
            });

            settings_section(ui, "Simulation Mode", true, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Physics:");
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

            settings_section(ui, "Oscillation", false, |ui| {
                ui.checkbox(&mut constants.oscillation.enabled, "Enabled");
                constants.oscillation.amplitude = constants.oscillation.amplitude.clamp(
                    ui_ranges.oscillation_amplitude.min,
                    ui_ranges.oscillation_amplitude.max,
                );
                length_slider(
                    ui,
                    &mut constants.oscillation.amplitude,
                    ui_ranges.oscillation_amplitude.min..=ui_ranges.oscillation_amplitude.max,
                    ui_ranges.oscillation_amplitude.step,
                    "Amplitude",
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

            settings_section(ui, "Container", false, |ui| {
                let container_height = (constants.container.half_extents.y * 2.0).max(0.001);
                let phys_min = 0.0_f32;
                let phys_max = (container_height - 0.001).max(phys_min + 0.001);

                let mut slider_min = ui_ranges.divider_height.min.max(phys_min);
                let mut slider_max = ui_ranges.divider_height.max.min(phys_max);
                let slider_step = ui_ranges.divider_height.step.max(0.0001);
                if slider_max <= slider_min {
                    slider_min = phys_min;
                    slider_max = phys_max;
                }

                constants.container.divider_height = constants
                    .container
                    .divider_height
                    .clamp(slider_min, slider_max);
                length_slider(
                    ui,
                    &mut constants.container.divider_height,
                    slider_min..=slider_max,
                    slider_step,
                    "Divider Height",
                );
            });

            settings_section(ui, "Contact Material", false, |ui| {
                constants.material.restitution = constants.material.restitution.clamp(
                    ui_ranges.particle_restitution.min,
                    ui_ranges.particle_restitution.max,
                );
                ui.add(
                    egui::Slider::new(
                        &mut constants.material.restitution,
                        ui_ranges.particle_restitution.min..=ui_ranges.particle_restitution.max,
                    )
                    .text("Particle Restitution")
                    .step_by(ui_ranges.particle_restitution.step as f64)
                    .fixed_decimals(2),
                );

                constants.material.friction = constants.material.friction.clamp(
                    ui_ranges.particle_friction.min,
                    ui_ranges.particle_friction.max,
                );
                ui.add(
                    egui::Slider::new(
                        &mut constants.material.friction,
                        ui_ranges.particle_friction.min..=ui_ranges.particle_friction.max,
                    )
                    .text("Particle Friction")
                    .step_by(ui_ranges.particle_friction.step as f64)
                    .fixed_decimals(2),
                );

                constants.wall.restitution = constants.wall.restitution.clamp(
                    ui_ranges.wall_restitution.min,
                    ui_ranges.wall_restitution.max,
                );
                ui.add(
                    egui::Slider::new(
                        &mut constants.wall.restitution,
                        ui_ranges.wall_restitution.min..=ui_ranges.wall_restitution.max,
                    )
                    .text("Wall Restitution")
                    .step_by(ui_ranges.wall_restitution.step as f64)
                    .fixed_decimals(2),
                );

                constants.wall.friction = constants
                    .wall
                    .friction
                    .clamp(ui_ranges.wall_friction.min, ui_ranges.wall_friction.max);
                ui.add(
                    egui::Slider::new(
                        &mut constants.wall.friction,
                        ui_ranges.wall_friction.min..=ui_ranges.wall_friction.max,
                    )
                    .text("Wall Friction")
                    .step_by(ui_ranges.wall_friction.step as f64)
                    .fixed_decimals(2),
                );
            });

            settings_section(ui, "Controls", true, |ui| {
                let pause_label = if sim_state.paused { "Resume" } else { "Pause" };
                if ui
                    .add_sized(
                        [ui.available_width(), layout.button_height],
                        egui::Button::new(egui::RichText::new(pause_label).strong())
                            .fill(egui::Color32::from_rgb(62, 86, 145)),
                    )
                    .clicked()
                {
                    sim_state.paused = !sim_state.paused;
                }

                if ui
                    .add_sized(
                        [ui.available_width(), layout.button_height],
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_rate(actual: Option<f64>, expected: f64) {
        assert!((actual.expect("rate should be available") - expected).abs() < 1e-9);
    }

    #[test]
    fn performance_uses_actual_progress_and_uncapped_wall_time() {
        let mut performance = PerfDisplayState::default();
        performance.record_frame(0.5, 4.0, 0.0004, false);
        assert_eq!(performance.smoothed_speed, None);
        assert_eq!(performance.smoothed_steps_per_sec, None);
        performance.record_frame(0.5, 4.15, 0.0004, false);
        assert_rate(performance.smoothed_fps, 2.0);
        assert_rate(performance.smoothed_speed, 0.3);
        assert_rate(performance.smoothed_steps_per_sec, 750.0);

        // A frame with no physics work contributes zero, regardless of the work cap.
        performance.record_frame(0.5, 4.15, 0.0004, false);
        let decay = (-0.5 / PERF_SMOOTHING_TAU_SEC).exp();
        assert_rate(performance.smoothed_speed, 0.3 * decay);
        assert_rate(performance.smoothed_steps_per_sec, 750.0 * decay);
    }

    #[test]
    fn pause_and_reset_clear_stale_simulation_rates() {
        let mut performance = PerfDisplayState::default();
        performance.record_frame(0.1, 0.0, 0.001, false);
        performance.record_frame(0.1, 0.1, 0.001, false);
        assert_rate(performance.smoothed_speed, 1.0);
        performance.record_frame(0.1, 0.1, 0.001, true);
        assert_rate(performance.smoothed_speed, 0.0);
        assert_rate(performance.smoothed_steps_per_sec, 0.0);
        assert_rate(performance.smoothed_fps, 10.0);

        performance.record_frame(0.1, 0.2, 0.001, false);
        assert!(performance.smoothed_speed.unwrap() > 0.0);
        performance.record_frame(0.1, 0.0, 0.001, false);
        assert_rate(performance.smoothed_speed, 0.0);
        assert_rate(performance.smoothed_steps_per_sec, 0.0);
        performance.record_frame(0.1, 0.0, 0.001, false);
        assert_rate(performance.smoothed_speed, 0.0);
    }

    #[test]
    fn invalid_performance_samples_do_not_produce_nonfinite_rates() {
        for bad_wall_dt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut performance = PerfDisplayState::default();
            performance.record_frame(0.1, 0.0, 0.001, false);
            performance.record_frame(bad_wall_dt, 0.1, 0.001, false);
            assert_eq!(performance.smoothed_fps, None);
            assert_eq!(performance.smoothed_speed, None);
            assert_eq!(performance.smoothed_steps_per_sec, None);
            performance.record_frame(0.1, 0.13, 0.001, false);
            assert_rate(performance.smoothed_speed, 0.3);
        }
        for bad_elapsed in [-1.0, f64::NAN, f64::INFINITY] {
            let mut performance = PerfDisplayState::default();
            performance.record_frame(0.1, 0.0, 0.001, false);
            performance.record_frame(0.1, bad_elapsed, 0.001, false);
            assert_eq!(performance.smoothed_speed, None);
            assert_eq!(performance.smoothed_steps_per_sec, None);
            performance.record_frame(0.1, 0.1, 0.001, false);
            assert_eq!(performance.smoothed_speed, None);
        }
        let mut performance = PerfDisplayState::default();
        performance.record_frame(0.1, 0.0, 0.0, false);
        performance.record_frame(0.1, 0.03, 0.0, false);
        assert_rate(performance.smoothed_speed, 0.3);
        assert_eq!(performance.smoothed_steps_per_sec, None);
    }

    #[test]
    fn repeated_gui_passes_only_display_recorded_performance() {
        let ctx = egui::Context::default();
        let mut performance = PerfDisplayState::default();
        performance.record_frame(0.1, 0.0, 0.001, false);
        performance.record_frame(0.1, 0.03, 0.001, false);
        let recorded = performance.clone();
        let mut constants = SimulationConstants::default();
        let ranges = UiControlRanges::default();
        let mut backend = PhysicsBackend::Gpu;
        let mut state = SimulationState::default();
        for frame in 0..4 {
            let mut output = ctx.run_pass(
                egui::RawInput {
                    time: Some(f64::from(frame)),
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1280.0, 900.0),
                    )),
                    ..Default::default()
                },
                |ctx| {
                    draw_control_panel(
                        ctx,
                        &performance,
                        &mut constants,
                        &ranges,
                        &mut backend,
                        &mut state,
                        None,
                    );
                },
            );
            output.textures_delta.clear();
            assert_eq!(performance, recorded);
            // egui hides a new window's initial sizing pass. Subsequent frames
            // are visible, and the deterministic time also completes fade-in.
            if frame == 0 {
                continue;
            }
            let text: Vec<_> = output
                .shapes
                .iter()
                .filter_map(|shape| match &shape.shape {
                    egui::Shape::Text(text) => Some(text.galley.job.text.as_str()),
                    _ => None,
                })
                .collect();
            assert!(text.contains(&"Speed: 0.30x"), "rendered text: {text:?}");
        }
    }

    fn draw_length_slider(
        ctx: &egui::Context,
        input: egui::RawInput,
        value_m: &mut f32,
    ) -> (egui::Response, egui::FullOutput) {
        let mut response = None;
        let mut output = ctx.run_ui(input, |ui| {
            response = Some(length_slider(ui, value_m, 0.001..=0.12, 0.001, "Amplitude"));
        });
        output.textures_delta.clear();
        (response.unwrap(), output)
    }

    #[test]
    fn displaying_millimeters_preserves_fractional_meter_values() {
        let ctx = egui::Context::default();
        for (original, displayed) in [(0.020475_f32, "20.475"), (0.0875, "87.500")] {
            let mut value = original;
            for _ in 0..3 {
                let (response, output) =
                    draw_length_slider(&ctx, egui::RawInput::default(), &mut value);
                assert!(!response.changed());
                assert_eq!(value.to_bits(), original.to_bits());
                let text: Vec<_> = output
                    .shapes
                    .iter()
                    .filter_map(|shape| match &shape.shape {
                        egui::Shape::Text(text) => Some(text.galley.job.text.as_str()),
                        _ => None,
                    })
                    .collect();
                assert!(text.contains(&"Amplitude [mm]"), "{text:?}");
                assert!(text.contains(&displayed), "{text:?}");
            }
        }
    }

    #[test]
    fn millimeter_slider_edits_write_meters_back() {
        let ctx = egui::Context::default();
        let mut value = 0.020_f32;
        draw_length_slider(&ctx, egui::RawInput::default(), &mut value)
            .0
            .request_focus();
        let input = egui::RawInput {
            events: vec![egui::Event::Key {
                key: egui::Key::ArrowRight,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }],
            ..Default::default()
        };
        let (response, _) = draw_length_slider(&ctx, input, &mut value);
        assert!(response.changed());
        assert!((value - 0.021).abs() < 1e-7);
    }
}

/// Draw the Particle Distribution panel.
pub fn draw_distribution_panel(
    ctx: &egui::Context,
    current: &CurrentDistribution,
    history: &DistributionHistory,
) {
    let layout = ui_layout_profile(ctx);

    let large_total = current.total_large();
    let small_total = current.total_small();
    let large_ratio = current.left_large_ratio() as f32;
    let small_ratio = current.left_small_ratio() as f32;

    egui::Window::new(egui::RichText::new("Particle Distribution").strong())
        .anchor(
            if layout.mobile {
                egui::Align2::LEFT_BOTTOM
            } else {
                egui::Align2::LEFT_TOP
            },
            layout.distribution_offset,
        )
        .default_width(layout.distribution_width)
        .min_width(layout.distribution_width)
        .max_width(layout.distribution_width)
        .collapsible(true)
        .default_open(!layout.mobile)
        .resizable(false)
        .frame(egui::Frame {
            fill: COLOR_PANEL_BG,
            stroke: egui::Stroke::new(1.0, COLOR_PANEL_BORDER),
            corner_radius: egui::CornerRadius::same(10),
            inner_margin: egui::Margin::same(layout.panel_margin),
            ..Default::default()
        })
        .show(ctx, |ui| {
            ui.spacing_mut().item_spacing = layout.item_spacing;
            settings_section(ui, "Current Distribution", true, |ui| {
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

            settings_section(ui, "History (Left ratio over time)", !layout.mobile, |ui| {
                let large_points: Vec<[f64; 2]> = history
                    .timestamps
                    .iter()
                    .zip(history.left_large_ratio.iter())
                    .map(|(t, r)| [*t, *r])
                    .collect();
                let small_points: Vec<[f64; 2]> = history
                    .timestamps
                    .iter()
                    .zip(history.left_small_ratio.iter())
                    .map(|(t, r)| [*t, *r])
                    .collect();

                Plot::new("distribution_history_plot")
                    .height(layout.plot_height)
                    .legend(Legend::default().position(Corner::LeftTop))
                    .allow_zoom(false)
                    .allow_drag([true, false])
                    .allow_scroll(false)
                    .include_y(0.0)
                    .include_y(1.0)
                    .show(ui, |plot_ui| {
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
}
