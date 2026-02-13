use bevy::prelude::*;
use bevy::ui::Checked;

use crate::analysis::DistributionHistory;
use crate::physics::{ParticleSize, ParticleStore};
use crate::simulation::{
    ContainerParams, OscillationParams, PhysicsBackend, SimulationConfig, SimulationSettings,
    SimulationState,
};

use super::markers::*;

/// シミュレーション時間の表示を更新
pub fn update_simulation_time_display(
    sim_state: Res<SimulationState>,
    mut time_text: Query<&mut Text, With<SimulationTimeText>>,
) {
    if let Ok(mut text) = time_text.single_mut() {
        let elapsed = sim_state.elapsed;
        text.0 = format!("Time: {:.1} s", elapsed);
    }
}

/// FPS とシミュレーション速度の表示を更新
pub fn update_fps_display(
    time: Res<Time>,
    settings: Res<SimulationSettings>,
    mut fps_text: Query<&mut Text, With<FpsText>>,
) {
    if let Ok(mut text) = fps_text.single_mut() {
        let dt = time.delta_secs();
        if dt > 0.0 {
            let fps = 1.0 / dt;
            let sim_steps = fps * settings.substeps_per_frame as f32;
            text.0 = format!("FPS: {:.0}  Sim: {:.0} steps/s", fps, sim_steps);
        }
    }
}

pub fn handle_oscillation_toggle(
    interaction: Query<&Interaction, (With<OscillationToggleButton>, Changed<Interaction>)>,
    mut toggle_btn: Query<
        (Entity, &mut BackgroundColor, Has<Checked>),
        With<OscillationToggleButton>,
    >,
    mut checkmark: Query<&mut Visibility, With<OscillationCheckMark>>,
    mut osc_params: ResMut<OscillationParams>,
    mut commands: Commands,
) {
    if let Ok(int) = interaction.single() {
        if *int == Interaction::Pressed {
            if let Ok((entity, mut bg, is_checked)) = toggle_btn.single_mut() {
                let new_checked = !is_checked;
                osc_params.enabled = new_checked;

                if new_checked {
                    commands.entity(entity).insert(Checked);
                    *bg = BackgroundColor(Color::srgb(0.2, 0.5, 0.2));
                } else {
                    commands.entity(entity).remove::<Checked>();
                    *bg = BackgroundColor(Color::srgb(0.3, 0.3, 0.3));
                }

                if let Ok(mut vis) = checkmark.single_mut() {
                    *vis = if new_checked {
                        Visibility::Visible
                    } else {
                        Visibility::Hidden
                    };
                }
            }
        }
    }
}

pub fn handle_physics_backend_toggle(
    interaction: Query<&Interaction, (With<PhysicsBackendToggleButton>, Changed<Interaction>)>,
    mut toggle_btn: Query<&mut BackgroundColor, With<PhysicsBackendToggleButton>>,
    mut backend_text: Query<&mut Text, With<PhysicsBackendText>>,
    mut backend: ResMut<PhysicsBackend>,
) {
    if let Ok(int) = interaction.single() {
        if *int == Interaction::Pressed {
            // トグル
            *backend = match *backend {
                PhysicsBackend::Gpu => PhysicsBackend::Cpu,
                PhysicsBackend::Cpu => PhysicsBackend::Gpu,
            };

            // UIを更新
            if let Ok(mut bg) = toggle_btn.single_mut() {
                *bg = match *backend {
                    PhysicsBackend::Gpu => BackgroundColor(Color::srgb(0.2, 0.4, 0.5)),
                    PhysicsBackend::Cpu => BackgroundColor(Color::srgb(0.5, 0.4, 0.2)),
                };
            }

            if let Ok(mut text) = backend_text.single_mut() {
                text.0 = match *backend {
                    PhysicsBackend::Gpu => "GPU".to_string(),
                    PhysicsBackend::Cpu => "CPU".to_string(),
                };
            }
        }
    }
}

pub fn handle_amplitude_buttons(
    up_btn: Query<&Interaction, (With<AmplitudeUpButton>, Changed<Interaction>)>,
    down_btn: Query<&Interaction, (With<AmplitudeDownButton>, Changed<Interaction>)>,
    mut amplitude_text: Query<&mut Text, With<AmplitudeText>>,
    mut osc_params: ResMut<OscillationParams>,
) {
    let mut changed = false;

    if let Ok(int) = up_btn.single() {
        if *int == Interaction::Pressed {
            osc_params.amplitude = (osc_params.amplitude + 0.001).min(0.1);
            changed = true;
        }
    }

    if let Ok(int) = down_btn.single() {
        if *int == Interaction::Pressed {
            osc_params.amplitude = (osc_params.amplitude - 0.001).max(0.001);
            changed = true;
        }
    }

    if changed {
        if let Ok(mut text) = amplitude_text.single_mut() {
            text.0 = format!("Amplitude: {:.3} m", osc_params.amplitude);
        }
    }
}

pub fn handle_frequency_buttons(
    up_btn: Query<&Interaction, (With<FrequencyUpButton>, Changed<Interaction>)>,
    down_btn: Query<&Interaction, (With<FrequencyDownButton>, Changed<Interaction>)>,
    mut frequency_text: Query<&mut Text, With<FrequencyText>>,
    mut osc_params: ResMut<OscillationParams>,
) {
    let mut changed = false;

    if let Ok(int) = up_btn.single() {
        if *int == Interaction::Pressed {
            osc_params.frequency = (osc_params.frequency + 1.0).min(20.0);
            changed = true;
        }
    }

    if let Ok(int) = down_btn.single() {
        if *int == Interaction::Pressed {
            osc_params.frequency = (osc_params.frequency - 1.0).max(1.0);
            changed = true;
        }
    }

    if changed {
        if let Ok(mut text) = frequency_text.single_mut() {
            text.0 = format!("Frequency: {:.1} Hz", osc_params.frequency);
        }
    }
}

pub fn handle_control_buttons(
    pause_btn: Query<&Interaction, (With<PauseButton>, Changed<Interaction>)>,
    reset_btn: Query<&Interaction, (With<ResetButton>, Changed<Interaction>)>,
    mut pause_text: Query<&mut Text, With<PauseButtonText>>,
    mut sim_state: ResMut<SimulationState>,
    mut history: ResMut<DistributionHistory>,
) {
    if let Ok(int) = pause_btn.single() {
        if *int == Interaction::Pressed {
            sim_state.paused = !sim_state.paused;
            if let Ok(mut text) = pause_text.single_mut() {
                text.0 = if sim_state.paused {
                    "Resume".to_string()
                } else {
                    "Pause".to_string()
                };
            }
        }
    }

    if let Ok(int) = reset_btn.single() {
        if *int == Interaction::Pressed {
            sim_state.reset_requested = true;
            history.clear();
        }
    }
}

#[allow(clippy::type_complexity)]
pub fn update_button_colors(
    mut buttons: Query<
        (&Interaction, &mut BackgroundColor),
        (
            With<Button>,
            Changed<Interaction>,
            Without<OscillationToggleButton>,
        ),
    >,
) {
    for (interaction, mut bg) in buttons.iter_mut() {
        let base_color = bg.0;
        *bg = match interaction {
            Interaction::Pressed => BackgroundColor(base_color.darker(0.2)),
            Interaction::Hovered => BackgroundColor(base_color.lighter(0.1)),
            Interaction::None => *bg,
        };
    }
}

/// シミュレーションのリセットを処理
pub fn handle_reset(
    mut sim_state: ResMut<SimulationState>,
    config: Res<SimulationConfig>,
    mut store: ResMut<ParticleStore>,
    container: Res<ContainerParams>,
) {
    if !sim_state.reset_requested {
        return;
    }

    sim_state.reset_requested = false;
    sim_state.reset_time();

    // 既存の粒子を削除
    store.clear();

    // 新しい粒子をスポーン
    use rand::Rng;
    let mut rng = rand::rng();

    let spawn_area_x = container.half_extents.x - config.large_radius;
    let spawn_area_z = container.half_extents.z - config.large_radius;
    let base_y = container.base_position.y - container.half_extents.y;

    // 大粒子をスポーン
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

    // 小粒子をスポーン
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
