use bevy::prelude::*;
use bevy::ui::Checked;

use crate::analysis::DistributionHistory;
use crate::physics::{ParticleBundle, ParticleSize, Position};
use crate::rendering::{ParticleMeshes, SimulationConfig};
use crate::simulation::{Container, OscillationParams, SimulationTime};

/// シミュレーションの状態
#[derive(Resource, Default)]
pub struct SimulationState {
    /// 一時停止中かどうか
    pub paused: bool,
    /// リセットが要求されているか
    pub reset_requested: bool,
}

/// bevy_ui control panel setup (ASCII only)
pub fn setup_bevy_ui_controls(mut commands: Commands) {
    // Root node (top-right)
    commands
        .spawn((
            ControlPanel,
            Node {
                position_type: PositionType::Absolute,
                right: Val::Px(10.0),
                top: Val::Px(10.0),
                width: Val::Px(280.0),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(Val::Px(10.0)),
                row_gap: Val::Px(8.0),
                ..default()
            },
            BackgroundColor(Color::srgba(0.1, 0.1, 0.1, 0.9)),
        ))
        .with_children(|parent| {
            // Title
            parent.spawn((
                Text::new("Simulation Settings"),
                TextFont {
                    font_size: 18.0,
                    ..default()
                },
                TextColor(Color::WHITE),
            ));

            // Oscillation toggle
            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(8.0),
                    ..default()
                })
                .with_children(|row| {
                    row.spawn((
                        OscillationToggleButton,
                        Checked,
                        Button,
                        Node {
                            width: Val::Px(24.0),
                            height: Val::Px(24.0),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            border: UiRect::all(Val::Px(2.0)),
                            ..default()
                        },
                        BorderColor::all(Color::WHITE),
                        BackgroundColor(Color::srgb(0.2, 0.5, 0.2)),
                    ))
                    .with_children(|btn| {
                        btn.spawn((
                            OscillationCheckMark,
                            Text::new("x"),
                            TextFont {
                                font_size: 14.0,
                                ..default()
                            },
                            TextColor(Color::WHITE),
                        ));
                    });
                    row.spawn((
                        Text::new("Oscillation"),
                        TextFont {
                            font_size: 14.0,
                            ..default()
                        },
                        TextColor(Color::WHITE),
                    ));
                });

            // Amplitude
            parent.spawn((
                AmplitudeText,
                Text::new("Amplitude: 0.005 m"),
                TextFont {
                    font_size: 14.0,
                    ..default()
                },
                TextColor(Color::WHITE),
            ));

            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(8.0),
                    ..default()
                })
                .with_children(|row| {
                    spawn_button(row, AmplitudeDownButton, "Amp -");
                    spawn_button(row, AmplitudeUpButton, "Amp +");
                });

            // Frequency
            parent.spawn((
                FrequencyText,
                Text::new("Frequency: 5.0 Hz"),
                TextFont {
                    font_size: 14.0,
                    ..default()
                },
                TextColor(Color::WHITE),
            ));

            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(8.0),
                    ..default()
                })
                .with_children(|row| {
                    spawn_button(row, FrequencyDownButton, "Freq -");
                    spawn_button(row, FrequencyUpButton, "Freq +");
                });

            // Separator
            parent.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(1.0),
                    margin: UiRect::vertical(Val::Px(5.0)),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.4, 0.4, 0.4)),
            ));

            // Pause button
            parent
                .spawn((
                    PauseButton,
                    Button,
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Px(32.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.3, 0.3, 0.5)),
                ))
                .with_children(|btn| {
                    btn.spawn((
                        PauseButtonText,
                        Text::new("Pause"),
                        TextFont {
                            font_size: 14.0,
                            ..default()
                        },
                        TextColor(Color::WHITE),
                    ));
                });

            // Reset button
            parent
                .spawn((
                    ResetButton,
                    Button,
                    Node {
                        width: Val::Percent(100.0),
                        height: Val::Px(32.0),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.5, 0.3, 0.3)),
                ))
                .with_children(|btn| {
                    btn.spawn((
                        Text::new("Reset"),
                        TextFont {
                            font_size: 14.0,
                            ..default()
                        },
                        TextColor(Color::WHITE),
                    ));
                });

            // Controls help
            parent.spawn((
                Node {
                    margin: UiRect::top(Val::Px(10.0)),
                    ..default()
                },
                Text::new("WASD: move, Mouse: look"),
                TextFont {
                    font_size: 11.0,
                    ..default()
                },
                TextColor(Color::srgb(0.6, 0.6, 0.6)),
            ));
        });
}

fn spawn_button<M: Component>(parent: &mut ChildSpawnerCommands, marker: M, label: &str) {
    parent
        .spawn((
            marker,
            Button,
            Node {
                width: Val::Px(80.0),
                height: Val::Px(28.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::srgb(0.3, 0.3, 0.4)),
        ))
        .with_children(|btn| {
            btn.spawn((
                Text::new(label),
                TextFont {
                    font_size: 12.0,
                    ..default()
                },
                TextColor(Color::WHITE),
            ));
        });
}

// Marker components
#[derive(Component)]
pub struct ControlPanel;

#[derive(Component)]
pub struct OscillationToggleButton;

#[derive(Component)]
pub struct OscillationCheckMark;

#[derive(Component)]
pub struct AmplitudeText;

#[derive(Component)]
pub struct AmplitudeUpButton;

#[derive(Component)]
pub struct AmplitudeDownButton;

#[derive(Component)]
pub struct FrequencyText;

#[derive(Component)]
pub struct FrequencyUpButton;

#[derive(Component)]
pub struct FrequencyDownButton;

#[derive(Component)]
pub struct PauseButton;

#[derive(Component)]
pub struct PauseButtonText;

#[derive(Component)]
pub struct ResetButton;

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

pub fn handle_amplitude_buttons(
    up_btn: Query<&Interaction, (With<AmplitudeUpButton>, Changed<Interaction>)>,
    down_btn: Query<&Interaction, (With<AmplitudeDownButton>, Changed<Interaction>)>,
    mut amplitude_text: Query<&mut Text, With<AmplitudeText>>,
    mut osc_params: ResMut<OscillationParams>,
) {
    let mut changed = false;

    if let Ok(int) = up_btn.single() {
        if *int == Interaction::Pressed {
            osc_params.amplitude = (osc_params.amplitude + 0.001).min(0.02);
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
    mut commands: Commands,
    mut sim_state: ResMut<SimulationState>,
    mut sim_time: ResMut<SimulationTime>,
    config: Res<SimulationConfig>,
    particles: Query<Entity, With<Position>>,
    meshes: Res<ParticleMeshes>,
    container: Res<Container>,
) {
    if !sim_state.reset_requested {
        return;
    }

    sim_state.reset_requested = false;
    sim_time.reset();

    // 既存の粒子を削除
    for entity in particles.iter() {
        commands.entity(entity).despawn();
    }

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

        let pos = Vec3::new(x, y, z);

        commands.spawn((
            ParticleBundle::new(pos, config.large_radius, config.density, ParticleSize::Large),
            Mesh3d(meshes.sphere.clone()),
            MeshMaterial3d(meshes.large_material.clone()),
            Transform::from_translation(pos).with_scale(Vec3::splat(config.large_radius)),
        ));
    }

    // 小粒子をスポーン
    for _ in 0..config.num_small {
        let x = rng.random_range(-spawn_area_x..spawn_area_x);
        let z = rng.random_range(-spawn_area_z..spawn_area_z);
        let y = base_y + config.small_radius + rng.random_range(0.0..0.2);

        let pos = Vec3::new(x, y, z);

        commands.spawn((
            ParticleBundle::new(pos, config.small_radius, config.density, ParticleSize::Small),
            Mesh3d(meshes.sphere.clone()),
            MeshMaterial3d(meshes.small_material.clone()),
            Transform::from_translation(pos).with_scale(Vec3::splat(config.small_radius)),
        ));
    }
}
