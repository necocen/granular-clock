use bevy::prelude::*;
use bevy::ui::Checked;

use super::markers::*;

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

            // Simulation time display
            parent.spawn((
                SimulationTimeText,
                Text::new("Time: 0.0 s"),
                TextFont {
                    font_size: 14.0,
                    ..default()
                },
                TextColor(Color::srgb(0.8, 0.8, 0.2)),
            ));

            // FPS / simulation rate display
            parent.spawn((
                FpsText,
                Text::new("FPS: --  Sim: -- steps/s"),
                TextFont {
                    font_size: 12.0,
                    ..default()
                },
                TextColor(Color::srgb(0.6, 0.8, 0.6)),
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

            // Physics backend toggle
            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: Val::Px(8.0),
                    ..default()
                })
                .with_children(|row| {
                    row.spawn((
                        PhysicsBackendToggleButton,
                        Button,
                        Node {
                            width: Val::Px(60.0),
                            height: Val::Px(24.0),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            border: UiRect::all(Val::Px(2.0)),
                            ..default()
                        },
                        BorderColor::all(Color::WHITE),
                        BackgroundColor(Color::srgb(0.2, 0.4, 0.5)),
                    ))
                    .with_children(|btn| {
                        btn.spawn((
                            PhysicsBackendText,
                            Text::new("GPU"),
                            TextFont {
                                font_size: 12.0,
                                ..default()
                            },
                            TextColor(Color::WHITE),
                        ));
                    });
                    row.spawn((
                        Text::new("Physics"),
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
                Text::new("Amplitude: 0.05 m"),
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
