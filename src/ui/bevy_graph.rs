use bevy::prelude::*;

use crate::analysis::{CurrentDistribution, DistributionHistory};

// Marker components
#[derive(Component)]
pub struct DistributionPanel;

#[derive(Component)]
pub struct LargeLeftBar;

#[derive(Component)]
pub struct SmallLeftBar;

#[derive(Component)]
pub struct LargeCountText;

#[derive(Component)]
pub struct SmallCountText;

#[derive(Component)]
pub struct GraphContainer;

#[derive(Component)]
#[allow(dead_code)]
pub struct GraphLine {
    pub is_large: bool,
}

const GRAPH_WIDTH: f32 = 380.0;
const GRAPH_HEIGHT: f32 = 150.0;
const MAX_POINTS: usize = 600; // 600サンプル x 0.5秒 = 300秒（5分）分

/// Setup distribution graph panel
pub fn setup_distribution_graph(mut commands: Commands) {
    commands
        .spawn((
            DistributionPanel,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(10.0),
                top: Val::Px(10.0),
                width: Val::Px(400.0),
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
                Text::new("Particle Distribution"),
                TextFont {
                    font_size: 18.0,
                    ..default()
                },
                TextColor(Color::WHITE),
            ));

            // Large particles section
            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(4.0),
                    ..default()
                })
                .with_children(|section| {
                    section.spawn((
                        LargeCountText,
                        Text::new("Large: L 0 / R 0 (50.0%)"),
                        TextFont {
                            font_size: 12.0,
                            ..default()
                        },
                        TextColor(Color::srgb(1.0, 0.4, 0.4)),
                    ));

                    // Bar container
                    section
                        .spawn((
                            Node {
                                width: Val::Percent(100.0),
                                height: Val::Px(16.0),
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.2, 0.2, 0.2)),
                        ))
                        .with_children(|bar_container| {
                            bar_container.spawn((
                                LargeLeftBar,
                                Node {
                                    width: Val::Percent(50.0),
                                    height: Val::Percent(100.0),
                                    ..default()
                                },
                                BackgroundColor(Color::srgb(0.8, 0.3, 0.3)),
                            ));
                        });
                });

            // Small particles section
            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Column,
                    row_gap: Val::Px(4.0),
                    ..default()
                })
                .with_children(|section| {
                    section.spawn((
                        SmallCountText,
                        Text::new("Small: L 0 / R 0 (50.0%)"),
                        TextFont {
                            font_size: 12.0,
                            ..default()
                        },
                        TextColor(Color::srgb(0.4, 0.4, 1.0)),
                    ));

                    // Bar container
                    section
                        .spawn((
                            Node {
                                width: Val::Percent(100.0),
                                height: Val::Px(16.0),
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.2, 0.2, 0.2)),
                        ))
                        .with_children(|bar_container| {
                            bar_container.spawn((
                                SmallLeftBar,
                                Node {
                                    width: Val::Percent(50.0),
                                    height: Val::Percent(100.0),
                                    ..default()
                                },
                                BackgroundColor(Color::srgb(0.3, 0.3, 0.8)),
                            ));
                        });
                });

            // Separator
            parent.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(1.0),
                    margin: UiRect::vertical(Val::Px(4.0)),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.4, 0.4, 0.4)),
            ));

            // Graph title
            parent.spawn((
                Text::new("History (Left ratio over time)"),
                TextFont {
                    font_size: 12.0,
                    ..default()
                },
                TextColor(Color::srgb(0.7, 0.7, 0.7)),
            ));

            // Graph container with border
            parent
                .spawn((
                    GraphContainer,
                    Node {
                        width: Val::Px(GRAPH_WIDTH),
                        height: Val::Px(GRAPH_HEIGHT),
                        overflow: Overflow::clip(),
                        ..default()
                    },
                    BackgroundColor(Color::srgb(0.15, 0.15, 0.15)),
                ))
                .with_children(|graph| {
                    // 50% reference line
                    graph.spawn((
                        Node {
                            position_type: PositionType::Absolute,
                            left: Val::Px(0.0),
                            top: Val::Px(GRAPH_HEIGHT / 2.0),
                            width: Val::Percent(100.0),
                            height: Val::Px(1.0),
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.3, 0.3, 0.3)),
                    ));
                });

            // Legend
            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(16.0),
                    ..default()
                })
                .with_children(|legend| {
                    // Large legend
                    legend
                        .spawn(Node {
                            flex_direction: FlexDirection::Row,
                            align_items: AlignItems::Center,
                            column_gap: Val::Px(4.0),
                            ..default()
                        })
                        .with_children(|item| {
                            item.spawn((
                                Node {
                                    width: Val::Px(12.0),
                                    height: Val::Px(12.0),
                                    ..default()
                                },
                                BackgroundColor(Color::srgb(0.8, 0.3, 0.3)),
                            ));
                            item.spawn((
                                Text::new("Large"),
                                TextFont {
                                    font_size: 11.0,
                                    ..default()
                                },
                                TextColor(Color::srgb(0.7, 0.7, 0.7)),
                            ));
                        });

                    // Small legend
                    legend
                        .spawn(Node {
                            flex_direction: FlexDirection::Row,
                            align_items: AlignItems::Center,
                            column_gap: Val::Px(4.0),
                            ..default()
                        })
                        .with_children(|item| {
                            item.spawn((
                                Node {
                                    width: Val::Px(12.0),
                                    height: Val::Px(12.0),
                                    ..default()
                                },
                                BackgroundColor(Color::srgb(0.3, 0.3, 0.8)),
                            ));
                            item.spawn((
                                Text::new("Small"),
                                TextFont {
                                    font_size: 11.0,
                                    ..default()
                                },
                                TextColor(Color::srgb(0.7, 0.7, 0.7)),
                            ));
                        });
                });
        });
}

/// Update distribution bars and text
pub fn update_distribution_display(
    current: Res<CurrentDistribution>,
    mut large_bar: Query<&mut Node, (With<LargeLeftBar>, Without<SmallLeftBar>)>,
    mut small_bar: Query<&mut Node, (With<SmallLeftBar>, Without<LargeLeftBar>)>,
    mut large_text: Query<&mut Text, (With<LargeCountText>, Without<SmallCountText>)>,
    mut small_text: Query<&mut Text, (With<SmallCountText>, Without<LargeCountText>)>,
) {
    let large_ratio = current.left_large_ratio() as f32;
    let small_ratio = current.left_small_ratio() as f32;

    if let Ok(mut node) = large_bar.single_mut() {
        node.width = Val::Percent(large_ratio * 100.0);
    }

    if let Ok(mut node) = small_bar.single_mut() {
        node.width = Val::Percent(small_ratio * 100.0);
    }

    if let Ok(mut text) = large_text.single_mut() {
        text.0 = format!(
            "Large: L {} / R {} ({:.1}%)",
            current.left_large,
            current.right_large,
            large_ratio * 100.0
        );
    }

    if let Ok(mut text) = small_text.single_mut() {
        text.0 = format!(
            "Small: L {} / R {} ({:.1}%)",
            current.left_small,
            current.right_small,
            small_ratio * 100.0
        );
    }
}

/// Update the graph lines based on history
pub fn update_graph_lines(
    history: Res<DistributionHistory>,
    graph_container: Query<Entity, With<GraphContainer>>,
    existing_lines: Query<Entity, With<GraphLine>>,
    mut commands: Commands,
) {
    if !history.is_changed() {
        return;
    }

    let Ok(container) = graph_container.single() else {
        return;
    };

    // Remove old lines
    for entity in existing_lines.iter() {
        commands.entity(entity).despawn();
    }

    let len = history.timestamps.len();
    if len < 2 {
        return;
    }

    // Determine visible range (last MAX_POINTS points)
    let start_idx = len.saturating_sub(MAX_POINTS);
    let _visible_len = len - start_idx;

    // Draw lines as small rectangles
    let point_width = GRAPH_WIDTH / MAX_POINTS as f32;

    commands.entity(container).with_children(|graph| {
        for i in start_idx..len {
            let x = ((i - start_idx) as f32) * point_width;

            // Large particles line
            let large_ratio = history.left_large_ratio[i] as f32;
            let large_y = (1.0 - large_ratio) * GRAPH_HEIGHT;
            graph.spawn((
                GraphLine { is_large: true },
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(x),
                    top: Val::Px((large_y - 1.0).max(0.0)),
                    width: Val::Px(point_width.max(2.0)),
                    height: Val::Px(2.0),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.9, 0.4, 0.4)),
            ));

            // Small particles line
            let small_ratio = history.left_small_ratio[i] as f32;
            let small_y = (1.0 - small_ratio) * GRAPH_HEIGHT;
            graph.spawn((
                GraphLine { is_large: false },
                Node {
                    position_type: PositionType::Absolute,
                    left: Val::Px(x),
                    top: Val::Px((small_y - 1.0).max(0.0)),
                    width: Val::Px(point_width.max(2.0)),
                    height: Val::Px(2.0),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.4, 0.4, 0.9)),
            ));
        }
    });
}
