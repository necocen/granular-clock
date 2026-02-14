use bevy::prelude::*;

pub mod bevy_graph;
pub mod handlers;
pub mod markers;
pub mod setup;

pub use bevy_graph::*;
pub use handlers::*;
pub use setup::*;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_bevy_ui_controls)
            .add_systems(Startup, setup_distribution_graph)
            .add_systems(Update, handle_oscillation_toggle)
            .add_systems(Update, handle_physics_backend_toggle)
            .add_systems(Update, handle_amplitude_buttons)
            .add_systems(Update, handle_frequency_buttons)
            .add_systems(Update, handle_control_buttons)
            .add_systems(Update, update_button_colors)
            .add_systems(Update, update_distribution_display)
            .add_systems(Update, update_graph_lines)
            .add_systems(Update, update_simulation_time_display)
            .add_systems(Update, update_fps_display)
            .add_systems(Update, handle_reset);
    }
}
