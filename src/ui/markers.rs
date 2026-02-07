use bevy::prelude::*;

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

#[derive(Component)]
pub struct SimulationTimeText;

#[derive(Component)]
pub struct PhysicsBackendToggleButton;

#[derive(Component)]
pub struct PhysicsBackendText;
