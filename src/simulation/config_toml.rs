use std::fmt::{Display, Formatter};
use std::path::Path;

use bevy::prelude::*;
use serde::Deserialize;

use crate::simulation::constants::{
    ContainerParams, GridSettings, MaterialProperties, OscillationParams, PhysicsConstants,
    SimulationConfig, SimulationConstants, SimulationSettings, SimulationTimeParams,
    UiControlRanges, UiSliderRange, WallProperties,
};

const EMBEDDED_CONFIG_TOML: &str = include_str!("../../simulation.toml");

#[derive(Clone)]
pub struct LoadedConfig {
    pub simulation: SimulationConstants,
    pub ui_ranges: UiControlRanges,
    pub warnings: Vec<String>,
}

#[derive(Debug)]
pub enum ConfigError {
    #[cfg(not(target_family = "wasm"))]
    Io(std::io::Error),
    Parse(toml::de::Error),
    #[cfg(target_family = "wasm")]
    UnsupportedPlatform(&'static str),
}

impl Display for ConfigError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(not(target_family = "wasm"))]
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Parse(err) => write!(f, "TOML parse error: {err}"),
            #[cfg(target_family = "wasm")]
            Self::UnsupportedPlatform(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for ConfigError {}

#[derive(Debug, Deserialize)]
struct RawRoot {
    simulation: RawSimulation,
    ui: RawUi,
}

#[derive(Debug, Deserialize)]
struct RawSimulation {
    config: RawSimulationConfig,
    container: RawContainer,
    oscillation: RawOscillation,
    time: RawTime,
    settings: RawSettings,
    physics: RawPhysics,
    material: RawMaterial,
    wall: RawWall,
    grid: RawGrid,
}

#[derive(Debug, Deserialize)]
struct RawSimulationConfig {
    large_radius: f32,
    small_radius: f32,
    density: f32,
    num_large: u32,
    num_small: u32,
}

#[derive(Debug, Deserialize)]
struct RawContainer {
    half_extents: [f32; 3],
    divider_height: f32,
    divider_thickness: f32,
    base_position: [f32; 3],
}

#[derive(Debug, Deserialize)]
struct RawOscillation {
    amplitude: f32,
    frequency: f32,
    enabled: bool,
}

#[derive(Debug, Deserialize)]
struct RawTime {
    dt: f32,
}

#[derive(Debug, Deserialize)]
struct RawSettings {
    substeps_per_frame: u32,
}

#[derive(Debug, Deserialize)]
struct RawPhysics {
    gravity: [f32; 3],
}

#[derive(Debug, Deserialize)]
struct RawMaterial {
    youngs_modulus: f32,
    poisson_ratio: f32,
    restitution: f32,
    friction: f32,
    rolling_friction: f32,
}

#[derive(Debug, Deserialize)]
struct RawWall {
    stiffness: f32,
    damping: f32,
    friction: f32,
    restitution: f32,
}

#[derive(Debug, Deserialize)]
struct RawGrid {
    cell_size: f32,
    table_size: usize,
}

#[derive(Debug, Deserialize)]
struct RawUi {
    oscillation: RawUiOscillation,
}

#[derive(Debug, Deserialize)]
struct RawUiOscillation {
    amplitude: RawUiSliderRange,
    frequency: RawUiSliderRange,
}

#[derive(Debug, Deserialize)]
struct RawUiSliderRange {
    min: f32,
    max: f32,
    step: f32,
}

pub fn load_embedded_config() -> LoadedConfig {
    match parse_loaded_config(EMBEDDED_CONFIG_TOML, "embedded config") {
        Ok(config) => config,
        Err(err) => LoadedConfig {
            simulation: SimulationConstants::default(),
            ui_ranges: UiControlRanges::default(),
            warnings: vec![format!(
                "Failed to parse embedded config: {err}. Falling back to Rust defaults."
            )],
        },
    }
}

#[cfg(not(target_family = "wasm"))]
pub fn load_config_from_path(path: &Path) -> Result<LoadedConfig, ConfigError> {
    let source = std::fs::read_to_string(path).map_err(ConfigError::Io)?;
    parse_loaded_config(&source, &format!("runtime config ({})", path.display()))
}

#[cfg(target_family = "wasm")]
pub fn load_config_from_path(_path: &Path) -> Result<LoadedConfig, ConfigError> {
    Err(ConfigError::UnsupportedPlatform(
        "runtime config path is not supported on wasm",
    ))
}

pub fn resolve_startup_config(config_path: Option<&Path>) -> LoadedConfig {
    if let Some(path) = config_path {
        match load_config_from_path(path) {
            Ok(config) => config,
            Err(err) => {
                let mut embedded = load_embedded_config();
                embedded.warnings.insert(
                    0,
                    format!(
                        "Failed to load runtime config '{}': {err}. Using embedded config.",
                        path.display()
                    ),
                );
                embedded
            }
        }
    } else {
        load_embedded_config()
    }
}

fn parse_loaded_config(source: &str, source_name: &str) -> Result<LoadedConfig, ConfigError> {
    let raw: RawRoot = toml::from_str(source).map_err(ConfigError::Parse)?;
    Ok(convert_raw_config(raw, source_name))
}

fn convert_raw_config(raw: RawRoot, source_name: &str) -> LoadedConfig {
    let defaults = SimulationConstants::default();
    let ui_defaults = UiControlRanges::default();
    let mut warnings = Vec::new();

    let amplitude_range = sanitize_slider_range(
        raw.ui.oscillation.amplitude,
        ui_defaults.oscillation_amplitude,
        source_name,
        "ui.oscillation.amplitude",
        &mut warnings,
    );
    let frequency_range = sanitize_slider_range(
        raw.ui.oscillation.frequency,
        ui_defaults.oscillation_frequency,
        source_name,
        "ui.oscillation.frequency",
        &mut warnings,
    );

    let ui_ranges = UiControlRanges {
        oscillation_amplitude: amplitude_range,
        oscillation_frequency: frequency_range,
    };

    let mut simulation = SimulationConstants {
        config: SimulationConfig {
            large_radius: sanitize_f32(
                raw.simulation.config.large_radius,
                defaults.config.large_radius,
                source_name,
                "simulation.config.large_radius",
                &mut warnings,
                |v| v > 0.0,
            ),
            small_radius: sanitize_f32(
                raw.simulation.config.small_radius,
                defaults.config.small_radius,
                source_name,
                "simulation.config.small_radius",
                &mut warnings,
                |v| v > 0.0,
            ),
            density: sanitize_f32(
                raw.simulation.config.density,
                defaults.config.density,
                source_name,
                "simulation.config.density",
                &mut warnings,
                |v| v > 0.0,
            ),
            num_large: sanitize_u32(
                raw.simulation.config.num_large,
                defaults.config.num_large,
                source_name,
                "simulation.config.num_large",
                &mut warnings,
                |v| v > 0,
            ),
            num_small: sanitize_u32(
                raw.simulation.config.num_small,
                defaults.config.num_small,
                source_name,
                "simulation.config.num_small",
                &mut warnings,
                |v| v > 0,
            ),
        },
        container: ContainerParams {
            half_extents: sanitize_positive_vec3(
                raw.simulation.container.half_extents,
                defaults.container.half_extents,
                source_name,
                "simulation.container.half_extents",
                &mut warnings,
            ),
            divider_height: sanitize_f32(
                raw.simulation.container.divider_height,
                defaults.container.divider_height,
                source_name,
                "simulation.container.divider_height",
                &mut warnings,
                |v| v > 0.0,
            ),
            divider_thickness: sanitize_f32(
                raw.simulation.container.divider_thickness,
                defaults.container.divider_thickness,
                source_name,
                "simulation.container.divider_thickness",
                &mut warnings,
                |v| v > 0.0,
            ),
            base_position: sanitize_vec3(
                raw.simulation.container.base_position,
                defaults.container.base_position,
                source_name,
                "simulation.container.base_position",
                &mut warnings,
            ),
        },
        oscillation: OscillationParams {
            amplitude: sanitize_f32(
                raw.simulation.oscillation.amplitude,
                defaults.oscillation.amplitude,
                source_name,
                "simulation.oscillation.amplitude",
                &mut warnings,
                |v| v >= 0.0,
            ),
            frequency: sanitize_f32(
                raw.simulation.oscillation.frequency,
                defaults.oscillation.frequency,
                source_name,
                "simulation.oscillation.frequency",
                &mut warnings,
                |v| v > 0.0,
            ),
            enabled: raw.simulation.oscillation.enabled,
        },
        time: SimulationTimeParams {
            dt: sanitize_f32(
                raw.simulation.time.dt,
                defaults.time.dt,
                source_name,
                "simulation.time.dt",
                &mut warnings,
                |v| v > 0.0,
            ),
        },
        settings: SimulationSettings {
            substeps_per_frame: sanitize_u32(
                raw.simulation.settings.substeps_per_frame,
                defaults.settings.substeps_per_frame,
                source_name,
                "simulation.settings.substeps_per_frame",
                &mut warnings,
                |v| v > 0,
            ),
        },
        physics: PhysicsConstants {
            gravity: sanitize_vec3(
                raw.simulation.physics.gravity,
                defaults.physics.gravity,
                source_name,
                "simulation.physics.gravity",
                &mut warnings,
            ),
        },
        material: MaterialProperties {
            youngs_modulus: sanitize_f32(
                raw.simulation.material.youngs_modulus,
                defaults.material.youngs_modulus,
                source_name,
                "simulation.material.youngs_modulus",
                &mut warnings,
                |v| v > 0.0,
            ),
            poisson_ratio: sanitize_f32(
                raw.simulation.material.poisson_ratio,
                defaults.material.poisson_ratio,
                source_name,
                "simulation.material.poisson_ratio",
                &mut warnings,
                |v| (-0.99..0.5).contains(&v),
            ),
            restitution: sanitize_f32(
                raw.simulation.material.restitution,
                defaults.material.restitution,
                source_name,
                "simulation.material.restitution",
                &mut warnings,
                |v| (0.0..=1.0).contains(&v),
            ),
            friction: sanitize_f32(
                raw.simulation.material.friction,
                defaults.material.friction,
                source_name,
                "simulation.material.friction",
                &mut warnings,
                |v| v >= 0.0,
            ),
            rolling_friction: sanitize_f32(
                raw.simulation.material.rolling_friction,
                defaults.material.rolling_friction,
                source_name,
                "simulation.material.rolling_friction",
                &mut warnings,
                |v| v >= 0.0,
            ),
        },
        wall: WallProperties {
            stiffness: sanitize_f32(
                raw.simulation.wall.stiffness,
                defaults.wall.stiffness,
                source_name,
                "simulation.wall.stiffness",
                &mut warnings,
                |v| v >= 0.0,
            ),
            damping: sanitize_f32(
                raw.simulation.wall.damping,
                defaults.wall.damping,
                source_name,
                "simulation.wall.damping",
                &mut warnings,
                |v| v >= 0.0,
            ),
            friction: sanitize_f32(
                raw.simulation.wall.friction,
                defaults.wall.friction,
                source_name,
                "simulation.wall.friction",
                &mut warnings,
                |v| v >= 0.0,
            ),
            restitution: sanitize_f32(
                raw.simulation.wall.restitution,
                defaults.wall.restitution,
                source_name,
                "simulation.wall.restitution",
                &mut warnings,
                |v| (0.0..=1.0).contains(&v),
            ),
        },
        grid: GridSettings {
            cell_size: sanitize_f32(
                raw.simulation.grid.cell_size,
                defaults.grid.cell_size,
                source_name,
                "simulation.grid.cell_size",
                &mut warnings,
                |v| v > 0.0,
            ),
            table_size: sanitize_usize(
                raw.simulation.grid.table_size,
                defaults.grid.table_size,
                source_name,
                "simulation.grid.table_size",
                &mut warnings,
                |v| v > 0,
            ),
        },
    };

    simulation.oscillation.amplitude = clamp_into_range(
        simulation.oscillation.amplitude,
        ui_ranges.oscillation_amplitude,
        source_name,
        "simulation.oscillation.amplitude",
        &mut warnings,
    );
    simulation.oscillation.frequency = clamp_into_range(
        simulation.oscillation.frequency,
        ui_ranges.oscillation_frequency,
        source_name,
        "simulation.oscillation.frequency",
        &mut warnings,
    );

    LoadedConfig {
        simulation,
        ui_ranges,
        warnings,
    }
}

fn sanitize_f32(
    value: f32,
    default: f32,
    source_name: &str,
    key: &str,
    warnings: &mut Vec<String>,
    predicate: impl Fn(f32) -> bool,
) -> f32 {
    if value.is_finite() && predicate(value) {
        value
    } else {
        warnings.push(format!(
            "{source_name}: invalid `{key}`={value:?}, using default {default:?}"
        ));
        default
    }
}

fn sanitize_u32(
    value: u32,
    default: u32,
    source_name: &str,
    key: &str,
    warnings: &mut Vec<String>,
    predicate: impl Fn(u32) -> bool,
) -> u32 {
    if predicate(value) {
        value
    } else {
        warnings.push(format!(
            "{source_name}: invalid `{key}`={value}, using default {default}"
        ));
        default
    }
}

fn sanitize_usize(
    value: usize,
    default: usize,
    source_name: &str,
    key: &str,
    warnings: &mut Vec<String>,
    predicate: impl Fn(usize) -> bool,
) -> usize {
    if predicate(value) {
        value
    } else {
        warnings.push(format!(
            "{source_name}: invalid `{key}`={value}, using default {default}"
        ));
        default
    }
}

fn sanitize_vec3(
    value: [f32; 3],
    default: Vec3,
    source_name: &str,
    key: &str,
    warnings: &mut Vec<String>,
) -> Vec3 {
    let mut out = [default.x, default.y, default.z];
    for (idx, item) in value.iter().enumerate() {
        if item.is_finite() {
            out[idx] = *item;
        } else {
            warnings.push(format!(
                "{source_name}: invalid `{key}[{idx}]`={item:?}, using default {}",
                out[idx]
            ));
        }
    }
    Vec3::new(out[0], out[1], out[2])
}

fn sanitize_positive_vec3(
    value: [f32; 3],
    default: Vec3,
    source_name: &str,
    key: &str,
    warnings: &mut Vec<String>,
) -> Vec3 {
    let mut out = [default.x, default.y, default.z];
    for (idx, item) in value.iter().enumerate() {
        if item.is_finite() && *item > 0.0 {
            out[idx] = *item;
        } else {
            warnings.push(format!(
                "{source_name}: invalid `{key}[{idx}]`={item:?}, using default {}",
                out[idx]
            ));
        }
    }
    Vec3::new(out[0], out[1], out[2])
}

fn sanitize_slider_range(
    raw: RawUiSliderRange,
    default: UiSliderRange,
    source_name: &str,
    key: &str,
    warnings: &mut Vec<String>,
) -> UiSliderRange {
    let min = sanitize_f32(
        raw.min,
        default.min,
        source_name,
        &format!("{key}.min"),
        warnings,
        |_| true,
    );
    let max = sanitize_f32(
        raw.max,
        default.max,
        source_name,
        &format!("{key}.max"),
        warnings,
        |_| true,
    );
    let mut step = sanitize_f32(
        raw.step,
        default.step,
        source_name,
        &format!("{key}.step"),
        warnings,
        |v| v > 0.0,
    );

    if min >= max {
        warnings.push(format!(
            "{source_name}: invalid range `{key}` (min >= max), using default [{}, {}] step {}",
            default.min, default.max, default.step
        ));
        return default;
    }

    if step > (max - min) {
        warnings.push(format!(
            "{source_name}: invalid `{key}.step`={step}, clamping to range width {}",
            max - min
        ));
        step = (max - min).max(f32::EPSILON);
    }

    UiSliderRange { min, max, step }
}

fn clamp_into_range(
    value: f32,
    range: UiSliderRange,
    source_name: &str,
    key: &str,
    warnings: &mut Vec<String>,
) -> f32 {
    if value < range.min || value > range.max {
        warnings.push(format!(
            "{source_name}: `{key}`={value} is outside UI range [{}, {}], clamped",
            range.min, range.max
        ));
        value.clamp(range.min, range.max)
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("granular_clock_{name}_{nanos}.toml"))
    }

    #[test]
    fn embedded_config_parses() {
        let loaded = load_embedded_config();
        assert!(!loaded.simulation.config.large_radius.is_nan());
        assert!(loaded.ui_ranges.oscillation_amplitude.min > 0.0);
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn load_config_from_path_success() {
        let path = temp_path("load_ok");
        std::fs::write(&path, EMBEDDED_CONFIG_TOML).expect("write config");
        let loaded = load_config_from_path(&path).expect("load config");
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.simulation.config.num_large, 250);
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn resolve_falls_back_to_embedded_when_runtime_missing() {
        let missing = temp_path("missing");
        let loaded = resolve_startup_config(Some(&missing));
        assert!(loaded.simulation.config.num_small > 0);
        assert!(!loaded.warnings.is_empty());
    }

    #[test]
    fn invalid_values_are_corrected() {
        let invalid = r#"
[simulation.config]
large_radius = -1.0
small_radius = 0.0
density = -10.0
num_large = 0
num_small = 0

[simulation.container]
half_extents = [0.2, -1.0, 0.1]
divider_height = -0.1
divider_thickness = -0.01
base_position = [0.0, 0.075, 0.0]

[simulation.oscillation]
amplitude = 10.0
frequency = -1.0
enabled = true

[simulation.time]
dt = -0.1

[simulation.settings]
substeps_per_frame = 0

[simulation.physics]
gravity = [0.0, -9.81, 0.0]

[simulation.material]
youngs_modulus = -1.0
poisson_ratio = 2.0
restitution = 2.0
friction = -1.0
rolling_friction = -1.0

[simulation.wall]
stiffness = -1.0
damping = -1.0
friction = -1.0
restitution = 2.0

[simulation.grid]
cell_size = 0.0
table_size = 0

[ui.oscillation.amplitude]
min = 2.0
max = 1.0
step = -1.0

[ui.oscillation.frequency]
min = 1.0
max = 10.0
step = 100.0
"#;

        let loaded = parse_loaded_config(invalid, "test config").expect("parse invalid config");
        let defaults = SimulationConstants::default();
        let ui_defaults = UiControlRanges::default();
        assert_eq!(
            loaded.simulation.config.large_radius,
            defaults.config.large_radius
        );
        assert_eq!(loaded.simulation.time.dt, defaults.time.dt);
        assert_eq!(
            loaded.ui_ranges.oscillation_amplitude.min,
            ui_defaults.oscillation_amplitude.min
        );
        assert!(!loaded.warnings.is_empty());
    }
}
