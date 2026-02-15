use std::fmt::{Display, Formatter};
use std::path::Path;

use bevy::prelude::*;
use serde::Deserialize;

use crate::simulation::constants::{
    ContainerParams, MaterialProperties, OscillationParams, PhysicsConstants, SimulationConfig,
    SimulationConstants, SimulationSettings, SimulationTimeParams, UiControlRanges, UiSliderRange,
    WallProperties,
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
    gravity: [f32; 3],
    particle: RawParticleConfig,
    container: RawContainer,
    oscillation: RawOscillation,
    step: RawStep,
}

#[derive(Debug, Deserialize)]
struct RawParticleConfig {
    large_radius: f32,
    small_radius: f32,
    num_large: u32,
    num_small: u32,
    material: RawParticleMaterial,
}

#[derive(Debug, Deserialize, Clone)]
struct RawContainer {
    r#box: RawContainerBox,
    divider: RawContainerDivider,
    material: RawContainerMaterial,
}

#[derive(Debug, Deserialize, Clone)]
struct RawContainerBox {
    size: [f32; 3],
    center: [f32; 3],
}

#[derive(Debug, Deserialize, Clone)]
struct RawContainerDivider {
    height: f32,
    thickness: f32,
}

#[derive(Debug, Deserialize)]
struct RawOscillation {
    amplitude: f32,
    frequency: f32,
}

#[derive(Debug, Deserialize)]
struct RawStep {
    dt: f32,
    substeps_per_frame: u32,
}

#[derive(Debug, Deserialize)]
struct RawParticleMaterial {
    density: f32,
    youngs_modulus: f32,
    poisson_ratio: f32,
    restitution: f32,
    friction: f32,
    rolling_friction: f32,
}

#[derive(Debug, Deserialize, Clone)]
struct RawWall {
    stiffness: f32,
    damping: f32,
    friction: f32,
    restitution: f32,
}

type RawContainerMaterial = RawWall;

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

    let particle = SimulationConfig {
        large_radius: sanitize_f32(
            raw.simulation.particle.large_radius,
            defaults.particle.large_radius,
            source_name,
            "simulation.particle.large_radius",
            &mut warnings,
            |v| v > 0.0,
        ),
        small_radius: sanitize_f32(
            raw.simulation.particle.small_radius,
            defaults.particle.small_radius,
            source_name,
            "simulation.particle.small_radius",
            &mut warnings,
            |v| v > 0.0,
        ),
        num_large: sanitize_u32(
            raw.simulation.particle.num_large,
            defaults.particle.num_large,
            source_name,
            "simulation.particle.num_large",
            &mut warnings,
            |v| v > 0,
        ),
        num_small: sanitize_u32(
            raw.simulation.particle.num_small,
            defaults.particle.num_small,
            source_name,
            "simulation.particle.num_small",
            &mut warnings,
            |v| v > 0,
        ),
    };
    let container = resolve_container_params(
        raw.simulation.container.clone(),
        defaults.container.clone(),
        source_name,
        &mut warnings,
    );
    let oscillation = OscillationParams {
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
        enabled: defaults.oscillation.enabled,
    };
    let time = SimulationTimeParams {
        dt: sanitize_f32(
            raw.simulation.step.dt,
            defaults.time.dt,
            source_name,
            "simulation.step.dt",
            &mut warnings,
            |v| v > 0.0,
        ),
    };
    let settings = SimulationSettings {
        substeps_per_frame: sanitize_u32(
            raw.simulation.step.substeps_per_frame,
            defaults.settings.substeps_per_frame,
            source_name,
            "simulation.step.substeps_per_frame",
            &mut warnings,
            |v| v > 0,
        ),
    };
    let physics = PhysicsConstants {
        gravity: sanitize_vec3(
            raw.simulation.gravity,
            defaults.physics.gravity,
            source_name,
            "simulation.gravity",
            &mut warnings,
        ),
    };
    let material = MaterialProperties {
        density: sanitize_f32(
            raw.simulation.particle.material.density,
            defaults.material.density,
            source_name,
            "simulation.particle.material.density",
            &mut warnings,
            |v| v > 0.0,
        ),
        youngs_modulus: sanitize_f32(
            raw.simulation.particle.material.youngs_modulus,
            defaults.material.youngs_modulus,
            source_name,
            "simulation.particle.material.youngs_modulus",
            &mut warnings,
            |v| v > 0.0,
        ),
        poisson_ratio: sanitize_f32(
            raw.simulation.particle.material.poisson_ratio,
            defaults.material.poisson_ratio,
            source_name,
            "simulation.particle.material.poisson_ratio",
            &mut warnings,
            |v| (-0.99..0.5).contains(&v),
        ),
        restitution: sanitize_f32(
            raw.simulation.particle.material.restitution,
            defaults.material.restitution,
            source_name,
            "simulation.particle.material.restitution",
            &mut warnings,
            |v| (0.0..=1.0).contains(&v),
        ),
        friction: sanitize_f32(
            raw.simulation.particle.material.friction,
            defaults.material.friction,
            source_name,
            "simulation.particle.material.friction",
            &mut warnings,
            |v| v >= 0.0,
        ),
        rolling_friction: sanitize_f32(
            raw.simulation.particle.material.rolling_friction,
            defaults.material.rolling_friction,
            source_name,
            "simulation.particle.material.rolling_friction",
            &mut warnings,
            |v| v >= 0.0,
        ),
    };
    let wall = WallProperties {
        stiffness: sanitize_f32(
            raw.simulation.container.material.stiffness,
            defaults.wall.stiffness,
            source_name,
            "simulation.container.material.stiffness",
            &mut warnings,
            |v| v >= 0.0,
        ),
        damping: sanitize_f32(
            raw.simulation.container.material.damping,
            defaults.wall.damping,
            source_name,
            "simulation.container.material.damping",
            &mut warnings,
            |v| v >= 0.0,
        ),
        friction: sanitize_f32(
            raw.simulation.container.material.friction,
            defaults.wall.friction,
            source_name,
            "simulation.container.material.friction",
            &mut warnings,
            |v| v >= 0.0,
        ),
        restitution: sanitize_f32(
            raw.simulation.container.material.restitution,
            defaults.wall.restitution,
            source_name,
            "simulation.container.material.restitution",
            &mut warnings,
            |v| (0.0..=1.0).contains(&v),
        ),
    };

    let mut simulation = defaults.clone();
    simulation.set_particle(particle);
    simulation.set_container(container);
    simulation.oscillation = oscillation;
    simulation.time = time;
    simulation.settings = settings;
    simulation.physics = physics;
    simulation.material = material;
    simulation.wall = wall;

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

fn resolve_container_params(
    raw: RawContainer,
    default: ContainerParams,
    source_name: &str,
    warnings: &mut Vec<String>,
) -> ContainerParams {
    let full_size = sanitize_positive_vec3(
        raw.r#box.size,
        default.half_extents * 2.0,
        source_name,
        "simulation.container.box.size",
        warnings,
    );
    let half_extents = full_size * 0.5;
    let base_position = sanitize_vec3(
        raw.r#box.center,
        default.base_position,
        source_name,
        "simulation.container.box.center",
        warnings,
    );
    let divider_height = sanitize_f32(
        raw.divider.height,
        default.divider_height,
        source_name,
        "simulation.container.divider.height",
        warnings,
        |v| v > 0.0,
    );
    let divider_thickness = sanitize_f32(
        raw.divider.thickness,
        default.divider_thickness,
        source_name,
        "simulation.container.divider.thickness",
        warnings,
        |v| v > 0.0,
    );

    ContainerParams {
        half_extents,
        divider_height,
        divider_thickness,
        base_position,
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
        assert!(!loaded.simulation.particle.large_radius.is_nan());
        assert!(loaded.ui_ranges.oscillation_amplitude.min > 0.0);
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn load_config_from_path_success() {
        let path = temp_path("load_ok");
        std::fs::write(&path, EMBEDDED_CONFIG_TOML).expect("write config");
        let loaded = load_config_from_path(&path).expect("load config");
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.simulation.particle.num_large, 250);
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn resolve_falls_back_to_embedded_when_runtime_missing() {
        let missing = temp_path("missing");
        let loaded = resolve_startup_config(Some(&missing));
        assert!(loaded.simulation.particle.num_small > 0);
        assert!(!loaded.warnings.is_empty());
    }

    #[test]
    fn invalid_values_are_corrected() {
        let invalid = r#"
[simulation]
gravity = [0.0, -9.81, 0.0]

[simulation.particle]
large_radius = -1.0
small_radius = 0.0
num_large = 0
num_small = 0

[simulation.particle.material]
density = -10.0
youngs_modulus = -1.0
poisson_ratio = 2.0
restitution = 2.0
friction = -1.0
rolling_friction = -1.0

[simulation.container.box]
size = [0.2, -1.0, 0.1]
center = [0.0, 0.075, 0.0]

[simulation.container.divider]
height = -0.1
thickness = -0.01

[simulation.container.material]
stiffness = -1.0
damping = -1.0
friction = -1.0
restitution = 2.0

[simulation.oscillation]
amplitude = 10.0
frequency = -1.0

[simulation.step]
dt = -0.1
substeps_per_frame = 0

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
            loaded.simulation.particle.large_radius,
            defaults.particle.large_radius
        );
        assert_eq!(
            loaded.simulation.material.density,
            defaults.material.density
        );
        assert_eq!(loaded.simulation.time.dt, defaults.time.dt);
        assert_eq!(
            loaded.simulation.oscillation.enabled,
            defaults.oscillation.enabled
        );
        assert_eq!(
            loaded.ui_ranges.oscillation_amplitude.min,
            ui_defaults.oscillation_amplitude.min
        );
        assert!(!loaded.warnings.is_empty());
    }
}
