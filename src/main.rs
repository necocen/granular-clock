mod analysis;
mod app;
mod physics;
mod rendering;
mod simulation;
mod ui;

#[cfg(not(target_family = "wasm"))]
use clap::Parser;
use mikage::wgpu;
use std::path::PathBuf;

use app::GranularClockApp;
use simulation::config_toml::resolve_startup_config;

#[cfg(not(target_family = "wasm"))]
#[derive(Debug, Parser)]
#[command(name = "granular-clock")]
struct CliArgs {
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,
}

#[cfg(not(target_family = "wasm"))]
fn main() {
    let cli = CliArgs::parse();
    run_with_config_path(cli.config);
}

#[cfg(target_family = "wasm")]
fn main() {
    run_with_config_path(None);
}

fn run_with_config_path(config_path: Option<PathBuf>) {
    let loaded = resolve_startup_config(config_path.as_deref());

    // Create OrbitCamera from CameraSettings
    let mut camera = mikage::OrbitCamera::default();
    let cam_pos = loaded.camera.position;
    let cam_target = loaded.camera.target;
    let diff = cam_pos - cam_target;
    camera.target = cam_target;
    camera.distance = diff.length();
    camera.yaw = diff.x.atan2(diff.z);
    camera.pitch = (diff.y / diff.length()).asin();

    mikage::run(
        move |ctx, target, size| {
            // mikage initializes logging before calling the app factory.
            for warning in &loaded.warnings {
                tracing::warn!("{warning}");
            }
            GranularClockApp::new(loaded, ctx, target, size)
        },
        mikage::RunConfig::with_defaults(camera)
            .with_title("Granular Clock")
            .with_size(1440, 960)
            .with_gpu(gpu_descriptor()),
    )
    .expect("failed to run Granular Clock");
}

fn gpu_descriptor() -> mikage::GpuDescriptor {
    mikage::GpuDescriptor {
        requirements: mikage::GpuRequirements {
            // The collision pipeline binds seven storage buffers.
            required_limits: Some(wgpu::Limits::default()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::CliArgs;
    use clap::Parser;
    use std::path::PathBuf;

    #[test]
    fn parse_cli_args_without_config() {
        let args = CliArgs::try_parse_from(["granular-clock"]).expect("parse args");
        assert_eq!(args.config, None);
    }

    #[test]
    fn parse_cli_args_with_config_path() {
        let args = CliArgs::try_parse_from(["granular-clock", "--config", "a.toml"])
            .expect("parse args with config");
        assert_eq!(args.config, Some(PathBuf::from("a.toml")));
    }

    #[test]
    fn parse_cli_args_with_missing_config_path_errors() {
        let result = CliArgs::try_parse_from(["granular-clock", "--config"]);
        assert!(result.is_err());
    }

    #[test]
    fn parse_cli_args_ignores_positionals() {
        let result = CliArgs::try_parse_from(["granular-clock", "unexpected"]);
        assert_eq!(
            result
                .expect_err("unexpected positional should be rejected")
                .kind(),
            clap::error::ErrorKind::UnknownArgument
        );
    }
}
