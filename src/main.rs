use clap::Parser;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "granular-clock")]
struct CliArgs {
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,
}

fn main() {
    let cli = CliArgs::parse();
    granular_clock::run_with_config_path(cli.config);
}

#[cfg(test)]
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
