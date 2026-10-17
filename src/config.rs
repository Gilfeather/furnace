//! Server options from a TOML file (`--config furnace.toml`, or the `FURNACE_CONFIG`
//! environment variable).
//!
//! The file is turned into command-line arguments placed before the ones actually given, and
//! the CLI lets a later occurrence of an option override an earlier one, so options on the
//! command line always win and file values go through exactly the same parsing and checks.
//! See `furnace.example.toml` for every key.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// Environment variable naming a config file when `--config` is not given
pub const CONFIG_ENV: &str = "FURNACE_CONFIG";

#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ConfigFile {
    #[serde(default)]
    pub server: Server,
    #[serde(default)]
    pub model: ModelSection,
    #[serde(default)]
    pub batching: Batching,
}

#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Server {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub max_concurrent_requests: Option<usize>,
    pub max_body_mb: Option<usize>,
    pub request_timeout_ms: Option<u64>,
    pub log_level: Option<String>,
    pub threads: Option<u64>,
    pub grpc_port: Option<u16>,
    /// Prefer the `FURNACE_ADMIN_TOKEN` environment variable over writing it here
    pub admin_token: Option<String>,
    /// Prefer the `FURNACE_API_KEYS` environment variable over writing them here
    pub api_keys: Option<Vec<String>>,
    pub rate_limit: Option<f64>,
    pub rate_limit_burst: Option<u32>,
    pub tls_cert: Option<PathBuf>,
    pub tls_key: Option<PathBuf>,
    pub tls_client_ca: Option<PathBuf>,
}

#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelSection {
    /// Built-in model or models (`--model-name`)
    pub name: Option<OneOrMany<String>>,
    /// File-based Burn model or models (`--model-path`)
    pub path: Option<OneOrMany<PathBuf>>,
    pub weights: Option<PathBuf>,
    pub backend: Option<String>,
}

/// `"a"` or `["a", "b"]`
#[derive(Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

impl<T> OneOrMany<T> {
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        match self {
            Self::One(one) => std::slice::from_ref(one).iter(),
            Self::Many(many) => many.iter(),
        }
    }
}

#[derive(Debug, Default, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Batching {
    pub max_batch_size: Option<usize>,
    pub timeout_us: Option<u64>,
    pub workers: Option<u64>,
}

impl ConfigFile {
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Self::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        toml::from_str(text).map_err(|e| e.to_string())
    }

    /// The file as command-line arguments. `with_model` is false when the command line picks a
    /// model itself, so that a `name` here cannot clash with a `--model-path` there.
    pub fn to_args(&self, with_model: bool) -> Vec<OsString> {
        let mut args = Vec::new();
        let mut push = |flag: &str, value: Option<String>| {
            if let Some(value) = value {
                args.push(OsString::from(flag));
                args.push(OsString::from(value));
            }
        };
        let s = &self.server;
        push("--host", s.host.clone());
        push("--port", s.port.map(|v| v.to_string()));
        push(
            "--max-concurrent-requests",
            s.max_concurrent_requests.map(|v| v.to_string()),
        );
        push("--max-body-mb", s.max_body_mb.map(|v| v.to_string()));
        push(
            "--request-timeout-ms",
            s.request_timeout_ms.map(|v| v.to_string()),
        );
        push("--log-level", s.log_level.clone());
        push("--threads", s.threads.map(|v| v.to_string()));
        push("--grpc-port", s.grpc_port.map(|v| v.to_string()));
        push("--admin-token", s.admin_token.clone());
        for key in s.api_keys.iter().flatten() {
            push("--api-key", Some(key.clone()));
        }
        push("--rate-limit", s.rate_limit.map(|v| v.to_string()));
        push(
            "--rate-limit-burst",
            s.rate_limit_burst.map(|v| v.to_string()),
        );
        push("--tls-cert", s.tls_cert.as_deref().map(path_string));
        push("--tls-key", s.tls_key.as_deref().map(path_string));
        push(
            "--tls-client-ca",
            s.tls_client_ca.as_deref().map(path_string),
        );
        let m = &self.model;
        if with_model {
            for name in m.name.iter().flat_map(OneOrMany::iter) {
                push("--model-name", Some(name.clone()));
            }
            for path in m.path.iter().flat_map(OneOrMany::iter) {
                push("--model-path", Some(path_string(path)));
            }
        }
        push("--weights", m.weights.as_deref().map(path_string));
        push("--backend", m.backend.clone());
        let b = &self.batching;
        push("--max-batch-size", b.max_batch_size.map(|v| v.to_string()));
        push("--batch-timeout-us", b.timeout_us.map(|v| v.to_string()));
        push("--batch-workers", b.workers.map(|v| v.to_string()));
        args
    }
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

/// The config file named by `--config <PATH>` / `--config=<PATH>` in `args` (the command line
/// without the program name), else by [`CONFIG_ENV`]
pub fn config_path(args: &[OsString]) -> Option<PathBuf> {
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let arg = arg.to_string_lossy();
        if arg == "--" {
            break;
        }
        if arg == "--config" {
            return it.next().map(PathBuf::from);
        }
        if let Some(path) = arg.strip_prefix("--config=") {
            return Some(PathBuf::from(path));
        }
    }
    std::env::var_os(CONFIG_ENV)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// Whether `args` (the command line without the program name) chooses a model itself
pub fn picks_model(args: &[OsString]) -> bool {
    args.iter().map(|a| a.to_string_lossy()).any(|arg| {
        arg == "-m"
            || (arg.starts_with("-m") && !arg.starts_with("--"))
            || arg == "--model-name"
            || arg.starts_with("--model-name=")
            || arg == "--model-path"
            || arg.starts_with("--model-path=")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<OsString> {
        list.iter().map(OsString::from).collect()
    }

    #[test]
    fn example_file_parses_and_maps_to_flags() {
        let config = ConfigFile::parse(include_str!("../furnace.example.toml")).unwrap();
        let flags = config.to_args(true);
        let flags: Vec<_> = flags
            .iter()
            .map(|f| f.to_string_lossy().into_owned())
            .collect();
        for flag in [
            "--host",
            "--port",
            "--model-name",
            "--backend",
            "--max-batch-size",
        ] {
            assert!(flags.iter().any(|f| f == flag), "{flag} missing: {flags:?}");
        }
    }

    #[test]
    fn empty_and_partial_files() {
        assert_eq!(ConfigFile::parse("").unwrap(), ConfigFile::default());
        let config = ConfigFile::parse("[server]\nport = 8080\n").unwrap();
        assert_eq!(config.to_args(true), args(&["--port", "8080"]));
    }

    #[test]
    fn unknown_keys_and_bad_types_are_errors() {
        let typo = ConfigFile::parse("[server]\nprot = 8080\n").unwrap_err();
        assert!(typo.contains("prot"), "{typo}");
        let section = ConfigFile::parse("[serverr]\nport = 8080\n").unwrap_err();
        assert!(section.contains("serverr"), "{section}");
        assert!(ConfigFile::parse("[server]\nport = \"x\"\n").is_err());
        assert!(ConfigFile::parse("[server]\nport = 70000\n").is_err());
    }

    #[test]
    fn model_is_left_out_when_the_command_line_picks_one() {
        let config =
            ConfigFile::parse("[model]\nname = \"resnet18\"\nbackend = \"cpu\"\n").unwrap();
        assert_eq!(config.to_args(false), args(&["--backend", "cpu"]),);
        assert!(picks_model(&args(&["--model-path", "x"])));
        assert!(picks_model(&args(&["-m", "mnist"])));
        assert!(picks_model(&args(&["-mmnist"])));
        assert!(picks_model(&args(&["--model-name=mnist"])));
        assert!(!picks_model(&args(&[
            "--port",
            "3000",
            "--max-body-mb",
            "5"
        ])));
    }

    #[test]
    fn several_models() {
        let config = ConfigFile::parse("[model]\nname = [\"a\", \"b\"]\n").unwrap();
        assert_eq!(
            config.to_args(true),
            args(&["--model-name", "a", "--model-name", "b"])
        );
        let config = ConfigFile::parse("[model]\npath = \"m\"\n").unwrap();
        assert_eq!(config.to_args(true), args(&["--model-path", "m"]));
    }

    #[test]
    fn config_path_from_flag() {
        assert_eq!(
            config_path(&args(&["--port", "1", "--config", "a.toml"])),
            Some(PathBuf::from("a.toml"))
        );
        assert_eq!(
            config_path(&args(&["--config=b.toml"])),
            Some(PathBuf::from("b.toml"))
        );
    }
}
