use clap::{Arg, ArgGroup, Command};
use std::net::IpAddr;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;
use tracing::{Level, error, info, warn};
use uuid::Uuid;

use error::{CliError, Result};
use furnace::{BackendKind, api, error, model, models::BuiltInModel};

// Freed operator outputs are reused instead of returned to the OS (see `furnace::allocator`)
#[global_allocator]
static ALLOCATOR: furnace::allocator::BlockCache = furnace::allocator::BlockCache::new();

#[derive(Debug)]
struct CliArgs {
    model_name: Option<String>,
    model_path: Option<PathBuf>,
    weights: Option<PathBuf>,
    host: String,
    port: u16,
    backend: BackendKind,
    max_concurrent_requests: usize,
    max_body_mb: usize,
    request_timeout_ms: Option<u64>,
    max_batch_size: usize,
    batch_timeout_us: u64,
    batch_workers: usize,
    log_level: String,
}

fn build_cli() -> Command {
    let backends = BackendKind::available_names();
    Command::new("furnace")
        .about("🔥 High-performance Rust ML inference server powered by Burn framework")
        .version(env!("CARGO_PKG_VERSION"))
        .long_about(
            "Furnace is a fast ML inference server that serves machine learning models \
            with zero Python dependencies. Built with Rust and the Burn framework.\n\n\
            EXAMPLES:\n  \
            furnace --model-name resnet18 --port 3000\n  \
            furnace --model-name resnet18 --weights ./resnet18-finetuned.bpk\n  \
            furnace --model-name resnet18 --backend wgpu --host 0.0.0.0\n  \
            furnace --model-path ./sample_model --max-concurrent-requests 8",
        )
        .arg(
            Arg::new("model-name")
                .long("model-name")
                .short('m')
                .value_name("NAME")
                .help(format!(
                    "Built-in model compiled into the binary from models/<NAME>.onnx                      (default: the only one, if there is just one). {}",
                    BuiltInModel::describe_available()
                )),
        )
        .arg(
            Arg::new("model-path")
                .long("model-path")
                .value_name("PATH")
                .help("File-based Burn model: <PATH>.bpk weights + <PATH>.json metadata")
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("weights")
                .long("weights")
                .short('w')
                .value_name("PATH")
                .help(
                    "Weight file (.bpk) for the built-in model \
                     [default: <exe dir>/<name>.bpk, then models/<name>.bpk]",
                )
                .value_parser(clap::value_parser!(PathBuf))
                .conflicts_with("model-path"),
        )
        .group(
            ArgGroup::new("model")
                .args(["model-name", "model-path"])
                .multiple(false),
        )
        .arg(
            Arg::new("port")
                .long("port")
                .short('p')
                .value_name("PORT")
                .help("Port to bind the server to (1024-65535)")
                .default_value("3000")
                .value_parser(clap::value_parser!(u16)),
        )
        .arg(
            Arg::new("host")
                .long("host")
                .value_name("HOST")
                .help("Host address to bind the server to ('0.0.0.0' for all interfaces)")
                .default_value("127.0.0.1"),
        )
        .arg(
            Arg::new("backend")
                .long("backend")
                .short('b')
                .value_name("BACKEND")
                .help(format!(
                    "Compute backend. Available in this build: {}",
                    backends.join(", ")
                ))
                .value_parser(backends)
                .default_value("cpu"),
        )
        .arg(
            Arg::new("max-concurrent-requests")
                .long("max-concurrent-requests")
                .value_name("NUM")
                .help("Maximum concurrent inferences (1-10000); extra requests get 503")
                .default_value("100")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("max-body-mb")
                .long("max-body-mb")
                .value_name("MB")
                .help("Maximum request body size in megabytes (1-1024)")
                .default_value("100")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("request-timeout-ms")
                .long("request-timeout-ms")
                .value_name("MS")
                .help("Per-request inference timeout in milliseconds; exceeded requests get 504")
                .value_parser(clap::value_parser!(u64)),
        )
        .arg(
            Arg::new("max-batch-size")
                .long("max-batch-size")
                .value_name("N")
                .help(
                    "Most items run together when concurrent requests are batched into one \
                     forward pass (1-1000); 1 turns dynamic batching off",
                )
                .default_value("8")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("batch-timeout-us")
                .long("batch-timeout-us")
                .value_name("US")
                .help(
                    "Longest a batch waits, in microseconds, for requests whose bodies are \
                     still arriving; a lone request never waits",
                )
                .default_value("2000")
                .value_parser(clap::value_parser!(u64).range(0..=1_000_000)),
        )
        .arg(
            Arg::new("batch-workers")
                .long("batch-workers")
                .value_name("N")
                .help("Batches that may run at the same time (1-64)")
                .default_value("4")
                .value_parser(clap::value_parser!(u64).range(1..=64)),
        )
        // Deprecated no-op flags kept so existing launch scripts keep working
        .arg(
            Arg::new("enable-kernel-fusion")
                .long("enable-kernel-fusion")
                .hide(true)
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("enable-autotuning")
                .long("enable-autotuning")
                .hide(true)
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("log-level")
                .long("log-level")
                .short('l')
                .value_name("LEVEL")
                .help("Set the logging level")
                .value_parser(["error", "warn", "info", "debug", "trace"])
                .default_value("info"),
        )
}

fn parse_args() -> Result<(CliArgs, Vec<&'static str>)> {
    let matches = build_cli().get_matches();

    let mut deprecated = Vec::new();
    if matches.get_flag("enable-kernel-fusion") {
        deprecated.push("--enable-kernel-fusion");
    }
    if matches.get_flag("enable-autotuning") {
        deprecated.push("--enable-autotuning");
    }

    let port = *matches.get_one::<u16>("port").unwrap();
    if port < 1024 {
        return Err(CliError::InvalidPort(format!(
            "Port {port} is below 1024 (requires root privileges)"
        ))
        .into());
    }

    let host = matches.get_one::<String>("host").unwrap().clone();
    validate_host(&host)?;

    let backend_name = matches.get_one::<String>("backend").unwrap();
    let backend = BackendKind::from_name(backend_name).map_err(|e| CliError::InvalidArgument {
        arg: "backend".to_string(),
        value: backend_name.clone(),
        reason: e.to_string(),
    })?;

    let max_concurrent_requests = *matches.get_one::<usize>("max-concurrent-requests").unwrap();
    if !(1..=10000).contains(&max_concurrent_requests) {
        return Err(CliError::InvalidArgument {
            arg: "max-concurrent-requests".to_string(),
            value: max_concurrent_requests.to_string(),
            reason: "must be between 1 and 10000".to_string(),
        }
        .into());
    }

    let max_body_mb = *matches.get_one::<usize>("max-body-mb").unwrap();
    if !(1..=1024).contains(&max_body_mb) {
        return Err(CliError::InvalidArgument {
            arg: "max-body-mb".to_string(),
            value: max_body_mb.to_string(),
            reason: "must be between 1 and 1024".to_string(),
        }
        .into());
    }

    let max_batch_size = *matches.get_one::<usize>("max-batch-size").unwrap();
    if !(1..=model::MAX_BATCH_SIZE).contains(&max_batch_size) {
        return Err(CliError::InvalidArgument {
            arg: "max-batch-size".to_string(),
            value: max_batch_size.to_string(),
            reason: format!("must be between 1 and {}", model::MAX_BATCH_SIZE),
        }
        .into());
    }

    let model_path = matches.get_one::<PathBuf>("model-path").cloned();
    let model_name = match (matches.get_one::<String>("model-name"), &model_path) {
        (Some(name), _) => Some(
            BuiltInModel::from_name(name)
                .map_err(|e| CliError::InvalidArgument {
                    arg: "model-name".to_string(),
                    value: name.clone(),
                    reason: e.to_string(),
                })?
                .name()
                .to_string(),
        ),
        // Neither a name nor a path: serve the built-in model if there is exactly one
        (None, None) => Some(
            BuiltInModel::default_model()
                .map_err(|e| CliError::InvalidArgument {
                    arg: "model-name".to_string(),
                    value: String::new(),
                    reason: e.to_string(),
                })?
                .name()
                .to_string(),
        ),
        (None, Some(_)) => None,
    };

    Ok((
        CliArgs {
            model_name,
            model_path,
            weights: matches.get_one::<PathBuf>("weights").cloned(),
            host,
            port,
            backend,
            max_concurrent_requests,
            max_body_mb,
            request_timeout_ms: matches.get_one::<u64>("request-timeout-ms").copied(),
            max_batch_size,
            batch_timeout_us: *matches.get_one::<u64>("batch-timeout-us").unwrap(),
            batch_workers: *matches.get_one::<u64>("batch-workers").unwrap() as usize,
            log_level: matches.get_one::<String>("log-level").unwrap().clone(),
        },
        deprecated,
    ))
}

/// Validate host address format
fn validate_host(host: &str) -> Result<()> {
    if host.is_empty() {
        return Err(CliError::InvalidHost("Host cannot be empty".to_string()).into());
    }
    if IpAddr::from_str(host).is_ok() || host == "localhost" {
        return Ok(());
    }
    if host.len() > 253 {
        return Err(
            CliError::InvalidHost("Hostname too long (max 253 characters)".to_string()).into(),
        );
    }
    if !host
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
    {
        return Err(
            CliError::InvalidHost("Hostname contains invalid characters".to_string()).into(),
        );
    }
    if host.starts_with(['-', '.']) || host.ends_with(['-', '.']) {
        return Err(CliError::InvalidHost(
            "Hostname cannot start or end with hyphen or dot".to_string(),
        )
        .into());
    }
    Ok(())
}

fn setup_logging(log_level: &str) -> Result<()> {
    let level = match log_level.to_lowercase().as_str() {
        "error" => Level::ERROR,
        "warn" => Level::WARN,
        "debug" => Level::DEBUG,
        "trace" => Level::TRACE,
        _ => Level::INFO,
    };

    // Production mode (FURNACE_ENV=production) switches to JSON logs
    let is_production = std::env::var("FURNACE_ENV")
        .map(|v| v.eq_ignore_ascii_case("production"))
        .unwrap_or(false);

    let env_filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(level.into())
        .from_env()
        .map_err(|e| CliError::InvalidArgument {
            arg: "log-level".to_string(),
            value: log_level.to_string(),
            reason: format!("failed to create log filter: {e}"),
        })?;

    if is_production {
        tracing_subscriber::fmt()
            .json()
            .with_current_span(false)
            .with_span_list(true)
            .with_target(true)
            .with_thread_ids(true)
            .with_file(false)
            .with_line_number(false)
            .with_env_filter(env_filter)
            .with_timer(tracing_subscriber::fmt::time::ChronoUtc::rfc_3339())
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_target(false)
            .with_thread_ids(true)
            .with_file(true)
            .with_line_number(true)
            .with_env_filter(env_filter)
            .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
            .init();
    }

    info!(log_level = %level, is_production, "🔧 Logging initialized");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let (args, deprecated) = parse_args()?;
    setup_logging(&args.log_level)?;

    let session_id = Uuid::new_v4();
    info!(
        session_id = %session_id,
        version = env!("CARGO_PKG_VERSION"),
        "🔥 Starting furnace inference server"
    );
    for flag in deprecated {
        warn!(
            "{flag} is deprecated and has no effect (Burn 0.22 configures fusion/autotune at build time)"
        );
    }

    info!(
        session_id = %session_id,
        model_name = ?args.model_name,
        model_path = ?args.model_path,
        weights = ?args.weights,
        backend = %args.backend,
        "📦 Loading model"
    );

    // Model loading (including embedded weight decoding) is blocking work
    let backend = args.backend;
    let model_name = args.model_name.clone();
    let model_path = args.model_path.clone();
    let weights = args.weights.clone();
    let loaded = tokio::task::spawn_blocking(move || match (model_path, model_name) {
        (Some(path), _) => model::load_model_with_backend(&path, backend),
        (None, Some(name)) => {
            model::load_built_in_model_with_backend(&name, backend, weights.as_deref())
        }
        (None, None) => unreachable!("parse_args defaults the model name"),
    })
    .await
    .expect("model loading task panicked");

    let model = match loaded {
        Ok(model) => {
            let model_info = model.get_info();
            info!(
                session_id = %session_id,
                model_name = %model_info.name,
                input_shape = ?model_info.input_spec.shape,
                output_shape = ?model_info.output_spec.shape,
                backend = %model_info.backend,
                model_type = %model_info.model_type,
                "✅ Model loaded successfully"
            );
            model
        }
        Err(e) => {
            error!(session_id = %session_id, error = %e, "❌ Failed to load model");
            std::process::exit(1);
        }
    };

    let server_config = api::ServerConfig {
        host: args.host.clone(),
        port: args.port,
        max_concurrent_requests: args.max_concurrent_requests,
        max_body_bytes: args.max_body_mb * 1024 * 1024,
        request_timeout: args.request_timeout_ms.map(Duration::from_millis),
        batching: api::BatchConfig {
            max_batch_size: args.max_batch_size,
            batch_timeout: Duration::from_micros(args.batch_timeout_us),
            workers: args.batch_workers,
        },
    };

    info!(
        session_id = %session_id,
        host = %args.host,
        port = args.port,
        max_concurrent_requests = args.max_concurrent_requests,
        max_body_mb = args.max_body_mb,
        request_timeout_ms = ?args.request_timeout_ms,
        max_batch_size = args.max_batch_size,
        batch_timeout_us = args.batch_timeout_us,
        batch_workers = args.batch_workers,
        "🚀 Starting HTTP server"
    );

    if let Err(e) = api::start_server_with_config(server_config, model).await {
        error!(session_id = %session_id, error = %e, "❌ Server failed to start");
        std::process::exit(1);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_definition_is_valid() {
        build_cli().debug_assert();
    }

    #[test]
    fn host_validation() {
        assert!(validate_host("127.0.0.1").is_ok());
        assert!(validate_host("::1").is_ok());
        assert!(validate_host("localhost").is_ok());
        assert!(validate_host("api.example.com").is_ok());
        assert!(validate_host("").is_err());
        assert!(validate_host("-bad.host").is_err());
        assert!(validate_host("bad_host").is_err());
    }
}
