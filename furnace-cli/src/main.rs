use clap::{Arg, ArgGroup, Command};
use std::ffi::OsString;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use tracing::{Level, info, warn};
use uuid::Uuid;

use furnace::{Server, api, config, model};
use furnace_burn::BackendKind;
use furnace_burn::models::BuiltInModel;

/// What stops the command before it serves: shown as `furnace: <message>`, exit code 2
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// A command-line option, or the config file, that cannot be used
#[derive(Debug, thiserror::Error)]
enum CliError {
    #[error("Invalid argument '{arg}': {value} - {reason}")]
    Argument {
        arg: String,
        value: String,
        reason: String,
    },

    #[error("Invalid port number: {0}")]
    Port(String),

    #[error("Invalid host address: {0}")]
    Host(String),
}

/// Loads a model again for `POST /admin/models/{model}/reload`, from the given weight file or,
/// with `None`, as configured. Blocking.
type Reloader = Arc<dyn Fn(Option<&Path>) -> furnace::Result<model::Model> + Send + Sync>;

// Freed operator outputs are reused instead of returned to the OS (see `furnace::allocator`)
#[global_allocator]
static ALLOCATOR: furnace::allocator::BlockCache = furnace::allocator::BlockCache::new();

#[derive(Debug)]
struct CliArgs {
    /// Built-in models to serve; the first is the default one
    model_names: Vec<String>,
    /// File-based models to serve (instead of built-in ones)
    model_paths: Vec<PathBuf>,
    weights: Option<PathBuf>,
    admin_token: Option<String>,
    api_keys: Vec<String>,
    rate_limit: Option<furnace::access::ClientRateLimit>,
    tls: Option<api::TlsSettings>,
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
    threads: Option<usize>,
    grpc_port: Option<u16>,
}

fn build_cli() -> Command {
    let backends = BackendKind::available_names();
    Command::new("furnace")
        // A later occurrence of an option wins, so the command line overrides the config file,
        // whose options are placed first (see `furnace::config`)
        .args_override_self(true)
        .about("🔥 High-performance Rust ML inference server powered by Burn framework")
        .version(env!("CARGO_PKG_VERSION"))
        .long_about(
            "Furnace is a fast ML inference server that serves machine learning models \
            with zero Python dependencies. Built with Rust and the Burn framework.\n\n\
            EXAMPLES:\n  \
            furnace --model-name resnet18 --port 3000\n  \
            furnace --model-name resnet18 --weights ./resnet18-finetuned.bpk\n  \
            furnace --model-name resnet18 --backend wgpu --host 0.0.0.0\n  \
            furnace --model-name resnet18,mnist --admin-token \"$TOKEN\"\n  \
            furnace --model-path ./sample_model --max-concurrent-requests 8\n  \
            furnace --config furnace.toml --port 8080",
        )
        .arg(
            Arg::new("config")
                .long("config")
                .value_name("PATH")
                .value_parser(clap::value_parser!(PathBuf))
                .help(format!(
                    "Read options from a TOML file (see furnace.example.toml); options given on \
                    the command line override it. Also read from ${}",
                    furnace::config::CONFIG_ENV
                )),
        )
        .arg(
            Arg::new("model-name")
                .long("model-name")
                .short('m')
                .value_name("NAME")
                .action(clap::ArgAction::Append)
                .value_delimiter(',')
                .help(format!(
                    "Built-in model compiled into the binary from furnace-burn/models/<NAME>.onnx (default: the only one, if there is just one). \
                    Several (comma-separated or repeated) are served side by side, the first by /predict. {}",
                    BuiltInModel::describe_available()
                )),
        )
        .arg(
            Arg::new("model-path")
                .long("model-path")
                .value_name("PATH")
                .action(clap::ArgAction::Append)
                .help(
                    "File-based Burn model: <PATH>.bpk weights + <PATH>.json metadata; repeat to \
                    serve several, the first by /predict",
                )
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("weights")
                .long("weights")
                .short('w')
                .value_name("PATH")
                .help(
                    "Weight file (.bpk) for the built-in model, when serving one \
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
                     forward pass (1-1000); the default, 1, leaves batching off. Try 8 for \
                     convolutional models and on GPUs",
                )
                .default_value("1")
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
            Arg::new("grpc-port")
                .long("grpc-port")
                .value_name("PORT")
                .help(
                    "Also serve the KServe / Triton Open Inference Protocol over gRPC on this \
                    port (builds with the `grpc` feature)",
                )
                .value_parser(clap::value_parser!(u16)),
        )
        .arg(
            Arg::new("admin-token")
                .long("admin-token")
                .value_name("TOKEN")
                .help(format!(
                    "Enable the admin API (POST /admin/models/{{model}}/reload) for requests with \
                    'Authorization: Bearer <TOKEN>'. Also read from ${ADMIN_TOKEN_ENV}, which \
                    keeps it out of the process list"
                )),
        )
        .arg(
            Arg::new("api-key")
                .long("api-key")
                .value_name("KEY")
                .action(clap::ArgAction::Append)
                .help(format!(
                    "Require this key ('Authorization: Bearer <KEY>' or 'X-API-Key: <KEY>') on \
                    every route but the health probes; repeat for several. Also read from \
                    ${API_KEYS_ENV} (comma-separated), which keeps them out of the process list"
                )),
        )
        .arg(
            Arg::new("rate-limit")
                .long("rate-limit")
                .value_name("PER_SECOND")
                .help(
                    "Requests a second each client may send on average, beyond which it gets \
                    429; clients are their API key, or their IP address without keys",
                )
                .value_parser(clap::value_parser!(f64)),
        )
        .arg(
            Arg::new("rate-limit-burst")
                .long("rate-limit-burst")
                .value_name("N")
                .help("Requests a client may send at once under --rate-limit [default: the rate, at least 1]")
                .value_parser(clap::value_parser!(u32).range(1..))
                .requires("rate-limit"),
        )
        .arg(
            Arg::new("tls-cert")
                .long("tls-cert")
                .value_name("PEM")
                .help(
                    "Serve HTTPS (and gRPC over TLS) with this PEM certificate chain; reloaded                     when the file changes (builds with the `tls` feature)",
                )
                .value_parser(clap::value_parser!(PathBuf))
                .requires("tls-key"),
        )
        .arg(
            Arg::new("tls-key")
                .long("tls-key")
                .value_name("PEM")
                .help("PEM private key for --tls-cert: ECDSA P-256 / P-384 or Ed25519")
                .value_parser(clap::value_parser!(PathBuf))
                .requires("tls-cert"),
        )
        .arg(
            Arg::new("tls-client-ca")
                .long("tls-client-ca")
                .value_name("PEM")
                .help("Mutual TLS: require client certificates signed by these PEM CA certificates")
                .value_parser(clap::value_parser!(PathBuf))
                .requires("tls-cert"),
        )
        .arg(
            Arg::new("threads")
                .long("threads")
                .value_name("N")
                .help(
                    "Threads for inference (default: one per physical core, or \
                    $RAYON_NUM_THREADS)",
                )
                .value_parser(clap::value_parser!(u64).range(1..=4096)),
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
    let given: Vec<OsString> = std::env::args_os().collect();
    let (program, given) = given.split_first().expect("program name");
    let mut argv = vec![program.clone()];
    if let Some(path) = config::config_path(given) {
        let file = config::ConfigFile::load(&path).map_err(|reason| CliError::Argument {
            arg: "config".to_string(),
            value: path.display().to_string(),
            reason,
        })?;
        argv.extend(file.to_args(!config::picks_model(given)));
    }
    argv.extend(given.iter().cloned());
    let matches = build_cli().get_matches_from(argv);

    let mut deprecated = Vec::new();
    if matches.get_flag("enable-kernel-fusion") {
        deprecated.push("--enable-kernel-fusion");
    }
    if matches.get_flag("enable-autotuning") {
        deprecated.push("--enable-autotuning");
    }

    let port = *matches.get_one::<u16>("port").unwrap();
    if port < 1024 {
        return Err(CliError::Port(format!(
            "Port {port} is below 1024 (requires root privileges)"
        ))
        .into());
    }

    let host = matches.get_one::<String>("host").unwrap().clone();
    validate_host(&host)?;

    let backend_name = matches.get_one::<String>("backend").unwrap();
    let backend = BackendKind::from_name(backend_name).map_err(|e| CliError::Argument {
        arg: "backend".to_string(),
        value: backend_name.clone(),
        reason: e.to_string(),
    })?;

    let max_concurrent_requests = *matches.get_one::<usize>("max-concurrent-requests").unwrap();
    if !(1..=10000).contains(&max_concurrent_requests) {
        return Err(CliError::Argument {
            arg: "max-concurrent-requests".to_string(),
            value: max_concurrent_requests.to_string(),
            reason: "must be between 1 and 10000".to_string(),
        }
        .into());
    }

    let max_body_mb = *matches.get_one::<usize>("max-body-mb").unwrap();
    if !(1..=1024).contains(&max_body_mb) {
        return Err(CliError::Argument {
            arg: "max-body-mb".to_string(),
            value: max_body_mb.to_string(),
            reason: "must be between 1 and 1024".to_string(),
        }
        .into());
    }

    let max_batch_size = *matches.get_one::<usize>("max-batch-size").unwrap();
    if !(1..=model::MAX_BATCH_SIZE).contains(&max_batch_size) {
        return Err(CliError::Argument {
            arg: "max-batch-size".to_string(),
            value: max_batch_size.to_string(),
            reason: format!("must be between 1 and {}", model::MAX_BATCH_SIZE),
        }
        .into());
    }

    let model_paths: Vec<PathBuf> = matches
        .get_many::<PathBuf>("model-path")
        .into_iter()
        .flatten()
        .cloned()
        .collect();
    let mut model_names = Vec::new();
    for name in matches
        .get_many::<String>("model-name")
        .into_iter()
        .flatten()
    {
        // Its error names the option already
        let name = BuiltInModel::from_name(name)?.name().to_string();
        if model_names.contains(&name) {
            return Err(CliError::Argument {
                arg: "model-name".to_string(),
                value: name,
                reason: "given twice".to_string(),
            }
            .into());
        }
        model_names.push(name);
    }
    // Neither a name nor a path: serve the built-in model if there is exactly one
    if model_names.is_empty() && model_paths.is_empty() {
        model_names.push(
            BuiltInModel::default_model()
                .ok_or_else(|| {
                    format!(
                        "no model to serve: choose one with --model-name or --model-path. {}",
                        BuiltInModel::describe_available()
                    )
                })?
                .name()
                .to_string(),
        );
    }

    let weights = matches.get_one::<PathBuf>("weights").cloned();
    if let Some(weights) = &weights
        && model_names.len() > 1
    {
        return Err(CliError::Argument {
            arg: "weights".to_string(),
            value: weights.display().to_string(),
            reason: "applies to a single built-in model; with several, reload one with new \
                weights through the admin API instead"
                .to_string(),
        }
        .into());
    }

    let admin_token = matches
        .get_one::<String>("admin-token")
        .cloned()
        .or_else(|| std::env::var(ADMIN_TOKEN_ENV).ok())
        .filter(|token| !token.is_empty());
    let mut api_keys: Vec<String> = matches
        .get_many::<String>("api-key")
        .into_iter()
        .flatten()
        .cloned()
        .collect();
    if let Ok(keys) = std::env::var(API_KEYS_ENV) {
        api_keys.extend(keys.split(',').map(str::trim).map(str::to_string));
    }
    api_keys.retain(|key| !key.is_empty());
    if api_keys.iter().any(|key| key.len() < 16) {
        return Err(CliError::Argument {
            arg: "api-key".to_string(),
            value: "<hidden>".to_string(),
            reason: "keys must be at least 16 characters".to_string(),
        }
        .into());
    }
    api_keys.sort();
    api_keys.dedup();

    let tls = matches
        .get_one::<PathBuf>("tls-cert")
        .map(|cert| api::TlsSettings {
            cert: cert.clone(),
            key: matches
                .get_one::<PathBuf>("tls-key")
                .cloned()
                .expect("--tls-cert requires --tls-key"),
            client_ca: matches.get_one::<PathBuf>("tls-client-ca").cloned(),
        });

    let rate_limit = match matches.get_one::<f64>("rate-limit").copied() {
        None => None,
        Some(per_second) if per_second.is_finite() && per_second > 0.0 => {
            Some(furnace::access::ClientRateLimit {
                per_second,
                burst: matches
                    .get_one::<u32>("rate-limit-burst")
                    .copied()
                    .unwrap_or_else(|| per_second.ceil().clamp(1.0, f64::from(u32::MAX)) as u32),
            })
        }
        Some(per_second) => {
            return Err(CliError::Argument {
                arg: "rate-limit".to_string(),
                value: per_second.to_string(),
                reason: "must be a positive number".to_string(),
            }
            .into());
        }
    };

    if let Some(token) = &admin_token
        && token.len() < 16
    {
        return Err(CliError::Argument {
            arg: "admin-token".to_string(),
            value: "<hidden>".to_string(),
            reason: "must be at least 16 characters".to_string(),
        }
        .into());
    }

    Ok((
        CliArgs {
            model_names,
            model_paths,
            weights,
            admin_token,
            api_keys,
            rate_limit,
            tls,
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
            threads: matches.get_one::<u64>("threads").map(|&n| n as usize),
            grpc_port: matches.get_one::<u16>("grpc-port").copied(),
        },
        deprecated,
    ))
}

/// Validate host address format
fn validate_host(host: &str) -> Result<()> {
    if host.is_empty() {
        return Err(CliError::Host("Host cannot be empty".to_string()).into());
    }
    if IpAddr::from_str(host).is_ok() || host == "localhost" {
        return Ok(());
    }
    if host.len() > 253 {
        return Err(CliError::Host("Hostname too long (max 253 characters)".to_string()).into());
    }
    if !host
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
    {
        return Err(CliError::Host("Hostname contains invalid characters".to_string()).into());
    }
    if host.starts_with(['-', '.']) || host.ends_with(['-', '.']) {
        return Err(
            CliError::Host("Hostname cannot start or end with hyphen or dot".to_string()).into(),
        );
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
        .map_err(|e| CliError::Argument {
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
async fn main() -> std::process::ExitCode {
    match run().await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("furnace: {e}");
            std::process::ExitCode::from(2)
        }
    }
}

async fn run() -> Result<()> {
    let (args, deprecated) = parse_args()?;
    setup_logging(&args.log_level)?;

    let session_id = Uuid::new_v4();
    info!(
        session_id = %session_id,
        version = env!("CARGO_PKG_VERSION"),
        "🔥 Starting furnace inference server"
    );
    let threads = furnace::threads::init(args.threads);
    info!("🧵 {threads} inference threads");
    for flag in deprecated {
        warn!(
            "{flag} is deprecated and has no effect (Burn 0.22 configures fusion/autotune at build time)"
        );
    }

    info!(
        session_id = %session_id,
        model_names = ?args.model_names,
        model_paths = ?args.model_paths,
        weights = ?args.weights,
        backend = %args.backend,
        "📦 Loading models"
    );

    // Model loading (including embedded weight decoding) is blocking work
    let reloaders = reloaders(&args);
    let loaded = tokio::task::spawn_blocking(move || {
        reloaders
            .into_iter()
            .map(|reload| reload(None).map(|model| (model, reload)))
            .collect::<furnace::Result<Vec<_>>>()
    })
    .await
    .expect("model loading task panicked");

    let models = loaded.map_err(|e| format!("cannot load the model: {e}"))?;
    let mut names = Vec::new();
    for (model, _) in &models {
        let model_info = model.get_info();
        if names.contains(&model_info.name) {
            return Err(format!("two models are named '{}'", model_info.name).into());
        }
        names.push(model_info.name.clone());
        info!(
            session_id = %session_id,
            model_name = %model_info.name,
            input_shape = ?model_info.input_spec.shape,
            output_shape = ?model_info.output_spec.shape,
            backend = %model_info.backend,
            model_type = %model_info.model_type,
            "✅ Model loaded successfully"
        );
    }
    if args.admin_token.is_some() {
        info!("🔑 Admin API enabled: POST /admin/models/{{model}}/reload");
    }

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
        grpc_port: args.grpc_port,
        admin_token: args.admin_token.clone(),
        api_keys: args.api_keys.clone(),
        rate_limit: args.rate_limit,
        tls: args.tls.clone(),
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

    let server = models.into_iter().fold(
        Server::with_config(server_config),
        |server, (model, reload)| server.reloadable_model(model, move |weights| reload(weights)),
    );
    server.serve().await?;
    Ok(())
}

/// Environment variable read for `--admin-token`
const ADMIN_TOKEN_ENV: &str = "FURNACE_ADMIN_TOKEN";
/// Environment variable with comma-separated `--api-key`s
const API_KEYS_ENV: &str = "FURNACE_API_KEYS";

/// One loader per model to serve. Called with `None` it loads the model as configured; the
/// admin API calls it again to reload, possibly with a new weight file.
fn reloaders(args: &CliArgs) -> Vec<Reloader> {
    let backend = args.backend;
    let built_in = args.model_names.iter().map(|name| {
        let name = name.clone();
        let weights = args.weights.clone();
        Arc::new(move |new_weights: Option<&Path>| {
            furnace_burn::load_built_in_model_with_backend(
                &name,
                backend,
                new_weights.or(weights.as_deref()),
            )
        }) as Reloader
    });
    let files = args.model_paths.iter().map(|path| {
        let path = path.clone();
        Arc::new(move |new_weights: Option<&Path>| match new_weights {
            // A file model is a .bpk/.json pair; a new path names another pair
            Some(other) => furnace_burn::load_model_with_backend(other, backend),
            None => furnace_burn::load_model_with_backend(&path, backend),
        }) as Reloader
    });
    built_in.chain(files).collect()
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
