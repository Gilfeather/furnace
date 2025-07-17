use clap::{Arg, Command};
use std::fs;
use std::net::IpAddr;
use std::path::PathBuf;
use std::str::FromStr;
use tracing::{error, info, Level};
use uuid::Uuid;

use error::{CliError, Result};
use furnace::{api, error, model};

#[derive(Debug)]
struct CliArgs {
    model_path: PathBuf,
    host: String,
    port: u16,
    backend: Option<String>,
    max_concurrent_requests: Option<usize>,
    enable_kernel_fusion: bool,
    enable_autotuning: bool,
    log_level: String,
}

fn parse_args() -> Result<CliArgs> {
    let matches = Command::new("furnace")
        .about("🔥 High-performance Rust ML inference server powered by Burn framework")
        .version(env!("CARGO_PKG_VERSION"))
        .long_about(
            "Furnace is a blazingly fast ML inference server that serves machine learning models \
            with zero Python dependencies. Built with Rust for maximum performance and the Burn \
            framework for native ML operations.\n\n\
            EXAMPLES:\n  \
            furnace --model-path ./model.burn --port 8080\n  \
            furnace --model-path ./model.burn --host 0.0.0.0 --backend wgpu\n  \
            furnace --model-path ./model.burn --enable-kernel-fusion --enable-autotuning"
        )
        .arg(
            Arg::new("model-path")
                .long("model-path")
                .short('m')
                .value_name("PATH")
                .help("Path to the .burn or .mpk model file")
                .long_help(
                    "Path to the model file to load for inference. \
                    Supports both .burn and .mpk (MessagePack) formats. \
                    The file must exist, be readable, and have a supported extension."
                )
                .required(true),
        )
        .arg(
            Arg::new("port")
                .long("port")
                .short('p')
                .value_name("PORT")
                .help("Port to bind the server to (1024-65535)")
                .long_help(
                    "TCP port number to bind the HTTP server to. Must be between 1024 and 65535. \
                    Ports below 1024 require root privileges and are not allowed."
                )
                .default_value("3000")
                .value_parser(clap::value_parser!(u16)),
        )
        .arg(
            Arg::new("host")
                .long("host")
                .value_name("HOST")
                .help("Host address to bind the server to")
                .long_help(
                    "IP address or hostname to bind the HTTP server to. \
                    Use '127.0.0.1' for localhost only, '0.0.0.0' for all interfaces."
                )
                .default_value("127.0.0.1"),
        )
        .arg(
            Arg::new("backend")
                .long("backend")
                .short('b')
                .value_name("BACKEND")
                .help("Backend to use for inference (cpu, wgpu, metal, cuda)")
                .long_help(
                    "Burn backend to use for model inference:\n  \
                    cpu   - CPU backend (always available)\n  \
                    wgpu  - WebGPU backend (cross-platform GPU)\n  \
                    metal - Metal backend (macOS GPU)\n  \
                    cuda  - CUDA backend (NVIDIA GPU)\n\n\
                    If the specified backend is not available, falls back to CPU."
                )
                .value_parser(["cpu", "wgpu", "metal", "cuda"])
                .default_value("cpu"),
        )
        .arg(
            Arg::new("max-concurrent-requests")
                .long("max-concurrent-requests")
                .value_name("NUM")
                .help("Maximum number of concurrent inference requests (1-10000)")
                .long_help(
                    "Maximum number of inference requests that can be processed concurrently. \
                    Higher values allow more throughput but consume more memory. \
                    Requests exceeding this limit will receive 503 Service Unavailable."
                )
                .default_value("100")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("enable-kernel-fusion")
                .long("enable-kernel-fusion")
                .help("Enable kernel fusion optimizations")
                .long_help(
                    "Enable Burn's kernel fusion optimizations to reduce memory copy overhead \
                    and GPU kernel launch overhead by fusing operations like GELU and MatMul."
                )
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("enable-autotuning")
                .long("enable-autotuning")
                .help("Enable autotuning cache optimizations")
                .long_help(
                    "Enable Burn's autotuning cache to optimize matrix operations based on size. \
                    Caches optimal kernel configurations for improved performance on repeated operations."
                )
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("log-level")
                .long("log-level")
                .short('l')
                .value_name("LEVEL")
                .help("Set the logging level")
                .long_help(
                    "Set the minimum logging level:\n  \
                    error - Only error messages\n  \
                    warn  - Warnings and errors\n  \
                    info  - Informational messages (default)\n  \
                    debug - Debug information\n  \
                    trace - Detailed trace information"
                )
                .value_parser(["error", "warn", "info", "debug", "trace"])
                .default_value("info"),
        )
        .get_matches();

    // Extract and validate arguments
    let model_path_str = matches
        .get_one::<String>("model-path")
        .ok_or_else(|| CliError::MissingArgument("model-path".to_string()))?;

    // Port validation with range check
    let port = *matches.get_one::<u16>("port").unwrap();
    if port < 1024 {
        return Err(CliError::InvalidPort(format!(
            "Port {port} is below 1024 (requires root privileges)"
        ))
        .into());
    }

    let host = matches.get_one::<String>("host").unwrap().clone();

    // Enhanced host validation
    validate_host(&host)?;

    let backend = matches.get_one::<String>("backend").cloned();

    // Max concurrent requests validation with range check
    let max_concurrent_requests = *matches.get_one::<usize>("max-concurrent-requests").unwrap();
    if max_concurrent_requests == 0 {
        return Err(CliError::InvalidArgument {
            arg: "max-concurrent-requests".to_string(),
            value: max_concurrent_requests.to_string(),
            reason: "must be greater than 0".to_string(),
        }
        .into());
    }
    if max_concurrent_requests > 10000 {
        return Err(CliError::InvalidArgument {
            arg: "max-concurrent-requests".to_string(),
            value: max_concurrent_requests.to_string(),
            reason: "must be 10000 or less".to_string(),
        }
        .into());
    }
    let max_concurrent_requests = Some(max_concurrent_requests);

    let enable_kernel_fusion = matches.get_flag("enable-kernel-fusion");
    let enable_autotuning = matches.get_flag("enable-autotuning");
    let log_level = matches.get_one::<String>("log-level").unwrap().clone();

    let model_path = PathBuf::from(model_path_str);

    // Enhanced model path validation
    validate_model_path(&model_path)?;

    Ok(CliArgs {
        model_path,
        host,
        port,
        backend,
        max_concurrent_requests,
        enable_kernel_fusion,
        enable_autotuning,
        log_level,
    })
}

/// Validate host address format and accessibility
fn validate_host(host: &str) -> Result<()> {
    if host.is_empty() {
        return Err(CliError::InvalidHost("Host cannot be empty".to_string()).into());
    }

    // Try to parse as IP address first
    if let Ok(_ip) = IpAddr::from_str(host) {
        return Ok(());
    }

    // If not an IP, validate as hostname
    if host == "localhost" {
        return Ok(());
    }

    // Basic hostname validation
    if host.len() > 253 {
        return Err(
            CliError::InvalidHost("Hostname too long (max 253 characters)".to_string()).into(),
        );
    }

    // Check for valid hostname characters
    if !host
        .chars()
        .all(|c| c.is_alphanumeric() || c == '.' || c == '-')
    {
        return Err(
            CliError::InvalidHost("Hostname contains invalid characters".to_string()).into(),
        );
    }

    // Check that hostname doesn't start or end with hyphen or dot
    if host.starts_with('-') || host.ends_with('-') || host.starts_with('.') || host.ends_with('.')
    {
        return Err(CliError::InvalidHost(
            "Hostname cannot start or end with hyphen or dot".to_string(),
        )
        .into());
    }

    Ok(())
}

/// Validate model path existence, permissions, and format
fn validate_model_path(model_path: &PathBuf) -> Result<()> {
    // Check if path exists
    if !model_path.exists() {
        return Err(CliError::InvalidModelPath {
            path: model_path.clone(),
            reason: "file does not exist".to_string(),
        }
        .into());
    }

    // Check if it's a file (not a directory)
    if !model_path.is_file() {
        return Err(CliError::InvalidModelPath {
            path: model_path.clone(),
            reason: "path is not a file".to_string(),
        }
        .into());
    }

    // Validate .burn or .mpk extension
    match model_path.extension() {
        Some(ext) if ext == "burn" || ext == "mpk" => {}
        Some(ext) => {
            return Err(CliError::InvalidModelPath {
                path: model_path.clone(),
                reason: format!(
                    "invalid extension '.{}', expected '.burn' or '.mpk'",
                    ext.to_string_lossy()
                ),
            }
            .into());
        }
        None => {
            return Err(CliError::InvalidModelPath {
                path: model_path.clone(),
                reason: "file must have .burn or .mpk extension".to_string(),
            }
            .into());
        }
    }

    // Check file permissions (readable)
    match fs::metadata(model_path) {
        Ok(metadata) => {
            if metadata.permissions().readonly() && !metadata.is_file() {
                return Err(CliError::InvalidModelPath {
                    path: model_path.clone(),
                    reason: "file is not readable".to_string(),
                }
                .into());
            }
        }
        Err(e) => {
            return Err(CliError::InvalidModelPath {
                path: model_path.clone(),
                reason: format!("cannot access file metadata: {e}"),
            }
            .into());
        }
    }

    // Check file size (basic sanity check)
    match fs::metadata(model_path) {
        Ok(metadata) => {
            let size = metadata.len();
            if size == 0 {
                return Err(CliError::InvalidModelPath {
                    path: model_path.clone(),
                    reason: "file is empty".to_string(),
                }
                .into());
            }
            if size > 10 * 1024 * 1024 * 1024 {
                // 10GB limit
                return Err(CliError::InvalidModelPath {
                    path: model_path.clone(),
                    reason: "file is too large (>10GB)".to_string(),
                }
                .into());
            }
        }
        Err(e) => {
            return Err(CliError::InvalidModelPath {
                path: model_path.clone(),
                reason: format!("cannot read file size: {e}"),
            }
            .into());
        }
    }

    Ok(())
}

fn setup_logging(log_level: &str) -> Result<()> {
    let level = match log_level.to_lowercase().as_str() {
        "error" => Level::ERROR,
        "warn" => Level::WARN,
        "info" => Level::INFO,
        "debug" => Level::DEBUG,
        "trace" => Level::TRACE,
        _ => Level::INFO,
    };

    // Check if we're in production mode (via environment variable)
    let is_production = std::env::var("FURNACE_ENV")
        .unwrap_or_else(|_| "development".to_string())
        .to_lowercase()
        == "production";

    // Create environment filter for better log control
    let env_filter = tracing_subscriber::EnvFilter::builder()
        .with_default_directive(level.into())
        .from_env()
        .map_err(|e| CliError::InvalidArgument {
            arg: "log-level".to_string(),
            value: log_level.to_string(),
            reason: format!("failed to create log filter: {e}"),
        })?;

    if is_production {
        // Production: JSON structured logging
        tracing_subscriber::fmt()
            .json()
            .with_max_level(level)
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
        // Development: Human-readable logging with colors
        tracing_subscriber::fmt()
            .with_max_level(level)
            .with_target(false)
            .with_thread_ids(true)
            .with_file(true)
            .with_line_number(true)
            .with_env_filter(env_filter)
            .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
            .init();
    }

    // Log the logging configuration
    tracing::info!(
        log_level = %level,
        is_production = is_production,
        "🔧 Logging initialized"
    );

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;
    setup_logging(&args.log_level)?;

    // Generate a unique session ID for this server instance
    let session_id = Uuid::new_v4();

    info!(
        session_id = %session_id,
        version = env!("CARGO_PKG_VERSION"),
        "🔥 Starting furnace inference server"
    );

    info!(
        session_id = %session_id,
        model_path = %args.model_path.display(),
        server_host = %args.host,
        server_port = args.port,
        backend = %args.backend.as_deref().unwrap_or("cpu"),
        max_concurrent_requests = ?args.max_concurrent_requests,
        kernel_fusion = args.enable_kernel_fusion,
        autotuning = args.enable_autotuning,
        log_level = %args.log_level,
        "📋 Server configuration"
    );

    // Load model with optimization settings
    info!(
        session_id = %session_id,
        model_path = %args.model_path.display(),
        backend = %args.backend.as_deref().unwrap_or("cpu"),
        kernel_fusion = args.enable_kernel_fusion,
        autotuning = args.enable_autotuning,
        "📦 Loading model"
    );

    let model_config = model::ModelConfig {
        backend: args.backend.clone(),
        enable_kernel_fusion: args.enable_kernel_fusion,
        enable_autotuning: args.enable_autotuning,
    };

    let model = match model::load_model_with_config(&args.model_path, model_config) {
        Ok(model) => {
            let model_info = model.get_info();
            info!(
                session_id = %session_id,
                model_name = %model_info.name,
                input_shape = ?model_info.input_spec.shape,
                output_shape = ?model_info.output_spec.shape,
                backend = %model_info.backend,
                model_type = %model_info.model_type,
                kernel_fusion = args.enable_kernel_fusion,
                autotuning = args.enable_autotuning,
                "✅ Model loaded successfully"
            );

            if args.enable_kernel_fusion {
                info!(
                    session_id = %session_id,
                    "🚀 Kernel fusion optimizations enabled"
                );
            }
            if args.enable_autotuning {
                info!(
                    session_id = %session_id,
                    "🎯 Autotuning cache optimizations enabled"
                );
            }
            model
        }
        Err(e) => {
            error!(
                session_id = %session_id,
                model_path = %args.model_path.display(),
                error = %e,
                "❌ Failed to load model"
            );
            std::process::exit(1);
        }
    };

    // Start server with concurrency limits
    let server_config = api::ServerConfig {
        host: args.host.clone(),
        port: args.port,
        max_concurrent_requests: args.max_concurrent_requests,
    };

    info!(
        session_id = %session_id,
        host = %args.host,
        port = args.port,
        max_concurrent_requests = ?args.max_concurrent_requests,
        "🚀 Starting HTTP server"
    );

    if let Err(e) = api::start_server_with_config(server_config, model).await {
        error!(
            session_id = %session_id,
            host = %args.host,
            port = args.port,
            error = %e,
            "❌ Server failed to start"
        );
        std::process::exit(1);
    }

    Ok(())
}
