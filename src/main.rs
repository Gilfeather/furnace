use clap::{Arg, Command};
use std::path::PathBuf;
use tracing::{error, info, Level};

use furnace::{api, error, model};
use error::{CliError, Result};

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
        .about("Rust-based Burn inference server")
        .version("0.1.0")
        .arg(
            Arg::new("model-path")
                .long("model-path")
                .value_name("PATH")
                .help("Path to the .burn model file")
                .required(true),
        )
        .arg(
            Arg::new("port")
                .long("port")
                .short('p')
                .value_name("PORT")
                .help("Port to bind the server to (1024-65535)")
                .default_value("3000"),
        )
        .arg(
            Arg::new("host")
                .long("host")
                .value_name("HOST")
                .help("Host to bind the server to")
                .default_value("127.0.0.1"),
        )
        .arg(
            Arg::new("backend")
                .long("backend")
                .value_name("BACKEND")
                .help("Backend to use for inference")
                .value_parser(["cpu", "wgpu", "metal", "cuda"])
                .default_value("cpu"),
        )
        .arg(
            Arg::new("max-concurrent-requests")
                .long("max-concurrent-requests")
                .value_name("NUM")
                .help("Maximum number of concurrent inference requests")
                .default_value("100"),
        )
        .arg(
            Arg::new("enable-kernel-fusion")
                .long("enable-kernel-fusion")
                .help("Enable kernel fusion optimizations")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("enable-autotuning")
                .long("enable-autotuning")
                .help("Enable autotuning cache optimizations")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("log-level")
                .long("log-level")
                .value_name("LEVEL")
                .help("Set the logging level")
                .value_parser(["error", "warn", "info", "debug", "trace"])
                .default_value("info"),
        )
        .get_matches();

    let model_path = matches
        .get_one::<String>("model-path")
        .ok_or_else(|| CliError::MissingArgument("model-path".to_string()))?;

    let port_str = matches.get_one::<String>("port").unwrap();
    let port = port_str
        .parse::<u16>()
        .map_err(|_| CliError::InvalidPort(port_str.clone()))?;

    if port < 1024 {
        return Err(CliError::InvalidPort(format!("Port {port} is below 1024")).into());
    }

    let host = matches.get_one::<String>("host").unwrap().clone();

    // Validate host format (basic check)
    if host.is_empty() {
        return Err(CliError::InvalidHost("Host cannot be empty".to_string()).into());
    }

    let backend = matches.get_one::<String>("backend").map(|s| s.clone());

    let max_concurrent_str = matches.get_one::<String>("max-concurrent-requests").unwrap();
    let max_concurrent_requests = Some(max_concurrent_str
        .parse::<usize>()
        .map_err(|_| CliError::InvalidArgument {
            arg: "max-concurrent-requests".to_string(),
            value: max_concurrent_str.clone(),
            reason: "must be a positive integer".to_string(),
        })?);

    if let Some(max_requests) = max_concurrent_requests {
        if max_requests == 0 {
            return Err(CliError::InvalidArgument {
                arg: "max-concurrent-requests".to_string(),
                value: max_concurrent_str.clone(),
                reason: "must be greater than 0".to_string(),
            }.into());
        }
    }

    let enable_kernel_fusion = matches.get_flag("enable-kernel-fusion");
    let enable_autotuning = matches.get_flag("enable-autotuning");
    let log_level = matches.get_one::<String>("log-level").unwrap().clone();

    let model_path = PathBuf::from(model_path);

    // Validate model path exists
    if !model_path.exists() {
        return Err(CliError::InvalidModelPath {
            path: model_path.clone(),
            reason: "file does not exist".to_string(),
        }.into());
    }

    if !model_path.is_file() {
        return Err(CliError::InvalidModelPath {
            path: model_path.clone(),
            reason: "path is not a file".to_string(),
        }.into());
    }

    // Validate .burn extension
    if let Some(extension) = model_path.extension() {
        if extension != "burn" {
            return Err(CliError::InvalidModelPath {
                path: model_path.clone(),
                reason: "file must have .burn extension".to_string(),
            }.into());
        }
    } else {
        return Err(CliError::InvalidModelPath {
            path: model_path.clone(),
            reason: "file must have .burn extension".to_string(),
        }.into());
    }

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

fn setup_logging(log_level: &str) {
    let level = match log_level.to_lowercase().as_str() {
        "error" => Level::ERROR,
        "warn" => Level::WARN,
        "info" => Level::INFO,
        "debug" => Level::DEBUG,
        "trace" => Level::TRACE,
        _ => Level::INFO,
    };

    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(false)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .with_env_filter(
            tracing_subscriber::EnvFilter::new(format!("{}={}", env!("CARGO_PKG_NAME"), level)),
        )
        .init();
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;
    setup_logging(&args.log_level);

    info!("🔥 Starting furnace inference server v{}", env!("CARGO_PKG_VERSION"));
    info!("Configuration:");
    info!("  Model path: {:?}", args.model_path);
    info!("  Server address: {}:{}", args.host, args.port);
    info!("  Backend: {:?}", args.backend.as_deref().unwrap_or("cpu"));
    info!("  Max concurrent requests: {:?}", args.max_concurrent_requests);
    info!("  Kernel fusion: {}", args.enable_kernel_fusion);
    info!("  Autotuning: {}", args.enable_autotuning);
    info!("  Log level: {}", args.log_level);

    // Load model with optimization settings
    info!("📦 Loading model from: {:?}", args.model_path);
    
    let model_config = model::ModelConfig {
        backend: args.backend.clone(),
        enable_kernel_fusion: args.enable_kernel_fusion,
        enable_autotuning: args.enable_autotuning,
    };

    let model = match model::load_model_with_config(&args.model_path, model_config) {
        Ok(model) => {
            info!("✅ Model loaded successfully: {}", model.get_info().name);
            info!("  Input shape: {:?}", model.get_info().input_spec.shape);
            info!("  Output shape: {:?}", model.get_info().output_spec.shape);
            info!("  Backend: {}", model.get_info().backend);
            if args.enable_kernel_fusion {
                info!("  🚀 Kernel fusion enabled");
            }
            if args.enable_autotuning {
                info!("  🎯 Autotuning cache enabled");
            }
            model
        }
        Err(e) => {
            error!("❌ Failed to load model: {}", e);
            std::process::exit(1);
        }
    };

    // Start server with concurrency limits
    let server_config = api::ServerConfig {
        host: args.host.clone(),
        port: args.port,
        max_concurrent_requests: args.max_concurrent_requests,
    };

    info!("🚀 Starting HTTP server on {}:{}", args.host, args.port);
    if let Err(e) = api::start_server_with_config(server_config, model).await {
        error!("❌ Server failed to start: {}", e);
        std::process::exit(1);
    }

    Ok(())
}
