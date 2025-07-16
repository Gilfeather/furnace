use clap::{Arg, Command};
use std::path::PathBuf;
use tracing::{info, error, Level};
use tracing_subscriber;

mod api;
mod model;
mod error;
mod burn_model;

use error::{CliError, Result};

#[derive(Debug)]
struct CliArgs {
    model_path: PathBuf,
    host: String,
    port: u16,
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
                .required(true)
        )
        .arg(
            Arg::new("port")
                .long("port")
                .short('p')
                .value_name("PORT")
                .help("Port to bind the server to (1024-65535)")
                .default_value("3000")
        )
        .arg(
            Arg::new("host")
                .long("host")
                .value_name("HOST")
                .help("Host to bind the server to")
                .default_value("127.0.0.1")
        )
        .get_matches();

    let model_path = matches.get_one::<String>("model-path")
        .ok_or_else(|| CliError::MissingArgument("model-path".to_string()))?;
    
    let port_str = matches.get_one::<String>("port").unwrap();
    let port = port_str.parse::<u16>()
        .map_err(|_| CliError::InvalidPort(port_str.clone()))?;
    
    if port < 1024 {
        return Err(CliError::InvalidPort(format!("Port {} is below 1024", port)).into());
    }

    let host = matches.get_one::<String>("host").unwrap().clone();
    
    // Validate host format (basic check)
    if host.is_empty() {
        return Err(CliError::InvalidHost("Host cannot be empty".to_string()).into());
    }

    let model_path = PathBuf::from(model_path);
    
    Ok(CliArgs {
        model_path,
        host,
        port,
    })
}

fn setup_logging() {
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_target(false)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();
}

#[tokio::main]
async fn main() -> Result<()> {
    setup_logging();
    
    let args = parse_args()?;
    
    info!("Starting furnace inference server");
    info!("Model path: {:?}", args.model_path);
    info!("Server will bind to: {}:{}", args.host, args.port);

    // Load model
    info!("Loading model from: {:?}", args.model_path);
    let model = match model::load_model(&args.model_path) {
        Ok(model) => {
            info!("Model loaded successfully: {}", model.get_info().name);
            info!("Input shape: {:?}", model.get_info().input_spec.shape);
            info!("Output shape: {:?}", model.get_info().output_spec.shape);
            model
        }
        Err(e) => {
            error!("Failed to load model: {}", e);
            std::process::exit(1);
        }
    };

    // Start server
    info!("Starting HTTP server on {}:{}", args.host, args.port);
    if let Err(e) = api::start_server(&args.host, args.port, model).await {
        error!("Server failed to start: {}", e);
        std::process::exit(1);
    }

    Ok(())
}