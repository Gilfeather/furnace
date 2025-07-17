use clap::{Arg, Command};
use furnace::benchmark::reports::ExportFormat;
use furnace::benchmark::{
    BenchmarkConfig, BenchmarkController, LoadPattern, ModelConfig, ModelSize, ModelType,
    ResourceLimits, ServerConfig, ServerType,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
use tracing::{error, info, Level};

#[derive(Debug)]
struct BenchmarkArgs {
    config_file: Option<PathBuf>,
    output_dir: String,
    duration: Duration,
    warmup_duration: Duration,
    servers: Vec<ServerType>,
    export_formats: Vec<ExportFormat>,
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    setup_logging(&args.log_level)?;

    info!("🔥 Starting Furnace Performance Benchmark");
    info!("Output directory: {}", args.output_dir);
    info!("Test duration: {:?}", args.duration);
    info!("Servers to test: {:?}", args.servers);

    // Create benchmark controller
    let mut controller = BenchmarkController::new(args.output_dir.clone())?;

    // Load or create benchmark configuration
    let config = if let Some(config_file) = args.config_file {
        load_config_from_file(&config_file).await?
    } else {
        create_default_config(args.duration, args.warmup_duration, args.servers)?
    };

    info!("Benchmark configuration loaded");
    info!("Servers: {}", config.servers.len());
    info!("Models: {}", config.models.len());
    info!("Load patterns: {}", config.load_patterns.len());

    // Run benchmark suite
    info!("🚀 Starting benchmark execution");
    let results = match controller.run_benchmark_suite(config).await {
        Ok(results) => {
            info!("✅ Benchmark suite completed successfully");
            results
        }
        Err(e) => {
            error!("❌ Benchmark suite failed: {}", e);
            return Err(e.into());
        }
    };

    // Generate and export reports
    info!("📊 Generating reports");
    let exported_files = controller
        .generate_and_export_report(&results, args.export_formats)
        .await?;

    info!("🎉 Benchmark completed successfully!");
    info!("Reports generated:");
    for file in exported_files {
        info!("  - {}", file);
    }

    // Print summary
    print_benchmark_summary(&results);

    Ok(())
}

fn parse_args() -> Result<BenchmarkArgs, Box<dyn std::error::Error>> {
    let matches = Command::new("furnace-benchmark")
        .about("🔥 Furnace Performance Benchmarking Tool")
        .version(env!("CARGO_PKG_VERSION"))
        .long_about(
            "Comprehensive performance benchmarking tool for Furnace inference server. \
            Compares Furnace against popular alternatives like TensorFlow Serving, \
            TorchServe, and ONNX Runtime with detailed metrics and professional reports.",
        )
        .arg(
            Arg::new("config")
                .long("config")
                .short('c')
                .value_name("FILE")
                .help("Path to benchmark configuration file (TOML format)")
                .long_help(
                    "Path to a TOML configuration file containing detailed benchmark settings. \
                    If not provided, a default configuration will be used.",
                ),
        )
        .arg(
            Arg::new("output-dir")
                .long("output-dir")
                .short('o')
                .value_name("DIR")
                .help("Output directory for reports and charts")
                .default_value("benchmark-results"),
        )
        .arg(
            Arg::new("duration")
                .long("duration")
                .short('d')
                .value_name("SECONDS")
                .help("Duration of each benchmark test in seconds")
                .default_value("300")
                .value_parser(clap::value_parser!(u64)),
        )
        .arg(
            Arg::new("warmup")
                .long("warmup")
                .value_name("SECONDS")
                .help("Warmup duration before each test in seconds")
                .default_value("30")
                .value_parser(clap::value_parser!(u64)),
        )
        .arg(
            Arg::new("servers")
                .long("servers")
                .short('s')
                .value_name("SERVERS")
                .help("Comma-separated list of servers to benchmark")
                .long_help(
                    "Servers to include in the benchmark:\n  \
                    furnace - Furnace inference server\n  \
                    tensorflow - TensorFlow Serving\n  \
                    torchserve - TorchServe\n  \
                    onnxruntime - ONNX Runtime Server\n\n\
                    Example: --servers furnace,tensorflow,torchserve",
                )
                .default_value("furnace,tensorflow,torchserve,onnxruntime"),
        )
        .arg(
            Arg::new("export")
                .long("export")
                .short('e')
                .value_name("FORMATS")
                .help("Export formats for reports (json,html,csv)")
                .default_value("json,html,csv"),
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
        .get_matches();

    let config_file = matches.get_one::<String>("config").map(PathBuf::from);
    let output_dir = matches.get_one::<String>("output-dir").unwrap().clone();
    let duration = Duration::from_secs(*matches.get_one::<u64>("duration").unwrap());
    let warmup_duration = Duration::from_secs(*matches.get_one::<u64>("warmup").unwrap());
    let log_level = matches.get_one::<String>("log-level").unwrap().clone();

    // Parse servers
    let servers_str = matches.get_one::<String>("servers").unwrap();
    let servers = parse_servers(servers_str)?;

    // Parse export formats
    let export_str = matches.get_one::<String>("export").unwrap();
    let export_formats = parse_export_formats(export_str)?;

    Ok(BenchmarkArgs {
        config_file,
        output_dir,
        duration,
        warmup_duration,
        servers,
        export_formats,
        log_level,
    })
}

fn parse_servers(servers_str: &str) -> Result<Vec<ServerType>, Box<dyn std::error::Error>> {
    let mut servers = Vec::new();

    for server in servers_str.split(',') {
        let server = server.trim().to_lowercase();
        match server.as_str() {
            "furnace" => servers.push(ServerType::Furnace),
            "tensorflow" => servers.push(ServerType::TensorFlowServing),
            "torchserve" => servers.push(ServerType::TorchServe),
            "onnxruntime" => servers.push(ServerType::OnnxRuntime),
            _ => return Err(format!("Unknown server type: {}", server).into()),
        }
    }

    if servers.is_empty() {
        return Err("At least one server must be specified".into());
    }

    Ok(servers)
}

fn parse_export_formats(
    formats_str: &str,
) -> Result<Vec<ExportFormat>, Box<dyn std::error::Error>> {
    let mut formats = Vec::new();

    for format in formats_str.split(',') {
        let format = format.trim().to_lowercase();
        match format.as_str() {
            "json" => formats.push(ExportFormat::Json),
            "html" => formats.push(ExportFormat::Html),
            "csv" => formats.push(ExportFormat::Csv),
            "pdf" => formats.push(ExportFormat::Pdf),
            _ => return Err(format!("Unknown export format: {}", format).into()),
        }
    }

    if formats.is_empty() {
        formats.push(ExportFormat::Json); // Default format
    }

    Ok(formats)
}

async fn load_config_from_file(
    config_file: &PathBuf,
) -> Result<BenchmarkConfig, Box<dyn std::error::Error>> {
    let config_content = tokio::fs::read_to_string(config_file).await?;
    let config: BenchmarkConfig = toml::from_str(&config_content)?;
    Ok(config)
}

fn create_default_config(
    duration: Duration,
    warmup_duration: Duration,
    servers: Vec<ServerType>,
) -> Result<BenchmarkConfig, Box<dyn std::error::Error>> {
    let mut server_configs = Vec::new();

    for server_type in servers {
        let server_config = ServerConfig {
            server_type: server_type.clone(),
            image: server_type.default_image().to_string(),
            port: server_type.default_port(),
            environment: create_default_environment(&server_type),
            resource_limits: ResourceLimits {
                cpu_limit: "2.0".to_string(),
                memory_limit: "4Gi".to_string(),
                cpu_request: Some("1.0".to_string()),
                memory_request: Some("2Gi".to_string()),
            },
        };
        server_configs.push(server_config);
    }

    let model_config = ModelConfig {
        name: "mnist_test".to_string(),
        size_category: ModelSize::Small,
        model_type: ModelType::ImageClassification,
        input_shape: vec![784],
        formats: HashMap::new(), // Will be populated based on available models
    };

    let load_patterns = vec![
        LoadPattern::Constant { rps: 10.0 },
        LoadPattern::Constant { rps: 50.0 },
        LoadPattern::Constant { rps: 100.0 },
        LoadPattern::RampUp {
            start_rps: 10.0,
            end_rps: 100.0,
            duration: Duration::from_secs(60),
        },
    ];

    Ok(BenchmarkConfig {
        servers: server_configs,
        models: vec![model_config],
        load_patterns,
        duration,
        warmup_duration,
        resource_limits: ResourceLimits {
            cpu_limit: "8.0".to_string(),
            memory_limit: "16Gi".to_string(),
            cpu_request: None,
            memory_request: None,
        },
    })
}

fn create_default_environment(server_type: &ServerType) -> HashMap<String, String> {
    let mut env = HashMap::new();

    match server_type {
        ServerType::Furnace => {
            env.insert("RUST_LOG".to_string(), "info".to_string());
        }
        ServerType::TensorFlowServing => {
            env.insert("MODEL_NAME".to_string(), "model".to_string());
            env.insert("TF_CPP_MIN_LOG_LEVEL".to_string(), "2".to_string());
        }
        ServerType::TorchServe => {
            env.insert(
                "TS_CONFIG_FILE".to_string(),
                "/opt/ml/config.properties".to_string(),
            );
        }
        ServerType::OnnxRuntime => {
            env.insert("ORT_LOGGING_LEVEL".to_string(), "2".to_string());
        }
    }

    env
}

fn setup_logging(log_level: &str) -> Result<(), Box<dyn std::error::Error>> {
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
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::rfc_3339())
        .init();

    Ok(())
}

fn print_benchmark_summary(results: &furnace::benchmark::BenchmarkResults) {
    println!("\n🎯 BENCHMARK SUMMARY");
    println!("==================");
    println!(
        "Timestamp: {}",
        results.timestamp.format("%Y-%m-%d %H:%M:%S UTC")
    );
    println!("Servers tested: {}", results.server_results.len());

    // Find the winner
    if let Some(winner) = results.comparison_analysis.performance_ranking.first() {
        println!("🏆 Winner: {}", winner.server_id);
        println!("   Score: {:.2}", winner.score);
    }

    println!("\n📊 DETAILED RESULTS");
    println!("===================");

    for (server_id, result) in &results.server_results {
        println!("\n🔥 {}", server_id);
        println!("   Latency (mean): {:.2}ms", result.latency_stats.mean);
        println!("   Latency (p95):  {:.2}ms", result.latency_stats.p95);
        println!("   Latency (p99):  {:.2}ms", result.latency_stats.p99);
        println!(
            "   Throughput:     {:.2} RPS",
            result.throughput_stats.requests_per_second
        );
        println!(
            "   Success rate:   {:.1}%",
            (1.0 - result.error_stats.error_rate) * 100.0
        );
        println!(
            "   CPU usage:      {:.1}%",
            result.resource_stats.avg_cpu_usage
        );
        println!(
            "   Memory usage:   {:.1}%",
            result.resource_stats.avg_memory_usage_percent
        );
    }

    println!("\n🚀 PERFORMANCE IMPROVEMENTS");
    println!("============================");

    for (server_id, improvement) in &results.comparison_analysis.relative_improvements {
        println!(
            "\n📈 {} vs {}",
            improvement.comparison_server, improvement.baseline_server
        );
        println!(
            "   Latency improvement:    {:+.1}%",
            improvement.latency_improvement
        );
        println!(
            "   Throughput improvement: {:+.1}%",
            improvement.throughput_improvement
        );
        println!(
            "   Memory improvement:     {:+.1}%",
            improvement.memory_improvement
        );
    }

    println!("\n✨ Benchmark completed! Check the generated reports for detailed analysis.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_servers() {
        let servers = parse_servers("furnace,tensorflow").unwrap();
        assert_eq!(servers.len(), 2);
        assert!(matches!(servers[0], ServerType::Furnace));
        assert!(matches!(servers[1], ServerType::TensorFlowServing));
    }

    #[test]
    fn test_parse_export_formats() {
        let formats = parse_export_formats("json,html,csv").unwrap();
        assert_eq!(formats.len(), 3);
    }

    #[test]
    fn test_create_default_config() {
        let servers = vec![ServerType::Furnace, ServerType::TensorFlowServing];
        let config =
            create_default_config(Duration::from_secs(300), Duration::from_secs(30), servers)
                .unwrap();

        assert_eq!(config.servers.len(), 2);
        assert_eq!(config.models.len(), 1);
        assert!(!config.load_patterns.is_empty());
    }
}
