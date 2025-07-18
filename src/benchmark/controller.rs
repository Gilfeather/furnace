use std::collections::HashMap;
use std::time::Duration;

use tracing::{error, info, warn};

use super::containers::ContainerManager;
use super::load_generator::{LoadConfig, LoadGenerator, LoadPattern, RequestConfig};
use super::metrics::MetricsCollector;
use super::reports::{ExportFormat, ReportGenerator};
use super::{
    BenchmarkConfig, BenchmarkResults, ComparisonAnalysis, ModelConfig, PerformanceRank,
    RelativeImprovement, ResourceLimits, ServerConfig, ServerResults,
};
use super::{BenchmarkError, Result};

pub struct BenchmarkController {
    container_manager: ContainerManager,
    load_generator: LoadGenerator,
    metrics_collector: MetricsCollector,
    report_generator: ReportGenerator,
}

impl BenchmarkController {
    pub fn new(output_dir: String) -> Result<Self> {
        let container_manager = ContainerManager::new()?;
        let load_generator = LoadGenerator::new(100, Duration::from_secs(30))?;
        let metrics_collector = MetricsCollector::new()?;
        let report_generator = ReportGenerator::new(output_dir)?;

        Ok(Self {
            container_manager,
            load_generator,
            metrics_collector,
            report_generator,
        })
    }

    pub async fn run_benchmark_suite(
        &mut self,
        config: BenchmarkConfig,
    ) -> Result<BenchmarkResults> {
        info!(
            "Starting benchmark suite with {} servers",
            config.servers.len()
        );

        // Validate environment
        self.validate_environment(&config).await?;

        let mut server_results = HashMap::new();
        let mut running_containers = Vec::new();

        // Start all servers
        for server_config in &config.servers {
            info!("Starting server: {:?}", server_config.server_type);

            let container_id = self
                .container_manager
                .start_server(
                    server_config.server_type.clone(),
                    Some(server_config.image.clone()),
                    Some(server_config.port),
                    server_config.resource_limits.clone(),
                    server_config.environment.clone(),
                )
                .await?;

            running_containers.push((container_id.clone(), server_config.clone()));
        }

        // Run benchmarks for each server
        for (container_id, server_config) in &running_containers {
            info!("Benchmarking server: {:?}", server_config.server_type);

            let server_result = self
                .run_single_benchmark(container_id, server_config, &config)
                .await?;

            server_results.insert(container_id.clone(), server_result);
        }

        // Stop all containers
        for (container_id, _) in running_containers {
            if let Err(e) = self.container_manager.stop_server(&container_id).await {
                warn!("Failed to stop container {}: {}", container_id, e);
            }
        }

        // Perform comparison analysis
        let comparison_analysis = self.perform_comparison_analysis(&server_results)?;

        let results = BenchmarkResults {
            timestamp: chrono::Utc::now(),
            config,
            server_results,
            comparison_analysis,
        };

        info!("Benchmark suite completed successfully");
        Ok(results)
    }

    pub async fn run_single_benchmark(
        &mut self,
        container_id: &str,
        server_config: &ServerConfig,
        benchmark_config: &BenchmarkConfig,
    ) -> Result<ServerResults> {
        info!("Running benchmark for container: {}", container_id);

        // Reset metrics collector
        self.metrics_collector.reset()?;

        // Warmup phase
        info!("Starting warmup phase");
        self.run_warmup_phase(container_id, server_config, benchmark_config)
            .await?;

        // Start resource monitoring
        let resource_monitor_handle = self.start_resource_monitoring(container_id);

        // Run load tests for each pattern
        let mut all_load_results = Vec::new();

        for load_pattern in &benchmark_config.load_patterns {
            info!("Running load pattern: {:?}", load_pattern);

            let load_config = LoadConfig {
                pattern: load_pattern.clone(),
                duration: benchmark_config.duration,
                max_concurrent: 100,
                timeout: Duration::from_secs(30),
            };

            let request_config = RequestConfig {
                server_type: server_config.server_type.clone(),
                base_url: format!("http://127.0.0.1:{}", server_config.port),
                model_input: self.generate_model_input(benchmark_config)?,
                headers: HashMap::new(),
            };

            let load_result = self
                .load_generator
                .run_load_test(load_config, request_config)
                .await?;
            self.metrics_collector
                .process_load_test_results(&load_result)?;
            all_load_results.push(load_result);
        }

        // Stop resource monitoring
        resource_monitor_handle.abort();

        // Collect final metrics
        let latency_stats = self.metrics_collector.get_latency_stats();
        let resource_stats = self.metrics_collector.get_resource_stats();

        // Calculate throughput stats from all load results
        let combined_load_result = self.combine_load_results(all_load_results)?;
        let throughput_stats = self
            .metrics_collector
            .calculate_throughput_stats(&combined_load_result);

        // Calculate error stats
        let error_stats = super::ErrorStats {
            total_requests: combined_load_result.total_requests,
            successful_requests: combined_load_result.successful_requests,
            failed_requests: combined_load_result.failed_requests,
            error_rate: combined_load_result.failed_requests as f64
                / combined_load_result.total_requests as f64,
            timeout_count: 0,     // TODO: Track timeouts separately
            connection_errors: 0, // TODO: Track connection errors separately
        };

        Ok(ServerResults {
            server_id: container_id.to_string(),
            server_type: server_config.server_type.clone(),
            latency_stats,
            throughput_stats,
            resource_stats,
            error_stats,
        })
    }

    pub async fn validate_environment(&self, config: &BenchmarkConfig) -> Result<()> {
        info!("Validating benchmark environment");

        // Check Docker availability
        // This is implicitly checked when creating ContainerManager

        // Validate server configurations
        for server_config in &config.servers {
            self.validate_server_config(server_config)?;
        }

        // Validate model configurations
        for model_config in &config.models {
            self.validate_model_config(model_config)?;
        }

        // Check available resources
        self.check_system_resources(config).await?;

        info!("Environment validation completed successfully");
        Ok(())
    }

    pub async fn generate_and_export_report(
        &self,
        results: &BenchmarkResults,
        formats: Vec<ExportFormat>,
    ) -> Result<Vec<String>> {
        info!("Generating benchmark report");

        let report = self.report_generator.generate_report(results)?;
        let mut exported_files = Vec::new();

        for format in formats {
            match self.report_generator.export_report(&report, format) {
                Ok(file_path) => {
                    info!("Report exported to: {}", file_path);
                    exported_files.push(file_path);
                }
                Err(e) => {
                    error!("Failed to export report: {}", e);
                    return Err(e);
                }
            }
        }

        Ok(exported_files)
    }

    async fn run_warmup_phase(
        &mut self,
        container_id: &str,
        server_config: &ServerConfig,
        benchmark_config: &BenchmarkConfig,
    ) -> Result<()> {
        let warmup_config = LoadConfig {
            pattern: LoadPattern::Constant { rps: 10.0 },
            duration: benchmark_config.warmup_duration,
            max_concurrent: 10,
            timeout: Duration::from_secs(30),
        };

        let request_config = RequestConfig {
            server_type: server_config.server_type.clone(),
            base_url: format!("http://127.0.0.1:{}", server_config.port),
            model_input: self.generate_model_input(benchmark_config)?,
            headers: HashMap::new(),
        };

        let _warmup_result = self
            .load_generator
            .run_load_test(warmup_config, request_config)
            .await?;

        info!("Warmup phase completed for container: {}", container_id);
        Ok(())
    }

    fn start_resource_monitoring(&mut self, _container_id: &str) -> tokio::task::JoinHandle<()> {
        // For now, return a dummy task that does nothing
        // In a production system, you'd want to use Arc<Mutex<>> or channels for safe sharing
        tokio::spawn(async move {
            // Placeholder implementation - resource monitoring disabled for now
            // TODO: Implement safe resource monitoring using Arc<Mutex<>> or channels
            tokio::time::sleep(Duration::from_secs(1)).await;
        })
    }

    fn perform_comparison_analysis(
        &self,
        server_results: &HashMap<String, ServerResults>,
    ) -> Result<ComparisonAnalysis> {
        let mut performance_ranking = Vec::new();
        let mut relative_improvements = HashMap::new();
        let statistical_significance = HashMap::new();

        // Calculate performance scores and rankings
        for (server_id, result) in server_results {
            let score = self.calculate_performance_score(result);
            performance_ranking.push(PerformanceRank {
                server_id: server_id.clone(),
                rank: 0, // Will be set after sorting
                score,
                category: "Overall".to_string(),
            });
        }

        // Sort by score (higher is better)
        performance_ranking.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Set ranks
        for (i, rank) in performance_ranking.iter_mut().enumerate() {
            rank.rank = i + 1;
        }

        // Calculate relative improvements (compare all servers to the worst performer)
        if let Some(baseline) = performance_ranking.last() {
            for rank in &performance_ranking {
                if rank.server_id != baseline.server_id {
                    if let (Some(current), Some(baseline_result)) = (
                        server_results.get(&rank.server_id),
                        server_results.get(&baseline.server_id),
                    ) {
                        let improvement = RelativeImprovement {
                            baseline_server: baseline.server_id.clone(),
                            comparison_server: rank.server_id.clone(),
                            latency_improvement: self.calculate_latency_improvement(
                                &baseline_result.latency_stats,
                                &current.latency_stats,
                            ),
                            throughput_improvement: self.calculate_throughput_improvement(
                                &baseline_result.throughput_stats,
                                &current.throughput_stats,
                            ),
                            memory_improvement: self.calculate_memory_improvement(
                                &baseline_result.resource_stats,
                                &current.resource_stats,
                            ),
                        };
                        relative_improvements.insert(rank.server_id.clone(), improvement);
                    }
                }
            }
        }

        Ok(ComparisonAnalysis {
            performance_ranking,
            relative_improvements,
            statistical_significance,
        })
    }

    fn calculate_performance_score(&self, result: &ServerResults) -> f64 {
        // Weighted performance score calculation
        let latency_score = 1000.0 / result.latency_stats.mean.max(1.0); // Lower latency = higher score
        let throughput_score = result.throughput_stats.requests_per_second; // Higher throughput = higher score
        let reliability_score = (1.0 - result.error_stats.error_rate) * 100.0; // Lower error rate = higher score
        let efficiency_score = 100.0 / result.resource_stats.avg_cpu_usage.max(1.0); // Lower CPU usage = higher score

        // Weighted average
        (latency_score * 0.3)
            + (throughput_score * 0.3)
            + (reliability_score * 0.2)
            + (efficiency_score * 0.2)
    }

    fn calculate_latency_improvement(
        &self,
        baseline: &super::LatencyStats,
        current: &super::LatencyStats,
    ) -> f64 {
        if baseline.mean > 0.0 {
            ((baseline.mean - current.mean) / baseline.mean) * 100.0
        } else {
            0.0
        }
    }

    fn calculate_throughput_improvement(
        &self,
        baseline: &super::ThroughputStats,
        current: &super::ThroughputStats,
    ) -> f64 {
        if baseline.requests_per_second > 0.0 {
            ((current.requests_per_second - baseline.requests_per_second)
                / baseline.requests_per_second)
                * 100.0
        } else {
            0.0
        }
    }

    fn calculate_memory_improvement(
        &self,
        baseline: &crate::benchmark::metrics::ResourceStats,
        current: &crate::benchmark::metrics::ResourceStats,
    ) -> f64 {
        if baseline.avg_memory_usage > 0.0 {
            ((baseline.avg_memory_usage - current.avg_memory_usage) / baseline.avg_memory_usage)
                * 100.0
        } else {
            0.0
        }
    }

    fn validate_server_config(&self, config: &ServerConfig) -> Result<()> {
        // Validate port range
        if config.port < 1024 {
            return Err(BenchmarkError::Configuration {
                message: format!("Invalid port number: {}", config.port),
            });
        }

        // Validate resource limits
        self.validate_resource_limits(&config.resource_limits)?;

        Ok(())
    }

    fn validate_model_config(&self, _config: &ModelConfig) -> Result<()> {
        // TODO: Validate model file existence and format
        Ok(())
    }

    fn validate_resource_limits(&self, limits: &ResourceLimits) -> Result<()> {
        // Basic validation of resource limit format
        if limits.cpu_limit.is_empty() || limits.memory_limit.is_empty() {
            return Err(BenchmarkError::Configuration {
                message: "CPU and memory limits must be specified".to_string(),
            });
        }

        Ok(())
    }

    async fn check_system_resources(&self, _config: &BenchmarkConfig) -> Result<()> {
        // TODO: Check available system resources (CPU, memory, disk space)
        // For now, just return Ok
        Ok(())
    }

    fn generate_model_input(&self, config: &BenchmarkConfig) -> Result<Vec<f32>> {
        // Use the first model's input shape to generate test data
        if let Some(model) = config.models.first() {
            Ok(LoadGenerator::generate_sample_input(&model.input_shape))
        } else {
            // Default MNIST-like input
            Ok(LoadGenerator::generate_sample_input(&[784]))
        }
    }

    fn combine_load_results(
        &self,
        results: Vec<super::load_generator::LoadTestResults>,
    ) -> Result<super::load_generator::LoadTestResults> {
        if results.is_empty() {
            return Err(BenchmarkError::Metrics {
                message: "No load test results to combine".to_string(),
            });
        }

        let mut combined = results[0].clone();

        for result in results.iter().skip(1) {
            combined.total_requests += result.total_requests;
            combined.successful_requests += result.successful_requests;
            combined.failed_requests += result.failed_requests;
            combined.total_duration += result.total_duration;
            combined.requests.extend(result.requests.clone());
        }

        // Recalculate actual RPS
        combined.actual_rps =
            combined.total_requests as f64 / combined.total_duration.as_secs_f64();

        Ok(combined)
    }
}

impl Drop for BenchmarkController {
    fn drop(&mut self) {
        // Cleanup any remaining containers
        if let Err(e) = futures::executor::block_on(self.container_manager.cleanup_all()) {
            error!("Failed to cleanup containers during drop: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::containers::ServerType;
    use crate::benchmark::metrics::{LatencyStats, ResourceStats, ThroughputStats};
    use crate::benchmark::ErrorStats;

    #[test]
    fn test_performance_score_calculation() {
        let controller = match BenchmarkController::new("test_output".to_string()) {
            Ok(c) => c,
            Err(_) => {
                // Skip test if Docker is not available
                println!("Docker not available, skipping performance score calculation test");
                return;
            }
        };

        let server_result = ServerResults {
            server_id: "test".to_string(),
            server_type: ServerType::Furnace,
            latency_stats: LatencyStats {
                mean: 10.0,
                median: 9.0,
                p50: 9.0,
                p90: 15.0,
                p95: 18.0,
                p99: 25.0,
                p99_9: 30.0,
                min: 5.0,
                max: 35.0,
                std_dev: 5.0,
                total_requests: 1000,
            },
            throughput_stats: ThroughputStats {
                requests_per_second: 100.0,
                successful_rps: 99.0,
                failed_rps: 1.0,
                total_requests: 1000,
                successful_requests: 990,
                failed_requests: 10,
                error_rate: 0.01,
                duration_seconds: 10.0,
            },
            resource_stats: ResourceStats {
                avg_cpu_usage: 50.0,
                max_cpu_usage: 75.0,
                min_cpu_usage: 25.0,
                avg_memory_usage: 1024.0,
                max_memory_usage: 1536.0,
                min_memory_usage: 512.0,
                avg_memory_usage_percent: 50.0,
                max_memory_usage_percent: 75.0,
                network_bytes_sent: 1000000,
                network_bytes_received: 2000000,
                samples_count: 100,
            },
            error_stats: ErrorStats {
                total_requests: 1000,
                successful_requests: 990,
                failed_requests: 10,
                error_rate: 0.01,
                timeout_count: 5,
                connection_errors: 5,
            },
        };

        let score = controller.calculate_performance_score(&server_result);
        assert!(score > 0.0);
    }

    #[test]
    fn test_latency_improvement_calculation() {
        let controller = match BenchmarkController::new("test_output".to_string()) {
            Ok(c) => c,
            Err(_) => {
                // Skip test if Docker is not available
                println!("Docker not available, skipping latency improvement calculation test");
                return;
            }
        };

        let baseline = LatencyStats {
            mean: 20.0,
            median: 18.0,
            p50: 18.0,
            p90: 30.0,
            p95: 35.0,
            p99: 45.0,
            p99_9: 50.0,
            min: 10.0,
            max: 60.0,
            std_dev: 8.0,
            total_requests: 1000,
        };

        let current = LatencyStats {
            mean: 10.0,
            median: 9.0,
            p50: 9.0,
            p90: 15.0,
            p95: 18.0,
            p99: 25.0,
            p99_9: 30.0,
            min: 5.0,
            max: 35.0,
            std_dev: 5.0,
            total_requests: 1000,
        };

        let improvement = controller.calculate_latency_improvement(&baseline, &current);
        assert_eq!(improvement, 50.0); // 50% improvement
    }
}
