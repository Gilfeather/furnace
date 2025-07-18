use futures_util::future::join_all;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::time::{interval, sleep};
use tracing::{debug, info, warn};

use super::{BenchmarkError, Result, ServerType};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadPattern {
    Constant {
        rps: f64,
    },
    RampUp {
        start_rps: f64,
        end_rps: f64,
        duration: Duration,
    },
    Burst {
        base_rps: f64,
        burst_rps: f64,
        burst_duration: Duration,
    },
}

#[derive(Debug, Clone)]
pub struct LoadConfig {
    pub pattern: LoadPattern,
    pub duration: Duration,
    pub max_concurrent: usize,
    pub timeout: Duration,
}

#[derive(Debug, Clone)]
pub struct RequestConfig {
    pub server_type: ServerType,
    pub base_url: String,
    pub model_input: Vec<f32>,
    pub headers: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct RequestResult {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub latency: Duration,
    pub success: bool,
    pub status_code: Option<u16>,
    pub error_message: Option<String>,
    pub response_size: usize,
}

#[derive(Debug, Clone)]
pub struct LoadTestResults {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_duration: Duration,
    pub requests: Vec<RequestResult>,
    pub actual_rps: f64,
}

pub struct LoadGenerator {
    client: Client,
    semaphore: Arc<Semaphore>,
}

impl LoadGenerator {
    pub fn new(max_concurrent: usize, timeout: Duration) -> Result<Self> {
        let client = Client::builder()
            .timeout(timeout)
            .pool_max_idle_per_host(max_concurrent)
            .build()
            .map_err(|e| BenchmarkError::LoadGeneration {
                message: format!("Failed to create HTTP client: {e}"),
            })?;

        let semaphore = Arc::new(Semaphore::new(max_concurrent));

        Ok(Self { client, semaphore })
    }

    pub async fn run_load_test(
        &self,
        config: LoadConfig,
        request_config: RequestConfig,
    ) -> Result<LoadTestResults> {
        info!(
            "Starting load test with pattern: {:?}, duration: {:?}",
            config.pattern, config.duration
        );

        let start_time = Instant::now();
        let results = match config.pattern {
            LoadPattern::Constant { rps } => {
                self.run_constant_load(rps, config.duration, &request_config)
                    .await?
            }
            LoadPattern::RampUp {
                start_rps,
                end_rps,
                duration,
            } => {
                self.run_ramp_up_load(start_rps, end_rps, duration, &request_config)
                    .await?
            }
            LoadPattern::Burst {
                base_rps,
                burst_rps,
                burst_duration,
            } => {
                self.run_burst_load(
                    base_rps,
                    burst_rps,
                    burst_duration,
                    config.duration,
                    &request_config,
                )
                .await?
            }
        };

        let total_requests = results.len() as u64;
        let successful_requests = results.iter().filter(|r| r.success).count() as u64;
        let failed_requests = total_requests - successful_requests;
        let total_duration = start_time.elapsed();
        let actual_rps = total_requests as f64 / total_duration.as_secs_f64();

        info!(
            "Load test completed: {} total requests, {} successful, {} failed, {:.2} RPS",
            total_requests, successful_requests, failed_requests, actual_rps
        );

        Ok(LoadTestResults {
            total_requests,
            successful_requests,
            failed_requests,
            total_duration,
            requests: results,
            actual_rps,
        })
    }

    async fn run_constant_load(
        &self,
        rps: f64,
        duration: Duration,
        request_config: &RequestConfig,
    ) -> Result<Vec<RequestResult>> {
        let interval_duration = Duration::from_secs_f64(1.0 / rps);
        let mut interval = interval(interval_duration);
        let end_time = Instant::now() + duration;
        let mut tasks = Vec::new();

        while Instant::now() < end_time {
            interval.tick().await;

            let task = self.send_request(request_config.clone());
            tasks.push(task);
        }

        let results = join_all(tasks).await;
        Ok(results.into_iter().collect())
    }

    async fn run_ramp_up_load(
        &self,
        start_rps: f64,
        end_rps: f64,
        duration: Duration,
        request_config: &RequestConfig,
    ) -> Result<Vec<RequestResult>> {
        let start_time = Instant::now();
        let mut tasks = Vec::new();
        let total_seconds = duration.as_secs_f64();

        while start_time.elapsed() < duration {
            let elapsed_seconds = start_time.elapsed().as_secs_f64();
            let progress = elapsed_seconds / total_seconds;
            let current_rps = start_rps + (end_rps - start_rps) * progress;

            let interval_duration = Duration::from_secs_f64(1.0 / current_rps);
            sleep(interval_duration).await;

            let task = self.send_request(request_config.clone());
            tasks.push(task);
        }

        let results = join_all(tasks).await;
        Ok(results.into_iter().collect())
    }

    async fn run_burst_load(
        &self,
        base_rps: f64,
        burst_rps: f64,
        burst_duration: Duration,
        total_duration: Duration,
        request_config: &RequestConfig,
    ) -> Result<Vec<RequestResult>> {
        let start_time = Instant::now();
        let mut tasks = Vec::new();
        let burst_interval = Duration::from_secs(10); // Burst every 10 seconds
        let mut last_burst = Instant::now();

        while start_time.elapsed() < total_duration {
            let should_burst = last_burst.elapsed() >= burst_interval;
            let current_rps = if should_burst && last_burst.elapsed() < burst_duration {
                burst_rps
            } else {
                if should_burst && last_burst.elapsed() >= burst_duration {
                    last_burst = Instant::now();
                }
                base_rps
            };

            let interval_duration = Duration::from_secs_f64(1.0 / current_rps);
            sleep(interval_duration).await;

            let task = self.send_request(request_config.clone());
            tasks.push(task);
        }

        let results = join_all(tasks).await;
        Ok(results.into_iter().collect())
    }

    async fn send_request(&self, config: RequestConfig) -> RequestResult {
        let _permit = self.semaphore.acquire().await.unwrap();
        let start_time = Instant::now();
        let timestamp = chrono::Utc::now();

        let url = format!(
            "{}{}",
            config.base_url,
            config.server_type.predict_endpoint()
        );
        let payload = self.create_request_payload(&config);

        debug!("Sending request to: {}", url);

        let mut request_builder = self.client.post(&url);

        // Add headers
        for (key, value) in &config.headers {
            request_builder = request_builder.header(key, value);
        }

        // Add content type
        request_builder = request_builder.header("Content-Type", "application/json");

        match request_builder.json(&payload).send().await {
            Ok(response) => {
                let latency = start_time.elapsed();
                let status_code = response.status().as_u16();
                let success = response.status().is_success();

                let response_size = match response.text().await {
                    Ok(text) => text.len(),
                    Err(_) => 0,
                };

                RequestResult {
                    timestamp,
                    latency,
                    success,
                    status_code: Some(status_code),
                    error_message: if success {
                        None
                    } else {
                        Some(format!("HTTP {status_code}"))
                    },
                    response_size,
                }
            }
            Err(e) => {
                let latency = start_time.elapsed();
                warn!("Request failed: {}", e);

                RequestResult {
                    timestamp,
                    latency,
                    success: false,
                    status_code: None,
                    error_message: Some(e.to_string()),
                    response_size: 0,
                }
            }
        }
    }

    fn create_request_payload(&self, config: &RequestConfig) -> serde_json::Value {
        match config.server_type {
            ServerType::Furnace => {
                serde_json::json!({
                    "input": config.model_input
                })
            }
            ServerType::TensorFlowServing => {
                serde_json::json!({
                    "instances": [config.model_input]
                })
            }
            ServerType::TorchServe => {
                serde_json::json!({
                    "data": config.model_input
                })
            }
            ServerType::OnnxRuntime => {
                serde_json::json!({
                    "input_data": [config.model_input]
                })
            }
        }
    }

    pub fn generate_sample_input(input_shape: &[usize]) -> Vec<f32> {
        let total_size: usize = input_shape.iter().product();
        (0..total_size)
            .map(|i| (i as f32 * 0.01) % 1.0) // Generate deterministic test data
            .collect()
    }

    pub fn generate_random_input(input_shape: &[usize]) -> Vec<f32> {
        use rand::Rng;
        let total_size: usize = input_shape.iter().product();
        let mut rng = rand::thread_rng();
        (0..total_size).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_sample_input() {
        let input_shape = vec![2, 3];
        let input = LoadGenerator::generate_sample_input(&input_shape);
        assert_eq!(input.len(), 6);
        assert_eq!(input[0], 0.0);
        assert_eq!(input[1], 0.01);
    }

    #[test]
    fn test_generate_random_input() {
        let input_shape = vec![10];
        let input = LoadGenerator::generate_random_input(&input_shape);
        assert_eq!(input.len(), 10);
        // Check that values are in expected range
        for value in input {
            assert!((-1.0..=1.0).contains(&value));
        }
    }

    #[test]
    fn test_load_pattern_serialization() {
        let pattern = LoadPattern::Constant { rps: 10.0 };
        let json = serde_json::to_string(&pattern).unwrap();
        let deserialized: LoadPattern = serde_json::from_str(&json).unwrap();

        match deserialized {
            LoadPattern::Constant { rps } => assert_eq!(rps, 10.0),
            _ => panic!("Unexpected pattern type"),
        }
    }

    #[tokio::test]
    async fn test_load_generator_creation() {
        let generator = LoadGenerator::new(10, Duration::from_secs(30));
        assert!(generator.is_ok());
    }
}
