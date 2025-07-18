use bollard::container::Stats;
use bollard::container::{
    Config, CreateContainerOptions, RemoveContainerOptions, StartContainerOptions, StatsOptions,
};
use bollard::image::CreateImageOptions;
use bollard::models::{HostConfig, RestartPolicy};
use bollard::Docker;
use futures_util::stream::StreamExt;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{error, info, warn};

use super::{BenchmarkError, ResourceLimits, Result};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ServerType {
    Furnace,
    TensorFlowServing,
    TorchServe,
    OnnxRuntime,
}

impl ServerType {
    pub fn default_image(&self) -> &'static str {
        match self {
            ServerType::Furnace => "furnace:latest",
            ServerType::TensorFlowServing => "tensorflow/serving:latest",
            ServerType::TorchServe => "pytorch/torchserve:latest",
            ServerType::OnnxRuntime => "mcr.microsoft.com/onnxruntime/server:latest",
        }
    }

    pub fn default_port(&self) -> u16 {
        match self {
            ServerType::Furnace => 3000,
            ServerType::TensorFlowServing => 8501,
            ServerType::TorchServe => 8080,
            ServerType::OnnxRuntime => 8001,
        }
    }

    pub fn health_endpoint(&self) -> &'static str {
        match self {
            ServerType::Furnace => "/healthz",
            ServerType::TensorFlowServing => "/v1/models",
            ServerType::TorchServe => "/ping",
            ServerType::OnnxRuntime => "/v1/models",
        }
    }

    pub fn predict_endpoint(&self) -> &'static str {
        match self {
            ServerType::Furnace => "/predict",
            ServerType::TensorFlowServing => "/v1/models/model:predict",
            ServerType::TorchServe => "/predictions/model",
            ServerType::OnnxRuntime => "/v1/models/model:predict",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Container {
    pub id: String,
    pub name: String,
    pub server_type: ServerType,
    pub port: u16,
    pub status: ContainerStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContainerStatus {
    Created,
    Running,
    Stopped,
    Error(String),
}

#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub is_healthy: bool,
    pub response_time_ms: f64,
    pub status_code: Option<u16>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ResourceMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_bytes: u64,
    pub memory_limit_bytes: u64,
    pub network_rx_bytes: u64,
    pub network_tx_bytes: u64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub struct ContainerManager {
    docker: Docker,
    containers: HashMap<String, Container>,
    client: reqwest::Client,
}

impl ContainerManager {
    pub fn new() -> Result<Self> {
        let docker = Docker::connect_with_local_defaults().map_err(BenchmarkError::Container)?;

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(BenchmarkError::Http)?;

        Ok(Self {
            docker,
            containers: HashMap::new(),
            client,
        })
    }

    pub async fn start_server(
        &mut self,
        server_type: ServerType,
        image: Option<String>,
        port: Option<u16>,
        resource_limits: ResourceLimits,
        environment: HashMap<String, String>,
    ) -> Result<String> {
        let image = image.unwrap_or_else(|| server_type.default_image().to_string());
        let port = port.unwrap_or_else(|| server_type.default_port());
        let container_name = format!("benchmark-{server_type:?}-{}", uuid::Uuid::new_v4());

        info!(
            "Starting container: {} with image: {}",
            container_name, image
        );

        // Pull image if not exists
        self.pull_image(&image).await?;

        // Create container configuration
        let host_config = HostConfig {
            port_bindings: Some({
                let mut bindings = HashMap::new();
                bindings.insert(
                    format!("{port}/tcp"),
                    Some(vec![bollard::models::PortBinding {
                        host_ip: Some("127.0.0.1".to_string()),
                        host_port: Some(port.to_string()),
                    }]),
                );
                bindings
            }),
            memory: Some(self.parse_memory_limit(&resource_limits.memory_limit)?),
            nano_cpus: Some(self.parse_cpu_limit(&resource_limits.cpu_limit)?),
            restart_policy: Some(RestartPolicy {
                name: Some(bollard::models::RestartPolicyNameEnum::NO),
                maximum_retry_count: Some(0),
            }),
            ..Default::default()
        };

        let config = Config {
            image: Some(image.clone()),
            env: Some(
                environment
                    .into_iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect(),
            ),
            host_config: Some(host_config),
            exposed_ports: Some({
                let mut ports = HashMap::new();
                ports.insert(format!("{port}/tcp"), HashMap::new());
                ports
            }),
            ..Default::default()
        };

        // Create container
        let container_id = self
            .docker
            .create_container(
                Some(CreateContainerOptions {
                    name: container_name.clone(),
                    platform: None,
                }),
                config,
            )
            .await
            .map_err(BenchmarkError::Container)?
            .id;

        // Start container
        self.docker
            .start_container(&container_id, None::<StartContainerOptions<String>>)
            .await
            .map_err(BenchmarkError::Container)?;

        let container = Container {
            id: container_id.clone(),
            name: container_name,
            server_type: server_type.clone(),
            port,
            status: ContainerStatus::Running,
        };

        self.containers.insert(container_id.clone(), container);

        info!("Container started successfully: {}", container_id);

        // Wait for container to be ready
        self.wait_for_health(&container_id, Duration::from_secs(60))
            .await?;

        Ok(container_id)
    }

    pub async fn stop_server(&mut self, container_id: &str) -> Result<()> {
        info!("Stopping container: {}", container_id);

        if let Some(container) = self.containers.get_mut(container_id) {
            container.status = ContainerStatus::Stopped;
        }

        // Stop container
        self.docker
            .stop_container(container_id, None)
            .await
            .map_err(BenchmarkError::Container)?;

        // Remove container
        self.docker
            .remove_container(
                container_id,
                Some(RemoveContainerOptions {
                    force: true,
                    ..Default::default()
                }),
            )
            .await
            .map_err(BenchmarkError::Container)?;

        self.containers.remove(container_id);

        info!("Container stopped and removed: {}", container_id);
        Ok(())
    }

    pub async fn health_check(&self, container_id: &str) -> Result<HealthStatus> {
        let container =
            self.containers
                .get(container_id)
                .ok_or_else(|| BenchmarkError::Configuration {
                    message: format!("Container not found: {container_id}"),
                })?;

        let url = format!(
            "http://127.0.0.1:{}{}",
            container.port,
            container.server_type.health_endpoint()
        );

        let start_time = std::time::Instant::now();

        match self.client.get(&url).send().await {
            Ok(response) => {
                let response_time = start_time.elapsed().as_millis() as f64;
                let status_code = response.status().as_u16();
                let is_healthy = response.status().is_success();

                Ok(HealthStatus {
                    is_healthy,
                    response_time_ms: response_time,
                    status_code: Some(status_code),
                    error_message: None,
                })
            }
            Err(e) => Ok(HealthStatus {
                is_healthy: false,
                response_time_ms: start_time.elapsed().as_millis() as f64,
                status_code: None,
                error_message: Some(e.to_string()),
            }),
        }
    }

    pub async fn get_resource_usage(&self, container_id: &str) -> Result<ResourceMetrics> {
        let mut stats_stream = self.docker.stats(
            container_id,
            Some(StatsOptions {
                stream: false,
                one_shot: true,
            }),
        );

        if let Some(stats_result) = stats_stream.next().await {
            let stats = stats_result.map_err(BenchmarkError::Container)?;
            Ok(self.parse_stats(stats))
        } else {
            Err(BenchmarkError::Metrics {
                message: format!("No stats available for container: {container_id}"),
            })
        }
    }

    pub async fn cleanup_all(&mut self) -> Result<()> {
        let container_ids: Vec<String> = self.containers.keys().cloned().collect();

        for container_id in container_ids {
            if let Err(e) = self.stop_server(&container_id).await {
                warn!("Failed to stop container {}: {}", container_id, e);
            }
        }

        Ok(())
    }

    async fn pull_image(&self, image: &str) -> Result<()> {
        info!("Pulling image: {}", image);

        let mut stream = self.docker.create_image(
            Some(CreateImageOptions {
                from_image: image,
                ..Default::default()
            }),
            None,
            None,
        );

        while let Some(result) = stream.next().await {
            match result {
                Ok(_) => {} // Progress update
                Err(e) => {
                    error!("Failed to pull image {}: {}", image, e);
                    return Err(BenchmarkError::Container(e));
                }
            }
        }

        info!("Image pulled successfully: {}", image);
        Ok(())
    }

    async fn wait_for_health(&self, container_id: &str, timeout: Duration) -> Result<()> {
        let start_time = std::time::Instant::now();
        let mut attempts = 0;
        const MAX_ATTEMPTS: u32 = 30;

        while start_time.elapsed() < timeout && attempts < MAX_ATTEMPTS {
            attempts += 1;

            match self.health_check(container_id).await {
                Ok(health) if health.is_healthy => {
                    info!("Container is healthy: {}", container_id);
                    return Ok(());
                }
                Ok(health) => {
                    info!(
                        "Container not ready yet (attempt {}): {} - {}",
                        attempts,
                        container_id,
                        health
                            .error_message
                            .unwrap_or_else(|| "Unknown error".to_string())
                    );
                }
                Err(e) => {
                    warn!("Health check failed (attempt {}): {}", attempts, e);
                }
            }

            sleep(Duration::from_secs(2)).await;
        }

        Err(BenchmarkError::Environment {
            message: format!(
                "Container failed to become healthy within {timeout:?}: {container_id}"
            ),
        })
    }

    fn parse_memory_limit(&self, limit: &str) -> Result<i64> {
        let limit = limit.to_lowercase();
        let (number, unit) = if limit.ends_with("gi") {
            (limit.trim_end_matches("gi"), 1024 * 1024 * 1024)
        } else if limit.ends_with("mi") {
            (limit.trim_end_matches("mi"), 1024 * 1024)
        } else if limit.ends_with("ki") {
            (limit.trim_end_matches("ki"), 1024)
        } else if limit.ends_with("g") {
            (limit.trim_end_matches("g"), 1000 * 1000 * 1000)
        } else if limit.ends_with("m") {
            (limit.trim_end_matches("m"), 1000 * 1000)
        } else if limit.ends_with("k") {
            (limit.trim_end_matches("k"), 1000)
        } else {
            (limit.as_str(), 1)
        };

        let number: f64 = number.parse().map_err(|_| BenchmarkError::Configuration {
            message: format!("Invalid memory limit format: {limit}"),
        })?;

        Ok((number * unit as f64) as i64)
    }

    fn parse_cpu_limit(&self, limit: &str) -> Result<i64> {
        let limit: f64 = limit.parse().map_err(|_| BenchmarkError::Configuration {
            message: format!("Invalid CPU limit format: {limit}"),
        })?;

        // Convert CPU cores to nanocpus (1 core = 1,000,000,000 nanocpus)
        Ok((limit * 1_000_000_000.0) as i64)
    }

    fn parse_stats(&self, stats: Stats) -> ResourceMetrics {
        let cpu_usage = {
            let (cpu_stats, precpu_stats) = (&stats.cpu_stats, &stats.precpu_stats);
            if let (cpu_usage, Some(system_usage), precpu_usage, Some(presystem_usage)) = (
                cpu_stats.cpu_usage.total_usage,
                cpu_stats.system_cpu_usage,
                precpu_stats.cpu_usage.total_usage,
                precpu_stats.system_cpu_usage,
            ) {
                let cpu_delta = cpu_usage.saturating_sub(precpu_usage) as f64;
                let system_delta = system_usage.saturating_sub(presystem_usage) as f64;
                let number_cpus = cpu_stats.online_cpus.unwrap_or(1) as f64;

                if system_delta > 0.0 {
                    (cpu_delta / system_delta) * number_cpus * 100.0
                } else {
                    0.0
                }
            } else {
                0.0
            }
        };

        let (memory_usage, memory_limit) = {
            let memory_stats = &stats.memory_stats;
            let usage = memory_stats.usage.unwrap_or(0);
            let limit = memory_stats.limit.unwrap_or(0);
            (usage, limit)
        };

        let (network_rx, network_tx) = if let Some(networks) = &stats.networks {
            let mut rx_bytes = 0;
            let mut tx_bytes = 0;
            for network in networks.values() {
                rx_bytes += network.rx_bytes;
                tx_bytes += network.tx_bytes;
            }
            (rx_bytes, tx_bytes)
        } else {
            (0, 0)
        };

        ResourceMetrics {
            cpu_usage_percent: cpu_usage,
            memory_usage_bytes: memory_usage,
            memory_limit_bytes: memory_limit,
            network_rx_bytes: network_rx,
            network_tx_bytes: network_tx,
            timestamp: chrono::Utc::now(),
        }
    }
}

impl Drop for ContainerManager {
    fn drop(&mut self) {
        // Note: In a real implementation, you might want to use a runtime
        // to properly clean up containers on drop
        warn!("ContainerManager dropped - containers may still be running");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_type_defaults() {
        assert_eq!(ServerType::Furnace.default_port(), 3000);
        assert_eq!(ServerType::TensorFlowServing.default_port(), 8501);
        assert_eq!(ServerType::Furnace.health_endpoint(), "/healthz");
        assert_eq!(ServerType::Furnace.predict_endpoint(), "/predict");
    }

    #[test]
    fn test_memory_limit_parsing() {
        let manager = match ContainerManager::new() {
            Ok(m) => m,
            Err(_) => {
                // Skip test if Docker is not available
                println!("Docker not available, skipping memory limit parsing test");
                return;
            }
        };

        assert_eq!(
            manager.parse_memory_limit("1Gi").unwrap(),
            1024 * 1024 * 1024
        );
        assert_eq!(
            manager.parse_memory_limit("512Mi").unwrap(),
            512 * 1024 * 1024
        );
        assert_eq!(
            manager.parse_memory_limit("2G").unwrap(),
            2 * 1000 * 1000 * 1000
        );
        assert_eq!(manager.parse_memory_limit("1024").unwrap(), 1024);
    }

    #[test]
    fn test_cpu_limit_parsing() {
        let manager = match ContainerManager::new() {
            Ok(m) => m,
            Err(_) => {
                // Skip test if Docker is not available
                println!("Docker not available, skipping CPU limit parsing test");
                return;
            }
        };

        assert_eq!(manager.parse_cpu_limit("1.0").unwrap(), 1_000_000_000);
        assert_eq!(manager.parse_cpu_limit("0.5").unwrap(), 500_000_000);
        assert_eq!(manager.parse_cpu_limit("2").unwrap(), 2_000_000_000);
    }
}
