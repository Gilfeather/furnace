use hdrhistogram::Histogram;
use serde::{Deserialize, Serialize};
use statistical::{mean, median, standard_deviation};
use std::collections::VecDeque;
use std::time::Duration;

use super::load_generator::{LoadTestResults, RequestResult};
use super::{BenchmarkError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub mean: f64,
    pub median: f64,
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub p99_9: f64,
    pub min: f64,
    pub max: f64,
    pub std_dev: f64,
    pub total_requests: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputStats {
    pub requests_per_second: f64,
    pub successful_rps: f64,
    pub failed_rps: f64,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub error_rate: f64,
    pub duration_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStats {
    pub avg_cpu_usage: f64,
    pub max_cpu_usage: f64,
    pub min_cpu_usage: f64,
    pub avg_memory_usage: f64,
    pub max_memory_usage: f64,
    pub min_memory_usage: f64,
    pub avg_memory_usage_percent: f64,
    pub max_memory_usage_percent: f64,
    pub network_bytes_sent: u64,
    pub network_bytes_received: u64,
    pub samples_count: usize,
}

#[derive(Debug)]
pub struct MetricsCollector {
    latency_histogram: Histogram<u64>,
    resource_samples: VecDeque<ResourceSample>,
    max_samples: usize,
}

#[derive(Debug, Clone)]
struct ResourceSample {
    timestamp: chrono::DateTime<chrono::Utc>,
    cpu_usage: f64,
    memory_usage: f64,
    memory_limit: f64,
    network_rx: u64,
    network_tx: u64,
}

impl MetricsCollector {
    pub fn new() -> Result<Self> {
        let latency_histogram =
            Histogram::new_with_bounds(1, 60_000, 3).map_err(|e| BenchmarkError::Metrics {
                message: format!("Failed to create latency histogram: {e}"),
            })?;

        Ok(Self {
            latency_histogram,
            resource_samples: VecDeque::new(),
            max_samples: 10000, // Keep last 10k samples
        })
    }

    pub fn record_latency(&mut self, latency: Duration) -> Result<()> {
        let latency_ms = latency.as_millis() as u64;
        self.latency_histogram
            .record(latency_ms)
            .map_err(|e| BenchmarkError::Metrics {
                message: format!("Failed to record latency: {e}"),
            })?;
        Ok(())
    }

    pub fn record_resource_usage(
        &mut self,
        cpu_usage: f64,
        memory_usage: f64,
        memory_limit: f64,
        network_rx: u64,
        network_tx: u64,
    ) {
        let sample = ResourceSample {
            timestamp: chrono::Utc::now(),
            cpu_usage,
            memory_usage,
            memory_limit,
            network_rx,
            network_tx,
        };

        self.resource_samples.push_back(sample);

        // Keep only the most recent samples
        while self.resource_samples.len() > self.max_samples {
            self.resource_samples.pop_front();
        }
    }

    pub fn get_latency_stats(&self) -> LatencyStats {
        if self.latency_histogram.is_empty() {
            return LatencyStats {
                mean: 0.0,
                median: 0.0,
                p50: 0.0,
                p90: 0.0,
                p95: 0.0,
                p99: 0.0,
                p99_9: 0.0,
                min: 0.0,
                max: 0.0,
                std_dev: 0.0,
                total_requests: 0,
            };
        }

        LatencyStats {
            mean: self.latency_histogram.mean(),
            median: self.latency_histogram.value_at_quantile(0.5) as f64,
            p50: self.latency_histogram.value_at_quantile(0.5) as f64,
            p90: self.latency_histogram.value_at_quantile(0.9) as f64,
            p95: self.latency_histogram.value_at_quantile(0.95) as f64,
            p99: self.latency_histogram.value_at_quantile(0.99) as f64,
            p99_9: self.latency_histogram.value_at_quantile(0.999) as f64,
            min: self.latency_histogram.min() as f64,
            max: self.latency_histogram.max() as f64,
            std_dev: self.latency_histogram.stdev(),
            total_requests: self.latency_histogram.len(),
        }
    }

    pub fn get_resource_stats(&self) -> ResourceStats {
        if self.resource_samples.is_empty() {
            return ResourceStats {
                avg_cpu_usage: 0.0,
                max_cpu_usage: 0.0,
                min_cpu_usage: 0.0,
                avg_memory_usage: 0.0,
                max_memory_usage: 0.0,
                min_memory_usage: 0.0,
                avg_memory_usage_percent: 0.0,
                max_memory_usage_percent: 0.0,
                network_bytes_sent: 0,
                network_bytes_received: 0,
                samples_count: 0,
            };
        }

        let cpu_values: Vec<f64> = self.resource_samples.iter().map(|s| s.cpu_usage).collect();
        let memory_values: Vec<f64> = self
            .resource_samples
            .iter()
            .map(|s| s.memory_usage)
            .collect();
        let memory_percent_values: Vec<f64> = self
            .resource_samples
            .iter()
            .map(|s| {
                if s.memory_limit > 0.0 {
                    (s.memory_usage / s.memory_limit) * 100.0
                } else {
                    0.0
                }
            })
            .collect();

        let latest_sample = self.resource_samples.back().unwrap();

        ResourceStats {
            avg_cpu_usage: mean(&cpu_values),
            max_cpu_usage: cpu_values.iter().fold(0.0, |a, &b| a.max(b)),
            min_cpu_usage: cpu_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            avg_memory_usage: mean(&memory_values),
            max_memory_usage: memory_values.iter().fold(0.0, |a, &b| a.max(b)),
            min_memory_usage: memory_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            avg_memory_usage_percent: mean(&memory_percent_values),
            max_memory_usage_percent: memory_percent_values.iter().fold(0.0, |a, &b| a.max(b)),
            network_bytes_sent: latest_sample.network_tx,
            network_bytes_received: latest_sample.network_rx,
            samples_count: self.resource_samples.len(),
        }
    }

    pub fn calculate_throughput_stats(&self, load_results: &LoadTestResults) -> ThroughputStats {
        let duration_seconds = load_results.total_duration.as_secs_f64();
        let error_rate = if load_results.total_requests > 0 {
            load_results.failed_requests as f64 / load_results.total_requests as f64
        } else {
            0.0
        };

        ThroughputStats {
            requests_per_second: load_results.actual_rps,
            successful_rps: load_results.successful_requests as f64 / duration_seconds,
            failed_rps: load_results.failed_requests as f64 / duration_seconds,
            total_requests: load_results.total_requests,
            successful_requests: load_results.successful_requests,
            failed_requests: load_results.failed_requests,
            error_rate,
            duration_seconds,
        }
    }

    pub fn process_load_test_results(&mut self, results: &LoadTestResults) -> Result<()> {
        // Record all latencies
        for request in &results.requests {
            if request.success {
                self.record_latency(request.latency)?;
            }
        }
        Ok(())
    }

    pub fn calculate_detailed_latency_stats(&self, requests: &[RequestResult]) -> LatencyStats {
        if requests.is_empty() {
            return LatencyStats {
                mean: 0.0,
                median: 0.0,
                p50: 0.0,
                p90: 0.0,
                p95: 0.0,
                p99: 0.0,
                p99_9: 0.0,
                min: 0.0,
                max: 0.0,
                std_dev: 0.0,
                total_requests: 0,
            };
        }

        let successful_requests: Vec<&RequestResult> =
            requests.iter().filter(|r| r.success).collect();

        if successful_requests.is_empty() {
            return LatencyStats {
                mean: 0.0,
                median: 0.0,
                p50: 0.0,
                p90: 0.0,
                p95: 0.0,
                p99: 0.0,
                p99_9: 0.0,
                min: 0.0,
                max: 0.0,
                std_dev: 0.0,
                total_requests: requests.len() as u64,
            };
        }

        let mut latencies: Vec<f64> = successful_requests
            .iter()
            .map(|r| r.latency.as_millis() as f64)
            .collect();

        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean_latency = mean(&latencies);
        let median_latency = median(&latencies);
        let std_dev_latency = standard_deviation(&latencies, Some(mean_latency));

        LatencyStats {
            mean: mean_latency,
            median: median_latency,
            p50: self.percentile(&latencies, 0.5),
            p90: self.percentile(&latencies, 0.9),
            p95: self.percentile(&latencies, 0.95),
            p99: self.percentile(&latencies, 0.99),
            p99_9: self.percentile(&latencies, 0.999),
            min: latencies.first().copied().unwrap_or(0.0),
            max: latencies.last().copied().unwrap_or(0.0),
            std_dev: std_dev_latency,
            total_requests: requests.len() as u64,
        }
    }

    fn percentile(&self, sorted_values: &[f64], percentile: f64) -> f64 {
        if sorted_values.is_empty() {
            return 0.0;
        }

        let index = percentile * (sorted_values.len() - 1) as f64;
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;
        
        if lower_index == upper_index {
            sorted_values[lower_index]
        } else {
            let lower_value = sorted_values[lower_index];
            let upper_value = sorted_values[upper_index];
            let weight = index - lower_index as f64;
            lower_value + weight * (upper_value - lower_value)
        }
    }

    pub fn reset(&mut self) -> Result<()> {
        self.latency_histogram.reset();
        self.resource_samples.clear();
        Ok(())
    }

    pub fn export_raw_data(&self) -> Result<serde_json::Value> {
        let latency_data: Vec<u64> = (0..self.latency_histogram.len())
            .filter_map(|_| {
                // Note: This is a simplified export. In a real implementation,
                // you'd want to iterate through the histogram properly
                None
            })
            .collect();

        let resource_data: Vec<serde_json::Value> = self
            .resource_samples
            .iter()
            .map(|sample| {
                serde_json::json!({
                    "timestamp": sample.timestamp,
                    "cpu_usage": sample.cpu_usage,
                    "memory_usage": sample.memory_usage,
                    "memory_limit": sample.memory_limit,
                    "memory_usage_percent": if sample.memory_limit > 0.0 {
                        (sample.memory_usage / sample.memory_limit) * 100.0
                    } else {
                        0.0
                    },
                    "network_rx": sample.network_rx,
                    "network_tx": sample.network_tx,
                })
            })
            .collect();

        Ok(serde_json::json!({
            "latency_samples": latency_data,
            "resource_samples": resource_data,
            "latency_stats": self.get_latency_stats(),
            "resource_stats": self.get_resource_stats(),
        }))
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new().expect("Failed to create default MetricsCollector")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new();
        assert!(collector.is_ok());
    }

    #[test]
    fn test_latency_recording() {
        let mut collector = MetricsCollector::new().unwrap();

        collector
            .record_latency(Duration::from_millis(100))
            .unwrap();
        collector
            .record_latency(Duration::from_millis(200))
            .unwrap();
        collector
            .record_latency(Duration::from_millis(150))
            .unwrap();

        let stats = collector.get_latency_stats();
        assert_eq!(stats.total_requests, 3);
        assert!(stats.mean > 0.0);
        assert!(stats.min <= stats.max);
    }

    #[test]
    fn test_resource_usage_recording() {
        let mut collector = MetricsCollector::new().unwrap();

        collector.record_resource_usage(50.0, 1024.0, 2048.0, 100, 200);
        collector.record_resource_usage(75.0, 1536.0, 2048.0, 150, 250);

        let stats = collector.get_resource_stats();
        assert_eq!(stats.samples_count, 2);
        assert_eq!(stats.avg_cpu_usage, 62.5);
        assert_eq!(stats.max_cpu_usage, 75.0);
        assert_eq!(stats.min_cpu_usage, 50.0);
    }

    #[test]
    fn test_percentile_calculation() {
        let collector = MetricsCollector::new().unwrap();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        assert!((collector.percentile(&values, 0.5) - 5.5).abs() < 0.001);
        assert!((collector.percentile(&values, 0.9) - 9.1).abs() < 0.001);
        assert!((collector.percentile(&values, 0.95) - 9.55).abs() < 0.001);
    }

    #[test]
    fn test_empty_stats() {
        let collector = MetricsCollector::new().unwrap();

        let latency_stats = collector.get_latency_stats();
        assert_eq!(latency_stats.total_requests, 0);
        assert_eq!(latency_stats.mean, 0.0);

        let resource_stats = collector.get_resource_stats();
        assert_eq!(resource_stats.samples_count, 0);
        assert_eq!(resource_stats.avg_cpu_usage, 0.0);
    }

    #[test]
    fn test_throughput_calculation() {
        let collector = MetricsCollector::new().unwrap();

        let load_results = LoadTestResults {
            total_requests: 100,
            successful_requests: 95,
            failed_requests: 5,
            total_duration: Duration::from_secs(10),
            requests: vec![],
            actual_rps: 10.0,
        };

        let throughput_stats = collector.calculate_throughput_stats(&load_results);
        assert_eq!(throughput_stats.total_requests, 100);
        assert_eq!(throughput_stats.successful_requests, 95);
        assert_eq!(throughput_stats.failed_requests, 5);
        assert_eq!(throughput_stats.error_rate, 0.05);
        assert_eq!(throughput_stats.successful_rps, 9.5);
    }
}
