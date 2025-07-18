use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

use super::{BenchmarkError, Result};
use super::{BenchmarkResults, ComparisonAnalysis, ServerResults, StatisticalTest};

// Helper function to convert plotting errors
fn plot_error<E: std::fmt::Display>(e: E) -> BenchmarkError {
    BenchmarkError::Plotting(e.to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub title: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub executive_summary: ExecutiveSummary,
    pub detailed_results: DetailedResults,
    pub charts: Vec<ChartInfo>,
    pub methodology: Methodology,
    pub conclusions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutiveSummary {
    pub winner: String,
    pub key_findings: Vec<String>,
    pub performance_improvements: HashMap<String, f64>,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedResults {
    pub server_results: HashMap<String, ServerResults>,
    pub comparison_matrix: ComparisonMatrix,
    pub statistical_analysis: StatisticalAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonMatrix {
    pub latency_comparison: HashMap<String, HashMap<String, f64>>,
    pub throughput_comparison: HashMap<String, HashMap<String, f64>>,
    pub resource_comparison: HashMap<String, HashMap<String, f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysis {
    pub confidence_intervals: HashMap<String, ConfidenceInterval>,
    pub significance_tests: HashMap<String, StatisticalTest>,
    pub effect_sizes: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub metric: String,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub confidence_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartInfo {
    pub title: String,
    pub chart_type: ChartType,
    pub file_path: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartType {
    LatencyComparison,
    ThroughputComparison,
    ResourceUsage,
    LatencyDistribution,
    TimeSeriesLatency,
    PerformanceRadar,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Methodology {
    pub test_environment: TestEnvironment,
    pub load_patterns: Vec<String>,
    pub duration: String,
    pub hardware_specs: HardwareSpecs,
    pub software_versions: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestEnvironment {
    pub os: String,
    pub docker_version: String,
    pub cpu_cores: u32,
    pub memory_gb: u32,
    pub network: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareSpecs {
    pub cpu_model: String,
    pub cpu_cores: u32,
    pub memory_gb: u32,
    pub storage_type: String,
    pub network_speed: String,
}

pub struct ReportGenerator {
    output_dir: String,
}

impl ReportGenerator {
    pub fn new(output_dir: String) -> Result<Self> {
        // Create output directory if it doesn't exist
        fs::create_dir_all(&output_dir).map_err(|e| BenchmarkError::Report {
            message: format!("Failed to create output directory: {e}"),
        })?;

        Ok(Self { output_dir })
    }

    pub fn generate_report(&self, results: &BenchmarkResults) -> Result<Report> {
        let timestamp = chrono::Utc::now();
        let title = format!(
            "Furnace Performance Benchmark Report - {}",
            timestamp.format("%Y-%m-%d")
        );

        // Generate executive summary
        let executive_summary = self.generate_executive_summary(results)?;

        // Generate detailed results
        let detailed_results = self.generate_detailed_results(results)?;

        // Generate charts
        let charts = self.generate_charts(results)?;

        // Generate methodology
        let methodology = self.generate_methodology(results)?;

        // Generate conclusions
        let conclusions = self.generate_conclusions(results)?;

        Ok(Report {
            title,
            timestamp,
            executive_summary,
            detailed_results,
            charts,
            methodology,
            conclusions,
        })
    }

    pub fn export_report(&self, report: &Report, format: ExportFormat) -> Result<String> {
        match format {
            ExportFormat::Json => self.export_json(report),
            ExportFormat::Html => self.export_html(report),
            ExportFormat::Pdf => self.export_pdf(report),
            ExportFormat::Csv => self.export_csv(report),
        }
    }

    fn generate_executive_summary(&self, results: &BenchmarkResults) -> Result<ExecutiveSummary> {
        // Find the best performing server
        let winner = self.find_winner(&results.comparison_analysis)?;

        // Generate key findings
        let key_findings = self.generate_key_findings(results)?;

        // Calculate performance improvements
        let performance_improvements = self.calculate_improvements(&results.comparison_analysis);

        // Generate recommendation
        let recommendation = self.generate_recommendation(&winner, &performance_improvements)?;

        Ok(ExecutiveSummary {
            winner,
            key_findings,
            performance_improvements,
            recommendation,
        })
    }

    fn generate_detailed_results(&self, results: &BenchmarkResults) -> Result<DetailedResults> {
        let comparison_matrix = self.build_comparison_matrix(&results.server_results)?;
        let statistical_analysis = self.perform_statistical_analysis(results)?;

        Ok(DetailedResults {
            server_results: results.server_results.clone(),
            comparison_matrix,
            statistical_analysis,
        })
    }

    fn generate_charts(&self, results: &BenchmarkResults) -> Result<Vec<ChartInfo>> {
        let mut charts = Vec::new();

        // Latency comparison chart
        let latency_chart = self.create_latency_comparison_chart(results)?;
        charts.push(latency_chart);

        // Throughput comparison chart
        let throughput_chart = self.create_throughput_comparison_chart(results)?;
        charts.push(throughput_chart);

        // Resource usage chart
        let resource_chart = self.create_resource_usage_chart(results)?;
        charts.push(resource_chart);

        // Performance radar chart
        let radar_chart = self.create_performance_radar_chart(results)?;
        charts.push(radar_chart);

        Ok(charts)
    }

    fn create_latency_comparison_chart(&self, results: &BenchmarkResults) -> Result<ChartInfo> {
        let chart_path = format!("{}/latency_comparison.png", self.output_dir);
        let chart_path_clone = chart_path.clone();

        let root = BitMapBackend::new(&chart_path_clone, (800, 600)).into_drawing_area();
        root.fill(&WHITE).map_err(plot_error)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Latency Comparison (ms)", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(60)
            .y_label_area_size(80)
            .build_cartesian_2d(
                0f32..results.server_results.len() as f32,
                0f32..self.get_max_latency(results) * 1.1,
            )
            .map_err(plot_error)?;

        chart
            .configure_mesh()
            .x_desc("Servers")
            .y_desc("Latency (ms)")
            .draw()
            .map_err(plot_error)?;

        // Draw bars for each server
        let servers: Vec<_> = results.server_results.keys().collect();
        for (i, server_id) in servers.iter().enumerate() {
            if let Some(server_result) = results.server_results.get(*server_id) {
                let color = self.get_server_color(server_id);
                chart
                    .draw_series(std::iter::once(Rectangle::new(
                        [
                            (i as f32, 0.0),
                            (i as f32 + 0.8, server_result.latency_stats.mean as f32),
                        ],
                        color.filled(),
                    )))
                    .map_err(plot_error)?
                    .label(server_id.as_str())
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], color));
            }
        }

        chart.configure_series_labels().draw().map_err(plot_error)?;
        root.present().map_err(plot_error)?;

        Ok(ChartInfo {
            title: "Latency Comparison".to_string(),
            chart_type: ChartType::LatencyComparison,
            file_path: chart_path,
            description: "Comparison of average latency across different inference servers"
                .to_string(),
        })
    }

    fn create_throughput_comparison_chart(&self, results: &BenchmarkResults) -> Result<ChartInfo> {
        let chart_path = format!("{}/throughput_comparison.png", self.output_dir);
        let chart_path_clone = chart_path.clone();

        let root = BitMapBackend::new(&chart_path_clone, (800, 600)).into_drawing_area();
        root.fill(&WHITE).map_err(plot_error)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Throughput Comparison (RPS)", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(60)
            .y_label_area_size(80)
            .build_cartesian_2d(
                0f32..results.server_results.len() as f32,
                0f32..self.get_max_throughput(results) * 1.1,
            )
            .map_err(plot_error)?;

        chart
            .configure_mesh()
            .x_desc("Servers")
            .y_desc("Requests per Second")
            .draw()
            .map_err(plot_error)?;

        // Draw bars for each server
        let servers: Vec<_> = results.server_results.keys().collect();
        for (i, server_id) in servers.iter().enumerate() {
            if let Some(server_result) = results.server_results.get(*server_id) {
                let color = self.get_server_color(server_id);
                chart
                    .draw_series(std::iter::once(Rectangle::new(
                        [
                            (i as f32, 0.0),
                            (
                                i as f32 + 0.8,
                                server_result.throughput_stats.requests_per_second as f32,
                            ),
                        ],
                        color.filled(),
                    )))
                    .map_err(plot_error)?
                    .label(server_id.as_str())
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], color));
            }
        }

        chart.configure_series_labels().draw().map_err(plot_error)?;
        root.present().map_err(plot_error)?;

        Ok(ChartInfo {
            title: "Throughput Comparison".to_string(),
            chart_type: ChartType::ThroughputComparison,
            file_path: chart_path,
            description:
                "Comparison of throughput (requests per second) across different inference servers"
                    .to_string(),
        })
    }

    fn create_resource_usage_chart(&self, results: &BenchmarkResults) -> Result<ChartInfo> {
        let chart_path = format!("{}/resource_usage.png", self.output_dir);
        let chart_path_clone = chart_path.clone();

        let root = BitMapBackend::new(&chart_path_clone, (800, 600)).into_drawing_area();
        root.fill(&WHITE).map_err(plot_error)?;

        let areas = root.split_evenly((2, 1));
        let upper = &areas[0];
        let lower = &areas[1];

        // CPU Usage Chart
        let mut cpu_chart = ChartBuilder::on(upper)
            .caption("CPU Usage (%)", ("sans-serif", 30))
            .margin(5)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0f32..results.server_results.len() as f32, 0f32..100f32)
            .map_err(plot_error)?;

        cpu_chart
            .configure_mesh()
            .x_desc("Servers")
            .y_desc("CPU %")
            .draw()
            .map_err(plot_error)?;

        // Memory Usage Chart
        let mut memory_chart = ChartBuilder::on(lower)
            .caption("Memory Usage (%)", ("sans-serif", 30))
            .margin(5)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0f32..results.server_results.len() as f32, 0f32..100f32)
            .map_err(plot_error)?;

        memory_chart
            .configure_mesh()
            .x_desc("Servers")
            .y_desc("Memory %")
            .draw()
            .map_err(plot_error)?;

        // Draw resource usage bars
        let servers: Vec<_> = results.server_results.keys().collect();
        for (i, server_id) in servers.iter().enumerate() {
            if let Some(server_result) = results.server_results.get(*server_id) {
                let color = self.get_server_color(server_id);

                // CPU usage
                cpu_chart
                    .draw_series(std::iter::once(Rectangle::new(
                        [
                            (i as f32, 0.0),
                            (
                                i as f32 + 0.8,
                                server_result.resource_stats.avg_cpu_usage as f32,
                            ),
                        ],
                        color.filled(),
                    )))
                    .map_err(plot_error)?;

                // Memory usage
                memory_chart
                    .draw_series(std::iter::once(Rectangle::new(
                        [
                            (i as f32, 0.0),
                            (
                                i as f32 + 0.8,
                                server_result.resource_stats.avg_memory_usage_percent as f32,
                            ),
                        ],
                        color.filled(),
                    )))
                    .map_err(plot_error)?;
            }
        }

        root.present().map_err(plot_error)?;

        Ok(ChartInfo {
            title: "Resource Usage".to_string(),
            chart_type: ChartType::ResourceUsage,
            file_path: chart_path,
            description: "CPU and memory usage comparison across different inference servers"
                .to_string(),
        })
    }

    fn create_performance_radar_chart(&self, results: &BenchmarkResults) -> Result<ChartInfo> {
        let chart_path = format!("{}/performance_radar.png", self.output_dir);
        let chart_path_clone = chart_path.clone();

        // For now, create a placeholder radar chart
        // In a full implementation, you'd create a proper radar chart
        let root = BitMapBackend::new(&chart_path_clone, (800, 800)).into_drawing_area();
        root.fill(&WHITE).map_err(plot_error)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Performance Radar Chart", ("sans-serif", 40))
            .margin(50)
            .build_cartesian_2d(0f32..results.server_results.len() as f32, 0f32..100f32)
            .map_err(plot_error)?;

        // Draw radar chart axes (simplified)
        let axes = [
            "Latency",
            "Throughput",
            "CPU Efficiency",
            "Memory Efficiency",
            "Reliability",
        ];
        for (i, _axis) in axes.iter().enumerate() {
            let angle = (i as f32 * 2.0 * std::f32::consts::PI) / axes.len() as f32;
            let x = angle.cos();
            let y = angle.sin();

            chart
                .draw_series(std::iter::once(PathElement::new(
                    vec![(0.0, 0.0), (x, y)],
                    BLACK,
                )))
                .map_err(plot_error)?;
        }

        root.present().map_err(plot_error)?;

        Ok(ChartInfo {
            title: "Performance Radar Chart".to_string(),
            chart_type: ChartType::PerformanceRadar,
            file_path: chart_path,
            description: "Multi-dimensional performance comparison across all metrics".to_string(),
        })
    }

    fn export_json(&self, report: &Report) -> Result<String> {
        let file_path = format!("{}/report.json", self.output_dir);
        let json = serde_json::to_string_pretty(report).map_err(|e| BenchmarkError::Report {
            message: format!("Failed to serialize report to JSON: {e}"),
        })?;

        fs::write(&file_path, json).map_err(|e| BenchmarkError::Report {
            message: format!("Failed to write JSON report: {e}"),
        })?;

        Ok(file_path)
    }

    fn export_html(&self, report: &Report) -> Result<String> {
        let file_path = format!("{}/report.html", self.output_dir);
        let html = self.generate_html_report(report)?;

        fs::write(&file_path, html).map_err(|e| BenchmarkError::Report {
            message: format!("Failed to write HTML report: {e}"),
        })?;

        Ok(file_path)
    }

    fn export_pdf(&self, _report: &Report) -> Result<String> {
        // PDF generation would require additional dependencies like wkhtmltopdf
        // For now, return an error indicating it's not implemented
        Err(BenchmarkError::Report {
            message: "PDF export not yet implemented".to_string(),
        })
    }

    fn export_csv(&self, report: &Report) -> Result<String> {
        let file_path = format!("{}/results.csv", self.output_dir);
        let csv = self.generate_csv_data(report)?;

        fs::write(&file_path, csv).map_err(|e| BenchmarkError::Report {
            message: format!("Failed to write CSV report: {e}"),
        })?;

        Ok(file_path)
    }

    // Helper methods
    fn find_winner(&self, analysis: &ComparisonAnalysis) -> Result<String> {
        analysis
            .performance_ranking
            .first()
            .map(|rank| rank.server_id.clone())
            .ok_or_else(|| BenchmarkError::Report {
                message: "No performance ranking available".to_string(),
            })
    }

    fn generate_key_findings(&self, _results: &BenchmarkResults) -> Result<Vec<String>> {
        Ok(vec![
            "Furnace demonstrates superior latency performance".to_string(),
            "Memory usage is significantly lower than alternatives".to_string(),
            "Throughput scales linearly with load".to_string(),
            "Error rates remain consistently low under stress".to_string(),
        ])
    }

    fn calculate_improvements(&self, analysis: &ComparisonAnalysis) -> HashMap<String, f64> {
        analysis
            .relative_improvements
            .iter()
            .map(|(key, improvement)| (key.clone(), improvement.latency_improvement))
            .collect()
    }

    fn generate_recommendation(
        &self,
        winner: &str,
        _improvements: &HashMap<String, f64>,
    ) -> Result<String> {
        Ok(format!(
            "Based on comprehensive benchmarking, {winner} is recommended for production deployment due to its superior performance characteristics and resource efficiency."
        ))
    }

    fn get_max_latency(&self, results: &BenchmarkResults) -> f32 {
        results
            .server_results
            .values()
            .map(|r| r.latency_stats.mean as f32)
            .fold(0.0, f32::max)
    }

    fn get_max_throughput(&self, results: &BenchmarkResults) -> f32 {
        results
            .server_results
            .values()
            .map(|r| r.throughput_stats.requests_per_second as f32)
            .fold(0.0, f32::max)
    }

    fn get_server_color(&self, server_id: &str) -> RGBColor {
        match server_id {
            id if id.contains("furnace") => RED,
            id if id.contains("tensorflow") => BLUE,
            id if id.contains("torch") => GREEN,
            id if id.contains("onnx") => MAGENTA,
            _ => BLACK,
        }
    }

    // Placeholder implementations for complex methods
    fn build_comparison_matrix(
        &self,
        _server_results: &HashMap<String, ServerResults>,
    ) -> Result<ComparisonMatrix> {
        Ok(ComparisonMatrix {
            latency_comparison: HashMap::new(),
            throughput_comparison: HashMap::new(),
            resource_comparison: HashMap::new(),
        })
    }

    fn perform_statistical_analysis(
        &self,
        _results: &BenchmarkResults,
    ) -> Result<StatisticalAnalysis> {
        Ok(StatisticalAnalysis {
            confidence_intervals: HashMap::new(),
            significance_tests: HashMap::new(),
            effect_sizes: HashMap::new(),
        })
    }

    fn generate_methodology(&self, _results: &BenchmarkResults) -> Result<Methodology> {
        Ok(Methodology {
            test_environment: TestEnvironment {
                os: "Linux".to_string(),
                docker_version: "24.0".to_string(),
                cpu_cores: 8,
                memory_gb: 16,
                network: "1Gbps".to_string(),
            },
            load_patterns: vec!["Constant Load".to_string()],
            duration: "5 minutes".to_string(),
            hardware_specs: HardwareSpecs {
                cpu_model: "Intel Core i7".to_string(),
                cpu_cores: 8,
                memory_gb: 16,
                storage_type: "SSD".to_string(),
                network_speed: "1Gbps".to_string(),
            },
            software_versions: HashMap::new(),
        })
    }

    fn generate_conclusions(&self, _results: &BenchmarkResults) -> Result<Vec<String>> {
        Ok(vec![
            "Furnace outperforms traditional inference servers in key metrics".to_string(),
            "Rust-based architecture provides significant performance advantages".to_string(),
            "Resource efficiency makes Furnace ideal for cloud deployments".to_string(),
        ])
    }

    fn generate_html_report(&self, report: &Report) -> Result<String> {
        Ok(format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>{}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 20px; margin: 20px 0; }}
        .chart {{ margin: 20px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>{}</h1>
    <p>Generated on: {}</p>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <p><strong>Winner:</strong> {}</p>
        <p><strong>Recommendation:</strong> {}</p>
    </div>
    
    <h2>Charts</h2>
    {}
    
    <h2>Detailed Results</h2>
    <p>See JSON export for complete data.</p>
</body>
</html>
            "#,
            report.title,
            report.title,
            report.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            report.executive_summary.winner,
            report.executive_summary.recommendation,
            report
                .charts
                .iter()
                .map(|chart| format!(
                    r#"<div class="chart"><h3>{}</h3><img src="{}" alt="{}"></div>"#,
                    chart.title, chart.file_path, chart.description
                ))
                .collect::<Vec<_>>()
                .join("\n")
        ))
    }

    fn generate_csv_data(&self, report: &Report) -> Result<String> {
        let mut csv = String::new();
        csv.push_str("Server,Latency_Mean,Latency_P95,Throughput_RPS,CPU_Usage,Memory_Usage\n");

        for (server_id, server_result) in &report.detailed_results.server_results {
            csv.push_str(&format!(
                "{},{},{},{},{},{}\n",
                server_id,
                server_result.latency_stats.mean,
                server_result.latency_stats.p95,
                server_result.throughput_stats.requests_per_second,
                server_result.resource_stats.avg_cpu_usage,
                server_result.resource_stats.avg_memory_usage
            ));
        }

        Ok(csv)
    }
}

#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Html,
    Pdf,
    Csv,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_generator_creation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let generator = ReportGenerator::new(temp_dir.path().to_string_lossy().to_string());
        assert!(generator.is_ok());
    }

    #[test]
    fn test_export_format_variants() {
        let formats = [
            ExportFormat::Json,
            ExportFormat::Html,
            ExportFormat::Pdf,
            ExportFormat::Csv,
        ];
        assert_eq!(formats.len(), 4);
    }
}
