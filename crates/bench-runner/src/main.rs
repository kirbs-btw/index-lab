mod scenarios;

use std::{
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{ensure, Context, Result};
use clap::Parser;
use index_core::{
    generate_query_set, generate_uniform_dataset, DistanceMetric, ScoredPoint, Vector, VectorIndex,
};
use index_linear::LinearIndex;
use serde::Serialize;
use scenarios::{ScenarioDetails, ScenarioKind};

#[derive(Debug, Parser)]
#[command(about = "Lightweight harness for experimenting with vector indexes")]
struct Cli {
    /// Dimensionality of the synthetic dataset
    #[arg(long, default_value = "128")]
    dimension: usize,
    /// Number of points to insert into the index
    #[arg(long, default_value = "10000")]
    points: usize,
    /// Number of query vectors to evaluate
    #[arg(long, default_value = "128")]
    queries: usize,
    /// Number of neighbours to request per query
    #[arg(long, default_value = "10")]
    limit: usize,
    /// Distance metric to use (euclidean | cosine)
    #[arg(long, default_value = "euclidean", value_parser = parse_metric)]
    metric: DistanceMetric,
    /// RNG seed used for dataset generation
    #[arg(long, default_value = "42")]
    seed: u64,
    /// List the baked-in benchmark scenarios
    #[arg(long)]
    list_scenarios: bool,
    /// Execute a named benchmark scenario instead of manual knobs
    #[arg(long, value_enum)]
    scenario: Option<ScenarioKind>,
    /// Export the generated dataset and queries as JSON for reuse
    #[arg(long)]
    export_testdata: Option<PathBuf>,
    /// Export timing + configuration data as JSON
    #[arg(long)]
    report_json: Option<PathBuf>,
}

fn parse_metric(value: &str) -> Result<DistanceMetric, String> {
    match value.to_ascii_lowercase().as_str() {
        "euclidean" | "l2" => Ok(DistanceMetric::Euclidean),
        "cosine" => Ok(DistanceMetric::Cosine),
        other => Err(format!("unsupported metric '{other}'")),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.list_scenarios {
        scenarios::print_available();
        return Ok(());
    }

    let mut runtime = RuntimeConfig::from_cli(&cli);

    if let Some(kind) = cli.scenario {
        let details = kind.details();
        println!("Using scenario '{}' – {}", details.slug, details.description);
        runtime.apply_scenario(&details);
    }

    ensure!(
        runtime.queries > 0,
        "queries must be greater than zero to run the benchmark"
    );

    let dataset = generate_uniform_dataset(
        runtime.dimension,
        runtime.points,
        -1.0_f32..1.0_f32,
        runtime.seed,
    );
    let queries = generate_query_set(
        runtime.dimension,
        runtime.queries,
        -1.0_f32..1.0_f32,
        runtime.seed.wrapping_add(1),
    );

    if let Some(path) = cli.export_testdata.as_deref() {
        export_testdata(path, &runtime, &dataset, &queries)?;
        println!("Wrote test data to {}", path.display());
    }

    let mut index = LinearIndex::new(runtime.metric);
    let build_start = Instant::now();
    index.build(dataset)?;
    let build_time = build_start.elapsed();

    let mut total_search_time = 0u128;
    let mut first_result: Option<Vec<ScoredPoint>> = None;
    for query in &queries {
        let search_start = Instant::now();
        let result = index.search(query, runtime.limit)?;
        total_search_time += search_start.elapsed().as_micros();
        if first_result.is_none() {
            first_result = Some(result.clone());
        }
    }

    if let Some(first) = &first_result {
        println!(
            "First query result ids: {:?}",
            first.iter().map(|point| point.id).collect::<Vec<_>>()
        );
    }

    let stats = BenchmarkStats::new(
        build_time,
        total_search_time,
        runtime.queries,
        estimate_dataset_bytes(runtime.dimension, runtime.points),
    );

    let scenario_note = runtime
        .scenario_slug
        .map(|slug| format!(" | Scenario: {slug}"))
        .unwrap_or_default();
    println!(
        "Build: {:.2?} | Avg search: {:.2} us | QPS: {:.1} | Dataset: {:.2} MiB | Metric: {:?} | Points: {} | Queries: {} | Dim: {} | Limit: {}{}",
        stats.build_time,
        stats.avg_search_micros,
        stats.queries_per_second,
        stats.dataset_bytes as f64 / (1024.0 * 1024.0),
        runtime.metric,
        runtime.points,
        runtime.queries,
        runtime.dimension,
        runtime.limit,
        scenario_note
    );

    if let Some(path) = cli.report_json.as_deref() {
        write_report(path, &runtime, &stats)?;
        println!("Wrote benchmark report to {}", path.display());
    }

    Ok(())
}

#[derive(Debug, Clone)]
struct RuntimeConfig {
    dimension: usize,
    points: usize,
    queries: usize,
    limit: usize,
    metric: DistanceMetric,
    seed: u64,
    scenario_slug: Option<&'static str>,
    scenario_label: Option<&'static str>,
    scenario_description: Option<&'static str>,
}

impl RuntimeConfig {
    fn from_cli(cli: &Cli) -> Self {
        Self {
            dimension: cli.dimension,
            points: cli.points,
            queries: cli.queries,
            limit: cli.limit,
            metric: cli.metric,
            seed: cli.seed,
            scenario_slug: None,
            scenario_label: None,
            scenario_description: None,
        }
    }

    fn apply_scenario(&mut self, details: &ScenarioDetails) {
        self.dimension = details.config.dimension;
        self.points = details.config.points;
        self.queries = details.config.queries;
        self.limit = details.config.limit;
        self.metric = details.config.metric;
        self.scenario_slug = Some(details.slug);
        self.scenario_label = Some(details.label);
        self.scenario_description = Some(details.description);
    }
}

struct BenchmarkStats {
    build_time: Duration,
    avg_search_micros: f64,
    queries_per_second: f64,
    dataset_bytes: usize,
}

impl BenchmarkStats {
    fn new(
        build_time: Duration,
        total_search_micros: u128,
        queries: usize,
        dataset_bytes: usize,
    ) -> Self {
        let avg_search_micros = if queries == 0 {
            0.0
        } else {
            total_search_micros as f64 / queries as f64
        };
        let queries_per_second = if avg_search_micros == 0.0 {
            f64::INFINITY
        } else {
            1_000_000.0 / avg_search_micros
        };
        Self {
            build_time,
            avg_search_micros,
            queries_per_second,
            dataset_bytes,
        }
    }
}

fn estimate_dataset_bytes(dimension: usize, points: usize) -> usize {
    dimension
        .saturating_mul(points)
        .saturating_mul(std::mem::size_of::<f32>())
}

fn write_report(path: &Path, runtime: &RuntimeConfig, stats: &BenchmarkStats) -> Result<()> {
    #[derive(Serialize)]
    struct ReportPayload<'a> {
        config: ReportConfig<'a>,
        timings: ReportTimings,
    }

    #[derive(Serialize)]
    struct ReportConfig<'a> {
        scenario: Option<&'a str>,
        label: Option<&'a str>,
        description: Option<&'a str>,
        dimension: usize,
        points: usize,
        queries: usize,
        limit: usize,
        metric: DistanceMetric,
        seed: u64,
        dataset_bytes: usize,
    }

    #[derive(Serialize)]
    struct ReportTimings {
        build_ms: f64,
        avg_search_micros: f64,
        queries_per_second: f64,
    }

    let payload = ReportPayload {
        config: ReportConfig {
            scenario: runtime.scenario_slug,
            label: runtime.scenario_label,
            description: runtime.scenario_description,
            dimension: runtime.dimension,
            points: runtime.points,
            queries: runtime.queries,
            limit: runtime.limit,
            metric: runtime.metric,
            seed: runtime.seed,
            dataset_bytes: stats.dataset_bytes,
        },
        timings: ReportTimings {
            build_ms: stats.build_time.as_secs_f64() * 1_000.0,
            avg_search_micros: stats.avg_search_micros,
            queries_per_second: stats.queries_per_second,
        },
    };

    let file = File::create(path)
        .with_context(|| format!("failed to create benchmark report at {}", path.display()))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &payload)
        .context("failed to write benchmark report JSON")?;
    Ok(())
}

fn export_testdata(
    path: &Path,
    runtime: &RuntimeConfig,
    dataset: &[(usize, Vector)],
    queries: &[Vector],
) -> Result<()> {
    #[derive(Serialize)]
    struct ExportMetadata<'a> {
        scenario: Option<&'a str>,
        label: Option<&'a str>,
        description: Option<&'a str>,
        dimension: usize,
        points: usize,
        queries: usize,
        limit: usize,
        metric: DistanceMetric,
        seed: u64,
    }

    #[derive(Serialize)]
    struct ExportPoint<'a> {
        id: usize,
        values: &'a [f32],
    }

    #[derive(Serialize)]
    struct TestDataExport<'a> {
        metadata: ExportMetadata<'a>,
        dataset: Vec<ExportPoint<'a>>,
        queries: Vec<&'a [f32]>,
    }

    let payload = TestDataExport {
        metadata: ExportMetadata {
            scenario: runtime.scenario_slug,
            label: runtime.scenario_label,
            description: runtime.scenario_description,
            dimension: runtime.dimension,
            points: runtime.points,
            queries: runtime.queries,
            limit: runtime.limit,
            metric: runtime.metric,
            seed: runtime.seed,
        },
        dataset: dataset
            .iter()
            .map(|(id, vector)| ExportPoint {
                id: *id,
                values: vector,
            })
            .collect(),
        queries: queries.iter().map(|vector| vector.as_slice()).collect(),
    };

    let file = File::create(path)
        .with_context(|| format!("failed to create test data file at {}", path.display()))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &payload)
        .context("failed to write test data JSON")?;
    Ok(())
}
