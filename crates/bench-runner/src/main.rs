mod scenarios;

use std::{
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{ensure, Context, Result};
use clap::{Parser, ValueEnum};
use index_core::{
    generate_query_set, generate_uniform_dataset, DistanceMetric, ScoredPoint, Vector, VectorIndex,
};
use index_apex::ApexIndex;
use index_armi::ArmiIndex;
use index_synthesis::SynthesisIndex;
use index_fusion::FusionIndex;
use index_hnsw::HnswIndex;
use index_hybrid::HybridIndex;
use index_ivf::IvfIndex;
use index_lim::LimIndex;
use index_linear::LinearIndex;
use index_nexus::NexusIndex;
use index_pq::PqIndex;
use index_prism::PrismIndex;
use index_seer::SeerIndex;
use index_swift::SwiftIndex;
use index_vortex::VortexIndex;
use scenarios::{ScenarioDetails, ScenarioKind};
use serde::Serialize;
use std::collections::HashSet;

/// Wrapper enum for different index types
enum IndexWrapper {
    Linear(LinearIndex),
    Hnsw(HnswIndex),
    Ivf(IvfIndex),
    Pq(PqIndex),
    Lim(LimIndex),
    Hybrid(HybridIndex),
    Seer(SeerIndex),
    Swift(SwiftIndex),
    Prism(PrismIndex),
    Nexus(NexusIndex),
    Fusion(FusionIndex),
    Vortex(VortexIndex),
    Armi(ArmiIndex),
    Apex(ApexIndex),
    Synthesis(SynthesisIndex),
}

impl VectorIndex for IndexWrapper {
    fn metric(&self) -> DistanceMetric {
        match self {
            Self::Linear(idx) => idx.metric(),
            Self::Hnsw(idx) => idx.metric(),
            Self::Ivf(idx) => idx.metric(),
            Self::Pq(idx) => idx.metric(),
            Self::Lim(idx) => idx.metric(),
            Self::Hybrid(idx) => idx.metric(),
            Self::Seer(idx) => idx.metric(),
            Self::Swift(idx) => idx.metric(),
            Self::Prism(idx) => idx.metric(),
            Self::Nexus(idx) => idx.metric(),
            Self::Fusion(idx) => idx.metric(),
            Self::Vortex(idx) => idx.metric(),
            Self::Armi(idx) => idx.metric(),
            Self::Apex(idx) => idx.metric(),
            Self::Synthesis(idx) => idx.metric(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Linear(idx) => idx.len(),
            Self::Hnsw(idx) => idx.len(),
            Self::Ivf(idx) => idx.len(),
            Self::Pq(idx) => idx.len(),
            Self::Lim(idx) => idx.len(),
            Self::Hybrid(idx) => idx.len(),
            Self::Seer(idx) => idx.len(),
            Self::Swift(idx) => idx.len(),
            Self::Prism(idx) => idx.len(),
            Self::Nexus(idx) => idx.len(),
            Self::Fusion(idx) => idx.len(),
            Self::Vortex(idx) => idx.len(),
            Self::Armi(idx) => idx.len(),
            Self::Apex(idx) => idx.len(),
            Self::Synthesis(idx) => idx.len(),
        }
    }

    fn insert(&mut self, id: usize, vector: Vector) -> Result<()> {
        match self {
            Self::Linear(idx) => idx.insert(id, vector),
            Self::Hnsw(idx) => idx.insert(id, vector),
            Self::Ivf(idx) => idx.insert(id, vector),
            Self::Pq(idx) => idx.insert(id, vector),
            Self::Lim(idx) => idx.insert(id, vector),
            Self::Hybrid(idx) => idx.insert(id, vector),
            Self::Seer(idx) => idx.insert(id, vector),
            Self::Swift(idx) => idx.insert(id, vector),
            Self::Prism(idx) => idx.insert(id, vector),
            Self::Nexus(idx) => idx.insert(id, vector),
            Self::Fusion(idx) => idx.insert(id, vector),
            Self::Vortex(idx) => idx.insert(id, vector),
            Self::Armi(idx) => idx.insert(id, vector),
            Self::Apex(idx) => idx.insert(id, vector),
            Self::Synthesis(idx) => idx.insert(id, vector),
        }
    }

    fn search(&self, query: &Vector, limit: usize) -> Result<Vec<ScoredPoint>> {
        match self {
            Self::Linear(idx) => idx.search(query, limit),
            Self::Hnsw(idx) => idx.search(query, limit),
            Self::Ivf(idx) => idx.search(query, limit),
            Self::Pq(idx) => idx.search(query, limit),
            Self::Lim(idx) => idx.search(query, limit),
            Self::Hybrid(idx) => idx.search(query, limit),
            Self::Seer(idx) => idx.search(query, limit),
            Self::Swift(idx) => idx.search(query, limit),
            Self::Prism(idx) => idx.search(query, limit),
            Self::Nexus(idx) => idx.search(query, limit),
            Self::Fusion(idx) => idx.search(query, limit),
            Self::Vortex(idx) => idx.search(query, limit),
            Self::Armi(idx) => idx.search(query, limit),
            Self::Apex(idx) => idx.search(query, limit),
            Self::Synthesis(idx) => idx.search(query, limit),
        }
    }
}

#[derive(Debug, Clone, ValueEnum)]
enum IndexType {
    Linear,
    Hnsw,
    Ivf,
    Pq,
    Lim,
    Hybrid,
    Seer,
    Swift,
    Prism,
    Nexus,
    Fusion,
    Vortex,
    Armi,
    Apex,
    Synthesis,
}

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
    /// Index type to use (linear | hnsw | ivf | pq)
    #[arg(long, default_value = "linear", value_enum)]
    index_type: IndexType,
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
    /// Save the built index to a JSON file
    #[arg(long)]
    save_index: Option<PathBuf>,
    /// Load an index from a JSON file instead of building it
    #[arg(long)]
    load_index: Option<PathBuf>,
}

fn parse_metric(value: &str) -> Result<DistanceMetric, String> {
    match value.to_ascii_lowercase().as_str() {
        "euclidean" | "l2" => Ok(DistanceMetric::Euclidean),
        "cosine" => Ok(DistanceMetric::Cosine),
        other => Err(format!("unsupported metric '{other}'")),
    }
}

/// Computes ground truth results using exhaustive linear search
fn compute_ground_truth(
    dataset: &[(usize, Vector)],
    queries: &[Vector],
    metric: DistanceMetric,
    limit: usize,
) -> Result<Vec<Vec<ScoredPoint>>> {
    let mut ground_truth_index = LinearIndex::new(metric);
    ground_truth_index.build(dataset.iter().cloned())?;

    let mut ground_truth_results = Vec::with_capacity(queries.len());
    for query in queries {
        let result = ground_truth_index.search(query, limit)?;
        ground_truth_results.push(result);
    }

    Ok(ground_truth_results)
}

/// Calculates recall@k by comparing approximate results with ground truth
fn calculate_recall(
    approximate_results: &[ScoredPoint],
    ground_truth: &[ScoredPoint],
    k: usize,
) -> f64 {
    if ground_truth.is_empty() || k == 0 {
        return 0.0;
    }

    // Create a set of IDs from ground truth (top k)
    let ground_truth_ids: HashSet<usize> =
        ground_truth.iter().take(k).map(|point| point.id).collect();

    // Count how many of the top k ground truth IDs are in approximate results
    let found_count = approximate_results
        .iter()
        .take(k)
        .filter(|point| ground_truth_ids.contains(&point.id))
        .count();

    found_count as f64 / k.min(ground_truth.len()) as f64
}

/// Computes recall metrics for all queries
fn compute_recall_metrics(
    approximate_results: &[Vec<ScoredPoint>],
    ground_truth: &[Vec<ScoredPoint>],
    limit: usize,
) -> (f64, f64, f64) {
    if approximate_results.len() != ground_truth.len() {
        return (0.0, 0.0, 0.0);
    }

    let mut recalls = Vec::with_capacity(approximate_results.len());
    for (approx, truth) in approximate_results.iter().zip(ground_truth.iter()) {
        let recall = calculate_recall(approx, truth, limit);
        recalls.push(recall);
    }

    let avg_recall = recalls.iter().sum::<f64>() / recalls.len() as f64;
    let min_recall = recalls.iter().copied().fold(f64::INFINITY, f64::min);
    let max_recall = recalls.iter().copied().fold(0.0, f64::max);

    (avg_recall, min_recall, max_recall)
}

/// Helper to save an index if a save path is provided
fn save_index_if_requested<F>(save_path: Option<&Path>, save_fn: F) -> Result<()>
where
    F: FnOnce(&Path) -> Result<()>,
{
    if let Some(path) = save_path {
        save_fn(path)?;
    }
    Ok(())
}

/// Generic benchmark runner that handles common logic for all index types
#[allow(clippy::too_many_arguments)]
fn run_benchmark<F1, F2, F3>(
    _index_name: &str,
    load_fn: F1,
    build_fn: F2,
    save_fn: F3,
    is_exhaustive: bool,
    dataset: &[(usize, Vector)],
    queries: &[Vector],
    runtime: &RuntimeConfig,
    cli: &Cli,
) -> Result<()>
where
    F1: FnOnce(&Path) -> Result<(IndexWrapper, Duration)>,
    F2: FnOnce() -> Result<(IndexWrapper, Duration)>,
    F3: FnOnce(&IndexWrapper, &Path) -> Result<()>,
{
    // Load or build index
    let (index, build_time) = if let Some(load_path) = cli.load_index.as_deref() {
        load_fn(load_path)?
    } else {
        build_fn()?
    };

    // Run search
    let mut total_search_time = 0u128;
    let mut all_results = Vec::with_capacity(queries.len());
    let mut first_result: Option<Vec<ScoredPoint>> = None;
    for query in queries {
        let search_start = Instant::now();
        let result = index.search(query, runtime.limit)?;
        total_search_time += search_start.elapsed().as_micros();
        all_results.push(result.clone());
        if first_result.is_none() {
            first_result = Some(result);
        }
    }

    // Compute ground truth and recall
    let recall_metrics = if is_exhaustive {
        // Exhaustive index (linear) always has perfect recall
        (1.0, 1.0, 1.0)
    } else {
        println!("Computing ground truth with exhaustive search...");
        let ground_truth_start = Instant::now();
        let ground_truth = compute_ground_truth(dataset, queries, runtime.metric, runtime.limit)?;
        let ground_truth_time = ground_truth_start.elapsed();
        println!("Ground truth computed in {:.2?}", ground_truth_time);

        let (avg_recall, min_recall, max_recall) =
            compute_recall_metrics(&all_results, &ground_truth, runtime.limit);
        (avg_recall, min_recall, max_recall)
    };

    // Save if requested
    save_index_if_requested(cli.save_index.as_deref(), |path| save_fn(&index, path))?;

    print_results(
        &index,
        &first_result,
        build_time,
        total_search_time,
        runtime,
        cli,
        recall_metrics,
    )?;

    Ok(())
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
        println!(
            "Using scenario '{}' – {}",
            details.slug, details.description
        );
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

    // Run benchmark with the selected index type
    match cli.index_type {
        IndexType::Linear => {
            run_benchmark(
                "linear",
                |load_path| {
                    println!("Loading linear index from {}...", load_path.display());
                    let load_start = Instant::now();
                    let loaded = LinearIndex::load(load_path).with_context(|| {
                        format!("failed to load index from {}", load_path.display())
                    })?;
                    let load_time = load_start.elapsed();
                    println!(
                        "Loaded index in {:.2?} ({} vectors, metric: {:?})",
                        load_time,
                        loaded.len(),
                        loaded.metric()
                    );
                    Ok((IndexWrapper::Linear(loaded), load_time))
                },
                || {
                    let mut index = LinearIndex::new(runtime.metric);
                    let build_start = Instant::now();
                    index.build(dataset.clone())?;
                    let build_time = build_start.elapsed();
                    Ok((IndexWrapper::Linear(index), build_time))
                },
                |idx, path| {
                    if let IndexWrapper::Linear(inner) = idx {
                        inner.save(path).with_context(|| {
                            format!("failed to save index to {}", path.display())
                        })?;
                        println!(
                            "Saved linear index to {} ({} vectors)",
                            path.display(),
                            inner.len()
                        );
                    }
                    Ok(())
                },
                true, // is_exhaustive
                &dataset,
                &queries,
                &runtime,
                &cli,
            )?;
        }
        IndexType::Hnsw => {
            run_benchmark(
                "HNSW",
                |load_path| {
                    println!("Loading HNSW index from {}...", load_path.display());
                    let load_start = Instant::now();
                    let loaded = HnswIndex::load(load_path).with_context(|| {
                        format!("failed to load index from {}", load_path.display())
                    })?;
                    let load_time = load_start.elapsed();
                    println!(
                        "Loaded index in {:.2?} ({} vectors, metric: {:?})",
                        load_time,
                        loaded.len(),
                        loaded.metric()
                    );
                    Ok((IndexWrapper::Hnsw(loaded), load_time))
                },
                || {
                    let mut index = HnswIndex::with_defaults(runtime.metric);
                    let build_start = Instant::now();
                    index.build(dataset.clone())?;
                    let build_time = build_start.elapsed();
                    Ok((IndexWrapper::Hnsw(index), build_time))
                },
                |idx, path| {
                    if let IndexWrapper::Hnsw(inner) = idx {
                        inner.save(path).with_context(|| {
                            format!("failed to save index to {}", path.display())
                        })?;
                        println!(
                            "Saved HNSW index to {} ({} vectors)",
                            path.display(),
                            inner.len()
                        );
                    }
                    Ok(())
                },
                false,
                &dataset,
                &queries,
                &runtime,
                &cli,
            )?;
        }
        IndexType::Ivf => {
            run_benchmark(
                "IVF",
                |load_path| {
                    println!("Loading IVF index from {}...", load_path.display());
                    let load_start = Instant::now();
                    let loaded = IvfIndex::load(load_path).with_context(|| {
                        format!("failed to load index from {}", load_path.display())
                    })?;
                    let load_time = load_start.elapsed();
                    println!(
                        "Loaded index in {:.2?} ({} vectors, metric: {:?})",
                        load_time,
                        loaded.len(),
                        loaded.metric()
                    );
                    Ok((IndexWrapper::Ivf(loaded), load_time))
                },
                || {
                    let mut index = IvfIndex::with_defaults(runtime.metric);
                    let build_start = Instant::now();
                    index.build(dataset.clone())?;
                    let build_time = build_start.elapsed();
                    Ok((IndexWrapper::Ivf(index), build_time))
                },
                |idx, path| {
                    if let IndexWrapper::Ivf(inner) = idx {
                        inner.save(path).with_context(|| {
                            format!("failed to save index to {}", path.display())
                        })?;
                        println!(
                            "Saved IVF index to {} ({} vectors)",
                            path.display(),
                            inner.len()
                        );
                    }
                    Ok(())
                },
                false,
                &dataset,
                &queries,
                &runtime,
                &cli,
            )?;
        }
        IndexType::Pq => {
            run_benchmark(
                "PQ",
                |load_path| {
                    println!("Loading PQ index from {}...", load_path.display());
                    let load_start = Instant::now();
                    let loaded = PqIndex::load(load_path).with_context(|| {
                        format!("failed to load index from {}", load_path.display())
                    })?;
                    let load_time = load_start.elapsed();
                    println!(
                        "Loaded index in {:.2?} ({} vectors, metric: {:?})",
                        load_time,
                        loaded.len(),
                        loaded.metric()
                    );
                    Ok((IndexWrapper::Pq(loaded), load_time))
                },
                || {
                    let mut index = PqIndex::with_defaults(runtime.metric);
                    let build_start = Instant::now();
                    index.build(dataset.clone())?;
                    let build_time = build_start.elapsed();
                    Ok((IndexWrapper::Pq(index), build_time))
                },
                |idx, path| {
                    if let IndexWrapper::Pq(inner) = idx {
                        inner.save(path).with_context(|| {
                            format!("failed to save index to {}", path.display())
                        })?;
                        println!(
                            "Saved PQ index to {} ({} vectors)",
                            path.display(),
                            inner.len()
                        );
                    }
                    Ok(())
                },
                false,
                &dataset,
                &queries,
                &runtime,
                &cli,
            )?;
        }
        IndexType::Lim => {
            run_benchmark(
                "LIM",
                |load_path| {
                    println!("Loading LIM index from {}...", load_path.display());
                    let load_start = Instant::now();
                    let loaded = LimIndex::load(load_path).with_context(|| {
                        format!("failed to load index from {}", load_path.display())
                    })?;
                    let load_time = load_start.elapsed();
                    println!(
                        "Loaded index in {:.2?} ({} vectors, metric: {:?})",
                        load_time,
                        loaded.len(),
                        loaded.metric()
                    );
                    Ok((IndexWrapper::Lim(loaded), load_time))
                },
                || {
                    let mut index = LimIndex::with_defaults(runtime.metric);
                    let build_start = Instant::now();
                    index.build(dataset.clone())?;
                    let build_time = build_start.elapsed();
                    Ok((IndexWrapper::Lim(index), build_time))
                },
                |idx, path| {
                    if let IndexWrapper::Lim(inner) = idx {
                        inner.save(path).with_context(|| {
                            format!("failed to save index to {}", path.display())
                        })?;
                        println!(
                            "Saved LIM index to {} ({} vectors)",
                            path.display(),
                            inner.len()
                        );
                    }
                    Ok(())
                },
                false, // Set to true if LIM is exhaustive (like linear)
                &dataset,
                &queries,
                &runtime,
                &cli,
            )?;
        }
        IndexType::Hybrid => {
            run_benchmark(
                "Hybrid",
                |load_path| {
                    println!("Loading Hybrid index from {}...", load_path.display());
                    let load_start = Instant::now();
                    let loaded = HybridIndex::load(load_path).with_context(|| {
                        format!("failed to load index from {}", load_path.display())
                    })?;
                    let load_time = load_start.elapsed();
                    println!(
                        "Loaded index in {:.2?} ({} vectors, metric: {:?})",
                        load_time,
                        loaded.len(),
                        loaded.metric()
                    );
                    Ok((IndexWrapper::Hybrid(loaded), load_time))
                },
                || {
                    let mut index = HybridIndex::with_defaults(runtime.metric);
                    let build_start = Instant::now();
                    index.build(dataset.clone())?;
                    let build_time = build_start.elapsed();
                    Ok((IndexWrapper::Hybrid(index), build_time))
                },
                |idx, path| {
                    if let IndexWrapper::Hybrid(inner) = idx {
                        inner.save(path).with_context(|| {
                            format!("failed to save index to {}", path.display())
                        })?;
                        println!(
                            "Saved Hybrid index to {} ({} vectors)",
                            path.display(),
                            inner.len()
                        );
                    }
                    Ok(())
                },
                false, // Not exhaustive (uses linear scan for now, but sparse matching is approximate)
                &dataset,
                &queries,
                &runtime,
                &cli,
            )?;
        }
        IndexType::Seer => {
            run_benchmark(
                "SEER",
                |load_path| {
                    println!("Loading SEER index from {}...", load_path.display());
                    let load_start = Instant::now();
                    let loaded = SeerIndex::load(load_path).with_context(|| {
                        format!("failed to load index from {}", load_path.display())
                    })?;
                    let load_time = load_start.elapsed();
                    println!(
                        "Loaded index in {:.2?} ({} vectors, metric: {:?})",
                        load_time,
                        loaded.len(),
                        loaded.metric()
                    );
                    Ok((IndexWrapper::Seer(loaded), load_time))
                },
                || {
                    let mut index = SeerIndex::with_defaults(runtime.metric);
                    let build_start = Instant::now();
                    index.build(dataset.clone())?;
                    let build_time = build_start.elapsed();
                    Ok((IndexWrapper::Seer(index), build_time))
                },
                |idx, path| {
                    if let IndexWrapper::Seer(inner) = idx {
                        inner.save(path).with_context(|| {
                            format!("failed to save index to {}", path.display())
                        })?;
                        println!(
                            "Saved SEER index to {} ({} vectors)",
                            path.display(),
                            inner.len()
                        );
                    }
                    Ok(())
                },
                false, // Not exhaustive (uses learned candidate filtering)
                &dataset,
                &queries,
                &runtime,
                &cli,
            )?;
        }
        IndexType::Swift => {
            run_benchmark(
                "SWIFT",
                |load_path| {
                    println!("Loading SWIFT index from {}...", load_path.display());
                    let load_start = Instant::now();
                    let loaded = SwiftIndex::load(load_path).with_context(|| {
                        format!("failed to load index from {}", load_path.display())
                    })?;
                    let load_time = load_start.elapsed();
                    println!(
                        "Loaded index in {:.2?} ({} vectors, metric: {:?})",
                        load_time,
                        loaded.len(),
                        loaded.metric()
                    );
                    Ok((IndexWrapper::Swift(loaded), load_time))
                },
                || {
                    let mut index = SwiftIndex::with_defaults(runtime.metric);
                    let build_start = Instant::now();
                    index.build(dataset.clone())?;
                    let build_time = build_start.elapsed();
                    Ok((IndexWrapper::Swift(index), build_time))
                },
                |idx, path| {
                    if let IndexWrapper::Swift(inner) = idx {
                        inner.save(path).with_context(|| {
                            format!("failed to save index to {}", path.display())
                        })?;
                        println!(
                            "Saved SWIFT index to {} ({} vectors)",
                            path.display(),
                            inner.len()
                        );
                    }
                    Ok(())
                },
                false, // Not exhaustive (uses LSH bucketing)
                &dataset,
                &queries,
                &runtime,
                &cli,
            )?;
        }
        IndexType::Prism => {
            run_benchmark(
                "PRISM",
                |load_path| {
                    println!("Loading PRISM index from {}...", load_path.display());
                    let load_start = Instant::now();
                    let loaded = PrismIndex::load(load_path).with_context(|| {
                        format!("failed to load index from {}", load_path.display())
                    })?;
                    let load_time = load_start.elapsed();
                    println!(
                        "Loaded index in {:.2?} ({} vectors, metric: {:?})",
                        load_time,
                        loaded.len(),
                        loaded.metric()
                    );
                    Ok((IndexWrapper::Prism(loaded), load_time))
                },
                || {
                    let mut index = PrismIndex::with_defaults(runtime.metric);
                    let build_start = Instant::now();
                    index.build(dataset.clone())?;
                    let build_time = build_start.elapsed();
                    Ok((IndexWrapper::Prism(index), build_time))
                },
                |idx, path| {
                    if let IndexWrapper::Prism(inner) = idx {
                        inner.save(path).with_context(|| {
                            format!("failed to save index to {}", path.display())
                        })?;
                        println!(
                            "Saved PRISM index to {} ({} vectors)",
                            path.display(),
                            inner.len()
                        );
                    }
                    Ok(())
                },
                false, // Not exhaustive (uses HNSW underneath with session optimization)
                &dataset,
                &queries,
                &runtime,
                &cli,
            )?;
        }
        IndexType::Nexus => {
            run_benchmark(
                "NEXUS",
                |load_path| {
                    println!("Loading NEXUS index from {}...", load_path.display());
                    let load_start = Instant::now();
                    let loaded = NexusIndex::load(load_path).with_context(|| {
                        format!("failed to load index from {}", load_path.display())
                    })?;
                    let load_time = load_start.elapsed();
                    println!(
                        "Loaded index in {:.2?} ({} vectors, metric: {:?})",
                        load_time,
                        loaded.len(),
                        loaded.metric()
                    );
                    Ok((IndexWrapper::Nexus(loaded), load_time))
                },
                || {
                    let mut index = NexusIndex::with_defaults(runtime.metric);
                    let build_start = Instant::now();
                    index.build(dataset.clone())?;
                    let build_time = build_start.elapsed();
                    Ok((IndexWrapper::Nexus(index), build_time))
                },
                |idx, path| {
                    if let IndexWrapper::Nexus(inner) = idx {
                        inner.save(path).with_context(|| {
                            format!("failed to save index to {}", path.display())
                        })?;
                        println!(
                            "Saved NEXUS index to {} ({} vectors)",
                            path.display(),
                            inner.len()
                        );
                    }
                    Ok(())
                },
                false, // Not exhaustive (uses spectral shortcuts)
                &dataset,
                &queries,
                &runtime,
                &cli,
            )?;
        }
        IndexType::Fusion => {
            run_benchmark(
                "FUSION",
                |load_path| {
                    println!("Loading FUSION index from {}...", load_path.display());
                    let load_start = Instant::now();
                    let loaded = FusionIndex::load(load_path).with_context(|| {
                        format!("failed to load index from {}", load_path.display())
                    })?;
                    let load_time = load_start.elapsed();
                    println!(
                        "Loaded index in {:.2?} ({} vectors, metric: {:?})",
                        load_time,
                        loaded.len(),
                        loaded.metric()
                    );
                    Ok((IndexWrapper::Fusion(loaded), load_time))
                },
                || {
                    let mut index = FusionIndex::with_defaults(runtime.metric);
                    let build_start = Instant::now();
                    index.build(dataset.clone())?;
                    let build_time = build_start.elapsed();
                    Ok((IndexWrapper::Fusion(index), build_time))
                },
                |idx, path| {
                    if let IndexWrapper::Fusion(inner) = idx {
                        inner.save(path).with_context(|| {
                            format!("failed to save index to {}", path.display())
                        })?;
                        println!(
                            "Saved FUSION index to {} ({} vectors)",
                            path.display(),
                            inner.len()
                        );
                    }
                    Ok(())
                },
                false, // Not exhaustive (uses LSH bucketing + mini-graphs)
                &dataset,
                &queries,
                &runtime,
                &cli,
            )?;
        }
        IndexType::Vortex => {
            run_benchmark(
                "VORTEX",
                |load_path| {
                    println!("Loading VORTEX index from {}...", load_path.display());
                    let load_start = Instant::now();
                    let loaded = VortexIndex::load(load_path).with_context(|| {
                        format!("failed to load index from {}", load_path.display())
                    })?;
                    let load_time = load_start.elapsed();
                    println!(
                        "Loaded index in {:.2?} ({} vectors, metric: {:?})",
                        load_time,
                        loaded.len(),
                        loaded.metric()
                    );
                    Ok((IndexWrapper::Vortex(loaded), load_time))
                },
                || {
                    let mut index = VortexIndex::with_defaults(runtime.metric);
                    let build_start = Instant::now();
                    index.build(dataset.iter().cloned())?;
                    let build_time = build_start.elapsed();
                    Ok((IndexWrapper::Vortex(index), build_time))
                },
                |idx, path| {
                    if let IndexWrapper::Vortex(inner) = idx {
                        inner.save(path).with_context(|| {
                            format!("failed to save index to {}", path.display())
                        })?;
                        println!(
                            "Saved VORTEX index to {} ({} vectors)",
                            path.display(),
                            inner.len()
                        );
                    }
                    Ok(())
                },
                false,
                &dataset,
                &queries,
                &runtime,
                &cli,
            )?;
        }
        IndexType::Armi => {
            run_benchmark(
                "ARMI",
                |load_path| {
                    println!("Loading ARMI index from {}...", load_path.display());
                    let load_start = Instant::now();
                    let loaded = ArmiIndex::load(load_path).with_context(|| {
                        format!("failed to load index from {}", load_path.display())
                    })?;
                    let load_time = load_start.elapsed();
                    println!(
                        "Loaded index in {:.2?} ({} vectors, metric: {:?})",
                        load_time,
                        loaded.len(),
                        loaded.metric()
                    );
                    Ok((IndexWrapper::Armi(loaded), load_time))
                },
                || {
                    let mut index = ArmiIndex::with_defaults(runtime.metric);
                    let build_start = Instant::now();
                    index.build(dataset.clone())?;
                    let build_time = build_start.elapsed();
                    Ok((IndexWrapper::Armi(index), build_time))
                },
                |idx, path| {
                    if let IndexWrapper::Armi(inner) = idx {
                        inner.save(path).with_context(|| {
                            format!("failed to save index to {}", path.display())
                        })?;
                        println!(
                            "Saved ARMI index to {} ({} vectors)",
                            path.display(),
                            inner.len()
                        );
                    }
                    Ok(())
                },
                false, // Not exhaustive (uses multi-modal graph with adaptive tuning)
                &dataset,
                &queries,
                &runtime,
                &cli,
            )?;
        }
        IndexType::Apex => {
            run_benchmark(
                "APEX",
                |load_path| {
                    println!("Loading APEX index from {}...", load_path.display());
                    let load_start = Instant::now();
                    let loaded = ApexIndex::load(load_path).with_context(|| {
                        format!("failed to load index from {}", load_path.display())
                    })?;
                    let load_time = load_start.elapsed();
                    println!(
                        "Loaded index in {:.2?} ({} vectors, metric: {:?})",
                        load_time,
                        loaded.len(),
                        loaded.metric()
                    );
                    Ok((IndexWrapper::Apex(loaded), load_time))
                },
                || {
                    let mut index = ApexIndex::with_defaults(runtime.metric);
                    let build_start = Instant::now();
                    index.build(dataset.clone())?;
                    let build_time = build_start.elapsed();
                    Ok((IndexWrapper::Apex(index), build_time))
                },
                |idx, path| {
                    if let IndexWrapper::Apex(inner) = idx {
                        inner.save(path).with_context(|| {
                            format!("failed to save index to {}", path.display())
                        })?;
                        println!(
                            "Saved APEX index to {} ({} vectors)",
                            path.display(),
                            inner.len()
                        );
                    }
                    Ok(())
                },
                false, // Not exhaustive (uses multi-modal graph with adaptive tuning)
                &dataset,
                &queries,
                &runtime,
                &cli,
            )?;
        }
        IndexType::Synthesis => {
            run_benchmark(
                "SYNTHESIS",
                |load_path| {
                    println!("Loading SYNTHESIS index from {}...", load_path.display());
                    let load_start = Instant::now();
                    let loaded = SynthesisIndex::load(load_path).with_context(|| {
                        format!("failed to load index from {}", load_path.display())
                    })?;
                    let load_time = load_start.elapsed();
                    println!(
                        "Loaded index in {:.2?} ({} vectors, metric: {:?})",
                        load_time,
                        loaded.len(),
                        loaded.metric()
                    );
                    Ok((IndexWrapper::Synthesis(loaded), load_time))
                },
                || {
                    let mut index = SynthesisIndex::with_defaults(runtime.metric);
                    let build_start = Instant::now();
                    index.build(dataset.clone())?;
                    let build_time = build_start.elapsed();
                    Ok((IndexWrapper::Synthesis(index), build_time))
                },
                |idx, path| {
                    if let IndexWrapper::Synthesis(inner) = idx {
                        inner.save(path).with_context(|| {
                            format!("failed to save index to {}", path.display())
                        })?;
                        println!(
                            "Saved SYNTHESIS index to {} ({} vectors)",
                            path.display(),
                            inner.len()
                        );
                    }
                    Ok(())
                },
                false, // Not exhaustive (uses LSH + hierarchical graph with adaptive tuning)
                &dataset,
                &queries,
                &runtime,
                &cli,
            )?;
        }
    }

    // Note: report_json is handled in print_results if needed
    // We calculate stats there to have access to search time

    Ok(())
}

fn print_results(
    index: &IndexWrapper,
    first_result: &Option<Vec<ScoredPoint>>,
    build_time: Duration,
    total_search_time: u128,
    runtime: &RuntimeConfig,
    cli: &Cli,
    recall_metrics: (f64, f64, f64),
) -> Result<()> {
    if let Some(first) = first_result {
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

    let index_type_str = match index {
        IndexWrapper::Linear(_) => "linear",
        IndexWrapper::Hnsw(_) => "hnsw",
        IndexWrapper::Ivf(_) => "ivf",
        IndexWrapper::Pq(_) => "pq",
        IndexWrapper::Lim(_) => "lim",
        IndexWrapper::Hybrid(_) => "hybrid",
        IndexWrapper::Seer(_) => "seer",
        IndexWrapper::Swift(_) => "swift",
        IndexWrapper::Prism(_) => "prism",
        IndexWrapper::Nexus(_) => "nexus",
        IndexWrapper::Fusion(_) => "fusion",
        IndexWrapper::Vortex(_) => "vortex",
        IndexWrapper::Armi(_) => "armi",
        IndexWrapper::Apex(_) => "apex",
        IndexWrapper::Synthesis(_) => "synthesis",
    };

    let (avg_recall, min_recall, max_recall) = recall_metrics;

    println!(
        "Build: {:.2?} | Avg search: {:.2} us | QPS: {:.1} | Dataset: {:.2} MiB | Metric: {:?} | Index: {} | Points: {} | Queries: {} | Dim: {} | Limit: {}{}",
        stats.build_time,
        stats.avg_search_micros,
        stats.queries_per_second,
        stats.dataset_bytes as f64 / (1024.0 * 1024.0),
        runtime.metric,
        index_type_str,
        runtime.points,
        runtime.queries,
        runtime.dimension,
        runtime.limit,
        scenario_note
    );

    if !matches!(index, IndexWrapper::Linear(_)) {
        println!(
            "Recall@{}: avg={:.4} | min={:.4} | max={:.4}",
            runtime.limit, avg_recall, min_recall, max_recall
        );
    }

    if let Some(path) = cli.report_json.as_deref() {
        write_report(path, runtime, &stats, recall_metrics)?;
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

fn write_report(
    path: &Path,
    runtime: &RuntimeConfig,
    stats: &BenchmarkStats,
    recall_metrics: (f64, f64, f64),
) -> Result<()> {
    #[derive(Serialize)]
    struct ReportPayload<'a> {
        config: ReportConfig<'a>,
        timings: ReportTimings,
        recall: ReportRecall,
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

    #[derive(Serialize)]
    struct ReportRecall {
        avg: f64,
        min: f64,
        max: f64,
        k: usize,
    }

    let (avg_recall, min_recall, max_recall) = recall_metrics;

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
        recall: ReportRecall {
            avg: avg_recall,
            min: min_recall,
            max: max_recall,
            k: runtime.limit,
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
    serde_json::to_writer_pretty(writer, &payload).context("failed to write test data JSON")?;
    Ok(())
}
