use std::time::Instant;

use anyhow::Result;
use clap::Parser;
use index_core::{
    generate_query_set, generate_uniform_dataset, DistanceMetric, ScoredPoint, VectorIndex,
};
use index_linear::LinearIndex;

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
}

fn parse_metric(value: &str) -> Result<DistanceMetric, String> {
    match value.to_ascii_lowercase().as_str() {
        "euclidean" | "l2" => Ok(DistanceMetric::Euclidean),
        "cosine" => Ok(DistanceMetric::Cosine),
        other => Err(format!("unsupported metric '{other}'")),
    }
}

fn main() -> Result<()> {
    let args = Cli::parse();

    let dataset = generate_uniform_dataset(
        args.dimension,
        args.points,
        -1.0_f32..1.0_f32,
        args.seed,
    );
    let queries =
        generate_query_set(args.dimension, args.queries, -1.0_f32..1.0_f32, args.seed.wrapping_add(1));

    let mut index = LinearIndex::new(args.metric);
    let build_start = Instant::now();
    index.build(dataset)?;
    let build_time = build_start.elapsed();

    let mut total_search_time = 0u128;
    let mut results: Vec<Vec<ScoredPoint>> = Vec::with_capacity(args.queries);
    for query in &queries {
        let search_start = Instant::now();
        let result = index.search(query, args.limit)?;
        total_search_time += search_start.elapsed().as_micros();
        results.push(result);
    }

    if let Some(first) = results.first() {
        println!(
            "First query result ids: {:?}",
            first.iter().map(|point| point.id).collect::<Vec<_>>()
        );
    }

    let avg_search_micros = total_search_time as f64 / args.queries as f64;

    println!(
        "Build time: {:.2?} | Avg search: {:.2}µs | Metric: {:?} | Points: {} | Queries: {} | Dim: {} | Limit: {}",
        build_time,
        avg_search_micros,
        args.metric,
        args.points,
        args.queries,
        args.dimension,
        args.limit
    );

    Ok(())
}
