//! Shared building blocks for experimental vector indexing algorithms.
//! This crate defines common traits, utilities, and dataset helpers that other
//! experimental crates can build upon.

use anyhow::{ensure, Result};
use rand::{distributions::Uniform, prelude::*};
use serde::{Deserialize, Serialize};

/// Canonical vector representation used throughout the playground.
pub type Vector = Vec<f32>;

/// Distance functions supported by the playground.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance.
    Euclidean,
    /// Cosine distance (1 - cosine similarity).
    Cosine,
}

impl DistanceMetric {
    /// True if the metric requires normalized vectors for correct semantics.
    pub fn requires_unit_vectors(self) -> bool {
        matches!(self, DistanceMetric::Cosine)
    }
}

/// Result of a nearest-neighbour query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredPoint {
    pub id: usize,
    pub distance: f32,
}

impl ScoredPoint {
    pub fn new(id: usize, distance: f32) -> Self {
        Self { id, distance }
    }
}

/// Trait implemented by all vector index prototypes in this repository.
pub trait VectorIndex {
    /// Distance metric the index was instantiated with.
    fn metric(&self) -> DistanceMetric;

    /// Total number of vectors managed by the index.
    fn len(&self) -> usize;

    /// Convenience helper indicating whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Builds the index from an initial dataset.
    fn build(&mut self, data: impl IntoIterator<Item = (usize, Vector)>) -> Result<()> {
        for (id, vector) in data {
            self.insert(id, vector)?;
        }
        Ok(())
    }

    /// Inserts a single vector with its identifier.
    fn insert(&mut self, id: usize, vector: Vector) -> Result<()>;

    /// Queries the index for the `limit` nearest neighbours to `query`.
    fn search(&self, query: &Vector, limit: usize) -> Result<Vec<ScoredPoint>>;
}

/// Computes the distance between two vectors according to the desired metric.
pub fn distance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> Result<f32> {
    ensure!(
        a.len() == b.len(),
        "vector dimension mismatch: {} != {}",
        a.len(),
        b.len()
    );

    match metric {
        DistanceMetric::Euclidean => Ok(a
            .iter()
            .zip(b.iter())
            .map(|(lhs, rhs)| {
                let diff = lhs - rhs;
                diff * diff
            })
            .sum::<f32>()
            .sqrt()),
        DistanceMetric::Cosine => {
            let (dot, norm_a, norm_b) = a.iter().zip(b.iter()).fold(
                (0.0_f32, 0.0_f32, 0.0_f32),
                |(dot_acc, norm_a_acc, norm_b_acc), (lhs, rhs)| {
                    (
                        dot_acc + lhs * rhs,
                        norm_a_acc + lhs * lhs,
                        norm_b_acc + rhs * rhs,
                    )
                },
            );

            ensure!(norm_a > 0.0 && norm_b > 0.0, "cosine metric is undefined for zero vectors");
            let similarity = dot / (norm_a.sqrt() * norm_b.sqrt());
            Ok(1.0 - similarity)
        }
    }
}

/// Generates a deterministic set of dense vectors uniformly sampled within the given range.
pub fn generate_uniform_dataset(
    dimension: usize,
    count: usize,
    bounds: std::ops::Range<f32>,
    seed: u64,
) -> Vec<(usize, Vector)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let distribution = Uniform::from(bounds);

    (0..count)
        .map(|id| {
            let vector = (0..dimension)
                .map(|_| rng.sample(distribution))
                .collect::<Vec<_>>();
            (id, vector)
        })
        .collect()
}

/// Produces a list of query vectors distinct from the dataset to benchmark search quality.
pub fn generate_query_set(
    dimension: usize,
    count: usize,
    bounds: std::ops::Range<f32>,
    seed: u64,
) -> Vec<Vector> {
    let mut rng = StdRng::seed_from_u64(seed);
    let distribution = Uniform::from(bounds);

    (0..count)
        .map(|_| (0..dimension).map(|_| rng.sample(distribution)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn euclidean_distance_matches_manual_result() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = distance(DistanceMetric::Euclidean, &a, &b).unwrap();
        assert!((dist - 5.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_distance_matches_manual_result() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let dist = distance(DistanceMetric::Cosine, &a, &b).unwrap();
        assert!((dist - 1.0).abs() < 1e-5);
    }

    #[test]
    fn dataset_generation_is_deterministic() {
        let first = generate_uniform_dataset(4, 2, -1.0..1.0, 42);
        let second = generate_uniform_dataset(4, 2, -1.0..1.0, 42);
        assert_eq!(first, second);
    }
}
