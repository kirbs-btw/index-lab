//! Shared building blocks for experimental vector indexing algorithms.
//! This crate defines common traits, utilities, and dataset helpers that other
//! experimental crates can build upon.

use anyhow::{ensure, Context, Result};
use rand::{distributions::Uniform, prelude::*};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

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

    /// Deletes a vector by its identifier.
    /// Returns `true` if the vector was found and deleted, `false` if not found.
    /// Default implementation returns an error indicating deletion is not supported.
    fn delete(&mut self, _id: usize) -> Result<bool> {
        Err(anyhow::anyhow!("delete not implemented for this index type"))
    }

    /// Updates an existing vector by its identifier.
    /// Returns `true` if the vector was found and updated, `false` if not found.
    /// Default implementation returns an error indicating update is not supported.
    fn update(&mut self, _id: usize, _vector: Vector) -> Result<bool> {
        Err(anyhow::anyhow!("update not implemented for this index type"))
    }
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

            ensure!(
                norm_a > 0.0 && norm_b > 0.0,
                "cosine metric is undefined for zero vectors"
            );
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

/// Validates that a vector has the expected dimension.
pub fn validate_dimension(expected: Option<usize>, actual: usize) -> Result<()> {
    if let Some(expected_dim) = expected {
        ensure!(
            actual == expected_dim,
            "vector dimension mismatch: expected {}, got {}",
            expected_dim,
            actual
        );
    }
    Ok(())
}

/// Macro to implement the common `validate_dimension` method for index types.
/// This eliminates duplication across all index implementations.
///
/// Usage:
/// ```ignore
/// impl_validate_dimension!(MyIndex, MyError);
/// ```
#[macro_export]
macro_rules! impl_validate_dimension {
    ($index_type:ty, $error_type:ty) => {
        impl $index_type {
            fn validate_dimension(&self, vector: &[f32]) -> $crate::Result<()> {
                if let Some(expected) = self.dimension {
                    $crate::validate_dimension(Some(expected), vector.len()).map_err(|_| {
                        <$error_type>::DimensionMismatch {
                            expected,
                            actual: vector.len(),
                        }
                    })?;
                }
                Ok(())
            }
        }
    };
}

/// Saves a serializable index to a JSON file.
pub fn save_index<T: Serialize>(index: &T, path: impl AsRef<Path>) -> Result<()> {
    let file = File::create(path.as_ref())
        .with_context(|| format!("failed to create index file at {}", path.as_ref().display()))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, index).context("failed to serialize index to JSON")?;
    Ok(())
}

/// Loads a deserializable index from a JSON file.
pub fn load_index<T: for<'de> Deserialize<'de>>(path: impl AsRef<Path>) -> Result<T> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("failed to open index file at {}", path.as_ref().display()))?;
    let reader = BufReader::new(file);
    let index = serde_json::from_reader(reader).with_context(|| {
        format!(
            "failed to deserialize index from {}",
            path.as_ref().display()
        )
    })?;
    Ok(index)
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
