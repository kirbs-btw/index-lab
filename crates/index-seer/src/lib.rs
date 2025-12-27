//! SEER (Similarity Estimation via Efficient Routing) - Learned Locality Prediction Index
//!
//! This implements a novel index structure that uses learned models to predict
//! nearest neighbor relationships without computing full distances. The algorithm:
//!
//! 1. During build: Samples vector pairs and learns feature projections that correlate
//!    with neighborhood relationships
//! 2. During search: Uses the learned predictor to score candidates quickly, then
//!    computes exact distances only for high-scoring candidates
//!
//! Research Gap Addressed: Gap 3A - Learned Index Structures

use anyhow::{ensure, Result};
use index_core::{
    distance, load_index, save_index, validate_dimension, DistanceMetric, ScoredPoint, Vector,
    VectorIndex,
};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SeerError {
    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("index is empty, cannot search")]
    EmptyIndex,
}

/// SEER index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeerConfig {
    /// Number of random projections for locality scoring
    pub n_projections: usize,
    /// Number of samples to use for learning projections
    pub n_samples: usize,
    /// Threshold for candidate selection (0.0 to 1.0)
    /// Higher = more selective, fewer candidates, faster but potentially lower recall
    pub candidate_threshold: f32,
    /// Minimum candidates to always consider (regardless of threshold)
    pub min_candidates: usize,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for SeerConfig {
    fn default() -> Self {
        Self {
            n_projections: 16,        // Number of random projection features
            n_samples: 1000,          // Samples for learning
            candidate_threshold: 0.3, // Select top 30% as candidates
            min_candidates: 50,       // Always consider at least 50
            seed: 42,
        }
    }
}

/// Locality predictor using random projections
///
/// This is a lightweight model that learns to predict whether two vectors
/// are likely to be neighbors based on their feature projections.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LocalityPredictor {
    /// Random projection vectors (n_projections x dimension)
    projections: Vec<Vector>,
    /// Learned weights for combining projection differences
    weights: Vec<f32>,
    /// Whether the predictor has been trained
    is_trained: bool,
}

impl LocalityPredictor {
    fn new(dimension: usize, n_projections: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        // Generate random projection vectors (unit normalized)
        let projections: Vec<Vector> = (0..n_projections)
            .map(|_| {
                let mut proj: Vector = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();
                // Normalize
                let norm: f32 = proj.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in &mut proj {
                        *v /= norm;
                    }
                }
                proj
            })
            .collect();

        // Initialize weights uniformly
        let weights = vec![1.0 / n_projections as f32; n_projections];

        Self {
            projections,
            weights,
            is_trained: false,
        }
    }

    /// Projects a vector onto the random projections
    fn project(&self, vector: &[f32]) -> Vec<f32> {
        self.projections
            .iter()
            .map(|proj| proj.iter().zip(vector.iter()).map(|(p, v)| p * v).sum())
            .collect()
    }

    /// Computes a locality score between two vectors (0 = likely far, 1 = likely near)
    fn score(&self, query: &[f32], candidate: &[f32]) -> f32 {
        let query_proj = self.project(query);
        let candidate_proj = self.project(candidate);

        // Compute weighted similarity of projections
        let mut score = 0.0f32;
        for i in 0..self.projections.len() {
            // Use absolute difference - smaller difference = higher similarity
            let diff = (query_proj[i] - candidate_proj[i]).abs();
            // Convert to similarity (exponential decay)
            let similarity = (-diff).exp();
            score += self.weights[i] * similarity;
        }

        score
    }

    /// Trains the predictor by adjusting weights based on true neighbor relationships
    fn train(
        &mut self,
        vectors: &[(usize, Vector)],
        metric: DistanceMetric,
        n_samples: usize,
        seed: u64,
    ) {
        if vectors.len() < 10 {
            self.is_trained = true;
            return; // Not enough data to train
        }

        let mut rng = StdRng::seed_from_u64(seed);
        let n = vectors.len();

        // Sample pairs and classify as neighbors vs non-neighbors
        let mut projection_correlations = vec![0.0f32; self.projections.len()];
        let mut sample_count = 0;

        for _ in 0..n_samples.min(n * n / 4) {
            let i = rng.gen_range(0..n);
            let j = rng.gen_range(0..n);
            if i == j {
                continue;
            }

            let (_, ref vi) = vectors[i];
            let (_, ref vj) = vectors[j];

            // Compute true distance
            let true_dist = distance(metric, vi, vj).unwrap_or(f32::MAX);

            // Compute projection features
            let pi = self.project(vi);
            let pj = self.project(vj);

            // For each projection, check if small projection difference correlates with small distance
            for k in 0..self.projections.len() {
                let proj_diff = (pi[k] - pj[k]).abs();
                // Correlation: negative if small proj_diff predicts small true_dist
                // We want projections where small differences indicate small distances
                // Use simple heuristic: reward if both are small or both are large
                let normalized_dist = true_dist / (true_dist + 1.0); // 0 to 1
                let normalized_proj_diff = proj_diff / (proj_diff + 1.0);

                // Agreement score: high if both indicate "near" or both indicate "far"
                let agreement = 1.0 - (normalized_dist - normalized_proj_diff).abs();
                projection_correlations[k] += agreement;
            }
            sample_count += 1;
        }

        // Normalize correlations to get weights
        if sample_count > 0 {
            let total: f32 = projection_correlations.iter().sum();
            if total > 0.0 {
                self.weights = projection_correlations.iter().map(|c| c / total).collect();
            }
        }

        self.is_trained = true;
    }
}

/// Stored vector entry
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VectorEntry {
    id: usize,
    vector: Vector,
}

/// SEER index implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeerIndex {
    metric: DistanceMetric,
    dimension: Option<usize>,
    config: SeerConfig,
    /// All indexed vectors
    vectors: Vec<VectorEntry>,
    /// Locality predictor
    predictor: Option<LocalityPredictor>,
    /// Whether build has been called (predictor trained)
    is_built: bool,
}

impl SeerIndex {
    /// Creates a new SEER index with the given metric and configuration
    pub fn new(metric: DistanceMetric, config: SeerConfig) -> Self {
        Self {
            metric,
            dimension: None,
            config,
            vectors: Vec::new(),
            predictor: None,
            is_built: false,
        }
    }

    /// Creates a new SEER index with default configuration
    pub fn with_defaults(metric: DistanceMetric) -> Self {
        Self::new(metric, SeerConfig::default())
    }

    fn validate_dimension(&self, vector: &[f32]) -> Result<()> {
        if let Some(expected) = self.dimension {
            validate_dimension(Some(expected), vector.len()).map_err(|_| {
                SeerError::DimensionMismatch {
                    expected,
                    actual: vector.len(),
                }
            })?;
        }
        Ok(())
    }

    /// Returns the dimensionality tracked by the index
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Trains the locality predictor on the current dataset
    fn train_predictor(&mut self) {
        if let Some(dim) = self.dimension {
            let mut predictor =
                LocalityPredictor::new(dim, self.config.n_projections, self.config.seed);

            let vectors_ref: Vec<(usize, Vector)> = self
                .vectors
                .iter()
                .map(|e| (e.id, e.vector.clone()))
                .collect();

            predictor.train(
                &vectors_ref,
                self.metric,
                self.config.n_samples,
                self.config.seed,
            );

            self.predictor = Some(predictor);
        }
    }

    /// Saves the index to a JSON file
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        save_index(self, path)
    }

    /// Loads an index from a JSON file
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        load_index(path)
    }
}

impl VectorIndex for SeerIndex {
    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn build(&mut self, data: impl IntoIterator<Item = (usize, Vector)>) -> Result<()> {
        // Insert all vectors first
        for (id, vector) in data {
            self.insert(id, vector)?;
        }

        // Train the predictor
        self.train_predictor();
        self.is_built = true;

        Ok(())
    }

    fn insert(&mut self, id: usize, vector: Vector) -> Result<()> {
        self.validate_dimension(&vector)?;

        if self.dimension.is_none() {
            self.dimension = Some(vector.len());
        }

        self.vectors.push(VectorEntry { id, vector });

        // If we've already built, mark as needing rebuild for optimal performance
        // (In practice, we'd want incremental updates, but for now just note it)
        if self.is_built && self.vectors.len().is_multiple_of(1000) {
            // Periodically retrain on new data
            self.train_predictor();
        }

        Ok(())
    }

    fn search(&self, query: &Vector, limit: usize) -> Result<Vec<ScoredPoint>> {
        ensure!(limit > 0, "limit must be greater than zero");
        ensure!(!self.vectors.is_empty(), SeerError::EmptyIndex);
        self.validate_dimension(query)?;

        // If we have a predictor, use it to filter candidates
        let candidates: Vec<&VectorEntry> = if let Some(ref predictor) = self.predictor {
            // Score all vectors with the predictor
            let mut scored: Vec<(&VectorEntry, f32)> = self
                .vectors
                .iter()
                .map(|entry| {
                    let score = predictor.score(query, &entry.vector);
                    (entry, score)
                })
                .collect();

            // Sort by score (descending - higher score = more likely neighbor)
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Select top candidates based on threshold
            let threshold_idx =
                ((1.0 - self.config.candidate_threshold) * scored.len() as f32) as usize;
            let n_candidates = threshold_idx
                .max(self.config.min_candidates)
                .max(limit)
                .min(scored.len());

            scored
                .into_iter()
                .take(n_candidates)
                .map(|(entry, _)| entry)
                .collect()
        } else {
            // No predictor, use all vectors (fallback to linear scan)
            self.vectors.iter().collect()
        };

        // Compute exact distances for candidates
        let mut results: Vec<ScoredPoint> = candidates
            .iter()
            .filter_map(|entry| {
                distance(self.metric, query, &entry.vector)
                    .ok()
                    .map(|d| ScoredPoint::new(entry.id, d))
            })
            .collect();

        // Sort by distance and take top limit
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.truncate(limit);

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_index_has_zero_length() {
        let index = SeerIndex::with_defaults(DistanceMetric::Euclidean);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn insert_sets_dimension() {
        let mut index = SeerIndex::with_defaults(DistanceMetric::Euclidean);
        index.insert(0, vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(index.dimension(), Some(3));
    }

    #[test]
    fn insert_rejects_dimension_mismatch() {
        let mut index = SeerIndex::with_defaults(DistanceMetric::Euclidean);
        index.insert(0, vec![1.0, 2.0, 3.0]).unwrap();
        let result = index.insert(1, vec![1.0, 2.0]);
        assert!(result.is_err());
    }

    #[test]
    fn build_and_search_returns_results() {
        let mut index = SeerIndex::with_defaults(DistanceMetric::Euclidean);
        let data = vec![
            (0, vec![0.0, 0.0]),
            (1, vec![1.0, 0.0]),
            (2, vec![0.0, 1.0]),
            (3, vec![10.0, 10.0]),
        ];
        index.build(data).unwrap();

        let result = index.search(&vec![0.0, 0.0], 2).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, 0); // Exact match first
    }

    #[test]
    fn predictor_is_trained_after_build() {
        let mut index = SeerIndex::with_defaults(DistanceMetric::Euclidean);
        let data: Vec<(usize, Vector)> = (0..100)
            .map(|i| (i, vec![i as f32 / 100.0, (100 - i) as f32 / 100.0]))
            .collect();
        index.build(data).unwrap();

        assert!(index.is_built);
        assert!(index.predictor.is_some());
    }

    #[test]
    fn search_with_large_dataset() {
        let mut index = SeerIndex::new(
            DistanceMetric::Euclidean,
            SeerConfig {
                n_projections: 8,
                n_samples: 500,
                candidate_threshold: 0.5, // Select top 50% as candidates
                min_candidates: 20,
                seed: 42,
            },
        );

        // Create a dataset with two clusters
        let mut data: Vec<(usize, Vector)> = Vec::new();
        for i in 0..500 {
            // Cluster 1: around origin
            data.push((
                i * 2,
                vec![i as f32 / 1000.0 + 0.1, i as f32 / 1000.0 + 0.1],
            ));
            // Cluster 2: far away
            data.push((
                i * 2 + 1,
                vec![10.0 + i as f32 / 1000.0, 10.0 + i as f32 / 1000.0],
            ));
        }
        index.build(data).unwrap();

        // Search near cluster 1
        let result = index.search(&vec![0.1, 0.1], 5).unwrap();
        assert_eq!(result.len(), 5);

        // All top results should be from cluster 1 (even IDs)
        for point in &result {
            assert!(
                point.id % 2 == 0,
                "Expected cluster 1 vectors (even IDs), got {}",
                point.id
            );
        }
    }

    #[test]
    fn cosine_metric_works() {
        let mut index = SeerIndex::with_defaults(DistanceMetric::Cosine);
        let data = vec![
            (0, vec![1.0, 0.0]),
            (1, vec![0.707, 0.707]),
            (2, vec![0.0, 1.0]),
        ];
        index.build(data).unwrap();

        let result = index.search(&vec![1.0, 0.0], 2).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, 0); // Exact match
    }

    #[test]
    fn save_and_load_roundtrip() {
        let mut index = SeerIndex::with_defaults(DistanceMetric::Euclidean);
        index
            .build(vec![(0, vec![1.0, 2.0]), (1, vec![3.0, 4.0])])
            .unwrap();

        let temp_path = std::env::temp_dir().join("seer_test_index.json");
        index.save(&temp_path).unwrap();

        let loaded = SeerIndex::load(&temp_path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.dimension(), Some(2));
        assert!(loaded.predictor.is_some());

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }
}
