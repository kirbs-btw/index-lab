//! Product Quantization (PQ) index implementation.
//!
//! This is a PQ implementation that compresses vectors by splitting them into
//! subvectors and quantizing each subvector using a codebook. During search,
//! approximate distances are computed using lookup tables for efficiency.

#![allow(unknown_lints)]
#![allow(clippy::manual_is_multiple_of)]

use anyhow::{ensure, Result};
use index_core::{
    distance, load_index, save_index, validate_dimension, DistanceMetric, ScoredPoint, Vector,
    VectorIndex,
};
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PqError {
    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("index is empty, cannot search")]
    EmptyIndex,
    #[error("index has not been trained yet (no codebooks)")]
    NotTrained,
    #[error("dimension {dimension} is not divisible by number of subvectors {m}")]
    DimensionNotDivisible { dimension: usize, m: usize },
}

/// PQ index configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PqConfig {
    /// Number of subvectors (M)
    pub m: usize,
    /// Number of centroids per subvector codebook (K, typically 256 for 1 byte codes)
    pub k: usize,
    /// Maximum iterations for K-means training
    pub max_iter: usize,
}

impl Default for PqConfig {
    fn default() -> Self {
        Self {
            m: 8,   // 8 subvectors
            k: 256, // 256 centroids per subvector (1 byte codes)
            max_iter: 20,
        }
    }
}

/// Product Quantization index implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PqIndex {
    metric: DistanceMetric,
    dimension: Option<usize>,
    config: PqConfig,
    /// Codebooks: one per subvector, each containing k centroids
    /// codebooks[i] is the codebook for subvector i
    codebooks: Vec<Vec<Vector>>,
    /// Encoded vectors: each vector is represented as M codes (one per subvector)
    /// codes[i] is a Vec<u8> of length M representing the encoded vector for entry i
    codes: Vec<Vec<u8>>,
    /// Vector IDs corresponding to each encoded vector
    ids: Vec<usize>,
    /// Subvector dimension (dimension / m)
    subvector_dim: Option<usize>,
}

impl PqIndex {
    /// Creates a new PQ index with the given metric and configuration
    pub fn new(metric: DistanceMetric, config: PqConfig) -> Self {
        Self {
            metric,
            dimension: None,
            config,
            codebooks: Vec::new(),
            codes: Vec::new(),
            ids: Vec::new(),
            subvector_dim: None,
        }
    }

    /// Creates a new PQ index with default configuration
    pub fn with_defaults(metric: DistanceMetric) -> Self {
        Self::new(metric, PqConfig::default())
    }

    fn validate_dimension(&self, vector: &[f32]) -> Result<()> {
        if let Some(expected) = self.dimension {
            validate_dimension(Some(expected), vector.len()).map_err(|_| {
                PqError::DimensionMismatch {
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

    /// Splits a vector into M subvectors
    fn split_vector<'a>(&self, vector: &'a [f32]) -> Result<Vec<&'a [f32]>> {
        let subvector_dim = self.subvector_dim.ok_or(PqError::NotTrained)?;
        let mut subvectors = Vec::with_capacity(self.config.m);

        for i in 0..self.config.m {
            let start = i * subvector_dim;
            let end = start + subvector_dim;
            subvectors.push(&vector[start..end]);
        }

        Ok(subvectors)
    }

    /// Trains codebooks for each subvector using K-means
    fn train_codebooks(&mut self, data: &[(usize, Vector)]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        let dimension = data[0].1.len();
        self.dimension = Some(dimension);

        // Validate dimension is divisible by m
        ensure!(
            dimension % self.config.m == 0,
            PqError::DimensionNotDivisible {
                dimension,
                m: self.config.m
            }
        );

        let subvector_dim = dimension / self.config.m;
        self.subvector_dim = Some(subvector_dim);

        // Extract subvectors for each position
        let mut subvector_data: Vec<Vec<Vector>> = vec![Vec::new(); self.config.m];

        for (_, vector) in data {
            for (i, subvec) in self.split_vector(vector)?.iter().enumerate() {
                subvector_data[i].push(subvec.to_vec());
            }
        }

        // Train codebook for each subvector position
        self.codebooks = Vec::with_capacity(self.config.m);
        for subvecs in subvector_data.iter() {
            let codebook = self.train_codebook(subvecs, subvector_dim)?;
            self.codebooks.push(codebook);
        }

        Ok(())
    }

    /// Trains a single codebook using K-means
    fn train_codebook(&self, data: &[Vector], dimension: usize) -> Result<Vec<Vector>> {
        if data.is_empty() {
            return Ok(vec![vec![0.0; dimension]; self.config.k]);
        }

        // Initialize centroids using K-means++
        let mut centroids = self.initialize_centroids(data, dimension);

        // Run K-means iterations
        for _iteration in 0..self.config.max_iter {
            // Assign vectors to nearest centroids
            let mut clusters: Vec<Vec<Vector>> = vec![Vec::new(); self.config.k];

            for vector in data {
                let mut min_dist = f32::MAX;
                let mut nearest_centroid = 0;

                for (centroid_idx, centroid) in centroids.iter().enumerate() {
                    let dist = distance(self.metric, vector, centroid)?;
                    if dist < min_dist {
                        min_dist = dist;
                        nearest_centroid = centroid_idx;
                    }
                }

                clusters[nearest_centroid].push(vector.clone());
            }

            // Update centroids
            let mut new_centroids = Vec::with_capacity(self.config.k);
            for (cluster_idx, cluster) in clusters.iter().enumerate() {
                if cluster.is_empty() {
                    // Keep old centroid if cluster is empty
                    if cluster_idx < centroids.len() {
                        new_centroids.push(centroids[cluster_idx].clone());
                    } else {
                        // Shouldn't happen, but duplicate a random existing centroid
                        let mut rng = rand::thread_rng();
                        if let Some(centroid) = centroids.choose(&mut rng) {
                            new_centroids.push(centroid.clone());
                        } else {
                            new_centroids.push(vec![0.0; dimension]);
                        }
                    }
                } else {
                    // Compute mean
                    let mut centroid = vec![0.0; dimension];
                    for vector in cluster {
                        for (i, &value) in vector.iter().enumerate() {
                            centroid[i] += value;
                        }
                    }
                    let count = cluster.len() as f32;
                    for value in &mut centroid {
                        *value /= count;
                    }

                    // Normalize for cosine distance if needed
                    if self.metric.requires_unit_vectors() {
                        let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if norm > 0.0 {
                            for value in &mut centroid {
                                *value /= norm;
                            }
                        } else {
                            // If somehow zero, create a unit vector
                            if dimension > 0 {
                                centroid[0] = 1.0;
                            }
                        }
                    }

                    new_centroids.push(centroid);
                }
            }

            // Check for convergence
            let mut converged = true;
            for (old, new) in centroids.iter().zip(new_centroids.iter()) {
                let dist = distance(self.metric, old, new)?;
                if dist > 1e-6 {
                    converged = false;
                    break;
                }
            }

            centroids = new_centroids;
            if converged {
                break;
            }
        }

        Ok(centroids)
    }

    /// Initializes centroids using K-means++ initialization
    fn initialize_centroids(&self, data: &[Vector], dimension: usize) -> Vec<Vector> {
        if data.is_empty() {
            // Return k zero vectors if no data
            return vec![vec![0.0; dimension]; self.config.k];
        }

        let mut rng = rand::thread_rng();
        let mut centroids = Vec::with_capacity(self.config.k);

        // First centroid: choose randomly
        if let Some(first_vector) = data.choose(&mut rng) {
            centroids.push(first_vector.clone());
        }

        // K-means++ initialization
        while centroids.len() < self.config.k.min(data.len()) {
            let mut distances = Vec::with_capacity(data.len());
            let mut total_distance = 0.0;

            for vector in data {
                let min_dist = centroids
                    .iter()
                    .map(|centroid| distance(self.metric, vector, centroid).unwrap_or(f32::MAX))
                    .fold(f32::MAX, f32::min);
                distances.push(min_dist);
                total_distance += min_dist;
            }

            if total_distance == 0.0 {
                // All points are at the same location, pick randomly or duplicate
                if let Some(vector) = data.choose(&mut rng) {
                    centroids.push(vector.clone());
                } else {
                    break;
                }
            } else {
                // Choose next centroid with probability proportional to distance^2
                let mut cumulative = 0.0;
                let threshold = rng.gen::<f64>() * total_distance as f64;
                for (i, &dist) in distances.iter().enumerate() {
                    cumulative += dist as f64;
                    if cumulative >= threshold {
                        centroids.push(data[i].clone());
                        break;
                    }
                }
            }
        }

        // If we still don't have k centroids (e.g., fewer data points than k), duplicate existing ones
        while centroids.len() < self.config.k {
            if let Some(centroid) = centroids.choose(&mut rng) {
                let mut new_centroid = centroid.clone();
                // For cosine distance, ensure it's normalized
                if self.metric.requires_unit_vectors() {
                    let norm: f32 = new_centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for value in &mut new_centroid {
                            *value /= norm;
                        }
                    } else {
                        // If somehow zero, create a unit vector
                        if dimension > 0 {
                            new_centroid[0] = 1.0;
                        }
                    }
                }
                centroids.push(new_centroid);
            } else {
                // Fallback: create a unit vector for cosine, or zero for euclidean
                let mut fallback = vec![0.0; dimension];
                if self.metric.requires_unit_vectors() && dimension > 0 {
                    fallback[0] = 1.0; // Unit vector
                }
                centroids.push(fallback);
            }
        }

        centroids
    }

    /// Encodes a vector into PQ codes
    fn encode_vector(&self, vector: &Vector) -> Result<Vec<u8>> {
        ensure!(!self.codebooks.is_empty(), PqError::NotTrained);

        let subvectors = self.split_vector(vector)?;
        let mut codes = Vec::with_capacity(self.config.m);

        for (subvec_idx, subvec) in subvectors.iter().enumerate() {
            let codebook = &self.codebooks[subvec_idx];

            // Find nearest centroid
            let mut min_dist = f32::MAX;
            let mut nearest_code = 0u8;

            for (code, centroid) in codebook.iter().enumerate() {
                if code >= 256 {
                    break; // Can't encode more than 256 codes in u8
                }
                let dist = distance(self.metric, subvec, centroid)?;
                if dist < min_dist {
                    min_dist = dist;
                    nearest_code = code as u8;
                }
            }

            codes.push(nearest_code);
        }

        Ok(codes)
    }

    /// Computes approximate distance between query and encoded vector using lookup tables
    fn approximate_distance(&self, query: &Vector, codes: &[u8]) -> Result<f32> {
        ensure!(!self.codebooks.is_empty(), PqError::NotTrained);

        let subvectors = self.split_vector(query)?;
        let mut total_distance = 0.0;

        for (subvec_idx, subvec) in subvectors.iter().enumerate() {
            let code = codes[subvec_idx] as usize;
            let codebook = &self.codebooks[subvec_idx];

            if code < codebook.len() {
                let centroid = &codebook[code];
                let dist = distance(self.metric, subvec, centroid)?;
                total_distance += dist * dist; // Sum of squared distances
            }
        }

        // For Euclidean, take square root of sum of squared distances
        // For Cosine, the sum is already the distance (since cosine distance is computed per subvector)
        if matches!(self.metric, DistanceMetric::Euclidean) {
            Ok(total_distance.sqrt())
        } else {
            Ok(total_distance)
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

impl VectorIndex for PqIndex {
    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn len(&self) -> usize {
        self.codes.len()
    }

    fn build(&mut self, data: impl IntoIterator<Item = (usize, Vector)>) -> Result<()> {
        let data: Vec<(usize, Vector)> = data.into_iter().collect();

        if data.is_empty() {
            return Ok(());
        }

        // Train codebooks
        self.train_codebooks(&data)?;

        // Encode all vectors
        self.codes.clear();
        self.ids.clear();

        for (id, vector) in data {
            let codes = self.encode_vector(&vector)?;
            self.codes.push(codes);
            self.ids.push(id);
        }

        Ok(())
    }

    fn insert(&mut self, id: usize, vector: Vector) -> Result<()> {
        self.validate_dimension(&vector)?;

        if self.dimension.is_none() {
            self.dimension = Some(vector.len());
            // Need to train first
            return Err(PqError::NotTrained.into());
        }

        ensure!(!self.codebooks.is_empty(), PqError::NotTrained);

        // Encode the vector
        let codes = self.encode_vector(&vector)?;
        self.codes.push(codes);
        self.ids.push(id);

        Ok(())
    }

    fn search(&self, query: &Vector, limit: usize) -> Result<Vec<ScoredPoint>> {
        ensure!(
            limit > 0,
            "limit must be greater than zero to execute a search"
        );
        ensure!(!self.codebooks.is_empty(), PqError::NotTrained);
        ensure!(!self.codes.is_empty(), PqError::EmptyIndex);
        self.validate_dimension(query)?;

        // Compute approximate distances to all encoded vectors
        let mut candidates = Vec::with_capacity(self.codes.len());

        for (idx, codes) in self.codes.iter().enumerate() {
            let dist = self.approximate_distance(query, codes)?;
            candidates.push(ScoredPoint::new(self.ids[idx], dist));
        }

        // Sort by distance and take top limit
        candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        candidates.truncate(limit.min(candidates.len()));

        Ok(candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_after_build_works() {
        let mut index = PqIndex::new(
            DistanceMetric::Euclidean,
            PqConfig {
                m: 2,
                k: 4,
                max_iter: 10,
            },
        );

        // Build with initial data (dimension must be divisible by m)
        let data = vec![
            (0, vec![0.0, 0.0, 1.0, 1.0]),
            (1, vec![1.0, 0.0, 0.0, 1.0]),
            (2, vec![0.0, 1.0, 1.0, 0.0]),
            (3, vec![1.0, 1.0, 0.0, 0.0]),
        ];
        index.build(data).unwrap();

        // Insert additional vector
        index.insert(4, vec![0.5, 0.5, 0.5, 0.5]).unwrap();

        assert_eq!(index.len(), 5);
    }

    #[test]
    fn search_returns_expected_results() {
        let mut index = PqIndex::new(
            DistanceMetric::Euclidean,
            PqConfig {
                m: 2,
                k: 4,
                max_iter: 10,
            },
        );

        let data = vec![
            (0, vec![0.0, 0.0, 0.0, 0.0]),
            (1, vec![1.0, 0.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0, 0.0]),
            (3, vec![0.0, 0.0, 1.0, 0.0]),
            (4, vec![0.5, 0.5, 0.5, 0.5]),
        ];
        index.build(data).unwrap();

        let result = index.search(&vec![0.0, 0.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(result[0].id, 0); // Closest should be the origin
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn empty_index_search_fails() {
        let index = PqIndex::with_defaults(DistanceMetric::Euclidean);
        assert!(index.search(&vec![0.0, 0.0], 1).is_err());
    }

    #[test]
    fn dimension_must_be_divisible_by_m() {
        let mut index = PqIndex::new(
            DistanceMetric::Euclidean,
            PqConfig {
                m: 8,
                k: 256,
                max_iter: 10,
            },
        );

        // Try to build with dimension 10, which is not divisible by 8
        let data = vec![(0, vec![0.0; 10])];
        assert!(index.build(data).is_err());
    }

    #[test]
    fn save_and_load_preserves_index() {
        use std::env::temp_dir;

        let mut original_index = PqIndex::new(
            DistanceMetric::Euclidean, // Use Euclidean to avoid cosine zero vector issues
            PqConfig {
                m: 2,
                k: 4,
                max_iter: 5, // Fewer iterations for faster test
            },
        );
        original_index
            .build(vec![
                (0, vec![1.0, 0.0, 0.0, 1.0]),
                (1, vec![0.0, 1.0, 1.0, 0.0]),
                (2, vec![1.0, 1.0, 0.0, 0.0]),
                (3, vec![0.0, 0.0, 1.0, 1.0]), // Add one more to have at least k data points
            ])
            .unwrap();

        let mut temp_path = temp_dir();
        temp_path.push(format!("test_pq_index_{}.json", std::process::id()));

        // Save the index
        original_index.save(&temp_path).unwrap();

        // Load the index
        let loaded_index = PqIndex::load(&temp_path).unwrap();

        // Verify the loaded index has the same properties
        assert_eq!(loaded_index.metric(), original_index.metric());
        assert_eq!(loaded_index.len(), original_index.len());
        assert_eq!(loaded_index.dimension(), original_index.dimension());

        // Verify search results are similar
        let query = vec![1.0, 0.0, 0.0, 1.0];
        let original_results = original_index.search(&query, 2).unwrap();
        let loaded_results = loaded_index.search(&query, 2).unwrap();
        assert_eq!(original_results.len(), loaded_results.len());
        // Results should match since it's the same index
        assert_eq!(original_results[0].id, loaded_results[0].id);

        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }
}
