//! Inverted File Index (IVF) implementation.
//!
//! This is an IVF implementation that uses K-means clustering to partition
//! the vector space into clusters. During search, it probes the nearest
//! clusters to find approximate nearest neighbors.

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
pub enum IvfError {
    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("index is empty, cannot search")]
    EmptyIndex,
    #[error("index has not been built yet (no centroids)")]
    NotBuilt,
}

/// IVF index configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IvfConfig {
    /// Number of clusters (centroids)
    pub n_clusters: usize,
    /// Number of clusters to probe during search
    pub n_probe: usize,
    /// Maximum iterations for K-means
    pub max_iter: usize,
    /// Number of K-means initializations
    pub n_init: usize,
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            n_clusters: 100,
            n_probe: 10,
            max_iter: 20,
            n_init: 1,
        }
    }
}

/// Inverted File Index implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IvfIndex {
    metric: DistanceMetric,
    dimension: Option<usize>,
    config: IvfConfig,
    /// Cluster centroids
    centroids: Vec<Vector>,
    /// Map from cluster index to list of (vector_id, vector) pairs
    clusters: Vec<Vec<(usize, Vector)>>,
    /// Total number of vectors indexed
    vector_count: usize,
}

impl IvfIndex {
    /// Creates a new IVF index with the given metric and configuration
    pub fn new(metric: DistanceMetric, config: IvfConfig) -> Self {
        Self {
            metric,
            dimension: None,
            config,
            centroids: Vec::new(),
            clusters: Vec::new(),
            vector_count: 0,
        }
    }

    /// Creates a new IVF index with default configuration
    pub fn with_defaults(metric: DistanceMetric) -> Self {
        Self::new(metric, IvfConfig::default())
    }

    fn validate_dimension(&self, vector: &[f32]) -> Result<()> {
        if let Some(expected) = self.dimension {
            validate_dimension(Some(expected), vector.len()).map_err(|_| {
                IvfError::DimensionMismatch {
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

    /// Initializes centroids using K-means++ initialization
    fn initialize_centroids(&self, data: &[(usize, Vector)]) -> Vec<Vector> {
        if data.is_empty() {
            return Vec::new();
        }

        let mut rng = rand::thread_rng();
        let mut centroids = Vec::with_capacity(self.config.n_clusters);

        // First centroid: choose randomly
        if let Some((_, first_vector)) = data.choose(&mut rng) {
            centroids.push(first_vector.clone());
        }

        // K-means++ initialization: choose centroids that are far from existing ones
        while centroids.len() < self.config.n_clusters.min(data.len()) {
            let mut distances = Vec::with_capacity(data.len());
            let mut total_distance = 0.0;

            for (_, vector) in data {
                let min_dist = centroids
                    .iter()
                    .map(|centroid| distance(self.metric, vector, centroid).unwrap_or(f32::MAX))
                    .fold(f32::MAX, f32::min);
                distances.push(min_dist);
                total_distance += min_dist;
            }

            if total_distance == 0.0 {
                // All points are at the same location, pick randomly
                if let Some((_, vector)) = data.choose(&mut rng) {
                    centroids.push(vector.clone());
                }
                break;
            }

            // Choose next centroid with probability proportional to distance^2
            let mut cumulative = 0.0;
            let threshold = rng.gen::<f64>() * total_distance as f64;
            for (i, &dist) in distances.iter().enumerate() {
                cumulative += dist as f64;
                if cumulative >= threshold {
                    centroids.push(data[i].1.clone());
                    break;
                }
            }
        }

        centroids
    }

    /// Assigns vectors to their nearest centroids
    fn assign_to_clusters(
        &self,
        data: &[(usize, Vector)],
        centroids: &[Vector],
    ) -> Result<Vec<Vec<(usize, Vector)>>> {
        let mut clusters: Vec<Vec<(usize, Vector)>> = vec![Vec::new(); centroids.len()];

        for (id, vector) in data {
            let mut min_dist = f32::MAX;
            let mut nearest_cluster = 0;

            for (cluster_idx, centroid) in centroids.iter().enumerate() {
                let dist = distance(self.metric, vector, centroid)?;
                if dist < min_dist {
                    min_dist = dist;
                    nearest_cluster = cluster_idx;
                }
            }

            clusters[nearest_cluster].push((*id, vector.clone()));
        }

        Ok(clusters)
    }

    /// Updates centroids based on current cluster assignments
    fn update_centroids(&self, clusters: &[Vec<(usize, Vector)>], dimension: usize) -> Vec<Vector> {
        clusters
            .iter()
            .map(|cluster| {
                if cluster.is_empty() {
                    // If cluster is empty, keep the old centroid or generate a random one
                    vec![0.0; dimension]
                } else {
                    // Compute mean of all vectors in the cluster
                    let mut centroid = vec![0.0; dimension];
                    for (_, vector) in cluster {
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
                        }
                    }

                    centroid
                }
            })
            .collect()
    }

    /// Trains the IVF index using K-means clustering
    fn train(&mut self, data: &[(usize, Vector)]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }

        let dimension = data[0].1.len();
        self.dimension = Some(dimension);

        // Initialize centroids
        let mut centroids = self.initialize_centroids(data);

        // Run K-means iterations
        for _iteration in 0..self.config.max_iter {
            // Assign vectors to clusters
            let clusters = self.assign_to_clusters(data, &centroids)?;

            // Update centroids
            let new_centroids = self.update_centroids(&clusters, dimension);

            // Check for convergence (centroids haven't changed much)
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

        // Final assignment
        self.clusters = self.assign_to_clusters(data, &centroids)?;
        self.centroids = centroids;

        Ok(())
    }

    /// Finds the nearest clusters to a query vector
    fn find_nearest_clusters(&self, query: &Vector, n_probe: usize) -> Result<Vec<usize>> {
        let mut cluster_distances: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(idx, centroid)| {
                let dist = distance(self.metric, query, centroid).unwrap_or(f32::MAX);
                (idx, dist)
            })
            .collect();

        // Sort by distance and take top n_probe
        cluster_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        cluster_distances.truncate(n_probe.min(self.centroids.len()));

        Ok(cluster_distances.into_iter().map(|(idx, _)| idx).collect())
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

impl VectorIndex for IvfIndex {
    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn len(&self) -> usize {
        self.vector_count
    }

    fn build(&mut self, data: impl IntoIterator<Item = (usize, Vector)>) -> Result<()> {
        let data: Vec<(usize, Vector)> = data.into_iter().collect();

        if data.is_empty() {
            return Ok(());
        }

        // Train the index (K-means clustering)
        self.train(&data)?;

        // Count total vectors
        self.vector_count = data.len();

        Ok(())
    }

    fn insert(&mut self, id: usize, vector: Vector) -> Result<()> {
        self.validate_dimension(&vector)?;

        if self.dimension.is_none() {
            self.dimension = Some(vector.len());
            // Initialize with empty clusters and centroids
            // This will require training before use
            self.centroids = Vec::new();
            self.clusters = Vec::new();
        }

        // For incremental insertion, we need to find the nearest centroid
        if self.centroids.is_empty() {
            // If no centroids exist, we can't insert properly
            // In a production system, you might want to retrain or use a different strategy
            return Err(IvfError::NotBuilt.into());
        }

        // Find nearest centroid
        let mut min_dist = f32::MAX;
        let mut nearest_cluster = 0;

        for (cluster_idx, centroid) in self.centroids.iter().enumerate() {
            let dist = distance(self.metric, &vector, centroid)?;
            if dist < min_dist {
                min_dist = dist;
                nearest_cluster = cluster_idx;
            }
        }

        // Add to the nearest cluster
        if nearest_cluster >= self.clusters.len() {
            self.clusters.resize(nearest_cluster + 1, Vec::new());
        }
        self.clusters[nearest_cluster].push((id, vector));
        self.vector_count += 1;

        Ok(())
    }

    fn search(&self, query: &Vector, limit: usize) -> Result<Vec<ScoredPoint>> {
        ensure!(
            limit > 0,
            "limit must be greater than zero to execute a search"
        );
        ensure!(!self.centroids.is_empty(), IvfError::NotBuilt);
        ensure!(self.vector_count > 0, IvfError::EmptyIndex);
        self.validate_dimension(query)?;

        // Find nearest clusters to probe
        let cluster_indices = self.find_nearest_clusters(query, self.config.n_probe)?;

        // Search within the selected clusters
        let mut candidates = Vec::new();
        for cluster_idx in cluster_indices {
            if cluster_idx >= self.clusters.len() {
                continue;
            }

            for (id, vector) in &self.clusters[cluster_idx] {
                let dist = distance(self.metric, query, vector)?;
                candidates.push(ScoredPoint::new(*id, dist));
            }
        }

        // Sort by distance and take top limit
        candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        candidates.truncate(limit.min(candidates.len()));

        Ok(candidates)
    }

    fn delete(&mut self, id: usize) -> Result<bool> {
        let mut found = false;
        
        // Search through all clusters to find and remove the vector
        for cluster in &mut self.clusters {
            let initial_len = cluster.len();
            cluster.retain(|(entry_id, _)| *entry_id != id);
            
            if cluster.len() < initial_len {
                found = true;
                self.vector_count -= 1;
                break;
            }
        }
        
        Ok(found)
    }

    fn update(&mut self, id: usize, vector: Vector) -> Result<bool> {
        self.validate_dimension(&vector)?;
        
        // Find and update the vector in clusters
        for cluster in &mut self.clusters {
            if let Some((entry_id, entry_vector)) = cluster.iter_mut().find(|(eid, _)| *eid == id) {
                *entry_vector = vector;
                return Ok(true);
            }
        }
        
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_after_build_works() {
        let mut index = IvfIndex::with_defaults(DistanceMetric::Euclidean);

        // Build with initial data
        let data = vec![
            (0, vec![0.0, 0.0]),
            (1, vec![1.0, 0.0]),
            (2, vec![0.0, 1.0]),
            (3, vec![1.0, 1.0]),
        ];
        index.build(data).unwrap();

        // Insert additional vector
        index.insert(4, vec![0.5, 0.5]).unwrap();

        assert_eq!(index.len(), 5);
    }

    #[test]
    fn search_returns_expected_results() {
        let mut index = IvfIndex::new(
            DistanceMetric::Euclidean,
            IvfConfig {
                n_clusters: 2,
                n_probe: 2,
                max_iter: 10,
                n_init: 1,
            },
        );

        let data = vec![
            (0, vec![0.0, 0.0]),
            (1, vec![1.0, 0.0]),
            (2, vec![0.0, 1.0]),
            (3, vec![1.0, 1.0]),
            (4, vec![0.5, 0.5]),
        ];
        index.build(data).unwrap();

        let result = index.search(&vec![0.0, 0.0], 3).unwrap();
        assert_eq!(result[0].id, 0); // Closest should be the origin
        assert!(result[0].distance < 0.1);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn empty_index_search_fails() {
        let index = IvfIndex::with_defaults(DistanceMetric::Euclidean);
        assert!(index.search(&vec![0.0, 0.0], 1).is_err());
    }

    #[test]
    fn save_and_load_preserves_index() {
        use std::env::temp_dir;

        let mut original_index = IvfIndex::new(
            DistanceMetric::Cosine,
            IvfConfig {
                n_clusters: 5,
                n_probe: 3,
                max_iter: 10,
                n_init: 1,
            },
        );
        original_index
            .build(vec![
                (0, vec![1.0, 0.0]),
                (1, vec![0.0, 1.0]),
                (2, vec![1.0, 1.0]),
            ])
            .unwrap();

        let mut temp_path = temp_dir();
        temp_path.push(format!("test_ivf_index_{}.json", std::process::id()));

        // Save the index
        original_index.save(&temp_path).unwrap();

        // Load the index
        let loaded_index = IvfIndex::load(&temp_path).unwrap();

        // Verify the loaded index has the same properties
        assert_eq!(loaded_index.metric(), original_index.metric());
        assert_eq!(loaded_index.len(), original_index.len());
        assert_eq!(loaded_index.dimension(), original_index.dimension());

        // Verify search results are similar (might not be identical due to clustering)
        let query = vec![1.0, 0.0];
        let original_results = original_index.search(&query, 2).unwrap();
        let loaded_results = loaded_index.search(&query, 2).unwrap();
        assert_eq!(original_results.len(), loaded_results.len());

        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }
}
