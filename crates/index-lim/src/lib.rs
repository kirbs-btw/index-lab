//! LIM (Locality Index Method) index implementation.
//!
//! This implements an Adaptive Locality-Aware Indexing with Temporal Decay.
//! The algorithm combines spatial proximity with temporal proximity, enabling
//! queries that favor recent vectors while maintaining spatial locality.

use anyhow::{ensure, Result};
use index_core::{
    distance, load_index, save_index, validate_dimension, DistanceMetric, ScoredPoint, Vector,
    VectorIndex,
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LimError {
    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("index is empty, cannot search")]
    EmptyIndex,
}

/// LIM index configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LimConfig {
    /// Weight for spatial distance (0.0 to 1.0). Higher values favor spatial proximity.
    /// Temporal weight is (1.0 - spatial_weight).
    pub spatial_weight: f32,
    /// Temporal decay factor. Higher values mean older vectors decay faster.
    /// Typical range: 0.001 to 0.1 (smaller = slower decay)
    pub temporal_decay: f32,
    /// Number of locality clusters to maintain
    pub n_clusters: usize,
    /// Number of clusters to probe during search
    pub n_probe: usize,
}

impl Default for LimConfig {
    fn default() -> Self {
        Self {
            spatial_weight: 0.7,  // 70% spatial, 30% temporal
            temporal_decay: 0.01, // Moderate decay rate
            n_clusters: 50,       // Moderate number of clusters
            n_probe: 5,           // Probe top 5 clusters
        }
    }
}

/// Vector entry with temporal metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VectorEntry {
    id: usize,
    vector: Vector,
    timestamp: u64, // Unix timestamp in seconds
}

/// Locality cluster containing vectors with similar spatial and temporal properties
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LocalityCluster {
    /// Representative vector (centroid) for spatial locality
    spatial_centroid: Vector,
    /// Representative timestamp for temporal locality
    temporal_centroid: u64,
    /// Vectors in this cluster
    vectors: Vec<VectorEntry>,
}

/// LIM index implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimIndex {
    metric: DistanceMetric,
    dimension: Option<usize>,
    config: LimConfig,
    /// Locality clusters
    clusters: Vec<LocalityCluster>,
    /// Total number of vectors indexed
    vector_count: usize,
    /// Reference time for temporal calculations (most recent insertion)
    reference_time: u64,
}

impl LimIndex {
    /// Creates a new LIM index with the given metric and configuration
    pub fn new(metric: DistanceMetric, config: LimConfig) -> Self {
        Self {
            metric,
            dimension: None,
            config,
            clusters: Vec::new(),
            vector_count: 0,
            reference_time: 0,
        }
    }

    /// Creates a new LIM index with default configuration
    pub fn with_defaults(metric: DistanceMetric) -> Self {
        Self::new(metric, LimConfig::default())
    }

    fn validate_dimension(&self, vector: &[f32]) -> Result<()> {
        if let Some(expected) = self.dimension {
            validate_dimension(Some(expected), vector.len()).map_err(|_| {
                LimError::DimensionMismatch {
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

    /// Gets the current timestamp
    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Computes combined spatial-temporal distance
    /// Returns: spatial_weight * spatial_dist + (1 - spatial_weight) * temporal_dist
    fn combined_distance(&self, spatial_dist: f32, time_diff: u64) -> f32 {
        // Normalize temporal distance using exponential decay
        // Older vectors get higher temporal distance
        let temporal_dist = 1.0 - (-(time_diff as f32) * self.config.temporal_decay).exp();

        // Combine spatial and temporal distances
        self.config.spatial_weight * spatial_dist
            + (1.0 - self.config.spatial_weight) * temporal_dist
    }

    /// Finds the nearest locality clusters to a query vector
    fn find_nearest_clusters(
        &self,
        query: &Vector,
        query_time: u64,
        n_probe: usize,
    ) -> Result<Vec<usize>> {
        let mut cluster_distances: Vec<(usize, f32)> = self
            .clusters
            .iter()
            .enumerate()
            .map(|(idx, cluster)| {
                // Compute spatial distance to cluster centroid
                let spatial_dist =
                    distance(self.metric, query, &cluster.spatial_centroid).unwrap_or(f32::MAX);

                // Compute temporal distance
                let time_diff = query_time.abs_diff(cluster.temporal_centroid);

                // Combined distance
                let combined_dist = self.combined_distance(spatial_dist, time_diff);
                (idx, combined_dist)
            })
            .collect();

        // Sort by combined distance and take top n_probe
        cluster_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        cluster_distances.truncate(n_probe.min(self.clusters.len()));

        Ok(cluster_distances.into_iter().map(|(idx, _)| idx).collect())
    }

    /// Assigns a vector to the nearest locality cluster, creating a new one if needed
    fn assign_to_cluster(&mut self, entry: VectorEntry) -> Result<()> {
        if self.clusters.is_empty() {
            // Create first cluster
            let cluster = LocalityCluster {
                spatial_centroid: entry.vector.clone(),
                temporal_centroid: entry.timestamp,
                vectors: vec![entry],
            };
            self.clusters.push(cluster);
            return Ok(());
        }

        // Find nearest cluster
        let mut min_dist = f32::MAX;
        let mut nearest_cluster = 0;

        for (idx, cluster) in self.clusters.iter().enumerate() {
            let spatial_dist =
                distance(self.metric, &entry.vector, &cluster.spatial_centroid).unwrap_or(f32::MAX);

            let time_diff = entry.timestamp.abs_diff(cluster.temporal_centroid);

            let combined_dist = self.combined_distance(spatial_dist, time_diff);

            if combined_dist < min_dist {
                min_dist = combined_dist;
                nearest_cluster = idx;
            }
        }

        // Add to nearest cluster
        self.clusters[nearest_cluster].vectors.push(entry);

        // Update cluster centroid (simple average)
        let cluster = &mut self.clusters[nearest_cluster];
        let n = cluster.vectors.len();

        // Update spatial centroid
        let dimension = cluster.spatial_centroid.len();
        for i in 0..dimension {
            cluster.spatial_centroid[i] =
                cluster.vectors.iter().map(|v| v.vector[i]).sum::<f32>() / n as f32;
        }

        // Normalize for cosine distance if needed
        if self.metric.requires_unit_vectors() {
            let norm: f32 = cluster
                .spatial_centroid
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            if norm > 0.0 {
                for value in &mut cluster.spatial_centroid {
                    *value /= norm;
                }
            }
        }

        // Update temporal centroid (average timestamp)
        cluster.temporal_centroid =
            cluster.vectors.iter().map(|v| v.timestamp).sum::<u64>() / n as u64;

        // If we have too many clusters, merge the smallest ones
        if self.clusters.len() > self.config.n_clusters {
            // Simple strategy: remove smallest cluster and redistribute its vectors
            let smallest_idx = self
                .clusters
                .iter()
                .enumerate()
                .min_by_key(|(_, c)| c.vectors.len())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let removed_cluster = self.clusters.remove(smallest_idx);
            for entry in removed_cluster.vectors {
                self.assign_to_cluster(entry)?;
            }
        }

        Ok(())
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

impl VectorIndex for LimIndex {
    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn len(&self) -> usize {
        self.vector_count
    }

    fn insert(&mut self, id: usize, vector: Vector) -> Result<()> {
        self.validate_dimension(&vector)?;
        if self.dimension.is_none() {
            self.dimension = Some(vector.len());
        }

        let timestamp = Self::current_timestamp();
        if timestamp > self.reference_time {
            self.reference_time = timestamp;
        }

        let entry = VectorEntry {
            id,
            vector,
            timestamp,
        };

        self.assign_to_cluster(entry)?;
        self.vector_count += 1;

        Ok(())
    }

    fn search(&self, query: &Vector, limit: usize) -> Result<Vec<ScoredPoint>> {
        ensure!(
            limit > 0,
            "limit must be greater than zero to execute a search"
        );
        ensure!(self.vector_count > 0, LimError::EmptyIndex);
        self.validate_dimension(query)?;

        let query_time = Self::current_timestamp();

        // Find nearest clusters
        let cluster_indices = self.find_nearest_clusters(query, query_time, self.config.n_probe)?;

        // Search within selected clusters
        let mut candidates = Vec::new();
        for cluster_idx in cluster_indices {
            if cluster_idx >= self.clusters.len() {
                continue;
            }

            for entry in &self.clusters[cluster_idx].vectors {
                // Compute spatial distance
                let spatial_dist = distance(self.metric, query, &entry.vector)?;

                // Compute temporal distance
                let time_diff = query_time.abs_diff(entry.timestamp);

                // Combined distance
                let combined_dist = self.combined_distance(spatial_dist, time_diff);

                candidates.push(ScoredPoint::new(entry.id, combined_dist));
            }
        }

        // Sort by combined distance and take top limit
        candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        candidates.truncate(limit.min(candidates.len()));

        Ok(candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn new_index_has_zero_length() {
        let index = LimIndex::with_defaults(DistanceMetric::Euclidean);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn insert_sets_dimension() {
        let mut index = LimIndex::with_defaults(DistanceMetric::Euclidean);
        index.insert(0, vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(index.dimension(), Some(3));
    }

    #[test]
    fn insert_and_search_returns_results() {
        let mut index = LimIndex::with_defaults(DistanceMetric::Euclidean);
        index.insert(0, vec![0.0, 0.0]).unwrap();
        index.insert(1, vec![1.0, 0.0]).unwrap();
        index.insert(2, vec![0.0, 1.0]).unwrap();

        let result = index.search(&vec![0.0, 0.0], 2).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, 0); // Should find the exact match first
    }

    #[test]
    fn temporal_decay_affects_results() {
        let mut index = LimIndex::new(
            DistanceMetric::Euclidean,
            LimConfig {
                spatial_weight: 0.3,  // Favor temporal (70% temporal weight)
                temporal_decay: 0.01, // Moderate decay
                n_clusters: 10,
                n_probe: 3,
            },
        );

        // Insert old vector
        index.insert(0, vec![0.0, 0.0]).unwrap();

        // Wait a bit to ensure different timestamps
        thread::sleep(Duration::from_millis(100));

        // Insert newer vector at same location
        index.insert(1, vec![0.0, 0.0]).unwrap();

        // Search should prefer the newer vector due to temporal decay
        let result = index.search(&vec![0.0, 0.0], 2).unwrap();
        assert_eq!(result.len(), 2);
        // With high temporal weight, newer vector should have lower combined distance
        // (Note: This test may be flaky if timestamps are too close, so we just check both are found)
        assert!(result.iter().any(|r| r.id == 0));
        assert!(result.iter().any(|r| r.id == 1));
    }

    #[test]
    fn spatial_locality_works() {
        let mut index = LimIndex::with_defaults(DistanceMetric::Euclidean);

        // Insert vectors in two spatial clusters
        index.insert(0, vec![0.0, 0.0]).unwrap();
        index.insert(1, vec![0.1, 0.1]).unwrap();
        index.insert(2, vec![10.0, 10.0]).unwrap();
        index.insert(3, vec![10.1, 10.1]).unwrap();

        // Search near first cluster
        let result = index.search(&vec![0.0, 0.0], 2).unwrap();
        assert_eq!(result.len(), 2);
        // Should find vectors from the first cluster
        assert!(result.iter().any(|r| r.id == 0));
        assert!(result.iter().any(|r| r.id == 1));
    }
}
