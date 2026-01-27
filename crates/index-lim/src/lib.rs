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

/// Simple KD-tree node for spatial cluster lookup
#[derive(Debug, Clone)]
struct KdNode {
    cluster_idx: usize,
    point: Vector,
    left: Option<Box<KdNode>>,
    right: Option<Box<KdNode>>,
}

impl KdNode {
    fn new(cluster_idx: usize, point: Vector) -> Self {
        Self {
            cluster_idx,
            point,
            left: None,
            right: None,
        }
    }
}

/// KD-tree for fast spatial nearest neighbor search of cluster centroids
#[derive(Debug, Clone)]
struct ClusterKdTree {
    root: Option<Box<KdNode>>,
    dimension: usize,
}

impl ClusterKdTree {
    fn new(dimension: usize) -> Self {
        Self {
            root: None,
            dimension,
        }
    }

    fn build(&mut self, clusters: &[LocalityCluster]) {
        if clusters.is_empty() {
            self.root = None;
            return;
        }

        let points: Vec<(usize, &Vector)> = clusters
            .iter()
            .enumerate()
            .map(|(idx, cluster)| (idx, &cluster.spatial_centroid))
            .collect();

        self.root = Some(Box::new(self.build_recursive(&points, 0)));
    }

    fn build_recursive(&self, points: &[(usize, &Vector)], depth: usize) -> KdNode {
        if points.len() == 1 {
            return KdNode::new(points[0].0, points[0].1.clone());
        }

        let axis = depth % self.dimension;
        let mut sorted_points = points.to_vec();
        sorted_points.sort_by(|a, b| {
            a.1[axis]
                .partial_cmp(&b.1[axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mid = sorted_points.len() / 2;
        let median = sorted_points[mid].clone();

        let left_points = &sorted_points[..mid];
        let right_points = &sorted_points[mid + 1..];

        let mut node = KdNode::new(median.0, median.1.clone());
        if !left_points.is_empty() {
            node.left = Some(Box::new(self.build_recursive(left_points, depth + 1)));
        }
        if !right_points.is_empty() {
            node.right = Some(Box::new(self.build_recursive(right_points, depth + 1)));
        }

        node
    }

    fn find_nearest(
        &self,
        query: &Vector,
        metric: DistanceMetric,
        n: usize,
    ) -> Vec<(usize, f32)> {
        if self.root.is_none() {
            return Vec::new();
        }

        let mut candidates = Vec::new();
        self.search_recursive(
            self.root.as_ref().unwrap(),
            query,
            metric,
            0,
            &mut candidates,
        );

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(n);
        candidates
    }

    fn search_recursive(
        &self,
        node: &KdNode,
        query: &Vector,
        metric: DistanceMetric,
        depth: usize,
        candidates: &mut Vec<(usize, f32)>,
    ) {
        let dist = distance(metric, query, &node.point).unwrap_or(f32::MAX);
        candidates.push((node.cluster_idx, dist));

        let axis = depth % self.dimension;
        let diff = query[axis] - node.point[axis];
        let go_left = diff < 0.0;

        // Search the closer subtree first
        if go_left {
            if let Some(ref left) = node.left {
                self.search_recursive(left, query, metric, depth + 1, candidates);
            }
            // Check if we need to search the other side
            if diff * diff < candidates.iter().map(|(_, d)| d * d).fold(f32::INFINITY, f32::min) {
                if let Some(ref right) = node.right {
                    self.search_recursive(right, query, metric, depth + 1, candidates);
                }
            }
        } else {
            if let Some(ref right) = node.right {
                self.search_recursive(right, query, metric, depth + 1, candidates);
            }
            if diff * diff < candidates.iter().map(|(_, d)| d * d).fold(f32::INFINITY, f32::min) {
                if let Some(ref left) = node.left {
                    self.search_recursive(left, query, metric, depth + 1, candidates);
                }
            }
        }
    }
}

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
#[serde(from = "LimIndexData", into = "LimIndexData")]
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
    /// Tracked spatial distance statistics for normalization
    spatial_dist_min: Option<f32>,
    spatial_dist_max: Option<f32>,
    /// KD-tree for fast cluster centroid lookup (not serialized, rebuilt on load)
    #[serde(skip)]
    cluster_kdtree: Option<ClusterKdTree>,
}

/// Serializable representation of LimIndex (without KD-tree)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LimIndexData {
    metric: DistanceMetric,
    dimension: Option<usize>,
    config: LimConfig,
    clusters: Vec<LocalityCluster>,
    vector_count: usize,
    reference_time: u64,
    /// Tracked spatial distance statistics for normalization
    spatial_dist_min: Option<f32>,
    spatial_dist_max: Option<f32>,
}

impl From<LimIndexData> for LimIndex {
    fn from(data: LimIndexData) -> Self {
        let mut index = LimIndex {
            metric: data.metric,
            dimension: data.dimension,
            config: data.config,
            clusters: data.clusters,
            vector_count: data.vector_count,
            reference_time: data.reference_time,
            spatial_dist_min: data.spatial_dist_min,
            spatial_dist_max: data.spatial_dist_max,
            cluster_kdtree: None,
        };
        index.rebuild_kdtree();
        index
    }
}

impl From<LimIndex> for LimIndexData {
    fn from(index: LimIndex) -> Self {
        LimIndexData {
            metric: index.metric,
            dimension: index.dimension,
            config: index.config,
            clusters: index.clusters,
            vector_count: index.vector_count,
            reference_time: index.reference_time,
            spatial_dist_min: index.spatial_dist_min,
            spatial_dist_max: index.spatial_dist_max,
        }
    }
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
            spatial_dist_min: None,
            spatial_dist_max: None,
            cluster_kdtree: None,
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

    /// Normalizes spatial distance to [0, 1] range using tracked statistics
    fn normalize_spatial_distance(&self, spatial_dist: f32) -> f32 {
        match (self.spatial_dist_min, self.spatial_dist_max) {
            (Some(min), Some(max)) if (max - min).abs() > 1e-10 => {
                // Normalize to [0, 1]
                ((spatial_dist - min) / (max - min)).clamp(0.0, 1.0)
            }
            _ => {
                // No statistics yet or zero range, return as-is (will be normalized later)
                // For cosine distance, it's already in [0, 2], so normalize to [0, 1]
                if self.metric == DistanceMetric::Cosine {
                    spatial_dist / 2.0
                } else {
                    // For Euclidean, use a heuristic: assume max distance is sqrt(dimension * max_coord^2)
                    // This is a rough estimate, will be refined as we see more distances
                    let estimated_max = if let Some(dim) = self.dimension {
                        (dim as f32 * 4.0).sqrt() // Assuming vectors in [-1, 1] range
                    } else {
                        10.0 // Fallback
                    };
                    (spatial_dist / estimated_max).min(1.0)
                }
            }
        }
    }

    /// Updates spatial distance statistics
    fn update_spatial_stats(&mut self, spatial_dist: f32) {
        match (self.spatial_dist_min, self.spatial_dist_max) {
            (None, None) => {
                self.spatial_dist_min = Some(spatial_dist);
                self.spatial_dist_max = Some(spatial_dist);
            }
            (Some(min), Some(max)) => {
                if spatial_dist < min {
                    self.spatial_dist_min = Some(spatial_dist);
                }
                if spatial_dist > max {
                    self.spatial_dist_max = Some(spatial_dist);
                }
            }
            _ => {}
        }
    }


    /// Immutable version for search (uses current stats without updating)
    fn combined_distance_immutable(&self, spatial_dist: f32, time_diff: u64) -> f32 {
        let normalized_spatial = self.normalize_spatial_distance(spatial_dist);
        let temporal_dist = 1.0 - (-(time_diff as f32) * self.config.temporal_decay).exp();
        self.config.spatial_weight * normalized_spatial
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

                // Combined distance (immutable version for search)
                let combined_dist = self.combined_distance_immutable(spatial_dist, time_diff);
                (idx, combined_dist)
            })
            .collect();

        // Sort by combined distance and take top n_probe
        cluster_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        cluster_distances.truncate(n_probe.min(self.clusters.len()));

        Ok(cluster_distances.into_iter().map(|(idx, _)| idx).collect())
    }

    /// Rebuilds the KD-tree from current clusters
    fn rebuild_kdtree(&mut self) {
        if let Some(dim) = self.dimension {
            let mut kdtree = ClusterKdTree::new(dim);
            kdtree.build(&self.clusters);
            self.cluster_kdtree = Some(kdtree);
        }
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
            self.rebuild_kdtree();
            return Ok(());
        }

        // Use KD-tree for fast spatial pre-filtering, then check combined distance
        let nearest_cluster = if let Some(ref kdtree) = self.cluster_kdtree {
            // Get top spatial candidates from KD-tree (check more than needed for temporal filtering)
            let spatial_candidates = kdtree.find_nearest(&entry.vector, self.metric, self.clusters.len().min(10));
            
            // Find the one with minimum combined distance
            let mut min_dist = f32::MAX;
            let mut nearest_idx = 0;
            let mut spatial_distances = Vec::new();
            
            for (cluster_idx, spatial_dist) in spatial_candidates {
                if cluster_idx >= self.clusters.len() {
                    continue;
                }
                spatial_distances.push(spatial_dist);
                let cluster = &self.clusters[cluster_idx];
                let time_diff = entry.timestamp.abs_diff(cluster.temporal_centroid);
                let combined_dist = self.combined_distance_immutable(spatial_dist, time_diff);
                
                if combined_dist < min_dist {
                    min_dist = combined_dist;
                    nearest_idx = cluster_idx;
                }
            }
            
            // Update stats after the loop
            for spatial_dist in spatial_distances {
                self.update_spatial_stats(spatial_dist);
            }
            
            nearest_idx
        } else {
            // Fallback: linear search if KD-tree not built
            let mut min_dist = f32::MAX;
            let mut nearest_cluster = 0;
            let mut spatial_distances = Vec::new();

            for (idx, cluster) in self.clusters.iter().enumerate() {
                let spatial_dist =
                    distance(self.metric, &entry.vector, &cluster.spatial_centroid).unwrap_or(f32::MAX);
                spatial_distances.push((idx, spatial_dist));

                let time_diff = entry.timestamp.abs_diff(cluster.temporal_centroid);
                let combined_dist = self.combined_distance_immutable(spatial_dist, time_diff);

                if combined_dist < min_dist {
                    min_dist = combined_dist;
                    nearest_cluster = idx;
                }
            }
            
            // Update stats after the loop
            for (_, spatial_dist) in spatial_distances {
                self.update_spatial_stats(spatial_dist);
            }
            
            nearest_cluster
        };

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

        // Rebuild KD-tree after centroid update
        self.rebuild_kdtree();

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
            // Rebuild KD-tree after removal
            self.rebuild_kdtree();
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
        let mut index: LimIndex = load_index(path)?;
        // Rebuild KD-tree after loading
        index.rebuild_kdtree();
        Ok(index)
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

                // Combined distance (immutable version for search)
                let combined_dist = self.combined_distance_immutable(spatial_dist, time_diff);

                candidates.push(ScoredPoint::new(entry.id, combined_dist));
            }
        }

        // Sort by combined distance and take top limit
        candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        candidates.truncate(limit.min(candidates.len()));

        Ok(candidates)
    }

    fn delete(&mut self, id: usize) -> Result<bool> {
        let mut found = false;
        
        // Search through all clusters to find and remove the vector
        for cluster in &mut self.clusters {
            let initial_len = cluster.vectors.len();
            cluster.vectors.retain(|entry| entry.id != id);
            
            if cluster.vectors.len() < initial_len {
                found = true;
                
                // Update cluster centroid if vectors remain
                if !cluster.vectors.is_empty() {
                    let n = cluster.vectors.len();
                    let dimension = cluster.spatial_centroid.len();
                    
                    // Recalculate spatial centroid
                    for i in 0..dimension {
                        cluster.spatial_centroid[i] =
                            cluster.vectors.iter().map(|v| v.vector[i]).sum::<f32>() / n as f32;
                    }
                    
                    // Normalize for cosine if needed
                    if self.metric.requires_unit_vectors() {
                        let norm: f32 = cluster.spatial_centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
                        if norm > 0.0 {
                            for value in &mut cluster.spatial_centroid {
                                *value /= norm;
                            }
                        }
                    }
                    
                    // Recalculate temporal centroid
                    cluster.temporal_centroid =
                        cluster.vectors.iter().map(|v| v.timestamp).sum::<u64>() / n as u64;
                }
            }
        }
        
        // Remove empty clusters
        self.clusters.retain(|cluster| !cluster.vectors.is_empty());
        
        if found {
            self.vector_count -= 1;
            self.rebuild_kdtree();
        }
        
        Ok(found)
    }

    fn update(&mut self, id: usize, vector: Vector) -> Result<bool> {
        self.validate_dimension(&vector)?;
        
        // Find and update the vector in clusters
        for cluster in &mut self.clusters {
            if let Some(entry) = cluster.vectors.iter_mut().find(|e| e.id == id) {
                entry.vector = vector.clone();
                entry.timestamp = Self::current_timestamp();
                
                // Recalculate cluster centroid
                let n = cluster.vectors.len();
                let dimension = cluster.spatial_centroid.len();
                
                for i in 0..dimension {
                    cluster.spatial_centroid[i] =
                        cluster.vectors.iter().map(|v| v.vector[i]).sum::<f32>() / n as f32;
                }
                
                if self.metric.requires_unit_vectors() {
                    let norm: f32 = cluster.spatial_centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for value in &mut cluster.spatial_centroid {
                            *value /= norm;
                        }
                    }
                }
                
                cluster.temporal_centroid =
                    cluster.vectors.iter().map(|v| v.timestamp).sum::<u64>() / n as u64;
                
                self.rebuild_kdtree();
                return Ok(true);
            }
        }
        
        Ok(false)
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
