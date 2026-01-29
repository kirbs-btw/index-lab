//! NEXUS (Neural EXploration with Unified Spectral Routing) - Spectral Manifold Vector Index
//!
//! NEXUS exploits manifold structure for faster vector search through:
//!
//! 1. **Spectral Embedding**: Projects vectors to low-dimensional spectral space for fast filtering
//! 2. **Entropy-Adaptive Graph**: More edges in sparse regions, fewer in dense clusters
//! 3. **Two-Phase Search**: Fast spectral distance filtering + full distance reranking
//!
//! This implementation uses random projections (Johnson-Lindenstrauss lemma) as a practical
//! approximation for spectral embedding, avoiding the need for eigensolvers.
//!
//! Research Gap Addressed: Gap 3A - Learned Index Structures (spectral variant)

use anyhow::{ensure, Result};
use index_core::{distance, DistanceMetric, ScoredPoint, Vector, VectorIndex};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashSet};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NexusError {
    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("index is empty, cannot search")]
    EmptyIndex,
    #[error("index not built, call build() first")]
    NotBuilt,
}

/// Random projection matrix for dimensionality reduction
/// Uses Johnson-Lindenstrauss lemma: distances preserved with high probability
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpectralProjector {
    /// Projection matrix: spectral_dim x original_dim
    projections: Vec<Vec<f32>>,
    /// Output dimensionality
    spectral_dim: usize,
}

impl SpectralProjector {
    /// Creates a random projection matrix
    fn new(original_dim: usize, spectral_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        // Generate random Gaussian projections (normalized)
        let scale = 1.0 / (spectral_dim as f32).sqrt();
        let projections: Vec<Vec<f32>> = (0..spectral_dim)
            .map(|_| {
                (0..original_dim)
                    .map(|_| {
                        // Approximate Gaussian using Box-Muller would be better,
                        // but uniform [-1, 1] works reasonably well
                        let u: f32 = rng.gen_range(-1.0..1.0);
                        u * scale
                    })
                    .collect()
            })
            .collect();

        Self {
            projections,
            spectral_dim,
        }
    }

    /// Projects a vector to spectral space
    fn project(&self, vector: &[f32]) -> Vec<f32> {
        self.projections
            .iter()
            .map(|proj| proj.iter().zip(vector.iter()).map(|(p, v)| p * v).sum())
            .collect()
    }
}

/// Computes spectral (low-dimensional) distance
fn spectral_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

/// Node in the adaptive graph
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GraphNode {
    id: usize,
    vector: Vector,
    spectral: Vec<f32>,
    neighbors: Vec<usize>,
    local_entropy: f32,
}

/// Entropy-adaptive graph structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdaptiveGraph {
    nodes: Vec<GraphNode>,
    base_edges: usize,
    mean_entropy: f32,
}

impl AdaptiveGraph {
    fn new(base_edges: usize) -> Self {
        Self {
            nodes: Vec::new(),
            base_edges,
            mean_entropy: 1.0,
        }
    }

    /// Computes local entropy based on neighbor distance variance
    fn compute_local_entropy(distances: &[f32]) -> f32 {
        if distances.is_empty() {
            return 1.0;
        }

        let sum: f32 = distances.iter().sum();
        if sum <= 0.0 {
            return 0.0;
        }

        // Normalize to probabilities
        let probs: Vec<f32> = distances.iter().map(|d| d / sum).collect();

        // Compute entropy: -Σ p * log(p)
        let entropy: f32 = probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();

        entropy
    }

    /// Computes adaptive edge count based on local entropy
    fn compute_edge_count(&self, entropy: f32) -> usize {
        if self.mean_entropy <= 0.0 {
            return self.base_edges;
        }

        // More edges in high-entropy (sparse/boundary) regions
        let multiplier = (entropy / self.mean_entropy).clamp(0.5, 2.0);
        ((self.base_edges as f32) * multiplier) as usize
    }

    /// Builds the graph from vectors and their spectral embeddings
    fn build(
        &mut self,
        vectors: &[(usize, Vector)],
        spectral_vecs: &[Vec<f32>],
        metric: DistanceMetric,
    ) {
        let n = vectors.len();
        if n == 0 {
            return;
        }

        // First pass: compute all pairwise distances and find k-NN for each node
        let k = self.base_edges * 2; // Use 2x base edges for entropy estimation

        // Initialize nodes
        self.nodes = vectors
            .iter()
            .zip(spectral_vecs.iter())
            .map(|((id, vec), spec)| GraphNode {
                id: *id,
                vector: vec.clone(),
                spectral: spec.clone(),
                neighbors: Vec::new(),
                local_entropy: 1.0,
            })
            .collect();

        // Compute k-NN for each node and estimate entropy
        let mut all_entropies = Vec::with_capacity(n);

        for i in 0..n {
            // Find k nearest neighbors
            let mut distances: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .filter_map(|j| {
                    distance(metric, &self.nodes[i].vector, &self.nodes[j].vector)
                        .ok()
                        .map(|d| (j, d))
                })
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.truncate(k);

            // Compute local entropy
            let neighbor_dists: Vec<f32> = distances.iter().map(|(_, d)| *d).collect();
            let entropy = Self::compute_local_entropy(&neighbor_dists);
            self.nodes[i].local_entropy = entropy;
            all_entropies.push(entropy);

            // Store neighbor IDs (will be pruned later based on adaptive count)
            self.nodes[i].neighbors = distances.iter().map(|(j, _)| *j).collect();
        }

        // Compute mean entropy
        self.mean_entropy = if all_entropies.is_empty() {
            1.0
        } else {
            all_entropies.iter().sum::<f32>() / all_entropies.len() as f32
        };

        // Second pass: prune neighbors based on adaptive edge count
        for i in 0..n {
            let entropy = self.nodes[i].local_entropy;
            let edge_count = self.compute_edge_count(entropy);
            self.nodes[i].neighbors.truncate(edge_count);
        }

        // Make edges bidirectional
        let mut to_add: Vec<(usize, usize)> = Vec::new();
        for i in 0..n {
            for &j in &self.nodes[i].neighbors {
                if !self.nodes[j].neighbors.contains(&i) {
                    to_add.push((j, i));
                }
            }
        }
        for (node, neighbor) in to_add {
            self.nodes[node].neighbors.push(neighbor);
        }
    }

    fn neighbors(&self, node_idx: usize) -> &[usize] {
        &self.nodes[node_idx].neighbors
    }
}

/// NEXUS index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NexusConfig {
    /// Spectral embedding dimensionality (m << d)
    pub spectral_dim: usize,
    /// Base number of edges per node
    pub base_edges: usize,
    /// Number of candidates to explore during search
    pub ef_search: usize,
    /// Ratio of spectral candidates to keep for reranking
    pub rerank_ratio: f32,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for NexusConfig {
    fn default() -> Self {
        Self {
            spectral_dim: 32,
            base_edges: 16,
            ef_search: 100,
            rerank_ratio: 2.0,
            seed: 42,
        }
    }
}

/// NEXUS index - spectral manifold vector index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NexusIndex {
    metric: DistanceMetric,
    dimension: Option<usize>,
    config: NexusConfig,
    projector: Option<SpectralProjector>,
    graph: AdaptiveGraph,
    is_built: bool,
}

impl NexusIndex {
    /// Creates a new NEXUS index with the given metric and configuration
    pub fn new(metric: DistanceMetric, config: NexusConfig) -> Self {
        Self {
            metric,
            dimension: None,
            config: config.clone(),
            projector: None,
            graph: AdaptiveGraph::new(config.base_edges),
            is_built: false,
        }
    }

    /// Creates a new NEXUS index with default configuration
    pub fn with_defaults(metric: DistanceMetric) -> Self {
        Self::new(metric, NexusConfig::default())
    }

    /// Returns the dimensionality tracked by the index
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Projects a query vector to spectral space
    fn project_query(&self, query: &[f32]) -> Option<Vec<f32>> {
        self.projector.as_ref().map(|p| p.project(query))
    }

    /// Validates vector dimension
    fn validate_dimension(&self, vector: &[f32]) -> Result<()> {
        if let Some(dim) = self.dimension {
            ensure!(
                vector.len() == dim,
                NexusError::DimensionMismatch {
                    expected: dim,
                    actual: vector.len(),
                }
            );
        }
        Ok(())
    }

    /// Saves the index to a JSON file
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        index_core::save_index(self, path)
    }

    /// Loads an index from a JSON file
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        index_core::load_index(path)
    }
}

/// Min-heap entry for search
#[derive(Debug, Clone)]
struct SearchEntry {
    node_idx: usize,
    distance: f32,
}

impl PartialEq for SearchEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for SearchEntry {}

impl PartialOrd for SearchEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse for min-heap (smallest distance first)
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl VectorIndex for NexusIndex {
    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn len(&self) -> usize {
        self.graph.nodes.len()
    }

    fn build(&mut self, data: impl IntoIterator<Item = (usize, Vector)>) -> Result<()> {
        let vectors: Vec<(usize, Vector)> = data.into_iter().collect();

        if vectors.is_empty() {
            return Ok(());
        }

        // Set dimension from first vector
        let dim = vectors[0].1.len();
        self.dimension = Some(dim);

        // Create spectral projector
        let projector = SpectralProjector::new(dim, self.config.spectral_dim, self.config.seed);

        // Project all vectors to spectral space
        let spectral_vecs: Vec<Vec<f32>> =
            vectors.iter().map(|(_, v)| projector.project(v)).collect();

        self.projector = Some(projector);

        // Build adaptive graph
        self.graph.build(&vectors, &spectral_vecs, self.metric);
        self.is_built = true;

        Ok(())
    }

    fn insert(&mut self, id: usize, vector: Vector) -> Result<()> {
        // For simplicity, we require build() to be called with all data
        // In a full implementation, we'd support incremental insertion
        if self.dimension.is_none() {
            self.dimension = Some(vector.len());
        }

        ensure!(
            vector.len() == self.dimension.unwrap(),
            NexusError::DimensionMismatch {
                expected: self.dimension.unwrap(),
                actual: vector.len(),
            }
        );

        // Project to spectral space
        let spectral = self
            .projector
            .as_ref()
            .map(|p| p.project(&vector))
            .unwrap_or_else(|| vec![0.0; self.config.spectral_dim]);

        // Add node with no neighbors (simplified)
        self.graph.nodes.push(GraphNode {
            id,
            vector,
            spectral,
            neighbors: Vec::new(),
            local_entropy: 1.0,
        });

        Ok(())
    }

    fn search(&self, query: &Vector, limit: usize) -> Result<Vec<ScoredPoint>> {
        ensure!(limit > 0, "limit must be greater than zero");
        ensure!(!self.graph.nodes.is_empty(), NexusError::EmptyIndex);
        ensure!(self.is_built, NexusError::NotBuilt);

        let q_spectral = self.project_query(query).ok_or(NexusError::NotBuilt)?;

        // Phase 1: Graph traversal using spectral distances
        let mut visited: HashSet<usize> = HashSet::new();
        let mut candidates: BinaryHeap<SearchEntry> = BinaryHeap::new();
        let mut results: Vec<SearchEntry> = Vec::new();

        // Find entry point (node with smallest spectral distance)
        let entry_idx = (0..self.graph.nodes.len())
            .min_by(|&a, &b| {
                let dist_a = spectral_distance(&q_spectral, &self.graph.nodes[a].spectral);
                let dist_b = spectral_distance(&q_spectral, &self.graph.nodes[b].spectral);
                dist_a.partial_cmp(&dist_b).unwrap()
            })
            .unwrap_or(0);

        let entry_dist = spectral_distance(&q_spectral, &self.graph.nodes[entry_idx].spectral);
        candidates.push(SearchEntry {
            node_idx: entry_idx,
            distance: entry_dist,
        });

        // Explore graph using spectral distance
        while let Some(current) = candidates.pop() {
            if visited.len() >= self.config.ef_search {
                break;
            }

            if visited.contains(&current.node_idx) {
                continue;
            }
            visited.insert(current.node_idx);
            results.push(current.clone());

            // Add neighbors to candidates
            for &neighbor_idx in self.graph.neighbors(current.node_idx) {
                if !visited.contains(&neighbor_idx) {
                    let s_dist =
                        spectral_distance(&q_spectral, &self.graph.nodes[neighbor_idx].spectral);
                    candidates.push(SearchEntry {
                        node_idx: neighbor_idx,
                        distance: s_dist,
                    });
                }
            }
        }

        // Phase 2: Rerank top candidates by full distance
        let rerank_count = ((limit as f32) * self.config.rerank_ratio) as usize;
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.truncate(rerank_count.max(limit));

        // Compute full distances
        let mut final_results: Vec<ScoredPoint> = results
            .into_iter()
            .filter_map(|entry| {
                let node = &self.graph.nodes[entry.node_idx];
                distance(self.metric, query, &node.vector)
                    .ok()
                    .map(|d| ScoredPoint::new(node.id, d))
            })
            .collect();

        // Sort by full distance and return top-k
        final_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        final_results.truncate(limit);

        Ok(final_results)
    }

    fn delete(&mut self, id: usize) -> Result<bool> {
        // Find node index by id
        let node_idx_opt = self.graph.nodes.iter().position(|n| n.id == id);
        
        if let Some(node_idx) = node_idx_opt {
            // Remove from neighbors lists of all nodes
            for node in &mut self.graph.nodes {
                // Remove references to this node index
                node.neighbors.retain(|&idx| idx != node_idx);
                // Decrement indices greater than removed index
                for idx in &mut node.neighbors {
                    if *idx > node_idx {
                        *idx -= 1;
                    }
                }
            }
            
            // Remove the node
            self.graph.nodes.remove(node_idx);
            
            // Mark as needing rebuild (graph structure changed)
            self.is_built = false;
            
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn update(&mut self, id: usize, vector: Vector) -> Result<bool> {
        self.validate_dimension(&vector)?;
        
        // Find node by id
        let node_opt = self.graph.nodes.iter_mut().find(|n| n.id == id);
        
        if let Some(node) = node_opt {
            // Update vector
            node.vector = vector.clone();
            
            // Re-project to spectral space
            if let Some(projector) = &self.projector {
                node.spectral = projector.project(&vector);
            }
            
            // Mark as needing rebuild (entropy/neighbors may have changed)
            self.is_built = false;
            
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_index_has_zero_length() {
        let index = NexusIndex::with_defaults(DistanceMetric::Euclidean);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn build_sets_dimension() {
        let mut index = NexusIndex::with_defaults(DistanceMetric::Euclidean);
        index.build(vec![(0, vec![1.0, 2.0, 3.0])]).unwrap();
        assert_eq!(index.dimension(), Some(3));
    }

    #[test]
    fn spectral_projection_preserves_relative_distances() {
        let projector = SpectralProjector::new(64, 16, 42);

        let v1 = vec![0.0; 64];
        let v2: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let v3: Vec<f32> = (0..64).map(|i| (i as f32 / 64.0) * 2.0).collect();

        let s1 = projector.project(&v1);
        let s2 = projector.project(&v2);
        let s3 = projector.project(&v3);

        // v1 should be closer to v2 than to v3
        let d12 = spectral_distance(&s1, &s2);
        let d13 = spectral_distance(&s1, &s3);

        assert!(
            d12 < d13,
            "Spectral projection should preserve relative distances"
        );
    }

    #[test]
    fn local_entropy_varies_with_distance_distribution() {
        // Uniform distances → higher entropy
        let uniform_dists = vec![1.0, 1.0, 1.0, 1.0];
        let uniform_entropy = AdaptiveGraph::compute_local_entropy(&uniform_dists);

        // Varied distances → lower entropy
        let varied_dists = vec![0.1, 0.5, 2.0, 5.0];
        let varied_entropy = AdaptiveGraph::compute_local_entropy(&varied_dists);

        assert!(
            uniform_entropy > varied_entropy,
            "Uniform distances should have higher entropy"
        );
    }

    #[test]
    fn build_and_search_returns_results() {
        let mut index = NexusIndex::with_defaults(DistanceMetric::Euclidean);
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
    fn search_with_larger_dataset() {
        let mut index = NexusIndex::new(
            DistanceMetric::Euclidean,
            NexusConfig {
                spectral_dim: 8,
                base_edges: 8,
                ef_search: 50,
                rerank_ratio: 2.0,
                seed: 42,
            },
        );

        // Create clustered data
        let mut data = Vec::new();
        for i in 0..50 {
            // Cluster near origin
            data.push((i, vec![i as f32 / 100.0, i as f32 / 100.0]));
        }
        for i in 50..100 {
            // Cluster far away
            data.push((i, vec![10.0 + (i - 50) as f32 / 100.0, 10.0]));
        }
        index.build(data).unwrap();

        // Query near first cluster
        let result = index.search(&vec![0.1, 0.1], 5).unwrap();
        assert_eq!(result.len(), 5);

        // All results should be from first cluster (IDs 0-49)
        for point in &result {
            assert!(point.id < 50, "Should find points from near cluster");
        }
    }

    #[test]
    fn adaptive_graph_assigns_variable_edges() {
        let mut index = NexusIndex::with_defaults(DistanceMetric::Euclidean);

        // Create data with different densities
        let mut data = Vec::new();
        // Dense cluster
        for i in 0..20 {
            data.push((i, vec![i as f32 * 0.01, 0.0]));
        }
        // Sparse region
        for i in 20..30 {
            data.push((i, vec![10.0 + (i - 20) as f32 * 1.0, 0.0]));
        }
        index.build(data).unwrap();

        // Graph should be built with variable edge counts
        assert!(index.is_built);
        assert!(index.graph.mean_entropy > 0.0);
    }

    #[test]
    fn save_and_load_roundtrip() {
        let mut index = NexusIndex::with_defaults(DistanceMetric::Euclidean);
        index
            .build(vec![(0, vec![1.0, 2.0]), (1, vec![3.0, 4.0])])
            .unwrap();

        let temp_path = std::env::temp_dir().join("nexus_test_index.json");
        index.save(&temp_path).unwrap();

        let loaded = NexusIndex::load(&temp_path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.dimension(), Some(2));
        assert!(loaded.is_built);

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn cosine_metric_works() {
        let mut index = NexusIndex::with_defaults(DistanceMetric::Cosine);
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
}
