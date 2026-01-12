//! FUSION: Fast Unified Search with Intelligent Orchestrated Navigation
//!
//! A novel vector index combining:
//! - **LSH bucketing** for O(1) candidate generation
//! - **Mini-graphs** for O(log b) navigation within buckets
//! - **Multi-probe** for high recall
//!
//! This addresses the O(n) issues found in SEER, LIM, and Hybrid indexes.
//!
//! # Architecture
//!
//! ```text
//! Layer 1: LSH Buckets (SimHash)    → O(1) candidate generation
//! Layer 2: Mini-Graphs (NSW)         → O(log b) navigation within buckets  
//! Layer 3: Exact Reranking           → Top-k from combined candidates
//! ```
//!
//! # Example
//!
//! ```rust
//! use index_fusion::{FusionIndex, FusionConfig};
//! use index_core::{DistanceMetric, VectorIndex};
//!
//! let mut index = FusionIndex::new(DistanceMetric::Euclidean, FusionConfig::default());
//! index.insert(0, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
//! index.insert(1, vec![1.1, 2.1, 3.1, 4.1]).unwrap();
//!
//! let query = vec![1.0, 2.0, 3.0, 4.0];
//! let results = index.search(&query, 2).unwrap();
//! assert_eq!(results[0].id, 0); // Exact match is closest
//! ```

use anyhow::Result;
use index_core::{distance, DistanceMetric, ScoredPoint, Vector, VectorIndex};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use thiserror::Error;

/// Errors specific to FUSION index operations.
#[derive(Debug, Error)]
pub enum FusionError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("index is empty, cannot search")]
    EmptyIndex,
    #[error("vector dimension must be at least 1")]
    ZeroDimension,
}

/// FUSION index configuration.
///
/// These parameters control the trade-off between recall, speed, and memory usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Number of hyperplanes for LSH (bucket count = 2^n_hyperplanes).
    /// More hyperplanes = more buckets = smaller buckets = faster search but potentially lower recall.
    /// Default: 8 (256 buckets)
    pub n_hyperplanes: usize,

    /// Number of buckets to probe during search.
    /// More probes = higher recall but slower search.
    /// Default: 8 (primary + 7 Hamming-1 neighbors)
    pub n_probes: usize,

    /// Maximum edges per node in mini-graphs.
    /// Higher = better navigation but more memory.
    /// Default: 16
    pub mini_graph_m: usize,

    /// Search beam width for mini-graph traversal.
    /// Higher = more thorough search but slower.
    /// Default: 64
    pub mini_graph_ef: usize,

    /// Minimum bucket size to build a mini-graph.
    /// Below this threshold, use linear scan within bucket.
    /// Default: 32
    pub min_bucket_for_graph: usize,

    /// Random seed for reproducible hyperplane generation.
    /// Default: 42
    pub seed: u64,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            n_hyperplanes: 3,        // 8 buckets (~1250 vectors per bucket for 10K dataset)
            n_probes: 8,             // Probe all buckets for maximum recall
            mini_graph_m: 20,        // Good edge count for navigation
            mini_graph_ef: 100,      // Moderate beam width for speed
            min_bucket_for_graph: 8, // Build graphs for reasonable bucket sizes
            seed: 42,
        }
    }
}

/// Entry for a stored vector with its ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VectorEntry {
    id: usize,
    vector: Vector,
}

/// LSH hasher using random hyperplane projections (SimHash).
///
/// Projects vectors onto random hyperplanes and uses the sign of each dot product
/// as a hash bit. Vectors with similar directions tend to hash to the same bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LshHasher {
    /// Random hyperplanes for projection (n_hyperplanes × dimension)
    hyperplanes: Vec<Vector>,
    /// Number of hash bits
    n_bits: usize,
}

impl LshHasher {
    /// Creates a new LSH hasher with random hyperplanes for the given dimension.
    fn new(dimension: usize, n_bits: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let hyperplanes = (0..n_bits)
            .map(|_| {
                let mut plane: Vector = (0..dimension)
                    .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
                    .collect();
                // Normalize hyperplane for consistent projection scale
                let norm: f32 = plane.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in &mut plane {
                        *x /= norm;
                    }
                }
                plane
            })
            .collect();

        Self {
            hyperplanes,
            n_bits,
        }
    }

    /// Hash a vector to a bucket index using SimHash.
    fn hash(&self, vector: &[f32]) -> usize {
        let mut bucket = 0usize;
        for (i, hyperplane) in self.hyperplanes.iter().enumerate() {
            let dot: f32 = vector
                .iter()
                .zip(hyperplane.iter())
                .map(|(a, b)| a * b)
                .sum();
            if dot > 0.0 {
                bucket |= 1 << i;
            }
        }
        bucket
    }

    /// Get buckets to probe (primary + Hamming-distance-1 and 2 neighbors).
    ///
    /// Multi-probe LSH improves recall by checking neighboring buckets
    /// that differ by one or two bits from the primary hash.
    fn get_probe_buckets(&self, primary: usize, n_probes: usize) -> Vec<usize> {
        let mut buckets = vec![primary];
        let n_buckets = self.n_buckets();

        // Add Hamming-1 neighbors (flip each bit)
        for bit in 0..self.n_bits {
            if buckets.len() >= n_probes {
                break;
            }
            let neighbor = primary ^ (1 << bit);
            if neighbor < n_buckets && !buckets.contains(&neighbor) {
                buckets.push(neighbor);
            }
        }

        // Add Hamming-2 neighbors (flip two bits)
        'outer: for bit1 in 0..self.n_bits {
            for bit2 in (bit1 + 1)..self.n_bits {
                if buckets.len() >= n_probes {
                    break 'outer;
                }
                let neighbor = primary ^ (1 << bit1) ^ (1 << bit2);
                if neighbor < n_buckets && !buckets.contains(&neighbor) {
                    buckets.push(neighbor);
                }
            }
        }

        buckets.truncate(n_probes);
        buckets
    }

    /// Total number of buckets (2^n_bits).
    fn n_buckets(&self) -> usize {
        1 << self.n_bits
    }
}

/// Entry for the search priority queue (min-heap by distance).
#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    local_id: usize,
    distance: f32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Invert for min-heap (we want smallest distance first)
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Small Navigable Small World (NSW) graph within a single bucket.
///
/// Each bucket maintains its own navigable graph for efficient local search.
/// This is simpler than full HNSW (single layer) but still provides O(log b) search.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MiniGraph {
    /// Global IDs of vectors in this bucket (in insertion order).
    vector_ids: Vec<usize>,
    /// Adjacency list: local_id -> neighbor local_ids.
    edges: Vec<Vec<usize>>,
    /// Entry point for search (local index of a well-connected node).
    entry_point: Option<usize>,
    /// Maximum edges per node.
    m: usize,
}

impl MiniGraph {
    /// Creates an empty mini-graph with the given edge limit.
    fn new(m: usize) -> Self {
        Self {
            vector_ids: Vec::new(),
            edges: Vec::new(),
            entry_point: None,
            m,
        }
    }

    /// Number of vectors in this bucket.
    fn len(&self) -> usize {
        self.vector_ids.len()
    }

    /// Whether this bucket is empty.
    fn is_empty(&self) -> bool {
        self.vector_ids.is_empty()
    }

    /// Insert a vector into this mini-graph.
    ///
    /// Builds bidirectional edges to the nearest neighbors already in the graph.
    fn insert(
        &mut self,
        global_id: usize,
        vector: &[f32],
        vectors: &[VectorEntry],
        metric: DistanceMetric,
    ) -> Result<()> {
        let local_id = self.vector_ids.len();
        self.vector_ids.push(global_id);
        self.edges.push(Vec::new());

        if local_id == 0 {
            // First node becomes entry point
            self.entry_point = Some(0);
            return Ok(());
        }

        // Find nearest neighbors among existing nodes
        let mut distances: Vec<(usize, f32)> = self
            .vector_ids
            .iter()
            .take(local_id) // Only existing nodes
            .enumerate()
            .map(|(lid, &gid)| {
                let dist = distance(metric, vector, &vectors[gid].vector).unwrap_or(f32::MAX);
                (lid, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Connect to nearest m neighbors
        let neighbors: Vec<usize> = distances.iter().take(self.m).map(|(lid, _)| *lid).collect();

        // Add forward edges
        self.edges[local_id] = neighbors.clone();

        // Add backward edges (bidirectional graph)
        for neighbor_lid in neighbors {
            if self.edges[neighbor_lid].len() < self.m {
                self.edges[neighbor_lid].push(local_id);
            } else {
                // Prune: replace worst neighbor if new connection is better
                let new_dist = distance(
                    metric,
                    &vectors[self.vector_ids[neighbor_lid]].vector,
                    vector,
                )
                .unwrap_or(f32::MAX);

                let mut worst_idx = None;
                let mut worst_dist = new_dist;

                for (i, &nid) in self.edges[neighbor_lid].iter().enumerate() {
                    let d = distance(
                        metric,
                        &vectors[self.vector_ids[neighbor_lid]].vector,
                        &vectors[self.vector_ids[nid]].vector,
                    )
                    .unwrap_or(0.0);
                    if d > worst_dist {
                        worst_dist = d;
                        worst_idx = Some(i);
                    }
                }

                if let Some(idx) = worst_idx {
                    self.edges[neighbor_lid][idx] = local_id;
                }
            }
        }

        // Update entry point to be the most connected node
        let current_ep = self.entry_point.unwrap_or(0);
        if self.edges[local_id].len() > self.edges[current_ep].len() {
            self.entry_point = Some(local_id);
        }

        Ok(())
    }

    /// Search within this mini-graph using greedy NSW search.
    ///
    /// Returns (global_id, distance) pairs sorted by distance.
    fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        vectors: &[VectorEntry],
        metric: DistanceMetric,
    ) -> Result<Vec<(usize, f32)>> {
        if self.is_empty() {
            return Ok(Vec::new());
        }

        // For small buckets, just do linear scan (simpler and effective)
        if self.vector_ids.len() <= ef {
            return self.linear_search(query, k, vectors, metric);
        }

        let entry = self.entry_point.unwrap_or(0);
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); // Min-heap (HeapEntry has inverted ord)

        // For results, we want max-heap behavior (keep smallest k by removing largest)
        // We'll just collect all visited and sort at the end for simplicity
        let mut all_visited_with_dist: Vec<(usize, f32)> = Vec::new();

        // Start from entry point
        let entry_dist =
            distance(metric, query, &vectors[self.vector_ids[entry]].vector).unwrap_or(f32::MAX);
        candidates.push(HeapEntry {
            local_id: entry,
            distance: entry_dist,
        });
        visited.insert(entry);
        all_visited_with_dist.push((entry, entry_dist));

        let mut iterations = 0;
        let max_iterations = ef * 3; // Prevent infinite loops

        while let Some(current) = candidates.pop() {
            iterations += 1;
            if iterations > max_iterations {
                break;
            }

            // Explore neighbors
            for &neighbor_lid in &self.edges[current.local_id] {
                if visited.insert(neighbor_lid) {
                    let dist = distance(
                        metric,
                        query,
                        &vectors[self.vector_ids[neighbor_lid]].vector,
                    )
                    .unwrap_or(f32::MAX);
                    candidates.push(HeapEntry {
                        local_id: neighbor_lid,
                        distance: dist,
                    });
                    all_visited_with_dist.push((neighbor_lid, dist));
                }
            }
        }

        // Convert to global IDs and return sorted top-k
        let mut result_vec: Vec<(usize, f32)> = all_visited_with_dist
            .into_iter()
            .map(|(lid, dist)| (self.vector_ids[lid], dist))
            .collect();
        result_vec.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        result_vec.truncate(k);

        Ok(result_vec)
    }

    /// Linear scan search for small buckets (below graph threshold).
    fn linear_search(
        &self,
        query: &[f32],
        k: usize,
        vectors: &[VectorEntry],
        metric: DistanceMetric,
    ) -> Result<Vec<(usize, f32)>> {
        let mut distances: Vec<(usize, f32)> = self
            .vector_ids
            .iter()
            .map(|&gid| {
                let dist = distance(metric, query, &vectors[gid].vector).unwrap_or(f32::MAX);
                (gid, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);

        Ok(distances)
    }
}

/// FUSION index: Fast Unified Search with Intelligent Orchestrated Navigation.
///
/// Combines LSH bucketing with mini-graph navigation for efficient approximate
/// nearest neighbor search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionIndex {
    /// Distance metric for similarity computation.
    metric: DistanceMetric,
    /// Configuration parameters.
    config: FusionConfig,
    /// LSH hasher (None until first vector determines dimension).
    hasher: Option<LshHasher>,
    /// Buckets, each containing a mini-graph.
    buckets: Vec<MiniGraph>,
    /// All vectors in insertion order (shared storage).
    vectors: Vec<VectorEntry>,
    /// Tracked dimension (set by first inserted vector).
    dimension: Option<usize>,
}

impl FusionIndex {
    /// Creates a new FUSION index with the given metric and configuration.
    pub fn new(metric: DistanceMetric, config: FusionConfig) -> Self {
        Self {
            metric,
            config,
            hasher: None,
            buckets: Vec::new(),
            vectors: Vec::new(),
            dimension: None,
        }
    }

    /// Creates a new FUSION index with default configuration.
    pub fn with_defaults(metric: DistanceMetric) -> Self {
        Self::new(metric, FusionConfig::default())
    }

    /// Returns the dimension of vectors in this index.
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Validates that a vector has the expected dimension.
    fn validate_dimension(&self, vector: &[f32]) -> Result<(), FusionError> {
        if vector.is_empty() {
            return Err(FusionError::ZeroDimension);
        }
        if let Some(expected) = self.dimension {
            if vector.len() != expected {
                return Err(FusionError::DimensionMismatch {
                    expected,
                    actual: vector.len(),
                });
            }
        }
        Ok(())
    }

    /// Initializes the LSH hasher when dimension is first known.
    fn init_hasher(&mut self, dimension: usize) {
        if self.hasher.is_none() {
            let hasher = LshHasher::new(dimension, self.config.n_hyperplanes, self.config.seed);
            let n_buckets = hasher.n_buckets();

            self.hasher = Some(hasher);
            self.buckets = (0..n_buckets)
                .map(|_| MiniGraph::new(self.config.mini_graph_m))
                .collect();
            self.dimension = Some(dimension);
        }
    }

    /// Saves the index to a JSON file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)?;
        Ok(())
    }

    /// Loads an index from a JSON file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let index = serde_json::from_reader(reader)?;
        Ok(index)
    }
}

impl VectorIndex for FusionIndex {
    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn insert(&mut self, id: usize, vector: Vector) -> Result<()> {
        self.validate_dimension(&vector)?;

        // Initialize on first insert
        if self.hasher.is_none() {
            self.init_hasher(vector.len());
        }

        let hasher = self.hasher.as_ref().unwrap();

        // Store vector globally
        let global_idx = self.vectors.len();
        self.vectors.push(VectorEntry {
            id,
            vector: vector.clone(),
        });

        // Hash to bucket and insert into mini-graph
        let bucket_id = hasher.hash(&vector);
        self.buckets[bucket_id].insert(global_idx, &vector, &self.vectors, self.metric)?;

        Ok(())
    }

    fn search(&self, query: &Vector, limit: usize) -> Result<Vec<ScoredPoint>> {
        if self.is_empty() {
            return Err(FusionError::EmptyIndex.into());
        }

        self.validate_dimension(query)?;

        let hasher = self.hasher.as_ref().unwrap();

        // Stage 1: Hash query to primary bucket
        let primary = hasher.hash(query);

        // Stage 2: Get probe buckets (primary + Hamming-1 neighbors)
        let probe_buckets = hasher.get_probe_buckets(primary, self.config.n_probes);

        // Stage 3: Search each probed bucket
        let mut all_candidates = Vec::new();
        for bucket_id in probe_buckets {
            let bucket = &self.buckets[bucket_id];
            if bucket.is_empty() {
                continue;
            }

            // Use graph search for large buckets, linear for small
            let candidates = if bucket.len() >= self.config.min_bucket_for_graph {
                bucket.search(
                    query,
                    limit * 2,
                    self.config.mini_graph_ef,
                    &self.vectors,
                    self.metric,
                )?
            } else {
                bucket.linear_search(query, limit * 2, &self.vectors, self.metric)?
            };

            all_candidates.extend(candidates);
        }

        // Stage 4: Deduplicate, sort, and take top-k
        let mut seen = HashSet::new();
        let mut unique_candidates: Vec<(usize, f32)> = all_candidates
            .into_iter()
            .filter(|(global_idx, _)| seen.insert(*global_idx))
            .collect();

        unique_candidates
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        unique_candidates.truncate(limit);

        // Convert to ScoredPoint with original IDs
        Ok(unique_candidates
            .into_iter()
            .map(|(global_idx, dist)| ScoredPoint::new(self.vectors[global_idx].id, dist))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsh_similar_vectors_same_bucket() {
        let hasher = LshHasher::new(4, 8, 42);

        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let v2 = vec![1.1, 2.1, 3.1, 4.1]; // Very similar
        let v3 = vec![-1.0, -2.0, -3.0, -4.0]; // Opposite direction

        let h1 = hasher.hash(&v1);
        let h2 = hasher.hash(&v2);
        let h3 = hasher.hash(&v3);

        // Similar vectors should hash to same bucket
        assert_eq!(h1, h2, "Similar vectors should hash to same bucket");

        // Opposite vectors should hash to different bucket
        assert_ne!(h1, h3, "Opposite vectors should hash to different buckets");
    }

    #[test]
    fn test_lsh_probe_buckets() {
        let hasher = LshHasher::new(4, 4, 42);

        let probes = hasher.get_probe_buckets(0b1010, 3);
        assert_eq!(probes.len(), 3);
        assert_eq!(probes[0], 0b1010); // Primary
        assert_eq!(probes[1], 0b1011); // Flip bit 0
        assert_eq!(probes[2], 0b1000); // Flip bit 1
    }

    #[test]
    fn test_fusion_basic_search() {
        // Use fewer hyperplanes (= fewer buckets) for small test dataset
        // to ensure vectors can be found with multi-probe
        let mut index = FusionIndex::new(
            DistanceMetric::Euclidean,
            FusionConfig {
                n_hyperplanes: 3,        // Only 8 buckets
                n_probes: 8,             // Probe all buckets
                min_bucket_for_graph: 1, // Allow graph even for tiny buckets
                ..Default::default()
            },
        );

        // Insert some vectors with clear distance relationships
        index.insert(0, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        index.insert(1, vec![2.0, 2.0, 2.0, 2.0]).unwrap();
        index.insert(2, vec![3.0, 3.0, 3.0, 3.0]).unwrap();
        index.insert(3, vec![100.0, 100.0, 100.0, 100.0]).unwrap(); // Very far

        // Search for vector near id=0
        let query = vec![1.0, 1.0, 1.0, 1.0];
        let results = index.search(&query, 3).unwrap();

        // Should find at least some results
        assert!(!results.is_empty(), "Should find some results");

        // First result should be id=0 (exact match with distance 0)
        assert_eq!(results[0].id, 0, "First result should be the exact match");
        assert!(
            results[0].distance < 0.001,
            "First result should have near-zero distance"
        );

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(
                results[i].distance >= results[i - 1].distance,
                "Results should be sorted by distance"
            );
        }
    }

    #[test]
    fn test_fusion_recall_with_more_vectors() {
        let mut index = FusionIndex::new(
            DistanceMetric::Euclidean,
            FusionConfig {
                n_hyperplanes: 6,
                n_probes: 8,
                ..Default::default()
            },
        );

        // Insert 100 vectors
        let mut rng = StdRng::seed_from_u64(123);
        for i in 0..100 {
            let vec: Vec<f32> = (0..32).map(|_| rng.gen::<f32>()).collect();
            index.insert(i, vec).unwrap();
        }

        // Generate a query and find ground truth
        let query: Vec<f32> = (0..32).map(|_| rng.gen::<f32>()).collect();

        // Compute all distances for ground truth
        let mut ground_truth: Vec<(usize, f32)> = index
            .vectors
            .iter()
            .map(|e| {
                let d = distance(DistanceMetric::Euclidean, &query, &e.vector).unwrap();
                (e.id, d)
            })
            .collect();
        ground_truth.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let true_top_10: HashSet<usize> = ground_truth.iter().take(10).map(|(id, _)| *id).collect();

        // Search
        let results = index.search(&query, 10).unwrap();
        let found: HashSet<usize> = results.iter().map(|r| r.id).collect();

        // Calculate recall
        let recall = found.intersection(&true_top_10).count() as f32 / 10.0;

        assert!(
            recall >= 0.7,
            "Recall should be at least 70%, got {:.1}%",
            recall * 100.0
        );
    }

    #[test]
    fn test_fusion_save_load() {
        let mut index = FusionIndex::with_defaults(DistanceMetric::Euclidean);
        index.insert(0, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        index.insert(1, vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        let temp_path = std::env::temp_dir().join("fusion_test_index.json");
        index.save(&temp_path).unwrap();

        let loaded = FusionIndex::load(&temp_path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.dimension(), Some(4));

        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_fusion_dimension_mismatch() {
        let mut index = FusionIndex::with_defaults(DistanceMetric::Euclidean);
        index.insert(0, vec![1.0, 2.0, 3.0]).unwrap();

        let result = index.insert(1, vec![1.0, 2.0]); // Wrong dimension
        assert!(result.is_err());
    }

    #[test]
    fn test_fusion_empty_search() {
        let index = FusionIndex::with_defaults(DistanceMetric::Euclidean);
        let result = index.search(&vec![1.0, 2.0, 3.0], 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_mini_graph_navigation() {
        let mut graph = MiniGraph::new(4);
        let mut vectors = Vec::new();

        // Create some test vectors
        for i in 0..20 {
            vectors.push(VectorEntry {
                id: i,
                vector: vec![i as f32, 0.0, 0.0, 0.0],
            });
        }

        // Insert into graph
        for i in 0..20 {
            graph
                .insert(i, &vectors[i].vector, &vectors, DistanceMetric::Euclidean)
                .unwrap();
        }

        // Search for vector near 10
        let query = vec![10.5, 0.0, 0.0, 0.0];
        let results = graph
            .search(&query, 3, 16, &vectors, DistanceMetric::Euclidean)
            .unwrap();

        assert!(!results.is_empty());
        // Best result should be id 10 or 11
        assert!(results[0].0 == 10 || results[0].0 == 11);
    }

    #[test]
    fn test_cosine_distance() {
        let mut index = FusionIndex::with_defaults(DistanceMetric::Cosine);

        // Insert unit vectors
        index.insert(0, vec![1.0, 0.0, 0.0]).unwrap();
        index.insert(1, vec![0.0, 1.0, 0.0]).unwrap();
        index.insert(2, vec![0.707, 0.707, 0.0]).unwrap(); // 45 degrees

        // Query with something between 0 and 2
        let query = vec![0.9, 0.1, 0.0];
        let results = index.search(&query, 2).unwrap();

        // Should prefer vectors aligned with the query
        assert_eq!(results[0].id, 0); // Most aligned with x-axis
    }
}
