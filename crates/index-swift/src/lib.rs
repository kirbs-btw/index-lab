//! SWIFT: Sparse-Weighted Index with Fast Traversal
//!
//! A novel vector index that combines:
//! - **LSH Bucketing** for O(1) candidate generation
//! - **Mini-Graphs** per bucket for O(log b) navigation
//! - **Multi-Probe** for improved recall
//!
//! This addresses the O(n) issues found in SEER, LIM, and Hybrid indexes.

use anyhow::{ensure, Result};
use index_core::{
    distance, load_index, save_index, validate_dimension, DistanceMetric, ScoredPoint, Vector,
    VectorIndex,
};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::path::Path;
use thiserror::Error;

/// Errors specific to the SWIFT index.
#[derive(Debug, Error)]
pub enum SwiftError {
    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("index is empty, cannot search")]
    EmptyIndex,
    #[error("index has not been built yet")]
    NotBuilt,
}

/// Configuration for the SWIFT index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwiftConfig {
    /// Number of hyperplanes for LSH (creates 2^n_hyperplanes buckets)
    pub n_hyperplanes: usize,
    /// Number of buckets to probe during search (higher = better recall, slower)
    pub n_probes: usize,
    /// Maximum edges per node in mini-graphs
    pub mini_graph_m: usize,
    /// Candidate list size during mini-graph construction
    pub ef_construction: usize,
    /// Candidate list size during mini-graph search
    pub ef_search: usize,
    /// Minimum bucket size before building a mini-graph (below this, use linear scan)
    pub min_bucket_size: usize,
}

impl Default for SwiftConfig {
    fn default() -> Self {
        Self {
            n_hyperplanes: 8,    // 256 buckets
            n_probes: 4,         // Check 4 buckets per query
            mini_graph_m: 8,     // 8 edges per node in mini-graphs
            ef_construction: 50, // Construction quality
            ef_search: 32,       // Search quality
            min_bucket_size: 16, // Linear scan for very small buckets
        }
    }
}

/// A stored vector with its metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VectorEntry {
    id: usize,
    vector: Vector,
    bucket: usize,
}

/// LSH bucketer using SimHash (random hyperplane projections).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LshBucketer {
    /// Random hyperplanes for projection (each is a unit vector)
    hyperplanes: Vec<Vector>,
    /// Number of buckets (2^n_hyperplanes)
    n_buckets: usize,
}

impl LshBucketer {
    /// Creates a new LSH bucketer with random hyperplanes.
    fn new(dimension: usize, n_hyperplanes: usize, rng: &mut impl Rng) -> Self {
        let n_buckets = 1 << n_hyperplanes;
        let hyperplanes: Vec<Vector> = (0..n_hyperplanes)
            .map(|_| {
                // Generate random unit vector
                let mut plane: Vector = (0..dimension).map(|_| rng.gen::<f32>() - 0.5).collect();
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
            n_buckets,
        }
    }

    /// Computes the hash bucket for a vector.
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

    /// Returns the primary bucket and probe buckets (neighbors in Hamming space).
    fn get_probe_buckets(&self, vector: &[f32], n_probes: usize) -> Vec<usize> {
        let primary = self.hash(vector);
        let mut buckets = vec![primary];

        // Add buckets that differ by 1 bit (Hamming distance 1)
        for bit in 0..self.hyperplanes.len() {
            if buckets.len() >= n_probes {
                break;
            }
            let neighbor = primary ^ (1 << bit);
            if neighbor < self.n_buckets {
                buckets.push(neighbor);
            }
        }

        // If we still need more, add 2-bit differences
        if buckets.len() < n_probes {
            'outer: for bit1 in 0..self.hyperplanes.len() {
                for bit2 in (bit1 + 1)..self.hyperplanes.len() {
                    if buckets.len() >= n_probes {
                        break 'outer;
                    }
                    let neighbor = primary ^ (1 << bit1) ^ (1 << bit2);
                    if neighbor < self.n_buckets && !buckets.contains(&neighbor) {
                        buckets.push(neighbor);
                    }
                }
            }
        }

        buckets.truncate(n_probes);
        buckets
    }
}

/// Min-heap entry for distance comparisons.
#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    local_idx: usize, // Index within the bucket's local_indices
    distance: f32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.local_idx == other.local_idx
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap (smallest distance first)
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// A mini-graph within a single bucket for efficient local search.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MiniGraph {
    /// Global indices of vectors in this bucket
    global_indices: Vec<usize>,
    /// Adjacency list: edges[local_idx] = list of local neighbor indices
    edges: Vec<Vec<usize>>,
    /// Entry point for search (local index of a well-connected node)
    entry_point: Option<usize>,
    /// Whether the graph has been built
    is_built: bool,
}

impl MiniGraph {
    fn new() -> Self {
        Self {
            global_indices: Vec::new(),
            edges: Vec::new(),
            entry_point: None,
            is_built: false,
        }
    }

    /// Adds a vector's global index to this bucket.
    fn add(&mut self, global_idx: usize) {
        self.global_indices.push(global_idx);
        self.edges.push(Vec::new());
        self.is_built = false;
    }

    /// Removes a vector's global index from this bucket.
    fn remove(&mut self, global_idx: usize) {
        if let Some(local_idx) = self.global_indices.iter().position(|&idx| idx == global_idx) {
            self.global_indices.remove(local_idx);
            self.edges.remove(local_idx);
            
            // Update edges: remove references to removed node and adjust indices
            for edges_list in &mut self.edges {
                edges_list.retain(|&idx| idx != local_idx);
                // Decrement indices greater than removed index
                for idx in edges_list.iter_mut() {
                    if *idx > local_idx {
                        *idx -= 1;
                    }
                }
            }
            
            // Update entry point if needed
            if self.entry_point == Some(local_idx) {
                self.entry_point = if !self.global_indices.is_empty() { Some(0) } else { None };
            } else if let Some(ref mut entry) = self.entry_point {
                if *entry > local_idx {
                    *entry -= 1;
                }
            }
            
            self.is_built = false;
        }
    }

    /// Updates global indices after a removal at a higher index.
    fn update_indices_after_removal(&mut self, removed_idx: usize) {
        // Update global indices
        for global_idx in &mut self.global_indices {
            if *global_idx > removed_idx {
                *global_idx -= 1;
            }
        }
    }

    /// Returns the number of vectors in this bucket.
    fn len(&self) -> usize {
        self.global_indices.len()
    }

    /// Builds the mini-graph structure using a simplified HNSW approach.
    fn build(
        &mut self,
        vectors: &[VectorEntry],
        metric: DistanceMetric,
        m: usize,
        ef_construction: usize,
    ) -> Result<()> {
        if self.global_indices.is_empty() {
            return Ok(());
        }

        // Reset edges
        self.edges = vec![Vec::new(); self.global_indices.len()];

        if self.global_indices.len() == 1 {
            self.entry_point = Some(0);
            self.is_built = true;
            return Ok(());
        }

        // Build graph incrementally
        self.entry_point = Some(0);

        for local_idx in 1..self.global_indices.len() {
            let global_idx = self.global_indices[local_idx];
            let query = &vectors[global_idx].vector;

            // Find nearest neighbors using current graph
            let neighbors = self.search_internal(
                query,
                vectors,
                metric,
                ef_construction.min(local_idx),
                local_idx,
            )?;

            // Connect to top-m neighbors
            for &neighbor_local_idx in neighbors.iter().take(m) {
                // Add bidirectional edge
                self.edges[local_idx].push(neighbor_local_idx);
                self.edges[neighbor_local_idx].push(local_idx);

                // Prune if over m connections
                if self.edges[neighbor_local_idx].len() > m * 2 {
                    self.edges[neighbor_local_idx].truncate(m * 2);
                }
            }
        }

        self.is_built = true;
        Ok(())
    }

    /// Internal search used for graph construction (searches only up to max_local_idx).
    fn search_internal(
        &self,
        query: &[f32],
        vectors: &[VectorEntry],
        metric: DistanceMetric,
        k: usize,
        max_local_idx: usize,
    ) -> Result<Vec<usize>> {
        if self.global_indices.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let entry = self.entry_point.unwrap_or(0);
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();

        // Initialize with entry point
        let entry_global = self.global_indices[entry];
        let entry_dist = distance(metric, query, &vectors[entry_global].vector)?;
        candidates.push(HeapEntry {
            local_idx: entry,
            distance: entry_dist,
        });
        visited.insert(entry);

        let mut results: Vec<HeapEntry> = Vec::new();

        while let Some(current) = candidates.pop() {
            // Add to results
            results.push(current);
            if results.len() >= k * 2 {
                break;
            }

            // Explore neighbors
            for &neighbor_local in &self.edges[current.local_idx] {
                if neighbor_local >= max_local_idx || visited.contains(&neighbor_local) {
                    continue;
                }
                visited.insert(neighbor_local);

                let neighbor_global = self.global_indices[neighbor_local];
                let dist = distance(metric, query, &vectors[neighbor_global].vector)?;
                candidates.push(HeapEntry {
                    local_idx: neighbor_local,
                    distance: dist,
                });
            }
        }

        // Sort by distance and return top-k local indices
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(Ordering::Equal)
        });
        Ok(results.into_iter().take(k).map(|e| e.local_idx).collect())
    }

    /// Searches the mini-graph for k-nearest neighbors.
    fn search(
        &self,
        query: &[f32],
        vectors: &[VectorEntry],
        metric: DistanceMetric,
        k: usize,
        ef: usize,
    ) -> Result<Vec<(usize, f32)>> {
        if self.global_indices.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        // Fall back to linear scan if graph not built or too small
        if !self.is_built || self.global_indices.len() <= 2 {
            return self.linear_search(query, vectors, metric, k);
        }

        let entry = self.entry_point.unwrap_or(0);
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();

        // Initialize with entry point
        let entry_global = self.global_indices[entry];
        let entry_dist = distance(metric, query, &vectors[entry_global].vector)?;
        candidates.push(HeapEntry {
            local_idx: entry,
            distance: entry_dist,
        });
        visited.insert(entry);

        let mut results: Vec<(usize, f32)> = Vec::new();

        let max_visited = ef.max(k) * 3;
        while let Some(current) = candidates.pop() {
            if visited.len() > max_visited {
                break;
            }

            // Only add to results if we have room or it's better than the worst
            if results.len() < k {
                let global_idx = self.global_indices[current.local_idx];
                results.push((global_idx, current.distance));
            } else if current.distance < results.last().unwrap().1 {
                let global_idx = self.global_indices[current.local_idx];
                results.pop();
                results.push((global_idx, current.distance));
                results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            }

            // Explore neighbors
            for &neighbor_local in &self.edges[current.local_idx] {
                if visited.contains(&neighbor_local) {
                    continue;
                }
                visited.insert(neighbor_local);

                let neighbor_global = self.global_indices[neighbor_local];
                let dist = distance(metric, query, &vectors[neighbor_global].vector)?;
                candidates.push(HeapEntry {
                    local_idx: neighbor_local,
                    distance: dist,
                });
            }
        }

        // Sort by distance
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        Ok(results)
    }

    /// Linear search within this bucket (fallback for small buckets).
    fn linear_search(
        &self,
        query: &[f32],
        vectors: &[VectorEntry],
        metric: DistanceMetric,
        k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        let mut results: Vec<(usize, f32)> = self
            .global_indices
            .iter()
            .map(|&global_idx| {
                let dist = distance(metric, query, &vectors[global_idx].vector).unwrap_or(f32::MAX);
                (global_idx, dist)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results.truncate(k);
        Ok(results)
    }
}

/// SWIFT Index: Fast vector search using LSH bucketing with mini-graphs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwiftIndex {
    metric: DistanceMetric,
    dimension: Option<usize>,
    config: SwiftConfig,
    /// All stored vectors
    vectors: Vec<VectorEntry>,
    /// LSH bucketer (initialized on first insert)
    bucketer: Option<LshBucketer>,
    /// Mini-graphs per bucket
    buckets: Vec<MiniGraph>,
    /// Whether the index has been fully built
    is_built: bool,
    /// RNG seed for reproducibility
    seed: u64,
}

impl SwiftIndex {
    /// Creates a new SWIFT index with the given metric and configuration.
    pub fn new(metric: DistanceMetric, config: SwiftConfig) -> Self {
        let n_buckets = 1 << config.n_hyperplanes;
        Self {
            metric,
            dimension: None,
            config,
            vectors: Vec::new(),
            bucketer: None,
            buckets: (0..n_buckets).map(|_| MiniGraph::new()).collect(),
            is_built: false,
            seed: 42,
        }
    }

    /// Creates a new SWIFT index with default configuration.
    pub fn with_defaults(metric: DistanceMetric) -> Self {
        Self::new(metric, SwiftConfig::default())
    }

    /// Sets the RNG seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Returns the dimensionality of the index.
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Returns the configuration.
    pub fn config(&self) -> &SwiftConfig {
        &self.config
    }

    fn validate_dimension(&self, vector: &[f32]) -> Result<()> {
        if let Some(expected) = self.dimension {
            validate_dimension(Some(expected), vector.len()).map_err(|_| {
                SwiftError::DimensionMismatch {
                    expected,
                    actual: vector.len(),
                }
            })?;
        }
        Ok(())
    }

    /// Initializes the LSH bucketer if not already done.
    fn ensure_bucketer(&mut self, dimension: usize) {
        if self.bucketer.is_none() {
            let mut rng = StdRng::seed_from_u64(self.seed);
            self.bucketer = Some(LshBucketer::new(
                dimension,
                self.config.n_hyperplanes,
                &mut rng,
            ));
        }
    }

    /// Builds the mini-graphs for all buckets.
    fn build_mini_graphs(&mut self) -> Result<()> {
        for bucket in &mut self.buckets {
            if bucket.len() >= self.config.min_bucket_size {
                bucket.build(
                    &self.vectors,
                    self.metric,
                    self.config.mini_graph_m,
                    self.config.ef_construction,
                )?;
            }
        }
        self.is_built = true;
        Ok(())
    }

    /// Saves the index to a JSON file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        save_index(self, path)
    }

    /// Loads an index from a JSON file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        load_index(path)
    }

    /// Returns bucket statistics for debugging.
    pub fn bucket_stats(&self) -> (usize, usize, f64) {
        let sizes: Vec<usize> = self.buckets.iter().map(|b| b.len()).collect();
        let max = *sizes.iter().max().unwrap_or(&0);
        let non_empty = sizes.iter().filter(|&&s| s > 0).count();
        let avg = if non_empty > 0 {
            sizes.iter().sum::<usize>() as f64 / non_empty as f64
        } else {
            0.0
        };
        (non_empty, max, avg)
    }
}

impl VectorIndex for SwiftIndex {
    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn build(&mut self, data: impl IntoIterator<Item = (usize, Vector)>) -> Result<()> {
        // Insert all vectors
        for (id, vector) in data {
            // Validate and set dimension
            self.validate_dimension(&vector)?;
            if self.dimension.is_none() {
                self.dimension = Some(vector.len());
            }

            // Ensure bucketer exists
            self.ensure_bucketer(vector.len());

            // Compute bucket assignment
            let bucket_idx = self.bucketer.as_ref().unwrap().hash(&vector);
            let global_idx = self.vectors.len();

            // Store vector
            self.vectors.push(VectorEntry {
                id,
                vector,
                bucket: bucket_idx,
            });

            // Add to bucket
            self.buckets[bucket_idx].add(global_idx);
        }

        // Build mini-graphs
        self.build_mini_graphs()?;
        Ok(())
    }

    fn insert(&mut self, id: usize, vector: Vector) -> Result<()> {
        self.validate_dimension(&vector)?;
        if self.dimension.is_none() {
            self.dimension = Some(vector.len());
        }

        // Ensure bucketer exists
        self.ensure_bucketer(vector.len());

        // Compute bucket assignment
        let bucket_idx = self.bucketer.as_ref().unwrap().hash(&vector);
        let global_idx = self.vectors.len();

        // Store vector
        self.vectors.push(VectorEntry {
            id,
            vector,
            bucket: bucket_idx,
        });

        // Add to bucket
        self.buckets[bucket_idx].add(global_idx);

        // Mark as needing rebuild (incremental updates could be added later)
        self.is_built = false;

        Ok(())
    }

    fn search(&self, query: &Vector, limit: usize) -> Result<Vec<ScoredPoint>> {
        ensure!(limit > 0, "limit must be greater than zero");
        ensure!(!self.vectors.is_empty(), SwiftError::EmptyIndex);
        self.validate_dimension(query)?;

        let bucketer = self
            .bucketer
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("index not initialized"))?;

        // Stage 1: Get probe buckets (O(1))
        let probe_buckets = bucketer.get_probe_buckets(query, self.config.n_probes);

        // Stage 2: Search each bucket (O(probes × log b))
        let mut all_candidates: Vec<(usize, f32)> = Vec::new();

        for bucket_idx in probe_buckets {
            let bucket = &self.buckets[bucket_idx];
            if bucket.len() == 0 {
                continue;
            }

            // Search the mini-graph or do linear scan
            let candidates = if bucket.is_built && bucket.len() >= self.config.min_bucket_size {
                bucket.search(
                    query,
                    &self.vectors,
                    self.metric,
                    limit * 2,
                    self.config.ef_search,
                )?
            } else {
                bucket.linear_search(query, &self.vectors, self.metric, limit * 2)?
            };

            all_candidates.extend(candidates);
        }

        // Stage 3: Dedup, sort, and take top-k (O(k log k))
        // Dedup by global index
        let mut seen = HashSet::new();
        all_candidates.retain(|(idx, _)| seen.insert(*idx));

        // Sort by distance
        all_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

        // Convert to ScoredPoints
        let results: Vec<ScoredPoint> = all_candidates
            .into_iter()
            .take(limit)
            .map(|(global_idx, dist)| ScoredPoint::new(self.vectors[global_idx].id, dist))
            .collect();

        Ok(results)
    }

    fn delete(&mut self, id: usize) -> Result<bool> {
        // Find the vector index by ID
        let idx_opt = self.vectors.iter().position(|entry| entry.id == id);
        
        if let Some(idx) = idx_opt {
            let removed_entry = self.vectors.remove(idx);
            let bucket_idx = removed_entry.bucket;
            
            // Remove from bucket's mini-graph
            if bucket_idx < self.buckets.len() {
                self.buckets[bucket_idx].remove(idx);
                
                // Update indices in all buckets (decrement indices > idx)
                for bucket in &mut self.buckets {
                    bucket.update_indices_after_removal(idx);
                }
            }
            
            // Mark as needing rebuild
            self.is_built = false;
            
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn update(&mut self, id: usize, vector: Vector) -> Result<bool> {
        self.validate_dimension(&vector)?;
        
        // Find the vector index by ID
        let idx_opt = self.vectors.iter().position(|entry| entry.id == id);
        
        if let Some(idx) = idx_opt {
            let old_entry = &self.vectors[idx];
            let old_bucket = old_entry.bucket;
            
            // Compute new bucket
            self.ensure_bucketer(vector.len());
            let new_bucket = self.bucketer.as_ref().unwrap().hash(&vector);
            
            // Update vector
            self.vectors[idx] = VectorEntry {
                id,
                vector,
                bucket: new_bucket,
            };
            
            // Update buckets if bucket changed
            if old_bucket != new_bucket {
                // Remove from old bucket
                if old_bucket < self.buckets.len() {
                    self.buckets[old_bucket].remove(idx);
                }
                
                // Add to new bucket
                if new_bucket < self.buckets.len() {
                    self.buckets[new_bucket].add(idx);
                }
            }
            
            // Mark as needing rebuild
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
    fn test_lsh_bucketer_deterministic() {
        let mut rng = StdRng::seed_from_u64(42);
        let bucketer = LshBucketer::new(4, 4, &mut rng);

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0, 0.0];

        assert_eq!(bucketer.hash(&v1), bucketer.hash(&v2));
    }

    #[test]
    fn test_lsh_similar_vectors_likely_same_bucket() {
        let mut rng = StdRng::seed_from_u64(42);
        let bucketer = LshBucketer::new(4, 4, &mut rng);

        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.99, 0.01, 0.0, 0.0];

        // Similar vectors should often (but not always) hash to same bucket
        // At minimum, they should be in each other's probe set
        let probes1 = bucketer.get_probe_buckets(&v1, 4);
        let bucket2 = bucketer.hash(&v2);

        assert!(probes1.contains(&bucket2) || bucketer.hash(&v1) == bucket2);
    }

    #[test]
    fn test_insert_and_search() {
        let mut index = SwiftIndex::with_defaults(DistanceMetric::Euclidean).with_seed(42);

        // Insert some vectors
        index.insert(0, vec![0.0, 0.0]).unwrap();
        index.insert(1, vec![1.0, 0.0]).unwrap();
        index.insert(2, vec![0.0, 1.0]).unwrap();
        index.insert(3, vec![1.0, 1.0]).unwrap();
        index.insert(4, vec![0.5, 0.5]).unwrap();

        // Build mini-graphs
        index.build_mini_graphs().unwrap();

        let result = index.search(&vec![0.0, 0.0], 2).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result[0].id, 0); // Closest should be itself
    }

    #[test]
    fn test_build_and_search() {
        let mut index = SwiftIndex::with_defaults(DistanceMetric::Euclidean).with_seed(42);

        let data = vec![
            (0, vec![0.0, 0.0]),
            (1, vec![1.0, 0.0]),
            (2, vec![0.0, 1.0]),
            (3, vec![1.0, 1.0]),
            (4, vec![0.5, 0.5]),
        ];

        index.build(data).unwrap();

        let result = index.search(&vec![0.0, 0.0], 2).unwrap();
        assert_eq!(result[0].id, 0);
        assert!(result[0].distance < 0.1);
    }

    #[test]
    fn test_larger_dataset() {
        let mut index = SwiftIndex::with_defaults(DistanceMetric::Euclidean).with_seed(42);

        // Create a larger dataset
        let mut data = Vec::new();
        for i in 0..1000 {
            let x = (i % 10) as f32;
            let y = (i / 10) as f32;
            data.push((i, vec![x, y]));
        }

        index.build(data).unwrap();

        // Search for something close to the center
        let result = index.search(&vec![5.0, 5.0], 5).unwrap();
        assert_eq!(result.len(), 5);

        // The closest should be (5, 5) which is id 55
        assert_eq!(result[0].id, 55);
        assert!(result[0].distance < 0.1);
    }

    #[test]
    fn test_empty_index_search_fails() {
        let index = SwiftIndex::with_defaults(DistanceMetric::Euclidean);
        assert!(index.search(&vec![0.0, 0.0], 1).is_err());
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut index = SwiftIndex::with_defaults(DistanceMetric::Euclidean);
        index.insert(0, vec![1.0, 2.0]).unwrap();
        assert!(index.insert(1, vec![1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn test_save_and_load() {
        use std::env::temp_dir;

        let mut original = SwiftIndex::with_defaults(DistanceMetric::Euclidean).with_seed(42);

        let data = vec![
            (0, vec![0.0, 0.0]),
            (1, vec![1.0, 0.0]),
            (2, vec![0.0, 1.0]),
        ];
        original.build(data).unwrap();

        let mut path = temp_dir();
        path.push(format!("test_swift_index_{}.json", std::process::id()));

        original.save(&path).unwrap();
        let loaded = SwiftIndex::load(&path).unwrap();

        assert_eq!(loaded.metric(), original.metric());
        assert_eq!(loaded.len(), original.len());
        assert_eq!(loaded.dimension(), original.dimension());

        // Verify search results match
        let query = vec![0.0, 0.0];
        let orig_results = original.search(&query, 2).unwrap();
        let loaded_results = loaded.search(&query, 2).unwrap();

        assert_eq!(orig_results.len(), loaded_results.len());
        assert_eq!(orig_results[0].id, loaded_results[0].id);

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_bucket_stats() {
        let mut index = SwiftIndex::with_defaults(DistanceMetric::Euclidean).with_seed(42);

        let mut data = Vec::new();
        for i in 0..100 {
            let x = (i % 10) as f32 / 10.0;
            let y = (i / 10) as f32 / 10.0;
            data.push((i, vec![x, y]));
        }

        index.build(data).unwrap();

        let (non_empty, max, avg) = index.bucket_stats();
        assert!(non_empty > 0);
        assert!(max > 0);
        assert!(avg > 0.0);
    }

    #[test]
    fn test_cosine_metric() {
        let mut index = SwiftIndex::with_defaults(DistanceMetric::Cosine).with_seed(42);

        // Normalized vectors for cosine
        let data = vec![
            (0, vec![1.0, 0.0]),
            (1, vec![0.0, 1.0]),
            (2, vec![0.707, 0.707]),
        ];

        index.build(data).unwrap();

        let result = index.search(&vec![1.0, 0.0], 3).unwrap();
        assert_eq!(result[0].id, 0); // Should match itself
    }
}
