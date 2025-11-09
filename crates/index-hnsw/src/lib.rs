//! Hierarchical Navigable Small World (HNSW) index implementation.
//!
//! This is a standard HNSW implementation based on the paper:
//! "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
//! by Yu. A. Malkov and D. A. Yashunin

use anyhow::{ensure, Context, Result};
use index_core::{distance, DistanceMetric, ScoredPoint, Vector, VectorIndex};
use rand::distributions::Distribution;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum HnswError {
    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("index is empty, cannot search")]
    EmptyIndex,
}

/// Node in the HNSW graph
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Node {
    id: usize,
    vector: Vector,
    /// Connections per layer: layers[0] is layer 0 (all nodes), layers[i] is layer i
    layers: Vec<Vec<usize>>, // neighbors per layer
}

impl Node {
    fn new(id: usize, vector: Vector, max_layer: usize) -> Self {
        Self {
            id,
            vector,
            layers: vec![Vec::new(); max_layer + 1],
        }
    }

    fn layer_count(&self) -> usize {
        self.layers.len()
    }

    fn neighbors_at_layer(&self, layer: usize) -> &[usize] {
        &self.layers[layer]
    }

    fn add_neighbor(&mut self, layer: usize, neighbor_id: usize) {
        if layer < self.layers.len() {
            self.layers[layer].push(neighbor_id);
        }
    }
}

/// Min-heap entry for distance comparisons (we want closest first)
#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    id: usize,
    distance: f32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse order for min-heap (smallest distance first)
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// HNSW index configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Maximum number of connections per node per layer
    pub m_max: usize,
    /// Size of candidate list during construction
    pub ef_construction: usize,
    /// Size of candidate list during search
    pub ef_search: usize,
    /// Layer normalization factor (typically 1/ln(2) ≈ 1.4427)
    pub ml: f64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m_max: 16,
            ef_construction: 200,
            ef_search: 50,
            ml: 1.0 / 2.0_f64.ln(),
        }
    }
}

/// HNSW index implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswIndex {
    metric: DistanceMetric,
    dimension: Option<usize>,
    config: HnswConfig,
    nodes: Vec<Node>,
    entry_point: Option<usize>, // ID of the entry point (highest layer node)
    max_layer: usize,
}

impl HnswIndex {
    /// Creates a new HNSW index with the given metric and configuration
    pub fn new(metric: DistanceMetric, config: HnswConfig) -> Self {
        Self {
            metric,
            dimension: None,
            config,
            nodes: Vec::new(),
            entry_point: None,
            max_layer: 0,
        }
    }

    /// Creates a new HNSW index with default configuration
    pub fn with_defaults(metric: DistanceMetric) -> Self {
        Self::new(metric, HnswConfig::default())
    }

    fn validate_dimension(&self, vector: &[f32]) -> Result<()> {
        if let Some(expected) = self.dimension {
            ensure!(
                vector.len() == expected,
                HnswError::DimensionMismatch {
                    expected,
                    actual: vector.len()
                }
            );
        }
        Ok(())
    }

    /// Returns the dimensionality tracked by the index
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Selects the layer for a new node using exponential distribution
    fn select_layer(&self) -> usize {
        let mut rng = rand::thread_rng();
        let uniform = rand::distributions::Uniform::new(0.0f64, 1.0);
        let r: f64 = uniform.sample(&mut rng);
        (-r.ln() * self.config.ml) as usize
    }

    /// Searches for the nearest neighbors in a specific layer
    fn search_layer(
        &self,
        query: &Vector,
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Result<Vec<HeapEntry>> {
        let mut visited = HashSet::new();
        // Use a vector to maintain candidates sorted by distance (closest first)
        // This makes it easier to check the worst candidate
        let mut candidates: Vec<HeapEntry> = Vec::new();
        let mut dynamic_candidates = BinaryHeap::new(); // min-heap for exploration

        // Initialize with entry points
        for &entry_id in entry_points {
            if entry_id < self.nodes.len() {
                let node = &self.nodes[entry_id];
                if layer < node.layer_count() {
                    let dist = distance(self.metric, query, &node.vector)?;
                    let entry = HeapEntry {
                        id: entry_id,
                        distance: dist,
                    };
                    candidates.push(entry);
                    dynamic_candidates.push(entry);
                    visited.insert(entry_id);
                }
            }
        }

        // Sort candidates by distance (closest first)
        candidates.sort();

        // Greedy search
        while let Some(current) = dynamic_candidates.pop() {
            let current_node = &self.nodes[current.id];
            if layer >= current_node.layer_count() {
                continue;
            }

            // Check all neighbors at this layer
            for &neighbor_id in current_node.neighbors_at_layer(layer) {
                if visited.contains(&neighbor_id) || neighbor_id >= self.nodes.len() {
                    continue;
                }

                visited.insert(neighbor_id);
                let neighbor_node = &self.nodes[neighbor_id];
                let dist = distance(self.metric, query, &neighbor_node.vector)?;

                // Check if we should add this candidate
                // Add if we have fewer than ef candidates, or if it's better than the worst
                let should_add = if candidates.len() < ef {
                    true
                } else {
                    // Compare with worst candidate (last in sorted vector)
                    dist < candidates.last().unwrap().distance
                };

                if should_add {
                    let entry = HeapEntry {
                        id: neighbor_id,
                        distance: dist,
                    };

                    // Insert in sorted order
                    match candidates.binary_search_by(|e| e.distance.partial_cmp(&dist).unwrap()) {
                        Ok(pos) | Err(pos) => {
                            candidates.insert(pos, entry);
                        }
                    }

                    // Keep only the best ef candidates
                    if candidates.len() > ef {
                        candidates.pop();
                    }

                    // Add to dynamic candidates for further exploration
                    dynamic_candidates.push(HeapEntry {
                        id: neighbor_id,
                        distance: dist,
                    });
                }
            }
        }

        Ok(candidates)
    }

    /// Selects neighbors using the HNSW heuristic (keep closest m nodes)
    fn select_neighbors_heuristic(
        &self,
        candidates: &[HeapEntry],
        m: usize,
        _layer: usize,
    ) -> Vec<usize> {
        // Sort candidates by distance (closest first)
        let mut sorted = candidates.to_vec();
        sorted.sort();

        // Simply take the closest m candidates
        // In a full implementation, you'd use a more sophisticated heuristic
        // that prunes redundant connections
        sorted
            .into_iter()
            .take(m)
            .map(|entry| entry.id)
            .collect()
    }

    /// Prunes connections to maintain m_max limit
    fn prune_connections(&mut self, node_id: usize, layer: usize, m: usize) {
        if node_id >= self.nodes.len() {
            return;
        }

        let node = &self.nodes[node_id];
        if layer >= node.layer_count() {
            return;
        }

        let neighbors = &node.layers[layer];
        if neighbors.len() <= m {
            return;
        }

        // Simple pruning: keep the first m connections
        // In a full implementation, you'd use a more sophisticated heuristic
        let node = &mut self.nodes[node_id];
        node.layers[layer].truncate(m);
    }

    /// Saves the index to a JSON file
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path.as_ref())
            .with_context(|| format!("failed to create index file at {}", path.as_ref().display()))?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, self)
            .context("failed to serialize index to JSON")?;
        Ok(())
    }

    /// Loads an index from a JSON file
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("failed to open index file at {}", path.as_ref().display()))?;
        let reader = BufReader::new(file);
        let index = serde_json::from_reader(reader)
            .with_context(|| format!("failed to deserialize index from {}", path.as_ref().display()))?;
        Ok(index)
    }
}

impl VectorIndex for HnswIndex {
    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn insert(&mut self, id: usize, vector: Vector) -> Result<()> {
        self.validate_dimension(&vector)?;
        if self.dimension.is_none() {
            self.dimension = Some(vector.len());
        }

        let layer = self.select_layer();
        let node_id = self.nodes.len();

        // Ensure we have enough layers for the new node
        let old_max_layer = self.max_layer;
        if layer > self.max_layer {
            self.max_layer = layer;
            // Expand all existing nodes to have this many layers
            for node in &mut self.nodes {
                while node.layers.len() <= layer {
                    node.layers.push(Vec::new());
                }
            }
        }

        // Create new node with appropriate number of layers (from 0 to layer, inclusive)
        let mut new_node = Node::new(id, vector.clone(), self.max_layer);

        if self.nodes.is_empty() {
            // First node becomes the entry point
            self.entry_point = Some(node_id);
            self.nodes.push(new_node);
            return Ok(());
        }

        // Determine starting layer for search
        // Start from min(max_layer, layer) - we search from the top layer that exists
        let search_start_layer = self.max_layer.min(layer);
        let mut entry_points = vec![self.entry_point.unwrap()];
        let mut current_layer = search_start_layer;

        // Search from top layer down to the insertion layer
        while current_layer > layer {
            let candidates = self.search_layer(&vector, &entry_points, 1, current_layer)?;
            if let Some(best) = candidates.first() {
                entry_points = vec![best.id];
            }
            current_layer -= 1;
        }

        // At the insertion layer and below, do full search and connect
        loop {
            let candidates = self.search_layer(
                &vector,
                &entry_points,
                self.config.ef_construction,
                current_layer,
            )?;

            // Select neighbors using heuristic
            let m = if current_layer == 0 {
                self.config.m_max * 2 // More connections at layer 0
            } else {
                self.config.m_max
            };

            let neighbors = self.select_neighbors_heuristic(&candidates, m, current_layer);

            // Add connections bidirectionally
            for &neighbor_id in &neighbors {
                new_node.add_neighbor(current_layer, neighbor_id);
                self.nodes[neighbor_id].add_neighbor(current_layer, node_id);

                // Prune connections if needed
                self.prune_connections(neighbor_id, current_layer, m);
            }

            if current_layer == 0 {
                break;
            }
            current_layer -= 1;

            // Update entry points for next layer
            if let Some(best) = candidates.first() {
                entry_points = vec![best.id];
            }
        }

        // Add the node to the graph
        self.nodes.push(new_node);

        // Update entry point if this node is at the highest layer
        // If it's at a higher layer than the old max, it's definitely the new entry point
        if layer > old_max_layer {
            self.entry_point = Some(node_id);
        } else if layer == self.max_layer {
            // If it's at the current max layer, we could use it as entry point
            // For simplicity, we keep the existing entry point unless this is higher
            // In a full implementation, you might want to balance entry points
        }

        Ok(())
    }

    fn search(&self, query: &Vector, limit: usize) -> Result<Vec<ScoredPoint>> {
        ensure!(limit > 0, "limit must be greater than zero to execute a search");
        ensure!(!self.nodes.is_empty(), HnswError::EmptyIndex);
        self.validate_dimension(query)?;

        let entry_point = self.entry_point.unwrap();
        let mut current_layer = self.max_layer;
        let mut current_candidates = vec![entry_point];

        // Search from top layer down to layer 0
        while current_layer > 0 {
            let candidates = self.search_layer(query, &current_candidates, 1, current_layer)?;
            if let Some(best) = candidates.first() {
                current_candidates = vec![best.id];
            }
            current_layer -= 1;
        }

        // Full search at layer 0
        let candidates = self.search_layer(
            query,
            &current_candidates,
            self.config.ef_search.max(limit),
            0,
        )?;

        // Convert to ScoredPoint and take top limit
        let mut results: Vec<ScoredPoint> = candidates
            .into_iter()
            .take(limit)
            .map(|entry| ScoredPoint::new(self.nodes[entry.id].id, entry.distance))
            .collect();

        // Sort by distance (closest first)
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_search_returns_expected_ids() {
        let mut index = HnswIndex::with_defaults(DistanceMetric::Euclidean);
        // Insert more points to ensure the graph is well-connected
        index.insert(0, vec![0.0, 0.0]).unwrap();
        index.insert(1, vec![1.0, 0.0]).unwrap();
        index.insert(2, vec![0.0, 1.0]).unwrap();
        index.insert(3, vec![1.0, 1.0]).unwrap();
        index.insert(4, vec![0.5, 0.5]).unwrap();

        let result = index.search(&vec![0.0, 0.0], 2).unwrap();
        // The closest should be node 0 (distance 0) and then node 1 or 2 (distance 1)
        assert_eq!(result[0].id, 0); // Closest point should be at origin
        assert!(result[0].distance < 0.1); // Should be very close to 0
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn empty_index_search_fails() {
        let index = HnswIndex::with_defaults(DistanceMetric::Euclidean);
        assert!(index.search(&vec![0.0, 0.0], 1).is_err());
    }

    #[test]
    fn save_and_load_preserves_index() {
        use std::env::temp_dir;

        let mut original_index = HnswIndex::with_defaults(DistanceMetric::Cosine);
        original_index.insert(0, vec![1.0, 0.0]).unwrap();
        original_index.insert(1, vec![0.0, 1.0]).unwrap();
        original_index.insert(2, vec![1.0, 1.0]).unwrap();

        let mut temp_path = temp_dir();
        temp_path.push(format!("test_hnsw_index_{}.json", std::process::id()));

        // Save the index
        original_index.save(&temp_path).unwrap();

        // Load the index
        let loaded_index = HnswIndex::load(&temp_path).unwrap();

        // Verify the loaded index has the same properties
        assert_eq!(loaded_index.metric(), original_index.metric());
        assert_eq!(loaded_index.len(), original_index.len());
        assert_eq!(loaded_index.dimension(), original_index.dimension());

        // Verify search results are identical
        let query = vec![1.0, 0.0];
        let original_results = original_index.search(&query, 2).unwrap();
        let loaded_results = loaded_index.search(&query, 2).unwrap();
        assert_eq!(original_results.len(), loaded_results.len());
        assert_eq!(original_results[0].id, loaded_results[0].id);

        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }
}

