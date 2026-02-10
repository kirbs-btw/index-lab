//! ZENITH: Zero-configuration Enhanced Navigable Index with Tuned Heuristics
//!
//! A refined HNSW that auto-tunes all parameters, uses diversity-aware neighbor
//! selection (RNG heuristic), and employs multiple entry points for robust search.
//!
//! # Design Principles
//!
//! - **One file, one structure**: Everything lives here. No separate modules.
//! - **HNSW is the answer**: The multi-layer graph is proven. Fix it, don't replace it.
//! - **Auto-tune, don't configure**: M, ef_construction, ef_search derived from data.
//! - **Serialize everything**: Complete state persistence from day 1.
//!
//! # Architecture
//!
//! ```text
//! Layer L (top):  few nodes, long-range connections
//! Layer L-1:      more nodes, medium-range connections
//! ...
//! Layer 0 (base): all nodes, short-range connections
//! ```
//!
//! Key innovations over standard HNSW:
//! 1. Zero-config auto-tuning (M, ef_build, ef_search from dataset properties)
//! 2. Diversity-aware neighbor selection (RNG/Vamana heuristic)
//! 3. Multiple entry points for robust search
//! 4. Adaptive search effort (ef scales with requested k)

use anyhow::Result;
use index_core::{distance, DistanceMetric, ScoredPoint, Vector, VectorIndex};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum ZenithError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("index is empty, cannot search")]
    EmptyIndex,
    #[error("vector dimension must be at least 1")]
    ZeroDimension,
}

// ---------------------------------------------------------------------------
// Auto-tuned parameters
// ---------------------------------------------------------------------------

/// Parameters automatically derived from dataset characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TunedParams {
    /// Max edges per node per layer (higher layers use m, layer 0 uses 2*m).
    m: usize,
    /// Candidate list size during construction.
    ef_construction: usize,
    /// Layer normalization factor: 1 / ln(m).
    ml: f64,
}

impl TunedParams {
    /// Derive optimal parameters from dataset dimension.
    fn from_dimension(dim: usize) -> Self {
        // M scales with sqrt(dim) * 1.5, clamped to sensible range
        // Higher M = more connections = better recall at cost of memory & build time
        let m = ((dim as f64).sqrt() * 1.5).round().clamp(12.0, 48.0) as usize;
        // Very generous build quality — this is the key to high recall
        let ef_construction = m * 16;
        // Standard HNSW layer normalization
        let ml = 1.0 / (m as f64).ln();
        Self {
            m,
            ef_construction,
            ml,
        }
    }

    /// Compute adaptive ef_search from the requested k.
    fn adaptive_ef(&self, k: usize) -> usize {
        // Generous search effort — recall matters more than marginal QPS
        (k * 10).max(self.m * 6).min(800)
    }

    /// Number of entry points to maintain (spread across the space).
    fn num_entry_points(&self) -> usize {
        5
    }
}

// ---------------------------------------------------------------------------
// Graph node
// ---------------------------------------------------------------------------

/// A node in the HNSW graph. Stores per-layer neighbor lists.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Node {
    /// External user-facing ID.
    id: usize,
    /// The vector data.
    vector: Vector,
    /// Neighbor lists per layer. `neighbors[0]` = layer 0 (base), etc.
    neighbors: Vec<Vec<usize>>,
    /// The highest layer this node appears in.
    max_layer: usize,
}

// ---------------------------------------------------------------------------
// Priority-queue helper (min-heap entry)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Candidate {
    /// Internal node index.
    node_idx: usize,
    dist: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reversed for min-heap behaviour (BinaryHeap is max-heap by default)
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// ZENITH Index
// ---------------------------------------------------------------------------

/// ZENITH: Zero-configuration Enhanced Navigable Index with Tuned Heuristics.
///
/// An enhanced HNSW with auto-tuned parameters, diversity-aware neighbor
/// selection, and multiple entry points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZenithIndex {
    metric: DistanceMetric,
    /// All graph nodes (internal index → Node).
    nodes: Vec<Node>,
    /// Map from external ID → internal node index.
    id_to_idx: HashMap<usize, usize>,
    /// Multiple entry points for robust search (internal indices).
    entry_points: Vec<usize>,
    /// Current maximum layer in the graph.
    max_layer: usize,
    /// Auto-tuned parameters (None until first vector sets dimension).
    params: Option<TunedParams>,
    /// Tracked dimension.
    dimension: Option<usize>,
    /// RNG seed for reproducibility.
    seed: u64,
    /// Current RNG state counter (incremented on each insert for layer selection).
    rng_counter: u64,
}

impl ZenithIndex {
    /// Creates a new ZENITH index with the given distance metric.
    pub fn new(metric: DistanceMetric) -> Self {
        Self::with_seed(metric, 42)
    }

    /// Creates a new ZENITH index with a specific seed for reproducibility.
    pub fn with_seed(metric: DistanceMetric, seed: u64) -> Self {
        Self {
            metric,
            nodes: Vec::new(),
            id_to_idx: HashMap::new(),
            entry_points: Vec::new(),
            max_layer: 0,
            params: None,
            dimension: None,
            seed,
            rng_counter: 0,
        }
    }

    /// Returns the tracked vector dimension.
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Saves the index to a JSON file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, self)?;
        Ok(())
    }

    /// Loads an index from a JSON file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let index = serde_json::from_reader(reader)?;
        Ok(index)
    }

    // -- internal helpers ---------------------------------------------------

    fn validate_dim(&self, vector: &[f32]) -> Result<()> {
        if vector.is_empty() {
            return Err(ZenithError::ZeroDimension.into());
        }
        if let Some(expected) = self.dimension {
            if vector.len() != expected {
                return Err(ZenithError::DimensionMismatch {
                    expected,
                    actual: vector.len(),
                }
                .into());
            }
        }
        Ok(())
    }

    /// Initialize parameters on first vector.
    fn ensure_init(&mut self, dim: usize) {
        if self.params.is_none() {
            self.dimension = Some(dim);
            self.params = Some(TunedParams::from_dimension(dim));
        }
    }

    /// Pick a random layer for a new node using the HNSW exponential distribution.
    fn random_layer(&mut self) -> usize {
        let ml = self.params.as_ref().unwrap().ml;
        let mut rng = StdRng::seed_from_u64(self.seed.wrapping_add(self.rng_counter));
        self.rng_counter += 1;
        let r: f64 = rng.gen::<f64>();
        // Guard against log(0)
        if r <= 0.0 {
            return 0;
        }
        (-r.ln() * ml).floor() as usize
    }

    /// Compute distance between query vector and a stored node.
    #[inline]
    fn dist_to_node(&self, query: &[f32], node_idx: usize) -> f32 {
        distance(self.metric, query, &self.nodes[node_idx].vector).unwrap_or(f32::MAX)
    }

    /// Greedy search within a single layer.
    /// Returns up to `ef` nearest candidates sorted by distance (closest first).
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        layer: usize,
    ) -> Vec<Candidate> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); // min-heap via reversed Ord
        let mut results: Vec<Candidate> = Vec::new();

        // Seed with entry points
        for &ep in entry_points {
            if visited.insert(ep) {
                let d = self.dist_to_node(query, ep);
                candidates.push(Candidate {
                    node_idx: ep,
                    dist: d,
                });
                results.push(Candidate {
                    node_idx: ep,
                    dist: d,
                });
            }
        }

        while let Some(current) = candidates.pop() {
            // If current is farther than the worst in our result set and we have
            // enough results, stop exploring.
            let worst_dist = if results.len() >= ef {
                results.iter().map(|c| c.dist).fold(f32::NEG_INFINITY, f32::max)
            } else {
                f32::MAX
            };

            if current.dist > worst_dist {
                break;
            }

            // Explore neighbors in this layer
            if layer < self.nodes[current.node_idx].neighbors.len() {
                for &neighbor_idx in &self.nodes[current.node_idx].neighbors[layer] {
                    if visited.insert(neighbor_idx) {
                        let d = self.dist_to_node(query, neighbor_idx);

                        // Only consider if better than worst result or results not full
                        if results.len() < ef || d < worst_dist {
                            candidates.push(Candidate {
                                node_idx: neighbor_idx,
                                dist: d,
                            });
                            results.push(Candidate {
                                node_idx: neighbor_idx,
                                dist: d,
                            });

                            // Keep results bounded to ef
                            if results.len() > ef {
                                results.sort_by(|a, b| {
                                    a.dist
                                        .partial_cmp(&b.dist)
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                });
                                results.truncate(ef);
                            }
                        }
                    }
                }
            }
        }

        results.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(ef);
        results
    }

    /// Diversity-aware neighbor selection (RNG / Vamana heuristic).
    ///
    /// Uses a two-phase approach:
    /// Phase 1: Select diverse neighbors using RNG heuristic (up to ~60% of M)
    /// Phase 2: Fill remaining slots with closest candidates (for connectivity)
    ///
    /// This balances diversity (better long-range navigation) with proximity
    /// (better local accuracy), which is the key to high recall.
    fn select_neighbors_heuristic(
        &self,
        node_idx: usize,
        candidates: &[Candidate],
        m: usize,
    ) -> Vec<usize> {
        // Sort candidates by distance (closest first)
        let mut sorted: Vec<Candidate> = candidates.to_vec();
        sorted.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));

        let mut selected: Vec<usize> = Vec::with_capacity(m);
        let diversity_budget = (m * 3) / 5; // ~60% diverse, ~40% closest

        // Phase 1: Diversity-aware selection (RNG heuristic)
        for cand in &sorted {
            if cand.node_idx == node_idx {
                continue;
            }

            let occluded = selected.iter().any(|&sel_idx| {
                let d_sel_cand = self.dist_to_node(&self.nodes[sel_idx].vector, cand.node_idx);
                // Use a relaxation factor (0.95) to be slightly less aggressive
                d_sel_cand < cand.dist * 0.95
            });

            if !occluded {
                selected.push(cand.node_idx);
            }

            if selected.len() >= diversity_budget {
                break;
            }
        }

        // Phase 2: Fill remaining with closest candidates (proximity for recall)
        for cand in &sorted {
            if cand.node_idx == node_idx {
                continue;
            }
            if !selected.contains(&cand.node_idx) {
                selected.push(cand.node_idx);
            }
            if selected.len() >= m {
                break;
            }
        }

        selected
    }

    /// Update entry points to maintain diversity across the vector space.
    fn update_entry_points(&mut self, new_idx: usize) {
        let max_eps = self.params.as_ref().map(|p| p.num_entry_points()).unwrap_or(5);

        if self.entry_points.is_empty() {
            self.entry_points.push(new_idx);
            return;
        }

        // If the new node is on the highest layer, it becomes a primary entry point
        if self.nodes[new_idx].max_layer >= self.max_layer {
            // Keep it at front
            if !self.entry_points.contains(&new_idx) {
                self.entry_points.insert(0, new_idx);
            }
        }

        // If we have fewer than max entry points, add this node if it's diverse enough
        if self.entry_points.len() < max_eps && !self.entry_points.contains(&new_idx) {
            // Check minimum distance to existing entry points
            let min_dist = self
                .entry_points
                .iter()
                .map(|&ep| self.dist_to_node(&self.nodes[new_idx].vector, ep))
                .fold(f32::MAX, f32::min);

            // Add if reasonably far from existing entry points
            // Use a threshold that adapts to the dataset spread
            if min_dist > 0.0 {
                self.entry_points.push(new_idx);
            }
        }

        // Trim to max
        if self.entry_points.len() > max_eps {
            self.entry_points.truncate(max_eps);
        }
    }

    /// Insert a node into the HNSW graph.
    fn insert_node(&mut self, id: usize, vector: Vector) -> Result<()> {
        let params = self.params.as_ref().unwrap().clone();
        let node_layer = self.random_layer();

        // Create node with empty neighbor lists for each layer
        let node = Node {
            id,
            vector: vector.clone(),
            neighbors: vec![Vec::new(); node_layer + 1],
            max_layer: node_layer,
        };

        let new_idx = self.nodes.len();
        self.nodes.push(node);
        self.id_to_idx.insert(id, new_idx);

        // First node — just set as entry point
        if self.nodes.len() == 1 {
            self.entry_points.push(new_idx);
            self.max_layer = node_layer;
            return Ok(());
        }

        let m = params.m;
        let m_max_0 = m * 2; // Layer 0 gets more connections
        let ef_construction = params.ef_construction;

        // Phase 1: Navigate from top layer down to node_layer + 1 using greedy search
        let mut current_eps: Vec<usize> = self.entry_points.clone();
        let top = self.max_layer;

        for layer in (node_layer + 1..=top).rev() {
            let nearest = self.search_layer(&vector, &current_eps, 1, layer);
            if !nearest.is_empty() {
                current_eps = nearest.iter().map(|c| c.node_idx).collect();
            }
        }

        // Phase 2: From node_layer down to layer 0, search and connect
        for layer in (0..=node_layer.min(top)).rev() {
            let candidates = self.search_layer(&vector, &current_eps, ef_construction, layer);
            let m_layer = if layer == 0 { m_max_0 } else { m };

            // Select diverse neighbors
            let neighbors = self.select_neighbors_heuristic(new_idx, &candidates, m_layer);

            // Set forward edges
            self.nodes[new_idx].neighbors[layer] = neighbors.clone();

            // Set backward edges (bidirectional)
            for &neighbor_idx in &neighbors {
                // Add new_idx as a neighbor of neighbor_idx in this layer
                if layer < self.nodes[neighbor_idx].neighbors.len() {
                    self.nodes[neighbor_idx].neighbors[layer].push(new_idx);

                    // Prune if over capacity
                    let max_neighbors = if layer == 0 { m_max_0 } else { m };
                    if self.nodes[neighbor_idx].neighbors[layer].len() > max_neighbors {
                        // Re-select using diversity heuristic
                        let neighbor_cands: Vec<Candidate> = self.nodes[neighbor_idx].neighbors
                            [layer]
                            .iter()
                            .map(|&nidx| Candidate {
                                node_idx: nidx,
                                dist: self.dist_to_node(
                                    &self.nodes[neighbor_idx].vector,
                                    nidx,
                                ),
                            })
                            .collect();
                        let pruned =
                            self.select_neighbors_heuristic(neighbor_idx, &neighbor_cands, max_neighbors);
                        self.nodes[neighbor_idx].neighbors[layer] = pruned;
                    }
                }
            }

            // Use this layer's results as entry points for the next layer down
            current_eps = candidates.iter().map(|c| c.node_idx).collect();
        }

        // Update max layer and entry points
        if node_layer > self.max_layer {
            self.max_layer = node_layer;
        }
        self.update_entry_points(new_idx);

        Ok(())
    }

    /// Core search: navigate from entry points through layers to find nearest neighbors.
    fn search_internal(&self, query: &[f32], limit: usize) -> Result<Vec<ScoredPoint>> {
        if self.nodes.is_empty() {
            return Err(ZenithError::EmptyIndex.into());
        }

        let params = self.params.as_ref().unwrap();
        let ef = params.adaptive_ef(limit);

        // Start from the closest entry point(s)
        let mut current_eps = self.select_best_entry_points(query);

        // Navigate from top layer down to layer 1
        for layer in (1..=self.max_layer).rev() {
            let nearest = self.search_layer(query, &current_eps, 1, layer);
            if !nearest.is_empty() {
                current_eps = nearest.iter().map(|c| c.node_idx).collect();
            }
        }

        // Full search at layer 0 with adaptive ef
        let results = self.search_layer(query, &current_eps, ef, 0);

        // Convert to ScoredPoint with external IDs
        Ok(results
            .into_iter()
            .take(limit)
            .map(|c| ScoredPoint::new(self.nodes[c.node_idx].id, c.dist))
            .collect())
    }

    /// Select the best entry points for a query by picking the closest among
    /// our maintained entry points.
    fn select_best_entry_points(&self, query: &[f32]) -> Vec<usize> {
        if self.entry_points.len() <= 1 {
            return self.entry_points.clone();
        }

        // Score each entry point by distance to query
        let mut scored: Vec<(usize, f32)> = self
            .entry_points
            .iter()
            .map(|&ep| (ep, self.dist_to_node(query, ep)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Use the closest 1-3 entry points (more for layer 0 search)
        let n = scored.len().min(3);
        scored.into_iter().take(n).map(|(idx, _)| idx).collect()
    }
}

// ---------------------------------------------------------------------------
// VectorIndex trait implementation
// ---------------------------------------------------------------------------

impl VectorIndex for ZenithIndex {
    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn insert(&mut self, id: usize, vector: Vector) -> Result<()> {
        self.validate_dim(&vector)?;
        self.ensure_init(vector.len());
        self.insert_node(id, vector)
    }

    fn search(&self, query: &Vector, limit: usize) -> Result<Vec<ScoredPoint>> {
        self.validate_dim(query)?;
        self.search_internal(query, limit)
    }

    fn delete(&mut self, id: usize) -> Result<bool> {
        let idx = match self.id_to_idx.remove(&id) {
            Some(idx) => idx,
            None => return Ok(false),
        };

        // Remove all edges pointing to this node
        for node in &mut self.nodes {
            for layer_neighbors in &mut node.neighbors {
                layer_neighbors.retain(|&n| n != idx);
            }
        }

        // Mark the node as deleted by clearing its neighbors and setting a tombstone.
        // We don't actually remove from the Vec to avoid invalidating indices.
        // Instead, we clear the node's data.
        self.nodes[idx].neighbors.clear();
        self.nodes[idx].vector.clear();
        self.nodes[idx].id = usize::MAX; // tombstone

        // Remove from entry points
        self.entry_points.retain(|&ep| ep != idx);
        if self.entry_points.is_empty() && !self.nodes.is_empty() {
            // Find a valid node to be entry point
            for (i, node) in self.nodes.iter().enumerate() {
                if node.id != usize::MAX {
                    self.entry_points.push(i);
                    break;
                }
            }
        }

        Ok(true)
    }

    fn update(&mut self, id: usize, vector: Vector) -> Result<bool> {
        self.validate_dim(&vector)?;

        if !self.id_to_idx.contains_key(&id) {
            return Ok(false);
        }

        // Simple strategy: delete old, insert new
        self.delete(id)?;
        self.insert(id, vector)?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_tune_params() {
        let p = TunedParams::from_dimension(64);
        assert_eq!(p.m, 12); // sqrt(64)*1.5 = 12
        assert_eq!(p.ef_construction, 192); // 12 * 16

        let p = TunedParams::from_dimension(256);
        assert_eq!(p.m, 24); // sqrt(256)*1.5 = 24

        let p = TunedParams::from_dimension(4);
        assert_eq!(p.m, 12); // clamped to min 12

        let p = TunedParams::from_dimension(10000);
        assert_eq!(p.m, 48); // clamped to max 48
    }

    #[test]
    fn test_adaptive_ef() {
        let p = TunedParams::from_dimension(64);
        // k=10: max(10*10, 12*6) = max(100, 72) = 100
        assert_eq!(p.adaptive_ef(10), 100);
        // k=1: max(10, 72) = 72
        assert_eq!(p.adaptive_ef(1), 72);
        // k=100: min(max(1000, 72), 800) = 800
        assert_eq!(p.adaptive_ef(100), 800);
    }
}
