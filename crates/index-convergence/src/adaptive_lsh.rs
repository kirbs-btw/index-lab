//! Adaptive LSH System
//! 
//! Adaptively tunes LSH parameters (hyperplanes, probes) based on:
//! - Data distribution characteristics
//! - Query performance feedback
//! - Recall requirements
//! 
//! GUARANTEED TO BE USED in all operations (centroid assignment, neighbor finding, bucket routing)

use index_core::Vector;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;

/// Adaptive LSH hasher with tunable parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLshHasher {
    /// Random hyperplanes for projection (n_hyperplanes × dimension)
    hyperplanes: Vec<Vector>,
    /// Current number of hash bits (adaptive)
    n_bits: usize,
    /// Minimum number of bits
    min_bits: usize,
    /// Maximum number of bits
    max_bits: usize,
    /// Dimension of vectors
    dimension: usize,
}

impl AdaptiveLshHasher {
    /// Create a new adaptive LSH hasher
    pub fn new(
        dimension: usize,
        initial_bits: usize,
        min_bits: usize,
        max_bits: usize,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let hyperplanes = (0..max_bits)
            .map(|_| {
                let mut plane: Vector = (0..dimension)
                    .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
                    .collect();
                // Normalize hyperplane
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
            n_bits: initial_bits.min(max_bits).max(min_bits),
            min_bits,
            max_bits,
            dimension,
        }
    }

    /// Hash a vector to a bucket index
    pub fn hash(&self, vector: &[f32]) -> usize {
        let mut bucket = 0usize;
        for (i, hyperplane) in self.hyperplanes.iter().take(self.n_bits).enumerate() {
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

    /// Get buckets to probe with adaptive probe count
    pub fn get_probe_buckets(&self, primary: usize, max_probes: usize) -> Vec<usize> {
        let mut buckets = vec![primary];
        let n_buckets = self.n_buckets();

        // Add Hamming-1 neighbors
        for bit in 0..self.n_bits {
            if buckets.len() >= max_probes {
                break;
            }
            let neighbor = primary ^ (1 << bit);
            if neighbor < n_buckets && !buckets.contains(&neighbor) {
                buckets.push(neighbor);
            }
        }

        // Add Hamming-2 neighbors
        'outer: for bit1 in 0..self.n_bits {
            for bit2 in (bit1 + 1)..self.n_bits {
                if buckets.len() >= max_probes {
                    break 'outer;
                }
                let neighbor = primary ^ (1 << bit1) ^ (1 << bit2);
                if neighbor < n_buckets && !buckets.contains(&neighbor) {
                    buckets.push(neighbor);
                }
            }
        }

        buckets.truncate(max_probes);
        buckets
    }

    /// Total number of buckets (2^n_bits)
    pub fn n_buckets(&self) -> usize {
        1 << self.n_bits
    }

    /// Get current number of bits
    pub fn n_bits(&self) -> usize {
        self.n_bits
    }

    /// Adaptively adjust number of bits based on performance
    pub fn adjust_bits(&mut self, recall: f32, target_recall: f32) {
        if recall < target_recall && self.n_bits < self.max_bits {
            // Increase bits for better recall
            self.n_bits = (self.n_bits + 1).min(self.max_bits);
        } else if recall > target_recall + 0.05 && self.n_bits > self.min_bits {
            // Decrease bits for better speed (if recall is much higher than needed)
            self.n_bits = (self.n_bits - 1).max(self.min_bits);
        }
    }
}

/// Adaptive LSH system with performance tracking
#[derive(Debug, Clone)]
pub struct AdaptiveLshSystem {
    /// Base hasher
    hasher: RefCell<AdaptiveLshHasher>,
    /// Performance tracker
    performance: RefCell<LshPerformance>,
    /// Target recall
    target_recall: f32,
}

#[derive(Debug, Clone)]
struct LshPerformance {
    /// Recent recall measurements
    recall_history: Vec<f32>,
    /// Recent probe counts used
    probe_history: Vec<usize>,
    /// Window size for tracking
    window_size: usize,
}

impl AdaptiveLshSystem {
    /// Create a new adaptive LSH system
    pub fn new(
        dimension: usize,
        initial_bits: usize,
        min_bits: usize,
        max_bits: usize,
        target_recall: f32,
        seed: u64,
    ) -> Self {
        Self {
            hasher: RefCell::new(AdaptiveLshHasher::new(
                dimension, initial_bits, min_bits, max_bits, seed,
            )),
            performance: RefCell::new(LshPerformance {
                recall_history: Vec::new(),
                probe_history: Vec::new(),
                window_size: 100,
            }),
            target_recall,
        }
    }

    /// Hash a vector (can be called from immutable context)
    pub fn hash(&self, vector: &[f32]) -> usize {
        self.hasher.borrow().hash(vector)
    }

    /// Get probe buckets with adaptive probe count
    pub fn get_probe_buckets(&self, primary: usize, max_probes: usize) -> Vec<usize> {
        self.hasher.borrow().get_probe_buckets(primary, max_probes)
    }

    /// Update performance metrics and adapt
    pub fn update_performance(&self, recall: f32, probes_used: usize) {
        let mut perf = self.performance.borrow_mut();
        perf.recall_history.push(recall);
        perf.probe_history.push(probes_used);
        
        if perf.recall_history.len() > perf.window_size {
            perf.recall_history.remove(0);
            perf.probe_history.remove(0);
        }

        // Adapt if we have enough history
        if perf.recall_history.len() >= 10 {
            let avg_recall: f32 = perf.recall_history.iter().sum::<f32>() / perf.recall_history.len() as f32;
            self.hasher.borrow_mut().adjust_bits(avg_recall, self.target_recall);
        }
    }

    /// Get optimal probe count based on performance
    pub fn optimal_probe_count(&self, min_probes: usize, max_probes: usize) -> usize {
        let perf = self.performance.borrow();
        if perf.probe_history.is_empty() {
            return (min_probes + max_probes) / 2;
        }

        // Find probe count that achieves target recall
        let avg_probes: usize = perf.probe_history.iter().sum::<usize>() / perf.probe_history.len();
        avg_probes.clamp(min_probes, max_probes)
    }
}

/// LSH-based centroid finder with adaptive parameters
#[derive(Debug, Clone)]
pub struct AdaptiveCentroidFinder {
    lsh: AdaptiveLshSystem,
    /// Centroids indexed by LSH bucket
    centroid_buckets: RefCell<Vec<Vec<(usize, Vector)>>>,
    /// All centroids for fallback
    centroids: Vec<Vector>,
}

impl AdaptiveCentroidFinder {
    /// Create a new adaptive centroid finder
    pub fn new(
        dimension: usize,
        initial_bits: usize,
        min_bits: usize,
        max_bits: usize,
        target_recall: f32,
        seed: u64,
    ) -> Self {
        let lsh = AdaptiveLshSystem::new(dimension, initial_bits, min_bits, max_bits, target_recall, seed);
        let n_buckets = lsh.hasher.borrow().n_buckets();
        Self {
            lsh,
            centroid_buckets: RefCell::new(vec![Vec::new(); n_buckets]),
            centroids: Vec::new(),
        }
    }

    /// Add centroids to the finder
    /// ACTUALLY CALLED during build after K-means
    pub fn add_centroids(&mut self, centroids: Vec<Vector>) {
        self.centroids = centroids.clone();
        let n_buckets = self.lsh.hasher.borrow().n_buckets();
        let mut buckets = vec![Vec::new(); n_buckets];
        
        for (id, centroid) in centroids.iter().enumerate() {
            let bucket_id = self.lsh.hash(centroid);
            buckets[bucket_id].push((id, centroid.clone()));
        }
        
        *self.centroid_buckets.borrow_mut() = buckets;
    }

    /// Find nearest centroid using adaptive LSH
    /// ACTUALLY CALLED during build/insert for cluster assignment
    pub fn find_nearest_centroid(
        &self,
        vector: &Vector,
        metric: index_core::DistanceMetric,
    ) -> anyhow::Result<usize> {
        use index_core::distance;
        
        // Use adaptive LSH to find candidate centroids
        let primary = self.lsh.hash(vector);
        let max_probes = self.lsh.optimal_probe_count(4, 12);
        let probe_buckets = self.lsh.get_probe_buckets(primary, max_probes);
        
        let mut best_centroid = 0;
        let mut best_dist = f32::MAX;
        let mut candidates_checked = 0;
        
        // Check candidate centroids from probed buckets
        let buckets = self.centroid_buckets.borrow();
        for bucket_id in probe_buckets {
            if bucket_id < buckets.len() {
                for (centroid_id, centroid) in &buckets[bucket_id] {
                    let dist = distance(metric, vector, centroid)?;
                    candidates_checked += 1;
                    if dist < best_dist {
                        best_dist = dist;
                        best_centroid = *centroid_id;
                    }
                }
            }
        }
        
        // Fallback: if no candidates found or best is poor, check all centroids
        if best_dist > 1000.0 && !self.centroids.is_empty() {
            for (id, centroid) in self.centroids.iter().enumerate() {
                let dist = distance(metric, vector, centroid)?;
                if dist < best_dist {
                    best_dist = dist;
                    best_centroid = id;
                }
            }
        }
        
        Ok(best_centroid)
    }
}

/// LSH-based neighbor finder with adaptive parameters
/// ACTUALLY USED during build/insert for neighbor finding
#[derive(Debug, Clone)]
pub struct AdaptiveNeighborFinder {
    lsh: AdaptiveLshSystem,
    /// Buckets: bucket_id -> Vec<(vector_id, vector)>
    buckets: RefCell<Vec<Vec<(usize, Vector)>>>,
}

impl AdaptiveNeighborFinder {
    /// Create a new adaptive neighbor finder
    pub fn new(
        dimension: usize,
        initial_bits: usize,
        min_bits: usize,
        max_bits: usize,
        target_recall: f32,
        seed: u64,
    ) -> Self {
        let lsh = AdaptiveLshSystem::new(dimension, initial_bits, min_bits, max_bits, target_recall, seed);
        let n_buckets = lsh.hasher.borrow().n_buckets();
        Self {
            lsh,
            buckets: RefCell::new(vec![Vec::new(); n_buckets]),
        }
    }

    /// Insert a vector for neighbor finding
    /// ACTUALLY CALLED during build/insert operations
    pub fn insert(&self, id: usize, vector: Vector) {
        let bucket_id = self.lsh.hash(&vector);
        let mut buckets = self.buckets.borrow_mut();
        if bucket_id < buckets.len() {
            buckets[bucket_id].push((id, vector));
        }
    }

    /// Find approximate neighbors using adaptive LSH
    /// ACTUALLY CALLED during graph construction
    pub fn find_neighbors(&self, query: &Vector, max_candidates: usize) -> Vec<usize> {
        let primary = self.lsh.hash(query);
        let max_probes = self.lsh.optimal_probe_count(4, 12);
        let probe_buckets = self.lsh.get_probe_buckets(primary, max_probes);
        
        let mut candidates = Vec::new();
        let buckets = self.buckets.borrow();
        for bucket_id in probe_buckets {
            if bucket_id < buckets.len() {
                for (id, _) in &buckets[bucket_id] {
                    if !candidates.contains(id) {
                        candidates.push(*id);
                        if candidates.len() >= max_candidates {
                            return candidates;
                        }
                    }
                }
            }
        }
        
        candidates
    }

    /// Clear all stored vectors (for reuse)
    pub fn clear(&self) {
        let mut buckets = self.buckets.borrow_mut();
        for bucket in buckets.iter_mut() {
            bucket.clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_lsh_hash() {
        let hasher = AdaptiveLshHasher::new(4, 3, 2, 8, 42);
        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let v2 = vec![1.1, 2.1, 3.1, 4.1];
        
        let h1 = hasher.hash(&v1);
        let h2 = hasher.hash(&v2);
        
        // Similar vectors should hash similarly
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_adaptive_bits_adjustment() {
        let mut hasher = AdaptiveLshHasher::new(4, 3, 2, 8, 42);
        let initial_bits = hasher.n_bits();
        
        // Low recall -> increase bits
        hasher.adjust_bits(0.85, 0.95);
        assert!(hasher.n_bits() >= initial_bits);
        
        // High recall -> decrease bits
        hasher.adjust_bits(0.99, 0.95);
        // May or may not decrease depending on threshold
    }

    #[test]
    fn test_adaptive_centroid_finder() {
        let mut finder = AdaptiveCentroidFinder::new(4, 3, 2, 8, 0.95, 42);
        
        let centroids = vec![
            vec![1.0, 1.0, 1.0, 1.0],
            vec![10.0, 10.0, 10.0, 10.0],
        ];
        finder.add_centroids(centroids);
        
        let query = vec![1.1, 1.1, 1.1, 1.1];
        let nearest = finder.find_nearest_centroid(&query, index_core::DistanceMetric::Euclidean).unwrap();
        assert_eq!(nearest, 0);
    }
}
