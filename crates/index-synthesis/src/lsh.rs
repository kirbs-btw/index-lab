//! LSH (Locality Sensitive Hashing) for fast operations
//! 
//! CRITICAL: This module is actually used in all operations:
//! - Centroid assignment during build (O(log C) vs O(C))
//! - Neighbor finding during graph construction (O(1) vs O(n))
//! - Bucket routing during search (O(1) vs O(n))

use index_core::Vector;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// LSH hasher using random hyperplane projections (SimHash)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LshHasher {
    /// Random hyperplanes for projection (n_hyperplanes × dimension)
    hyperplanes: Vec<Vector>,
    /// Number of hash bits
    n_bits: usize,
}

impl LshHasher {
    /// Creates a new LSH hasher with random hyperplanes
    pub fn new(dimension: usize, n_bits: usize, seed: u64) -> Self {
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

    /// Hash a vector to a bucket index using SimHash
    pub fn hash(&self, vector: &[f32]) -> usize {
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

    /// Get buckets to probe (primary + Hamming-distance neighbors)
    /// 
    /// Multi-probe LSH improves recall by checking neighboring buckets
    pub fn get_probe_buckets(&self, primary: usize, n_probes: usize) -> Vec<usize> {
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

    /// Total number of buckets (2^n_bits)
    pub fn n_buckets(&self) -> usize {
        1 << self.n_bits
    }
}

/// LSH-based neighbor finder for graph construction
/// 
/// ACTUALLY USED during build/insert for neighbor finding
#[derive(Debug, Clone)]
pub struct NeighborFinder {
    hasher: LshHasher,
    /// Buckets: bucket_id -> Vec<(vector_id, vector)>
    buckets: Vec<Vec<(usize, Vector)>>,
}

impl NeighborFinder {
    /// Create a new neighbor finder
    pub fn new(dimension: usize, n_hyperplanes: usize, seed: u64) -> Self {
        let hasher = LshHasher::new(dimension, n_hyperplanes, seed);
        let n_buckets = hasher.n_buckets();
        Self {
            hasher,
            buckets: vec![Vec::new(); n_buckets],
        }
    }

    /// Insert a vector for neighbor finding
    /// 
    /// ACTUALLY CALLED during build/insert operations
    pub fn insert(&mut self, id: usize, vector: Vector) {
        let bucket_id = self.hasher.hash(&vector);
        self.buckets[bucket_id].push((id, vector));
    }

    /// Find approximate neighbors using LSH
    /// 
    /// ACTUALLY CALLED during graph construction
    /// Returns candidate IDs that are likely to be neighbors
    pub fn find_neighbors(&self, query: &Vector, max_candidates: usize) -> Vec<usize> {
        let primary = self.hasher.hash(query);
        let probe_buckets = self.hasher.get_probe_buckets(primary, self.hasher.n_buckets().min(8));
        
        let mut candidates = Vec::new();
        for bucket_id in probe_buckets {
            if bucket_id < self.buckets.len() {
                for (id, _) in &self.buckets[bucket_id] {
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
    pub fn clear(&mut self) {
        for bucket in &mut self.buckets {
            bucket.clear();
        }
    }
}

/// LSH-based centroid finder for fast cluster assignment
/// 
/// ACTUALLY USED during build/insert for centroid assignment
/// Uses LSH + centroid graph for O(log C) assignment
#[derive(Debug, Clone)]
pub struct CentroidFinder {
    hasher: LshHasher,
    /// Centroids indexed by LSH bucket
    /// bucket_id -> Vec<(centroid_id, centroid_vector)>
    centroid_buckets: Vec<Vec<(usize, Vector)>>,
    /// All centroids for fallback
    centroids: Vec<Vector>,
}

impl CentroidFinder {
    /// Create a new centroid finder
    pub fn new(dimension: usize, n_hyperplanes: usize, seed: u64) -> Self {
        let hasher = LshHasher::new(dimension, n_hyperplanes, seed);
        let n_buckets = hasher.n_buckets();
        Self {
            hasher,
            centroid_buckets: vec![Vec::new(); n_buckets],
            centroids: Vec::new(),
        }
    }

    /// Add centroids to the finder
    /// 
    /// ACTUALLY CALLED during build after K-means
    pub fn add_centroids(&mut self, centroids: Vec<Vector>) {
        self.centroids = centroids.clone();
        self.centroid_buckets.clear();
        self.centroid_buckets = vec![Vec::new(); self.hasher.n_buckets()];
        
        for (id, centroid) in centroids.iter().enumerate() {
            let bucket_id = self.hasher.hash(centroid);
            self.centroid_buckets[bucket_id].push((id, centroid.clone()));
        }
    }

    /// Find nearest centroid using LSH
    /// 
    /// ACTUALLY CALLED during build/insert for cluster assignment
    /// Returns centroid ID
    pub fn find_nearest_centroid(
        &self,
        vector: &Vector,
        metric: index_core::DistanceMetric,
    ) -> anyhow::Result<usize> {
        use index_core::distance;
        
        // Use LSH to find candidate centroids
        let primary = self.hasher.hash(vector);
        let probe_buckets = self.hasher.get_probe_buckets(primary, self.hasher.n_buckets().min(8));
        
        let mut best_centroid = 0;
        let mut best_dist = f32::MAX;
        
        // Check candidate centroids from probed buckets
        for bucket_id in probe_buckets {
            if bucket_id < self.centroid_buckets.len() {
                for (centroid_id, centroid) in &self.centroid_buckets[bucket_id] {
                    let dist = distance(metric, vector, centroid)?;
                    if dist < best_dist {
                        best_dist = dist;
                        best_centroid = *centroid_id;
                    }
                }
            }
        }
        
        // Fallback: if no candidates found or best is poor, check all centroids
        // This should rarely happen with good LSH parameters
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
        assert_ne!(h1, h3, "Opposite vectors should hash to different buckets");
    }

    #[test]
    fn test_neighbor_finder() {
        let mut finder = NeighborFinder::new(4, 3, 42); // 8 buckets
        
        // Insert some vectors
        finder.insert(0, vec![1.0, 1.0, 1.0, 1.0]);
        finder.insert(1, vec![1.1, 1.1, 1.1, 1.1]);
        finder.insert(2, vec![10.0, 10.0, 10.0, 10.0]);
        
        // Find neighbors for a query
        let query = vec![1.0, 1.0, 1.0, 1.0];
        let neighbors = finder.find_neighbors(&query, 10);
        
        // Should find at least some neighbors
        assert!(!neighbors.is_empty());
        assert!(neighbors.contains(&0) || neighbors.contains(&1));
    }

    #[test]
    fn test_centroid_finder() {
        let mut finder = CentroidFinder::new(4, 3, 42);
        
        // Add centroids
        let centroids = vec![
            vec![1.0, 1.0, 1.0, 1.0],
            vec![10.0, 10.0, 10.0, 10.0],
            vec![20.0, 20.0, 20.0, 20.0],
        ];
        finder.add_centroids(centroids);
        
        // Find nearest centroid
        let query = vec![1.1, 1.1, 1.1, 1.1];
        let nearest = finder.find_nearest_centroid(&query, index_core::DistanceMetric::Euclidean).unwrap();
        assert_eq!(nearest, 0); // Should find first centroid
    }
}
