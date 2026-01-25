use index_core::{DistanceMetric, ScoredPoint, VectorIndex};
use index_hnsw::{HnswConfig, HnswIndex};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::error::{AtlasError, Result};
use crate::sparse::SparseVector;

/// Configuration for a hybrid bucket
#[derive(Debug, Clone)]
pub struct BucketConfig {
    pub hnsw_config: HnswConfig,
    pub dense_weight: f32,
    pub metric: DistanceMetric,
}

/// Hybrid bucket containing both dense (HNSW) and sparse (inverted index) components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridBucket {
    /// Bucket ID
    pub id: usize,

    /// Cluster centroid
    pub centroid: Vec<f32>,

    /// Dense vector index (mini-HNSW)
    dense_index: HnswIndex,

    /// Sparse inverted index: term_id → [vector_ids]
    inverted_index: HashMap<u32, Vec<usize>>,

    /// Sparse vectors storage: vector_id → sparse_vector
    sparse_vectors: HashMap<usize, SparseVector>,

    /// Number of vectors in this bucket
    size: usize,

    /// Fusion weight for dense component
    dense_weight: f32,
}

impl HybridBucket {
    /// Create a new hybrid bucket
    pub fn new(id: usize, centroid: Vec<f32>, config: &BucketConfig) -> Self {
        Self {
            id,
            centroid,
            dense_index: HnswIndex::new(config.metric, config.hnsw_config),
            inverted_index: HashMap::new(),
            sparse_vectors: HashMap::new(),
            size: 0,
            dense_weight: config.dense_weight,
        }
    }

    /// Insert a dense-only vector
    pub fn insert(&mut self, id: usize, vector: Vec<f32>) -> Result<()> {
        self.dense_index
            .insert(id, vector)
            .map_err(|e| AtlasError::HnswError(e.to_string()))?;
        self.size += 1;
        Ok(())
    }

    /// Insert a hybrid vector (dense + sparse)
    pub fn insert_hybrid(
        &mut self,
        id: usize,
        dense: Vec<f32>,
        sparse: SparseVector,
    ) -> Result<()> {
        // Insert dense component
        self.dense_index
            .insert(id, dense)
            .map_err(|e| AtlasError::HnswError(e.to_string()))?;

        // Update inverted index for each sparse term
        for &term_id in sparse.term_ids().iter() {
            self.inverted_index
                .entry(term_id)
                .or_insert_with(Vec::new)
                .push(id);
        }

        // Store sparse vector
        self.sparse_vectors.insert(id, sparse);
        self.size += 1;
        Ok(())
    }

    /// Pure dense search using HNSW
    pub fn search_dense(&self, query: &[f32], k: usize) -> Result<Vec<ScoredPoint>> {
        self.dense_index
            .search(&query.to_vec(), k)
            .map_err(|e| AtlasError::HnswError(e.to_string()))
    }

    /// Hybrid search: dense + sparse fusion
    pub fn search_hybrid(
        &self,
        dense_query: &[f32],
        sparse_query: &SparseVector,
        k: usize,
    ) -> Result<Vec<ScoredPoint>> {
        if sparse_query.is_empty() {
            // No sparse query, just do dense search
            return self.search_dense(dense_query, k);
        }

        // Strategy: Get dense candidates, filter/boost by sparse
        let dense_candidates = self
            .dense_index
            .search(&dense_query.to_vec(), k * 3)
            .map_err(|e| AtlasError::HnswError(e.to_string()))?;

        // Get sparse candidates from inverted index
        let sparse_candidates = self.get_sparse_candidates(sparse_query);

        // Combine candidates (union for now)
        let mut candidate_set: HashSet<usize> = dense_candidates.iter().map(|sp| sp.id).collect();
        candidate_set.extend(sparse_candidates);

        // Compute hybrid scores for all candidates
        let _metric = self.dense_index.metric();
        let mut scored_results = Vec::new();

        for candidate_id in candidate_set {
            // Get dense score (recompute or use cached)
            let dense_score = if let Some(sp) = dense_candidates.iter().find(|sp| sp.id == candidate_id) {
                sp.distance
            } else {
                // Not in dense top-k, need to compute distance
                // This requires access to the vector, which HNSW doesn't expose directly
                // For now, we'll skip candidates not in dense results
                continue;
            };

            // Get sparse score
            let sparse_score = if let Some(sv) = self.sparse_vectors.get(&candidate_id) {
                1.0 - sparse_query.cosine_similarity(sv) // Convert similarity to distance
            } else {
                1.0 // Worst case if no sparse vector
            };

            // Normalize scores (simple min-max for now)
            // In production, should track statistics
            let normalized_dense = dense_score / (dense_score + 1.0);
            let normalized_sparse = sparse_score;

            // Fuse scores
            let fused_score =
                self.dense_weight * normalized_dense + (1.0 - self.dense_weight) * normalized_sparse;

            scored_results.push(ScoredPoint::new(candidate_id, fused_score));
        }

        // Sort by fused score and return top-k
        scored_results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored_results.truncate(k);

        Ok(scored_results)
    }

    /// Get candidate IDs that contain at least one query term
    fn get_sparse_candidates(&self, sparse_query: &SparseVector) -> HashSet<usize> {
        let mut candidates = HashSet::new();

        for &term_id in sparse_query.term_ids().iter() {
            if let Some(doc_ids) = self.inverted_index.get(&term_id) {
                candidates.extend(doc_ids);
            }
        }

        candidates
    }

    /// Get the number of vectors in this bucket
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if bucket is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get the centroid
    pub fn centroid(&self) -> &[f32] {
        &self.centroid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> BucketConfig {
        let config = BucketConfig {
            hnsw_config: HnswConfig {
                m_max: 16,
                ef_construction: 50,
                ef_search: 30,
                ml: 1.0 / 2.0_f64.ln(),
            },
            dense_weight: 0.6,
            metric: DistanceMetric::Euclidean,
        };
        config
    }

    #[test]
    fn test_bucket_creation() {
        let config = create_test_config();
        let centroid = vec![0.0; 64];
        let bucket = HybridBucket::new(0, centroid.clone(), &config);

        assert_eq!(bucket.id, 0);
        assert_eq!(bucket.centroid(), &centroid[..]);
        assert_eq!(bucket.len(), 0);
        assert!(bucket.is_empty());
    }

    #[test]
    fn test_dense_only_insert_and_search() {
        let config = create_test_config();
        let centroid = vec![0.0; 64];
        let mut bucket = HybridBucket::new(0, centroid, &config);

        // Insert some vectors
        bucket.insert(0, vec![1.0; 64]).unwrap();
        bucket.insert(1, vec![2.0; 64]).unwrap();
        bucket.insert(2, vec![3.0; 64]).unwrap();

        assert_eq!(bucket.len(), 3);

        // Search for vector closest to [1.0; 64]
        let query_vec = vec![1.0; 64];
        let results = bucket.search_dense(&query_vec, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 0); // Should be closest
    }

    #[test]
    fn test_hybrid_insert() {
        let config = create_test_config();
        let centroid = vec![0.0; 64];
        let mut bucket = HybridBucket::new(0, centroid, &config);

        let dense = vec![1.0; 64];
        let sparse = SparseVector::new(vec![(1, 0.5), (2, 0.3)]);

        bucket.insert_hybrid(0, dense, sparse.clone()).unwrap();

        assert_eq!(bucket.len(), 1);
        assert!(bucket.sparse_vectors.contains_key(&0));
        assert!(bucket.inverted_index.contains_key(&1));
        assert!(bucket.inverted_index.contains_key(&2));
    }

    #[test]
    fn test_inverted_index_correctness() {
        let config = create_test_config();
        let centroid = vec![0.0; 64];
        let mut bucket = HybridBucket::new(0, centroid, &config);

        // Insert vectors with overlapping sparse terms
        bucket
            .insert_hybrid(0, vec![1.0; 64], SparseVector::new(vec![(1, 0.5), (2, 0.3)]))
            .unwrap();
        bucket
            .insert_hybrid(1, vec![2.0; 64], SparseVector::new(vec![(2, 0.4), (3, 0.6)]))
            .unwrap();

        // Check inverted index
        assert_eq!(bucket.inverted_index.get(&1), Some(&vec![0]));
        assert_eq!(bucket.inverted_index.get(&2).unwrap().len(), 2); // Both vectors
        assert_eq!(bucket.inverted_index.get(&3), Some(&vec![1]));
    }

    #[test]
    fn test_sparse_candidates() {
        let config = create_test_config();
        let centroid = vec![0.0; 64];
        let mut bucket = HybridBucket::new(0, centroid, &config);

        bucket
            .insert_hybrid(0, vec![1.0; 64], SparseVector::new(vec![(1, 0.5)]))
            .unwrap();
        bucket
            .insert_hybrid(1, vec![2.0; 64], SparseVector::new(vec![(2, 0.5)]))
            .unwrap();
        bucket
            .insert_hybrid(2, vec![3.0; 64], SparseVector::new(vec![(1, 0.5), (2, 0.5)]))
            .unwrap();

        // Query with term 1
        let query = SparseVector::new(vec![(1, 1.0)]);
        let candidates = bucket.get_sparse_candidates(&query);

        assert_eq!(candidates.len(), 2); // Vectors 0 and 2
        assert!(candidates.contains(&0));
        assert!(candidates.contains(&2));
    }

    #[test]
    fn test_hybrid_search_with_empty_sparse_query() {
        let config = create_test_config();
        let centroid = vec![0.0; 64];
        let mut bucket = HybridBucket::new(0, centroid, &config);

        bucket.insert(0, vec![1.0; 64]).unwrap();
        bucket.insert(1, vec![2.0; 64]).unwrap();

        // Empty sparse query should fall back to dense search
        let results = bucket
            .search_hybrid(&vec![1.0; 64], &SparseVector::empty(), 2)
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 0);
    }
}
