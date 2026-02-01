use crate::modality::{MultiModalQuery, MultiModalVector};
use crate::error::Result;
use index_core::{DistanceMetric, ScoredPoint, VectorIndex};
use index_hnsw::{HnswConfig, HnswIndex};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Configuration for a hybrid bucket
#[derive(Debug, Clone)]
pub struct BucketConfig {
    pub hnsw_config: HnswConfig,
    pub dense_weight: f32,
    pub metric: DistanceMetric,
}

/// Sparse vector representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector {
    terms: Vec<(u32, f32)>, // term_id -> weight
}

impl SparseVector {
    pub fn new(terms: Vec<(u32, f32)>) -> Self {
        Self { terms }
    }

    pub fn empty() -> Self {
        Self { terms: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn term_ids(&self) -> Vec<u32> {
        self.terms.iter().map(|(id, _)| *id).collect()
    }

    pub fn cosine_similarity(&self, other: &SparseVector) -> f32 {
        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;

        let a_map: HashMap<u32, f32> = self.terms.iter().cloned().collect();
        let b_map: HashMap<u32, f32> = other.terms.iter().cloned().collect();

        for (term_id, weight_a) in &a_map {
            norm_a += weight_a * weight_a;
            if let Some(weight_b) = b_map.get(term_id) {
                dot_product += weight_a * weight_b;
            }
        }

        for weight_b in b_map.values() {
            norm_b += weight_b * weight_b;
        }

        norm_a = norm_a.sqrt();
        norm_b = norm_b.sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }
}

/// Hybrid bucket containing dense (HNSW), sparse (inverted index), and audio (HNSW) components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridBucket {
    /// Bucket ID
    pub id: usize,

    /// Cluster centroid
    pub centroid: Vec<f32>,

    /// Dense vector index (mini-HNSW)
    dense_index: HnswIndex,

    /// Audio vector index (mini-HNSW)
    audio_index: Option<HnswIndex>,

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
            dense_index: HnswIndex::new(config.metric, config.hnsw_config.clone()),
            audio_index: None,
            inverted_index: HashMap::new(),
            sparse_vectors: HashMap::new(),
            size: 0,
            dense_weight: config.dense_weight,
        }
    }

    /// Insert a dense-only vector
    pub fn insert_dense(&mut self, id: usize, vector: Vec<f32>) -> Result<()> {
        self.dense_index
            .insert(id, vector)
            .map_err(|e| crate::error::ApexError::BucketError(e.to_string()))?;
        self.size += 1;
        Ok(())
    }

    /// Insert an audio vector
    pub fn insert_audio(&mut self, id: usize, vector: Vec<f32>, config: &BucketConfig) -> Result<()> {
        if self.audio_index.is_none() {
            self.audio_index = Some(HnswIndex::new(config.metric, config.hnsw_config.clone()));
        }
        
        self.audio_index.as_mut().unwrap()
            .insert(id, vector)
            .map_err(|e| crate::error::ApexError::BucketError(e.to_string()))?;
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
            .map_err(|e| crate::error::ApexError::BucketError(e.to_string()))?;

        // Update inverted index for each sparse term
        for &term_id in sparse.term_ids().iter() {
            self.inverted_index.entry(term_id).or_default().push(id);
        }

        // Store sparse vector
        self.sparse_vectors.insert(id, sparse);
        self.size += 1;
        Ok(())
    }

    /// Insert a multi-modal vector
    pub fn insert_multi_modal(
        &mut self,
        vector: &MultiModalVector,
        config: &BucketConfig,
    ) -> Result<()> {
        let id = vector.id;

        // Insert dense component if present
        if let Some(dense) = &vector.dense {
            self.dense_index
                .insert(id, dense.clone())
                .map_err(|e| crate::error::ApexError::BucketError(e.to_string()))?;
        }

        // Insert audio component if present
        if let Some(audio) = &vector.audio {
            self.insert_audio(id, audio.clone(), config)?;
        }

        // Insert sparse component if present
        if let Some(sparse_map) = &vector.sparse {
            let sparse = SparseVector::new(
                sparse_map.iter().map(|(k, v)| (*k, *v)).collect()
            );
            
            // Update inverted index
            for &term_id in sparse.term_ids().iter() {
                self.inverted_index.entry(term_id).or_default().push(id);
            }
            
            self.sparse_vectors.insert(id, sparse);
        }

        self.size += 1;
        Ok(())
    }

    /// Pure dense search using HNSW
    pub fn search_dense(&self, query: &[f32], k: usize) -> Result<Vec<ScoredPoint>> {
        self.dense_index
            .search(&query.to_vec(), k)
            .map_err(|e| crate::error::ApexError::BucketError(e.to_string()))
    }

    /// Search with multi-modal query
    pub fn search_multi_modal(
        &self,
        query: &MultiModalQuery,
        k: usize,
    ) -> Result<Vec<ScoredPoint>> {
        let mut all_candidates = Vec::new();

        // Search dense component
        if let Some(dense_query) = &query.dense {
            let dense_results = self.search_dense(dense_query, k * 2)?;
            all_candidates.extend(dense_results);
        }

        // Search audio component
        if let Some(audio_query) = &query.audio {
            if let Some(audio_idx) = &self.audio_index {
                let audio_results = audio_idx
                    .search(&audio_query.clone(), k * 2)
                    .map_err(|e| crate::error::ApexError::BucketError(e.to_string()))?;
                all_candidates.extend(audio_results);
            }
        }

        // Get sparse candidates
        if let Some(sparse_query_map) = &query.sparse {
            let sparse_query = SparseVector::new(
                sparse_query_map.iter().map(|(k, v)| (*k, *v)).collect()
            );
            let sparse_candidates = self.get_sparse_candidates(&sparse_query);
            
            // Add sparse candidates with placeholder scores
            for candidate_id in sparse_candidates {
                if !all_candidates.iter().any(|sp| sp.id == candidate_id) {
                    all_candidates.push(ScoredPoint::new(candidate_id, 0.5)); // Placeholder
                }
            }
        }

        // Deduplicate and sort
        let mut seen = HashSet::new();
        let mut unique_results: Vec<ScoredPoint> = all_candidates
            .into_iter()
            .filter(|sp| seen.insert(sp.id))
            .collect();

        unique_results.sort_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });
        unique_results.truncate(k);

        Ok(unique_results)
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
            .map_err(|e| crate::error::ApexError::BucketError(e.to_string()))?;

        // Get sparse candidates from inverted index
        let sparse_candidates = self.get_sparse_candidates(sparse_query);

        // Combine candidates (union for now)
        let mut candidate_set: HashSet<usize> = dense_candidates.iter().map(|sp| sp.id).collect();
        candidate_set.extend(sparse_candidates);

        // Compute hybrid scores for all candidates
        let mut scored_results = Vec::new();

        for candidate_id in candidate_set {
            // Get dense score
            let dense_score =
                if let Some(sp) = dense_candidates.iter().find(|sp| sp.id == candidate_id) {
                    sp.distance
                } else {
                    continue; // Skip if not in dense results
                };

            // Get sparse score
            let sparse_score = if let Some(sv) = self.sparse_vectors.get(&candidate_id) {
                1.0 - sparse_query.cosine_similarity(sv) // Convert similarity to distance
            } else {
                1.0 // Worst case if no sparse vector
            };

            // Normalize scores (simple min-max for now)
            let normalized_dense = dense_score / (dense_score + 1.0);
            let normalized_sparse = sparse_score;

            // Fuse scores
            let fused_score = self.dense_weight * normalized_dense
                + (1.0 - self.dense_weight) * normalized_sparse;

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

    /// Delete a vector from this bucket
    pub fn delete(&mut self, id: usize) -> Result<bool> {
        // Try to delete from dense index
        let dense_deleted = self.dense_index.delete(id).unwrap_or(false);
        
        // Try to delete from audio index
        let audio_deleted = if let Some(audio_idx) = &mut self.audio_index {
            audio_idx.delete(id).unwrap_or(false)
        } else {
            false
        };
        
        if dense_deleted || audio_deleted {
            // Remove from sparse vectors
            let had_sparse = self.sparse_vectors.remove(&id).is_some();
            
            // Remove from inverted index
            if had_sparse {
                for term_ids in self.inverted_index.values_mut() {
                    term_ids.retain(|&vid| vid != id);
                }
            }
            
            self.size = self.size.saturating_sub(1);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Update a vector in this bucket
    pub fn update(&mut self, id: usize, vector: Vec<f32>) -> Result<bool> {
        // Try to update in dense index
        let updated = self.dense_index.update(id, vector).unwrap_or(false);
        Ok(updated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> BucketConfig {
        BucketConfig {
            hnsw_config: HnswConfig {
                m_max: 16,
                ef_construction: 50,
                ef_search: 30,
                ml: 1.0 / 2.0_f64.ln(),
            },
            dense_weight: 0.6,
            metric: DistanceMetric::Euclidean,
        }
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
        bucket.insert_dense(0, vec![1.0; 64]).unwrap();
        bucket.insert_dense(1, vec![2.0; 64]).unwrap();
        bucket.insert_dense(2, vec![3.0; 64]).unwrap();

        assert_eq!(bucket.len(), 3);

        // Search for vector closest to [1.0; 64]
        let query_vec = vec![1.0; 64];
        let results = bucket.search_dense(&query_vec, 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 0); // Should be closest
    }
}
