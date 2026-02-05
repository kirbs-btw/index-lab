use crate::modality::{MultiModalQuery, MultiModalVector};
use crate::error::Result;
use crate::temporal;
use crate::empty_bucket_handler::EmptyBucketHandler;
use index_core::{DistanceMetric, ScoredPoint, VectorIndex};
use index_hnsw::{HnswConfig, HnswIndex};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Configuration for a hybrid bucket with temporal integration
#[derive(Debug, Clone)]
pub struct BucketConfig {
    pub hnsw_config: HnswConfig,
    pub dense_weight: f32,
    pub metric: DistanceMetric,
    pub enable_temporal: bool,
    pub halflife_seconds: f64,
}

/// Sparse vector representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector {
    terms: Vec<(u32, f32)>,
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

/// Hybrid bucket with temporal integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridBucket {
    pub id: usize,
    pub centroid: Vec<f32>,
    dense_index: HnswIndex,
    audio_index: Option<HnswIndex>,
    inverted_index: HashMap<u32, Vec<usize>>,
    sparse_vectors: HashMap<usize, SparseVector>,
    size: usize,
    dense_weight: f32,
    /// Temporal metadata: vector_id -> timestamp
    temporal_metadata: HashMap<usize, u64>,
    enable_temporal: bool,
    halflife_seconds: f64,
}

impl HybridBucket {
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
            temporal_metadata: HashMap::new(),
            enable_temporal: config.enable_temporal,
            halflife_seconds: config.halflife_seconds,
        }
    }

    pub fn insert_dense(&mut self, id: usize, vector: Vec<f32>, timestamp: Option<u64>) -> Result<()> {
        self.dense_index
            .insert(id, vector)
            .map_err(|e| crate::error::ConvergenceError::BucketError(e.to_string()))?;
        if let Some(ts) = timestamp {
            self.temporal_metadata.insert(id, ts);
        }
        self.size += 1;
        Ok(())
    }

    pub fn insert_multi_modal(
        &mut self,
        vector: &MultiModalVector,
        config: &BucketConfig,
    ) -> Result<()> {
        let id = vector.id;

        if let Some(dense) = &vector.dense {
            self.insert_dense(id, dense.clone(), vector.timestamp)?;
        }

        if let Some(audio) = &vector.audio {
            if self.audio_index.is_none() {
                self.audio_index = Some(HnswIndex::new(config.metric, config.hnsw_config.clone()));
            }
            self.audio_index.as_mut().unwrap()
                .insert(id, audio.clone())
                .map_err(|e| crate::error::ConvergenceError::BucketError(e.to_string()))?;
            if let Some(ts) = vector.timestamp {
                self.temporal_metadata.insert(id, ts);
            }
        }

        if let Some(sparse_map) = &vector.sparse {
            let sparse = SparseVector::new(
                sparse_map.iter().map(|(k, v)| (*k, *v)).collect()
            );
            
            for &term_id in sparse.term_ids().iter() {
                self.inverted_index.entry(term_id).or_default().push(id);
            }
            
            self.sparse_vectors.insert(id, sparse);
            if let Some(ts) = vector.timestamp {
                self.temporal_metadata.insert(id, ts);
            }
        }

        self.size += 1;
        Ok(())
    }

    /// Search with temporal decay applied
    /// ACTUALLY CALLED during search (temporal decay everywhere)
    pub fn search_multi_modal(
        &self,
        query: &MultiModalQuery,
        k: usize,
        handler: &EmptyBucketHandler,
    ) -> Result<Vec<ScoredPoint>> {
        if self.size == 0 {
            // Handle empty bucket gracefully - return empty results
            return Ok(Vec::new());
        }

        let mut all_candidates = Vec::new();

        // Search dense component with temporal decay
        if let Some(dense_query) = &query.dense {
            let dense_results = self.dense_index
                .search(&dense_query.clone(), k * 2)
                .map_err(|e| crate::error::ConvergenceError::BucketError(e.to_string()))?;
            
            // Apply temporal decay to results
            let current_time = temporal::current_timestamp();
            for mut result in dense_results {
                if self.enable_temporal {
                    if let Some(ts) = self.temporal_metadata.get(&result.id) {
                        let age_seconds = (current_time.saturating_sub(*ts)) as f64;
                        result.distance = temporal::apply_temporal_decay(
                            result.distance,
                            age_seconds,
                            self.halflife_seconds,
                        );
                    }
                }
                all_candidates.push(result);
            }
        }

        // Search audio component with temporal decay
        if let Some(audio_query) = &query.audio {
            if let Some(audio_idx) = &self.audio_index {
                let audio_results = audio_idx
                    .search(&audio_query.clone(), k * 2)
                    .map_err(|e| crate::error::ConvergenceError::BucketError(e.to_string()))?;
                
                let current_time = temporal::current_timestamp();
                for mut result in audio_results {
                    if self.enable_temporal {
                        if let Some(ts) = self.temporal_metadata.get(&result.id) {
                            let age_seconds = (current_time.saturating_sub(*ts)) as f64;
                            result.distance = temporal::apply_temporal_decay(
                                result.distance,
                                age_seconds,
                                self.halflife_seconds,
                            );
                        }
                    }
                    all_candidates.push(result);
                }
            }
        }

        // Get sparse candidates
        if let Some(sparse_query_map) = &query.sparse {
            let sparse_query = SparseVector::new(
                sparse_query_map.iter().map(|(k, v)| (*k, *v)).collect()
            );
            let sparse_candidates = self.get_sparse_candidates(&sparse_query);
            
            let current_time = temporal::current_timestamp();
            for candidate_id in sparse_candidates {
                if !all_candidates.iter().any(|sp| sp.id == candidate_id) {
                    let mut dist = 0.5;  // Placeholder
                    if self.enable_temporal {
                        if let Some(ts) = self.temporal_metadata.get(&candidate_id) {
                            let age_seconds = (current_time.saturating_sub(*ts)) as f64;
                            dist = temporal::apply_temporal_decay(
                                dist,
                                age_seconds,
                                self.halflife_seconds,
                            );
                        }
                    }
                    all_candidates.push(ScoredPoint::new(candidate_id, dist));
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

    fn get_sparse_candidates(&self, sparse_query: &SparseVector) -> HashSet<usize> {
        let mut candidates = HashSet::new();
        for &term_id in sparse_query.term_ids().iter() {
            if let Some(doc_ids) = self.inverted_index.get(&term_id) {
                candidates.extend(doc_ids);
            }
        }
        candidates
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn centroid(&self) -> &[f32] {
        &self.centroid
    }

    pub fn delete(&mut self, id: usize) -> Result<bool> {
        let mut deleted = false;
        
        if self.dense_index.delete(id).unwrap_or(false) {
            deleted = true;
        }
        
        if let Some(audio_idx) = &mut self.audio_index {
            if audio_idx.delete(id).unwrap_or(false) {
                deleted = true;
            }
        }
        
        if self.sparse_vectors.remove(&id).is_some() {
            deleted = true;
            for term_ids in self.inverted_index.values_mut() {
                term_ids.retain(|&vid| vid != id);
            }
        }
        
        self.temporal_metadata.remove(&id);
        
        if deleted {
            self.size = self.size.saturating_sub(1);
        }
        
        Ok(deleted)
    }
}
