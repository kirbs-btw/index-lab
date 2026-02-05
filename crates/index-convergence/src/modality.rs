use index_core::{distance, DistanceMetric, Vector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type of vector modality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModalityType {
    /// Dense embedding vectors (e.g., BERT, CLIP)
    Dense,
    /// Sparse term vectors (e.g., BM25, TF-IDF)
    Sparse,
    /// Audio embeddings (e.g., Wav2Vec)
    Audio,
}

impl ModalityType {
    pub fn all() -> Vec<ModalityType> {
        vec![ModalityType::Dense, ModalityType::Sparse, ModalityType::Audio]
    }
}

/// Multi-modal query containing vectors for different modalities
#[derive(Debug, Clone)]
pub struct MultiModalQuery {
    pub dense: Option<Vector>,
    pub sparse: Option<HashMap<u32, f32>>,
    pub audio: Option<Vector>,
}

impl MultiModalQuery {
    pub fn new() -> Self {
        Self {
            dense: None,
            sparse: None,
            audio: None,
        }
    }
    
    pub fn with_dense(dense: Vector) -> Self {
        Self {
            dense: Some(dense),
            sparse: None,
            audio: None,
        }
    }
    
    pub fn with_sparse(sparse: HashMap<u32, f32>) -> Self {
        Self {
            dense: None,
            sparse: Some(sparse),
            audio: None,
        }
    }
    
    pub fn with_hybrid(dense: Vector, sparse: HashMap<u32, f32>) -> Self {
        Self {
            dense: Some(dense),
            sparse: Some(sparse),
            audio: None,
        }
    }
    
    pub fn modalities(&self) -> Vec<ModalityType> {
        let mut mods = Vec::new();
        if self.dense.is_some() {
            mods.push(ModalityType::Dense);
        }
        if self.sparse.is_some() {
            mods.push(ModalityType::Sparse);
        }
        if self.audio.is_some() {
            mods.push(ModalityType::Audio);
        }
        mods
    }
}

impl Default for MultiModalQuery {
    fn default() -> Self {
        Self::new()
    }
}

/// Multi-modal vector storage for a single document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalVector {
    pub id: usize,
    pub dense: Option<Vector>,
    pub sparse: Option<HashMap<u32, f32>>,
    pub audio: Option<Vector>,
    /// Timestamp for temporal decay
    pub timestamp: Option<u64>,
}

impl MultiModalVector {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            dense: None,
            sparse: None,
            audio: None,
            timestamp: None,
        }
    }
    
    pub fn with_dense(id: usize, dense: Vector) -> Self {
        Self {
            id,
            dense: Some(dense),
            sparse: None,
            audio: None,
            timestamp: None,
        }
    }
    
    pub fn modalities(&self) -> Vec<ModalityType> {
        let mut mods = Vec::new();
        if self.dense.is_some() {
            mods.push(ModalityType::Dense);
        }
        if self.sparse.is_some() {
            mods.push(ModalityType::Sparse);
        }
        if self.audio.is_some() {
            mods.push(ModalityType::Audio);
        }
        mods
    }
    
    /// Compute distance between query and this vector across all modalities
    pub fn distance(
        &self,
        query: &MultiModalQuery,
        metric: DistanceMetric,
        dense_weight: f32,
    ) -> anyhow::Result<f32> {
        let mut distances = Vec::new();
        let mut weights = Vec::new();
        
        if let (Some(q_dense), Some(v_dense)) = (&query.dense, &self.dense) {
            let dist = distance(metric, q_dense, v_dense)?;
            distances.push(dist);
            weights.push(dense_weight);
        }
        
        if let (Some(q_sparse), Some(v_sparse)) = (&query.sparse, &self.sparse) {
            let sparse_dist = sparse_distance(q_sparse, v_sparse);
            distances.push(sparse_dist);
            weights.push(1.0 - dense_weight);
        }
        
        if let (Some(q_audio), Some(v_audio)) = (&query.audio, &self.audio) {
            let dist = distance(metric, q_audio, v_audio)?;
            distances.push(dist);
            weights.push(0.1);
        }
        
        if distances.is_empty() {
            return Err(anyhow::anyhow!("no matching modalities between query and vector"));
        }
        
        let total_weight: f32 = weights.iter().sum();
        let normalized_weights: Vec<f32> = weights.iter().map(|w| w / total_weight).collect();
        
        let combined_dist: f32 = distances
            .iter()
            .zip(normalized_weights.iter())
            .map(|(d, w)| d * w)
            .sum();
        
        Ok(combined_dist)
    }
}

fn sparse_distance(a: &HashMap<u32, f32>, b: &HashMap<u32, f32>) -> f32 {
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    
    for (term_id, weight_a) in a {
        norm_a += weight_a * weight_a;
        if let Some(weight_b) = b.get(term_id) {
            dot_product += weight_a * weight_b;
        }
    }
    
    for weight_b in b.values() {
        norm_b += weight_b * weight_b;
    }
    
    norm_a = norm_a.sqrt();
    norm_b = norm_b.sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }
    
    let cosine_sim = dot_product / (norm_a * norm_b);
    1.0 - cosine_sim.clamp(-1.0, 1.0)
}
