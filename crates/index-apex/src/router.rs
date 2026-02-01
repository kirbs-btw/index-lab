//! Multi-modal learned router for cluster prediction
//! 
//! Extends ATLAS's router to handle multi-modal queries (dense/sparse/audio)
//! and integrates with shift detection for adaptive learning.

use crate::modality::{ModalityType, MultiModalQuery};
use crate::error::Result;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Prediction result from the cluster router
#[derive(Debug, Clone)]
pub struct ClusterPrediction {
    /// Probability distribution over clusters
    pub probabilities: Vec<f32>,
    /// Maximum confidence score
    pub max_confidence: f32,
}

/// Multi-modal cluster router that handles dense/sparse/audio queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalRouter {
    // Dense router (for dense queries)
    dense_router: Option<ClusterRouter>,
    // Sparse router (for sparse queries) - uses term frequencies
    sparse_router: Option<SparseRouter>,
    // Audio router (for audio queries)
    audio_router: Option<ClusterRouter>,
    
    // Configuration
    learning_rate: f32,
    hidden_dim: usize,
    num_clusters: usize,
    seed: u64,
}

/// Standard cluster router (for dense/audio vectors)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClusterRouter {
    w1: Vec<Vec<f32>>, // input_dim × hidden_dim
    w2: Vec<Vec<f32>>, // hidden_dim × num_clusters
    b1: Vec<f32>,
    b2: Vec<f32>,
    input_dim: usize,
    hidden_dim: usize,
    num_clusters: usize,
    learning_rate: f32,
    training_samples: usize,
}

/// Sparse router (for sparse queries using term frequencies)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SparseRouter {
    // Term frequency → cluster mapping
    term_cluster_weights: std::collections::HashMap<u32, Vec<f32>>, // term_id → cluster_weights
    num_clusters: usize,
    learning_rate: f32,
}

impl ClusterRouter {
    fn new(input_dim: usize, hidden_dim: usize, num_clusters: usize, learning_rate: f32, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let scale_1 = (2.0 / (input_dim + hidden_dim) as f32).sqrt();
        let scale_2 = (2.0 / (hidden_dim + num_clusters) as f32).sqrt();

        let w1 = (0..input_dim)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale_1)
                    .collect()
            })
            .collect();

        let w2 = (0..hidden_dim)
            .map(|_| {
                (0..num_clusters)
                    .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale_2)
                    .collect()
            })
            .collect();

        Self {
            w1,
            w2,
            b1: vec![0.0; hidden_dim],
            b2: vec![0.0; num_clusters],
            input_dim,
            hidden_dim,
            num_clusters,
            learning_rate,
            training_samples: 0,
        }
    }

    fn predict(&self, query: &[f32]) -> Result<Vec<f32>> {
        if query.len() != self.input_dim {
            return Err(crate::error::ApexError::RouterTrainingError(
                format!("dimension mismatch: expected {}, got {}", self.input_dim, query.len())
            ));
        }

        // Layer 1: ReLU
        let mut hidden = vec![0.0; self.hidden_dim];
        for j in 0..self.hidden_dim {
            let mut sum = self.b1[j];
            for i in 0..self.input_dim {
                sum += query[i] * self.w1[i][j];
            }
            hidden[j] = sum.max(0.0);
        }

        // Layer 2: logits
        let mut logits = vec![0.0; self.num_clusters];
        for k in 0..self.num_clusters {
            let mut sum = self.b2[k];
            for j in 0..self.hidden_dim {
                sum += hidden[j] * self.w2[j][k];
            }
            logits[k] = sum;
        }

        Ok(softmax(&logits))
    }

    fn update(&mut self, query: &[f32], target: &[f32]) -> Result<()> {
        if query.len() != self.input_dim || target.len() != self.num_clusters {
            return Err(crate::error::ApexError::RouterTrainingError(
                "dimension mismatch in update".to_string()
            ));
        }

        // Forward pass
        let mut hidden = vec![0.0; self.hidden_dim];
        for j in 0..self.hidden_dim {
            let mut sum = self.b1[j];
            for i in 0..self.input_dim {
                sum += query[i] * self.w1[i][j];
            }
            hidden[j] = sum.max(0.0);
        }

        let mut logits = vec![0.0; self.num_clusters];
        for k in 0..self.num_clusters {
            let mut sum = self.b2[k];
            for j in 0..self.hidden_dim {
                sum += hidden[j] * self.w2[j][k];
            }
            logits[k] = sum;
        }

        let probs = softmax(&logits);

        // Backpropagation (simplified SGD)
        let output_grad: Vec<f32> = probs.iter().zip(target.iter())
            .map(|(p, t)| p - t)
            .collect();

        // Update W2, b2
        for j in 0..self.hidden_dim {
            for k in 0..self.num_clusters {
                self.w2[j][k] -= self.learning_rate * output_grad[k] * hidden[j];
            }
        }
        for k in 0..self.num_clusters {
            self.b2[k] -= self.learning_rate * output_grad[k];
        }

        // Hidden layer gradients
        let mut hidden_grad = vec![0.0; self.hidden_dim];
        for j in 0..self.hidden_dim {
            let mut grad = 0.0;
            for k in 0..self.num_clusters {
                grad += output_grad[k] * self.w2[j][k];
            }
            hidden_grad[j] = if hidden[j] > 0.0 { grad } else { 0.0 };
        }

        // Update W1, b1
        for i in 0..self.input_dim {
            for j in 0..self.hidden_dim {
                self.w1[i][j] -= self.learning_rate * hidden_grad[j] * query[i];
            }
        }
        for j in 0..self.hidden_dim {
            self.b1[j] -= self.learning_rate * hidden_grad[j];
        }

        self.training_samples += 1;
        Ok(())
    }
}

impl SparseRouter {
    fn new(num_clusters: usize, learning_rate: f32) -> Self {
        Self {
            term_cluster_weights: std::collections::HashMap::new(),
            num_clusters,
            learning_rate,
        }
    }

    fn predict(&self, query: &std::collections::HashMap<u32, f32>) -> Vec<f32> {
        let mut cluster_scores = vec![0.0; self.num_clusters];
        let mut total_weight = 0.0;

        for (term_id, weight) in query {
            if let Some(cluster_weights) = self.term_cluster_weights.get(term_id) {
                for (k, &w) in cluster_weights.iter().enumerate() {
                    if k < cluster_scores.len() {
                        cluster_scores[k] += weight * w;
                    }
                }
                total_weight += weight;
            }
        }

        // Normalize
        if total_weight > 0.0 {
            for score in &mut cluster_scores {
                *score /= total_weight;
            }
        } else {
            // Uniform distribution if no matching terms
            let uniform = 1.0 / self.num_clusters as f32;
            cluster_scores.fill(uniform);
        }

        softmax(&cluster_scores)
    }

    fn update(&mut self, query: &std::collections::HashMap<u32, f32>, target: &[f32]) {
        for (term_id, weight) in query {
            let cluster_weights = self.term_cluster_weights
                .entry(*term_id)
                .or_insert_with(|| vec![0.0; self.num_clusters]);

            // Update weights towards target
            for (k, &target_val) in target.iter().enumerate() {
                if k < cluster_weights.len() {
                    cluster_weights[k] += self.learning_rate * weight * (target_val - cluster_weights[k]);
                }
            }
        }
    }
}

impl MultiModalRouter {
    /// Create a new multi-modal router
    pub fn new(
        dense_dim: Option<usize>,
        audio_dim: Option<usize>,
        hidden_dim: usize,
        num_clusters: usize,
        learning_rate: f32,
        seed: u64,
    ) -> Self {
        Self {
            dense_router: dense_dim.map(|dim| ClusterRouter::new(dim, hidden_dim, num_clusters, learning_rate, seed)),
            sparse_router: Some(SparseRouter::new(num_clusters, learning_rate)),
            audio_router: audio_dim.map(|dim| ClusterRouter::new(dim, hidden_dim, num_clusters, learning_rate, seed.wrapping_add(1))),
            learning_rate,
            hidden_dim,
            num_clusters,
            seed,
        }
    }

    /// Predict cluster probabilities for a multi-modal query
    pub fn predict(&self, query: &MultiModalQuery) -> Result<ClusterPrediction> {
        let mut all_probs = Vec::new();
        let mut weights = Vec::new();

        // Dense prediction
        if let Some(dense_query) = &query.dense {
            if let Some(router) = &self.dense_router {
                let probs = router.predict(dense_query)?;
                all_probs.push(probs);
                weights.push(0.6); // Dense weight
            }
        }

        // Sparse prediction
        if let Some(sparse_query) = &query.sparse {
            if let Some(router) = &self.sparse_router {
                let probs = router.predict(sparse_query);
                all_probs.push(probs);
                weights.push(0.3); // Sparse weight
            }
        }

        // Audio prediction
        if let Some(audio_query) = &query.audio {
            if let Some(router) = &self.audio_router {
                let probs = router.predict(audio_query)?;
                all_probs.push(probs);
                weights.push(0.1); // Audio weight
            }
        }

        if all_probs.is_empty() {
            // No modalities, return uniform distribution
            let uniform = vec![1.0 / self.num_clusters as f32; self.num_clusters];
            return Ok(ClusterPrediction {
                probabilities: uniform,
                max_confidence: 1.0 / self.num_clusters as f32,
            });
        }

        // Weighted average of predictions
        let total_weight: f32 = weights.iter().sum();
        let normalized_weights: Vec<f32> = weights.iter().map(|w| w / total_weight).collect();

        let mut fused_probs = vec![0.0; self.num_clusters];
        for (probs, &weight) in all_probs.iter().zip(normalized_weights.iter()) {
            for (k, &p) in probs.iter().enumerate() {
                if k < fused_probs.len() {
                    fused_probs[k] += weight * p;
                }
            }
        }

        let max_confidence = fused_probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        Ok(ClusterPrediction {
            probabilities: fused_probs,
            max_confidence,
        })
    }

    /// Get top-k cluster predictions
    pub fn route(&self, query: &MultiModalQuery, k: usize) -> Result<Vec<(usize, f32)>> {
        let pred = self.predict(query)?;

        let mut clusters: Vec<_> = pred.probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();

        clusters.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        clusters.truncate(k.min(self.num_clusters));

        Ok(clusters)
    }

    /// Update router with feedback
    pub fn update(&mut self, query: &MultiModalQuery, true_clusters: &[usize]) -> Result<()> {
        if true_clusters.is_empty() {
            return Ok(());
        }

        // Create target distribution
        let mut target = vec![0.0; self.num_clusters];
        let weight_per_cluster = 1.0 / true_clusters.len() as f32;
        for &cluster_id in true_clusters {
            if cluster_id < self.num_clusters {
                target[cluster_id] = weight_per_cluster;
            }
        }

        // Update each modality router
        if let Some(dense_query) = &query.dense {
            if let Some(router) = &mut self.dense_router {
                router.update(dense_query, &target)?;
            }
        }

        if let Some(sparse_query) = &query.sparse {
            if let Some(router) = &mut self.sparse_router {
                router.update(sparse_query, &target);
            }
        }

        if let Some(audio_query) = &query.audio {
            if let Some(router) = &mut self.audio_router {
                router.update(audio_query, &target)?;
            }
        }

        Ok(())
    }

    /// Initialize routers when dimensions are known
    pub fn initialize_modalities(&mut self, dense_dim: Option<usize>, audio_dim: Option<usize>) {
        if self.dense_router.is_none() && dense_dim.is_some() {
            self.dense_router = Some(ClusterRouter::new(
                dense_dim.unwrap(),
                self.hidden_dim,
                self.num_clusters,
                self.learning_rate,
                self.seed,
            ));
        }

        if self.audio_router.is_none() && audio_dim.is_some() {
            self.audio_router = Some(ClusterRouter::new(
                audio_dim.unwrap(),
                self.hidden_dim,
                self.num_clusters,
                self.learning_rate,
                self.seed.wrapping_add(1),
            ));
        }
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sums: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exp_sums.iter().sum();

    exp_sums.iter().map(|&e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_multimodal_router_dense_only() {
        let mut router = MultiModalRouter::new(
            Some(64),
            None,
            128,
            10,
            0.001,
            42,
        );

        let query = MultiModalQuery::with_dense(vec![0.5; 64]);
        let pred = router.predict(&query).unwrap();

        assert_eq!(pred.probabilities.len(), 10);
        let sum: f32 = pred.probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_multimodal_router_hybrid() {
        let mut router = MultiModalRouter::new(
            Some(64),
            None,
            128,
            10,
            0.001,
            42,
        );

        let mut sparse = HashMap::new();
        sparse.insert(1, 0.5);
        sparse.insert(2, 0.3);
        let query = MultiModalQuery::with_hybrid(vec![0.5; 64], sparse);

        let pred = router.predict(&query).unwrap();
        assert_eq!(pred.probabilities.len(), 10);
    }

    #[test]
    fn test_route_top_k() {
        let mut router = MultiModalRouter::new(
            Some(64),
            None,
            128,
            10,
            0.001,
            42,
        );

        let query = MultiModalQuery::with_dense(vec![0.5; 64]);
        let top_3 = router.route(&query, 3).unwrap();

        assert_eq!(top_3.len(), 3);
        for i in 0..top_3.len() - 1 {
            assert!(top_3[i].1 >= top_3[i + 1].1);
        }
    }
}
