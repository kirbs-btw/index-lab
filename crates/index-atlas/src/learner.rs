use rand::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::{AtlasError, Result};

/// Prediction result from the cluster router
#[derive(Debug, Clone)]
pub struct ClusterPrediction {
    /// Probability distribution over clusters
    pub probabilities: Vec<f32>,
    /// Maximum confidence score
    pub max_confidence: f32,
}

/// Two-layer MLP for predicting which clusters contain relevant results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterRouter {
    // Network weights
    w1: Vec<Vec<f32>>,  // input_dim × hidden_dim
    w2: Vec<Vec<f32>>,  // hidden_dim × num_clusters
    b1: Vec<f32>,       // hidden_dim
    b2: Vec<f32>,       // num_clusters

    // Configuration
    learning_rate: f32,
    training_samples: usize,

    // Dimensions
    input_dim: usize,
    hidden_dim: usize,
    num_clusters: usize,
}

impl ClusterRouter {
    /// Create a new cluster router with Xavier initialization
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        num_clusters: usize,
        learning_rate: f32,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        // Xavier initialization
        let scale_1 = (2.0 / (input_dim + hidden_dim) as f32).sqrt();
        let scale_2 = (2.0 / (hidden_dim + num_clusters) as f32).sqrt();

        // Initialize W1: input_dim × hidden_dim
        let w1 = (0..input_dim)
            .map(|_| {
                (0..hidden_dim)
                    .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale_1)
                    .collect()
            })
            .collect();

        // Initialize W2: hidden_dim × num_clusters
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
            learning_rate,
            training_samples: 0,
            input_dim,
            hidden_dim,
            num_clusters,
        }
    }

    /// Forward pass: predict cluster probabilities
    pub fn predict(&self, query: &[f32]) -> Result<ClusterPrediction> {
        if query.len() != self.input_dim {
            return Err(AtlasError::DimensionMismatch {
                expected: self.input_dim,
                actual: query.len(),
            });
        }

        // Layer 1: h = ReLU(query × W1 + b1)
        let mut hidden = vec![0.0; self.hidden_dim];
        for j in 0..self.hidden_dim {
            let mut sum = self.b1[j];
            for i in 0..self.input_dim {
                sum += query[i] * self.w1[i][j];
            }
            hidden[j] = sum.max(0.0); // ReLU
        }

        // Layer 2: logits = h × W2 + b2
        let mut logits = vec![0.0; self.num_clusters];
        for k in 0..self.num_clusters {
            let mut sum = self.b2[k];
            for j in 0..self.hidden_dim {
                sum += hidden[j] * self.w2[j][k];
            }
            logits[k] = sum;
        }

        // Softmax
        let probabilities = softmax(&logits);
        let max_confidence = probabilities
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        Ok(ClusterPrediction {
            probabilities,
            max_confidence,
        })
    }

    /// Get top-k cluster predictions
    pub fn route(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
        let pred = self.predict(query)?;

        let mut clusters: Vec<_> = pred
            .probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();

        // Sort by probability (descending)
        clusters.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k
        clusters.truncate(k.min(self.num_clusters));

        Ok(clusters)
    }

    /// Online learning: update weights based on feedback
    /// 
    /// `true_clusters`: cluster IDs that actually contained relevant results
    pub fn update(&mut self, query: &[f32], true_clusters: &[usize]) -> Result<()> {
        if query.len() != self.input_dim {
            return Err(AtlasError::DimensionMismatch {
                expected: self.input_dim,
                actual: query.len(),
            });
        }

        if true_clusters.is_empty() {
            return Ok(()); // Nothing to learn
        }

        // Create target distribution (uniform over true clusters)
        let mut target = vec![0.0; self.num_clusters];
        let weight_per_cluster = 1.0 / true_clusters.len() as f32;
        for &cluster_id in true_clusters {
            if cluster_id < self.num_clusters {
                target[cluster_id] = weight_per_cluster;
            }
        }

        // Forward pass (keeping intermediate values for backprop)
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
        // Output layer gradients: (prob - target)
        let output_grad: Vec<f32> = probs
            .iter()
            .zip(target.iter())
            .map(|(p, t)| p - t)
            .collect();

        // Update W2 and b2
        for j in 0..self.hidden_dim {
            for k in 0..self.num_clusters {
                self.w2[j][k] -= self.learning_rate * output_grad[k] * hidden[j];
            }
        }
        for k in 0..self.num_clusters {
            self.b2[k] -= self.learning_rate * output_grad[k];
        }

        // Hidden layer gradients (with ReLU derivative)
        let mut hidden_grad = vec![0.0; self.hidden_dim];
        for j in 0..self.hidden_dim {
            let mut grad = 0.0;
            for k in 0..self.num_clusters {
                grad += output_grad[k] * self.w2[j][k];
            }
            // ReLU derivative: gradient only flows if hidden[j] > 0
            hidden_grad[j] = if hidden[j] > 0.0 { grad } else { 0.0 };
        }

        // Update W1 and b1
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

    /// Get number of training samples processed
    pub fn training_samples(&self) -> usize {
        self.training_samples
    }

    /// Get dimensions
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.input_dim, self.hidden_dim, self.num_clusters)
    }
}

/// Apply softmax to a vector of logits
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }

    let max_logit = logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let exp_sums: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum: f32 = exp_sums.iter().sum();

    exp_sums.iter().map(|&e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_router_initialization() {
        let router = ClusterRouter::new(64, 128, 10, 0.001, 42);
        assert_eq!(router.dimensions(), (64, 128, 10));
        assert_eq!(router.training_samples(), 0);
    }

    #[test]
    fn test_predict_valid_probabilities() {
        let router = ClusterRouter::new(64, 128, 10, 0.001, 42);
        let query = vec![0.5; 64];

        let pred = router.predict(&query).unwrap();

        // Check probabilities sum to 1.0
        let sum: f32 = pred.probabilities.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-5);

        // Check all probabilities are in [0, 1]
        for &p in &pred.probabilities {
            assert!(p >= 0.0 && p <= 1.0);
        }

        // Check max_confidence is valid
        assert!(pred.max_confidence >= 0.0 && pred.max_confidence <= 1.0);
    }

    #[test]
    fn test_route_top_k() {
        let router = ClusterRouter::new(64, 128, 10, 0.001, 42);
        let query = vec![0.5; 64];

        let top_3 = router.route(&query, 3).unwrap();

        assert_eq!(top_3.len(), 3);

        // Check returned in descending order of probability
        for i in 0..top_3.len() - 1 {
            assert!(top_3[i].1 >= top_3[i + 1].1);
        }

        // Check cluster IDs are valid
        for &(cluster_id, _) in &top_3 {
            assert!(cluster_id < 10);
        }
    }

    #[test]
    fn test_dimension_mismatch() {
        let router = ClusterRouter::new(64, 128, 10, 0.001, 42);
        let wrong_query = vec![0.5; 32]; // Wrong dimension

        let result = router.predict(&wrong_query);
        assert!(result.is_err());
    }

    #[test]
    fn test_online_learning() {
        let mut router = ClusterRouter::new(64, 128, 10, 0.01, 42);
        let query = vec![0.5; 64];

        // Get initial prediction for cluster 0
        let pred_before = router.predict(&query).unwrap();
        let prob_0_before = pred_before.probabilities[0];

        // Update with feedback that cluster 0 is relevant
        for _ in 0..10 {
            router.update(&query, &[0]).unwrap();
        }

        // Get prediction after training
        let pred_after = router.predict(&query).unwrap();
        let prob_0_after = pred_after.probabilities[0];

        // Probability for cluster 0 should increase
        assert!(prob_0_after > prob_0_before);
        assert_eq!(router.training_samples(), 10);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Sum should be 1.0
        let sum: f32 = probs.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

        // Higher logit should have higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_empty() {
        let logits = vec![];
        let probs = softmax(&logits);
        assert_eq!(probs.len(), 0);
    }
}
