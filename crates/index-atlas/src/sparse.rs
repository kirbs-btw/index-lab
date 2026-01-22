use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Sparse vector representation using term ID → weight mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector {
    /// Map of term_id → weight
    terms: HashMap<u32, f32>,
}

impl SparseVector {
    /// Create a new sparse vector from term-weight pairs
    pub fn new(terms: Vec<(u32, f32)>) -> Self {
        Self {
            terms: terms.into_iter().collect(),
        }
    }

    /// Create an empty sparse vector
    pub fn empty() -> Self {
        Self {
            terms: HashMap::new(),
        }
    }

    /// Get the weight for a specific term
    pub fn get(&self, term_id: u32) -> Option<f32> {
        self.terms.get(&term_id).copied()
    }

    /// Get all terms
    pub fn terms(&self) -> &HashMap<u32, f32> {
        &self.terms
    }

    /// Number of non-zero terms
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Compute dot product similarity with another sparse vector
    pub fn dot_product(&self, other: &SparseVector) -> f32 {
        let mut score = 0.0;

        // Iterate over the smaller vector for efficiency
        let (smaller, larger) = if self.terms.len() <= other.terms.len() {
            (&self.terms, &other.terms)
        } else {
            (&other.terms, &self.terms)
        };

        for (term_id, weight) in smaller {
            if let Some(other_weight) = larger.get(term_id) {
                score += weight * other_weight;
            }
        }

        score
    }

    /// Compute cosine similarity with another sparse vector
    pub fn cosine_similarity(&self, other: &SparseVector) -> f32 {
        let dot = self.dot_product(other);
        let norm_self = self.l2_norm();
        let norm_other = other.l2_norm();

        if norm_self == 0.0 || norm_other == 0.0 {
            0.0
        } else {
            dot / (norm_self * norm_other)
        }
    }

    /// Compute L2 norm
    pub fn l2_norm(&self) -> f32 {
        self.terms.values().map(|w| w * w).sum::<f32>().sqrt()
    }

    /// Get all term IDs
    pub fn term_ids(&self) -> Vec<u32> {
        self.terms.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sparse_vector_creation() {
        let sv = SparseVector::new(vec![(1, 0.5), (2, 0.3), (10, 0.8)]);
        assert_eq!(sv.len(), 3);
        assert_eq!(sv.get(1), Some(0.5));
        assert_eq!(sv.get(2), Some(0.3));
        assert_eq!(sv.get(10), Some(0.8));
        assert_eq!(sv.get(100), None);
    }

    #[test]
    fn test_empty_sparse_vector() {
        let sv = SparseVector::empty();
        assert_eq!(sv.len(), 0);
        assert!(sv.is_empty());
    }

    #[test]
    fn test_dot_product() {
        let sv1 = SparseVector::new(vec![(1, 1.0), (2, 2.0), (3, 3.0)]);
        let sv2 = SparseVector::new(vec![(2, 1.0), (3, 1.0), (4, 1.0)]);
        
        // (2*1) + (3*1) = 5.0
        let dot = sv1.dot_product(&sv2);
        assert_relative_eq!(dot, 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_dot_product_no_overlap() {
        let sv1 = SparseVector::new(vec![(1, 1.0), (2, 2.0)]);
        let sv2 = SparseVector::new(vec![(3, 1.0), (4, 1.0)]);
        
        let dot = sv1.dot_product(&sv2);
        assert_relative_eq!(dot, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_l2_norm() {
        let sv = SparseVector::new(vec![(1, 3.0), (2, 4.0)]);
        // sqrt(9 + 16) = 5.0
        assert_relative_eq!(sv.l2_norm(), 5.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let sv1 = SparseVector::new(vec![(1, 1.0), (2, 0.0)]);
        let sv2 = SparseVector::new(vec![(1, 1.0), (2, 0.0)]);
        
        // Identical vectors should have cosine = 1.0
        assert_relative_eq!(sv1.cosine_similarity(&sv2), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let sv1 = SparseVector::new(vec![(1, 1.0)]);
        let sv2 = SparseVector::new(vec![(2, 1.0)]);
        
        // Orthogonal vectors should have cosine = 0.0
        assert_relative_eq!(sv1.cosine_similarity(&sv2), 0.0, epsilon = 1e-6);
    }
}
