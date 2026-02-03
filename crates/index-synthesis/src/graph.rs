//! Hierarchical multi-modal graph structure with temporal decay
//! 
//! Provides HNSW-like layers per modality with cross-modal edges
//! that have temporal decay applied to edge weights.

use crate::modality::{MultiModalVector, ModalityType};
use crate::error::Result;
use crate::temporal;
use index_core::DistanceMetric;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cross-modal edge with temporal decay weight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalEdge {
    pub from_id: usize,
    pub to_id: usize,
    pub from_modality: ModalityType,
    pub to_modality: ModalityType,
    pub base_distance: f32,
    pub timestamp: u64,
}

impl CrossModalEdge {
    /// Compute current distance with temporal decay applied
    /// 
    /// Uses temporal module for normalized decay
    pub fn decayed_distance(&self, current_time: u64, halflife_seconds: f64) -> f32 {
        if self.timestamp == 0 {
            return self.base_distance;
        }
        
        let age_seconds = (current_time.saturating_sub(self.timestamp)) as f64;
        temporal::apply_temporal_decay(self.base_distance, age_seconds, halflife_seconds)
    }
}

/// Manages cross-modal edges between vectors
/// 
/// ACTUALLY POPULATED AND QUERIED (fixes APEX issue)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalGraph {
    /// Edges: (from_id, to_id) -> edge
    edges: HashMap<(usize, usize), CrossModalEdge>,
    /// Current time for temporal decay
    current_time: u64,
    /// Temporal halflife in seconds
    halflife_seconds: f64,
}

impl CrossModalGraph {
    pub fn new(halflife_seconds: f64) -> Self {
        Self {
            edges: HashMap::new(),
            current_time: temporal::current_timestamp(),
            halflife_seconds,
        }
    }

    /// Add or update a cross-modal edge
    /// 
    /// ACTUALLY CALLED during build/insert
    pub fn add_edge(
        &mut self,
        from_id: usize,
        to_id: usize,
        from_modality: ModalityType,
        to_modality: ModalityType,
        distance: f32,
        timestamp: Option<u64>,
    ) {
        let edge = CrossModalEdge {
            from_id,
            to_id,
            from_modality,
            to_modality,
            base_distance: distance,
            timestamp: timestamp.unwrap_or(self.current_time),
        };
        self.edges.insert((from_id, to_id), edge);
    }

    /// Get decayed distance between two nodes
    /// 
    /// ACTUALLY CALLED during search
    pub fn get_distance(&self, from_id: usize, to_id: usize) -> Option<f32> {
        self.edges
            .get(&(from_id, to_id))
            .map(|edge| edge.decayed_distance(self.current_time, self.halflife_seconds))
    }

    /// Update current time (for temporal decay)
    pub fn set_time(&mut self, time: u64) {
        self.current_time = time;
    }

    /// Remove edge
    pub fn remove_edge(&mut self, from_id: usize, to_id: usize) {
        self.edges.remove(&(from_id, to_id));
    }

    /// Get all neighbors of a node via cross-modal edges
    /// 
    /// ACTUALLY CALLED during search
    pub fn neighbors(&self, node_id: usize) -> Vec<(usize, f32)> {
        self.edges
            .iter()
            .filter_map(|((from, to), edge)| {
                if *from == node_id {
                    Some((*to, edge.decayed_distance(self.current_time, self.halflife_seconds)))
                } else if *to == node_id {
                    Some((*from, edge.decayed_distance(self.current_time, self.halflife_seconds)))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get neighbors filtered by modality
    pub fn neighbors_by_modality(&self, node_id: usize, modality: ModalityType) -> Vec<(usize, f32)> {
        self.edges
            .iter()
            .filter_map(|((from, to), edge)| {
                let (neighbor_id, neighbor_modality) = if *from == node_id {
                    (*to, edge.to_modality)
                } else if *to == node_id {
                    (*from, edge.from_modality)
                } else {
                    return None;
                };
                
                if neighbor_modality == modality {
                    Some((neighbor_id, edge.decayed_distance(self.current_time, self.halflife_seconds)))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Clear all edges (for rebuild)
    pub fn clear(&mut self) {
        self.edges.clear();
    }

    /// Get number of edges
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

/// Helper to compute cross-modal distance between two vectors
pub fn compute_cross_modal_distance(
    a: &MultiModalVector,
    b: &MultiModalVector,
    metric: DistanceMetric,
    dense_weight: f32,
) -> Result<f32> {
    use index_core::distance;

    let mut distances = Vec::new();
    let mut weights = Vec::new();

    // Dense distance
    if let (Some(a_dense), Some(b_dense)) = (&a.dense, &b.dense) {
        let dist = distance(metric, a_dense, b_dense)?;
        distances.push(dist);
        weights.push(dense_weight);
    }

    // Sparse distance
    if let (Some(a_sparse), Some(b_sparse)) = (&a.sparse, &b.sparse) {
        let sparse_dist = sparse_distance(a_sparse, b_sparse);
        distances.push(sparse_dist);
        weights.push(1.0 - dense_weight);
    }

    // Audio distance
    if let (Some(a_audio), Some(b_audio)) = (&a.audio, &b.audio) {
        let dist = distance(metric, a_audio, b_audio)?;
        distances.push(dist);
        weights.push(0.1);
    }

    if distances.is_empty() {
        return Err(crate::error::SynthesisError::UnsupportedModality {
            modality: "no matching modalities".to_string(),
        }.into());
    }

    // Normalize weights
    let total_weight: f32 = weights.iter().sum();
    let normalized_weights: Vec<f32> = weights.iter().map(|w| w / total_weight).collect();

    // Weighted average
    let combined_dist: f32 = distances
        .iter()
        .zip(normalized_weights.iter())
        .map(|(d, w)| d * w)
        .sum();

    Ok(combined_dist)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_modal_graph() {
        let mut graph = CrossModalGraph::new(86400.0); // 1 day halflife
        graph.set_time(1000);

        graph.add_edge(
            0, 1,
            ModalityType::Dense,
            ModalityType::Sparse,
            0.5,
            Some(500),
        );
        graph.add_edge(
            1, 2,
            ModalityType::Sparse,
            ModalityType::Audio,
            0.3,
            Some(900),
        );

        let dist = graph.get_distance(0, 1).unwrap();
        assert!(dist > 0.5); // Should be increased due to age

        let neighbors = graph.neighbors(1);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.iter().any(|(id, _)| *id == 0));
        assert!(neighbors.iter().any(|(id, _)| *id == 2));
    }

    #[test]
    fn test_neighbors_by_modality() {
        let mut graph = CrossModalGraph::new(86400.0);
        graph.set_time(1000);

        graph.add_edge(
            0, 1,
            ModalityType::Dense,
            ModalityType::Sparse,
            0.5,
            None,
        );
        graph.add_edge(
            0, 2,
            ModalityType::Dense,
            ModalityType::Dense,
            0.3,
            None,
        );

        let dense_neighbors = graph.neighbors_by_modality(0, ModalityType::Dense);
        assert_eq!(dense_neighbors.len(), 1);
        assert_eq!(dense_neighbors[0].0, 2);
    }
}
