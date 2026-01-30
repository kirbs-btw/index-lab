use crate::modality::{ModalityType, MultiModalVector};
use index_core::{distance, DistanceMetric};
use std::collections::{HashMap, HashSet};

/// Unified multi-modal graph structure
#[derive(Debug, Clone)]
pub struct UnifiedGraph {
    /// Graph nodes: id -> MultiModalVector
    nodes: HashMap<usize, MultiModalVector>,
    
    /// Edges: node_id -> Vec<neighbor_ids>
    /// Edges can connect nodes across modalities
    edges: HashMap<usize, Vec<usize>>,
    
    /// Modality-specific entry points for faster routing
    entry_points: HashMap<ModalityType, Vec<usize>>,
    
    /// Maximum edges per node
    m_max: usize,
    
    metric: DistanceMetric,
}

impl UnifiedGraph {
    pub fn new(m_max: usize, metric: DistanceMetric) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            entry_points: HashMap::new(),
            m_max,
            metric,
        }
    }
    
    pub fn insert_multi_modal(&mut self, vector: MultiModalVector) -> anyhow::Result<()> {
        let id = vector.id;
        
        // Store the vector
        self.nodes.insert(id, vector.clone());
        self.edges.insert(id, Vec::new());
        
        // Update entry points for each modality
        for modality in vector.modalities() {
            self.entry_points.entry(modality).or_insert_with(Vec::new).push(id);
        }
        
        // Find nearest neighbors and connect
        if self.nodes.len() > 1 {
            self.connect_to_neighbors(id)?;
        }
        
        Ok(())
    }
    
    fn connect_to_neighbors(&mut self, new_id: usize) -> anyhow::Result<()> {
        let new_vector = self.nodes.get(&new_id).unwrap();
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        
        // Find candidates from all nodes (simplified - in production would use spatial index)
        for (id, vector) in &self.nodes {
            if *id == new_id {
                continue;
            }
            
            // Compute distance across all shared modalities
            let dist = self.compute_cross_modal_distance(new_vector, vector)?;
            candidates.push((*id, dist));
        }
        
        // Sort by distance and take top m_max
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.m_max);
        
        // Add bidirectional edges
        for (neighbor_id, _) in &candidates {
            // Add edge from new node to neighbor
            if let Some(edges) = self.edges.get_mut(&new_id) {
                edges.push(*neighbor_id);
            }
            
            // Add edge from neighbor to new node (if not full)
            if let Some(edges) = self.edges.get_mut(neighbor_id) {
                if edges.len() < self.m_max {
                    edges.push(new_id);
                } else {
                    // Prune: replace worst edge if new one is better
                    self.prune_edge(*neighbor_id, new_id)?;
                }
            }
        }
        
        Ok(())
    }
    
    fn prune_edge(&mut self, node_id: usize, new_neighbor: usize) -> anyhow::Result<()> {
        // Clone necessary data to avoid borrowing conflicts
        let node = self.nodes.get(&node_id).cloned().unwrap();
        let new_neighbor_vec = self.nodes.get(&new_neighbor).cloned().unwrap();
        let new_dist = self.compute_cross_modal_distance(&node, &new_neighbor_vec)?;
        
        // Collect neighbor IDs and compute distances before mutating edges
        let neighbor_ids: Vec<usize> = self.edges.get(&node_id)
            .map(|e| e.iter().copied().collect())
            .unwrap_or_default();
        
        let mut worst_idx = 0;
        let mut worst_dist = f32::MAX;
        
        for (idx, neighbor_id) in neighbor_ids.iter().enumerate() {
            let neighbor = self.nodes.get(neighbor_id).cloned().unwrap();
            let dist = self.compute_cross_modal_distance(&node, &neighbor)?;
            if dist > worst_dist {
                worst_dist = dist;
                worst_idx = idx;
            }
        }
        
        // Now mutate edges if needed
        if let Some(edges) = self.edges.get_mut(&node_id) {
            if new_dist < worst_dist && worst_idx < edges.len() {
                edges[worst_idx] = new_neighbor;
            }
        }
        
        Ok(())
    }
    
    fn compute_cross_modal_distance(
        &self,
        a: &MultiModalVector,
        b: &MultiModalVector,
    ) -> anyhow::Result<f32> {
        let mut distances = Vec::new();
        
        // Dense distance
        if let (Some(a_dense), Some(b_dense)) = (&a.dense, &b.dense) {
            let dist = distance(self.metric, a_dense, b_dense)?;
            distances.push((dist, 0.6));
        }
        
        // Sparse distance
        if let (Some(a_sparse), Some(b_sparse)) = (&a.sparse, &b.sparse) {
            let sparse_dist = sparse_distance(a_sparse, b_sparse);
            distances.push((sparse_dist, 0.3));
        }
        
        // Audio distance
        if let (Some(a_audio), Some(b_audio)) = (&a.audio, &b.audio) {
            let dist = distance(self.metric, a_audio, b_audio)?;
            distances.push((dist, 0.1));
        }
        
        if distances.is_empty() {
            // No shared modalities - return max distance
            return Ok(f32::MAX);
        }
        
        // Weighted average
        let total_weight: f32 = distances.iter().map(|(_, w)| w).sum();
        let combined: f32 = distances
            .iter()
            .map(|(d, w)| d * w / total_weight)
            .sum();
        
        Ok(combined)
    }
    
    pub fn search_unified(
        &self,
        query: &crate::modality::MultiModalQuery,
        ef: usize,
        _precision: Precision,
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }
        
        // Find entry point based on query modalities
        let entry_point = self.find_entry_point(query);
        
        // Greedy graph search
        let mut visited = HashSet::new();
        let mut candidates = std::collections::BinaryHeap::new();
        let mut results = Vec::new();
        
        // Initialize with entry point
        let entry_dist = self.compute_query_distance(query, entry_point)?;
        candidates.push(HeapEntry {
            id: entry_point,
            distance: entry_dist,
        });
        visited.insert(entry_point);
        results.push((entry_point, entry_dist));
        
        // Explore graph
        while let Some(current) = candidates.pop() {
            if visited.len() >= ef {
                break;
            }
            
            // Explore neighbors
            if let Some(neighbors) = self.edges.get(&current.id) {
                for &neighbor_id in neighbors {
                    if visited.insert(neighbor_id) {
                        let dist = self.compute_query_distance(query, neighbor_id)?;
                        candidates.push(HeapEntry {
                            id: neighbor_id,
                            distance: dist,
                        });
                        results.push((neighbor_id, dist));
                    }
                }
            }
        }
        
        // Sort by distance and return
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }
    
    fn find_entry_point(&self, query: &crate::modality::MultiModalQuery) -> usize {
        // Find best entry point based on query modalities
        let query_modalities = query.modalities();
        
        if let Some(first_modality) = query_modalities.first() {
            if let Some(entry_points) = self.entry_points.get(first_modality) {
                if !entry_points.is_empty() {
                    return entry_points[0];
                }
            }
        }
        
        // Fallback: use first node
        self.nodes.keys().next().copied().unwrap_or(0)
    }
    
    fn compute_query_distance(
        &self,
        query: &crate::modality::MultiModalQuery,
        node_id: usize,
    ) -> anyhow::Result<f32> {
        let node = self.nodes.get(&node_id)
            .ok_or_else(|| anyhow::anyhow!("node not found"))?;
        node.distance(query, self.metric)
    }
    
    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    
    pub fn rebuild_region(&mut self, _region: Vec<usize>) -> anyhow::Result<()> {
        // Simplified: rebuild all edges for affected nodes
        // In production, would only rebuild affected subgraph
        let node_ids: Vec<usize> = self.nodes.keys().copied().collect();
        for id in node_ids {
            if let Some(edges) = self.edges.get_mut(&id) {
                edges.clear();
            }
            self.connect_to_neighbors(id)?;
        }
        Ok(())
    }
}

fn sparse_distance(a: &std::collections::HashMap<u32, f32>, b: &std::collections::HashMap<u32, f32>) -> f32 {
    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    
    for (term_id, weight_a) in a {
        norm_a += weight_a * weight_a;
        if let Some(weight_b) = b.get(term_id) {
            dot_product += weight_a * weight_b;
        }
    }
    
    for (_, weight_b) in b {
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Precision {
    FP32,
    FP16,
    INT8,
}

impl Default for Precision {
    fn default() -> Self {
        Precision::FP32
    }
}

#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    id: usize,
    distance: f32,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && (self.distance - other.distance).abs() < 1e-6
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Min-heap (smallest distance first)
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap (smallest distance first)
        other.distance.partial_cmp(&self.distance)
            .unwrap_or_else(|| self.id.cmp(&other.id))
    }
}

impl Eq for HeapEntry {}
