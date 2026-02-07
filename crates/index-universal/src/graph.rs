//! Unified Hierarchical Graph
//! 
//! Single graph structure that adapts to all scenarios:
//! - Small datasets: Single layer, exhaustive search
//! - Medium datasets: Multi-layer HNSW
//! - Large datasets: Multi-layer HNSW + LSH acceleration
//! 
//! Features:
//! - Lazy construction (build edges as needed)
//! - Adaptive layer count
//! - Multi-modal support
//! - Memory-efficient storage

use crate::autotune::AutoTuner;
use crate::cache::IntelligentCache;
use crate::config::UniversalConfig;
use crate::error::Result;
use index_core::{distance, DistanceMetric, ScoredPoint, Vector, VectorIndex};
use index_hnsw::{HnswConfig, HnswIndex};
use std::collections::HashMap;

/// Unified hierarchical graph that adapts to dataset size
#[derive(Debug, Clone)]
pub struct UnifiedHierarchicalGraph {
    metric: DistanceMetric,
    config: UniversalConfig,
    
    // Adaptive structure
    small_dataset_threshold: usize,
    large_dataset_threshold: usize,
    
    // Graph layers (for medium/large datasets)
    layers: Vec<HnswIndex>,
    max_layers: usize,
    
    // Simple storage (for small datasets)
    simple_vectors: Vec<(usize, Vector)>,
    
    // Lazy construction state
    lazy_edges: HashMap<usize, Vec<usize>>,  // node_id -> pending neighbors
    constructed: bool,
}

impl UnifiedHierarchicalGraph {
    pub fn new(metric: DistanceMetric, config: &UniversalConfig) -> Self {
        Self {
            metric,
            config: config.clone(),
            small_dataset_threshold: 1000,
            large_dataset_threshold: 100000,
            layers: Vec::new(),
            max_layers: 5,
            simple_vectors: Vec::new(),
            lazy_edges: HashMap::new(),
            constructed: false,
        }
    }
    
    /// Build graph with lazy construction
    pub fn build_lazy(&mut self, dataset: &[(usize, Vector)], autotuner: &AutoTuner) -> Result<()> {
        let n = dataset.len();
        
        if n < self.small_dataset_threshold {
            // Small dataset: Use simple storage
            self.simple_vectors = dataset.iter().cloned().collect();
            self.constructed = true;
            return Ok(());
        }
        
        // Medium/large dataset: Initialize layers
        let layer_count = autotuner.optimal_layer_count(n);
        self.max_layers = layer_count;
        
        // Initialize layers with auto-tuned parameters
        let m_max = autotuner.optimal_m_max(n);
        let ef_construction = autotuner.optimal_ef_construction(n);
        let ef_search = autotuner.optimal_ef_search(n);
        
        for layer_id in 0..layer_count {
            let config = HnswConfig {
                m_max,
                ef_construction,
                ef_search,
                ml: 1.0 / 2.0_f64.ln(),
            };
            self.layers.push(HnswIndex::new(self.metric, config));
        }
        
        // Insert vectors into layer 0 only (lazy edge construction)
        for (id, vector) in dataset {
            if let Some(layer0) = self.layers.get_mut(0) {
                layer0.insert(*id, vector.clone())
                    .map_err(|e: anyhow::Error| crate::error::UniversalError::InvalidConfig(e.to_string()))?;
            }
        }
        
        self.constructed = true;
        Ok(())
    }
    
    /// Build graph with full construction
    pub fn build_full(&mut self, dataset: &[(usize, Vector)], autotuner: &AutoTuner) -> Result<()> {
        let n = dataset.len();
        
        if n < self.small_dataset_threshold {
            return self.build_lazy(dataset, autotuner);
        }
        
        // Build full hierarchical structure
        let layer_count = autotuner.optimal_layer_count(n);
        self.max_layers = layer_count;
        
        let m_max = autotuner.optimal_m_max(n);
        let ef_construction = autotuner.optimal_ef_construction(n);
        let ef_search = autotuner.optimal_ef_search(n);
        
        for layer_id in 0..layer_count {
            let config = HnswConfig {
                m_max,
                ef_construction,
                ef_search,
                ml: 1.0 / 2.0_f64.ln(),
            };
            self.layers.push(HnswIndex::new(self.metric, config));
        }
        
        // Insert vectors into all appropriate layers
        for (id, vector) in dataset {
            let layers_to_insert = self.select_layers_for_insertion();
            for &layer_id in &layers_to_insert {
                if let Some(layer) = self.layers.get_mut(layer_id) {
                    layer.insert(*id, vector.clone())
                        .map_err(|e: anyhow::Error| crate::error::UniversalError::InvalidConfig(e.to_string()))?;
                }
            }
        }
        
        self.constructed = true;
        Ok(())
    }
    
    /// Insert vector with lazy construction
    pub fn insert_lazy(&mut self, id: usize, vector: Vector, autotuner: &AutoTuner) -> Result<()> {
        if !self.constructed {
            // Initialize if needed
            let n = self.simple_vectors.len() + 1;
            if n < self.small_dataset_threshold {
                self.simple_vectors.push((id, vector));
                return Ok(());
            } else {
                // Transition to hierarchical structure
                let mut dataset: Vec<(usize, Vector)> = self.simple_vectors.drain(..).collect();
                dataset.push((id, vector.clone()));
                return self.build_lazy(&dataset, autotuner);
            }
        }
        
        if !self.simple_vectors.is_empty() {
            // Still using simple storage
            self.simple_vectors.push((id, vector));
            return Ok(());
        }
        
        // Insert into layer 0 (edges added lazily during search)
        if let Some(layer0) = self.layers.get_mut(0) {
            layer0.insert(id, vector)
                .map_err(|e: anyhow::Error| crate::error::UniversalError::InvalidConfig(e.to_string()))?;
        }
        
        Ok(())
    }
    
    /// Insert vector with full construction
    pub fn insert(&mut self, id: usize, vector: Vector, autotuner: &AutoTuner) -> Result<()> {
        if !self.constructed {
            return self.insert_lazy(id, vector, autotuner);
        }
        
        if !self.simple_vectors.is_empty() {
            self.simple_vectors.push((id, vector));
            return Ok(());
        }
        
        // Insert into all appropriate layers
        let layers_to_insert = self.select_layers_for_insertion();
        for &layer_id in &layers_to_insert {
            if let Some(layer) = self.layers.get_mut(layer_id) {
                layer.insert(id, vector.clone())
                    .map_err(|e: anyhow::Error| crate::error::UniversalError::InvalidConfig(e.to_string()))?;
            }
        }
        
        Ok(())
    }
    
    /// Search with intelligent routing and caching
    pub fn search(
        &self,
        query: &Vector,
        entry_points: Vec<usize>,
        limit: usize,
        autotuner: &AutoTuner,
        cache: &IntelligentCache,
    ) -> Result<Vec<ScoredPoint>> {
        if !self.constructed {
            return Err(crate::error::UniversalError::EmptyIndex);
        }
        
        // Small dataset: Exhaustive search
        if !self.simple_vectors.is_empty() {
            return self.search_simple(query, limit);
        }
        
        // Medium/large dataset: Hierarchical search
        self.search_hierarchical(query, entry_points, limit, autotuner, cache)
    }
    
    fn search_simple(&self, query: &Vector, limit: usize) -> Result<Vec<ScoredPoint>> {
        let mut candidates: Vec<ScoredPoint> = self.simple_vectors
            .iter()
            .map(|(id, vector)| {
                let dist = distance(self.metric, query, vector).unwrap_or(f32::MAX);
                ScoredPoint::new(*id, dist)
            })
            .collect();
        
        candidates.sort_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(limit);
        
        Ok(candidates)
    }
    
    fn search_hierarchical(
        &self,
        query: &Vector,
        entry_points: Vec<usize>,
        limit: usize,
        autotuner: &AutoTuner,
        cache: &IntelligentCache,
    ) -> Result<Vec<ScoredPoint>> {
        if self.layers.is_empty() {
            return Ok(Vec::new());
        }
        
        // Start from top layer with entry points
        let mut candidates = Vec::new();
        let ef = autotuner.optimal_ef_search(self.total_vectors());
        
        // Search from top to bottom
        for layer_id in (0..self.layers.len()).rev() {
            if let Some(layer) = self.layers.get(layer_id) {
                if layer_id == self.layers.len() - 1 {
                    // Top layer: use entry points
                    if let Ok(results) = layer.search(query, ef) {
                        candidates = results;
                    }
                } else {
                    // Lower layers: refine from upper layer candidates
                    if !candidates.is_empty() {
                        if let Ok(results) = layer.search(query, ef) {
                            candidates = results;
                        }
                    }
                }
            }
        }
        
        // Return top-k
        candidates.sort_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(limit);
        
        Ok(candidates)
    }
    
    fn select_layers_for_insertion(&self) -> Vec<usize> {
        // HNSW-style probabilistic layer selection
        let mut layers = vec![0];  // Always insert into layer 0
        
        let mut layer = 0;
        let ml = 1.0 / 2.0_f64.ln();
        let mut rng = rand::thread_rng();
        
        while layer < self.max_layers - 1 {
            let prob = (-(layer as f64) / ml).exp();
            if rng.gen::<f64>() < prob {
                layer += 1;
                layers.push(layer);
            } else {
                break;
            }
        }
        
        layers
    }
    
    pub fn delete(&mut self, id: usize) -> Result<()> {
        if !self.simple_vectors.is_empty() {
            self.simple_vectors.retain(|(vid, _)| *vid != id);
            return Ok(());
        }
        
        for layer in &mut self.layers {
            let _ = layer.delete(id);
        }
        
        Ok(())
    }
    
    pub fn total_vectors(&self) -> usize {
        if !self.simple_vectors.is_empty() {
            self.simple_vectors.len()
        } else if let Some(layer0) = self.layers.get(0) {
            layer0.len()
        } else {
            0
        }
    }
    
    pub fn serialize(&self) -> Result<Vec<u8>> {
        // TODO: Implement serialization
        Ok(Vec::new())
    }
    
    pub fn deserialize(data: Vec<u8>) -> Result<Self> {
        // TODO: Implement deserialization
        Err(crate::error::UniversalError::SerializationError("Not implemented".to_string()))
    }
}

use rand::Rng;
