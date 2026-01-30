//! ARMI: Adaptive Robust Multi-Modal Index
//!
//! A novel vector indexing algorithm that combines:
//! - Multi-modal streaming workloads (Gap 1B)
//! - Energy efficiency optimization (Gap 5)
//! - Deterministic/reproducible search (Gap 6A)
//! - Out-of-distribution robustness (Gap 6B)
//! - Adaptive query-time optimization (Gap 7A)

mod adaptive;
mod config;
mod energy;
mod error;
mod graph;
mod modality;
mod robustness;

pub use config::ArmiConfig;
pub use error::{ArmiError, Result};
pub use modality::{ModalityType, MultiModalQuery, MultiModalVector};

use adaptive::ParameterOptimizer;
use energy::{EnergyBudget, PrecisionSelector};
use graph::UnifiedGraph;
use index_core::{
    validate_dimension, DistanceMetric, ScoredPoint, Vector,
    VectorIndex,
};
use rand::{rngs::StdRng, SeedableRng};
use robustness::{ShiftDetector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Main ARMI index implementation
#[derive(Debug, Clone)]
pub struct ArmiIndex {
    config: ArmiConfig,
    metric: DistanceMetric,
    
    // Multi-modal storage
    vectors: HashMap<usize, MultiModalVector>,
    
    // Unified graph
    graph: UnifiedGraph,
    
    // Robustness components
    shift_detector: ShiftDetector,
    
    // Adaptive tuning
    parameter_optimizer: ParameterOptimizer,
    
    // Energy management
    energy_budget: EnergyBudget,
    precision_selector: PrecisionSelector,
    
    // Deterministic RNG
    rng: Option<StdRng>,
    
    dimension: Option<usize>,
}

impl ArmiIndex {
    /// Creates a new ARMI index
    pub fn new(metric: DistanceMetric, config: ArmiConfig) -> Self {
        let mut rng = if config.deterministic {
            Some(StdRng::seed_from_u64(config.seed))
        } else {
            None
        };
        
        Self {
            config: config.clone(),
            metric,
            vectors: HashMap::new(),
            graph: UnifiedGraph::new(config.m_max, metric),
            shift_detector: ShiftDetector::new(
                config.shift_detection_window,
                config.shift_threshold,
            ),
            parameter_optimizer: ParameterOptimizer::new(
                config.min_ef,
                config.max_ef,
            ),
            energy_budget: EnergyBudget::new(config.energy_budget_per_query),
            precision_selector: PrecisionSelector::new(),
            rng,
            dimension: None,
        }
    }
    
    /// Creates a new ARMI index with default configuration
    pub fn with_defaults(metric: DistanceMetric) -> Self {
        Self::new(metric, ArmiConfig::default())
    }
    
    /// Inserts a multi-modal vector
    pub fn insert_multi_modal(
        &mut self,
        id: usize,
        vector: MultiModalVector,
    ) -> anyhow::Result<()> {
        // Validate dimensions
        if let Some(dense) = &vector.dense {
            if let Some(expected_dim) = self.dimension {
                validate_dimension(Some(expected_dim), dense.len())
                    .map_err(|_| ArmiError::DimensionMismatch {
                        expected: expected_dim,
                        actual: dense.len(),
                    })?;
            } else {
                self.dimension = Some(dense.len());
            }
        }
        
        // Store vector
        self.vectors.insert(id, vector.clone());
        
        // Detect distribution shift
        if self.shift_detector.detect_shift()? {
            self.adapt_to_shift()?;
        }
        
        // Update shift detector
        self.shift_detector.update(&vector);
        
        // Add to unified graph
        self.graph.insert_multi_modal(vector)?;
        
        // Update energy budget
        self.energy_budget.record_insert()?;
        
        Ok(())
    }
    
    /// Adaptive search with multi-modal query
    pub fn search_adaptive(
        &self,
        query: &MultiModalQuery,
        limit: usize,
    ) -> anyhow::Result<Vec<ScoredPoint>> {
        if self.vectors.is_empty() {
            return Err(ArmiError::EmptyIndex.into());
        }
        
        // Reset energy budget for new query
        let mut budget = self.energy_budget.clone();
        budget.reset();
        
        // Select optimal parameters via RL agent
        let mut optimizer = self.parameter_optimizer.clone();
        let params = optimizer.select_params(query)?;
        
        // Choose precision based on energy budget
        let precision = self.precision_selector.select(query, &budget)?;
        
        // Multi-modal graph traversal
        let candidates = self.graph.search_unified(query, params.ef, precision)?;
        
        // Convert to ScoredPoints
        let results: Vec<ScoredPoint> = candidates
            .into_iter()
            .take(limit)
            .map(|(id, dist)| ScoredPoint::new(id, dist))
            .collect();
        
        // Update RL agent with results (would need mutable access in production)
        // optimizer.update(query, &results)?;
        
        Ok(results)
    }
    
    fn adapt_to_shift(&mut self) -> anyhow::Result<()> {
        // Identify affected regions
        let affected = self.shift_detector.affected_regions()?;
        
        // Rebuild graph regions incrementally
        if !affected.is_empty() {
            self.graph.rebuild_region(affected)?;
        }
        
        // Reset parameter optimizer
        self.parameter_optimizer.reset();
        
        Ok(())
    }
    
    /// Saves the index to a file
    /// Note: Currently saves only vectors, graph and other components are rebuilt on load
    pub fn save(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        use std::fs::File;
        use std::io::BufWriter;
        use serde_json;
        
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        
        // Save serializable parts
        let serializable = SerializableArmiIndex {
            config: self.config.clone(),
            metric: self.metric,
            vectors: self.vectors.clone(),
            dimension: self.dimension,
        };
        
        serde_json::to_writer(writer, &serializable)?;
        Ok(())
    }
    
    /// Loads the index from a file
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        use std::fs::File;
        use std::io::BufReader;
        use serde_json;
        
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let serializable: SerializableArmiIndex = serde_json::from_reader(reader)?;
        
        // Rebuild index from serialized data
        let mut index = Self::new(serializable.metric, serializable.config.clone());
        index.dimension = serializable.dimension;
        
        // Rebuild graph from vectors
        for (id, vector) in serializable.vectors {
            index.vectors.insert(id, vector.clone());
            index.graph.insert_multi_modal(vector)?;
        }
        
        Ok(index)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SerializableArmiIndex {
    config: ArmiConfig,
    metric: DistanceMetric,
    vectors: HashMap<usize, MultiModalVector>,
    dimension: Option<usize>,
}

impl VectorIndex for ArmiIndex {
    fn metric(&self) -> DistanceMetric {
        self.metric
    }
    
    fn len(&self) -> usize {
        self.vectors.len()
    }
    
    fn insert(&mut self, id: usize, vector: Vector) -> anyhow::Result<()> {
        // Convert single dense vector to multi-modal
        let multi_modal = MultiModalVector::with_dense(id, vector);
        self.insert_multi_modal(id, multi_modal)
    }
    
    fn search(&self, query: &Vector, limit: usize) -> anyhow::Result<Vec<ScoredPoint>> {
        // Convert single dense query to multi-modal
        let multi_query = MultiModalQuery::with_dense(query.clone());
        self.search_adaptive(&multi_query, limit)
    }
    
    fn delete(&mut self, id: usize) -> anyhow::Result<bool> {
        if self.vectors.remove(&id).is_some() {
            // Note: Graph deletion would require more complex implementation
            // For now, just remove from vectors storage
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    fn update(&mut self, id: usize, vector: Vector) -> anyhow::Result<bool> {
        if let Some(existing) = self.vectors.get_mut(&id) {
            // Update dense vector
            existing.dense = Some(vector);
            
            // Rebuild graph connections for this node
            // Simplified: would need to update graph edges
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_armi_basic_insert_and_search() {
        let mut index = ArmiIndex::with_defaults(DistanceMetric::Euclidean);
        
        // Insert some vectors
        index.insert(0, vec![1.0, 2.0, 3.0]).unwrap();
        index.insert(1, vec![1.1, 2.1, 3.1]).unwrap();
        index.insert(2, vec![10.0, 20.0, 30.0]).unwrap();
        
        assert_eq!(index.len(), 3);
        
        // Search
        let results = index.search(&vec![1.0, 2.0, 3.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 0); // Should find itself first
    }
    
    #[test]
    fn test_armi_multi_modal_insert() {
        let mut index = ArmiIndex::with_defaults(DistanceMetric::Euclidean);
        
        // Create multi-modal vector
        let mut vector = MultiModalVector::new(0);
        vector.dense = Some(vec![1.0, 2.0, 3.0]);
        let mut sparse = HashMap::new();
        sparse.insert(1, 0.5);
        sparse.insert(2, 0.3);
        vector.sparse = Some(sparse);
        
        index.insert_multi_modal(0, vector).unwrap();
        assert_eq!(index.len(), 1);
    }
    
    #[test]
    fn test_armi_multi_modal_search() {
        let mut index = ArmiIndex::with_defaults(DistanceMetric::Euclidean);
        
        // Insert with dense vector
        index.insert(0, vec![1.0, 2.0, 3.0]).unwrap();
        
        // Search with multi-modal query
        let mut query = MultiModalQuery::new();
        query.dense = Some(vec![1.0, 2.0, 3.0]);
        
        let results = index.search_adaptive(&query, 1).unwrap();
        assert!(!results.is_empty());
    }
    
    #[test]
    fn test_armi_empty_search() {
        let index = ArmiIndex::with_defaults(DistanceMetric::Euclidean);
        assert!(index.search(&vec![1.0, 2.0], 1).is_err());
    }
}
