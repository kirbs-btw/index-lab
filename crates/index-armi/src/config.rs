use serde::{Deserialize, Serialize};

/// ARMI index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmiConfig {
    /// Base HNSW configuration for graph structure
    pub base_ef_construction: usize,
    pub base_ef_search: usize,
    pub m_max: usize,
    
    /// Distribution shift detection
    pub shift_detection_window: usize,
    pub shift_threshold: f32,
    
    /// Adaptive tuning
    pub enable_adaptive_tuning: bool,
    pub min_ef: usize,
    pub max_ef: usize,
    
    /// Energy optimization
    pub enable_energy_optimization: bool,
    pub energy_budget_per_query: Option<f32>,
    
    /// Deterministic mode
    pub deterministic: bool,
    pub seed: u64,
}

impl Default for ArmiConfig {
    fn default() -> Self {
        Self {
            base_ef_construction: 200,
            base_ef_search: 50,
            m_max: 16,
            shift_detection_window: 1000,
            shift_threshold: 0.1,
            enable_adaptive_tuning: true,
            min_ef: 10,
            max_ef: 200,
            enable_energy_optimization: true,
            energy_budget_per_query: None,
            deterministic: true,
            seed: 42,
        }
    }
}
