use serde::{Deserialize, Serialize};

/// Configuration for the CONVERGENCE index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceConfig {
    // Clustering
    /// Number of clusters (default: sqrt(N), will be set during build)
    pub num_clusters: Option<usize>,
    
    // Router
    /// Hidden dimension for the learned router MLP
    pub router_hidden_dim: usize,
    /// Learning rate for router training
    pub router_learning_rate: f32,
    /// Enable continuous online learning (updates router during queries)
    pub enable_online_learning: bool,
    /// Mini-batch size for online learning
    pub online_learning_batch_size: usize,
    /// Confidence threshold for ensemble routing
    pub ensemble_confidence_threshold: f32,
    
    // Adaptive LSH
    /// Initial number of hyperplanes for LSH hashing
    pub lsh_initial_hyperplanes: usize,
    /// Enable adaptive LSH parameter tuning
    pub enable_adaptive_lsh: bool,
    /// Minimum hyperplanes (adaptive)
    pub lsh_min_hyperplanes: usize,
    /// Maximum hyperplanes (adaptive)
    pub lsh_max_hyperplanes: usize,
    /// Initial probe count
    pub lsh_initial_probes: usize,
    /// Minimum probes (adaptive)
    pub lsh_min_probes: usize,
    /// Maximum probes (adaptive)
    pub lsh_max_probes: usize,
    
    // Hierarchical Graph
    /// Number of edges per node in graphs
    pub graph_m_max: usize,
    /// ef_construction for graph building
    pub graph_ef_construction: usize,
    /// ef_search for graph search
    pub graph_ef_search: usize,
    /// Maximum layer level (true hierarchical)
    pub graph_max_layers: usize,
    /// Layer selection probability (for hierarchical construction)
    pub graph_ml: f64,
    
    // Temporal
    /// Enable temporal decay everywhere
    pub enable_temporal_decay: bool,
    /// Temporal decay rate
    pub temporal_decay_rate: f32,
    /// Half-life in seconds for temporal decay
    pub temporal_halflife_seconds: f64,
    
    // Edge Pruning
    /// Enable smart edge pruning
    pub enable_edge_pruning: bool,
    /// Distance threshold for edge creation
    pub edge_distance_threshold: f32,
    /// Maximum degree per node
    pub max_node_degree: usize,
    /// Enable importance-based pruning
    pub enable_importance_pruning: bool,
    
    // Adaptive
    /// Enable adaptive parameter tuning
    pub enable_adaptive_tuning: bool,
    /// Minimum ef value
    pub min_ef: usize,
    /// Maximum ef value
    pub max_ef: usize,
    
    // Energy
    /// Enable energy optimization
    pub enable_energy_optimization: bool,
    /// Energy budget per query (None = unlimited)
    pub energy_budget_per_query: Option<f32>,
    
    // Robustness
    /// Window size for distribution shift detection
    pub shift_detection_window: usize,
    /// Threshold for shift detection
    pub shift_threshold: f32,
    
    // Hybrid
    /// Initial weight for dense component (will be learned)
    pub initial_dense_weight: f32,
    /// Enable sparse indexing
    pub enable_sparse: bool,
    /// Learn fusion weights adaptively
    pub learn_fusion_weights: bool,
    
    // Empty Bucket Handling
    /// Enable automatic bucket merging
    pub enable_bucket_merging: bool,
    /// Minimum bucket size before merging
    pub min_bucket_size: usize,
    
    // Multi-Strategy Search
    /// Enable multi-strategy search
    pub enable_multi_strategy: bool,
    /// Strategy selection method
    pub strategy_selection: StrategySelection,
    
    // Build
    /// Random seed for reproducibility
    pub seed: u64,
    /// Maximum K-Means iterations during initial clustering
    pub max_kmeans_iters: usize,
    /// Subsample size for router training (None = use all)
    pub router_training_subsample: Option<usize>,
    
    // Deterministic
    /// Use deterministic RNG
    pub deterministic: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrategySelection {
    /// Always use best strategy
    Best,
    /// Use ensemble of strategies
    Ensemble,
    /// Adaptive selection based on query
    Adaptive,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            // Clustering
            num_clusters: None,
            
            // Router
            router_hidden_dim: 128,
            router_learning_rate: 0.001,
            enable_online_learning: true,  // NEW: Enabled by default
            online_learning_batch_size: 32,
            ensemble_confidence_threshold: 0.7,
            
            // Adaptive LSH
            lsh_initial_hyperplanes: 3,
            enable_adaptive_lsh: true,  // NEW: Adaptive enabled
            lsh_min_hyperplanes: 2,
            lsh_max_hyperplanes: 8,
            lsh_initial_probes: 8,
            lsh_min_probes: 4,
            lsh_max_probes: 16,
            
            // Hierarchical Graph
            graph_m_max: 16,
            graph_ef_construction: 200,
            graph_ef_search: 50,
            graph_max_layers: 5,  // NEW: True hierarchical
            graph_ml: 1.0 / 2.0_f64.ln(),  // HNSW layer selection probability
            
            // Temporal
            enable_temporal_decay: true,
            temporal_decay_rate: 0.1,
            temporal_halflife_seconds: 86400.0,
            
            // Edge Pruning
            enable_edge_pruning: true,  // NEW: Enabled
            edge_distance_threshold: 0.5,
            max_node_degree: 32,
            enable_importance_pruning: true,
            
            // Adaptive
            enable_adaptive_tuning: true,
            min_ef: 10,
            max_ef: 200,
            
            // Energy
            enable_energy_optimization: true,
            energy_budget_per_query: None,
            
            // Robustness
            shift_detection_window: 1000,
            shift_threshold: 0.1,
            
            // Hybrid
            initial_dense_weight: 0.6,
            enable_sparse: true,
            learn_fusion_weights: true,  // NEW: Learn weights
            
            // Empty Bucket Handling
            enable_bucket_merging: true,  // NEW: Enabled
            min_bucket_size: 10,
            
            // Multi-Strategy Search
            enable_multi_strategy: true,  // NEW: Enabled
            strategy_selection: StrategySelection::Adaptive,
            
            // Build
            seed: 42,
            max_kmeans_iters: 20,
            router_training_subsample: None,
            
            // Deterministic
            deterministic: true,
        }
    }
}

impl ConvergenceConfig {
    /// Create a configuration optimized for high recall
    pub fn high_recall() -> Self {
        Self {
            lsh_initial_probes: 12,
            lsh_max_probes: 20,
            graph_m_max: 24,
            graph_ef_construction: 200,
            graph_ef_search: 100,
            max_ef: 300,
            edge_distance_threshold: 0.7,  // More permissive
            ..Default::default()
        }
    }

    /// Create a configuration optimized for speed
    pub fn high_speed() -> Self {
        Self {
            lsh_initial_probes: 4,
            lsh_max_probes: 8,
            graph_m_max: 12,
            graph_ef_construction: 50,
            graph_ef_search: 30,
            min_ef: 5,
            max_ef: 100,
            edge_distance_threshold: 0.3,  // More restrictive
            max_node_degree: 16,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for memory efficiency
    pub fn memory_efficient() -> Self {
        Self {
            graph_max_layers: 3,
            max_node_degree: 16,
            edge_distance_threshold: 0.4,
            enable_edge_pruning: true,
            enable_importance_pruning: true,
            ..Default::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.router_hidden_dim == 0 {
            return Err(crate::error::ConvergenceError::InvalidConfig(
                "router_hidden_dim must be > 0".to_string(),
            ));
        }

        if self.router_learning_rate <= 0.0 || self.router_learning_rate > 1.0 {
            return Err(crate::error::ConvergenceError::InvalidConfig(
                "router_learning_rate must be in (0, 1]".to_string(),
            ));
        }

        if self.lsh_min_hyperplanes > self.lsh_max_hyperplanes {
            return Err(crate::error::ConvergenceError::InvalidConfig(
                "lsh_min_hyperplanes must be <= lsh_max_hyperplanes".to_string(),
            ));
        }

        if self.lsh_min_probes > self.lsh_max_probes {
            return Err(crate::error::ConvergenceError::InvalidConfig(
                "lsh_min_probes must be <= lsh_max_probes".to_string(),
            ));
        }

        if self.graph_max_layers == 0 {
            return Err(crate::error::ConvergenceError::InvalidConfig(
                "graph_max_layers must be > 0".to_string(),
            ));
        }

        if self.edge_distance_threshold < 0.0 || self.edge_distance_threshold > 1.0 {
            return Err(crate::error::ConvergenceError::InvalidConfig(
                "edge_distance_threshold must be in [0, 1]".to_string(),
            ));
        }

        if self.min_ef > self.max_ef {
            return Err(crate::error::ConvergenceError::InvalidConfig(
                "min_ef must be <= max_ef".to_string(),
            ));
        }

        if self.initial_dense_weight < 0.0 || self.initial_dense_weight > 1.0 {
            return Err(crate::error::ConvergenceError::InvalidConfig(
                "initial_dense_weight must be in [0, 1]".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = ConvergenceConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_high_recall_config_is_valid() {
        let config = ConvergenceConfig::high_recall();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_high_speed_config_is_valid() {
        let config = ConvergenceConfig::high_speed();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_memory_efficient_config_is_valid() {
        let config = ConvergenceConfig::memory_efficient();
        assert!(config.validate().is_ok());
    }
}
