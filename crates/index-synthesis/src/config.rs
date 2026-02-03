use serde::{Deserialize, Serialize};

/// Configuration for the SYNTHESIS index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisConfig {
    // Clustering
    /// Number of clusters (default: sqrt(N), will be set during build)
    pub num_clusters: Option<usize>,
    
    // Router
    /// Hidden dimension for the learned router MLP
    pub router_hidden_dim: usize,
    /// Learning rate for router training
    pub router_learning_rate: f32,
    /// Confidence threshold for using learned router vs graph fallback
    pub confidence_threshold: f32,
    /// Enable online learning (update router during queries)
    pub enable_online_learning: bool,
    /// Use learned routing (if false, always use graph)
    pub use_learned_routing: bool,
    /// Number of buckets to probe during search
    pub n_probes: usize,
    
    // LSH - Actually used in all operations
    /// Number of hyperplanes for LSH hashing
    pub lsh_hyperplanes: usize,
    /// Number of buckets to probe
    pub lsh_probes: usize,
    
    // Graph
    /// Number of edges per node in graphs
    pub graph_m_max: usize,
    /// ef_construction for graph building
    pub graph_ef_construction: usize,
    /// ef_search for graph search
    pub graph_ef_search: usize,
    /// Maximum layer level
    pub graph_max_layers: usize,
    
    // Temporal
    /// Enable temporal decay in edge weights
    pub enable_temporal_decay: bool,
    /// Temporal decay rate (higher = faster decay)
    pub temporal_decay_rate: f32,
    /// Half-life in seconds for temporal decay
    pub temporal_halflife_seconds: f64,
    
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
    /// Weight for dense component (0.0 = all sparse, 1.0 = all dense)
    pub dense_weight: f32,
    /// Enable sparse indexing
    pub enable_sparse: bool,
    /// Learn fusion weights adaptively
    pub learn_fusion_weights: bool,
    
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

impl Default for SynthesisConfig {
    fn default() -> Self {
        Self {
            // Clustering
            num_clusters: None, // Will be set to sqrt(N)
            
            // Router
            router_hidden_dim: 128,
            router_learning_rate: 0.001,
            confidence_threshold: 0.7,
            enable_online_learning: false,
            use_learned_routing: true,
            n_probes: 3,
            
            // LSH
            lsh_hyperplanes: 3, // 2^3 = 8 buckets
            lsh_probes: 8,
            
            // Graph
            graph_m_max: 16,
            graph_ef_construction: 200,
            graph_ef_search: 50,
            graph_max_layers: 5,
            
            // Temporal
            enable_temporal_decay: true,
            temporal_decay_rate: 0.1,
            temporal_halflife_seconds: 86400.0, // 1 day
            
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
            dense_weight: 0.6,
            enable_sparse: true,
            learn_fusion_weights: true,
            
            // Build
            seed: 42,
            max_kmeans_iters: 20,
            router_training_subsample: None,
            
            // Deterministic
            deterministic: true,
        }
    }
}

impl SynthesisConfig {
    /// Create a configuration optimized for high recall
    pub fn high_recall() -> Self {
        Self {
            lsh_probes: 10,
            graph_m_max: 24,
            graph_ef_construction: 200,
            graph_ef_search: 100,
            max_ef: 300,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for speed
    pub fn high_speed() -> Self {
        Self {
            lsh_probes: 4,
            graph_m_max: 12,
            graph_ef_construction: 50,
            graph_ef_search: 30,
            min_ef: 5,
            max_ef: 100,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for energy efficiency
    pub fn energy_efficient() -> Self {
        Self {
            enable_energy_optimization: true,
            energy_budget_per_query: Some(100.0),
            graph_ef_search: 30,
            max_ef: 100,
            ..Default::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.router_hidden_dim == 0 {
            return Err(crate::error::SynthesisError::InvalidConfig(
                "router_hidden_dim must be > 0".to_string(),
            ));
        }

        if self.router_learning_rate <= 0.0 || self.router_learning_rate > 1.0 {
            return Err(crate::error::SynthesisError::InvalidConfig(
                "router_learning_rate must be in (0, 1]".to_string(),
            ));
        }

        if self.confidence_threshold < 0.0 || self.confidence_threshold > 1.0 {
            return Err(crate::error::SynthesisError::InvalidConfig(
                "confidence_threshold must be in [0, 1]".to_string(),
            ));
        }

        if self.lsh_hyperplanes == 0 {
            return Err(crate::error::SynthesisError::InvalidConfig(
                "lsh_hyperplanes must be > 0".to_string(),
            ));
        }

        if self.lsh_probes == 0 {
            return Err(crate::error::SynthesisError::InvalidConfig(
                "lsh_probes must be > 0".to_string(),
            ));
        }

        if self.graph_m_max == 0 {
            return Err(crate::error::SynthesisError::InvalidConfig(
                "graph_m_max must be > 0".to_string(),
            ));
        }

        if self.temporal_decay_rate < 0.0 || self.temporal_decay_rate > 1.0 {
            return Err(crate::error::SynthesisError::InvalidConfig(
                "temporal_decay_rate must be in [0, 1]".to_string(),
            ));
        }

        if self.min_ef > self.max_ef {
            return Err(crate::error::SynthesisError::InvalidConfig(
                "min_ef must be <= max_ef".to_string(),
            ));
        }

        if self.dense_weight < 0.0 || self.dense_weight > 1.0 {
            return Err(crate::error::SynthesisError::InvalidConfig(
                "dense_weight must be in [0, 1]".to_string(),
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
        let config = SynthesisConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_high_recall_config_is_valid() {
        let config = SynthesisConfig::high_recall();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_high_speed_config_is_valid() {
        let config = SynthesisConfig::high_speed();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_energy_efficient_config_is_valid() {
        let config = SynthesisConfig::energy_efficient();
        assert!(config.validate().is_ok());
    }
}
