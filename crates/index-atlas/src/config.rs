use serde::{Deserialize, Serialize};

/// Configuration for the ATLAS index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtlasConfig {
    // Router configuration
    /// Number of clusters (default: sqrt(N), will be set during build)
    pub num_clusters: Option<usize>,

    /// Hidden dimension for the learned router MLP
    pub router_hidden_dim: usize,

    /// Learning rate for router training
    pub router_learning_rate: f32,

    /// Confidence threshold for using learned router vs graph fallback
    pub confidence_threshold: f32,

    /// Enable online learning (update router during queries)
    pub enable_online_learning: bool,

    // Search configuration
    /// Number of buckets to probe during search
    pub n_probes: usize,

    /// Use learned routing (if false, always use graph)
    pub use_learned_routing: bool,

    // Bucket configuration (mini-HNSW)
    /// Number of edges per node in mini-HNSW
    pub mini_hnsw_m: usize,

    /// ef_construction for mini-HNSW
    pub mini_hnsw_ef_construction: usize,

    /// ef_search for mini-HNSW
    pub mini_hnsw_ef_search: usize,

    // Hybrid configuration
    /// Weight for dense component (0.0 = all sparse, 1.0 = all dense)
    pub dense_weight: f32,

    /// Enable sparse indexing
    pub enable_sparse: bool,

    // Build configuration
    /// Random seed for reproducibility
    pub seed: u64,

    /// Maximum K-Means iterations during initial clustering
    pub max_kmeans_iters: usize,

    /// Subsample size for router training (None = use all)
    pub router_training_subsample: Option<usize>,
}

impl Default for AtlasConfig {
    fn default() -> Self {
        Self {
            // Router
            num_clusters: None, // Will be set to sqrt(N)
            router_hidden_dim: 128,
            router_learning_rate: 0.001,
            confidence_threshold: 0.7,
            enable_online_learning: false, // Disabled by default for stability

            // Search
            n_probes: 3,
            use_learned_routing: true,

            // Bucket
            mini_hnsw_m: 16,
            mini_hnsw_ef_construction: 100,
            mini_hnsw_ef_search: 50,

            // Hybrid
            dense_weight: 0.6,
            enable_sparse: true,

            // Build
            seed: 42,
            max_kmeans_iters: 20,
            router_training_subsample: None,
        }
    }
}

impl AtlasConfig {
    /// Create a configuration optimized for high recall
    pub fn high_recall() -> Self {
        Self {
            n_probes: 5,
            mini_hnsw_m: 24,
            mini_hnsw_ef_construction: 200,
            mini_hnsw_ef_search: 100,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for speed
    pub fn high_speed() -> Self {
        Self {
            n_probes: 2,
            mini_hnsw_m: 12,
            mini_hnsw_ef_construction: 50,
            mini_hnsw_ef_search: 30,
            ..Default::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.router_hidden_dim == 0 {
            return Err(crate::error::AtlasError::InvalidConfig(
                "router_hidden_dim must be > 0".to_string(),
            ));
        }

        if self.router_learning_rate <= 0.0 || self.router_learning_rate > 1.0 {
            return Err(crate::error::AtlasError::InvalidConfig(
                "router_learning_rate must be in (0, 1]".to_string(),
            ));
        }

        if self.confidence_threshold < 0.0 || self.confidence_threshold > 1.0 {
            return Err(crate::error::AtlasError::InvalidConfig(
                "confidence_threshold must be in [0, 1]".to_string(),
            ));
        }

        if self.n_probes == 0 {
            return Err(crate::error::AtlasError::InvalidConfig(
                "n_probes must be > 0".to_string(),
            ));
        }

        if self.mini_hnsw_m == 0 {
            return Err(crate::error::AtlasError::InvalidConfig(
                "mini_hnsw_m must be > 0".to_string(),
            ));
        }

        if self.dense_weight < 0.0 || self.dense_weight > 1.0 {
            return Err(crate::error::AtlasError::InvalidConfig(
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
        let config = AtlasConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_high_recall_config_is_valid() {
        let config = AtlasConfig::high_recall();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_high_speed_config_is_valid() {
        let config = AtlasConfig::high_speed();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_learning_rate() {
        let config = AtlasConfig {
            router_learning_rate: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_dense_weight() {
        let config = AtlasConfig {
            dense_weight: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }
}
