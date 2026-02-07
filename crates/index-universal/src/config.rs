use serde::{Deserialize, Serialize};

/// Configuration for UNIVERSAL index
/// 
/// All parameters are auto-tuned by default.
/// Manual configuration is optional and only for advanced use cases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalConfig {
    /// Enable auto-tuning (default: true)
    pub auto_tune: bool,
    
    /// Target recall (default: 0.95)
    pub target_recall: f32,
    
    /// Memory budget in bytes (None = unlimited)
    pub memory_budget: Option<usize>,
    
    /// Enable caching (default: true)
    pub enable_caching: bool,
    
    /// Cache size factor (default: sqrt(N))
    pub cache_size_factor: f64,
    
    /// Enable compression (default: true for large datasets)
    pub enable_compression: bool,
    
    /// Compression threshold (compress vectors if dataset > threshold)
    pub compression_threshold: usize,
    
    /// Random seed for reproducibility
    pub seed: u64,
    
    /// Enable lazy construction (default: true)
    pub lazy_construction: bool,
    
    /// Background optimization interval (None = disabled)
    pub optimization_interval: Option<u64>,
}

impl Default for UniversalConfig {
    fn default() -> Self {
        Self {
            auto_tune: true,
            target_recall: 0.95,
            memory_budget: None,
            enable_caching: true,
            cache_size_factor: 1.0,  // sqrt(N) by default
            enable_compression: true,
            compression_threshold: 10000,
            seed: 42,
            lazy_construction: true,
            optimization_interval: None,
        }
    }
}

impl UniversalConfig {
    /// Create configuration optimized for speed
    pub fn fast() -> Self {
        Self {
            target_recall: 0.90,
            cache_size_factor: 2.0,
            ..Default::default()
        }
    }
    
    /// Create configuration optimized for accuracy
    pub fn accurate() -> Self {
        Self {
            target_recall: 0.99,
            cache_size_factor: 0.5,
            ..Default::default()
        }
    }
    
    /// Create configuration optimized for memory
    pub fn memory_efficient() -> Self {
        Self {
            enable_compression: true,
            compression_threshold: 1000,
            cache_size_factor: 0.5,
            ..Default::default()
        }
    }
    
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.target_recall < 0.0 || self.target_recall > 1.0 {
            return Err(crate::error::UniversalError::InvalidConfig(
                "target_recall must be in [0, 1]".to_string(),
            ));
        }
        
        if self.cache_size_factor < 0.0 {
            return Err(crate::error::UniversalError::InvalidConfig(
                "cache_size_factor must be >= 0".to_string(),
            ));
        }
        
        Ok(())
    }
}
