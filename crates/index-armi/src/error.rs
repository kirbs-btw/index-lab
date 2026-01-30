use thiserror::Error;

#[derive(Debug, Error)]
pub enum ArmiError {
    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("index is empty, cannot search")]
    EmptyIndex,
    
    #[error("modality {modality} not supported")]
    UnsupportedModality { modality: String },
    
    #[error("distribution shift detected but adaptation failed: {0}")]
    AdaptationFailed(String),
    
    #[error("energy budget exhausted")]
    EnergyBudgetExhausted,
    
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, ArmiError>;
