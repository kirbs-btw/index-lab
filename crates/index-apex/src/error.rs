use thiserror::Error;

#[derive(Debug, Error)]
pub enum ApexError {
    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("index is empty, cannot search")]
    EmptyIndex,
    
    #[error("modality {modality} not supported")]
    UnsupportedModality { modality: String },
    
    #[error("invalid cluster ID: {0}")]
    InvalidClusterId(usize),
    
    #[error("distribution shift detected but adaptation failed: {0}")]
    AdaptationFailed(String),
    
    #[error("energy budget exhausted")]
    EnergyBudgetExhausted,
    
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    
    #[error("router training failed: {0}")]
    RouterTrainingError(String),
    
    #[error("bucket error: {0}")]
    BucketError(String),
    
    #[error("LSH error: {0}")]
    LshError(String),
    
    #[error("serialization error: {0}")]
    SerializationError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("anyhow error: {0}")]
    AnyhowError(#[from] anyhow::Error),
    
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

pub type Result<T> = std::result::Result<T, ApexError>;
