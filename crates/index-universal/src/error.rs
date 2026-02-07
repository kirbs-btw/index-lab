use thiserror::Error;

#[derive(Debug, Error)]
pub enum UniversalError {
    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("index is empty, cannot search")]
    EmptyIndex,
    
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    
    #[error("serialization error: {0}")]
    SerializationError(String),
    
    #[error("cache error: {0}")]
    CacheError(String),
}

pub type Result<T> = std::result::Result<T, UniversalError>;
