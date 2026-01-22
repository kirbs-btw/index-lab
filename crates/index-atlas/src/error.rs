use thiserror::Error;

#[derive(Error, Debug)]
pub enum AtlasError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Empty index: cannot search or train router")]
    EmptyIndex,

    #[error("Invalid cluster ID: {0}")]
    InvalidClusterId(usize),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Router training failed: {0}")]
    RouterTrainingError(String),

    #[error("Bucket error: {0}")]
    BucketError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("HNSW error: {0}")]
    HnswError(String),

    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

pub type Result<T> = std::result::Result<T, AtlasError>;
