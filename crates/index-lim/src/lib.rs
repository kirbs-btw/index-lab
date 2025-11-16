//! LIM (Locality Index Method) index implementation.
//!
//! TODO: Implement the LIM algorithm here.

use anyhow::{ensure, Result};
use index_core::{
    DistanceMetric, load_index, save_index, ScoredPoint,
    validate_dimension, Vector, VectorIndex,
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LimError {
    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("index is empty, cannot search")]
    EmptyIndex,
}

/// LIM index configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LimConfig {
    // TODO: Add configuration parameters for your algorithm
}

impl Default for LimConfig {
    fn default() -> Self {
        Self {
            // TODO: Set default values
        }
    }
}

/// LIM index implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimIndex {
    metric: DistanceMetric,
    dimension: Option<usize>,
    config: LimConfig,
    // TODO: Add your algorithm-specific fields here
    // Example: vectors: Vec<(usize, Vector)>,
}

impl LimIndex {
    /// Creates a new LIM index with the given metric and configuration
    pub fn new(metric: DistanceMetric, config: LimConfig) -> Self {
        Self {
            metric,
            dimension: None,
            config,
            // TODO: Initialize your algorithm-specific fields
        }
    }

    /// Creates a new LIM index with default configuration
    pub fn with_defaults(metric: DistanceMetric) -> Self {
        Self::new(metric, LimConfig::default())
    }

    fn validate_dimension(&self, vector: &[f32]) -> Result<()> {
        if let Some(expected) = self.dimension {
            validate_dimension(Some(expected), vector.len())
                .map_err(|_| LimError::DimensionMismatch {
                    expected,
                    actual: vector.len(),
                })?;
        }
        Ok(())
    }

    /// Returns the dimensionality tracked by the index
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Saves the index to a JSON file
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        save_index(self, path)
    }

    /// Loads an index from a JSON file
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        load_index(path)
    }
}

impl VectorIndex for LimIndex {
    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn len(&self) -> usize {
        // TODO: Return the actual number of vectors in the index
        0
    }

    fn insert(&mut self, _id: usize, vector: Vector) -> Result<()> {
        self.validate_dimension(&vector)?;
        if self.dimension.is_none() {
            self.dimension = Some(vector.len());
        }
        // TODO: Implement your insertion logic here
        Ok(())
    }

    fn search(&self, query: &Vector, limit: usize) -> Result<Vec<ScoredPoint>> {
        ensure!(limit > 0, "limit must be greater than zero to execute a search");
        ensure!(self.len() > 0, LimError::EmptyIndex);
        self.validate_dimension(query)?;

        // TODO: Implement your search logic here
        // For now, return empty results
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_index_has_zero_length() {
        let index = LimIndex::with_defaults(DistanceMetric::Euclidean);
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn insert_sets_dimension() {
        let mut index = LimIndex::with_defaults(DistanceMetric::Euclidean);
        index.insert(0, vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(index.dimension(), Some(3));
    }

    // TODO: Add more tests as you implement the algorithm
}

