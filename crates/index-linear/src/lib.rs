//! Straight-forward baseline index that performs a linear scan over all vectors.

use anyhow::{ensure, Result};
use index_core::{
    distance, load_index, save_index, validate_dimension, DistanceMetric, ScoredPoint, Vector,
    VectorIndex,
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LinearIndexError {
    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearIndex {
    metric: DistanceMetric,
    dimension: Option<usize>,
    entries: Vec<(usize, Vector)>,
}

impl LinearIndex {
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            metric,
            dimension: None,
            entries: Vec::new(),
        }
    }

    fn validate_dimension(&self, vector: &[f32]) -> Result<()> {
        if let Some(expected) = self.dimension {
            validate_dimension(Some(expected), vector.len()).map_err(|_| {
                LinearIndexError::DimensionMismatch {
                    expected,
                    actual: vector.len(),
                }
            })?;
        }
        Ok(())
    }

    /// Returns the dimensionality tracked by the index.
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Saves the index to a JSON file at the specified path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or if serialization fails.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        save_index(self, path)
    }

    /// Loads an index from a JSON file at the specified path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or if deserialization fails.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        load_index(path)
    }
}

impl Default for LinearIndex {
    fn default() -> Self {
        Self::new(DistanceMetric::Euclidean)
    }
}

impl VectorIndex for LinearIndex {
    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn insert(&mut self, id: usize, vector: Vector) -> Result<()> {
        self.validate_dimension(&vector)?;
        if self.dimension.is_none() {
            self.dimension = Some(vector.len());
        }

        self.entries.push((id, vector));
        Ok(())
    }

    fn search(&self, query: &Vector, limit: usize) -> Result<Vec<ScoredPoint>> {
        ensure!(
            limit > 0,
            "limit must be greater than zero to execute a search"
        );
        self.validate_dimension(query)?;

        let mut candidates = Vec::with_capacity(self.entries.len());
        for (id, vector) in &self.entries {
            let distance = distance(self.metric, query, vector)?;
            candidates.push(ScoredPoint::new(*id, distance));
        }

        candidates.sort_by(|left, right| left.distance.partial_cmp(&right.distance).unwrap());
        candidates.truncate(limit.min(candidates.len()));
        Ok(candidates)
    }

    fn delete(&mut self, id: usize) -> Result<bool> {
        let initial_len = self.entries.len();
        self.entries.retain(|(entry_id, _)| *entry_id != id);
        Ok(self.entries.len() < initial_len)
    }

    fn update(&mut self, id: usize, vector: Vector) -> Result<bool> {
        self.validate_dimension(&vector)?;
        
        for (entry_id, entry_vector) in &mut self.entries {
            if *entry_id == id {
                *entry_vector = vector;
                return Ok(true);
            }
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;

    #[test]
    fn insert_and_search_returns_expected_ids() {
        let mut index = LinearIndex::new(DistanceMetric::Euclidean);
        index.insert(0, vec![0.0, 0.0]).unwrap();
        index.insert(1, vec![1.0, 0.0]).unwrap();
        index.insert(2, vec![0.0, 1.0]).unwrap();

        let result = index.search(&vec![0.0, 0.0], 2).unwrap();
        let ids: Vec<usize> = result.into_iter().map(|item| item.id).collect();
        assert_eq!(ids, vec![0, 1]);
    }

    #[test]
    fn save_and_load_preserves_index() {
        let mut original_index = LinearIndex::new(DistanceMetric::Cosine);
        original_index.insert(0, vec![1.0, 0.0]).unwrap();
        original_index.insert(1, vec![0.0, 1.0]).unwrap();
        original_index.insert(2, vec![1.0, 1.0]).unwrap();

        let mut temp_path = temp_dir();
        temp_path.push(format!("test_index_{}.json", std::process::id()));

        // Save the index
        original_index.save(&temp_path).unwrap();

        // Load the index
        let loaded_index = LinearIndex::load(&temp_path).unwrap();

        // Verify the loaded index has the same properties
        assert_eq!(loaded_index.metric(), original_index.metric());
        assert_eq!(loaded_index.len(), original_index.len());
        assert_eq!(loaded_index.dimension(), original_index.dimension());

        // Verify search results are identical
        let query = vec![1.0, 0.0];
        let original_results = original_index.search(&query, 2).unwrap();
        let loaded_results = loaded_index.search(&query, 2).unwrap();
        assert_eq!(original_results, loaded_results);

        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }
}
