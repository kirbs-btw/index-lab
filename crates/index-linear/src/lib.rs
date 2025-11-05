//! Straight-forward baseline index that performs a linear scan over all vectors.

use anyhow::{ensure, Result};
use index_core::{distance, DistanceMetric, ScoredPoint, Vector, VectorIndex};
use serde::{Deserialize, Serialize};
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
            ensure!(
                vector.len() == expected,
                LinearIndexError::DimensionMismatch {
                    expected,
                    actual: vector.len()
                }
            );
        }
        Ok(())
    }

    /// Returns the dimensionality tracked by the index.
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
