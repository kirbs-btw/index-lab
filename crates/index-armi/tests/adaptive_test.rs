//! Adaptive parameter tuning tests for ARMI

use index_armi::ArmiIndex;
use index_core::{DistanceMetric, ScoredPoint, VectorIndex};

#[test]
fn test_adaptive_ef_selection() {
    let mut index = ArmiIndex::with_defaults(DistanceMetric::Euclidean);
    
    // Build index with some vectors
    for i in 0..100 {
        let vector: Vec<f32> = (0..64)
            .map(|j| (i + j) as f32 * 0.01)
            .collect();
        index.insert(i, vector).unwrap();
    }
    
    // Run multiple queries - optimizer should learn
    for _ in 0..10 {
        let query: Vec<f32> = (0..64).map(|j| j as f32 * 0.01).collect();
        let _results = index.search(&query, 10).unwrap();
    }
    
    // Index should still work
    let results = index.search(&vec![0.0; 64], 10).unwrap();
    assert_eq!(results.len(), 10);
}

#[test]
fn test_query_type_learning() {
    let mut index = ArmiIndex::with_defaults(DistanceMetric::Euclidean);
    
    // Insert diverse vectors
    for i in 0..200 {
        let vector: Vec<f32> = if i % 2 == 0 {
            // Easy queries - close vectors
            (0..32).map(|j| (i + j) as f32 * 0.01).collect()
        } else {
            // Hard queries - distant vectors
            (0..32).map(|j| 100.0 + (i + j) as f32 * 0.01).collect()
        };
        index.insert(i, vector).unwrap();
    }
    
    // Run queries with varying difficulty
    for i in 0..20 {
        let query: Vec<f32> = if i % 2 == 0 {
            vec![0.0; 32] // Easy
        } else {
            vec![100.0; 32] // Hard
        };
        let _results = index.search(&query, 10).unwrap();
    }
    
    // Verify results
    let results = index.search(&vec![0.0; 32], 10).unwrap();
    assert_eq!(results.len(), 10);
}
