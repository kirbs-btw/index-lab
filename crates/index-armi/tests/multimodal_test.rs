//! Multi-modal functionality tests for ARMI

use index_armi::{ArmiIndex, MultiModalQuery, MultiModalVector};
use index_core::{DistanceMetric, ScoredPoint, VectorIndex};
use std::collections::HashMap;

#[test]
fn test_dense_only_queries() {
    let mut index = ArmiIndex::with_defaults(DistanceMetric::Euclidean);
    
    // Insert dense-only vectors
    index.insert(0, vec![1.0, 2.0, 3.0]).unwrap();
    index.insert(1, vec![1.1, 2.1, 3.1]).unwrap();
    index.insert(2, vec![10.0, 20.0, 30.0]).unwrap();
    
    // Query with dense vector
    let results = index.search(&vec![1.0, 2.0, 3.0], 2).unwrap();
    
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, 0); // Should be closest
    assert!(results[0].distance < results[1].distance);
}

#[test]
fn test_hybrid_dense_sparse_insert() {
    let mut index = ArmiIndex::with_defaults(DistanceMetric::Euclidean);
    
    // Insert hybrid vector with dense and sparse components
    let mut vector1 = MultiModalVector::with_dense(0, vec![1.0, 2.0, 3.0]);
    let mut sparse1 = HashMap::new();
    sparse1.insert(0, 0.5);
    sparse1.insert(5, 0.8);
    vector1.sparse = Some(sparse1);
    index.insert_multi_modal(0, vector1).unwrap();
    
    let mut vector2 = MultiModalVector::with_dense(1, vec![1.1, 2.1, 3.1]);
    let mut sparse2 = HashMap::new();
    sparse2.insert(0, 0.6);
    sparse2.insert(5, 0.7);
    vector2.sparse = Some(sparse2);
    index.insert_multi_modal(1, vector2).unwrap();
    
    assert_eq!(index.len(), 2);
}

#[test]
fn test_hybrid_dense_sparse_query() {
    let mut index = ArmiIndex::with_defaults(DistanceMetric::Euclidean);
    
    // Insert hybrid vectors
    let mut vector1 = MultiModalVector::with_dense(0, vec![1.0, 2.0, 3.0]);
    let mut sparse1 = HashMap::new();
    sparse1.insert(0, 0.5);
    sparse1.insert(5, 0.8);
    vector1.sparse = Some(sparse1);
    index.insert_multi_modal(0, vector1).unwrap();
    
    let mut vector2 = MultiModalVector::with_dense(1, vec![10.0, 20.0, 30.0]);
    let mut sparse2 = HashMap::new();
    sparse2.insert(10, 0.9);
    vector2.sparse = Some(sparse2);
    index.insert_multi_modal(1, vector2).unwrap();
    
    // Query with hybrid query
    let mut query = MultiModalQuery::with_dense(vec![1.0, 2.0, 3.0]);
    let mut query_sparse = HashMap::new();
    query_sparse.insert(0, 0.5);
    query_sparse.insert(5, 0.8);
    query.sparse = Some(query_sparse);
    
    let results = index.search_adaptive(&query, 2).unwrap();
    
    assert_eq!(results.len(), 2);
    // Vector 0 should be closer due to both dense and sparse match
    assert_eq!(results[0].id, 0);
}

#[test]
fn test_cross_modal_search() {
    let mut index = ArmiIndex::with_defaults(DistanceMetric::Euclidean);
    
    // Insert vectors with different modality combinations
    let mut vector1 = MultiModalVector::with_dense(0, vec![1.0, 2.0, 3.0]);
    let mut sparse1 = HashMap::new();
    sparse1.insert(0, 0.5);
    vector1.sparse = Some(sparse1);
    index.insert_multi_modal(0, vector1).unwrap();
    
    let mut vector2 = MultiModalVector::with_dense(1, vec![1.1, 2.1, 3.1]);
    vector2.audio = Some(vec![0.1, 0.2, 0.3]);
    index.insert_multi_modal(1, vector2).unwrap();
    
    // Query with only dense component - should find both via cross-modal edges
    let results = index.search(&vec![1.0, 2.0, 3.0], 2).unwrap();
    
    assert_eq!(results.len(), 2);
    // Both should be found despite different modality combinations
    assert!(results.iter().any(|r| r.id == 0));
    assert!(results.iter().any(|r| r.id == 1));
}
