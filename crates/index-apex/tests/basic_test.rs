use index_apex::{ApexIndex, MultiModalQuery, MultiModalVector};
use index_core::{DistanceMetric, VectorIndex};
use std::collections::HashMap;

#[test]
fn test_apex_basic_insert_and_search() {
    let mut index = ApexIndex::with_defaults(DistanceMetric::Euclidean);
    
    index.insert(0, vec![1.0, 2.0, 3.0]).unwrap();
    index.insert(1, vec![1.1, 2.1, 3.1]).unwrap();
    index.insert(2, vec![10.0, 20.0, 30.0]).unwrap();
    
    assert_eq!(index.len(), 3);
    
    let results = index.search(&vec![1.0, 2.0, 3.0], 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, 0);
}

#[test]
fn test_apex_empty_search() {
    let index = ApexIndex::with_defaults(DistanceMetric::Euclidean);
    let result = index.search(&vec![1.0, 2.0, 3.0], 5);
    assert!(result.is_err());
}

#[test]
fn test_apex_multimodal_query() {
    let mut index = ApexIndex::with_defaults(DistanceMetric::Euclidean);
    
    // Insert dense vector
    index.insert(0, vec![1.0, 2.0, 3.0]).unwrap();
    
    // Create multi-modal query
    let mut sparse = HashMap::new();
    sparse.insert(1, 0.5);
    let query = MultiModalQuery::with_hybrid(vec![1.0, 2.0, 3.0], sparse);
    
    // Search with adaptive method (requires mutable)
    let mut temp_index = index.clone();
    let results = temp_index.search_adaptive(&query, 1).unwrap();
    assert!(!results.is_empty());
}

#[test]
fn test_apex_build() {
    let mut index = ApexIndex::with_defaults(DistanceMetric::Euclidean);
    
    let dataset = vec![
        (0, vec![1.0, 2.0, 3.0]),
        (1, vec![1.1, 2.1, 3.1]),
        (2, vec![10.0, 20.0, 30.0]),
    ];
    
    index.build(dataset).unwrap();
    assert_eq!(index.len(), 3);
}

#[test]
fn test_apex_delete() {
    let mut index = ApexIndex::with_defaults(DistanceMetric::Euclidean);
    
    index.insert(0, vec![1.0, 2.0, 3.0]).unwrap();
    index.insert(1, vec![1.1, 2.1, 3.1]).unwrap();
    
    assert_eq!(index.len(), 2);
    
    let deleted = index.delete(0).unwrap();
    assert!(deleted);
    assert_eq!(index.len(), 1);
}

#[test]
fn test_apex_update() {
    let mut index = ApexIndex::with_defaults(DistanceMetric::Euclidean);
    
    index.insert(0, vec![1.0, 2.0, 3.0]).unwrap();
    
    let updated = index.update(0, vec![2.0, 3.0, 4.0]).unwrap();
    assert!(updated);
    
    let results = index.search(&vec![2.0, 3.0, 4.0], 1).unwrap();
    assert_eq!(results[0].id, 0);
}
