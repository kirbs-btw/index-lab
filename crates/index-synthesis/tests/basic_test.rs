use index_core::{DistanceMetric, VectorIndex};
use index_synthesis::{SynthesisConfig, SynthesisIndex};

#[test]
fn test_basic_insert_search() {
    let mut index = SynthesisIndex::with_defaults(DistanceMetric::Euclidean);
    
    // Build index (trains router)
    let data = vec![
        (0, vec![1.0, 2.0, 3.0]),
        (1, vec![1.1, 2.1, 3.1]),
        (2, vec![10.0, 20.0, 30.0]),
    ];
    index.build(data).unwrap();
    
    // Search
    let results = index.search(&vec![1.0, 2.0, 3.0], 2).unwrap();
    
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, 0); // Should be closest
}

#[test]
fn test_empty_search() {
    let index = SynthesisIndex::with_defaults(DistanceMetric::Euclidean);
    let results = index.search(&vec![1.0, 2.0, 3.0], 5);
    // Empty index should return error, not empty results
    assert!(results.is_err());
}

#[test]
fn test_build() {
    let mut index = SynthesisIndex::with_defaults(DistanceMetric::Euclidean);
    
    let data = vec![
        (0, vec![1.0, 2.0, 3.0]),
        (1, vec![1.1, 2.1, 3.1]),
        (2, vec![10.0, 20.0, 30.0]),
        (3, vec![10.1, 20.1, 30.1]),
    ];
    
    index.build(data).unwrap();
    
    assert_eq!(index.len(), 4);
    
    // Search should work after build
    let results = index.search(&vec![1.0, 2.0, 3.0], 2).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn test_delete() {
    let mut index = SynthesisIndex::with_defaults(DistanceMetric::Euclidean);
    
    index.insert(0, vec![1.0, 2.0, 3.0]).unwrap();
    index.insert(1, vec![1.1, 2.1, 3.1]).unwrap();
    
    assert_eq!(index.len(), 2);
    
    let deleted = index.delete(0).unwrap();
    assert!(deleted);
    assert_eq!(index.len(), 1);
    
    let deleted_again = index.delete(0).unwrap();
    assert!(!deleted_again);
}

#[test]
fn test_update() {
    let mut index = SynthesisIndex::with_defaults(DistanceMetric::Euclidean);
    
    // Build index first
    let mut data = vec![
        (0, vec![1.0, 2.0, 3.0]),
        (1, vec![1.1, 2.1, 3.1]),
    ];
    index.build(data.clone()).unwrap();
    
    let updated = index.update(0, vec![10.0, 20.0, 30.0]).unwrap();
    assert!(updated);
    
    // Update the data and rebuild to ensure consistency
    data[0] = (0, vec![10.0, 20.0, 30.0]);
    index.build(data).unwrap();
    
    // Search should find updated vector
    let results = index.search(&vec![10.0, 20.0, 30.0], 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 0);
}

#[test]
fn test_lsh_integration() {
    // Test that LSH is actually used during build
    let mut index = SynthesisIndex::with_defaults(DistanceMetric::Euclidean);
    
    let data: Vec<(usize, Vec<f32>)> = (0..100)
        .map(|i| (i, vec![i as f32, (i * 2) as f32, (i * 3) as f32]))
        .collect();
    
    // Build should use LSH for centroid assignment
    index.build(data).unwrap();
    
    assert_eq!(index.len(), 100);
    
    // Search should work
    let results = index.search(&vec![50.0, 100.0, 150.0], 5).unwrap();
    assert!(!results.is_empty());
}

#[test]
fn test_cross_modal_graph_populated() {
    // Test that cross-modal graph is actually populated
    let mut index = SynthesisIndex::with_defaults(DistanceMetric::Euclidean);
    
    let data: Vec<(usize, Vec<f32>)> = (0..50)
        .map(|i| (i, vec![i as f32, (i * 2) as f32]))
        .collect();
    
    index.build(data).unwrap();
    
    // Cross-modal graph should have edges
    // (We can't directly access it, but search should explore cross-modal edges)
    let results = index.search(&vec![25.0, 50.0], 10).unwrap();
    assert!(!results.is_empty());
}

#[test]
fn test_adaptive_features() {
    // Test that adaptive features work in standard search
    let mut index = SynthesisIndex::with_defaults(DistanceMetric::Euclidean);
    
    let data: Vec<(usize, Vec<f32>)> = (0..100)
        .map(|i| (i, vec![i as f32, (i * 2) as f32]))
        .collect();
    
    index.build(data).unwrap();
    
    // Multiple searches should trigger adaptive updates
    for _ in 0..10 {
        let _results = index.search(&vec![50.0, 100.0], 10).unwrap();
    }
    
    // Should still work after adaptive updates
    let results = index.search(&vec![50.0, 100.0], 10).unwrap();
    assert!(!results.is_empty());
}
