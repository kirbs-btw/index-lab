use index_convergence::{ConvergenceIndex, ConvergenceConfig};
use index_core::{DistanceMetric, VectorIndex};

#[test]
fn test_basic_insert_search() {
    let mut index = ConvergenceIndex::with_defaults(DistanceMetric::Euclidean);
    
    // Insert vectors
    index.insert(0, vec![1.0, 0.0, 0.0]).unwrap();
    index.insert(1, vec![0.0, 1.0, 0.0]).unwrap();
    index.insert(2, vec![0.0, 0.0, 1.0]).unwrap();
    
    // Build index
    let dataset: Vec<(usize, Vec<f32>)> = vec![
        (0, vec![1.0, 0.0, 0.0]),
        (1, vec![0.0, 1.0, 0.0]),
        (2, vec![0.0, 0.0, 1.0]),
    ];
    index.build(dataset).unwrap();
    
    // Search
    let results = index.search(&vec![1.0, 0.0, 0.0], 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].id, 0);
}

#[test]
fn test_empty_search() {
    let index = ConvergenceIndex::with_defaults(DistanceMetric::Euclidean);
    let result = index.search(&vec![1.0, 0.0, 0.0], 5);
    assert!(result.is_err());  // Empty index should return error
}

#[test]
fn test_update() {
    let mut index = ConvergenceIndex::with_defaults(DistanceMetric::Euclidean);
    
    // Build with initial data
    let dataset: Vec<(usize, Vec<f32>)> = vec![
        (0, vec![1.0, 0.0, 0.0]),
        (1, vec![0.0, 1.0, 0.0]),
    ];
    index.build(dataset).unwrap();
    
    // Update vector
    let updated = index.update(0, vec![0.0, 0.0, 1.0]).unwrap();
    assert!(updated);
    
    // Verify update
    let results = index.search(&vec![0.0, 0.0, 1.0], 1).unwrap();
    assert_eq!(results[0].id, 0);
}

#[test]
fn test_delete() {
    let mut index = ConvergenceIndex::with_defaults(DistanceMetric::Euclidean);
    
    // Build with initial data
    let dataset: Vec<(usize, Vec<f32>)> = vec![
        (0, vec![1.0, 0.0, 0.0]),
        (1, vec![0.0, 1.0, 0.0]),
    ];
    index.build(dataset).unwrap();
    
    // Delete vector
    let deleted = index.delete(0).unwrap();
    assert!(deleted);
    
    // Verify deletion
    let results = index.search(&vec![1.0, 0.0, 0.0], 5).unwrap();
    assert!(!results.iter().any(|r| r.id == 0));
}

#[test]
fn test_cosine_metric() {
    let mut index = ConvergenceIndex::with_defaults(DistanceMetric::Cosine);
    
    let dataset: Vec<(usize, Vec<f32>)> = vec![
        (0, vec![1.0, 0.0]),
        (1, vec![0.0, 1.0]),
        (2, vec![1.0, 1.0]),
    ];
    index.build(dataset).unwrap();
    
    let results = index.search(&vec![1.0, 0.0], 2).unwrap();
    assert_eq!(results.len(), 2);
}

#[test]
fn test_large_dataset() {
    let mut index = ConvergenceIndex::with_defaults(DistanceMetric::Euclidean);
    
    let mut dataset = Vec::new();
    for i in 0..100 {
        let mut vec = vec![0.0; 64];
        vec[i % 64] = 1.0;
        dataset.push((i, vec));
    }
    
    index.build(dataset).unwrap();
    
    let query = vec![1.0; 64];
    let results = index.search(&query, 10).unwrap();
    assert_eq!(results.len(), 10);
}

#[test]
fn test_custom_config() {
    let config = ConvergenceConfig::high_recall();
    let mut index = ConvergenceIndex::new(DistanceMetric::Euclidean, config).unwrap();
    
    let dataset: Vec<(usize, Vec<f32>)> = vec![
        (0, vec![1.0, 0.0, 0.0]),
        (1, vec![0.0, 1.0, 0.0]),
        (2, vec![0.0, 0.0, 1.0]),
    ];
    index.build(dataset).unwrap();
    
    let results = index.search(&vec![1.0, 0.0, 0.0], 2).unwrap();
    assert_eq!(results.len(), 2);
}
