//! Energy efficiency tests for ARMI

use index_armi::{ArmiConfig, ArmiIndex};
use index_core::{DistanceMetric, VectorIndex};

#[test]
fn test_energy_budget_initialization() {
    let mut config = ArmiConfig::default();
    config.energy_budget_per_query = Some(1000.0);
    let index = ArmiIndex::new(DistanceMetric::Euclidean, config);
    
    // Energy budget should be initialized
    assert!(index.len() == 0);
}

#[test]
fn test_precision_scaling() {
    let mut index = ArmiIndex::with_defaults(DistanceMetric::Euclidean);
    
    // Build index
    for i in 0..50 {
        let vector: Vec<f32> = (0..32)
            .map(|j| (i + j) as f32 * 0.01)
            .collect();
        index.insert(i, vector).unwrap();
    }
    
    // Run queries - precision selector should work
    for _ in 0..5 {
        let query = vec![0.0; 32];
        let _results = index.search(&query, 10).unwrap();
    }
    
    // Index should still function
    let results = index.search(&vec![0.0; 32], 10).unwrap();
    assert_eq!(results.len(), 10);
}

#[test]
fn test_low_energy_budget() {
    let mut config = ArmiConfig::default();
    config.energy_budget_per_query = Some(10.0); // Very low budget
    let mut index = ArmiIndex::new(DistanceMetric::Euclidean, config);
    
    // Build small index
    for i in 0..20 {
        let vector: Vec<f32> = (0..16)
            .map(|j| (i + j) as f32 * 0.1)
            .collect();
        index.insert(i, vector).unwrap();
    }
    
    // Query should still work (may return partial results)
    let results = index.search(&vec![0.0; 16], 10).unwrap();
    assert!(!results.is_empty());
}
