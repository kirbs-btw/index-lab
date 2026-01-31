//! Deterministic behavior tests for ARMI

use index_armi::{ArmiConfig, ArmiIndex};
use index_core::{DistanceMetric, VectorIndex};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[test]
fn test_deterministic_build() {
    // Create two indexes with same seed
    let mut config1 = ArmiConfig::default();
    config1.deterministic = true;
    config1.seed = 42;
    let mut index1 = ArmiIndex::new(DistanceMetric::Euclidean, config1);
    
    let mut config2 = ArmiConfig::default();
    config2.deterministic = true;
    config2.seed = 42;
    let mut index2 = ArmiIndex::new(DistanceMetric::Euclidean, config2);
    
    // Insert same vectors in same order
    let mut rng = StdRng::seed_from_u64(42);
    for i in 0..50 {
        let vector: Vec<f32> = (0..32)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        index1.insert(i, vector.clone()).unwrap();
        index2.insert(i, vector).unwrap();
    }
    
    // Query both with same query
    let query: Vec<f32> = (0..32).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let results1 = index1.search(&query, 10).unwrap();
    let results2 = index2.search(&query, 10).unwrap();
    
    // Results should be identical (same IDs and distances)
    assert_eq!(results1.len(), results2.len());
    for (r1, r2) in results1.iter().zip(results2.iter()) {
        assert_eq!(r1.id, r2.id);
        assert!((r1.distance - r2.distance).abs() < 1e-5);
    }
}

#[test]
fn test_reproducible_results() {
    let mut config = ArmiConfig::default();
    config.deterministic = true;
    config.seed = 123;
    let mut index = ArmiIndex::new(DistanceMetric::Euclidean, config);
    
    // Build index
    for i in 0..100 {
        let vector: Vec<f32> = (0..64)
            .map(|j| (i + j) as f32 * 0.01)
            .collect();
        index.insert(i, vector).unwrap();
    }
    
    // Run same query multiple times
    let query = vec![0.0; 64];
    let results1 = index.search(&query, 10).unwrap();
    let results2 = index.search(&query, 10).unwrap();
    
    // Results should be identical
    assert_eq!(results1.len(), results2.len());
    for (r1, r2) in results1.iter().zip(results2.iter()) {
        assert_eq!(r1.id, r2.id);
        assert!((r1.distance - r2.distance).abs() < 1e-5);
    }
}
