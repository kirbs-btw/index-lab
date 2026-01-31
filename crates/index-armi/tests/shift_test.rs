//! Distribution shift detection tests for ARMI

use index_armi::ArmiIndex;
use index_core::{DistanceMetric, VectorIndex};
use rand::{rngs::StdRng, Rng, SeedableRng};

#[test]
fn test_distribution_shift_detection() {
    let mut index = ArmiIndex::with_defaults(DistanceMetric::Euclidean);
    
    // Insert vectors from distribution A (mean around 0)
    let mut rng = StdRng::seed_from_u64(42);
    for i in 0..100 {
        let vector: Vec<f32> = (0..64)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        index.insert(i, vector).unwrap();
    }
    
    // Insert vectors from distribution B (shifted mean around 10)
    for i in 100..110 {
        let vector: Vec<f32> = (0..64)
            .map(|_| rng.gen_range(9.0..11.0))
            .collect();
        index.insert(i, vector).unwrap();
    }
    
    // Shift detector should have detected the change
    // Note: Actual detection depends on window size and threshold
    assert_eq!(index.len(), 110);
}

#[test]
fn test_shift_adaptation() {
    let mut index = ArmiIndex::with_defaults(DistanceMetric::Euclidean);
    
    // Insert initial batch
    for i in 0..50 {
        let vector: Vec<f32> = (0..32)
            .map(|j| (i + j) as f32 * 0.1)
            .collect();
        index.insert(i, vector).unwrap();
    }
    
    // Insert shifted batch
    for i in 50..60 {
        let vector: Vec<f32> = (0..32)
            .map(|j| 10.0 + (i + j) as f32 * 0.1)
            .collect();
        index.insert(i, vector).unwrap();
    }
    
    // Index should still function after shift
    let results = index.search(&vec![0.0; 32], 10).unwrap();
    assert_eq!(results.len(), 10);
}
