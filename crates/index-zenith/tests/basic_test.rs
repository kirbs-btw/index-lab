use index_core::{distance, DistanceMetric, VectorIndex};
use index_zenith::ZenithIndex;
use rand::prelude::*;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Basic operations
// ---------------------------------------------------------------------------

#[test]
fn test_basic_insert_search() {
    let mut index = ZenithIndex::new(DistanceMetric::Euclidean);

    index.insert(0, vec![1.0, 0.0, 0.0, 0.0]).unwrap();
    index.insert(1, vec![0.0, 1.0, 0.0, 0.0]).unwrap();
    index.insert(2, vec![0.0, 0.0, 1.0, 0.0]).unwrap();
    index.insert(3, vec![0.0, 0.0, 0.0, 1.0]).unwrap();

    assert_eq!(index.len(), 4);

    let results = index.search(&vec![1.0, 0.0, 0.0, 0.0], 2).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].id, 0); // exact match
    assert!(results[0].distance < 0.001);
}

#[test]
fn test_exact_match_distance_zero() {
    let mut index = ZenithIndex::new(DistanceMetric::Euclidean);

    for i in 0..10 {
        index
            .insert(i, vec![i as f32, 0.0, 0.0, 0.0])
            .unwrap();
    }

    let results = index.search(&vec![5.0, 0.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(results[0].id, 5);
    assert!(results[0].distance < 1e-6);
}

#[test]
fn test_results_sorted_by_distance() {
    let mut index = ZenithIndex::new(DistanceMetric::Euclidean);

    for i in 0..20 {
        index
            .insert(i, vec![i as f32, 0.0, 0.0, 0.0])
            .unwrap();
    }

    let results = index.search(&vec![10.0, 0.0, 0.0, 0.0], 5).unwrap();
    for i in 1..results.len() {
        assert!(
            results[i].distance >= results[i - 1].distance,
            "Results must be sorted by distance"
        );
    }
}

// ---------------------------------------------------------------------------
// Empty index
// ---------------------------------------------------------------------------

#[test]
fn test_empty_search() {
    let index = ZenithIndex::new(DistanceMetric::Euclidean);
    let result = index.search(&vec![1.0, 2.0, 3.0], 5);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Dimension validation
// ---------------------------------------------------------------------------

#[test]
fn test_dimension_mismatch() {
    let mut index = ZenithIndex::new(DistanceMetric::Euclidean);
    index.insert(0, vec![1.0, 2.0, 3.0]).unwrap();

    let result = index.insert(1, vec![1.0, 2.0]);
    assert!(result.is_err());
}

#[test]
fn test_zero_dimension_rejected() {
    let mut index = ZenithIndex::new(DistanceMetric::Euclidean);
    let result = index.insert(0, vec![]);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Delete
// ---------------------------------------------------------------------------

#[test]
fn test_delete() {
    let mut index = ZenithIndex::new(DistanceMetric::Euclidean);

    index.insert(0, vec![1.0, 0.0, 0.0]).unwrap();
    index.insert(1, vec![0.0, 1.0, 0.0]).unwrap();
    index.insert(2, vec![0.0, 0.0, 1.0]).unwrap();

    assert!(index.delete(1).unwrap());
    assert!(!index.delete(99).unwrap()); // non-existent

    // Search should not return deleted ID
    let results = index.search(&vec![0.0, 1.0, 0.0], 3).unwrap();
    let ids: HashSet<usize> = results.iter().map(|r| r.id).collect();
    assert!(!ids.contains(&1), "Deleted node should not appear in results");
}

// ---------------------------------------------------------------------------
// Update
// ---------------------------------------------------------------------------

#[test]
fn test_update() {
    let mut index = ZenithIndex::new(DistanceMetric::Euclidean);

    index.insert(0, vec![0.0, 0.0, 0.0]).unwrap();
    index.insert(1, vec![10.0, 0.0, 0.0]).unwrap();

    // Update node 0 to be close to node 1
    assert!(index.update(0, vec![9.9, 0.0, 0.0]).unwrap());
    assert!(!index.update(99, vec![1.0, 2.0, 3.0]).unwrap()); // non-existent

    let results = index.search(&vec![10.0, 0.0, 0.0], 2).unwrap();
    // Both should be close to (10,0,0) now
    assert!(results[0].distance < 0.2);
    assert!(results[1].distance < 0.2);
}

// ---------------------------------------------------------------------------
// Cosine metric
// ---------------------------------------------------------------------------

#[test]
fn test_cosine_metric() {
    let mut index = ZenithIndex::new(DistanceMetric::Cosine);

    // Orthogonal unit-ish vectors
    index.insert(0, vec![1.0, 0.0, 0.0]).unwrap();
    index.insert(1, vec![0.0, 1.0, 0.0]).unwrap();
    index.insert(2, vec![0.707, 0.707, 0.0]).unwrap(); // 45 degrees

    let results = index.search(&vec![0.9, 0.1, 0.0], 2).unwrap();
    assert_eq!(results[0].id, 0); // most aligned with x-axis
}

// ---------------------------------------------------------------------------
// Save / Load
// ---------------------------------------------------------------------------

#[test]
fn test_save_load() {
    let mut index = ZenithIndex::new(DistanceMetric::Euclidean);
    for i in 0..10 {
        index
            .insert(i, vec![i as f32, 0.0, 0.0, 0.0])
            .unwrap();
    }

    let tmp = std::env::temp_dir().join("zenith_test_save_load.json");
    index.save(&tmp).unwrap();

    let loaded = ZenithIndex::load(&tmp).unwrap();
    assert_eq!(loaded.len(), 10);
    assert_eq!(loaded.dimension(), Some(4));

    // Search should work on loaded index
    let results = loaded.search(&vec![5.0, 0.0, 0.0, 0.0], 3).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].id, 5);

    std::fs::remove_file(tmp).ok();
}

// ---------------------------------------------------------------------------
// Recall validation — the critical test
// ---------------------------------------------------------------------------

#[test]
fn test_recall_100_vectors() {
    let mut index = ZenithIndex::with_seed(DistanceMetric::Euclidean, 123);

    let mut rng = StdRng::seed_from_u64(456);
    let dim = 32;
    let n = 100;

    // Build dataset
    let mut dataset = Vec::new();
    for i in 0..n {
        let vec: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        dataset.push((i, vec));
    }

    for (id, vec) in &dataset {
        index.insert(*id, vec.clone()).unwrap();
    }

    // Run multiple queries and compute average recall
    let n_queries = 20;
    let k = 10;
    let mut total_recall = 0.0;

    for _ in 0..n_queries {
        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        // Ground truth via brute force
        let mut ground_truth: Vec<(usize, f32)> = dataset
            .iter()
            .map(|(id, vec)| {
                let d = distance(DistanceMetric::Euclidean, &query, vec).unwrap();
                (*id, d)
            })
            .collect();
        ground_truth.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let true_top_k: HashSet<usize> = ground_truth.iter().take(k).map(|(id, _)| *id).collect();

        // Index search
        let results = index.search(&query, k).unwrap();
        let found: HashSet<usize> = results.iter().map(|r| r.id).collect();

        let recall = found.intersection(&true_top_k).count() as f32 / k as f32;
        total_recall += recall;
    }

    let avg_recall = total_recall / n_queries as f32;
    assert!(
        avg_recall >= 0.70,
        "Average recall@{} should be >= 70%, got {:.1}%",
        k,
        avg_recall * 100.0
    );
}

#[test]
fn test_recall_1000_vectors() {
    let mut index = ZenithIndex::with_seed(DistanceMetric::Euclidean, 789);

    let mut rng = StdRng::seed_from_u64(1011);
    let dim = 64;
    let n = 1000;

    // Build dataset
    let mut dataset = Vec::new();
    for i in 0..n {
        let vec: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        dataset.push((i, vec));
    }

    for (id, vec) in &dataset {
        index.insert(*id, vec.clone()).unwrap();
    }

    // Run queries
    let n_queries = 30;
    let k = 20;
    let mut total_recall = 0.0;

    for _ in 0..n_queries {
        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        // Ground truth
        let mut ground_truth: Vec<(usize, f32)> = dataset
            .iter()
            .map(|(id, vec)| {
                let d = distance(DistanceMetric::Euclidean, &query, vec).unwrap();
                (*id, d)
            })
            .collect();
        ground_truth.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let true_top_k: HashSet<usize> = ground_truth.iter().take(k).map(|(id, _)| *id).collect();

        let results = index.search(&query, k).unwrap();
        let found: HashSet<usize> = results.iter().map(|r| r.id).collect();

        let recall = found.intersection(&true_top_k).count() as f32 / k as f32;
        total_recall += recall;
    }

    let avg_recall = total_recall / n_queries as f32;
    assert!(
        avg_recall >= 0.50,
        "Average recall@{} on 1K vectors should be >= 50%, got {:.1}%",
        k,
        avg_recall * 100.0
    );
}

// ---------------------------------------------------------------------------
// Larger dataset correctness
// ---------------------------------------------------------------------------

#[test]
fn test_larger_dataset() {
    let mut index = ZenithIndex::new(DistanceMetric::Euclidean);

    let mut rng = StdRng::seed_from_u64(42);
    for i in 0..500 {
        let vec: Vec<f32> = (0..16).map(|_| rng.gen::<f32>()).collect();
        index.insert(i, vec).unwrap();
    }

    assert_eq!(index.len(), 500);

    // Should return correct number of results
    let results = index.search(&vec![0.5; 16], 10).unwrap();
    assert_eq!(results.len(), 10);
}

// ---------------------------------------------------------------------------
// Single vector
// ---------------------------------------------------------------------------

#[test]
fn test_single_vector() {
    let mut index = ZenithIndex::new(DistanceMetric::Euclidean);
    index.insert(42, vec![1.0, 2.0, 3.0]).unwrap();

    let results = index.search(&vec![1.0, 2.0, 3.0], 5).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, 42);
    assert!(results[0].distance < 1e-6);
}
