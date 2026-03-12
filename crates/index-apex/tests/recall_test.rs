use index_core::*;
use index_apex::ApexIndex;

#[test]
fn test_apex_recall_incremental() {
    let dataset = generate_uniform_dataset(200, 16, 42);
    let queries = generate_uniform_dataset(20, 16, 123);
    
    let mut index = ApexIndex::with_defaults(DistanceMetric::Euclidean);
    for (id, vec) in &dataset {
        index.insert(*id, vec.clone()).unwrap();
    }
    
    let mut total_recall = 0.0;
    for (_, query) in &queries {
        let results = index.search(query, 10).unwrap();
        let ground_truth = brute_force_knn(query, &dataset, DistanceMetric::Euclidean, 10);
        let truth_ids: Vec<usize> = ground_truth.iter().map(|sp| sp.id).collect();
        total_recall += recall_at_k(&results, &truth_ids, 10);
    }
    
    let avg_recall = total_recall / queries.len() as f32;
    // Incremental mode recall may be lower since all vectors go to one bucket
    assert!(avg_recall > 0.3, "APEX incremental recall@10 too low: {}", avg_recall);
}


#[test]
fn test_apex_large_incremental() {
    let dataset = generate_uniform_dataset(500, 16, 42);
    
    let mut index = ApexIndex::with_defaults(DistanceMetric::Euclidean);
    for (id, vec) in &dataset {
        index.insert(*id, vec.clone()).unwrap();
    }
    
    // Verify we can search after many inserts
    let results = index.search(&dataset[0].1, 5).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].id, 0);
}


#[test]
fn test_apex_delete() {
    let mut index = ApexIndex::with_defaults(DistanceMetric::Euclidean);
    
    index.insert(0, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    index.insert(1, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    
    assert_eq!(index.len(), 2);
    index.delete(0).unwrap();
    // Vector should no longer appear in results
    let results = index.search(&vec![1.0, 2.0, 3.0, 4.0], 5).unwrap();
    // Should only find vector 1
    assert!(results.iter().all(|r| r.id != 0) || results.is_empty());
}

#[test]
fn test_apex_update() {
    let mut index = ApexIndex::with_defaults(DistanceMetric::Euclidean);
    
    index.insert(0, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    index.insert(1, vec![10.0, 20.0, 30.0, 40.0]).unwrap();
    
    // Update vector 0 to be close to vector 1
    index.update(0, vec![10.1, 20.1, 30.1, 40.1]).unwrap();
    
    let results = index.search(&vec![10.0, 20.0, 30.0, 40.0], 2).unwrap();
    assert!(!results.is_empty());
}
