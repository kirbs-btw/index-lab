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
