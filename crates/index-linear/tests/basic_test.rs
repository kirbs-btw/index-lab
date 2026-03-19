use index_core::*;
use index_linear::LinearIndex;

#[test]
fn test_linear_perfect_recall() {
    let dataset = generate_uniform_dataset(100, 16, 42);
    let queries = generate_uniform_dataset(10, 16, 123);
    
    let mut index = LinearIndex::new(DistanceMetric::Euclidean);
    for (id, vec) in &dataset {
        index.insert(*id, vec.clone()).unwrap();
    }
    
    for (_, query) in &queries {
        let results = index.search(query, 10).unwrap();
        let ground_truth = brute_force_knn(query, &dataset, DistanceMetric::Euclidean, 10);
        let truth_ids: Vec<usize> = ground_truth.iter().map(|sp| sp.id).collect();
        let recall = recall_at_k(&results, &truth_ids, 10);
        assert_eq!(recall, 1.0, "Linear scan should have perfect recall");
    }
}

#[test]
fn test_linear_delete() {
    let mut index = LinearIndex::new(DistanceMetric::Euclidean);
    index.insert(0, vec![1.0, 2.0]).unwrap();
    index.insert(1, vec![3.0, 4.0]).unwrap();
    
    assert_eq!(index.len(), 2);
    index.delete(0).unwrap();
    assert_eq!(index.len(), 1);
}
