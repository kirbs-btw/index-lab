use index_core::*;
use index_ivf::IvfIndex;

#[test]
fn test_ivf_recall_uniform() {
    let dataset = generate_uniform_dataset(500, 32, 42);
    let queries = generate_uniform_dataset(50, 32, 123);
    
    let mut index = IvfIndex::with_defaults(DistanceMetric::Euclidean);
    let data_for_build: Vec<(usize, Vec<f32>)> = dataset.clone();
    index.build(data_for_build).unwrap();
    
    let mut total_recall = 0.0;
    for (_, query) in &queries {
        let results = index.search(query, 10).unwrap();
        let ground_truth = brute_force_knn(query, &dataset, DistanceMetric::Euclidean, 10);
        let truth_ids: Vec<usize> = ground_truth.iter().map(|sp| sp.id).collect();
        total_recall += recall_at_k(&results, &truth_ids, 10);
    }
    
    let avg_recall = total_recall / queries.len() as f32;
    assert!(avg_recall > 0.5, "IVF recall@10 too low: {}", avg_recall);
}
