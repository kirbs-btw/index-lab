use index_core::*;
use index_hnsw::{HnswConfig, HnswIndex};

#[test]
fn test_hnsw_recall_uniform() {
    let dataset = generate_uniform_dataset(500, 32, 42);
    let queries = generate_uniform_dataset(50, 32, 123);
    
    let config = HnswConfig {
        m_max: 16,
        ef_construction: 100,
        ef_search: 50,
        ml: 1.0 / 2.0_f64.ln(),
    };
    
    let mut index = HnswIndex::new(DistanceMetric::Euclidean, config);
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
    assert!(avg_recall > 0.8, "HNSW recall@10 too low: {}", avg_recall);
}


#[test]
fn test_hnsw_results_sorted() {
    let dataset = generate_uniform_dataset(100, 16, 42);
    
    let config = index_hnsw::HnswConfig {
        m_max: 16,
        ef_construction: 100,
        ef_search: 50,
        ml: 1.0 / 2.0_f64.ln(),
    };
    
    let mut index = index_hnsw::HnswIndex::new(DistanceMetric::Euclidean, config);
    for (id, vec) in &dataset {
        index.insert(*id, vec.clone()).unwrap();
    }
    
    let query = vec![0.0; 16];
    let results = index.search(&query, 10).unwrap();
    assert_sorted_by_distance(&results);
}
