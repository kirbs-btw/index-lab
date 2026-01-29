use crate::kmeans::KMeans;
use anyhow::Result;
use index_core::{
    distance, load_index, save_index, validate_dimension, DistanceMetric, ScoredPoint, Vector,
    VectorIndex,
};
use index_hnsw::{HnswConfig, HnswIndex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

mod kmeans;

#[derive(Debug, Error)]
pub enum VortexError {
    #[error("index is not trained")]
    NotTrained,
    #[error("dimension mismatch")]
    DimensionMismatch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VortexConfig {
    pub num_clusters: usize,
    pub n_probes: usize,
    pub hnsw_config: HnswConfig,
    pub max_kmeans_iters: usize,
}

impl Default for VortexConfig {
    fn default() -> Self {
        Self {
            num_clusters: 100, // Default, likely overridden based on N
            n_probes: 5,
            hnsw_config: HnswConfig::default(),
            max_kmeans_iters: 20,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VortexIndex {
    metric: DistanceMetric,
    config: VortexConfig,
    dimension: Option<usize>,

    // Core components
    centroid_index: HnswIndex,
    buckets: Vec<Vec<usize>>, // Inverted index: cluster_idx -> [vector_ids]
    vectors: HashMap<usize, Vector>, // Storage for reranking

    trained: bool,
}

impl VortexIndex {
    pub fn new(metric: DistanceMetric, config: VortexConfig) -> Self {
        Self {
            metric,
            config: config.clone(),
            dimension: None,
            centroid_index: HnswIndex::new(metric, config.hnsw_config),
            buckets: Vec::new(),
            vectors: HashMap::new(),
            trained: false,
        }
    }

    pub fn with_defaults(metric: DistanceMetric) -> Self {
        Self::new(metric, VortexConfig::default())
    }

    /// Build the index from a dataset.
    /// This performs training (K-Means) + indexing.
    pub fn build(&mut self, data: impl Iterator<Item = (usize, Vector)>) -> Result<()> {
        let dataset: Vec<(usize, Vector)> = data.collect();
        if dataset.is_empty() {
            return Ok(());
        }

        let first_dim = dataset[0].1.len();
        self.dimension = Some(first_dim);

        // 1. Train K-Means
        // Determine number of clusters if not set or default?
        // Heuristic: sqrt(N) is common.
        // If config.num_clusters is default 100, checking if we should scale it.
        // For benchmarking with 10k points, 100 is indeed sqrt(10k).
        // Let's stick to config for now, but user might want dynamic.
        let k = self.config.num_clusters;

        let vectors_only: Vec<Vector> = dataset.iter().map(|(_, v)| v.clone()).collect();

        let mut kmeans = KMeans::new(self.metric);
        kmeans.train(&vectors_only, k, self.config.max_kmeans_iters);

        // 2. Build HNSW on centroids
        self.centroid_index = HnswIndex::new(self.metric, self.config.hnsw_config);
        for (i, centroid) in kmeans.centroids.iter().enumerate() {
            self.centroid_index.insert(i, centroid.clone())?;
        }

        // 3. Assign vectors to buckets
        self.buckets = vec![Vec::new(); kmeans.centroids.len()];
        self.vectors.reserve(dataset.len());

        // We can reuse the kmeans instance for nearest centroid search
        // But we put them in HNSW. We can use HNSW to assign, or just linear scan if K is small.
        // HNSW search is O(log K), linear is O(K).
        // For building, exact assignment is better for quality, but approximate is faster.
        // Standard IVF uses exact assignment (linear scan of centroids).
        // Since we have an HNSW on centroids, we CAN use it for assignment too!
        // Let's use the HNSW index on centroids for assignment to be consistent with search?
        // Actually, for *assignment*, we usually want the true nearest centroid to balance recall.
        // But using HNSW is also fine if recall is high enough.
        // Let's use the HNSW index to find nearest centroid for assignment.

        for (id, vector) in dataset {
            // Save vector for reranking
            self.vectors.insert(id, vector.clone());

            // Find nearest centroid
            // We only need top-1
            let result = self.centroid_index.search(&vector, 1)?;
            if let Some(closest) = result.first() {
                self.buckets[closest.id].push(id);
            } else {
                // Should not happen if index is not empty
                // Fallback to bucket 0?
                if !self.buckets.is_empty() {
                    self.buckets[0].push(id);
                }
            }
        }

        self.trained = true;
        Ok(())
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        save_index(self, path)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        load_index(path)
    }
}

impl VectorIndex for VortexIndex {
    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn insert(&mut self, id: usize, vector: Vector) -> Result<()> {
        if let Some(dim) = self.dimension {
            validate_dimension(Some(dim), vector.len())
                .map_err(|_| VortexError::DimensionMismatch)?;
        } else {
            self.dimension = Some(vector.len());
        }

        // If not trained, we can't properly assign.
        // For this proof of concept, we assume build() is called first, or we just fail/panic?
        // Or we just store it and wait for build?
        // Index trait implies online updates.
        // If trained, we can insert.
        if self.trained {
            let result = self.centroid_index.search(&vector, 1)?;
            if let Some(closest) = result.first() {
                self.buckets[closest.id].push(id);
                self.vectors.insert(id, vector);
            }
        } else {
            // If strictly online, we would need to buffer or init randomly.
            // For the benchmark runner, it calls build() anyway.
            // So we can just error or no-op if not trained?
            // Actually, let's just store it in a fallback bucket or bucket 0 if we really have to.
            // But let's assume trained for now for the primary use case.
            // Or better: just add to vectors map, but not into any bucket? Then it's unreachable.
            // Let's just panic or error if not trained, to be safe.
            // But wait, `insert` returns Result.
            return Err(VortexError::NotTrained.into());
        }
        Ok(())
    }

    fn search(&self, query: &Vector, limit: usize) -> Result<Vec<ScoredPoint>> {
        if !self.trained {
            return Err(VortexError::NotTrained.into());
        }

        // 1. Route: Find nearest `n_probes` centroids
        let centroid_results = self.centroid_index.search(query, self.config.n_probes)?;

        let mut candidates = Vec::new();

        // 2. Scan: Retrieve vectors from these buckets
        for centroid_match in centroid_results {
            let bucket_idx = centroid_match.id;
            if bucket_idx < self.buckets.len() {
                for &vector_id in &self.buckets[bucket_idx] {
                    if let Some(vector) = self.vectors.get(&vector_id) {
                        // 3. Rerank: Compute exact distance
                        let dist = distance(self.metric, query, vector)?;
                        candidates.push(ScoredPoint::new(vector_id, dist));
                    }
                }
            }
        }

        // Sort and take top-k
        candidates.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(limit);

        Ok(candidates)
    }

    fn delete(&mut self, id: usize) -> Result<bool> {
        if !self.trained {
            return Err(VortexError::NotTrained.into());
        }
        
        // Remove from vectors map
        if self.vectors.remove(&id).is_some() {
            // Find and remove from bucket
            for bucket in &mut self.buckets {
                bucket.retain(|&bucket_id| bucket_id != id);
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn update(&mut self, id: usize, vector: Vector) -> Result<bool> {
        if !self.trained {
            return Err(VortexError::NotTrained.into());
        }
        
        if let Some(dim) = self.dimension {
            validate_dimension(Some(dim), vector.len())
                .map_err(|_| VortexError::DimensionMismatch)?;
        } else {
            self.dimension = Some(vector.len());
        }
        
        // Check if vector exists
        if !self.vectors.contains_key(&id) {
            return Ok(false);
        }
        
        // Find old bucket
        let old_vector = self.vectors.get(&id).cloned();
        let old_bucket_idx = if let Some(old_vec) = &old_vector {
            let result = self.centroid_index.search(old_vec, 1)?;
            result.first().map(|sp| sp.id)
        } else {
            None
        };
        
        // Find new bucket
        let result = self.centroid_index.search(&vector, 1)?;
        let new_bucket_idx = result.first().map(|sp| sp.id);
        
        // Update vector
        self.vectors.insert(id, vector);
        
        // Update bucket assignment if changed
        if let (Some(old_bucket), Some(new_bucket)) = (old_bucket_idx, new_bucket_idx) {
            if old_bucket != new_bucket {
                // Remove from old bucket
                if old_bucket < self.buckets.len() {
                    self.buckets[old_bucket].retain(|&bucket_id| bucket_id != id);
                }
                // Add to new bucket
                if new_bucket < self.buckets.len() {
                    self.buckets[new_bucket].push(id);
                }
            }
        }
        
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vortex_basic_flow() {
        let mut index = VortexIndex::with_defaults(DistanceMetric::Euclidean);

        // Generate some dummy data
        let mut data = Vec::new();
        for i in 0..100 {
            let v = vec![i as f32, (i as f32).sqrt()];
            data.push((i, v));
        }

        // Build index
        index.build(data.clone().into_iter()).unwrap();

        // Search for a point that exists
        let query = vec![50.0, (50.0f32).sqrt()];
        let results = index.search(&query, 5).unwrap();

        // Check finding itself
        // It might not be exact 0 distance due to float, but < 1e-5
        // bucket assignment should work

        // Since we insert 100 points and use default 100 clusters, it might be 1 point per cluster or so.
        // HNSW on 100 centroids.

        assert!(!results.is_empty());
        // Verify we found the exact point
        let found = results.iter().any(|p| p.id == 50 && p.distance < 1e-5);
        assert!(
            found,
            "Did not find vector 50 with near-zero distance. Results: {:?}",
            results
        );
    }
}
