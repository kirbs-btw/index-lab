use index_core::{distance, DistanceMetric, Vector};
use rand::seq::SliceRandom;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct KMeans {
    pub centroids: Vec<Vector>,
    pub metric: DistanceMetric,
}

impl KMeans {
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            centroids: Vec::new(),
            metric,
        }
    }

    pub fn train(&mut self, data: &[Vector], k: usize, max_iter: usize) {
        if data.is_empty() || k == 0 {
            return;
        }

        let mut rng = rand::thread_rng();
        
        // 1. Initialize centroids by sampling k random points
        // If data is smaller than k, just use all data
        if data.len() <= k {
            self.centroids = data.to_vec();
            // Pad with duplicates if strictly required, but for clustering typically we just take what we have
            // However, to ensure exactly k centroids might optionally duplicate, but let's stick to unique if possible.
            // If data < k, effectively we have data.len() clusters.
        } else {
            self.centroids = data
                .choose_multiple(&mut rng, k)
                .cloned()
                .collect();
        }

        // 2. Iterative optimization
        for _ in 0..max_iter {
            // Assignment step
            let mut assignments = vec![Vec::new(); self.centroids.len()];
            
            for point in data {
                let (nearest_idx, _) = self.find_nearest(point);
                assignments[nearest_idx].push(point);
            }

            // Update step
            let mut new_centroids = Vec::with_capacity(self.centroids.len());
            let mut changed = false;

            for (i, cluster_points) in assignments.into_iter().enumerate() {
                if cluster_points.is_empty() {
                    // Handle empty cluster: currently preserve old centroid
                    // Alternatively, re-init to a random point to avoid dead clusters
                    new_centroids.push(self.centroids[i].clone());
                    continue;
                }

                // Compute mean
                let dim = cluster_points[0].len();
                let mut sum = vec![0.0; dim];
                for p in &cluster_points {
                    for (d, val) in p.iter().enumerate() {
                        sum[d] += val;
                    }
                }
                
                let count = cluster_points.len() as f32;
                let mean: Vector = sum.into_iter().map(|v| v / count).collect();
                
                if distance(self.metric, &mean, &self.centroids[i]).unwrap_or(0.0) > 1e-5 {
                    changed = true;
                }
                new_centroids.push(mean);
            }

            self.centroids = new_centroids;

            if !changed {
                break;
            }
        }
    }

    pub fn find_nearest(&self, query: &Vector) -> (usize, f32) {
        let mut min_dist = f32::MAX;
        let mut nearest_idx = 0;

        for (i, centroid) in self.centroids.iter().enumerate() {
            if let Ok(dist) = distance(self.metric, query, centroid) {
                if dist < min_dist {
                    min_dist = dist;
                    nearest_idx = i;
                }
            }
        }

        (nearest_idx, min_dist)
    }
}
