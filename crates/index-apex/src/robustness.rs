use crate::modality::{ModalityType, MultiModalVector};
use std::collections::HashMap;

/// Tracks distribution statistics for shift detection
#[derive(Debug, Clone)]
pub struct DistributionTracker {
    /// Statistics per modality
    stats: HashMap<ModalityType, ModalityStats>,
    /// Window size for tracking
    window_size: usize,
    /// Recent vectors for comparison
    recent_vectors: Vec<MultiModalVector>,
}

#[derive(Debug, Clone)]
struct ModalityStats {
    /// Mean vector (for dense/audio)
    mean: Option<Vec<f32>>,
    /// Variance vector (for dense/audio)
    variance: Option<Vec<f32>>,
    /// Count of vectors seen
    count: usize,
    /// Recent samples for comparison
    recent_samples: Vec<Vec<f32>>,
}

impl DistributionTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            stats: HashMap::new(),
            window_size,
            recent_vectors: Vec::new(),
        }
    }
    
    pub fn update(&mut self, vector: &MultiModalVector) {
        // Update statistics for each modality
        let modalities = vector.modalities();
        for modality in modalities {
            let stats = self.stats.entry(modality).or_insert_with(|| ModalityStats {
                mean: None,
                variance: None,
                count: 0,
                recent_samples: Vec::new(),
            });
            
            match modality {
                ModalityType::Dense | ModalityType::Audio => {
                    if let Some(dense_vec) = vector.dense.as_ref().or(vector.audio.as_ref()) {
                        Self::update_dense_stats(stats, dense_vec, self.window_size);
                    }
                }
                ModalityType::Sparse => {
                    // For sparse, track term frequency distribution
                    // Simplified: just track count
                    stats.count += 1;
                }
            }
        }
        
        // Store recent vector
        self.recent_vectors.push(vector.clone());
        if self.recent_vectors.len() > self.window_size {
            self.recent_vectors.remove(0);
        }
    }
    
    fn update_dense_stats(stats: &mut ModalityStats, vector: &Vec<f32>, window_size: usize) {
        stats.count += 1;
        
        // Online mean and variance update
        if stats.mean.is_none() {
            stats.mean = Some(vector.clone());
            stats.variance = Some(vec![0.0; vector.len()]);
        } else {
            let mean = stats.mean.as_mut().unwrap();
            let variance = stats.variance.as_mut().unwrap();
            let n = stats.count as f32;
            
            // Update mean
            for (i, &val) in vector.iter().enumerate() {
                let old_mean = mean[i];
                mean[i] = old_mean + (val - old_mean) / n;
                
                // Update variance (Welford's algorithm)
                if n > 1.0 {
                    variance[i] = variance[i] + (val - old_mean) * (val - mean[i]);
                }
            }
        }
        
        // Store recent sample
        stats.recent_samples.push(vector.clone());
        if stats.recent_samples.len() > window_size {
            stats.recent_samples.remove(0);
        }
    }
}

/// Detects distribution shifts using statistical tests
#[derive(Debug, Clone)]
pub struct ShiftDetector {
    tracker: DistributionTracker,
    threshold: f32,
}

impl ShiftDetector {
    pub fn new(window_size: usize, threshold: f32) -> Self {
        Self {
            tracker: DistributionTracker::new(window_size),
            threshold,
        }
    }
    
    pub fn update(&mut self, vector: &MultiModalVector) {
        self.tracker.update(vector);
    }
    
    pub fn detect_shift(&self) -> anyhow::Result<bool> {
        // Simplified shift detection: compare recent samples to historical mean
        // In production, would use KS test or KL divergence
        
        for (modality, stats) in &self.tracker.stats {
            if stats.count < self.tracker.window_size {
                continue; // Not enough data
            }
            
            match modality {
                ModalityType::Dense | ModalityType::Audio => {
                    if let Some(mean) = &stats.mean {
                        // Compute average distance of recent samples from mean
                        let mut total_dist = 0.0;
                        for sample in &stats.recent_samples {
                            let dist = euclidean_distance(sample, mean);
                            total_dist += dist;
                        }
                        let avg_dist = total_dist / stats.recent_samples.len() as f32;
                        
                        // Compare to historical variance
                        if let Some(variance) = &stats.variance {
                            let avg_variance: f32 = variance.iter().sum::<f32>() / variance.len() as f32;
                            let std_dev = avg_variance.sqrt();
                            
                            // Shift detected if recent samples are significantly far from mean
                            if avg_dist > std_dev * self.threshold {
                                return Ok(true);
                            }
                        }
                    }
                }
                ModalityType::Sparse => {
                    // For sparse, detect shift in term frequency distribution
                    // Simplified: just check if count changed significantly
                    // In production, would track term distribution
                }
            }
        }
        
        Ok(false)
    }
    
    pub fn affected_regions(&self) -> anyhow::Result<Vec<usize>> {
        // Return IDs of vectors in recent window (affected by shift)
        Ok(self.tracker.recent_vectors.iter().map(|v| v.id).collect())
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}
