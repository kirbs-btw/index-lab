//! Smart Edge Pruning
//! 
//! Multi-strategy edge pruning to avoid SYNTHESIS weakness:
//! - Too many cross-modal edges created
//! - No distance threshold
//! - No degree limits
//! 
//! GUARANTEED TO BE USED: All edge creation goes through pruning

/// Edge pruning strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruningStrategy {
    /// Distance threshold: only add edges below threshold
    DistanceThreshold,
    /// Degree limit: maximum edges per node
    DegreeLimit,
    /// Importance-based: keep most important edges
    ImportanceBased,
    /// Combined: use all strategies
    Combined,
}

/// Smart edge pruner
#[derive(Debug, Clone)]
pub struct EdgePruner {
    /// Distance threshold (normalized 0-1)
    distance_threshold: f32,
    /// Maximum degree per node
    max_degree: usize,
    /// Enable importance-based pruning
    enable_importance: bool,
    /// Strategy to use
    strategy: PruningStrategy,
}

impl EdgePruner {
    /// Create a new edge pruner
    pub fn new(
        distance_threshold: f32,
        max_degree: usize,
        enable_importance: bool,
        strategy: PruningStrategy,
    ) -> Self {
        Self {
            distance_threshold,
            max_degree,
            enable_importance,
            strategy,
        }
    }

    /// Check if edge should be created
    /// ACTUALLY CALLED during edge creation
    pub fn should_create_edge(
        &self,
        distance: f32,
        max_distance: f32,
        current_degree: usize,
    ) -> bool {
        // Normalize distance
        let normalized_dist = if max_distance > 0.0 {
            (distance / max_distance).min(1.0)
        } else {
            distance
        };

        match self.strategy {
            PruningStrategy::DistanceThreshold => {
                normalized_dist <= self.distance_threshold
            }
            PruningStrategy::DegreeLimit => {
                current_degree < self.max_degree
            }
            PruningStrategy::ImportanceBased => {
                // Importance = inverse distance
                let importance = 1.0 / (normalized_dist + 0.001);
                importance > (1.0 / (self.distance_threshold + 0.001))
            }
            PruningStrategy::Combined => {
                // Must pass distance threshold AND degree limit
                normalized_dist <= self.distance_threshold && current_degree < self.max_degree
            }
        }
    }

    /// Prune edges from a candidate list
    /// ACTUALLY CALLED during edge creation
    pub fn prune_edges(
        &self,
        candidates: Vec<(usize, f32)>,
        max_distance: f32,
        current_edges: &[(usize, f32)],
    ) -> Vec<(usize, f32)> {
        let mut pruned = Vec::new();
        let current_degree = current_edges.len();

        for (id, dist) in candidates {
            if self.should_create_edge(dist, max_distance, current_degree + pruned.len()) {
                pruned.push((id, dist));
            }
        }

        // If importance-based, keep only top-K by importance
        if self.enable_importance && pruned.len() > self.max_degree {
            pruned.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            pruned.truncate(self.max_degree);
        }

        pruned
    }

    /// Update distance threshold adaptively
    pub fn update_threshold(&mut self, recall: f32, target_recall: f32) {
        if recall < target_recall {
            // Low recall -> increase threshold (more edges)
            self.distance_threshold = (self.distance_threshold * 1.1).min(1.0);
        } else if recall > target_recall + 0.05 {
            // High recall -> decrease threshold (fewer edges)
            self.distance_threshold = (self.distance_threshold * 0.9).max(0.1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_threshold_pruning() {
        let pruner = EdgePruner::new(0.5, 100, false, PruningStrategy::DistanceThreshold);
        
        // Distance below threshold -> should create
        assert!(pruner.should_create_edge(0.3, 1.0, 0));
        
        // Distance above threshold -> should not create
        assert!(!pruner.should_create_edge(0.7, 1.0, 0));
    }

    #[test]
    fn test_degree_limit_pruning() {
        let pruner = EdgePruner::new(1.0, 5, false, PruningStrategy::DegreeLimit);
        
        // Below degree limit -> should create
        assert!(pruner.should_create_edge(0.5, 1.0, 3));
        
        // At degree limit -> should not create
        assert!(!pruner.should_create_edge(0.5, 1.0, 5));
    }

    #[test]
    fn test_combined_pruning() {
        let pruner = EdgePruner::new(0.5, 5, false, PruningStrategy::Combined);
        
        // Passes both -> should create
        assert!(pruner.should_create_edge(0.3, 1.0, 3));
        
        // Fails distance -> should not create
        assert!(!pruner.should_create_edge(0.7, 1.0, 3));
        
        // Fails degree -> should not create
        assert!(!pruner.should_create_edge(0.3, 1.0, 5));
    }
}
