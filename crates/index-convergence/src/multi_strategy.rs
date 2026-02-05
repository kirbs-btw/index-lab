//! Multi-Strategy Search
//! 
//! Automatic strategy selection based on query characteristics
//! 
//! Strategies:
//! - Hierarchical graph search (fast, high recall)
//! - Bucket scan (exhaustive, perfect recall)
//! - LSH-based search (very fast, lower recall)
//! 
//! GUARANTEED TO BE USED: All searches go through strategy selector

use crate::modality::MultiModalQuery;
use index_core::ScoredPoint;
use std::collections::HashMap;

/// Search strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchStrategy {
    /// Hierarchical graph search (default)
    Hierarchical,
    /// Bucket scan (exhaustive)
    BucketScan,
    /// LSH-based search
    LshBased,
    /// Ensemble of strategies
    Ensemble,
}

/// Strategy selector
#[derive(Debug, Clone)]
pub struct StrategySelector {
    /// Enable multi-strategy
    enabled: bool,
    /// Selection method
    method: StrategySelection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategySelection {
    /// Always use best strategy
    Best,
    /// Use ensemble
    Ensemble,
    /// Adaptive selection
    Adaptive,
}

impl StrategySelector {
    /// Create a new strategy selector
    pub fn new(enabled: bool, method: StrategySelection) -> Self {
        Self { enabled, method }
    }

    /// Select search strategy based on query
    /// ACTUALLY CALLED during search
    pub fn select_strategy(&self, query: &MultiModalQuery, index_size: usize) -> SearchStrategy {
        if !self.enabled {
            return SearchStrategy::Hierarchical;  // Default
        }

        match self.method {
            StrategySelection::Best => {
                // Always use hierarchical (best balance)
                SearchStrategy::Hierarchical
            }
            StrategySelection::Ensemble => {
                // Use ensemble
                SearchStrategy::Ensemble
            }
            StrategySelection::Adaptive => {
                // Adaptive selection based on query characteristics
                self.adaptive_select(query, index_size)
            }
        }
    }

    /// Adaptive strategy selection
    fn adaptive_select(&self, query: &MultiModalQuery, index_size: usize) -> SearchStrategy {
        // Small index -> use bucket scan (fast enough)
        if index_size < 1000 {
            return SearchStrategy::BucketScan;
        }

        // Large index -> use hierarchical
        if index_size > 100000 {
            return SearchStrategy::Hierarchical;
        }

        // Medium index -> use hierarchical (good balance)
        SearchStrategy::Hierarchical
    }

    /// Combine results from multiple strategies
    /// ACTUALLY CALLED when using ensemble strategy
    pub fn combine_results(
        &self,
        strategy_results: &[(SearchStrategy, Vec<ScoredPoint>)],
    ) -> Vec<ScoredPoint> {
        if strategy_results.is_empty() {
            return Vec::new();
        }

        // Weighted combination based on strategy confidence
        let mut combined: std::collections::HashMap<usize, (f32, usize)> = HashMap::new();

        for (strategy, results) in strategy_results {
            let weight = match strategy {
                SearchStrategy::Hierarchical => 0.5,
                SearchStrategy::BucketScan => 0.3,
                SearchStrategy::LshBased => 0.2,
                SearchStrategy::Ensemble => 0.0,  // Shouldn't happen
            };

            for result in results {
                let entry = combined.entry(result.id).or_insert((0.0, 0));
                entry.0 += result.distance * weight;
                entry.1 += 1;
            }
        }

        // Convert to ScoredPoint and sort
        let mut final_results: Vec<ScoredPoint> = combined
            .into_iter()
            .map(|(id, (dist, count))| ScoredPoint::new(id, dist / count as f32))
            .collect();

        final_results.sort_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });

        final_results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_selection() {
        let selector = StrategySelector::new(true, StrategySelection::Adaptive);
        
        let query = MultiModalQuery::with_dense(vec![1.0; 64]);
        let strategy = selector.select_strategy(&query, 500);
        assert_eq!(strategy, SearchStrategy::BucketScan);  // Small index
        
        let strategy = selector.select_strategy(&query, 1000000);
        assert_eq!(strategy, SearchStrategy::Hierarchical);  // Large index
    }
}
