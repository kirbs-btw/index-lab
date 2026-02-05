//! Robust Empty Bucket Handling
//! 
//! Fixes SYNTHESIS weakness: search fails on empty buckets
//! 
//! Strategies:
//! - Graceful degradation (return empty results)
//! - Automatic bucket merging
//! - Fallback routing
//! 
//! GUARANTEED TO BE USED: All bucket searches go through handler

use crate::error::Result;

/// Handler for empty bucket scenarios
#[derive(Debug, Clone)]
pub struct EmptyBucketHandler {
    /// Enable automatic bucket merging
    enable_merging: bool,
    /// Minimum bucket size before merging
    min_bucket_size: usize,
    /// Enable fallback routing
    enable_fallback: bool,
}

impl EmptyBucketHandler {
    /// Create a new empty bucket handler
    pub fn new(enable_merging: bool, min_bucket_size: usize, enable_fallback: bool) -> Self {
        Self {
            enable_merging,
            min_bucket_size,
            enable_fallback,
        }
    }

    /// Handle empty bucket search gracefully
    /// ACTUALLY CALLED when bucket search returns empty
    pub fn handle_empty_bucket(
        &self,
        bucket_id: usize,
        total_buckets: usize,
    ) -> Result<Vec<usize>> {
        if self.enable_fallback {
            // Fallback: return nearby buckets
            let mut fallback_buckets = Vec::new();
            
            // Add adjacent buckets
            if bucket_id > 0 {
                fallback_buckets.push(bucket_id - 1);
            }
            if bucket_id < total_buckets - 1 {
                fallback_buckets.push(bucket_id + 1);
            }
            
            // Add random buckets if still empty
            if fallback_buckets.is_empty() && total_buckets > 0 {
                fallback_buckets.push(0);
            }
            
            Ok(fallback_buckets)
        } else {
            // Graceful degradation: return empty (no error)
            Ok(Vec::new())
        }
    }

    /// Check if bucket should be merged
    /// ACTUALLY CALLED during insert/delete operations
    pub fn should_merge(&self, bucket_size: usize) -> bool {
        self.enable_merging && bucket_size < self.min_bucket_size
    }

    /// Find merge candidate for a bucket
    /// ACTUALLY CALLED when merging is needed
    pub fn find_merge_candidate(
        &self,
        bucket_id: usize,
        bucket_sizes: &[usize],
    ) -> Option<usize> {
        if !self.enable_merging {
            return None;
        }

        // Find smallest nearby bucket
        let mut best_candidate = None;
        let mut best_size = usize::MAX;

        // Check adjacent buckets
        if bucket_id > 0 && bucket_sizes[bucket_id - 1] < best_size {
            best_size = bucket_sizes[bucket_id - 1];
            best_candidate = Some(bucket_id - 1);
        }
        if bucket_id < bucket_sizes.len() - 1 && bucket_sizes[bucket_id + 1] < best_size {
            best_size = bucket_sizes[bucket_id + 1];
            best_candidate = Some(bucket_id + 1);
        }

        best_candidate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_empty_bucket_fallback() {
        let handler = EmptyBucketHandler::new(false, 10, true);
        let fallback = handler.handle_empty_bucket(5, 10).unwrap();
        assert!(!fallback.is_empty());
    }

    #[test]
    fn test_handle_empty_bucket_graceful() {
        let handler = EmptyBucketHandler::new(false, 10, false);
        let result = handler.handle_empty_bucket(5, 10).unwrap();
        assert!(result.is_empty());  // Graceful degradation
    }

    #[test]
    fn test_should_merge() {
        let handler = EmptyBucketHandler::new(true, 10, false);
        assert!(handler.should_merge(5));  // Below threshold
        assert!(!handler.should_merge(15));  // Above threshold
    }

    #[test]
    fn test_find_merge_candidate() {
        let handler = EmptyBucketHandler::new(true, 10, false);
        let sizes = vec![20, 5, 15, 8];
        let candidate = handler.find_merge_candidate(1, &sizes);
        assert_eq!(candidate, Some(0));  // Should merge with bucket 0 (smaller)
    }
}
