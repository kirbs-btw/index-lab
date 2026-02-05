//! Complete Temporal Integration
//! 
//! Temporal decay applied EVERYWHERE:
//! - Bucket edges
//! - Centroid graph edges
//! - Cross-modal edges
//! - All search operations
//! 
//! Fixes SYNTHESIS weakness: temporal decay only on cross-modal edges

/// Apply temporal decay to edge weight or distance
pub fn apply_temporal_decay(
    base_distance: f32,
    age_seconds: f64,
    halflife_seconds: f64,
) -> f32 {
    if age_seconds <= 0.0 {
        return base_distance;
    }
    
    let decay_factor = (-age_seconds / halflife_seconds * std::f64::consts::LN_2).exp();
    let penalty = 1.0 - decay_factor as f32;
    base_distance * (1.0 + penalty)
}

/// Normalize spatial distance to [0, 1] range for temporal fusion
pub fn normalize_spatial_distance(distance: f32, max_distance: f32) -> f32 {
    if max_distance <= 0.0 {
        return distance;
    }
    (distance / max_distance).min(1.0).max(0.0)
}

/// Compute combined temporal-spatial distance
pub fn combine_temporal_spatial(
    spatial_dist: f32,
    temporal_dist: f32,
    spatial_weight: f32,
    max_spatial_dist: f32,
) -> f32 {
    let normalized_spatial = normalize_spatial_distance(spatial_dist, max_spatial_dist);
    spatial_weight * normalized_spatial + (1.0 - spatial_weight) * temporal_dist
}

/// Get current timestamp in seconds since epoch
pub fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_decay_recent() {
        let decayed = apply_temporal_decay(0.5, 0.0, 86400.0);
        assert!((decayed - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_temporal_decay_old() {
        let decayed = apply_temporal_decay(0.5, 172800.0, 86400.0);
        assert!(decayed > 0.5);
    }

    #[test]
    fn test_normalize_spatial() {
        let normalized = normalize_spatial_distance(0.5, 1.0);
        assert_eq!(normalized, 0.5);
        
        let normalized = normalize_spatial_distance(2.0, 1.0);
        assert_eq!(normalized, 1.0);
    }
}
