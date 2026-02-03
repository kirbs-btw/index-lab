//! Temporal decay utilities for edge weights and distances
//! 
//! Provides normalized temporal decay that avoids scale mismatch issues
//! found in LIM algorithm.

/// Apply temporal decay to edge weight or distance
/// 
/// Uses exponential decay with half-life for normalized behavior.
/// Older edges/vectors get increased distance (penalized).
/// 
/// # Arguments
/// * `base_distance` - Base distance/weight before temporal decay
/// * `age_seconds` - Age of the vector/edge in seconds
/// * `halflife_seconds` - Half-life in seconds (time for decay to reach 50%)
/// 
/// # Returns
/// Decayed distance (always >= base_distance for older vectors)
pub fn apply_temporal_decay(
    base_distance: f32,
    age_seconds: f64,
    halflife_seconds: f64,
) -> f32 {
    if age_seconds <= 0.0 {
        return base_distance; // No decay for current/recent vectors
    }
    
    // Exponential decay: decay_factor = 2^(-age/halflife)
    // This gives 0.5 at halflife, 0.25 at 2×halflife, etc.
    let decay_factor = (-age_seconds / halflife_seconds * std::f64::consts::LN_2).exp();
    
    // Increase distance for older vectors
    // Recent vectors (decay_factor ≈ 1.0): no penalty
    // Old vectors (decay_factor ≈ 0.0): maximum penalty
    let penalty = 1.0 - decay_factor as f32;
    base_distance * (1.0 + penalty)
}

/// Normalize spatial distance to [0, 1] range for temporal fusion
/// 
/// Fixes LIM's scale mismatch issue where spatial distances (0→∞) 
/// and temporal distances (0→1) are incompatible.
/// 
/// # Arguments
/// * `distance` - Raw spatial distance
/// * `max_distance` - Maximum expected distance (for normalization)
/// 
/// # Returns
/// Normalized distance in [0, 1] range
pub fn normalize_spatial_distance(distance: f32, max_distance: f32) -> f32 {
    if max_distance <= 0.0 {
        return distance;
    }
    (distance / max_distance).min(1.0).max(0.0)
}

/// Compute combined temporal-spatial distance
/// 
/// Combines normalized spatial and temporal distances with learned weights.
/// 
/// # Arguments
/// * `spatial_dist` - Spatial distance (will be normalized)
/// * `temporal_dist` - Temporal distance (already normalized)
/// * `spatial_weight` - Weight for spatial component
/// * `max_spatial_dist` - Maximum spatial distance for normalization
/// 
/// # Returns
/// Combined distance
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
        // Recent vector (age = 0) should have no penalty
        let decayed = apply_temporal_decay(0.5, 0.0, 86400.0);
        assert!((decayed - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_temporal_decay_old() {
        // Old vector (age = 2×halflife) should have significant penalty
        let decayed = apply_temporal_decay(0.5, 172800.0, 86400.0); // 2 days, 1 day halflife
        assert!(decayed > 0.5); // Should be penalized
    }

    #[test]
    fn test_normalize_spatial() {
        let normalized = normalize_spatial_distance(0.5, 1.0);
        assert_eq!(normalized, 0.5);
        
        let normalized = normalize_spatial_distance(2.0, 1.0);
        assert_eq!(normalized, 1.0); // Clamped to max
        
        let normalized = normalize_spatial_distance(-0.1, 1.0);
        assert_eq!(normalized, 0.0); // Clamped to min
    }
}
