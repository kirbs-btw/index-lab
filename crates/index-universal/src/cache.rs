//! Intelligent Caching System
//! 
//! Multi-level caching for performance:
//! - Distance cache (frequent distance computations)
//! - Neighbor cache (frequent neighbor lookups)
//! - Result cache (recent search results)
//! - Adaptive cache sizing

use crate::config::UniversalConfig;
use crate::error::Result;
use index_core::{ScoredPoint, Vector};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Intelligent cache with multiple cache types
#[derive(Debug, Clone)]
pub struct IntelligentCache {
    config: UniversalConfig,
    
    // Result cache: query -> results
    result_cache: HashMap<CacheKey, Vec<ScoredPoint>>,
    result_cache_size: usize,
    
    // Distance cache: (id1, id2) -> distance
    distance_cache: HashMap<(usize, usize), f32>,
    distance_cache_size: usize,
    
    // Neighbor cache: id -> neighbors
    neighbor_cache: HashMap<usize, Vec<usize>>,
    neighbor_cache_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    query_hash: u64,
    limit: usize,
}

impl IntelligentCache {
    pub fn new(config: &UniversalConfig) -> Self {
        Self {
            config: config.clone(),
            result_cache: HashMap::new(),
            result_cache_size: 1000,
            distance_cache: HashMap::new(),
            distance_cache_size: 10000,
            neighbor_cache: HashMap::new(),
            neighbor_cache_size: 5000,
        }
    }
    
    pub fn initialize(&mut self, dataset_size: usize) {
        // Adaptive cache sizing based on dataset size
        let cache_factor = self.config.cache_size_factor;
        let base_size = (dataset_size as f64).sqrt() as usize;
        
        self.result_cache_size = (base_size as f64 * cache_factor) as usize;
        self.distance_cache_size = (base_size as f64 * cache_factor * 10.0) as usize;
        self.neighbor_cache_size = (base_size as f64 * cache_factor * 5.0) as usize;
    }
    
    pub fn get_results(&self, query: &Vector, limit: usize) -> Option<Vec<ScoredPoint>> {
        if !self.config.enable_caching {
            return None;
        }
        
        let key = self.hash_query(query, limit);
        self.result_cache.get(&key).cloned()
    }
    
    pub fn cache_results(&mut self, query: &Vector, limit: usize, results: &[ScoredPoint]) {
        if !self.config.enable_caching {
            return;
        }
        
        // Evict if cache is full
        if self.result_cache.len() >= self.result_cache_size {
            self.evict_result_cache();
        }
        
        let key = self.hash_query(query, limit);
        self.result_cache.insert(key, results.to_vec());
    }
    
    pub fn get_distance(&self, id1: usize, id2: usize) -> Option<f32> {
        if !self.config.enable_caching {
            return None;
        }
        
        // Try both orders
        self.distance_cache.get(&(id1, id2))
            .or_else(|| self.distance_cache.get(&(id2, id1)))
            .copied()
    }
    
    pub fn cache_distance(&mut self, id1: usize, id2: usize, distance: f32) {
        if !self.config.enable_caching {
            return;
        }
        
        // Evict if cache is full
        if self.distance_cache.len() >= self.distance_cache_size {
            self.evict_distance_cache();
        }
        
        // Store in canonical order
        let key = if id1 < id2 { (id1, id2) } else { (id2, id1) };
        self.distance_cache.insert(key, distance);
    }
    
    pub fn get_neighbors(&self, id: usize) -> Option<Vec<usize>> {
        if !self.config.enable_caching {
            return None;
        }
        
        self.neighbor_cache.get(&id).cloned()
    }
    
    pub fn cache_neighbors(&mut self, id: usize, neighbors: Vec<usize>) {
        if !self.config.enable_caching {
            return;
        }
        
        // Evict if cache is full
        if self.neighbor_cache.len() >= self.neighbor_cache_size {
            self.evict_neighbor_cache();
        }
        
        self.neighbor_cache.insert(id, neighbors);
    }
    
    pub fn invalidate(&mut self) {
        // Clear all caches
        self.result_cache.clear();
        self.distance_cache.clear();
        self.neighbor_cache.clear();
    }
    
    fn hash_query(&self, query: &Vector, limit: usize) -> CacheKey {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        
        // Hash first few dimensions for cache key
        for &val in query.iter().take(8) {
            ((val * 1000.0) as u64).hash(&mut hasher);
        }
        limit.hash(&mut hasher);
        
        CacheKey {
            query_hash: hasher.finish(),
            limit,
        }
    }
    
    fn evict_result_cache(&mut self) {
        // Simple eviction: remove 10% oldest entries
        let to_remove = self.result_cache_size / 10;
        let keys: Vec<_> = self.result_cache.keys().take(to_remove).cloned().collect();
        for key in keys {
            self.result_cache.remove(&key);
        }
    }
    
    fn evict_distance_cache(&mut self) {
        // Simple eviction: remove 10% oldest entries
        let to_remove = self.distance_cache_size / 10;
        let keys: Vec<_> = self.distance_cache.keys().take(to_remove).cloned().collect();
        for key in keys {
            self.distance_cache.remove(&key);
        }
    }
    
    fn evict_neighbor_cache(&mut self) {
        // Simple eviction: remove 10% oldest entries
        let to_remove = self.neighbor_cache_size / 10;
        let keys: Vec<_> = self.neighbor_cache.keys().take(to_remove).cloned().collect();
        for key in keys {
            self.neighbor_cache.remove(&key);
        }
    }
    
    pub fn serialize(&self) -> Result<Vec<u8>> {
        // TODO: Implement serialization
        Ok(Vec::new())
    }
    
    pub fn deserialize(data: Vec<u8>) -> Result<Self> {
        // TODO: Implement deserialization
        Err(crate::error::UniversalError::SerializationError("Not implemented".to_string()))
    }
}
