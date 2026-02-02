# APEX Implementation Analysis

> **APEX**: Adaptive Performance-Enhanced eXploration  
> **Analysis Date**: 2026-02-08  
> **Focus**: Implementation quality, code review, and improvement recommendations

---

## Executive Summary

This document provides a comprehensive analysis of the APEX algorithm implementation, evaluating code quality, feature completeness, and identifying critical issues that prevent the algorithm from achieving its design goals.

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Concept** | ⭐⭐⭐⭐⭐ | Excellent synthesis of multiple algorithms |
| **Implementation** | ⭐⭐⭐☆☆ | Good architecture, but incomplete integration |
| **Code Quality** | ⭐⭐⭐☆☆ | Clean structure, but unused code and partial features |
| **Performance** | ⭐⭐⭐⭐☆ | Good complexity, but optimizations missing |
| **Research Value** | ⭐⭐⭐⭐⭐ | Demonstrates synthesis approach effectively |

### Key Findings

1. **Critical Issue**: LSH neighbor finder is implemented but not utilized during build/insert operations
2. **Partial Implementation**: Adaptive features exist but aren't integrated into standard search path
3. **Missing Integration**: Cross-modal graph created but never populated or queried
4. **Good Architecture**: Clean modular design with proper separation of concerns
5. **Strong Foundation**: Core concepts are sound, needs refinement for production use

---

## Architecture Deep Dive

### Four-Tier Design

APEX implements a four-tier architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                         APEX INDEX                               │
├─────────────────────────────────────────────────────────────────┤
│ Tier 1: Learned Multi-Modal Router (MLP)                        │
│  - MultiModalRouter with separate routers per modality          │
│  - Fuses predictions with learned weights                        │
│  - Adaptive learning via backpropagation                        │
├─────────────────────────────────────────────────────────────────┤
│ Tier 2: LSH-Accelerated Neighbor Finding                        │
│  - NeighborFinder with LshHasher                               │
│  - Multi-probe LSH for high recall                             │
│  - ⚠️ IMPLEMENTED BUT NOT USED                                 │
├─────────────────────────────────────────────────────────────────┤
│ Tier 3: Centroid Graph (HNSW)                                   │
│  - Fallback routing when router confidence low                 │
│  - Built on cluster centroids                                  │
│  - Used correctly                                              │
├─────────────────────────────────────────────────────────────────┤
│ Tier 4: Hybrid Buckets (per cluster)                            │
│  - Mini-HNSW for dense/audio vectors                           │
│  - Inverted index for sparse vectors                           │
│  - Properly implemented and used                               │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interactions

**Build Flow**:
1. K-Means clustering → centroids
2. LSH initialization → `NeighborFinder` created (but not used)
3. Centroid graph construction → HNSW on centroids
4. Router initialization → `MultiModalRouter` with correct dimensions
5. Bucket creation → `HybridBucket` per cluster
6. Vector assignment → **Uses brute-force centroid search** (should use LSH)
7. Router training → 5 epochs on dataset

**Search Flow**:
1. Router prediction → cluster probabilities
2. Cluster selection → top-K clusters
3. Bucket search → parallel search across buckets
4. Result fusion → deduplication and sorting
5. **Adaptive features skipped** (only in `search_adaptive`, not `search`)

---

## Critical Innovation Analysis

### LSH-Accelerated Neighbor Finding

**Theoretical Design**: Use LSH to find approximate neighbors in O(1) time, eliminating ARMI's O(n²) build bottleneck.

**Actual Implementation**: LSH infrastructure exists but is not utilized.

#### Code Evidence

**File**: `crates/index-apex/src/lib.rs`

**Line 145-149**: LSH NeighborFinder is created:
```rust
self.neighbor_finder = Some(NeighborFinder::new(
    dimension,
    self.config.lsh_hyperplanes,
    self.config.seed,
));
```

**Line 210-225**: But build process uses brute-force:
```rust
// Assign vectors to buckets using LSH-accelerated neighbor finding
for (id, vector) in &dataset {
    // Find nearest centroid
    let cluster_id = self.find_best_cluster(&vector)?;  // ⚠️ O(C) brute-force scan!
    
    // Convert to multi-modal vector
    let multi_modal = MultiModalVector::with_dense(*id, vector.clone());
    
    // Insert into bucket
    buckets[cluster_id].insert_multi_modal(&multi_modal, &bucket_config)?;
    // ...
}
```

**Line 237-251**: `find_best_cluster` does linear scan:
```rust
fn find_best_cluster(&self, vector: &[f32]) -> Result<usize> {
    let mut best_cluster = 0;
    let mut best_dist = f32::MAX;

    for (cluster_id, centroid) in self.centroids.iter().enumerate() {  // O(C) scan
        let dist = distance(self.metric, vector, centroid)?;
        if dist < best_dist {
            best_dist = dist;
            best_cluster = cluster_id;
        }
    }
    Ok(best_cluster)
}
```

**Impact**: 
- Expected: O(log C) via LSH bucket lookup
- Actual: O(C) linear scan through all centroids
- For C = √N = 100, this is 100× slower than needed

**Unused Code**: `NeighborFinder::insert()` and `NeighborFinder::find_neighbors()` are never called.

### Multi-Modal Router Design

**Implementation**: Well-designed extension of ATLAS router.

**File**: `crates/index-apex/src/router.rs`

**Strengths**:
- Separate routers for each modality (dense, sparse, audio)
- Proper fusion with weighted averaging
- Adaptive learning via backpropagation

**Potential Issue**: Fixed fusion weights (0.6/0.3/0.1) may not be optimal:
```rust
// Line 89-91: Fixed weights
weights.push(0.6); // Dense weight
weights.push(0.3); // Sparse weight  
weights.push(0.1); // Audio weight
```

**Recommendation**: Learn fusion weights adaptively based on query performance.

---

## Implementation Quality Assessment

### Strengths

#### 1. Modular Design
- Clear separation: `router.rs`, `lsh.rs`, `bucket.rs`, `adaptive.rs`, `robustness.rs`
- Each module has single responsibility
- Easy to test and maintain

#### 2. Error Handling
**File**: `crates/index-apex/src/error.rs`

- Custom `ApexError` enum with `thiserror`
- Proper error propagation with `?` operator
- Includes IO and serialization errors:
```rust
#[error("IO error: {0}")]
IoError(#[from] std::io::Error),

#[error("anyhow error: {0}")]
AnyhowError(#[from] anyhow::Error),
```

#### 3. Configuration System
**File**: `crates/index-apex/src/config.rs`

- Comprehensive `ApexConfig` with 20+ parameters
- Validation method checks all constraints
- Presets: `high_recall()`, `high_speed()`, `energy_efficient()`
- Good defaults for most use cases

#### 4. Integration
- Implements `VectorIndex` trait correctly
- Added to bench-runner for benchmarking
- Serialization support for save/load

### Code Quality Issues

#### 1. Unused Imports (24 warnings)

**File**: `crates/index-apex/src/lib.rs`
- Line 27: `Precision`, `SearchParams` imported but not used
- Line 35: `ClusterPrediction` imported but not used
- Line 33: `Rng` imported but not used

**File**: `crates/index-apex/src/bucket.rs`
- Line 1: `ModalityType` imported but not used
- Line 3: `distance` imported but not used

**File**: `crates/index-apex/src/router.rs`
- Line 6: `ModalityType` imported but not used

**File**: `crates/index-apex/src/graph.rs`
- Line 6: `ModalityType` imported but not used

#### 2. Unused Methods

**File**: `crates/index-apex/src/lsh.rs`
- `NeighborFinder::insert()` - Created but never called
- `NeighborFinder::find_neighbors()` - Created but never called

**File**: `crates/index-apex/src/robustness.rs`
- `ShiftDetector::affected_regions()` - Created but never called

#### 3. Inefficient Cloning

**File**: `crates/index-apex/src/lib.rs`, Line 317-318:
```rust
// Reset energy budget
let mut budget = self.energy_budget.clone();  // ⚠️ Expensive clone on every search
budget.reset();
```

**Issue**: Cloning `EnergyBudget` on every adaptive search is inefficient.

**Fix**: Use interior mutability (`RefCell`/`Mutex`) or reset in-place.

---

## Feature Completeness Analysis

### Fully Implemented Features ✅

1. **K-Means Clustering** (`lib.rs:598-677`)
   - Proper implementation with convergence checking
   - Uses seeded RNG for reproducibility
   - ⚠️ Uses Euclidean distance even if metric is Cosine

2. **Hybrid Buckets** (`bucket.rs`)
   - Mini-HNSW for dense/audio vectors
   - Inverted index for sparse vectors
   - Multi-modal insertion and search

3. **Router Training** (`lib.rs:255-270`)
   - 5 epochs with subsampling
   - Backpropagation learning
   - ⚠️ Hardcoded epochs, no convergence check

4. **Basic Search** (`lib.rs:524-558`)
   - Router-based cluster selection
   - Bucket search and fusion
   - Deduplication and sorting

### Partially Implemented Features ⚠️

#### 1. LSH Neighbor Finding
- **Status**: Infrastructure exists, not integrated
- **Location**: `lsh.rs` - fully implemented
- **Issue**: Never called during build or insert
- **Impact**: Missing the main performance improvement

#### 2. Adaptive Search Features
- **Status**: Implemented but not in standard search path
- **Location**: `lib.rs:307-383` - `search_adaptive()` method
- **Issue**: `VectorIndex::search()` doesn't use adaptive features
- **Reason**: `search_adaptive()` requires `&mut self`, but trait uses `&self`
- **Impact**: Energy optimization, precision scaling, adaptive tuning not applied

#### 3. Temporal Decay
- **Status**: Implemented but applied post-search
- **Location**: `lib.rs:352-375`
- **Issue**: Decay applied to results AFTER search, not during graph traversal
- **Impact**: Less efficient than decayed edge weights during search
- **Better Approach**: Apply decay to edge weights in `CrossModalGraph`

#### 4. Cross-Modal Graph
- **Status**: Created but never populated
- **Location**: `lib.rs:105` - `CrossModalGraph::new()`
- **Issue**: No edges added during insert, never queried during search
- **Impact**: Cross-modal connections not utilized

#### 5. Shift Detection
- **Status**: Detection works, adaptation doesn't
- **Location**: `lib.rs:476-480`
- **Issue**: Detects shift but only logs, doesn't trigger adaptation
```rust
if self.shift_detector.detect_shift()? {
    println!("Distribution shift detected");  // Just logs!
    // No actual adaptation triggered
}
```
- **Impact**: Robustness feature incomplete

### Missing Features ❌

1. **LSH-based Centroid Assignment**: Should use LSH to find nearest centroid
2. **Interior Mutability**: For adaptive features in immutable search
3. **Learned Fusion Weights**: Currently fixed at 0.6/0.3/0.1
4. **Metric-Aware Clustering**: K-means uses Euclidean regardless of metric
5. **Early Stopping**: Router training has no convergence check

---

## Performance Characteristics

### Expected Complexity

**Build**: O(N√N)
- K-Means: O(N×C×iters) where C ≈ √N
- LSH Build: O(N×h×d) where h=hyperplanes
- Graph Construction: O(N×log B) using LSH neighbors
- Router Training: O(N×d×hidden_dim)

**Search**: O(log C + log B)
- Router: O(d×hidden_dim + hidden_dim×C)
- Cluster Selection: O(C log C)
- Bucket Search: O(n_probes × log B)

**Insert**: O(log B)
- LSH Hash: O(h×d)
- Router: O(d×hidden_dim)
- Neighbor Finding: O(1) via LSH
- Graph Insert: O(log B)

### Actual Complexity

**Build**: O(N√N) ✓ + O(N×C) ⚠️
- K-Means: O(N√N) ✓
- **Centroid Assignment**: O(N×C) ⚠️ (should be O(N log C) with LSH)
- Graph Construction: O(N×log B) ✓
- Router Training: O(N×d×hidden_dim) ✓

**Search**: O(log C + log B) ✓
- Router: O(d×hidden_dim + hidden_dim×C) ✓
- Cluster Selection: O(C log C) ✓
- Bucket Search: O(n_probes × log B) ✓

**Insert**: O(C) ⚠️ (should be O(log C))
- **Centroid Finding**: O(C) ⚠️ (should use centroid graph: O(log C))
- Graph Insert: O(log B) ✓

### Bottlenecks Identified

1. **Centroid Assignment** (Build): O(N×C) instead of O(N log C)
   - Impact: For N=10K, C=100, this is 100× slower than needed
   - Fix: Use LSH or centroid graph for assignment

2. **Centroid Finding** (Insert): O(C) instead of O(log C)
   - Impact: For C=100, this is ~7× slower than needed
   - Fix: Use centroid graph search

3. **Energy Budget Cloning** (Search): Clone on every adaptive search
   - Impact: Unnecessary allocation overhead
   - Fix: Use interior mutability

---

## Comparison Matrix

| Feature | APEX | ARMI | ATLAS | FUSION |
|---------|------|------|-------|--------|
| **Build Time** | O(n√n) | O(n²) 🔴 | O(n log n) | O(n) |
| **Multi-Modal** | ✅ Full | ✅ Full | ⚠️ Dense+Sparse | ❌ Dense only |
| **Learned Routing** | ✅ Multi-modal | ❌ | ✅ Dense only | ❌ |
| **LSH Acceleration** | ⚠️ Partial | ❌ | ❌ | ✅ |
| **Temporal Decay** | ⚠️ Partial | ❌ | ❌ | ❌ |
| **Adaptive Tuning** | ⚠️ Partial | ⚠️ Partial | ❌ | ❌ |
| **Energy Efficiency** | ⚠️ Partial | ⚠️ Partial | ❌ | ❌ |
| **Shift Detection** | ⚠️ Partial | ✅ | ❌ | ❌ |
| **Deterministic** | ✅ | ✅ | ⚠️ Partial | ✅ |

**Legend**:
- ✅ Fully implemented
- ⚠️ Partially implemented
- ❌ Not implemented
- 🔴 Critical issue

### Key Differentiators

**APEX Advantages**:
- Multi-modal learned routing (extends ATLAS)
- LSH infrastructure (though not fully utilized)
- Temporal decay (though applied post-search)
- Comprehensive feature set

**APEX Disadvantages**:
- Incomplete LSH integration (main innovation not realized)
- Adaptive features not in standard search path
- More complex than simpler algorithms

---

## Specific Issues Documented

### Issue 1: LSH NeighborFinder Not Utilized

**Severity**: 🔴 Critical

**Location**: `crates/index-apex/src/lib.rs`

**Problem**: LSH infrastructure exists but is never used during build or insert operations.

**Evidence**:
- Line 145-149: `NeighborFinder` created
- Line 210-225: Build uses `find_best_cluster()` (brute-force)
- Line 237-251: `find_best_cluster()` does O(C) linear scan
- `NeighborFinder::insert()` never called
- `NeighborFinder::find_neighbors()` never called

**Impact**: 
- Build time: O(N×C) instead of O(N log C)
- Insert time: O(C) instead of O(log C)
- Missing the main performance improvement

**Fix Required**:
```rust
// Instead of:
let cluster_id = self.find_best_cluster(&vector)?;

// Should use:
let cluster_id = if let Some(finder) = &self.neighbor_finder {
    // Use LSH to find approximate nearest centroid
    let candidates = finder.find_neighbors(&vector, 10);
    // Find best among candidates
    // ...
} else {
    self.find_best_cluster(&vector)?
};
```

### Issue 2: Adaptive Features Not Used in Standard Search

**Severity**: 🟠 High

**Location**: `crates/index-apex/src/lib.rs`

**Problem**: `VectorIndex::search()` doesn't use adaptive features (energy budget, precision scaling, adaptive tuning).

**Evidence**:
- Line 524-558: `search()` method doesn't call adaptive components
- Line 307-383: `search_adaptive()` exists but requires `&mut self`
- Trait signature: `fn search(&self, ...)` prevents mutation

**Impact**: 
- Energy optimization not applied
- Precision scaling not used
- Adaptive parameter tuning not active

**Fix Required**: Use interior mutability (`RefCell`/`Mutex`) for adaptive components:
```rust
use std::cell::RefCell;

struct ApexIndex {
    // ...
    parameter_optimizer: RefCell<ParameterOptimizer>,
    energy_budget: RefCell<EnergyBudget>,
    // ...
}
```

### Issue 3: Temporal Decay Applied Post-Search

**Severity**: 🟡 Medium

**Location**: `crates/index-apex/src/lib.rs:352-375`

**Problem**: Temporal decay applied to results AFTER search, not during graph traversal.

**Current Implementation**:
```rust
// Search completes first
let results = bucket.search_multi_modal(query, limit * 2)?;

// Then decay is applied
if self.config.enable_temporal_decay {
    for result in &mut unique_results {
        result.distance = apply_temporal_decay(...);
    }
    // Re-sort after decay
}
```

**Issue**: Less efficient than decayed edge weights during search.

**Better Approach**: Apply decay to edge weights in `CrossModalGraph` during graph construction, so search naturally prioritizes recent edges.

### Issue 4: Cross-Modal Graph Not Populated

**Severity**: 🟡 Medium

**Location**: `crates/index-apex/src/lib.rs:105`

**Problem**: `CrossModalGraph` is created but never populated with edges or queried.

**Evidence**:
- Line 105: `CrossModalGraph::new()` called
- No calls to `cross_modal_graph.add_edge()` found
- No calls to `cross_modal_graph.get_distance()` found
- Graph remains empty throughout execution

**Impact**: Cross-modal connections not utilized, reducing recall for multi-modal queries.

**Fix Required**: Populate graph during insert:
```rust
// During insert, add cross-modal edges
if let Some(existing_vector) = self.vectors.get(&neighbor_id) {
    let dist = compute_cross_modal_distance(&multi_modal, existing_vector, ...)?;
    self.cross_modal_graph.add_edge(id, neighbor_id, dist, timestamp);
}
```

### Issue 5: Shift Detection Doesn't Trigger Adaptation

**Severity**: 🟡 Medium

**Location**: `crates/index-apex/src/lib.rs:476-480`

**Problem**: Shift detection works but only logs, doesn't trigger adaptation.

**Current Code**:
```rust
if self.shift_detector.detect_shift()? {
    println!("Distribution shift detected");  // Just logs!
    // No actual adaptation triggered
}
```

**Impact**: Robustness feature incomplete, no automatic adaptation to distribution shifts.

**Fix Required**: Trigger adaptation:
```rust
if self.shift_detector.detect_shift()? {
    // Retrain router
    self.train_router(&recent_vectors)?;
    
    // Optionally rebuild affected clusters
    // ...
}
```

---

## Recommendations

### Priority 1: Fix LSH Integration (Critical)

**Goal**: Actually use LSH for neighbor finding during build and insert.

**Tasks**:
1. Use `NeighborFinder` during build for centroid assignment
2. Use LSH to find approximate nearest centroid instead of brute-force
3. Use `NeighborFinder` during insert for graph neighbor discovery
4. Remove unused `NeighborFinder` methods or integrate them

**Expected Impact**:
- Build time: O(N×C) → O(N log C)
- Insert time: O(C) → O(log C)
- Realizes the main performance improvement

**Files to Modify**:
- `crates/index-apex/src/lib.rs` (build and insert methods)

### Priority 2: Complete Adaptive Features (High)

**Goal**: Make adaptive features work in standard search path.

**Tasks**:
1. Use interior mutability (`RefCell`/`Mutex`) for adaptive components
2. Integrate adaptive features into `VectorIndex::search()`
3. Apply precision scaling to distance computations
4. Update optimizer after each search

**Expected Impact**:
- Energy optimization active
- Adaptive tuning functional
- Better query performance over time

**Files to Modify**:
- `crates/index-apex/src/lib.rs` (search method, struct definition)
- `crates/index-apex/src/adaptive.rs` (if needed for interior mutability)

### Priority 3: Implement Cross-Modal Graph (Medium)

**Goal**: Populate and use cross-modal graph for better recall.

**Tasks**:
1. Add edges during insert operations
2. Query graph during search for cross-modal connections
3. Apply temporal decay to edge weights
4. Use graph for better multi-modal recall

**Expected Impact**:
- Better recall for multi-modal queries
- Temporal decay applied correctly
- Cross-modal connections utilized

**Files to Modify**:
- `crates/index-apex/src/lib.rs` (insert and search methods)
- `crates/index-apex/src/graph.rs` (if needed)

### Priority 4: Code Cleanup (Low)

**Goal**: Remove unused code and fix warnings.

**Tasks**:
1. Remove unused imports (24 warnings)
2. Remove or integrate unused methods
3. Fix K-means to use configured metric
4. Add convergence check to router training

**Expected Impact**:
- Cleaner codebase
- Fewer warnings
- Better maintainability

**Files to Modify**:
- All files with unused imports
- `crates/index-apex/src/lib.rs` (K-means, router training)

---

## Code Examples

### Example 1: Fixing LSH Integration

**Current (Brute-Force)**:
```rust
// lib.rs:237-251
fn find_best_cluster(&self, vector: &[f32]) -> Result<usize> {
    let mut best_cluster = 0;
    let mut best_dist = f32::MAX;
    
    for (cluster_id, centroid) in self.centroids.iter().enumerate() {  // O(C)
        let dist = distance(self.metric, vector, centroid)?;
        if dist < best_dist {
            best_dist = dist;
            best_cluster = cluster_id;
        }
    }
    Ok(best_cluster)
}
```

**Proposed (LSH-Accelerated)**:
```rust
fn find_best_cluster(&self, vector: &[f32]) -> Result<usize> {
    if let Some(finder) = &self.neighbor_finder {
        // Use LSH to find candidate centroids
        let candidates = finder.find_neighbors(vector, 10.min(self.centroids.len()));
        
        let mut best_cluster = 0;
        let mut best_dist = f32::MAX;
        
        // Only check candidate centroids
        for &cluster_id in &candidates {
            if cluster_id < self.centroids.len() {
                let dist = distance(self.metric, vector, &self.centroids[cluster_id])?;
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = cluster_id;
                }
            }
        }
        Ok(best_cluster)
    } else {
        // Fallback to brute-force if LSH not initialized
        // ... existing code ...
    }
}
```

### Example 2: Using Interior Mutability for Adaptive Features

**Current (Requires &mut self)**:
```rust
pub fn search_adaptive(&mut self, query: &MultiModalQuery, limit: usize) -> Result<Vec<ScoredPoint>> {
    let mut budget = self.energy_budget.clone();  // Clone
    budget.reset();
    // ...
}
```

**Proposed (Works with &self)**:
```rust
use std::cell::RefCell;

struct ApexIndex {
    // ...
    parameter_optimizer: RefCell<ParameterOptimizer>,
    energy_budget: RefCell<EnergyBudget>,
    // ...
}

fn search(&self, query: &Vector, limit: usize) -> anyhow::Result<Vec<ScoredPoint>> {
    // Reset energy budget
    self.energy_budget.borrow_mut().reset();
    
    // Select precision
    let precision = self.precision_selector.select(query, &self.energy_budget.borrow())?;
    
    // Select adaptive parameters
    let search_params = self.parameter_optimizer.borrow_mut().select_params(query)?;
    
    // ... rest of search logic ...
    
    // Update optimizer
    self.parameter_optimizer.borrow_mut().update(query, &results)?;
    
    Ok(results)
}
```

### Example 3: Populating Cross-Modal Graph

**Current (Empty Graph)**:
```rust
// lib.rs:105
cross_modal_graph: CrossModalGraph::new(config.temporal_decay_rate),
// Never populated or queried
```

**Proposed (Populated During Insert)**:
```rust
// During insert, after finding neighbors
if let Some(neighbors) = self.neighbor_finder.as_ref()
    .and_then(|f| Some(f.find_neighbors(&vector, 10))) 
{
    for neighbor_id in neighbors {
        if let Some(neighbor_vector) = self.vectors.get(&neighbor_id) {
            let dist = compute_cross_modal_distance(
                &multi_modal, 
                neighbor_vector, 
                self.metric,
                self.config.dense_weight
            )?;
            
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            
            self.cross_modal_graph.add_edge(id, neighbor_id, dist, Some(timestamp));
        }
    }
}
```

---

## Conclusion

APEX represents an ambitious synthesis of multiple vector indexing algorithms, with a strong architectural foundation and clear design goals. However, the implementation is incomplete, with several critical features (especially LSH integration) not fully realized.

### Strengths
- Excellent modular design
- Comprehensive feature set
- Good error handling and configuration
- Strong theoretical foundation

### Critical Gaps
- LSH neighbor finder not utilized (main innovation)
- Adaptive features not in standard search path
- Cross-modal graph not populated
- Several features partially implemented

### Path Forward
With the recommended fixes (especially Priority 1: LSH integration), APEX can achieve its performance goals and serve as a strong multi-modal vector index. The codebase is well-structured for these improvements, and the core concepts are sound.

**Recommendation**: Address Priority 1 and Priority 2 issues before benchmarking, as these are fundamental to the algorithm's performance claims.

---

## References

- **APEX Design Document**: `research/apex_analysis.md`
- **ARMI Analysis**: `research/armi_analysis.md` (for O(n²) problem context)
- **Source Code**: `crates/index-apex/src/`
- **Benchmark Integration**: `crates/bench-runner/src/main.rs`
