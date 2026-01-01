# SEER Algorithm Critical Analysis

> **SEER**: Similarity Estimation via Efficient Routing  
> **Research Gap Addressed**: Gap 3A – Learned Index Structures  
> **Analysis Date**: 2025-12-17

---

## Executive Summary

SEER introduces a novel approach to vector indexing using learned locality prediction via random projections. While conceptually sound and well-implemented, **the current implementation is 25× slower than linear scan** due to fundamental algorithmic issues that need addressing before it can be considered production-ready.

| Verdict | Rating |
|---------|--------|
| **Concept** | ⭐⭐⭐⭐☆ |
| **Implementation Quality** | ⭐⭐⭐⭐☆ |
| **Performance** | ⭐☆☆☆☆ |
| **Scalability** | ⭐⭐☆☆☆ |
| **Research Value** | ⭐⭐⭐☆☆ |

---

## Algorithm Overview

### Core Idea

Use lightweight machine learning to **predict locality relationships** between vectors rather than computing all pairwise distances:

```
predicted_locality = predictor.score(query, candidate)
if predicted_locality > threshold:
    exact_distance = compute_distance(query, candidate)  // Only compute when likely match
```

### Components

1. **LocalityPredictor** (lines 59-201 in `lib.rs`)
   - Projects vectors onto `n_projections` random unit vectors
   - Learns weights via correlation with true distances during training
   - Scores candidates based on weighted projection similarity

2. **SeerIndex** (lines 210-385)
   - Stores all vectors with their IDs
   - Trains predictor during `build()`
   - Uses predictor to filter candidates during `search()`

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_projections` | 16 | Number of random projection features |
| `n_samples` | 1000 | Training samples for weight learning |
| `candidate_threshold` | 0.3 | Candidate selection ratio (inverted semantics!) |
| `min_candidates` | 50 | Minimum candidates to always consider |

---

## Benchmark Results

### Actual Performance (Verified 2025-12-17)

| Scenario | Points | Dim | SEER QPS | Linear QPS | Speedup | Recall |
|----------|--------|-----|----------|------------|---------|--------|
| `smoke` | 1,000 | 32 | 48.5 | 1,105.5 | **0.04×** | 98.75% |
| `recall-baseline` | 10,000 | 64 | 2.7 | 67.2 | **0.04×** | 96.48% |

> [!CAUTION]
> SEER is currently **~25× slower** than brute-force linear scan, negating its intended purpose.

### Build Time

| Scenario | Points | Build Time |
|----------|--------|------------|
| `smoke` | 1,000 | 24.38ms |
| `recall-baseline` | 10,000 | 42.98ms |

Build time is reasonable, but the training phase adds latency compared to linear index.

---

## Critical Issues

### 🔴 Issue 1: No Actual Pruning (High Severity)

**Location**: `search()` method, lines 341-368

**Problem**: The algorithm scores *every* vector before filtering, resulting in O(n) complexity:

```rust
// Current implementation scores ALL vectors first
let mut scored: Vec<(&VectorEntry, f32)> = self.vectors
    .iter()
    .map(|entry| {
        let score = predictor.score(query, &entry.vector);  // O(n) calls
        (entry, score)
    })
    .collect();
```

**Complexity Analysis**:
- Linear scan: `O(n × d)` distance computations
- SEER: `O(n × n_projections)` projections + `O(candidates × d)` distances + sorting

With `n_projections=16` and `d=64`, the per-vector overhead is comparable, and SEER adds sorting overhead.

**Fix Required**: Implement spatial partitioning (LSH buckets, VP-trees) to avoid scoring all vectors.

---

### 🔴 Issue 2: Training Provides No Benefit (High Severity)

**Location**: `train()` method, lines 136-200

**Problem**: The correlation heuristic doesn't learn discriminative weights:

```rust
// Lines 178-182: Weak learning signal
let normalized_dist = true_dist / (true_dist + 1.0);
let normalized_proj_diff = proj_diff / (proj_diff + 1.0);
let agreement = 1.0 - (normalized_dist - normalized_proj_diff).abs();
```

**Why It Fails**:
1. Random projections already preserve distances by Johnson-Lindenstrauss lemma
2. The "agreement score" converges to near-uniform weights after normalization
3. No actual learning algorithm (gradient descent, etc.) is employed

**Evidence**: Uniform weights (`1/n_projections` each) perform identically to "learned" weights.

**Fix Required**: Implement proper metric learning (e.g., Mahalanobis distance) or remove training entirely.

---

### 🟠 Issue 3: Inverted Threshold Semantics (Medium Severity)

**Location**: Line 355

**Problem**: The threshold calculation is counterintuitive:

```rust
let threshold_idx = ((1.0 - self.config.candidate_threshold) * scored.len() as f32) as usize;
```

With `candidate_threshold = 0.3`, this selects **top 70%** of candidates, not 30%.

| `candidate_threshold` | Candidates Selected | Intuitive? |
|-----------------------|---------------------|------------|
| 0.3 (default) | 70% | ❌ |
| 0.5 | 50% | ✓ |
| 0.7 | 30% | ❌ |

**Fix Required**: Rename to `candidate_ratio` or invert the calculation:
```rust
let threshold_idx = (self.config.candidate_threshold * scored.len() as f32) as usize;
```

---

### 🟠 Issue 4: Hidden Retraining Latency (Medium Severity)

**Location**: `insert()` method, lines 327-330

**Problem**: Periodic retraining during insertions can cause latency spikes:

```rust
if self.is_built && self.vectors.len() % 1000 == 0 {
    self.train_predictor();  // O(n_samples × n_vectors) hidden work
}
```

**Impact**: Every 1000th insert triggers full retraining, causing unpredictable latency.

**Fix Required**: 
- Remove automatic retraining from `insert()`
- Expose explicit `optimize()` or `retrain()` method
- Consider incremental training strategies

---

### 🟡 Issue 5: O(n) Scalability (Medium Severity)

**Problem**: Unlike HNSW (O(log n)) or IVF (O(probes × k)), SEER is fundamentally O(n):

| Algorithm | Search Complexity | SEER Compatible? |
|-----------|-------------------|------------------|
| Linear | O(n) | ✓ (equivalent) |
| IVF | O(probes × k) | Could integrate |
| HNSW | O(log n) | Could integrate |
| SEER | O(n) | Current state |

**Fix Required**: Add hierarchical structure or integrate with existing index types.

---

## What Works Well

### ✅ Clean Implementation
- Idiomatic Rust with proper error handling via `thiserror`
- Well-documented with module-level docs
- Standard patterns for serialization (`serde`)

### ✅ Good Test Coverage
- 8 unit tests covering core functionality
- Tests for cluster separation, save/load, dimension validation
- All tests pass

### ✅ Reasonable Recall
- 96-99% recall@k across benchmarks
- Comparable to exact search quality
- Correctly identifies nearest neighbors (when it finishes)

### ✅ Novel Research Direction
- First attempt at learned locality prediction in this codebase
- Addresses unexplored Research Gap 3A
- Provides foundation for future improvements

---

## Recommendations

### Immediate Fixes (Priority 1)

1. **Add LSH-based bucketing** for O(1) candidate lookup:
   ```rust
   fn hash_to_bucket(&self, vector: &[f32]) -> usize {
       let projections = self.predictor.project(vector);
       // Binary hash based on sign of projections
       projections.iter().enumerate().fold(0, |hash, (i, &p)| {
           hash | ((p > 0.0) as usize) << i
       })
   }
   ```

2. **Fix threshold semantics** with clear naming

3. **Remove training or implement proper learning**

### Medium-term Improvements (Priority 2)

4. **Integrate with HNSW** for graph-accelerated search
5. **Add multi-probe LSH** for better recall with bucketing
6. **Implement early termination** based on prediction confidence

### Research Extensions (Priority 3)

7. **Learned hash functions** (instead of random projections)
8. **Neural locality prediction** (lightweight MLP)
9. **Query-adaptive thresholds** based on data distribution

---

## Comparison with Other Indexes

| Metric | Linear | SEER | IVF | HNSW |
|--------|--------|------|-----|------|
| **Search Complexity** | O(n) | O(n) | O(probes×k) | O(log n) |
| **Build Complexity** | O(n) | O(n×samples) | O(n×clusters) | O(n log n) |
| **Memory Overhead** | 0 | O(n_proj×d) | O(clusters×d) | O(n×edges) |
| **Recall@10** | 100% | ~97% | ~95% | ~98% |
| **QPS (10K points)** | 67 | 2.7 | ~100 | ~500 |
| **Learned** | ❌ | ✅ | ❌ | ❌ |

---

## Files Modified

- [`crates/index-seer/src/lib.rs`](file:///Users/bastianlipka/Desktop/index-lab/crates/index-seer/src/lib.rs) — Main implementation
- [`crates/index-seer/Cargo.toml`](file:///Users/bastianlipka/Desktop/index-lab/crates/index-seer/Cargo.toml) — Dependencies

---

## Conclusion

SEER represents a promising research direction for learned vector indexing, but the current implementation fails to achieve its primary goal of faster search. The algorithm needs fundamental changes—particularly **spatial partitioning** and **improved learning**—before it can compete with established methods like HNSW or IVF.

**Next Steps**:
1. Implement LSH bucketing (estimated: 2-3 hours)
2. Benchmark against HNSW with equivalent recall targets
3. Update [algorithm_findings.md](./algorithm_findings.md) with SEER results

--
