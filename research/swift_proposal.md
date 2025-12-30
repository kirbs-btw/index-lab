# SWIFT: Sparse-Weighted Index with Fast Traversal

> A practical vector index that actually works – combining LSH bucketing, hierarchical navigation, and optional learned routing

---

## The Problem: All Our Algorithms Are O(n)

| Algorithm | Core Issue | Why It's Slow |
|-----------|------------|---------------|
| **SEER** | No pruning | Scores ALL vectors before filtering |
| **LIM** | No spatial index | Checks ALL clusters on insert |
| **Hybrid** | No inverted index | Linear scan for sparse matching |
| **NEXUS** | Too complex | Spectral decomposition is expensive |

**The Pattern**: Every algorithm iterates over too much data. We need **O(1) or O(log n) access** to relevant candidates.

---

## Core Insight: Layered Filtering

Real-world similarity search can be decomposed into three stages:

```
Stage 1: BUCKETING     → O(1) to find candidate buckets
Stage 2: NAVIGATION    → O(log n) to traverse within buckets  
Stage 3: REFINEMENT    → O(k) to rerank final candidates
```

HNSW only does stage 2. LSH only does stage 1. SWIFT does all three.

---

## SWIFT Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        SWIFT INDEX                          │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: LSH Buckets (O(1) candidate generation)           │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                   │
│  │ B00 │ │ B01 │ │ B10 │ │ B11 │ │ ... │  ← 2^h buckets    │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘                   │
│     │       │       │       │       │                       │
├─────┼───────┼───────┼───────┼───────┼───────────────────────┤
│  Layer 2: Mini-Graphs per Bucket (O(log b) navigation)      │
│     ▼       ▼       ▼       ▼       ▼                       │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                   │
│  │HNSW │ │HNSW │ │HNSW │ │HNSW │ │ ... │  ← Mini-graphs    │
│  │ 20  │ │ 50  │ │ 30  │ │ 100 │ │     │    per bucket     │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                   │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Optional Extensions                               │
│  • Sparse Inverted Index (for hybrid search)                │
│  • Temporal Decay (for time-aware search)                   │
│  • Learned Router (for query-adaptive bucket selection)     │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer 1: LSH Bucketing (O(1) Access)

### Why LSH Fixes SEER's Problem

SEER scores all vectors because it has no spatial structure. LSH provides immediate O(1) candidate generation.

### Implementation: SimHash for Dense Vectors

```rust
struct LshBucketer {
    hyperplanes: Vec<Vec<f32>>,  // h random hyperplanes
    n_buckets: usize,            // 2^h buckets
}

impl LshBucketer {
    fn hash(&self, vector: &[f32]) -> usize {
        let mut bucket = 0usize;
        for (i, hyperplane) in self.hyperplanes.iter().enumerate() {
            let dot = dot_product(vector, hyperplane);
            if dot > 0.0 {
                bucket |= 1 << i;
            }
        }
        bucket
    }
}
```

### Multi-Probe LSH for Better Recall

Don't just check one bucket – probe neighboring buckets too:

```rust
fn get_probe_buckets(&self, query: &[f32], n_probes: usize) -> Vec<usize> {
    let primary = self.hash(query);
    let mut buckets = vec![primary];
    
    // Add buckets that differ by 1 bit (Hamming distance 1)
    for bit in 0..self.hyperplanes.len() {
        buckets.push(primary ^ (1 << bit));
    }
    
    buckets.truncate(n_probes);
    buckets
}
```

### Complexity Analysis

| Operation | SEER | SWIFT |
|-----------|------|-------|
| Find candidates | O(n) | O(1) hash + O(b) bucket size |
| With multi-probe | O(n) | O(probes × b/2^h) |

For n=1M vectors, h=10 bits: ~1000 vectors per bucket instead of 1M.

---

## Layer 2: Mini-Graphs per Bucket (O(log b) Navigation)

### Why This Beats Flat Buckets

Pure LSH still requires linear scan within buckets. Each bucket gets its own small HNSW graph for O(log b) search.

### Implementation

```rust
struct SwiftIndex {
    bucketer: LshBucketer,
    buckets: Vec<MiniGraph>,  // One mini-graph per bucket
    vectors: Vec<VectorEntry>,
}

struct MiniGraph {
    local_ids: Vec<usize>,     // IDs of vectors in this bucket
    edges: Vec<Vec<usize>>,    // Navigable graph within bucket
    entry_point: usize,
}

impl SwiftIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<usize> {
        // Stage 1: O(1) bucket lookup
        let probe_buckets = self.bucketer.get_probe_buckets(query, 4);
        
        // Stage 2: O(log b) search per bucket
        let mut candidates = Vec::new();
        for bucket_id in probe_buckets {
            let bucket = &self.buckets[bucket_id];
            let local_results = bucket.search(query, k * 2, &self.vectors);
            candidates.extend(local_results);
        }
        
        // Stage 3: O(k log k) final reranking
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(k);
        candidates.into_iter().map(|(id, _)| id).collect()
    }
}
```

### Why Mini-Graphs Instead of Full HNSW?

| Approach | Build Time | Memory | Search |
|----------|------------|--------|--------|
| Single HNSW | O(n log n) | O(n × M) | O(log n) |
| LSH only | O(n) | O(n) | O(n/2^h) |
| **SWIFT** | O(n log b) | O(n × M/4) | O(probes × log b) |

Mini-graphs are smaller (avg ~1000 vectors), so:
- Faster to build (less graph navigation per insert)
- Lower memory (fewer edges needed for small graphs)
- Can rebuild individual buckets without touching others

---

## Layer 3: Optional Extensions

### 3A. Sparse Inverted Index (Fixes Hybrid's Problem)

```rust
struct SparseIndex {
    inverted: HashMap<u32, Vec<(usize, f32)>>,  // term_id -> [(doc_id, weight)]
}

impl SparseIndex {
    fn search(&self, query_sparse: &HashMap<u32, f32>, k: usize) -> Vec<(usize, f32)> {
        let mut scores: HashMap<usize, f32> = HashMap::new();
        
        // Only iterate query terms, not all documents
        for (term_id, query_weight) in query_sparse {
            if let Some(posting_list) = self.inverted.get(term_id) {
                for (doc_id, doc_weight) in posting_list {
                    *scores.entry(*doc_id).or_default() += query_weight * doc_weight;
                }
            }
        }
        
        // Return top-k by score
        let mut results: Vec<_> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        results
    }
}
```

**Complexity**: O(query_terms × avg_posting_list_len) instead of O(n × query_terms)

### 3B. Temporal Decay (Fixes LIM's Problems)

Integrate timestamps without O(n) cluster search:

```rust
struct TemporalEntry {
    id: usize,
    vector: Vec<f32>,
    timestamp: u64,
    bucket: usize,  // Pre-computed bucket assignment
}

fn temporal_search(&self, query: &[f32], k: usize, reference_time: u64) -> Vec<usize> {
    let candidates = self.search_dense(query, k * 3);  // Get more candidates
    
    // Apply temporal reranking
    candidates.into_iter()
        .map(|(id, dist)| {
            let entry = &self.vectors[id];
            let age = (reference_time - entry.timestamp) as f32;
            let temporal_weight = (-age * self.decay_rate).exp();
            let combined = dist * (1.0 - self.temporal_weight) 
                         + (1.0 - temporal_weight) * self.temporal_weight;
            (id, combined)
        })
        .sorted_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .take(k)
        .map(|(id, _)| id)
        .collect()
}
```

**Key Fix**: Temporal decay is applied AFTER fast candidate generation, not during O(n) scan.

### 3C. Learned Router (Optional, Inspired by SEER/NEXUS)

For the 1% of queries where standard probing fails, learn which buckets to check:

```rust
struct LearnedRouter {
    // Tiny MLP: query features -> bucket probabilities
    weights: Vec<Vec<f32>>,
}

impl LearnedRouter {
    fn predict_buckets(&self, query: &[f32], n: usize) -> Vec<usize> {
        // Forward pass through tiny network
        let probs = self.forward(query);
        
        // Return top-n buckets by probability
        probs.iter()
            .enumerate()
            .sorted_by(|a, b| b.1.partial_cmp(&a.1).unwrap())
            .take(n)
            .map(|(idx, _)| idx)
            .collect()
    }
}
```

**Training**: Collect (query, true_nearest_bucket) pairs from search logs.

**When to use**: Enable only after basic SWIFT is working well.

---

## Why SWIFT Beats the Alternatives

### vs. SEER (25× slower)

| Issue | SEER | SWIFT |
|-------|------|-------|
| Candidate generation | O(n) scoring | O(1) hash lookup |
| Distance computation | All then filter | Only bucket contents |
| Learning | Broken correlation | Optional router |

### vs. LIM (O(n) cluster search)

| Issue | LIM | SWIFT |
|-------|-----|-------|
| Cluster assignment | O(n_clusters) per insert | O(1) hash |
| Cluster merging | Expensive, latency spikes | Not needed |
| Temporal integration | Core algorithm | Post-hoc reranking |

### vs. Hybrid (Linear sparse scan)

| Issue | Hybrid | SWIFT |
|-------|--------|-------|
| Sparse matching | O(n × terms) | O(query_terms × posting_len) |
| Dense matching | O(n) | O(probes × log b) |
| Score fusion | During O(n) scan | After candidate generation |

### vs. NEXUS (Too complex)

| Issue | NEXUS | SWIFT |
|-------|-------|-------|
| Build time | O(n²) spectral decomposition | O(n log b) |
| Training | Requires optimal path generation | Optional, simple |
| Implementation | Complex (eigenvectors, NRP) | Standard components |

---

## Theoretical Analysis

### Search Complexity

```
Total = O(hash) + O(probes × (bucket_lookup + graph_search)) + O(rerank)
      = O(h)   + O(p × (1 + log(n/2^h)))                     + O(k log k)
      = O(p × log(n/2^h))
```

With default parameters (h=10, p=4, n=1M):
- SWIFT: O(4 × log(1000)) ≈ O(40) distance computations
- HNSW: O(log 1M) ≈ O(20) distance computations
- Linear: O(1M) distance computations

### Recall Guarantees

Using standard LSH theory:
- P(collision | d(a,b) < r) ≥ 1 - (1 - p₁^h)^L
- Multi-probe adds approximate factor of (h choose t) for t-bit differences

**Target**: 95%+ recall@10 with 4-probe configuration.

### Memory Overhead

| Component | Size |
|-----------|------|
| Hyperplanes | h × d floats = 10 × 768 × 4 = 30 KB |
| Bucket assignments | n × 2 bytes = 2 MB for 1M vectors |
| Mini-graph edges | n × M/4 × 4 bytes = 4 MB for 1M vectors |
| **Total overhead** | ~2% over raw vectors |

---

## Implementation Plan

### Phase 1: Core LSH + Flat Buckets (1 day)

```
[ ] Implement LshBucketer with SimHash
[ ] Create SwiftIndex with bucket storage
[ ] Flat search within buckets (no mini-graphs yet)
[ ] Basic benchmarks vs. Linear
```

**Success Criteria**: Faster than linear scan with >90% recall

### Phase 2: Mini-Graphs (1 day)

```
[ ] Implement MiniGraph with simplified HNSW
[ ] Integrate with SwiftIndex
[ ] Benchmark vs. full HNSW
```

**Success Criteria**: Within 2× of HNSW speed, better build time

### Phase 3: Hybrid Extension (1 day)

```
[ ] Add SparseIndex inverted index
[ ] Implement hybrid search (dense + sparse)
[ ] Benchmark vs. current Hybrid index
```

**Success Criteria**: 10× faster than current Hybrid on hybrid queries

### Phase 4: Temporal Extension (0.5 days)

```
[ ] Add timestamp support
[ ] Implement temporal reranking
[ ] Benchmark vs. LIM
```

**Success Criteria**: Faster than LIM, equivalent temporal behavior

### Phase 5: Learned Router (Optional, 1 day)

```
[ ] Implement tiny MLP router
[ ] Training data collection
[ ] Benchmark improvement from routing
```

**Success Criteria**: 5%+ recall improvement on hard queries

---

## Configuration & Defaults

```rust
struct SwiftConfig {
    // LSH parameters
    n_hyperplanes: usize,       // Default: 10 (1024 buckets)
    n_probes: usize,            // Default: 4
    
    // Mini-graph parameters  
    mini_graph_m: usize,        // Default: 8 (edges per node)
    mini_graph_ef: usize,       // Default: 32
    min_bucket_size: usize,     // Default: 16 (below this, no graph)
    
    // Optional features
    enable_sparse: bool,        // Default: false
    enable_temporal: bool,      // Default: false
    enable_learned_router: bool,// Default: false
    
    // Sparse config (if enabled)
    dense_weight: f32,          // Default: 0.6
    
    // Temporal config (if enabled)
    temporal_decay: f32,        // Default: 0.01
    temporal_weight: f32,       // Default: 0.3
}
```

---

## Potential Weaknesses & Mitigations

| Weakness | Mitigation |
|----------|------------|
| **LSH bucket imbalance** | Use consistent hashing or dynamic buckets |
| **High-dimensional curse** | More hyperplanes (h∝d) or random projections |
| **Cold start** | Fall back to linear scan for first 1000 vectors |
| **Dynamic updates** | Incremental bucket reassignment on drift detection |
| **Memory fragmentation** | Periodic compaction of mini-graphs |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Recall@10** | ≥95% | vs. brute-force results |
| **QPS (10K)** | ≥500 | bench-runner recall-baseline |
| **QPS (1M)** | ≥100 | large-scale benchmark |
| **Build time** | ≤HNSW | seconds to build index |
| **Memory** | ≤1.1× raw | bytes per vector |

---

## Conclusion

SWIFT addresses the core problem shared by SEER, LIM, Hybrid, and NEXUS: **O(n) operations where O(log n) or O(1) is possible**.

The key innovations:
1. **LSH bucketing** for O(1) candidate generation (fixes SEER)
2. **Mini-graphs** for O(log b) navigation within buckets
3. **Inverted index** for sparse matching (fixes Hybrid)
4. **Post-hoc temporal reranking** (fixes LIM)
5. **Optional learned routing** (captures SEER/NEXUS ideas safely)

By layering simple, well-understood components, SWIFT should be:
- **Faster to build** than HNSW or NEXUS
- **Faster to query** than SEER, LIM, or Hybrid
- **Easier to understand** than NEXUS
- **More practical** than any single algorithm

---

*Proposal created: 2025-12-30*
*Status: Ready for implementation*
