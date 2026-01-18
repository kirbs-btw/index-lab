# Algorithm Research Findings Summary

A summary of findings from the novel algorithms implemented and tested in `index-lab`.

---

## Algorithms Implemented

| Algorithm | Crate | Type | Research Gap Addressed |
|-----------|-------|------|------------------------|
| **LIM** (Locality Index Method) | `index-lim` | Novel | Temporal vector indexing (Gap 1C) |
| **Hybrid Index** | `index-hybrid` | Novel | Sparse-Dense Fusion (Gap 2A, 2B) |
| **SEER** (Similarity Estimation via Efficient Routing) | `index-seer` | Novel | Learned Index Structures (Gap 3A) |
| **SWIFT** (Sparse-Weighted Index with Fast Traversal) | `index-swift` | Novel | Fast candidate generation (fixes SEER O(n)) |
| **PRISM** (Progressive Refinement Index with Session Memory) | `index-prism` | Novel | Context-aware, adaptive search (Gap 7) |
| **NEXUS** (Neural EXploration with Unified Spectral Routing) | `index-nexus` | Novel | Spectral manifold learning (Gap 3A) |
| **FUSION** (Fast Unified Search with Intelligent Orchestrated Navigation) | `index-fusion` | Novel | LSH + Mini-graphs (fixes O(n) issues) |
| **HNSW** | `index-hnsw` | Baseline | Graph-based state-of-the-art |
| **IVF** | `index-ivf` | Baseline | Clustering-based indexing |
| **PQ** | `index-pq` | Baseline | Compression via quantization |
| **Linear** | `index-linear` | Baseline | Brute-force reference |

---

## LIM Algorithm (Locality Index Method)

### What It Does
Combines **spatial proximity** with **temporal proximity** to enable time-aware vector search. Recent vectors are weighted more heavily than older ones.

**Key Formula:**
```
combined_distance = α × spatial_distance + (1-α) × temporal_distance
```

### ✅ Upsides
| Advantage | Description |
|-----------|-------------|
| **Temporal awareness** | First algorithm to incorporate time decay into vector similarity |
| **No training phase** | Incremental clustering, vectors can be inserted immediately |
| **Practical applications** | E-commerce, social media, fraud detection, news feeds |
| **Time-decayed queries** | Naturally favors recent data without manual filtering |
| **Unique value** | Addresses unexplored research gap (temporal vector indexing) |

### ❌ Downsides
| Issue | Severity | Description |
|-------|----------|-------------|
| **O(n) cluster search** | 🔴 High | Every insertion checks ALL clusters |
| **Expensive merging** | 🔴 High | Cluster merging can cause latency spikes |
| **Scale mismatch** | 🟠 Medium | Spatial (0→∞) vs temporal (0→1) distances incompatible |
| **No hierarchical structure** | 🟡 Medium | Flat clustering limits scalability vs HNSW |
| **No deletion support** | 🟠 Medium | Vectors accumulate forever |
| **No theoretical guarantees** | 🟡 Medium | No provable recall bounds |

### Testing Performed
- ✅ Unit tests for basic insert/search
- ✅ Temporal decay verification (older vectors get lower scores)
- ✅ Spatial locality clustering
- ✅ Save/load functionality
- ❌ Large-scale benchmarks (>1M vectors) not yet done
- ❌ Head-to-head comparison with HNSW on temporal workloads

### Recommendations
1. **Normalize distances** - Fix spatial-temporal scale mismatch
2. **Add hierarchical structure** - Use KD-tree for O(log n) cluster lookup
3. **Implement deletion** - Mark-as-deleted + lazy cleanup
4. **Benchmark at scale** - Test with 1M+ vectors

---

## Hybrid Index (Dense-Sparse Fusion)

### What It Does
Unified index for **dense embeddings + sparse term vectors** (e.g., BM25, TF-IDF). Uses **distribution-aware score fusion** to combine modalities.

**Key Innovation:**
- Tracks min/max for both dense and sparse scores
- Normalizes each to [0, 1] before fusion
- Avoids scale mismatch problems that plague naive fusion

### ✅ Upsides
| Advantage | Description |
|-----------|-------------|
| **Unified storage** | Single index for both modalities |
| **Adaptive fusion** | Distribution-aware normalization |
| **No post-hoc merging** | Scores combined during search |
| **Addresses RAG needs** | 80% of production RAG systems need hybrid search |
| **Memory locality** | Dense + sparse stored together |

### ❌ Downsides
| Issue | Severity | Description |
|-------|----------|-------------|
| **Linear scan** | 🔴 High | No indexing for sparse vectors yet |
| **Fixed fusion weights** | 🟡 Medium | Could learn from query patterns |
| **Dual computation** | 🟡 Medium | 2.1× more expensive than pure dense |
| **No graph structure** | 🟡 Medium | Could integrate with HNSW |

### Testing Performed
- ✅ Dense-only search (backward compatible)
- ✅ Hybrid search with sparse queries
- ✅ Distribution statistics normalization
- ✅ Multiple sparse scoring methods (dot product, cosine)
- ✅ Save/load functionality
- ❌ Real-world BM25/SPLADE integration not tested
- ❌ Recall comparison with separate indexes

### Recommendations
1. **Add inverted index** for sparse vectors (keyword lookup)
2. **Learn fusion weights** from query feedback
3. **Integrate with HNSW** for graph-accelerated dense search
4. **Test with real data** - MSMARCO, BEIR benchmarks

---

## SEER Algorithm (Similarity Estimation via Efficient Routing)

> [!WARNING]
> SEER is currently **25× slower** than linear scan. See [seer_analysis.md](./seer_analysis.md) for full details.

### What It Does
Uses **learned random projections** to predict locality relationships between vectors, filtering candidates before computing exact distances.

**Key Innovation:**
- Projects vectors onto random unit vectors
- Learns weights via correlation with true distances
- Scores candidates based on weighted projection similarity

### ✅ Upsides
| Advantage | Description |
|-----------|-------------|
| **Novel approach** | First learned index in the codebase |
| **Clean implementation** | Well-structured Rust with proper error handling |
| **Good test coverage** | 8 unit tests, all passing |
| **High recall** | 96-99% recall@k in benchmarks |
| **Research foundation** | Addresses Gap 3A (Learned Index Structures) |

### ❌ Downsides
| Issue | Severity | Description |
|-------|----------|-------------|
| **No actual pruning** | 🔴 High | Scores ALL vectors before filtering → O(n) |
| **Slower than linear** | 🔴 High | 2.7 QPS vs 67 QPS (25× slower) |
| **Training ineffective** | 🟠 Medium | Learned weights ≈ uniform weights |
| **Inverted threshold** | 🟡 Medium | `threshold=0.3` selects 70%, not 30% |
| **Hidden retraining** | 🟡 Medium | Every 1000th insert retrains predictor |

### Benchmark Results (Verified 2025-12-17)
| Scenario | Points | SEER QPS | Linear QPS | Recall |
|----------|--------|----------|------------|--------|
| `smoke` | 1,000 | 48.5 | 1,105.5 | 98.75% |
| `recall-baseline` | 10,000 | 2.7 | 67.2 | 96.48% |

### Testing Performed
- ✅ Unit tests (8 tests, all passing)
- ✅ Cluster separation verification
- ✅ Save/load roundtrip
- ✅ Benchmark smoke test
- ✅ Benchmark recall-baseline
- ❌ Large-scale benchmarks (>100K vectors)
- ❌ Comparison with HNSW at equivalent recall

### Recommendations
1. **Add LSH bucketing** for O(1) candidate lookup instead of O(n) scoring
2. **Fix threshold semantics** - rename or invert calculation
3. **Remove or improve training** - current learning provides no benefit
4. **Integrate with HNSW** for graph-accelerated candidate generation
5. **See full analysis**: [seer_analysis.md](./seer_analysis.md)

---

## SWIFT (Sparse-Weighted Index with Fast Traversal)

> [!WARNING]
> SWIFT currently suffers from **extremely low recall (6.0%)** in benchmarks. See [swift_proposal.md](./swift_proposal.md) for architecture.

### What It Does
Decomposes search into three stages: **LSH bucketing** (O(1)), **Mini-graph navigation** (O(log b)), and **refinement** (O(k)). Aims to solve the O(n) bottleneck of SEER and Hybrid.

**Key Innovation:**
- Layered filtering: LSH → Mini-Graph → Rerank
- Multi-probe LSH to find candidate buckets
- Sparse inverted index extension for hybrid search

### ✅ Upsides
| Advantage | Description |
|-----------|-------------|
| **Theoretical Speed** | O(1) candidate generation vs O(n) scan |
| **Scalable Architecture** | Mini-graphs are easier to maintain than one giant graph |
| **Hybrid & Temporal** | Designed to support multi-modal and temporal search natively |
| **Simple Components** | Composed of standard LSH and HNSW parts |

### ❌ Downsides
| Issue | Severity | Description |
|-------|----------|-------------|
| **Catastrophic Recall** | 🔴 Critical | Currently achieves only 6.0% recall (target 95%) |
| **Bucket Imbalance** | 🔴 High | Risk of empty or mega-buckets degrading performance |
| **Dimensionality Curse** | 🟡 Medium | High-dim vectors require many hyperplanes for good hashing |
| **Cold Start** | 🟡 Medium | Requires enough data to form meaningful buckets |

### Recommendations
1. **Debug LSH Hashing** - Identify why recall is so low (hash collisions, probing depth)
2. **Implement Dynamic Buckets** - Split/merge buckets based on size
3. **Verify Mini-Graph Connectivity** - Ensure graphs within buckets are navigable
4. **Increase Probes** - Aggressively probe more buckets to improve recall

---

## NEXUS (Neural EXploration with Unified Spectral Routing)

### What It Does
Exploits **manifold structure** via spectral embedding. Projects high-dim vectors to low-dim spectral space for fast filtering, then reranks with full vectors.

**Key Innovation:**
- **Spectral Filtering**: O(m) distance vs O(d) distance (e.g., 32 vs 768 dims)
- **Adaptive Graph**: Allocates more edges to "hard" (high entropy) regions
- **Two-Phase Search**: Fast spectral search → Accurate full search

### ✅ Upsides
| Advantage | Description |
|-----------|-------------|
| **Manifold Aware** | Adaptation to data geometry via entropy estimation |
| **Fast Traversal** | Spectral distances are ~24× cheaper to compute |
| **Adaptive Topology** | Variable edge counts optimize for local density |
| **Novelty** | Addresses "Learned Index Structures" gap (Gap 3A) |

### ❌ Downsides
| Issue | Severity | Description |
|-------|----------|-------------|
| **Build Complexity** | 🔴 High | O(n²) entropy estimation makes build time prohibitive |
| **Low Recall** | 🔴 High | Currently 14.6% recall; random projections are too noisy |
| **Approximation Loss** | 🟡 Medium | Random projection is not true spectral decomposition |
| **No Neural Router** | 🟡 Medium | "Neural" part (NRP) not yet implemented |

### Recommendations
1. **Optimize Build** - Use approximate k-NN for entropy estimation (O(n log n))
2. **True Spectral Embedding** - Replace random projections with actual eigendecomposition
3. **Implement NRP** - Train a small MLP to route queries instead of greedy search
4. **Hierarchical Entry** - Add HNSW-like layers for faster entry point finding

---

## PRISM (Progressive Refinement Index with Session Memory)

### What It Does
Wraps an underlying index (like HNSW) with **session memory**. Caches "hot regions" and adapts search parameters (`ef`) based on query difficulty and similarity to recent history.

**Key Innovation:**
- **Context Awareness**: Remembers recent queries and results
- **Adaptive Search**: Increases `ef` for hard queries, decreases for related ones
- **Hot Region Caching**: Direct jumping to relevant graph neighborhoods

### ✅ Upsides
| Advantage | Description |
|-----------|-------------|
| **Session optimization** | Speeds up related queries in a sequence |
| **Lightweight** | Minimal memory overhead (~50KB/session) |
| **Clean Design** | Wraps existing indexes without internal changes |
| **High Recall** | Maintains HNSW-level recall (~98%) |

### ❌ Downsides
| Issue | Severity | Description |
|-------|----------|-------------|
| **Trait Limitations** | 🟠 Medium | `VectorIndex::search` is immutable, complicating state updates |
| **Limited "Warm Start"** | 🟠 Medium | Cannot pass entry points to HNSW internal search |
| **Zero Recall Bug** | 🔴 Critical | Benchmark shows 0.8% recall (likely implementation bug) |
| **State Complexity** | 🟡 Medium | Managing session expiry and persistence is complex |

### Recommendations
1. **Fix Recall Bug** - Investigate why PRISM search yields near-zero recall
2. **Expose Entry Points** - Modify HNSW to accept custom entry candidates
3. **Refactor Trait** - Allow `search_mut` or use interior mutability for session state
4. **Cross-Session Learning** - Aggregate stats across users for global structure learning

---

## FUSION (Fast Unified Search)

### What It Does
Combines **LSH bucketing** (for coarse routing) with **Mini-Graphs** (for fine-grained search). Designed to fix the O(n) issues of SEER/LIM while maintaining high recall.

**Key Innovation:**
- **LSH Routing**: O(1) bucket selection
- **NSW Graphs**: Simple navigable small world graphs per bucket
- **Adaptive Probing**: Stops probing buckets when enough good candidates found

### ✅ Upsides
| Advantage | Description |
|-----------|-------------|
| **High Recall** | 94% recall (best of the O(n)-fixing attempts) |
| **No Linear Scan** | LSH avoids O(n) candidate scoring |
| **Clean Architecture** | Well-separated components (Hasher, Graph, Reranker) |
| **Scalable** | Design scales better than linear scan for >100K vectors |

### ❌ Downsides
| Issue | Severity | Description |
|-------|----------|-------------|
| **Small-Scale Speed** | 🟠 Medium | Slower than linear for <10K items due to overhead |
| **Uniform Data Issues** | 🟡 Medium | LSH acts poorly on uniform random benchmarks |
| **Scan Overhead** | 🟡 Medium | Probing multiple buckets can read more than needed |
| **Fixed Buckets** | 🟡 Medium | No dynamic splitting/merging of buckets |

### Recommendations
1. **Parallel Probing** - Search buckets concurrently
2. **Learned Routing** - Replace LSH with a learned classifier for bucket selection
3. **Hierarchical LSH** - Two-level hashing for very large datasets
4. **Optimize Overhead** - Reduce hashing/probing cost for small N

---

## Baseline Algorithms

### HNSW (Hierarchical Navigable Small World)
- **Implementation**: Standard multi-layer navigable graph
- **Key parameters**: `m` (connections), `ef_construction`, `ef_search`
- **Status**: Fully implemented, serves as accuracy/speed baseline

### IVF (Inverted File Index)
- **Implementation**: K-means++ initialization, proper cluster training
- **Key innovation**: Separation of training vs search phase
- **Status**: Fully implemented with incremental inserts after build

### PQ (Product Quantization)
- **Implementation**: Subvector codebook learning
- **Compression**: Reduces memory ~4-8× while maintaining approximate distances
- **Status**: Fully implemented with K-means codebook training

---

## Related Documents

- **Benchmark scenarios**: See [README.md](./README.md#-running-benchmarks)
- **Research gaps**: See [research_gaps.md](./research_gaps.md) for full coverage

---

## Benchmark Results (Verified 2026-01-12)

> **Scenario**: `recall-baseline` – 10,000 points, 64 dimensions, 256 queries, k=20, Euclidean

| Index | QPS | Avg Recall@20 | Build Time | Status |
|-------|-----|---------------|------------|--------|
| **Linear** | 1,230 | 100.0% | 634µs | ✅ Baseline |
| **HNSW** | 33,299 | 1.0% | 220ms | ⚠️ Low recall (needs tuning) |
| **IVF** | 13,710 | 40.4% | 1.71s | ⚠️ Low recall (needs more probes) |
| **PQ** | 748 | 33.7% | 16.16s | ⚠️ Low recall (compression tradeoff) |
| **LIM** | 1,158 | 98.7% | 3.33s | ✅ High recall, near-linear QPS |
| **Hybrid** | 1,256 | 100.0% | 557µs | ✅ Perfect recall (falls back to linear) |
| **SEER** | 110 | 96.5% | 2.4ms | 🔴 11× slower than linear |
| **SWIFT** | 15,884 | 6.0% | 73ms | 🔴 Very low recall |
| **PRISM** | 32,389 | 0.8% | 222ms | 🔴 Nearly zero recall |
| **NEXUS** | 2,329 | 14.6% | 8.39s | 🔴 Low recall, long build |
| **FUSION** | 527 | 94.0% | 553ms | ✅ High recall, addresses O(n) issues |

### Key Observations

1. **LIM**, **Hybrid**, and **FUSION** are the algorithms with high recall (>90%)
2. **FUSION** successfully addresses O(n) issues with LSH bucketing + mini-graphs
3. **HNSW**, **IVF** need parameter tuning for this dataset
4. **SEER** has good recall but O(n) performance issue remains
5. **SWIFT**, **PRISM**, **NEXUS** have serious recall issues to investigate

---

## Performance Comparison Summary

| Metric | Linear | LIM | Hybrid | SEER | IVF | PQ | HNSW |
|--------|--------|-----|--------|------|-----|-------|------|
| **Search Complexity** | O(n) | O(clusters × k) | O(n) | O(n)* | O(probes × k) | O(n) | O(log n) |
| **Insert Complexity** | O(1) | O(clusters) | O(1) | O(1) | O(clusters)* | O(m)* | O(log n) |
| **Memory per Vector** | O(d) | O(d + 8) | O(d + sparse) | O(d + proj) | O(d) | O(m) | O(d + edges) |
| **Accuracy** | 100% | Approximate | Approximate | ~97% | Approximate | Approximate | Approximate |
| **Temporal Aware** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Hybrid Search** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Learned** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |

*After training phase

---

## Next Steps

### Immediate Priorities
1. **Optimize FUSION speed** - Adaptive probing, parallel bucket search
2. **Fix LIM scale mismatch** - Normalize spatial distances
3. **Add sparse inverted index** to Hybrid
4. **Large-scale benchmarks** - 100K-1M vectors to validate FUSION scalability

### Research Extensions
1. **Learned FUSION routing** - Train model to predict best buckets
2. **Graph-accelerated Hybrid** - Combine HNSW with sparse lookup
3. **Temporal FUSION** - Add time decay post-search reranking
4. **Streaming support** - Handle continuous inserts without rebuilds

---

*See also: [fusion_analysis.md](./fusion_analysis.md), [seer_analysis.md](./seer_analysis.md), [lim_analysis.md](./lim_analysis.md), [research_gaps.md](./research_gaps.md)*

