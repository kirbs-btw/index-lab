# Algorithm Research Findings Summary

A summary of findings from the novel algorithms implemented and tested in `index-lab`.

---

## Algorithms Implemented

| Algorithm | Crate | Type | Research Gap Addressed |
|-----------|-------|------|------------------------|
| **LIM** (Locality Index Method) | `index-lim` | Novel | Temporal vector indexing (Gap 1C) |
| **Hybrid Index** | `index-hybrid` | Novel | Sparse-Dense Fusion (Gap 2A, 2B) |
| **SEER** (Similarity Estimation via Efficient Routing) | `index-seer` | Novel | Learned Index Structures (Gap 3A) |
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
1. **Fix SEER performance** - Add LSH bucketing for O(1) candidate lookup (currently 25× slower than linear)
2. **Fix LIM scale mismatch** - Normalize spatial distances
3. **Add sparse inverted index** to Hybrid
4. **Large-scale benchmarks** - 100K-1M vectors

### Research Extensions
1. **Graph-accelerated Hybrid** - Combine HNSW with sparse lookup
2. **Improved SEER learning** - Proper metric learning or neural locality prediction
3. **Streaming support** - Handle continuous inserts without rebuilds

---

*Document updated: 2025-12-28*
*See also: [seer_analysis.md](./seer_analysis.md), [lim_analysis.md](./lim_analysis.md), [research_gaps.md](./research_gaps.md)*

