# VORTEX Algorithm Analysis

> **VORTEX**: Voronoi-Optimized Routing for Traversal with Efficient indeXing  
> **Research Gap Addressed**: Gap 2B – Graph-based Cluster Routing  
> **Implementation Date**: 2026-01-20

---

## Executive Summary

VORTEX addresses the "Recall vs Speed" dilemma by combining IVF (clustering) with HNSW (graphs). Instead of using unreliable LSH or linear cluster scanning, it builds a navigable small-world graph on the cluster centroids themselves, enabling O(log C) routing to the most relevant buckets.

| Verdict | Rating |
|---------|--------|
| **Concept** | ⭐⭐⭐⭐⭐ |
| **Implementation Quality** | ⭐⭐⭐☆☆ |
| **Performance** | ⭐⭐⭐⭐☆ |
| **Scalability** | ⭐⭐⭐⭐⭐ |
| **Research Value** | ⭐⭐⭐⭐⭐ |

---

## Algorithm Overview

### Core Idea

Solve the scalability issues of both Flat-IVF (O(C) scan) and HNSW (O(log N) insert/build) by treating clusters as graph nodes:

```
Layer 1: Centroid Graph (HNSW)   ──► O(log C) Search
      [C1] ── [C2]
       │       │
      [C3] ── [C4]

Layer 2: Inverted Buckets        ──► O(bucket_size) Scan
      ▼       ▼
   ┌─────┐ ┌─────┐
   │Vectors│ │Vectors│
   └─────┘ └─────┘
```

### Components

| Component | Purpose |
|-----------|---------|
| `VortexConfig` | Configuration (num_clusters, n_probes, HNSW settings) |
| `KMeans` | Trains centroids to define semantic clusters |
| `CentroidGraph` | HNSW index storing only the centroids |
| `Buckets` | Inverted index storing vectors assigned to each centroid |
| `VortexIndex` | Main coordinator |

### Key Features

1.  **Graph-Based Routing**: Uses HNSW to find nearest centroids instead of linear scan.
2.  **Density-Aware Partitioning**: Uses K-Means to adapt to data distribution (unlike LSH).
3.  **Two-Stage Search**: Fast routing → Exhaustive bucket scan → Exact reranking.
4.  **Hybrid Complexity**: $O(\log C + \frac{N}{C} \cdot P)$ where $C$ is cluster count and $P$ is probes.

---

## Implementation Details

### Build Process

1.  **Training**: K-Means clustering runs on the dataset to identify $C$ centroids ($C \approx \sqrt{N}$).
2.  **Graph Construction**: The centroids are inserted into an HNSW graph. This graph is tiny ($C$ nodes) compared to a full HNSW ($N$ nodes).
3.  **Assignment**: Vectors are routed via the Centroid Graph to their nearest cluster and appended to the corresponding bucket.

### Search Process

```rust
// 1. Route: Find nearest `n_probes` centroids using HNSW
let centroid_results = self.centroid_index.search(query, self.config.n_probes)?;

// 2. Scan: Retrieve vectors from these buckets
for centroid_match in centroid_results {
    let bucket_idx = centroid_match.id;
    // ... accumulate candidates ...
}

// 3. Rerank: Compute exact distance and top-k
candidates.sort_by(...)
```

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_clusters` | 100 | Number of centroids/buckets |
| `n_probes` | 5 | Number of buckets to search |
| `hnsw_config` | Default | Config for the centroid graph |
| `max_kmeans_iters` | 20 | Training iterations |

---

## Strengths

### ✅ Scalability via Hierarchical Routing
- **Routing**: $O(\log C)$ is negligible (e.g., $\log 1000 \approx 10$ ops).
- **Scanning**: Only need to scan a small fraction of data.
- **Build Speed**: Much faster than HNSW because we only build the graph on centroids.

### ✅ Robust Recall
- K-Means adapts to data density, unlike random projections (SWIFT).
- Graph routing is more accurate than tree-based routing for high-dim data.

### ✅ Verified Functionality
- Passed `test_vortex_basic_flow` covering build, search, and exact match retrieval.

---

## Current Limitations

### 🟠 Training Overhead
- K-Means is $O(N \cdot C \cdot \text{iters})$. For very large $N$, this can be slow.
- **Mitigation**: Train on a subsample of $N$.

### 🟠 Memory Usage
- Stores vectors twice in current implementation (once in buckets, once in `vectors` map for reranking).
- **Fix**: Optimize storage to hold vectors only once.

---

## Comparison with Other Indexes

| Feature | LIM (IVF) | HNSW | VORTEX |
|---------|-----------|------|--------|
| **Routing** | Linear Scan $O(C)$ | Graph $O(\log N)$ | Graph $O(\log C)$ |
| **Build Time** | Fast | Slow | Medium/Fast |
| **Recall** | Good | Excellent | Good/Excellent |
| **Memory** | Low | High | Medium |

---

## Files

- [`crates/index-vortex/src/lib.rs`](file:///Users/bastianlipka/Desktop/index-lab/crates/index-vortex/src/lib.rs) — Main implementation
- [`crates/index-vortex/src/kmeans.rs`](file:///Users/bastianlipka/Desktop/index-lab/crates/index-vortex/src/kmeans.rs) — K-Means implementation

---

## Future Extensions

1.  **Subsampling for Training**: Speed up K-Means by training on 10% of data.
2.  **Parallel Build**: Assign vectors to buckets in parallel.
3.  **Product Quantization**: Compress vectors in buckets for reduced memory.

---

## Conclusion

VORTEX successfuly implements a "best of both worlds" approach. By keeping the navigation graph small (only on centroids) and using simple buckets for storage, it offers HNSW-like navigation speed with IVF-like build times and scalability.

---

*Status: ✅ Implemented and tested*
*Implementation: [`index-vortex`](file:///Users/bastianlipka/Desktop/index-lab/crates/index-vortex/)*
