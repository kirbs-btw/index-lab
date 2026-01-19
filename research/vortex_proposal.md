# VORTEX: Voronoi-Optimized Routing for Traversal with Efficient indeXing

> **Key Idea**: Solve the "Recall vs Speed" dilemma by replacing unreliable LSH/Learning with robust **Graph-based Cluster Routing**.

---

## The Problem with Current Solutions

Our valid "fixes" for the O(n) bottleneck have critical flaws:

| Algorithm | Method | Fatal Flaw |
|-----------|--------|------------|
| **SWIFT** | LSH + Buckets | **Recall (6%)**: Random hyperplanes don't capture real data manifolds. |
| **NEXUS** | Spectral + Graph | **Build Time**: O(n²) entropy estimation is too slow. |
| **SEER** | Learned Routing | **O(n) Scoring**: "Pruning" requires scoring every vector. |
| **LIM** | Flat Clustering | **O(C) Scan**: Scanning centroids scales poorly. |

**Theme**: We need O(log N) access to semantic neighborhoods *without* sacrificing recall.

---

## VORTEX Architecture

VORTEX combines **IVF (clustering)** with **HNSW (graphs)** in a novel way: treating the clusters themselves as nodes in a navigable graph.

```
┌─────────────────────────────────────────────────────────────┐
│                       VORTEX INDEX                          │
│                                                             │
│  Layer 1: Centroid Graph (HNSW)                             │
│       [C1] ─── [C2] ─── [C5]                                │
│        │        │                                           │
│       [C3] ─── [C4]         <-- O(log C) Search             │
│                                                             │
│  Layer 2: Inverted Buckets                                  │
│       ▼         ▼                                           │
│    ┌─────┐   ┌─────┐                                        │
│    │Vectors│   │Vectors│      <-- O(bucket_size) Scan       │
│    │ ...   │   │ ...   │                                    │
│    └─────┘   └─────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

### Components

1.  **Centroids**: Use K-Means (or hierarchical clustering) to define $C \approx \sqrt{N}$ semantic clusters.
2.  **Routing Graph**: Build a small HNSW graph on the $C$ centroids.
3.  **Inverted Index**: Store vectors in the clusters.

---

## Why VORTEX Wins

### 1. Fixes SEER's O(n)
Instead of scoring $N$ vectors to find candidates, we search the Centroid Graph.
*   Cost: $O(\log C)$ distance computations.
*   For 1M vectors, 1000 clusters: $\log(1000) \approx 10$ comps vs $1,000,000$ comps.

### 2. Fixes SWIFT's Low Recall
LSH partitions randomly. K-Means partitions by **density**.
*   Recall is robust because clusters naturally conform to the data distribution.
*   Multi-probing is efficient: Traverse the graph to find the nearest *k* centroids, then scan them.

### 3. Fixes LIM's Scalability
LIM scans all clusters linearly. VORTEX navigates them logarithmically.

### 4. Fixes HNSW's Build Cost
HNSW inserts are $O(\log N)$. VORTEX inserts are:
*   $O(\log C)$ to find nearest centroid.
*   $O(1)$ append to bucket.
*   Graph on $C$ is tiny, so maintenance is cheap.

---

## Proposed Implementation

### Config
```rust
struct VortexConfig {
    num_clusters: usize,   // e.g., sqrt(N)
    centroid_m: usize,     // HNSW M for centroid graph (e.g., 24)
    centroid_ef: usize,    // HNSW ef for routing (e.g., 64)
    n_probes: usize,       // Number of clusters to scan (e.g., 4)
}
```

### Build Process
1.  **Train**: Sample subset, run K-Means to find $C$ centroids.
2.  **Build Graph**: Insert centroids into HNSW graph.
3.  **Assign**: Stream vectors, find nearest centroid via graph, append to bucket.

### Search Process
1.  **Route**: Search centroid graph for `n_probes` nearest centroids.
2.  **Scan**: Retrieve vectors from these buckets.
3.  **Rerank**: Compute exact distances and return top-k.

---

## Comparison Prediction

| Metric | Hybrid (Current) | SWIFT (Current) | VORTEX (Predicted) |
|--------|------------------|-----------------|--------------------|
| **Recall** | 100% | 6% | **~95%** |
| **QPS** | ~1,200 | ~15,000 | **~10,000** |
| **Build Time** | Fast | Medium | **Fast** |
| **Complexity** | O(n) | O(log N) | **O(log C)** |

## Next Steps
1. Create `index-vortex` crate.
2. Implement `CentroidGraph` using `index-hnsw` components? Or a lightweight custom graph.
3. Implement `InvertedIndex` storage.
4. Benchmark against Hybrid and SWIFT.
