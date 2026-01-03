# NEXUS: Neural EXploration with Unified Spectral Routing

> A novel vector index algorithm designed to beat HNSW by exploiting manifold structure and learned navigation

---

## The Core Insight

**HNSW's fundamental weakness**: It treats high-dimensional space as uniformly difficult to navigate. Every query uses the same greedy traversal strategy, regardless of:
- Local data density
- Underlying manifold structure  
- Query difficulty (easy queries near cluster centers vs. hard queries in sparse regions)

**Our thesis**: Real-world embeddings lie on low-dimensional manifolds embedded in high-dimensional space. By *learning* this manifold structure and using *spectral shortcuts*, we can outperform graph-only navigation.

---

## Algorithm Overview: NEXUS

```
NEXUS = Spectral Manifold Learning + Neural Route Predictor + Adaptive Graph
```

### Three Key Innovations

| Innovation | What HNSW Does | What NEXUS Does |
|------------|----------------|-----------------|
| **Structure Discovery** | Ignores manifold | Learns spectral embedding of k-NN graph |
| **Navigation Strategy** | Fixed greedy search | Neural network predicts per-query strategy |
| **Graph Topology** | Uniform edge budget | Entropy-adaptive edges (more in hard regions) |

---

## Phase 1: Spectral Manifold Embedding

### The Idea

Build the k-NN graph, then compute its **Laplacian eigenvectors**. These eigenvectors encode the "natural coordinates" of the data manifold.

```
1. Build k-NN graph G = (V, E) with edge weights w_ij = exp(-||v_i - v_j||²)
2. Compute graph Laplacian: L = D - W  (D = degree matrix, W = weight matrix)
3. Find top-m eigenvectors of L: φ₁, φ₂, ..., φₘ
4. Project each vector: v_i → (φ₁(i), φ₂(i), ..., φₘ(i))
```

### Why This Helps

The spectral embedding has **dimensionality m << d** (typically m=16-64 vs d=768-1536). 

Key insight: **Points close in spectral space are close along the manifold**, even if their Euclidean distance is large due to manifold curvature.

```
     Original Space (d=768)           Spectral Space (m=32)
     
    A -------- long path -------- B      A ---- short ---- B
         (geodesic distance)              (spectral distance)
```

### Spectral Shortcuts

During search, we can compute `spectral_dist(query, candidate)` in O(m) instead of O(d), then only compute full distance for promising candidates.

---

## Phase 2: Neural Route Predictor (NRP)

### The Problem with Greedy Search

HNSW always picks the neighbor closest to the query. This fails when:
- The closest neighbor leads to a local minimum
- The query is in a "saddle point" region between clusters
- Multiple paths have similar quality but different endpoints

### The Solution: Learn to Navigate

Train a **tiny neural network** (1-2 layers, ~10K parameters) to predict:

```
Input:  [query_spectral, current_node_spectral, neighbor_spectral_distances]
Output: [probability distribution over neighbors]
```

The NRP learns patterns like:
- "When at a cluster boundary, prefer outward-facing edges"
- "When in a dense region, take larger steps"
- "When approaching the target, switch to fine-grained search"

### Training the NRP

```python
# Generate training data from successful searches
for query in training_queries:
    optimal_path = dijkstra(graph, entry_point, nearest_neighbor(query))
    for (current, next_hop) in optimal_path:
        training_data.append({
            'features': extract_spectral_features(query, current, neighbors(current)),
            'label': next_hop
        })

# Train lightweight MLP
nrp = MLP(input=3*m, hidden=64, output=max_neighbors)
nrp.train(training_data, epochs=10)
```

### Why This is Different from LIDER/SEER

| Approach | What It Learns | Limitation |
|----------|----------------|------------|
| LIDER | Cluster assignments | Static, doesn't adapt to query |
| SEER | Distance correlations | Still scores ALL vectors |
| **NEXUS NRP** | Navigation decisions | Per-step routing, O(log n) |

---

## Phase 3: Entropy-Adaptive Graph

### The Problem with Uniform Edge Counts

HNSW uses the same number of edges (M=16 typically) everywhere. But:
- Dense clusters need **fewer** edges (neighbors are easy to find)
- Sparse regions need **more** edges (must reach distant points)
- Boundary regions need **directional** edges (avoid local minima)

### Local Entropy Estimation

```
entropy(node) = -Σ p(neighbor) * log(p(neighbor))
where p(neighbor) = distance_to_neighbor / Σ distances
```

- **Low entropy**: Neighbors at uniform distances → dense region → fewer edges needed
- **High entropy**: Neighbors at varied distances → sparse/boundary → more edges needed

### Adaptive Edge Budget

```rust
fn compute_edge_count(node: &Node, base_m: usize) -> usize {
    let entropy = local_entropy(node);
    let density = local_density(node);
    
    // More edges in high-entropy, low-density regions
    let multiplier = (entropy / mean_entropy) * (mean_density / density);
    
    (base_m as f32 * multiplier.clamp(0.5, 2.0)) as usize
}
```

---

## Search Algorithm: NEXUS Query

```python
def nexus_search(query, k, index):
    # 1. Project query to spectral space
    q_spectral = spectral_project(query)
    
    # 2. Find entry point using spectral distance (fast, O(m))
    entry = find_spectral_nearest(q_spectral, index.level_0_nodes)
    
    # 3. Navigate using NRP
    visited = set()
    candidates = MinHeap()  # by spectral distance
    candidates.push((spectral_dist(q_spectral, entry), entry))
    
    while len(visited) < ef_search:
        _, current = candidates.pop()
        if current in visited:
            continue
        visited.add(current)
        
        # If spectral distance is promising, compute full distance
        if should_compute_full(q_spectral, current):
            full_dist = euclidean_dist(query, index.vectors[current])
            candidates.push((full_dist, current))
        
        # Use NRP to select promising neighbors
        neighbor_probs = nrp.predict(q_spectral, current.spectral, current.neighbors)
        for neighbor, prob in zip(current.neighbors, neighbor_probs):
            if prob > threshold and neighbor not in visited:
                candidates.push((spectral_dist(q_spectral, neighbor), neighbor))
    
    # 4. Rerank top candidates by full distance
    return rerank_by_full_distance(query, candidates.top(k * 2))[:k]
```

---

## Theoretical Analysis

### Time Complexity

| Phase | HNSW | NEXUS |
|-------|------|-------|
| **Entry point** | O(log n) | O(n/b) spectral scan of bucket |
| **Per step** | O(M × d) distances | O(M × m) spectral + NRP inference |
| **Total search** | O(log n × M × d) | O(log n × M × m + k × d) |

When m << d (e.g., m=32, d=768): **NEXUS is ~20× faster per step**.

The k × d term at the end is for reranking - computing full distances only for final candidates.

### Memory Overhead

| Component | Size |
|-----------|------|
| Spectral embedding | n × m floats |
| NRP model | ~10K parameters |
| Adaptive graph | ~n × avg_M edges |

Total: ~**1.3× HNSW memory** for m=32.

---

## Why NEXUS Could Beat HNSW

### 1. Exploits Hidden Structure
HNSW is "structure-blind". NEXUS discovers and exploits the manifold.

### 2. Adaptive Navigation
HNSW uses one strategy for all queries. NEXUS adapts per-query:
- Easy queries (near cluster centers) → aggressive pruning, fewer steps
- Hard queries (boundaries, outliers) → careful exploration, more candidates

### 3. Cheaper Per-Step Computation
m-dimensional spectral distance vs d-dimensional Euclidean:
- 32 FLOPs vs 768 FLOPs per distance computation
- NRP inference: ~100 FLOPs per step

### 4. Better Graph Topology
Entropy-adaptive edges mean:
- Faster navigation in easy regions
- Better coverage in hard regions

---

## Potential Weaknesses & Mitigations

| Weakness | Mitigation |
|----------|------------|
| **Spectral computation is expensive** | One-time cost during index build; can use approximate methods (Nyström, random projections) |
| **NRP training requires labeled data** | Generate synthetic paths via exhaustive search on subset |
| **Manifold assumption may not hold** | Fallback to HNSW-style greedy when spectral distance unreliable |
| **Memory overhead** | Spectral vectors can be quantized to int8 |

---

## Implementation Roadmap

### Phase 1: Proof of Concept (2-3 weeks)
- [ ] Implement spectral embedding using eigendecomposition
- [ ] Build basic graph with uniform edges
- [ ] Measure spectral distance correlation with true distance

### Phase 2: Neural Navigator (2-3 weeks)
- [ ] Generate training data from optimal paths
- [ ] Train NRP model (PyTorch → ONNX → Rust inference)
- [ ] Integrate NRP into search loop

### Phase 3: Adaptive Graph (1-2 weeks)
- [ ] Implement local entropy estimation
- [ ] Modify graph construction for adaptive edge counts
- [ ] Tune hyperparameters

### Phase 4: Benchmarking (1 week)
- [ ] Compare vs HNSW on standard datasets (SIFT1M, GIST1M, Deep1M)
- [ ] Measure recall@k, QPS, memory usage
- [ ] Profile where time is spent

---

## Related Work & Novelty

| Approach | Key Idea | NEXUS Difference |
|----------|----------|------------------|
| **HNSW** | Hierarchical graph | No hierarchy; spectral shortcuts instead |
| **DiskANN** | SSD-optimized graph | Focus on latency, not I/O |
| **LIDER** | Learned cluster prediction | Learns navigation, not clustering |
| **Spectral Hashing** | LSH via eigenvectors | Full graph + neural router, not just hashing |
| **Graph Neural Networks** | Learn on graphs | Tiny predictor, not full GNN |

### The Novel Combination
NEXUS is the first to combine:
1. Spectral manifold discovery
2. Per-step neural routing
3. Entropy-adaptive graph topology

This three-way integration doesn't exist in the literature.

---

## Appendix: Quick Pseudocode

```rust
struct NexusIndex {
    vectors: Vec<Vector>,           // Original d-dimensional vectors
    spectral: Vec<SpectralVec>,     // m-dimensional spectral projections
    graph: AdaptiveGraph,           // Entropy-adaptive edges
    nrp: NeuralRoutePredictor,      // Tiny MLP for navigation
    eigenvectors: Matrix,           // Φ matrix for query projection
}

impl NexusIndex {
    fn build(vectors: Vec<Vector>) -> Self {
        // 1. Build k-NN graph
        let knn = build_knn_graph(&vectors, k=64);
        
        // 2. Compute spectral embedding
        let laplacian = compute_laplacian(&knn);
        let (eigenvalues, eigenvectors) = eig(laplacian, m=32);
        let spectral: Vec<_> = (0..vectors.len())
            .map(|i| project_to_spectral(i, &eigenvectors))
            .collect();
        
        // 3. Build entropy-adaptive graph
        let graph = build_adaptive_graph(&vectors, &spectral, base_m=16);
        
        // 4. Train NRP on synthetic paths
        let nrp = train_nrp(&vectors, &spectral, &graph);
        
        Self { vectors, spectral, graph, nrp, eigenvectors }
    }
    
    fn search(&self, query: &Vector, k: usize) -> Vec<usize> {
        let q_spectral = self.project_query(query);
        let entry = self.find_entry(&q_spectral);
        
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        candidates.push((spectral_dist(&q_spectral, &self.spectral[entry]), entry));
        
        while candidates.len() < ef_search {
            let (_, current) = candidates.pop().unwrap();
            if visited.contains(&current) { continue; }
            visited.insert(current);
            
            // NRP predicts which neighbors to explore
            let probs = self.nrp.predict(&q_spectral, current, &self.graph.neighbors(current));
            for (neighbor, prob) in self.graph.neighbors(current).zip(probs) {
                if prob > 0.1 && !visited.contains(&neighbor) {
                    let s_dist = spectral_dist(&q_spectral, &self.spectral[neighbor]);
                    candidates.push((s_dist, neighbor));
                }
            }
        }
        
        // Rerank by full distance
        self.rerank(query, candidates.into_vec(), k)
    }
}
```

---

## Conclusion

NEXUS represents a paradigm shift from "graph traversal" to "manifold navigation". By learning the hidden structure of the embedding space and training a neural guide, we can potentially achieve:

- **2-5× faster search** via cheap spectral distances
- **Better recall** via adaptive exploration
- **Robustness** via learned navigation strategies

The key bet: **The manifold is real, and learning to navigate it beats blindly walking the graph.**

---

*Status: Conceptual - awaiting implementation*
