# Flaws of Current Vector Indexing Methods

This document catalogs all known flaws, weaknesses, and limitations across every indexing algorithm in the index-lab repository. It synthesizes findings from benchmark results, critical analyses, and the research summary.

---

## Structure

| Section | Algorithms |
|---------|------------|
| [1. Baseline](#1-baseline-and-reference-algorithms) | Linear, HNSW |
| [2. Compression](#2-compression-and-quantization) | IVF, PQ |
| [3. Novel / Research](#3-novel--research-algorithms) | SWIFT, PRISM, NEXUS, SEER, LIM, Hybrid, VORTEX, ARMI, FUSION |
| [4. Research-Generation](#4-research-generation-algorithms-apex--universal) | APEX, SYNTHESIS, CONVERGENCE, UNIVERSAL, ZENITH |
| [5. Cross-Cutting](#5-cross-cutting-flaws) | Implementation, parameters, scalability, data |
| [6. Summary Matrix](#6-summary-matrix) | Quick comparison table |

---

## 1. Baseline and Reference Algorithms

### 1.1 Linear

| Flaw | Severity | Description |
|------|----------|-------------|
| **O(n) search** | Critical | Every query scans all N vectors. At 50K points: 404 QPS vs ZENITH's 1,250 QPS. Does not scale. |
| **No structure** | Design | Pure brute force. No acceleration possible. Serves only as correctness baseline. |
| **Memory-bound** | Medium | Every query touches all vector data. Cache-unfriendly at scale. |

---

### 1.2 HNSW

| Flaw | Severity | Description |
|------|----------|-------------|
| **Catastrophically low recall** | Critical | 1.09% recall at 10K, 0.31% at 50K with default parameters. Extremely fast (33K QPS) but produces incorrect results. |
| **Parameter sensitivity** | Critical | Requires manual tuning of `m_max`, `ef_construction`, `ef_search`. Wrong defaults make it useless out of the box. |
| **No auto-tuning** | Medium | Algorithm is sound but needs expert knowledge to configure. |
| **Single entry point** | Medium | One entry point can be in a bad neighborhood; search may converge to wrong region. |

---

## 2. Compression and Quantization

### 2.1 IVF (Inverted File)

| Flaw | Severity | Description |
|------|----------|-------------|
| **Low recall** | Critical | 40.4% recall with default probe count. Needs more probes for acceptable quality. |
| **Fixed probe count** | Medium | `n_probes` is configurable but not adaptive to query difficulty. |
| **K-means limitations** | Medium | Centroids assume spherical clusters; fails on non-convex or high-dimensional manifolds. |
| **Training phase** | Medium | Must train centroids before use; not suitable for streaming inserts. |

---

### 2.2 PQ (Product Quantization)

| Flaw | Severity | Description |
|------|----------|-------------|
| **Low recall** | Critical | 33.7% recall. Compression inherently loses information. |
| **Very slow build** | Critical | 16.16s for 10K vectors vs 634µs for Linear. K-means codebook training is expensive. |
| **Slow search** | Critical | 748 QPS—slower than Linear (1,230 QPS). Compression adds overhead. |
| **Fixed subvector split** | Medium | Subvector boundaries are fixed; not adaptive to data distribution. |

---

## 3. Novel / Research Algorithms

### 3.1 SWIFT

| Flaw | Severity | Description |
|------|----------|-------------|
| **Very low recall** | Critical | 6.0% recall. Fundamental recall issue. |
| **Needs investigation** | High | Benchmarks indicate serious recall problems. Root cause undocumented. |

---

### 3.2 PRISM

| Flaw | Severity | Description |
|------|----------|-------------|
| **Nearly zero recall** | Critical | 0.8% recall at 32K QPS. Fast but useless. |
| **Session optimization mismatch** | Critical | HNSW underneath with session-specific optimizations; default parameters broken. |

---

### 3.3 NEXUS

| Flaw | Severity | Description |
|------|----------|-------------|
| **Low recall** | Critical | 14.6% recall. Spectral shortcuts not effective for this dataset. |
| **Long build time** | High | 8.39s for 10K vectors. Spectral methods are expensive. |
| **Complexity** | Medium | Multiple components; unclear which dominates performance. |

---

### 3.4 SEER

| Flaw | Severity | Description |
|------|----------|-------------|
| **11× slower than linear** | Critical | 110 QPS vs 1,230 Linear. Learns locality predictor but adds massive overhead. |
| **O(n) candidate filtering** | Critical | Still iterates over all vectors during search; learned filter does not reduce complexity. |
| **Training overhead** | Medium | Requires training phase; adds build latency. |
| **Fixed candidate threshold** | Medium | `candidate_threshold` and `min_candidates` fixed; not adaptive. |

---

### 3.5 LIM

| Flaw | Severity | Description |
|------|----------|-------------|
| **O(n_clusters) per insert** | Critical | Every insert checks distance to ALL clusters. No spatial index for centroids. |
| **Expensive cluster merging** | Critical | When cluster limit hit, reassigns ALL vectors of smallest cluster—O(n²) worst case. |
| **No hierarchical structure** | High | Flat clustering; search is O(n_probe × avg_cluster_size), not O(log n). |
| **Scale mismatch** | High | Spatial distances not normalized; temporal decay combined with raw spatial distances causes bias. |
| **Fixed cluster count** | Medium | `n_clusters` fixed; merging strategy is reactive, not proactive. |

---

### 3.6 Hybrid

| Flaw | Severity | Description |
|------|----------|-------------|
| **Fallback to linear** | Design | Achieves 100% recall by falling back to linear scan. No approximate acceleration when sparse matching fails. |
| **No sparse inverted index** | Medium | Research gaps note missing sparse inverted index for keyword matching. |
| **Single-modality dense** | Medium | Dense component is separate; no unified graph for hybrid vectors. |

---

### 3.7 VORTEX

| Flaw | Severity | Description |
|------|----------|-------------|
| **K-Means training overhead** | High | O(N × C × iters). For large N, training is slow. |
| **Memory usage** | Medium | Stores full vectors in buckets + centroid graph. No compression. |
| **n_probes fixed** | Medium | Default 5 probes; not adaptive to query or recall requirements. |
| **Bucket scan cost** | Medium | After routing, must scan all vectors in probed buckets. No mini-graph within buckets. |
| **Not in benchmark summary** | Low | No verified QPS/recall in algorithm_findings.md. |

---

### 3.8 ARMI

| Flaw | Severity | Description |
|------|----------|-------------|
| **O(n²) build time** | Critical | Each insertion computes distance to ALL existing nodes. No spatial index for neighbor finding. |
| **Performance needs optimization** | High | Analysis notes "Performance: ⭐⭐☆☆☆ (needs optimization)." |
| **Unused code** | High | 8 compiler warnings: unused `rng`, unused `update`/`estimate_recall`, unused `record_distance`/`record_traversal`, etc. |
| **Energy budget never read** | Medium | `operation_costs` field never read; precision scaling may not be applied. |
| **Complex architecture** | Medium | Multi-modal + shift detection + RL tuning + energy management. Many moving parts. |

---

### 3.9 FUSION

| Flaw | Severity | Description |
|------|----------|-------------|
| **Slower than linear at 10K** | High | 527–637 QPS vs 1,230–2,087 for Linear. LSH probing overhead dominates at small scale. |
| **Probing all buckets** | Medium | With 8 buckets and 8 probes, often probes all buckets—no early termination benefit. |
| **Fixed LSH parameters** | Medium | `n_hyperplanes`, `n_probes` fixed; not adaptive to data distribution. |
| **Mini-graph per bucket** | Medium | Each bucket builds its own NSW; memory overhead for many buckets. |
| **Adaptive probing complexity** | Low | `candidate_threshold_factor` can cause early termination; may reduce recall in edge cases. |

---

## 4. Research-Generation Algorithms (APEX → UNIVERSAL)

### 4.1 APEX (Generation 1)

| Flaw | Severity | Description |
|------|----------|-------------|
| **LSH not used** | Critical | LSH neighbor finder implemented but never called during build/insert. Dead code. |
| **Cross-modal graph empty** | Critical | Graph structure created but never populated or queried. Dead code. |
| **Adaptive features isolated** | Critical | Only available in `search_adaptive()`, not standard `search()`. Users get non-adaptive behavior. |
| **Dead code** | High | Significant portions of implementation unused. Wasted effort, no performance benefit. |
| **Never properly benchmarked** | High | Test failures; performance claims unvalidated. |

---

### 4.2 SYNTHESIS (Generation 2)

| Flaw | Severity | Description |
|------|----------|-------------|
| **Router training gaps** | Critical | Router only trained during `build()`, not during individual inserts. Stale routing over time. |
| **Fixed LSH parameters** | High | `lsh_hyperplanes`, `lsh_probes` fixed; not adaptive to data. |
| **Single-layer "hierarchical"** | High | Called hierarchical but only single-layer graphs. No O(log n) search. |
| **Fixed fusion weights** | High | Weights not learned adaptively. |
| **Empty bucket failures** | High | Search may fail if bucket is empty (e.g., after deletions). |
| **Excessive cross-modal edges** | Medium | Adds 10 edges per vector; no distance threshold or pruning. Memory and traversal overhead. |
| **Temporal decay only cross-modal** | Medium | Edges within buckets don't get temporal decay; inconsistent. |
| **Centroid graph underutilized** | Medium | Built but only used as fallback when learned router confidence is low. |
| **Incomplete serialization** | Medium | Does not rebuild full index state (graphs, buckets, router weights). |
| **Shift detection oversimplified** | Low | Statistical tests for distribution shift may be too simple. |

---

### 4.3 CONVERGENCE (Generation 3)

| Flaw | Severity | Description |
|------|----------|-------------|
| **20+ configuration parameters** | Critical | Extreme configuration burden. Requires expert tuning. |
| **High complexity** | Critical | Buckets + graphs + routers + LSH + ensemble. Many components to maintain. |
| **O(N log N) upfront build** | High | Slow initial build. No lazy construction. |
| **Memory overhead** | High | Multiple data structures; redundancy. |
| **36 compiler warnings** | High | Dead code, unused fields, unused methods. Evidence of incomplete integration. |
| **Serialization incomplete** | High | Basic only; full graph/cache/parameter serialization pending. |
| **Test failures** | High | Multiple basic tests fail (router init, RefCell borrowing). |
| **Unused variants** | Medium | `LshBased` strategy, `DegreeLimit`/`ImportanceBased` pruning never constructed. |
| **Unused methods** | Medium | `update_weights`, `train_batch`, `route`, `combine_results`, etc. |

---

### 4.4 UNIVERSAL (Generation 4)

| Flaw | Severity | Description |
|------|----------|-------------|
| **Never benchmarked** | Critical | Theoretical design only. No actual QPS/recall numbers. |
| **Cache coherence** | Critical | Cache updates require `&mut self`, but search is `&self`. Caching disabled in practice. |
| **Incomplete serialization** | High | Graph, caches, auto-tuner parameters not fully serialized. |
| **Over-engineered** | High | Router, Cache, AutoTuner, Graph—too many abstractions for unvalidated benefit. |
| **Lazy construction complexity** | Medium | Conditional logic for when to build edges. Unclear when optimization triggers. |
| **No multi-modal support** | Medium | Dense vectors only. Not "universal." |
| **Parameters set once** | Medium | Auto-tuning uses heuristics at init; no online learning from actual performance. |
| **Edge quality** | Medium | Lazy edges may not be optimal; no background optimization. |

---

### 4.5 ZENITH (Generation 5)

| Flaw | Severity | Description |
|------|----------|-------------|
| **Recall degrades at scale** | High | 94.82% at 10K drops to 81.31% at 50K. M and ef not scaled with dataset size. |
| **Slow build time** | High | 4.85s at 10K, 35.57s at 50K. Diversity heuristic adds O(M²) per insertion. |
| **M derived from dimension only** | Medium | Should scale with dataset size (N), not just dimension. |
| **Diversity heuristic cost** | Medium | Inter-neighbor distance checks during neighbor selection; adds build overhead. |
| **Single-threaded** | Medium | No parallelism in build or search. |
| **Fusion has better recall at 50K** | Medium | 93.30% vs 81.31%. LSH bucketing better for large-scale candidate generation. |

---

## 5. Cross-Cutting Flaws

### 5.1 Implementation Quality

| Flaw | Affected | Description |
|------|----------|-------------|
| **Dead code** | APEX, SYNTHESIS, CONVERGENCE | Features implemented but not used. Wasted effort. |
| **Compiler warnings** | APEX, ARMI, SYNTHESIS, CONVERGENCE | Unused variables, unused methods, unused fields. Indicates incomplete integration. |
| **Test failures** | SYNTHESIS, CONVERGENCE | Basic tests fail. Correctness not validated. |
| **Incomplete serialization** | SYNTHESIS, CONVERGENCE, UNIVERSAL | Cannot save/load full index state. Limits production use. |

---

### 5.2 Parameter and Configuration

| Flaw | Affected | Description |
|------|----------|-------------|
| **Manual tuning required** | HNSW, IVF, PQ, CONVERGENCE | Many parameters; optimal values dataset-dependent. |
| **Fixed parameters** | SYNTHESIS, FUSION, LIM | Parameters that should adapt (LSH, probes, clusters) are fixed. |
| **No auto-tuning** | Most algorithms | Only ZENITH derives parameters from data. |

---

### 5.3 Scalability

| Flaw | Affected | Description |
|------|----------|-------------|
| **O(n) search** | Linear, SEER | Does not scale. |
| **O(n) insert** | LIM (cluster search), ARMI | Per-insert cost grows with dataset. |
| **Recall degradation** | ZENITH, HNSW (with wrong params) | Quality drops as dataset grows. |
| **Not tested > 1M** | All | Large-scale behavior unknown. |

---

### 5.4 Data and Validation

| Flaw | Affected | Description |
|------|----------|-------------|
| **Synthetic data only** | All | No real-world dataset testing. Generalizability unknown. |
| **Single distribution** | All | Uniform random vectors. Clustered, skewed, or sparse data untested. |
| **No multi-modal benchmarks** | APEX, SYNTHESIS, CONVERGENCE | Dense/sparse/audio fusion not benchmarked. |

---

## 6. Summary Matrix

| Algorithm | Recall | QPS (10K) | Build Time | Config | Dead Code | Serialization | Tests |
|-----------|--------|-----------|------------|--------|-----------|---------------|-------|
| Linear | 100% | 2,087 | 516µs | None | None | Complete | Pass |
| HNSW | 1% | 33,970 | 223ms | 3–4 | None | Complete | Pass |
| IVF | 40% | 13,710 | 1.71s | Multiple | Unknown | Complete | Pass |
| PQ | 34% | 748 | 16.16s | Multiple | Unknown | Complete | Pass |
| LIM | 95% | 1,829 | 3.49s | Multiple | None | Complete | Pass |
| Hybrid | 100% | 1,256 | 557µs | Few | None | Complete | Pass |
| SEER | 96.5% | 110 | 2.4ms | Few | None | Complete | Pass |
| SWIFT | 6% | 15,884 | 73ms | Few | Unknown | Complete | Pass |
| PRISM | 0.8% | 32,389 | 222ms | Few | Unknown | Complete | Pass |
| NEXUS | 14.6% | 2,329 | 8.39s | Few | Unknown | Complete | Pass |
| FUSION | 94% | 637 | 382ms | 6+ | None | Complete | Pass |
| VORTEX | ? | ? | ? | 4+ | Unknown | Complete | Pass |
| ARMI | ? | ? | ? | Many | Some | Custom | Pass |
| ZENITH | 95% | 2,113 | 4.85s | 0 | None | Complete | Pass |
| APEX | ? | ? | ? | Many | Critical | Partial | Fail |
| SYNTHESIS | ? | ? | ? | Many | Some | Incomplete | Fail |
| CONVERGENCE | ? | ? | ? | 20+ | Significant | Incomplete | Fail |
| UNIVERSAL | ? | ? | ? | 5+ | Some | Incomplete | Unknown |

---

*Last updated: 2026-02-08. Sources: algorithm_findings.md, research_summary.md, synthesis_critical_analysis.md, convergence_analysis.md, universal_analysis.md, zenith_analysis.md, lim_analysis.md, seer_analysis.md, fusion_analysis.md, research_gaps.md.*
