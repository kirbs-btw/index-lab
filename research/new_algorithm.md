# Index Algorithms

## index-lim (Locality Index Method)

### Novel Research Directions to Explore

#### 1. **Adaptive Locality-Aware Indexing with Temporal Decay**
**Research Gap Addressed**: Dynamic Indexing and Streaming Vector Workloads (Gap 1C)

**Novel Idea**: Design a locality-based index structure that incorporates **temporal decay** into similarity calculations. Unlike static indexes, LIM could:
- Weight vector similarities by recency (recent vectors have higher influence)
- Maintain locality clusters that adapt over time without full reconstruction
- Support time-decayed queries: "find similar vectors from the last N days"

**Key Innovation**: Instead of treating all vectors equally, LIM could use a **locality function** that combines spatial proximity with temporal proximity, enabling queries like "find vectors similar to X that were inserted recently."

**Research Questions**:
- How to balance spatial vs. temporal locality in edge construction?
- What decay functions (exponential, linear, step) work best for different workloads?
- Can we maintain logarithmic search complexity with temporal-aware structures?

---

#### 2. **Multi-Modal Locality Fusion**
**Research Gap Addressed**: Hybrid Retrieval (Gap 2A, 2B)

**Novel Idea**: Create a unified index structure that natively handles **dense, sparse, and hybrid vectors** within a single locality-based framework. LIM could:
- Maintain separate locality clusters for different vector modalities
- Use learned fusion functions to combine dense and sparse distances
- Enable joint optimization of multi-modal queries without post-hoc merging

**Key Innovation**: A **locality-aware fusion layer** that learns optimal combination weights for different vector types based on query characteristics, rather than using fixed score fusion.

**Research Questions**:
- How to define locality across heterogeneous vector spaces (dense vs sparse)?
- Can we learn modality-specific locality functions that adapt to query patterns?
- What theoretical guarantees can we provide for multi-modal recall?

---

#### 3. **Learned Locality Prediction**
**Research Gap Addressed**: Learned and Adaptive Indexing (Gap 3A, 3B)

**Novel Idea**: Use lightweight machine learning models to **predict locality relationships** between vectors, rather than computing all pairwise distances. LIM could:
- Train small neural networks or gradient-boosted trees to predict "should these vectors be neighbors?"
- Adaptively update locality predictions as data distributions shift
- Use predicted locality for fast approximate search, falling back to exact distance when needed

**Key Innovation**: **Predictive locality** - instead of computing expensive distance functions, learn to predict which vectors are likely neighbors based on vector features, enabling faster index construction and updates.

**Research Questions**:
- What model architectures (MLPs, transformers, graph neural networks) best predict locality?
- How to balance prediction accuracy vs. model size for real-time inference?
- Can we provide theoretical bounds on recall when using predicted vs. exact locality?

---

#### 4. **Energy-Aware Locality Optimization**
**Research Gap Addressed**: Sustainability and Energy Efficiency (Gap 5A, 5B)

**Novel Idea**: Design LIM to explicitly optimize for **energy efficiency** alongside accuracy and latency. This could involve:
- **Adaptive precision**: Use lower precision (FP16, INT8) for distant vectors, full precision only for close neighbors
- **Lazy locality computation**: Defer expensive distance calculations until necessary
- **Hardware-aware locality**: Structure the index to minimize cache misses and memory bandwidth

**Key Innovation**: **Energy-proportional locality** - the index structure adapts its computational intensity based on query difficulty, using less energy for easy queries and more for hard ones.

**Research Questions**:
- How to measure and model energy consumption of different locality computation strategies?
- What are the energy-accuracy trade-offs for different precision levels?
- Can we design locality structures optimized for specific hardware (ARM, GPU, neuromorphic)?

---

#### 5. **Context-Aware Personalized Locality**
**Research Gap Addressed**: Context-Aware Personalized Retrieval (Gap 7A, 7B)

**Novel Idea**: Make locality relationships **context-dependent** and **user-personalized**. LIM could:
- Maintain multiple locality views: one global view and per-user/context local views
- Dynamically adjust locality weights based on user query history
- Support session-aware locality where recent queries influence current search

**Key Innovation**: **Personalized locality spaces** - the same vector might have different "neighbors" depending on the user's context, enabling more relevant results without changing the underlying data.

**Research Questions**:
- How to efficiently maintain multiple locality views without excessive memory overhead?
- What learning algorithms can adapt locality relationships from user feedback?
- How to balance personalization with privacy (federated locality learning)?

---

#### 6. **Robust Locality with Distribution Shift Adaptation**
**Research Gap Addressed**: Robustness and Reproducibility (Gap 6C)

**Novel Idea**: Design LIM to be **robust to distribution shifts** by maintaining locality relationships that generalize across domains. This could involve:
- **Domain-agnostic locality**: Learn locality functions that work across different data distributions
- **Incremental adaptation**: Update locality relationships as new data arrives without full retraining
- **Uncertainty-aware locality**: Track confidence in locality predictions and use it to guide search

**Key Innovation**: **Adaptive locality** that maintains good performance even when query distributions differ from training data, addressing the OOD brittleness problem.

**Research Questions**:
- How to detect distribution shifts in vector data?
- What locality update strategies maintain performance under distribution drift?
- Can we provide theoretical guarantees on robustness?

---

### Recommended Starting Point

**Start with #1 (Adaptive Locality-Aware Indexing with Temporal Decay)** because:
1. It addresses a clear gap (temporal vector indexing is unexplored)
2. It's implementable with the current infrastructure
3. It has practical applications (e-commerce, social media, fraud detection)
4. It can be extended to other directions later

**Initial Implementation Plan**:
1. Add temporal metadata (insertion timestamp) to vectors
2. Modify distance function to include temporal decay: `d_combined = α * d_spatial + (1-α) * d_temporal`
3. Update locality clustering to consider temporal proximity
4. Implement time-windowed queries

---

### Success Metrics

For any LIM variant, measure:
- **Recall@k**: Accuracy compared to ground truth
- **Latency**: Query time and build time
- **Update throughput**: Inserts/updates per second
- **Memory efficiency**: Bytes per vector
- **Energy consumption**: Joules per query (if exploring #4)
- **Robustness**: Performance under distribution shifts (if exploring #6)

---

## index-seer (SEER: Similarity Estimation via Efficient Routing)

### Novel Research Direction: Learned Locality Prediction

**Research Gap Addressed**: Learned Index Structures (Gap 3A)

**Novel Idea**: Use lightweight machine learning models to **predict locality relationships** between vectors, rather than computing all pairwise distances. SEER:
- Uses random projections to create feature representations
- Learns which projection differences correlate with true distance
- Filters candidates before exact distance computation

**Key Innovation**: **Predictive locality** - instead of computing expensive distance functions, learn to predict which vectors are likely neighbors based on vector features, enabling faster index construction and updates.

### Algorithm

```
predicted_locality = predictor.score(query, candidate)
if predicted_locality > threshold:
    exact_distance = compute_distance(query, candidate)  // Only compute when likely match
```

### Implementation Details

1. **LocalityPredictor**: Random projection-based scorer
   - Projects vectors onto `n_projections` random unit vectors
   - Learns weights via correlation with true distances
   - Scores based on weighted projection similarity

2. **Candidate Filtering**: 
   - Score all vectors with predictor (O(n × projections) — fast)
   - Select top candidates based on threshold
   - Compute exact distances only for candidates

> [!WARNING]
> **Critical Analysis (2025-12-17)**: SEER is currently **25× slower** than linear scan due to O(n) scoring overhead. See [seer_analysis.md](./seer_analysis.md) for full details and recommended fixes.

### Benchmark Results

| Scenario | Points | Dim | SEER QPS | Linear QPS | Speedup | Recall |
|----------|--------|-----|----------|------------|---------|--------|
| `smoke` | 1,000 | 32 | 48.5 | 1,105.5 | **0.04×** | 98.75% |
| `recall-baseline` | 10,000 | 64 | 2.7 | 67.2 | **0.04×** | 96.48% |

**Key Findings**:
- Recall is good (~97%), but performance degrades linearly with dataset size
- The O(n) scoring phase negates any pruning benefit
- Needs LSH bucketing or hierarchical structure to be competitive

### Research Questions
- What model architectures (MLPs, transformers, graph neural networks) best predict locality?
- How to balance prediction accuracy vs. model size for real-time inference?
- Can we provide theoretical bounds on recall when using predicted vs. exact locality?
- **NEW**: How to add spatial partitioning (LSH/VP-trees) to avoid O(n) scoring?

### Configuration Parameters
| Parameter | Default | Description | Issue |
|-----------|---------|-------------|-------|
| `n_projections` | 16 | Number of random projections | ✓ OK |
| `n_samples` | 1000 | Training samples for weight learning | ⚠️ Training ineffective |
| `candidate_threshold` | 0.3 | Select top 30% as candidates | ⚠️ Actually selects 70% |
| `min_candidates` | 50 | Minimum candidates to always consider | ✓ OK |

