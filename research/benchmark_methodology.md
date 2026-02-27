# Benchmark Methodology

## Metrics

### Recall@k
Fraction of true nearest neighbors found in the top-k results.
Ground truth computed via brute-force linear scan.

### QPS (Queries Per Second)
Number of queries processed per second. Measured after index build,
using a warm cache. Each query is timed individually.

### Build Time
Wall-clock time to build the index from a dataset. Includes all
preprocessing (clustering, graph construction, router training).

## Datasets

### Synthetic
- Random uniform: `rand_uniform(n, dim)` — uniformly distributed vectors
- Clustered: `rand_clustered(n, dim, k)` — k Gaussian clusters

### Real-world (planned)
- SIFT1M: 1M 128-dim vectors
- GloVe-100: ~1.2M 100-dim word embeddings

## Test Protocol

1. Build index on training set
2. Compute ground truth via brute-force on test queries
3. Measure recall@k for k ∈ {1, 10, 100}
4. Measure QPS over 1000 queries (median of 3 runs)
5. Report recall-QPS tradeoff curve by varying ef_search
