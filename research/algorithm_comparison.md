# Algorithm Comparison

## Overview

| Algorithm | Build Complexity | Search Complexity | Recall@10 | Multi-Modal | Incremental |
|-----------|-----------------|-------------------|-----------|-------------|-------------|
| Linear | O(1) | O(n) | 1.0 | No | Yes |
| HNSW | O(n log n) | O(log n) | ~0.95 | No | Yes |
| IVF | O(n) | O(n/k) | ~0.85 | No | Partial |
| ATLAS | O(n log n) | O(log C + log B) | ~0.88 | Yes | No |
| ARMI | O(n) | O(log n) | ~0.82 | Yes | Yes |
| APEX | O(n log n) | O(log C + log B) | ~0.90 | Yes | Yes* |
| Synthesis | O(n log n) | O(log C + log B) | ~0.87 | Yes | No |
| Convergence | O(n log n) | O(log C + log B) | ~0.89 | Yes | No |

*APEX incremental mode now supported with bucket splitting (Mar 2026)

## Key Findings

### Strengths by Algorithm
- **HNSW**: Best pure recall, simple API, good incremental support
- **APEX**: Best balance of features — multi-modal, incremental, adaptive
- **Convergence**: Most complete feature set, but complex build
- **Fusion**: Interesting LSH + mini-graph hybrid, fast build

### Common Weaknesses
1. Multi-modal algorithms have high build overhead
2. LSH parameters are often hardcoded (Synthesis, Fusion)
3. Temporal decay features are largely untested in practice
4. Energy efficiency metrics are placeholder — no real power measurement
