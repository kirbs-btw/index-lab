# SOTA Assessment: Are We Beating State of the Art?

**Short answer**: No — not yet. Our best benchmarked algorithm (ZENITH) is a well-tuned HNSW variant. It improves usability (zero config) and recall vs. poorly tuned HNSW, but has not been validated against properly tuned HNSW or other SOTA systems.

---

## 1. The Bottom Line

| Question | Answer |
|----------|--------|
| **Do we beat HNSW?** | Only vs. HNSW with default params; HNSW is not tuned in our setup |
| **Do we beat Linear?** | Yes — ZENITH is ~3.1× faster at 50K vectors |
| **Do we beat FUSION?** | On speed: yes (5.1× faster at 50K). On recall at scale: no (FUSION 93.3% vs ZENITH 81.3%) |
| **Novel contributions?** | Yes — zero-config auto-tuning, diversity heuristic, RNG neighbor selection |

---

## 2. Benchmarked Results (10K points, 64 dims)

| Algorithm | QPS | Recall@20 | Config | Notes |
|-----------|-----|----------|--------|-------|
| **ZENITH** | 2,113 | 94.82% | Zero | Best balance of speed/recall |
| Linear | 2,087 | 100.0% | None | Correctness baseline |
| HNSW | 33,970 | 1.09% | Manual | Fast but broken with defaults |
| FUSION | 637 | 93.96% | 6+ | High recall, slower |
| LIM | 1,829 | 95.14% | Multiple | Best recall, O(n) insert |

---

## 3. Why Our HNSW Looks Bad

Our HNSW uses default parameters and shows ~1% recall. With tuned `m_max`, `ef_construction`, `ef_search`, HNSW typically reaches 95%+ recall. ZENITH’s 94.82% comes from auto-tuning those same parameters.

ZENITH’s advantage is **zero configuration**, not a fundamentally new architecture.

---

## 4. Theoretical vs. Practical

| Algorithm | Theoretical SOTA Claim | Practical Status |
|-----------|------------------------|------------------|
| **UNIVERSAL** | Should beat HNSW (O(N) build, zero config, lazy construction) | Never benchmarked |
| **CONVERGENCE** | Complex but “comprehensive” | Test failures, 36 warnings |
| **ZENITH** | Auto-tuned HNSW | Benchmarked; improves usability, not proven SOTA |

---

## 5. What We Can Claim

1. **Usability**: Zero-config auto-tuning is a real improvement for production use.
2. **Correctness**: Fixes HNSW’s broken default behavior.
3. **Simplicity**: ~450 lines vs. thousands in previous generations.
4. **Scalability**: ~3.1× faster than Linear at 50K.

---

## 6. What We Cannot Claim (Yet)

1. **SOTA recall**: Not validated against tuned HNSW, DiskANN, or FAISS.
2. **SOTA speed**: Not validated against tuned HNSW or GPU-optimized systems.
3. **Billion-scale**: No benchmarks > 1M vectors.

---

## 7. Next Steps to Validate SOTA

1. **Benchmark vs. tuned HNSW** — Same dataset, same recall target.
2. **ANN-Benchmarks** — Run on standard benchmarks (e.g., glove-100).
3. **Large-scale** — 100K–1M vectors to validate scaling.
4. **Real data** — Text/image embeddings instead of synthetic vectors.

---

*Last updated: 2026-02-12. Source: zenith_analysis.md, benchmark results, research_summary.md.*
