# Index-Lab Research Hub

> Central navigation for the index-lab vector indexing project.

---

## 📁 Document Structure

```
research/
├── README.md                    ← You are here (navigation)
├── SOTA_SUMMARY.md              ← Are we beating state of the art? (short answer)
│
├── CORE OVERVIEWS
│   ├── algorithm_findings.md   ← All algorithms: pros, cons, benchmarks
│   ├── research_summary.md     ← APEX → SYNTHESIS → CONVERGENCE → UNIVERSAL evolution
│   └── research_gaps.md        ← Research opportunities + literature
│
├── REFERENCE & SYNTHESIS
│   ├── algorithm_flaws_documentation.md  ← All known flaws across algorithms
│   └── research_gaps.md                  ← 7 research gaps (see above)
│
├── ALGORITHM DEEP DIVES (by category)
│   │
│   ├── Best performing (benchmarked)
│   │   └── zenith_analysis.md   ← ZENITH: zero-config HNSW (94.82% recall)
│   │
│   ├── Research-generation (APEX → UNIVERSAL)
│   │   ├── apex_analysis.md
│   │   ├── apex_implementation_analysis.md
│   │   ├── synthesis_analysis.md
│   │   ├── synthesis_critical_analysis.md
│   │   ├── convergence_analysis.md
│   │   └── universal_analysis.md
│   │
│   ├── Novel algorithms (by research gap)
│   │   ├── lim_analysis.md      ← Gap 1C: Temporal indexing
│   │   ├── hybrid_analysis.md   ← Gap 2: Sparse-dense fusion
│   │   ├── seer_analysis.md     ← Gap 3A: Learned index
│   │   ├── swift_analysis.md    ← Gap 3A: LSH + mini-graphs
│   │   ├── nexus_analysis.md    ← Gap 3A: Spectral routing
│   │   ├── prism_analysis.md    ← Gap 7: Context-aware
│   │   ├── vortex_analysis.md   ← Gap 2B: Cluster routing
│   │   ├── atlas_analysis.md    ← Gaps 1A, 2C, 3A, 7A
│   │   ├── armi_analysis.md     ← Gaps 1B, 5, 6A, 6B, 7A
│   │   └── fusion_analysis.md   ← LSH + mini-graphs
│   │
│   └── Baselines
│       └── (HNSW, IVF, PQ, Linear — see algorithm_findings.md)
```

---

## 🎯 Quick Start

| Goal | Read |
|------|------|
| **Are we beating SOTA?** | [SOTA_SUMMARY.md](./SOTA_SUMMARY.md) |
| **Overview of all algorithms** | [algorithm_findings.md](./algorithm_findings.md) |
| **Find research ideas** | [research_gaps.md](./research_gaps.md) |
| **Algorithm evolution story** | [research_summary.md](./research_summary.md) |
| **Known flaws & limitations** | [algorithm_flaws_documentation.md](./algorithm_flaws_documentation.md) |
| **Best performing algorithm** | [zenith_analysis.md](./zenith_analysis.md) |

---

## 📊 Algorithm Status Matrix

| Algorithm | Recall | QPS (10K) | Config | Status |
|-----------|--------|-----------|--------|--------|
| **ZENITH** | 94.82% | 2,113 | 0 (auto) | ✅ Best balanced |
| **LIM** | 95.14% | 1,829 | Multiple | ✅ High recall |
| **FUSION** | 93.96% | 637 | 6+ | ✅ High recall |
| **HNSW** | 1.09% | 33,970 | Manual | ⚠️ Broken defaults |
| **SEER** | 96.5% | 110 | Few | 🔴 11× slower than linear |
| **SWIFT** | 6.0% | 15,884 | Few | 🔴 Recall issues |
| **PRISM** | 0.8% | 32,389 | Few | 🔴 Recall bug |
| **NEXUS** | 14.6% | 2,329 | Few | 🔴 Recall issues |

*Full data: [algorithm_findings.md](./algorithm_findings.md), [algorithm_flaws_documentation.md](./algorithm_flaws_documentation.md)*

---

## 🚨 Critical Issues to Fix

| Algorithm | Issue | Fix | Doc |
|-----------|-------|-----|-----|
| SEER | 11× slower than linear | Add LSH bucketing | [seer_analysis.md](./seer_analysis.md) |
| LIM | O(n) cluster search | KD-tree for centroids | [lim_analysis.md](./lim_analysis.md) |
| Hybrid | Linear sparse scan | Build inverted index | [algorithm_findings.md](./algorithm_findings.md) |
| HNSW | 1% recall (defaults) | Use ZENITH or tune params | [zenith_analysis.md](./zenith_analysis.md) |

---

## 🏃 Running Benchmarks

```bash
cargo run --release -p bench-runner -- --scenario smoke          # Quick validation
cargo run --release -p bench-runner -- --scenario recall-baseline # Accuracy test
cargo run --release -p bench-runner -- --scenario io-heavy        # Stress test
```

---

## 📖 Key Papers Referenced

| Paper | Authors | Year | Relevance |
|-------|---------|------|-----------|
| HNSW | Malkov & Yashunin | 2016 | Graph-based baseline |
| DiskANN | Subramanya et al. | 2019 | Billion-scale reference |
| FAISS | Johnson et al. | 2017 | GPU-accelerated baseline |
| RaBitQ | Gao & Long | 2024 | SIGMOD Best Paper, 1-bit quantization |

Full citations: [research_gaps.md](./research_gaps.md#appendix-a-state-of-the-art-algorithms)
