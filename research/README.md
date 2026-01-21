# Index-Lab Research Hub

> Central navigation for the index-lab vector indexing project.

---

## 📁 Files (9 total)

| File | Purpose |
|------|---------|
| **README.md** | Navigation (you're here) |
| [**algorithm_findings.md**](./algorithm_findings.md) | ⭐ Summary of all algorithms |
| [**research_gaps.md**](./research_gaps.md) | Research opportunities + literature |
| [**lim_analysis.md**](./lim_analysis.md) | LIM deep dive |
| [**seer_analysis.md**](./seer_analysis.md) | SEER deep dive |
| [**hybrid_analysis.md**](./hybrid_analysis.md) | Hybrid Index deep dive |
| [**swift_analysis.md**](./swift_analysis.md) | ✅ SWIFT algorithm analysis (LSH + mini-graphs) |
| [**nexus_analysis.md**](./nexus_analysis.md) | ✅ NEXUS algorithm analysis (spectral + adaptive) |
| [**prism_analysis.md**](./prism_analysis.md) | ✅ PRISM algorithm analysis (session-aware) |
| [**vortex_analysis.md**](./vortex_analysis.md) | ✅ VORTEX algorithm analysis (cluster routing) |

---

## 🎯 Quick Start

| Goal | Read |
|------|------|
| **Understand the project** | [algorithm_findings.md](./algorithm_findings.md) |
| **Find research ideas** | [research_gaps.md](./research_gaps.md) |
| **Fix LIM issues** | [lim_analysis.md](./lim_analysis.md) |
| **Fix SEER issues** | [seer_analysis.md](./seer_analysis.md) |
| **Fix Hybrid issues** | [hybrid_analysis.md](./hybrid_analysis.md) |

---

## 📊 Project Status at a Glance

### Novel Algorithms We Implemented

| Algorithm | What It Does | Status | Issue |
|-----------|--------------|--------|-------|
| **LIM** | Temporal-aware vector search | ✅ Works | ⚠️ O(n) insertion |
| **Hybrid** | Dense + sparse fusion | ✅ Works | ⚠️ Linear scan |
| **SEER** | Learned locality prediction | ✅ Works | 🔴 25× slower than baseline |
| **SWIFT** | LSH bucketing + mini-graphs | ✅ Works | ⚠️ LSH data distribution sensitivity |
| **PRISM** | Session-aware adaptive search | ✅ Works | Session state requires mutable access |
| **NEXUS** | Spectral embedding + adaptive graph | ✅ Works | O(n²) graph build time |
| **VORTEX** | Cluster-driven graph routing | ✅ Works | O(N*C) training overhead |

### Research Gaps Addressed

| Gap | Description | Algorithm | Status |
|-----|-------------|-----------|--------|
| 1C | Temporal vector indexing | **LIM** | ✅ Implemented |
| 2A, 2B | Sparse-Dense Fusion | **Hybrid** | ✅ Implemented |
| 3A | Learned index structures | **SEER** | ⚠️ Needs optimization |
| 3A | Fast candidate generation | **SWIFT** | ✅ Implemented |
| 3A | Spectral manifold learning | **NEXUS** | ✅ Implemented |
| 7 | Context-aware, adaptive search | **PRISM** | ✅ Implemented |
| 2B | Graph-based cluster routing | **VORTEX** | ✅ Implemented |

### Gaps Not Yet Explored
- Gap 4: Privacy-preserving search
- Gap 5: Energy efficiency  
- Gap 6: Robustness/reproducibility

---

## 🚨 Critical Issues to Fix

### 1. SEER: 25× Slower Than Linear Scan
**File**: [seer_analysis.md](./seer_analysis.md)

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| 2.7 QPS vs 67 QPS (linear) | Scores ALL vectors before filtering | Add LSH bucketing for O(1) lookup |

**Estimated time**: 2-3 hours

---

### 2. LIM: O(n) Cluster Search
**File**: [lim_analysis.md](./lim_analysis.md)

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| Every insert checks ALL clusters | No spatial index for centroids | Use KD-tree for O(log n) lookup |
| Spatial/temporal scale mismatch | Spatial 0→∞, temporal 0→1 | Normalize spatial to [0,1] |

**Estimated time**: 2-3 hours

---

### 3. Hybrid: Linear Sparse Scan
**File**: [algorithm_findings.md](./algorithm_findings.md)

| Problem | Root Cause | Fix |
|---------|-----------|-----|
| Slow sparse term matching | No inverted index | Build term → [doc_ids] index |

**Estimated time**: 3-4 hours

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

Full citations in [research_gaps.md](./research_gaps.md#appendix-a-state-of-the-art-algorithms).

---
