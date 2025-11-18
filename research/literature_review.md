# State-of-the-Art Vector Indexing Algorithms with Primary Research Papers

A comprehensive reference of current state-of-the-art vector indexing algorithms and their foundational research papers.

---

## Graph-Based Algorithms

| Algorithm | Authors | Publication | Year | Key Innovation |
|-----------|---------|-------------|------|----------------|
| **HNSW** (Hierarchical Navigable Small World) | Yu. A. Malkov, D. A. Yashunin | IEEE TPAMI, Vol. 42 | 2016/2020 | Multi-layer navigable graph with logarithmic search complexity |
| **NSG** (Navigating Spreading-out Graph) | Cong Fu, Chao Xiang, Changxu Wang, Deng Cai | VLDB 2019 | 2017 | Single-layer graph with spreading-out property for better navigation |
| **Vamana** (DiskANN) | Suhas Jayaram Subramanya, Devvrit, Rohan Kadekodi, Ravishankar Krishnaswamy, Harsha Simhadri | NeurIPS 2019 | 2019 | Disk-based billion-scale search with aggressive edge pruning |
| **CAGRA** (CUDA ANN Graph) | Hiroyuki Ootomo et al., NVIDIA RAPIDS | arXiv:2308.15136 | 2023 | GPU-optimized graph construction with 33-77× speedup |
| **NGT** (Neighborhood Graph and Tree) | Masajiro Iwasaki, Yahoo Japan | Open source library | 2016 | Hybrid approach combining graph and tree structures |

---

## Quantization-Based Algorithms

| Algorithm | Authors | Publication | Year | Key Innovation |
|-----------|---------|-------------|------|----------------|
| **Product Quantization (PQ)** | Hervé Jégou, Matthijs Douze, Cordelia Schmid | IEEE TPAMI, Vol. 33 | 2011 | Vector compression via subvector quantization with codebooks |
| **FAISS** | Jeff Johnson, Matthijs Douze, Hervé Jégou | IEEE Trans. on Big Data (2019); arXiv:2401.08281 (2024) | 2017/2024 | Comprehensive library with GPU-accelerated similarity search |
| **ScaNN** (Scalable Nearest Neighbors) | Ruiqi Guo, Philip Sun, Erik Lindgren, Quan Geng, David Simcha, Felix Chern, Sanjiv Kumar (Google) | ICML 2020 | 2020 | Anisotropic vector quantization for improved accuracy |
| **SOAR** | Philip Sun, David Simcha, Dave Dopson, Ruiqi Guo, Sanjiv Kumar (Google) | NeurIPS 2023 / arXiv:2404.00774 | 2023/2024 | Orthogonality-amplified indexing improving upon ScaNN |
| **RaBitQ** | Jianyang Gao, Cheng Long | SIGMOD 2024 / arXiv:2405.12497 | 2024 | 1-bit quantization with theoretical error bounds |

---

## Hybrid and Inverted File Algorithms

| Algorithm | Authors | Publication | Year | Key Innovation |
|-----------|---------|-------------|------|----------------|
| **IVF** (Inverted File Index) | Hervé Jégou et al. | Various papers | 2008-2011 | Cluster-based partitioning for efficient search |
| **SPANN** | Qi Chen, Bing Zhao, Haidong Wang, et al. (Microsoft Research, Peking University) | NeurIPS 2021 / arXiv:2111.08566 | 2021 | Highly-efficient billion-scale search with memory-disk hybrid |
| **Filtered-DiskANN** | Siddharth Gollapudi, Neel Karia, et al. (Microsoft Research) | WWW 2023 | 2023 | Graph algorithms supporting metadata filter constraints |

---

## Hash-Based Algorithms

| Algorithm | Authors | Publication | Year | Key Innovation |
|-----------|---------|-------------|------|----------------|
| **LSH** (Locality-Sensitive Hashing) | Piotr Indyk, Rajeev Motwani | ACM STOC | 1998 | Hash functions mapping similar vectors to same buckets |
| **p-Stable LSH** | Mayur Datar, Nicole Immorlica, Piotr Indyk, Vahab S. Mirrokni | VLDB 2004 | 2004 | LSH using p-stable distributions for Lp distances |
| **Annoy** (Approximate Nearest Neighbors Oh Yeah) | Erik Bernhardsson (Spotify) | Open source library | 2013 | Random projection trees optimized for static datasets |

---

## Research Paper Citations

### Graph-Based Algorithms

**HNSW (Hierarchical Navigable Small World)**
- Malkov, Y. A., & Yashunin, D. A. (2016). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. *arXiv preprint arXiv:1603.09320*.
- Malkov, Y. A., & Yashunin, D. A. (2020). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(4), 824-836.

**NSG (Navigating Spreading-out Graph)**
- Fu, C., Xiang, C., Wang, C., & Cai, D. (2019). Fast approximate nearest neighbor search with the navigating spreading-out graph. *Proceedings of the VLDB Endowment*, 12(5), 461-474.
- arXiv preprint: arXiv:1707.00143 (2017)

**Vamana (DiskANN)**
- Subramanya, S. J., Devvrit, F., Kadekodi, R., Krishnaswamy, R., & Simhadri, H. V. (2019). DiskANN: Fast accurate billion-point nearest neighbor search on a single node. *Advances in Neural Information Processing Systems (NeurIPS)*, 32, 13766-13776.
- Microsoft Research technical report (2019)

**CAGRA (CUDA ANN Graph)**
- Ootomo, H., et al. (2023). CAGRA: Highly parallel graph construction and approximate nearest neighbor search for GPUs. *arXiv preprint arXiv:2308.15136*.
- Part of NVIDIA RAPIDS cuVS library

**NGT (Neighborhood Graph and Tree)**
- Iwasaki, M. (2016). NGT: Neighborhood graph and tree for indexing high-dimensional data. *GitHub repository and documentation*.
- Yahoo Japan open source project: https://github.com/yahoojapan/NGT

### Quantization-Based Algorithms

**Product Quantization (PQ)**
- Jégou, H., Douze, M., & Schmid, C. (2011). Product quantization for nearest neighbor search. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 33(1), 117-128.
- arXiv preprint: arXiv:1102.3828

**FAISS**
- Johnson, J., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535-547.
- arXiv preprint: arXiv:1702.08734 (2017)
- Douze, M., et al. (2024). The Faiss library. *arXiv preprint arXiv:2401.08281*.
- Facebook AI Research (Meta) open source project

**ScaNN (Scalable Nearest Neighbors)**
- Guo, R., Sun, P., Lindgren, E., Geng, Q., Simcha, D., Chern, F., & Kumar, S. (2020). Accelerating large-scale inference with anisotropic vector quantization. *International Conference on Machine Learning (ICML)*, 3887-3896.
- Google Research open source library

**SOAR**
- Sun, P., Simcha, D., Dopson, D., Guo, R., & Kumar, S. (2024). SOAR: Improved indexing for approximate nearest neighbor search. *Advances in Neural Information Processing Systems (NeurIPS)*, 36.
- arXiv preprint: arXiv:2404.00774 (2023)
- Google Research algorithmic improvement to ScaNN

**RaBitQ**
- Gao, J., & Long, C. (2024). RaBitQ: Quantizing high-dimensional vectors with a theoretical error bound for approximate nearest neighbor search. *Proceedings of the ACM SIGMOD International Conference on Management of Data*, 1-14.
- arXiv preprint: arXiv:2405.12497
- SIGMOD 2024 Best Paper Award

### Hybrid and Inverted File Algorithms

**IVF (Inverted File Index)**
- Jégou, H., Douze, M., & Schmid, C. (2008). Hamming embedding and weak geometric consistency for large scale image search. *European Conference on Computer Vision (ECCV)*, 304-317.
- Foundational work integrated into FAISS library

**SPANN**
- Chen, Q., Zhao, B., Wang, H., Li, M., Liu, C., Li, Z., Yang, M., & Wang, J. (2021). SPANN: Highly-efficient billion-scale approximate nearest neighbor search. *Advances in Neural Information Processing Systems (NeurIPS)*, 34, 5199-5212.
- arXiv preprint: arXiv:2111.08566
- Microsoft Research & Peking University collaboration

**Filtered-DiskANN**
- Gollapudi, S., Karia, N., Sivashankar, V., Krishnaswamy, R., Begum, N., Rao, S., Najork, M., & Simhadri, H. V. (2023). Filtered-DiskANN: Graph algorithms for approximate nearest neighbor search with filters. *Proceedings of the ACM Web Conference (WWW)*, 3406-3416.
- Microsoft Research extension of DiskANN

### Hash-Based Algorithms

**LSH (Locality-Sensitive Hashing)**
- Indyk, P., & Motwani, R. (1998). Approximate nearest neighbors: Towards removing the curse of dimensionality. *Proceedings of the 30th Annual ACM Symposium on Theory of Computing (STOC)*, 604-613.
- Foundational paper establishing LSH framework

**p-Stable LSH**
- Datar, M., Immorlica, N., Indyk, P., & Mirrokni, V. S. (2004). Locality-sensitive hashing scheme based on p-stable distributions. *Proceedings of the 20th Annual Symposium on Computational Geometry (SCG)*, 253-262.
- Also published in modified form in VLDB 2004

**Annoy (Approximate Nearest Neighbors Oh Yeah)**
- Bernhardsson, E. (2013). Annoy: Approximate nearest neighbors in C++/Python. *GitHub repository*.
- Spotify open source project: https://github.com/spotify/annoy
- Blog post: "Nearest neighbor methods and vector models" (2015)

---

## Additional Key References

### Survey Papers

- **Comprehensive Vector Database Survey**: Wang, J., et al. (2023). A comprehensive survey on vector database: Storage and retrieval techniques, challenges. *arXiv preprint arXiv:2310.11703*.

- **Graph-Based ANN Evaluation**: Li, W., et al. (2019). Approximate nearest neighbor search on high dimensional data—experiments, analyses, and improvement. *IEEE Transactions on Knowledge and Data Engineering*, 32(8), 1475-1488.

- **Filtered Vector Search**: Hadian, A., & Herodotou, H. (2023). Filtered vector search: State-of-the-art and research directions. *VLDB Tutorial*.

### Benchmarking Resources

- **ANN-Benchmarks**: Aumüller, M., Bernhardsson, E., & Faithfull, A. (2020). ANN-Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms. *Information Systems*, 87, 101374.
  - Website: http://ann-benchmarks.com
  - GitHub: https://github.com/erikbern/ann-benchmarks

- **Big-ANN Challenge**: Simhadri, H. V., et al. (2021). Results of the NeurIPS'21 Challenge on Billion-Scale Approximate Nearest Neighbor Search. *NeurIPS Competition Track*.
  - Website: https://big-ann-benchmarks.com

---

## Notes on Current State-of-the-Art (November 2025)

**Dominant Production Systems**: HNSW remains the most widely deployed for in-memory search, while DiskANN/Vamana is standard for billion-scale deployments requiring SSD-based indexing.

**GPU Acceleration**: CAGRA (2023) represents the current state-of-the-art for GPU-accelerated search, with 33-77× speedups over CPU-based HNSW on appropriate hardware.

**Extreme Compression**: RaBitQ (2024) achieves 32× compression with 1-bit quantization while maintaining competitive recall, representing the most aggressive compression with theoretical guarantees.

**Emerging Directions**: Research is actively exploring filtered search, multi-modal indexing, privacy-preserving vector search, learned index structures, and energy-efficient algorithms for sustainable AI infrastructure.

---

*This document represents the state of research as of November 2025. For the most current developments, consult recent publications from major venues including NeurIPS, ICML, SIGMOD, VLDB, and WWW.*