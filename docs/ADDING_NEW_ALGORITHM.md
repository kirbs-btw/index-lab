# Adding a New Algorithm Type

This guide outlines all the steps needed to add a new vector index algorithm to the testing infrastructure.

## Overview

To add a new algorithm (e.g., `index-lsh` for Locality-Sensitive Hashing), you need to:

1. Create a new crate for the algorithm
2. Implement the `VectorIndex` trait
3. Register it in the workspace
4. Integrate it into the benchmark runner
5. Add tests

## Step-by-Step Instructions

### 1. Create the New Crate

Create a new directory and Cargo.toml:

```bash
mkdir -p crates/index-lsh/src
```

**`crates/index-lsh/Cargo.toml`**:
```toml
[package]
name = "index-lsh"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = { workspace = true }
index-core = { path = "../index-core" }
rand = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
thiserror = { workspace = true }
```

### 2. Implement the Algorithm

**`crates/index-lsh/src/lib.rs`** (template):
```rust
//! Locality-Sensitive Hashing (LSH) index implementation.

use anyhow::{ensure, Result};
use index_core::{
    distance, DistanceMetric, load_index, save_index, ScoredPoint, 
    validate_dimension, Vector, VectorIndex
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LshError {
    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("index is empty, cannot search")]
    EmptyIndex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LshIndex {
    metric: DistanceMetric,
    dimension: Option<usize>,
    // Add your algorithm-specific fields here
    vectors: Vec<(usize, Vector)>,
}

impl LshIndex {
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            metric,
            dimension: None,
            vectors: Vec::new(),
        }
    }

    fn validate_dimension(&self, vector: &[f32]) -> Result<()> {
        if let Some(expected) = self.dimension {
            validate_dimension(Some(expected), vector.len())
                .map_err(|_| LshError::DimensionMismatch {
                    expected,
                    actual: vector.len(),
                })?;
        }
        Ok(())
    }

    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    /// Saves the index to a JSON file
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        save_index(self, path)
    }

    /// Loads an index from a JSON file
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        load_index(path)
    }
}

impl VectorIndex for LshIndex {
    fn metric(&self) -> DistanceMetric {
        self.metric
    }

    fn len(&self) -> usize {
        self.vectors.len()
    }

    fn insert(&mut self, id: usize, vector: Vector) -> Result<()> {
        self.validate_dimension(&vector)?;
        if self.dimension.is_none() {
            self.dimension = Some(vector.len());
        }
        // Implement your insertion logic here
        self.vectors.push((id, vector));
        Ok(())
    }

    fn search(&self, query: &Vector, limit: usize) -> Result<Vec<ScoredPoint>> {
        ensure!(limit > 0, "limit must be greater than zero to execute a search");
        ensure!(!self.vectors.is_empty(), LshError::EmptyIndex);
        self.validate_dimension(query)?;

        // Implement your search logic here
        let mut candidates = Vec::with_capacity(self.vectors.len());
        for (id, vector) in &self.vectors {
            let dist = distance(self.metric, query, vector)?;
            candidates.push(ScoredPoint::new(*id, dist));
        }

        candidates.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        candidates.truncate(limit.min(candidates.len()));
        Ok(candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_search_returns_expected_ids() {
        let mut index = LshIndex::new(DistanceMetric::Euclidean);
        index.insert(0, vec![0.0, 0.0]).unwrap();
        index.insert(1, vec![1.0, 0.0]).unwrap();
        
        let result = index.search(&vec![0.0, 0.0], 2).unwrap();
        assert_eq!(result[0].id, 0);
    }
}
```

### 3. Register in Workspace

**Update `Cargo.toml`** (root):
```toml
[workspace]
members = [
    "crates/index-core",
    "crates/index-linear",
    "crates/index-hnsw",
    "crates/index-ivf",
    "crates/index-pq",
    "crates/index-lsh",  # ADD THIS LINE
    "crates/bench-runner",
]
```

### 4. Add to Benchmark Runner Dependencies

**Update `crates/bench-runner/Cargo.toml`**:
```toml
[dependencies]
# ... existing dependencies ...
index-lsh = { path = "../index-lsh" }  # ADD THIS LINE
```

### 5. Integrate into Benchmark Runner

**Update `crates/bench-runner/src/main.rs`**:

#### 5a. Add import:
```rust
use index_lsh::LshIndex;  // ADD THIS LINE
```

#### 5b. Add to `IndexWrapper` enum:
```rust
enum IndexWrapper {
    Linear(LinearIndex),
    Hnsw(HnswIndex),
    Ivf(IvfIndex),
    Pq(PqIndex),
    Lsh(LshIndex),  // ADD THIS LINE
}
```

#### 5c. Update `VectorIndex` impl for `IndexWrapper`:
```rust
impl VectorIndex for IndexWrapper {
    fn metric(&self) -> DistanceMetric {
        match self {
            IndexWrapper::Linear(idx) => idx.metric(),
            IndexWrapper::Hnsw(idx) => idx.metric(),
            IndexWrapper::Ivf(idx) => idx.metric(),
            IndexWrapper::Pq(idx) => idx.metric(),
            IndexWrapper::Lsh(idx) => idx.metric(),  // ADD THIS LINE
        }
    }

    fn len(&self) -> usize {
        match self {
            IndexWrapper::Linear(idx) => idx.len(),
            IndexWrapper::Hnsw(idx) => idx.len(),
            IndexWrapper::Ivf(idx) => idx.len(),
            IndexWrapper::Pq(idx) => idx.len(),
            IndexWrapper::Lsh(idx) => idx.len(),  // ADD THIS LINE
        }
    }

    fn insert(&mut self, id: usize, vector: Vector) -> Result<()> {
        match self {
            IndexWrapper::Linear(idx) => idx.insert(id, vector),
            IndexWrapper::Hnsw(idx) => idx.insert(id, vector),
            IndexWrapper::Ivf(idx) => idx.insert(id, vector),
            IndexWrapper::Pq(idx) => idx.insert(id, vector),
            IndexWrapper::Lsh(idx) => idx.insert(id, vector),  // ADD THIS LINE
        }
    }

    fn search(&self, query: &Vector, limit: usize) -> Result<Vec<ScoredPoint>> {
        match self {
            IndexWrapper::Linear(idx) => idx.search(query, limit),
            IndexWrapper::Hnsw(idx) => idx.search(query, limit),
            IndexWrapper::Ivf(idx) => idx.search(query, limit),
            IndexWrapper::Pq(idx) => idx.search(query, limit),
            IndexWrapper::Lsh(idx) => idx.search(query, limit),  // ADD THIS LINE
        }
    }
}
```

#### 5d. Add to `IndexType` enum:
```rust
#[derive(Debug, Clone, ValueEnum)]
enum IndexType {
    Linear,
    Hnsw,
    Ivf,
    Pq,
    Lsh,  // ADD THIS LINE
}
```

#### 5e. Add match arm in `main()`:
```rust
match cli.index_type {
    // ... existing arms ...
    IndexType::Lsh => {
        run_benchmark(
            "LSH",
            |load_path| {
                println!("Loading LSH index from {}...", load_path.display());
                let load_start = Instant::now();
                let loaded = LshIndex::load(load_path)
                    .with_context(|| format!("failed to load index from {}", load_path.display()))?;
                let load_time = load_start.elapsed();
                println!("Loaded index in {:.2?} ({} vectors, metric: {:?})", 
                         load_time, loaded.len(), loaded.metric());
                Ok((IndexWrapper::Lsh(loaded), load_time))
            },
            || {
                let mut index = LshIndex::new(runtime.metric);
                let build_start = Instant::now();
                index.build(dataset.clone())?;
                let build_time = build_start.elapsed();
                Ok((IndexWrapper::Lsh(index), build_time))
            },
            |idx| {
                if let Some(save_path) = cli.save_index.as_deref() {
                    if let IndexWrapper::Lsh(inner) = idx {
                        inner.save(save_path)
                            .with_context(|| format!("failed to save index to {}", save_path.display()))?;
                        println!("Saved LSH index to {} ({} vectors)", save_path.display(), inner.len());
                    }
                }
                Ok(())
            },
            false, // Set to true if your algorithm is exhaustive (like linear)
            &dataset,
            &queries,
            &runtime,
            &cli,
        )?;
    }
}
```

#### 5f. Update `print_results()` function:
```rust
let index_type_str = match index {
    IndexWrapper::Linear(_) => "linear",
    IndexWrapper::Hnsw(_) => "hnsw",
    IndexWrapper::Ivf(_) => "ivf",
    IndexWrapper::Pq(_) => "pq",
    IndexWrapper::Lsh(_) => "lsh",  // ADD THIS LINE
};
```

#### 5g. Update exhaustive check (if needed):
```rust
if !matches!(index, IndexWrapper::Linear(_)) {
    // ... recall printing ...
}
// If LSH is also exhaustive, update to:
if !matches!(index, IndexWrapper::Linear(_) | IndexWrapper::Lsh(_)) {
    // ...
}
```

### 6. Verify Everything Works

```bash
# Check that it compiles
cargo check

# Run a quick test
cargo run --bin bench-runner -- --index-type lsh --points 100 --queries 10

# Run with a scenario
cargo run --bin bench-runner -- --scenario smoke --index-type lsh
```

## Summary Checklist

- [ ] Created new crate directory and `Cargo.toml`
- [ ] Implemented `VectorIndex` trait for the new algorithm
- [ ] Added `save()` and `load()` methods (using shared helpers)
- [ ] Added crate to workspace `members` in root `Cargo.toml`
- [ ] Added dependency in `bench-runner/Cargo.toml`
- [ ] Added import in `bench-runner/src/main.rs`
- [ ] Added variant to `IndexWrapper` enum
- [ ] Updated all 4 methods in `IndexWrapper` impl
- [ ] Added variant to `IndexType` enum
- [ ] Added match arm in `main()` function
- [ ] Updated `index_type_str` in `print_results()`
- [ ] Updated exhaustive check if algorithm is exhaustive
- [ ] Added unit tests in the algorithm crate
- [ ] Verified compilation with `cargo check`
- [ ] Tested with benchmark runner

## Notes

- **Save/Load**: Use the shared `save_index()` and `load_index()` helpers from `index-core`
- **Dimension Validation**: Use the shared `validate_dimension()` helper
- **Error Types**: Follow the pattern of other algorithms (DimensionMismatch, EmptyIndex, etc.)
- **Serialization**: Make sure your index struct derives `Serialize` and `Deserialize`
- **Testing**: Include at least basic insert/search tests, and save/load tests

## Example: Minimal Working Algorithm

For a quick test, you can start with a simple wrapper around `LinearIndex`:

```rust
pub struct LshIndex {
    inner: LinearIndex,
}

impl VectorIndex for LshIndex {
    // Delegate all methods to inner
}
```

This lets you verify the integration works before implementing the actual algorithm.

