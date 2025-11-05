# index-lab
Experimental playground for designing and benchmarking novel vector indexing algorithms.

## Layout
- `crates/index-core` — shared traits, distance metrics, and dataset generators.
- `crates/index-linear` — baseline linear scan implementation of the index traits.
- `crates/bench-runner` — tiny CLI harness for building indexes and timing queries.

## Getting started
1. Install the Rust toolchain (`rustup` + stable `cargo`).
2. Check the workspace builds: `cargo check`.
3. Run the sample benchmark harness:
   ```bash
   cargo run -p bench-runner -- --points 5000 --queries 256 --dimension 128 --limit 20
   ```
   The CLI accepts `--metric euclidean|cosine` and `--seed <u64>` to experiment with different setups.

## Adding new ideas
- Add a new crate under `crates/` and depend on `index-core` for the shared abstractions.
- Implement the `VectorIndex` trait so the bench runner (or custom harnesses) can reuse your algorithm.
- Extend `bench-runner` or add Criterion benches when you need more rigorous measurements.
