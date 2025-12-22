# index-la
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

## Benchmark scenarios & reusable test data
- Discover built-in benchmark cases with `cargo run -p bench-runner -- --list-scenarios`.
- Execute a predefined workload via `--scenario <name>` to get consistent recall/latency comparisons.
- Persist deterministic datasets and query vectors with `--export-testdata path/to/file.json`.
- Save timing + configuration summaries for later analysis using `--report-json path/to/report.json`.

| Scenario name     | Description                               | Metric    | Dim | Points | Queries | Limit |
|-------------------|-------------------------------------------|-----------|-----|--------|---------|-------|
| `smoke`           | 1k×32 Euclidean sanity check              | Euclidean | 32  | 1,000  | 32      | 5     |
| `recall-baseline` | 10k×64 recall/latency baseline            | Euclidean | 64  | 10,000 | 256     | 20    |
| `cosine-quality`  | 15k×128 cosine accuracy sweep             | Cosine    | 128 | 15,000 | 256     | 25    |
| `io-heavy`        | 50k×256 throughput + I/O stress           | Euclidean | 256 | 50,000 | 512     | 15    |

Example workflows:
```bash
# Run the recall baseline and emit both a timing summary and raw vectors.
cargo run -p bench-runner -- --scenario recall-baseline \
  --report-json reports/recall-baseline.json \
  --export-testdata fixtures/recall-baseline.json

# Generate a cosine-friendly dataset for offline recall testing without running the benchmark.
cargo run -p bench-runner -- --scenario cosine-quality --export-testdata data/cosine-quality.json
```

## Adding new ideas
- Add a new crate under `crates/` and depend on `index-core` for the shared abstractions.
- Implement the `VectorIndex` trait so the bench runner (or custom harnesses) can reuse your algorithm.
- Extend `bench-runner` or add Criterion benches when you need more rigorous measurements.
- See [CONTRIBUTING.md](CONTRIBUTING.md) for a step-by-step guide.
