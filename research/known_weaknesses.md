# Known Algorithm Weaknesses

## APEX
- ~~Incremental insert routing broken~~ (Fixed Feb 13)
- ~~Single bucket for all incremental inserts~~ (Fixed Mar 4 with splitting)
- Router training ineffective for small datasets (<100 vectors)
- Temporal decay not tested with real timestamps

## Synthesis
- Fixed LSH parameters (not adaptive)
- Single-layer graph (no hierarchy)
- Router OR graph routing, never ensemble
- Incomplete serialization (graph edges lost)

## Convergence
- Dead code in ensemble router
- Edge pruning thresholds hardcoded
- Online router learning never converges (learning rate too high)
- Multi-strategy search always falls back to default

## Fusion
- LSH parameters not validated (could be 0 or negative)
- Mini-graphs rebuilt on every insert (no incremental)
- Multi-probe only works with SimHash (not cross-polytope)

## Universal
- AutoTuner records data but never actually adapts
- All public methods are unused (dead code)

## General Issues
- No crate uses logging (all println! or silent)
- Error types inconsistent across crates
- No serialization tests
- Benchmark runner not wired to any real dataset
