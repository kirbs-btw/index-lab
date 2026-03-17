# Convergence vs Synthesis: Detailed Comparison

## Architecture Differences

### Routing
- **Synthesis**: MLP router OR centroid graph (one or the other)
- **Convergence**: Ensemble routing (combines both with learned weights)
  - Weakness: ensemble weights are initialized but never properly trained

### Graph Structure
- **Synthesis**: Single-layer NSW graph per bucket
- **Convergence**: Hierarchical graph with edge pruning
  - Weakness: pruning thresholds are hardcoded

### LSH
- **Synthesis**: Fixed hyperplane count at build time
- **Convergence**: Adaptive LSH that can rehash based on recall feedback
  - But: rehashing is expensive and rarely triggered

### Temporal
- **Synthesis**: Cross-modal temporal edges only
- **Convergence**: Full temporal integration across all components
  - Most complete implementation but adds ~30% build overhead

## Benchmark Results (Synthetic, 1000 vectors, 32-dim)

Both algorithms show similar recall@10 (~0.87-0.89) on uniform data.
Convergence is ~20% slower to build due to additional components.
The ensemble routing provides marginal improvement (+2% recall) at
the cost of significant complexity.

## Recommendation

For most use cases, APEX provides a better balance. Convergence should
be considered only when the ensemble routing or full temporal integration
is specifically needed.
