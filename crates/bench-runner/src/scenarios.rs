use clap::ValueEnum;
use index_core::DistanceMetric;

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab_case")]
pub enum ScenarioKind {
    /// Fastest possible run – verifies plumbing and I/O quickly.
    Smoke,
    /// Balanced Euclidean case meant for recall/latency comparisons.
    RecallBaseline,
    /// Cosine-distance workload to exercise normalization-sensitive indexes.
    CosineQuality,
    /// Larger dataset that stresses memory bandwidth and I/O.
    IoHeavy,
}

#[derive(Clone, Debug)]
pub struct ScenarioConfig {
    pub dimension: usize,
    pub points: usize,
    pub queries: usize,
    pub limit: usize,
    pub metric: DistanceMetric,
}

#[derive(Clone, Debug)]
pub struct ScenarioDetails {
    pub slug: &'static str,
    pub label: &'static str,
    pub description: &'static str,
    pub config: ScenarioConfig,
}

impl ScenarioKind {
    pub fn details(self) -> ScenarioDetails {
        match self {
            ScenarioKind::Smoke => ScenarioDetails {
                slug: "smoke",
                label: "Smoke test (5k pts)",
                description: "Quick correctness sanity check that fits easily into CI.",
                config: ScenarioConfig {
                    dimension: 64,
                    points: 5_000,
                    queries: 64,
                    limit: 10,
                    metric: DistanceMetric::Euclidean,
                },
            },
            ScenarioKind::RecallBaseline => ScenarioDetails {
                slug: "recall-baseline",
                label: "Recall baseline (100k pts)",
                description: "Large Euclidean dataset for evaluating recall vs. latency at scale.",
                config: ScenarioConfig {
                    dimension: 128,
                    points: 100_000,
                    queries: 500,
                    limit: 50,
                    metric: DistanceMetric::Euclidean,
                },
            },
            ScenarioKind::CosineQuality => ScenarioDetails {
                slug: "cosine-quality",
                label: "Cosine quality (75k pts)",
                description: "Cosine distance case with higher dimensionality for quality sweeps.",
                config: ScenarioConfig {
                    dimension: 256,
                    points: 75_000,
                    queries: 500,
                    limit: 50,
                    metric: DistanceMetric::Cosine,
                },
            },
            ScenarioKind::IoHeavy => ScenarioDetails {
                slug: "io-heavy",
                label: "I/O heavy (500k pts)",
                description: "Large memory-bound setup to observe throughput and cache behavior.",
                config: ScenarioConfig {
                    dimension: 384,
                    points: 500_000,
                    queries: 1_000,
                    limit: 25,
                    metric: DistanceMetric::Euclidean,
                },
            },
        }
    }

    pub fn list_details() -> Vec<ScenarioDetails> {
        ScenarioKind::value_variants()
            .iter()
            .map(|kind| kind.details())
            .collect()
    }
}

pub fn print_available() {
    println!("Available benchmark scenarios:");
    for details in ScenarioKind::list_details() {
        println!(
            "  {:<15} {:<26} dim={:<4} points={:<7} queries={:<5} limit={:<3} metric={:?}",
            details.slug,
            details.label,
            details.config.dimension,
            details.config.points,
            details.config.queries,
            details.config.limit,
            details.config.metric
        );
        println!("      {}", details.description);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_scenarios_have_unique_slugs() {
        let mut slugs = ScenarioKind::list_details()
            .into_iter()
            .map(|detail| detail.slug)
            .collect::<Vec<_>>();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), ScenarioKind::value_variants().len());
    }
}
