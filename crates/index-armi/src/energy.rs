use crate::graph::Precision;

/// Tracks energy budget and consumption
#[derive(Debug, Clone)]
pub struct EnergyBudget {
    /// Total energy budget per query (in arbitrary units)
    budget_per_query: Option<f32>,
    /// Current energy consumed in this query
    current_consumption: f32,
    /// Energy cost per operation type
    operation_costs: OperationCosts,
}

#[derive(Debug, Clone)]
struct OperationCosts {
    distance_fp32: f32,
    distance_fp16: f32,
    distance_int8: f32,
    graph_traversal: f32,
}

impl Default for OperationCosts {
    fn default() -> Self {
        Self {
            distance_fp32: 1.0,
            distance_fp16: 0.5,
            distance_int8: 0.25,
            graph_traversal: 0.1,
        }
    }
}

impl EnergyBudget {
    pub fn new(budget_per_query: Option<f32>) -> Self {
        Self {
            budget_per_query,
            current_consumption: 0.0,
            operation_costs: OperationCosts::default(),
        }
    }
    
    pub fn record_distance(&mut self, precision: Precision) -> anyhow::Result<()> {
        let cost = match precision {
            Precision::FP32 => self.operation_costs.distance_fp32,
            Precision::FP16 => self.operation_costs.distance_fp16,
            Precision::INT8 => self.operation_costs.distance_int8,
        };
        
        self.current_consumption += cost;
        
        if let Some(budget) = self.budget_per_query {
            if self.current_consumption > budget {
                return Err(anyhow::anyhow!("energy budget exhausted"));
            }
        }
        
        Ok(())
    }
    
    pub fn record_traversal(&mut self) -> anyhow::Result<()> {
        self.current_consumption += self.operation_costs.graph_traversal;
        
        if let Some(budget) = self.budget_per_query {
            if self.current_consumption > budget {
                return Err(anyhow::anyhow!("energy budget exhausted"));
            }
        }
        
        Ok(())
    }
    
    pub fn reset(&mut self) {
        self.current_consumption = 0.0;
    }
    
    pub fn remaining(&self) -> Option<f32> {
        self.budget_per_query.map(|budget| (budget - self.current_consumption).max(0.0))
    }
    
    pub fn record_insert(&mut self) -> anyhow::Result<()> {
        // Insert operations are typically cheaper (no graph traversal)
        self.current_consumption += 0.05;
        Ok(())
    }
}

/// Selects appropriate precision based on query difficulty and energy budget
#[derive(Debug, Clone)]
pub struct PrecisionSelector {
    /// Base precision to use
    base_precision: Precision,
}

impl PrecisionSelector {
    pub fn new() -> Self {
        Self {
            base_precision: Precision::FP32,
        }
    }
    
    pub fn select(
        &self,
        _query: &crate::modality::MultiModalQuery,
        budget: &EnergyBudget,
    ) -> anyhow::Result<Precision> {
        // Simplified: select precision based on remaining budget
        // In production, would also consider query difficulty
        
        if let Some(remaining) = budget.remaining() {
            if remaining < 0.3 {
                Ok(Precision::INT8) // Low budget: use lowest precision
            } else if remaining < 0.6 {
                Ok(Precision::FP16) // Medium budget: use medium precision
            } else {
                Ok(Precision::FP32) // High budget: use full precision
            }
        } else {
            Ok(self.base_precision) // No budget constraint
        }
    }
}

impl Default for PrecisionSelector {
    fn default() -> Self {
        Self::new()
    }
}
