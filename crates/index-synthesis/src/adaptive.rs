//! Adaptive parameter tuning and energy efficiency
//! 
//! Uses RefCell for interior mutability so adaptive features work
//! in standard search operations (fixes APEX issue)

use crate::modality::MultiModalQuery;
use index_core::ScoredPoint;
use std::cell::RefCell;
use std::collections::HashMap;

/// Precision levels for energy-aware computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    FP32,
    FP16,
    INT8,
}

impl Default for Precision {
    fn default() -> Self {
        Precision::FP32
    }
}

/// RL-based parameter optimizer for adaptive tuning
/// 
/// Wrapped in RefCell for interior mutability
#[derive(Debug, Clone)]
pub struct ParameterOptimizer {
    /// Learned parameters per query type
    query_performance: HashMap<String, QueryPerformance>,
    /// Current ef value
    current_ef: usize,
    /// Min and max ef bounds
    min_ef: usize,
    max_ef: usize,
}

#[derive(Debug, Clone)]
struct QueryPerformance {
    /// Average recall achieved
    avg_recall: f32,
    /// Number of queries seen
    count: usize,
    /// Optimal ef for this query type
    optimal_ef: usize,
}

impl ParameterOptimizer {
    pub fn new(min_ef: usize, max_ef: usize) -> Self {
        Self {
            query_performance: HashMap::new(),
            current_ef: (min_ef + max_ef) / 2,
            min_ef,
            max_ef,
        }
    }
    
    pub fn select_params(&mut self, query: &MultiModalQuery) -> anyhow::Result<SearchParams> {
        // Simplified: use query signature to identify query type
        let query_type = self.query_signature(query);
        
        // Look up optimal ef for this query type
        let ef = if let Some(perf) = self.query_performance.get(&query_type) {
            perf.optimal_ef
        } else {
            self.current_ef
        };
        
        Ok(SearchParams { ef })
    }
    
    pub fn update(
        &mut self,
        query: &MultiModalQuery,
        results: &[ScoredPoint],
    ) -> anyhow::Result<()> {
        // Simplified: estimate recall based on result quality
        // In production, would compare to ground truth
        
        let query_type = self.query_signature(query);
        let estimated_recall = self.estimate_recall(results);
        
        let perf = self.query_performance.entry(query_type).or_insert_with(|| {
            QueryPerformance {
                avg_recall: 0.0,
                count: 0,
                optimal_ef: self.current_ef,
            }
        });
        
        // Update running average
        perf.avg_recall = (perf.avg_recall * perf.count as f32 + estimated_recall)
            / (perf.count + 1) as f32;
        perf.count += 1;
        
        // Adjust ef based on recall
        if perf.avg_recall < 0.9 && perf.optimal_ef < self.max_ef {
            perf.optimal_ef = (perf.optimal_ef + 10).min(self.max_ef);
        } else if perf.avg_recall > 0.98 && perf.optimal_ef > self.min_ef {
            perf.optimal_ef = (perf.optimal_ef - 5).max(self.min_ef);
        }
        
        Ok(())
    }
    
    fn query_signature(&self, query: &MultiModalQuery) -> String {
        // Create signature based on query modalities
        let mods = query.modalities();
        format!("{:?}", mods)
    }
    
    fn estimate_recall(&self, results: &[ScoredPoint]) -> f32 {
        // Simplified: assume high recall if we have many results with low distances
        if results.is_empty() {
            return 0.0;
        }
        
        let avg_distance: f32 = results.iter().map(|r| r.distance).sum::<f32>() / results.len() as f32;
        
        // Lower average distance suggests better recall
        // This is a heuristic - in production would use ground truth
        if avg_distance < 0.1 {
            0.98
        } else if avg_distance < 0.5 {
            0.95
        } else {
            0.90
        }
    }
    
    pub fn reset(&mut self) {
        self.query_performance.clear();
        self.current_ef = (self.min_ef + self.max_ef) / 2;
    }
}

/// Wrapper for ParameterOptimizer with interior mutability
/// 
/// Allows adaptive tuning in standard search operations
#[derive(Debug, Clone)]
pub struct AdaptiveOptimizer {
    optimizer: RefCell<ParameterOptimizer>,
}

impl AdaptiveOptimizer {
    pub fn new(min_ef: usize, max_ef: usize) -> Self {
        Self {
            optimizer: RefCell::new(ParameterOptimizer::new(min_ef, max_ef)),
        }
    }
    
    /// Select parameters (can be called from immutable context)
    pub fn select_params(&self, query: &MultiModalQuery) -> anyhow::Result<SearchParams> {
        self.optimizer.borrow_mut().select_params(query)
    }
    
    /// Update optimizer (can be called from immutable context)
    pub fn update(&self, query: &MultiModalQuery, results: &[ScoredPoint]) -> anyhow::Result<()> {
        self.optimizer.borrow_mut().update(query, results)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SearchParams {
    pub ef: usize,
}

/// Tracks energy budget and consumption
/// 
/// Wrapped in RefCell for interior mutability
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

/// Wrapper for EnergyBudget with interior mutability
#[derive(Debug, Clone)]
pub struct AdaptiveEnergyBudget {
    budget: RefCell<EnergyBudget>,
}

impl AdaptiveEnergyBudget {
    pub fn new(budget_per_query: Option<f32>) -> Self {
        Self {
            budget: RefCell::new(EnergyBudget::new(budget_per_query)),
        }
    }
    
    pub fn record_distance(&self, precision: Precision) -> anyhow::Result<()> {
        self.budget.borrow_mut().record_distance(precision)
    }
    
    pub fn record_traversal(&self) -> anyhow::Result<()> {
        self.budget.borrow_mut().record_traversal()
    }
    
    pub fn reset(&self) {
        self.budget.borrow_mut().reset();
    }
    
    pub fn remaining(&self) -> Option<f32> {
        self.budget.borrow().remaining()
    }
    
    pub fn record_insert(&self) -> anyhow::Result<()> {
        self.budget.borrow_mut().record_insert()
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
        _query: &MultiModalQuery,
        budget: &AdaptiveEnergyBudget,
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
