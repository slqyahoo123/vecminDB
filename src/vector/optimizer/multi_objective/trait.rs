use crate::Result;
use super::types::{MultiObjectiveConfig, MultiObjectiveResult};
use crate::vector::optimizer::{ParameterSpace, ParameterValue};
use crate::vector::optimizer::optimization_result::OptimizationResult;
use crate::vector::index::types::IndexType;
use std::collections::HashMap;
use super::types::Pareto;

/// 多目标优化器特征
/// 
/// 这个特征定义了所有多目标优化器必须实现的方法
pub trait MultiObjectiveOptimizer: Send + Sync {
    /// 优化给定的目标函数集合，在指定的参数范围内查找最优解
    fn optimize(
        &mut self, 
        objective_functions: &[Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>],
        bounds: &[(f64, f64)]
    ) -> Result<MultiObjectiveResult>;
    
    /// 获取优化器配置
    fn get_config(&self) -> &MultiObjectiveConfig;
    
    /// 验证解是否满足约束条件
    fn validate_solution(&self, solution: &Vec<f64>, bounds: &[(f64, f64)]) -> Result<bool>;
    
    /// 计算解的优势值
    fn calculate_dominance(
        &self,
        solution1_objectives: &[f64],
        solution2_objectives: &[f64]
    ) -> i32;
    
    /// 计算质量指标（如超体积、广义距离等）
    fn calculate_metrics(&self, solutions: &[Vec<f64>], objectives: &[Vec<f64>]) -> Result<MultiObjectiveResult>;
    
    /// 评估特定配置的性能
    fn evaluate_configuration(&self, index_type: IndexType, parameters: &HashMap<String, ParameterValue>, parameter_space: &ParameterSpace) -> Result<OptimizationResult>;
    
    /// 计算帕累托前沿
    fn compute_pareto_front(&self) -> Vec<Pareto>;
    
    /// 选择最佳配置
    fn select_best_configuration(&self, pareto_front: &[Pareto]) -> usize;
    
    /// 优化索引
    /// 
    /// # 参数
    /// - `index_type`: 索引类型
    /// - `parameter_space`: 参数空间
    /// 
    /// # 返回值
    /// 返回帕累托前沿和优化结果
    fn optimize_index(&mut self, index_type: IndexType, parameter_space: &ParameterSpace) -> Result<(Vec<Pareto>, OptimizationResult)>;
} 