// 优化结果模块
// 定义优化过程的结果结构

use crate::vector::{
    index::{IndexType, IndexConfig},
    benchmark::BenchmarkResult
};
use super::config::OptimizationTarget;

/// 优化结果
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// 索引类型
    pub index_type: IndexType,
    /// 最佳配置
    pub best_config: IndexConfig,
    /// 性能结果
    pub performance: BenchmarkResult,
    /// 评估的配置数量
    pub evaluated_configs: usize,
    /// 优化目标
    pub target: OptimizationTarget,
    /// 优化分数
    pub score: f64,
} 