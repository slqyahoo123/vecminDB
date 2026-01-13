// 向量优化器配置模块
// 包含优化器的配置结构和默认实现

use serde::{Serialize, Deserialize};
use crate::vector::benchmark::BenchmarkConfig;

/// 优化目标
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OptimizationTarget {
    /// 优化查询速度
    QuerySpeed,
    /// 优化构建速度
    BuildSpeed,
    /// 优化内存使用
    MemoryUsage,
    /// 优化准确率
    Accuracy,
    /// 平衡速度和准确率
    BalancedPerformance,
}

impl Default for OptimizationTarget {
    fn default() -> Self {
        OptimizationTarget::BalancedPerformance
    }
}

/// 参数优化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// 基准测试配置
    pub benchmark_config: BenchmarkConfig,
    /// 优化目标
    pub target: OptimizationTarget,
    /// 最大迭代次数
    pub max_iterations: usize,
    /// 并行优化
    pub parallel: bool,
    /// 随机种子
    pub random_seed: Option<u64>,
    /// 是否使用贝叶斯优化
    pub use_bayesian: bool,
    /// 是否使用网格搜索
    pub use_grid_search: bool,
    /// 是否使用随机搜索
    pub use_random_search: bool,
    /// 是否使用遗传算法
    pub use_genetic_algorithm: bool,
    /// 数据集配置
    pub dataset: Option<DatasetConfig>,
}

/// 数据集配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// 数据集名称
    pub name: String,
    /// 数据集路径
    pub path: String,
    /// 数据集大小
    pub size: usize,
    /// 向量维度
    pub dimensions: usize,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            benchmark_config: BenchmarkConfig::default(),
            target: OptimizationTarget::default(),
            max_iterations: 50,
            parallel: true,
            random_seed: None,
            use_bayesian: false,
            use_grid_search: true,
            use_random_search: true,
            use_genetic_algorithm: false,
            dataset: None,
        }
    }
} 