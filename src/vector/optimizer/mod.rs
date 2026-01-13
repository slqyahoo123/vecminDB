// 向量优化器模块
//
// 包含各种向量优化算法，包括单目标和多目标优化
// 多目标优化算法开发路线图: docs/algorithm_roadmap.md

mod config;
mod parameter_space;
mod optimization_result;
mod optimizers;
mod scoring;
mod utils;
mod auto_tune;
pub mod multi_objective;
pub mod gradient;
pub mod genetic;

// 从各个子模块重新导出公共接口
pub use config::{OptimizationTarget, OptimizerConfig, DatasetConfig};
pub use parameter_space::{ParameterRange, ParameterSpace, ParameterValue, ParameterType, Parameter, OptimizationDimension};
pub use optimization_result::OptimizationResult;
pub use optimizers::{IndexOptimizer, ConfigValue};
pub use auto_tune::{VectorIndexAutoTuner, AutoTuneConfig, create_auto_tuner, run_optimizer_example as run_auto_tune_example};

// 从multi_objective模块重新导出主要接口
pub use multi_objective::{
    MultiObjectiveAlgorithm,
    MultiObjectiveConfig,
    MultiObjectiveResult,
    MultiObjectiveOptimizer,
    NSGA2Optimizer,
    MOEADOptimizer,
    MOPSOOptimizer,
    MultiObjectiveOptimizerFactory,
    ObjectiveDirection
};

// 导出示例函数
pub use optimizers::run_optimizer_example;

// 工厂函数
pub fn create_optimizer(config: OptimizerConfig) -> IndexOptimizer {
    IndexOptimizer::new(config)
}

/// 创建向量优化器
pub fn create_vector_optimizer(config: AutoTuneConfig) -> VectorIndexAutoTuner {
    auto_tune::create_auto_tuner(config)
} 