/*
 * 多目标优化算法模块
 * 
 * 已实现算法：NSGA2, MOEA/D, MOPSO, MOSA, MOBO, MultiGradient
 * 
 * 详细开发路线图见: docs/algorithm_roadmap.md
 */

mod types;
mod r#trait;
mod nsga2;
mod moead;
mod mopso;
mod mosa;
mod mobo;
mod multi_gradient;
mod factory;

// 公开接口
pub use types::{
    MultiObjectiveAlgorithm,
    MultiObjectiveConfig,
    MultiObjectiveResult,
    ObjectiveDirection,
    ParetoMetric, 
    Solution,
    WeightStrategy,
    MOBOConfig, 
    MOBOParameters,
    MultiGradientConfig,
    MultiGradientParameters
};
pub use r#trait::MultiObjectiveOptimizer;
pub use nsga2::NSGA2Optimizer;
pub use moead::MOEADOptimizer;
pub use mopso::MOPSOOptimizer;
pub use mosa::MOSAOptimizer;
pub use mobo::MOBO;
pub use multi_gradient::MultiGradientOptimizer;
pub use factory::MultiObjectiveOptimizerFactory; 

/// 工厂函数，创建特定算法的优化器
pub fn create_optimizer(algorithm: MultiObjectiveAlgorithm, config: MultiObjectiveConfig) -> Result<Box<dyn MultiObjectiveOptimizer>, crate::error::Error> {
    MultiObjectiveOptimizerFactory::create(algorithm, config)
} 