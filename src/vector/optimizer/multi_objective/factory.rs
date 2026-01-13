use crate::{Error, Result};
use super::types::{
    MultiObjectiveConfig,
    MultiObjectiveAlgorithm,
    MultiObjectiveResult,
    ObjectiveDirection,
    MOBOConfig,
    MOBOParameters,
};
// NSGA2Optimizer 未实现 MultiObjectiveOptimizer，暂不引入以避免未使用警告
use super::moead::MOEADOptimizer;
use super::mopso::MOPSOOptimizer;
use super::mosa::MOSAOptimizer;
use super::multi_gradient::MultiGradientOptimizer;
use super::r#trait::MultiObjectiveOptimizer;
use super::mobo::MOBO;

/// 多目标优化器工厂，用于创建不同类型的多目标优化算法
pub struct MultiObjectiveOptimizerFactory;

impl MultiObjectiveOptimizerFactory {
    /// 创建指定类型的多目标优化器
    pub fn create(algorithm: MultiObjectiveAlgorithm, config: MultiObjectiveConfig) -> Result<Box<dyn MultiObjectiveOptimizer>> {
        let algorithm_config = config.clone();

        match algorithm {
            MultiObjectiveAlgorithm::NSGA2 => {
                let population_size = algorithm_config.population_size;
                let mutation_rate = algorithm_config.mutation_probability;
                let crossover_rate = algorithm_config.crossover_probability;

                // NSGA2Optimizer 当前未实现 MultiObjectiveOptimizer，暂返回未实现错误
                return Err(Error::not_implemented(
                    "NSGA2Optimizer 未实现 MultiObjectiveOptimizer 接口，暂不支持工厂创建".to_string(),
                ));
            },
            MultiObjectiveAlgorithm::MoeaD => {
                let optimizer = MOEADOptimizer::new(algorithm_config)?;
                Ok(Box::new(optimizer))
            },
            MultiObjectiveAlgorithm::MOPSO => {
                let optimizer = MOPSOOptimizer::new(algorithm_config)?;
                Ok(Box::new(optimizer))
            },
            MultiObjectiveAlgorithm::MOSA => {
                let optimizer = MOSAOptimizer::new(algorithm_config);
                Ok(Box::new(optimizer))
            },
            MultiObjectiveAlgorithm::MOBO => {
                println!("创建MOBO多目标贝叶斯优化器");
                
                // 从通用配置转换为 MOBO 特定配置
                let params = MOBOParameters {
                    max_iterations: config.max_iterations,
                    initial_samples: config
                        .algorithm_params
                        .get("initial_samples")
                        .map(|v| v.round() as usize)
                        .unwrap_or(10),
                    candidate_count: config
                        .algorithm_params
                        .get("candidate_count")
                        .map(|v| v.round() as usize)
                        .unwrap_or(config.population_size.max(10)),
                    objective_weights: (
                        config
                            .algorithm_params
                            .get("weight_query")
                            .cloned()
                            .unwrap_or(0.7),
                        config
                            .algorithm_params
                            .get("weight_size")
                            .cloned()
                            .unwrap_or(0.3),
                    ),
                    time_limit: None,
                };

                let mobo_config = MOBOConfig {
                    seed: config.seed,
                    parameters: params,
                };
                
                let optimizer = MOBO::new(mobo_config);
                Ok(Box::new(optimizer))
            },
            MultiObjectiveAlgorithm::MultiGradient => {
                let optimizer = MultiGradientOptimizer::new(algorithm_config);
                Ok(Box::new(optimizer))
            },
            MultiObjectiveAlgorithm::Custom => {
                Err(Error::not_implemented(
                    "Custom 多目标算法工厂尚未实现，需提供 custom_name 对应实现".to_string(),
                ))
            },
            algorithm => Err(Error::not_implemented(format!("算法 {:?} 尚未实现", algorithm))),
        }
    }
    
    /// 运行一个简单的多目标优化示例
    pub fn run_example() -> Result<MultiObjectiveResult> {
        println!("运行多目标优化示例...");
        
        // 创建一个NSGA2的配置
        let config = MultiObjectiveConfig {
            algorithm: MultiObjectiveAlgorithm::NSGA2,
            objectives_count: 2,
            population_size: 100,
            max_iterations: 100,
            mutation_probability: 0.1,
            crossover_probability: 0.9,
            seed: Some(42),
            objective_directions: vec![ObjectiveDirection::Minimize, ObjectiveDirection::Minimize],
            ..Default::default()
        };
        
        // 创建优化器实例
        let mut optimizer = Self::create(config.algorithm, config)?;
        
        // 定义测试函数 - 使用ZDT1测试函数
        let f1 = |x: &Vec<f64>| -> f64 { x[0] };
        let f2 = |x: &Vec<f64>| -> f64 {
            let g = 1.0 + 9.0 * (x.iter().skip(1).sum::<f64>() / (x.len() - 1) as f64);
            g * (1.0 - (x[0] / g).sqrt())
        };
        
        // 目标函数向量
        let objectives: Vec<Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>> = vec![
            Box::new(f1) as Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>,
            Box::new(f2) as Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>,
        ];
        
        // 定义搜索空间 - 每个维度在[0,1]之间
        let dimensions = 30;
        let mut bounds = Vec::new();
        for _ in 0..dimensions {
            bounds.push((0.0, 1.0));
        }
        
        // 运行优化
        let result = optimizer.optimize(objectives.as_slice(), &bounds)?;
        
        // 输出结果
        println!("优化完成! 找到{}个Pareto最优解", result.pareto_front.len());
        println!("指标:");
        println!("  超体积: {}", result.hypervolume.unwrap_or(0.0));
        println!("  世代距离: {}", result.generational_distance.unwrap_or(0.0));
        println!("  反向世代距离: {}", result.inverted_generational_distance.unwrap_or(0.0));
        println!("  分散度: {}", result.spread.unwrap_or(0.0));
        println!("  运行时间: {}ms", result.runtime_ms);
        
        Ok(result)
    }
    
    /// 运行MOSA多目标优化示例
    pub fn run_mosa_example() -> Result<MultiObjectiveResult> {
        println!("运行MOSA多目标优化示例...");
        
        // 创建一个MOSA的配置
        let mut config = MultiObjectiveConfig {
            algorithm: MultiObjectiveAlgorithm::MOSA,
            objectives_count: 2,
            max_iterations: 50,
            seed: Some(42),
            objective_directions: vec![ObjectiveDirection::Minimize, ObjectiveDirection::Minimize],
            ..Default::default()
        };
        
        // MOSA特定参数
        config.algorithm_params.insert("initial_temperature".to_string(), 100.0);
        config.algorithm_params.insert("cooling_rate".to_string(), 0.95);
        config.algorithm_params.insert("iterations_per_temperature".to_string(), 10.0);
        config.algorithm_params.insert("archive_size".to_string(), 100.0);
        
        // 创建优化器实例
        let mut optimizer = Self::create(config.algorithm, config)?;
        
        // 定义测试函数 - 使用ZDT2测试函数
        let f1 = |x: &Vec<f64>| -> f64 { x[0] };
        let f2 = |x: &Vec<f64>| -> f64 {
            let g = 1.0 + 9.0 * (x.iter().skip(1).sum::<f64>() / (x.len() - 1) as f64);
            g * (1.0 - (x[0] / g).powi(2))
        };
        
        // 目标函数向量
        let objectives: Vec<Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>> = vec![
            Box::new(f1) as Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>,
            Box::new(f2) as Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>,
        ];
        
        // 定义搜索空间 - 每个维度在[0,1]之间
        let dimensions = 30;
        let mut bounds = Vec::new();
        for _ in 0..dimensions {
            bounds.push((0.0, 1.0));
        }
        
        // 运行优化
        let result = optimizer.optimize(objectives.as_slice(), &bounds)?;
        
        // 输出结果
        println!("MOSA优化完成! 找到{}个Pareto最优解", result.pareto_front.len());
        println!("指标:");
        println!("  超体积: {}", result.hypervolume.unwrap_or(0.0));
        println!("  运行时间: {}ms", result.runtime_ms);
        
        Ok(result)
    }
    
    /// 运行MOBO多目标贝叶斯优化示例
    pub fn run_mobo_example() -> Result<MultiObjectiveResult> {
        println!("运行MOBO多目标贝叶斯优化示例...");
        
        // 创建一个MOBO的配置
        let mut config = MultiObjectiveConfig {
            algorithm: MultiObjectiveAlgorithm::MOBO,
            objectives_count: 2,
            max_iterations: 30,
            seed: Some(42),
            objective_directions: vec![ObjectiveDirection::Minimize, ObjectiveDirection::Minimize],
            ..Default::default()
        };
        
        // MOBO特定参数
        config.algorithm_params.insert("initial_samples".to_string(), 10.0);
        config.algorithm_params.insert("iteration_samples".to_string(), 5.0);
        config.algorithm_params.insert("exploration_weight".to_string(), 0.5);
        config.algorithm_params.insert("kernel_param".to_string(), 0.1);
        
        // 创建优化器实例
        let mut optimizer = Self::create(config.algorithm, config)?;
        
        // 定义测试函数 - 使用ZDT3测试函数
        let f1 = |x: &Vec<f64>| -> f64 { x[0] };
        let f2 = |x: &Vec<f64>| -> f64 {
            let g = 1.0 + 9.0 * (x.iter().skip(1).sum::<f64>() / (x.len() - 1) as f64);
            let h = 1.0 - (x[0] / g).sqrt() - (x[0] / g) * (10.0 * x[0] * std::f64::consts::PI).sin();
            g * h
        };
        
        // 目标函数向量
        let objectives: Vec<Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>> = vec![
            Box::new(f1) as Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>,
            Box::new(f2) as Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>,
        ];
        
        // 定义搜索空间 - 每个维度在[0,1]之间
        let dimensions = 10;  // MOBO使用较少的维度以加快计算速度
        let mut bounds = Vec::new();
        for _ in 0..dimensions {
            bounds.push((0.0, 1.0));
        }
        
        // 运行优化
        let result = optimizer.optimize(objectives.as_slice(), &bounds)?;
        
        // 输出结果
        println!("MOBO优化完成! 找到{}个Pareto最优解", result.pareto_front.len());
        println!("指标:");
        println!("  超体积: {}", result.hypervolume.unwrap_or(0.0));
        println!("  运行时间: {}ms", result.runtime_ms);
        
        Ok(result)
    }
    
    /// 运行多目标梯度下降示例
    pub fn run_multi_gradient_example() -> Result<MultiObjectiveResult> {
        println!("运行多目标梯度下降优化示例...");
        
        // 创建一个MultiGradient的配置
        let mut config = MultiObjectiveConfig {
            algorithm: MultiObjectiveAlgorithm::MultiGradient,
            objectives_count: 2,
            max_iterations: 100,
            population_size: 50,
            seed: Some(42),
            objective_directions: vec![ObjectiveDirection::Minimize, ObjectiveDirection::Minimize],
            ..Default::default()
        };
        
        // MultiGradient特定参数
        config.algorithm_params.insert("learning_rate".to_string(), 0.01);
        config.algorithm_params.insert("momentum".to_string(), 0.9);
        
        // 创建优化器实例
        let mut optimizer = Self::create(config.algorithm, config)?;
        
        // 定义测试函数 - 使用ZDT4测试函数
        let f1 = |x: &Vec<f64>| -> f64 { x[0] };
        let f2 = |x: &Vec<f64>| -> f64 {
            let g = 1.0 + 10.0 * (x.len() - 1) as f64 + 
                x.iter().skip(1).map(|&xi| {
                    xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos()
                }).sum::<f64>();
            
            g * (1.0 - (x[0] / g).sqrt())
        };
        
        // 目标函数向量（需要 Send + Sync）
        let objectives: Vec<Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>> = vec![
            Box::new(f1) as Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>,
            Box::new(f2) as Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>,
        ];
        
        // 定义搜索空间 - 第一个维度在[0,1]，其他维度在[-5,5]
        let dimensions = 10;
        let mut bounds = Vec::new();
        bounds.push((0.0, 1.0)); // 第一个变量
        for _ in 1..dimensions {
            bounds.push((-5.0, 5.0)); // 其他变量
        }
        
        // 运行优化
        let result = optimizer.optimize(objectives.as_slice(), &bounds)?;
        
        // 输出结果
        println!("多目标梯度下降优化完成! 找到{}个Pareto最优解", result.pareto_front.len());
        println!("指标:");
        println!("  超体积: {}", result.hypervolume.unwrap_or(0.0));
        println!("  世代距离: {}", result.generational_distance.unwrap_or(0.0));
        println!("  运行时间: {}ms", result.runtime_ms);
        
        Ok(result)
    }
} 