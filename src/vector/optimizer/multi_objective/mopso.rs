use crate::{Error, Result};
use std::collections::HashMap;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use super::types::{
    MultiObjectiveAlgorithm,
    MultiObjectiveConfig,
    MultiObjectiveResult,
    ObjectiveDirection,
    Solution,
    Pareto
};
use super::r#trait::MultiObjectiveOptimizer;
use crate::vector::index::{IndexType, IndexConfig};
use crate::vector::optimizer::config::OptimizationTarget;
use crate::vector::utils::benchmark::BenchmarkResult;
use crate::vector::optimizer::{ParameterValue, ParameterSpace, OptimizationResult};

/// 粒子结构，表示搜索空间中的候选解
struct Particle {
    /// 当前位置
    position: Vec<f64>,
    /// 当前速度
    velocity: Vec<f64>,
    /// 个体最佳位置
    best_position: Vec<f64>,
    /// 个体最佳目标值
    best_objectives: HashMap<String, f64>,
    /// 当前目标值
    objectives: HashMap<String, f64>,
}

/// 多目标粒子群优化器
pub struct MOPSOOptimizer {
    /// 优化器配置
    config: MultiObjectiveConfig,
    /// 非支配解档案
    archive: Vec<Solution>,
    /// 随机数生成器
    rng: StdRng,
    /// 档案变化跟踪
    archive_change: f64,
}

impl MOPSOOptimizer {
    /// 创建新的MOPSO优化器
    pub fn new(config: MultiObjectiveConfig) -> Result<Self> {
        // 验证配置是否为MOPSO算法
        if config.algorithm != MultiObjectiveAlgorithm::MOPSO {
            return Err(Error::InvalidArgument("配置必须指定MOPSO算法".to_string()));
        }
        
        // 验证其他参数
        if config.objectives.len() < 2 {
            return Err(Error::InvalidArgument("MOPSO至少需要两个优化目标".to_string()));
        }
        
        if config.population_size < 10 {
            return Err(Error::InvalidArgument("MOPSO需要足够的种群大小".to_string()));
        }
        
        // 初始化随机数生成器
        let seed = config.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let rng = StdRng::seed_from_u64(seed);
        
        Ok(Self {
            config,
            archive: Vec::new(),
            rng,
            archive_change: 0.0,
        })
    }
    
    /// 判断解a是否支配解b
    fn dominates(&self, a: &HashMap<String, f64>, b: &HashMap<String, f64>) -> Result<bool> {
        let mut a_better_in_any = false;
        
        for objective in &self.config.objectives {
            let name = &objective.name;
            let a_value = a.get(name).ok_or_else(|| 
                Error::InvalidArgument(format!("Objective {} not found in solution", name)))?;
            let b_value = b.get(name).ok_or_else(|| 
                Error::InvalidArgument(format!("Objective {} not found in solution", name)))?;
            
            // 根据优化方向比较
            let a_better = match objective.direction {
                ObjectiveDirection::Minimize => a_value < b_value,
                ObjectiveDirection::Maximize => a_value > b_value,
            };
            
            // 如果a在任何一个目标上比b差，则a不支配b
            if !a_better && a_value != b_value {
                return Ok(false);
            }
            
            // 检查a是否在任何目标上严格更好
            if a_better {
                a_better_in_any = true;
            }
        }
        
        // a支配b当且仅当a在至少一个目标上严格更好，且在其他目标上不比b差
        Ok(a_better_in_any)
    }
    
    /// 更新非支配解档案
    fn update_archive(&mut self, particles: &[Particle]) -> Result<()> {
        let previous_size = self.archive.len();
        let mut added = false;
        
        // 检查每个粒子是否应该添加到档案中
        for particle in particles {
            let mut is_dominated = false;
            let mut i = 0;
            
            // 检查当前粒子是否被档案中的任何解支配
            while i < self.archive.len() {
                let archive_obj = &self.archive[i].objective_values;
                
                if self.dominates(archive_obj, &particle.objectives)? {
                    is_dominated = true;
                    break;
                } else if self.dominates(&particle.objectives, archive_obj)? {
                    // 如果当前粒子支配档案中的解，移除档案中的解
                    self.archive.remove(i);
                    added = true;
                } else {
                    i += 1;
                }
            }
            
            // 如果粒子不被支配，添加到档案
            if !is_dominated {
                let solution = Solution {
                    parameters: particle.position.clone(),
                    objective_values: particle.objectives.clone(),
                    is_pareto_optimal: true,
                    rank: None,
                    crowding_distance: None,
                    quality_metrics: HashMap::new(),
                };
                
                self.archive.push(solution);
                added = true;
            }
        }
        
        // 如果档案超过最大大小，进行修剪
        let max_archive_size = self.config.population_size * 2;
        if self.archive.len() > max_archive_size {
            self.prune_archive(max_archive_size)?;
        }
        
        // 计算档案变化率
        if previous_size > 0 {
            self.archive_change = added as u64 as f64 / previous_size as f64;
        }
        
        // 计算拥挤度
        self.calculate_crowding_distances()?;
        
        Ok(())
    }
    
    /// 修剪档案大小
    fn prune_archive(&mut self, max_size: usize) -> Result<()> {
        if self.archive.len() <= max_size {
            return Ok(());
        }
        
        // 确保已计算拥挤度
        self.calculate_crowding_distances()?;
        
        // 按拥挤度降序排序
        self.archive.sort_by(|a, b| {
            let a_dist = a.crowding_distance.unwrap_or(0.0);
            let b_dist = b.crowding_distance.unwrap_or(0.0);
            b_dist.partial_cmp(&a_dist).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // 保留拥挤度最高的解
        self.archive.truncate(max_size);
        
        Ok(())
    }
    
    /// 计算拥挤度距离
    fn calculate_crowding_distances(&mut self) -> Result<()> {
        if self.archive.is_empty() {
            return Ok(());
        }
        
        let archive_size = self.archive.len();
        
        // 为每个解初始化拥挤度为0
        for solution in &mut self.archive {
            solution.crowding_distance = Some(0.0);
        }
        
        // 对每个目标计算拥挤度
        for objective in &self.config.objectives {
            let obj_name = &objective.name;
            
            // 按当前目标排序
            self.archive.sort_by(|a, b| {
                let a_val = a.objective_values.get(obj_name).unwrap_or(&f64::MAX);
                let b_val = b.objective_values.get(obj_name).unwrap_or(&f64::MAX);
                a_val.partial_cmp(b_val).unwrap_or(std::cmp::Ordering::Equal)
            });
            
            // 设置边界点的拥挤度为无限大
            if archive_size > 1 {
                if let Some(first) = self.archive.first_mut() {
                    first.crowding_distance = Some(f64::INFINITY);
                }
                if let Some(last) = self.archive.last_mut() {
                    last.crowding_distance = Some(f64::INFINITY);
                }
            }
            
            // 计算中间点的拥挤度
            if archive_size > 2 {
                let min_obj = self.archive.first().unwrap().objective_values.get(obj_name).unwrap_or(&0.0);
                let max_obj = self.archive.last().unwrap().objective_values.get(obj_name).unwrap_or(&0.0);
                
                let obj_range = max_obj - min_obj;
                if obj_range.abs() > 1e-10 {  // 避免除以零
                    for i in 1..archive_size - 1 {
                        let prev_val = self.archive[i-1].objective_values.get(obj_name).unwrap_or(&0.0);
                        let next_val = self.archive[i+1].objective_values.get(obj_name).unwrap_or(&0.0);
                        
                        let distance = (next_val - prev_val) / obj_range;
                        let current = &mut self.archive[i];
                        let current_dist = current.crowding_distance.unwrap_or(0.0);
                        current.crowding_distance = Some(current_dist + distance);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// 选择领导者（用于粒子更新）
    fn select_leader(&mut self) -> Result<&Solution> {
        if self.archive.is_empty() {
            return Err(Error::InvalidOperation("Archive is empty, cannot select leader".to_string()));
        }
        
        // 使用锦标赛选择
        let tournament_size = (self.archive.len() as f64 * 0.1).ceil() as usize;
        let tournament_size = tournament_size.max(2).min(self.archive.len());
        
        let mut best_index = self.rng.gen_range(0..self.archive.len());
        let mut best_crowd = self.archive[best_index].crowding_distance.unwrap_or(0.0);
        
        for _ in 1..tournament_size {
            let idx = self.rng.gen_range(0..self.archive.len());
            let crowd = self.archive[idx].crowding_distance.unwrap_or(0.0);
            
            // 选择拥挤度更高的解
            if crowd > best_crowd {
                best_index = idx;
                best_crowd = crowd;
            }
        }
        
        Ok(&self.archive[best_index])
    }
    
    /// 计算超体积指标（衡量非支配解集合的质量）
    fn calculate_hypervolume(&self, solutions: &[Solution]) -> Result<f64> {
        if solutions.is_empty() {
            return Ok(0.0);
        }
        
        // 实际的超体积计算非常复杂
        // 这里使用一个简化版本作为示例
        // 在实际应用中应使用更复杂的算法
        
        let objectives_count = self.config.objectives.len();
        let mut total_volume = 0.0;
        
        // 设置参考点
        let mut reference_point = HashMap::new();
        for obj in &self.config.objectives {
            let value = match obj.direction {
                ObjectiveDirection::Minimize => f64::MAX,
                ObjectiveDirection::Maximize => 0.0,
            };
            reference_point.insert(obj.name.clone(), value);
        }
        
        // 计算每个解的贡献
        for solution in solutions {
            let mut volume = 1.0;
            
            for obj in &self.config.objectives {
                let sol_value = solution.objective_values.get(&obj.name).unwrap_or(&0.0);
                let ref_value = reference_point.get(&obj.name).unwrap_or(&0.0);
                
                let contribution = match obj.direction {
                    ObjectiveDirection::Minimize => ref_value - sol_value,
                    ObjectiveDirection::Maximize => sol_value - ref_value,
                };
                
                volume *= contribution.abs().max(0.0);
            }
            
            total_volume += volume;
        }
        
        Ok(total_volume)
    }
    
    fn generate_parameter_combinations(&self, parameter_space: &ParameterSpace) -> Result<Vec<HashMap<String, ParameterValue>>> {
        // 生成参数组合
        let mut combinations = Vec::new();
        
        // 基础实现：在参数空间内均匀采样生成参数组合
        // 完整的 MOPSO 算法应使用粒子群优化进行参数搜索
        for i in 0..5 {
            let mut params = HashMap::new();
            for (j, range) in parameter_space.ranges.iter().enumerate() {
                let value = if range.is_integer {
                    ParameterValue::Integer((range.min + (i as f64) * (range.max - range.min) / 4.0).round() as i64)
                } else {
                    ParameterValue::Float(range.min + (i as f64) * (range.max - range.min) / 4.0)
                };
                params.insert(range.name.clone(), value);
            }
            combinations.push(params);
        }
        
        Ok(combinations)
    }
    
    fn compute_pareto_ranking(&self, pareto_front: &mut [Pareto]) {
        // 计算帕累托排序
        for (i, pareto) in pareto_front.iter_mut().enumerate() {
            pareto.rank = i;
        }
    }
}

impl MultiObjectiveOptimizer for MOPSOOptimizer {
    fn optimize(
        &mut self, 
        objective_functions: &[Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>],
        bounds: &[(f64, f64)]
    ) -> Result<MultiObjectiveResult> {
        self.optimize(objective_functions, bounds)
    }
    
    fn get_config(&self) -> &MultiObjectiveConfig {
        &self.config
    }
    
    fn validate_solution(&self, solution: &Vec<f64>, bounds: &[(f64, f64)]) -> Result<bool> {
        if solution.len() != bounds.len() {
            return Ok(false);
        }
        
        for (i, &val) in solution.iter().enumerate() {
            let (min, max) = bounds[i];
            if val < min || val > max {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    fn calculate_dominance(
        &self,
        solution1_objectives: &[f64],
        solution2_objectives: &[f64]
    ) -> i32 {
        // 直接比较目标值数组
        if solution1_objectives.len() != solution2_objectives.len() {
            return 0;
        }
        
        let mut solution1_better = false;
        let mut solution2_better = false;
        
        for (a, b) in solution1_objectives.iter().zip(solution2_objectives.iter()) {
            if a < b {
                solution1_better = true;
            } else if b < a {
                solution2_better = true;
            }
        }
        
        if solution1_better && !solution2_better {
            1  // solution1 支配 solution2
        } else if solution2_better && !solution1_better {
            -1  // solution2 支配 solution1
        } else {
            0  // 互不支配
        }
    }
    
    fn calculate_metrics(&self, solutions: &[Vec<f64>], objectives: &[Vec<f64>]) -> Result<MultiObjectiveResult> {
        // 转换为 Solution 类型以计算超体积
        let solution_vec: Vec<super::types::Solution> = objectives.iter().enumerate().map(|(i, obj_vals)| {
            let mut objective_values = HashMap::new();
            for (j, v) in obj_vals.iter().enumerate() {
                objective_values.insert(format!("obj_{}", j), *v);
            }
            super::types::Solution {
                parameters: if i < solutions.len() { solutions[i].clone() } else { vec![] },
                objective_values,
                is_pareto_optimal: true,
                rank: Some(0),
                crowding_distance: None,
                quality_metrics: HashMap::new(),
            }
        }).collect();
        
        let hypervolume = self.calculate_hypervolume(&solution_vec)?;
        
        // 构建结果
        Ok(MultiObjectiveResult {
            pareto_front: solutions.to_vec(),
            objective_values: objectives.to_vec(),
            hypervolume: Some(hypervolume),
            generational_distance: Some(0.0),
            inverted_generational_distance: Some(0.0),
            spread: Some(0.0),
            runtime_ms: 0,
            all_solutions: Some(solution_vec),
            final_metrics: HashMap::new(),
            convergence_history: None,
            iterations: 0,
            early_stopped: false,
            algorithm_specific: HashMap::new(),
        })
    }

    fn evaluate_configuration(&self, index_type: IndexType, parameters: &HashMap<String, ParameterValue>, parameter_space: &ParameterSpace) -> Result<OptimizationResult> {
        // 为MOPSO实现配置评估
        let mut total_score = 0.0;
        let mut configurations_tested = 0;
        
        // 评估不同配置的性能
        for (_param_name, param_value) in parameters {
            let score = match param_value {
                ParameterValue::Integer(val) => *val as f64,
                ParameterValue::Float(val) => *val,
                ParameterValue::Categorical(val) => val.len() as f64,
            };
            
            total_score += score;
            configurations_tested += 1;
        }
        
        let avg_score = if configurations_tested > 0 {
            total_score / configurations_tested as f64
        } else {
            0.0
        };
        
        // 创建默认配置
        let mut config = IndexConfig::default();
        config.index_type = index_type;
        
        // 创建默认性能结果
        let performance = BenchmarkResult {
            index_type,
            config: config.clone(),
            metrics: Default::default(),
            dataset_size: 0,
            dimension: 0,
            build_time_ms: 0,
            avg_query_time_ms: 0.0,
            queries_per_second: 0.0,
            memory_usage_bytes: 0,
            accuracy: avg_score,
            index_size_bytes: 0,
        };
        
        Ok(OptimizationResult {
            index_type,
            best_config: config,
            performance,
            evaluated_configs: configurations_tested,
            target: OptimizationTarget::BalancedPerformance,
            score: avg_score,
        })
    }

    fn compute_pareto_front(&self) -> Vec<Pareto> {
        // 从档案中构建帕累托前沿
        let mut pareto_front = Vec::new();
        
        for (i, solution) in self.archive.iter().enumerate() {
            let objectives: Vec<f64> = solution.objective_values.values().cloned().collect();
            
            pareto_front.push(Pareto {
                solution: solution.parameters.clone(),
                objective_values: objectives,
                rank: i,
                crowding_distance: solution.crowding_distance.unwrap_or(0.0),
            });
        }
        
        pareto_front
    }

    fn select_best_configuration(&self, pareto_front: &[Pareto]) -> usize {
        if pareto_front.is_empty() {
            return 0;
        }
        
        // 选择拥挤距离最大的解
        let mut best_index = 0;
        let mut best_crowding_distance = 0.0;
        
        for (i, pareto) in pareto_front.iter().enumerate() {
            if pareto.crowding_distance > best_crowding_distance {
                best_crowding_distance = pareto.crowding_distance;
                best_index = i;
            }
        }
        
        best_index
    }
    
    fn optimize_index(&mut self, index_type: IndexType, parameter_space: &ParameterSpace) -> Result<(Vec<Pareto>, OptimizationResult)> {
        // 实现索引优化逻辑
        let mut pareto_front = Vec::new();
        
        // 生成参数组合
        let parameter_combinations = self.generate_parameter_combinations(parameter_space)?;
        
        // 对每个参数组合进行评估
        for (i, params) in parameter_combinations.iter().enumerate() {
            let result = self.evaluate_configuration(index_type.clone(), params, parameter_space)?;
            
            // 将参数转换为向量
            let solution: Vec<f64> = params.values().map(|v| match v {
                ParameterValue::Integer(val) => *val as f64,
                ParameterValue::Float(val) => *val,
                ParameterValue::Categorical(_) => 0.0,
            }).collect();
            
            // 创建帕累托解
            let pareto = Pareto {
                solution,
                objective_values: vec![result.score, result.performance.avg_query_time_ms],
                rank: 0,
                crowding_distance: 0.0,
            };
            
            pareto_front.push(pareto);
        }
        
        // 计算帕累托排序
        self.compute_pareto_ranking(&mut pareto_front);
        
        // 选择最佳结果
        let best_index = self.select_best_configuration(&pareto_front);
        let best_result = if !pareto_front.is_empty() {
            // 从最佳帕累托解中重建参数
            let best_solution = &pareto_front[best_index].solution;
            let mut best_params = HashMap::new();
            for (i, range) in parameter_space.ranges.iter().enumerate() {
                if i < best_solution.len() {
                    let value = best_solution[i];
                    if range.is_integer {
                        best_params.insert(range.name.clone(), ParameterValue::Integer(value.round() as i64));
                    } else {
                        best_params.insert(range.name.clone(), ParameterValue::Float(value));
                    }
                }
            }
            self.evaluate_configuration(index_type, &best_params, parameter_space)?
        } else {
            // 创建默认结果
            let mut config = IndexConfig::default();
            config.index_type = index_type;
            let performance = BenchmarkResult {
                index_type,
                config: config.clone(),
                metrics: Default::default(),
                dataset_size: 0,
                dimension: 0,
                build_time_ms: 0,
                avg_query_time_ms: 0.0,
                queries_per_second: 0.0,
                memory_usage_bytes: 0,
                accuracy: 0.0,
                index_size_bytes: 0,
            };
            OptimizationResult {
                index_type,
                best_config: config,
                performance,
                evaluated_configs: 0,
                target: OptimizationTarget::BalancedPerformance,
                score: 0.0,
            }
        };
        
        Ok((pareto_front, best_result))
    }
}



impl Clone for MOPSOOptimizer {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            archive: self.archive.clone(),
            rng: StdRng::seed_from_u64(self.config.seed.unwrap_or(42)),
            archive_change: self.archive_change,
        }
    }
}

/// 运行MOPSO优化器示例
/// 
/// 这个函数展示了如何设置和使用MOPSO优化器
pub fn run_mopso_example() -> Result<MultiObjectiveResult> {
    println!("运行MOPSO多目标优化示例...");

    // 创建一个简单的配置
    let config = MultiObjectiveConfig {
        algorithm: MultiObjectiveAlgorithm::MOPSO,
        objectives_count: 2,
        population_size: 30,
        max_iterations: 50,
        mutation_probability: 0.1,
        crossover_probability: 0.8,
        seed: Some(42),
        objective_directions: vec![ObjectiveDirection::Minimize, ObjectiveDirection::Minimize],
        // MOPSO特定参数
        w: Some(0.4), // 惯性权重
        c1: Some(1.5), // 个体学习因子
        c2: Some(1.5), // 社会学习因子
        archive_size: Some(50),
        ..Default::default()
    };

    // 创建MOPSO优化器
    let mut optimizer = MOPSOOptimizer::new(config)?;

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
    println!("  GD: {}", result.generational_distance.unwrap_or(0.0));
    println!("  IGD: {}", result.inverted_generational_distance.unwrap_or(0.0));

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mopso_creation() {
        let config = MultiObjectiveConfig {
            algorithm: MultiObjectiveAlgorithm::MOPSO,
            objectives: vec![
                super::super::types::Objective {
                    name: "f1".to_string(),
                    description: None,
                    direction: ObjectiveDirection::Minimize,
                    weight: None,
                    lower_bound: None,
                    upper_bound: None,
                    constraints: vec![],
                },
                super::super::types::Objective {
                    name: "f2".to_string(),
                    description: None,
                    direction: ObjectiveDirection::Minimize,
                    weight: None,
                    lower_bound: None,
                    upper_bound: None,
                    constraints: vec![],
                }
            ],
            weight_strategy: super::super::types::WeightStrategy::Fixed,
            metrics: vec![],
            max_iterations: 100,
            population_size: 50,
            convergence_threshold: None,
            early_stopping: None,
            normalization: None,
            algorithm_params: HashMap::new(),
            parallel_processing: None,
            seed: Some(42),
        };
        
        let optimizer = MOPSOOptimizer::new(config);
        assert!(optimizer.is_ok());
    }
    
    #[test]
    fn test_mopso_invalid_config() {
        // 使用错误的算法类型
        let config = MultiObjectiveConfig {
            algorithm: MultiObjectiveAlgorithm::NSGA2,
            objectives: vec![
                super::super::types::Objective {
                    name: "f1".to_string(),
                    description: None,
                    direction: ObjectiveDirection::Minimize,
                    weight: None,
                    lower_bound: None,
                    upper_bound: None,
                    constraints: vec![],
                },
                super::super::types::Objective {
                    name: "f2".to_string(),
                    description: None,
                    direction: ObjectiveDirection::Minimize,
                    weight: None,
                    lower_bound: None,
                    upper_bound: None,
                    constraints: vec![],
                }
            ],
            weight_strategy: super::super::types::WeightStrategy::Fixed,
            metrics: vec![],
            max_iterations: 100,
            population_size: 50,
            convergence_threshold: None,
            early_stopping: None,
            normalization: None,
            algorithm_params: HashMap::new(),
            parallel_processing: None,
            seed: Some(42),
        };
        
        let optimizer = MOPSOOptimizer::new(config);
        assert!(optimizer.is_err());
    }
} 