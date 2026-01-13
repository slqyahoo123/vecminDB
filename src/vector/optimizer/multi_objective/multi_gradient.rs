// Copyright 2023 VecMind Technologies
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::error::{Result, Error};
use crate::vector::optimizer::multi_objective::types::{MultiObjectiveConfig, MultiObjectiveResult, Pareto, Solution, ObjectiveDirection};
use crate::vector::optimizer::multi_objective::MultiObjectiveOptimizer;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info};

/// 多目标梯度下降优化器
///
/// 多目标梯度下降算法（Multi-Objective Gradient Descent）利用目标函数的梯度信息
/// 引导搜索过程，适用于目标函数可微分的优化问题。该算法同时优化多个目标，
/// 通过梯度聚合和帕累托支配关系来平衡不同目标。
pub struct MultiGradientOptimizer {
    /// 优化配置
    config: MultiObjectiveConfig,
    /// 随机数生成器
    rng: StdRng,
    /// 当前解集
    solutions: Vec<Solution>,
    /// 梯度步长
    learning_rate: f64,
    /// 动量系数
    momentum: f64,
    /// 梯度历史
    gradient_history: Vec<Vec<f64>>,
}

impl MultiGradientOptimizer {
    /// 创建新的多目标梯度下降优化器
    pub fn new(config: MultiObjectiveConfig) -> Self {
        let seed = config.seed.unwrap_or_else(|| rand::random());
        let rng = StdRng::seed_from_u64(seed);
        
        // 从配置中提取算法特定参数
        let learning_rate = config.algorithm_params.get("learning_rate").cloned().unwrap_or(0.01);
        let momentum = config.algorithm_params.get("momentum").cloned().unwrap_or(0.9);
        
        Self {
            config,
            rng,
            solutions: Vec::new(),
            learning_rate,
            momentum,
            gradient_history: Vec::new(),
        }
    }
    
    /// 初始化解集
    fn initialize_solutions(&mut self, bounds: &[(f64, f64)]) -> Result<()> {
        self.solutions.clear();
        
        // 生成初始解
        for _ in 0..self.config.population_size {
            let mut parameters = Vec::with_capacity(bounds.len());
            
            // 随机生成参数值
            for &(lower, upper) in bounds {
                let value = self.rng.gen_range(lower..=upper);
                parameters.push(value);
            }
            
            // 创建解
            let solution = Solution {
                parameters,
                objective_values: HashMap::new(),
                is_pareto_optimal: false,
                rank: None,
                crowding_distance: None,
                quality_metrics: HashMap::new(),
            };
            
            self.solutions.push(solution);
        }
        
        // 初始化梯度历史
        self.gradient_history = vec![vec![0.0; bounds.len()]; self.solutions.len()];
        
        Ok(())
    }
    
    /// 计算梯度
    fn calculate_gradients(
        &self, 
        solution: &[f64], 
        objective_functions: &[Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>],
        bounds: &[(f64, f64)]
    ) -> Vec<Vec<f64>> {
        let n_objectives = objective_functions.len();
        let n_params = solution.len();
        let mut gradients = vec![vec![0.0; n_params]; n_objectives];
        
        // 梯度计算的步长
        let epsilon = 1e-6;
        
        // 为每个目标函数计算梯度
        for (obj_idx, objective_fn) in objective_functions.iter().enumerate() {
            let base_value = objective_fn(&solution.to_vec());
            
            // 为每个参数计算偏导数
            for param_idx in 0..n_params {
                // 创建扰动的参数向量
                let mut perturbed = solution.to_vec();
                perturbed[param_idx] += epsilon;
                
                // 在边界范围内调整
                if perturbed[param_idx] > bounds[param_idx].1 {
                    perturbed[param_idx] = bounds[param_idx].1;
                }
                
                // 计算扰动后的目标值
                let perturbed_value = objective_fn(&perturbed);
                
                // 计算梯度 (偏导数)
                let gradient = (perturbed_value - base_value) / epsilon;
                
                // 调整最小化/最大化方向
                let direction_factor = if self.config.objective_directions.get(obj_idx) == Some(&ObjectiveDirection::Minimize) {
                    1.0
                } else {
                    -1.0 // 对于最大化目标，梯度方向相反
                };
                
                gradients[obj_idx][param_idx] = gradient * direction_factor;
            }
        }
        
        gradients
    }
    
    /// 聚合多个目标的梯度
    fn aggregate_gradients(&self, gradients: &[Vec<f64>]) -> Vec<f64> {
        let n_objectives = gradients.len();
        let n_params = gradients[0].len();
        let mut aggregated = vec![0.0; n_params];
        
        // 使用权重策略聚合梯度
        match self.config.weight_strategy {
            crate::vector::optimizer::multi_objective::types::WeightStrategy::Fixed => {
                // 使用固定权重
                for obj_idx in 0..n_objectives {
                    let weight = if let Some(obj) = self.config.objectives.get(obj_idx) {
                        obj.weight.unwrap_or(1.0 / n_objectives as f64)
                    } else {
                        1.0 / n_objectives as f64
                    };
                    
                    for param_idx in 0..n_params {
                        aggregated[param_idx] += weight * gradients[obj_idx][param_idx];
                    }
                }
            },
            _ => {
                // 其他权重策略可在此扩展
                // 默认使用均等权重
                for obj_idx in 0..n_objectives {
                    let weight = 1.0 / n_objectives as f64;
                    
                    for param_idx in 0..n_params {
                        aggregated[param_idx] += weight * gradients[obj_idx][param_idx];
                    }
                }
            }
        }
        
        aggregated
    }
    
    /// 更新解
    fn update_solution(
        &mut self,
        solution_idx: usize,
        aggregated_gradient: Vec<f64>,
        bounds: &[(f64, f64)]
    ) -> Result<()> {
        if solution_idx >= self.solutions.len() {
            return Err(Error::invalid_argument("解索引超出范围"));
        }
        
        let solution = &mut self.solutions[solution_idx];
        let history = &mut self.gradient_history[solution_idx];
        
        // 对每个参数应用梯度更新
        for param_idx in 0..solution.parameters.len() {
            // 使用动量更新梯度
            history[param_idx] = self.momentum * history[param_idx] + (1.0 - self.momentum) * aggregated_gradient[param_idx];
            
            // 应用梯度更新
            solution.parameters[param_idx] -= self.learning_rate * history[param_idx];
            
            // 确保参数在边界范围内
            let (lower, upper) = bounds[param_idx];
            solution.parameters[param_idx] = solution.parameters[param_idx].clamp(lower, upper);
        }
        
        Ok(())
    }
    
    /// 评估解集
    fn evaluate_solutions(
        &mut self,
        objective_functions: &[Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>]
    ) -> Result<()> {
        for solution in &mut self.solutions {
            // 创建目标值映射
            let mut objective_values = HashMap::new();
            
            // 对每个目标函数计算值
            for (idx, objective_fn) in objective_functions.iter().enumerate() {
                let value = objective_fn(&solution.parameters);
                
                // 获取目标名称
                let name = if let Some(obj) = self.config.objectives.get(idx) {
                    obj.name.clone()
                } else {
                    format!("objective_{}", idx)
                };
                
                objective_values.insert(name, value);
            }
            
            solution.objective_values = objective_values;
        }
        
        // 更新解的帕累托最优性
        self.update_pareto_optimality()?;
        
        Ok(())
    }
    
    /// 更新解的帕累托最优性
    fn update_pareto_optimality(&mut self) -> Result<()> {
        if self.solutions.is_empty() {
            return Ok(());
        }
        
        // 复位所有解的帕累托最优标志
        for solution in &mut self.solutions {
            solution.is_pareto_optimal = true;
        }
        
        // 非支配排序
        for i in 0..self.solutions.len() {
            if !self.solutions[i].is_pareto_optimal {
                continue;
            }
            
            for j in 0..self.solutions.len() {
                if i == j || !self.solutions[j].is_pareto_optimal {
                    continue;
                }
                
                // 检查j是否被i支配
                let mut i_dominates_j = true;
                let mut j_dominates_i = true;
                
                for (idx, direction) in self.config.objective_directions.iter().enumerate() {
                    let obj_name = if let Some(obj) = self.config.objectives.get(idx) {
                        obj.name.clone()
                    } else {
                        format!("objective_{}", idx)
                    };
                    
                    let i_value = *self.solutions[i].objective_values.get(&obj_name).unwrap_or(&0.0);
                    let j_value = *self.solutions[j].objective_values.get(&obj_name).unwrap_or(&0.0);
                    
                    match direction {
                        ObjectiveDirection::Minimize => {
                            if i_value > j_value {
                                i_dominates_j = false;
                            }
                            if j_value > i_value {
                                j_dominates_i = false;
                            }
                        },
                        ObjectiveDirection::Maximize => {
                            if i_value < j_value {
                                i_dominates_j = false;
                            }
                            if j_value < i_value {
                                j_dominates_i = false;
                            }
                        }
                    }
                }
                
                if i_dominates_j {
                    self.solutions[j].is_pareto_optimal = false;
                }
                
                if j_dominates_i {
                    self.solutions[i].is_pareto_optimal = false;
                    break;
                }
            }
        }
        
        Ok(())
    }
    
    /// 构建帕累托前沿
    fn build_pareto_front(&self) -> Result<Vec<Pareto>> {
        let mut pareto_front = Vec::new();
        
        for solution in &self.solutions {
            if solution.is_pareto_optimal {
                // 构建帕累托解
                let objective_values = solution.objective_values.values().cloned().collect();

                let pareto = Pareto {
                    solution: solution.parameters.clone(),
                    objective_values,
                    rank: 0,
                    crowding_distance: 0.0,
                };

                pareto_front.push(pareto);
            }
        }
        
        Ok(pareto_front)
    }
}

impl MultiObjectiveOptimizer for MultiGradientOptimizer {
    fn optimize(
        &mut self, 
        objective_functions: &[Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>],
        bounds: &[(f64, f64)]
    ) -> Result<MultiObjectiveResult> {
        info!("开始多目标梯度下降优化");
        
        // 初始化解集
        self.initialize_solutions(bounds)?;
        
        // 开始计时
        let start_time = Instant::now();
        
        // 初始评估解集
        self.evaluate_solutions(objective_functions)?;
        
        // 迭代优化
        let mut early_stopped = false;
        let mut iterations = 0;
        let mut convergence_history = Vec::new();
        
        // 记录最佳超体积
        let mut best_hypervolume = 0.0;
        let mut no_improvement_count = 0;
        
        for iter in 0..self.config.max_iterations {
            debug!("多目标梯度下降迭代 {}/{}", iter + 1, self.config.max_iterations);
            
            // 对每个解进行梯度更新
            for i in 0..self.solutions.len() {
                // 计算所有目标的梯度
                let gradients = self.calculate_gradients(
                    &self.solutions[i].parameters,
                    objective_functions,
                    bounds,
                );
                
                // 聚合梯度
                let aggregated = self.aggregate_gradients(&gradients);
                
                // 更新解
                self.update_solution(i, aggregated, bounds)?;
            }
            
            // 评估更新后的解集
            self.evaluate_solutions(objective_functions)?;
            
            // 计算当前迭代的超体积
            let hypervolume = self.calculate_hypervolume()?;
            convergence_history.push(hypervolume);
            
            // 检查是否需要提前停止
            if let Some(ref early_stopping) = self.config.early_stopping {
                if early_stopping.enabled {
                    if hypervolume > best_hypervolume + early_stopping.min_delta {
                        best_hypervolume = hypervolume;
                        no_improvement_count = 0;
                    } else {
                        no_improvement_count += 1;
                        if no_improvement_count >= early_stopping.patience {
                            info!("多目标梯度下降算法提前停止，连续{}次迭代没有改善", no_improvement_count);
                            early_stopped = true;
                            break;
                        }
                    }
                }
            }
            
            iterations = iter + 1;
        }
        
        // 构建帕累托前沿
        let pareto_solutions = self.build_pareto_front()?;
        
        // 提取帕累托前沿和目标值
        let mut pareto_front = Vec::new();
        let mut objective_values = Vec::new();
        
        for solution in &self.solutions {
            if solution.is_pareto_optimal {
                pareto_front.push(solution.parameters.clone());
                
                let mut objectives = Vec::new();
                for idx in 0..self.config.objectives_count {
                    let obj_name = if let Some(obj) = self.config.objectives.get(idx) {
                        obj.name.clone()
                    } else {
                        format!("objective_{}", idx)
                    };
                    
                    objectives.push(*solution.objective_values.get(&obj_name).unwrap_or(&0.0));
                }
                
                objective_values.push(objectives);
            }
        }
        
        // 计算帕累托指标
        let hypervolume = Some(self.calculate_hypervolume()?);
        let generational_distance = Some(self.calculate_generational_distance()?);
        let inverted_generational_distance = Some(self.calculate_inverted_generational_distance()?);
        let spread = Some(self.calculate_spread()?);
        
        // 构建结果
        let runtime_ms = start_time.elapsed().as_millis() as u64;
        
        let mut result = MultiObjectiveResult {
            pareto_front,
            objective_values,
            hypervolume,
            generational_distance,
            inverted_generational_distance,
            spread,
            runtime_ms,
            all_solutions: Some(self.solutions.clone()),
            final_metrics: HashMap::new(),
            convergence_history: Some(convergence_history),
            iterations,
            early_stopped,
            algorithm_specific: HashMap::new(),
        };
        
        // 添加算法特定结果
        result.algorithm_specific.insert(
            "learning_rate".to_string(),
            serde_json::to_value(self.learning_rate).unwrap_or_default()
        );
        
        result.algorithm_specific.insert(
            "momentum".to_string(),
            serde_json::to_value(self.momentum).unwrap_or_default()
        );
        
        Ok(result)
    }
    
    fn get_config(&self) -> &MultiObjectiveConfig {
        &self.config
    }
    
    fn validate_solution(&self, solution: &Vec<f64>, bounds: &[(f64, f64)]) -> Result<bool> {
        if solution.len() != bounds.len() {
            return Err(Error::invalid_argument("解的维度与边界不匹配"));
        }
        
        // 检查所有参数是否在边界范围内
        for (i, &value) in solution.iter().enumerate() {
            let (lower, upper) = bounds[i];
            if value < lower || value > upper {
                return Ok(false);
            }
        }
        
        Ok(true)
    }

    fn calculate_dominance(
        &self,
        _solution1_objectives: &[f64],
        _solution2_objectives: &[f64],
    ) -> i32 {
        // 简化处理：未提供支配度计算，返回0表示不可比
        0
    }

    fn calculate_metrics(&self, solutions: &[Vec<f64>], objectives: &[Vec<f64>]) -> Result<MultiObjectiveResult> {
        // 计算帕累托前沿
        let pareto_solutions = self.build_pareto_front()?;
        let pareto_front: Vec<Vec<f64>> = pareto_solutions.iter().map(|p| p.solution.clone()).collect();
        
        // 计算超体积
        let hypervolume = Some(self.calculate_hypervolume()?);
        let generational_distance = Some(self.calculate_generational_distance()?);
        let inverted_generational_distance = Some(self.calculate_inverted_generational_distance()?);
        let spread = Some(self.calculate_spread()?);
        
        // 构建指标
        let mut final_metrics = HashMap::new();
        final_metrics.insert("pareto_front_size".to_string(), pareto_front.len() as f64);
        if let Some(hv) = hypervolume {
            final_metrics.insert("hypervolume".to_string(), hv);
        }
        
        Ok(MultiObjectiveResult {
            pareto_front,
            objective_values: objectives.to_vec(),
            hypervolume,
            generational_distance,
            inverted_generational_distance,
            spread,
            runtime_ms: 0,
            all_solutions: None,
            final_metrics,
            convergence_history: None,
            iterations: 0,
            early_stopped: false,
            algorithm_specific: HashMap::new(),
        })
    }

    fn evaluate_configuration(
        &self,
        index_type: crate::vector::index::types::IndexType,
        parameters: &std::collections::HashMap<String, crate::vector::optimizer::ParameterValue>,
        parameter_space: &crate::vector::optimizer::ParameterSpace,
    ) -> Result<crate::vector::optimizer::OptimizationResult> {
        use crate::vector::index::types::IndexConfig;
        use crate::vector::benchmark::{IndexBenchmark, BenchmarkConfig};
        use crate::vector::optimizer::{OptimizationResult, OptimizationTarget};
        
        // 检查参数是否有效
        for (name, _value) in parameters {
            let exists = parameter_space
                .parameters()
                .any(|p| p.name() == name.as_str());
            if !exists {
                return Err(Error::InvalidArgument(format!("Unknown parameter: {}", name)));
            }
        }
        
        // 将参数映射到索引配置
        let mut index_config = IndexConfig::default();
        index_config.index_type = index_type;
        
        for (name, value) in parameters {
            let v_usize = match value {
                crate::vector::optimizer::ParameterValue::Integer(i) => *i as usize,
                crate::vector::optimizer::ParameterValue::Float(f) => *f as usize,
                crate::vector::optimizer::ParameterValue::Categorical(_) => continue,
            };
            index_config.set_param(name, v_usize);
        }
        
        // 使用基准测试器评估配置
        let bench_config = BenchmarkConfig::default();
        let benchmark = IndexBenchmark::new(bench_config);
        let bench_result = benchmark.benchmark_index(index_type, index_config.clone())?;
        
        // 计算综合评分
        let time_term = bench_result.avg_query_time_ms.max(1e-6);
        let size_term = (bench_result.index_size_bytes as f64 / 1_000_000.0).max(1e-6);
        let score = 1.0 / (time_term + size_term);
        
        Ok(OptimizationResult {
            index_type,
            best_config: index_config,
            performance: bench_result,
            evaluated_configs: 1,
            target: OptimizationTarget::BalancedPerformance,
            score,
        })
    }

    fn compute_pareto_front(&self) -> Vec<crate::vector::optimizer::multi_objective::types::Pareto> {
        // Pareto类型已通过模块导入，这里不需要重复导入
        // use crate::vector::optimizer::multi_objective::types::Pareto;
        
        if let Ok(pareto_solutions) = self.build_pareto_front() {
            pareto_solutions
        } else {
            Vec::new()
        }
    }

    fn select_best_configuration(&self, pareto_front: &[crate::vector::optimizer::multi_objective::types::Pareto]) -> usize {
        if pareto_front.is_empty() {
            return 0;
        }
        
        // 使用加权和方法选择最佳配置（平衡所有目标）
        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = 0;
        
        // 归一化目标值
        let obj_count = if let Some(first) = pareto_front.first() {
            first.objective_values.len()
        } else {
            return 0;
        };
        
        let mut min_vals = vec![f64::INFINITY; obj_count];
        let mut max_vals = vec![f64::NEG_INFINITY; obj_count];
        
        for pareto in pareto_front {
            for (i, &val) in pareto.objective_values.iter().enumerate() {
                min_vals[i] = min_vals[i].min(val);
                max_vals[i] = max_vals[i].max(val);
            }
        }
        
        // 计算加权和
        for (i, pareto) in pareto_front.iter().enumerate() {
            let mut score = 0.0;
            for (j, &val) in pareto.objective_values.iter().enumerate() {
                let range = (max_vals[j] - min_vals[j]).max(1e-10);
                let normalized = 1.0 - (val - min_vals[j]) / range; // 假设最小化
                let weight = 1.0 / obj_count as f64;
                score += weight * normalized;
            }
            
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        
        best_idx
    }

    fn optimize_index(
        &mut self,
        index_type: crate::vector::index::types::IndexType,
        parameter_space: &crate::vector::optimizer::ParameterSpace,
    ) -> Result<(Vec<crate::vector::optimizer::multi_objective::types::Pareto>, crate::vector::optimizer::OptimizationResult)> {
        use crate::vector::optimizer::{OptimizationResult, OptimizationTarget};
        use crate::vector::index::types::IndexConfig;
        use crate::vector::benchmark::{IndexBenchmark, BenchmarkConfig};
        use std::time::Instant;
        
        let start_time = Instant::now();
        
        // 从参数空间构建边界
        let mut bounds = Vec::new();
        let mut param_names = Vec::new();
        for param in parameter_space.parameters() {
            match param.parameter_type() {
                crate::vector::optimizer::parameter_space::ParameterType::Integer(min, max) => {
                    bounds.push((*min as f64, *max as f64));
                    param_names.push(param.name().to_string());
                },
                crate::vector::optimizer::parameter_space::ParameterType::Float(min, max) => {
                    bounds.push((*min, *max));
                    param_names.push(param.name().to_string());
                },
                crate::vector::optimizer::parameter_space::ParameterType::Categorical(_) => {
                    // 类别参数暂时跳过
                },
            }
        }
        
        if bounds.is_empty() {
            return Err(Error::InvalidArgument("参数空间为空".to_string()));
        }
        
        // 构建目标函数（查询时间和索引大小）
        let mut eval_results: Vec<OptimizationResult> = Vec::new();
        let mut evaluated_configs = Vec::new();
        
        // 生成并评估初始配置
        for _ in 0..self.config.population_size {
            let mut params = std::collections::HashMap::new();
            let mut param_vector = Vec::new();
            
            for (i, param_name) in param_names.iter().enumerate() {
                if i >= bounds.len() {
                    break;
                }
                let (min, max) = bounds[i];
                let value = self.rng.gen_range(min..=max);
                param_vector.push(value);
                
                // 转换为参数值
                if let Some(param) = parameter_space.parameters().find(|p| p.name() == param_name.as_str()) {
                    match param.parameter_type() {
                        crate::vector::optimizer::parameter_space::ParameterType::Integer(_, _) => {
                            params.insert(param_name.clone(), 
                                crate::vector::optimizer::ParameterValue::Integer(value as i64));
                        },
                        crate::vector::optimizer::parameter_space::ParameterType::Float(_, _) => {
                            params.insert(param_name.clone(), 
                                crate::vector::optimizer::ParameterValue::Float(value));
                        },
                        _ => {},
                    }
                }
            }
            
            let result = self.evaluate_configuration(index_type, &params, parameter_space)?;
            eval_results.push(result);
            evaluated_configs.push((param_vector, vec![
                eval_results.last().unwrap().performance.avg_query_time_ms,
                eval_results.last().unwrap().performance.index_size_bytes as f64,
            ]));
        }
        
        // 使用 Arc 共享 evaluated_configs
        use std::sync::Arc;
        let evaluated_configs_arc = Arc::new(evaluated_configs);
        
        // 构建目标函数（查询时间和索引大小）
        let objective_functions: Vec<Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>> = vec![
            Box::new({
                let evaluated_configs = evaluated_configs_arc.clone();
                move |params: &Vec<f64>| {
                    // 查找对应的评估结果
                    for (eval_params, objectives) in evaluated_configs.iter() {
                        if eval_params == params {
                            return objectives[0];
                        }
                    }
                    f64::INFINITY
                }
            }),
            Box::new({
                let evaluated_configs = evaluated_configs_arc.clone();
                move |params: &Vec<f64>| {
                    for (eval_params, objectives) in evaluated_configs.iter() {
                        if eval_params == params {
                            return objectives[1];
                        }
                    }
                    f64::INFINITY
                }
            }),
        ];
        
        // 运行优化
        let multi_result = self.optimize(&objective_functions, &bounds)?;
        
        // 构建帕累托前沿
        let pareto_front = self.compute_pareto_front();
        
        // 选择最佳配置（从帕累托前沿中找到对应的评估结果）
        let best_idx = self.select_best_configuration(&pareto_front);
        let best_result = if !pareto_front.is_empty() {
            // 尝试找到与最佳帕累托解对应的评估结果
            let best_pareto = &pareto_front[best_idx];
            let mut found = false;
            let mut result = None;
            
            for (i, eval_result) in eval_results.iter().enumerate() {
                // 检查参数是否匹配（简化比较）
                if i < evaluated_configs_arc.len() {
                    let (eval_params, _) = &evaluated_configs_arc[i];
                    if eval_params.len() == best_pareto.solution.len() {
                        let mut matches = true;
                        for (j, &val) in eval_params.iter().enumerate() {
                            if (val - best_pareto.solution[j]).abs() > 1e-6 {
                                matches = false;
                                break;
                            }
                        }
                        if matches {
                            result = Some(eval_result.clone());
                            found = true;
                            break;
                        }
                    }
                }
            }
            
            if found {
                result.unwrap()
            } else if !eval_results.is_empty() {
                eval_results.last().unwrap().clone()
            } else {
                // 创建默认结果
                let mut index_config = IndexConfig::default();
                index_config.index_type = index_type;
                let bench_config = BenchmarkConfig::default();
                let benchmark = IndexBenchmark::new(bench_config);
                let bench_result = benchmark.benchmark_index(index_type, index_config.clone())?;
                
                OptimizationResult {
                    index_type,
                    best_config: index_config,
                    performance: bench_result,
                    evaluated_configs: evaluated_configs_arc.len(),
                    target: OptimizationTarget::BalancedPerformance,
                    score: 0.0,
                }
            }
        } else if !eval_results.is_empty() {
            eval_results.last().unwrap().clone()
        } else {
            // 创建默认结果
            let mut index_config = IndexConfig::default();
            index_config.index_type = index_type;
            let bench_config = BenchmarkConfig::default();
            let benchmark = IndexBenchmark::new(bench_config);
            let bench_result = benchmark.benchmark_index(index_type, index_config.clone())?;
            
            OptimizationResult {
                index_type,
                best_config: index_config,
                performance: bench_result,
                evaluated_configs: evaluated_configs_arc.len(),
                target: OptimizationTarget::BalancedPerformance,
                score: 0.0,
            }
        };
        
        Ok((pareto_front, best_result))
    }
}

impl MultiGradientOptimizer {
    /// 计算超体积指标
    fn calculate_hypervolume(&self) -> Result<f64> {
        // 检查是否有帕累托最优解
        if self.solutions.iter().filter(|s| s.is_pareto_optimal).count() == 0 {
            return Ok(0.0);
        }
        
        // 提取帕累托最优解的目标值
        let pareto_solutions: Vec<&Solution> = self.solutions.iter()
            .filter(|s| s.is_pareto_optimal)
            .collect();
            
        // 获取目标维度
        let obj_count = self.config.objectives_count;
        if obj_count == 0 {
            return Err(Error::invalid_argument("目标数量不能为零"));
        }
        
        // 确定参考点（对于最小化问题，使用每个目标维度的最大值；对于最大化问题，使用每个目标维度的最小值）
        let mut reference_point = vec![0.0; obj_count];
        
        for obj_idx in 0..obj_count {
            let obj_key = format!("objective_{}", obj_idx);
            
            if self.config.objective_directions.get(obj_idx) == Some(&ObjectiveDirection::Minimize) {
                // 对于最小化目标，参考点应该大于所有解
                reference_point[obj_idx] = pareto_solutions.iter()
                    .filter_map(|s| s.objective_values.get(&obj_key))
                    .fold(std::f64::NEG_INFINITY, |a, &b| a.max(b)) * 1.1; // 增加10%的余量
            } else {
                // 对于最大化目标，参考点应该小于所有解
                reference_point[obj_idx] = pareto_solutions.iter()
                    .filter_map(|s| s.objective_values.get(&obj_key))
                    .fold(std::f64::INFINITY, |a, &b| a.min(b)) * 0.9; // 减少10%的余量
            }
        }
        
        // 构建目标值矩阵（规范化为最小化问题）
        let mut objective_matrix: Vec<Vec<f64>> = Vec::with_capacity(pareto_solutions.len());
        for solution in &pareto_solutions {
            let mut obj_values = Vec::with_capacity(obj_count);
            
            for obj_idx in 0..obj_count {
                let obj_key = format!("objective_{}", obj_idx);
                let obj_value = solution.objective_values.get(&obj_key).cloned().unwrap_or(0.0);
                
                // 对于最大化目标，取反转化为最小化问题
                if self.config.objective_directions.get(obj_idx) == Some(&ObjectiveDirection::Maximize) {
                    obj_values.push(-obj_value);
                } else {
                    obj_values.push(obj_value);
                }
            }
            
            objective_matrix.push(obj_values);
        }
        
        // 计算超体积（使用Monte Carlo采样方法进行近似计算）
        const SAMPLES: usize = 10000;
        let mut inside_count = 0;
        
        let mut rng = rand::thread_rng();
        
        for _ in 0..SAMPLES {
            // 在参考点和理想点之间随机采样
            let mut sample_point = Vec::with_capacity(obj_count);
            for obj_idx in 0..obj_count {
                let min_val = objective_matrix.iter()
                    .map(|v| v[obj_idx])
                    .fold(std::f64::INFINITY, |a, b| a.min(b));
                
                let range_min = min_val;
                let range_max = if self.config.objective_directions.get(obj_idx) == Some(&ObjectiveDirection::Maximize) {
                    -reference_point[obj_idx]
                } else {
                    reference_point[obj_idx]
                };
                
                let sample = rng.gen_range(range_min..=range_max);
                sample_point.push(sample);
            }
            
            // 检查该点是否被帕累托集支配
            let is_dominated = objective_matrix.iter().any(|solution| {
                // 检查是否每个目标上都不劣于采样点，且至少一个目标上优于采样点
                let mut is_better = false;
                let mut is_not_worse = true;
                
                for obj_idx in 0..obj_count {
                    if solution[obj_idx] < sample_point[obj_idx] {
                        is_better = true;
                    } else if solution[obj_idx] > sample_point[obj_idx] {
                        is_not_worse = false;
                        break;
                    }
                }
                
                is_not_worse && is_better
            });
            
            if is_dominated {
                inside_count += 1;
            }
        }
        
        // 计算超体积
        let total_volume = (0..obj_count).map(|obj_idx| {
            let min_val = objective_matrix.iter()
                .map(|v| v[obj_idx])
                .fold(std::f64::INFINITY, |a, b| a.min(b));
                
            let range = if self.config.objective_directions.get(obj_idx) == Some(&ObjectiveDirection::Maximize) {
                -reference_point[obj_idx] - min_val
            } else {
                reference_point[obj_idx] - min_val
            };
            
            range
        }).product::<f64>();
        
        let hypervolume = total_volume * (inside_count as f64 / SAMPLES as f64);
        
        Ok(hypervolume)
    }
    
    /// 计算世代距离
    fn calculate_generational_distance(&self) -> Result<f64> {
        // 检查是否有帕累托最优解
        let pareto_solutions: Vec<&Solution> = self.solutions.iter()
            .filter(|s| s.is_pareto_optimal)
            .collect();
            
        if pareto_solutions.is_empty() {
            return Ok(0.0);
        }
        
        // 在实际项目中，这里需要与真实的帕累托前沿进行比较
        // 对于没有已知真实帕累托前沿的问题，可以使用近似方法或参考集
        
        // 简化版实现：假设当前的帕累托集就是参考集，返回0
        Ok(0.0)
    }
    
    /// 计算反向世代距离
    fn calculate_inverted_generational_distance(&self) -> Result<f64> {
        // 类似于世代距离，但是计算方向相反
        // 在实际项目中，这里需要与真实的帕累托前沿进行比较
        
        // 简化版实现：假设当前的帕累托集就是参考集，返回0
        Ok(0.0)
    }
    
    /// 计算分散度指标
    fn calculate_spread(&self) -> Result<f64> {
        // 检查是否有帕累托最优解
        let pareto_solutions: Vec<&Solution> = self.solutions.iter()
            .filter(|s| s.is_pareto_optimal)
            .collect();
            
        if pareto_solutions.len() < 2 {
            return Ok(0.0); // 至少需要两个解才能计算分散度
        }
        
        let obj_count = self.config.objectives_count;
        if obj_count == 0 {
            return Err(Error::invalid_argument("目标数量不能为零"));
        }
        
        // 计算解之间的欧几里得距离
        let mut distances = Vec::new();
        
        for i in 0..pareto_solutions.len() {
            for j in (i+1)..pareto_solutions.len() {
                let mut squared_distance = 0.0;
                
                for obj_idx in 0..obj_count {
                    let obj_key = format!("objective_{}", obj_idx);
                    
                    let value_i = pareto_solutions[i].objective_values.get(&obj_key).cloned().unwrap_or(0.0);
                    let value_j = pareto_solutions[j].objective_values.get(&obj_key).cloned().unwrap_or(0.0);
                    
                    squared_distance += (value_i - value_j).powi(2);
                }
                
                distances.push(squared_distance.sqrt());
            }
        }
        
        // 计算平均距离
        if distances.is_empty() {
            return Ok(0.0);
        }
        
        let mean_distance = distances.iter().sum::<f64>() / distances.len() as f64;
        
        // 计算分散度（使用变异系数：标准差/平均值）
        let variance = distances.iter()
            .map(|&d| (d - mean_distance).powi(2))
            .sum::<f64>() / distances.len() as f64;
            
        let std_deviation = variance.sqrt();
        
        // 分散度指标
        let spread = if mean_distance > 0.0 {
            std_deviation / mean_distance
        } else {
            0.0
        };
        
        Ok(spread)
    }
} 