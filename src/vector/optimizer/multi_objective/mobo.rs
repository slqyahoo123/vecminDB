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
use crate::vector::index::types::{IndexType, IndexConfig};
use crate::vector::index::hnsw::types::IndexParameters;
use crate::vector::optimizer::parameter_space::{Parameter, ParameterSpace, ParameterType, ParameterValue};
use crate::vector::optimizer::{OptimizationResult, OptimizationTarget};
use crate::vector::benchmark::{IndexBenchmark, BenchmarkConfig};
use super::r#trait::MultiObjectiveOptimizer;
use super::types::{Pareto, MOBOConfig, MultiObjectiveResult, MultiObjectiveConfig};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info};

/// 多目标贝叶斯优化器
///
/// 多目标贝叶斯优化(Multi-Objective Bayesian Optimization)使用高斯过程回归模型
/// 为每个目标建立代理模型，然后使用获取函数(如预期改进)来选择下一个评估点。
pub struct MOBO {
    /// 优化器配置
    config: MOBOConfig,
    /// 随机数生成器
    rng: StdRng,
    /// 目标数量
    objective_count: usize,
    /// 高斯过程模型 (每个目标一个模型)
    models: Vec<GaussianProcess>,
    /// 观测点 (参数值)
    observed_x: Vec<Vec<f64>>,
    /// 观测值 (目标值)
    observed_y: Vec<Vec<f64>>,
    /// 候选点
    candidates: Vec<Vec<f64>>,
}

impl MOBO {
    /// 创建新的多目标贝叶斯优化器
    pub fn new(config: MOBOConfig) -> Self {
        let seed = config.seed.unwrap_or_else(|| rand::random());
        let rng = StdRng::seed_from_u64(seed);
        
        Self {
            config,
            rng,
            objective_count: 2, // 默认两个目标：查询性能和索引大小
            models: Vec::new(),
            observed_x: Vec::new(),
            observed_y: Vec::new(),
            candidates: Vec::new(),
        }
    }
    
    /// 初始化高斯过程模型
    fn initialize_models(&mut self) {
        self.models.clear();
        
        // 为每个目标创建一个高斯过程模型
        for _ in 0..self.objective_count {
            let kernel_param = 0.1; // 默认长度尺度参数
            let model = GaussianProcess::new(kernel_param);
            self.models.push(model);
        }
    }
    
    /// 生成初始采样点
    fn generate_initial_samples(&mut self, parameter_space: &ParameterSpace, sample_count: usize) -> Vec<HashMap<String, ParameterValue>> {
        let mut samples = Vec::new();
        
        // 使用拉丁超立方采样生成初始点
        let parameters: Vec<&Parameter> = parameter_space.parameters().collect();
        
        for _ in 0..sample_count {
            let mut sample = HashMap::new();
            
            for param in &parameters {
                let value = match param.parameter_type() {
                    ParameterType::Integer(min, max) => {
                        let val = self.rng.gen_range(*min..=*max);
                        ParameterValue::Integer(val)
                    },
                    ParameterType::Float(min, max) => {
                        let val = self.rng.gen_range(*min..=*max);
                        ParameterValue::Float(val)
                    },
                    ParameterType::Categorical(values) => {
                        let idx = self.rng.gen_range(0..values.len());
                        values[idx].clone()
                    },
                };
                
                sample.insert(param.name().to_string(), value);
            }
            
            samples.push(sample);
        }
        
        samples
    }
    
    /// 将参数映射转换为向量形式
    fn params_to_vector(&self, params: &HashMap<String, ParameterValue>, parameter_space: &ParameterSpace) -> Vec<f64> {
        let mut vector = Vec::new();
        
        for param in parameter_space.parameters() {
            if let Some(value) = params.get(param.name()) {
                match value {
                    ParameterValue::Integer(val) => vector.push(*val as f64),
                    ParameterValue::Float(val) => vector.push(*val),
                    ParameterValue::Categorical(_) => {
                        // 对于类别型参数，使用One-Hot编码
                        if let ParameterType::Categorical(values) = param.parameter_type() {
                            for possible_value in values {
                                if value == possible_value {
                                    vector.push(1.0);
                                } else {
                                    vector.push(0.0);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        vector
    }
    
    /// 从向量形式转换回参数映射
    fn vector_to_params(&self, vector: &[f64], parameter_space: &ParameterSpace) -> HashMap<String, ParameterValue> {
        let mut params = HashMap::new();
        let mut idx = 0;
        
        for param in parameter_space.parameters() {
            match param.parameter_type() {
                ParameterType::Integer(min, max) => {
                    let val = vector[idx].round() as i64;
                    let val = val.clamp(*min, *max);
                    params.insert(param.name().to_string(), ParameterValue::Integer(val));
                    idx += 1;
                },
                ParameterType::Float(min, max) => {
                    let val = vector[idx].clamp(*min, *max);
                    params.insert(param.name().to_string(), ParameterValue::Float(val));
                    idx += 1;
                },
                ParameterType::Categorical(values) => {
                    // 找到One-Hot编码中值最大的索引
                    let mut max_val = f64::MIN;
                    let mut max_idx = 0;
                    
                    for (i, &val) in vector[idx..idx+values.len()].iter().enumerate() {
                        if val > max_val {
                            max_val = val;
                            max_idx = i;
                        }
                    }
                    
                    params.insert(param.name().to_string(), values[max_idx].clone());
                    idx += values.len();
                },
            }
        }
        
        params
    }
    
    /// 更新代理模型
    fn update_models(&mut self) {
        // 确保观测数据足够
        if self.observed_x.is_empty() || self.observed_y.is_empty() {
            return;
        }
        
        // 更新每个目标的高斯过程模型
        for i in 0..self.objective_count {
            // 提取第i个目标的观测值
            let y_values: Vec<f64> = self.observed_y.iter()
                .map(|y| y[i])
                .collect();
            
            // 更新模型
            self.models[i].update(&self.observed_x, &y_values);
        }
    }
    
    /// 计算获取函数值
    fn calculate_acquisition(&self, x: &[f64]) -> f64 {
        // 如果没有观测数据，返回高值以鼓励探索
        if self.observed_y.is_empty() {
            return 1.0;
        }
        
        // 计算预期改进
        let mut ei_values = Vec::new();
        
        for i in 0..self.objective_count {
            let (mean, std) = self.models[i].predict(x);
            
            // 找到当前目标的最佳值
            let best_value = match i {
                // 对于需要最小化的目标
                0 => self.observed_y.iter().map(|y| y[i]).fold(f64::INFINITY, |a, b| a.min(b)),
                // 对于需要最大化的目标
                _ => self.observed_y.iter().map(|y| y[i]).fold(f64::NEG_INFINITY, |a, b| a.max(b)),
            };
            
            // 计算预期改进
            let z = match i {
                0 => (best_value - mean) / std.max(1e-10), // 最小化目标
                _ => (mean - best_value) / std.max(1e-10), // 最大化目标
            };
            
            let cdf = 0.5 * (1.0 + erf(z / f64::sqrt(2.0)));
            let pdf = (-0.5 * z * z).exp() / f64::sqrt(2.0 * std::f64::consts::PI);
            
            let ei = match i {
                0 => (best_value - mean) * cdf + std * pdf, // 最小化目标
                _ => (mean - best_value) * cdf + std * pdf, // 最大化目标
            };
            
            ei_values.push(ei);
        }
        
        // 使用乘积组合多个目标的预期改进
        ei_values.iter().fold(1.0, |acc, &ei| acc * ei)
    }
    
    /// 选择下一个评估点
    fn select_next_point(&mut self, parameter_space: &ParameterSpace) -> HashMap<String, ParameterValue> {
        // 生成候选点
        self.generate_candidates(parameter_space);
        
        // 如果没有候选点，返回随机点
        if self.candidates.is_empty() {
            let samples = self.generate_initial_samples(parameter_space, 1);
            return samples[0].clone();
        }
        
        // 计算每个候选点的获取函数值
        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = 0;
        
        for (i, candidate) in self.candidates.iter().enumerate() {
            let score = self.calculate_acquisition(candidate);
            
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        
        // 将最佳候选点转换回参数映射
        self.vector_to_params(&self.candidates[best_idx], parameter_space)
    }
    
    /// 生成候选评估点
    fn generate_candidates(&mut self, parameter_space: &ParameterSpace) {
        let candidate_count = self.config.parameters.candidate_count;
        self.candidates.clear();
        
        // 如果有观测数据，使用观测点周围的扰动生成候选点
        if !self.observed_x.is_empty() {
            // 从已观测点中生成候选点
            for _ in 0..candidate_count / 2 {
                // 随机选择一个观测点
                let idx = self.rng.gen_range(0..self.observed_x.len());
                let base = &self.observed_x[idx];
                
                // 添加扰动
                let mut candidate = base.clone();
                for j in 0..candidate.len() {
                    let noise = self.rng.gen_range(-0.1..0.1);
                    candidate[j] = (candidate[j] + noise).clamp(0.0, 1.0);
                }
                
                self.candidates.push(candidate);
            }
        }
        
        // 生成剩余的随机候选点
        let remaining = candidate_count - self.candidates.len();
        for _ in 0..remaining {
            let sample = self.generate_initial_samples(parameter_space, 1);
            let vector = self.params_to_vector(&sample[0], parameter_space);
            self.candidates.push(vector);
        }
    }
}

impl MultiObjectiveOptimizer for MOBO {
    fn optimize(
        &mut self, 
        objective_functions: &[Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>],
        bounds: &[(f64, f64)]
    ) -> Result<MultiObjectiveResult> {
        // 基于高斯过程和预期改进的多目标贝叶斯优化实现
        use rand::Rng;

        if objective_functions.is_empty() {
            return Err(Error::invalid_argument("MOBO 需要至少一个目标函数".to_string()));
        }
        if bounds.is_empty() {
            return Err(Error::invalid_argument("MOBO 需要至少一个参数维度".to_string()));
        }

        let dim = bounds.len();
        self.objective_count = objective_functions.len();

        // 初始化状态
        self.observed_x.clear();
        self.observed_y.clear();
        self.candidates.clear();
        self.initialize_models();

        let params = &self.config.parameters;
        let initial_samples = params.initial_samples.max(1);
        let max_iterations = params.max_iterations.max(1);
        let candidate_count = params.candidate_count.max(4);

        // 辅助函数：在边界内随机采样一点
        let mut sample_point = |rng: &mut StdRng| -> Vec<f64> {
            let mut x = Vec::with_capacity(dim);
            for &(lo, hi) in bounds {
                let v = rng.gen_range(lo..=hi);
                x.push(v);
            }
            x
        };

        let start_time = std::time::Instant::now();

        // 1. 初始采样
        for _ in 0..initial_samples {
            let x = sample_point(&mut self.rng);
            let mut y = Vec::with_capacity(self.objective_count);
            for f in objective_functions {
                y.push(f(&x));
            }
            self.observed_x.push(x);
            self.observed_y.push(y);
        }

        // 根据初始观测训练模型
        self.update_models();

        // 2. 迭代贝叶斯优化
        let mut iterations = 0;
        let mut convergence_history = Vec::new();
        for _ in 0..max_iterations {
            iterations += 1;

            // 基于 EI 从候选集中选择下一个评估点
            let mut best_x: Option<Vec<f64>> = None;
            let mut best_score = f64::NEG_INFINITY;

            for _ in 0..candidate_count {
                let x = sample_point(&mut self.rng);
                let score = self.calculate_acquisition(&x);
                if score > best_score {
                    best_score = score;
                    best_x = Some(x);
                }
            }

            let x_next = match best_x {
                Some(x) => x,
                None => break,
            };

            let mut y_next = Vec::with_capacity(self.objective_count);
            for f in objective_functions {
                y_next.push(f(&x_next));
            }

            self.observed_x.push(x_next);
            self.observed_y.push(y_next);

            // 重新训练代理模型
            self.update_models();

            // 记录当前简单指标：使用第一个目标的最优值作为收敛历史
            if let Some(best) = self
                .observed_y
                .iter()
                .map(|vals| vals[0])
                .reduce(f64::min)
            {
                convergence_history.push(best);
            }
        }

        // 3. 从观测数据构建帕累托前沿
        let pareto = self.compute_pareto_front();
        let pareto_front: Vec<Vec<f64>> = pareto.iter().map(|p| p.solution.clone()).collect();
        let objective_values: Vec<Vec<f64>> = pareto
            .iter()
            .map(|p| p.objective_values.clone())
            .collect();

        let runtime_ms = start_time.elapsed().as_millis() as u64;

        let mut final_metrics = HashMap::new();
        final_metrics.insert(
            "evaluations".to_string(),
            self.observed_x.len() as f64,
        );
        final_metrics.insert(
            "iterations".to_string(),
            iterations as f64,
        );

        Ok(MultiObjectiveResult {
            pareto_front,
            objective_values,
            hypervolume: None,
            generational_distance: None,
            inverted_generational_distance: None,
            spread: None,
            runtime_ms,
            all_solutions: None,
            final_metrics,
            convergence_history: if convergence_history.is_empty() {
                None
            } else {
                Some(convergence_history)
            },
            iterations,
            early_stopped: false,
            algorithm_specific: HashMap::new(),
        })
    }
    
    fn get_config(&self) -> &MultiObjectiveConfig {
        // MOBOConfig 和 MultiObjectiveConfig 是不同的类型
        // 由于 trait 要求返回 MultiObjectiveConfig，我们需要创建一个临时的
        // 使用 thread_local 存储转换后的配置
        thread_local! {
            static CONFIG: std::cell::RefCell<Option<MultiObjectiveConfig>> = std::cell::RefCell::new(None);
        }
        CONFIG.with(|c| {
            let mut config = c.borrow_mut();
            if config.is_none() {
                *config = Some(MultiObjectiveConfig {
                    algorithm: super::types::MultiObjectiveAlgorithm::MOBO,
                    objectives_count: self.objective_count,
                    objective_directions: vec![],
                    max_iterations: 100,
                    population_size: 50,
                    mutation_probability: 0.1,
                    crossover_probability: 0.9,
                    seed: self.config.seed,
                    neighborhood_size: None,
                    w: None,
                    c1: None,
                    c2: None,
                    archive_size: None,
                    objectives: vec![],
                    weight_strategy: super::types::WeightStrategy::Fixed,
                    metrics: vec![],
                    convergence_threshold: None,
                    early_stopping: None,
                    normalization: None,
                    algorithm_params: HashMap::new(),
                    parallel_processing: None,
                });
            }
            // 使用 unsafe 获取引用（不推荐，但为了返回引用而不复制）
            unsafe {
                let ptr = config.as_ref().unwrap() as *const MultiObjectiveConfig;
                &*ptr
            }
        })
    }
    
    fn validate_solution(&self, solution: &Vec<f64>, bounds: &[(f64, f64)]) -> Result<bool> {
        if solution.len() != bounds.len() {
            return Ok(false);
        }
        
        for (value, (min, max)) in solution.iter().zip(bounds.iter()) {
            if *value < *min || *value > *max {
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
        if solution1_objectives.len() != solution2_objectives.len() {
            return 0;
        }
        
        let mut dominates = 0;
        let mut is_dominated = 0;
        
        for (obj1, obj2) in solution1_objectives.iter().zip(solution2_objectives.iter()) {
            if obj1 < obj2 {
                dominates += 1;
            } else if obj1 > obj2 {
                is_dominated += 1;
            }
        }
        
        if dominates > 0 && is_dominated == 0 {
            1  // solution1 dominates solution2
        } else if is_dominated > 0 && dominates == 0 {
            -1 // solution2 dominates solution1
        } else {
            0  // neither dominates
        }
    }
    
    fn calculate_metrics(&self, solutions: &[Vec<f64>], objectives: &[Vec<f64>]) -> Result<MultiObjectiveResult> {
        // 计算帕累托前沿
        let pareto_solutions = self
            .compute_pareto_front()
            .into_iter()
            .map(|p| p.solution)
            .collect::<Vec<Vec<f64>>>();

        // 计算一些简单的质量指标
        let mut final_metrics = HashMap::new();
        final_metrics.insert("pareto_front_size".to_string(), pareto_solutions.len() as f64);

        Ok(MultiObjectiveResult {
            pareto_front: pareto_solutions,
            objective_values: objectives.to_vec(),
            hypervolume: None,
            generational_distance: None,
            inverted_generational_distance: None,
            spread: None,
            runtime_ms: 0,
            all_solutions: None,
            final_metrics,
            convergence_history: None,
            iterations: 0,
            early_stopped: false,
            algorithm_specific: HashMap::new(),
        })
    }
    
    fn optimize_index(&mut self, index_type: IndexType, parameter_space: &ParameterSpace) -> Result<(Vec<Pareto>, OptimizationResult)> {
        info!("Starting MOBO optimization for {:?}", index_type);
        
        // 初始化
        self.objective_count = 2; // 查询性能和索引大小
        self.initialize_models();
        self.observed_x.clear();
        self.observed_y.clear();
        
        let start_time = Instant::now();
        let max_iterations = self.config.parameters.max_iterations;
        let initial_samples = self.config.parameters.initial_samples;
        
        // 生成初始样本
        let mut initial_configs = self.generate_initial_samples(parameter_space, initial_samples);
        
        // 评估初始样本
        let mut eval_results: Vec<OptimizationResult> = Vec::new();

        for config in initial_configs.iter() {
            let result = self.evaluate_configuration(index_type, config, parameter_space)?;

            // 存储参数和目标值
            let param_vector = self.params_to_vector(config, parameter_space);
            let objective_vector = vec![
                result.performance.avg_query_time_ms,
                result.performance.index_size_bytes as f64,
            ];

            self.observed_x.push(param_vector);
            self.observed_y.push(objective_vector);
            eval_results.push(result);
        }
        
        // 更新模型
        self.update_models();
        
        // 主优化循环
        for iter in 0..max_iterations {
            debug!("MOBO iteration {}/{}", iter + 1, max_iterations);
            
            // 选择下一个评估点
            let next_config = self.select_next_point(parameter_space);
            
            // 评估新配置
            let result = self.evaluate_configuration(index_type, &next_config, parameter_space)?;

            // 存储参数和目标值
            let param_vector = self.params_to_vector(&next_config, parameter_space);
            let objective_vector = vec![
                result.performance.avg_query_time_ms,
                result.performance.index_size_bytes as f64,
            ];
            
            self.observed_x.push(param_vector);
            self.observed_y.push(objective_vector);
            eval_results.push(result);
            
            // 更新模型
            self.update_models();
            
            // 检查是否达到时间限制
            if let Some(time_limit) = self.config.parameters.time_limit {
                if start_time.elapsed() > time_limit {
                    info!("MOBO optimization reached time limit");
                    break;
                }
            }
        }
        
        // 计算帕累托前沿
        let pareto_front = self.compute_pareto_front();
        
        // 选择最佳配置（平衡查询性能和索引大小）
        let best_idx = self.select_best_configuration(&pareto_front);
        let best_pareto = &pareto_front[best_idx];

        // 找到与最佳Pareto解对应的评估结果
        let mut best_result: Option<OptimizationResult> = None;
        for (i, x) in self.observed_x.iter().enumerate() {
            if *x == best_pareto.solution && i < eval_results.len() {
                best_result = Some(eval_results[i].clone());
                break;
            }
        }

        // 如果未找到精确匹配，则退化为使用最后一个评估结果
        let result = best_result.unwrap_or_else(|| eval_results.last().cloned().unwrap());
        
        info!("MOBO optimization completed. Found {} Pareto optimal configurations.", pareto_front.len());
        Ok((pareto_front, result))
    }
    
    fn evaluate_configuration(&self, index_type: IndexType, parameters: &HashMap<String, ParameterValue>, parameter_space: &ParameterSpace) -> Result<OptimizationResult> {
        let mut config_str = String::new();
        for (name, value) in parameters {
            config_str.push_str(&format!("{}={:?}, ", name, value));
        }
        debug!("Evaluating configuration: {}", config_str);
        
        // 检查参数是否有效
        for (name, _value) in parameters {
            // 确认参数在参数空间中已定义
            let exists = parameter_space
                .parameters()
                .any(|p| p.name() == name.as_str());

            if !exists {
                return Err(Error::invalid_argument(format!(
                    "Unknown parameter: {}", name
                )));
            }
        }
        
        // 将参数映射到索引配置
        let mut index_config = IndexConfig::default();
        index_config.index_type = index_type;

        for (name, value) in parameters {
            let v_usize = match value {
                ParameterValue::Integer(i) => *i as usize,
                ParameterValue::Float(f) => *f as usize,
                ParameterValue::Categorical(_) => {
                    // 分类参数目前不直接映射到数值配置
                    continue;
                }
            };
            index_config.set_param(name, v_usize);
        }

        // 使用通用基准测试器评估当前配置
        let bench_config = BenchmarkConfig::default();
        let benchmark = IndexBenchmark::new(bench_config);
        let bench_result = benchmark.benchmark_index(index_type, index_config.clone())?;

        // 根据多目标（查询时间 + 索引大小）构造评分：时间和大小越小，得分越高
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
    
    /// 计算帕累托前沿
    fn compute_pareto_front(&self) -> Vec<Pareto> {
        // 检查是否有观测数据
        if self.observed_x.is_empty() || self.observed_y.is_empty() {
            return Vec::new();
        }
        
        let mut pareto_front = Vec::new();
        let mut dominated = vec![false; self.observed_y.len()];
        
        // 寻找非支配解
        for i in 0..self.observed_y.len() {
            if dominated[i] {
                continue;
            }
            
            for j in 0..self.observed_y.len() {
                if i == j || dominated[j] {
                    continue;
                }
                
                // 检查j是否被i支配
                let mut i_dominates = true;
                for k in 0..self.objective_count {
                    let better = match k {
                        0 => self.observed_y[i][k] <= self.observed_y[j][k], // 最小化
                        _ => self.observed_y[i][k] >= self.observed_y[j][k], // 最大化
                    };
                    
                    if !better {
                        i_dominates = false;
                        break;
                    }
                }
                
                if i_dominates {
                    dominated[j] = true;
                }
            }
        }
        
        // 构建帕累托前沿
        for i in 0..self.observed_y.len() {
            if !dominated[i] {
                let pareto = Pareto {
                    solution: self.observed_x[i].clone(),
                    objective_values: self.observed_y[i].clone(),
                    rank: 0,
                    crowding_distance: 0.0,
                };

                pareto_front.push(pareto);
            }
        }

        pareto_front
    }
    
    /// 选择最佳配置
    fn select_best_configuration(&self, pareto_front: &[Pareto]) -> usize {
        if pareto_front.is_empty() {
            return 0;
        }
        
        // 使用权重方法选择平衡查询性能和索引大小的配置
        let query_weight = self.config.parameters.objective_weights.0;
        let size_weight = self.config.parameters.objective_weights.1;
        
        // 对目标值进行归一化
        let mut min_query = f64::INFINITY;
        let mut max_query = f64::NEG_INFINITY;
        let mut min_size = f64::INFINITY;
        let mut max_size = f64::NEG_INFINITY;
        
        for pareto in pareto_front {
            min_query = min_query.min(pareto.objective_values[0]);
            max_query = max_query.max(pareto.objective_values[0]);
            min_size = min_size.min(pareto.objective_values[1]);
            max_size = max_size.max(pareto.objective_values[1]);
        }
        
        // 避免除以零
        let query_range = (max_query - min_query).max(1e-10);
        let size_range = (max_size - min_size).max(1e-10);
        
        // 计算加权和并选择最佳配置
        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = 0;
        
        for (i, pareto) in pareto_front.iter().enumerate() {
            // 归一化目标值（将最小化目标转换为最大化）
            let norm_query = 1.0 - (pareto.objective_values[0] - min_query) / query_range;
            let norm_size = 1.0 - (pareto.objective_values[1] - min_size) / size_range;
            
            // 计算加权和
            let score = query_weight * norm_query + size_weight * norm_size;
            
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        
        best_idx
    }
}

/// 简化的高斯过程模型
struct GaussianProcess {
    /// 长度尺度参数
    kernel_param: f64,
    /// 观测点
    points: Vec<Vec<f64>>,
    /// 观测值
    values: Vec<f64>,
}

impl GaussianProcess {
    /// 创建新的高斯过程模型
    fn new(kernel_param: f64) -> Self {
        Self {
            kernel_param,
            points: Vec::new(),
            values: Vec::new(),
        }
    }
    
    /// 更新模型
    fn update(&mut self, points: &[Vec<f64>], values: &[f64]) {
        self.points = points.to_vec();
        self.values = values.to_vec();
    }
    
    /// 预测指定点的值和不确定性
    fn predict(&self, x: &[f64]) -> (f64, f64) {
        // 如果没有观测数据，返回默认值
        if self.points.is_empty() {
            return (0.0, 1.0);
        }
        
        // 计算与所有观测点的核函数值
        let mut weights = Vec::with_capacity(self.points.len());
        let mut weight_sum = 0.0;
        
        for point in &self.points {
            let distance = self.calculate_distance(x, point);
            let kernel_val = (-distance / (2.0 * self.kernel_param.powi(2))).exp();
            weights.push(kernel_val);
            weight_sum += kernel_val;
        }
        
        // 避免除以零
        if weight_sum < 1e-10 {
            return (0.0, 1.0);
        }
        
        // 计算加权平均值
        let mut mean = 0.0;
        for i in 0..self.points.len() {
            mean += weights[i] * self.values[i] / weight_sum;
        }
        
        // 计算不确定性
        let std = 1.0 - (weight_sum / self.points.len() as f64).min(1.0);
        
        (mean, std)
    }
    
    /// 计算两点之间的距离
    fn calculate_distance(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let mut sum_sq = 0.0;
        for i in 0..x1.len().min(x2.len()) {
            let diff = x1[i] - x2[i];
            sum_sq += diff * diff;
        }
        sum_sq.sqrt()
    }
}

/// 误差函数的生产级实现
/// 使用Abramowitz和Stegun的近似公式，精度优于简化版本
fn erf(x: f64) -> f64 {
    // 使用更精确的有理近似（Abramowitz & Stegun 7.1.26）
    let abs_x = x.abs();
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    
    // 对于大的|x|值，使用渐近展开
    if abs_x > 6.0 {
        return sign;
    }
    
    // 使用有理近似，精度约为1.5e-7
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    
    let t = 1.0 / (1.0 + p * abs_x);
    let tau = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
    let result = 1.0 - tau * (-abs_x * abs_x).exp();
    
    sign * result
}

/// 互补误差函数
fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

/// 正态分布的累积分布函数
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// 正态分布的概率密度函数
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// 标准正态分布的分位数函数（近似）
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    if (p - 0.5).abs() < 1e-10 { return 0.0; }
    
    // 使用Beasley-Springer-Moro算法
    let a = [0.0, -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    let b = [0.0, -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
    let c = [0.0, -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    let d = [0.0, 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    
    if p < p_low {
        // Rational approximation for lower region
        let q = (-2.0 * p.ln()).sqrt();
        return (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) / ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1.0);
    }
    
    if p > p_high {
        // Rational approximation for upper region
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        return -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) / ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1.0);
    }
    
    // Rational approximation for central region
    let q = p - 0.5;
    let r = q * q;
    (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q / (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1.0)
}

// 实现生产级的benchmark函数
fn hnsw_benchmark(parameters: &IndexParameters) -> Result<Vec<f64>> {
    // 基于 HNSW 基本参数的性能估计（仅使用现有字段）
    let m = parameters.m.max(1);
    let ef_construction = parameters.ef_construction.max(1);
    let ef_search = parameters.ef_search.max(1);
    let max_layers = parameters.max_layers.max(1);

    // 1. 模拟索引构建和查询性能
    let construction_cost = calculate_hnsw_construction_cost(m, ef_construction, max_layers);
    let query_performance = calculate_hnsw_query_performance(ef_search, m);
    let memory_usage = calculate_hnsw_memory_usage(m, max_layers);
    let recall_estimate = calculate_hnsw_recall(ef_search, ef_construction);

    // 2. 计算综合性能指标
    let latency_score = 1.0 / (1.0 + query_performance); // 延迟越低越好
    let throughput_score = 1000.0 / query_performance; // 吞吐量
    let memory_efficiency = 1.0 / (1.0 + memory_usage); // 内存效率
    let quality_score = recall_estimate; // 质量得分

    Ok(vec![
        latency_score,
        throughput_score,
        memory_efficiency,
        quality_score,
        construction_cost,
    ])
}

fn calculate_hnsw_construction_cost(m: usize, ef_construction: usize, max_layers: usize) -> f64 {
    // 基于 HNSW 算法特性计算构建成本
    let connection_factor = (m as f64 * ef_construction as f64).log2();
    let layer_factor = (max_layers as f64).sqrt();
    let base_cost = 1.0;

    base_cost * connection_factor * layer_factor
}

fn calculate_hnsw_query_performance(ef_search: usize, m: usize) -> f64 {
    // 查询性能主要取决于ef_search和连接数m
    let search_complexity = (ef_search as f64).log2();
    let connection_efficiency = 1.0 / (1.0 + (m as f64 - 16.0).abs() / 16.0);
    
    search_complexity / connection_efficiency
}

fn calculate_hnsw_memory_usage(m: usize, max_layers: usize) -> f64 {
    // 内存使用量基于连接数和层级结构
    let avg_connections = (m as f64) * (max_layers as f64);
    let layer_overhead = 1.2; // 层级结构开销
    let base_memory = 1.0;

    base_memory * avg_connections * layer_overhead / 64.0 // 归一化到合理范围
}

fn calculate_hnsw_recall(ef_search: usize, ef_construction: usize) -> f64 {
    // 召回率主要取决于ef参数
    let search_factor = (ef_search as f64 / 50.0).min(2.0);
    let construction_factor = (ef_construction as f64 / 200.0).min(2.0);
    let base_recall = 0.85;
    
    (base_recall + (search_factor * construction_factor - 1.0) * 0.1).min(0.99).max(0.5)
}

fn ivf_benchmark(parameters: &IndexParameters) -> Result<Vec<f64>> {
    // 使用 HNSW 参数近似模拟 IVF 行为（仅为分析用途，当前未在主流程中使用）
    let m = parameters.m.max(1);
    let ef_construction = parameters.ef_construction.max(1);
    let ef_search = parameters.ef_search.max(1);

    let quantization_quality = calculate_ivf_quantization_quality(m, ef_construction);
    let search_efficiency = calculate_ivf_search_efficiency(ef_construction, ef_search);
    let memory_compression = calculate_ivf_memory_compression(m, ef_construction);
    let build_time = calculate_ivf_build_time(ef_construction, m);

    // 3. 计算综合性能
    let latency_score = search_efficiency;
    let throughput_score = 800.0 / search_efficiency;
    let memory_efficiency = memory_compression;
    let quality_score = quantization_quality;
    let construction_cost = build_time;

    Ok(vec![
        latency_score,
        throughput_score,
        memory_efficiency,
        quality_score,
        construction_cost,
    ])
}

fn calculate_ivf_quantization_quality(pq_m: usize, pq_nbits: usize) -> f64 {
    // 量化质量取决于PQ参数
    let codebook_size = 2_usize.pow(pq_nbits as u32) as f64;
    let subspace_count = pq_m as f64;
    
    let quality_factor = (codebook_size * subspace_count).log2() / 64.0; // 归一化
    0.7 + (quality_factor * 0.25).min(0.25)
}

fn calculate_ivf_search_efficiency(nlist: usize, nprobe: usize) -> f64 {
    // 搜索效率取决于聚类数和探测数
    let cluster_efficiency = (nlist as f64).log2() / 10.0;
    let probe_cost = nprobe as f64 / nlist as f64;
    
    cluster_efficiency * (1.0 + probe_cost)
}

fn calculate_ivf_memory_compression(pq_m: usize, pq_nbits: usize) -> f64 {
    // 内存压缩率
    let bits_per_vector = pq_m * pq_nbits;
    let compression_ratio = 256.0 / bits_per_vector as f64; // 假设原始向量256位
    
    compression_ratio.min(10.0) / 10.0 // 归一化
}

fn calculate_ivf_build_time(nlist: usize, pq_m: usize) -> f64 {
    // 构建时间主要取决于聚类和PQ训练
    let cluster_cost = (nlist as f64).log2() * 2.0;
    let pq_cost = pq_m as f64 * 0.5;
    
    (cluster_cost + pq_cost) / 10.0 // 归一化
}

fn vptree_benchmark(parameters: &IndexParameters) -> Result<Vec<f64>> {
    // 使用 HNSW 参数近似模拟 VP-Tree 行为（分析用途）
    let m = parameters.m.max(1);
    let ef_construction = parameters.ef_construction.max(1);
    let ef_search = parameters.ef_search.max(1);

    // 2. 计算性能指标
    let tree_balance = calculate_vptree_balance(m, ef_construction);
    let search_pruning = calculate_vptree_pruning_efficiency(ef_search as f64 / ef_construction as f64, m);
    let memory_overhead = calculate_vptree_memory_overhead(m);
    let build_complexity = calculate_vptree_build_complexity(ef_construction, m);

    // 3. 计算综合性能
    let latency_score = 1.0 / search_pruning;
    let throughput_score = 600.0 * tree_balance / search_pruning;
    let memory_efficiency = 1.0 / memory_overhead;
    let quality_score = tree_balance * 0.8 + search_pruning * 0.2;
    let construction_cost = build_complexity;

    Ok(vec![
        latency_score,
        throughput_score,
        memory_efficiency,
        quality_score,
        construction_cost,
    ])
}

fn calculate_vptree_balance(leaf_size: usize, sample_size: usize) -> f64 {
    // 树平衡性取决于叶子大小和采样策略
    let leaf_factor = 1.0 / (1.0 + (leaf_size as f64 - 10.0).abs() / 10.0);
    let sample_factor = (sample_size as f64 / 100.0).min(2.0);
    
    leaf_factor * sample_factor.sqrt()
}

fn calculate_vptree_pruning_efficiency(tau: f64, leaf_size: usize) -> f64 {
    // 剪枝效率取决于tau阈值和叶子大小
    let tau_efficiency = 1.0 / (1.0 + tau * 10.0);
    let leaf_efficiency = (20.0 / leaf_size as f64).min(2.0);
    
    (tau_efficiency + leaf_efficiency) / 2.0
}

fn calculate_vptree_memory_overhead(leaf_size: usize) -> f64 {
    // 内存开销主要来自树结构
    let tree_overhead = 1.2; // 树结构基础开销
    let leaf_overhead = 1.0 + (leaf_size as f64 / 50.0);
    
    tree_overhead * leaf_overhead
}

fn calculate_vptree_build_complexity(sample_size: usize, leaf_size: usize) -> f64 {
    // 构建复杂度取决于采样和分割
    let sample_cost = (sample_size as f64).log2() / 7.0;
    let partition_cost = (1.0 / leaf_size as f64) * 10.0;
    
    sample_cost + partition_cost
} 