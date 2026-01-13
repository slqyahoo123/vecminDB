// 向量优化器实现
// 包含不同的优化算法实现

use crate::vector::{
    index::{IndexType, IndexConfig},
    benchmark::{IndexBenchmark, BenchmarkResult},
};
use super::{
    config::{OptimizerConfig, OptimizationTarget},
    parameter_space::{ParameterSpace, ParameterRange, OptimizationDimension},
    scoring,
    utils,
    optimization_result::OptimizationResult,
};
use crate::Result;
use std::collections::{HashMap, HashSet};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand::seq::SliceRandom;
use rayon::prelude::*;
use rand::thread_rng;
use rand::rngs::ThreadRng;
use serde::{Serialize, Deserialize};

/// 配置值类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConfigValue {
    /// 整数值
    Int(i64),
    /// 浮点值
    Float(f64),
    /// 布尔值
    Bool(bool),
    /// 字符串值
    String(String),
}

/// 索引优化器结构
pub struct IndexOptimizer {
    /// 优化配置
    config: OptimizerConfig,
    /// 基准测试器
    benchmark: IndexBenchmark,
    /// 评分函数
    scoring_fn: Option<Box<dyn Fn(&BenchmarkResult) -> f64>>,
    /// 优化维度
    dimensions: Vec<OptimizationDimension>,
    /// 随机状态
    rng: ThreadRng,
    /// 最佳配置
    best_config: Option<IndexConfig>,
    /// 最佳得分
    best_score: Option<f64>,
    /// 已尝试的配置
    tried_configs: HashSet<String>,
    /// 观察结果
    observed_results: Vec<(HashMap<String, ConfigValue>, f64, IndexConfig, BenchmarkResult)>,
    /// 当前迭代
    current_iteration: usize,
    /// 初始化样本大小
    initial_sample_size: usize,
}

impl IndexOptimizer {
    /// 创建新的优化器实例
    pub fn new(config: OptimizerConfig) -> Self {
        let benchmark = IndexBenchmark::new(config.benchmark_config.clone());
        Self {
            config,
            benchmark,
            scoring_fn: None,
            dimensions: Vec::new(),
            rng: thread_rng(),
            best_config: None,
            best_score: None,
            tried_configs: HashSet::new(),
            observed_results: Vec::new(),
            current_iteration: 0,
            initial_sample_size: 0,
        }
    }

    /// 根据给定的索引类型进行优化
    pub fn optimize(&self, index_type: IndexType) -> Result<OptimizationResult> {
        let parameter_space = ParameterSpace::for_index_type(index_type);
        
        // 根据配置选择优化策略
        match (
            self.config.use_grid_search,
            self.config.use_random_search,
            self.config.use_bayesian,
            self.config.use_genetic_algorithm
        ) {
            (true, _, _, _) => self.optimize_grid_search(index_type, &parameter_space),
            (_, true, _, _) => self.optimize_random_search(index_type, &parameter_space),
            (_, _, true, _) => self.optimize_bayesian(index_type, &parameter_space),
            (_, _, _, true) => self.optimize_genetic(index_type, &parameter_space),
            _ => self.optimize_grid_search(index_type, &parameter_space), // 默认使用网格搜索
        }
    }

    /// 贝叶斯优化实现
    fn optimize_bayesian(&self, index_type: IndexType, parameter_space: &ParameterSpace) -> Result<OptimizationResult> {
        // 贝叶斯优化实现
        
        // 初始化随机数生成器
        let seed = self.config.random_seed.unwrap_or_else(|| rand::random());
        let mut rng = StdRng::seed_from_u64(seed);
        
        // 如果参数空间为空，使用默认配置
        if parameter_space.ranges.is_empty() {
            let mut default_config = IndexConfig::default();
            default_config.index_type = index_type;
            let result = self.benchmark.benchmark_index(index_type, default_config.clone())?;
            let score = scoring::calculate_score(&result, self.config.target);
            
            return Ok(OptimizationResult {
                index_type,
                best_config: default_config,
                performance: result,
                evaluated_configs: 1,
                target: self.config.target,
                score,
            });
        }
        
        // 初始随机点采样（探索阶段）
        let init_points = 5.min(self.config.max_iterations / 4).max(2);
        let mut observed_points = Vec::with_capacity(self.config.max_iterations);
        
        // 收集初始观察点
        for _ in 0..init_points {
            let params_f64 = parameter_space.random_params(&mut rng);
            // Convert f64 params to String for apply_params
            let params_string: HashMap<String, String> = params_f64.iter()
                .map(|(k, v)| (k.clone(), v.to_string()))
                .collect();
            // Convert f64 params to ConfigValue for observed_points
            let params_config: HashMap<String, ConfigValue> = params_f64.iter()
                .map(|(k, v)| (k.clone(), ConfigValue::Float(*v)))
                .collect();
            
            let mut config = IndexConfig::default();
            config.index_type = index_type;
            utils::apply_params(&mut config, &params_string);
            
            if let Ok(result) = self.benchmark.benchmark_index(index_type, config.clone()) {
                let score = scoring::calculate_score(&result, self.config.target);
                observed_points.push((params_config, score, config, result));
            }
        }
        
        // 如果没有有效的观察点，返回默认结果
        if observed_points.is_empty() {
            return self.optimize_random_search(index_type, parameter_space);
        }
        
        // 高斯过程回归模型
        let mut best_params = observed_points[0].0.clone();
        let mut best_score = observed_points[0].1;
        let mut best_config = observed_points[0].2.clone();
        let mut best_result = observed_points[0].3.clone();
        
        // 找到初始最佳点
        for (params, score, config, result) in &observed_points {
            if *score > best_score {
                best_params = params.clone();
                best_score = *score;
                best_config = config.clone();
                best_result = result.clone();
            }
        }
        
        // 贝叶斯优化的主循环（利用阶段）
        let remaining_iterations = self.config.max_iterations - init_points;
        for _ in 0..remaining_iterations {
            // 1. 构建高斯过程模型
            let gp_model = self.build_gaussian_process(&observed_points);
            
            // 2. 使用获取函数找到下一个采样点
            let next_params_config = self.find_next_point(&gp_model, parameter_space, &observed_points, &mut rng);
            
            // 3. 评估新点
            // Convert ConfigValue params to String for apply_params
            let next_params_string: HashMap<String, String> = next_params_config.iter()
                .map(|(k, v)| {
                    let val_str = match v {
                        ConfigValue::Int(i) => i.to_string(),
                        ConfigValue::Float(f) => f.to_string(),
                        ConfigValue::Bool(b) => b.to_string(),
                        ConfigValue::String(s) => s.clone(),
                    };
                    (k.clone(), val_str)
                })
                .collect();
            
            let mut config = IndexConfig::default();
            config.index_type = index_type;
            utils::apply_params(&mut config, &next_params_string);
            
            if let Ok(result) = self.benchmark.benchmark_index(index_type, config.clone()) {
                let score = scoring::calculate_score(&result, self.config.target);
                
                // 4. 更新观察点和最佳点
                observed_points.push((next_params_config.clone(), score, config.clone(), result.clone()));
                
                if score > best_score {
                    best_params = next_params_config;
                    best_score = score;
                    best_config = config;
                    best_result = result;
                }
            }
        }
        
        // 返回最佳结果
        Ok(OptimizationResult {
            index_type,
            best_config,
            performance: best_result,
            evaluated_configs: observed_points.len(),
            target: self.config.target,
            score: best_score,
        })
    }
    
    // 构建高斯过程模型
    fn build_gaussian_process(&self, observed_points: &[(HashMap<String, ConfigValue>, f64, IndexConfig, BenchmarkResult)]) -> GaussianProcess {
        // 提取输入和输出数据
        let inputs: Vec<Vec<f64>> = observed_points.iter()
            .map(|(params, _, _, _)| self.params_to_vector(params))
            .collect();
        let outputs: Vec<f64> = observed_points.iter()
            .map(|(_, score, _, _)| *score)
            .collect();

        // 计算输出均值和标准差进行标准化
        let output_mean = outputs.iter().sum::<f64>() / outputs.len() as f64;
        let output_var = outputs.iter()
            .map(|y| (y - output_mean).powi(2))
            .sum::<f64>() / outputs.len() as f64;
        let output_std = output_var.sqrt().max(1e-6);

        // 标准化输出
        let normalized_outputs: Vec<f64> = outputs.iter()
            .map(|y| (y - output_mean) / output_std)
            .collect();

        // 构建协方差矩阵
        let n = inputs.len();
        let mut covariance_matrix = vec![vec![0.0; n]; n];
        
        for i in 0..n {
            for j in 0..n {
                covariance_matrix[i][j] = self.rbf_kernel(&inputs[i], &inputs[j], 1.0, 0.1);
                if i == j {
                    covariance_matrix[i][j] += 1e-6; // 数值稳定性
                }
            }
        }

        // 计算协方差矩阵的逆（使用Cholesky分解）
        let inverted_covariance = self.invert_matrix(&covariance_matrix);

        GaussianProcess {
            inputs,
            outputs: normalized_outputs,
            output_mean,
            output_std,
            covariance_matrix,
            inverted_covariance,
            length_scale: 0.1,
            signal_variance: 1.0,
            noise_variance: 1e-6,
        }
    }

    // 将参数映射转换为数值向量
    fn params_to_vector(&self, params: &HashMap<String, ConfigValue>) -> Vec<f64> {
        let mut vector = Vec::new();
        
        // 预定义的参数顺序，确保一致性
        let param_order = ["m", "ef_construction", "ef_search", "max_m", "max_m0", 
                          "ml", "seed", "nlist", "nprobe", "quantizer_type"];
        
        for &param_name in &param_order {
            if let Some(value) = params.get(param_name) {
                match value {
                    ConfigValue::Int(i) => vector.push(*i as f64 / 100.0), // 归一化
                    ConfigValue::Float(f) => vector.push(*f),
                    ConfigValue::Bool(b) => vector.push(if *b { 1.0 } else { 0.0 }),
                    ConfigValue::String(s) => {
                        // 简单的字符串哈希转换
                        let hash = s.chars().fold(0u32, |acc, c| acc.wrapping_mul(31).wrapping_add(c as u32));
                        vector.push((hash % 100) as f64 / 100.0);
                    }
                }
            } else {
                vector.push(0.0); // 默认值
            }
        }
        
        vector
    }

    // RBF（径向基函数）核函数
    fn rbf_kernel(&self, x1: &[f64], x2: &[f64], signal_variance: f64, length_scale: f64) -> f64 {
        let squared_distance: f64 = x1.iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        
        signal_variance * (-squared_distance / (2.0 * length_scale.powi(2))).exp()
    }

    // 矩阵求逆（使用LU分解）
    fn invert_matrix(&self, matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = matrix.len();
        let mut augmented = vec![vec![0.0; 2 * n]; n];
        
        // 构建增广矩阵 [A|I]
        for i in 0..n {
            for j in 0..n {
                augmented[i][j] = matrix[i][j];
                augmented[i][j + n] = if i == j { 1.0 } else { 0.0 };
            }
        }
        
        // 高斯-约旦消元法
        for i in 0..n {
            // 选择主元
            let mut max_row = i;
            for k in i + 1..n {
                if augmented[k][i].abs() > augmented[max_row][i].abs() {
                    max_row = k;
                }
            }
            
            // 交换行
            if max_row != i {
                augmented.swap(i, max_row);
            }
            
            // 防止除零
            if augmented[i][i].abs() < 1e-10 {
                continue;
            }
            
            // 缩放当前行
            let pivot = augmented[i][i];
            for j in 0..2 * n {
                augmented[i][j] /= pivot;
            }
            
            // 消元
            for k in 0..n {
                if k != i {
                    let factor = augmented[k][i];
                    for j in 0..2 * n {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }
        
        // 提取逆矩阵
        let mut inverse = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                inverse[i][j] = augmented[i][j + n];
            }
        }
        
        inverse
    }
    
    // 使用获取函数找到下一个采样点
    fn find_next_point(
        &self,
        gp: &GaussianProcess,
        parameter_space: &ParameterSpace,
        observed_points: &[(HashMap<String, ConfigValue>, f64, IndexConfig, BenchmarkResult)],
        rng: &mut StdRng
    ) -> HashMap<String, ConfigValue> {
        // 使用上界置信区间(UCB)作为获取函数
        // 生成候选点
        let num_candidates = 10;
        let mut candidates = Vec::with_capacity(num_candidates);
        
        for _ in 0..num_candidates {
            let params_f64 = parameter_space.random_params(rng);
            // Convert f64 params to ConfigValue
            let params_config: HashMap<String, ConfigValue> = params_f64.iter()
                .map(|(k, v)| (k.clone(), ConfigValue::Float(*v)))
                .collect();
            candidates.push(params_config);
        }
        
        // 找到UCB值最高的候选点
        let mut best_candidate = candidates[0].clone();
        let mut best_ucb = f64::NEG_INFINITY;
        
        for candidate in candidates {
            let (mean, std) = gp.predict(&candidate);
            let ucb = mean + 2.0 * std; // 探索-利用平衡参数
            
            if ucb > best_ucb {
                best_ucb = ucb;
                best_candidate = candidate;
            }
        }
        
        best_candidate
    }
}

// 高斯过程模型生产级实现
struct GaussianProcess {
    /// 输入数据（标准化后的参数向量）
    inputs: Vec<Vec<f64>>,
    /// 输出数据（标准化后的分数）
    outputs: Vec<f64>,
    /// 输出数据的均值（用于反标准化）
    output_mean: f64,
    /// 输出数据的标准差（用于反标准化）
    output_std: f64,
    /// 协方差矩阵
    covariance_matrix: Vec<Vec<f64>>,
    /// 协方差矩阵的逆
    inverted_covariance: Vec<Vec<f64>>,
    /// 核函数长度尺度参数
    length_scale: f64,
    /// 信号方差
    signal_variance: f64,
    /// 噪声方差
    noise_variance: f64,
}

impl GaussianProcess {
    // 预测给定点的均值和标准差
    fn predict(&self, params: &HashMap<String, crate::vector::optimizer::optimizers::ConfigValue>) -> (f64, f64) {
        if self.inputs.is_empty() {
            return (0.0, 1.0);
        }
        
        // 将参数转换为数值向量
        let x_star = self.params_to_vector(params);
        
        // 计算测试点与训练点的协方差向量
        let k_star: Vec<f64> = self.inputs.iter()
            .map(|x_train| self.rbf_kernel(&x_star, x_train))
            .collect();
        
        // 计算测试点的自协方差
        let k_star_star = self.signal_variance + self.noise_variance;
        
        // 计算预测均值：μ* = k*^T K^(-1) y
        let mean = self.vector_matrix_multiply(&k_star, &self.inverted_covariance)
            .iter()
            .zip(self.outputs.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
        
        // 计算预测方差：σ*² = k** - k*^T K^(-1) k*
        let k_inv_k_star = self.matrix_vector_multiply(&self.inverted_covariance, &k_star);
        let variance_reduction: f64 = k_star.iter()
            .zip(k_inv_k_star.iter())
            .map(|(a, b)| a * b)
            .sum();
        
        let variance = (k_star_star - variance_reduction).max(self.noise_variance);
        let std = variance.sqrt();
        
        // 反标准化
        let final_mean = mean * self.output_std + self.output_mean;
        let final_std = std * self.output_std;
        
        (final_mean, final_std)
    }
    
    // 将参数映射转换为数值向量
    fn params_to_vector(&self, params: &HashMap<String, crate::vector::optimizer::optimizers::ConfigValue>) -> Vec<f64> {
        let mut vector = Vec::new();
        
        // 预定义的参数顺序，确保一致性
        let param_order = ["m", "ef_construction", "ef_search", "max_m", "max_m0", 
                          "ml", "seed", "nlist", "nprobe", "quantizer_type"];
        
        for &param_name in &param_order {
            if let Some(value) = params.get(param_name) {
                match value {
                    crate::vector::optimizer::optimizers::ConfigValue::Int(i) => vector.push(*i as f64 / 100.0), // 归一化
                    crate::vector::optimizer::optimizers::ConfigValue::Float(f) => vector.push(*f),
                    crate::vector::optimizer::optimizers::ConfigValue::Bool(b) => vector.push(if *b { 1.0 } else { 0.0 }),
                    crate::vector::optimizer::optimizers::ConfigValue::String(s) => {
                        // 简单的字符串哈希转换
                        let hash = s.chars().fold(0u32, |acc, c| acc.wrapping_mul(31).wrapping_add(c as u32));
                        vector.push((hash % 100) as f64 / 100.0);
                    }
                }
            } else {
                vector.push(0.0); // 默认值
            }
        }
        
        vector
    }

    // RBF核函数
    fn rbf_kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let squared_distance: f64 = x1.iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        
        self.signal_variance * (-squared_distance / (2.0 * self.length_scale.powi(2))).exp()
    }

    // 向量与矩阵相乘
    fn vector_matrix_multiply(&self, vector: &[f64], matrix: &[Vec<f64>]) -> Vec<f64> {
        let mut result = vec![0.0; matrix[0].len()];
        
        for i in 0..matrix.len() {
            for j in 0..matrix[0].len() {
                result[j] += vector[i] * matrix[i][j];
            }
        }
        
        result
    }
    
    // 矩阵与向量相乘
    fn matrix_vector_multiply(&self, matrix: &[Vec<f64>], vector: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; matrix.len()];
        
        for i in 0..matrix.len() {
            for j in 0..vector.len() {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        
        result
    }
}

impl IndexOptimizer {
    /// 使用网格搜索优化
    fn optimize_grid_search(&self, index_type: IndexType, parameter_space: &ParameterSpace) -> Result<OptimizationResult> {
        // 生成所有可能的参数组合
        let all_params = parameter_space.grid_params();
        
        if all_params.is_empty() {
            // 如果没有参数空间，使用默认配置
            let mut default_config = IndexConfig::default();
            default_config.index_type = index_type;
            let result = self.benchmark.benchmark_index(index_type, default_config.clone())?;
            let score = scoring::calculate_score(&result, self.config.target);
            
            return Ok(OptimizationResult {
                index_type,
                best_config: default_config,
                performance: result,
                evaluated_configs: 1,
                target: self.config.target,
                score,
            });
        }
        
        // 限制测试的配置数量
        let max_configs = self.config.max_iterations.min(all_params.len());
        let params_to_test = if max_configs < all_params.len() {
            // 随机采样
            let seed = self.config.random_seed.unwrap_or_else(|| rand::random());
            let mut rng = StdRng::seed_from_u64(seed);
            let mut indices: Vec<usize> = (0..all_params.len()).collect();
            indices.shuffle(&mut rng);
            indices.truncate(max_configs);
            
            indices.into_iter().map(|i| all_params[i].clone()).collect()
        } else {
            all_params
        };
        
        // 串行执行（为保证线程安全，这里暂不启用并行逻辑）
        let (best_config, best_result, best_score) = {
            let mut best_config = IndexConfig::default();
            best_config.index_type = index_type;
            let mut best_result = match self.benchmark.benchmark_index(index_type, best_config.clone()) {
                Ok(result) => result,
                Err(_) => BenchmarkResult::default(),
            };
            let mut best_score = scoring::calculate_score(&best_result, self.config.target);

            for params in &params_to_test {
                let mut config = IndexConfig::default();
                config.index_type = index_type;
                let params_string: HashMap<String, String> = params.iter()
                    .map(|(k, v)| (k.clone(), v.to_string()))
                    .collect();
                utils::apply_params(&mut config, &params_string);

                if let Ok(result) = self.benchmark.benchmark_index(index_type, config.clone()) {
                    let score = scoring::calculate_score(&result, self.config.target);
                    if score > best_score {
                        best_config = config;
                        best_result = result;
                        best_score = score;
                    }
                }
            }

            (best_config, best_result, best_score)
        };
        
        Ok(OptimizationResult {
            index_type,
            best_config,
            performance: best_result,
            evaluated_configs: params_to_test.len(),
            target: self.config.target,
            score: best_score,
        })
    }

    /// 使用随机搜索优化
    fn optimize_random_search(&self, index_type: IndexType, parameter_space: &ParameterSpace) -> Result<OptimizationResult> {
        // 如果参数空间为空，直接返回默认配置
        if parameter_space.ranges.is_empty() {
            let mut default_config = IndexConfig::default();
            default_config.index_type = index_type;
            let result = self.benchmark.benchmark_index(index_type, default_config.clone())?;
            let score = scoring::calculate_score(&result, self.config.target);
            
            return Ok(OptimizationResult {
                index_type,
                best_config: default_config,
                performance: result,
                evaluated_configs: 1,
                target: self.config.target,
                score,
            });
        }
        
        // 初始化随机数生成器
        let seed = self.config.random_seed.unwrap_or_else(|| rand::random());
        let mut rng = StdRng::seed_from_u64(seed);
        
        // 生成随机参数集
        let mut params_list = Vec::with_capacity(self.config.max_iterations);
        for _ in 0..self.config.max_iterations {
            params_list.push(parameter_space.random_params(&mut rng));
        }
        
        // 串行执行（为保证线程安全，这里暂不启用并行逻辑）
        let (best_config, best_result, best_score) = {
            let mut best_config = IndexConfig::default();
            best_config.index_type = index_type;
            let mut best_result = match self.benchmark.benchmark_index(index_type, best_config.clone()) {
                Ok(result) => result,
                Err(_) => BenchmarkResult::default(),
            };
            let mut best_score = scoring::calculate_score(&best_result, self.config.target);

            for params in &params_list {
                let mut config = IndexConfig::default();
                config.index_type = index_type;
                let params_string: HashMap<String, String> = params.iter()
                    .map(|(k, v)| (k.clone(), v.to_string()))
                    .collect();
                utils::apply_params(&mut config, &params_string);

                if let Ok(result) = self.benchmark.benchmark_index(index_type, config.clone()) {
                    let score = scoring::calculate_score(&result, self.config.target);
                    if score > best_score {
                        best_config = config;
                        best_result = result;
                        best_score = score;
                    }
                }
            }

            (best_config, best_result, best_score)
        };
        
        Ok(OptimizationResult {
            index_type,
            best_config,
            performance: best_result,
            evaluated_configs: params_list.len(),
            target: self.config.target,
            score: best_score,
        })
    }

    /// 遗传算法优化（基于参数空间的完整实现）
    fn optimize_genetic(&self, index_type: IndexType, parameter_space: &ParameterSpace) -> Result<OptimizationResult> {
        // 实现遗传算法优化
        use rand::Rng;
        use std::time::Instant;
        
        log::info!("开始使用遗传算法优化 {:?} 索引", index_type);
        
        let ranges = &parameter_space.ranges;
        if ranges.is_empty() {
            return self.optimize_random_search(index_type, parameter_space);
        }
        
        // 定义遗传算法参数
        let population_size = 20usize;
        let max_generations = 15usize;
        let crossover_rate = 0.8f64;
        let mutation_rate = 0.15f64;
        let elitism_count = 2usize.min(population_size);
        
        // 随机种子
        let seed = self.config.random_seed.unwrap_or_else(|| rand::random());
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        
        // 采样一个随机个体（每个维度对应 ranges 里的一个参数）
        let mut sample_individual = |rng: &mut rand::rngs::StdRng| -> Vec<f64> {
            ranges
                .iter()
                .map(|r: &ParameterRange| {
                    let (min, max) = (r.min, r.max);
                    let value = if r.is_log_scale && min > 0.0 && max > 0.0 {
                        let log_min = min.ln();
                        let log_max = max.ln();
                        rng.gen_range(log_min..log_max).exp()
                    } else {
                        rng.gen_range(min..max)
                    };
                    if r.is_integer { value.round() } else { value }
                })
                .collect()
        };
        
        // 基因 -> IndexConfig -> BenchmarkResult -> score
        let mut eval_individual = |genes: &Vec<f64>| -> (IndexConfig, BenchmarkResult, f64) {
            let mut config = IndexConfig::default();
            config.index_type = index_type;
            utils::apply_params_to_config(&mut config, genes, ranges);
            
            let bench = self
                .benchmark
                .benchmark_index(index_type, config.clone())
                .unwrap_or_default();
            let score = scoring::calculate_score(&bench, self.config.target);
            (config, bench, score)
        };
        
        // 初始化种群
        let mut population: Vec<Vec<f64>> = Vec::with_capacity(population_size);
        let mut scores: Vec<f64> = Vec::with_capacity(population_size);
        let mut configs: Vec<IndexConfig> = Vec::with_capacity(population_size);
        let mut results: Vec<BenchmarkResult> = Vec::with_capacity(population_size);
        
        for _ in 0..population_size {
            let genes = sample_individual(&mut rng);
            let (cfg, bench, score) = eval_individual(&genes);
            population.push(genes);
            configs.push(cfg);
            results.push(bench);
            scores.push(score);
        }
        
        // 当前最佳
        let mut best_idx = 0usize;
        let mut best_score = scores[0];
        for (i, &s) in scores.iter().enumerate().skip(1) {
            if s > best_score {
                best_score = s;
                best_idx = i;
            }
        }
        let mut best_config = configs[best_idx].clone();
        let mut best_result = results[best_idx].clone();
        
        let mut total_evaluated = population_size;
        let start_time = Instant::now();
        
        for gen in 0..max_generations {
            let fitness_sum: f64 = scores.iter().sum();
            
            let select_parent = |rng: &mut rand::rngs::StdRng, scores: &Vec<f64>| -> usize {
                if fitness_sum <= 1e-9 {
                    rng.gen_range(0..scores.len())
                } else {
                    let mut r = rng.gen::<f64>() * fitness_sum;
                    for (i, s) in scores.iter().enumerate() {
                        r -= *s;
                        if r <= 0.0 {
                            return i;
                        }
                    }
                    scores.len() - 1
                }
            };
            
            let mut new_population = Vec::with_capacity(population_size);
            let mut new_scores = Vec::with_capacity(population_size);
            let mut new_configs = Vec::with_capacity(population_size);
            let mut new_results = Vec::with_capacity(population_size);
            
            // 精英保留
            let mut elite_idx: Vec<usize> = (0..population_size).collect();
            elite_idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
            for &idx in elite_idx.iter().take(elitism_count) {
                new_population.push(population[idx].clone());
                new_scores.push(scores[idx]);
                new_configs.push(configs[idx].clone());
                new_results.push(results[idx].clone());
            }
            
            // 生成剩余个体
            while new_population.len() < population_size {
                let p1 = select_parent(&mut rng, &scores);
                let mut p2 = select_parent(&mut rng, &scores);
                if p1 == p2 && population_size > 1 {
                    p2 = (p2 + 1) % population_size;
                }
                let parent1 = &population[p1];
                let parent2 = &population[p2];
                
                // 交叉
                let mut child = parent1.clone();
                if rng.gen::<f64>() < crossover_rate {
                    let point = rng.gen_range(0..parent1.len());
                    for i in point..parent1.len() {
                        child[i] = parent2[i];
                    }
                }
                
                // 变异：重新在该维度的范围内采样
                for (i, gene) in child.iter_mut().enumerate() {
                    if rng.gen::<f64>() < mutation_rate {
                        let r = &ranges[i];
                        let (min, max) = (r.min, r.max);
                        let new_value = if r.is_log_scale && min > 0.0 && max > 0.0 {
                            let log_min = min.ln();
                            let log_max = max.ln();
                            rng.gen_range(log_min..log_max).exp()
                        } else {
                            rng.gen_range(min..max)
                        };
                        *gene = if r.is_integer { new_value.round() } else { new_value };
                    }
                }
                
                let (cfg, bench, score) = eval_individual(&child);
                new_population.push(child);
                new_scores.push(score);
                new_configs.push(cfg);
                new_results.push(bench);
                total_evaluated += 1;
            }
            
            population = new_population;
            scores = new_scores;
            configs = new_configs;
            results = new_results;
            
            // 更新全局最佳
            for (i, &s) in scores.iter().enumerate() {
                if s > best_score {
                    best_score = s;
                    best_config = configs[i].clone();
                    best_result = results[i].clone();
                }
            }
            
            log::info!(
                "遗传算法第 {} 代完成，当前最佳分数: {}，总评估配置数: {}",
                gen + 1,
                best_score,
                total_evaluated
            );
        }
        
        let elapsed = start_time.elapsed();
        log::info!(
            "遗传算法优化完成，总评估 {} 个配置，耗时: {:?}",
            total_evaluated,
            elapsed
        );
        
        Ok(OptimizationResult {
            index_type,
            best_config,
            performance: best_result,
            evaluated_configs: total_evaluated,
            target: self.config.target,
            score: best_score,
        })
    }

    /// 优化所有索引类型
    pub fn optimize_all(&self) -> Result<HashMap<IndexType, OptimizationResult>> {
        let index_types = vec![
            IndexType::HNSW,
            IndexType::IVF,
            IndexType::PQ,
            IndexType::LSH,
            IndexType::Flat,
        ];
        
        let mut results = HashMap::new();
        
        for index_type in index_types {
            let result = self.optimize(index_type)?;
            results.insert(index_type, result);
        }
        
        Ok(results)
    }

    /// 生成优化报告
    pub fn generate_report(&self, results: &HashMap<IndexType, OptimizationResult>) -> String {
        let mut report = String::new();
        
        report.push_str("# 向量索引优化报告\n\n");
        report.push_str("## 优化配置\n\n");
        report.push_str(&format!("- 优化目标: {:?}\n", self.config.target));
        report.push_str(&format!("- 最大迭代次数: {}\n", self.config.max_iterations));
        report.push_str(&format!("- 并行优化: {}\n", self.config.parallel));
        report.push_str("\n## 优化结果\n\n");
        
        for (index_type, result) in results {
            report.push_str(&format!("### {:?} 索引\n\n", index_type));
            report.push_str(&format!("- 优化分数: {:.4}\n", result.score));
            report.push_str(&format!("- 查询时间: {:.4} ms\n", result.performance.avg_query_time_ms));
            report.push_str(&format!("- 构建时间: {:.4} ms\n", result.performance.build_time_ms));
            report.push_str(&format!("- 内存使用: {:.4} MB\n", result.performance.memory_usage_bytes as f64 / (1024.0 * 1024.0)));
            report.push_str(&format!("- 平均召回率: {:.4}\n", result.performance.accuracy));
            report.push_str(&format!("- 评估的配置数量: {}\n", result.evaluated_configs));
            
            report.push_str("\n#### 最佳配置\n\n");
            match index_type {
                IndexType::HNSW => {
                    report.push_str(&format!("- hnsw_m: {}\n", result.best_config.hnsw_m));
                    report.push_str(&format!("- hnsw_ef_construction: {}\n", result.best_config.hnsw_ef_construction));
                    report.push_str(&format!("- hnsw_ef_search: {}\n", result.best_config.hnsw_ef_search));
                },
                IndexType::IVF => {
                    report.push_str(&format!("- ivf_nlist: {}\n", result.best_config.ivf_nlist));
                    report.push_str(&format!("- ivf_nprobe: {}\n", result.best_config.ivf_nprobe));
                },
                IndexType::PQ => {
                    report.push_str(&format!("- pq_subvector_count: {}\n", result.best_config.pq_subvector_count));
                    report.push_str(&format!("- pq_subvector_bits: {}\n", result.best_config.pq_subvector_bits));
                },
                IndexType::LSH => {
                    report.push_str(&format!("- lsh_hash_count: {}\n", result.best_config.lsh_hash_count));
                    report.push_str(&format!("- lsh_hash_length: {}\n", result.best_config.lsh_hash_length));
                },
                _ => {}
            }
            
            report.push_str("\n");
        }
        
        report.push_str("\n## 比较表格\n\n");
        
        if results.len() > 1 {
            report.push_str("| 索引类型 | 查询时间(ms) | 构建时间(ms) | 内存使用(MB) | 准确率 |\n");
            report.push_str("|----------|--------------|--------------|--------------|--------|\n");
            
            for (index_type, result) in results {
                report.push_str(&format!("| {:?} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
                    index_type,
                    result.performance.avg_query_time_ms,
                    result.performance.build_time_ms,
                    result.performance.memory_usage_bytes as f64 / (1024.0 * 1024.0),
                    result.performance.accuracy
                ));
            }
        }
        
        report
    }
}

/// 运行优化器示例
pub fn run_optimizer_example() -> Result<String> {
    let config = OptimizerConfig {
        max_iterations: 20,
        parallel: true,
        use_grid_search: false,
        use_random_search: true,
        target: OptimizationTarget::BalancedPerformance,
        ..Default::default()
    };
    
    let optimizer = IndexOptimizer::new(config);
    let results = optimizer.optimize_all()?;
    
    let report = optimizer.generate_report(&results);
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_optimizer() {
        let config = OptimizerConfig {
            max_iterations: 5,
            use_random_search: true,
            ..Default::default()
        };
        
        let optimizer = IndexOptimizer::new(config);
        let result = optimizer.optimize(IndexType::HNSW).unwrap();
        
        // 基本验证
        assert_eq!(result.index_type, IndexType::HNSW);
        assert!(result.evaluated_configs > 0);
        assert!(result.score >= 0.0);
    }
}