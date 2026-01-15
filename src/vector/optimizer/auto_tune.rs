// 向量索引自动调优器
//
// 提供高级向量索引优化功能，包括：
// 1. 自动参数搜索和配置
// 2. 多目标优化
// 3. 索引策略自动选择
// 4. 查询参数自动优化

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, RwLock},
    time::{Duration, Instant},
};

use log::{info, warn};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    error::Error,
    vector::{
        index::{IndexConfig, IndexType, VectorIndex},
        utils::benchmark::{BenchmarkConfig, BenchmarkResult, IndexBenchmark},
    },
    Result,
};

use super::{
    config::{DatasetConfig, OptimizerConfig, OptimizationTarget},
    optimization_result::OptimizationResult,
    optimizers::IndexOptimizer,
    parameter_space::{OptimizationDimension, ParameterSpace},
    scoring,
    multi_objective::{
        MultiObjectiveAlgorithm, MultiObjectiveConfig, MultiObjectiveOptimizer,
        create_optimizer as create_mo_optimizer,
    },
};

/// 向量索引自动调优器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTuneConfig {
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
    /// 是否使用多目标优化
    pub use_multi_objective: bool,
    /// 多目标优化算法
    pub multi_objective_algorithm: MultiObjectiveAlgorithm,
    /// 基准测试配置
    pub benchmark_config: BenchmarkConfig,
    /// 学习率（用于自适应搜索）
    pub learning_rate: f64,
    /// 探索率（用于平衡探索与利用）
    pub exploration_rate: f64,
    /// 目标权重映射
    pub objective_weights: HashMap<String, f64>,
    /// 权重衰减（用于迭代优化）
    pub weight_decay: f64,
    /// 提前停止条件
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// 自适应搜索设置
    pub adaptive_search: bool,
    /// 索引类型筛选
    pub index_type_filter: Option<Vec<IndexType>>,
    /// 自动选择索引类型
    pub auto_select_index_type: bool,
    /// 缓存优化结果
    pub cache_results: bool,
    /// 优化后验证
    pub validate_after_optimize: bool,
    /// 训练数据集配置
    pub training_dataset: Option<DatasetConfig>,
    /// 验证数据集配置
    pub validation_dataset: Option<DatasetConfig>,
}

/// 提前停止配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// 停止的耐心值
    pub patience: usize,
    /// 最小提升阈值
    pub min_improvement: f64,
    /// 是否监视验证性能
    pub monitor_validation: bool,
}

impl Default for AutoTuneConfig {
    fn default() -> Self {
        Self {
            target: OptimizationTarget::BalancedPerformance,
            max_iterations: 50,
            parallel: true,
            random_seed: None,
            use_bayesian: true,
            use_grid_search: false,
            use_random_search: false,
            use_genetic_algorithm: false,
            use_multi_objective: false,
            multi_objective_algorithm: MultiObjectiveAlgorithm::NSGA2,
            benchmark_config: BenchmarkConfig::default(),
            learning_rate: 0.01,
            exploration_rate: 0.2,
            objective_weights: HashMap::new(),
            weight_decay: 0.95,
            early_stopping: Some(EarlyStoppingConfig {
                patience: 5,
                min_improvement: 0.01,
                monitor_validation: true,
            }),
            adaptive_search: true,
            index_type_filter: None,
            auto_select_index_type: true,
            cache_results: true,
            validate_after_optimize: true,
            training_dataset: None,
            validation_dataset: None,
        }
    }
}

/// 自动调优优化状态
#[derive(Debug, Clone)]
pub struct OptimizationState {
    /// 当前迭代
    pub current_iteration: usize,
    /// 当前最佳分数
    pub best_score: f64,
    /// 当前最佳配置
    pub best_config: IndexConfig,
    /// 当前最佳索引类型
    pub best_index_type: IndexType,
    /// 无改进的迭代次数
    pub iterations_without_improvement: usize,
    /// 尝试过的配置
    pub tried_configs: Vec<(IndexType, IndexConfig, f64)>,
    /// 搜索空间探索状态
    pub search_space_state: HashMap<String, SearchSpaceStat>,
    /// 参数敏感度分析
    pub parameter_sensitivity: HashMap<String, f64>,
    /// 开始时间
    pub start_time: Instant,
    /// 已消耗时间
    pub elapsed_time: Duration,
}

/// 参数搜索空间统计
#[derive(Debug, Clone)]
pub struct SearchSpaceStat {
    /// 参数名称
    pub name: String,
    /// 尝试的值
    pub tried_values: Vec<f64>,
    /// 最佳值
    pub best_value: f64,
    /// 值分布
    pub distribution: HashMap<String, usize>,
    /// 探索覆盖率 (0-1)
    pub exploration_coverage: f64,
}

/// 向量索引自动调优器
pub struct VectorIndexAutoTuner {
    /// 调优配置
    config: AutoTuneConfig,
    /// 基准测试器
    benchmark: IndexBenchmark,
    /// 评分函数
    scoring_fn: Option<Box<dyn Fn(&BenchmarkResult) -> f64 + Send + Sync>>,
    /// 优化维度
    dimensions: Vec<OptimizationDimension>,
    /// 随机数生成器
    rng: StdRng,
    /// 优化状态
    state: Arc<Mutex<OptimizationState>>,
    /// 结果缓存
    result_cache: Arc<RwLock<HashMap<String, OptimizationResult>>>,
    /// 多目标优化器
    multi_objective_optimizer: Option<Box<dyn MultiObjectiveOptimizer>>,
    /// 索引类型优化器映射
    index_optimizers: HashMap<IndexType, IndexOptimizer>,
}

impl VectorIndexAutoTuner {
    /// 创建新的向量索引自动调优器
    pub fn new(config: AutoTuneConfig) -> Self {
        let benchmark = IndexBenchmark::new(config.benchmark_config.clone());
        let seed = config.random_seed.unwrap_or_else(|| rand::random());
        let rng = StdRng::seed_from_u64(seed);
        
        let state = Arc::new(Mutex::new(OptimizationState {
            current_iteration: 0,
            best_score: 0.0,
            best_config: IndexConfig::default(),
            best_index_type: IndexType::Flat,
            iterations_without_improvement: 0,
            tried_configs: Vec::new(),
            search_space_state: HashMap::new(),
            parameter_sensitivity: HashMap::new(),
            start_time: Instant::now(),
            elapsed_time: Duration::from_secs(0),
        }));
        
        Self {
            config,
            benchmark,
            scoring_fn: None,
            dimensions: Vec::new(),
            rng,
            state,
            result_cache: Arc::new(RwLock::new(HashMap::new())),
            multi_objective_optimizer: None,
            index_optimizers: HashMap::new(),
        }
    }
    
    /// 设置评分函数
    pub fn with_scoring_function<F>(mut self, scoring_fn: F) -> Self 
    where
        F: Fn(&BenchmarkResult) -> f64 + Send + Sync + 'static,
    {
        self.scoring_fn = Some(Box::new(scoring_fn));
        self
    }
    
    /// 添加优化维度
    pub fn add_dimension(&mut self, dimension: OptimizationDimension) {
        self.dimensions.push(dimension);
    }
    
    /// 设置多目标优化器
    pub fn set_multi_objective_optimizer(&mut self, algorithm: MultiObjectiveAlgorithm) -> Result<()> {
        let mo_config = MultiObjectiveConfig {
            algorithm,
            population_size: 50,
            max_iterations: self.config.max_iterations,
            objectives_count: 3, // 默认三个目标：速度、内存、准确度
            seed: self.config.random_seed,
            ..Default::default()
        };
        
        self.multi_objective_optimizer = Some(create_mo_optimizer(algorithm, mo_config)?);
        Ok(())
    }
    
    /// 运行自动调优
    pub fn auto_tune(&mut self) -> Result<HashMap<IndexType, OptimizationResult>> {
        info!("开始向量索引自动调优...");
        let start_time = Instant::now();
        
        // 如果启用多目标优化，使用多目标方法
        if self.config.use_multi_objective {
            self.run_multi_objective_optimization()
        } else {
            // 否则使用单目标优化
            self.run_single_objective_optimization()
        }
    }
    
    /// 运行单目标优化
    fn run_single_objective_optimization(&mut self) -> Result<HashMap<IndexType, OptimizationResult>> {
        let mut results = HashMap::new();
        let index_types = self.get_index_types_to_optimize();
        
        if self.config.parallel {
            // 先基于当前调优器配置构建每个索引类型对应的优化任务，避免在并行闭包中捕获 &self
            let tasks: Vec<(IndexType, OptimizerConfig)> = index_types
                .iter()
                .map(|&index_type| {
                    let optimizer_config = self.create_optimizer_config();
                    (index_type, optimizer_config)
                })
                .collect();

            // 并行执行各索引类型的优化任务
            let thread_results: Vec<(IndexType, Result<OptimizationResult>)> = tasks
                .into_par_iter()
                .map(|(index_type, optimizer_config)| {
                    let optimizer = IndexOptimizer::new(optimizer_config);
                    let result = optimizer.optimize(index_type);
                    (index_type, result)
                })
                .collect();

            for (index_type, result) in thread_results {
                match result {
                    Ok(optimization_result) => {
                        results.insert(index_type, optimization_result);
                    }
                    Err(e) => {
                        warn!("优化索引类型 {:?} 时发生错误: {}", index_type, e);
                    }
                }
            }
        } else {
            // 串行优化各索引类型
            for &index_type in &index_types {
                let optimizer_config = self.create_optimizer_config();
                let optimizer = IndexOptimizer::new(optimizer_config);

                match optimizer.optimize(index_type) {
                    Ok(optimization_result) => {
                        results.insert(index_type, optimization_result);
                    }
                    Err(e) => {
                        warn!("优化索引类型 {:?} 时发生错误: {}", index_type, e);
                    }
                }
            }
        }
        
        // 如果启用自动选择索引类型，选择最佳结果
        if self.config.auto_select_index_type && !results.is_empty() {
            let mut best_index_type = None;
            let mut best_score = 0.0;
            
            for (&index_type, result) in &results {
                if result.score > best_score {
                    best_score = result.score;
                    best_index_type = Some(index_type);
                }
            }
            
            if let Some(best_type) = best_index_type {
                info!("自动选择最佳索引类型: {:?}", best_type);
                
                // 更新优化状态
                let mut state = self.state.lock().unwrap();
                state.best_index_type = best_type;
                if let Some(result) = results.get(&best_type) {
                    state.best_config = result.best_config.clone();
                    state.best_score = result.score;
                }
            }
        }
        
        // 如果启用验证，验证优化结果
        if self.config.validate_after_optimize {
            self.validate_optimization_results(&results)?;
        }
        
        // 缓存结果
        if self.config.cache_results {
            let mut cache = self.result_cache.write().unwrap();
            for (index_type, result) in &results {
                cache.insert(format!("{:?}", index_type), result.clone());
            }
        }
        
        Ok(results)
    }
    
    /// 运行多目标优化
    fn run_multi_objective_optimization(&mut self) -> Result<HashMap<IndexType, OptimizationResult>> {
        if self.multi_objective_optimizer.is_none() {
            self.set_multi_objective_optimizer(self.config.multi_objective_algorithm)?;
        }
        
        let mut results = HashMap::new();
        let index_types = self.get_index_types_to_optimize();
        
        // 使用 Arc 包装需要的字段以便闭包捕获
        let benchmark = Arc::new(self.benchmark.clone());
        let target = self.config.target;
        
        for &index_type in &index_types {
            let parameter_space = ParameterSpace::for_index_type(index_type);
            let bounds = parameter_space.ranges.iter()
                .map(|range| (range.min, range.max))
                .collect::<Vec<(f64, f64)>>();
            
            let benchmark_clone = benchmark.clone();
            let query_speed_objective: Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync> = Box::new({
                let index_type = index_type;
                let benchmark = benchmark_clone.clone();
                let parameter_space = parameter_space.clone();
                move |params: &Vec<f64>| -> f64 {
                    let mut config = IndexConfig::default();
                    config.index_type = index_type;
                    for (i, range) in parameter_space.ranges.iter().enumerate() {
                        if i < params.len() {
                            // 简化参数应用逻辑
                            let param_name = range.name.as_str();
                            let value = params[i];
                            match param_name {
                                "M" => config.hnsw_m = value as usize,
                                "ef_construction" => config.hnsw_ef_construction = value as usize,
                                "ef" => config.hnsw_ef_search = value as usize,
                                "nlist" => config.ivf_nlist = value as usize,
                                "nprobe" => config.ivf_nprobe = value as usize,
                                "m" => config.pq_m = value as usize,
                                "nbits" => config.pq_nbits = value as usize,
                                "tables" => config.lsh_ntables = value as usize,
                                "bits_per_table" => config.lsh_hash_length = value as usize,
                                "k" => config.search_k = value as usize,
                                _ => {}
                            }
                        }
                    }
                    match benchmark.benchmark_index(index_type, config) {
                        Ok(result) => -1.0 * result.avg_query_time_ms,
                        Err(_) => 0.0,
                    }
                }
            });
            
            let benchmark_clone = benchmark.clone();
            let memory_usage_objective: Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync> = Box::new({
                let index_type = index_type;
                let benchmark = benchmark_clone.clone();
                let parameter_space = parameter_space.clone();
                move |params: &Vec<f64>| -> f64 {
                    let mut config = IndexConfig::default();
                    config.index_type = index_type;
                    for (i, range) in parameter_space.ranges.iter().enumerate() {
                        if i < params.len() {
                            let param_name = range.name.as_str();
                            let value = params[i];
                            match param_name {
                                "M" => config.hnsw_m = value as usize,
                                "ef_construction" => config.hnsw_ef_construction = value as usize,
                                "ef" => config.hnsw_ef_search = value as usize,
                                "nlist" => config.ivf_nlist = value as usize,
                                "nprobe" => config.ivf_nprobe = value as usize,
                                "m" => config.pq_m = value as usize,
                                "nbits" => config.pq_nbits = value as usize,
                                "tables" => config.lsh_ntables = value as usize,
                                "bits_per_table" => config.lsh_hash_length = value as usize,
                                "k" => config.search_k = value as usize,
                                _ => {}
                            }
                        }
                    }
                    match benchmark.benchmark_index(index_type, config) {
                        Ok(result) => -1.0 * (result.memory_usage_bytes as f64),
                        Err(_) => 0.0,
                    }
                }
            });
            
            let benchmark_clone = benchmark.clone();
            let accuracy_objective: Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync> = Box::new({
                let index_type = index_type;
                let benchmark = benchmark_clone.clone();
                let parameter_space = parameter_space.clone();
                move |params: &Vec<f64>| -> f64 {
                    let mut config = IndexConfig::default();
                    config.index_type = index_type;
                    for (i, range) in parameter_space.ranges.iter().enumerate() {
                        if i < params.len() {
                            let param_name = range.name.as_str();
                            let value = params[i];
                            match param_name {
                                "M" => config.hnsw_m = value as usize,
                                "ef_construction" => config.hnsw_ef_construction = value as usize,
                                "ef" => config.hnsw_ef_search = value as usize,
                                "nlist" => config.ivf_nlist = value as usize,
                                "nprobe" => config.ivf_nprobe = value as usize,
                                "m" => config.pq_m = value as usize,
                                "nbits" => config.pq_nbits = value as usize,
                                "tables" => config.lsh_ntables = value as usize,
                                "bits_per_table" => config.lsh_hash_length = value as usize,
                                "k" => config.search_k = value as usize,
                                _ => {}
                            }
                        }
                    }
                    match benchmark.benchmark_index(index_type, config) {
                        Ok(result) => result.accuracy,
                        Err(_) => 0.0,
                    }
                }
            });
            
            let objectives: Vec<Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>> = match target {
                OptimizationTarget::QuerySpeed => {
                    vec![query_speed_objective]
                },
                OptimizationTarget::MemoryUsage => {
                    vec![memory_usage_objective]
                },
                OptimizationTarget::Accuracy => {
                    vec![accuracy_objective]
                },
                OptimizationTarget::BalancedPerformance => {
                    vec![query_speed_objective, memory_usage_objective, accuracy_objective]
                },
                OptimizationTarget::BuildSpeed => {
                    let benchmark_clone = benchmark.clone();
                    let build_speed_objective: Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync> = Box::new({
                        let index_type = index_type;
                        let benchmark = benchmark_clone.clone();
                        let parameter_space = parameter_space.clone();
                        move |params: &Vec<f64>| -> f64 {
                            let mut config = IndexConfig::default();
                            config.index_type = index_type;
                            for (i, range) in parameter_space.ranges.iter().enumerate() {
                                if i < params.len() {
                                    let param_name = range.name.as_str();
                                    let value = params[i];
                                    match param_name {
                                        "M" => config.hnsw_m = value as usize,
                                        "ef_construction" => config.hnsw_ef_construction = value as usize,
                                        "ef" => config.hnsw_ef_search = value as usize,
                                        "nlist" => config.ivf_nlist = value as usize,
                                        "nprobe" => config.ivf_nprobe = value as usize,
                                        "m" => config.pq_m = value as usize,
                                        "nbits" => config.pq_nbits = value as usize,
                                        "tables" => config.lsh_ntables = value as usize,
                                        "bits_per_table" => config.lsh_hash_length = value as usize,
                                        "k" => config.search_k = value as usize,
                                        _ => {}
                                    }
                                }
                            }
                            match benchmark.benchmark_index(index_type, config) {
                                Ok(result) => -1.0 * (result.build_time_ms as f64),
                                Err(_) => 0.0,
                            }
                        }
                    });
                    vec![build_speed_objective]
                },
            };
            
            // 运行多目标优化
            let mo_result = if let Some(mo_optimizer) = &mut self.multi_objective_optimizer {
                mo_optimizer.optimize(objectives.as_slice(), &bounds)?
            } else {
                return Err(Error::internal("多目标优化器未初始化"));
            };
            
            // 转换为单目标优化结果
            if let Some(best_solution) = mo_result.pareto_front.first() {
                let mut config = IndexConfig::default();
                config.index_type = index_type;
                
                for (i, range) in parameter_space.ranges.iter().enumerate() {
                    if i < best_solution.len() {
                        let param_value = best_solution[i];
                        self.apply_parameter_to_config(&mut config, range.name.as_str(), param_value)?;
                    }
                }
                
                // 评估最佳配置
                let benchmark_result = self.benchmark.benchmark_index(index_type, config.clone())?;
                let score = if let Some(scoring_fn) = &self.scoring_fn {
                    scoring_fn(&benchmark_result)
                } else {
                    scoring::calculate_score(&benchmark_result, self.config.target)
                };
                
                let optimization_result = OptimizationResult {
                    index_type,
                    best_config: config,
                    performance: benchmark_result,
                    evaluated_configs: mo_result.iterations as usize,
                    target: self.config.target,
                    score,
                };
                
                results.insert(index_type, optimization_result);
            }
        }
        
        // 如果启用自动选择索引类型，选择最佳结果
        if self.config.auto_select_index_type && !results.is_empty() {
            let mut best_index_type = None;
            let mut best_score = 0.0;
            
            for (&index_type, result) in &results {
                if result.score > best_score {
                    best_score = result.score;
                    best_index_type = Some(index_type);
                }
            }
            
            if let Some(best_type) = best_index_type {
                info!("多目标优化自动选择最佳索引类型: {:?}", best_type);
                
                // 更新优化状态
                let mut state = self.state.lock().unwrap();
                state.best_index_type = best_type;
                if let Some(result) = results.get(&best_type) {
                    state.best_config = result.best_config.clone();
                    state.best_score = result.score;
                }
            }
        }
        
        Ok(results)
    }
    
    /// 创建优化器配置
    fn create_optimizer_config(&self) -> OptimizerConfig {
        OptimizerConfig {
            benchmark_config: self.config.benchmark_config.clone(),
            target: self.config.target,
            max_iterations: self.config.max_iterations,
            parallel: self.config.parallel,
            random_seed: self.config.random_seed,
            use_bayesian: self.config.use_bayesian,
            use_grid_search: self.config.use_grid_search,
            use_random_search: self.config.use_random_search,
            use_genetic_algorithm: self.config.use_genetic_algorithm,
            dataset: self.config.training_dataset.clone(),
        }
    }
    
    /// 获取要优化的索引类型
    fn get_index_types_to_optimize(&self) -> Vec<IndexType> {
        if let Some(filter) = &self.config.index_type_filter {
            filter.clone()
        } else {
            vec![
                IndexType::Flat,
                IndexType::HNSW,
                IndexType::IVF,
                IndexType::IVFPQ,
                IndexType::PQ,
                IndexType::LSH,
                IndexType::VPTree,
            ]
        }
    }
    
    /// 验证优化结果
    fn validate_optimization_results(&self, results: &HashMap<IndexType, OptimizationResult>) -> Result<()> {
        if let Some(validation_dataset) = &self.config.validation_dataset {
            info!("使用验证数据集验证优化结果...");
            // 这里可以实现验证逻辑
        }
        
        Ok(())
    }
    
    /// 评估查询速度
    fn evaluate_query_speed(&self, index_type: IndexType, params: &Vec<f64>) -> f64 {
        let mut config = IndexConfig::default();
        config.index_type = index_type;
        
        let parameter_space = ParameterSpace::for_index_type(index_type);
        for (i, range) in parameter_space.ranges.iter().enumerate() {
            if i < params.len() {
                if let Err(e) = self.apply_parameter_to_config(&mut config, &range.name, params[i]) {
                    warn!("应用参数 {} 时出错: {}", range.name, e);
                    return 0.0;
                }
            }
        }
        
        match self.benchmark.benchmark_index(index_type, config) {
            Ok(result) => {
                // 查询速度倒数，乘以-1使其成为最小化目标
                -1.0 * result.avg_query_time_ms
            },
            Err(_) => 0.0,
        }
    }
    
    /// 评估内存使用
    fn evaluate_memory_usage(&self, index_type: IndexType, params: &Vec<f64>) -> f64 {
        let mut config = IndexConfig::default();
        config.index_type = index_type;
        
        let parameter_space = ParameterSpace::for_index_type(index_type);
        for (i, range) in parameter_space.ranges.iter().enumerate() {
            if i < params.len() {
                if let Err(e) = self.apply_parameter_to_config(&mut config, &range.name, params[i]) {
                    warn!("应用参数 {} 时出错: {}", range.name, e);
                    return 0.0;
                }
            }
        }
        
        match self.benchmark.benchmark_index(index_type, config) {
            Ok(result) => {
                // 内存使用字节数，乘以-1使其成为最小化目标
                -1.0 * (result.memory_usage_bytes as f64)
            },
            Err(_) => 0.0,
        }
    }
    
    /// 评估准确度
    fn evaluate_accuracy(&self, index_type: IndexType, params: &Vec<f64>) -> f64 {
        let mut config = IndexConfig::default();
        config.index_type = index_type;
        
        let parameter_space = ParameterSpace::for_index_type(index_type);
        for (i, range) in parameter_space.ranges.iter().enumerate() {
            if i < params.len() {
                if let Err(e) = self.apply_parameter_to_config(&mut config, &range.name, params[i]) {
                    warn!("应用参数 {} 时出错: {}", range.name, e);
                    return 0.0;
                }
            }
        }
        
        match self.benchmark.benchmark_index(index_type, config) {
            Ok(result) => {
                // 准确度是0-1之间的值，越大越好
                result.accuracy
            },
            Err(_) => 0.0,
        }
    }
    
    /// 评估构建速度
    fn evaluate_build_speed(&self, index_type: IndexType, params: &Vec<f64>) -> f64 {
        let mut config = IndexConfig::default();
        config.index_type = index_type;
        
        let parameter_space = ParameterSpace::for_index_type(index_type);
        for (i, range) in parameter_space.ranges.iter().enumerate() {
            if i < params.len() {
                if let Err(e) = self.apply_parameter_to_config(&mut config, &range.name, params[i]) {
                    warn!("应用参数 {} 时出错: {}", range.name, e);
                    return 0.0;
                }
            }
        }
        
        match self.benchmark.benchmark_index(index_type, config) {
            Ok(result) => {
                // 构建时间倒数，乘以-1使其成为最小化目标
                -1.0 * (result.build_time_ms as f64)
            },
            Err(_) => 0.0,
        }
    }
    
    /// 将参数应用到配置
    fn apply_parameter_to_config(&self, config: &mut IndexConfig, param_name: &str, value: f64) -> Result<()> {
        match param_name {
            // HNSW参数
            "M" => config.hnsw_m = value as usize,
            "ef_construction" => config.hnsw_ef_construction = value as usize,
            "ef" => config.hnsw_ef_search = value as usize,
            
            // IVF参数
            "nlist" => config.ivf_nlist = value as usize,
            "nprobe" => config.ivf_nprobe = value as usize,
            
            // PQ参数
            "m" => config.pq_m = value as usize,
            "nbits" => config.pq_nbits = value as usize,
            
            // LSH参数
            "tables" => config.lsh_ntables = value as usize,
            "bits_per_table" => config.lsh_hash_length = value as usize,
            
            // 通用参数
            "k" => config.search_k = value as usize,
            
            // 其他参数，根据需要扩展
            _ => return Err(Error::invalid_argument(format!("未知参数: {}", param_name))),
        }
        
        Ok(())
    }
    
    /// 生成优化报告
    pub fn generate_report(&self, results: &HashMap<IndexType, OptimizationResult>) -> String {
        let mut report = String::new();
        report.push_str("# 向量索引优化报告\n\n");
        
        // 添加配置信息
        report.push_str("## 优化配置\n\n");
        report.push_str(&format!("- 优化目标: {:?}\n", self.config.target));
        report.push_str(&format!("- 最大迭代次数: {}\n", self.config.max_iterations));
        report.push_str(&format!("- 使用贝叶斯优化: {}\n", self.config.use_bayesian));
        report.push_str(&format!("- 使用网格搜索: {}\n", self.config.use_grid_search));
        report.push_str(&format!("- 使用随机搜索: {}\n", self.config.use_random_search));
        report.push_str(&format!("- 使用遗传算法: {}\n", self.config.use_genetic_algorithm));
        report.push_str(&format!("- 使用多目标优化: {}\n", self.config.use_multi_objective));
        if self.config.use_multi_objective {
            report.push_str(&format!("- 多目标算法: {:?}\n", self.config.multi_objective_algorithm));
        }
        report.push_str("\n");
        
        // 对结果排序
        let mut sorted_results: Vec<(&IndexType, &OptimizationResult)> = results.iter().collect();
        sorted_results.sort_by(|(_, a), (_, b)| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        // 添加最佳结果
        if let Some((best_index_type, best_result)) = sorted_results.first() {
            report.push_str("## 最佳索引配置\n\n");
            report.push_str(&format!("- 索引类型: {:?}\n", best_index_type));
            report.push_str(&format!("- 优化分数: {:.4}\n", best_result.score));
            report.push_str(&format!("- 平均查询时间: {:.4} ms\n", best_result.performance.avg_query_time_ms));
            report.push_str(&format!("- 查询吞吐量: {:.2} QPS\n", best_result.performance.queries_per_second));
            report.push_str(&format!("- 准确率: {:.4}\n", best_result.performance.accuracy));
            report.push_str(&format!("- 内存使用: {:.2} MB\n", best_result.performance.memory_usage_bytes as f64 / (1024.0 * 1024.0)));
            report.push_str(&format!("- 索引大小: {:.2} MB\n", best_result.performance.index_size_bytes as f64 / (1024.0 * 1024.0)));
            report.push_str(&format!("- 构建时间: {:.2} ms\n", best_result.performance.build_time_ms));
            report.push_str("\n### 最佳配置参数\n\n");
            
            let config = &best_result.best_config;
            report.push_str(&format!("```json\n{}\n```\n\n", serde_json::to_string_pretty(config).unwrap_or_default()));
        }
        
        // 添加所有索引类型的结果比较
        report.push_str("## 索引类型比较\n\n");
        report.push_str("| 索引类型 | 优化分数 | 查询时间(ms) | 准确率 | 内存使用(MB) | 索引大小(MB) |\n");
        report.push_str("|----------|----------|--------------|--------|--------------|-------------|\n");
        
        for (index_type, result) in &sorted_results {
            report.push_str(&format!("| {:?} | {:.4} | {:.4} | {:.4} | {:.2} | {:.2} |\n",
                index_type,
                result.score,
                result.performance.avg_query_time_ms,
                result.performance.accuracy,
                result.performance.memory_usage_bytes as f64 / (1024.0 * 1024.0),
                result.performance.index_size_bytes as f64 / (1024.0 * 1024.0)
            ));
        }
        
        // 添加参数敏感度分析
        if let Ok(state) = self.state.lock() {
            if !state.parameter_sensitivity.is_empty() {
                report.push_str("\n## 参数敏感度分析\n\n");
                report.push_str("| 参数 | 敏感度 |\n");
                report.push_str("|------|--------|\n");
                
                // 对敏感度排序
                let mut sensitivities: Vec<(&String, &f64)> = state.parameter_sensitivity.iter().collect();
                sensitivities.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                
                for (param, sensitivity) in sensitivities {
                    report.push_str(&format!("| {} | {:.4} |\n", param, sensitivity));
                }
            }
        }
        
        // 添加优化过程
        report.push_str("\n## 优化过程\n\n");
        if let Ok(state) = self.state.lock() {
            report.push_str(&format!("- 迭代次数: {}\n", state.current_iteration));
            report.push_str(&format!("- 运行时间: {:.2} 秒\n", state.elapsed_time.as_secs_f64()));
            report.push_str(&format!("- 尝试的配置数: {}\n", state.tried_configs.len()));
        }
        
        report
    }
    
    /// 运行优化示例
    pub fn run_autotune_example() -> Result<String> {
        info!("运行向量索引自动调优示例...");
        
        // 创建调优器配置
        let config = AutoTuneConfig {
            target: OptimizationTarget::BalancedPerformance,
            max_iterations: 20,
            use_bayesian: true,
            use_multi_objective: false,
            benchmark_config: BenchmarkConfig {
                dataset_size: 5000,
                dimension: 128,
                query_size: 50,
                ..Default::default()
            },
            ..Default::default()
        };
        
        // 创建调优器
        let mut tuner = VectorIndexAutoTuner::new(config);
        
        // 运行优化
        let results = tuner.auto_tune()?;
        
        // 生成报告
        let report = tuner.generate_report(&results);
        info!("自动调优完成");
        
        Ok(report)
    }
}

/// 工厂函数
pub fn create_auto_tuner(config: AutoTuneConfig) -> VectorIndexAutoTuner {
    VectorIndexAutoTuner::new(config)
}

/// 导出任务示例
pub fn run_optimizer_example() -> Result<String> {
    VectorIndexAutoTuner::run_autotune_example()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_auto_tuner() {
        let config = AutoTuneConfig {
            max_iterations: 2, // 减少迭代次数以加快测试
            benchmark_config: BenchmarkConfig {
                dataset_size: 100, // 减少数据集大小以加快测试
                dimension: 32,
                query_size: 10,
                ..Default::default()
            },
            ..Default::default()
        };
        
        let mut tuner = VectorIndexAutoTuner::new(config);
        match tuner.auto_tune() {
            Ok(results) => {
                assert!(!results.is_empty(), "优化结果不应为空");
                // 检查结果中是否至少有一个索引类型
                assert!(results.contains_key(&IndexType::Flat) || 
                        results.contains_key(&IndexType::HNSW) || 
                        results.contains_key(&IndexType::IVF));
            },
            Err(e) => {
                panic!("自动调优测试失败: 预期成功完成自动调优，但遇到错误: {}", e);
            }
        }
    }
} 