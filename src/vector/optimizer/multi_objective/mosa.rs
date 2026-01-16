use std::collections::HashMap;
use std::time::Instant;
use rand::{rngs::StdRng, SeedableRng, Rng, seq::SliceRandom};
use crate::error::{Error, Result};
use super::types::{MultiObjectiveConfig, MultiObjectiveResult, ObjectiveDirection, Solution};
use super::MultiObjectiveOptimizer;

/// 多目标模拟退火算法优化器
pub struct MOSAOptimizer {
    /// 配置
    config: MultiObjectiveConfig,
    /// 随机数生成器
    rng: StdRng,
    /// 当前温度
    temperature: f64,
    /// 初始温度
    initial_temperature: f64,
    /// 冷却率
    cooling_rate: f64,
    /// 每个温度下的迭代次数
    iterations_per_temperature: usize,
    /// 存档大小
    archive_size: usize,
    /// 非支配解集（帕累托前沿）存档
    archive: Vec<(Vec<f64>, Vec<f64>)>,
}

impl MOSAOptimizer {
    /// 创建一个新的多目标模拟退火优化器
    pub fn new(config: MultiObjectiveConfig) -> Self {
        let seed = config.seed.unwrap_or_else(|| rand::random::<u64>());
        let rng = StdRng::seed_from_u64(seed);
        
        // 提取MOSA算法特定参数（如果未设置则使用默认值）
        let initial_temperature = *config.algorithm_params.get("initial_temperature").unwrap_or(&100.0);
        let cooling_rate = *config.algorithm_params.get("cooling_rate").unwrap_or(&0.95);
        let iterations_per_temperature = *config.algorithm_params.get("iterations_per_temperature").unwrap_or(&10.0) as usize;
        let archive_size = config.algorithm_params.get("archive_size")
            .map(|&v| v as usize)
            .unwrap_or(100);
        
        Self {
            config,
            rng,
            temperature: initial_temperature,
            initial_temperature,
            cooling_rate,
            iterations_per_temperature,
            archive_size,
            archive: Vec::new(),
        }
    }
    
    /// 生成初始解
    fn generate_initial_solution(&mut self, bounds: &[(f64, f64)]) -> Vec<f64> {
        let mut solution = Vec::with_capacity(bounds.len());
        
        for &(lower, upper) in bounds {
            let value = self.rng.gen_range(lower..upper);
            solution.push(value);
        }
        
        solution
    }
    
    /// 生成邻居解
    fn generate_neighbor(&mut self, current: &[f64], bounds: &[(f64, f64)], temperature_ratio: f64) -> Vec<f64> {
        let mut neighbor = current.to_vec();
        let dimension = current.len();
        
        // 计算扰动幅度（随温度降低而减小）
        let perturbation_scale = temperature_ratio; // 扰动幅度随温度比例变化
        
        // 选择要扰动的维度数量（随机）
        let dimensions_to_perturb = (1.0 + self.rng.gen::<f64>() * (dimension as f64 * 0.5)) as usize;
        let dimensions_to_perturb = dimensions_to_perturb.max(1).min(dimension);
        
        // 随机选择维度进行扰动
        let mut indices: Vec<usize> = (0..dimension).collect();
        indices.shuffle(&mut self.rng);
        
        for &idx in indices.iter().take(dimensions_to_perturb) {
            let (lower, upper) = bounds[idx];
            let range = upper - lower;
            
            // 扰动大小随温度变化
            let perturbation = (self.rng.gen::<f64>() - 0.5) * range * perturbation_scale;
            
            // 应用扰动并保证在边界内
            neighbor[idx] = (neighbor[idx] + perturbation).max(lower).min(upper);
        }
        
        neighbor
    }
    
    /// 检查一个解是否支配另一个解
    fn dominates(&self, a: &[f64], b: &[f64]) -> bool {
        let mut has_better = false;
        
        for i in 0..a.len() {
            let is_better = match self.config.objective_directions.get(i).unwrap_or(&ObjectiveDirection::Minimize) {
                ObjectiveDirection::Minimize => a[i] < b[i],
                ObjectiveDirection::Maximize => a[i] > b[i],
            };
            
            let is_worse = match self.config.objective_directions.get(i).unwrap_or(&ObjectiveDirection::Minimize) {
                ObjectiveDirection::Minimize => a[i] > b[i],
                ObjectiveDirection::Maximize => a[i] < b[i],
            };
            
            if is_worse {
                return false;
            }
            
            if is_better {
                has_better = true;
            }
        }
        
        has_better
    }
    
    /// 计算模拟退火接受概率
    fn calculate_acceptance_probability(&self, current_energy: &[f64], new_energy: &[f64], temperature: f64) -> f64 {
        // 如果新解支配当前解，则一定接受
        if self.dominates(new_energy, current_energy) {
            return 1.0;
        }
        
        // 如果当前解支配新解，则根据温度和能量差计算接受概率
        if self.dominates(current_energy, new_energy) {
            // 计算能量差的平均值（考虑所有目标）
            let mut energy_diff = 0.0;
            let mut objectives_count = 0;
            
            for i in 0..current_energy.len() {
                let diff = match self.config.objective_directions.get(i).unwrap_or(&ObjectiveDirection::Minimize) {
                    ObjectiveDirection::Minimize => new_energy[i] - current_energy[i],
                    ObjectiveDirection::Maximize => current_energy[i] - new_energy[i],
                };
                
                // 只考虑负向差异（即变差的目标）
                if diff > 0.0 {
                    energy_diff += diff;
                    objectives_count += 1;
                }
            }
            
            // 如果没有变差的目标，则接受概率为0.5
            if objectives_count == 0 {
                return 0.5;
            }
            
            // 计算平均能量差
            let avg_energy_diff = energy_diff / objectives_count as f64;
            
            // 计算接受概率（Boltzmann分布）
            let probability = (-avg_energy_diff / temperature).exp();
            return probability;
        }
        
        // 如果两个解互不支配，则接受概率为0.5
        0.5
    }
    
    /// 更新非支配解集（帕累托集）
    fn update_archive(&self, archive: &mut Vec<(Vec<f64>, Vec<f64>)>, solution: Vec<f64>, objectives: Vec<f64>) {
        // 检查新解是否被现有解支配
        let mut is_dominated = false;
        let mut dominated_indices = Vec::new();
        
        for (i, (_, existing_obj)) in archive.iter().enumerate() {
            if self.dominates(existing_obj, &objectives) {
                is_dominated = true;
                break;
            } else if self.dominates(&objectives, existing_obj) {
                dominated_indices.push(i);
            }
        }
        
        // 如果新解不被支配，则添加到存档
        if !is_dominated {
            // 移除被新解支配的解
            dominated_indices.sort_unstable_by(|a, b| b.cmp(a)); // 从后往前删除
            for idx in dominated_indices {
                archive.remove(idx);
            }
            
            // 添加新解
            archive.push((solution, objectives));
            
            // 如果存档大小超过限制，则根据拥挤度进行裁剪
            if archive.len() > self.archive_size {
                self.prune_archive(archive);
            }
        }
    }
    
    /// 裁剪存档，保持解的多样性
    fn prune_archive(&self, archive: &mut Vec<(Vec<f64>, Vec<f64>)>) {
        if archive.is_empty() {
            return;
        }
        
        // 计算每个解的拥挤度
        let mut crowding_distances = vec![0.0; archive.len()];
        let objectives_count = archive[0].1.len();
        
        for obj_idx in 0..objectives_count {
            // 按当前目标函数值排序
            let mut sorted_indices: Vec<usize> = (0..archive.len()).collect();
            sorted_indices.sort_by(|&a, &b| {
                let a_val = archive[a].1[obj_idx];
                let b_val = archive[b].1[obj_idx];
                a_val.partial_cmp(&b_val).unwrap_or(std::cmp::Ordering::Equal)
            });
            
            // 边界解的拥挤度设为无穷大
            crowding_distances[sorted_indices[0]] = f64::INFINITY;
            crowding_distances[sorted_indices[archive.len() - 1]] = f64::INFINITY;
            
            // 计算中间解的拥挤度
            let obj_range = archive[sorted_indices[archive.len() - 1]].1[obj_idx] - 
                           archive[sorted_indices[0]].1[obj_idx];
            
            if obj_range > 1e-10 {
                for i in 1..archive.len() - 1 {
                    let next_val = archive[sorted_indices[i + 1]].1[obj_idx];
                    let prev_val = archive[sorted_indices[i - 1]].1[obj_idx];
                    
                    crowding_distances[sorted_indices[i]] += (next_val - prev_val) / obj_range;
                }
            }
        }
        
        // 根据拥挤度排序（拥挤度低的先删除）
        let mut indices_with_distances: Vec<(usize, f64)> = crowding_distances.iter()
            .enumerate()
            .map(|(i, &d)| (i, d))
            .collect();
        
        indices_with_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // 删除多余解
        let to_remove = archive.len() - self.archive_size;
        let mut removed = 0;
        let mut remove_indices = Vec::new();
        
        for (idx, _) in indices_with_distances {
            remove_indices.push(idx);
            removed += 1;
            if removed >= to_remove {
                break;
            }
        }
        
        // 从后往前删除，避免索引变化
        remove_indices.sort_unstable_by(|a, b| b.cmp(a));
        for idx in remove_indices {
            archive.remove(idx);
        }
    }
    
    /// 计算超体积指标
    fn calculate_hypervolume(&self, objective_values: &[Vec<f64>]) -> Option<f64> {
        if objective_values.is_empty() {
            return None;
        }
        
        // 对于二维问题，使用简单的面积计算
        if objective_values[0].len() == 2 {
            // 找到参考点
            let mut reference_point = vec![0.0; 2];
            for i in 0..2 {
                match self.config.objective_directions.get(i).unwrap_or(&ObjectiveDirection::Minimize) {
                    ObjectiveDirection::Minimize => {
                        reference_point[i] = objective_values.iter()
                            .map(|v| v[i])
                            .fold(f64::MAX, |a, b| a.max(b)) * 1.1;
                    },
                    ObjectiveDirection::Maximize => {
                        reference_point[i] = objective_values.iter()
                            .map(|v| v[i])
                            .fold(f64::MIN, |a, b| a.min(b)) * 0.9;
                    }
                }
            }
            
            // 按第一个目标排序
            let mut sorted_points = objective_values.to_vec();
            sorted_points.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
            
            // 计算面积
            let mut hypervolume = 0.0;
            let mut prev_x = reference_point[0];
            
            for point in &sorted_points {
                let x = match self.config.objective_directions.get(0).unwrap_or(&ObjectiveDirection::Minimize) {
                    ObjectiveDirection::Minimize => point[0],
                    ObjectiveDirection::Maximize => reference_point[0] - point[0],
                };
                
                let y = match self.config.objective_directions.get(1).unwrap_or(&ObjectiveDirection::Minimize) {
                    ObjectiveDirection::Minimize => reference_point[1] - point[1],
                    ObjectiveDirection::Maximize => point[1],
                };
                
                hypervolume += (prev_x - x).abs() * y;
                prev_x = x;
            }
            
            return Some(hypervolume);
        }
        
        // 对于更高维度，返回None（需要更复杂的实现）
        None
    }
}

impl MultiObjectiveOptimizer for MOSAOptimizer {
    fn optimize(
        &mut self, 
        objective_functions: &[Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>],
        bounds: &[(f64, f64)]
    ) -> Result<MultiObjectiveResult> {
        // 验证输入
        if objective_functions.len() != self.config.objectives_count {
            return Err(Error::invalid_argument(format!(
                "预期的目标函数数量为 {}，但实际提供了 {}", 
                self.config.objectives_count, 
                objective_functions.len()
            )));
        }
        
        if bounds.is_empty() {
            return Err(Error::invalid_argument("参数边界不能为空".to_string()));
        }
        
        // 记录开始时间
        let start_time = Instant::now();
        
        // 初始化温度
        self.temperature = self.initial_temperature;
        
        // 生成初始解
        let mut current_solution = self.generate_initial_solution(bounds);
        
        // 计算初始解的目标函数值
        let mut current_objectives = Vec::with_capacity(objective_functions.len());
        for obj_fn in objective_functions {
            current_objectives.push(obj_fn(&current_solution));
        }
        
        // 初始化非支配解集（帕累托前沿）
        self.archive.clear();
        self.update_archive(&mut self.archive, current_solution.clone(), current_objectives.clone());
        
        // 收敛历史记录
        let mut convergence_history = Vec::new();
        let mut early_stopped = false;
        let mut no_improvement_count = 0;
        let max_no_improvement = self.config.early_stopping
            .as_ref()
            .map(|es| es.patience)
            .unwrap_or(self.config.max_iterations / 10);
        
        // 主循环 - 多目标模拟退火
        for iteration in 0..self.config.max_iterations {
            let temperature_ratio = self.temperature / self.initial_temperature;
            
            // 记录当前迭代的超体积
            let current_hypervolume = self.calculate_hypervolume(&self.archive.iter().map(|(_, obj)| obj.clone()).collect::<Vec<_>>());
            if let Some(hv) = current_hypervolume {
                convergence_history.push(hv);
                
                // 检查早停条件
                if convergence_history.len() > max_no_improvement {
                    let prev_best = convergence_history[convergence_history.len() - max_no_improvement - 1];
                    let min_delta = self.config.early_stopping
                        .as_ref()
                        .map(|es| es.min_delta)
                        .unwrap_or(1e-4);
                    
                    if (hv - prev_best).abs() < min_delta {
                        no_improvement_count += 1;
                        if no_improvement_count >= max_no_improvement {
                            early_stopped = true;
                            break;
                        }
                    } else {
                        no_improvement_count = 0;
                    }
                }
            }
            
            // 每个温度下执行多次迭代
            for _ in 0..self.iterations_per_temperature {
                // 生成邻居解
                let neighbor_solution = self.generate_neighbor(&current_solution, bounds, temperature_ratio);
                
                // 计算邻居解的目标函数值
                let mut neighbor_objectives = Vec::with_capacity(objective_functions.len());
                for obj_fn in objective_functions {
                    neighbor_objectives.push(obj_fn(&neighbor_solution));
                }
                
                // 计算接受概率
                let acceptance_prob = self.calculate_acceptance_probability(
                    &current_objectives, &neighbor_objectives, self.temperature);
                
                // 决定是否接受新解
                if self.rng.gen::<f64>() < acceptance_prob {
                    // 避免将 neighbor_solution 移动走，后续还需要用于更新存档
                    current_solution = neighbor_solution.clone();
                    current_objectives = neighbor_objectives.clone();
                }
                
                // 更新非支配解集（帕累托前沿）
                self.update_archive(&mut self.archive, neighbor_solution, neighbor_objectives);
            }
            
            // 降温
            self.temperature *= self.cooling_rate;
            
            // 日志输出
            if (iteration + 1) % 10 == 0 || iteration == 0 {
                log::info!(
                    "MOSA迭代 {}/{}, 温度: {:.2e}, 存档大小: {}, 超体积: {:.4}", 
                    iteration + 1, 
                    self.config.max_iterations,
                    self.temperature,
                    self.archive.len(),
                    current_hypervolume.unwrap_or(-1.0)
                );
            }
            
            // 如果温度太低，提前终止
            if self.temperature < 1e-10 {
                log::info!("温度过低，MOSA提前终止");
                break;
            }
        }
        
        // 计算运行时间
        let runtime_ms = start_time.elapsed().as_millis() as u64;
        
        // 提取结果
        let pareto_front: Vec<Vec<f64>> = self.archive.iter().map(|(sol, _)| sol.clone()).collect();
        let objective_values: Vec<Vec<f64>> = self.archive.iter().map(|(_, obj)| obj.clone()).collect();
        
        // 计算超体积指标
        let hypervolume = self.calculate_hypervolume(&objective_values);
        
        // 构建所有解的详细信息
        let all_solutions = self.archive.iter().enumerate().map(|(i, (params, obj_vals))| {
            Solution {
                parameters: params.clone(),
                objective_values: obj_vals.iter().enumerate()
                    .map(|(i, &v)| (format!("f{}", i + 1), v))
                    .collect(),
                is_pareto_optimal: true, // 存档中的所有解都是非支配的
                rank: Some(1), // 所有解都是第一层
                crowding_distance: None, // 这里不计算拥挤度
                quality_metrics: HashMap::new(),
            }
        }).collect();
        
        // 收集最终指标
        let mut final_metrics = HashMap::new();
        if let Some(hv) = hypervolume {
            final_metrics.insert("hypervolume".to_string(), hv);
        }
        
        // 返回结果
        Ok(MultiObjectiveResult {
            pareto_front,
            objective_values,
            hypervolume,
            generational_distance: None, // 这些指标需要更复杂的实现
            inverted_generational_distance: None,
            spread: None,
            runtime_ms,
            all_solutions: Some(all_solutions),
            final_metrics,
            convergence_history: Some(convergence_history),
            iterations: self.config.max_iterations,
            early_stopped,
            algorithm_specific: HashMap::new(),
        })
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
        _solution1_objectives: &[f64],
        _solution2_objectives: &[f64],
    ) -> i32 {
        // MOSA 当前未提供显式支配度比较，这里返回0表示不可比
        0
    }

    fn calculate_metrics(&self, solutions: &[Vec<f64>], objectives: &[Vec<f64>]) -> Result<MultiObjectiveResult> {
        // 计算帕累托前沿
        let pareto_solutions = self.compute_pareto_front();
        let pareto_front: Vec<Vec<f64>> = pareto_solutions.iter().map(|p| p.solution.clone()).collect();
        
        // 计算超体积
        let hypervolume = if !objectives.is_empty() {
            self.calculate_hypervolume(objectives)
        } else {
            None
        };
        
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
                return Err(Error::invalid_argument(format!("Unknown parameter: {}", name)));
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
        use crate::vector::optimizer::multi_objective::types::Pareto;
        
        let mut pareto_front = Vec::new();
        
        for (solution, objectives) in &self.archive {
            pareto_front.push(Pareto {
                solution: solution.clone(),
                objective_values: objectives.clone(),
                rank: 0,
                crowding_distance: 0.0,
            });
        }
        
        pareto_front
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
            return Err(Error::invalid_argument("参数空间为空".to_string()));
        }
        
        // 生成初始样本并评估
        let initial_samples = self.config.algorithm_params.get("initial_samples")
            .map(|&v| v as usize)
            .unwrap_or(10);
        
        let mut eval_results: Vec<OptimizationResult> = Vec::new();
        let mut evaluated_configs = Vec::new();
        
        // 生成并评估初始配置
        for _ in 0..initial_samples {
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
        
        // 更新存档
        for (params, objectives) in &evaluated_configs {
            self.update_archive(&mut self.archive, params.clone(), objectives.clone());
        }
        
        // 主优化循环
        for iter in 0..self.config.max_iterations {
            // 生成邻居解
            let current_solution = if !evaluated_configs.is_empty() {
                let idx = self.rng.gen_range(0..evaluated_configs.len());
                evaluated_configs[idx].0.clone()
            } else {
                self.generate_initial_solution(&bounds)
            };
            
            let neighbor_solution = self.generate_neighbor(&current_solution, &bounds, 
                self.temperature / self.initial_temperature);
            
            // 转换为参数映射并评估
            let mut params = std::collections::HashMap::new();
            for (i, param_name) in param_names.iter().enumerate() {
                if i >= neighbor_solution.len() {
                    break;
                }
                if let Some(param) = parameter_space.parameters().find(|p| p.name() == param_name.as_str()) {
                    match param.parameter_type() {
                        crate::vector::optimizer::parameter_space::ParameterType::Integer(_, _) => {
                            params.insert(param_name.clone(), 
                                crate::vector::optimizer::ParameterValue::Integer(neighbor_solution[i] as i64));
                        },
                        crate::vector::optimizer::parameter_space::ParameterType::Float(_, _) => {
                            params.insert(param_name.clone(), 
                                crate::vector::optimizer::ParameterValue::Float(neighbor_solution[i]));
                        },
                        _ => {},
                    }
                }
            }
            
            let result = self.evaluate_configuration(index_type, &params, parameter_space)?;
            let objectives = vec![
                result.performance.avg_query_time_ms,
                result.performance.index_size_bytes as f64,
            ];
            
            // 计算接受概率
            let current_objectives = if !evaluated_configs.is_empty() {
                let idx = self.rng.gen_range(0..evaluated_configs.len());
                evaluated_configs[idx].1.clone()
            } else {
                objectives.clone()
            };
            
            let acceptance_prob = self.calculate_acceptance_probability(
                &current_objectives, &objectives, self.temperature);
            
            if self.rng.gen::<f64>() < acceptance_prob {
                evaluated_configs.push((neighbor_solution.clone(), objectives.clone()));
                eval_results.push(result);
            }
            
            // 更新存档
            self.update_archive(&mut self.archive, neighbor_solution, objectives);
            
            // 降温
            self.temperature *= self.cooling_rate;
            
            if self.temperature < 1e-10 {
                break;
            }
        }
        
        // 构建帕累托前沿
        let pareto_front = self.compute_pareto_front();
        
        // 选择最佳配置
        let best_idx = self.select_best_configuration(&pareto_front);
        let best_result = if !pareto_front.is_empty() && best_idx < eval_results.len() {
            eval_results[best_idx].clone()
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
                evaluated_configs: evaluated_configs.len(),
                target: OptimizationTarget::BalancedPerformance,
                score: 0.0,
            }
        };
        
        Ok((pareto_front, best_result))
    }
} 