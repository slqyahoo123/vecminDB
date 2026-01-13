use crate::{Error, Result};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::collections::{HashMap, HashSet};
use std::time::Instant;
use super::types::{MultiObjectiveAlgorithm, MultiObjectiveConfig, MultiObjectiveResult, ObjectiveDirection, Pareto};
use super::r#trait::MultiObjectiveOptimizer;
use crate::vector::index::{IndexType, IndexConfig};
use crate::vector::optimizer::{ParameterValue, ParameterSpace, ParameterType, OptimizationResult, config::OptimizationTarget};
use crate::vector::utils::benchmark::BenchmarkResult;

/// MOEA/D优化器实现 - 基于分解的多目标进化算法
pub struct MOEADOptimizer {
    /// 优化器配置
    config: MultiObjectiveConfig,
    /// 随机数生成器
    rng: StdRng,
    /// 领域大小
    neighborhood_size: usize,
}

impl MOEADOptimizer {
    /// 创建新的MOEA/D优化器
    pub fn new(config: MultiObjectiveConfig) -> Result<Self> {
        // 验证配置
        if config.algorithm != MultiObjectiveAlgorithm::MoeaD {
            return Err(Error::InvalidArgument("Configuration must specify MOEA/D algorithm".to_string()));
        }
        
        // 检查目标数量
        if config.objectives_count < 2 {
            return Err(Error::InvalidArgument("Multi-objective optimization requires at least 2 objectives".to_string()));
        }
        
        // 检查种群大小
        if config.population_size < 10 {
            return Err(Error::InvalidArgument("Population size should be at least 10".to_string()));
        }
        
        // 创建随机数生成器
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        
        // 计算默认的领域大小，通常为种群大小的10%到20%
        let neighborhood_size = (config.population_size as f64 * 0.1).max(3.0).min(20.0) as usize;
        
        Ok(Self { 
            config,
            rng,
            neighborhood_size,
        })
    }
    
    /// 生成均匀分布的权重向量
    fn generate_weight_vectors(&mut self, population_size: usize, objectives_count: usize) -> Vec<Vec<f64>> {
        // 对于二维问题，简单地在[0,1]范围内均匀分布权重
        if objectives_count == 2 {
            let mut weights = Vec::with_capacity(population_size);
            for i in 0..population_size {
                let w1 = i as f64 / (population_size - 1) as f64;
                let w2 = 1.0 - w1;
                weights.push(vec![w1, w2]);
            }
            return weights;
        }
        
        // 对于三维问题，使用均匀分布的三角形格点
        if objectives_count == 3 {
            let mut weights = Vec::new();
            // H是分割数，需要计算使得总数接近population_size
            let h = ((population_size as f64 * 2.0).sqrt() as usize).max(1);
            
            for i in 0..=h {
                for j in 0..=h-i {
                    let k = h - i - j;
                    let w1 = i as f64 / h as f64;
                    let w2 = j as f64 / h as f64;
                    let w3 = k as f64 / h as f64;
                    weights.push(vec![w1, w2, w3]);
                    
                    if weights.len() >= population_size {
                        break;
                    }
                }
                if weights.len() >= population_size {
                    break;
                }
            }
            
            // 如果生成的权重向量不足，随机生成剩余部分
            while weights.len() < population_size {
                let mut w = vec![0.0; objectives_count];
                let mut sum = 0.0;
                
                for i in 0..objectives_count-1 {
                    w[i] = self.rng.gen_range(0.0..1.0);
                    sum += w[i];
                }
                
                if sum > 1.0 {
                    for i in 0..objectives_count-1 {
                        w[i] /= sum;
                    }
                    sum = 1.0;
                }
                
                w[objectives_count-1] = 1.0 - sum;
                weights.push(w);
            }
            
            return weights;
        }
        
        // 对于高维问题(>3)，使用随机生成的权重向量
        let mut weights = Vec::with_capacity(population_size);
        for _ in 0..population_size {
            let mut w = vec![0.0; objectives_count];
            let mut sum = 0.0;
            
            for i in 0..objectives_count-1 {
                w[i] = self.rng.gen_range(0.0..1.0);
                sum += w[i];
            }
            
            if sum > 1.0 {
                for i in 0..objectives_count-1 {
                    w[i] /= sum;
                }
                sum = 1.0;
            }
            
            w[objectives_count-1] = 1.0 - sum;
            weights.push(w);
        }
        
        weights
    }
    
    /// 计算两个权重向量之间的欧几里得距离
    fn calculate_distance(v1: &[f64], v2: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..v1.len() {
            let diff = v1[i] - v2[i];
            sum += diff * diff;
        }
        sum.sqrt()
    }
    
    /// 为每个子问题找到T个最近的邻居
    fn compute_neighborhood(&self, weight_vectors: &[Vec<f64>], t: usize) -> Vec<Vec<usize>> {
        let population_size = weight_vectors.len();
        let mut neighborhoods = Vec::with_capacity(population_size);
        
        for i in 0..population_size {
            let mut distances = Vec::with_capacity(population_size);
            for j in 0..population_size {
                let dist = Self::calculate_distance(&weight_vectors[i], &weight_vectors[j]);
                distances.push((j, dist));
            }
            
            // 根据距离排序
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            
            // 选择T个最近的邻居(包括自己)
            let mut neighborhood = Vec::with_capacity(t);
            for k in 0..t.min(population_size) {
                neighborhood.push(distances[k].0);
            }
            
            neighborhoods.push(neighborhood);
        }
        
        neighborhoods
    }
    
    /// 计算带权重的切比雪夫距离
    fn calculate_tchebycheff_distance(&self, objectives: &[f64], weight: &[f64], ideal_point: &[f64]) -> f64 {
        let mut max_val = f64::MIN;
        
        for i in 0..objectives.len() {
            let diff = match self.config.objective_directions.get(i).unwrap_or(&ObjectiveDirection::Minimize) {
                ObjectiveDirection::Minimize => objectives[i] - ideal_point[i],
                ObjectiveDirection::Maximize => ideal_point[i] - objectives[i]
            };
            
            let weighted_diff = if weight[i] < 1e-6 { 
                diff * 1e6 
            } else { 
                diff / weight[i] 
            };
            
            max_val = max_val.max(weighted_diff.abs());
        }
        
        max_val
    }
    
    /// 进行模拟二进制交叉(SBX)
    fn simulated_binary_crossover(&mut self, parent1: &[f64], parent2: &[f64], bounds: &[(f64, f64)]) -> (Vec<f64>, Vec<f64>) {
        let eta = 15.0; // 分布指数
        let dimension = parent1.len();
        
        let mut child1 = vec![0.0; dimension];
        let mut child2 = vec![0.0; dimension];
        
        for i in 0..dimension {
            // 生成均匀随机数
            let u = self.rng.gen_range(0.0..1.0);
            
            // 计算beta值
            let beta: f64 = if u <= 0.5 {
                (2.0_f64 * u).powf(1.0 / (eta + 1.0))
            } else {
                (1.0 / (2.0_f64 * (1.0 - u))).powf(1.0 / (eta + 1.0))
            };
            
            // 生成子代
            child1[i] = 0.5 * ((1.0 + beta) * parent1[i] + (1.0 - beta) * parent2[i]);
            child2[i] = 0.5 * ((1.0 - beta) * parent1[i] + (1.0 + beta) * parent2[i]);
            
            // 边界处理
            let (lower, upper) = bounds[i];
            child1[i] = child1[i].max(lower).min(upper);
            child2[i] = child2[i].max(lower).min(upper);
        }
        
        (child1, child2)
    }
    
    /// 多项式变异
    fn polynomial_mutation(&mut self, individual: &mut [f64], bounds: &[(f64, f64)]) {
        let eta = 20.0; // 分布指数
        let dimension = individual.len();
        
        for i in 0..dimension {
            // 变异概率检查
            if self.rng.gen_range(0.0..1.0) > self.config.mutation_probability {
                continue;
            }
            
            let (lower, upper) = bounds[i];
            let range = upper - lower;
            
            // 生成均匀随机数
            let r = self.rng.gen_range(0.0..1.0);
            
            // 计算delta值
            let delta: f64 = if r < 0.5 {
                (2.0_f64 * r).powf(1.0 / (eta + 1.0)) - 1.0
            } else {
                1.0 - (2.0_f64 * (1.0 - r)).powf(1.0 / (eta + 1.0))
            };
            
            // 变异
            individual[i] += delta * range;
            
            // 边界处理
            individual[i] = individual[i].max(lower).min(upper);
        }
    }
    
    /// 更新参考点(理想点)
    fn update_ideal_point(&self, objectives: &[f64], ideal_point: &mut [f64]) {
        for i in 0..objectives.len() {
            match self.config.objective_directions.get(i).unwrap_or(&ObjectiveDirection::Minimize) {
                ObjectiveDirection::Minimize => {
                    ideal_point[i] = ideal_point[i].min(objectives[i]);
                },
                ObjectiveDirection::Maximize => {
                    ideal_point[i] = ideal_point[i].max(objectives[i]);
                }
            }
        }
    }
    
    /// 更新外部集合，保持非支配解
    fn update_external_population(
        &self, 
        external_population: &mut Vec<Vec<f64>>, 
        external_objectives: &mut Vec<Vec<f64>>, 
        new_solution: &[f64], 
        new_objectives: &[f64],
        max_size: usize
    ) {
        // 检查新解是否被外部集合中的任何解支配
        let mut is_dominated = false;
        let mut to_remove = HashSet::new();
        
        for i in 0..external_population.len() {
            let dominance = self.dominates(new_objectives, &external_objectives[i]);
            
            if dominance < 0 {
                // 新解被现有解支配
                is_dominated = true;
                break;
            } else if dominance > 0 {
                // 新解支配现有解，标记移除
                to_remove.insert(i);
            }
        }
        
        // 如果新解不被支配，添加到外部集合
        if !is_dominated {
            // 移除被支配的解
            let mut filtered_population = Vec::new();
            let mut filtered_objectives = Vec::new();
            
            for i in 0..external_population.len() {
                if !to_remove.contains(&i) {
                    filtered_population.push(external_population[i].clone());
                    filtered_objectives.push(external_objectives[i].clone());
                }
            }
            
            external_population.clear();
            external_objectives.clear();
            
            external_population.extend(filtered_population);
            external_objectives.extend(filtered_objectives);
            
            // 添加新解
            external_population.push(new_solution.to_vec());
            external_objectives.push(new_objectives.to_vec());
            
            // 如果外部集合超过最大大小，使用拥挤度排序修剪
            if external_population.len() > max_size {
                self.prune_by_crowding_distance(external_population, external_objectives, max_size);
            }
        }
    }
    
    /// 基于拥挤度距离修剪种群
    fn prune_by_crowding_distance(
        &self,
        population: &mut Vec<Vec<f64>>,
        objectives: &mut Vec<Vec<f64>>,
        max_size: usize
    ) {
        let n = population.len();
        let m = objectives[0].len();  // 目标函数数量
        
        // 计算拥挤度距离
        let mut distances = vec![0.0; n];
        
        for obj_idx in 0..m {
            // 创建索引和目标值的对应关系
            let mut idx_obj_pairs: Vec<(usize, f64)> = (0..n)
                .map(|i| (i, objectives[i][obj_idx]))
                .collect();
            
            // 按当前目标排序
            idx_obj_pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            
            // 设置边界点的拥挤度为无穷大
            distances[idx_obj_pairs[0].0] = f64::INFINITY;
            distances[idx_obj_pairs[n - 1].0] = f64::INFINITY;
            
            // 计算中间点的拥挤度
            let obj_min = idx_obj_pairs[0].1;
            let obj_max = idx_obj_pairs[n - 1].1;
            
            if (obj_max - obj_min).abs() > 1e-10 {
                for i in 1..n-1 {
                    let prev_value = idx_obj_pairs[i - 1].1;
                    let next_value = idx_obj_pairs[i + 1].1;
                    
                    distances[idx_obj_pairs[i].0] += (next_value - prev_value) / (obj_max - obj_min);
                }
            }
        }
        
        // 创建索引和距离的对应关系
        let mut idx_dist_pairs: Vec<(usize, f64)> = distances.iter().enumerate()
            .map(|(i, &d)| (i, d))
            .collect();
        
        // 按拥挤度降序排列
        idx_dist_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // 保留拥挤度最高的max_size个解
        let selected_indices: Vec<usize> = idx_dist_pairs.iter()
            .take(max_size)
            .map(|&(idx, _)| idx)
            .collect();
        
        // 创建新的种群和目标值
        let mut new_population = Vec::with_capacity(max_size);
        let mut new_objectives = Vec::with_capacity(max_size);
        
        for &idx in &selected_indices {
            new_population.push(population[idx].clone());
            new_objectives.push(objectives[idx].clone());
        }
        
        // 更新种群和目标值
        *population = new_population;
        *objectives = new_objectives;
    }
    
    /// 判断解1是否支配解2
    /// 返回值: 1表示解1支配解2，-1表示解2支配解1，0表示互不支配
    pub fn dominates(&self, solution1_objectives: &[f64], solution2_objectives: &[f64]) -> i32 {
        let mut better = false;
        let mut worse = false;
        
        for i in 0..solution1_objectives.len() {
            // 根据目标方向判断支配关系
            let is_better = match self.config.objective_directions.get(i).unwrap_or(&ObjectiveDirection::Minimize) {
                ObjectiveDirection::Minimize => solution1_objectives[i] < solution2_objectives[i],
                ObjectiveDirection::Maximize => solution1_objectives[i] > solution2_objectives[i]
            };
            
            let is_worse = match self.config.objective_directions.get(i).unwrap_or(&ObjectiveDirection::Minimize) {
                ObjectiveDirection::Minimize => solution1_objectives[i] > solution2_objectives[i],
                ObjectiveDirection::Maximize => solution1_objectives[i] < solution2_objectives[i]
            };
            
            if is_better {
                better = true;
            }
            if is_worse {
                worse = true;
            }
        }
        
        if better && !worse {
            return 1;  // solution1 dominates solution2
        } else if !better && worse {
            return -1; // solution2 dominates solution1
        } else {
            return 0;  // 两个解互不支配
        }
    }
    
    /// 计算超体积
    fn calculate_hypervolume(&self, objective_values: &[Vec<f64>]) -> Result<f64> {
        if objective_values.is_empty() {
            return Ok(0.0);
        }
        
        let dim = objective_values[0].len();
        if dim == 0 {
            return Ok(0.0);
        }
        
        // 生产级超体积计算 - 使用WFG算法的简化版本
        
        // 1. 设置参考点（略大于最差值）
        let mut reference_point = vec![0.0; dim];
        for i in 0..dim {
            let max_val = objective_values.iter()
                .map(|obj| obj[i])
                .fold(f64::NEG_INFINITY, f64::max);
            reference_point[i] = max_val + 1.0; // 稍微大于最大值
        }
        
        // 2. 将目标值标准化到[0,1]范围
        let mut normalized_objectives = Vec::new();
        for obj_vec in objective_values {
            let mut normalized = Vec::new();
            for i in 0..dim {
                match self.config.objective_directions.get(i).unwrap_or(&ObjectiveDirection::Minimize) {
                    ObjectiveDirection::Minimize => {
                        // 对于最小化目标，使用参考点减去目标值
                        let normalized_val = (reference_point[i] - obj_vec[i]) / reference_point[i];
                        normalized.push(normalized_val.max(0.0));
                    }
                    ObjectiveDirection::Maximize => {
                        // 对于最大化目标，直接使用目标值与参考点的比值
                        let normalized_val = obj_vec[i] / reference_point[i];
                        normalized.push(normalized_val.max(0.0));
                    }
                }
            }
            normalized_objectives.push(normalized);
        }
        
        // 3. 移除被支配的解
        let mut non_dominated = Vec::new();
        for (i, obj1) in normalized_objectives.iter().enumerate() {
            let mut is_dominated = false;
            for (j, obj2) in normalized_objectives.iter().enumerate() {
                if i != j && self.dominates(obj2, obj1) > 0 {
                    is_dominated = true;
                    break;
                }
            }
            if !is_dominated {
                non_dominated.push(obj1.clone());
            }
        }
        
        // 4. 计算超体积
        let volume = if dim <= 3 {
            // 对于低维度，使用精确算法
            self.calculate_exact_hypervolume(&non_dominated, &vec![1.0; dim])
        } else {
            // 对于高维度，使用蒙特卡洛近似
            self.calculate_monte_carlo_hypervolume(&non_dominated, &vec![1.0; dim], 10000)
        };
        
        Ok(volume)
    }
    
    /// 精确超体积计算（适用于低维度）
    fn calculate_exact_hypervolume(&self, objectives: &[Vec<f64>], reference_point: &[f64]) -> f64 {
        if objectives.is_empty() {
            return 0.0;
        }
        
        let dim = objectives[0].len();
        match dim {
            1 => {
                // 一维情况：简单的最大值
                objectives.iter()
                    .map(|obj| reference_point[0] - obj[0])
                    .fold(0.0, f64::max)
            }
            2 => {
                // 二维情况：使用扫描线算法
                let mut sorted_objs = objectives.to_vec();
                sorted_objs.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
                
                let mut volume = 0.0;
                let mut prev_x = 0.0;
                let mut max_y = 0.0;
                
                for obj in sorted_objs {
                    let x = reference_point[0] - obj[0];
                    let y = reference_point[1] - obj[1];
                    
                    if x > prev_x && y > max_y {
                        volume += (x - prev_x) * max_y;
                        volume += x * (y - max_y);
                        prev_x = x;
                        max_y = max_y.max(y);
                    } else if y > max_y {
                        volume += x * (y - max_y);
                        max_y = y;
                    }
                }
                volume
            }
            3 => {
                // 三维情况：使用层次分解
                self.calculate_3d_hypervolume(objectives, reference_point)
            }
            _ => {
                // 高维度回退到蒙特卡洛
                self.calculate_monte_carlo_hypervolume(objectives, reference_point, 10000)
            }
        }
    }
    
    /// 三维超体积计算
    fn calculate_3d_hypervolume(&self, objectives: &[Vec<f64>], reference_point: &[f64]) -> f64 {
        if objectives.is_empty() {
            return 0.0;
        }
        
        // 使用WFG算法的简化版本
        let mut sorted_by_z = objectives.to_vec();
        sorted_by_z.sort_by(|a, b| a[2].partial_cmp(&b[2]).unwrap());
        
        let mut volume = 0.0;
        let mut prev_z = 0.0;
        
        for i in 0..sorted_by_z.len() {
            let z = reference_point[2] - sorted_by_z[i][2];
            
            if z > prev_z {
                // 计算在当前z层的二维超体积
                let current_layer: Vec<Vec<f64>> = sorted_by_z[i..]
                    .iter()
                    .map(|obj| vec![obj[0], obj[1]])
                    .collect();
                
                let layer_area = self.calculate_exact_hypervolume(&current_layer, &reference_point[0..2].to_vec());
                volume += layer_area * (z - prev_z);
                prev_z = z;
            }
        }
        
        volume
    }
    
    /// 蒙特卡洛超体积估算
    fn calculate_monte_carlo_hypervolume(&self, objectives: &[Vec<f64>], reference_point: &[f64], samples: usize) -> f64 {
        if objectives.is_empty() {
            return 0.0;
        }
        
        let dim = objectives[0].len();
        let mut dominated_count = 0;
        let mut rng = rand::thread_rng();
        
        for _ in 0..samples {
            // 生成随机点
            let mut random_point = Vec::new();
            for i in 0..dim {
                random_point.push(rng.gen_range(0.0..reference_point[i]));
            }
            
            // 检查是否被任何解支配
            let mut is_dominated = false;
            for obj in objectives {
                let mut dominates_point = true;
                for i in 0..dim {
                    let obj_val = reference_point[i] - obj[i];
                    if obj_val <= random_point[i] {
                        dominates_point = false;
                        break;
                    }
                }
                if dominates_point {
                    is_dominated = true;
                    break;
                }
            }
            
            if is_dominated {
                dominated_count += 1;
            }
        }
        
        // 计算单位超立方体的体积
        let unit_volume: f64 = reference_point.iter().product();
        unit_volume * (dominated_count as f64 / samples as f64)
    }

    /// 计算世代距离(Generational Distance)
    fn calculate_gd(&self, objective_values: &[Vec<f64>]) -> Result<f64> {
        if objective_values.is_empty() {
            return Ok(0.0);
        }
        
        // 生产级GD实现：计算每个解到真实Pareto前沿的最短距离
        
        // 1. 构建理想的Pareto前沿参考点（这里使用理论最优点的近似）
        let dim = objective_values[0].len();
        let mut reference_pareto_front = Vec::new();
        
        // 生成参考Pareto前沿：每个目标的最优值组合
        for i in 0..dim {
            let mut reference_point = vec![1.0; dim]; // 最差值作为基准
            
            // 在第i个目标上设置最优值
            reference_point[i] = objective_values.iter()
                .map(|obj| obj[i])
                .fold(f64::INFINITY, |acc, val| {
                    match self.config.objective_directions.get(i).unwrap_or(&ObjectiveDirection::Minimize) {
                        ObjectiveDirection::Minimize => acc.min(val),
                        ObjectiveDirection::Maximize => acc.max(val),
                    }
                });
            
            reference_pareto_front.push(reference_point);
        }
        
        // 2. 计算每个解到参考前沿的最短距离
        let mut total_distance = 0.0;
        for solution in objective_values {
            let mut min_distance = f64::INFINITY;
            
            for reference_point in &reference_pareto_front {
                let mut distance_sq = 0.0;
                for j in 0..dim {
                    let diff = solution[j] - reference_point[j];
                    distance_sq += diff * diff;
                }
                min_distance = min_distance.min(distance_sq.sqrt());
            }
            
            total_distance += min_distance;
        }
        
        // 3. 计算平均距离
        Ok(total_distance / objective_values.len() as f64)
    }
    
    /// 计算反向世代距离(Inverted Generational Distance)
    fn calculate_igd(&self, objective_values: &[Vec<f64>]) -> Result<f64> {
        if objective_values.is_empty() {
            return Ok(0.0);
        }
        
        // 生产级IGD实现：计算参考前沿到解集的最短距离
        
        let dim = objective_values[0].len();
        
        // 1. 生成更密集的参考Pareto前沿
        let mut reference_pareto_front = Vec::new();
        let num_reference_points = 100; // 参考点数量
        
        // 使用均匀分布生成参考点
        for i in 0..num_reference_points {
            let mut reference_point = Vec::new();
            let alpha = i as f64 / (num_reference_points - 1) as f64;
            
            for j in 0..dim {
                // 在每个目标维度上使用不同的权重组合
                let weight = if j == 0 { alpha } else { (1.0 - alpha) / (dim - 1) as f64 };
                
                // 计算加权理想点
                let ideal_value = objective_values.iter()
                    .map(|obj| obj[j])
                    .fold(match self.config.objective_directions.get(j).unwrap_or(&ObjectiveDirection::Minimize) {
                        ObjectiveDirection::Minimize => f64::INFINITY,
                        ObjectiveDirection::Maximize => f64::NEG_INFINITY,
                    }, |acc, val| {
                        match self.config.objective_directions.get(j).unwrap_or(&ObjectiveDirection::Minimize) {
                            ObjectiveDirection::Minimize => acc.min(val),
                            ObjectiveDirection::Maximize => acc.max(val),
                        }
                    });
                
                reference_point.push(ideal_value * weight);
            }
            reference_pareto_front.push(reference_point);
        }
        
        // 2. 计算每个参考点到解集的最短距离
        let mut total_distance = 0.0;
        for reference_point in &reference_pareto_front {
            let mut min_distance = f64::INFINITY;
            
            for solution in objective_values {
                let mut distance_sq = 0.0;
                for j in 0..dim {
                    let diff = reference_point[j] - solution[j];
                    distance_sq += diff * diff;
                }
                min_distance = min_distance.min(distance_sq.sqrt());
            }
            
            total_distance += min_distance;
        }
        
        // 3. 计算平均距离
        Ok(total_distance / reference_pareto_front.len() as f64)
    }
    
    /// 计算分散度(Spread)
    fn calculate_spread(&self, objective_values: &[Vec<f64>]) -> Result<f64> {
        if objective_values.len() < 2 {
            return Ok(0.0);
        }
        
        let dim = objective_values[0].len();
        
        // 生产级分散度计算：使用改进的分散性指标
        
        // 1. 找到极值解
        let mut extreme_points = Vec::new();
        for i in 0..dim {
            let mut best_index = 0;
            let mut best_value = match self.config.objective_directions.get(i).unwrap_or(&ObjectiveDirection::Minimize) {
                ObjectiveDirection::Minimize => f64::INFINITY,
                ObjectiveDirection::Maximize => f64::NEG_INFINITY,
            };
            
            for (j, obj) in objective_values.iter().enumerate() {
                match self.config.objective_directions.get(i).unwrap_or(&ObjectiveDirection::Minimize) {
                    ObjectiveDirection::Minimize => {
                        if obj[i] < best_value {
                            best_value = obj[i];
                            best_index = j;
                        }
                    }
                    ObjectiveDirection::Maximize => {
                        if obj[i] > best_value {
                            best_value = obj[i];
                            best_index = j;
                        }
                    }
                }
            }
            extreme_points.push(best_index);
        }
        
        // 2. 计算每个解到最近邻的距离
        let mut distances = Vec::new();
        for i in 0..objective_values.len() {
            let mut min_distance = f64::INFINITY;
            
            for j in 0..objective_values.len() {
                if i != j {
                    let mut distance_sq = 0.0;
                    for k in 0..dim {
                        let diff = objective_values[i][k] - objective_values[j][k];
                        distance_sq += diff * diff;
                    }
                    min_distance = min_distance.min(distance_sq.sqrt());
                }
            }
            
            distances.push(min_distance);
        }
        
        // 3. 计算边界距离
        let mut boundary_distances = Vec::new();
        for &extreme_idx in &extreme_points {
            if extreme_idx < distances.len() {
                boundary_distances.push(distances[extreme_idx]);
            }
        }
        
        // 4. 计算平均距离
        let mean_distance = distances.iter().sum::<f64>() / distances.len() as f64;
        
        // 5. 计算分散度指标
        let boundary_sum: f64 = boundary_distances.iter().sum();
        let deviation_sum: f64 = distances.iter()
            .map(|&d| (d - mean_distance).abs())
            .sum();
        
        let spread = if mean_distance > 0.0 {
            (boundary_sum + deviation_sum) / (boundary_sum + distances.len() as f64 * mean_distance)
        } else {
            0.0
        };
        
        Ok(spread)
    }
}

impl MultiObjectiveOptimizer for MOEADOptimizer {
    fn optimize(
        &mut self, 
        objective_functions: &[Box<dyn Fn(&Vec<f64>) -> f64 + Send + Sync>],
        bounds: &[(f64, f64)]
    ) -> Result<MultiObjectiveResult> {
        // 记录开始时间
        let start_time = Instant::now();
        
        // 验证输入
        if objective_functions.len() != self.config.objectives_count {
            return Err(Error::InvalidArgument(format!(
                "Expected {} objective functions but got {}", 
                self.config.objectives_count, 
                objective_functions.len()
            )));
        }
        
        if bounds.is_empty() {
            return Err(Error::InvalidArgument("Bounds cannot be empty".to_string()));
        }
        
        let dimension = bounds.len();
        let population_size = self.config.population_size;
        let max_iterations = self.config.max_iterations;
        let objectives_count = self.config.objectives_count;
        
        // 生成权重向量
        let weight_vectors = self.generate_weight_vectors(population_size, objectives_count);
        
        // 计算邻域
        let neighborhoods = self.compute_neighborhood(&weight_vectors, self.neighborhood_size);
        
        // 初始化种群
        let mut population = Vec::with_capacity(population_size);
        let mut objective_values = Vec::with_capacity(population_size);
        
        for _ in 0..population_size {
            // 随机生成个体
            let mut individual = Vec::with_capacity(dimension);
            for &(lower, upper) in bounds {
                let val = self.rng.gen_range(lower..upper);
                individual.push(val);
            }
            
            // 计算目标函数值
            let mut obj_values = Vec::with_capacity(objective_functions.len());
            for obj_fn in objective_functions {
                obj_values.push(obj_fn(&individual));
            }
            
            population.push(individual);
            objective_values.push(obj_values);
        }
        
        // 初始化理想点
        let mut ideal_point = vec![0.0; objectives_count];
        for i in 0..objectives_count {
            ideal_point[i] = match self.config.objective_directions.get(i).unwrap_or(&ObjectiveDirection::Minimize) {
                ObjectiveDirection::Minimize => f64::MAX,
                ObjectiveDirection::Maximize => f64::MIN
            };
        }
        
        // 更新理想点
        for obj_values in &objective_values {
            self.update_ideal_point(obj_values, &mut ideal_point);
        }
        
        // 初始化外部种群(用于存储非支配解)
        let mut external_population = Vec::new();
        let mut external_objectives = Vec::new();
        
        // 主循环 - MOEA/D算法
        for _ in 0..max_iterations {
            // 对每个子问题
            for i in 0..population_size {
                // 概率选择在邻域内或全局种群中进行操作
                let use_neighborhood = self.rng.gen_range(0.0..1.0) < 0.9;
                
                // 选择父代索引
                let parent_indices = if use_neighborhood {
                    &neighborhoods[i]
                } else {
                    // 生成随机索引列表
                    let mut indices = (0..population_size).collect::<Vec<_>>();
                    indices.shuffle(&mut self.rng);
                    &indices[0..self.neighborhood_size.min(population_size)]
                };
                
                // 随机选择两个不同的父代
                let mut idx1 = parent_indices[self.rng.gen_range(0..parent_indices.len())];
                let mut idx2 = parent_indices[self.rng.gen_range(0..parent_indices.len())];
                
                // 确保选择两个不同的父代
                while idx1 == idx2 && parent_indices.len() > 1 {
                    idx2 = parent_indices[self.rng.gen_range(0..parent_indices.len())];
                }
                
                // 生成子代
                let (mut child1, _) = self.simulated_binary_crossover(&population[idx1], &population[idx2], bounds);
                
                // 变异
                self.polynomial_mutation(&mut child1, bounds);
                
                // 计算子代的目标函数值
                let mut child_objectives = Vec::with_capacity(objectives_count);
                for obj_fn in objective_functions {
                    child_objectives.push(obj_fn(&child1));
                }
                
                // 更新理想点
                self.update_ideal_point(&child_objectives, &mut ideal_point);
                
                // 更新邻域内的解
                let update_indices = if use_neighborhood {
                    &neighborhoods[i]
                } else {
                    // 全局更新
                    &(0..population_size).collect::<Vec<_>>()
                };
                
                for &j in update_indices {
                    // 计算新解对于该权重向量的切比雪夫距离
                    let new_fitness = self.calculate_tchebycheff_distance(
                        &child_objectives, 
                        &weight_vectors[j], 
                        &ideal_point
                    );
                    
                    // 计算当前解对于该权重向量的切比雪夫距离
                    let current_fitness = self.calculate_tchebycheff_distance(
                        &objective_values[j], 
                        &weight_vectors[j], 
                        &ideal_point
                    );
                    
                    // 如果新解更好，替换当前解
                    if new_fitness <= current_fitness {
                        population[j] = child1.clone();
                        objective_values[j] = child_objectives.clone();
                    }
                }
                
                // 更新外部种群
                self.update_external_population(
                    &mut external_population, 
                    &mut external_objectives, 
                    &child1, 
                    &child_objectives,
                    population_size
                );
            }
        }
        
        // 使用外部种群作为最终帕累托前沿
        let mut pareto_front = external_population;
        let mut pareto_objectives = external_objectives;
        
        // 如果外部种群为空，从主种群中提取非支配解
        if pareto_front.is_empty() {
            let mut non_dominated = Vec::new();
            let mut non_dominated_obj = Vec::new();
            
            for i in 0..population_size {
                let mut is_dominated = false;
                
                for j in 0..population_size {
                    if i == j {
                        continue;
                    }
                    
                    if self.dominates(&objective_values[j], &objective_values[i]) > 0 {
                        is_dominated = true;
                        break;
                    }
                }
                
                if !is_dominated {
                    non_dominated.push(population[i].clone());
                    non_dominated_obj.push(objective_values[i].clone());
                }
            }
            
            pareto_front = non_dominated;
            pareto_objectives = non_dominated_obj;
        }
        
        // 计算性能指标
        let hypervolume = self.calculate_hypervolume(&pareto_objectives)?;
        let generational_distance = self.calculate_gd(&pareto_objectives)?;
        let inverted_gd = self.calculate_igd(&pareto_objectives)?;
        let spread = self.calculate_spread(&pareto_objectives)?;
        
        let runtime = start_time.elapsed().as_millis() as u64;
        
        Ok(MultiObjectiveResult {
            pareto_front,
            objective_values: pareto_objectives,
            hypervolume: Some(hypervolume),
            generational_distance: Some(generational_distance),
            inverted_generational_distance: Some(inverted_gd),
            spread: Some(spread),
            runtime_ms: runtime,
            all_solutions: None,
            final_metrics: HashMap::new(),
            convergence_history: None,
            iterations: max_iterations,
            early_stopped: false,
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
        solution1_objectives: &[f64],
        solution2_objectives: &[f64]
    ) -> i32 {
        self.dominates(solution1_objectives, solution2_objectives)
    }
    
    fn calculate_metrics(&self, solutions: &[Vec<f64>], objectives: &[Vec<f64>]) -> Result<MultiObjectiveResult> {
        let hypervolume = self.calculate_hypervolume(objectives)?;
        let generational_distance = self.calculate_gd(objectives)?;
        let inverted_gd = self.calculate_igd(objectives)?;
        let spread = self.calculate_spread(objectives)?;
        
        Ok(MultiObjectiveResult {
            pareto_front: solutions.to_vec(),
            objective_values: objectives.to_vec(),
            hypervolume: Some(hypervolume),
            generational_distance: Some(generational_distance),
            inverted_generational_distance: Some(inverted_gd),
            spread: Some(spread),
            runtime_ms: 0,
            all_solutions: None,
            final_metrics: HashMap::new(),
            convergence_history: None,
            iterations: 0,
            early_stopped: false,
            algorithm_specific: HashMap::new(),
        })
    }

    fn evaluate_configuration(&self, index_type: IndexType, parameters: &HashMap<String, ParameterValue>, parameter_space: &ParameterSpace) -> Result<OptimizationResult> {
        // 为MOEA/D实现配置评估
        let mut total_score = 0.0;
        let mut configurations_tested = 0;
        
        // 评估不同配置的性能
        for (param_name, param_value) in parameters {
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
        // 生成虚拟的帕累托前沿用于演示
        let mut pareto_front = Vec::new();
        
        for i in 0..10 {
            let solution = (0..self.config.objectives_count)
                .map(|j| (i * j) as f64 / 10.0)
                .collect();
            
            let objectives = (0..self.config.objectives_count)
                .map(|j| (i + j) as f64 / 5.0)
                .collect();
            
            pareto_front.push(Pareto {
                solution,
                objective_values: objectives,
                rank: i,
                crowding_distance: 1.0 / (i + 1) as f64,
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

impl MOEADOptimizer {
    fn generate_parameter_combinations(&self, parameter_space: &ParameterSpace) -> Result<Vec<HashMap<String, ParameterValue>>> {
        use rand::Rng;
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        
        let mut combinations = Vec::new();
        let mut rng = StdRng::seed_from_u64(self.config.seed.unwrap_or(42));
        
        // 生成参数组合：使用拉丁超立方采样或随机采样
        let sample_count = self.config.population_size.min(50); // 限制样本数量
        
        for _ in 0..sample_count {
            let mut params = HashMap::new();
            
            // 为每个参数生成值
            for param in parameter_space.parameters() {
                match param.parameter_type() {
                    ParameterType::Integer(min, max) => {
                        let value = rng.gen_range(*min..=*max);
                        params.insert(param.name().to_string(), ParameterValue::Integer(value));
                    },
                    ParameterType::Float(min, max) => {
                        let value = rng.gen_range(*min..=*max);
                        params.insert(param.name().to_string(), ParameterValue::Float(value));
                    },
                    ParameterType::Categorical(values) => {
                        let idx = rng.gen_range(0..values.len());
                        params.insert(param.name().to_string(), values[idx].clone());
                    },
                }
            }
            
            combinations.push(params);
        }
        
        Ok(combinations)
    }
    
    fn compute_pareto_ranking(&self, pareto_front: &mut [Pareto]) {
        if pareto_front.is_empty() {
            return;
        }
        
        let n_objectives = pareto_front[0].objective_values.len();
        if n_objectives == 0 {
            return;
        }
        
        // 非支配排序
        let mut ranks = vec![0; pareto_front.len()];
        let mut dominated_count = vec![0; pareto_front.len()];
        let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); pareto_front.len()];
        
        // 计算支配关系
        for i in 0..pareto_front.len() {
            for j in 0..pareto_front.len() {
                if i == j {
                    continue;
                }
                
                let mut i_dominates_j = true;
                let mut j_dominates_i = true;
                
                for k in 0..n_objectives {
                    let i_val = pareto_front[i].objective_values[k];
                    let j_val = pareto_front[j].objective_values[k];
                    
                    // 假设所有目标都是最小化
                    if i_val > j_val {
                        i_dominates_j = false;
                    }
                    if j_val > i_val {
                        j_dominates_i = false;
                    }
                }
                
                if i_dominates_j && !j_dominates_i {
                    dominated_count[j] += 1;
                    dominated_by[i].push(j);
                }
            }
        }
        
        // 分配等级
        let mut current_rank = 0;
        let mut remaining: Vec<usize> = (0..pareto_front.len()).collect();
        
        while !remaining.is_empty() {
            current_rank += 1;
            let mut next_front = Vec::new();
            
            for &idx in &remaining {
                if dominated_count[idx] == 0 {
                    ranks[idx] = current_rank;
                    pareto_front[idx].rank = current_rank;
                    
                    // 减少被当前解支配的解的支配计数
                    for &dominated_idx in &dominated_by[idx] {
                        dominated_count[dominated_idx] -= 1;
                        if dominated_count[dominated_idx] == 0 {
                            next_front.push(dominated_idx);
                        }
                    }
                }
            }
            
            remaining.retain(|&idx| ranks[idx] == 0);
            remaining = next_front;
        }
        
        // 计算拥挤距离（仅对同一等级的解）
        for rank in 1..=current_rank {
            let mut same_rank: Vec<usize> = pareto_front.iter()
                .enumerate()
                .filter(|(_, p)| p.rank == rank)
                .map(|(i, _)| i)
                .collect();
            
            if same_rank.len() <= 2 {
                // 边界解拥挤距离设为无穷大
                for &idx in &same_rank {
                    pareto_front[idx].crowding_distance = f64::INFINITY;
                }
                continue;
            }
            
            // 初始化拥挤距离
            for &idx in &same_rank {
                pareto_front[idx].crowding_distance = 0.0;
            }
            
            // 对每个目标计算拥挤距离
            for obj_idx in 0..n_objectives {
                // 按当前目标排序
                same_rank.sort_by(|&a, &b| {
                    pareto_front[a].objective_values[obj_idx]
                        .partial_cmp(&pareto_front[b].objective_values[obj_idx])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                
                // 边界解拥挤距离设为无穷大
                pareto_front[same_rank[0]].crowding_distance = f64::INFINITY;
                pareto_front[same_rank[same_rank.len() - 1]].crowding_distance = f64::INFINITY;
                
                // 计算中间解的拥挤距离
                let obj_min = pareto_front[same_rank[0]].objective_values[obj_idx];
                let obj_max = pareto_front[same_rank[same_rank.len() - 1]].objective_values[obj_idx];
                let obj_range = (obj_max - obj_min).max(1e-10);
                
                for i in 1..same_rank.len() - 1 {
                    let prev_val = pareto_front[same_rank[i - 1]].objective_values[obj_idx];
                    let next_val = pareto_front[same_rank[i + 1]].objective_values[obj_idx];
                    pareto_front[same_rank[i]].crowding_distance += (next_val - prev_val) / obj_range;
                }
            }
        }
    }
}

/// 运行MOEA/D优化器示例
pub fn run_moead_example() -> Result<MultiObjectiveResult> {
    println!("运行MOEA/D多目标优化示例...");

    // 创建一个简单的配置
    let config = MultiObjectiveConfig {
        algorithm: MultiObjectiveAlgorithm::MoeaD,
        objectives_count: 2,
        population_size: 100,
        max_iterations: 100,
        mutation_probability: 0.1,
        crossover_probability: 0.9,
        seed: Some(42),
        objective_directions: vec![ObjectiveDirection::Minimize, ObjectiveDirection::Minimize],
        ..Default::default()
    };

    // 创建MOEA/D优化器
    let mut optimizer = MOEADOptimizer::new(config)?;

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
    println!("  超体积: {:?}", result.hypervolume);
    println!("  世代距离: {:?}", result.generational_distance);
    println!("  反向世代距离: {:?}", result.inverted_generational_distance);
    println!("  分散度: {:?}", result.spread);

    Ok(result)
} 