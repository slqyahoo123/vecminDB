//! NSGA-II多目标优化算法实现
//!
//! 该模块实现了Non-dominated Sorting Genetic Algorithm II (NSGA-II)，
//! 是一种广泛使用的多目标进化算法。

use rand::Rng;
use serde::{Serialize, Deserialize};
use std::fmt;

/// 个体表示（用于优化）
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Individual {
    /// 向量数据
    pub vector: Vec<f64>,
    /// 适应度值
    pub fitness: f64,
    /// 目标函数值（多目标优化）
    pub objectives: Vec<f64>,
    /// 支配等级
    pub rank: usize,
    /// 拥挤距离
    pub crowding_distance: f64,
}

impl Individual {
    /// 创建新个体
    pub fn new(vector: Vec<f64>) -> Self {
        Self {
            vector,
            fitness: 0.0,
            objectives: Vec::new(),
            rank: 0,
            crowding_distance: 0.0,
        }
    }

    /// 设置目标函数值
    pub fn set_objectives(&mut self, objectives: Vec<f64>) {
        // 先读取适应度所需的值，避免在移动到 self 之后再借用已被移动的对象
        let first_fitness = objectives.get(0).copied().unwrap_or(0.0);
        self.objectives = objectives;
        // 简单地用第一个目标作为适应度（可根据需要修改）
        self.fitness = first_fitness;
    }

    /// 检查是否支配另一个个体（Pareto支配）
    pub fn dominates(&self, other: &Individual) -> bool {
        let mut at_least_one_better = false;
        for (obj1, obj2) in self.objectives.iter().zip(&other.objectives) {
            if obj1 > obj2 {
                return false; // 假设最小化问题
            }
            if obj1 < obj2 {
                at_least_one_better = true;
            }
        }
        at_least_one_better
    }
}

impl fmt::Display for Individual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Individual(fitness: {:.4}, objectives: {:?})", 
               self.fitness, self.objectives)
    }
}

/// NSGA-II优化器
#[derive(Clone)]
pub struct NSGA2Optimizer {
    population_size: usize,
    mutation_rate: f64,
    crossover_rate: f64,
}

impl NSGA2Optimizer {
    /// 创建新的NSGA-II优化器
    pub fn new(population_size: usize, mutation_rate: f64, crossover_rate: f64) -> Self {
        Self {
            population_size,
            mutation_rate,
            crossover_rate,
        }
    }

    /// 运行优化算法
    pub fn optimize(
        &self,
        initial_population: Vec<Individual>,
        max_generations: usize,
        objective_functions: &[Box<dyn Fn(&[f64]) -> f64>],
    ) -> Vec<Individual> {
        let mut population = initial_population;
        let mut rng = rand::thread_rng();

        // 评估初始种群
        for individual in &mut population {
            let objectives: Vec<f64> = objective_functions
                .iter()
                .map(|f| f(&individual.vector))
                .collect();
            individual.set_objectives(objectives);
        }

        for generation in 0..max_generations {
            // 生成子代种群
            let offspring = self.generate_offspring(&population, &mut rng);
            
            // 合并父代和子代
            let mut combined_population = population;
            combined_population.extend(offspring);
            
            // 评估新个体的目标函数
            for individual in &mut combined_population {
                if individual.objectives.is_empty() {
                    let objectives: Vec<f64> = objective_functions
                        .iter()
                        .map(|f| f(&individual.vector))
                        .collect();
                    individual.set_objectives(objectives);
                }
            }

            // NSGA-II环境选择
            population = self.environmental_selection(combined_population);
        }

        population
    }

    /// 生成子代种群
    fn generate_offspring(&self, population: &[Individual], rng: &mut impl Rng) -> Vec<Individual> {
        let mut offspring = Vec::new();

        while offspring.len() < self.population_size {
            // 锦标赛选择
            let parent1 = self.tournament_selection(population, rng);
            let parent2 = self.tournament_selection(population, rng);

            // 交叉
            let (mut child1, mut child2) = if rng.gen::<f64>() < self.crossover_rate {
                self.crossover(&parent1, &parent2, rng)
            } else {
                (parent1.clone(), parent2.clone())
            };

            // 变异
            if rng.gen::<f64>() < self.mutation_rate {
                self.mutate(&mut child1, rng);
            }
                if rng.gen::<f64>() < self.mutation_rate {
                self.mutate(&mut child2, rng);
            }

            offspring.push(child1);
            if offspring.len() < self.population_size {
                offspring.push(child2);
            }
        }

        offspring.truncate(self.population_size);
        offspring
    }

    /// 锦标赛选择
    fn tournament_selection(&self, population: &[Individual], rng: &mut impl Rng) -> Individual {
        let tournament_size = 2;
        let mut best = &population[rng.gen_range(0..population.len())];

        for _ in 1..tournament_size {
            let candidate = &population[rng.gen_range(0..population.len())];
            if self.is_better(candidate, best) {
                best = candidate;
            }
        }

        best.clone()
    }

    /// 判断个体是否更优（基于支配关系和拥挤距离）
    fn is_better(&self, a: &Individual, b: &Individual) -> bool {
        if a.rank < b.rank {
            true
        } else if a.rank > b.rank {
            false
        } else {
            a.crowding_distance > b.crowding_distance
        }
    }

    /// 交叉操作（模拟二进制交叉）
    fn crossover(&self, parent1: &Individual, parent2: &Individual, rng: &mut impl Rng) -> (Individual, Individual) {
        let mut child1_vector = parent1.vector.clone();
        let mut child2_vector = parent2.vector.clone();

        let eta = 2.0; // 分布指数
        
        for i in 0..child1_vector.len() {
            if rng.gen::<f64>() <= 0.5 {
                let y1 = child1_vector[i];
                let y2 = child2_vector[i];
                
                if (y1 - y2).abs() > 1e-14 {
                    let beta = if rng.gen::<f64>() <= 0.5 {
                        let u = rng.gen::<f64>();
                        (2.0 * u).powf(1.0 / (eta + 1.0))
                    } else {
                        let u = rng.gen::<f64>();
                        (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
                    };

                    child1_vector[i] = 0.5 * ((y1 + y2) - beta * (y1 - y2).abs());
                    child2_vector[i] = 0.5 * ((y1 + y2) + beta * (y1 - y2).abs());
                }
            }
        }

        (Individual::new(child1_vector), Individual::new(child2_vector))
    }

    /// 变异操作（多项式变异）
    fn mutate(&self, individual: &mut Individual, rng: &mut impl Rng) {
        let eta = 20.0; // 分布指数
        
        // 预先获取长度，避免在可变借用迭代过程中对 individual.vector 进行不可变借用
        let vector_len = individual.vector.len();
        for gene in &mut individual.vector {
            if rng.gen::<f64>() <= (1.0 / vector_len as f64) {
                let u = rng.gen::<f64>();
                let delta = if u < 0.5 {
                    (2.0 * u).powf(1.0 / (eta + 1.0)) - 1.0
                } else {
                    1.0 - (2.0 * (1.0 - u)).powf(1.0 / (eta + 1.0))
                };
                
                *gene += delta;
                *gene = gene.clamp(0.0, 1.0); // 假设变量范围为[0,1]
            }
        }
    }

    /// 环境选择（NSGA-II核心）
    fn environmental_selection(&self, mut population: Vec<Individual>) -> Vec<Individual> {
        // 快速非支配排序
        let fronts = fast_non_dominated_sort(&mut population);
        
        let mut new_population = Vec::new();
        
        for front_indices in fronts {
            if new_population.len() + front_indices.len() <= self.population_size {
                // 整个前沿都可以加入
                for &i in &front_indices {
                    new_population.push(population[i].clone());
                }
            } else {
                // 需要从当前前沿中选择部分个体
                let remaining = self.population_size - new_population.len();
                if remaining > 0 {
                    // 计算拥挤距离
                    calculate_crowding_distance(&mut population, &front_indices);
                    
                    // 按拥挤距离排序
                    let mut front_sorted = front_indices;
                    front_sorted.sort_by(|&a, &b| {
                        population[b].crowding_distance.partial_cmp(&population[a].crowding_distance)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    
                    // 选择拥挤距离最大的个体
                    for i in 0..remaining {
                        new_population.push(population[front_sorted[i]].clone());
                    }
                }
                break;
            }
        }
        
        new_population
    }
}

/// 快速非支配排序
pub fn fast_non_dominated_sort(population: &mut [Individual]) -> Vec<Vec<usize>> {
    let n = population.len();
    let mut fronts = Vec::new();
    let mut domination_count = vec![0; n];
    let mut dominated_solutions = vec![Vec::new(); n];
    
    // 计算支配关系
    for i in 0..n {
        for j in 0..n {
            if i != j {
                if population[i].dominates(&population[j]) {
                    dominated_solutions[i].push(j);
                } else if population[j].dominates(&population[i]) {
                    domination_count[i] += 1;
                }
            }
        }
    }
    
    // 第一个前沿
    let mut current_front = Vec::new();
    for i in 0..n {
        if domination_count[i] == 0 {
            population[i].rank = 1;
            current_front.push(i);
        }
    }
    
    fronts.push(current_front.clone());
    
    // 构建其他前沿
    let mut front_number = 1;
    while !current_front.is_empty() {
        let mut next_front = Vec::new();
        
        for &i in &current_front {
            for &j in &dominated_solutions[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    population[j].rank = front_number + 1;
                    next_front.push(j);
                }
            }
        }
        
        if !next_front.is_empty() {
            fronts.push(next_front.clone());
            front_number += 1;
        }
        current_front = next_front;
    }
    
    fronts
}

/// 计算拥挤距离
pub fn calculate_crowding_distance(population: &mut [Individual], front: &[usize]) {
    let front_size = front.len();
    
    if front_size <= 2 {
        for &i in front {
            population[i].crowding_distance = f64::INFINITY;
        }
        return;
    }
    
    // 初始化拥挤距离
    for &i in front {
        population[i].crowding_distance = 0.0;
    }
    
    let num_objectives = if !front.is_empty() && !population[front[0]].objectives.is_empty() {
        population[front[0]].objectives.len()
    } else {
        return;
    };
    
    // 对每个目标函数计算拥挤距离
    for obj_index in 0..num_objectives {
        let mut sorted_indices = front.to_vec();
        sorted_indices.sort_by(|&a, &b| {
            population[a].objectives[obj_index]
                .partial_cmp(&population[b].objectives[obj_index])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // 边界个体设为无穷大
        population[sorted_indices[0]].crowding_distance = f64::INFINITY;
        population[sorted_indices[front_size - 1]].crowding_distance = f64::INFINITY;
        
        // 计算目标范围
        let min_obj = population[sorted_indices[0]].objectives[obj_index];
        let max_obj = population[sorted_indices[front_size - 1]].objectives[obj_index];
        
        if (max_obj - min_obj).abs() < f64::EPSILON {
            continue;
        }
        
        // 计算中间个体的拥挤距离
        for i in 1..front_size - 1 {
            let current = sorted_indices[i];
            let prev = sorted_indices[i - 1];
            let next = sorted_indices[i + 1];
            
            population[current].crowding_distance += 
                (population[next].objectives[obj_index] - population[prev].objectives[obj_index]) / (max_obj - min_obj);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_individual_creation() {
        let vector = vec![1.0, 2.0, 3.0];
        let individual = Individual::new(vector.clone());
        assert_eq!(individual.vector, vector);
        assert_eq!(individual.fitness, 0.0);
    }

    #[test]
    fn test_domination() {
        let mut ind1 = Individual::new(vec![1.0, 2.0]);
        let mut ind2 = Individual::new(vec![2.0, 3.0]);
        
        ind1.set_objectives(vec![1.0, 2.0]);
        ind2.set_objectives(vec![2.0, 3.0]);
        
        assert!(ind1.dominates(&ind2));
        assert!(!ind2.dominates(&ind1));
    }
} 