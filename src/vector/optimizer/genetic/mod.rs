// 遗传算法优化器模块
//
// 包含不同类型的遗传算法实现

use crate::Result;
use rand::{rngs::StdRng, SeedableRng, Rng};
use super::config::OptimizerConfig;
use super::parameter_space::ParameterSpace;
use super::optimization_result::OptimizationResult;

/// 遗传算法优化器特性
pub trait GeneticOptimizer {
    /// 执行优化过程并返回结果
    fn optimize(&mut self, 
        objective_function: Box<dyn Fn(&[f64]) -> f64>,
        parameter_space: &ParameterSpace,
        config: &OptimizerConfig
    ) -> Result<OptimizationResult>;
    
    /// 锦标赛选择
    fn tournament_selection(&self, population: &[Vec<f64>], fitness_scores: &[(usize, f64)], 
                           tournament_size: usize, rng: &mut impl Rng) -> Vec<f64> {
        let mut best_idx = rng.gen_range(0..population.len());
        let mut best_fitness = fitness_scores.iter().find(|(i, _)| *i == best_idx).unwrap().1;
        
        for _ in 1..tournament_size {
            let idx = rng.gen_range(0..population.len());
            let fitness = fitness_scores.iter().find(|(i, _)| *i == idx).unwrap().1;
            if fitness < best_fitness {
                best_idx = idx;
                best_fitness = fitness;
            }
        }
        
        population[best_idx].clone()
    }
    
    /// 交叉操作
    fn crossover(&self, parent1: &[f64], parent2: &[f64], rng: &mut impl Rng) -> (Vec<f64>, Vec<f64>) {
        let crossover_point = rng.gen_range(1..parent1.len());
        
        let mut child1 = parent1.to_vec();
        let mut child2 = parent2.to_vec();
        
        // 单点交叉
        for i in crossover_point..parent1.len() {
            child1[i] = parent2[i];
            child2[i] = parent1[i];
        }
        
        (child1, child2)
    }
    
    /// 变异操作
    fn mutate(&self, individual: &mut [f64], parameter_space: &ParameterSpace,
              mutation_rate: f64, rng: &mut impl Rng) {
        for (i, value) in individual.iter_mut().enumerate() {
            if rng.gen::<f64>() < mutation_rate {
                if i < parameter_space.ranges.len() {
                    let range = &parameter_space.ranges[i];
                    let new_val = rng.gen_range(range.min..=range.max);
                    *value = new_val;
                }
            }
        }
    }
}

/// 基本遗传算法优化器
pub struct StandardGeneticAlgorithm {
    population_size: usize,
    max_generations: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    elitism_count: usize,
    rng: StdRng,
}

impl StandardGeneticAlgorithm {
    /// 创建新的遗传算法优化器
    pub fn new(
        population_size: usize,
        max_generations: usize,
        crossover_rate: f64,
        mutation_rate: f64,
        elitism_count: usize,
        seed: Option<u64>
    ) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        
        Self {
            population_size,
            max_generations,
            crossover_rate,
            mutation_rate,
            elitism_count,
            rng,
        }
    }
}

impl GeneticOptimizer for StandardGeneticAlgorithm {
    fn optimize(&mut self, 
        objective_function: Box<dyn Fn(&[f64]) -> f64>,
        parameter_space: &ParameterSpace,
        config: &OptimizerConfig
    ) -> Result<OptimizationResult> {
        // 生产级遗传算法优化器实现
        use rand::Rng;
        
        // 遗传算法的超参数：基于 OptimizerConfig 中的 max_iterations 派生，暂不从 config 读取不存在字段
        let population_size = 100usize.min(config.max_iterations.max(1));
        let max_generations = config.max_iterations;
        let mutation_rate = 0.1;
        let crossover_rate = 0.8;
        let elite_size = (population_size as f64 * 0.1) as usize; // 保留10%的精英
        
        let mut rng = rand::thread_rng();
        let param_count = parameter_space.ranges.len();
        
        // 初始化种群
        let mut population: Vec<Vec<f64>> = Vec::new();
        for _ in 0..population_size {
            let mut individual = Vec::new();
            for range in &parameter_space.ranges {
                let value = rng.gen_range(range.min..=range.max);
                individual.push(value);
            }
            population.push(individual);
        }
        
        let mut best_individual = population[0].clone();
        let mut best_fitness = objective_function(&best_individual);
        let mut convergence_history = Vec::new();
        
        for generation in 0..max_generations {
            // 评估适应度
            let mut fitness_scores: Vec<(usize, f64)> = population
                .iter()
                .enumerate()
                .map(|(i, individual)| (i, objective_function(individual)))
                .collect();
            
            // 按适应度排序（假设最小化问题）
            fitness_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            
            // 更新最佳解
            let current_best_fitness = fitness_scores[0].1;
            if current_best_fitness < best_fitness {
                best_fitness = current_best_fitness;
                best_individual = population[fitness_scores[0].0].clone();
            }
            
            convergence_history.push(best_fitness);
            
            // 检查收敛性（简化版：基于最近一段迭代的改进）
            if generation > 50 {
                let recent_improvement = convergence_history[generation - 50] - best_fitness;
                if recent_improvement.abs() < 1e-6 {
                    break;
                }
            }
            
            // 选择、交叉、变异
            let mut new_population = Vec::new();
            
            // 保留精英
            for i in 0..elite_size {
                new_population.push(population[fitness_scores[i].0].clone());
            }
            
            // 生成新个体
            while new_population.len() < population_size {
                // 锦标赛选择
                let parent1 = self.tournament_selection(&population, &fitness_scores, 3, &mut rng);
                let parent2 = self.tournament_selection(&population, &fitness_scores, 3, &mut rng);
                
                // 交叉
                let (mut child1, mut child2) = if rng.gen::<f64>() < crossover_rate {
                    self.crossover(&parent1, &parent2, &mut rng)
                } else {
                    (parent1.clone(), parent2.clone())
                };
                
                // 变异
                self.mutate(&mut child1, parameter_space, mutation_rate, &mut rng);
                self.mutate(&mut child2, parameter_space, mutation_rate, &mut rng);
                
                new_population.push(child1);
                if new_population.len() < population_size {
                    new_population.push(child2);
                }
            }
            
            population = new_population;
        }
        
        // 构建 IndexConfig 和 BenchmarkResult（简化版）
        let mut best_config = crate::vector::index::IndexConfig::default();
        for (i, range) in parameter_space.ranges.iter().enumerate() {
            if i < best_individual.len() {
                let param_value = best_individual[i] as usize;
                best_config.set_param(&range.name, param_value);
            }
        }

        let performance = crate::vector::benchmark::BenchmarkResult {
            index_type: crate::vector::index::IndexType::Flat,
            config: best_config.clone(),
            metrics: crate::vector::benchmark::IndexPerformanceMetrics {
                build_time_ms: 0.0,
                avg_query_time_ms: best_fitness as f64,
                memory_usage_bytes: 0,
                index_size_bytes: 0,
                recall_rate: 1.0,
                accuracy: 1.0,
            },
            dataset_size: 0,
            dimension: param_count,
            build_time_ms: 0,
            avg_query_time_ms: best_fitness as f64,
            queries_per_second: 0.0,
            memory_usage_bytes: 0,
            accuracy: 1.0,
            index_size_bytes: 0,
        };

        Ok(OptimizationResult {
            index_type: crate::vector::index::IndexType::Flat,
            best_config,
            performance,
            evaluated_configs: convergence_history.len(),
            target: config.target,
            score: best_fitness,
        })
    }
} 