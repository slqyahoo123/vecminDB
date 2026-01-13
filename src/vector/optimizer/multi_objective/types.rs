// 保留类型模块为纯类型定义文件，不引入未使用的错误类型
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// 多目标优化算法类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MultiObjectiveAlgorithm {
    /// 基于支配排序的遗传算法（非支配排序遗传算法II）
    NSGA2,
    /// 基于分解的多目标进化算法
    MoeaD,
    /// 多目标粒子群优化算法
    MOPSO,
    /// 多目标模拟退火算法
    MOSA,
    /// 多目标贝叶斯优化算法 - 该算法使用高斯过程对多个目标函数进行建模，适用于计算成本高的优化问题
    /// 注意：完整实现计划在v1.2版本
    MOBO,
    /// 多目标梯度下降算法 - 利用各目标函数的梯度信息指导搜索，适用于目标函数可微分的场景
    /// 注意：完整实现计划在v1.3版本
    MultiGradient,
    /// 自定义算法
    Custom,
}

/// 目标类型及优化方向
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectiveDirection {
    /// 最小化目标
    Minimize,
    /// 最大化目标
    Maximize,
}

/// 帕累托优化度量
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParetoMetric {
    /// 超体积指标
    Hypervolume,
    /// 世代距离
    GenerationalDistance,
    /// 反向世代距离
    InverseGenerationalDistance,
    /// 分布均匀度
    Spacing,
    /// 分集度
    Spread,
    /// 基于参考点的覆盖度
    RCoverage,
    /// 自定义指标
    Custom,
}

/// 目标权重策略
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeightStrategy {
    /// 固定权重
    Fixed,
    /// 自适应权重
    Adaptive,
    /// 动态权重
    Dynamic,
    /// 随机权重
    Random,
    /// 基于学习的权重
    Learning,
}

/// 单个优化目标的定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Objective {
    /// 目标名称
    pub name: String,
    /// 目标描述
    pub description: Option<String>,
    /// 目标优化方向
    pub direction: ObjectiveDirection,
    /// 目标权重（在某些算法中使用）
    pub weight: Option<f64>,
    /// 目标下界（用于归一化）
    pub lower_bound: Option<f64>,
    /// 目标上界（用于归一化）
    pub upper_bound: Option<f64>,
    /// 目标相关的约束条件
    pub constraints: Vec<String>,
}

/// 多目标优化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiObjectiveConfig {
    /// 算法类型
    pub algorithm: MultiObjectiveAlgorithm,
    
    /// 目标数量
    pub objectives_count: usize,
    
    /// 目标优化方向
    pub objective_directions: Vec<ObjectiveDirection>,
    
    /// 迭代次数上限
    pub max_iterations: usize,
    
    /// 种群/样本数量
    pub population_size: usize,
    
    /// 变异概率
    pub mutation_probability: f64,
    
    /// 交叉概率
    pub crossover_probability: f64,
    
    /// 随机种子
    pub seed: Option<u64>,
    
    /// MOEA-D特定参数：邻域大小比例
    pub neighborhood_size: Option<f64>,
    
    /// MOPSO特定参数：惯性权重
    pub w: Option<f64>,
    
    /// MOPSO特定参数：个体学习因子
    pub c1: Option<f64>,
    
    /// MOPSO特定参数：社会学习因子
    pub c2: Option<f64>,
    
    /// MOPSO特定参数：存档大小
    pub archive_size: Option<usize>,
    
    /// 优化目标
    pub objectives: Vec<Objective>,
    
    /// 权重策略
    pub weight_strategy: WeightStrategy,
    
    /// 质量度量
    pub metrics: Vec<ParetoMetric>,
    
    /// 收敛阈值
    pub convergence_threshold: Option<f64>,
    
    /// 早停参数
    pub early_stopping: Option<EarlyStoppingConfig>,
    
    /// 归一化方式
    pub normalization: Option<NormalizationMethod>,
    
    /// 算法特定参数
    pub algorithm_params: HashMap<String, f64>,
    
    /// 并行处理设置
    pub parallel_processing: Option<ParallelConfig>,
}

impl Default for MultiObjectiveConfig {
    fn default() -> Self {
        Self {
            algorithm: MultiObjectiveAlgorithm::NSGA2,
            objectives_count: 2,
            objective_directions: vec![ObjectiveDirection::Minimize, ObjectiveDirection::Minimize],
            max_iterations: 100,
            population_size: 50,
            mutation_probability: 0.1,
            crossover_probability: 0.8,
            seed: None,
            neighborhood_size: None,
            w: None,
            c1: None,
            c2: None,
            archive_size: None,
            objectives: Vec::new(),
            weight_strategy: WeightStrategy::Fixed,
            metrics: vec![ParetoMetric::Hypervolume],
            convergence_threshold: None,
            early_stopping: None,
            normalization: None,
            algorithm_params: HashMap::new(),
            parallel_processing: None,
        }
    }
}

/// 提前停止配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// 监控的指标
    pub monitor: String,
    /// 最小改善幅度
    pub min_delta: f64,
    /// 耐心值（连续多少个迭代没有改善才停止）
    pub patience: usize,
    /// 是否启用
    pub enabled: bool,
}

/// 归一化方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// 最小最大标准化
    MinMax,
    /// Z-score标准化
    ZScore,
    /// 不进行归一化
    None,
}

/// 并行处理配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// 是否启用并行
    pub enabled: bool,
    /// 并行度（线程数）
    pub num_workers: usize,
    /// 分块大小
    pub chunk_size: Option<usize>,
}

/// 多目标优化结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiObjectiveResult {
    /// 帕累托最优解集
    pub pareto_front: Vec<Vec<f64>>,
    
    /// 对应的目标函数值
    pub objective_values: Vec<Vec<f64>>,
    
    /// 超体积指标
    pub hypervolume: Option<f64>,
    
    /// 世代距离
    pub generational_distance: Option<f64>,
    
    /// 反向世代距离
    pub inverted_generational_distance: Option<f64>,
    
    /// 分散度
    pub spread: Option<f64>,
    
    /// 运行时间(毫秒)
    pub runtime_ms: u64,
    
    /// 所有解集
    pub all_solutions: Option<Vec<Solution>>,
    
    /// 最终指标
    pub final_metrics: HashMap<String, f64>,
    
    /// 收敛历史
    pub convergence_history: Option<Vec<f64>>,
    
    /// 迭代次数
    pub iterations: usize,
    
    /// 是否提前停止
    pub early_stopped: bool,
    
    /// 算法特定结果
    pub algorithm_specific: HashMap<String, serde_json::Value>,
}

/// 单个解的定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    /// 参数向量
    pub parameters: Vec<f64>,
    /// 目标函数值
    pub objective_values: HashMap<String, f64>,
    /// 是否为帕累托最优
    pub is_pareto_optimal: bool,
    /// 解的排名
    pub rank: Option<usize>,
    /// 拥挤度距离
    pub crowding_distance: Option<f64>,
    /// 解的质量指标
    pub quality_metrics: HashMap<String, f64>,
}

/// 帕累托解
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pareto {
    /// 解向量
    pub solution: Vec<f64>,
    /// 目标函数值
    pub objective_values: Vec<f64>,
    /// 解的排名
    pub rank: usize,
    /// 拥挤度距离
    pub crowding_distance: f64,
}

/// MOBO算法参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MOBOParameters {
    /// 最大迭代次数
    pub max_iterations: usize,
    /// 初始采样数量
    pub initial_samples: usize,
    /// 候选点数量
    pub candidate_count: usize,
    /// 目标权重（查询性能权重, 索引大小权重）
    pub objective_weights: (f64, f64),
    /// 时间限制
    pub time_limit: Option<std::time::Duration>,
}

impl Default for MOBOParameters {
    fn default() -> Self {
        Self {
            max_iterations: 30,
            initial_samples: 10,
            candidate_count: 50,
            objective_weights: (0.7, 0.3), // 默认更重视查询性能
            time_limit: None,
        }
    }
}

/// MOBO算法配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MOBOConfig {
    /// 随机种子
    pub seed: Option<u64>,
    /// 算法参数
    pub parameters: MOBOParameters,
}

/// 多目标梯度下降算法参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiGradientParameters {
    /// 学习率
    pub learning_rate: f64,
    /// 动量系数
    pub momentum: f64,
    /// 最大迭代次数
    pub max_iterations: usize,
    /// 初始种群大小
    pub population_size: usize,
    /// 目标权重
    pub objective_weights: Vec<f64>,
    /// 是否使用自适应学习率
    pub adaptive_learning_rate: bool,
    /// 学习率衰减系数
    pub learning_rate_decay: Option<f64>,
    /// 时间限制
    pub time_limit: Option<std::time::Duration>,
}

impl Default for MultiGradientParameters {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.9,
            max_iterations: 100,
            population_size: 30,
            objective_weights: vec![],
            adaptive_learning_rate: false,
            learning_rate_decay: None,
            time_limit: None,
        }
    }
}

/// 多目标梯度下降算法配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiGradientConfig {
    /// 随机种子
    pub seed: Option<u64>,
    /// 算法参数
    pub parameters: MultiGradientParameters,
} 