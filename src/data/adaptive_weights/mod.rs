use crate::{Error, Result};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use log::{info, warn};
use crate::core::common_types::PerformanceMetrics;

// 在文件开头添加接口导出
pub mod interface;
pub use interface::{WeightAdjuster, create_weight_adjuster, create_default_weight_adjuster, FeatureImportanceConverter};

/// 自适应权重调整配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveWeightsConfig {
    /// 初始权重值
    pub initial_weights: HashMap<String, f32>,
    /// 最小权重限制
    pub min_weight: f32,
    /// 最大权重限制
    pub max_weight: f32,
    /// 学习率（权重调整速度）
    pub learning_rate: f32,
    /// 调整阈值（只有当性能变化超过该阈值时才调整权重）
    pub adjustment_threshold: f32,
    /// 权重衰减系数
    pub weight_decay: f32,
    /// 是否使用动态学习率
    pub use_dynamic_lr: bool,
    /// 调整策略
    pub strategy: AdaptiveStrategy,
    /// 归一化方法
    pub normalization: NormalizationMethod,
}

/// 自适应调整策略
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdaptiveStrategy {
    /// 性能比例调整（根据性能贡献调整权重）
    PerformanceBased,
    /// 梯度下降
    GradientDescent,
    /// 指数移动平均
    ExponentialMovingAverage,
    /// 自适应矩估计
    AdaptiveMomentEstimation,
    /// 自定义策略
    Custom,
}

/// 归一化方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// 总和为1
    SumToOne,
    /// Min-Max缩放
    MinMax,
    /// Softmax
    Softmax,
    /// 不进行归一化
    None,
}

/// 权重更新状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightUpdateState {
    /// 当前权重
    pub current_weights: HashMap<String, f32>,
    /// 累积梯度
    pub accumulated_gradients: HashMap<String, f32>,
    /// 梯度平方累积（用于AdamW）
    pub accumulated_squared_gradients: HashMap<String, f32>,
    /// 学习率调整系数
    pub learning_rate_factor: f32,
    /// 迭代次数
    pub iterations: usize,
    /// 上次权重
    pub previous_weights: HashMap<String, f32>,
    /// 上次性能指标
    pub previous_performance: HashMap<String, f32>,
}

/// 自适应权重调整器
pub struct AdaptiveWeightAdjuster {
    config: AdaptiveWeightsConfig,
    state: WeightUpdateState,
}

impl Default for AdaptiveWeightsConfig {
    fn default() -> Self {
        Self {
            initial_weights: HashMap::new(),
            min_weight: 0.01,
            max_weight: 1.0,
            learning_rate: 0.01,
            adjustment_threshold: 0.001,
            weight_decay: 0.0001,
            use_dynamic_lr: true,
            strategy: AdaptiveStrategy::PerformanceBased,
            normalization: NormalizationMethod::SumToOne,
        }
    }
}

impl AdaptiveWeightAdjuster {
    /// 创建新的自适应权重调整器
    pub fn new(config: AdaptiveWeightsConfig) -> Result<Self> {
        // 验证配置
        if config.min_weight < 0.0 {
            return Err(Error::invalid_argument("最小权重不能为负".to_string()));
        }
        
        if config.min_weight >= config.max_weight {
            return Err(Error::invalid_argument("最小权重必须小于最大权重".to_string()));
        }
        
        if config.learning_rate <= 0.0 || config.learning_rate > 1.0 {
            return Err(Error::invalid_argument("学习率必须在(0,1]范围内".to_string()));
        }
        
        // 初始化状态
        let state = WeightUpdateState {
            current_weights: config.initial_weights.clone(),
            accumulated_gradients: HashMap::new(),
            accumulated_squared_gradients: HashMap::new(),
            learning_rate_factor: 1.0,
            iterations: 0,
            previous_weights: config.initial_weights.clone(),
            previous_performance: HashMap::new(),
        };
        
        Ok(Self { config, state })
    }
    
    /// 更新权重
    pub fn update_weights(&mut self, performance_metrics: &[PerformanceMetrics]) -> Result<HashMap<String, f32>> {
        // 1. 计算性能分数
        let performance_scores = self.calculate_performance_scores(performance_metrics)?;
        
        // 2. 根据策略调整权重
        match self.config.strategy {
            AdaptiveStrategy::PerformanceBased => {
                self.update_weights_performance_based(&performance_scores)?
            },
            AdaptiveStrategy::GradientDescent => {
                self.update_weights_gradient_descent(&performance_scores)?
            },
            AdaptiveStrategy::ExponentialMovingAverage => {
                self.update_weights_ema(&performance_scores)?
            },
            AdaptiveStrategy::AdaptiveMomentEstimation => {
                self.update_weights_adam(&performance_scores)?
            },
            AdaptiveStrategy::Custom => {
                // 自定义策略实现
                self.update_weights_custom(&performance_scores)?
            },
        }
        
        // 3. 应用权重衰减（仅在衰减系数为正时生效）
        self.apply_weight_decay()?;
        
        // 4. 应用权重限制（裁剪）
        self.clip_weights()?;
        
        // 5. 归一化权重
        self.normalize_weights()?;
        
        // 6. 更新状态
        self.state.iterations += 1;
        self.state.previous_weights = self.state.current_weights.clone();
        for (k, v) in &performance_scores {
            self.state.previous_performance.insert(k.clone(), *v);
        }
        
        // 7. 动态调整学习率
        if self.config.use_dynamic_lr {
            self.adjust_learning_rate()?;
        }
        
        Ok(self.state.current_weights.clone())
    }
    
    /// 获取当前权重
    pub fn get_weights(&self) -> &HashMap<String, f32> {
        &self.state.current_weights
    }
    
    /// 重置权重到初始状态
    pub fn reset(&mut self) {
        self.state.current_weights = self.config.initial_weights.clone();
        self.state.accumulated_gradients.clear();
        self.state.accumulated_squared_gradients.clear();
        self.state.learning_rate_factor = 1.0;
        self.state.iterations = 0;
        self.state.previous_weights = self.config.initial_weights.clone();
        self.state.previous_performance.clear();
        info!("AdaptiveWeightAdjuster has been reset to initial state");
    }
    
    /// 更新配置
    pub fn update_config(&mut self, config: AdaptiveWeightsConfig) -> Result<()> {
        // 验证新配置
        if config.min_weight < 0.0 {
            return Err(Error::invalid_argument("最小权重不能为负".to_string()));
        }
        
        if config.min_weight >= config.max_weight {
            return Err(Error::invalid_argument("最小权重必须小于最大权重".to_string()));
        }
        
        if config.learning_rate <= 0.0 || config.learning_rate > 1.0 {
            return Err(Error::invalid_argument("学习率必须在(0,1]范围内".to_string()));
        }
        
        // 更新配置
        self.config = config;
        
        Ok(())
    }
    
    // 私有辅助方法
    
    /// 计算性能分数
    fn calculate_performance_scores(&self, metrics: &[PerformanceMetrics]) -> Result<HashMap<String, f32>> {
        let mut scores = HashMap::new();
        
        for metric in metrics {
            let component_id = &metric.component_id;
            let mut score = 0.0;
            let mut count = 0;
            
            // 计算综合评分（简单平均）
            if let Some(accuracy) = metric.accuracy {
                score += accuracy;
                count += 1;
            }
            
            if let Some(precision) = metric.precision {
                score += precision;
                count += 1;
            }
            
            if let Some(recall) = metric.recall {
                score += recall;
                count += 1;
            }
            
            if let Some(f1) = metric.f1_score {
                score += f1;
                count += 1;
            }
            
            if let Some(mse) = metric.mse {
                // 均方误差越小越好，所以用1减去归一化后的均方误差
                let normalized_mse = 1.0 - (mse.min(1.0));
                score += normalized_mse;
                count += 1;
            }
            
            // 添加自定义指标
            for (_, value) in &metric.custom_metrics {
                score += value;
                count += 1;
            }
            
            // 计算平均分数
            if count > 0 {
                score /= count as f32;
            }
            
            scores.insert(component_id.clone(), score);
        }
        
        // 检查是否所有组件都有分数
        for component_id in self.state.current_weights.keys() {
            if !scores.contains_key(component_id) {
                // 如果没有找到某个组件的分数，给它一个默认分数（当前权重）
                // 这样可以确保所有组件都有分数，并且保持当前权重不变
                let default_score = self.state.current_weights.get(component_id).unwrap_or(&0.5);
                scores.insert(component_id.clone(), *default_score);
            }
        }
        
        Ok(scores)
    }
    
    /// 基于性能比例更新权重
    fn update_weights_performance_based(&mut self, performance_scores: &HashMap<String, f32>) -> Result<()> {
        // 获取所有组件ID
        let component_ids: Vec<String> = self.state.current_weights.keys()
            .cloned()
            .collect();
            
        // 计算性能总分
        let mut total_score = 0.0;
        for component_id in &component_ids {
            let score = performance_scores.get(component_id).unwrap_or(&0.0);
            total_score += score;
        }
        
        // 如果总分为0，保持权重不变
        if total_score <= 0.0 {
            warn!("AdaptiveWeightAdjuster: total performance score is non-positive; skipping weight update");
            return Ok(());
        }
        
        // 根据性能比例调整权重
        for component_id in &component_ids {
            let current_weight = self.state.current_weights.get(component_id).unwrap_or(&0.5);
            let score = performance_scores.get(component_id).unwrap_or(&0.0);
            
            // 计算权重增量
            let weight_ratio = score / total_score;
            let weight_diff = weight_ratio - *current_weight;
            let weight_delta = weight_diff * self.config.learning_rate * self.state.learning_rate_factor;
            
            // 更新权重
            let new_weight = current_weight + weight_delta;
            self.state.current_weights.insert(component_id.clone(), new_weight);
        }
        
        Ok(())
    }
    
    /// 梯度下降更新权重
    fn update_weights_gradient_descent(&mut self, performance_scores: &HashMap<String, f32>) -> Result<()> {
        // 如果是第一次迭代，没有足够信息计算梯度，直接返回
        if self.state.iterations == 0 {
            for (component_id, score) in performance_scores {
                if !self.state.current_weights.contains_key(component_id) {
                    self.state.current_weights.insert(component_id.clone(), *score);
                }
                // 初始化前一次性能指标
                self.state.previous_performance.insert(component_id.clone(), *score);
            }
            return Ok(());
        }
        
        // 计算每个组件的梯度
        for (component_id, &current_score) in performance_scores {
            let current_weight = *self.state.current_weights.get(component_id).unwrap_or(&0.5);
            let previous_score = *self.state.previous_performance.get(component_id).unwrap_or(&current_score);
            let previous_weight = *self.state.previous_weights.get(component_id).unwrap_or(&current_weight);
            
            // 计算性能变化
            let performance_change = current_score - previous_score;
            
            // 计算权重变化
            let weight_change = current_weight - previous_weight;
            
            // 计算梯度（权重变化对性能的影响）
            let gradient = if weight_change.abs() > 1e-6 {
                performance_change / weight_change
            } else {
                0.0 // 如果权重几乎没变，梯度为0
            };
            
            // 累积梯度
            let accumulated_gradient = self.state.accumulated_gradients
                .entry(component_id.clone())
                .or_insert(0.0);
            *accumulated_gradient = *accumulated_gradient * 0.9 + gradient * 0.1; // 简单平滑处理
            
            // 计算权重调整量
            let adjustment = self.config.learning_rate * self.state.learning_rate_factor * *accumulated_gradient;
            
            // 更新权重
            let new_weight = current_weight + adjustment;
            self.state.current_weights.insert(component_id.clone(), new_weight);
        }
        
        Ok(())
    }
    
    /// 指数移动平均更新权重
    fn update_weights_ema(&mut self, performance_scores: &HashMap<String, f32>) -> Result<()> {
        // EMA衰减系数
        let alpha = self.config.learning_rate;
        
        for (component_id, &score) in performance_scores {
            let current_weight = *self.state.current_weights.get(component_id).unwrap_or(&0.5);
            
            // 计算目标权重
            let target_weight = score;
            
            // EMA更新公式: new = alpha * target + (1 - alpha) * current
            let new_weight = alpha * target_weight + (1.0 - alpha) * current_weight;
            
            // 更新权重
            self.state.current_weights.insert(component_id.clone(), new_weight);
        }
        
        Ok(())
    }
    
    /// Adam优化器更新权重
    fn update_weights_adam(&mut self, performance_scores: &HashMap<String, f32>) -> Result<()> {
        // Adam超参数
        let beta1 = 0.9; // 一阶矩估计的指数衰减率
        let beta2 = 0.999; // 二阶矩估计的指数衰减率
        let epsilon = 1e-8; // 防止除零
        
        // 如果是第一次迭代，初始化状态
        if self.state.iterations == 0 {
            for (component_id, score) in performance_scores {
                if !self.state.current_weights.contains_key(component_id) {
                    self.state.current_weights.insert(component_id.clone(), *score);
                }
                self.state.previous_performance.insert(component_id.clone(), *score);
                self.state.accumulated_gradients.insert(component_id.clone(), 0.0);
                self.state.accumulated_squared_gradients.insert(component_id.clone(), 0.0);
            }
            return Ok(());
        }
        
        // 当前迭代次数
        let t = self.state.iterations + 1;
        
        // 计算每个组件的梯度并更新权重
        for (component_id, &current_score) in performance_scores {
            let current_weight = *self.state.current_weights.get(component_id).unwrap_or(&0.5);
            let previous_score = *self.state.previous_performance.get(component_id).unwrap_or(&current_score);
            let previous_weight = *self.state.previous_weights.get(component_id).unwrap_or(&current_weight);
            
            // 计算性能变化
            let performance_change = current_score - previous_score;
            
            // 计算权重变化
            let weight_change = current_weight - previous_weight;
            
            // 计算梯度
            let gradient = if weight_change.abs() > 1e-6 {
                performance_change / weight_change
            } else {
                0.0
            };
            
            // 更新一阶矩估计
            let m = self.state.accumulated_gradients
                .entry(component_id.clone())
                .or_insert(0.0);
            *m = beta1 * *m + (1.0 - beta1) * gradient;
            
            // 更新二阶矩估计
            let v = self.state.accumulated_squared_gradients
                .entry(component_id.clone())
                .or_insert(0.0);
            *v = beta2 * *v + (1.0 - beta2) * gradient * gradient;
            
            // 计算偏差修正
            let m_hat = *m / (1.0 - beta1.powi(t as i32));
            let v_hat = *v / (1.0 - beta2.powi(t as i32));
            
            // 计算权重调整量
            let adjustment = self.config.learning_rate * m_hat / (v_hat.sqrt() + epsilon);
            
            // 更新权重
            let new_weight = current_weight + adjustment;
            self.state.current_weights.insert(component_id.clone(), new_weight);
        }
        
        Ok(())
    }
    
    /// 自定义策略更新权重
    fn update_weights_custom(&mut self, performance_scores: &HashMap<String, f32>) -> Result<()> {
        // 这里实现自定义策略
        // 默认实现是性能比例和梯度下降的混合
        
        // 1. 计算当前总分
        let mut total_score = 0.0;
        for &score in performance_scores.values() {
            total_score += score;
        }
        
        if total_score <= 0.0 {
            warn!("AdaptiveWeightAdjuster: total performance score is non-positive; skipping weight update");
            return Ok(());
        }
        
        // 2. 计算每个组件的目标权重（基于性能比例）
        let mut target_weights = HashMap::new();
        for (component_id, &score) in performance_scores {
            let target_weight = score / total_score;
            target_weights.insert(component_id.clone(), target_weight);
        }
        
        // 3. 应用梯度下降（朝目标权重逐步移动）
        for (component_id, &target_weight) in &target_weights {
            let current_weight = *self.state.current_weights.get(component_id).unwrap_or(&0.5);
            
            // 计算权重差
            let weight_diff = target_weight - current_weight;
            
            // 应用阻尼系数以平滑变化
            let damping = 0.8;
            let adjustment = weight_diff * self.config.learning_rate * damping;
            
            // 更新权重
            let new_weight = current_weight + adjustment;
            self.state.current_weights.insert(component_id.clone(), new_weight);
        }
        
        Ok(())
    }
    
    /// 应用权重衰减
    fn apply_weight_decay(&mut self) -> Result<()> {
        if self.config.weight_decay <= 0.0 {
            return Ok(());
        }
        
        for weight in self.state.current_weights.values_mut() {
            *weight *= 1.0 - self.config.weight_decay;
        }
        
        Ok(())
    }
    
    /// 裁剪权重到允许范围
    fn clip_weights(&mut self) -> Result<()> {
        for weight in self.state.current_weights.values_mut() {
            *weight = weight.max(self.config.min_weight).min(self.config.max_weight);
        }
        
        Ok(())
    }
    
    /// 归一化权重
    fn normalize_weights(&mut self) -> Result<()> {
        match self.config.normalization {
            NormalizationMethod::SumToOne => {
                // 计算总和
                let mut sum = 0.0;
                for &weight in self.state.current_weights.values() {
                    sum += weight;
                }
                
                if sum > 0.0 {
                    // 归一化
                    for weight in self.state.current_weights.values_mut() {
                        *weight /= sum;
                    }
                }
            },
            NormalizationMethod::MinMax => {
                // 找出最小值和最大值
                let mut min_val = f32::MAX;
                let mut max_val = f32::MIN;
                
                for &weight in self.state.current_weights.values() {
                    min_val = min_val.min(weight);
                    max_val = max_val.max(weight);
                }
                
                let range = max_val - min_val;
                if range > 0.0 {
                    // 应用Min-Max归一化
                    for weight in self.state.current_weights.values_mut() {
                        *weight = ((*weight - min_val) / range) * 
                            (self.config.max_weight - self.config.min_weight) + 
                            self.config.min_weight;
                    }
                }
            },
            NormalizationMethod::Softmax => {
                // 计算指数和
                let mut exp_sum = 0.0;
                let mut exp_values = HashMap::new();
                
                for (component_id, &weight) in &self.state.current_weights {
                    let exp_val = weight.exp();
                    exp_values.insert(component_id.clone(), exp_val);
                    exp_sum += exp_val;
                }
                
                if exp_sum > 0.0 {
                    // 应用Softmax
                    for (component_id, exp_val) in exp_values {
                        let softmax_value = exp_val / exp_sum;
                        self.state.current_weights.insert(component_id, softmax_value);
                    }
                }
            },
            NormalizationMethod::None => {
                // 不进行归一化
            },
        }
        
        Ok(())
    }
    
    /// 动态调整学习率
    fn adjust_learning_rate(&mut self) -> Result<()> {
        // 简单的动态学习率调整策略
        // 如果连续多次迭代权重变化不大，减小学习率
        
        // 检查权重是否有显著变化
        let mut has_significant_change = false;
        for (component_id, &current_weight) in &self.state.current_weights {
            let previous_weight = self.state.previous_weights.get(component_id).unwrap_or(&current_weight);
            let weight_change = (current_weight - *previous_weight).abs();
            
            if weight_change > self.config.adjustment_threshold {
                has_significant_change = true;
                break;
            }
        }
        
        // 如果没有显著变化，减小学习率
        if !has_significant_change && self.state.iterations > 0 {
            self.state.learning_rate_factor *= 0.95; // 逐渐减小
            // 确保学习率不会太小
            self.state.learning_rate_factor = self.state.learning_rate_factor.max(0.01);
        } else {
            // 否则，缓慢恢复学习率
            self.state.learning_rate_factor = (self.state.learning_rate_factor * 1.01).min(1.0);
        }
        
        Ok(())
    }
}

// 在AdaptiveWeightAdjuster实现部分添加WeightAdjuster接口实现
impl interface::WeightAdjuster for AdaptiveWeightAdjuster {
    fn update_weights(&mut self, performance_metrics: &[PerformanceMetrics]) -> Result<HashMap<String, f32>> {
        self.update_weights(performance_metrics)
    }
    
    fn get_weights(&self) -> &HashMap<String, f32> {
        self.get_weights()
    }
    
    fn reset(&mut self) {
        self.reset()
    }
    
    fn update_config(&mut self, config: AdaptiveWeightsConfig) -> Result<()> {
        self.update_config(config)
    }
    
    fn adjust_for_data_characteristics(&mut self, characteristics: &HashMap<String, f64>) -> Result<HashMap<String, f32>> {
        // 基于数据特性调整权重的实现
        let mut adjusted_weights = self.state.current_weights.clone();
        
        // 处理各种特性
        for (characteristic, value) in characteristics {
            match characteristic.as_str() {
                "text_complexity" => {
                    // 文本复杂度越高，文本特征权重越大
                    if let Some(weight) = adjusted_weights.get_mut("text") {
                        let factor = (value.min(1.0).max(0.0) * 0.5 + 0.5) as f32;  // 将0-1映射到0.5-1范围
                        *weight *= factor;
                    }
                },
                "numeric_variance" => {
                    // 数值方差越大，数值特征权重越大
                    if let Some(weight) = adjusted_weights.get_mut("numeric") {
                        let factor = (value.min(1.0).max(0.1)) as f32;
                        *weight *= factor;
                    }
                },
                "categorical_entropy" => {
                    // 类别熵越大，类别特征权重越大
                    if let Some(weight) = adjusted_weights.get_mut("categorical") {
                        let factor = (value.min(1.0).max(0.1)) as f32;
                        *weight *= factor;
                    }
                },
                "data_quality" => {
                    // 数据质量影响所有权重
                    let quality_factor = (value.min(1.0).max(0.5)) as f32;
                    for w in adjusted_weights.values_mut() {
                        *w *= quality_factor;
                    }
                },
                _ => {
                    // 忽略未知特性
                }
            }
        }
        
        // 归一化调整后的权重
        let sum: f32 = adjusted_weights.values().sum();
        if sum > 0.0 {
            for w in adjusted_weights.values_mut() {
                *w /= sum;
            }
        }
        
        Ok(adjusted_weights)
    }
} 