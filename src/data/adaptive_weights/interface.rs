use crate::Result;
use std::collections::HashMap;
use super::{
    AdaptiveWeightsConfig, PerformanceMetrics,
};

/// 权重调整器接口
pub trait WeightAdjuster {
    /// 更新权重
    fn update_weights(&mut self, performance_metrics: &[PerformanceMetrics]) -> Result<HashMap<String, f32>>;
    
    /// 获取当前权重
    fn get_weights(&self) -> &HashMap<String, f32>;
    
    /// 重置权重到初始状态
    fn reset(&mut self);
    
    /// 更新配置
    fn update_config(&mut self, config: AdaptiveWeightsConfig) -> Result<()>;
    
    /// 按数据特性调整权重
    fn adjust_for_data_characteristics(&mut self, characteristics: &HashMap<String, f64>) -> Result<HashMap<String, f32>>;
}

/// 创建新的权重调整器
pub fn create_weight_adjuster(config: AdaptiveWeightsConfig) -> Result<Box<dyn WeightAdjuster>> {
    use super::AdaptiveWeightAdjuster;
    let adjuster = AdaptiveWeightAdjuster::new(config)?;
    Ok(Box::new(adjuster))
}

/// 创建默认权重调整器
pub fn create_default_weight_adjuster() -> Box<dyn WeightAdjuster> {
    use super::AdaptiveWeightAdjuster;
    let config = AdaptiveWeightsConfig::default();
    let adjuster = AdaptiveWeightAdjuster::new(config).unwrap();
    Box::new(adjuster)
}

/// 特征重要性权重转换器
pub struct FeatureImportanceConverter;

impl FeatureImportanceConverter {
    /// 将特征重要性分数转换为权重
    pub fn scores_to_weights(importance_scores: &HashMap<String, f64>) -> HashMap<String, f32> {
        let mut weights = HashMap::new();
        let mut total = 0.0;
        
        // 按重要性分数计算初始权重
        for (feature, score) in importance_scores {
            let weight = *score as f32;
            weights.insert(feature.clone(), weight);
            total += weight;
        }
        
        // 归一化权重
        if total > 0.0 {
            for weight in weights.values_mut() {
                *weight /= total;
            }
        }
        
        weights
    }
    
    /// 应用权重裁剪和调整
    pub fn apply_constraints(
        weights: &mut HashMap<String, f32>,
        min_weight: Option<f32>,
        max_weight: Option<f32>
    ) {
        let min = min_weight.unwrap_or(0.01);
        let max = max_weight.unwrap_or(1.0);
        
        // 应用限制
        for weight in weights.values_mut() {
            if *weight < min {
                *weight = min;
            } else if *weight > max {
                *weight = max;
            }
        }
        
        // 重新归一化
        let total: f32 = weights.values().sum();
        if total > 0.0 {
            for weight in weights.values_mut() {
                *weight /= total;
            }
        }
    }
} 