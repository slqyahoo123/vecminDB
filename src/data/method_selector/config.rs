// Method Selector 配置
// 方法选择器的配置模块

use crate::data::text_features::config::TextFeatureMethod;
use crate::data::method_selector::types::PriorityStrategy;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// 方法选择器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodSelectorConfig {
    /// 自动选择算法
    pub auto_select: bool,
    /// 自适应选择算法
    pub adaptive_selection: bool,
    /// 性能监控
    pub performance_monitoring: bool,
    /// 缓存评估结果
    pub cache_evaluations: bool,
    /// 评估样本大小
    pub evaluation_sample_size: usize,
    /// 性能权重（0-1）
    pub performance_weight: f64,
    /// 质量权重（0-1）
    pub quality_weight: f64,
    /// 内存权重（0-1）
    pub memory_weight: f64,
    /// 维度权重（0-1）
    pub dimension_weight: f64,
    /// 稀疏度权重（0-1）
    pub sparsity_weight: f64,
    /// 最小评估间隔（秒）
    pub min_evaluation_interval_secs: u64,
    /// 自动重新评估阈值
    pub reevaluation_threshold: f64,
    /// 候选方法列表
    pub candidates: Vec<TextFeatureMethod>,
    /// 评估指标
    pub evaluation_metrics: Option<HashMap<String, f64>>,
    /// 是否允许使用外部模型
    pub allow_external_models: bool,
    /// 优先级策略
    pub priority_strategy: PriorityStrategy,
    /// 最小要求分数
    pub min_score_threshold: f64,
    /// 最大允许内存使用（MB）
    pub max_memory_usage: Option<usize>,
    /// 预估执行时间限制（秒）
    pub time_limit: Option<f64>,
}

/// 默认配置实现
impl Default for MethodSelectorConfig {
    fn default() -> Self {
        Self {
            auto_select: true,
            adaptive_selection: true,
            performance_monitoring: true,
            cache_evaluations: true,
            evaluation_sample_size: 1000,
            performance_weight: 0.4,
            quality_weight: 0.3,
            memory_weight: 0.1,
            dimension_weight: 0.1,
            sparsity_weight: 0.1,
            min_evaluation_interval_secs: 3600,
            reevaluation_threshold: 0.1,
            candidates: vec![
                TextFeatureMethod::TfIdf,
                TextFeatureMethod::BagOfWords,
                TextFeatureMethod::Word2Vec,
                TextFeatureMethod::Bert,
                TextFeatureMethod::FastText,
            ],
            evaluation_metrics: None,
            allow_external_models: true,
            priority_strategy: PriorityStrategy::Balanced,
            min_score_threshold: 0.6,
            max_memory_usage: Some(1024),
            time_limit: Some(60.0),
        }
    }
}

impl MethodSelectorConfig {
    /// 创建新的方法选择器配置
    pub fn new() -> Self {
        Self::default()
    }
    
    /// 设置自适应选择
    pub fn with_adaptive_selection(mut self, adaptive: bool) -> Self {
        self.adaptive_selection = adaptive;
        self
    }
    
    /// 设置性能监控
    pub fn with_performance_monitoring(mut self, monitoring: bool) -> Self {
        self.performance_monitoring = monitoring;
        self
    }
    
    /// 设置评估缓存
    pub fn with_cache_evaluations(mut self, cache: bool) -> Self {
        self.cache_evaluations = cache;
        self
    }
    
    /// 设置评估样本大小
    pub fn with_evaluation_sample_size(mut self, size: usize) -> Self {
        self.evaluation_sample_size = size;
        self
    }
    
    /// 设置性能权重
    pub fn with_performance_weight(mut self, weight: f64) -> Self {
        self.performance_weight = weight;
        self
    }
    
    /// 设置质量权重
    pub fn with_quality_weight(mut self, weight: f64) -> Self {
        self.quality_weight = weight;
        self
    }
    
    /// 设置内存权重
    pub fn with_memory_weight(mut self, weight: f64) -> Self {
        self.memory_weight = weight;
        self
    }
    
    /// 设置维度权重
    pub fn with_dimension_weight(mut self, weight: f64) -> Self {
        self.dimension_weight = weight;
        self
    }
    
    /// 设置稀疏性权重
    pub fn with_sparsity_weight(mut self, weight: f64) -> Self {
        self.sparsity_weight = weight;
        self
    }
    
    /// 设置最小评估间隔
    pub fn with_min_evaluation_interval(mut self, interval: u64) -> Self {
        self.min_evaluation_interval_secs = interval;
        self
    }
    
    /// 设置重新评估阈值
    pub fn with_reevaluation_threshold(mut self, threshold: f64) -> Self {
        self.reevaluation_threshold = threshold;
        self
    }

    /// 设置权重
    fn set_weights(&mut self, performance: f64, quality: f64, memory: f64, 
                  dimension: f64, sparsity: f64) -> &mut Self {
        self.performance_weight = performance;
        self.quality_weight = quality;
        self.memory_weight = memory;
        self.dimension_weight = dimension;
        self.sparsity_weight = sparsity;
        self.normalize_weights();
        self
    }
    
    /// 获取性能权重
    pub fn get_performance_weight(&self) -> f64 {
        self.performance_weight
    }

    /// 获取质量权重
    pub fn get_quality_weight(&self) -> f64 {
        self.quality_weight
    }

    /// 获取内存权重
    pub fn get_memory_weight(&self) -> f64 {
        self.memory_weight
    }

    /// 获取维度权重
    pub fn get_dimension_weight(&self) -> f64 {
        self.dimension_weight
    }

    /// 获取稀疏度权重
    pub fn get_sparsity_weight(&self) -> f64 {
        self.sparsity_weight
    }

    /// 获取最小评估间隔
    pub fn get_min_evaluation_interval_secs(&self) -> u64 {
        self.min_evaluation_interval_secs
    }

    /// 获取重新评估阈值
    pub fn get_reevaluation_threshold(&self) -> f64 {
        self.reevaluation_threshold
    }

    /// 规范化权重
    pub fn normalize_weights(&mut self) {
        let sum = self.performance_weight + self.quality_weight + 
                 self.memory_weight + self.dimension_weight + self.sparsity_weight;
        
        if sum > 0.0 {
            self.performance_weight /= sum;
            self.quality_weight /= sum;
            self.memory_weight /= sum;
            self.dimension_weight /= sum;
            self.sparsity_weight /= sum;
        } else {
            // 如果所有权重都为0，则平均分配
            self.performance_weight = 0.2;
            self.quality_weight = 0.2;
            self.memory_weight = 0.2;
            self.dimension_weight = 0.2;
            self.sparsity_weight = 0.2;
        }
    }

    /// 验证配置是否有效
    pub fn validate(&self) -> bool {
        // 验证权重范围
        let weights_valid = self.performance_weight >= 0.0 && self.performance_weight <= 1.0
            && self.quality_weight >= 0.0 && self.quality_weight <= 1.0
            && self.memory_weight >= 0.0 && self.memory_weight <= 1.0
            && self.dimension_weight >= 0.0 && self.dimension_weight <= 1.0
            && self.sparsity_weight >= 0.0 && self.sparsity_weight <= 1.0;
        
        // 验证评估间隔
        let interval_valid = self.min_evaluation_interval_secs > 0;
        
        // 验证重新评估阈值
        let threshold_valid = self.reevaluation_threshold > 0.0 && self.reevaluation_threshold <= 1.0;
        
        weights_valid && interval_valid && threshold_valid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = MethodSelectorConfig::default();
        assert!(config.validate());
    }
    
    #[test]
    fn test_with_options() {
        let config = MethodSelectorConfig::default()
            .with_adaptive_selection(true)
            .with_evaluation_sample_size(200)
            .with_performance_weight(0.7)
            .with_quality_weight(0.2)
            .with_memory_weight(0.1);
        
        assert!(config.validate());
        assert_eq!(config.evaluation_sample_size, 200);
        assert!(config.adaptive_selection);
    }
    
    #[test]
    fn test_normalize_weights() {
        let mut config = MethodSelectorConfig::default()
            .with_performance_weight(5.0)
            .with_quality_weight(3.0)
            .with_memory_weight(2.0);
        
        config.normalize_weights();
        
        // 检查权重是否已归一化
        assert!(config.performance_weight > 0.0 && config.performance_weight < 1.0);
        assert!(config.quality_weight > 0.0 && config.quality_weight < 1.0);
        assert!(config.memory_weight > 0.0 && config.memory_weight < 1.0);
        
        // 检查权重总和是否为1
        let sum = config.performance_weight + config.quality_weight + config.memory_weight + 
                 config.dimension_weight + config.sparsity_weight;
        assert!((sum - 1.0).abs() < 1e-6);
    }
} 