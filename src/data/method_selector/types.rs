// Types for Method Selector
// 方法选择器的数据类型定义

// FeatureImportance 未在此模块直接使用，保留时应实现使用
use crate::data::text_features::NumericStats;
use crate::data::text_features::config::TextFeatureMethod;
use serde_json::Value;
use std::collections::HashMap;
// Instant 未直接使用，移除
use serde::{Serialize, Deserialize};
use chrono;
// fmt 未直接使用，移除

/// 优先级策略枚举
/// 定义如何权衡不同特性进行方法选择
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PriorityStrategy {
    /// 偏重性能的策略
    Performance,
    /// 偏重准确性的策略
    Accuracy,
    /// 偏重内存效率的策略
    MemoryEfficient,
    /// 平衡性能和准确性的策略
    Balanced,
    /// 用户自定义权重的策略
    Custom,
}

/// 方法评估结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodEvaluation {
    /// 方法名称
    pub method: TextFeatureMethod,
    /// 处理时间（毫秒）
    pub processing_time_ms: u64,
    /// 处理时间（毫秒）- 别名
    pub processing_time: u64,
    /// 内存使用（字节）
    pub memory_usage_bytes: usize,
    /// 内存使用（字节）- 别名
    pub memory_usage: usize,
    /// 特征向量维度
    pub feature_dimension: usize,
    /// 特征向量稀疏度（0-1）
    pub sparsity: f64,
    /// 特征向量质量评分（0-1）
    pub quality_score: f64,
    /// 综合评分（0-1）
    pub overall_score: f64,
}

impl MethodEvaluation {
    /// 创建新的方法评估结果
    pub fn new(method: TextFeatureMethod) -> Self {
        Self {
            method,
            processing_time_ms: 0,
            processing_time: 0,
            memory_usage_bytes: 0,
            memory_usage: 0,
            feature_dimension: 0,
            sparsity: 0.0,
            quality_score: 0.0,
            overall_score: 0.0,
        }
    }
    
    /// 设置处理时间
    pub fn with_processing_time(mut self, time_ms: u64) -> Self {
        self.processing_time_ms = time_ms;
        self.processing_time = time_ms;
        self
    }
    
    /// 设置内存使用
    pub fn with_memory_usage(mut self, memory_bytes: usize) -> Self {
        self.memory_usage_bytes = memory_bytes;
        self.memory_usage = memory_bytes;
        self
    }
}

/// 方法选择器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodSelectorConfig {
    /// 是否启用自适应选择
    pub adaptive_selection: bool,
    /// 是否启用性能监控
    pub performance_monitoring: bool,
    /// 是否监控性能指标
    pub monitor_performance: bool,
    /// 是否缓存评估结果
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
    /// 自动重新评估阈值（数据项数量变化百分比）
    pub reevaluation_threshold: f64,
    /// 默认方法
    pub default_method: Option<TextFeatureMethod>,
    /// 是否应用领域规则
    pub apply_domain_rules: bool,
}

impl Default for MethodSelectorConfig {
    fn default() -> Self {
        Self {
            adaptive_selection: true,
            performance_monitoring: true,
            monitor_performance: true,
            cache_evaluations: true,
            evaluation_sample_size: 1000,
            performance_weight: 0.4,
            quality_weight: 0.3,
            memory_weight: 0.1,
            dimension_weight: 0.1,
            sparsity_weight: 0.1,
            min_evaluation_interval_secs: 3600,
            reevaluation_threshold: 0.1,
            default_method: Some(TextFeatureMethod::TfIdf),
            apply_domain_rules: true,
        }
    }
}

/// 数据特征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCharacteristics {
    /// 数据类型
    pub data_type: String,
    /// 平均文本长度
    pub avg_text_length: f64,
    /// 词汇量大小
    pub vocabulary_size: usize,
    /// 数值特征数量
    pub numeric_feature_count: usize,
    /// 分类特征数量
    pub categorical_feature_count: usize,
    /// 是否包含结构化数据
    pub has_structured_data: bool,
    /// 是否包含非结构化数据
    pub has_unstructured_data: bool,
    /// 语言（如果是文本）
    pub language: Option<String>,
    /// 领域（如果已知）
    pub domain: Option<String>,
    /// 缺失值比例
    pub missing_ratio: f64,
    /// 数据质量得分
    pub quality_score: f64,
    /// 文本字段统计
    pub text_fields: HashMap<String, OtherTextFieldStats>,
    /// 数值字段统计
    pub numeric_fields: HashMap<String, NumericStats>,
    /// 分类字段统计
    pub categorical_fields: HashMap<String, OtherCategoricalFieldStats>,
    /// 样本总数
    pub sample_count: usize,
    /// 字段总数
    pub field_count: usize,
    /// 是否包含序列模式
    pub contains_sequential_patterns: bool,
    /// 是否包含时间相关文本
    pub contains_time_related_text: bool,
    /// 是否包含复杂语义
    pub contains_complex_semantics: bool,
    /// 是否包含歧义表达
    pub contains_ambiguous_meanings: bool,
    /// 文本字段比例
    pub text_field_ratio: f64,
    /// 数值字段比例
    pub numeric_field_ratio: f64,
    /// 类别字段比例
    pub categorical_field_ratio: f64,
    /// 是否包含混合数据类型
    pub has_mixed_data_types: bool,
    /// 文本多样性得分
    pub text_diversity_score: f64,
}

impl Default for DataCharacteristics {
    fn default() -> Self {
        DataCharacteristics {
            text_fields: HashMap::new(),
            numeric_fields: HashMap::new(),
            categorical_fields: HashMap::new(),
            sample_count: 0,
            field_count: 0,
            data_type: String::new(),
            avg_text_length: 0.0,
            vocabulary_size: 0,
            numeric_feature_count: 0,
            categorical_feature_count: 0,
            has_structured_data: false,
            has_unstructured_data: false,
            language: None,
            domain: None,
            missing_ratio: 0.0,
            quality_score: 0.0,
            contains_sequential_patterns: false,
            contains_time_related_text: false,
            contains_complex_semantics: false,
            contains_ambiguous_meanings: false,
            text_field_ratio: 0.0,
            numeric_field_ratio: 0.0,
            categorical_field_ratio: 0.0,
            has_mixed_data_types: false,
            text_diversity_score: 0.0,
        }
    }
}

/// 性能数据点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDataPoint {
    /// 时间戳
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// 方法
    pub method: TextFeatureMethod,
    /// 处理时间（毫秒）
    pub processing_time_ms: u64,
    /// 内存使用（字节）
    pub memory_usage_bytes: usize,
    /// 数据项数量
    pub data_count: usize,
}

/// 文本字段统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtherTextFieldStats {
    /// 平均长度
    pub avg_length: f64,
    /// 平均词数
    pub avg_word_count: f64,
    /// 特殊字符比例
    pub special_char_ratio: f64,
    /// 缺失值比例
    pub missing_ratio: f64,
    /// 词汇频率统计
    pub vocabulary: Option<HashMap<String, usize>>,
    /// 样本数量
    pub sample_count: usize,
    /// 预计算的稀疏度
    pub sparsity_score: f64,
}

/// 分类字段统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtherCategoricalFieldStats {
    /// 基数（不同值的数量）
    pub cardinality: usize,
    /// 最常见值的频率
    pub top_frequency: f64,
    /// 缺失值比例
    pub missing_ratio: f64,
}

/// 领域规则
pub struct DomainRule {
    /// 规则条件
    pub(crate) condition: Box<dyn Fn(&[Value]) -> bool + Send + Sync>,
    /// 推荐方法
    pub(crate) recommended_method: TextFeatureMethod,
    /// 规则优先级 (0-100)
    pub(crate) priority: u8,
}

impl Clone for DomainRule {
    fn clone(&self) -> Self {
        // Domain rules 不能直接克隆，因为它们包含闭包
        // 但我们可以提供一个简化版本，复制基本属性并创建一个始终返回false的条件
        DomainRule {
            condition: Box::new(|_| false),  // 默认条件，始终返回false
            recommended_method: self.recommended_method.clone(),
            priority: self.priority,
        }
    }
} 