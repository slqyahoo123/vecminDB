use serde::{Serialize, Deserialize};
use std::collections::HashMap;
// remove unused fmt and Result imports

// 使用config模块中的TextFeatureMethod
pub use crate::data::text_features::config::TextFeatureMethod;

/// 特征类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureType {
    /// 文本特征
    Text,
    /// 数值特征
    Numeric,
    /// 类别特征
    Categorical,
    /// 日期时间特征
    DateTime,
    /// 复合特征
    Mixed,
    /// 自定义特征
    Custom,
}

impl Default for FeatureType {
    fn default() -> Self {
        FeatureType::Text
    }
}

/// 字段类型
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FieldType {
    /// 文本类型
    Text,
    /// 数值类型
    Numeric,
    /// 分类类型
    Categorical,
    /// 未知类型
    Unknown,
    /// 混合类型
    Mixed,
    /// 自定义类型
    Custom,
}

impl Default for FieldType {
    fn default() -> Self {
        FieldType::Unknown
    }
}

/// 特征重要性
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    /// 特征名称
    pub name: String,
    /// 重要性分数
    pub importance: f64,
    /// 特征类型
    pub feature_type: FeatureType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextFeatures {
    pub features: HashMap<String, f32>,
}

impl TextFeatures {
    pub fn new() -> Self {
        TextFeatures {
            features: HashMap::new(),
        }
    }
    
    pub fn extract(text: &str) -> crate::Result<Self> {
        // 简单实现，实际项目中应该有更复杂的特征提取
        let mut features = HashMap::new();
        features.insert("length".to_string(), text.len() as f32);
        features.insert("word_count".to_string(), text.split_whitespace().count() as f32);
        
        Ok(TextFeatures { features })
    }
    
    pub fn merge(&mut self, other: &TextFeatures) {
        for (key, value) in &other.features {
            self.features.insert(key.clone(), *value);
        }
    }
}

/// 特征提取结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractionResult {
    /// 提取的特征向量
    pub features: Vec<f32>,
    /// 使用的特征提取方法
    pub method: TextFeatureMethod,
    /// 特征维度
    pub dimension: usize,
    /// 提取时间（毫秒）
    pub extraction_time_ms: u64,
    /// 额外元数据
    pub metadata: HashMap<String, String>,
}

impl FeatureExtractionResult {
    /// 创建新的特征提取结果
    pub fn new(features: Vec<f32>, method: TextFeatureMethod) -> Self {
        Self {
            dimension: features.len(),
            features,
            method,
            extraction_time_ms: 0,
            metadata: HashMap::new(),
        }
    }
    
    /// 添加元数据
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
    
    /// 设置提取时间
    pub fn with_extraction_time(mut self, time_ms: u64) -> Self {
        self.extraction_time_ms = time_ms;
        self
    }
}

/// 文本预处理器类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TextPreprocessorType {
    /// 清洗器
    Cleaner,
    /// 标准化器
    Normalizer,
    /// 分词器
    Tokenizer,
}

/// 文本清洗策略枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CleaningStrategy {
    Basic,
    Advanced,
    Custom,
}

/// 文本规范化策略枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormalizationStrategy {
    Basic,
    Advanced,
    Custom,
}

/// 分词策略枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TokenizationStrategy {
    /// 空格分词
    WhitespaceSplit,
    /// N元组分词
    NGram,
    /// 字符N元组
    CharNGram,
    /// 正则表达式分词
    Regex,
    /// 自定义分词
    Custom,
}

/// 特征融合策略枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureFusionStrategy {
    /// 特征连接
    Concatenation,
    /// 加权平均
    WeightedAverage,
    /// 最大值
    Maximum,
    /// 最小值
    Minimum,
    /// 乘积
    Product,
    /// 特征选择
    FeatureSelection,
    /// 主成分分析
    PCA,
    /// 堆叠
    Stacking,
    /// 自定义融合
    Custom,
}

impl Default for FeatureFusionStrategy {
    fn default() -> Self {
        FeatureFusionStrategy::Concatenation
    }
} 