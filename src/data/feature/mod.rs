// Feature Module
// 特征模块

// 导出子模块
pub mod types;
pub mod extractor;
pub mod factory;
pub mod adapter;
pub mod factory_impl;
pub mod interface;
pub mod evaluator;
pub mod adapters;
pub mod unified_factory;
pub mod config;
pub mod fusion;
pub mod utils;
pub mod validator;
pub mod validating_extractor;
// 以下模块需要创建
// pub mod pipeline;
// pub mod vector;

// 重导出类型
pub use types::FeatureType;

// 正确重导出结构体和特性，使用别名避免冲突
pub use types::{
    Feature, FeatureVector as TypesFeatureVector, FeatureGroup,
    FeatureExtractor as TypesFeatureExtractor, FeatureTransformer
};

// 重导出工厂
pub use factory::{
    FeatureFactory, FeatureConfig
};

// 重导出提取器
pub use extractor::{
    GenericFeatureExtractor,
    TextFeatureExtractor, ImageFeatureExtractor,
    AudioFeatureExtractor, MultiModalExtractor
};

// 重导出适配器
pub use adapter::{
    FeatureAdapter, GenericAdapter, AdapterFactory
};

// 重导出验证器和验证提取器
pub use validator::{FeatureValidator, BasicFeatureValidator, StatisticalFeatureValidator, ValidatorError};
pub use validating_extractor::{ValidatingFeatureExtractor, create_validating_extractor, ValidationConfig};

// 以下导出可能需要注释掉，因为config模块可能不包含这些类型
// pub use config::{
//     FeatureExtractorConfig, TextConfig, ImageConfig,
//     AudioConfig, MultiModalConfig
// };

// 以下是旧接口，为了向后兼容保留
// 这些接口在后续版本中可能会被废弃
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use crate::Result;
use crate::Error;
// use crate::data::{DataBatch, DataSchema};
// 移除对model模块的直接依赖，使用core类型
// use crate::model::tensor::TensorData;
use crate::core::CoreTensorData;

/// 特征向量表示
#[derive(Clone, Debug)]
pub struct FeatureVector {
    /// 特征向量数据
    pub data: Vec<f32>,
    /// 特征向量维度
    pub dimension: usize,
    /// 特征类型
    pub feature_type: String,
    /// 特征元数据
    pub metadata: HashMap<String, String>,
}

impl FeatureVector {
    /// 创建新的特征向量
    pub fn new(data: Vec<f32>, feature_type: &str) -> Self {
        let dimension = data.len();
        Self {
            data,
            dimension,
            feature_type: feature_type.to_string(),
            metadata: HashMap::new(),
        }
    }
    
    /// 创建空的特征向量
    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            dimension: 0,
            feature_type: "empty".to_string(),
            metadata: HashMap::new(),
        }
    }
    
    /// 获取特征向量数据
    pub fn values(&self) -> &[f32] {
        &self.data
    }
    
    /// 从TensorData创建特征向量
    pub fn from_tensor(tensor: &CoreTensorData, feature_type: &str) -> Result<Self> {
        let data = tensor.data.clone();
        let dimension = data.len();
        
        Ok(Self {
            data,
            dimension,
            feature_type: feature_type.to_string(),
            metadata: HashMap::new(),
        })
    }
    
    /// 添加元数据
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
    
    /// 转换为CoreTensorData
    pub fn to_tensor(&self) -> Result<CoreTensorData> {
        Ok(CoreTensorData::new(vec![self.data.len()], self.data.clone()))
    }
    
    /// 计算与另一个特征向量的余弦相似度
    pub fn cosine_similarity(&self, other: &FeatureVector) -> Result<f32> {
        if self.dimension != other.dimension {
            return Err(Error::invalid_argument(format!(
                "特征向量维度不匹配: {} vs {}", 
                self.dimension, other.dimension
            )));
        }
        
        let mut dot_product = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        
        for i in 0..self.dimension {
            dot_product += self.data[i] * other.data[i];
            norm_a += self.data[i] * self.data[i];
            norm_b += other.data[i] * other.data[i];
        }
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return Ok(0.0);
        }
        
        Ok(dot_product / (norm_a.sqrt() * norm_b.sqrt()))
    }
}

/// 特征提取器特性
/// 所有特征提取器都应实现此特性
pub trait FeatureExtractor: Send + Sync {
    /// 提取特征
    fn extract(&self, data: &[u8], metadata: Option<&HashMap<String, String>>) -> Result<FeatureVector>;
    
    /// 批量提取特征
    fn batch_extract(&self, data_batch: &[Vec<u8>], metadata_batch: Option<&[HashMap<String, String>]>) -> Result<Vec<FeatureVector>> {
        let mut results = Vec::with_capacity(data_batch.len());
        
        for (i, data) in data_batch.iter().enumerate() {
            let metadata = if let Some(meta_batch) = metadata_batch {
                if i < meta_batch.len() {
                    Some(&meta_batch[i])
                } else {
                    None
                }
            } else {
                None
            };
            
            let vector = self.extract(data, metadata)?;
            results.push(vector);
        }
        
        Ok(results)
    }
    
    /// 获取特征维度
    fn dimension(&self) -> usize;
    
    /// 获取提取器类型
    fn extractor_type(&self) -> &str;
    
    /// 获取提取器配置
    fn get_config(&self) -> FeatureExtractorConfig;
    
    /// 从配置创建提取器（默认实现返回错误，子类需要覆盖）
    fn from_config(_config: &FeatureExtractorConfig) -> Result<Self> where Self: Sized {
        Err(Error::not_implemented("from_config 未实现"))
    }
}

/// 特征融合策略
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// 简单连接
    Concatenation,
    /// 加权平均
    WeightedAverage,
    /// 注意力机制
    Attention,
    /// 自定义
    Custom(String),
}

/// 特征融合器配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureFuserConfig {
    /// 融合策略
    pub strategy: FusionStrategy,
    /// 融合权重
    pub weights: Option<HashMap<String, f32>>,
    /// 输出维度
    pub output_dimension: usize,
    /// 其他参数
    pub params: HashMap<String, serde_json::Value>,
}

/// 特征融合器特征
/// 用于将多个特征向量融合为一个
pub trait FeatureFuser: Send + Sync {
    /// 融合特征
    fn fuse(&self, features: &HashMap<String, FeatureVector>) -> Result<FeatureVector>;
    
    /// 获取融合策略
    fn strategy(&self) -> FusionStrategy;
    
    /// 获取输出维度
    fn output_dimension(&self) -> usize;
    
    /// 获取融合器配置
    fn get_config(&self) -> FeatureFuserConfig;
    
    /// 从配置创建融合器（默认实现返回错误，子类需要覆盖）
    fn from_config(_config: &FeatureFuserConfig) -> Result<Self> where Self: Sized {
        Err(Error::not_implemented("from_config 未实现"))
    }
}

/// 特征提取器配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureExtractorConfig {
    /// 提取器类型
    pub extractor_type: String,
    /// 输出特征维度
    pub dimension: usize,
    /// 特征提取器参数
    pub params: HashMap<String, serde_json::Value>,
}

/// 特征提取器工厂
/// 用于创建和管理特征提取器
pub struct FeatureExtractorFactory {
    // 特征提取器注册表
    extractors: HashMap<String, Box<dyn Fn(&FeatureExtractorConfig) -> Result<Box<dyn FeatureExtractor>> + Send + Sync>>,
}

impl FeatureExtractorFactory {
    /// 创建新的工厂实例
    pub fn new() -> Self {
        Self {
            extractors: HashMap::new(),
        }
    }
    
    /// 注册特征提取器
    pub fn register<F>(&mut self, extractor_type: &str, creator: F) -> Result<()>
    where
        F: Fn(&FeatureExtractorConfig) -> Result<Box<dyn FeatureExtractor>> + Send + Sync + 'static
    {
        if self.extractors.contains_key(extractor_type) {
            return Err(Error::invalid_argument(format!("特征提取器类型已注册: {}", extractor_type)));
        }
        
        self.extractors.insert(extractor_type.to_string(), Box::new(creator));
        Ok(())
    }
    
    /// 创建特征提取器
    pub fn create(&self, config: &FeatureExtractorConfig) -> Result<Box<dyn FeatureExtractor>> {
        if let Some(creator) = self.extractors.get(&config.extractor_type) {
            creator(config)
        } else {
            Err(Error::invalid_argument(format!("未知的特征提取器类型: {}", config.extractor_type)))
        }
    }

    /// 注册默认的特征提取器工厂
    pub fn register_default_factories(&mut self) -> &mut Self {
        // 注册文本特征提取器
        let _ = self.register("text", |config| {
            let text_factory = adapters::TextFeatureExtractorFactoryAdapter::new();
            text_factory.create_text_extractor(config)
        });
        
        // 注册音频特征提取器
        let _ = self.register("audio", |config| {
            let audio_factory = adapters::AudioFeatureExtractorFactoryAdapter::new();
            audio_factory.create_audio_extractor(config)
        });
        
        // 可以注册更多默认工厂...
        
        self
    }
}

/// 特征融合工厂
/// 用于创建和管理特征融合器
pub struct FeatureFuserFactory {
    // 特征融合器注册表
    fusers: HashMap<String, Box<dyn Fn(&FeatureFuserConfig) -> Result<Box<dyn FeatureFuser>> + Send + Sync>>,
}

impl FeatureFuserFactory {
    /// 创建新的工厂实例
    pub fn new() -> Self {
        Self {
            fusers: HashMap::new(),
        }
    }
    
    /// 注册特征融合器
    pub fn register<F>(&mut self, strategy_name: &str, creator: F) -> Result<()>
    where
        F: Fn(&FeatureFuserConfig) -> Result<Box<dyn FeatureFuser>> + Send + Sync + 'static
    {
        if self.fusers.contains_key(strategy_name) {
            return Err(Error::invalid_argument(format!("特征融合策略已注册: {}", strategy_name)));
        }
        
        self.fusers.insert(strategy_name.to_string(), Box::new(creator));
        Ok(())
    }
    
    /// 创建特征融合器
    pub fn create(&self, config: &FeatureFuserConfig) -> Result<Box<dyn FeatureFuser>> {
        let strategy_name = match &config.strategy {
            FusionStrategy::Concatenation => "concatenation",
            FusionStrategy::WeightedAverage => "weighted_average",
            FusionStrategy::Attention => "attention",
            FusionStrategy::Custom(name) => name,
        };
        
        if let Some(creator) = self.fusers.get(strategy_name) {
            creator(config)
        } else {
            Err(Error::invalid_argument(format!("未知的特征融合策略: {}", strategy_name)))
        }
    }
}

// 创建空工厂
impl Default for FeatureExtractorFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for FeatureFuserFactory {
    fn default() -> Self {
        Self::new()
    }
}

// 用于测试的模块
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_vector_creation() {
        let vec = FeatureVector::new(vec![1.0, 2.0, 3.0], "test");
        assert_eq!(vec.dimension, 3);
        assert_eq!(vec.data, vec![1.0, 2.0, 3.0]);
    }
    
    #[test]
    fn test_feature_vector_operations() {
        let vec1 = FeatureVector::new(vec![1.0, 2.0, 3.0], "test");
        let vec2 = FeatureVector::new(vec![4.0, 5.0, 6.0], "test");
        
        let similarity = vec1.cosine_similarity(&vec2).unwrap();
        assert!(similarity > 0.97 && similarity < 0.98);
    }
} 