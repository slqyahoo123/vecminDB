// Feature Extractor Module
// 特征提取器模块

// 导出子模块
pub mod tfidf;
pub mod numeric;
pub mod categorical;
pub mod generic;
pub mod advanced;
pub mod model_based;

// 用于直接导入
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::any::Any;
use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use tracing::error;

use crate::data::feature::types::{ExtractorType, FeatureType};
// use crate::Error; // not needed; use fully-qualified crate::Error in conversions

/// 输入数据类型
#[derive(Debug, Clone)]
pub enum InputData {
    /// 原始数据
    Raw(Vec<u8>),
    /// 文本输入
    Text(String),
    /// 文本数组输入
    TextArray(Vec<String>),
    /// 二进制数据
    Binary(Vec<u8>),
    /// 图像数据
    Image(Vec<u8>),
    /// 音频数据
    Audio(Vec<u8>),
    /// 多模态数据
    MultiModal(HashMap<String, Box<InputData>>),
    /// 张量数据
    Tensor(Vec<f32>, Vec<usize>),
}

impl InputData {
    /// 获取输入数据类型的字符串表示
    pub fn type_name(&self) -> &'static str {
        match self {
            InputData::Raw(_) => "raw",
            InputData::Text(_) => "text",
            InputData::TextArray(_) => "text_array",
            InputData::Binary(_) => "binary",
            InputData::Image(_) => "image",
            InputData::Audio(_) => "audio",
            InputData::MultiModal(_) => "multimodal",
            InputData::Tensor(_, _) => "tensor",
        }
    }
    
    /// 转换为字节向量
    pub fn to_vec(&self) -> Vec<u8> {
        match self {
            InputData::Raw(data) => data.clone(),
            InputData::Text(text) => text.as_bytes().to_vec(),
            InputData::TextArray(texts) => {
                let mut result = Vec::new();
                for text in texts {
                    result.extend_from_slice(text.as_bytes());
                    result.push(b'\n'); // 用换行符分隔
                }
                result
            },
            InputData::Binary(data) => data.clone(),
            InputData::Image(data) => data.clone(),
            InputData::Audio(data) => data.clone(),
            InputData::MultiModal(_) => {
                // 对于多模态数据，返回空向量或序列化数据
                Vec::new()
            },
            InputData::Tensor(data, _) => {
                // 将f32转换为字节
                let mut result = Vec::with_capacity(data.len() * 4);
                for value in data {
                    result.extend_from_slice(&value.to_le_bytes());
                }
                result
            },
        }
    }
    
    /// 尝试获取原始数据
    pub fn as_raw(&self) -> Option<&Vec<u8>> {
        match self {
            InputData::Raw(data) => Some(data),
            _ => None,
        }
    }
    
    /// 尝试获取文本数据
    pub fn as_text(&self) -> Option<&String> {
        match self {
            InputData::Text(text) => Some(text),
            _ => None,
        }
    }
    
    /// 尝试获取文本数组数据
    pub fn as_text_array(&self) -> Option<&Vec<String>> {
        match self {
            InputData::TextArray(text_array) => Some(text_array),
            _ => None,
        }
    }
    
    /// 尝试获取二进制数据
    pub fn as_binary(&self) -> Option<&Vec<u8>> {
        match self {
            InputData::Binary(data) => Some(data),
            _ => None,
        }
    }
    
    /// 尝试获取图像数据
    pub fn as_image(&self) -> Option<&Vec<u8>> {
        match self {
            InputData::Image(data) => Some(data),
            _ => None,
        }
    }
    
    /// 尝试获取音频数据
    pub fn as_audio(&self) -> Option<&Vec<u8>> {
        match self {
            InputData::Audio(data) => Some(data),
            _ => None,
        }
    }
    
    /// 尝试获取多模态数据
    pub fn as_multimodal(&self) -> Option<&HashMap<String, Box<InputData>>> {
        match self {
            InputData::MultiModal(data) => Some(data),
            _ => None,
        }
    }
    
    /// 尝试获取张量数据
    pub fn as_tensor(&self) -> Option<(&Vec<f32>, &Vec<usize>)> {
        match self {
            InputData::Tensor(data, shape) => Some((data, shape)),
            _ => None,
        }
    }
}

/// 特征向量
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// 特征值
    pub values: Vec<f32>,
    /// 特征类型
    pub feature_type: FeatureType,
    /// 提取器类型
    pub extractor_type: Option<ExtractorType>,
    /// 元数据
    pub metadata: HashMap<String, String>,
    /// 特征列表（用于复杂特征组合）
    pub features: Vec<crate::data::feature::types::Feature>,
}

impl FeatureVector {
    /// 创建新的特征向量
    pub fn new(feature_type: FeatureType, values: Vec<f32>) -> Self {
        Self {
            values,
            feature_type,
            extractor_type: None,
            metadata: HashMap::new(),
            features: Vec::new(),
        }
    }
    
    /// 创建空的特征向量
    pub fn empty() -> Self {
        Self {
            values: Vec::new(),
            feature_type: FeatureType::Generic,
            extractor_type: None,
            metadata: HashMap::new(),
            features: Vec::new(),
        }
    }
    
    /// 设置提取器类型
    pub fn with_extractor_type(mut self, extractor_type: ExtractorType) -> Self {
        self.extractor_type = Some(extractor_type);
        self
    }
    
    /// 添加元数据
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    
    /// 获取维度
    pub fn dimension(&self) -> usize {
        if !self.features.is_empty() {
            self.features.iter().map(|f| f.data.len()).sum()
        } else {
            self.values.len()
        }
    }
    
    /// 添加特征
    pub fn add_feature(&mut self, feature: crate::data::feature::types::Feature) {
        self.features.push(feature);
        // 更新values以保持一致性
        self.update_values_from_features();
    }
    
    /// 从特征列表更新values
    fn update_values_from_features(&mut self) {
        if !self.features.is_empty() {
            self.values.clear();
            for feature in &self.features {
                self.values.extend_from_slice(&feature.data);
            }
        }
    }
}

/// 特征批次
#[derive(Debug, Clone)]
pub struct FeatureBatch {
    /// 批次大小
    pub batch_size: usize,
    /// 特征值列表
    pub values: Vec<Vec<f32>>,
    /// 特征类型
    pub feature_type: FeatureType,
    /// 提取器类型
    pub extractor_type: Option<ExtractorType>,
    /// 元数据
    pub metadata: HashMap<String, String>,
    /// 特征维度
    pub dimension: usize,
}

impl FeatureBatch {
    /// 创建新的特征批次
    pub fn new(values: Vec<Vec<f32>>, feature_type: FeatureType) -> Self {
        let batch_size = values.len();
        let dimension = if !values.is_empty() && !values[0].is_empty() {
            values[0].len()
        } else {
            0
        };
        
        Self {
            batch_size,
            values,
            feature_type,
            extractor_type: None,
            metadata: HashMap::new(),
            dimension,
        }
    }
    
    /// 设置提取器类型
    pub fn with_extractor_type(mut self, extractor_type: ExtractorType) -> Self {
        self.extractor_type = Some(extractor_type);
        self
    }
    
    /// 添加元数据
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    
    /// 获取特定索引的特征向量
    pub fn get_vector(&self, index: usize) -> Option<FeatureVector> {
        if index >= self.batch_size {
            return None;
        }
        
        Some(FeatureVector {
            values: self.values[index].clone(),
            feature_type: self.feature_type.clone(),
            extractor_type: self.extractor_type.clone(),
            metadata: self.metadata.clone(),
            features: Vec::new(),
        })
    }
    
    /// 转换为特征向量列表
    pub fn to_vectors(&self) -> Vec<FeatureVector> {
        (0..self.batch_size)
            .filter_map(|i| self.get_vector(i))
            .collect()
    }
}

/// 提取器上下文
#[derive(Debug, Clone, Default)]
pub struct ExtractorContext {
    /// 上下文参数
    params: HashMap<String, String>,
}

impl ExtractorContext {
    /// 创建新的提取器上下文
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }
    
    /// 设置参数
    pub fn set_param(&mut self, key: impl Into<String>, value: impl Into<String>) -> std::result::Result<(), ExtractorError> {
        self.params.insert(key.into(), value.into());
        Ok(())
    }
    
    /// 获取参数
    pub fn get_param(&self, key: &str) -> Option<&String> {
        self.params.get(key)
    }
    
    /// 获取参数并解析为指定类型
    pub fn get_param_as<T: std::str::FromStr>(&self, key: &str) -> Option<std::result::Result<T, ()>> {
        self.params.get(key).map(|s| s.parse().map_err(|_| ()))
    }
    
    /// 移除参数
    pub fn remove_param(&mut self, key: &str) -> Option<String> {
        self.params.remove(key)
    }
    
    /// 获取所有参数
    pub fn get_params(&self) -> &HashMap<String, String> {
        &self.params
    }
    
    /// 添加参数并返回self
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }
    
    /// 添加输入数据信息到上下文
    pub fn with_input_data(mut self, input_type: &str) -> Self {
        self.params.insert("input_type".to_string(), input_type.to_string());
        self
    }
    
    /// 获取唯一ID
    pub fn get_id(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        // 将所有参数按key排序后hash
        let mut params_vec: Vec<_> = self.params.iter().collect();
        params_vec.sort_by_key(|(k, _)| *k);
        params_vec.hash(&mut hasher);
        
        format!("ctx_{:016x}", hasher.finish())
    }
    
    /// 添加批次索引
    pub fn with_batch_index(mut self, index: usize) -> Self {
        self.params.insert("batch_index".to_string(), index.to_string());
        self
    }
    
    /// 设置输入数据类型
    pub fn set_input_data_type(&mut self, data_type: &str) -> &mut Self {
        self.params.insert("input_data_type".to_string(), data_type.to_string());
        self
    }
    
    /// 获取输入数据类型
    pub fn get_input_data_type(&self) -> Option<&String> {
        self.params.get("input_data_type")
    }
}

/// 提取器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractorConfig {
    /// 提取器类型
    pub extractor_type: ExtractorType,
    /// 提取器名称
    pub name: String,
    /// 输出维度（可选）
    pub output_dimension: Option<usize>,
    /// 附加参数
    pub params: HashMap<String, String>,
    /// 序列化的字典参数
    pub dict_params: HashMap<String, serde_json::Value>,
}

impl ExtractorConfig {
    /// 创建新的提取器配置
    pub fn new(extractor_type: ExtractorType) -> Self {
        Self {
            extractor_type,
            name: format!("{:?}", extractor_type),
            output_dimension: None,
            params: HashMap::new(),
            dict_params: HashMap::new(),
        }
    }
    
    /// 设置名称
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
    
    /// 设置输出维度
    pub fn with_output_dim(mut self, dim: usize) -> Self {
        self.output_dimension = Some(dim);
        self
    }
    
    /// 添加参数
    pub fn with_param<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }
    
    /// 添加序列化参数
    pub fn with_serialized_param<T: Serialize>(mut self, key: impl Into<String>, value: T) -> std::result::Result<Self, ExtractorError> {
        match serde_json::to_value(value) {
            Ok(json_value) => {
                self.dict_params.insert(key.into(), json_value);
                Ok(self)
            }
            Err(e) => Err(ExtractorError::Config(format!("序列化参数失败: {}", e))),
        }
    }
    
    /// 获取参数
    pub fn get_param<T: std::str::FromStr>(&self, key: &str) -> Option<std::result::Result<T, ExtractorError>> {
        self.params.get(key).map(|value| {
            value.parse::<T>().map_err(|_| {
                ExtractorError::Config(format!("无法将参数 {} 转换为请求的类型", key))
            })
        })
    }
    
    /// 获取序列化参数
    pub fn get_serialized_param<T: for<'a> Deserialize<'a>>(&self, key: &str) -> Option<std::result::Result<T, ExtractorError>> {
        self.dict_params.get(key).map(|value| {
            serde_json::from_value(value.clone()).map_err(|e| {
                ExtractorError::Config(format!("反序列化参数 {} 失败: {}", key, e))
            })
        })
    }
    
    /// 获取字符串参数
    pub fn get_string_param(&self, key: &str) -> Option<String> {
        self.params.get(key).cloned()
    }
    
    /// 获取i32参数
    pub fn get_i32_param(&self, key: &str) -> Option<std::result::Result<i32, ExtractorError>> {
        self.get_param::<i32>(key)
    }
    
    /// 获取f32参数
    pub fn get_f32_param(&self, key: &str) -> Option<std::result::Result<f32, ExtractorError>> {
        self.get_param::<f32>(key)
    }
    
    /// 获取bool参数
    pub fn get_bool_param(&self, key: &str) -> Option<std::result::Result<bool, ExtractorError>> {
        self.params.get(key).map(|s| {
            match s.to_lowercase().as_str() {
                "true" | "1" | "yes" | "y" => Ok(true),
                "false" | "0" | "no" | "n" => Ok(false),
                _ => Err(ExtractorError::Config(format!(
                    "无法将参数值 {} 解析为布尔值", s
                ))),
            }
        })
    }
}

/// 提取器错误类型
#[derive(Debug, thiserror::Error)]
pub enum ExtractorError {
    #[error("提取错误: {0}")]
    Extract(String),
    
    #[error("配置错误: {0}")]
    Config(String),
    
    #[error("验证错误: {0}")]
    Validation(String),
    
    #[error("输入错误: {0}")]
    InvalidInput(String),
    
    #[error("输入数据错误: {0}")]
    InputData(String),
    
    #[error("内部错误: {0}")]
    Internal(String),
    
    #[error("不支持的操作: {0}")]
    Unsupported(String),
    
    #[error("未找到: {0}")]
    NotFound(String),
    
    #[error("序列化错误: {0}")]
    Serialization(String),
    
    #[error("维度不匹配: {0}")]
    DimensionMismatch(String),
    
    #[error("处理错误: {0}")]
    ProcessingError(String),
    
    #[error("其他错误: {0}")]
    Other(String),
}

impl From<crate::Error> for ExtractorError {
    fn from(err: crate::Error) -> Self {
        match err {
            crate::Error::Validation(msg) => ExtractorError::Validation(msg),
            crate::Error::InvalidInput(msg) => ExtractorError::InvalidInput(msg),
            crate::Error::Processing(msg) => ExtractorError::ProcessingError(msg),
            crate::Error::Internal(msg) => ExtractorError::Internal(msg),
            crate::Error::Config(msg) => ExtractorError::Config(msg),
            crate::Error::Serialization(msg) => ExtractorError::Serialization(msg),
            crate::Error::NotFound(msg) => ExtractorError::NotFound(msg),
            _ => ExtractorError::Other(format!("转换错误: {}", err)),
        }
    }
}

impl From<ExtractorError> for crate::Error {
    fn from(err: ExtractorError) -> Self {
        match err {
            ExtractorError::Extract(msg) => crate::Error::processing(msg),
            ExtractorError::Config(msg) => crate::Error::config(msg),
            ExtractorError::Validation(msg) => crate::Error::validation(msg),
            ExtractorError::InvalidInput(msg) => crate::Error::invalid_input(msg),
            ExtractorError::InputData(msg) => crate::Error::invalid_data(msg),
            ExtractorError::Internal(msg) => crate::Error::internal(msg),
            ExtractorError::Unsupported(msg) => crate::Error::not_implemented(msg),
            ExtractorError::NotFound(msg) => crate::Error::not_found(msg),
            ExtractorError::Serialization(msg) => crate::Error::serialization(msg),
            ExtractorError::DimensionMismatch(msg) => crate::Error::invalid_argument(msg),
            ExtractorError::ProcessingError(msg) => crate::Error::processing(msg),
            ExtractorError::Other(msg) => crate::Error::other(msg),
        }
    }
}

/// 特征提取器特性
#[async_trait]
pub trait FeatureExtractor: Send + Sync + Debug {
    /// 获取提取器类型
    fn extractor_type(&self) -> ExtractorType;
    
    /// 获取提取器配置
    fn config(&self) -> &ExtractorConfig;
    
    /// 检查输入数据是否兼容
    fn is_compatible(&self, input: &InputData) -> bool;
    
    /// 提取特征
    async fn extract(&self, input: InputData, context: Option<ExtractorContext>) -> std::result::Result<FeatureVector, ExtractorError>;
    
    /// 批量提取特征
    async fn batch_extract(&self, inputs: Vec<InputData>, context: Option<ExtractorContext>) -> std::result::Result<FeatureBatch, ExtractorError>;
    
    /// 获取输出特征类型
    fn output_feature_type(&self) -> FeatureType;
    
    /// 获取输出维度
    fn output_dimension(&self) -> Option<usize>;
    
    /// 转换为Any类型
    fn as_any(&self) -> &dyn Any;
}

// 重导出TF-IDF提取器相关组件
pub use tfidf::{TfIdfExtractor, TfIdfConfig, TfIdfVocabulary, create_tfidf_extractor};

// 通用特征提取器
pub struct GenericFeatureExtractor;
pub struct TextFeatureExtractor;
pub struct ImageFeatureExtractor;
pub struct AudioFeatureExtractor;
pub struct MultiModalExtractor;

// 虚拟提取器，用于测试
pub struct DummyExtractor {
    config: ExtractorConfig,
    feature_type: FeatureType,
    output_dim: usize,
    test_values: Vec<f32>,
}

impl DummyExtractor {
    pub fn new(config: ExtractorConfig) -> Self {
        Self {
            config,
            feature_type: FeatureType::Generic,
            output_dim: 2,
            test_values: vec![1.0, 2.0],
        }
    }
    
    pub fn with_feature_type(mut self, feature_type: FeatureType) -> Self {
        self.feature_type = feature_type;
        self
    }
    
    pub fn with_output_dim(mut self, dim: usize) -> Self {
        self.output_dim = dim;
        let mut values = Vec::with_capacity(dim);
        for i in 0..dim {
            values.push((i as f32) + 1.0);
        }
        self.test_values = values;
        self
    }
    
    pub fn with_test_values(mut self, values: Vec<f32>) -> Self {
        self.test_values = values;
        self.output_dim = values.len();
        self
    }
}

impl Debug for DummyExtractor {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("DummyExtractor")
            .field("type", &self.config.extractor_type)
            .field("feature_type", &self.feature_type)
            .field("output_dim", &self.output_dim)
            .finish()
    }
}

#[async_trait]
impl FeatureExtractor for DummyExtractor {
    fn extractor_type(&self) -> ExtractorType {
        self.config.extractor_type.clone()
    }
    
    fn config(&self) -> &ExtractorConfig {
        &self.config
    }
    
    fn is_compatible(&self, _input: &InputData) -> bool {
        true // 虚拟提取器兼容所有输入
    }
    
    async fn extract(&self, _input: InputData, _context: Option<ExtractorContext>) -> std::result::Result<FeatureVector, ExtractorError> {
        Ok(FeatureVector::new(self.feature_type.clone(), self.test_values.clone())
            .with_extractor_type(self.config.extractor_type.clone()))
    }
    
    async fn batch_extract(&self, inputs: Vec<InputData>, _context: Option<ExtractorContext>) -> std::result::Result<FeatureBatch, ExtractorError> {
        let batch_size = inputs.len();
        let values = vec![self.test_values.clone(); batch_size];
        
        Ok(FeatureBatch::new(values, self.feature_type.clone())
            .with_extractor_type(self.config.extractor_type.clone()))
    }
    
    fn output_feature_type(&self) -> FeatureType {
        self.feature_type.clone()
    }
    
    fn output_dimension(&self) -> Option<usize> {
        Some(self.output_dim)
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
} 