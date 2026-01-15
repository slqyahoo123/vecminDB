// 特征提取器统一接口
// 提供所有特征提取器的统一抽象接口

use std::collections::HashMap;
use std::fmt::Debug;
use std::time::Duration;

use async_trait::async_trait;

use crate::Result;
use crate::Error;
use crate::data::feature::types::{ExtractorType, FeatureType};

/// 特征提取结果
#[derive(Debug, Clone)]
pub struct FeatureExtractionResult {
    /// 特征向量
    pub feature_vector: Vec<f32>,
    /// 特征维度
    pub dimension: usize,
    /// 特征类型
    pub feature_type: FeatureType,
    /// 提取时间
    pub extraction_time: Option<Duration>,
    /// 元数据
    pub metadata: HashMap<String, String>,
    /// 提取器类型
    pub extractor_type: ExtractorType,
}

impl FeatureExtractionResult {
    /// 创建新的特征提取结果
    pub fn new(feature_vector: Vec<f32>, feature_type: FeatureType, extractor_type: ExtractorType) -> Self {
        let dimension = feature_vector.len();
        Self {
            feature_vector,
            dimension,
            feature_type,
            extraction_time: None,
            metadata: HashMap::new(),
            extractor_type,
        }
    }
    
    /// 设置提取时间
    pub fn with_extraction_time(mut self, time: Duration) -> Self {
        self.extraction_time = Some(time);
        self
    }
    
    /// 添加元数据
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
    
    /// 添加多个元数据
    pub fn with_metadata_map(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata.extend(metadata);
        self
    }
    
    /// 转换为特征向量
    pub fn to_feature_vector(&self) -> FeatureVector {
        FeatureVector::new(self.feature_vector.clone())
            .with_metadata(self.metadata.clone())
    }
    
    /// 判断结果是否为空
    pub fn is_empty(&self) -> bool {
        self.feature_vector.is_empty()
    }
    
    /// 获取特征长度
    pub fn len(&self) -> usize {
        self.dimension
    }
}

/// 特征提取配置
#[derive(Debug, Clone)]
pub struct ExtractorConfig {
    /// 提取器类型
    pub extractor_type: ExtractorType,
    /// 特征类型
    pub feature_type: FeatureType,
    /// 目标维度
    pub target_dimension: Option<usize>,
    /// 是否启用缓存
    pub enable_cache: bool,
    /// 超时设置（毫秒）
    pub timeout_ms: Option<u64>,
    /// 批处理大小
    pub batch_size: Option<usize>,
    /// 其他配置选项
    pub options: HashMap<String, String>,
}

impl ExtractorConfig {
    /// 创建新的提取器配置
    pub fn new(extractor_type: ExtractorType, feature_type: FeatureType) -> Self {
        Self {
            extractor_type,
            feature_type,
            target_dimension: None,
            enable_cache: true,
            timeout_ms: None,
            batch_size: None,
            options: HashMap::new(),
        }
    }
    
    /// 设置目标维度
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.target_dimension = Some(dimension);
        self
    }
    
    /// 设置缓存启用状态
    pub fn with_cache_enabled(mut self, enable: bool) -> Self {
        self.enable_cache = enable;
        self
    }
    
    /// 设置超时
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }
    
    /// 设置批处理大小
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }
    
    /// 添加配置选项
    pub fn with_option(mut self, key: &str, value: &str) -> Self {
        self.options.insert(key.to_string(), value.to_string());
        self
    }
    
    /// 添加多个配置选项
    pub fn with_options(mut self, options: HashMap<String, String>) -> Self {
        self.options.extend(options);
        self
    }
    
    /// 获取选项值
    pub fn get_option(&self, key: &str) -> Option<&String> {
        self.options.get(key)
    }
    
    /// 获取选项值，如果不存在则返回默认值
    pub fn get_option_or(&self, key: &str, default: &str) -> String {
        self.options.get(key).map_or_else(|| default.to_string(), |s| s.clone())
    }
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        use crate::data::feature::types::TextExtractorType;
        Self::new(ExtractorType::Text(TextExtractorType::TfIdf), FeatureType::Text)
    }
}

/// 特征提取器
#[async_trait]
pub trait FeatureExtractor: Send + Sync + Debug {
    /// 获取提取器类型
    fn extractor_type(&self) -> ExtractorType;
    
    /// 获取特征类型
    fn feature_type(&self) -> FeatureType;
    
    /// 获取提取器名称
    fn name(&self) -> &str;
    
    /// 获取提取器配置
    fn config(&self) -> &ExtractorConfig;
    
    /// 获取提取器配置（可变）
    fn get_config(&self) -> ExtractorConfig;
    
    /// 提取特征
    async fn extract(&self, input: &str) -> Result<FeatureExtractionResult>;
    
    /// 批量提取特征
    async fn extract_batch(&self, inputs: &[String]) -> Result<Vec<FeatureExtractionResult>> {
        let mut results = Vec::with_capacity(inputs.len());
        
        for input in inputs {
            let result = self.extract(input).await?;
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// 提取并返回特征向量
    async fn extract_vector(&self, input: &str) -> Result<FeatureVector> {
        let result = self.extract(input).await?;
        Ok(result.to_feature_vector())
    }
    
    /// 批量提取并返回特征向量
    async fn extract_vectors(&self, inputs: &[String]) -> Result<Vec<FeatureVector>> {
        let results = self.extract_batch(inputs).await?;
        Ok(results.into_iter().map(|r| r.to_feature_vector()).collect())
    }
    
    /// 重置提取器状态
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
    
    /// 关闭提取器并释放资源
    fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

/// 特征融合策略
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusionStrategy {
    /// 向量拼接
    Concatenate,
    /// 加权平均
    WeightedAverage,
    /// 最大池化
    MaxPooling,
    /// 平均池化
    AveragePooling,
    /// 加权投票
    WeightedVoting,
}

/// 特征提取评估器
pub trait FeatureEvaluator: Send + Sync {
    /// 评估特征提取结果
    fn evaluate(&self, result: &FeatureExtractionResult) -> Result<crate::data::feature::evaluator::FeatureEvaluationResult>;
    
    /// 批量评估特征提取结果
    fn evaluate_batch(&self, results: &[FeatureExtractionResult]) -> Result<Vec<crate::data::feature::evaluator::FeatureEvaluationResult>>;
    
    /// 计算特征质量评分
    fn calculate_quality_score(&self, result: &FeatureExtractionResult) -> f32;
}

/// 特征提取器创建函数类型
pub type ExtractorCreator = fn(config: ExtractorConfig) -> Result<Box<dyn FeatureExtractor>>;

/// 特征提取器工厂
pub trait FeatureExtractorFactory: Send + Sync {
    /// 注册特征提取器创建函数
    fn register(&mut self, extractor_type: ExtractorType, creator: ExtractorCreator) -> Result<()>;
    
    /// 创建特征提取器
    fn create(&self, config: ExtractorConfig) -> Result<Box<dyn FeatureExtractor>>;
    
    /// 创建文本特征提取器
    fn create_text_extractor(&self, config: ExtractorConfig) -> Result<Box<dyn FeatureExtractor>> {
        use crate::data::feature::types::TextExtractorType;
        let config = ExtractorConfig {
            extractor_type: if matches!(config.extractor_type, ExtractorType::Text(_)) {
                config.extractor_type
            } else {
                ExtractorType::Text(TextExtractorType::TfIdf)
            },
            feature_type: FeatureType::Text,
            ..config
        };
        self.create(config)
    }
    
    /// 创建图像特征提取器
    fn create_image_extractor(&self, config: ExtractorConfig) -> Result<Box<dyn FeatureExtractor>> {
        let config = ExtractorConfig {
            extractor_type: ExtractorType::Image,
            feature_type: FeatureType::Image,
            ..config
        };
        self.create(config)
    }
    
    /// 创建通用特征提取器
    fn create_generic_extractor(&self, config: ExtractorConfig) -> Result<Box<dyn FeatureExtractor>> {
        use crate::data::feature::types::GenericExtractorType;
        let config = ExtractorConfig {
            extractor_type: if matches!(config.extractor_type, ExtractorType::Generic(_)) {
                config.extractor_type
            } else {
                ExtractorType::Generic(GenericExtractorType::Identity)
            },
            feature_type: FeatureType::Generic,
            ..config
        };
        self.create(config)
    }
    
    /// 获取所有已注册的提取器类型
    fn get_registered_types(&self) -> Vec<ExtractorType>;
    
    /// 检查指定提取器类型是否已注册
    fn is_registered(&self, extractor_type: ExtractorType) -> bool;
}

/// 特征评估结果
#[derive(Debug, Clone)]
pub struct FeatureEvaluation {
    /// 特征维度
    pub dimension: usize,
    /// 提取时间（毫秒）
    pub extraction_time_ms: u64,
    /// 质量得分（0.0-1.0）
    pub quality_score: f64,
    /// 内存使用（KB）
    pub memory_usage_kb: u64,
    /// 额外评估指标
    pub metrics: HashMap<String, f64>,
}

/// 增强特征评估器接口
pub trait EnhancedFeatureEvaluator: Send + Sync {
    /// 评估特征提取结果
    fn evaluate(&self, features: &FeatureExtractionResult) -> Result<FeatureEvaluation>;
    
    /// 批量评估特征提取结果
    fn batch_evaluate(&self, features: &[FeatureExtractionResult]) -> Result<Vec<FeatureEvaluation>> {
        let mut results = Vec::with_capacity(features.len());
        
        for feature in features {
            let evaluation = self.evaluate(feature)?;
            results.push(evaluation);
        }
        
        Ok(results)
    }
    
    /// 计算特征质量得分
    fn calculate_quality_score(&self, features: &[f32]) -> f64;
    
    /// 获取评估器名称
    fn name(&self) -> &str;
}

// 添加FeatureVector结构体定义
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// 特征向量
    pub data: Vec<f32>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

impl FeatureVector {
    /// 创建新的特征向量
    pub fn new(data: Vec<f32>) -> Self {
        Self {
            data,
            metadata: HashMap::new(),
        }
    }
    
    /// 添加元数据
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }
    
    /// 获取特征维度
    pub fn dimension(&self) -> usize {
        self.data.len()
    }
}

/// 提取器上下文
/// 提供特征提取过程中的上下文信息
#[derive(Debug, Clone)]
pub struct ExtractorContext {
    /// 上下文ID
    id: String,
    /// 批次索引
    batch_index: Option<usize>,
    /// 输入数据引用
    input_data_id: Option<String>,
    /// 上下文元数据
    metadata: HashMap<String, String>,
    /// 创建时间戳
    created_at: std::time::SystemTime,
}

impl ExtractorContext {
    /// 创建新的上下文
    pub fn new() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            batch_index: None,
            input_data_id: None,
            metadata: HashMap::new(),
            created_at: std::time::SystemTime::now(),
        }
    }

    /// 设置批次索引
    pub fn with_batch_index(mut self, index: usize) -> Self {
        self.batch_index = Some(index);
        self
    }

    /// 设置输入数据引用
    pub fn with_input_data(mut self, data: &InputData) -> Self {
        self.input_data_id = Some(data.id.clone());
        self
    }

    /// 添加元数据
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// 获取上下文ID
    pub fn get_id(&self) -> String {
        self.id.clone()
    }

    /// 获取批次索引
    pub fn get_batch_index(&self) -> Option<usize> {
        self.batch_index
    }

    /// 获取输入数据ID
    pub fn get_input_data_id(&self) -> Option<&String> {
        self.input_data_id.as_ref()
    }

    /// 获取元数据
    pub fn get_metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// 获取创建时间
    pub fn get_created_at(&self) -> std::time::SystemTime {
        self.created_at
    }
}

/// 输入数据
/// 封装特征提取的输入数据和相关信息
#[derive(Debug, Clone)]
pub struct InputData {
    /// 数据ID
    pub id: String,
    /// 数据内容
    pub content: String,
    /// 数据类型
    pub data_type: String,
    /// 数据大小（字节）
    pub size: usize,
    /// 数据元数据
    pub metadata: HashMap<String, String>,
    /// 创建时间戳
    pub created_at: std::time::SystemTime,
}

impl InputData {
    /// 创建新的输入数据
    pub fn new(content: String, data_type: String) -> Self {
        let size = content.len();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content,
            data_type,
            size,
            metadata: HashMap::new(),
            created_at: std::time::SystemTime::now(),
        }
    }

    /// 从字节数据创建
    pub fn from_bytes(data: &[u8], data_type: String) -> Result<Self> {
        let content = String::from_utf8(data.to_vec())
            .map_err(|e| Error::invalid_argument(format!("无效的UTF-8数据: {}", e)))?;
        Ok(Self::new(content, data_type))
    }

    /// 添加元数据
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// 获取数据内容作为字节
    pub fn as_bytes(&self) -> &[u8] {
        self.content.as_bytes()
    }

    /// 判断数据是否为空
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// 获取数据长度
    pub fn len(&self) -> usize {
        self.size
    }
}

/// 特征提取错误
#[derive(Debug, Clone)]
pub enum ExtractorError {
    /// 无效输入
    InvalidInput(String),
    /// 维度不匹配
    DimensionMismatch(String),
    /// 处理错误
    ProcessingError(String),
    /// 超时错误
    Timeout(String),
    /// 配置错误
    ConfigurationError(String),
    /// 资源不足
    ResourceExhausted(String),
    /// 网络错误
    NetworkError(String),
    /// IO错误
    IoError(String),
    /// 解析错误
    ParseError(String),
    /// 验证错误
    ValidationError(String),
    /// 未知错误
    Unknown(String),
}

impl std::fmt::Display for ExtractorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtractorError::InvalidInput(msg) => write!(f, "无效输入: {}", msg),
            ExtractorError::DimensionMismatch(msg) => write!(f, "维度不匹配: {}", msg),
            ExtractorError::ProcessingError(msg) => write!(f, "处理错误: {}", msg),
            ExtractorError::Timeout(msg) => write!(f, "超时错误: {}", msg),
            ExtractorError::ConfigurationError(msg) => write!(f, "配置错误: {}", msg),
            ExtractorError::ResourceExhausted(msg) => write!(f, "资源不足: {}", msg),
            ExtractorError::NetworkError(msg) => write!(f, "网络错误: {}", msg),
            ExtractorError::IoError(msg) => write!(f, "IO错误: {}", msg),
            ExtractorError::ParseError(msg) => write!(f, "解析错误: {}", msg),
            ExtractorError::ValidationError(msg) => write!(f, "验证错误: {}", msg),
            ExtractorError::Unknown(msg) => write!(f, "未知错误: {}", msg),
        }
    }
}

impl std::error::Error for ExtractorError {}

impl From<ExtractorError> for Error {
    fn from(err: ExtractorError) -> Self {
        match err {
            ExtractorError::InvalidInput(msg) => Error::invalid_argument(msg),
            ExtractorError::DimensionMismatch(msg) => Error::invalid_argument(msg),
            ExtractorError::ProcessingError(msg) => Error::processing(msg),
            ExtractorError::Timeout(msg) => Error::timeout(msg),
            ExtractorError::ConfigurationError(msg) => Error::config(msg),
            ExtractorError::ResourceExhausted(msg) => Error::Resource(msg),
            ExtractorError::NetworkError(msg) => Error::Network(msg),
            ExtractorError::IoError(msg) => Error::Io(std::io::Error::new(std::io::ErrorKind::Other, msg)),
            ExtractorError::ParseError(msg) => Error::invalid_data(msg),
            ExtractorError::ValidationError(msg) => Error::validation(msg),
            ExtractorError::Unknown(msg) => Error::other(msg),
        }
    }
}

/// 特征批次
/// 用于批量处理特征向量
#[derive(Debug, Clone)]
pub struct FeatureBatch {
    /// 批次ID
    pub id: String,
    /// 特征向量集合
    pub feature_vectors: Vec<FeatureVector>,
    /// 批次元数据
    pub metadata: HashMap<String, String>,
    /// 批次大小
    pub batch_size: usize,
    /// 创建时间戳
    pub created_at: std::time::SystemTime,
}

impl FeatureBatch {
    /// 创建新的特征批次
    pub fn new(feature_vectors: Vec<FeatureVector>) -> Self {
        let batch_size = feature_vectors.len();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            feature_vectors,
            metadata: HashMap::new(),
            batch_size,
            created_at: std::time::SystemTime::now(),
        }
    }

    /// 创建空的特征批次
    pub fn empty() -> Self {
        Self::new(Vec::new())
    }

    /// 添加特征向量
    pub fn add_feature_vector(&mut self, vector: FeatureVector) {
        self.feature_vectors.push(vector);
        self.batch_size = self.feature_vectors.len();
    }

    /// 添加多个特征向量
    pub fn extend_feature_vectors(&mut self, vectors: Vec<FeatureVector>) {
        self.feature_vectors.extend(vectors);
        self.batch_size = self.feature_vectors.len();
    }

    /// 添加元数据
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// 获取批次大小
    pub fn len(&self) -> usize {
        self.batch_size
    }

    /// 判断批次是否为空
    pub fn is_empty(&self) -> bool {
        self.batch_size == 0
    }

    /// 获取特征向量迭代器
    pub fn iter(&self) -> std::slice::Iter<FeatureVector> {
        self.feature_vectors.iter()
    }

    /// 获取特征向量可变迭代器
    pub fn iter_mut(&mut self) -> std::slice::IterMut<FeatureVector> {
        self.feature_vectors.iter_mut()
    }

    /// 分割批次
    pub fn split(self, chunk_size: usize) -> Vec<FeatureBatch> {
        if chunk_size == 0 {
            return vec![self];
        }

        let mut batches = Vec::new();
        let chunks: Vec<_> = self.feature_vectors.chunks(chunk_size).collect();
        
        for chunk in chunks {
            let mut batch = FeatureBatch::new(chunk.to_vec());
            batch.metadata = self.metadata.clone();
            batches.push(batch);
        }

        batches
    }

    /// 合并多个批次
    pub fn merge(batches: Vec<FeatureBatch>) -> Self {
        let mut all_vectors = Vec::new();
        let mut combined_metadata = HashMap::new();

        for batch in batches {
            all_vectors.extend(batch.feature_vectors);
            combined_metadata.extend(batch.metadata);
        }

        let mut merged = FeatureBatch::new(all_vectors);
        merged.metadata = combined_metadata;
        merged
    }
}

impl IntoIterator for FeatureBatch {
    type Item = FeatureVector;
    type IntoIter = std::vec::IntoIter<FeatureVector>;

    fn into_iter(self) -> Self::IntoIter {
        self.feature_vectors.into_iter()
    }
} 