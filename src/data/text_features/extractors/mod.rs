// Feature Extractors Module
// Provides various types of feature extraction capabilities

// Basic Feature Extractors
pub mod basic;
pub mod tfidf;
pub mod bert;

// Advanced Feature Extractors
pub mod mixed;
pub mod advanced_representations;
pub mod enhanced_representations;
pub mod multimodal;

// Text Specific Feature Extractors
pub mod ngram;
pub mod entity;
pub mod topic;
pub mod sentiment;

// Models and Factory
pub mod models;
pub mod factory;

// Re-export public APIs
pub use self::mixed::MixedFeatureExtractor;
pub use self::enhanced_representations::EnhancedFeatureExtractor;
pub use self::basic::BasicExtractor;
pub use self::bert::BertExtractor;
pub use self::tfidf::TfIdfExtractor;
pub use self::ngram::NGramExtractor;
pub use self::entity::EntityExtractor;
pub use self::topic::TopicExtractor;
pub use self::sentiment::SentimentExtractor;

// Export factory functions
pub use factory::{create_extractor, create_extractor_from_config, register_extractor};

use crate::data::text_features::config::TextFeatureConfig;
use crate::data::text_features::methods::TextFeatureMethod;
use crate::Result;
use std::sync::Arc;

/// Feature Extractor trait
/// 
/// All feature extractors should implement this trait
pub trait FeatureExtractor: Send + Sync {
    /// Extract features from text
    fn extract(&self, text: &str) -> Result<Vec<f32>>;
    
    /// Get feature dimension
    fn dimension(&self) -> usize;
    
    /// Get extractor name
    fn name(&self) -> &str;
    
    /// Create extractor from config
    fn from_config(config: &TextFeatureConfig) -> Result<Self> where Self: Sized;
    
    /// Get output dimension of the extractor
    fn get_output_dimension(&self) -> Result<usize> {
        Ok(self.dimension())
    }
    
    /// Get extractor type identifier
    fn get_extractor_type(&self) -> String {
        self.name().to_string()
    }
}

/// 包装原始的TextFeatureExtractor以便与其他接口集成
pub struct TextExtractorWrapper {
    inner: Arc<dyn FeatureExtractor>,
}

impl TextExtractorWrapper {
    /// 创建新的包装器
    pub fn new(extractor: Arc<dyn FeatureExtractor>) -> Self {
        Self {
            inner: extractor,
        }
    }
    
    /// 获取内部提取器的引用
    pub fn inner(&self) -> &Arc<dyn FeatureExtractor> {
        &self.inner
    }
    
    /// 提取特征向量
    pub fn extract(&self, text: &str) -> Result<Vec<f32>> {
        self.inner.extract(text)
    }
    
    /// 获取特征维度
    pub fn dimension(&self) -> usize {
        self.inner.dimension()
    }
    
    /// 获取提取器名称
    pub fn name(&self) -> &str {
        self.inner.name()
    }
    
    /// 获取输出维度
    pub fn get_output_dimension(&self) -> Result<usize> {
        self.inner.get_output_dimension()
    }
    
    /// 获取提取器类型
    pub fn get_extractor_type(&self) -> String {
        self.inner.get_extractor_type()
    }
    
    /// 从配置创建
    pub fn from_text_config(config: &TextFeatureConfig, extractor: Arc<dyn FeatureExtractor>) -> Self {
        Self {
            inner: extractor,
        }
    }
    
    /// 批量处理文本
    pub fn batch_extract(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        
        for text in texts {
            let features = self.inner.extract(text)?;
            results.push(features);
        }
        
        Ok(results)
    }
    
    /// 从JSON数据中提取文本特征
    pub async fn extract_from_json(&self, data: &[serde_json::Value], field: &str) -> Result<Vec<Vec<f64>>> {
        let mut features = Vec::new();
        
        for item in data {
            let text = match item {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Object(obj) => {
                    if let Some(serde_json::Value::String(s)) = obj.get(field) {
                        s.clone()
                    } else {
                        continue; // 跳过没有指定字段的项
                    }
                },
                _ => continue, // 跳过不支持的类型
            };
            
            // 提取特征并转换为f64
            let feature_vec = self.inner.extract(&text)?;
            let feature_f64: Vec<f64> = feature_vec.into_iter().map(|f| f as f64).collect();
            features.push(feature_f64);
        }
        
        Ok(features)
    }
}

/// 实现Debug特性
impl std::fmt::Debug for TextExtractorWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextExtractorWrapper")
            .field("name", &self.name())
            .field("dimension", &self.dimension())
            .finish()
    }
}

/// 根据方法创建文本特征提取器
pub fn create_text_extractor(method: TextFeatureMethod) -> Result<TextExtractorWrapper> {
    match method {
        TextFeatureMethod::TfIdf => {
            let config = TextFeatureConfig::default();
            Ok(TextExtractorWrapper::new(Arc::new(TfIdfExtractor::from_config(&config)?)))
        },
        TextFeatureMethod::Bert => {
            let config = TextFeatureConfig::default();
            Ok(TextExtractorWrapper::new(Arc::new(BertExtractor::from_config(&config)?)))
        },
        TextFeatureMethod::BertEmbedding => {
            let config = TextFeatureConfig::default();
            Ok(TextExtractorWrapper::new(Arc::new(BertExtractor::from_config(&config)?)))
        },
        TextFeatureMethod::Word2Vec => {
            let config = TextFeatureConfig::default();
            Ok(TextExtractorWrapper::new(Arc::new(BasicExtractor::from_config(&config)?)))
        },
        TextFeatureMethod::FastText => {
            let config = TextFeatureConfig::default();
            Ok(TextExtractorWrapper::new(Arc::new(BasicExtractor::from_config(&config)?)))
        },
        TextFeatureMethod::Count => {
            let config = TextFeatureConfig::default();
            Ok(TextExtractorWrapper::new(Arc::new(BasicExtractor::from_config(&config)?)))
        },
        TextFeatureMethod::NGram => {
            let config = TextFeatureConfig::default();
            Ok(TextExtractorWrapper::new(Arc::new(NGramExtractor::from_config(&config)?)))
        },
        TextFeatureMethod::BagOfWords => {
            let config = TextFeatureConfig::default();
            Ok(TextExtractorWrapper::new(Arc::new(BasicExtractor::from_config(&config)?)))
        },
        TextFeatureMethod::WordFrequency => {
            let config = TextFeatureConfig::default();
            Ok(TextExtractorWrapper::new(Arc::new(BasicExtractor::from_config(&config)?)))
        },
        TextFeatureMethod::CharacterLevel => {
            let config = TextFeatureConfig::default();
            Ok(TextExtractorWrapper::new(Arc::new(BasicExtractor::from_config(&config)?)))
        },
        TextFeatureMethod::Statistical => {
            let config = TextFeatureConfig::default();
            Ok(TextExtractorWrapper::new(Arc::new(BasicExtractor::from_config(&config)?)))
        },
        TextFeatureMethod::GloVe => {
            let config = TextFeatureConfig::default();
            Ok(TextExtractorWrapper::new(Arc::new(BasicExtractor::from_config(&config)?)))
        },
        TextFeatureMethod::Universal => {
            let config = TextFeatureConfig::default();
            Ok(TextExtractorWrapper::new(Arc::new(BasicExtractor::from_config(&config)?)))
        },
        TextFeatureMethod::Elmo => {
            let config = TextFeatureConfig::default();
            Ok(TextExtractorWrapper::new(Arc::new(BasicExtractor::from_config(&config)?)))
        },
        TextFeatureMethod::AutoSelect => {
            let config = TextFeatureConfig::default();
            Ok(TextExtractorWrapper::new(Arc::new(BasicExtractor::from_config(&config)?)))
        },
        TextFeatureMethod::Custom(name) => {
            let config = TextFeatureConfig::default();
            // 对于自定义方法，使用基本提取器作为回退
            Ok(TextExtractorWrapper::new(Arc::new(BasicExtractor::from_config(&config)?)))
        },
    }
} 