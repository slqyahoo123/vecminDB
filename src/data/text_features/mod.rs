// Text Feature Extraction Module
// Provides various text feature extraction methods and configuration options

// Basic Types
pub mod types;
pub mod stats;
pub mod config;
pub mod api;
pub mod features;
pub mod evaluation;

// Transformer modules
pub mod error;
pub mod tokenizer;
pub mod encoder;
pub mod similarity;
pub mod language;
pub mod model;
// pub mod training; // Removed: vector database does not need training functionality
pub mod inference;

// Feature Extractors
pub mod extractors;

// Preprocessing and Utilities
pub mod preprocessing;
pub mod utils;
pub mod processors;

// Bridge Modules
pub mod cleaners;
pub mod normalizers;
pub mod tokenizers;
pub mod filters;
pub mod transformer;
pub mod transformers;
pub mod augmentors;
pub mod analyzers;
pub mod custom;

// Text Representation
pub mod vectorizer;
pub mod embedding;

// Feature Fusion and Storage
pub mod fusion;
pub mod incremental;
pub mod pipeline;
pub mod vector_store;
pub mod vector_index;

// Method Implementation
pub mod methods;

// Export Public API from Submodules
pub use self::config::{TextFeatureConfig, MixedFeatureConfig, TextFeatureMethod, TransformerConfig};
pub use self::preprocessing::{TextPreprocessor as PreprocessingTextPreprocessor};
pub use crate::data::text_features::config::PreprocessingConfig;
pub use crate::data::text_features::types::{CleaningStrategy, NormalizationStrategy};
pub use self::types::{FieldType};
pub use self::extractors::models::{TextFieldStats, NumericStats, CategoricalFieldStats, DataCharacteristics};
pub use self::extractors::mixed::MixedFeatureExtractor;
pub use self::extractors::FeatureExtractor as ExtractorsFeatureExtractor;
pub use self::extractors::basic::BasicExtractor;
pub use self::extractors::tfidf::TfIdfExtractor;
pub use self::extractors::bert::BertExtractor;
pub use self::extractors::advanced_representations::EnhancedFeatureExtractor;
pub use self::extractors::multimodal::{MultimodalFeatureExtractor as MultiModalExtractor, NumericFeatureConfig};
pub use self::fusion::{FusionStrategy, FeatureFusion, AdaptiveFusion, AdaptiveFusionConfig, AttentionMethod};
pub use self::incremental::IncrementalLearningState;
pub use self::processors::{TextProcessor};
pub use self::evaluation::EvaluationWeights;
pub use self::features::TextFeatures;
pub use self::types::FeatureExtractionResult;

// Export Transformer module components
pub use self::transformer::{
    // Sub-module exports
    create_transformer_model,
    create_encoder,
    create_tokenizer,
    create_feature_extractor,
    create_similarity_calculator,
    create_language_processor,
    create_trainer,
    create_inference_engine,
    create_inference_predictor,
    
    // Convenience functions
    process_text,
    compute_similarity,
    process_batch,
    train_model,
    infer_text,
    batch_infer,
    predict_class,
    predict_similarity,
    
    // Configuration functions
    get_default_config,
    get_default_encoder_config,
    get_default_feature_config,
    get_default_similarity_config,
    get_default_language_config,
    get_default_training_config,
    get_default_inference_config,
    get_default_prediction_config,
    validate_config,
    
    // Utility functions
    get_supported_languages,
    get_supported_feature_types,
    get_supported_similarity_methods,
    get_model_info,
};

// Re-export sub-module types for convenience
pub use self::config::{TransformerConfig as TransformerModelConfig};
pub use self::error::TransformerError;
pub use self::model::TransformerModel;
pub use self::tokenizer::Tokenizer;
pub use self::encoder::{Encoder, EncoderConfig};
pub use self::features::{FeatureExtractor, FeatureExtractionConfig, FeatureVector, FeatureType};
pub use self::similarity::{SimilarityCalculator, SimilarityConfig, SimilarityMethod};
pub use self::language::{LanguageProcessor, LanguageProcessingConfig, Language};
pub use self::model::{ProcessedText, ModelState};
// Training types removed: vector database does not need training functionality
pub use self::inference::{InferenceEngine, InferenceConfig, InferenceResult, InferencePredictor, PredictionConfig};

use std::collections::HashSet;
use std::sync::Arc;
use regex::Regex;
use std::error::Error as StdError;
use serde::{Serialize, Deserialize};
use lazy_static::lazy_static;

/// 文本特征提取器特征
pub trait TextFeatureExtractor: Send + Sync {
    /// 提取文本特征向量
    /// 
    /// # 参数
    /// * `text` - 待提取特征的文本
    /// * `options` - 文本处理选项，如果为None则使用默认选项
    /// 
    /// # 返回
    /// 返回提取的特征向量，如果提取失败则返回错误
    fn extract_features(&self, text: &str, options: Option<&TextProcessingOptions>) -> std::result::Result<Vec<f32>, Box<dyn StdError>>;
    
    /// 批量提取文本特征向量
    /// 
    /// # 参数
    /// * `texts` - 待提取特征的文本列表
    /// * `options` - 文本处理选项，如果为None则使用默认选项
    /// 
    /// # 返回
    /// 返回提取的特征向量列表，如果提取失败则返回错误
    fn batch_extract_features(&self, texts: &[&str], options: Option<&TextProcessingOptions>) -> std::result::Result<Vec<Vec<f32>>, Box<dyn StdError>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            let features = self.extract_features(text, options)?;
            results.push(features);
        }
        Ok(results)
    }
    
    /// 计算两个文本之间的相似度
    /// 
    /// # 参数
    /// * `text1` - 第一个文本
    /// * `text2` - 第二个文本
    /// * `options` - 文本处理选项，如果为None则使用默认选项
    /// 
    /// # 返回
    /// 返回两个文本之间的相似度（0-1之间的值），如果计算失败则返回错误
    fn compute_similarity(&self, text1: &str, text2: &str, options: Option<&TextProcessingOptions>) -> std::result::Result<f32, Box<dyn StdError>>;
    
    /// 批量计算文本相似度
    /// 
    /// # 参数
    /// * `texts` - 文本列表
    /// * `query` - 查询文本
    /// * `options` - 文本处理选项，如果为None则使用默认选项
    /// 
    /// # 返回
    /// 返回查询文本与每个文本的相似度列表，如果计算失败则返回错误
    fn batch_compute_similarity(&self, texts: &[&str], query: &str, options: Option<&TextProcessingOptions>) -> std::result::Result<Vec<f32>, Box<dyn StdError>> {
        let mut similarities = Vec::with_capacity(texts.len());
        for text in texts {
            let similarity = self.compute_similarity(text, query, options)?;
            similarities.push(similarity);
        }
        Ok(similarities)
    }
    
    /// 获取特征维度
    /// 
    /// # 返回
    /// 返回特征向量的维度
    fn feature_dimension(&self) -> usize;
    
    /// 获取提取器名称
    /// 
    /// # 返回
    /// 返回特征提取器的名称
    fn name(&self) -> &str;
}

/// 文本处理选项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextProcessingOptions {
    /// 是否转换为小写
    pub lowercase: bool,
    /// 是否去除停用词
    pub remove_stopwords: bool,
    /// 是否去除标点符号
    pub remove_punctuation: bool,
    /// 是否去除数字
    pub remove_numbers: bool,
    /// 是否去除HTML标签
    pub remove_html: bool,
    /// 是否进行词干提取
    pub stemming: bool,
    /// 语言（影响停用词和词干提取）
    pub language: String,
    /// 最大文本长度（0表示不限制）
    pub max_length: usize,
    /// 自定义停用词列表
    pub custom_stopwords: Option<HashSet<String>>,
}

impl Default for TextProcessingOptions {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_stopwords: true,
            remove_punctuation: true,
            remove_numbers: true,
            remove_html: true,
            stemming: false,
            language: "en".to_string(),
            max_length: 0,
            custom_stopwords: None,
        }
    }
}

/// 获取默认文本处理选项
pub fn default_processing_options() -> TextProcessingOptions {
    TextProcessingOptions::default()
}

/// 文本预处理器
pub struct TextPreprocessor {
    /// 处理选项
    options: TextProcessingOptions,
    /// 停用词集
    stopwords: HashSet<String>,
    /// HTML标签正则表达式
    html_regex: Regex,
    /// 标点符号正则表达式
    punctuation_regex: Regex,
    /// 数字正则表达式
    number_regex: Regex,
}

impl TextPreprocessor {
    /// 创建新的文本预处理器
    pub fn new(options: TextProcessingOptions) -> Self {
        // 初始化停用词集
        let mut stopwords = match options.language.as_str() {
            "en" => DEFAULT_ENGLISH_STOPWORDS.iter().map(|&s| s.to_string()).collect(),
            "zh" => DEFAULT_CHINESE_STOPWORDS.iter().map(|&s| s.to_string()).collect(),
            _ => HashSet::new(),
        };
        
        // 添加自定义停用词
        if let Some(custom_stopwords) = &options.custom_stopwords {
            stopwords.extend(custom_stopwords.clone());
        }
        
        Self {
            options,
            stopwords,
            html_regex: Regex::new(r"<[^>]*>").unwrap(),
            punctuation_regex: Regex::new(r"[^\w\s]").unwrap(),
            number_regex: Regex::new(r"\d+").unwrap(),
        }
    }
    
    /// 创建使用默认选项的预处理器
    pub fn default() -> Self {
        Self::new(TextProcessingOptions::default())
    }
    
    /// 处理文本
    pub fn process(&self, text: &str) -> String {
        let mut processed = text.to_string();
        
        // 去除HTML标签
        if self.options.remove_html {
            processed = self.remove_html(&processed);
        }
        
        // 转换为小写
        if self.options.lowercase {
            processed = processed.to_lowercase();
        }
        
        // 去除标点符号
        if self.options.remove_punctuation {
            processed = self.remove_punctuation(&processed);
        }
        
        // 去除数字
        if self.options.remove_numbers {
            processed = self.remove_numbers(&processed);
        }
        
        // 分词
        let mut tokens: Vec<String> = processed.split_whitespace()
            .map(|s| s.to_string())
            .collect();
        
        // 去除停用词
        if self.options.remove_stopwords {
            tokens = self.remove_stopwords(&tokens);
        }
        
        // 词干提取
        if self.options.stemming {
            tokens = self.apply_stemming(&tokens);
        }
        
        // 限制长度
        if self.options.max_length > 0 && tokens.len() > self.options.max_length {
            tokens.truncate(self.options.max_length);
        }
        
        // 重新组合为文本
        tokens.join(" ")
    }
    
    /// 去除HTML标签
    pub fn remove_html(&self, text: &str) -> String {
        self.html_regex.replace_all(text, "").to_string()
    }
    
    /// 去除标点符号
    pub fn remove_punctuation(&self, text: &str) -> String {
        self.punctuation_regex.replace_all(text, "").to_string()
    }
    
    /// 去除数字
    pub fn remove_numbers(&self, text: &str) -> String {
        self.number_regex.replace_all(text, "").to_string()
    }
    
    /// 去除停用词
    pub fn remove_stopwords(&self, tokens: &[String]) -> Vec<String> {
        tokens.iter()
            .filter(|token| !self.stopwords.contains(*token))
            .cloned()
            .collect()
    }
    
    /// 应用词干提取
    pub fn apply_stemming(&self, tokens: &[String]) -> Vec<String> {
        // 这里应该使用词干提取库，如rust-stemmers
        // 但为了简化示例，我们只返回原始token
        tokens.to_vec()
    }
}

// 默认英文停用词
lazy_static! {
    static ref DEFAULT_ENGLISH_STOPWORDS: [&'static str; 25] = [
        "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", 
        "be", "been", "being", "in", "on", "at", "to", "for", "with", "by", 
        "about", "against", "between", "into", "through"
    ];
    
    static ref DEFAULT_CHINESE_STOPWORDS: [&'static str; 15] = [
        "的", "了", "和", "是", "在", "有", "我", "你", "他", "她", 
        "它", "们", "这", "那", "都"
    ];
}

/// TransformerModel实现TextFeatureExtractor特性
impl TextFeatureExtractor for TransformerModel {
    fn extract_features(&self, text: &str, options: Option<&TextProcessingOptions>) -> std::result::Result<Vec<f32>, Box<dyn StdError>> {
        // 使用预处理选项
        let processed_text = if let Some(opts) = options {
            let preprocessor = TextPreprocessor::new(opts.clone());
            preprocessor.process(text)
        } else {
            text.to_string()
        };
        
        // 使用Transformer模型提取特征
        let processed = self.process_text(&processed_text)
            .map_err(|e| Box::new(e) as Box<dyn StdError>)?;
        
        Ok(processed.encoded)
    }
    
    fn compute_similarity(&self, text1: &str, text2: &str, options: Option<&TextProcessingOptions>) -> std::result::Result<f32, Box<dyn StdError>> {
        // 使用预处理选项
        let (processed_text1, processed_text2) = if let Some(opts) = options {
            let preprocessor = TextPreprocessor::new(opts.clone());
            (preprocessor.process(text1), preprocessor.process(text2))
        } else {
            (text1.to_string(), text2.to_string())
        };
        
        // 使用Transformer模型计算相似度
        // 提取两个文本的特征向量
        let features1 = self.extract_features(&processed_text1, None)?;
        let features2 = self.extract_features(&processed_text2, None)?;
        
        // 计算余弦相似度
        if features1.len() != features2.len() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("特征向量维度不匹配: {} vs {}", features1.len(), features2.len())
            )) as Box<dyn StdError>);
        }
        
        let dot_product: f32 = features1.iter().zip(features2.iter())
            .map(|(a, b)| a * b)
            .sum();
        
        let norm1: f32 = features1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = features2.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }
        
        let similarity = dot_product / (norm1 * norm2);
        Ok(similarity)
    }
    
    fn feature_dimension(&self) -> usize {
        self.get_config().hidden_size
    }
    
    fn name(&self) -> &str {
        self.get_config().model_name.as_str()
    }
}

/// 创建Transformer特征提取器
pub fn create_transformer_extractor(config: Option<TransformerConfig>) -> std::result::Result<Arc<dyn TextFeatureExtractor>, Box<dyn StdError>> {
    let config = config.unwrap_or_default();
    let model = TransformerModel::new(config)
        .map_err(|e| Box::new(e) as Box<dyn StdError>)?;
    
    Ok(Arc::new(model) as Arc<dyn TextFeatureExtractor>)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_preprocessor() {
        let options = TextProcessingOptions {
            lowercase: true,
            remove_stopwords: true,
            remove_punctuation: true,
            remove_numbers: true,
            remove_html: true,
            stemming: false,
            language: "en".to_string(),
            max_length: 0,
            custom_stopwords: None,
        };
        
        let preprocessor = TextPreprocessor::new(options);
        let text = "Hello World! This is a <b>test</b> sentence with 123 numbers.";
        let processed = preprocessor.process(text);
        
        assert_eq!(processed, "hello world test sentence numbers");
    }
    
    #[test]
    fn test_custom_stopwords() {
        let mut custom_stopwords = HashSet::new();
        custom_stopwords.insert("hello".to_string());
        custom_stopwords.insert("world".to_string());
        
        let options = TextProcessingOptions {
            lowercase: true,
            remove_stopwords: true,
            remove_punctuation: true,
            remove_numbers: true,
            remove_html: true,
            stemming: false,
            language: "en".to_string(),
            max_length: 0,
            custom_stopwords: Some(custom_stopwords),
        };
        
        let preprocessor = TextPreprocessor::new(options);
        let text = "Hello World! This is a test.";
        let processed = preprocessor.process(text);
        
        assert_eq!(processed, "test");
    }
    
    #[test]
    fn test_transformer_integration() {
        let config = TransformerConfig::default();
        let model = TransformerModel::new(config).unwrap();
        
        let result = model.process_text("Hello world");
        assert!(result.is_ok());
        
        let processed = result.unwrap();
        assert_eq!(processed.original_text, "Hello world");
        assert!(!processed.processed_text.is_empty());
    }
}

