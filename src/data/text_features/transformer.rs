// src/data/text_features/transformer.rs
//
// Transformer 模型主模块
// 提供基于Transformer架构的特征提取功能

use std::collections::HashMap;
use super::error::TransformerError;
use super::config::TransformerConfig;
use super::model::{TransformerModel, ProcessedText};
use super::encoder::{Encoder, EncoderConfig};
use super::tokenizer::Tokenizer;
use super::features::{FeatureExtractor, FeatureExtractionConfig, FeatureType};
use super::similarity::{SimilarityCalculator, SimilarityConfig, SimilarityMethod};
use super::language::{LanguageProcessor, LanguageProcessingConfig, Language};
// Training imports removed: vector database does not need training functionality
use super::inference::{InferenceEngine, InferenceConfig, InferenceResult, InferencePredictor, PredictionConfig};

/// 创建新的Transformer模型
pub fn create_transformer_model(config: TransformerConfig) -> Result<TransformerModel, TransformerError> {
    TransformerModel::new(config)
}

/// 创建新的编码器
pub fn create_encoder(config: EncoderConfig, tokenizer: Tokenizer, model_config: &TransformerConfig) -> Encoder {
    Encoder::new(config, tokenizer, model_config)
}

/// 创建新的分词器
pub fn create_tokenizer(config: TransformerConfig) -> Result<Tokenizer, TransformerError> {
    let mut tokenizer = Tokenizer::new(config);
    tokenizer.initialize()?;
    Ok(tokenizer)
}

/// 创建新的特征提取器
pub fn create_feature_extractor(config: FeatureExtractionConfig) -> FeatureExtractor {
    FeatureExtractor::new(config)
}

/// 创建新的相似度计算器
pub fn create_similarity_calculator(config: SimilarityConfig) -> SimilarityCalculator {
    SimilarityCalculator::new(config)
}

/// 创建新的语言处理器
pub fn create_language_processor(config: LanguageProcessingConfig) -> LanguageProcessor {
    LanguageProcessor::new(config)
}

// Training functions removed: vector database does not need training functionality

/// 创建新的推理引擎
pub fn create_inference_engine(config: InferenceConfig, model: TransformerModel) -> InferenceEngine {
    InferenceEngine::new(config, model)
}

/// 创建新的推理预测器
pub fn create_inference_predictor(
    engine: InferenceEngine,
    config: PredictionConfig,
) -> InferencePredictor {
    InferencePredictor::new(engine, config)
}

/// 便捷的文本处理函数
pub fn process_text(text: &str) -> Result<ProcessedText, TransformerError> {
    let config = TransformerConfig::default();
    let model = TransformerModel::new(config)?;
    model.process_text(text)
}

/// 便捷的相似度计算函数
pub fn compute_similarity(text1: &str, text2: &str) -> Result<f32, TransformerError> {
    let config = TransformerConfig::default();
    let mut model = TransformerModel::new(config)?;
    model.compute_similarity(text1, text2)
}

/// 便捷的批量处理函数
pub fn process_batch(texts: &[String]) -> Result<Vec<ProcessedText>, TransformerError> {
    let config = TransformerConfig::default();
    let model = TransformerModel::new(config)?;
    model.process_batch(texts)
}

// Training functions removed: vector database does not need training functionality

/// 便捷的推理函数
pub fn infer_text(text: &str, config: Option<InferenceConfig>) -> Result<InferenceResult, TransformerError> {
    let model_config = TransformerConfig::default();
    let model = TransformerModel::new(model_config)?;
    
    let inference_config = config.unwrap_or_default();
    let mut engine = InferenceEngine::new(inference_config, model);
    engine.infer(text)
}

/// 便捷的批量推理函数
pub fn batch_infer(texts: &[String], config: Option<InferenceConfig>) -> Result<Vec<InferenceResult>, TransformerError> {
    let model_config = TransformerConfig::default();
    let model = TransformerModel::new(model_config)?;
    
    let inference_config = config.unwrap_or_default();
    let mut engine = InferenceEngine::new(inference_config, model);
    engine.batch_infer(texts)
}

/// 便捷的预测函数
pub fn predict_class(text: &str, config: Option<PredictionConfig>) -> Result<super::inference::PredictionResult, TransformerError> {
    let model_config = TransformerConfig::default();
    let model = TransformerModel::new(model_config)?;
    
    let inference_config = InferenceConfig::default();
    let engine = InferenceEngine::new(inference_config, model);
    
    let prediction_config = config.unwrap_or_default();
    let mut predictor = InferencePredictor::new(engine, prediction_config);
    predictor.predict_class(text)
}

/// 便捷的相似度预测函数
pub fn predict_similarity(text1: &str, text2: &str, config: Option<PredictionConfig>) -> Result<super::inference::SimilarityPrediction, TransformerError> {
    let model_config = TransformerConfig::default();
    let model = TransformerModel::new(model_config)?;
    
    let inference_config = InferenceConfig::default();
    let engine = InferenceEngine::new(inference_config, model);
    
    let prediction_config = config.unwrap_or_default();
    let mut predictor = InferencePredictor::new(engine, prediction_config);
    predictor.predict_similarity(text1, text2)
}

/// 获取默认配置
pub fn get_default_config() -> TransformerConfig {
    TransformerConfig::default()
}

/// 获取默认编码器配置
pub fn get_default_encoder_config() -> EncoderConfig {
    EncoderConfig::default()
}

/// 获取默认特征提取配置
pub fn get_default_feature_config() -> FeatureExtractionConfig {
    FeatureExtractionConfig::default()
}

/// 获取默认相似度配置
pub fn get_default_similarity_config() -> SimilarityConfig {
    SimilarityConfig::default()
}

/// 获取默认语言处理配置
pub fn get_default_language_config() -> LanguageProcessingConfig {
    LanguageProcessingConfig::default()
}

// Training config functions removed: vector database does not need training functionality

/// 获取默认推理配置
pub fn get_default_inference_config() -> InferenceConfig {
    InferenceConfig::default()
}

/// 获取默认预测配置
pub fn get_default_prediction_config() -> PredictionConfig {
    PredictionConfig::default()
}

/// 验证配置的有效性
pub fn validate_config(config: &TransformerConfig) -> Result<(), TransformerError> {
    if config.hidden_size == 0 {
        return Err(TransformerError::config_error("隐藏层大小不能为0"));
    }
    
    if config.max_seq_length == 0 {
        return Err(TransformerError::config_error("最大序列长度不能为0"));
    }
    
    if config.vocab_size == 0 {
        return Err(TransformerError::config_error("词汇表大小不能为0"));
    }
    
    Ok(())
}

/// 获取支持的语言列表
pub fn get_supported_languages() -> Vec<Language> {
    vec![
        Language::English,
        Language::Chinese,
        Language::Japanese,
        Language::Korean,
        Language::Arabic,
        Language::Russian,
        Language::French,
        Language::German,
        Language::Spanish,
        Language::Italian,
        Language::Portuguese,
    ]
}

/// 获取支持的特征类型列表
pub fn get_supported_feature_types() -> Vec<FeatureType> {
    vec![
        FeatureType::WordFrequency,
        FeatureType::TfIdf,
        FeatureType::Ngram,
        FeatureType::Character,
        FeatureType::Semantic,
        FeatureType::Statistical,
    ]
}

/// 获取支持的相似度方法列表
pub fn get_supported_similarity_methods() -> Vec<SimilarityMethod> {
    vec![
        SimilarityMethod::Cosine,
        SimilarityMethod::Euclidean,
        SimilarityMethod::Manhattan,
        SimilarityMethod::Pearson,
        SimilarityMethod::Jaccard,
        SimilarityMethod::EditDistance,
    ]
}

/// 获取模型信息
pub fn get_model_info() -> HashMap<String, String> {
    let mut info = HashMap::new();
    info.insert("name".to_string(), "VecMind Transformer".to_string());
    info.insert("version".to_string(), "1.0.0".to_string());
    info.insert("description".to_string(), "基于Transformer架构的文本特征提取模型".to_string());
    info.insert("author".to_string(), "VecMind Team".to_string());
    info.insert("license".to_string(), "MIT".to_string());
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_transformer_model() {
        let config = TransformerConfig::default();
        let model = create_transformer_model(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_process_text() {
        let result = process_text("Hello world");
        assert!(result.is_ok());
        
        let processed = result.unwrap();
        assert_eq!(processed.original_text, "Hello world");
        assert!(!processed.processed_text.is_empty());
    }

    #[test]
    fn test_compute_similarity() {
        let result = compute_similarity("Hello world", "Hello universe");
        assert!(result.is_ok());
        
        let similarity = result.unwrap();
        assert!(similarity >= 0.0 && similarity <= 1.0);
    }

    #[test]
    fn test_batch_processing() {
        let texts = vec!["Hello".to_string(), "World".to_string()];
        let result = process_batch(&texts);
        assert!(result.is_ok());
        
        let processed = result.unwrap();
        assert_eq!(processed.len(), 2);
    }

    #[test]
    fn test_infer_text() {
        let result = infer_text("Hello world", None);
        assert!(result.is_ok());
        
        let inference = result.unwrap();
        assert_eq!(inference.input_text, "Hello world");
        assert!(!inference.output_encoding.is_empty());
    }

    #[test]
    fn test_validate_config() {
        let config = TransformerConfig::default();
        let result = validate_config(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_supported_languages() {
        let languages = get_supported_languages();
        assert!(!languages.is_empty());
        assert!(languages.contains(&Language::English));
        assert!(languages.contains(&Language::Chinese));
    }

    #[test]
    fn test_get_model_info() {
        let info = get_model_info();
        assert!(info.contains_key("name"));
        assert!(info.contains_key("version"));
        assert!(info.contains_key("description"));
    }
} 