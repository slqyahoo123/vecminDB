// 特征提取器工厂模块
// 提供特征提取器的创建和管理功能

use crate::Result;
use crate::Error;
use crate::data::text_features::types::TextFeatureMethod;
use crate::data::text_features::config::TextFeatureConfig;
use std::collections::HashMap;
use std::sync::RwLock;
use once_cell::sync::Lazy;
// Arc and lazy_static are not used here

use super::FeatureExtractor;
use super::tfidf::TfIdfExtractor;
use super::mixed::MixedFeatureExtractor;
use super::enhanced_representations::EnhancedFeatureExtractor;
use super::basic::BasicExtractor;
use super::bert::BertExtractor;
// use super::word2vec::Word2VecExtractor;
use super::ngram::NGramExtractor;
use super::entity::EntityExtractor;
use super::topic::TopicExtractor;
use super::sentiment::SentimentExtractor;

// 提取器创建函数类型
type ExtractorCreator = fn(&TextFeatureConfig) -> Result<Box<dyn FeatureExtractor>>;

// 提取器注册表
static EXTRACTOR_REGISTRY: Lazy<RwLock<HashMap<TextFeatureMethod, ExtractorCreator>>> = Lazy::new(|| {
    let mut registry = HashMap::new();
    
    // 注册默认提取器
    registry.insert(TextFeatureMethod::TfIdf, create_tfidf_extractor as ExtractorCreator);
    registry.insert(TextFeatureMethod::BagOfWords, create_bow_extractor as ExtractorCreator);
    registry.insert(TextFeatureMethod::Word2Vec, create_basic_extractor as ExtractorCreator);
    registry.insert(TextFeatureMethod::FastText, create_basic_extractor as ExtractorCreator);
    registry.insert(TextFeatureMethod::Bert, create_bert_extractor as ExtractorCreator);
    registry.insert(TextFeatureMethod::NGram, create_ngram_extractor as ExtractorCreator);
    registry.insert(TextFeatureMethod::Mixed, create_mixed_extractor as ExtractorCreator);
    registry.insert(TextFeatureMethod::EntityExtraction, create_entity_extractor as ExtractorCreator);
    registry.insert(TextFeatureMethod::TopicModeling, create_topic_extractor as ExtractorCreator);
    registry.insert(TextFeatureMethod::SentimentAnalysis, create_sentiment_extractor as ExtractorCreator);
    registry.insert(TextFeatureMethod::EnhancedRepresentation, create_enhanced_extractor as ExtractorCreator);
    
    RwLock::new(registry)
});

/// 创建特征提取器
/// 根据配置创建相应的特征提取器
pub fn create_extractor(config: &TextFeatureConfig) -> Result<Box<dyn FeatureExtractor>> {
    let registry = EXTRACTOR_REGISTRY.read().unwrap();
    
    // 根据配置的提取方法确定使用哪种提取器
    let method = match config.extraction_method.as_str() {
        "tfidf" => TextFeatureMethod::TfIdf,
        "bow" | "bagofwords" => TextFeatureMethod::BagOfWords,
        "word2vec" => TextFeatureMethod::Word2Vec,
        "fasttext" => TextFeatureMethod::FastText,
        "bert" => TextFeatureMethod::Bert,
        "ngram" => TextFeatureMethod::NGram,
        "mixed" => TextFeatureMethod::Mixed,
        "entity" => TextFeatureMethod::EntityExtraction,
        "topic" => TextFeatureMethod::TopicModeling,
        "sentiment" => TextFeatureMethod::SentimentAnalysis,
        "enhanced" => TextFeatureMethod::EnhancedRepresentation,
        _ => TextFeatureMethod::TfIdf, // 默认使用TF-IDF
    };
    
    if let Some(creator) = registry.get(&method) {
        creator(config)
    } else {
        // 默认使用基础特征提取器
        create_basic_extractor(config)
    }
}

/// 从配置创建特征提取器的工厂函数
pub fn create_text_feature_extractor(config: &TextFeatureConfig) -> Result<Box<dyn FeatureExtractor>> {
    create_extractor(config)
}

/// 从配置创建特征提取器的工厂函数(相同功能但保留以兼容现有代码)
pub fn create_extractor_from_config(config: &TextFeatureConfig) -> Result<Box<dyn FeatureExtractor>> {
    create_extractor(config)
}

/// 注册新的特征提取器
pub fn register_extractor(method: TextFeatureMethod, creator: ExtractorCreator) -> Result<()> {
    let mut registry = EXTRACTOR_REGISTRY.write().unwrap();
    
    if registry.contains_key(&method) {
        return Err(Error::AlreadyExists(format!("特征提取器类型 {:?} 已注册", method)));
    }
    
    registry.insert(method, creator);
    Ok(())
}

// 以下是各种提取器的创建函数

/// 创建TF-IDF特征提取器
pub fn create_tfidf_extractor(config: &TextFeatureConfig) -> Result<Box<dyn FeatureExtractor>> {
    Ok(Box::new(TfIdfExtractor::from_config(config)?))
}

/// 创建词袋模型特征提取器
pub fn create_bow_extractor(config: &TextFeatureConfig) -> Result<Box<dyn FeatureExtractor>> {
    let mut bow_config = config.clone();
    // 注意：TextFeatureConfig中没有use_idf字段，这里需要调整
    Ok(Box::new(TfIdfExtractor::from_config(&bow_config)?))
}

/// 创建基础特征提取器
pub fn create_basic_extractor(config: &TextFeatureConfig) -> Result<Box<dyn FeatureExtractor>> {
    Ok(Box::new(BasicExtractor::from_config(config)?))
}

/// 创建BERT特征提取器
pub fn create_bert_extractor(config: &TextFeatureConfig) -> Result<Box<dyn FeatureExtractor>> {
    Ok(Box::new(BertExtractor::from_config(config)?))
}

fn create_ngram_extractor(config: &TextFeatureConfig) -> Result<Box<dyn FeatureExtractor>> {
    Ok(Box::new(NGramExtractor::from_config(config)?))
}

fn create_mixed_extractor(config: &TextFeatureConfig) -> Result<Box<dyn FeatureExtractor>> {
    Ok(Box::new(MixedFeatureExtractor::from_config(config)?))
}

fn create_entity_extractor(config: &TextFeatureConfig) -> Result<Box<dyn FeatureExtractor>> {
    Ok(Box::new(EntityExtractor::from_config(config)?))
}

fn create_topic_extractor(config: &TextFeatureConfig) -> Result<Box<dyn FeatureExtractor>> {
    Ok(Box::new(TopicExtractor::from_config(config)?))
}

fn create_sentiment_extractor(config: &TextFeatureConfig) -> Result<Box<dyn FeatureExtractor>> {
    Ok(Box::new(SentimentExtractor::from_config(config)?))
}

fn create_enhanced_extractor(config: &TextFeatureConfig) -> Result<Box<dyn FeatureExtractor>> {
    Ok(Box::new(EnhancedFeatureExtractor::from_config(config)?))
}
