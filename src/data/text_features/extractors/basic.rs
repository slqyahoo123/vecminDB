//! 基本文本特征提取器实现
//! 
//! 提供简单高效的文本特征提取方法

use std::collections::HashMap;
use crate::error::Result;
use crate::data::text_features::config::{TextFeatureConfig, PreprocessingConfig};
use crate::data::text_features::preprocessing::PreprocessingPipeline;
use crate::data::text_features::methods::{MethodConfig, FeatureMethod};
use crate::data::text_features::methods::tfidf::TfIdfExtractor;
use crate::data::text_features::methods::stats::StatsExtractor;
use super::FeatureExtractor;

/// 基础特征提取器
/// 结合了统计特征和TF-IDF特征
pub struct BasicExtractor {
    config: TextFeatureConfig,
    tfidf_extractor: TfIdfExtractor,
    stats_extractor: StatsExtractor,
    preprocessor: PreprocessingPipeline,
}

impl BasicExtractor {
    /// 创建新的基础特征提取器
    pub fn new(config: TextFeatureConfig) -> Result<Self> {
        let preprocess_config = PreprocessingConfig {
            lowercase: !config.case_sensitive,
            remove_punctuation: config.preprocessing.remove_punctuation,
            remove_numbers: config.preprocessing.remove_numbers,
            remove_stopwords: config.preprocessing.remove_stopwords,
            remove_html: config.preprocessing.remove_html,
            stemming: config.preprocessing.stemming,
            lemmatization: config.preprocessing.lemmatization,
            language: config.language.clone(),
            case_sensitive: config.case_sensitive,
            max_length: config.preprocessing.max_length,
            min_word_length: config.preprocessing.min_word_length,
            max_word_length: config.preprocessing.max_word_length,
            custom_stopwords: config.preprocessing.custom_stopwords.clone(),
            stopwords_path: config.preprocessing.stopwords_path.clone(),
            regex_patterns: config.preprocessing.regex_patterns.clone(),
            replacement_patterns: config.preprocessing.replacement_patterns.clone(),
            preserve_original: config.preprocessing.preserve_original,
            metadata: config.preprocessing.metadata.clone(),
        };
        
        let preprocessor = PreprocessingPipeline::new();
        
        // 创建方法配置
        let tfidf_config = MethodConfig {
            max_features: Some(config.max_features),
            case_sensitive: config.case_sensitive,
            vector_dim: config.feature_dimension,
            aggregation_method: "mean".to_string(),
            normalize: config.use_tfidf,
            params: HashMap::new(),
        };
        
        let stats_config = MethodConfig {
            max_features: None,
            case_sensitive: config.case_sensitive,
            vector_dim: 14, // 统计特征维度
            aggregation_method: "mean".to_string(),
            normalize: false,
            params: HashMap::new(),
        };
        
        let tfidf_extractor = TfIdfExtractor::new(tfidf_config);
        let stats_extractor = StatsExtractor::new(stats_config);
        
        Ok(Self {
            config,
            tfidf_extractor,
            stats_extractor,
            preprocessor,
        })
    }
    
    /// 合并两种类型的特征
    fn merge_features(&self, tfidf_features: Vec<f32>, stats_features: Vec<f32>) -> Vec<f32> {
        let mut merged = Vec::with_capacity(tfidf_features.len() + stats_features.len());
        merged.extend_from_slice(&tfidf_features);
        merged.extend_from_slice(&stats_features);
        merged
    }
}

impl FeatureExtractor for BasicExtractor {
    fn extract(&self, text: &str) -> Result<Vec<f32>> {
        // 预处理文本
        let processed_text = self.preprocessor.process(text)?;
        
        // 空文本处理
        if processed_text.is_empty() {
            return Ok(vec![0.0; self.dimension()]);
        }
        
        // 提取两种特征
        let tfidf_features = self.tfidf_extractor.extract(&processed_text)?;
        let stats_features = self.stats_extractor.extract(&processed_text)?;
        
        // 合并特征
        Ok(self.merge_features(tfidf_features, stats_features))
    }
    
    fn dimension(&self) -> usize {
        self.config.feature_dimension + 14 // TF-IDF维度 + 统计特征维度
    }
    
    fn name(&self) -> &str {
        "basic"
    }
    
    fn from_config(config: &TextFeatureConfig) -> Result<Self> {
        Self::new(config.clone())
    }
}

/// 创建基本特征提取器
pub fn create_basic_extractor(config: &TextFeatureConfig) -> Result<Box<dyn FeatureExtractor>> {
    Ok(Box::new(BasicExtractor::from_config(config)?))
} 