//! 文本特征提取器工厂模块
//! 提供创建不同类型文本特征提取器的工厂方法

use crate::data::text_features::{TextFeatureMethod, TextFeatureConfig};
use crate::data::text_features::methods::{FeatureMethod, MethodConfig};
use crate::data::text_features::extractors::bert::BertExtractor;
use crate::data::text_features::extractors::FeatureExtractor;
use super::tfidf::TfIdfExtractor;
use super::word2vec::Word2VecExtractor;
use crate::Error;
use crate::Result;
use std::collections::HashMap;

/// BertExtractor 适配器，用于实现 FeatureMethod trait
struct BertExtractorAdapter {
    extractor: BertExtractor,
    config: MethodConfig,
}

impl FeatureMethod for BertExtractorAdapter {
    fn extract(&self, text: &str) -> Result<Vec<f32>> {
        self.extractor.extract(text)
    }
    
    fn name(&self) -> &str {
        "BertExtractor"
    }
    
    fn config(&self) -> &MethodConfig {
        &self.config
    }
    
    fn reset(&mut self) {
        // BertExtractor 没有公开的 reset 方法，这里清空缓存
        // 注意：这需要 BertExtractor 提供公开的方法
    }
    
    fn batch_extract(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.extractor.extract(text)?);
        }
        Ok(results)
    }
    
    fn tokenize_text(&self, text: &str) -> Result<Vec<String>> {
        // 简单的分词实现
        Ok(text.split_whitespace().map(|s| s.to_string()).collect())
    }
    
    fn get_word_embeddings(&self, words: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(words.len());
        for word in words {
            // 使用 extract 方法获取单词的嵌入
            embeddings.push(self.extractor.extract(word)?);
        }
        Ok(embeddings)
    }
    
    fn lookup_word_embedding(&self, word: &str) -> Result<Vec<f32>> {
        self.extractor.extract(word)
    }
    
    fn get_unknown_word_embedding(&self) -> Result<Vec<f32>> {
        // 返回零向量作为未知词嵌入
        Ok(vec![0.0; self.config.vector_dim])
    }
    
    fn simple_hash(&self, text: &str) -> Result<u64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        Ok(hasher.finish())
    }
}

/// 从 TextFeatureConfig 创建 MethodConfig
fn config_to_method_config(config: &TextFeatureConfig) -> MethodConfig {
    let mut params = HashMap::new();
    params.insert("min_df".to_string(), config.min_df.to_string());
    params.insert("max_df".to_string(), config.max_df.to_string());
    params.insert("use_idf".to_string(), config.use_idf.to_string());
    params.insert("smooth_idf".to_string(), config.smooth_idf.to_string());
    
    // 从元数据中获取normalize，如果没有则默认为false
    let normalize = config.metadata
        .get("normalize")
        .and_then(|v| v.parse::<bool>().ok())
        .unwrap_or(false);
    
    MethodConfig {
        max_features: Some(config.max_features),
        case_sensitive: config.preprocessing.case_sensitive,
        vector_dim: config.feature_dimension,
        aggregation_method: "mean".to_string(), // 默认聚合方法
        normalize,
        params,
    }
}

/// 创建文本特征提取器
/// 
/// 根据指定的方法和配置创建对应的特征提取器
pub fn create_extractor(method: TextFeatureMethod, config: TextFeatureConfig) -> Result<Box<dyn FeatureMethod>> {
    let method_config = config_to_method_config(&config);
    
    match method {
        TextFeatureMethod::TfIdf => {
            let extractor = TfIdfExtractor::new(method_config);
            Ok(Box::new(extractor))
        },
        TextFeatureMethod::Word2Vec => {
            let extractor = Word2VecExtractor::new(method_config);
            Ok(Box::new(extractor))
        },
        TextFeatureMethod::Bert => {
            let extractor = BertExtractor::new(config.clone())?;
            let adapter = BertExtractorAdapter {
                extractor,
                config: method_config,
            };
            Ok(Box::new(adapter))
        },
        _ => Err(Error::invalid_argument(&format!("不支持的特征提取方法: {:?}", method)))
    }
}

/// 获取所有支持的特征提取方法
pub fn get_supported_methods() -> Vec<TextFeatureMethod> {
    vec![
        TextFeatureMethod::TfIdf,
        TextFeatureMethod::Word2Vec,
        TextFeatureMethod::Bert,
    ]
}

/// 检查特征提取方法是否支持
pub fn is_method_supported(method: TextFeatureMethod) -> bool {
    match method {
        TextFeatureMethod::TfIdf | 
        TextFeatureMethod::Word2Vec | 
        TextFeatureMethod::Bert => true,
        _ => false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_extractor() {
        let config = TextFeatureConfig::default();
        
        // 测试创建TfIdf提取器
        let extractor = create_extractor(TextFeatureMethod::TfIdf, config.clone());
        assert!(extractor.is_ok());
        
        // 测试创建Word2Vec提取器
        let extractor = create_extractor(TextFeatureMethod::Word2Vec, config.clone());
        assert!(extractor.is_ok());
        
        // 测试创建Bert提取器
        let extractor = create_extractor(TextFeatureMethod::Bert, config.clone());
        assert!(extractor.is_ok());
    }
    
    #[test]
    fn test_supported_methods() {
        let methods = get_supported_methods();
        assert!(methods.contains(&TextFeatureMethod::TfIdf));
        assert!(methods.contains(&TextFeatureMethod::Word2Vec));
        assert!(methods.contains(&TextFeatureMethod::Bert));
    }
    
    #[test]
    fn test_is_method_supported() {
        assert!(is_method_supported(TextFeatureMethod::TfIdf));
        assert!(is_method_supported(TextFeatureMethod::Word2Vec));
        assert!(is_method_supported(TextFeatureMethod::Bert));
    }
} 