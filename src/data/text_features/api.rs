use crate::Result;
use crate::Error;
use serde_json::Value;
use std::collections::HashMap;
// unused concurrent/async imports removed; API here is sync facade
use log::debug;

use super::config::TextFeatureConfig;
use super::extractors::{create_extractor, MixedFeatureExtractor, FeatureExtractor};
use super::types::TextFeatureMethod;
use crate::vector::utils::vector;
// remove unused data batch/format and processed batch imports
use crate::core::CoreTensorData;

// 为了向后兼容，创建类型别名
pub type TensorData = CoreTensorData;

/// 提取文本特征
/// 
/// 使用默认配置从文本中提取特征
/// 
/// # 参数
/// * `text` - 输入文本
/// 
/// # 返回
/// * `Result<Vec<f32>>` - 特征向量
pub fn extract_features(text: &str) -> Result<Vec<f32>> {
    debug!("text_features.api.extract_features: len={}", text.len());
    let config = TextFeatureConfig::default();
    let extractor = create_extractor(&config)?;
    extractor.extract(text)
}

/// 提取批量文本特征
/// 
/// 使用默认配置从多个文本中提取特征
/// 
/// # 参数
/// * `texts` - 输入文本列表
/// 
/// # 返回
/// * `Result<Vec<Vec<f32>>>` - 特征向量列表
pub fn extract_batch_features(texts: &[String]) -> Result<Vec<Vec<f32>>> {
    debug!("text_features.api.extract_batch_features: batch={}", texts.len());
    let config = TextFeatureConfig::default();
    let extractor = create_extractor(&config)?;
    
    let mut results = Vec::with_capacity(texts.len());
    for text in texts {
        results.push(extractor.extract(text)?);
    }
    
    Ok(results)
}

/// 使用指定方法提取特征
/// 
/// # 参数
/// * `text` - 输入文本
/// * `method` - 特征提取方法
/// 
/// # 返回
/// * `Result<Vec<f32>>` - 特征向量
pub fn extract_features_with_method(text: &str, method: TextFeatureMethod) -> Result<Vec<f32>> {
    debug!("text_features.api.extract_features_with_method: method={:?}", method);
    let config = TextFeatureConfig {
        method: method.into(),
        ..TextFeatureConfig::default()
    };
    
    let extractor = create_extractor(&config)?;
    extractor.extract(text)
}

/// 提取JSON数据中的特征
/// 
/// # 参数
/// * `json` - JSON数据
/// * `config` - 配置信息
/// 
/// # 返回
/// * `Result<Vec<f32>>` - 特征向量
pub fn extract_features_from_json(json: &Value, config: &TextFeatureConfig) -> Result<Vec<f32>> {
    debug!("text_features.api.extract_features_from_json");
    // 对于JSON数据，使用MixedFeatureExtractor
    let mixed_config = super::config::MixedFeatureConfig {
        text_config: config.clone(),
        numeric_config: super::config::NumericFeatureConfig::default(),
        categorical_config: super::config::CategoricalFeatureConfig::default(),
        fusion_strategy: super::config::FusionStrategy::Concatenation,
        normalize: true,
        feature_selection: false,
        selection_method: "variance".to_string(),
        selected_feature_count: 100,
        metadata: HashMap::new(),
    };
    
    let extractor = MixedFeatureExtractor::new(mixed_config)?;
    
    // 将JSON转换为字符串
    let text = json.to_string();
    extractor.extract(&text)
}

/// 比较两个文本的相似度
/// 
/// # 参数
/// * `text1` - 第一个文本
/// * `text2` - 第二个文本
/// * `method` - 特征提取方法
/// 
/// # 返回
/// * `Result<f32>` - 相似度 (0-1)
pub fn compare_texts(text1: &str, text2: &str, method: TextFeatureMethod) -> Result<f32> {
    debug!("text_features.api.compare_texts: method={:?}", method);
    let config = TextFeatureConfig {
        method: method.into(),
        ..TextFeatureConfig::default()
    };
    
    let extractor = create_extractor(&config)?;
    
    let features1 = extractor.extract(text1)?;
    let features2 = extractor.extract(text2)?;
    
    // 计算余弦相似度
    vector::feature_similarity(&features1, &features2)
        .map_err(|e| Error::data(format!("计算相似度失败: {}", e)))
}

/// 创建特征提取器配置
/// 
/// # 参数
/// * `method` - 特征提取方法
/// * `max_features` - 最大特征数
/// * `min_df` - 最小文档频率
/// 
/// # 返回
/// * `TextFeatureConfig` - 配置
pub fn create_config(method: TextFeatureMethod, max_features: Option<usize>, min_df: Option<f64>) -> TextFeatureConfig {
    TextFeatureConfig {
        method,
        max_features: max_features.unwrap_or(1000),
        min_df: min_df.unwrap_or(1.0),
        ..TextFeatureConfig::default()
    }
}

/// 创建默认的文本特征配置
pub fn default_text_feature_config() -> TextFeatureConfig {
    TextFeatureConfig::default()
}

/// 处理文本
/// 
/// # 参数
/// * `text` - 输入文本
/// * `config` - 配置信息
/// 
/// # 返回
/// * `Result<Vec<f32>>` - 特征向量
pub fn process_text(text: &str, config: &TextFeatureConfig) -> Result<Vec<f32>> {
    debug!("text_features.api.process_text");
    let extractor = create_extractor(config)?;
    extractor.extract(text)
}

/// 批量处理文本
/// 
/// # 参数
/// * `texts` - 输入文本列表
/// * `config` - 配置信息
/// 
/// # 返回
/// * `Result<Vec<Vec<f32>>>` - 特征向量列表
pub fn batch_process_text(texts: &[String], config: &TextFeatureConfig) -> Result<Vec<Vec<f32>>> {
    debug!("text_features.api.batch_process_text: batch={}", texts.len());
    let extractor = create_extractor(config)?;
    
    let mut results = Vec::with_capacity(texts.len());
    for text in texts {
        results.push(extractor.extract(text)?);
    }
    
    Ok(results)
}

/// 处理JSON数据
/// 
/// # 参数
/// * `data` - JSON数据列表
/// * `field` - 要处理的字段名
/// * `config` - 配置信息
/// 
/// # 返回
/// * `Result<Vec<Vec<f32>>>` - 特征向量列表
pub fn process_json_data(data: &[Value], field: &str, config: &TextFeatureConfig) -> Result<Vec<Vec<f32>>> {
    debug!("text_features.api.process_json_data: field={}", field);
    let extractor = create_extractor(config)?;
    
    let mut results = Vec::with_capacity(data.len());
    for item in data {
        if let Some(value) = item.get(field) {
            if let Some(text) = value.as_str() {
                results.push(extractor.extract(text)?);
            }
        }
    }
    
    Ok(results)
} 