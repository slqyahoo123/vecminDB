// src/data/text_features/features.rs
//
// Transformer 特征提取模块

use std::collections::HashMap;
use log::info;
use serde::{Serialize, Deserialize};

// crate::Error not referenced here
use super::error::TransformerError;
// use super::config::TransformerConfig;
use super::encoder::Encoder;

/// 特征提取配置
#[derive(Debug, Clone)]
pub struct FeatureExtractionConfig {
    /// 是否提取词频特征
    pub extract_word_frequency: bool,
    /// 是否提取TF-IDF特征
    pub extract_tfidf: bool,
    /// 是否提取n-gram特征
    pub extract_ngrams: bool,
    /// n-gram范围
    pub ngram_range: (usize, usize),
    /// 是否提取字符级特征
    pub extract_char_features: bool,
    /// 是否提取语义特征
    pub extract_semantic_features: bool,
    /// 特征维度
    pub feature_dimension: usize,
    /// 是否归一化特征
    pub normalize_features: bool,
    /// 是否使用特征选择
    pub use_feature_selection: bool,
    /// 最大特征数
    pub max_features: usize,
}

impl Default for FeatureExtractionConfig {
    fn default() -> Self {
        Self {
            extract_word_frequency: true,
            extract_tfidf: true,
            extract_ngrams: false,
            ngram_range: (1, 2),
            extract_char_features: false,
            extract_semantic_features: true,
            feature_dimension: 768,
            normalize_features: true,
            use_feature_selection: false,
            max_features: 1000,
        }
    }
}

/// 特征向量
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// 特征值
    pub values: Vec<f32>,
    /// 特征名称
    pub names: Vec<String>,
    /// 特征类型
    pub feature_types: Vec<FeatureType>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

impl FeatureVector {
    /// 创建新的特征向量
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
            names: Vec::new(),
            feature_types: Vec::new(),
            metadata: HashMap::new(),
        }
    }
    
    /// 添加特征
    pub fn add_feature(&mut self, name: String, value: f32, feature_type: FeatureType) {
        self.values.push(value);
        self.names.push(name);
        self.feature_types.push(feature_type);
    }
    
    /// 获取特征维度
    pub fn dimension(&self) -> usize {
        self.values.len()
    }
    
    /// 归一化特征
    pub fn normalize(&mut self) -> Result<(), TransformerError> {
        if self.values.is_empty() {
            return Err(TransformerError::InputError("特征向量为空".to_string()));
        }
        
        let mean = self.values.iter().sum::<f32>() / self.values.len() as f32;
        let variance = self.values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / self.values.len() as f32;
        let std_dev = (variance + 1e-8).sqrt();
        
        for value in &mut self.values {
            *value = (*value - mean) / std_dev;
        }
        
        Ok(())
    }
    
    /// 转换为向量
    pub fn to_vec(&self) -> Vec<f32> {
        self.values.clone()
    }
    
    /// 从向量创建
    pub fn from_vec(values: Vec<f32>, names: Vec<String>, feature_types: Vec<FeatureType>) -> Self {
        Self {
            values,
            names,
            feature_types,
            metadata: HashMap::new(),
        }
    }
}

/// 特征类型
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeatureType {
    /// 词频特征
    WordFrequency,
    /// TF-IDF特征
    TfIdf,
    /// N-gram特征
    Ngram,
    /// 字符特征
    Character,
    /// 语义特征
    Semantic,
    /// 统计特征
    Statistical,
    /// 自定义特征
    Custom(String),
}

/// 特征提取器
#[derive(Debug)]
pub struct FeatureExtractor {
    /// 配置
    config: FeatureExtractionConfig,
    /// 编码器
    encoder: Option<Encoder>,
    /// 词汇表
    vocabulary: HashMap<String, usize>,
    /// 文档频率
    document_frequency: HashMap<String, usize>,
    /// 总文档数
    total_documents: usize,
    /// 特征选择器
    feature_selector: Option<FeatureSelector>,
}

impl FeatureExtractor {
    /// 创建新的特征提取器
    pub fn new(config: FeatureExtractionConfig) -> Self {
        Self {
            config,
            encoder: None,
            vocabulary: HashMap::new(),
            document_frequency: HashMap::new(),
            total_documents: 0,
            feature_selector: None,
        }
    }
    
    /// 设置编码器
    pub fn set_encoder(&mut self, encoder: Encoder) {
        self.encoder = Some(encoder);
    }
    
    /// 训练特征提取器
    pub fn fit(&mut self, documents: &[String]) -> Result<(), TransformerError> {
        if documents.is_empty() {
            return Err(TransformerError::InputError("训练文档为空".to_string()));
        }
        
        self.total_documents = documents.len();
        
        // 构建词汇表
        self.build_vocabulary(documents)?;
        
        // 计算文档频率
        self.calculate_document_frequency(documents)?;
        
        // 初始化特征选择器
        if self.config.use_feature_selection {
            self.feature_selector = Some(FeatureSelector::new(self.config.max_features));
        }
        
        Ok(())
    }
    
    /// 构建词汇表
    fn build_vocabulary(&mut self, documents: &[String]) -> Result<(), TransformerError> {
        let mut word_counts = HashMap::new();
        
        for document in documents {
            let tokens = self.tokenize_document(document);
            for token in tokens {
                *word_counts.entry(token).or_insert(0) += 1;
            }
        }
        
        // 按频率排序并选择top词汇
        let mut sorted_words: Vec<_> = word_counts.into_iter().collect();
        sorted_words.sort_by(|a, b| b.1.cmp(&a.1));
        
        for (i, (word, _)) in sorted_words.into_iter().take(self.config.max_features).enumerate() {
            self.vocabulary.insert(word, i);
        }
        
        info!("词汇表构建完成，大小: {}", self.vocabulary.len());
        Ok(())
    }
    
    /// 计算文档频率
    fn calculate_document_frequency(&mut self, documents: &[String]) -> Result<(), TransformerError> {
        for document in documents {
            let tokens = self.tokenize_document(document);
            let unique_tokens: std::collections::HashSet<_> = tokens.into_iter().collect();
            
            for token in unique_tokens {
                if self.vocabulary.contains_key(&token) {
                    *self.document_frequency.entry(token).or_insert(0) += 1;
                }
            }
        }
        
        Ok(())
    }
    
    /// 分词文档
    fn tokenize_document(&self, document: &str) -> Vec<String> {
        // 简单的分词实现
        document.to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect()
    }
    
    /// 提取特征
    pub fn extract_features(&self, text: &str) -> Result<FeatureVector, TransformerError> {
        let mut features = FeatureVector::new();
        
        // 提取词频特征
        if self.config.extract_word_frequency {
            self.extract_word_frequency_features(text, &mut features)?;
        }
        
        // 提取TF-IDF特征
        if self.config.extract_tfidf {
            self.extract_tfidf_features(text, &mut features)?;
        }
        
        // 提取n-gram特征
        if self.config.extract_ngrams {
            self.extract_ngram_features(text, &mut features)?;
        }
        
        // 提取字符级特征
        if self.config.extract_char_features {
            self.extract_character_features(text, &mut features)?;
        }
        
        // 提取语义特征
        if self.config.extract_semantic_features {
            self.extract_semantic_features(text, &mut features)?;
        }
        
        // 归一化特征
        if self.config.normalize_features {
            features.normalize()?;
        }
        
        // 特征选择
        if let Some(selector) = &self.feature_selector {
            selector.select_features(&mut features)?;
        }
        
        Ok(features)
    }
    
    /// 提取词频特征
    fn extract_word_frequency_features(&self, text: &str, features: &mut FeatureVector) -> Result<(), TransformerError> {
        let tokens = self.tokenize_document(text);
        let mut word_counts = HashMap::new();
        
        for token in tokens {
            *word_counts.entry(token).or_insert(0) += 1;
        }
        
        for (word, &id) in &self.vocabulary {
            let count = word_counts.get(word).unwrap_or(&0);
            let frequency = *count as f32 / tokens.len() as f32;
            
            features.add_feature(
                format!("word_freq_{}", word),
                frequency,
                FeatureType::WordFrequency
            );
        }
        
        Ok(())
    }
    
    /// 提取TF-IDF特征
    fn extract_tfidf_features(&self, text: &str, features: &mut FeatureVector) -> Result<(), TransformerError> {
        let tokens = self.tokenize_document(text);
        let mut word_counts = HashMap::new();
        
        for token in tokens {
            *word_counts.entry(token).or_insert(0) += 1;
        }
        
        for (word, &id) in &self.vocabulary {
            let tf = *word_counts.get(word).unwrap_or(&0) as f32;
            let df = *self.document_frequency.get(word).unwrap_or(&1) as f32;
            let idf = (self.total_documents as f32 / df).ln();
            let tfidf = tf * idf;
            
            features.add_feature(
                format!("tfidf_{}", word),
                tfidf,
                FeatureType::TfIdf
            );
        }
        
        Ok(())
    }
    
    /// 提取n-gram特征
    fn extract_ngram_features(&self, text: &str, features: &mut FeatureVector) -> Result<(), TransformerError> {
        let tokens = self.tokenize_document(text);
        let (min_n, max_n) = self.config.ngram_range;
        
        for n in min_n..=max_n {
            if tokens.len() < n {
                continue;
            }
            
            for i in 0..=tokens.len() - n {
                let ngram: Vec<_> = tokens[i..i+n].iter().cloned().collect();
                let ngram_str = ngram.join(" ");
                
                // 计算n-gram频率
                let mut ngram_count = 0;
                for j in 0..=tokens.len() - n {
                    let current_ngram: Vec<_> = tokens[j..j+n].iter().cloned().collect();
                    if current_ngram == ngram {
                        ngram_count += 1;
                    }
                }
                
                let frequency = ngram_count as f32 / (tokens.len() - n + 1) as f32;
                
                features.add_feature(
                    format!("ngram_{}_{}", n, ngram_str),
                    frequency,
                    FeatureType::Ngram
                );
            }
        }
        
        Ok(())
    }
    
    /// 提取字符级特征
    fn extract_character_features(&self, text: &str, features: &mut FeatureVector) -> Result<(), TransformerError> {
        let chars: Vec<char> = text.chars().collect();
        
        // 字符频率
        let mut char_counts = HashMap::new();
        for &ch in &chars {
            if ch.is_alphabetic() {
                *char_counts.entry(ch.to_lowercase().next().unwrap()).or_insert(0) += 1;
            }
        }
        
        // 添加字符频率特征
        for ch in 'a'..='z' {
            let count = char_counts.get(&ch).unwrap_or(&0);
            let frequency = *count as f32 / chars.len() as f32;
            
            features.add_feature(
                format!("char_freq_{}", ch),
                frequency,
                FeatureType::Character
            );
        }
        
        // 文本长度特征
        features.add_feature(
            "text_length".to_string(),
            text.len() as f32,
            FeatureType::Statistical
        );
        
        // 单词数量特征
        let word_count = text.split_whitespace().count();
        features.add_feature(
            "word_count".to_string(),
            word_count as f32,
            FeatureType::Statistical
        );
        
        // 平均单词长度
        let words: Vec<_> = text.split_whitespace().collect();
        let total_length: usize = words.iter().map(|w| w.len()).sum();
        let avg_word_length = if words.is_empty() { 0.0 } else { total_length as f32 / words.len() as f32 };
        
        features.add_feature(
            "avg_word_length".to_string(),
            avg_word_length,
            FeatureType::Statistical
        );
        
        Ok(())
    }
    
    /// 提取语义特征
    fn extract_semantic_features(&self, text: &str, features: &mut FeatureVector) -> Result<(), TransformerError> {
        if let Some(encoder) = &self.encoder {
            // 使用编码器提取语义特征
            let semantic_features = encoder.encode(text)?;
            
            // 添加语义特征
            for (i, &value) in semantic_features.iter().enumerate() {
                features.add_feature(
                    format!("semantic_{}", i),
                    value,
                    FeatureType::Semantic
                );
            }
        } else {
            // 如果没有编码器，使用简单的统计特征
            let tokens = self.tokenize_document(text);
            
            // 词汇多样性
            let unique_tokens: std::collections::HashSet<_> = tokens.iter().cloned().collect();
            let diversity = unique_tokens.len() as f32 / tokens.len() as f32;
            
            features.add_feature(
                "vocabulary_diversity".to_string(),
                diversity,
                FeatureType::Statistical
            );
            
            // 句子复杂度（基于标点符号）
            let punctuation_count = text.chars().filter(|&c| ".,;:!?".contains(c)).count();
            let complexity = punctuation_count as f32 / text.len() as f32;
            
            features.add_feature(
                "sentence_complexity".to_string(),
                complexity,
                FeatureType::Statistical
            );
        }
        
        Ok(())
    }
    
    /// 获取配置
    pub fn config(&self) -> &FeatureExtractionConfig {
        &self.config
    }
    
    /// 获取词汇表大小
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }
}

/// 特征选择器
#[derive(Debug)]
pub struct FeatureSelector {
    /// 最大特征数
    max_features: usize,
    /// 特征重要性
    feature_importance: HashMap<String, f32>,
}

impl FeatureSelector {
    /// 创建新的特征选择器
    pub fn new(max_features: usize) -> Self {
        Self {
            max_features,
            feature_importance: HashMap::new(),
        }
    }
    
    /// 计算特征重要性
    pub fn calculate_importance(&mut self, features: &[FeatureVector], labels: &[i32]) -> Result<(), TransformerError> {
        if features.is_empty() || features.len() != labels.len() {
            return Err(TransformerError::InputError("特征和标签数量不匹配".to_string()));
        }
        
        let feature_count = features[0].dimension();
        
        // 计算每个特征的重要性（使用方差作为简单的重要性指标）
        for feature_idx in 0..feature_count {
            let feature_name = &features[0].names[feature_idx];
            let values: Vec<f32> = features.iter().map(|f| f.values[feature_idx]).collect();
            
            // 计算方差
            let mean = values.iter().sum::<f32>() / values.len() as f32;
            let variance = values.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / values.len() as f32;
            
            self.feature_importance.insert(feature_name.clone(), variance);
        }
        
        Ok(())
    }
    
    /// 选择特征
    pub fn select_features(&self, features: &mut FeatureVector) -> Result<(), TransformerError> {
        if self.feature_importance.is_empty() {
            return Ok(()); // 如果没有重要性信息，保留所有特征
        }
        
        // 按重要性排序特征
        let mut feature_indices: Vec<_> = (0..features.dimension()).collect();
        feature_indices.sort_by(|&a, &b| {
            let importance_a = self.feature_importance.get(&features.names[a]).unwrap_or(&0.0);
            let importance_b = self.feature_importance.get(&features.names[b]).unwrap_or(&0.0);
            importance_b.partial_cmp(importance_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // 选择top特征
        let selected_count = std::cmp::min(self.max_features, feature_indices.len());
        let selected_indices = &feature_indices[..selected_count];
        
        // 重新构建特征向量
        let mut new_values = Vec::new();
        let mut new_names = Vec::new();
        let mut new_types = Vec::new();
        
        for &idx in selected_indices {
            new_values.push(features.values[idx]);
            new_names.push(features.names[idx].clone());
            new_types.push(features.feature_types[idx].clone());
        }
        
        features.values = new_values;
        features.names = new_names;
        features.feature_types = new_types;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_vector_creation() {
        let mut features = FeatureVector::empty();
        features.add_feature("test_feature".to_string(), 1.0, FeatureType::WordFrequency);
        
        assert_eq!(features.dimension(), 1);
        assert_eq!(features.values[0], 1.0);
    }

    #[test]
    fn test_feature_extractor_creation() {
        let config = FeatureExtractionConfig::default();
        let extractor = FeatureExtractor::new(config);
        
        assert_eq!(extractor.vocabulary_size(), 0);
    }

    #[test]
    fn test_feature_extraction() {
        let config = FeatureExtractionConfig::default();
        let mut extractor = FeatureExtractor::new(config);
        
        let documents = vec![
            "Hello world".to_string(),
            "Hello universe".to_string(),
        ];
        
        extractor.fit(&documents).unwrap();
        
        let features = extractor.extract_features("Hello world").unwrap();
        assert!(features.dimension() > 0);
    }

    #[test]
    fn test_feature_normalization() {
        let mut features = FeatureVector::empty();
        features.add_feature("f1".to_string(), 1.0, FeatureType::Statistical);
        features.add_feature("f2".to_string(), 2.0, FeatureType::Statistical);
        features.add_feature("f3".to_string(), 3.0, FeatureType::Statistical);
        
        features.normalize().unwrap();
        
        // 检查归一化后的均值为0，标准差为1
        let mean = features.values.iter().sum::<f32>() / features.values.len() as f32;
        assert!((mean.abs() < 1e-6));
    }
} 

/// 文本特征结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextFeatures {
    /// 特征向量
    pub features: HashMap<String, f32>,
    /// 特征维度
    pub dimension: usize,
    /// 特征类型
    pub feature_type: FeatureType,
    /// 元数据
    pub metadata: HashMap<String, String>,
    /// 提取时间戳
    pub timestamp: u64,
    /// 文本长度
    pub text_length: usize,
    /// 词汇表大小
    pub vocabulary_size: usize,
    /// 特征提取方法
    pub extraction_method: String,
    /// 预处理配置
    pub preprocessing_config: Option<String>,
    /// 质量分数
    pub quality_score: Option<f32>,
}

impl TextFeatures {
    /// 创建新的文本特征
    pub fn new() -> Self {
        Self {
            features: HashMap::new(),
            dimension: 0,
            feature_type: FeatureType::Statistical,
            metadata: HashMap::new(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            text_length: 0,
            vocabulary_size: 0,
            extraction_method: String::new(),
            preprocessing_config: None,
            quality_score: None,
        }
    }

    /// 从特征向量创建文本特征
    pub fn from_feature_vector(vector: FeatureVector, text_length: usize, vocabulary_size: usize) -> Self {
        let mut features = HashMap::new();
        for (name, value) in vector.names.iter().zip(vector.values.iter()) {
            features.insert(name.clone(), *value);
        }

        Self {
            features,
            dimension: vector.dimension(),
            feature_type: FeatureType::Statistical,
            metadata: vector.metadata,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            text_length,
            vocabulary_size,
            extraction_method: String::new(),
            preprocessing_config: None,
            quality_score: None,
        }
    }

    /// 添加特征
    pub fn add_feature(&mut self, name: String, value: f32) {
        self.features.insert(name, value);
        self.dimension = self.features.len();
    }

    /// 获取特征值
    pub fn get_feature(&self, name: &str) -> Option<&f32> {
        self.features.get(name)
    }

    /// 获取所有特征值
    pub fn get_features(&self) -> &HashMap<String, f32> {
        &self.features
    }

    /// 转换为向量
    pub fn to_vector(&self) -> Vec<f32> {
        self.features.values().cloned().collect()
    }

    /// 计算特征统计信息
    pub fn calculate_statistics(&self) -> HashMap<String, f32> {
        let values: Vec<f32> = self.features.values().cloned().collect();
        if values.is_empty() {
            return HashMap::new();
        }

        let sum: f32 = values.iter().sum();
        let mean = sum / values.len() as f32;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();
        let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let mut stats = HashMap::new();
        stats.insert("count".to_string(), values.len() as f32);
        stats.insert("sum".to_string(), sum);
        stats.insert("mean".to_string(), mean);
        stats.insert("variance".to_string(), variance);
        stats.insert("std_dev".to_string(), std_dev);
        stats.insert("min".to_string(), min);
        stats.insert("max".to_string(), max);

        stats
    }

    /// 归一化特征
    pub fn normalize(&mut self) -> Result<(), TransformerError> {
        if self.features.is_empty() {
            return Ok(());
        }

        let values: Vec<f32> = self.features.values().cloned().collect();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();

        if std_dev < 1e-8 {
            return Ok(());
        }

        for value in self.features.values_mut() {
            *value = (*value - mean) / std_dev;
        }

        Ok(())
    }

    /// 设置质量分数
    pub fn set_quality_score(&mut self, score: f32) {
        self.quality_score = Some(score);
    }

    /// 获取质量分数
    pub fn get_quality_score(&self) -> Option<f32> {
        self.quality_score
    }

    /// 设置提取方法
    pub fn set_extraction_method(&mut self, method: String) {
        self.extraction_method = method;
    }

    /// 设置预处理配置
    pub fn set_preprocessing_config(&mut self, config: String) {
        self.preprocessing_config = Some(config);
    }

    /// 合并另一个文本特征
    pub fn merge(&mut self, other: &TextFeatures) {
        for (key, value) in &other.features {
            self.features.insert(key.clone(), *value);
        }
        self.dimension = self.features.len();
        
        // 合并元数据
        for (key, value) in &other.metadata {
            self.metadata.insert(key.clone(), value.clone());
        }
    }

    /// 验证特征质量
    pub fn validate(&self) -> Result<bool, TransformerError> {
        if self.features.is_empty() {
            return Ok(false);
        }

        // 检查是否有NaN或无穷大值
        for value in self.features.values() {
            if value.is_nan() || value.is_infinite() {
                return Ok(false);
            }
        }

        // 检查特征维度
        if self.dimension == 0 {
            return Ok(false);
        }

        Ok(true)
    }
}

impl Default for TextFeatures {
    fn default() -> Self {
        Self::new()
    }
} 