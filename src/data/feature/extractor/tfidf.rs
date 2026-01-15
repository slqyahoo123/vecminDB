// TF-IDF 特征提取器实现
// 提供文本的TF-IDF特征提取功能

use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::RwLock;
// No direct IO here
use serde::{Serialize, Deserialize};
// serde_json not used directly here
use log::{debug, info};
use async_trait::async_trait;

use crate::data::feature::extractor::{
    FeatureExtractor, ExtractorError, InputData, ExtractorContext,
    FeatureVector, FeatureBatch, ExtractorConfig
};
use crate::data::feature::types::{
    ExtractorType, FeatureType, TextExtractorType
};
use crate::core::CoreTensorData;

// 为了向后兼容，创建类型别名
pub type TensorData = CoreTensorData;

/// TF-IDF提取器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TfIdfConfig {
    /// 最大特征维度
    pub max_features: usize,
    /// 最小文档频率
    pub min_df: f32,
    /// 最大文档频率
    pub max_df: f32,
    /// 是否使用IDF
    pub use_idf: bool,
    /// IDF平滑项
    pub smooth_idf: bool,
    /// 是否对IDF取对数
    pub sublinear_tf: bool,
    /// 是否二值化TF
    pub binary: bool,
    /// 停用词列表
    pub stop_words: Option<Vec<String>>,
    /// 是否进行词干提取
    pub use_stemming: bool,
    /// n-gram范围
    pub ngram_range: (usize, usize),
    /// 词汇表大小限制
    pub max_vocabulary_size: Option<usize>,
    /// 归一化方式 ('none', 'l1', 'l2')
    pub norm: String,
    /// 语言
    pub language: String,
    /// 文本分割器
    pub tokenizer: Option<String>,
    /// 是否保留稀疏结果
    pub sparse_output: bool,
}

impl Default for TfIdfConfig {
    fn default() -> Self {
        Self {
            max_features: 10000,
            min_df: 0.0,
            max_df: 1.0,
            use_idf: true,
            smooth_idf: true,
            sublinear_tf: false,
            binary: false,
            stop_words: None,
            use_stemming: false,
            ngram_range: (1, 1),
            max_vocabulary_size: None,
            norm: "l2".to_string(),
            language: "english".to_string(),
            tokenizer: None,
            sparse_output: true,
        }
    }
}

/// TF-IDF提取器的词汇表
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TfIdfVocabulary {
    /// 词到索引的映射
    pub word_to_index: HashMap<String, usize>,
    /// 索引到词的映射
    pub index_to_word: HashMap<usize, String>,
    /// 每个词的文档频率
    pub document_frequency: Vec<usize>,
    /// 总文档数
    pub total_documents: usize,
    /// IDF值
    pub idf: Option<Vec<f32>>,
}

impl TfIdfVocabulary {
    /// 创建新的词汇表
    pub fn new() -> Self {
        Self {
            word_to_index: HashMap::new(),
            index_to_word: HashMap::new(),
            document_frequency: Vec::new(),
            total_documents: 0,
            idf: None,
        }
    }
    
    /// 获取词汇表大小
    pub fn size(&self) -> usize {
        self.word_to_index.len()
    }
    
    /// 添加词汇
    pub fn add_word(&mut self, word: &str) -> usize {
        if let Some(&index) = self.word_to_index.get(word) {
            return index;
        }
        
        let index = self.word_to_index.len();
        self.word_to_index.insert(word.to_string(), index);
        self.index_to_word.insert(index, word.to_string());
        self.document_frequency.push(0);
        
        index
    }
    
    /// 获取词汇索引
    pub fn get_word_index(&self, word: &str) -> Option<usize> {
        self.word_to_index.get(word).copied()
    }
    
    /// 增加词汇的文档频率
    pub fn increment_document_frequency(&mut self, index: usize) {
        if index < self.document_frequency.len() {
            self.document_frequency[index] += 1;
        }
    }
    
    /// 计算IDF值
    pub fn calculate_idf(&mut self, smooth: bool) {
        let n_docs = self.total_documents as f32;
        
        if n_docs == 0.0 {
            self.idf = Some(vec![0.0; self.size()]);
            return;
        }
        
        // 计算IDF值
        let mut idf = Vec::with_capacity(self.size());
        
        for &df in &self.document_frequency {
            let df = df as f32;
            let idf_value = if smooth {
                ((n_docs + 1.0) / (df + 1.0)).ln() + 1.0
            } else {
                (n_docs / df.max(1.0)).ln()
            };
            
            idf.push(idf_value);
        }
        
        self.idf = Some(idf);
    }
    
    /// 获取IDF值
    pub fn get_idf(&self, index: usize) -> Option<f32> {
        self.idf.as_ref().and_then(|idf| idf.get(index).copied())
    }
    
    /// 添加文档
    pub fn add_document(&mut self, document: &str, stop_words: Option<&HashSet<String>>) {
        self.total_documents += 1;
        
        // 分词
        let tokens = tokenize(document);
        
        // 去重，避免多次计算同一个词的频率
        let mut seen = HashSet::new();
        
        for token in tokens {
            // 跳过停用词
            if let Some(stop_words) = stop_words {
                if stop_words.contains(&token) {
                    continue;
                }
            }
            
            // 记录文档频率
            if !seen.contains(&token) {
                let index = self.add_word(&token);
                self.increment_document_frequency(index);
                seen.insert(token);
            }
        }
    }
    
    /// 转换为特征向量
    pub fn transform(&self, document: &str, config: &TfIdfConfig) -> Vec<f32> {
        // 初始化特征向量
        let mut features = vec![0.0; self.size()];
        
        // 如果词汇表为空，返回空向量
        if self.size() == 0 {
            return features;
        }
        
        // 分词
        let tokens = tokenize(document);
        
        // 计算词频
        let mut term_freq = HashMap::new();
        let stop_words = config.stop_words.as_ref().map(|words| {
            words.iter().map(|w| w.to_string()).collect::<HashSet<_>>()
        });
        
        for token in tokens {
            // 跳过停用词
            if let Some(ref stop_words) = stop_words {
                if stop_words.contains(&token) {
                    continue;
                }
            }
            
            // 更新词频
            if let Some(index) = self.get_word_index(&token) {
                *term_freq.entry(index).or_insert(0.0) += 1.0;
            }
        }
        
        // 计算TF-IDF
        for (index, freq) in term_freq {
            // 计算TF
            let tf = if config.binary {
                1.0
            } else if config.sublinear_tf {
                1.0 + (freq as f32).ln()
            } else {
                freq as f32
            };
            
            // 计算IDF
            let idf = if config.use_idf {
                self.get_idf(index).unwrap_or(1.0)
            } else {
                1.0
            };
            
            // 计算TF-IDF
            features[index] = tf * idf;
        }
        
        // 归一化
        match config.norm.as_str() {
            "l1" => normalize_l1(&mut features),
            "l2" => normalize_l2(&mut features),
            _ => {}
        }
        
        features
    }
    
    /// 保留排名前N的特征
    pub fn keep_top_n_features(&mut self, n: usize) {
        if n >= self.size() {
            return;
        }
        
        // 计算每个词的重要性得分（使用文档频率）
        let mut scores: Vec<(usize, usize)> = self.document_frequency
            .iter()
            .enumerate()
            .map(|(idx, &df)| (idx, df))
            .collect();
        
        // 按文档频率排序（降序）
        scores.sort_by(|a, b| b.1.cmp(&a.1));
        
        // 保留前N个特征
        let top_n: HashSet<usize> = scores.iter()
            .take(n)
            .map(|&(idx, _)| idx)
            .collect();
        
        // 更新词汇表
        let mut new_word_to_index = HashMap::new();
        let mut new_index_to_word = HashMap::new();
        let mut new_document_frequency = Vec::with_capacity(n);
        let mut new_idf = self.idf.as_ref().map(|_| Vec::with_capacity(n));
        
        // 创建旧索引到新索引的映射
        let mut old_to_new = HashMap::new();
        let mut new_idx = 0;
        
        for old_idx in top_n {
            if let Some(word) = self.index_to_word.get(&old_idx) {
                new_word_to_index.insert(word.clone(), new_idx);
                new_index_to_word.insert(new_idx, word.clone());
                new_document_frequency.push(self.document_frequency[old_idx]);
                
                if let Some(idf) = &self.idf {
                    if let Some(idf_value) = idf.get(old_idx) {
                        if let Some(ref mut new_idf_vec) = new_idf {
                            new_idf_vec.push(*idf_value);
                        }
                    }
                }
                
                old_to_new.insert(old_idx, new_idx);
                new_idx += 1;
            }
        }
        
        // 更新词汇表
        self.word_to_index = new_word_to_index;
        self.index_to_word = new_index_to_word;
        self.document_frequency = new_document_frequency;
        self.idf = new_idf;
    }
    
    /// 过滤文档频率
    pub fn filter_by_document_frequency(&mut self, min_df: f32, max_df: f32) {
        let n_docs = self.total_documents as f32;
        
        if n_docs == 0.0 {
            return;
        }
        
        // 计算绝对文档频率阈值
        let min_df_abs = if min_df < 1.0 {
            (min_df * n_docs).ceil() as usize
        } else {
            min_df as usize
        };
        
        let max_df_abs = if max_df < 1.0 {
            (max_df * n_docs).floor() as usize
        } else {
            max_df as usize
        };
        
        // 过滤词汇表
        let mut indices_to_keep = HashSet::new();
        
        for (idx, &df) in self.document_frequency.iter().enumerate() {
            if df >= min_df_abs && df <= max_df_abs {
                indices_to_keep.insert(idx);
            }
        }
        
        // 更新词汇表
        let mut new_word_to_index = HashMap::new();
        let mut new_index_to_word = HashMap::new();
        let mut new_document_frequency = Vec::with_capacity(indices_to_keep.len());
        let mut new_idf = self.idf.as_ref().map(|_| Vec::with_capacity(indices_to_keep.len()));
        
        // 创建旧索引到新索引的映射
        let mut old_to_new = HashMap::new();
        let mut new_idx = 0;
        
        for old_idx in indices_to_keep {
            if let Some(word) = self.index_to_word.get(&old_idx) {
                new_word_to_index.insert(word.clone(), new_idx);
                new_index_to_word.insert(new_idx, word.clone());
                new_document_frequency.push(self.document_frequency[old_idx]);
                
                if let Some(idf) = &self.idf {
                    if let Some(idf_value) = idf.get(old_idx) {
                        if let Some(ref mut new_idf_vec) = new_idf {
                            new_idf_vec.push(*idf_value);
                        }
                    }
                }
                
                old_to_new.insert(old_idx, new_idx);
                new_idx += 1;
            }
        }
        
        // 更新词汇表
        self.word_to_index = new_word_to_index;
        self.index_to_word = new_index_to_word;
        self.document_frequency = new_document_frequency;
        self.idf = new_idf;
    }
}

impl Default for TfIdfVocabulary {
    fn default() -> Self {
        Self::new()
    }
}

/// TF-IDF特征提取器
pub struct TfIdfExtractor {
    /// 提取器配置
    config: ExtractorConfig,
    /// TF-IDF配置
    tfidf_config: TfIdfConfig,
    /// 是否已训练
    trained: bool,
    /// 词汇表
    vocabulary: RwLock<TfIdfVocabulary>,
    /// 停用词集合
    stop_words: Option<HashSet<String>>,
}

impl TfIdfExtractor {
    /// 创建新的TF-IDF提取器
    pub fn new(config: ExtractorConfig) -> Self {
        // 解析TF-IDF特定配置
        let tfidf_config = if let Some(Ok(config_value)) = config.get_serialized_param::<TfIdfConfig>("tfidf_config") {
            config_value
        } else {
            TfIdfConfig::default()
        };
        
        // 解析停用词
        let stop_words = tfidf_config.stop_words.as_ref().map(|words| {
            words.iter().map(|w| w.to_string()).collect::<HashSet<_>>()
        });
        
        Self {
            config,
            tfidf_config,
            trained: false,
            vocabulary: RwLock::new(TfIdfVocabulary::new()),
            stop_words,
        }
    }
    
    /// 训练TF-IDF模型
    pub fn fit(&self, documents: &[String]) -> std::result::Result<(), ExtractorError> {
        let mut vocabulary = self.vocabulary.write().map_err(|e| {
            ExtractorError::Internal(format!("获取词汇表写锁失败: {}", e))
        })?;
        
        // 重置词汇表
        *vocabulary = TfIdfVocabulary::new();
        
        // 添加文档到词汇表
        for document in documents {
            vocabulary.add_document(document, self.stop_words.as_ref());
        }
        
        debug!("初始词汇表大小: {}", vocabulary.size());
        
        // 过滤文档频率
        vocabulary.filter_by_document_frequency(
            self.tfidf_config.min_df,
            self.tfidf_config.max_df
        );
        
        debug!("过滤后词汇表大小: {}", vocabulary.size());
        
        // 保留前N个特征
        if let Some(max_size) = self.tfidf_config.max_vocabulary_size {
            vocabulary.keep_top_n_features(max_size);
            debug!("限制后词汇表大小: {}", vocabulary.size());
        }
        
        // 计算IDF
        vocabulary.calculate_idf(self.tfidf_config.smooth_idf);
        
        debug!("TF-IDF模型训练完成，词汇表大小: {}", vocabulary.size());
        
        Ok(())
    }
    
    /// 转换文档为TF-IDF特征向量
    pub fn transform(&self, document: &str) -> std::result::Result<Vec<f32>, ExtractorError> {
        let vocabulary = self.vocabulary.read().map_err(|e| {
            ExtractorError::Internal(format!("获取词汇表读锁失败: {}", e))
        })?;
        
        if !self.trained && vocabulary.size() == 0 {
            return Err(ExtractorError::Internal(
                "TF-IDF提取器未训练".to_string()
            ));
        }
        
        let features = vocabulary.transform(document, &self.tfidf_config);
        Ok(features)
    }
    
    /// 批量转换文档为TF-IDF特征向量
    pub fn transform_batch(&self, documents: &[String]) -> std::result::Result<Vec<Vec<f32>>, ExtractorError> {
        let mut result = Vec::with_capacity(documents.len());
        
        for document in documents {
            result.push(self.transform(document)?);
        }
        
        Ok(result)
    }
}

impl Debug for TfIdfExtractor {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("TfIdfExtractor")
            .field("config", &self.config)
            .field("tfidf_config", &self.tfidf_config)
            .field("trained", &self.trained)
            .field("vocabulary_size", &self.vocabulary.read().map(|v| v.size()).unwrap_or(0))
            .finish()
    }
}

#[async_trait]
impl FeatureExtractor for TfIdfExtractor {
    fn extractor_type(&self) -> ExtractorType {
        ExtractorType::Text(TextExtractorType::TfIdf)
    }
    
    fn config(&self) -> &ExtractorConfig {
        &self.config
    }
    
    fn is_compatible(&self, input: &InputData) -> bool {
        match input {
            InputData::Text(_) => true,
            InputData::TextArray(_) => true,
            _ => false,
        }
    }
    
    async fn extract(&self, input: InputData, context: Option<ExtractorContext>) -> std::result::Result<FeatureVector, ExtractorError> {
        // 处理输入
        let document = match input {
            InputData::Text(text) => text,
            InputData::Binary(bytes) => {
                String::from_utf8(bytes).map_err(|e| {
                    ExtractorError::InputData(format!("无法将二进制数据转换为文本: {}", e))
                })?
            },
            _ => return Err(ExtractorError::InputData(
                format!("不支持的输入类型: {:?}", input.type_name())
            )),
        };
        
        // 是否需要训练
        if let Some(ctx) = &context {
            if let Some(train_documents_str) = ctx.get_param("train_documents") {
                // 尝试从JSON字符串解析文档列表
                if let Ok(train_documents) = serde_json::from_str::<Vec<String>>(train_documents_str) {
                    info!("使用 {} 个文档训练TF-IDF模型", train_documents.len());
                    self.fit(&train_documents)?;
                }
            }
        }
        
        // 提取特征
        let features = self.transform(&document)?;
        
        // 创建特征向量
        let result = FeatureVector {
            feature_type: FeatureType::Text,
            features: Vec::new(),
            values: features,
            extractor_type: Some(self.extractor_type()),
            metadata: HashMap::new(),
        };
        
        Ok(result)
    }
    
    async fn batch_extract(&self, inputs: Vec<InputData>, context: Option<ExtractorContext>) -> std::result::Result<FeatureBatch, ExtractorError> {
        // 处理批量输入
        let mut documents = Vec::with_capacity(inputs.len());
        
        for input in inputs {
            let document = match input {
                InputData::Text(text) => text,
                InputData::Binary(bytes) => {
                    String::from_utf8(bytes).map_err(|e| {
                        ExtractorError::InputData(format!("无法将二进制数据转换为文本: {}", e))
                    })?
                },
                _ => return Err(ExtractorError::InputData(
                    format!("不支持的输入类型: {:?}", input.type_name())
                )),
            };
            
            documents.push(document);
        }
        
        // 是否需要训练
        if let Some(ctx) = &context {
            if let Some(train_documents_str) = ctx.get_param("train_documents") {
                // 尝试从JSON字符串解析文档列表
                if let Ok(train_documents) = serde_json::from_str::<Vec<String>>(train_documents_str) {
                    info!("使用 {} 个文档训练TF-IDF模型", train_documents.len());
                    self.fit(&train_documents)?;
                }
            }
        }
        
        // 批量提取特征
        let features = self.transform_batch(&documents)?;
        
        // 计算维度
        let dimension = if features.is_empty() { 0 } else { features[0].len() };
        
        // 创建特征批次
        let result = FeatureBatch {
            feature_type: FeatureType::Text,
            values: features,
            extractor_type: Some(self.extractor_type()),
            batch_size: documents.len(),
            dimension,
            metadata: HashMap::new(),
        };
        
        Ok(result)
    }
    
    fn output_feature_type(&self) -> FeatureType {
        FeatureType::Text
    }
    
    fn output_dimension(&self) -> Option<usize> {
        self.vocabulary.read().ok().map(|v| v.size())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// 创建TF-IDF特征提取器
pub fn create_tfidf_extractor(config: ExtractorConfig) -> std::result::Result<TfIdfExtractor, ExtractorError> {
    // 检查配置是否匹配
    if !matches!(config.extractor_type, ExtractorType::Text(TextExtractorType::TfIdf)) {
        return Err(ExtractorError::Config(format!(
            "提取器类型不匹配: 期望 TfIdf, 实际 {:?}", config.extractor_type
        )));
    }
    
    Ok(TfIdfExtractor::new(config))
}

/// 简单的分词函数
fn tokenize(text: &str) -> Vec<String> {
    // 在实际应用中，这里应该使用更复杂的分词器
    // 这里仅做简单处理：转换为小写、移除标点符号、按空格分割
    let lowercase = text.to_lowercase();
    let without_punctuation = lowercase
        .chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .collect::<String>();
    
    without_punctuation
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

/// L1归一化
fn normalize_l1(features: &mut [f32]) {
    let norm: f32 = features.iter().map(|&x| x.abs()).sum();
    
    if norm > 0.0 {
        for feature in features.iter_mut() {
            *feature /= norm;
        }
    }
}

/// L2归一化
fn normalize_l2(features: &mut [f32]) {
    let norm: f32 = features.iter().map(|&x| x * x).sum::<f32>().sqrt();
    
    if norm > 0.0 {
        for feature in features.iter_mut() {
            *feature /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tokenize() {
        let text = "Hello, world! This is a test.";
        let tokens = tokenize(text);
        assert_eq!(tokens, vec!["hello", "world", "this", "is", "a", "test"]);
    }
    
    #[test]
    fn test_normalize_l1() {
        let mut features = vec![1.0, 2.0, 3.0];
        normalize_l1(&mut features);
        
        let sum: f32 = features.iter().map(|&x| x.abs()).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_normalize_l2() {
        let mut features = vec![1.0, 2.0, 3.0];
        normalize_l2(&mut features);
        
        let sum: f32 = features.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((sum - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_vocabulary() {
        let mut vocabulary = TfIdfVocabulary::new();
        
        vocabulary.add_document("hello world", None);
        vocabulary.add_document("hello rust", None);
        vocabulary.add_document("hello world rust", None);
        
        assert_eq!(vocabulary.size(), 3);
        assert_eq!(vocabulary.total_documents, 3);
        
        // 检查文档频率
        assert_eq!(vocabulary.document_frequency[vocabulary.word_to_index["hello"]], 3);
        assert_eq!(vocabulary.document_frequency[vocabulary.word_to_index["world"]], 2);
        assert_eq!(vocabulary.document_frequency[vocabulary.word_to_index["rust"]], 2);
        
        // 计算IDF
        vocabulary.calculate_idf(true);
        
        // 检查IDF值
        let hello_idf = vocabulary.get_idf(vocabulary.word_to_index["hello"]).unwrap();
        let world_idf = vocabulary.get_idf(vocabulary.word_to_index["world"]).unwrap();
        let rust_idf = vocabulary.get_idf(vocabulary.word_to_index["rust"]).unwrap();
        
        // hello出现在所有文档中，IDF应该最小
        assert!(hello_idf < world_idf);
        assert!(hello_idf < rust_idf);
    }
    
    #[tokio::test]
    async fn test_tfidf_extractor() {
        // 创建配置
        let mut config = ExtractorConfig::new(ExtractorType::Text(TextExtractorType::TfIdf));
        let tfidf_config = TfIdfConfig::default();
        config = config.with_param("tfidf_config", tfidf_config).unwrap();
        
        // 创建提取器
        let extractor = TfIdfExtractor::new(config);
        
        // 准备训练数据
        let documents = vec![
            "This is the first document.".to_string(),
            "This document is the second document.".to_string(),
            "And this is the third one.".to_string(),
            "Is this the first document?".to_string(),
        ];
        
        // 训练
        assert!(extractor.fit(&documents).is_ok());
        
        // 提取特征
        let features = extractor.transform("This is the first document.").unwrap();
        
        // 特征维度应该等于词汇表大小
        assert_eq!(features.len(), extractor.vocabulary.read().unwrap().size());
        
        // 批量提取特征
        let batch_features = extractor.transform_batch(&documents).unwrap();
        
        // 批量特征数量应该等于文档数量
        assert_eq!(batch_features.len(), documents.len());
        
        // 每个特征向量的维度应该一致
        for features in &batch_features {
            assert_eq!(features.len(), extractor.vocabulary.read().unwrap().size());
        }
        
        // 测试extract方法
        let input = InputData::Text("This is a test.".to_string());
        let context = None;
        let result = extractor.extract(input, context).await;
        
        assert!(result.is_ok());
        let feature_vector = result.unwrap();
        assert_eq!(feature_vector.values.len(), extractor.vocabulary.read().unwrap().size());
    }
} 