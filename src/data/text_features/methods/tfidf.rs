// TF-IDF Feature Extraction Method
// 实现基于TF-IDF的特征提取

use crate::Result;
use crate::data::text_features::methods::{FeatureMethod, MethodConfig};
use crate::data::text_features::preprocessing::TextPreprocessor;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use rayon::prelude::*;

/// TF-IDF向量化器配置
pub struct TfidfVectorizerConfig {
    /// 是否使用IDF权重
    pub use_idf: bool,
    /// 是否对IDF权重进行平滑处理
    pub smooth_idf: bool,
    /// 是否对文档向量进行归一化
    pub normalize: bool,
    /// 最小文档频率
    pub min_df: usize,
    /// 最大文档频率
    pub max_df: f64,
    /// 最大特征数量
    pub max_features: Option<usize>,
    /// 预处理器，可选
    pub preprocessor: Option<Box<dyn TextPreprocessor>>,
    /// 是否使用二元特征
    pub binary: bool,
    /// N-gram范围，例如(1,2)表示使用一元和二元词组
    pub ngram_range: (usize, usize),
    /// 停用词列表
    pub stop_words: Option<HashSet<String>>,
}

impl Clone for TfidfVectorizerConfig {
    fn clone(&self) -> Self {
        Self {
            use_idf: self.use_idf,
            smooth_idf: self.smooth_idf,
            normalize: self.normalize,
            min_df: self.min_df,
            max_df: self.max_df,
            max_features: self.max_features,
            preprocessor: None, // 不能克隆 trait 对象，设置为 None
            binary: self.binary,
            ngram_range: self.ngram_range,
            stop_words: self.stop_words.clone(),
        }
    }
}

impl std::fmt::Debug for TfidfVectorizerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TfidfVectorizerConfig")
            .field("use_idf", &self.use_idf)
            .field("smooth_idf", &self.smooth_idf)
            .field("normalize", &self.normalize)
            .field("min_df", &self.min_df)
            .field("max_df", &self.max_df)
            .field("max_features", &self.max_features)
            .field("preprocessor", &self.preprocessor.as_ref().map(|_| "Some(TextPreprocessor)"))
            .field("binary", &self.binary)
            .field("ngram_range", &self.ngram_range)
            .field("stop_words", &self.stop_words)
            .finish()
    }
}

/// TF-IDF特征提取器
pub struct TfIdfExtractor {
    /// 方法配置
    config: MethodConfig,
    /// 词汇表
    vocabulary: HashMap<String, usize>,
    /// 文档频率
    document_frequency: HashMap<String, usize>,
    /// 文档总数
    document_count: usize,
    /// 预处理器
    preprocessor: Option<Box<dyn TextPreprocessor>>,
    /// 缓存
    cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
}

impl Clone for TfIdfExtractor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            vocabulary: self.vocabulary.clone(),
            document_frequency: self.document_frequency.clone(),
            document_count: self.document_count,
            preprocessor: None, // 不能克隆 trait 对象，设置为 None
            cache: Arc::clone(&self.cache),
        }
    }
}

impl TfIdfExtractor {
    /// 创建新的TF-IDF提取器
    pub fn new(config: MethodConfig) -> Self {
        Self {
            config,
            vocabulary: HashMap::new(),
            document_frequency: HashMap::new(),
            document_count: 0,
            preprocessor: None,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// 设置预处理器
    pub fn with_preprocessor(mut self, preprocessor: Box<dyn TextPreprocessor>) -> Self {
        self.preprocessor = Some(preprocessor);
        self
    }
    
    /// 从语料库构建词汇表和文档频率
    pub fn fit(&mut self, corpus: &[String]) -> Result<()> {
        self.vocabulary.clear();
        self.document_frequency.clear();
        self.document_count = corpus.len();
        
        let mut word_counts = HashMap::new();
        let mut document_frequency = HashMap::new();
        
        // 统计词频和文档频率
        for document in corpus {
            let processed_text = if let Some(preprocessor) = &self.preprocessor {
                preprocessor.preprocess(document)?
            } else {
                document.to_string()
            };
            
            let tokens = self.tokenize(&processed_text);
            let mut unique_tokens = HashSet::new();
            
            for token in &tokens {
                *word_counts.entry(token.clone()).or_insert(0) += 1;
                unique_tokens.insert(token.clone());
            }
            
            for token in unique_tokens {
                *document_frequency.entry(token).or_insert(0) += 1;
            }
        }
        
        // 构建词汇表，选择最常见的 max_features 个词
        let mut word_count_pairs: Vec<_> = word_counts.into_iter().collect();
        word_count_pairs.sort_by(|a, b| b.1.cmp(&a.1));
        
        let max_features = self.config.max_features.unwrap_or(word_count_pairs.len()).min(word_count_pairs.len());
        
        for (i, (word, _)) in word_count_pairs.into_iter().take(max_features).enumerate() {
            self.vocabulary.insert(word, i);
        }
        
        // 更新文档频率
        for (word, count) in document_frequency {
            if self.vocabulary.contains_key(&word) {
                self.document_frequency.insert(word, count);
            }
        }
        
        // 清除缓存
        self.cache.write().unwrap().clear();
        
        Ok(())
    }
    
    /// 分词
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|token| {
                if self.config.case_sensitive {
                    token.to_string()
                } else {
                    token.to_lowercase()
                }
            })
            .collect()
    }
    
    /// 计算TF-IDF向量
    fn compute_tfidf(&self, text: &str) -> Result<Vec<f32>> {
        let processed_text = if let Some(preprocessor) = &self.preprocessor {
            preprocessor.preprocess(text)?
        } else {
            text.to_string()
        };
        
        let tokens = self.tokenize(&processed_text);
        
        // 计算词频
        let mut term_frequency = vec![0.0; self.vocabulary.len()];
        for token in tokens {
            if let Some(&index) = self.vocabulary.get(&token) {
                term_frequency[index] += 1.0;
            }
        }
        
        // 计算TF-IDF
        let mut tfidf = vec![0.0; self.vocabulary.len()];
        for (token, &index) in &self.vocabulary {
            if let Some(&df) = self.document_frequency.get(token) {
                let tf: f32 = term_frequency[index];
                let idf = (self.document_count as f32 / (df as f32 + 1.0)).ln() + 1.0;
                
                if self.config.params.get("sublinear_tf").map_or(false, |v| v == "true") {
                    // 次线性TF
                    if tf > 0.0 {
                        tfidf[index] = (1.0 + tf.ln()) * idf;
                    }
                } else {
                    tfidf[index] = tf * idf;
                }
            }
        }
        
        // 归一化
        let norm: f32 = tfidf.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in tfidf.iter_mut() {
                *value /= norm;
            }
        }
        
        Ok(tfidf)
    }
}

impl FeatureMethod for TfIdfExtractor {
    /// 提取特征
    fn extract(&self, text: &str) -> Result<Vec<f32>> {
        // 检查缓存
        if let Some(cached) = self.cache.read().unwrap().get(text) {
            return Ok(cached.clone());
        }
        
        let features = self.compute_tfidf(text)?;
        
        // 更新缓存
        let mut cache = self.cache.write().unwrap();
        if cache.len() > 1000 {  // 简单的缓存限制
            cache.clear();
        }
        cache.insert(text.to_string(), features.clone());
        
        Ok(features)
    }
    
    /// 获取方法名称
    fn name(&self) -> &str {
        "TF-IDF"
    }
    
    /// 获取方法配置
    fn config(&self) -> &MethodConfig {
        &self.config
    }
    
    /// 重置状态
    fn reset(&mut self) {
        self.vocabulary.clear();
        self.document_frequency.clear();
        self.document_count = 0;
        self.cache.write().unwrap().clear();
    }
    
    /// 批量提取特征
    fn batch_extract(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let use_parallel = texts.len() > 100;  // 简单的并行处理阈值
        
        if use_parallel {
            // 并行处理
            texts.par_iter()
                .map(|text| self.extract(text))
                .collect()
        } else {
            // 顺序处理
            texts.iter()
                .map(|text| self.extract(text))
                .collect()
        }
    }
    
    /// 分词方法（对外接口）
    fn tokenize_text(&self, text: &str) -> Result<Vec<String>> {
        Ok(self.tokenize(text))
    }
    
    /// 获取词嵌入（TF-IDF不使用词嵌入，返回错误）
    fn get_word_embeddings(&self, _tokens: &[String]) -> Result<Vec<Vec<f32>>> {
        Err(crate::error::Error::not_implemented("TF-IDF不支持词嵌入"))
    }
    
    /// 查找词嵌入（TF-IDF不使用词嵌入，返回错误）
    fn lookup_word_embedding(&self, _word: &str) -> Result<Vec<f32>> {
        Err(crate::error::Error::not_implemented("TF-IDF不支持词嵌入"))
    }
    
    /// 获取未知词嵌入（TF-IDF不使用词嵌入，返回错误）
    fn get_unknown_word_embedding(&self) -> Result<Vec<f32>> {
        Err(crate::error::Error::not_implemented("TF-IDF不支持词嵌入"))
    }
    
    /// 简单哈希函数
    fn simple_hash(&self, s: &str) -> Result<u64> {
        let mut hash = 5381u64;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }
        Ok(hash)
    }
} 