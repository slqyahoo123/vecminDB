use crate::Result;
use crate::data::text_features::methods::{FeatureMethod, MethodConfig};
use crate::data::text_features::preprocessing::TextPreprocessor;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use rayon::prelude::*;

/// 词频特征提取器
pub struct WordFrequencyExtractor {
    /// 方法配置
    config: MethodConfig,
    /// 词汇表
    vocabulary: HashMap<String, usize>,
    /// 预处理器
    preprocessor: Option<Box<dyn TextPreprocessor>>,
    /// 缓存
    cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
}

impl Clone for WordFrequencyExtractor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            vocabulary: self.vocabulary.clone(),
            preprocessor: None, // 不能克隆 trait 对象，设置为 None
            cache: Arc::clone(&self.cache),
        }
    }
}

impl WordFrequencyExtractor {
    /// 创建新的词频提取器
    pub fn new(config: MethodConfig) -> Self {
        Self {
            config,
            vocabulary: HashMap::new(),
            preprocessor: None,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// 设置预处理器
    pub fn with_preprocessor(mut self, preprocessor: Box<dyn TextPreprocessor>) -> Self {
        self.preprocessor = Some(preprocessor);
        self
    }
    
    /// 构建词汇表
    pub fn fit(&mut self, corpus: &[String]) -> Result<()> {
        self.vocabulary.clear();
        
        let mut word_counts = HashMap::new();
        
        // 统计词频
        for document in corpus {
            let processed_text = if let Some(preprocessor) = &self.preprocessor {
                preprocessor.preprocess(document)?
            } else {
                document.to_string()
            };
            
            let tokens = self.tokenize(&processed_text);
            
            for token in tokens {
                *word_counts.entry(token).or_insert(0) += 1;
            }
        }
        
        // 构建词汇表，选择最常见的 max_features 个词
        let mut word_count_pairs: Vec<_> = word_counts.into_iter().collect();
        word_count_pairs.sort_by(|a, b| b.1.cmp(&a.1));
        
        let max_features = self.config.max_features.unwrap_or(word_count_pairs.len()).min(word_count_pairs.len());
        
        for (i, (word, _)) in word_count_pairs.into_iter().take(max_features).enumerate() {
            self.vocabulary.insert(word, i);
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
    
    /// 计算词频向量
    fn compute_frequency(&self, text: &str) -> Result<Vec<f32>> {
        let processed_text = if let Some(preprocessor) = &self.preprocessor {
            preprocessor.preprocess(text)?
        } else {
            text.to_string()
        };
        
        let tokens = self.tokenize(&processed_text);
        
        // 计算词频
        let mut token_counts = HashMap::new();
        for token in tokens {
            *token_counts.entry(token).or_insert(0) += 1;
        }
        
        let total_tokens = token_counts.values().sum::<usize>().max(1) as f32;
        
        // 创建词频向量
        let mut frequency = vec![0.0; self.vocabulary.len()];
        for (token, count) in token_counts {
            if let Some(&index) = self.vocabulary.get(&token) {
                frequency[index] = count as f32 / total_tokens;
            }
        }
        
        // 根据配置决定是否要归一化
        if self.config.params.get("normalize").map_or(true, |v| v == "true") {
            let norm: f32 = frequency.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for value in frequency.iter_mut() {
                    *value /= norm;
                }
            }
        }
        
        Ok(frequency)
    }
}

impl FeatureMethod for WordFrequencyExtractor {
    /// 提取特征
    fn extract(&self, text: &str) -> Result<Vec<f32>> {
        // 检查缓存
        if let Some(cached) = self.cache.read().unwrap().get(text) {
            return Ok(cached.clone());
        }
        
        let features = self.compute_frequency(text)?;
        
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
        "词频特征"
    }
    
    /// 获取方法配置
    fn config(&self) -> &MethodConfig {
        &self.config
    }
    
    /// 重置状态
    fn reset(&mut self) {
        self.vocabulary.clear();
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
    
    /// 分词方法
    fn tokenize_text(&self, text: &str) -> Result<Vec<String>> {
        let processed_text = if let Some(preprocessor) = &self.preprocessor {
            preprocessor.preprocess(text)?
        } else {
            text.to_string()
        };
        Ok(self.tokenize(&processed_text))
    }
    
    /// 获取词嵌入（词频特征不使用词嵌入，返回错误）
    fn get_word_embeddings(&self, _tokens: &[String]) -> Result<Vec<Vec<f32>>> {
        Err(crate::error::Error::not_implemented("词频特征不支持词嵌入"))
    }
    
    /// 查找词嵌入（词频特征不使用词嵌入，返回错误）
    fn lookup_word_embedding(&self, _word: &str) -> Result<Vec<f32>> {
        Err(crate::error::Error::not_implemented("词频特征不支持词嵌入"))
    }
    
    /// 获取未知词嵌入（词频特征不使用词嵌入，返回错误）
    fn get_unknown_word_embedding(&self) -> Result<Vec<f32>> {
        Err(crate::error::Error::not_implemented("词频特征不支持词嵌入"))
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