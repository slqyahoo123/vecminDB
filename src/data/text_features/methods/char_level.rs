use crate::Result;
use crate::data::text_features::methods::{FeatureMethod, MethodConfig};
use crate::data::text_features::preprocessing::TextPreprocessor;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use rayon::prelude::*;

/// 字符级特征提取器
pub struct CharLevelExtractor {
    /// 方法配置
    config: MethodConfig,
    /// 字符表
    char_vocabulary: HashMap<char, usize>,
    /// 预处理器
    preprocessor: Option<Box<dyn TextPreprocessor>>,
    /// 缓存
    cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
}

impl Clone for CharLevelExtractor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            char_vocabulary: self.char_vocabulary.clone(),
            preprocessor: None, // 不能克隆 trait 对象，设置为 None
            cache: Arc::clone(&self.cache),
        }
    }
}

impl CharLevelExtractor {
    /// 创建新的字符级特征提取器
    pub fn new(config: MethodConfig) -> Self {
        Self {
            config,
            char_vocabulary: HashMap::new(),
            preprocessor: None,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// 设置预处理器
    pub fn with_preprocessor(mut self, preprocessor: Box<dyn TextPreprocessor>) -> Self {
        self.preprocessor = Some(preprocessor);
        self
    }
    
    /// 从语料库构建字符表
    pub fn fit(&mut self, corpus: &[String]) -> Result<()> {
        self.char_vocabulary.clear();
        
        let mut char_counts = HashMap::new();
        
        // 统计字符频率
        for document in corpus {
            let processed_text = if let Some(preprocessor) = &self.preprocessor {
                preprocessor.preprocess(document)?
            } else {
                document.to_string()
            };
            
            for ch in processed_text.chars() {
                if !self.config.case_sensitive && ch.is_alphabetic() {
                    *char_counts.entry(ch.to_lowercase().next().unwrap()).or_insert(0) += 1;
                } else {
                    *char_counts.entry(ch).or_insert(0) += 1;
                }
            }
        }
        
        // 构建字符表，选择最常见的 max_features 个字符
        let mut char_count_pairs: Vec<_> = char_counts.into_iter().collect();
        char_count_pairs.sort_by(|a, b| b.1.cmp(&a.1));
        
        let max_features = self.config.max_features.unwrap_or(char_count_pairs.len()).min(char_count_pairs.len());
        
        for (i, (ch, _)) in char_count_pairs.into_iter().take(max_features).enumerate() {
            self.char_vocabulary.insert(ch, i);
        }
        
        // 清除缓存
        self.cache.write().unwrap().clear();
        
        Ok(())
    }
    
    /// 计算字符级特征向量
    fn compute_char_features(&self, text: &str) -> Result<Vec<f32>> {
        let processed_text = if let Some(preprocessor) = &self.preprocessor {
            preprocessor.preprocess(text)?
        } else {
            text.to_string()
        };
        
        let text_len = processed_text.chars().count().max(1) as f32;
        let mut char_counts = HashMap::new();
        
        // 计算字符频率
        for ch in processed_text.chars() {
            let ch = if !self.config.case_sensitive && ch.is_alphabetic() {
                ch.to_lowercase().next().unwrap()
            } else {
                ch
            };
            
            *char_counts.entry(ch).or_insert(0) += 1;
        }
        
        // 创建特征向量
        let mut features = vec![0.0; self.char_vocabulary.len()];
        for (ch, count) in char_counts {
            if let Some(&index) = self.char_vocabulary.get(&ch) {
                features[index] = count as f32 / text_len;
            }
        }
        
        // 根据聚合方法处理
        if self.config.aggregation_method == "sum" {
            // 已经是总和形式
        } else if self.config.aggregation_method == "mean" {
            // 已经是平均形式 (频率)
        } else if self.config.aggregation_method == "max" {
            // 对于字符级特征，max 不太适用，但我们可以实现为最大出现频率
            let max_value = features.iter().cloned().fold(0.0, f32::max);
            if max_value > 0.0 {
                for value in features.iter_mut() {
                    if *value > 0.0 {
                        *value = *value / max_value;
                    }
                }
            }
        }
        
        // 归一化
        if self.config.params.get("normalize").map_or(true, |v| v == "true") {
            let norm: f32 = features.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for value in features.iter_mut() {
                    *value /= norm;
                }
            }
        }
        
        Ok(features)
    }
}

impl FeatureMethod for CharLevelExtractor {
    /// 提取特征
    fn extract(&self, text: &str) -> Result<Vec<f32>> {
        // 检查缓存
        if let Some(cached) = self.cache.read().unwrap().get(text) {
            return Ok(cached.clone());
        }
        
        let features = self.compute_char_features(text)?;
        
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
        "字符级特征"
    }
    
    /// 获取方法配置
    fn config(&self) -> &MethodConfig {
        &self.config
    }
    
    /// 重置状态
    fn reset(&mut self) {
        self.char_vocabulary.clear();
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
    
    /// 分词方法（字符级特征不需要分词，返回字符列表）
    fn tokenize_text(&self, text: &str) -> Result<Vec<String>> {
        let chars: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        Ok(chars)
    }
    
    /// 获取词嵌入（字符级特征不使用词嵌入，返回错误）
    fn get_word_embeddings(&self, _tokens: &[String]) -> Result<Vec<Vec<f32>>> {
        Err(crate::error::Error::not_implemented("字符级特征不支持词嵌入"))
    }
    
    /// 查找词嵌入（字符级特征不使用词嵌入，返回错误）
    fn lookup_word_embedding(&self, _word: &str) -> Result<Vec<f32>> {
        Err(crate::error::Error::not_implemented("字符级特征不支持词嵌入"))
    }
    
    /// 获取未知词嵌入（字符级特征不使用词嵌入，返回错误）
    fn get_unknown_word_embedding(&self) -> Result<Vec<f32>> {
        Err(crate::error::Error::not_implemented("字符级特征不支持词嵌入"))
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