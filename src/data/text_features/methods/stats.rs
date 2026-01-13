// Statistical Feature Extraction Method
// 实现基于统计的特征提取

use crate::Error;
use crate::data::text_features::methods::{FeatureMethod, MethodConfig};
use crate::data::text_features::preprocessing::TextPreprocessor;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use rayon::prelude::*;

/// 统计特征提取器
pub struct StatsExtractor {
    /// 方法配置
    config: MethodConfig,
    /// 预处理器
    preprocessor: Option<Box<dyn TextPreprocessor>>,
    /// 缓存
    cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
}

impl Clone for StatsExtractor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            preprocessor: None, // 不能克隆 trait 对象，设置为 None
            cache: Arc::clone(&self.cache),
        }
    }
}

impl StatsExtractor {
    /// 创建新的统计特征提取器
    pub fn new(config: MethodConfig) -> Self {
        Self {
            config,
            preprocessor: None,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// 设置预处理器
    pub fn with_preprocessor(mut self, preprocessor: Box<dyn TextPreprocessor>) -> Self {
        self.preprocessor = Some(preprocessor);
        self
    }
    
    /// 预处理文本
    fn preprocess_text(&self, text: &str) -> Result<String, Error> {
        if let Some(preprocessor) = &self.preprocessor {
            preprocessor.preprocess(text)
        } else {
            Ok(text.to_string())
        }
    }
    
    /// 计算文本统计特征
    fn compute_stats(&self, text: &str) -> Result<Vec<f32>, Error> {
        let processed_text = self.preprocess_text(text)?;
        
        if processed_text.is_empty() {
            return Ok(vec![0.0; 10]); // 返回一个全零的默认特征向量
        }
        
        // 计算基本统计特征
        let char_count = processed_text.chars().count() as f32;
        let word_count = processed_text.split_whitespace().count() as f32;
        let sentence_count = processed_text.split(|c| c == '.' || c == '?' || c == '!').filter(|s| !s.trim().is_empty()).count() as f32;
        let line_count = processed_text.lines().count() as f32;
        
        let words: Vec<&str> = processed_text.split_whitespace().collect();
        let _avg_word_length = if word_count > 0.0 {
            words.iter().map(|w| w.chars().count() as f32).sum::<f32>() / word_count
        } else {
            0.0
        };
        
        // 计算词频
        let mut word_freq = HashMap::new();
        for word in &words {
            *word_freq.entry(word.to_lowercase()).or_insert(0) += 1;
        }
        
        // 唯一词比例
        let unique_word_ratio = if word_count > 0.0 {
            word_freq.len() as f32 / word_count
        } else {
            0.0
        };
        
        // 计算句子长度的统计信息
        let sentences: Vec<&str> = processed_text
            .split(|c| c == '.' || c == '?' || c == '!')
            .filter(|s| !s.trim().is_empty())
            .collect();
            
        let avg_sentence_length = if sentence_count > 0.0 {
            sentences.iter().map(|s| s.split_whitespace().count() as f32).sum::<f32>() / sentence_count
        } else {
            0.0
        };
        
        // 计算停用词比例
        let stopwords = self.get_stopwords();
        let stopword_count = words
            .iter()
            .filter(|&w| stopwords.contains(&w.to_lowercase()))
            .count() as f32;
        let stopword_ratio = if word_count > 0.0 {
            stopword_count / word_count
        } else {
            0.0
        };
        
        // 计算符号比例
        let punctuation_count = processed_text
            .chars()
            .filter(|c| c.is_ascii_punctuation())
            .count() as f32;
        let punctuation_ratio = if char_count > 0.0 {
            punctuation_count / char_count
        } else {
            0.0
        };
        
        // 计算大写字母比例
        let uppercase_count = processed_text
            .chars()
            .filter(|c| c.is_uppercase())
            .count() as f32;
        let uppercase_ratio = if char_count > 0.0 {
            uppercase_count / char_count
        } else {
            0.0
        };
        
        // 计算数字比例
        let digit_count = processed_text
            .chars()
            .filter(|c| c.is_digit(10))
            .count() as f32;
        let digit_ratio = if char_count > 0.0 {
            digit_count / char_count
        } else {
            0.0
        };
        
        // 计算词性多样性（简化版）
        // 完整实现需要POS标注器
        let word_diversity = unique_word_ratio;
        
        // 计算词长分布
        let word_lengths: Vec<usize> = words
            .iter()
            .map(|w| w.chars().count())
            .collect();
            
        let avg_word_length = if !word_lengths.is_empty() {
            word_lengths.iter().sum::<usize>() as f32 / word_lengths.len() as f32
        } else {
            0.0
        };
        
        // 计算词长标准差
        let word_length_variance = if !word_lengths.is_empty() {
            let mean = avg_word_length;
            let variance = word_lengths
                .iter()
                .map(|&l| {
                    let diff = l as f32 - mean;
                    diff * diff
                })
                .sum::<f32>() / word_lengths.len() as f32;
            variance.sqrt()
        } else {
            0.0
        };
        
        // 计算特殊字符比例
        let special_chars_count = processed_text
            .chars()
            .filter(|&c| {
                !c.is_alphanumeric() && !c.is_whitespace() && !c.is_ascii_punctuation()
            })
            .count() as f32;
        let special_chars_ratio = if char_count > 0.0 {
            special_chars_count / char_count
        } else {
            0.0
        };
        
        // 返回一组统计特征
        let features = vec![
            char_count / 1000.0,  // 归一化字符数
            word_count / 100.0,   // 归一化词数
            sentence_count / 10.0, // 归一化句子数
            line_count / 10.0,     // 归一化行数
            avg_word_length,      // 平均词长
            unique_word_ratio,    // 唯一词比例
            avg_sentence_length / 20.0, // 归一化平均句子长度
            stopword_ratio,       // 停用词比例
            punctuation_ratio,    // 标点符号比例
            uppercase_ratio,      // 大写字母比例
            digit_ratio,          // 数字比例
            word_diversity,       // 词多样性
            word_length_variance, // 词长标准差
            special_chars_ratio,  // 特殊字符比例
        ];
        
        Ok(features)
    }
    
    /// 获取停用词列表
    fn get_stopwords(&self) -> HashSet<String> {
        // 简单的英文停用词列表
        let stopwords = vec![
            "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "shall", "should", "can", "could", "may", "might",
            "must", "to", "of", "in", "on", "at", "by", "for", "with", "about",
            "against", "between", "into", "through", "during", "before", "after",
            "above", "below", "from", "up", "down", "out", "off", "over", "under",
            "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "any", "both", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "s", "t", "just", "don", "now",
            // 中文停用词
            "的", "了", "和", "是", "在", "我", "有", "这", "个", "们",
            "中", "也", "就", "来", "到", "你", "说", "为", "着", "如",
            "那", "要", "自", "以", "会", "对", "可", "她", "里", "所",
            "他", "而", "么", "去", "之", "于", "把", "等", "被", "一",
            "没", "什", "麼", "这个", "那个", "只是", "因为", "所以", "但是", "如果",
        ];
        
        stopwords.into_iter().map(|s| s.to_string()).collect()
    }
}

impl FeatureMethod for StatsExtractor {
    /// 提取特征
    fn extract(&self, text: &str) -> Result<Vec<f32>, Error> {
        // 检查缓存
        if let Some(cached) = self.cache.read().unwrap().get(text) {
            return Ok(cached.clone());
        }
        
        let features = self.compute_stats(text)?;
        
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
        "Statistical"
    }
    
    /// 获取方法配置
    fn config(&self) -> &MethodConfig {
        &self.config
    }
    
    /// 重置状态
    fn reset(&mut self) {
        self.cache.write().unwrap().clear();
    }
    
    /// 批量提取特征
    fn batch_extract(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, Error> {
        let use_parallel = texts.len() > 50;  // 简单的并行处理阈值
        
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
    fn tokenize_text(&self, text: &str) -> Result<Vec<String>, Error> {
        let processed_text = self.preprocess_text(text)?;
        let tokens: Vec<String> = processed_text
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        Ok(tokens)
    }
    
    /// 获取词嵌入（统计特征不使用词嵌入，返回错误）
    fn get_word_embeddings(&self, _tokens: &[String]) -> Result<Vec<Vec<f32>>, Error> {
        Err(crate::error::Error::not_implemented("统计特征不支持词嵌入"))
    }
    
    /// 查找词嵌入（统计特征不使用词嵌入，返回错误）
    fn lookup_word_embedding(&self, _word: &str) -> Result<Vec<f32>, Error> {
        Err(crate::error::Error::not_implemented("统计特征不支持词嵌入"))
    }
    
    /// 获取未知词嵌入（统计特征不使用词嵌入，返回错误）
    fn get_unknown_word_embedding(&self) -> Result<Vec<f32>, Error> {
        Err(crate::error::Error::not_implemented("统计特征不支持词嵌入"))
    }
    
    /// 简单哈希函数
    fn simple_hash(&self, s: &str) -> Result<u64, Error> {
        let mut hash = 5381u64;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }
        Ok(hash)
    }
} 