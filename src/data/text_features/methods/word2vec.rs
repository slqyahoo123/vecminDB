// Word2Vec Feature Extraction Method
// 实现基于Word2Vec的特征提取

use crate::Error;
use crate::data::text_features::methods::{FeatureMethod, MethodConfig};
use crate::data::text_features::preprocessing::TextPreprocessor;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
// use ndarray::{Array1, Array2}; // not used in current simplified aggregation path
use rayon::prelude::*;

/// Word2Vec 特征提取器
pub struct Word2VecExtractor {
    /// 方法配置
    config: MethodConfig,
    /// 词向量映射
    word_vectors: HashMap<String, Vec<f32>>,
    /// 向量维度
    vector_dim: usize,
    /// 预处理器
    preprocessor: Option<Box<dyn TextPreprocessor>>,
    /// 缓存
    cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    /// 是否已加载模型
    loaded: bool,
}

impl Clone for Word2VecExtractor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            word_vectors: self.word_vectors.clone(),
            vector_dim: self.vector_dim,
            preprocessor: None, // 不能克隆 trait 对象，设置为 None
            cache: Arc::clone(&self.cache),
            loaded: self.loaded,
        }
    }
}

impl Word2VecExtractor {
    /// 创建新的Word2Vec提取器
    pub fn new(config: MethodConfig) -> Self {
        Self {
            config,
            word_vectors: HashMap::new(),
            vector_dim: 0,
            preprocessor: None,
            cache: Arc::new(RwLock::new(HashMap::new())),
            loaded: false,
        }
    }
    
    /// 设置预处理器
    pub fn with_preprocessor(mut self, preprocessor: Box<dyn TextPreprocessor>) -> Self {
        self.preprocessor = Some(preprocessor);
        self
    }
    
    /// 从文件加载预训练的词向量模型
    pub fn load_pretrained(&mut self, file_path: &str) -> Result<(), Error> {
        // 这里是示例实现，实际中需要根据文件格式解析
        // 通常Word2Vec模型是文本文件，每行包含一个词和对应的向量
        
        // 模拟加载预训练模型
        self.word_vectors.clear();
        
        // 假设我们读取文件并解析
        // 第一行通常包含词汇量和向量维度
        // 格式: 词汇量 向量维度
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| Error::IoError(format!("无法读取模型文件: {}", e)))?;
            
        let lines: Vec<&str> = content.lines().collect();
        if lines.is_empty() {
            return Err(Error::invalid_argument("模型文件为空".to_string()));
        }
        
        // 解析第一行获取向量维度
        let header_parts: Vec<&str> = lines[0].split_whitespace().collect();
        if header_parts.len() == 2 {
            // 标准格式：词汇量 向量维度
            self.vector_dim = header_parts[1].parse::<usize>()
                .map_err(|_| Error::invalid_argument("无效的向量维度".to_string()))?;
        } else {
            // 推断向量维度
            let first_line_parts: Vec<&str> = lines[0].split_whitespace().collect();
            self.vector_dim = first_line_parts.len() - 1;
        }
        
        // 解析每一行
        for line in lines.iter().skip(if header_parts.len() == 2 { 1 } else { 0 }) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < self.vector_dim + 1 {
                continue; // 跳过格式不正确的行
            }
            
            let word = parts[0].to_string();
            let vector = parts[1..].iter()
                .map(|&s| s.parse::<f32>().unwrap_or(0.0))
                .collect::<Vec<f32>>();
                
            // 确保向量维度一致
            if vector.len() == self.vector_dim {
                self.word_vectors.insert(word, vector);
            }
        }
        
        if self.word_vectors.is_empty() {
            return Err(Error::InvalidState("未能加载任何词向量".to_string()));
        }
        
        self.loaded = true;
        self.cache.write().unwrap().clear();
        
        Ok(())
    }
    
    /// 预处理文本
    fn preprocess_text(&self, text: &str) -> Result<String, Error> {
        if let Some(preprocessor) = &self.preprocessor {
            preprocessor.preprocess(text)
        } else {
            Ok(text.to_string())
        }
    }
    
    /// 将文本分词
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|s| {
                if self.config.case_sensitive {
                    s.to_string()
                } else {
                    s.to_lowercase()
                }
            })
            .collect()
    }
    
    /// 计算文本的词向量
    fn compute_word_vectors(&self, text: &str) -> Result<Vec<f32>, Error> {
        if !self.loaded {
            return Err(Error::InvalidState("Word2Vec模型未加载".to_string()));
        }
        
        let processed_text = self.preprocess_text(text)?;
        let words = self.tokenize(&processed_text);
        
        if words.is_empty() {
            return Ok(vec![0.0; self.vector_dim]);
        }
        
        // 构建单词向量矩阵
        let mut vectors = Vec::new();
        
        for word in &words {
            if let Some(vector) = self.word_vectors.get(word) {
                vectors.push(vector.clone());
            }
        }
        
        if vectors.is_empty() {
            // 没有找到任何词的向量，返回零向量
            return Ok(vec![0.0; self.vector_dim]);
        }
        
        // 根据聚合方法计算最终向量
        match self.config.aggregation_method.as_str() {
            "mean" => {
                // 计算平均向量
                let mut result = vec![0.0; self.vector_dim];
                for vec in &vectors {
                    for (i, val) in vec.iter().enumerate() {
                        result[i] += val;
                    }
                }
                
                // 归一化
                let n = vectors.len() as f32;
                for val in result.iter_mut() {
                    *val /= n;
                }
                
                Ok(result)
            }
            "max" => {
                // 计算最大值向量
                let mut result = vec![f32::NEG_INFINITY; self.vector_dim];
                for vec in &vectors {
                    for (i, &val) in vec.iter().enumerate() {
                        if val > result[i] {
                            result[i] = val;
                        }
                    }
                }
                
                // 替换无穷大值
                for val in result.iter_mut() {
                    if val.is_infinite() {
                        *val = 0.0;
                    }
                }
                
                Ok(result)
            }
            "sum" => {
                // 计算和向量
                let mut result = vec![0.0; self.vector_dim];
                for vec in &vectors {
                    for (i, val) in vec.iter().enumerate() {
                        result[i] += val;
                    }
                }
                
                Ok(result)
            }
            "weighted" => {
                // 加权平均，根据词在文本中的位置
                let mut result = vec![0.0; self.vector_dim];
                let n = vectors.len() as f32;
                
                for (i, vec) in vectors.iter().enumerate() {
                    let weight = (n - i as f32) / n;  // 前面的词权重更高
                    for (j, val) in vec.iter().enumerate() {
                        result[j] += val * weight;
                    }
                }
                
                // 归一化
                let sum_weights = (1..=vectors.len()).sum::<usize>() as f32 / vectors.len() as f32;
                for val in result.iter_mut() {
                    *val /= sum_weights;
                }
                
                Ok(result)
            }
            _ => {
                // 默认使用平均值
                let mut result = vec![0.0; self.vector_dim];
                for vec in &vectors {
                    for (i, val) in vec.iter().enumerate() {
                        result[i] += val;
                    }
                }
                
                let n = vectors.len() as f32;
                for val in result.iter_mut() {
                    *val /= n;
                }
                
                Ok(result)
            }
        }
    }
}

impl FeatureMethod for Word2VecExtractor {
    /// 提取特征
    fn extract(&self, text: &str) -> Result<Vec<f32>, Error> {
        // 检查缓存
        if let Some(cached) = self.cache.read().unwrap().get(text) {
            return Ok(cached.clone());
        }
        
        let features = self.compute_word_vectors(text)?;
        
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
        "Word2Vec"
    }
    
    /// 获取方法配置
    fn config(&self) -> &MethodConfig {
        &self.config
    }
    
    /// 重置状态
    fn reset(&mut self) {
        self.word_vectors.clear();
        self.vector_dim = 0;
        self.loaded = false;
        self.cache.write().unwrap().clear();
    }
    
    /// 批量提取特征
    fn batch_extract(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, Error> {
        if !self.loaded {
            return Err(Error::InvalidState("Word2Vec模型未加载".to_string()));
        }
        
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
    fn tokenize_text(&self, text: &str) -> Result<Vec<String>, Error> {
        Ok(self.tokenize(text))
    }
    
    /// 获取词嵌入
    fn get_word_embeddings(&self, tokens: &[String]) -> Result<Vec<Vec<f32>>, Error> {
        let mut embeddings = Vec::new();
        for token in tokens {
            if let Some(vector) = self.word_vectors.get(token) {
                embeddings.push(vector.clone());
            } else {
                embeddings.push(vec![0.0; self.vector_dim]);
            }
        }
        Ok(embeddings)
    }
    
    /// 查找词嵌入
    fn lookup_word_embedding(&self, word: &str) -> Result<Vec<f32>, Error> {
        if let Some(vector) = self.word_vectors.get(word) {
            Ok(vector.clone())
        } else {
            Ok(vec![0.0; self.vector_dim])
        }
    }
    
    /// 获取未知词嵌入
    fn get_unknown_word_embedding(&self) -> Result<Vec<f32>, Error> {
        Ok(vec![0.0; self.vector_dim])
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