// Text Feature Methods Module
// 文本特征提取方法模块

pub mod tfidf;
pub mod word2vec;
pub mod char_level;
pub mod stats;
pub mod word_frequency;
pub mod fasttext;
pub mod embeddings;
pub mod transformer;
pub mod combinations;
pub mod misc;
pub mod types;
pub mod factory;
// pub mod selection;
// pub mod evaluation;

pub use tfidf::TfIdfExtractor as TfIdfFeatureExtractor;
pub use word2vec::Word2VecExtractor as Word2VecFeatureExtractor;
pub use stats::StatsExtractor as StatsFeatureExtractor;
pub use word_frequency::WordFrequencyExtractor as WordFrequencyFeatureExtractor;
pub use factory::create_extractor;
// pub use self::factory::{create_feature_extractor, register_method};
// pub use self::selection::{select_best_method, evaluate_method_performance};
// pub use self::evaluation::{MethodEvaluation, EvaluationMetrics};

use crate::Result;
use crate::Error;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
pub use crate::data::text_features::types::TextFeatureMethod;
use crate::data::text_features::config::TextFeatureConfig;

/// 特征提取方法的配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodConfig {
    /// 最大特征数量
    pub max_features: Option<usize>,
    /// 是否区分大小写
    pub case_sensitive: bool,
    /// 向量维度
    pub vector_dim: usize,
    /// 聚合方法
    pub aggregation_method: String,
    /// 是否标准化
    pub normalize: bool,
    /// 其他参数
    pub params: HashMap<String, String>,
}

impl Default for MethodConfig {
    fn default() -> Self {
        MethodConfig {
            max_features: None,
            case_sensitive: false,
            vector_dim: 100,
            aggregation_method: "mean".to_string(),
            normalize: false,
            params: HashMap::new(),
        }
    }
}

/// 特征提取方法trait
pub trait FeatureMethod: Send + Sync {
    /// 提取特征
    fn extract(&self, text: &str) -> Result<Vec<f32>>;
    
    /// 获取方法名称
    fn name(&self) -> &str;
    
    /// 获取方法配置
    fn config(&self) -> &MethodConfig;
    
    /// 重置状态
    fn reset(&mut self);
    
    /// 批量提取特征
    fn batch_extract(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    
    /// 分词
    fn tokenize_text(&self, text: &str) -> Result<Vec<String>>;
    
    /// 获取词嵌入
    fn get_word_embeddings(&self, words: &[String]) -> Result<Vec<Vec<f32>>>;
    
    /// 查找单词嵌入
    fn lookup_word_embedding(&self, word: &str) -> Result<Vec<f32>>;
    
    /// 获取未知词嵌入
    fn get_unknown_word_embedding(&self) -> Result<Vec<f32>>;
    
    /// 简单哈希
    fn simple_hash(&self, text: &str) -> Result<u64>;
    
    /// 平均池化
    fn mean_pooling(&self, embeddings: &[Vec<f32>]) -> Result<Vec<f32>> {
        if embeddings.is_empty() {
            return Ok(vec![0.0; self.config().vector_dim]);
        }
        
        let dim = embeddings[0].len();
        let mut result = vec![0.0; dim];
        
        for embedding in embeddings {
            for (i, &val) in embedding.iter().enumerate() {
                result[i] += val;
            }
        }
        
        let count = embeddings.len() as f32;
        for val in result.iter_mut() {
            *val /= count;
        }
        
        Ok(result)
    }
    
    /// 最大池化
    fn max_pooling(&self, embeddings: &[Vec<f32>]) -> Result<Vec<f32>> {
        if embeddings.is_empty() {
            return Ok(vec![0.0; self.config().vector_dim]);
        }
        
        let dim = embeddings[0].len();
        let mut result = vec![f32::NEG_INFINITY; dim];
        
        for embedding in embeddings {
            for (i, &val) in embedding.iter().enumerate() {
                if val > result[i] {
                    result[i] = val;
                }
            }
        }
        
        Ok(result)
    }
    
    /// 求和池化
    fn sum_pooling(&self, embeddings: &[Vec<f32>]) -> Result<Vec<f32>> {
        if embeddings.is_empty() {
            return Ok(vec![0.0; self.config().vector_dim]);
        }
        
        let dim = embeddings[0].len();
        let mut result = vec![0.0; dim];
        
        for embedding in embeddings {
            for (i, &val) in embedding.iter().enumerate() {
                result[i] += val;
            }
        }
        
        Ok(result)
    }
    
    /// 加权池化
    fn weighted_pooling(&self, embeddings: &[Vec<f32>], tokens: &[String]) -> Result<Vec<f32>> {
        if embeddings.is_empty() {
            return Ok(vec![0.0; self.config().vector_dim]);
        }
        
        let dim = embeddings[0].len();
        let mut result = vec![0.0; dim];
        let mut total_weight = 0.0;
        
        for (embedding, token) in embeddings.iter().zip(tokens.iter()) {
            // 简单的权重计算：基于词长度
            let weight = token.len() as f32;
            total_weight += weight;
            
            for (i, &val) in embedding.iter().enumerate() {
                result[i] += val * weight;
            }
        }
        
        if total_weight > 0.0 {
            for val in result.iter_mut() {
                *val /= total_weight;
            }
        }
        
        Ok(result)
    }
    
    /// 注意力池化
    fn attention_pooling(&self, embeddings: &[Vec<f32>]) -> Result<Vec<f32>> {
        if embeddings.is_empty() {
            return Ok(vec![0.0; self.config().vector_dim]);
        }
        
        // 计算查询向量（所有embedding的平均值）
        let query = self.mean_pooling(embeddings)?;
        
        // 计算注意力分数
        let mut attention_scores = Vec::new();
        for embedding in embeddings {
            let score = self.compute_attention_score(&query, embedding)?;
            attention_scores.push(score);
        }
        
        // 归一化注意力分数
        let sum_scores: f32 = attention_scores.iter().sum();
        if sum_scores > 0.0 {
            for score in attention_scores.iter_mut() {
                *score /= sum_scores;
            }
        }
        
        // 计算加权平均
        let dim = embeddings[0].len();
        let mut result = vec![0.0; dim];
        
        for (embedding, &weight) in embeddings.iter().zip(attention_scores.iter()) {
            for (i, &val) in embedding.iter().enumerate() {
                result[i] += val * weight;
            }
        }
        
        Ok(result)
    }
    
    /// 计算查询向量
    fn compute_query_vector(&self, embeddings: &[Vec<f32>]) -> Result<Vec<f32>> {
        // 简单实现：使用均值作为查询向量
        self.mean_pooling(embeddings)
    }
    
    /// 计算注意力分数
    fn compute_attention_score(&self, query: &[f32], key: &[f32]) -> Result<f32> {
        // 使用点积作为注意力分数
        let score = query.iter().zip(key.iter()).map(|(&q, &k)| q * k).sum();
        Ok(score)
    }
    
    /// L2正规化
    fn l2_normalize(&self, vector: &[f32]) -> Result<Vec<f32>> {
        let norm: f32 = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
        
        if norm == 0.0 {
            return Ok(vector.to_vec());
        }
        
        Ok(vector.iter().map(|&x| x / norm).collect())
    }
}

/// 方法管理器，用于创建和管理特征提取方法
pub struct MethodManager {
    /// 缓存的特征提取方法
    methods: HashMap<String, Box<dyn FeatureMethod>>,
}

impl std::fmt::Debug for MethodManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MethodManager")
            .field("methods_count", &self.methods.len())
            .finish()
    }
}

impl Clone for MethodManager {
    fn clone(&self) -> Self {
        // 由于 Box<dyn FeatureMethod> 无法直接克隆，我们创建一个新的空管理器
        // 在实际使用中，方法会在需要时重新创建
        MethodManager {
            methods: HashMap::new(),
        }
    }
}

impl MethodManager {
    /// 创建一个新的方法管理器
    pub fn new() -> Self {
        MethodManager {
            methods: HashMap::new(),
        }
    }
    
    /// 获取指定名称的特征提取方法
    pub fn get_method(&mut self, name: &str, config: Option<MethodConfig>) -> Result<&Box<dyn FeatureMethod>> {
        if !self.methods.contains_key(name) {
            // 将字符串转换为 TextFeatureMethod
            let method_enum = match name.to_lowercase().as_str() {
                "tfidf" => TextFeatureMethod::TfIdf,
                "word2vec" => TextFeatureMethod::Word2Vec,
                "bert" => TextFeatureMethod::Bert,
                _ => return Err(Error::invalid_argument(&format!("不支持的特征提取方法: {}", name))),
            };
            
            // 将 MethodConfig 转换为 TextFeatureConfig
            let text_config = if let Some(method_config) = config {
                let mut text_config = TextFeatureConfig::default();
                text_config.method = method_enum;
                text_config.max_features = method_config.max_features.unwrap_or(1000);
                text_config.case_sensitive = method_config.case_sensitive;
                text_config.feature_dimension = method_config.vector_dim;
                text_config.metadata.insert("normalize".to_string(), method_config.normalize.to_string());
                text_config.metadata.insert("aggregation_method".to_string(), method_config.aggregation_method.clone());
                // 将 params 复制到 metadata
                for (k, v) in method_config.params {
                    text_config.metadata.insert(k, v);
                }
                text_config
            } else {
                let mut text_config = TextFeatureConfig::default();
                text_config.method = method_enum;
                text_config
            };
            
            let method = create_extractor(method_enum, text_config)?;
            self.methods.insert(name.to_string(), method);
        }
        Ok(self.methods.get(name).unwrap())
    }
    
    /// 添加自定义方法
    pub fn add_method(&mut self, name: String, method: Box<dyn FeatureMethod>) {
        self.methods.insert(name, method);
    }
    
    /// 清除所有缓存的方法
    pub fn clear(&mut self) {
        self.methods.clear();
    }
} 