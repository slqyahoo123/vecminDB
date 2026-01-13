use crate::data::text_features::methods::{FeatureMethod, MethodConfig};
use crate::Result;
use crate::Error;

/// 特征融合策略
pub enum FeatureFusionStrategy {
    /// 连接向量
    Concatenate,
    /// 平均向量
    Average,
    /// 加权平均
    WeightedAverage,
    /// 最大值
    Max,
    /// PCA降维
    PCA,
    /// 自定义策略
    Custom(String),
}

/// 组合特征提取器
pub struct CombinationExtractor {
    /// 配置
    config: MethodConfig,
    /// 特征提取器列表
    extractors: Vec<Box<dyn FeatureMethod + Send + Sync>>,
    /// 融合策略
    fusion_strategy: FeatureFusionStrategy,
    /// 权重（用于加权平均）
    weights: Option<Vec<f32>>,
}

impl CombinationExtractor {
    /// 创建新的组合特征提取器
    pub fn new(config: MethodConfig, fusion_strategy: FeatureFusionStrategy) -> Self {
        CombinationExtractor {
            config,
            extractors: Vec::new(),
            fusion_strategy,
            weights: None,
        }
    }
    
    /// 添加特征提取器
    pub fn add_extractor(&mut self, extractor: Box<dyn FeatureMethod + Send + Sync>) {
        self.extractors.push(extractor);
    }
    
    /// 设置权重
    pub fn set_weights(&mut self, weights: Vec<f32>) -> Result<()> {
        if weights.len() != self.extractors.len() {
            return Err(Error::InvalidArgument("权重数量必须与提取器数量相同".to_string()));
        }
        self.weights = Some(weights);
        Ok(())
    }
    
    /// 融合特征
    fn fuse_features(&self, features: Vec<Vec<f32>>) -> Result<Vec<f32>> {
        match self.fusion_strategy {
            FeatureFusionStrategy::Concatenate => {
                // 简单连接所有特征向量
                let mut result = Vec::new();
                for feat in features {
                    result.extend(feat);
                }
                Ok(result)
            },
            FeatureFusionStrategy::Average => {
                // 确保所有特征向量长度相同
                if features.is_empty() {
                    return Ok(vec![]);
                }
                
                let vec_dim = features[0].len();
                let mut result = vec![0.0; vec_dim];
                
                for feat in &features {
                    if feat.len() != vec_dim {
                        return Err(Error::InvalidArgument("所有特征向量长度必须相同".to_string()));
                    }
                    
                    for (i, &val) in feat.iter().enumerate() {
                        result[i] += val;
                    }
                }
                
                // 求平均
                for val in &mut result {
                    *val /= features.len() as f32;
                }
                
                Ok(result)
            },
            FeatureFusionStrategy::WeightedAverage => {
                // 确保有权重
                let weights = match &self.weights {
                    Some(w) => w,
                    None => return Err(Error::InvalidArgument("需要设置权重".to_string())),
                };
                
                if features.is_empty() {
                    return Ok(vec![]);
                }
                
                let vec_dim = features[0].len();
                let mut result = vec![0.0; vec_dim];
                
                // 计算权重和
                let weight_sum: f32 = weights.iter().sum();
                
                for (i, feat) in features.iter().enumerate() {
                    if feat.len() != vec_dim {
                        return Err(Error::InvalidArgument("所有特征向量长度必须相同".to_string()));
                    }
                    
                    let weight = weights[i] / weight_sum;
                    
                    for (j, &val) in feat.iter().enumerate() {
                        result[j] += val * weight;
                    }
                }
                
                Ok(result)
            },
            FeatureFusionStrategy::Max => {
                // 取每个维度的最大值
                if features.is_empty() {
                    return Ok(vec![]);
                }
                
                let vec_dim = features[0].len();
                let mut result = vec![std::f32::MIN; vec_dim];
                
                for feat in &features {
                    if feat.len() != vec_dim {
                        return Err(Error::InvalidArgument("所有特征向量长度必须相同".to_string()));
                    }
                    
                    for (i, &val) in feat.iter().enumerate() {
                        if val > result[i] {
                            result[i] = val;
                        }
                    }
                }
                
                Ok(result)
            },
            FeatureFusionStrategy::PCA => {
                // 简化实现，实际应该使用PCA算法
                // 这里只是返回平均值
                if features.is_empty() {
                    return Ok(vec![]);
                }
                
                let vec_dim = features[0].len();
                let mut result = vec![0.0; vec_dim];
                
                for feat in &features {
                    if feat.len() != vec_dim {
                        return Err(Error::InvalidArgument("所有特征向量长度必须相同".to_string()));
                    }
                    
                    for (i, &val) in feat.iter().enumerate() {
                        result[i] += val;
                    }
                }
                
                // 求平均
                for val in &mut result {
                    *val /= features.len() as f32;
                }
                
                Ok(result)
            },
            FeatureFusionStrategy::Custom(_) => {
                // 默认使用连接
                let mut result = Vec::new();
                for feat in features {
                    result.extend(feat);
                }
                Ok(result)
            },
        }
    }
}

impl FeatureMethod for CombinationExtractor {
    /// 提取特征
    fn extract(&self, text: &str) -> Result<Vec<f32>> {
        let mut features = Vec::with_capacity(self.extractors.len());
        
        // 从每个提取器获取特征
        for extractor in &self.extractors {
            let feat = extractor.extract(text)?;
            features.push(feat);
        }
        
        // 融合特征
        self.fuse_features(features)
    }
    
    /// 获取方法名称
    fn name(&self) -> &str {
        "CombinationExtractor"
    }
    
    /// 获取方法配置
    fn config(&self) -> &MethodConfig {
        &self.config
    }
    
    /// 重置状态
    fn reset(&mut self) {
        for extractor in &mut self.extractors {
            extractor.reset();
        }
    }
    
    /// 批量提取特征
    fn batch_extract(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            let features = self.extract(text)?;
            results.push(features);
        }
        Ok(results)
    }
    
    /// 分词方法
    fn tokenize_text(&self, text: &str) -> Result<Vec<String>> {
        // 使用第一个提取器的分词功能，如果没有则返回简单分词
        if let Some(extractor) = self.extractors.first() {
            extractor.tokenize_text(text)
        } else {
            Ok(text.split_whitespace().map(|s| s.to_string()).collect())
        }
    }
    
    /// 获取词嵌入
    fn get_word_embeddings(&self, tokens: &[String]) -> Result<Vec<Vec<f32>>> {
        // 使用第一个支持词嵌入的提取器
        for extractor in &self.extractors {
            if let Ok(embeddings) = extractor.get_word_embeddings(tokens) {
                return Ok(embeddings);
            }
        }
        Err(crate::error::Error::not_implemented("没有支持词嵌入的提取器"))
    }
    
    /// 查找词嵌入
    fn lookup_word_embedding(&self, word: &str) -> Result<Vec<f32>> {
        // 使用第一个支持词嵌入的提取器
        for extractor in &self.extractors {
            if let Ok(embedding) = extractor.lookup_word_embedding(word) {
                return Ok(embedding);
            }
        }
        Err(crate::error::Error::not_implemented("没有支持词嵌入的提取器"))
    }
    
    /// 获取未知词嵌入
    fn get_unknown_word_embedding(&self) -> Result<Vec<f32>> {
        // 使用第一个支持词嵌入的提取器
        for extractor in &self.extractors {
            if let Ok(embedding) = extractor.get_unknown_word_embedding() {
                return Ok(embedding);
            }
        }
        Err(crate::error::Error::not_implemented("没有支持词嵌入的提取器"))
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