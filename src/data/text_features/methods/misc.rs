use crate::data::text_features::methods::{FeatureMethod, MethodConfig};
use crate::Result;
use std::collections::HashMap;

/// 混合特征提取器
pub struct MiscExtractor {
    /// 配置
    config: MethodConfig,
    /// 使用统计特征
    use_statistical: bool,
    /// 使用N-gram特征
    use_ngram: bool,
    /// 使用情感分析
    use_sentiment: bool,
    /// 特征缓存
    cache: HashMap<String, Vec<f32>>,
}

impl MiscExtractor {
    /// 创建新的混合特征提取器
    pub fn new(config: MethodConfig) -> Self {
        MiscExtractor {
            config,
            use_statistical: false,
            use_ngram: false,
            use_sentiment: false,
            cache: HashMap::new(),
        }
    }
    
    /// 启用统计特征
    pub fn enable_statistical(&mut self, enabled: bool) {
        self.use_statistical = enabled;
    }
    
    /// 启用N-gram特征
    pub fn enable_ngram(&mut self, enabled: bool) {
        self.use_ngram = enabled;
    }
    
    /// 启用情感分析
    pub fn enable_sentiment(&mut self, enabled: bool) {
        self.use_sentiment = enabled;
    }
    
    /// 提取统计特征
    fn extract_statistical_features(&self, text: &str) -> Vec<f32> {
        let mut features = Vec::new();
        
        // 文本长度
        features.push(text.len() as f32);
        
        // 词数
        let word_count = text.split_whitespace().count();
        features.push(word_count as f32);
        
        // 平均词长
        let avg_word_len = if word_count > 0 {
            text.split_whitespace().map(|w| w.len()).sum::<usize>() as f32 / word_count as f32
        } else {
            0.0
        };
        features.push(avg_word_len);
        
        // 大写字母比例
        let uppercase_ratio = if text.len() > 0 {
            text.chars().filter(|c| c.is_uppercase()).count() as f32 / text.len() as f32
        } else {
            0.0
        };
        features.push(uppercase_ratio);
        
        // 数字比例
        let digit_ratio = if text.len() > 0 {
            text.chars().filter(|c| c.is_digit(10)).count() as f32 / text.len() as f32
        } else {
            0.0
        };
        features.push(digit_ratio);
        
        // 标点符号比例
        let punct_ratio = if text.len() > 0 {
            text.chars().filter(|c| c.is_ascii_punctuation()).count() as f32 / text.len() as f32
        } else {
            0.0
        };
        features.push(punct_ratio);
        
        features
    }
    
    /// 提取N-gram特征
    fn extract_ngram_features(&self, text: &str) -> Vec<f32> {
        let mut features = Vec::new();
        
        // 简单实现，只返回一个占位向量
        let vec_dim = self.config.vector_dim;
        features = vec![0.0; vec_dim];
        
        features
    }
    
    /// 提取情感特征
    fn extract_sentiment_features(&self, text: &str) -> Vec<f32> {
        // 简单实现，只返回两个值：积极和消极概率
        let mut features = vec![0.5, 0.5]; // 默认中性
        
        features
    }
}

impl FeatureMethod for MiscExtractor {
    /// 提取特征
    fn extract(&self, text: &str) -> Result<Vec<f32>> {
        // 检查缓存
        if let Some(cached) = self.cache.get(text) {
            return Ok(cached.clone());
        }
        
        let mut features = Vec::new();
        
        // 提取各类特征
        if self.use_statistical {
            features.extend(self.extract_statistical_features(text));
        }
        
        if self.use_ngram {
            features.extend(self.extract_ngram_features(text));
        }
        
        if self.use_sentiment {
            features.extend(self.extract_sentiment_features(text));
        }
        
        // 如果没有启用任何特征，返回默认向量
        if features.is_empty() {
            features = vec![0.0; self.config.vector_dim];
        }
        
        Ok(features)
    }
    
    /// 获取方法名称
    fn name(&self) -> &str {
        "MiscExtractor"
    }
    
    /// 获取方法配置
    fn config(&self) -> &MethodConfig {
        &self.config
    }
    
    /// 重置状态
    fn reset(&mut self) {
        self.cache.clear();
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
    
    /// 分词
    fn tokenize_text(&self, text: &str) -> Result<Vec<String>> {
        // 简单的分词实现
        let tokens: Vec<String> = text
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        Ok(tokens)
    }
    
    /// 获取词嵌入
    fn get_word_embeddings(&self, words: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::with_capacity(words.len());
        for word in words {
            let embedding = self.lookup_word_embedding(word)?;
            embeddings.push(embedding);
        }
        Ok(embeddings)
    }
    
    /// 查找单词嵌入
    fn lookup_word_embedding(&self, word: &str) -> Result<Vec<f32>> {
        // 简单的嵌入实现：基于哈希值生成向量
        let hash = self.simple_hash(word)?;
        let dim = self.config.vector_dim.min(100); // 限制维度
        let mut embedding = Vec::with_capacity(dim);
        
        // 使用哈希值作为种子生成向量
        let mut seed = hash;
        for _ in 0..dim {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let value = (seed % 1000) as f32 / 1000.0 - 0.5; // 生成-0.5到0.5的值
            embedding.push(value);
        }
        
        Ok(embedding)
    }
    
    /// 获取未知词嵌入
    fn get_unknown_word_embedding(&self) -> Result<Vec<f32>> {
        // 返回零向量作为未知词嵌入
        let dim = self.config.vector_dim.min(100);
        Ok(vec![0.0; dim])
    }
    
    /// 简单哈希
    fn simple_hash(&self, text: &str) -> Result<u64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        Ok(hasher.finish())
    }
} 