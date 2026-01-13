// src/data/text_features/similarity.rs
//
// Transformer 相似度计算模块

use std::collections::HashMap;
use super::error::TransformerError;
use super::features::FeatureVector;

/// 相似度计算配置
#[derive(Debug, Clone)]
pub struct SimilarityConfig {
    /// 相似度计算方法
    pub method: SimilarityMethod,
    /// 是否归一化向量
    pub normalize_vectors: bool,
    /// 是否使用阈值过滤
    pub use_threshold: bool,
    /// 相似度阈值
    pub similarity_threshold: f32,
    /// 是否缓存计算结果
    pub use_caching: bool,
    /// 缓存大小
    pub cache_size: usize,
}

impl Default for SimilarityConfig {
    fn default() -> Self {
        Self {
            method: SimilarityMethod::Cosine,
            normalize_vectors: true,
            use_threshold: false,
            similarity_threshold: 0.5,
            use_caching: true,
            cache_size: 1000,
        }
    }
}

/// 相似度计算方法
#[derive(Debug, Clone, PartialEq, Hash)]
pub enum SimilarityMethod {
    /// 余弦相似度
    Cosine,
    /// 欧几里得距离
    Euclidean,
    /// 曼哈顿距离
    Manhattan,
    /// 皮尔逊相关系数
    Pearson,
    /// Jaccard相似度
    Jaccard,
    /// 编辑距离
    EditDistance,
    /// 自定义相似度
    Custom(String),
}

/// 相似度计算器
#[derive(Debug)]
pub struct SimilarityCalculator {
    /// 配置
    config: SimilarityConfig,
    /// 缓存
    cache: HashMap<String, f32>,
}

impl SimilarityCalculator {
    /// 创建新的相似度计算器
    pub fn new(config: SimilarityConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
        }
    }
    
    /// 计算两个特征向量的相似度
    pub fn calculate_similarity(&mut self, vec1: &FeatureVector, vec2: &FeatureVector) -> Result<f32, TransformerError> {
        // 检查向量维度是否匹配
        if vec1.dimension() != vec2.dimension() {
            return Err(TransformerError::dimension_mismatch(
                format!("{}", vec1.dimension()),
                format!("{}", vec2.dimension())
            ));
        }
        
        // 生成缓存键
        let cache_key = self.generate_cache_key(vec1, vec2);
        
        // 检查缓存
        if self.config.use_caching {
            if let Some(&cached_similarity) = self.cache.get(&cache_key) {
                return Ok(cached_similarity);
            }
        }
        
        // 计算相似度
        let similarity = match self.config.method {
            SimilarityMethod::Cosine => self.cosine_similarity(vec1, vec2)?,
            SimilarityMethod::Euclidean => self.euclidean_similarity(vec1, vec2)?,
            SimilarityMethod::Manhattan => self.manhattan_similarity(vec1, vec2)?,
            SimilarityMethod::Pearson => self.pearson_similarity(vec1, vec2)?,
            SimilarityMethod::Jaccard => self.jaccard_similarity(vec1, vec2)?,
            SimilarityMethod::EditDistance => self.edit_distance_similarity(vec1, vec2)?,
            SimilarityMethod::Custom(ref method) => self.custom_similarity(vec1, vec2, method)?,
        };
        
        // 应用阈值过滤
        if self.config.use_threshold && similarity < self.config.similarity_threshold {
            return Ok(0.0);
        }
        
        // 缓存结果
        if self.config.use_caching {
            self.cache_result(cache_key, similarity);
        }
        
        Ok(similarity)
    }
    
    /// 生成缓存键
    fn generate_cache_key(&self, vec1: &FeatureVector, vec2: &FeatureVector) -> String {
        // 使用向量的哈希值作为缓存键
        // 由于 f32 不能直接 hash，我们将向量转换为字节数组进行 hash
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // 将 f32 向量转换为字节数组进行 hash
        let vec1_bytes: Vec<u8> = vec1.values.iter()
            .flat_map(|&f| f.to_ne_bytes().to_vec())
            .collect();
        let vec2_bytes: Vec<u8> = vec2.values.iter()
            .flat_map(|&f| f.to_ne_bytes().to_vec())
            .collect();
        
        vec1_bytes.hash(&mut hasher);
        vec2_bytes.hash(&mut hasher);
        self.config.method.hash(&mut hasher);
        
        format!("{:x}", hasher.finish())
    }
    
    /// 缓存结果
    fn cache_result(&mut self, key: String, similarity: f32) {
        if self.cache.len() >= self.config.cache_size {
            // 简单的LRU策略：移除第一个元素
            if let Some(first_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&first_key);
            }
        }
        self.cache.insert(key, similarity);
    }
    
    /// 余弦相似度
    fn cosine_similarity(&self, vec1: &FeatureVector, vec2: &FeatureVector) -> Result<f32, TransformerError> {
        let v1 = if self.config.normalize_vectors {
            self.normalize_vector(&vec1.values)?
        } else {
            vec1.values.clone()
        };
        
        let v2 = if self.config.normalize_vectors {
            self.normalize_vector(&vec2.values)?
        } else {
            vec2.values.clone()
        };
        
        let dot_product = v1.iter().zip(v2.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();
        
        let norm1 = (v1.iter().map(|x| x * x).sum::<f32>()).sqrt();
        let norm2 = (v2.iter().map(|x| x * x).sum::<f32>()).sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }
        
        Ok(dot_product / (norm1 * norm2))
    }
    
    /// 欧几里得相似度（转换为相似度）
    fn euclidean_similarity(&self, vec1: &FeatureVector, vec2: &FeatureVector) -> Result<f32, TransformerError> {
        let distance = self.euclidean_distance(vec1, vec2)?;
        
        // 将距离转换为相似度：1 / (1 + distance)
        Ok(1.0 / (1.0 + distance))
    }
    
    /// 欧几里得距离
    fn euclidean_distance(&self, vec1: &FeatureVector, vec2: &FeatureVector) -> Result<f32, TransformerError> {
        let v1 = if self.config.normalize_vectors {
            self.normalize_vector(&vec1.values)?
        } else {
            vec1.values.clone()
        };
        
        let v2 = if self.config.normalize_vectors {
            self.normalize_vector(&vec2.values)?
        } else {
            vec2.values.clone()
        };
        
        let distance = v1.iter().zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        
        Ok(distance)
    }
    
    /// 曼哈顿相似度
    fn manhattan_similarity(&self, vec1: &FeatureVector, vec2: &FeatureVector) -> Result<f32, TransformerError> {
        let v1 = if self.config.normalize_vectors {
            self.normalize_vector(&vec1.values)?
        } else {
            vec1.values.clone()
        };
        
        let v2 = if self.config.normalize_vectors {
            self.normalize_vector(&vec2.values)?
        } else {
            vec2.values.clone()
        };
        
        let distance = v1.iter().zip(v2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>();
        
        // 转换为相似度
        Ok(1.0 / (1.0 + distance))
    }
    
    /// 皮尔逊相关系数
    fn pearson_similarity(&self, vec1: &FeatureVector, vec2: &FeatureVector) -> Result<f32, TransformerError> {
        let v1 = &vec1.values;
        let v2 = &vec2.values;
        
        let n = v1.len() as f32;
        let sum1: f32 = v1.iter().sum();
        let sum2: f32 = v2.iter().sum();
        let sum1_sq: f32 = v1.iter().map(|x| x * x).sum();
        let sum2_sq: f32 = v2.iter().map(|x| x * x).sum();
        let p_sum: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        
        let numerator = p_sum - (sum1 * sum2 / n);
        let denominator = ((sum1_sq - sum1 * sum1 / n) * (sum2_sq - sum2 * sum2 / n)).sqrt();
        
        if denominator == 0.0 {
            return Ok(0.0);
        }
        
        let correlation = numerator / denominator;
        
        // 将相关系数转换为相似度（0到1之间）
        Ok((correlation + 1.0) / 2.0)
    }
    
    /// Jaccard相似度
    fn jaccard_similarity(&self, vec1: &FeatureVector, vec2: &FeatureVector) -> Result<f32, TransformerError> {
        // 将向量转换为集合（非零元素）
        let set1: std::collections::HashSet<_> = vec1.values.iter()
            .enumerate()
            .filter(|(_, &val)| val > 0.0)
            .map(|(i, _)| i)
            .collect();
        
        let set2: std::collections::HashSet<_> = vec2.values.iter()
            .enumerate()
            .filter(|(_, &val)| val > 0.0)
            .map(|(i, _)| i)
            .collect();
        
        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();
        
        if union == 0 {
            return Ok(0.0);
        }
        
        Ok(intersection as f32 / union as f32)
    }
    
    /// 编辑距离相似度
    fn edit_distance_similarity(&self, vec1: &FeatureVector, vec2: &FeatureVector) -> Result<f32, TransformerError> {
        // 将向量转换为字符串进行比较
        let str1 = self.vector_to_string(vec1);
        let str2 = self.vector_to_string(vec2);
        
        let distance = self.levenshtein_distance(&str1, &str2);
        let max_len = std::cmp::max(str1.len(), str2.len());
        
        if max_len == 0 {
            return Ok(1.0);
        }
        
        Ok(1.0 - (distance as f32 / max_len as f32))
    }
    
    /// 自定义相似度
    fn custom_similarity(&self, vec1: &FeatureVector, vec2: &FeatureVector, method: &str) -> Result<f32, TransformerError> {
        match method {
            "weighted_cosine" => self.weighted_cosine_similarity(vec1, vec2),
            "hamming" => self.hamming_similarity(vec1, vec2),
            "chebyshev" => self.chebyshev_similarity(vec1, vec2),
            _ => Err(TransformerError::config_error(format!("未知的相似度方法: {}", method))),
        }
    }
    
    /// 加权余弦相似度
    fn weighted_cosine_similarity(&self, vec1: &FeatureVector, vec2: &FeatureVector) -> Result<f32, TransformerError> {
        let v1 = &vec1.values;
        let v2 = &vec2.values;
        
        // 使用特征重要性作为权重
        let weights: Vec<f32> = (0..v1.len()).map(|i| 1.0 + (i as f32 / v1.len() as f32)).collect();
        
        let weighted_dot_product: f32 = v1.iter().zip(v2.iter()).zip(weights.iter())
            .map(|((a, b), w)| a * b * w)
            .sum();
        
        let weighted_norm1: f32 = v1.iter().zip(weights.iter())
            .map(|(a, w)| a * a * w)
            .sum::<f32>()
            .sqrt();
        
        let weighted_norm2: f32 = v2.iter().zip(weights.iter())
            .map(|(a, w)| a * a * w)
            .sum::<f32>()
            .sqrt();
        
        if weighted_norm1 == 0.0 || weighted_norm2 == 0.0 {
            return Ok(0.0);
        }
        
        Ok(weighted_dot_product / (weighted_norm1 * weighted_norm2))
    }
    
    /// 汉明相似度
    fn hamming_similarity(&self, vec1: &FeatureVector, vec2: &FeatureVector) -> Result<f32, TransformerError> {
        let v1 = &vec1.values;
        let v2 = &vec2.values;
        
        let mut distance = 0;
        for (a, b) in v1.iter().zip(v2.iter()) {
            if (a - b).abs() > 1e-6 {
                distance += 1;
            }
        }
        
        Ok(1.0 - (distance as f32 / v1.len() as f32))
    }
    
    /// 切比雪夫相似度
    fn chebyshev_similarity(&self, vec1: &FeatureVector, vec2: &FeatureVector) -> Result<f32, TransformerError> {
        let v1 = &vec1.values;
        let v2 = &vec2.values;
        
        let max_distance = v1.iter().zip(v2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        
        Ok(1.0 / (1.0 + max_distance))
    }
    
    /// 归一化向量
    fn normalize_vector(&self, vector: &[f32]) -> Result<Vec<f32>, TransformerError> {
        let norm = (vector.iter().map(|x| x * x).sum::<f32>()).sqrt();
        
        if norm == 0.0 {
            return Ok(vec![0.0; vector.len()]);
        }
        
        Ok(vector.iter().map(|&x| x / norm).collect())
    }
    
    /// 向量转字符串
    fn vector_to_string(&self, vector: &FeatureVector) -> String {
        vector.values.iter()
            .map(|&x| if x > 0.5 { '1' } else { '0' })
            .collect()
    }
    
    /// 莱文斯坦距离
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();
        
        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }
        
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];
        
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }
        
        for (i, c1) in s1.chars().enumerate() {
            for (j, c2) in s2.chars().enumerate() {
                let cost = if c1 == c2 { 0 } else { 1 };
                matrix[i + 1][j + 1] = std::cmp::min(
                    matrix[i][j + 1] + 1,
                    std::cmp::min(
                        matrix[i + 1][j] + 1,
                        matrix[i][j] + cost
                    )
                );
            }
        }
        
        matrix[len1][len2]
    }
    
    /// 批量计算相似度
    pub fn calculate_batch_similarity(&mut self, query: &FeatureVector, candidates: &[FeatureVector]) -> Result<Vec<f32>, TransformerError> {
        let mut similarities = Vec::with_capacity(candidates.len());
        
        for candidate in candidates {
            let similarity = self.calculate_similarity(query, candidate)?;
            similarities.push(similarity);
        }
        
        Ok(similarities)
    }
    
    /// 找到最相似的向量
    pub fn find_most_similar(&mut self, query: &FeatureVector, candidates: &[FeatureVector], top_k: usize) -> Result<Vec<(usize, f32)>, TransformerError> {
        let similarities = self.calculate_batch_similarity(query, candidates)?;
        
        let mut indexed_similarities: Vec<(usize, f32)> = similarities.into_iter().enumerate().collect();
        indexed_similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(indexed_similarities.into_iter().take(top_k).collect())
    }
    
    /// 获取配置
    pub fn config(&self) -> &SimilarityConfig {
        &self.config
    }
    
    /// 清空缓存
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
    
    /// 获取缓存大小
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::features::FeatureType;

    #[test]
    fn test_cosine_similarity() {
        let config = SimilarityConfig::default();
        let mut calculator = SimilarityCalculator::new(config);
        
        let mut vec1 = FeatureVector::empty();
        vec1.add_feature("f1".to_string(), 1.0, FeatureType::Statistical);
        vec1.add_feature("f2".to_string(), 2.0, FeatureType::Statistical);
        
        let mut vec2 = FeatureVector::empty();
        vec2.add_feature("f1".to_string(), 2.0, FeatureType::Statistical);
        vec2.add_feature("f2".to_string(), 4.0, FeatureType::Statistical);
        
        let similarity = calculator.calculate_similarity(&vec1, &vec2).unwrap();
        assert!((similarity - 1.0).abs() < 1e-6); // 应该完全相似
    }

    #[test]
    fn test_euclidean_similarity() {
        let mut config = SimilarityConfig::default();
        config.method = SimilarityMethod::Euclidean;
        let mut calculator = SimilarityCalculator::new(config);
        
        let mut vec1 = FeatureVector::empty();
        vec1.add_feature("f1".to_string(), 1.0, FeatureType::Statistical);
        vec1.add_feature("f2".to_string(), 2.0, FeatureType::Statistical);
        
        let mut vec2 = FeatureVector::empty();
        vec2.add_feature("f1".to_string(), 3.0, FeatureType::Statistical);
        vec2.add_feature("f2".to_string(), 4.0, FeatureType::Statistical);
        
        let similarity = calculator.calculate_similarity(&vec1, &vec2).unwrap();
        assert!(similarity > 0.0 && similarity < 1.0);
    }

    #[test]
    fn test_jaccard_similarity() {
        let mut config = SimilarityConfig::default();
        config.method = SimilarityMethod::Jaccard;
        let mut calculator = SimilarityCalculator::new(config);
        
        let mut vec1 = FeatureVector::empty();
        vec1.add_feature("f1".to_string(), 1.0, FeatureType::Statistical);
        vec1.add_feature("f2".to_string(), 0.0, FeatureType::Statistical);
        vec1.add_feature("f3".to_string(), 1.0, FeatureType::Statistical);
        
        let mut vec2 = FeatureVector::empty();
        vec2.add_feature("f1".to_string(), 1.0, FeatureType::Statistical);
        vec2.add_feature("f2".to_string(), 1.0, FeatureType::Statistical);
        vec2.add_feature("f3".to_string(), 0.0, FeatureType::Statistical);
        
        let similarity = calculator.calculate_similarity(&vec1, &vec2).unwrap();
        assert_eq!(similarity, 0.5); // 交集1个，并集2个
    }

    #[test]
    fn test_batch_similarity() {
        let config = SimilarityConfig::default();
        let mut calculator = SimilarityCalculator::new(config);
        
        let mut query = FeatureVector::empty();
        query.add_feature("f1".to_string(), 1.0, FeatureType::Statistical);
        query.add_feature("f2".to_string(), 2.0, FeatureType::Statistical);
        
        let mut candidates = Vec::new();
        for i in 0..3 {
            let mut candidate = FeatureVector::empty();
            candidate.add_feature("f1".to_string(), (i + 1) as f32, FeatureType::Statistical);
            candidate.add_feature("f2".to_string(), (i + 2) as f32, FeatureType::Statistical);
            candidates.push(candidate);
        }
        
        let similarities = calculator.calculate_batch_similarity(&query, &candidates).unwrap();
        assert_eq!(similarities.len(), 3);
    }
} 