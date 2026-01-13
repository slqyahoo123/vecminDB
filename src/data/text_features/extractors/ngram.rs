use crate::{Result, Error};
use crate::data::text_features::config::TextFeatureConfig;
use super::FeatureExtractor;
use std::collections::HashMap;

/// N-gram特征提取器
/// 
/// 提取文本中的N-gram特征,支持字符级和词级N-gram
#[derive(Debug, Clone)]
pub struct NGramExtractor {
    /// 配置信息
    config: TextFeatureConfig,
    /// N-gram大小
    n: usize,
    /// 是否使用字符级N-gram
    char_level: bool,
    /// 最大特征数
    max_features: usize,
    /// 特征维度
    dimension: usize,
    /// 特征映射
    feature_map: HashMap<String, usize>,
}

impl NGramExtractor {
    /// 创建新的N-gram特征提取器
    pub fn new(n: usize, char_level: bool, max_features: usize) -> Result<Self> {
        if n < 1 || n > 5 {
            return Err(Error::InvalidParameter(format!(
                "N-gram大小必须在1-5之间,当前值: {}",
                n
            )));
        }
        
        if max_features < 1 {
            return Err(Error::InvalidParameter(format!(
                "最大特征数必须大于0,当前值: {}",
                max_features
            )));
        }
        
        Ok(Self {
            config: TextFeatureConfig::default(),
            n,
            char_level,
            max_features,
            dimension: 0,
            feature_map: HashMap::new(),
        })
    }
    
    /// 从配置创建N-gram特征提取器
    pub fn from_config(config: &TextFeatureConfig) -> Result<Self> {
        // 从配置中获取参数
        // 从metadata中获取ngram_size，默认为2
        let n = config.metadata.get("ngram_size")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2);
        // 从metadata中获取char_level，默认为false
        let char_level = config.metadata.get("char_level")
            .and_then(|s| s.parse::<bool>().ok())
            .unwrap_or(false);
        // max_features是直接字段，不是Option
        let max_features = config.max_features;
        
        Self::new(n, char_level, max_features)
    }
    
    /// 提取N-gram特征
    fn extract_ngrams(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        
        let mut ngrams = Vec::new();
        
        if self.char_level {
            // 字符级N-gram
            let chars: Vec<char> = text.chars().collect();
            for i in 0..chars.len().saturating_sub(self.n - 1) {
                let ngram: String = chars[i..i + self.n].iter().collect();
                ngrams.push(ngram);
            }
        } else {
            // 词级N-gram
            let words: Vec<&str> = text.split_whitespace().collect();
            for i in 0..words.len().saturating_sub(self.n - 1) {
                let ngram = words[i..i + self.n].join(" ");
                ngrams.push(ngram);
            }
        }
        
        ngrams
    }
    
    /// 更新特征映射
    fn update_feature_map(&mut self, ngrams: &[String]) {
        // 统计N-gram频率
        let mut frequencies: HashMap<String, usize> = HashMap::new();
        for ngram in ngrams {
            *frequencies.entry(ngram.clone()).or_insert(0) += 1;
        }
        
        // 按频率排序
        let mut sorted_ngrams: Vec<_> = frequencies.into_iter().collect();
        sorted_ngrams.sort_by(|a, b| b.1.cmp(&a.1));
        
        // 更新特征映射
        self.feature_map.clear();
        for (i, (ngram, _)) in sorted_ngrams.into_iter().take(self.max_features).enumerate() {
            self.feature_map.insert(ngram, i);
        }
        
        // 更新特征维度
        self.dimension = self.feature_map.len();
    }
    
    /// 将N-gram转换为特征向量
    fn ngrams_to_features(&self, ngrams: &[String]) -> Vec<f32> {
        let mut features = vec![0.0; self.dimension];
        
        // 计算N-gram频率
        for ngram in ngrams {
            if let Some(&idx) = self.feature_map.get(ngram) {
                features[idx] += 1.0;
            }
        }
        
        // 归一化
        let sum: f32 = features.iter().sum();
        if sum > 0.0 {
            for feature in features.iter_mut() {
                *feature /= sum;
            }
        }
        
        features
    }
}

impl FeatureExtractor for NGramExtractor {
    fn extract(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(Error::InvalidData("输入文本为空".to_string()));
        }
        
        // 提取N-gram
        let ngrams = self.extract_ngrams(text);
        
        // 转换为特征向量
        Ok(self.ngrams_to_features(&ngrams))
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    fn name(&self) -> &str {
        "ngram"
    }
    
    fn from_config(config: &TextFeatureConfig) -> Result<Self> {
        NGramExtractor::from_config(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_char_ngram() {
        let extractor = NGramExtractor::new(2, true, 100).unwrap();
        let text = "hello";
        let ngrams = extractor.extract_ngrams(text);
        
        assert_eq!(ngrams, vec!["he", "el", "ll", "lo"]);
    }
    
    #[test]
    fn test_word_ngram() {
        let extractor = NGramExtractor::new(2, false, 100).unwrap();
        let text = "hello world";
        let ngrams = extractor.extract_ngrams(text);
        
        assert_eq!(ngrams, vec!["hello world"]);
    }
    
    #[test]
    fn test_feature_extraction() {
        let mut extractor = NGramExtractor::new(2, true, 100).unwrap();
        let text = "hello";
        let ngrams = extractor.extract_ngrams(text);
        
        // 更新特征映射
        extractor.update_feature_map(&ngrams);
        
        // 提取特征
        let features = extractor.extract(text).unwrap();
        
        // 验证特征
        assert_eq!(features.len(), extractor.dimension());
        assert!(features.iter().sum::<f32>() - 1.0 < 1e-6);
    }
} 