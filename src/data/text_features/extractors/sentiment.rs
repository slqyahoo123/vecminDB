use crate::error::{Result, Error};
use crate::data::text_features::config::TextFeatureConfig;
use super::FeatureExtractor;
use std::collections::{HashMap, HashSet};
use rust_stemmers::{Algorithm, Stemmer};
use stop_words::{get, LANGUAGE};
use serde_json::from_str;

/// 情感分析提取器
/// 
/// 使用词典和规则方法提取文本的情感特征
pub struct SentimentExtractor {
    /// 配置信息
    config: TextFeatureConfig,
    /// 词干提取器
    stemmer: Stemmer,
    /// 停用词集合
    stop_words: HashSet<String>,
    /// 情感词典
    sentiment_dict: HashMap<String, f32>,
    /// 否定词集合
    negation_words: HashSet<String>,
    /// 增强词集合
    intensifier_words: HashMap<String, f32>,
    /// 特征维度
    dimension: usize,
}

impl SentimentExtractor {
    /// 创建新的情感分析提取器
    pub fn new() -> Result<Self> {
        // 加载情感词典
        let sentiment_dict = load_sentiment_dict()?;
        
        // 加载否定词
        let negation_words: HashSet<String> = vec![
            "not", "no", "never", "none", "nobody", "nothing", "neither", "nowhere",
            "hardly", "scarcely", "barely", "don't", "doesn't", "didn't", "won't",
            "wouldn't", "shouldn't", "can't", "cannot", "couldn't",
        ].into_iter().map(String::from).collect();
        
        // 加载增强词
        let intensifier_words: HashMap<String, f32> = vec![
            ("very", 1.5),
            ("extremely", 2.0),
            ("really", 1.5),
            ("quite", 1.2),
            ("somewhat", 0.8),
            ("slightly", 0.5),
        ].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        
        Ok(Self {
            config: TextFeatureConfig::default(),
            stemmer: Stemmer::create(Algorithm::English),
            stop_words: get(LANGUAGE::English).into_iter().collect(),
            sentiment_dict,
            negation_words,
            intensifier_words,
            dimension: 6, // 积极、消极、中性、复合、主观性、强度
        })
    }
    
    /// 从配置创建情感分析提取器
    pub fn from_config(config: &TextFeatureConfig) -> Result<Self> {
        let mut extractor = Self::new()?;
        extractor.config = config.clone();
        Ok(extractor)
    }
    
    /// 预处理文本
    fn preprocess_text(&self, text: &str) -> Vec<String> {
        // 分词
        let words: Vec<String> = text
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();
        
        // 过滤停用词和标点符号
        let filtered_words: Vec<String> = words
            .into_iter()
            .filter(|w| {
                !self.stop_words.contains(w) && 
                !w.chars().all(|c| c.is_ascii_punctuation())
            })
            .collect();
        
        // 词干提取
        filtered_words
            .into_iter()
            .map(|w| self.stemmer.stem(&w).to_string())
            .collect()
    }
    
    /// 分析情感特征
    fn analyze_sentiment(&self, words: &[String]) -> Vec<f32> {
        let mut features = vec![0.0; self.dimension];
        let mut pos_count = 0.0;
        let mut neg_count = 0.0;
        let mut neu_count = 0.0;
        let mut subjectivity = 0.0;
        let mut intensity = 0.0;
        
        let mut i = 0;
        while i < words.len() {
            let word = &words[i];
            let mut sentiment = 0.0;
            let mut is_negated = false;
            let mut intensity_multiplier = 1.0;
            
            // 检查否定词
            if i > 0 && self.negation_words.contains(&words[i - 1]) {
                is_negated = true;
            }
            
            // 检查增强词
            if i > 0 {
                if let Some(&multiplier) = self.intensifier_words.get(&words[i - 1]) {
                    intensity_multiplier = multiplier;
                }
            }
            
            // 获取情感值
            if let Some(&value) = self.sentiment_dict.get(word) {
                sentiment = value * intensity_multiplier;
                if is_negated {
                    sentiment = -sentiment;
                }
                
                // 更新计数
                if sentiment > 0.0 {
                    pos_count += 1.0;
                } else if sentiment < 0.0 {
                    neg_count += 1.0;
                } else {
                    neu_count += 1.0;
                }
                
                subjectivity += sentiment.abs();
                intensity += intensity_multiplier;
            }
            
            i += 1;
        }
        
        // 计算特征
        let total = pos_count + neg_count + neu_count;
        if total > 0.0 {
            features[0] = pos_count / total; // 积极
            features[1] = neg_count / total; // 消极
            features[2] = neu_count / total; // 中性
            features[3] = (pos_count - neg_count) / total; // 复合
            features[4] = subjectivity / total; // 主观性
            features[5] = intensity / total; // 强度
        }
        
        features
    }
}

impl FeatureExtractor for SentimentExtractor {
    fn extract(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(Error::invalid_data("输入文本为空".to_string()));
        }
        
        // 预处理文本
        let words = self.preprocess_text(text);
        if words.is_empty() {
            return Err(Error::invalid_data("预处理后文本为空".to_string()));
        }
        
        // 分析情感
        Ok(self.analyze_sentiment(&words))
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    fn name(&self) -> &str {
        "sentiment"
    }
    
    fn from_config(config: &TextFeatureConfig) -> Result<Self> {
        SentimentExtractor::from_config(config)
    }
}

/// 加载情感词典
fn load_sentiment_dict() -> Result<HashMap<String, f32>> {
    // 这里使用一个简单的情感词典,实际应用中应该使用更完整的词典
    let dict_str = r#"{
        "good": 1.0,
        "great": 1.5,
        "excellent": 2.0,
        "amazing": 2.0,
        "bad": -1.0,
        "terrible": -2.0,
        "awful": -2.0,
        "poor": -1.0,
        "happy": 1.0,
        "sad": -1.0,
        "angry": -1.5,
        "love": 2.0,
        "hate": -2.0
    }"#;
    
    from_str(dict_str).map_err(|e| Error::invalid_data(format!("加载情感词典失败: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_preprocess() {
        let extractor = SentimentExtractor::new().unwrap();
        let text = "Hello World! This is a test.";
        let words = extractor.preprocess_text(text);
        
        assert!(!words.is_empty());
        assert!(words.iter().all(|w| !w.chars().any(|c| c.is_ascii_punctuation())));
    }
    
    #[test]
    fn test_sentiment_analysis() {
        let extractor = SentimentExtractor::new().unwrap();
        
        // 积极文本
        let text = "This is a very good and excellent product!";
        let features = extractor.extract(text).unwrap();
        assert!(features[0] > features[1]); // 积极 > 消极
        
        // 消极文本
        let text = "This is a terrible and awful product!";
        let features = extractor.extract(text).unwrap();
        assert!(features[1] > features[0]); // 消极 > 积极
        
        // 中性文本
        let text = "This is a product.";
        let features = extractor.extract(text).unwrap();
        assert!(features[2] > features[0] && features[2] > features[1]); // 中性 > 积极/消极
    }
    
    #[test]
    fn test_negation() {
        let extractor = SentimentExtractor::new().unwrap();
        
        // 否定词改变情感
        let text = "This is not a good product.";
        let features = extractor.extract(text).unwrap();
        assert!(features[1] > features[0]); // 消极 > 积极
    }
    
    #[test]
    fn test_intensifier() {
        let extractor = SentimentExtractor::new().unwrap();
        
        // 增强词增加情感强度
        let text = "This is a very good product.";
        let features1 = extractor.extract(text).unwrap();
        
        let text = "This is a good product.";
        let features2 = extractor.extract(text).unwrap();
        
        assert!(features1[5] > features2[5]); // 强度1 > 强度2
    }
}

impl std::fmt::Debug for SentimentExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SentimentExtractor")
            .field("config", &self.config)
            .field("stemmer", &"<Stemmer>")
            .field("stop_words", &self.stop_words)
            .field("sentiment_dict", &self.sentiment_dict)
            .field("negation_words", &self.negation_words)
            .field("intensifier_words", &self.intensifier_words)
            .field("dimension", &self.dimension)
            .finish()
    }
}

impl Clone for SentimentExtractor {
    fn clone(&self) -> Self {
        // 重新创建Stemmer，因为它不能直接克隆
        let stemmer = Stemmer::create(Algorithm::English);
        Self {
            config: self.config.clone(),
            stemmer,
            stop_words: self.stop_words.clone(),
            sentiment_dict: self.sentiment_dict.clone(),
            negation_words: self.negation_words.clone(),
            intensifier_words: self.intensifier_words.clone(),
            dimension: self.dimension,
        }
    }
} 