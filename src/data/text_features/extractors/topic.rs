use crate::{Result, Error};
use crate::data::text_features::config::TextFeatureConfig;
use super::FeatureExtractor;
use std::collections::{HashMap, HashSet};
use rust_stemmers::{Algorithm, Stemmer};
use stop_words::{get, LANGUAGE};

/// 主题提取器
/// 
/// 使用LDA(Latent Dirichlet Allocation)算法提取文本主题特征
pub struct TopicExtractor {
    /// 配置信息
    config: TextFeatureConfig,
    /// 主题数量
    num_topics: usize,
    /// 词干提取器
    stemmer: Stemmer,
    /// 停用词集合
    stop_words: HashSet<String>,
    /// 词汇表
    vocabulary: HashMap<String, usize>,
    /// 主题-词分布
    topic_word_dist: Vec<Vec<f32>>,
    /// 特征维度
    dimension: usize,
}

impl TopicExtractor {
    /// 创建新的主题提取器
    pub fn new(num_topics: usize) -> Result<Self> {
        if num_topics < 1 {
            return Err(Error::invalid_argument(format!(
                "主题数量必须大于0,当前值: {}",
                num_topics
            )));
        }
        
        Ok(Self {
            config: TextFeatureConfig::default(),
            num_topics,
            stemmer: Stemmer::create(Algorithm::English),
            stop_words: get(LANGUAGE::English).into_iter().collect(),
            vocabulary: HashMap::new(),
            topic_word_dist: Vec::new(),
            dimension: num_topics,
        })
    }
    
    /// 从配置创建主题提取器
    pub fn from_config(config: &TextFeatureConfig) -> Result<Self> {
        // 从配置的metadata中获取主题数量，默认为10
        let num_topics = config.metadata.get("num_topics")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(10);
        
        Self::new(num_topics)
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
    
    /// 更新词汇表
    fn update_vocabulary(&mut self, words: &[String]) {
        for word in words {
            if !self.vocabulary.contains_key(word) {
                let idx = self.vocabulary.len();
                self.vocabulary.insert(word.clone(), idx);
            }
        }
    }
    
    /// 将文本转换为词袋表示
    fn text_to_bow(&self, words: &[String]) -> Vec<usize> {
        let mut bow = vec![0; self.vocabulary.len()];
        
        for word in words {
            if let Some(&idx) = self.vocabulary.get(word) {
                bow[idx] += 1;
            }
        }
        
        bow
    }
    
    /// 训练LDA模型
    fn train_lda(&mut self, corpus: &[Vec<String>], num_iterations: usize) {
        // 初始化主题-词分布
        let vocab_size = self.vocabulary.len();
        self.topic_word_dist = vec![vec![1.0 / vocab_size as f32; vocab_size]; self.num_topics];
        
        // 初始化文档-主题分布
        let mut doc_topic_dist = vec![vec![1.0 / self.num_topics as f32; self.num_topics]; corpus.len()];
        
        // Gibbs采样迭代
        for _ in 0..num_iterations {
            // 更新主题分布
            for (doc_id, words) in corpus.iter().enumerate() {
                let bow = self.text_to_bow(words);
                
                // 更新文档-主题分布
                for (word_id, &count) in bow.iter().enumerate() {
                    if count > 0 {
                        // 计算主题概率
                        let mut topic_probs = vec![0.0; self.num_topics];
                        for topic in 0..self.num_topics {
                            topic_probs[topic] = doc_topic_dist[doc_id][topic] * 
                                               self.topic_word_dist[topic][word_id];
                        }
                        
                        // 归一化
                        let sum: f32 = topic_probs.iter().sum();
                        if sum > 0.0 {
                            for prob in topic_probs.iter_mut() {
                                *prob /= sum;
                            }
                        }
                        
                        // 更新分布
                        doc_topic_dist[doc_id] = topic_probs;
                    }
                }
            }
        }
    }
    
    /// 推断文档主题分布
    fn infer_topics(&self, bow: &[usize]) -> Vec<f32> {
        let mut topic_dist = vec![1.0 / self.num_topics as f32; self.num_topics];
        
        // 使用变分推断
        for _ in 0..10 {
            let mut new_dist = vec![0.0; self.num_topics];
            
            for (word_id, &count) in bow.iter().enumerate() {
                if count > 0 {
                    for topic in 0..self.num_topics {
                        new_dist[topic] += topic_dist[topic] * 
                                         self.topic_word_dist[topic][word_id] * 
                                         count as f32;
                    }
                }
            }
            
            // 归一化
            let sum: f32 = new_dist.iter().sum();
            if sum > 0.0 {
                for prob in new_dist.iter_mut() {
                    *prob /= sum;
                }
            }
            
            topic_dist = new_dist;
        }
        
        topic_dist
    }
}

impl FeatureExtractor for TopicExtractor {
    fn extract(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(Error::InvalidData("输入文本为空".to_string()));
        }
        
        // 预处理文本
        let words = self.preprocess_text(text);
        if words.is_empty() {
            return Err(Error::InvalidData("预处理后文本为空".to_string()));
        }
        
        // 转换为词袋表示
        let bow = self.text_to_bow(&words);
        
        // 推断主题分布
        Ok(self.infer_topics(&bow))
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    fn name(&self) -> &str {
        "topic"
    }
    
    fn from_config(config: &TextFeatureConfig) -> Result<Self> {
        TopicExtractor::from_config(config)
    }
}

impl std::fmt::Debug for TopicExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TopicExtractor")
            .field("config", &self.config)
            .field("num_topics", &self.num_topics)
            .field("stemmer", &"<Stemmer>")
            .field("stop_words", &self.stop_words)
            .field("vocabulary", &self.vocabulary)
            .field("topic_word_dist", &self.topic_word_dist)
            .field("dimension", &self.dimension)
            .finish()
    }
}

impl Clone for TopicExtractor {
    fn clone(&self) -> Self {
        // 重新创建Stemmer，因为它不能直接克隆
        let stemmer = Stemmer::create(Algorithm::English);
        Self {
            config: self.config.clone(),
            num_topics: self.num_topics,
            stemmer,
            stop_words: self.stop_words.clone(),
            vocabulary: self.vocabulary.clone(),
            topic_word_dist: self.topic_word_dist.clone(),
            dimension: self.dimension,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_preprocess() {
        let extractor = TopicExtractor::new(10).unwrap();
        let text = "Hello World! This is a test.";
        let words = extractor.preprocess_text(text);
        
        assert!(!words.is_empty());
        assert!(words.iter().all(|w| !w.chars().any(|c| c.is_ascii_punctuation())));
    }
    
    #[test]
    fn test_vocabulary() {
        let mut extractor = TopicExtractor::new(10).unwrap();
        let words = vec!["hello".to_string(), "world".to_string()];
        
        extractor.update_vocabulary(&words);
        assert_eq!(extractor.vocabulary.len(), 2);
        
        let bow = extractor.text_to_bow(&words);
        assert_eq!(bow.len(), 2);
        assert_eq!(bow.iter().sum::<usize>(), 2);
    }
    
    #[test]
    fn test_feature_extraction() {
        let mut extractor = TopicExtractor::new(10).unwrap();
        let text = "Hello World! This is a test.";
        
        // 更新词汇表
        let words = extractor.preprocess_text(text);
        extractor.update_vocabulary(&words);
        
        // 训练模型
        extractor.train_lda(&[words], 10);
        
        // 提取特征
        let features = extractor.extract(text).unwrap();
        
        assert_eq!(features.len(), extractor.dimension());
        assert!(features.iter().sum::<f32>() - 1.0 < 1e-6);
    }
} 