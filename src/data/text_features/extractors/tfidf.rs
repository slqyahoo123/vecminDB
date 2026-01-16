// TF-IDF特征提取器实现

use crate::error::{Error, Result};
use crate::data::text_features::config::TextFeatureConfig;
use crate::data::text_features::preprocessing::PreprocessingPipeline;
use super::FeatureExtractor;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};

/// TF-IDF特征提取器
pub struct TfIdfExtractor {
    /// 配置信息
    config: TextFeatureConfig,
    /// 提取器名称
    name: String,
    /// 词汇表
    vocabulary: HashMap<String, usize>,
    /// 文档频率 (DF)
    document_frequencies: HashMap<String, usize>,
    /// 逆文档频率 (IDF)
    idf_values: HashMap<String, f32>,
    /// 处理过的文档总数
    document_count: usize,
    /// 文本预处理器
    preprocessor: PreprocessingPipeline,
    /// 特征维度
    dimension: usize,
    /// 是否已拟合
    is_fitted: bool,
}

/// TF-IDF模型状态
#[derive(Serialize, Deserialize)]
struct TfIdfModelState {
    vocabulary: HashMap<String, usize>,
    document_frequencies: HashMap<String, usize>,
    idf_values: HashMap<String, f32>,
    document_count: usize,
    dimension: usize,
}

impl TfIdfExtractor {
    /// 创建新的TF-IDF特征提取器
    pub fn new(config: TextFeatureConfig) -> Result<Self> {
        let preprocessor = PreprocessingPipeline::new();
        
        Ok(Self {
            config,
            name: "TfIdfExtractor".to_string(),
            vocabulary: HashMap::new(),
            document_frequencies: HashMap::new(),
            idf_values: HashMap::new(),
            document_count: 0,
            preprocessor,
            dimension: 0,
            is_fitted: false,
        })
    }
    
    /// 拟合模型
    pub fn fit(&mut self, texts: &[String]) -> Result<()> {
        if texts.is_empty() {
            return Err(Error::invalid_data("训练数据为空".to_string()));
        }
        
        // 重置状态
        self.vocabulary.clear();
        self.document_frequencies.clear();
        self.idf_values.clear();
        self.document_count = 0;
        
        // 计算文档频率
        for text in texts {
            let processed_text = self.preprocessor.process(text)?;
            let tokens = self.tokenize(&processed_text)?;
            
            // 统计每个词在文档中的出现情况
            let mut seen = HashSet::new();
            for token in &tokens {
                seen.insert(token.clone());
            }
            
            // 更新文档频率
            for token in seen {
                *self.document_frequencies.entry(token).or_insert(0) += 1;
            }
            
            self.document_count += 1;
        }
        
        // 应用min_df和max_df过滤
        self.apply_frequency_filter()?;
        
        // 构建词汇表
        self.build_vocabulary()?;
        
        // 计算IDF值
        self.compute_idf()?;
        
        self.is_fitted = true;
        Ok(())
    }
    
    /// 转换文本为特征向量
    pub fn transform(&self, text: &str) -> Result<Vec<f32>> {
        if !self.is_fitted {
            return Err(Error::invalid_state("模型尚未拟合".to_string()));
        }
        
        // 预处理文本
        let processed_text = self.preprocessor.process(text)?;
        let tokens = self.tokenize(&processed_text)?;
        
        // 计算词频 (TF)
        let mut token_counts = HashMap::new();
        let total_tokens = tokens.len().max(1);
        
        for token in tokens {
            *token_counts.entry(token).or_insert(0) += 1;
        }
        
        // 计算TF-IDF向量
        let mut feature_vector = vec![0.0; self.dimension];
        
        for (token, count) in token_counts {
            if let Some(&idx) = self.vocabulary.get(&token) {
                // 使用标准TF计算（binary 和 sublinear_tf 不在 TextFeatureConfig 中，使用默认行为）
                let tf = count as f32 / total_tokens as f32;
                
                let idf = self.idf_values.get(&token).cloned().unwrap_or(0.0);
                let tfidf = if self.config.use_idf { tf * idf } else { tf };
                
                feature_vector[idx] = tfidf;
            }
        }
        
        // L2规范化（norm 不在 TextFeatureConfig 中，默认进行规范化）
        normalize_vector(&mut feature_vector);
        
        Ok(feature_vector)
    }
    
    /// 批量转换文本
    pub fn batch_transform(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());
        
        for text in texts {
            let features = self.transform(text)?;
            results.push(features);
        }
        
        Ok(results)
    }
    
    /// 保存模型状态
    pub fn save(&self) -> Result<String> {
        if !self.is_fitted {
            return Err(Error::invalid_state("模型尚未拟合".to_string()));
        }
        
        let state = TfIdfModelState {
            vocabulary: self.vocabulary.clone(),
            document_frequencies: self.document_frequencies.clone(),
            idf_values: self.idf_values.clone(),
            document_count: self.document_count,
            dimension: self.dimension,
        };
        
        serde_json::to_string(&state)
            .map_err(|e| Error::serialization(format!("无法序列化模型: {}", e)))
    }
    
    /// 加载模型状态
    pub fn load(&mut self, state_json: &str) -> Result<()> {
        let state: TfIdfModelState = serde_json::from_str(state_json)
            .map_err(|e| Error::serialization(format!("无法反序列化模型: {}", e)))?;
        
        self.vocabulary = state.vocabulary;
        self.document_frequencies = state.document_frequencies;
        self.idf_values = state.idf_values;
        self.document_count = state.document_count;
        self.dimension = state.dimension;
        
        self.is_fitted = true;
        Ok(())
    }
    
    // 内部辅助方法
    
    /// 分词
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        
        let tokens = text.split_whitespace()
            .map(|s| s.to_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
            
        Ok(tokens)
    }
    
    /// 应用频率过滤
    fn apply_frequency_filter(&mut self) -> Result<()> {
        let min_df = if self.config.min_df < 1.0 {
            // 解释为比例
            (self.config.min_df * self.document_count as f64) as usize
        } else {
            // 解释为绝对计数
            self.config.min_df as usize
        };
        
        let max_df = if self.config.max_df <= 1.0 {
            // 解释为比例
            (self.config.max_df * self.document_count as f64) as usize
        } else {
            // 解释为绝对计数
            self.config.max_df as usize
        };
        
        if min_df > max_df {
            return Err(Error::invalid_data(format!(
                "最小文档频率({})大于最大文档频率({})",
                min_df, max_df
            )));
        }
        
        // 应用过滤
        self.document_frequencies.retain(|_, &mut freq| {
            freq >= min_df && freq <= max_df
        });
        
        Ok(())
    }
    
    /// 构建词汇表
    fn build_vocabulary(&mut self) -> Result<()> {
        // 按文档频率排序
        let mut words: Vec<_> = self.document_frequencies.keys().cloned().collect();
        words.sort_by(|a, b| {
            let freq_a = self.document_frequencies.get(a).unwrap_or(&0);
            let freq_b = self.document_frequencies.get(b).unwrap_or(&0);
            freq_b.cmp(freq_a) // 按频率降序排序
        });
        
        // 应用max_features限制
        if self.config.max_features > 0 && words.len() > self.config.max_features {
            words.truncate(self.config.max_features);
        }
        
        // 构建词汇表
        self.vocabulary.clear();
        for (idx, word) in words.into_iter().enumerate() {
            self.vocabulary.insert(word, idx);
        }
        
        self.dimension = self.vocabulary.len();
        Ok(())
    }
    
    /// 计算IDF值
    fn compute_idf(&mut self) -> Result<()> {
        if self.document_count == 0 {
            return Err(Error::invalid_data("文档计数为0".to_string()));
        }
        
        let doc_count = self.document_count as f32;
        
        for (token, freq) in &self.document_frequencies {
            let idf = if self.config.smooth_idf {
                // 平滑IDF
                ((doc_count + 1.0) / (*freq as f32 + 1.0)).ln()
            } else {
                // 标准IDF
                (doc_count / *freq as f32).ln()
            };
            
            self.idf_values.insert(token.clone(), idf);
        }
        
        Ok(())
    }
}

impl FeatureExtractor for TfIdfExtractor {
    fn extract(&self, text: &str) -> Result<Vec<f32>> {
        self.transform(text)
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn from_config(config: &TextFeatureConfig) -> Result<Self> {
        Self::new(config.clone())
    }
}

/// 向量L2规范化
fn normalize_vector(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
} 