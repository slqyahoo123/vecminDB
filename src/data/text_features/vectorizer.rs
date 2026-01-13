// 文本向量化器桥接模块
// Text Vectorizer Bridge Module
// 
// 重新导出vectorization中的功能以保持向后兼容性
// Re-exports functionality from vectorization module for backward compatibility
//
// 这个模块提供了将文本转换为向量表示的功能，支持多种向量化方法
// This module provides functionality to convert text into vector representations
// using various vectorization methods

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::fs;
use std::io::{self, Read, Write};

use serde::{Serialize, Deserialize};
use ndarray::{Array1, Array2};
use thiserror::Error;

// 重导出原始模块的公共API
// pub use crate::data::text_features::vectorization::*;

/// 向量化错误类型
#[derive(Error, Debug)]
pub enum VectorizerError {
    #[error("IO错误: {0}")]
    IoError(#[from] io::Error),
    
    #[error("序列化错误: {0}")]
    SerializationError(String),
    
    #[error("词汇表为空")]
    EmptyVocabulary,
    
    #[error("模型未训练")]
    NotFitted,
    
    #[error("无效参数: {0}")]
    InvalidParameters(String),
    
    #[error("维度不匹配: 期望 {expected}，实际 {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("外部错误: {0}")]
    ExternalError(String),
}

/// 向量化方法
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum VectorizerMethod {
    /// TF-IDF (词频-逆文档频率)
    TfIdf,
    /// 词袋模型
    BagOfWords,
    /// Word2Vec词向量
    Word2Vec,
    /// GloVe词向量
    GloVe,
    /// FastText模型
    FastText,
    /// 自定义方法
    Custom(String),
}

impl Default for VectorizerMethod {
    fn default() -> Self {
        VectorizerMethod::TfIdf
    }
}

/// 向量化器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorizerOptions {
    /// 最小文档频率（绝对数或比例）
    pub min_df: Option<f64>,
    /// 最大文档频率（绝对数或比例）
    pub max_df: Option<f64>,
    /// 是否使用IDF权重
    pub use_idf: bool,
    /// 是否使用L2正则化
    pub normalize: bool,
    /// 是否使用二值化TF
    pub binary: bool,
    /// 是否使用次线性TF缩放
    pub sublinear_tf: bool,
    /// 最大特征数
    pub max_features: Option<usize>,
    /// 停用词
    pub stop_words: Option<HashSet<String>>,
    /// N元语法范围
    pub ngram_range: Option<(usize, usize)>,
    /// 向量维度 (用于Word2Vec/GloVe等)
    pub vector_size: Option<usize>,
    /// 上下文窗口大小 (用于Word2Vec)
    pub window_size: Option<usize>,
    /// 最小词频
    pub min_count: Option<usize>,
    /// 自定义选项
    pub custom_options: Option<HashMap<String, String>>,
}

impl Default for VectorizerOptions {
    fn default() -> Self {
        Self {
            min_df: Some(1.0),
            max_df: Some(1.0),
            use_idf: true,
            normalize: true,
            binary: false,
            sublinear_tf: false,
            max_features: None,
            stop_words: None,
            ngram_range: Some((1, 1)),
            vector_size: Some(100),
            window_size: Some(5),
            min_count: Some(5),
            custom_options: None,
        }
    }
}

/// 文本向量化器
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextVectorizer {
    /// 向量化方法
    pub method: VectorizerMethod,
    /// 向量维度
    pub dimension: usize,
    /// 向量化选项
    pub options: VectorizerOptions,
    /// 词汇表 (词语 -> 索引映射)
    pub vocabulary: Option<HashMap<String, usize>>,
    /// 文档频率 (用于TF-IDF)
    pub document_frequencies: Option<HashMap<String, usize>>,
    /// 逆文档频率 (用于TF-IDF)
    pub idf_values: Option<HashMap<String, f32>>,
    /// 是否已拟合
    pub is_fitted: bool,
    /// 嵌入矩阵 (用于Word2Vec/GloVe)
    pub embedding_matrix: Option<Array2<f32>>,
    /// 文档计数
    pub document_count: usize,
}

impl TextVectorizer {
    /// 创建新的文本向量化器
    pub fn new(method: VectorizerMethod, dimension: usize) -> Self {
        Self {
            method,
            dimension,
            options: VectorizerOptions::default(),
            vocabulary: None,
            document_frequencies: None,
            idf_values: None,
            is_fitted: false,
            embedding_matrix: None,
            document_count: 0,
        }
    }
    
    /// 使用自定义选项创建向量化器
    pub fn with_options(method: VectorizerMethod, dimension: usize, options: VectorizerOptions) -> Self {
        Self {
            method,
            dimension,
            options,
            vocabulary: None,
            document_frequencies: None,
            idf_values: None,
            is_fitted: false,
            embedding_matrix: None,
            document_count: 0,
        }
    }
    
    /// 拟合向量化器
    pub fn fit(&mut self, texts: &[String]) -> Result<(), VectorizerError> {
        if texts.is_empty() {
            return Err(VectorizerError::InvalidParameters("输入文本不能为空".to_string()));
        }
        
        match self.method {
            VectorizerMethod::TfIdf | VectorizerMethod::BagOfWords => {
                self.fit_count_based(texts)?;
            },
            VectorizerMethod::Word2Vec => {
                self.fit_word2vec(texts)?;
            },
            VectorizerMethod::GloVe => {
                self.fit_glove(texts)?;
            },
            VectorizerMethod::FastText => {
                self.fit_fasttext(texts)?;
            },
            VectorizerMethod::Custom(ref name) => {
                self.fit_custom(name, texts)?;
            }
        }
        
        self.is_fitted = true;
        Ok(())
    }
    
    /// 转换文本为向量
    pub fn transform(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VectorizerError> {
        if !self.is_fitted {
            return Err(VectorizerError::NotFitted);
        }
        
        match self.method {
            VectorizerMethod::TfIdf | VectorizerMethod::BagOfWords => {
                self.transform_count_based(texts)
            },
            VectorizerMethod::Word2Vec => {
                self.transform_word2vec(texts)
            },
            VectorizerMethod::GloVe => {
                self.transform_glove(texts)
            },
            VectorizerMethod::FastText => {
                self.transform_fasttext(texts)
            },
            VectorizerMethod::Custom(ref name) => {
                self.transform_custom(name, texts)
            }
        }
    }
    
    /// 转换单个文本为向量
    pub fn transform_single(&self, text: &str) -> Result<Vec<f32>, VectorizerError> {
        let result = self.transform(&[text.to_string()])?;
        Ok(result.into_iter().next().unwrap_or_default())
    }
    
    /// 拟合并转换
    pub fn fit_transform(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, VectorizerError> {
        self.fit(texts)?;
        self.transform(texts)
    }
    
    /// 保存向量化器到文件
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), VectorizerError> {
        let serialized = serde_json::to_string(self)
            .map_err(|e| VectorizerError::SerializationError(e.to_string()))?;
            
        fs::write(path, serialized)?;
        Ok(())
    }
    
    /// 从文件加载向量化器
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, VectorizerError> {
        let mut contents = String::new();
        let mut file = fs::File::open(path)?;
        file.read_to_string(&mut contents)?;
        
        let vectorizer: TextVectorizer = serde_json::from_str(&contents)
            .map_err(|e| VectorizerError::SerializationError(e.to_string()))?;
            
        Ok(vectorizer)
    }
    
    /// 拟合基于计数的向量化器 (TF-IDF, Bag of Words)
    fn fit_count_based(&mut self, texts: &[String]) -> Result<(), VectorizerError> {
        // 初始化数据结构
        let mut vocabulary = HashMap::new();
        let mut document_frequencies = HashMap::new();
        let mut token_counter = 0;
        
        // 处理每个文档
        for text in texts {
            let tokens = self.tokenize(text)?;
            let mut seen_tokens = HashSet::new();
            
            for token in &tokens {
                // 更新词汇表
                if !vocabulary.contains_key(token) {
                    vocabulary.insert(token.clone(), token_counter);
                    token_counter += 1;
                }
                
                // 更新文档频率 (每个文档中只计算一次)
                if !seen_tokens.contains(token) {
                    *document_frequencies.entry(token.clone()).or_insert(0) += 1;
                    seen_tokens.insert(token.clone());
                }
            }
        }
        
        self.document_count = texts.len();
        
        // 应用最小和最大文档频率过滤
        self.apply_frequency_filter(&mut vocabulary, &mut document_frequencies)?;
        
        // 如果设置了最大特征数，限制词汇表大小
        if let Some(max_features) = self.options.max_features {
            if vocabulary.len() > max_features {
                self.limit_features(max_features, &mut vocabulary, &mut document_frequencies)?;
            }
        }
        
        // 重新索引词汇表
        let mut new_vocab = HashMap::new();
        for (i, (token, _)) in vocabulary.iter().enumerate() {
            new_vocab.insert(token.clone(), i);
        }
        
        // 计算IDF值 (仅TF-IDF需要)
        let mut idf_values = None;
        if self.method == VectorizerMethod::TfIdf && self.options.use_idf {
            let mut idfs = HashMap::new();
            let n_docs = self.document_count as f32;
            
            for (token, doc_freq) in &document_frequencies {
                if let Some(&token_idx) = new_vocab.get(token) {
                    // 使用平滑的IDF公式: log((N+1)/(df+1)) + 1
                    let idf = ((n_docs + 1.0) / (*doc_freq as f32 + 1.0)).ln() + 1.0;
                    idfs.insert(token.clone(), idf);
                }
            }
            
            idf_values = Some(idfs);
        }
        
        // 更新向量化器状态
        self.vocabulary = Some(new_vocab);
        self.document_frequencies = Some(document_frequencies);
        self.idf_values = idf_values;
        self.dimension = vocabulary.len();
        
        Ok(())
    }
    
    /// 应用文档频率过滤
    fn apply_frequency_filter(
        &self,
        vocabulary: &mut HashMap<String, usize>,
        document_frequencies: &mut HashMap<String, usize>
    ) -> Result<(), VectorizerError> {
        let n_docs = self.document_count as f64;
        
        // 计算最小文档频率阈值
        let min_df = if let Some(min_df_value) = self.options.min_df {
            if min_df_value < 1.0 {
                // 解释为比例
                (min_df_value * n_docs).ceil() as usize
            } else {
                // 解释为绝对数
                min_df_value as usize
            }
        } else {
            1 // 默认值
        };
        
        // 计算最大文档频率阈值
        let max_df = if let Some(max_df_value) = self.options.max_df {
            if max_df_value <= 1.0 {
                // 解释为比例
                (max_df_value * n_docs).floor() as usize
            } else {
                // 解释为绝对数
                max_df_value as usize
            }
        } else {
            n_docs as usize // 默认值
        };
        
        // 验证阈值
        if min_df > max_df {
            return Err(VectorizerError::InvalidParameters(
                format!("min_df ({}) 大于 max_df ({})", min_df, max_df)
            ));
        }
        
        // 过滤词汇表和文档频率
        let mut tokens_to_remove = Vec::new();
        for (token, &doc_freq) in document_frequencies.iter() {
            if doc_freq < min_df || doc_freq > max_df {
                tokens_to_remove.push(token.clone());
            }
        }
        
        for token in tokens_to_remove {
            vocabulary.remove(&token);
            document_frequencies.remove(&token);
        }
        
        Ok(())
    }
    
    /// 限制特征数量
    fn limit_features(
        &self,
        max_features: usize,
        vocabulary: &mut HashMap<String, usize>,
        document_frequencies: &mut HashMap<String, usize>
    ) -> Result<(), VectorizerError> {
        // 按文档频率排序特征
        let mut freq_pairs: Vec<_> = document_frequencies.iter().collect();
        freq_pairs.sort_by(|a, b| b.1.cmp(a.1)); // 降序排序
        
        // 保留前max_features个特征
        let selected_tokens: HashSet<_> = freq_pairs.iter()
                                                  .take(max_features)
                                                  .map(|(token, _)| (*token).clone())
                                                  .collect();
        
        // 移除未选择的特征
        vocabulary.retain(|token, _| selected_tokens.contains(token));
        document_frequencies.retain(|token, _| selected_tokens.contains(token));
        
        Ok(())
    }
    
    /// 基于计数的向量化转换 (TF-IDF, Bag of Words)
    fn transform_count_based(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VectorizerError> {
        let vocabulary = self.vocabulary.as_ref()
            .ok_or(VectorizerError::NotFitted)?;
            
        let mut result = Vec::with_capacity(texts.len());
        
        for text in texts {
            let tokens = self.tokenize(text)?;
            let mut vector = vec![0.0; self.dimension];
            
            // 计算词频
            let mut token_counts = HashMap::new();
            for token in tokens {
                if let Some(&token_idx) = vocabulary.get(&token) {
                    *token_counts.entry(token_idx).or_insert(0) += 1;
                }
            }
            
            // 填充向量
            for (token_idx, count) in token_counts {
                let tf = if self.options.binary {
                    1.0
                } else if self.options.sublinear_tf {
                    1.0 + (count as f32).ln()
                } else {
                    count as f32
                };
                
                if self.method == VectorizerMethod::TfIdf && self.options.use_idf {
                    if let Some(idf_values) = &self.idf_values {
                        if let Some(token) = vocabulary.iter().find(|(_, &idx)| idx == token_idx).map(|(k, _)| k) {
                            if let Some(&idf) = idf_values.get(token) {
                                vector[token_idx] = tf * idf;
                            }
                        }
                    }
                } else {
                    vector[token_idx] = tf;
                }
            }
            
            // 规范化
            if self.options.normalize {
                normalize_vector(&mut vector);
            }
            
            result.push(vector);
        }
        
        Ok(result)
    }
    
    /// 拟合Word2Vec向量化器
    fn fit_word2vec(&mut self, texts: &[String]) -> Result<(), VectorizerError> {
        // 这里应该使用实际的Word2Vec实现
        // 由于复杂度较高，这里仅提供框架
        
        // 1. 创建词汇表
        let mut vocabulary = HashMap::new();
        let mut word_counts = HashMap::new();
        
        // 预处理和计数
        for text in texts {
            let tokens = self.tokenize(text)?;
            for token in tokens {
                *word_counts.entry(token).or_insert(0) += 1;
            }
        }
        
        // 应用最小词频
        let min_count = self.options.min_count.unwrap_or(5);
        let filtered_words: Vec<_> = word_counts.iter()
            .filter(|(_, &count)| count >= min_count)
            .map(|(word, _)| word.clone())
            .collect();
            
        // 构建词汇表
        for (i, word) in filtered_words.iter().enumerate() {
            vocabulary.insert(word.clone(), i);
        }
        
        // 2. 训练Word2Vec模型
        // 这里是一个简化的实现，实际应使用专门的库
        let vector_size = self.options.vector_size.unwrap_or(100);
        let mut embeddings = Array2::zeros((vocabulary.len(), vector_size));
        
        // ... 实际的Word2Vec训练 ...
        // 此处应集成真实的Word2Vec训练逻辑
        // 一般会使用随机初始化的词向量，然后根据上下文进行迭代训练
        
        // 3. 更新状态
        self.vocabulary = Some(vocabulary);
        self.embedding_matrix = Some(embeddings);
        self.dimension = vector_size;
        self.document_count = texts.len();
        
        Ok(())
    }
    
    /// Word2Vec向量化转换
    fn transform_word2vec(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VectorizerError> {
        let vocabulary = self.vocabulary.as_ref()
            .ok_or(VectorizerError::NotFitted)?;
            
        let embedding_matrix = self.embedding_matrix.as_ref()
            .ok_or(VectorizerError::NotFitted)?;
            
        let mut result = Vec::with_capacity(texts.len());
        
        for text in texts {
            let tokens = self.tokenize(text)?;
            
            // 计算平均词向量
            let mut sum_vector = Array1::zeros(self.dimension);
            let mut count = 0;
            
            for token in tokens {
                if let Some(&token_idx) = vocabulary.get(&token) {
                    if token_idx < embedding_matrix.nrows() {
                        let word_vector = embedding_matrix.row(token_idx);
                        sum_vector += &word_vector;
                        count += 1;
                    }
                }
            }
            
            let mut avg_vector: Vec<f32> = Vec::with_capacity(self.dimension);
            if count > 0 {
                // 计算平均值
                sum_vector /= count as f32;
                avg_vector = sum_vector.iter().cloned().collect();
            } else {
                // 如果没有词在词汇表中，返回零向量
                avg_vector = vec![0.0; self.dimension];
            }
            
            // 规范化
            if self.options.normalize {
                normalize_vector(&mut avg_vector);
            }
            
            result.push(avg_vector);
        }
        
        Ok(result)
    }
    
    /// 拟合GloVe向量化器
    fn fit_glove(&mut self, texts: &[String]) -> Result<(), VectorizerError> {
        // GloVe实现与Word2Vec类似，但训练目标不同
        // 这里简化起见，调用Word2Vec实现
        // 实际实现应考虑全局词共现统计等GloVe特定算法
        
        // 1. 构建词共现矩阵
        // 2. 应用GloVe训练方法
        // 3. 生成词向量
        
        self.fit_word2vec(texts)
    }
    
    /// GloVe向量化转换
    fn transform_glove(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VectorizerError> {
        // 与Word2Vec转换逻辑相同
        self.transform_word2vec(texts)
    }
    
    /// 拟合FastText向量化器
    fn fit_fasttext(&mut self, texts: &[String]) -> Result<(), VectorizerError> {
        // FastText实现，增加了子词特性
        // 这里简化起见，调用Word2Vec实现
        // 实际实现应考虑FastText的子词特性
        
        // 1. 生成子词
        // 2. 训练子词和词向量
        // 3. 构建词向量查找表
        
        self.fit_word2vec(texts)
    }
    
    /// FastText向量化转换
    fn transform_fasttext(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, VectorizerError> {
        // 与Word2Vec转换逻辑类似，但有子词处理
        // 实际实现应考虑FastText的子词表示
        
        self.transform_word2vec(texts)
    }
    
    /// 拟合自定义向量化器
    fn fit_custom(&mut self, name: &str, texts: &[String]) -> Result<(), VectorizerError> {
        // 实现自定义向量化逻辑
        Err(VectorizerError::InvalidParameters(format!("自定义方法 '{}' 未实现", name)))
    }
    
    /// 自定义向量化转换
    fn transform_custom(&self, name: &str, texts: &[String]) -> Result<Vec<Vec<f32>>, VectorizerError> {
        // 实现自定义向量化逻辑
        Err(VectorizerError::InvalidParameters(format!("自定义方法 '{}' 未实现", name)))
    }
    
    /// 分词方法
    fn tokenize(&self, text: &str) -> Result<Vec<String>, VectorizerError> {
        // 简单的空格分词，实际项目中应使用更复杂的分词器
        let tokens: Vec<String> = text
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
            
        // 如果有停用词，过滤掉
        let filtered_tokens = if let Some(stop_words) = &self.options.stop_words {
            tokens.into_iter()
                .filter(|token| !stop_words.contains(token))
                .collect()
        } else {
            tokens
        };
        
        // 生成n-gram特征
        if let Some((min_n, max_n)) = self.options.ngram_range {
            if min_n > 1 || max_n > 1 {
                return self.generate_ngrams(&filtered_tokens, min_n, max_n);
            }
        }
        
        Ok(filtered_tokens)
    }
    
    /// 生成n-gram特征
    fn generate_ngrams(&self, tokens: &[String], min_n: usize, max_n: usize) -> Result<Vec<String>, VectorizerError> {
        let mut results = Vec::new();
        
        // 首先添加单个token（如果min_n为1）
        if min_n <= 1 {
            results.extend(tokens.iter().cloned());
        }
        
        // 然后添加多元语法
        for n in min_n.max(2)..=max_n {
            if n > tokens.len() {
                continue;
            }
            
            for i in 0..=tokens.len() - n {
                let ngram = tokens[i..i+n].join(" ");
                results.push(ngram);
            }
        }
        
        Ok(results)
    }
}

/// 规范化向量
fn normalize_vector(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in vec.iter_mut() {
            *val /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tfidf_vectorizer() {
        let texts = vec![
            "这是第一篇文档".to_string(),
            "这是第二篇文档".to_string(),
            "这个文档和其他文档不同".to_string(),
        ];
        
        let mut vectorizer = TextVectorizer::new(VectorizerMethod::TfIdf, 0);
        let vectors = vectorizer.fit_transform(&texts).unwrap();
        
        assert_eq!(vectors.len(), 3);
        assert!(vectorizer.is_fitted);
        assert!(vectorizer.dimension > 0);
    }
    
    #[test]
    fn test_bow_vectorizer() {
        let texts = vec![
            "这是第一篇文档".to_string(),
            "这是第二篇文档".to_string(),
            "这个文档和其他文档不同".to_string(),
        ];
        
        let mut options = VectorizerOptions::default();
        options.use_idf = false;
        
        let mut vectorizer = TextVectorizer::with_options(
            VectorizerMethod::BagOfWords,
            0,
            options
        );
        
        let vectors = vectorizer.fit_transform(&texts).unwrap();
        
        assert_eq!(vectors.len(), 3);
        assert!(vectorizer.is_fitted);
        assert!(vectorizer.dimension > 0);
    }
    
    #[test]
    fn test_normalize_vector() {
        let mut vec = vec![1.0, 2.0, 3.0];
        normalize_vector(&mut vec);
        
        let norm: f32 = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_tokenize() {
        let vectorizer = TextVectorizer::new(VectorizerMethod::TfIdf, 0);
        let tokens = vectorizer.tokenize("这是 一个 测试").unwrap();
        
        assert_eq!(tokens, vec!["这是", "一个", "测试"]);
    }
    
    #[test]
    fn test_ngrams() {
        let mut options = VectorizerOptions::default();
        options.ngram_range = Some((1, 2));
        
        let vectorizer = TextVectorizer::with_options(
            VectorizerMethod::TfIdf,
            0,
            options
        );
        
        let tokens = vectorizer.tokenize("这是 测试 文本").unwrap();
        
        assert!(tokens.contains(&"这是".to_string()));
        assert!(tokens.contains(&"测试".to_string()));
        assert!(tokens.contains(&"文本".to_string()));
        assert!(tokens.contains(&"这是 测试".to_string()));
        assert!(tokens.contains(&"测试 文本".to_string()));
    }
    
    #[test]
    fn test_transform_single() {
        let texts = vec![
            "这是第一篇文档".to_string(),
            "这是第二篇文档".to_string(),
            "这个文档和其他文档不同".to_string(),
        ];
        
        let mut vectorizer = TextVectorizer::new(VectorizerMethod::TfIdf, 0);
        vectorizer.fit(&texts).unwrap();
        
        let vector = vectorizer.transform_single("这是一篇测试文档").unwrap();
        
        assert_eq!(vector.len(), vectorizer.dimension);
    }
} 