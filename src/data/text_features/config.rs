// src/data/text_features/config.rs
//
// Transformer 模型配置模块

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// 文本特征提取方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TextFeatureMethod {
    /// TF-IDF (词频-逆文档频率)
    TfIdf,
    /// 词袋模型
    BagOfWords,
    /// 词频统计
    WordFrequency,
    /// 字符级特征
    CharacterLevel,
    /// 统计特征
    Statistical,
    /// Word2Vec词向量
    Word2Vec,
    /// BERT预训练模型
    Bert,
    /// 混合特征方法
    Mixed,
    /// BERT嵌入
    BertEmbedding,
    /// FastText模型
    FastText,
    /// GloVe词向量
    GloVe,
    /// Universal Sentence Encoder
    Universal,
    /// ELMo模型
    Elmo,
    /// 计数向量
    Count,
    /// N-gram特征
    NGram,
    /// 实体提取
    EntityExtraction,
    /// 主题建模
    TopicModeling,
    /// 情感分析
    SentimentAnalysis,
    /// 上下文感知特征
    ContextAware,
    /// 增强特征表示
    EnhancedRepresentation,
    /// 自动选择最佳方法
    AutoSelect,
    /// 自定义方法
    Custom(usize),
}

impl Default for TextFeatureMethod {
    fn default() -> Self {
        TextFeatureMethod::TfIdf
    }
}

impl AsRef<str> for TextFeatureMethod {
    fn as_ref(&self) -> &str {
        match self {
            TextFeatureMethod::TfIdf => "tfidf",
            TextFeatureMethod::BagOfWords => "bag_of_words",
            TextFeatureMethod::WordFrequency => "word_frequency",
            TextFeatureMethod::CharacterLevel => "character_level",
            TextFeatureMethod::Statistical => "statistical",
            TextFeatureMethod::Word2Vec => "word2vec",
            TextFeatureMethod::Bert => "bert",
            TextFeatureMethod::Mixed => "mixed",
            TextFeatureMethod::BertEmbedding => "bert_embedding",
            TextFeatureMethod::FastText => "fasttext",
            TextFeatureMethod::GloVe => "glove",
            TextFeatureMethod::Universal => "universal",
            TextFeatureMethod::Elmo => "elmo",
            TextFeatureMethod::Count => "count",
            TextFeatureMethod::NGram => "ngram",
            TextFeatureMethod::EntityExtraction => "entity_extraction",
            TextFeatureMethod::TopicModeling => "topic_modeling",
            TextFeatureMethod::SentimentAnalysis => "sentiment_analysis",
            TextFeatureMethod::ContextAware => "context_aware",
            TextFeatureMethod::EnhancedRepresentation => "enhanced_representation",
            TextFeatureMethod::AutoSelect => "auto_select",
            TextFeatureMethod::Custom(id) => {
                // 对于自定义方法，使用数字作为标识
                // 注意：这里需要静态字符串，所以使用固定格式
                "custom"
            }
        }
    }
}

/// Transformer模型配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// 模型名称
    pub model_name: String,
    /// 模型路径
    pub model_path: String,
    /// 最大序列长度
    pub max_seq_length: usize,
    /// 隐藏层大小
    pub hidden_size: usize,
    /// 词汇表大小
    pub vocab_size: usize,
    /// 是否区分大小写
    pub case_sensitive: bool,
    /// 注意力头数
    pub num_heads: usize,
    /// 层数
    pub num_layers: usize,
    /// 前馈网络维度
    pub feed_forward_dim: usize,
    /// Dropout率
    pub dropout_rate: f32,
    /// 学习率
    pub learning_rate: f32,
    /// 批处理大小
    pub batch_size: usize,
    /// 训练轮次
    pub epochs: usize,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            model_name: "default-transformer".to_string(),
            model_path: "./models/transformer".to_string(),
            max_seq_length: 512,
            hidden_size: 768,
            vocab_size: 30000,
            case_sensitive: false,
            num_heads: 12,
            num_layers: 6,
            feed_forward_dim: 3072,
            dropout_rate: 0.1,
            learning_rate: 0.0001,
            batch_size: 32,
            epochs: 10,
        }
    }
}

impl TransformerConfig {
    /// 创建新的配置
    pub fn new(
        model_name: String,
        model_path: String,
        max_seq_length: usize,
        hidden_size: usize,
        vocab_size: usize,
    ) -> Self {
        Self {
            model_name,
            model_path,
            max_seq_length,
            hidden_size,
            vocab_size,
            case_sensitive: false,
            num_heads: 12,
            num_layers: 6,
            feed_forward_dim: hidden_size * 4,
            dropout_rate: 0.1,
            learning_rate: 0.0001,
            batch_size: 32,
            epochs: 10,
        }
    }

    /// 设置注意力头数
    pub fn with_num_heads(mut self, num_heads: usize) -> Self {
        self.num_heads = num_heads;
        self
    }

    /// 设置层数
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// 设置前馈网络维度
    pub fn with_feed_forward_dim(mut self, feed_forward_dim: usize) -> Self {
        self.feed_forward_dim = feed_forward_dim;
        self
    }

    /// 设置Dropout率
    pub fn with_dropout_rate(mut self, dropout_rate: f32) -> Self {
        self.dropout_rate = dropout_rate;
        self
    }

    /// 设置学习率
    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// 设置批处理大小
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// 设置训练轮次
    pub fn with_epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// 设置是否区分大小写
    pub fn with_case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }

    /// 验证配置的有效性
    pub fn validate(&self) -> Result<(), String> {
        if self.max_seq_length == 0 {
            return Err("max_seq_length must be greater than 0".to_string());
        }
        if self.hidden_size == 0 {
            return Err("hidden_size must be greater than 0".to_string());
        }
        if self.vocab_size == 0 {
            return Err("vocab_size must be greater than 0".to_string());
        }
        if self.num_heads == 0 {
            return Err("num_heads must be greater than 0".to_string());
        }
        if self.num_layers == 0 {
            return Err("num_layers must be greater than 0".to_string());
        }
        if self.hidden_size % self.num_heads != 0 {
            return Err("hidden_size must be divisible by num_heads".to_string());
        }
        if self.dropout_rate < 0.0 || self.dropout_rate > 1.0 {
            return Err("dropout_rate must be between 0.0 and 1.0".to_string());
        }
        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be greater than 0.0".to_string());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be greater than 0".to_string());
        }
        if self.epochs == 0 {
            return Err("epochs must be greater than 0".to_string());
        }
        Ok(())
    }

    /// 获取每个注意力头的维度
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// 获取模型参数总数
    pub fn total_parameters(&self) -> usize {
        // 简化的参数计算
        let embedding_params = self.vocab_size * self.hidden_size;
        let transformer_params = self.num_layers * (
            self.hidden_size * self.hidden_size * 4 + // 注意力层
            self.hidden_size * self.feed_forward_dim * 2 + // 前馈网络
            self.hidden_size * 2 // 层归一化
        );
        embedding_params + transformer_params
    }
}

/// 文本特征配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextFeatureConfig {
    /// 特征提取方法
    pub extraction_method: String,
    /// 特征提取方法枚举
    pub method: TextFeatureMethod,
    /// 最大特征数量
    pub max_features: usize,
    /// 最小文档频率
    pub min_df: f64,
    /// 最大文档频率
    pub max_df: f64,
    /// 是否使用IDF
    pub use_idf: bool,
    /// 是否平滑IDF
    pub smooth_idf: bool,
    /// 停用词列表
    pub stop_words: Option<Vec<String>>,
    /// 预处理配置
    pub preprocessing: PreprocessingConfig,
    /// 特征维度
    pub feature_dimension: usize,
    /// 是否使用TF-IDF
    pub use_tfidf: bool,
    /// 是否使用词向量
    pub use_word_vectors: bool,
    /// 词向量维度
    pub word_vector_dimension: usize,
    /// 最大词汇表大小
    pub max_vocabulary_size: usize,
    /// 最小词频
    pub min_word_frequency: usize,
    /// 停用词文件路径
    pub stopwords_path: Option<String>,
    /// 自定义停用词列表
    pub custom_stopwords: Vec<String>,
    /// 是否进行词干提取
    pub stemming: bool,
    /// 是否进行词形还原
    pub lemmatization: bool,
    /// 语言设置
    pub language: String,
    /// 是否区分大小写
    pub case_sensitive: bool,
    /// 最大文本长度
    pub max_text_length: usize,
    /// 批处理大小
    pub batch_size: usize,
    /// 并行处理线程数
    pub num_threads: usize,
    /// 缓存大小
    pub cache_size: usize,
    /// 是否启用缓存
    pub enable_cache: bool,
    /// 模型路径
    pub model_path: Option<String>,
    /// 配置元数据
    pub metadata: HashMap<String, String>,
}

impl Default for TextFeatureConfig {
    fn default() -> Self {
        Self {
            extraction_method: "tfidf".to_string(),
            method: TextFeatureMethod::TfIdf,
            max_features: 1000,
            min_df: 1.0,
            max_df: 1.0,
            use_idf: true,
            smooth_idf: true,
            stop_words: None,
            preprocessing: PreprocessingConfig::default(),
            feature_dimension: 1000,
            use_tfidf: true,
            use_word_vectors: false,
            word_vector_dimension: 100,
            max_vocabulary_size: 10000,
            min_word_frequency: 2,
            stopwords_path: None,
            custom_stopwords: Vec::new(),
            stemming: false,
            lemmatization: false,
            language: "en".to_string(),
            case_sensitive: false,
            max_text_length: 1000,
            batch_size: 32,
            num_threads: 4,
            cache_size: 1000,
            enable_cache: true,
            model_path: None,
            metadata: HashMap::new(),
        }
    }
}

impl TextFeatureConfig {
    /// 创建新的文本特征配置
    pub fn new(extraction_method: String) -> Self {
        Self {
            extraction_method,
            ..Default::default()
        }
    }

    /// 设置预处理配置
    pub fn with_preprocessing(mut self, preprocessing: PreprocessingConfig) -> Self {
        self.preprocessing = preprocessing;
        self
    }

    /// 设置特征维度
    pub fn with_feature_dimension(mut self, dimension: usize) -> Self {
        self.feature_dimension = dimension;
        self
    }

    /// 设置TF-IDF
    pub fn with_tfidf(mut self, use_tfidf: bool) -> Self {
        self.use_tfidf = use_tfidf;
        self
    }

    /// 设置词向量
    pub fn with_word_vectors(mut self, use_word_vectors: bool, dimension: usize) -> Self {
        self.use_word_vectors = use_word_vectors;
        self.word_vector_dimension = dimension;
        self
    }

    /// 设置词汇表大小
    pub fn with_vocabulary_size(mut self, max_size: usize, min_freq: usize) -> Self {
        self.max_vocabulary_size = max_size;
        self.min_word_frequency = min_freq;
        self
    }

    /// 设置停用词
    pub fn with_stopwords(mut self, stopwords: Vec<String>) -> Self {
        self.custom_stopwords = stopwords;
        self
    }

    /// 设置语言处理
    pub fn with_language_processing(mut self, language: String, stemming: bool, lemmatization: bool) -> Self {
        self.language = language;
        self.stemming = stemming;
        self.lemmatization = lemmatization;
        self
    }

    /// 设置文本处理
    pub fn with_text_processing(mut self, case_sensitive: bool, max_length: usize) -> Self {
        self.case_sensitive = case_sensitive;
        self.max_text_length = max_length;
        self
    }

    /// 设置性能参数
    pub fn with_performance(mut self, batch_size: usize, num_threads: usize, cache_size: usize) -> Self {
        self.batch_size = batch_size;
        self.num_threads = num_threads;
        self.cache_size = cache_size;
        self
    }

    /// 设置模型路径
    pub fn with_model_path(mut self, model_path: String) -> Self {
        self.model_path = Some(model_path);
        self
    }

    /// 添加元数据
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// 验证配置
    pub fn validate(&self) -> Result<(), String> {
        if self.feature_dimension == 0 {
            return Err("Feature dimension must be greater than 0".to_string());
        }
        if self.max_vocabulary_size == 0 {
            return Err("Max vocabulary size must be greater than 0".to_string());
        }
        if self.batch_size == 0 {
            return Err("Batch size must be greater than 0".to_string());
        }
        if self.num_threads == 0 {
            return Err("Number of threads must be greater than 0".to_string());
        }
        if self.max_text_length == 0 {
            return Err("Max text length must be greater than 0".to_string());
        }
        Ok(())
    }
}

/// 预处理配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// 是否转换为小写
    pub lowercase: bool,
    /// 是否去除停用词
    pub remove_stopwords: bool,
    /// 是否去除标点符号
    pub remove_punctuation: bool,
    /// 是否去除数字
    pub remove_numbers: bool,
    /// 是否去除HTML标签
    pub remove_html: bool,
    /// 是否进行词干提取
    pub stemming: bool,
    /// 是否进行词形还原
    pub lemmatization: bool,
    /// 语言设置
    pub language: String,
    /// 是否区分大小写
    pub case_sensitive: bool,
    /// 最大文本长度
    pub max_length: usize,
    /// 最小词长度
    pub min_word_length: usize,
    /// 最大词长度
    pub max_word_length: usize,
    /// 自定义停用词列表
    pub custom_stopwords: Vec<String>,
    /// 停用词文件路径
    pub stopwords_path: Option<String>,
    /// 正则表达式模式
    pub regex_patterns: Vec<String>,
    /// 替换模式
    pub replacement_patterns: Vec<(String, String)>,
    /// 是否保留原始文本
    pub preserve_original: bool,
    /// 配置元数据
    pub metadata: HashMap<String, String>,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_stopwords: true,
            remove_punctuation: false,
            remove_numbers: false,
            remove_html: true,
            stemming: false,
            lemmatization: false,
            language: "en".to_string(),
            case_sensitive: false,
            max_length: 1000,
            min_word_length: 2,
            max_word_length: 50,
            custom_stopwords: Vec::new(),
            stopwords_path: None,
            regex_patterns: Vec::new(),
            replacement_patterns: Vec::new(),
            preserve_original: false,
            metadata: HashMap::new(),
        }
    }
}

impl PreprocessingConfig {
    /// 创建新的预处理配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置大小写处理
    pub fn with_case_handling(mut self, lowercase: bool, case_sensitive: bool) -> Self {
        self.lowercase = lowercase;
        self.case_sensitive = case_sensitive;
        self
    }

    /// 设置文本清理
    pub fn with_text_cleaning(mut self, remove_punctuation: bool, remove_numbers: bool, remove_html: bool) -> Self {
        self.remove_punctuation = remove_punctuation;
        self.remove_numbers = remove_numbers;
        self.remove_html = remove_html;
        self
    }

    /// 设置停用词处理
    pub fn with_stopwords(mut self, remove_stopwords: bool, custom_stopwords: Vec<String>) -> Self {
        self.remove_stopwords = remove_stopwords;
        self.custom_stopwords = custom_stopwords;
        self
    }

    /// 设置语言处理
    pub fn with_language_processing(mut self, language: String, stemming: bool, lemmatization: bool) -> Self {
        self.language = language;
        self.stemming = stemming;
        self.lemmatization = lemmatization;
        self
    }

    /// 设置长度限制
    pub fn with_length_limits(mut self, max_length: usize, min_word_length: usize, max_word_length: usize) -> Self {
        self.max_length = max_length;
        self.min_word_length = min_word_length;
        self.max_word_length = max_word_length;
        self
    }

    /// 设置正则表达式模式
    pub fn with_regex_patterns(mut self, patterns: Vec<String>) -> Self {
        self.regex_patterns = patterns;
        self
    }

    /// 设置替换模式
    pub fn with_replacement_patterns(mut self, patterns: Vec<(String, String)>) -> Self {
        self.replacement_patterns = patterns;
        self
    }

    /// 设置停用词文件路径
    pub fn with_stopwords_path(mut self, path: String) -> Self {
        self.stopwords_path = Some(path);
        self
    }

    /// 添加元数据
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// 验证配置
    pub fn validate(&self) -> Result<(), String> {
        if self.max_length == 0 {
            return Err("Max length must be greater than 0".to_string());
        }
        if self.min_word_length == 0 {
            return Err("Min word length must be greater than 0".to_string());
        }
        if self.max_word_length < self.min_word_length {
            return Err("Max word length must be greater than or equal to min word length".to_string());
        }
        Ok(())
    }
}

/// 混合特征配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedFeatureConfig {
    /// 文本特征配置
    pub text_config: TextFeatureConfig,
    /// 数值特征配置
    pub numeric_config: NumericFeatureConfig,
    /// 分类特征配置
    pub categorical_config: CategoricalFeatureConfig,
    /// 特征融合策略
    pub fusion_strategy: FusionStrategy,
    /// 是否标准化
    pub normalize: bool,
    /// 是否进行特征选择
    pub feature_selection: bool,
    /// 特征选择方法
    pub selection_method: String,
    /// 选择的特征数量
    pub selected_feature_count: usize,
    /// 配置元数据
    pub metadata: HashMap<String, String>,
}

impl Default for MixedFeatureConfig {
    fn default() -> Self {
        Self {
            text_config: TextFeatureConfig::default(),
            numeric_config: NumericFeatureConfig::default(),
            categorical_config: CategoricalFeatureConfig::default(),
            fusion_strategy: FusionStrategy::Concatenation,
            normalize: true,
            feature_selection: false,
            selection_method: "variance".to_string(),
            selected_feature_count: 100,
            metadata: HashMap::new(),
        }
    }
}

/// 数值特征配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericFeatureConfig {
    /// 是否标准化
    pub normalize: bool,
    /// 标准化方法
    pub normalization_method: String,
    /// 是否处理缺失值
    pub handle_missing: bool,
    /// 缺失值填充方法
    pub missing_fill_method: String,
    /// 是否处理异常值
    pub handle_outliers: bool,
    /// 异常值检测方法
    pub outlier_detection_method: String,
    /// 是否进行特征缩放
    pub scale_features: bool,
    /// 缩放方法
    pub scaling_method: String,
    /// 配置元数据
    pub metadata: HashMap<String, String>,
}

impl Default for NumericFeatureConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            normalization_method: "zscore".to_string(),
            handle_missing: true,
            missing_fill_method: "mean".to_string(),
            handle_outliers: false,
            outlier_detection_method: "iqr".to_string(),
            scale_features: true,
            scaling_method: "minmax".to_string(),
            metadata: HashMap::new(),
        }
    }
}

/// 分类特征配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalFeatureConfig {
    /// 编码方法
    pub encoding_method: String,
    /// 是否处理缺失值
    pub handle_missing: bool,
    /// 缺失值编码
    pub missing_encoding: String,
    /// 是否处理未知值
    pub handle_unknown: bool,
    /// 未知值编码
    pub unknown_encoding: String,
    /// 是否进行特征哈希
    pub feature_hashing: bool,
    /// 哈希维度
    pub hash_dimension: usize,
    /// 配置元数据
    pub metadata: HashMap<String, String>,
}

impl Default for CategoricalFeatureConfig {
    fn default() -> Self {
        Self {
            encoding_method: "onehot".to_string(),
            handle_missing: true,
            missing_encoding: "missing".to_string(),
            handle_unknown: true,
            unknown_encoding: "unknown".to_string(),
            feature_hashing: false,
            hash_dimension: 100,
            metadata: HashMap::new(),
        }
    }
}

/// 特征融合策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// 简单拼接
    Concatenation,
    /// 加权融合
    Weighted,
    /// 注意力融合
    Attention,
    /// 自适应融合
    Adaptive,
    /// 自定义融合
    Custom(String),
}

impl Default for FusionStrategy {
    fn default() -> Self {
        Self::Concatenation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = TransformerConfig::default();
        assert_eq!(config.model_name, "default-transformer");
        assert_eq!(config.max_seq_length, 512);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.vocab_size, 30000);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.num_layers, 6);
    }

    #[test]
    fn test_config_new() {
        let config = TransformerConfig::new(
            "test-model".to_string(),
            "./test".to_string(),
            256,
            512,
            15000,
        );
        assert_eq!(config.model_name, "test-model");
        assert_eq!(config.max_seq_length, 256);
        assert_eq!(config.hidden_size, 512);
        assert_eq!(config.vocab_size, 15000);
    }

    #[test]
    fn test_config_validation() {
        let mut config = TransformerConfig::default();
        assert!(config.validate().is_ok());

        config.max_seq_length = 0;
        assert!(config.validate().is_err());

        config = TransformerConfig::default();
        config.hidden_size = 10;
        config.num_heads = 3;
        assert!(config.validate().is_err()); // 10不能被3整除
    }

    #[test]
    fn test_head_dim() {
        let config = TransformerConfig::default();
        assert_eq!(config.head_dim(), 768 / 12);
    }
} 