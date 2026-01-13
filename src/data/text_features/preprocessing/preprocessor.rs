// Text Preprocessor Implementation
// 文本预处理器实现

use crate::data::text_features::preprocessing::cleaner::TextCleaner;
use crate::data::text_features::preprocessing::normalizer::TextNormalizer;
use crate::data::text_features::preprocessing::tokenizer::Tokenizer;
use crate::data::text_features::preprocessing::filter::TextFilter;
use crate::data::text_features::preprocessing::TextProcessor;
use crate::data::text_features::preprocessing::text_preprocessor_trait::TextPreprocessor;
use crate::Result;
use std::sync::Arc;
use log::warn;

/// 预处理配置
#[derive(Clone, Debug, Default)]
pub struct PreprocessingConfig {
    /// 清洁策略
    pub cleaning_strategies: Option<Vec<String>>,
    /// 规范化策略
    pub normalization_strategies: Option<Vec<String>>,
    /// 使用NGram分词
    pub use_ngrams: bool,
    /// NGram范围
    pub ngram_range: Option<(usize, usize)>,
    /// 使用字符NGram
    pub use_char_ngrams: bool,
    /// 字符NGram范围
    pub char_ngram_range: (usize, usize),
    /// 使用过滤
    pub use_filtering: bool,
    /// 移除停用词
    pub remove_stopwords: bool,
    /// 最小令牌长度
    pub min_token_length: usize,
    /// 最大令牌长度
    pub max_token_length: Option<usize>,
    /// 语言
    pub language: String,
}

/// 文本预处理器
pub struct TextPreprocessorImpl {
    /// 配置
    config: PreprocessingConfig,
    /// 预处理流水线
    pipeline: super::PreprocessingPipeline,
    /// 分词器
    tokenizer: Option<Box<dyn Tokenizer>>,
}

impl TextPreprocessorImpl {
    /// 创建新的文本预处理器
    pub fn new(config: Option<PreprocessingConfig>) -> Result<Self> {
        let config = config.unwrap_or_default();
        let pipeline = super::PreprocessingPipeline::from_preprocessing_config(&config);
        
        // 创建分词器
        let tokenizer = if config.use_ngrams {
            let range = config.ngram_range.unwrap_or((1, 3));
            Some(super::create_tokenizer("ngram"))
        } else {
            Some(super::create_tokenizer("whitespace"))
        };
        
        Ok(Self {
            config,
            pipeline,
            tokenizer,
        })
    }
    
    /// 分词
    pub fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        if let Some(tokenizer) = &self.tokenizer {
            tokenizer.tokenize(text)
        } else {
            // 默认空格分词
            Ok(text.split_whitespace().map(String::from).collect())
        }
    }
    
    /// 预处理并分词
    pub fn preprocess_and_tokenize(&self, text: &str) -> Result<Vec<String>> {
        let processed = self.preprocess(text)?;
        self.tokenize(&processed)
    }
}

impl TextPreprocessor for TextPreprocessorImpl {
    fn preprocess(&self, text: &str) -> Result<String> {
        self.pipeline.process(text)
    }
    
    fn name(&self) -> &str {
        "text_preprocessor"
    }
}

/// 文本处理器适配器 - 用于连接旧系统和新系统
pub struct TextProcessorAdapter {
    /// 预处理器
    preprocessor: Arc<dyn TextPreprocessor>,
    /// 名称
    name: String,
}

impl TextProcessorAdapter {
    /// 创建新的适配器
    pub fn new(preprocessor: Arc<dyn TextPreprocessor>, name: &str) -> Self {
        Self {
            preprocessor,
            name: name.to_string(),
        }
    }
}

impl TextProcessor for TextProcessorAdapter {
    fn process(&self, text: &str) -> Result<String> {
        self.preprocessor.preprocess(text)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn processor_type(&self) -> crate::data::text_features::pipeline::ProcessingStageType {
        crate::data::text_features::pipeline::ProcessingStageType::Transformation
    }
    
    fn box_clone(&self) -> Box<dyn TextProcessor> {
        Box::new(Self {
            preprocessor: self.preprocessor.clone(),
            name: self.name.clone(),
        })
    }
}

/// 创建共享预处理器
pub fn create_shared_preprocessor(config: Option<PreprocessingConfig>) -> Arc<dyn TextPreprocessor> {
    match TextPreprocessorImpl::new(config) {
        Ok(preprocessor) => Arc::new(preprocessor),
        Err(e) => {
            warn!("创建预处理器失败: {}", e);
            // 创建默认预处理器
            Arc::new(TextPreprocessorImpl::new(None).unwrap())
        }
    }
}

/// 创建默认预处理器
pub fn create_default_preprocessor() -> Arc<dyn TextPreprocessor> {
    create_shared_preprocessor(None)
} 