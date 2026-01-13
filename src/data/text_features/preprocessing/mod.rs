// Text Preprocessing Module
// 文本预处理模块

// 子模块
pub mod cleaner;
pub mod normalizer;
pub mod tokenizer;
pub mod filter;
pub mod preprocessor;
pub mod text_preprocessor_trait;

// 重导出
pub use self::cleaner::{TextCleaner, StandardTextCleaner};
pub use self::normalizer::{TextNormalizer, StandardTextNormalizer};
pub use self::tokenizer::{Tokenizer, TokenizerImpl, TokenizerConfig, TokenizeMethod};
pub use self::filter::{TextFilter, LengthFilter, StopwordFilter};
pub use self::preprocessor::{TextPreprocessorImpl, PreprocessingConfig, TextProcessorAdapter};
pub use self::text_preprocessor_trait::TextPreprocessor;

use crate::Result;
use log::{debug, info};
use std::sync::Arc;
use std::collections::HashMap;

/// 预处理流水线
pub struct PreprocessingPipeline {
    /// 清洁器集合
    cleaners: Vec<Box<dyn TextCleaner>>,
    /// 规范化器集合
    normalizers: Vec<Box<dyn TextNormalizer>>,
    /// 分词器
    tokenizer: Option<Box<dyn Tokenizer>>,
    /// 过滤器集合
    filters: Vec<Box<dyn TextFilter>>,
}

impl PreprocessingPipeline {
    /// 创建新的预处理流水线
    pub fn new() -> Self {
        Self {
            cleaners: Vec::new(),
            normalizers: Vec::new(),
            tokenizer: None,
            filters: Vec::new(),
        }
    }
    
    /// 添加清洁器
    pub fn add_cleaner<T: TextCleaner + 'static>(&mut self, cleaner: T) {
        self.cleaners.push(Box::new(cleaner));
    }
    
    /// 添加规范化器
    pub fn add_normalizer<T: TextNormalizer + 'static>(&mut self, normalizer: T) {
        self.normalizers.push(Box::new(normalizer));
    }
    
    /// 设置分词器
    pub fn set_tokenizer<T: Tokenizer + 'static>(&mut self, tokenizer: T) {
        self.tokenizer = Some(Box::new(tokenizer));
    }
    
    /// 添加过滤器
    pub fn add_filter<T: TextFilter + 'static>(&mut self, filter: T) {
        self.filters.push(Box::new(filter));
    }
    
    /// 执行预处理流水线
    pub fn process(&self, text: &str) -> Result<String> {
        debug!("开始文本预处理, 长度: {}", text.len());
        
        // 1. 清洁文本
        let mut processed = text.to_string();
        for cleaner in &self.cleaners {
            processed = cleaner.clean(&processed)?;
        }
        
        // 2. 规范化
        for normalizer in &self.normalizers {
            processed = normalizer.normalize(&processed)?;
        }
        
        // 3. 分词
        let mut tokens = if let Some(tokenizer) = &self.tokenizer {
            tokenizer.tokenize(&processed)?
        } else {
            // 默认按空格分词
            processed.split_whitespace().map(String::from).collect()
        };
        
        // 4. 过滤
        for filter in &self.filters {
            tokens = filter.filter(&tokens)?;
        }
        
        // 5. 重组为文本
        let result = tokens.join(" ");
        
        debug!("预处理完成, 结果长度: {}", result.len());
        info!("文本预处理: 原始长度 {} -> 处理后长度 {}", text.len(), result.len());
        
        Ok(result)
    }
    
    /// 处理一批文本
    pub fn batch_process(&self, texts: &[&str]) -> Result<Vec<String>> {
        texts.iter().map(|&text| self.process(text)).collect()
    }
    
    /// 获取默认实例
    pub fn default() -> Self {
        Self::new()
    }
    
    /// 从预处理配置创建
    pub fn from_preprocessing_config(config: &PreprocessingConfig) -> Self {
        let mut pipeline = Self::default();
        
        // 添加清洁器
        if let Some(cleaning_strategies) = &config.cleaning_strategies {
            let cleaners: Vec<Box<dyn TextCleaner>> = cleaning_strategies.iter()
                .map(|_| Box::new(StandardTextCleaner::new()) as Box<dyn TextCleaner>)
                .collect();
            for cleaner in cleaners {
                pipeline.cleaners.push(cleaner);
            }
        }
        
        // 添加规范化器
        if let Some(normalization_strategies) = &config.normalization_strategies {
            let normalizers: Vec<Box<dyn TextNormalizer>> = normalization_strategies.iter()
                .map(|_| Box::new(StandardTextNormalizer::new()) as Box<dyn TextNormalizer>)
                .collect();
            for normalizer in normalizers {
                pipeline.normalizers.push(normalizer);
            }
        }
        
        // 设置分词器
        if config.use_ngrams {
            let ngram_range = config.ngram_range.unwrap_or((1, 3));
            if let Ok(tokenizer) = tokenizer::NgramTokenizer::new(ngram_range.0, ngram_range.1) {
                pipeline.set_tokenizer(tokenizer);
            }
        } else {
            pipeline.set_tokenizer(tokenizer::DefaultTokenizer::new());
        }
        
        // 添加过滤器
        if config.use_filtering {
            if config.remove_stopwords {
                pipeline.add_filter(StopwordFilter::new(config.language.clone()));
            }
            
            if config.min_token_length > 0 {
                pipeline.add_filter(LengthFilter::new(config.min_token_length, config.max_token_length));
            }
        }
        
        pipeline
    }
}

impl TextProcessor for PreprocessingPipeline {
    fn process(&self, text: &str) -> Result<String> {
        self.process(text)
    }
    
    fn name(&self) -> &str {
        "PreprocessingPipeline"
    }
    
    fn processor_type(&self) -> crate::data::text_features::pipeline::ProcessingStageType {
        crate::data::text_features::pipeline::ProcessingStageType::Custom
    }
    
    fn box_clone(&self) -> Box<dyn TextProcessor> {
        // 由于 Box<dyn TextCleaner> 等无法直接克隆，我们创建一个新的空 pipeline
        // 如果需要完整克隆，需要为这些 trait 添加 DynClone 支持
        Box::new(Self {
            cleaners: Vec::new(),
            normalizers: Vec::new(),
            tokenizer: None,
            filters: Vec::new(),
        })
    }
}

/// 文本处理器特征
pub trait TextProcessor {
    /// 处理文本
    fn process(&self, text: &str) -> Result<String>;
    
    /// 处理文本并返回附加信息
    fn process_with_metadata(&self, text: &str) -> Result<(String, HashMap<String, String>)> {
        let processed = self.process(text)?;
        Ok((processed, HashMap::new()))
    }
    
    /// 获取处理器名称
    fn name(&self) -> &str;
    
    /// 获取处理器类型
    fn processor_type(&self) -> crate::data::text_features::pipeline::ProcessingStageType;
    
    /// 克隆处理器
    fn box_clone(&self) -> Box<dyn TextProcessor>;
}

impl Clone for Box<dyn TextProcessor> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

/// 创建分词器
pub fn create_tokenizer<T: AsRef<str>>(strategy: T) -> Box<dyn Tokenizer> {
    match strategy.as_ref().to_lowercase().as_str() {
        "whitespace" => Box::new(tokenizer::WhitespaceTokenizer::new()),
        "word" => Box::new(tokenizer::WordTokenizer::new()),
        "char" => Box::new(tokenizer::CharTokenizer::new()),
        "wordpiece" => Box::new(tokenizer::WordPieceTokenizer::new(
            None, "[UNK]".to_string(), 100, true
        )),
        "sentencepiece" => Box::new(tokenizer::SentencePieceTokenizer::new("", 16000)),
        _ => Box::new(tokenizer::DefaultTokenizer::new()),
    }
}

/// 创建预处理器
pub fn create_preprocessor(config: Option<PreprocessingConfig>) -> Arc<dyn TextPreprocessor> {
    preprocessor::create_shared_preprocessor(config)
} 