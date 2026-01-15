// Text Tokenizer Implementation
// 文本分词器实现

use crate::{Error, Result};
use crate::data::text_features::preprocessing::TextPreprocessor;
use std::collections::{HashSet, HashMap};
use unicode_segmentation::UnicodeSegmentation;
use regex::Regex;
use dyn_clone::DynClone;
use crate::data::text_features::pipeline::ProcessingStageType;
use crate::data::text_features::preprocessing::TextProcessor;

/// 分词器特征
pub trait Tokenizer: Send + Sync + DynClone {
    /// 将文本分词
    fn tokenize(&self, text: &str) -> Result<Vec<String>>;
    
    /// 获取分词器名称
    fn name(&self) -> &str;
}

dyn_clone::clone_trait_object!(Tokenizer);

/// 文本分词器
pub struct TokenizerImpl {
    /// 分词器配置
    config: TokenizerConfig,
    /// 停用词集合
    stopwords: Option<HashSet<String>>,
    tokenizer: Option<Box<dyn Tokenizer>>,
}

/// 分词器配置
#[derive(Clone, Debug)]
pub struct TokenizerConfig {
    /// 分词方式
    pub tokenize_method: TokenizeMethod,
    /// 是否移除停用词
    pub remove_stopwords: bool,
    /// 是否区分大小写
    pub case_sensitive: bool,
    /// 语言
    pub language: String,
    /// 最小词长度
    pub min_token_length: usize,
    /// 自定义分隔符
    pub custom_delimiter: Option<String>,
}

/// 分词方式
#[derive(Clone, Debug, PartialEq)]
pub enum TokenizeMethod {
    /// 按空格分词
    Whitespace,
    /// 按NGram分词
    NGram(usize, usize),
    /// 按字符分词
    Character,
    /// 按自定义分隔符分词
    Delimiter,
    /// 按Unicode单元分词
    Unicode,
    /// 基于正则表达式的分词
    Regex(String),
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            tokenize_method: TokenizeMethod::Whitespace,
            remove_stopwords: true,
            case_sensitive: false,
            language: "en".to_string(),
            min_token_length: 1,
            custom_delimiter: None,
        }
    }
}

impl TokenizerImpl {
    /// 创建新的分词器
    pub fn new() -> Self {
        Self {
            config: TokenizerConfig::default(),
            stopwords: None,
            tokenizer: None,
        }
    }
    
    /// 使用指定配置创建分词器
    pub fn with_config(config: TokenizerConfig) -> Self {
        Self {
            config,
            stopwords: None,
            tokenizer: None,
        }
    }
    
    /// 设置停用词
    pub fn with_stopwords(mut self, stopwords: HashSet<String>) -> Self {
        self.stopwords = Some(stopwords);
        self
    }
    
    /// 分词处理
    fn tokenize(&self, text: &str) -> Vec<String> {
        let processed_text = if !self.config.case_sensitive {
            text.to_lowercase()
        } else {
            text.to_string()
        };
        
        let tokens = match self.config.tokenize_method {
            TokenizeMethod::Whitespace => {
                processed_text.split_whitespace()
                    .map(|s| s.to_string())
                    .collect()
            },
            TokenizeMethod::NGram(min_n, max_n) => {
                self.ngram_tokenize(&processed_text, min_n, max_n)
            },
            TokenizeMethod::Character => {
                processed_text.chars()
                    .map(|c| c.to_string())
                    .collect()
            },
            TokenizeMethod::Delimiter => {
                if let Some(delimiter) = &self.config.custom_delimiter {
                    processed_text.split(delimiter)
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                        .collect()
                } else {
                    processed_text.split_whitespace()
                        .map(|s| s.to_string())
                        .collect()
                }
            },
            TokenizeMethod::Unicode => {
                processed_text.graphemes(true)
                    .map(|g| g.to_string())
                    .collect()
            }
            TokenizeMethod::Regex(ref pattern) => {
                let re = Regex::new(pattern).unwrap_or_else(|_| Regex::new(r"\w+").unwrap());
                re.find_iter(&processed_text)
                    .map(|m| m.as_str().to_string())
                    .collect()
            }
        };
        
        // 处理后的令牌
        let filtered_tokens = tokens.into_iter()
            .filter(|token| {
                // 过滤长度过短的令牌
                if token.len() < self.config.min_token_length {
                    return false;
                }
                
                // 过滤停用词
                if self.config.remove_stopwords {
                    if let Some(stopwords) = &self.stopwords {
                        let t = if self.config.case_sensitive {
                            token.clone()
                        } else {
                            token.to_lowercase()
                        };
                        
                        if stopwords.contains(&t) {
                            return false;
                        }
                    }
                }
                
                true
            })
            .collect();
            
        filtered_tokens
    }
    
    /// NGram分词
    fn ngram_tokenize(&self, text: &str, min_n: usize, max_n: usize) -> Vec<String> {
        let mut result = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        
        for n in min_n..=max_n {
            if n > chars.len() {
                continue;
            }
            
            for i in 0..=chars.len() - n {
                let ngram: String = chars[i..i+n].iter().collect();
                result.push(ngram);
            }
        }
        
        result
    }
    
    /// 获取默认停用词
    fn get_default_stopwords(&self) -> HashSet<String> {
        match self.config.language.as_str() {
            "en" => {
                vec![
                    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
                    "be", "been", "being", "have", "has", "had", "do", "does", "did",
                    "will", "would", "shall", "should", "can", "could", "may", "might",
                    "must", "to", "of", "in", "on", "at", "by", "for", "with", "about",
                    "against", "between", "into", "through", "during", "before", "after",
                    "above", "below", "from", "up", "down", "out", "off", "over", "under",
                    "again", "further", "then", "once", "here", "there", "when", "where",
                    "why", "how", "all", "any", "both", "each", "few", "more", "most",
                    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
                    "so", "than", "too", "very", "s", "t", "just", "don", "now",
                ]
            },
            "zh" => {
                vec![
                    "的", "了", "和", "是", "在", "我", "有", "这", "个", "们",
                    "中", "也", "就", "来", "到", "你", "说", "为", "着", "如",
                    "那", "要", "自", "以", "会", "对", "可", "她", "里", "所",
                    "他", "而", "么", "去", "之", "于", "把", "等", "被", "一",
                    "没", "什", "麼", "这个", "那个", "只是", "因为", "所以", "但是", "如果",
                ]
            },
            _ => vec![]
        }.into_iter().map(|s| s.to_string()).collect()
    }
}

impl TextProcessor for TokenizerImpl {
    fn process(&self, text: &str) -> Result<String> {
        // tokenize 方法返回 Vec<String>，不是 Result
        let tokens = self.tokenize(text);
        Ok(tokens.join(" "))
    }
    
    fn name(&self) -> &str {
        "tokenizer"
    }
    
    fn processor_type(&self) -> ProcessingStageType {
        ProcessingStageType::Tokenization
    }
    
    fn box_clone(&self) -> Box<dyn TextProcessor> {
        Box::new(TokenizerImpl {
            config: self.config.clone(),
            stopwords: self.stopwords.clone(),
            tokenizer: None, // tokenizer 是 Box<dyn Tokenizer>，无法克隆，设为 None
        })
    }
}

/// 默认分词器
#[derive(Clone)]
pub struct DefaultTokenizer {
    name: String,
}

impl DefaultTokenizer {
    /// 创建新的默认分词器
    pub fn new() -> Self {
        Self {
            name: "DefaultTokenizer".to_string(),
        }
    }
}

impl Tokenizer for DefaultTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        // 简单按空白字符分词
        Ok(text.split_whitespace()
            .map(|s| s.to_string())
            .collect())
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// N-Gram分词器
#[derive(Clone)]
pub struct NgramTokenizer {
    name: String,
    min_n: usize,
    max_n: usize,
}

impl NgramTokenizer {
    /// 创建新的N-Gram分词器
    pub fn new(min_n: usize, max_n: usize) -> Result<Self> {
        if min_n > max_n || min_n == 0 {
            return Err(Error::InvalidArgument(
                format!("无效的n-gram范围: {} - {}", min_n, max_n)
            ));
        }
        
        Ok(Self {
            name: format!("NgramTokenizer({},{})", min_n, max_n),
            min_n,
            max_n,
        })
    }
}

impl Tokenizer for NgramTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut ngrams = Vec::new();
        
        for n in self.min_n..=self.max_n {
            if n > words.len() {
                continue;
            }
            
            for i in 0..=(words.len() - n) {
                let ngram = words[i..(i + n)].join(" ");
                ngrams.push(ngram);
            }
        }
        
        Ok(ngrams)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// 字符N-Gram分词器
#[derive(Clone)]
pub struct CharNgramTokenizer {
    name: String,
    min_n: usize,
    max_n: usize,
}

impl CharNgramTokenizer {
    /// 创建新的字符N-Gram分词器
    pub fn new(min_n: usize, max_n: usize) -> Result<Self> {
        if min_n > max_n || min_n == 0 {
            return Err(Error::InvalidArgument(
                format!("无效的字符n-gram范围: {} - {}", min_n, max_n)
            ));
        }
        
        Ok(Self {
            name: format!("CharNgramTokenizer({},{})", min_n, max_n),
            min_n,
            max_n,
        })
    }
}

impl Tokenizer for CharNgramTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        let chars: Vec<char> = text.chars().collect();
        let mut ngrams = Vec::new();
        
        for n in self.min_n..=self.max_n {
            if n > chars.len() {
                continue;
            }
            
            for i in 0..=(chars.len() - n) {
                let ngram: String = chars[i..(i + n)].iter().collect();
                ngrams.push(ngram);
            }
        }
        
        Ok(ngrams)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// 自定义分词器
#[derive(Clone)]
pub struct CustomTokenizer {
    name: String,
    path: String,
}

impl CustomTokenizer {
    /// 创建新的自定义分词器
    pub fn new(path: &str) -> Self {
        Self {
            name: "CustomTokenizer".to_string(),
            path: path.to_string(),
        }
    }
}

impl Tokenizer for CustomTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        // 简单实现，实际应加载自定义分词逻辑
        Ok(text.split_whitespace()
            .map(|s| s.to_string())
            .collect())
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

// WhitespaceTokenizer实现
#[derive(Clone)]
pub struct WhitespaceTokenizer {
    name: String,
}

impl WhitespaceTokenizer {
    pub fn new() -> Self {
        Self {
            name: "whitespace_tokenizer".to_string(),
        }
    }
}

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        Ok(text.split_whitespace().map(|s| s.to_string()).collect())
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

// WordTokenizer实现
#[derive(Clone)]
pub struct WordTokenizer {
    name: String,
}

impl WordTokenizer {
    pub fn new() -> Self {
        Self {
            name: "word_tokenizer".to_string(),
        }
    }
}

impl Tokenizer for WordTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        // 使用简单的正则表达式来分词，仅保留字母和数字组成的词
        let words: Vec<String> = text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        Ok(words)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

// CharTokenizer实现
#[derive(Clone)]
pub struct CharTokenizer {
    name: String,
}

impl CharTokenizer {
    pub fn new() -> Self {
        Self {
            name: "char_tokenizer".to_string(),
        }
    }
}

impl Tokenizer for CharTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        Ok(text.chars().map(|c| c.to_string()).collect())
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

// WordPieceTokenizer实现
#[derive(Clone)]
pub struct WordPieceTokenizer {
    name: String,
    vocab: Option<HashMap<String, usize>>,
    unk_token: String,
    max_input_chars_per_word: usize,
    with_offsets: bool,
}

impl WordPieceTokenizer {
    pub fn new(
        vocab: Option<HashMap<String, usize>>, 
        unk_token: String, 
        max_input_chars_per_word: usize, 
        with_offsets: bool
    ) -> Self {
        Self {
            name: "wordpiece_tokenizer".to_string(),
            vocab,
            unk_token,
            max_input_chars_per_word,
            with_offsets,
        }
    }
}

impl Tokenizer for WordPieceTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        // 基础实现：使用空格分词
        // 完整的 WordPiece 算法需要词汇表和子词切分逻辑
        // 生产环境应集成 transformers 库或加载预训练 tokenizer
        let words = text.split_whitespace().map(|s| s.to_string()).collect();
        Ok(words)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

// SentencePieceTokenizer实现
#[derive(Clone)]
pub struct SentencePieceTokenizer {
    name: String,
    model_path: String,
    vocab_size: usize,
}

impl SentencePieceTokenizer {
    pub fn new(model_path: &str, vocab_size: usize) -> Self {
        Self {
            name: "sentencepiece_tokenizer".to_string(),
            model_path: model_path.to_string(),
            vocab_size,
        }
    }
}

impl Tokenizer for SentencePieceTokenizer {
    fn tokenize(&self, text: &str) -> Result<Vec<String>> {
        // 基础实现：使用空格分词
        // 完整的 SentencePiece 需要加载 .model 文件并使用 sentencepiece-rs 库
        // 生产环境应调用 sentencepiece 库进行子词切分
        let words = text.split_whitespace().map(|s| s.to_string()).collect();
        Ok(words)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
} 