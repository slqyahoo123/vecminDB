// 文本处理器实现
// 提供各类文本处理功能，用于高级文本处理管道

use crate::Result;
use crate::Error;
use crate::data::text_features::pipeline::{TextProcessor, ProcessingStageType};
use std::collections::HashMap;
use std::sync::Arc;
use regex::Regex;
use serde::{Serialize, Deserialize};

/// HTML清理处理器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HtmlCleanerConfig {
    /// 是否移除HTML标签
    pub remove_html: bool,
    /// 是否移除脚本
    pub remove_scripts: bool,
    /// 是否移除样式
    pub remove_styles: bool,
    /// 是否保留特定标签内容
    pub preserve_tags: Vec<String>,
    /// 是否解码HTML实体
    pub decode_entities: bool,
}

impl Default for HtmlCleanerConfig {
    fn default() -> Self {
        Self {
            remove_html: true,
            remove_scripts: true,
            remove_styles: true,
            preserve_tags: vec!["p".to_string(), "a".to_string(), "h1".to_string(), "h2".to_string(), "h3".to_string()],
            decode_entities: true,
        }
    }
}

/// HTML清理处理器
pub struct HtmlCleaner {
    /// 配置
    config: HtmlCleanerConfig,
    /// 正则表达式缓存
    regex_cache: HashMap<String, Regex>,
    /// 名称
    name: String,
}

impl HtmlCleaner {
    /// 创建新的HTML清理处理器
    pub fn new(config: HtmlCleanerConfig, name: Option<String>) -> Result<Self> {
        let mut regex_cache = HashMap::new();
        
        // 预编译常用正则表达式
        if config.remove_scripts {
            regex_cache.insert(
                "script".to_string(),
                Regex::new(r"<script\b[^>]*>([\s\S]*?)<\/script>").unwrap(),
            );
        }
        
        if config.remove_styles {
            regex_cache.insert(
                "style".to_string(),
                Regex::new(r"<style\b[^>]*>([\s\S]*?)<\/style>").unwrap(),
            );
        }
        
        if config.remove_html {
            regex_cache.insert(
                "html".to_string(),
                Regex::new(r"<[^>]*>").unwrap(),
            );
        }
        
        Ok(Self {
            config,
            regex_cache,
            name: name.unwrap_or_else(|| "HTMLCleaner".to_string()),
        })
    }
    
    /// 解码HTML实体
    fn decode_html_entities(&self, text: &str) -> String {
        let mut result = text.to_string();
        
        // 常见HTML实体解码
        let entities = [
            ("&amp;", "&"),
            ("&lt;", "<"),
            ("&gt;", ">"),
            ("&quot;", "\""),
            ("&apos;", "'"),
            ("&nbsp;", " "),
        ];
        
        for (entity, replacement) in entities.iter() {
            result = result.replace(entity, replacement);
        }
        
        // 十进制和十六进制编码
        let decimal_regex = Regex::new(r"&#(\d+);").unwrap();
        result = decimal_regex.replace_all(&result, |caps: &regex::Captures| {
            let code = caps[1].parse::<u32>().unwrap_or(0);
            match std::char::from_u32(code) {
                Some(c) => c.to_string(),
                None => caps[0].to_string(),
            }
        }).to_string();
        
        let hex_regex = Regex::new(r"&#x([0-9a-fA-F]+);").unwrap();
        result = hex_regex.replace_all(&result, |caps: &regex::Captures| {
            let code = u32::from_str_radix(&caps[1], 16).unwrap_or(0);
            match std::char::from_u32(code) {
                Some(c) => c.to_string(),
                None => caps[0].to_string(),
            }
        }).to_string();
        
        result
    }
}

impl TextProcessor for HtmlCleaner {
    fn process(&self, text: &str) -> Result<String> {
        let mut result = text.to_string();
        
        // 移除脚本
        if self.config.remove_scripts {
            if let Some(re) = self.regex_cache.get("script") {
                result = re.replace_all(&result, "").to_string();
            }
        }
        
        // 移除样式
        if self.config.remove_styles {
            if let Some(re) = self.regex_cache.get("style") {
                result = re.replace_all(&result, "").to_string();
            }
        }
        
        // 移除HTML标签
        if self.config.remove_html {
            if let Some(re) = self.regex_cache.get("html") {
                result = re.replace_all(&result, "").to_string();
            }
        }
        
        // 解码HTML实体
        if self.config.decode_entities {
            result = self.decode_html_entities(&result);
        }
        
        // 规范化空白
        result = result.replace('\n', " ");
        result = Regex::new(r"\s+").unwrap().replace_all(&result, " ").to_string();
        result = result.trim().to_string();
        
        Ok(result)
    }
    
    fn process_with_metadata(&self, text: &str) -> Result<(String, HashMap<String, String>)> {
        let processed = self.process(text)?;
        
        let mut metadata = HashMap::new();
        metadata.insert("original_length".to_string(), text.len().to_string());
        metadata.insert("processed_length".to_string(), processed.len().to_string());
        metadata.insert("html_removed".to_string(), self.config.remove_html.to_string());
        
        Ok((processed, metadata))
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn processor_type(&self) -> ProcessingStageType {
        ProcessingStageType::Cleaning
    }
    
    fn box_clone(&self) -> Box<dyn TextProcessor> {
        Box::new(Self {
            config: self.config.clone(),
            regex_cache: self.regex_cache.clone(),
            name: self.name.clone(),
        })
    }
}

/// 文本规范化处理器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextNormalizerConfig {
    /// 是否转为小写
    pub lowercase: bool,
    /// 是否去除重音符号
    pub remove_accents: bool,
    /// 是否规范化空白
    pub normalize_whitespace: bool,
    /// 是否去除标点符号
    pub remove_punctuation: bool,
    /// 是否移除数字
    pub remove_numbers: bool,
    /// 是否规范化新行
    pub normalize_newlines: bool,
}

impl Default for TextNormalizerConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_accents: false,
            normalize_whitespace: true,
            remove_punctuation: false,
            remove_numbers: false,
            normalize_newlines: true,
        }
    }
}

/// 文本规范化处理器
pub struct TextNormalizer {
    /// 配置
    config: TextNormalizerConfig,
    /// 正则表达式缓存
    regex_cache: HashMap<String, Regex>,
    /// 名称
    name: String,
}

impl TextNormalizer {
    /// 创建新的文本规范化处理器
    pub fn new(config: TextNormalizerConfig, name: Option<String>) -> Result<Self> {
        let mut regex_cache = HashMap::new();
        
        // 预编译常用正则表达式
        if config.normalize_whitespace {
            regex_cache.insert(
                "whitespace".to_string(),
                Regex::new(r"\s+").unwrap(),
            );
        }
        
        if config.remove_punctuation {
            regex_cache.insert(
                "punctuation".to_string(),
                Regex::new(r"[^\w\s]").unwrap(),
            );
        }
        
        if config.remove_numbers {
            regex_cache.insert(
                "numbers".to_string(),
                Regex::new(r"\d+").unwrap(),
            );
        }
        
        Ok(Self {
            config,
            regex_cache,
            name: name.unwrap_or_else(|| "TextNormalizer".to_string()),
        })
    }
    
    /// 去除重音符号
    fn remove_accents(&self, text: &str) -> String {
        // 注意：此实现较为简化，完整版应使用更全面的字符映射
        let mut result = text.to_string();
        
        // 常见重音字符映射
        let accent_map = [
            ("á", "a"), ("à", "a"), ("â", "a"), ("ä", "a"), ("ã", "a"), ("å", "a"),
            ("é", "e"), ("è", "e"), ("ê", "e"), ("ë", "e"),
            ("í", "i"), ("ì", "i"), ("î", "i"), ("ï", "i"),
            ("ó", "o"), ("ò", "o"), ("ô", "o"), ("ö", "o"), ("õ", "o"),
            ("ú", "u"), ("ù", "u"), ("û", "u"), ("ü", "u"),
            ("ý", "y"), ("ÿ", "y"),
            ("ç", "c"), ("ñ", "n"),
        ];
        
        for (accented, plain) in accent_map.iter() {
            result = result.replace(accented, plain);
            // 处理大写形式
            result = result.replace(&accented.to_uppercase(), &plain.to_uppercase());
        }
        
        result
    }
}

impl TextProcessor for TextNormalizer {
    fn process(&self, text: &str) -> Result<String> {
        let mut result = text.to_string();
        
        // 转为小写
        if self.config.lowercase {
            result = result.to_lowercase();
        }
        
        // 去除重音符号
        if self.config.remove_accents {
            result = self.remove_accents(&result);
        }
        
        // 规范化空白
        if self.config.normalize_whitespace {
            if let Some(re) = self.regex_cache.get("whitespace") {
                result = re.replace_all(&result, " ").to_string();
            }
            result = result.trim().to_string();
        }
        
        // 去除标点符号
        if self.config.remove_punctuation {
            if let Some(re) = self.regex_cache.get("punctuation") {
                result = re.replace_all(&result, "").to_string();
            }
        }
        
        // 移除数字
        if self.config.remove_numbers {
            if let Some(re) = self.regex_cache.get("numbers") {
                result = re.replace_all(&result, "").to_string();
            }
        }
        
        // 规范化新行
        if self.config.normalize_newlines {
            result = result.replace('\r', "");
            result = result.replace("\n\n", "\n");
        }
        
        Ok(result)
    }
    
    fn process_with_metadata(&self, text: &str) -> Result<(String, HashMap<String, String>)> {
        let processed = self.process(text)?;
        
        let mut metadata = HashMap::new();
        metadata.insert("original_length".to_string(), text.len().to_string());
        metadata.insert("processed_length".to_string(), processed.len().to_string());
        metadata.insert("lowercase".to_string(), self.config.lowercase.to_string());
        
        if self.config.remove_punctuation {
            // 计算移除的标点符号数量
            let original_punctuation_count = Regex::new(r"[^\w\s]").unwrap().find_iter(text).count();
            metadata.insert("removed_punctuation".to_string(), original_punctuation_count.to_string());
        }
        
        if self.config.remove_numbers {
            // 计算移除的数字数量
            let original_number_count = Regex::new(r"\d+").unwrap().find_iter(text).count();
            metadata.insert("removed_numbers".to_string(), original_number_count.to_string());
        }
        
        Ok((processed, metadata))
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn processor_type(&self) -> ProcessingStageType {
        ProcessingStageType::Normalization
    }
    
    fn box_clone(&self) -> Box<dyn TextProcessor> {
        Box::new(Self {
            config: self.config.clone(),
            regex_cache: self.regex_cache.clone(),
            name: self.name.clone(),
        })
    }
}

/// 分词处理器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// 分词模式
    pub mode: TokenizationMode,
    /// 是否小写化
    pub lowercase: bool,
    /// 是否保留停用词
    pub keep_stopwords: bool,
    /// 最小词长度
    pub min_token_length: usize,
    /// 最大词长度
    pub max_token_length: Option<usize>,
    /// 分隔符（仅用于Split模式）
    pub split_pattern: Option<String>,
}

/// 分词模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenizationMode {
    /// 简单空格分割
    Whitespace,
    /// 自定义分隔符分割
    Split,
    /// 正则表达式分割
    Regex,
    /// Word Piece分词
    WordPiece,
    /// BPE分词
    BPE,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            mode: TokenizationMode::Whitespace,
            lowercase: true,
            keep_stopwords: false,
            min_token_length: 1,
            max_token_length: None,
            split_pattern: None,
        }
    }
}

/// 分词处理器
pub struct Tokenizer {
    /// 配置
    config: TokenizerConfig,
    /// 停用词集合
    stopwords: HashSet<String>,
    /// 词表（用于WordPiece和BPE）
    vocabulary: Option<HashMap<String, usize>>,
    /// 名称
    name: String,
}

impl Tokenizer {
    /// 创建新的分词处理器
    pub fn new(config: TokenizerConfig, name: Option<String>) -> Result<Self> {
        // 初始化默认英文停用词
        let default_stopwords = vec![
            "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
            "at", "by", "for", "with", "about", "against", "between", "into",
            "through", "during", "before", "after", "above", "below", "to", "from",
            "up", "down", "in", "out", "on", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "any", "both", "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't",
            "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
            "aren't", "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't",
            "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't",
            "ma", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't",
            "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren",
            "weren't", "won", "won't", "wouldn", "wouldn't",
        ];
        
        let stopwords: HashSet<String> = default_stopwords.into_iter().map(String::from).collect();
        
        Ok(Self {
            config,
            stopwords,
            vocabulary: None,
            name: name.unwrap_or_else(|| "Tokenizer".to_string()),
        })
    }
    
    /// 加载自定义停用词
    pub fn load_stopwords(&mut self, stopwords: Vec<String>) {
        self.stopwords = stopwords.into_iter().collect();
    }
    
    /// 加载词表
    pub fn load_vocabulary(&mut self, vocabulary: HashMap<String, usize>) {
        self.vocabulary = Some(vocabulary);
    }
    
    /// 过滤token
    fn filter_token(&self, token: &str) -> Option<String> {
        let token = token.trim();
        
        // 检查长度限制
        if token.len() < self.config.min_token_length {
            return None;
        }
        
        if let Some(max_len) = self.config.max_token_length {
            if token.len() > max_len {
                return None;
            }
        }
        
        // 检查是否为停用词
        if !self.config.keep_stopwords && self.stopwords.contains(token) {
            return None;
        }
        
        Some(token.to_string())
    }
    
    /// 分词
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let text = if self.config.lowercase { text.to_lowercase() } else { text.to_string() };
        
        match self.config.mode {
            TokenizationMode::Whitespace => {
                for token in text.split_whitespace() {
                    if let Some(token) = self.filter_token(token) {
                        tokens.push(token);
                    }
                }
            },
            TokenizationMode::Split => {
                let pattern = self.config.split_pattern.as_deref().unwrap_or(" ");
                for token in text.split(pattern) {
                    if let Some(token) = self.filter_token(token) {
                        tokens.push(token);
                    }
                }
            },
            TokenizationMode::Regex => {
                // 使用正则表达式为分隔符
                if let Some(pattern) = &self.config.split_pattern {
                    if let Ok(re) = Regex::new(pattern) {
                        for token in re.split(&text) {
                            if let Some(token) = self.filter_token(token) {
                                tokens.push(token);
                            }
                        }
                    }
                } else {
                    // 默认使用非字母数字作为分隔符
                    let re = Regex::new(r"[^\w\s]+").unwrap();
                    for token in re.split(&text) {
                        if let Some(token) = self.filter_token(token) {
                            tokens.push(token);
                        }
                    }
                }
            },
            TokenizationMode::WordPiece => {
                // 简化的WordPiece实现
                if self.vocabulary.is_none() {
                    // 如果没有词表，回退到空格分割
                    for token in text.split_whitespace() {
                        if let Some(token) = self.filter_token(token) {
                            tokens.push(token);
                        }
                    }
                } else {
                    // WordPiece分词（简化版）
                    let vocabulary = self.vocabulary.as_ref().unwrap();
                    for word in text.split_whitespace() {
                        if vocabulary.contains_key(word) {
                            tokens.push(word.to_string());
                        } else {
                            // 简单的贪心分词
                            let mut start = 0;
                            let mut end = 1;
                            let chars: Vec<char> = word.chars().collect();
                            
                            while start < chars.len() {
                                while end <= chars.len() {
                                    let subword: String = chars[start..end].iter().collect();
                                    if vocabulary.contains_key(&subword) {
                                        tokens.push(subword);
                                        start = end;
                                        end = start + 1;
                                        break;
                                    }
                                    end += 1;
                                }
                                
                                if end > chars.len() {
                                    // 未知词，以单字符形式添加
                                    tokens.push(chars[start].to_string());
                                    start += 1;
                                    end = start + 1;
                                }
                            }
                        }
                    }
                }
            },
            TokenizationMode::BPE => {
                // 简化的BPE实现
                if self.vocabulary.is_none() {
                    // 如果没有词表，回退到空格分割
                    for token in text.split_whitespace() {
                        if let Some(token) = self.filter_token(token) {
                            tokens.push(token);
                        }
                    }
                } else {
                    // BPE分词（简化版）
                    let vocabulary = self.vocabulary.as_ref().unwrap();
                    for word in text.split_whitespace() {
                        if vocabulary.contains_key(word) {
                            tokens.push(word.to_string());
                        } else {
                            // 简单的字符级BPE
                            let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
                            
                            // 合并最常见的相邻对
                            while chars.len() > 1 {
                                let mut best_pair = (0, 0);
                                let mut best_score = 0;
                                
                                for i in 0..chars.len() - 1 {
                                    let pair = format!("{}{}", chars[i], chars[i + 1]);
                                    if let Some(score) = vocabulary.get(&pair) {
                                        if *score > best_score {
                                            best_score = *score;
                                            best_pair = (i, i + 1);
                                        }
                                    }
                                }
                                
                                if best_score > 0 {
                                    let (i, j) = best_pair;
                                    let merged = format!("{}{}", chars[i], chars[j]);
                                    chars[i] = merged;
                                    chars.remove(j);
                                } else {
                                    break;
                                }
                            }
                            
                            tokens.extend(chars);
                        }
                    }
                }
            },
        }
        
        tokens
    }
}

impl TextProcessor for Tokenizer {
    fn process(&self, text: &str) -> Result<String> {
        let tokens = self.tokenize(text);
        Ok(tokens.join(" "))
    }
    
    fn process_with_metadata(&self, text: &str) -> Result<(String, HashMap<String, String>)> {
        let tokens = self.tokenize(text);
        let result = tokens.join(" ");
        
        let mut metadata = HashMap::new();
        metadata.insert("token_count".to_string(), tokens.len().to_string());
        metadata.insert("original_length".to_string(), text.len().to_string());
        metadata.insert("tokenization_mode".to_string(), format!("{:?}", self.config.mode));
        
        Ok((result, metadata))
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn processor_type(&self) -> ProcessingStageType {
        ProcessingStageType::Tokenization
    }
    
    fn box_clone(&self) -> Box<dyn TextProcessor> {
        Box::new(Self {
            config: self.config.clone(),
            stopwords: self.stopwords.clone(),
            vocabulary: self.vocabulary.clone(),
            name: self.name.clone(),
        })
    }
}

// 添加必要的导入
use std::collections::HashSet;

/// 创建处理器工厂 - 用于根据配置创建文本处理器
pub struct TextProcessorFactory;

impl TextProcessorFactory {
    /// 根据配置创建处理器
    pub fn create(stage_type: ProcessingStageType, config: &HashMap<String, String>, name: Option<String>) -> Result<Box<dyn TextProcessor>> {
        match stage_type {
            ProcessingStageType::Cleaning => {
                let html_config = HtmlCleanerConfig {
                    remove_html: config.get("remove_html").map_or(true, |v| v == "true"),
                    remove_scripts: config.get("remove_scripts").map_or(true, |v| v == "true"),
                    remove_styles: config.get("remove_styles").map_or(true, |v| v == "true"),
                    preserve_tags: config.get("preserve_tags")
                        .map_or_else(Vec::new, |v| v.split(',').map(String::from).collect()),
                    decode_entities: config.get("decode_entities").map_or(true, |v| v == "true"),
                };
                
                Ok(Box::new(HtmlCleaner::new(html_config, name)?))
            },
            ProcessingStageType::Normalization => {
                let normalizer_config = TextNormalizerConfig {
                    lowercase: config.get("lowercase").map_or(true, |v| v == "true"),
                    remove_accents: config.get("remove_accents").map_or(false, |v| v == "true"),
                    normalize_whitespace: config.get("normalize_whitespace").map_or(true, |v| v == "true"),
                    remove_punctuation: config.get("remove_punctuation").map_or(false, |v| v == "true"),
                    remove_numbers: config.get("remove_numbers").map_or(false, |v| v == "true"),
                    normalize_newlines: config.get("normalize_newlines").map_or(true, |v| v == "true"),
                };
                
                Ok(Box::new(TextNormalizer::new(normalizer_config, name)?))
            },
            ProcessingStageType::Tokenization => {
                let mode = match config.get("mode").map(|s| s.as_str()) {
                    Some("whitespace") => TokenizationMode::Whitespace,
                    Some("split") => TokenizationMode::Split,
                    Some("regex") => TokenizationMode::Regex,
                    Some("wordpiece") => TokenizationMode::WordPiece,
                    Some("bpe") => TokenizationMode::BPE,
                    _ => TokenizationMode::Whitespace,
                };
                
                let tokenizer_config = TokenizerConfig {
                    mode,
                    lowercase: config.get("lowercase").map_or(true, |v| v == "true"),
                    keep_stopwords: config.get("keep_stopwords").map_or(false, |v| v == "true"),
                    min_token_length: config.get("min_token_length").and_then(|v| v.parse().ok()).unwrap_or(1),
                    max_token_length: config.get("max_token_length").and_then(|v| v.parse().ok()),
                    split_pattern: config.get("split_pattern").cloned(),
                };
                
                Ok(Box::new(Tokenizer::new(tokenizer_config, name)?))
            },
            _ => Err(Error::InvalidOperation(format!("不支持的处理器类型: {:?}", stage_type))),
        }
    }
} 