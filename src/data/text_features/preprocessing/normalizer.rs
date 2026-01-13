// Text Normalizer Implementation
// 文本规范化器实现

// use crate::Error; // errors are propagated via crate::Result in this module
use crate::data::text_features::preprocessing::TextPreprocessor;
use unicode_segmentation::UnicodeSegmentation;
use unicode_normalization::UnicodeNormalization;
use regex::Regex;
use std::collections::HashMap;
use lazy_static::lazy_static;
use crate::Result;
use std::collections::HashSet;
use crate::data::text_features::pipeline::ProcessingStageType;
use crate::data::text_features::preprocessing::TextProcessor;

lazy_static! {
    static ref PUNCTUATION_REGEX: Regex = Regex::new(r"[^\w\s]").unwrap();
    static ref NUMBER_REGEX: Regex = Regex::new(r"\d+").unwrap();
    static ref WHITESPACE_REGEX: Regex = Regex::new(r"\s+").unwrap();
    static ref URL_REGEX: Regex = Regex::new(r"https?://\S+|www\.\S+").unwrap();
    static ref EMAIL_REGEX: Regex = Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b").unwrap();
}

/// 文本规范化器特征
pub trait TextNormalizer: Send + Sync {
    /// 规范化文本
    fn normalize(&self, text: &str) -> Result<String>;
    
    /// 获取规范化器名称
    fn name(&self) -> &str;
}

/// 文本规范化器
#[derive(Clone)]
pub struct StandardTextNormalizer {
    /// 规范化配置
    config: NormalizerConfig,
    /// 自定义替换映射
    replacements: HashMap<String, String>,
}

/// 规范化配置
#[derive(Clone, Debug)]
pub struct NormalizerConfig {
    /// 是否转为小写
    pub lowercase: bool,
    /// 是否移除标点符号
    pub remove_punctuation: bool,
    /// 是否移除数字
    pub remove_numbers: bool,
    /// 是否移除URLs
    pub remove_urls: bool,
    /// 是否移除邮件地址
    pub remove_emails: bool,
    /// 是否进行Unicode规范化
    pub unicode_normalize: bool,
    /// Unicode规范化方式
    pub unicode_normalization_form: UnicodeNormalizationForm,
    /// 是否移除额外空白
    pub remove_extra_whitespace: bool,
    /// 是否取代全角字符
    pub replace_fullwidth_chars: bool,
}

/// Unicode规范化方式
#[derive(Clone, Debug, PartialEq)]
pub enum UnicodeNormalizationForm {
    /// NFD - Canonical Decomposition
    NFD,
    /// NFC - Canonical Decomposition followed by Canonical Composition
    NFC,
    /// NFKD - Compatibility Decomposition
    NFKD,
    /// NFKC - Compatibility Decomposition followed by Canonical Composition
    NFKC,
}

impl Default for NormalizerConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_punctuation: false,
            remove_numbers: false,
            remove_urls: true,
            remove_emails: false,
            unicode_normalize: true,
            unicode_normalization_form: UnicodeNormalizationForm::NFC,
            remove_extra_whitespace: true,
            replace_fullwidth_chars: true,
        }
    }
}

impl StandardTextNormalizer {
    /// 创建新的文本规范化器
    pub fn new() -> Self {
        Self {
            config: NormalizerConfig::default(),
            replacements: HashMap::new(),
        }
    }
    
    /// 使用自定义配置创建
    pub fn with_config(config: NormalizerConfig) -> Self {
        Self {
            config,
            replacements: HashMap::new(),
        }
    }
    
    /// 添加自定义替换规则
    pub fn add_replacement(&mut self, pattern: &str, replacement: &str) {
        self.replacements.insert(pattern.to_string(), replacement.to_string());
    }
    
    /// 使用自定义替换规则创建
    pub fn with_replacements(mut self, replacements: HashMap<String, String>) -> Self {
        self.replacements = replacements;
        self
    }
    
    /// 去除标点符号
    fn remove_punctuation(&self, text: &str) -> String {
        PUNCTUATION_REGEX.replace_all(text, "").to_string()
    }
    
    /// 去除数字
    fn remove_numbers(&self, text: &str) -> String {
        NUMBER_REGEX.replace_all(text, "").to_string()
    }
    
    /// 去除多余空白
    fn remove_extra_whitespace(&self, text: &str) -> String {
        WHITESPACE_REGEX.replace_all(text, " ").trim().to_string()
    }
    
    /// 去除URL
    fn remove_urls(&self, text: &str) -> String {
        URL_REGEX.replace_all(text, "").to_string()
    }
    
    /// 去除邮件地址
    fn remove_emails(&self, text: &str) -> String {
        EMAIL_REGEX.replace_all(text, "").to_string()
    }
    
    /// Unicode规范化
    fn normalize_unicode(&self, text: &str) -> String {
        match self.config.unicode_normalization_form {
            UnicodeNormalizationForm::NFD => text.nfd().collect(),
            UnicodeNormalizationForm::NFC => text.nfc().collect(),
            UnicodeNormalizationForm::NFKD => text.nfkd().collect(),
            UnicodeNormalizationForm::NFKC => text.nfkc().collect(),
        }
    }
    
    /// 替换全角字符
    fn replace_fullwidth_chars(&self, text: &str) -> String {
        let mut result = String::with_capacity(text.len());
        for c in text.chars() {
            if c >= '\u{FF01}' && c <= '\u{FF5E}' {
                // 全角ASCII字符区域
                let halfwidth = char::from_u32(c as u32 - 0xFEE0).unwrap_or(c);
                result.push(halfwidth);
            } else if c == '\u{3000}' {
                // 全角空格
                result.push(' ');
            } else {
                result.push(c);
            }
        }
        result
    }
    
    /// 应用自定义替换规则
    fn apply_custom_replacements(&self, text: &str) -> String {
        let mut result = text.to_string();
        for (pattern, replacement) in &self.replacements {
            result = result.replace(pattern, replacement);
        }
        result
    }
}

impl TextProcessor for StandardTextNormalizer {
    fn process(&self, text: &str) -> Result<String> {
        // 如果输入为空，返回空
        if text.is_empty() {
            return Ok(String::new());
        }
        
        let mut result = text.to_string();
        
        // 应用配置的规范化操作
        if self.config.lowercase {
            result = result.to_lowercase();
        }
        
        if self.config.remove_punctuation {
            result = self.remove_punctuation(&result);
        }
        
        if self.config.remove_numbers {
            result = self.remove_numbers(&result);
        }
        
        if self.config.remove_urls {
            result = self.remove_urls(&result);
        }
        
        if self.config.remove_emails {
            result = self.remove_emails(&result);
        }
        
        if self.config.unicode_normalize {
            result = self.normalize_unicode(&result);
        }
        
        if self.config.replace_fullwidth_chars {
            result = self.replace_fullwidth_chars(&result);
        }

        // 字素级清理：移除孤立的控制字符和零宽字符
        let cleaned: String = result
            .graphemes(true)
            .filter(|g| {
                // 过滤常见不可见字符
                !matches!(*g, "\u{200B}" | "\u{200C}" | "\u{200D}" | "\u{2060}")
            })
            .collect();
        let mut result = cleaned;
        
        // 应用自定义规范化规则
        result = self.apply_custom_replacements(&result);
        
        if self.config.remove_extra_whitespace {
            result = self.remove_extra_whitespace(&result);
        }
        
        Ok(result)
    }
    
    fn name(&self) -> &str {
        "text_normalizer"
    }
    
    fn processor_type(&self) -> ProcessingStageType {
        ProcessingStageType::Normalization
    }
    
    fn box_clone(&self) -> Box<dyn TextProcessor> {
        Box::new(self.clone())
    }
}

impl TextNormalizer for StandardTextNormalizer {
    fn normalize(&self, text: &str) -> Result<String> {
        self.process(text)
    }
    
    fn name(&self) -> &str {
        "text_normalizer"
    }
}

/// 小写转换规范化器
pub struct LowercaseNormalizer {
    name: String,
}

impl LowercaseNormalizer {
    /// 创建新的小写转换规范化器
    pub fn new() -> Self {
        Self {
            name: "LowercaseNormalizer".to_string(),
        }
    }
}

impl TextNormalizer for LowercaseNormalizer {
    fn normalize(&self, text: &str) -> Result<String> {
        Ok(text.to_lowercase())
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// 词干提取规范化器
pub struct StemmingNormalizer {
    name: String,
}

impl StemmingNormalizer {
    /// 创建新的词干提取规范化器
    pub fn new() -> Self {
        Self {
            name: "StemmingNormalizer".to_string(),
        }
    }
    
    /// 对单词进行词干提取
    fn stem_word(&self, word: &str) -> String {
        // 简单实现，实际应使用专业词干提取库
        let word = word.to_lowercase();
        
        // 一些简单的词尾处理规则
        let mut result = word.clone();
        if word.ends_with("ing") && word.len() > 4 {
            result = word[0..word.len() - 3].to_string();
        } else if word.ends_with("ly") && word.len() > 3 {
            result = word[0..word.len() - 2].to_string();
        } else if word.ends_with("s") && word.len() > 2 {
            result = word[0..word.len() - 1].to_string();
        } else if word.ends_with("ed") && word.len() > 3 {
            result = word[0..word.len() - 2].to_string();
        }
        
        result
    }
}

impl TextNormalizer for StemmingNormalizer {
    fn normalize(&self, text: &str) -> Result<String> {
        let words: Vec<String> = text.split_whitespace()
            .map(|word| self.stem_word(word))
            .collect();
        
        Ok(words.join(" "))
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// 词形还原规范化器
pub struct LemmatizationNormalizer {
    name: String,
}

impl LemmatizationNormalizer {
    /// 创建新的词形还原规范化器
    pub fn new() -> Self {
        Self {
            name: "LemmatizationNormalizer".to_string(),
        }
    }
    
    /// 对单词进行词形还原
    fn lemmatize_word(&self, word: &str) -> String {
        // 简单实现，实际应使用专业词形还原库
        let word = word.to_lowercase();
        
        // 一些简单的词形还原规则
        let mut result = word.clone();
        if word == "running" {
            result = "run".to_string();
        } else if word == "went" || word == "going" || word == "goes" {
            result = "go".to_string();
        } else if word == "better" || word == "best" {
            result = "good".to_string();
        } else if word == "worse" || word == "worst" {
            result = "bad".to_string();
        } else if word == "am" || word == "are" || word == "is" || word == "was" || word == "were" {
            result = "be".to_string();
        }
        
        result
    }
}

impl TextNormalizer for LemmatizationNormalizer {
    fn normalize(&self, text: &str) -> Result<String> {
        let words: Vec<String> = text.split_whitespace()
            .map(|word| self.lemmatize_word(word))
            .collect();
        
        Ok(words.join(" "))
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// 停用词移除规范化器
pub struct StopwordNormalizer {
    name: String,
    stop_words: HashSet<String>,
}

impl StopwordNormalizer {
    /// 创建新的停用词移除规范化器
    pub fn new(stop_words: HashSet<String>) -> Self {
        Self {
            name: "StopwordNormalizer".to_string(),
            stop_words,
        }
    }
}

impl TextNormalizer for StopwordNormalizer {
    fn normalize(&self, text: &str) -> Result<String> {
        let words: Vec<String> = text.split_whitespace()
            .filter(|word| !self.stop_words.contains(&word.to_lowercase()))
            .map(|word| word.to_string())
            .collect();
        
        Ok(words.join(" "))
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// 最大长度限制规范化器
pub struct LengthLimitNormalizer {
    name: String,
    max_length: usize,
}

impl LengthLimitNormalizer {
    /// 创建新的最大长度限制规范化器
    pub fn new(max_length: usize) -> Self {
        Self {
            name: format!("LengthLimitNormalizer({})", max_length),
            max_length,
        }
    }
}

impl TextNormalizer for LengthLimitNormalizer {
    fn normalize(&self, text: &str) -> Result<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        
        if words.len() <= self.max_length {
            return Ok(text.to_string());
        }
        
        Ok(words[0..self.max_length].join(" "))
    }
    
    fn name(&self) -> &str {
        &self.name
    }
} 