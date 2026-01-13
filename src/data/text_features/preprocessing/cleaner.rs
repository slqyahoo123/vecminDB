// Text Cleaner Implementation
// 文本清洁器实现

use crate::Error;
use crate::data::text_features::preprocessing::TextPreprocessor;
use regex::Regex;
use std::collections::HashSet;
use lazy_static::lazy_static;
use crate::Result;
use crate::data::text_features::pipeline::ProcessingStageType;
use crate::data::text_features::preprocessing::TextProcessor;

lazy_static! {
    static ref HTML_TAG_REGEX: Regex = Regex::new(r"<[^>]*>").unwrap();
    static ref SPECIAL_CHARS_REGEX: Regex = Regex::new(r"[^\w\s]").unwrap();
    static ref REPEATED_CHARS_REGEX: Regex = Regex::new(r"(.)\1{2,}").unwrap();
    static ref EMOJI_REGEX: Regex = Regex::new(
        r"[\x{1F600}-\x{1F64F}|\x{1F300}-\x{1F5FF}|\x{1F680}-\x{1F6FF}|\x{1F700}-\x{1F77F}|\x{1F780}-\x{1F7FF}|\x{1F800}-\x{1F8FF}|\x{1F900}-\x{1F9FF}|\x{1FA00}-\x{1FA6F}|\x{1FA70}-\x{1FAFF}|\x{2600}-\x{26FF}|\x{2700}-\x{27BF}]"
    ).unwrap();
}

/// 文本清洗器特征
pub trait TextCleaner: Send + Sync {
    /// 清洗文本
    fn clean(&self, text: &str) -> Result<String>;
    
    /// 获取清洗器名称
    fn name(&self) -> &str;
}

/// 文本清理器
#[derive(Clone)]
pub struct StandardTextCleaner {
    /// 清理配置
    config: CleanerConfig,
    /// 自定义替换规则
    custom_patterns: Vec<(Regex, String)>,
}

/// 清理配置
#[derive(Clone, Debug)]
pub struct CleanerConfig {
    /// 是否移除HTML标记
    pub remove_html_tags: bool,
    /// 是否移除Emoji
    pub remove_emojis: bool,
    /// 是否移除特殊字符
    pub remove_special_chars: bool,
    /// 是否移除重复字符
    pub collapse_repeated_chars: bool,
    /// 是否移除非ASCII字符
    pub remove_non_ascii: bool,
    /// 是否替换常见拼写错误
    pub fix_common_misspellings: bool,
    /// 保留的特殊字符
    pub allowed_special_chars: HashSet<char>,
    /// 最小文本长度
    pub min_text_length: usize,
}

impl Default for CleanerConfig {
    fn default() -> Self {
        Self {
            remove_html_tags: true,
            remove_emojis: true,
            remove_special_chars: false,
            collapse_repeated_chars: true,
            remove_non_ascii: false,
            fix_common_misspellings: true,
            allowed_special_chars: ['.', ',', '!', '?', '-', ':', ';', '\'', '"'].iter().cloned().collect(),
            min_text_length: 1,
        }
    }
}

impl StandardTextCleaner {
    /// 创建新的文本清理器
    pub fn new() -> Self {
        Self {
            config: CleanerConfig::default(),
            custom_patterns: Vec::new(),
        }
    }
    
    /// 使用自定义配置创建
    pub fn with_config(config: CleanerConfig) -> Self {
        Self {
            config,
            custom_patterns: Vec::new(),
        }
    }
    
    /// 添加自定义清理模式
    pub fn add_pattern(&mut self, pattern: &str, replacement: &str) -> Result<(), Error> {
        let regex = Regex::new(pattern).map_err(|e| Error::InvalidParameter(format!("无效的正则表达式模式: {}", e)))?;
        self.custom_patterns.push((regex, replacement.to_string()));
        Ok(())
    }
    
    /// 使用自定义模式创建
    pub fn with_patterns(mut self, patterns: Vec<(String, String)>) -> Result<Self, Error> {
        for (pattern, replacement) in patterns {
            self.add_pattern(&pattern, &replacement)?;
        }
        Ok(self)
    }
    
    /// 移除HTML标签
    fn remove_html_tags(&self, text: &str) -> String {
        HTML_TAG_REGEX.replace_all(text, "").to_string()
    }
    
    /// 移除表情
    fn remove_emojis(&self, text: &str) -> String {
        EMOJI_REGEX.replace_all(text, "").to_string()
    }
    
    /// 移除特殊字符
    fn remove_special_chars(&self, text: &str) -> String {
        if self.config.allowed_special_chars.is_empty() {
            SPECIAL_CHARS_REGEX.replace_all(text, "").to_string()
        } else {
            text.chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace() || self.config.allowed_special_chars.contains(c))
                .collect()
        }
    }
    
    /// 折叠重复字符
    fn collapse_repeated_chars(&self, text: &str) -> String {
        REPEATED_CHARS_REGEX.replace_all(text, "$1$1").to_string()
    }
    
    /// 移除非ASCII字符
    fn remove_non_ascii(&self, text: &str) -> String {
        text.chars().filter(|c| c.is_ascii()).collect()
    }
    
    /// 修复常见拼写错误
    fn fix_common_misspellings(&self, text: &str) -> String {
        // 简单的常见拼写错误修复
        let mut corrected = text.to_string();
        
        // 实现简单的拼写修正，更复杂的情况可以使用专门的库
        let common_errors = [
            ("teh", "the"),
            ("recieve", "receive"),
            ("alot", "a lot"),
            ("dont", "don't"),
            ("didnt", "didn't"),
            ("wasnt", "wasn't"),
            ("isnt", "isn't"),
            ("wont", "won't"),
        ];
        
        for (error, correction) in common_errors.iter() {
            corrected = corrected.replace(error, correction);
        }
        
        corrected
    }
    
    /// 应用自定义替换模式
    fn apply_custom_patterns(&self, text: &str) -> String {
        let mut result = text.to_string();
        
        for (pattern, replacement) in &self.custom_patterns {
            result = pattern.replace_all(&result, replacement).to_string();
        }
        
        result
    }
}

impl TextProcessor for StandardTextCleaner {
    /// 处理文本
    fn process(&self, text: &str) -> Result<String, Error> {
        // 如果输入为空，返回空
        if text.is_empty() {
            return Ok(String::new());
        }
        
        let mut result = text.to_string();
        
        // 应用配置的清理操作
        if self.config.remove_html_tags {
            result = self.remove_html_tags(&result);
        }
        
        if self.config.remove_emojis {
            result = self.remove_emojis(&result);
        }
        
        if self.config.remove_special_chars {
            result = self.remove_special_chars(&result);
        }
        
        if self.config.collapse_repeated_chars {
            result = self.collapse_repeated_chars(&result);
        }
        
        if self.config.remove_non_ascii {
            result = self.remove_non_ascii(&result);
        }
        
        if self.config.fix_common_misspellings {
            result = self.fix_common_misspellings(&result);
        }
        
        // 应用自定义替换规则
        result = self.apply_custom_patterns(&result);
        
        // 检查是否达到最小长度要求
        if result.trim().len() < self.config.min_text_length {
            return Err(Error::InvalidInput(format!(
                "Text length after cleaning is below minimum requirement of {} characters", 
                self.config.min_text_length
            )));
        }
        
        Ok(result)
    }
    
    /// 获取预处理器名称
    fn name(&self) -> &str {
        "text_cleaner"
    }
    
    /// 获取处理器类型
    fn processor_type(&self) -> ProcessingStageType {
        ProcessingStageType::Cleaning
    }
    
    /// 克隆处理器
    fn box_clone(&self) -> Box<dyn TextProcessor> {
        Box::new(self.clone())
    }
}

impl TextCleaner for StandardTextCleaner {
    fn clean(&self, text: &str) -> Result<String> {
        self.process(text)
    }
    
    fn name(&self) -> &str {
        "standard_text_cleaner"
    }
}

/// 标点符号清洗器
pub struct PunctuationCleaner {
    name: String,
}

impl PunctuationCleaner {
    /// 创建新的标点符号清洗器
    pub fn new() -> Self {
        Self {
            name: "PunctuationCleaner".to_string(),
        }
    }
}

impl TextCleaner for PunctuationCleaner {
    fn clean(&self, text: &str) -> Result<String> {
        lazy_static! {
            static ref RE: Regex = Regex::new(r"[^\w\s]").unwrap();
        }
        
        Ok(RE.replace_all(text, "").to_string())
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// 数字清洗器
pub struct DigitCleaner {
    name: String,
}

impl DigitCleaner {
    /// 创建新的数字清洗器
    pub fn new() -> Self {
        Self {
            name: "DigitCleaner".to_string(),
        }
    }
}

impl TextCleaner for DigitCleaner {
    fn clean(&self, text: &str) -> Result<String> {
        lazy_static! {
            static ref RE: Regex = Regex::new(r"\d").unwrap();
        }
        
        Ok(RE.replace_all(text, "").to_string())
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// HTML标签清洗器
pub struct HtmlCleaner {
    name: String,
}

impl HtmlCleaner {
    /// 创建新的HTML标签清洗器
    pub fn new() -> Self {
        Self {
            name: "HtmlCleaner".to_string(),
        }
    }
}

impl TextCleaner for HtmlCleaner {
    fn clean(&self, text: &str) -> Result<String> {
        lazy_static! {
            static ref RE: Regex = Regex::new(r"<[^>]*>").unwrap();
        }
        
        Ok(RE.replace_all(text, "").to_string())
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// URL清洗器
pub struct UrlCleaner {
    name: String,
}

impl UrlCleaner {
    /// 创建新的URL清洗器
    pub fn new() -> Self {
        Self {
            name: "UrlCleaner".to_string(),
        }
    }
}

impl TextCleaner for UrlCleaner {
    fn clean(&self, text: &str) -> Result<String> {
        lazy_static! {
            static ref RE: Regex = Regex::new(
                r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
            ).unwrap();
        }
        
        Ok(RE.replace_all(text, "").to_string())
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// 邮箱清洗器
pub struct EmailCleaner {
    name: String,
}

impl EmailCleaner {
    /// 创建新的邮箱清洗器
    pub fn new() -> Self {
        Self {
            name: "EmailCleaner".to_string(),
        }
    }
}

impl TextCleaner for EmailCleaner {
    fn clean(&self, text: &str) -> Result<String> {
        lazy_static! {
            static ref RE: Regex = Regex::new(
                r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
            ).unwrap();
        }
        
        Ok(RE.replace_all(text, "").to_string())
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// 表情符号清洗器
pub struct EmojiCleaner {
    name: String,
}

impl EmojiCleaner {
    /// 创建新的表情符号清洗器
    pub fn new() -> Self {
        Self {
            name: "EmojiCleaner".to_string(),
        }
    }
}

impl TextCleaner for EmojiCleaner {
    fn clean(&self, text: &str) -> Result<String> {
        // 简单实现，完整实现需要更复杂的表情符号检测
        lazy_static! {
            static ref RE: Regex = Regex::new(
                r"[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{1F700}-\u{1F77F}]|[\u{1F780}-\u{1F7FF}]|[\u{1F800}-\u{1F8FF}]|[\u{1F900}-\u{1F9FF}]|[\u{1FA00}-\u{1FA6F}]|[\u{1FA70}-\u{1FAFF}]"
            ).unwrap();
        }
        
        Ok(RE.replace_all(text, "").to_string())
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// 额外空白字符清洗器
pub struct WhitespaceCleaner {
    name: String,
}

impl WhitespaceCleaner {
    /// 创建新的额外空白字符清洗器
    pub fn new() -> Self {
        Self {
            name: "WhitespaceCleaner".to_string(),
        }
    }
}

impl TextCleaner for WhitespaceCleaner {
    fn clean(&self, text: &str) -> Result<String> {
        lazy_static! {
            static ref RE: Regex = Regex::new(r"\s+").unwrap();
        }
        
        Ok(RE.replace_all(text, " ").trim().to_string())
    }
    
    fn name(&self) -> &str {
        &self.name
    }
} 