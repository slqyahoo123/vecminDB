// src/data/text_features/language.rs
//
// Transformer 语言处理模块

use std::collections::HashMap;
use super::error::TransformerError;

/// 语言类型
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Language {
    /// 英语
    English,
    /// 中文
    Chinese,
    /// 日语
    Japanese,
    /// 韩语
    Korean,
    /// 阿拉伯语
    Arabic,
    /// 俄语
    Russian,
    /// 法语
    French,
    /// 德语
    German,
    /// 西班牙语
    Spanish,
    /// 意大利语
    Italian,
    /// 葡萄牙语
    Portuguese,
    /// 未知语言
    Unknown,
}

impl Language {
    /// 获取语言代码
    pub fn code(&self) -> &'static str {
        match self {
            Language::English => "en",
            Language::Chinese => "zh",
            Language::Japanese => "ja",
            Language::Korean => "ko",
            Language::Arabic => "ar",
            Language::Russian => "ru",
            Language::French => "fr",
            Language::German => "de",
            Language::Spanish => "es",
            Language::Italian => "it",
            Language::Portuguese => "pt",
            Language::Unknown => "unknown",
        }
    }
    
    /// 获取语言名称
    pub fn name(&self) -> &'static str {
        match self {
            Language::English => "English",
            Language::Chinese => "Chinese",
            Language::Japanese => "Japanese",
            Language::Korean => "Korean",
            Language::Arabic => "Arabic",
            Language::Russian => "Russian",
            Language::French => "French",
            Language::German => "German",
            Language::Spanish => "Spanish",
            Language::Italian => "Italian",
            Language::Portuguese => "Portuguese",
            Language::Unknown => "Unknown",
        }
    }
    
    /// 是否为RTL语言
    pub fn is_rtl(&self) -> bool {
        matches!(self, Language::Arabic)
    }
    
    /// 是否为CJK语言
    pub fn is_cjk(&self) -> bool {
        matches!(self, Language::Chinese | Language::Japanese | Language::Korean)
    }
}

/// 语言处理配置
#[derive(Debug, Clone)]
pub struct LanguageProcessingConfig {
    /// 是否启用语言检测
    pub enable_language_detection: bool,
    /// 是否启用多语言处理
    pub enable_multilingual_processing: bool,
    /// 是否启用语言特定预处理
    pub enable_language_specific_preprocessing: bool,
    /// 是否启用混合语言处理
    pub enable_mixed_language_processing: bool,
    /// 主要语言
    pub primary_language: Option<Language>,
    /// 次要语言
    pub secondary_language: Option<Language>,
    /// 语言权重
    pub language_weights: HashMap<Language, f32>,
}

impl Default for LanguageProcessingConfig {
    fn default() -> Self {
        let mut language_weights = HashMap::new();
        language_weights.insert(Language::English, 1.0);
        language_weights.insert(Language::Chinese, 0.8);
        language_weights.insert(Language::Japanese, 0.7);
        language_weights.insert(Language::Korean, 0.7);
        language_weights.insert(Language::Arabic, 0.6);
        language_weights.insert(Language::Russian, 0.6);
        
        Self {
            enable_language_detection: true,
            enable_multilingual_processing: true,
            enable_language_specific_preprocessing: true,
            enable_mixed_language_processing: true,
            primary_language: None,
            secondary_language: None,
            language_weights,
        }
    }
}

/// 语言处理器
pub struct LanguageProcessor {
    /// 配置
    config: LanguageProcessingConfig,
    /// 语言检测器
    language_detector: LanguageDetector,
    /// 语言特定预处理器
    preprocessors: HashMap<Language, Box<dyn LanguagePreprocessor>>,
}

impl std::fmt::Debug for LanguageProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanguageProcessor")
            .field("config", &self.config)
            .field("language_detector", &self.language_detector)
            .field("preprocessors_count", &self.preprocessors.len())
            .finish()
    }
}

impl LanguageProcessor {
    /// 创建新的语言处理器
    pub fn new(config: LanguageProcessingConfig) -> Self {
        let mut processor = Self {
            config,
            language_detector: LanguageDetector::new(),
            preprocessors: HashMap::new(),
        };
        
        // 初始化语言特定预处理器
        processor.initialize_preprocessors();
        
        processor
    }
    
    /// 初始化预处理器
    fn initialize_preprocessors(&mut self) {
        self.preprocessors.insert(Language::English, Box::new(EnglishPreprocessor));
        self.preprocessors.insert(Language::Chinese, Box::new(ChinesePreprocessor));
        self.preprocessors.insert(Language::Japanese, Box::new(JapanesePreprocessor));
        self.preprocessors.insert(Language::Korean, Box::new(KoreanPreprocessor));
        self.preprocessors.insert(Language::Arabic, Box::new(ArabicPreprocessor));
        self.preprocessors.insert(Language::Russian, Box::new(RussianPreprocessor));
    }
    
    /// 处理文本
    pub fn process_text(&self, text: &str) -> Result<ProcessedText, TransformerError> {
        let mut processed = ProcessedText::new(text.to_string());
        
        // 语言检测
        if self.config.enable_language_detection {
            let detected_language = self.language_detector.detect_language(text)?;
            processed.detected_language = Some(detected_language);
        }
        
        // 语言特定预处理
        if self.config.enable_language_specific_preprocessing {
            if let Some(language) = processed.detected_language {
                if let Some(preprocessor) = self.preprocessors.get(&language) {
                    processed.processed_text = preprocessor.preprocess(&processed.original_text)?;
                }
            }
        }
        
        // 多语言处理
        if self.config.enable_multilingual_processing {
            self.process_multilingual_text(&mut processed)?;
        }
        
        Ok(processed)
    }
    
    /// 处理多语言文本
    fn process_multilingual_text(&self, processed: &mut ProcessedText) -> Result<(), TransformerError> {
        if let Some(language) = processed.detected_language {
            if language == Language::Unknown {
                // 尝试检测混合语言
                let mixed_languages = self.detect_mixed_languages(&processed.original_text)?;
                if mixed_languages.len() > 1 {
                    processed.mixed_languages = mixed_languages;
                    self.process_mixed_language_text(processed)?;
                }
            }
        }
        
        Ok(())
    }
    
    /// 检测混合语言
    fn detect_mixed_languages(&self, text: &str) -> Result<Vec<Language>, TransformerError> {
        let mut languages = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        
        for word in words {
            if let Ok(language) = self.language_detector.detect_word_language(word) {
                if !languages.contains(&language) {
                    languages.push(language);
                }
            }
        }
        
        Ok(languages)
    }
    
    /// 处理混合语言文本
    fn process_mixed_language_text(&self, processed: &mut ProcessedText) -> Result<(), TransformerError> {
        if processed.mixed_languages.is_empty() {
            return Ok(());
        }
        
        // 按语言分割文本
        let mut language_segments = HashMap::new();
        let words: Vec<&str> = processed.original_text.split_whitespace().collect();
        
        for word in words {
            if let Ok(language) = self.language_detector.detect_word_language(word) {
                language_segments.entry(language).or_insert_with(Vec::new).push(word);
            }
        }
        
        // 处理每个语言段
        for (language, words) in language_segments {
            if let Some(preprocessor) = self.preprocessors.get(&language) {
                let segment_text = words.join(" ");
                let processed_segment = preprocessor.preprocess(&segment_text)?;
                
                // 应用语言权重
                let weight = self.config.language_weights.get(&language).unwrap_or(&1.0);
                processed.language_weights.insert(language, *weight);
                
                // 合并处理后的文本
                if !processed.processed_text.is_empty() {
                    processed.processed_text.push(' ');
                }
                processed.processed_text.push_str(&processed_segment);
            }
        }
        
        Ok(())
    }
    
    /// 获取配置
    pub fn config(&self) -> &LanguageProcessingConfig {
        &self.config
    }
}

/// 处理后的文本
#[derive(Debug, Clone)]
pub struct ProcessedText {
    /// 原始文本
    pub original_text: String,
    /// 处理后的文本
    pub processed_text: String,
    /// 检测到的语言
    pub detected_language: Option<Language>,
    /// 混合语言列表
    pub mixed_languages: Vec<Language>,
    /// 语言权重
    pub language_weights: HashMap<Language, f32>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

impl ProcessedText {
    /// 创建新的处理后文本
    pub fn new(original_text: String) -> Self {
        Self {
            original_text,
            processed_text: String::new(),
            detected_language: None,
            mixed_languages: Vec::new(),
            language_weights: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
    
    /// 获取主要语言
    pub fn primary_language(&self) -> Option<Language> {
        self.detected_language
    }
    
    /// 是否为混合语言
    pub fn is_mixed_language(&self) -> bool {
        self.mixed_languages.len() > 1
    }
    
    /// 获取语言权重
    pub fn get_language_weight(&self, language: &Language) -> f32 {
        self.language_weights.get(language).copied().unwrap_or(1.0)
    }
}

/// 语言检测器
#[derive(Debug)]
pub struct LanguageDetector {
    /// 语言特征
    language_features: HashMap<Language, Vec<String>>,
}

impl LanguageDetector {
    /// 创建新的语言检测器
    pub fn new() -> Self {
        let mut detector = Self {
            language_features: HashMap::new(),
        };
        
        detector.initialize_language_features();
        detector
    }
    
    /// 初始化语言特征
    fn initialize_language_features(&mut self) {
        // 英语特征
        self.language_features.insert(Language::English, vec![
            "the".to_string(), "a".to_string(), "an".to_string(), "and".to_string(), "or".to_string(), "but".to_string(), "in".to_string(), "on".to_string(), "at".to_string(), "to".to_string(), "for".to_string(),
            "of".to_string(), "with".to_string(), "by".to_string(), "is".to_string(), "are".to_string(), "was".to_string(), "were".to_string(), "be".to_string(), "been".to_string(), "being".to_string()
        ]);
        
        // 中文特征
        self.language_features.insert(Language::Chinese, vec![
            "的".to_string(), "了".to_string(), "在".to_string(), "是".to_string(), "我".to_string(), "有".to_string(), "和".to_string(), "他".to_string(), "她".to_string(), "你".to_string(), "们".to_string(),
            "这".to_string(), "那".to_string(), "个".to_string(), "不".to_string(), "也".to_string(), "就".to_string(), "说".to_string(), "到".to_string(), "去".to_string(), "来".to_string()
        ]);
        
        // 日语特征
        self.language_features.insert(Language::Japanese, vec![
            "の".to_string(), "に".to_string(), "は".to_string(), "を".to_string(), "が".to_string(), "で".to_string(), "と".to_string(), "から".to_string(), "まで".to_string(), "より".to_string(),
            "です".to_string(), "ます".to_string(), "た".to_string(), "て".to_string(), "いる".to_string(), "ある".to_string(), "する".to_string(), "なる".to_string()
        ]);
        
        // 韩语特征
        self.language_features.insert(Language::Korean, vec![
            "의".to_string(), "에".to_string(), "가".to_string(), "을".to_string(), "를".to_string(), "은".to_string(), "는".to_string(), "이".to_string(), "그".to_string(), "저".to_string(),
            "하다".to_string(), "있다".to_string(), "없다".to_string(), "되다".to_string(), "하다".to_string(), "보다".to_string(), "보다".to_string()
        ]);
        
        // 阿拉伯语特征
        self.language_features.insert(Language::Arabic, vec![
            "في".to_string(), "من".to_string(), "إلى".to_string(), "على".to_string(), "أن".to_string(), "هذا".to_string(), "التي".to_string(), "كان".to_string(), "ليس".to_string(), "عند".to_string(),
            "مع".to_string(), "أو".to_string(), "و".to_string(), "ف".to_string(), "ثم".to_string(), "إذا".to_string(), "حيث".to_string(), "كيف".to_string(), "متى".to_string(), "لماذا".to_string()
        ]);
        
        // 俄语特征
        self.language_features.insert(Language::Russian, vec![
            "в".to_string(), "на".to_string(), "с".to_string(), "по".to_string(), "для".to_string(), "от".to_string(), "до".to_string(), "из".to_string(), "к".to_string(), "у".to_string(),
            "быть".to_string(), "быть".to_string(), "быть".to_string(), "иметь".to_string(), "делать".to_string(), "говорить".to_string(), "знать".to_string()
        ]);
    }
    
    /// 检测语言
    pub fn detect_language(&self, text: &str) -> Result<Language, TransformerError> {
        if text.is_empty() {
            return Ok(Language::Unknown);
        }
        
        let mut language_scores = HashMap::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        
        for (language, features) in &self.language_features {
            let mut score = 0.0;
            for word in &words {
                let word_lower = word.to_lowercase();
                if features.iter().any(|feature| word_lower.contains(feature)) {
                    score += 1.0;
                }
            }
            if score > 0.0 {
                language_scores.insert(language.clone(), score / words.len() as f32);
            }
        }
        
        // 如果没有找到匹配的语言，使用字符特征检测
        if language_scores.is_empty() {
            return self.detect_language_by_characters(text);
        }
        
        // 返回得分最高的语言
        let best_language = language_scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(lang, _)| lang.clone())
            .unwrap_or(Language::Unknown);
        
        Ok(best_language)
    }
    
    /// 检测单词语言
    pub fn detect_word_language(&self, word: &str) -> Result<Language, TransformerError> {
        if word.is_empty() {
            return Ok(Language::Unknown);
        }
        
        // 检查是否为CJK字符
        if self.is_cjk_character(word) {
            return self.detect_cjk_language(word);
        }
        
        // 检查是否为阿拉伯语字符
        if self.is_arabic_character(word) {
            return Ok(Language::Arabic);
        }
        
        // 检查是否为俄语字符
        if self.is_russian_character(word) {
            return Ok(Language::Russian);
        }
        
        // 默认为英语
        Ok(Language::English)
    }
    
    /// 基于字符检测语言
    fn detect_language_by_characters(&self, text: &str) -> Result<Language, TransformerError> {
        let mut cjk_count = 0;
        let mut arabic_count = 0;
        let mut russian_count = 0;
        let mut latin_count = 0;
        
        for ch in text.chars() {
            if self.is_cjk_character(&ch.to_string()) {
                cjk_count += 1;
            } else if self.is_arabic_character(&ch.to_string()) {
                arabic_count += 1;
            } else if self.is_russian_character(&ch.to_string()) {
                russian_count += 1;
            } else if ch.is_ascii_alphabetic() {
                latin_count += 1;
            }
        }
        
        let total_chars = text.chars().count();
        if total_chars == 0 {
            return Ok(Language::Unknown);
        }
        
        let cjk_ratio = cjk_count as f32 / total_chars as f32;
        let arabic_ratio = arabic_count as f32 / total_chars as f32;
        let russian_ratio = russian_count as f32 / total_chars as f32;
        let latin_ratio = latin_count as f32 / total_chars as f32;
        
        if cjk_ratio > 0.3 {
            return self.detect_cjk_language(text);
        } else if arabic_ratio > 0.3 {
            return Ok(Language::Arabic);
        } else if russian_ratio > 0.3 {
            return Ok(Language::Russian);
        } else if latin_ratio > 0.5 {
            return Ok(Language::English);
        }
        
        Ok(Language::Unknown)
    }
    
    /// 检测CJK语言
    fn detect_cjk_language(&self, text: &str) -> Result<Language, TransformerError> {
        // 简单的CJK语言检测
        let mut japanese_count = 0;
        let mut korean_count = 0;
        let mut chinese_count = 0;
        
        for ch in text.chars() {
            let code = ch as u32;
            
            // 日语假名范围
            if (0x3040..=0x309F).contains(&code) || (0x30A0..=0x30FF).contains(&code) {
                japanese_count += 1;
            }
            // 韩语谚文范围
            else if (0xAC00..=0xD7AF).contains(&code) {
                korean_count += 1;
            }
            // 中文字符范围
            else if (0x4E00..=0x9FFF).contains(&code) {
                chinese_count += 1;
            }
        }
        
        if japanese_count > korean_count && japanese_count > chinese_count {
            Ok(Language::Japanese)
        } else if korean_count > chinese_count {
            Ok(Language::Korean)
        } else {
            Ok(Language::Chinese)
        }
    }
    
    /// 检查是否为CJK字符
    fn is_cjk_character(&self, text: &str) -> bool {
        text.chars().any(|ch| {
            let code = ch as u32;
            (0x4E00..=0x9FFF).contains(&code) || // 中文
            (0x3040..=0x309F).contains(&code) || // 日语平假名
            (0x30A0..=0x30FF).contains(&code) || // 日语片假名
            (0xAC00..=0xD7AF).contains(&code)    // 韩语谚文
        })
    }
    
    /// 检查是否为阿拉伯语字符
    fn is_arabic_character(&self, text: &str) -> bool {
        text.chars().any(|ch| {
            let code = ch as u32;
            (0x0600..=0x06FF).contains(&code) || // 阿拉伯语
            (0x0750..=0x077F).contains(&code) || // 阿拉伯语补充
            (0x08A0..=0x08FF).contains(&code)    // 阿拉伯语扩展
        })
    }
    
    /// 检查是否为俄语字符
    fn is_russian_character(&self, text: &str) -> bool {
        text.chars().any(|ch| {
            let code = ch as u32;
            (0x0400..=0x04FF).contains(&code) // 西里尔字母
        })
    }
}

/// 语言预处理器特征
pub trait LanguagePreprocessor: Send + Sync {
    /// 预处理文本
    fn preprocess(&self, text: &str) -> Result<String, TransformerError>;
    
    /// 检查字符是否为中文
    fn is_chinese_char(&self, ch: char) -> bool {
        let code = ch as u32;
        (0x4E00..=0x9FFF).contains(&code) || // 基本汉字
        (0x3400..=0x4DBF).contains(&code) || // 汉字扩展A
        (0x20000..=0x2A6DF).contains(&code)  // 汉字扩展B
    }
    
    /// 检查字符是否为日语
    fn is_japanese_char(&self, ch: char) -> bool {
        let code = ch as u32;
        (0x3040..=0x309F).contains(&code) || // 平假名
        (0x30A0..=0x30FF).contains(&code) || // 片假名
        (0x4E00..=0x9FFF).contains(&code)    // 汉字
    }
    
    /// 检查字符是否为韩语
    fn is_korean_char(&self, ch: char) -> bool {
        let code = ch as u32;
        (0xAC00..=0xD7AF).contains(&code) || // 谚文
        (0x1100..=0x11FF).contains(&code) || // 谚文字母
        (0x3130..=0x318F).contains(&code)    // 谚文兼容字母
    }
    
    /// 检查字符是否为阿拉伯语
    fn is_arabic_char(&self, ch: char) -> bool {
        let code = ch as u32;
        (0x0600..=0x06FF).contains(&code) || // 阿拉伯语
        (0x0750..=0x077F).contains(&code) || // 阿拉伯语补充
        (0x08A0..=0x08FF).contains(&code)    // 阿拉伯语扩展
    }
}

/// 英语预处理器
#[derive(Debug)]
pub struct EnglishPreprocessor;

impl LanguagePreprocessor for EnglishPreprocessor {
    fn preprocess(&self, text: &str) -> Result<String, TransformerError> {
        let mut processed = text.to_lowercase();
        
        // 移除标点符号
        processed = processed.chars()
            .filter(|&c| c.is_alphanumeric() || c.is_whitespace())
            .collect();
        
        // 清理多余空格
        processed = processed.split_whitespace().collect::<Vec<&str>>().join(" ");
        
        Ok(processed)
    }
}

/// 中文预处理器
#[derive(Debug)]
pub struct ChinesePreprocessor;

impl LanguagePreprocessor for ChinesePreprocessor {
    fn preprocess(&self, text: &str) -> Result<String, TransformerError> {
        let mut processed = String::new();
        
        for ch in text.chars() {
            if ch.is_whitespace() || self.is_chinese_char(ch) {
                processed.push(ch);
            }
        }
        
        // 清理多余空格
        processed = processed.split_whitespace().collect::<Vec<&str>>().join(" ");
        
        Ok(processed)
    }
    
    fn is_chinese_char(&self, ch: char) -> bool {
        let code = ch as u32;
        (0x4E00..=0x9FFF).contains(&code) || // 基本汉字
        (0x3400..=0x4DBF).contains(&code) || // 汉字扩展A
        (0x20000..=0x2A6DF).contains(&code)  // 汉字扩展B
    }
}

/// 日语预处理器
#[derive(Debug)]
pub struct JapanesePreprocessor;

impl LanguagePreprocessor for JapanesePreprocessor {
    fn preprocess(&self, text: &str) -> Result<String, TransformerError> {
        let mut processed = String::new();
        
        for ch in text.chars() {
            if ch.is_whitespace() || self.is_japanese_char(ch) {
                processed.push(ch);
            }
        }
        
        // 清理多余空格
        processed = processed.split_whitespace().collect::<Vec<&str>>().join(" ");
        
        Ok(processed)
    }
    
    fn is_japanese_char(&self, ch: char) -> bool {
        let code = ch as u32;
        (0x3040..=0x309F).contains(&code) || // 平假名
        (0x30A0..=0x30FF).contains(&code) || // 片假名
        (0x4E00..=0x9FFF).contains(&code)    // 汉字
    }
}

/// 韩语预处理器
#[derive(Debug)]
pub struct KoreanPreprocessor;

impl LanguagePreprocessor for KoreanPreprocessor {
    fn preprocess(&self, text: &str) -> Result<String, TransformerError> {
        let mut processed = String::new();
        
        for ch in text.chars() {
            if ch.is_whitespace() || self.is_korean_char(ch) {
                processed.push(ch);
            }
        }
        
        // 清理多余空格
        processed = processed.split_whitespace().collect::<Vec<&str>>().join(" ");
        
        Ok(processed)
    }
    
    fn is_korean_char(&self, ch: char) -> bool {
        let code = ch as u32;
        (0xAC00..=0xD7AF).contains(&code) || // 谚文
        (0x1100..=0x11FF).contains(&code) || // 谚文字母
        (0x3130..=0x318F).contains(&code)    // 谚文兼容字母
    }
}

/// 阿拉伯语预处理器
#[derive(Debug)]
pub struct ArabicPreprocessor;

impl LanguagePreprocessor for ArabicPreprocessor {
    fn preprocess(&self, text: &str) -> Result<String, TransformerError> {
        let mut processed = String::new();
        
        for ch in text.chars() {
            if ch.is_whitespace() || self.is_arabic_char(ch) {
                processed.push(ch);
            }
        }
        
        // 清理多余空格
        processed = processed.split_whitespace().collect::<Vec<&str>>().join(" ");
        
        Ok(processed)
    }
    
    fn is_arabic_char(&self, ch: char) -> bool {
        let code = ch as u32;
        (0x0600..=0x06FF).contains(&code) || // 阿拉伯语
        (0x0750..=0x077F).contains(&code) || // 阿拉伯语补充
        (0x08A0..=0x08FF).contains(&code)    // 阿拉伯语扩展
    }
}

/// 俄语预处理器
#[derive(Debug)]
pub struct RussianPreprocessor;

impl LanguagePreprocessor for RussianPreprocessor {
    fn preprocess(&self, text: &str) -> Result<String, TransformerError> {
        let mut processed = text.to_lowercase();
        
        // 移除标点符号
        processed = processed.chars()
            .filter(|&c| c.is_alphanumeric() || c.is_whitespace())
            .collect();
        
        // 清理多余空格
        processed = processed.split_whitespace().collect::<Vec<&str>>().join(" ");
        
        Ok(processed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_detection() {
        let detector = LanguageDetector::new();
        
        let english_text = "Hello world";
        let detected = detector.detect_language(english_text).unwrap();
        assert_eq!(detected, Language::English);
        
        let chinese_text = "你好世界";
        let detected = detector.detect_language(chinese_text).unwrap();
        assert_eq!(detected, Language::Chinese);
    }

    #[test]
    fn test_language_processor() {
        let config = LanguageProcessingConfig::default();
        let processor = LanguageProcessor::new(config);
        
        let processed = processor.process_text("Hello world").unwrap();
        assert_eq!(processed.detected_language, Some(Language::English));
        assert!(!processed.processed_text.is_empty());
    }

    #[test]
    fn test_mixed_language_detection() {
        let detector = LanguageDetector::new();
        let text = "Hello 世界";
        let languages = detector.detect_mixed_languages(text).unwrap();
        assert!(languages.len() > 1);
    }
} 