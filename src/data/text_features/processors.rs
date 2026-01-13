use crate::Result;
// 导入TextFeatures类型
use crate::data::text_features::types::TextFeatures;
// use super::TextFeatures;

pub trait TextProcessor {
    fn process(&self, text: &str) -> Result<TextFeatures>;
}

pub struct BasicTextProcessor;

impl BasicTextProcessor {
    pub fn new() -> Self {
        BasicTextProcessor
    }
}

impl TextProcessor for BasicTextProcessor {
    fn process(&self, text: &str) -> Result<TextFeatures> {
        TextFeatures::extract(text)
    }
}

pub struct AdvancedTextProcessor {
    pub remove_stopwords: bool,
    pub lowercase: bool,
}

impl AdvancedTextProcessor {
    pub fn new(remove_stopwords: bool, lowercase: bool) -> Self {
        AdvancedTextProcessor {
            remove_stopwords,
            lowercase,
        }
    }
    
    fn preprocess(&self, text: &str) -> String {
        let mut processed = text.to_string();
        
        // 1. 移除多余的空白字符
        processed = self.normalize_whitespace(&processed);
        
        // 2. 转换为小写
        if self.lowercase {
            processed = processed.to_lowercase();
        }
        
        // 3. 清理标点符号
        processed = self.clean_punctuation(&processed);
        
        // 4. 移除停用词
        if self.remove_stopwords {
            processed = self.remove_stopwords_advanced(&processed);
        }
        
        // 5. 标准化词汇
        processed = self.normalize_tokens(&processed);
        
        processed
    }
    
    /// 规范化空白字符
    fn normalize_whitespace(&self, text: &str) -> String {
        // 将多个空白字符替换为单个空格
        let mut result = String::new();
        let mut prev_was_space = false;
        
        for ch in text.chars() {
            if ch.is_whitespace() {
                if !prev_was_space {
                    result.push(' ');
                    prev_was_space = true;
                }
            } else {
                result.push(ch);
                prev_was_space = false;
            }
        }
        
        result.trim().to_string()
    }
    
    /// 清理标点符号
    fn clean_punctuation(&self, text: &str) -> String {
        let mut result = String::new();
        
        for ch in text.chars() {
            if ch.is_ascii_punctuation() {
                // 保留一些重要的标点符号，如句号、问号、感叹号
                if matches!(ch, '.' | '?' | '!' | ',' | ';' | ':') {
                    result.push(' ');
                    result.push(ch);
                    result.push(' ');
                } else {
                    result.push(' ');
                }
            } else {
                result.push(ch);
            }
        }
        
        self.normalize_whitespace(&result)
    }
    
    /// 高级停用词移除
    fn remove_stopwords_advanced(&self, text: &str) -> String {
        let stopwords = self.get_stopwords();
        let words: Vec<&str> = text.split_whitespace().collect();
        
        let filtered_words: Vec<&str> = words
            .into_iter()
            .filter(|word| {
                let clean_word = word.trim_matches(|c: char| c.is_ascii_punctuation()).to_lowercase();
                !stopwords.contains(&clean_word.as_str())
            })
            .collect();
        
        filtered_words.join(" ")
    }
    
    /// 获取停用词列表
    fn get_stopwords(&self) -> std::collections::HashSet<&'static str> {
        let stopwords = [
            // 常见英语停用词
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "will", "with", "would", "could", "should", "may",
            "might", "can", "had", "have", "been", "being", "do", "does",
            "did", "this", "these", "those", "they", "them", "their", "his",
            "her", "him", "she", "we", "you", "your", "our", "us", "me",
            "my", "mine", "yours", "theirs", "ours", "himself", "herself",
            "itself", "myself", "yourself", "ourselves", "themselves",
            "what", "which", "who", "whom", "whose", "where", "when", "why",
            "how", "all", "any", "both", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "now", "then",
            "here", "there", "up", "down", "out", "off", "over", "under",
            "again", "further", "once", "before", "after", "above", "below",
            "between", "through", "during", "about", "against", "into",
            "onto", "upon", "within", "without", "across", "along", "around",
            "behind", "beside", "beyond", "inside", "outside", "toward",
            "towards", "underneath", "until", "since", "while", "although",
            "because", "if", "unless", "until", "whether", "though", "even"
        ];
        
        stopwords.iter().cloned().collect()
    }
    
    /// 标准化词汇
    fn normalize_tokens(&self, text: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        
        let normalized_words: Vec<String> = words
            .into_iter()
            .map(|word| {
                let mut normalized = word.to_string();
                
                // 移除前后的标点符号
                normalized = normalized.trim_matches(|c: char| c.is_ascii_punctuation()).to_string();
                
                // 处理缩写
                normalized = self.expand_contractions(&normalized);
                
                // 处理数字
                normalized = self.normalize_numbers(&normalized);
                
                normalized
            })
            .filter(|word| !word.is_empty() && word.len() > 1) // 过滤空字符串和单字符
            .collect();
        
        normalized_words.join(" ")
    }
    
    /// 展开缩写
    fn expand_contractions(&self, word: &str) -> String {
        let contractions = [
            ("don't", "do not"),
            ("won't", "will not"),
            ("can't", "cannot"),
            ("n't", " not"),
            ("'re", " are"),
            ("'ve", " have"),
            ("'ll", " will"),
            ("'d", " would"),
            ("'m", " am"),
            ("'s", " is"),
        ];
        
        let mut expanded = word.to_string();
        for (contraction, expansion) in contractions.iter() {
            if expanded.contains(contraction) {
                expanded = expanded.replace(contraction, expansion);
            }
        }
        
        expanded
    }
    
    /// 标准化数字
    fn normalize_numbers(&self, word: &str) -> String {
        if word.chars().all(|c| c.is_ascii_digit()) {
            // 将纯数字替换为标记
            "<NUMBER>".to_string()
        } else if word.chars().any(|c| c.is_ascii_digit()) && word.chars().any(|c| c.is_ascii_alphabetic()) {
            // 包含数字和字母的混合词（如版本号、型号等）
            "<ALPHANUM>".to_string()
        } else {
            word.to_string()
        }
    }
}

impl TextProcessor for AdvancedTextProcessor {
    fn process(&self, text: &str) -> Result<TextFeatures> {
        let preprocessed = self.preprocess(text);
        TextFeatures::extract(&preprocessed)
    }
} 