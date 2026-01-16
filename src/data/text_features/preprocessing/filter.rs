// Text Filter Implementation
// 文本过滤器实现

use crate::Result;
use std::collections::HashSet;
use crate::Error;
use std::path::Path;
use std::fs;
use log::{debug, info};

/// 文本过滤器特征
pub trait TextFilter: Send + Sync {
    /// 过滤文本令牌
    fn filter(&self, tokens: &[String]) -> Result<Vec<String>>;
    
    /// 获取过滤器名称
    fn name(&self) -> &str;
}

/// 长度过滤器 - 过滤长度不符合条件的令牌
pub struct LengthFilter {
    /// 最小长度
    min_length: usize,
    /// 最大长度（如果有）
    max_length: Option<usize>,
    /// 过滤器名称
    name: String,
}

impl LengthFilter {
    /// 创建新的长度过滤器
    pub fn new(min_length: usize, max_length: Option<usize>) -> Self {
        Self {
            min_length,
            max_length,
            name: "length_filter".to_string(),
        }
    }
}

impl TextFilter for LengthFilter {
    fn filter(&self, tokens: &[String]) -> Result<Vec<String>> {
        let filtered = tokens.iter()
            .filter(|token| {
                let len = token.len();
                len >= self.min_length && self.max_length.map_or(true, |max| len <= max)
            })
            .cloned()
            .collect();
        
        Ok(filtered)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// 停用词过滤器 - 过滤停用词
pub struct StopwordFilter {
    /// 停用词集合
    stopwords: HashSet<String>,
    /// 过滤器名称
    name: String,
}

impl StopwordFilter {
    /// 创建新的停用词过滤器
    pub fn new(language: String) -> Self {
        // 简单实现，在实际应用中应该加载特定语言的停用词列表
        let mut stopwords = HashSet::new();
        
        // 默认英文停用词
        if language == "en" {
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
                "weren't", "won", "won't", "wouldn", "wouldn't"
            ];
            
            for word in default_stopwords {
                stopwords.insert(word.to_string());
            }
        } else if language == "zh" {
            // 简体中文停用词（示例）
            let default_stopwords = vec![
                "的", "了", "和", "是", "在", "我", "有", "个", "这", "那",
                "你", "们", "他", "她", "它", "啊", "吧", "呢", "吗", "嗯",
                "哦", "哪", "什么", "怎么", "为什么", "如何", "这个", "那个",
                "一个", "一些", "一样", "一直", "才", "把", "被", "比", "不",
                "从", "但", "打", "到", "对", "多", "而", "儿", "方", "给",
                "跟", "还", "好", "和", "后", "会", "或", "几", "家", "将"
            ];
            
            for word in default_stopwords {
                stopwords.insert(word.to_string());
            }
        }
        
        Self {
            stopwords,
            name: format!("stopword_filter_{}", language),
        }
    }
    
    /// 使用自定义停用词列表
    pub fn with_custom_stopwords(stopwords: HashSet<String>) -> Self {
        Self {
            stopwords,
            name: "custom_stopword_filter".to_string(),
        }
    }
    
    /// 加载停用词列表
    pub fn load_stopwords(&mut self, path: &str) -> Result<()> {
        debug!("加载停用词列表: {}", path);
        
        if !Path::new(path).exists() {
            return Err(Error::IoError(format!("停用词文件不存在: {}", path)));
        }
        
        let content = fs::read_to_string(path)
            .map_err(|e| Error::IoError(format!("读取停用词文件失败: {}", e)))?;
        
        for line in content.lines() {
            let word = line.trim();
            if !word.is_empty() {
                self.stopwords.insert(word.to_string());
            }
        }
        
        info!("已加载 {} 个停用词", self.stopwords.len());
        
        Ok(())
    }
}

impl TextFilter for StopwordFilter {
    fn filter(&self, tokens: &[String]) -> Result<Vec<String>> {
        let filtered = tokens.iter()
            .filter(|token| !self.stopwords.contains(&token.to_lowercase()))
            .cloned()
            .collect();
        
        Ok(filtered)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// 正则表达式过滤器 - 使用正则表达式过滤令牌
pub struct RegexFilter {
    /// 正则表达式
    regex: regex::Regex,
    /// 是否保留匹配项
    keep_matches: bool,
    /// 过滤器名称
    name: String,
}

impl RegexFilter {
    /// 创建新的正则表达式过滤器
    pub fn new(pattern: &str, keep_matches: bool) -> Result<Self> {
        let regex = regex::Regex::new(pattern)
            .map_err(|e| Error::invalid_argument(format!("无效的正则表达式: {}", e)))?;
        
        Ok(Self {
            regex,
            keep_matches,
            name: "regex_filter".to_string(),
        })
    }
}

impl TextFilter for RegexFilter {
    fn filter(&self, tokens: &[String]) -> Result<Vec<String>> {
        let filtered = tokens.iter()
            .filter(|token| {
                let matches = self.regex.is_match(token);
                (matches && self.keep_matches) || (!matches && !self.keep_matches)
            })
            .cloned()
            .collect();
        
        Ok(filtered)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// 自定义过滤器 - 使用自定义函数过滤令牌
pub struct CustomFilter<F>
where
    F: Fn(&str) -> bool + Send + Sync,
{
    /// 过滤函数
    filter_fn: F,
    /// 过滤器名称
    name: String,
}

impl<F> CustomFilter<F>
where
    F: Fn(&str) -> bool + Send + Sync,
{
    /// 创建新的自定义过滤器
    pub fn new(filter_fn: F, name: &str) -> Self {
        Self {
            filter_fn,
            name: name.to_string(),
        }
    }
}

impl<F> TextFilter for CustomFilter<F>
where
    F: Fn(&str) -> bool + Send + Sync,
{
    fn filter(&self, tokens: &[String]) -> Result<Vec<String>> {
        let filtered = tokens.iter()
            .filter(|token| (self.filter_fn)(token))
            .cloned()
            .collect();
        
        Ok(filtered)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

/// 创建基于语言的停用词过滤器
pub fn create_stopword_filter(language: &str) -> StopwordFilter {
    StopwordFilter::new(language.to_string())
}

/// 创建长度过滤器
pub fn create_length_filter(min_length: usize, max_length: Option<usize>) -> LengthFilter {
    LengthFilter::new(min_length, max_length)
}

/// 创建正则表达式过滤器
pub fn create_regex_filter(pattern: &str, keep_matches: bool) -> Result<RegexFilter> {
    RegexFilter::new(pattern, keep_matches)
} 