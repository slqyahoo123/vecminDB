// src/data/text_features/tokenizer.rs
//
// Transformer 分词器模块

use std::collections::HashMap;
use std::path::Path;
use std::fs::File;
use std::io::{BufRead, BufReader};
use log::info;

// use crate::Error;
use super::error::TransformerError;
use super::config::TransformerConfig;

/// 词汇表管理器
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// 词汇表（词对应ID）
    vocab: HashMap<String, usize>,
    /// 反向词汇表（ID对应词）
    id_to_token: HashMap<usize, String>,
    /// 特殊标记
    special_tokens: HashMap<String, usize>,
    /// 词汇表大小
    size: usize,
}

impl Vocabulary {
    /// 创建新的词汇表
    pub fn new() -> Self {
        let mut special_tokens = HashMap::new();
        special_tokens.insert("[PAD]".to_string(), 0);
        special_tokens.insert("[UNK]".to_string(), 1);
        special_tokens.insert("[CLS]".to_string(), 2);
        special_tokens.insert("[SEP]".to_string(), 3);
        special_tokens.insert("[MASK]".to_string(), 4);
        
        Self {
            vocab: HashMap::new(),
            id_to_token: HashMap::new(),
            special_tokens,
            size: 0,
        }
    }
    
    /// 从文件加载词汇表
    pub fn load_from_file(&mut self, vocab_file: &Path) -> Result<(), TransformerError> {
        info!("从文件加载词汇表: {:?}", vocab_file);
        
        let file = File::open(vocab_file)
            .map_err(|e| TransformerError::IoError(e))?;
        
        let reader = BufReader::new(file);
        let mut next_id = 0;
        
        for line in reader.lines() {
            let token = line.map_err(|e| TransformerError::IoError(e))?;
            let token = token.trim();
            
            if !token.is_empty() {
                self.vocab.insert(token.to_string(), next_id);
                self.id_to_token.insert(next_id, token.to_string());
                next_id += 1;
            }
        }
        
        self.size = next_id;
        info!("词汇表加载完成，大小: {}", self.size);
        Ok(())
    }
    
    /// 生成BPE词汇表
    pub fn generate_bpe_vocabulary(&mut self, config: &TransformerConfig) -> Result<(), TransformerError> {
        info!("生成BPE词汇表");
        
        // 添加特殊标记到词汇表
        for (token, id) in &self.special_tokens {
            self.vocab.insert(token.clone(), *id);
            self.id_to_token.insert(*id, token.clone());
        }
        
        let mut next_id = self.special_tokens.len();
        
        // 添加基础词汇
        let basic_words = vec![
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "this", "that", "these", "those",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
            "us", "them", "my", "your", "his", "her", "its", "our", "their",
            "mine", "yours", "hers", "ours", "theirs", "am", "is", "are",
            "was", "were", "been", "being", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may", "might", "can",
            "shall", "must", "ought", "need", "dare", "used", "going", "gonna",
            "wanna", "gotta", "lemme", "gimme", "gonna", "wanna", "gotta",
            "lemme", "gimme", "gonna", "wanna", "gotta", "lemme", "gimme"
        ];
        
        for word in basic_words {
            if next_id < config.vocab_size {
                self.vocab.insert(word.to_string(), next_id);
                self.id_to_token.insert(next_id, word.to_string());
                next_id += 1;
            }
        }
        
        // 添加字符级别的词汇
        let chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        for ch in chars.chars() {
            if next_id < config.vocab_size {
                let token = ch.to_string();
                if !self.vocab.contains_key(&token) {
                    self.vocab.insert(token.clone(), next_id);
                    self.id_to_token.insert(next_id, token);
                    next_id += 1;
                }
            }
        }
        
        // 添加标点符号
        let punctuation = ".,;:!?\"'()[]{}+-*/=<>@#$%^&|\\~`";
        for ch in punctuation.chars() {
            if next_id < config.vocab_size {
                let token = ch.to_string();
                if !self.vocab.contains_key(&token) {
                    self.vocab.insert(token.clone(), next_id);
                    self.id_to_token.insert(next_id, token);
                    next_id += 1;
                }
            }
        }
        
        self.size = next_id;
        info!("BPE词汇表生成完成，大小: {}", self.size);
        Ok(())
    }
    
    /// 获取词汇表大小
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// 检查词汇表是否为空
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
    
    /// 获取token的ID
    pub fn get_id(&self, token: &str) -> Option<usize> {
        self.vocab.get(token).copied()
    }
    
    /// 根据ID获取token
    pub fn get_token(&self, id: usize) -> Option<&String> {
        self.id_to_token.get(&id)
    }
    
    /// 获取未知token的ID
    pub fn get_unk_id(&self) -> usize {
        *self.special_tokens.get("[UNK]").unwrap_or(&1)
    }
    
    /// 获取填充token的ID
    pub fn get_pad_id(&self) -> usize {
        *self.special_tokens.get("[PAD]").unwrap_or(&0)
    }
    
    /// 获取分类token的ID
    pub fn get_cls_id(&self) -> usize {
        *self.special_tokens.get("[CLS]").unwrap_or(&2)
    }
    
    /// 获取分隔token的ID
    pub fn get_sep_id(&self) -> usize {
        *self.special_tokens.get("[SEP]").unwrap_or(&3)
    }
    
    /// 获取掩码token的ID
    pub fn get_mask_id(&self) -> usize {
        *self.special_tokens.get("[MASK]").unwrap_or(&4)
    }
    
    /// 检查是否为特殊token
    pub fn is_special_token(&self, token: &str) -> bool {
        self.special_tokens.contains_key(token)
    }
    
    /// 添加新token到词汇表
    pub fn add_token(&mut self, token: String) -> usize {
        if let Some(&id) = self.vocab.get(&token) {
            id
        } else {
            let id = self.size;
            self.vocab.insert(token.clone(), id);
            self.id_to_token.insert(id, token);
            self.size += 1;
            id
        }
    }
}

/// 分词器
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// 词汇表
    vocabulary: Vocabulary,
    /// 配置
    config: TransformerConfig,
    /// 是否区分大小写
    case_sensitive: bool,
}

impl Tokenizer {
    /// 创建新的分词器
    pub fn new(config: TransformerConfig) -> Self {
        Self {
            vocabulary: Vocabulary::new(),
            case_sensitive: config.case_sensitive,
            config,
        }
    }
    
    /// 初始化分词器
    pub fn initialize(&mut self) -> Result<(), TransformerError> {
        // 尝试从文件加载词汇表
        let vocab_path = std::path::Path::new(&self.config.model_path).join("vocab.txt");
        
        if vocab_path.exists() {
            self.vocabulary.load_from_file(&vocab_path)?;
        } else {
            // 生成BPE词汇表
            self.vocabulary.generate_bpe_vocabulary(&self.config)?;
        }
        
        Ok(())
    }
    
    /// 对文本进行分词
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return vec![];
        }
        
        let text = if !self.case_sensitive {
            text.to_lowercase()
        } else {
            text.to_string()
        };
        
        // 预处理文本
        let preprocessed = self.preprocess_text(&text);
        
        // 基于空格的分词
        let mut tokens = Vec::new();
        
        for word in preprocessed.split_whitespace() {
            if word.is_empty() {
                continue;
            }
            
            let word_tokens = self.tokenize_word(word);
            tokens.extend(word_tokens);
        }
        
        tokens
    }
    
    /// 对单词进行分词
    fn tokenize_word(&self, word: &str) -> Vec<String> {
        if word.is_empty() {
            return vec![];
        }
        
        // 如果词在词汇表中直接存在，返回它
        if self.vocabulary.vocab.contains_key(word) {
            return vec![word.to_string()];
        }
        
        // 简单的字符级分词作为后备
        let mut tokens = Vec::new();
        let mut current_token = String::new();
        
        for ch in word.chars() {
            if ch.is_alphanumeric() {
                current_token.push(ch);
            } else {
                if !current_token.is_empty() {
                    tokens.push(current_token.clone());
                    current_token.clear();
                }
                if !ch.is_whitespace() {
                    tokens.push(ch.to_string());
                }
            }
        }
        
        if !current_token.is_empty() {
            tokens.push(current_token);
        }
        
        tokens
    }
    
    /// 文本预处理
    fn preprocess_text(&self, text: &str) -> String {
        let mut result = String::new();
        let mut chars = text.chars().peekable();
        
        while let Some(ch) = chars.next() {
            match ch {
                // 处理标点符号，在前后添加空格
                '.' | ',' | ';' | ':' | '!' | '?' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' => {
                    result.push(' ');
                    result.push(ch);
                    result.push(' ');
                }
                // 处理数字和字母
                c if c.is_alphanumeric() => {
                    result.push(c);
                }
                // 处理空白字符
                c if c.is_whitespace() => {
                    result.push(' ');
                }
                // 其他字符
                _ => {
                    result.push(' ');
                    result.push(ch);
                    result.push(' ');
                }
            }
        }
        
        // 清理多余的空格
        result.split_whitespace().collect::<Vec<&str>>().join(" ")
    }
    
    /// 将文本编码为token ID序列
    pub fn encode(&self, text: &str) -> Result<Vec<i32>, TransformerError> {
        if self.vocabulary.is_empty() {
            return Err(TransformerError::vocabulary_error("词汇表未初始化"));
        }
        
        // 分词
        let tokens = self.tokenize(text);
        
        // 转换为ID
        let mut ids = Vec::with_capacity(tokens.len() + 2); // +2 for [CLS] and [SEP]
        
        // 添加[CLS]标记
        ids.push(self.vocabulary.get_cls_id() as i32);
        
        // 添加文本token
        for token in &tokens {
            let id = self.vocabulary.get_id(token).unwrap_or(self.vocabulary.get_unk_id());
            ids.push(id as i32);
            
            // 如果超过最大长度，则截断
            if ids.len() >= self.config.max_seq_length - 1 { // -1 留给[SEP]
                break;
            }
        }
        
        // 添加[SEP]标记
        ids.push(self.vocabulary.get_sep_id() as i32);
        
        // 填充或截断
        self.pad_or_truncate(&mut ids);
        
        Ok(ids)
    }
    
    /// 填充或截断序列
    fn pad_or_truncate(&self, ids: &mut Vec<i32>) {
        let max_length = self.config.max_seq_length;
        
        if ids.len() > max_length {
            // 截断
            ids.truncate(max_length);
            // 确保最后一个token是[SEP]
            ids[max_length - 1] = self.vocabulary.get_sep_id() as i32;
        } else if ids.len() < max_length {
            // 填充
            let pad_id = self.vocabulary.get_pad_id() as i32;
            while ids.len() < max_length {
                ids.push(pad_id);
            }
        }
    }
    
    /// 将token ID序列解码为文本
    pub fn decode(&self, ids: &[i32]) -> Result<String, TransformerError> {
        let mut tokens = Vec::new();
        
        for &id in ids {
            if let Some(token) = self.vocabulary.get_token(id as usize) {
                if !self.vocabulary.is_special_token(token) {
                    tokens.push(token.clone());
                }
            }
        }
        
        Ok(tokens.join(" "))
    }
    
    /// 获取词汇表
    pub fn vocabulary(&self) -> &Vocabulary {
        &self.vocabulary
    }
    
    /// 获取配置
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary_creation() {
        let vocab = Vocabulary::new();
        assert!(!vocab.is_empty());
        assert_eq!(vocab.get_unk_id(), 1);
        assert_eq!(vocab.get_pad_id(), 0);
    }

    #[test]
    fn test_tokenizer_initialization() {
        let config = TransformerConfig::default();
        let mut tokenizer = Tokenizer::new(config);
        assert!(tokenizer.initialize().is_ok());
    }

    #[test]
    fn test_tokenization() {
        let config = TransformerConfig::default();
        let mut tokenizer = Tokenizer::new(config);
        tokenizer.initialize().unwrap();
        
        let tokens = tokenizer.tokenize("Hello world!");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_encoding_decoding() {
        let config = TransformerConfig::default();
        let mut tokenizer = Tokenizer::new(config);
        tokenizer.initialize().unwrap();
        
        let text = "Hello world";
        let ids = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&ids).unwrap();
        
        assert!(!ids.is_empty());
        assert!(!decoded.is_empty());
    }
} 