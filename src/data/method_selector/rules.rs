// Rules for Method Selector
// 方法选择器的规则系统

use crate::data::text_features::config::TextFeatureMethod;
use crate::data::method_selector::types::DomainRule;
use serde_json::Value;

/// 规则构建器
pub struct RuleBuilder;

impl RuleBuilder {
    /// 创建新的规则构建器
    pub fn new() -> Self {
        Self
    }
    
    /// 创建文本长度规则
    pub fn text_length_rule(field: &str, min_length: usize, max_length: usize, method: TextFeatureMethod, priority: u8) -> DomainRule {
        let field = field.to_string();
        let condition = Box::new(move |data: &[Value]| -> bool {
            if data.is_empty() {
                return false;
            }
            
            // 检查字段是否存在
            if !data[0].get(&field).is_some() {
                return false;
            }
            
            // 统计符合条件的数据比例
            let mut match_count = 0;
            let mut valid_count = 0;
            
            for item in data.iter() {
                if let Some(value) = item.get(&field) {
                    if let Some(text) = value.as_str() {
                        valid_count += 1;
                        let length = text.len();
                        if length >= min_length && length <= max_length {
                            match_count += 1;
                        }
                    }
                }
            }
            
            // 如果有效字段不足，返回false
            if valid_count < 10 {
                return false;
            }
            
            // 如果超过75%的数据符合条件，返回true
            ((match_count as f64) / (valid_count as f64)) > 0.75
        });
        
        DomainRule {
            condition,
            recommended_method: method,
            priority,
        }
    }
    
    /// 创建词汇量规则
    pub fn vocabulary_size_rule(field: &str, min_vocab: usize, max_vocab: usize, method: TextFeatureMethod, priority: u8) -> DomainRule {
        let field = field.to_string();
        let condition = Box::new(move |data: &[Value]| -> bool {
            if data.is_empty() {
                return false;
            }
            
            // 检查字段是否存在
            if !data[0].get(&field).is_some() {
                return false;
            }
            
            // 统计唯一词汇
            use std::collections::HashSet;
            let mut all_words = HashSet::new();
            let mut valid_count = 0;
            
            for item in data.iter().take(100) { // 只检查前100个样本
                if let Some(value) = item.get(&field) {
                    if let Some(text) = value.as_str() {
                        valid_count += 1;
                        for word in text.split_whitespace() {
                            all_words.insert(word.to_lowercase());
                        }
                    }
                }
            }
            
            // 如果有效字段不足，返回false
            if valid_count < 10 {
                return false;
            }
            
            let vocab_size = all_words.len();
            vocab_size >= min_vocab && vocab_size <= max_vocab
        });
        
        DomainRule {
            condition,
            recommended_method: method,
            priority,
        }
    }
    
    /// 创建语言规则
    pub fn language_rule(field: &str, language_patterns: Vec<&str>, method: TextFeatureMethod, priority: u8) -> DomainRule {
        let field = field.to_string();
        let patterns: Vec<String> = language_patterns.iter().map(|&s| s.to_string()).collect();
        
        let condition = Box::new(move |data: &[Value]| -> bool {
            if data.is_empty() {
                return false;
            }
            
            // 检查字段是否存在
            if !data[0].get(&field).is_some() {
                return false;
            }
            
            // 统计匹配特定语言模式的样本
            let mut match_count = 0;
            let mut valid_count = 0;
            
            for item in data.iter().take(50) { // 只检查前50个样本
                if let Some(value) = item.get(&field) {
                    if let Some(text) = value.as_str() {
                        valid_count += 1;
                        
                        // 检查是否匹配任何语言模式
                        let matches_pattern = patterns.iter().any(|pattern| {
                            text.to_lowercase().contains(&pattern.to_lowercase())
                        });
                        
                        if matches_pattern {
                            match_count += 1;
                        }
                    }
                }
            }
            
            // 如果有效字段不足，返回false
            if valid_count < 5 {
                return false;
            }
            
            // 如果超过60%的样本匹配语言模式，返回true
            ((match_count as f64) / (valid_count as f64)) > 0.6
        });
        
        DomainRule {
            condition,
            recommended_method: method,
            priority,
        }
    }
    
    /// 创建数值特征比例规则
    pub fn numeric_ratio_rule(min_ratio: f64, max_ratio: f64, method: TextFeatureMethod, priority: u8) -> DomainRule {
        let condition = Box::new(move |data: &[Value]| -> bool {
            if data.is_empty() {
                return false;
            }
            
            // 统计数值字段vs文本字段
            let mut numeric_fields = 0;
            let mut total_fields = 0;
            
            if let Some(obj) = data[0].as_object() {
                for (_, value) in obj {
                    total_fields += 1;
                    if value.is_number() {
                        numeric_fields += 1;
                    }
                }
            }
            
            if total_fields == 0 {
                return false;
            }
            
            let ratio = numeric_fields as f64 / total_fields as f64;
            ratio >= min_ratio && ratio <= max_ratio
        });
        
        DomainRule {
            condition,
            recommended_method: method,
            priority,
        }
    }
    
    /// 创建自定义规则
    pub fn custom_rule<F>(condition: F, method: TextFeatureMethod, priority: u8) -> DomainRule
    where
        F: Fn(&[Value]) -> bool + Send + Sync + 'static,
    {
        DomainRule {
            condition: Box::new(condition),
            recommended_method: method,
            priority,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_text_length_rule() {
        // 创建测试数据
        let data = vec![
            serde_json::json!({"text": "short text"}),
            serde_json::json!({"text": "this is a medium length text for testing"}),
        ];
        
        // 创建长度规则
        let rule = RuleBuilder::text_length_rule("text", 5, 50, TextFeatureMethod::TfIdf, 50);
        
        // 测试规则
        assert!(rule.condition(&data));
    }
} 