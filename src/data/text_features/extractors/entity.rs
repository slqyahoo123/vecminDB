use crate::{Result, Error};
use crate::data::text_features::config::TextFeatureConfig;
use super::FeatureExtractor;
use std::collections::HashMap;
use regex::Regex;

/// 实体特征提取器
/// 
/// 提取文本中的命名实体特征,支持多种实体类型
#[derive(Debug, Clone)]
pub struct EntityExtractor {
    /// 配置信息
    config: TextFeatureConfig,
    /// 实体类型列表
    entity_types: Vec<String>,
    /// 实体模式
    patterns: HashMap<String, Regex>,
    /// 特征维度
    dimension: usize,
    /// 实体映射
    entity_map: HashMap<String, usize>,
}

impl EntityExtractor {
    /// 创建新的实体特征提取器
    pub fn new(entity_types: Vec<String>) -> Result<Self> {
        if entity_types.is_empty() {
            return Err(Error::invalid_argument("实体类型列表不能为空".to_string()));
        }
        
        // 创建实体模式
        let mut patterns = HashMap::new();
        for entity_type in &entity_types {
            let pattern = match entity_type.as_str() {
                "email" => r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                "url" => r"https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
                "phone" => r"\+?[\d\s-()]{10,}",
                "date" => r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
                "time" => r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?",
                "number" => r"-?\d+(?:\.\d+)?",
                "currency" => r"[$€£¥]\s*\d+(?:\.\d{2})?",
                "percentage" => r"\d+(?:\.\d+)?%",
                _ => continue,
            };
            
            patterns.insert(entity_type.clone(), Regex::new(pattern).unwrap());
        }
        
        Ok(Self {
            config: TextFeatureConfig::default(),
            entity_types: entity_types.clone(),
            patterns,
            dimension: entity_types.len(),
            entity_map: HashMap::new(),
        })
    }
    
    /// 从配置创建实体特征提取器
    pub fn from_config(config: &TextFeatureConfig) -> Result<Self> {
        // 从配置的metadata中获取实体类型
        let entity_types = config.metadata.get("entity_types")
            .and_then(|s| {
                // 尝试从JSON字符串解析
                serde_json::from_str::<Vec<String>>(s).ok()
            })
            .unwrap_or_else(|| {
            vec![
                "email".to_string(),
                "url".to_string(),
                "phone".to_string(),
                "date".to_string(),
                "time".to_string(),
                "number".to_string(),
                "currency".to_string(),
                "percentage".to_string(),
            ]
        });
        
        Self::new(entity_types)
    }
    
    /// 提取实体特征
    fn extract_entities(&self, text: &str) -> HashMap<String, Vec<String>> {
        let mut entities: HashMap<String, Vec<String>> = HashMap::new();
        
        // 初始化实体列表
        for entity_type in &self.entity_types {
            entities.insert(entity_type.clone(), Vec::new());
        }
        
        // 提取每种类型的实体
        for (entity_type, pattern) in &self.patterns {
            let matches: Vec<String> = pattern
                .find_iter(text)
                .map(|m| m.as_str().to_string())
                .collect();
            
            if let Some(entity_list) = entities.get_mut(entity_type) {
                *entity_list = matches;
            }
        }
        
        entities
    }
    
    /// 更新实体映射
    fn update_entity_map(&mut self, entities: &HashMap<String, Vec<String>>) {
        // 统计实体频率
        let mut frequencies: HashMap<String, usize> = HashMap::new();
        
        for (entity_type, entity_list) in entities {
            for entity in entity_list {
                let key = format!("{}:{}", entity_type, entity);
                *frequencies.entry(key).or_insert(0) += 1;
            }
        }
        
        // 按频率排序
        let mut sorted_entities: Vec<_> = frequencies.into_iter().collect();
        sorted_entities.sort_by(|a, b| b.1.cmp(&a.1));
        
        // 更新实体映射
        self.entity_map.clear();
        for (i, (entity, _)) in sorted_entities.into_iter().enumerate() {
            self.entity_map.insert(entity, i);
        }
        
        // 更新特征维度
        self.dimension = self.entity_map.len();
    }
    
    /// 将实体转换为特征向量
    fn entities_to_features(&self, entities: &HashMap<String, Vec<String>>) -> Vec<f32> {
        let mut features = vec![0.0; self.dimension];
        
        // 计算实体频率
        for (entity_type, entity_list) in entities {
            for entity in entity_list {
                let key = format!("{}:{}", entity_type, entity);
                if let Some(&idx) = self.entity_map.get(&key) {
                    features[idx] += 1.0;
                }
            }
        }
        
        // 归一化
        let sum: f32 = features.iter().sum();
        if sum > 0.0 {
            for feature in features.iter_mut() {
                *feature /= sum;
            }
        }
        
        features
    }
}

impl FeatureExtractor for EntityExtractor {
    fn extract(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(Error::invalid_data("输入文本为空".to_string()));
        }
        
        // 提取实体
        let entities = self.extract_entities(text);
        
        // 转换为特征向量
        Ok(self.entities_to_features(&entities))
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    fn name(&self) -> &str {
        "entity"
    }
    
    fn from_config(config: &TextFeatureConfig) -> Result<Self> {
        EntityExtractor::from_config(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_email_extraction() {
        let extractor = EntityExtractor::new(vec!["email".to_string()]).unwrap();
        let text = "Contact us at test@example.com or support@company.com";
        let entities = extractor.extract_entities(text);
        
        assert_eq!(entities["email"], vec!["test@example.com", "support@company.com"]);
    }
    
    #[test]
    fn test_url_extraction() {
        let extractor = EntityExtractor::new(vec!["url".to_string()]).unwrap();
        let text = "Visit https://example.com or http://www.company.com";
        let entities = extractor.extract_entities(text);
        
        assert_eq!(entities["url"], vec!["https://example.com", "http://www.company.com"]);
    }
    
    #[test]
    fn test_feature_extraction() {
        let mut extractor = EntityExtractor::new(vec!["email".to_string(), "url".to_string()]).unwrap();
        let text = "Contact us at test@example.com or visit https://example.com";
        let entities = extractor.extract_entities(text);
        
        // 更新实体映射
        extractor.update_entity_map(&entities);
        
        // 提取特征
        let features = extractor.extract(text).unwrap();
        
        // 验证特征
        assert_eq!(features.len(), extractor.dimension());
        assert!(features.iter().sum::<f32>() - 1.0 < 1e-6);
    }
} 