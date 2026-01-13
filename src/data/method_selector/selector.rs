// Method Selector Core
// 方法选择器核心逻辑

use crate::data::method_selector::analyzer::DataAnalyzer;
use crate::data::method_selector::evaluator::MethodEvaluator;
use crate::data::method_selector::rules::RuleBuilder;
use crate::data::method_selector::types::{DataCharacteristics, DomainRule, MethodEvaluation, MethodSelectorConfig};
use crate::data::text_features::config::TextFeatureMethod;
use serde_json::Value;
use std::collections::HashMap;
// serde traits are not used directly in this module
use tracing as log;

/// 方法选择器
pub struct MethodSelector {
    config: MethodSelectorConfig,
    evaluator: MethodEvaluator,
    analyzer: DataAnalyzer,
    domain_rules: Vec<DomainRule>,
    evaluation_cache: HashMap<String, MethodEvaluation>,
}

impl MethodSelector {
    /// 创建新的方法选择器
    pub fn new(config: MethodSelectorConfig) -> Self {
        let evaluator = MethodEvaluator::new(config.clone());
        let analyzer = DataAnalyzer::new();
        
        // 初始化领域规则
        let domain_rules = Self::init_domain_rules();
        
        Self {
            config,
            evaluator,
            analyzer,
            domain_rules,
            evaluation_cache: HashMap::new(),
        }
    }
    
    /// 初始化默认的领域规则
    fn init_domain_rules() -> Vec<DomainRule> {
        let mut rules = Vec::new();
        
        // 文本长度规则
        rules.push(RuleBuilder::text_length_rule("text", 0, 50, TextFeatureMethod::BagOfWords, 40));
        rules.push(RuleBuilder::text_length_rule("text", 50, 200, TextFeatureMethod::TfIdf, 50));
        rules.push(RuleBuilder::text_length_rule("text", 200, 500, TextFeatureMethod::Word2Vec, 60));
        rules.push(RuleBuilder::text_length_rule("text", 500, 1000, TextFeatureMethod::Bert, 70));
        rules.push(RuleBuilder::text_length_rule("text", 1000, std::usize::MAX, TextFeatureMethod::ContextAware, 80));
        
        // 词汇量规则
        rules.push(RuleBuilder::vocabulary_size_rule("text", 0, 100, TextFeatureMethod::BagOfWords, 30));
        rules.push(RuleBuilder::vocabulary_size_rule("text", 100, 1000, TextFeatureMethod::TfIdf, 40));
        rules.push(RuleBuilder::vocabulary_size_rule("text", 1000, 5000, TextFeatureMethod::FastText, 50));
        rules.push(RuleBuilder::vocabulary_size_rule("text", 5000, 10000, TextFeatureMethod::Bert, 60));
        rules.push(RuleBuilder::vocabulary_size_rule("text", 10000, std::usize::MAX, TextFeatureMethod::EnhancedRepresentation, 70));
        
        // 语言规则
        rules.push(RuleBuilder::language_rule("text", vec!["code", "function", "class", "import"], TextFeatureMethod::BagOfWords, 50));
        
        // 数值特征比例规则
        rules.push(RuleBuilder::numeric_ratio_rule(0.5, 0.8, TextFeatureMethod::Mixed, 40));
        
        // 上下文相关性规则
        rules.push(RuleBuilder::custom_rule(
            |data: &[Value]| -> bool {
                // 检测是否存在强上下文相关性
                // 如有序列化数据、时序数据或结构化文本
                // 这里是一个简化的实现
                if data.is_empty() {
                    return false;
                }
                
                let mut context_related_count = 0;
                let num_samples = std::cmp::min(data.len(), 50);
                
                for i in 0..num_samples {
                    if let Some(item) = data.get(i) {
                        if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                            if text.contains("next") || text.contains("previous") || 
                               text.contains("following") || text.contains("before") ||
                               text.contains("after") || text.contains("sequence") ||
                               text.contains("step") || text.contains("time") {
                                context_related_count += 1;
                            }
                        }
                    }
                }
                
                (context_related_count as f64) / (num_samples as f64) > 0.3
            },
            TextFeatureMethod::ContextAware,
            75
        ));
        
        // 复杂语义规则
        rules.push(RuleBuilder::custom_rule(
            |data: &[Value]| -> bool {
                // 检测是否存在复杂语义需求
                // 如多义词、隐喻、复杂情感等
                // 这里是一个简化的实现
                if data.is_empty() {
                    return false;
                }
                
                let mut complex_semantics_count = 0;
                let num_samples = std::cmp::min(data.len(), 50);
                
                for i in 0..num_samples {
                    if let Some(item) = data.get(i) {
                        if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                            let words = text.split_whitespace().count();
                            if words > 20 && (text.contains("meaning") || text.contains("understand") || 
                                             text.contains("context") || text.contains("concept") ||
                                             text.contains("similar") || text.contains("different") ||
                                             text.contains("relation") || text.contains("compare")) {
                                complex_semantics_count += 1;
                            }
                        }
                    }
                }
                
                (complex_semantics_count as f64) / (num_samples as f64) > 0.25
            },
            TextFeatureMethod::EnhancedRepresentation,
            65
        ));
        
        rules
    }
    
    /// 添加自定义领域规则
    pub fn add_domain_rule(&mut self, rule: DomainRule) {
        self.domain_rules.push(rule);
    }
    
    /// 设置配置
    pub fn set_config(&mut self, config: MethodSelectorConfig) {
        self.config = config;
    }
    
    /// 为给定数据选择最佳方法
    pub async fn select_best_method(&mut self, data: &[Value], field: &str) -> TextFeatureMethod {
        // 1. 如果自适应选择被禁用，返回默认方法
        if !self.config.adaptive_selection {
            return self.config.default_method.unwrap_or(TextFeatureMethod::TfIdf);
        }
        
        log::info!("开始为字段 '{}' 选择最佳文本特征提取方法", field);
        
        // 2. 如果启用了域规则，先检查它们
        if self.config.apply_domain_rules {
            if let Some(method) = self.apply_domain_rules(data) {
                log::info!("基于领域规则选择方法: {:?}", method);
                return method;
            }
        }
        
        // 3. 提取数据样本用于评估
        let sample_size = self.config.evaluation_sample_size.min(data.len());
        let sample_data = self.analyzer.extract_sample(data, sample_size);
        
        // 4. 获取数据特征
        let characteristics = self.analyzer.analyze_data(&sample_data);
        
        // 5. 基于数据特征初步筛选方法
        let candidate_methods = self.filter_methods_by_characteristics(&characteristics);
        
        // 6. 如果筛选后只有一个方法，直接返回
        if candidate_methods.len() == 1 {
            let method = candidate_methods[0];
            log::info!("基于数据特征筛选后只剩下一个方法: {:?}", method);
            return method;
        }
        
        // 7. 评估候选方法并选择最佳方法
        let mut best_evaluation: Option<MethodEvaluation> = None;
        
        for &method in &candidate_methods {
            // 检查缓存中是否已有评估结果
            let cache_key = format!("{}-{}-{}", method.as_ref(), field, sample_data.len());
            
            let evaluation = if self.config.cache_evaluations && self.evaluation_cache.contains_key(&cache_key) {
                log::info!("使用缓存的评估结果: {:?}", method);
                self.evaluation_cache.get(&cache_key).unwrap().clone()
            } else {
                // 评估方法
                match self.evaluator.evaluate_method(method, &sample_data, field).await {
                    Ok(eval_result) => {
                        // 缓存结果
                        if self.config.cache_evaluations {
                            self.evaluation_cache.insert(cache_key, eval_result.clone());
                        }
                        eval_result
                    },
                    Err(e) => {
                        log::warn!("评估方法 {:?} 失败: {}", method, e);
                        // 返回默认的低评分评估结果
                        MethodEvaluation::new(method)
                            .with_processing_time(u64::MAX)
                            .with_memory_usage(usize::MAX)
                    }
                }
            };
            
            // 更新最佳方法
            if let Some(ref current_best) = best_evaluation {
                if evaluation.overall_score > current_best.overall_score {
                    best_evaluation = Some(evaluation);
                }
            } else {
                best_evaluation = Some(evaluation);
            }
        }
        
        // 8. 返回最佳方法或默认方法
        let selected_method = best_evaluation
            .map(|e| e.method)
            .unwrap_or_else(|| {
                log::warn!("无法评估任何方法，使用默认方法");
                self.config.default_method.unwrap_or(TextFeatureMethod::TfIdf)
            });
            
        log::info!("选择了最佳特征提取方法: {:?}", selected_method);
        selected_method
    }
    
    /// 应用领域规则
    fn apply_domain_rules(&self, data: &[Value]) -> Option<TextFeatureMethod> {
        // 按优先级从高到低排序
        let mut sorted_rules = self.domain_rules.clone();
        sorted_rules.sort_by(|a, b| b.priority.cmp(&a.priority));
        
        // 应用规则
        for rule in &sorted_rules {
            if (rule.condition)(data) {
                return Some(rule.recommended_method);
            }
        }
        
        None
    }
    
    /// 基于数据特征筛选方法
    fn filter_methods_by_characteristics(&self, characteristics: &DataCharacteristics) -> Vec<TextFeatureMethod> {
        let mut candidates = Vec::new();
        
        // 基于数据类型进行初步筛选
        match characteristics.data_type.as_str() {
            "text" => {
                // 对于纯文本数据
                if characteristics.avg_text_length < 30.0 {
                    // 短文本
                    candidates.push(TextFeatureMethod::TfIdf);
                    candidates.push(TextFeatureMethod::Count);
                    candidates.push(TextFeatureMethod::NGram);
                } else if characteristics.avg_text_length < 200.0 {
                    // 中等长度文本
                    candidates.push(TextFeatureMethod::TfIdf);
                    candidates.push(TextFeatureMethod::Word2Vec);
                    candidates.push(TextFeatureMethod::FastText);
                    candidates.push(TextFeatureMethod::GloVe);
                } else {
                    // 长文本
                    candidates.push(TextFeatureMethod::Word2Vec);
                    candidates.push(TextFeatureMethod::BertEmbedding);
                    candidates.push(TextFeatureMethod::Universal);
                    candidates.push(TextFeatureMethod::Elmo);
                }
                
                // 基于词汇量大小
                if characteristics.vocabulary_size > 10000 {
                    // 大词汇量，优先考虑深度学习模型
                    if !candidates.contains(&TextFeatureMethod::BertEmbedding) {
                        candidates.push(TextFeatureMethod::BertEmbedding);
                    }
                }
            },
            "mixed" => {
                // 对于混合数据类型
                candidates.push(TextFeatureMethod::TfIdf);
                candidates.push(TextFeatureMethod::Word2Vec);
                candidates.push(TextFeatureMethod::BertEmbedding);
                candidates.push(TextFeatureMethod::FastText);
            },
            _ => {
                // 默认情况
                candidates.push(TextFeatureMethod::TfIdf);
                candidates.push(TextFeatureMethod::Word2Vec);
                candidates.push(TextFeatureMethod::Count);
            }
        }
        
        // 如果候选为空，添加所有方法
        if candidates.is_empty() {
            candidates.push(TextFeatureMethod::TfIdf);
            candidates.push(TextFeatureMethod::Word2Vec);
            candidates.push(TextFeatureMethod::BertEmbedding);
            candidates.push(TextFeatureMethod::FastText);
            candidates.push(TextFeatureMethod::GloVe);
            candidates.push(TextFeatureMethod::Universal);
            candidates.push(TextFeatureMethod::Elmo);
            candidates.push(TextFeatureMethod::Count);
            candidates.push(TextFeatureMethod::NGram);
        }
        
        candidates
    }
    
    /// 清除评估缓存
    pub fn clear_cache(&mut self) {
        self.evaluation_cache.clear();
    }

    /// 根据数据特征生成候选方法
    fn generate_candidate_methods(&self, characteristics: &DataCharacteristics) -> Vec<(TextFeatureMethod, f64)> {
        let mut candidates = Vec::new();
        
        // 1. 基础候选方法
        candidates.push((TextFeatureMethod::BagOfWords, 0.5));
        candidates.push((TextFeatureMethod::TfIdf, 0.6));
        candidates.push((TextFeatureMethod::WordFrequency, 0.4));
        
        // 2. 基于数据特征添加候选方法
        let avg_length = characteristics.avg_text_length;
        let vocab_size = characteristics.vocabulary_size;
        
        // 短文本
        if avg_length < 50.0 {
            candidates.push((TextFeatureMethod::CharacterLevel, 0.7));
        }
        // 中等长度文本
        else if avg_length < 200.0 {
            candidates.push((TextFeatureMethod::Word2Vec, 0.65));
            candidates.push((TextFeatureMethod::FastText, 0.7));
        }
        // 长文本
        else if avg_length < 500.0 {
            candidates.push((TextFeatureMethod::Bert, 0.75));
            candidates.push((TextFeatureMethod::Universal, 0.7));
        }
        // 超长文本
        else {
            candidates.push((TextFeatureMethod::ContextAware, 0.85));
            candidates.push((TextFeatureMethod::EnhancedRepresentation, 0.8));
        }
        
        // 小词汇量
        if vocab_size < 1000 {
            candidates.push((TextFeatureMethod::BagOfWords, 0.75));
        }
        // 中等词汇量
        else if vocab_size < 5000 {
            candidates.push((TextFeatureMethod::TfIdf, 0.8));
            candidates.push((TextFeatureMethod::FastText, 0.75));
        }
        // 大词汇量
        else if vocab_size < 10000 {
            candidates.push((TextFeatureMethod::Bert, 0.8));
            candidates.push((TextFeatureMethod::Universal, 0.75));
        }
        // 超大词汇量
        else {
            candidates.push((TextFeatureMethod::EnhancedRepresentation, 0.9));
        }
        
        // 3. 基于上下文相关性添加候选方法
        if characteristics.contains_sequential_patterns || characteristics.contains_time_related_text {
            candidates.push((TextFeatureMethod::ContextAware, 0.9));
        }
        
        // 4. 基于复杂语义需求添加候选方法
        if characteristics.contains_complex_semantics || characteristics.contains_ambiguous_meanings {
            candidates.push((TextFeatureMethod::EnhancedRepresentation, 0.85));
            candidates.push((TextFeatureMethod::Bert, 0.8));
        }
        
        // 5. 对混合数据类型进行调整
        if characteristics.has_mixed_data_types {
            candidates.push((TextFeatureMethod::Mixed, 0.9));
            
            // 如果混合数据中文本比例较大
            if characteristics.text_field_ratio > 0.5 {
                candidates.push((TextFeatureMethod::EnhancedRepresentation, 0.85));
            }
        }
        
        // 去重并按评分排序
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.dedup_by(|a, b| a.0 == b.0); // 只保留每种方法的最高评分
        
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_select_best_method() {
        let config = MethodSelectorConfig::default();
        let mut selector = MethodSelector::new(config);
        
        let data = vec![
            serde_json::json!({ "text": "Short sample text for testing" }),
            serde_json::json!({ "text": "Another example" }),
        ];
        
        let method = selector.select_best_method(&data, "text").await;
        
        // 检查是否返回了有效的方法
        assert!(matches!(
            method,
            TextFeatureMethod::TfIdf | 
            TextFeatureMethod::Word2Vec | 
            TextFeatureMethod::BertEmbedding | 
            TextFeatureMethod::FastText | 
            TextFeatureMethod::GloVe | 
            TextFeatureMethod::Universal | 
            TextFeatureMethod::Elmo | 
            TextFeatureMethod::Count | 
            TextFeatureMethod::NGram
        ));
    }
} 