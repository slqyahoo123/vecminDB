// Method Evaluator for Method Selector
// 方法选择器的方法评估模块

use crate::data::method_selector::types::{MethodEvaluation, MethodSelectorConfig, PerformanceDataPoint};
use crate::data::text_features::config::TextFeatureMethod;
use serde_json::Value;
use std::collections::HashMap;
use std::time::Instant;
use std::sync::{Arc, Mutex};
use tracing as log;

/// 方法评估器
#[derive(Debug)]
pub struct MethodEvaluator {
    config: MethodSelectorConfig,
    performance_cache: Arc<Mutex<HashMap<TextFeatureMethod, Vec<PerformanceDataPoint>>>>,
}

impl MethodEvaluator {
    /// 创建新的方法评估器
    pub fn new(config: MethodSelectorConfig) -> Self {
        Self {
            config,
            performance_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// 评估指定方法的性能
    pub async fn evaluate_method(&self, method: TextFeatureMethod, data: &[Value], field: &str) -> crate::Result<MethodEvaluation> {
        log::info!("开始评估方法: {:?}", method);
        
        // 创建特征提取器
        let extractor = crate::data::text_features::extractors::create_text_extractor(
            method.clone().into()
        )?;
        
        // 测量性能
        let start_time = Instant::now();
        let start_memory = self.get_memory_usage();
        
        // 提取特征
        let features = match extractor.extract_from_json(data, field).await {
            Ok(features) => features,
            Err(_) => vec![], // 处理错误情况
        };
        
        // 记录时间和内存使用
        let processing_time = start_time.elapsed().as_millis() as u64;
        let memory_used = (self.get_memory_usage() - start_memory) as usize;
        
        // 计算特征维度和稀疏性
        let (dimension, sparsity) = self.calculate_feature_metrics(&features);
        
        // 估计质量分数 (简化版)
        let quality_score = self.estimate_quality(&features, method);
        
        // 计算整体分数
        let overall_score = self.calculate_overall_score(
            processing_time as f64,
            memory_used as f64,
            dimension,
            sparsity,
            quality_score,
            method,
        );
        
        // 如果配置为监控性能，则更新性能缓存
        if self.config.monitor_performance {
            self.update_performance_cache(method, processing_time as f64, memory_used as f64, data.len());
        }
        
        log::info!("方法 {:?} 评估完成，分数: {:.2}", method, overall_score);
        
        Ok(MethodEvaluation {
            method,
            processing_time_ms: processing_time,
            processing_time,
            memory_usage_bytes: memory_used,
            memory_usage: memory_used,
            feature_dimension: dimension,
            sparsity,
            quality_score,
            overall_score,
        })
    }
    
    /// 获取当前内存使用量 (MB)
    fn get_memory_usage(&self) -> f64 {
        // 在实际应用中，这里应该返回实际的内存使用量
        // 但为了简单起见，返回一个模拟值
        1.0
    }
    
    /// 计算特征维度和稀疏性
    fn calculate_feature_metrics(&self, features: &Vec<Vec<f64>>) -> (usize, f64) {
        if features.is_empty() {
            return (0, 0.0);
        }
        
        let dimension = features[0].len();
        
        // 计算稀疏性（非零元素的比例）
        let mut non_zero_count = 0;
        let mut total_elements = 0;
        
        for vec in features {
            for value in vec {
                total_elements += 1;
                if *value != 0.0 {
                    non_zero_count += 1;
                }
            }
        }
        
        let sparsity = if total_elements > 0 {
            1.0 - (non_zero_count as f64 / total_elements as f64)
        } else {
            0.0
        };
        
        (dimension, sparsity)
    }
    
    /// 估计方法质量
    fn estimate_quality(&self, features: &Vec<Vec<f64>>, method: TextFeatureMethod) -> f64 {
        // 简单的质量估计，基于特征分布、方法类型等
        // 实际应用中，这可能涉及更复杂的度量
        
        if features.is_empty() {
            return 0.0;
        }
        
        // 基于方法类型的基本评分
        let base_score = match method {
            TextFeatureMethod::TfIdf => 0.8,
            TextFeatureMethod::Word2Vec => 0.85,
            TextFeatureMethod::BertEmbedding => 0.9,
            TextFeatureMethod::FastText => 0.82,
            TextFeatureMethod::GloVe => 0.83,
            TextFeatureMethod::Universal => 0.92,
            TextFeatureMethod::Elmo => 0.93,
            TextFeatureMethod::Count => 0.75,
            TextFeatureMethod::NGram => 0.78,
            TextFeatureMethod::BagOfWords => 0.76,
            TextFeatureMethod::WordFrequency => 0.72,
            TextFeatureMethod::CharacterLevel => 0.74,
            TextFeatureMethod::Statistical => 0.7,
            TextFeatureMethod::Bert => 0.88,
            TextFeatureMethod::AutoSelect => 0.85,
            TextFeatureMethod::Custom(_) => 0.8,
        };
        
        // 检查特征向量的一致性和质量
        let dimensions = features[0].len();
        let mut variance_sum = 0.0;
        let mut feature_means: Vec<f64> = vec![0.0; dimensions];
        
        // 计算每个维度的平均值
        for feature_vec in features {
            for (i, &value) in feature_vec.iter().enumerate() {
                feature_means[i] += value / features.len() as f64;
            }
        }
        
        // 计算方差和
        for feature_vec in features {
            for (i, &value) in feature_vec.iter().enumerate() {
                variance_sum += (value - feature_means[i]).powi(2);
            }
        }
        
        // 归一化方差 (0-1)
        let normalized_variance = if variance_sum.is_normal() && variance_sum > 0.0 {
            (variance_sum / (features.len() * dimensions) as f64).min(1.0)
        } else {
            0.5 // 默认中等方差
        };
        
        // 结合基础分数与方差评估
        base_score * (0.5 + normalized_variance / 2.0)
    }
    
    /// 计算总体评分
    fn calculate_overall_score(
        &self,
        processing_time: f64,
        memory_used: f64,
        dimension: usize,
        sparsity: f64,
        quality_score: f64,
        method: TextFeatureMethod,
    ) -> f64 {
        // 归一化处理时间 (更低更好)
        let time_score = if processing_time > 0.0 {
            let normalized_time = 100.0 / processing_time.max(1.0);
            normalized_time.min(1.0)
        } else {
            0.5 // 默认中等分数
        };
        
        // 归一化内存使用 (更低更好)
        let memory_score = if memory_used > 0.0 {
            let normalized_memory = 10.0 / memory_used.max(1.0);
            normalized_memory.min(1.0)
        } else {
            0.5 // 默认中等分数
        };
        
        // 评估维度(基于用例，中等维度通常较好)
        let dimension_score = match dimension {
            0 => 0.0,
            1..=10 => 0.3,
            11..=100 => 0.6,
            101..=500 => 0.9,
            501..=2000 => 0.7,
            _ => 0.5,
        };
        
        // 稀疏性评分 (取决于方法，某些方法更适合稀疏表示)
        let sparsity_score = match method {
            TextFeatureMethod::TfIdf | TextFeatureMethod::Count | TextFeatureMethod::NGram => {
                if sparsity > 0.9 { 0.9 } else if sparsity > 0.7 { 0.8 } else { 0.6 }
            },
            _ => {
                if sparsity < 0.2 { 0.9 } else if sparsity < 0.5 { 0.7 } else { 0.5 }
            }
        };
        
        // 计算性能分数
        let performance_score = 
            time_score * self.config.performance_weight +
            dimension_score * self.config.dimension_weight +
            sparsity_score * self.config.sparsity_weight;
            
        // 应用方法权重
        let method_weight = 1.0; // 简化处理，不使用method_weights
        
        // 计算最终分数
        let final_score = 
            performance_score * self.config.performance_weight +
            quality_score * self.config.quality_weight +
            memory_score * self.config.memory_weight;
            
        // 应用方法权重
        final_score * method_weight
    }
    
    /// 更新性能缓存
    fn update_performance_cache(&self, method: TextFeatureMethod, time: f64, memory: f64, data_count: usize) {
        if let Ok(mut cache) = self.performance_cache.lock() {
            let entry = cache.entry(method).or_insert_with(Vec::new);
            
            let data_point = PerformanceDataPoint {
                timestamp: chrono::Utc::now(),
                method,
                processing_time_ms: time as u64,
                memory_usage_bytes: memory as usize,
                data_count,
            };
            
            entry.push(data_point);
            
            // 保持缓存大小适中
            if entry.len() > 100 {
                entry.remove(0);
            }
        }
    }
    
    /// 获取性能历史
    pub fn get_performance_history(&self, method: TextFeatureMethod) -> Vec<PerformanceDataPoint> {
        if let Ok(cache) = self.performance_cache.lock() {
            if let Some(history) = cache.get(&method) {
                return history.clone();
            }
        }
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_evaluate_method() {
        let config = MethodSelectorConfig::default();
        let evaluator = MethodEvaluator::new(config);
        
        let data = vec![
            serde_json::json!({ "text": "Sample text for testing" }),
            serde_json::json!({ "text": "Another example" }),
        ];
        
        let evaluation = evaluator.evaluate_method(TextFeatureMethod::TfIdf, &data, "text").await;
        
        assert_eq!(evaluation.method, TextFeatureMethod::TfIdf);
        assert!(evaluation.processing_time >= 0.0);
        assert!(evaluation.memory_usage >= 0.0);
        assert!(evaluation.overall_score >= 0.0 && evaluation.overall_score <= 1.0);
    }
} 