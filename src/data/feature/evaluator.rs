use std::collections::HashMap;
use std::time::Duration;

use crate::Result;
use crate::data::feature::types::FeatureType;
use crate::data::feature::interface::{FeatureExtractionResult, FeatureEvaluator};

/// 特征评估结果
#[derive(Debug, Clone)]
pub struct FeatureEvaluationResult {
    /// 特征名称
    pub feature_name: String,
    /// 特征类型
    pub feature_type: FeatureType,
    /// 特征维度
    pub dimension: usize,
    /// 质量评分 (0.0-1.0)
    pub quality_score: f32,
    /// 提取时间 (毫秒)
    pub extraction_time_ms: u64,
    /// 内存使用 (KB)
    pub memory_usage_kb: u64,
    /// 稀疏度 (0.0-1.0, 0表示密集，1表示完全稀疏)
    pub sparsity: f32,
    /// 信噪比 (dB)
    pub signal_to_noise_ratio: Option<f32>,
    /// 其他指标
    pub metrics: HashMap<String, f32>,
}

impl FeatureEvaluationResult {
    /// 创建新的特征评估结果
    pub fn new(feature_name: &str, feature_type: FeatureType, dimension: usize) -> Self {
        Self {
            feature_name: feature_name.to_string(),
            feature_type,
            dimension,
            quality_score: 0.0,
            extraction_time_ms: 0,
            memory_usage_kb: 0,
            sparsity: 0.0,
            signal_to_noise_ratio: None,
            metrics: HashMap::new(),
        }
    }
    
    /// 设置质量评分
    pub fn with_quality_score(mut self, score: f32) -> Self {
        self.quality_score = score.max(0.0).min(1.0);
        self
    }
    
    /// 设置提取时间
    pub fn with_extraction_time(mut self, time: Duration) -> Self {
        self.extraction_time_ms = time.as_millis() as u64;
        self
    }
    
    /// 设置内存使用
    pub fn with_memory_usage(mut self, memory_kb: u64) -> Self {
        self.memory_usage_kb = memory_kb;
        self
    }
    
    /// 设置稀疏度
    pub fn with_sparsity(mut self, sparsity: f32) -> Self {
        self.sparsity = sparsity.max(0.0).min(1.0);
        self
    }
    
    /// 设置信噪比
    pub fn with_snr(mut self, snr: f32) -> Self {
        self.signal_to_noise_ratio = Some(snr);
        self
    }
    
    /// 添加自定义指标
    pub fn with_metric(mut self, name: &str, value: f32) -> Self {
        self.metrics.insert(name.to_string(), value);
        self
    }
    
    /// 计算综合评分
    pub fn calculate_overall_score(&self) -> f32 {
        // 基础权重
        let quality_weight = 0.5;
        let time_weight = 0.2;
        let memory_weight = 0.2;
        let sparsity_weight = 0.1;
        
        // 归一化时间评分 (越快越好)
        let time_score = if self.extraction_time_ms <= 10 {
            1.0
        } else if self.extraction_time_ms >= 5000 {
            0.0
        } else {
            1.0 - (self.extraction_time_ms as f32 - 10.0) / 4990.0
        };
        
        // 归一化内存评分 (越小越好)
        let memory_score = if self.memory_usage_kb <= 1024 {
            1.0
        } else if self.memory_usage_kb >= 1024 * 100 {
            0.0
        } else {
            1.0 - (self.memory_usage_kb as f32 - 1024.0) / (1024.0 * 99.0)
        };
        
        // 稀疏度评分 (适度稀疏最好，通常0.1-0.3是较好的范围)
        let sparsity_score = if self.sparsity >= 0.1 && self.sparsity <= 0.3 {
            1.0
        } else if self.sparsity > 0.3 && self.sparsity <= 0.7 {
            0.8
        } else if self.sparsity > 0.7 {
            0.5
        } else {
            0.7 // 极度密集的特征
        };
        
        // 计算加权得分
        let weighted_score = 
            quality_weight * self.quality_score +
            time_weight * time_score +
            memory_weight * memory_score +
            sparsity_weight * sparsity_score;
            
        weighted_score
    }
}

/// 标准特征评估器实现
pub struct StandardFeatureEvaluator {
    /// 基准内存使用 (KB)
    baseline_memory_kb: u64,
    /// 基准时间 (毫秒)
    baseline_time_ms: u64,
    /// 最低质量阈值
    min_quality_threshold: f32,
}

impl StandardFeatureEvaluator {
    /// 创建新的标准特征评估器
    pub fn new() -> Self {
        Self {
            baseline_memory_kb: 1024, // 1MB
            baseline_time_ms: 100,    // 100ms
            min_quality_threshold: 0.5,
        }
    }
    
    /// 设置基准内存使用
    pub fn with_baseline_memory(mut self, memory_kb: u64) -> Self {
        self.baseline_memory_kb = memory_kb;
        self
    }
    
    /// 设置基准时间
    pub fn with_baseline_time(mut self, time_ms: u64) -> Self {
        self.baseline_time_ms = time_ms;
        self
    }
    
    /// 设置最低质量阈值
    pub fn with_min_quality_threshold(mut self, threshold: f32) -> Self {
        self.min_quality_threshold = threshold.max(0.0).min(1.0);
        self
    }
    
    /// 计算特征稀疏度
    fn calculate_sparsity(&self, feature_vector: &[f32]) -> f32 {
        let total = feature_vector.len();
        if total == 0 {
            return 0.0;
        }
        
        let zero_count = feature_vector.iter().filter(|&&v| v.abs() < 1e-6).count();
        zero_count as f32 / total as f32
    }
    
    /// 计算信噪比
    fn calculate_snr(&self, feature_vector: &[f32]) -> Option<f32> {
        let total = feature_vector.len();
        if total < 2 {
            return None;
        }
        
        // 计算信号能量（均方值）
        let signal_power: f32 = feature_vector.iter().map(|&v| v * v).sum::<f32>() / total as f32;
        if signal_power < 1e-10 {
            return None;
        }
        
        // 估计噪声能量（使用相邻样本差异的方差作为噪声估计）
        let mut noise_power = 0.0;
        for i in 1..total {
            let diff = feature_vector[i] - feature_vector[i-1];
            noise_power += diff * diff;
        }
        noise_power = noise_power / (total - 1) as f32 / 2.0; // 除以2是基于差分噪声估计的校正
        
        if noise_power < 1e-10 {
            return Some(100.0); // 几乎没有噪声，返回高SNR
        }
        
        // 计算SNR（分贝）
        let snr_db = 10.0 * (signal_power / noise_power).log10();
        Some(snr_db)
    }
    
    /// 计算特征质量评分
    fn calculate_quality_score(&self, result: &FeatureExtractionResult) -> f32 {
        // 基本假设：高维特征通常可以更好地表达复杂语义，但维度过高可能带来冗余
        let dimension = result.dimension;
        let dimension_score = if dimension < 16 {
            0.6 // 低维特征
        } else if dimension <= 256 {
            0.9 // 中等维度特征
        } else if dimension <= 1024 {
            1.0 // 较高维度特征，通常是最佳范围
        } else if dimension <= 4096 {
            0.8 // 高维特征
        } else {
            0.7 // 超高维特征，可能存在冗余
        };
        
        // 计算稀疏度
        let sparsity = self.calculate_sparsity(&result.feature_vector);
        let sparsity_score = if sparsity >= 0.1 && sparsity <= 0.3 {
            1.0 // 适度稀疏
        } else if sparsity > 0.3 && sparsity <= 0.7 {
            0.8 // 较稀疏
        } else if sparsity > 0.7 {
            0.6 // 极稀疏
        } else {
            0.7 // 极度密集
        };
        
        // 基于元数据的评分
        let metadata_score = if result.metadata.contains_key("quality_score") {
            result.metadata.get("quality_score")
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(0.8)
        } else {
            0.8 // 默认分数
        };
        
        // 综合评分
        let combined_score = 0.4 * dimension_score + 0.3 * sparsity_score + 0.3 * metadata_score;
        combined_score.max(0.0).min(1.0)
    }
}

impl Default for StandardFeatureEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureEvaluator for StandardFeatureEvaluator {
    fn evaluate(&self, result: &FeatureExtractionResult) -> Result<FeatureEvaluationResult> {
        let feature_name = result.metadata.get("name")
            .map(|s| s.to_string())
            .unwrap_or_else(|| "unknown".to_string());
            
        let mut evaluation = FeatureEvaluationResult::new(
            &feature_name,
            result.feature_type.clone(),
            result.dimension
        );
        
        // 设置提取时间
        if let Some(time) = result.extraction_time {
            evaluation = evaluation.with_extraction_time(time);
        }
        
        // 计算稀疏度
        let sparsity = self.calculate_sparsity(&result.feature_vector);
        evaluation = evaluation.with_sparsity(sparsity);
        
        // 计算信噪比
        if let Some(snr) = self.calculate_snr(&result.feature_vector) {
            evaluation = evaluation.with_snr(snr);
        }
        
        // 计算质量评分
        let quality_score = self.calculate_quality_score(result);
        evaluation = evaluation.with_quality_score(quality_score);
        
        // 估计内存使用
        let memory_usage_kb = (result.dimension * std::mem::size_of::<f32>()) / 1024;
        evaluation = evaluation.with_memory_usage(memory_usage_kb as u64);
        
        // 添加自定义指标
        for (key, value) in &result.metadata {
            if key.starts_with("metric_") && key.len() > 7 {
                if let Ok(metric_value) = value.parse::<f32>() {
                    let metric_name = &key[7..];
                    evaluation = evaluation.with_metric(metric_name, metric_value);
                }
            }
        }
        
        Ok(evaluation)
    }
    
    fn evaluate_batch(&self, results: &[FeatureExtractionResult]) -> Result<Vec<FeatureEvaluationResult>> {
        let mut evaluations = Vec::with_capacity(results.len());
        
        for result in results {
            let evaluation = self.evaluate(result)?;
            evaluations.push(evaluation);
        }
        
        Ok(evaluations)
    }
    
    fn calculate_quality_score(&self, result: &FeatureExtractionResult) -> f32 {
        self.calculate_quality_score(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use crate::data::multimodal::extractors::interface::ExtractorType;
    
    #[test]
    fn test_feature_evaluation_result() {
        let eval_result = FeatureEvaluationResult::new("test_feature", FeatureType::Text, 128)
            .with_quality_score(0.85)
            .with_extraction_time(Duration::from_millis(50))
            .with_memory_usage(2048)
            .with_sparsity(0.2)
            .with_snr(15.5)
            .with_metric("accuracy", 0.92);
            
        assert_eq!(eval_result.feature_name, "test_feature");
        assert_eq!(eval_result.dimension, 128);
        assert_eq!(eval_result.quality_score, 0.85);
        assert_eq!(eval_result.extraction_time_ms, 50);
        assert_eq!(eval_result.memory_usage_kb, 2048);
        assert_eq!(eval_result.sparsity, 0.2);
        assert_eq!(eval_result.signal_to_noise_ratio, Some(15.5));
        assert_eq!(eval_result.metrics.get("accuracy"), Some(&0.92));
        
        let score = eval_result.calculate_overall_score();
        assert!(score > 0.0 && score <= 1.0);
    }
    
    #[test]
    fn test_standard_feature_evaluator() {
        let evaluator = StandardFeatureEvaluator::new();
        
        // 创建测试特征提取结果
        let mut metadata = HashMap::new();
        metadata.insert("name".to_string(), "test_feature".to_string());
        
        let result = FeatureExtractionResult {
            feature_vector: vec![0.1, 0.5, 0.0, 0.8, 0.0, 0.3, 0.0, 0.9],
            dimension: 8,
            feature_type: FeatureType::Text,
            extraction_time: Some(Duration::from_millis(30)),
            metadata,
            extractor_type: ExtractorType::Text,
        };
        
        let evaluation = evaluator.evaluate(&result).unwrap();
        
        assert_eq!(evaluation.feature_name, "test_feature");
        assert_eq!(evaluation.dimension, 8);
        assert!(evaluation.quality_score > 0.0);
        assert_eq!(evaluation.extraction_time_ms, 30);
        assert!(evaluation.memory_usage_kb > 0);
        assert!(evaluation.sparsity > 0.0);
    }
    
    #[test]
    fn test_sparsity_calculation() {
        let evaluator = StandardFeatureEvaluator::new();
        
        // 测试稠密向量
        let dense = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let dense_sparsity = evaluator.calculate_sparsity(&dense);
        assert_eq!(dense_sparsity, 0.0);
        
        // 测试稀疏向量
        let sparse = vec![0.0, 0.5, 0.0, 0.0, 0.8, 0.0];
        let sparse_sparsity = evaluator.calculate_sparsity(&sparse);
        assert!(sparse_sparsity > 0.6 && sparse_sparsity < 0.7);
        
        // 测试完全稀疏向量
        let fully_sparse = vec![0.0, 0.0, 0.0, 0.0];
        let fully_sparse_sparsity = evaluator.calculate_sparsity(&fully_sparse);
        assert_eq!(fully_sparse_sparsity, 1.0);
    }
} 