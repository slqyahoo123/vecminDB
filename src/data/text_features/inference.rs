// src/data/text_features/inference.rs
//
// Transformer 推理模块

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use log::{info, warn};
use super::error::TransformerError;
use super::config::TransformerConfig;
use super::model::{TransformerModel, ProcessedText};
use super::encoder::Encoder;
use super::features::FeatureVector;

/// 推理配置
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// 批处理大小
    pub batch_size: usize,
    /// 是否使用缓存
    pub use_caching: bool,
    /// 缓存大小
    pub cache_size: usize,
    /// 是否启用并行处理
    pub enable_parallel: bool,
    /// 最大序列长度
    pub max_sequence_length: usize,
    /// 是否使用量化
    pub use_quantization: bool,
    /// 量化精度
    pub quantization_bits: u8,
    /// 是否使用模型融合
    pub use_model_fusion: bool,
    /// 推理超时时间（毫秒）
    pub timeout_ms: u64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            use_caching: true,
            cache_size: 1000,
            enable_parallel: false,
            max_sequence_length: 512,
            use_quantization: false,
            quantization_bits: 8,
            use_model_fusion: false,
            timeout_ms: 5000,
        }
    }
}

/// 推理引擎
#[derive(Debug)]
pub struct InferenceEngine {
    /// 推理配置
    config: InferenceConfig,
    /// 模型
    model: Arc<Mutex<TransformerModel>>,
    /// 缓存
    cache: HashMap<String, InferenceResult>,
    /// 推理统计
    stats: InferenceStats,
}

impl InferenceEngine {
    /// 创建新的推理引擎
    pub fn new(config: InferenceConfig, model: TransformerModel) -> Self {
        Self {
            config,
            model: Arc::new(Mutex::new(model)),
            cache: HashMap::new(),
            stats: InferenceStats::new(),
        }
    }
    
    /// 通过配置与可插拔编码器构造
    pub fn from_configs(
        infer_cfg: InferenceConfig,
        model_cfg: TransformerConfig,
        encoder: Option<Encoder>,
    ) -> Result<Self, TransformerError> {
        let mut model = TransformerModel::new(model_cfg)?;
        if let Some(enc) = encoder {
            model.set_encoder(enc);
        }
        Ok(Self::new(infer_cfg, model))
    }
    
    /// 推理单个文本
    pub fn infer(&mut self, text: &str) -> Result<InferenceResult, TransformerError> {
        let start_time = std::time::Instant::now();
        
        // 检查缓存
        if self.config.use_caching {
            if let Some(cached_result) = self.cache.get(text) {
                self.stats.cache_hits += 1;
                return Ok(cached_result.clone());
            }
        }
        
        // 检查超时
        if start_time.elapsed().as_millis() > self.config.timeout_ms as u128 {
            return Err(TransformerError::timeout_error("推理超时"));
        }
        
        // 预处理文本
        let processed = self.preprocess_text(text)?;
        
        // 执行推理
        let result = self.execute_inference(&processed)?;
        
        // 后处理结果
        let final_result = self.postprocess_result(result)?;
        
        // 缓存结果
        if self.config.use_caching {
            self.cache_result(text, &final_result);
        }
        
        // 更新统计
        self.stats.total_inferences += 1;
        self.stats.total_time += start_time.elapsed();
        
        Ok(final_result)
    }
    
    /// 批量推理
    pub fn batch_infer(&mut self, texts: &[String]) -> Result<Vec<InferenceResult>, TransformerError> {
        if texts.is_empty() {
            return Err(TransformerError::InputError("输入文本列表为空".to_string()));
        }
        
        let mut results = Vec::with_capacity(texts.len());
        
        if self.config.enable_parallel {
            // 并行处理
            results = self.parallel_inference(texts)?;
        } else {
            // 顺序处理
            for text in texts {
                let result = self.infer(text)?;
                results.push(result);
            }
        }
        
        Ok(results)
    }
    
    /// 并行推理
    fn parallel_inference(&self, texts: &[String]) -> Result<Vec<InferenceResult>, TransformerError> {
        use std::thread;
        use std::sync::mpsc;
        
        let (tx, rx) = mpsc::channel();
        let mut handles = Vec::new();
        
        // 创建线程池
        for text in texts {
            let tx = tx.clone();
            let text = text.clone();
            let model = self.model.clone();
            
            let handle = thread::spawn(move || {
                let mut model = model.lock().unwrap();
                let result = model.process_text(&text);
                tx.send((text, result)).unwrap();
            });
            
            handles.push(handle);
        }
        
        // 收集结果
        let mut results = Vec::new();
        for _ in 0..texts.len() {
            let (text, result) = rx.recv().unwrap();
            match result {
                Ok(processed) => {
                    let inference_result = InferenceResult {
                        input_text: text,
                        output_features: processed.features,
                        output_encoding: processed.encoded,
                        confidence: 1.0,
                        processing_time: std::time::Duration::from_millis(0),
                        metadata: processed.metadata,
                    };
                    results.push(inference_result);
                }
                Err(e) => return Err(e),
            }
        }
        
        // 等待所有线程完成
        for handle in handles {
            handle.join().unwrap();
        }
        
        Ok(results)
    }
    
    /// 预处理文本
    fn preprocess_text(&self, text: &str) -> Result<ProcessedText, TransformerError> {
        let mut model = self.model.lock().unwrap();
        
        // 检查序列长度
        if text.len() > self.config.max_sequence_length {
            let truncated = &text[..self.config.max_sequence_length];
            warn!("文本长度超过限制，已截断: {} -> {}", text.len(), truncated.len());
            model.process_text(truncated)
        } else {
            model.process_text(text)
        }
    }
    
    /// 执行推理
    fn execute_inference(&self, processed: &ProcessedText) -> Result<InferenceResult, TransformerError> {
        let start_time = std::time::Instant::now();
        
        // 量化处理（如果启用）
        let encoded = if self.config.use_quantization {
            self.quantize_features(&processed.encoded)?
        } else {
            processed.encoded.clone()
        };
        
        // 模型融合（如果启用）
        let final_encoding = if self.config.use_model_fusion {
            self.fuse_model_outputs(&encoded)?
        } else {
            encoded
        };
        
        let processing_time = start_time.elapsed();
        
        Ok(InferenceResult {
            input_text: processed.original_text.clone(),
            output_features: processed.features.clone(),
            output_encoding: final_encoding,
            confidence: self.calculate_confidence(&processed.features),
            processing_time,
            metadata: processed.metadata.clone(),
        })
    }
    
    /// 后处理结果
    fn postprocess_result(&self, result: InferenceResult) -> Result<InferenceResult, TransformerError> {
        // 应用后处理逻辑
        let mut processed_result = result;
        
        // 特征归一化
        processed_result.output_encoding = self.normalize_encoding(&processed_result.output_encoding)?;
        
        // 置信度调整
        processed_result.confidence = self.adjust_confidence(processed_result.confidence);
        
        Ok(processed_result)
    }
    
    /// 量化特征
    fn quantize_features(&self, features: &[f32]) -> Result<Vec<f32>, TransformerError> {
        let bits = self.config.quantization_bits;
        let max_value = (1 << bits) - 1;
        let scale = max_value as f32;
        
        let mut quantized = Vec::with_capacity(features.len());
        
        for &feature in features {
            let normalized = (feature + 1.0) / 2.0; // 归一化到[0,1]
            let quantized_value = (normalized * scale).round() / scale;
            let denormalized = quantized_value * 2.0 - 1.0; // 反归一化
            quantized.push(denormalized);
        }
        
        Ok(quantized)
    }
    
    /// 模型融合
    fn fuse_model_outputs(&self, encoding: &[f32]) -> Result<Vec<f32>, TransformerError> {
        // 简化的模型融合实现
        // 在实际应用中，这里应该实现真正的模型融合逻辑
        Ok(encoding.to_vec())
    }
    
    /// 计算置信度
    fn calculate_confidence(&self, features: &FeatureVector) -> f32 {
        if features.values.is_empty() {
            return 0.0;
        }
        
        // 基于特征强度的置信度计算
        let total_strength: f32 = features.values.iter().map(|&v| v.abs()).sum();
        let avg_strength = total_strength / features.values.len() as f32;
        
        // 归一化到[0,1]范围
        (avg_strength / 10.0).min(1.0).max(0.0)
    }
    
    /// 归一化编码
    fn normalize_encoding(&self, encoding: &[f32]) -> Result<Vec<f32>, TransformerError> {
        if encoding.is_empty() {
            return Ok(Vec::new());
        }
        
        let norm = encoding.iter().map(|&x| x * x).sum::<f32>().sqrt();
        
        if norm == 0.0 {
            return Ok(encoding.to_vec());
        }
        
        Ok(encoding.iter().map(|&x| x / norm).collect())
    }
    
    /// 调整置信度
    fn adjust_confidence(&self, confidence: f32) -> f32 {
        // 应用sigmoid函数进行平滑
        let sigmoid = 1.0 / (1.0 + (-confidence * 5.0).exp());
        sigmoid
    }
    
    /// 缓存结果
    fn cache_result(&mut self, text: &str, result: &InferenceResult) {
        if self.cache.len() >= self.config.cache_size {
            // 简单的LRU策略：移除第一个元素
            if let Some(first_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&first_key);
            }
        }
        
        self.cache.insert(text.to_string(), result.clone());
    }
    
    /// 获取推理统计
    pub fn get_stats(&self) -> &InferenceStats {
        &self.stats
    }
    
    /// 清空缓存
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        info!("推理缓存已清空");
    }
    
    /// 获取缓存大小
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

/// 推理结果
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// 输入文本
    pub input_text: String,
    /// 输出特征
    pub output_features: FeatureVector,
    /// 输出编码
    pub output_encoding: Vec<f32>,
    /// 置信度
    pub confidence: f32,
    /// 处理时间
    pub processing_time: std::time::Duration,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

impl InferenceResult {
    /// 获取主要特征
    pub fn get_main_features(&self) -> Vec<f32> {
        self.output_features.values.clone()
    }
    
    /// 获取编码向量
    pub fn get_encoding(&self) -> &[f32] {
        &self.output_encoding
    }
    
    /// 是否高置信度
    pub fn is_high_confidence(&self) -> bool {
        self.confidence > 0.8
    }
    
    /// 获取处理时间（毫秒）
    pub fn get_processing_time_ms(&self) -> u64 {
        self.processing_time.as_millis() as u64
    }
}

/// 推理统计
#[derive(Debug)]
pub struct InferenceStats {
    /// 总推理次数
    pub total_inferences: usize,
    /// 缓存命中次数
    pub cache_hits: usize,
    /// 总处理时间
    pub total_time: std::time::Duration,
    /// 平均处理时间
    pub avg_processing_time: std::time::Duration,
    /// 最大处理时间
    pub max_processing_time: std::time::Duration,
    /// 最小处理时间
    pub min_processing_time: std::time::Duration,
}

impl InferenceStats {
    pub fn new() -> Self {
        Self {
            total_inferences: 0,
            cache_hits: 0,
            total_time: std::time::Duration::from_millis(0),
            avg_processing_time: std::time::Duration::from_millis(0),
            max_processing_time: std::time::Duration::from_millis(0),
            min_processing_time: std::time::Duration::from_millis(0),
        }
    }
    
    /// 更新统计
    pub fn update(&mut self, processing_time: std::time::Duration) {
        self.total_inferences += 1;
        self.total_time += processing_time;
        
        if self.total_inferences == 1 {
            self.avg_processing_time = processing_time;
            self.max_processing_time = processing_time;
            self.min_processing_time = processing_time;
        } else {
            // 更新平均时间
            let total_nanos = self.total_time.as_nanos() as f64;
            let avg_nanos = total_nanos / self.total_inferences as f64;
            self.avg_processing_time = std::time::Duration::from_nanos(avg_nanos as u64);
            
            // 更新最大/最小时间
            if processing_time > self.max_processing_time {
                self.max_processing_time = processing_time;
            }
            if processing_time < self.min_processing_time {
                self.min_processing_time = processing_time;
            }
        }
    }
    
    /// 获取缓存命中率
    pub fn get_cache_hit_rate(&self) -> f32 {
        if self.total_inferences == 0 {
            return 0.0;
        }
        self.cache_hits as f32 / self.total_inferences as f32
    }
    
    /// 获取平均推理时间（毫秒）
    pub fn get_avg_processing_time_ms(&self) -> u64 {
        self.avg_processing_time.as_millis() as u64
    }
}

/// 推理预测器
#[derive(Debug)]
pub struct InferencePredictor {
    /// 推理引擎
    engine: InferenceEngine,
    /// 预测配置
    config: PredictionConfig,
}

/// 预测配置
#[derive(Debug, Clone)]
pub struct PredictionConfig {
    /// 预测阈值
    pub prediction_threshold: f32,
    /// 最大预测数量
    pub max_predictions: usize,
    /// 是否启用不确定性估计
    pub enable_uncertainty_estimation: bool,
    /// 不确定性样本数
    pub uncertainty_samples: usize,
}

impl Default for PredictionConfig {
    fn default() -> Self {
        Self {
            prediction_threshold: 0.5,
            max_predictions: 10,
            enable_uncertainty_estimation: false,
            uncertainty_samples: 10,
        }
    }
}

impl InferencePredictor {
    /// 创建新的预测器
    pub fn new(engine: InferenceEngine, config: PredictionConfig) -> Self {
        Self { engine, config }
    }
    
    /// 预测文本类别
    pub fn predict_class(&mut self, text: &str) -> Result<PredictionResult, TransformerError> {
        let inference_result = self.engine.infer(text)?;
        
        // 基于编码向量进行类别预测
        let predictions = self.classify_encoding(&inference_result.output_encoding)?;
        
        Ok(PredictionResult {
            input_text: text.to_string(),
            predictions,
            confidence: inference_result.confidence,
            processing_time: inference_result.processing_time,
            uncertainty: self.estimate_uncertainty(&inference_result.output_encoding)?,
        })
    }
    
    /// 预测文本相似度
    pub fn predict_similarity(&mut self, text1: &str, text2: &str) -> Result<SimilarityPrediction, TransformerError> {
        let result1 = self.engine.infer(text1)?;
        let result2 = self.engine.infer(text2)?;
        
        let similarity = self.calculate_similarity(&result1.output_encoding, &result2.output_encoding)?;
        
        Ok(SimilarityPrediction {
            text1: text1.to_string(),
            text2: text2.to_string(),
            similarity,
            confidence: (result1.confidence + result2.confidence) / 2.0,
        })
    }
    
    /// 分类编码
    fn classify_encoding(&self, encoding: &[f32]) -> Result<Vec<ClassPrediction>, TransformerError> {
        let mut predictions = Vec::new();
        
        // 简化的分类逻辑
        // 在实际应用中，这里应该使用训练好的分类器
        for (i, &value) in encoding.iter().enumerate() {
            if value.abs() > self.config.prediction_threshold {
                predictions.push(ClassPrediction {
                    class_id: i,
                    class_name: format!("class_{}", i),
                    confidence: value.abs(),
                    probability: (value + 1.0) / 2.0,
                });
            }
        }
        
        // 按置信度排序
        predictions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        
        // 限制预测数量
        predictions.truncate(self.config.max_predictions);
        
        Ok(predictions)
    }
    
    /// 计算相似度
    fn calculate_similarity(&self, encoding1: &[f32], encoding2: &[f32]) -> Result<f32, TransformerError> {
        if encoding1.len() != encoding2.len() {
            return Err(TransformerError::dimension_mismatch(
                format!("{}", encoding1.len()),
                format!("{}", encoding2.len())
            ));
        }
        
        // 余弦相似度
        let dot_product: f32 = encoding1.iter().zip(encoding2.iter())
            .map(|(a, b)| a * b)
            .sum();
        
        let norm1: f32 = encoding1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = encoding2.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm1 == 0.0 || norm2 == 0.0 {
            return Ok(0.0);
        }
        
        Ok(dot_product / (norm1 * norm2))
    }
    
    /// 估计不确定性
    fn estimate_uncertainty(&self, encoding: &[f32]) -> Result<f32, TransformerError> {
        if !self.config.enable_uncertainty_estimation {
            return Ok(0.0);
        }
        
        // 简化的不确定性估计
        // 基于编码向量的方差
        let mean = encoding.iter().sum::<f32>() / encoding.len() as f32;
        let variance = encoding.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / encoding.len() as f32;
        
        Ok(variance.sqrt())
    }
}

/// 预测结果
#[derive(Debug, Clone)]
pub struct PredictionResult {
    /// 输入文本
    pub input_text: String,
    /// 预测结果
    pub predictions: Vec<ClassPrediction>,
    /// 置信度
    pub confidence: f32,
    /// 处理时间
    pub processing_time: std::time::Duration,
    /// 不确定性
    pub uncertainty: f32,
}

/// 类别预测
#[derive(Debug, Clone)]
pub struct ClassPrediction {
    /// 类别ID
    pub class_id: usize,
    /// 类别名称
    pub class_name: String,
    /// 置信度
    pub confidence: f32,
    /// 概率
    pub probability: f32,
}

/// 相似度预测
#[derive(Debug, Clone)]
pub struct SimilarityPrediction {
    /// 文本1
    pub text1: String,
    /// 文本2
    pub text2: String,
    /// 相似度
    pub similarity: f32,
    /// 置信度
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config::TransformerConfig;
    use super::super::model::TransformerModel;

    #[test]
    fn test_inference_engine_creation() {
        let config = InferenceConfig::default();
        let model = TransformerModel::new(TransformerConfig::default()).unwrap();
        let engine = InferenceEngine::new(config, model);
        
        assert_eq!(engine.config.batch_size, 1);
        assert!(engine.config.use_caching);
    }

    #[test]
    fn test_inference_execution() {
        let config = InferenceConfig::default();
        let model = TransformerModel::new(TransformerConfig::default()).unwrap();
        let mut engine = InferenceEngine::new(config, model);
        
        let result = engine.infer("Hello world");
        assert!(result.is_ok());
        
        let inference_result = result.unwrap();
        assert_eq!(inference_result.input_text, "Hello world");
        assert!(!inference_result.output_encoding.is_empty());
    }

    #[test]
    fn test_batch_inference() {
        let config = InferenceConfig::default();
        let model = TransformerModel::new(TransformerConfig::default()).unwrap();
        let mut engine = InferenceEngine::new(config, model);
        
        let texts = vec!["Hello".to_string(), "World".to_string()];
        let results = engine.batch_infer(&texts);
        assert!(results.is_ok());
        
        let inference_results = results.unwrap();
        assert_eq!(inference_results.len(), 2);
    }

    #[test]
    fn test_prediction() {
        let config = InferenceConfig::default();
        let model = TransformerModel::new(TransformerConfig::default()).unwrap();
        let engine = InferenceEngine::new(config, model);
        let predictor_config = PredictionConfig::default();
        let mut predictor = InferencePredictor::new(engine, predictor_config);
        
        let result = predictor.predict_class("Hello world");
        assert!(result.is_ok());
        
        let prediction = result.unwrap();
        assert_eq!(prediction.input_text, "Hello world");
    }
} 