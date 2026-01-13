// 特征融合模块
// 用于实现不同的特征融合策略

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use tracing::{info, warn, debug, error, instrument, span, Level};
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;

use crate::Result;
use crate::data::feature::extractor::{
    FeatureVector, FeatureBatch, ExtractorError, InputData, ExtractorContext
};
use crate::data::feature::types::FeatureType;

/// 特征融合策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// 简单连接
    Concatenate,
    /// 加权平均
    WeightedAverage,
    /// 注意力机制
    Attention,
    /// 门控融合
    Gated,
    /// 深度融合
    DeepFusion,
    /// 自定义融合
    Custom(String),
}

impl ToString for FusionStrategy {
    fn to_string(&self) -> String {
        match self {
            FusionStrategy::Concatenate => "concatenate".to_string(),
            FusionStrategy::WeightedAverage => "weighted_average".to_string(),
            FusionStrategy::Attention => "attention".to_string(),
            FusionStrategy::Gated => "gated".to_string(),
            FusionStrategy::DeepFusion => "deep_fusion".to_string(),
            FusionStrategy::Custom(name) => format!("custom:{}", name),
        }
    }
}

impl FusionStrategy {
    /// 从字符串解析融合策略
    pub fn from_str(s: &str) -> Result<Self, ExtractorError> {
        let strategy = match s.to_lowercase().as_str() {
            "concatenate" | "concat" => FusionStrategy::Concatenate,
            "weighted_average" | "weighted" | "average" => FusionStrategy::WeightedAverage,
            "attention" | "attn" => FusionStrategy::Attention,
            "gated" => FusionStrategy::Gated,
            "deep_fusion" => FusionStrategy::DeepFusion,
            s if s.starts_with("custom:") => {
                let name = s.trim_start_matches("custom:").to_string();
                FusionStrategy::Custom(name)
            },
            _ => {
                return Err(ExtractorError::Config(format!(
                    "不支持的融合策略: {}", s
                )));
            }
        };
        
        Ok(strategy)
    }
}

/// 特征融合器
/// 用于融合多个特征向量
#[derive(Debug)]
pub struct FeatureFusionEngine {
    /// 融合策略
    strategy: FusionStrategy,
    /// 权重配置
    weights: HashMap<String, f32>,
    /// 配置信息
    config: HashMap<String, String>,
    /// 运行时统计
    stats: Arc<RwLock<FusionStats>>,
}

/// 融合统计信息
#[derive(Debug, Clone, Default)]
pub struct FusionStats {
    /// 融合次数
    pub fusion_count: u64,
    /// 平均融合时间
    pub avg_fusion_time_ms: f64,
    /// 成功次数
    pub success_count: u64,
    /// 失败次数
    pub error_count: u64,
}

impl FeatureFusionEngine {
    /// 创建新的特征融合引擎
    pub fn new(strategy: FusionStrategy) -> Self {
        info!("创建特征融合引擎，策略: {:?}", strategy);
        
        Self {
            strategy,
            weights: HashMap::new(),
            config: HashMap::new(),
            stats: Arc::new(RwLock::new(FusionStats::default())),
        }
    }

    /// 设置权重
    pub fn set_weights(&mut self, weights: HashMap<String, f32>) {
        debug!("设置融合权重: {:?}", weights);
        self.weights = weights;
    }

    /// 设置配置
    pub fn set_config(&mut self, config: HashMap<String, String>) {
        info!("更新融合引擎配置");
        self.config = config;
    }

    /// 融合特征向量
    #[instrument(skip(self, feature_vectors))]
    pub async fn fuse_features(
        &self, 
        feature_vectors: Vec<FeatureVector>,
        context: Option<ExtractorContext>
    ) -> Result<FeatureVector, ExtractorError> {
        let _span = span!(Level::DEBUG, "feature_fusion").entered();
        
        debug!("开始融合 {} 个特征向量", feature_vectors.len());
        
        if feature_vectors.is_empty() {
            error!("输入特征向量为空");
            return Err(ExtractorError::InvalidInput("空特征向量列表".to_string()));
        }

        let start_time = std::time::Instant::now();
        
        let result = match &self.strategy {
            FusionStrategy::Concatenate => self.concatenate_fusion(feature_vectors).await,
            FusionStrategy::WeightedAverage => self.weighted_average_fusion(feature_vectors).await,
            FusionStrategy::Attention => self.attention_fusion(feature_vectors, context).await,
            FusionStrategy::Gated => self.gated_fusion(feature_vectors).await,
            FusionStrategy::DeepFusion => self.deep_fusion(feature_vectors).await,
            FusionStrategy::Custom(name) => self.custom_fusion(feature_vectors, name).await,
        };

        let fusion_time = start_time.elapsed();
        
        // 更新统计信息
        let mut stats = self.stats.write().await;
        stats.fusion_count += 1;
        stats.avg_fusion_time_ms = (stats.avg_fusion_time_ms * (stats.fusion_count - 1) as f64 + 
                                   fusion_time.as_millis() as f64) / stats.fusion_count as f64;
        
        match &result {
            Ok(_) => {
                stats.success_count += 1;
                info!("特征融合成功，耗时: {:?}", fusion_time);
            },
            Err(e) => {
                stats.error_count += 1;
                error!("特征融合失败: {:?}, 耗时: {:?}", e, fusion_time);
            }
        }

        result
    }

    /// 连接融合
    async fn concatenate_fusion(&self, feature_vectors: Vec<FeatureVector>) -> Result<FeatureVector, ExtractorError> {
        debug!("执行连接融合");
        
        let mut result = FeatureVector::new(FeatureType::Generic, vec![]);
        
        for (index, vector) in feature_vectors.into_iter().enumerate() {
            for feature in vector.features {
                let mut new_feature = feature;
                new_feature.name = format!("concat_{}_{}", index, new_feature.name);
                result.add_feature(new_feature);
            }
        }
        
        result.metadata.insert("fusion_strategy".to_string(), "concatenate".to_string());
        Ok(result)
    }

    /// 加权平均融合
    async fn weighted_average_fusion(&self, feature_vectors: Vec<FeatureVector>) -> Result<FeatureVector, ExtractorError> {
        debug!("执行加权平均融合");
        
        if feature_vectors.is_empty() {
            return Err(ExtractorError::InvalidInput("空特征向量".to_string()));
        }

        let mut result = FeatureVector::new(FeatureType::Generic, vec![]);
        let total_features = feature_vectors[0].features.len();
        
        // 验证所有向量具有相同的特征数量
        for vector in &feature_vectors {
            if vector.features.len() != total_features {
                return Err(ExtractorError::DimensionMismatch(
                    format!("特征数量不匹配: 期望 {}, 实际 {}", total_features, vector.features.len())
                ));
            }
        }

        for feature_idx in 0..total_features {
            let mut weighted_data = vec![0.0; feature_vectors[0].features[feature_idx].data.len()];
            let mut total_weight = 0.0;
            
            for (vector_idx, vector) in feature_vectors.iter().enumerate() {
                let weight = self.weights.get(&vector_idx.to_string()).copied().unwrap_or(1.0);
                total_weight += weight;
                
                let feature_data = &vector.features[feature_idx].data;
                for (data_idx, &value) in feature_data.iter().enumerate() {
                    weighted_data[data_idx] += value * weight;
                }
            }
            
            // 归一化
            if total_weight > 0.0 {
                for value in &mut weighted_data {
                    *value /= total_weight;
                }
            }
            
            let feature = crate::data::feature::types::Feature::new(
                format!("weighted_avg_{}", feature_idx),
                feature_vectors[0].features[feature_idx].feature_type.clone(),
                weighted_data,
            );
            
            result.add_feature(feature);
        }
        
        result.metadata.insert("fusion_strategy".to_string(), "weighted_average".to_string());
        Ok(result)
    }

    /// 注意力融合
    async fn attention_fusion(
        &self, 
        feature_vectors: Vec<FeatureVector>,
        context: Option<ExtractorContext>
    ) -> Result<FeatureVector, ExtractorError> {
        debug!("执行注意力融合");
        
        // 简化的注意力机制实现
        let mut result = FeatureVector::new(FeatureType::Generic, vec![]);
        
        if feature_vectors.is_empty() {
            return Err(ExtractorError::InvalidInput("空特征向量".to_string()));
        }

        // 计算注意力权重（简化版本）
        let mut attention_weights = Vec::with_capacity(feature_vectors.len());
        let total_features = feature_vectors.iter().map(|v| v.features.len()).sum::<usize>() as f32;
        
        for vector in &feature_vectors {
            let weight = vector.features.len() as f32 / total_features;
            attention_weights.push(weight);
        }
        
        // 应用注意力权重进行融合
        for (vector_idx, vector) in feature_vectors.into_iter().enumerate() {
            let attention_weight = attention_weights[vector_idx];
            
            for (feature_idx, feature) in vector.features.into_iter().enumerate() {
                let mut weighted_data = feature.data;
                for value in &mut weighted_data {
                    *value *= attention_weight;
                }
                
                let new_feature = crate::data::feature::types::Feature::new(
                    format!("attention_{}_{}", vector_idx, feature_idx),
                    feature.feature_type,
                    weighted_data,
                );
                
                result.add_feature(new_feature);
            }
        }
        
        if let Some(ctx) = context {
            result.metadata.insert("context_id".to_string(), ctx.get_id());
        }
        result.metadata.insert("fusion_strategy".to_string(), "attention".to_string());
        
        Ok(result)
    }

    /// 门控融合
    async fn gated_fusion(&self, feature_vectors: Vec<FeatureVector>) -> Result<FeatureVector, ExtractorError> {
        debug!("执行门控融合");
        
        // 简化的门控机制
        let gate_threshold = self.config.get("gate_threshold")
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.5);
            
        let mut result = FeatureVector::new(FeatureType::Generic, vec![]);
        
        for (vector_idx, vector) in feature_vectors.into_iter().enumerate() {
            for (feature_idx, feature) in vector.features.into_iter().enumerate() {
                // 计算门控值（基于特征均值）
                let mean_value = feature.data.iter().sum::<f32>() / feature.data.len() as f32;
                let gate_value = 1.0 / (1.0 + (-mean_value).exp()); // sigmoid
                
                if gate_value > gate_threshold {
                    let mut gated_data = feature.data;
                    for value in &mut gated_data {
                        *value *= gate_value;
                    }
                    
                    let new_feature = crate::data::feature::types::Feature::new(
                        format!("gated_{}_{}", vector_idx, feature_idx),
                        feature.feature_type,
                        gated_data,
                    );
                    
                    result.add_feature(new_feature);
                }
            }
        }
        
        result.metadata.insert("fusion_strategy".to_string(), "gated".to_string());
        result.metadata.insert("gate_threshold".to_string(), gate_threshold.to_string());
        
        Ok(result)
    }

    /// 深度融合
    async fn deep_fusion(&self, feature_vectors: Vec<FeatureVector>) -> Result<FeatureVector, ExtractorError> {
        debug!("执行深度融合");
        
        // 多层特征变换和融合
        let mut current_vectors = feature_vectors;
        let layers = self.config.get("fusion_layers")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2);
        
        for layer in 0..layers {
            debug!("处理深度融合层 {}", layer);
            
            // 在每一层应用不同的融合策略
            match layer % 3 {
                0 => current_vectors = vec![self.concatenate_fusion(current_vectors).await?],
                1 => current_vectors = vec![self.weighted_average_fusion(current_vectors).await?],
                _ => current_vectors = vec![self.attention_fusion(current_vectors, None).await?],
            }
        }
        
        if let Some(mut result) = current_vectors.into_iter().next() {
            result.metadata.insert("fusion_strategy".to_string(), "deep_fusion".to_string());
            result.metadata.insert("fusion_layers".to_string(), layers.to_string());
            Ok(result)
        } else {
            Err(ExtractorError::ProcessingError("深度融合失败".to_string()))
        }
    }

    /// 自定义融合
    async fn custom_fusion(&self, feature_vectors: Vec<FeatureVector>, fusion_name: &str) -> Result<FeatureVector, ExtractorError> {
        warn!("使用自定义融合策略: {}", fusion_name);
        
        // 默认回退到连接融合
        self.concatenate_fusion(feature_vectors).await
    }

    /// 获取统计信息
    pub async fn get_stats(&self) -> FusionStats {
        self.stats.read().await.clone()
    }

    /// 重置统计信息
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().await;
        *stats = FusionStats::default();
        info!("融合统计信息已重置");
    }
}

/// 批量特征融合处理器
#[derive(Debug)]
pub struct BatchFeatureFusion {
    engine: FeatureFusionEngine,
    batch_size: usize,
}

impl BatchFeatureFusion {
    /// 创建新的批量处理器
    pub fn new(strategy: FusionStrategy, batch_size: usize) -> Self {
        info!("创建批量特征融合处理器，批次大小: {}", batch_size);
        
        Self {
            engine: FeatureFusionEngine::new(strategy),
            batch_size,
        }
    }

    /// 批量处理特征向量
    #[instrument(skip(self, feature_batches))]
    pub async fn process_batches(
        &self,
        feature_batches: Vec<Vec<FeatureVector>>,
        input_data: Vec<InputData>,
    ) -> Result<Vec<FeatureVector>, ExtractorError> {
        let _span = span!(Level::INFO, "batch_feature_fusion").entered();
        
        info!("开始批量处理 {} 个特征批次", feature_batches.len());
        
        if feature_batches.len() != input_data.len() {
            return Err(ExtractorError::InvalidInput(
                format!("特征批次数量({})与输入数据数量({})不匹配", 
                        feature_batches.len(), input_data.len())
            ));
        }

        let mut results = Vec::with_capacity(feature_batches.len());
        
        for (batch_idx, (features, data)) in feature_batches.into_iter().zip(input_data.into_iter()).enumerate() {
            debug!("处理批次 {}/{}", batch_idx + 1, results.capacity());
            
            let context = ExtractorContext::new()
                .with_batch_index(batch_idx)
                .with_input_data(data.type_name());
                
            let fused_result = self.engine.fuse_features(features, Some(context)).await?;
            results.push(fused_result);
        }
        
        info!("批量处理完成，共处理 {} 个批次", results.len());
        Ok(results)
    }
}

/// 特征融合器特征
/// 用于融合多个特征向量
#[async_trait]
pub trait FeatureFuser: Send + Sync {
    /// 融合多个特征向量
    async fn fuse(&self, features: Vec<FeatureVector>) -> Result<FeatureVector, ExtractorError>;
    
    /// 批量融合特征
    async fn batch_fuse(&self, features_batch: Vec<Vec<FeatureVector>>) -> Result<FeatureBatch, ExtractorError> {
        let mut results = Vec::with_capacity(features_batch.len());
        
        for features in features_batch {
            let fused = self.fuse(features).await?;
            results.push(fused.values);
        }
        
        // 获取特征类型和维度
        let feature_type = if !results.is_empty() {
            self.output_feature_type()
        } else {
            FeatureType::Generic
        };
        
        let dimension = if !results.is_empty() && !results[0].is_empty() {
            results[0].len()
        } else {
            0
        };
        
        Ok(FeatureBatch {
            feature_type,
            values: results,
            extractor_type: None,
            batch_size: results.len(),
            dimension,
            metadata: HashMap::new(),
        })
    }
    
    /// 获取融合策略
    fn strategy(&self) -> FusionStrategy;
    
    /// 获取输出特征类型
    fn output_feature_type(&self) -> FeatureType {
        FeatureType::Generic
    }
    
    /// 获取输出维度
    fn output_dimension(&self) -> Option<usize>;
}

/// 连接融合器
/// 将多个特征向量简单连接在一起
pub struct ConcatenationFuser {
    /// 输出维度（可选）
    output_dim: Option<usize>,
}

impl ConcatenationFuser {
    /// 创建新的连接融合器
    pub fn new() -> Self {
        Self {
            output_dim: None,
        }
    }
    
    /// 设置输出维度
    pub fn with_output_dim(mut self, dim: usize) -> Self {
        self.output_dim = Some(dim);
        self
    }
}

#[async_trait]
impl FeatureFuser for ConcatenationFuser {
    async fn fuse(&self, features: Vec<FeatureVector>) -> Result<FeatureVector, ExtractorError> {
        if features.is_empty() {
            return Err(ExtractorError::ProcessingError("没有提供特征向量".to_string()));
        }
        
        // 计算总维度
        let total_dim: usize = features.iter()
            .map(|f| f.values.len())
            .sum();
        
        // 连接所有特征
        let mut combined = Vec::with_capacity(total_dim);
        for feature in &features {
            combined.extend_from_slice(&feature.values);
        }
        
        // 如果设置了输出维度，检查是否匹配
        if let Some(dim) = self.output_dim {
            if dim != combined.len() {
                warn!(
                    "连接融合器: 设置的输出维度 {} 与实际维度 {} 不匹配",
                    dim, combined.len()
                );
            }
        }
        
        // 获取第一个特征的类型（如果有多个不同类型，以第一个为准）
        let feature_type = features[0].feature_type;
        
        Ok(FeatureVector {
            feature_type,
            values: combined,
            extractor_type: None,
            metadata: HashMap::new(),
            features: Vec::new(),
        })
    }
    
    fn strategy(&self) -> FusionStrategy {
        FusionStrategy::Concatenate
    }
    
    fn output_dimension(&self) -> Option<usize> {
        self.output_dim
    }
}

/// 加权平均融合器
/// 对多个特征向量进行加权平均
pub struct WeightedAverageFuser {
    /// 权重
    weights: Option<Vec<f32>>,
    /// 输出维度（可选）
    output_dim: Option<usize>,
    /// 是否归一化权重
    normalize_weights: bool,
}

impl WeightedAverageFuser {
    /// 创建新的加权平均融合器
    pub fn new() -> Self {
        Self {
            weights: None,
            output_dim: None,
            normalize_weights: true,
        }
    }
    
    /// 设置权重
    pub fn with_weights(mut self, weights: Vec<f32>) -> Self {
        self.weights = Some(weights);
        self
    }
    
    /// 设置输出维度
    pub fn with_output_dim(mut self, dim: usize) -> Self {
        self.output_dim = Some(dim);
        self
    }
    
    /// 设置是否归一化权重
    pub fn with_normalize_weights(mut self, normalize: bool) -> Self {
        self.normalize_weights = normalize;
        self
    }
}

#[async_trait]
impl FeatureFuser for WeightedAverageFuser {
    async fn fuse(&self, features: Vec<FeatureVector>) -> Result<FeatureVector, ExtractorError> {
        if features.is_empty() {
            return Err(ExtractorError::ProcessingError("没有提供特征向量".to_string()));
        }
        
        // 获取权重，如果没有提供则使用平均权重
        let mut weights = match &self.weights {
            Some(w) => {
                if w.len() != features.len() {
                    return Err(ExtractorError::Config(format!(
                        "权重数量 {} 与特征向量数量 {} 不匹配",
                        w.len(), features.len()
                    )));
                }
                w.clone()
            },
            None => vec![1.0 / features.len() as f32; features.len()],
        };
        
        // 归一化权重
        if self.normalize_weights {
            let sum: f32 = weights.iter().sum();
            if sum > 0.0 {
                for w in &mut weights {
                    *w /= sum;
                }
            }
        }
        
        // 检查所有特征向量维度是否相同
        let dim = features[0].values.len();
        for (i, f) in features.iter().enumerate().skip(1) {
            if f.values.len() != dim {
                return Err(ExtractorError::ProcessingError(format!(
                    "特征向量维度不一致: 第一个维度 {}, 第 {} 个维度 {}",
                    dim, i + 1, f.values.len()
                )));
            }
        }
        
        // 计算加权平均
        let mut result = vec![0.0; dim];
        for (i, feature) in features.iter().enumerate() {
            let weight = weights[i];
            for (j, &value) in feature.values.iter().enumerate() {
                result[j] += value * weight;
            }
        }
        
        // 获取特征类型（取第一个）
        let feature_type = features[0].feature_type;
        
        Ok(FeatureVector {
            feature_type,
            values: result,
            extractor_type: None,
            metadata: HashMap::new(),
            features: Vec::new(),
        })
    }
    
    fn strategy(&self) -> FusionStrategy {
        FusionStrategy::WeightedAverage
    }
    
    fn output_dimension(&self) -> Option<usize> {
        if let Some(dim) = self.output_dim {
            return Some(dim);
        }
        
        // 如果没有设置输出维度，但有特征权重，返回第一个非零权重特征的维度
        if let Some(weights) = &self.weights {
            for (i, &w) in weights.iter().enumerate() {
                if w > 0.0 {
                    return Some(0); // 这里应该在实际实现中获取对应特征的维度
                }
            }
        }
        
        None
    }
}

/// 注意力融合器
/// 使用注意力机制融合多个特征向量
pub struct AttentionFuser {
    /// 注意力温度参数
    temperature: f32,
    /// 输出维度（可选）
    output_dim: Option<usize>,
    /// 是否使用多头注意力
    use_multi_head: bool,
    /// 头数
    num_heads: usize,
}

impl AttentionFuser {
    /// 创建新的注意力融合器
    pub fn new() -> Self {
        Self {
            temperature: 1.0,
            output_dim: None,
            use_multi_head: false,
            num_heads: 1,
        }
    }
    
    /// 设置温度参数
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }
    
    /// 设置输出维度
    pub fn with_output_dim(mut self, dim: usize) -> Self {
        self.output_dim = Some(dim);
        self
    }
    
    /// 设置是否使用多头注意力
    pub fn with_multi_head(mut self, use_multi_head: bool, num_heads: usize) -> Self {
        self.use_multi_head = use_multi_head;
        self.num_heads = num_heads;
        self
    }
    
    /// 计算注意力分数
    fn compute_attention_scores(&self, features: &[FeatureVector]) -> Vec<f32> {
        // 简化实现：计算每个特征向量的二范数，将其作为注意力分数
        let mut scores = Vec::with_capacity(features.len());
        
        for feature in features {
            let norm_squared: f32 = feature.values.iter()
                .map(|&x| x * x)
                .sum();
            let norm = norm_squared.sqrt();
            scores.push(norm);
        }
        
        // 应用温度缩放
        for score in &mut scores {
            *score /= self.temperature;
        }
        
        // Softmax归一化
        let max_score = scores.iter()
            .fold(std::f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let mut exp_scores = Vec::with_capacity(scores.len());
        let mut sum_exp = 0.0;
        
        for &score in &scores {
            let exp_score = ((score - max_score) / self.temperature).exp();
            exp_scores.push(exp_score);
            sum_exp += exp_score;
        }
        
        // 归一化
        for exp_score in &mut exp_scores {
            *exp_score /= sum_exp;
        }
        
        exp_scores
    }
}

#[async_trait]
impl FeatureFuser for AttentionFuser {
    async fn fuse(&self, features: Vec<FeatureVector>) -> Result<FeatureVector, ExtractorError> {
        if features.is_empty() {
            return Err(ExtractorError::ProcessingError("没有提供特征向量".to_string()));
        }
        
        // 检查所有特征向量维度是否相同
        let dim = features[0].values.len();
        for (i, f) in features.iter().enumerate().skip(1) {
            if f.values.len() != dim {
                return Err(ExtractorError::ProcessingError(format!(
                    "特征向量维度不一致: 第一个维度 {}, 第 {} 个维度 {}",
                    dim, i + 1, f.values.len()
                )));
            }
        }
        
        // 计算注意力分数
        let attention_scores = self.compute_attention_scores(&features);
        
        // 应用注意力融合
        let mut result = vec![0.0; dim];
        for (i, feature) in features.iter().enumerate() {
            let weight = attention_scores[i];
            for (j, &value) in feature.values.iter().enumerate() {
                result[j] += value * weight;
            }
        }
        
        // 获取特征类型（取第一个）
        let feature_type = features[0].feature_type;
        
        Ok(FeatureVector {
            feature_type,
            values: result,
            extractor_type: None,
            metadata: HashMap::new(),
            features: Vec::new(),
        })
    }
    
    fn strategy(&self) -> FusionStrategy {
        FusionStrategy::Attention
    }
    
    fn output_dimension(&self) -> Option<usize> {
        self.output_dim
    }
}

/// 创建融合器
/// 根据融合策略创建相应的融合器
pub fn create_fuser(strategy: FusionStrategy) -> Box<dyn FeatureFuser> {
    match strategy {
        FusionStrategy::Concatenate => Box::new(ConcatenationFuser::new()),
        FusionStrategy::WeightedAverage => Box::new(WeightedAverageFuser::new()),
        FusionStrategy::Attention => Box::new(AttentionFuser::new()),
        FusionStrategy::Gated => Box::new(WeightedAverageFuser::new()),
        FusionStrategy::DeepFusion => Box::new(WeightedAverageFuser::new()),
        FusionStrategy::Custom(name) => {
            // 实际应用中应该从注册表中查找自定义融合器
            // 这里简单地返回加权平均融合器作为默认
            warn!("未找到名为 {} 的自定义融合器，使用加权平均融合器作为替代", name);
            Box::new(WeightedAverageFuser::new())
        }
    }
}

/// 使用融合策略名称创建融合器
pub fn create_fuser_by_name(name: &str) -> Result<Box<dyn FeatureFuser>, ExtractorError> {
    let strategy = FusionStrategy::from_str(name)?;
    Ok(create_fuser(strategy))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_concatenation_fuser() {
        let fuser = ConcatenationFuser::new();
        
        let feature1 = FeatureVector {
            feature_type: FeatureType::Text,
            values: vec![1.0, 2.0, 3.0],
            extractor_type: None,
            metadata: HashMap::new(),
        };
        
        let feature2 = FeatureVector {
            feature_type: FeatureType::Text,
            values: vec![4.0, 5.0],
            extractor_type: None,
            metadata: HashMap::new(),
        };
        
        let result = fuser.fuse(vec![feature1, feature2]).await.unwrap();
        
        assert_eq!(result.values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(result.feature_type, FeatureType::Text);
    }
    
    #[tokio::test]
    async fn test_weighted_average_fuser() {
        let fuser = WeightedAverageFuser::new()
            .with_weights(vec![0.7, 0.3]);
        
        let feature1 = FeatureVector {
            feature_type: FeatureType::Text,
            values: vec![1.0, 2.0, 3.0],
            extractor_type: None,
            metadata: HashMap::new(),
        };
        
        let feature2 = FeatureVector {
            feature_type: FeatureType::Text,
            values: vec![4.0, 5.0, 6.0],
            extractor_type: None,
            metadata: HashMap::new(),
        };
        
        let result = fuser.fuse(vec![feature1, feature2]).await.unwrap();
        
        // 期望结果：[1.0*0.7 + 4.0*0.3, 2.0*0.7 + 5.0*0.3, 3.0*0.7 + 6.0*0.3]
        assert_eq!(result.values.len(), 3);
        assert!((result.values[0] - 1.9).abs() < 1e-5);
        assert!((result.values[1] - 2.9).abs() < 1e-5);
        assert!((result.values[2] - 3.9).abs() < 1e-5);
    }
    
    #[tokio::test]
    async fn test_attention_fuser() {
        let fuser = AttentionFuser::new()
            .with_temperature(0.5);
        
        let feature1 = FeatureVector {
            feature_type: FeatureType::Text,
            values: vec![1.0, 0.0, 0.0],
            extractor_type: None,
            metadata: HashMap::new(),
        };
        
        let feature2 = FeatureVector {
            feature_type: FeatureType::Text,
            values: vec![0.0, 1.0, 0.0],
            extractor_type: None,
            metadata: HashMap::new(),
        };
        
        let feature3 = FeatureVector {
            feature_type: FeatureType::Text,
            values: vec![0.0, 0.0, 1.0],
            extractor_type: None,
            metadata: HashMap::new(),
        };
        
        let result = fuser.fuse(vec![feature1, feature2, feature3]).await.unwrap();
        
        // 由于注意力机制基于向量范数，且这里所有向量的范数都是1，
        // 所以期望结果接近于[1/3, 1/3, 1/3]
        assert_eq!(result.values.len(), 3);
        assert!((result.values[0] - 1.0/3.0).abs() < 1e-1);
        assert!((result.values[1] - 1.0/3.0).abs() < 1e-1);
        assert!((result.values[2] - 1.0/3.0).abs() < 1e-1);
    }
} 