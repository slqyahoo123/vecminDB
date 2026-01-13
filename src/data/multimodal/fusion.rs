// Multimodal Fusion Module
// 多模态融合模块：负责融合不同模态的特征（如文本、图像、音频等）

use crate::error::{Error, Result};
use crate::compat::{Model, tensor::TensorData};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;

/// 融合策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// 早期融合（Early Fusion）：在特征提取之前融合不同模态
    Early,
    /// 晚期融合（Late Fusion）：在特征提取之后融合不同模态
    Late,
    /// 混合融合（Hybrid Fusion）：综合早期和晚期融合的特点
    Hybrid,
    /// 注意力融合（Attention-based Fusion）：使用注意力机制融合不同模态
    Attention,
    /// 权重融合（Weighted Fusion）：基于权重融合不同模态的特征
    Weighted(HashMap<String, f32>),
    /// 自适应融合（Adaptive Fusion）：根据数据特点自适应调整融合策略
    Adaptive,
    /// 自定义融合（Custom Fusion）：用户自定义的融合策略
    Custom(String),
}

/// 模态类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ModalityType {
    Text,
    Image,
    Video,
    Audio,
    TimeSeries,
    Tabular,
    Custom(String),
}

/// 多模态特征
#[derive(Debug, Clone)]
pub struct MultiModalFeature {
    /// 模态类型
    pub modality: ModalityType,
    /// 特征数据
    pub features: TensorData,
    /// 特征维度
    pub dimensions: Vec<usize>,
    /// 特征元数据
    pub metadata: HashMap<String, String>,
}

impl MultiModalFeature {
    pub fn new(modality: ModalityType, features: TensorData) -> Self {
        let dimensions = features.get_shape().to_vec();
        Self {
            modality,
            features,
            dimensions,
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// 融合结果
#[derive(Debug, Clone)]
pub struct FusionResult {
    /// 融合后的特征
    pub features: TensorData,
    /// 原始特征
    pub original_features: HashMap<ModalityType, MultiModalFeature>,
    /// 融合策略
    pub strategy: FusionStrategy,
    /// 融合权重
    pub weights: Option<HashMap<ModalityType, f32>>,
    /// 融合元数据
    pub metadata: HashMap<String, String>,
}

/// 多模态融合接口
pub trait MultiModalFusion: Send + Sync {
    /// 融合多个模态的特征
    fn fuse(&self, features: &HashMap<ModalityType, MultiModalFeature>) -> Result<FusionResult>;
    
    /// 获取融合策略
    fn get_strategy(&self) -> FusionStrategy;
}

/// 早期融合实现
pub struct EarlyFusion;

impl MultiModalFusion for EarlyFusion {
    fn fuse(&self, features: &HashMap<ModalityType, MultiModalFeature>) -> Result<FusionResult> {
        if features.is_empty() {
            return Err(Error::invalid_input("没有提供特征进行融合"));
        }
        
        // 收集所有特征向量并连接
        let mut all_features = Vec::new();
        let mut total_dim = 0;
        
        for feature in features.values() {
            let feature_data = feature.features.get_data();
            all_features.push(feature_data.clone());
            total_dim += feature_data.len();
        }
        
        // 创建融合特征
        let mut fused_features = Vec::with_capacity(total_dim);
        for feat in all_features {
            fused_features.extend_from_slice(&feat);
        }
        
        // 创建张量数据
        let tensor = TensorData::new(vec![1, total_dim], fused_features);
        
        // 创建结果
        let result = FusionResult {
            features: tensor,
            original_features: features.clone(),
            strategy: self.get_strategy(),
            weights: None,
            metadata: HashMap::new(),
        };
        
        Ok(result)
    }
    
    fn get_strategy(&self) -> FusionStrategy {
        FusionStrategy::Early
    }
}

/// 晚期融合实现
pub struct LateFusion;

impl MultiModalFusion for LateFusion {
    fn fuse(&self, features: &HashMap<ModalityType, MultiModalFeature>) -> Result<FusionResult> {
        if features.is_empty() {
            return Err(Error::invalid_input("没有提供特征进行融合"));
        }
        
        // 晚期融合保留每个模态的特征，返回一个映射
        let combined_features = features.values().next().unwrap().features.clone();
        
        // 创建结果
        let result = FusionResult {
            features: combined_features,
            original_features: features.clone(),
            strategy: self.get_strategy(),
            weights: None,
            metadata: HashMap::from([
                ("fusion_type".to_string(), "late".to_string()),
                ("modality_count".to_string(), features.len().to_string()),
            ]),
        };
        
        Ok(result)
    }
    
    fn get_strategy(&self) -> FusionStrategy {
        FusionStrategy::Late
    }
}

/// 加权融合实现
pub struct WeightedFusion {
    weights: HashMap<ModalityType, f32>,
}

impl WeightedFusion {
    pub fn new(weights: HashMap<ModalityType, f32>) -> Self {
        Self { weights }
    }
}

impl MultiModalFusion for WeightedFusion {
    fn fuse(&self, features: &HashMap<ModalityType, MultiModalFeature>) -> Result<FusionResult> {
        if features.is_empty() {
            return Err(Error::invalid_input("没有提供特征进行融合"));
        }
        
        // 验证所有模态都有权重
        for modality in features.keys() {
            if !self.weights.contains_key(modality) {
                return Err(Error::invalid_input(format!("模态 {:?} 没有指定权重", modality)));
            }
        }
        
        // 加权融合特征
        let mut weighted_features = Vec::new();
        let feature_dim = features.values().next().unwrap().features.get_data().len();
        weighted_features.resize(feature_dim, 0.0);
        
        for (modality, feature) in features.iter() {
            let weight = self.weights.get(modality).unwrap();
            let feature_data = feature.features.get_data();
            
            for i in 0..feature_dim {
                weighted_features[i] += feature_data[i] * weight;
            }
        }
        
        // 创建张量数据
        let tensor = TensorData::new(vec![1, feature_dim], weighted_features);
        
        // 创建结果
        let result = FusionResult {
            features: tensor,
            original_features: features.clone(),
            strategy: self.get_strategy(),
            weights: Some(self.weights.clone()),
            metadata: HashMap::from([
                ("fusion_type".to_string(), "weighted".to_string()),
                ("weight_count".to_string(), self.weights.len().to_string()),
            ]),
        };
        
        Ok(result)
    }
    
    fn get_strategy(&self) -> FusionStrategy {
        FusionStrategy::Weighted(self.weights.clone())
    }
}

/// 注意力融合实现
pub struct AttentionFusion {
    attention_model: Arc<Model>,
}

impl AttentionFusion {
    pub fn new(model: Arc<Model>) -> Self {
        Self { attention_model: model }
    }
}

impl MultiModalFusion for AttentionFusion {
    fn fuse(&self, features: &HashMap<ModalityType, MultiModalFeature>) -> Result<FusionResult> {
        if features.is_empty() {
            return Err(Error::invalid_input("没有提供特征进行融合"));
        }
        
        // 实际系统中，这里会使用注意力模型计算权重
        // 这里简化为生成随机权重
        let mut weights = HashMap::new();
        let weight_sum = features.len() as f32;
        
        for modality in features.keys() {
            weights.insert(modality.clone(), 1.0 / weight_sum);
        }
        
        // 使用加权融合实现融合
        let weighted_fusion = WeightedFusion::new(weights.clone());
        let mut result = weighted_fusion.fuse(features)?;
        
        // 更新策略
        result.strategy = self.get_strategy();
        result.metadata.insert("fusion_type".to_string(), "attention".to_string());
        
        Ok(result)
    }
    
    fn get_strategy(&self) -> FusionStrategy {
        FusionStrategy::Attention
    }
}

/// 融合工厂，创建不同的融合策略
pub struct FusionFactory;

impl FusionFactory {
    pub fn create_fusion(strategy: FusionStrategy) -> Box<dyn MultiModalFusion> {
        match strategy {
            FusionStrategy::Early => Box::new(EarlyFusion),
            FusionStrategy::Late => Box::new(LateFusion),
            FusionStrategy::Weighted(weights) => {
                let modality_weights = weights.iter()
                    .map(|(k, v)| {
                        let modality = match k.as_str() {
                            "text" => ModalityType::Text,
                            "image" => ModalityType::Image,
                            "video" => ModalityType::Video,
                            "audio" => ModalityType::Audio,
                            "timeseries" => ModalityType::TimeSeries,
                            "tabular" => ModalityType::Tabular,
                            _ => ModalityType::Custom(k.clone()),
                        };
                        (modality, *v)
                    })
                    .collect();
                Box::new(WeightedFusion::new(modality_weights))
            },
            // 其他策略暂时使用早期融合实现
            _ => Box::new(EarlyFusion),
        }
    }
}

/// 融合管道，处理多模态融合流程
pub struct FusionPipeline {
    strategy: FusionStrategy,
    fusion: Box<dyn MultiModalFusion>,
}

impl FusionPipeline {
    pub fn new(strategy: FusionStrategy) -> Self {
        let fusion = FusionFactory::create_fusion(strategy.clone());
        Self { strategy, fusion }
    }
    
    pub fn fuse(&self, features: &HashMap<ModalityType, MultiModalFeature>) -> Result<FusionResult> {
        self.fusion.fuse(features)
    }
    
    pub fn with_strategy(self, strategy: FusionStrategy) -> Self {
        let fusion = FusionFactory::create_fusion(strategy.clone());
        Self { strategy, fusion }
    }
} 