use crate::{Error, Result};
use std::collections::HashMap;
use super::interface::ModalityFusion;
use super::super::FusionStrategy as ParentFusionStrategy;
use super::interface::FeatureFusion;
use crate::core::types::CoreTensorData;

/// 特征融合模块
pub struct FusionModule {
    /// 融合策略
    strategy: ParentFusionStrategy,
    /// 输出维度
    output_dimension: usize,
    /// 权重
    weights: Option<HashMap<String, f32>>,
}

impl FusionModule {
    /// 创建新的融合模块
    pub fn new(strategy: ParentFusionStrategy, output_dimension: usize) -> Self {
        Self {
            strategy,
            output_dimension,
            weights: None,
        }
    }
    
    /// 设置权重
    pub fn with_weights(mut self, weights: HashMap<String, f32>) -> Self {
        self.weights = Some(weights);
        self
    }
    
    /// 获取融合策略
    pub fn strategy(&self) -> &ParentFusionStrategy {
        &self.strategy
    }
}

impl ModalityFusion for FusionModule {
    fn fuse_features(&self, features: HashMap<String, CoreTensorData>) -> Result<CoreTensorData> {
        match self.strategy {
            ParentFusionStrategy::Concatenation => {
                self.concat_fusion(features)
            },
            ParentFusionStrategy::Attention => {
                self.attention_fusion(features)
            },
            ParentFusionStrategy::Weighted => {
                self.weighted_fusion(features)
            },
            ParentFusionStrategy::Gated => {
                self.gated_fusion(features)
            },
            ParentFusionStrategy::TensorFusion => {
                self.tensor_fusion(features)
            },
            ParentFusionStrategy::Custom(_) => {
                self.custom_fusion(features)
            },
        }
    }
    
    fn get_fusion_strategy(&self) -> ParentFusionStrategy {
        self.strategy.clone()
    }
    
    fn get_output_dimension(&self) -> usize {
        self.output_dimension
    }
}

impl FusionModule {
    // 连接融合
    fn concat_fusion(&self, features: HashMap<String, CoreTensorData>) -> Result<CoreTensorData> {
        if features.is_empty() {
            return Err(Error::invalid_argument("Empty features for fusion".to_string()));
        }
        
        // 简单拼接所有特征
        let mut concatenated_data = Vec::new();
        let mut total_dim = 0;
        
        for tensor in features.values() {
            if tensor.shape.len() != 2 || tensor.shape[0] != 1 {
                return Err(Error::invalid_argument("Invalid tensor shape for fusion".to_string()));
            }
            
            concatenated_data.extend_from_slice(&tensor.data);
            total_dim += tensor.shape[1];
        }
        
        // 如果设置了输出维度，可能需要进行降维操作
        let result_data = if total_dim > self.output_dimension && self.output_dimension > 0 {
            // 简单截断（实际应用中可能使用PCA或其他降维技术）
            concatenated_data[0..self.output_dimension].to_vec()
        } else {
            concatenated_data
        };
        
        // 构建结果
        let actual_dim = result_data.len();
        let now = chrono::Utc::now();
        let tensor = CoreTensorData {
            id: uuid::Uuid::new_v4().to_string(),
            shape: vec![1, actual_dim],
            data: result_data,
            dtype: "float32".to_string(),
            device: "cpu".to_string(),
            requires_grad: false,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        };
        
        Ok(tensor)
    }
    
    // 加权融合
    fn weighted_fusion(&self, features: HashMap<String, CoreTensorData>) -> Result<CoreTensorData> {
        if features.is_empty() {
            return Err(Error::invalid_argument("Empty features for fusion".to_string()));
        }
        
        // 获取权重，如果没有设置则平均
        let weights = if let Some(w) = &self.weights {
            w.clone()
        } else {
            let weight = 1.0 / features.len() as f32;
            features.keys().map(|k| (k.clone(), weight)).collect()
        };
        
        // 首先确定输出维度
        let max_dim = features.values()
            .map(|t| t.shape[1])
            .max()
            .unwrap_or(0);
        
        let out_dim = if self.output_dimension > 0 {
            self.output_dimension.min(max_dim)
        } else {
            max_dim
        };
        
        // 初始化输出
        let mut result = vec![0.0; out_dim];
        
        // 加权融合
        for (name, tensor) in &features {
            let weight = weights.get(name).copied().unwrap_or(1.0 / features.len() as f32);
            let dim = tensor.shape[1].min(out_dim);
            
            for i in 0..dim {
                result[i] += tensor.data[i] * weight;
            }
        }
        
        // 构建结果
        let now = chrono::Utc::now();
        let tensor = CoreTensorData {
            id: uuid::Uuid::new_v4().to_string(),
            shape: vec![1, out_dim],
            data: result,
            dtype: "float32".to_string(),
            device: "cpu".to_string(),
            requires_grad: false,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        };
        
        Ok(tensor)
    }
    
    // 注意力融合
    fn attention_fusion(&self, features: HashMap<String, CoreTensorData>) -> Result<CoreTensorData> {
        if features.is_empty() {
            return Err(Error::invalid_argument("Empty features for attention fusion".to_string()));
        }
        
        // 计算注意力权重（基于特征向量的L2范数）
        let mut attention_weights = Vec::with_capacity(features.len());
        let mut feature_vecs: Vec<(String, &CoreTensorData, f32)> = Vec::new();
        
        for (name, tensor) in &features {
            if tensor.shape.len() != 2 || tensor.shape[0] != 1 {
                return Err(Error::invalid_argument(format!("Invalid tensor shape for attention fusion: {:?}", tensor.shape)));
            }
            
            // 计算L2范数作为注意力分数
            let norm = tensor.data.iter().map(|x| x * x).sum::<f32>().sqrt();
            feature_vecs.push((name.clone(), tensor, norm));
        }
        
        // 归一化注意力权重（softmax）
        let max_norm = feature_vecs.iter().map(|(_, _, norm)| *norm).fold(0.0f32, f32::max);
        let exp_sum: f32 = feature_vecs.iter()
            .map(|(_, _, norm)| (norm - max_norm).exp())
            .sum();
        
        for (_, _, norm) in &feature_vecs {
            let weight = ((norm - max_norm).exp()) / exp_sum;
            attention_weights.push(weight);
        }
        
        // 确定输出维度
        let max_dim = feature_vecs.iter()
            .map(|(_, tensor, _)| tensor.shape[1])
            .max()
            .unwrap_or(0);
        
        let out_dim = if self.output_dimension > 0 {
            self.output_dimension.min(max_dim)
        } else {
            max_dim
        };
        
        // 应用注意力权重进行加权融合
        let mut result = vec![0.0; out_dim];
        for (idx, (_, tensor, _)) in feature_vecs.iter().enumerate() {
            let weight = attention_weights[idx];
            let dim = tensor.shape[1].min(out_dim);
            
            for i in 0..dim {
                result[i] += tensor.data[i] * weight;
            }
        }
        
        // 构建结果
        let now = chrono::Utc::now();
        let mut tensor = CoreTensorData {
            id: uuid::Uuid::new_v4().to_string(),
            shape: vec![1, out_dim],
            data: result,
            dtype: "float32".to_string(),
            device: "cpu".to_string(),
            requires_grad: false,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        };
        tensor.metadata.insert("fusion_strategy".to_string(), "attention".to_string());
        
        Ok(tensor)
    }
    
    // 门控融合
    fn gated_fusion(&self, features: HashMap<String, CoreTensorData>) -> Result<CoreTensorData> {
        if features.is_empty() {
            return Err(Error::invalid_argument("Empty features for gated fusion".to_string()));
        }
        
        // 门控阈值（从配置或使用默认值）
        let gate_threshold = 0.5;
        
        // 确定输出维度
        let max_dim = features.values()
            .map(|t| if t.shape.len() == 2 && t.shape[0] == 1 { t.shape[1] } else { 0 })
            .max()
            .unwrap_or(0);
        
        if max_dim == 0 {
            return Err(Error::invalid_argument("Invalid tensor shapes for gated fusion".to_string()));
        }
        
        let out_dim = if self.output_dimension > 0 {
            self.output_dimension.min(max_dim)
        } else {
            max_dim
        };
        
        // 初始化输出
        let mut result = vec![0.0; out_dim];
        let mut total_gate_weight = 0.0;
        
        // 对每个特征应用门控机制
        for (name, tensor) in &features {
            if tensor.shape.len() != 2 || tensor.shape[0] != 1 {
                continue;
            }
            
            // 计算门控值（基于特征均值的sigmoid）
            let mean_value = if !tensor.data.is_empty() {
                tensor.data.iter().sum::<f32>() / tensor.data.len() as f32
            } else {
                0.0
            };
            let gate_value = 1.0 / (1.0 + (-mean_value).exp()); // sigmoid
            
            // 如果门控值超过阈值，则应用门控权重
            if gate_value > gate_threshold {
                let dim = tensor.shape[1].min(out_dim);
                for i in 0..dim {
                    result[i] += tensor.data[i] * gate_value;
                }
                total_gate_weight += gate_value;
            }
        }
        
        // 归一化（如果总权重大于0）
        if total_gate_weight > 1e-6 {
            for val in &mut result {
                *val /= total_gate_weight;
            }
        }
        
        // 构建结果
        let now = chrono::Utc::now();
        let mut tensor = CoreTensorData {
            id: uuid::Uuid::new_v4().to_string(),
            shape: vec![1, out_dim],
            data: result,
            dtype: "float32".to_string(),
            device: "cpu".to_string(),
            requires_grad: false,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        };
        tensor.metadata.insert("fusion_strategy".to_string(), "gated".to_string());
        tensor.metadata.insert("gate_threshold".to_string(), gate_threshold.to_string());
        
        Ok(tensor)
    }
    
    // 张量融合
    fn tensor_fusion(&self, features: HashMap<String, CoreTensorData>) -> Result<CoreTensorData> {
        if features.is_empty() {
            return Err(Error::invalid_argument("Empty features for tensor fusion".to_string()));
        }
        
        // 验证所有张量的形状
        let mut feature_vecs: Vec<Vec<f32>> = Vec::new();
        let mut max_dim = 0;
        
        for (name, tensor) in &features {
            if tensor.shape.len() != 2 || tensor.shape[0] != 1 {
                return Err(Error::invalid_argument(format!(
                    "Invalid tensor shape for tensor fusion: {} has shape {:?}",
                    name, tensor.shape
                )));
            }
            
            let dim = tensor.shape[1];
            if dim > max_dim {
                max_dim = dim;
            }
            feature_vecs.push(tensor.data.clone());
        }
        
        // 如果只有一个特征，直接返回
        if feature_vecs.len() == 1 {
            let tensor = features.values().next().unwrap();
            return Ok(tensor.clone());
        }
        
        // 计算外积（简化版本：逐元素乘积）
        // 对于多个特征向量，我们计算它们的逐元素乘积的平均值
        let out_dim = if self.output_dimension > 0 {
            self.output_dimension.min(max_dim)
        } else {
            max_dim
        };
        
        let mut result = vec![1.0; out_dim];
        
        // 计算逐元素乘积
        for feature_vec in &feature_vecs {
            let dim = feature_vec.len().min(out_dim);
            for i in 0..dim {
                result[i] *= feature_vec[i];
            }
        }
        
        // 归一化（取n次根，其中n是特征数量）
        let n = feature_vecs.len() as f32;
        for val in &mut result {
            *val = val.signum() * val.abs().powf(1.0 / n);
        }
        
        // 构建结果
        let now = chrono::Utc::now();
        let mut tensor = CoreTensorData {
            id: uuid::Uuid::new_v4().to_string(),
            shape: vec![1, out_dim],
            data: result,
            dtype: "float32".to_string(),
            device: "cpu".to_string(),
            requires_grad: false,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        };
        tensor.metadata.insert("fusion_strategy".to_string(), "tensor".to_string());
        
        Ok(tensor)
    }
    
    // 自定义融合
    fn custom_fusion(&self, features: HashMap<String, CoreTensorData>) -> Result<CoreTensorData> {
        // 自定义融合默认使用加权融合
        self.weighted_fusion(features)
    }
}

/// 特征融合策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionStrategy {
    /// 简单拼接
    Concatenation,
    /// 加权平均
    WeightedAverage,
    /// 加权求和
    WeightedSum,
    /// 最大池化
    MaxPooling,
    /// 平均池化
    AveragePooling,
}

/// 融合器配置
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// 融合策略
    pub strategy: FusionStrategy,
    /// 输出维度
    pub output_dimension: usize,
    /// 权重配置（对于加权方法）
    pub weights: Option<HashMap<String, f32>>,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            strategy: FusionStrategy::Concatenation,
            output_dimension: 0, // 拼接时会自动计算
            weights: None,
        }
    }
}

/// 特征融合提取器
#[derive(Debug)]
pub struct FeatureFusionExtractor {
    config: FusionConfig,
}

impl FeatureFusionExtractor {
    /// 创建新的融合器
    pub fn new(config: FusionConfig) -> Result<Self> {
        Ok(Self { config })
    }
    
    /// 使用默认策略创建融合器
    pub fn concatenation() -> Self {
        Self {
            config: FusionConfig::default(),
        }
    }
    
    /// 使用加权平均策略创建融合器
    pub fn weighted_average(weights: HashMap<String, f32>) -> Self {
        Self {
            config: FusionConfig {
                strategy: FusionStrategy::WeightedAverage,
                output_dimension: 0, // 会根据第一个特征自动设置
                weights: Some(weights),
            },
        }
    }
}

impl FeatureFusion for FeatureFusionExtractor {
    fn fuse_features(&self, features: &[Vec<f32>]) -> Result<Vec<f32>> {
        if features.is_empty() {
            return Err(Error::invalid_data("没有提供特征进行融合".to_string()));
        }
        
        match self.config.strategy {
            FusionStrategy::Concatenation => {
                // 简单拼接所有特征
                let mut result = Vec::new();
                for feature in features {
                    result.extend_from_slice(feature);
                }
                Ok(result)
            },
            FusionStrategy::WeightedAverage | FusionStrategy::WeightedSum => {
                // 检查所有特征维度是否相同
                let dim = features[0].len();
                for (i, feature) in features.iter().enumerate().skip(1) {
                    if feature.len() != dim {
                        return Err(Error::invalid_data(
                            format!("特征维度不一致: 第0个特征维度={}, 第{}个特征维度={}", 
                                    dim, i, feature.len())
                        ));
                    }
                }
                
                // 获取权重
                let weights = if let Some(weights) = &self.config.weights {
                    // 使用配置的权重
                    let mut w = Vec::with_capacity(features.len());
                    for i in 0..features.len() {
                        w.push(*weights.get(&i.to_string()).unwrap_or(&1.0));
                    }
                    w
                } else {
                    // 使用平均权重
                    vec![1.0 / features.len() as f32; features.len()]
                };
                
                // 执行加权操作
                let mut result = vec![0.0; dim];
                for (i, feature) in features.iter().enumerate() {
                    let weight = weights[i];
                    for j in 0..dim {
                        result[j] += feature[j] * weight;
                    }
                }
                
                // 对于加权平均，需要归一化
                if self.config.strategy == FusionStrategy::WeightedAverage {
                    let weight_sum: f32 = weights.iter().sum();
                    if weight_sum > 1e-6 {
                        for val in &mut result {
                            *val /= weight_sum;
                        }
                    }
                }
                
                Ok(result)
            },
            FusionStrategy::MaxPooling => {
                // 检查所有特征维度是否相同
                let dim = features[0].len();
                for (i, feature) in features.iter().enumerate().skip(1) {
                    if feature.len() != dim {
                        return Err(Error::invalid_data(
                            format!("特征维度不一致: 第0个特征维度={}, 第{}个特征维度={}", 
                                    dim, i, feature.len())
                        ));
                    }
                }
                
                // 执行最大池化
                let mut result = vec![std::f32::NEG_INFINITY; dim];
                for feature in features {
                    for j in 0..dim {
                        result[j] = result[j].max(feature[j]);
                    }
                }
                
                Ok(result)
            },
            FusionStrategy::AveragePooling => {
                // 检查所有特征维度是否相同
                let dim = features[0].len();
                for (i, feature) in features.iter().enumerate().skip(1) {
                    if feature.len() != dim {
                        return Err(Error::invalid_data(
                            format!("特征维度不一致: 第0个特征维度={}, 第{}个特征维度={}", 
                                    dim, i, feature.len())
                        ));
                    }
                }
                
                // 执行平均池化
                let mut result = vec![0.0; dim];
                for feature in features {
                    for j in 0..dim {
                        result[j] += feature[j];
                    }
                }
                
                // 进行平均化
                let n = features.len() as f32;
                for val in &mut result {
                    *val /= n;
                }
                
                Ok(result)
            },
        }
    }
    
    fn get_output_dim(&self) -> usize {
        match self.config.strategy {
            FusionStrategy::Concatenation => {
                // 拼接模式下，输出维度取决于输入特征
                // 这里返回配置的值，如果为0表示尚未确定
                self.config.output_dimension
            },
            _ => {
                // 其他模式下，输出维度等于输入特征维度
                // 同样，如果为0表示尚未确定
                self.config.output_dimension
            }
        }
    }
    
    fn get_fusion_type(&self) -> String {
        match self.config.strategy {
            FusionStrategy::Concatenation => "concatenation",
            FusionStrategy::WeightedAverage => "weighted_average",
            FusionStrategy::WeightedSum => "weighted_sum",
            FusionStrategy::MaxPooling => "max_pooling",
            FusionStrategy::AveragePooling => "average_pooling",
        }.to_string()
    }
} 