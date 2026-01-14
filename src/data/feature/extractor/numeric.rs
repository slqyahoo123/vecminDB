// 数值特征提取器实现
// 提供标准化、归一化、对数变换等数值特征提取功能

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::{Arc, RwLock};
use async_trait::async_trait;
use serde_json;

use crate::{Result, Error};
use crate::data::feature::extractor::{
    FeatureExtractor, ExtractorError, InputData, ExtractorContext,
    FeatureVector, FeatureBatch, ExtractorConfig
};
use crate::data::feature::types::{
    ExtractorType, FeatureType, NumericExtractorType
};

/// 标准化特征提取器（Z-score标准化）
/// 
/// 将特征值标准化为均值为0、标准差为1的分布
/// 公式: (x - mean) / std
pub struct StandardizeExtractor {
    config: ExtractorConfig,
    output_dim: usize,
    // 训练时的统计信息（均值、标准差）
    stats: Arc<RwLock<Option<StandardizeStats>>>,
}

#[derive(Debug, Clone)]
struct StandardizeStats {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl StandardizeExtractor {
    /// 创建新的标准化提取器
    pub fn new(config: ExtractorConfig) -> Result<Self> {
        let output_dim = config.output_dimension.ok_or_else(|| {
            Error::from(ExtractorError::Config("标准化提取器必须指定输出维度".to_string()))
        })?;
        
        Ok(Self {
            config,
            output_dim,
            stats: Arc::new(RwLock::new(None)),
        })
    }
    
    /// 从训练数据计算统计信息
    fn fit(&self, training_data: &[Vec<f32>]) -> Result<()> {
        if training_data.is_empty() {
            return Err(Error::from(ExtractorError::Internal("训练数据不能为空".to_string())));
        }
        
        let dim = training_data[0].len();
        if dim != self.output_dim {
            return Err(ExtractorError::Config(format!(
                "训练数据维度 {} 与输出维度 {} 不匹配",
                dim, self.output_dim
            )).into());
        }
        
        // 计算均值和标准差
        let mut mean = vec![0.0f32; dim];
        let mut variance = vec![0.0f32; dim];
        
        // 计算均值
        for sample in training_data {
            for (i, &value) in sample.iter().enumerate() {
                mean[i] += value;
            }
        }
        let n = training_data.len() as f32;
        for m in &mut mean {
            *m /= n;
        }
        
        // 计算方差
        for sample in training_data {
            for (i, &value) in sample.iter().enumerate() {
                let diff = value - mean[i];
                variance[i] += diff * diff;
            }
        }
        
        // 计算标准差
        let mut std = vec![0.0f32; dim];
        for (i, v) in variance.iter().enumerate() {
            let var = v / n;
            std[i] = if var > 1e-8 {
                var.sqrt()
            } else {
                1.0 // 避免除零，使用1.0作为默认值
            };
        }
        
        let stats = StandardizeStats { mean, std };
        *self.stats.write().unwrap() = Some(stats);
        
        Ok(())
    }
    
    /// 应用标准化变换
    fn transform(&self, data: &[f32]) -> Result<Vec<f32>> {
        let stats = self.stats.read().unwrap();
        let stats = stats.as_ref().ok_or_else(|| {
            Error::from(ExtractorError::Internal("标准化提取器尚未拟合，请先调用fit方法".to_string()))
        })?;
        
        if data.len() != stats.mean.len() {
            return Err(Error::from(ExtractorError::Internal(format!(
                "输入数据维度 {} 与统计信息维度 {} 不匹配",
                data.len(), stats.mean.len()
            ))));
        }
        
        let mut result = Vec::with_capacity(data.len());
        for (i, &value) in data.iter().enumerate() {
            let normalized = (value - stats.mean[i]) / stats.std[i];
            result.push(normalized);
        }
        
        Ok(result)
    }
    
    /// 从输入数据提取数值向量
    fn extract_numeric_vector(&self, input: &InputData) -> Result<Vec<f32>> {
        match input {
            InputData::Tensor(data, shape) => {
                // 验证维度
                let total_elements: usize = shape.iter().product();
                if data.len() != total_elements {
                    return Err(Error::from(ExtractorError::Internal(format!(
                        "张量数据长度 {} 与形状 {:?} 不匹配",
                        data.len(), shape
                    ))));
                }
                
                // 如果是一维，直接返回
                if shape.len() == 1 && shape[0] == self.output_dim {
                    Ok(data.clone())
                } else {
                    // 展平为一维向量
                    Ok(data.clone())
                }
            },
            InputData::Raw(_) | InputData::Binary(_) => {
                // 尝试将字节数据转换为数值向量
                Err(Error::from(ExtractorError::Internal(
                    "无法从原始/二进制数据提取数值向量，请使用Tensor类型".to_string()
                )))
            },
            _ => Err(Error::from(ExtractorError::Internal(format!(
                "标准化提取器不支持输入类型: {}",
                input.type_name()
            )))),
        }
    }
}

impl Debug for StandardizeExtractor {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("StandardizeExtractor")
            .field("output_dim", &self.output_dim)
            .field("fitted", &self.stats.read().unwrap().is_some())
            .finish()
    }
}

#[async_trait]
impl FeatureExtractor for StandardizeExtractor {
    fn extractor_type(&self) -> ExtractorType {
        ExtractorType::Numeric(NumericExtractorType::Standardize)
    }
    
    fn config(&self) -> &ExtractorConfig {
        &self.config
    }
    
    fn is_compatible(&self, input: &InputData) -> bool {
        matches!(input, InputData::Tensor(_, _))
    }
    
    async fn extract(&self, input: InputData, context: Option<ExtractorContext>) -> std::result::Result<FeatureVector, ExtractorError> {
        // 检查是否有训练数据需要拟合
        if let Some(ctx) = &context {
            if let Some(training_data_str) = ctx.get_param("training_data") {
                if let Ok(training_data) = serde_json::from_str::<Vec<Vec<f32>>>(training_data_str) {
                    self.fit(&training_data)?;
                }
            }
        }
        
        let numeric_data = self.extract_numeric_vector(&input)?;
        let transformed = self.transform(&numeric_data)?;
        
        Ok(FeatureVector::new(FeatureType::Numeric, transformed)
            .with_extractor_type(self.extractor_type()))
    }
    
    async fn batch_extract(&self, inputs: Vec<InputData>, context: Option<ExtractorContext>) -> std::result::Result<FeatureBatch, ExtractorError> {
        // 检查是否有训练数据需要拟合
        if let Some(ctx) = &context {
            if let Some(training_data_str) = ctx.get_param("training_data") {
                if let Ok(training_data) = serde_json::from_str::<Vec<Vec<f32>>>(&training_data_str) {
                    self.fit(&training_data)?;
                }
            }
        }
        
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            let numeric_data = self.extract_numeric_vector(&input)?;
            let transformed = self.transform(&numeric_data)?;
            results.push(transformed);
        }
        
        Ok(FeatureBatch::new(results, FeatureType::Numeric)
            .with_extractor_type(self.extractor_type()))
    }
    
    fn output_feature_type(&self) -> FeatureType {
        FeatureType::Numeric
    }
    
    fn output_dimension(&self) -> Option<usize> {
        Some(self.output_dim)
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// 归一化特征提取器（MinMax归一化）
/// 
/// 将特征值缩放到[0, 1]区间
/// 公式: (x - min) / (max - min)
pub struct NormalizeExtractor {
    config: ExtractorConfig,
    output_dim: usize,
    // 训练时的统计信息（最小值、最大值）
    stats: Arc<RwLock<Option<NormalizeStats>>>,
}

#[derive(Debug, Clone)]
struct NormalizeStats {
    min: Vec<f32>,
    max: Vec<f32>,
    range: Vec<f32>, // max - min，用于避免重复计算
}

impl NormalizeExtractor {
    /// 创建新的归一化提取器
    pub fn new(config: ExtractorConfig) -> Result<Self> {
        let output_dim = config.output_dimension.ok_or_else(|| {
            Error::from(ExtractorError::Config("归一化提取器必须指定输出维度".to_string()))
        })?;
        
        Ok(Self {
            config,
            output_dim,
            stats: Arc::new(RwLock::new(None)),
        })
    }
    
    /// 从训练数据计算统计信息
    fn fit(&self, training_data: &[Vec<f32>]) -> Result<()> {
        if training_data.is_empty() {
            return Err(Error::from(ExtractorError::Internal("训练数据不能为空".to_string())));
        }
        
        let dim = training_data[0].len();
        if dim != self.output_dim {
            return Err(ExtractorError::Config(format!(
                "训练数据维度 {} 与输出维度 {} 不匹配",
                dim, self.output_dim
            )).into());
        }
        
        // 初始化最小值和最大值
        let mut min = training_data[0].clone();
        let mut max = training_data[0].clone();
        
        // 计算最小值和最大值
        for sample in training_data.iter().skip(1) {
            for (i, &value) in sample.iter().enumerate() {
                if value < min[i] {
                    min[i] = value;
                }
                if value > max[i] {
                    max[i] = value;
                }
            }
        }
        
        // 计算范围
        let mut range = Vec::with_capacity(dim);
        for i in 0..dim {
            let r = max[i] - min[i];
            range.push(if r > 1e-8 {
                r
            } else {
                1.0 // 避免除零，使用1.0作为默认值
            });
        }
        
        let stats = NormalizeStats { min, max, range };
        *self.stats.write().unwrap() = Some(stats);
        
        Ok(())
    }
    
    /// 应用归一化变换
    fn transform(&self, data: &[f32]) -> Result<Vec<f32>> {
        let stats = self.stats.read().unwrap();
        let stats = stats.as_ref().ok_or_else(|| {
            Error::from(ExtractorError::Internal("归一化提取器尚未拟合，请先调用fit方法".to_string()))
        })?;
        
        if data.len() != stats.min.len() {
            return Err(Error::from(ExtractorError::Internal(format!(
                "输入数据维度 {} 与统计信息维度 {} 不匹配",
                data.len(), stats.min.len()
            ))));
        }
        
        let mut result = Vec::with_capacity(data.len());
        for (i, &value) in data.iter().enumerate() {
            let normalized = (value - stats.min[i]) / stats.range[i];
            result.push(normalized.max(0.0).min(1.0)); // 确保在[0, 1]范围内
        }
        
        Ok(result)
    }
    
    /// 从输入数据提取数值向量
    fn extract_numeric_vector(&self, input: &InputData) -> Result<Vec<f32>> {
        match input {
            InputData::Tensor(data, shape) => {
                // 验证维度
                let total_elements: usize = shape.iter().product();
                if data.len() != total_elements {
                    return Err(Error::from(ExtractorError::Internal(format!(
                        "张量数据长度 {} 与形状 {:?} 不匹配",
                        data.len(), shape
                    ))));
                }
                
                Ok(data.clone())
            },
            _ => Err(Error::from(ExtractorError::Internal(format!(
                "归一化提取器不支持输入类型: {}",
                input.type_name()
            )))),
        }
    }
}

impl Debug for NormalizeExtractor {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("NormalizeExtractor")
            .field("output_dim", &self.output_dim)
            .field("fitted", &self.stats.read().unwrap().is_some())
            .finish()
    }
}

#[async_trait]
impl FeatureExtractor for NormalizeExtractor {
    fn extractor_type(&self) -> ExtractorType {
        ExtractorType::Numeric(NumericExtractorType::Normalize)
    }
    
    fn config(&self) -> &ExtractorConfig {
        &self.config
    }
    
    fn is_compatible(&self, input: &InputData) -> bool {
        matches!(input, InputData::Tensor(_, _))
    }
    
    async fn extract(&self, input: InputData, context: Option<ExtractorContext>) -> std::result::Result<FeatureVector, ExtractorError> {
        // 检查是否有训练数据需要拟合
        if let Some(ctx) = &context {
            if let Some(training_data_str) = ctx.get_param("training_data") {
                if let Ok(training_data) = serde_json::from_str::<Vec<Vec<f32>>>(training_data_str) {
                    self.fit(&training_data)?;
                }
            }
        }
        
        let numeric_data = self.extract_numeric_vector(&input)?;
        let transformed = self.transform(&numeric_data)?;
        
        Ok(FeatureVector::new(FeatureType::Numeric, transformed)
            .with_extractor_type(self.extractor_type()))
    }
    
    async fn batch_extract(&self, inputs: Vec<InputData>, context: Option<ExtractorContext>) -> std::result::Result<FeatureBatch, ExtractorError> {
        // 检查是否有训练数据需要拟合
        if let Some(ctx) = &context {
            if let Some(training_data_str) = ctx.get_param("training_data") {
                if let Ok(training_data) = serde_json::from_str::<Vec<Vec<f32>>>(training_data_str) {
                    self.fit(&training_data)?;
                }
            }
        }
        
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            let numeric_data = self.extract_numeric_vector(&input)?;
            let transformed = self.transform(&numeric_data)?;
            results.push(transformed);
        }
        
        Ok(FeatureBatch::new(results, FeatureType::Numeric)
            .with_extractor_type(self.extractor_type()))
    }
    
    fn output_feature_type(&self) -> FeatureType {
        FeatureType::Numeric
    }
    
    fn output_dimension(&self) -> Option<usize> {
        Some(self.output_dim)
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// 对数变换特征提取器
/// 
/// 对特征值应用对数变换
/// 公式: log(x + epsilon)
pub struct LogTransformExtractor {
    config: ExtractorConfig,
    output_dim: usize,
    epsilon: f32, // 用于处理零值和负值的小常数
}

impl LogTransformExtractor {
    /// 创建新的对数变换提取器
    pub fn new(config: ExtractorConfig) -> Result<Self> {
        let output_dim = config.output_dimension.ok_or_else(|| {
            Error::from(ExtractorError::Config("对数变换提取器必须指定输出维度".to_string()))
        })?;
        
        // 从配置获取epsilon，默认为1e-8
        let epsilon = config.get_param::<f32>("epsilon")
            .and_then(|r| r.ok())
            .unwrap_or(1e-8);
        
        Ok(Self {
            config,
            output_dim,
            epsilon,
        })
    }
    
    /// 应用对数变换
    fn transform(&self, data: &[f32]) -> Result<Vec<f32>> {
        let mut result = Vec::with_capacity(data.len());
        for &value in data {
            // 处理负值：如果值为负，先取绝对值再变换，然后加负号
            let transformed = if value < 0.0 {
                -((value.abs() + self.epsilon).ln())
            } else {
                (value + self.epsilon).ln()
            };
            result.push(transformed);
        }
        Ok(result)
    }
    
    /// 从输入数据提取数值向量
    fn extract_numeric_vector(&self, input: &InputData) -> Result<Vec<f32>> {
        match input {
            InputData::Tensor(data, shape) => {
                // 验证维度
                let total_elements: usize = shape.iter().product();
                if data.len() != total_elements {
                    return Err(Error::from(ExtractorError::Internal(format!(
                        "张量数据长度 {} 与形状 {:?} 不匹配",
                        data.len(), shape
                    ))));
                }
                
                Ok(data.clone())
            },
            _ => Err(Error::from(ExtractorError::Internal(format!(
                "对数变换提取器不支持输入类型: {}",
                input.type_name()
            )))),
        }
    }
}

impl Debug for LogTransformExtractor {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("LogTransformExtractor")
            .field("output_dim", &self.output_dim)
            .field("epsilon", &self.epsilon)
            .finish()
    }
}

#[async_trait]
impl FeatureExtractor for LogTransformExtractor {
    fn extractor_type(&self) -> ExtractorType {
        ExtractorType::Numeric(NumericExtractorType::LogTransform)
    }
    
    fn config(&self) -> &ExtractorConfig {
        &self.config
    }
    
    fn is_compatible(&self, input: &InputData) -> bool {
        matches!(input, InputData::Tensor(_, _))
    }
    
    async fn extract(&self, input: InputData, _context: Option<ExtractorContext>) -> std::result::Result<FeatureVector, ExtractorError> {
        let numeric_data = self.extract_numeric_vector(&input)?;
        let transformed = self.transform(&numeric_data)?;
        
        Ok(FeatureVector::new(FeatureType::Numeric, transformed)
            .with_extractor_type(self.extractor_type()))
    }
    
    async fn batch_extract(&self, inputs: Vec<InputData>, _context: Option<ExtractorContext>) -> std::result::Result<FeatureBatch, ExtractorError> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            let numeric_data = self.extract_numeric_vector(&input)?;
            let transformed = self.transform(&numeric_data)?;
            results.push(transformed);
        }
        
        Ok(FeatureBatch::new(results, FeatureType::Numeric)
            .with_extractor_type(self.extractor_type()))
    }
    
    fn output_feature_type(&self) -> FeatureType {
        FeatureType::Numeric
    }
    
    fn output_dimension(&self) -> Option<usize> {
        Some(self.output_dim)
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

