// 通用特征提取器实现
// 提供Identity、FeatureSelection等通用特征提取功能

use std::fmt::{Debug, Formatter, Result as FmtResult};
use async_trait::async_trait;

use crate::{Result, Error};
use crate::data::feature::extractor::{
    FeatureExtractor, ExtractorError, InputData, ExtractorContext,
    FeatureVector, FeatureBatch, ExtractorConfig
};
use crate::data::feature::types::{
    ExtractorType, FeatureType, GenericExtractorType
};

/// 身份特征提取器
/// 
/// 直接返回输入数据，不进行任何变换（用于测试和调试）
pub struct IdentityExtractor {
    config: ExtractorConfig,
    output_dim: usize,
}

impl IdentityExtractor {
    /// 创建新的身份提取器
    pub fn new(config: ExtractorConfig) -> Result<Self> {
        let output_dim = config.output_dimension.ok_or_else(|| {
            Error::from(ExtractorError::Config("身份提取器必须指定输出维度".to_string()))
        })?;
        
        if output_dim == 0 {
            return Err(Error::from(ExtractorError::Config("输出维度必须大于0".to_string())));
        }
        
        Ok(Self {
            config,
            output_dim,
        })
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
                
                // 验证输出维度
                if data.len() != self.output_dim {
                    return Err(Error::from(ExtractorError::Internal(format!(
                        "输入数据维度 {} 与输出维度 {} 不匹配",
                        data.len(), self.output_dim
                    ))));
                }
                
                Ok(data.clone())
            },
            _ => Err(Error::from(ExtractorError::Internal(format!(
                "身份提取器不支持输入类型: {}",
                input.type_name()
            )))),
        }
    }
}

impl Debug for IdentityExtractor {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("IdentityExtractor")
            .field("output_dim", &self.output_dim)
            .finish()
    }
}

#[async_trait]
impl FeatureExtractor for IdentityExtractor {
    fn extractor_type(&self) -> ExtractorType {
        ExtractorType::Generic(GenericExtractorType::Identity)
    }
    
    fn config(&self) -> &ExtractorConfig {
        &self.config
    }
    
    fn is_compatible(&self, input: &InputData) -> bool {
        matches!(input, InputData::Tensor(_, _))
    }
    
    async fn extract(&self, input: InputData, _context: Option<ExtractorContext>) -> Result<FeatureVector, ExtractorError> {
        let numeric_data = self.extract_numeric_vector(&input)?;
        
        // 身份变换：直接返回输入数据
        Ok(FeatureVector::new(FeatureType::Generic, numeric_data)
            .with_extractor_type(self.extractor_type()))
    }
    
    async fn batch_extract(&self, inputs: Vec<InputData>, _context: Option<ExtractorContext>) -> Result<FeatureBatch, ExtractorError> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            let numeric_data = self.extract_numeric_vector(&input)?;
            results.push(numeric_data);
        }
        
        Ok(FeatureBatch::new(results, FeatureType::Generic)
            .with_extractor_type(self.extractor_type()))
    }
    
    fn output_feature_type(&self) -> FeatureType {
        FeatureType::Generic
    }
    
    fn output_dimension(&self) -> Option<usize> {
        Some(self.output_dim)
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// 特征选择提取器
/// 
/// 使用特征重要性评分选择最重要的特征子集
pub struct FeatureSelectionExtractor {
    config: ExtractorConfig,
    input_dim: usize,
    output_dim: usize,
    // 选择的特征索引（从训练数据学习）
    selected_indices: Option<Vec<usize>>,
}

impl FeatureSelectionExtractor {
    /// 创建新的特征选择提取器
    pub fn new(config: ExtractorConfig) -> Result<Self> {
        let output_dim = config.output_dimension.ok_or_else(|| {
            Error::from(ExtractorError::Config("特征选择提取器必须指定输出维度".to_string()))
        })?;
        
        let input_dim = config.get_param::<usize>("input_dim")
            .and_then(|r| r.ok())
            .ok_or_else(|| {
                Error::from(ExtractorError::Config("特征选择提取器必须指定input_dim参数".to_string()))
            })?;
        
        if output_dim == 0 {
            return Err(Error::from(ExtractorError::Config("输出维度必须大于0".to_string())));
        }
        
        if input_dim == 0 {
            return Err(Error::from(ExtractorError::Config("输入维度必须大于0".to_string())));
        }
        
        if output_dim > input_dim {
            return Err(Error::from(ExtractorError::Config(format!(
                "输出维度 {} 不能大于输入维度 {}",
                output_dim, input_dim
            ))));
        }
        
        Ok(Self {
            config,
            input_dim,
            output_dim,
            selected_indices: None,
        })
    }
    
    /// 从训练数据学习特征选择（使用方差作为重要性评分）
    /// 注意：由于trait方法签名限制，fit方法需要单独调用
    pub fn fit(&mut self, training_data: &[Vec<f32>]) -> Result<()> {
        if training_data.is_empty() {
            return Err(Error::from(ExtractorError::Internal("训练数据不能为空".to_string())));
        }
        
        // 验证输入维度
        for sample in training_data {
            if sample.len() != self.input_dim {
                return Err(Error::from(ExtractorError::Internal(format!(
                        "训练数据维度 {} 与输入维度 {} 不匹配",
                        sample.len(), self.input_dim
                    ))));
            }
        }
        
        // 计算每个特征的方差（作为重要性评分）
        let mut variances = vec![0.0f32; self.input_dim];
        
        // 计算均值
        let mut means = vec![0.0f32; self.input_dim];
        for sample in training_data {
            for (i, &value) in sample.iter().enumerate() {
                means[i] += value;
            }
        }
        let n = training_data.len() as f32;
        for mean in &mut means {
            *mean /= n;
        }
        
        // 计算方差
        for sample in training_data {
            for (i, &value) in sample.iter().enumerate() {
                let diff = value - means[i];
                variances[i] += diff * diff;
            }
        }
        for variance in &mut variances {
            *variance /= n;
        }
        
        // 选择方差最大的output_dim个特征
        let mut indexed_variances: Vec<(usize, f32)> = variances
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        
        indexed_variances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let selected_indices: Vec<usize> = indexed_variances
            .iter()
            .take(self.output_dim)
            .map(|(i, _)| *i)
            .collect();
        
        self.selected_indices = Some(selected_indices);
        Ok(())
    }
    
    /// 应用特征选择
    fn transform(&self, data: &[f32]) -> Result<Vec<f32>> {
        let indices = self.selected_indices.as_ref().ok_or_else(|| {
            Error::from(ExtractorError::Internal("特征选择提取器尚未拟合，请先调用fit方法".to_string()))
        })?;
        
        if data.len() != self.input_dim {
            return Err(Error::from(ExtractorError::Internal(format!(
                "输入数据维度 {} 与输入维度 {} 不匹配",
                data.len(), self.input_dim
            ))));
        }
        
        let mut result = Vec::with_capacity(self.output_dim);
        for &index in indices {
            if index >= data.len() {
                return Err(Error::from(ExtractorError::Internal(format!(
                    "特征索引 {} 超出数据范围",
                    index
                ))));
            }
            result.push(data[index]);
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
                
                if data.len() != self.input_dim {
                    return Err(Error::from(ExtractorError::Internal(format!(
                "输入数据维度 {} 与输入维度 {} 不匹配",
                data.len(), self.input_dim
            ))));
                }
                
                Ok(data.clone())
            },
            _ => Err(Error::from(ExtractorError::Internal(format!(
                "特征选择提取器不支持输入类型: {}",
                input.type_name()
            )))),
        }
    }
}

impl Debug for FeatureSelectionExtractor {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("FeatureSelectionExtractor")
            .field("input_dim", &self.input_dim)
            .field("output_dim", &self.output_dim)
            .field("fitted", &self.selected_indices.is_some())
            .finish()
    }
}

#[async_trait]
impl FeatureExtractor for FeatureSelectionExtractor {
    fn extractor_type(&self) -> ExtractorType {
        ExtractorType::Generic(GenericExtractorType::FeatureSelection)
    }
    
    fn config(&self) -> &ExtractorConfig {
        &self.config
    }
    
    fn is_compatible(&self, input: &InputData) -> bool {
        matches!(input, InputData::Tensor(_, _))
    }
    
    async fn extract(&self, input: InputData, context: Option<ExtractorContext>) -> Result<FeatureVector, ExtractorError> {
        // 注意：这里需要&mut self来调用fit，但trait方法签名不允许
        // 实际使用中，fit应该在创建提取器后单独调用
        // 这里我们假设已经拟合过了
        
        let numeric_data = self.extract_numeric_vector(&input)?;
        let transformed = self.transform(&numeric_data)?;
        
        Ok(FeatureVector::new(FeatureType::Generic, transformed)
            .with_extractor_type(self.extractor_type()))
    }
    
    async fn batch_extract(&self, inputs: Vec<InputData>, _context: Option<ExtractorContext>) -> Result<FeatureBatch, ExtractorError> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            let numeric_data = self.extract_numeric_vector(&input)?;
            let transformed = self.transform(&numeric_data)?;
            results.push(transformed);
        }
        
        Ok(FeatureBatch::new(results, FeatureType::Generic)
            .with_extractor_type(self.extractor_type()))
    }
    
    fn output_feature_type(&self) -> FeatureType {
        FeatureType::Generic
    }
    
    fn output_dimension(&self) -> Option<usize> {
        Some(self.output_dim)
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

