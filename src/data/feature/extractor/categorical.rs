// 分类特征提取器实现
// 提供OneHot、LabelEncoding、FrequencyEncoding等分类特征提取功能

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use async_trait::async_trait;

use crate::{Result, Error};
use crate::data::feature::extractor::{
    FeatureExtractor, ExtractorError, InputData, ExtractorContext,
    FeatureVector, FeatureBatch, ExtractorConfig
};
use crate::data::feature::types::{
    ExtractorType, FeatureType, CategoricalExtractorType
};

/// OneHot编码特征提取器
/// 
/// 将类别值转换为稀疏的二进制向量，只有一个位置为1，其余为0
pub struct OneHotExtractor {
    config: ExtractorConfig,
    categories: usize,
    // 类别到索引的映射（从训练数据学习）
    category_to_index: Arc<RwLock<Option<HashMap<String, usize>>>>,
}

impl OneHotExtractor {
    /// 创建新的OneHot编码提取器
    pub fn new(config: ExtractorConfig) -> Result<Self> {
        let categories = config.get_param::<usize>("categories")
            .and_then(|r| r.ok())
            .ok_or_else(|| {
                Error::from(ExtractorError::Config("OneHot编码提取器必须指定categories参数".to_string()))
            })?;
        
        if categories == 0 {
            return Err(Error::from(ExtractorError::Config("categories必须大于0".to_string())));
        }
        
        Ok(Self {
            config,
            categories,
            category_to_index: Arc::new(RwLock::new(None)),
        })
    }
    
    /// 从训练数据学习类别映射
    fn fit(&self, training_data: &[String]) -> Result<()> {
        let mut category_to_index = HashMap::new();
        let mut index = 0;
        
        for category in training_data {
            if !category_to_index.contains_key(category) {
                if index >= self.categories {
                    return Err(Error::from(ExtractorError::Config(format!(
                        "训练数据中的类别数量 {} 超过了指定的categories {}",
                        category_to_index.len() + 1,
                        self.categories
                    ))));
                }
                category_to_index.insert(category.clone(), index);
                index += 1;
            }
        }
        
        *self.category_to_index.write().unwrap() = Some(category_to_index);
        Ok(())
    }
    
    /// 应用OneHot编码
    fn transform(&self, category: &str) -> Result<Vec<f32>> {
        let mapping = self.category_to_index.read().unwrap();
        let mapping = mapping.as_ref().ok_or_else(|| {
            ExtractorError::Internal("OneHot提取器尚未拟合，请先调用fit方法".to_string())
        })?;
        
        let mut result = vec![0.0f32; self.categories];
        
        if let Some(&index) = mapping.get(category) {
            if index < self.categories {
                result[index] = 1.0;
            }
        } else {
            // 未知类别，返回全零向量（或可以返回错误）
            // 这里选择返回全零向量，表示未知类别
        }
        
        Ok(result)
    }
    
    /// 从输入数据提取类别字符串
    fn extract_category(&self, input: &InputData) -> Result<String> {
        match input {
            InputData::Text(text) => Ok(text.clone()),
            InputData::TextArray(texts) => {
                if texts.is_empty() {
                    return Err(Error::from(ExtractorError::Internal("文本数组不能为空".to_string())));
                }
                Ok(texts[0].clone()) // 取第一个文本作为类别
            },
            _ => Err(Error::from(ExtractorError::Internal(format!(
                "OneHot提取器不支持输入类型: {}",
                input.type_name()
            )))),
        }
    }
}

impl Debug for OneHotExtractor {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("OneHotExtractor")
            .field("categories", &self.categories)
            .field("fitted", &self.category_to_index.read().unwrap().is_some())
            .finish()
    }
}

#[async_trait]
impl FeatureExtractor for OneHotExtractor {
    fn extractor_type(&self) -> ExtractorType {
        ExtractorType::Categorical(CategoricalExtractorType::OneHot)
    }
    
    fn config(&self) -> &ExtractorConfig {
        &self.config
    }
    
    fn is_compatible(&self, input: &InputData) -> bool {
        matches!(input, InputData::Text(_) | InputData::TextArray(_))
    }
    
    async fn extract(&self, input: InputData, context: Option<ExtractorContext>) -> Result<FeatureVector, ExtractorError> {
        // 检查是否有训练数据需要拟合
        if let Some(ctx) = &context {
            if let Some(training_data_str) = ctx.get_param("training_data") {
                if let Ok(training_data) = serde_json::from_str::<Vec<String>>(training_data_str) {
                    self.fit(&training_data)?;
                }
            }
        }
        
        let category = self.extract_category(&input)?;
        let encoded = self.transform(&category)?;
        
        Ok(FeatureVector::new(FeatureType::Categorical, encoded)
            .with_extractor_type(self.extractor_type()))
    }
    
    async fn batch_extract(&self, inputs: Vec<InputData>, context: Option<ExtractorContext>) -> Result<FeatureBatch, ExtractorError> {
        // 检查是否有训练数据需要拟合
        if let Some(ctx) = &context {
            if let Some(training_data_str) = ctx.get_param("training_data") {
                if let Ok(training_data) = serde_json::from_str::<Vec<String>>(training_data_str) {
                    self.fit(&training_data)?;
                }
            }
        }
        
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            let category = self.extract_category(&input)?;
            let encoded = self.transform(&category)?;
            results.push(encoded);
        }
        
        Ok(FeatureBatch::new(results, FeatureType::Categorical)
            .with_extractor_type(self.extractor_type()))
    }
    
    fn output_feature_type(&self) -> FeatureType {
        FeatureType::Categorical
    }
    
    fn output_dimension(&self) -> Option<usize> {
        Some(self.categories)
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// 标签编码特征提取器
/// 
/// 将类别值映射到整数标签（0到categories-1）
pub struct LabelEncodingExtractor {
    config: ExtractorConfig,
    categories: usize,
    // 类别到标签的映射（从训练数据学习）
    category_to_label: Arc<RwLock<Option<HashMap<String, usize>>>>,
}

impl LabelEncodingExtractor {
    /// 创建新的标签编码提取器
    pub fn new(config: ExtractorConfig) -> Result<Self> {
        let categories = config.get_param::<usize>("categories")
            .and_then(|r| r.ok())
            .ok_or_else(|| {
                ExtractorError::Config("标签编码提取器必须指定categories参数".to_string())
            })?;
        
        if categories == 0 {
            return Err(Error::from(ExtractorError::Config("categories必须大于0".to_string())));
        }
        
        Ok(Self {
            config,
            categories,
            category_to_label: Arc::new(RwLock::new(None)),
        })
    }
    
    /// 从训练数据学习类别映射
    fn fit(&self, training_data: &[String]) -> Result<()> {
        let mut category_to_label = HashMap::new();
        let mut label = 0;
        
        for category in training_data {
            if !category_to_label.contains_key(category) {
                if label >= self.categories {
                    return Err(Error::from(ExtractorError::Config(format!(
                        "训练数据中的类别数量 {} 超过了指定的categories {}",
                        category_to_label.len() + 1,
                        self.categories
                    ))));
                }
                category_to_label.insert(category.clone(), label);
                label += 1;
            }
        }
        
        *self.category_to_label.write().unwrap() = Some(category_to_label);
        Ok(())
    }
    
    /// 应用标签编码
    fn transform(&self, category: &str) -> Result<Vec<f32>> {
        let mapping = self.category_to_label.read().unwrap();
        let mapping = mapping.as_ref().ok_or_else(|| {
            Error::from(ExtractorError::Internal("标签编码提取器尚未拟合，请先调用fit方法".to_string()))
        })?;
        
        let label = mapping.get(category)
            .copied()
            .unwrap_or(self.categories); // 未知类别使用categories作为标签
        
        Ok(vec![label as f32])
    }
    
    /// 从输入数据提取类别字符串
    fn extract_category(&self, input: &InputData) -> Result<String> {
        match input {
            InputData::Text(text) => Ok(text.clone()),
            InputData::TextArray(texts) => {
                if texts.is_empty() {
                    return Err(Error::from(ExtractorError::Internal("文本数组不能为空".to_string())));
                }
                Ok(texts[0].clone())
            },
            _ => Err(Error::from(ExtractorError::Internal(format!(
                "标签编码提取器不支持输入类型: {}",
                input.type_name()
            )))),
        }
    }
}

impl Debug for LabelEncodingExtractor {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("LabelEncodingExtractor")
            .field("categories", &self.categories)
            .field("fitted", &self.category_to_label.read().unwrap().is_some())
            .finish()
    }
}

#[async_trait]
impl FeatureExtractor for LabelEncodingExtractor {
    fn extractor_type(&self) -> ExtractorType {
        ExtractorType::Categorical(CategoricalExtractorType::LabelEncoding)
    }
    
    fn config(&self) -> &ExtractorConfig {
        &self.config
    }
    
    fn is_compatible(&self, input: &InputData) -> bool {
        matches!(input, InputData::Text(_) | InputData::TextArray(_))
    }
    
    async fn extract(&self, input: InputData, context: Option<ExtractorContext>) -> Result<FeatureVector, ExtractorError> {
        // 检查是否有训练数据需要拟合
        if let Some(ctx) = &context {
            if let Some(training_data_str) = ctx.get_param("training_data") {
                if let Ok(training_data) = serde_json::from_str::<Vec<String>>(training_data_str) {
                    self.fit(&training_data)?;
                }
            }
        }
        
        let category = self.extract_category(&input)?;
        let encoded = self.transform(&category)?;
        
        Ok(FeatureVector::new(FeatureType::Categorical, encoded)
            .with_extractor_type(self.extractor_type()))
    }
    
    async fn batch_extract(&self, inputs: Vec<InputData>, context: Option<ExtractorContext>) -> Result<FeatureBatch, ExtractorError> {
        // 检查是否有训练数据需要拟合
        if let Some(ctx) = &context {
            if let Some(training_data_str) = ctx.get_param("training_data") {
                if let Ok(training_data) = serde_json::from_str::<Vec<String>>(training_data_str) {
                    self.fit(&training_data)?;
                }
            }
        }
        
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            let category = self.extract_category(&input)?;
            let encoded = self.transform(&category)?;
            results.push(encoded);
        }
        
        Ok(FeatureBatch::new(results, FeatureType::Categorical)
            .with_extractor_type(self.extractor_type()))
    }
    
    fn output_feature_type(&self) -> FeatureType {
        FeatureType::Categorical
    }
    
    fn output_dimension(&self) -> Option<usize> {
        Some(1) // 标签编码输出单个值
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// 频率编码特征提取器
/// 
/// 将类别值映射到其出现频率（归一化到[0,1]）
pub struct FrequencyEncodingExtractor {
    config: ExtractorConfig,
    // 类别到频率的映射（从训练数据学习）
    category_to_frequency: Arc<RwLock<Option<HashMap<String, f32>>>>,
}

impl FrequencyEncodingExtractor {
    /// 创建新的频率编码提取器
    pub fn new(config: ExtractorConfig) -> Result<Self> {
        Ok(Self {
            config,
            category_to_frequency: Arc::new(RwLock::new(None)),
        })
    }
    
    /// 从训练数据学习频率映射
    fn fit(&self, training_data: &[String]) -> Result<()> {
        if training_data.is_empty() {
            return Err(Error::from(ExtractorError::Internal("训练数据不能为空".to_string())));
        }
        
        let mut category_counts = HashMap::new();
        let total = training_data.len() as f32;
        
        // 统计每个类别的出现次数
        for category in training_data {
            *category_counts.entry(category.clone()).or_insert(0) += 1;
        }
        
        // 计算频率（归一化到[0,1]）
        let mut category_to_frequency = HashMap::new();
        for (category, count) in category_counts {
            category_to_frequency.insert(category, count as f32 / total);
        }
        
        *self.category_to_frequency.write().unwrap() = Some(category_to_frequency);
        Ok(())
    }
    
    /// 应用频率编码
    fn transform(&self, category: &str) -> Result<Vec<f32>> {
        let mapping = self.category_to_frequency.read().unwrap();
        let mapping = mapping.as_ref().ok_or_else(|| {
            Error::from(ExtractorError::Internal("频率编码提取器尚未拟合，请先调用fit方法".to_string()))
        })?;
        
        let frequency = mapping.get(category)
            .copied()
            .unwrap_or(0.0); // 未知类别使用0.0作为频率
        
        Ok(vec![frequency])
    }
    
    /// 从输入数据提取类别字符串
    fn extract_category(&self, input: &InputData) -> Result<String> {
        match input {
            InputData::Text(text) => Ok(text.clone()),
            InputData::TextArray(texts) => {
                if texts.is_empty() {
                    return Err(Error::from(ExtractorError::Internal("文本数组不能为空".to_string())));
                }
                Ok(texts[0].clone())
            },
            _ => Err(Error::from(ExtractorError::Internal(format!(
                "频率编码提取器不支持输入类型: {}",
                input.type_name()
            )))),
        }
    }
}

impl Debug for FrequencyEncodingExtractor {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("FrequencyEncodingExtractor")
            .field("fitted", &self.category_to_frequency.read().unwrap().is_some())
            .finish()
    }
}

#[async_trait]
impl FeatureExtractor for FrequencyEncodingExtractor {
    fn extractor_type(&self) -> ExtractorType {
        ExtractorType::Categorical(CategoricalExtractorType::FrequencyEncoding)
    }
    
    fn config(&self) -> &ExtractorConfig {
        &self.config
    }
    
    fn is_compatible(&self, input: &InputData) -> bool {
        matches!(input, InputData::Text(_) | InputData::TextArray(_))
    }
    
    async fn extract(&self, input: InputData, context: Option<ExtractorContext>) -> Result<FeatureVector, ExtractorError> {
        // 检查是否有训练数据需要拟合
        if let Some(ctx) = &context {
            if let Some(training_data_str) = ctx.get_param("training_data") {
                if let Ok(training_data) = serde_json::from_str::<Vec<String>>(training_data_str) {
                    self.fit(&training_data)?;
                }
            }
        }
        
        let category = self.extract_category(&input)?;
        let encoded = self.transform(&category)?;
        
        Ok(FeatureVector::new(FeatureType::Categorical, encoded)
            .with_extractor_type(self.extractor_type()))
    }
    
    async fn batch_extract(&self, inputs: Vec<InputData>, context: Option<ExtractorContext>) -> Result<FeatureBatch, ExtractorError> {
        // 检查是否有训练数据需要拟合
        if let Some(ctx) = &context {
            if let Some(training_data_str) = ctx.get_param("training_data") {
                if let Ok(training_data) = serde_json::from_str::<Vec<String>>(training_data_str) {
                    self.fit(&training_data)?;
                }
            }
        }
        
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            let category = self.extract_category(&input)?;
            let encoded = self.transform(&category)?;
            results.push(encoded);
        }
        
        Ok(FeatureBatch::new(results, FeatureType::Categorical)
            .with_extractor_type(self.extractor_type()))
    }
    
    fn output_feature_type(&self) -> FeatureType {
        FeatureType::Categorical
    }
    
    fn output_dimension(&self) -> Option<usize> {
        Some(1) // 频率编码输出单个值
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

