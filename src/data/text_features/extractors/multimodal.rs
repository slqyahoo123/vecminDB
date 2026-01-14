use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use serde_json::Value;
use crate::error::{Error, Result};
use crate::data::text_features::config::TextFeatureConfig;
use crate::data::text_features::extractors::FeatureExtractor as TextFeatureExtractor;
use crate::data::text_features::stats::DataCharacteristics;
use crate::data::text_features::types::TextFeatureMethod;
use crate::data::text_features::extractors::factory::create_extractor;
// ndarray arrays not directly used here; keep types minimal
use async_trait::async_trait;

/// 数据模态类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModalityType {
    /// 文本模态
    Text,
    /// 数值模态
    Numeric,
    /// 类别模态
    Categorical,
    /// 混合模态
    Mixed,
}

/// 模态权重配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityWeights {
    /// 文本特征权重
    pub text_weight: f32,
    /// 数值特征权重
    pub numeric_weight: f32,
    /// 类别特征权重
    pub categorical_weight: f32,
    /// 是否自动调整权重
    pub auto_adjust: bool,
}

impl Default for ModalityWeights {
    fn default() -> Self {
        Self {
            text_weight: 0.4,
            numeric_weight: 0.3,
            categorical_weight: 0.3,
            auto_adjust: true,
        }
    }
}

/// 多模态特征提取器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextMultimodalConfig {
    /// 文本特征提取配置
    pub text_config: TextFeatureConfig,
    /// 数值特征处理配置
    pub numeric_config: NumericFeatureConfig,
    /// 类别特征处理配置
    pub categorical_config: CategoricalFeatureConfig,
    /// 模态权重
    pub weights: ModalityWeights,
    /// 是否启用特征融合
    pub enable_fusion: bool,
    /// 最大特征维度
    pub max_dimension: usize,
    /// 是否正则化特征
    pub normalize_features: bool,
}

impl Default for TextMultimodalConfig {
    fn default() -> Self {
        Self {
            text_config: TextFeatureConfig::default(),
            numeric_config: NumericFeatureConfig::default(),
            categorical_config: CategoricalFeatureConfig::default(),
            weights: ModalityWeights::default(),
            enable_fusion: true,
            max_dimension: 300,
            normalize_features: true,
        }
    }
}

/// 数值特征配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericFeatureConfig {
    /// 是否启用缩放
    pub enable_scaling: bool,
    /// 缩放方法
    pub scaling_method: ScalingMethod,
    /// 是否处理缺失值
    pub handle_missing: bool,
    /// 最大特征数
    pub max_features: usize,
}

impl Default for NumericFeatureConfig {
    fn default() -> Self {
        Self {
            enable_scaling: true,
            scaling_method: ScalingMethod::StandardScaler,
            handle_missing: true,
            max_features: 50,
        }
    }
}

/// 缩放方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalingMethod {
    /// 标准缩放 (平均值=0，标准差=1)
    StandardScaler,
    /// 最小最大缩放 (范围0-1)
    MinMaxScaler,
    /// 鲁棒缩放 (基于四分位数)
    RobustScaler,
    /// 不进行缩放
    NoScaling,
}

/// 类别特征配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalFeatureConfig {
    /// 编码方法
    pub encoding_method: EncodingMethod,
    /// 最大类别数量 (超过则用其他表示)
    pub max_categories: usize,
    /// 最小频率 (低于则归为其他)
    pub min_frequency: f64,
    /// 是否处理未知类别
    pub handle_unknown: bool,
}

impl Default for CategoricalFeatureConfig {
    fn default() -> Self {
        Self {
            encoding_method: EncodingMethod::OneHot,
            max_categories: 100,
            min_frequency: 0.001,
            handle_unknown: true,
        }
    }
}

/// 类别编码方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncodingMethod {
    /// 独热编码
    OneHot,
    /// 标签编码
    LabelEncoding,
    /// 目标编码
    TargetEncoding,
    /// 计数编码
    CountEncoding,
}

/// 多模态特征提取结果
#[derive(Debug, Clone)]
pub struct MultimodalFeatures {
    /// 联合特征向量
    pub combined_features: Vec<f32>,
    /// 文本特征向量
    pub text_features: Option<Vec<f32>>,
    /// 数值特征向量
    pub numeric_features: Option<Vec<f32>>,
    /// 类别特征向量
    pub categorical_features: Option<Vec<f32>>,
    /// 各模态权重
    pub weights: ModalityWeights,
    /// 特征元数据
    pub metadata: HashMap<String, String>,
}

/// 多模态特征提取器
/// 
/// 支持同时处理文本、数值和类别特征，并融合为统一的特征向量
pub struct MultimodalFeatureExtractor {
    /// 提取器配置
    pub config: TextMultimodalConfig,
    /// 文本特征提取器
    text_extractor: Box<dyn TextFeatureExtractor>,
    /// 数据特征状态
    data_characteristics: Option<DataCharacteristics>,
    /// 上次自动调整时间
    last_adjustment: Option<std::time::SystemTime>,
    /// 当前权重
    current_weights: ModalityWeights,
    /// 类别到索引的映射表（用于OneHot和LabelEncoding）
    category_mapping: HashMap<String, usize>,
    /// 类别频率统计
    category_frequencies: HashMap<String, usize>,
}

impl Default for MultimodalFeatureExtractor {
    fn default() -> Self {
        let config = TextMultimodalConfig::default();
        // 使用工厂函数创建文本特征提取器
        let text_extractor = create_extractor(&config.text_config)
            .unwrap_or_else(|_| {
                // 如果创建失败，使用默认配置创建一个基本提取器
                let default_config = TextFeatureConfig::default();
                create_extractor(&default_config).unwrap_or_else(|_| {
                    // 如果仍然失败，返回一个错误而不是panic
                    // 注意：Default trait 不能返回Result，所以这里使用一个基本的TfIdf提取器作为后备
                    // 在实际使用中，应该确保配置正确
                    log::warn!("无法创建文本特征提取器，使用空配置重试");
                    // 创建一个最小配置的提取器
                    let mut minimal_config = TextFeatureConfig::default();
                    minimal_config.method = TextFeatureMethod::TfIdf;
                    create_extractor(&minimal_config).expect("无法创建最小配置的文本特征提取器，这不应该发生")
                })
            });
        
        Self {
            config: config.clone(),
            text_extractor,
            data_characteristics: None,
            last_adjustment: None,
            current_weights: config.weights,
            category_mapping: HashMap::new(),
            category_frequencies: HashMap::new(),
        }
    }
}

impl MultimodalFeatureExtractor {
    /// 从配置创建多模态特征提取器
    pub fn from_config(config: &TextMultimodalConfig) -> Result<Self> {
        let text_extractor = create_extractor(&config.text_config)?;
        
        Ok(Self {
            config: config.clone(),
            text_extractor,
            data_characteristics: None,
            last_adjustment: None,
            current_weights: config.weights.clone(),
            category_mapping: HashMap::new(),
            category_frequencies: HashMap::new(),
        })
    }
    
    /// 提取多模态特征
    pub fn extract_multimodal(&self, input: &MultimodalInput) -> Result<MultimodalFeatures> {
        // 提取各模态特征
        let text_features = self.extract_text_features(input)?;
        let numeric_features = self.extract_numeric_features(input)?;
        let categorical_features = self.extract_categorical_features(input)?;
        
        // 组合特征
        let combined_features = self.combine_features(
            text_features.as_deref(),
            numeric_features.as_deref(),
            categorical_features.as_deref(),
        )?;
        
        // 构建结果
        let features = MultimodalFeatures {
            combined_features,
            text_features,
            numeric_features,
            categorical_features,
            weights: self.current_weights.clone(),
            metadata: HashMap::new(),
        };
        
        Ok(features)
    }
    
    /// 提取文本特征
    fn extract_text_features(&self, input: &MultimodalInput) -> Result<Option<Vec<f32>>> {
        if let Some(text) = &input.text {
            if !text.is_empty() {
                let features = self.text_extractor.extract(text)?;
                return Ok(Some(features));
            }
        }
        
        Ok(None)
    }
    
    /// 提取数值特征
    fn extract_numeric_features(&self, input: &MultimodalInput) -> Result<Option<Vec<f32>>> {
        if let Some(numeric_values) = &input.numeric_values {
            if !numeric_values.is_empty() {
                let mut features = Vec::with_capacity(numeric_values.len());
                
                for value in numeric_values {
                    let processed_value = if self.config.numeric_config.enable_scaling {
                        match self.config.numeric_config.scaling_method {
                            ScalingMethod::StandardScaler => {
                                // 简化版标准化处理
                                (*value - 0.0) / 1.0
                            },
                            ScalingMethod::MinMaxScaler => {
                                // 简化版最小最大缩放
                                (*value - 0.0) / (1.0 - 0.0)
                            },
                            _ => *value,
                        }
                    } else {
                        *value
                    };
                    
                    features.push(processed_value);
                }
                
                return Ok(Some(features));
            }
        }
        
        Ok(None)
    }
    
    /// 提取类别特征
    fn extract_categorical_features(&self, input: &MultimodalInput) -> Result<Option<Vec<f32>>> {
        if let Some(categories) = &input.categories {
            if !categories.is_empty() {
                let mut features = Vec::new();
                
                match self.config.categorical_config.encoding_method {
                    EncodingMethod::OneHot => {
                        // 完整的One-Hot编码实现
                        // 获取所有唯一类别并构建映射
                        let max_categories = self.config.categorical_config.max_categories;
                        let num_categories = self.category_mapping.len().max(max_categories);
                        
                        for category in categories {
                            // 获取类别索引，如果不存在则使用未知类别索引
                            let category_index = self.category_mapping.get(category)
                                .copied()
                                .unwrap_or(num_categories); // 未知类别使用最后一个索引
                            
                            // 创建One-Hot向量
                            let mut one_hot = vec![0.0; num_categories + 1]; // +1 for unknown
                            if category_index < num_categories {
                                one_hot[category_index] = 1.0;
                            } else if self.config.categorical_config.handle_unknown {
                                // 未知类别放在最后一个位置
                                one_hot[num_categories] = 1.0;
                            }
                            
                            features.extend(one_hot);
                        }
                    },
                    EncodingMethod::LabelEncoding => {
                        // 完整的标签编码实现
                        let max_categories = self.config.categorical_config.max_categories;
                        
                        for category in categories {
                            // 获取类别索引
                            let category_index = self.category_mapping.get(category)
                                .copied();
                            
                            let encoded_value = if let Some(idx) = category_index {
                                idx as f32
                            } else if self.config.categorical_config.handle_unknown {
                                // 未知类别使用-1或最大索引+1
                                max_categories as f32
                            } else {
                                return Err(Error::InvalidData(format!("未知类别: {}", category)));
                            };
                            
                            features.push(encoded_value);
                        }
                    },
                    EncodingMethod::TargetEncoding => {
                        // 目标编码需要目标值，这里使用频率作为替代
                        for category in categories {
                            let frequency = self.category_frequencies.get(category)
                                .copied()
                                .unwrap_or(0) as f32;
                            let total = self.category_frequencies.values().sum::<usize>() as f32;
                            let encoded_value = if total > 0.0 {
                                frequency / total
                            } else {
                                0.0
                            };
                            features.push(encoded_value);
                        }
                    },
                    EncodingMethod::CountEncoding => {
                        // 计数编码：使用类别出现次数
                        for category in categories {
                            let count = self.category_frequencies.get(category)
                                .copied()
                                .unwrap_or(0) as f32;
                            features.push(count);
                        }
                    },
                }
                
                return Ok(Some(features));
            }
        }
        
        Ok(None)
    }
    
    /// 更新类别映射和频率统计
    pub fn update_category_mapping(&mut self, categories: &[String]) {
        for category in categories {
            // 更新频率
            *self.category_frequencies.entry(category.clone()).or_insert(0) += 1;
            
            // 更新映射（如果类别数量未超过限制）
            if self.category_mapping.len() < self.config.categorical_config.max_categories {
                self.category_mapping.entry(category.clone())
                    .or_insert_with(|| self.category_mapping.len());
            }
        }
    }
    
    /// 组合各模态特征
    fn combine_features(
        &self,
        text_features: Option<&[f32]>,
        numeric_features: Option<&[f32]>,
        categorical_features: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let mut combined = Vec::new();
        
        // 添加文本特征
        if let Some(features) = text_features {
            let weight = self.current_weights.text_weight;
            for &value in features {
                combined.push(value * weight);
            }
        }
        
        // 添加数值特征
        if let Some(features) = numeric_features {
            let weight = self.current_weights.numeric_weight;
            for &value in features {
                combined.push(value * weight);
            }
        }
        
        // 添加类别特征
        if let Some(features) = categorical_features {
            let weight = self.current_weights.categorical_weight;
            for &value in features {
                combined.push(value * weight);
            }
        }
        
        // 处理结果为空的情况
        if combined.is_empty() {
            return Err(Error::InvalidData("没有可用特征".to_string()));
        }
        
        // 如果维度过大，进行降维
        if combined.len() > self.config.max_dimension {
            // 简单截断降维，实际应使用PCA或其他降维方法
            combined.truncate(self.config.max_dimension);
        }
        
        // 正则化结果
        if self.config.normalize_features {
            // ... existing code ...
        }
        
        Ok(combined)
    }
    
    /// 根据数据特征自动调整模态权重
    pub fn auto_adjust_weights(&mut self, data_characteristics: &DataCharacteristics) -> Result<()> {
        if !self.config.weights.auto_adjust {
            return Ok(());
        }
        
        // 存储数据特征
        self.data_characteristics = Some(data_characteristics.clone());
        self.last_adjustment = Some(std::time::SystemTime::now());
        
        // 基于数据字段比例调整权重
        let text_ratio = data_characteristics.text_field_ratio;
        let numeric_ratio = data_characteristics.numeric_field_ratio;
        let categorical_ratio = data_characteristics.categorical_field_ratio;
        
        // 最小权重确保所有模态都有一定贡献
        let min_weight = 0.1;
        
        // 调整初始权重
        let mut text_weight = (text_ratio * 0.8).max(min_weight);
        let mut numeric_weight = (numeric_ratio * 0.8).max(min_weight);
        let mut categorical_weight = (categorical_ratio * 0.8).max(min_weight);
        
        // 进一步基于数据特征调整
        if data_characteristics.is_primarily_classification() {
            // 分类任务中类别特征更重要
            categorical_weight *= 1.2;
        } else if data_characteristics.is_primarily_regression() {
            // 回归任务中数值特征更重要
            numeric_weight *= 1.2;
        }
        
        // 如果文本字段具有高复杂性，增加其权重
        if let Some(complexity_score) = self.calculate_text_complexity(data_characteristics) {
            if complexity_score > 0.7 {
                text_weight *= 1.3;
            }
        }
        
        // 归一化权重
        let sum = text_weight + numeric_weight + categorical_weight;
        text_weight /= sum;
        numeric_weight /= sum;
        categorical_weight /= sum;
        
        // 更新当前权重（转换为f32）
        self.current_weights = ModalityWeights {
            text_weight: text_weight as f32,
            numeric_weight: numeric_weight as f32,
            categorical_weight: categorical_weight as f32,
            auto_adjust: true,
        };
        
        Ok(())
    }
    
    /// 计算文本复杂度得分
    fn calculate_text_complexity(&self, data_characteristics: &DataCharacteristics) -> Option<f64> {
        if data_characteristics.text_fields.is_empty() {
            return None;
        }
        
        // 计算平均文本长度
        let avg_length: f64 = data_characteristics.text_fields.values()
            .map(|stats| stats.avg_length)
            .sum::<f64>() / data_characteristics.text_fields.len() as f64;
            
        // 计算独特词比例
        let unique_ratio: f64 = data_characteristics.text_fields.values()
            .map(|stats| stats.unique_values as f64 / stats.count.max(1) as f64)
            .sum::<f64>() / data_characteristics.text_fields.len() as f64;
            
        // 综合计算复杂度
        let length_score = (avg_length / 500.0).min(1.0);
        let unique_score = (unique_ratio * 2.0).min(1.0);
        
        Some((length_score + unique_score) / 2.0)
    }
}

/// 多模态输入数据
#[derive(Debug, Clone)]
pub struct MultimodalInput {
    /// 文本数据
    pub text: Option<String>,
    /// 数值数据
    pub numeric_values: Option<Vec<f32>>,
    /// 类别数据
    pub categories: Option<Vec<String>>,
    /// 输入元数据
    pub metadata: HashMap<String, String>,
}

impl MultimodalInput {
    /// 创建新的多模态输入
    pub fn new() -> Self {
        Self {
            text: None,
            numeric_values: None,
            categories: None,
            metadata: HashMap::new(),
        }
    }
    
    /// 设置文本数据
    pub fn with_text(mut self, text: String) -> Self {
        self.text = Some(text);
        self
    }
    
    /// 设置数值数据
    pub fn with_numeric_values(mut self, values: Vec<f32>) -> Self {
        self.numeric_values = Some(values);
        self
    }
    
    /// 设置类别数据
    pub fn with_categories(mut self, categories: Vec<String>) -> Self {
        self.categories = Some(categories);
        self
    }
    
    /// 添加元数据
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    /// 检查是否包含任何有效数据
    pub fn has_data(&self) -> bool {
        self.text.as_ref().map_or(false, |t| !t.is_empty()) ||
        self.numeric_values.as_ref().map_or(false, |v| !v.is_empty()) ||
        self.categories.as_ref().map_or(false, |c| !c.is_empty())
    }
}

impl std::fmt::Debug for MultimodalFeatureExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultimodalFeatureExtractor")
            .field("config", &self.config)
            .field("text_extractor", &"<Box<dyn TextFeatureExtractor>>")
            .field("data_characteristics", &self.data_characteristics)
            .field("last_adjustment", &self.last_adjustment)
            .field("current_weights", &self.current_weights)
            .field("category_mapping", &self.category_mapping)
            .field("category_frequencies", &self.category_frequencies)
            .finish()
    }
}

#[async_trait]
impl crate::data::feature::extractor::FeatureExtractor for MultimodalFeatureExtractor {
    fn extractor_type(&self) -> crate::data::feature::types::ExtractorType {
        crate::data::feature::types::ExtractorType::Text(crate::data::feature::types::TextExtractorType::TfIdf)
    }
    
    fn config(&self) -> &crate::data::feature::extractor::ExtractorConfig {
        // 创建一个静态配置，实际项目中应该从self.config转换
        static CONFIG: std::sync::OnceLock<crate::data::feature::extractor::ExtractorConfig> = std::sync::OnceLock::new();
        CONFIG.get_or_init(|| {
            crate::data::feature::extractor::ExtractorConfig::new(crate::data::feature::types::ExtractorType::Text(crate::data::feature::types::TextExtractorType::TfIdf))
                .with_name("multimodal_extractor")
                .with_output_dim(512)
        })
    }
    
    fn is_compatible(&self, input: &crate::data::feature::extractor::InputData) -> bool {
        matches!(input, crate::data::feature::extractor::InputData::Text(_) | 
                        crate::data::feature::extractor::InputData::MultiModal(_))
    }

    async fn extract(&self, input: crate::data::feature::extractor::InputData, _context: Option<crate::data::feature::extractor::ExtractorContext>) -> std::result::Result<crate::data::feature::extractor::FeatureVector, crate::data::feature::extractor::ExtractorError> {
        match input {
            crate::data::feature::extractor::InputData::Text(text) => {
                let multimodal_input = MultimodalInput::new().with_text(text);
                let features = self.extract_multimodal(&multimodal_input)
                    .map_err(|e| crate::data::feature::extractor::ExtractorError::Extract(e.to_string()))?;
                
                Ok(crate::data::feature::extractor::FeatureVector::new(
                    crate::data::feature::types::FeatureType::Text,
                    features.combined_features
                ).with_extractor_type(self.extractor_type()))
            },
            crate::data::feature::extractor::InputData::MultiModal(data) => {
                // 从MultiModal数据中提取特征
                let mut multimodal_input = MultimodalInput::new();
                
                if let Some(text_data) = data.get("text") {
                    if let Some(text) = text_data.as_text() {
                        multimodal_input = multimodal_input.with_text(text.clone());
                    }
                }
                
                let features = self.extract_multimodal(&multimodal_input)
                    .map_err(|e| crate::data::feature::extractor::ExtractorError::Extract(e.to_string()))?;
                
                Ok(crate::data::feature::extractor::FeatureVector::new(
                    crate::data::feature::types::FeatureType::Text,
                    features.combined_features
                ).with_extractor_type(self.extractor_type()))
            },
            _ => Err(crate::data::feature::extractor::ExtractorError::InvalidInput(
                "Unsupported input type for multimodal extractor".to_string()
            ))
        }
    }

    async fn batch_extract(&self, inputs: Vec<crate::data::feature::extractor::InputData>, _context: Option<crate::data::feature::extractor::ExtractorContext>) -> std::result::Result<crate::data::feature::extractor::FeatureBatch, crate::data::feature::extractor::ExtractorError> {
        let mut batch_features = Vec::new();
        
        for input in inputs {
            let feature_vector = self.extract(input, None).await?;
            batch_features.push(feature_vector.values);
        }
        
        Ok(crate::data::feature::extractor::FeatureBatch::new(
            batch_features,
            crate::data::feature::types::FeatureType::Text
        ).with_extractor_type(self.extractor_type()))
    }

    fn output_feature_type(&self) -> crate::data::feature::types::FeatureType {
        crate::data::feature::types::FeatureType::Text
    }

    fn output_dimension(&self) -> Option<usize> {
        Some(self.config.max_dimension)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// 添加辅助方法
impl MultimodalFeatureExtractor {
    /// 获取特征维度
    pub fn dimension(&self) -> usize {
        self.config.max_dimension
    }
    
    /// 获取提取器名称
    pub fn name(&self) -> &str {
        "multimodal"
    }
    

    
    /// 获取输出维度
    pub fn get_output_dimension(&self) -> Result<usize> {
        Ok(self.dimension())
    }

    /// 使用自适应方法处理输入
    pub fn auto_extract(&mut self, input: &MultimodalInput, data_characteristics: Option<&DataCharacteristics>) -> Result<MultimodalFeatures> {
        // 如果提供了数据特征，先调整权重
        if let Some(characteristics) = data_characteristics {
            self.auto_adjust_weights(characteristics)?;
        }
        
        // 提取特征
        let features = self.extract_multimodal(input)?;
        
        Ok(features)
    }
}

impl MultimodalFeatureExtractor {
    /// 更新配置
    pub fn update_config(&mut self, config: TextMultimodalConfig) -> Result<()> {
        self.config = config.clone();
        self.text_extractor = create_extractor(&config.text_config)?;
        self.current_weights = config.weights;
        Ok(())
    }
    
    /// 检测最合适的特征处理方法
    pub fn detect_best_method(&self, input: &MultimodalInput) -> TextFeatureMethod {
        let modality = self.detect_primary_modality(input);
        
        match modality {
            ModalityType::Text => {
                // 针对文本选择合适的方法
                if let Some(text) = &input.text {
                    if text.len() > 1000 {
                        TextFeatureMethod::Bert
                    } else if text.contains(" ") {
                        TextFeatureMethod::TfIdf
                    } else {
                        TextFeatureMethod::BagOfWords
                    }
                } else {
                    TextFeatureMethod::default()
                }
            },
            ModalityType::Numeric => {
                // 数值数据默认使用词袋模型
                TextFeatureMethod::BagOfWords
            },
            ModalityType::Categorical => {
                // 类别数据默认使用词袋模型
                TextFeatureMethod::BagOfWords
            },
            ModalityType::Mixed => {
                // 混合数据推荐使用更高级的方法
                TextFeatureMethod::Word2Vec
            },
        }
    }
    
    /// 检测主要模态类型
    pub fn detect_primary_modality(&self, input: &MultimodalInput) -> ModalityType {
        if let Some(ref text) = input.text {
            if !text.is_empty() {
                return ModalityType::Text;
            }
        }
        
        if let Some(ref values) = input.numeric_values {
            if !values.is_empty() {
                return ModalityType::Numeric;
            }
        }
        
        if let Some(ref cats) = input.categories {
            if !cats.is_empty() {
                return ModalityType::Categorical;
            }
        }
        
        // 默认返回混合类型
        ModalityType::Mixed
    }
    
    /// 提取JSON数据中的特征
    pub fn extract_features(&self, data: &Value) -> Result<Vec<f32>> {
        let mut features = Vec::new();
        
        match data {
            Value::Object(obj) => {
                // 提取文本特征
                if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        let mut text_features = self.text_extractor.extract(text)?;
                        features.append(&mut text_features);
                    }
                }
                
                // 提取数值特征
                if let Some(values) = obj.get("numeric").and_then(|v| v.as_array()) {
                    let mut numeric_features = Vec::new();
                    for value in values {
                        if let Some(num) = value.as_f64() {
                            numeric_features.push(num as f32);
                        }
                    }
                    features.append(&mut numeric_features);
                }
                
                // 提取类别特征
                if let Some(categories) = obj.get("categories").and_then(|v| v.as_array()) {
                    for (i, cat) in categories.iter().enumerate().take(10) {
                        if let Some(c) = cat.as_str() {
                            // 简单编码：类别索引作为特征值
                            features.push(i as f32);
                        }
                    }
                }
            },
            Value::String(text) => {
                // 直接处理文本
                let mut text_features = self.text_extractor.extract(text)?;
                features.append(&mut text_features);
            },
            _ => {
                // 其他类型尝试转换为字符串处理
                let text = data.to_string();
                if !text.is_empty() {
                    let mut text_features = self.text_extractor.extract(&text)?;
                    features.append(&mut text_features);
                }
            }
        }
        
        // 确保提取了特征
        if features.is_empty() {
            return Err(Error::InvalidData("没有可用特征".to_string()));
        }
        
        Ok(features)
    }
} 