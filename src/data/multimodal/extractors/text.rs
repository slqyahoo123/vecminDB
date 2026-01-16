use std::collections::HashMap;
use std::any::Any;
use crate::error::{Error, Result};
use crate::core::types::CoreTensorData;
use crate::data::text_features::TextFeatureExtractor as BaseTextExtractor;
use async_trait::async_trait;
use super::interface::ModalityExtractor;
use super::super::ModalityType;
use crate::data::feature::extractor::{FeatureExtractor, InputData, ExtractorContext, ExtractorConfig, ExtractorError, FeatureVector};
use crate::data::feature::types::{FeatureType, ExtractorType};

/// 文本特征提取配置
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MultimodalTextFeatureConfig {
    /// 特征维度
    pub dimension: usize,
    /// 提取器类型
    pub extractor_type: String,
    /// 使用缓存
    pub use_cache: bool,
    /// 提取器参数
    pub params: HashMap<String, String>,
}

impl Default for MultimodalTextFeatureConfig {
    fn default() -> Self {
        Self {
            dimension: 768,
            extractor_type: "bert".to_string(),
            use_cache: true,
            params: HashMap::new(),
        }
    }
}

/// 文本特征提取器
pub struct TextFeatureExtractor {
    /// 配置
    config: MultimodalTextFeatureConfig,
    /// 基础提取器
    extractor: Option<Box<dyn BaseTextExtractor>>,
    /// 提取器配置
    extractor_config: ExtractorConfig,
}

impl std::fmt::Debug for TextFeatureExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextFeatureExtractor")
            .field("config", &self.config)
            .field("extractor", &if self.extractor.is_some() { "<Some>" } else { "<None>" })
            .field("extractor_config", &self.extractor_config)
            .finish()
    }
}

impl TextFeatureExtractor {
    /// 创建新的文本特征提取器
    pub fn new(config: MultimodalTextFeatureConfig) -> Result<Self> {
        // 创建ExtractorConfig
        let mut extractor_config = ExtractorConfig::new(
            ExtractorType::TextBERT
        );
        extractor_config = extractor_config.with_output_dim(config.dimension);
        
        // 实际实现中，这里会根据config创建对应的BaseTextExtractor
        // 为了MVP，先返回一个空实现
        Ok(Self {
            config,
            extractor: None,
            extractor_config,
        })
    }

    /// 确保提取器初始化
    fn ensure_initialized(&mut self) -> Result<()> {
        if self.extractor.is_none() {
            // 尝试创建基础文本提取器
            // 注意：这里使用模拟实现，因为BaseTextExtractor的具体实现可能不可用
            // 在实际生产环境中，应该根据config创建对应的提取器实例
            log::debug!("TextFeatureExtractor 使用内置特征提取实现，无需外部提取器初始化");
        }
        Ok(())
    }
}

impl ModalityExtractor for TextFeatureExtractor {
    fn extract_features(&self, data: &serde_json::Value) -> Result<CoreTensorData> {
        // 将JSON数据转换为文本
        let text = match data {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Object(obj) => {
                if let Some(serde_json::Value::String(s)) = obj.get("text") {
                    s.clone()
                } else {
                    return Err(Error::invalid_argument("Text data not found in JSON object".to_string()));
                }
            },
            _ => return Err(Error::invalid_argument("Invalid text data format".to_string())),
        };
        
        // 提取特征
        let input = InputData::Text(text);
        let feature_vector = tokio::runtime::Runtime::new()
            .map_err(|e| Error::system(format!("Failed to create runtime: {}", e)))?
            .block_on(async {
                self.extract(input, None)
                    .await
                    .map_err(|e| Error::invalid_data(format!("Failed to extract text features: {}", e)))
            })?;
        
        // 转换为CoreTensorData
        use chrono::Utc;
        use uuid::Uuid;
        let tensor = CoreTensorData {
            id: Uuid::new_v4().to_string(),
            shape: vec![1, feature_vector.values.len()],
            data: feature_vector.values,
            dtype: "float32".to_string(),
            device: "cpu".to_string(),
            requires_grad: false,
            metadata: HashMap::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        
        Ok(tensor)
    }
    
    fn get_config(&self) -> Result<serde_json::Value> {
        let config = self.config.clone();
        Ok(serde_json::to_value(config)?)
    }
    
    fn get_modality_type(&self) -> ModalityType {
        ModalityType::Text
    }
    
    fn get_dimension(&self) -> usize {
        self.config.dimension
    }
}

#[async_trait]
impl FeatureExtractor for TextFeatureExtractor {
    fn extractor_type(&self) -> ExtractorType {
        ExtractorType::TextBERT
    }
    
    fn config(&self) -> &ExtractorConfig {
        &self.extractor_config
    }
    
    fn is_compatible(&self, input: &InputData) -> bool {
        matches!(input, InputData::Text(_) | InputData::TextArray(_))
    }
    
    async fn extract(&self, input: InputData, _context: Option<ExtractorContext>) -> std::result::Result<FeatureVector, ExtractorError> {
        // 处理输入数据
        let text = match input {
            InputData::Text(text) => text,
            InputData::TextArray(texts) if !texts.is_empty() => texts[0].clone(),
            _ => return Err(ExtractorError::InputData("不支持的输入数据类型".to_string())),
        };
        
        // 生成模拟特征数据
        let mut features = Vec::with_capacity(self.config.dimension);
            
        // 使用文本内容生成简单hash
        let mut hash_value: u64 = 0;
        for b in text.bytes() {
            hash_value = hash_value.wrapping_mul(31).wrapping_add(b as u64);
        }
        
        // 使用hash值生成伪随机特征，确保相同输入产生相同输出
        let mut state = hash_value;
        for _ in 0..self.config.dimension {
            // 简单的xorshift随机数生成
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            
            // 将u64转换为-1到1之间的f32
            let value = (state as f32 / std::u64::MAX as f32) * 2.0 - 1.0;
            features.push(value);
        }
        
        // L2归一化
        let mut norm = 0.0;
        for val in &features {
            norm += val * val;
        }
        norm = norm.sqrt();
        
        // 应用归一化
        if norm > 1e-8 {
            for val in &mut features {
                *val /= norm;
            }
        }
        
        // 创建FeatureVector
        let feature_vector = FeatureVector::new(
            FeatureType::Text, 
            features
        ).with_extractor_type(self.extractor_type());
        
        Ok(feature_vector)
    }
    
    async fn batch_extract(&self, inputs: Vec<InputData>, _context: Option<ExtractorContext>) -> std::result::Result<crate::data::feature::extractor::FeatureBatch, ExtractorError> {
        let mut feature_vectors = Vec::with_capacity(inputs.len());
        
        for input in inputs {
            let feature_vector = self.extract(input, None).await?;
            feature_vectors.push(feature_vector.values);
        }
        
        Ok(crate::data::feature::extractor::FeatureBatch::new(
            feature_vectors,
            FeatureType::Text
        ).with_extractor_type(self.extractor_type()))
    }
    
    fn output_feature_type(&self) -> FeatureType {
        FeatureType::Text
    }
    
    fn output_dimension(&self) -> Option<usize> {
        Some(self.config.dimension)
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
} 