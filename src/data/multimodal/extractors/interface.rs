use crate::{Error, Result};
use std::collections::HashMap;
use crate::core::CoreTensorData;
use super::super::{ModalityType, FusionStrategy, AlignmentMethod};
use std::fmt::Debug;

// 为了向后兼容，创建类型别名
pub type TensorData = CoreTensorData;

/// 特征提取器接口
/// 
/// 定义了所有特征提取器必须实现的基本方法
pub trait FeatureExtractor: Send + Sync + Debug {
    /// 从输入数据中提取特征
    fn extract_features(&self, data: &[u8], metadata: Option<&HashMap<String, String>>) -> Result<Vec<f32>>;
    
    /// 批量提取特征，默认实现为顺序处理，子类可以重写为并行处理
    fn batch_extract(&self, data_batch: &[Vec<u8>], metadata_batch: Option<&[HashMap<String, String>]>) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(data_batch.len());
        
        let metadata_iter = if let Some(meta_batch) = metadata_batch {
            Some(meta_batch.iter())
        } else {
            None
        };
        
        for (i, data) in data_batch.iter().enumerate() {
            let metadata = if let Some(iter) = &metadata_iter {
                iter.nth(i).map(|m| m)
            } else {
                None
            };
            
            results.push(self.extract_features(data, metadata)?);
        }
        
        Ok(results)
    }
    
    /// 获取提取器输出特征的维度
    fn get_output_dim(&self) -> usize;
    
    /// 获取提取器类型
    fn get_extractor_type(&self) -> String;
}

/// 特征融合接口
///
/// 用于融合多个特征向量
pub trait FeatureFusion: Send + Sync + Debug {
    /// 融合多个特征向量
    fn fuse_features(&self, features: &[Vec<f32>]) -> Result<Vec<f32>>;
    
    /// 获取融合后特征的维度
    fn get_output_dim(&self) -> usize;
    
    /// 获取融合器类型
    fn get_fusion_type(&self) -> String;
}

/// 可用的提取器类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractorType {
    /// 图像特征
    Image,
    /// 文本特征
    Text,
    /// 视频特征
    Video,
    /// 音频特征
    Audio,
    /// 融合特征
    Fusion,
}

impl ToString for ExtractorType {
    fn to_string(&self) -> String {
        match self {
            ExtractorType::Image => "image".to_string(),
            ExtractorType::Text => "text".to_string(),
            ExtractorType::Video => "video".to_string(),
            ExtractorType::Audio => "audio".to_string(),
            ExtractorType::Fusion => "fusion".to_string(),
        }
    }
}

impl From<&str> for ExtractorType {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "image" => ExtractorType::Image,
            "text" => ExtractorType::Text,
            "video" => ExtractorType::Video,
            "audio" => ExtractorType::Audio,
            "fusion" => ExtractorType::Fusion,
            _ => ExtractorType::Text, // 默认为文本
        }
    }
}

/// 模态提取器接口
pub trait ModalityExtractor {
    /// 提取特征
    fn extract_features(&self, data: &serde_json::Value) -> Result<TensorData>;
    
    /// 获取配置
    fn get_config(&self) -> Result<serde_json::Value>;
    
    /// 获取模态类型
    fn get_modality_type(&self) -> ModalityType;
    
    /// 获取特征维度
    fn get_dimension(&self) -> usize;
}

/// 模态对齐接口
pub trait ModalityAlignment {
    /// 对齐特征
    fn align_features(&self, features: HashMap<String, TensorData>) -> Result<HashMap<String, TensorData>>;
    
    /// 获取对齐方法
    fn get_alignment_method(&self) -> AlignmentMethod;
    
    /// 获取参考模态
    fn get_reference_modality(&self) -> Option<String>;
}

/// 模态融合接口
pub trait ModalityFusion {
    /// 融合特征
    fn fuse_features(&self, features: HashMap<String, TensorData>) -> Result<TensorData>;
    
    /// 获取融合策略
    fn get_fusion_strategy(&self) -> FusionStrategy;
    
    /// 获取输出维度
    fn get_output_dimension(&self) -> usize;
}

/// 多模态提取结果
#[derive(Debug, Clone)]
pub struct MultimodalExtractionResult {
    /// 融合后的特征
    pub fused_features: TensorData,
    /// 各模态原始特征
    pub modality_features: HashMap<String, TensorData>,
    /// 权重
    pub weights: HashMap<String, f32>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

/// 多模态处理器接口
pub trait MultimodalProcessor {
    /// 处理多模态数据
    fn process(&self, data: HashMap<String, serde_json::Value>) -> Result<MultimodalExtractionResult>;
    
    /// 获取支持的模态
    fn supported_modalities(&self) -> Vec<ModalityType>;
    
    /// 更新配置
    fn update_config(&mut self, config: serde_json::Value) -> Result<()>;
}

/// 创建特定模态的特征提取器
pub fn create_modality_extractor(
    modality_type: ModalityType,
    config: serde_json::Value
) -> Result<Box<dyn ModalityExtractor + Send + Sync>> {
    use super::super::{
        TextModalityExtractor, 
        ImageModalityExtractor, AudioModalityExtractor
    };
    
    match modality_type {
        ModalityType::Text => {
            let text_config = serde_json::from_value(config)
                .map_err(|e| Error::invalid_argument(format!("无效的文本特征配置: {}", e)))?;
            let extractor = TextModalityExtractor::new(text_config)?;
            Ok(Box::new(extractor))
        },
        ModalityType::Image => {
            let image_config = serde_json::from_value(config)
                .map_err(|e| Error::invalid_argument(format!("无效的图像特征配置: {}", e)))?;
            let extractor = ImageModalityExtractor::new(image_config)?;
            Ok(Box::new(extractor))
        },
        ModalityType::Audio => {
            let audio_config = serde_json::from_value(config)
                .map_err(|e| Error::invalid_argument(format!("无效的音频特征配置: {}", e)))?;
            let extractor = AudioModalityExtractor::new(audio_config)?;
            Ok(Box::new(extractor))
        },
        ModalityType::Custom(name) => {
            Err(Error::invalid_argument(format!("暂不支持自定义模态类型: {}", name)))
        },
        _ => {
            Err(Error::invalid_argument(format!("不支持的模态类型: {:?}", modality_type)))
        }
    }
} 