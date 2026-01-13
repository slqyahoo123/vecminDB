/// 特征提取器模块
/// 
/// 本模块包含各种多模态数据的特征提取器实现，用于从不同类型的
/// 原始数据中提取特征向量，以支持后续的AI模型训练和推理。

// 基础接口定义
pub mod interface;

// 各类型提取器实现
pub mod image;
pub mod text;
pub mod video;
pub mod audio;
pub mod fusion;

// 重新导出关键接口与类型
pub use interface::{ModalityExtractor, FeatureExtractor};

// 导出具体提取器
pub use image::ImageFeatureExtractor;

pub use text::TextFeatureExtractor;

pub use video::{
    VideoFeatureExtractor,
    create_default_video_extractor,
    VideoSource,
    VideoProcessingConfig,
    VideoFeatureType,
};

pub use audio::{
    AudioFeatureExtractor,
    create_default_audio_extractor,
    AudioProcessingConfig,
    AudioFeatureType,
    AudioSource,
};

pub use fusion::{
    FusionStrategy,
    FeatureFusionExtractor,
};

use crate::{Error, Result};
use std::collections::HashMap;
use super::{ModalityType, MultiModalConfig, ModalityAlignment, ModalityFusion};

/// 工厂方法：创建多模态处理器
pub fn create_multimodal_processor(config: MultiModalConfig) -> Result<Box<dyn interface::MultimodalProcessor>> {
    use super::MultiModalExtractor;
    let extractor = MultiModalExtractor::new(config)?;
    Ok(Box::new(extractor))
}

// 实现MultimodalProcessor接口
impl interface::MultimodalProcessor for super::MultiModalExtractor {
    fn process(&self, data: HashMap<String, serde_json::Value>) -> Result<interface::MultimodalExtractionResult> {
        // 提取各模态特征
        let mut modality_features = HashMap::new();
        
        for (modality_name, modality_data) in &data {
            // 将字符串转换为 ModalityType
            let modality_type = match modality_name.as_str() {
                "text" => ModalityType::Text,
                "image" => ModalityType::Image,
                "audio" => ModalityType::Audio,
                "video" => ModalityType::Video,
                _ => continue,
            };
            if let Some(extractor) = self.modality_extractors.get(&modality_type) {
                // 创建 MultiModalData
                let multi_modal_data = super::MultiModalData {
                    id: None,
                    text: if modality_type == ModalityType::Text {
                        serde_json::from_value(modality_data.clone()).ok()
                    } else { None },
                    images: if modality_type == ModalityType::Image {
                        serde_json::from_value(modality_data.clone()).ok()
                    } else { None },
                    audio: if modality_type == ModalityType::Audio {
                        serde_json::from_value(modality_data.clone()).ok()
                    } else { None },
                    video: if modality_type == ModalityType::Video {
                        serde_json::from_value(modality_data.clone()).ok()
                    } else { None },
                    metadata: HashMap::new(),
                };
                let feature = extractor.extract_features(&multi_modal_data)?;
                modality_features.insert(modality_type, feature);
            }
        }
        
        // 对齐特征（如果需要）
        let aligned_features = if let Some(alignment) = &self.alignment_module {
            alignment.align_features(modality_features.clone())?
        } else {
            modality_features.clone()
        };
        
        // 融合特征
        let fused_features = if let Some(ref fusion_module) = self.fusion_module {
            // 将 HashMap 转换为 Vec<ModalTensorData>
            let mut feature_vec = Vec::new();
            for (_, features) in &aligned_features {
                feature_vec.extend(features.clone());
            }
            let modal_tensor = fusion_module.fuse_features(feature_vec)?;
            // 将 ModalTensorData 转换为 CoreTensorData
            let now = chrono::Utc::now();
            interface::TensorData {
                id: uuid::Uuid::new_v4().to_string(),
                shape: modal_tensor.shape.clone(),
                data: modal_tensor.tensor.clone(),
                dtype: "float32".to_string(),
                device: "cpu".to_string(),
                requires_grad: false,
                metadata: modal_tensor.metadata.clone(),
                created_at: now,
                updated_at: now,
            }
        } else {
            return Err(Error::data("融合模块未配置".to_string()));
        };
        
        // 获取当前权重
        let mut weights = HashMap::new();
        let modality_count = aligned_features.len();
        if modality_count > 0 {
            let weight = 1.0 / modality_count as f32;
            for modality_type in aligned_features.keys() {
                weights.insert(format!("{:?}", modality_type), weight);
            }
        }
        
        // 转换 modality_features 为正确的类型
        let mut modality_features_map: HashMap<String, interface::TensorData> = HashMap::new();
        for (modality_type, features) in &modality_features {
            // 将 Vec<ModalTensorData> 转换为单个 TensorData
            // 这里需要将多个特征合并或选择第一个
            if let Some(first_feature) = features.first() {
                // 将 ModalTensorData 转换为 CoreTensorData
                let now = chrono::Utc::now();
                let tensor_data = interface::TensorData {
                    id: uuid::Uuid::new_v4().to_string(),
                    shape: first_feature.shape.clone(),
                    data: first_feature.tensor.clone(),
                    dtype: "float32".to_string(),
                    device: "cpu".to_string(),
                    requires_grad: false,
                    metadata: first_feature.metadata.clone(),
                    created_at: now,
                    updated_at: now,
                };
                modality_features_map.insert(format!("{:?}", modality_type), tensor_data);
            }
        }
        
        // 构建返回结果
        let result = interface::MultimodalExtractionResult {
            fused_features,
            modality_features: modality_features_map,
            weights,
            metadata: HashMap::new(),
        };
        
        Ok(result)
    }
    
    fn supported_modalities(&self) -> Vec<ModalityType> {
        // 返回所有配置的模态类型
        let mut modalities = Vec::new();
        if self.config.text.is_some() {
            modalities.push(ModalityType::Text);
        }
        if self.config.image.is_some() {
            modalities.push(ModalityType::Image);
        }
        if self.config.audio.is_some() {
            modalities.push(ModalityType::Audio);
        }
        if self.config.video.is_some() {
            modalities.push(ModalityType::Video);
        }
        modalities
    }
    
    fn update_config(&mut self, config: serde_json::Value) -> Result<()> {
        let new_config: MultiModalConfig = serde_json::from_value(config)
            .map_err(|e| Error::InvalidArgument(format!("无效的多模态配置: {}", e)))?;
            
        // 替换现有配置（实际可能需要更复杂的实现）
        *self = super::MultiModalExtractor::new(new_config)?;
        
        Ok(())
    }
} 