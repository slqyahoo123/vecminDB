// 特征提取器配置
// 定义用于创建特征提取器的配置结构

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::data::feature::types::ExtractorType;

/// 通用提取器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractorConfig {
    /// 提取器类型
    pub extractor_type: ExtractorType,
    /// 输出特征维度
    pub dimension: usize,
    /// 特征提取器参数
    pub params: HashMap<String, serde_json::Value>,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        use crate::data::feature::types::TextExtractorType;
        Self {
            extractor_type: ExtractorType::Text(TextExtractorType::TfIdf),
            dimension: 100,
            params: HashMap::new(),
        }
    }
}

impl ExtractorConfig {
    /// 创建新的提取器配置
    pub fn new(extractor_type: ExtractorType) -> Self {
        Self {
            extractor_type,
            dimension: 100,
            params: HashMap::new(),
        }
    }
    
    /// 设置维度
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }
    
    /// 添加参数
    pub fn with_param<K: Into<String>, V: Into<serde_json::Value>>(mut self, key: K, value: V) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }
    
    /// 获取参数
    pub fn get_param(&self, key: &str) -> Option<&serde_json::Value> {
        self.params.get(key)
    }
    
    /// 获取参数作为字符串
    pub fn get_param_as_string(&self, key: &str) -> Option<String> {
        self.params.get(key).and_then(|v| v.as_str().map(|s| s.to_string()))
    }
    
    /// 获取参数作为数字
    pub fn get_param_as_usize(&self, key: &str) -> Option<usize> {
        self.params.get(key).and_then(|v| v.as_u64().map(|n| n as usize))
    }
}

/// 文本特征提取器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextExtractorConfig {
    /// 基础配置
    #[serde(flatten)]
    pub base: ExtractorConfig,
    /// 文本特征方法
    pub method: String,
    /// 是否区分大小写
    pub case_sensitive: bool,
    /// 最大特征数量
    pub max_features: usize,
    /// 停用词列表
    pub stop_words: Option<Vec<String>>,
    /// 是否使用IDF
    pub use_idf: bool,
    /// 最小文档频率
    pub min_df: Option<f32>,
    /// 最大文档频率
    pub max_df: Option<f32>,
    /// n-gram范围
    pub ngram_range: Option<(usize, usize)>,
    /// 是否启用缓存
    pub cache_enabled: bool,
}

impl Default for TextExtractorConfig {
    fn default() -> Self {
        use crate::data::feature::types::TextExtractorType;
        Self {
            base: ExtractorConfig {
                extractor_type: ExtractorType::Text(TextExtractorType::TfIdf),
                dimension: 100,
                params: HashMap::new(),
            },
            method: "tfidf".to_string(),
            case_sensitive: false,
            max_features: 10000,
            stop_words: None,
            use_idf: true,
            min_df: Some(0.0),
            max_df: Some(1.0),
            ngram_range: Some((1, 1)),
            cache_enabled: true,
        }
    }
}

/// 图像特征提取器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageExtractorConfig {
    /// 基础配置
    #[serde(flatten)]
    pub base: ExtractorConfig,
    /// 模型名称
    pub model_name: String,
    /// 目标大小
    pub target_size: (usize, usize),
    /// 是否使用预训练模型
    pub use_pretrained: bool,
    /// 模型路径
    pub model_path: Option<String>,
    /// 是否使用GPU
    pub use_gpu: bool,
    /// 归一化方式
    pub normalization_type: String,
}

impl Default for ImageExtractorConfig {
    fn default() -> Self {
        Self {
            base: ExtractorConfig {
                extractor_type: ExtractorType::Image,
                dimension: 512,
                params: HashMap::new(),
            },
            model_name: "resnet50".to_string(),
            target_size: (224, 224),
            use_pretrained: true,
            model_path: None,
            use_gpu: true,
            normalization_type: "standard".to_string(),
        }
    }
}

/// 音频特征提取器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioExtractorConfig {
    /// 基础配置
    #[serde(flatten)]
    pub base: ExtractorConfig,
    /// 特征类型
    pub feature_type: String,
    /// 采样率
    pub sample_rate: usize,
    /// 帧大小
    pub frame_size: usize,
    /// 帧间距
    pub hop_length: usize,
    /// MFCC系数数量
    pub n_mfcc: Option<usize>,
    /// 梅尔过滤器数量
    pub n_mels: Option<usize>,
    /// 是否使用delta特征
    pub use_delta: bool,
}

impl Default for AudioExtractorConfig {
    fn default() -> Self {
        Self {
            base: ExtractorConfig {
                extractor_type: ExtractorType::Audio,
                dimension: 40,
                params: HashMap::new(),
            },
            feature_type: "mfcc".to_string(),
            sample_rate: 16000,
            frame_size: 1024,
            hop_length: 512,
            n_mfcc: Some(13),
            n_mels: Some(128),
            use_delta: true,
        }
    }
}

/// 视频特征提取器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoExtractorConfig {
    /// 基础配置
    #[serde(flatten)]
    pub base: ExtractorConfig,
    /// 特征类型
    pub feature_type: String,
    /// 帧率
    pub fps: f32,
    /// 是否提取RGB特征
    pub extract_rgb: bool,
    /// 是否提取光流特征
    pub extract_flow: bool,
    /// 是否使用I3D模型
    pub use_i3d: bool,
    /// 是否使用GPU
    pub use_gpu: bool,
    /// 时间池化类型
    pub temporal_pooling: String,
}

impl Default for VideoExtractorConfig {
    fn default() -> Self {
        Self {
            base: ExtractorConfig {
                extractor_type: ExtractorType::Video,
                dimension: 1024,
                params: HashMap::new(),
            },
            feature_type: "rgb".to_string(),
            fps: 25.0,
            extract_rgb: true,
            extract_flow: false,
            use_i3d: true,
            use_gpu: true,
            temporal_pooling: "mean".to_string(),
        }
    }
}

/// 多模态特征提取器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalExtractorConfig {
    /// 基础配置
    #[serde(flatten)]
    pub base: ExtractorConfig,
    /// 模态权重
    pub modality_weights: HashMap<String, f32>,
    /// 是否使用文本特征
    pub use_text: bool,
    /// 是否使用图像特征
    pub use_image: bool,
    /// 是否使用音频特征
    pub use_audio: bool,
    /// 是否使用视频特征
    pub use_video: bool,
    /// 是否使用跨模态特征
    pub use_cross_modal: bool,
    /// 融合方法
    pub fusion_method: String,
}

impl Default for MultimodalExtractorConfig {
    fn default() -> Self {
        let mut weights = HashMap::new();
        weights.insert("text".to_string(), 1.0);
        weights.insert("image".to_string(), 1.0);
        weights.insert("audio".to_string(), 1.0);
        weights.insert("video".to_string(), 1.0);
        
        use crate::data::feature::types::MultiModalExtractorType;
        Self {
            base: ExtractorConfig {
                extractor_type: ExtractorType::MultiModal(MultiModalExtractorType::Fusion),
                dimension: 2048,
                params: HashMap::new(),
            },
            modality_weights: weights,
            use_text: true,
            use_image: true,
            use_audio: false,
            use_video: false,
            use_cross_modal: true,
            fusion_method: "concatenation".to_string(),
        }
    }
} 