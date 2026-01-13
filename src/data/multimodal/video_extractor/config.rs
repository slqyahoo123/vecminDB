//! 视频特征提取器配置模块
//! 
//! 本模块定义了与视频特征提取相关的配置结构体和默认值

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use super::types::*;
use std::fmt;
use std::path::{Path, PathBuf};
use std::default::Default;

/// 视频特征配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoFeatureConfig {
    /// 每秒帧数
    pub fps: usize,
    /// 帧宽度
    pub frame_width: usize,
    /// 帧高度
    pub frame_height: usize,
    /// 要提取的特征类型
    pub feature_types: Vec<VideoFeatureType>,
    /// 时间池化方法
    pub temporal_pooling: TemporalPoolingType,
    /// 空间池化方法
    pub spatial_pooling: SpatialPoolingType,
    /// 是否使用预训练模型
    pub use_pretrained: bool,
    /// 预训练模型名称
    pub pretrained_model: Option<String>,
    /// 最大帧数
    pub max_frames: Option<usize>,
    /// 提取关键帧间隔
    pub keyframe_interval: Option<usize>,
    /// 是否提取音频特征
    pub extract_audio: bool,
    /// 视频解码质量
    pub decode_quality: DecodeQuality,
    /// 归一化方法
    pub normalization: Option<NormalizationType>,
    /// 自定义参数
    pub custom_params: HashMap<String, String>,
    /// 关键帧提取方法
    #[serde(default)]
    pub keyframe_method: KeyframeExtractionMethod,
    /// 自定义特征配置
    #[serde(default)]
    pub custom_features_config: HashMap<String, HashMap<String, String>>,
    /// 并行处理线程数
    #[serde(default = "default_parallel_threads")]
    pub parallel_threads: usize,
    /// 缓存大小（条目数）
    #[serde(default = "default_cache_size")]
    pub cache_size: usize,
    /// 启用内存优化
    #[serde(default)]
    pub memory_optimized: bool,
    /// 模型类型
    #[serde(default)]
    pub model_type: ModelType,
    /// 最大线程数
    #[serde(default = "default_max_threads")]
    pub max_threads: usize,
    /// 是否使用缓存
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
    /// 是否使用GPU
    #[serde(default)]
    pub use_gpu: bool,
    /// 内存限制(MB)
    #[serde(default = "default_memory_limit")]
    pub memory_limit_mb: usize,
    /// 批处理大小
    #[serde(default)]
    pub batch_size: Option<usize>,
    /// 失败时重试次数
    #[serde(default = "default_retry_count")]
    pub retry_count: u8,
    /// 批处理超时时间(秒)
    #[serde(default = "default_batch_timeout")]
    pub batch_timeout_seconds: u32,
    /// 导出格式
    #[serde(default)]
    pub export_format: Option<super::export::ExportFormat>,
}

/// 并行处理线程数默认值
pub fn default_parallel_threads() -> usize {
    std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4)
}

/// 默认缓存大小
pub fn default_cache_size() -> usize {
    1000
}

/// 默认最大线程数
pub fn default_max_threads() -> usize {
    std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4)
}

/// 默认使用缓存
pub fn default_use_cache() -> bool {
    true
}

/// 默认内存限制(MB)
pub fn default_memory_limit() -> usize {
    4096 // 4GB
}

/// 默认重试次数
pub fn default_retry_count() -> u8 {
    3
}

/// 默认批处理超时时间(秒)
pub fn default_batch_timeout() -> u32 {
    3600 // 1小时
}

impl Default for VideoFeatureConfig {
    fn default() -> Self {
        Self {
            fps: 15,
            frame_width: 224,
            frame_height: 224,
            feature_types: vec![VideoFeatureType::RGB],
            temporal_pooling: TemporalPoolingType::Mean,
            spatial_pooling: SpatialPoolingType::GlobalAverage,
            use_pretrained: true,
            pretrained_model: Some("resnet50".to_string()),
            max_frames: Some(100),
            keyframe_interval: Some(10),
            extract_audio: false,
            decode_quality: DecodeQuality::Medium,
            normalization: Some(NormalizationType::MinMax),
            custom_params: HashMap::new(),
            keyframe_method: KeyframeExtractionMethod::FixedInterval,
            custom_features_config: HashMap::new(),
            parallel_threads: default_parallel_threads(),
            cache_size: default_cache_size(),
            memory_optimized: false,
            model_type: ModelType::ResNet50,
            max_threads: default_max_threads(),
            use_cache: default_use_cache(),
            use_gpu: false,
            memory_limit_mb: default_memory_limit(),
            batch_size: Some(10),
            retry_count: default_retry_count(),
            batch_timeout_seconds: default_batch_timeout(),
            export_format: None,
        }
    }
}

/// 配置比较结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigComparisonResult {
    /// 基准配置
    pub baseline_config: VideoFeatureConfig,
    /// 测试配置
    pub test_config: VideoFeatureConfig,
    /// 性能差异百分比
    pub performance_diff_percent: f64,
    /// 质量差异百分比
    pub quality_diff_percent: f64,
    /// 内存使用差异百分比
    pub memory_diff_percent: f64,
    /// 是否推荐使用此配置
    pub recommended: bool,
    /// 推荐原因
    pub recommendation_reason: String,
}

impl VideoFeatureConfig {
    /// 创建新配置
    pub fn new(feature_type: VideoFeatureType) -> Self {
        let mut config = Self::default();
        config.feature_types = vec![feature_type];
        
        // 根据特征类型调整默认配置
        match feature_type {
            VideoFeatureType::RGB => {
                // 默认值已设置
            },
            VideoFeatureType::OpticalFlow => {
                config.normalization = Some(NormalizationType::MinMax);
            },
            VideoFeatureType::Audio => {
                config.frame_width = 96;
                config.frame_height = 64;
                config.model_type = ModelType::VGGish;
            },
            _ => {}
        }
        
        config
    }
    
    /// 验证配置是否有效
    pub fn validate(&self) -> Result<(), ConfigError> {
        // 验证分辨率
        if self.frame_width < 16 || self.frame_width > 4096 {
            return Err(ConfigError::InvalidValue(format!(
                "帧宽度必须在16到4096之间，当前值: {}", self.frame_width
            )));
        }
        
        if self.frame_height < 16 || self.frame_height > 4096 {
            return Err(ConfigError::InvalidValue(format!(
                "帧高度必须在16到4096之间，当前值: {}", self.frame_height
            )));
        }
        
        // 验证帧率
        if self.fps < 1 || self.fps > 120 {
            return Err(ConfigError::InvalidValue(format!(
                "帧率必须在1到120之间，当前值: {}", self.fps
            )));
        }
        
        // 验证特征类型
        if self.feature_types.is_empty() {
            return Err(ConfigError::InvalidValue(
                "必须指定至少一种特征类型".to_string()
            ));
        }
        
        // 验证线程数
        if self.parallel_threads == 0 {
            return Err(ConfigError::InvalidValue(
                "并行线程数必须大于0".to_string()
            ));
        }
        
        // 验证缓存大小
        if self.cache_size > 10000 {
            return Err(ConfigError::InvalidValue(format!(
                "缓存大小不应超过10000，当前值: {}", self.cache_size
            )));
        }
        
        // 验证最大线程数
        if self.max_threads == 0 {
            return Err(ConfigError::InvalidValue(
                "最大线程数必须大于0".to_string()
            ));
        }
        
        // 验证内存限制
        if self.memory_limit_mb < 512 {
            return Err(ConfigError::InvalidValue(format!(
                "内存限制不应低于512MB，当前值: {}MB", self.memory_limit_mb
            )));
        }
        
        // 验证批处理超时时间
        if self.batch_timeout_seconds < 60 {
            return Err(ConfigError::InvalidValue(format!(
                "批处理超时时间不应低于60秒，当前值: {}秒", self.batch_timeout_seconds
            )));
        }
        
        Ok(())
    }
    
    /// 合并两个配置，优先使用other中的非默认值
    pub fn merge(&self, other: &Self) -> Self {
        let default = Self::default();
        
        let is_default_width = other.frame_width == default.frame_width;
        let is_default_height = other.frame_height == default.frame_height;
        let is_default_fps = other.fps == default.fps;
        let is_default_feature_types = other.feature_types == default.feature_types;
        let is_default_temporal = other.temporal_pooling == default.temporal_pooling;
        let is_default_spatial = other.spatial_pooling == default.spatial_pooling;
        let is_default_norm = other.normalization == default.normalization;
        let is_default_threads = other.parallel_threads == default.parallel_threads;
        let is_default_cache = other.cache_size == default.cache_size;
        let is_default_memory = other.memory_optimized == default.memory_optimized;
        let is_default_model = other.model_type == default.model_type;
        let is_default_quality = other.decode_quality == default.decode_quality;
        let is_default_max_threads = other.max_threads == default.max_threads;
        let is_default_use_cache = other.use_cache == default.use_cache;
        let is_default_use_gpu = other.use_gpu == default.use_gpu;
        let is_default_memory_limit = other.memory_limit_mb == default.memory_limit_mb;
        let is_default_batch_size = other.batch_size == default.batch_size;
        let is_default_retry_count = other.retry_count == default.retry_count;
        let is_default_batch_timeout = other.batch_timeout_seconds == default.batch_timeout_seconds;
        
        let mut custom_params = self.custom_params.clone();
        for (k, v) in &other.custom_params {
            custom_params.insert(k.clone(), v.clone());
        }
        
        let mut custom_features_config = self.custom_features_config.clone();
        for (k, v) in &other.custom_features_config {
            custom_features_config.insert(k.clone(), v.clone());
        }
        
        Self {
            frame_width: if is_default_width { self.frame_width } else { other.frame_width },
            frame_height: if is_default_height { self.frame_height } else { other.frame_height },
            fps: if is_default_fps { self.fps } else { other.fps },
            feature_types: if is_default_feature_types { self.feature_types.clone() } else { other.feature_types.clone() },
            temporal_pooling: if is_default_temporal { self.temporal_pooling } else { other.temporal_pooling },
            spatial_pooling: if is_default_spatial { self.spatial_pooling } else { other.spatial_pooling },
            use_pretrained: if other.use_pretrained != default.use_pretrained { other.use_pretrained } else { self.use_pretrained },
            pretrained_model: if other.pretrained_model != default.pretrained_model { other.pretrained_model.clone() } else { self.pretrained_model.clone() },
            max_frames: if other.max_frames != default.max_frames { other.max_frames } else { self.max_frames },
            keyframe_interval: if other.keyframe_interval != default.keyframe_interval { other.keyframe_interval } else { self.keyframe_interval },
            extract_audio: if other.extract_audio != default.extract_audio { other.extract_audio } else { self.extract_audio },
            decode_quality: if !is_default_quality { other.decode_quality } else { self.decode_quality },
            normalization: if is_default_norm { self.normalization } else { other.normalization },
            custom_params,
            keyframe_method: if other.keyframe_method != default.keyframe_method { other.keyframe_method } else { self.keyframe_method },
            custom_features_config,
            parallel_threads: if is_default_threads { self.parallel_threads } else { other.parallel_threads },
            cache_size: if is_default_cache { self.cache_size } else { other.cache_size },
            memory_optimized: if is_default_memory { self.memory_optimized } else { other.memory_optimized },
            model_type: if is_default_model { self.model_type } else { other.model_type },
            max_threads: if is_default_max_threads { self.max_threads } else { other.max_threads },
            use_cache: if is_default_use_cache { self.use_cache } else { other.use_cache },
            use_gpu: if is_default_use_gpu { self.use_gpu } else { other.use_gpu },
            memory_limit_mb: if is_default_memory_limit { self.memory_limit_mb } else { other.memory_limit_mb },
            batch_size: if is_default_batch_size { self.batch_size } else { other.batch_size },
            retry_count: if is_default_retry_count { self.retry_count } else { other.retry_count },
            batch_timeout_seconds: if is_default_batch_timeout { self.batch_timeout_seconds } else { other.batch_timeout_seconds },
            export_format: if other.export_format.is_some() { other.export_format } else { self.export_format },
        }
    }
    
    /// 获取预设配置：高性能模式
    pub fn high_performance() -> Self {
        Self {
            fps: 5,
            frame_width: 128,
            frame_height: 128,
            feature_types: vec![VideoFeatureType::RGB],
            temporal_pooling: TemporalPoolingType::Mean,
            spatial_pooling: SpatialPoolingType::GlobalAverage,
            use_pretrained: true,
            pretrained_model: Some("resnet50".to_string()),
            max_frames: Some(100),
            keyframe_interval: Some(10),
            extract_audio: false,
            decode_quality: DecodeQuality::Low,
            normalization: Some(NormalizationType::MinMax),
            custom_params: HashMap::new(),
            keyframe_method: KeyframeExtractionMethod::FixedInterval,
            custom_features_config: HashMap::new(),
            parallel_threads: num_cpus::get().max(2),
            cache_size: 100,
            memory_optimized: true,
            model_type: ModelType::ResNet50,
            max_threads: default_max_threads(),
            use_cache: default_use_cache(),
            use_gpu: false,
            memory_limit_mb: default_memory_limit(),
            batch_size: Some(10),
            retry_count: default_retry_count(),
            batch_timeout_seconds: default_batch_timeout(),
            export_format: None,
        }
    }
    
    /// 获取预设配置：高质量模式
    pub fn high_quality() -> Self {
        Self {
            fps: 30,
            frame_width: 320,
            frame_height: 320,
            feature_types: vec![VideoFeatureType::RGB, VideoFeatureType::OpticalFlow],
            temporal_pooling: TemporalPoolingType::Attention,
            spatial_pooling: SpatialPoolingType::RegionBased,
            use_pretrained: true,
            pretrained_model: Some("i3d".to_string()),
            max_frames: Some(300),
            keyframe_interval: Some(5),
            extract_audio: true,
            decode_quality: DecodeQuality::High,
            normalization: Some(NormalizationType::ZScore),
            custom_params: HashMap::new(),
            keyframe_method: KeyframeExtractionMethod::SceneChange,
            custom_features_config: HashMap::new(),
            parallel_threads: (num_cpus::get() / 2).max(1),
            cache_size: 500,
            memory_optimized: false,
            model_type: ModelType::I3D,
            max_threads: default_max_threads(),
            use_cache: default_use_cache(),
            use_gpu: false,
            memory_limit_mb: default_memory_limit(),
            batch_size: Some(5),
            retry_count: default_retry_count(),
            batch_timeout_seconds: default_batch_timeout(),
            export_format: None,
        }
    }
    
    /// 获取预设配置：平衡模式
    pub fn balanced() -> Self {
        Self {
            fps: 15,
            frame_width: 224,
            frame_height: 224,
            feature_types: vec![VideoFeatureType::RGB],
            temporal_pooling: TemporalPoolingType::Mean,
            spatial_pooling: SpatialPoolingType::GlobalAverage,
            use_pretrained: true,
            pretrained_model: Some("resnet50".to_string()),
            max_frames: Some(200),
            keyframe_interval: Some(8),
            extract_audio: false,
            decode_quality: DecodeQuality::Medium,
            normalization: Some(NormalizationType::MinMax),
            custom_params: HashMap::new(),
            keyframe_method: KeyframeExtractionMethod::FixedInterval,
            custom_features_config: HashMap::new(),
            parallel_threads: num_cpus::get(),
            cache_size: 1000,
            memory_optimized: false,
            model_type: ModelType::ResNet50,
            max_threads: default_max_threads(),
            use_cache: default_use_cache(),
            use_gpu: false,
            memory_limit_mb: default_memory_limit(),
            batch_size: Some(8),
            retry_count: default_retry_count(),
            batch_timeout_seconds: default_batch_timeout(),
            export_format: None,
        }
    }
    
    /// 基于视频特性自动生成最佳配置
    pub fn auto_config_for_video<P: AsRef<Path>>(video_path: P) -> Result<Self, ConfigError> {
        // 尝试获取视频信息
        let video_info = match extract_video_metadata(video_path.as_ref()) {
            Ok(info) => info,
            Err(_) => return Ok(Self::default()), // 无法读取视频信息，使用默认配置
        };
        
        // 基于视频分辨率选择配置
        let resolution = video_info.width * video_info.height;
        let mut config = if resolution > 1280 * 720 {
            // 高分辨率视频
            Self::high_quality()
        } else if resolution < 480 * 360 {
            // 低分辨率视频
            Self::high_performance()
        } else {
            // 中等分辨率
            Self::balanced()
        };
        
        // 调整帧率，不超过原始视频帧率
        if config.fps > video_info.fps as usize {
            config.fps = video_info.fps as usize;
        }
        
        // 如果视频时长超过5分钟，启用内存优化
        if video_info.duration > 300.0 {
            config.memory_optimized = true;
        }
        
        // 如果视频有音频流，考虑添加音频特征
        if video_info.audio_channels.is_some() {
            let mut feature_types = config.feature_types.clone();
            if !feature_types.contains(&VideoFeatureType::Audio) {
                feature_types.push(VideoFeatureType::Audio);
            }
            config.feature_types = feature_types;
        }
        
        Ok(config)
    }
    
    /// 生成配置哈希值，用于标识配置变化
    pub fn hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        self.frame_width.hash(&mut hasher);
        self.frame_height.hash(&mut hasher);
        self.fps.hash(&mut hasher);
        format!("{:?}", self.feature_types).hash(&mut hasher);
        format!("{:?}", self.temporal_pooling).hash(&mut hasher);
        format!("{:?}", self.spatial_pooling).hash(&mut hasher);
        format!("{:?}", self.normalization).hash(&mut hasher);
        self.parallel_threads.hash(&mut hasher);
        format!("{:?}", self.model_type).hash(&mut hasher);
        format!("{:?}", self.decode_quality).hash(&mut hasher);
        hasher.finish()
    }
    
    /// 转换为键值对格式，方便序列化
    pub fn to_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("frame_width".to_string(), self.frame_width.to_string());
        map.insert("frame_height".to_string(), self.frame_height.to_string());
        map.insert("fps".to_string(), self.fps.to_string());
        map.insert("feature_types".to_string(), format!("{:?}", self.feature_types));
        map.insert("temporal_pooling".to_string(), format!("{:?}", self.temporal_pooling));
        map.insert("spatial_pooling".to_string(), format!("{:?}", self.spatial_pooling));
        map.insert("normalization".to_string(), format!("{:?}", self.normalization));
        map.insert("parallel_threads".to_string(), self.parallel_threads.to_string());
        map.insert("cache_size".to_string(), self.cache_size.to_string());
        map.insert("memory_optimized".to_string(), self.memory_optimized.to_string());
        map.insert("model_type".to_string(), format!("{:?}", self.model_type));
        map.insert("decode_quality".to_string(), format!("{:?}", self.decode_quality));
        map
    }
    
    /// 从键值对格式创建配置
    pub fn from_map(map: &HashMap<String, String>) -> Result<Self, ConfigError> {
        let mut config = Self::default();
        
        if let Some(value) = map.get("frame_width") {
            config.frame_width = value.parse().map_err(|_| 
                ConfigError::ParseError("帧宽度解析失败".to_string()))?;
        }
        
        if let Some(value) = map.get("frame_height") {
            config.frame_height = value.parse().map_err(|_| 
                ConfigError::ParseError("帧高度解析失败".to_string()))?;
        }
        
        if let Some(value) = map.get("fps") {
            config.fps = value.parse().map_err(|_| 
                ConfigError::ParseError("帧率解析失败".to_string()))?;
        }
        
        if let Some(value) = map.get("memory_optimized") {
            config.memory_optimized = value.parse().map_err(|_| 
                ConfigError::ParseError("内存优化标志解析失败".to_string()))?;
        }
        
        if let Some(value) = map.get("cache_size") {
            config.cache_size = value.parse().map_err(|_| 
                ConfigError::ParseError("缓存大小解析失败".to_string()))?;
        }
        
        if let Some(value) = map.get("parallel_threads") {
            config.parallel_threads = value.parse().map_err(|_| 
                ConfigError::ParseError("并行线程数解析失败".to_string()))?;
        }
        
        // 其他字段需要更复杂的解析，这里简化处理
        
        config.validate()?;
        Ok(config)
    }
    
    /// 获取字符串参数
    pub fn get_string_param(&self, key: &str) -> Option<String> {
        self.custom_params.get(key).cloned()
    }
    
    /// 获取布尔参数
    pub fn get_bool_param(&self, key: &str) -> Option<bool> {
        self.custom_params.get(key)
            .map(|v| v == "true" || v == "1" || v == "yes")
    }
    
    /// 获取usize参数
    pub fn get_usize_param(&self, key: &str) -> Option<usize> {
        self.custom_params.get(key)
            .and_then(|v| v.parse().ok())
    }
    
    /// 设置字符串参数
    pub fn set_string_param(&mut self, key: &str, value: String) {
        self.custom_params.insert(key.to_string(), value);
    }
    
    /// 设置布尔参数
    pub fn set_bool_param(&mut self, key: &str, value: bool) {
        self.custom_params.insert(key.to_string(), if value { "true".to_string() } else { "false".to_string() });
    }
    
    /// 设置usize参数
    pub fn set_usize_param(&mut self, key: &str, value: usize) {
        self.custom_params.insert(key.to_string(), value.to_string());
    }
    
    /// 获取浮点数参数
    pub fn get_float_param(&self, key: &str) -> Option<f64> {
        self.custom_params.get(key)
            .and_then(|v| v.parse().ok())
    }
    
    /// 设置浮点数参数
    pub fn set_float_param(&mut self, key: &str, value: f64) {
        self.custom_params.insert(key.to_string(), value.to_string());
    }
    
    /// 生成缓存键
    pub fn to_cache_key(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        self.frame_width.hash(&mut hasher);
        self.frame_height.hash(&mut hasher);
        self.fps.hash(&mut hasher);
        format!("{:?}", self.feature_types).hash(&mut hasher);
        format!("{:?}", self.temporal_pooling).hash(&mut hasher);
        format!("{:?}", self.spatial_pooling).hash(&mut hasher);
        format!("{:?}", self.model_type).hash(&mut hasher);
        format!("{:?}", self.normalization).hash(&mut hasher);
        hasher.finish().to_string()
    }
}

/// 配置错误类型
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("配置值无效: {0}")]
    InvalidValue(String),
    
    #[error("配置解析错误: {0}")]
    ParseError(String),
    
    #[error("配置验证错误: {0}")]
    ValidationError(String),
    
    #[error("配置I/O错误: {0}")]
    IoError(#[from] std::io::Error),
}

// 从字符串转换
impl From<String> for ConfigError {
    fn from(err: String) -> Self {
        ConfigError::ParseError(err)
    }
}

// 配置管理器
pub struct ConfigManager {
    config_path: PathBuf,
    current_config: VideoFeatureConfig,
    default_config: VideoFeatureConfig,
}

impl ConfigManager {
    /// 创建新的配置管理器
    pub fn new<P: AsRef<std::path::Path>>(config_path: P) -> Self {
        Self {
            config_path: config_path.as_ref().to_path_buf(),
            current_config: VideoFeatureConfig::default(),
            default_config: VideoFeatureConfig::default(),
        }
    }
    
    /// 加载配置
    pub fn load(&mut self) -> Result<&VideoFeatureConfig, ConfigError> {
        if !self.config_path.exists() {
            return Ok(&self.current_config);
        }
        
        let config_str = std::fs::read_to_string(&self.config_path)?;
        let config: VideoFeatureConfig = serde_json::from_str(&config_str)
            .map_err(|e| ConfigError::ParseError(format!("JSON解析失败: {}", e)))?;
        
        config.validate()?;
        self.current_config = config;
        Ok(&self.current_config)
    }
    
    /// 保存配置
    pub fn save(&self) -> Result<(), ConfigError> {
        self.current_config.validate()?;
        
        let config_str = serde_json::to_string_pretty(&self.current_config)
            .map_err(|e| ConfigError::ParseError(format!("JSON序列化失败: {}", e)))?;
        
        // 确保目录存在
        if let Some(parent) = self.config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        std::fs::write(&self.config_path, config_str)?;
        Ok(())
    }
    
    /// 获取当前配置
    pub fn get_config(&self) -> &VideoFeatureConfig {
        &self.current_config
    }
    
    /// 获取可变当前配置
    pub fn get_config_mut(&mut self) -> &mut VideoFeatureConfig {
        &mut self.current_config
    }
    
    /// 更新配置
    pub fn update_config(&mut self, config: VideoFeatureConfig) -> Result<(), ConfigError> {
        config.validate()?;
        self.current_config = config;
        Ok(())
    }
    
    /// 更新特定配置项
    pub fn update_config_item<T>(&mut self, key: &str, value: T) -> Result<(), ConfigError> 
    where
        T: ToString + std::fmt::Display
    {
        let value_str = value.to_string();
        
        match key {
            "frame_width" => {
                self.current_config.frame_width = value_str.parse()
                    .map_err(|_| ConfigError::ParseError(format!("无法解析帧宽度: {}", value_str)))?;
            },
            "frame_height" => {
                self.current_config.frame_height = value_str.parse()
                    .map_err(|_| ConfigError::ParseError(format!("无法解析帧高度: {}", value_str)))?;
            },
            "fps" => {
                self.current_config.fps = value_str.parse()
                    .map_err(|_| ConfigError::ParseError(format!("无法解析帧率: {}", value_str)))?;
            },
            "parallel_threads" => {
                self.current_config.parallel_threads = value_str.parse()
                    .map_err(|_| ConfigError::ParseError(format!("无法解析并行线程数: {}", value_str)))?;
            },
            "cache_size" => {
                self.current_config.cache_size = value_str.parse()
                    .map_err(|_| ConfigError::ParseError(format!("无法解析缓存大小: {}", value_str)))?;
            },
            "memory_optimized" => {
                self.current_config.memory_optimized = value_str.parse()
                    .map_err(|_| ConfigError::ParseError(format!("无法解析内存优化设置: {}", value_str)))?;
            },
            "decode_quality" => {
                self.current_config.decode_quality = match value_str.to_lowercase().as_str() {
                    "low" => DecodeQuality::Low,
                    "medium" => DecodeQuality::Medium,
                    "high" => DecodeQuality::High,
                    _ => return Err(ConfigError::InvalidValue(format!("无效的解码质量: {}", value)))
                };
            },
            "model_type" => {
                self.current_config.model_type = match value_str.to_lowercase().as_str() {
                    "standard" | "resnet50" => ModelType::ResNet50,
                    "lightweight" => ModelType::ResNet50,
                    "highprecision" | "high_precision" | "i3d" => ModelType::I3D,
                    "slowfast" => ModelType::SlowFast,
                    "vggish" => ModelType::VGGish,
                    "custom" => ModelType::Custom,
                    _ => return Err(ConfigError::InvalidValue(format!("无效的模型类型: {}", value)))
                };
            },
            _ => return Err(ConfigError::InvalidValue(format!("未知配置项: {}", key))),
        }
        
        self.current_config.validate()?;
        Ok(())
    }
    
    /// 从环境变量加载配置覆盖
    pub fn load_from_env(&mut self, prefix: &str) -> Result<(), ConfigError> {
        // 遍历环境变量
        for (key, value) in std::env::vars() {
            if key.starts_with(prefix) {
                let config_key = key[prefix.len()..].to_lowercase();
                self.try_update_from_env(&config_key, &value)?;
            }
        }
        
        self.current_config.validate()?;
        Ok(())
    }
    
    /// 尝试更新特定配置项
    fn try_update_from_env(&mut self, key: &str, value: &str) -> Result<(), ConfigError> {
        match key {
            "frame_width" => {
                self.current_config.frame_width = value.parse()
                    .map_err(|_| ConfigError::ParseError(format!("无法解析帧宽度: {}", value)))?;
            },
            "frame_height" => {
                self.current_config.frame_height = value.parse()
                    .map_err(|_| ConfigError::ParseError(format!("无法解析帧高度: {}", value)))?;
            },
            "fps" => {
                self.current_config.fps = value.parse()
                    .map_err(|_| ConfigError::ParseError(format!("无法解析帧率: {}", value)))?;
            },
            "parallel_threads" => {
                self.current_config.parallel_threads = value.parse()
                    .map_err(|_| ConfigError::ParseError(format!("无法解析并行线程数: {}", value)))?;
            },
            "cache_size" => {
                self.current_config.cache_size = value.parse()
                    .map_err(|_| ConfigError::ParseError(format!("无法解析缓存大小: {}", value)))?;
            },
            "memory_optimized" => {
                self.current_config.memory_optimized = value.parse()
                    .map_err(|_| ConfigError::ParseError(format!("无法解析内存优化设置: {}", value)))?;
            },
            "decode_quality" => {
                self.current_config.decode_quality = match value.to_lowercase().as_str() {
                    "low" => DecodeQuality::Low,
                    "medium" => DecodeQuality::Medium,
                    "high" => DecodeQuality::High,
                    _ => return Err(ConfigError::InvalidValue(format!("无效的解码质量: {}", value)))
                };
            },
            "model_type" => {
                self.current_config.model_type = match value.to_lowercase().as_str() {
                    "standard" => ModelType::ResNet50,
                    "lightweight" => ModelType::ResNet50,
                    "highprecision" | "high_precision" | "i3d" => ModelType::I3D,
                    _ => return Err(ConfigError::InvalidValue(format!("无效的模型类型: {}", value)))
                };
            },
            _ => {}  // 忽略未知配置项
        }
        
        Ok(())
    }
    
    /// 重置为默认配置
    pub fn reset_to_default(&mut self) {
        self.current_config = self.default_config.clone();
    }
    
    /// 设置为高性能配置
    pub fn set_high_performance(&mut self) {
        self.current_config = VideoFeatureConfig::high_performance();
    }
    
    /// 设置为高质量配置
    pub fn set_high_quality(&mut self) {
        self.current_config = VideoFeatureConfig::high_quality();
    }
    
    /// 设置为平衡配置
    pub fn set_balanced(&mut self) {
        self.current_config = VideoFeatureConfig::default();
    }
    
    /// 为特定视频自动设置配置
    pub fn auto_config_for_video<P: AsRef<std::path::Path>>(&mut self, video_path: P) -> Result<(), ConfigError> {
        let metadata = extract_video_metadata(video_path.as_ref())
            .map_err(|e| ConfigError::ParseError(format!("无法提取视频元数据: {}", e)))?;
        
        // 基于视频分辨率选择配置
        let resolution = metadata.width * metadata.height;
        let mut config = if resolution > 1280 * 720 {
            // 高分辨率视频
            VideoFeatureConfig::high_quality()
        } else if resolution < 480 * 360 {
            // 低分辨率视频
            VideoFeatureConfig::high_performance()
        } else {
            // 中等分辨率
            VideoFeatureConfig::default()
        };
        
        // 调整帧率，不超过原始视频帧率
        if config.fps > metadata.fps as usize {
            config.fps = metadata.fps as usize;
        }
        
        // 如果视频时长超过5分钟，启用内存优化
        if metadata.duration > 300.0 {
            config.memory_optimized = true;
        }
        
        // 如果视频有音频流，考虑添加音频特征
        if metadata.audio_channels.is_some() {
            config.extract_audio = true;
            let mut feature_types = config.feature_types.clone();
            if !feature_types.contains(&VideoFeatureType::Audio) {
                feature_types.push(VideoFeatureType::Audio);
            }
            config.feature_types = feature_types;
        }
        
        self.current_config = config;
        Ok(())
    }
}

fn extract_video_metadata(path: &std::path::Path) -> Result<VideoMetadata, String> {
    // 调用processing模块中的完整实现
    use crate::data::multimodal::video_extractor::processing::extract_video_metadata;
    use crate::data::multimodal::video_extractor::error::VideoExtractionError;
    
    let path_str = path.to_string_lossy().to_string();
    extract_video_metadata(&path_str)
        .map_err(|e| match e {
            VideoExtractionError::FileError(msg) => msg,
            VideoExtractionError::ProcessingError(msg) => msg,
            _ => format!("提取视频元数据失败: {:?}", e),
        })
}

fn simple_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a初始值
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3); // FNV素数
    }
    hash
}

impl fmt::Display for VideoFeatureConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VideoFeatureConfig {{ 类型: {:?}, 分辨率: {}x{}, FPS: {}, 模型: {:?}, 池化: {:?}, GPU: {}, 帧数: {} }}",
            self.feature_types[0],
            self.frame_width,
            self.frame_height,
            self.fps,
            self.model_type,
            self.temporal_pooling,
            self.use_pretrained,
            self.max_frames.unwrap_or(0)
        )
    }
} 