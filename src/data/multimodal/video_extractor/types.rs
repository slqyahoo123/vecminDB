//! 视频特征提取器类型定义模块
//!
//! 本模块定义视频特征提取过程中使用的各种数据类型

use std::fmt;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use super::config::VideoFeatureConfig;

/// 视频特征类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum VideoFeatureType {
    /// RGB特征
    RGB,
    /// 光流特征
    OpticalFlow,
    /// I3D特征
    I3D,
    /// SlowFast特征
    SlowFast,
    /// 音频特征
    Audio,
    /// 通用特征类型
    Generic,
    /// 自定义特征
    Custom(u8),
}

impl VideoFeatureType {
    /// 转换为 u8 值
    pub fn as_u8(self) -> u8 {
        match self {
            VideoFeatureType::RGB => 0,
            VideoFeatureType::OpticalFlow => 1,
            VideoFeatureType::I3D => 2,
            VideoFeatureType::SlowFast => 3,
            VideoFeatureType::Audio => 4,
            VideoFeatureType::Generic => 5,
            VideoFeatureType::Custom(v) => v,
        }
    }
}

/// 视频元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoMetadata {
    /// 视频ID
    pub id: String,
    /// 文件路径
    pub file_path: String,
    /// 文件大小(字节)
    pub file_size: u64,
    /// 视频时长(秒)
    pub duration: f64,
    /// 视频宽度
    pub width: u32,
    /// 视频高度
    pub height: u32,
    /// 视频帧率
    pub fps: f32,
    /// 总帧数
    pub frame_count: u32,
    /// 视频编解码器
    pub codec: String,
    /// 音频编解码器
    pub audio_codec: Option<String>,
    /// 音频采样率
    pub audio_sample_rate: Option<u32>,
    /// 音频通道数
    pub audio_channels: Option<u8>,
    /// 创建时间
    pub created_at: u64,
    /// 自定义元数据
    pub custom_metadata: Option<HashMap<String, String>>,
}

/// 视频特征结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoFeatureResult {
    /// 特征类型
    pub feature_type: VideoFeatureType,
    /// 特征数据
    pub features: Vec<f32>,
    /// 视频元数据
    pub metadata: Option<VideoMetadata>,
    /// 处理信息
    pub processing_info: Option<ProcessingInfo>,
    /// 特征维度
    pub dimensions: usize,
    /// 时间戳
    pub timestamp: u64,
}

impl Default for VideoFeatureResult {
    fn default() -> Self {
        Self {
            feature_type: VideoFeatureType::RGB,
            features: Vec::new(),
            metadata: None,
            processing_info: None,
            dimensions: 0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

/// 视频特征（用于extractor.rs接口）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoFeature {
    /// 特征类型
    pub feature_type: VideoFeatureType,
    /// 特征数据
    pub features: Vec<f32>,
    /// 视频元数据
    pub metadata: Option<VideoMetadata>,
    /// 特征维度
    pub dimensions: usize,
    /// 时间戳
    pub timestamp: u64,
}

/// 处理信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingInfo {
    /// 特征类型
    pub feature_type: VideoFeatureType,
    /// 提取配置
    pub config: VideoFeatureConfig,
    /// 提取时间(毫秒)
    pub extraction_time_ms: u64,
    /// 提取方法
    pub extraction_method: String,
}

/// 时间间隔
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimeInterval {
    /// 开始时间(秒)
    pub start: f64,
    /// 结束时间(秒)
    pub end: f64,
}

/// 提取状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtractionStatus {
    /// 排队中
    Queued,
    /// 处理中
    Processing,
    /// 已完成
    Completed,
    /// 失败
    Failed,
    /// 已取消
    Canceled,
}

/// 处理进度
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingProgress {
    /// 视频ID
    pub video_id: String,
    /// 状态
    pub status: ExtractionStatus,
    /// 进度百分比(0-100)
    pub percentage: f32,
    /// 已处理的帧数
    pub frames_processed: u32,
    /// 总帧数
    pub total_frames: u32,
    /// 已处理的时间(秒)
    pub elapsed_seconds: f64,
    /// 估计剩余时间(秒)
    pub estimated_remaining_seconds: f64,
    /// 当前处理阶段
    pub current_stage: String,
    /// 错误消息(如果有)
    pub error_message: Option<String>,
}

/// 池化方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PoolingMethod {
    /// 平均池化
    Mean,
    /// 最大池化
    Max,
    /// 最小池化
    Min,
    /// 中值池化
    Median,
    /// 无池化(保留所有特征)
    None,
    /// 自适应池化
    Adaptive,
}

/// 模型类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelType {
    /// ResNet50
    ResNet50,
    /// I3D
    I3D,
    /// SlowFast
    SlowFast,
    /// VGGish(音频)
    VGGish,
    /// 自定义模型
    Custom,
}

impl Default for ModelType {
    fn default() -> Self {
        ModelType::ResNet50
    }
}

/// 缩放方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScaleMethod {
    /// 双线性插值
    Bilinear,
    /// 最近邻插值
    Nearest,
    /// 双三次插值
    Bicubic,
    /// Lanczos插值
    Lanczos,
}

/// 资源使用情况
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU使用率(百分比)
    pub cpu_usage_percent: f32,
    /// 内存使用量(MB)
    pub memory_usage_mb: f32,
    /// GPU使用率(百分比,如果可用)
    pub gpu_usage_percent: Option<f32>,
    /// GPU内存使用量(MB,如果可用)
    pub gpu_memory_usage_mb: Option<f32>,
    /// 磁盘读写速度(MB/s)
    pub disk_io_mbps: f32,
    /// 线程数
    pub thread_count: usize,
    /// 采样时间
    pub sample_time: u64,
}

/// 特征提取请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionRequest {
    /// 请求ID
    pub request_id: String,
    /// 视频路径或URL
    pub video_source: String,
    /// 特征类型
    pub feature_type: VideoFeatureType,
    /// 提取配置
    pub config: Option<String>,
    /// 时间间隔(可选)
    pub intervals: Option<Vec<TimeInterval>>,
    /// 优先级(0-10,默认5)
    pub priority: u8,
    /// 回调URL(可选)
    pub callback_url: Option<String>,
    /// 超时时间(秒)
    pub timeout_seconds: u32,
    /// 创建时间
    pub created_at: u64,
}

/// 视频帧结构
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// 帧宽度
    pub width: usize,
    /// 帧高度
    pub height: usize,
    /// 通道数
    pub channels: usize,
    /// 帧数据
    pub data: Vec<u8>,
    /// 时间戳(秒)
    pub timestamp: f64,
}

impl VideoFrame {
    /// 创建新帧
    pub fn new(width: usize, height: usize, channels: usize, timestamp: f64) -> Self {
        Self {
            width,
            height,
            channels,
            data: vec![0; width * height * channels],
            timestamp,
        }
    }
    
    /// 获取像素
    pub fn get_pixel(&self, x: usize, y: usize) -> Option<&[u8]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        
        let idx = (y * self.width + x) * self.channels;
        Some(&self.data[idx..idx + self.channels])
    }
    
    /// 设置像素
    pub fn set_pixel(&mut self, x: usize, y: usize, pixel: &[u8]) -> bool {
        if x >= self.width || y >= self.height || pixel.len() != self.channels {
            return false;
        }
        
        let idx = (y * self.width + x) * self.channels;
        self.data[idx..idx + self.channels].copy_from_slice(pixel);
        true
    }
    
    /// 转换为灰度图
    pub fn to_grayscale(&self) -> Self {
        if self.channels == 1 {
            return self.clone();
        }
        
        let mut gray_frame = Self::new(self.width, self.height, 1, self.timestamp);
        
        for y in 0..self.height {
            for x in 0..self.width {
                if let Some(pixel) = self.get_pixel(x, y) {
                    let gray = match self.channels {
                        3 | 4 => {
                            // RGB或RGBA转灰度公式: 0.299*R + 0.587*G + 0.114*B
                            let r = pixel[0] as f32 * 0.299;
                            let g = pixel[1] as f32 * 0.587;
                            let b = pixel[2] as f32 * 0.114;
                            (r + g + b) as u8
                        },
                        _ => pixel[0],
                    };
                    
                    gray_frame.set_pixel(x, y, &[gray]);
                }
            }
        }
        
        gray_frame
    }
    
    /// 调整大小
    pub fn resize(&self, new_width: usize, new_height: usize) -> Self {
        let mut resized = Self::new(new_width, new_height, self.channels, self.timestamp);
        
        // 简单的最近邻插值
        let x_ratio = self.width as f32 / new_width as f32;
        let y_ratio = self.height as f32 / new_height as f32;
        
        for y in 0..new_height {
            for x in 0..new_width {
                let px = (x as f32 * x_ratio) as usize;
                let py = (y as f32 * y_ratio) as usize;
                
                if let Some(pixel) = self.get_pixel(px, py) {
                    resized.set_pixel(x, y, pixel);
                }
            }
        }
        
        resized
    }
    
    /// 获取帧大小(字节)
    pub fn size_bytes(&self) -> usize {
        self.width * self.height * self.channels
    }
}

/// 解码质量
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecodeQuality {
    /// 低质量(快速)
    Low,
    /// 中等质量(平衡)
    Medium,
    /// 高质量(慢速)
    High,
}

/// 归一化类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormalizationType {
    /// 最小-最大归一化
    MinMax,
    /// Z-Score归一化
    ZScore,
    /// L2归一化
    L2,
    /// 无归一化
    None,
}

/// 时间池化类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemporalPoolingType {
    /// 平均池化
    Mean,
    /// 最大池化
    Max,
    /// 注意力池化
    Attention,
    /// 自定义池化
    Custom(u8),
}

/// 空间池化类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpatialPoolingType {
    /// 全局平均池化
    GlobalAverage,
    /// 全局最大池化
    GlobalMax,
    /// 区域池化
    RegionBased,
    /// 自定义池化
    Custom(u8),
}

/// 关键帧提取方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KeyframeExtractionMethod {
    /// 固定间隔
    FixedInterval,
    /// 场景变化检测
    SceneChange,
    /// 内容重要性
    ContentImportance,
    /// 自适应采样
    AdaptiveSampling,
}

impl Default for KeyframeExtractionMethod {
    fn default() -> Self {
        KeyframeExtractionMethod::FixedInterval
    }
}

/// 输出格式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputFormat {
    /// JSON格式
    JSON,
    /// 二进制格式
    Binary,
    /// HDF5格式
    HDF5,
    /// CSV格式
    CSV,
    /// 自定义格式
    Custom(u8),
}

/// 特征转换
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureTransformation {
    /// 转换类型
    pub transformation_type: TransformationType,
    /// 参数
    pub parameters: HashMap<String, String>,
}

/// 转换类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransformationType {
    /// 降维
    DimensionReduction,
    /// 特征选择
    FeatureSelection,
    /// 归一化
    Normalization,
    /// 特征增强
    FeatureAugmentation,
    /// 自定义转换
    Custom(u8),
}

/// 批处理配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessingConfig {
    /// 批大小
    pub batch_size: usize,
    /// 并行处理线程数
    pub parallel_threads: usize,
    /// 内存限制(MB)
    pub memory_limit_mb: usize,
    /// 超时时间(秒)
    pub timeout_seconds: u32,
    /// 错误处理策略
    pub error_strategy: ErrorStrategy,
    /// 进度回调
    pub progress_callback: Option<String>,
}

/// 错误处理策略
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorStrategy {
    /// 失败时停止
    StopOnFailure,
    /// 继续处理其余项
    ContinueOnFailure,
    /// 重试(指定次数)
    Retry(u8),
    /// 跳过
    Skip,
}

/// 特征聚合策略
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureAggregationStrategy {
    /// 拼接
    Concatenate,
    /// 平均
    Average,
    /// 加权平均
    WeightedAverage,
    /// 投票
    Voting,
    /// 学习聚合
    Learned,
}

/// 多模态特征集
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalFeatureSet {
    /// 视频ID
    pub video_id: String,
    /// 视觉特征
    pub visual_features: Option<Vec<VideoFeatureResult>>,
    /// 音频特征
    pub audio_features: Option<Vec<AudioFeatureResult>>,
    /// 文本特征
    pub text_features: Option<Vec<TextFeatureResult>>,
    /// 聚合特征
    pub aggregated_features: Option<Vec<f32>>,
    /// 聚合策略
    pub aggregation_strategy: Option<FeatureAggregationStrategy>,
    /// 元数据
    pub metadata: Option<VideoMetadata>,
    /// 创建时间
    pub created_at: u64,
}

/// 音频特征结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFeatureResult {
    /// 音频ID
    pub audio_id: String,
    /// 特征类型
    pub feature_type: String,
    /// 特征数据
    pub features: Vec<f32>,
    /// 采样率
    pub sample_rate: u32,
    /// 处理时间(毫秒)
    pub processing_time_ms: u64,
    /// 创建时间
    pub created_at: u64,
}

/// 文本特征结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextFeatureResult {
    /// 文本ID
    pub text_id: String,
    /// 特征类型
    pub feature_type: String,
    /// 特征数据
    pub features: Vec<f32>,
    /// 原始文本
    pub text: String,
    /// 处理时间(毫秒)
    pub processing_time_ms: u64,
    /// 创建时间
    pub created_at: u64,
}

/// 提取作业
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionJob {
    /// 作业ID
    pub job_id: String,
    /// 视频源
    pub video_source: String,
    /// 特征类型
    pub feature_types: Vec<VideoFeatureType>,
    /// 配置
    pub config: HashMap<String, String>,
    /// 状态
    pub status: ExtractionStatus,
    /// 进度
    pub progress: f32,
    /// 创建时间
    pub created_at: u64,
    /// 开始时间
    pub started_at: Option<u64>,
    /// 完成时间
    pub completed_at: Option<u64>,
    /// 错误消息
    pub error_message: Option<String>,
    /// 结果ID
    pub result_ids: Vec<String>,
}

impl Default for VideoFrame {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            channels: 0,
            data: Vec::new(),
            timestamp: 0.0,
        }
    }
}

impl fmt::Display for VideoFeatureType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VideoFeatureType::RGB => write!(f, "RGB"),
            VideoFeatureType::OpticalFlow => write!(f, "OpticalFlow"),
            VideoFeatureType::I3D => write!(f, "I3D"),
            VideoFeatureType::SlowFast => write!(f, "SlowFast"),
            VideoFeatureType::Audio => write!(f, "Audio"),
            VideoFeatureType::Generic => write!(f, "Generic"),
            VideoFeatureType::Custom(id) => write!(f, "Custom({})", id),
        }
    }
}

impl fmt::Display for PoolingMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PoolingMethod::Mean => write!(f, "Mean"),
            PoolingMethod::Max => write!(f, "Max"),
            PoolingMethod::Min => write!(f, "Min"),
            PoolingMethod::Median => write!(f, "Median"),
            PoolingMethod::None => write!(f, "None"),
            PoolingMethod::Adaptive => write!(f, "Adaptive"),
        }
    }
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelType::ResNet50 => write!(f, "ResNet50"),
            ModelType::I3D => write!(f, "I3D"),
            ModelType::SlowFast => write!(f, "SlowFast"),
            ModelType::VGGish => write!(f, "VGGish"),
            ModelType::Custom => write!(f, "Custom"),
        }
    }
}

impl fmt::Display for ExtractionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExtractionStatus::Queued => write!(f, "排队中"),
            ExtractionStatus::Processing => write!(f, "处理中"),
            ExtractionStatus::Completed => write!(f, "已完成"),
            ExtractionStatus::Failed => write!(f, "失败"),
            ExtractionStatus::Canceled => write!(f, "已取消"),
        }
    }
}

impl fmt::Display for DecodeQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeQuality::Low => write!(f, "Low"),
            DecodeQuality::Medium => write!(f, "Medium"),
            DecodeQuality::High => write!(f, "High"),
        }
    }
}

impl fmt::Display for NormalizationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NormalizationType::MinMax => write!(f, "MinMax"),
            NormalizationType::ZScore => write!(f, "ZScore"),
            NormalizationType::L2 => write!(f, "L2"),
            NormalizationType::None => write!(f, "None"),
        }
    }
}

impl fmt::Display for TemporalPoolingType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TemporalPoolingType::Mean => write!(f, "Mean"),
            TemporalPoolingType::Max => write!(f, "Max"),
            TemporalPoolingType::Attention => write!(f, "Attention"),
            TemporalPoolingType::Custom(id) => write!(f, "Custom({})", id),
        }
    }
}

impl fmt::Display for SpatialPoolingType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpatialPoolingType::GlobalAverage => write!(f, "GlobalAverage"),
            SpatialPoolingType::GlobalMax => write!(f, "GlobalMax"),
            SpatialPoolingType::RegionBased => write!(f, "RegionBased"),
            SpatialPoolingType::Custom(id) => write!(f, "Custom({})", id),
        }
    }
} 