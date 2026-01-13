//! 视频处理相关类型定义

use std::path::PathBuf;

/// 关键帧信息
#[derive(Debug, Clone)]
pub struct KeyframeInfo {
    /// 时间戳（秒）
    pub timestamp: f64,
    /// 在视频中的位置（字节偏移）
    pub position: u64,
    /// 帧类型（I/P/B）
    pub frame_type: String,
    /// 帧大小（字节）
    pub frame_size: usize,
    /// 帧索引
    pub frame_index: usize,
}

/// 场景变化信息
#[derive(Debug, Clone)]
pub struct SceneChange {
    /// 开始时间戳（秒）
    pub start_timestamp: f64,
    /// 结束时间戳（秒）
    pub end_timestamp: f64,
    /// 变化置信度 (0.0-1.0)
    pub confidence: f32,
    /// 对应的帧索引
    pub frame_index: usize,
    /// 场景描述（可选）
    pub description: Option<String>,
}

/// 缩略图信息
#[derive(Debug, Clone)]
pub struct Thumbnail {
    /// 时间戳（秒）
    pub timestamp: f64,
    /// 缩略图文件路径
    pub file_path: PathBuf,
    /// 宽度（像素）
    pub width: usize,
    /// 高度（像素）
    pub height: usize,
    /// 图像质量 (1-100)
    pub quality: usize,
    /// 图像数据（可选）
    pub data: Option<Vec<u8>>,
} 