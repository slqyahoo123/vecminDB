//! 视频处理模块
//! 
//! 提供视频帧处理和特征提取的核心功能

mod feature_extractor;
mod rgb;
mod optical_flow;
pub mod frames;
mod pooling;
mod batch;
pub mod metadata;
mod types;

// 重新导出所有公共组件
pub use feature_extractor::FeatureExtractor;
pub use rgb::RGBFeatureExtractor;
pub use optical_flow::OpticalFlowExtractor;
pub use frames::{preprocess_frame, simulate_video_frames, simulate_interval_frames};
pub use pooling::{temporal_mean_pooling, temporal_max_pooling, spatial_average_pooling, spatial_max_pooling};
pub use batch::{extract_features_batch, extract_features_batch_with_progress};
pub use metadata::extract_video_metadata;
pub use types::{KeyframeInfo, SceneChange, Thumbnail};

use crate::data::multimodal::video_extractor::types::*;
use crate::data::multimodal::video_extractor::config::VideoFeatureConfig;
use crate::data::multimodal::video_extractor::error::VideoExtractionError;
use crate::data::multimodal::video_extractor::util;
use std::collections::HashMap;
use std::time::Instant;
use log::{info, warn, debug};

/// 从文件提取特征
pub fn extract_features_from_file(
    video_path: &str,
    config: &VideoFeatureConfig
) -> Result<VideoFeatureResult, VideoExtractionError> {
    info!("从文件提取特征: {}", video_path);
    let start_time = Instant::now();
    
    // 提取视频元数据
    let metadata = extract_video_metadata(video_path)?;
    
    // 根据特征类型选择提取器（使用第一个特征类型）
    let feature_type = config.feature_types.first().copied().unwrap_or(VideoFeatureType::RGB);
    let mut extractor: Box<dyn FeatureExtractor> = match feature_type {
        VideoFeatureType::RGB => Box::new(RGBFeatureExtractor::new(config)),
        VideoFeatureType::OpticalFlow => Box::new(OpticalFlowExtractor::new(config)),
        _ => return Err(VideoExtractionError::NotImplementedError(
            format!("特征类型不支持: {:?}", feature_type)
        )),
    };
    
    // 初始化提取器
    extractor.initialize()?;
    
    // 生成模拟帧（实际应用中应该从视频中读取真实帧）
    let frame_count = config.max_frames.unwrap_or(30);
    let frames = simulate_video_frames(
        frame_count,
        config.frame_width,
        config.frame_height
    )?;
    
    // 处理帧并提取特征
    let frame_features = frames.iter().map(|frame| {
        extractor.process_frame(frame)
    }).collect::<Result<Vec<_>, _>>()?;
    
    // 时间池化
    let pooled_features = extractor.temporal_pooling(&frame_features, &config.temporal_pooling)?;
    
    // 计算处理时间
    let processing_time_ms = start_time.elapsed().as_millis() as u64;
    
    // 提取结束，释放资源
    extractor.release()?;
    
    // 创建结果
    let result = VideoFeatureResult {
        feature_type,
        features: pooled_features.clone(),
        metadata: Some(metadata),
        processing_info: Some(ProcessingInfo {
            feature_type,
            config: config.clone(),
            extraction_time_ms: processing_time_ms,
            extraction_method: "extract_features_from_file".to_string(),
        }),
        dimensions: pooled_features.len(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    };
    
    info!("特征提取完成，耗时: {}ms, 特征维度: {}", processing_time_ms, result.dimensions);
    
    Ok(result)
}

/// 按时间间隔提取特征
pub fn extract_features_by_intervals(
    video_path: &str,
    intervals: &[TimeInterval],
    config: &VideoFeatureConfig
) -> Result<HashMap<String, VideoFeatureResult>, VideoExtractionError> {
    info!("按时间间隔提取特征: {}, 区间数: {}", video_path, intervals.len());
    
    // 提取视频元数据
    let metadata = extract_video_metadata(video_path)?;
    
    // 根据特征类型选择提取器（使用第一个特征类型）
    let feature_type = config.feature_types.first().copied().unwrap_or(VideoFeatureType::RGB);
    let mut extractor: Box<dyn FeatureExtractor> = match feature_type {
        VideoFeatureType::RGB => Box::new(RGBFeatureExtractor::new(config)),
        VideoFeatureType::OpticalFlow => Box::new(OpticalFlowExtractor::new(config)),
        _ => return Err(VideoExtractionError::NotImplementedError(
            format!("特征类型不支持: {:?}", feature_type)
        )),
    };
    
    // 初始化提取器
    extractor.initialize()?;
    
    let mut results: HashMap<String, VideoFeatureResult> = HashMap::new();
    
    // 处理每个时间间隔
    for interval in intervals {
        let start_time = Instant::now();
        
        // 检查间隔有效性
        if interval.start < 0.0 || interval.end > metadata.duration || interval.start >= interval.end {
            warn!("无效的时间间隔: {:?}", interval);
            continue;
        }
        
        // 生成当前间隔的帧
        let frame_count = config.max_frames.unwrap_or(30);
        let frames = frames::simulate_interval_frames(
            frame_count,
            config.frame_width,
            config.frame_height,
            interval.start,
            interval.end
        )?;
        
        // 处理帧并提取特征
        let frame_features = frames.iter().map(|frame| {
            extractor.process_frame(frame)
        }).collect::<Result<Vec<_>, _>>()?;
        
        // 时间池化
        let pooled_features = extractor.temporal_pooling(&frame_features, &config.temporal_pooling)?;
        
        // 计算处理时间
        let processing_time_ms = start_time.elapsed().as_millis() as u64;
        
        // 为当前间隔创建间隔ID
        let interval_id = format!("{}_{:.2}_{:.2}", metadata.id, interval.start, interval.end);
        
        // 创建结果
        let result = VideoFeatureResult {
            feature_type,
            features: pooled_features.clone(),
            metadata: Some(metadata.clone()),
            processing_info: Some(ProcessingInfo {
                feature_type,
                config: config.clone(),
                extraction_time_ms: processing_time_ms,
                extraction_method: format!("extract_interval_{}_{}", interval.start, interval.end),
            }),
            dimensions: pooled_features.len(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        
        results.insert(interval_id, result);
        
        debug!("间隔 [{:.2}-{:.2}] 处理完成，耗时: {}ms", interval.start, interval.end, processing_time_ms);
    }
    
    // 提取结束，释放资源
    extractor.release()?;
    
    info!("所有时间间隔处理完成，共 {} 个间隔", results.len());
    
    Ok(results)
}

/// 生成视频ID
pub fn generate_video_id(video_path: &str) -> String {
    let path = std::path::Path::new(video_path);
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");
    
    // 使用文件名和路径的哈希值组合生成ID
    let path_hash = util::hash_str(video_path);
    format!("{}_{:x}", file_name, path_hash)
} 