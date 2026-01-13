//! # 视频特征提取器
//! 
//! 本模块提供了从视频中提取各种特征的功能，支持多种特征类型、并行处理和性能优化。
//! 主要功能包括：
//! 
//! - **多种特征类型提取**：支持RGB、光流、深度特征、场景分类、对象检测等
//! - **批量处理**：高效处理大量视频，支持自动资源优化
//! - **错误处理与恢复**：提供完善的错误诊断和故障转移机制
//! - **性能基准测试**：评估不同配置和特征类型的性能表现
//! - **缓存管理**：优化内存使用和处理速度
//! - **关键帧提取**：支持多种关键帧选择策略
//! - **自适应配置**：根据视频特性和系统资源自动优化参数
//! 
//! ## 快速开始
//! 
//! ```rust
//! // 创建并配置特征提取器
//! let mut config = VideoFeatureConfig::default();
//! config.feature_types = vec![VideoFeatureType::RGB, VideoFeatureType::DeepFeatures];
//! let mut extractor = VideoFeatureExtractor::new(config).unwrap();
//! 
//! // 提取特征
//! let features = extractor.extract_from_file("video.mp4", None).unwrap();
//! println!("特征维度: {}", features.features.len());
//! 
//! // 批量处理多个视频
//! let videos = vec!["video1.mp4", "video2.mp4", "video3.mp4"];
//! let results = extractor.batch_extract_from_files(&videos, None).unwrap();
//! ```
//! 
//! ## 错误处理与恢复
//! 
//! 模块提供了完善的错误处理机制，支持自动重试和故障转移：
//! 
//! ```rust
//! // 安全提取(带重试)
//! let result = extractor.extract_safely_from_file("video.mp4", None);
//! 
//! // 故障转移(质量逐级降级)
//! let result = extractor.extract_with_fallback("video.mp4", None);
//! 
//! // 错误诊断
//! if let Ok(diagnostics) = extractor.diagnose_extraction_error("video.mp4") {
//!     println!("诊断信息: {:?}", diagnostics);
//! }
//! ```
//! 
//! ## 性能优化
//! 
//! 模块提供多种性能优化功能：
//! 
//! ```rust
//! // 运行性能基准测试
//! let benchmarks = extractor.run_benchmark(&["video1.mp4"], None, Some(3));
//! 
//! // 比较不同配置的性能
//! let configs = vec![config1, config2, config3];
//! let results = extractor.compare_configurations(&["test.mp4"], &configs, VideoFeatureType::RGB);
//! 
//! // 获取推荐配置
//! let optimal_config = extractor.generate_config_recommendation("sample.mp4", "speed");
//! extractor.config = optimal_config;
//! ```
//! 
//! ## 批量处理优化
//! 
//! 对于大量视频处理，模块提供了批处理优化功能：
//! 
//! ```rust
//! // 创建优化的批处理计划
//! let plan = extractor.optimize_batch_processing(&videos_paths);
//! 
//! // 执行批处理计划并显示进度
//! extractor.execute_batch_plan(&plan, |processed, total| {
//!     println!("进度: {}%", (processed * 100) / total);
//! });
//! ```

// 导入核心依赖
use crate::{Error, Result};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
// current_timestamp 提供于 util 模块，无需在此直接导入 SystemTime/UNIX_EPOCH
use rayon::prelude::*;
// 修复log导入，将外部crate重命名为logging
extern crate log as logging;
use logging::{info, warn};

// 导出子模块
pub mod config;
pub mod types;
pub mod extractor;
mod error;
mod benchmark;
mod util;
mod cache;
mod batch;
mod processing;
#[path = "log.rs"]
pub mod video_log;  // 将内部log模块重命名为video_log避免冲突
mod export;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod integration_test;

// 导入所需的Duration和Instant类型
use std::time::{Duration, Instant};

// 从子模块重新导出公共组件
pub use config::*;
pub use types::*;
pub use extractor::{VideoFeatureExtractor as ExtractorTrait, ExtractorFactory, CompositeExtractor};
pub use error::*;
pub use benchmark::*;
pub use util::*;
pub use batch::*;
pub use processing::*;
pub use export::*;
pub use cache::{FeatureCache, CacheStats};

/// 视频特征提取器
pub struct VideoFeatureExtractor {
    /// 配置
    config: VideoFeatureConfig,
    /// 特征缓存
    cache: FeatureCache,
    /// 性能基准记录
    benchmarks: Vec<PerformanceBenchmark>,
    /// 最后一次提取时间
    last_extraction_time: Option<Duration>,
    /// 已提取的视频数量
    processed_videos_count: usize,
    /// 总处理字节数
    total_bytes_processed: u64,
    /// 最后错误
    last_error: Option<VideoExtractionError>,
    /// 资源使用情况
    resource_usage: ResourceUsage,
}

impl VideoFeatureExtractor {
    /// 创建新的视频特征提取器
    pub fn new(config: VideoFeatureConfig) -> Result<Self> {
        // 验证配置
        config.validate().map_err(|e| {
            Error::Data(format!("配置验证失败: {}", e))
        })?;
        
        Ok(Self {
            config,
            cache: FeatureCache::new(1000), // 默认缓存大小
            benchmarks: Vec::new(),
            last_extraction_time: None,
            processed_videos_count: 0,
            total_bytes_processed: 0,
            last_error: None,
            resource_usage: ResourceUsage {
                cpu_usage_percent: 0.0,
                memory_usage_mb: 0.0,
                gpu_usage_percent: None,
                gpu_memory_usage_mb: None,
                disk_io_mbps: 0.0,
                thread_count: 0,
                sample_time: current_timestamp(),
            },
        })
    }
    
    /// 提取视频特征
    pub fn extract_features(&mut self, video_path: &str) -> Result<VideoFeatureResult> {
        // 记录开始时间
        let start_time = Instant::now();
        
        // 检查文件是否存在
        if !util::check_video_file(video_path) {
            let error = VideoExtractionError::FileError(format!("视频文件不存在或无法访问: {}", video_path));
            self.last_error = Some(error.clone());
            return Err(Error::Data(format!("{}", error)));
        }
        
        // 尝试从缓存获取
        let video_id = processing::generate_video_id(video_path);
        let cache_key = format!("{}-{}", video_id, self.config.to_cache_key());
        
        if let Some(cached_result) = self.cache.get(&cache_key) {
            info!("从缓存中获取到视频特征: {}", video_path);
            return Ok(cached_result);
        }
        
        // 提取特征
        info!("开始提取视频特征: {}", video_path);
        let result = processing::extract_features_from_file(video_path, &self.config)
            .map_err(|e| {
                self.last_error = Some(e.clone());
                Error::Data(format!("{}", e))
            })?;
        
        // 更新统计信息
        self.processed_videos_count += 1;
        self.last_extraction_time = Some(start_time.elapsed());
        
        if let Some(metadata) = &result.metadata {
            self.total_bytes_processed += metadata.file_size;
        }
        
        // 添加到缓存
        if self.config.use_cache {
            self.cache.put(&cache_key, result.clone());
        }
        
        // 记录资源使用情况
        self.update_resource_usage();
        
        // 返回结果
        Ok(result)
    }
    
    /// 批量提取视频特征
    pub fn extract_features_batch(&mut self, video_paths: &[String]) -> Result<Vec<Result<VideoFeatureResult>>> {
        let start_time = Instant::now();
        info!("开始批量提取视频特征，视频数量: {}", video_paths.len());
        
        // 创建批处理计划
        let plan = BatchProcessingPlan::new(
            video_paths.iter().map(|p| processing::generate_video_id(p)).collect(),
            self.config.batch_size.unwrap_or(10),
            None
        );
        
        let mut results = Vec::with_capacity(video_paths.len());
        
        // 执行批处理
        for batch_idx in 0..plan.get_batch_count() {
            let batch_video_ids = plan.get_batch_videos(batch_idx);
            let batch_paths: Vec<&String> = batch_video_ids.iter()
                .filter_map(|id| video_paths.iter().find(|p| processing::generate_video_id(p) == *id))
                .collect();
            
            info!("处理批次 {}/{}，视频数量: {}", batch_idx + 1, plan.get_batch_count(), batch_paths.len());
            
            // 使用rayon并行处理批次中的视频，而不是简化的顺序处理
            let batch_results: Vec<Result<VideoFeatureResult>> = batch_paths
                .par_iter()
                .map(|video_path| {
                    // 为每个视频创建独立的提取上下文以确保线程安全
                    let mut local_context = self.clone();
                    local_context.extract_features(video_path)
                })
                .collect();
            
            // 合并结果
            results.extend(batch_results);
        }
        
        let elapsed = start_time.elapsed();
        info!("批量提取完成，总耗时: {:?}，成功: {}，失败: {}", 
            elapsed,
            results.iter().filter(|r| r.is_ok()).count(),
            results.iter().filter(|r| r.is_err()).count()
        );
        
        // 创建性能基准记录
        self.record_benchmark(video_paths.len(), elapsed);
        
        Ok(results)
    }
    
    /// 提取指定时间段的视频特征
    pub fn extract_features_by_interval(&mut self, video_path: &str, intervals: &[TimeInterval]) -> Result<HashMap<String, VideoFeatureResult>> {
        let start_time = Instant::now();
        
        // 检查文件是否存在
        if !util::check_video_file(video_path) {
            let error = VideoExtractionError::FileError(format!("视频文件不存在或无法访问: {}", video_path));
            self.last_error = Some(error.clone());
            return Err(Error::Data(format!("{}", error)));
        }
        
        // 获取视频元数据
        let metadata = processing::extract_video_metadata(video_path).map_err(|e| {
            self.last_error = Some(e.clone());
            Error::Data(format!("{}", e))
        })?;
        
        let mut results = HashMap::new();
        
        // 处理每个时间段
        for interval in intervals {
            if interval.start >= interval.end || interval.start < 0.0 || interval.end > metadata.duration {
                warn!("无效的时间段: {:?}，跳过", interval);
                continue;
            }
            
            // 构建此时间段的缓存键
            let cache_key = format!("{}-{}-{}-{}", 
                metadata.id, 
                self.config.to_cache_key(),
                interval.start,
                interval.end
            );
            
            // 尝试从缓存获取
            let interval_key = format!("{}-{}", interval.start, interval.end);
            if let Some(cached_result) = self.cache.get(&cache_key) {
                results.insert(interval_key.clone(), cached_result);
                continue;
            }
            
            // 直接处理指定时间段的帧，而不是处理整个视频
            let interval_result = self.extract_interval_features(video_path, *interval, &metadata)?;
            
            // 添加到缓存和结果
            if self.config.use_cache {
                self.cache.put(&cache_key, interval_result.clone());
            }
            
            results.insert(interval_key, interval_result);
        }
        
        let elapsed = start_time.elapsed();
        info!("时间段提取完成，总耗时: {:?}，处理的时间段数量: {}", elapsed, results.len());
        
        Ok(results)
    }
    
    /// 提取单个时间段的视频特征
    fn extract_interval_features(&mut self, video_path: &str, interval: TimeInterval, metadata: &VideoMetadata) -> Result<VideoFeatureResult> {
        info!("提取时间段 [{:.2}-{:.2}] 的特征: {}", interval.start, interval.end, video_path);
        
        // 计算此间隔内的帧数
        let interval_duration = interval.end - interval.start;
        let frame_count = (interval_duration * self.config.fps as f64) as usize;
        let frame_count = frame_count.min(self.config.max_frames.unwrap_or(300));
        
        if frame_count == 0 {
            warn!("时间段过短，无法提取帧: {:?}", interval);
            return Err(Error::Data(format!("时间段过短，无法提取帧: {:?}", interval)));
        }
        
        // 根据特征类型选择提取器
        let feature_type = self.config.feature_types.first().ok_or_else(|| {
            Error::Data(format!("{}", VideoExtractionError::ConfigError("未指定特征类型".to_string())))
        })?;
        
        // 初始化提取器
        let mut extractor: Box<dyn processing::FeatureExtractor> = match feature_type {
            VideoFeatureType::RGB => Box::new(processing::RGBFeatureExtractor::new(&self.config)),
            VideoFeatureType::OpticalFlow => Box::new(processing::OpticalFlowExtractor::new(&self.config)),
            VideoFeatureType::I3D => return Err(Error::Data(format!("{}", VideoExtractionError::NotImplementedError(
                "I3D特征提取尚未实现".to_string()
            )))),
            VideoFeatureType::SlowFast => return Err(Error::Data(format!("{}", VideoExtractionError::NotImplementedError(
                "SlowFast特征提取尚未实现".to_string()
            )))),
            VideoFeatureType::Audio => return Err(Error::Data(format!("{}", VideoExtractionError::NotImplementedError(
                "音频特征提取尚未实现".to_string()
            )))),
            VideoFeatureType::Custom(_) => return Err(Error::Data(format!("{}", VideoExtractionError::NotImplementedError(
                "自定义特征提取尚未实现".to_string()
            )))),
        };
        
        // 初始化提取器
        extractor.initialize().map_err(|e| Error::Data(format!("{}", e)))?;
        
        // 使用ffmpeg直接提取指定时间段的帧
        #[cfg(feature = "ffmpeg")]
        {
            use std::path::Path;
            use ffmpeg_next as ffmpeg;
            
            // 初始化ffmpeg
            ffmpeg::init().map_err(|e| {
                Error::Data(format!("{}", VideoExtractionError::CodecError(format!("初始化ffmpeg失败: {}", e))))
            })?;
            
            // 打开输入文件
            let mut input_context = ffmpeg::format::input(&Path::new(video_path)).map_err(|e| {
                Error::Data(format!("{}", VideoExtractionError::CodecError(format!("打开视频文件失败: {}", e))))
            })?;
            
            // 查找视频流
            let video_stream_index = input_context.streams()
                .best(ffmpeg::media::Type::Video)
                .ok_or_else(|| {
                    Error::Data(format!("{}", VideoExtractionError::CodecError("无法找到视频流".to_string())))
                })?
                .index();
            
            // 创建解码器
            let context = input_context.stream(video_stream_index)
                .ok_or_else(|| {
                    Error::Data(format!("{}", VideoExtractionError::CodecError("无法获取视频流".to_string())))
                })?
                .codec();
            
            let mut decoder = context.decoder().video().map_err(|e| {
                Error::Data(format!("{}", VideoExtractionError::CodecError(format!("无法创建视频解码器: {}", e))))
            })?;
            
            // 计算时间戳
            let time_base = input_context.stream(video_stream_index)
                .unwrap()
                .time_base();
            
            let start_ts = (interval.start * time_base.den as f64 / time_base.num as f64) as i64;
            let duration_ts = (interval_duration * time_base.den as f64 / time_base.num as f64) as i64;
            
            // 设置读取位置
            input_context.seek(start_ts, 0).map_err(|e| {
                Error::Data(format!("{}", VideoExtractionError::CodecError(format!("无法定位到指定时间点: {}", e))))
            })?;
            
            // 读取和处理帧
            let frame_width = self.config.frame_width.unwrap_or(224);
            let frame_height = self.config.frame_height.unwrap_or(224);
            let mut frames = Vec::with_capacity(frame_count);
            let mut packet = ffmpeg::packet::Packet::empty();
            let end_ts = start_ts + duration_ts;
            
            while frames.len() < frame_count && input_context.read(&mut packet).is_ok() {
                // 检查是否到达时间段末尾
                if packet.pts().unwrap_or(0) > end_ts {
                    break;
                }
                
                // 只处理视频流
                if packet.stream() != video_stream_index {
                    continue;
                }
                
                // 解码帧
                decoder.send_packet(&packet).map_err(|e| {
                    Error::Data(format!("{}", VideoExtractionError::CodecError(format!("发送数据包到解码器失败: {}", e))))
                })?;
                
                let mut decoded = ffmpeg::frame::Video::empty();
                while decoder.receive_frame(&mut decoded).is_ok() {
                    // 转换帧格式并调整大小
                    let mut rgb_frame = ffmpeg::frame::Video::empty();
                    let mut scaler = ffmpeg::software::scaling::context::Context::get(
                        decoder.format(),
                        decoded.width(),
                        decoded.height(),
                        ffmpeg::format::Pixel::RGB24,
                        frame_width as u32,
                        frame_height as u32,
                        ffmpeg::software::scaling::flag::Flags::BILINEAR,
                    ).map_err(|e| {
                        Error::Data(format!("{}", VideoExtractionError::CodecError(format!("创建缩放器失败: {}", e))))
                    })?;
                    
                    scaler.run(&decoded, &mut rgb_frame).map_err(|e| {
                        Error::Data(format!("{}", VideoExtractionError::CodecError(format!("缩放帧失败: {}", e))))
                    })?;
                    
                    // 将帧转换为我们的内部格式
                    let frame_data = rgb_frame.data(0).to_vec();
                    let video_frame = VideoFrame {
                        width: frame_width,
                        height: frame_height,
                        channels: 3,
                        data: frame_data,
                        format: PixelFormat::RGB,
                        timestamp: interval.start + frames.len() as f64 * (interval_duration / frame_count as f64),
                    };
                    
                    frames.push(video_frame);
                    
                    // 如果已经收集了足够的帧，就停止
                    if frames.len() >= frame_count {
                        break;
                    }
                }
            }
            
            // 调用静态方法，使用Self::方法名
            Self::process_frames_and_extract_features(frames, extractor, metadata, interval, &self.config)
        }
        
        #[cfg(not(feature = "ffmpeg"))]
        {
            // 如果没有ffmpeg支持，则使用模拟帧
            let frames = processing::frames::simulate_interval_frames(
                frame_count,
                self.config.frame_width,
                self.config.frame_height,
                interval.start,
                interval.end
            ).map_err(|e| Error::Data(format!("{}", e)))?;
            
            Self::process_frames_and_extract_features(frames, extractor, metadata, interval, &self.config)
        }
    }
    
    /// 处理视频帧并提取特征
    fn process_frames_and_extract_features(
        frames: Vec<VideoFrame>,
        mut extractor: Box<dyn processing::FeatureExtractor>,
        metadata: &VideoMetadata,
        interval: TimeInterval,
        config: &VideoFeatureConfig
    ) -> Result<VideoFeatureResult> {
        let start_time = Instant::now();
        
        // 如果没有足够的帧，则返回错误
        if frames.is_empty() {
            return Err(Error::Data(format!("{}", VideoExtractionError::ProcessingError(
                format!("时间段 [{:.2}-{:.2}] 未提取到任何帧", interval.start, interval.end)
            ))));
        }
        
        info!("处理时间段 [{:.2}-{:.2}] 的 {} 帧", interval.start, interval.end, frames.len());
        
        // 处理每个帧并提取特征
        let frame_features = frames.iter().map(|frame| {
            extractor.process_frame(frame)
        }).collect::<Result<Vec<_>, _>>().map_err(|e| Error::Data(format!("{}", e)))?;
        
        // 时间池化
        let temporal_pooling_type = config.temporal_pooling;
        let pooled_features = extractor.temporal_pooling(&frame_features, &temporal_pooling_type)
            .map_err(|e| Error::Data(format!("{}", e)))?;
        
        // 计算处理时间
        let processing_time_ms = start_time.elapsed().as_millis() as u64;
        
        // 生成结果
        let feature_type = *config.feature_types.first().unwrap_or(&VideoFeatureType::Generic);
        let result = VideoFeatureResult {
            feature_type,
            features: pooled_features,
            metadata: Some(metadata.clone()),
            processing_info: Some(ProcessingInfo {
                feature_type,
                config: config.clone(),
                extraction_time_ms: processing_time_ms,
                extraction_method: format!("extract_interval_{:.2}_{:.2}", interval.start, interval.end),
            }),
            dimensions: pooled_features.len(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        
        info!("时间段 [{:.2}-{:.2}] 处理完成，特征维度: {}，耗时: {}ms", 
              interval.start, interval.end, result.dimensions, processing_time_ms);
        
        Ok(result)
    }
    
    /// 清空缓存
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        info!("已清空特征缓存");
    }
    
    /// 获取缓存统计信息
    pub fn get_cache_stats(&self) -> CacheStats {
        self.cache.get_stats()
    }
    
    /// 获取性能基准历史
    pub fn get_benchmarks(&self) -> &[PerformanceBenchmark] {
        &self.benchmarks
    }
    
    /// 比较两次基准测试的性能差异
    pub fn compare_benchmarks(&self, index1: usize, index2: usize) -> Option<BenchmarkComparison> {
        if index1 >= self.benchmarks.len() || index2 >= self.benchmarks.len() {
            return None;
        }
        
        Some(BenchmarkComparison::new(
            self.benchmarks[index1].clone(), 
            self.benchmarks[index2].clone()
        ))
    }
    
    /// 获取资源使用情况
    pub fn get_resource_usage(&self) -> &ResourceUsage {
        &self.resource_usage
    }
    
    /// 获取最后一次错误
    pub fn get_last_error(&self) -> Option<&VideoExtractionError> {
        self.last_error.as_ref()
    }
    
    /// 获取处理统计信息
    pub fn get_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        
        stats.insert("processed_videos".to_string(), self.processed_videos_count.to_string());
        stats.insert("total_bytes_processed".to_string(), self.total_bytes_processed.to_string());
        
        if let Some(time) = self.last_extraction_time {
            stats.insert("last_extraction_time_ms".to_string(), time.as_millis().to_string());
        }
        
        let cache_stats = self.get_cache_stats();
        stats.insert("cache_size".to_string(), cache_stats.size.to_string());
        stats.insert("cache_hit_rate".to_string(), format!("{:.2}", cache_stats.hit_rate));
        
        stats
    }
    
    /// 更新资源使用情况
    fn update_resource_usage(&mut self) {
        let mut usage = ResourceUsage {
            cpu_usage_percent: 0.0,
            memory_usage_mb: 0.0,
            gpu_usage_percent: None,
            gpu_memory_usage_mb: None,
            disk_io_mbps: 0.0,
            thread_count: self.config.max_threads,
            sample_time: current_timestamp(),
        };
        
        // 获取CPU使用率
        #[cfg(target_os = "linux")]
        {
            if let Ok(cpu_info) = std::fs::read_to_string("/proc/stat") {
                if let Some(cpu_line) = cpu_info.lines().next() {
                    let values: Vec<u64> = cpu_line.split_whitespace()
                        .skip(1) // 跳过"cpu"标签
                        .filter_map(|v| v.parse::<u64>().ok())
                        .collect();
                    
                    if values.len() >= 4 {
                        // 计算CPU使用时间和总时间
                        let idle = values[3];
                        let total: u64 = values.iter().sum();
                        
                        // 存储当前值，下次更新时计算差值
                        static mut PREV_IDLE: u64 = 0;
                        static mut PREV_TOTAL: u64 = 0;
                        
                        unsafe {
                            if PREV_TOTAL > 0 {
                                let idle_delta = idle - PREV_IDLE;
                                let total_delta = total - PREV_TOTAL;
                                
                                if total_delta > 0 {
                                    usage.cpu_usage_percent = 100.0 * (1.0 - (idle_delta as f64 / total_delta as f64));
                                }
                            }
                            
                            PREV_IDLE = idle;
                            PREV_TOTAL = total;
                        }
                    }
                }
            }
        }
        
        #[cfg(all(target_os = "windows", feature = "winapi"))]
        {
            use winapi::um::processthreadsapi::{GetCurrentProcess, GetProcessTimes};
            use winapi::shared::minwindef::FILETIME;
            
            // 获取当前进程的 CPU 时间
            unsafe {
                let process_handle = GetCurrentProcess();
                let mut creation_time = std::mem::zeroed::<FILETIME>();
                let mut exit_time = std::mem::zeroed::<FILETIME>();
                let mut kernel_time = std::mem::zeroed::<FILETIME>();
                let mut user_time = std::mem::zeroed::<FILETIME>();
                
                if GetProcessTimes(
                    process_handle,
                    &mut creation_time,
                    &mut exit_time,
                    &mut kernel_time,
                    &mut user_time
                ) != 0 {
                    // 将 FILETIME 转换为 u64 (100纳秒单位)
                    let filetime_to_u64 = |ft: &FILETIME| -> u64 {
                        ((ft.dwHighDateTime as u64) << 32) | (ft.dwLowDateTime as u64)
                    };
                    
                    // 计算CPU使用率
                    let cpu_time = (filetime_to_u64(&kernel_time) + filetime_to_u64(&user_time)) as f64 / 10000000.0;  // 转换为秒
                    usage.cpu_usage_percent = ((cpu_time / self.last_extraction_time.unwrap().as_secs_f64()) * 100.0) as f32;
                }
            }
        }
        
        // 获取内存使用情况
        #[cfg(target_os = "linux")]
        {
            if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                if let Some(line) = status.lines().find(|l| l.starts_with("VmRSS:")) {
                    if let Some(mem_kb) = line.split_whitespace().nth(1) {
                        if let Ok(mem_kb) = mem_kb.parse::<f64>() {
                            usage.memory_usage_mb = mem_kb / 1024.0;
                        }
                    }
                }
            }
        }
        
        #[cfg(target_os = "windows")]
        {
            #[cfg(feature = "winapi")]
            use winapi::um::processthreadsapi::GetCurrentProcess;
            #[cfg(feature = "winapi")]
            {
                use winapi::um::psapi::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
                
                unsafe {
                    let h_process = GetCurrentProcess();
                    let mut pmc: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
                    let cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
                    
                    if GetProcessMemoryInfo(
                        h_process,
                        &mut pmc as *mut _ as *mut _,
                        cb
                    ) != 0 {
                        // WorkingSetSize是进程当前驻留在物理内存中的大小
                        usage.memory_usage_mb = (pmc.WorkingSetSize as f64 / (1024.0 * 1024.0)) as f32;
                    }
                }
            }
        }
        
        // 获取线程数
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            usage.thread_count = num_cpus::get_physical().min(self.config.max_threads);
        }
        
        // 如果配置了GPU使用，尝试获取GPU使用情况
        if self.config.use_gpu {
            #[cfg(feature = "cuda")]
            {
                // 使用CUDA API获取GPU使用情况
                if let Some(gpu_info) = self.get_cuda_gpu_info() {
                    usage.gpu_usage_percent = Some(gpu_info.0);
                    usage.gpu_memory_usage_mb = Some(gpu_info.1);
                }
            }
            
            #[cfg(not(feature = "cuda"))]
            {
                // 没有CUDA支持时提供默认值
                usage.gpu_usage_percent = Some(0.0);
                usage.gpu_memory_usage_mb = Some(0.0);
            }
        }
        
        // 估算磁盘I/O
        if let Some(last_time) = self.last_extraction_time {
            if last_time.as_millis() > 0 {
                let bytes_per_sec = self.total_bytes_processed as f64 / last_time.as_secs_f64();
                usage.disk_io_mbps = (bytes_per_sec / (1024.0 * 1024.0)) as f32;
            }
        }
        
        // 更新资源使用情况
        self.resource_usage = usage;
    }
    
    #[cfg(feature = "cuda")]
    fn get_cuda_gpu_info(&self) -> Option<(f64, f64)> {
        use std::ffi::CString;
        use std::os::raw::{c_int, c_uint, c_ulonglong};
        
        extern "C" {
            fn cudaGetDevice(device: *mut c_int) -> c_int;
            fn cudaGetDeviceProperties(props: *mut CudaDeviceProperties, device: c_int) -> c_int;
            fn cudaMemGetInfo(free: *mut c_ulonglong, total: *mut c_ulonglong) -> c_int;
        }
        
        #[repr(C)]
        struct CudaDeviceProperties {
            name: [u8; 256],
            totalGlobalMem: c_ulonglong,
            clockRate: c_int,
            // 其他属性...
        }
        
        unsafe {
            // 获取当前设备ID
            let mut device: c_int = 0;
            if cudaGetDevice(&mut device) != 0 {
                return None;
            }
            
            // 获取设备属性
            let mut props: CudaDeviceProperties = std::mem::zeroed();
            if cudaGetDeviceProperties(&mut props, device) != 0 {
                return None;
            }
            
            // 获取内存使用情况
            let mut free: c_ulonglong = 0;
            let mut total: c_ulonglong = 0;
            if cudaMemGetInfo(&mut free, &mut total) != 0 {
                return None;
            }
            
            // 计算内存使用率和内存使用量(MB)
            let used = total - free;
            let usage_percent = 100.0 * (used as f64 / total as f64);
            let memory_mb = used as f64 / (1024.0 * 1024.0);
            
            Some((usage_percent, memory_mb))
        }
    }
    
    /// 记录性能基准
    fn record_benchmark(&mut self, video_count: usize, elapsed: Duration) {
        let total_size = self.total_bytes_processed;
        let processing_time_ms = elapsed.as_millis() as u64;
        
        // 计算处理速度（MB/s）
        let processing_speed_mbps = if processing_time_ms > 0 {
            (total_size as f64 / 1024.0 / 1024.0) / (processing_time_ms as f64 / 1000.0)
        } else {
            0.0
        };
        
        // 平均处理时间
        let avg_processing_time_ms = if video_count > 0 {
            processing_time_ms / video_count as u64
        } else {
            0
        };
        
        // 创建基准记录
        let benchmark = PerformanceBenchmark {
            feature_type: self.config.feature_types[0],
            video_count,
            total_size_bytes: total_size,
            processing_speed_mbps,
            avg_processing_time_ms: avg_processing_time_ms as f64,
            peak_memory_mb: self.resource_usage.memory_usage_mb as f64,
            thread_count: self.resource_usage.thread_count,
            model_type: self.config.model_type,
            metrics: HashMap::new(),
            timestamp: current_timestamp(),
        };
        
        // 添加到历史记录
        self.benchmarks.push(benchmark);
        
        // 限制历史记录大小
        if self.benchmarks.len() > 10 {
            self.benchmarks.remove(0);
        }
    }
    
    /// 导出特征到指定路径
    pub fn export_features(&self, result: &VideoFeatureResult, output_path: impl AsRef<Path>, format: ExportFormat) -> Result<PathBuf> {
        let options = ExportOptions {
            format,
            include_metadata: true,
            include_processing_info: true,
            compress: false,
            batch_size: None,
            custom_options: HashMap::new(),
        };
        
        export::export_features(vec![result], output_path.as_ref(), options)
            .map_err(|e| Error::Data(format!("{}", e)))
            .map(|_| output_path.as_ref().to_path_buf())
    }
    
    /// 导出特征带自定义选项
    pub fn export_features_with_options(&self, result: &VideoFeatureResult, output_path: impl AsRef<Path>, options: &ExportOptions) -> Result<PathBuf> {
        export::export_features(vec![result], output_path.as_ref(), options.clone())
            .map_err(|e| Error::Data(format!("{}", e)))
            .map(|_| output_path.as_ref().to_path_buf())
    }
    
    /// 批量导出特征
    pub fn export_features_batch(&self, results: &[VideoFeatureResult], output_dir: impl AsRef<Path>, format: ExportFormat) -> Result<Vec<PathBuf>> {
        let options = ExportOptions {
            format,
            include_metadata: true,
            include_processing_info: true,
            compress: false,
            batch_size: None,
            custom_options: HashMap::new(),
        };
        
        export::export_features_batch(results, output_dir.as_ref(), &options)
            .map_err(|e| Error::Data(format!("{}", e)))
    }
    
    /// 批量导出特征(带自定义选项)
    pub fn export_features_batch_with_options(&self, results: &[VideoFeatureResult], output_dir: impl AsRef<Path>, options: &ExportOptions) -> Result<Vec<PathBuf>> {
        export::export_features_batch(results, output_dir.as_ref(), options)
            .map_err(|e| Error::Data(format!("{}", e)))
    }
    
    /// 直接提取并导出特征
    pub fn extract_and_export(&mut self, video_path: &str, output_path: impl AsRef<Path>, format: ExportFormat) -> Result<PathBuf> {
        let result = self.extract_features(video_path)?;
        self.export_features(&result, output_path, format)
    }
    
    /// 直接批量提取并导出特征
    pub fn extract_and_export_batch(&mut self, video_paths: &[String], output_dir: impl AsRef<Path>, format: ExportFormat) -> Result<Vec<PathBuf>> {
        let results = self.extract_features_batch(video_paths)?;
        
        // 过滤出成功的结果
        let successful_results: Vec<_> = results.into_iter()
            .filter_map(|r| r.ok())
            .collect();
        
        self.export_features_batch(&successful_results, output_dir, format)
    }
    
    /// 创建批处理执行器
    pub fn create_batch_executor(&self) -> batch::BatchExecutor {
        let config = batch::BatchProcessorConfig {
            threads: self.config.max_threads,
            memory_limit_mb: Some(self.config.memory_limit_mb),
            timeout_seconds: 3600,
            retry_count: 3,
        };
        
        batch::BatchExecutor::new(self.clone(), Some(config))
    }
    
    /// 执行批处理任务
    pub fn batch_process(&mut self, video_paths: Vec<String>) -> Result<batch::BatchSummary> {
        let mut executor = self.create_batch_executor();
        executor.execute(video_paths).map_err(|e| Error::Data(format!("{}", e)))
    }
    
    /// 执行批处理任务并提供进度回调
    pub fn batch_process_with_progress<F>(&mut self, video_paths: Vec<String>, progress_callback: F) 
        -> Result<batch::BatchSummary>
    where 
        F: Fn(f32, HashMap<String, ProcessingProgress>) -> bool + Send + Sync + 'static
    {
        let mut executor = self.create_batch_executor();
        executor.execute_with_callback(video_paths, Some(progress_callback))
            .map_err(|e| Error::Data(format!("{}", e)))
    }
    
    /// 创建批处理计划
    pub fn create_batch_plan(&self, video_paths: Vec<String>) -> batch::BatchProcessingPlan {
        batch::BatchProcessingPlan::new(
            video_paths,
            self.config.batch_size.unwrap_or(10),
            None
        )
    }
    
    /// 优化批处理计划
    pub fn optimize_batch_plan(&self, plan: &mut batch::BatchProcessingPlan) {
        plan.optimize(Some(self.config.memory_limit_mb));
    }
    
    /// 估计内存使用
    pub fn estimate_batch_memory(&self, video_count: usize, frames_per_video: usize) -> usize {
        let estimator = batch::MemoryEstimator::new();
        estimator.estimate_batch_processing(
            self.config.batch_size.unwrap_or(10),
            self.config.max_threads,
            frames_per_video,
            self.config.use_cache
        )
    }
    
    /// 建议最佳批处理参数
    pub fn suggest_optimal_batch_parameters(&self, video_count: usize, frames_per_video: usize) 
        -> (usize, usize, bool) 
    {
        let estimator = batch::MemoryEstimator::new();
        estimator.optimize_parameters(
            self.config.memory_limit_mb,
            video_count,
            frames_per_video
        )
    }
    
    /// 复制提取器
    pub fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            cache: FeatureCache::new(self.config.cache_size),
            benchmarks: self.benchmarks.clone(),
            last_extraction_time: self.last_extraction_time,
            processed_videos_count: self.processed_videos_count,
            total_bytes_processed: self.total_bytes_processed,
            last_error: self.last_error.clone(),
            resource_usage: self.resource_usage.clone(),
        }
    }
    
    /// 导出特征到TensorBoard可视化
    pub fn export_to_tensorboard(&self, results: &[VideoFeatureResult], log_dir: impl AsRef<Path>) -> Result<PathBuf> {
        // 可选的标签映射
        let label_map = None;
        
        export::export_to_tensorboard(results, log_dir, label_map)
            .map_err(|e| Error::Data(format!("{}", e)))
    }
    
    /// 导出特征到TensorBoard可视化(带标签)
    pub fn export_to_tensorboard_with_labels(&self, 
        results: &[VideoFeatureResult], 
        log_dir: impl AsRef<Path>,
        labels: HashMap<String, String>
    ) -> Result<PathBuf> {
        export::export_to_tensorboard(results, log_dir, Some(labels))
            .map_err(|e| Error::Data(format!("{}", e)))
    }
    
    /// 获取可用的导出格式
    pub fn get_available_export_formats(&self) -> Vec<ExportFormat> {
        export::get_available_export_formats()
    }
    
    /// 检查导出格式是否支持
    pub fn is_format_supported(&self, format: ExportFormat) -> bool {
        export::is_format_supported(format)
    }
    
    /// 创建导出选项
    pub fn create_export_options(&self, 
        format: ExportFormat,
        include_metadata: bool,
        include_processing_info: bool,
        compress: bool
    ) -> ExportOptions {
        export::create_export_options(
            format, 
            include_metadata, 
            include_processing_info, 
            compress
        )
    }

    /// 配置提取器
    pub fn configure(&mut self, config: VideoFeatureConfig) -> Result<()> {
        // 验证新配置
        config.validate().map_err(|e| Error::Data(format!("配置验证失败: {}", e)))?;
        
        // 如果缓存配置发生变化，需要调整缓存
        if self.config.cache_size != config.cache_size {
            let new_cache = FeatureCache::new(config.cache_size);
            self.cache = new_cache;
        }
        
        // 应用新配置
        self.config = config;
        
        Ok(())
    }
}

// current_timestamp 已由 util 模块提供并通过 `pub use util::*` 导出，此处不再重复定义