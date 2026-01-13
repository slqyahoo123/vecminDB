//! 视频特征批处理模块
//! 
//! 本模块提供批量处理视频特征提取的功能，支持并行处理和进度跟踪

use crate::data::multimodal::video_extractor::types::*;
use crate::data::multimodal::video_extractor::config::VideoFeatureConfig;
use crate::data::multimodal::video_extractor::error::VideoExtractionError;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::path::Path;
use std::time::Instant;
use std::collections::HashMap;
use log::{info, debug, warn, error};

/// 进度回调函数类型
pub type ProgressCallback = Box<dyn Fn(usize, usize, &str) + Send + Sync>;

/// 批量提取视频特征
pub fn extract_features_batch(
    video_paths: &[String],
    config: &VideoFeatureConfig
) -> Result<Vec<Result<VideoFeatureResult, VideoExtractionError>>, VideoExtractionError> {
    info!("批量处理{}个视频文件", video_paths.len());
    
    if video_paths.is_empty() {
        return Ok(Vec::new());
    }
    
    let start_time = Instant::now();
    
    // 确定并行处理线程数
    let threads = config.get_usize_param("parallel_threads").unwrap_or_else(|| {
        let num_cpus = num_cpus::get();
        info!("未指定并行线程数，使用系统CPU核心数: {}", num_cpus);
        num_cpus
    });
    
    // 设置Rayon线程池
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| VideoExtractionError::ProcessingError(
            format!("无法创建线程池: {}", e)
        ))?;
    
    let results = thread_pool.install(|| {
        video_paths.par_iter()
            .map(|path| {
                debug!("处理视频文件: {}", path);
                match super::extract_features_from_file(path, config) {
                    Ok(result) => {
                        debug!("视频处理成功: {}", path);
                        Ok(result)
                    },
                    Err(err) => {
                        error!("视频处理失败: {}, 错误: {:?}", path, err);
                        Err(err)
                    }
                }
            })
            .collect::<Vec<_>>()
    });
    
    let elapsed = start_time.elapsed();
    info!("批量处理{}个视频文件完成，耗时: {:?}，平均每个视频: {:?}", 
        video_paths.len(), elapsed, elapsed.div_f32(video_paths.len() as f32));
    
    Ok(results)
}

/// 批量提取视频特征，带进度回调
pub fn extract_features_batch_with_progress(
    video_paths: &[String],
    config: &VideoFeatureConfig,
    progress_callback: ProgressCallback
) -> Result<Vec<Result<VideoFeatureResult, VideoExtractionError>>, VideoExtractionError> {
    info!("批量处理{}个视频文件（带进度跟踪）", video_paths.len());
    
    if video_paths.is_empty() {
        return Ok(Vec::new());
    }
    
    let start_time = Instant::now();
    
    // 确定并行处理线程数
    let threads = config.get_usize_param("parallel_threads").unwrap_or_else(|| {
        let num_cpus = num_cpus::get();
        info!("未指定并行线程数，使用系统CPU核心数: {}", num_cpus);
        num_cpus
    });
    
    // 设置Rayon线程池
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| VideoExtractionError::ProcessingError(
            format!("无法创建线程池: {}", e)
        ))?;
    
    // 创建共享的进度计数器
    let processed = Arc::new(Mutex::new(0usize));
    let total = video_paths.len();
    
    let results = thread_pool.install(|| {
        video_paths.par_iter()
            .map(|path| {
                let path_str = path.clone();
                debug!("处理视频文件: {}", path);
                
                // 更新进度
                progress_callback(
                    *processed.lock().unwrap(),
                    total,
                    &format!("处理 {}", Path::new(path).file_name().unwrap_or_default().to_string_lossy())
                );
                
                let result = match super::extract_features_from_file(&path_str, config) {
                    Ok(result) => {
                        debug!("视频处理成功: {}", path);
                        Ok(result)
                    },
                    Err(err) => {
                        error!("视频处理失败: {}, 错误: {:?}", path, err);
                        Err(err)
                    }
                };
                
                // 增加计数并更新进度
                let mut counter = processed.lock().unwrap();
                *counter += 1;
                progress_callback(
                    *counter,
                    total,
                    &format!("已完成 {}", Path::new(path).file_name().unwrap_or_default().to_string_lossy())
                );
                
                result
            })
            .collect::<Vec<_>>()
    });
    
    let elapsed = start_time.elapsed();
    info!("批量处理{}个视频文件完成，耗时: {:?}，平均每个视频: {:?}", 
        video_paths.len(), elapsed, elapsed.div_f32(video_paths.len() as f32));
    
    Ok(results)
}

/// 批量提取视频特征，支持自定义错误处理策略
pub fn extract_features_batch_with_strategy(
    video_paths: &[String],
    config: &VideoFeatureConfig,
    error_strategy: ErrorStrategy,
    progress_callback: Option<ProgressCallback>
) -> Result<Vec<Result<VideoFeatureResult, VideoExtractionError>>, VideoExtractionError> {
    info!("批量处理{}个视频文件（带错误处理策略）", video_paths.len());
    
    if video_paths.is_empty() {
        return Ok(Vec::new());
    }
    
    let start_time = Instant::now();
    
    // 确定并行处理线程数
    let threads = config.get_usize_param("parallel_threads").unwrap_or_else(|| {
        let num_cpus = num_cpus::get();
        info!("未指定并行线程数，使用系统CPU核心数: {}", num_cpus);
        num_cpus
    });
    
    // 设置Rayon线程池
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|e| VideoExtractionError::ProcessingError(
            format!("无法创建线程池: {}", e)
        ))?;
    
    // 创建共享的进度计数器和错误计数器
    let processed = Arc::new(Mutex::new(0usize));
    let error_count = Arc::new(Mutex::new(0usize));
    let total = video_paths.len();
    
    let results = thread_pool.install(|| {
        video_paths.par_iter()
            .map(|path| {
                let path_str = path.clone();
                debug!("处理视频文件: {}", path);
                
                // 更新进度（如果有回调）
                if let Some(ref callback) = progress_callback {
                    callback(
                        *processed.lock().unwrap(),
                        total,
                        &format!("处理 {}", Path::new(path).file_name().unwrap_or_default().to_string_lossy())
                    );
                }
                
                let mut result = process_with_strategy(&path_str, config, &error_strategy);
                
                // 如果出错并且策略是重试
                if let (Err(_), ErrorStrategy::Retry(max_retries)) = (&result, &error_strategy) {
                    let mut retries = 0;
                    while let Err(err) = result {
                        if retries >= *max_retries {
                            error!("视频处理失败，已达最大重试次数: {}, 错误: {:?}", path, err);
                            break;
                        }
                        
                        retries += 1;
                        warn!("重试处理视频 ({}): {}", retries, path);
                        result = super::extract_features_from_file(&path_str, config);
                    }
                }
                
                // 增加计数并更新进度
                let mut counter = processed.lock().unwrap();
                *counter += 1;
                
                // 统计错误数
                if result.is_err() {
                    let mut errors = error_count.lock().unwrap();
                    *errors += 1;
                    
                    // 检查是否需要停止处理
                    if let ErrorStrategy::StopOnFailure = error_strategy {
                        if *errors == 1 {  // 只打印一次
                            error!("遇到错误，根据策略停止后续处理");
                        }
                        // 这里无法真正停止并行处理，只能在后续检查
                    }
                }
                
                // 更新进度（如果有回调）
                if let Some(ref callback) = progress_callback {
                    callback(
                        *counter,
                        total,
                        &format!("已完成 {}", Path::new(path).file_name().unwrap_or_default().to_string_lossy())
                    );
                }
                
                result
            })
            .collect::<Vec<_>>()
    });
    
    // 检查是否需要提前返回错误
    if let ErrorStrategy::StopOnFailure = error_strategy {
        let errors = *error_count.lock().unwrap();
        if errors > 0 {
            return Err(VideoExtractionError::ProcessingError(
                format!("批处理中止：遇到{}个错误", errors)
            ));
        }
    }
    
    let elapsed = start_time.elapsed();
    let errors = *error_count.lock().unwrap();
    info!("批量处理{}个视频文件完成，成功: {}，失败: {}，耗时: {:?}", 
        video_paths.len(), video_paths.len() - errors, errors, elapsed);
    
    Ok(results)
}

/// 按策略处理单个视频
fn process_with_strategy(
    path: &str,
    config: &VideoFeatureConfig,
    strategy: &ErrorStrategy
) -> Result<VideoFeatureResult, VideoExtractionError> {
    match super::extract_features_from_file(path, config) {
        Ok(result) => {
            debug!("视频处理成功: {}", path);
            Ok(result)
        },
        Err(err) => {
            match strategy {
                ErrorStrategy::StopOnFailure => {
                    error!("视频处理失败，策略：停止处理: {}, 错误: {:?}", path, err);
                    Err(err)
                },
                ErrorStrategy::ContinueOnFailure => {
                    warn!("视频处理失败，策略：继续处理: {}, 错误: {:?}", path, err);
                    Err(err)
                },
                ErrorStrategy::Retry(_) => {
                    // 重试逻辑在外层处理
                    Err(err)
                },
                ErrorStrategy::Skip => {
                    warn!("视频处理失败，策略：跳过: {}, 错误: {:?}", path, err);
                    // 这里返回错误，外层会统计但不会中断处理
                    Err(err)
                },
            }
        }
    }
}

/// 批量提取视频特征，支持不同特征类型
pub fn extract_multiple_features_batch(
    video_paths: &[String],
    feature_types: &[VideoFeatureType],
    base_config: &VideoFeatureConfig
) -> Result<HashMap<String, HashMap<VideoFeatureType, Result<VideoFeatureResult, VideoExtractionError>>>, VideoExtractionError> {
    info!("批量处理{}个视频文件的{}种特征类型", video_paths.len(), feature_types.len());
    
    if video_paths.is_empty() || feature_types.is_empty() {
        return Ok(HashMap::new());
    }
    
    let start_time = Instant::now();
    
    // 创建结果容器
    let mut results: HashMap<String, HashMap<VideoFeatureType, Result<VideoFeatureResult, VideoExtractionError>>> = HashMap::new();
    
    // 对每种特征类型单独处理
    for &feature_type in feature_types {
        info!("处理特征类型: {:?}", feature_type);
        
        // 创建针对该特征类型的配置
        let mut config = base_config.clone();
        config.feature_types = vec![feature_type];
        
        // 批量处理
        let type_results = extract_features_batch(video_paths, &config)?;
        
        // 合并结果
        for (path, result) in video_paths.iter().zip(type_results.into_iter()) {
            let path_results = results.entry(path.clone()).or_insert_with(HashMap::new);
            path_results.insert(feature_type, result);
        }
    }
    
    let elapsed = start_time.elapsed();
    info!("批量处理{}个视频文件的{}种特征类型完成，耗时: {:?}", 
        video_paths.len(), feature_types.len(), elapsed);
    
    Ok(results)
}

/// 提取单个视频的多种特征
pub fn extract_multiple_features(
    video_path: &str,
    feature_types: &[VideoFeatureType],
    base_config: &VideoFeatureConfig
) -> Result<HashMap<VideoFeatureType, Result<VideoFeatureResult, VideoExtractionError>>, VideoExtractionError> {
    info!("处理视频{}的{}种特征类型", video_path, feature_types.len());
    
    if feature_types.is_empty() {
        return Ok(HashMap::new());
    }
    
    let start_time = Instant::now();
    let mut results = HashMap::new();
    
    // 对每种特征类型单独处理
    for &feature_type in feature_types {
        debug!("处理特征类型: {:?}", feature_type);
        
        // 创建针对该特征类型的配置
        let mut config = base_config.clone();
        config.feature_types = vec![feature_type];
        
        // 提取特征
        let result = super::extract_features_from_file(video_path, &config);
        results.insert(feature_type, result);
    }
    
    let elapsed = start_time.elapsed();
    info!("处理视频{}的{}种特征类型完成，耗时: {:?}", 
        video_path, feature_types.len(), elapsed);
    
    Ok(results)
} 