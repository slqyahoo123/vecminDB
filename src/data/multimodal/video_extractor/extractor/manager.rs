//! 提取器管理器
//! 
//! 负责管理多个提取器实例和缓存

use super::VideoFeatureExtractor;
use super::ExtractorFactory;
use crate::data::multimodal::video_extractor::types::*;
use crate::data::multimodal::video_extractor::config::VideoFeatureConfig;
use crate::data::multimodal::video_extractor::error::VideoExtractionError;
use crate::data::multimodal::video_extractor::cache::FeatureCache;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use log::{info, warn, error, debug};

/// 提取器管理器，用于管理多个提取器
pub struct ExtractorManager {
    factory: ExtractorFactory,
    active_extractors: HashMap<String, Arc<Mutex<Box<dyn VideoFeatureExtractor + Send + Sync>>>>,
    default_config: VideoFeatureConfig,
    cache: FeatureCache,
}

impl ExtractorManager {
    /// 创建新的提取器管理器
    pub fn new(factory: ExtractorFactory, default_config: VideoFeatureConfig) -> Self {
        Self {
            factory,
            active_extractors: HashMap::new(),
            default_config,
            cache: FeatureCache::new(default_config.cache_size),
        }
    }
    
    /// 获取提取器，如果不存在则创建
    pub fn get_extractor(&mut self, name: &str) -> Result<Arc<Mutex<Box<dyn VideoFeatureExtractor + Send + Sync>>>, VideoExtractionError> {
        if !self.active_extractors.contains_key(name) {
            let extractor = self.factory.create_extractor(name)?;
            self.active_extractors.insert(name.to_string(), Arc::new(Mutex::new(extractor)));
        }
        
        Ok(self.active_extractors.get(name).unwrap().clone())
    }
    
    /// 释放所有提取器资源
    pub fn release_all(&mut self) -> Result<(), VideoExtractionError> {
        let mut errors = Vec::new();
        
        for (name, extractor) in &self.active_extractors {
            match extractor.lock() {
                Ok(mut extractor) => {
                    if let Err(e) = extractor.release() {
                        warn!("释放提取器资源失败: {}, 错误: {}", name, e);
                        errors.push(e);
                    }
                },
                Err(e) => {
                    warn!("获取提取器锁失败: {}, 错误: {}", name, e);
                    errors.push(VideoExtractionError::ProcessingError(format!("获取提取器锁失败: {}", e)));
                }
            }
        }
        
        self.active_extractors.clear();
        
        if errors.is_empty() {
            Ok(())
        } else {
            Err(VideoExtractionError::ProcessingError(format!("释放资源时发生{}个错误", errors.len())))
        }
    }
    
    /// 提取视频特征
    pub fn extract_features(&mut self, video_path: &Path, extractor_name: &str, config: Option<&VideoFeatureConfig>) 
        -> Result<VideoFeature, VideoExtractionError> {
        let config = match config {
            Some(cfg) => cfg,
            None => &self.default_config,
        };
        
        // 尝试从缓存获取
        if config.use_cache {
            let cache_key = format!("{}_{}_{}", video_path.display(), extractor_name, config.to_cache_key());
            if let Some(result) = self.cache.get(&cache_key) {
                info!("从缓存中获取特征: {}", cache_key);
                
                // 从VideoFeatureResult转换为VideoFeature
                return Ok(VideoFeature {
                    feature_type: match extractor_name {
                        "rgb" => VideoFeatureType::RGB,
                        "optical_flow" => VideoFeatureType::OpticalFlow,
                        _ => VideoFeatureType::Generic,
                    },
                    features: result.features.clone(),
                    metadata: result.metadata.clone(),
                    dimensions: result.dimensions,
                    timestamp: result.timestamp,
                });
            }
        }
        
        let extractor = self.get_extractor(extractor_name)?;
        
        let mut extractor_guard = extractor.lock().map_err(|e| {
            VideoExtractionError::ProcessingError(format!("获取提取器锁失败: {}", e))
        })?;
        
        // 测量提取时间
        let start_time = std::time::Instant::now();
        let result = extractor_guard.extract_features(video_path, config)?;
        let extraction_time_ms = start_time.elapsed().as_millis() as u64;
        
        // 添加到缓存
        if config.use_cache {
            let cache_key = format!("{}_{}_{}", video_path.display(), extractor_name, config.to_cache_key());
            
            // 将VideoFeature转换为VideoFeatureResult用于缓存
            let result_for_cache = VideoFeatureResult {
                feature_type: result.feature_type,
                features: result.features.clone(),
                metadata: result.metadata.clone(),
                processing_info: Some(ProcessingInfo {
                    feature_type: result.feature_type,
                    config: config.clone(),
                    extraction_time_ms,
                    extraction_method: "提取器直接提取".to_string(),
                }),
                dimensions: result.dimensions,
                timestamp: result.timestamp,
            };
            
            self.cache.put(&cache_key, result_for_cache);
        }
        
        Ok(result)
    }
    
    /// 批量提取视频特征
    pub fn batch_extract(&mut self, video_paths: &[PathBuf], extractor_name: &str, config: Option<&VideoFeatureConfig>) 
        -> Result<HashMap<PathBuf, Result<VideoFeature, VideoExtractionError>>, VideoExtractionError> {
        let config = match config {
            Some(cfg) => cfg,
            None => &self.default_config,
        };
        
        let extractor = self.get_extractor(extractor_name)?;
        
        let mut results = HashMap::new();
        
        // 首先检查缓存
        let mut paths_to_process = Vec::new();
        if config.use_cache {
            for path in video_paths {
                let cache_key = format!("{}_{}_{}", path.display(), extractor_name, config.to_cache_key());
                if let Some(cached_result) = self.cache.get(&cache_key) {
                    info!("从缓存中获取特征: {}", cache_key);
                    
                    // 从VideoFeatureResult转换为VideoFeature
                    let feature = VideoFeature {
                        feature_type: match extractor_name {
                            "rgb" => VideoFeatureType::RGB,
                            "optical_flow" => VideoFeatureType::OpticalFlow,
                            _ => VideoFeatureType::Generic,
                        },
                        features: cached_result.features.clone(),
                        metadata: cached_result.metadata.clone(),
                        dimensions: cached_result.dimensions,
                        timestamp: cached_result.timestamp,
                    };
                    
                    results.insert(path.clone(), Ok(feature));
                } else {
                    paths_to_process.push(path.clone());
                }
            }
        } else {
            paths_to_process = video_paths.to_vec();
        }
        
        if paths_to_process.is_empty() {
            return Ok(results);
        }
        
        // 处理未缓存的路径
        let mut extractor_guard = extractor.lock().map_err(|e| {
            VideoExtractionError::ProcessingError(format!("获取提取器锁失败: {}", e))
        })?;
        
        // 测量批处理时间
        let start_time = std::time::Instant::now();
        let batch_results = extractor_guard.batch_extract(&paths_to_process, config)?;
        let total_extraction_time_ms = start_time.elapsed().as_millis() as u64;
        
        // 计算每个视频的平均处理时间
        let avg_extraction_time_ms = if !paths_to_process.is_empty() {
            total_extraction_time_ms / paths_to_process.len() as u64
        } else {
            0
        };
        
        // 添加到结果和缓存
        for (path, result) in batch_results {
            if config.use_cache && result.is_ok() {
                let cache_key = format!("{}_{}_{}", path.display(), extractor_name, config.to_cache_key());
                let feature = result.as_ref().unwrap();
                
                // 将VideoFeature转换为VideoFeatureResult用于缓存
                let result_for_cache = VideoFeatureResult {
                    feature_type: feature.feature_type,
                    features: feature.features.clone(),
                    metadata: feature.metadata.clone(),
                    processing_info: Some(ProcessingInfo {
                        feature_type: feature.feature_type,
                        config: config.clone(),
                        extraction_time_ms: avg_extraction_time_ms,
                        extraction_method: "提取器批量提取".to_string(),
                    }),
                    dimensions: feature.dimensions,
                    timestamp: feature.timestamp,
                };
                
                self.cache.put(&cache_key, result_for_cache);
            }
            
            results.insert(path, result);
        }
        
        Ok(results)
    }
    
    /// 运行特征提取器基准测试
    pub fn run_benchmark(&mut self, video_path: &Path, extractor_name: &str, config: Option<&VideoFeatureConfig>, iterations: usize) 
        -> Result<super::benchmark::BenchmarkResult, VideoExtractionError> {
        let config = match config {
            Some(cfg) => cfg,
            None => &self.default_config,
        };
        
        let extractor = self.get_extractor(extractor_name)?;
        
        let mut extractor_guard = extractor.lock().map_err(|e| {
            VideoExtractionError::ProcessingError(format!("获取提取器锁失败: {}", e))
        })?;
        
        super::benchmark::run_benchmark(&mut **extractor_guard, video_path, config, iterations)
    }
    
    /// 获取所有可用的提取器
    pub fn get_available_extractors(&self) -> Vec<String> {
        self.factory.get_available_extractors()
    }
    
    /// 检查提取器是否可用
    pub fn is_extractor_available(&self, name: &str) -> bool {
        self.factory.is_extractor_available(name)
    }
    
    /// 清除特征缓存
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
    
    /// 获取缓存统计信息
    pub fn get_cache_stats(&self) -> crate::data::multimodal::video_extractor::cache::CacheStats {
        self.cache.get_stats()
    }
    
    /// 设置默认配置
    pub fn set_default_config(&mut self, config: VideoFeatureConfig) {
        self.default_config = config;
    }
    
    /// 获取默认配置
    pub fn get_default_config(&self) -> &VideoFeatureConfig {
        &self.default_config
    }
    
    /// 注册一个新的提取器
    pub fn register_extractor<F>(&mut self, name: &str, factory_fn: F)
    where
        F: Fn() -> Box<dyn VideoFeatureExtractor + Send + Sync> + Send + Sync + 'static
    {
        self.factory.register_extractor(name, factory_fn);
    }
    
    /// 获取诊断信息
    pub fn get_diagnostics(&mut self, extractor_name: &str) -> Result<super::benchmark::DiagnosticInfo, VideoExtractionError> {
        debug!("获取提取器诊断信息: {}", extractor_name);
        
        // 获取提取器实例
        let extractor = self.get_extractor(extractor_name)?;
        
        // 获取提取器锁
        let mut extractor_guard = extractor.lock().map_err(|e| {
            VideoExtractionError::ProcessingError(format!("获取提取器锁失败: {}", e))
        })?;
        
        // 执行诊断
        let diagnostics = super::benchmark::diagnose_extractor(&mut **extractor_guard)?;
        
        // 添加缓存统计信息
        let mut enhanced_diagnostics = diagnostics;
        
        // 添加额外系统信息
        enhanced_diagnostics.system_info.insert(
            "cache_hit_rate".to_string(), 
            format!("{:.2}%", self.cache.get_stats().hit_rate * 100.0)
        );
        
        enhanced_diagnostics.system_info.insert(
            "cache_size".to_string(), 
            format!("{}", self.cache.get_stats().size)
        );
        
        enhanced_diagnostics.system_info.insert(
            "cache_max_size".to_string(), 
            format!("{}", self.cache.get_stats().capacity)
        );
        
        // 添加提取器特定信息
        enhanced_diagnostics.system_info.insert(
            "extractors_registered".to_string(), 
            format!("{}", self.factory.get_available_extractors().len())
        );
        
        // 添加性能建议
        self.add_performance_recommendations(&mut enhanced_diagnostics);
        
        // 生成诊断报告
        let report = enhanced_diagnostics.generate_diagnosis_report();
        debug!("生成诊断报告: {} 字节", report.len());
        
        Ok(enhanced_diagnostics)
    }
    
    /// 添加性能优化建议
    fn add_performance_recommendations(&self, diagnostics: &mut super::benchmark::DiagnosticInfo) {
        // 基于缓存状况添加建议
        let cache_stats = self.cache.get_stats();
        let access_count = cache_stats.hit_count + cache_stats.miss_count;
        if cache_stats.hit_rate < 0.5 && access_count > 10 {
            diagnostics.add_recommendation(format!(
                "缓存命中率较低({}%)，考虑增加缓存大小或优化缓存策略", 
                (cache_stats.hit_rate * 100.0) as u32
            ));
        }
        
        // 基于CPU核心数和线程数添加建议
        if let Some(cores) = diagnostics.system_info.get("cpu_cores") {
            if let Ok(cpu_cores) = cores.parse::<usize>() {
                if diagnostics.resource_usage.thread_count < cpu_cores / 2 {
                    diagnostics.add_recommendation(format!(
                        "当前线程数({})远小于CPU核心数({}), 考虑增加并行提取任务数",
                        diagnostics.resource_usage.thread_count, cpu_cores
                    ));
                }
            }
        }
        
        // 基于提取器类型添加建议
        match diagnostics.extractor_name.as_str() {
            name if name.contains("RGB") => {
                // RGB特定建议
                if diagnostics.resource_usage.memory_usage_mb > 2000.0 {
                    diagnostics.add_recommendation(
                        "RGB提取器内存使用较高，考虑减小处理分辨率或批量大小".to_string()
                    );
                }
                
                // GPU相关建议
                if diagnostics.hardware_capabilities.gpu_model.is_some() && 
                   (diagnostics.resource_usage.gpu_usage_percent.is_none() || 
                    diagnostics.resource_usage.gpu_usage_percent.unwrap() < 5.0) {
                    diagnostics.add_recommendation(
                        "检测到GPU但未充分利用，确认已启用GPU加速功能".to_string()
                    );
                }
            },
            name if name.contains("OpticalFlow") => {
                // 光流特定建议
                if diagnostics.resource_usage.cpu_usage_percent > 90.0 {
                    diagnostics.add_recommendation(
                        "光流计算CPU使用率高，考虑使用更高效的算法或启用硬件加速".to_string()
                    );
                }
            },
            name if name.contains("Composite") => {
                // 复合提取器特定建议
                diagnostics.add_recommendation(
                    "复合提取器可以通过调整聚合策略来优化性能和特征质量".to_string()
                );
            },
            _ => {
                // 通用建议
                diagnostics.add_recommendation(
                    "考虑使用缓存和批处理来提高特征提取效率".to_string()
                );
            }
        }
        
        // 基于整体资源使用情况添加建议
        if diagnostics.resource_usage.disk_io_mbps > 50.0 {
            diagnostics.add_recommendation(
                "磁盘IO较高，考虑使用内存缓存或固态硬盘提高性能".to_string()
            );
        }
    }
}

impl Drop for ExtractorManager {
    fn drop(&mut self) {
        if let Err(e) = self.release_all() {
            error!("释放提取器管理器资源失败: {}", e);
        }
    }
} 