//! 视频特征提取器批处理模块
//!
//! 本模块提供了批量处理视频和优化资源使用的功能

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use rayon::prelude::*;
use super::error::VideoExtractionError;
use super::types::{VideoFeatureResult, ProcessingProgress, ExtractionStatus};
// super::video_log and super::util are not directly used here
use log::{info, warn, error};
use crate::data::multimodal::video_extractor::ResourceUsage;
use crate::data::multimodal::video_extractor::VideoFeatureExtractor;
// VideoFeatureType not directly used in this module

/// 批处理计划，用于组织和安排视频处理任务
#[derive(Clone)]
pub struct BatchProcessingPlan {
    /// 视频ID或路径列表
    video_paths: Vec<String>,
    /// 每批处理的视频数量
    batch_size: usize,
    /// 处理优先级（可选）
    priorities: Option<HashMap<String, u8>>,
    /// 处理顺序
    processing_order: Vec<usize>,
}

impl BatchProcessingPlan {
    /// 创建新的批处理计划
    pub fn new(
        video_paths: Vec<String>,
        batch_size: usize,
        priorities: Option<HashMap<String, u8>>
    ) -> Self {
        let count = video_paths.len();
        let mut plan = Self {
            video_paths,
            batch_size: batch_size.max(1),
            priorities,
            processing_order: (0..count).collect(),
        };
        
        // 根据优先级排序
        plan.sort_by_priority();
        
        plan
    }
    
    /// 根据优先级对处理顺序进行排序
    fn sort_by_priority(&mut self) {
        if let Some(priorities) = &self.priorities {
            // 为每个视频分配优先级
            let mut priority_scores: Vec<(usize, u8)> = self.processing_order
                .iter()
                .map(|&idx| {
                    let priority = priorities
                        .get(&self.video_paths[idx])
                        .copied()
                        .unwrap_or(5); // 默认优先级5
                    (idx, priority)
                })
                .collect();
            
            // 按优先级排序（高优先级在前）
            priority_scores.sort_by(|a, b| b.1.cmp(&a.1));
            
            // 更新处理顺序
            self.processing_order = priority_scores.into_iter().map(|(idx, _)| idx).collect();
        }
    }
    
    /// 获取批次数量
    pub fn get_batch_count(&self) -> usize {
        (self.video_paths.len() + self.batch_size - 1) / self.batch_size
    }
    
    /// 获取指定批次的视频路径
    pub fn get_batch_videos(&self, batch_index: usize) -> Vec<String> {
        let start = batch_index * self.batch_size;
        let end = (start + self.batch_size).min(self.video_paths.len());
        
        self.processing_order[start..end]
            .iter()
            .map(|&idx| self.video_paths[idx].clone())
            .collect()
    }
    
    /// 获取所有视频路径
    pub fn get_all_videos(&self) -> &[String] {
        &self.video_paths
    }
    
    /// 获取批大小
    pub fn get_batch_size(&self) -> usize {
        self.batch_size
    }
    
    /// 设置新的批大小
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size.max(1);
    }
    
    /// 修改视频优先级
    pub fn set_priority(&mut self, video_path: &str, priority: u8) {
        if self.priorities.is_none() {
            self.priorities = Some(HashMap::new());
        }
        
        if let Some(priorities) = &mut self.priorities {
            priorities.insert(video_path.to_string(), priority);
        }
        
        // 重新排序
        self.sort_by_priority();
    }
    
    /// 添加视频到计划中
    pub fn add_video(&mut self, video_path: String, priority: Option<u8>) {
        self.video_paths.push(video_path.clone());
        self.processing_order.push(self.video_paths.len() - 1);
        
        if let Some(p) = priority {
            if self.priorities.is_none() {
                self.priorities = Some(HashMap::new());
            }
            
            if let Some(priorities) = &mut self.priorities {
                priorities.insert(video_path, p);
            }
        }
        
        // 重新排序
        self.sort_by_priority();
    }
    
    /// 移除视频
    pub fn remove_video(&mut self, video_path: &str) -> bool {
        if let Some(index) = self.video_paths.iter().position(|path| path == video_path) {
            self.video_paths.remove(index);
            
            // 更新处理顺序
            self.processing_order = (0..self.video_paths.len()).collect();
            
            // 如果有优先级信息，也需要更新
            if let Some(priorities) = &mut self.priorities {
                priorities.remove(video_path);
            }
            
            // 重新排序
            self.sort_by_priority();
            
            true
        } else {
            false
        }
    }
    
    /// 优化批处理计划
    pub fn optimize(&mut self, memory_constraint: Option<usize>) {
        // 根据可用内存调整批大小
        if let Some(memory_mb) = memory_constraint {
            // 假设每个视频处理大约需要200MB内存
            let estimated_batch_size = (memory_mb / 200).max(1);
            self.batch_size = estimated_batch_size;
        }
        
        info!("批处理计划已优化，批大小: {}", self.batch_size);
    }
}

/// 批处理执行器
pub struct BatchProcessor {
    /// 批处理计划
    plan: BatchProcessingPlan,
    /// 处理配置
    config: BatchProcessorConfig,
    /// 处理进度
    progress: Arc<Mutex<HashMap<String, ProcessingProgress>>>,
    /// 处理结果
    results: Arc<Mutex<HashMap<String, Result<VideoFeatureResult, VideoExtractionError>>>>,
    /// 处理队列
    queue: Arc<Mutex<VecDeque<String>>>,
    /// 开始时间
    start_time: Instant,
    /// 处理状态
    state: Arc<Mutex<BatchProcessingState>>,
}

/// 批处理器配置
#[derive(Clone)]
pub struct BatchProcessorConfig {
    /// 并行处理线程数
    pub threads: usize,
    /// 内存限制（MB）
    pub memory_limit_mb: Option<usize>,
    /// 超时时间（秒）
    pub timeout_seconds: u32,
    /// 失败重试次数
    pub retry_count: u8,
}

impl Default for BatchProcessorConfig {
    fn default() -> Self {
        Self {
            threads: num_cpus::get(),
            memory_limit_mb: None,
            timeout_seconds: 3600,
            retry_count: 3,
        }
    }
}

/// 批处理状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchProcessingState {
    /// 初始化
    Initialized,
    /// 运行中
    Running,
    /// 暂停
    Paused,
    /// 已完成
    Completed,
    /// 已取消
    Canceled,
    /// 发生错误
    Error,
}

impl BatchProcessor {
    /// 创建新的批处理执行器
    pub fn new(plan: BatchProcessingPlan, config: BatchProcessorConfig) -> Self {
        let video_paths = plan.get_all_videos();
        let progress = Arc::new(Mutex::new(HashMap::new()));
        let results = Arc::new(Mutex::new(HashMap::new()));
        
        // 初始化进度信息
        for video_path in video_paths {
            let progress_entry = ProcessingProgress {
                video_id: video_path.clone(),
                status: ExtractionStatus::Queued,
                percentage: 0.0,
                frames_processed: 0,
                total_frames: 0,
                elapsed_seconds: 0.0,
                estimated_remaining_seconds: 0.0,
                current_stage: "等待处理".to_string(),
                error_message: None,
            };
            
            progress.lock().unwrap().insert(video_path.clone(), progress_entry);
        }
        
        // 初始化处理队列
        let mut queue = VecDeque::new();
        for batch_idx in 0..plan.get_batch_count() {
            let batch_videos = plan.get_batch_videos(batch_idx);
            for video_path in batch_videos {
                queue.push_back(video_path);
            }
        }
        
        Self {
            plan,
            config,
            progress,
            results,
            queue: Arc::new(Mutex::new(queue)),
            start_time: Instant::now(),
            state: Arc::new(Mutex::new(BatchProcessingState::Initialized)),
        }
    }
    
    /// 获取处理进度
    pub fn get_progress(&self, video_path: &str) -> Option<ProcessingProgress> {
        self.progress.lock().unwrap().get(video_path).cloned()
    }
    
    /// 获取所有处理进度
    pub fn get_all_progress(&self) -> HashMap<String, ProcessingProgress> {
        self.progress.lock().unwrap().clone()
    }
    
    /// 获取处理结果
    pub fn get_result(&self, video_path: &str) -> Option<Result<VideoFeatureResult, VideoExtractionError>> {
        self.results.lock().unwrap().get(video_path).cloned()
    }
    
    /// 获取所有处理结果
    pub fn get_all_results(&self) -> HashMap<String, Result<VideoFeatureResult, VideoExtractionError>> {
        self.results.lock().unwrap().clone()
    }
    
    /// 更新处理进度
    fn update_progress(&self, video_path: &str, update: ProcessingProgress) {
        let mut progress = self.progress.lock().unwrap();
        progress.insert(video_path.to_string(), update);
    }
    
    /// 添加处理结果
    fn add_result(&self, video_path: &str, result: Result<VideoFeatureResult, VideoExtractionError>) {
        let mut results = self.results.lock().unwrap();
        results.insert(video_path.to_string(), result);
    }
    
    /// 开始处理
    pub fn process<F, P>(&mut self, process_fn: F, progress_callback: Option<P>) -> Result<(), VideoExtractionError>
    where
        F: Fn(&str) -> Result<VideoFeatureResult, VideoExtractionError> + Send + Sync,
        P: Fn(&str, &ProcessingProgress) -> bool + Send + Sync
    {
        // 记录开始时间
        self.start_time = Instant::now();
        
        // 设置状态为运行中
        {
            let mut state = self.state.lock().unwrap();
            *state = BatchProcessingState::Running;
        }
        
        info!("开始批处理，线程数: {}", self.config.threads);
        
        // 创建线程池
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.config.threads)
            .build()
            .map_err(|e| VideoExtractionError::GenericError(
                format!("无法创建线程池: {}", e)
            ))?;
        
        // 创建状态引用以便在线程中使用
        let state_ref = self.state.clone();
        
        pool.install(|| {
            // 获取队列副本用于并行处理
            let queue = {
                let queue_guard = self.queue.lock().unwrap();
                queue_guard.clone()
            };
            
            // 并行处理视频
            queue.into_par_iter().for_each(|video_path| {
                let video_path = video_path.clone();
                
                // 检查处理状态
                if *state_ref.lock().unwrap() == BatchProcessingState::Canceled {
                    // 处理已被取消，不处理此视频
                    return;
                }
                
                // 如果处理被暂停，等待恢复
                while *state_ref.lock().unwrap() == BatchProcessingState::Paused {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    // 再次检查是否被取消
                    if *state_ref.lock().unwrap() == BatchProcessingState::Canceled {
                        return;
                    }
                }
                
                // 更新状态为处理中
                let mut progress = ProcessingProgress {
                    video_id: video_path.clone(),
                    status: ExtractionStatus::Processing,
                    percentage: 0.0,
                    frames_processed: 0,
                    total_frames: 0,
                    elapsed_seconds: 0.0,
                    estimated_remaining_seconds: 0.0,
                    current_stage: "开始处理".to_string(),
                    error_message: None,
                };
                
                self.update_progress(&video_path, progress.clone());
                
                // 调用回调函数
                if let Some(callback) = &progress_callback {
                    if !callback(&video_path, &progress) {
                        // 用户取消了处理
                        progress.status = ExtractionStatus::Canceled;
                        self.update_progress(&video_path, progress);
                        return;
                    }
                }
                
                // 记录开始时间
                let start_time = Instant::now();
                
                // 尝试处理视频
                let mut retry_count = 0;
                let result = loop {
                    // 检查处理状态
                    if *state_ref.lock().unwrap() == BatchProcessingState::Canceled {
                        // 处理已被取消
                        progress.status = ExtractionStatus::Canceled;
                        self.update_progress(&video_path, progress.clone());
                        return;
                    }
                    
                    // 如果处理被暂停，等待恢复
                    while *state_ref.lock().unwrap() == BatchProcessingState::Paused {
                        std::thread::sleep(std::time::Duration::from_millis(500));
                        // 再次检查是否被取消
                        if *state_ref.lock().unwrap() == BatchProcessingState::Canceled {
                            progress.status = ExtractionStatus::Canceled;
                            self.update_progress(&video_path, progress.clone());
                            return;
                        }
                    }
                    
                    let process_result = process_fn(&video_path);
                    
                    match &process_result {
                        Ok(_) => break process_result,
                        Err(e) => {
                            retry_count += 1;
                            if retry_count >= self.config.retry_count {
                                // 超过重试次数
                                error!("视频 {} 处理失败，已重试 {} 次: {}", video_path, retry_count, e);
                                break process_result;
                            }
                            
                            // 更新进度
                            progress.error_message = Some(format!("处理失败，正在重试 ({}/{}): {}", 
                                retry_count, self.config.retry_count, e));
                            self.update_progress(&video_path, progress.clone());
                            
                            // 调用回调函数
                            if let Some(callback) = &progress_callback {
                                if !callback(&video_path, &progress) {
                                    // 用户取消了处理
                                    progress.status = ExtractionStatus::Canceled;
                                    self.update_progress(&video_path, progress.clone());
                                    return;
                                }
                            }
                            
                            // 等待一段时间后重试
                            std::thread::sleep(std::time::Duration::from_secs(2));
                            info!("重试处理视频 {}: {}/{}", video_path, retry_count, self.config.retry_count);
                        }
                    }
                };
                
                // 处理完成，更新状态
                let elapsed = start_time.elapsed();
                
                match &result {
                    Ok(_) => {
                        progress.status = ExtractionStatus::Completed;
                        progress.percentage = 100.0;
                        progress.elapsed_seconds = elapsed.as_secs_f64();
                        progress.estimated_remaining_seconds = 0.0;
                        progress.current_stage = "处理完成".to_string();
                    },
                    Err(e) => {
                        progress.status = ExtractionStatus::Failed;
                        progress.elapsed_seconds = elapsed.as_secs_f64();
                        progress.error_message = Some(format!("{}", e));
                        progress.current_stage = "处理失败".to_string();
                    }
                }
                
                self.update_progress(&video_path, progress.clone());
                
                // 添加结果
                self.add_result(&video_path, result);
                
                // 最终回调
                if let Some(callback) = &progress_callback {
                    callback(&video_path, &progress);
                }
            });
        });
        
        // 更新处理状态
        {
            let mut state = self.state.lock().unwrap();
            
            // 检查是否被取消
            if *state == BatchProcessingState::Canceled {
                info!("批处理已取消");
            } else {
                // 标记为已完成
                *state = BatchProcessingState::Completed;
                info!("批处理完成");
            }
        }
        
        Ok(())
    }
    
    /// 获取总体处理进度百分比
    pub fn get_overall_progress(&self) -> f32 {
        let progress = self.progress.lock().unwrap();
        let total = progress.len();
        
        if total == 0 {
            return 0.0;
        }
        
        let completed_count = progress.values()
            .filter(|p| matches!(p.status, ExtractionStatus::Completed))
            .count();
            
        let in_progress_sum: f32 = progress.values()
            .filter(|p| matches!(p.status, ExtractionStatus::Processing))
            .map(|p| p.percentage)
            .sum();
            
        let in_progress_count = progress.values()
            .filter(|p| matches!(p.status, ExtractionStatus::Processing))
            .count();
            
        // 计算总体进度
        let base_progress = (completed_count as f32 * 100.0) / total as f32;
        let additional_progress = if in_progress_count > 0 {
            (in_progress_sum / in_progress_count as f32) / total as f32
        } else {
            0.0
        };
        
        base_progress + additional_progress
    }
    
    /// 获取批处理运行时间（秒）
    pub fn get_runtime_seconds(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }
    
    /// 获取当前处理状态
    pub fn get_state(&self) -> BatchProcessingState {
        *self.state.lock().unwrap()
    }
    
    /// 暂停处理
    pub fn pause(&self) -> Result<(), VideoExtractionError> {
        let mut state = self.state.lock().unwrap();
        
        // 仅在运行状态下可以暂停
        if *state == BatchProcessingState::Running {
            *state = BatchProcessingState::Paused;
            
            // 更新所有进行中任务的状态为暂停
            let mut progress = self.progress.lock().unwrap();
            for (_, entry) in progress.iter_mut() {
                if entry.status == ExtractionStatus::Processing {
                    entry.current_stage = "已暂停".to_string();
                }
            }
            
            info!("批处理已暂停");
            Ok(())
        } else {
            Err(VideoExtractionError::ProcessingError(
                format!("无法暂停批处理，当前状态: {:?}", *state)
            ))
        }
    }
    
    /// 恢复处理
    pub fn resume(&self) -> Result<(), VideoExtractionError> {
        let mut state = self.state.lock().unwrap();
        
        // 仅在暂停状态下可以恢复
        if *state == BatchProcessingState::Paused {
            *state = BatchProcessingState::Running;
            
            // 更新所有暂停任务的状态为处理中
            let mut progress = self.progress.lock().unwrap();
            for (_, entry) in progress.iter_mut() {
                if entry.status == ExtractionStatus::Processing && entry.current_stage == "已暂停" {
                    entry.current_stage = "处理中".to_string();
                }
            }
            
            info!("批处理已恢复");
            Ok(())
        } else {
            Err(VideoExtractionError::ProcessingError(
                format!("无法恢复批处理，当前状态: {:?}", *state)
            ))
        }
    }
    
    /// 取消处理
    pub fn cancel(&self) -> Result<(), VideoExtractionError> {
        let mut state = self.state.lock().unwrap();
        
        // 仅在运行或暂停状态下可以取消
        if *state == BatchProcessingState::Running || *state == BatchProcessingState::Paused {
            *state = BatchProcessingState::Canceled;
            
            // 更新所有进行中任务和排队任务的状态为取消
            let mut progress = self.progress.lock().unwrap();
            for (_, entry) in progress.iter_mut() {
                if entry.status == ExtractionStatus::Processing || entry.status == ExtractionStatus::Queued {
                    entry.status = ExtractionStatus::Canceled;
                    entry.current_stage = "已取消".to_string();
                }
            }
            
            info!("批处理已取消");
            Ok(())
        } else {
            Err(VideoExtractionError::ProcessingError(
                format!("无法取消批处理，当前状态: {:?}", *state)
            ))
        }
    }
}

/// 内存使用估算器
pub struct MemoryEstimator;

impl MemoryEstimator {
    /// 创建新的内存估算器
    pub fn new() -> Self {
        Self {}
    }
    
    /// 估算批处理所需内存
    pub fn estimate_batch_processing(
        &self,
        batch_size: usize,
        threads: usize,
        frame_count: usize,
        use_cache: bool
    ) -> usize {
        // 基础内存需求(MB)
        let base_memory = 100;
        
        // 每个线程的内存开销(MB)
        let thread_memory = threads * 50;
        
        // 每帧的内存使用(MB)，假设每帧10MB
        let frame_memory = frame_count * 10;
        
        // 缓存内存(MB)
        let cache_memory = if use_cache { 200 } else { 0 };
        
        // 总内存估计(MB)
        base_memory + thread_memory + frame_memory + cache_memory
    }
    
    /// 根据可用内存建议批大小
    pub fn suggest_batch_size(
        &self,
        available_memory_mb: usize,
        threads: usize,
        frames_per_video: usize,
        use_cache: bool
    ) -> usize {
        // 基础内存需求(MB)
        let base_memory = 100;
        
        // 每个线程的内存开销(MB)
        let thread_memory = threads * 50;
        
        // 缓存内存(MB)
        let cache_memory = if use_cache { 200 } else { 0 };
        
        // 固定内存开销
        let fixed_memory = base_memory + thread_memory + cache_memory;
        
        // 如果可用内存小于固定开销，至少处理一个视频
        if available_memory_mb <= fixed_memory {
            return 1;
        }
        
        // 每个视频的内存需求(MB)
        let memory_per_video = frames_per_video * 10;
        
        // 可用于视频处理的内存
        let available_for_videos = available_memory_mb - fixed_memory;
        
        // 计算最大批大小
        let max_batch_size = if memory_per_video > 0 {
            available_for_videos / memory_per_video
        } else {
            10 // 默认值
        };
        
        // 确保至少处理一个视频
        max_batch_size.max(1)
    }
    
    /// 优化处理参数
    pub fn optimize_parameters(
        &self,
        available_memory_mb: usize,
        video_count: usize,
        frames_per_video: usize
    ) -> (usize, usize, bool) {
        // 尝试不同的参数组合
        let possible_threads = [1, 2, 4, 8, 16, 32];
        let mut best_throughput = 0.0;
        let mut best_params = (1, 1, false);
        
        for &threads in &possible_threads {
            // 不考虑超过可用CPU核心数的线程数
            if threads > num_cpus::get() {
                continue;
            }
            
            // 尝试启用和禁用缓存的情况
            for use_cache in [false, true] {
                // 根据内存限制计算批大小
                let batch_size = self.suggest_batch_size(
                    available_memory_mb,
                    threads,
                    frames_per_video,
                    use_cache
                );
                
                // 估算吞吐量（每秒处理视频数）
                // 吞吐量与线程数和批大小正相关，与是否启用缓存有关
                let base_throughput = threads as f32 * 0.5; // 基础吞吐量
                let cache_factor = if use_cache { 1.5 } else { 1.0 }; // 缓存加速因子
                let throughput = base_throughput * batch_size as f32 * cache_factor;
                
                if throughput > best_throughput {
                    best_throughput = throughput;
                    best_params = (batch_size, threads, use_cache);
                }
            }
        }
        
        info!("优化参数: 批大小={}, 线程数={}, 缓存启用={}", 
            best_params.0, best_params.1, best_params.2);
        
        best_params
    }
}

/// 资源监控器
pub struct ResourceMonitor {
    /// 开始时间
    start_time: std::time::Instant,
    /// CPU使用率历史
    cpu_usage_history: Vec<f32>,
    /// 内存使用历史
    memory_usage_history: Vec<f32>,
    /// 磁盘I/O历史
    disk_io_history: Vec<f32>,
    /// 当前批处理器
    processor: Option<Arc<BatchProcessor>>,
    /// 系统信息采集器
    #[cfg(feature = "system_stats")]
    system: sysinfo::System,
    /// 上次I/O统计
    #[cfg(feature = "system_stats")]
    last_disk_stats: Option<std::time::Instant>,
}

impl ResourceMonitor {
    /// 创建新的资源监控器
    pub fn new() -> Self {
        #[cfg(feature = "system_stats")]
        let mut system = sysinfo::System::new();
        
        #[cfg(feature = "system_stats")]
        system.refresh_all();
        
        Self {
            start_time: std::time::Instant::now(),
            cpu_usage_history: Vec::new(),
            memory_usage_history: Vec::new(),
            disk_io_history: Vec::new(),
            processor: None,
            #[cfg(feature = "system_stats")]
            system,
            #[cfg(feature = "system_stats")]
            last_disk_stats: None,
        }
    }
    
    /// 设置批处理器
    pub fn set_processor(&mut self, processor: Arc<BatchProcessor>) {
        self.processor = Some(processor);
    }
    
    /// 开始监控
    pub fn start_monitoring(&mut self) {
        self.start_time = std::time::Instant::now();
        self.cpu_usage_history.clear();
        self.memory_usage_history.clear();
        self.disk_io_history.clear();
        
        #[cfg(feature = "system_stats")]
        {
            self.system.refresh_all();
            self.last_disk_stats = Some(std::time::Instant::now());
        }
    }
    
    /// 采集系统指标
    pub fn collect_metrics(&mut self) -> ResourceUsage {
        #[cfg(feature = "system_stats")]
        {
            // 刷新系统信息
            self.system.refresh_all();
            
            // 采集CPU使用率
            let cpu_usage = self.system.global_cpu_info().cpu_usage();
            self.cpu_usage_history.push(cpu_usage);
            
            // 采集内存使用率
            let total_memory = self.system.total_memory() as f32 / 1024.0 / 1024.0; // 转换为MB
            let used_memory = self.system.used_memory() as f32 / 1024.0 / 1024.0;   // 转换为MB
            let memory_usage = 100.0 * used_memory / total_memory;
            self.memory_usage_history.push(memory_usage);
            
            // 采集磁盘I/O
            let mut disk_io = 0.0;
            if let Some(last_time) = self.last_disk_stats {
                let elapsed = last_time.elapsed().as_secs_f32();
                if elapsed > 0.0 {
                    let mut total_read_bytes = 0;
                    let mut total_written_bytes = 0;
                    
                    for disk in self.system.disks() {
                        total_read_bytes += disk.read_bytes();
                        total_written_bytes += disk.written_bytes();
                    }
                    
                    // MB/s
                    disk_io = (total_read_bytes + total_written_bytes) as f32 / (1024.0 * 1024.0) / elapsed;
                    self.disk_io_history.push(disk_io);
                }
                
                self.last_disk_stats = Some(std::time::Instant::now());
            } else {
                self.last_disk_stats = Some(std::time::Instant::now());
            }
            
            // 计算线程数
            let mut thread_count = 0;
            for process in self.system.processes().values() {
                thread_count += process.thread_count();
            }
            
            // 如果历史太长，清理旧数据
            if self.cpu_usage_history.len() > 100 {
                self.cpu_usage_history.drain(0..50);
                self.memory_usage_history.drain(0..50);
                self.disk_io_history.drain(0..50);
            }
            
            // 构建资源使用对象
            ResourceUsage {
                cpu_usage_percent: cpu_usage,
                memory_usage_mb: used_memory,
                gpu_usage_percent: None, // 需要特殊库支持，如NVML
                gpu_memory_usage_mb: None,
                disk_io_mbps: disk_io,
                thread_count,
                sample_time: chrono::Utc::now().timestamp() as u64,
            }
        }
        
        #[cfg(not(feature = "system_stats"))]
        {
            use rand::Rng;
            
            let mut rng = rand::thread_rng();
            
            // 模拟CPU使用率（0-100%）
            let cpu_usage = (30.0 + rng.gen::<f32>() * 40.0).min(100.0);
            self.cpu_usage_history.push(cpu_usage);
            
            // 模拟内存使用率（0-100%）和内存使用量
            let memory_usage = (40.0 + rng.gen::<f32>() * 30.0).min(100.0);
            let memory_usage_mb = 1000.0 + rng.gen::<f32>() * 3000.0; // 1GB - 4GB
            self.memory_usage_history.push(memory_usage);
            
            // 模拟磁盘I/O（MB/s）
            let disk_io = 5.0 + rng.gen::<f32>() * 25.0; // 5 - 30 MB/s
            self.disk_io_history.push(disk_io);
            
            // 模拟线程数
            let thread_count = 20 + rng.gen_range(0..20);
            
            warn!("使用模拟数据，启用'system_stats'特性可获取真实系统资源信息");
            
            // 如果历史太长，清理旧数据
            if self.cpu_usage_history.len() > 100 {
                self.cpu_usage_history.drain(0..50);
                self.memory_usage_history.drain(0..50);
                self.disk_io_history.drain(0..50);
            }
            
            // 构建资源使用对象
            ResourceUsage {
                cpu_usage_percent: cpu_usage,
                memory_usage_mb: memory_usage_mb,
                gpu_usage_percent: Some(20.0 + rng.gen::<f32>() * 40.0), // 模拟GPU使用
                gpu_memory_usage_mb: Some(500.0 + rng.gen::<f32>() * 1500.0), // 模拟GPU内存
                disk_io_mbps: disk_io,
                thread_count,
                sample_time: chrono::Utc::now().timestamp() as u64,
            }
        }
    }
    
    /// 获取CPU使用率
    pub fn get_cpu_usage(&self) -> f32 {
        if self.cpu_usage_history.is_empty() {
            return 0.0;
        }
        self.cpu_usage_history.iter().sum::<f32>() / self.cpu_usage_history.len() as f32
    }
    
    /// 获取内存使用率
    pub fn get_memory_usage(&self) -> f32 {
        if self.memory_usage_history.is_empty() {
            return 0.0;
        }
        self.memory_usage_history.iter().sum::<f32>() / self.memory_usage_history.len() as f32
    }
    
    /// 获取磁盘I/O
    pub fn get_disk_io(&self) -> f32 {
        if self.disk_io_history.is_empty() {
            return 0.0;
        }
        self.disk_io_history.iter().sum::<f32>() / self.disk_io_history.len() as f32
    }
    
    /// 获取运行时间（秒）
    pub fn get_runtime_seconds(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }
    
    /// 获取总体进度
    pub fn get_overall_progress(&self) -> f32 {
        if let Some(processor) = &self.processor {
            processor.get_overall_progress()
        } else {
            0.0
        }
    }
    
    /// 获取监控报告
    pub fn get_report(&self) -> HashMap<String, String> {
        let mut report = HashMap::new();
        
        report.insert("运行时间".to_string(), format!("{:.1}秒", self.get_runtime_seconds()));
        report.insert("CPU使用率".to_string(), format!("{:.1}%", self.get_cpu_usage()));
        report.insert("内存使用率".to_string(), format!("{:.1}%", self.get_memory_usage()));
        report.insert("磁盘I/O".to_string(), format!("{:.1} MB/s", self.get_disk_io()));
        
        if let Some(processor) = &self.processor {
            report.insert("总体进度".to_string(), format!("{:.1}%", processor.get_overall_progress()));
            
            let progress = processor.get_all_progress();
            let completed = progress.values().filter(|p| matches!(p.status, ExtractionStatus::Completed)).count();
            let failed = progress.values().filter(|p| matches!(p.status, ExtractionStatus::Failed)).count();
            let processing = progress.values().filter(|p| matches!(p.status, ExtractionStatus::Processing)).count();
            let queued = progress.values().filter(|p| matches!(p.status, ExtractionStatus::Queued)).count();
            
            report.insert("总任务数".to_string(), format!("{}", progress.len()));
            report.insert("已完成".to_string(), format!("{}", completed));
            report.insert("处理中".to_string(), format!("{}", processing));
            report.insert("排队中".to_string(), format!("{}", queued));
            report.insert("失败".to_string(), format!("{}", failed));
        }
        
        report
    }
}

/// 批处理结果摘要
#[derive(Debug, Clone)]
pub struct BatchSummary {
    /// 总视频数量
    pub total_videos: usize,
    /// 成功处理数量
    pub success_count: usize,
    /// 失败数量
    pub failure_count: usize,
    /// 取消数量
    pub canceled_count: usize,
    /// 总处理时间(秒)
    pub total_duration_seconds: f64,
    /// 平均处理时间(秒/视频)
    pub avg_processing_seconds: f64,
    /// 错误摘要
    pub error_summary: HashMap<String, usize>,
    /// 成功率
    pub success_rate: f64,
    /// 处理速度(视频/分钟)
    pub processing_speed: f64,
}

/// 批处理执行器，提供更高级别的批处理接口
pub struct BatchExecutor {
    /// 视频提取器
    extractor: Arc<Mutex<VideoFeatureExtractor>>,
    /// 资源监控器
    resource_monitor: ResourceMonitor,
    /// 批处理配置
    config: BatchProcessorConfig,
    /// 完成的批处理结果
    completed_batches: Vec<BatchSummary>,
}

impl BatchExecutor {
    /// 创建新的批处理执行器
    pub fn new(extractor: VideoFeatureExtractor, config: Option<BatchProcessorConfig>) -> Self {
        let config = config.unwrap_or_default();
        
        Self {
            extractor: Arc::new(Mutex::new(extractor)),
            resource_monitor: ResourceMonitor::new(),
            config,
            completed_batches: Vec::new(),
        }
    }
    
    /// 执行批处理
    pub fn execute(&mut self, video_paths: Vec<String>) -> Result<BatchSummary, VideoExtractionError> {
        self.execute_with_callback::<fn(f32, HashMap<String, ProcessingProgress>) -> bool>(video_paths, None)
    }
    
    /// 执行批处理并提供进度回调
    pub fn execute_with_callback<F>(&mut self, video_paths: Vec<String>, progress_callback: Option<F>) 
        -> Result<BatchSummary, VideoExtractionError>
    where 
        F: Fn(f32, HashMap<String, ProcessingProgress>) -> bool + Send + Sync + 'static 
    {
        // 创建批处理计划
        let plan = BatchProcessingPlan::new(
            video_paths,
            self.config.threads.min(10), // 默认批大小为线程数，但不超过10
            None
        );
        
        // 优化批处理计划
        let mut optimized_plan = plan.clone();
        optimized_plan.optimize(self.config.memory_limit_mb);
        
        // 创建批处理器
        let mut processor = BatchProcessor::new(optimized_plan, self.config.clone());
        
        // 设置资源监控
        let processor_ref = Arc::new(processor);
        self.resource_monitor.set_processor(Arc::clone(&processor_ref));
        self.resource_monitor.start_monitoring();
        
        // 准备处理函数
        let extractor_ref = Arc::clone(&self.extractor);
        let process_fn = move |video_path: &str| -> Result<VideoFeatureResult, VideoExtractionError> {
            let mut extractor = extractor_ref.lock().unwrap();
            extractor.extract_features(video_path)
                .map_err(|e| VideoExtractionError::from(e))
        };
        
        // 处理进度回调
        let progress_tracking: Arc<Mutex<HashMap<String, ProcessingProgress>>> = Arc::new(Mutex::new(HashMap::new()));
        let progress_tracking_ref = Arc::clone(&progress_tracking);
        
        let callback_wrapper = if let Some(callback) = progress_callback {
            let processor_ref2 = Arc::clone(&processor_ref);
            let progress_tracking_ref2 = Arc::clone(&progress_tracking_ref);
            
            Some(move |video_path: &str, progress: &ProcessingProgress| -> bool {
                // 更新进度跟踪
                let mut tracking = progress_tracking_ref2.lock().unwrap();
                tracking.insert(video_path.to_string(), progress.clone());
                
                // 获取总体进度
                let overall_progress = processor_ref2.get_overall_progress();
                
                // 调用用户回调
                callback(overall_progress, tracking.clone())
            })
        } else {
            None
        };
        
        // 执行批处理
        info!("开始执行批处理，视频数量: {}", processor_ref.plan.get_all_videos().len());
        let start_time = Instant::now();
        
        // 我们需要将processor_ref转换回可变引用以调用process方法
        // 因为Arc没有提供get_mut方法，我们需要使用unsafe来实现
        let processor_ptr = Arc::into_raw(processor_ref);
        let processor_mut = unsafe { &mut *processor_ptr };
        
        let result = processor_mut.process(process_fn, callback_wrapper);
        
        // 恢复Arc以便正确管理内存
        let _ = unsafe { Arc::from_raw(processor_ptr) };
        
        let elapsed = start_time.elapsed();
        
        if let Err(e) = result {
            error!("批处理执行失败: {}", e);
            return Err(e);
        }
        
        // 生成批处理摘要
        let summary = self.generate_summary(processor_mut, elapsed.as_secs_f64());
        
        // 保存至完成批次
        self.completed_batches.push(summary.clone());
        
        info!("批处理执行完成: 总数 {}，成功 {}，失败 {}，取消 {}，耗时 {:.2}秒",
            summary.total_videos,
            summary.success_count,
            summary.failure_count,
            summary.canceled_count,
            summary.total_duration_seconds
        );
        
        Ok(summary)
    }
    
    /// 生成批处理摘要
    fn generate_summary(&self, processor: &BatchProcessor, elapsed: f64) -> BatchSummary {
        let all_progress = processor.get_all_progress();
        let all_results = processor.get_all_results();
        
        let total_videos = all_progress.len();
        let success_count = all_results.values()
            .filter(|r| r.is_ok())
            .count();
        let failure_count = all_results.values()
            .filter(|r| r.is_err())
            .count();
        let canceled_count = all_progress.values()
            .filter(|p| p.status == ExtractionStatus::Canceled)
            .count();
        
        // 计算错误摘要
        let mut error_summary = HashMap::new();
        for result in all_results.values() {
            if let Err(e) = result {
                let error_type = format!("{:?}", e);
                let count = error_summary.entry(error_type).or_insert(0);
                *count += 1;
            }
        }
        
        // 计算平均处理时间
        let avg_processing_seconds = if success_count > 0 {
            let total_processing_time: f64 = all_progress.values()
                .filter(|p| p.status == ExtractionStatus::Completed)
                .map(|p| p.elapsed_seconds)
                .sum();
            total_processing_time / success_count as f64
        } else {
            0.0
        };
        
        // 计算成功率
        let success_rate = if total_videos > 0 {
            success_count as f64 / total_videos as f64
        } else {
            0.0
        };
        
        // 计算处理速度(视频/分钟)
        let processing_speed = if elapsed > 0.0 {
            (success_count as f64 / elapsed) * 60.0
        } else {
            0.0
        };
        
        BatchSummary {
            total_videos,
            success_count,
            failure_count,
            canceled_count,
            total_duration_seconds: elapsed,
            avg_processing_seconds,
            error_summary,
            success_rate,
            processing_speed,
        }
    }
    
    /// 获取完成的批次摘要
    pub fn get_completed_batches(&self) -> &[BatchSummary] {
        &self.completed_batches
    }
    
    /// 获取最后一个批处理摘要
    pub fn get_last_batch_summary(&self) -> Option<&BatchSummary> {
        self.completed_batches.last()
    }
    
    /// 获取资源监控信息
    pub fn get_resource_metrics(&self) -> HashMap<String, String> {
        self.resource_monitor.get_report()
    }
    
    /// 执行批处理计划
    pub fn execute_plan(&mut self, plan: BatchProcessingPlan) -> Result<BatchSummary, VideoExtractionError> {
        let video_paths = plan.get_all_videos().to_vec();
        self.execute(video_paths)
    }
    
    /// 暂停当前批处理
    pub fn pause(&self) -> Result<(), VideoExtractionError> {
        let processor_ref = self.resource_monitor.get_processor()?;
        processor_ref.pause()
    }
    
    /// 恢复当前批处理
    pub fn resume(&self) -> Result<(), VideoExtractionError> {
        let processor_ref = self.resource_monitor.get_processor()?;
        processor_ref.resume()
    }
    
    /// 取消当前批处理
    pub fn cancel(&self) -> Result<(), VideoExtractionError> {
        let processor_ref = self.resource_monitor.get_processor()?;
        processor_ref.cancel()
    }
    
    /// 获取当前批处理状态
    pub fn get_state(&self) -> Result<BatchProcessingState, VideoExtractionError> {
        let processor_ref = self.resource_monitor.get_processor()?;
        Ok(processor_ref.get_state())
    }
    
    /// 获取当前批处理进度
    pub fn get_progress(&self) -> Result<f32, VideoExtractionError> {
        let processor_ref = self.resource_monitor.get_processor()?;
        Ok(processor_ref.get_overall_progress())
    }
}

// 扩展ResourceMonitor，添加获取当前处理器的方法
impl ResourceMonitor {
    pub fn get_processor(&self) -> Result<Arc<BatchProcessor>, VideoExtractionError> {
        if let Some(processor) = &self.processor {
            Ok(Arc::clone(processor))
        } else {
            Err(VideoExtractionError::ProcessingError("没有活动的批处理器".to_string()))
        }
    }
} 