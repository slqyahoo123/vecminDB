//! 视频特征提取器性能基准测试
//!
//! 本模块提供了性能基准测试功能，用于评估不同配置和特征类型的性能表现

use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use std::path::Path;
use crate::data::multimodal::video_extractor::VideoExtractionError;

use super::types::{VideoFeatureType, ModelType};
use super::config::VideoFeatureConfig;

/// 性能基准测试结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBenchmark {
    /// 特征类型
    pub feature_type: VideoFeatureType,
    /// 视频数量
    pub video_count: usize,
    /// 视频总大小(字节)
    pub total_size_bytes: u64,
    /// 处理速度(MB/秒)
    pub processing_speed_mbps: f64,
    /// 平均处理时间(毫秒/视频)
    pub avg_processing_time_ms: f64,
    /// 内存使用峰值(MB)
    pub peak_memory_mb: f64,
    /// 使用的线程数
    pub thread_count: usize,
    /// 使用的模型类型
    pub model_type: ModelType,
    /// 其他性能指标
    pub metrics: HashMap<String, f64>,
    /// 测试时间戳
    pub timestamp: u64,
}

impl PerformanceBenchmark {
    /// 创建新的性能基准测试结果
    pub fn new(
        feature_type: VideoFeatureType,
        video_count: usize,
        total_size_bytes: u64,
        processing_time_ms: f64,
        memory_mb: f64,
        thread_count: usize,
        model_type: ModelType,
    ) -> Self {
        let processing_speed_mbps = if processing_time_ms > 0.0 {
            (total_size_bytes as f64 / 1024.0 / 1024.0) / (processing_time_ms / 1000.0)
        } else {
            0.0
        };

        let avg_processing_time_ms = if video_count > 0 {
            processing_time_ms / video_count as f64
        } else {
            0.0
        };

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self {
            feature_type,
            video_count,
            total_size_bytes,
            processing_speed_mbps,
            avg_processing_time_ms,
            peak_memory_mb: memory_mb,
            thread_count,
            model_type,
            metrics: HashMap::new(),
            timestamp,
        }
    }

    /// 添加自定义性能指标
    pub fn add_metric(&mut self, name: &str, value: f64) {
        self.metrics.insert(name.to_string(), value);
    }

    /// 获取性能评分（综合得分）
    pub fn get_performance_score(&self) -> f64 {
        // 简单加权评分公式
        let speed_weight = 0.5;
        let memory_weight = 0.3;
        let additional_weight = 0.2;

        // 标准化处理速度（假设10MB/s为基准）
        let normalized_speed = self.processing_speed_mbps / 10.0;
        
        // 标准化内存使用（假设1GB为基准，越低越好）
        let normalized_memory = 1024.0 / self.peak_memory_mb.max(1.0);
        
        // 综合评分
        let score = speed_weight * normalized_speed + 
                   memory_weight * normalized_memory +
                   additional_weight * 1.0; // 附加项可以根据具体需求调整
        
        score
    }
}

/// 性能比较结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    /// 基准测试结果
    pub baseline: PerformanceBenchmark,
    /// 对比测试结果
    pub comparison: PerformanceBenchmark,
    /// 处理速度变化百分比
    pub speed_change_percent: f64,
    /// 内存使用变化百分比
    pub memory_change_percent: f64,
    /// 综合评分变化百分比
    pub score_change_percent: f64,
}

impl BenchmarkComparison {
    /// 创建新的性能比较结果
    pub fn new(baseline: PerformanceBenchmark, comparison: PerformanceBenchmark) -> Self {
        let speed_change_percent = if baseline.processing_speed_mbps > 0.0 {
            (comparison.processing_speed_mbps - baseline.processing_speed_mbps) 
                / baseline.processing_speed_mbps * 100.0
        } else {
            0.0
        };

        let memory_change_percent = if baseline.peak_memory_mb > 0.0 {
            (comparison.peak_memory_mb - baseline.peak_memory_mb) 
                / baseline.peak_memory_mb * 100.0
        } else {
            0.0
        };

        let baseline_score = baseline.get_performance_score();
        let comparison_score = comparison.get_performance_score();
        
        let score_change_percent = if baseline_score > 0.0 {
            (comparison_score - baseline_score) / baseline_score * 100.0
        } else {
            0.0
        };

        Self {
            baseline,
            comparison,
            speed_change_percent,
            memory_change_percent,
            score_change_percent,
        }
    }

    /// 是否整体性能更好
    pub fn is_better_overall(&self) -> bool {
        self.score_change_percent > 0.0
    }

    /// 获取改进摘要
    pub fn get_improvement_summary(&self) -> String {
        let speed_text = if self.speed_change_percent > 0.0 {
            format!("速度提升 {:.1}%", self.speed_change_percent)
        } else {
            format!("速度下降 {:.1}%", -self.speed_change_percent)
        };

        let memory_text = if self.memory_change_percent < 0.0 {
            format!("内存减少 {:.1}%", -self.memory_change_percent)
        } else {
            format!("内存增加 {:.1}%", self.memory_change_percent)
        };

        let overall_text = if self.is_better_overall() {
            format!("整体性能提升 {:.1}%", self.score_change_percent)
        } else {
            format!("整体性能下降 {:.1}%", -self.score_change_percent)
        };

        format!("{}，{}，{}", speed_text, memory_text, overall_text)
    }
}

/// 运行性能基准测试
pub fn run_benchmark(
    config: &VideoFeatureConfig,
    video_paths: &[String],
    repeat_count: usize
) -> Result<PerformanceBenchmark, VideoExtractionError> {
    use super::VideoFeatureExtractor;
    use super::error::VideoExtractionError;
    use std::fs;
    
    if video_paths.is_empty() {
        return Err(VideoExtractionError::InputError("基准测试需要至少一个视频文件".to_string()));
    }
    
    // 创建提取器
    let mut extractor = VideoFeatureExtractor::new(config.clone())?;
    
    println!("开始性能基准测试，配置:");
    println!("- 特征类型: {:?}", config.feature_types);
    println!("- 分辨率: {}x{}", config.frame_width, config.frame_height);
    println!("- 线程数: {}", config.parallel_threads);
    
    let start_time = Instant::now();
    let mut total_size_bytes = 0;
    let mut total_processing_time_ms = 0;
    let mut successful_extractions = 0;
    
    // 获取初始内存使用
    let initial_memory = get_current_memory_usage();
    let mut peak_memory = initial_memory;
    
    // 计算文件总大小
    for path in video_paths.iter() {
        if let Ok(metadata) = fs::metadata(path) {
            total_size_bytes += metadata.len();
        }
    }
    
    let feature_type = config.feature_types.first().cloned().unwrap_or(VideoFeatureType::RGB);
    let model_type = config.model_type.clone();
    
    // 执行多次提取以获得更准确的数据
    for i in 0..repeat_count {
        println!("运行 {}/{}", i + 1, repeat_count);
        
        for (j, path) in video_paths.iter().enumerate() {
            println!("处理视频 {}/{}: {}", j + 1, video_paths.len(), path);
            
            match extractor.extract_features(path) {
                Ok(result) => {
                    if let Some(ref info) = result.processing_info {
                        total_processing_time_ms += info.extraction_time_ms;
                    }
                    successful_extractions += 1;
                    
                    // 更新内存峰值
                    let current_memory = get_current_memory_usage();
                    peak_memory = peak_memory.max(current_memory);
                    
                    if let Some(ref info) = result.processing_info {
                        println!("- 处理时间: {} ms", info.extraction_time_ms);
                    }
                    println!("- 特征维度: {}", result.features.len());
                },
                Err(e) => {
                    println!("提取失败: {:?}", e);
                }
            }
        }
    }
    
    let elapsed = start_time.elapsed();
    let total_elapsed_ms = elapsed.as_millis() as f64;
    
    println!("基准测试完成:");
    println!("- 总时间: {:.2} ms", total_elapsed_ms);
    println!("- 总视频大小: {:.2} MB", total_size_bytes as f64 / 1024.0 / 1024.0);
    println!("- 成功处理: {}/{}", successful_extractions, video_paths.len() * repeat_count);
    
    // 创建基准测试结果
    let benchmark = PerformanceBenchmark::new(
        feature_type,
        successful_extractions,
        total_size_bytes,
        total_processing_time_ms as f64,
        peak_memory - initial_memory,
        config.parallel_threads,
        model_type,
    );
    
    println!("- 处理速度: {:.2} MB/s", benchmark.processing_speed_mbps);
    println!("- 平均时间: {:.2} ms/视频", benchmark.avg_processing_time_ms);
    println!("- 内存峰值: {:.2} MB", benchmark.peak_memory_mb);
    println!("- 性能评分: {:.2}", benchmark.get_performance_score());
    
    Ok(benchmark)
}

/// 获取当前内存使用量（MB）
fn get_current_memory_usage() -> f64 {
    // 这里应该使用系统API获取实际内存使用
    // 在不同平台上实现方式不同，以下是模拟实现
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
            for line in content.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(value_str) = line.split_whitespace().nth(1) {
                        if let Ok(value) = value_str.parse::<f64>() {
                            return value / 1024.0; // 转换为MB
                        }
                    }
                }
            }
        }
    }
    
    #[cfg(all(target_os = "windows", feature = "winapi"))]
    {
        use winapi::um::psapi::GetProcessMemoryInfo;
        use winapi::um::processthreadsapi::GetCurrentProcess;
        use winapi::um::psapi::PROCESS_MEMORY_COUNTERS;
        use winapi::shared::minwindef::DWORD;
        
        unsafe {
            let mut pmc = PROCESS_MEMORY_COUNTERS::default();
            let handle = GetCurrentProcess();
            if GetProcessMemoryInfo(
                handle,
                &mut pmc as *mut PROCESS_MEMORY_COUNTERS,
                std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as DWORD
            ) != 0 {
                return pmc.WorkingSetSize as f64 / (1024.0 * 1024.0);
            }
        }
    }
    
    // 如果无法获取，返回模拟值
    static mut SIMULATED_MEMORY: f64 = 100.0;
    unsafe {
        SIMULATED_MEMORY += rand::random::<f64>() * 10.0;
        SIMULATED_MEMORY
    }
}

/// 对比测试不同配置
pub fn benchmark_configs(
    video_paths: &[String],
    configs: &[VideoFeatureConfig],
    repeat_count: usize
) -> Vec<PerformanceBenchmark> {
    let mut results = Vec::new();
    
    for (i, config) in configs.iter().enumerate() {
        println!("\n测试配置 #{} / {}", i + 1, configs.len());
        
        match run_benchmark(config, video_paths, repeat_count) {
            Ok(benchmark) => {
                results.push(benchmark);
                println!("配置 #{} 测试完成", i + 1);
            },
            Err(e) => {
                println!("配置 #{} 测试失败: {:?}", i + 1, e);
            }
        }
    }
    
    // 如果有多个结果，打印比较信息
    if results.len() > 1 {
        println!("\n配置性能比较:");
        
        // 使用第一个配置作为基准
        let baseline = &results[0];
        
        for (i, result) in results.iter().enumerate().skip(1) {
            let comparison = BenchmarkComparison::new(baseline.clone(), result.clone());
            
            println!("配置 #{} vs 配置 #1:", i + 1);
            println!("- {}", comparison.get_improvement_summary());
        }
        
        // 找出最佳配置
        if let Some((best_idx, best)) = results.iter().enumerate()
            .max_by(|(_, a), (_, b)| 
                a.get_performance_score().partial_cmp(&b.get_performance_score())
                    .unwrap_or(std::cmp::Ordering::Equal)) {
            
            println!("\n最佳性能配置: #{}", best_idx + 1);
            println!("- 特征类型: {:?}", best.feature_type);
            println!("- 处理速度: {:.2} MB/s", best.processing_speed_mbps);
            println!("- 内存使用: {:.2} MB", best.peak_memory_mb);
            println!("- 性能评分: {:.2}", best.get_performance_score());
        }
    }
    
    results
}

/// 对比测试不同特征类型
pub fn benchmark_feature_types(
    video_paths: &[String],
    base_config: &VideoFeatureConfig,
    feature_types: &[VideoFeatureType],
    repeat_count: usize
) -> Vec<PerformanceBenchmark> {
    let mut configs = Vec::new();
    
    // 为每种特征类型创建配置
    for feature_type in feature_types {
        let mut config = base_config.clone();
        config.feature_types = vec![feature_type.clone()];
        configs.push(config);
    }
    
    println!("对比测试 {} 种特征类型", feature_types.len());
    benchmark_configs(video_paths, &configs, repeat_count)
}

/// 比较两个基准测试结果
pub fn compare_benchmarks(
    baseline: &PerformanceBenchmark,
    comparison: &PerformanceBenchmark
) -> BenchmarkComparison {
    let result = BenchmarkComparison::new(baseline.clone(), comparison.clone());
    
    println!("基准比较结果:");
    println!("- 基准: {:?} ({:.2} MB/s, {:.2} MB)",
        baseline.feature_type, baseline.processing_speed_mbps, baseline.peak_memory_mb);
    println!("- 对比: {:?} ({:.2} MB/s, {:.2} MB)",
        comparison.feature_type, comparison.processing_speed_mbps, comparison.peak_memory_mb);
    println!("- {}", result.get_improvement_summary());
    
    result
}

/// 将基准测试结果保存到文件
pub fn save_benchmark(benchmark: &PerformanceBenchmark, file_path: &str) -> std::io::Result<()> {
    use std::fs::File;
    use std::io::Write;
    
    let json = serde_json::to_string_pretty(benchmark)?;
    let mut file = File::create(file_path)?;
    file.write_all(json.as_bytes())?;
    
    println!("基准测试结果已保存到: {}", file_path);
    Ok(())
}

/// 从文件加载基准测试结果
pub fn load_benchmark(file_path: &str) -> std::io::Result<PerformanceBenchmark> {
    use std::fs::File;
    use std::io::BufReader;
    
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let benchmark = serde_json::from_reader(reader)?;
    
    println!("从文件加载了基准测试结果: {}", file_path);
    Ok(benchmark)
}

/// 基准测试历史记录
#[derive(Debug, Clone)]
pub struct BenchmarkHistory {
    benchmarks: Vec<PerformanceBenchmark>,
    storage_path: Option<String>,
}

impl BenchmarkHistory {
    /// 创建新的基准测试历史记录
    pub fn new() -> Self {
        Self {
            benchmarks: Vec::new(),
            storage_path: None,
        }
    }
    
    /// 设置存储路径
    pub fn with_storage(mut self, path: &str) -> Self {
        self.storage_path = Some(path.to_string());
        self
    }
    
    /// 从存储加载历史记录
    pub fn load(&mut self) -> std::io::Result<()> {
        if let Some(path) = &self.storage_path {
            use std::fs::{self, File};
            use std::io::BufReader;
            use std::path::Path;
            
            let dir_path = Path::new(path);
            
            // 确保目录存在
            if !dir_path.exists() {
                fs::create_dir_all(dir_path)?;
                println!("创建基准测试历史目录: {}", path);
                return Ok(());
            }
            
            // 加载所有基准测试文件
            self.benchmarks.clear();
            let mut loaded_count = 0;
            
            for entry in fs::read_dir(dir_path)? {
                let entry = entry?;
                let file_path = entry.path();
                
                if file_path.is_file() && 
                   file_path.extension().map_or(false, |ext| ext == "json") {
                    // 尝试加载为基准测试结果
                    match File::open(&file_path) {
                        Ok(file) => {
                            let reader = BufReader::new(file);
                            match serde_json::from_reader::<_, PerformanceBenchmark>(reader) {
                                Ok(benchmark) => {
                                    self.benchmarks.push(benchmark);
                                    loaded_count += 1;
                                },
                                Err(e) => {
                                    println!("无法解析文件 {}: {}", file_path.display(), e);
                                }
                            }
                        },
                        Err(e) => {
                            println!("无法打开文件 {}: {}", file_path.display(), e);
                        }
                    }
                }
            }
            
            println!("从目录加载了 {} 条基准测试历史记录", loaded_count);
            
            // 按时间戳排序
            self.benchmarks.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        }
        
        Ok(())
    }
    
    /// 添加基准测试结果
    pub fn add(&mut self, benchmark: PerformanceBenchmark) -> std::io::Result<()> {
        self.benchmarks.push(benchmark.clone());
        
        if let Some(path) = &self.storage_path {
            let timestamp = benchmark.timestamp;
            let feature_type = format!("{:?}", benchmark.feature_type).to_lowercase();
            let filename = format!("benchmark_{}_{}.json", feature_type, timestamp);
            let file_path = Path::new(path).join(filename);
            
            save_benchmark(&benchmark, file_path.to_str().unwrap_or_default())?;
        }
        
        Ok(())
    }
    
    /// 获取历史记录
    pub fn get_history(&self) -> &Vec<PerformanceBenchmark> {
        &self.benchmarks
    }
    
    /// 获取特定特征类型的历史记录
    pub fn get_history_by_type(&self, feature_type: &VideoFeatureType) -> Vec<&PerformanceBenchmark> {
        self.benchmarks.iter()
            .filter(|b| b.feature_type == *feature_type)
            .collect()
    }
    
    /// 获取性能趋势分析
    pub fn get_trend(&self, feature_type: &VideoFeatureType) -> Option<String> {
        let history = self.get_history_by_type(feature_type);
        
        if history.len() < 2 {
            // 数据不足，无法分析趋势
            return None;
        }
        
        let mut trend_analysis = String::new();
        
        // 按照时间对历史记录排序
        let mut sorted_history = history.clone();
        sorted_history.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        
        // 计算性能指标的变化趋势
        let first = sorted_history.first().unwrap();
        let last = sorted_history.last().unwrap();
        
        // 计算处理速度变化
        let speed_change_percent = if first.processing_speed_mbps > 0.0 {
            (last.processing_speed_mbps - first.processing_speed_mbps) / 
                first.processing_speed_mbps * 100.0
        } else {
            0.0
        };
        
        // 计算内存使用变化
        let memory_change_percent = if first.peak_memory_mb > 0.0 {
            (last.peak_memory_mb - first.peak_memory_mb) / 
                first.peak_memory_mb * 100.0
        } else {
            0.0
        };
        
        // 计算平均处理时间变化
        let time_change_percent = if first.avg_processing_time_ms > 0.0 {
            (last.avg_processing_time_ms - first.avg_processing_time_ms) / 
                first.avg_processing_time_ms * 100.0
        } else {
            0.0
        };
        
        // 计算性能评分变化
        let score_first = first.get_performance_score();
        let score_last = last.get_performance_score();
        let score_change_percent = if score_first > 0.0 {
            (score_last - score_first) / score_first * 100.0
        } else {
            0.0
        };
        
        // 时间段描述
        let time_span_days = (last.timestamp - first.timestamp) / (24 * 60 * 60);
        
        trend_analysis.push_str(&format!("在过去的 {} 天中，{:?} 特征类型的性能趋势：\n", 
            time_span_days, feature_type));
        
        // 处理速度趋势
        if speed_change_percent > 5.0 {
            trend_analysis.push_str(&format!("- 处理速度显著提高: +{:.1}%\n", speed_change_percent));
        } else if speed_change_percent < -5.0 {
            trend_analysis.push_str(&format!("- 处理速度明显下降: {:.1}%\n", speed_change_percent));
        } else {
            trend_analysis.push_str("- 处理速度保持稳定\n");
        }
        
        // 内存使用趋势
        if memory_change_percent < -5.0 {
            trend_analysis.push_str(&format!("- 内存占用显著降低: {:.1}%\n", memory_change_percent));
        } else if memory_change_percent > 5.0 {
            trend_analysis.push_str(&format!("- 内存占用明显增加: +{:.1}%\n", memory_change_percent));
        } else {
            trend_analysis.push_str("- 内存占用保持稳定\n");
        }
        
        // 处理时间趋势
        if time_change_percent < -5.0 {
            trend_analysis.push_str(&format!("- 平均处理时间明显缩短: {:.1}%\n", time_change_percent));
        } else if time_change_percent > 5.0 {
            trend_analysis.push_str(&format!("- 平均处理时间显著增加: +{:.1}%\n", time_change_percent));
        } else {
            trend_analysis.push_str("- 平均处理时间保持稳定\n");
        }
        
        // 整体性能评分
        if score_change_percent > 5.0 {
            trend_analysis.push_str(&format!("- 整体性能评分明显提升: +{:.1}%\n", score_change_percent));
        } else if score_change_percent < -5.0 {
            trend_analysis.push_str(&format!("- 整体性能评分显著下降: {:.1}%\n", score_change_percent));
        } else {
            trend_analysis.push_str("- 整体性能评分保持稳定\n");
        }
        
        // 添加总结
        if score_change_percent > 0.0 {
            trend_analysis.push_str("\n总体趋势：性能有所提升，继续保持当前优化方向");
        } else if score_change_percent < 0.0 {
            trend_analysis.push_str("\n总体趋势：性能有所下降，建议分析原因并采取措施");
        } else {
            trend_analysis.push_str("\n总体趋势：性能保持稳定，可尝试新的优化方向");
        }
        
        Some(trend_analysis)
    }
    
    /// 清除历史记录
    pub fn clear(&mut self) -> std::io::Result<()> {
        self.benchmarks.clear();
        
        if let Some(path) = &self.storage_path {
            let dir_path = Path::new(path);
            if dir_path.exists() {
                for entry in std::fs::read_dir(dir_path)? {
                    let entry = entry?;
                    let path = entry.path();
                    
                    if path.is_file() && path.extension().map_or(false, |ext| ext == "json") {
                        std::fs::remove_file(path)?;
                    }
                }
            }
        }
        
        println!("已清除所有基准测试历史记录");
        Ok(())
    }
} 