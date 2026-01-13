//! 特征提取器基准测试
//! 
//! 提供特征提取器性能测试和诊断功能

use super::VideoFeatureExtractor;
use crate::data::multimodal::video_extractor::config::VideoFeatureConfig;
use crate::data::multimodal::video_extractor::error::VideoExtractionError;
use crate::data::multimodal::video_extractor::types::VideoFeatureType;
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};
use log::{info, error, debug};

/// 基准测试结果
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub extractor_name: String,
    pub iterations: usize,
    pub success_count: usize,
    pub avg_time: Duration,
    pub results: Vec<Result<Duration, VideoExtractionError>>,
    pub config: VideoFeatureConfig,
}

impl BenchmarkResult {
    /// 计算性能统计信息
    pub fn get_statistics(&self) -> BenchmarkStatistics {
        let mut durations = Vec::new();
        for result in &self.results {
            if let Ok(duration) = result {
                durations.push(duration.as_millis() as u64);
            }
        }
        
        durations.sort();
        let min = durations.first().copied().unwrap_or(0);
        let max = durations.last().copied().unwrap_or(0);
        let median = if !durations.is_empty() {
            durations[durations.len() / 2]
        } else {
            0
        };
        
        let mean = if !durations.is_empty() {
            durations.iter().sum::<u64>() / durations.len() as u64
        } else {
            0
        };
        
        let variance = if durations.len() > 1 {
            let squared_diffs: u64 = durations.iter()
                .map(|&d| {
                    let diff = if d > mean { d - mean } else { mean - d };
                    diff * diff
                })
                .sum();
            squared_diffs / (durations.len() - 1) as u64
        } else {
            0
        };
        
        let std_dev = (variance as f64).sqrt() as u64;
        
        BenchmarkStatistics {
            success_rate: if self.iterations > 0 {
                self.success_count as f64 / self.iterations as f64
            } else {
                0.0
            },
            min_time_ms: min,
            max_time_ms: max,
            median_time_ms: median,
            mean_time_ms: mean,
            std_dev_ms: std_dev,
        }
    }
    
    /// 判断基准测试是否稳定
    pub fn is_stable(&self, threshold: f64) -> bool {
        let stats = self.get_statistics();
        if stats.mean_time_ms == 0 {
            return false;
        }
        
        let variation = stats.std_dev_ms as f64 / stats.mean_time_ms as f64;
        variation < threshold
    }
    
    /// 输出基准测试报告
    pub fn generate_report(&self) -> String {
        let stats = self.get_statistics();
        let mut report = String::new();
        
        report.push_str(&format!("提取器: {}\n", self.extractor_name));
        report.push_str(&format!("迭代次数: {}\n", self.iterations));
        report.push_str(&format!("成功次数: {} ({}%)\n", self.success_count, (stats.success_rate * 100.0) as u64));
        report.push_str(&format!("平均时间: {}ms\n", stats.mean_time_ms));
        report.push_str(&format!("中位数时间: {}ms\n", stats.median_time_ms));
        report.push_str(&format!("最小时间: {}ms\n", stats.min_time_ms));
        report.push_str(&format!("最大时间: {}ms\n", stats.max_time_ms));
        report.push_str(&format!("标准差: {}ms\n", stats.std_dev_ms));
        report.push_str(&format!("变异系数: {:.2}%\n", (stats.std_dev_ms as f64 / stats.mean_time_ms as f64) * 100.0));
        report.push_str(&format!("稳定性评估: {}\n", if self.is_stable(0.1) { "稳定" } else { "不稳定" }));
        
        report
    }
}

/// 基准测试统计信息
#[derive(Debug, Clone)]
pub struct BenchmarkStatistics {
    pub success_rate: f64,
    pub min_time_ms: u64,
    pub max_time_ms: u64,
    pub median_time_ms: u64,
    pub mean_time_ms: u64,
    pub std_dev_ms: u64,
}

/// 诊断信息
#[derive(Debug, Clone)]
pub struct DiagnosticInfo {
    pub extractor_name: String,
    pub is_available: bool,
    pub supported_features: Vec<VideoFeatureType>,
    pub error_details: Option<String>,
    pub system_info: HashMap<String, String>,
    pub recommendations: Vec<String>,
    pub resource_usage: ResourceUsage,
    pub hardware_capabilities: HardwareCapabilities,
}

/// 系统资源使用情况
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub gpu_usage_percent: Option<f64>,
    pub gpu_memory_usage_mb: Option<f64>,
    pub disk_io_mbps: f64,
    pub thread_count: usize,
    pub sample_time: u64,
}

/// 硬件能力信息
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    pub cpu_model: String,
    pub cpu_features: Vec<String>,
    pub total_memory_mb: u64,
    pub gpu_model: Option<String>,
    pub gpu_memory_mb: Option<u64>,
    pub gpu_capabilities: Option<Vec<String>>,
    pub disk_type: String,
    pub disk_speed_mbps: Option<f64>,
}

impl DiagnosticInfo {
    /// 添加建议
    pub fn add_recommendation(&mut self, recommendation: String) {
        self.recommendations.push(recommendation);
    }
    
    /// 设置错误详情
    pub fn set_error_details(&mut self, details: String) {
        self.error_details = Some(details);
    }
    
    /// 获取可能导致性能问题的因素
    pub fn get_performance_bottlenecks(&self) -> Vec<String> {
        let mut bottlenecks = Vec::new();
        
        // 检查CPU使用率
        if self.resource_usage.cpu_usage_percent > 80.0 {
            bottlenecks.push(format!("CPU使用率过高({}%)，可能导致性能瓶颈", 
                                     self.resource_usage.cpu_usage_percent));
        }
        
        // 检查内存使用率
        let memory_usage_percent = if let Some(total) = self.system_info.get("memory_total_mb") {
            if let Ok(total_mb) = total.parse::<f64>() {
                if total_mb > 0.0 {
                    (self.resource_usage.memory_usage_mb / total_mb) * 100.0
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        if memory_usage_percent > 80.0 {
            bottlenecks.push(format!("内存使用率过高({}%)，可能导致性能瓶颈", 
                                     memory_usage_percent));
        }
        
        // 检查GPU使用率
        if let Some(gpu_usage) = self.resource_usage.gpu_usage_percent {
            if gpu_usage > 80.0 {
                bottlenecks.push(format!("GPU使用率过高({}%)，可能导致性能瓶颈", 
                                         gpu_usage));
            }
        }
        
        // 检查磁盘IO
        if self.resource_usage.disk_io_mbps > 100.0 {
            bottlenecks.push(format!("磁盘IO过高({}MB/s)，可能导致性能瓶颈", 
                                     self.resource_usage.disk_io_mbps));
        }
        
        bottlenecks
    }
    
    /// 生成性能优化建议
    pub fn generate_optimization_suggestions(&self) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        // 根据当前硬件和资源使用情况提供优化建议
        let bottlenecks = self.get_performance_bottlenecks();
        if !bottlenecks.is_empty() {
            for bottleneck in &bottlenecks {
                suggestions.push(format!("解决性能瓶颈: {}", bottleneck));
            }
        }
        
        // 检查是否有GPU但未使用
        if self.hardware_capabilities.gpu_model.is_some() && 
           (self.resource_usage.gpu_usage_percent.is_none() || 
            self.resource_usage.gpu_usage_percent.unwrap() < 10.0) {
            suggestions.push("检测到GPU但未充分利用，考虑启用GPU加速".to_string());
        }
        
        // 内存优化建议
        if self.resource_usage.memory_usage_mb > 1000.0 {
            suggestions.push("内存使用较高，考虑优化内存管理或增加批处理大小".to_string());
        }
        
        // 线程优化建议
        let cpu_cores = if let Some(cores) = self.system_info.get("cpu_cores") {
            cores.parse::<usize>().unwrap_or(1)
        } else {
            1
        };
        
        if self.resource_usage.thread_count < cpu_cores / 2 {
            suggestions.push(format!("线程数({})少于可用CPU核心数({}),考虑增加并行度",
                                    self.resource_usage.thread_count, cpu_cores));
        } else if self.resource_usage.thread_count > cpu_cores * 2 {
            suggestions.push(format!("线程数({})远高于CPU核心数({}),可能导致线程切换开销",
                                    self.resource_usage.thread_count, cpu_cores));
        }
        
        suggestions
    }
    
    /// 生成诊断报告
    pub fn generate_diagnosis_report(&self) -> String {
        let mut report = String::new();
        
        // 基本信息
        report.push_str(&format!("## 提取器诊断报告: {}\n\n", self.extractor_name));
        report.push_str(&format!("可用状态: {}\n", if self.is_available { "可用" } else { "不可用" }));
        
        // 支持的特征类型
        if !self.supported_features.is_empty() {
            report.push_str("\n### 支持的特征类型:\n");
            for feature in &self.supported_features {
                report.push_str(&format!("- {:?}\n", feature));
            }
        }
        
        // 错误详情
        if let Some(ref error) = self.error_details {
            report.push_str("\n### 错误详情:\n");
            report.push_str(&format!("{}\n", error));
        }
        
        // 系统信息
        report.push_str("\n### 系统信息:\n");
        for (key, value) in &self.system_info {
            report.push_str(&format!("- {}: {}\n", key, value));
        }
        
        // 硬件能力
        report.push_str("\n### 硬件能力:\n");
        report.push_str(&format!("- CPU型号: {}\n", self.hardware_capabilities.cpu_model));
        report.push_str(&format!("- 内存总量: {}MB\n", self.hardware_capabilities.total_memory_mb));
        
        if let Some(ref gpu_model) = self.hardware_capabilities.gpu_model {
            report.push_str(&format!("- GPU型号: {}\n", gpu_model));
            if let Some(gpu_memory) = self.hardware_capabilities.gpu_memory_mb {
                report.push_str(&format!("- GPU内存: {}MB\n", gpu_memory));
            }
        }
        
        report.push_str(&format!("- 磁盘类型: {}\n", self.hardware_capabilities.disk_type));
        
        // 资源使用情况
        report.push_str("\n### 资源使用情况:\n");
        report.push_str(&format!("- CPU使用率: {:.2}%\n", self.resource_usage.cpu_usage_percent));
        report.push_str(&format!("- 内存使用: {:.2}MB\n", self.resource_usage.memory_usage_mb));
        if let Some(gpu_usage) = self.resource_usage.gpu_usage_percent {
            report.push_str(&format!("- GPU使用率: {:.2}%\n", gpu_usage));
        }
        if let Some(gpu_memory) = self.resource_usage.gpu_memory_usage_mb {
            report.push_str(&format!("- GPU内存使用: {:.2}MB\n", gpu_memory));
        }
        report.push_str(&format!("- 磁盘IO: {:.2}MB/s\n", self.resource_usage.disk_io_mbps));
        report.push_str(&format!("- 线程数: {}\n", self.resource_usage.thread_count));
        
        // 性能瓶颈
        let bottlenecks = self.get_performance_bottlenecks();
        if !bottlenecks.is_empty() {
            report.push_str("\n### 性能瓶颈:\n");
            for bottleneck in bottlenecks {
                report.push_str(&format!("- {}\n", bottleneck));
            }
        }
        
        // 优化建议
        if !self.recommendations.is_empty() {
            report.push_str("\n### 优化建议:\n");
            for recommendation in &self.recommendations {
                report.push_str(&format!("- {}\n", recommendation));
            }
        }
        
        report
    }
}

/// 执行基准测试
pub fn run_benchmark(
    extractor: &mut dyn VideoFeatureExtractor,
    video_path: &Path,
    config: &VideoFeatureConfig,
    iterations: usize
) -> Result<BenchmarkResult, VideoExtractionError> {
    info!("开始对'{}'执行{}次基准测试", video_path.display(), iterations);
    let mut total_time = Duration::new(0, 0);
    let mut success_count = 0;
    let mut results = Vec::with_capacity(iterations);
    
    // 确保提取器已初始化
    if !extractor.is_available() {
        extractor.initialize()?;
    }
    
    for i in 0..iterations {
        let start = Instant::now();
        match extractor.extract_features(video_path, config) {
            Ok(_) => {
                let duration = start.elapsed();
                total_time += duration;
                success_count += 1;
                results.push(Ok(duration));
                debug!("基准测试迭代 {}/{}: {:?}", i+1, iterations, duration);
            },
            Err(e) => {
                error!("基准测试迭代 {}/{} 失败: {}", i+1, iterations, e);
                results.push(Err(e));
            }
        }
    }
    
    let avg_time = if success_count > 0 {
        total_time / success_count as u32
    } else {
        Duration::new(0, 0)
    };
    
    Ok(BenchmarkResult {
        extractor_name: extractor.name().to_string(),
        iterations,
        success_count,
        avg_time,
        results,
        config: config.clone(),
    })
}

/// 诊断提取器问题
pub fn diagnose_extractor(extractor: &mut dyn VideoFeatureExtractor) -> Result<DiagnosticInfo, VideoExtractionError> {
    info!("开始诊断'{}'提取器", extractor.name());
    // 基本诊断信息
    let mut diagnostics = DiagnosticInfo {
        extractor_name: extractor.name().to_string(),
        is_available: extractor.is_available(),
        supported_features: extractor.supported_features(),
        error_details: None,
        system_info: collect_system_info(),
        recommendations: Vec::new(),
        resource_usage: collect_resource_usage(),
        hardware_capabilities: detect_hardware_capabilities(),
    };
    
    // 尝试初始化
    if !diagnostics.is_available {
        match extractor.initialize() {
            Ok(_) => {
                diagnostics.is_available = true;
                diagnostics.add_recommendation("提取器已成功初始化".to_string());
            },
            Err(e) => {
                diagnostics.set_error_details(format!("初始化失败: {}", e));
                diagnostics.add_recommendation("检查初始化参数和依赖项".to_string());
            }
        }
    }
    
    // 生成建议
    if !diagnostics.is_available {
        diagnostics.add_recommendation("提取器不可用，请检查依赖库是否正确安装".to_string());
    }
    
    if diagnostics.supported_features.is_empty() {
        diagnostics.add_recommendation("提取器未声明支持的特征类型，可能未正确实现接口".to_string());
    }
    
    // 添加性能优化建议
    for suggestion in diagnostics.generate_optimization_suggestions() {
        diagnostics.add_recommendation(suggestion);
    }
    
    // 生成诊断报告
    let report = diagnostics.generate_diagnosis_report();
    debug!("诊断报告已生成");
    
    Ok(diagnostics)
}

/// 收集系统信息
fn collect_system_info() -> HashMap<String, String> {
    let mut info = HashMap::new();
    
    // 收集基本系统信息
    info.insert("os".to_string(), std::env::consts::OS.to_string());
    info.insert("cpu_cores".to_string(), num_cpus::get().to_string());
    
    // 内存信息
    #[cfg(target_os = "linux")]
    {
        if let Ok(mem_info) = std::fs::read_to_string("/proc/meminfo") {
            if let Some(line) = mem_info.lines().find(|l| l.starts_with("MemTotal:")) {
                if let Some(mem_kb) = line.split_whitespace().nth(1) {
                    info.insert("memory_total_kb".to_string(), mem_kb.to_string());
                }
            }
        }
    }
    
    // Rust版本
    info.insert("rust_version".to_string(), env!("CARGO_PKG_RUST_VERSION", "unknown").to_string());
    
    // 可用磁盘空间
    if let Ok(path) = std::env::current_dir() {
        if let Ok(available_space) = available_space_for_path(&path) {
            info.insert("available_disk_space_mb".to_string(), format!("{}", available_space / (1024 * 1024)));
        }
    }
    
    info
}

/// 获取指定路径的可用空间
fn available_space_for_path(path: &Path) -> Result<u64, std::io::Error> {
    #[cfg(target_family = "unix")]
    {
        #[cfg(unix)]
use std::os::unix::fs::MetadataExt;
        let fs_stats = std::fs::metadata(path)?;
        Ok(fs_stats.blocks() * 512)
    }
    
    #[cfg(all(target_family = "windows", feature = "winapi"))]
    {
        use winapi::um::fileapi::GetDiskFreeSpaceExW;
        use winapi::shared::ntdef::PULARGE_INTEGER;
        
        // 将路径转换为宽字符字符串
        let path_str = path.to_string_lossy().to_string();
        let mut wide_path: Vec<u16> = path_str.encode_utf16().collect();
        wide_path.push(0); // 确保以null结尾
        
        // 声明存储结果的变量
        let mut free_bytes_available = 0u64;
        let mut total_number_of_bytes = 0u64;
        let mut total_number_of_free_bytes = 0u64;
        
        // 调用Windows API获取磁盘空间信息
        let result = unsafe {
            GetDiskFreeSpaceExW(
                wide_path.as_ptr(),
                &mut free_bytes_available as *mut u64 as PULARGE_INTEGER,
                &mut total_number_of_bytes as *mut u64 as PULARGE_INTEGER,
                &mut total_number_of_free_bytes as *mut u64 as PULARGE_INTEGER
            )
        };
        
        if result == 0 {
            // 获取失败时返回最后一个错误
            return Err(std::io::Error::last_os_error());
        }
        
        Ok(free_bytes_available)
    }
    
    #[cfg(not(any(target_family = "unix", target_family = "windows")))]
    {
        Ok(0)
    }
}

/// 收集当前系统资源使用情况
fn collect_resource_usage() -> ResourceUsage {
    let mut usage = ResourceUsage {
        cpu_usage_percent: 0.0,
        memory_usage_mb: 0.0,
        gpu_usage_percent: None,
        gpu_memory_usage_mb: None,
        disk_io_mbps: 0.0,
        thread_count: num_cpus::get(),
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
                    let idle = values[3];
                    let total: u64 = values.iter().sum();
                    
                    // 采样一小段时间后再次测量
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    
                    if let Ok(cpu_info2) = std::fs::read_to_string("/proc/stat") {
                        if let Some(cpu_line2) = cpu_info2.lines().next() {
                            let values2: Vec<u64> = cpu_line2.split_whitespace()
                                .skip(1)
                                .filter_map(|v| v.parse::<u64>().ok())
                                .collect();
                            
                            if values2.len() >= 4 {
                                let idle2 = values2[3];
                                let total2: u64 = values2.iter().sum();
                                
                                let idle_diff = idle2 - idle;
                                let total_diff = total2 - total;
                                
                                if total_diff > 0 {
                                    usage.cpu_usage_percent = 100.0 * (1.0 - (idle_diff as f64 / total_diff as f64));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // 获取内存使用情况
    #[cfg(target_os = "linux")]
    {
        if let Ok(mem_info) = std::fs::read_to_string("/proc/meminfo") {
            let mut mem_total = 0u64;
            let mut mem_free = 0u64;
            let mut mem_available = 0u64;
            
            for line in mem_info.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        mem_total = value.parse::<u64>().unwrap_or(0);
                    }
                } else if line.starts_with("MemFree:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        mem_free = value.parse::<u64>().unwrap_or(0);
                    }
                } else if line.starts_with("MemAvailable:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        mem_available = value.parse::<u64>().unwrap_or(0);
                    }
                }
            }
            
            if mem_total > 0 && mem_available > 0 {
                let mem_used = mem_total - mem_available;
                usage.memory_usage_mb = mem_used as f64 / 1024.0; // KB to MB
            }
        }
    }
    
    #[cfg(all(target_os = "windows", feature = "winapi"))]
    {
        use winapi::um::sysinfoapi::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        use winapi::um::processthreadsapi::{GetCurrentProcess, GetProcessTimes};
        use winapi::um::psapi::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
        
        // 获取系统内存使用情况
        unsafe {
            let mut memory_status = MEMORYSTATUSEX {
                dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
                ..std::mem::zeroed()
            };
            
            if GlobalMemoryStatusEx(&mut memory_status) != 0 {
                usage.memory_usage_mb = (memory_status.ullTotalPhys - memory_status.ullAvailPhys) as f64 / (1024.0 * 1024.0);
            }
            
            // 获取进程CPU和内存使用情况
            let process_handle = GetCurrentProcess();
            let mut process_memory = std::mem::zeroed::<PROCESS_MEMORY_COUNTERS>();
            
            if GetProcessMemoryInfo(
                process_handle,
                &mut process_memory as *mut PROCESS_MEMORY_COUNTERS,
                std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32
            ) != 0 {
                usage.memory_usage_mb = process_memory.WorkingSetSize as f64 / (1024.0 * 1024.0);
            }
            
            // CPU使用情况测量
            use winapi::shared::minwindef::FILETIME;
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
                
                // 初始CPU时间
                let cpu_time1 = filetime_to_u64(&kernel_time) + filetime_to_u64(&user_time);
                
                // 等待一小段时间
                std::thread::sleep(std::time::Duration::from_millis(100));
                
                // 再次获取CPU时间
                let mut kernel_time2 = std::mem::zeroed::<FILETIME>();
                let mut user_time2 = std::mem::zeroed::<FILETIME>();
                
                if GetProcessTimes(
                    process_handle,
                    &mut creation_time,
                    &mut exit_time,
                    &mut kernel_time2,
                    &mut user_time2
                ) != 0 {
                    let cpu_time2 = filetime_to_u64(&kernel_time2) + filetime_to_u64(&user_time2);
                    let elapsed_cpu = cpu_time2 - cpu_time1;
                    
                    // 转换为CPU使用百分比 (100ms = 1,000,000 * 100ns单位)
                    // Windows的时间单位是100纳秒
                    usage.cpu_usage_percent = (elapsed_cpu as f64 / 1_000_000.0) * 10.0;
                }
            }
        }
    }

    // 尝试获取GPU使用情况 - NVML为例
    #[cfg(feature = "nvml")]
    {
        if let Ok(_) = nvml_sys::nvml::nvml_init() {
            let mut device_count = 0u32;
            if nvml_sys::nvml::nvml_device_get_count(&mut device_count) == nvml_sys::nvml::NVML_SUCCESS {
                if device_count > 0 {
                    let mut device = std::ptr::null_mut();
                    if nvml_sys::nvml::nvml_device_get_handle_by_index(0, &mut device) == nvml_sys::nvml::NVML_SUCCESS {
                        // 获取GPU使用率
                        let mut utilization = nvml_sys::nvml::nvmlUtilization_t {
                            gpu: 0,
                            memory: 0
                        };
                        if nvml_sys::nvml::nvml_device_get_utilization_rates(device, &mut utilization) == nvml_sys::nvml::NVML_SUCCESS {
                            usage.gpu_usage_percent = Some(utilization.gpu as f64);
                            
                            // 获取GPU内存使用情况
                            let mut memory = nvml_sys::nvml::nvmlMemory_t {
                                total: 0,
                                free: 0,
                                used: 0
                            };
                            if nvml_sys::nvml::nvml_device_get_memory_info(device, &mut memory) == nvml_sys::nvml::NVML_SUCCESS {
                                usage.gpu_memory_usage_mb = Some(memory.used as f64 / (1024.0 * 1024.0));
                            }
                        }
                    }
                }
            }
            nvml_sys::nvml::nvml_shutdown();
        }
    }
    
    // 如果没有启用NVML特性，使用简化的实现
    #[cfg(not(feature = "nvml"))]
    {
        // 在Linux上尝试使用nvidia-smi命令
        #[cfg(target_os = "linux")]
        {
            use std::process::Command;
            
            if let Ok(output) = Command::new("nvidia-smi")
                .args(&["--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"])
                .output() {
                if output.status.success() {
                    if let Ok(output_str) = String::from_utf8(output.stdout) {
                        let parts: Vec<&str> = output_str.trim().split(',').collect();
                        if parts.len() >= 2 {
                            if let Ok(gpu_util) = parts[0].trim().parse::<f64>() {
                                usage.gpu_usage_percent = Some(gpu_util);
                            }
                            if let Ok(gpu_mem) = parts[1].trim().parse::<f64>() {
                                usage.gpu_memory_usage_mb = Some(gpu_mem);
                            }
                        }
                    }
                }
            }
        }
        
        // 在Windows上尝试使用typeperf
        #[cfg(target_os = "windows")]
        {
            use std::process::Command;
            
            if let Ok(output) = Command::new("typeperf")
                .args(&["-sc", "1", "\\GPU Engine(*)\\Utilization Percentage"])
                .output() {
                if output.status.success() {
                    if let Ok(output_str) = String::from_utf8(output.stdout) {
                        let lines: Vec<&str> = output_str.lines().collect();
                        if lines.len() >= 2 {
                            let data_line = lines[1];
                            let parts: Vec<&str> = data_line.split(',').collect();
                            if parts.len() >= 2 {
                                if let Ok(gpu_util) = parts[1].trim_matches('"').parse::<f64>() {
                                    usage.gpu_usage_percent = Some(gpu_util);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // 获取磁盘IO统计信息
    #[cfg(target_os = "linux")]
    {
        if let Ok(disk_stats) = std::fs::read_to_string("/proc/diskstats") {
            let mut total_read_sectors = 0u64;
            let mut total_write_sectors = 0u64;
            
            for line in disk_stats.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 14 {
                    if let (Ok(read_sectors), Ok(write_sectors)) = (parts[5].parse::<u64>(), parts[9].parse::<u64>()) {
                        total_read_sectors += read_sectors;
                        total_write_sectors += write_sectors;
                    }
                }
            }
            
            // 采样一段时间后再次测量
            std::thread::sleep(std::time::Duration::from_millis(100));
            
            if let Ok(disk_stats2) = std::fs::read_to_string("/proc/diskstats") {
                let mut total_read_sectors2 = 0u64;
                let mut total_write_sectors2 = 0u64;
                
                for line in disk_stats2.lines() {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 14 {
                        if let (Ok(read_sectors), Ok(write_sectors)) = (parts[5].parse::<u64>(), parts[9].parse::<u64>()) {
                            total_read_sectors2 += read_sectors;
                            total_write_sectors2 += write_sectors;
                        }
                    }
                }
                
                // 计算100ms内的IO速率
                let read_sectors_diff = total_read_sectors2 - total_read_sectors;
                let write_sectors_diff = total_write_sectors2 - total_write_sectors;
                
                // 扇区通常为512字节
                let bytes_per_second = (read_sectors_diff + write_sectors_diff) * 512 * 10; // *10因为观测了100ms
                usage.disk_io_mbps = bytes_per_second as f64 / (1024.0 * 1024.0);
            }
        }
    }
    
    // Windows磁盘IO统计
    #[cfg(target_os = "windows")]
    {
        use std::process::Command;
        
        if let Ok(output) = Command::new("typeperf")
            .args(&["-sc", "1", "\\PhysicalDisk(_Total)\\Disk Bytes/sec"])
            .output() {
            if output.status.success() {
                if let Ok(output_str) = String::from_utf8(output.stdout) {
                    let lines: Vec<&str> = output_str.lines().collect();
                    if lines.len() >= 2 {
                        let data_line = lines[1];
                        let parts: Vec<&str> = data_line.split(',').collect();
                        if parts.len() >= 2 {
                            if let Ok(io_bytes) = parts[1].trim_matches('"').parse::<f64>() {
                                usage.disk_io_mbps = io_bytes / (1024.0 * 1024.0);
                            }
                        }
                    }
                }
            }
        }
    }
    
    usage
}

/// 检测硬件能力
fn detect_hardware_capabilities() -> HardwareCapabilities {
    let mut capabilities = HardwareCapabilities {
        cpu_model: "Unknown".to_string(),
        cpu_features: Vec::new(),
        total_memory_mb: 0,
        gpu_model: None,
        gpu_memory_mb: None,
        gpu_capabilities: None,
        disk_type: "Unknown".to_string(),
        disk_speed_mbps: None,
    };
    
    // CPU信息
    #[cfg(target_os = "linux")]
    {
        if let Ok(cpu_info) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in cpu_info.lines() {
                if line.starts_with("model name") {
                    if let Some(model) = line.split(':').nth(1) {
                        capabilities.cpu_model = model.trim().to_string();
                    }
                } else if line.starts_with("flags") {
                    if let Some(flags) = line.split(':').nth(1) {
                        capabilities.cpu_features = flags
                            .trim()
                            .split_whitespace()
                            .map(|s| s.to_string())
                            .collect();
                    }
                }
            }
        }
        
        // 内存信息
        if let Ok(mem_info) = std::fs::read_to_string("/proc/meminfo") {
            if let Some(line) = mem_info.lines().find(|l| l.starts_with("MemTotal:")) {
                if let Some(mem_kb) = line.split_whitespace().nth(1) {
                    if let Ok(mem) = mem_kb.parse::<u64>() {
                        capabilities.total_memory_mb = mem / 1024;
                    }
                }
            }
        }
    }
    
    #[cfg(all(target_os = "windows", feature = "winapi"))]
    {
        use winapi::um::sysinfoapi::{GetSystemInfo, SYSTEM_INFO};
        use winapi::um::sysinfoapi::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        
        unsafe {
            // CPU信息
            let mut system_info: SYSTEM_INFO = std::mem::zeroed();
            GetSystemInfo(&mut system_info);
            
            // 使用WMI查询CPU模型和功能
            if let Ok(output) = std::process::Command::new("wmic")
                .args(&["cpu", "get", "Name,NumberOfCores,NumberOfLogicalProcessors", "/format:csv"])
                .output() {
                if let Ok(output_str) = String::from_utf8(output.stdout) {
                    let lines: Vec<&str> = output_str.lines().collect();
                    if lines.len() >= 2 {
                        let parts: Vec<&str> = lines[1].split(',').collect();
                        if parts.len() >= 4 {
                            capabilities.cpu_model = parts[1].to_string();
                        }
                    }
                }
            }
            
            // 内存信息
            let mut memory_status = MEMORYSTATUSEX {
                dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
                ..std::mem::zeroed()
            };
            
            if GlobalMemoryStatusEx(&mut memory_status) != 0 {
                capabilities.total_memory_mb = (memory_status.ullTotalPhys / (1024 * 1024)) as u64;
            }
        }
    }
    
    // GPU检测 - 如果有NVML
    #[cfg(feature = "nvml")]
    {
        if let Ok(_) = nvml_sys::nvml::nvml_init() {
            let mut device_count = 0u32;
            if nvml_sys::nvml::nvml_device_get_count(&mut device_count) == nvml_sys::nvml::NVML_SUCCESS {
                if device_count > 0 {
                    let mut device = std::ptr::null_mut();
                    if nvml_sys::nvml::nvml_device_get_handle_by_index(0, &mut device) == nvml_sys::nvml::NVML_SUCCESS {
                        // 获取GPU名称
                        let mut name_buffer = [0u8; 100];
                        if nvml_sys::nvml::nvml_device_get_name(device, name_buffer.as_mut_ptr() as *mut i8, name_buffer.len() as u32) == nvml_sys::nvml::NVML_SUCCESS {
                            let name_cstr = std::ffi::CStr::from_ptr(name_buffer.as_ptr() as *const i8);
                            if let Ok(name_str) = name_cstr.to_str() {
                                capabilities.gpu_model = Some(name_str.to_string());
                            }
                        }
                        
                        // 获取GPU内存大小
                        let mut memory = nvml_sys::nvml::nvmlMemory_t {
                            total: 0,
                            free: 0,
                            used: 0
                        };
                        if nvml_sys::nvml::nvml_device_get_memory_info(device, &mut memory) == nvml_sys::nvml::NVML_SUCCESS {
                            capabilities.gpu_memory_mb = Some((memory.total / (1024 * 1024)) as u64);
                        }
                    }
                }
            }
            nvml_sys::nvml::nvml_shutdown();
        }
    }
    
    // 非NVML的GPU检测
    #[cfg(not(feature = "nvml"))]
    {
        // Linux上使用nvidia-smi
        #[cfg(target_os = "linux")]
        {
            use std::process::Command;
            
            if let Ok(output) = Command::new("nvidia-smi")
                .args(&["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
                .output() {
                if output.status.success() {
                    if let Ok(output_str) = String::from_utf8(output.stdout) {
                        let parts: Vec<&str> = output_str.trim().split(',').collect();
                        if parts.len() >= 2 {
                            capabilities.gpu_model = Some(parts[0].trim().to_string());
                            if let Ok(mem) = parts[1].trim().parse::<u64>() {
                                capabilities.gpu_memory_mb = Some(mem);
                            }
                        }
                    }
                }
            }
        }
        
        // Windows上使用wmic
        #[cfg(target_os = "windows")]
        {
            use std::process::Command;
            
            if let Ok(output) = Command::new("wmic")
                .args(&["path", "win32_VideoController", "get", "Name,AdapterRAM", "/format:csv"])
                .output() {
                if output.status.success() {
                    if let Ok(output_str) = String::from_utf8(output.stdout) {
                        let lines: Vec<&str> = output_str.lines().collect();
                        if lines.len() >= 2 {
                            let parts: Vec<&str> = lines[1].split(',').collect();
                            if parts.len() >= 3 {
                                capabilities.gpu_model = Some(parts[1].to_string());
                                if let Ok(ram) = parts[2].parse::<u64>() {
                                    capabilities.gpu_memory_mb = Some(ram / (1024 * 1024));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    // 检测磁盘类型和速度
    #[cfg(target_os = "linux")]
    {
        use std::process::Command;
        
        // 尝试使用lsblk命令
        if let Ok(output) = Command::new("lsblk")
            .args(&["-d", "-o", "NAME,ROTA", "-n"])
            .output() {
            if output.status.success() {
                if let Ok(output_str) = String::from_utf8(output.stdout) {
                    let lines: Vec<&str> = output_str.lines().collect();
                    if !lines.is_empty() {
                        let parts: Vec<&str> = lines[0].split_whitespace().collect();
                        if parts.len() >= 2 {
                            let is_rotational = parts[1] == "1";
                            capabilities.disk_type = if is_rotational {
                                "HDD".to_string()
                            } else {
                                "SSD".to_string()
                            };
                        }
                    }
                }
            }
        }
        
        // 尝试使用hdparm测试读取速度
        if let Ok(output) = Command::new("hdparm")
            .args(&["-t", "/dev/sda"])
            .output() {
            if output.status.success() {
                if let Ok(output_str) = String::from_utf8(output.stdout) {
                    if let Some(line) = output_str.lines().find(|l| l.contains("MB/sec")) {
                        if let Some(speed_str) = line.split_whitespace().find(|w| w.parse::<f64>().is_ok()) {
                            if let Ok(speed) = speed_str.parse::<f64>() {
                                capabilities.disk_speed_mbps = Some(speed);
                            }
                        }
                    }
                }
            }
        }
    }
    
    capabilities
}

/// 获取当前时间戳（毫秒）
fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis() as u64
} 