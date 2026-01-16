// src/data/pipeline/monitor/performance_monitor.rs
//
// 性能监控器实现
// 提供对流水线执行过程中CPU、内存和IO使用情况的监控

use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};
// no shared state in this monitor; remove unused sync imports
use log::{debug, info, warn};

use crate::data::pipeline::traits::{PipelineStage, PipelineStageStatus, PipelineContext};
use crate::error::Result;
use std::error::Error;

#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::io::{BufRead, BufReader};
#[cfg(all(target_os = "windows", feature = "winapi"))]
use winapi::um::processthreadsapi::GetCurrentProcess;
#[cfg(all(target_os = "windows", feature = "winapi"))]
use winapi::um::psapi::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};

/// 性能指标类型
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricType {
    /// CPU使用率（百分比）
    CpuUsage,
    /// 内存使用（MB）
    MemoryUsage,
    /// 磁盘读取（KB/s）
    DiskRead,
    /// 磁盘写入（KB/s）
    DiskWrite,
    /// 执行时间（毫秒）
    ExecutionTime,
    /// 处理记录数
    RecordsProcessed,
    /// 每秒处理记录数
    RecordsPerSecond,
}

/// 资源使用指标
#[derive(Debug, Clone)]
pub struct ResourceMetrics {
    /// CPU使用率（0-100%）
    pub cpu_usage: f64,
    /// 内存使用量（MB）
    pub memory_usage_mb: f64,
    /// 磁盘读取速率（KB/s）
    pub disk_read_kbps: f64,
    /// 磁盘写入速率（KB/s）
    pub disk_write_kbps: f64,
    /// 采集时间
    pub timestamp: SystemTime,
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage_mb: 0.0,
            disk_read_kbps: 0.0,
            disk_write_kbps: 0.0,
            timestamp: std::time::SystemTime::now(),
        }
    }
}

/// 性能监控阶段
pub struct AdvancedPerformanceMonitorStage {
    /// 阶段名称
    name: String,
    /// 阶段状态
    status: PipelineStageStatus,
    /// 监控间隔（毫秒）
    monitor_interval_ms: u64,
    /// 是否收集CPU指标
    collect_cpu: bool,
    /// 是否收集内存指标
    collect_memory: bool,
    /// 是否收集磁盘指标
    collect_disk: bool,
    /// 开始时间
    start_time: Option<Instant>,
    /// 收集的指标
    metrics: Vec<ResourceMetrics>,
    /// 阶段执行时间
    stage_times: HashMap<String, Duration>,
    /// 上次资源采集时间
    last_resource_collection: Option<Instant>,
    /// 处理的记录数
    processed_records: usize,
    /// 上次CPU总时间（用于计算CPU使用率）
    last_cpu_total_time: Option<u64>,
    /// 上次CPU空闲时间（用于计算CPU使用率）
    last_cpu_idle_time: Option<u64>,
    /// 上次磁盘读取字节数（用于计算I/O速率）
    last_disk_read_bytes: Option<u64>,
    /// 上次磁盘写入字节数（用于计算I/O速率）
    last_disk_write_bytes: Option<u64>,
}

impl AdvancedPerformanceMonitorStage {
    /// 创建新的性能监控阶段
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            status: PipelineStageStatus::NotInitialized,
            monitor_interval_ms: 1000, // 默认1秒采集一次
            collect_cpu: true,
            collect_memory: true,
            collect_disk: true,
            start_time: None,
            metrics: Vec::new(),
            stage_times: HashMap::new(),
            last_resource_collection: None,
            processed_records: 0,
            last_cpu_total_time: None,
            last_cpu_idle_time: None,
            last_disk_read_bytes: None,
            last_disk_write_bytes: None,
        }
    }

    /// 设置监控间隔
    pub fn with_monitor_interval(mut self, interval_ms: u64) -> Self {
        self.monitor_interval_ms = interval_ms;
        self
    }

    /// 设置是否收集CPU指标
    pub fn with_cpu_monitoring(mut self, enabled: bool) -> Self {
        self.collect_cpu = enabled;
        self
    }

    /// 设置是否收集内存指标
    pub fn with_memory_monitoring(mut self, enabled: bool) -> Self {
        self.collect_memory = enabled;
        self
    }

    /// 设置是否收集磁盘指标
    pub fn with_disk_monitoring(mut self, enabled: bool) -> Self {
        self.collect_disk = enabled;
        self
    }

    /// 获取CPU使用率
    #[cfg(target_os = "linux")]
    fn get_cpu_usage(&mut self) -> Result<f64> {
        // 读取/proc/stat获取CPU使用情况
        let file = File::open("/proc/stat")?;
        let reader = BufReader::new(file);
        let first_line = reader.lines().next().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::Other, "无法读取/proc/stat")
        })??;

        let values: Vec<&str> = first_line.split_whitespace().collect();
        if values.len() < 5 {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "无效的/proc/stat格式",
            )));
        }

        // 计算总CPU时间和空闲时间
        let user = values[1].parse::<u64>()?;
        let nice = values[2].parse::<u64>()?;
        let system = values[3].parse::<u64>()?;
        let idle = values[4].parse::<u64>()?;

        let total_time = user + nice + system + idle;
        let idle_time = idle;

        // 计算CPU使用率：需要与上次采集的值进行比较
        if let (Some(last_total), Some(last_idle)) = (self.last_cpu_total_time, self.last_cpu_idle_time) {
            let total_delta = total_time.saturating_sub(last_total);
            let idle_delta = idle_time.saturating_sub(last_idle);
            
            if total_delta > 0 {
                let cpu_usage = 100.0 * (1.0 - (idle_delta as f64 / total_delta as f64));
                // 更新上次的值
                self.last_cpu_total_time = Some(total_time);
                self.last_cpu_idle_time = Some(idle_time);
                Ok(cpu_usage.max(0.0).min(100.0))
            } else {
                Ok(0.0)
            }
        } else {
            // 首次采集，保存当前值，返回0
            self.last_cpu_total_time = Some(total_time);
            self.last_cpu_idle_time = Some(idle_time);
            Ok(0.0)
        }
    }

    /// 获取CPU使用率（Windows）
    #[cfg(all(target_os = "windows", feature = "winapi"))]
    fn get_cpu_usage(&mut self) -> Result<f64> {
        // Windows下获取CPU使用率：读取性能计数器
        // 使用GetSystemTimes API获取系统时间信息
        use winapi::um::processthreadsapi::GetSystemTimes;
        use winapi::shared::minwindef::FILETIME;
        
        unsafe {
            let mut idle_time = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
            let mut kernel_time = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
            let mut user_time = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
            
            if GetSystemTimes(&mut idle_time, &mut kernel_time, &mut user_time) == 0 {
                return Err(crate::Error::system("无法获取系统时间信息"));
            }
            
            // 将FILETIME转换为u64（100纳秒单位）
            let idle = ((idle_time.dwHighDateTime as u64) << 32) | (idle_time.dwLowDateTime as u64);
            let kernel = ((kernel_time.dwHighDateTime as u64) << 32) | (kernel_time.dwLowDateTime as u64);
            let user = ((user_time.dwHighDateTime as u64) << 32) | (user_time.dwLowDateTime as u64);
            let total = kernel + user;
            
            // 计算CPU使用率：需要与上次采集的值进行比较
            if let (Some(last_total), Some(last_idle)) = (self.last_cpu_total_time, self.last_cpu_idle_time) {
                let total_delta = total.saturating_sub(last_total);
                let idle_delta = idle.saturating_sub(last_idle);
                
                if total_delta > 0 {
                    let cpu_usage: f64 = 100.0 * (1.0 - (idle_delta as f64 / total_delta as f64));
                    // 更新上次的值
                    self.last_cpu_total_time = Some(total);
                    self.last_cpu_idle_time = Some(idle);
                    Ok(cpu_usage.max(0.0).min(100.0))
                } else {
                    Ok(0.0)
                }
            } else {
                // 首次采集，保存当前值，返回0
                self.last_cpu_total_time = Some(total);
                self.last_cpu_idle_time = Some(idle);
                Ok(0.0)
            }
        }
    }

    /// 获取CPU使用率（Windows，无winapi feature）
    #[cfg(all(target_os = "windows", not(feature = "winapi")))]
    fn get_cpu_usage(&mut self) -> Result<f64> {
        warn!("Windows平台需要winapi feature才能获取CPU使用率");
        Ok(0.0)
    }

    /// 获取CPU使用率（其他平台）
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    fn get_cpu_usage(&mut self) -> Result<f64> {
        warn!("当前平台不支持CPU使用率监控");
        Ok(0.0)
    }

    /// 获取内存使用情况（Linux）
    #[cfg(target_os = "linux")]
    fn get_memory_usage(&self) -> Result<f64> {
        // 读取/proc/self/status获取内存使用情况
        let file = File::open("/proc/self/status")?;
        let reader = BufReader::new(file);
        
        let mut vm_rss = None;
        
        for line in reader.lines() {
            let line = line?;
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    vm_rss = Some(parts[1].parse::<f64>()?);
                    break;
                }
            }
        }
        
        if let Some(rss) = vm_rss {
            // 转换为MB
            Ok(rss / 1024.0)
        } else {
            Err(crate::Error::system("无法获取内存使用情况"))
        }
    }

    /// 获取内存使用情况（Windows）
    #[cfg(all(target_os = "windows", feature = "winapi"))]
    fn get_memory_usage(&self) -> Result<f64> {
        unsafe {
            let mut pmc: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
            pmc.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
            
            if GetProcessMemoryInfo(
                GetCurrentProcess(),
                &mut pmc,
                std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            ) == 0 {
                return Err(crate::Error::system("无法获取内存使用情况"));
            }
            
            // 转换为MB
            Ok(pmc.WorkingSetSize as f64 / (1024.0 * 1024.0))
        }
    }

    /// 获取内存使用情况（Windows，无winapi feature）
    #[cfg(all(target_os = "windows", not(feature = "winapi")))]
    fn get_memory_usage(&self) -> Result<f64> {
        warn!("Windows平台需要winapi feature才能获取内存使用情况");
        Ok(0.0)
    }

    /// 获取内存使用情况（其他平台）
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    fn get_memory_usage(&self) -> Result<f64> {
        warn!("当前平台不支持内存使用情况监控");
        Ok(0.0)
    }

    /// 获取磁盘IO情况（Linux）
    #[cfg(target_os = "linux")]
    fn get_disk_io(&mut self) -> Result<(f64, f64)> {
        // 读取/proc/self/io获取磁盘IO情况
        let file = File::open("/proc/self/io")?;
        let reader = BufReader::new(file);
        
        let mut read_bytes = None;
        let mut write_bytes = None;
        
        for line in reader.lines() {
            let line = line?;
            if line.starts_with("read_bytes:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    read_bytes = Some(parts[1].parse::<u64>()?);
                }
            } else if line.starts_with("write_bytes:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    write_bytes = Some(parts[1].parse::<u64>()?);
                }
            }
        }
        
        if let (Some(read), Some(write)) = (read_bytes, write_bytes) {
            // 计算I/O速率：需要与上次采集的值进行比较，并考虑时间间隔
            let elapsed_secs = if let Some(last_collection) = self.last_resource_collection {
                last_collection.elapsed().as_secs_f64()
            } else {
                1.0 // 默认1秒
            };
            
            if let (Some(last_read), Some(last_write)) = (self.last_disk_read_bytes, self.last_disk_write_bytes) {
                let read_delta = read.saturating_sub(last_read);
                let write_delta = write.saturating_sub(last_write);
                
                let read_kbps = (read_delta as f64 / 1024.0) / elapsed_secs.max(0.001);
                let write_kbps = (write_delta as f64 / 1024.0) / elapsed_secs.max(0.001);
                
                // 更新上次的值
                self.last_disk_read_bytes = Some(read);
                self.last_disk_write_bytes = Some(write);
                
                Ok((read_kbps.max(0.0), write_kbps.max(0.0)))
            } else {
                // 首次采集，保存当前值，返回0
                self.last_disk_read_bytes = Some(read);
                self.last_disk_write_bytes = Some(write);
                Ok((0.0, 0.0))
            }
        } else {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "无法获取磁盘IO情况",
            )))
        }
    }

    /// 获取磁盘IO情况（Windows和其他平台）
    #[cfg(not(target_os = "linux"))]
    fn get_disk_io(&mut self) -> Result<(f64, f64)> {
        warn!("当前平台不支持磁盘IO监控");
        Ok((0.0, 0.0))
    }

    /// 收集资源指标
    fn collect_resources(&mut self) -> Result<()> {
        let now = Instant::now();
        
        // 检查是否应该采集
        if let Some(last) = self.last_resource_collection {
            let elapsed = now.duration_since(last).as_millis() as u64;
            if elapsed < self.monitor_interval_ms {
                return Ok(());
            }
        }
        
        // 更新采集时间
        self.last_resource_collection = Some(now);
        
        let mut metrics = ResourceMetrics {
            timestamp: SystemTime::now(),
            ..Default::default()
        };
        
        // 收集CPU使用率
        if self.collect_cpu {
            metrics.cpu_usage = self.get_cpu_usage()?;
        }
        
        // 收集内存使用情况
        if self.collect_memory {
            metrics.memory_usage_mb = self.get_memory_usage()?;
        }
        
        // 收集磁盘IO情况
        if self.collect_disk {
            let (read, write) = self.get_disk_io()?;
            metrics.disk_read_kbps = read;
            metrics.disk_write_kbps = write;
        }
        
        // 保存指标
        self.metrics.push(metrics);
        
        Ok(())
    }

    /// 计算总执行时间
    fn calculate_total_time(&self) -> Duration {
        if let Some(start) = self.start_time {
            Instant::now().duration_since(start)
        } else {
            Duration::from_secs(0)
        }
    }
    
    /// 获取收集的指标
    pub fn get_metrics(&self) -> &[ResourceMetrics] {
        &self.metrics
    }
    
    /// 获取阶段执行时间
    pub fn get_stage_times(&self) -> &HashMap<String, Duration> {
        &self.stage_times
    }
    
    /// 记录阶段执行时间
    pub fn record_stage_time(&mut self, stage_name: &str, duration: Duration) {
        self.stage_times.insert(stage_name.to_string(), duration);
    }
    
    /// 获取元数据
    pub fn metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        
        // 添加监控间隔
        metadata.insert("monitor_interval_ms".to_string(), self.monitor_interval_ms.to_string());
        
        // 添加收集的指标类型
        let mut metric_types = Vec::new();
        if self.collect_cpu {
            metric_types.push("cpu");
        }
        if self.collect_memory {
            metric_types.push("memory");
        }
        if self.collect_disk {
            metric_types.push("disk");
        }
        metadata.insert("metric_types".to_string(), metric_types.join(","));
        
        // 添加总执行时间
        if let Some(start) = self.start_time {
            let duration = Instant::now().duration_since(start);
            metadata.insert("total_time_ms".to_string(), duration.as_millis().to_string());
        }
        
        // 添加处理的记录数
        metadata.insert("processed_records".to_string(), self.processed_records.to_string());
        
        // 添加平均指标
        if !self.metrics.is_empty() {
            let cpu_avg = self.metrics.iter().map(|m| m.cpu_usage).sum::<f64>() / self.metrics.len() as f64;
            let mem_avg = self.metrics.iter().map(|m| m.memory_usage_mb).sum::<f64>() / self.metrics.len() as f64;
            metadata.insert("avg_cpu_usage".to_string(), format!("{:.2}", cpu_avg));
            metadata.insert("avg_memory_usage_mb".to_string(), format!("{:.2}", mem_avg));
        }
        
        metadata
    }
}

impl PipelineStage for AdvancedPerformanceMonitorStage {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn init(&mut self, _context: &mut PipelineContext) -> std::result::Result<(), Box<dyn Error>> {
        info!("初始化性能监控阶段: {}", self.name);
        self.status = PipelineStageStatus::Initialized;
        self.start_time = Some(Instant::now());
        self.last_resource_collection = Some(Instant::now());
        Ok(())
    }
    
    fn process(&mut self, context: &mut PipelineContext) -> std::result::Result<(), Box<dyn Error>> {
        debug!("执行性能监控阶段: {}", self.name);
        self.status = PipelineStageStatus::Processing;
        
        // 收集资源使用情况
        if let Err(e) = self.collect_resources() {
            warn!("收集资源指标失败: {}", e);
        }
        
        // 更新处理的记录数
        self.processed_records = context.records.len();
        
        // 记录各阶段执行时间
        if let Some(stages) = context.shared_data.get("pipeline_stages") {
            if let Some(stage_times) = context.shared_data.get("stage_times") {
                if let Ok(times) = stage_times.lock() {
                    if let Some(times_map) = times.downcast_ref::<HashMap<String, Duration>>() {
                        for (stage, time) in times_map {
                            self.record_stage_time(stage, *time);
                        }
                    }
                }
            }
        }
        
        // 计算总执行时间并添加到上下文
        let total_time = self.calculate_total_time();
        context.metadata.insert("total_execution_time_ms".to_string(), total_time.as_millis().to_string());
        
        // 计算每秒处理记录数
        let records_per_second = if total_time.as_secs() > 0 {
            self.processed_records as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };
        context.metadata.insert("records_per_second".to_string(), format!("{:.2}", records_per_second));
        
        // 将收集的资源指标添加到上下文
        let metrics_clone = self.metrics.clone();
        context.add_shared_data("resource_metrics", metrics_clone);
        
        self.status = PipelineStageStatus::Completed;
        Ok(())
    }
    
    fn cleanup(&mut self, _context: &mut PipelineContext) -> std::result::Result<(), Box<dyn Error>> {
        debug!("清理性能监控阶段: {}", self.name);
        self.status = PipelineStageStatus::Cleaned;
        Ok(())
    }
    
    fn status(&self) -> PipelineStageStatus {
        self.status
    }
    
    fn set_status(&mut self, status: PipelineStageStatus) {
        self.status = status;
    }
    
    fn metadata(&self) -> HashMap<String, String> {
        self.metadata()
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

// 实现 Default 特性
impl Default for AdvancedPerformanceMonitorStage {
    fn default() -> Self {
        Self::new("性能监控")
    }
} 