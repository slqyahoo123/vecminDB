use std::time::{SystemTime, Duration};
use std::sync::Arc;
use sysinfo::System;
use crate::algorithm::base_types::{ResourceType, ResourceUsageTrend};
use crate::algorithm::gpu::GpuInfoProvider;
use log::info;

/// 算法资源使用情况
#[derive(Debug, Clone)]
pub struct AlgorithmResourceUsage {
    /// 内存使用 (bytes)
    pub memory_used: usize,
    /// 内存使用率(%)
    pub memory_usage_percent: f32,
    /// CPU使用率(%)
    pub cpu_usage: f32,
    /// CPU核心使用情况
    pub cpu_core_usage: Vec<f32>,
    /// GPU使用率(%)，如果可用
    pub gpu_usage: Option<f32>,
    /// GPU内存使用 (bytes)，如果可用
    pub gpu_memory_used: Option<usize>,
    /// 磁盘I/O读取 (bytes/s)
    pub disk_read_bytes: usize,
    /// 磁盘I/O写入 (bytes/s)
    pub disk_write_bytes: usize,
    /// 网络I/O下载 (bytes/s)
    pub network_receive_bytes: usize,
    /// 网络I/O上传 (bytes/s)
    pub network_send_bytes: usize,
    /// 时间戳
    pub timestamp: SystemTime,
}

impl AlgorithmResourceUsage {
    /// 创建新的资源使用记录
    pub fn new() -> Self {
        Self {
            memory_used: 0,
            memory_usage_percent: 0.0,
            cpu_usage: 0.0,
            cpu_core_usage: Vec::new(),
            gpu_usage: None,
            gpu_memory_used: None,
            disk_read_bytes: 0,
            disk_write_bytes: 0,
            network_receive_bytes: 0,
            network_send_bytes: 0,
            timestamp: SystemTime::now(),
        }
    }
    
    /// 转换为摘要字符串
    pub fn to_summary_string(&self) -> String {
        let gpu_str = if let Some(gpu) = self.gpu_usage {
            format!(", GPU: {:.1}%", gpu)
        } else {
            String::new()
        };
        
        format!(
            "CPU: {:.1}%, Memory: {:.1}% ({} MB){}, Disk: {} KB/s read, {} KB/s write, Network: {} KB/s down, {} KB/s up",
            self.cpu_usage,
            self.memory_usage_percent,
            self.memory_used / (1024 * 1024),
            gpu_str,
            self.disk_read_bytes / 1024,
            self.disk_write_bytes / 1024,
            self.network_receive_bytes / 1024,
            self.network_send_bytes / 1024
        )
    }
    
    /// 检查是否超过资源限制
    pub fn check_exceeds_limits(&self, limits: &AlgorithmResourceLimits) -> Option<String> {
        let mut violations = Vec::new();
        
        if let Some(max_memory) = limits.max_memory_mb {
            let memory_mb = self.memory_used / (1024 * 1024);
            if memory_mb > max_memory {
                violations.push(format!("内存使用超限: {} MB > {} MB", memory_mb, max_memory));
            }
        }
        
        if let Some(max_cpu) = limits.max_cpu_percent {
            if self.cpu_usage > max_cpu {
                violations.push(format!("CPU使用率超限: {:.1}% > {:.1}%", self.cpu_usage, max_cpu));
            }
        }
        
        if let Some(max_gpu) = limits.max_gpu_percent {
            if let Some(gpu_usage) = self.gpu_usage {
                if gpu_usage > max_gpu {
                    violations.push(format!("GPU使用率超限: {:.1}% > {:.1}%", gpu_usage, max_gpu));
                }
            }
        }
        
        if violations.is_empty() {
            None
        } else {
            Some(violations.join("; "))
        }
    }
}

impl Default for AlgorithmResourceUsage {
    fn default() -> Self {
        Self::new()
    }
}

/// 资源限制
#[derive(Debug, Clone)]
pub struct AlgorithmResourceLimits {
    /// 最大内存使用(MB)
    pub max_memory_mb: Option<usize>,
    /// 最大CPU使用率(%)
    pub max_cpu_percent: Option<f32>,
    /// 最大GPU使用率(%)
    pub max_gpu_percent: Option<f32>,
    /// 最大磁盘I/O (MB/s)
    pub max_disk_io_mb_per_sec: Option<f32>,
    /// 最大网络I/O (MB/s)
    pub max_network_io_mb_per_sec: Option<f32>,
}

impl Default for AlgorithmResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: None,
            max_cpu_percent: None,
            max_gpu_percent: None,
            max_disk_io_mb_per_sec: None,
            max_network_io_mb_per_sec: None,
        }
    }
}

/// 资源使用历史
#[derive(Debug, Clone)]
pub struct ResourceUsageHistory {
    /// 资源使用历史记录
    pub history: Vec<AlgorithmResourceUsage>,
    /// 最大历史记录数量
    max_history: usize,
}

impl ResourceUsageHistory {
    /// 创建新的资源使用历史
    pub fn new(max_history: usize) -> Self {
        Self {
            history: Vec::with_capacity(max_history),
            max_history,
        }
    }
    
    /// 添加资源使用记录
    pub fn add(&mut self, usage: AlgorithmResourceUsage) {
        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(usage);
    }
    
    /// 获取最新的资源使用记录
    pub fn latest(&self) -> Option<&AlgorithmResourceUsage> {
        self.history.last()
    }
    
    /// 获取平均CPU使用率
    pub fn average_cpu_usage(&self) -> Option<f32> {
        if self.history.is_empty() {
            return None;
        }
        
        let sum: f32 = self.history.iter().map(|u| u.cpu_usage).sum();
        Some(sum / self.history.len() as f32)
    }
    
    /// 获取平均内存使用
    pub fn average_memory_usage(&self) -> Option<usize> {
        if self.history.is_empty() {
            return None;
        }
        
        let sum: usize = self.history.iter().map(|u| u.memory_used).sum();
        Some(sum / self.history.len())
    }
    
    /// 计算资源使用趋势
    pub fn calculate_trend(&self) -> ResourceUsageTrend {
        if self.history.len() < 2 {
            return ResourceUsageTrend::Unknown;
        }
        
        let recent_count = std::cmp::min(5, self.history.len());
        let recent = &self.history[self.history.len() - recent_count..];
        
        let first_half_avg = recent[..recent_count/2].iter()
            .map(|u| u.cpu_usage).sum::<f32>() / (recent_count/2) as f32;
        let second_half_avg = recent[recent_count/2..].iter()
            .map(|u| u.cpu_usage).sum::<f32>() / (recent_count - recent_count/2) as f32;
        
        let diff = second_half_avg - first_half_avg;
        if diff > 5.0 {
            ResourceUsageTrend::Increasing
        } else if diff < -5.0 {
            ResourceUsageTrend::Decreasing
        } else {
            ResourceUsageTrend::Stable
        }
    }
}

/// 资源监控配置
#[derive(Debug, Clone)]
pub struct ResourceMonitoringConfig {
    /// 资源监控间隔（秒）
    pub interval: u64,
    /// 是否启用GPU监控
    pub enable_gpu_monitoring: bool,
    /// 是否启用详细的磁盘I/O监控
    pub enable_disk_io_monitoring: bool,
    /// 是否启用网络I/O监控
    pub enable_network_monitoring: bool,
    /// 资源使用历史记录数量
    pub history_size: usize,
    /// 是否记录详细日志
    pub verbose_logging: bool,
    /// 资源预警阈值
    pub warning_thresholds: ResourceWarningThresholds,
}

impl Default for ResourceMonitoringConfig {
    fn default() -> Self {
        Self {
            interval: 10,
            enable_gpu_monitoring: true,
            enable_disk_io_monitoring: true,
            enable_network_monitoring: true,
            history_size: 60, // 保留60条历史记录
            verbose_logging: false,
            warning_thresholds: ResourceWarningThresholds::default(),
        }
    }
}

/// 资源预警阈值
#[derive(Debug, Clone)]
pub struct ResourceWarningThresholds {
    /// CPU使用率预警阈值(%)
    pub cpu_percent: f32,
    /// 内存使用率预警阈值(%)
    pub memory_percent: f32,
    /// GPU使用率预警阈值(%)
    pub gpu_percent: f32,
    /// 磁盘使用率预警阈值(%)
    pub disk_percent: f32,
}

impl Default for ResourceWarningThresholds {
    fn default() -> Self {
        Self {
            cpu_percent: 85.0,
            memory_percent: 80.0,
            gpu_percent: 90.0,
            disk_percent: 85.0,
        }
    }
}

/// 资源预警
#[derive(Debug, Clone)]
pub struct ResourceWarning {
    /// 资源类型
    pub resource_type: ResourceType,
    /// 当前值
    pub current_value: f32,
    /// 预警阈值
    pub threshold: f32,
    /// 预警信息
    pub message: String,
}

/// 资源报告
#[derive(Debug, Clone)]
pub struct ResourceReport {
    /// 当前资源使用情况
    pub current_usage: Option<AlgorithmResourceUsage>,
    /// 平均CPU使用率
    pub average_cpu_usage: Option<f32>,
    /// 平均内存使用
    pub average_memory_usage: Option<usize>,
    /// 资源使用趋势
    pub trend: ResourceUsageTrend,
    /// 资源预警
    pub warnings: Vec<ResourceWarning>,
}

/// 资源监控器
pub struct ResourceMonitor {
    /// 资源使用历史
    pub usage_history: ResourceUsageHistory,
    /// 监控配置
    pub config: ResourceMonitoringConfig,
    /// 资源限制
    pub limits: AlgorithmResourceLimits,
    /// 系统信息
    system: System,
    /// 上次监控时间
    last_check: SystemTime,
    /// 上次磁盘读取字节数
    last_disk_read_bytes: Option<u64>,
    /// 上次磁盘写入字节数
    last_disk_write_bytes: Option<u64>,
    /// 上次网络接收字节数
    last_network_receive_bytes: Option<u64>,
    /// 上次网络发送字节数
    last_network_send_bytes: Option<u64>,
    /// 进程ID
    process_id: Option<u32>,
    /// GPU信息提供者
    #[allow(dead_code)]
    gpu_provider: Option<Arc<dyn GpuInfoProvider>>,
    /// 是否启用
    is_enabled: bool,
}

impl ResourceMonitor {
    /// 创建新的资源监控器
    pub fn new(config: ResourceMonitoringConfig, limits: AlgorithmResourceLimits) -> Self {
        let mut system = System::new();
        system.refresh_all();
        
        Self {
            usage_history: ResourceUsageHistory::new(config.history_size),
            config,
            limits,
            system,
            last_check: SystemTime::now(),
            last_disk_read_bytes: None,
            last_disk_write_bytes: None,
            last_network_receive_bytes: None,
            last_network_send_bytes: None,
            process_id: None,
            gpu_provider: None,
            is_enabled: true,
        }
    }
    
    /// 创建支持GPU监控的资源监控器
    pub fn with_gpu_monitoring(
        config: ResourceMonitoringConfig, 
        limits: AlgorithmResourceLimits,
        gpu_index: usize
    ) -> Self {
        let mut monitor = Self::new(config, limits);
        
        // 尝试创建GPU提供者
        if monitor.config.enable_gpu_monitoring {
            let gpu_provider = crate::algorithm::gpu::NvidiaGpuInfoProvider::new(gpu_index);
            monitor.gpu_provider = Some(Arc::new(gpu_provider));
        }
        
        monitor
    }
    
    /// 设置GPU提供者
    pub fn set_gpu_provider(&mut self, provider: Box<dyn GpuInfoProvider>) {
        self.gpu_provider = Some(Arc::from(provider));
    }
    
    /// 移除GPU提供者
    pub fn remove_gpu_provider(&mut self) {
        self.gpu_provider = None;
    }
    
    /// 检查是否支持GPU监控
    pub fn supports_gpu_monitoring(&self) -> bool {
        self.gpu_provider.is_some() && self.config.enable_gpu_monitoring
    }
    
    /// 停止监控
    pub fn stop_monitoring(&mut self) {
        self.is_enabled = false;
    }
    
    /// 更新资源使用情况
    pub fn update(&mut self) -> AlgorithmResourceUsage {
        if !self.is_enabled {
            return AlgorithmResourceUsage::default();
        }
        
        // 刷新系统信息
        self.system.refresh_all();
        
        let current_time = SystemTime::now();
        let time_diff = current_time.duration_since(self.last_check)
            .unwrap_or(Duration::from_secs(1));
        
        // 获取CPU使用率
        let cpu_usage = self.system.global_cpu_info().cpu_usage();
        let cpu_core_usage = self.system.cpus().iter()
            .map(|cpu| cpu.cpu_usage())
            .collect();
        
        // 获取内存使用情况
        let total_memory = self.system.total_memory();
        let used_memory = self.system.used_memory();
        let memory_usage_percent = if total_memory > 0 {
            (used_memory as f32 / total_memory as f32) * 100.0
        } else {
            0.0
        };
        
        // 获取GPU信息
        let (gpu_usage, gpu_memory_used) = if let Some(ref provider) = self.gpu_provider {
            let gpu_info = provider.get_gpu_info();
            (gpu_info.usage, gpu_info.memory_used)
        } else {
            (None, None)
        };
        
        // 获取磁盘I/O信息（简化实现）
        let (disk_read_bytes, disk_write_bytes) = self.get_disk_io_delta(time_diff);
        
        // 获取网络I/O信息（简化实现）
        let (network_receive_bytes, network_send_bytes) = self.get_network_io_delta(time_diff);
        
        let usage = AlgorithmResourceUsage {
            memory_used: used_memory as usize,
            memory_usage_percent,
            cpu_usage,
            cpu_core_usage,
            gpu_usage,
            gpu_memory_used,
            disk_read_bytes,
            disk_write_bytes,
            network_receive_bytes,
            network_send_bytes,
            timestamp: current_time,
        };
        
        // 添加到历史记录
        self.usage_history.add(usage.clone());
        self.last_check = current_time;
        
        usage
    }
    
    /// 获取磁盘I/O增量
    fn get_disk_io_delta(&mut self, time_diff: Duration) -> (usize, usize) {
        if !self.config.enable_disk_io_monitoring {
            return (0, 0);
        }
        
        // 获取当前磁盘I/O统计
        let mut total_read_bytes = 0u64;
        let mut total_write_bytes = 0u64;
        
        for (_name, disk) in self.system.disks() {
            // 注意：sysinfo crate可能需要不同的API来获取I/O统计
            // 这里使用简化的模拟实现
            total_read_bytes += disk.total_space() / 1000; // 模拟读取
            total_write_bytes += disk.available_space() / 1000; // 模拟写入
        }
        
        let read_delta = self.last_disk_read_bytes.map_or(0, |last| {
            if total_read_bytes >= last {
                total_read_bytes - last
            } else {
                0
            }
        });
        
        let write_delta = self.last_disk_write_bytes.map_or(0, |last| {
            if total_write_bytes >= last {
                total_write_bytes - last
            } else {
                0
            }
        });
        
        self.last_disk_read_bytes = Some(total_read_bytes);
        self.last_disk_write_bytes = Some(total_write_bytes);
        
        let time_sec = time_diff.as_secs() as f64 + time_diff.subsec_nanos() as f64 / 1_000_000_000.0;
        let read_bytes_per_sec = if time_sec > 0.0 {
            (read_delta as f64 / time_sec) as usize
        } else {
            0
        };
        let write_bytes_per_sec = if time_sec > 0.0 {
            (write_delta as f64 / time_sec) as usize
        } else {
            0
        };
        
        (read_bytes_per_sec, write_bytes_per_sec)
    }
    
    /// 获取网络I/O增量
    fn get_network_io_delta(&mut self, time_diff: Duration) -> (usize, usize) {
        if !self.config.enable_network_monitoring {
            return (0, 0);
        }
        
        // 获取当前网络I/O统计
        let mut total_receive_bytes = 0u64;
        let mut total_send_bytes = 0u64;
        
        for (_interface_name, data) in self.system.networks() {
            total_receive_bytes += data.received();
            total_send_bytes += data.transmitted();
        }
        
        let receive_delta = self.last_network_receive_bytes.map_or(0, |last| {
            if total_receive_bytes >= last {
                total_receive_bytes - last
            } else {
                0
            }
        });
        
        let send_delta = self.last_network_send_bytes.map_or(0, |last| {
            if total_send_bytes >= last {
                total_send_bytes - last
            } else {
                0
            }
        });
        
        self.last_network_receive_bytes = Some(total_receive_bytes);
        self.last_network_send_bytes = Some(total_send_bytes);
        
        let time_sec = time_diff.as_secs() as f64 + time_diff.subsec_nanos() as f64 / 1_000_000_000.0;
        let receive_bytes_per_sec = if time_sec > 0.0 {
            (receive_delta as f64 / time_sec) as usize
        } else {
            0
        };
        let send_bytes_per_sec = if time_sec > 0.0 {
            (send_delta as f64 / time_sec) as usize
        } else {
            0
        };
        
        (receive_bytes_per_sec, send_bytes_per_sec)
    }
    
    /// 生成资源报告
    pub fn get_resource_report(&self) -> ResourceReport {
        let current_usage = self.usage_history.latest().cloned();
        let average_cpu_usage = self.usage_history.average_cpu_usage();
        let average_memory_usage = self.usage_history.average_memory_usage();
        let trend = self.usage_history.calculate_trend();
        
        let mut warnings = Vec::new();
        
        if let Some(ref usage) = current_usage {
            // 检查CPU使用率预警
            if usage.cpu_usage > self.config.warning_thresholds.cpu_percent {
                warnings.push(ResourceWarning {
                    resource_type: ResourceType::Cpu,
                    current_value: usage.cpu_usage,
                    threshold: self.config.warning_thresholds.cpu_percent,
                    message: format!("CPU使用率过高: {:.1}%", usage.cpu_usage),
                });
            }
            
            // 检查内存使用率预警
            if usage.memory_usage_percent > self.config.warning_thresholds.memory_percent {
                warnings.push(ResourceWarning {
                    resource_type: ResourceType::Memory,
                    current_value: usage.memory_usage_percent,
                    threshold: self.config.warning_thresholds.memory_percent,
                    message: format!("内存使用率过高: {:.1}%", usage.memory_usage_percent),
                });
            }
            
            // 检查GPU使用率预警
            if let Some(gpu_usage) = usage.gpu_usage {
                if gpu_usage > self.config.warning_thresholds.gpu_percent {
                    warnings.push(ResourceWarning {
                        resource_type: ResourceType::Gpu,
                        current_value: gpu_usage,
                        threshold: self.config.warning_thresholds.gpu_percent,
                        message: format!("GPU使用率过高: {:.1}%", gpu_usage),
                    });
                }
            }
        }
        
        ResourceReport {
            current_usage,
            average_cpu_usage,
            average_memory_usage,
            trend,
            warnings,
        }
    }
    
    /// 启动监控
    pub async fn start_monitoring(&mut self) -> crate::Result<()> {
        self.is_enabled = true;
        info!("资源监控已启动");
        Ok(())
    }
    
    /// 获取资源使用情况
    pub fn get_usage(&self) -> AlgorithmResourceUsage {
        if let Some(latest) = self.usage_history.latest() {
            latest.clone()
        } else {
            AlgorithmResourceUsage::default()
        }
    }
    
    /// 强制终止所有进程（用于安全清理）
    pub fn force_terminate_all_processes(&self) -> crate::Result<()> {
        // 简化实现：通过系统调用终止相关进程
        if let Some(pid) = self.process_id {
            #[cfg(unix)]
            {
                use std::process::Command;
                let _ = Command::new("kill")
                    .arg("-9")
                    .arg(pid.to_string())
                    .output();
            }
            
            #[cfg(windows)]
            {
                use std::process::Command;
                let _ = Command::new("taskkill")
                    .arg("/F")
                    .arg("/PID")
                    .arg(pid.to_string())
                    .output();
            }
        }
        
        self.stop_monitoring();
        Ok(())
    }

    /// 获取活跃进程数量
    pub fn get_active_process_count(&self) -> usize {
        if self.process_id.is_some() && self.is_enabled {
            1
        } else {
            0
        }
    }

    /// 获取统计信息
    pub fn get_statistics(&self) -> EnforcementStatistics {
        EnforcementStatistics {
            total_processes_monitored: if self.process_id.is_some() { 1 } else { 0 },
            active_processes: self.get_active_process_count(),
            resource_violations: 0, // TODO: 实现违规统计
            enforcement_actions: 0, // TODO: 实现执行动作统计
            average_cpu_usage: self.usage_history.average_cpu_usage().unwrap_or(0.0),
            average_memory_usage: self.usage_history.average_memory_usage().unwrap_or(0),
            last_update: chrono::Utc::now(),
        }
    }

    /// 获取最近的监控事件
    pub fn get_recent_events(&self, limit: usize) -> Vec<ResourceMonitoringEvent> {
        // 简化实现：根据最近的资源使用情况生成事件
        let mut events = Vec::new();
        
        if let Some(latest_usage) = self.usage_history.latest() {
            // 检查CPU使用率是否过高
            if latest_usage.cpu_usage > self.config.warning_thresholds.cpu_percent {
                events.push(ResourceMonitoringEvent {
                    event_type: "HighCpuUsage".to_string(),
                    description: format!("CPU使用率过高: {:.1}%", latest_usage.cpu_usage),
                    severity: if latest_usage.cpu_usage > 90.0 { "Critical" } else { "Warning" }.to_string(),
                    timestamp: chrono::Utc::now(),
                    process_id: self.process_id,
                    resource_values: std::collections::HashMap::from([
                        ("cpu_usage".to_string(), latest_usage.cpu_usage as f64),
                        ("threshold".to_string(), self.config.warning_thresholds.cpu_percent as f64),
                    ]),
                });
            }
            
            // 检查内存使用率是否过高
            if latest_usage.memory_usage_percent > self.config.warning_thresholds.memory_percent {
                events.push(ResourceMonitoringEvent {
                    event_type: "HighMemoryUsage".to_string(),
                    description: format!("内存使用率过高: {:.1}%", latest_usage.memory_usage_percent),
                    severity: if latest_usage.memory_usage_percent > 90.0 { "Critical" } else { "Warning" }.to_string(),
                    timestamp: chrono::Utc::now(),
                    process_id: self.process_id,
                    resource_values: std::collections::HashMap::from([
                        ("memory_usage_percent".to_string(), latest_usage.memory_usage_percent as f64),
                        ("threshold".to_string(), self.config.warning_thresholds.memory_percent as f64),
                    ]),
                });
            }
        }
        
        // 只返回指定数量的事件
        events.truncate(limit);
        events
    }
}

/// 执行统计信息
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnforcementStatistics {
    /// 监控的进程总数
    pub total_processes_monitored: usize,
    /// 活跃进程数量
    pub active_processes: usize,
    /// 资源违规次数
    pub resource_violations: u64,
    /// 执行动作次数
    pub enforcement_actions: u64,
    /// 平均CPU使用率
    pub average_cpu_usage: f32,
    /// 平均内存使用量
    pub average_memory_usage: usize,
    /// 最后更新时间
    pub last_update: chrono::DateTime<chrono::Utc>,
}

/// 资源监控事件
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResourceMonitoringEvent {
    /// 事件类型
    pub event_type: String,
    /// 事件描述
    pub description: String,
    /// 严重程度
    pub severity: String,
    /// 时间戳
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// 进程ID
    pub process_id: Option<u32>,
    /// 资源值
    pub resource_values: std::collections::HashMap<String, f64>,
} 