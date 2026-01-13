use serde::{Serialize, Deserialize};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use crate::algorithm::types::ResourceUsage;

/// 资源监控器
#[derive(Debug)]
pub struct ResourceMonitor {
    /// 开始时间
    start_time: Instant,
    
    /// CPU使用量(百分比 * 100)
    cpu_usage: AtomicU64,
    
    /// 内存使用量(字节)
    memory_usage: AtomicUsize,
    
    /// 峰值内存使用量(字节)
    peak_memory: AtomicUsize,
    
    /// IO读取字节数
    io_reads: AtomicUsize,
    
    /// IO写入字节数
    io_writes: AtomicUsize,
    
    /// 网络传输字节数
    network_bytes: AtomicUsize,
}

impl ResourceMonitor {
    /// 创建新的资源监控器
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            cpu_usage: AtomicU64::new(0),
            memory_usage: AtomicUsize::new(0),
            peak_memory: AtomicUsize::new(0),
            io_reads: AtomicUsize::new(0),
            io_writes: AtomicUsize::new(0),
            network_bytes: AtomicUsize::new(0),
        }
    }
    
    /// 重置监控器
    pub fn reset(&mut self) {
        self.start_time = Instant::now();
        self.cpu_usage.store(0, Ordering::SeqCst);
        self.memory_usage.store(0, Ordering::SeqCst);
        self.peak_memory.store(0, Ordering::SeqCst);
        self.io_reads.store(0, Ordering::SeqCst);
        self.io_writes.store(0, Ordering::SeqCst);
        self.network_bytes.store(0, Ordering::SeqCst);
    }
    
    /// 更新CPU使用率
    pub fn update_cpu_usage(&self, percent: f64) {
        self.cpu_usage.store((percent * 100.0) as u64, Ordering::SeqCst);
    }
    
    /// 更新内存使用量
    pub fn update_memory_usage(&self, bytes: usize) {
        self.memory_usage.store(bytes, Ordering::SeqCst);
        
        // 更新峰值内存
        let current_peak = self.peak_memory.load(Ordering::SeqCst);
        if bytes > current_peak {
            self.peak_memory.store(bytes, Ordering::SeqCst);
        }
    }
    
    /// 记录IO读取
    pub fn record_io_read(&self, bytes: usize) {
        self.io_reads.fetch_add(bytes, Ordering::SeqCst);
    }
    
    /// 记录IO写入
    pub fn record_io_write(&self, bytes: usize) {
        self.io_writes.fetch_add(bytes, Ordering::SeqCst);
    }
    
    /// 记录网络传输
    pub fn record_network_io(&self, bytes: usize) {
        self.network_bytes.fetch_add(bytes, Ordering::SeqCst);
    }
    
    /// 获取CPU使用率
    pub fn get_cpu_usage(&self) -> f64 {
        self.cpu_usage.load(Ordering::SeqCst) as f64 / 100.0
    }
    
    /// 获取内存使用量
    pub fn get_memory_usage(&self) -> usize {
        self.memory_usage.load(Ordering::SeqCst)
    }
    
    /// 获取峰值内存使用量
    pub fn get_peak_memory(&self) -> usize {
        self.peak_memory.load(Ordering::SeqCst)
    }
    
    /// 获取IO读取字节数
    pub fn get_io_reads(&self) -> usize {
        self.io_reads.load(Ordering::SeqCst)
    }
    
    /// 获取IO写入字节数
    pub fn get_io_writes(&self) -> usize {
        self.io_writes.load(Ordering::SeqCst)
    }
    
    /// 获取网络传输字节数
    pub fn get_network_bytes(&self) -> usize {
        self.network_bytes.load(Ordering::SeqCst)
    }
    
    /// 获取经过的时间
    pub fn get_elapsed_time(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    /// 获取当前资源使用情况
    pub fn get_current_usage(&self) -> ResourceUsage {
        ResourceUsage {
            cpu_usage_percent: self.get_cpu_usage(),
            memory_usage_bytes: self.get_memory_usage(),
            peak_memory_bytes: self.get_peak_memory(),
            io_read_bytes: self.get_io_reads(),
            io_write_bytes: self.get_io_writes(),
            network_bytes: self.get_network_bytes(),
            execution_time_ms: self.get_elapsed_time().as_millis() as u64,
            limits_exceeded: false,
            exceeded_resource: None,
            code_size_bytes: 0,
            instruction_count: 0,
            output_size_bytes: 0,
        }
    }
    
    /// 开始监控
    pub fn start_monitoring(&self) -> crate::error::Result<()> {
        // 这里可以启动后台监控线程或任务
        // 目前简单实现为重置计数器
        self.cpu_usage.store(0, Ordering::SeqCst);
        self.memory_usage.store(0, Ordering::SeqCst);
        self.io_reads.store(0, Ordering::SeqCst);
        self.io_writes.store(0, Ordering::SeqCst);
        self.network_bytes.store(0, Ordering::SeqCst);
        Ok(())
    }
    
    /// 获取使用情况（别名方法）
    pub fn get_usage(&self) -> ResourceUsage {
        self.get_current_usage()
    }
}

/// 资源阈值设置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceThresholds {
    /// 内存使用警告阈值（占最大值的百分比）
    pub memory_warning_threshold: f64,
    /// CPU使用警告阈值（占最大值的百分比）
    pub cpu_warning_threshold: f64,
    /// 指令计数警告阈值（占最大值的百分比）
    pub instruction_warning_threshold: f64,
    /// 内存使用危险阈值（占最大值的百分比）
    pub memory_critical_threshold: f64,
    /// CPU使用危险阈值（占最大值的百分比）
    pub cpu_critical_threshold: f64,
    /// 指令计数危险阈值（占最大值的百分比）
    pub instruction_critical_threshold: f64,
}

impl Default for ResourceThresholds {
    fn default() -> Self {
        Self {
            memory_warning_threshold: 0.7,      // 70%
            cpu_warning_threshold: 0.7,         // 70%
            instruction_warning_threshold: 0.7,  // 70%
            memory_critical_threshold: 0.9,      // 90%
            cpu_critical_threshold: 0.9,         // 90%
            instruction_critical_threshold: 0.9,  // 90%
        }
    }
}

/// 资源监控器（详细版）
pub struct DetailedResourceMonitor {
    /// 开始时间
    start_time: std::time::Instant,
    /// 当前资源使用
    current_usage: ResourceUsage,
    /// 资源限制
    limits: crate::algorithm::types::ResourceLimits,
    /// 资源阈值
    thresholds: ResourceThresholds,
}

impl DetailedResourceMonitor {
    /// 创建新的资源监控器
    pub fn new(limits: crate::algorithm::types::ResourceLimits) -> Self {
        Self {
            start_time: std::time::Instant::now(),
            current_usage: ResourceUsage::default(),
            limits,
            thresholds: ResourceThresholds::default(),
        }
    }
    
    /// 设置资源阈值
    pub fn with_thresholds(mut self, thresholds: ResourceThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }
    
    /// 更新内存使用量
    pub fn update_memory_usage(&mut self, memory_used: usize) {
        self.current_usage.peak_memory_bytes = std::cmp::max(self.current_usage.peak_memory_bytes, memory_used);
        
        // 检查是否超出限制
        if memory_used > self.limits.max_memory && self.limits.track_memory {
            self.current_usage.limits_exceeded = true;
            self.current_usage.exceeded_resource = Some(format!(
                "内存使用量超出限制: {}MB > {}MB", 
                memory_used / (1024 * 1024), 
                self.limits.max_memory / (1024 * 1024)
            ));
        }
    }
    
    /// 更新CPU使用时间
    pub fn update_cpu_time(&mut self) {
        let elapsed = self.start_time.elapsed().as_millis() as u64;
        self.current_usage.execution_time_ms = elapsed;
        
        // 检查是否超出限制
        if elapsed > self.limits.max_cpu_time && self.limits.track_cpu_time {
            self.current_usage.limits_exceeded = true;
            self.current_usage.exceeded_resource = Some(format!(
                "CPU时间超出限制: {}ms > {}ms", 
                elapsed, 
                self.limits.max_cpu_time
            ));
        }
    }
    
    /// 更新指令计数
    pub fn update_instruction_count(&mut self, count: u64) {
        self.current_usage.instruction_count = count;
        
        // 检查是否超出限制
        if count > self.limits.max_instruction_count && self.limits.track_instructions {
            self.current_usage.limits_exceeded = true;
            self.current_usage.exceeded_resource = Some(format!(
                "指令计数超出限制: {} > {}", 
                count, 
                self.limits.max_instruction_count
            ));
        }
    }
    
    /// 更新代码大小
    pub fn set_code_size(&mut self, size: usize) {
        self.current_usage.code_size = size;
        
        // 检查是否超出限制
        if size > self.limits.max_code_size {
            self.current_usage.limits_exceeded = true;
            self.current_usage.exceeded_resource = Some(format!(
                "代码大小超出限制: {}KB > {}KB", 
                size / 1024, 
                self.limits.max_code_size / 1024
            ));
        }
    }
    
    /// 更新输出大小
    pub fn update_output_size(&mut self, size: usize) {
        self.current_usage.output_size = size;
        
        // 检查是否超出限制
        if size > self.limits.max_output_size {
            self.current_usage.limits_exceeded = true;
            self.current_usage.exceeded_resource = Some(format!(
                "输出大小超出限制: {}KB > {}KB", 
                size / 1024, 
                self.limits.max_output_size / 1024
            ));
        }
    }
    
    /// 获取当前资源使用情况
    pub fn get_usage(&mut self) -> ResourceUsage {
        // 更新CPU时间
        self.update_cpu_time();
        
        self.current_usage.clone()
    }
    
    /// 检查是否接近资源限制
    pub fn check_warning_thresholds(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        
        // 检查内存使用
        if self.limits.track_memory {
            let memory_ratio = self.current_usage.peak_memory_bytes as f64 / self.limits.max_memory as f64;
            if memory_ratio > self.thresholds.memory_warning_threshold {
                warnings.push(format!(
                    "内存使用接近限制: {:.1}%", 
                    memory_ratio * 100.0
                ));
            }
        }
        
        // 检查CPU时间
        if self.limits.track_cpu_time {
            let cpu_ratio = self.current_usage.execution_time_ms as f64 / self.limits.max_cpu_time as f64;
            if cpu_ratio > self.thresholds.cpu_warning_threshold {
                warnings.push(format!(
                    "CPU时间接近限制: {:.1}%", 
                    cpu_ratio * 100.0
                ));
            }
        }
        
        // 检查指令计数
        if self.limits.track_instructions {
            let inst_ratio = self.current_usage.instruction_count as f64 / self.limits.max_instruction_count as f64;
            if inst_ratio > self.thresholds.instruction_warning_threshold {
                warnings.push(format!(
                    "指令计数接近限制: {:.1}%", 
                    inst_ratio * 100.0
                ));
            }
        }
        
        warnings
    }
    
    /// 检查是否超出资源限制
    pub fn is_exceeded(&self) -> bool {
        self.current_usage.limits_exceeded
    }
    
    /// 重置监控器
    pub fn reset(&mut self) {
        self.start_time = Instant::now();
    }
} 