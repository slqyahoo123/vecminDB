use serde::{Serialize, Deserialize};
use std::time::{Duration, SystemTime};

/// 执行器性能指标
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ExecutorMetrics {
    /// 总执行次数
    pub total_executions: usize,
    /// 成功执行数
    pub successful_executions: usize,
    /// 失败执行数
    pub failed_executions: usize,
    /// 编译缓存命中次数
    pub cache_hits: usize,
    /// 编译缓存未命中次数
    pub cache_misses: usize,
    /// 平均编译时间(毫秒)
    pub avg_compilation_time_ms: f64,
    /// 平均执行时间(毫秒)
    pub avg_execution_time_ms: f64,
    /// 资源限制超出次数
    pub resource_limit_exceeded_count: usize,
    /// 超时次数
    pub timeout_count: usize,
    /// 平均内存使用量(字节)
    pub avg_memory_usage: f64,
    /// 最大内存使用量(字节)
    pub max_memory_usage: usize,
    /// 平均指令计数
    pub avg_instruction_count: f64,
    /// 最大指令计数
    pub max_instruction_count: u64,
    /// 指标重置时间
    pub reset_time: Option<SystemTime>,
}

impl ExecutorMetrics {
    /// 创建新的性能指标实例
    pub fn new() -> Self {
        let mut metrics = Self::default();
        metrics.reset_time = Some(SystemTime::now());
        metrics
    }
    
    /// 重置所有指标
    pub fn reset(&mut self) {
        *self = Self::new();
    }
    
    /// 记录执行开始
    pub fn record_execution_start(&mut self) {
        self.total_executions += 1;
    }
    
    /// 记录执行成功
    pub fn record_execution_success(&mut self) {
        self.successful_executions += 1;
    }
    
    /// 记录执行失败
    pub fn record_execution_failure(&mut self) {
        self.failed_executions += 1;
    }
    
    /// 记录缓存命中
    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }
    
    /// 记录缓存未命中
    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }
    
    /// 记录编译时间
    pub fn record_compilation_time(&mut self, time_ms: f64) {
        let total_compilations = self.cache_misses;
        if total_compilations == 0 {
            self.avg_compilation_time_ms = time_ms;
        } else {
            self.avg_compilation_time_ms = 
                (self.avg_compilation_time_ms * (total_compilations - 1) as f64 + time_ms) /
                total_compilations as f64;
        }
    }
    
    /// 记录执行时间
    pub fn record_execution_time(&mut self, time_ms: f64) {
        let total = self.total_executions;
        if total == 0 {
            self.avg_execution_time_ms = time_ms;
        } else {
            self.avg_execution_time_ms = 
                (self.avg_execution_time_ms * (total - 1) as f64 + time_ms) /
                total as f64;
        }
    }
    
    /// 记录资源限制超出
    pub fn record_resource_limit_exceeded(&mut self) {
        self.resource_limit_exceeded_count += 1;
    }
    
    /// 记录超时
    pub fn record_timeout(&mut self) {
        self.timeout_count += 1;
    }
    
    /// 记录内存使用
    pub fn record_memory_usage(&mut self, memory_bytes: usize) {
        let total = self.total_executions;
        
        // 更新平均内存使用
        if total == 0 {
            self.avg_memory_usage = memory_bytes as f64;
        } else {
            self.avg_memory_usage = 
                (self.avg_memory_usage * (total - 1) as f64 + memory_bytes as f64) /
                total as f64;
        }
        
        // 更新最大内存使用
        self.max_memory_usage = std::cmp::max(self.max_memory_usage, memory_bytes);
    }
    
    /// 记录指令计数
    pub fn record_instruction_count(&mut self, count: u64) {
        let total = self.total_executions;
        
        // 更新平均指令计数
        if total == 0 {
            self.avg_instruction_count = count as f64;
        } else {
            self.avg_instruction_count = 
                (self.avg_instruction_count * (total - 1) as f64 + count as f64) /
                total as f64;
        }
        
        // 更新最大指令计数
        self.max_instruction_count = std::cmp::max(self.max_instruction_count, count);
    }
    
    /// 获取指标运行时长
    pub fn uptime(&self) -> Duration {
        if let Some(reset_time) = self.reset_time {
            SystemTime::now()
                .duration_since(reset_time)
                .unwrap_or(Duration::from_secs(0))
        } else {
            Duration::from_secs(0)
        }
    }
    
    /// 获取缓存命中率
    pub fn cache_hit_ratio(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            self.cache_hits as f64 / total as f64
        }
    }
    
    /// 获取成功率
    pub fn success_ratio(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            self.successful_executions as f64 / self.total_executions as f64
        }
    }
    
    /// 生成性能报告
    pub fn generate_report(&self) -> String {
        let uptime = self.uptime();
        let uptime_secs = uptime.as_secs();
        let days = uptime_secs / 86400;
        let hours = (uptime_secs % 86400) / 3600;
        let minutes = (uptime_secs % 3600) / 60;
        let seconds = uptime_secs % 60;
        
        format!(
            "执行器性能指标报告\n\
            ==================\n\
            运行时间: {}天 {:02}:{:02}:{:02}\n\
            总执行次数: {}\n\
            成功执行数: {} ({:.1}%)\n\
            失败执行数: {} ({:.1}%)\n\
            缓存命中率: {:.1}%\n\
            平均编译时间: {:.2}ms\n\
            平均执行时间: {:.2}ms\n\
            资源限制超出次数: {}\n\
            超时次数: {}\n\
            平均内存使用: {:.2}MB\n\
            最大内存使用: {:.2}MB\n\
            平均指令计数: {:.0}\n\
            最大指令计数: {}\n",
            days, hours, minutes, seconds,
            self.total_executions,
            self.successful_executions, self.success_ratio() * 100.0,
            self.failed_executions, (1.0 - self.success_ratio()) * 100.0,
            self.cache_hit_ratio() * 100.0,
            self.avg_compilation_time_ms,
            self.avg_execution_time_ms,
            self.resource_limit_exceeded_count,
            self.timeout_count,
            self.avg_memory_usage / (1024.0 * 1024.0),
            self.max_memory_usage as f64 / (1024.0 * 1024.0),
            self.avg_instruction_count,
            self.max_instruction_count
        )
    }
}

/// 执行指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    /// 执行时间
    pub execution_time: Duration,
    
    /// CPU使用率(百分比)
    pub cpu_usage: f64,
    
    /// 内存使用量(字节)
    pub memory_usage: usize,
    
    /// IO读取字节数
    pub io_reads: u64,
    
    /// IO写入字节数
    pub io_writes: u64,
}

impl Default for ExecutionMetrics {
    fn default() -> Self {
        Self {
            execution_time: Duration::from_secs(0),
            cpu_usage: 0.0,
            memory_usage: 0,
            io_reads: 0u64,
            io_writes: 0u64,
        }
    }
}

impl ExecutionMetrics {
    /// 创建新的指标对象
    pub fn new() -> Self {
        Self::default()
    }
    
    /// 合并多个指标
    pub fn merge(&mut self, other: &Self) {
        self.execution_time = std::cmp::max(self.execution_time, other.execution_time);
        self.cpu_usage = (self.cpu_usage + other.cpu_usage) / 2.0;
        self.memory_usage = std::cmp::max(self.memory_usage, other.memory_usage);
        self.io_reads += other.io_reads;
        self.io_writes += other.io_writes;
    }
    
    /// 格式化执行时间
    pub fn format_execution_time(&self) -> String {
        let millis = self.execution_time.as_millis();
        if millis < 1000 {
            format!("{}ms", millis)
        } else if millis < 60_000 {
            format!("{:.2}s", millis as f64 / 1000.0)
        } else {
            format!("{:.2}m", millis as f64 / 60_000.0)
        }
    }
    
    /// 格式化内存使用量
    pub fn format_memory_usage(&self) -> String {
        let bytes = self.memory_usage as f64;
        if bytes < 1024.0 {
            format!("{:.0}B", bytes)
        } else if bytes < 1024.0 * 1024.0 {
            format!("{:.2}KB", bytes / 1024.0)
        } else if bytes < 1024.0 * 1024.0 * 1024.0 {
            format!("{:.2}MB", bytes / (1024.0 * 1024.0))
        } else {
            format!("{:.2}GB", bytes / (1024.0 * 1024.0 * 1024.0))
        }
    }
    
    /// 格式化IO读取
    pub fn format_io_reads(&self) -> String {
        self.format_bytes(self.io_reads as usize)
    }
    
    /// 格式化IO写入
    pub fn format_io_writes(&self) -> String {
        self.format_bytes(self.io_writes as usize)
    }
    
    /// 格式化字节数
    fn format_bytes(&self, bytes: usize) -> String {
        let bytes = bytes as f64;
        if bytes < 1024.0 {
            format!("{:.0}B", bytes)
        } else if bytes < 1024.0 * 1024.0 {
            format!("{:.2}KB", bytes / 1024.0)
        } else if bytes < 1024.0 * 1024.0 * 1024.0 {
            format!("{:.2}MB", bytes / (1024.0 * 1024.0))
        } else {
            format!("{:.2}GB", bytes / (1024.0 * 1024.0 * 1024.0))
        }
    }
    
    /// 获取简要报告
    pub fn get_summary(&self) -> String {
        format!(
            "执行时间: {}, CPU使用率: {:.1}%, 内存: {}, IO读取: {}, IO写入: {}",
            self.format_execution_time(),
            self.cpu_usage,
            self.format_memory_usage(),
            self.format_io_reads(),
            self.format_io_writes()
        )
    }
} 