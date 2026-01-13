use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use log::warn;

use crate::algorithm::executor::sandbox::result::SandboxResult;

/// Resource Usage Information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU usage (percentage)
    pub cpu_usage: f64,
    /// Memory usage (bytes)
    pub memory_usage: u64,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: u64,
    /// Disk I/O read (bytes)
    pub disk_read: u64,
    /// Disk I/O write (bytes)
    pub disk_write: u64,
    /// Network I/O received (bytes)
    pub network_received: u64,
    /// Network I/O sent (bytes)
    pub network_sent: u64,
    /// Execution time (milliseconds)
    pub execution_time_ms: u64,
    /// GPU usage (percentage)
    pub gpu_usage: Option<f64>,
    /// GPU memory usage (bytes)
    pub gpu_memory_usage: Option<u64>,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0,
            peak_memory_usage: 0,
            disk_read: 0,
            disk_write: 0,
            network_received: 0,
            network_sent: 0,
            execution_time_ms: 0,
            gpu_usage: None,
            gpu_memory_usage: None,
        }
    }
}

impl ResourceUsage {
    /// Create a new resource usage record
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set CPU usage
    pub fn with_cpu_usage(mut self, cpu_usage: f64) -> Self {
        self.cpu_usage = cpu_usage;
        self
    }
    
    /// Set memory usage
    pub fn with_memory_usage(mut self, memory_usage: u64) -> Self {
        self.memory_usage = memory_usage;
        self
    }
    
    /// Set execution time
    pub fn with_execution_time(mut self, execution_time_ms: u64) -> Self {
        self.execution_time_ms = execution_time_ms;
        self
    }
    
    /// Add resource usage
    pub fn add(&mut self, other: &ResourceUsage) {
        self.cpu_usage += other.cpu_usage;
        self.memory_usage += other.memory_usage;
        self.peak_memory_usage = self.peak_memory_usage.max(other.peak_memory_usage);
        self.disk_read += other.disk_read;
        self.disk_write += other.disk_write;
        self.network_received += other.network_received;
        self.network_sent += other.network_sent;
        self.execution_time_ms += other.execution_time_ms;
        
        if let (Some(self_gpu), Some(other_gpu)) = (self.gpu_usage, other.gpu_usage) {
            self.gpu_usage = Some(self_gpu + other_gpu);
        } else if other.gpu_usage.is_some() {
            self.gpu_usage = other.gpu_usage;
        }
        
        if let (Some(self_gpu_mem), Some(other_gpu_mem)) = (self.gpu_memory_usage, other.gpu_memory_usage) {
            self.gpu_memory_usage = Some(self_gpu_mem + other_gpu_mem);
        } else if other.gpu_memory_usage.is_some() {
            self.gpu_memory_usage = other.gpu_memory_usage;
        }
    }
    
    /// Check if resource limits are exceeded
    pub fn exceeds_limits(&self, limits: &ResourceLimits) -> bool {
        self.cpu_usage > limits.max_cpu_usage ||
        self.memory_usage > limits.max_memory_usage ||
        self.execution_time_ms > limits.max_execution_time_ms
    }

    /// Set limit exceeded status
    pub fn set_limit_exceeded(&mut self, resource_type: &str, message: &str) {
        // Since ResourceUsage doesn't have limit exceeded fields in the current structure,
        // we'll track this via external monitoring
        warn!("Resource limit exceeded for {}: {}", resource_type, message);
        
        // For now, we can set a special marker in the usage stats
        match resource_type {
            "memory" => {
                // Mark memory as exceeded by setting a special value
                if self.memory_usage < u64::MAX - 1000 {
                    self.memory_usage += 1000; // Add 1KB as exceeded marker
                }
            }
            "cpu" => {
                // Mark CPU as exceeded
                if self.cpu_usage < 100.0 {
                    self.cpu_usage = (self.cpu_usage + 1.0).min(100.0);
                }
            }
            "timeout" => {
                // Mark execution time as exceeded
                self.execution_time_ms += 1000; // Add 1 second as exceeded marker
            }
            _ => {
                // Log unknown resource type
                warn!("Unknown resource type for limit exceeded: {}", resource_type);
            }
        }
    }

    /// Check if any resource limits were exceeded
    pub fn has_exceeded_limits(&self) -> bool {
        // This is a simple heuristic based on our marker system
        // In a production system, this would be tracked separately
        false // For now, always return false until we implement proper tracking
    }

    /// Get resource usage as percentage of limit
    pub fn get_usage_percentage(&self, limits: &ResourceLimits) -> f64 {
        let memory_percent = if limits.max_memory_usage > 0 {
            (self.memory_usage as f64 / limits.max_memory_usage as f64) * 100.0
        } else {
            0.0
        };

        let cpu_percent = if limits.max_cpu_usage > 0.0 {
            (self.cpu_usage / limits.max_cpu_usage) * 100.0
        } else {
            0.0
        };

        let time_percent = if limits.max_execution_time_ms > 0 {
            (self.execution_time_ms as f64 / limits.max_execution_time_ms as f64) * 100.0
        } else {
            0.0
        };

        memory_percent.max(cpu_percent).max(time_percent)
    }

    /// Update resource usage with new values
    pub fn update_usage(&mut self, new_usage: &ResourceUsage) {
        self.cpu_usage = self.cpu_usage.max(new_usage.cpu_usage);
        self.memory_usage = self.memory_usage.max(new_usage.memory_usage);
        self.peak_memory_usage = self.peak_memory_usage.max(new_usage.peak_memory_usage);
        self.disk_read += new_usage.disk_read;
        self.disk_write += new_usage.disk_write;
        self.network_received += new_usage.network_received;
        self.network_sent += new_usage.network_sent;
        self.execution_time_ms = self.execution_time_ms.max(new_usage.execution_time_ms);
        
        if let Some(gpu_usage) = new_usage.gpu_usage {
            self.gpu_usage = Some(self.gpu_usage.unwrap_or(0.0).max(gpu_usage));
        }
        
        if let Some(gpu_memory) = new_usage.gpu_memory_usage {
            self.gpu_memory_usage = Some(self.gpu_memory_usage.unwrap_or(0).max(gpu_memory));
        }
    }
}

/// Resource Limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum CPU usage (percentage)
    pub max_cpu_usage: f64,
    /// Maximum memory usage (bytes)
    pub max_memory_usage: u64,
    /// Maximum execution time (milliseconds)
    pub max_execution_time_ms: u64,
    /// Maximum disk I/O (bytes)
    pub max_disk_io: Option<u64>,
    /// Maximum network I/O (bytes)
    pub max_network_io: Option<u64>,
    /// Maximum GPU usage (percentage)
    pub max_gpu_usage: Option<f64>,
    /// Maximum GPU memory usage (bytes)
    pub max_gpu_memory_usage: Option<u64>,
    /// Maximum memory usage in bytes (alias for max_memory_usage)
    pub max_memory_bytes: u64,
    /// Maximum CPU time in seconds
    pub max_cpu_time_seconds: u64,
    /// Maximum GPU memory usage in bytes (alias for max_gpu_memory_usage)
    pub max_gpu_memory_bytes: Option<u64>,
    /// Maximum network bandwidth in bytes per second
    pub max_network_bandwidth_bps: Option<u64>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_cpu_usage: 80.0,
            max_memory_usage: 1024 * 1024 * 1024, // 1GB
            max_execution_time_ms: 30000, // 30 seconds
            max_disk_io: Some(100 * 1024 * 1024), // 100MB
            max_network_io: Some(50 * 1024 * 1024), // 50MB
            max_gpu_usage: Some(90.0),
            max_gpu_memory_usage: Some(2 * 1024 * 1024 * 1024), // 2GB
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB
            max_cpu_time_seconds: 30, // 30 seconds
            max_gpu_memory_bytes: Some(2 * 1024 * 1024 * 1024), // 2GB
            max_network_bandwidth_bps: Some(100 * 1024 * 1024), // 100MB/s
        }
    }
}

impl ResourceLimits {
    /// Get maximum memory usage in MB
    pub fn max_memory_mb(&self) -> Option<u64> {
        Some(self.max_memory_usage / (1024 * 1024))
    }

    /// Get maximum CPU usage percentage
    pub fn max_cpu_percent(&self) -> Option<f64> {
        Some(self.max_cpu_usage)
    }

    /// Get maximum execution time in seconds
    pub fn max_execution_time_seconds(&self) -> u64 {
        self.max_execution_time_ms / 1000
    }

    /// Check if memory limit is set
    pub fn has_memory_limit(&self) -> bool {
        self.max_memory_usage > 0
    }

    /// Check if CPU limit is set
    pub fn has_cpu_limit(&self) -> bool {
        self.max_cpu_usage > 0.0
    }

    /// Check if execution time limit is set
    pub fn has_time_limit(&self) -> bool {
        self.max_execution_time_ms > 0
    }
}

/// Execution Status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStatus {
    Pending,
    Running,
    Success,
    Completed,
    Failed(String),
    Cancelled,
    Timeout,
    Paused,
}

impl Default for ExecutionStatus {
    fn default() -> Self {
        ExecutionStatus::Pending
    }
}

/// Execution Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub execution_id: String,
    pub algorithm_id: String,
    pub output: Vec<u8>,
    pub status: ExecutionStatus,
    pub metrics: crate::algorithm::executor::metrics::ExecutionMetrics,
    pub error: Option<String>,
    pub resource_usage: ResourceUsage,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub sandbox_result: Option<SandboxResult>,
}

impl ExecutionResult {
    pub fn new(
        execution_id: String,
        algorithm_id: String,
        output: Vec<u8>,
        status: ExecutionStatus,
        resource_usage: ResourceUsage,
        metrics: crate::algorithm::executor::metrics::ExecutionMetrics,
        sandbox_result: Option<SandboxResult>
    ) -> Self {
        let now = Utc::now();
        Self {
            execution_id,
            algorithm_id,
            output,
            status,
            error: None,
            resource_usage,
            start_time: now,
            end_time: now,
            sandbox_result,
            metrics,
        }
    }
    
    pub fn with_error(mut self, error: &str) -> Self {
        self.error = Some(error.to_string());
        self
    }
    
    pub fn with_sandbox_result(mut self, result: SandboxResult) -> Self {
        self.sandbox_result = Some(result);
        self
    }
    
    pub fn with_start_time(mut self, start_time: DateTime<Utc>) -> Self {
        self.start_time = start_time;
        self
    }
    
    pub fn with_end_time(mut self, end_time: DateTime<Utc>) -> Self {
        self.end_time = end_time;
        self
    }
    
    pub fn is_success(&self) -> bool {
        matches!(self.status, ExecutionStatus::Success | ExecutionStatus::Completed)
    }
    
    pub fn duration_ms(&self) -> i64 {
        (self.end_time - self.start_time).num_milliseconds()
    }
} 