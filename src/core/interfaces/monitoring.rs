use std::collections::HashMap;
use chrono::{DateTime, Utc};
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use crate::error::Result;

/// 性能监控接口
#[async_trait]
pub trait PerformanceMonitor: Send + Sync {
    async fn record_metric(&self, name: &str, value: f64, tags: &HashMap<String, String>) -> Result<()>;
    async fn get_metrics(&self, name: &str, time_range: &TimeRange) -> Result<Vec<MetricPoint>>;
    async fn create_alert(&self, condition: &AlertCondition) -> Result<String>;
    async fn get_system_health(&self) -> Result<SystemHealth>;
}

/// 资源监控接口
#[async_trait]
pub trait ResourceMonitor: Send + Sync {
    async fn get_cpu_usage(&self) -> Result<f32>;
    async fn get_memory_usage(&self) -> Result<MemoryUsage>;
    async fn get_disk_usage(&self) -> Result<DiskUsage>;
    async fn get_network_stats(&self) -> Result<NetworkStats>;
    async fn get_gpu_usage(&self) -> Result<Option<GpuUsage>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertCondition {
    pub metric_name: String,
    pub operator: String, // "gt", "lt", "eq", etc.
    pub threshold: f64,
    pub duration: u64, // seconds
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub overall_status: String,
    pub components: HashMap<String, ComponentHealth>,
    pub last_updated: DateTime<Utc>,
}

// ComponentHealth 已在 core::types 中定义，这里引用它
pub use crate::core::types::ComponentHealth;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub total: usize,
    pub used: usize,
    pub available: usize,
    pub percentage: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskUsage {
    pub total: u64,
    pub used: u64,
    pub available: u64,
    pub percentage: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuUsage {
    pub utilization: f32,
    pub memory_used: usize,
    pub memory_total: usize,
    pub temperature: f32,
}


