// 通用模型管理功能

use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use serde::{Serialize, Deserialize};
use crate::compat::{TrainingStatus, TrainingConfig};

/// 模型管理器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManagerConfig {
    pub cache_size: usize,
    pub cache_ttl: Duration,
    pub max_concurrent_training: usize,
    pub backup_retention_days: u32,
    pub enable_monitoring: bool,
}

impl Default for ModelManagerConfig {
    fn default() -> Self {
        Self {
            cache_size: 100,
            cache_ttl: Duration::from_secs(3600), // 1小时
            max_concurrent_training: 5,
            backup_retention_days: 30,
            enable_monitoring: true,
        }
    }
}

/// 模型缓存条目
#[derive(Clone)]
pub struct ModelCacheEntry {
    pub model: std::sync::Arc<crate::model::Model>,
    pub last_accessed: SystemTime,
    pub access_count: u64,
}

/// 训练会话
#[derive(Debug, Clone)]
pub struct TrainingSession {
    pub id: String,
    pub model_id: String,
    pub status: TrainingStatus,
    pub config: TrainingConfig,
    pub start_time: SystemTime,
    pub progress: f32,
    pub metrics: HashMap<String, f64>,
}

/// 监控任务
#[derive(Debug, Clone)]
pub struct MonitoringTask {
    pub id: String,
    pub model_id: String,
    pub config: crate::model::manager::metrics::ModelMonitoringConfig,
    pub status: crate::algorithm::manager::types::model::MonitoringStatus,
    pub metrics: HashMap<String, f64>,
}

// 添加缺失的类型定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersionInfo {
    pub version: String,
    pub created_at: SystemTime,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelHealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceMetrics {
    pub latency_ms: f64,
    pub throughput: f64,
    pub accuracy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enabled: bool,
    pub interval_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackupType {
    Full,
    Incremental,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub enabled: bool,
    pub algorithm: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub enabled: bool,
    pub bits: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub compression: CompressionConfig,
    pub quantization: QuantizationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDependency {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityCheckResult {
    pub passed: bool,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestConfig {
    pub enabled: bool,
    pub traffic_split: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABTestResults {
    pub variant_a_metrics: ModelPerformanceMetrics,
    pub variant_b_metrics: ModelPerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfig {
    pub replicas: u32,
    pub resources: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStatus {
    Pending,
    Running,
    Failed,
    Completed,
}

/// 备份信息
#[derive(Debug, Clone)]
pub struct BackupInfo {
    pub id: String,
    pub model_id: String,
    pub backup_type: crate::model::manager::enums::BackupType,
    pub created_at: SystemTime,
    pub size_bytes: u64,
    pub checksum: String,
}

/// 部署信息
#[derive(Debug, Clone)]
pub struct DeploymentInfo {
    pub id: String,
    pub model_id: String,
    pub status: crate::algorithm::manager::types::model::DeploymentStatus,
    pub config: crate::model::manager::config::DeploymentConfig,
    pub endpoint: String,
    pub created_at: SystemTime,
}

/// A/B测试信息
#[derive(Debug, Clone)]
pub struct ABTestInfo {
    pub id: String,
    pub model_a_id: String,
    pub model_b_id: String,
    pub config: crate::model::manager::config::ABTestConfig,
    pub status: crate::algorithm::manager::types::model::ABTestStatus,
    pub results: Option<crate::model::manager::metrics::ABTestResults>,
}

/// 推理结果
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub id: String,
    pub model_id: String,
    pub input_hash: String,
    pub output: crate::data::DataBatch,
    pub created_at: SystemTime,
} 