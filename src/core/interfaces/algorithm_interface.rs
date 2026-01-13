use super::*;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use crate::core::types::{CoreTensorData, DataType};
use crate::error::Result;
use async_trait::async_trait;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmInterface {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub algorithm_type: AlgorithmTypeInterface,
    pub version: String,
    pub author: String,
    pub source_code: String,
    pub language: String,
    pub dependencies: Vec<String>,
    pub input_schema: Vec<TensorSchemaInterface>,
    pub output_schema: Vec<TensorSchemaInterface>,
    pub resource_requirements: ResourceRequirementsInterface,
    pub security_level: SecurityLevelInterface,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlgorithmTypeInterface {
    Classification,
    Regression,
    Clustering,
    DimensionReduction,
    AnomalyDetection,
    Recommendation,
    DataProcessing,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSchemaInterface {
    pub name: String,
    pub shape: Vec<usize>,
    pub data_type: DataType,
    pub optional: bool,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirementsInterface {
    pub max_memory_mb: usize,
    pub max_cpu_percent: f32,
    pub max_execution_time_seconds: u64,
    pub requires_gpu: bool,
    pub max_gpu_memory_mb: Option<usize>,
    pub network_access: bool,
    pub file_system_access: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityLevelInterface {
    Safe,      // 完全安全的算法
    Limited,   // 有限制的算法
    Dangerous, // 需要特殊权限的算法
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmExecutionRequestInterface {
    pub algorithm_id: String,
    pub inputs: Vec<CoreTensorData>,
    pub parameters: HashMap<String, String>,
    pub execution_config: ExecutionConfigInterface,
    pub callback_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfigInterface {
    pub timeout_seconds: Option<u64>,
    pub resource_limits: Option<ResourceRequirementsInterface>,
    pub security_context: Option<SecurityContextInterface>,
    pub debug_mode: bool,
    pub priority: ExecutionPriorityInterface,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContextInterface {
    pub sandbox_type: String,
    pub allowed_operations: Vec<String>,
    pub restricted_modules: Vec<String>,
    pub network_policy: NetworkPolicyInterface,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkPolicyInterface {
    None,           // 无网络访问
    Limited(Vec<String>), // 限制访问的域名列
    Full,           // 完全网络访问
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionPriorityInterface {
    Low,
    Normal,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmExecutionResultInterface {
    pub execution_id: String,
    pub algorithm_id: String,
    pub status: ExecutionStatusInterface,
    pub outputs: Option<Vec<CoreTensorData>>,
    pub error_message: Option<String>,
    pub execution_time_ms: u64,
    pub resource_usage: ResourceUsageInterface,
    pub logs: Vec<LogEntryInterface>,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStatusInterface {
    Pending,
    Running,
    Completed,
    Failed,
    Timeout,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageInterface {
    pub peak_memory_mb: usize,
    pub avg_cpu_percent: f32,
    pub peak_cpu_percent: f32,
    pub gpu_memory_mb: Option<usize>,
    pub disk_read_mb: usize,
    pub disk_write_mb: usize,
    pub network_sent_mb: usize,
    pub network_received_mb: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntryInterface {
    pub level: LogLevelInterface,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub context: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevelInterface {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

/// 算法服务接口
#[async_trait]
pub trait AlgorithmService: Send + Sync {
    async fn register_algorithm(&self, algorithm: AlgorithmInterface) -> Result<String>;
    async fn get_algorithm(&self, algorithm_id: &str) -> Result<Option<AlgorithmInterface>>;
    async fn update_algorithm(&self, algorithm: AlgorithmInterface) -> Result<()>;
    async fn delete_algorithm(&self, algorithm_id: &str) -> Result<()>;
    async fn list_algorithms(&self, algorithm_type: Option<AlgorithmTypeInterface>) -> Result<Vec<AlgorithmInterface>>;
    async fn validate_algorithm(&self, algorithm: &AlgorithmInterface) -> Result<ValidationResult>;
    async fn execute_algorithm(&self, request: AlgorithmExecutionRequestInterface) -> Result<String>;
    async fn get_execution_result(&self, execution_id: &str) -> Result<Option<AlgorithmExecutionResultInterface>>;
    async fn cancel_execution(&self, execution_id: &str) -> Result<()>;
}


