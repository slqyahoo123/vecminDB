use super::*;
use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use crate::error::Result;
use crate::core::types::CoreTensorData;

/// 算法编译器接口
#[async_trait]
pub trait AlgorithmCompiler: Send + Sync {
    async fn compile(&self, source_code: &str, language: &str) -> Result<CompiledAlgorithm>;
    async fn validate_syntax(&self, source_code: &str, language: &str) -> Result<ValidationResult>;
    async fn optimize(&self, algorithm: &CompiledAlgorithm) -> Result<CompiledAlgorithm>;
}

/// 算法运行时接口
#[async_trait]
pub trait AlgorithmRuntime: Send + Sync {
    async fn execute(&self, algorithm: &CompiledAlgorithm, inputs: &[CoreTensorData]) -> Result<ExecutionResult>;
    async fn get_memory_usage(&self) -> Result<usize>;
    async fn get_execution_stats(&self) -> Result<ExecutionStats>;
    async fn set_resource_limits(&self, limits: &ResourceLimits) -> Result<()>;
}

/// 安全沙箱接口
#[async_trait]
pub trait SecuritySandbox: Send + Sync {
    async fn create_sandbox(&self, config: &SandboxConfig) -> Result<String>;
    async fn execute_in_sandbox(&self, sandbox_id: &str, algorithm: &CompiledAlgorithm, inputs: &[CoreTensorData]) -> Result<ExecutionResult>;
    async fn destroy_sandbox(&self, sandbox_id: &str) -> Result<()>;
    async fn get_sandbox_status(&self, sandbox_id: &str) -> Result<SandboxStatus>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledAlgorithm {
    pub algorithm_id: String,
    pub bytecode: Vec<u8>,
    pub metadata: AlgorithmMetadata,
    pub dependencies: Vec<String>,
    pub resource_requirements: ResourceRequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmMetadata {
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub input_schema: Vec<TensorSchema>,
    pub output_schema: Vec<TensorSchema>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSchema {
    pub name: String,
    pub shape: Vec<i32>, // -1 for dynamic dimensions
    pub dtype: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub max_memory: usize,
    pub max_cpu_time: u64,
    pub max_gpu_memory: Option<usize>,
    pub network_access: bool,
    pub file_system_access: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub outputs: Vec<CoreTensorData>,
    pub execution_time: u64,
    pub memory_used: usize,
    pub metadata: HashMap<String, String>,
    pub logs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStats {
    pub total_executions: u64,
    pub successful_executions: u64,
    pub failed_executions: u64,
    pub average_execution_time: f64,
    pub peak_memory_usage: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_memory: usize,
    pub max_cpu_time: u64,
    pub max_network_requests: Option<u32>,
    pub max_file_operations: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    pub resource_limits: ResourceLimits,
    pub network_policy: NetworkPolicy,
    pub file_system_policy: FileSystemPolicy,
    pub security_level: SecurityLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicy {
    pub allow_outbound: bool,
    pub allowed_hosts: Vec<String>,
    pub blocked_ports: Vec<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSystemPolicy {
    pub read_only: bool,
    pub allowed_paths: Vec<String>,
    pub blocked_paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityLevel {
    Low,
    Medium,
    High,
    Paranoid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxStatus {
    pub sandbox_id: String,
    pub status: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub resource_usage: ResourceUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub memory_used: usize,
    pub cpu_time_used: u64,
    pub network_requests: u32,
    pub file_operations: u32,
}


