//! 核心接口模块
//!
//! 统一导出所有核心接口定义

// 接口子模块
pub mod algorithm_interface;
pub mod algorithm_model;
pub mod algorithm;
pub mod api_upload;
pub mod data_interface;
pub mod data;
pub mod gpu_impl;
pub mod monitoring;
pub mod service_registry;
pub mod storage_impl;
pub mod storage_interface;
pub mod storage;

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::Result;

// 重新导出所有接口
pub use algorithm_interface::*;
pub use algorithm_model::*;
pub use algorithm::*;
pub use api_upload::*;
pub use data_interface::*;
pub use data::*;
pub use gpu_impl::*;
// 重新导出监控相关类型，避免 MetricPoint 的模糊重新导出
pub use monitoring::{AlertCondition, SystemHealth, TimeRange, MetricPoint, MemoryUsage, DiskUsage, NetworkStats, GpuUsage};
pub use service_registry::*;
pub use storage_impl::*;
pub use storage_interface::*;
pub use storage::*;

// 重新导出现有的接口定义，避免重复定义
// 注意：避免循环依赖，这些接口应该从core模块内部定义
// MonitoringInterface 已在上面定义，不需要重新导入
pub use crate::core::EventBusInterface;
pub use crate::core::interfaces::data_interface::DataProcessorInterface;

// 重新添加核心接口定义，避免循环依赖
// 模型相关接口stub定义（vecminDB中不需要完整实现，但需要接口定义以保持兼容性）
use async_trait::async_trait;

/// 模型管理器接口（stub定义）
#[async_trait]
pub trait ModelManagerInterface: Send + Sync {
    async fn create_model(&self, model_id: &str, config: &HashMap<String, serde_json::Value>) -> crate::Result<String>;
    async fn get_model(&self, model_id: &str) -> crate::Result<Option<crate::core::types::ModelInfo>>;
    async fn delete_model(&self, model_id: &str) -> crate::Result<()>;
    async fn list_models(&self) -> crate::Result<Vec<crate::core::types::ModelInfo>>;
}

// 训练相关接口已移除：向量数据库系统不需要训练功能
// 如需使用训练功能，请使用 compat 模块中的 stub 类型

pub use crate::interfaces::algorithm::AlgorithmExecutorInterface;

// 重新导出算法相关接口
pub use crate::core::interfaces::algorithm_interface::{
    AlgorithmInterface, TensorSchemaInterface, 
    ResourceRequirementsInterface, SecurityLevelInterface, AlgorithmExecutionRequestInterface,
    ExecutionConfigInterface, SecurityContextInterface, NetworkPolicyInterface,
    AlgorithmExecutionResultInterface, ExecutionStatusInterface, ResourceUsageInterface
};

// 重新导出interfaces模块中的算法接口（避免重复定义）
pub use crate::interfaces::algorithm::{
    AlgorithmTypeInterface, AlgorithmParametersInterface, AlgorithmResultInterface,
    AlgorithmDefinitionInterface, SchemaInterface,
    FieldDefinitionInterface, FieldTypeInterface, ConstraintInterface,
    ValidationResultInterface
};

// 添加缺失的接口定义
/// 模型服务接口
#[async_trait::async_trait]
pub trait ModelServiceInterface: Send + Sync {
    /// 创建模型实例
    async fn create_model_instance(&self, model_id: &str, config: &HashMap<String, serde_json::Value>) -> Result<String>;
    
    /// 获取模型实例
    async fn get_model_instance(&self, instance_id: &str) -> Result<Option<ModelInstance>>;
    
    /// 更新模型实例
    async fn update_model_instance(&self, instance_id: &str, config: &HashMap<String, serde_json::Value>) -> Result<()>;
    
    /// 删除模型实例
    async fn delete_model_instance(&self, instance_id: &str) -> Result<()>;
    
    /// 列出所有模型实例
    async fn list_model_instances(&self, model_id: Option<&str>) -> Result<Vec<ModelInstance>>;
    
    /// 执行模型推理
    async fn execute_inference(&self, instance_id: &str, input_data: &[u8]) -> Result<Vec<u8>>;
    
    /// 获取模型状态
    async fn get_model_status(&self, instance_id: &str) -> Result<ModelInstanceStatus>;
    
    /// 启动模型服务
    async fn start_model_service(&self, instance_id: &str) -> Result<()>;
    
    /// 停止模型服务
    async fn stop_model_service(&self, instance_id: &str) -> Result<()>;
}

/// 存储服务接口
#[async_trait::async_trait]
pub trait StorageServiceInterface: Send + Sync {
    /// 创建存储实例
    async fn create_storage_instance(&self, config: &StorageInstanceConfig) -> Result<String>;
    
    /// 获取存储实例
    async fn get_storage_instance(&self, instance_id: &str) -> Result<Option<StorageInstance>>;
    
    /// 更新存储配置
    async fn update_storage_config(&self, instance_id: &str, config: &StorageInstanceConfig) -> Result<()>;
    
    /// 删除存储实例
    async fn delete_storage_instance(&self, instance_id: &str) -> Result<()>;
    
    /// 列出所有存储实例
    async fn list_storage_instances(&self) -> Result<Vec<StorageInstance>>;
    
    /// 存储数据
    async fn store_data(&self, instance_id: &str, key: &str, data: &[u8]) -> Result<()>;
    
    /// 检索数据
    async fn retrieve_data(&self, instance_id: &str, key: &str) -> Result<Option<Vec<u8>>>;
    
    /// 删除数据
    async fn delete_data(&self, instance_id: &str, key: &str) -> Result<()>;
    
    /// 列出数据键
    async fn list_keys(&self, instance_id: &str, prefix: Option<&str>) -> Result<Vec<String>>;
    
    /// 获取存储统计
    async fn get_storage_stats(&self, instance_id: &str) -> Result<StorageStats>;
}

/// 模型实例
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInstance {
    pub id: String,
    pub model_id: String,
    pub config: HashMap<String, serde_json::Value>,
    pub status: ModelInstanceStatus,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub metrics: Option<ModelInstanceMetrics>,
}

/// 模型实例状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelInstanceStatus {
    Creating,
    Ready,
    Running,
    Stopped,
    Error(String),
}

/// 模型实例指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInstanceMetrics {
    pub inference_count: u64,
    pub avg_inference_time_ms: f64,
    pub error_count: u64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
}

/// 存储实例
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageInstance {
    pub id: String,
    pub config: StorageInstanceConfig,
    pub status: StorageInstanceStatus,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// 存储实例配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageInstanceConfig {
    pub storage_type: StorageType,
    pub connection_string: String,
    pub max_connections: u32,
    pub timeout_seconds: u64,
    pub encryption_enabled: bool,
    pub backup_enabled: bool,
}

/// 存储类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageType {
    RocksDB,
    Redis,
    PostgreSQL,
    MongoDB,
    S3,
}

/// 存储实例状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageInstanceStatus {
    Initializing,
    Ready,
    Error(String),
    Maintenance,
}

/// 存储统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_keys: u64,
    pub total_size_bytes: u64,
    pub read_operations: u64,
    pub write_operations: u64,
    pub delete_operations: u64,
    pub avg_read_time_ms: f64,
    pub avg_write_time_ms: f64,
}
pub use crate::core::interfaces::storage_interface::{StorageService, StorageTransaction, TransactionState, IsolationLevel};

// 重新导出现有的类型定义
pub use crate::core::types::{ProcessedData};
pub use crate::core::common_types::UnifiedModelParameters;
pub use crate::data::loader::types::DataSchema;
pub use crate::data::text_features::preprocessing::preprocessor::PreprocessingConfig;

// 创建类型别名
pub type CoreModelInfo = ModelInfo;

// Stub types for removed modules (compatibility only)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingMetrics {
    pub loss: f64,
    pub accuracy: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingConfiguration {
    pub epochs: usize,
    pub batch_size: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum TrainingStatus {
    Idle,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ModelState {
    Uninitialized,
    Initialized,
    Trained,
}

// 性能指标结构
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PerformanceMetric {
    pub name: String,
    pub value: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub tags: std::collections::HashMap<String, String>,
    pub unit: Option<String>,
    pub description: Option<String>,
}

impl PerformanceMetric {
    pub fn new(name: String, value: f64) -> Self {
        Self {
            name,
            value,
            timestamp: chrono::Utc::now(),
            tags: std::collections::HashMap::new(),
            unit: None,
            description: None,
        }
    }
    
    pub fn with_tags(mut self, tags: std::collections::HashMap<String, String>) -> Self {
        self.tags = tags;
        self
    }
    
    pub fn with_unit(mut self, unit: String) -> Self {
        self.unit = Some(unit);
        self
    }
    
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }
}

// 验证结果定义（如果不存在则创建）
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub score: Option<f32>,
    pub metadata: std::collections::HashMap<String, String>,
}

impl ValidationResult {
    pub fn success() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            score: Some(1.0),
            metadata: std::collections::HashMap::new(),
        }
    }
    
    pub fn failure(errors: Vec<String>) -> Self {
        Self {
            is_valid: false,
            errors,
            warnings: Vec::new(),
            score: Some(0.0),
            metadata: std::collections::HashMap::new(),
        }
    }
    
    /// 添加详细信息
    pub fn with_detail(mut self, key: &str, value: String) -> Self {
        self.metadata.insert(key.to_string(), value);
        self
    }
    
    /// 合并另一个验证结果
    pub fn merge(&mut self, other: ValidationResult) {
        self.is_valid = self.is_valid && other.is_valid;
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
        if let Some(other_score) = other.score {
            self.score = self.score.map(|s| (s + other_score) / 2.0).or(Some(other_score));
        }
        self.metadata.extend(other.metadata);
    }
    
    /// 添加错误
    pub fn add_error(&mut self, error: crate::data::validation::ValidationError) {
        self.is_valid = false;
        self.errors.push(error.message);
    }
}

// 模型定义（如果不存在则创建）
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelDefinition {
    pub name: String,
    pub description: Option<String>,
    pub model_type: String,
    pub architecture: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub hyperparameters: std::collections::HashMap<String, String>,
    pub training_config: Option<TrainingConfiguration>,
    pub metadata: std::collections::HashMap<String, String>,
}

// 算法模型接口（如果不存在则创建）
#[async_trait::async_trait]
pub trait AlgorithmModelInterface: Send + Sync {
    /// 将算法应用到模型
    async fn apply_algorithm_to_model(&self, algorithm_id: &str, model_id: &str) -> Result<String, crate::Error>;
    
    /// 获取模型算法映射
    async fn get_model_algorithms(&self, model_id: &str) -> Result<Vec<String>, crate::Error>;
    
    /// 获取算法模型映射
    async fn get_algorithm_models(&self, algorithm_id: &str) -> Result<Vec<String>, crate::Error>;
    
    /// 移除模型算法映射
    async fn remove_algorithm_from_model(&self, algorithm_id: &str, model_id: &str) -> Result<(), crate::Error>;
    
    /// 验证算法模型兼容性
    async fn validate_compatibility(&self, algorithm_id: &str, model_id: &str) -> Result<ValidationResult, crate::Error>;
    
    /// 训练模型
    async fn train(&self, data: &[f32], labels: &[f32]) -> Result<(), crate::Error>;
    
    /// 预测
    async fn predict(&self, data: &[f32]) -> Result<Vec<f32>, crate::Error>;
    
    /// 保存模型
    async fn save_model(&self, path: &str) -> Result<(), crate::Error>;
    
    /// 加载模型
    async fn load_model(&self, path: &str) -> Result<(), crate::Error>;
    
    /// 获取参数
    async fn get_parameters(&self) -> Result<std::collections::HashMap<String, f32>, crate::Error>;
    
    /// 设置参数
    async fn set_parameters(&self, parameters: std::collections::HashMap<String, f32>) -> Result<(), crate::Error>;
}

/// 监控接口
/// 提供统一的监控能力，支持指标记录、查询、告警管理和系统健康监控
#[async_trait::async_trait]
pub trait MonitoringInterface: Send + Sync {
    /// 记录指标
    async fn record_metric(&self, name: &str, value: f64, tags: &std::collections::HashMap<String, String>) -> Result<(), crate::Error>;
    
    /// 获取指标
    async fn get_metrics(&self, name: &str) -> Result<Vec<crate::core::interfaces::MetricPoint>, crate::Error>;
    
    /// 创建告警
    async fn create_alert(&self, condition: &crate::core::interfaces::monitoring::AlertCondition) -> Result<(), crate::Error>;
    
    /// 获取系统健康状态
    async fn get_system_health(&self) -> Result<crate::core::interfaces::monitoring::SystemHealth, crate::Error>;
    
    /// 获取统计信息
    async fn get_stats(&self) -> Result<std::collections::HashMap<String, f64>, crate::Error>;
    
    /// 获取计数器值
    async fn get_counter(&self, name: &str) -> Result<u64, crate::Error>;
    
    /// 增加计数器
    async fn increment_counter(&self, name: &str, value: u64) -> Result<(), crate::Error>;
    
    /// 统计模型数量
    async fn count_models(&self) -> Result<u64, crate::Error>;
    
    /// 按类型统计模型数量
    async fn count_models_by_type(&self, model_type: &str) -> Result<u64, crate::Error>;
    
    /// 获取最近的模型
    async fn get_recent_models(&self, limit: usize) -> Result<Vec<String>, crate::Error>;
    
    /// 统计任务数量
    async fn count_tasks(&self) -> Result<u64, crate::Error>;
    
    /// 按状态统计任务数量
    async fn count_tasks_by_status(&self, status: &str) -> Result<u64, crate::Error>;
    
    /// 获取最近的任务
    async fn get_recent_tasks(&self, limit: usize) -> Result<Vec<String>, crate::Error>;
    
    /// 获取日志
    async fn get_logs(&self, level: &str, limit: usize) -> Result<Vec<String>, crate::Error>;
    
    /// 统计活跃任务数量
    async fn count_active_tasks(&self) -> Result<u64, crate::Error>;
    
    /// 获取API统计信息
    async fn get_api_stats(&self) -> Result<std::collections::HashMap<String, f64>, crate::Error>;
    
    /// 检查健康状态
    async fn check_health(&self) -> Result<bool, crate::Error>;
}

/// 模型接口trait
#[async_trait::async_trait]
pub trait ModelInterface: Send + Sync {
    fn get_model_id(&self) -> String;
    fn get_model_parameters(&self) -> Vec<f32>;
    fn set_model_parameters(&self, parameters: Vec<f32>);
    fn train_step(&self, inputs: &[f32], targets: &[f32]) -> Result<f32, crate::Error>;
    fn evaluate(&self, inputs: &[f32], targets: &[f32]) -> Result<f32, crate::Error>;
    fn predict(&self, inputs: &[f32]) -> Result<Vec<f32>, crate::Error>;
    fn save_model(&self, path: &str) -> Result<(), crate::Error>;
    fn load_model(&self, path: &str) -> Result<(), crate::Error>;
    
    /// 获取模型ID
    fn id(&self) -> &str;
    
    /// 获取模型名称
    fn name(&self) -> &str;
    
    /// 获取模型描述
    fn description(&self) -> Option<&str>;
    
    /// 获取模型架构
    fn architecture(&self) -> Result<String>;
    
    /// 获取模型类型
    fn model_type(&self) -> crate::model::interface::ModelType;
    
    /// 获取模型状态
    fn state(&self) -> crate::model::interface::ModelState;
    
    /// 设置模型状态
    fn set_state(&mut self, state: crate::model::interface::ModelState) -> Result<()>;
    
    /// 获取模型参数
    async fn get_parameters(&self) -> Result<HashMap<String, crate::model::tensor::TensorData>>;
    
    /// 设置模型参数
    async fn set_parameters(&mut self, parameters: HashMap<String, crate::model::tensor::TensorData>) -> Result<()>;
    
    /// 获取优化器状态
    async fn get_optimizer_state(&self) -> Result<HashMap<String, crate::model::tensor::TensorData>>;
    
    /// 设置优化器状态
    async fn set_optimizer_state(&mut self, state: HashMap<String, crate::model::tensor::TensorData>) -> Result<()>;
    
    /// 获取模型元数据
    async fn get_metadata(&self) -> Result<HashMap<String, String>>;
    
    /// 设置模型元数据
    async fn set_metadata(&mut self, metadata: HashMap<String, String>) -> Result<()>;
    
    /// 获取模型ID
    async fn get_id(&self) -> Result<String>;
    
    /// 获取模型名称
    async fn get_name(&self) -> Result<String>;
    
    /// 获取模型版本
    async fn get_version(&self) -> Result<String>;
    
    /// 获取模型状态
    async fn get_status(&self) -> Result<crate::model::interface::ModelState>;
}

// ModelInfo 已在 core::types 中定义，这里使用类型别名
pub use crate::core::types::ModelInfo;

/// 模型状态接口
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ModelStatusInterface {
    Created,
    Training,
    Trained,
    Failed,
    Deleted,
}

/// 数据信息结构
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DataInfo {
    pub id: String,
    pub name: String,
    pub format: String,
    pub size: u64,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub metadata: std::collections::HashMap<String, String>,
}

/// 算法定义
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AlgorithmDefinition {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub algorithm_type: String,
    pub version: String,
    pub author: String,
    pub source_code: String,
    pub language: String,
    pub dependencies: Vec<String>,
    pub input_schema: Vec<crate::core::interfaces::TensorSchemaInterface>,
    pub output_schema: Vec<crate::core::interfaces::TensorSchemaInterface>,
    pub resource_requirements: crate::core::interfaces::ResourceRequirementsInterface,
    pub security_level: crate::core::interfaces::SecurityLevelInterface,
    pub metadata: std::collections::HashMap<String, String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// 任务数据结构
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TaskData {
    pub task_id: String,
    pub status: String, // TaskStatus stub - 训练相关功能已移除
    pub context: String, // TrainingContext stub - 训练相关功能已移除
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub metadata: std::collections::HashMap<String, String>,
}

/// 优化器接口
/// 提供统一的优化器能力，支持各种优化算法
#[async_trait::async_trait]
pub trait OptimizerInterface: Send + Sync {
    /// 更新参数
    async fn update_parameters(&self, parameters: &mut std::collections::HashMap<String, crate::core::types::CoreTensorData>, gradients: &std::collections::HashMap<String, crate::core::types::CoreTensorData>) -> Result<(), crate::Error>;
    
    /// 获取迭代次数
    async fn get_iterations(&self) -> Result<u32, crate::Error>;
    
    /// 执行优化步骤
    async fn step(&self) -> Result<(), crate::Error>;
    
    /// 获取学习率
    async fn get_learning_rate(&self) -> Result<f64, crate::Error>;
    
    /// 设置学习率
    async fn set_learning_rate(&self, lr: f64) -> Result<(), crate::Error>;
    
    /// 获取优化器状态
    async fn get_optimizer_state(&self) -> Result<OptimizerState, crate::Error>;
    
    /// 重置优化器
    async fn reset(&self) -> Result<(), crate::Error>;
}

/// 损失函数接口
/// 提供统一的损失函数能力，支持各种损失计算
#[async_trait::async_trait]
pub trait LossFunctionInterface: Send + Sync {
    /// 计算损失
    async fn compute_loss(&self, predictions: &crate::core::types::CoreTensorData, targets: &crate::core::types::CoreTensorData) -> Result<f64, crate::Error>;
    
    /// 计算梯度
    async fn compute_gradients(&self, predictions: &crate::core::types::CoreTensorData, targets: &crate::core::types::CoreTensorData) -> Result<std::collections::HashMap<String, crate::core::types::CoreTensorData>, crate::Error>;
    
    /// 获取损失函数名称
    async fn get_loss_function_name(&self) -> Result<String, crate::Error>;
    
    /// 获取损失函数参数
    async fn get_loss_function_parameters(&self) -> Result<std::collections::HashMap<String, f64>, crate::Error>;
    
    /// 设置损失函数参数
    async fn set_loss_function_parameters(&self, parameters: std::collections::HashMap<String, f64>) -> Result<(), crate::Error>;
}

/// 优化器状态
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OptimizerState {
    pub optimizer_type: String,
    pub learning_rate: f64,
    pub momentum: Option<f64>,
    pub weight_decay: Option<f64>,
    pub state_dict: std::collections::HashMap<String, crate::core::types::CoreTensorData>,
}

// 接口模型信息 (用于兼容性)
pub type InterfaceModelInfo = ModelInfo;

// 接口模型状态 (用于兼容性)
pub type InterfaceModelState = String;
