//! Compatibility module
//!
//! Provides stub types for removed training/model modules to maintain
//! backward compatibility during the cleanup process.
//! 
//! These types will be gradually removed as we refactor the codebase.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Model-related stub types
// ============================================================================

pub type Model = ModelStub;
pub type Layer = String;
pub type Connection = String;
pub type ConnectionType = String;
pub type LayerId = String;
pub type Activation = String;
pub type Padding = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStub {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: String,
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelState {
    Uninitialized,
    Initialized,
    Trained,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    pub data: HashMap<String, Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    pub layers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStatus {
    Uninitialized,
    Initialized,
    Training,
    Trained,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartModelParameters {
    pub data: HashMap<String, Vec<f32>>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct ModelMemoryMonitor {
    pub current_usage: usize,
    pub peak_usage: usize,
}

impl ModelMemoryMonitor {
    pub fn new() -> Self {
        Self {
            current_usage: 0,
            peak_usage: 0,
        }
    }
}

// Tensor-related types
pub type TensorData = Vec<f32>;
pub type DataType = String;

pub type ManagedTensorData = Vec<f32>;
pub type TensorDataType = String;

// ============================================================================
// Training-related stub types
// ============================================================================
// 注意：向量数据库系统不需要训练功能，这些类型仅用于向后兼容

use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 10,
            batch_size: 32,
            learning_rate: 0.001,
        }
    }
}

/// 训练结果详情（stub类型，向量数据库系统不需要训练功能）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResultDetail {
    pub id: String,
    pub model_id: String,
    pub model_name: String,
    pub status: String, // 简化为 String，避免依赖 TrainingStatus
    pub metrics: HashMap<String, f64>,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub duration_seconds: Option<u64>,
    pub epochs_completed: usize,
    pub total_epochs: usize,
    pub loss_history: Vec<f64>,
    pub accuracy_history: Vec<f64>,
    pub config: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub loss: f64,
    pub accuracy: f64,
    pub epoch: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParams {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub epochs: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    pub metrics: TrainingMetrics,
    pub success: bool,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingStatus {
    Idle,
    Running,
    Completed,
    Failed,
}

// ============================================================================
// Loss function stub
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossFunction {
    MSE,
    CrossEntropy,
    Custom(String),
}

impl Default for LossFunction {
    fn default() -> Self {
        Self::MSE
    }
}

// ============================================================================
// Re-exports for convenience
// ============================================================================

pub mod loss {
    pub mod base {
        pub use super::super::LossFunction;
    }
}

pub mod types {
    pub use super::{TrainingMetrics, TrainingParams, TrainingResult, TrainingStatus};
}

pub mod state {
    pub use super::ModelState;
}

pub mod parameters {
    pub use super::{ModelParameters, TrainingMetrics};
}

// Additional stubs for various modules
pub mod interface {
    use super::*;
    
    pub type ModelType = String;
    pub use super::ModelState;
}

pub mod tensor {
    use super::{Serialize, Deserialize, HashMap};
    
    // TensorData stub - 简化的张量数据结构
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TensorData {
        pub data: TensorValues,
        pub shape: Vec<usize>,
        pub dtype: DataType,
        pub metadata: HashMap<String, String>,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum TensorValues {
        F32(Vec<f32>),
        F64(Vec<f64>),
        I32(Vec<i32>),
        I64(Vec<i64>),
        U8(Vec<u8>),
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum DataType {
        Float32,
        Float64,
        Int32,
        Int64,
        UInt8,
    }
}

pub mod memory_management {
    pub type ManagedTensorData = Vec<f32>;
}

pub mod manager {
    pub mod traits {
        use std::path::Path;
        use crate::error::Result;
        use super::super::{Model, ModelParameters, ModelArchitecture};
        use crate::storage::models::implementation::{ModelInfo, StorageFormat};
        
        pub trait ModelManager: Send + Sync {
            // 基本 CRUD 操作
            fn create_model(&self, name: &str, model_type: &str, description: Option<&str>) -> Result<Model>;
            fn get_model(&self, model_id: &str) -> Result<Option<Model>>;
            fn update_model(&self, model: &Model) -> Result<()>;
            fn delete_model(&self, model_id: &str) -> Result<bool>;
            fn save_model_parameters(&self, model_id: &str, parameters: &ModelParameters) -> Result<()>;
            fn get_model_parameters(&self, model_id: &str) -> Result<Option<ModelParameters>>;
            fn save_model_architecture(&self, model_id: &str, architecture: &ModelArchitecture) -> Result<()>;
            fn get_model_architecture(&self, model_id: &str) -> Result<Option<ModelArchitecture>>;
            fn list_models(&self) -> Result<Vec<ModelInfo>>;
            fn import_model(&self, path: &Path, format: StorageFormat) -> Result<String>;
            fn export_model(&self, model_id: &str, format: StorageFormat, path: &Path) -> Result<()>;
            fn search_models(&self, query: &str) -> Result<Vec<ModelInfo>>;
            fn find_models_by_tag(&self, tag: &str) -> Result<Vec<ModelInfo>>;
            
            // 版本管理
            fn create_model_version(&self, model_id: &str, version: &str, description: Option<&str>) -> Result<String>;
            fn list_model_versions(&self, model_id: &str) -> Result<Vec<crate::algorithm::manager::models::common::ModelVersionInfo>>;
            fn switch_to_version(&self, model_id: &str, version: &str) -> Result<()>;
            fn delete_model_version(&self, model_id: &str, version: &str) -> Result<bool>;
            
            // 健康检查和监控
            fn health_check(&self, model_id: &str) -> Result<crate::algorithm::manager::models::common::ModelHealthStatus>;
            fn get_model_metrics(&self, model_id: &str) -> Result<crate::algorithm::manager::models::common::ModelPerformanceMetrics>;
            fn start_monitoring(&self, model_id: &str, config: &crate::algorithm::manager::models::common::MonitoringConfig) -> Result<String>;
            fn stop_monitoring(&self, model_id: &str, monitor_id: &str) -> Result<()>;
            fn start_monitoring_task(&self, model_id: &str, monitor_id: &str, config: &crate::algorithm::manager::models::common::MonitoringConfig) -> Result<crate::algorithm::manager::models::common::MonitoringTask>;
            fn start_performance_monitoring(&self, model_id: &str, monitor_id: &str) -> Result<()>;
            fn start_accuracy_monitoring(&self, model_id: &str, monitor_id: &str) -> Result<()>;
            fn start_resource_monitoring(&self, model_id: &str, monitor_id: &str) -> Result<()>;
            fn start_prediction_monitoring(&self, model_id: &str, monitor_id: &str) -> Result<()>;
            
            // 备份和恢复
            fn create_backup(&self, model_id: &str, backup_type: crate::algorithm::manager::models::common::BackupType) -> Result<String>;
            fn restore_from_backup(&self, model_id: &str, backup_id: &str) -> Result<()>;
            fn list_backups(&self, model_id: &str) -> Result<Vec<crate::algorithm::manager::models::common::BackupInfo>>;
            fn delete_backup(&self, backup_id: &str) -> Result<bool>;
            
            // 模型优化
            fn warm_up_model(&self, model_id: &str, warm_up_data: &crate::data::DataBatch) -> Result<()>;
            fn compress_model(&self, model_id: &str, compression_config: &crate::algorithm::manager::models::common::CompressionConfig) -> Result<String>;
            fn quantize_model(&self, model_id: &str, quantization_config: &crate::algorithm::manager::models::common::QuantizationConfig) -> Result<String>;
            fn optimize_model(&self, model_id: &str, optimization_config: &crate::algorithm::manager::models::common::OptimizationConfig) -> Result<String>;
            
            // 依赖和完整性
            fn get_model_dependencies(&self, model_id: &str) -> Result<Vec<crate::algorithm::manager::models::common::ModelDependency>>;
            fn validate_model_integrity(&self, model_id: &str) -> Result<crate::algorithm::manager::models::common::IntegrityCheckResult>;
            
            // A/B 测试
            fn create_ab_test(&self, model_a_id: &str, model_b_id: &str, test_config: &crate::algorithm::manager::models::common::ABTestConfig) -> Result<String>;
            fn get_ab_test_results(&self, test_id: &str) -> Result<crate::algorithm::manager::models::common::ABTestResults>;
            
            // 部署
            fn deploy_model(&self, model_id: &str, deployment_config: &crate::algorithm::manager::models::common::DeploymentConfig) -> Result<String>;
            fn undeploy_model(&self, model_id: &str, deployment_id: &str) -> Result<()>;
            fn get_deployment_status(&self, deployment_id: &str) -> Result<crate::algorithm::manager::models::common::DeploymentStatus>;
            
            // 推理
            fn inference(&self, model_id: &str, input_data: &crate::data::DataBatch) -> Result<crate::data::DataBatch>;
            fn batch_inference(&self, model_id: &str, input_batches: &[crate::data::DataBatch]) -> Result<Vec<crate::data::DataBatch>>;
            fn async_inference(&self, model_id: &str, input_data: &crate::data::DataBatch) -> Result<String>;
            fn get_inference_result(&self, inference_id: &str) -> Result<Option<crate::data::DataBatch>>;
        }
    }
    
    pub mod enums {
        use serde::{Deserialize, Serialize};
        
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
        pub enum BackupType {
            Full,
            Incremental,
            Differential,
        }
    }
    
    pub mod metrics {
        use serde::{Deserialize, Serialize};
        use std::collections::HashMap;
        
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct ModelHealthStatus {
            pub is_healthy: bool,
            pub status_message: String,
        }
        
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct ModelPerformanceMetrics {
            pub latency_ms: f64,
            pub throughput: f64,
            pub accuracy: f64,
        }
        
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct ModelMonitoringConfig {
            pub enabled: bool,
            pub interval_seconds: u64,
        }
        
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct IntegrityCheckResult {
            pub is_valid: bool,
            pub errors: Vec<String>,
        }
        
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct ABTestResults {
            pub variant_a_metrics: HashMap<String, f64>,
            pub variant_b_metrics: HashMap<String, f64>,
        }
    }
    
    pub mod config {
        use serde::{Deserialize, Serialize};
        
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct MonitoringConfig {
            pub enabled: bool,
            pub interval_seconds: u64,
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
            pub enabled: bool,
            pub level: String,
        }
        
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct ABTestConfig {
            pub enabled: bool,
            pub variants: Vec<String>,
        }
        
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct DeploymentConfig {
            pub strategy: String,
            pub replicas: u32,
        }
    }
}

pub mod coordinator {
    pub use super::TrainingStatus;
}

pub mod config {
    pub use super::TrainingConfig;
}

pub mod engine {
    pub type TaskStatus = String;
}

pub mod unified {
    pub use super::ModelArchitecture;
    pub struct UnifiedModelAdapter;
}

pub mod adapters {
    pub struct ModelInterfaceAdapter;
}

// ============================================================================
// data_to_model_engine stub types
// ============================================================================

pub mod data_to_model_engine {
    use super::{Serialize, Deserialize};
    
    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    pub struct ConversionConfig {
        pub auto_detect_schema: bool,
        pub normalization_enabled: bool,
        pub feature_extraction_enabled: bool,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct DataToModelConversionResult {
        pub success: bool,
        pub model_id: Option<String>,
        pub message: String,
    }
    
    impl Default for DataToModelConversionResult {
        fn default() -> Self {
            Self {
                success: false,
                model_id: None,
                message: String::new(),
            }
        }
    }
}

// ============================================================================
// end_to_end_pipeline stub types
// ============================================================================

pub mod end_to_end_pipeline {
    use super::{Serialize, Deserialize};
    
    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    pub struct PipelineConfig {
        pub auto_train: bool,
        pub auto_deploy: bool,
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EndToEndTrainingResult {
        pub success: bool,
        pub model_id: Option<String>,
        pub training_time_seconds: f64,
        pub message: String,
    }
    
    impl Default for EndToEndTrainingResult {
        fn default() -> Self {
            Self {
                success: false,
                model_id: None,
                training_time_seconds: 0.0,
                message: String::new(),
            }
        }
    }
    
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum DataSource {
        File { path: String },
        Database { connection_string: String },
        Api { url: String },
    }
    
    impl Default for DataSource {
        fn default() -> Self {
            DataSource::File { path: String::new() }
        }
    }
}

// ============================================================================
// task_scheduler stub types
// ============================================================================

pub mod task_scheduler {
    pub mod core {
        use serde::{Deserialize, Serialize};
        use std::str::FromStr;
        
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub struct TaskId(pub String);
        
        impl FromStr for TaskId {
            type Err = String;
            
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                Ok(TaskId(s.to_string()))
            }
        }
        
        impl From<String> for TaskId {
            fn from(s: String) -> Self {
                TaskId(s)
            }
        }
        
        impl From<&str> for TaskId {
            fn from(s: &str) -> Self {
                TaskId(s.to_string())
            }
        }
    }
}

// ============================================================================
// API routes stub types
// ============================================================================

pub mod api {
    pub mod routes {
        pub mod system {
            pub mod types {
                use serde::{Deserialize, Serialize};
                use chrono::{DateTime, Utc};
                
                #[derive(Debug, Clone, Serialize, Deserialize)]
                pub struct StorageMetrics {
                    pub total_objects: u64,
                    pub total_size_bytes: u64,
                    pub read_operations: u64,
                    pub write_operations: u64,
                    pub operations_per_second: f64,
                }
                
                #[derive(Debug, Clone, Serialize, Deserialize)]
                pub struct StorageStatsResponse {
                    pub timestamp: DateTime<Utc>,
                    pub data_files_count: u64,
                    pub model_files_count: u64,
                    pub algorithm_files_count: u64,
                    pub total_size_bytes: u64,
                    pub usage_by_type: std::collections::HashMap<String, u64>,
                }
            }
        }
    }
}
