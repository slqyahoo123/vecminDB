use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use crate::Result;

/// 算法类型接口
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AlgorithmTypeInterface {
    LinearRegression,
    NeuralNetwork,
    DecisionTree,
    SVM,
    KMeans,
    Custom(String),
}

/// 算法参数接口
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmParametersInterface {
    pub algorithm_type: AlgorithmTypeInterface,
    pub parameters: HashMap<String, serde_json::Value>,
    pub hyperparameters: HashMap<String, serde_json::Value>,
    pub metadata: HashMap<String, String>,
}

/// 算法执行结果接口
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmResultInterface {
    pub success: bool,
    pub result_data: Option<serde_json::Value>,
    pub error_message: Option<String>,
    pub execution_time: f64,
    pub resource_usage: ResourceUsageInterface,
    pub metadata: HashMap<String, String>,
}

/// 资源使用情况接口
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageInterface {
    pub memory_used: usize,
    pub cpu_time: f64,
    pub gpu_time: Option<f64>,
    pub disk_io: usize,
    pub network_io: usize,
}

/// 算法管理接口
#[async_trait]
pub trait AlgorithmManagerInterface: Send + Sync {
    /// 注册算法
    async fn register_algorithm(&self, algorithm_id: &str, algorithm: AlgorithmDefinitionInterface) -> Result<()>;
    
    /// 获取算法定义
    async fn get_algorithm(&self, algorithm_id: &str) -> Result<Option<AlgorithmDefinitionInterface>>;
    
    /// 执行算法
    async fn execute_algorithm(&self, algorithm_id: &str, inputs: HashMap<String, serde_json::Value>) -> Result<AlgorithmResultInterface>;
    
    /// 删除算法
    async fn delete_algorithm(&self, algorithm_id: &str) -> Result<()>;
    
    /// 列出所有算法
    async fn list_algorithms(&self) -> Result<Vec<String>>;
    
    /// 按类型列出算法
    async fn list_algorithms_by_type(&self, algorithm_type: AlgorithmTypeInterface) -> Result<Vec<String>>;
    
    /// 验证算法
    async fn validate_algorithm(&self, algorithm: &AlgorithmDefinitionInterface) -> Result<ValidationResultInterface>;
}

/// 算法定义接口
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmDefinitionInterface {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub algorithm_type: AlgorithmTypeInterface,
    pub version: String,
    pub code: String,
    pub parameters: AlgorithmParametersInterface,
    pub input_schema: SchemaInterface,
    pub output_schema: SchemaInterface,
    pub resource_requirements: ResourceRequirementsInterface,
    pub security_level: SecurityLevelInterface,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub metadata: HashMap<String, String>,
}

/// 数据模式接口
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaInterface {
    pub fields: Vec<FieldDefinitionInterface>,
    pub required_fields: Vec<String>,
    pub optional_fields: Vec<String>,
    pub constraints: HashMap<String, ConstraintInterface>,
}

/// 字段定义接口
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinitionInterface {
    pub name: String,
    pub field_type: FieldTypeInterface,
    pub description: Option<String>,
    pub default_value: Option<serde_json::Value>,
    pub constraints: Vec<ConstraintInterface>,
}

/// 字段类型接口
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldTypeInterface {
    Integer,
    Float,
    String,
    Boolean,
    Array(Box<FieldTypeInterface>),
    Object(HashMap<String, FieldTypeInterface>),
}

/// 约束接口
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintInterface {
    MinValue(f64),
    MaxValue(f64),
    MinLength(usize),
    MaxLength(usize),
    Pattern(String),
    Enum(Vec<serde_json::Value>),
    Custom(String),
}

/// 资源需求接口
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirementsInterface {
    pub max_memory: Option<usize>,
    pub max_cpu_time: Option<f64>,
    pub max_gpu_time: Option<f64>,
    pub max_disk_space: Option<usize>,
    pub max_network_bandwidth: Option<usize>,
    pub required_capabilities: Vec<String>,
}

/// 安全级别接口
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum SecurityLevelInterface {
    Low,
    Medium,
    High,
    Critical,
}

/// 验证结果接口
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResultInterface {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub security_score: u8,
    pub performance_score: u8,
    pub recommendations: Vec<String>,
}

/// 算法执行接口
#[async_trait]
pub trait AlgorithmExecutorInterface: Send + Sync {
    /// 执行算法
    async fn execute(&self, algorithm: &AlgorithmDefinitionInterface, inputs: HashMap<String, serde_json::Value>) -> Result<AlgorithmResultInterface>;
    
    /// 验证输入
    async fn validate_inputs(&self, algorithm: &AlgorithmDefinitionInterface, inputs: &HashMap<String, serde_json::Value>) -> Result<ValidationResultInterface>;
    
    /// 获取执行状态
    async fn get_execution_status(&self, execution_id: &str) -> Result<ExecutionStatusInterface>;
    
    /// 取消执行
    async fn cancel_execution(&self, execution_id: &str) -> Result<()>;
    
    /// 执行算法（新方法）
    async fn execute_algorithm(&self, algo_id: &str, inputs: &[crate::core::types::CoreTensorData]) -> Result<Vec<crate::core::types::CoreTensorData>>;
    
    /// 验证算法
    async fn validate_algorithm(&self, algo_code: &str) -> Result<crate::core::interfaces::ValidationResult>;
    
    /// 验证算法代码
    async fn validate_algorithm_code(&self, algo_code: &str) -> Result<crate::core::interfaces::ValidationResult>;
    
    /// 注册算法
    async fn register_algorithm(&self, algo_def: &crate::core::unified_system::AlgorithmDefinition) -> Result<String>;
    
    /// 获取算法
    async fn get_algorithm(&self, algo_id: &str) -> Result<Option<crate::core::unified_system::AlgorithmDefinition>>;
    
    /// 列出算法
    async fn list_algorithms(&self) -> Result<Vec<crate::api::routes::algorithm::types::AlgorithmInfo>>;
    
    /// 更新算法
    async fn update_algorithm(&self, algo_id: &str, algo_def: &crate::core::unified_system::AlgorithmDefinition) -> Result<()>;
}

/// 执行状态接口
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionStatusInterface {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
    Timeout,
}

/// 算法任务接口
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmTaskInterface {
    pub id: String,
    pub algorithm_id: String,
    pub inputs: HashMap<String, serde_json::Value>,
    pub status: ExecutionStatusInterface,
    pub result: Option<AlgorithmResultInterface>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub metadata: HashMap<String, String>,
} 