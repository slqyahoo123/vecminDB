/// 核心类型定义
/// 定义系统中使用的核心数据类型和结构

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// 核心张量数据
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CoreTensorData {
    pub id: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub dtype: String,
    pub device: String,
    pub requires_grad: bool,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Default for CoreTensorData {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            shape: Vec::new(),
            data: Vec::new(),
            dtype: "float32".to_string(),
            device: "cpu".to_string(),
            requires_grad: false,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

impl CoreTensorData {
    /// 创建空的张量数据
    pub fn empty() -> Self {
        Self::default()
    }
    
    /// 获取张量数据的长度
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// 检查张量是否为空
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            shape,
            data,
            dtype: "float32".to_string(),
            device: "cpu".to_string(),
            requires_grad: false,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn reshape(&mut self, new_shape: Vec<usize>) -> Result<(), String> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.data.len() {
            return Err(format!("Cannot reshape tensor: size mismatch {} vs {}", new_size, self.data.len()));
        }
        self.shape = new_shape;
        Ok(())
    }
}

/// 核心数据批次
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreDataBatch {
    pub id: String,
    pub data: Vec<CoreTensorData>,
    pub labels: Option<Vec<CoreTensorData>>,
    pub batch_size: usize,
    pub metadata: Option<HashMap<String, String>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Default for CoreDataBatch {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            data: Vec::new(),
            labels: None,
            batch_size: 0,
            metadata: None,
            created_at: now,
            updated_at: now,
        }
    }
}

impl CoreDataBatch {
    pub fn new(data: Vec<CoreTensorData>, labels: Option<Vec<CoreTensorData>>) -> Self {
        let batch_size = data.len();
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            data,
            labels,
            batch_size,
            metadata: None,
            created_at: now,
            updated_at: now,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get_feature_count(&self) -> usize {
        self.data.len()
    }

    pub fn get_label_count(&self) -> usize {
        self.labels.as_ref().map_or(0, |labels| labels.len())
    }
}

/// 核心模型参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreModelParameters {
    pub id: String,
    pub parameters: HashMap<String, CoreTensorData>,
    pub optimizer_state: Option<HashMap<String, CoreTensorData>>,
    pub learning_rate: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Default for CoreModelParameters {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            parameters: HashMap::new(),
            optimizer_state: None,
            learning_rate: 0.001,
            momentum: 0.9,
            weight_decay: 0.0,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

impl CoreModelParameters {
    pub fn new(parameters: HashMap<String, CoreTensorData>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            parameters,
            optimizer_state: None,
            learning_rate: 0.001,
            momentum: 0.9,
            weight_decay: 0.0,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    pub fn get_parameter_count(&self) -> usize {
        self.parameters.len()
    }

    pub fn get_total_size(&self) -> usize {
        self.parameters.values().map(|tensor| tensor.data.len()).sum()
    }

    pub fn has_parameter(&self, name: &str) -> bool {
        self.parameters.contains_key(name)
    }

    pub fn get_parameter(&self, name: &str) -> Option<&CoreTensorData> {
        self.parameters.get(name)
    }

    pub fn set_parameter(&mut self, name: String, parameter: CoreTensorData) {
        self.parameters.insert(name, parameter);
        self.updated_at = Utc::now();
    }
}

/// 核心训练配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreTrainingConfig {
    pub id: String,
    pub model_id: String,
    pub dataset_id: String,
    pub epochs: u32,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub optimizer: String,
    pub loss_function: String,
    pub metrics: Vec<String>,
    pub validation_split: f64,
    pub early_stopping: bool,
    pub checkpoint_enabled: bool,
    pub mixed_precision: bool,
    pub device: String,
    pub parallel: bool,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Default for CoreTrainingConfig {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            model_id: String::new(),
            dataset_id: String::new(),
            epochs: 100,
            batch_size: 32,
            learning_rate: 0.001,
            optimizer: "adam".to_string(),
            loss_function: "mse".to_string(),
            metrics: vec!["accuracy".to_string()],
            validation_split: 0.2,
            early_stopping: true,
            checkpoint_enabled: true,
            mixed_precision: false,
            device: "cpu".to_string(),
            parallel: true,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

impl CoreTrainingConfig {
    pub fn new(model_id: String, dataset_id: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            model_id,
            dataset_id,
            epochs: 100,
            batch_size: 32,
            learning_rate: 0.001,
            optimizer: "adam".to_string(),
            loss_function: "mse".to_string(),
            metrics: vec!["accuracy".to_string()],
            validation_split: 0.2,
            early_stopping: true,
            checkpoint_enabled: true,
            mixed_precision: false,
            device: "cpu".to_string(),
            parallel: true,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    pub fn with_epochs(mut self, epochs: u32) -> Self {
        self.epochs = epochs;
        self.updated_at = Utc::now();
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self.updated_at = Utc::now();
        self
    }

    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self.updated_at = Utc::now();
        self
    }

    pub fn with_optimizer(mut self, optimizer: String) -> Self {
        self.optimizer = optimizer;
        self.updated_at = Utc::now();
        self
    }

    pub fn with_loss_function(mut self, loss_function: String) -> Self {
        self.loss_function = loss_function;
        self.updated_at = Utc::now();
        self
    }

    pub fn with_device(mut self, device: String) -> Self {
        self.device = device;
        self.updated_at = Utc::now();
        self
    }
}

/// 核心资源使用情况
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreResourceUsage {
    pub id: String,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub network_usage: f64,
    pub gpu_usage: Option<f64>,
    pub gpu_memory_usage: Option<f64>,
    pub active_processes: u32,
    pub open_files: u32,
    pub network_connections: u32,
    pub timestamp: DateTime<Utc>,
}

impl Default for CoreResourceUsage {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            cpu_usage: 0.0,
            memory_usage: 0.0,
            disk_usage: 0.0,
            network_usage: 0.0,
            gpu_usage: None,
            gpu_memory_usage: None,
            active_processes: 0,
            open_files: 0,
            network_connections: 0,
            timestamp: Utc::now(),
        }
    }
}

impl CoreResourceUsage {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_cpu_usage(mut self, cpu_usage: f64) -> Self {
        self.cpu_usage = cpu_usage;
        self
    }

    pub fn with_memory_usage(mut self, memory_usage: f64) -> Self {
        self.memory_usage = memory_usage;
        self
    }

    pub fn with_disk_usage(mut self, disk_usage: f64) -> Self {
        self.disk_usage = disk_usage;
        self
    }

    pub fn with_network_usage(mut self, network_usage: f64) -> Self {
        self.network_usage = network_usage;
        self
    }

    pub fn with_gpu_usage(mut self, gpu_usage: f64, gpu_memory_usage: f64) -> Self {
        self.gpu_usage = Some(gpu_usage);
        self.gpu_memory_usage = Some(gpu_memory_usage);
        self
    }

    pub fn is_healthy(&self) -> bool {
        self.cpu_usage < 90.0 && 
        self.memory_usage < 90.0 && 
        self.disk_usage < 90.0
    }

    pub fn get_total_usage(&self) -> f64 {
        (self.cpu_usage + self.memory_usage + self.disk_usage) / 3.0
    }
}

/// 核心执行结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreExecutionResult {
    pub id: String,
    pub success: bool,
    pub result: Option<CoreTensorData>,
    pub error: Option<String>,
    pub execution_time: f64,
    pub memory_used: f64,
    pub cpu_time: f64,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
}

impl Default for CoreExecutionResult {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            success: false,
            result: None,
            error: None,
            execution_time: 0.0,
            memory_used: 0.0,
            cpu_time: 0.0,
            metadata: HashMap::new(),
            created_at: Utc::now(),
        }
    }
}

impl CoreExecutionResult {
    pub fn success(result: CoreTensorData, execution_time: f64) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            success: true,
            result: Some(result),
            error: None,
            execution_time,
            memory_used: 0.0,
            cpu_time: 0.0,
            metadata: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    pub fn failure(error: String, execution_time: f64) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            success: false,
            result: None,
            error: Some(error),
            execution_time,
            memory_used: 0.0,
            cpu_time: 0.0,
            metadata: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    pub fn is_success(&self) -> bool {
        self.success
    }

    pub fn get_result(&self) -> Option<&CoreTensorData> {
        self.result.as_ref()
    }

    pub fn get_error(&self) -> Option<&String> {
        self.error.as_ref()
    }
}

/// 核心统一模型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreUnifiedModel {
    pub id: String,
    pub name: String,
    pub model_type: String,
    pub parameters: CoreModelParameters,
    pub architecture: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// 数据类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataType {
    Float32,
    Float64,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Bool,
    String,
    Binary,
}

impl Default for DataType {
    fn default() -> Self {
        DataType::Float32
    }
}

/// 设备类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DeviceType {
    CPU,
    GPU,
    CUDA,
    OpenCL,
    Metal,
    TPU,
}

impl Default for DeviceType {
    fn default() -> Self {
        DeviceType::CPU
    }
}

/// 模型状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelStatus {
    Created,
    Training,
    Trained,
    Deployed,
    Failed,
    Archived,
}

impl Default for ModelStatus {
    fn default() -> Self {
        ModelStatus::Created
    }
}

/// 模型信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub model_type: String,
    pub status: ModelStatus,
    pub version: String,
    pub description: Option<String>,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Default for ModelInfo {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name: "untitled_model".to_string(),
            model_type: "neural_network".to_string(),
            status: ModelStatus::Created,
            version: "1.0.0".to_string(),
            description: None,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

/// 核心训练结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreTrainingResult {
    pub id: String,
    pub model_id: String,
    pub status: TrainingStatus,
    pub metrics: HashMap<String, f64>,
    pub loss: f64,
    pub accuracy: f64,
    pub training_time: f64,
    pub epoch: u32,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Default for CoreTrainingResult {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            model_id: String::new(),
            status: TrainingStatus::Pending,
            metrics: HashMap::new(),
            loss: 0.0,
            accuracy: 0.0,
            training_time: 0.0,
            epoch: 0,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

/// 训练状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrainingStatus {
    Pending,
    Running,
    Paused,
    Completed,
    Failed,
    Stopped,
}

impl Default for TrainingStatus {
    fn default() -> Self {
        TrainingStatus::Pending
    }
}

/// 优化器类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OptimizerType {
    SGD,
    Adam,
    AdamW,
    RMSprop,
    Adagrad,
    Adadelta,
    Adamax,
    Nadam,
}

impl Default for OptimizerType {
    fn default() -> Self {
        OptimizerType::Adam
    }
}

/// 损失函数类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LossFunctionType {
    MSE,
    MAE,
    CrossEntropy,
    BinaryCrossEntropy,
    Hinge,
    Huber,
    KLDivergence,
    Poisson,
    MeanSquaredError,
    HuberLoss { delta: f64 },
    Custom { name: String },
}

impl Default for LossFunctionType {
    fn default() -> Self {
        LossFunctionType::MSE
    }
}

/// 核心早停配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreEarlyStoppingConfig {
    pub enabled: bool,
    pub patience: u32,
    pub min_delta: f64,
    pub monitor: String,
    pub mode: String,
    pub restore_best_weights: bool,
}

impl Default for CoreEarlyStoppingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            patience: 10,
            min_delta: 0.0,
            monitor: "val_loss".to_string(),
            mode: "min".to_string(),
            restore_best_weights: true,
        }
    }
}

/// 核心数据模式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreDataSchema {
    pub id: String,
    pub name: String,
    pub fields: Vec<CoreSchemaField>,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Default for CoreDataSchema {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name: "untitled_schema".to_string(),
            fields: Vec::new(),
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

/// 核心模式字段
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreSchemaField {
    pub name: String,
    pub field_type: CoreFieldType,
    pub nullable: bool,
    pub constraints: Option<FieldConstraints>,
    pub description: Option<String>,
}

impl Default for CoreSchemaField {
    fn default() -> Self {
        Self {
            name: String::new(),
            field_type: CoreFieldType::String,
            nullable: true,
            constraints: None,
            description: None,
        }
    }
}

/// 核心字段类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CoreFieldType {
    String,
    Integer,
    Float,
    Boolean,
    Date,
    DateTime,
    Array,
    Object,
}

impl Default for CoreFieldType {
    fn default() -> Self {
        CoreFieldType::String
    }
}

/// 字段约束
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldConstraints {
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub pattern: Option<String>,
    pub enum_values: Option<Vec<String>>,
}

impl Default for FieldConstraints {
    fn default() -> Self {
        Self {
            min_length: None,
            max_length: None,
            min_value: None,
            max_value: None,
            pattern: None,
            enum_values: None,
        }
    }
}

/// 核心算法定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreAlgorithmDefinition {
    pub id: String,
    pub name: String,
    pub algorithm_type: AlgorithmType,
    pub parameters: Vec<CoreAlgorithmParameter>,
    pub description: String,
    pub version: String,
    pub source_code: String,
    pub language: String,
    pub input_schema: Vec<crate::core::interfaces::TensorSchemaInterface>,
    pub output_schema: Vec<crate::core::interfaces::TensorSchemaInterface>,
    pub resource_requirements: crate::core::interfaces::ResourceRequirementsInterface,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Default for CoreAlgorithmDefinition {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name: "untitled_algorithm".to_string(),
            algorithm_type: AlgorithmType::MachineLearning,
            parameters: Vec::new(),
            description: String::new(),
            version: "1.0.0".to_string(),
            source_code: String::new(),
            language: "rust".to_string(),
            input_schema: Vec::new(),
            output_schema: Vec::new(),
            resource_requirements: crate::core::interfaces::ResourceRequirementsInterface {
                max_memory_mb: 512,
                max_cpu_percent: 100.0,
                max_execution_time_seconds: 3600,
                requires_gpu: false,
                max_gpu_memory_mb: None,
                network_access: false,
                file_system_access: Vec::new(),
            },
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

/// 算法类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlgorithmType {
    MachineLearning,
    DeepLearning,
    Statistical,
    Optimization,
    Clustering,
    Classification,
    Regression,
    Reinforcement,
    DataProcessing,
    FeatureExtraction,
    Custom,
}

impl Default for AlgorithmType {
    fn default() -> Self {
        AlgorithmType::MachineLearning
    }
}

impl std::fmt::Display for AlgorithmType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlgorithmType::MachineLearning => write!(f, "MachineLearning"),
            AlgorithmType::DeepLearning => write!(f, "DeepLearning"),
            AlgorithmType::Statistical => write!(f, "Statistical"),
            AlgorithmType::Optimization => write!(f, "Optimization"),
            AlgorithmType::Clustering => write!(f, "Clustering"),
            AlgorithmType::Classification => write!(f, "Classification"),
            AlgorithmType::Regression => write!(f, "Regression"),
            AlgorithmType::Reinforcement => write!(f, "Reinforcement"),
        }
    }
}

/// 核心算法参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreAlgorithmParameter {
    pub name: String,
    pub parameter_type: ParameterType,
    pub required: bool,
    pub default_value: Option<String>,
    pub description: String,
}

impl Default for CoreAlgorithmParameter {
    fn default() -> Self {
        Self {
            name: String::new(),
            parameter_type: ParameterType::String,
            required: false,
            default_value: None,
            description: String::new(),
        }
    }
}

/// 参数类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ParameterType {
    String,
    Integer,
    Float,
    Boolean,
    Array,
    Object,
}

impl Default for ParameterType {
    fn default() -> Self {
        ParameterType::String
    }
}

/// 资源需求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_cores: u32,
    pub memory_gb: f64,
    pub gpu_count: u32,
    pub gpu_memory_gb: f64,
    pub storage_gb: f64,
    pub network_bandwidth_mbps: f64,
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu_cores: 1,
            memory_gb: 1.0,
            gpu_count: 0,
            gpu_memory_gb: 0.0,
            storage_gb: 1.0,
            network_bandwidth_mbps: 100.0,
        }
    }
}

/// 核心日志条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreLogEntry {
    pub id: String,
    pub level: LogLevel,
    pub message: String,
    pub source: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

impl Default for CoreLogEntry {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            level: LogLevel::Info,
            message: String::new(),
            source: String::new(),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }
}

/// 日志级别
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Fatal,
}

impl Default for LogLevel {
    fn default() -> Self {
        LogLevel::Info
    }
}

/// 核心组件状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreComponentStatus {
    pub component_id: String,
    pub component_type: ComponentType,
    pub status: ComponentStatusValue,
    pub health_score: f64,
    pub last_heartbeat: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

impl Default for CoreComponentStatus {
    fn default() -> Self {
        Self {
            component_id: String::new(),
            component_type: ComponentType::Service,
            status: ComponentStatusValue::Unknown,
            health_score: 0.0,
            last_heartbeat: Utc::now(),
            metadata: HashMap::new(),
        }
    }
}

/// 组件类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComponentType {
    Service,
    Database,
    Cache,
    Queue,
    Storage,
    Network,
    Compute,
    Monitor,
}

impl Default for ComponentType {
    fn default() -> Self {
        ComponentType::Service
    }
}

/// 组件状态值
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComponentStatusValue {
    Healthy,
    Warning,
    Critical,
    Unknown,
    Offline,
}

impl Default for ComponentStatusValue {
    fn default() -> Self {
        ComponentStatusValue::Unknown
    }
}

/// 核心问题
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreIssue {
    pub id: String,
    pub title: String,
    pub description: String,
    pub severity: IssueSeverity,
    pub component_id: String,
    pub status: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

impl Default for CoreIssue {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            title: String::new(),
            description: String::new(),
            severity: IssueSeverity::Low,
            component_id: String::new(),
            status: "open".to_string(),
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
        }
    }
}

/// 问题严重性
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl Default for IssueSeverity {
    fn default() -> Self {
        IssueSeverity::Low
    }
}

/// 处理后的数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedData {
    pub id: String,
    pub data: Vec<u8>,
    pub format: String,
    pub size: usize,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Default for ProcessedData {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            data: Vec::new(),
            format: "raw".to_string(),
            size: 0,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

/// 接口预处理配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfacePreprocessingConfig {
    pub id: String,
    pub name: String,
    pub config_type: String,
    pub parameters: HashMap<String, String>,
    pub enabled: bool,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Default for InterfacePreprocessingConfig {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name: "default_preprocessing".to_string(),
            config_type: "standard".to_string(),
            parameters: HashMap::new(),
            enabled: true,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

/// 组件健康状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub component_id: String,
    pub status: String,
    pub health_score: f64,
    pub last_check: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

impl Default for ComponentHealth {
    fn default() -> Self {
        Self {
            component_id: String::new(),
            status: "unknown".to_string(),
            health_score: 0.0,
            last_check: Utc::now(),
            metadata: HashMap::new(),
        }
    }
}

/// 健康状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Unknown,
    Offline,
}

impl Default for HealthStatus {
    fn default() -> Self {
        HealthStatus::Unknown
    }
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            HealthStatus::Healthy => "Healthy",
            HealthStatus::Warning => "Warning",
            HealthStatus::Critical => "Critical",
            HealthStatus::Unknown => "Unknown",
            HealthStatus::Offline => "Offline",
        };
        write!(f, "{}", s)
    }
}

/// 训练配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub id: String,
    pub name: String,
    pub model_id: String,
    pub algorithm_id: String,
    pub parameters: HashMap<String, String>,
    pub batch_size: u32,
    pub learning_rate: f64,
    pub epochs: u32,
    pub validation_split: f64,
    pub early_stopping: bool,
    pub checkpoint_enabled: bool,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name: "default_training".to_string(),
            model_id: String::new(),
            algorithm_id: String::new(),
            parameters: HashMap::new(),
            batch_size: 32,
            learning_rate: 0.001,
            epochs: 100,
            validation_split: 0.2,
            early_stopping: true,
            checkpoint_enabled: true,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

/// 模型状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelState {
    Created,
    Training,
    Trained,
    Deployed,
    Failed,
    Archived,
}

impl Default for ModelState {
    fn default() -> Self {
        ModelState::Created
    }
}


/// 算法参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmParameter {
    pub name: String,
    pub parameter_type: String,
    pub required: bool,
    pub default_value: Option<String>,
    pub description: String,
}

impl Default for AlgorithmParameter {
    fn default() -> Self {
        Self {
            name: String::new(),
            parameter_type: "string".to_string(),
            required: false,
            default_value: None,
            description: String::new(),
        }
    }
}

/// 核心训练配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreTrainingConfiguration {
    pub id: String,
    pub name: String,
    pub model_id: String,
    pub algorithm_id: String,
    pub parameters: HashMap<String, String>,
    pub batch_size: u32,
    pub learning_rate: f64,
    pub epochs: u32,
    pub validation_split: f64,
    pub early_stopping: bool,
    pub checkpoint_enabled: bool,
    pub device_type: DeviceType,
    pub optimization_config: OptimizationConfig,
    pub loss_function: LossFunctionType,
    pub regularization: RegularizationConfig,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Default for CoreTrainingConfiguration {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name: "default_training_config".to_string(),
            model_id: String::new(),
            algorithm_id: String::new(),
            parameters: HashMap::new(),
            batch_size: 32,
            learning_rate: 0.001,
            epochs: 100,
            validation_split: 0.2,
            early_stopping: true,
            checkpoint_enabled: true,
            device_type: DeviceType::CPU,
            optimization_config: OptimizationConfig::default(),
            loss_function: LossFunctionType::default(),
            regularization: RegularizationConfig::default(),
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

/// 优化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub optimizer_type: String,
    pub learning_rate_schedule: String,
    pub weight_decay: f64,
    pub momentum: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub amsgrad: bool,
    pub parameters: HashMap<String, String>,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            optimizer_type: "adam".to_string(),
            learning_rate_schedule: "constant".to_string(),
            weight_decay: 0.0,
            momentum: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            amsgrad: false,
            parameters: HashMap::new(),
        }
    }
}

// 删除重复定义，使用已有的ModelStatus定义

/// 层定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDefinition {
    pub id: String,
    pub name: String,
    pub layer_type: String,
    pub parameters: HashMap<String, String>,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Default for LayerDefinition {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name: "untitled_layer".to_string(),
            layer_type: "dense".to_string(),
            parameters: HashMap::new(),
            input_shape: Vec::new(),
            output_shape: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

/// 连接定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionDefinition {
    pub id: String,
    pub from_layer: String,
    pub to_layer: String,
    pub connection_type: String,
    pub parameters: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Default for ConnectionDefinition {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            from_layer: String::new(),
            to_layer: String::new(),
            connection_type: "direct".to_string(),
            parameters: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

// 删除重复定义，使用已有的OptimizerType定义

// 删除重复定义，使用已有的LossFunctionType定义

// 删除重复定义，使用已有的DeviceType定义

/// 正则化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    pub l1_regularization: f64,
    pub l2_regularization: f64,
    pub dropout_rate: f64,
    pub batch_normalization: bool,
    pub weight_decay: f64,
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            l1_regularization: 0.0,
            l2_regularization: 0.0,
            dropout_rate: 0.0,
            batch_normalization: false,
            weight_decay: 0.0,
        }
    }
}

// 删除重复定义，使用已有的DataType定义

impl Default for CoreUnifiedModel {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name: "untitled_model".to_string(),
            model_type: "neural_network".to_string(),
            parameters: CoreModelParameters::default(),
            architecture: "sequential".to_string(),
            input_shape: Vec::new(),
            output_shape: Vec::new(),
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

impl CoreUnifiedModel {
    pub fn new(name: String, model_type: String, architecture: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            model_type,
            parameters: CoreModelParameters::default(),
            architecture,
            input_shape: Vec::new(),
            output_shape: Vec::new(),
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }

    pub fn with_input_shape(mut self, input_shape: Vec<usize>) -> Self {
        self.input_shape = input_shape;
        self.updated_at = Utc::now();
        self
    }

    pub fn with_output_shape(mut self, output_shape: Vec<usize>) -> Self {
        self.output_shape = output_shape;
        self.updated_at = Utc::now();
        self
    }

    pub fn with_parameters(mut self, parameters: CoreModelParameters) -> Self {
        self.parameters = parameters;
        self.updated_at = Utc::now();
        self
    }

    pub fn get_parameter_count(&self) -> usize {
        self.parameters.get_parameter_count()
    }

    pub fn get_total_parameters(&self) -> usize {
        self.parameters.get_total_size()
    }

    pub fn is_empty(&self) -> bool {
        self.parameters.parameters.is_empty()
    }
}

// 这些类型已在其他模块中定义，避免重复定义
// CoreProcessingConfig, CorePreprocessingConfig, CorePostprocessingConfig 已在 interfaces.rs 中定义
// CoreProcessedData 已在 interfaces.rs 中定义
// CoreDataInfo 已在 data_to_model_engine.rs 中定义
