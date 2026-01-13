use std::collections::HashMap;
use std::time::SystemTime;
use serde::{Serialize, Deserialize};

/// 算法模型定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmModel {
    pub id: String,
    pub name: String,
    pub description: String,
    pub algorithm_type: AlgorithmType,
    pub parameters: HashMap<String, ModelParameter>,
    pub architecture: ModelArchitecture,
    pub metadata: HashMap<String, String>,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
    pub version: String,
}

/// 算法类型枚举
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlgorithmType {
    Classification,
    Regression,
    Clustering,
    DeepLearning,
    NeuralNetwork,
    DecisionTree,
    SVM,
    LinearRegression,
    LogisticRegression,
    RandomForest,
    GradientBoosting,
    KMeans,
    DBSCAN,
    Custom(String),
}

/// 模型参数定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameter {
    pub name: String,
    pub value: ParameterValue,
    pub param_type: ParameterType,
    pub is_required: bool,
    pub default_value: Option<ParameterValue>,
    pub constraints: Option<ParameterConstraints>,
    pub description: String,
}

/// 参数值类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Array(Vec<ParameterValue>),
    Object(HashMap<String, ParameterValue>),
}

/// 参数类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    Integer,
    Float,
    String,
    Boolean,
    Array,
    Object,
}

/// 参数约束
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterConstraints {
    pub min_value: Option<ParameterValue>,
    pub max_value: Option<ParameterValue>,
    pub allowed_values: Option<Vec<ParameterValue>>,
    pub pattern: Option<String>,
}

/// 模型架构定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    pub layers: Vec<LayerDefinition>,
    pub connections: Vec<LayerConnection>,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub loss_function: String,
    pub optimizer: OptimizerConfig,
    pub metrics: Vec<String>,
}

/// 层定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDefinition {
    pub id: String,
    pub layer_type: LayerType,
    pub input_shape: Option<Vec<usize>>,
    pub output_shape: Option<Vec<usize>>,
    pub parameters: HashMap<String, ParameterValue>,
    pub activation: Option<ActivationFunction>,
}

/// 层类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    Dense,
    Convolutional2D,
    MaxPooling2D,
    AveragePooling2D,
    LSTM,
    GRU,
    Dropout,
    BatchNormalization,
    Embedding,
    Flatten,
    Reshape,
    Input,
    Output,
    Custom(String),
}

/// 激活函数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Linear,
    LeakyReLU,
    ELU,
    Swish,
    GELU,
    Custom(String),
}

/// 层连接定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConnection {
    pub from_layer: String,
    pub to_layer: String,
    pub connection_type: ConnectionType,
}

/// 连接类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectionType {
    Forward,
    Residual,
    Skip,
    Attention,
    Custom(String),
}

/// 优化器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub optimizer_type: OptimizerType,
    pub learning_rate: f64,
    pub parameters: HashMap<String, ParameterValue>,
}

/// 优化器类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerType {
    SGD,
    Adam,
    AdamW,
    RMSprop,
    Adagrad,
    Adadelta,
    Custom(String),
}

/// 模型配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfiguration {
    pub model_id: String,
    pub training_config: TrainingConfiguration,
    pub validation_config: ValidationConfiguration,
    pub deployment_config: DeploymentConfiguration,
    pub resource_requirements: ResourceRequirements,
}

/// 训练配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfiguration {
    pub epochs: usize,
    pub batch_size: usize,
    pub validation_split: f64,
    pub early_stopping: Option<EarlyStoppingConfig>,
    pub checkpointing: Option<CheckpointConfig>,
    pub data_augmentation: Option<DataAugmentationConfig>,
}

/// 早停配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    pub monitor: String,
    pub patience: usize,
    pub min_delta: f64,
    pub mode: MonitorMode,
}

/// 监控模式
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MonitorMode {
    Min,
    Max,
    Auto,
}

/// 检查点配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    pub save_best_only: bool,
    pub save_frequency: usize,
    pub monitor: String,
    pub mode: MonitorMode,
}

/// 数据增强配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAugmentationConfig {
    pub enabled: bool,
    pub transforms: Vec<TransformConfig>,
}

/// 变换配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformConfig {
    pub transform_type: String,
    pub parameters: HashMap<String, ParameterValue>,
    pub probability: f64,
}

/// 验证配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfiguration {
    pub validation_type: ValidationType,
    pub metrics: Vec<String>,
    pub cross_validation_folds: Option<usize>,
    pub test_split: f64,
}

/// 验证类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationType {
    HoldOut,
    CrossValidation,
    TimeSeriesSplit,
    Custom(String),
}

/// 部署配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentConfiguration {
    pub deployment_type: DeploymentType,
    pub scaling_config: ScalingConfig,
    pub performance_requirements: PerformanceRequirements,
}

/// 部署类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentType {
    Local,
    Cloud,
    Edge,
    Distributed,
    Custom(String),
}

/// 扩展配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConfig {
    pub auto_scaling: bool,
    pub min_instances: usize,
    pub max_instances: usize,
    pub target_utilization: f64,
}

/// 性能要求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    pub max_latency_ms: u64,
    pub min_throughput: u64,
    pub memory_limit_mb: u64,
    pub cpu_limit: f64,
}

/// 资源需求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_cores: f64,
    pub memory_gb: f64,
    pub gpu_count: usize,
    pub storage_gb: f64,
    pub network_bandwidth_mbps: f64,
}

/// 模型验证结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelValidation {
    pub is_valid: bool,
    pub validation_errors: Vec<ValidationError>,
    pub validation_warnings: Vec<ValidationWarning>,
    pub performance_metrics: HashMap<String, f64>,
    pub validation_timestamp: SystemTime,
}

/// 验证错误
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub error_type: String,
    pub message: String,
    pub location: Option<String>,
    pub severity: ErrorSeverity,
}

/// 验证警告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    pub warning_type: String,
    pub message: String,
    pub location: Option<String>,
    pub recommendation: Option<String>,
}

/// 错误严重程度
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl AlgorithmModel {
    /// 创建新的算法模型
    pub fn new(id: String, name: String, algorithm_type: AlgorithmType) -> Self {
        let now = SystemTime::now();
        Self {
            id,
            name,
            description: String::new(),
            algorithm_type,
            parameters: HashMap::new(),
            architecture: ModelArchitecture::default(),
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
            version: "1.0.0".to_string(),
        }
    }

    /// 添加参数
    pub fn add_parameter(&mut self, parameter: ModelParameter) {
        self.parameters.insert(parameter.name.clone(), parameter);
        self.updated_at = SystemTime::now();
    }

    /// 移除参数
    pub fn remove_parameter(&mut self, name: &str) -> Option<ModelParameter> {
        self.updated_at = SystemTime::now();
        self.parameters.remove(name)
    }

    /// 验证模型
    pub fn validate(&self) -> ModelValidation {
        let mut validation = ModelValidation {
            is_valid: true,
            validation_errors: Vec::new(),
            validation_warnings: Vec::new(),
            performance_metrics: HashMap::new(),
            validation_timestamp: SystemTime::now(),
        };

        // 验证必需参数
        for (name, param) in &self.parameters {
            if param.is_required && matches!(param.value, ParameterValue::String(ref s) if s.is_empty()) {
                validation.validation_errors.push(ValidationError {
                    error_type: "MissingRequiredParameter".to_string(),
                    message: format!("Required parameter '{}' is missing or empty", name),
                    location: Some(format!("parameters.{}", name)),
                    severity: ErrorSeverity::High,
                });
                validation.is_valid = false;
            }
        }

        // 验证架构
        if self.architecture.layers.is_empty() {
            validation.validation_errors.push(ValidationError {
                error_type: "EmptyArchitecture".to_string(),
                message: "Model architecture has no layers".to_string(),
                location: Some("architecture.layers".to_string()),
                severity: ErrorSeverity::Critical,
            });
            validation.is_valid = false;
        }

        validation
    }
}

impl Default for ModelArchitecture {
    fn default() -> Self {
        Self {
            layers: Vec::new(),
            connections: Vec::new(),
            input_shape: Vec::new(),
            output_shape: Vec::new(),
            loss_function: "mse".to_string(),
            optimizer: OptimizerConfig {
                optimizer_type: OptimizerType::Adam,
                learning_rate: 0.001,
                parameters: HashMap::new(),
            },
            metrics: vec!["accuracy".to_string()],
        }
    }
} 