use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;

// Import the ResourceLimits type which is used in SecurityRule
use crate::algorithm::execution_types::ResourceLimits;

/// Task identifier
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub String);

impl TaskId {
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    pub fn from_string(s: String) -> Self {
        Self(s)
    }
}

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for TaskId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for TaskId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Task status enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Initialized,
    Running,
    Paused,
    Completed,
    Failed(String),
    Cancelled,
    Error,
}

impl Default for TaskStatus {
    fn default() -> Self {
        TaskStatus::Pending
    }
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskStatus::Pending => write!(f, "Pending"),
            TaskStatus::Initialized => write!(f, "Initialized"),
            TaskStatus::Running => write!(f, "Running"),
            TaskStatus::Paused => write!(f, "Paused"),
            TaskStatus::Completed => write!(f, "Completed"),
            TaskStatus::Failed(err) => write!(f, "Failed: {}", err),
            TaskStatus::Cancelled => write!(f, "Cancelled"),
            TaskStatus::Error => write!(f, "Error"),
        }
    }
}

impl TaskStatus {
    /// Parse TaskStatus from string
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "pending" => Ok(TaskStatus::Pending),
            "initialized" => Ok(TaskStatus::Initialized),
            "running" => Ok(TaskStatus::Running),
            "paused" => Ok(TaskStatus::Paused),
            "completed" => Ok(TaskStatus::Completed),
            "cancelled" => Ok(TaskStatus::Cancelled),
            "error" => Ok(TaskStatus::Error),
            s if s.starts_with("failed:") => {
                let error_msg = s.strip_prefix("failed:").unwrap_or("").trim();
                Ok(TaskStatus::Failed(error_msg.to_string()))
            },
            _ => Err(format!("Unknown task status: {}", s)),
        }
    }

    /// Convert TaskStatus to string representation
    pub fn to_string_simple(&self) -> String {
        match self {
            TaskStatus::Pending => "pending".to_string(),
            TaskStatus::Initialized => "initialized".to_string(),
            TaskStatus::Running => "running".to_string(),
            TaskStatus::Paused => "paused".to_string(),
            TaskStatus::Completed => "completed".to_string(),
            TaskStatus::Failed(_) => "failed".to_string(),
            TaskStatus::Cancelled => "cancelled".to_string(),
            TaskStatus::Error => "error".to_string(),
        }
    }

    /// Check if status represents a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(self, TaskStatus::Completed | TaskStatus::Failed(_) | TaskStatus::Cancelled | TaskStatus::Error)
    }

    /// Check if status represents an active state
    pub fn is_active(&self) -> bool {
        matches!(self, TaskStatus::Running | TaskStatus::Paused)
    }

    /// Check if status represents a failure state
    pub fn is_failed(&self) -> bool {
        matches!(self, TaskStatus::Failed(_) | TaskStatus::Error)
    }
}

impl std::str::FromStr for TaskStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        TaskStatus::from_str(s)
    }
}

/// Algorithm task structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmTask {
    pub id: TaskId,
    pub name: String,
    pub algorithm_id: String,
    pub status: TaskStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub priority: crate::algorithm::sandbox_types::TaskPriority,
    pub parameters: HashMap<String, serde_json::Value>,
    pub metadata: HashMap<String, String>,
    pub progress: f64,
    pub error_message: Option<String>,
    /// 错误信息（兼容字段）
    pub error: Option<String>,
    pub result: Option<serde_json::Value>,
    /// 完成时间
    pub completed_at: Option<DateTime<Utc>>,
}

impl AlgorithmTask {
    pub fn new(name: String, algorithm_id: String) -> Self {
        let now = Utc::now();
        Self {
            id: TaskId::new(),
            name,
            algorithm_id,
            status: TaskStatus::Pending,
            created_at: now,
            updated_at: now,
            priority: crate::algorithm::sandbox_types::TaskPriority::Normal,
            parameters: HashMap::new(),
            metadata: HashMap::new(),
            progress: 0.0,
            error_message: None,
            error: None,
            result: None,
            completed_at: None,
        }
    }

    pub fn with_id(id: TaskId, name: String, algorithm_id: String) -> Self {
        let now = Utc::now();
        Self {
            id,
            name,
            algorithm_id,
            status: TaskStatus::Pending,
            created_at: now,
            updated_at: now,
            priority: crate::algorithm::sandbox_types::TaskPriority::Normal,
            parameters: HashMap::new(),
            metadata: HashMap::new(),
            progress: 0.0,
            error_message: None,
            error: None,
            result: None,
            completed_at: None,
        }
    }

    pub fn update_status(&mut self, status: TaskStatus) {
        self.status = status;
        self.updated_at = Utc::now();
    }

    pub fn set_progress(&mut self, progress: f64) {
        self.progress = progress.clamp(0.0, 100.0);
        self.updated_at = Utc::now();
    }

    pub fn set_error(&mut self, error: String) {
        self.status = TaskStatus::Failed(error.clone());
        self.error_message = Some(error);
        self.error = Some(error);
        self.updated_at = Utc::now();
    }

    pub fn set_result(&mut self, result: serde_json::Value) {
        self.result = Some(result);
        self.status = TaskStatus::Completed;
        self.completed_at = Some(Utc::now());
        self.updated_at = Utc::now();
    }
}

/// Algorithm metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmMetadata {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub author: Option<String>,
    pub tags: Vec<String>,
    pub algorithm_type: crate::algorithm::algorithm_types::AlgorithmType,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub dependencies: Vec<String>,
    pub parameters: HashMap<String, ParameterDefinition>,
    pub performance_metrics: Option<PerformanceMetrics>,
    pub validation_status: ValidationStatus,
}

/// Parameter definition for algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDefinition {
    pub name: String,
    pub parameter_type: ParameterType,
    pub required: bool,
    pub default_value: Option<serde_json::Value>,
    pub description: Option<String>,
    pub constraints: Option<ParameterConstraints>,
}

/// Parameter type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParameterType {
    String,
    Integer,
    Float,
    Boolean,
    Array,
    Object,
}

/// Parameter constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterConstraints {
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub allowed_values: Option<Vec<serde_json::Value>>,
    pub pattern: Option<String>,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub execution_time_ms: u64,
    pub memory_usage_mb: u64,
    pub accuracy: Option<f64>,
    pub throughput: Option<f64>,
    pub error_rate: Option<f64>,
}

/// Validation status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationStatus {
    Pending,
    Valid,
    Invalid(String),
    Warning(String),
}

/// Algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmConfig {
    pub id: String,
    pub parameters: HashMap<String, serde_json::Value>,
    pub optimization_config: Option<AlgorithmOptimizationConfig>,
    pub execution_config: crate::algorithm::sandbox_types::ExecutionConfig,
    pub validation_rules: Vec<ValidationRule>,
    pub resource_limits: crate::algorithm::execution_types::ResourceLimits,
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            parameters: HashMap::new(),
            optimization_config: None,
            execution_config: crate::algorithm::sandbox_types::ExecutionConfig::default(),
            validation_rules: Vec::new(),
            resource_limits: crate::algorithm::execution_types::ResourceLimits::default(),
        }
    }
}

/// Algorithm optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmOptimizationConfig {
    pub optimization_type: crate::algorithm::algorithm_types::AlgorithmOptimizationType,
    pub target_metric: String,
    pub max_iterations: u32,
    pub convergence_threshold: f64,
    pub early_stopping: bool,
    pub hyperparameters: HashMap<String, serde_json::Value>,
}

impl Default for AlgorithmOptimizationConfig {
    fn default() -> Self {
        Self {
            optimization_type: crate::algorithm::algorithm_types::AlgorithmOptimizationType::Gradient,
            target_metric: "accuracy".to_string(),
            max_iterations: 1000,
            convergence_threshold: 1e-6,
            early_stopping: true,
            hyperparameters: HashMap::new(),
        }
    }
}

/// Validation rule for algorithm parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub name: String,
    pub rule_type: ValidationRuleType,
    pub condition: String,
    pub error_message: String,
}

/// Validation rule type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationRuleType {
    Range,
    Pattern,
    Custom,
    Dependency,
}

/// Algorithm metrics for monitoring and evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmMetrics {
    pub algorithm_id: String,
    pub task_id: TaskId,
    pub timestamp: DateTime<Utc>,
    pub execution_time_ms: u64,
    pub memory_usage_bytes: u64,
    pub cpu_usage_percent: f64,
    pub accuracy: Option<f64>,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1_score: Option<f64>,
    pub loss: Option<f64>,
    pub custom_metrics: HashMap<String, f64>,
}

impl AlgorithmMetrics {
    pub fn new(algorithm_id: String, task_id: TaskId) -> Self {
        Self {
            algorithm_id,
            task_id,
            timestamp: Utc::now(),
            execution_time_ms: 0,
            memory_usage_bytes: 0,
            cpu_usage_percent: 0.0,
            accuracy: None,
            precision: None,
            recall: None,
            f1_score: None,
            loss: None,
            custom_metrics: HashMap::new(),
        }
    }

    pub fn set_performance(&mut self, execution_time_ms: u64, memory_usage_bytes: u64, cpu_usage_percent: f64) {
        self.execution_time_ms = execution_time_ms;
        self.memory_usage_bytes = memory_usage_bytes;
        self.cpu_usage_percent = cpu_usage_percent;
    }

    pub fn set_accuracy_metrics(&mut self, accuracy: f64, precision: f64, recall: f64, f1_score: f64) {
        self.accuracy = Some(accuracy);
        self.precision = Some(precision);
        self.recall = Some(recall);
        self.f1_score = Some(f1_score);
    }

    pub fn add_custom_metric(&mut self, name: String, value: f64) {
        self.custom_metrics.insert(name, value);
    }
}

/// Execution context for algorithm execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionContext {
    pub task_id: TaskId,
    pub algorithm_id: String,
    pub session_id: String,
    pub user_id: Option<String>,
    pub environment: ExecutionEnvironment,
    pub configuration: AlgorithmConfig,
    pub resources: ResourceAllocation,
    pub security_context: SecurityContext,
    pub started_at: DateTime<Utc>,
    pub timeout: Option<std::time::Duration>,
}

/// Execution environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionEnvironment {
    pub platform: String,
    pub version: String,
    pub architecture: String,
    pub available_memory_bytes: u64,
    pub available_cpu_cores: u32,
    pub available_gpu_count: u32,
    pub environment_variables: HashMap<String, String>,
}

/// Resource allocation for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub memory_limit_bytes: u64,
    pub cpu_cores: u32,
    pub gpu_count: u32,
    pub disk_space_bytes: u64,
    pub network_bandwidth_mbps: u32,
    pub priority: crate::algorithm::sandbox_types::TaskPriority,
}

/// Security context for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContext {
    pub security_level: crate::algorithm::sandbox_types::SandboxSecurityLevel,
    pub allowed_operations: Vec<String>,
    pub restricted_operations: Vec<String>,
    pub network_policy: crate::algorithm::sandbox_types::NetworkPolicy,
    pub filesystem_policy: crate::algorithm::sandbox_types::FilesystemPolicy,
    pub encryption_required: bool,
}

impl ExecutionContext {
    pub fn new(task_id: TaskId, algorithm_id: String) -> Self {
        Self {
            task_id,
            algorithm_id,
            session_id: Uuid::new_v4().to_string(),
            user_id: None,
            environment: ExecutionEnvironment::default(),
            configuration: AlgorithmConfig::default(),
            resources: ResourceAllocation::default(),
            security_context: SecurityContext::default(),
            started_at: Utc::now(),
            timeout: None,
        }
    }

    pub fn with_config(task_id: TaskId, algorithm_id: String, config: AlgorithmConfig) -> Self {
        let mut ctx = Self::new(task_id, algorithm_id);
        ctx.configuration = config;
        ctx
    }

    pub fn is_expired(&self) -> bool {
        if let Some(timeout) = self.timeout {
            Utc::now().signed_duration_since(self.started_at) > chrono::Duration::from_std(timeout).unwrap()
        } else {
            false
        }
    }
}

impl Default for ExecutionEnvironment {
    fn default() -> Self {
        Self {
            platform: std::env::consts::OS.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            architecture: std::env::consts::ARCH.to_string(),
            available_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB default
            available_cpu_cores: num_cpus::get() as u32,
            available_gpu_count: 0,
            environment_variables: HashMap::new(),
        }
    }
}

impl Default for ResourceAllocation {
    fn default() -> Self {
        Self {
            memory_limit_bytes: 1024 * 1024 * 1024, // 1GB default
            cpu_cores: 1,
            gpu_count: 0,
            disk_space_bytes: 10 * 1024 * 1024 * 1024, // 10GB default
            network_bandwidth_mbps: 100,
            priority: crate::algorithm::sandbox_types::TaskPriority::Normal,
        }
    }
}

impl Default for SecurityContext {
    fn default() -> Self {
        Self {
            security_level: crate::algorithm::sandbox_types::SandboxSecurityLevel::Medium,
            allowed_operations: Vec::new(),
            restricted_operations: Vec::new(),
            network_policy: crate::algorithm::sandbox_types::NetworkPolicy::Denied,
            filesystem_policy: crate::algorithm::sandbox_types::FilesystemPolicy::ReadOnly,
            encryption_required: false,
        }
    }
}

/// Task error for algorithm operations
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskError {
    ValidationError(String),
    ExecutionError(String),
    TimeoutError,
    ResourceExhausted(String),
    SecurityViolation(String),
    InternalError(String),
    DependencyError(String),
    ConfigurationError(String),
}

impl std::fmt::Display for TaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TaskError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            TaskError::ExecutionError(msg) => write!(f, "Execution error: {}", msg),
            TaskError::TimeoutError => write!(f, "Task timeout"),
            TaskError::ResourceExhausted(msg) => write!(f, "Resource exhausted: {}", msg),
            TaskError::SecurityViolation(msg) => write!(f, "Security violation: {}", msg),
            TaskError::InternalError(msg) => write!(f, "Internal error: {}", msg),
            TaskError::DependencyError(msg) => write!(f, "Dependency error: {}", msg),
            TaskError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for TaskError {}

/// Algorithm result containing execution output and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmResult {
    pub task_id: TaskId,
    pub algorithm_id: String,
    pub status: TaskStatus,
    pub output: Option<serde_json::Value>,
    pub metrics: AlgorithmMetrics,
    pub error: Option<TaskError>,
    pub execution_time_ms: u64,
    pub memory_used_bytes: u64,
    pub cpu_usage_percent: f64,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub logs: Vec<LogEntry>,
    pub artifacts: Vec<ArtifactReference>,
}

impl AlgorithmResult {
    pub fn new(task_id: TaskId, algorithm_id: String) -> Self {
        Self {
            task_id: task_id.clone(),
            algorithm_id,
            status: TaskStatus::Pending,
            output: None,
            metrics: AlgorithmMetrics::new(algorithm_id.clone(), task_id),
            error: None,
            execution_time_ms: 0,
            memory_used_bytes: 0,
            cpu_usage_percent: 0.0,
            started_at: Utc::now(),
            completed_at: None,
            logs: Vec::new(),
            artifacts: Vec::new(),
        }
    }

    pub fn with_success(mut self, output: serde_json::Value) -> Self {
        self.status = TaskStatus::Completed;
        self.output = Some(output);
        self.completed_at = Some(Utc::now());
        self
    }

    pub fn with_error(mut self, error: TaskError) -> Self {
        self.status = TaskStatus::Failed(error.to_string());
        self.error = Some(error);
        self.completed_at = Some(Utc::now());
        self
    }

    pub fn add_log(&mut self, level: LogLevel, message: String) {
        self.logs.push(LogEntry {
            timestamp: Utc::now(),
            level,
            message,
            source: "algorithm_executor".to_string(),
        });
    }

    pub fn add_artifact(&mut self, artifact: ArtifactReference) {
        self.artifacts.push(artifact);
    }

    pub fn is_success(&self) -> bool {
        matches!(self.status, TaskStatus::Completed)
    }

    pub fn is_failed(&self) -> bool {
        matches!(self.status, TaskStatus::Failed(_))
    }
}

/// Log entry for algorithm execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: DateTime<Utc>,
    pub level: LogLevel,
    pub message: String,
    pub source: String,
}

/// Log level enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

/// Artifact reference for algorithm outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactReference {
    pub id: String,
    pub name: String,
    pub artifact_type: ArtifactType,
    pub path: String,
    pub size_bytes: u64,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// Artifact type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtifactType {
    Model,
    Dataset,
    Checkpoint,
    Log,
    Config,
    Report,
    Custom(String),
}

/// Algorithm apply configuration for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmApplyConfig {
    pub algorithm_id: String,
    pub version: Option<String>,
    pub parameters: HashMap<String, serde_json::Value>,
    pub execution_config: crate::algorithm::sandbox_types::ExecutionConfig,
    pub resource_limits: crate::algorithm::execution_types::ResourceLimits,
    pub timeout_seconds: Option<u64>,
    pub retry_config: Option<RetryConfig>,
    pub checkpoint_config: Option<CheckpointConfig>,
    pub logging_config: LoggingConfig,
    pub artifact_config: ArtifactConfig,
}

impl Default for AlgorithmApplyConfig {
    fn default() -> Self {
        Self {
            algorithm_id: String::new(),
            version: None,
            parameters: HashMap::new(),
            execution_config: crate::algorithm::sandbox_types::ExecutionConfig::default(),
            resource_limits: crate::algorithm::execution_types::ResourceLimits::default(),
            timeout_seconds: Some(3600), // 1 hour default
            retry_config: Some(RetryConfig::default()),
            checkpoint_config: None,
            logging_config: LoggingConfig::default(),
            artifact_config: ArtifactConfig::default(),
        }
    }
}

/// Retry configuration for algorithm execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f64,
    pub retry_on_errors: Vec<TaskError>,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 1000,
            max_delay_ms: 30000,
            backoff_multiplier: 2.0,
            retry_on_errors: vec![
                TaskError::ResourceExhausted("temporary".to_string()),
                TaskError::InternalError("temporary".to_string()),
            ],
        }
    }
}

/// Checkpoint configuration for algorithm execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    pub enabled: bool,
    pub interval_seconds: u64,
    pub max_checkpoints: u32,
    pub checkpoint_path: String,
    pub include_optimizer_state: bool,
    pub compression_enabled: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            interval_seconds: 300, // 5 minutes
            max_checkpoints: 5,
            checkpoint_path: "./checkpoints".to_string(),
            include_optimizer_state: true,
            compression_enabled: true,
        }
    }
}

/// Logging configuration for algorithm execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: LogLevel,
    pub include_metrics: bool,
    pub include_parameters: bool,
    pub include_environment: bool,
    pub max_log_entries: Option<u32>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            include_metrics: true,
            include_parameters: false,
            include_environment: false,
            max_log_entries: Some(1000),
        }
    }
}

/// Artifact configuration for algorithm execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactConfig {
    pub enabled: bool,
    pub base_path: String,
    pub auto_save_models: bool,
    pub auto_save_checkpoints: bool,
    pub auto_save_logs: bool,
    pub compression_enabled: bool,
    pub retention_policy: RetentionPolicy,
}

impl Default for ArtifactConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_path: "./artifacts".to_string(),
            auto_save_models: true,
            auto_save_checkpoints: false,
            auto_save_logs: true,
            compression_enabled: true,
            retention_policy: RetentionPolicy::default(),
        }
    }
}

/// Retention policy for artifacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    pub max_artifacts: Option<u32>,
    pub max_age_days: Option<u32>,
    pub cleanup_strategy: CleanupStrategy,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_artifacts: Some(100),
            max_age_days: Some(30),
            cleanup_strategy: CleanupStrategy::OldestFirst,
        }
    }
}

/// Cleanup strategy for retention policy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CleanupStrategy {
    OldestFirst,
    LargestFirst,
    LeastUsed,
    Manual,
}

/// Optimizer configuration and state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Optimizer {
    pub optimizer_type: OptimizerType,
    pub learning_rate: f64,
    pub parameters: HashMap<String, serde_json::Value>,
    pub state: OptimizerState,
    pub statistics: OptimizerStatistics,
}

impl Default for Optimizer {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::Adam,
            learning_rate: 0.001,
            parameters: HashMap::new(),
            state: OptimizerState::default(),
            statistics: OptimizerStatistics::default(),
        }
    }
}

/// Optimizer type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerType {
    SGD,
    Adam,
    RMSprop,
    AdaGrad,
    AdaDelta,
    Custom(String),
}

/// Optimizer state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    pub step: u64,
    pub momentum: HashMap<String, serde_json::Value>,
    pub velocity: HashMap<String, serde_json::Value>,
    pub accumulated_gradients: HashMap<String, serde_json::Value>,
}

impl Default for OptimizerState {
    fn default() -> Self {
        Self {
            step: 0,
            momentum: HashMap::new(),
            velocity: HashMap::new(),
            accumulated_gradients: HashMap::new(),
        }
    }
}

/// Optimizer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerStatistics {
    pub total_steps: u64,
    pub average_gradient_norm: f64,
    pub learning_rate_history: Vec<f64>,
    pub convergence_metrics: ConvergenceMetrics,
}

impl Default for OptimizerStatistics {
    fn default() -> Self {
        Self {
            total_steps: 0,
            average_gradient_norm: 0.0,
            learning_rate_history: Vec::new(),
            convergence_metrics: ConvergenceMetrics::default(),
        }
    }
}

/// Convergence metrics for optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceMetrics {
    pub loss_history: Vec<f64>,
    pub gradient_norm_history: Vec<f64>,
    pub is_converged: bool,
    pub convergence_threshold: f64,
    pub patience: u32,
    pub best_loss: Option<f64>,
}

impl Default for ConvergenceMetrics {
    fn default() -> Self {
        Self {
            loss_history: Vec::new(),
            gradient_norm_history: Vec::new(),
            is_converged: false,
            convergence_threshold: 1e-6,
            patience: 10,
            best_loss: None,
        }
    }
}

impl Optimizer {
    pub fn new(optimizer_type: OptimizerType, learning_rate: f64) -> Self {
        Self {
            optimizer_type,
            learning_rate,
            parameters: HashMap::new(),
            state: OptimizerState::default(),
            statistics: OptimizerStatistics::default(),
        }
    }

    pub fn step(&mut self) {
        self.state.step += 1;
        self.statistics.total_steps += 1;
        self.statistics.learning_rate_history.push(self.learning_rate);
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    pub fn get_state(&self) -> &OptimizerState {
        &self.state
    }

    pub fn get_statistics(&self) -> &OptimizerStatistics {
        &self.statistics
    }

    pub fn reset(&mut self) {
        self.state = OptimizerState::default();
        self.statistics = OptimizerStatistics::default();
    }

    pub fn update_convergence(&mut self, loss: f64, gradient_norm: f64) {
        self.statistics.convergence_metrics.loss_history.push(loss);
        self.statistics.convergence_metrics.gradient_norm_history.push(gradient_norm);
        self.statistics.average_gradient_norm = gradient_norm;

        // Update best loss
        if self.statistics.convergence_metrics.best_loss.is_none() || 
           loss < self.statistics.convergence_metrics.best_loss.unwrap() {
            self.statistics.convergence_metrics.best_loss = Some(loss);
        }

        // Check convergence
        self.check_convergence();
    }

    fn check_convergence(&mut self) {
        let metrics = &mut self.statistics.convergence_metrics;
        if metrics.loss_history.len() < metrics.patience as usize + 1 {
            return;
        }

        let recent_losses = &metrics.loss_history[metrics.loss_history.len() - metrics.patience as usize..];
        let loss_change = recent_losses.iter().fold(0.0, |acc, &loss| {
            acc + (loss - recent_losses[0]).abs()
        }) / metrics.patience as f64;

        metrics.is_converged = loss_change < metrics.convergence_threshold;
    }
}

/// Production model options for advanced configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionModelOptions {
    pub enable_model_versioning: bool,
    pub enable_model_compression: bool,
    pub enable_model_encryption: bool,
    pub enable_model_validation: bool,
    pub enable_model_optimization: bool,
    pub enable_distributed_storage: bool,
    pub model_cache_size: usize,
    pub model_timeout: std::time::Duration,
    pub enable_model_analytics: bool,
    pub enable_model_backup: bool,
}

impl Default for ProductionModelOptions {
    fn default() -> Self {
        Self {
            enable_model_versioning: true,
            enable_model_compression: true,
            enable_model_encryption: true,
            enable_model_validation: true,
            enable_model_optimization: true,
            enable_distributed_storage: false,
            model_cache_size: 1000,
            model_timeout: std::time::Duration::from_secs(300),
            enable_model_analytics: true,
            enable_model_backup: true,
        }
    }
}

/// Executor pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorPoolConfig {
    pub core_pool_size: usize,
    pub max_pool_size: usize,
    pub keep_alive_time: std::time::Duration,
    pub queue_capacity: usize,
    pub enable_work_stealing: bool,
    pub enable_load_balancing: bool,
    pub priority_scheduling: bool,
}

impl Default for ExecutorPoolConfig {
    fn default() -> Self {
        Self {
            core_pool_size: 4,
            max_pool_size: 16,
            keep_alive_time: std::time::Duration::from_secs(300),
            queue_capacity: 1000,
            enable_work_stealing: true,
            enable_load_balancing: true,
            priority_scheduling: true,
        }
    }
}

/// Security constraints for algorithm execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConstraints {
    pub max_memory_usage: u64,
    pub max_cpu_time: std::time::Duration,
    pub max_network_connections: u32,
    pub allowed_file_operations: Vec<FileOperation>,
    pub sandbox_level: SandboxLevel,
    pub enable_resource_monitoring: bool,
    pub enable_anomaly_detection: bool,
}

impl Default for SecurityConstraints {
    fn default() -> Self {
        Self {
            max_memory_usage: 2048 * 1024 * 1024, // 2GB
            max_cpu_time: std::time::Duration::from_secs(600), // 10 minutes
            max_network_connections: 100,
            allowed_file_operations: vec![FileOperation::Read],
            sandbox_level: SandboxLevel::High,
            enable_resource_monitoring: true,
            enable_anomaly_detection: true,
        }
    }
}

/// File operation permissions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileOperation {
    Read,
    Write,
    Create,
    Delete,
    Execute,
    Modify,
}

/// Sandbox security level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SandboxLevel {
    Low,
    Medium,
    High,
    Maximum,
}

impl Default for SandboxLevel {
    fn default() -> Self {
        SandboxLevel::Medium
    }
}

/// Security policy definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityPolicy {
    pub id: String,
    pub name: String,
    pub policy_type: SecurityPolicyType,
    pub rules: Vec<SecurityRule>,
    pub priority: u32,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl SecurityPolicy {
    pub fn new_resource_limit_policy(limits: ResourceLimits) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: "Resource Limit Policy".to_string(),
            policy_type: SecurityPolicyType::ResourceLimit,
            rules: vec![SecurityRule::ResourceLimit { limits }],
            priority: 100,
            enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    pub fn new_network_access_policy(network_policy: crate::algorithm::sandbox_types::NetworkPolicy) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: "Network Access Policy".to_string(),
            policy_type: SecurityPolicyType::NetworkAccess,
            rules: vec![SecurityRule::NetworkAccess { policy: network_policy }],
            priority: 200,
            enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    pub fn new_filesystem_policy(filesystem_policy: crate::algorithm::sandbox_types::FilesystemPolicy) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: "Filesystem Policy".to_string(),
            policy_type: SecurityPolicyType::FilesystemAccess,
            rules: vec![SecurityRule::FilesystemAccess { policy: filesystem_policy }],
            priority: 300,
            enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    pub fn new_api_access_policy(api_policy: ApiAccessPolicy) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: "API Access Policy".to_string(),
            policy_type: SecurityPolicyType::ApiAccess,
            rules: vec![SecurityRule::ApiAccess { policy: api_policy }],
            priority: 400,
            enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    pub fn new_data_protection_policy(data_policy: DataProtectionPolicy) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: "Data Protection Policy".to_string(),
            policy_type: SecurityPolicyType::DataProtection,
            rules: vec![SecurityRule::DataProtection { policy: data_policy }],
            priority: 500,
            enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }
}

/// Security policy type enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityPolicyType {
    ResourceLimit,
    NetworkAccess,
    FilesystemAccess,
    ApiAccess,
    DataProtection,
    Custom(String),
}

/// Security rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityRule {
    ResourceLimit { limits: ResourceLimits },
    NetworkAccess { policy: crate::algorithm::sandbox_types::NetworkPolicy },
    FilesystemAccess { policy: crate::algorithm::sandbox_types::FilesystemPolicy },
    ApiAccess { policy: ApiAccessPolicy },
    DataProtection { policy: DataProtectionPolicy },
    Custom { rule_type: String, configuration: HashMap<String, serde_json::Value> },
}

/// API access policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiAccessPolicy {
    pub allowed_endpoints: Vec<String>,
    pub blocked_endpoints: Vec<String>,
    pub rate_limits: HashMap<String, RateLimit>,
    pub require_authentication: bool,
    pub allowed_methods: Vec<HttpMethod>,
}

impl Default for ApiAccessPolicy {
    fn default() -> Self {
        Self {
            allowed_endpoints: Vec::new(),
            blocked_endpoints: Vec::new(),
            rate_limits: HashMap::new(),
            require_authentication: true,
            allowed_methods: vec![HttpMethod::Get, HttpMethod::Post],
        }
    }
}

/// HTTP method enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
    Patch,
    Head,
    Options,
}

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    pub requests_per_minute: u32,
    pub burst_capacity: u32,
    pub window_size: std::time::Duration,
}

/// Data protection policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataProtectionPolicy {
    pub encryption_required: bool,
    pub encryption_algorithm: String,
    pub key_rotation_interval: std::time::Duration,
    pub data_classification: DataClassification,
    pub retention_policy: DataRetentionPolicy,
    pub access_controls: Vec<DataAccessControl>,
}

impl Default for DataProtectionPolicy {
    fn default() -> Self {
        Self {
            encryption_required: true,
            encryption_algorithm: "AES-256-GCM".to_string(),
            key_rotation_interval: std::time::Duration::from_secs(86400 * 30), // 30 days
            data_classification: DataClassification::Internal,
            retention_policy: DataRetentionPolicy::default(),
            access_controls: Vec::new(),
        }
    }
}

/// Data classification levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataClassification {
    Public,
    Internal,
    Confidential,
    Restricted,
}

/// Data retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRetentionPolicy {
    pub retention_period: std::time::Duration,
    pub auto_deletion: bool,
    pub archive_before_deletion: bool,
    pub legal_hold_supported: bool,
}

impl Default for DataRetentionPolicy {
    fn default() -> Self {
        Self {
            retention_period: std::time::Duration::from_secs(86400 * 365), // 1 year
            auto_deletion: false,
            archive_before_deletion: true,
            legal_hold_supported: true,
        }
    }
}

/// Data access control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAccessControl {
    pub user_id: String,
    pub permissions: Vec<DataPermission>,
    pub expiry: Option<DateTime<Utc>>,
    pub conditions: Vec<AccessCondition>,
}

/// Data permission enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataPermission {
    Read,
    Write,
    Delete,
    Share,
    Export,
    Print,
}

/// Access condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessCondition {
    pub condition_type: AccessConditionType,
    pub value: String,
}

/// Access condition type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessConditionType {
    IpAddress,
    TimeRange,
    Location,
    DeviceType,
    Custom(String),
}

/// Compliance requirements configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRequirements {
    pub enable_audit_logging: bool,
    pub enable_data_encryption: bool,
    pub enable_access_control: bool,
    pub retention_period: std::time::Duration,
    pub compliance_standards: Vec<ComplianceStandard>,
    pub privacy_protection_level: PrivacyLevel,
}

impl Default for ComplianceRequirements {
    fn default() -> Self {
        Self {
            enable_audit_logging: true,
            enable_data_encryption: true,
            enable_access_control: true,
            retention_period: std::time::Duration::from_secs(86400 * 90), // 90 days
            compliance_standards: Vec::new(),
            privacy_protection_level: PrivacyLevel::High,
        }
    }
}

/// Compliance standard enumeration
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplianceStandard {
    Gdpr,
    Hipaa,
    Sox,
    Pci,
    Iso27001,
    Custom(String),
}

/// Privacy protection level
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrivacyLevel {
    Low,
    Medium,
    High,
    Maximum,
}

/// Adaptive policy for dynamic security adjustment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptivePolicy {
    pub id: String,
    pub name: String,
    pub policy_type: AdaptivePolicyType,
    pub triggers: Vec<PolicyTrigger>,
    pub actions: Vec<PolicyAction>,
    pub conditions: Vec<PolicyCondition>,
    pub priority: u32,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl AdaptivePolicy {
    pub fn new_performance_policy(thresholds: PerformanceThresholds, actions: PerformanceActions) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: "Performance Adaptive Policy".to_string(),
            policy_type: AdaptivePolicyType::Performance,
            triggers: vec![PolicyTrigger::PerformanceThreshold { thresholds }],
            actions: vec![PolicyAction::PerformanceOptimization { actions }],
            conditions: Vec::new(),
            priority: 100,
            enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    pub fn new_security_policy(thresholds: SecurityThresholds, actions: SecurityActions) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: "Security Adaptive Policy".to_string(),
            policy_type: AdaptivePolicyType::Security,
            triggers: vec![PolicyTrigger::SecurityThreat { thresholds }],
            actions: vec![PolicyAction::SecurityResponse { actions }],
            conditions: Vec::new(),
            priority: 200,
            enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    pub fn new_resource_policy(thresholds: ResourceThresholds, actions: ResourceActions) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: "Resource Adaptive Policy".to_string(),
            policy_type: AdaptivePolicyType::Resource,
            triggers: vec![PolicyTrigger::ResourceUsage { thresholds }],
            actions: vec![PolicyAction::ResourceManagement { actions }],
            conditions: Vec::new(),
            priority: 300,
            enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }
}

/// Adaptive policy type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptivePolicyType {
    Performance,
    Security,
    Resource,
    Custom(String),
}

/// Policy trigger definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyTrigger {
    PerformanceThreshold { thresholds: PerformanceThresholds },
    SecurityThreat { thresholds: SecurityThresholds },
    ResourceUsage { thresholds: ResourceThresholds },
    Custom { trigger_type: String, configuration: HashMap<String, serde_json::Value> },
}

/// Policy action definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyAction {
    PerformanceOptimization { actions: PerformanceActions },
    SecurityResponse { actions: SecurityActions },
    ResourceManagement { actions: ResourceActions },
    Custom { action_type: String, configuration: HashMap<String, serde_json::Value> },
}

/// Policy condition definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyCondition {
    pub condition_type: PolicyConditionType,
    pub operator: ComparisonOperator,
    pub value: serde_json::Value,
}

/// Policy condition type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyConditionType {
    Time,
    Load,
    ResourceUsage,
    SecurityLevel,
    Custom(String),
}

/// Comparison operator for policy conditions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Contains,
    NotContains,
}

/// Performance thresholds for adaptive policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub cpu_usage_threshold: f64,
    pub memory_usage_threshold: f64,
    pub response_time_threshold: std::time::Duration,
    pub throughput_threshold: f64,
    pub error_rate_threshold: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            cpu_usage_threshold: 80.0,
            memory_usage_threshold: 85.0,
            response_time_threshold: std::time::Duration::from_millis(1000),
            throughput_threshold: 100.0,
            error_rate_threshold: 5.0,
        }
    }
}

/// Performance actions for adaptive policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceActions {
    pub scale_up: bool,
    pub scale_down: bool,
    pub optimize_algorithms: bool,
    pub adjust_cache_size: bool,
    pub enable_compression: bool,
}

impl Default for PerformanceActions {
    fn default() -> Self {
        Self {
            scale_up: true,
            scale_down: true,
            optimize_algorithms: true,
            adjust_cache_size: true,
            enable_compression: true,
        }
    }
}

/// Security thresholds for adaptive policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityThresholds {
    pub failed_login_threshold: u32,
    pub suspicious_activity_threshold: u32,
    pub anomaly_score_threshold: f64,
    pub threat_level_threshold: ThreatLevel,
}

impl Default for SecurityThresholds {
    fn default() -> Self {
        Self {
            failed_login_threshold: 5,
            suspicious_activity_threshold: 10,
            anomaly_score_threshold: 0.8,
            threat_level_threshold: ThreatLevel::Medium,
        }
    }
}

/// Threat level enumeration
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThreatLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Security actions for adaptive policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityActions {
    pub block_ip: bool,
    pub increase_monitoring: bool,
    pub require_additional_auth: bool,
    pub isolate_suspicious_processes: bool,
    pub alert_administrators: bool,
}

impl Default for SecurityActions {
    fn default() -> Self {
        Self {
            block_ip: false,
            increase_monitoring: true,
            require_additional_auth: false,
            isolate_suspicious_processes: true,
            alert_administrators: true,
        }
    }
}

/// Resource thresholds for adaptive policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceThresholds {
    pub memory_threshold: f64,
    pub cpu_threshold: f64,
    pub disk_threshold: f64,
    pub network_threshold: f64,
}

impl Default for ResourceThresholds {
    fn default() -> Self {
        Self {
            memory_threshold: 85.0,
            cpu_threshold: 80.0,
            disk_threshold: 90.0,
            network_threshold: 75.0,
        }
    }
}

/// Resource actions for adaptive policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceActions {
    pub free_memory: bool,
    pub scale_resources: bool,
    pub optimize_storage: bool,
    pub throttle_operations: bool,
}

impl Default for ResourceActions {
    fn default() -> Self {
        Self {
            free_memory: true,
            scale_resources: true,
            optimize_storage: true,
            throttle_operations: false,
        }
    }
}

/// Threat detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatDetectionConfig {
    pub enable_behavioral_analysis: bool,
    pub enable_pattern_detection: bool,
    pub enable_anomaly_detection: bool,
    pub detection_sensitivity: f64,
    pub response_time_threshold: std::time::Duration,
    pub max_detection_queue_size: usize,
}

impl Default for ThreatDetectionConfig {
    fn default() -> Self {
        Self {
            enable_behavioral_analysis: true,
            enable_pattern_detection: true,
            enable_anomaly_detection: true,
            detection_sensitivity: 0.8,
            response_time_threshold: std::time::Duration::from_millis(100),
            max_detection_queue_size: 10000,
        }
    }
}

/// 执行历史记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionHistory {
    pub execution_id: String,
    pub algorithm_id: String,
    pub execution_time: DateTime<Utc>,
    pub duration: std::time::Duration,
    pub status: crate::algorithm::execution_types::ExecutionStatus,
    pub input_size: usize,
    pub output_size: Option<usize>,
    pub resource_usage: crate::algorithm::execution_types::ResourceUsage,
    pub error_message: Option<String>,
    pub performance_metrics: PerformanceMetrics,
}

// ExecutionStatus is defined in execution_types.rs to avoid duplication

/// 系统资源使用情况
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResourceUsage {
    pub cpu_usage_percent: f64,
    pub memory_usage_bytes: usize,
    pub memory_total_bytes: usize,
    pub disk_usage_bytes: usize,
    pub disk_total_bytes: usize,
    pub network_in_bytes: usize,
    pub network_out_bytes: usize,
    pub gpu_usage_percent: Option<f64>,
    pub gpu_memory_usage_bytes: Option<usize>,
    pub gpu_memory_total_bytes: Option<usize>,
    pub timestamp: DateTime<Utc>,
    pub process_count: usize,
    pub load_average: (f64, f64, f64),
} 