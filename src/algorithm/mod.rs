// 算法模块
// 负责算法的管理、验证、执行和存储

// 现有的子模块定义
pub mod executor;
pub mod manager;
pub mod validator;
pub mod worker;
pub mod security;
pub mod security_auto;
pub mod storage;
pub mod types;
pub mod utils;
pub mod wasm;
// 注释掉：模型训练相关模块，vecminDB不需要
// pub mod linear;
// pub mod neural;
pub mod tree;
pub mod validation;
pub mod optimization;

// 新增的拆分模块
pub mod traits;
pub mod base_types;
pub mod resource;
pub mod status;
pub mod gpu;
pub mod enhanced_executor;
pub mod adapter;
pub mod factory;

// 新的拆分类型模块
pub mod algorithm_types;
pub mod execution_types;
pub mod sandbox_types;
pub mod management_types;
pub mod algorithm_impl;
pub mod algorithm;

// 重新导出基础trait和类型
pub use traits::{Algorithm, ValidationInterface, AlgorithmTrait};
pub use base_types::{
    Node, Connection, ConnectionType, SecurityCheck, SecurityLevel,
    AlgorithmStatus, LogLevel, StatusLog, ResourceType, ResourceUsageTrend,
    ExecutorControl, ExecutorType, ExecutionParams
};

// 导入WASM安全相关类型和函数
pub use wasm_security::{
    WasmSecuritySeverity, WasmExecutionStatistics,
    create_high_security_wasm_executor
};

// 重新导出资源相关
pub use resource::{
    AlgorithmResourceUsage, AlgorithmResourceLimits, ResourceUsageHistory,
    ResourceMonitoringConfig, ResourceWarningThresholds, ResourceWarning,
    ResourceReport, ResourceMonitor
};

// 重新导出状态相关
pub use status::{AlgorithmStatusTracker, AlgorithmStatusReport};

// 重新导出GPU相关
pub use gpu::{GpuInfo, GpuInfoProvider, NvidiaGpuInfoProvider};

// 从子模块重导出
pub use types::{
    Algorithm as ImportedAlgorithm, AlgorithmType, AlgorithmTask, 
    TaskStatus, TaskError, AlgorithmResult, 
    AlgorithmApplyConfig, AlgorithmMetadata,
    AlgorithmConfig, Optimizer
};

// 从core模块重新导出统一的AlgorithmDefinition类型
pub use crate::core::types::CoreAlgorithmDefinition as AlgorithmDefinition;

// 导出AlgorithmManager及相关类型
pub use manager::{
    AlgorithmManager, 
    AlgorithmManagerConfig,
};
pub use manager::config::TaskResourceLimits;
pub use manager::adapter::AlgorithmTraitAdapter;

// Avoid ambiguous glob re-exports by explicitly listing and not duplicating names elsewhere
// 注意：ThreatLevel已在types.rs中重新导出，这里不再重复导出
pub use security::{SecurityPolicyManager, SecurityPolicy, SecurityPolicyLevel, SecurityAuditEvent, SecurityAuditEventType, create_high_security_manager};
pub use security_auto::{AutoSecurityAdjuster, RiskLevel, RiskAssessment, ResourceThresholds, AutoAdjustConfig};
pub use storage::{AlgorithmStorage, LocalAlgorithmStorage};
pub use utils::{generate_id, AlgorithmValidationError};
pub use validator::AlgorithmValidator;
pub use worker::AlgorithmWorker;
pub use validation::{
    AlgorithmValidator as ValidationAlgorithmValidator, ValidationIssue, 
    ValidationSeverity, ValidationRule
};
pub use optimization::{
    AlgorithmOptimizer, AlgorithmOptimizationConfig, OptimizationType, OptimizationResult
};
pub use executor::{
    ExecutorConfig,
};
// 从新拆分的模块导出类型（AlgorithmType已经在第62行导入，这里只导入其他类型）
pub use algorithm_types::{AlgorithmOptimizationType};
// 注意：这些类型已在types.rs中重新导出，这里使用别名避免冲突
// 避免重复导出，使用明确的别名
// 统一以 core::types::CoreExecutionResult 为唯一来源，避免与算法模块内的执行结果命名冲突
pub use execution_types::{ResourceLimits as CoreResourceLimits, ExecutionStatus as CoreExecutionStatus};
// pub use sandbox_types::{SandboxStatus as NewSandboxStatus, SandboxSecurityLevel as NewSandboxSecurityLevel, ExecutionMode as NewExecutionMode, NetworkPolicy as NewNetworkPolicy, FilesystemPolicy as NewFilesystemPolicy, SandboxType as NewSandboxType, TaskPriority as NewTaskPriority, ExecutionConfig as NewExecutionConfig, SandboxConfig as NewSandboxConfig};
pub use algorithm_impl::Algorithm as AlgorithmImpl;

// 导出具体算法实现
// pub use linear::{LinearRegression, LinearRegressionParams};
// pub use neural::{NeuralNetwork, NeuralNetworkParams};
pub use tree::{DecisionTree, DecisionTreeParams};

// 重新导出增强执行器
pub use enhanced_executor::{EnhancedAlgorithmExecutor, EnhancedExecutorConfig};

// 重新导出适配器
pub use adapter::AlgorithmServiceAdapter;

// 重新导出工厂函数
pub use factory::{create_enhanced_algorithm_manager, create_executor};

/// 算法执行模块
/// 
/// 提供安全的算法执行环境，包括沙箱隔离、资源监控、恶意代码检测等功能。

use crate::error::{Error, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use log::{info, warn, error, debug};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use crate::core::unified_algorithm_service::ExecutionConfig;
// removed unused imports: Uuid, async_trait


// removed unused StatusTracker import; status types are re-exported separately

/// 算法模块专用错误类型
#[derive(Debug, thiserror::Error)]
pub enum AlgorithmError {
    #[error("算法验证失败: {0}")]
    ValidationFailed(String),
    
    #[error("算法执行失败: {0}")]
    ExecutionFailed(String),
    
    #[error("安全检查失败: {0}")]
    SecurityCheckFailed(String),
    
    #[error("资源限制错误: {0}")]
    ResourceLimitError(String),
    
    #[error("进程错误: {0}")]
    ProcessError(String),
    
    #[error("WASM执行错误: {0}")]
    WasmExecutionError(String),
    
    #[error("算法存储错误: {0}")]
    StorageError(String),
    
    #[error("配置错误: {0}")]
    ConfigurationError(String),
    
    #[error("超时错误: {0}")]
    TimeoutError(String),
    
    #[error("IO错误: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("序列化错误: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("其他错误: {0}")]
    Other(String),
}

// 新增的安全模块
pub mod security_engine;
pub mod wasm_security;
pub mod resource_enforcer;
pub use resource_enforcer::create_strict_resource_enforcer;

// 重新导出核心类型 - 避免重复导出
// pub use types::*;  // 已在上面明确导出
// pub use security::*;  // 已在上面明确导出
// pub use executor::sandbox::*;  // 已在上面明确导出
// 避免重复导出ResourceUsage，使用明确的导出
pub use security_engine::{
    AlgorithmSecurityEngine, ThreatType, ThreatDetectionResult, 
    TerminationStrategy, ProcessController, ExecutionSession, 
    SessionStatus, SecurityConfig, SecurityStats, MaliciousCodeDetector
};
pub use wasm_security::{
    WasmSecurityExecutor, WasmSecurityConfig, SecureWasmExecutionResult,
    WasmSecurityEvent, WasmSecurityEventType
};
pub use resource_enforcer::{
    ResourceEnforcer, ResourceMonitoringEvent, EnforcementStatistics,
    ResourceEnforcerConfig, ResourceLimit, ResourceEventType,
    EventSeverity
};

/// 执行统计信息
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExecutionStatistics {
    /// 总执行次数
    pub total_executions: u64,
    /// 成功执行次数
    pub successful_executions: u64,
    /// 失败执行次数
    pub failed_executions: u64,
    /// 平均执行时间（毫秒）
    pub average_execution_time: f64,
    /// 最短执行时间（毫秒）
    pub min_execution_time: u64,
    /// 最长执行时间（毫秒）
    pub max_execution_time: u64,
    /// 总执行时间（毫秒）
    pub total_execution_time: u64,
    /// 最后执行时间
    pub last_execution_time: DateTime<Utc>,
    /// 错误统计
    pub error_counts: HashMap<String, u64>,
    /// 算法类型统计
    pub algorithm_type_counts: HashMap<String, u64>,
    /// 资源使用统计
    pub resource_usage_stats: HashMap<String, f64>,
}

/// 统一的算法安全执行接口
pub struct SecureAlgorithmExecutor {
    /// 安全执行引擎
    security_engine: Arc<AlgorithmSecurityEngine>,
    /// WASM安全执行器
    wasm_executor: Arc<WasmSecurityExecutor>,
    /// 资源强制执行器
    resource_enforcer: Arc<RwLock<ResourceEnforcer>>,
    /// 全局配置
    config: Arc<RwLock<SecureExecutorConfig>>,
}

/// 安全执行器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureExecutorConfig {
    /// 默认安全级别
    pub default_security_level: SecurityLevel,
    /// 是否启用详细日志
    pub enable_detailed_logging: bool,
    /// 是否启用性能监控
    pub enable_performance_monitoring: bool,
    /// 是否启用自动威胁处理
    pub enable_auto_threat_handling: bool,
    /// 默认执行超时(秒)
    pub default_execution_timeout_seconds: u64,
    /// 最大并发执行数
    pub max_concurrent_executions: usize,
    /// 是否启用执行历史记录
    pub enable_execution_history: bool,
}

impl Default for SecureExecutorConfig {
    fn default() -> Self {
        Self {
            default_security_level: SecurityLevel::Standard,
            enable_detailed_logging: true,
            enable_performance_monitoring: true,
            enable_auto_threat_handling: true,
            default_execution_timeout_seconds: 30,
            max_concurrent_executions: 10,
            enable_execution_history: true,
        }
    }
}

/// 统一执行结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedExecutionResult {
    /// 执行ID
    pub execution_id: String,
    /// 算法ID
    pub algorithm_id: String,
    /// 是否成功
    pub success: bool,
    /// 输出数据
    pub output_data: Vec<u8>,
    /// 执行时间(毫秒)
    pub execution_time_ms: u64,
    /// 内存使用峰值(字节)
    pub peak_memory_bytes: usize,
    /// 安全事件
    pub security_events: Vec<SecurityEventSummary>,
    /// 性能指标
    pub performance_metrics: HashMap<String, f64>,
    /// 错误信息(如果失败)
    pub error_message: Option<String>,
    /// 安全评分 (0-100)
    pub security_score: u8,
}

/// 安全事件摘要
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEventSummary {
    /// 事件类型
    pub event_type: String,
    /// 严重程度
    pub severity: u8,
    /// 事件描述
    pub description: String,
    /// 处理结果
    pub handled: bool,
}

impl SecureAlgorithmExecutor {
    /// 创建新的安全算法执行器
    pub fn new() -> Result<Self> {
        let policy_manager = Arc::new(SecurityPolicyManager::new_default());
        let security_engine = Arc::new(AlgorithmSecurityEngine::new(policy_manager));
        let wasm_executor = Arc::new(WasmSecurityExecutor::new());
        
        let mut resource_enforcer = ResourceEnforcer::new();
        resource_enforcer.start()?;
        
        Ok(Self {
            security_engine,
            wasm_executor,
            resource_enforcer: Arc::new(RwLock::new(resource_enforcer)),
            config: Arc::new(RwLock::new(SecureExecutorConfig::default())),
        })
    }
    
    /// 使用自定义配置创建
    pub fn with_config(config: SecureExecutorConfig) -> Result<Self> {
        let mut executor = Self::new()?;
        *executor.config.write().unwrap() = config;
        Ok(executor)
    }
    
    /// 创建高安全级别执行器
    pub fn create_high_security() -> Result<Self> {
        let policy_manager = Arc::new(create_high_security_manager());
        let security_engine = Arc::new(AlgorithmSecurityEngine::new(policy_manager));
        let wasm_executor = Arc::new(create_high_security_wasm_executor());
        
        let mut resource_enforcer = create_strict_resource_enforcer();
        resource_enforcer.start()?;
        
        let mut config = SecureExecutorConfig::default();
        config.default_security_level = SecurityLevel::High;
        config.default_execution_timeout_seconds = 10;
        config.max_concurrent_executions = 5;
        
        Ok(Self {
            security_engine,
            wasm_executor,
            resource_enforcer: Arc::new(RwLock::new(resource_enforcer)),
            config: Arc::new(RwLock::new(config)),
        })
    }
    
    /// 安全执行算法
    pub async fn execute_algorithm_safely(
        &self,
        algorithm: &dyn crate::algorithm::traits::Algorithm,
        input_data: &[u8],
        custom_config: Option<ExecutionConfig>,
    ) -> Result<UnifiedExecutionResult> {
        let execution_id = uuid::Uuid::new_v4().to_string();
        let config = self.config.read().unwrap().clone();
        
        info!("开始安全执行算法: {} ({})", algorithm.get_name(), execution_id);
        
        // 检查并发限制
        if self.get_active_execution_count() >= config.max_concurrent_executions {
            return Err(Error::resource("并发执行数已达上限"));
        }
        
        // 确定执行超时
        let timeout_duration = custom_config
            .as_ref()
            .and_then(|c| c.timeout_seconds)
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(config.default_execution_timeout_seconds));
        
        // 根据算法类型选择执行路径
        let result = match algorithm.get_algorithm_type() {
            crate::algorithm::types::AlgorithmType::Wasm => {
                self.execute_wasm_algorithm(&execution_id, algorithm, input_data, timeout_duration).await
            },
            _ => {
                self.execute_native_algorithm(&execution_id, algorithm, input_data, timeout_duration).await
            }
        };
        
        info!("算法执行完成: {} - {}", execution_id, if result.is_ok() { "成功" } else { "失败" });
        result
    }
    
    /// 执行WASM算法
    async fn execute_wasm_algorithm(
        &self,
        execution_id: &str,
        algorithm: &dyn crate::algorithm::traits::Algorithm,
        input_data: &[u8],
        _timeout: Duration,
    ) -> Result<UnifiedExecutionResult> {
        debug!("执行WASM算法: {} ({})", algorithm.get_name(), execution_id);
        
        // 对于WASM算法，我们需要从algorithm的代码中获取二进制数据
        // 这里假设WASM二进制数据存储在algorithm的code字段中
        let wasm_binary = algorithm.get_code().as_bytes();
        if wasm_binary.is_empty() {
            return Err(Error::validation("WASM算法缺少二进制数据"));
        }
        
        // 执行WASM
        let wasm_result = self.wasm_executor.execute_secure(
            execution_id,
            wasm_binary,
            "main", // 默认入口函数
            input_data,
            None, // 使用默认配置
        ).await?;
        
        // 转换为统一结果
        Ok(UnifiedExecutionResult {
            execution_id: execution_id.to_string(),
            algorithm_id: algorithm.get_id().to_string(),
            success: wasm_result.success,
            output_data: wasm_result.output,
            execution_time_ms: wasm_result.execution_time_ms,
            peak_memory_bytes: wasm_result.stats.memory_usage,
            security_events: {
                let events = wasm_result.security_events.iter()
                    .map(|e| SecurityEventSummary {
                        event_type: format!("{:?}", e.event_type),
                        severity: match e.severity {
                            WasmSecuritySeverity::Info => 20,
                            WasmSecuritySeverity::Warning => 50,
                            WasmSecuritySeverity::Error => 80,
                            WasmSecuritySeverity::Critical => 100,
                        },
                        description: e.description.clone(),
                        handled: true,
                    }).collect::<Vec<_>>();
                events
            },
            performance_metrics: HashMap::from([
                ("fuel_consumed".to_string(), wasm_result.stats.fuel_consumed as f64),
                ("function_calls".to_string(), wasm_result.stats.function_calls as f64),
                ("instructions_executed".to_string(), wasm_result.stats.instructions_executed as f64),
            ]),
            error_message: if !wasm_result.success {
                Some("WASM执行失败".to_string())
            } else {
                None
            },
            security_score: self.calculate_security_score(&wasm_result.security_events),
        })
    }
    
    /// 执行原生算法
    async fn execute_native_algorithm(
        &self,
        execution_id: &str,
        algorithm: &dyn crate::algorithm::traits::Algorithm,
        input_data: &[u8],
        timeout: Duration,
    ) -> Result<UnifiedExecutionResult> {
        debug!("执行原生算法: {} ({})", algorithm.get_name(), execution_id);
        
        // 使用安全引擎执行
        let engine_result = self.security_engine.execute_algorithm_securely(
            algorithm,
            input_data,
            timeout,
        ).await?;
        
        // 转换为统一结果
        Ok(UnifiedExecutionResult {
            execution_id: execution_id.to_string(),
            algorithm_id: algorithm.get_id().to_string(),
            success: engine_result.success,
            output_data: engine_result.output.as_bytes().to_vec(),
            execution_time_ms: engine_result.execution_time.as_millis() as u64,
            peak_memory_bytes: engine_result.resource_usage.peak_memory_bytes as usize,
            security_events: vec![], // 原生算法没有详细的安全事件
            performance_metrics: {
                // 从resource_usage中提取可用的性能指标（统一为 f64）
                let mut metrics: HashMap<String, f64> = HashMap::new();
                metrics.insert(
                    "cpu_time_ms".to_string(),
                    engine_result.resource_usage.cpu_time_ms as f64,
                );
                metrics.insert(
                    "disk_read_bytes".to_string(),
                    engine_result.resource_usage.disk_read_bytes as f64,
                );
                metrics.insert(
                    "disk_write_bytes".to_string(),
                    engine_result.resource_usage.disk_write_bytes as f64,
                );
                metrics.insert(
                    "network_received_bytes".to_string(),
                    engine_result.resource_usage.network_received_bytes as f64,
                );
                metrics.insert(
                    "network_sent_bytes".to_string(),
                    engine_result.resource_usage.network_sent_bytes as f64,
                );
                metrics
            },
            error_message: if !engine_result.success {
                Some("原生算法执行失败".to_string())
            } else {
                None
            },
            security_score: 80, // 默认安全分数
        })
    }
    
    /// 计算WASM安全分数
    fn calculate_security_score(&self, events: &[WasmSecurityEvent]) -> u8 {
        if events.is_empty() {
            return 100;
        }
        
        let total_penalty: u32 = events.iter()
            .map(|e| match e.severity {
                WasmSecuritySeverity::Info => 1,
                WasmSecuritySeverity::Warning => 5,
                WasmSecuritySeverity::Error => 15,
                WasmSecuritySeverity::Critical => 30,
            })
            .sum();
        
        100_u8.saturating_sub(total_penalty as u8)
    }
    
    /// 强制终止指定执行
    pub async fn force_terminate_execution(&self, execution_id: &str, reason: &str) -> Result<()> {
        warn!("强制终止执行: {} - {}", execution_id, reason);
        
        // 终止安全引擎中的会话
        if let Err(e) = self.security_engine.force_terminate_session(execution_id, reason).await {
            warn!("安全引擎终止失败: {}", e);
        }
        
        // 终止WASM执行
        if let Err(e) = self.wasm_executor.force_terminate_execution(execution_id, reason).await {
            warn!("WASM执行器终止失败: {}", e);
        }
        
        Ok(())
    }
    
    /// 强制终止所有执行
    pub async fn force_terminate_all_executions(&self, reason: &str) -> Result<()> {
        warn!("强制终止所有执行 - {}", reason);
        
        // 终止安全引擎中的所有会话
        if let Err(e) = self.security_engine.force_terminate_all_sessions(reason).await {
            error!("安全引擎批量终止失败: {}", e);
        }
        
        // 终止所有WASM执行
        if let Err(e) = self.wasm_executor.force_terminate_all(reason).await {
            error!("WASM执行器批量终止失败: {}", e);
        }
        
        // 终止所有进程监控
        if let Err(e) = self.resource_enforcer.read().unwrap()
            .force_terminate_all_processes() {
            error!("资源执行器批量终止失败: {}", e);
        }
        
        Ok(())
    }
    
    /// 获取活动执行数量
    pub fn get_active_execution_count(&self) -> usize {
        let security_count = self.security_engine.get_active_session_count();
        let wasm_count = self.wasm_executor.get_active_execution_count();
        let process_count = self.resource_enforcer.read().unwrap().get_active_process_count();
        
        security_count + wasm_count + process_count
    }
    
    /// 获取系统统计信息
    pub fn get_system_statistics(&self) -> SystemStatistics {
        SystemStatistics {
            security_engine_stats: self.security_engine.get_execution_statistics(),
            wasm_executor_stats: self.wasm_executor.get_execution_statistics(),
            resource_enforcer_stats: self.resource_enforcer.read().unwrap().get_statistics(),
            active_executions: self.get_active_execution_count(),
        }
    }
    
    /// 获取最近的安全事件
    pub fn get_recent_security_events(&self, limit: usize) -> Vec<ResourceMonitoringEvent> {
        self.resource_enforcer.read().unwrap().get_recent_events(limit)
    }
    
    /// 更新配置
    pub fn update_config(&self, new_config: SecureExecutorConfig) {
        *self.config.write().unwrap() = new_config;
        info!("安全执行器配置已更新");
    }
    
    /// 获取当前配置
    pub fn get_config(&self) -> SecureExecutorConfig {
        self.config.read().unwrap().clone()
    }
    
    /// 健康检查
    pub fn health_check(&self) -> HealthCheckResult {
        HealthCheckResult {
            overall_status: HealthStatus::Healthy,
            security_engine_healthy: true,
            wasm_executor_healthy: true,
            resource_enforcer_healthy: true,
            active_executions: self.get_active_execution_count(),
            last_check: chrono::Utc::now(),
        }
    }
}

impl Drop for SecureAlgorithmExecutor {
    fn drop(&mut self) {
        // 确保所有资源被正确清理
        if let Err(e) = futures::executor::block_on(
            self.force_terminate_all_executions("系统关闭")
        ) {
            error!("清理执行器时出错: {}", e);
        }
    }
}

// ExecutionConfig 现在从 types 模块导出，不再在这里重复定义

/// 系统统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatistics {
    /// 安全引擎统计
    pub security_engine_stats: ExecutionStatistics,
    /// WASM执行器统计
    pub wasm_executor_stats: WasmExecutionStatistics,
    /// 资源执行器统计
    pub resource_enforcer_stats: EnforcementStatistics,
    /// 活动执行数量
    pub active_executions: usize,
}

/// 健康检查结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// 总体状态
    pub overall_status: HealthStatus,
    /// 安全引擎是否健康
    pub security_engine_healthy: bool,
    /// WASM执行器是否健康
    pub wasm_executor_healthy: bool,
    /// 资源执行器是否健康
    pub resource_enforcer_healthy: bool,
    /// 活动执行数量
    pub active_executions: usize,
    /// 检查时间
    pub last_check: chrono::DateTime<chrono::Utc>,
}

/// 健康状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// 健康
    Healthy,
    /// 警告
    Warning,
    /// 错误
    Error,
    /// 严重错误
    Critical,
}

/// 创建默认安全算法执行器
pub fn create_secure_algorithm_executor() -> Result<SecureAlgorithmExecutor> {
    SecureAlgorithmExecutor::new()
}

/// 创建高安全级别执行器
pub fn create_high_security_algorithm_executor() -> Result<SecureAlgorithmExecutor> {
    SecureAlgorithmExecutor::create_high_security()
}

/// 创建开发环境执行器 (宽松的安全限制)
pub fn create_development_algorithm_executor() -> Result<SecureAlgorithmExecutor> {
    let mut config = SecureExecutorConfig::default();
    config.default_security_level = SecurityLevel::Low;
    config.default_execution_timeout_seconds = 60;
    config.max_concurrent_executions = 20;
    config.enable_auto_threat_handling = false;
    
    SecureAlgorithmExecutor::with_config(config)
}

/// 创建生产环境执行器 (严格的安全限制)
pub fn create_production_algorithm_executor() -> Result<SecureAlgorithmExecutor> {
    let mut config = SecureExecutorConfig::default();
    config.default_security_level = SecurityLevel::High;
    config.default_execution_timeout_seconds = 15;
    config.max_concurrent_executions = 5;
    config.enable_auto_threat_handling = true;
    
    SecureAlgorithmExecutor::with_config(config)
}

// 为Box<dyn Algorithm>实现Algorithm trait
// 这个实现会造成递归，Box<dyn Algorithm> 已经自动支持 Algorithm trait
// 删除这个实现，因为它是不必要的且有问题的

// Core algorithm modules already declared above

// Re-export key types for easy access (removing duplicates)
pub use algorithm::Algorithm as NewAlgorithm;
pub use algorithm_types::{AlgorithmType as CoreAlgorithmType, AlgorithmOptimizationType as CoreAlgorithmOptimizationType};
pub use sandbox_types::{SandboxStatus as CoreSandboxStatus};
