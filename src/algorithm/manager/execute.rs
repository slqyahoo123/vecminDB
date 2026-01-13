// 算法执行器模块
// 负责调用底层执行器执行算法

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use log::{info, warn, error};
use serde_json::Value;
use chrono::{DateTime, Utc};

use crate::error::{Result, Error};
use crate::algorithm::types::{
    Algorithm, SandboxSecurityLevel, TaskStatus, TaskId, ResourceUsage, ResourceLimits, ExecutionStatus
};
use crate::algorithm::executor::config::{SandboxConfig as ExecutorSandboxConfig};
use crate::algorithm::executor::{ExecutorTrait};
use crate::algorithm::types::ExecutionResult;
use crate::algorithm::manager::config::AlgorithmManagerConfig;
use crate::algorithm::manager::{SecurityPolicyManager, AutoSecurityAdjuster};
// removed unused SecurityAssessment import; kept module references elsewhere

/// 任务执行结果
#[derive(Debug, Clone)]
pub struct TaskExecutionResult {
    /// 任务ID
    pub task_id: TaskId,
    /// 执行状态
    pub status: TaskStatus,
    /// 执行结果
    pub result: Option<String>,
    /// 错误信息
    pub error: Option<String>,
    /// 执行时间(秒)
    pub execution_time: f64,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 完成时间
    pub completed_at: Option<DateTime<Utc>>,
    /// 资源使用情况
    pub resource_usage: Option<ResourceUsage>,
}

/// 算法执行器配置
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// 任务ID
    pub task_id: Option<TaskId>,
    /// 沙箱配置
    pub sandbox_config: ExecutorSandboxConfig,
    /// 资源限制
    pub resource_limits: ResourceLimits,
    /// 安全级别
    pub security_level: SandboxSecurityLevel,
    /// 超时时间(毫秒)
    pub timeout_ms: u64,
    /// 是否启用监控
    pub enable_metrics: bool,
    /// 调试模式
    pub debug_mode: bool,
    /// 工作目录
    pub work_dir: std::path::PathBuf,
    /// 环境变量
    pub environment_variables: HashMap<String, String>,
    /// 最大并发任务数
    pub max_concurrent_tasks: Option<u32>,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            task_id: None,
            sandbox_config: ExecutorSandboxConfig::default(),
            resource_limits: ResourceLimits::default(),
            security_level: SandboxSecurityLevel::default(),
            timeout_ms: 30000, // 30秒
            enable_metrics: true,
            debug_mode: false,
            work_dir: std::env::temp_dir().join("vecmind_executor"),
            environment_variables: HashMap::new(),
            max_concurrent_tasks: Some(4),
        }
    }
}

/// 安全策略级别
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityPolicyLevel {
    /// 无安全限制
    None,
    /// 低级别安全
    Low,
    /// 中级别安全
    Medium,
    /// 高级别安全
    High,
}

impl Default for SecurityPolicyLevel {
    fn default() -> Self {
        SecurityPolicyLevel::Medium
    }
}

/// 算法执行器
pub struct AlgorithmExecutor {
    /// 底层执行器
    executor: Arc<dyn ExecutorTrait>,
    /// 安全策略管理器
    security_manager: Arc<SecurityPolicyManager>,
    /// 自动安全调整器
    auto_adjuster: Arc<AutoSecurityAdjuster>,
    /// 配置
    config: AlgorithmManagerConfig,
}

impl AlgorithmExecutor {
    /// 创建新的算法执行器
    pub fn new(
        executor: Arc<dyn ExecutorTrait>,
        security_manager: Arc<SecurityPolicyManager>,
        auto_adjuster: Arc<AutoSecurityAdjuster>,
        config: AlgorithmManagerConfig,
    ) -> Self {
        Self {
            executor,
            security_manager,
            auto_adjuster,
            config,
        }
    }

    /// 执行算法任务
    pub async fn execute(
        &self,
        task_id: &TaskId,
        algorithm: Arc<Algorithm>,
        parameters: HashMap<String, Value>,
        model_id: Option<&str>,
    ) -> Result<TaskExecutionResult> {
        info!("开始执行算法任务, 任务ID: {}, 算法: {}", task_id, algorithm.name);
        
        let start_time = Instant::now();
        
        // 1. 安全策略评估
        let security_assessment = self.security_manager
            .assess_algorithm_security(&algorithm, &parameters).await?;
        
        warn!("安全评估完成: {:?}", security_assessment);
        
        // 2. 调整安全配置
        let adjusted_config = self.auto_adjuster.adjust_security_config(
            &algorithm,
            &security_assessment,
        ).await?;

        info!("安全配置调整完成");
        
        // 3. 构建执行器配置
        let executor_config = crate::algorithm::executor::config::ExecutorConfig {
            task_id: Some(task_id.to_string()),
            sandbox_config: convert_sandbox_config(adjusted_config.sandbox_config),
            resource_limits: convert_resource_limits(adjusted_config.resource_limits),
            security_level: convert_security_level(adjusted_config.security_level),
            timeout_ms: self.config.algorithm_timeout_ms,
            enable_metrics: true,
            debug_mode: self.config.debug_mode,
            work_dir: std::env::temp_dir().join("vecmind_executor"),
            environment_variables: HashMap::new(),
            max_concurrent_tasks: Some(self.config.max_concurrent_tasks as u32),
        };
        
        // 4. 执行算法
        let execution_result = self.executor.execute(
            &algorithm,
            &parameters,
            model_id,
            &executor_config,
        ).await;
        
        let execution_time = start_time.elapsed();
        
        // 5. 处理执行结果
        match execution_result {
            Ok(result) => {
                info!("算法 {} 执行成功, 任务ID: {}, 耗时: {:?}", 
                     algorithm.name, task_id, execution_time);
                
                // 6. 验证执行结果
                if let Err(validation_error) = self.validate_execution_result(&result).await {
                    warn!("执行结果验证失败: {}", validation_error);
                    
                    // 构建验证失败的结果
                    let task_result = TaskExecutionResult {
                        task_id: task_id.clone(),
                        status: TaskStatus::Failed(format!("结果验证失败: {}", validation_error)),
                        result: None,
                        error: Some(format!("结果验证失败: {}", validation_error)),
                        execution_time: execution_time.as_secs_f64(),
                        created_at: chrono::Utc::now(),
                        completed_at: Some(chrono::Utc::now()),
                        resource_usage: None,
                    };
                    
                    return Ok(task_result);
                }
                
                // 7. 构建任务执行结果
                let task_result = TaskExecutionResult {
                    task_id: task_id.clone(),
                    status: TaskStatus::Completed,
                    result: Some(String::from_utf8_lossy(&result.output).to_string()),
                    error: None,
                    execution_time: execution_time.as_secs_f64(),
                    created_at: chrono::Utc::now(),
                    completed_at: Some(chrono::Utc::now()),
                    resource_usage: Some(result.resource_usage),
                };
                
                Ok(task_result)
            },
            Err(e) => {
                // 记录错误
                error!("算法 {} 执行失败, 任务ID: {}, 错误: {}", algorithm.name, task_id, e);
                
                // 构建失败的任务结果
                let task_result = TaskExecutionResult {
                    task_id: task_id.clone(),
                    status: TaskStatus::Failed(format!("算法执行失败: {}", e)),
                    result: None,
                    error: Some(format!("算法执行失败: {}", e)),
                    execution_time: start_time.elapsed().as_secs_f64(),
                    created_at: chrono::Utc::now(),
                    completed_at: Some(chrono::Utc::now()),
                    resource_usage: None,
                };
                
                Ok(task_result)
            }
        }
    }
    
    /// 取消算法执行
    pub async fn cancel_execution(&self, task_id: &TaskId) -> Result<()> {
        info!("取消算法执行, 任务ID: {}", task_id);
        // 转换TaskId类型
        let scheduler_task_id = crate::task_scheduler::core::TaskId::from_str(&task_id.0)
            .map_err(|e| Error::invalid_parameter(format!("无效的任务ID: {}", e)))?;
        self.executor.cancel_execution(&scheduler_task_id).await
    }
    
    /// 获取算法执行状态
    pub async fn get_execution_status(&self, task_id: &TaskId) -> Result<TaskStatus> {
        info!("获取算法执行状态, 任务ID: {}", task_id);
        // 转换TaskId类型
        let scheduler_task_id = crate::task_scheduler::core::TaskId::from_str(&task_id.0)
            .map_err(|e| Error::invalid_parameter(format!("无效的任务ID: {}", e)))?;
        let status = self.executor.get_execution_status(&scheduler_task_id).await?;
        Ok(convert_execution_status(status))
    }
    
    /// 验证执行结果
    async fn validate_execution_result(&self, result: &ExecutionResult) -> Result<()> {
        if matches!(result.status, ExecutionStatus::Failed(_)) {
            return Err(Error::execution(format!(
                "执行失败: {}",
                result.error.as_deref().unwrap_or("未知错误")
            )));
        }

        if result.output.is_empty() {
            warn!("执行结果输出为空");
        }
        
        Ok(())
    }
}

fn convert_execution_status(status: ExecutionStatus) -> TaskStatus {
    match status {
        ExecutionStatus::Pending => TaskStatus::Pending,
        ExecutionStatus::Running => TaskStatus::Running,
        ExecutionStatus::Success | ExecutionStatus::Completed => TaskStatus::Completed,
        ExecutionStatus::Failed(err) => TaskStatus::Failed(err),
        ExecutionStatus::Cancelled => TaskStatus::Cancelled,
        ExecutionStatus::Timeout => TaskStatus::Failed("Execution timeout".to_string()),
        ExecutionStatus::Paused => TaskStatus::Paused,
    }
}

fn convert_security_level(result_level: SandboxSecurityLevel) -> SandboxSecurityLevel {
    // 现在类型统一了，直接返回
    result_level
}

fn convert_resource_limits(result_limits: ResourceLimits) -> ResourceLimits {
    // 现在类型统一了，直接返回
    result_limits
}

fn convert_sandbox_config(result_config: crate::algorithm::types::SandboxConfig) -> crate::algorithm::executor::config::SandboxConfig {
    // 转换algorithm::types::SandboxConfig到algorithm::executor::config::SandboxConfig
    crate::algorithm::executor::config::SandboxConfig {
        sandbox_type: convert_sandbox_type_to_executor(result_config.sandbox_type),
        security_level: result_config.security_level,
        network_policy: result_config.network_policy,
        filesystem_policy: result_config.filesystem_policy,
        timeout: std::time::Duration::from_secs(30), // 默认30秒超时
        extra_params: std::collections::HashMap::new(),
    }
}

fn convert_sandbox_type_to_executor(sandbox_type: crate::algorithm::types::SandboxType) -> crate::algorithm::executor::config::SandboxType {
    match sandbox_type {
        crate::algorithm::types::SandboxType::LocalProcess => crate::algorithm::executor::config::SandboxType::LocalProcess,
        crate::algorithm::types::SandboxType::IsolatedProcess => crate::algorithm::executor::config::SandboxType::IsolatedProcess,
        crate::algorithm::types::SandboxType::Process => crate::algorithm::executor::config::SandboxType::Process,
        crate::algorithm::types::SandboxType::Wasm => crate::algorithm::executor::config::SandboxType::Wasm,
        crate::algorithm::types::SandboxType::Docker => crate::algorithm::executor::config::SandboxType::Docker,
    }
}

fn convert_network_policy(result_policy: crate::algorithm::types::NetworkPolicy) -> crate::algorithm::types::NetworkPolicy {
    // 现在类型统一了，直接返回
    result_policy
}

fn convert_filesystem_policy(result_policy: crate::algorithm::types::FilesystemPolicy) -> crate::algorithm::types::FilesystemPolicy {
    // 现在类型统一了，直接返回
    result_policy
} 