//! 算法执行结果相关类型定义
//!
//! 定义算法执行的结果、状态、任务等核心类型

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use thiserror::Error;
use uuid::Uuid;
use chrono;

use crate::algorithm::types::{
    ExecutionConfig, TaskPriority, ResourceUsage,
};

/// 执行模式
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutionMode {
    /// 本地执行模式
    Local,
    /// 远程执行模式
    Remote,
    /// 分布式执行模式
    Distributed,
    /// 沙箱隔离执行模式
    Sandboxed,
    /// 容器化执行模式
    Containerized,
    /// 虚拟机执行模式
    Virtualized,
}

impl Default for ExecutionMode {
    fn default() -> Self {
        ExecutionMode::Sandboxed
    }
}

/// 执行器专用网络策略（比通用策略更详细）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutorNetworkPolicy {
    /// 完全禁止网络访问
    Denied,
    /// 允许所有网络访问
    Allowed,
    /// 仅允许HTTP/HTTPS访问
    HttpOnly,
    /// 仅允许本地网络访问
    LocalOnly,
    /// 自定义白名单访问
    Whitelist(Vec<String>),
    /// 自定义黑名单访问
    Blacklist(Vec<String>),
}

impl Default for ExecutorNetworkPolicy {
    fn default() -> Self {
        ExecutorNetworkPolicy::Denied
    }
}

/// 执行器专用文件系统策略（比通用策略更详细）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutorFilesystemPolicy {
    /// 完全禁止文件系统访问
    Denied,
    /// 允许只读访问
    ReadOnly,
    /// 允许读写访问
    ReadWrite,
    /// 仅允许临时目录访问
    TempOnly,
    /// 自定义路径白名单
    Whitelist(Vec<String>),
    /// 自定义路径黑名单
    Blacklist(Vec<String>),
}

impl Default for ExecutorFilesystemPolicy {
    fn default() -> Self {
        ExecutorFilesystemPolicy::ReadOnly
    }
}

/// 执行器专用沙箱状态（详细的执行状态）
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutorSandboxStatus {
    /// 成功执行
    Success,
    /// 执行失败
    Error,
    /// 执行超时
    Timeout,
    /// 被取消
    Cancelled,
    /// 资源超限
    ResourceExceeded,
    /// 安全违规
    SecurityViolation,
}

impl Default for ExecutorSandboxStatus {
    fn default() -> Self {
        ExecutorSandboxStatus::Success
    }
}

/// 执行器专用沙箱类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutorSandboxType {
    /// 无沙箱隔离
    None,
    /// 基础沙箱
    Basic,
    /// 严格沙箱
    Strict,
    /// 自定义沙箱
    Custom(String),
}

impl Default for ExecutorSandboxType {
    fn default() -> Self {
        ExecutorSandboxType::Basic
    }
}

/// 执行器专用沙箱配置（详细配置选项）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorSandboxConfig {
    /// 沙箱类型
    pub sandbox_type: ExecutorSandboxType,
    /// 允许的系统调用
    pub allowed_syscalls: Vec<String>,
    /// 阻止的系统调用
    pub blocked_syscalls: Vec<String>,
    /// 网络策略
    pub network_policy: ExecutorNetworkPolicy,
    /// 文件系统策略
    pub filesystem_policy: ExecutorFilesystemPolicy,
    /// 内存限制（字节）
    pub memory_limit: Option<usize>,
    /// CPU时间限制（秒）
    pub cpu_time_limit: Option<u64>,
    /// 最大文件描述符数
    pub max_file_descriptors: Option<u32>,
    /// 最大进程数
    pub max_processes: Option<u32>,
    /// 环境变量控制
    pub environment_variables: HashMap<String, String>,
    /// 工作目录
    pub working_directory: Option<String>,
}

impl Default for ExecutorSandboxConfig {
    fn default() -> Self {
        Self {
            sandbox_type: ExecutorSandboxType::Basic,
            allowed_syscalls: vec![
                "read".to_string(),
                "write".to_string(),
                "open".to_string(),
                "close".to_string(),
                "stat".to_string(),
                "fstat".to_string(),
                "lstat".to_string(),
                "mmap".to_string(),
                "mprotect".to_string(),
                "munmap".to_string(),
                "brk".to_string(),
                "exit".to_string(),
                "exit_group".to_string(),
            ],
            blocked_syscalls: vec![
                "socket".to_string(),
                "connect".to_string(),
                "bind".to_string(),
                "listen".to_string(),
                "accept".to_string(),
                "fork".to_string(),
                "execve".to_string(),
                "kill".to_string(),
                "ptrace".to_string(),
            ],
            network_policy: ExecutorNetworkPolicy::Denied,
            filesystem_policy: ExecutorFilesystemPolicy::ReadOnly,
            memory_limit: Some(100 * 1024 * 1024), // 100MB
            cpu_time_limit: Some(60), // 60秒
            max_file_descriptors: Some(1024),
            max_processes: Some(1),
            environment_variables: HashMap::new(),
            working_directory: None,
        }
    }
}

/// 执行错误
#[derive(Debug, Clone, Error, Serialize, Deserialize, PartialEq)]
pub enum ExecutionError {
    #[error("资源限制超出: {0}")]
    ResourceExceeded(String),
    
    #[error("执行超时: {0}ms")]
    Timeout(u64),
    
    #[error("安全验证失败: {0}")]
    SecurityViolation(String),
    
    #[error("代码编译失败: {0}")]
    CompilationFailed(String),
    
    #[error("运行时错误: {0}")]
    RuntimeError(String),
    
    #[error("系统错误: {0}")]
    SystemError(String),
    
    #[error("未知错误: {0}")]
    Unknown(String),
}

/// 执行任务
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTask {
    /// 任务ID
    pub id: String,
    /// 算法ID
    pub algorithm_id: String,
    /// 算法代码
    pub algorithm_code: String,
    /// 输入数据
    pub input_data: Vec<u8>,
    /// 任务参数
    pub parameters: HashMap<String, serde_json::Value>,
    /// 执行配置
    pub config: ExecutionConfig,
    /// 任务状态
    pub status: crate::algorithm::types::ExecutionStatus,
    /// 创建时间
    pub created_at: SystemTime,
    /// 开始时间
    pub started_at: Option<SystemTime>,
    /// 完成时间
    pub completed_at: Option<SystemTime>,
    /// 优先级
    pub priority: TaskPriority,
    /// 重试次数
    pub retry_count: u32,
    /// 最大重试次数
    pub max_retries: u32,
    /// 任务标签
    pub tags: HashMap<String, String>,
    /// 执行结果
    pub result: Option<crate::algorithm::types::ExecutionResult>,
}

/// 执行状态信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionState {
    /// 当前状态
    pub status: crate::algorithm::types::ExecutionStatus,
    /// 进度百分比
    pub progress: f32,
    /// 已处理项目数
    pub processed_items: usize,
    /// 总项目数
    pub total_items: usize,
    /// 当前步骤
    pub current_step: String,
    /// 步骤详情
    pub step_details: HashMap<String, String>,
    /// 执行时间
    pub elapsed_time: Duration,
    /// 估算剩余时间
    pub estimated_remaining: Option<Duration>,
    /// 资源使用情况
    pub resource_usage: ResourceUsage,
    /// 最后更新时间
    pub last_updated: SystemTime,
    /// 错误信息
    pub error_info: Option<ExecutionError>,
    /// 警告信息
    pub warnings: Vec<String>,
}

impl ExecutionTask {
    /// 创建一个新的执行任务
    pub fn new(
        algorithm_id: String,
        algorithm_code: String,
        input_data: Vec<u8>,
        parameters: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            algorithm_id,
            algorithm_code,
            input_data,
            parameters,
            config: ExecutionConfig::default(),
            status: crate::algorithm::types::ExecutionStatus::Pending,
            created_at: SystemTime::now(),
            started_at: None,
            completed_at: None,
            priority: TaskPriority::Normal,
            retry_count: 0,
            max_retries: 3,
            tags: HashMap::new(),
            result: None,
        }
    }

    /// 将任务标记为开始执行
    pub fn start(&mut self) {
        if self.status == crate::algorithm::types::ExecutionStatus::Pending {
            self.status = crate::algorithm::types::ExecutionStatus::Running;
            self.started_at = Some(SystemTime::now());
        }
    }

    /// 将任务标记为成功完成
    pub fn complete(&mut self, result: crate::algorithm::types::ExecutionResult) {
        self.status = crate::algorithm::types::ExecutionStatus::Completed;
        self.completed_at = Some(SystemTime::now());
        self.result = Some(result);
    }

    /// 将任务标记为失败
    pub fn fail(&mut self, error_message: String) {
        self.status = crate::algorithm::types::ExecutionStatus::Failed(error_message.clone());
        self.completed_at = Some(SystemTime::now());
        
        let execution_result = crate::algorithm::types::ExecutionResult {
             algorithm_id: self.algorithm_id.clone(),
             execution_id: self.id.clone(),
             output: vec![],
             status: self.status.clone(),
             metrics: Default::default(),
             error: Some(error_message),
             resource_usage: Default::default(),
             start_time: chrono::Utc::now(),
             end_time: chrono::Utc::now(),
             sandbox_result: None,
        };
        self.result = Some(execution_result);
    }
    
    /// 检查任务是否可以重试
    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
            && matches!(
                self.status,
                crate::algorithm::types::ExecutionStatus::Failed(_) | crate::algorithm::types::ExecutionStatus::Timeout
            )
    }

    /// 为重试准备任务，增加重试计数器并重置状态
    pub fn prepare_for_retry(&mut self) {
        if self.can_retry() {
            self.retry_count += 1;
            self.status = crate::algorithm::types::ExecutionStatus::Pending;
            self.started_at = None;
            self.completed_at = None;
            self.result = None;
        }
    }

    /// 获取任务的执行时长
    pub fn duration(&self) -> Option<Duration> {
        match (self.started_at, self.completed_at) {
            (Some(start), Some(end)) => end.duration_since(start).ok(),
            (Some(start), None) => SystemTime::now().duration_since(start).ok(),
            _ => None,
        }
    }
}

impl ExecutionState {
    /// 创建一个新的执行状态
    pub fn new() -> Self {
        Self::default()
    }

    /// 更新进度
    pub fn update_progress(&mut self, processed: usize, total: usize) {
        self.processed_items = processed;
        self.total_items = total;
        self.progress = if total > 0 {
            (processed as f32 / total as f32) * 100.0
        } else {
            0.0
        };
        self.last_updated = SystemTime::now();
    }

    /// 更新当前步骤
    pub fn update_step(&mut self, step: String, details: Option<HashMap<String, String>>) {
        self.current_step = step;
        if let Some(details) = details {
            self.step_details = details;
        }
        self.last_updated = SystemTime::now();
    }
    
    /// 开始执行
    pub fn start(&mut self) {
        self.status = crate::algorithm::types::ExecutionStatus::Running;
        self.last_updated = SystemTime::now();
        self.elapsed_time = Duration::from_secs(0);
    }

    /// 标记为成功
    pub fn complete(&mut self) {
        self.status = crate::algorithm::types::ExecutionStatus::Completed;
        self.progress = 100.0;
        self.last_updated = SystemTime::now();
    }
    
    /// 标记为失败
    pub fn fail(&mut self, error: ExecutionError) {
        self.status = crate::algorithm::types::ExecutionStatus::Failed(format!("{:?}", error));
        self.error_info = Some(error);
        self.last_updated = SystemTime::now();
    }
}

impl Default for ExecutionState {
    fn default() -> Self {
        Self {
            status: crate::algorithm::types::ExecutionStatus::Pending,
            progress: 0.0,
            processed_items: 0,
            total_items: 0,
            current_step: "初始化".to_string(),
            step_details: HashMap::new(),
            elapsed_time: Duration::from_secs(0),
            estimated_remaining: None,
            resource_usage: ResourceUsage::default(),
            last_updated: SystemTime::now(),
            error_info: None,
            warnings: Vec::new(),
        }
    }
}

// ExecutionMetrics in metrics.rs, re-export it.
pub use super::metrics::ExecutionMetrics;
