use serde::{Serialize, Deserialize};

/// Sandbox Status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SandboxStatus {
    Created,
    Uninitialized,
    Initialized,
    Ready,
    Running,
    Success,
    Completed,
    Failure(String),
    Failed,
    Timeout,
    Paused,
    Cancelled,
    Cleaned,
    Disposed,
}

impl Default for SandboxStatus {
    fn default() -> Self {
        SandboxStatus::Initialized
    }
}

/// Sandbox security level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SandboxSecurityLevel {
    Low,
    Standard,
    Medium,
    High,
    Strict,
}

impl Default for SandboxSecurityLevel {
    fn default() -> Self {
        SandboxSecurityLevel::Medium
    }
}

/// Execution mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode {
    Synchronous,
    Asynchronous,
    Parallel,
    Distributed,
}

impl Default for ExecutionMode {
    fn default() -> Self {
        ExecutionMode::Synchronous
    }
}

/// Network policy for sandbox
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkPolicy {
    Denied,
    LocalOnly,
    Restricted(Vec<String>),
    Allowed,
}

impl Default for NetworkPolicy {
    fn default() -> Self {
        NetworkPolicy::Denied
    }
}

/// Filesystem policy for sandbox
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilesystemPolicy {
    ReadOnly,
    WriteTemp,
    Restricted(Vec<String>),
    Full,
}

impl Default for FilesystemPolicy {
    fn default() -> Self {
        FilesystemPolicy::ReadOnly
    }
}

/// Sandbox type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SandboxType {
    Process,
    Container,
    Virtual,
    Native,
}

impl Default for SandboxType {
    fn default() -> Self {
        SandboxType::Process
    }
}

/// Task priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

impl Default for TaskPriority {
    fn default() -> Self {
        TaskPriority::Normal
    }
}

/// Execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    pub security_level: SandboxSecurityLevel,
    pub execution_mode: ExecutionMode,
    pub network_policy: NetworkPolicy,
    pub filesystem_policy: FilesystemPolicy,
    pub sandbox_type: SandboxType,
    pub priority: TaskPriority,
    pub timeout_ms: u64,
    pub max_memory_mb: u64,
    pub max_cpu_percent: f64,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            security_level: SandboxSecurityLevel::default(),
            execution_mode: ExecutionMode::default(),
            network_policy: NetworkPolicy::default(),
            filesystem_policy: FilesystemPolicy::default(),
            sandbox_type: SandboxType::default(),
            priority: TaskPriority::default(),
            timeout_ms: 30000,
            max_memory_mb: 1024,
            max_cpu_percent: 80.0,
        }
    }
}

/// Sandbox configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    pub sandbox_type: SandboxType,
    pub security_level: SandboxSecurityLevel,
    pub network_policy: NetworkPolicy,
    pub filesystem_policy: FilesystemPolicy,
    pub memory_limit_mb: u64,
    pub cpu_limit_percent: f64,
    pub timeout_seconds: u64,
    pub environment_variables: std::collections::HashMap<String, String>,
    pub working_directory: Option<String>,
    pub allowed_syscalls: Option<Vec<String>>,
    pub blocked_syscalls: Option<Vec<String>>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            sandbox_type: SandboxType::default(),
            security_level: SandboxSecurityLevel::default(),
            network_policy: NetworkPolicy::default(),
            filesystem_policy: FilesystemPolicy::default(),
            memory_limit_mb: 1024,
            cpu_limit_percent: 80.0,
            timeout_seconds: 30,
            environment_variables: std::collections::HashMap::new(),
            working_directory: None,
            allowed_syscalls: None,
            blocked_syscalls: None,
        }
    }
} 