use thiserror::Error;
use crate::Error;
use std::sync::PoisonError;

/// 沙箱错误类型
#[derive(Debug, Error)]
pub enum SandboxError {
    #[error("资源限制超出: {0}")]
    ResourceExceeded(String),
    
    #[error("执行超时: {0}ms")]
    Timeout(u64),
    
    #[error("安全违规: {0}")]
    SecurityViolation(String),
    
    #[error("初始化失败: {0}")]
    InitializationFailed(String),
    
    #[error("执行失败: {0}")]
    ExecutionFailed(String),
    
    #[error("系统错误: {0}")]
    SystemError(String),
    
    #[error("WASM编译错误: {0}")]
    WasmCompilation(String),
    
    #[error("WASM实例化错误: {0}")]
    WasmInstantiation(String),
    
    #[error("WASM执行错误: {0}")]
    WasmExecution(String),
    
    #[error("函数不存在: {0}")]
    FunctionNotFound(String),
    
    #[error("内存访问错误: {0}")]
    MemoryAccess(String),
    
    #[error("IO错误: {0}")]
    IoError(String),
    
    #[error("取消执行: {0}")]
    Cancelled(String),
    
    #[error("参数错误: {0}")]
    InvalidArgument(String),
    
    #[error("状态错误: {0}")]
    InvalidState(String),
    
    #[error("配置错误: {0}")]
    ConfigurationError(String),
    
    #[error("代码验证错误: {0}")]
    ValidationError(String),
    
    #[error("环境错误: {0}")]
    EnvironmentError(String),
    
    #[error("依赖错误: {0}")]
    DependencyError(String),
    
    #[error("并发错误: {0}")]
    ConcurrencyError(String),
    
    #[error("锁错误: {0}")]
    LockError(String),
    
    #[error("网络错误: {0}")]
    NetworkError(String),
    
    #[error("权限错误: {0}")]
    PermissionDenied(String),
    
    #[error("未找到: {0}")]
    NotFound(String),
    
    #[error("不支持: {0}")]
    Unsupported(String),
}

impl From<std::io::Error> for SandboxError {
    fn from(error: std::io::Error) -> Self {
        SandboxError::IoError(error.to_string())
    }
}

impl From<wasmtime::Error> for SandboxError {
    fn from(error: wasmtime::Error) -> Self {
        if error.to_string().contains("instantiate") {
            SandboxError::WasmInstantiation(error.to_string())
        } else if error.to_string().contains("compile") {
            SandboxError::WasmCompilation(error.to_string())
        } else {
            SandboxError::WasmExecution(error.to_string())
        }
    }
}

impl<T> From<PoisonError<T>> for SandboxError {
    fn from(error: PoisonError<T>) -> Self {
        SandboxError::LockError(format!("锁被毒化: {}", error))
    }
}

impl From<tokio::sync::TryLockError> for SandboxError {
    fn from(error: tokio::sync::TryLockError) -> Self {
        SandboxError::LockError(format!("无法获取锁: {}", error))
    }
}

impl From<tokio::task::JoinError> for SandboxError {
    fn from(error: tokio::task::JoinError) -> Self {
        SandboxError::ConcurrencyError(format!("任务执行失败: {}", error))
    }
}

impl From<Box<bincode::ErrorKind>> for SandboxError {
    fn from(error: Box<bincode::ErrorKind>) -> Self {
        SandboxError::SystemError(format!("序列化错误: {}", error))
    }
}

impl From<serde_json::Error> for SandboxError {
    fn from(error: serde_json::Error) -> Self {
        SandboxError::SystemError(format!("JSON错误: {}", error))
    }
}

impl From<std::string::FromUtf8Error> for SandboxError {
    fn from(error: std::string::FromUtf8Error) -> Self {
        SandboxError::SystemError(format!("UTF-8解码错误: {}", error))
    }
}

impl From<SandboxError> for Error {
    fn from(error: SandboxError) -> Self {
        match error {
            SandboxError::ResourceExceeded(msg) => Error::resource_exhausted(msg),
            SandboxError::Timeout(ms) => Error::deadline_exceeded(format!("执行超时: {}ms", ms)),
            SandboxError::SecurityViolation(msg) => Error::permission_denied(msg),
            SandboxError::Cancelled(msg) => Error::cancelled(msg),
            SandboxError::InvalidArgument(msg) => Error::invalid_argument(msg),
            SandboxError::NotFound(msg) => Error::not_found(msg),
            SandboxError::PermissionDenied(msg) => Error::permission_denied(msg),
            SandboxError::LockError(msg) => Error::lock(msg),
            SandboxError::Unsupported(msg) => Error::not_implemented(msg),
            _ => Error::internal(error.to_string()),
        }
    }
}

/// 为常见错误类型创建沙箱错误的辅助函数
pub fn function_not_found(name: &str) -> Error {
    SandboxError::FunctionNotFound(format!("函数不存在: {}", name)).into()
}

/// 为内存访问错误创建沙箱错误的辅助函数
pub fn memory_access(message: &str) -> Error {
    SandboxError::MemoryAccess(message.to_string()).into()
}

/// 为执行错误创建沙箱错误的辅助函数
pub fn execution_failed(message: &str) -> Error {
    SandboxError::ExecutionFailed(message.to_string()).into()
}

/// 为系统错误创建沙箱错误的辅助函数
pub fn system_error(message: &str) -> Error {
    SandboxError::SystemError(message.to_string()).into()
}

/// 为超时错误创建沙箱错误的辅助函数
pub fn timeout(ms: u64) -> Error {
    SandboxError::Timeout(ms).into()
}

/// 为资源超限错误创建沙箱错误的辅助函数
pub fn resource_exceeded(message: &str) -> Error {
    SandboxError::ResourceExceeded(message.to_string()).into()
}

/// 为安全违规错误创建沙箱错误的辅助函数
pub fn security_violation(message: &str) -> Error {
    SandboxError::SecurityViolation(message.to_string()).into()
}

/// 为初始化失败错误创建沙箱错误的辅助函数
pub fn initialization_failed(message: &str) -> Error {
    SandboxError::InitializationFailed(message.to_string()).into()
}

/// 为WASM相关错误创建沙箱错误的辅助函数
pub mod wasm {
    use super::*;
    
    /// 为WASM编译错误创建沙箱错误
    pub fn compilation(message: &str) -> Error {
        SandboxError::WasmCompilation(message.to_string()).into()
    }
    
    /// 为WASM实例化错误创建沙箱错误
    pub fn instantiation(message: &str) -> Error {
        SandboxError::WasmInstantiation(message.to_string()).into()
    }
    
    /// 为WASM执行错误创建沙箱错误
    pub fn execution(message: &str) -> Error {
        SandboxError::WasmExecution(message.to_string()).into()
    }
} 