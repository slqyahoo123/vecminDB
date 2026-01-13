// 类型定义模块
// 包含算法管理器相关的所有类型定义

pub mod storage;
pub mod model;
pub mod execution;
pub mod models;
pub mod progress;
pub mod security;

pub use storage::*;
pub use model::*;
// keep single consolidated re-export from this module
pub use execution::{TaskExecutionRecord, ExecutionStatistics};
pub use models::{AlgorithmModel, ModelConfiguration, ModelValidation};
pub use progress::{TaskProgress, ProgressInfo, ProgressCallback};
pub use security::{SecurityContext, SecurityLevel, SecurityPolicy};

// 直接从顶层类型模块重导出核心类型
pub use crate::algorithm::types::{TaskStatus, ResourceUsage};

// 重导出核心类型（已在上方导出，避免重复导出导致的E0252）