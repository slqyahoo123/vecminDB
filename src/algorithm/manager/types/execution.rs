// 执行相关类型定义

use std::time::{Duration, SystemTime};
use serde::{Serialize, Deserialize};

// 重新导出核心类型以供外部使用
pub use crate::algorithm::types::{ResourceUsage, TaskStatus};

/// 任务执行记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskExecutionRecord {
    pub task_id: String,
    pub algorithm_id: String,
    pub start_time: SystemTime,
    pub end_time: Option<SystemTime>,
    pub status: TaskStatus,
    pub resource_usage: ResourceUsage,
    pub error_message: Option<String>,
}

/// 执行统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStatistics {
    pub total_tasks: u64,
    pub successful_tasks: u64,
    pub failed_tasks: u64,
    pub cancelled_tasks: u64,
    pub average_execution_time: Duration,
    pub peak_memory_usage: u64,
    pub total_cpu_time: Duration,
}

impl Default for ExecutionStatistics {
    fn default() -> Self {
        Self {
            total_tasks: 0,
            successful_tasks: 0,
            failed_tasks: 0,
            cancelled_tasks: 0,
            average_execution_time: Duration::from_secs(0),
            peak_memory_usage: 0,
            total_cpu_time: Duration::from_secs(0),
        }
    }
} 