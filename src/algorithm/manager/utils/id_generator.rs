// ID生成器工具

use uuid::Uuid;
use std::time::{SystemTime, UNIX_EPOCH};

/// 生成唯一ID
pub fn generate_id() -> String {
    Uuid::new_v4().to_string()
}

/// 生成带时间戳的ID
pub fn generate_timestamped_id(prefix: &str) -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    
    format!("{}_{}_{}", prefix, timestamp, Uuid::new_v4().to_string().split('-').next().unwrap_or(""))
}

/// 生成算法ID
pub fn generate_algorithm_id() -> String {
    generate_timestamped_id("algo")
}

/// 生成执行ID
pub fn generate_execution_id() -> String {
    generate_timestamped_id("exec")
}

/// 生成模型ID
pub fn generate_model_id() -> String {
    generate_timestamped_id("model")
}

/// 生成任务ID
pub fn generate_task_id() -> String {
    generate_timestamped_id("task")
} 