use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use uuid::Uuid;
use crate::{Error, Result};
use crate::algorithm::executor::{ExecutionStatus, ExecutionResult};
use crate::algorithm::types::ResourceUsage;

/// 沙箱执行状态
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SandboxStatus {
    /// 成功执行
    Success,
    /// 执行错误
    Error,
    /// 执行超时
    Timeout,
    /// 执行被取消
    Cancelled,
    /// 资源超限
    ResourceExceeded,
    /// 安全违规
    SecurityViolation,
}

impl Default for SandboxStatus {
    fn default() -> Self {
        SandboxStatus::Success
    }
}

/// 沙箱执行结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxResult {
    /// 是否成功
    pub success: bool,
    
    /// 退出码
    pub exit_code: i32,
    
    /// 标准输出
    pub stdout: String,
    
    /// 标准错误
    pub stderr: String,
    
    /// 错误信息
    pub error: Option<String>,
    
    /// 执行时间(毫秒)
    pub execution_time_ms: u64,
    
    /// 资源使用情况
    pub resource_usage: ResourceUsage,
    
    /// 运行指标
    pub metrics: HashMap<String, f64>,
    
    /// 沙箱执行状态
    pub status: SandboxStatus,
    
    /// 执行状态（用于兼容性）
    pub execution_status: ExecutionStatus,
    
    /// 警告信息
    pub warnings: Vec<String>,
    
    /// 开始时间
    pub start_time: chrono::DateTime<chrono::Utc>,
    
    /// 结束时间
    pub end_time: chrono::DateTime<chrono::Utc>,
    
    /// 执行ID
    pub execution_id: String,
}

impl SandboxResult {
    /// 创建新的基础结果
    pub fn new(
        success: bool,
        exit_code: i32,
        stdout: String,
        stderr: String,
        execution_time_ms: u64,
        resource_usage: ResourceUsage,
    ) -> Self {
        let sandbox_status = if success { SandboxStatus::Success } else { SandboxStatus::Error };
        let execution_status = if success { ExecutionStatus::Completed } else { ExecutionStatus::Failed("执行失败".to_string()) };
        
        Self {
            success,
            exit_code,
            stdout,
            stderr,
            error: if success { None } else { Some("执行失败".to_string()) },
            execution_time_ms,
            resource_usage,
            metrics: HashMap::new(),
            status: sandbox_status,
            execution_status,
            warnings: Vec::new(),
            start_time: chrono::Utc::now() - chrono::Duration::milliseconds(execution_time_ms as i64),
            end_time: chrono::Utc::now(),
            execution_id: Uuid::new_v4().to_string(),
        }
    }
    
    /// 创建新的成功结果
    pub fn success(stdout: String, stderr: String, execution_time_ms: u64, resource_usage: ResourceUsage) -> Self {
        Self {
            success: true,
            exit_code: 0,
            stdout,
            stderr,
            error: None,
            execution_time_ms,
            resource_usage,
            metrics: HashMap::new(),
            status: SandboxStatus::Success,
            execution_status: ExecutionStatus::Completed,
            warnings: Vec::new(),
            start_time: chrono::Utc::now() - chrono::Duration::milliseconds(execution_time_ms as i64),
            end_time: chrono::Utc::now(),
            execution_id: Uuid::new_v4().to_string(),
        }
    }
    
    /// 创建新的失败结果
    pub fn failure(exit_code: i32, stdout: String, stderr: String, error: String, execution_time_ms: u64, resource_usage: ResourceUsage) -> Self {
        Self {
            success: false,
            exit_code,
            stdout,
            stderr,
            error: Some(error.clone()),
            execution_time_ms,
            resource_usage,
            metrics: HashMap::new(),
            status: SandboxStatus::Error,
            execution_status: ExecutionStatus::Failed(error),
            warnings: Vec::new(),
            start_time: chrono::Utc::now() - chrono::Duration::milliseconds(execution_time_ms as i64),
            end_time: chrono::Utc::now(),
            execution_id: Uuid::new_v4().to_string(),
        }
    }
    
    /// 创建超时结果
    pub fn timeout(timeout_ms: u64, stdout: String, stderr: String, resource_usage: ResourceUsage) -> Self {
        let error_msg = format!("执行超时: {}ms", timeout_ms);
        Self {
            success: false,
            exit_code: 124, // 标准超时退出码
            stdout,
            stderr,
            error: Some(error_msg.clone()),
            execution_time_ms: timeout_ms,
            resource_usage,
            metrics: HashMap::new(),
            status: SandboxStatus::Timeout,
            execution_status: ExecutionStatus::Timeout,
            warnings: vec!["执行超时".to_string()],
            start_time: chrono::Utc::now() - chrono::Duration::milliseconds(timeout_ms as i64),
            end_time: chrono::Utc::now(),
            execution_id: Uuid::new_v4().to_string(),
        }
    }
    
    /// 创建取消结果
    pub fn cancelled(stdout: String, stderr: String, execution_time_ms: u64, resource_usage: ResourceUsage) -> Self {
        Self {
            success: false,
            exit_code: 130, // 标准取消退出码
            stdout,
            stderr,
            error: Some("执行被取消".to_string()),
            execution_time_ms,
            resource_usage,
            metrics: HashMap::new(),
            status: SandboxStatus::Cancelled,
            execution_status: ExecutionStatus::Cancelled,
            warnings: vec!["执行被用户取消".to_string()],
            start_time: chrono::Utc::now() - chrono::Duration::milliseconds(execution_time_ms as i64),
            end_time: chrono::Utc::now(),
            execution_id: Uuid::new_v4().to_string(),
        }
    }
    
    /// 添加警告信息
    pub fn add_warning(&mut self, warning: &str) {
        self.warnings.push(warning.to_string());
    }
    
    /// 添加指标
    pub fn add_metric(&mut self, key: &str, value: f64) {
        self.metrics.insert(key.to_string(), value);
    }
    
    /// 转换为ExecutionResult
    pub fn to_execution_result(&self) -> Result<ExecutionResult> {
        if !self.success {
            return Err(Error::execution(self.error.clone().unwrap_or_else(|| "执行失败".to_string())));
        }
        
        // 解析输出结果
        let output = match serde_json::from_str::<serde_json::Value>(&self.stdout) {
            Ok(value) => value,
            Err(e) => {
                return Err(Error::serialization(format!(
                    "无法解析算法输出: {}，错误: {}", 
                    self.stdout, e
                )));
            }
        };
        
        Ok(ExecutionResult {
            output,
            resource_usage: self.resource_usage.clone(),
            execution_time_ms: self.execution_time_ms,
            status: self.execution_status.clone(),
            logs: Some(self.stderr.clone()),
        })
    }
} 