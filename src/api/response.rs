//! API响应模块
//! 定义各种API响应结构

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// 通用API响应结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    /// 操作是否成功
    pub success: bool,
    /// 响应数据
    pub data: Option<T>,
    /// 错误信息
    pub error: Option<String>,
    /// 响应代码
    pub code: Option<i32>,
    /// 响应消息
    pub message: Option<String>,
}

impl<T> ApiResponse<T> {
    /// 创建成功响应
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            code: Some(200),
            message: None,
        }
    }

    /// 创建带消息的成功响应
    pub fn success_with_message(message: impl Into<String>, data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            code: Some(200),
            message: Some(message.into()),
        }
    }

    /// 创建错误响应（双参数版本）
    pub fn error(msg: impl Into<String>, code: i32) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(msg.into()),
            code: Some(code),
            message: None,
        }
    }
    
    /// 创建错误响应（带详细信息版本）
    pub fn error_with_details(msg: impl Into<String>, details: impl Into<String>) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(msg.into()),
            code: Some(400),
            message: Some(details.into()),
        }
    }
    
    /// 创建错误响应（单参数版本，用于向后兼容）
    pub fn error_msg(msg: impl Into<String>) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(msg.into()),
            code: Some(400),
            message: None,
        }
    }
}

impl<T> std::fmt::Display for ApiResponse<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.success {
            write!(f, "Success")
        } else {
            write!(f, "Error: {}", self.error.as_deref().unwrap_or("Unknown error"))
        }
    }
}

/// 健康检查响应
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// 服务状态
    pub status: String,
    /// 版本信息
    pub version: String,
    /// 上线时间
    pub uptime: u64,
    /// 系统指标
    pub metrics: HashMap<String, serde_json::Value>,
}

