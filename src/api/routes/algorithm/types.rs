// 算法相关类型定义（向量数据库精简版）

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

/// 算法信息（核心类型，被executor等模块使用）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmInfo {
    /// 算法ID
    pub id: String,
    /// 算法名称
    pub name: String,
    /// 算法类别
    pub category: String,
    /// 版本
    pub version: String,
    /// 描述
    pub description: Option<String>,
}

/// 算法参数定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmParameter {
    /// 参数名称
    pub name: String,
    /// 参数类型
    pub param_type: String,
    /// 是否必须
    pub required: bool,
    /// 默认值
    pub default_value: Option<serde_json::Value>,
    /// 描述
    pub description: Option<String>,
}

/// 算法详情
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmDetails {
    /// 算法ID
    pub id: String,
    /// 算法名称
    pub name: String,
    /// 算法类别
    pub category: String,
    /// 版本
    pub version: String,
    /// 描述
    pub description: Option<String>,
    /// 参数列表
    pub parameters: Vec<AlgorithmParameter>,
    /// 输入要求
    pub input_requirements: Option<HashMap<String, serde_json::Value>>,
    /// 输出格式
    pub output_format: Option<HashMap<String, serde_json::Value>>,
    /// 是否异步
    pub is_async: bool,
}

