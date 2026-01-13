use std::collections::HashMap;
use std::time::SystemTime;
use serde::{Serialize, Deserialize};
use crate::algorithm::utils;

/// DSL节点类型
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Node {
    Input { name: String, data_type: String },
    Output { name: String, data_type: String },
    Operation { op_type: String, config: HashMap<String, String> },
    Control { control_type: String, config: HashMap<String, String> },
    Custom { name: String, code: String },
}

/// 连接类型
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum ConnectionType {
    /// 串行
    Serial,
    /// 并行
    Parallel,
    /// 分支
    Branch,
    /// 合并
    Merge,
}

/// 连接结构
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Connection {
    /// 连接ID
    pub id: String,
    /// 源节点ID
    pub source_id: String,
    /// 目标节点ID
    pub target_id: String,
    /// 连接类型
    pub connection_type: ConnectionType,
    /// 条件表达式
    pub condition: Option<String>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

impl Connection {
    /// 创建新连接
    pub fn new(source_id: &str, target_id: &str, connection_type: ConnectionType) -> Self {
        Self {
            id: utils::generate_id(),
            source_id: source_id.to_string(),
            target_id: target_id.to_string(),
            connection_type,
            condition: None,
            metadata: HashMap::new(),
        }
    }
    
    /// 设置条件
    pub fn with_condition(mut self, condition: &str) -> Self {
        self.condition = Some(condition.to_string());
        self
    }
    
    /// 添加元数据
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }
}

/// 安全检查结果
#[derive(Debug, Clone)]
pub struct SecurityCheck {
    pub security_level: SecurityLevel,
    pub potential_risks: Vec<String>,
    pub recommended_policies: Vec<String>,
}

/// 安全级别
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SecurityLevel {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// 日志级别
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// 调试信息
    Debug,
    /// 普通信息
    Info,
    /// 警告信息
    Warning,
    /// 错误信息
    Error,
}

/// 算法状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgorithmStatus {
    /// 队列中
    Queued,
    /// 待处理
    Pending,
    /// 初始化中
    Initializing,
    /// 正在运行
    Running,
    /// 已暂停
    Paused,
    /// 已完成
    Completed,
    /// 已失败
    Failed,
    /// 已取消
    Canceled,
    /// 未知状态
    Unknown,
}

/// 执行器控制
#[derive(Debug, Clone)]
pub enum ExecutorControl {
    /// 暂停执行
    Pause,
    /// 恢复执行
    Resume,
    /// 取消执行
    Cancel,
    /// 更新资源限制
    UpdateResourceLimits(crate::algorithm::resource::AlgorithmResourceLimits),
}

/// 执行器类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutorType {
    /// 本地执行器
    Local,
    /// Docker沙箱
    Docker,
    /// WebAssembly沙箱
    Wasm,
    /// 安全的V8沙箱
    V8,
    /// 远程执行器
    Remote,
}

/// 执行参数
#[derive(Debug, Clone)]
pub struct ExecutionParams {
    /// 模型ID（可选）
    pub model_id: Option<String>,
    /// 执行参数
    pub parameters: HashMap<String, serde_json::Value>,
    /// 超时时间（秒）
    pub timeout: Option<u64>,
    /// 自定义配置
    pub config: Option<HashMap<String, String>>,
}

/// 状态日志
#[derive(Debug, Clone)]
pub struct StatusLog {
    /// 时间戳
    pub timestamp: SystemTime,
    /// 日志级别
    pub level: LogLevel,
    /// 日志信息
    pub message: String,
    /// 相关组件
    pub component: String,
    /// 上下文数据
    pub context: Option<HashMap<String, String>>,
}

/// 资源类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceType {
    /// CPU
    Cpu,
    /// 内存
    Memory,
    /// GPU
    Gpu,
    /// 磁盘
    Disk,
    /// 网络
    Network,
}

/// 资源使用趋势
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceUsageTrend {
    /// 上升
    Increasing,
    /// 下降
    Decreasing,
    /// 稳定
    Stable,
    /// 未知
    Unknown,
} 