use std::collections::HashMap;
use std::time::Duration;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

/// 资源类型枚举
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// CPU资源
    CPU,
    /// 内存资源
    Memory,
    /// 存储资源
    Storage,
    /// 网络带宽
    Network,
    /// GPU资源
    GPU,
    /// 算法执行
    AlgorithmExecution,
    /// 自定义资源
    Custom(String),
}

impl Default for ResourceType {
    fn default() -> Self {
        ResourceType::CPU
    }
}

impl std::fmt::Display for ResourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResourceType::CPU => write!(f, "CPU"),
            ResourceType::Memory => write!(f, "Memory"),
            ResourceType::Storage => write!(f, "Storage"),
            ResourceType::Network => write!(f, "Network"),
            ResourceType::GPU => write!(f, "GPU"),
            ResourceType::AlgorithmExecution => write!(f, "AlgorithmExecution"),
            ResourceType::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// 资源分配请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequest {
    /// 请求ID
    pub id: String,
    /// 请求者ID
    pub requester_id: String,
    /// 资源类型
    pub resource_type: ResourceType,
    /// 请求数量
    pub amount: usize,
    /// 优先级
    pub priority: crate::types::TaskPriority,
    /// 是否可以等待
    pub can_wait: bool,
    /// 任务ID
    pub task_id: Option<String>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 超时时间
    pub timeout: Option<Duration>,
    /// 标签
    pub tags: Vec<String>,
    /// 是否可抢占
    pub is_preemptible: bool,
}

impl ResourceRequest {
    /// 创建新的资源请求
    pub fn new(
        requester_id: String,
        resource_type: ResourceType,
        amount: usize,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            requester_id,
            resource_type,
            amount,
            priority: crate::types::TaskPriority::Normal,
            can_wait: false,
            task_id: None,
            created_at: Utc::now(),
            timeout: None,
            tags: Vec::new(),
            is_preemptible: false,
        }
    }

    /// 设置优先级
    pub fn with_priority(mut self, priority: crate::types::TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    /// 设置是否可等待
    pub fn with_wait(mut self, can_wait: bool) -> Self {
        self.can_wait = can_wait;
        self
    }

    /// 设置任务ID
    pub fn with_task_id(mut self, task_id: String) -> Self {
        self.task_id = Some(task_id);
        self
    }

    /// 设置超时时间
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// 添加标签
    pub fn add_tag(mut self, tag: String) -> Self {
        self.tags.push(tag);
        self
    }

    /// 设置可抢占
    pub fn with_preemptible(mut self, is_preemptible: bool) -> Self {
        self.is_preemptible = is_preemptible;
        self
    }
}

/// 资源分配结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// 分配ID
    pub allocation_id: String,
    /// 对应的请求ID
    pub request_id: String,
    /// 资源类型
    pub resource_type: ResourceType,
    /// 分配数量
    pub amount: usize,
    /// 分配时间
    pub allocated_at: DateTime<Utc>,
    /// 过期时间
    pub expires_at: Option<DateTime<Utc>>,
    /// 资源位置
    pub resource_location: Option<String>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

// 为训练引擎等模块提供统一的资源/网络/磁盘使用类型桥接（避免跨模块路径分裂）
// 分析与意图：部分代码以 `crate::resource::types::ResourceUsage` 作为返回类型，
// 而权威定义位于 `training::engine::core::traits`。为保持最小侵入与路径稳定，
// 在资源类型模块下重导出这些类型，统一对外可见路径。
// 注意：training模块已移除，使用本地定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu: f64,
    pub memory: usize,
    pub gpu: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkUsage {
    pub bytes_sent: usize,
    pub bytes_received: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskUsage {
    pub bytes_read: usize,
    pub bytes_written: usize,
}

impl ResourceAllocation {
    /// 创建新的资源分配
    pub fn new(
        request_id: String,
        resource_type: ResourceType,
        amount: usize,
    ) -> Self {
        Self {
            allocation_id: uuid::Uuid::new_v4().to_string(),
            request_id,
            resource_type,
            amount,
            allocated_at: Utc::now(),
            expires_at: None,
            resource_location: None,
            metadata: HashMap::new(),
        }
    }

    /// 设置过期时间
    pub fn with_expiry(mut self, expires_at: DateTime<Utc>) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// 设置资源位置
    pub fn with_location(mut self, location: String) -> Self {
        self.resource_location = Some(location);
        self
    }

    /// 添加元数据
    pub fn add_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// 检查是否过期
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            Utc::now() > expires_at
        } else {
            false
        }
    }

    /// 获取分配持续时间
    pub fn duration(&self) -> chrono::Duration {
        Utc::now().signed_duration_since(self.allocated_at)
    }
} 