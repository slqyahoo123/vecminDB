/// 资源管理模块
/// 
/// 提供向量数据库操作所需的计算资源管理、分配、监控和优化功能。
/// 
/// ## 功能特性
/// 
/// - CPU/GPU/内存资源分配
/// - 向量操作专用资源管理
/// - 资源使用监控
/// - 资源限制和配额
/// - 自动资源调优

use crate::{Result, Error};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

use chrono::{DateTime, Utc};
use uuid::Uuid;

pub mod types;
pub mod manager;
pub mod cache;
pub mod coordinator;
pub mod vector_resource;

// 重新导出主要类型
pub use manager::{ResourceManager, ResourceManagerConfig, MultiLevelCacheConfig as CacheConfig};
pub use cache::{CacheManager as ResourceCache};
pub use coordinator::{ResourceCoordinator, ResourceLimiter, ResourceMonitor};

// 重新导出types中的公共类型
pub use types::{ResourceType, ResourceRequest, ResourceAllocation};
pub use crate::types::TaskPriority;

// 向量专用资源管理
pub use vector_resource::{
    VectorResourceManager,
    VectorOperationResource,
    VectorOperationType,
    VectorResourceAllocation,
};

/// 内部资源类型（用于资源子系统内部）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InternalResourceType {
    /// 内存资源
    Memory,
    /// CPU资源
    CPU,
    /// GPU资源
    GPU,
    /// 存储资源
    Storage,
    /// 网络带宽
    NetworkBandwidth,
    /// 磁盘IO
    DiskIO,
    /// 自定义资源
    Custom(u32),
}

impl std::fmt::Display for InternalResourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InternalResourceType::Memory => write!(f, "Memory"),
            InternalResourceType::CPU => write!(f, "CPU"),
            InternalResourceType::GPU => write!(f, "GPU"), 
            InternalResourceType::Storage => write!(f, "Storage"),
            InternalResourceType::NetworkBandwidth => write!(f, "NetworkBandwidth"),
            InternalResourceType::DiskIO => write!(f, "DiskIO"),
            InternalResourceType::Custom(id) => write!(f, "Custom({})", id),
        }
    }
}

/// 内部资源请求（用于资源子系统内部）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalResourceRequest {
    /// 请求ID
    pub id: Option<String>,
    /// 请求者ID
    pub requester_id: Option<String>,
    /// 资源类型
    pub resource_type: InternalResourceType,
    /// 请求的资源数量
    pub amount: usize,
    /// 优先级
    pub priority: Option<u8>,
    /// 请求创建时间
    pub created_at: Option<DateTime<Utc>>,
    /// 超时时间
    pub timeout: Option<DateTime<Utc>>,
    /// 资源标签
    pub tags: Option<HashMap<String, String>>,
    /// 预留标志
    pub is_preemptible: Option<bool>,
    /// 任务ID
    pub task_id: Option<String>,
    /// 是否可以等待
    pub can_wait: bool,
}

impl InternalResourceRequest {
    /// 创建新的资源请求
    pub fn new(
        resource_type: InternalResourceType,
        amount: usize,
    ) -> Self {
        Self {
            id: Some(Uuid::new_v4().to_string()),
            requester_id: None,
            resource_type,
            amount,
            priority: Some(5), // 默认中等优先级
            created_at: Some(Utc::now()),
            timeout: None,
            tags: Some(HashMap::new()),
            is_preemptible: Some(false),
            task_id: None,
            can_wait: false,
        }
    }

    /// 创建简单的资源请求
    pub fn simple(resource_type: InternalResourceType, amount: usize) -> Self {
        Self {
            id: None,
            requester_id: None,
            resource_type,
            amount,
            priority: None,
            created_at: None,
            timeout: None,
            tags: None,
            is_preemptible: None,
            task_id: None,
            can_wait: false,
        }
    }

    /// 设置请求者ID
    pub fn with_requester(mut self, requester_id: String) -> Self {
        self.requester_id = Some(requester_id);
        self
    }

    /// 设置优先级
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = Some(priority.min(10)); // 最大优先级为10
        self
    }

    /// 设置超时时间
    pub fn with_timeout(mut self, timeout: DateTime<Utc>) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// 添加标签
    pub fn add_tag(mut self, key: String, value: String) -> Self {
        if self.tags.is_none() {
            self.tags = Some(HashMap::new());
        }
        self.tags.as_mut().unwrap().insert(key, value);
        self
    }

    /// 设置可抢占
    pub fn with_preemptible(mut self, preemptible: bool) -> Self {
        self.is_preemptible = Some(preemptible);
        self
    }

    /// 设置任务ID
    pub fn with_task_id(mut self, task_id: String) -> Self {
        self.task_id = Some(task_id);
        self
    }

    /// 设置是否可等待
    pub fn with_wait(mut self, can_wait: bool) -> Self {
        self.can_wait = can_wait;
        self
    }

    /// 检查请求是否过期
    pub fn is_expired(&self) -> bool {
        if let Some(timeout) = self.timeout {
            Utc::now() > timeout
        } else {
            false
        }
    }
}

/// 资源分配状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationStatus {
    /// 待分配
    Pending,
    /// 已分配
    Allocated,
    /// 使用中
    InUse,
    /// 已释放
    Released,
    /// 分配失败
    Failed,
    /// 被抢占
    Preempted,
}

/// 内部资源分配（用于资源子系统内部）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalResourceAllocation {
    /// 分配ID
    pub allocation_id: String,
    /// 对应的请求ID
    pub request_id: Option<String>,
    /// 资源类型
    pub resource_type: InternalResourceType,
    /// 分配的资源数量
    pub amount: usize,
    /// 分配状态
    pub status: Option<AllocationStatus>,
    /// 分配时间
    pub allocated_at: Option<DateTime<Utc>>,
    /// 释放时间
    pub released_at: Option<DateTime<Utc>>,
    /// 资源位置/地址
    pub resource_location: Option<String>,
    /// 使用统计
    pub usage_stats: Option<ResourceUsageStats>,
    /// 过期时间
    pub expires_at: Option<DateTime<Utc>>,
}

impl InternalResourceAllocation {
    /// 创建新的资源分配
    pub fn new(resource_type: InternalResourceType, amount: usize, allocation_id: String) -> Self {
        Self {
            allocation_id,
            request_id: None,
            resource_type,
            amount,
            status: Some(AllocationStatus::Allocated),
            allocated_at: Some(Utc::now()),
            released_at: None,
            resource_location: None,
            usage_stats: Some(ResourceUsageStats::new()),
            expires_at: None,
        }
    }

    /// 从请求创建分配
    pub fn from_request(request_id: String, resource_type: InternalResourceType, amount: usize) -> Self {
        Self {
            allocation_id: Uuid::new_v4().to_string(),
            request_id: Some(request_id),
            resource_type,
            amount,
            status: Some(AllocationStatus::Allocated),
            allocated_at: Some(Utc::now()),
            released_at: None,
            resource_location: None,
            usage_stats: Some(ResourceUsageStats::new()),
            expires_at: None,
        }
    }

    /// 标记为已分配
    pub fn mark_allocated(&mut self, location: Option<String>) {
        self.status = Some(AllocationStatus::Allocated);
        self.allocated_at = Some(Utc::now());
        self.resource_location = location;
    }

    /// 标记为使用中
    pub fn mark_in_use(&mut self) {
        self.status = Some(AllocationStatus::InUse);
        if let Some(ref mut stats) = self.usage_stats {
            stats.mark_start_usage();
        }
    }

    /// 标记为已释放
    pub fn mark_released(&mut self) {
        self.status = Some(AllocationStatus::Released);
        self.released_at = Some(Utc::now());
        if let Some(ref mut stats) = self.usage_stats {
            stats.mark_end_usage();
        }
    }

    /// 标记为失败
    pub fn mark_failed(&mut self) {
        self.status = Some(AllocationStatus::Failed);
    }

    /// 标记为被抢占
    pub fn mark_preempted(&mut self) {
        self.status = Some(AllocationStatus::Preempted);
        self.released_at = Some(Utc::now());
    }

    /// 获取分配持续时间
    pub fn duration(&self) -> Option<chrono::Duration> {
        if let (Some(start), Some(end)) = (self.allocated_at, self.released_at) {
            Some(end.signed_duration_since(start))
        } else {
            None
        }
    }
}

/// 资源使用统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageStats {
    /// 开始使用时间
    pub start_time: Option<DateTime<Utc>>,
    /// 结束使用时间
    pub end_time: Option<DateTime<Utc>>,
    /// 峰值使用量
    pub peak_usage: u64,
    /// 平均使用量
    pub average_usage: f64,
    /// 使用计数器
    pub usage_count: u64,
    /// 错误计数
    pub error_count: u64,
}

impl ResourceUsageStats {
    /// 创建新的使用统计
    pub fn new() -> Self {
        Self {
            start_time: None,
            end_time: None,
            peak_usage: 0,
            average_usage: 0.0,
            usage_count: 0,
            error_count: 0,
        }
    }

    /// 标记开始使用
    pub fn mark_start_usage(&mut self) {
        self.start_time = Some(Utc::now());
    }

    /// 标记结束使用
    pub fn mark_end_usage(&mut self) {
        self.end_time = Some(Utc::now());
    }

    /// 更新使用量
    pub fn update_usage(&mut self, current_usage: u64) {
        self.usage_count += 1;
        if current_usage > self.peak_usage {
            self.peak_usage = current_usage;
        }
        
        // 更新平均使用量
        let total = self.average_usage * (self.usage_count - 1) as f64 + current_usage as f64;
        self.average_usage = total / self.usage_count as f64;
    }

    /// 增加错误计数
    pub fn increment_error(&mut self) {
        self.error_count += 1;
    }
}

/// 资源池
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePool {
    /// 资源类型
    pub resource_type: InternalResourceType,
    /// 总容量
    pub total_capacity: u64,
    /// 已分配容量
    pub allocated_capacity: u64,
    /// 可用容量
    pub available_capacity: u64,
    /// 分配列表
    pub allocations: Vec<InternalResourceAllocation>,
    /// 最后更新时间
    pub last_updated: DateTime<Utc>,
}

impl ResourcePool {
    /// 创建新的资源池
    pub fn new(resource_type: InternalResourceType, capacity: u64) -> Self {
        Self {
            resource_type,
            total_capacity: capacity,
            allocated_capacity: 0,
            available_capacity: capacity,
            allocations: Vec::new(),
            last_updated: Utc::now(),
        }
    }

    /// 尝试分配资源
    pub fn try_allocate(&mut self, request: &InternalResourceRequest) -> Result<InternalResourceAllocation> {
        if request.amount as u64 > self.available_capacity {
            return Err(Error::resource(format!(
                "Insufficient resources: requested {}, available {}",
                request.amount, self.available_capacity
            )));
        }

        let allocation = InternalResourceAllocation::new(
            request.resource_type.clone(),
            request.amount,
            Uuid::new_v4().to_string(),
        );

        self.allocated_capacity += request.amount as u64;
        self.available_capacity -= request.amount as u64;
        self.allocations.push(allocation.clone());
        self.last_updated = Utc::now();

        Ok(allocation)
    }

    /// 释放资源
    pub fn release(&mut self, allocation_id: &str) -> Result<()> {
        if let Some(index) = self.allocations.iter().position(|a| a.allocation_id == allocation_id) {
            let allocation = &self.allocations[index];
            self.allocated_capacity -= allocation.amount as u64;
            self.available_capacity += allocation.amount as u64;
            self.allocations.remove(index);
            self.last_updated = Utc::now();
            Ok(())
        } else {
            Err(Error::NotFound(format!("Allocation not found: {}", allocation_id)))
        }
    }

    /// 计算使用率
    pub fn utilization(&self) -> f64 {
        if self.total_capacity == 0 {
            0.0
        } else {
            self.allocated_capacity as f64 / self.total_capacity as f64
        }
    }

    /// 清理过期的分配
    pub fn cleanup_expired(&mut self) {
        let now = Utc::now();
        self.allocations.retain(|allocation| {
            if let Some(expires_at) = allocation.expires_at {
                if now > expires_at {
                    // 释放过期的资源
                    self.allocated_capacity -= allocation.amount as u64;
                    self.available_capacity += allocation.amount as u64;
                    false
                } else {
                    true
                }
            } else {
                true
            }
        });
        self.last_updated = now;
    }
}

/// 资源协调器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCoordinatorConfig {
    /// 最大待处理请求数
    pub max_pending_requests: usize,
    /// 资源清理间隔（秒）
    pub cleanup_interval_seconds: u64,
    /// 启用抢占
    pub enable_preemption: bool,
    /// 最大分配历史记录数
    pub max_allocation_history: usize,
}

impl Default for ResourceCoordinatorConfig {
    fn default() -> Self {
        Self {
            max_pending_requests: 1000,
            cleanup_interval_seconds: 300, // 5分钟
            enable_preemption: false,
            max_allocation_history: 10000,
        }
    }
}

/// 资源状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStatus {
    /// 资源类型
    pub resource_type: InternalResourceType,
    /// 总容量
    pub total_capacity: u64,
    /// 已分配容量
    pub allocated_capacity: u64,
    /// 可用容量
    pub available_capacity: u64,
    /// 使用率
    pub utilization: f64,
    /// 活跃分配数
    pub active_allocations: usize,
    /// 最后更新时间
    pub last_updated: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_request_creation() {
        let request = InternalResourceRequest::new(InternalResourceType::CPU, 4);
        assert_eq!(request.resource_type, InternalResourceType::CPU);
        assert_eq!(request.amount, 4);
        assert!(request.id.is_some());
    }

    #[test]
    fn test_resource_pool_allocation() {
        let mut pool = ResourcePool::new(InternalResourceType::Memory, 1024);
        let request = InternalResourceRequest::new(InternalResourceType::Memory, 256);
        
        let allocation = pool.try_allocate(&request).unwrap();
        assert_eq!(allocation.amount, 256);
        assert_eq!(pool.available_capacity, 768);
    }

    #[test]
    fn test_resource_coordinator() {
        let config = ResourceCoordinatorConfig::default();
        assert_eq!(config.max_pending_requests, 1000);
        assert_eq!(config.cleanup_interval_seconds, 300);
    }
} 