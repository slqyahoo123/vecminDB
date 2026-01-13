//! 事件系统模块
//! 
//! 提供完整的事件发布订阅功能，包括：
//! - 基础事件系统
//! - 增强事件系统
//! - 分布式事件系统
//! - 统一事件系统（推荐使用）
//! - 旧版事件系统兼容

use std::fmt;
use std::collections::HashMap;
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use crate::Result;

/// 事件类型枚举
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    // 数据相关事件
    DataUploaded,
    DataProcessed,
    DataDeleted,
    DataValidated,
    DataTransformed,
    DataCacheHit,
    DataCacheMiss,
    DataCacheUpdate,
    
    // 通用缓存事件
    CacheHit,
    CacheMiss,
    CacheEviction,
    CacheRemoval,
    CacheClear,
    CacheResize,
    CacheWarmup,
    CacheWrite,
    CacheDelete,
    CacheCleanup,

    // 模型相关事件
    ModelCreated,
    ModelUpdated,
    ModelDeleted,
    ModelRegistered,
    ModelRemoved,
    ModelStatusChanged,
    ModelValidated,
    ModelDeployed,

    // 训练相关事件
    TrainingTaskCreated,
    TrainingTaskStarted,
    TrainingTaskStarting,
    TrainingTaskStatusChanged,
    TrainingTaskProgressUpdated,
    TrainingTaskCompleted,
    TrainingTaskFailed,
    TrainingTaskCanceled,
    TrainingTaskPaused,
    TrainingTaskResumed,
    TrainingEpochCompleted,
    TrainingMetricsUpdated,
    TrainingStarted,
    TrainingCompleted,
    TrainingProgress,
    TrainingManagerCreated,
    
    // 分布式训练相关事件
    DistributedEnvironmentInitialized,
    DistributedEnvironmentShutdown,
    DistributedTrainingTaskCreated,
    DistributedTrainingStarted,
    DistributedTrainingCompleted,
    DistributedTrainingFailed,
    DistributedTrainingStopped,
    DistributedTrainingJoined, // 添加缺失的事件类型
    JoiningDistributedTraining,
    JoinedDistributedTraining,
    
    // 检查点相关事件
    CheckpointSaved,
    CheckpointLoaded,
    CheckpointDeleted,
    
    // 算法相关事件
    AlgorithmCreated,
    AlgorithmUpdated,
    AlgorithmDeleted,
    AlgorithmTaskCreated,
    AlgorithmTaskStarted,
    AlgorithmTaskStatusChanged,
    AlgorithmTaskProgressUpdated,
    AlgorithmTaskCompleted,
    AlgorithmTaskFailed,
    AlgorithmTaskCanceled,
    AlgorithmValidated,
    
    // 连接相关事件
    ConnectionCreated,
    ConnectionUpdated,
    ConnectionDeleted,
    ConnectionEstablished,
    ConnectionLost,

    // 系统事件
    SystemStarted,
    SystemStopped,
    SystemStatusChanged,
    SystemHealthCheck,
    SystemResourceUsage,
    
    // 错误事件
    ErrorOccurred,
    WarningIssued,
    CriticalError,

    // 用户事件
    UserLoggedIn,
    UserLoggedOut,
    UserActionPerformed,
    
    // 自定义事件
    Custom(String),
}

impl fmt::Display for EventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            EventType::DataUploaded => "data_uploaded",
            EventType::DataProcessed => "data_processed",
            EventType::DataDeleted => "data_deleted",
            EventType::DataValidated => "data_validated",
            EventType::DataTransformed => "data_transformed",
            EventType::DataCacheHit => "data_cache_hit",
            EventType::DataCacheMiss => "data_cache_miss",
            EventType::DataCacheUpdate => "data_cache_update",
            EventType::CacheHit => "cache_hit",
            EventType::CacheMiss => "cache_miss",
            EventType::CacheEviction => "cache_eviction",
            EventType::CacheRemoval => "cache_removal",
            EventType::CacheClear => "cache_clear",
            EventType::CacheResize => "cache_resize",
            EventType::CacheWarmup => "cache_warmup",
            EventType::CacheWrite => "cache_write",
            EventType::CacheDelete => "cache_delete",
            EventType::CacheCleanup => "cache_cleanup",
            EventType::ModelCreated => "model_created",
            EventType::ModelUpdated => "model_updated",
            EventType::ModelDeleted => "model_deleted",
            EventType::ModelRegistered => "model_registered",
            EventType::ModelRemoved => "model_removed",
            EventType::ModelStatusChanged => "model_status_changed",
            EventType::ModelValidated => "model_validated",
            EventType::ModelDeployed => "model_deployed",
            EventType::TrainingTaskCreated => "training_task_created",
            EventType::TrainingTaskStarted => "training_task_started",
            EventType::TrainingTaskStarting => "training_task_starting",
            EventType::TrainingTaskStatusChanged => "training_task_status_changed",
            EventType::TrainingTaskProgressUpdated => "training_task_progress_updated",
            EventType::TrainingTaskCompleted => "training_task_completed",
            EventType::TrainingTaskFailed => "training_task_failed",
            EventType::TrainingTaskCanceled => "training_task_canceled",
            EventType::TrainingTaskPaused => "training_task_paused",
            EventType::TrainingTaskResumed => "training_task_resumed",
            EventType::TrainingEpochCompleted => "training_epoch_completed",
            EventType::TrainingMetricsUpdated => "training_metrics_updated",
            EventType::TrainingStarted => "training_started",
            EventType::TrainingCompleted => "training_completed",
            EventType::TrainingProgress => "training_progress",
            EventType::TrainingManagerCreated => "training_manager_created",
            EventType::DistributedEnvironmentInitialized => "distributed_environment_initialized",
            EventType::DistributedEnvironmentShutdown => "distributed_environment_shutdown",
            EventType::DistributedTrainingTaskCreated => "distributed_training_task_created",
            EventType::DistributedTrainingStarted => "distributed_training_started",
            EventType::DistributedTrainingCompleted => "distributed_training_completed",
            EventType::DistributedTrainingFailed => "distributed_training_failed",
            EventType::DistributedTrainingStopped => "distributed_training_stopped",
            EventType::DistributedTrainingJoined => "distributed_training_joined",
            EventType::JoiningDistributedTraining => "joining_distributed_training",
            EventType::JoinedDistributedTraining => "joined_distributed_training",
            EventType::CheckpointSaved => "checkpoint_saved",
            EventType::CheckpointLoaded => "checkpoint_loaded",
            EventType::CheckpointDeleted => "checkpoint_deleted",
            EventType::AlgorithmCreated => "algorithm_created",
            EventType::AlgorithmUpdated => "algorithm_updated",
            EventType::AlgorithmDeleted => "algorithm_deleted",
            EventType::AlgorithmTaskCreated => "algorithm_task_created",
            EventType::AlgorithmTaskStarted => "algorithm_task_started",
            EventType::AlgorithmTaskStatusChanged => "algorithm_task_status_changed",
            EventType::AlgorithmTaskProgressUpdated => "algorithm_task_progress_updated",
            EventType::AlgorithmTaskCompleted => "algorithm_task_completed",
            EventType::AlgorithmTaskFailed => "algorithm_task_failed",
            EventType::AlgorithmTaskCanceled => "algorithm_task_canceled",
            EventType::AlgorithmValidated => "algorithm_validated",
            EventType::ConnectionCreated => "connection_created",
            EventType::ConnectionUpdated => "connection_updated",
            EventType::ConnectionDeleted => "connection_deleted",
            EventType::ConnectionEstablished => "connection_established",
            EventType::ConnectionLost => "connection_lost",
            EventType::SystemStarted => "system_started",
            EventType::SystemStopped => "system_stopped",
            EventType::SystemStatusChanged => "system_status_changed",
            EventType::SystemHealthCheck => "system_health_check",
            EventType::SystemResourceUsage => "system_resource_usage",
            EventType::ErrorOccurred => "error_occurred",
            EventType::WarningIssued => "warning_issued",
            EventType::CriticalError => "critical_error",
            EventType::UserLoggedIn => "user_logged_in",
            EventType::UserLoggedOut => "user_logged_out",
            EventType::UserActionPerformed => "user_action_performed",
            EventType::Custom(name) => name,
        };
        write!(f, "{}", name)
    }
}

/// 事件结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// 事件唯一标识
    pub id: String,
    /// 事件类型
    pub event_type: EventType,
    /// 事件源
    pub source: String,
    /// 事件时间戳（Unix时间戳）
    pub timestamp: u64,
    /// 事件数据
    pub data: HashMap<String, String>,
    /// 事件元数据
    pub metadata: Option<HashMap<String, String>>,
}

impl Event {
    /// 创建新事件
    pub fn new(event_type: EventType, source: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            event_type,
            source: source.to_string(),
            timestamp: chrono::Utc::now().timestamp() as u64,
            data: HashMap::new(),
            metadata: None,
        }
    }
    
    /// 添加事件数据
    pub fn with_data<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.data.insert(key.into(), value.into());
        self
    }
    
    /// 添加多个事件数据
    pub fn with_data_map(mut self, data: HashMap<String, String>) -> Self {
        self.data.extend(data);
        self
    }
    
    /// 添加元数据
    pub fn with_metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        if self.metadata.is_none() {
            self.metadata = Some(HashMap::new());
        }
        self.metadata.as_mut().unwrap().insert(key.into(), value.into());
        self
        }
        
    /// 设置时间戳
    pub fn with_timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// 获取事件年龄（秒）
    pub fn age_seconds(&self) -> u64 {
        let now = chrono::Utc::now().timestamp() as u64;
        now.saturating_sub(self.timestamp)
    }

    /// 检查事件是否过期
    pub fn is_expired(&self, max_age_seconds: u64) -> bool {
        self.age_seconds() > max_age_seconds
    }
}

/// 事件回调trait
pub trait EventCallback: Send + Sync {
    /// 处理事件
    fn on_event(&self, event: &Event) -> Result<()>;
}

/// 事件过滤器trait
pub trait EventFilter: Send + Sync {
    /// 检查事件是否通过过滤器
    fn matches(&self, event: &Event) -> bool;
}

/// 基础事件系统trait
pub trait EventSystem: Send + Sync {
    /// 发布事件
    fn publish(&self, event: Event) -> Result<()>;
    
    /// 订阅特定类型的事件
    fn subscribe(&self, event_type: EventType, callback: Arc<dyn EventCallback>) -> Result<String>;
    
    /// 订阅所有事件
    fn subscribe_all(&self, callback: Arc<dyn EventCallback>) -> Result<String>;
    
    /// 取消订阅
    fn unsubscribe(&self, subscription_id: &str) -> Result<()>;
    
    /// 获取待处理的事件
    fn get_pending_events(&self) -> Result<Vec<Event>>;
    
    /// 启动事件系统
    fn start(&self) -> Result<()>;
    
    /// 停止事件系统
    fn stop(&self) -> Result<()>;
}

/// 简单的事件过滤器实现
pub struct SimpleEventFilter {
    event_types: Vec<EventType>,
    source_patterns: Vec<String>,
}

impl SimpleEventFilter {
    pub fn new() -> Self {
        Self {
            event_types: Vec::new(),
            source_patterns: Vec::new(),
        }
    }

    pub fn with_event_types(mut self, event_types: Vec<EventType>) -> Self {
        self.event_types = event_types;
        self
    }

    pub fn with_source_patterns(mut self, patterns: Vec<String>) -> Self {
        self.source_patterns = patterns;
        self
    }
}

impl EventFilter for SimpleEventFilter {
    fn matches(&self, event: &Event) -> bool {
        // 检查事件类型
        if !self.event_types.is_empty() && !self.event_types.contains(&event.event_type) {
            return false;
        }

        // 检查源模式
        if !self.source_patterns.is_empty() {
            let matches_source = self.source_patterns.iter().any(|pattern| {
                event.source.contains(pattern)
            });
            if !matches_source {
                return false;
            }
        }

        true
    }
}

/// 函数式事件回调
pub struct FunctionCallback<F>
where
    F: Fn(&Event) -> Result<()> + Send + Sync,
{
    func: F,
}

impl<F> FunctionCallback<F>
where
    F: Fn(&Event) -> Result<()> + Send + Sync,
{
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<F> EventCallback for FunctionCallback<F>
where
    F: Fn(&Event) -> Result<()> + Send + Sync,
{
    fn on_event(&self, event: &Event) -> Result<()> {
        (self.func)(event)
    }
}

/// 创建函数式回调的便捷函数
pub fn callback<F>(func: F) -> Arc<dyn EventCallback>
where
    F: Fn(&Event) -> Result<()> + Send + Sync + 'static,
{
    Arc::new(FunctionCallback::new(func))
}

/// 创建默认的事件系统
/// 
/// 推荐使用统一事件系统，它整合了所有功能
pub fn create_default_event_system() -> Result<Arc<dyn EventSystem>> {
    let unified_system = unified::create_unified_event_system()?;
    Ok(unified_system as Arc<dyn EventSystem>)
}

/// 创建内存事件系统
pub fn create_memory_event_system() -> Result<Arc<dyn EventSystem>> {
    let system = MemoryEventSystem::new(1000);
    Ok(Arc::new(system))
}

/// 创建增强事件系统
pub fn create_enhanced_event_system(source_name: &str, queue_capacity: usize) -> Result<Arc<dyn EventSystem>> {
    use std::time::Duration;
    let system = EnhancedEventSystem::new(source_name, queue_capacity, Duration::from_secs(30));
    Ok(Arc::new(system))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_event_creation() {
        let event = Event::new(EventType::DataUploaded, "test_source")
            .with_data("key1", "value1")
            .with_data("key2", "value2")
            .with_metadata("meta1", "metavalue1");

        assert_eq!(event.event_type, EventType::DataUploaded);
        assert_eq!(event.source, "test_source");
        assert_eq!(event.data.get("key1"), Some(&"value1".to_string()));
        assert_eq!(event.data.get("key2"), Some(&"value2".to_string()));
        assert!(event.metadata.is_some());
        assert_eq!(
            event.metadata.as_ref().unwrap().get("meta1"),
            Some(&"metavalue1".to_string())
        );
    }

    #[test]
    fn test_event_filter() {
        let filter = SimpleEventFilter::new()
            .with_event_types(vec![EventType::DataUploaded, EventType::ModelCreated])
            .with_source_patterns(vec!["test".to_string()]);

        let event1 = Event::new(EventType::DataUploaded, "test_source");
        let event2 = Event::new(EventType::DataProcessed, "test_source");
        let event3 = Event::new(EventType::DataUploaded, "other_source");

        assert!(filter.matches(&event1));
        assert!(!filter.matches(&event2));
        assert!(!filter.matches(&event3));
    }

    #[test]
    fn test_function_callback() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let callback = callback(move |_event: &Event| -> Result<()> {
            counter_clone.fetch_add(1, Ordering::Relaxed);
            Ok(())
        });

        let event = Event::new(EventType::DataUploaded, "test");
        callback.on_event(&event).unwrap();

        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }
}

// 导出子模块
pub mod memory;
pub mod file;
pub mod distributed;
pub mod enhanced; // 添加新的增强事件系统模块
pub mod unified; // 添加统一事件系统模块

// 导出关键类型
pub use memory::MemoryEventSystem;
pub use file::FileEventSystem;
pub use distributed::DistributedEventSystem;
pub use enhanced::{EnhancedEventSystem, EnhancedEvent, EventPriority, EventProcessingState};

/// 事件系统枚举
#[derive(Clone)]
pub enum EventSystemEnum {
    Memory(memory::MemoryEventSystem),
    File(file::FileEventSystem),
    Distributed(distributed::DistributedEventSystem),
    Enhanced(enhanced::EnhancedEventSystem),
    // 注释掉不存在的DummyEventSystem
    // DummyEvent(crate::algorithm::manager::core::DummyEventSystem),
}

impl EventSystem for EventSystemEnum {
    fn publish(&self, event: Event) -> Result<()> {
        match self {
            Self::Memory(system) => system.publish(event),
            Self::File(system) => system.publish(event),
            Self::Distributed(system) => system.publish(event),
            Self::Enhanced(system) => system.publish(event),
            // Self::DummyEvent(system) => system.publish(event),
        }
    }
    
    fn start(&self) -> Result<()> {
        match self {
            Self::Memory(system) => system.start(),
            Self::File(system) => system.start(),
            Self::Distributed(system) => system.start(),
            Self::Enhanced(system) => system.start(),
            // Self::DummyEvent(system) => system.start(),
        }
    }
    
    fn stop(&self) -> Result<()> {
        match self {
            Self::Memory(system) => system.stop(),
            Self::File(system) => system.stop(),
            Self::Distributed(system) => system.stop(),
            Self::Enhanced(system) => system.stop(),
            // Self::DummyEvent(system) => system.stop(),
        }
    }
    
    fn subscribe(&self, event_type: EventType, callback: Arc<dyn EventCallback>) -> Result<String> {
        match self {
            Self::Memory(system) => system.subscribe(event_type, callback),
            Self::File(system) => system.subscribe(event_type, callback),
            Self::Distributed(system) => system.subscribe(event_type, callback),
            Self::Enhanced(system) => system.subscribe(event_type, callback),
            // Self::DummyEvent(system) => system.subscribe(event_type, callback),
        }
    }
    
    fn subscribe_all(&self, callback: Arc<dyn EventCallback>) -> Result<String> {
        match self {
            Self::Memory(system) => system.subscribe_all(callback),
            Self::File(system) => system.subscribe_all(callback),
            Self::Distributed(system) => system.subscribe_all(callback),
            Self::Enhanced(system) => system.subscribe_all(callback),
            // Self::DummyEvent(system) => system.subscribe_all(callback),
        }
    }
    
    fn unsubscribe(&self, subscription_id: &str) -> Result<()> {
        match self {
            Self::Memory(system) => system.unsubscribe(subscription_id),
            Self::File(system) => system.unsubscribe(subscription_id),
            Self::Distributed(system) => system.unsubscribe(subscription_id),
            Self::Enhanced(system) => system.unsubscribe(subscription_id),
            // Self::DummyEvent(system) => system.unsubscribe(subscription_id),
        }
    }
    
    fn get_pending_events(&self) -> Result<Vec<Event>> {
        match self {
            Self::Memory(system) => system.get_pending_events(),
            Self::File(system) => system.get_pending_events(),
            Self::Distributed(system) => system.get_pending_events(),
            Self::Enhanced(system) => system.get_pending_events(),
            // Self::DummyEvent(system) => system.get_pending_events(),
        }
    }
} 