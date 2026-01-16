use crate::event::{Event, EventType, EventSystem};
use crate::event::enhanced::{EnhancedEventSystem, EventPriority};
use crate::status::{StatusType, StatusTrackerTrait};
use uuid::Uuid;
use std::sync::Arc;
use std::time::Duration;
use serde::{Serialize, Deserialize};

/// 事件监听器类型
pub type EventListener = Box<dyn Fn(&Event) -> crate::error::Result<()> + Send + Sync + 'static>;

/// 模块协调器，负责处理特定模块的事件和状态协调
pub struct ModuleCoordinator {
    module_name: String,
    event_system: Arc<dyn EventSystem>,
    enhanced_event_system: Arc<EnhancedEventSystem>,
    status_tracker: Arc<dyn StatusTrackerTrait>,
}

impl ModuleCoordinator {
    pub fn new(
        module_name: &str,
        event_system: Arc<dyn EventSystem>,
        enhanced_event_system: Arc<EnhancedEventSystem>,
        status_tracker: Arc<dyn StatusTrackerTrait>,
    ) -> Self {
        Self {
            module_name: module_name.to_string(),
            event_system,
            enhanced_event_system,
            status_tracker,
        }
    }
    
    /// 注册模块对特定事件类型的订阅
    pub fn register_event_subscription(&self, event_types: Vec<EventType>) -> crate::error::Result<()> {
        for event_type in event_types {
            let module_name = self.module_name.clone();
            let enhanced_system = self.enhanced_event_system.clone();
            
            self.event_system.subscribe(event_type, Arc::new(move |event| {
                // 将事件转换为增强事件并发布
                let enhanced_event = event.clone().into();
                enhanced_system.publish_enhanced(enhanced_event)
            }))?;
        }
        Ok(())
    }
    
    /// 发布模块事件
    pub fn publish_event(&self, event: Event) -> crate::error::Result<()> {
        // 发布到增强事件系统
        self.enhanced_event_system.publish_as_enhanced(event)
    }
    
    /// 发布增强模块事件
    pub fn publish_enhanced_event(&self, event: Event, tags: Vec<&str>, priority: EventPriority) -> crate::error::Result<()> {
        use crate::event::enhanced::EnhancedEvent;
        
        let mut enhanced_event: EnhancedEvent = event.into();
        
        // 添加标签
        for tag in tags {
            enhanced_event = enhanced_event.with_tag(tag);
        }
        
        // 设置优先级
        enhanced_event = enhanced_event.with_priority(priority);
        
        // 设置源模块
        enhanced_event.routing.source_module = self.module_name.clone();
        
        self.enhanced_event_system.publish_enhanced(enhanced_event)
    }
    
    /// 更新任务状态
    pub fn update_task_status(&self, task_id: &Uuid, status: StatusType, message: &str) -> crate::error::Result<()> {
        use crate::status::StatusEvent;
        use chrono::Utc;
        let event = StatusEvent {
            task_id: *task_id,
            status,
            progress: None,
            message: message.to_string(),
            updated_at: Utc::now(),
            metadata: std::collections::HashMap::new(),
        };
        tokio::runtime::Handle::current().block_on(async {
            self.status_tracker.update_status(event).await
        })
    }
    
    /// 创建任务并记录状态
    pub fn create_task(&self, task_type: &str) -> crate::error::Result<Uuid> {
        use crate::status::StatusEvent;
        use crate::status::StatusType;
        use chrono::Utc;
        let task_id = Uuid::new_v4();
        let event = StatusEvent {
            task_id,
            status: StatusType::Pending,
            progress: None,
            message: format!("Task created: {}", task_type),
            updated_at: Utc::now(),
            metadata: std::collections::HashMap::new(),
        };
        tokio::runtime::Handle::current().block_on(async {
            self.status_tracker.update_status(event).await?;
            Ok(task_id)
        })
    }
}



/// 统一组件接口标准
pub trait Component: Send + Sync {
    /// 获取组件名称
    fn name(&self) -> &str;
    /// 获取组件类型
    fn component_type(&self) -> ComponentType;
    /// 获取组件状态
    fn status(&self) -> ComponentStatus;
    /// 启动组件
    fn start(&mut self) -> crate::error::Result<()> {
        Ok(())
    }
    /// 停止组件
    fn stop(&mut self) -> crate::error::Result<()> {
        Ok(())
    }
    /// 获取组件运行时间
    fn get_uptime(&self) -> Duration {
        // 这里应该实现获取组件运行时间的逻辑
        // 暂时返回一个默认值
        Duration::from_secs(0)
    }
}

/// 组件类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentType {
    Vector,
    Data,
    Storage,
    Algorithm,
    Training,
    Service,
}

/// 组件状态枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentStatus {
    Initializing,
    Ready,
    Running,
    Error,
    Shutdown,
}

/// 异步适配器 - 为同步API提供异步接口
pub struct AsyncAdapter<T> {
    inner: Arc<T>,
}

impl<T> AsyncAdapter<T> {
    /// 创建新的异步适配器
    pub fn new(inner: Arc<T>) -> Self {
        Self { inner }
    }
    
    /// 获取内部组件引用
    pub fn inner(&self) -> Arc<T> {
        self.inner.clone()
    }
}

/// 任务优先级
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl Default for TaskPriority {
    fn default() -> Self {
        TaskPriority::Normal
    }
}

impl From<u8> for TaskPriority {
    fn from(value: u8) -> Self {
        match value {
            0 => TaskPriority::Low,
            1 => TaskPriority::Normal,
            2 => TaskPriority::High,
            3..=255 => TaskPriority::Critical,
        }
    }
}

impl From<TaskPriority> for u8 {
    fn from(priority: TaskPriority) -> Self {
        match priority {
            TaskPriority::Low => 0,
            TaskPriority::Normal => 1,
            TaskPriority::High => 2,
            TaskPriority::Critical => 3,
        }
    }
}

/// 模型ID类型
pub type ModelId = String;

/// 算法ID类型
pub type AlgorithmId = String;

/// 任务ID类型
pub type TaskId = String;

/// 执行ID类型
pub type ExecutionId = String;

/// 会话ID类型
pub type SessionId = String;

/// 用户ID类型别名
pub type UserId = String;

// 重导出资源相关类型
pub use crate::resource::types::{ResourceType, ResourceRequest, ResourceAllocation};

 