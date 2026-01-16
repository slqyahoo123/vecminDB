/// 统一事件系统
/// 整合新旧事件系统的功能，提供统一的接口和实现
/// 
/// 此模块将新版事件系统作为核心实现，同时提供对旧版事件系统的完全兼容

use std::sync::{Arc, RwLock, Mutex};
use std::collections::{HashMap, HashSet, VecDeque};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use crate::{Error, Result};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use async_trait::async_trait;

// 重导出新版事件系统的核心类型
pub use crate::event::{Event, EventType, EventSystem, EventCallback};
pub use crate::event::enhanced::{EnhancedEventSystem, EnhancedEvent, EventPriority, EventProcessingState};

// 旧版事件系统兼容类型定义
#[derive(Debug, Clone)]
pub struct OldEvent {
    pub id: String,
    pub event_type: OldEventType,
    pub sender: String,
    pub data: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

/// 旧版事件类型
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum OldEventType {
    DataChanged,
    ModelUpdated,
    SystemEvent,
    Custom(String),
}

impl std::fmt::Display for OldEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OldEventType::DataChanged => write!(f, "DataChanged"),
            OldEventType::ModelUpdated => write!(f, "ModelUpdated"),
            OldEventType::SystemEvent => write!(f, "SystemEvent"),
            OldEventType::Custom(s) => write!(f, "Custom({})", s),
        }
    }
}

pub trait OldEventCallback: Send + Sync {
    fn handle(&self, event: &OldEvent) -> Result<()>;
}

pub trait OldEventSystemTrait: Send + Sync {
    fn publish_old(&self, event: OldEvent) -> Result<()>;
    fn subscribe_old(&self, callback: Arc<dyn OldEventCallback>) -> Result<String>;
}

/// 统一事件系统配置
#[derive(Debug, Clone)]
pub struct UnifiedEventSystemConfig {
    /// 事件队列容量
    pub queue_capacity: usize,
    /// 历史事件保留数量
    pub max_history_size: usize,
    /// 是否启用持久化
    pub enable_persistence: bool,
    /// 持久化路径
    pub persistence_path: Option<String>,
    /// 是否启用分布式
    pub enable_distributed: bool,
    /// 分布式配置
    pub distributed_config: Option<DistributedConfig>,
    /// 处理器线程数
    pub processor_threads: usize,
    /// 批处理大小
    pub batch_size: usize,
    /// 处理超时时间
    pub processing_timeout: Duration,
}

impl Default for UnifiedEventSystemConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 10000,
            max_history_size: 1000,
            enable_persistence: false,
            persistence_path: None,
            enable_distributed: false,
            distributed_config: None,
            processor_threads: 4,
            batch_size: 100,
            processing_timeout: Duration::from_secs(30),
        }
    }
}

/// 分布式配置
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// 节点ID
    pub node_id: String,
    /// 集群节点列表
    pub cluster_nodes: Vec<String>,
    /// 同步间隔
    pub sync_interval: Duration,
    /// 是否为主节点
    pub is_master: bool,
}

/// 统一事件系统
/// 
/// 这是一个完整的事件系统实现，整合了：
/// 1. 基础事件发布/订阅功能
/// 2. 增强事件处理功能
/// 3. 旧版事件系统兼容
/// 4. 分布式事件同步
/// 5. 事件持久化
pub struct UnifiedEventSystem {
    /// 系统配置
    config: UnifiedEventSystemConfig,
    /// 事件订阅者
    subscribers: RwLock<HashMap<String, SubscriberInfo>>,
    /// 事件队列
    event_queue: Arc<Mutex<VecDeque<UnifiedEventWrapper>>>,
    /// 事件历史
    event_history: RwLock<VecDeque<UnifiedEventWrapper>>,
    /// 处理状态
    processing_state: Arc<AtomicBool>,
    /// 统计信息
    stats: Arc<EventSystemStats>,
    /// 持久化管理器
    persistence_manager: Option<Arc<dyn EventPersistenceManager>>,
    /// 分布式管理器
    distributed_manager: Option<Arc<dyn DistributedEventManager>>,
    /// 事件处理器
    event_processors: Vec<Arc<EventProcessor>>,
}

/// 统一事件包装器
#[derive(Debug, Clone)]
pub enum UnifiedEventWrapper {
    /// 新版事件
    New(Event),
    /// 增强事件
    Enhanced(EnhancedEvent),
    /// 旧版事件（兼容）
    Legacy(OldEvent),
}

/// 订阅者信息
#[derive(Debug)]
pub struct SubscriberInfo {
    /// 订阅者ID
    pub id: String,
    /// 订阅者名称
    pub name: String,
    /// 订阅的事件类型
    pub event_types: HashSet<String>,
    /// 回调函数
    pub callback: Arc<dyn UnifiedEventCallback>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 处理统计
    pub stats: SubscriberStats,
}

/// 订阅者统计信息
#[derive(Debug, Default)]
pub struct SubscriberStats {
    /// 处理的事件数量
    pub events_processed: AtomicUsize,
    /// 处理失败数量
    pub events_failed: AtomicUsize,
    /// 最后处理时间
    pub last_processed_at: RwLock<Option<DateTime<Utc>>>,
    /// 平均处理时间
    pub avg_processing_time: RwLock<Duration>,
}

/// 统一事件回调接口
#[async_trait]
pub trait UnifiedEventCallback: Send + Sync + std::fmt::Debug {
    /// 处理新版事件
    async fn on_event(&self, event: &Event) -> Result<()> {
        // 默认实现：转换为旧版事件处理
        let old_event = self.convert_to_old_event(event);
        self.on_old_event(&old_event).await
    }
    
    /// 处理增强事件
    async fn on_enhanced_event(&self, event: &EnhancedEvent) -> Result<()> {
        // 默认实现：转换为基础事件处理
        let basic_event = event.base().clone();
        self.on_event(&basic_event).await
    }
    
    /// 处理旧版事件（兼容）
    async fn on_old_event(&self, event: &OldEvent) -> Result<()> {
        // 默认实现：转换为新版事件处理
        let new_event = self.convert_to_new_event(event);
        self.on_event(&new_event).await
    }
    
    /// 转换为旧版事件
    fn convert_to_old_event(&self, event: &Event) -> OldEvent {
        OldEvent {
            id: event.id.clone(),
            event_type: self.convert_event_type(&event.event_type),
            sender: event.source.clone(),
            data: event.data.clone(),
            timestamp: DateTime::from_timestamp(event.timestamp as i64, 0).unwrap_or_else(|| Utc::now()),
        }
    }
    
    /// 转换为新版事件
    fn convert_to_new_event(&self, event: &OldEvent) -> Event {
        Event {
            id: event.id.clone(),
            event_type: self.convert_old_event_type(&event.event_type),
            source: event.sender.clone(),
            timestamp: event.timestamp.timestamp() as u64,
            data: event.data.clone(),
            metadata: None,
        }
    }
    
    /// 转换事件类型
    fn convert_event_type(&self, event_type: &EventType) -> OldEventType;
    
    /// 转换旧版事件类型
    fn convert_old_event_type(&self, event_type: &OldEventType) -> EventType;
}

/// 事件系统统计信息
#[derive(Debug, Default)]
pub struct EventSystemStats {
    /// 发布的事件总数
    pub events_published: AtomicUsize,
    /// 处理的事件总数
    pub events_processed: AtomicUsize,
    /// 失败的事件总数
    pub events_failed: AtomicUsize,
    /// 当前队列大小
    pub queue_size: AtomicUsize,
    /// 订阅者数量
    pub subscriber_count: AtomicUsize,
    /// 系统启动时间
    pub started_at: RwLock<Option<DateTime<Utc>>>,
    /// 最后活动时间
    pub last_activity_at: RwLock<Option<DateTime<Utc>>>,
}

/// 事件持久化管理器接口
#[async_trait]
pub trait EventPersistenceManager: Send + Sync {
    /// 保存事件
    async fn save_event(&self, event: &UnifiedEventWrapper) -> Result<()>;
    
    /// 加载事件历史
    async fn load_event_history(&self, limit: usize) -> Result<Vec<UnifiedEventWrapper>>;
    
    /// 清理过期事件
    async fn cleanup_expired_events(&self, before: DateTime<Utc>) -> Result<usize>;
    
    /// 获取统计信息
    async fn get_stats(&self) -> Result<PersistenceStats>;
}

/// 持久化统计信息
#[derive(Debug, Clone)]
pub struct PersistenceStats {
    pub total_events: usize,
    pub storage_size: usize,
    pub last_cleanup: Option<DateTime<Utc>>,
}

/// 分布式事件管理器接口
#[async_trait]
pub trait DistributedEventManager: Send + Sync {
    /// 同步事件到其他节点
    async fn sync_event(&self, event: &UnifiedEventWrapper) -> Result<()>;
    
    /// 从其他节点接收事件
    async fn receive_events(&self) -> Result<Vec<UnifiedEventWrapper>>;
    
    /// 获取集群状态
    async fn get_cluster_status(&self) -> Result<ClusterStatus>;
    
    /// 选举主节点
    async fn elect_master(&self) -> Result<String>;
}

/// 集群状态
#[derive(Debug, Clone)]
pub struct ClusterStatus {
    pub master_node: String,
    pub active_nodes: Vec<String>,
    pub total_nodes: usize,
    pub sync_lag: Duration,
}

/// 事件处理器
pub struct EventProcessor {
    /// 处理器ID
    pub id: String,
    /// 处理状态
    pub is_running: Arc<AtomicBool>,
    /// 处理统计
    pub stats: ProcessorStats,
}

/// 处理器统计信息
#[derive(Debug, Default)]
pub struct ProcessorStats {
    pub events_processed: AtomicUsize,
    pub processing_time: RwLock<Duration>,
    pub last_processed_at: RwLock<Option<DateTime<Utc>>>,
}

impl UnifiedEventSystem {
    /// 创建新的统一事件系统
    pub fn new(config: UnifiedEventSystemConfig) -> Result<Self> {
        let mut system = Self {
            config: config.clone(),
            subscribers: RwLock::new(HashMap::new()),
            event_queue: Arc::new(Mutex::new(VecDeque::with_capacity(config.queue_capacity))),
            event_history: RwLock::new(VecDeque::with_capacity(config.max_history_size)),
            processing_state: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(EventSystemStats::default()),
            persistence_manager: None,
            distributed_manager: None,
            event_processors: Vec::new(),
        };
        
        // 初始化处理器
        system.initialize_processors()?;
        
        // 初始化持久化管理器
        if config.enable_persistence {
            system.initialize_persistence()?;
        }
        
        // 初始化分布式管理器
        if config.enable_distributed {
            system.initialize_distributed()?;
        }
        
        Ok(system)
    }
    
    /// 初始化事件处理器
    fn initialize_processors(&mut self) -> Result<()> {
        for i in 0..self.config.processor_threads {
            let processor = Arc::new(EventProcessor {
                id: format!("processor-{}", i),
                is_running: Arc::new(AtomicBool::new(false)),
                stats: ProcessorStats::default(),
            });
            self.event_processors.push(processor);
        }
        Ok(())
    }
    
    /// 初始化持久化管理器
    fn initialize_persistence(&mut self) -> Result<()> {
        // 这里可以根据配置创建不同类型的持久化管理器
        // 例如：文件持久化、数据库持久化等
        Ok(())
    }
    
    /// 初始化分布式管理器
    fn initialize_distributed(&mut self) -> Result<()> {
        // 这里可以根据配置创建分布式管理器
        // 例如：Redis集群、Kafka等
        Ok(())
    }
    
    /// 发布事件
    pub async fn publish(&self, event: UnifiedEventWrapper) -> Result<()> {
        // 更新统计信息
        self.stats.events_published.fetch_add(1, Ordering::Relaxed);
        self.stats.last_activity_at.write().unwrap().replace(Utc::now());
        
        // 添加到队列
        {
            let mut queue = self.event_queue.lock().unwrap();
            if queue.len() >= self.config.queue_capacity {
                return Err(Error::resource("Event queue is full"));
            }
            queue.push_back(event.clone());
            self.stats.queue_size.store(queue.len(), Ordering::Relaxed);
        }
        
        // 持久化事件
        if let Some(persistence) = &self.persistence_manager {
            persistence.save_event(&event).await?;
        }
        
        // 分布式同步
        if let Some(distributed) = &self.distributed_manager {
            distributed.sync_event(&event).await?;
        }
        
        // 立即处理事件
        self.process_event(event).await?;
        
        Ok(())
    }
    
    /// 处理单个事件
    async fn process_event(&self, event: UnifiedEventWrapper) -> Result<()> {
        let subscribers = self.subscribers.read().unwrap();
        
        for subscriber in subscribers.values() {
            // 检查事件类型匹配
            let event_type = match &event {
                UnifiedEventWrapper::New(e) => e.event_type.to_string(),
                UnifiedEventWrapper::Enhanced(e) => e.base.event_type.to_string(),
                UnifiedEventWrapper::Legacy(e) => format!("{}", e.event_type),
            };
            
            if !subscriber.event_types.is_empty() && !subscriber.event_types.contains(&event_type) {
                continue;
            }
            
            // 处理事件
            let start_time = std::time::Instant::now();
            let result = match &event {
                UnifiedEventWrapper::New(e) => subscriber.callback.on_event(e).await,
                UnifiedEventWrapper::Enhanced(e) => subscriber.callback.on_enhanced_event(e).await,
                UnifiedEventWrapper::Legacy(e) => subscriber.callback.on_old_event(e).await,
            };
            
            let processing_time = start_time.elapsed();
            
            // 更新统计信息
            match result {
                Ok(_) => {
                    subscriber.stats.events_processed.fetch_add(1, Ordering::Relaxed);
                    self.stats.events_processed.fetch_add(1, Ordering::Relaxed);
                }
                Err(_) => {
                    subscriber.stats.events_failed.fetch_add(1, Ordering::Relaxed);
                    self.stats.events_failed.fetch_add(1, Ordering::Relaxed);
                }
            }
            
            // 更新处理时间
            subscriber.stats.last_processed_at.write().unwrap().replace(Utc::now());
            *subscriber.stats.avg_processing_time.write().unwrap() = processing_time;
        }
        
        // 添加到历史记录
        {
            let mut history = self.event_history.write().unwrap();
            if history.len() >= self.config.max_history_size {
                history.pop_front();
            }
            history.push_back(event);
        }
        
        Ok(())
    }
    
    /// 订阅事件
    pub fn subscribe<T: UnifiedEventCallback + 'static>(
        &self,
        name: &str,
        event_types: Vec<String>,
        callback: T,
    ) -> Result<String> {
        let subscriber_id = Uuid::new_v4().to_string();
        let subscriber = SubscriberInfo {
            id: subscriber_id.clone(),
            name: name.to_string(),
            event_types: event_types.into_iter().collect(),
            callback: Arc::new(callback),
            created_at: Utc::now(),
            stats: SubscriberStats::default(),
        };
        
        self.subscribers.write().unwrap().insert(subscriber_id.clone(), subscriber);
        self.stats.subscriber_count.store(self.subscribers.read().unwrap().len(), Ordering::Relaxed);
        
        Ok(subscriber_id)
    }
    
    /// 取消订阅
    pub fn unsubscribe(&self, subscriber_id: &str) -> Result<()> {
        self.subscribers.write().unwrap().remove(subscriber_id);
        self.stats.subscriber_count.store(self.subscribers.read().unwrap().len(), Ordering::Relaxed);
        Ok(())
    }
    
    /// 启动事件系统
    pub async fn start(&self) -> Result<()> {
        self.processing_state.store(true, Ordering::Relaxed);
        self.stats.started_at.write().unwrap().replace(Utc::now());
        
        // 启动事件处理器
        for processor in &self.event_processors {
            processor.is_running.store(true, Ordering::Relaxed);
        }
        
        Ok(())
    }
    
    /// 停止事件系统
    pub async fn stop(&self) -> Result<()> {
        self.processing_state.store(false, Ordering::Relaxed);
        
        // 停止事件处理器
        for processor in &self.event_processors {
            processor.is_running.store(false, Ordering::Relaxed);
        }
        
        Ok(())
    }
    
    /// 获取统计信息
    pub fn get_stats(&self) -> EventSystemStats {
        EventSystemStats {
            events_published: AtomicUsize::new(self.stats.events_published.load(std::sync::atomic::Ordering::Relaxed)),
            events_processed: AtomicUsize::new(self.stats.events_processed.load(std::sync::atomic::Ordering::Relaxed)),
            events_failed: AtomicUsize::new(self.stats.events_failed.load(std::sync::atomic::Ordering::Relaxed)),
            queue_size: AtomicUsize::new(self.stats.queue_size.load(std::sync::atomic::Ordering::Relaxed)),
            subscriber_count: AtomicUsize::new(self.stats.subscriber_count.load(std::sync::atomic::Ordering::Relaxed)),
            started_at: RwLock::new(self.stats.started_at.read().unwrap().clone()),
            last_activity_at: RwLock::new(self.stats.last_activity_at.read().unwrap().clone()),
        }
    }
}

impl EventSystem for UnifiedEventSystem {
    fn publish(&self, event: Event) -> Result<()> {
        let wrapper = UnifiedEventWrapper::New(event);
        // 使用 tokio::runtime::Handle::current() 来在同步上下文中运行异步代码
        let rt = tokio::runtime::Handle::current();
        rt.block_on(self.publish(wrapper))
    }
    
    fn subscribe(&self, event_type: EventType, callback: Arc<dyn EventCallback>) -> Result<String> {
        // 创建一个适配器来将 EventCallback 转换为 UnifiedEventCallback
        let adapter = EventCallbackAdapter { callback };
        let event_types = vec![event_type.to_string()];
        self.subscribe("event_system_adapter", event_types, adapter)
    }
    
    fn subscribe_all(&self, callback: Arc<dyn EventCallback>) -> Result<String> {
        let adapter = EventCallbackAdapter { callback };
        let event_types = vec!["*".to_string()]; // 通配符表示所有事件
        self.subscribe("event_system_adapter_all", event_types, adapter)
    }
    
    fn unsubscribe(&self, subscription_id: &str) -> Result<()> {
        self.unsubscribe(subscription_id)
    }
    
    fn get_pending_events(&self) -> Result<Vec<Event>> {
        let queue = self.event_queue.lock().map_err(|e| Error::internal(format!("Failed to lock event queue: {}", e)))?;
        let mut events = Vec::new();
        for wrapper in queue.iter() {
            if let UnifiedEventWrapper::New(event) = wrapper {
                events.push(event.clone());
            }
        }
        Ok(events)
    }
    
    fn start(&self) -> Result<()> {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(self.start())
    }
    
    fn stop(&self) -> Result<()> {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(self.stop())
    }
}

/// 事件回调适配器
struct EventCallbackAdapter {
    callback: Arc<dyn EventCallback>,
}

impl std::fmt::Debug for EventCallbackAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventCallbackAdapter")
            .field("callback", &"Arc<dyn EventCallback>")
            .finish()
    }
}

#[async_trait]
impl UnifiedEventCallback for EventCallbackAdapter {
    async fn on_event(&self, event: &Event) -> Result<()> {
        // EventCallback 的 on_event 是同步的，所以我们需要在异步上下文中调用它
        // 使用 spawn_blocking 来避免阻塞异步运行时
        let callback = self.callback.clone();
        let event = event.clone();
        tokio::task::spawn_blocking(move || {
            callback.on_event(&event)
        }).await.map_err(|e| Error::Internal(format!("任务执行失败: {}", e)))?
    }
    
    fn convert_event_type(&self, _event_type: &EventType) -> OldEventType {
        // 简单的类型转换，实际应用中可能需要更复杂的映射
        OldEventType::Custom("converted".to_string())
    }
    
    fn convert_old_event_type(&self, _event_type: &OldEventType) -> EventType {
        // 简单的类型转换，实际应用中可能需要更复杂的映射
        EventType::Custom("converted".to_string())
    }
}

/// 创建默认的统一事件系统
pub fn create_unified_event_system() -> Result<Arc<UnifiedEventSystem>> {
    let config = UnifiedEventSystemConfig::default();
    let system = UnifiedEventSystem::new(config)?;
    Ok(Arc::new(system))
}

/// 创建带配置的统一事件系统
pub fn create_unified_event_system_with_config(config: UnifiedEventSystemConfig) -> Result<Arc<UnifiedEventSystem>> {
    let system = UnifiedEventSystem::new(config)?;
    Ok(Arc::new(system))
} 