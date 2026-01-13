use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock, atomic::{AtomicBool, Ordering}};
use std::time::Duration;
use uuid::Uuid;
use log::{debug, info, warn, error};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

use crate::{Result, Error};
use crate::event::{Event, EventType, EventSystem, EventCallback, EventSystemEnum};

/// 增强事件订阅者
pub struct EnhancedEventSubscriber {
    /// 订阅者ID
    pub id: String,
    /// 事件回调函数
    pub callback: Arc<dyn EventCallback>,
    /// 订阅者的事件过滤器
    pub filter: Option<Box<dyn crate::event::EventFilter>>,
    /// 创建时间
    pub created_at: std::time::Instant,
    /// 最后一次触发时间
    pub last_triggered: Option<std::time::Instant>,
    /// 处理的事件计数
    pub event_count: usize,
}

impl std::fmt::Debug for EnhancedEventSubscriber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnhancedEventSubscriber")
            .field("id", &self.id)
            .field("callback", &"<dyn EventCallback>")
            .field("filter", &"<dyn EventFilter>")
            .field("created_at", &self.created_at)
            .field("last_triggered", &self.last_triggered)
            .field("event_count", &self.event_count)
            .finish()
    }
}

impl Clone for EnhancedEventSubscriber {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            callback: self.callback.clone(),
            filter: None, // Cannot clone trait objects, set to None
            created_at: self.created_at,
            last_triggered: self.last_triggered,
            event_count: self.event_count,
        }
    }
}

/// 事件处理器trait
pub trait EventProcessor: Send + Sync {
    /// 处理增强事件
    fn process(&self, event: &EnhancedEvent) -> Result<()>;
    
    /// 获取处理器名称
    fn name(&self) -> &str;
    
    /// 获取处理器优先级
    fn priority(&self) -> EventPriority;
}

/// 增强事件系统配置
#[derive(Debug, Clone)]
pub struct EnhancedEventConfig {
    /// 队列容量
    pub queue_capacity: usize,
    /// 事件超时时间
    pub event_timeout: Duration,
    /// 是否启用优先级处理
    pub enable_priority_processing: bool,
    /// 是否启用事件统计
    pub enable_statistics: bool,
    /// 最大并发处理器数量
    pub max_concurrent_processors: usize,
    /// 是否启用事件路由
    pub enable_event_routing: bool,
}

impl Default for EnhancedEventConfig {
    fn default() -> Self {
        Self {
            queue_capacity: 1000,
            event_timeout: Duration::from_secs(30),
            enable_priority_processing: true,
            enable_statistics: true,
            max_concurrent_processors: 4,
            enable_event_routing: true,
        }
    }
}

/// 事件统计信息
#[derive(Debug, Clone)]
pub struct EventStats {
    /// 总事件数量
    pub total_events: usize,
    /// 已处理事件数量
    pub processed_events: usize,
    /// 失败事件数量
    pub failed_events: usize,
    /// 平均处理时间（毫秒）
    pub avg_processing_time_ms: f64,
    /// 最大处理时间（毫秒）
    pub max_processing_time_ms: u64,
    /// 最小处理时间（毫秒）
    pub min_processing_time_ms: u64,
    /// 最后更新时间
    pub last_updated: std::time::Instant,
}

impl Default for EventStats {
    fn default() -> Self {
        Self {
            total_events: 0,
            processed_events: 0,
            failed_events: 0,
            avg_processing_time_ms: 0.0,
            max_processing_time_ms: 0,
            min_processing_time_ms: u64::MAX,
            last_updated: std::time::Instant::now(),
        }
    }
}

/// 域事件结构 - 用于业务领域事件发布
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainEvent {
    /// 事件ID
    pub id: String,
    /// 事件类型
    pub event_type: String,
    /// 事件源
    pub source: String,
    /// 事件数据
    pub data: HashMap<String, String>,
    /// 事件元数据
    pub metadata: Option<HashMap<String, String>>,
    /// 创建时间戳
    pub timestamp: DateTime<Utc>,
    /// 事件版本
    pub version: String,
    /// 聚合根ID
    pub aggregate_id: Option<String>,
    /// 事件优先级
    pub priority: EventPriority,
}

impl DomainEvent {
    /// 创建新的域事件
    pub fn new(event_type: String, source: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            event_type,
            source,
            data: HashMap::new(),
            metadata: None,
            timestamp: Utc::now(),
            version: "1.0".to_string(),
            aggregate_id: None,
            priority: EventPriority::Normal,
        }
    }
    
    /// 添加事件数据
    pub fn with_data<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.data.insert(key.into(), value.into());
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
    
    /// 设置聚合根ID
    pub fn with_aggregate_id(mut self, aggregate_id: String) -> Self {
        self.aggregate_id = Some(aggregate_id);
        self
    }
    
    /// 设置优先级
    pub fn with_priority(mut self, priority: EventPriority) -> Self {
        self.priority = priority;
        self
    }
    
    /// 设置版本
    pub fn with_version(mut self, version: String) -> Self {
        self.version = version;
        self
    }
    
    /// 添加数据字段（链式调用）
    pub fn add_data<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.data.insert(key.into(), value.into());
        self
    }
    
    /// 发布到事件系统
    pub fn publish(&self, event_system: &EnhancedEventSystem) -> Result<()> {
        let enhanced_event = EnhancedEvent::from(self.to_event())
            .with_priority(self.priority);
        event_system.publish_enhanced(enhanced_event)
    }
    
    /// 转换为基础事件
    pub fn to_event(&self) -> Event {
        // 将字符串转换为 EventType，如果失败则使用 Custom
        let event_type = EventType::Custom(self.event_type.clone());
        let mut event = Event::new(event_type, &self.source);
        for (k, v) in &self.data {
            event = event.with_data(k, v);
        }
        event
    }
    
    /// 获取数据字段
    pub fn get_data(&self, key: &str) -> Option<&String> {
        self.data.get(key)
    }
    
    /// 获取元数据字段
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.as_ref()?.get(key)
    }
}

impl From<Event> for DomainEvent {
    fn from(event: Event) -> Self {
        Self {
            id: event.id.clone(),
            event_type: format!("{:?}", event.event_type),
            source: event.source.clone(),
            data: event.data.clone(),
            metadata: None,
            timestamp: Utc::now(),
            version: "1.0".to_string(),
            aggregate_id: None,
            priority: EventPriority::Normal,
        }
    }
}

/// 事件优先级
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventPriority {
    /// 低优先级
    Low = 0,
    /// 普通优先级
    Normal = 1,
    /// 高优先级
    High = 2,
    /// 紧急优先级
    Critical = 3,
}

impl Default for EventPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// 事件处理状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventProcessingState {
    /// 已创建
    Created,
    /// 已发送
    Dispatched,
    /// 正在处理
    Processing,
    /// 已处理
    Processed,
    /// 已完成
    Completed,
    /// 已失败
    Failed,
}

/// 事件处理步骤
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventProcessingStep {
    /// 处理模块
    pub module: String,
    /// 处理时间
    pub timestamp: DateTime<Utc>,
    /// 处理状态
    pub state: EventProcessingState,
    /// 处理结果
    pub result: Option<String>,
}

/// 事件路由信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRouting {
    /// 源模块
    pub source_module: String,
    /// 目标模块（如果为空则广播）
    pub target_modules: Vec<String>,
    /// 处理状态
    pub processing_state: EventProcessingState,
    /// 处理历史
    pub processing_history: Vec<EventProcessingStep>,
}

/// 增强的事件结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedEvent {
    /// 基础事件
    pub base: Event,
    /// 事件标签
    pub tags: HashSet<String>,
    /// 事件优先级
    pub priority: EventPriority,
    /// 关联事件ID
    pub related_events: Vec<String>,
    /// 路由信息
    pub routing: EventRouting,
}

impl EnhancedEvent {
    /// 使用标签创建增强事件
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.insert(tag.to_string());
        self
    }
    
    /// 设置事件优先级
    pub fn with_priority(mut self, priority: EventPriority) -> Self {
        self.priority = priority;
        self
    }
    
    /// 添加关联事件
    pub fn with_related_event(mut self, event_id: &str) -> Self {
        self.related_events.push(event_id.to_string());
        self
    }
    
    /// 添加目标模块
    pub fn with_target(mut self, target_module: &str) -> Self {
        self.routing.target_modules.push(target_module.to_string());
        self
    }
    
    /// 添加处理步骤
    pub fn add_processing_step(&mut self, module: &str, state: EventProcessingState, result: Option<String>) {
        self.routing.processing_state = state;
        self.routing.processing_history.push(EventProcessingStep {
            module: module.to_string(),
            timestamp: Utc::now(),
            state,
            result,
        });
    }
    
    /// 获取基础事件
    pub fn base(&self) -> &Event {
        &self.base
    }
}

impl From<Event> for EnhancedEvent {
    fn from(event: Event) -> Self {
        EnhancedEvent {
            base: event,
            tags: HashSet::new(),
            priority: EventPriority::Normal,
            related_events: Vec::new(),
            routing: EventRouting {
                source_module: "unknown".to_string(),
                target_modules: Vec::new(),
                processing_state: EventProcessingState::Created,
                processing_history: Vec::new(),
            },
        }
    }
}

/// 增强事件回调
pub type EnhancedEventHandler = Arc<dyn Fn(&EnhancedEvent) -> Result<()> + Send + Sync>;

/// 增强事件系统
/// 
/// 提供高级事件处理功能，包括优先级、过滤、增强事件等
pub struct EnhancedEventSystem {
    /// 内部事件系统
    inner: Box<EventSystemEnum>,
    /// 模块订阅
    module_subscriptions: RwLock<HashMap<String, HashSet<EventType>>>,
    /// 标签订阅
    tag_subscriptions: RwLock<HashMap<String, HashSet<String>>>,
    /// 优先级处理器
    priority_handlers: RwLock<HashMap<EventPriority, Vec<Arc<dyn Fn(&EnhancedEvent) -> Result<()> + Send + Sync>>>>,
    /// 源名称
    source_name: String,
    /// 处理运行状态
    processing_running: Arc<AtomicBool>,
    /// 处理器注册表
    handler_registry: Arc<RwLock<HashMap<String, Vec<Arc<dyn Fn(Event) -> Result<()> + Send + Sync>>>>>,
    /// 事件队列
    event_queue: Arc<Mutex<VecDeque<Event>>>,
    /// 事件超时时间
    event_timeout: Duration,
}

impl Clone for EnhancedEventSystem {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            module_subscriptions: RwLock::new(HashMap::new()), // Cannot clone RwLock contents
            tag_subscriptions: RwLock::new(HashMap::new()), // Cannot clone RwLock contents
            priority_handlers: RwLock::new(HashMap::new()), // Cannot clone RwLock contents
            source_name: self.source_name.clone(),
            processing_running: self.processing_running.clone(),
            handler_registry: Arc::new(RwLock::new(HashMap::new())), // Cannot clone RwLock contents
            event_queue: Arc::new(Mutex::new(VecDeque::new())), // Cannot clone Mutex contents
            event_timeout: self.event_timeout,
        }
    }
}

impl EnhancedEventSystem {
    /// 创建新的增强事件系统
    pub fn new(source_name: &str, queue_capacity: usize, event_timeout: Duration) -> Self {
        // 默认使用内存事件系统
        let inner = EventSystemEnum::Memory(crate::event::memory::MemoryEventSystem::new(queue_capacity));
        
        Self {
            inner: Box::new(inner),
            module_subscriptions: RwLock::new(HashMap::new()),
            tag_subscriptions: RwLock::new(HashMap::new()),
            priority_handlers: RwLock::new(HashMap::new()),
            source_name: source_name.to_string(),
            processing_running: Arc::new(AtomicBool::new(false)),
            handler_registry: Arc::new(RwLock::new(HashMap::new())),
            event_queue: Arc::new(Mutex::new(VecDeque::with_capacity(queue_capacity))),
            event_timeout,
        }
    }
    
    /// 使用指定的后端创建增强事件系统
    pub fn with_backend(source_name: &str, backend: EventSystemEnum, queue_capacity: usize, event_timeout: Duration) -> Self {
        Self {
            inner: Box::new(backend),
            module_subscriptions: RwLock::new(HashMap::new()),
            tag_subscriptions: RwLock::new(HashMap::new()),
            priority_handlers: RwLock::new(HashMap::new()),
            source_name: source_name.to_string(),
            processing_running: Arc::new(AtomicBool::new(false)),
            handler_registry: Arc::new(RwLock::new(HashMap::new())),
            event_queue: Arc::new(Mutex::new(VecDeque::with_capacity(queue_capacity))),
            event_timeout,
        }
    }
    
    /// 注册模块订阅
    pub fn register_module_subscription(&self, module: &str, event_types: Vec<EventType>) -> Result<()> {
        let mut subscriptions = self.module_subscriptions.write().map_err(|e| {
            Error::Internal(format!("无法获取模块订阅锁: {}", e))
        })?;
        
        let entry = subscriptions.entry(module.to_string()).or_insert_with(HashSet::new);
        for event_type in event_types {
            entry.insert(event_type);
        }
        
        Ok(())
    }
    
    /// 注册标签订阅
    pub fn register_tag_subscription(&self, module: &str, tags: Vec<String>) -> Result<()> {
        let mut subscriptions = self.tag_subscriptions.write().map_err(|e| {
            Error::Internal(format!("无法获取标签订阅锁: {}", e))
        })?;
        
        let entry = subscriptions.entry(module.to_string()).or_insert_with(HashSet::new);
        for tag in tags {
            entry.insert(tag);
        }
        
        Ok(())
    }
    
    /// 注册优先级处理器
    pub fn register_priority_handler<F>(&self, priority: EventPriority, handler: F) -> Result<()>
    where
        F: Fn(&EnhancedEvent) -> Result<()> + Send + Sync + 'static,
    {
        let mut handlers = self.priority_handlers.write().map_err(|e| {
            Error::Internal(format!("无法获取优先级处理器锁: {}", e))
        })?;
        
        let entry = handlers.entry(priority).or_insert_with(Vec::new);
        entry.push(Arc::new(handler));
        
        Ok(())
    }

    /// 最小回调与过滤器注册：用于行使 EventCallback / EventFilter 等导入
    pub fn readiness_probe(&self) -> Result<()> {
        // 注册一个高优先级的空处理器
        self.register_priority_handler(EventPriority::High, |_e: &EnhancedEvent| {
            Ok(())
        })?;

        // 注册一个基础的事件处理器到内部注册表，行使 EventCallback 类型
        self.register_handler("system.probe", |_evt: crate::event::Event| {
            Ok(())
        })?;

        // 发布一个探针事件
        let evt = crate::event::Event::new(crate::event::EventType::Custom("system.probe".to_string()), &self.source_name);
        let _ = self.publish_as_enhanced(evt);
        Ok(())
    }
    
    /// 发布增强事件
    pub fn publish_enhanced(&self, mut event: EnhancedEvent) -> Result<()> {
        // 添加处理步骤
        event.add_processing_step(
            &self.source_name, 
            EventProcessingState::Dispatched,
            None
        );
        
        // 1. 优先级处理
        let handlers = self.priority_handlers.read().map_err(|e| {
            Error::Internal(format!("无法获取优先级处理器锁: {}", e))
        })?;
        
        if let Some(priority_handlers) = handlers.get(&event.priority) {
            for handler in priority_handlers {
                if let Err(e) = handler(&event) {
                    error!("优先级处理器执行失败: {}", e);
                }
            }
        }
        
        // 2. 目标模块处理
        if !event.routing.target_modules.is_empty() {
            // 生产级实现：遍历目标模块，分发事件到对应模块的订阅处理器
            let module_subs = self.module_subscriptions.read().map_err(|e| {
                Error::Internal(format!("无法获取模块订阅锁: {}", e))
            })?;
            let tag_subs = self.tag_subscriptions.read().map_err(|e| {
                Error::Internal(format!("无法获取标签订阅锁: {}", e))
            })?;
            for module in &event.routing.target_modules {
                // 1. 按事件类型分发
                if let Some(types) = module_subs.get(module) {
                    for event_type in types {
                        if &event.base.event_type == event_type {
                            // 查找注册的处理器
                            let registry = self.handler_registry.read().map_err(|e| {
                                Error::Internal(format!("无法获取处理器注册表锁: {}", e))
                            })?;
                            let event_type_str = match event_type {
                                EventType::Custom(name) => name.clone(),
                                _ => format!("{:?}", event_type),
                            };
                            if let Some(handlers) = registry.get(&event_type_str) {
                                for handler in handlers {
                                    if let Err(e) = handler(event.base.clone()) {
                                        error!("目标模块事件处理失败: {}", e);
                                    }
                                }
                            }
                        }
                    }
                }
                // 2. 按标签分发
                if let Some(tags) = tag_subs.get(module) {
                    for tag in tags {
                        if event.tags.contains(tag) {
                            // 查找注册的处理器
                            let registry = self.handler_registry.read().map_err(|e| {
                                Error::Internal(format!("无法获取处理器注册表锁: {}", e))
                            })?;
                            if let Some(handlers) = registry.get(tag) {
                                for handler in handlers {
                                    if let Err(e) = handler(event.base.clone()) {
                                        error!("目标模块标签事件处理失败: {}", e);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 3. 将事件发布到基础事件系统
        let base_event = event.base.clone();
        self.inner.publish(base_event)?;
        
        Ok(())
    }
    
    /// 从Event创建并发布增强事件
    pub fn publish_as_enhanced(&self, event: Event) -> Result<()> {
        let enhanced = EnhancedEvent::from(event);
        self.publish_enhanced(enhanced)
    }
    
    /// 启动事件处理
    pub fn start_processing(&self) -> Result<()> {
        // 避免重复启动
        if self.processing_running.load(Ordering::SeqCst) {
            debug!("增强事件系统处理已在运行");
            return Ok(());
        }
        
        // 设置处理标志
        self.processing_running.store(true, Ordering::SeqCst);
        info!("增强事件系统处理启动");
        
        // 启动内部事件系统
        self.inner.start()?;
        
        // 创建处理线程
        let queue = self.event_queue.clone();
        let registry = Arc::clone(&self.handler_registry);
        let running = self.processing_running.clone();
        let timeout = self.event_timeout;
        
        std::thread::spawn(move || {
            let sleep_duration = Duration::from_millis(10);
            
            while running.load(Ordering::SeqCst) {
                // 获取事件队列中的下一个事件
                let event = {
                    let mut queue_guard = match queue.lock() {
                        Ok(guard) => guard,
                        Err(e) => {
                            error!("无法获取事件队列锁: {}", e);
                            std::thread::sleep(sleep_duration);
                            continue;
                        }
                    };
                    
                    queue_guard.pop_front()
                };
                
                // 如果队列为空，则休眠一段时间
                if event.is_none() {
                    std::thread::sleep(sleep_duration);
                    continue;
                }
                
                let event = event.unwrap();
                debug!("处理事件: {:?}", event.event_type);
                
                // 获取处理器注册表
                let registry_guard = match registry.read() {
                    Ok(guard) => guard,
                    Err(e) => {
                        error!("无法获取处理器注册表锁: {}", e);
                        continue;
                    }
                };
                
                // 获取事件类型字符串
                let event_type_str = match &event.event_type {
                    EventType::Custom(name) => name.clone(),
                    _ => format!("{:?}", event.event_type),
                };
                
                // 查找处理器
                if let Some(handlers) = registry_guard.get(&event_type_str) {
                    for handler in handlers {
                        let event_clone = event.clone();
                        let handler_clone = handler.clone();
                        
                        // 在单独的线程中处理事件
                        std::thread::spawn(move || {
                            if let Err(e) = handler_clone(event_clone) {
                                error!("事件处理失败: {}", e);
                            }
                        });
                    }
                } else {
                    warn!("未找到事件处理器: {}", event_type_str);
                }
            }
        });
        
        Ok(())
    }
    
    /// 停止事件处理
    pub fn stop_processing(&self) -> Result<()> {
        // 设置处理标志
        self.processing_running.store(false, Ordering::SeqCst);
        info!("增强事件系统处理停止");
        
        // 停止内部事件系统
        self.inner.stop()?;
        
        Ok(())
    }
    
    /// 注册事件处理器
    pub fn register_handler<F>(&self, event_type: &str, handler: F) -> Result<()>
    where
        F: Fn(Event) -> Result<()> + Send + Sync + 'static,
    {
        let mut registry = self.handler_registry.write().map_err(|e| {
            Error::Internal(format!("无法获取处理器注册表锁: {}", e))
        })?;
        
        let entry = registry.entry(event_type.to_string()).or_insert_with(Vec::new);
        entry.push(Arc::new(handler));
        
        Ok(())
    }
    
    /// 发布事件到队列
    pub fn publish_to_queue(&self, event: Event) -> Result<()> {
        let mut queue = self.event_queue.lock().map_err(|e| {
            Error::Internal(format!("无法获取事件队列锁: {}", e))
        })?;
        
        queue.push_back(event);
        
        Ok(())
    }
    
    /// 获取队列中的事件数量
    pub fn queue_size(&self) -> Result<usize> {
        let queue = self.event_queue.lock().map_err(|e| {
            Error::Internal(format!("无法获取事件队列锁: {}", e))
        })?;
        
        Ok(queue.len())
    }
    
    /// 清空事件队列
    pub fn clear_queue(&self) -> Result<()> {
        let mut queue = self.event_queue.lock().map_err(|e| {
            Error::Internal(format!("无法获取事件队列锁: {}", e))
        })?;
        
        queue.clear();
        
        Ok(())
    }
    
    /// 发布域事件
    pub fn publish_domain_event(&self, domain_event: &DomainEvent) -> Result<()> {
        let enhanced_event = EnhancedEvent::from(domain_event.to_event())
            .with_priority(domain_event.priority);
        self.publish_enhanced(enhanced_event)
    }
    
    /// 创建并发布域事件的便捷方法
    pub fn create_and_publish_domain_event(&self, event_type: &str, source: &str, data: HashMap<String, String>) -> Result<()> {
        let mut domain_event = DomainEvent::new(event_type.to_string(), source.to_string());
        for (key, value) in data {
            domain_event = domain_event.with_data(key, value);
        }
        self.publish_domain_event(&domain_event)
    }
}

// 实现EventSystem接口以保持兼容性
impl EventSystem for EnhancedEventSystem {
    fn publish(&self, event: Event) -> Result<()> {
        // 发布到队列
        self.publish_to_queue(event.clone())?;
        
        // 同时也发布到内部事件系统
        self.inner.publish(event)
    }
    
    fn start(&self) -> Result<()> {
        self.start_processing()
    }
    
    fn stop(&self) -> Result<()> {
        self.stop_processing()
    }
    
    fn subscribe(&self, event_type: EventType, callback: Arc<dyn EventCallback>) -> Result<String> {
        // 创建订阅ID
        let subscription_id = Uuid::new_v4().to_string();
        
        // 注册到内部事件系统
        self.inner.subscribe(event_type, callback)?;
        
        Ok(subscription_id)
    }
    
    fn subscribe_all(&self, callback: Arc<dyn EventCallback>) -> Result<String> {
        // 创建订阅ID
        let subscription_id = Uuid::new_v4().to_string();
        
        // 注册到内部事件系统
        self.inner.subscribe_all(callback)?;
        
        Ok(subscription_id)
    }
    
    fn unsubscribe(&self, subscription_id: &str) -> Result<()> {
        // 取消内部事件系统的订阅
        self.inner.unsubscribe(subscription_id)
    }
    
    fn get_pending_events(&self) -> Result<Vec<Event>> {
        self.inner.get_pending_events()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    
    struct TestCallback {
        counter: Arc<AtomicUsize>,
    }
    
    impl EventCallback for TestCallback {
        fn on_event(&self, _event: &Event) -> Result<()> {
            self.counter.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }
    
    #[test]
    fn test_enhanced_event_system() {
        // 创建增强事件系统
        let system = EnhancedEventSystem::new("test", 100, Duration::from_secs(1));
        
        // 启动处理
        system.start_processing().unwrap();
        
        // 创建计数器
        let counter = Arc::new(AtomicUsize::new(0));
        let callback = Arc::new(TestCallback { counter: counter.clone() });
        
        // 订阅事件
        system.subscribe(EventType::Custom("test_event".to_string()), callback).unwrap();
        
        // 发布事件
        let event = Event::new(EventType::Custom("test_event".to_string()), "test");
        system.publish(event).unwrap();
        
        // 等待处理
        std::thread::sleep(Duration::from_millis(100));
        
        // 验证结果
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        
        // 停止处理
        system.stop_processing().unwrap();
    }
    
    #[test]
    fn test_enhanced_event_with_priority() {
        // 创建增强事件系统
        let system = EnhancedEventSystem::new("test", 100, Duration::from_secs(1));
        
        // 创建计数器
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        
        // 注册优先级处理器
        system.register_priority_handler(EventPriority::High, move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }).unwrap();
        
        // 创建高优先级事件
        let event = Event::new(EventType::Custom("test_event".to_string()), "test");
        let enhanced = EnhancedEvent::from(event).with_priority(EventPriority::High);
        
        // 发布增强事件
        system.publish_enhanced(enhanced).unwrap();
        
        // 验证结果
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
} 