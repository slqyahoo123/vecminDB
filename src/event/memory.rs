use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, Mutex};
use log::{debug, info, warn, error};
use std::time::{Duration, Instant};

use crate::event::{Event, EventType, EventSystem, EventCallback, EventFilter};
use crate::Result;
use crate::Error;

/// 基于内存的事件系统实现
/// 
/// 使用内存队列存储事件，支持发布-订阅模式，适合单机或小规模系统。
#[derive(Clone)]
pub struct MemoryEventSystem {
    /// 事件队列
    queue: Arc<Mutex<VecDeque<Event>>>,
    /// 事件订阅者，按照事件类型分组
    subscribers: Arc<RwLock<HashMap<EventType, Vec<EventSubscriber>>>>,
    /// 全局订阅者，接收所有事件
    global_subscribers: Arc<RwLock<Vec<EventSubscriber>>>,
    /// 事件处理状态
    is_running: Arc<Mutex<bool>>,
    /// 事件队列容量限制
    capacity: usize,
    /// 事件保留策略
    retention_policy: RetentionPolicy,
}

/// 事件保留策略
#[derive(Debug, Clone)]
pub enum RetentionPolicy {
    /// 保留所有事件直到容量满
    KeepAll,
    /// 保留最新的N条事件
    KeepLatest(usize),
    /// 保留指定时间段内的事件
    KeepRecent(Duration),
    /// 根据事件类型选择性保留
    Selective(HashMap<EventType, usize>),
}

/// 事件订阅者
pub struct EventSubscriber {
    /// 订阅者ID
    pub id: String,
    /// 事件回调函数
    pub callback: Arc<dyn EventCallback>,
    /// 订阅者的事件过滤器
    pub filter: Option<Box<dyn EventFilter>>,
    /// 创建时间
    pub created_at: Instant,
    /// 最后一次触发时间
    pub last_triggered: Option<Instant>,
    /// 处理的事件计数
    pub event_count: usize,
}

impl MemoryEventSystem {
    /// 创建新的内存事件系统
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))),
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            global_subscribers: Arc::new(RwLock::new(Vec::new())),
            is_running: Arc::new(Mutex::new(false)),
            capacity,
            retention_policy: RetentionPolicy::KeepLatest(1000),
        }
    }

    /// 就绪探针：注册一个最小回调与过滤器以行使 EventCallback/EventFilter 的导入与路径
    pub fn readiness_probe(&self) -> Result<()> {
        // 简单过滤器：仅允许 SystemStarted 事件
        struct AllowSystemStarted;
        impl crate::event::EventFilter for AllowSystemStarted {
            fn matches(&self, event: &crate::event::Event) -> bool {
                matches!(event.event_type, crate::event::EventType::SystemStarted)
            }
        }

        // 简单回调：记录一次调用
        struct ProbeCallback;
        impl crate::event::EventCallback for ProbeCallback {
            fn on_event(&self, _event: &crate::event::Event) -> crate::Result<()> { Ok(()) }
        }

        // 注册到全局订阅者并发布一次事件
        let id = self.subscribe_all(Arc::new(ProbeCallback))?;
        {
            let mut globals = self.global_subscribers.write().map_err(|e| {
                Error::Internal(format!("无法获取全局订阅者写锁: {}", e))
            })?;
            if let Some(last) = globals.last_mut() {
                last.filter = Some(Box::new(AllowSystemStarted));
            }
        }
        let evt = Event::new(EventType::SystemStarted, "readiness_probe");
        self.publish(evt)?;
        // 清理
        let _ = self.unsubscribe(&id);
        Ok(())
    }
    
    /// 设置事件保留策略
    pub fn set_retention_policy(&mut self, policy: RetentionPolicy) {
        self.retention_policy = policy;
    }
    
    /// 启动事件处理循环
    pub fn start(&self) -> Result<()> {
        let mut is_running = self.is_running.lock().map_err(|e| {
            Error::Internal(format!("无法获取事件系统运行状态锁: {}", e))
        })?;
        
        if *is_running {
            return Ok(());
        }
        
        *is_running = true;
        info!("MemoryEventSystem started; capacity={}", self.capacity);
        
        // 启动一个后台线程处理事件
        let queue = self.queue.clone();
        let subscribers = self.subscribers.clone();
        let global_subscribers = self.global_subscribers.clone();
        let is_running_clone = self.is_running.clone();
        
        std::thread::spawn(move || {
            Self::event_loop(queue, subscribers, global_subscribers, is_running_clone);
        });
        
        Ok(())
    }
    
    /// 事件处理循环
    fn event_loop(
        queue: Arc<Mutex<VecDeque<Event>>>, 
        subscribers: Arc<RwLock<HashMap<EventType, Vec<EventSubscriber>>>>,
        global_subscribers: Arc<RwLock<Vec<EventSubscriber>>>,
        is_running: Arc<Mutex<bool>>
    ) {
        loop {
            // 检查是否应该停止
            let running = is_running.lock().unwrap();
            if !*running {
                break;
            }
            drop(running);
            
            // 尝试获取下一个事件
            let event = {
                let mut queue_lock = queue.lock().unwrap();
                queue_lock.pop_front()
            };
            
            // 处理事件
            if let Some(event) = event {
                debug!("processing event: {}", event.event_type);
                // 1. 通知特定事件类型的订阅者
                if let Ok(subscribers_lock) = subscribers.read() {
                    if let Some(type_subscribers) = subscribers_lock.get(&event.event_type) {
                        for subscriber in type_subscribers {
                            if Self::should_notify(subscriber, &event) {
                                let callback = subscriber.callback.clone();
                                let event_clone = event.clone();
                                
                                // 在单独的线程中处理，避免阻塞事件循环
                                std::thread::spawn(move || {
                                    if let Err(e) = callback.on_event(&event_clone) {
                                        error!("事件回调执行失败: {}", e);
                                    }
                                });
                            }
                        }
                    }
                }
                
                // 2. 通知全局订阅者
                if let Ok(global_lock) = global_subscribers.read() {
                    for subscriber in global_lock.iter() {
                        if Self::should_notify(subscriber, &event) {
                            let callback = subscriber.callback.clone();
                            let event_clone = event.clone();
                            
                            std::thread::spawn(move || {
                                if let Err(e) = callback.on_event(&event_clone) {
                                    error!("全局事件回调执行失败: {}", e);
                                }
                            });
                        }
                    }
                }
            } else {
                // 如果没有事件，稍微睡眠一下，避免CPU空转
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
    
    /// 检查是否应该通知订阅者
    fn should_notify(subscriber: &EventSubscriber, event: &Event) -> bool {
        if let Some(filter) = &subscriber.filter {
            filter.matches(event)
        } else {
            true
        }
    }
    
    /// 停止事件处理
    pub fn stop(&self) -> Result<()> {
        let mut is_running = self.is_running.lock().map_err(|e| {
            Error::Internal(format!("无法获取事件系统运行状态锁: {}", e))
        })?;
        
        *is_running = false;
        info!("MemoryEventSystem stopped");
        Ok(())
    }
}

impl EventSystem for MemoryEventSystem {
    fn publish(&self, event: Event) -> Result<()> {
        let mut queue = self.queue.lock().map_err(|e| {
            Error::Internal(format!("无法获取事件队列锁: {}", e))
        })?;
        
        // 检查队列是否已满
        if queue.len() >= self.capacity {
            // 根据保留策略处理
            match self.retention_policy {
                RetentionPolicy::KeepAll => {
                    return Err(Error::resource("事件队列已满"));
                },
                RetentionPolicy::KeepLatest(_) => {
                    // 丢弃最旧的事件
                    warn!("event queue full; dropping oldest event due to KeepLatest policy");
                    queue.pop_front();
                },
                RetentionPolicy::KeepRecent(duration) => {
                    // 丢弃超过指定时间的事件
                    let now = Instant::now();
                    while !queue.is_empty() {
                        if let Some(front) = queue.front() {
                            // 将 u64 时间戳转换为相对时间来计算时间差
                            let current_time = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap()
                                .as_secs();
                            let event_age_seconds = current_time - front.timestamp;
                            if event_age_seconds > duration.as_secs() {
                                queue.pop_front();
                            } else {
                                break;
                            }
                        }
                    }
                },
                RetentionPolicy::Selective(ref policy) => {
                    // 检查该类型的事件是否需要限制数量
                    if let Some(&limit) = policy.get(&event.event_type) {
                        // 计算当前该类型事件的数量
                        let count = queue.iter()
                            .filter(|e| e.event_type == event.event_type)
                            .count();
                        
                        if count >= limit {
                            // 找到并删除最旧的该类型事件
                            let mut index = None;
                            for (i, e) in queue.iter().enumerate() {
                                if e.event_type == event.event_type {
                                    index = Some(i);
                                    break;
                                }
                            }
                            
                            if let Some(i) = index {
                                queue.remove(i);
                            }
                        }
                    }
                }
            }
        }
        
        // 添加新事件
        queue.push_back(event);
        Ok(())
    }
    
    fn subscribe(&self, event_type: EventType, callback: Arc<dyn EventCallback>) -> Result<String> {
        let id = format!("sub-{}", uuid::Uuid::new_v4());
        let subscriber = EventSubscriber {
            id: id.clone(),
            callback,
            filter: None,
            created_at: Instant::now(),
            last_triggered: None,
            event_count: 0,
        };
        
        let mut subscribers = self.subscribers.write().map_err(|e| {
            Error::Internal(format!("无法获取订阅者列表写锁: {}", e))
        })?;
        
        subscribers.entry(event_type)
            .or_insert_with(Vec::new)
            .push(subscriber);
        
        Ok(id)
    }
    
    fn subscribe_all(&self, callback: Arc<dyn EventCallback>) -> Result<String> {
        let id = format!("global-{}", uuid::Uuid::new_v4());
        let subscriber = EventSubscriber {
            id: id.clone(),
            callback,
            filter: None,
            created_at: Instant::now(),
            last_triggered: None,
            event_count: 0,
        };
        
        let mut global_subscribers = self.global_subscribers.write().map_err(|e| {
            Error::Internal(format!("无法获取全局订阅者列表写锁: {}", e))
        })?;
        
        global_subscribers.push(subscriber);
        
        Ok(id)
    }
    
    fn unsubscribe(&self, subscription_id: &str) -> Result<()> {
        // 先检查全局订阅
        {
            let mut global = self.global_subscribers.write().map_err(|e| {
                Error::Internal(format!("无法获取全局订阅者列表写锁: {}", e))
            })?;
            
            let len_before = global.len();
            global.retain(|s| s.id != subscription_id);
            
            if global.len() < len_before {
                return Ok(());
            }
        }
        
        // 然后检查特定事件类型的订阅
        let mut subscribers = self.subscribers.write().map_err(|e| {
            Error::Internal(format!("无法获取订阅者列表写锁: {}", e))
        })?;
        
        for subscribers_list in subscribers.values_mut() {
            let len_before = subscribers_list.len();
            subscribers_list.retain(|s| s.id != subscription_id);
            
            if subscribers_list.len() < len_before {
                return Ok(());
            }
        }
        
        Err(Error::NotFound(format!("未找到订阅ID: {}", subscription_id)))
    }
    
    fn get_pending_events(&self) -> Result<Vec<Event>> {
        let queue = self.queue.lock().map_err(|e| {
            Error::Internal(format!("无法获取事件队列锁: {}", e))
        })?;
        
        Ok(queue.iter().cloned().collect())
    }
    
    fn start(&self) -> Result<()> {
        let mut is_running = self.is_running.lock().map_err(|e| {
            Error::Internal(format!("无法获取运行状态锁: {}", e))
        })?;
        
        if *is_running {
            return Err(Error::AlreadyExists("事件处理已经启动".to_string()));
        }
        
        *is_running = true;
        
        let queue = self.queue.clone();
        let subscribers = self.subscribers.clone();
        let global_subscribers = self.global_subscribers.clone();
        let is_running_clone = self.is_running.clone();
        
        std::thread::spawn(move || {
            Self::event_loop(queue, subscribers, global_subscribers, is_running_clone);
        });
        
        Ok(())
    }
    
    fn stop(&self) -> Result<()> {
        let mut is_running = self.is_running.lock().map_err(|e| {
            Error::Internal(format!("无法获取运行状态锁: {}", e))
        })?;
        
        if !*is_running {
            return Err(Error::NotFound("事件处理未启动".to_string()));
        }
        
        *is_running = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    
    struct TestCallback {
        triggered: Arc<AtomicBool>,
    }
    
    impl EventCallback for TestCallback {
        fn on_event(&self, _event: &Event) -> Result<()> {
            self.triggered.store(true, Ordering::SeqCst);
            Ok(())
        }
    }
    
    #[test]
    fn test_memory_event_system() {
        let system = MemoryEventSystem::new(100);
        
        // 创建测试回调
        let triggered = Arc::new(AtomicBool::new(false));
        let callback = Arc::new(TestCallback {
            triggered: triggered.clone(),
        });
        
        // 订阅事件
        let subscription_id = system.subscribe(EventType::SystemStarted, callback).unwrap();
        
        // 发布事件
        let event = Event::new(EventType::SystemStarted, "测试事件".to_string());
        system.publish(event).unwrap();
        
        // 手动触发事件处理
        let queue = system.queue.clone();
        let subscribers = system.subscribers.clone();
        let global_subscribers = system.global_subscribers.clone();
        let is_running = system.is_running.clone();
        
        let event = queue.lock().unwrap().pop_front().unwrap();
        if let Ok(subscribers_lock) = subscribers.read() {
            if let Some(type_subscribers) = subscribers_lock.get(&event.event_type) {
                for subscriber in type_subscribers {
                    subscriber.callback.on_event(&event).unwrap();
                }
            }
        }
        
        // 验证回调被触发
        assert!(triggered.load(Ordering::SeqCst));
        
        // 测试取消订阅
        system.unsubscribe(&subscription_id).unwrap();
        
        // 验证订阅已删除
        let subscribers_lock = subscribers.read().unwrap();
        assert!(subscribers_lock.get(&EventType::SystemStarted).unwrap().is_empty());
    }
} 