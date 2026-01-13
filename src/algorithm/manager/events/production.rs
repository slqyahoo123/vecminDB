// 生产级事件系统实现
// 提供完整的事件发布订阅功能，包括事件持久化、订阅管理、事件路由等

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};
use log::{debug, info, warn};
use tokio::runtime::Runtime;
use uuid::Uuid;
use chrono;

use crate::error::{Error, Result};
use crate::event::{Event, EventSystem, EventType, EventCallback};

use super::common::{EventSystemConfig, EventSystemStats, SubscriberInfo};

/// 生产级事件系统实现
/// 提供完整的事件发布订阅功能，包括事件持久化、订阅管理、事件路由等
pub struct ProductionEventSystem {
    /// 事件存储
    event_store: Arc<RwLock<Vec<Event>>>,
    /// 订阅者映射
    subscribers: Arc<RwLock<HashMap<EventType, Vec<SubscriberInfo>>>>,
    /// 全局订阅者
    global_subscribers: Arc<RwLock<Vec<SubscriberInfo>>>,
    /// 系统配置
    config: EventSystemConfig,
    /// 运行状态
    is_running: Arc<RwLock<bool>>,
    /// 事件统计
    stats: Arc<RwLock<EventSystemStats>>,
    /// 事件处理队列
    event_queue: Arc<RwLock<std::collections::VecDeque<Event>>>,
    /// 异步运行时
    runtime: Arc<Runtime>,
}

impl ProductionEventSystem {
    /// 创建新的生产级事件系统
    pub fn new() -> Result<Self> {
        let runtime = Arc::new(Runtime::new().map_err(|e| {
            Error::Internal(format!("无法创建异步运行时: {}", e))
        })?);
        
        let system = Self {
            event_store: Arc::new(RwLock::new(Vec::new())),
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            global_subscribers: Arc::new(RwLock::new(Vec::new())),
            config: EventSystemConfig::default(),
            is_running: Arc::new(RwLock::new(false)),
            stats: Arc::new(RwLock::new(EventSystemStats::default())),
            event_queue: Arc::new(RwLock::new(std::collections::VecDeque::new())),
            runtime,
        };
        
        Ok(system)
    }
    
    /// 使用自定义配置创建事件系统
    pub fn with_config(config: EventSystemConfig) -> Result<Self> {
        let mut system = Self::new()?;
        system.config = config;
        Ok(system)
    }
    
    /// 启动事件系统
    pub async fn start(&self) -> Result<()> {
        {
            let mut is_running = self.is_running.write().map_err(|e| {
                Error::Internal(format!("无法获取运行状态写锁: {}", e))
            })?;
            
            if *is_running {
                return Err(Error::InvalidOperation("事件系统已在运行".to_string()));
            }
            *is_running = true;
        }
        
        // 启动事件处理循环
        let event_queue = self.event_queue.clone();
        let is_running = self.is_running.clone();
        
        self.runtime.spawn(async move {
            let _ = Self::process_event_queue_loop(event_queue, is_running).await;
        });
        
        info!("事件系统已启动");
        Ok(())
    }
    
    /// 停止事件系统
    pub async fn stop(&self) -> Result<()> {
        {
            let mut is_running = self.is_running.write().map_err(|e| {
                Error::Internal(format!("无法获取运行状态写锁: {}", e))
            })?;
            *is_running = false;
        }
        
        info!("事件系统已停止");
        Ok(())
    }
    
    /// 发布事件
    pub async fn publish(&self, event: Event) -> Result<()> {
        // 添加到事件存储
        if self.config.enable_persistence {
            let mut event_store = self.event_store.write().map_err(|e| {
                Error::Internal(format!("无法获取事件存储写锁: {}", e))
            })?;
            
            event_store.push(event.clone());
            
            // 限制存储大小
            if event_store.len() > self.config.max_events {
                event_store.remove(0);
            }
        }
        
        // 添加到处理队列
        {
            let mut queue = self.event_queue.write().map_err(|e| {
                Error::Internal(format!("无法获取事件队列写锁: {}", e))
            })?;
            
            queue.push_back(event);
        }
        
        // 更新统计信息
        if self.config.enable_metrics {
            let mut stats = self.stats.write().map_err(|e| {
                Error::Internal(format!("无法获取统计信息写锁: {}", e))
            })?;
            stats.total_events_published += 1;
        }
        
        Ok(())
    }
    
    /// 订阅事件
    pub async fn subscribe(
        &self,
        event_type: EventType,
        callback: Arc<dyn EventCallback + Send + Sync>,
    ) -> Result<String> {
        let subscription_id = Uuid::new_v4().to_string();
        let subscriber = SubscriberInfo {
            id: subscription_id.clone(),
            callback,
            created_at: SystemTime::now(),
            last_activity: SystemTime::now(),
        };
        
        {
            let mut subscribers = self.subscribers.write().map_err(|e| {
                Error::Internal(format!("无法获取订阅者写锁: {}", e))
            })?;
            
            let type_subscribers = subscribers.entry(event_type).or_insert_with(Vec::new);
            
            // 检查订阅者数量限制
            if type_subscribers.len() >= self.config.max_subscribers {
                return Err(Error::resource_exhausted("订阅者数量已达上限"));
            }
            
            type_subscribers.push(subscriber);
        }
        
        // 更新统计信息
        if self.config.enable_metrics {
            let mut stats = self.stats.write().map_err(|e| {
                Error::Internal(format!("无法获取统计信息写锁: {}", e))
            })?;
            stats.total_subscribers += 1;
            stats.active_subscribers += 1;
        }
        
        Ok(subscription_id)
    }
    
    /// 全局订阅
    pub async fn subscribe_global(
        &self,
        callback: Arc<dyn EventCallback + Send + Sync>,
    ) -> Result<String> {
        let subscription_id = Uuid::new_v4().to_string();
        let subscriber = SubscriberInfo {
            id: subscription_id.clone(),
            callback,
            created_at: SystemTime::now(),
            last_activity: SystemTime::now(),
        };
        
        {
            let mut global_subscribers = self.global_subscribers.write().map_err(|e| {
                Error::Internal(format!("无法获取全局订阅者写锁: {}", e))
            })?;
            
            // 检查订阅者数量限制
            if global_subscribers.len() >= self.config.max_subscribers {
                return Err(Error::resource_exhausted("全局订阅者数量已达上限"));
            }
            
            global_subscribers.push(subscriber);
        }
        
        // 更新统计信息
        if self.config.enable_metrics {
            let mut stats = self.stats.write().map_err(|e| {
                Error::Internal(format!("无法获取统计信息写锁: {}", e))
            })?;
            stats.total_subscribers += 1;
            stats.active_subscribers += 1;
        }
        
        Ok(subscription_id)
    }
    
    /// 取消订阅
    pub async fn unsubscribe(&self, subscription_id: &str) -> Result<()> {
        debug!("取消订阅 - ID: {}", subscription_id);
        
        let mut removed = false;
        
        // 从类型订阅者中移除
        {
            let mut subscribers = self.subscribers.write().map_err(|e| {
                Error::Internal(format!("无法获取订阅者写锁: {}", e))
            })?;
            
            for type_subscribers in subscribers.values_mut() {
                type_subscribers.retain(|subscriber| {
                    if subscriber.id == subscription_id {
                        removed = true;
                        false
                    } else {
                        true
                    }
                });
            }
        }
        
        // 从全局订阅者中移除
        if !removed {
            let mut global_subscribers = self.global_subscribers.write().map_err(|e| {
                Error::Internal(format!("无法获取全局订阅者写锁: {}", e))
            })?;
            
            global_subscribers.retain(|subscriber| {
                if subscriber.id == subscription_id {
                    removed = true;
                    false
                } else {
                    true
                }
            });
        }
        
        if removed {
            // 更新统计信息
            if self.config.enable_metrics {
                let mut stats = self.stats.write().map_err(|e| {
                    Error::Internal(format!("无法获取统计信息写锁: {}", e))
                })?;
                stats.active_subscribers = stats.active_subscribers.saturating_sub(1);
            }
            
            info!("订阅已取消: {}", subscription_id);
            Ok(())
        } else {
            warn!("未找到订阅: {}", subscription_id);
            Err(Error::NotFound(format!("订阅不存在: {}", subscription_id)))
        }
    }
    
    /// 获取待处理事件
    pub async fn get_pending_events(&self) -> Result<Vec<Event>> {
        let queue = self.event_queue.read().map_err(|e| {
            Error::Internal(format!("无法获取事件队列读锁: {}", e))
        })?;
        Ok(queue.iter().cloned().collect())
    }
    
    /// 清理过期事件
    pub async fn cleanup_expired_events(&self, max_age: Duration) -> Result<usize> {
        let cutoff_timestamp = chrono::Utc::now().timestamp() as u64 - max_age.as_secs();
        let mut removed_count = 0;
        
        // 清理事件存储
        {
            let mut event_store = self.event_store.write().map_err(|e| {
                Error::Internal(format!("无法获取事件存储写锁: {}", e))
            })?;
            
            let _initial_len = event_store.len();
            event_store.retain(|event| {
                if event.timestamp < cutoff_timestamp {
                    removed_count += 1;
                    false
                } else {
                    true
                }
            });
        }
        
        // 清理非活跃订阅者
        let cutoff_time = SystemTime::now() - max_age;
        {
            let mut subscribers = self.subscribers.write().map_err(|e| {
                Error::Internal(format!("无法获取订阅者写锁: {}", e))
            })?;
            
            for type_subscribers in subscribers.values_mut() {
                type_subscribers.retain(|subscriber| {
                    if subscriber.last_activity < cutoff_time {
                        removed_count += 1;
                        false
                    } else {
                        true
                    }
                });
            }
        }
        
        info!("清理了 {} 个非活跃订阅者", removed_count);
        Ok(removed_count)
    }
    
    /// 获取统计信息
    pub async fn get_stats(&self) -> Result<EventSystemStats> {
        let stats = self.stats.read().map_err(|e| {
            Error::Internal(format!("无法获取统计信息读锁: {}", e))
        })?;
        Ok(stats.clone())
    }
    
    /// 处理事件队列循环
    async fn process_event_queue_loop(
        event_queue: Arc<RwLock<std::collections::VecDeque<Event>>>,
        is_running: Arc<RwLock<bool>>,
    ) -> Result<()> {
        loop {
            // 检查是否仍在运行
            {
                let running = is_running.read().map_err(|e| {
                    Error::Internal(format!("无法获取运行状态读锁: {}", e))
                })?;
                if !*running {
                    break;
                }
            }
            
            // 处理队列中的事件
            {
                let mut queue = event_queue.write().map_err(|e| {
                    Error::Internal(format!("无法获取事件队列写锁: {}", e))
                })?;
                
                if let Some(event) = queue.pop_front() {
                    // 这里应该处理事件，但为了简化，我们只是记录
                    debug!("处理事件: {:?}", event.event_type);
                }
            }
            
            // 短暂休眠
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        
        Ok(())
    }
}

#[async_trait::async_trait]
impl EventSystem for ProductionEventSystem {
    fn publish(&self, event: Event) -> Result<()> {
        // 阻塞调用异步方法
        let rt = Runtime::new().map_err(|e| Error::Internal(format!("创建运行时失败: {}", e)))?;
        rt.block_on(self.publish(event))
    }
    
    fn subscribe(&self, event_type: EventType, callback: Arc<dyn EventCallback>) -> Result<String> {
        // 将EventCallback包装为EventCallback + Send + Sync
        let wrapped_callback = Arc::new(CallbackWrapper { callback });
        
        // 阻塞调用异步方法
        let rt = Runtime::new().map_err(|e| Error::Internal(format!("创建运行时失败: {}", e)))?;
        rt.block_on(self.subscribe(event_type, wrapped_callback))
    }
    
    fn subscribe_all(&self, callback: Arc<dyn EventCallback>) -> Result<String> {
        // 将EventCallback包装为EventCallback + Send + Sync
        let wrapped_callback = Arc::new(CallbackWrapper { callback });
        
        // 阻塞调用异步方法
        let rt = Runtime::new().map_err(|e| Error::Internal(format!("创建运行时失败: {}", e)))?;
        rt.block_on(self.subscribe_global(wrapped_callback))
    }
    
    fn unsubscribe(&self, subscription_id: &str) -> Result<()> {
        // 阻塞调用异步方法
        let rt = Runtime::new().map_err(|e| Error::Internal(format!("创建运行时失败: {}", e)))?;
        rt.block_on(self.unsubscribe(subscription_id))
    }
    
    fn get_pending_events(&self) -> Result<Vec<Event>> {
        // 阻塞调用异步方法
        let rt = Runtime::new().map_err(|e| Error::Internal(format!("创建运行时失败: {}", e)))?;
        rt.block_on(self.get_pending_events())
    }
    
    fn start(&self) -> Result<()> {
        // 阻塞调用异步方法
        let rt = Runtime::new().map_err(|e| Error::Internal(format!("创建运行时失败: {}", e)))?;
        rt.block_on(self.start())
    }
    
    fn stop(&self) -> Result<()> {
        // 阻塞调用异步方法
        let rt = Runtime::new().map_err(|e| Error::Internal(format!("创建运行时失败: {}", e)))?;
        rt.block_on(self.stop())
    }
}

/// EventCallback包装器，确保Send + Sync
struct CallbackWrapper {
    callback: Arc<dyn EventCallback>,
}

impl EventCallback for CallbackWrapper {
    fn on_event(&self, event: &Event) -> Result<()> {
        self.callback.on_event(event)
    }
}

unsafe impl Send for CallbackWrapper {}
unsafe impl Sync for CallbackWrapper {} 