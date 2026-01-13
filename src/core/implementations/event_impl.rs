/// 事件驱动架构的完整生产级实现
/// 提供事件总线、发布订阅、事件存储等功能

use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use tokio::sync::{mpsc, broadcast};
use tokio::time::{sleep, Duration};

use crate::{Result, Error};
use crate::core::interfaces::event::*;

/// 生产级事件总线实现
pub struct ProductionEventBus {
    subscribers: Arc<RwLock<HashMap<String, Vec<Arc<dyn EventHandler>>>>>,
    event_queue: Arc<Mutex<mpsc::UnboundedSender<InternalEvent>>>,
    processor: Arc<EventProcessor>,
    storage: Option<Arc<dyn EventStorage>>,
    metrics: Arc<RwLock<EventMetrics>>,
    is_running: Arc<RwLock<bool>>,
}

impl ProductionEventBus {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let processor = Arc::new(EventProcessor::new(rx));
        
        Self {
            subscribers: Arc::new(RwLock::new(HashMap::new())),
            event_queue: Arc::new(Mutex::new(tx)),
            processor,
            storage: None,
            metrics: Arc::new(RwLock::new(EventMetrics::new())),
            is_running: Arc::new(RwLock::new(false)),
        }
    }

    pub fn with_storage(mut self, storage: Arc<dyn EventStorage>) -> Self {
        self.storage = Some(storage);
        self
    }

    pub async fn start(&self) -> Result<()> {
        *self.is_running.write().unwrap() = true;
        
        // 启动事件处理器
        self.processor.start(
            self.subscribers.clone(),
            self.storage.clone(),
            self.metrics.clone(),
            self.is_running.clone(),
        ).await?;

        Ok(())
    }

    pub async fn shutdown(&self) -> Result<()> {
        *self.is_running.write().unwrap() = false;
        self.processor.shutdown().await
    }

    fn update_metrics(&self, event_type: &str, processing_time_ms: u64, success: bool) {
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.total_events += 1;
            metrics.total_processing_time_ms += processing_time_ms;
            
            if success {
                metrics.successful_events += 1;
            } else {
                metrics.failed_events += 1;
            }
            
            metrics.events_by_type
                .entry(event_type.to_string())
                .and_modify(|e| *e += 1)
                .or_insert(1);
        }
    }
}

#[async_trait]
impl EventBus for ProductionEventBus {
    async fn publish(&self, event: Event) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        // 创建内部事件
        let internal_event = InternalEvent {
            id: Uuid::new_v4().to_string(),
            event,
            published_at: Utc::now(),
            processing_attempts: 0,
        };

        // 发送到处理队列
        if let Ok(sender) = self.event_queue.lock() {
            sender.send(internal_event.clone())
                .map_err(|e| Error::InvalidInput(e.to_string()))?;
        }

        // 存储事件（如果配置了存储）
        if let Some(ref storage) = self.storage {
            storage.store_event(&internal_event).await?;
        }

        let processing_time = start_time.elapsed().as_millis() as u64;
        self.update_metrics(&internal_event.event.event_type, processing_time, true);

        Ok(())
    }

    async fn subscribe(&self, event_type: &str, handler: Arc<dyn EventHandler>) -> Result<()> {
        let mut subscribers = self.subscribers.write().unwrap();
        let handlers = subscribers.entry(event_type.to_string()).or_insert_with(Vec::new);
        handlers.push(handler);
        Ok(())
    }

    async fn unsubscribe(&self, event_type: &str, handler_id: &str) -> Result<()> {
        let mut subscribers = self.subscribers.write().unwrap();
        if let Some(handlers) = subscribers.get_mut(event_type) {
            handlers.retain(|h| h.handler_id() != handler_id);
        }
        Ok(())
    }

    async fn get_metrics(&self) -> Result<EventBusMetrics> {
        let metrics = self.metrics.read().unwrap();
        Ok(EventBusMetrics {
            total_events: metrics.total_events,
            successful_events: metrics.successful_events,
            failed_events: metrics.failed_events,
            average_processing_time_ms: metrics.average_processing_time_ms(),
            events_by_type: metrics.events_by_type.clone(),
        })
    }
}

/// 生产级事件发布器实现
pub struct ProductionEventPublisher {
    event_bus: Arc<ProductionEventBus>,
    source: String,
}

impl ProductionEventPublisher {
    pub fn new(event_bus: Arc<ProductionEventBus>, source: String) -> Self {
        Self {
            event_bus,
            source,
        }
    }
}

#[async_trait]
impl EventPublisher for ProductionEventPublisher {
    async fn publish_event(&self, event_type: &str, data: EventData) -> Result<()> {
        let event = Event {
            event_type: event_type.to_string(),
            source: self.source.clone(),
            data,
            timestamp: Utc::now(),
            correlation_id: None,
            metadata: HashMap::new(),
        };

        self.event_bus.publish(event).await
    }

    async fn publish_event_with_correlation(&self, event_type: &str, data: EventData, correlation_id: &str) -> Result<()> {
        let event = Event {
            event_type: event_type.to_string(),
            source: self.source.clone(),
            data,
            timestamp: Utc::now(),
            correlation_id: Some(correlation_id.to_string()),
            metadata: HashMap::new(),
        };

        self.event_bus.publish(event).await
    }

    async fn publish_domain_event(&self, domain: &str, event_type: &str, aggregate_id: &str, data: EventData) -> Result<()> {
        let mut metadata = HashMap::new();
        metadata.insert("domain".to_string(), domain.to_string());
        metadata.insert("aggregate_id".to_string(), aggregate_id.to_string());

        let event = Event {
            event_type: format!("{}.{}", domain, event_type),
            source: self.source.clone(),
            data,
            timestamp: Utc::now(),
            correlation_id: None,
            metadata,
        };

        self.event_bus.publish(event).await
    }
}

/// 生产级事件订阅器实现
pub struct ProductionEventSubscriber {
    handler: Arc<dyn EventHandler>,
    event_types: Vec<String>,
    is_active: Arc<RwLock<bool>>,
}

impl ProductionEventSubscriber {
    pub fn new(handler: Arc<dyn EventHandler>, event_types: Vec<String>) -> Self {
        Self {
            handler,
            event_types,
            is_active: Arc::new(RwLock::new(true)),
        }
    }

    pub async fn subscribe_to_bus(&self, event_bus: &ProductionEventBus) -> Result<()> {
        for event_type in &self.event_types {
            event_bus.subscribe(event_type, self.handler.clone()).await?;
        }
        Ok(())
    }

    pub fn set_active(&self, active: bool) {
        *self.is_active.write().unwrap() = active;
    }

    pub fn is_active(&self) -> bool {
        *self.is_active.read().unwrap()
    }
}

/// 事件处理器
struct EventProcessor {
    receiver: Arc<Mutex<mpsc::UnboundedReceiver<InternalEvent>>>,
}

impl EventProcessor {
    fn new(receiver: mpsc::UnboundedReceiver<InternalEvent>) -> Self {
        Self {
            receiver: Arc::new(Mutex::new(receiver)),
        }
    }

    async fn start(
        &self,
        subscribers: Arc<RwLock<HashMap<String, Vec<Arc<dyn EventHandler>>>>>,
        storage: Option<Arc<dyn EventStorage>>,
        metrics: Arc<RwLock<EventMetrics>>,
        is_running: Arc<RwLock<bool>>,
    ) -> Result<()> {
        let receiver = self.receiver.clone();
        
        tokio::spawn(async move {
            while *is_running.read().unwrap() {
                if let Ok(mut rx) = receiver.try_lock() {
                    while let Some(mut internal_event) = rx.recv().await {
                        let start_time = std::time::Instant::now();
                        let mut success = true;

                        // 获取订阅者
                        let handlers = {
                            let subs = subscribers.read().unwrap();
                            subs.get(&internal_event.event.event_type).cloned().unwrap_or_default()
                        };

                        // 处理事件
                        for handler in handlers {
                            match handler.handle(&internal_event.event).await {
                                Ok(_) => {
                                    log::debug!("事件处理成功: {} by {}", 
                                        internal_event.event.event_type, 
                                        handler.handler_id()
                                    );
                                },
                                Err(e) => {
                                    log::error!("事件处理失败: {} by {}: {}", 
                                        internal_event.event.event_type, 
                                        handler.handler_id(),
                                        e
                                    );
                                    success = false;
                                }
                            }
                        }

                        // 更新处理状态
                        internal_event.processing_attempts += 1;

                        // 更新存储中的事件状态
                        if let Some(ref storage) = storage {
                            if success {
                                let _ = storage.mark_event_processed(&internal_event.id).await;
                            } else {
                                let _ = storage.mark_event_failed(&internal_event.id, &format!("处理失败，尝试次数: {}", internal_event.processing_attempts)).await;
                            }
                        }

                        // 更新指标
                        let processing_time = start_time.elapsed().as_millis() as u64;
                        if let Ok(mut m) = metrics.write() {
                            m.total_processing_time_ms += processing_time;
                            if success {
                                m.successful_events += 1;
                            } else {
                                m.failed_events += 1;
                            }
                        }
                    }
                }
                
                sleep(Duration::from_millis(10)).await;
            }
        });

        Ok(())
    }

    async fn shutdown(&self) -> Result<()> {
        // 处理剩余的事件
        if let Ok(mut rx) = self.receiver.try_lock() {
            rx.close();
        }
        Ok(())
    }
}

/// 内存事件存储实现
pub struct InMemoryEventStorage {
    events: Arc<RwLock<HashMap<String, StoredEvent>>>,
}

impl InMemoryEventStorage {
    pub fn new() -> Self {
        Self {
            events: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl EventStorage for InMemoryEventStorage {
    async fn store_event(&self, event: &InternalEvent) -> Result<()> {
        let stored_event = StoredEvent {
            id: event.id.clone(),
            event: event.event.clone(),
            published_at: event.published_at,
            processing_attempts: event.processing_attempts,
            status: EventStatus::Pending,
            processed_at: None,
            error_message: None,
        };

        let mut events = self.events.write().unwrap();
        events.insert(event.id.clone(), stored_event);
        Ok(())
    }

    async fn get_event(&self, event_id: &str) -> Result<Option<StoredEvent>> {
        let events = self.events.read().unwrap();
        Ok(events.get(event_id).cloned())
    }

    async fn mark_event_processed(&self, event_id: &str) -> Result<()> {
        let mut events = self.events.write().unwrap();
        if let Some(event) = events.get_mut(event_id) {
            event.status = EventStatus::Processed;
            event.processed_at = Some(Utc::now());
        }
        Ok(())
    }

    async fn mark_event_failed(&self, event_id: &str, error: &str) -> Result<()> {
        let mut events = self.events.write().unwrap();
        if let Some(event) = events.get_mut(event_id) {
            event.status = EventStatus::Failed;
            event.error_message = Some(error.to_string());
            event.processed_at = Some(Utc::now());
        }
        Ok(())
    }

    async fn get_events_by_type(&self, event_type: &str) -> Result<Vec<StoredEvent>> {
        let events = self.events.read().unwrap();
        let filtered_events: Vec<StoredEvent> = events
            .values()
            .filter(|e| e.event.event_type == event_type)
            .cloned()
            .collect();
        Ok(filtered_events)
    }

    async fn get_pending_events(&self) -> Result<Vec<StoredEvent>> {
        let events = self.events.read().unwrap();
        let pending_events: Vec<StoredEvent> = events
            .values()
            .filter(|e| e.status == EventStatus::Pending)
            .cloned()
            .collect();
        Ok(pending_events)
    }
}

/// 简单事件处理器实现
pub struct SimpleEventHandler {
    id: String,
    handler_fn: Box<dyn Fn(&Event) -> Result<()> + Send + Sync>,
}

impl SimpleEventHandler {
    pub fn new<F>(id: String, handler_fn: F) -> Self 
    where
        F: Fn(&Event) -> Result<()> + Send + Sync + 'static,
    {
        Self {
            id,
            handler_fn: Box::new(handler_fn),
        }
    }
}

#[async_trait]
impl EventHandler for SimpleEventHandler {
    async fn handle(&self, event: &Event) -> Result<()> {
        (self.handler_fn)(event)
    }

    fn handler_id(&self) -> &str {
        &self.id
    }
}

/// 批量事件处理器
pub struct BatchEventHandler {
    id: String,
    batch_size: usize,
    batch_timeout: Duration,
    events: Arc<Mutex<Vec<Event>>>,
    processor: Box<dyn Fn(&[Event]) -> Result<()> + Send + Sync>,
}

impl BatchEventHandler {
    pub fn new<F>(id: String, batch_size: usize, batch_timeout: Duration, processor: F) -> Self 
    where
        F: Fn(&[Event]) -> Result<()> + Send + Sync + 'static,
    {
        Self {
            id,
            batch_size,
            batch_timeout,
            events: Arc::new(Mutex::new(Vec::new())),
            processor: Box::new(processor),
        }
    }

    async fn process_batch(&self) -> Result<()> {
        let events_to_process = {
            let mut events = self.events.lock().unwrap();
            if events.is_empty() {
                return Ok(());
            }
            
            let batch = events.drain(..).collect::<Vec<_>>();
            batch
        };

        if !events_to_process.is_empty() {
            (self.processor)(&events_to_process)?;
        }

        Ok(())
    }
}

#[async_trait]
impl EventHandler for BatchEventHandler {
    async fn handle(&self, event: &Event) -> Result<()> {
        let should_process = {
            let mut events = self.events.lock().unwrap();
            events.push(event.clone());
            events.len() >= self.batch_size
        };

        if should_process {
            self.process_batch().await?;
        }

        Ok(())
    }

    fn handler_id(&self) -> &str {
        &self.id
    }
}

/// 内部事件
#[derive(Debug, Clone)]
struct InternalEvent {
    id: String,
    event: Event,
    published_at: DateTime<Utc>,
    processing_attempts: u32,
}

/// 存储的事件
#[derive(Debug, Clone)]
struct StoredEvent {
    id: String,
    event: Event,
    published_at: DateTime<Utc>,
    processing_attempts: u32,
    status: EventStatus,
    processed_at: Option<DateTime<Utc>>,
    error_message: Option<String>,
}

/// 事件状态
#[derive(Debug, Clone, PartialEq)]
enum EventStatus {
    Pending,
    Processed,
    Failed,
}

/// 事件指标
#[derive(Debug, Clone)]
struct EventMetrics {
    total_events: u64,
    successful_events: u64,
    failed_events: u64,
    total_processing_time_ms: u64,
    events_by_type: HashMap<String, u64>,
}

impl EventMetrics {
    fn new() -> Self {
        Self {
            total_events: 0,
            successful_events: 0,
            failed_events: 0,
            total_processing_time_ms: 0,
            events_by_type: HashMap::new(),
        }
    }

    fn average_processing_time_ms(&self) -> f64 {
        if self.total_events > 0 {
            self.total_processing_time_ms as f64 / self.total_events as f64
        } else {
            0.0
        }
    }
}

/// 事件存储接口
#[async_trait]
pub trait EventStorage: Send + Sync {
    async fn store_event(&self, event: &InternalEvent) -> Result<()>;
    async fn get_event(&self, event_id: &str) -> Result<Option<StoredEvent>>;
    async fn mark_event_processed(&self, event_id: &str) -> Result<()>;
    async fn mark_event_failed(&self, event_id: &str, error: &str) -> Result<()>;
    async fn get_events_by_type(&self, event_type: &str) -> Result<Vec<StoredEvent>>;
    async fn get_pending_events(&self) -> Result<Vec<StoredEvent>>;
} 