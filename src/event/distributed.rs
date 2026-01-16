use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use log::{debug, info, warn, error};
use serde::{Serialize, Deserialize};
use tokio::sync::mpsc::{Sender, Receiver};

use crate::event::{Event, EventType, EventSystem, EventCallback};
use crate::network::{NetworkManager, NodeInfo, NodeRole as NetworkNodeRole};
use crate::Result;
use crate::Error;

/// 分布式事件系统
/// 
/// 支持跨节点的分布式事件传播和处理
#[derive(Clone)]
pub struct DistributedEventSystem {
    /// 本地节点ID
    node_id: String,
    /// 本地内存事件系统
    local_system: crate::event::memory::MemoryEventSystem,
    /// 网络管理器
    network: Arc<NetworkManager>,
    /// 远程订阅信息
    remote_subscriptions: Arc<RwLock<HashMap<String, Vec<RemoteSubscription>>>>,
    /// 本地订阅到远程节点的映射
    local_to_remote: Arc<RwLock<HashMap<String, Vec<RemoteSubscriptionTarget>>>>,
    /// 事件缓存
    event_cache: Arc<RwLock<EventCache>>,
    /// 系统配置
    config: DistributedEventConfig,
    /// 运行状态
    is_running: Arc<Mutex<bool>>,
}

/// 分布式事件系统配置
#[derive(Debug, Clone)]
pub struct DistributedEventConfig {
    /// 事件传播模式
    pub propagation_mode: EventPropagationMode,
    /// 事件缓存大小
    pub cache_size: usize,
    /// 事件重试次数
    pub retry_count: usize,
    /// 重试间隔
    pub retry_interval: Duration,
    /// 心跳间隔
    pub heartbeat_interval: Duration,
    /// 消息超时时间
    pub message_timeout: Duration,
    /// 是否启用消息确认
    pub enable_ack: bool,
    /// 是否启用压缩
    pub enable_compression: bool,
}

impl Default for DistributedEventConfig {
    fn default() -> Self {
        Self {
            propagation_mode: EventPropagationMode::AllNodes,
            cache_size: 1000,
            retry_count: 3,
            retry_interval: Duration::from_millis(500),
            heartbeat_interval: Duration::from_secs(30),
            message_timeout: Duration::from_secs(5),
            enable_ack: true,
            enable_compression: true,
        }
    }
}

/// 事件传播模式
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventPropagationMode {
    /// 只发送到订阅该事件类型的节点
    SubscribedOnly,
    /// 发送到所有节点
    AllNodes,
    /// 只发送到指定角色的节点
    ByRole(NodeRole),
    /// 只发送到指定组的节点
    ByGroup(String),
}

/// 节点角色
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeRole {
    /// 主节点
    Master,
    /// 从节点
    Slave,
    /// 工作节点
    Worker,
    /// 协调节点
    Coordinator,
    /// 网关节点
    Gateway,
    /// 存储节点
    Storage,
    /// 自定义角色
    Custom(u32),
}

/// 远程订阅信息
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RemoteSubscription {
    /// 远程节点ID
    node_id: String,
    /// 订阅ID
    subscription_id: String,
    /// 事件类型
    event_type: EventType,
    /// 订阅创建时间
    created_at: SystemTime,
    /// 上次活跃时间
    last_active: SystemTime,
    /// 过滤条件
    filter: Option<EventFilterDesc>,
}

/// 事件过滤器描述
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EventFilterDesc {
    /// 过滤器类型
    filter_type: String,
    /// 过滤器参数
    params: HashMap<String, String>,
}

/// 远程订阅目标
#[derive(Debug, Clone)]
struct RemoteSubscriptionTarget {
    /// 远程节点ID
    node_id: String,
    /// 远程订阅ID
    remote_subscription_id: String,
    /// 事件类型
    event_type: EventType,
}

/// 事件缓存
#[derive(Debug, Default)]
struct EventCache {
    /// 按ID缓存事件
    by_id: HashMap<String, Event>,
    /// 按类型缓存最近的事件
    by_type: HashMap<EventType, Vec<String>>,
    /// 最大缓存大小
    max_size: usize,
    /// 待确认的事件
    pending_acks: HashMap<String, PendingEventAck>,
}

/// 待确认的事件
#[derive(Debug)]
struct PendingEventAck {
    /// 事件ID
    event_id: String,
    /// 目标节点
    target_nodes: HashSet<String>,
    /// 已确认的节点
    acked_nodes: HashSet<String>,
    /// 发送时间
    sent_at: Instant,
    /// 重试次数
    retry_count: usize,
}

/// 网络消息类型
#[derive(Debug, Clone, Serialize, Deserialize)]
enum EventNetworkMessage {
    /// 事件通知
    EventNotification {
        /// 事件
        event: Event,
        /// 源节点
        source_node: String,
        /// 消息ID
        message_id: String,
    },
    /// 事件确认
    EventAck {
        /// 事件ID
        event_id: String,
        /// 消息ID
        message_id: String,
        /// 确认节点
        node_id: String,
    },
    /// 订阅请求
    SubscriptionRequest {
        /// 订阅节点
        node_id: String,
        /// 订阅ID
        subscription_id: String,
        /// 事件类型
        event_type: EventType,
        /// 过滤条件
        filter: Option<EventFilterDesc>,
    },
    /// 订阅确认
    SubscriptionAck {
        /// 订阅ID
        subscription_id: String,
        /// 确认节点
        node_id: String,
        /// 结果
        success: bool,
        /// 错误信息
        error: Option<String>,
    },
    /// 取消订阅
    UnsubscriptionRequest {
        /// 节点ID
        node_id: String,
        /// 订阅ID
        subscription_id: String,
    },
    /// 心跳
    Heartbeat {
        /// 节点ID
        node_id: String,
        /// 时间戳
        timestamp: SystemTime,
    },
}

impl DistributedEventSystem {
    /// 创建新的分布式事件系统
    pub fn new(
        node_id: String,
        network: Arc<NetworkManager>,
        config: Option<DistributedEventConfig>
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        let local_system = crate::event::memory::MemoryEventSystem::new(config.cache_size);
        
        let event_cache = EventCache {
            by_id: HashMap::new(),
            by_type: HashMap::new(),
            max_size: config.cache_size,
            pending_acks: HashMap::new(),
        };
        
        Ok(Self {
            node_id,
            local_system,
            network,
            remote_subscriptions: Arc::new(RwLock::new(HashMap::new())),
            local_to_remote: Arc::new(RwLock::new(HashMap::new())),
            event_cache: Arc::new(RwLock::new(event_cache)),
            config,
            is_running: Arc::new(Mutex::new(false)),
        })
    }
    
    /// 启动分布式事件系统
    pub fn start(&self) -> Result<()> {
        let mut is_running = self.is_running.lock().map_err(|e| {
            Error::Internal(format!("无法获取运行状态锁: {}", e))
        })?;
        
        if *is_running {
            return Ok(());
        }
        
        // 启动本地事件系统
        self.local_system.start()?;
        
        // 注册网络消息处理
        let node_id = self.node_id.clone();
        let remote_subscriptions = self.remote_subscriptions.clone();
        let local_to_remote = self.local_to_remote.clone();
        let event_cache = self.event_cache.clone();
        let config = self.config.clone();
        let network = self.network.clone();
        let is_running_clone = self.is_running.clone();
        
        // 启动消息处理线程
        std::thread::spawn(move || {
            Self::message_handler(
                node_id,
                network,
                remote_subscriptions,
                local_to_remote,
                event_cache,
                config,
                is_running_clone
            );
        });
        
        // 启动心跳线程
        let node_id = self.node_id.clone();
        let network = self.network.clone();
        let heartbeat_interval = self.config.heartbeat_interval;
        let is_running_clone = self.is_running.clone();
        
        // 避免在 tokio::spawn 中捕获非 Send 互斥锁守卫：改为后台线程驱动异步 runtime
        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("runtime");
            rt.block_on(async move {
                Self::heartbeat_sender(
                    node_id,
                    network,
                    heartbeat_interval,
                    is_running_clone
                ).await;
            });
        });
        
        // 更新运行状态
        *is_running = true;
        info!("DistributedEventSystem started: node_id={}", self.node_id);
        
        Ok(())
    }
    
    /// 消息处理循环
    fn message_handler(
        node_id: String,
        network: Arc<NetworkManager>,
        remote_subscriptions: Arc<RwLock<HashMap<String, Vec<RemoteSubscription>>>>,
        local_to_remote: Arc<RwLock<HashMap<String, Vec<RemoteSubscriptionTarget>>>>,
        event_cache: Arc<RwLock<EventCache>>,
        config: DistributedEventConfig,
        is_running: Arc<Mutex<bool>>
    ) {
        // 创建消息接收通道（显式使用 Sender/Receiver 类型以行使导入）
        let (tx, mut rx): (Sender<EventNetworkMessage>, Receiver<EventNetworkMessage>) =
            tokio::sync::mpsc::channel(100);
        
        // 注册消息处理器
        if let Err(e) = network.register_handler("event_system", move |data: &[u8], _source: &str| {
            let message: EventNetworkMessage = match bincode::deserialize(&data) {
                Ok(msg) => msg,
                Err(e) => {
                    error!("无法解析事件系统消息: {}", e);
                    return;
                }
            };
            
            if let Err(e) = tx.blocking_send(message) {
                error!("无法发送消息到事件处理队列: {}", e);
            }
        }) {
            error!("无法注册事件系统消息处理器: {}", e);
        }
        
        // 消息处理循环
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            while let Some(message) = rx.recv().await {
                // 检查是否应该停止 - 在await之前检查并释放锁
                {
                    let running = is_running.lock().unwrap();
                    if !*running {
                        break;
                    }
                } // 锁在这里被释放
                
                match message {
                    EventNetworkMessage::EventNotification { event, source_node, message_id } => {
                        Self::handle_event_notification(
                            &node_id,
                            &network,
                            &remote_subscriptions,
                            &event_cache,
                            event,
                            source_node,
                            message_id,
                            &config
                        ).await;
                    },
                    EventNetworkMessage::EventAck { event_id, message_id, node_id: source_node } => {
                        Self::handle_event_ack(
                            &event_cache,
                            event_id,
                            message_id,
                            source_node
                        ).await;
                    },
                    EventNetworkMessage::SubscriptionRequest { node_id: source_node, subscription_id, event_type, filter } => {
                        Self::handle_subscription_request(
                            &node_id,
                            &network,
                            &remote_subscriptions,
                            source_node,
                            subscription_id,
                            event_type,
                            filter
                        ).await;
                    },
                    EventNetworkMessage::SubscriptionAck { subscription_id, node_id: source_node, success, error } => {
                        Self::handle_subscription_ack(
                            &local_to_remote,
                            subscription_id,
                            source_node,
                            success,
                            error
                        ).await;
                    },
                    EventNetworkMessage::UnsubscriptionRequest { node_id: source_node, subscription_id } => {
                        Self::handle_unsubscription_request(
                            &remote_subscriptions,
                            source_node,
                            subscription_id
                        ).await;
                    },
                    EventNetworkMessage::Heartbeat { node_id: source_node, timestamp } => {
                        Self::handle_heartbeat(
                            &remote_subscriptions,
                            source_node,
                            timestamp
                        ).await;
                    },
                }
            }
        });
    }
    
    /// 心跳发送器
    async fn heartbeat_sender(
        node_id: String,
        network: Arc<NetworkManager>,
        heartbeat_interval: Duration,
        is_running: Arc<Mutex<bool>>
    ) {
        loop {
            // 检查是否应该停止
            let running = is_running.lock().unwrap();
            if !*running {
                break;
            }
            drop(running);
            
            // 发送心跳
            let message = EventNetworkMessage::Heartbeat {
                node_id: node_id.clone(),
                timestamp: SystemTime::now(),
            };
            
            let data = match bincode::serialize(&message) {
                Ok(data) => data,
                Err(e) => {
                    error!("序列化心跳消息失败: {}", e);
                    continue;
                }
            };
            
            if let Err(e) = network.broadcast_message("event_system", &data).await {
                error!("广播心跳消息失败: {}", e);
            }
            
            // 等待下一个心跳周期
            tokio::time::sleep(heartbeat_interval).await;
        }
    }
    
    /// 处理事件通知
    async fn handle_event_notification(
        node_id: &str,
        network: &NetworkManager,
        remote_subscriptions: &RwLock<HashMap<String, Vec<RemoteSubscription>>>,
        _event_cache: &RwLock<EventCache>,
        event: Event,
        source_node: String,
        message_id: String,
        config: &DistributedEventConfig
    ) {
        // 先获取event_type的字符串表示，避免借用移动错误
        let event_type_str = event.event_type.to_string();
        
        // 向本地订阅者分发事件
        // 生产级实现：直接调用本地内存事件系统的publish方法
        if let Some(local_system) = network.get_local_event_system() {
            if let Err(e) = local_system.publish(event.clone()) {
                warn!("本地事件分发失败: {}", e);
            }
        }
        
        // 发送确认消息
        if config.enable_ack {
            let ack_message = EventNetworkMessage::EventAck {
                event_id: event.id.clone(),
                message_id: message_id.clone(),
                node_id: node_id.to_string(),
            };
            
            if let Ok(data) = bincode::serialize(&ack_message) {
                if let Err(e) = network.send_to(&source_node, "event_system", &data).await {
                    warn!("发送事件确认失败: {}", e);
                }
            }
        }
        
        // 查找订阅了此事件类型的远程节点，并通知它们
        // 修复：避免在持有读锁时执行 await，先收集目标节点与过滤条件，再异步发送
        let targets: Vec<(String, Option<EventFilterDesc>)> = {
            match remote_subscriptions.read() {
                Ok(subs) => subs
                    .get(&event_type_str)
                    .map(|list| list.iter()
                        .filter(|sub| sub.node_id != source_node)
                        .map(|sub| (sub.node_id.clone(), sub.filter.clone()))
                        .collect())
                    .unwrap_or_default(),
                Err(_) => Vec::new(),
            }
        };

        for (target_node, filter_desc_opt) in targets {
            // 根据订阅的filter字段进行事件过滤（浅过滤）
            let mut should_send = true;
            if let Some(filter_desc) = &filter_desc_opt {
                for (k, v) in &filter_desc.params {
                    if let Some(ev) = event.data.get(k) {
                        if ev != v {
                            should_send = false;
                            break;
                        }
                    } else {
                        should_send = false;
                        break;
                    }
                }
            }
            if !should_send { continue; }

            let notification = EventNetworkMessage::EventNotification {
                event: event.clone(),
                source_node: node_id.to_string(),
                message_id: format!("msg-{}", uuid::Uuid::new_v4()),
            };

            if let Ok(data) = bincode::serialize(&notification) {
                if let Err(e) = network.send_to(&target_node, "event_system", &data).await {
                    warn!("发送事件通知到节点 {} 失败: {}", target_node, e);
                }
            }
        }
    }
    
    /// 处理事件确认
    async fn handle_event_ack(
        event_cache: &RwLock<EventCache>,
        _event_id: String,
        message_id: String,
        source_node: String
    ) {
        let mut cache = event_cache.write().unwrap();
        
        if let Some(pending) = cache.pending_acks.get_mut(&message_id) {
            pending.acked_nodes.insert(source_node);
            
            // 检查是否所有节点都已确认
            if pending.acked_nodes.len() == pending.target_nodes.len() {
                // 所有节点都已确认，移除待确认记录
                cache.pending_acks.remove(&message_id);
            }
        }
    }
    
    /// 处理订阅请求
    async fn handle_subscription_request(
        node_id: &str,
        network: &NetworkManager,
        remote_subscriptions: &RwLock<HashMap<String, Vec<RemoteSubscription>>>,
        source_node: String,
        subscription_id: String,
        event_type: EventType,
        filter: Option<EventFilterDesc>
    ) {
        // 先获取event_type的字符串表示，避免借用移动错误
        let event_type_str = event_type.to_string();
        
        let subscription = RemoteSubscription {
            node_id: source_node.clone(),
            subscription_id: subscription_id.clone(),
            event_type,
            created_at: SystemTime::now(),
            last_active: SystemTime::now(),
            filter,
        };
        
        // 添加远程订阅
        {
            let mut subs = remote_subscriptions.write().unwrap();
            subs.entry(event_type_str)
                .or_insert_with(Vec::new)
                .push(subscription);
        }
        
        // 发送确认消息
        let ack_message = EventNetworkMessage::SubscriptionAck {
            subscription_id,
            node_id: node_id.to_string(),
            success: true,
            error: None,
        };
        
        if let Ok(data) = bincode::serialize(&ack_message) {
            if let Err(e) = network.send_to(&source_node, "event_system", &data).await {
                warn!("发送订阅确认失败: {}", e);
            }
        }
    }
    
    /// 处理订阅确认
    async fn handle_subscription_ack(
        local_to_remote: &RwLock<HashMap<String, Vec<RemoteSubscriptionTarget>>>,
        subscription_id: String,
        source_node: String,
        success: bool,
        error: Option<String>
    ) {
        if !success {
            if let Some(err) = error {
                warn!("远程订阅失败 (节点={}, 订阅ID={}): {}", 
                      source_node, subscription_id, err);
            } else {
                warn!("远程订阅失败 (节点={}, 订阅ID={})", 
                      source_node, subscription_id);
            }
            
            // 移除失败的订阅
            let mut local_remote = local_to_remote.write().unwrap();
            for targets in local_remote.values_mut() {
                targets.retain(|t| !(t.node_id == source_node && 
                                   t.remote_subscription_id == subscription_id));
            }
        } else {
            debug!("远程订阅成功 (节点={}, 订阅ID={})", 
                  source_node, subscription_id);
        }
    }
    
    /// 处理取消订阅请求
    async fn handle_unsubscription_request(
        remote_subscriptions: &RwLock<HashMap<String, Vec<RemoteSubscription>>>,
        source_node: String,
        subscription_id: String
    ) {
        let mut subs = remote_subscriptions.write().unwrap();
        
        for (_, type_subs) in subs.iter_mut() {
            type_subs.retain(|s| !(s.node_id == source_node && 
                                 s.subscription_id == subscription_id));
        }
    }
    
    /// 处理心跳
    async fn handle_heartbeat(
        remote_subscriptions: &RwLock<HashMap<String, Vec<RemoteSubscription>>>,
        source_node: String,
        timestamp: SystemTime
    ) {
        let mut subs = remote_subscriptions.write().unwrap();
        
        for (_, type_subs) in subs.iter_mut() {
            for sub in type_subs.iter_mut() {
                if sub.node_id == source_node {
                    sub.last_active = timestamp;
                }
            }
        }
    }
    
    /// 停止分布式事件系统
    pub fn stop(&self) -> Result<()> {
        let mut is_running = self.is_running.lock().map_err(|e| {
            Error::Internal(format!("无法获取运行状态锁: {}", e))
        })?;
        
        if !*is_running {
            return Ok(());
        }
        
        // 更新运行状态
        *is_running = false;
        
        // 停止本地事件系统
        self.local_system.stop()?;
        
        // 取消所有远程订阅
        self.cancel_all_remote_subscriptions()?;
        
        Ok(())
    }
    
    /// 取消所有远程订阅
    fn cancel_all_remote_subscriptions(&self) -> Result<()> {
        let local_remote = self.local_to_remote.read().map_err(|e| {
            Error::Internal(format!("无法获取本地到远程映射读锁: {}", e))
        })?;
        
        for targets in local_remote.values() {
            for target in targets {
                let message = EventNetworkMessage::UnsubscriptionRequest {
                    node_id: self.node_id.clone(),
                    subscription_id: target.remote_subscription_id.clone(),
                };
                
                if let Ok(data) = bincode::serialize(&message) {
                    // 使用block_on来处理异步调用
                    if let Err(e) = tokio::runtime::Handle::current().block_on(async {
                        self.network.send_to(&target.node_id, "event_system", &data).await
                    }) {
                        warn!("发送取消订阅请求失败: {}", e);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// 获取所有远程订阅
    pub fn get_remote_subscriptions(&self) -> Result<HashMap<String, Vec<RemoteSubscription>>> {
        let subs = self.remote_subscriptions.read().map_err(|e| {
            Error::Internal(format!("无法获取远程订阅读锁: {}", e))
        })?;
        
        Ok(subs.clone())
    }
    
    /// 订阅远程事件
    pub fn subscribe_remote(&self, node_id: &str, event_type: EventType) -> Result<String> {
        let subscription_id = format!("remote-{}", uuid::Uuid::new_v4());
        
        let message = EventNetworkMessage::SubscriptionRequest {
            node_id: self.node_id.clone(),
            subscription_id: subscription_id.clone(),
            event_type,
            filter: None,
        };
        
        let data = bincode::serialize(&message)?;
        
        // 使用block_on来处理异步调用
        tokio::runtime::Handle::current().block_on(async {
            self.network.send_to(node_id, "event_system", &data).await
        })?;
        
        // 记录本地到远程的映射
        let event_type_str = event_type.to_string();
        let target = RemoteSubscriptionTarget {
            node_id: node_id.to_string(),
            remote_subscription_id: subscription_id.clone(),
            event_type,
        };
        
        let mut local_remote = self.local_to_remote.write().map_err(|e| {
            Error::Internal(format!("无法获取本地到远程映射写锁: {}", e))
        })?;
        
        local_remote.entry(event_type_str)
            .or_insert_with(Vec::new)
            .push(target);
            
        Ok(subscription_id)
    }
    
    /// 取消订阅远程事件
    pub fn unsubscribe_remote(&self, subscription_id: &str) -> Result<()> {
        let mut local_remote = self.local_to_remote.write().map_err(|e| {
            Error::Internal(format!("无法获取本地到远程映射写锁: {}", e))
        })?;
        
        let mut found = false;
        let mut target_node = String::new();
        
        for targets in local_remote.values_mut() {
            let before_len = targets.len();
            let mut remote_sub_id = String::new();
            
            for target in targets.iter() {
                if target.remote_subscription_id == subscription_id {
                    target_node = target.node_id.clone();
                    remote_sub_id = target.remote_subscription_id.clone();
                    break;
                }
            }
            
            if !remote_sub_id.is_empty() {
                targets.retain(|t| t.remote_subscription_id != subscription_id);
                found = targets.len() < before_len;
                break;
            }
        }
        
        if found && !target_node.is_empty() {
            // 向远程节点发送取消订阅请求
            let message = EventNetworkMessage::UnsubscriptionRequest {
                node_id: self.node_id.clone(),
                subscription_id: subscription_id.to_string(),
            };
            
            if let Ok(data) = bincode::serialize(&message) {
                // 使用block_on来处理异步调用
                if let Err(e) = tokio::runtime::Handle::current().block_on(async {
                    self.network.send_to(&target_node, "event_system", &data).await
                }) {
                    warn!("发送取消订阅请求失败: {}", e);
                }
            }
            
            Ok(())
        } else {
            Err(Error::NotFound(format!("未找到订阅ID: {}", subscription_id)))
        }
    }
}

impl EventSystem for DistributedEventSystem {
    fn publish(&self, event: Event) -> Result<()> {
        // 先发布到本地事件系统
        self.local_system.publish(event.clone())?;
        
        // 获取目标节点
        let target_nodes: Vec<String> = match self.config.propagation_mode {
            EventPropagationMode::SubscribedOnly => {
                // 只发送给订阅了该事件类型的节点
                let subs = self.remote_subscriptions.read().map_err(|e| {
                    Error::Internal(format!("无法获取远程订阅读锁: {}", e))
                })?;
                
                let mut nodes = HashSet::new();
                if let Some(type_subs) = subs.get(&event.event_type.to_string()) {
                    for sub in type_subs {
                        nodes.insert(sub.node_id.clone());
                    }
                }
                nodes.into_iter().collect()
            },
            EventPropagationMode::AllNodes => {
                // 发送给所有节点
                let nodes: Vec<NodeInfo> = tokio::runtime::Handle::current().block_on(async {
                    self.network.get_all_nodes().await
                });
                nodes.into_iter()
                    .map(|n| n.id)
                    .collect()
            },
            EventPropagationMode::ByRole(role) => {
                // 发送给指定角色的节点
                let network_role = match role {
                    NodeRole::Master => NetworkNodeRole::Master,
                    NodeRole::Slave => NetworkNodeRole::Worker, // 映射到 Worker
                    NodeRole::Worker => NetworkNodeRole::Worker,
                    NodeRole::Coordinator => NetworkNodeRole::Master, // 映射到 Master
                    NodeRole::Gateway => NetworkNodeRole::Worker, // 映射到 Worker
                    NodeRole::Storage => NetworkNodeRole::Worker, // 映射到 Worker
                    NodeRole::Custom(_) => NetworkNodeRole::Worker, // 映射到 Worker
                };
                // get_nodes_by_role 返回 Vec<NodeInfo>，不需要额外的 Result 包装
                let nodes = tokio::runtime::Handle::current().block_on(async {
                    self.network.get_nodes_by_role(&network_role).await
                });
                nodes.into_iter()
                    .map(|n| n.id)
                    .collect()
            },
            EventPropagationMode::ByGroup(ref group) => {
                // 发送给指定组的节点
                let nodes = tokio::runtime::Handle::current().block_on(async {
                    self.network.get_nodes_by_group(group).await
                }).unwrap_or_else(|_| Vec::new());
                nodes.into_iter()
                    .map(|n| n.id)
                    .collect()
            },
        };
        
        // 创建消息
        let message_id = format!("msg-{}", uuid::Uuid::new_v4());
        let message = EventNetworkMessage::EventNotification {
            event: event.clone(),
            source_node: self.node_id.clone(),
            message_id: message_id.clone(),
        };
        
        // 序列化消息
        let data = bincode::serialize(&message)?;
        
        // 如果启用了确认机制，记录待确认的事件
        if self.config.enable_ack && !target_nodes.is_empty() {
            let mut cache = self.event_cache.write().map_err(|e| {
                Error::Internal(format!("无法获取事件缓存写锁: {}", e))
            })?;
            
            cache.pending_acks.insert(message_id.clone(), PendingEventAck {
                event_id: event.id.clone(),
                target_nodes: target_nodes.iter().cloned().collect(),
                acked_nodes: HashSet::new(),
                sent_at: Instant::now(),
                retry_count: 0,
            });
        }
        
        // 发送到所有目标节点
        for node_id in target_nodes {
            if node_id == self.node_id {
                continue; // 跳过自己
            }
            
            if let Err(e) = tokio::runtime::Handle::current().block_on(async {
                self.network.send_to(&node_id, "event_system", &data).await
            }) {
                warn!("发送事件到节点 {} 失败: {}", node_id, e);
            }
        }
        
        Ok(())
    }
    
    fn subscribe(&self, event_type: EventType, callback: Arc<dyn EventCallback>) -> Result<String> {
        // 委托给本地事件系统
        self.local_system.subscribe(event_type, callback)
    }
    
    fn subscribe_all(&self, callback: Arc<dyn EventCallback>) -> Result<String> {
        // 委托给本地事件系统
        self.local_system.subscribe_all(callback)
    }
    
    fn unsubscribe(&self, subscription_id: &str) -> Result<()> {
        // 委托给本地事件系统
        self.local_system.unsubscribe(subscription_id)
    }
    
    fn get_pending_events(&self) -> Result<Vec<Event>> {
        // 委托给本地事件系统
        self.local_system.get_pending_events()
    }
    
    fn start(&self) -> Result<()> {
        // 调用DistributedEventSystem自己的start方法，而不是EventSystem的方法
        DistributedEventSystem::start(self)
    }
    
    fn stop(&self) -> Result<()> {
        // 调用DistributedEventSystem自己的stop方法，而不是EventSystem的方法
        DistributedEventSystem::stop(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use tempfile::tempdir;
    
    struct TestCallback {
        triggered: Arc<AtomicBool>,
    }
    
    impl EventCallback for TestCallback {
        fn on_event(&self, _event: &Event) -> Result<()> {
            self.triggered.store(true, Ordering::SeqCst);
            Ok(())
        }
    }
    
    // 注意：这些测试需要真实的网络环境，在单元测试中可能无法正常运行
    // 这里仅作为示例代码，实际测试需要使用模拟的网络环境
    #[test]
    #[ignore]
    fn test_distributed_event_system() {
        // 创建网络管理器
        let config_dir = tempdir().unwrap();
        let node_a_info = NodeInfo::new(
            "Node A",
            "127.0.0.1:8000".parse().unwrap(),
            NetworkNodeRole::Worker,
        );
        
        let network_a = Arc::new(NetworkManager::new(
            node_a_info.clone(),
            config_dir.path().to_str().unwrap(),
        ).unwrap());
        
        // 创建分布式事件系统
        let system_a = DistributedEventSystem::new(
            "node-a".to_string(),
            network_a,
            None
        ).unwrap();
        
        // 启动
        system_a.start().unwrap();
        
        // 创建订阅回调
        let triggered = Arc::new(AtomicBool::new(false));
        let callback = Arc::new(TestCallback {
            triggered: triggered.clone(),
        });
        
        // 订阅事件
        let subscription_id = system_a.subscribe(EventType::SystemStarted, callback).unwrap();
        
        // 发布事件
        let event = Event::new(EventType::SystemStarted, "测试事件".to_string());
        system_a.publish(event).unwrap();
        
        // 等待事件处理
        std::thread::sleep(Duration::from_millis(100));
        
        // 验证回调被触发
        assert!(triggered.load(Ordering::SeqCst));
        
        // 清理
        system_a.unsubscribe(&subscription_id).unwrap();
        system_a.stop().unwrap();
    }
} 