// 分布式状态同步协议
//
// 提供分布式环境下的状态同步机制，确保数据一致性

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use std::thread;
use log::{debug, error, info, warn};
use uuid::Uuid;

use crate::error::{Error, Result};

/// 节点ID类型
pub type NodeId = String;

/// 状态版本类型
pub type StateVersion = u64;

/// 同步优先级
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SyncPriority {
    /// 低优先级
    Low = 0,
    /// 中优先级
    Medium = 1,
    /// 高优先级
    High = 2,
    /// 紧急优先级
    Critical = 3,
}

impl Default for SyncPriority {
    fn default() -> Self {
        SyncPriority::Medium
    }
}

/// 同步状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncState {
    /// 空闲
    Idle,
    /// 同步中
    Syncing,
    /// 验证中
    Validating,
    /// 完成
    Completed,
    /// 失败
    Failed,
}

/// 同步操作类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncOperationType {
    /// 完整同步
    FullSync,
    /// 增量同步
    IncrementalSync,
    /// 仅元数据同步
    MetadataOnly,
    /// 按需同步
    OnDemand,
}

/// 状态同步配置
#[derive(Debug, Clone)]
pub struct StateSyncConfig {
    /// 同步间隔（毫秒）
    pub sync_interval_ms: u64,
    /// 超时时间（毫秒）
    pub timeout_ms: u64,
    /// 重试次数
    pub retry_count: usize,
    /// 重试间隔（毫秒）
    pub retry_interval_ms: u64,
    /// 批处理大小
    pub batch_size: usize,
    /// 是否启用压缩
    pub enable_compression: bool,
    /// 是否启用增量同步
    pub enable_incremental_sync: bool,
    /// 最大并发同步数
    pub max_concurrent_syncs: usize,
    /// 心跳间隔（毫秒）
    pub heartbeat_interval_ms: u64,
    /// 节点超时时间（毫秒）
    pub node_timeout_ms: u64,
    /// 数据验证级别
    pub verification_level: VerificationLevel,
}

impl Default for StateSyncConfig {
    fn default() -> Self {
        Self {
            sync_interval_ms: 5000,      // 5秒同步一次
            timeout_ms: 30000,           // 30秒超时
            retry_count: 3,              // 重试3次
            retry_interval_ms: 1000,     // 1秒重试间隔
            batch_size: 1000,            // 每批1000条
            enable_compression: true,    // 启用压缩
            enable_incremental_sync: true, // 启用增量同步
            max_concurrent_syncs: 5,     // 最多5个并发同步
            heartbeat_interval_ms: 1000, // 1秒心跳间隔
            node_timeout_ms: 5000,       // 5秒节点超时
            verification_level: VerificationLevel::Checksum,
        }
    }
}

/// 数据验证级别
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationLevel {
    /// 无验证
    None,
    /// 校验和验证
    Checksum,
    /// 完整内容验证
    Full,
    /// 抽样验证
    Sampling,
}

/// 同步请求
#[derive(Debug, Clone)]
pub struct SyncRequest {
    /// 请求ID
    pub id: String,
    /// 目标节点
    pub target_node: NodeId,
    /// 请求时间
    pub request_time: Instant,
    /// 超时时间
    pub timeout: Duration,
    /// 同步类型
    pub sync_type: SyncOperationType,
    /// 优先级
    pub priority: SyncPriority,
    /// 请求的起始版本
    pub start_version: StateVersion,
    /// 请求的结束版本
    pub end_version: StateVersion,
    /// 请求的数据键集合（如果为空，则同步所有数据）
    pub keys: Option<HashSet<Vec<u8>>>,
}

/// 同步响应
#[derive(Debug, Clone)]
pub struct SyncResponse {
    /// 请求ID
    pub request_id: String,
    /// 源节点
    pub source_node: NodeId,
    /// 响应时间
    pub response_time: Instant,
    /// 状态码
    pub status: SyncStatus,
    /// 同步的起始版本
    pub start_version: StateVersion,
    /// 同步的结束版本
    pub end_version: StateVersion,
    /// 同步的数据条数
    pub entry_count: usize,
    /// 数据校验和
    pub checksum: String,
}

/// 同步状态码
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStatus {
    /// 成功
    Success,
    /// 部分成功
    PartialSuccess,
    /// 版本冲突
    VersionConflict,
    /// 数据不可用
    DataUnavailable,
    /// 节点忙
    NodeBusy,
    /// 权限错误
    PermissionError,
    /// 超时
    Timeout,
    /// 网络错误
    NetworkError,
    /// 未知错误
    UnknownError,
}

/// 节点状态
#[derive(Debug, Clone)]
pub struct NodeState {
    /// 节点ID
    pub id: NodeId,
    /// 节点地址
    pub address: String,
    /// 角色
    pub role: NodeRole,
    /// 最后心跳时间
    pub last_heartbeat: Instant,
    /// 当前状态
    pub state: NodeStatus,
    /// 当前版本
    pub current_version: StateVersion,
    /// 延迟（毫秒）
    pub latency_ms: u64,
    /// 负载
    pub load: f64,
    /// 同步成功率
    pub sync_success_rate: f64,
}

/// 节点角色
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    /// 主节点
    Primary,
    /// 从节点
    Secondary,
    /// 仲裁节点
    Arbiter,
    /// 观察者节点
    Observer,
}

/// 节点状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    /// 在线
    Online,
    /// 离线
    Offline,
    /// 同步中
    Syncing,
    /// 降级
    Degraded,
    /// 故障
    Failed,
    /// 恢复中
    Recovering,
    /// 维护中
    Maintenance,
}

/// 状态同步管理器
pub struct StateSyncManager {
    /// 节点ID
    node_id: NodeId,
    
    /// 配置
    config: StateSyncConfig,
    
    /// 已知节点
    nodes: Arc<RwLock<HashMap<NodeId, NodeState>>>,
    
    /// 当前版本
    current_version: Arc<RwLock<StateVersion>>,
    
    /// 同步请求队列
    sync_requests: Arc<Mutex<VecDeque<SyncRequest>>>,
    
    /// 进行中的同步
    active_syncs: Arc<RwLock<HashMap<String, SyncState>>>,
    
    /// 同步历史
    sync_history: Arc<Mutex<VecDeque<SyncResponse>>>,
    
    /// 状态提供者（用于获取当前状态）
    state_provider: Arc<Mutex<Option<Box<dyn StateProvider + Send + Sync>>>>,
    
    /// 状态应用器（用于应用同步的状态）
    state_applier: Arc<Mutex<Option<Box<dyn StateApplier + Send + Sync>>>>,
    
    /// 运行标志
    running: Arc<RwLock<bool>>,
}

/// 状态提供者接口
pub trait StateProvider: Send + Sync {
    /// 获取当前版本
    fn get_current_version(&self) -> Result<StateVersion>;
    
    /// 获取指定版本范围的状态数据
    fn get_state_data(&self, start_version: StateVersion, end_version: StateVersion, keys: Option<&HashSet<Vec<u8>>>) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;
    
    /// 获取状态校验和
    fn get_state_checksum(&self, version: StateVersion) -> Result<String>;
}

/// 状态应用器接口
pub trait StateApplier: Send + Sync {
    /// 应用状态数据
    fn apply_state_data(&self, version: StateVersion, data: Vec<(Vec<u8>, Vec<u8>)>) -> Result<()>;
    
    /// 验证状态
    fn verify_state(&self, version: StateVersion, checksum: &str) -> Result<bool>;
}

impl StateSyncManager {
    /// 创建新的状态同步管理器
    pub fn new(node_id: &str, config: Option<StateSyncConfig>) -> Self {
        let config = config.unwrap_or_default();
        
        Self {
            node_id: node_id.to_string(),
            config,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            current_version: Arc::new(RwLock::new(0)),
            sync_requests: Arc::new(Mutex::new(VecDeque::new())),
            active_syncs: Arc::new(RwLock::new(HashMap::new())),
            sync_history: Arc::new(Mutex::new(VecDeque::with_capacity(100))),
            state_provider: Arc::new(Mutex::new(None)),
            state_applier: Arc::new(Mutex::new(None)),
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// 设置状态提供者
    pub fn set_state_provider<P: StateProvider + 'static>(&self, provider: P) -> Result<()> {
        let mut state_provider = self.state_provider.lock().map_err(|e| {
            Error::LockError(format!("Failed to acquire state provider lock: {}", e))
        })?;
        
        *state_provider = Some(Box::new(provider));
        
        Ok(())
    }

    /// 设置状态应用器
    pub fn set_state_applier<A: StateApplier + 'static>(&self, applier: A) -> Result<()> {
        let mut state_applier = self.state_applier.lock().map_err(|e| {
            Error::LockError(format!("Failed to acquire state applier lock: {}", e))
        })?;
        
        *state_applier = Some(Box::new(applier));
        
        Ok(())
    }

    /// 启动状态同步
    pub fn start(&self) -> Result<()> {
        let mut running = self.running.write().map_err(|e| {
            Error::LockError(format!("Failed to acquire running lock: {}", e))
        })?;
        
        if *running {
            return Ok(());
        }
        
        *running = true;
        
        // 启动同步处理线程
        self.start_sync_processor()?;
        
        // 启动心跳发送线程
        self.start_heartbeat_sender()?;
        
        info!("State sync manager started for node {}", self.node_id);
        Ok(())
    }

    /// 停止状态同步
    pub fn stop(&self) -> Result<()> {
        let mut running = self.running.write().map_err(|e| {
            Error::LockError(format!("Failed to acquire running lock: {}", e))
        })?;
        
        if !*running {
            return Ok(());
        }
        
        *running = false;
        
        info!("State sync manager stopped for node {}", self.node_id);
        Ok(())
    }

    /// 启动同步处理线程
    fn start_sync_processor(&self) -> Result<()> {
        let sync_requests = self.sync_requests.clone();
        let active_syncs = self.active_syncs.clone();
        let sync_history = self.sync_history.clone();
        let running = self.running.clone();
        let config = self.config.clone();
        let node_id = self.node_id.clone();
        let state_provider = self.state_provider.clone();
        let state_applier = self.state_applier.clone();
        
        thread::spawn(move || {
            let sync_interval = Duration::from_millis(config.sync_interval_ms);
            
            while {
                let running_guard = running.read().unwrap();
                *running_guard
            } {
                // 处理同步请求
                Self::process_sync_requests(
                    &sync_requests,
                    &active_syncs,
                    &sync_history,
                    &node_id,
                    &config,
                    &state_provider,
                    &state_applier,
                ).unwrap_or_else(|e| {
                    error!("Error processing sync requests: {}", e);
                });
                
                // 清理过期的同步请求
                Self::cleanup_expired_syncs(
                    &active_syncs,
                    &config,
                ).unwrap_or_else(|e| {
                    error!("Error cleaning up expired syncs: {}", e);
                });
                
                // 等待下一个同步周期
                thread::sleep(sync_interval);
            }
        });
        
        Ok(())
    }

    /// 启动心跳发送线程
    fn start_heartbeat_sender(&self) -> Result<()> {
        let nodes = self.nodes.clone();
        let running = self.running.clone();
        let config = self.config.clone();
        let node_id = self.node_id.clone();
        let current_version = self.current_version.clone();
        
        thread::spawn(move || {
            let heartbeat_interval = Duration::from_millis(config.heartbeat_interval_ms);
            
            while {
                let running_guard = running.read().unwrap();
                *running_guard
            } {
                // 发送心跳到所有节点
                Self::send_heartbeats(
                    &nodes,
                    &node_id,
                    &current_version,
                ).unwrap_or_else(|e| {
                    error!("Error sending heartbeats: {}", e);
                });
                
                // 检测节点超时
                Self::detect_node_timeouts(
                    &nodes,
                    &config,
                ).unwrap_or_else(|e| {
                    error!("Error detecting node timeouts: {}", e);
                });
                
                // 等待下一个心跳周期
                thread::sleep(heartbeat_interval);
            }
        });
        
        Ok(())
    }

    /// 处理同步请求
    fn process_sync_requests(
        sync_requests: &Mutex<VecDeque<SyncRequest>>,
        active_syncs: &RwLock<HashMap<String, SyncState>>,
        sync_history: &Mutex<VecDeque<SyncResponse>>,
        node_id: &str,
        config: &StateSyncConfig,
        state_provider: &Mutex<Option<Box<dyn StateProvider + Send + Sync>>>,
        state_applier: &Mutex<Option<Box<dyn StateApplier + Send + Sync>>>,
    ) -> Result<()> {
        // 获取当前活跃同步数量
        let active_count = {
            let active_syncs_guard = active_syncs.read().map_err(|e| {
                Error::LockError(format!("Failed to acquire active syncs lock: {}", e))
            })?;
            
            active_syncs_guard.len()
        };
        
        // 检查是否达到最大并发同步数
        if active_count >= config.max_concurrent_syncs {
            return Ok(());
        }
        
        // 计算可处理的请求数量
        let available_slots = config.max_concurrent_syncs - active_count;
        
        // 获取待处理的请求
        let requests_to_process = {
            let mut sync_requests_guard = sync_requests.lock().map_err(|e| {
                Error::LockError(format!("Failed to acquire sync requests lock: {}", e))
            })?;
            
            // 按优先级排序
            let mut requests: Vec<SyncRequest> = sync_requests_guard.drain(..).collect();
            requests.sort_by(|a, b| b.priority.cmp(&a.priority));
            
            // 限制处理数量
            requests.truncate(available_slots);
            
            requests
        };
        
        for request in requests_to_process {
            // 标记为活跃同步
            {
                let mut active_syncs_guard = active_syncs.write().map_err(|e| {
                    Error::LockError(format!("Failed to acquire active syncs write lock: {}", e))
                })?;
                
                active_syncs_guard.insert(request.id.clone(), SyncState::Syncing);
            }
            
            // 处理同步请求
            let response = match request.sync_type {
                SyncOperationType::FullSync => {
                    Self::handle_full_sync(&request, state_provider, state_applier, config)
                },
                SyncOperationType::IncrementalSync => {
                    Self::handle_incremental_sync(&request, state_provider, state_applier, config)
                },
                SyncOperationType::MetadataOnly => {
                    Self::handle_metadata_sync(&request, state_provider, config)
                },
                SyncOperationType::OnDemand => {
                    Self::handle_on_demand_sync(&request, state_provider, state_applier, config)
                },
            };
            
            // 更新同步状态和历史
            match response {
                Ok(sync_response) => {
                    // 更新活跃同步状态
                    {
                        let mut active_syncs_guard = active_syncs.write().map_err(|e| {
                            Error::LockError(format!("Failed to acquire active syncs write lock: {}", e))
                        })?;
                        
                        active_syncs_guard.insert(request.id.clone(), SyncState::Completed);
                    }
                    
                    // 添加到同步历史
                    {
                        let mut sync_history_guard = sync_history.lock().map_err(|e| {
                            Error::LockError(format!("Failed to acquire sync history lock: {}", e))
                        })?;
                        
                        sync_history_guard.push_back(sync_response);
                        
                        // 限制历史记录数量
                        while sync_history_guard.len() > 100 {
                            sync_history_guard.pop_front();
                        }
                    }
                    
                    info!("Sync request {} completed successfully", request.id);
                },
                Err(e) => {
                    error!("Failed to process sync request {}: {}", request.id, e);
                    
                    // 更新活跃同步状态为失败
                    {
                        let mut active_syncs_guard = active_syncs.write().map_err(|_| {
                            Error::internal(format!("Failed to acquire active syncs write lock"))
                        })?;
                        
                        active_syncs_guard.insert(request.id.clone(), SyncState::Failed);
                    }
                },
            }
        }
        
        Ok(())
    }

    /// 清理过期的同步
    fn cleanup_expired_syncs(
        active_syncs: &RwLock<HashMap<String, SyncState>>,
        config: &StateSyncConfig,
    ) -> Result<()> {
        let now = Instant::now();
        let timeout = Duration::from_millis(config.timeout_ms);
        
        let expired_syncs = {
            let active_syncs_guard = active_syncs.read().map_err(|e| {
                Error::LockError(format!("Failed to acquire active syncs lock: {}", e))
            })?;
            
            active_syncs_guard.iter()
                .filter(|(_, state)| **state == SyncState::Syncing || **state == SyncState::Validating)
                .map(|(id, _)| id.clone())
                .collect::<Vec<String>>()
        };
        
        if !expired_syncs.is_empty() {
            let mut active_syncs_guard = active_syncs.write().map_err(|e| {
                Error::LockError(format!("Failed to acquire active syncs write lock: {}", e))
            })?;
            
            for id in expired_syncs {
                active_syncs_guard.remove(&id);
                warn!("Expired sync request {} removed", id);
            }
        }
        
        Ok(())
    }

    /// 发送心跳到所有节点
    fn send_heartbeats(
        nodes: &RwLock<HashMap<NodeId, NodeState>>,
        node_id: &str,
        current_version: &RwLock<StateVersion>,
    ) -> Result<()> {
        let version = {
            let version_guard = current_version.read().map_err(|e| {
                Error::LockError(format!("Failed to acquire current version lock: {}", e))
            })?;
            
            *version_guard
        };
        
        let nodes_to_heartbeat = {
            let nodes_guard = nodes.read().map_err(|e| {
                Error::LockError(format!("Failed to acquire nodes lock: {}", e))
            })?;
            
            nodes_guard.iter()
                .filter(|(id, _)| **id != node_id)
                .map(|(id, state)| (id.clone(), state.address.clone()))
                .collect::<Vec<(NodeId, String)>>()
        };
        
        for (id, address) in nodes_to_heartbeat {
            // 在实际实现中，这里应该发送心跳
            debug!("Sending heartbeat to node {} at {} with version {}", id, address, version);
            
            // 更新节点的最后心跳时间
            {
                let mut nodes_guard = nodes.write().map_err(|e| {
                    Error::LockError(format!("Failed to acquire nodes write lock: {}", e))
                })?;
                
                if let Some(node) = nodes_guard.get_mut(&id) {
                    node.last_heartbeat = Instant::now();
                }
            }
        }
        
        Ok(())
    }

    /// 检测节点超时
    fn detect_node_timeouts(
        nodes: &RwLock<HashMap<NodeId, NodeState>>,
        config: &StateSyncConfig,
    ) -> Result<()> {
        let now = Instant::now();
        let timeout = Duration::from_millis(config.node_timeout_ms);
        
        let timed_out_nodes = {
            let nodes_guard = nodes.read().map_err(|e| {
                Error::LockError(format!("Failed to acquire nodes lock: {}", e))
            })?;
            
            nodes_guard.iter()
                .filter(|(_, state)| {
                    state.state == NodeStatus::Online && 
                    now.duration_since(state.last_heartbeat) > timeout
                })
                .map(|(id, _)| id.clone())
                .collect::<Vec<NodeId>>()
        };
        
        if !timed_out_nodes.is_empty() {
            let mut nodes_guard = nodes.write().map_err(|e| {
                Error::LockError(format!("Failed to acquire nodes write lock: {}", e))
            })?;
            
            for id in timed_out_nodes {
                if let Some(node) = nodes_guard.get_mut(&id) {
                    node.state = NodeStatus::Offline;
                    warn!("Node {} marked as offline due to timeout", id);
                }
            }
        }
        
        Ok(())
    }

    /// 处理完整同步
    fn handle_full_sync(
        request: &SyncRequest,
        state_provider: &Mutex<Option<Box<dyn StateProvider + Send + Sync>>>,
        state_applier: &Mutex<Option<Box<dyn StateApplier + Send + Sync>>>,
        config: &StateSyncConfig,
    ) -> Result<SyncResponse> {
        // 获取状态提供者
        let provider = {
            let provider_guard = state_provider.lock().map_err(|e| {
                Error::LockError(format!("Failed to acquire state provider lock: {}", e))
            })?;
            
            if provider_guard.is_none() {
                return Err(Error::invalid_state("State provider not set"));
            }
            
            provider_guard.as_ref().unwrap()
        };
        
        // 获取状态应用器
        let applier = {
            let applier_guard = state_applier.lock().map_err(|e| {
                Error::LockError(format!("Failed to acquire state applier lock: {}", e))
            })?;
            
            if applier_guard.is_none() {
                return Err(Error::invalid_state("State applier not set"));
            }
            
            applier_guard.as_ref().unwrap()
        };
        
        // 获取状态数据
        let state_data = provider.get_state_data(
            request.start_version,
            request.end_version,
            request.keys.as_ref(),
        )?;
        
        let entry_count = state_data.len();
        
        // 应用状态数据
        applier.apply_state_data(request.end_version, state_data)?;
        
        // 获取校验和
        let checksum = provider.get_state_checksum(request.end_version)?;
        
        // 验证状态
        let verified = applier.verify_state(request.end_version, &checksum)?;
        
        if !verified {
            return Err(Error::data_corruption("State verification failed"));
        }
        
        // 创建响应
        let response = SyncResponse {
            request_id: request.id.clone(),
            source_node: request.target_node.clone(),
            response_time: Instant::now(),
            status: SyncStatus::Success,
            start_version: request.start_version,
            end_version: request.end_version,
            entry_count,
            checksum,
        };
        
        Ok(response)
    }

    /// 处理增量同步
    fn handle_incremental_sync(
        request: &SyncRequest,
        state_provider: &Mutex<Option<Box<dyn StateProvider + Send + Sync>>>,
        state_applier: &Mutex<Option<Box<dyn StateApplier + Send + Sync>>>,
        config: &StateSyncConfig,
    ) -> Result<SyncResponse> {
        // 实现增量同步逻辑
        // 简化处理，重用完整同步实现
        Self::handle_full_sync(request, state_provider, state_applier, config)
    }

    /// 处理元数据同步
    fn handle_metadata_sync(
        request: &SyncRequest,
        state_provider: &Mutex<Option<Box<dyn StateProvider + Send + Sync>>>,
        config: &StateSyncConfig,
    ) -> Result<SyncResponse> {
        // 获取状态提供者
        let provider = {
            let provider_guard = state_provider.lock().map_err(|e| {
                Error::LockError(format!("Failed to acquire state provider lock: {}", e))
            })?;
            
            if provider_guard.is_none() {
                return Err(Error::invalid_state("State provider not set"));
            }
            
            provider_guard.as_ref().unwrap()
        };
        
        // 获取校验和
        let checksum = provider.get_state_checksum(request.end_version)?;
        
        // 创建响应
        let response = SyncResponse {
            request_id: request.id.clone(),
            source_node: request.target_node.clone(),
            response_time: Instant::now(),
            status: SyncStatus::Success,
            start_version: request.start_version,
            end_version: request.end_version,
            entry_count: 0, // 元数据同步不传输数据条目
            checksum,
        };
        
        Ok(response)
    }

    /// 处理按需同步
    fn handle_on_demand_sync(
        request: &SyncRequest,
        state_provider: &Mutex<Option<Box<dyn StateProvider + Send + Sync>>>,
        state_applier: &Mutex<Option<Box<dyn StateApplier + Send + Sync>>>,
        config: &StateSyncConfig,
    ) -> Result<SyncResponse> {
        // 实现按需同步逻辑
        // 简化处理，重用完整同步实现
        Self::handle_full_sync(request, state_provider, state_applier, config)
    }

    /// 添加同步请求
    pub fn add_sync_request(&self, target_node: &str, sync_type: SyncOperationType, start_version: StateVersion, end_version: StateVersion, priority: Option<SyncPriority>, keys: Option<HashSet<Vec<u8>>>) -> Result<String> {
        let request_id = Uuid::new_v4().to_string();
        
        let request = SyncRequest {
            id: request_id.clone(),
            target_node: target_node.to_string(),
            request_time: Instant::now(),
            timeout: Duration::from_millis(self.config.timeout_ms),
            sync_type,
            priority: priority.unwrap_or_default(),
            start_version,
            end_version,
            keys,
        };
        
        // 添加到请求队列
        {
            let mut sync_requests = self.sync_requests.lock().map_err(|e| {
                Error::LockError(format!("Failed to acquire sync requests lock: {}", e))
            })?;
            
            sync_requests.push_back(request);
        }
        
        debug!("Added sync request {} to queue", request_id);
        Ok(request_id)
    }

    /// 添加节点
    pub fn add_node(&self, id: &str, address: &str, role: NodeRole) -> Result<()> {
        let mut nodes = self.nodes.write().map_err(|e| {
            Error::LockError(format!("Failed to acquire nodes write lock: {}", e))
        })?;
        
        let node_state = NodeState {
            id: id.to_string(),
            address: address.to_string(),
            role,
            last_heartbeat: Instant::now(),
            state: NodeStatus::Online,
            current_version: 0,
            latency_ms: 0,
            load: 0.0,
            sync_success_rate: 1.0,
        };
        
        nodes.insert(id.to_string(), node_state);
        
        info!("Added node {} at {} with role {:?}", id, address, role);
        Ok(())
    }

    /// 获取节点状态
    pub fn get_node_status(&self, node_id: &str) -> Result<Option<NodeStatus>> {
        let nodes = self.nodes.read().map_err(|e| {
            Error::LockError(format!("Failed to acquire nodes lock: {}", e))
        })?;
        
        if let Some(node) = nodes.get(node_id) {
            Ok(Some(node.state))
        } else {
            Ok(None)
        }
    }

    /// 获取所有节点
    pub fn get_all_nodes(&self) -> Result<HashMap<NodeId, NodeState>> {
        let nodes = self.nodes.read().map_err(|e| {
            Error::LockError(format!("Failed to acquire nodes lock: {}", e))
        })?;
        
        Ok(nodes.clone())
    }

    /// 获取同步状态
    pub fn get_sync_status(&self, request_id: &str) -> Result<Option<SyncState>> {
        let active_syncs = self.active_syncs.read().map_err(|e| {
            Error::LockError(format!("Failed to acquire active syncs lock: {}", e))
        })?;
        
        if let Some(state) = active_syncs.get(request_id) {
            Ok(Some(*state))
        } else {
            Ok(None)
        }
    }

    /// 获取同步历史
    pub fn get_sync_history(&self) -> Result<Vec<SyncResponse>> {
        let sync_history = self.sync_history.lock().map_err(|e| {
            Error::LockError(format!("Failed to acquire sync history lock: {}", e))
        })?;
        
        Ok(sync_history.iter().cloned().collect())
    }

    /// 更新当前版本
    pub fn update_current_version(&self, version: StateVersion) -> Result<()> {
        let mut current_version = self.current_version.write().map_err(|e| {
            Error::LockError(format!("Failed to acquire current version write lock: {}", e))
        })?;
        
        *current_version = version;
        
        Ok(())
    }

    /// 获取当前版本
    pub fn get_current_version(&self) -> Result<StateVersion> {
        let current_version = self.current_version.read().map_err(|e| {
            Error::LockError(format!("Failed to acquire current version lock: {}", e))
        })?;
        
        Ok(*current_version)
    }

    /// 强制同步
    pub fn force_sync(&self, target_node: &str, end_version: Option<StateVersion>) -> Result<String> {
        let current_version = self.get_current_version()?;
        let end_version = end_version.unwrap_or(current_version);
        
        self.add_sync_request(
            target_node,
            SyncOperationType::FullSync,
            0,
            end_version,
            Some(SyncPriority::High),
            None,
        )
    }

    /// 获取当前版本并更新
    pub fn get_and_update_current_version(&self, new_version: Option<StateVersion>) -> Result<StateVersion> {
        let mut current_version = self.current_version.write().map_err(|e| {
            Error::LockError(format!("Failed to acquire current version write lock: {}", e))
        })?;
        
        let old_version = *current_version;
        
        if let Some(version) = new_version {
            if version > old_version {
                *current_version = version;
                debug!("Updated current version from {} to {}", old_version, version);
            }
        }
        
        Ok(old_version)
    }
    
    /// 获取节点角色
    pub fn get_node_role(&self, node_id: &str) -> Result<Option<NodeRole>> {
        let nodes = self.nodes.read().map_err(|e| {
            Error::LockError(format!("Failed to acquire nodes lock: {}", e))
        })?;
        
        Ok(nodes.get(node_id).map(|node| node.role))
    }
    
    /// 设置本节点角色
    pub fn set_local_role(&self, role: NodeRole) -> Result<()> {
        let mut nodes = self.nodes.write().map_err(|e| {
            Error::LockError(format!("Failed to acquire nodes write lock: {}", e))
        })?;
        
        if let Some(node) = nodes.get_mut(&self.node_id) {
            node.role = role;
            info!("Set local node role to {:?}", role);
        } else {
            // 创建本地节点记录
            let node = NodeState {
                id: self.node_id.clone(),
                address: format!("local://{}", self.node_id),
                role,
                last_heartbeat: Instant::now(),
                state: NodeStatus::Online,
                current_version: self.get_current_version()?,
                latency_ms: 0,
                load: 0.0,
                sync_success_rate: 1.0,
            };
            
            nodes.insert(self.node_id.clone(), node);
            info!("Created local node with role {:?}", role);
        }
        
        Ok(())
    }
    
    /// 检测和处理节点状态变化
    pub fn detect_node_status_changes(&self) -> Result<Vec<(NodeId, NodeStatus, NodeStatus)>> {
        let mut changes = Vec::new();
        let now = Instant::now();
        let timeout = Duration::from_millis(self.config.node_timeout_ms);
        
        let mut nodes = self.nodes.write().map_err(|e| {
            Error::LockError(format!("Failed to acquire nodes write lock: {}", e))
        })?;
        
        for (node_id, node) in nodes.iter_mut() {
            // 跳过本地节点
            if node_id == &self.node_id {
                continue;
            }
            
            let old_status = node.state;
            
            // 检查超时
            if old_status == NodeStatus::Online && now.duration_since(node.last_heartbeat) > timeout {
                node.state = NodeStatus::Offline;
                warn!("Node {} marked as offline due to timeout", node_id);
                changes.push((node_id.clone(), old_status, NodeStatus::Offline));
            }
            // 可以添加其他状态检测逻辑
        }
        
        Ok(changes)
    }
    
    /// 处理同步冲突
    pub fn handle_sync_conflict(&self, target_node: &str, conflict_type: &str, details: &str) -> Result<()> {
        // 记录冲突
        warn!("Sync conflict with node {}: {} - {}", target_node, conflict_type, details);
        
        // 根据冲突类型处理
        match conflict_type {
            "version_conflict" => {
                // 版本冲突，强制完全同步
                let request_id = self.add_sync_request(
                    target_node,
                    SyncOperationType::FullSync,
                    0,
                    self.get_current_version()?,
                    Some(SyncPriority::High),
                    None
                )?;
                
                info!("Scheduled full sync with node {} due to version conflict, request ID: {}", 
                     target_node, request_id);
            },
            "data_corruption" => {
                // 数据损坏，标记节点为降级状态
                let mut nodes = self.nodes.write().map_err(|e| {
                    Error::LockError(format!("Failed to acquire nodes write lock: {}", e))
                })?;
                
                if let Some(node) = nodes.get_mut(target_node) {
                    node.state = NodeStatus::Degraded;
                    warn!("Node {} marked as degraded due to data corruption", target_node);
                }
            },
            _ => {
                // 其他冲突，记录但不采取特殊操作
                info!("Unknown conflict type: {}", conflict_type);
            }
        }
        
        Ok(())
    }
    
    /// 发送心跳到所有节点
    pub fn send_heartbeat_to_all(&self) -> Result<usize> {
        let nodes_to_heartbeat = {
            let nodes = self.nodes.read().map_err(|e| {
                Error::LockError(format!("Failed to acquire nodes lock: {}", e))
            })?;
            
            nodes.iter()
                .filter(|(id, _)| **id != self.node_id)
                .map(|(id, _)| id.clone())
                .collect::<Vec<_>>()
        };
        
        let current_version = self.get_current_version()?;
        let mut success_count = 0;
        
        for node_id in nodes_to_heartbeat {
            // 实际实现中，这里应该发送真实的心跳消息
            debug!("Sending heartbeat to node {} with version {}", node_id, current_version);
            
            // 模拟心跳发送成功
            success_count += 1;
        }
        
        Ok(success_count)
    }
    
    /// 创建增量同步请求
    pub fn create_incremental_sync(&self, target_node: &str, from_version: StateVersion) -> Result<String> {
        let current_version = self.get_current_version()?;
        
        if from_version >= current_version {
            debug!("No need for incremental sync with node {}, already at current version", target_node);
            return Ok("no_sync_needed".to_string());
        }
        
        let request_id = self.add_sync_request(
            target_node,
            SyncOperationType::IncrementalSync,
            from_version,
            current_version,
            Some(SyncPriority::Medium),
            None
        )?;
        
        info!("Created incremental sync request {} for node {} from version {} to {}",
             request_id, target_node, from_version, current_version);
        
        Ok(request_id)
    }
    
    /// 注册自定义验证回调
    pub fn register_validation_callback<F>(&self, callback: F) -> Result<()>
    where
        F: Fn(StateVersion, &str) -> Result<bool> + Send + Sync + 'static,
    {
        // 在实际实现中，应该将回调存储起来并在验证时调用
        info!("Registered custom validation callback");
        
        // 这里简化处理
        Ok(())
    }
    
    /// 计算同步效率指标
    pub fn calculate_sync_efficiency(&self) -> Result<f64> {
        let sync_history = self.sync_history.lock().map_err(|e| {
            Error::LockError(format!("Failed to acquire sync history lock: {}", e))
        })?;
        
        if sync_history.is_empty() {
            return Ok(1.0); // 默认值
        }
        
        let mut success_count = 0;
        let mut total_count = 0;
        
        for response in sync_history.iter().take(100) { // 只考虑最近100条记录
            total_count += 1;
            if response.status == SyncStatus::Success {
                success_count += 1;
            }
        }
        
        let efficiency = if total_count > 0 {
            success_count as f64 / total_count as f64
        } else {
            1.0
        };
        
        Ok(efficiency)
    }
    
    /// 清理过期同步历史
    pub fn cleanup_sync_history(&self, max_age_secs: u64) -> Result<usize> {
        let now = Instant::now();
        let max_age = Duration::from_secs(max_age_secs);
        
        let mut sync_history = self.sync_history.lock().map_err(|e| {
            Error::LockError(format!("Failed to acquire sync history lock: {}", e))
        })?;
        
        let original_len = sync_history.len();
        
        // 移除过期记录
        sync_history.retain(|response| {
            now.duration_since(response.response_time) <= max_age
        });
        
        let removed_count = original_len - sync_history.len();
        
        if removed_count > 0 {
            debug!("Removed {} expired sync history records", removed_count);
        }
        
        Ok(removed_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // 添加测试用例
} 