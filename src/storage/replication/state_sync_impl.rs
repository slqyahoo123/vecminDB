// 状态同步管理器实现
//
// 提供StateSyncManager的完整实现

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use std::thread;
use log::{debug, error, info, warn};
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::storage::engine::StorageEngine;
use crate::storage::replication::state_sync::{
    NodeId, StateVersion, SyncPriority, SyncState, SyncOperationType, SyncStatus,
    NodeRole, NodeStatus, StateSyncManager, StateProvider, StateApplier,
    SyncRequest, SyncResponse, StateSyncConfig, VerificationLevel,
    NodeState
};
use crate::storage::replication::sync_protocol::{
    SyncProtocol, SyncSession, SyncChunk, SyncStats,
    DefaultStateProvider, DefaultStateApplier
};

/// 状态同步管理器实现
pub struct StateSyncManagerImpl {
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
    
    /// 同步协议
    sync_protocol: Arc<SyncProtocol>,
    
    /// 状态提供者
    state_provider: Arc<DefaultStateProvider>,
    
    /// 状态应用器
    state_applier: Arc<DefaultStateApplier>,
    
    /// 运行标志
    running: Arc<RwLock<bool>>,
}

impl StateSyncManagerImpl {
    /// 创建新的状态同步管理器
    pub fn new(
        node_id: &str, 
        storage: Arc<dyn StorageEngine>,
        config: Option<StateSyncConfig>
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        
        // 创建状态提供者和应用器
        let state_provider = Arc::new(DefaultStateProvider::new(storage.clone()));
        let state_applier = Arc::new(DefaultStateApplier::new(
            storage.clone(),
            config.verification_level
        ));
        
                // 创建同步协议        let sync_protocol = Arc::new(SyncProtocol::new(            node_id,            Arc::new(StateSyncManager::new(node_id, Some(config.clone()))),            config.max_concurrent_syncs,            1024 * 1024, // 1MB块大小            config.enable_compression        ));
        
        // 设置协议的状态提供者和应用器
        sync_protocol.set_state_provider(state_provider.clone())?;
        sync_protocol.set_state_applier(state_applier.clone())?;
        
        let manager = Self {
            node_id: node_id.to_string(),
            config,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            current_version: Arc::new(RwLock::new(0)),
            sync_requests: Arc::new(Mutex::new(VecDeque::new())),
            active_syncs: Arc::new(RwLock::new(HashMap::new())),
            sync_history: Arc::new(Mutex::new(VecDeque::with_capacity(100))),
            sync_protocol,
            state_provider,
            state_applier,
            running: Arc::new(RwLock::new(false)),
        };
        
        Ok(manager)
    }

    /// 开始节点间的同步
    pub fn start_sync_with_node(&self, target_node: &str, sync_type: SyncOperationType, priority: SyncPriority) -> Result<String> {
        // 获取当前版本
        let current_version = self.get_current_version()?;
        
        // 获取目标节点的版本
        let target_version = match self.get_node_version(target_node)? {
            Some(version) => version,
            None => {
                warn!("Unknown version for node {}, assuming 0", target_node);
                0
            }
        };
        
        // 检查是否需要同步
        if target_version >= current_version && sync_type != SyncOperationType::FullSync {
            info!("Node {} is already at version {}, no sync needed", target_node, target_version);
            return Ok("no_sync_needed".to_string());
        }
        
        // 创建同步会话
        let session_id = self.sync_protocol.create_sync_session(
            target_node,
            sync_type,
            target_version,
            current_version,
            priority
        )?;
        
        // 记录活跃同步
        {
            let mut active_syncs = self.active_syncs.write().map_err(|e| {
                Error::LockError(format!("Failed to acquire active syncs write lock: {}", e))
            })?;
            
            active_syncs.insert(session_id.clone(), SyncState::Syncing);
        }
        
        info!("Started sync session {} with node {}", session_id, target_node);
        Ok(session_id)
    }

    /// 检查同步状态
    pub fn check_sync_status(&self, session_id: &str) -> Result<SyncState> {
        match self.sync_protocol.get_session_status(session_id)? {
            Some(state) => Ok(state),
            None => {
                // 检查历史记录
                let sync_history = self.sync_history.lock().map_err(|e| {
                    Error::LockError(format!("Failed to acquire sync history lock: {}", e))
                })?;
                
                for response in sync_history.iter() {
                    if response.request_id == session_id {
                        return Ok(if response.status == SyncStatus::Success {
                            SyncState::Completed
                        } else {
                            SyncState::Failed
                        });
                    }
                }
                
                Err(Error::not_found(format!("Sync session {} not found", session_id)))
            }
        }
    }

    /// 获取节点版本
    fn get_node_version(&self, node_id: &str) -> Result<Option<StateVersion>> {
        let nodes = self.nodes.read().map_err(|e| {
            Error::LockError(format!("Failed to acquire nodes lock: {}", e))
        })?;
        
        Ok(nodes.get(node_id).map(|node| node.current_version))
    }

    /// 更新节点状态
    pub fn update_node_state(&self, node_id: &str, status: NodeStatus, version: StateVersion) -> Result<()> {
        let mut nodes = self.nodes.write().map_err(|e| {
            Error::LockError(format!("Failed to acquire nodes write lock: {}", e))
        })?;
        
        if let Some(node) = nodes.get_mut(node_id) {
            node.state = status;
            node.current_version = version;
            node.last_heartbeat = Instant::now();
        } else {
            return Err(Error::not_found(format!("Node {} not found", node_id)));
        }
        
        Ok(())
    }

    /// 处理来自节点的心跳
    pub fn handle_heartbeat(&self, node_id: &str, version: StateVersion) -> Result<()> {
        let mut nodes = self.nodes.write().map_err(|e| {
            Error::LockError(format!("Failed to acquire nodes write lock: {}", e))
        })?;
        
        if let Some(node) = nodes.get_mut(node_id) {
            node.last_heartbeat = Instant::now();
            node.current_version = version;
            
            if node.state == NodeStatus::Offline {
                node.state = NodeStatus::Online;
                info!("Node {} is now online", node_id);
            }
        } else {
            // 新节点，添加到已知节点列表
            let new_node = NodeState {
                id: node_id.to_string(),
                address: format!("unknown://{}", node_id), // 实际实现中应该提供真实地址
                role: NodeRole::Secondary, // 假设新节点是从节点
                last_heartbeat: Instant::now(),
                state: NodeStatus::Online,
                current_version: version,
                latency_ms: 0,
                load: 0.0,
                sync_success_rate: 1.0,
            };
            
            nodes.insert(node_id.to_string(), new_node);
            info!("Added new node {} with version {}", node_id, version);
        }
        
        Ok(())
    }

    /// 获取同步统计信息
    pub fn get_sync_statistics(&self) -> Result<SyncStats> {
        self.sync_protocol.get_sync_stats()
    }

    /// 强制与所有节点同步
    pub fn force_sync_all_nodes(&self, sync_type: SyncOperationType) -> Result<Vec<String>> {
        let nodes = self.nodes.read().map_err(|e| {
            Error::LockError(format!("Failed to acquire nodes lock: {}", e))
        })?;
        
        let mut session_ids = Vec::new();
        
        for (node_id, node) in nodes.iter() {
            if node.id != self.node_id && node.state == NodeStatus::Online {
                match self.start_sync_with_node(&node.id, sync_type, SyncPriority::High) {
                    Ok(session_id) => {
                        if session_id != "no_sync_needed" {
                            session_ids.push(session_id);
                        }
                    },
                    Err(e) => {
                        error!("Failed to start sync with node {}: {}", node_id, e);
                    }
                }
            }
        }
        
        Ok(session_ids)
    }

    /// 等待同步完成
    pub fn wait_for_sync_completion(&self, session_ids: &[String], timeout_ms: u64) -> Result<bool> {
        let timeout = Duration::from_millis(timeout_ms);
        let start_time = Instant::now();
        let check_interval = Duration::from_millis(100);
        
        let mut completed_sessions = HashSet::new();
        
        while start_time.elapsed() < timeout {
            // 检查所有会话状态
            for session_id in session_ids {
                if completed_sessions.contains(session_id) {
                    continue;
                }
                
                match self.check_sync_status(session_id) {
                    Ok(state) => {
                        if state == SyncState::Completed || state == SyncState::Failed {
                            completed_sessions.insert(session_id.clone());
                        }
                    },
                    Err(_) => {
                        // 会话不存在，可能已完成
                        completed_sessions.insert(session_id.clone());
                    }
                }
            }
            
            // 检查是否所有会话都已完成
            if completed_sessions.len() == session_ids.len() {
                return Ok(true);
            }
            
            // 等待一段时间再检查
            thread::sleep(check_interval);
        }
        
        // 超时
        Ok(false)
    }

    /// 获取可用的状态提供者
    pub fn get_state_provider(&self) -> Arc<DefaultStateProvider> {
        self.state_provider.clone()
    }

    /// 获取可用的状态应用器
    pub fn get_state_applier(&self) -> Arc<DefaultStateApplier> {
        self.state_applier.clone()
    }
}

// 为StateSyncManager实现特定方法
impl StateSyncManager {
    /// 创建带有实现的新实例
    pub fn with_impl(
        node_id: &str, 
        storage: Arc<dyn StorageEngine>,
        config: Option<StateSyncConfig>
    ) -> Result<Self> {
        let impl_manager = StateSyncManagerImpl::new(node_id, storage, config)?;
        
        // 将实现封装在基本结构中
        let manager = Self {
            node_id: node_id.to_string(),
            config: impl_manager.config.clone(),
            nodes: impl_manager.nodes.clone(),
            current_version: impl_manager.current_version.clone(),
            sync_requests: impl_manager.sync_requests.clone(),
            active_syncs: impl_manager.active_syncs.clone(),
            sync_history: impl_manager.sync_history.clone(),
            state_provider: Arc::new(Mutex::new(Some(Box::new(impl_manager.state_provider.clone())))),
            state_applier: Arc::new(Mutex::new(Some(Box::new(impl_manager.state_applier.clone())))),
            running: impl_manager.running.clone(),
        };
        
        Ok(manager)
    }

    /// 使用同步协议启动同步
    pub fn start_sync_protocol(&self, storage: Arc<dyn StorageEngine>) -> Result<()> {
        // 创建实现管理器
        let impl_manager = StateSyncManagerImpl::new(&self.node_id, storage, Some(self.config.clone()))?;
        
        // 启动同步协议
        impl_manager.sync_protocol.start()?;
        
        // 设置运行标志
        let mut running = self.running.write().map_err(|e| {
            Error::LockError(format!("Failed to acquire running lock: {}", e))
        })?;
        
        *running = true;
        
        info!("Started sync protocol for node {}", self.node_id);
        Ok(())
    }

    /// 强制所有节点同步
    pub fn force_sync_all(&self, storage: Arc<dyn StorageEngine>) -> Result<Vec<String>> {
        // 创建实现管理器
        let impl_manager = StateSyncManagerImpl::new(&self.node_id, storage, Some(self.config.clone()))?;
        
        // 强制同步所有节点
        impl_manager.force_sync_all_nodes(SyncOperationType::FullSync)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // 添加测试用例
} 