// 分布式状态同步协议实现
//
// 提供高效的分布式状态同步机制，支持增量同步和快照同步

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use std::thread;
use log::{debug, error, info};
use uuid::Uuid;
use rocksdb::WriteBatch;

use crate::error::{Error, Result};
use crate::storage::replication::state_sync::{
    NodeId, StateVersion, SyncPriority, SyncState, SyncOperationType,
    StateSyncManager, StateProvider, StateApplier
};

/// 同步会话
#[derive(Debug, Clone)]
pub struct SyncSession {
    /// 会话ID
    pub id: String,
    /// 源节点
    pub source_node: NodeId,
    /// 目标节点
    pub target_node: NodeId,
    /// 开始时间
    pub start_time: Instant,
    /// 最后活动时间
    pub last_activity: Instant,
    /// 同步类型
    pub sync_type: SyncOperationType,
    /// 开始版本
    pub start_version: StateVersion,
    /// 结束版本
    pub end_version: StateVersion,
    /// 当前状态
    pub state: SyncState,
    /// 数据校验和
    pub checksum: Option<String>,
    /// 传输的数据块数
    pub transferred_chunks: usize,
    /// 总数据块数
    pub total_chunks: usize,
    /// 重试次数
    pub retry_count: usize,
}

/// 同步块
#[derive(Debug, Clone)]
pub struct SyncChunk {
    /// 块ID
    pub id: String,
    /// 会话ID
    pub session_id: String,
    /// 块索引
    pub index: usize,
    /// 块数据
    pub data: Vec<(Vec<u8>, Vec<u8>)>,
    /// 校验和
    pub checksum: String,
    /// 版本
    pub version: StateVersion,
}

/// 同步协议实现
pub struct SyncProtocol {
    /// 节点ID
    node_id: NodeId,
    
    /// 活跃会话
    active_sessions: Arc<RwLock<HashMap<String, SyncSession>>>,
    
    /// 挂起的数据块
    pending_chunks: Arc<Mutex<HashMap<String, Vec<SyncChunk>>>>,
    
    /// 状态提供者
    state_provider: Arc<Mutex<Option<Box<dyn StateProvider + Send + Sync>>>>,
    
    /// 状态应用器
    state_applier: Arc<Mutex<Option<Box<dyn StateApplier + Send + Sync>>>>,
    
    /// 运行标志
    running: Arc<RwLock<bool>>,
    
    /// 同步管理器引用
    sync_manager: Arc<StateSyncManager>,
    
    /// 最大会话数
    max_sessions: usize,
    
    /// 最大块大小（字节）
    max_chunk_size: usize,
    
    /// 是否启用压缩
    enable_compression: bool,
}

impl SyncProtocol {
    /// 创建新的同步协议实例
    pub fn new(
        node_id: &str, 
        sync_manager: Arc<StateSyncManager>,
        max_sessions: usize,
        max_chunk_size: usize,
        enable_compression: bool
    ) -> Self {
        Self {
            node_id: node_id.to_string(),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            pending_chunks: Arc::new(Mutex::new(HashMap::new())),
            state_provider: Arc::new(Mutex::new(None)),
            state_applier: Arc::new(Mutex::new(None)),
            running: Arc::new(RwLock::new(false)),
            sync_manager,
            max_sessions,
            max_chunk_size,
            enable_compression,
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

    /// 启动同步协议
    pub fn start(&self) -> Result<()> {
        let mut running = self.running.write().map_err(|e| {
            Error::LockError(format!("Failed to acquire running lock: {}", e))
        })?;
        
        if *running {
            return Ok(());
        }
        
        *running = true;
        
        // 启动会话清理线程
        self.start_session_cleanup()?;
        
        // 启动块处理线程
        self.start_chunk_processor()?;
        
        info!("Sync protocol started for node {}", self.node_id);
        Ok(())
    }

    /// 停止同步协议
    pub fn stop(&self) -> Result<()> {
        let mut running = self.running.write().map_err(|e| {
            Error::LockError(format!("Failed to acquire running lock: {}", e))
        })?;
        
        if !*running {
            return Ok(());
        }
        
        *running = false;
        
        info!("Sync protocol stopped for node {}", self.node_id);
        Ok(())
    }

    /// 启动会话清理线程
    fn start_session_cleanup(&self) -> Result<()> {
        let active_sessions = self.active_sessions.clone();
        let running = self.running.clone();
        
        thread::spawn(move || {
            let cleanup_interval = Duration::from_secs(60); // 每分钟清理一次
            
            while {
                let running_guard = running.read().unwrap();
                *running_guard
            } {
                // 清理过期会话
                if let Err(e) = Self::cleanup_expired_sessions(&active_sessions) {
                    error!("Error cleaning up expired sessions: {}", e);
                }
                
                // 等待下一次清理
                thread::sleep(cleanup_interval);
            }
        });
        
        Ok(())
    }

    /// 启动块处理线程
    fn start_chunk_processor(&self) -> Result<()> {
        let pending_chunks = self.pending_chunks.clone();
        let active_sessions = self.active_sessions.clone();
        let state_applier = self.state_applier.clone();
        let running = self.running.clone();
        let node_id = self.node_id.clone();
        
        thread::spawn(move || {
            let process_interval = Duration::from_millis(100); // 每100毫秒处理一次
            
            while {
                let running_guard = running.read().unwrap();
                *running_guard
            } {
                // 处理挂起的数据块
                if let Err(e) = Self::process_pending_chunks(
                    &pending_chunks,
                    &active_sessions,
                    &state_applier,
                    &node_id
                ) {
                    error!("Error processing pending chunks: {}", e);
                }
                
                // 等待下一次处理
                thread::sleep(process_interval);
            }
        });
        
        Ok(())
    }

    /// 清理过期会话
    fn cleanup_expired_sessions(
        active_sessions: &RwLock<HashMap<String, SyncSession>>
    ) -> Result<usize> {
        let now = Instant::now();
        let timeout = Duration::from_secs(3600); // 1小时超时
        let mut expired_sessions = Vec::new();
        
        // 找出过期会话
        {
            let sessions = active_sessions.read().map_err(|e| {
                Error::LockError(format!("Failed to acquire active sessions lock: {}", e))
            })?;
            
            for (id, session) in sessions.iter() {
                if now.duration_since(session.last_activity) > timeout {
                    expired_sessions.push(id.clone());
                }
            }
        }
        
        // 删除过期会话
        if !expired_sessions.is_empty() {
            let mut sessions = active_sessions.write().map_err(|e| {
                Error::LockError(format!("Failed to acquire active sessions write lock: {}", e))
            })?;
            
            for id in &expired_sessions {
                sessions.remove(id);
                info!("Removed expired sync session: {}", id);
            }
        }
        
        Ok(expired_sessions.len())
    }

    /// 处理挂起的数据块
    fn process_pending_chunks(
        pending_chunks: &Mutex<HashMap<String, Vec<SyncChunk>>>,
        active_sessions: &RwLock<HashMap<String, SyncSession>>,
        state_applier: &Mutex<Option<Box<dyn StateApplier + Send + Sync>>>,
        node_id: &str
    ) -> Result<usize> {
        let mut processed_count = 0;
        let mut sessions_to_update = HashMap::new();
        let mut chunks_to_remove = Vec::new();
        
        // 获取待处理的块
        let chunks_by_session = {
            let chunks = pending_chunks.lock().map_err(|e| {
                Error::LockError(format!("Failed to acquire pending chunks lock: {}", e))
            })?;
            
            chunks.clone()
        };
        
        // 检查没有待处理块的情况
        if chunks_by_session.is_empty() {
            return Ok(0);
        }
        
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
        
        // 处理每个会话的块
        for (session_id, chunks) in chunks_by_session.iter() {
            // 获取会话信息
            let session = {
                let sessions = active_sessions.read().map_err(|e| {
                    Error::LockError(format!("Failed to acquire active sessions lock: {}", e))
                })?;
                
                match sessions.get(session_id) {
                    Some(s) => s.clone(),
                    None => {
                        // 会话不存在，移除所有相关块
                        chunks_to_remove.push(session_id.clone());
                        continue;
                    }
                }
            };
            
            // 检查是否是目标节点
            if session.target_node != node_id {
                continue;
            }
            
            // 按索引排序块
            let mut sorted_chunks = chunks.clone();
            sorted_chunks.sort_by_key(|c| c.index);
            
            // 应用块数据
            for chunk in &sorted_chunks {
                match applier.apply_state_data(chunk.version, chunk.data.clone()) {
                    Ok(()) => {
                        processed_count += 1;
                        info!("Applied chunk {} from session {}", chunk.index, session_id);
                        
                        // 记录要删除的块
                        chunks_to_remove.push(session_id.clone());
                        
                        // 更新会话状态
                        let mut session_clone = session.clone();
                        session_clone.last_activity = Instant::now();
                        session_clone.transferred_chunks += 1;
                        
                        if session_clone.transferred_chunks >= session_clone.total_chunks {
                            session_clone.state = SyncState::Completed;
                        }
                        
                        sessions_to_update.insert(session_id.clone(), session_clone);
                    },
                    Err(e) => {
                        error!("Failed to apply chunk {} from session {}: {}", 
                               chunk.index, session_id, e);
                    }
                }
            }
        }
        
        // 更新会话状态
        if !sessions_to_update.is_empty() {
            let mut sessions = active_sessions.write().map_err(|e| {
                Error::LockError(format!("Failed to acquire active sessions write lock: {}", e))
            })?;
            
            for (id, updated_session) in sessions_to_update {
                sessions.insert(id, updated_session);
            }
        }
        
        // 删除已处理的块
        if !chunks_to_remove.is_empty() {
            let mut chunks = pending_chunks.lock().map_err(|e| {
                Error::LockError(format!("Failed to acquire pending chunks lock: {}", e))
            })?;
            
            for session_id in chunks_to_remove {
                chunks.remove(&session_id);
            }
        }
        
        Ok(processed_count)
    }

    /// 创建同步会话
    pub fn create_sync_session(
        &self,
        target_node: &str,
        sync_type: SyncOperationType,
        start_version: StateVersion,
        end_version: StateVersion,
        priority: SyncPriority
    ) -> Result<String> {
        // 检查是否已达到最大会话数
        {
            let sessions = self.active_sessions.read().map_err(|e| {
                Error::LockError(format!("Failed to acquire active sessions lock: {}", e))
            })?;
            
            if sessions.len() >= self.max_sessions {
                return Err(Error::resource_exhausted("Maximum number of sync sessions reached"));
            }
        }
        
        // 创建新会话
        let session_id = Uuid::new_v4().to_string();
        let now = Instant::now();
        
        let session = SyncSession {
            id: session_id.clone(),
            source_node: self.node_id.clone(),
            target_node: target_node.to_string(),
            start_time: now,
            last_activity: now,
            sync_type,
            start_version,
            end_version,
            state: SyncState::Syncing,
            checksum: None,
            transferred_chunks: 0,
            total_chunks: 0, // 将在后续计算
            retry_count: 0,
        };
        
        // 添加到活跃会话
        {
            let mut sessions = self.active_sessions.write().map_err(|e| {
                Error::LockError(format!("Failed to acquire active sessions write lock: {}", e))
            })?;
            
            sessions.insert(session_id.clone(), session);
        }
        
        // 启动同步过程
        self.start_sync_process(&session_id)?;
        
        info!("Created sync session {} to node {}", session_id, target_node);
        Ok(session_id)
    }

    /// 启动同步过程
    fn start_sync_process(&self, session_id: &str) -> Result<()> {
        let session = {
            let sessions = self.active_sessions.read().map_err(|e| {
                Error::LockError(format!("Failed to acquire active sessions lock: {}", e))
            })?;
            
            match sessions.get(session_id) {
                Some(s) => s.clone(),
                None => return Err(Error::not_found(format!("Sync session {} not found", session_id))),
            }
        };
        
        // 获取状态提供者
        let provider = {
            let provider_guard = self.state_provider.lock().map_err(|e| {
                Error::LockError(format!("Failed to acquire state provider lock: {}", e))
            })?;
            
            if provider_guard.is_none() {
                return Err(Error::invalid_state("State provider not set"));
            }
            
            provider_guard.as_ref().unwrap()
        };
        
        // 获取状态数据
        let state_data = provider.get_state_data(
            session.start_version,
            session.end_version,
            None, // 获取所有数据
        )?;
        
        // 计算总块数
        let total_chunks = (state_data.len() + self.max_chunk_size - 1) / self.max_chunk_size;
        
        // 更新会话信息
        {
            let mut sessions = self.active_sessions.write().map_err(|e| {
                Error::LockError(format!("Failed to acquire active sessions write lock: {}", e))
            })?;
            
            if let Some(s) = sessions.get_mut(session_id) {
                s.total_chunks = total_chunks;
            }
        }
        
        // 将数据拆分为多个块
        let mut chunks = Vec::new();
        for i in 0..total_chunks {
            let start_idx = i * self.max_chunk_size;
            let end_idx = std::cmp::min(start_idx + self.max_chunk_size, state_data.len());
            
            let chunk_data = state_data[start_idx..end_idx].to_vec();
            let chunk_id = format!("{}_{}", session_id, i);
            
            // 计算校验和
            let checksum = Self::calculate_checksum(&chunk_data)?;
            
            let chunk = SyncChunk {
                id: chunk_id,
                session_id: session_id.to_string(),
                index: i,
                data: chunk_data,
                checksum,
                version: session.end_version,
            };
            
            chunks.push(chunk);
        }
        
        // 发送数据块
        self.send_chunks(session.target_node.as_str(), chunks)?;
        
        info!("Started sync process for session {}", session_id);
        Ok(())
    }

    /// 发送数据块
    fn send_chunks(&self, target_node: &str, chunks: Vec<SyncChunk>) -> Result<()> {
        // 在实际实现中，这里应该使用网络传输将块发送到目标节点
        // 这里简化为直接将块添加到目标节点的挂起块队列中
        
        // 注意：在分布式系统中，这里需要使用适当的网络协议和消息格式
        
        // 模拟将块发送到目标节点
        info!("Sending {} chunks to node {}", chunks.len(), target_node);
        
        // 实际实现可能需要使用消息队列、gRPC或其他RPC机制
        
        Ok(())
    }

    /// 接收数据块
    pub fn receive_chunk(&self, chunk: SyncChunk) -> Result<()> {
        // 验证块校验和
        let calculated_checksum = Self::calculate_checksum(&chunk.data)?;
        if calculated_checksum != chunk.checksum {
            return Err(Error::data_corruption(
                format!("Chunk checksum mismatch: expected {}, got {}", 
                       chunk.checksum, calculated_checksum)
            ));
        }
        
        // 添加到挂起块队列
        let mut chunks = self.pending_chunks.lock().map_err(|e| {
            Error::LockError(format!("Failed to acquire pending chunks lock: {}", e))
        })?;
        
        let session_chunks = chunks.entry(chunk.session_id.clone()).or_insert_with(Vec::new);
        session_chunks.push(chunk.clone());
        
        debug!("Received chunk {} for session {}", chunk.index, chunk.session_id);
        Ok(())
    }

    /// 获取会话状态
    pub fn get_session_status(&self, session_id: &str) -> Result<Option<SyncState>> {
        let sessions = self.active_sessions.read().map_err(|e| {
            Error::LockError(format!("Failed to acquire active sessions lock: {}", e))
        })?;
        
        if let Some(session) = sessions.get(session_id) {
            Ok(Some(session.state))
        } else {
            Ok(None)
        }
    }

    /// 取消同步会话
    pub fn cancel_sync_session(&self, session_id: &str) -> Result<bool> {
        let mut session_found = false;
        
        // 从活跃会话中删除
        {
            let mut sessions = self.active_sessions.write().map_err(|e| {
                Error::LockError(format!("Failed to acquire active sessions write lock: {}", e))
            })?;
            
            if sessions.remove(session_id).is_some() {
                session_found = true;
            }
        }
        
        // 清理挂起的块
        if session_found {
            let mut chunks = self.pending_chunks.lock().map_err(|e| {
                Error::LockError(format!("Failed to acquire pending chunks lock: {}", e))
            })?;
            
            chunks.remove(session_id);
            
            info!("Cancelled sync session {}", session_id);
        }
        
        Ok(session_found)
    }

    /// 计算校验和
    fn calculate_checksum<T: AsRef<[u8]>>(data: &[T]) -> Result<String> {
        use sha2::{Sha256, Digest};
        
        let mut hasher = Sha256::new();
        
        for item in data {
            let bytes = item.as_ref();
            hasher.update(bytes);
        }
        
        let result = hasher.finalize();
        let checksum = format!("{:x}", result);
        
        Ok(checksum)
    }

    /// 获取所有活跃会话
    pub fn get_active_sessions(&self) -> Result<Vec<SyncSession>> {
        let sessions = self.active_sessions.read().map_err(|e| {
            Error::LockError(format!("Failed to acquire active sessions lock: {}", e))
        })?;
        
        let session_list = sessions.values().cloned().collect();
        Ok(session_list)
    }

    /// 获取同步统计信息
    pub fn get_sync_stats(&self) -> Result<SyncStats> {
        let sessions = self.active_sessions.read().map_err(|e| {
            Error::LockError(format!("Failed to acquire active sessions lock: {}", e))
        })?;
        
        let total_sessions = sessions.len();
        let mut completed_sessions = 0;
        let mut failed_sessions = 0;
        let mut active_sessions = 0;
        let mut total_chunks_transferred = 0;
        let mut total_chunks_pending = 0;
        
        for session in sessions.values() {
            match session.state {
                SyncState::Completed => completed_sessions += 1,
                SyncState::Failed => failed_sessions += 1,
                _ => active_sessions += 1,
            }
            
            total_chunks_transferred += session.transferred_chunks;
            total_chunks_pending += session.total_chunks - session.transferred_chunks;
        }
        
        Ok(SyncStats {
            total_sessions,
            completed_sessions,
            failed_sessions,
            active_sessions,
            total_chunks_transferred,
            total_chunks_pending,
        })
    }
}

/// 同步统计信息
#[derive(Debug, Clone)]
pub struct SyncStats {
    /// 总会话数
    pub total_sessions: usize,
    /// 已完成会话数
    pub completed_sessions: usize,
    /// 失败会话数
    pub failed_sessions: usize,
    /// 活跃会话数
    pub active_sessions: usize,
    /// 已传输块数
    pub total_chunks_transferred: usize,
    /// 待传输块数
    pub total_chunks_pending: usize,
}

/// 状态同步实现提供者
pub struct DefaultStateProvider {
    /// 存储引擎
    storage: Arc<dyn crate::storage::engine::StorageEngine>,
    /// 版本索引
    version_index: Arc<RwLock<HashMap<StateVersion, HashSet<Vec<u8>>>>>,
}

impl DefaultStateProvider {
    /// 创建新的状态提供者
    pub fn new(storage: Arc<dyn crate::storage::engine::StorageEngine>) -> Self {
        Self {
            storage,
            version_index: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 记录键的版本
    pub fn record_version(&self, key: &[u8], version: StateVersion) -> Result<()> {
        let mut version_index = self.version_index.write().map_err(|e| {
            Error::LockError(format!("Failed to acquire version index lock: {}", e))
        })?;
        
        let keys = version_index.entry(version).or_insert_with(HashSet::new);
        keys.insert(key.to_vec());
        
        Ok(())
    }

    /// 获取版本中的所有键
    pub fn get_keys_for_version(&self, version: StateVersion) -> Result<Option<HashSet<Vec<u8>>>> {
        let version_index = self.version_index.read().map_err(|e| {
            Error::LockError(format!("Failed to acquire version index lock: {}", e))
        })?;
        
        Ok(version_index.get(&version).cloned())
    }
}

impl StateProvider for DefaultStateProvider {
    fn get_current_version(&self) -> Result<StateVersion> {
        // 简单实现：返回最高版本
        let version_index = self.version_index.read().map_err(|e| {
            Error::LockError(format!("Failed to acquire version index lock: {}", e))
        })?;
        
        if version_index.is_empty() {
            return Ok(0);
        }
        
        let max_version = *version_index.keys().max().unwrap_or(&0);
        Ok(max_version)
    }
    
    fn get_state_data(&self, start_version: StateVersion, end_version: StateVersion, keys: Option<&HashSet<Vec<u8>>>) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut result = Vec::new();
        let version_index = self.version_index.read().map_err(|e| {
            Error::LockError(format!("Failed to acquire version index lock: {}", e))
        })?;
        
        // 收集指定版本范围内的所有键
        let mut all_keys = HashSet::new();
        for version in start_version..=end_version {
            if let Some(version_keys) = version_index.get(&version) {
                for key in version_keys {
                    all_keys.insert(key.clone());
                }
            }
        }
        
        // 如果指定了键集合，过滤结果
        let keys_to_fetch = if let Some(filter_keys) = keys {
            all_keys.iter()
                .filter(|k| filter_keys.contains(*k))
                .cloned()
                .collect::<Vec<_>>()
        } else {
            all_keys.into_iter().collect::<Vec<_>>()
        };
        
        // 获取键值对
        for key in keys_to_fetch {
            if let Some(value) = self.storage.get(&key)? {
                result.push((key, value));
            }
        }
        
        Ok(result)
    }
    
    fn get_state_checksum(&self, version: StateVersion) -> Result<String> {
        // 获取版本的所有数据
        let data = self.get_state_data(version, version, None)?;
        
        // 计算校验和
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        
        // 确保数据按键排序，以确保校验和的一致性
        let mut sorted_data = data.clone();
        sorted_data.sort_by(|a, b| a.0.cmp(&b.0));
        
        for (key, value) in sorted_data {
            hasher.update(&key);
            hasher.update(&value);
        }
        
        let result = hasher.finalize();
        let checksum = format!("{:x}", result);
        
        Ok(checksum)
    }
}

/// 状态应用实现
pub struct DefaultStateApplier {
    /// 存储引擎
    storage: Arc<dyn crate::storage::engine::StorageEngine>,
    /// 版本索引
    version_index: Arc<RwLock<HashMap<StateVersion, HashSet<Vec<u8>>>>>,
    /// 当前版本
    current_version: Arc<RwLock<StateVersion>>,
    /// 验证级别
    verification_level: crate::storage::replication::state_sync::VerificationLevel,
}

impl DefaultStateApplier {
    /// 创建新的状态应用器
    pub fn new(
        storage: Arc<dyn crate::storage::engine::StorageEngine>,
        verification_level: crate::storage::replication::state_sync::VerificationLevel
    ) -> Self {
        Self {
            storage,
            version_index: Arc::new(RwLock::new(HashMap::new())),
            current_version: Arc::new(RwLock::new(0)),
            verification_level,
        }
    }

    /// 更新当前版本
    pub fn update_current_version(&self, version: StateVersion) -> Result<()> {
        let mut current_version = self.current_version.write().map_err(|e| {
            Error::LockError(format!("Failed to acquire current version lock: {}", e))
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

    /// 记录版本索引
    fn record_version(&self, keys: &[Vec<u8>], version: StateVersion) -> Result<()> {
        let mut version_index = self.version_index.write().map_err(|e| {
            Error::LockError(format!("Failed to acquire version index lock: {}", e))
        })?;
        
        let version_keys = version_index.entry(version).or_insert_with(HashSet::new);
        
        for key in keys {
            version_keys.insert(key.clone());
        }
        
        Ok(())
    }
}

impl StateApplier for DefaultStateApplier {
    fn apply_state_data(&self, version: StateVersion, data: Vec<(Vec<u8>, Vec<u8>)>) -> Result<()> {
        // 检查版本顺序
        let current_version = self.get_current_version()?;
        if version < current_version {
            return Err(Error::invalid_argument(
                format!("Cannot apply version {} as current version is {}", version, current_version)
            ));
        }
        
        // 批量写入数据
        let mut batch = WriteBatch::default();
        let keys = data.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>();
        
        for (key, value) in data {
            batch.put(&key, &value);
        }
        
        self.storage.write_batch(&batch)?;
        
        // 记录版本索引
        self.record_version(&keys, version)?;
        
        // 更新当前版本
        self.update_current_version(version)?;
        
        Ok(())
    }
    
    fn verify_state(&self, version: StateVersion, checksum: &str) -> Result<bool> {
        match self.verification_level {
            crate::storage::replication::state_sync::VerificationLevel::None => {
                // 不进行验证，直接返回成功
                Ok(true)
            },
            crate::storage::replication::state_sync::VerificationLevel::Checksum => {
                // 获取版本的所有键
                let version_index = self.version_index.read().map_err(|e| {
                    Error::LockError(format!("Failed to acquire version index lock: {}", e))
                })?;
                
                let keys = match version_index.get(&version) {
                    Some(ks) => ks.clone(),
                    None => return Err(Error::not_found(format!("Version {} not found", version))),
                };
                
                // 获取所有数据
                let mut data = Vec::new();
                for key in keys {
                    if let Some(value) = self.storage.get(&key)? {
                        data.push((key, value));
                    }
                }
                
                // 计算校验和
                use sha2::{Sha256, Digest};
                let mut hasher = Sha256::new();
                
                // 确保数据按键排序，以确保校验和的一致性
                let mut sorted_data = data;
                sorted_data.sort_by(|a, b| a.0.cmp(&b.0));
                
                for (key, value) in sorted_data {
                    hasher.update(&key);
                    hasher.update(&value);
                }
                
                let result = hasher.finalize();
                let calculated = format!("{:x}", result);
                
                Ok(calculated == checksum)
            },
            crate::storage::replication::state_sync::VerificationLevel::Full => {
                // 全面验证需要比较所有数据，此处简化为校验和验证
                self.verify_state(version, checksum)
            },
            crate::storage::replication::state_sync::VerificationLevel::Sampling => {
                // 抽样验证，简化为校验和验证
                self.verify_state(version, checksum)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // 添加测试用例
} 