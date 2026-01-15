/// 跨节点数据一致性模块
/// 
/// 此模块实现了分布式环境下的数据一致性保证，包括：
/// - MVCC（多版本并发控制）
/// - 分布式锁管理
/// - 事务协调
/// - 冲突检测和解决
/// - 共识协议

use std::sync::{Arc, Mutex, RwLock};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
 
use uuid::Uuid;
use tokio::time::sleep;
use chrono::{DateTime, Utc};
use log::{info, warn, debug};

use crate::error::{Error, Result};

/// 事务ID类型
pub type TransactionId = String;

/// 节点ID类型  
pub type NodeId = String;

/// 时间戳类型
pub type Timestamp = u64;

/// MVCC配置
#[derive(Debug, Clone)]
pub struct MVCCConfig {
    /// 最大版本保留数量
    pub max_versions: usize,
    /// 垃圾回收间隔
    pub gc_interval: Duration,
    /// 读超时
    pub read_timeout: Duration,
    /// 写超时
    pub write_timeout: Duration,
}

impl Default for MVCCConfig {
    fn default() -> Self {
        Self {
            max_versions: 10,
            gc_interval: Duration::from_secs(300),
            read_timeout: Duration::from_secs(10),
            write_timeout: Duration::from_secs(30),
        }
    }
}

/// MVCC控制器
pub struct MVCCController {
    config: MVCCConfig,
    versions: Arc<RwLock<HashMap<String, Vec<DataVersion>>>>,
    active_transactions: Arc<RwLock<HashMap<TransactionId, TransactionInfo>>>,
    timestamp_oracle: Arc<Mutex<TimestampOracle>>,
}

/// 数据版本
#[derive(Debug, Clone)]
pub struct DataVersion {
    pub key: String,
    pub value: Vec<u8>,
    pub version: Timestamp,
    pub transaction_id: TransactionId,
    pub created_at: DateTime<Utc>,
    pub is_deleted: bool,
}

/// 时间戳预言机
#[derive(Debug)]
struct TimestampOracle {
    current: Timestamp,
}

impl TimestampOracle {
    fn new() -> Self {
        Self { current: 0 }
    }

    fn next_timestamp(&mut self) -> Timestamp {
        self.current += 1;
        self.current
    }
}

impl MVCCController {
    pub fn new(config: MVCCConfig) -> Self {
        Self {
            config,
            versions: Arc::new(RwLock::new(HashMap::new())),
            active_transactions: Arc::new(RwLock::new(HashMap::new())),
            timestamp_oracle: Arc::new(Mutex::new(TimestampOracle::new())),
        }
    }

    /// 开始事务
    pub fn begin_transaction(&self) -> Result<TransactionId> {
        let transaction_id = Uuid::new_v4().to_string();
        let timestamp = self.timestamp_oracle.lock().unwrap().next_timestamp();
        
        let transaction = TransactionInfo {
            id: transaction_id.clone(),
            start_timestamp: timestamp,
            status: TransactionStatus::Active,
            read_set: HashSet::new(),
            write_set: HashMap::new(),
            created_at: Utc::now(),
        };

        self.active_transactions.write().unwrap()
            .insert(transaction_id.clone(), transaction);

        info!("开始事务: {}", transaction_id);
        Ok(transaction_id)
    }

    /// 读取数据
    pub fn read(&self, transaction_id: &TransactionId, key: &str) -> Result<Option<Vec<u8>>> {
        let mut transactions = self.active_transactions.write().unwrap();
        let transaction = transactions.get_mut(transaction_id)
            .ok_or_else(|| Error::validation_error("事务不存在"))?;

        if transaction.status != TransactionStatus::Active {
            return Err(Error::validation_error("事务不是活跃状态"));
        }

        // 添加到读集合
        transaction.read_set.insert(key.to_string());

        let versions = self.versions.read().unwrap();
        if let Some(version_list) = versions.get(key) {
            // 找到最新的可见版本
            for version in version_list.iter().rev() {
                if version.version <= transaction.start_timestamp && !version.is_deleted {
                    debug!("事务 {} 读取键 {} 的版本 {}", transaction_id, key, version.version);
                    return Ok(Some(version.value.clone()));
                }
            }
        }

        Ok(None)
    }

    /// 写入数据
    pub fn write(&self, transaction_id: &TransactionId, key: &str, value: Vec<u8>) -> Result<()> {
        let mut transactions = self.active_transactions.write().unwrap();
        let transaction = transactions.get_mut(transaction_id)
            .ok_or_else(|| Error::validation_error("事务不存在"))?;

        if transaction.status != TransactionStatus::Active {
            return Err(Error::validation_error("事务不是活跃状态"));
        }

        // 添加到写集合
        transaction.write_set.insert(key.to_string(), value);
        debug!("事务 {} 写入键 {}", transaction_id, key);
        Ok(())
    }

    /// 提交事务
    pub fn commit_transaction(&self, transaction_id: &TransactionId) -> Result<()> {
        let commit_timestamp = self.timestamp_oracle.lock().unwrap().next_timestamp();
        
        let mut transactions = self.active_transactions.write().unwrap();
        let mut transaction = transactions.remove(transaction_id)
            .ok_or_else(|| Error::validation_error("事务不存在"))?;

        if transaction.status != TransactionStatus::Active {
            return Err(Error::validation_error("事务不是活跃状态"));
        }

        // 验证读集合（快照隔离）
        let versions = self.versions.read().unwrap();
        for key in &transaction.read_set {
            if let Some(version_list) = versions.get(key) {
                for version in version_list {
                    if version.version > transaction.start_timestamp && 
                       version.version < commit_timestamp {
                        transaction.status = TransactionStatus::Aborted;
                        return Err(Error::validation_error("读写冲突，事务中止"));
                    }
                }
            }
        }

        // 提交写集合
        drop(versions);
        let mut versions = self.versions.write().unwrap();
        
        for (key, value) in transaction.write_set {
            let version = DataVersion {
                key: key.clone(),
                value,
                version: commit_timestamp,
                transaction_id: transaction_id.clone(),
                created_at: Utc::now(),
                is_deleted: false,
            };

            versions.entry(key).or_insert_with(Vec::new).push(version);
        }

        transaction.status = TransactionStatus::Committed;
        info!("事务 {} 提交成功", transaction_id);
        Ok(())
    }

    /// 中止事务
    pub fn abort_transaction(&self, transaction_id: &TransactionId) -> Result<()> {
        let mut transactions = self.active_transactions.write().unwrap();
        if let Some(mut transaction) = transactions.remove(transaction_id) {
            transaction.status = TransactionStatus::Aborted;
            info!("事务 {} 已中止", transaction_id);
        }
        Ok(())
    }

    /// 垃圾回收
    pub fn garbage_collect(&self) -> Result<usize> {
        let mut versions = self.versions.write().unwrap();
        let mut collected = 0;

        for (_, version_list) in versions.iter_mut() {
            if version_list.len() > self.config.max_versions {
                let keep_count = self.config.max_versions;
                version_list.sort_by_key(|v| v.version);
                let removed = version_list.len() - keep_count;
                version_list.drain(0..removed);
                collected += removed;
            }
        }

        if collected > 0 {
            info!("垃圾回收清理了 {} 个版本", collected);
        }
        Ok(collected)
    }
}

/// 事务信息
#[derive(Debug, Clone)]
pub struct TransactionInfo {
    pub id: TransactionId,
    pub start_timestamp: Timestamp,
    pub status: TransactionStatus,
    pub read_set: HashSet<String>,
    pub write_set: HashMap<String, Vec<u8>>,
    pub created_at: DateTime<Utc>,
}

/// 事务状态
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransactionStatus {
    Active,
    Committed,
    Aborted,
}

/// 锁配置
#[derive(Debug, Clone)]
pub struct LockConfig {
    pub default_timeout: Duration,
    pub max_concurrent_locks: usize,
    pub deadlock_detection_interval: Duration,
}

impl Default for LockConfig {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_secs(30),
            max_concurrent_locks: 1000,
            deadlock_detection_interval: Duration::from_secs(5),
        }
    }
}

/// 分布式锁管理器
pub struct DistributedLockManager {
    config: LockConfig,
    locks: Arc<RwLock<HashMap<String, LockInfo>>>,
    wait_queue: Arc<Mutex<VecDeque<LockRequest>>>,
    deadlock_detector: Arc<Mutex<DeadlockDetector>>,
}

/// 锁信息
#[derive(Debug, Clone)]
pub struct LockInfo {
    pub resource: String,
    pub holder: String,
    pub lock_type: LockType,
    pub acquired_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub status: LockStatus,
}

/// 锁类型
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockType {
    Shared,
    Exclusive,
}

/// 锁状态
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockStatus {
    Acquired,
    Waiting,
    Released,
    Expired,
}

/// 锁请求
#[derive(Debug, Clone)]
pub struct LockRequest {
    pub id: String,
    pub resource: String,
    pub requester: String,
    pub lock_type: LockType,
    pub timeout: Duration,
    pub requested_at: DateTime<Utc>,
}

/// 死锁检测器
#[derive(Debug)]
struct DeadlockDetector {
    wait_graph: HashMap<String, HashSet<String>>,
}

impl DeadlockDetector {
    fn new() -> Self {
        Self {
            wait_graph: HashMap::new(),
        }
    }

    fn add_wait_edge(&mut self, waiter: &str, holder: &str) {
        self.wait_graph.entry(waiter.to_string())
            .or_insert_with(HashSet::new)
            .insert(holder.to_string());
    }

    fn remove_wait_edge(&mut self, waiter: &str, holder: &str) {
        if let Some(edges) = self.wait_graph.get_mut(waiter) {
            edges.remove(holder);
            if edges.is_empty() {
                self.wait_graph.remove(waiter);
            }
        }
    }

    fn detect_deadlock(&self) -> Option<Vec<String>> {
        // 简化的死锁检测算法
        for node in self.wait_graph.keys() {
            if let Some(cycle) = self.dfs_cycle_detection(node, &mut HashSet::new(), &mut Vec::new()) {
                return Some(cycle);
            }
        }
        None
    }

    fn dfs_cycle_detection(
        &self,
        current: &str,
        visited: &mut HashSet<String>,
        path: &mut Vec<String>,
    ) -> Option<Vec<String>> {
        if path.contains(&current.to_string()) {
            // 找到环
            let cycle_start = path.iter().position(|x| x == current).unwrap();
            return Some(path[cycle_start..].to_vec());
        }

        if visited.contains(current) {
            return None;
        }

        visited.insert(current.to_string());
        path.push(current.to_string());

        if let Some(neighbors) = self.wait_graph.get(current) {
            for neighbor in neighbors {
                if let Some(cycle) = self.dfs_cycle_detection(neighbor, visited, path) {
                    return Some(cycle);
                }
            }
        }

        path.pop();
        None
    }
}

impl DistributedLockManager {
    pub fn new(config: LockConfig) -> Self {
        Self {
            config,
            locks: Arc::new(RwLock::new(HashMap::new())),
            wait_queue: Arc::new(Mutex::new(VecDeque::new())),
            deadlock_detector: Arc::new(Mutex::new(DeadlockDetector::new())),
        }
    }

    /// 请求锁
    pub async fn acquire_lock(
        &self,
        resource: &str,
        requester: &str,
        lock_type: LockType,
        timeout: Option<Duration>,
    ) -> Result<String> {
        let request_id = Uuid::new_v4().to_string();
        let timeout = timeout.unwrap_or(self.config.default_timeout);

        let request = LockRequest {
            id: request_id.clone(),
            resource: resource.to_string(),
            requester: requester.to_string(),
            lock_type: lock_type.clone(),
            timeout,
            requested_at: Utc::now(),
        };

        // 尝试立即获取锁
        if self.try_acquire_immediate(&request)? {
            return Ok(request_id);
        }

        // 加入等待队列
        self.wait_queue.lock().unwrap().push_back(request.clone());

        // 等待锁可用
        let start_time = Instant::now();
        loop {
            if start_time.elapsed() > timeout {
                self.remove_from_wait_queue(&request_id);
                return Err(Error::timeout("锁获取超时"));
            }

            // 检查死锁
            if let Some(cycle) = self.deadlock_detector.lock().unwrap().detect_deadlock() {
                warn!("检测到死锁: {:?}", cycle);
                if cycle.contains(&requester.to_string()) {
                    self.remove_from_wait_queue(&request_id);
                    return Err(Error::lock("检测到死锁"));
                }
            }

            if self.try_acquire_immediate(&request)? {
                self.remove_from_wait_queue(&request_id);
                return Ok(request_id);
            }

            sleep(Duration::from_millis(100)).await;
        }
    }

    fn try_acquire_immediate(&self, request: &LockRequest) -> Result<bool> {
        let mut locks = self.locks.write().unwrap();
        
        if let Some(existing_lock) = locks.get(&request.resource) {
            // 检查锁是否过期
            if let Some(expires_at) = existing_lock.expires_at {
                if Utc::now() > expires_at {
                    locks.remove(&request.resource);
                } else {
                    // 检查锁兼容性
                    match (&existing_lock.lock_type, &request.lock_type) {
                        (LockType::Shared, LockType::Shared) => {
                            // 共享锁兼容
                        }
                        _ => {
                            // 添加等待边到死锁检测器
                            self.deadlock_detector.lock().unwrap()
                                .add_wait_edge(&request.requester, &existing_lock.holder);
                            return Ok(false);
                        }
                    }
                }
            }
        }

        // 创建新锁
        let lock_info = LockInfo {
            resource: request.resource.clone(),
            holder: request.requester.clone(),
            lock_type: request.lock_type.clone(),
            acquired_at: Utc::now(),
            expires_at: Some(Utc::now() + chrono::Duration::from_std(request.timeout).unwrap()),
            status: LockStatus::Acquired,
        };

        locks.insert(request.resource.clone(), lock_info);
        info!("锁已获取: {} by {}", request.resource, request.requester);
        Ok(true)
    }

    fn remove_from_wait_queue(&self, request_id: &str) {
        let mut queue = self.wait_queue.lock().unwrap();
        queue.retain(|req| req.id != request_id);
    }

    /// 释放锁
    pub fn release_lock(&self, resource: &str, holder: &str) -> Result<()> {
        let mut locks = self.locks.write().unwrap();
        
        if let Some(lock_info) = locks.get(resource) {
            if lock_info.holder == holder {
                locks.remove(resource);
                
                // 从死锁检测器中移除相关边
                let mut detector = self.deadlock_detector.lock().unwrap();
                detector.wait_graph.retain(|_, holders| {
                    holders.remove(holder);
                    !holders.is_empty()
                });

                info!("锁已释放: {} by {}", resource, holder);
                Ok(())
            } else {
                Err(Error::permission_denied("不是锁的持有者"))
            }
        } else {
            Err(Error::not_found("锁不存在"))
        }
    }

    /// 获取锁信息
    pub fn get_lock_info(&self, resource: &str) -> Option<LockInfo> {
        self.locks.read().unwrap().get(resource).cloned()
    }

    /// 清理过期锁
    pub fn cleanup_expired_locks(&self) -> usize {
        let mut locks = self.locks.write().unwrap();
        let now = Utc::now();
        let mut removed = 0;

        locks.retain(|_, lock_info| {
            if let Some(expires_at) = lock_info.expires_at {
                if now > expires_at {
                    removed += 1;
                    false
                } else {
                    true
                }
            } else {
                true
            }
        });

        if removed > 0 {
            info!("清理了 {} 个过期锁", removed);
        }
        removed
    }
}

/// 一致性配置
#[derive(Debug, Clone)]
pub struct ConsistencyConfig {
    pub consistency_level: ConsistencyLevel,
    pub read_strategy: ReadStrategy,
    pub write_strategy: WriteStrategy,
    pub conflict_resolution: ConflictResolution,
    pub replication_factor: usize,
    pub timeout: Duration,
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        Self {
            consistency_level: ConsistencyLevel::EventualConsistency,
            read_strategy: ReadStrategy::ReadOne,
            write_strategy: WriteStrategy::WriteQuorum,
            conflict_resolution: ConflictResolution::LastWriteWins,
            replication_factor: 3,
            timeout: Duration::from_secs(30),
        }
    }
}

/// 一致性级别
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsistencyLevel {
    StrongConsistency,
    EventualConsistency,
    SessionConsistency,
    MonotonicReadConsistency,
    MonotonicWriteConsistency,
}

/// 读策略
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadStrategy {
    ReadOne,
    ReadQuorum,
    ReadAll,
    ReadLocal,
}

/// 写策略
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteStrategy {
    WriteOne,
    WriteQuorum,
    WriteAll,
    WriteLocal,
}

/// 冲突解决策略
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictResolution {
    LastWriteWins,
    FirstWriteWins,
    Merge,
    Manual,
    Timestamp,
}

/// 一致性协调器
pub struct ConsistencyCoordinator {
    config: ConsistencyConfig,
    mvcc: Arc<MVCCController>,
    lock_manager: Arc<DistributedLockManager>,
    nodes: Arc<RwLock<HashMap<NodeId, NodeInfo>>>,
}

/// 节点信息
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub id: NodeId,
    pub address: String,
    pub status: NodeStatus,
    pub last_heartbeat: DateTime<Utc>,
    pub weight: f64,
}

/// 节点状态
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeStatus {
    Online,
    Offline,
    Suspected,
    Maintenance,
}

impl ConsistencyCoordinator {
    pub fn new(config: ConsistencyConfig) -> Self {
        let mvcc_config = MVCCConfig::default();
        let lock_config = LockConfig::default();

        Self {
            config,
            mvcc: Arc::new(MVCCController::new(mvcc_config)),
            lock_manager: Arc::new(DistributedLockManager::new(lock_config)),
            nodes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 添加节点
    pub fn add_node(&self, node: NodeInfo) -> Result<()> {
        self.nodes.write().unwrap().insert(node.id.clone(), node);
        Ok(())
    }

    /// 移除节点
    pub fn remove_node(&self, node_id: &NodeId) -> Result<()> {
        self.nodes.write().unwrap().remove(node_id);
        Ok(())
    }

    /// 获取活跃节点
    pub fn get_active_nodes(&self) -> Vec<NodeInfo> {
        self.nodes.read().unwrap()
            .values()
            .filter(|node| node.status == NodeStatus::Online)
            .cloned()
            .collect()
    }

    /// 计算仲裁数
    pub fn quorum_size(&self) -> usize {
        let total_nodes = self.nodes.read().unwrap().len();
        (total_nodes / 2) + 1
    }

    /// 开始分布式事务
    pub async fn begin_distributed_transaction(&self) -> Result<TransactionId> {
        let transaction_id = self.mvcc.begin_transaction()?;
        
        match self.config.consistency_level {
            ConsistencyLevel::StrongConsistency => {
                // 强一致性需要在所有节点上开始事务
                self.coordinate_transaction_start(&transaction_id).await?;
            }
            _ => {
                // 其他一致性级别可以本地开始
            }
        }

        Ok(transaction_id)
    }

    async fn coordinate_transaction_start(&self, _transaction_id: &TransactionId) -> Result<()> {
        // 分布式模式下需要向所有节点发送事务开始请求
        // 当前单节点模式下无需协调，直接返回成功
        Ok(())
    }

    /// 分布式读取
    pub async fn distributed_read(&self, transaction_id: &TransactionId, key: &str) -> Result<Option<Vec<u8>>> {
        match self.config.read_strategy {
            ReadStrategy::ReadOne => {
                self.mvcc.read(transaction_id, key)
            }
            ReadStrategy::ReadQuorum => {
                self.read_with_quorum(transaction_id, key).await
            }
            ReadStrategy::ReadAll => {
                self.read_from_all_nodes(transaction_id, key).await
            }
            ReadStrategy::ReadLocal => {
                self.mvcc.read(transaction_id, key)
            }
        }
    }

    async fn read_with_quorum(&self, transaction_id: &TransactionId, key: &str) -> Result<Option<Vec<u8>>> {
        let quorum_size = self.quorum_size();
        let active_nodes = self.get_active_nodes();

        if active_nodes.len() < quorum_size {
            return Err(Error::resource("没有足够的节点形成仲裁"));
        }

        // 单节点模式下从本地 MVCC 读取
        // 分布式模式下需要从多个节点读取并比较版本
        self.mvcc.read(transaction_id, key)
    }

    async fn read_from_all_nodes(&self, transaction_id: &TransactionId, key: &str) -> Result<Option<Vec<u8>>> {
        // 单节点模式下从本地 MVCC 读取
        // 分布式模式下需要从所有节点读取并合并结果
        self.mvcc.read(transaction_id, key)
    }

    /// 分布式写入
    pub async fn distributed_write(&self, transaction_id: &TransactionId, key: &str, value: Vec<u8>) -> Result<()> {
        match self.config.write_strategy {
            WriteStrategy::WriteOne => {
                self.mvcc.write(transaction_id, key, value)
            }
            WriteStrategy::WriteQuorum => {
                self.write_with_quorum(transaction_id, key, value).await
            }
            WriteStrategy::WriteAll => {
                self.write_to_all_nodes(transaction_id, key, value).await
            }
            WriteStrategy::WriteLocal => {
                self.mvcc.write(transaction_id, key, value)
            }
        }
    }

    async fn write_with_quorum(&self, transaction_id: &TransactionId, key: &str, value: Vec<u8>) -> Result<()> {
        let quorum_size = self.quorum_size();
        let active_nodes = self.get_active_nodes();

        if active_nodes.len() < quorum_size {
            return Err(Error::resource("没有足够的节点形成仲裁"));
        }

        // 单节点模式下写入本地 MVCC
        // 分布式模式下需要写入多个节点并等待确认
        self.mvcc.write(transaction_id, key, value)
    }

    async fn write_to_all_nodes(&self, transaction_id: &TransactionId, key: &str, value: Vec<u8>) -> Result<()> {
        // 单节点模式下写入本地 MVCC
        // 分布式模式下需要写入所有节点并等待全部确认
        self.mvcc.write(transaction_id, key, value)
    }

    /// 提交分布式事务
    pub async fn commit_distributed_transaction(&self, transaction_id: &TransactionId) -> Result<()> {
        match self.config.consistency_level {
            ConsistencyLevel::StrongConsistency => {
                self.two_phase_commit(transaction_id).await
            }
            _ => {
                self.mvcc.commit_transaction(transaction_id)
            }
        }
    }

    async fn two_phase_commit(&self, transaction_id: &TransactionId) -> Result<()> {
        // 第一阶段：准备
        let prepare_success = self.prepare_transaction(transaction_id).await?;
        
        if prepare_success {
            // 第二阶段：提交
            self.commit_transaction_all_nodes(transaction_id).await
        } else {
            // 中止事务
            self.abort_transaction_all_nodes(transaction_id).await?;
            Err(Error::transaction("事务在准备阶段失败"))
        }
    }

    async fn prepare_transaction(&self, _transaction_id: &TransactionId) -> Result<bool> {
        // 单节点模式下无需两阶段提交的准备阶段，直接返回成功
        // 分布式模式下需要向所有节点发送准备请求并等待响应
        Ok(true)
    }

    async fn commit_transaction_all_nodes(&self, transaction_id: &TransactionId) -> Result<()> {
        // 单节点模式下在本地 MVCC 提交
        // 分布式模式下需要向所有节点发送提交请求
        self.mvcc.commit_transaction(transaction_id)
    }

    async fn abort_transaction_all_nodes(&self, transaction_id: &TransactionId) -> Result<()> {
        // 单节点模式下在本地 MVCC 中止
        // 分布式模式下需要向所有节点发送中止请求
        self.mvcc.abort_transaction(transaction_id)
    }
}

/// 工厂函数
pub fn create_consistency_coordinator(config: ConsistencyConfig) -> ConsistencyCoordinator {
    ConsistencyCoordinator::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mvcc_basic_operations() {
        let mvcc = MVCCController::new(MVCCConfig::default());
        
        // 开始事务
        let tx1 = mvcc.begin_transaction().unwrap();
        
        // 写入数据
        mvcc.write(&tx1, "key1", b"value1".to_vec()).unwrap();
        
        // 读取数据（应该为空，因为事务未提交）
        let result = mvcc.read(&tx1, "key1").unwrap();
        assert!(result.is_none());
        
        // 提交事务
        mvcc.commit_transaction(&tx1).unwrap();
        
        // 新事务读取数据
        let tx2 = mvcc.begin_transaction().unwrap();
        let result = mvcc.read(&tx2, "key1").unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), b"value1");
    }

    #[tokio::test]
    async fn test_distributed_lock_manager() {
        let lock_manager = DistributedLockManager::new(LockConfig::default());
        
        // 获取独占锁
        let lock_id = lock_manager.acquire_lock(
            "resource1", 
            "client1", 
            LockType::Exclusive,
            Some(Duration::from_secs(5))
        ).await.unwrap();
        
        // 尝试获取相同资源的锁（应该超时）
        let result = lock_manager.acquire_lock(
            "resource1", 
            "client2", 
            LockType::Exclusive,
            Some(Duration::from_millis(100))
        ).await;
        
        assert!(result.is_err());
        
        // 释放锁
        lock_manager.release_lock("resource1", "client1").unwrap();
        
        // 现在应该可以获取锁了
        let _lock_id2 = lock_manager.acquire_lock(
            "resource1", 
            "client2", 
            LockType::Exclusive,
            Some(Duration::from_secs(5))
        ).await.unwrap();
    }

    #[tokio::test]
    async fn test_consistency_coordinator() {
        let config = ConsistencyConfig::default();
        let coordinator = ConsistencyCoordinator::new(config);
        
        // 开始分布式事务
        let tx_id = coordinator.begin_distributed_transaction().await.unwrap();
        
        // 分布式写入
        coordinator.distributed_write(&tx_id, "test_key", b"test_value".to_vec()).await.unwrap();
        
        // 分布式读取
        let result = coordinator.distributed_read(&tx_id, "test_key").await.unwrap();
        assert!(result.is_none()); // 事务未提交，应该读不到
        
        // 提交事务
        coordinator.commit_distributed_transaction(&tx_id).await.unwrap();
        
        // 新事务读取
        let tx_id2 = coordinator.begin_distributed_transaction().await.unwrap();
        let result = coordinator.distributed_read(&tx_id2, "test_key").await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), b"test_value");
    }
} 