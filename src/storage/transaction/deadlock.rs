// 死锁检测和恢复模块
//
// 提供事务死锁检测和自动恢复机制

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use std::thread;
use log::{debug, error, info, warn};

use crate::error::{Error, Result};
use crate::storage::transaction::TransactionId;

/// 资源ID类型
pub type ResourceId = Vec<u8>;

/// 死锁检测器配置
#[derive(Debug, Clone)]
pub struct DeadlockDetectorConfig {
    /// 检测间隔（毫秒）
    pub detection_interval_ms: u64,
    /// 等待图清理时间（秒）
    pub wait_graph_cleanup_secs: u64,
    /// 最大等待事务数
    pub max_waiting_transactions: usize,
    /// 最大等待时间（毫秒）
    pub max_wait_time_ms: u64,
    /// 死锁解决策略
    pub resolution_strategy: DeadlockResolutionStrategy,
}

impl Default for DeadlockDetectorConfig {
    fn default() -> Self {
        Self {
            detection_interval_ms: 1000,   // 默认每秒检测一次
            wait_graph_cleanup_secs: 300,  // 5分钟清理一次
            max_waiting_transactions: 1000, // 最多1000个等待事务
            max_wait_time_ms: 30000,       // 最多等待30秒
            resolution_strategy: DeadlockResolutionStrategy::AbortYoungest,
        }
    }
}

/// 死锁解决策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeadlockResolutionStrategy {
    /// 中止最年轻的事务
    AbortYoungest,
    /// 中止最老的事务
    AbortOldest,
    /// 中止涉及资源最少的事务
    AbortLeastResources,
    /// 中止优先级最低的事务
    AbortLowestPriority,
}

/// 等待关系信息
#[derive(Debug, Clone)]
struct WaitInfo {
    /// 等待的事务ID
    txn_id: TransactionId,
    /// 等待的资源ID
    resource_id: ResourceId,
    /// 等待开始时间
    wait_start: Instant,
    /// 持有资源的事务ID
    holder_txn_id: TransactionId,
    /// 事务优先级（用于死锁解决）
    priority: u32,
    /// 事务开始时间
    txn_start_time: Instant,
    /// 涉及的资源数量
    resource_count: usize,
}

/// 死锁检测器
/// 
/// 用于检测事务之间的死锁并自动解决
pub struct DeadlockDetector {
    /// 等待图：事务 -> 它等待的事务集合
    wait_for_graph: Arc<RwLock<HashMap<TransactionId, HashSet<TransactionId>>>>,
    
    /// 详细等待信息
    wait_info: Arc<RwLock<HashMap<(TransactionId, ResourceId), WaitInfo>>>,
    
    /// 持有资源信息：资源ID -> 持有的事务
    resource_holders: Arc<RwLock<HashMap<ResourceId, TransactionId>>>,
    
    /// 配置
    config: DeadlockDetectorConfig,
    
    /// 正在运行标志
    running: Arc<RwLock<bool>>,
    
    /// 上次清理时间
    last_cleanup: Arc<RwLock<Instant>>,
    
    /// 死锁处理回调
    deadlock_handler: Arc<Mutex<Option<Box<dyn Fn(TransactionId) -> Result<()> + Send + Sync>>>>,
}

impl DeadlockDetector {
    /// 创建新的死锁检测器
    pub fn new(config: Option<DeadlockDetectorConfig>) -> Self {
        let config = config.unwrap_or_default();
        
        Self {
            wait_for_graph: Arc::new(RwLock::new(HashMap::new())),
            wait_info: Arc::new(RwLock::new(HashMap::new())),
            resource_holders: Arc::new(RwLock::new(HashMap::new())),
            config,
            running: Arc::new(RwLock::new(false)),
            last_cleanup: Arc::new(RwLock::new(Instant::now())),
            deadlock_handler: Arc::new(Mutex::new(None)),
        }
    }

    /// 启动死锁检测
    pub fn start(&self) -> Result<()> {
        let mut running = self.running.write().map_err(|e| {
            Error::LockError(format!("运行状态锁错误: {}", e))
        })?;
        
        if *running {
            return Ok(());
        }
        
        *running = true;
        
        // 启动死锁检测线程
        let wait_for_graph = self.wait_for_graph.clone();
        let wait_info = self.wait_info.clone();
        let resource_holders = self.resource_holders.clone();
        let running_clone = self.running.clone();
        let config = self.config.clone();
        let deadlock_handler = self.deadlock_handler.clone();
        let last_cleanup = self.last_cleanup.clone();
        
        thread::spawn(move || {
            let detection_interval = Duration::from_millis(config.detection_interval_ms);
            
            while {
                let running = running_clone.read().unwrap();
                *running
            } {
                // 检测死锁
                if let Err(e) = Self::detect_deadlocks(
                    &wait_for_graph,
                    &wait_info,
                    &resource_holders,
                    &config,
                    &deadlock_handler,
                ) {
                    error!("Error in deadlock detection: {}", e);
                }
                
                // 清理过期等待
                if let Err(e) = Self::cleanup_expired_waits(
                    &wait_for_graph,
                    &wait_info,
                    &resource_holders,
                    &config,
                ) {
                    error!("Error in wait cleanup: {}", e);
                }
                
                // 检查是否需要清理等待图
                {
                    let mut last_cleanup_time = last_cleanup.write().unwrap();
                    let now = Instant::now();
                    
                    if now.duration_since(*last_cleanup_time).as_secs() > config.wait_graph_cleanup_secs {
                        if let Err(e) = Self::cleanup_wait_graph(
                            &wait_for_graph,
                            &wait_info,
                            &resource_holders,
                        ) {
                            error!("Error in wait graph cleanup: {}", e);
                        }
                        
                        *last_cleanup_time = now;
                    }
                }
                
                // 等待下一次检测
                thread::sleep(detection_interval);
            }
        });
        
        info!("Deadlock detector started");
        Ok(())
    }

    /// 停止死锁检测
    pub fn stop(&self) -> Result<()> {
        let mut running = self.running.write().map_err(|e| {
            Error::LockError(format!("运行状态锁错误: {}", e))
        })?;
        
        if !*running {
            return Ok(());
        }
        
        *running = false;
        
        info!("Deadlock detector stopped");
        Ok(())
    }

    /// 设置死锁处理回调
    pub fn set_deadlock_handler<F>(&self, handler: F) -> Result<()>
    where
        F: Fn(TransactionId) -> Result<()> + Send + Sync + 'static,
    {
        let mut deadlock_handler = self.deadlock_handler.lock().map_err(|e| {
            Error::LockError(format!("死锁处理器锁错误: {}", e))
        })?;
        
        *deadlock_handler = Some(Box::new(handler));
        
        Ok(())
    }

    /// 注册资源锁请求
    pub fn register_lock_request(&self, txn_id: &TransactionId, resource_id: ResourceId, txn_start_time: Instant, priority: u32, resource_count: usize) -> Result<bool> {
        // 检查资源是否已被持有
        let holder_opt = {
                let resource_holders = self.resource_holders.read().map_err(|e| {
                    Error::LockError(format!("资源持有者读锁错误: {}", e))
                })?;
            
            resource_holders.get(&resource_id).cloned()
        };
        
        if let Some(holder_txn_id) = holder_opt {
            // 资源已被其他事务持有
            if holder_txn_id == *txn_id {
                // 自己已经持有该资源，无需等待
                return Ok(true);
            }
            
            // 检查是否存在死锁
            if self.would_cause_deadlock(txn_id, &holder_txn_id)? {
                warn!("Deadlock detected: transaction {} would wait for {}", txn_id, holder_txn_id);
                // 存在死锁，拒绝锁请求
                return Ok(false);
            }
            
            // 记录等待信息
            self.record_wait(txn_id, &resource_id, &holder_txn_id, txn_start_time, priority, resource_count)?;
            
            // 资源被占用，需要等待
            return Ok(false);
        } else {
            // 资源未被持有，可以获取锁
            self.acquire_resource(txn_id, &resource_id)?;
            return Ok(true);
        }
    }

    /// 获取资源锁
    pub fn acquire_resource(&self, txn_id: &TransactionId, resource_id: &ResourceId) -> Result<()> {
        let mut resource_holders = self.resource_holders.write().map_err(|e| {
            Error::LockError(format!("资源持有者写锁错误: {}", e))
        })?;
        
        resource_holders.insert(resource_id.clone(), txn_id.clone());
        
        debug!("Transaction {} acquired lock on resource {:?}", txn_id, resource_id);
        Ok(())
    }

    /// 释放资源锁
    pub fn release_resource(&self, txn_id: &TransactionId, resource_id: &ResourceId) -> Result<()> {
        // 释放资源
        {
            let mut resource_holders = self.resource_holders.write().map_err(|e| {
                Error::LockError(format!("资源持有者写锁错误: {}", e))
            })?;
            
            // 检查是否仍然持有该资源
            if let Some(holder) = resource_holders.get(resource_id) {
                if holder == txn_id {
                    resource_holders.remove(resource_id);
                    debug!("Transaction {} released lock on resource {:?}", txn_id, resource_id);
                }
            }
        }
        
        // 清理等待信息
        {
            let mut wait_for_graph = self.wait_for_graph.write().map_err(|e| {
                Error::LockError(format!("等待图写锁错误: {}", e))
            })?;
            
            // 移除其他事务对该事务的等待
            for (_, waiters) in wait_for_graph.iter_mut() {
                waiters.remove(txn_id);
            }
            
            // 移除该事务的等待项
            wait_for_graph.remove(txn_id);
        }
        
        {
            let mut wait_info = self.wait_info.write().map_err(|e| {
                Error::LockError(format!("等待信息写锁错误: {}", e))
            })?;
            
            // 移除所有与该资源相关的等待信息
            wait_info.retain(|&(waiter_id, ref res_id), _| {
                !(res_id == resource_id && waiter_id == *txn_id)
            });
        }
        
        Ok(())
    }

    /// 清理事务相关的所有资源和等待
    pub fn cleanup_transaction(&self, txn_id: &TransactionId) -> Result<()> {
        // 找出事务持有的所有资源
        let resources_to_release = {
                let resource_holders = self.resource_holders.read().map_err(|e| {
                    Error::LockError(format!("资源持有者读锁错误: {}", e))
                })?;
            
            resource_holders.iter()
                .filter(|(_, holder)| *holder == txn_id)
                .map(|(res_id, _)| res_id.clone())
                .collect::<Vec<_>>()
        };
        
        // 释放所有资源
        for resource_id in resources_to_release {
            self.release_resource(txn_id, &resource_id)?;
        }
        
        // 清理等待图
        {
            let mut wait_for_graph = self.wait_for_graph.write().map_err(|e| {
                Error::LockError(format!("等待图写锁错误: {}", e))
            })?;
            
            // 移除其他事务对该事务的等待
            for (_, waiters) in wait_for_graph.iter_mut() {
                waiters.remove(txn_id);
            }
            
            // 移除该事务的等待项
            wait_for_graph.remove(txn_id);
        }
        
        // 清理等待信息
        {
            let mut wait_info = self.wait_info.write().map_err(|e| {
                Error::LockError(format!("等待信息写锁错误: {}", e))
            })?;
            
            // 移除所有与该事务相关的等待信息
            wait_info.retain(|&(waiter_id, _), info| {
                waiter_id != *txn_id && info.holder_txn_id != *txn_id
            });
        }
        
        debug!("Cleaned up all resources and waits for transaction {}", txn_id);
        Ok(())
    }

    /// 记录等待信息
    fn record_wait(&self, txn_id: &TransactionId, resource_id: &ResourceId, holder_txn_id: &TransactionId, txn_start_time: Instant, priority: u32, resource_count: usize) -> Result<()> {
        // 记录等待信息
        {
            let mut wait_info = self.wait_info.write().map_err(|e| {
                Error::LockError(format!("等待信息写锁错误: {}", e))
            })?;
            
            wait_info.insert(
                (txn_id.clone(), resource_id.clone()),
                WaitInfo {
                    txn_id: txn_id.clone(),
                    resource_id: resource_id.clone(),
                    wait_start: Instant::now(),
                    holder_txn_id: holder_txn_id.clone(),
                    priority,
                    txn_start_time,
                    resource_count,
                },
            );
        }
        
        // 更新等待图
        {
            let mut wait_for_graph = self.wait_for_graph.write().map_err(|e| {
                Error::LockError(format!("等待图写锁错误: {}", e))
            })?;
            
            let waiting_set = wait_for_graph.entry(txn_id.clone()).or_insert_with(HashSet::new);
            waiting_set.insert(holder_txn_id.clone());
        }
        
        debug!("Transaction {} waiting for resource {:?} held by {}", txn_id, resource_id, holder_txn_id);
        Ok(())
    }

    /// 检查请求是否会导致死锁
    fn would_cause_deadlock(&self, txn_id: &TransactionId, holder_txn_id: &TransactionId) -> Result<bool> {
            let wait_for_graph = self.wait_for_graph.read().map_err(|e| {
                Error::LockError(format!("等待图读锁错误: {}", e))
            })?;
        
        // 检查是否存在环：holder_txn_id 是否直接或间接等待 txn_id
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        // 从holder_txn_id开始BFS
        if let Some(waiters) = wait_for_graph.get(holder_txn_id) {
            queue.extend(waiters.iter().cloned());
            visited.insert(holder_txn_id.clone());
        }
        
        while let Some(current) = queue.pop_front() {
            if current == *txn_id {
                // 找到环，会导致死锁
                return Ok(true);
            }
            
            visited.insert(current.clone());
            
            if let Some(waiters) = wait_for_graph.get(&current) {
                for waiter in waiters {
                    if !visited.contains(waiter) {
                        queue.push_back(waiter.clone());
                    }
                }
            }
        }
        
        // 没有找到环，不会导致死锁
        Ok(false)
    }

    /// 检测死锁
    fn detect_deadlocks(
        wait_for_graph: &RwLock<HashMap<TransactionId, HashSet<TransactionId>>>,
        wait_info: &RwLock<HashMap<(TransactionId, ResourceId), WaitInfo>>,
        resource_holders: &RwLock<HashMap<ResourceId, TransactionId>>,
        config: &DeadlockDetectorConfig,
        deadlock_handler: &Mutex<Option<Box<dyn Fn(TransactionId) -> Result<()> + Send + Sync>>>,
    ) -> Result<Vec<Vec<TransactionId>>> {
        let graph = wait_for_graph.read().map_err(|e| {
            Error::LockError(format!("等待图读锁错误: {}", e))
        })?;
        
        // 使用Tarjan算法查找强连通分量（环）
        let mut cycles = Vec::new();
        let mut visited = HashMap::new();
        let mut stack = Vec::new();
        let mut index = 0;
        let mut low_link = HashMap::new();
        let mut on_stack = HashSet::new();
        
        for node in graph.keys() {
            if !visited.contains_key(node) {
                Self::tarjan_scc(
                    node,
                    &graph,
                    &mut index,
                    &mut visited,
                    &mut low_link,
                    &mut stack,
                    &mut on_stack,
                    &mut cycles,
                )?;
            }
        }
        
        // 解决死锁
        for cycle in &cycles {
            // 跳过自循环
            if cycle.len() <= 1 {
                continue;
            }
            
            info!("Deadlock detected: {:?}", cycle);
            
            // 选择一个事务中止
            if let Some(victim) = Self::choose_victim(cycle, wait_info, config)? {
                info!("Choosing transaction {} as victim to resolve deadlock", victim);
                
                // 调用死锁处理回调
                let handler = deadlock_handler.lock().map_err(|e| {
                    Error::LockError(format!("死锁处理器锁错误: {}", e))
                })?;
                
                if let Some(ref handler_fn) = *handler {
                    if let Err(e) = handler_fn(victim.clone()) {
                        error!("Error in deadlock handler: {}", e);
                    }
                } else {
                    warn!("No deadlock handler registered, deadlock may persist");
                }
            }
        }
        
        Ok(cycles)
    }

    /// Tarjan算法查找强连通分量（环）
    fn tarjan_scc(
        node: &TransactionId,
        graph: &HashMap<TransactionId, HashSet<TransactionId>>,
        index: &mut usize,
        visited: &mut HashMap<TransactionId, usize>,
        low_link: &mut HashMap<TransactionId, usize>,
        stack: &mut Vec<TransactionId>,
        on_stack: &mut HashSet<TransactionId>,
        cycles: &mut Vec<Vec<TransactionId>>,
    ) -> Result<()> {
        // 设置当前节点的索引和低链接值
        let current_index = *index;
        visited.insert(node.clone(), current_index);
        low_link.insert(node.clone(), current_index);
        *index += 1;
        stack.push(node.clone());
        on_stack.insert(node.clone());
        
        // 检查所有邻居
        if let Some(neighbors) = graph.get(node) {
            for neighbor in neighbors {
                if !visited.contains_key(neighbor) {
                    // 邻居尚未访问，递归
                    Self::tarjan_scc(
                        neighbor,
                        graph,
                        index,
                        visited,
                        low_link,
                        stack,
                        on_stack,
                        cycles,
                    )?;
                    
                    // 更新当前节点的低链接值
                    let neighbor_low = *low_link.get(neighbor).unwrap_or(&usize::MAX);
                    let current_low = *low_link.get(node).unwrap_or(&usize::MAX);
                    if neighbor_low < current_low {
                        low_link.insert(node.clone(), neighbor_low);
                    }
                } else if on_stack.contains(neighbor) {
                    // 邻居在栈上，更新低链接值
                    let neighbor_index = *visited.get(neighbor).unwrap_or(&usize::MAX);
                    let current_low = *low_link.get(node).unwrap_or(&usize::MAX);
                    if neighbor_index < current_low {
                        low_link.insert(node.clone(), neighbor_index);
                    }
                }
            }
        }
        
        // 检查是否找到SCC
        if let Some(&node_low) = low_link.get(node) {
            if let Some(&node_index) = visited.get(node) {
                if node_low == node_index {
                    // 找到一个SCC（可能是环）
                    let mut cycle = Vec::new();
                    
                    while let Some(w) = stack.pop() {
                        on_stack.remove(&w);
                        cycle.push(w.clone());
                        
                        if &w == node {
                            break;
                        }
                    }
                    
                    // 如果有多个节点，那么这是一个环
                    if cycle.len() > 1 {
                        cycles.push(cycle);
                    }
                }
            }
        }
        
        Ok(())
    }

    /// 选择要中止的事务（死锁受害者）
    fn choose_victim(
        cycle: &[TransactionId],
        wait_info: &RwLock<HashMap<(TransactionId, ResourceId), WaitInfo>>,
        config: &DeadlockDetectorConfig,
    ) -> Result<Option<TransactionId>> {
        let wait_info_read = wait_info.read().map_err(|e| {
            Error::LockError(format!("等待信息读锁错误: {}", e))
        })?;
        
        // 收集所有事务的信息
        let mut txn_info = HashMap::new();
        
        for txn_id in cycle {
            for ((wait_txn, _), info) in wait_info_read.iter() {
                if wait_txn == txn_id {
                    txn_info.insert(txn_id.clone(), (
                        info.txn_start_time,
                        info.priority,
                        info.resource_count,
                    ));
                    break;
                }
            }
        }
        
        if txn_info.is_empty() {
            return Ok(None);
        }
        
        // 根据策略选择受害者
        match config.resolution_strategy {
            DeadlockResolutionStrategy::AbortYoungest => {
                // 中止最年轻的事务（开始时间最晚的）
                let victim = txn_info.iter()
                    .max_by_key(|(_, (start_time, _, _))| start_time)
                    .map(|(txn_id, _)| txn_id.clone());
                
                Ok(victim)
            },
            DeadlockResolutionStrategy::AbortOldest => {
                // 中止最老的事务（开始时间最早的）
                let victim = txn_info.iter()
                    .min_by_key(|(_, (start_time, _, _))| start_time)
                    .map(|(txn_id, _)| txn_id.clone());
                
                Ok(victim)
            },
            DeadlockResolutionStrategy::AbortLeastResources => {
                // 中止涉及资源最少的事务
                let victim = txn_info.iter()
                    .min_by_key(|(_, (_, _, resource_count))| resource_count)
                    .map(|(txn_id, _)| txn_id.clone());
                
                Ok(victim)
            },
            DeadlockResolutionStrategy::AbortLowestPriority => {
                // 中止优先级最低的事务
                let victim = txn_info.iter()
                    .min_by_key(|(_, (_, priority, _))| priority)
                    .map(|(txn_id, _)| txn_id.clone());
                
                Ok(victim)
            },
        }
    }

    /// 清理过期等待
    fn cleanup_expired_waits(
        wait_for_graph: &RwLock<HashMap<TransactionId, HashSet<TransactionId>>>,
        wait_info: &RwLock<HashMap<(TransactionId, ResourceId), WaitInfo>>,
        resource_holders: &RwLock<HashMap<ResourceId, TransactionId>>,
        config: &DeadlockDetectorConfig,
    ) -> Result<usize> {
        let max_wait_time = Duration::from_millis(config.max_wait_time_ms);
        let now = Instant::now();
        let mut expired_waits = Vec::new();
        
        // 找出所有过期的等待
        {
            let wait_info_read = wait_info.read().map_err(|e| {
                Error::LockError(format!("等待信息读锁错误: {}", e))
            })?;
            
            for ((txn_id, resource_id), info) in wait_info_read.iter() {
                if now.duration_since(info.wait_start) > max_wait_time {
                    expired_waits.push((txn_id.clone(), resource_id.clone()));
                }
            }
        }
        
        // 清理过期等待
        let mut count = 0;
        for (txn_id, resource_id) in expired_waits {
            // 移除等待信息
            {
                let mut wait_info_write = wait_info.write().map_err(|e| {
                    Error::LockError(format!("等待信息写锁错误: {}", e))
                })?;
                
                wait_info_write.remove(&(txn_id.clone(), resource_id.clone()));
            }
            
            // 更新等待图
            {
                let mut wait_for_graph_write = wait_for_graph.write().map_err(|e| {
                    Error::LockError(format!("等待图写锁错误: {}", e))
                })?;
                
                let resource_holders_read = resource_holders.read().map_err(|e| {
                    Error::LockError(format!("资源持有者读锁错误: {}", e))
                })?;
                
                if let Some(holder) = resource_holders_read.get(&resource_id) {
                    if let Some(waiters) = wait_for_graph_write.get_mut(&txn_id) {
                        waiters.remove(holder);
                        
                        if waiters.is_empty() {
                            wait_for_graph_write.remove(&txn_id);
                        }
                    }
                }
            }
            
            count += 1;
            warn!("Expired wait detected: transaction {} waiting for resource {:?}", txn_id, resource_id);
        }
        
        Ok(count)
    }

    /// 清理等待图
    fn cleanup_wait_graph(
        wait_for_graph: &RwLock<HashMap<TransactionId, HashSet<TransactionId>>>,
        wait_info: &RwLock<HashMap<(TransactionId, ResourceId), WaitInfo>>,
        resource_holders: &RwLock<HashMap<ResourceId, TransactionId>>,
    ) -> Result<usize> {
        // 获取当前所有有效的等待
        let valid_waits = {
            let wait_info_read = wait_info.read().map_err(|e| {
                Error::LockError(format!("等待信息读锁错误: {}", e))
            })?;
            
            wait_info_read.keys()
                .map(|(txn_id, _)| txn_id.clone())
                .collect::<HashSet<_>>()
        };
        
        // 清理无效的等待图节点
        let mut count = 0;
        {
            let mut wait_for_graph_write = wait_for_graph.write().map_err(|e| {
                Error::LockError(format!("等待图写锁错误: {}", e))
            })?;
            
            let invalid_nodes: Vec<TransactionId> = wait_for_graph_write.keys()
                .filter(|txn_id| !valid_waits.contains(*txn_id))
                .cloned()
                .collect();
            
            for txn_id in invalid_nodes {
                wait_for_graph_write.remove(&txn_id);
                count += 1;
            }
            
            // 清理无效的边
            for (_, waiters) in wait_for_graph_write.iter_mut() {
                let before_count = waiters.len();
                
                waiters.retain(|txn_id| valid_waits.contains(txn_id));
                
                count += before_count - waiters.len();
            }
        }
        
        debug!("Cleaned up {} invalid wait graph entries", count);
        Ok(count)
    }

    /// 获取当前等待图
    pub fn get_wait_graph(&self) -> Result<HashMap<TransactionId, HashSet<TransactionId>>> {
            let wait_for_graph = self.wait_for_graph.read().map_err(|e| {
                Error::LockError(format!("等待图读锁错误: {}", e))
            })?;
        
        Ok(wait_for_graph.clone())
    }

    /// 获取等待超时的事务
    pub fn get_long_waiting_transactions(&self, threshold_ms: u64) -> Result<Vec<(TransactionId, ResourceId, Duration)>> {
            let wait_info = self.wait_info.read().map_err(|e| {
                Error::LockError(format!("等待信息读锁错误: {}", e))
            })?;
        
        let now = Instant::now();
        let threshold = Duration::from_millis(threshold_ms);
        
        let long_waiting = wait_info.iter()
            .filter_map(|((txn_id, resource_id), info)| {
                let wait_time = now.duration_since(info.wait_start);
                if wait_time > threshold {
                    Some((txn_id.clone(), resource_id.clone(), wait_time))
                } else {
                    None
                }
            })
            .collect();
        
        Ok(long_waiting)
    }
}

#[cfg(test)]
mod tests {
    // 添加测试用例
} 