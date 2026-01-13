// 事务隔离级别实现
//
// 提供不同事务隔离级别的具体实现和行为控制

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::Instant;
// removed unused debug import; add when logging is introduced here

use crate::error::{Error, Result};
use crate::storage::transaction::{IsolationLevel, TransactionId};

/// 事务可见性管理器
/// 
/// 管理不同隔离级别下的数据可见性，避免并发事务的异常情况。
pub struct TransactionVisibilityManager {
    /// 活跃事务集合
    active_transactions: Arc<RwLock<HashSet<TransactionId>>>,
    
    /// 提交的事务信息：事务ID -> (提交时间，提交版本)
    committed_transactions: Arc<RwLock<HashMap<TransactionId, (Instant, u64)>>>,
    
    /// 全局事务计数器，用于分配事务版本号
    global_txn_counter: Arc<RwLock<u64>>,
    
    /// 事务读取的数据版本: (事务ID, 键) -> 版本号
    read_versions: Arc<RwLock<HashMap<(TransactionId, Vec<u8>), u64>>>,
    
    /// 事务写入的数据版本: (事务ID, 键) -> 版本号
    write_versions: Arc<RwLock<HashMap<(TransactionId, Vec<u8>), u64>>>,
    
    /// 事务的开始时间戳
    transaction_start_times: Arc<RwLock<HashMap<TransactionId, Instant>>>,
}

impl TransactionVisibilityManager {
    /// 创建新的事务可见性管理器
    pub fn new() -> Self {
        Self {
            active_transactions: Arc::new(RwLock::new(HashSet::new())),
            committed_transactions: Arc::new(RwLock::new(HashMap::new())),
            global_txn_counter: Arc::new(RwLock::new(1)), // 从1开始
            read_versions: Arc::new(RwLock::new(HashMap::new())),
            write_versions: Arc::new(RwLock::new(HashMap::new())),
            transaction_start_times: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// 开始新事务
    pub fn begin_transaction(&self, txn_id: &TransactionId) -> Result<u64> {
        // 获取全局事务版本
        let version = {
            let mut counter = self.global_txn_counter.write().map_err(|e| {
                Error::LockError(format!("全局事务计数器锁错误: {}", e))
            })?;
            
            let version = *counter;
            *counter += 1;
            version
        };
        
        // 记录事务开始时间
        {
            let mut start_times = self.transaction_start_times.write().map_err(|e| {
                Error::LockError(format!("事务开始时间锁错误: {}", e))
            })?;
            
            start_times.insert(txn_id.clone(), Instant::now());
        }
        
        // 添加到活跃事务集合
        {
            let mut active_txns = self.active_transactions.write().map_err(|e| {
                Error::LockError(format!("活跃事务锁错误: {}", e))
            })?;
            
            active_txns.insert(txn_id.clone());
        }
        
        Ok(version)
    }

    /// 提交事务
    pub fn commit_transaction(&self, txn_id: &TransactionId, txn_version: u64) -> Result<()> {
        // 更新提交事务信息
        {
            let mut committed_txns = self.committed_transactions.write().map_err(|e| {
                Error::lock(format!("Failed to acquire committed transactions lock: {}", e))
            })?;
            
            committed_txns.insert(txn_id.clone(), (Instant::now(), txn_version));
        }
        
        // 从活跃事务集合中移除
        {
            let mut active_txns = self.active_transactions.write().map_err(|e| {
                Error::lock(format!("Failed to acquire active transactions lock: {}", e))
            })?;
            
            active_txns.remove(txn_id);
        }
        
        Ok(())
    }

    /// 回滚事务
    pub fn rollback_transaction(&self, txn_id: &TransactionId) -> Result<()> {
        // 从活跃事务集合中移除
        {
            let mut active_txns = self.active_transactions.write().map_err(|e| {
                Error::lock(format!("Failed to acquire active transactions lock: {}", e))
            })?;
            
            active_txns.remove(txn_id);
        }
        
        // 清理事务相关信息
        {
            let mut read_versions = self.read_versions.write().map_err(|e| {
                Error::lock(format!("Failed to acquire read versions lock: {}", e))
            })?;
            
            read_versions.retain(|(t, _), _| t != txn_id);
        }
        
        {
            let mut write_versions = self.write_versions.write().map_err(|e| {
                Error::lock(format!("Failed to acquire write versions lock: {}", e))
            })?;
            
            write_versions.retain(|(t, _), _| t != txn_id);
        }
        
        {
            let mut start_times = self.transaction_start_times.write().map_err(|e| {
                Error::lock(format!("Failed to acquire transaction start times lock: {}", e))
            })?;
            
            start_times.remove(txn_id);
        }
        
        Ok(())
    }

    /// 检查数据是否可见（用于读取操作）
    pub fn is_visible(&self, txn_id: &TransactionId, key: &[u8], version: u64, isolation_level: IsolationLevel) -> Result<bool> {
        match isolation_level {
            IsolationLevel::ReadUncommitted => {
                // 读未提交：可以看到所有未提交的数据
                Ok(true)
            },
            IsolationLevel::ReadCommitted => {
                // 读已提交：只能看到已提交的数据
                if let Some((_, commit_version)) = self.get_commit_info(key, version)? {
                    Ok(commit_version <= version)
                } else {
                    // 未提交数据对读已提交事务不可见
                    let active_txns = self.active_transactions.read().map_err(|e| {
                        Error::lock(format!("Failed to acquire active transactions lock: {}", e))
                    })?;
                    
                    // 如果是自己写的未提交数据，仍然可见
                    let is_own_write = self.is_own_write(txn_id, key)?;
                    
                    Ok(is_own_write)
                }
            },
            IsolationLevel::RepeatableRead => {
                // 可重复读：只能看到事务开始前已提交的数据，或自己写的数据
                let txn_start_time = self.get_transaction_start_time(txn_id)?;
                
                if let Some((commit_time, commit_version)) = self.get_commit_info(key, version)? {
                    // 如果数据在事务开始前已提交，则可见
                    Ok(commit_time <= txn_start_time && commit_version <= version)
                } else {
                    // 如果是自己写的未提交数据，仍然可见
                    let is_own_write = self.is_own_write(txn_id, key)?;
                    
                    Ok(is_own_write)
                }
            },
            IsolationLevel::Serializable => {
                // 串行化：最严格的隔离级别，防止所有并发问题
                // 实现方式通常涉及到锁或MVCC
                
                // 检查是否有其他活跃事务修改了同一数据
                let has_write_conflict = self.has_write_conflict(txn_id, key)?;
                
                if has_write_conflict {
                    return Err(Error::Concurrency(
                        format!("Serializable isolation conflict: another transaction is modifying the same data")
                    ));
                }
                
                // 与可重复读相同的可见性规则
                let txn_start_time = self.get_transaction_start_time(txn_id)?;
                
                if let Some((commit_time, commit_version)) = self.get_commit_info(key, version)? {
                    // 如果数据在事务开始前已提交，则可见
                    Ok(commit_time <= txn_start_time && commit_version <= version)
                } else {
                    // 如果是自己写的未提交数据，仍然可见
                    let is_own_write = self.is_own_write(txn_id, key)?;
                    
                    Ok(is_own_write)
                }
            },
        }
    }

    /// 检查是否可以写入数据（防止脏写）
    pub fn can_write(&self, txn_id: &TransactionId, key: &[u8], isolation_level: IsolationLevel) -> Result<bool> {
        match isolation_level {
            IsolationLevel::ReadUncommitted | IsolationLevel::ReadCommitted => {
                // 检查是否有其他事务持有写锁
                self.check_write_lock(key, txn_id)
            },
            IsolationLevel::RepeatableRead => {
                // 检查是否有冲突的写操作
                if self.has_write_conflict(txn_id, key)? {
                    return Ok(false);
                }
                
                // 检查写锁
                self.check_write_lock(key, txn_id)
            },
            IsolationLevel::Serializable => {
                // 检查是否有任何冲突（读或写）
                if self.has_any_conflict(txn_id, key)? {
                    return Ok(false);
                }
                
                // 检查写锁
                self.check_write_lock(key, txn_id)
            },
        }
    }

    /// 记录读取版本
    pub fn record_read(&self, txn_id: &TransactionId, key: &[u8], version: u64) -> Result<()> {
        let mut read_versions = self.read_versions.write().map_err(|e| {
            Error::Storage(format!("Failed to acquire read versions lock: {}", e))
        })?;
        
        read_versions.insert((txn_id.clone(), key.to_vec()), version);
        
        Ok(())
    }

    /// 记录写入版本
    pub fn record_write(&self, txn_id: &TransactionId, key: &[u8], version: u64) -> Result<()> {
        let mut write_versions = self.write_versions.write().map_err(|e| {
            Error::Storage(format!("Failed to acquire write versions lock: {}", e))
        })?;
        
        write_versions.insert((txn_id.clone(), key.to_vec()), version);
        
        Ok(())
    }

    /// 获取数据的提交信息
    fn get_commit_info(&self, key: &[u8], version: u64) -> Result<Option<(Instant, u64)>> {
        // 从已提交事务记录中查找该键的提交信息
        // 遍历所有已提交的事务，查找是否有写入该键的事务
        let committed_txns = self.committed_transactions.read().map_err(|e| {
            Error::lock(format!("Failed to acquire committed transactions lock: {}", e))
        })?;
        
        // 查找写入版本记录中该键对应的已提交事务
        let write_versions = self.write_versions.read().map_err(|e| {
            Error::lock(format!("Failed to acquire write versions lock: {}", e))
        })?;
        
        // 查找所有写入该键的事务，然后检查哪些已提交
        for ((txn_id, written_key), write_version) in write_versions.iter() {
            if written_key == key && write_version <= &version {
                // 检查该事务是否已提交
                if let Some((commit_time, commit_version)) = committed_txns.get(txn_id) {
                    // 如果提交版本小于等于请求的版本，返回提交信息
                    if commit_version <= &version {
                        return Ok(Some((*commit_time, *commit_version)));
                    }
                }
            }
        }
        
        // 如果未找到已提交的写入，返回 None
        Ok(None)
    }

    /// 检查是否有写冲突（防止脏读、不可重复读和幻读）
    fn has_write_conflict(&self, txn_id: &TransactionId, key: &[u8]) -> Result<bool> {
        let write_versions = self.write_versions.read().map_err(|e| {
            Error::lock(format!("Failed to acquire write versions lock: {}", e))
        })?;
        
        // 检查是否有其他活跃事务写入了相同的键
        for ((other_txn_id, other_key), _) in write_versions.iter() {
            if other_txn_id != txn_id && other_key == key {
                let active_txns = self.active_transactions.read().map_err(|e| {
                    Error::lock(format!("Failed to acquire active transactions lock: {}", e))
                })?;
                
                if active_txns.contains(other_txn_id) {
                    return Ok(true);
                }
            }
        }
        
        Ok(false)
    }

    /// 检查是否有任何冲突（读或写）
    fn has_any_conflict(&self, txn_id: &TransactionId, key: &[u8]) -> Result<bool> {
        // 检查写冲突
        if self.has_write_conflict(txn_id, key)? {
            return Ok(true);
        }
        
        // 检查读冲突（防止幻读）
        let read_versions = self.read_versions.read().map_err(|e| {
            Error::Storage(format!("Failed to acquire read versions lock: {}", e))
        })?;
        
        // 检查是否有其他活跃事务读取了相同的键
        for ((other_txn_id, other_key), _) in read_versions.iter() {
            if other_txn_id != txn_id && other_key == key {
                let active_txns = self.active_transactions.read().map_err(|e| {
                    Error::lock(format!("Failed to acquire active transactions lock: {}", e))
                })?;
                
                if active_txns.contains(other_txn_id) {
                    return Ok(true);
                }
            }
        }
        
        Ok(false)
    }

    /// 检查是否是自己的写操作
    fn is_own_write(&self, txn_id: &TransactionId, key: &[u8]) -> Result<bool> {
        let write_versions = self.write_versions.read().map_err(|e| {
            Error::lock(format!("Failed to acquire write versions lock: {}", e))
        })?;
        
        Ok(write_versions.contains_key(&(txn_id.clone(), key.to_vec())))
    }

    /// 获取事务开始时间
    fn get_transaction_start_time(&self, txn_id: &TransactionId) -> Result<Instant> {
        let start_times = self.transaction_start_times.read().map_err(|e| {
            Error::lock(format!("Failed to acquire transaction start times lock: {}", e))
        })?;
        
        start_times.get(txn_id)
            .cloned()
            .ok_or_else(|| Error::invalid_state(format!("Transaction {} not found", txn_id)))
    }

    /// 检查写锁
    fn check_write_lock(&self, key: &[u8], txn_id: &TransactionId) -> Result<bool> {
        // 检查是否有其他活跃事务正在写入该键
        let write_versions = self.write_versions.read().map_err(|e| {
            Error::lock(format!("Failed to acquire write versions lock: {}", e))
        })?;
        
        // 查找是否有其他事务写入该键
        for ((other_txn_id, written_key), _) in write_versions.iter() {
            if written_key == key && other_txn_id != txn_id {
                // 检查该事务是否仍然活跃
                let active_txns = self.active_transactions.read().map_err(|e| {
                    Error::lock(format!("Failed to acquire active transactions lock: {}", e))
                })?;
                
                if active_txns.contains(other_txn_id) {
                    // 有其他活跃事务正在写入该键，存在写锁冲突
                    return Ok(false);
                }
            }
        }
        
        // 没有冲突，可以写入
        Ok(true)
    }
    
    /// 清理已完成事务的数据
    pub fn cleanup_old_transactions(&self, max_age_secs: u64) -> Result<usize> {
        let now = Instant::now();
        let mut count = 0;
        
        // 清理已提交事务记录
        {
            let mut committed_txns = self.committed_transactions.write().map_err(|e| {
                Error::LockError(format!("已提交事务锁错误: {}", e))
            })?;
            
            let old_txns: Vec<TransactionId> = committed_txns.iter()
                .filter(|(_, (commit_time, _))| {
                    now.duration_since(*commit_time).as_secs() > max_age_secs
                })
                .map(|(txn_id, _)| txn_id.clone())
                .collect();
                
            for txn_id in old_txns {
                committed_txns.remove(&txn_id);
                count += 1;
            }
        }
        
        // 清理读写版本记录
        {
            let mut read_versions = self.read_versions.write().map_err(|e| {
                Error::lock(format!("Failed to acquire read versions lock: {}", e))
            })?;
            
            let active_txns = self.active_transactions.read().map_err(|e| {
                Error::lock(format!("Failed to acquire active transactions lock: {}", e))
            })?;
            
            // 仅保留活跃事务的读记录
            let before_count = read_versions.len();
            read_versions.retain(|(txn_id, _), _| active_txns.contains(txn_id));
            count += before_count - read_versions.len();
        }
        
        {
            let mut write_versions = self.write_versions.write().map_err(|e| {
                Error::lock(format!("Failed to acquire write versions lock: {}", e))
            })?;
            
            let active_txns = self.active_transactions.read().map_err(|e| {
                Error::lock(format!("Failed to acquire active transactions lock: {}", e))
            })?;
            
            // 仅保留活跃事务的写记录
            let before_count = write_versions.len();
            write_versions.retain(|(txn_id, _), _| active_txns.contains(txn_id));
            count += before_count - write_versions.len();
        }
        
        Ok(count)
    }
}

/// 事务隔离控制器，根据隔离级别控制事务行为
pub struct IsolationController {
    /// 可见性管理器
    visibility_manager: Arc<TransactionVisibilityManager>,
}

impl IsolationController {
    /// 创建新的隔离控制器
    pub fn new(visibility_manager: Arc<TransactionVisibilityManager>) -> Self {
        Self {
            visibility_manager,
        }
    }

    /// 验证读操作的合法性
    pub fn validate_read(&self, txn_id: &TransactionId, key: &[u8], version: u64, isolation_level: IsolationLevel) -> Result<bool> {
        self.visibility_manager.is_visible(txn_id, key, version, isolation_level)
    }

    /// 验证写操作的合法性
    pub fn validate_write(&self, txn_id: &TransactionId, key: &[u8], isolation_level: IsolationLevel) -> Result<bool> {
        self.visibility_manager.can_write(txn_id, key, isolation_level)
    }

    /// 记录读操作
    pub fn record_read(&self, txn_id: &TransactionId, key: &[u8], version: u64) -> Result<()> {
        self.visibility_manager.record_read(txn_id, key, version)
    }

    /// 记录写操作
    pub fn record_write(&self, txn_id: &TransactionId, key: &[u8], version: u64) -> Result<()> {
        self.visibility_manager.record_write(txn_id, key, version)
    }

    /// 根据隔离级别获取操作模式描述
    pub fn get_isolation_behavior(isolation_level: IsolationLevel) -> IsolationBehavior {
        match isolation_level {
            IsolationLevel::ReadUncommitted => IsolationBehavior {
                prevents_dirty_read: false,
                prevents_non_repeatable_read: false,
                prevents_phantom_read: false,
                description: "允许脏读、不可重复读和幻读。性能最好但隔离性最弱。".to_string(),
            },
            IsolationLevel::ReadCommitted => IsolationBehavior {
                prevents_dirty_read: true,
                prevents_non_repeatable_read: false,
                prevents_phantom_read: false,
                description: "防止脏读，但允许不可重复读和幻读。平衡性能和隔离性。".to_string(),
            },
            IsolationLevel::RepeatableRead => IsolationBehavior {
                prevents_dirty_read: true,
                prevents_non_repeatable_read: true,
                prevents_phantom_read: false,
                description: "防止脏读和不可重复读，但允许幻读。较强的隔离性。".to_string(),
            },
            IsolationLevel::Serializable => IsolationBehavior {
                prevents_dirty_read: true,
                prevents_non_repeatable_read: true,
                prevents_phantom_read: true,
                description: "防止所有并发问题，包括脏读、不可重复读和幻读。最强的隔离性但性能最差。".to_string(),
            },
        }
    }
}

/// 隔离级别行为描述
#[derive(Debug, Clone)]
pub struct IsolationBehavior {
    /// 是否防止脏读
    pub prevents_dirty_read: bool,
    /// 是否防止不可重复读
    pub prevents_non_repeatable_read: bool,
    /// 是否防止幻读
    pub prevents_phantom_read: bool,
    /// 行为描述
    pub description: String,
}

#[cfg(test)]
mod tests {
    // 添加测试用例
} 