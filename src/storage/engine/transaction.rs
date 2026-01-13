use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use uuid::Uuid;
use log::{debug, info, warn, error};
use crate::Result;
use crate::Error;

/// 事务状态枚举
/// 
/// 定义事务的生命周期状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TransactionState {
    /// 活跃状态 - 事务正在进行中
    Active,
    /// 已提交状态 - 事务已成功提交
    Committed,
    /// 已中断状态 - 事务被中断或回滚
    Aborted,
}

impl Default for TransactionState {
    fn default() -> Self {
        TransactionState::Active
    }
}

/// 事务操作类型
/// 
/// 定义事务中可以执行的操作类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionOperation {
    /// 插入或更新操作
    Put { 
        /// 键
        key: Vec<u8>, 
        /// 值
        value: Vec<u8> 
    },
    /// 删除操作
    Delete { 
        /// 要删除的键
        key: Vec<u8> 
    },
}

impl TransactionOperation {
    /// 创建Put操作
    pub fn put(key: Vec<u8>, value: Vec<u8>) -> Self {
        TransactionOperation::Put { key, value }
    }
    
    /// 创建Delete操作
    pub fn delete(key: Vec<u8>) -> Self {
        TransactionOperation::Delete { key }
    }
    
    /// 获取操作的键
    pub fn get_key(&self) -> &[u8] {
        match self {
            TransactionOperation::Put { key, .. } => key,
            TransactionOperation::Delete { key } => key,
        }
    }
    
    /// 判断是否为写操作
    pub fn is_write_operation(&self) -> bool {
        matches!(self, TransactionOperation::Put { .. })
    }
    
    /// 判断是否为删除操作
    pub fn is_delete_operation(&self) -> bool {
        matches!(self, TransactionOperation::Delete { .. })
    }
}

/// 事务实现
/// 
/// 表示一个完整的事务，包含状态、操作列表和时间信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    /// 事务唯一标识符
    pub id: String,
    /// 事务当前状态
    pub state: TransactionState,
    /// 事务包含的操作列表
    pub operations: Vec<TransactionOperation>,
    /// 事务创建时间
    pub created_at: std::time::SystemTime,
    /// 事务最后修改时间
    pub last_modified: std::time::SystemTime,
    /// 事务超时时间（可选）
    pub timeout: Option<std::time::Duration>,
}

impl Transaction {
    /// 创建新的事务
    /// 
    /// # 返回值
    /// 返回一个新的活跃状态事务
    pub fn new() -> Self {
        let now = std::time::SystemTime::now();
        Self {
            id: Uuid::new_v4().to_string(),
            state: TransactionState::Active,
            operations: Vec::new(),
            created_at: now,
            last_modified: now,
            timeout: None,
        }
    }
    
    /// 创建带有超时的事务
    /// 
    /// # 参数
    /// - `timeout`: 事务超时时间
    /// 
    /// # 返回值
    /// 返回一个新的带超时的活跃状态事务
    pub fn new_with_timeout(timeout: std::time::Duration) -> Self {
        let mut transaction = Self::new();
        transaction.timeout = Some(timeout);
        transaction
    }
    
    /// 向事务添加操作
    /// 
    /// # 参数
    /// - `operation`: 要添加的事务操作
    /// 
    /// # 返回值
    /// 如果事务处于活跃状态，返回Ok()，否则返回错误
    pub fn add_operation(&mut self, operation: TransactionOperation) -> Result<()> {
        if self.state != TransactionState::Active {
            error!("事务 {} 不处于活跃状态，无法添加操作。当前状态: {:?}", self.id, self.state);
            return Err(Error::Transaction(format!(
                "事务 {} 不处于活跃状态，无法添加操作。当前状态: {:?}", 
                self.id, self.state
            )));
        }
        
        // 检查是否超时
        if let Some(timeout) = self.timeout {
            if let Ok(elapsed) = self.created_at.elapsed() {
                if elapsed > timeout {
                    self.state = TransactionState::Aborted;
                    error!("事务 {} 已超时，无法添加操作 (elapsed: {:?}, timeout: {:?})", self.id, elapsed, timeout);
                    return Err(Error::Transaction(format!(
                        "事务 {} 已超时，无法添加操作", self.id
                    )));
                }
            }
        }
        
        debug!("向事务 {} 添加操作: {:?}", self.id, operation);
        self.operations.push(operation);
        self.last_modified = std::time::SystemTime::now();
        Ok(())
    }
    
    /// 提交事务
    /// 
    /// # 返回值
    /// 如果事务可以提交，返回Ok()，否则返回错误
    pub fn commit(&mut self) -> Result<()> {
        if self.state != TransactionState::Active {
            error!("事务 {} 不处于活跃状态，无法提交。当前状态: {:?}", self.id, self.state);
            return Err(Error::Transaction(format!(
                "事务 {} 不处于活跃状态，无法提交。当前状态: {:?}", 
                self.id, self.state
            )));
        }
        
        info!("提交事务 {}，包含 {} 个操作", self.id, self.operations.len());
        self.state = TransactionState::Committed;
        self.last_modified = std::time::SystemTime::now();
        Ok(())
    }
    
    /// 中断事务
    /// 
    /// # 返回值
    /// 如果事务可以中断，返回Ok()，否则返回错误
    pub fn abort(&mut self) -> Result<()> {
        if self.state == TransactionState::Committed {
            error!("事务 {} 已提交，无法中断", self.id);
            return Err(Error::Transaction(format!(
                "事务 {} 已提交，无法中断", self.id
            )));
        }
        
        warn!("中断事务 {}，包含 {} 个操作", self.id, self.operations.len());
        self.state = TransactionState::Aborted;
        self.last_modified = std::time::SystemTime::now();
        Ok(())
    }
    
    /// 检查事务是否处于活跃状态
    pub fn is_active(&self) -> bool {
        self.state == TransactionState::Active
    }
    
    /// 检查事务是否已提交
    pub fn is_committed(&self) -> bool {
        self.state == TransactionState::Committed
    }
    
    /// 检查事务是否已中断
    pub fn is_aborted(&self) -> bool {
        self.state == TransactionState::Aborted
    }
    
    /// 检查事务是否已超时
    pub fn is_timed_out(&self) -> bool {
        if let Some(timeout) = self.timeout {
            if let Ok(elapsed) = self.created_at.elapsed() {
                return elapsed > timeout;
            }
        }
        false
    }
    
    /// 获取事务的存续时间
    pub fn get_duration(&self) -> Result<std::time::Duration> {
        self.created_at.elapsed()
            .map_err(|e| Error::Transaction(format!("获取事务存续时间失败: {}", e)))
    }
    
    /// 获取事务操作数量
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }
    
    /// 清空事务操作（仅限活跃状态）
    pub fn clear_operations(&mut self) -> Result<()> {
        if self.state != TransactionState::Active {
            return Err(Error::Transaction(format!(
                "事务 {} 不处于活跃状态，无法清空操作", self.id
            )));
        }
        
        debug!("清空事务 {} 的所有操作", self.id);
        self.operations.clear();
        self.last_modified = std::time::SystemTime::now();
        Ok(())
    }
    
    /// 获取指定键的最后操作
    pub fn get_last_operation_for_key(&self, key: &[u8]) -> Option<&TransactionOperation> {
        self.operations.iter()
            .rev()
            .find(|op| op.get_key() == key)
    }
    
    /// 获取所有涉及的键
    pub fn get_affected_keys(&self) -> Vec<Vec<u8>> {
        let mut keys = Vec::new();
        let mut seen_keys = std::collections::HashSet::new();
        
        for operation in &self.operations {
            let key = operation.get_key().to_vec();
            if seen_keys.insert(key.clone()) {
                keys.push(key);
            }
        }
        
        keys
    }
}

impl Default for Transaction {
    fn default() -> Self {
        Self::new()
    }
}

/// 事务管理器
/// 
/// 管理系统中的所有事务，提供事务的创建、查询、清理等功能
#[derive(Debug)]
pub struct TransactionManager {
    /// 活跃事务映射
    transactions: HashMap<String, Transaction>,
    /// 最大活跃事务数
    max_active_transactions: usize,
    /// 事务默认超时时间
    default_timeout: Option<std::time::Duration>,
}

impl TransactionManager {
    /// 创建新的事务管理器
    /// 
    /// # 参数
    /// - `max_active_transactions`: 最大活跃事务数
    /// 
    /// # 返回值
    /// 返回新的事务管理器实例
    pub fn new(max_active_transactions: usize) -> Self {
        Self {
            transactions: HashMap::new(),
            max_active_transactions,
            default_timeout: Some(std::time::Duration::from_secs(300)), // 默认5分钟超时
        }
    }
    
    /// 设置默认超时时间
    pub fn set_default_timeout(&mut self, timeout: Option<std::time::Duration>) {
        self.default_timeout = timeout;
    }
    
    /// 创建新的事务
    /// 
    /// # 返回值
    /// 返回事务ID
    pub fn begin_transaction(&mut self) -> Result<String> {
        // 检查活跃事务数量限制
        let active_count = self.transactions.values()
            .filter(|t| t.is_active())
            .count();
            
        if active_count >= self.max_active_transactions {
            return Err(Error::Transaction(format!(
                "达到最大活跃事务数量限制: {}", self.max_active_transactions
            )));
        }
        
        let transaction = if let Some(timeout) = self.default_timeout {
            Transaction::new_with_timeout(timeout)
        } else {
            Transaction::new()
        };
        
        let transaction_id = transaction.id.clone();
        info!("开始新事务: {}", transaction_id);
        
        self.transactions.insert(transaction_id.clone(), transaction);
        Ok(transaction_id)
    }
    
    /// 获取事务
    pub fn get_transaction(&self, transaction_id: &str) -> Option<&Transaction> {
        self.transactions.get(transaction_id)
    }
    
    /// 获取可变事务
    pub fn get_transaction_mut(&mut self, transaction_id: &str) -> Option<&mut Transaction> {
        self.transactions.get_mut(transaction_id)
    }
    
    /// 提交事务
    pub fn commit_transaction(&mut self, transaction_id: &str) -> Result<()> {
        if let Some(transaction) = self.transactions.get_mut(transaction_id) {
            transaction.commit()
        } else {
            Err(Error::Transaction(format!("事务不存在: {}", transaction_id)))
        }
    }
    
    /// 回滚事务
    pub fn rollback_transaction(&mut self, transaction_id: &str) -> Result<()> {
        if let Some(transaction) = self.transactions.get_mut(transaction_id) {
            transaction.abort()
        } else {
            Err(Error::Transaction(format!("事务不存在: {}", transaction_id)))
        }
    }
    
    /// 清理已完成或超时的事务
    pub fn cleanup_transactions(&mut self) -> usize {
        let mut removed_count = 0;
        let mut to_remove = Vec::new();
        
        for (id, transaction) in &mut self.transactions {
            let should_remove = match transaction.state {
                TransactionState::Committed | TransactionState::Aborted => true,
                TransactionState::Active => {
                    if transaction.is_timed_out() {
                        warn!("事务 {} 已超时，自动中断", id);
                        let _ = transaction.abort();
                        true
                    } else {
                        false
                    }
                }
            };
            
            if should_remove {
                to_remove.push(id.clone());
            }
        }
        
        for id in to_remove {
            self.transactions.remove(&id);
            removed_count += 1;
        }
        
        if removed_count > 0 {
            info!("清理了 {} 个已完成或超时的事务", removed_count);
        }
        
        removed_count
    }
    
    /// 获取活跃事务数量
    pub fn get_active_transaction_count(&self) -> usize {
        self.transactions.values()
            .filter(|t| t.is_active())
            .count()
    }
    
    /// 获取所有事务的统计信息
    pub fn get_transaction_stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        
        let mut active = 0;
        let mut committed = 0;
        let mut aborted = 0;
        
        for transaction in self.transactions.values() {
            match transaction.state {
                TransactionState::Active => active += 1,
                TransactionState::Committed => committed += 1,
                TransactionState::Aborted => aborted += 1,
            }
        }
        
        stats.insert("active".to_string(), active);
        stats.insert("committed".to_string(), committed);
        stats.insert("aborted".to_string(), aborted);
        stats.insert("total".to_string(), self.transactions.len());
        
        stats
    }
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new(1000) // 默认最大1000个活跃事务
    }
} 