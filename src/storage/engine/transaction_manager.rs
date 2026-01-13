use std::sync::{Arc, Mutex};
use crate::Result;
use crate::Error;
use super::transaction::{TransactionManager, TransactionState};

/// 事务管理器服务
#[derive(Clone)]
pub struct TransactionManagerService {
    transaction_manager: Arc<Mutex<TransactionManager>>,
}

impl TransactionManagerService {
    /// 创建新的事务管理器服务
    pub fn new(transaction_manager: Arc<Mutex<TransactionManager>>) -> Self {
        Self { transaction_manager }
    }

    /// 开始事务
    pub fn begin_transaction(&self) -> Result<String> {
        let mut manager = self.transaction_manager.lock()
            .map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
        manager.begin_transaction()
    }

    /// 事务中写入数据
    pub fn transaction_put(&self, transaction_id: &str, key: &[u8], value: &[u8]) -> Result<()> {
        let mut manager = self.transaction_manager.lock()
            .map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
        if let Some(transaction) = manager.get_transaction_mut(transaction_id) {
            transaction.add_operation(super::transaction::TransactionOperation::Put {
                key: key.to_vec(),
                value: value.to_vec(),
            });
            Ok(())
        } else {
            Err(Error::Transaction(format!("事务不存在: {}", transaction_id)))
        }
    }

    /// 事务中删除数据
    pub fn transaction_delete(&self, transaction_id: &str, key: &[u8]) -> Result<()> {
        let mut manager = self.transaction_manager.lock()
            .map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
        if let Some(transaction) = manager.get_transaction_mut(transaction_id) {
            transaction.add_operation(super::transaction::TransactionOperation::Delete {
                key: key.to_vec(),
            });
            Ok(())
        } else {
            Err(Error::Transaction(format!("事务不存在: {}", transaction_id)))
        }
    }

    /// 提交事务
    pub fn commit_transaction(&self, transaction_id: &str) -> Result<()> {
        let mut manager = self.transaction_manager.lock()
            .map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
        manager.commit_transaction(transaction_id)
    }

    /// 回滚事务
    pub fn rollback_transaction(&self, transaction_id: &str) -> Result<()> {
        let mut manager = self.transaction_manager.lock()
            .map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
        manager.rollback_transaction(transaction_id)
    }

    /// 获取事务状态
    pub fn get_transaction_state(&self, transaction_id: &str) -> Result<TransactionState> {
        let manager = self.transaction_manager.lock()
            .map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
        if let Some(transaction) = manager.get_transaction(transaction_id) {
            Ok(transaction.state)
        } else {
            Err(Error::Transaction(format!("事务不存在: {}", transaction_id)))
        }
    }

    /// 清理过期事务
    pub fn cleanup_transactions(&self) -> Result<usize> {
        let mut manager = self.transaction_manager.lock()
            .map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
        Ok(manager.cleanup_transactions())
    }

    /// 获取活跃事务数量
    pub fn get_active_transaction_count(&self) -> Result<usize> {
        let manager = self.transaction_manager.lock()
            .map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
        Ok(manager.get_active_transaction_count())
    }
} 