// 事务实现模块
//
// 提供数据库事务处理功能，支持原子性操作和错误恢复

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use std::fs;
use std::io::Write;
// use chrono::{DateTime, Utc};
use log::{info, warn, error, debug};
use uuid::Uuid;
use serde::{Serialize, Deserialize};

use crate::error::{Error, Result};
use crate::storage::engine::StorageEngine;
use super::{IsolationLevel, TransactionId};
use crate::core::interfaces::StorageInterface as CoreStorageTransaction;

/// 事务状态枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TransactionState {
    /// 活跃状态，可执行操作
    Active,
    
    /// 已提交状态
    Committed,
    
    /// 已回滚状态
    RolledBack,
    
    /// 超时状态
    TimedOut,
    
    /// 执行失败状态
    Failed,
    
    /// 事务已过期
    Expired,
    
    /// 事务已终止
    Aborted,
}

/// 冲突处理策略
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConflictStrategy {
    /// 中止当前事务
    Abort,
    /// 重试当前事务
    Retry,
    /// 等待冲突解决
    Wait,
}

/// 事务选项配置
#[derive(Debug, Clone)]
pub struct TransactionOptions {
    /// 超时时间
    pub timeout: Option<Duration>,
    /// 隔离级别
    pub isolation_level: IsolationLevel,
    /// 是否为只读事务
    pub read_only: bool,
    /// 最大重试次数
    pub max_retries: usize,
    /// 冲突处理策略
    pub conflict_strategy: ConflictStrategy,
}

impl Default for TransactionOptions {
    fn default() -> Self {
        Self {
            timeout: Some(Duration::from_secs(30)),
            isolation_level: IsolationLevel::ReadCommitted,
            read_only: false,
            max_retries: 3,
            conflict_strategy: ConflictStrategy::Abort,
        }
    }
}

/// 事务操作类型
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Operation {
    /// 写入操作
    Put {
        key: Vec<u8>,
        value: Vec<u8>,
    },
    
    /// 删除操作
    Delete {
        key: Vec<u8>,
    },
    
    /// 多项写入操作
    BatchPut {
        /// 键值对列表
        items: Vec<(Vec<u8>, Vec<u8>)>,
    },
    
    /// 多项删除操作
    BatchDelete {
        /// 键列表
        keys: Vec<Vec<u8>>,
    },
}

/// 事务结构体
pub struct Transaction {
    /// 事务ID
    id: TransactionId,
    
    /// 事务开始时间
    start_time: Instant,
    
    /// 事务隔离级别
    isolation_level: IsolationLevel,
    
    /// 事务状态
    state: TransactionState,
    
    /// 事务操作列表
    operations: Vec<Operation>,
    
    /// 锁定的键
    locked_keys: HashSet<Vec<u8>>,
    
    /// 存储引擎引用（当不使用上层事务对象时）
    db: Arc<dyn StorageEngine>,
    /// 上层存储事务对象（来自统一接口的事务），优先于直接引擎操作
    inner_tx: Option<Box<dyn CoreStorageTransaction>>, 
    
    /// 事务选项
    options: TransactionOptions,
    
    /// 重试次数
    retry_count: usize,
    
    /// 已读取数据的快照
    snapshot: HashMap<Vec<u8>, Option<Vec<u8>>>,
}

impl Transaction {
    /// 创建新事务
    pub fn new(id: TransactionId, db: Arc<dyn StorageEngine>, isolation_level: IsolationLevel) -> Self {
        let options = TransactionOptions {
            isolation_level,
            ..Default::default()
        };
        
        Self {
            id,
            start_time: Instant::now(),
            isolation_level,
            state: TransactionState::Active,
            operations: Vec::new(),
            locked_keys: HashSet::new(),
            db,
            inner_tx: None,
            options,
            retry_count: 0,
            snapshot: HashMap::new(),
        }
    }
    
    /// 创建带选项的新事务
    pub fn with_options(id: TransactionId, db: Arc<dyn StorageEngine>, options: TransactionOptions) -> Self {
        Self {
            id,
            start_time: Instant::now(),
            isolation_level: options.isolation_level,
            state: TransactionState::Active,
            operations: Vec::new(),
            locked_keys: HashSet::new(),
            db,
            inner_tx: None,
            options,
            retry_count: 0,
            snapshot: HashMap::new(),
        }
    }

    /// 设置事务超时时间（秒）
    pub fn set_timeout(&mut self, timeout_seconds: u64) {
        self.options.timeout = Some(Duration::from_secs(timeout_seconds));
    }

    /// 使用统一存储接口的事务对象创建事务
    pub fn from_storage_transaction(
        storage_tx: Box<dyn CoreStorageTransaction>,
        isolation_level: IsolationLevel,
    ) -> Self {
        // 占位的引擎引用不会被直接使用（inner_tx 优先），这里使用一个空实现的占位引擎
        struct NoopEngine;
        impl StorageEngine for NoopEngine {
            fn get(&self, _key: &[u8]) -> crate::Result<Option<Vec<u8>>> { Ok(None) }
            fn put(&self, _key: &[u8], _value: &[u8]) -> crate::Result<()> { Ok(()) }
            fn delete(&self, _key: &[u8]) -> crate::Result<()> { Ok(()) }
            fn scan_prefix(&self, _prefix: &[u8]) -> Box<dyn Iterator<Item = crate::Result<(Vec<u8>, Vec<u8>)>> + '_> { Box::new(std::iter::empty()) }
            fn set_options(&mut self, _options: &crate::storage::engine::implementation::StorageOptions) -> crate::Result<()> { Ok(()) }
            fn dataset_exists(&self, _dataset_id: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<bool>> + Send + '_>> { Box::pin(async { Ok(false) }) }
            fn get_dataset_data(&self, _dataset_id: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<Vec<u8>>> + Send + '_>> { Box::pin(async { Ok(Vec::new()) }) }
            fn get_dataset_metadata(&self, _dataset_id: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<serde_json::Value>> + Send + '_>> { Box::pin(async { Ok(serde_json::Value::Null) }) }
            fn save_dataset_data(&self, _dataset_id: &str, _data: &[u8]) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<()>> + Send + '_>> { Box::pin(async { Ok(()) }) }
            fn save_dataset_metadata(&self, _dataset_id: &str, _metadata: &serde_json::Value) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<()>> + Send + '_>> { Box::pin(async { Ok(()) }) }
            fn get_dataset_schema(&self, _dataset_id: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<serde_json::Value>> + Send + '_>> { Box::pin(async { Ok(serde_json::Value::Null) }) }
            fn save_dataset_schema(&self, _dataset_id: &str, _schema: &serde_json::Value) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<()>> + Send + '_>> { Box::pin(async { Ok(()) }) }
            fn delete_dataset_complete(&self, _dataset_id: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<()>> + Send + '_>> { Box::pin(async { Ok(()) }) }
            fn list_datasets(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<Vec<String>>> + Send + '_>> { Box::pin(async { Ok(Vec::new()) }) }
            fn store(&self, _key: &str, _data: &[u8]) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<()>> + Send + '_>> { Box::pin(async { Ok(()) }) }
            fn list_models_with_filters(&self, _filters: std::collections::HashMap<String, String>, _limit: usize, _offset: usize) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<Vec<crate::model::Model>>> + Send + '_>> { Box::pin(async { Ok(Vec::new()) }) }
            fn count_models(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<usize>> + Send + '_>> { Box::pin(async { Ok(0) }) }
            fn get_data_batch(&self, _batch_id: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<crate::data::DataBatch>> + Send + '_>> { Box::pin(async { Err(crate::Error::not_found("NoopEngine")) }) }
            fn get_batch_data(&self, _batch_id: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<crate::data::DataBatch>> + Send + '_>> { Box::pin(async { Err(crate::Error::not_found("NoopEngine")) }) }
            fn save_processed_batch(&self, _model_id: &str, _batch: &crate::data::ProcessedBatch) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<()>> + Send + '_>> { Box::pin(async { Ok(()) }) }
            fn get_training_metrics_history(&self, _model_id: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<Vec<crate::model::parameters::TrainingMetrics>>> + Send + '_>> { Box::pin(async { Ok(Vec::new()) }) }
            fn record_training_metrics(&self, _model_id: &str, _metrics: &crate::model::parameters::TrainingMetrics) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<()>> + Send + '_>> { Box::pin(async { Ok(()) }) }
            fn query_dataset(&self, _name: &str, _limit: Option<usize>, _offset: Option<usize>) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<Vec<serde_json::Value>>> + Send + '_>> { Box::pin(async { Ok(Vec::new()) }) }
            fn exists(&self, _key: &[u8]) -> crate::Result<bool> { Ok(false) }
            fn get_dataset_size(&self, _dataset_id: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<usize>> + Send + '_>> { Box::pin(async { Ok(0) }) }
            fn get_dataset_chunk(&self, _dataset_id: &str, _start: usize, _end: usize) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<Vec<u8>>> + Send + '_>> { Box::pin(async { Ok(Vec::new()) }) }
            fn close(&self) -> crate::Result<()> { Ok(()) }
            fn save_processed_data(&self, _model_id: &str, _data: &[Vec<f32>]) -> std::pin::Pin<Box<dyn std::future::Future<Output = crate::Result<()>> + Send + '_>> { Box::pin(async { Ok(()) }) }
        }
        let id = Uuid::new_v4().to_string();
        Self {
            id,
            start_time: Instant::now(),
            isolation_level,
            state: TransactionState::Active,
            operations: Vec::new(),
            locked_keys: HashSet::new(),
            db: Arc::new(NoopEngine),
            inner_tx: Some(storage_tx),
            options: TransactionOptions { isolation_level, ..Default::default() },
            retry_count: 0,
            snapshot: HashMap::new(),
        }
    }
    
    /// 获取事务ID
    pub fn id(&self) -> &TransactionId {
        &self.id
    }
    
    /// 获取事务状态
    pub fn state(&self) -> TransactionState {
        self.state
    }
    
    /// 获取事务隔离级别
    pub fn isolation_level(&self) -> IsolationLevel {
        self.isolation_level
    }
    
    /// 获取事务已运行时间
    pub fn duration(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    /// 检查事务是否活跃
    pub fn is_active(&self) -> bool {
        self.state == TransactionState::Active
    }
    
    /// 检查事务是否已超时
    pub fn is_timeout(&self) -> bool {
        if let Some(timeout) = self.options.timeout {
            self.start_time.elapsed() > timeout
        } else {
            false
        }
    }
    
    /// 写入数据
    pub fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        if !self.is_active() {
            return Err(Error::invalid_state("事务不在活跃状态"));
        }
        
        if self.options.read_only {
            return Err(Error::invalid_operation("不能在只读事务中修改数据"));
        }
        
        // 如果隔离级别高于读未提交，需要锁定键
        if matches!(self.isolation_level, IsolationLevel::ReadCommitted | IsolationLevel::RepeatableRead | IsolationLevel::Serializable) {
            self.locked_keys.insert(key.to_vec());
        }
        
        self.operations.push(Operation::Put {
            key: key.to_vec(),
            value: value.to_vec(),
        });
        
        Ok(())
    }
    
    /// 删除数据
    pub fn delete(&mut self, key: &[u8]) -> Result<()> {
        if !self.is_active() {
            return Err(Error::invalid_state("事务不在活跃状态"));
        }
        
        if self.options.read_only {
            return Err(Error::invalid_operation("不能在只读事务中修改数据"));
        }
        
        // 如果隔离级别高于读未提交，需要锁定键
        if matches!(self.isolation_level, IsolationLevel::ReadCommitted | IsolationLevel::RepeatableRead | IsolationLevel::Serializable) {
            self.locked_keys.insert(key.to_vec());
        }
        
        self.operations.push(Operation::Delete {
            key: key.to_vec(),
        });
        
        Ok(())
    }
    
    /// 批量写入
    pub fn batch_put(&mut self, items: &[(Vec<u8>, Vec<u8>)]) -> Result<()> {
        if !self.is_active() {
            return Err(Error::invalid_state("事务不在活跃状态"));
        }
        
        if self.options.read_only {
            return Err(Error::invalid_operation("不能在只读事务中修改数据"));
        }
        
        // 如果隔离级别高于读未提交，需要锁定所有键
        if matches!(self.isolation_level, IsolationLevel::ReadCommitted | IsolationLevel::RepeatableRead | IsolationLevel::Serializable) {
            for (key, _) in items {
                self.locked_keys.insert(key.clone());
            }
        }
        
        self.operations.push(Operation::BatchPut {
            items: items.to_vec(),
        });
        
        Ok(())
    }
    
    /// 批量删除
    pub fn batch_delete(&mut self, keys: &[Vec<u8>]) -> Result<()> {
        if !self.is_active() {
            return Err(Error::invalid_state("事务不在活跃状态"));
        }
        
        if self.options.read_only {
            return Err(Error::invalid_operation("不能在只读事务中修改数据"));
        }
        
        // 如果隔离级别高于读未提交，需要锁定所有键
        if matches!(self.isolation_level, IsolationLevel::ReadCommitted | IsolationLevel::RepeatableRead | IsolationLevel::Serializable) {
            for key in keys {
                self.locked_keys.insert(key.clone());
            }
        }
        
        self.operations.push(Operation::BatchDelete {
            keys: keys.to_vec(),
        });
        
        Ok(())
    }
    
    /// 读取数据
    pub fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        if !self.is_active() {
            return Err(Error::invalid_state("事务不在活跃状态"));
        }
        
        if self.is_timeout() {
            self.state = TransactionState::TimedOut;
            return Err(Error::Transaction("事务已超时".to_string()));
        }
        
        // 根据隔离级别处理读取
        match self.isolation_level {
            // 读未提交：直接读取，不考虑其他事务的影响
            IsolationLevel::ReadUncommitted => {
                // 先检查本事务中是否有修改
                for op in self.operations.iter().rev() {
                    match op {
                        Operation::Put { key: k, value } if k == key => {
                            return Ok(Some(value.clone()));
                        }
                        Operation::Delete { key: k } if k == key => {
                            return Ok(None);
                        }
                        Operation::BatchPut { items } => {
                            for (k, v) in items {
                                if k == key {
                                    return Ok(Some(v.clone()));
                                }
                            }
                        }
                        Operation::BatchDelete { keys } => {
                            for k in keys {
                                if k == key {
                                    return Ok(None);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                
                // 否则读取存储中的数据
                if let Some(tx) = &self.inner_tx { 
                    tx.retrieve(&String::from_utf8_lossy(key))
                        .map(|opt| opt.map(|v| v))
                } else { 
                    self.db.get(key) 
                }
            }
            
            // 读已提交：只读取已提交的数据
            IsolationLevel::ReadCommitted => {
                // 先检查本事务中是否有修改
                for op in self.operations.iter().rev() {
                    match op {
                        Operation::Put { key: k, value } if k == key => {
                            return Ok(Some(value.clone()));
                        }
                        Operation::Delete { key: k } if k == key => {
                            return Ok(None);
                        }
                        Operation::BatchPut { items } => {
                            for (k, v) in items {
                                if k == key {
                                    return Ok(Some(v.clone()));
                                }
                            }
                        }
                        Operation::BatchDelete { keys } => {
                            for k in keys {
                                if k == key {
                                    return Ok(None);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                
                // 否则读取存储中的数据
                if let Some(tx) = &self.inner_tx { 
                    tx.retrieve(&String::from_utf8_lossy(key))
                        .map(|opt| opt.map(|v| v))
                } else { 
                    self.db.get(key) 
                }
            }
            
            // 可重复读：确保在事务中多次读取相同数据得到相同结果
            IsolationLevel::RepeatableRead | IsolationLevel::Serializable => {
                // 如果隔离级别是可重复读或更高，且已经有快照，则从快照读取
                if self.snapshot.contains_key(key) {
                    return Ok(self.snapshot.get(key).cloned().unwrap_or(None));
                }
                
                // 先检查本事务中是否有修改
                for op in self.operations.iter().rev() {
                    match op {
                        Operation::Put { key: k, value } if k == key => {
                            // 记录快照
                            let result = Some(value.clone());
                            self.snapshot.insert(key.to_vec(), result.clone());
                            return Ok(result);
                        }
                        Operation::Delete { key: k } if k == key => {
                            // 记录快照
                            self.snapshot.insert(key.to_vec(), None);
                            return Ok(None);
                        }
                        Operation::BatchPut { items } => {
                            for (k, v) in items {
                                if k == key {
                                    // 记录快照
                                    let result = Some(v.clone());
                                    self.snapshot.insert(key.to_vec(), result.clone());
                                    return Ok(result);
                                }
                            }
                        }
                        Operation::BatchDelete { keys } => {
                            for k in keys {
                                if k == key {
                                    // 记录快照
                                    self.snapshot.insert(key.to_vec(), None);
                                    return Ok(None);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                
                // 从存储引擎读取数据
                let value = self.db.get(key)?;
                
                // 记录快照
                self.snapshot.insert(key.to_vec(), value.clone());
                
                Ok(value)
            }
        }
    }
    
    /// 提交事务
    pub fn commit(&mut self) -> Result<()> {
        if !self.is_active() {
            return Err(Error::invalid_state("事务不在活跃状态"));
        }
        
        if self.is_timeout() {
            self.state = TransactionState::TimedOut;
            return Err(Error::Transaction("事务已超时".to_string()));
        }
        
        // 执行所有操作
        for op in &self.operations {
            match op {
                Operation::Put { key, value } => {
                    if let Some(tx) = &mut self.inner_tx { tx.store(&String::from_utf8_lossy(key), value)?; } else { self.db.put(key, value)?; }
                }
                Operation::Delete { key } => {
                    if let Some(tx) = &mut self.inner_tx { tx.delete(&String::from_utf8_lossy(key))?; } else { self.db.delete(key)?; }
                }
                Operation::BatchPut { items } => {
                    for (key, value) in items {
                        if let Some(tx) = &mut self.inner_tx { tx.store(&String::from_utf8_lossy(key), value)?; } else { self.db.put(key, value)?; }
                    }
                }
                Operation::BatchDelete { keys } => {
                    for key in keys {
                        if let Some(tx) = &mut self.inner_tx { tx.delete(&String::from_utf8_lossy(key))?; } else { self.db.delete(key)?; }
                    }
                }
            }
        }
        
        // 更新状态
        self.state = TransactionState::Committed;
        
        // 释放所有锁
        self.locked_keys.clear();
        
        debug!("事务 {} 已提交，耗时: {:?}", self.id, self.duration());
        Ok(())
    }
    
    /// 回滚事务
    pub fn rollback(&mut self) -> Result<()> {
        if self.state == TransactionState::Active {
            // CoreStorageTransaction 没有 rollback 方法，需要手动清理
            // if let Some(tx) = &mut self.inner_tx { let _ = tx.rollback(); }
            self.state = TransactionState::RolledBack;
            self.locked_keys.clear();
            debug!("事务 {} 已回滚", self.id);
        }
        Ok(())
    }
    
    /// 检查事务是否超时
    pub fn check_timeout(&mut self, timeout: Duration) -> bool {
        if self.is_active() && self.duration() > timeout {
            self.state = TransactionState::TimedOut;
            warn!("事务 {} 已超时", self.id);
            true
        } else {
            false
        }
    }
    
    /// 获取事务统计信息
    pub fn get_stats(&self) -> TransactionStats {
        TransactionStats {
            id: self.id.clone(),
            state: self.state,
            isolation_level: self.isolation_level,
            start_time: self.start_time,
            duration: self.duration(),
            operations_count: self.operations.len(),
            locked_keys_count: self.locked_keys.len(),
            retry_count: self.retry_count,
            snapshot_size: self.snapshot.len(),
        }
    }
    
    /// 获取锁定的键列表
    pub fn get_locked_keys(&self) -> Vec<Vec<u8>> {
        self.locked_keys.iter().cloned().collect()
    }
    
    /// 检查是否与其他事务存在锁冲突
    pub fn has_lock_conflict(&self, other_keys: &HashSet<Vec<u8>>) -> bool {
        !self.locked_keys.is_disjoint(other_keys)
    }
    
    /// 强制终止事务
    pub fn abort(&mut self) {
        self.state = TransactionState::Aborted;
        self.locked_keys.clear();
        warn!("事务 {} 被强制终止", self.id);
    }
    
    /// 获取事务日志
    pub fn get_transaction_log(&self) -> TransactionLog {
        TransactionLog {
            id: self.id.clone(),
            start_time: self.start_time,
            state: self.state,
            isolation_level: self.isolation_level,
            operations: self.operations.clone(),
            locked_keys: self.locked_keys.iter().cloned().collect(),
            retry_count: self.retry_count,
            duration: self.duration(),
        }
    }
    
    /// 从日志恢复事务状态
    pub fn restore_from_log(log: &TransactionLog, db: Arc<dyn StorageEngine>) -> Self {
        let mut transaction = Self {
            id: log.id.clone(),
            start_time: log.start_time,
            isolation_level: log.isolation_level,
            state: log.state,
            operations: log.operations.clone(),
            locked_keys: log.locked_keys.iter().cloned().collect(),
            db: db.clone(),
            inner_tx: None, // 从日志恢复时，inner_tx 为 None
            options: TransactionOptions::default(),
            retry_count: log.retry_count,
            snapshot: HashMap::new(),
        };
        
        // 如果事务处于活跃状态但已超时，标记为超时
        if transaction.is_active() && transaction.duration() > Duration::from_secs(300) {
            transaction.state = TransactionState::TimedOut;
        }
        
        transaction
    }

    /// 重试事务
    pub fn retry(&mut self) -> Result<()> {
        if self.retry_count >= self.options.max_retries {
            return Err(Error::Transaction("超过最大重试次数".to_string()));
        }
        
        // 重置事务状态
        self.state = TransactionState::Active;
        self.start_time = Instant::now();
        self.retry_count += 1;
        self.operations.clear();
        self.locked_keys.clear();
        self.snapshot.clear();
        
        info!("事务 {} 开始第 {} 次重试", self.id, self.retry_count);
        Ok(())
    }
    
    /// 预提交检查
    pub fn pre_commit_check(&self) -> Result<()> {
        if !self.is_active() {
            return Err(Error::invalid_state("事务不在活跃状态"));
        }
        
        if self.is_timeout() {
            return Err(Error::Transaction("事务已超时".to_string()));
        }
        
        if self.options.read_only && !self.operations.is_empty() {
            return Err(Error::invalid_operation("只读事务不能包含写操作"));
        }
        
        // 检查资源限制
        if self.operations.len() > 10000 {
            return Err(Error::resource_exhausted("事务操作数量过多"));
        }
        
        Ok(())
    }
    
    /// 获取事务影响的键范围
    pub fn get_affected_keys(&self) -> HashSet<Vec<u8>> {
        let mut keys = HashSet::new();
        
        for op in &self.operations {
            match op {
                Operation::Put { key, .. } => {
                    keys.insert(key.clone());
                }
                Operation::Delete { key } => {
                    keys.insert(key.clone());
                }
                Operation::BatchPut { items } => {
                    for (key, _) in items {
                        keys.insert(key.clone());
                    }
                }
                Operation::BatchDelete { keys: op_keys } => {
                    for key in op_keys {
                        keys.insert(key.clone());
                    }
                }
            }
        }
        
        keys
    }
}

/// 事务统计信息
#[derive(Debug, Clone)]
pub struct TransactionStats {
    /// 事务ID
    pub id: TransactionId,
    /// 事务状态
    pub state: TransactionState,
    /// 隔离级别
    pub isolation_level: IsolationLevel,
    /// 开始时间
    pub start_time: Instant,
    /// 持续时间
    pub duration: Duration,
    /// 操作数量
    pub operations_count: usize,
    /// 锁定键数量
    pub locked_keys_count: usize,
    /// 重试次数
    pub retry_count: usize,
    /// 快照大小
    pub snapshot_size: usize,
}

/// 事务日志
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionLog {
    /// 事务ID
    pub id: TransactionId,
    /// 开始时间（序列化为时间戳）
    #[serde(skip, default = "default_instant")]
    pub start_time: Instant,
    /// 事务状态（序列化为字符串）
    #[serde(skip, default = "default_transaction_state")]
    pub state: TransactionState,
    /// 隔离级别
    pub isolation_level: IsolationLevel,
    /// 操作列表
    pub operations: Vec<Operation>,
    /// 锁定的键
    pub locked_keys: Vec<Vec<u8>>,
    /// 重试次数
    pub retry_count: usize,
    /// 持续时间（序列化为秒数）
    #[serde(skip)]
    pub duration: Duration,
}

// 默认值辅助函数
fn default_instant() -> Instant {
    Instant::now()
}

fn default_transaction_state() -> TransactionState {
    TransactionState::Active
}

/// 事务包装器，提供线程安全的事务访问
pub struct TransactionWrapper {
    inner: Mutex<Transaction>,
}

impl TransactionWrapper {
    /// 创建新的事务包装器
    pub fn new(transaction: Transaction) -> Self {
        Self {
            inner: Mutex::new(transaction),
        }
    }
    
    /// 获取事务锁
    pub fn lock(&self) -> Result<MutexGuard<'_, Transaction>> {
        self.inner.lock().map_err(|_| Error::lock("无法获取事务锁"))
    }
}

/// 死锁检测器
#[derive(Debug)]
pub struct DeadlockDetector {
    /// 等待图：事务ID -> 等待的事务ID列表
    wait_graph: Mutex<HashMap<TransactionId, HashSet<TransactionId>>>,
}

impl DeadlockDetector {
    /// 创建新的死锁检测器
    pub fn new() -> Self {
        Self {
            wait_graph: Mutex::new(HashMap::new()),
        }
    }
    
    /// 添加等待关系
    pub fn add_wait(&self, waiter: &TransactionId, holder: &TransactionId) -> Result<()> {
        let mut graph = self.wait_graph.lock().map_err(|_| Error::lock("无法获取等待图锁"))?;
        graph.entry(waiter.clone()).or_insert_with(HashSet::new).insert(holder.clone());
        Ok(())
    }
    
    /// 移除等待关系
    pub fn remove_wait(&self, waiter: &TransactionId, holder: &TransactionId) -> Result<()> {
        let mut graph = self.wait_graph.lock().map_err(|_| Error::lock("无法获取等待图锁"))?;
        if let Some(waiters) = graph.get_mut(waiter) {
            waiters.remove(holder);
            if waiters.is_empty() {
                graph.remove(waiter);
            }
        }
        Ok(())
    }
    
    /// 检测死锁
    pub fn detect_deadlock(&self) -> Result<Option<Vec<TransactionId>>> {
        let graph = self.wait_graph.lock().map_err(|_| Error::lock("无法获取等待图锁"))?;
        
        // 使用DFS检测环
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        
        for tx_id in graph.keys() {
            if !visited.contains(tx_id) {
                if let Some(cycle) = self.dfs_detect_cycle(&graph, tx_id, &mut visited, &mut rec_stack, &mut Vec::new()) {
                    return Ok(Some(cycle));
                }
            }
        }
        
        Ok(None)
    }
    
    /// DFS检测环
    fn dfs_detect_cycle(
        &self,
        graph: &HashMap<TransactionId, HashSet<TransactionId>>,
        current: &TransactionId,
        visited: &mut HashSet<TransactionId>,
        rec_stack: &mut HashSet<TransactionId>,
        path: &mut Vec<TransactionId>,
    ) -> Option<Vec<TransactionId>> {
        visited.insert(current.clone());
        rec_stack.insert(current.clone());
        path.push(current.clone());
        
        if let Some(neighbors) = graph.get(current) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    if let Some(cycle) = self.dfs_detect_cycle(graph, neighbor, visited, rec_stack, path) {
                        return Some(cycle);
                    }
                } else if rec_stack.contains(neighbor) {
                    // 找到环，返回环路径
                    let cycle_start = path.iter().position(|x| x == neighbor).unwrap();
                    return Some(path[cycle_start..].to_vec());
                }
            }
        }
        
        rec_stack.remove(current);
        path.pop();
        None
    }
    
    /// 清理事务的所有等待关系
    pub fn cleanup_transaction(&self, tx_id: &TransactionId) -> Result<()> {
        let mut graph = self.wait_graph.lock().map_err(|_| Error::lock("无法获取等待图锁"))?;
        
        // 移除该事务的等待关系
        graph.remove(tx_id);
        
        // 移除其他事务对该事务的等待
        for waiters in graph.values_mut() {
            waiters.remove(tx_id);
        }
        
        // 清理空的等待集合
        graph.retain(|_, waiters| !waiters.is_empty());
        
        Ok(())
    }
}

/// 事务管理器，负责管理所有活跃事务
pub struct TransactionManager {
    /// 活跃事务
    active_transactions: Mutex<HashMap<TransactionId, Arc<TransactionWrapper>>>,
    
    /// 事务日志目录
    log_dir: PathBuf,
    
    /// 存储引擎
    db: Arc<dyn StorageEngine>,
    
    /// 事务超时设置
    timeout: Duration,
    
    /// 死锁检测器
    deadlock_detector: DeadlockDetector,
    
    /// 事务日志写入器
    log_writer: Mutex<Option<fs::File>>,
}

impl TransactionManager {
    /// 创建新的事务管理器
    pub fn new(db: Arc<dyn StorageEngine>, log_dir: PathBuf, timeout_secs: u64) -> Self {
        // 确保日志目录存在
        if !log_dir.exists() {
            if let Err(e) = fs::create_dir_all(&log_dir) {
                error!("无法创建事务日志目录: {}", e);
            }
        }
        
        Self {
            active_transactions: Mutex::new(HashMap::new()),
            log_dir,
            db,
            timeout: Duration::from_secs(timeout_secs),
            deadlock_detector: DeadlockDetector::new(),
            log_writer: Mutex::new(None),
        }
    }
    
    /// 开始新事务
    pub fn begin_transaction(&self) -> Result<Arc<TransactionWrapper>> {
        self.begin_transaction_with_isolation(IsolationLevel::ReadCommitted)
    }
    
    /// 开始指定隔离级别的事务
    pub fn begin_transaction_with_isolation(&self, isolation_level: IsolationLevel) -> Result<Arc<TransactionWrapper>> {
        let tx_id = Uuid::new_v4().to_string();
        let transaction = Transaction::new(tx_id.clone(), self.db.clone(), isolation_level);
        let wrapper = Arc::new(TransactionWrapper::new(transaction));
        
        // 记录到活跃事务列表
        let mut active = self.active_transactions.lock().map_err(|_| Error::lock("无法获取活跃事务锁"))?;
        active.insert(tx_id.clone(), wrapper.clone());
        
        // 写入事务日志
        self.write_transaction_log(&tx_id, "BEGIN")?;
        
        info!("开始新事务: {}, 隔离级别: {:?}", tx_id, isolation_level);
        Ok(wrapper)
    }
    
    /// 获取事务
    pub fn get_transaction(&self, tx_id: &str) -> Result<Option<Arc<TransactionWrapper>>> {
        let active = self.active_transactions.lock().map_err(|_| Error::lock("无法获取活跃事务锁"))?;
        Ok(active.get(tx_id).cloned())
    }
    
    /// 结束事务
    pub fn end_transaction(&self, tx_id: &str) -> Result<()> {
        let mut active = self.active_transactions.lock().map_err(|_| Error::lock("无法获取活跃事务锁"))?;
        
        if let Some(wrapper) = active.remove(tx_id) {
            // 清理死锁检测器中的相关信息
            self.deadlock_detector.cleanup_transaction(&tx_id.to_string())?;
            
            // 写入事务日志
            self.write_transaction_log(tx_id, "END")?;
            
            debug!("结束事务: {}", tx_id);
        }
        
        Ok(())
    }
    
    /// 清理超时事务
    pub fn cleanup_timed_out_transactions(&self) -> Result<usize> {
        let mut active = self.active_transactions.lock().map_err(|_| Error::lock("无法获取活跃事务锁"))?;
        let mut timed_out = Vec::new();
        
        for (tx_id, wrapper) in active.iter() {
            if let Ok(mut tx) = wrapper.lock() {
                if tx.check_timeout(self.timeout) {
                    timed_out.push(tx_id.clone());
                }
            }
        }
        
        let count = timed_out.len();
        for tx_id in timed_out {
            active.remove(&tx_id);
            self.deadlock_detector.cleanup_transaction(&tx_id)?;
            self.write_transaction_log(&tx_id, "TIMEOUT")?;
            warn!("清理超时事务: {}", tx_id);
        }
        
        Ok(count)
    }
    
    /// 获取活跃事务数量
    pub fn count_active(&self) -> Result<usize> {
        let active = self.active_transactions.lock().map_err(|_| Error::lock("无法获取活跃事务锁"))?;
        Ok(active.len())
    }
    
    /// 获取活跃事务ID列表
    pub fn get_active_transaction_ids(&self) -> Result<Vec<String>> {
        let active = self.active_transactions.lock().map_err(|_| Error::lock("无法获取活跃事务锁"))?;
        Ok(active.keys().cloned().collect())
    }
    
    /// 检测并解决死锁
    pub fn detect_and_resolve_deadlocks(&self) -> Result<usize> {
        if let Some(cycle) = self.deadlock_detector.detect_deadlock()? {
            warn!("检测到死锁，涉及事务: {:?}", cycle);
            
            // 选择最年轻的事务进行回滚
            if let Some(victim_id) = cycle.last() {
                if let Some(wrapper) = self.get_transaction(victim_id)? {
                    if let Ok(mut tx) = wrapper.lock() {
                        tx.abort();
                        self.write_transaction_log(victim_id, "DEADLOCK_ABORT")?;
                        warn!("因死锁回滚事务: {}", victim_id);
                    }
                }
                self.end_transaction(victim_id)?;
                return Ok(1);
            }
        }
        
        Ok(0)
    }
    
    /// 写入事务日志
    fn write_transaction_log(&self, tx_id: &str, action: &str) -> Result<()> {
        let log_entry = format!(
            "{} {} {}\n",
            chrono::Utc::now().to_rfc3339(),
            tx_id,
            action
        );
        
        let log_file_path = self.log_dir.join("transaction.log");
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_file_path)
            .map_err(|e| Error::io(format!("无法打开事务日志文件: {}", e)))?;
        
        file.write_all(log_entry.as_bytes())
            .map_err(|e| Error::io(format!("无法写入事务日志: {}", e)))?;
        
        file.flush()
            .map_err(|e| Error::io(format!("无法刷新事务日志: {}", e)))?;
        
        Ok(())
    }
    
    /// 从日志恢复事务
    pub fn recover_from_log(&self) -> Result<usize> {
        let log_file_path = self.log_dir.join("transaction.log");
        if !log_file_path.exists() {
            return Ok(0);
        }
        
        let content = fs::read_to_string(log_file_path)
            .map_err(|e| Error::io(format!("无法读取事务日志: {}", e)))?;
        
        let mut recovered = 0;
        let mut pending_transactions = HashMap::new();
        
        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                let _timestamp = parts[0];
                let tx_id = parts[1];
                let action = parts[2];
                
                match action {
                    "BEGIN" => {
                        pending_transactions.insert(tx_id.to_string(), true);
                    }
                    "END" | "COMMIT" | "ROLLBACK" | "TIMEOUT" | "DEADLOCK_ABORT" => {
                        pending_transactions.remove(tx_id);
                    }
                    _ => {}
                }
            }
        }
        
        // 对于未完成的事务，创建恢复事务进行回滚
        for tx_id in pending_transactions.keys() {
            warn!("恢复未完成事务: {}", tx_id);
            // 这里可以添加具体的恢复逻辑
            recovered += 1;
        }
        
        info!("从日志恢复了 {} 个事务", recovered);
        Ok(recovered)
    }
    
    /// 获取事务统计信息
    pub fn get_transaction_stats(&self) -> Result<Vec<TransactionStats>> {
        let active = self.active_transactions.lock().map_err(|_| Error::lock("无法获取活跃事务锁"))?;
        let mut stats = Vec::new();
        
        for wrapper in active.values() {
            if let Ok(tx) = wrapper.lock() {
                stats.push(tx.get_stats());
            }
        }
        
        Ok(stats)
    }
}

/// 事务函数，自动处理提交和回滚
pub fn transactional<F, T, E>(
    tx_manager: &TransactionManager,
    func: F,
) -> std::result::Result<T, E>
where
    F: FnOnce(&mut Transaction) -> std::result::Result<T, E>,
    E: From<Error>,
{
    transactional_with_isolation(tx_manager, IsolationLevel::default(), func)
}

/// 以指定隔离级别执行事务函数
pub fn transactional_with_isolation<F, T, E>(
    tx_manager: &TransactionManager,
    isolation_level: IsolationLevel,
    func: F,
) -> std::result::Result<T, E>
where
    F: FnOnce(&mut Transaction) -> std::result::Result<T, E>,
    E: From<Error>,
{
    // 开始事务
    let tx_wrapper = tx_manager.begin_transaction_with_isolation(isolation_level)
        .map_err(E::from)?;
    
    let mut tx = tx_wrapper.lock().map_err(E::from)?;
    let tx_id = tx.id().clone();
    
    // 执行函数
    let result = func(&mut tx);
    
    // 根据结果提交或回滚
    match &result {
        Ok(_) => {
            // 提交事务
            if let Err(e) = tx.commit() {
                error!("事务 {} 提交失败: {}", tx_id, e);
                return Err(E::from(e));
            }
        },
        Err(_) => {
            // 回滚事务
            tx.rollback();
        }
    }
    
    // 结束事务
    drop(tx);
    if let Err(e) = tx_manager.end_transaction(&tx_id) {
        error!("结束事务 {} 失败: {}", tx_id, e);
        return Err(E::from(e));
    }
    
    result
} 