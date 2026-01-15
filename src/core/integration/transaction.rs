/// 跨模块事务处理模块
/// 
/// 提供跨模块的分布式事务协调机制，确保数据一致性

use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::time::Duration;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use log::{info, error, debug, warn};
use crate::Result;
use tokio::time::timeout;

/// 事务状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TransactionStatus {
    Created,
    Preparing,
    Prepared,
    Committing,
    Committed,
    Aborting,
    Aborted,
    Failed,
    Timeout,
}

/// 事务参与者状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ParticipantStatus {
    Idle,
    Preparing,
    Prepared,
    Committed,
    Aborted,
    Failed,
}

/// 事务操作类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionOperation {
    ModelUpdate {
        model_id: String,
        parameters: HashMap<String, String>,
    },
    DataProcess {
        batch_id: String,
        operations: Vec<String>,
    },
    AlgorithmExecute {
        algorithm_id: String,
        params: HashMap<String, String>,
    },
    StorageWrite {
        key: String,
        value: String,
    },
    StorageDelete {
        key: String,
    },
}

/// 跨模块事务
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModuleTransaction {
    pub id: String,
    pub name: String,
    pub status: TransactionStatus,
    pub operations: Vec<TransactionOperation>,
    pub participants: HashMap<String, ParticipantStatus>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub timeout_seconds: u64,
    pub retry_count: usize,
    pub max_retries: usize,
    pub metadata: HashMap<String, String>,
    pub error_message: Option<String>,
}

impl CrossModuleTransaction {
    pub fn new(name: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.to_string(),
            status: TransactionStatus::Created,
            operations: Vec::new(),
            participants: HashMap::new(),
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            timeout_seconds: 300, // 5分钟默认超时
            retry_count: 0,
            max_retries: 3,
            metadata: HashMap::new(),
            error_message: None,
        }
    }
    
    pub fn add_operation(mut self, operation: TransactionOperation) -> Self {
        self.operations.push(operation);
        self
    }
    
    pub fn add_participant(mut self, participant_id: String) -> Self {
        self.participants.insert(participant_id, ParticipantStatus::Idle);
        self
    }
    
    pub fn with_timeout(mut self, timeout_seconds: u64) -> Self {
        self.timeout_seconds = timeout_seconds;
        self
    }
    
    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }
    
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// 事务参与者接口
#[async_trait]
pub trait TransactionParticipant: Send + Sync {
    /// 参与者ID
    fn participant_id(&self) -> &str;
    
    /// 准备阶段 - 检查是否可以执行操作
    async fn prepare(&self, transaction_id: &str, operations: &[TransactionOperation]) -> Result<bool>;
    
    /// 提交阶段 - 执行实际操作
    async fn commit(&self, transaction_id: &str) -> Result<()>;
    
    /// 回滚阶段 - 撤销操作
    async fn abort(&self, transaction_id: &str) -> Result<()>;
    
    /// 获取参与者状态
    fn get_status(&self, transaction_id: &str) -> ParticipantStatus;
}

/// 事务协调器
pub struct TransactionCoordinator {
    active_transactions: Arc<RwLock<HashMap<String, CrossModuleTransaction>>>,
    participants: Arc<RwLock<HashMap<String, Arc<dyn TransactionParticipant>>>>,
    timeout_duration: Duration,
    cleanup_interval: Duration,
}

impl TransactionCoordinator {
    pub fn new(timeout_seconds: u64) -> Result<Self> {
        Ok(Self {
            active_transactions: Arc::new(RwLock::new(HashMap::new())),
            participants: Arc::new(RwLock::new(HashMap::new())),
            timeout_duration: Duration::from_secs(timeout_seconds),
            cleanup_interval: Duration::from_secs(60), // 每分钟清理一次
        })
    }
    
    /// 注册事务参与者
    pub fn register_participant(&self, participant: Arc<dyn TransactionParticipant>) {
        let participant_id = participant.participant_id().to_string();
        let mut participants = self.participants.write()
            .expect("参与者列表写入锁获取失败：无法更新参与者");
        participants.insert(participant_id.clone(), participant);
        info!("注册事务参与者: {}", participant_id);
    }
    
    /// 开始事务
    pub async fn begin_transaction(&self, mut transaction: CrossModuleTransaction) -> Result<String> {
        // 验证参与者是否存在
        {
            let participants = self.participants.read()
                .expect("参与者列表读取锁获取失败：无法读取参与者");
            for participant_id in transaction.participants.keys() {
                if !participants.contains_key(participant_id) {
                    return Err(crate::Error::transaction(format!(
                        "事务参与者不存在: {}", 
                        participant_id
                    )));
                }
            }
        }
        
        let transaction_id = transaction.id.clone();
        transaction.started_at = Some(Utc::now());
        transaction.status = TransactionStatus::Preparing;
        
        // 保存事务
        {
            let mut transactions = self.active_transactions.write()
                .expect("事务列表写入锁获取失败：无法更新事务");
            transactions.insert(transaction_id.clone(), transaction);
        }
        
        info!("开始事务: {}", transaction_id);
        
        // 启动事务执行
        let coordinator = Arc::new(self.clone_ref());
        let tx_id = transaction_id.clone();
        tokio::spawn(async move {
            if let Err(e) = coordinator.execute_transaction(&tx_id).await {
                error!("事务执行失败: {}, 错误: {}", tx_id, e);
            }
        });
        
        Ok(transaction_id)
    }
    
    /// 执行事务
    async fn execute_transaction(&self, transaction_id: &str) -> Result<()> {
        info!("执行事务: {}", transaction_id);
        
        // 两阶段提交协议
        match self.two_phase_commit(transaction_id).await {
            Ok(_) => {
                // 更新事务状态为已提交
                {
                    let mut transactions = self.active_transactions.write()
                .expect("事务列表写入锁获取失败：无法更新事务");
                    if let Some(transaction) = transactions.get_mut(transaction_id) {
                        transaction.status = TransactionStatus::Committed;
                        transaction.completed_at = Some(Utc::now());
                    }
                }
                info!("事务提交成功: {}", transaction_id);
                Ok(())
            },
            Err(e) => {
                // 回滚事务
                warn!("事务执行失败，开始回滚: {}", transaction_id);
                if let Err(rollback_err) = self.abort_transaction(transaction_id).await {
                    error!("事务回滚失败: {}, 错误: {}", transaction_id, rollback_err);
                }
                
                // 更新事务状态为失败
                {
                    let mut transactions = self.active_transactions.write()
                .expect("事务列表写入锁获取失败：无法更新事务");
                    if let Some(transaction) = transactions.get_mut(transaction_id) {
                        transaction.status = TransactionStatus::Failed;
                        transaction.error_message = Some(e.to_string());
                        transaction.completed_at = Some(Utc::now());
                    }
                }
                
                Err(e)
            }
        }
    }
    
    /// 两阶段提交协议
    async fn two_phase_commit(&self, transaction_id: &str) -> Result<()> {
        // 阶段一：准备阶段
        self.prepare_phase(transaction_id).await?;
        
        // 阶段二：提交阶段
        self.commit_phase(transaction_id).await?;
        
        Ok(())
    }
    
    /// 准备阶段
    async fn prepare_phase(&self, transaction_id: &str) -> Result<()> {
        debug!("事务准备阶段: {}", transaction_id);
        
        let (participants, operations) = {
            let transactions = self.active_transactions.read().unwrap();
            let transaction = transactions.get(transaction_id)
                .ok_or_else(|| crate::Error::NotFound(format!("事务不存在: {}", transaction_id)))?;
            
            (transaction.participants.keys().cloned().collect::<Vec<_>>(), 
             transaction.operations.clone())
        };
        
        // 先收集参与者引用，避免在 await 期间持有锁
        let prepared_participants: Vec<(String, Arc<dyn TransactionParticipant>)> = {
            let participants_map = self.participants.read().unwrap();
            participants
                .iter()
                .filter_map(|id| participants_map.get(id).cloned().map(|p| (id.clone(), p)))
                .collect()
        };
        
        // 并行向所有参与者发送准备请求
        let mut prepare_tasks = Vec::new();
        for (participant_id, participant) in prepared_participants.into_iter() {
            let tx_id = transaction_id.to_string();
            let ops = operations.clone();
            let task = tokio::spawn(async move {
                participant.prepare(&tx_id, &ops).await
            });
            prepare_tasks.push((participant_id, task));
        }
        
        // 等待所有参与者响应
        let timeout_duration = self.timeout_duration;
        for (participant_id, task) in prepare_tasks {
            let participant_id_clone = participant_id.clone();
            match timeout(timeout_duration, task).await {
                Ok(Ok(Ok(prepared))) => {
                    if prepared {
                        // 更新参与者状态为已准备
                        let mut transactions = self.active_transactions.write()
                .expect("事务列表写入锁获取失败：无法更新事务");
                        if let Some(transaction) = transactions.get_mut(transaction_id) {
                            transaction.participants.insert(participant_id_clone.clone(), ParticipantStatus::Prepared);
                        }
                        debug!("参与者准备成功: {}", participant_id_clone);
                    } else {
                        return Err(crate::Error::transaction(format!(
                            "参与者拒绝准备: {}", 
                            participant_id_clone
                        )));
                    }
                },
                Ok(Ok(Err(e))) => {
                    return Err(crate::Error::Transaction(format!(
                        "参与者准备失败: {}, 错误: {}", 
                        participant_id_clone, 
                        e
                    )));
                },
                Ok(Err(e)) => {
                    return Err(crate::Error::Transaction(format!(
                        "参与者任务失败: {}, 错误: {}", 
                        participant_id_clone, 
                        e
                    )));
                },
                Err(_) => {
                    return Err(crate::Error::Transaction(format!(
                        "参与者准备超时: {}", 
                        participant_id_clone
                    )));
                }
            }
        }
        
        // 更新事务状态为已准备
        {
            let mut transactions = self.active_transactions.write()
                .expect("事务列表写入锁获取失败：无法更新事务");
            if let Some(transaction) = transactions.get_mut(transaction_id) {
                transaction.status = TransactionStatus::Prepared;
            }
        }
        
        info!("事务准备阶段完成: {}", transaction_id);
        Ok(())
    }
    
    /// 提交阶段
    async fn commit_phase(&self, transaction_id: &str) -> Result<()> {
        debug!("事务提交阶段: {}", transaction_id);
        
        let participants = {
            let transactions = self.active_transactions.read().unwrap();
            let transaction = transactions.get(transaction_id)
                .ok_or_else(|| crate::Error::NotFound(format!("事务不存在: {}", transaction_id)))?;
            
            transaction.participants.keys().cloned().collect::<Vec<_>>()
        };
        
        // 更新事务状态为提交中
        {
            let mut transactions = self.active_transactions.write()
                .expect("事务列表写入锁获取失败：无法更新事务");
            if let Some(transaction) = transactions.get_mut(transaction_id) {
                transaction.status = TransactionStatus::Committing;
            }
        }
        
        // 先收集参与者引用，避免在 await 期间持有锁
        let commit_participants: Vec<(String, Arc<dyn TransactionParticipant>)> = {
            let participants_map = self.participants.read().unwrap();
            participants
                .iter()
                .filter_map(|id| participants_map.get(id).cloned().map(|p| (id.clone(), p)))
                .collect()
        };
        
        // 并行向所有参与者发送提交请求
        let mut commit_tasks = Vec::new();
        for (participant_id, participant) in commit_participants.into_iter() {
            let tx_id = transaction_id.to_string();
            let task = tokio::spawn(async move {
                participant.commit(&tx_id).await
            });
            commit_tasks.push((participant_id, task));
        }
        
        // 等待所有参与者提交完成
        for (participant_id, task) in commit_tasks {
            match timeout(self.timeout_duration, task).await {
                Ok(Ok(Ok(()))) => {
                    // 更新参与者状态为已提交
                    let mut transactions = self.active_transactions.write()
                .expect("事务列表写入锁获取失败：无法更新事务");
                    if let Some(transaction) = transactions.get_mut(transaction_id) {
                        transaction.participants.insert(participant_id.clone(), ParticipantStatus::Committed);
                    }
                    debug!("参与者提交成功: {}", participant_id);
                },
                Ok(Ok(Err(e))) => {
                    error!("参与者提交失败: {}, 错误: {}", participant_id, e);
                    // 注意：在两阶段提交中，如果准备阶段成功但提交阶段失败，
                    // 这是一个严重的不一致状态，需要人工干预
                },
                Ok(Err(e)) => {
                    error!("参与者任务失败: {}, 错误: {}", participant_id, e);
                },
                Err(_) => {
                    error!("参与者提交超时: {}", participant_id);
                }
            }
        }
        
        info!("事务提交阶段完成: {}", transaction_id);
        Ok(())
    }
    
    /// 回滚事务
    async fn abort_transaction(&self, transaction_id: &str) -> Result<()> {
        info!("回滚事务: {}", transaction_id);
        
        let participants = {
            let transactions = self.active_transactions.read().unwrap();
            let transaction = transactions.get(transaction_id)
                .ok_or_else(|| crate::Error::NotFound(format!("事务不存在: {}", transaction_id)))?;
            
            transaction.participants.keys().cloned().collect::<Vec<_>>()
        };
        
        // 更新事务状态为回滚中
        {
            let mut transactions = self.active_transactions.write()
                .expect("事务列表写入锁获取失败：无法更新事务");
            if let Some(transaction) = transactions.get_mut(transaction_id) {
                transaction.status = TransactionStatus::Aborting;
            }
        }
        
        // 获取参与者列表，避免跨await边界持有锁
        let participants_list: Vec<Arc<dyn TransactionParticipant>> = {
            let participants_map = self.participants.read().unwrap();
            participants.iter()
                .filter_map(|id| participants_map.get(id).map(|p| p.clone()))
                .collect()
        };
        
        // 并行向所有参与者发送回滚请求
        let mut abort_tasks = Vec::new();
        for (i, participant) in participants_list.iter().enumerate() {
            let participant = participant.clone();
            let tx_id = transaction_id.to_string();
            let participant_id = participants[i].clone();
            
            let task = tokio::spawn(async move {
                participant.abort(&tx_id).await
            });
            abort_tasks.push((participant_id, task));
        }
        
        // 等待所有参与者回滚完成
        for (participant_id, task) in abort_tasks {
            match timeout(self.timeout_duration, task).await {
                Ok(Ok(Ok(()))) => {
                    // 更新参与者状态为已回滚
                    let mut transactions = self.active_transactions.write()
                .expect("事务列表写入锁获取失败：无法更新事务");
                    if let Some(transaction) = transactions.get_mut(transaction_id) {
                        transaction.participants.insert(participant_id.clone(), ParticipantStatus::Aborted);
                    }
                    debug!("参与者回滚成功: {}", participant_id);
                },
                Ok(Ok(Err(e))) => {
                    error!("参与者回滚失败: {}, 错误: {}", participant_id, e);
                },
                Ok(Err(e)) => {
                    error!("参与者任务失败: {}, 错误: {}", participant_id, e);
                },
                Err(_) => {
                    error!("参与者回滚超时: {}", participant_id);
                }
            }
        }
        
        // 更新事务状态为已回滚
        {
            let mut transactions = self.active_transactions.write()
                .expect("事务列表写入锁获取失败：无法更新事务");
            if let Some(transaction) = transactions.get_mut(transaction_id) {
                transaction.status = TransactionStatus::Aborted;
                transaction.completed_at = Some(Utc::now());
            }
        }
        
        info!("事务回滚完成: {}", transaction_id);
        Ok(())
    }
    
    /// 获取事务状态
    pub fn get_transaction_status(&self, transaction_id: &str) -> Option<TransactionStatus> {
        let transactions = self.active_transactions.read().unwrap();
        transactions.get(transaction_id).map(|t| t.status.clone())
    }
    
    /// 获取事务详情
    pub fn get_transaction(&self, transaction_id: &str) -> Option<CrossModuleTransaction> {
        let transactions = self.active_transactions.read().unwrap();
        transactions.get(transaction_id).cloned()
    }
    
    /// 列出活跃事务
    pub fn list_active_transactions(&self) -> Vec<String> {
        let transactions = self.active_transactions.read().unwrap();
        transactions.keys().cloned().collect()
    }
    
    /// 清理完成的事务
    pub fn cleanup_completed_transactions(&self) -> usize {
        let mut transactions = self.active_transactions.write().unwrap();
        let initial_count = transactions.len();
        
        let cutoff_time = Utc::now() - chrono::Duration::hours(1); // 保留1小时内的事务记录
        
        transactions.retain(|_, transaction| {
            match transaction.status {
                TransactionStatus::Committed | 
                TransactionStatus::Aborted | 
                TransactionStatus::Failed | 
                TransactionStatus::Timeout => {
                    if let Some(completed_at) = transaction.completed_at {
                        completed_at > cutoff_time
                    } else {
                        false
                    }
                },
                _ => true, // 保留未完成的事务
            }
        });
        
        let cleaned_count = initial_count - transactions.len();
        if cleaned_count > 0 {
            info!("清理了 {} 个已完成的事务", cleaned_count);
        }
        
        cleaned_count
    }
    
    /// 启动清理任务
    pub async fn start_cleanup_task(&self) {
        let coordinator = Arc::new(self.clone_ref());
        let cleanup_interval = self.cleanup_interval;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(cleanup_interval);
            loop {
                interval.tick().await;
                coordinator.cleanup_completed_transactions();
            }
        });
        
        info!("事务清理任务已启动，间隔: {:?}", cleanup_interval);
    }
    
    /// 创建引用副本
    ///
    /// 克隆所有 Arc 包装的共享状态，创建一个新的 TransactionManager 实例
    /// 指向相同的底层数据。这是一个轻量级操作，不会复制实际的事务数据。
    fn clone_ref(&self) -> Self {
        Self {
            active_transactions: self.active_transactions.clone(),
            participants: self.participants.clone(),
            timeout_duration: self.timeout_duration,
            cleanup_interval: self.cleanup_interval,
        }
    }
}

/// 模型服务事务参与者
pub struct ModelServiceParticipant {
    participant_id: String,
    // 实际实现中会包含模型服务的引用
}

impl ModelServiceParticipant {
    pub fn new(participant_id: String) -> Self {
        Self { participant_id }
    }
}

#[async_trait]
impl TransactionParticipant for ModelServiceParticipant {
    fn participant_id(&self) -> &str {
        &self.participant_id
    }
    
    async fn prepare(&self, transaction_id: &str, operations: &[TransactionOperation]) -> Result<bool> {
        debug!("模型服务准备事务: {}", transaction_id);
        
        // 检查操作是否可执行
        for operation in operations {
            match operation {
                TransactionOperation::ModelUpdate { model_id, .. } => {
                    debug!("准备更新模型: {}", model_id);
                    // 实际实现中会检查模型是否存在、是否可更新等
                },
                _ => {
                    // 忽略不相关的操作
                }
            }
        }
        
        Ok(true)
    }
    
    async fn commit(&self, transaction_id: &str) -> Result<()> {
        debug!("模型服务提交事务: {}", transaction_id);
        
        // 实际实现中会执行模型更新操作
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        
        Ok(())
    }
    
    async fn abort(&self, transaction_id: &str) -> Result<()> {
        debug!("模型服务回滚事务: {}", transaction_id);
        
        // 实际实现中会撤销模型更新操作
        tokio::time::sleep(tokio::time::Duration::from_millis(30)).await;
        
        Ok(())
    }
    
    fn get_status(&self, _transaction_id: &str) -> ParticipantStatus {
        ParticipantStatus::Idle
    }
}

/// 存储服务事务参与者
pub struct StorageServiceParticipant {
    participant_id: String,
    // 实际实现中会包含存储服务的引用
}

impl StorageServiceParticipant {
    pub fn new(participant_id: String) -> Self {
        Self { participant_id }
    }
}

#[async_trait]
impl TransactionParticipant for StorageServiceParticipant {
    fn participant_id(&self) -> &str {
        &self.participant_id
    }
    
    async fn prepare(&self, transaction_id: &str, operations: &[TransactionOperation]) -> Result<bool> {
        debug!("存储服务准备事务: {}", transaction_id);
        
        // 检查存储操作是否可执行
        for operation in operations {
            match operation {
                TransactionOperation::StorageWrite { key, .. } => {
                    debug!("准备写入数据: {}", key);
                    // 实际实现中会检查存储空间、权限等
                },
                TransactionOperation::StorageDelete { key } => {
                    debug!("准备删除数据: {}", key);
                    // 实际实现中会检查数据是否存在
                },
                _ => {
                    // 忽略不相关的操作
                }
            }
        }
        
        Ok(true)
    }
    
    async fn commit(&self, transaction_id: &str) -> Result<()> {
        debug!("存储服务提交事务: {}", transaction_id);
        
        // 实际实现中会执行存储操作
        tokio::time::sleep(tokio::time::Duration::from_millis(30)).await;
        
        Ok(())
    }
    
    async fn abort(&self, transaction_id: &str) -> Result<()> {
        debug!("存储服务回滚事务: {}", transaction_id);
        
        // 实际实现中会撤销存储操作
        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
        
        Ok(())
    }
    
    fn get_status(&self, _transaction_id: &str) -> ParticipantStatus {
        ParticipantStatus::Idle
    }
} 