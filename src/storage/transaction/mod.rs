// 事务管理模块
//
// 提供数据库事务处理功能，支持原子性操作和错误恢复

pub mod implementation;
mod isolation;
mod deadlock;

// 重新导出事务管理组件
pub use implementation::{
    Transaction,
    TransactionManager,
    TransactionState,
    Operation,
    transactional,
    ConflictStrategy,
    TransactionOptions,
};

// 重新导出隔离级别相关组件
pub use isolation::{
    TransactionVisibilityManager,
    IsolationController,
    IsolationBehavior,
};

// 重新导出死锁检测相关组件
pub use deadlock::{
    DeadlockDetector,
    DeadlockDetectorConfig,
    DeadlockResolutionStrategy,
    ResourceId,
};

// 使用本模块中定义的IsolationLevel

// 事务相关常量
pub mod constants {
    /// 默认事务日志目录
    pub const DEFAULT_TRANSACTION_LOG_DIR: &str = "transaction_logs";
    
    /// 默认最大活跃事务数量
    pub const DEFAULT_MAX_ACTIVE_TRANSACTIONS: usize = 100;
    
    /// 默认事务超时时间（秒）
    pub const DEFAULT_TRANSACTION_TIMEOUT_SECS: u64 = 60;
}

// 定义事务隔离级别
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum IsolationLevel {
    /// 读未提交 - 最低隔离级别，允许脏读
    ReadUncommitted,
    
    /// 读已提交 - 防止脏读，允许不可重复读和幻读
    ReadCommitted,
    
    /// 可重复读 - 防止脏读和不可重复读，允许幻读
    RepeatableRead,
    
    /// 串行化 - 最高隔离级别，防止所有并发问题
    Serializable,
}

impl Default for IsolationLevel {
    fn default() -> Self {
        IsolationLevel::ReadCommitted
    }
}

impl IsolationLevel {
    /// 将隔离级别转换为字符串描述
    pub fn as_str(&self) -> &'static str {
        match self {
            IsolationLevel::ReadUncommitted => "read_uncommitted",
            IsolationLevel::ReadCommitted => "read_committed",
            IsolationLevel::RepeatableRead => "repeatable_read",
            IsolationLevel::Serializable => "serializable",
        }
    }
    
    /// 从字符串解析隔离级别
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "read_uncommitted" => Some(IsolationLevel::ReadUncommitted),
            "read_committed" => Some(IsolationLevel::ReadCommitted),
            "repeatable_read" => Some(IsolationLevel::RepeatableRead),
            "serializable" => Some(IsolationLevel::Serializable),
            _ => None,
        }
    }
}

// 事务配置
#[derive(Debug, Clone)]
pub struct TransactionConfig {
    /// 事务隔离级别
    pub isolation_level: IsolationLevel,
    
    /// 最大活跃事务数
    pub max_active_transactions: usize,
    
    /// 事务超时时间（秒）
    pub transaction_timeout_secs: u64,
    
    /// 事务日志目录
    pub transaction_log_dir: String,
    
    /// 是否启用异步提交
    pub enable_async_commit: bool,
    
    /// 过期事务超时时间（秒）
    pub stale_transaction_timeout: u64,
}

impl Default for TransactionConfig {
    fn default() -> Self {
        Self {
            isolation_level: IsolationLevel::default(),
            max_active_transactions: constants::DEFAULT_MAX_ACTIVE_TRANSACTIONS,
            transaction_timeout_secs: constants::DEFAULT_TRANSACTION_TIMEOUT_SECS,
            transaction_log_dir: constants::DEFAULT_TRANSACTION_LOG_DIR.to_string(),
            enable_async_commit: false,
            stale_transaction_timeout: 3600, // 默认1小时
        }
    }
}

// 事务ID类型
pub type TransactionId = String;

// 提供事务辅助宏
#[macro_export]
macro_rules! with_transaction_manager {
    ($tx_manager:expr, $body:expr) => {
        $crate::storage::transaction::transactional($tx_manager, |tx| $body(tx))
    };
    ($tx_manager:expr, $isolation:expr, $body:expr) => {
        $crate::storage::transaction::transactional_with_isolation($tx_manager, $isolation, |tx| $body(tx))
    };
}

// 导出带隔离级别的事务函数
pub fn transactional_with_isolation<F, T, E>(
    tx_manager: &implementation::TransactionManager,
    isolation_level: IsolationLevel,
    func: F,
) -> std::result::Result<T, E>
where
    F: FnOnce(&mut implementation::Transaction) -> std::result::Result<T, E>,
    E: From<crate::error::Error>,
{
    let tx = tx_manager.begin_transaction_with_isolation(isolation_level)
        .map_err(|e| E::from(e))?;
    
    let mut tx_guard = tx.lock().unwrap();
    
    match func(&mut tx_guard) {
        Ok(result) => {
            tx_guard.commit().map_err(|e| E::from(e))?;
            Ok(result)
        },
        Err(e) => {
            tx_guard.rollback();
            Err(e)
        }
    }
}
