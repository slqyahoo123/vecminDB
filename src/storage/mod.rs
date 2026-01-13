//! 存储模块
//! 
//! 提供向量数据库的持久化存储功能。
//! 
//! ## 核心组件
//! 
//! - `KeyValueStorageEngine` - 键值存储引擎（Sled/RocksDB）
//! - `TransactionManager` - 事务管理
//! - `VectorStorage` - 向量数据存储
//! 
//! ## 功能特性
//! 
//! - 支持多种存储后端（Sled、RocksDB）
//! - ACID事务支持
//! - 高效的序列化/反序列化
//! - 向量索引持久化
//! - 元数据管理

// 核心存储引擎
pub mod kv_storage;
pub mod transaction;
pub mod types;
pub mod config;

// 向量专用存储
pub mod vector_storage;
pub mod index_storage;
pub mod metadata_storage;

// 工具模块
pub mod serialization;
pub mod compression;

// 其他模块
pub mod engine;
pub mod constants;
pub mod cache;
pub mod models;
pub mod module;
pub mod permissions;
pub mod replication;
pub mod examples;
pub mod tests;
pub mod utils;

// 重新导出核心接口
pub use kv_storage::{
    KeyValueStorageEngine,
    KeyValueStorageConfig,
    // StorageBackend,  // 类型不存在，已移除
};

pub use transaction::{
    TransactionManager,
    Transaction,
    Operation,
    IsolationLevel,
};

// pub use types::{
//     StorageKey,
//     StorageValue,
//     StorageError,
//     StorageResult,
// };  // 这些类型不存在，已移除

pub use config::StorageConfig;

// 向量存储接口
pub use vector_storage::{
    VectorStorage,
    VectorStorageConfig,
    VectorRecord,
};

pub use index_storage::{
    IndexStorage,
    IndexStorageConfig,
    IndexMetadata,
};

pub use metadata_storage::{
    MetadataStorage,
    CollectionMetadata,
};

// 重新导出引擎相关
pub use engine::{StorageEngine, StorageEngineImpl};
pub use module::Storage;

// 重新导出模型相关（保留兼容性）
pub use models::implementation::{
    ModelInfo,
    ModelMetrics,
    StorageFormat,
    ModelStorage,
};

use std::sync::Arc;
use serde::{Serialize, Deserialize};

/// 缓存策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CachePolicy {
    LRU,
    FIFO,
    Random,
}

/// 存储引擎选项
#[derive(Debug, Clone)]
pub struct StorageEngineOptions {
    /// 存储路径
    pub path: std::path::PathBuf,
    /// 缓存大小（MB）
    pub cache_size_mb: Option<usize>,
    /// 最大打开文件数
    pub max_open_files: Option<i32>,
    /// 是否创建不存在的目录
    pub create_if_missing: bool,
    /// 是否使用WAL
    pub use_wal: bool,
}

impl Default for StorageEngineOptions {
    fn default() -> Self {
        Self {
            path: std::path::PathBuf::from("./data"),
            cache_size_mb: Some(512),
            max_open_files: Some(1000),
            create_if_missing: true,
            use_wal: true,
        }
    }
}

/// 初始化默认存储引擎
pub fn init_default_storage(_config: &crate::config::Config) -> crate::Result<Arc<Storage>> {
    let storage_config = module::config::StorageConfig::default();
    Storage::new(storage_config)
}
