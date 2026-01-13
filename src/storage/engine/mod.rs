// VecMind 存储引擎模块
// 
// 该模块提供了完整的存储引擎功能，包括：
// - 核心存储引擎实现
// - 事务管理
// - 模型存储服务
// - 数据集存储服务  
// - 监控统计服务
// - 存储引擎接口定义

// 核心模块
pub mod core;
pub mod transaction;
pub mod interfaces;
pub mod models;
pub mod datasets;
pub mod monitoring;
pub mod types;
// 注释掉：模型管理器，vecminDB不需要
// pub mod model_manager;
pub mod algorithm_manager;
pub mod distributed_manager;
pub mod trait_implementations;
pub mod monitoring_integration;
pub mod dataset_integration;
pub mod transaction_manager;
pub mod advanced_operations;
pub mod config_manager;

// 引入存储引擎子模块（保持向后兼容）
pub mod rocksdb;
pub mod memory;
pub mod factory;
pub mod implementation;

// 统一导出接口
pub use interfaces::{
    StorageService, 
    StorageEngine, 
    DatasetStorageInterface
};
pub use crate::core::interfaces::MonitoringInterface;

// 导出事务管理
pub use transaction::{
    Transaction, 
    TransactionState, 
    TransactionOperation, 
    TransactionManager
};

// 导出核心实现
pub use core::StorageEngineImpl;
pub use core::Storage;

// 导出类型定义
pub use types::*;

// 导出管理器
// pub use model_manager::ModelManager;  // 模块已移除
pub use algorithm_manager::AlgorithmManager;
pub use distributed_manager::DistributedManager;
pub use transaction_manager::TransactionManagerService;
pub use advanced_operations::AdvancedOperationsService;
pub use config_manager::ConfigManagerService;

// 导出trait实现（按需在使用处显式导入，避免无意扩大API面）

// 导出存储服务
pub use models::ModelStorageService;
pub use datasets::DatasetStorageService;
pub use monitoring::MonitoringStorageService;

// 导出配置
pub use crate::storage::config::{StorageConfig, StorageConfigUpdate};

// 导出错误类型
pub use crate::Result;
pub use crate::Error;

// 明确重导出核心存储契约（KV/事务），供API等上层以统一别名使用
pub use crate::core::interfaces::StorageInterface as CoreStorageService;