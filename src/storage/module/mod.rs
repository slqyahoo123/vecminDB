// 存储模块
// 提供存储引擎的核心功能实现

pub mod config;
mod cache;
mod data;
// 注释掉：模型相关模块，vecminDB不需要
// mod model;
mod algorithm;
mod transaction;
mod utils;
mod permissions;
mod core;
mod data_ops;
// mod model_ops;
mod algorithm_ops;
mod training_ops;
mod permissions_ops;

// 从各个子模块重新导出公共接口
pub use config::StorageConfig as ModuleStorageConfig;
pub use cache::{CachePolicy, CacheEntry, CacheManager};
pub use data::{DataInfo, DataOperations, CF_RAW_DATA, CF_PROCESSED_DATA, DATA_RAW_PREFIX, DATA_PROCESSED_PREFIX};
// 注释掉：模型操作，vecminDB不需要
// pub use model::{ModelOperations, CF_MODEL_PARAMS, CF_CHECKPOINTS, MODEL_PARAMS_PREFIX, MODEL_ARCH_PREFIX};
pub use algorithm::{AlgorithmOperations, CF_ALGORITHMS, ALGO_CODE_PREFIX};
pub use transaction::{StorageTransaction, WriteBatchExt, CF_VERSIONS, CF_INFO, VERSION_PREFIX};
pub use utils::{str_to_bytes, bytes_to_str};
pub use core::Storage;
pub use permissions::PermissionManager;

// 重新导出包外可见接口 