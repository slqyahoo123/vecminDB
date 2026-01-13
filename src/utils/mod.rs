/// 通用工具模块

pub mod compression;
pub mod config;
pub mod validation;
pub mod batch;
pub mod files;
pub mod hash;
pub mod concurrency;
pub mod id;
pub mod json;
pub mod log_utils;
pub mod metrics;
pub mod time;
pub mod path;
pub mod parameters;
pub mod errors;

// 添加hashing模块重新导出
pub mod hashing {
    // 重新导出根目录下hashing.rs中的功能
    pub use crate::hashing::{
        HashFactory, ModelHasher, GlobalHasher,
        HashAlgorithm, HashResult
    };
    
    // 提供便利函数
    pub fn generate_model_hash(
        model_id: &str, 
        version: Option<&str>, 
        metadata: Option<&std::collections::HashMap<String, String>>
    ) -> String {
        ModelHasher::generate_model_hash(model_id, version, metadata)
    }
    
    pub fn compute_hash_with_algorithm(algorithm: HashAlgorithm, input: &[u8]) -> crate::Result<Vec<u8>> {
        GlobalHasher::hash(algorithm, input)
    }
}

// 重新导出常用功能
pub use hash::{compute_hash, HashAlgorithm as LocalHashAlgorithm, HashResult as LocalHashResult};
pub use concurrency::{
    ThreadPool, ThreadPoolConfig, TaskPriority, AsyncTaskScheduler,
    DistributedLockManager, ConcurrencyLimiter, LockType
};

// 重新导出工具功能
pub use validation::{ValidationError, FieldValidator, DataValidator, NumberValidator};
pub use batch::{BatchProcessor, BatchOperations, DataShard, BatchResult};
pub use files::{FileUtils, PathUtils, FileType, FileResult};
pub use id::{
    UuidGenerator, SnowflakeGenerator, IncrementalIdGenerator, 
    NanoIdGenerator, TimestampIdGenerator, IdFactory
};
pub use time::{
    TimeUtils, Timer, TimeWindow, RateLimiter, Timeout, 
    TimeFormatter
};
 
