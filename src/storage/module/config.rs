// 存储配置模块
// 包含存储引擎配置结构和默认实现

use super::CachePolicy;
use serde::{Serialize, Deserialize};

/// Configuration for storage engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub path: String,
    pub max_background_jobs: i32,
    pub create_if_missing: bool,
    pub enable_compression: bool,
    pub compression_level: i32,
    pub enable_cache: bool,
    pub max_cache_size: usize,
    pub cache_policy: CachePolicy,
    pub permissions_path: String,
    pub max_open_files: i32,
    pub keep_log_file_num: i32,
    pub max_total_wal_size: i64,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            path: "./db_data".to_string(),
            max_background_jobs: 4,
            create_if_missing: true,
            enable_compression: true,
            compression_level: 6,
            enable_cache: true,
            max_cache_size: 1024 * 1024 * 100, // 100MB
            cache_policy: CachePolicy::LRU,
            permissions_path: "./permissions.db".to_string(),
            max_open_files: 1000,
            keep_log_file_num: 10,
            max_total_wal_size: 1 << 30, // 1GB
        }
    }
}

impl StorageConfig {
    pub fn new(path: String) -> Self {
        Self {
            path,
            ..Default::default()
        }
    }
    
    pub fn with_cache_policy(mut self, policy: CachePolicy) -> Self {
        self.cache_policy = policy;
        self
    }
    
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.max_cache_size = size;
        self
    }
}