// src/storage/constants/mod.rs
//
// 存储引擎常量定义
// 包含系统级配置常量和默认值

/// 默认数据目录
pub const DEFAULT_DATA_DIRECTORY: &str = "./data/storage";

/// 默认最大打开文件数
pub const DEFAULT_MAX_OPEN_FILES: i32 = 1000;

/// 默认后台线程数
pub const DEFAULT_BACKGROUND_THREADS: i32 = 4;

/// 默认写缓冲区大小 (64MB)
pub const DEFAULT_WRITE_BUFFER_SIZE: usize = 64 * 1024 * 1024;

/// 默认块缓存大小 (256MB)
pub const DEFAULT_BLOCK_CACHE_SIZE: usize = 256 * 1024 * 1024;

/// 默认表缓存大小 (64MB)
pub const DEFAULT_TABLE_CACHE_SIZE: usize = 64 * 1024 * 1024;

/// 默认压缩等级
pub const DEFAULT_COMPRESSION_LEVEL: i32 = 6;

/// 默认最大日志文件数
pub const DEFAULT_MAX_LOG_FILES: usize = 10;

/// 默认WAL最大大小 (1GB)
pub const DEFAULT_MAX_WAL_SIZE: u64 = 1024 * 1024 * 1024;

/// 默认Level 0文件数触发压缩阈值
pub const DEFAULT_LEVEL0_FILE_NUM_COMPACTION_TRIGGER: i32 = 4;

/// 默认Level 0慢写触发阈值
pub const DEFAULT_LEVEL0_SLOWDOWN_WRITES_TRIGGER: i32 = 20;

/// 默认Level 0停止写入触发阈值
pub const DEFAULT_LEVEL0_STOP_WRITES_TRIGGER: i32 = 36;

/// 默认目标文件大小基数 (64MB)
pub const DEFAULT_TARGET_FILE_SIZE_BASE: u64 = 64 * 1024 * 1024;

/// 默认最大字节数基数 (256MB)
pub const DEFAULT_MAX_BYTES_FOR_LEVEL_BASE: u64 = 256 * 1024 * 1024;

/// 默认最大压缩字节数 (1.6GB)
pub const DEFAULT_MAX_COMPACTION_BYTES: u64 = 1600 * 1024 * 1024;

/// 默认同步间隔 (毫秒)
pub const DEFAULT_SYNC_INTERVAL_MS: u64 = 1000;

/// 默认批处理大小
pub const DEFAULT_BATCH_SIZE: usize = 1000;

/// 默认超时时间 (秒)
pub const DEFAULT_TIMEOUT_SECONDS: u64 = 30;

/// 默认重试次数
pub const DEFAULT_MAX_RETRIES: usize = 3;

/// 默认连接池大小
pub const DEFAULT_CONNECTION_POOL_SIZE: usize = 10;

/// 默认内存限制 (512MB)
pub const DEFAULT_MEMORY_LIMIT: usize = 512 * 1024 * 1024;

/// 默认磁盘限制 (10GB)
pub const DEFAULT_DISK_LIMIT: u64 = 10 * 1024 * 1024 * 1024;

/// 默认检查点间隔 (分钟)
pub const DEFAULT_CHECKPOINT_INTERVAL_MINUTES: u64 = 15;

/// 默认统计更新间隔 (秒)
pub const DEFAULT_STATS_UPDATE_INTERVAL_SECONDS: u64 = 60;

/// 默认清理间隔 (小时)
pub const DEFAULT_CLEANUP_INTERVAL_HOURS: u64 = 24;

/// 系统常量
pub mod system {
    /// 最小文件大小
    pub const MIN_FILE_SIZE: u64 = 1024; // 1KB
    
    /// 最大文件大小
    pub const MAX_FILE_SIZE: u64 = 2 * 1024 * 1024 * 1024; // 2GB
    
    /// 最小内存需求
    pub const MIN_MEMORY_REQUIREMENT: usize = 64 * 1024 * 1024; // 64MB
    
    /// 默认页面大小
    pub const DEFAULT_PAGE_SIZE: usize = 4096; // 4KB
    
    /// 最大键长度
    pub const MAX_KEY_LENGTH: usize = 1024;
    
    /// 最大值长度
    pub const MAX_VALUE_LENGTH: usize = 64 * 1024 * 1024; // 64MB
}

/// 性能常量
pub mod performance {
    /// 默认预读大小
    pub const DEFAULT_READAHEAD_SIZE: usize = 4 * 1024 * 1024; // 4MB
    
    /// 默认I/O超时
    pub const DEFAULT_IO_TIMEOUT_MS: u64 = 5000; // 5秒
    
    /// 默认网络超时
    pub const DEFAULT_NETWORK_TIMEOUT_MS: u64 = 10000; // 10秒
    
    /// 默认CPU核心数
    pub const DEFAULT_CPU_CORES: usize = 4;
    
    /// 默认并发度
    pub const DEFAULT_CONCURRENCY: usize = 100;
}

/// 安全常量
pub mod security {
    /// 默认加密密钥长度
    pub const DEFAULT_ENCRYPTION_KEY_LENGTH: usize = 32; // 256位
    
    /// 默认哈希轮数
    pub const DEFAULT_HASH_ROUNDS: u32 = 12;
    
    /// 默认令牌生存时间 (小时)
    pub const DEFAULT_TOKEN_LIFETIME_HOURS: u64 = 24;
    
    /// 最大重试次数
    pub const MAX_AUTH_RETRIES: usize = 3;
}

/// 监控常量
pub mod monitoring {
    /// 默认指标收集间隔 (秒)
    pub const DEFAULT_METRICS_COLLECTION_INTERVAL_SECONDS: u64 = 10;
    
    /// 默认日志保留天数
    pub const DEFAULT_LOG_RETENTION_DAYS: u64 = 30;
    
    /// 默认警报阈值
    pub const DEFAULT_ALERT_THRESHOLD_PERCENT: f64 = 80.0;
    
    /// 默认健康检查间隔 (秒)
    pub const DEFAULT_HEALTH_CHECK_INTERVAL_SECONDS: u64 = 30;
}

/// 业务常量
pub mod business {
    /// 默认模型版本
    pub const DEFAULT_MODEL_VERSION: &str = "1.0.0";
    
    /// 默认数据集版本
    pub const DEFAULT_DATASET_VERSION: &str = "1.0.0";
    
    /// 默认算法版本
    pub const DEFAULT_ALGORITHM_VERSION: &str = "1.0.0";
    
    /// 最大模型大小 (1GB)
    pub const MAX_MODEL_SIZE: u64 = 1024 * 1024 * 1024;
    
    /// 最大数据集大小 (10GB)
    pub const MAX_DATASET_SIZE: u64 = 10 * 1024 * 1024 * 1024;
}
