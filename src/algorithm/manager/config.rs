// 算法管理器配置模块
// 包含配置结构体和默认实现

/// 算法管理器配置
#[derive(Debug, Clone)]
pub struct AlgorithmManagerConfig {
    /// 默认任务超时时间（秒）
    pub default_task_timeout: u64,
    /// 任务最大并发数
    pub max_concurrent_tasks: usize,
    /// 最大缓存算法数量
    pub max_cached_algorithms: usize,
    /// 任务结果保留时间（秒）
    pub task_result_ttl: u64,
    /// 任务默认重试次数
    pub default_retry_count: u32,
    /// 任务重试延迟（秒）
    pub retry_delay: u64,
    /// 是否启用安全检查
    pub enable_security_check: bool,
    /// 是否启用资源限制
    pub enable_resource_limits: bool,
    /// 模型分片大小 
    pub model_chunk_size: usize,
    /// 算法执行超时时间（毫秒）
    pub algorithm_timeout_ms: u64,
    /// 调试模式
    pub debug_mode: bool,
    /// 最大输出大小（字节）
    pub max_output_size: usize,
    /// 最大内存使用量（字节）
    pub max_memory_usage: usize,
}

impl Default for AlgorithmManagerConfig {
    fn default() -> Self {
        Self {
            default_task_timeout: 3600, // 默认1小时超时
            max_concurrent_tasks: 10,   // 最多10个并发任务
            max_cached_algorithms: 100, // 最多缓存100个算法
            task_result_ttl: 86400,     // 默认保留一天
            default_retry_count: 3,     // 默认重试3次
            retry_delay: 5,             // 默认重试延迟5秒
            enable_security_check: true,
            enable_resource_limits: true,
            model_chunk_size: 1024 * 1024, // 1MB
            algorithm_timeout_ms: 60000, // 默认60秒超时
            debug_mode: false,          // 默认关闭调试模式
            max_output_size: 100 * 1024 * 1024, // 100MB输出限制
            max_memory_usage: 1024 * 1024 * 1024, // 1GB内存限制
        }
    }
}

/// 任务资源限制
#[derive(Debug, Clone)]
pub struct TaskResourceLimits {
    /// 内存限制 (bytes)
    pub memory_limit: usize,
    /// CPU时间限制 (秒)
    pub cpu_time_limit: u64,
    /// 网络带宽限制 (bytes/sec)
    pub network_bandwidth_limit: Option<usize>,
    /// 磁盘限制 (bytes)
    pub disk_space_limit: Option<usize>,
    /// GPU内存限制 (bytes)
    pub gpu_memory_limit: Option<usize>,
}

impl Default for TaskResourceLimits {
    fn default() -> Self {
        Self {
            memory_limit: 1024 * 1024 * 1024, // 1GB
            cpu_time_limit: 3600,             // 1小时
            network_bandwidth_limit: Some(10 * 1024 * 1024), // 10MB/s
            disk_space_limit: Some(10 * 1024 * 1024 * 1024), // 10GB
            gpu_memory_limit: None,
        }
    }
} 