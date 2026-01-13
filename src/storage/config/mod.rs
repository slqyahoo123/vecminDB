use std::path::PathBuf;
use serde::{Serialize, Deserialize};

/// 压缩策略类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CompactionStyle {
    /// 级别压缩
    Level,
    /// 通用压缩
    Universal,
    /// 无压缩
    None,
}

/// 访问提示类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AccessHint {
    /// 正常访问模式
    Normal,
    /// 顺序访问模式
    Sequential,
    /// 随机访问模式
    Random,
    /// 批量访问模式
    Bulk,
}

/// 存储引擎配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// 数据存储路径
    pub path: PathBuf,
    /// 是否创建不存在的目录
    pub create_if_missing: bool,
    /// 最大后台工作线程数
    pub max_background_jobs: i32,
    /// 写缓冲大小（字节）
    pub write_buffer_size: usize,
    /// 最大打开文件数
    pub max_open_files: i32,
    /// 使用压缩
    pub use_compression: bool,
    /// 最大文件大小（字节）
    pub max_file_size: u64,
    
    // 新增字段用于修复编译错误
    /// 最大连接数
    pub max_connections: u32,
    /// 连接超时时间（毫秒）
    pub connection_timeout: u64,
    /// 是否启用WAL（Write-Ahead Log）
    pub enable_wal: bool,
    /// 缓存大小（MB）
    pub cache_size: usize,
    /// 压缩级别
    pub compression: Option<u32>,
    
    // 添加缺失的字段
    /// 缓存大小（MB）
    pub cache_size_mb: usize,
    /// WAL缓冲区大小（字节）
    pub wal_buffer_size: usize,
    /// 最大WAL大小（字节）
    pub max_wal_size: usize,
    /// 是否启用压缩
    pub enable_compaction: bool,
    /// 压缩样式
    pub compaction_style: CompactionStyle,
    /// 目标文件大小基数（字节）
    pub target_file_size_base: usize,
    /// 最大级别基数（字节）
    pub max_bytes_for_level_base: usize,
    /// Level0文件数压缩触发
    pub level0_file_num_compaction_trigger: i32,
    /// 引擎ID
    pub engine_id: String,
    /// Level0减速写入触发
    pub level0_slowdown_writes_trigger: i32,
    /// Level0停止写入触发
    pub level0_stop_writes_trigger: i32,
    /// 最大写缓冲区数量
    pub max_write_buffer_number: i32,
    /// 最小写缓冲区合并数量
    pub min_write_buffer_number_to_merge: i32,
    /// 最大压缩字节数
    pub max_compaction_bytes: usize,
    /// 使用直接读取
    pub use_direct_reads: bool,
    /// 使用直接IO进行刷新和压缩
    pub use_direct_io_for_flush_and_compaction: bool,
    /// 允许并发内存表写入
    pub allow_concurrent_memtable_write: bool,
    
    // 高级RocksDB配置选项
    /// 启用管道写入
    pub enable_pipelined_write: bool,
    /// 启用写入线程自适应让步
    pub enable_write_thread_adaptive_yield: bool,
    /// 写入线程最大让步时间（微秒）
    pub write_thread_max_yield_usec: u64,
    /// 写入线程慢速让步时间（微秒）
    pub write_thread_slow_yield_usec: u64,
    /// 跳过数据库打开时的统计更新
    pub skip_stats_update_on_db_open: bool,
    /// 跳过数据库打开时的SST文件大小检查
    pub skip_checking_sst_file_sizes_on_db_open: bool,
    /// 允许两阶段提交
    pub allow_2pc: bool,
    /// 选项文件错误时失败
    pub fail_if_options_file_error: bool,
    /// 转储malloc统计信息
    pub dump_malloc_stats: bool,
    /// 恢复期间避免刷新
    pub avoid_flush_during_recovery: bool,
    /// 关闭期间避免刷新
    pub avoid_flush_during_shutdown: bool,
    /// 允许fallocate
    pub allow_fallocate: bool,
    /// 文件描述符在exec时关闭
    pub is_fd_close_on_exec: bool,
    /// 统计信息转储周期（秒）
    pub stats_dump_period_sec: u64,
    /// 统计信息持久化周期（秒）
    pub stats_persist_period_sec: u64,
    /// 统计历史缓冲区大小（字节）
    pub stats_history_buffer_size: usize,
    /// 打开时建议随机访问
    pub advise_random_on_open: bool,
    /// 数据库写缓冲区大小（字节）
    pub db_write_buffer_size: usize,
    /// 压缩开始时的访问提示
    pub access_hint_on_compaction_start: AccessHint,
    /// 为压缩输入创建新表读取器
    pub new_table_reader_for_compaction_inputs: bool,
    /// 压缩预读大小（字节）
    pub compaction_readahead_size: usize,
    /// 随机访问最大缓冲区大小（字节）
    pub random_access_max_buffer_size: usize,
    /// 可写文件最大缓冲区大小（字节）
    pub writable_file_max_buffer_size: usize,
    /// 使用自适应互斥锁
    pub use_adaptive_mutex: bool,
    /// 每次同步字节数
    pub bytes_per_sync: u64,
    /// WAL每次同步字节数
    pub wal_bytes_per_sync: u64,
    /// 严格字节同步
    pub strict_bytes_per_sync: bool,
    /// 启用线程跟踪
    pub enable_thread_tracking: bool,
    /// 允许mmap读取
    pub allow_mmap_reads: bool,
    /// 允许mmap写入
    pub allow_mmap_writes: bool,
    /// 使用fsync
    pub use_fsync: bool,
}

/// 存储配置更新结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfigUpdate {
    /// 数据存储路径
    pub path: Option<PathBuf>,
    /// 是否创建不存在的目录
    pub create_if_missing: Option<bool>,
    /// 最大后台工作线程数
    pub max_background_jobs: Option<i32>,
    /// 写缓冲大小（字节）
    pub write_buffer_size: Option<usize>,
    /// 最大打开文件数
    pub max_open_files: Option<i32>,
    /// 使用压缩
    pub use_compression: Option<bool>,
    /// 缓存大小（MB）
    pub cache_size_mb: Option<usize>,
}

impl StorageConfig {
    /// 创建新的存储配置
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        Self {
            path: path.into(),
            create_if_missing: true,
            max_background_jobs: 4,
            write_buffer_size: 64 * 1024 * 1024, // 64MB
            max_open_files: 1000,
            use_compression: true,
            max_file_size: 1024 * 1024 * 1024, // 1GB
            max_connections: 100,
            connection_timeout: 5000,
            enable_wal: true,
            engine_id: uuid::Uuid::new_v4().to_string(),
            cache_size: 1024 * 1024 * 1024, // 1GB
            compression: Some(3),
            cache_size_mb: 256,
            wal_buffer_size: 16 * 1024 * 1024, // 16MB
            max_wal_size: 128 * 1024 * 1024, // 128MB
            enable_compaction: true,
            compaction_style: CompactionStyle::Level,
            target_file_size_base: 64 * 1024 * 1024, // 64MB
            max_bytes_for_level_base: 256 * 1024 * 1024, // 256MB
            level0_file_num_compaction_trigger: 4,
            level0_slowdown_writes_trigger: 20,
            level0_stop_writes_trigger: 36,
            max_write_buffer_number: 3,
            min_write_buffer_number_to_merge: 1,
            max_compaction_bytes: 2 * 1024 * 1024 * 1024, // 2GB
            use_direct_reads: false,
            use_direct_io_for_flush_and_compaction: false,
            allow_concurrent_memtable_write: true,
            enable_pipelined_write: true,
            enable_write_thread_adaptive_yield: true,
            write_thread_max_yield_usec: 100,
            write_thread_slow_yield_usec: 3,
            skip_stats_update_on_db_open: false,
            skip_checking_sst_file_sizes_on_db_open: false,
            allow_2pc: false,
            fail_if_options_file_error: false,
            dump_malloc_stats: false,
            avoid_flush_during_recovery: false,
            avoid_flush_during_shutdown: false,
            allow_fallocate: true,
            is_fd_close_on_exec: true,
            stats_dump_period_sec: 600,
            stats_persist_period_sec: 600,
            stats_history_buffer_size: 1024 * 1024, // 1MB
            advise_random_on_open: false,
            db_write_buffer_size: 0,
            access_hint_on_compaction_start: AccessHint::Normal,
            new_table_reader_for_compaction_inputs: false,
            compaction_readahead_size: 0,
            random_access_max_buffer_size: 1024 * 1024, // 1MB
            writable_file_max_buffer_size: 1024 * 1024, // 1MB
            use_adaptive_mutex: false,
            bytes_per_sync: 0,
            wal_bytes_per_sync: 0,
            strict_bytes_per_sync: false,
            enable_thread_tracking: false,
            allow_mmap_reads: false,
            allow_mmap_writes: false,
            use_fsync: false,
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("data/storage"),
            create_if_missing: true,
            max_background_jobs: 4,
            write_buffer_size: 64 * 1024 * 1024, // 64MB
            max_open_files: 1000,
            use_compression: true,
            max_file_size: 1024 * 1024 * 1024, // 1GB
            engine_id: uuid::Uuid::new_v4().to_string(),
            max_connections: 100,
            connection_timeout: 5000,
            enable_wal: true,
            cache_size: 1024 * 1024 * 1024, // 1GB
            compression: Some(3),
            cache_size_mb: 256,
            wal_buffer_size: 16 * 1024 * 1024, // 16MB
            max_wal_size: 128 * 1024 * 1024, // 128MB
            enable_compaction: true,
            compaction_style: CompactionStyle::Level,
            target_file_size_base: 64 * 1024 * 1024, // 64MB
            max_bytes_for_level_base: 256 * 1024 * 1024, // 256MB
            level0_file_num_compaction_trigger: 4,
            level0_slowdown_writes_trigger: 20,
            level0_stop_writes_trigger: 36,
            max_write_buffer_number: 3,
            min_write_buffer_number_to_merge: 1,
            max_compaction_bytes: 2 * 1024 * 1024 * 1024, // 2GB
            use_direct_reads: false,
            use_direct_io_for_flush_and_compaction: false,
            allow_concurrent_memtable_write: true,
            enable_pipelined_write: true,
            enable_write_thread_adaptive_yield: true,
            write_thread_max_yield_usec: 100,
            write_thread_slow_yield_usec: 3,
            skip_stats_update_on_db_open: false,
            skip_checking_sst_file_sizes_on_db_open: false,
            allow_2pc: false,
            fail_if_options_file_error: false,
            dump_malloc_stats: false,
            avoid_flush_during_recovery: false,
            avoid_flush_during_shutdown: false,
            allow_fallocate: true,
            is_fd_close_on_exec: true,
            stats_dump_period_sec: 600,
            stats_persist_period_sec: 600,
            stats_history_buffer_size: 1024 * 1024, // 1MB
            advise_random_on_open: false,
            db_write_buffer_size: 0,
            access_hint_on_compaction_start: AccessHint::Normal,
            new_table_reader_for_compaction_inputs: false,
            compaction_readahead_size: 0,
            random_access_max_buffer_size: 1024 * 1024, // 1MB
            writable_file_max_buffer_size: 1024 * 1024, // 1MB
            use_adaptive_mutex: false,
            bytes_per_sync: 0,
            wal_bytes_per_sync: 0,
            strict_bytes_per_sync: false,
            enable_thread_tracking: false,
            allow_mmap_reads: false,
            allow_mmap_writes: false,
            use_fsync: false,
        }
    }
} 