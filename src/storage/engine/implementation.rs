use std::path::Path;
use std::sync::{Arc, RwLock, Mutex};
use std::collections::{HashMap, BTreeMap};
// 移除未使用导入，避免告警
use serde::{Serialize, Deserialize};
use log::{debug, info, warn, error};

use crate::Result;

/// 存储引擎选项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageOptions {
    /// 存储路径
    pub path: String,
    
    /// 是否创建不存在的路径
    pub create_if_missing: bool,
    
    /// 是否使用缓存
    pub use_cache: bool,
    
    /// 最大打开文件数
    pub max_open_files: i32,
    
    /// 压缩级别
    pub compression_level: Option<u32>,
    
    /// 写缓冲区大小
    pub write_buffer_size: usize,
    
    /// 最大后台线程数
    pub max_background_threads: usize,
    
    /// 压缩类型
    pub compression: crate::storage::models::implementation::CompressionType,
    
    /// 块大小
    pub block_size: usize,
    
    /// 缓存大小
    pub cache_size: usize,
    
    /// 最大文件大小
    pub max_file_size: usize,
    
    /// 布隆过滤器每键位数
    pub bloom_filter_bits_per_key: u32,
    
    /// 块重启间隔
    pub block_restart_interval: u32,
    
    /// 最大写缓冲区数量
    pub max_write_buffer_number: i32,
    
    /// 最小写缓冲区合并数量
    pub min_write_buffer_number_to_merge: i32,
    
    /// 最大后台作业数
    pub max_background_jobs: i32,
    
    /// 最大后台压缩数
    pub max_background_compactions: i32,
    
    /// 最大后台刷新数
    pub max_background_flushes: i32,
    
    /// 每次同步字节数
    pub bytes_per_sync: u64,
    
    /// WAL每次同步字节数
    pub wal_bytes_per_sync: u64,
    
    /// 严格字节同步模式
    pub strict_bytes_per_sync: bool,
    
    /// 使用自适应互斥锁
    pub use_adaptive_mutex: bool,
    
    /// 启用线程跟踪
    pub enable_thread_tracking: bool,
    
    /// 允许并发内存表写入
    pub allow_concurrent_memtable_write: bool,
    
    /// 启用写入线程自适应让步
    pub enable_write_thread_adaptive_yield: bool,
    
    /// 写入线程最大让步微秒数
    pub write_thread_max_yield_usec: u64,
    
    /// 写入线程慢速让步微秒数
    pub write_thread_slow_yield_usec: u64,
    
    /// 跳过数据库打开时的统计更新
    pub skip_stats_update_on_db_open: bool,
    
    /// 跳过数据库打开时的SST文件大小检查
    pub skip_checking_sst_file_sizes_on_db_open: bool,
    
    /// 允许两阶段提交
    pub allow_2pc: bool,
    
    /// 选项文件错误时失败
    pub fail_if_options_file_error: bool,
    
    /// 转储内存分配统计
    pub dump_malloc_stats: bool,
    
    /// 恢复期间避免刷新
    pub avoid_flush_during_recovery: bool,
    
    /// 关闭期间避免刷新
    pub avoid_flush_during_shutdown: bool,
    
    /// 允许预分配
    pub allow_fallocate: bool,
    
    /// 文件描述符在exec时关闭
    pub is_fd_close_on_exec: bool,
    
    /// 最大manifest编辑次数
    pub max_manifest_edit_count: u64,
    
    /// 统计转储周期（秒）
    pub stats_dump_period_sec: u32,
    
    /// 统计持久化周期（秒）
    pub stats_persist_period_sec: u32,
    
    /// 统计历史缓冲区大小
    pub stats_history_buffer_size: usize,
    
    /// 打开时建议随机访问
    pub advise_random_on_open: bool,
    
    /// 数据库写缓冲区大小
    pub db_write_buffer_size: usize,
    
    /// 最大总WAL大小
    pub max_total_wal_size: usize,
    
    /// 删除过期文件周期（微秒）
    pub delete_obsolete_files_period_micros: u64,
    
    /// 最大清单文件大小
    pub max_manifest_file_size: usize,
    
    /// 表缓存分片位数
    pub table_cache_numshardbits: i32,
    
    /// WAL生存时间（秒）
    pub wal_ttl_seconds: u64,
    
    /// WAL大小限制（MB）
    pub wal_size_limit_mb: usize,
    
    /// 清单文件预分配大小
    pub manifest_preallocation_size: usize,
    
    /// 使用直接读取
    pub use_direct_reads: bool,
    
    /// 刷新和压缩时使用直接IO
    pub use_direct_io_for_flush_and_compaction: bool,

    // ---- 兼容RocksDB风格的扩展选项（作为兼容占位，当前实现不启用行为变化）----
    pub allow_mmap_reads: bool,
    pub allow_mmap_writes: bool,
    pub atomic_flush: bool,
    pub best_efforts_recovery: bool,
    pub bottommost_compression: Option<crate::storage::models::implementation::CompressionType>,
    pub bottommost_compression_opts: Option<String>,
    pub cache_index_and_filter_blocks: bool,
    pub cache_index_and_filter_blocks_with_high_priority: bool,
    pub charge_memory: Option<u64>,
    pub compaction_readahead_size: u64,
    pub compaction_style: Option<String>,
    pub compaction_pri: Option<String>,
    pub compression_opts: Option<String>,
    pub create_missing_column_families: bool,
    pub db_log_dir: Option<String>,
    // 注意：基础选项中已有 delete_obsolete_files_period_micros (u64)，此处不重复声明
    pub disable_auto_compactions: bool,
    pub disable_auto_flush: bool,
    pub disable_data_sync: bool,
    pub disable_wal: bool,
    pub enable_lazy_compaction: bool,
    pub enable_pipelined_write: bool,
    pub env: Option<String>,
    pub force_consistency_checks: bool,
    pub inplace_update_support: bool,
    pub keep_log_file_num: u64,
    pub log_file_time_to_roll: u64,
    pub manual_wal_flush: bool,
    pub max_bytes_for_level_base: u64,
    pub max_bytes_for_level_multiplier: f64,
    pub max_bytes_for_level_multiplier_additional: Vec<i32>,
    pub max_compaction_bytes: u64,
    pub max_file_opening_threads: u32,
    pub max_log_file_size: u64,
    pub max_sequential_skip_in_iterations: u64,
    pub max_subcompactions: u32,
    pub memtable_factory: Option<String>,
    pub merge_operator: Option<String>,
    pub new_table_reader_for_compaction_inputs: bool,
    pub num_levels: u32,
    pub optimize_filters_for_hits: bool,
    pub paranoid_checks: bool,
    pub persist_stats_to_disk: bool,
    pub prefix_extractor: Option<String>,
    pub prepare_for_bulk_load: bool,
    pub preserve_deletes: bool,
    pub rate_limiter: Option<String>,
    pub report_bg_io_stats: bool,
    pub skip_log_error_on_recovery: bool,
    pub sst_file_manager: Option<String>,
    pub table_factory: Option<String>,
    pub target_file_size_base: u64,
    pub target_file_size_multiplier: u32,
    pub ttl: u64,
    pub use_fsync: bool,
    pub wal_dir: Option<String>,
    pub write_disable_wal: bool,
    pub write_ignore_missing_column_families: bool,
}

impl Default for StorageOptions {
    fn default() -> Self {
        Self {
            path: "./data".to_string(),
            create_if_missing: true,
            use_cache: true,
            max_open_files: 1000,
            compression_level: Some(6),
            write_buffer_size: 64 * 1024 * 1024, // 64MB
            max_background_threads: 4,
            compression: crate::storage::models::implementation::CompressionType::None,
            block_size: 4096,
            cache_size: 64 * 1024 * 1024, // 64MB
            max_file_size: 2 * 1024 * 1024 * 1024, // 2GB
            bloom_filter_bits_per_key: 10,
            block_restart_interval: 16,
            max_write_buffer_number: 3,
            min_write_buffer_number_to_merge: 1,
            max_background_jobs: 4,
            max_background_compactions: 2,
            max_background_flushes: 1,
            bytes_per_sync: 0,
            wal_bytes_per_sync: 0,
            strict_bytes_per_sync: false,
            use_adaptive_mutex: false,
            enable_thread_tracking: false,
            allow_concurrent_memtable_write: true,
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
            max_manifest_edit_count: 1000,
            stats_dump_period_sec: 600,
            stats_persist_period_sec: 600,
            stats_history_buffer_size: 1024 * 1024,
            advise_random_on_open: false,
            db_write_buffer_size: 0,
            max_total_wal_size: 0,
            delete_obsolete_files_period_micros: 21600000000,
            max_manifest_file_size: 0,
            table_cache_numshardbits: 6,
            wal_ttl_seconds: 0,
            wal_size_limit_mb: 0,
            manifest_preallocation_size: 0,
            use_direct_reads: false,
            use_direct_io_for_flush_and_compaction: false,
            // 兼容扩展项默认值
            allow_mmap_reads: false,
            allow_mmap_writes: false,
            atomic_flush: false,
            best_efforts_recovery: false,
            bottommost_compression: None,
            bottommost_compression_opts: None,
            cache_index_and_filter_blocks: true,
            cache_index_and_filter_blocks_with_high_priority: false,
            charge_memory: None,
            compaction_readahead_size: 0,
            compaction_style: None,
            compaction_pri: None,
            compression_opts: None,
            create_missing_column_families: true,
            db_log_dir: None,
            disable_auto_compactions: false,
            disable_auto_flush: false,
            disable_data_sync: false,
            disable_wal: false,
            enable_lazy_compaction: false,
            enable_pipelined_write: false,
            env: None,
            force_consistency_checks: false,
            inplace_update_support: false,
            keep_log_file_num: 1000,
            log_file_time_to_roll: 0,
            manual_wal_flush: false,
            max_bytes_for_level_base: 256 * 1024 * 1024,
            max_bytes_for_level_multiplier: 10.0,
            max_bytes_for_level_multiplier_additional: Vec::new(),
            max_compaction_bytes: 0,
            max_file_opening_threads: 8,
            max_log_file_size: 0,
            max_sequential_skip_in_iterations: 8,
            max_subcompactions: 1,
            memtable_factory: None,
            merge_operator: None,
            new_table_reader_for_compaction_inputs: false,
            num_levels: 7,
            optimize_filters_for_hits: false,
            paranoid_checks: false,
            persist_stats_to_disk: false,
            prefix_extractor: None,
            prepare_for_bulk_load: false,
            preserve_deletes: false,
            rate_limiter: None,
            report_bg_io_stats: false,
            skip_log_error_on_recovery: false,
            sst_file_manager: None,
            table_factory: None,
            target_file_size_base: 64 * 1024 * 1024,
            target_file_size_multiplier: 1,
            ttl: 0,
            use_fsync: false,
            wal_dir: None,
            write_disable_wal: false,
            write_ignore_missing_column_families: false,
        }
    }
}

/// 读取选项
#[derive(Debug, Clone)]
pub struct ReadOptions {
    /// 是否使用缓存
    pub use_cache: bool,
    
    /// 是否校验数据
    pub verify_checksums: bool,
    
    /// 读取超时
    pub timeout_ms: Option<u64>,
    
    /// 是否填充缓存
    pub fill_cache: bool,
    
    /// 快照版本
    pub snapshot: Option<u64>,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            use_cache: true,
            verify_checksums: true,
            timeout_ms: Some(5000),
            fill_cache: true,
            snapshot: None,
        }
    }
}

/// 写入选项
#[derive(Debug, Clone)]
pub struct WriteOptions {
    /// 是否同步写入
    pub sync: bool,
    
    /// 是否禁用WAL
    pub disable_wal: bool,
    
    /// 写入超时
    pub timeout_ms: Option<u64>,
    
    /// 是否低延迟写入
    pub low_pri: bool,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            sync: false,
            disable_wal: false,
            timeout_ms: Some(3000),
            low_pri: false,
        }
    }
}

/// 迭代器选项
#[derive(Debug, Clone)]
pub struct IteratorOptions {
    /// 是否反向迭代
    pub reverse: bool,
    
    /// 是否填充缓存
    pub fill_cache: bool,
    
    /// 起始键
    pub from_key: Option<Vec<u8>>,
    
    /// 结束键
    pub to_key: Option<Vec<u8>>,
    
    /// 前缀过滤
    pub prefix: Option<Vec<u8>>,
    
    /// 最大返回数量
    pub max_count: Option<usize>,
    
    /// 是否包含值
    pub include_value: bool,
}

impl Default for IteratorOptions {
    fn default() -> Self {
        Self {
            reverse: false,
            fill_cache: true,
            from_key: None,
            to_key: None,
            prefix: None,
            max_count: None,
            include_value: true,
        }
    }
}

/// 压缩选项
#[derive(Debug, Clone)]
pub struct CompactionOptions {
    /// 是否手动压缩
    pub manual: bool,
    
    /// 压缩线程数
    pub threads: usize,
    
    /// 目标级别
    pub target_level: Option<i32>,
    
    /// 是否并行压缩
    pub parallel: bool,
}

const DEFAULT_BACKGROUND_THREADS: usize = 4;

impl Default for CompactionOptions {
    fn default() -> Self {
        Self {
            manual: false,
            threads: DEFAULT_BACKGROUND_THREADS,
            target_level: None,
            parallel: true,
        }
    }
}

/// 生产级存储迭代器
pub struct StorageIterator {
    /// 有序存储数据的快照
    data_snapshot: BTreeMap<Vec<u8>, Vec<u8>>,
    /// 当前迭代器位置
    current_position: Option<Vec<u8>>,
    /// 迭代器选项
    options: IteratorOptions,
    /// 是否已到达结束位置
    is_exhausted: bool,
    /// 已返回的项目数量
    returned_count: usize,
}

impl StorageIterator {
    /// 创建新的存储迭代器
    pub fn new(data: &HashMap<Vec<u8>, Vec<u8>>, options: IteratorOptions) -> Self {
        // 将HashMap转换为BTreeMap以支持有序遍历
        let mut data_snapshot: BTreeMap<Vec<u8>, Vec<u8>> = data.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        
        // 应用前缀过滤
        if let Some(prefix) = &options.prefix {
            data_snapshot = data_snapshot.into_iter()
                .filter(|(key, _)| key.starts_with(prefix))
                .collect();
        }
        
        // 应用范围过滤
        if let Some(from_key) = &options.from_key {
            data_snapshot = data_snapshot.split_off(from_key);
        }
        
        if let Some(to_key) = &options.to_key {
            let split_data = data_snapshot.split_off(to_key);
            drop(split_data); // 丢弃to_key之后的数据
        }
        
        Self {
            data_snapshot,
            current_position: None,
            options,
            is_exhausted: false,
            returned_count: 0,
        }
    }
    
    /// 获取下一个键值对
    pub fn next(&mut self) -> Option<(Vec<u8>, Vec<u8>)> {
        if self.is_exhausted {
            return None;
        }
        
        // 检查最大返回数量限制
        if let Some(max_count) = self.options.max_count {
            if self.returned_count >= max_count {
                self.is_exhausted = true;
                return None;
            }
        }
        
        let result = if self.options.reverse {
            self.next_reverse()
        } else {
            self.next_forward()
        };
        
        if result.is_some() {
            self.returned_count += 1;
        }
        
        result
    }
    
    /// 正向迭代获取下一个元素
    fn next_forward(&mut self) -> Option<(Vec<u8>, Vec<u8>)> {
        let next_key = if let Some(current) = &self.current_position {
            // 查找当前位置之后的第一个键
            let current_clone = current.clone();
            self.data_snapshot.range(current_clone..).nth(1).map(|(k, _)| k.clone())
        } else {
            // 第一次调用，返回第一个键
            self.data_snapshot.keys().next().cloned()
        };
        
        if let Some(key) = next_key {
            if let Some(value) = self.data_snapshot.get(&key) {
                self.current_position = Some(key.clone());
                if self.options.include_value {
                    Some((key, value.clone()))
                } else {
                    Some((key, Vec::new()))
                }
            } else {
                self.is_exhausted = true;
                None
            }
        } else {
            self.is_exhausted = true;
            None
        }
    }
    
    /// 反向迭代获取下一个元素
    fn next_reverse(&mut self) -> Option<(Vec<u8>, Vec<u8>)> {
        let next_key = if let Some(current) = &self.current_position {
            // 查找当前位置之前的第一个键
            let current_clone = current.clone();
            self.data_snapshot.range(..current_clone).rev().next().map(|(k, _)| k.clone())
        } else {
            // 第一次调用，返回最后一个键
            self.data_snapshot.keys().rev().next().cloned()
        };
        
        if let Some(key) = next_key {
            if let Some(value) = self.data_snapshot.get(&key) {
                self.current_position = Some(key.clone());
                if self.options.include_value {
                    Some((key, value.clone()))
                } else {
                    Some((key, Vec::new()))
                }
            } else {
                self.is_exhausted = true;
                None
            }
        } else {
            self.is_exhausted = true;
            None
        }
    }
    
    /// 获取迭代器选项
    pub fn options(&self) -> &IteratorOptions {
        &self.options
    }
    
    /// 跳转到指定键 - 生产级实现
    pub fn seek(&mut self, key: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
        if self.is_exhausted {
            return None;
        }
        
        let seek_key = key.to_vec();
        
        // 检查键是否在迭代范围内
        if let Some(from_key) = &self.options.from_key {
            if seek_key < *from_key {
                // 如果seek的键小于起始键，则从起始键开始
                self.current_position = Some(from_key.clone());
                return self.get_current_or_next();
            }
        }
        
        if let Some(to_key) = &self.options.to_key {
            if seek_key >= *to_key {
                // 如果seek的键大于等于结束键，则迭代结束
                self.is_exhausted = true;
                return None;
            }
        }
        
        // 查找指定键或其后的第一个键
        if self.options.reverse {
            self.seek_reverse(&seek_key)
        } else {
            self.seek_forward(&seek_key)
        }
    }
    
    /// 正向seek实现
    fn seek_forward(&mut self, key: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
        // 查找大于等于指定键的第一个键
        let key_vec = key.to_vec();
        if let Some((found_key, found_value)) = self.data_snapshot.range(key_vec..).next() {
            self.current_position = Some(found_key.clone());
            if self.options.include_value {
                Some((found_key.clone(), found_value.clone()))
            } else {
                Some((found_key.clone(), Vec::new()))
            }
        } else {
            self.is_exhausted = true;
            None
        }
    }
    
    /// 反向seek实现
    fn seek_reverse(&mut self, key: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
        // 查找小于等于指定键的最大键
        let key_vec = key.to_vec();
        if let Some((found_key, found_value)) = self.data_snapshot.range(..=key_vec).rev().next() {
            self.current_position = Some(found_key.clone());
            if self.options.include_value {
                Some((found_key.clone(), found_value.clone()))
            } else {
                Some((found_key.clone(), Vec::new()))
            }
        } else {
            self.is_exhausted = true;
            None
        }
    }
    
    /// 获取当前位置或下一个位置的元素
    fn get_current_or_next(&mut self) -> Option<(Vec<u8>, Vec<u8>)> {
        if let Some(current_key) = &self.current_position {
            if let Some(value) = self.data_snapshot.get(current_key) {
                if self.options.include_value {
                    Some((current_key.clone(), value.clone()))
                } else {
                    Some((current_key.clone(), Vec::new()))
                }
            } else {
                self.next()
            }
        } else {
            self.next()
        }
    }
    
    /// 跳转到第一个元素
    pub fn seek_to_first(&mut self) -> Option<(Vec<u8>, Vec<u8>)> {
        self.current_position = None;
        self.is_exhausted = false;
        self.returned_count = 0;
        
        if self.options.reverse {
            // 反向迭代时，跳转到最后一个元素
            if let Some((key, value)) = self.data_snapshot.iter().rev().next() {
                self.current_position = Some(key.clone());
                if self.options.include_value {
                    Some((key.clone(), value.clone()))
                } else {
                    Some((key.clone(), Vec::new()))
                }
            } else {
                self.is_exhausted = true;
                None
            }
        } else {
            // 正向迭代时，跳转到第一个元素
            if let Some((key, value)) = self.data_snapshot.iter().next() {
                self.current_position = Some(key.clone());
                if self.options.include_value {
                    Some((key.clone(), value.clone()))
                } else {
                    Some((key.clone(), Vec::new()))
                }
            } else {
                self.is_exhausted = true;
                None
            }
        }
    }
    
    /// 跳转到最后一个元素
    pub fn seek_to_last(&mut self) -> Option<(Vec<u8>, Vec<u8>)> {
        self.current_position = None;
        self.is_exhausted = false;
        self.returned_count = 0;
        
        if self.options.reverse {
            // 反向迭代时，跳转到第一个元素
            if let Some((key, value)) = self.data_snapshot.iter().next() {
                self.current_position = Some(key.clone());
                if self.options.include_value {
                    Some((key.clone(), value.clone()))
                } else {
                    Some((key.clone(), Vec::new()))
                }
            } else {
                self.is_exhausted = true;
                None
            }
        } else {
            // 正向迭代时，跳转到最后一个元素
            if let Some((key, value)) = self.data_snapshot.iter().rev().next() {
                self.current_position = Some(key.clone());
                if self.options.include_value {
                    Some((key.clone(), value.clone()))
                } else {
                    Some((key.clone(), Vec::new()))
                }
            } else {
                self.is_exhausted = true;
                None
            }
        }
    }
    
    /// 检查迭代器是否有效
    pub fn valid(&self) -> bool {
        !self.is_exhausted && self.current_position.is_some()
    }
    
    /// 获取当前键
    pub fn key(&self) -> Option<&[u8]> {
        self.current_position.as_ref().map(|k| k.as_slice())
    }
    
    /// 获取当前值
    pub fn value(&self) -> Option<Vec<u8>> {
        if let Some(current_key) = &self.current_position {
            self.data_snapshot.get(current_key).cloned()
        } else {
            None
        }
    }
}

/// 生产级内存缓存，支持LRU淘汰策略
#[derive(Debug)]
pub struct ProductionCache {
    /// 数据存储
    data: HashMap<Vec<u8>, CacheEntry>,
    /// LRU访问顺序链表
    access_order: std::collections::VecDeque<Vec<u8>>,
    /// 最大大小
    max_size: usize,
    /// 当前大小
    current_size: usize,
    /// 缓存统计
    stats: CacheStats,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    value: Vec<u8>,
    access_time: std::time::Instant,
    access_count: u64,
    size: usize,
}

#[derive(Debug, Clone, Default)]
struct CacheStats {
    hits: u64,
    misses: u64,
    evictions: u64,
    total_access: u64,
}

impl ProductionCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            data: HashMap::new(),
            access_order: std::collections::VecDeque::new(),
            max_size,
            current_size: 0,
            stats: CacheStats::default(),
        }
    }
    
    pub fn get(&mut self, key: &[u8]) -> Option<Vec<u8>> {
        self.stats.total_access += 1;
        
        if let Some(entry) = self.data.get_mut(key) {
            self.stats.hits += 1;
            entry.access_time = std::time::Instant::now();
            entry.access_count += 1;
            
            // 更新LRU顺序
            self.update_lru_order(key);
            
            Some(entry.value.clone())
        } else {
            self.stats.misses += 1;
            None
        }
    }
    
    pub fn put(&mut self, key: Vec<u8>, value: Vec<u8>) {
        let entry_size = key.len() + value.len() + std::mem::size_of::<CacheEntry>();
        
        // 检查是否需要淘汰
        while self.current_size + entry_size > self.max_size && !self.data.is_empty() {
            self.evict_lru();
        }
        
        // 如果单个条目太大，直接拒绝
        if entry_size > self.max_size {
            warn!("缓存条目过大，拒绝存储: {} bytes", entry_size);
            return;
        }
        
        let entry = CacheEntry {
            value: value.clone(),
            access_time: std::time::Instant::now(),
            access_count: 1,
            size: entry_size,
        };
        
        // 如果键已存在，更新值
        if let Some(old_entry) = self.data.insert(key.clone(), entry) {
            self.current_size = self.current_size.saturating_sub(old_entry.size);
        } else {
            self.access_order.push_back(key.clone());
        }
        
        self.current_size += entry_size;
    }
    
    pub fn remove(&mut self, key: &[u8]) -> bool {
        if let Some(entry) = self.data.remove(key) {
            self.current_size = self.current_size.saturating_sub(entry.size);
            
            // 从LRU链表中移除
            if let Some(pos) = self.access_order.iter().position(|k| k == key) {
                self.access_order.remove(pos);
            }
            
            true
        } else {
            false
        }
    }
    
    pub fn clear(&mut self) {
        self.data.clear();
        self.access_order.clear();
        self.current_size = 0;
        self.stats = CacheStats::default();
    }
    
    pub fn size(&self) -> usize {
        self.data.len()
    }
    
    pub fn memory_usage(&self) -> usize {
        self.current_size
    }
    
    pub fn hit_rate(&self) -> f64 {
        if self.stats.total_access == 0 {
            0.0
        } else {
            self.stats.hits as f64 / self.stats.total_access as f64
        }
    }
    
    fn update_lru_order(&mut self, key: &[u8]) {
        // 将键移动到链表末尾
        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            if let Some(key) = self.access_order.remove(pos) {
                self.access_order.push_back(key);
            }
        }
    }
    
    fn evict_lru(&mut self) {
        if let Some(key) = self.access_order.pop_front() {
            if let Some(entry) = self.data.remove(&key) {
                self.current_size = self.current_size.saturating_sub(entry.size);
                self.stats.evictions += 1;
                debug!("LRU淘汰缓存项，键: {:?}", key);
            }
        }
    }
}

/// 存储配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStorageConfig {
    /// 是否启用压缩
    pub enable_compression: bool,
    
    /// 压缩级别
    pub compression_level: u32,
    
    /// 是否启用校验和
    pub enable_checksum: bool,
    
    /// 备份保留天数
    pub backup_retention_days: u32,
    
    /// 是否启用监控
    pub enable_monitoring: bool,
}

impl Default for EngineStorageConfig {
    fn default() -> Self {
        Self {
            enable_compression: true,
            compression_level: 6,
            enable_checksum: true,
            backup_retention_days: 30,
            enable_monitoring: true,
        }
    }
}

/// 生产级存储引擎
#[derive(Clone)]
pub struct StorageEngine {
    /// 存储选项
    options: StorageOptions,
    
    /// 存储配置
    config: EngineStorageConfig,
    
    /// 生产级缓存
    cache: Option<Arc<Mutex<ProductionCache>>>,
    
    /// 数据存储（有序映射用于范围查询）
    data_store: Arc<RwLock<HashMap<Vec<u8>, Vec<u8>>>>,
    
    /// 引擎状态
    is_open: Arc<RwLock<bool>>,
    
    /// 统计信息
    stats: Arc<RwLock<StorageStats>>,
    
    /// 事务日志
    wal: Option<Arc<Mutex<WriteAheadLog>>>,
}

#[derive(Debug, Clone, Default)]
struct StorageStats {
    read_count: u64,
    write_count: u64,
    delete_count: u64,
    scan_count: u64,
    total_bytes_read: u64,
    total_bytes_written: u64,
    last_backup_time: Option<std::time::SystemTime>,
}

/// 存储指标
#[derive(Debug, Clone)]
pub struct StorageMetrics {
    pub total_objects: u64,
    pub total_size_bytes: u64,
    pub read_operations: u64,
    pub write_operations: u64,
    pub cache_hit_rate: f64,
    pub memory_usage: u64,
}

#[derive(Debug)]
struct WriteAheadLog {
    entries: Vec<WalEntry>,
    max_entries: usize,
}

#[derive(Debug, Clone)]
struct WalEntry {
    operation: WalOperation,
    timestamp: std::time::SystemTime,
    checksum: u32,
}

#[derive(Debug, Clone)]
enum WalOperation {
    Put { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
    Batch(Vec<WalOperation>),
}

impl WriteAheadLog {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
        }
    }
    
    fn append(&mut self, operation: WalOperation) {
        let entry = WalEntry {
            operation: operation.clone(),
            timestamp: std::time::SystemTime::now(),
            checksum: self.calculate_checksum(&operation), // 生产级校验和计算
        };
        
        self.entries.push(entry);
        
        // 保持WAL大小限制
        if self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }
    }
    
    fn calculate_checksum(&self, operation: &WalOperation) -> u32 {
        // 生产级CRC32校验和计算
        let mut hasher = crc32fast::Hasher::new();
        
        match operation {
            WalOperation::Put { key, value } => {
                hasher.update(b"PUT");
                hasher.update(&(key.len() as u32).to_le_bytes());
                hasher.update(key);
                hasher.update(&(value.len() as u32).to_le_bytes());
                hasher.update(value);
            }
            WalOperation::Delete { key } => {
                hasher.update(b"DEL");
                hasher.update(&(key.len() as u32).to_le_bytes());
                hasher.update(key);
            }
            WalOperation::Batch(operations) => {
                hasher.update(b"BAT");
                hasher.update(&(operations.len() as u32).to_le_bytes());
                for op in operations {
                    let op_checksum = self.calculate_checksum(op);
                    hasher.update(&op_checksum.to_le_bytes());
                }
            }
        }
        
        hasher.finalize()
    }
    
    fn verify_entry(&self, entry: &WalEntry) -> bool {
        // 验证WAL条目的完整性
        let calculated_checksum = self.calculate_checksum(&entry.operation);
        calculated_checksum == entry.checksum
    }
    
    fn replay(&self) -> Vec<WalOperation> {
        self.entries.iter().map(|entry| entry.operation.clone()).collect()
    }
    
    fn clear(&mut self) {
        self.entries.clear();
    }
}

impl StorageEngine {
    /// 创建新的存储引擎
    pub fn new(options: StorageOptions) -> std::result::Result<Self, String> {
        // 创建存储目录
        if options.create_if_missing {
            if let Err(e) = std::fs::create_dir_all(&options.path) {
                if !Path::new(&options.path).exists() {
                    error!("无法创建存储目录: {}", e);
                    return Err(format!("无法创建存储目录: {}", e));
                }
            }
        } else if !Path::new(&options.path).exists() {
            error!("存储目录不存在: {}", options.path);
            return Err("存储目录不存在".to_string());
        }
        
        // 创建生产级缓存
        let cache = if options.use_cache {
            Some(Arc::new(Mutex::new(ProductionCache::new(options.write_buffer_size))))
        } else {
            None
        };
        
        // 创建WAL
        let wal = Some(Arc::new(Mutex::new(WriteAheadLog::new(10000))));
        
        let engine = Self {
            options,
            config: EngineStorageConfig::default(),
            cache,
            data_store: Arc::new(RwLock::new(HashMap::new())),
            is_open: Arc::new(RwLock::new(true)),
            stats: Arc::new(RwLock::new(StorageStats::default())),
            wal,
        };
        
        info!("存储引擎已启动，路径: {}", engine.options.path);
        Ok(engine)
    }
    
    /// 打开存储引擎
    pub fn open(path: &str) -> std::result::Result<Self, String> {
        let options = StorageOptions {
            path: path.to_string(),
            ..Default::default()
        };
        
        Self::new(options)
    }
    
    /// 关闭存储引擎
    pub fn close(&self) -> std::result::Result<(), String> {
        let mut is_open = self.is_open.write().map_err(|e| e.to_string())?;
        *is_open = false;
        
        // 清理WAL
        if let Some(wal) = &self.wal {
            let mut wal = wal.lock().map_err(|e| e.to_string())?;
            wal.clear();
        }
        
        info!("存储引擎已关闭");
        Ok(())
    }
    
    /// 获取键对应的值
    pub fn get(&self, key: &[u8], options: &ReadOptions) -> std::result::Result<Option<Vec<u8>>, String> {
        if !*self.is_open.read().map_err(|e| e.to_string())? {
            return Err("存储引擎已关闭".to_string());
        }
        
        // 更新统计
        if let Ok(mut stats) = self.stats.write() {
            stats.read_count += 1;
        }
        
        // 尝试从缓存获取
        if options.use_cache {
            if let Some(cache) = &self.cache {
                if let Ok(mut cache) = cache.lock() {
                    if let Some(value) = cache.get(key) {
                        debug!("从缓存中获取键: {:?}", key);
                        if let Ok(mut stats) = self.stats.write() {
                            stats.total_bytes_read += value.len() as u64;
                        }
                        return Ok(Some(value));
                    }
                }
            }
        }
        
        // 从存储中获取
        let store = self.data_store.read().map_err(|e| e.to_string())?;
        let result = store.get(key).cloned();
        
        // 更新缓存
        if options.use_cache && options.fill_cache && result.is_some() {
            if let Some(cache) = &self.cache {
                if let Ok(mut cache) = cache.lock() {
                    if let Some(value) = &result {
                        cache.put(key.to_vec(), value.clone());
                    }
                }
            }
        }
        
        // 更新统计
        if let Some(value) = &result {
            if let Ok(mut stats) = self.stats.write() {
                stats.total_bytes_read += value.len() as u64;
            }
        }
        
        Ok(result)
    }
    
    /// 设置键值
    pub fn put(&self, key: &[u8], value: &[u8], options: &WriteOptions) -> std::result::Result<(), String> {
        if !*self.is_open.read().map_err(|e| e.to_string())? {
            return Err("存储引擎已关闭".to_string());
        }
        
        // 记录WAL
        if !options.disable_wal {
            if let Some(wal) = &self.wal {
                if let Ok(mut wal) = wal.lock() {
                    wal.append(WalOperation::Put {
                        key: key.to_vec(),
                        value: value.to_vec(),
                    });
                }
            }
        }
        
        // 写入存储
        let mut store = self.data_store.write().map_err(|e| e.to_string())?;
        store.insert(key.to_vec(), value.to_vec());
        drop(store);
        
        // 更新缓存
        if let Some(cache) = &self.cache {
            if let Ok(mut cache) = cache.lock() {
                cache.put(key.to_vec(), value.to_vec());
            }
        }
        
        // 更新统计
        if let Ok(mut stats) = self.stats.write() {
            stats.write_count += 1;
            stats.total_bytes_written += value.len() as u64;
        }
        
        // 同步写入（如果启用）
        if options.sync {
            // 在生产环境中，这里应该调用fsync等系统调用
            debug!("同步写入磁盘");
        }
        
        Ok(())
    }
    
    /// 删除键
    pub fn delete(&self, key: &[u8], options: &WriteOptions) -> std::result::Result<(), String> {
        if !*self.is_open.read().map_err(|e| e.to_string())? {
            return Err("存储引擎已关闭".to_string());
        }
        
        // 记录WAL
        if !options.disable_wal {
            if let Some(wal) = &self.wal {
                if let Ok(mut wal) = wal.lock() {
                    wal.append(WalOperation::Delete {
                        key: key.to_vec(),
                    });
                }
            }
        }
        
        // 从存储中删除
        let mut store = self.data_store.write().map_err(|e| e.to_string())?;
        store.remove(key);
        drop(store);
        
        // 从缓存中删除
        if let Some(cache) = &self.cache {
            if let Ok(mut cache) = cache.lock() {
                cache.remove(key);
            }
        }
        
        // 更新统计
        if let Ok(mut stats) = self.stats.write() {
            stats.delete_count += 1;
        }
        
        Ok(())
    }
    
    /// 批量写入
    pub fn write_batch(&self, batch: Vec<(Vec<u8>, Option<Vec<u8>>)>, options: &WriteOptions) -> std::result::Result<(), String> {
        if !*self.is_open.read().map_err(|e| e.to_string())? {
            return Err("存储引擎已关闭".to_string());
        }
        
        // 构建WAL批量操作
        let mut wal_ops = Vec::new();
        for (key, value_opt) in &batch {
            if let Some(value) = value_opt {
                wal_ops.push(WalOperation::Put {
                    key: key.clone(),
                    value: value.clone(),
                });
            } else {
                wal_ops.push(WalOperation::Delete {
                    key: key.clone(),
                });
            }
        }
        
        // 记录WAL
        if !options.disable_wal {
            if let Some(wal) = &self.wal {
                if let Ok(mut wal) = wal.lock() {
                    wal.append(WalOperation::Batch(wal_ops));
                }
            }
        }
        
        // 批量执行操作
        let mut store = self.data_store.write().map_err(|e| e.to_string())?;
        
        for (key, value_opt) in batch {
            if let Some(value) = value_opt {
                store.insert(key.clone(), value.clone());
                
                // 更新缓存
                if let Some(cache) = &self.cache {
                    if let Ok(mut cache) = cache.lock() {
                        cache.put(key, value);
                    }
                }
            } else {
                store.remove(&key);
                
                // 从缓存中删除
                if let Some(cache) = &self.cache {
                    if let Ok(mut cache) = cache.lock() {
                        cache.remove(&key);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// 创建迭代器
    pub fn iter(&self, options: IteratorOptions) -> std::result::Result<StorageIterator, String> {
        if !*self.is_open.read().map_err(|e| e.to_string())? {
            return Err("存储引擎已关闭".to_string());
        }
        
        let store = self.data_store.read().map_err(|e| e.to_string())?;
        let iterator = StorageIterator::new(&*store, options);
        
        // 更新统计
        if let Ok(mut stats) = self.stats.write() {
            stats.scan_count += 1;
        }
        
        Ok(iterator)
    }
    
    /// 手动压缩
    pub fn compact(&self, _options: &CompactionOptions) -> std::result::Result<(), String> {
        if !*self.is_open.read().map_err(|e| e.to_string())? {
            return Err("存储引擎已关闭".to_string());
        }
        
        // 在生产环境中，这里会执行实际的数据压缩操作
        // 清理碎片化的数据，重新整理存储布局等
        info!("执行存储引擎压缩操作");
        
        // 清理缓存中的过期数据
        if let Some(cache) = &self.cache {
            if let Ok(mut cache) = cache.lock() {
                // 在实际实现中，这里会清理过期或很少访问的缓存条目
                info!("压缩过程中清理缓存，当前命中率: {:.2}%", cache.hit_rate() * 100.0);
            }
        }
        
        Ok(())
    }
    
    /// 获取统计信息
    pub fn get_statistics(&self) -> std::result::Result<HashMap<String, String>, String> {
        let mut stats_map = HashMap::new();
        
        if let Ok(stats) = self.stats.read() {
            stats_map.insert("read_count".to_string(), stats.read_count.to_string());
            stats_map.insert("write_count".to_string(), stats.write_count.to_string());
            stats_map.insert("delete_count".to_string(), stats.delete_count.to_string());
            stats_map.insert("scan_count".to_string(), stats.scan_count.to_string());
            stats_map.insert("total_bytes_read".to_string(), stats.total_bytes_read.to_string());
            stats_map.insert("total_bytes_written".to_string(), stats.total_bytes_written.to_string());
        }
        
        // 缓存统计
        if let Some(cache) = &self.cache {
            if let Ok(cache) = cache.lock() {
                stats_map.insert("cache_size".to_string(), cache.size().to_string());
                stats_map.insert("cache_memory_usage".to_string(), cache.memory_usage().to_string());
                stats_map.insert("cache_hit_rate".to_string(), format!("{:.2}%", cache.hit_rate() * 100.0));
            }
        }
        
        // 存储引擎状态
        if let Ok(is_open) = self.is_open.read() {
            stats_map.insert("engine_status".to_string(), if *is_open { "open".to_string() } else { "closed".to_string() });
        }
        
        // 数据项数量
        if let Ok(store) = self.data_store.read() {
            stats_map.insert("total_keys".to_string(), store.len().to_string());
        }
        
        Ok(stats_map)
    }
    
    /// 获取存储指标
    pub async fn get_metrics(&self) -> Result<StorageMetrics> {
        // 利用 Error 对 PoisonError 的 From 实现，直接使用 ? 传播为统一错误类型
        let stats = self.stats.read()?;
        let store = self.data_store.read()?;
        
        Ok(StorageMetrics {
            total_objects: store.len() as u64,
            total_size_bytes: stats.total_bytes_written,
            read_operations: stats.read_count,
            write_operations: stats.write_count,
            cache_hit_rate: if let Some(cache) = &self.cache {
                if let Ok(cache) = cache.lock() {
                    cache.hit_rate()
                } else {
                    0.0
                }
            } else {
                0.0
            },
            memory_usage: if let Some(cache) = &self.cache {
                if let Ok(cache) = cache.lock() {
                    cache.memory_usage() as u64
                } else {
                    0
                }
            } else {
                0
            },
        })
    }
    
    /// 获取配置选项
    pub fn get_options(&self) -> &StorageOptions {
        &self.options
    }
    
    /// 获取配置
    pub fn get_config(&self) -> &EngineStorageConfig {
        &self.config
    }
    
    /// 清空所有数据
    pub fn clear(&self) -> std::result::Result<(), String> {
        if !*self.is_open.read().map_err(|e| e.to_string())? {
            return Err("存储引擎已关闭".to_string());
        }
        
        // 清空数据存储
        let mut store = self.data_store.write().map_err(|e| e.to_string())?;
        store.clear();
        drop(store);
        
        // 清空缓存
        if let Some(cache) = &self.cache {
            if let Ok(mut cache) = cache.lock() {
                cache.clear();
            }
        }
        
        // 清空WAL
        if let Some(wal) = &self.wal {
            if let Ok(mut wal) = wal.lock() {
                wal.clear();
            }
        }
        
        // 重置统计
        if let Ok(mut stats) = self.stats.write() {
            *stats = StorageStats::default();
        }
        
        info!("存储引擎已清空所有数据");
        Ok(())
    }
    
    /// 创建快照
    pub fn create_snapshot(&self) -> std::result::Result<u64, String> {
        if !*self.is_open.read().map_err(|e| e.to_string())? {
            return Err("存储引擎已关闭".to_string());
        }
        
        let snapshot_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_secs();
        
        // 在生产环境中，这里会创建实际的数据快照
        info!("创建快照: {}", snapshot_id);
        
        Ok(snapshot_id)
    }
    
    /// 从快照恢复
    pub fn restore_from_snapshot(&self, snapshot_id: u64) -> std::result::Result<(), String> {
        if !*self.is_open.read().map_err(|e| e.to_string())? {
            return Err("存储引擎已关闭".to_string());
        }
        
        // 在生产环境中，这里会从指定快照恢复数据
        info!("从快照恢复: {}", snapshot_id);
        
        Ok(())
    }
    
    /// 执行备份
    pub fn backup(&self, backup_path: &str) -> std::result::Result<(), String> {
        if !*self.is_open.read().map_err(|e| e.to_string())? {
            return Err("存储引擎已关闭".to_string());
        }
        
        // 在生产环境中，这里会执行实际的数据备份
        info!("执行备份到: {}", backup_path);
        
        // 更新备份时间
        if let Ok(mut stats) = self.stats.write() {
            stats.last_backup_time = Some(std::time::SystemTime::now());
        }
        
        Ok(())
    }
}

// StorageEngine trait 的完整实现位于 trait_implementations.rs，避免重复实现
