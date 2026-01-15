use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use log::{debug, warn};
use serde::{Deserialize, Serialize};
use async_trait::async_trait;

use crate::error::{Error, Result};
use crate::event::{Event, EventSystem, EventType};
use crate::status::{StatusTracker, StatusTrackerTrait};
use crate::cache_common::EvictionPolicy;

/// 缓存层级枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheTier {
    /// 内存缓存（最快）
    Memory,
    /// 磁盘缓存（中速）
    Disk,
    /// 分布式缓存（较慢但容量大）
    Distributed,
    /// 远程缓存（最慢，用于备份）
    Remote,
}

/// 预取统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchStats {
    /// 预取请求总数
    pub total_requests: usize,
    /// 预取命中次数
    pub hits: usize,
    /// 预取未命中次数
    pub misses: usize,
    /// 预取命中率
    pub hit_ratio: f64,
    /// 预取数据大小（字节）
    pub total_prefetched_bytes: usize,
    /// 平均预取延迟（微秒）
    pub avg_prefetch_latency_us: u64,
    /// 最后更新时间
    pub last_updated: DateTime<Utc>,
    /// 缓存层级
    pub tier: CacheTier,
    /// 预取策略
    pub strategy: PrefetchStrategy,
    /// 预取窗口大小
    pub window_size: usize,
    /// 当前预取队列长度
    pub queue_length: usize,
}

impl PrefetchStats {
    /// 创建新的预取统计对象
    pub fn new(tier: CacheTier, strategy: PrefetchStrategy, window_size: usize) -> Self {
        Self {
            total_requests: 0,
            hits: 0,
            misses: 0,
            hit_ratio: 0.0,
            total_prefetched_bytes: 0,
            avg_prefetch_latency_us: 0,
            last_updated: Utc::now(),
            tier,
            strategy,
            window_size,
            queue_length: 0,
        }
    }

    /// 记录预取请求
    pub fn record_request(&mut self, latency_us: u64, prefetched_bytes: usize) {
        self.total_requests += 1;
        self.total_prefetched_bytes += prefetched_bytes;
        self.update_avg_latency(latency_us);
        self.last_updated = Utc::now();
    }

    /// 记录预取命中
    pub fn record_hit(&mut self) {
        self.hits += 1;
        self.update_hit_ratio();
        self.last_updated = Utc::now();
    }

    /// 记录预取未命中
    pub fn record_miss(&mut self) {
        self.misses += 1;
        self.update_hit_ratio();
        self.last_updated = Utc::now();
    }

    /// 更新队列长度
    pub fn update_queue_length(&mut self, length: usize) {
        self.queue_length = length;
        self.last_updated = Utc::now();
    }

    /// 更新平均延迟
    fn update_avg_latency(&mut self, latency_us: u64) {
        if self.total_requests > 1 {
            self.avg_prefetch_latency_us = 
                ((self.avg_prefetch_latency_us * (self.total_requests - 1) as u64) + latency_us) 
                / self.total_requests as u64;
        } else {
            self.avg_prefetch_latency_us = latency_us;
        }
    }

    /// 更新命中率
    fn update_hit_ratio(&mut self) {
        let total = self.hits + self.misses;
        if total > 0 {
            self.hit_ratio = self.hits as f64 / total as f64;
        }
    }
}

/// 预取策略枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrefetchStrategy {
    /// 无预取
    None,
    /// 顺序预取
    Sequential,
    /// 随机预取
    Random,
    /// 基于访问模式的预取
    PatternBased,
    /// 自适应预取
    Adaptive,
    /// 机器学习驱动的预取
    MLDriven,
}

/// 缓存操作统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// 缓存命中次数
    pub hits: usize,
    /// 缓存未命中次数
    pub misses: usize,
    /// 写入操作次数
    pub writes: usize,
    /// 删除操作次数
    pub deletes: usize,
    /// 缓存淘汰次数
    pub evictions: usize,
    /// 平均访问时间（微秒）
    pub avg_access_time_us: u64,
    /// 命中率
    pub hit_ratio: f64,
    /// 最后统计时间
    pub last_updated: DateTime<Utc>,
    /// 缓存层级
    pub tier: CacheTier,
    /// 当前缓存条目数
    pub entries: usize,
    /// 当前缓存大小（字节）
    pub size_bytes: usize,
    /// 最大容量（字节）
    pub capacity_bytes: usize,
}

impl CacheMetrics {
    /// 创建新的缓存指标对象
    pub fn new(tier: CacheTier, capacity_bytes: usize) -> Self {
        Self {
            hits: 0,
            misses: 0,
            writes: 0,
            deletes: 0,
            evictions: 0,
            avg_access_time_us: 0,
            hit_ratio: 0.0,
            last_updated: Utc::now(),
            tier,
            entries: 0,
            size_bytes: 0,
            capacity_bytes,
        }
    }

    /// 更新命中次数
    pub fn record_hit(&mut self, access_time_us: u64) {
        self.hits += 1;
        self.update_avg_access_time(access_time_us);
        self.update_hit_ratio();
        self.last_updated = Utc::now();
    }

    /// 更新未命中次数
    pub fn record_miss(&mut self, access_time_us: u64) {
        self.misses += 1;
        self.update_avg_access_time(access_time_us);
        self.update_hit_ratio();
        self.last_updated = Utc::now();
    }

    /// 更新写入次数
    pub fn record_write(&mut self, size_bytes: usize) {
        self.writes += 1;
        self.size_bytes += size_bytes;
        self.entries += 1;
        self.last_updated = Utc::now();
    }

    /// 更新删除次数
    pub fn record_delete(&mut self, size_bytes: usize) {
        self.deletes += 1;
        self.size_bytes = self.size_bytes.saturating_sub(size_bytes);
        self.entries = self.entries.saturating_sub(1);
        self.last_updated = Utc::now();
    }

    /// 更新淘汰次数
    pub fn record_eviction(&mut self, count: usize, size_bytes: usize) {
        self.evictions += count;
        self.size_bytes = self.size_bytes.saturating_sub(size_bytes);
        self.entries = self.entries.saturating_sub(count);
        self.last_updated = Utc::now();
    }

    /// 更新平均访问时间
    fn update_avg_access_time(&mut self, access_time_us: u64) {
        let total = self.hits + self.misses;
        if total == 1 {
            self.avg_access_time_us = access_time_us;
        } else {
            self.avg_access_time_us = ((self.avg_access_time_us as u128 * (total - 1) as u128 + 
                                        access_time_us as u128) / total as u128) as u64;
        }
    }

    /// 更新命中率
    fn update_hit_ratio(&mut self) {
        let total = self.hits + self.misses;
        if total > 0 {
            self.hit_ratio = self.hits as f64 / total as f64;
        }
    }
}

/// 缓存配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// 内存缓存配置
    pub memory: TierConfig,
    /// 磁盘缓存配置
    pub disk: Option<TierConfig>,
    /// 分布式缓存配置
    pub distributed: Option<TierConfig>,
    /// 远程缓存配置
    pub remote: Option<TierConfig>,
    /// 过期清理间隔（秒）
    pub cleanup_interval_secs: u64,
    /// 是否启用后台自动淘汰
    pub enable_background_eviction: bool,
    /// 是否启用指标收集
    pub enable_metrics: bool,
    /// 是否启用缓存预热
    pub enable_warmup: bool,
    /// 是否启用自适应缓存
    pub enable_adaptive_caching: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            memory: TierConfig {
                max_size_bytes: 100 * 1024 * 1024, // 100MB
                max_items: 10000,
                ttl_secs: Some(3600),              // 1小时
                eviction_policy: EvictionPolicy::LRU,
                enabled: true,
                cache_dir: String::new(), // 内存缓存不需要目录
            },
            disk: Some(TierConfig {
                max_size_bytes: 1024 * 1024 * 1024, // 1GB
                max_items: 100000,
                ttl_secs: Some(86400),              // 1天
                eviction_policy: EvictionPolicy::LRU,
                enabled: true,
                cache_dir: "./cache/disk".to_string(), // 默认磁盘缓存目录
            }),
            distributed: None,
            remote: None,
            cleanup_interval_secs: 300,            // 5分钟
            enable_background_eviction: true,
            enable_metrics: true,
            enable_warmup: true,
            enable_adaptive_caching: false,
        }
    }
}

/// 缓存层级配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierConfig {
    /// 最大缓存大小（字节）
    pub max_size_bytes: usize,
    /// 最大条目数
    pub max_items: usize,
    /// 默认TTL（秒）
    pub ttl_secs: Option<u64>,
    /// 淘汰策略
    pub eviction_policy: EvictionPolicy,
    /// 是否启用
    pub enabled: bool,
    /// 缓存目录（仅用于磁盘缓存）
    pub cache_dir: String,
}

/// 缓存访问结果
#[derive(Debug)]
pub enum CacheResult<T> {
    /// 缓存命中
    Hit(T, CacheTier),
    /// 缓存未命中
    Miss,
}

/// 缓存管理器
pub struct CacheManager {
    /// 内存缓存
    memory_cache: MemoryCache,
    /// 磁盘缓存
    disk_cache: Option<DiskCache>,
    /// 分布式缓存
    distributed_cache: Option<Arc<dyn DistributedCache>>,
    /// 远程缓存
    remote_cache: Option<Arc<dyn RemoteCache>>,
    /// 缓存配置
    config: CacheConfig,
    /// 事件系统
    event_system: Option<Arc<dyn EventSystem>>,
    /// 状态跟踪器
    status_tracker: Option<Arc<dyn StatusTrackerTrait>>,
    /// 缓存层级指标
    metrics: HashMap<CacheTier, Arc<RwLock<CacheMetrics>>>,
    /// 最后清理时间
    last_cleanup: Mutex<Instant>,
    /// 清理锁，防止并发清理
    cleanup_lock: Mutex<()>,
    /// 预取统计信息
    prefetch_stats: Arc<RwLock<HashMap<CacheTier, PrefetchStats>>>,
}

impl CacheManager {
    /// 获取所有层级的指标
    pub fn get_all_metrics(&self) -> Result<HashMap<CacheTier, CacheMetrics>> {
        let mut result = HashMap::new();
        for (tier, metric_lock) in &self.metrics {
            let metric = metric_lock.read().map_err(|e| Error::lock(e.to_string()))?;
            result.insert(*tier, metric.clone());
        }
        Ok(result)
    }

    /// 创建新的缓存管理器
    pub fn new(config: CacheConfig) -> Result<Self> {
        // 创建内存缓存
        let memory_cache = MemoryCache::new(
            config.memory.max_size_bytes,
            config.memory.max_items,
            config.memory.ttl_secs.map(Duration::from_secs),
            config.memory.eviction_policy.clone(),
        )?;

        // 创建磁盘缓存（如果配置）
        let disk_cache = if let Some(disk_config) = &config.disk {
            if disk_config.enabled {
                let cache_dir = std::path::PathBuf::from(&disk_config.cache_dir);
                Some(DiskCache::new(
                    disk_config.max_size_bytes,
                    disk_config.max_items,
                    disk_config.ttl_secs.map(Duration::from_secs),
                    disk_config.eviction_policy.clone(),
                    cache_dir,
                )?)
            } else {
                None
            }
        } else {
            None
        };

        // 初始化指标
        let mut metrics = HashMap::new();
        metrics.insert(
            CacheTier::Memory,
            Arc::new(RwLock::new(CacheMetrics::new(
                CacheTier::Memory,
                config.memory.max_size_bytes,
            ))),
        );

        if let Some(disk_config) = &config.disk {
            if disk_config.enabled {
                metrics.insert(
                    CacheTier::Disk,
                    Arc::new(RwLock::new(CacheMetrics::new(
                        CacheTier::Disk,
                        disk_config.max_size_bytes,
                    ))),
                );
            }
        }

        if let Some(dist_config) = &config.distributed {
            if dist_config.enabled {
                metrics.insert(
                    CacheTier::Distributed,
                    Arc::new(RwLock::new(CacheMetrics::new(
                        CacheTier::Distributed,
                        dist_config.max_size_bytes,
                    ))),
                );
            }
        }

        if let Some(remote_config) = &config.remote {
            if remote_config.enabled {
                metrics.insert(
                    CacheTier::Remote,
                    Arc::new(RwLock::new(CacheMetrics::new(
                        CacheTier::Remote,
                        remote_config.max_size_bytes,
                    ))),
                );
            }
        }

        // 初始化预取统计
        let mut prefetch_stats = HashMap::new();
        prefetch_stats.insert(
            CacheTier::Memory,
            PrefetchStats::new(CacheTier::Memory, PrefetchStrategy::Adaptive, 10),
        );

        if let Some(disk_config) = &config.disk {
            if disk_config.enabled {
                prefetch_stats.insert(
                    CacheTier::Disk,
                    PrefetchStats::new(CacheTier::Disk, PrefetchStrategy::Sequential, 5),
                );
            }
        }

        if let Some(dist_config) = &config.distributed {
            if dist_config.enabled {
                prefetch_stats.insert(
                    CacheTier::Distributed,
                    PrefetchStats::new(CacheTier::Distributed, PrefetchStrategy::PatternBased, 8),
                );
            }
        }

        if let Some(remote_config) = &config.remote {
            if remote_config.enabled {
                prefetch_stats.insert(
                    CacheTier::Remote,
                    PrefetchStats::new(CacheTier::Remote, PrefetchStrategy::None, 0),
                );
            }
        }

        // 创建缓存管理器
        let cache_manager = Self {
            memory_cache,
            disk_cache,
            distributed_cache: None, // 将在后续方法中设置
            remote_cache: None,      // 将在后续方法中设置
            config,
            event_system: None,      // 将在后续方法中设置
            status_tracker: None,    // 将在后续方法中设置
            metrics,
            last_cleanup: Mutex::new(Instant::now()),
            cleanup_lock: Mutex::new(()),
            prefetch_stats: Arc::new(RwLock::new(prefetch_stats)),
        };

        Ok(cache_manager)
    }

    /// 设置事件系统
    pub fn with_event_system(mut self, event_system: Arc<dyn EventSystem>) -> Self {
        self.event_system = Some(event_system);
        self
    }

    /// 设置状态跟踪器
    pub fn with_status_tracker(mut self, status_tracker: Arc<StatusTracker>) -> Self {
        self.status_tracker = Some(status_tracker);
        self
    }

    /// 设置分布式缓存
    pub fn with_distributed_cache(mut self, cache: Arc<dyn DistributedCache>) -> Result<Self> {
        if let Some(config) = &self.config.distributed {
            if config.enabled {
                self.distributed_cache = Some(cache);
            }
        }
        Ok(self)
    }

    /// 设置远程缓存
    pub fn with_remote_cache(mut self, cache: Arc<dyn RemoteCache>) -> Result<Self> {
        if let Some(config) = &self.config.remote {
            if config.enabled {
                self.remote_cache = Some(cache);
            }
        }
        Ok(self)
    }

    /// 从缓存获取值
    pub fn get(&self, key: &str) -> Result<CacheResult<Vec<u8>>> {
        // 记录开始时间
        let start = Instant::now();

        // 首先从内存缓存获取
        match self.memory_cache.get(key) {
            Ok(Some(value)) => {
                // 更新指标
                if let Some(metrics) = self.metrics.get(&CacheTier::Memory) {
                    if let Ok(mut m) = metrics.write() {
                        m.record_hit(start.elapsed().as_micros() as u64);
                    }
                }

                // 发送事件
                self.emit_cache_event(EventType::CacheHit, key, CacheTier::Memory)?;

                return Ok(CacheResult::Hit(value, CacheTier::Memory));
            }
            Ok(None) => {
                // 内存缓存未命中
                if let Some(metrics) = self.metrics.get(&CacheTier::Memory) {
                    if let Ok(mut m) = metrics.write() {
                        m.record_miss(start.elapsed().as_micros() as u64);
                    }
                }
            }
            Err(e) => {
                warn!("从内存缓存获取值时发生错误: {}", e);
            }
        }

        // 尝试从磁盘缓存获取
        if let Some(disk_cache) = &self.disk_cache {
            match disk_cache.get(key) {
                Ok(Some(value)) => {
                    // 更新指标
                    if let Some(metrics) = self.metrics.get(&CacheTier::Disk) {
                        if let Ok(mut m) = metrics.write() {
                            m.record_hit(start.elapsed().as_micros() as u64);
                        }
                    }

                    // 发送事件
                    self.emit_cache_event(EventType::CacheHit, key, CacheTier::Disk)?;

                    // 写入内存缓存（多级缓存策略）
                    if let Err(e) = self.memory_cache.set(key, &value) {
                        warn!("将磁盘缓存的值写入内存缓存时发生错误: {}", e);
                    }

                    return Ok(CacheResult::Hit(value, CacheTier::Disk));
                }
                Ok(None) => {
                    // 磁盘缓存未命中
                    if let Some(metrics) = self.metrics.get(&CacheTier::Disk) {
                        if let Ok(mut m) = metrics.write() {
                            m.record_miss(start.elapsed().as_micros() as u64);
                        }
                    }
                }
                Err(e) => {
                    warn!("从磁盘缓存获取值时发生错误: {}", e);
                }
            }
        }

        // 尝试从分布式缓存获取
        if let Some(distributed_cache) = &self.distributed_cache {
            match distributed_cache.get(key) {
                Ok(Some(value)) => {
                    // 更新指标
                    if let Some(metrics) = self.metrics.get(&CacheTier::Distributed) {
                        if let Ok(mut m) = metrics.write() {
                            m.record_hit(start.elapsed().as_micros() as u64);
                        }
                    }

                    // 发送事件
                    self.emit_cache_event(EventType::CacheHit, key, CacheTier::Distributed)?;

                    // 写入更底层的缓存（多级缓存策略）
                    if let Err(e) = self.memory_cache.set(key, &value) {
                        warn!("将分布式缓存的值写入内存缓存时发生错误: {}", e);
                    }

                    if let Some(disk_cache) = &self.disk_cache {
                        if let Err(e) = disk_cache.set(key, &value) {
                            warn!("将分布式缓存的值写入磁盘缓存时发生错误: {}", e);
                        }
                    }

                    return Ok(CacheResult::Hit(value, CacheTier::Distributed));
                }
                Ok(None) => {
                    // 分布式缓存未命中
                    if let Some(metrics) = self.metrics.get(&CacheTier::Distributed) {
                        if let Ok(mut m) = metrics.write() {
                            m.record_miss(start.elapsed().as_micros() as u64);
                        }
                    }
                }
                Err(e) => {
                    warn!("从分布式缓存获取值时发生错误: {}", e);
                }
            }
        }

        // 尝试从远程缓存获取
        if let Some(remote_cache) = &self.remote_cache {
            match remote_cache.get(key) {
                Ok(Some(value)) => {
                    // 更新指标
                    if let Some(metrics) = self.metrics.get(&CacheTier::Remote) {
                        if let Ok(mut m) = metrics.write() {
                            m.record_hit(start.elapsed().as_micros() as u64);
                        }
                    }

                    // 发送事件
                    self.emit_cache_event(EventType::CacheHit, key, CacheTier::Remote)?;

                    // 写入更底层的缓存（多级缓存策略）
                    if let Err(e) = self.memory_cache.set(key, &value) {
                        warn!("将远程缓存的值写入内存缓存时发生错误: {}", e);
                    }

                    if let Some(disk_cache) = &self.disk_cache {
                        if let Err(e) = disk_cache.set(key, &value) {
                            warn!("将远程缓存的值写入磁盘缓存时发生错误: {}", e);
                        }
                    }

                    if let Some(distributed_cache) = &self.distributed_cache {
                        if let Err(e) = distributed_cache.set(key, &value) {
                            warn!("将远程缓存的值写入分布式缓存时发生错误: {}", e);
                        }
                    }

                    return Ok(CacheResult::Hit(value, CacheTier::Remote));
                }
                Ok(None) => {
                    // 远程缓存未命中
                    if let Some(metrics) = self.metrics.get(&CacheTier::Remote) {
                        if let Ok(mut m) = metrics.write() {
                            m.record_miss(start.elapsed().as_micros() as u64);
                        }
                    }
                }
                Err(e) => {
                    warn!("从远程缓存获取值时发生错误: {}", e);
                }
            }
        }

        // 发送未命中事件
        self.emit_cache_event(EventType::CacheMiss, key, CacheTier::Memory)?;

        // 所有缓存都未命中
        Ok(CacheResult::Miss)
    }

    /// 设置缓存值
    pub fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        // 记录开始时间
        let start = Instant::now();
        let value_size = value.len();

        // 写入所有可用缓存层
        let mut errors = Vec::new();

        // 写入内存缓存
        if let Err(e) = self.memory_cache.set(key, value) {
            errors.push(format!("内存缓存写入错误: {}", e));
        } else {
            // 更新指标
            if let Some(metrics) = self.metrics.get(&CacheTier::Memory) {
                if let Ok(mut m) = metrics.write() {
                    m.record_write(value_size);
                }
            }
        }

        // 写入磁盘缓存
        if let Some(disk_cache) = &self.disk_cache {
            if let Err(e) = disk_cache.set(key, value) {
                errors.push(format!("磁盘缓存写入错误: {}", e));
            } else {
                // 更新指标
                if let Some(metrics) = self.metrics.get(&CacheTier::Disk) {
                    if let Ok(mut m) = metrics.write() {
                        m.record_write(value_size);
                    }
                }
            }
        }

        // 写入分布式缓存
        if let Some(distributed_cache) = &self.distributed_cache {
            if let Err(e) = distributed_cache.set(key, value) {
                errors.push(format!("分布式缓存写入错误: {}", e));
            } else {
                // 更新指标
                if let Some(metrics) = self.metrics.get(&CacheTier::Distributed) {
                    if let Ok(mut m) = metrics.write() {
                        m.record_write(value_size);
                    }
                }
            }
        }

        // 写入远程缓存
        if let Some(remote_cache) = &self.remote_cache {
            if let Err(e) = remote_cache.set(key, value) {
                errors.push(format!("远程缓存写入错误: {}", e));
            } else {
                // 更新指标
                if let Some(metrics) = self.metrics.get(&CacheTier::Remote) {
                    if let Ok(mut m) = metrics.write() {
                        m.record_write(value_size);
                    }
                }
            }
        }

        // 发送缓存写入事件
        self.emit_cache_event(EventType::CacheWrite, key, CacheTier::Memory)?;

        // 检查是否需要执行自动清理
        self.check_cleanup()?;

        // 如果有错误，记录但不阻止操作
        if !errors.is_empty() {
            warn!("缓存写入过程中发生以下错误: {}", errors.join(", "));
        }

        Ok(())
    }

    /// 删除缓存值
    pub fn delete(&self, key: &str) -> Result<bool> {
        let mut deleted = false;
        let mut value_size = 0;

        // 从内存缓存中获取值大小（如果可能）
        if let Ok(Some(value)) = self.memory_cache.get(key) {
            value_size = value.len();
        }

        // 从所有缓存层删除
        // 从内存缓存删除
        if let Ok(d) = self.memory_cache.delete(key) {
            deleted |= d;
            // 更新指标
            if let Some(metrics) = self.metrics.get(&CacheTier::Memory) {
                if let Ok(mut m) = metrics.write() {
                    m.record_delete(value_size);
                }
            }
        }

        // 从磁盘缓存删除
        if let Some(disk_cache) = &self.disk_cache {
            if let Ok(d) = disk_cache.delete(key) {
                deleted |= d;
                // 更新指标
                if let Some(metrics) = self.metrics.get(&CacheTier::Disk) {
                    if let Ok(mut m) = metrics.write() {
                        m.record_delete(value_size);
                    }
                }
            }
        }

        // 从分布式缓存删除
        if let Some(distributed_cache) = &self.distributed_cache {
            if let Ok(d) = distributed_cache.delete(key) {
                deleted |= d;
                // 更新指标
                if let Some(metrics) = self.metrics.get(&CacheTier::Distributed) {
                    if let Ok(mut m) = metrics.write() {
                        m.record_delete(value_size);
                    }
                }
            }
        }

        // 从远程缓存删除
        if let Some(remote_cache) = &self.remote_cache {
            if let Ok(d) = remote_cache.delete(key) {
                deleted |= d;
                // 更新指标
                if let Some(metrics) = self.metrics.get(&CacheTier::Remote) {
                    if let Ok(mut m) = metrics.write() {
                        m.record_delete(value_size);
                    }
                }
            }
        }

        // 发送缓存删除事件
        if deleted {
            self.emit_cache_event(EventType::CacheDelete, key, CacheTier::Memory)?;
        }

        Ok(deleted)
    }

    /// 清空所有缓存
    pub fn clear(&self) -> Result<()> {
        // 清空内存缓存
        self.memory_cache.clear()?;

        // 清空磁盘缓存
        if let Some(disk_cache) = &self.disk_cache {
            disk_cache.clear()?;
        }

        // 清空分布式缓存
        if let Some(distributed_cache) = &self.distributed_cache {
            distributed_cache.clear()?;
        }

        // 清空远程缓存
        if let Some(remote_cache) = &self.remote_cache {
            remote_cache.clear()?;
        }

        // 重置所有指标
        for (tier, metrics) in &self.metrics {
            if let Ok(mut m) = metrics.write() {
                *m = CacheMetrics::new(*tier, m.capacity_bytes);
            }
        }

        // 发送缓存清空事件
        if let Some(event_system) = &self.event_system {
            let event = Event::new(
                EventType::CacheClear,
                serde_json::json!({
                    "time": Utc::now().to_rfc3339(),
                }),
            );
            if let Err(e) = event_system.publish(event) {
                warn!("发送缓存清空事件失败: {}", e);
            }
        }

        Ok(())
    }

    /// 获取缓存指标
    pub fn get_metrics(&self) -> Result<HashMap<CacheTier, CacheMetrics>> {
        let mut result = HashMap::new();
        for (tier, metrics) in &self.metrics {
            if let Ok(m) = metrics.read() {
                result.insert(*tier, m.clone());
            }
        }
        Ok(result)
    }

    /// 获取缓存配置
    pub fn get_config(&self) -> &CacheConfig {
        &self.config
    }

    /// 检查是否需要执行缓存清理
    fn check_cleanup(&self) -> Result<()> {
        // 获取互斥锁，但不阻塞
        if let Ok(lock) = self.cleanup_lock.try_lock() {
            let now = Instant::now();
            let mut last_cleanup = self.last_cleanup.lock().unwrap();
            let cleanup_interval = Duration::from_secs(self.config.cleanup_interval_secs);

            if now.duration_since(*last_cleanup) > cleanup_interval {
                // 执行清理
                let _result = self.cleanup_expired();
                *last_cleanup = now;
                drop(lock);
            }
        }

        Ok(())
    }

    /// 清理过期的缓存项
    pub fn cleanup_expired(&self) -> Result<usize> {
        let mut total_cleaned = 0;

        // 清理内存缓存
        if let Ok(cleaned) = self.memory_cache.cleanup_expired() {
            total_cleaned += cleaned;
        }

        // 清理磁盘缓存
        if let Some(disk_cache) = &self.disk_cache {
            if let Ok(cleaned) = disk_cache.cleanup_expired() {
                total_cleaned += cleaned;
            }
        }

        // 记录清理事件
        if total_cleaned > 0 {
            if let Some(event_system) = &self.event_system {
                let event = Event::new(
                    EventType::CacheCleanup,
                    serde_json::json!({
                        "cleaned_items": total_cleaned,
                        "time": Utc::now().to_rfc3339(),
                    }),
                );
                if let Err(e) = event_system.publish(event) {
                    warn!("发送缓存清理事件失败: {}", e);
                }
            }
        }

        Ok(total_cleaned)
    }

    /// 发送缓存事件
    fn emit_cache_event(&self, event_type: EventType, key: &str, tier: CacheTier) -> Result<()> {
        if let Some(event_system) = &self.event_system {
            let event = Event::new(
                event_type,
                serde_json::json!({
                    "key": key,
                    "tier": format!("{:?}", tier),
                    "time": Utc::now().to_rfc3339(),
                }),
            );
            if let Err(e) = event_system.publish(event) {
                warn!("发送缓存事件失败: {}", e);
            }
        }
        Ok(())
    }

    /// 调整缓存层级大小
    pub fn resize_tier(&mut self, tier: CacheTier, new_size_bytes: usize) -> Result<()> {
        match tier {
            CacheTier::Memory => {
                if let Ok(mut data) = self.memory_cache.data.write() {
                    self.memory_cache.max_size_bytes = new_size_bytes;
                    debug!("内存缓存大小已调整为 {} 字节", new_size_bytes);
                }
            }
            CacheTier::Disk => {
                if let Some(ref mut disk_cache) = self.disk_cache {
                    disk_cache.max_size_bytes = new_size_bytes;
                    debug!("磁盘缓存大小已调整为 {} 字节", new_size_bytes);
                }
            }
            _ => {
                warn!("暂不支持调整 {:?} 层级大小", tier);
                return Err(Error::Unsupported(format!("不支持调整 {:?} 层级大小", tier)));
            }
        }
        Ok(())
    }

    /// 获取预取统计
    pub fn get_prefetch_count(&self, tier: CacheTier) -> Result<usize> {
        if let Ok(stats) = self.prefetch_stats.read() {
            if let Some(prefetch_stat) = stats.get(&tier) {
                Ok(prefetch_stat.total_requests)
            } else {
                Ok(0)
            }
        } else {
            Err(Error::InvalidState("无法获取预取统计".to_string()))
        }
    }

    /// 更新预取统计
    pub fn update_prefetch_count(&self, tier: CacheTier, count: usize) -> Result<()> {
        if let Ok(mut stats) = self.prefetch_stats.write() {
            if let Some(prefetch_stat) = stats.get_mut(&tier) {
                // 模拟预取请求的延迟和数据大小
                let latency_us = 1000; // 1ms
                let prefetched_bytes = count * 1024; // 假设每个预取项1KB
                
                prefetch_stat.record_request(latency_us, prefetched_bytes);
                prefetch_stat.update_queue_length(count);
                
                debug!("已更新 {:?} 层级的预取统计: 请求数={}, 队列长度={}", 
                       tier, prefetch_stat.total_requests, count);
            } else {
                warn!("未找到 {:?} 层级的预取统计", tier);
            }
        } else {
            return Err(Error::InvalidState("无法更新预取统计".to_string()));
        }
        Ok(())
    }

    /// 获取预取统计详情
    pub fn get_prefetch_stats(&self, tier: CacheTier) -> Result<PrefetchStats> {
        if let Ok(stats) = self.prefetch_stats.read() {
            if let Some(prefetch_stat) = stats.get(&tier) {
                Ok(prefetch_stat.clone())
            } else {
                Err(Error::NotFound(format!("未找到 {:?} 层级的预取统计", tier)))
            }
        } else {
            Err(Error::InvalidState("无法获取预取统计".to_string()))
        }
    }

    /// 记录预取命中
    pub fn record_prefetch_hit(&self, tier: CacheTier) -> Result<()> {
        if let Ok(mut stats) = self.prefetch_stats.write() {
            if let Some(prefetch_stat) = stats.get_mut(&tier) {
                prefetch_stat.record_hit();
                debug!("记录 {:?} 层级预取命中", tier);
            } else {
                warn!("未找到 {:?} 层级的预取统计", tier);
            }
        } else {
            return Err(Error::InvalidState("无法记录预取命中".to_string()));
        }
        Ok(())
    }

    /// 记录预取未命中
    pub fn record_prefetch_miss(&self, tier: CacheTier) -> Result<()> {
        if let Ok(mut stats) = self.prefetch_stats.write() {
            if let Some(prefetch_stat) = stats.get_mut(&tier) {
                prefetch_stat.record_miss();
                debug!("记录 {:?} 层级预取未命中", tier);
            } else {
                warn!("未找到 {:?} 层级的预取统计", tier);
            }
        } else {
            return Err(Error::InvalidState("无法记录预取未命中".to_string()));
        }
        Ok(())
    }

    /// 执行预取操作
    pub fn execute_prefetch(&self, tier: CacheTier, keys: Vec<String>) -> Result<usize> {
        let start_time = Instant::now();
        let mut prefetched_count = 0;

        match tier {
            CacheTier::Memory => {
                // 内存缓存预取：直接加载到内存
                for key in &keys {
                    if let Ok(Some(_)) = self.memory_cache.get(&key) {
                        prefetched_count += 1;
                        self.record_prefetch_hit(tier)?;
                    } else {
                        self.record_prefetch_miss(tier)?;
                    }
                }
            }
            CacheTier::Disk => {
                // 磁盘缓存预取：检查文件是否存在
                if let Some(disk_cache) = &self.disk_cache {
                    for key in &keys {
                        if let Ok(Some(_)) = disk_cache.get(&key) {
                            prefetched_count += 1;
                            self.record_prefetch_hit(tier)?;
                        } else {
                            self.record_prefetch_miss(tier)?;
                        }
                    }
                }
            }
            CacheTier::Distributed => {
                // 分布式缓存预取：检查远程节点
                if let Some(dist_cache) = &self.distributed_cache {
                    for key in &keys {
                        match dist_cache.get(&key) {
                            Ok(Some(_)) => {
                                prefetched_count += 1;
                                self.record_prefetch_hit(tier)?;
                            }
                            _ => {
                                self.record_prefetch_miss(tier)?;
                            }
                        }
                    }
                }
            }
            CacheTier::Remote => {
                // 远程缓存预取：异步检查远程服务
                warn!("暂不支持远程缓存预取");
                return Err(Error::Unsupported("远程缓存预取暂未实现".to_string()));
            }
        }

        // 更新预取统计
        let latency_us = start_time.elapsed().as_micros() as u64;
        if let Ok(mut stats) = self.prefetch_stats.write() {
            if let Some(prefetch_stat) = stats.get_mut(&tier) {
                prefetch_stat.record_request(latency_us, prefetched_count * 1024);
                prefetch_stat.update_queue_length(keys.len());
            }
        }

        debug!("完成 {:?} 层级预取操作: 预取 {} 个键，耗时 {}ms", 
               tier, prefetched_count, latency_us / 1000);
        
        Ok(prefetched_count)
    }

    /// 获取所有预取统计
    pub fn get_all_prefetch_stats(&self) -> Result<HashMap<CacheTier, PrefetchStats>> {
        if let Ok(stats) = self.prefetch_stats.read() {
            Ok(stats.clone())
        } else {
            Err(Error::InvalidState("无法获取预取统计".to_string()))
        }
    }

    /// 强制淘汰指定数量的缓存项
    pub fn force_eviction(&self, tier: CacheTier, count: usize) -> Result<()> {
        match tier {
            CacheTier::Memory => {
                self.memory_cache.evict_entries(count)?;
            }
            CacheTier::Disk => {
                if let Some(ref disk_cache) = self.disk_cache {
                    disk_cache.evict_entries(count)?;
                }
            }
            _ => {
                warn!("暂不支持对 {:?} 层级进行强制淘汰", tier);
                return Err(Error::Unsupported(format!("不支持对 {:?} 层级进行强制淘汰", tier)));
            }
        }
        debug!("已对 {:?} 层级强制淘汰 {} 个缓存项", tier, count);
        Ok(())
    }

    /// 获取指定层级的缓存项数量
    pub fn get_tier_items(&self, tier: CacheTier) -> Result<usize> {
        match tier {
            CacheTier::Memory => {
                if let Ok(data) = self.memory_cache.data.read() {
                    Ok(data.len())
                } else {
                    Err(Error::InvalidState("无法获取内存缓存数据".to_string()))
                }
            }
            CacheTier::Disk => {
                if let Some(ref disk_cache) = self.disk_cache {
                    if let Ok(index) = disk_cache.index.read() {
                        Ok(index.len())
                    } else {
                        Err(Error::InvalidState("无法获取磁盘缓存索引".to_string()))
                    }
                } else {
                    Ok(0)
                }
            }
            _ => {
                warn!("暂不支持获取 {:?} 层级项数", tier);
                Ok(0)
            }
        }
    }

    /// 更新缓存项
    pub fn update_item(&self, tier: CacheTier, key: &str, value: &[u8]) -> Result<()> {
        // 直接使用set方法更新项
        match tier {
            CacheTier::Memory => {
                self.memory_cache.set(key, value)?;
            }
            CacheTier::Disk => {
                if let Some(ref disk_cache) = self.disk_cache {
                    disk_cache.set(key, value)?;
                }
            }
            _ => {
                warn!("暂不支持更新 {:?} 层级项", tier);
                return Err(Error::Unsupported(format!("不支持更新 {:?} 层级项", tier)));
            }
        }
        debug!("已更新 {:?} 层级的缓存项: {}", tier, key);
        Ok(())
    }

    /// 获取指定层级的容量
    pub fn get_tier_capacity(&self, tier: &CacheTier) -> Result<usize> {
        match tier {
            CacheTier::Memory => Ok(self.memory_cache.max_size()),
            CacheTier::Disk => {
                if let Some(ref disk_cache) = self.disk_cache {
                    Ok(disk_cache.max_size_bytes)
                } else {
                    Err(Error::InvalidState("磁盘缓存未启用".to_string()))
                }
            }
            _ => {
                warn!("暂不支持获取 {:?} 层级容量", tier);
                Err(Error::Unsupported(format!("不支持获取 {:?} 层级容量", tier)))
            }
        }
    }
}

/// 内存缓存实现
struct MemoryCache {
    /// 缓存数据存储，使用RwLock提供并发安全
    data: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// 最大内存字节数
    max_size_bytes: usize,
    /// 最大条目数量
    max_items: usize,
    /// 过期时间
    ttl: Option<Duration>,
    /// 淘汰策略
    eviction_policy: EvictionPolicy,
    /// 当前使用的字节数
    current_size: Arc<Mutex<usize>>,
    /// 访问时间记录(用于LRU)
    access_order: Arc<Mutex<Vec<String>>>,
    /// 缓存统计信息
    metrics: Arc<Mutex<CacheMetrics>>,
}

/// 缓存条目
#[derive(Debug, Clone)]
struct CacheEntry {
    /// 数据内容
    data: Vec<u8>,
    /// 创建时间
    created_at: Instant,
    /// 最后访问时间
    last_accessed: Instant,
    /// 过期时间
    expires_at: Option<Instant>,
    /// 访问次数
    access_count: usize,
}

impl CacheEntry {
    fn new(data: Vec<u8>, ttl: Option<Duration>) -> Self {
        let now = Instant::now();
        Self {
            data,
            created_at: now,
            last_accessed: now,
            expires_at: ttl.map(|ttl| now + ttl),
            access_count: 1,
        }
    }

    fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            Instant::now() > expires_at
        } else {
            false
        }
    }

    fn update_access(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

impl MemoryCache {
    /// 创建新的内存缓存
    fn new(
        max_size_bytes: usize,
        max_items: usize,
        ttl: Option<Duration>,
        eviction_policy: EvictionPolicy,
    ) -> Result<Self> {
        Ok(Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            max_size_bytes,
            max_items,
            ttl,
            eviction_policy,
            current_size: Arc::new(Mutex::new(0)),
            access_order: Arc::new(Mutex::new(Vec::new())),
            metrics: Arc::new(Mutex::new(CacheMetrics {
                hits: 0,
                misses: 0,
                writes: 0,
                deletes: 0,
                evictions: 0,
                avg_access_time_us: 0,
                hit_ratio: 0.0,
                last_updated: Utc::now(),
                tier: CacheTier::Memory,
                entries: 0,
                size_bytes: 0,
                capacity_bytes: max_size_bytes,
            })),
        })
    }

    /// 获取缓存项
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let start_time = Instant::now();
        
        let mut data = self.data.write()
            .map_err(|_| Error::internal("Failed to acquire write lock"))?;
        
        if let Some(entry) = data.get_mut(key) {
            if entry.is_expired() {
                // 记录大小，然后删除过期条目
                let entry_size = entry.data.len();
                data.remove(key);
                self.update_size_after_removal(entry_size)?;
                self.update_metrics(false, start_time)?;
                return Ok(None);
            }
            
            // 更新访问信息
            entry.update_access();
            self.update_access_order(key)?;
            
            let result = entry.data.clone();
            self.update_metrics(true, start_time)?;
            Ok(Some(result))
        } else {
            self.update_metrics(false, start_time)?;
            Ok(None)
        }
    }

    /// 设置缓存项
    fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        let entry = CacheEntry::new(value.to_vec(), self.ttl);
        let entry_size = entry.data.len();

        // 检查是否需要淘汰
        self.ensure_capacity(key, entry_size)?;

        let mut data = self.data.write()
            .map_err(|_| Error::internal("Failed to acquire write lock"))?;
        
        // 如果已存在，先减去旧条目大小
        if let Some(old_entry) = data.get(key) {
            self.update_size_after_removal(old_entry.data.len())?;
        }

        data.insert(key.to_string(), entry);
        self.update_size_after_insertion(entry_size)?;
        self.update_access_order(key)?;
        
        // 更新写入统计
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.writes += 1;
            metrics.last_updated = Utc::now();
        }

        Ok(())
    }

    /// 删除缓存项
    fn delete(&self, key: &str) -> Result<bool> {
        let mut data = self.data.write()
            .map_err(|_| Error::internal("Failed to acquire write lock"))?;
        
        if let Some(entry) = data.remove(key) {
            self.update_size_after_removal(entry.data.len())?;
            self.remove_from_access_order(key)?;
            
            // 更新删除统计
            if let Ok(mut metrics) = self.metrics.lock() {
                metrics.deletes += 1;
                metrics.last_updated = Utc::now();
            }
            
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// 清空缓存
    fn clear(&self) -> Result<()> {
        let mut data = self.data.write()
            .map_err(|_| Error::internal("Failed to acquire write lock"))?;
        
        data.clear();
        
        if let Ok(mut size) = self.current_size.lock() {
            *size = 0;
        }
        
        if let Ok(mut order) = self.access_order.lock() {
            order.clear();
        }
        
        // 重置统计信息
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.size_bytes = 0;
            metrics.last_updated = Utc::now();
        }

        Ok(())
    }

    /// 清理过期项
    fn cleanup_expired(&self) -> Result<usize> {
        let mut data = self.data.write()
            .map_err(|_| Error::internal("Failed to acquire write lock"))?;
        
        let mut expired_keys = Vec::new();
        let mut total_size_freed = 0;

        for (key, entry) in data.iter() {
            if entry.is_expired() {
                expired_keys.push(key.clone());
                total_size_freed += entry.data.len();
            }
        }

        let count = expired_keys.len();
        for key in expired_keys {
            data.remove(&key);
            self.remove_from_access_order(&key)?;
        }

        if total_size_freed > 0 {
            self.update_size_after_removal(total_size_freed)?;
        }

        Ok(count)
    }

    /// 确保有足够容量
    fn ensure_capacity(&self, new_key: &str, new_size: usize) -> Result<()> {
        let current_size = *self.current_size.lock()
            .map_err(|_| Error::internal("Failed to acquire size lock"))?;
        
        let (data_len, contains_key) = {
            let data = self.data.read()
                .map_err(|_| Error::internal("Failed to acquire read lock"))?;
            (data.len(), data.contains_key(new_key))
        };
        
        // 检查条目数量限制
        if data_len >= self.max_items && !contains_key {
            self.evict_entries(1)?;
        }

        // 检查内存大小限制
        let needed_space = new_size;

        if current_size + needed_space > self.max_size_bytes {
            let space_to_free = (current_size + needed_space) - self.max_size_bytes;
            self.evict_by_size(space_to_free)?;
        }

        Ok(())
    }

    /// 按数量淘汰条目
    fn evict_entries(&self, count: usize) -> Result<()> {
        let mut evicted = 0;
        
        while evicted < count {
            let key_to_evict = match self.eviction_policy {
                EvictionPolicy::LRU => self.find_lru_key()?,
                EvictionPolicy::LFU => self.find_lfu_key()?,
                EvictionPolicy::Random => self.find_random_key()?,
                EvictionPolicy::FIFO => self.find_fifo_key()?,
            };

            if let Some(key) = key_to_evict {
                self.delete(&key)?;
                evicted += 1;
                
                // 更新淘汰统计
                if let Ok(mut metrics) = self.metrics.lock() {
                    metrics.evictions += 1;
                }
            } else {
                break;
            }
        }

        Ok(())
    }

    /// 按大小淘汰条目
    fn evict_by_size(&self, target_size: usize) -> Result<()> {
        let mut freed_size = 0;
        
        while freed_size < target_size {
            let key_to_evict = match self.eviction_policy {
                EvictionPolicy::LRU => self.find_lru_key()?,
                EvictionPolicy::LFU => self.find_lfu_key()?,
                EvictionPolicy::Random => self.find_random_key()?,
                EvictionPolicy::FIFO => self.find_fifo_key()?,
            };

            if let Some(key) = key_to_evict {
                let data = self.data.read()
                    .map_err(|_| Error::internal("Failed to acquire read lock"))?;
                if let Some(entry) = data.get(&key) {
                    freed_size += entry.data.len();
                }
                drop(data);
                
                self.delete(&key)?;
                
                // 更新淘汰统计
                if let Ok(mut metrics) = self.metrics.lock() {
                    metrics.evictions += 1;
                }
            } else {
                break;
            }
        }

        Ok(())
    }

    /// 查找LRU条目
    fn find_lru_key(&self) -> Result<Option<String>> {
        let data = self.data.read()
            .map_err(|_| Error::internal("Failed to acquire read lock"))?;
        
        let mut oldest_key: Option<String> = None;
        let mut oldest_time = Instant::now();

        for (key, entry) in data.iter() {
            if entry.last_accessed < oldest_time {
                oldest_time = entry.last_accessed;
                oldest_key = Some(key.clone());
            }
        }

        Ok(oldest_key)
    }

    /// 查找LFU条目
    fn find_lfu_key(&self) -> Result<Option<String>> {
        let data = self.data.read()
            .map_err(|_| Error::internal("Failed to acquire read lock"))?;
        
        let mut least_used_key: Option<String> = None;
        let mut least_count = usize::MAX;

        for (key, entry) in data.iter() {
            if entry.access_count < least_count {
                least_count = entry.access_count;
                least_used_key = Some(key.clone());
            }
        }

        Ok(least_used_key)
    }

    /// 查找随机条目
    fn find_random_key(&self) -> Result<Option<String>> {
        let data = self.data.read()
            .map_err(|_| Error::internal("Failed to acquire read lock"))?;
        
        if data.is_empty() {
            return Ok(None);
        }

        // 简单的随机选择实现
        let keys: Vec<String> = data.keys().cloned().collect();
        let index = chrono::Utc::now().timestamp_millis() as usize % keys.len();
        Ok(Some(keys[index].clone()))
    }

    /// 查找FIFO条目
    fn find_fifo_key(&self) -> Result<Option<String>> {
        let data = self.data.read()
            .map_err(|_| Error::internal("Failed to acquire read lock"))?;
        
        let mut oldest_key: Option<String> = None;
        let mut oldest_time = Instant::now();

        for (key, entry) in data.iter() {
            if entry.created_at < oldest_time {
                oldest_time = entry.created_at;
                oldest_key = Some(key.clone());
            }
        }

        Ok(oldest_key)
    }

    /// 更新访问顺序
    fn update_access_order(&self, key: &str) -> Result<()> {
        if let Ok(mut order) = self.access_order.lock() {
            // 移除旧位置
            order.retain(|k| k != key);
            // 添加到末尾
            order.push(key.to_string());
        }
        Ok(())
    }

    /// 从访问顺序中移除
    fn remove_from_access_order(&self, key: &str) -> Result<()> {
        if let Ok(mut order) = self.access_order.lock() {
            order.retain(|k| k != key);
        }
        Ok(())
    }

    /// 更新大小(插入后)
    fn update_size_after_insertion(&self, size: usize) -> Result<()> {
        if let Ok(mut current_size) = self.current_size.lock() {
            *current_size += size;
        }
        
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.size_bytes = *self.current_size.lock().unwrap_or(&mut 0);
        }
        
        Ok(())
    }

    /// 更新大小(删除后)
    fn update_size_after_removal(&self, size: usize) -> Result<()> {
        if let Ok(mut current_size) = self.current_size.lock() {
            *current_size = current_size.saturating_sub(size);
        }
        
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.size_bytes = *self.current_size.lock().unwrap_or(&mut 0);
        }
        
        Ok(())
    }

    /// 更新统计信息
    fn update_metrics(&self, hit: bool, start_time: Instant) -> Result<()> {
        if let Ok(mut metrics) = self.metrics.lock() {
            if hit {
                metrics.hits += 1;
            } else {
                metrics.misses += 1;
            }
            
            let total_requests = metrics.hits + metrics.misses;
            metrics.hit_ratio = if total_requests > 0 {
                metrics.hits as f64 / total_requests as f64
            } else {
                0.0
            };
            
            let access_time_us = start_time.elapsed().as_micros() as u64;
            metrics.avg_access_time_us = (metrics.avg_access_time_us + access_time_us) / 2;
            metrics.last_updated = Utc::now();
        }
        Ok(())
    }
}

/// 磁盘缓存实现
struct DiskCache {
    /// 缓存目录路径
    cache_dir: std::path::PathBuf,
    /// 索引，存储文件路径和元数据
    index: Arc<RwLock<HashMap<String, DiskCacheEntry>>>,
    /// 最大磁盘字节数
    max_size_bytes: usize,
    /// 最大条目数量
    max_items: usize,
    /// 过期时间
    ttl: Option<Duration>,
    /// 淘汰策略
    eviction_policy: EvictionPolicy,
    /// 当前使用的字节数
    current_size: Arc<Mutex<usize>>,
    /// 缓存统计信息
    metrics: Arc<Mutex<CacheMetrics>>,
}

/// 磁盘缓存条目元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiskCacheEntry {
    /// 文件路径
    file_path: std::path::PathBuf,
    /// 数据大小
    size_bytes: usize,
    /// 创建时间
    created_at: DateTime<Utc>,
    /// 最后访问时间
    last_accessed: DateTime<Utc>,
    /// 过期时间
    expires_at: Option<DateTime<Utc>>,
    /// 访问次数
    access_count: usize,
}

impl DiskCacheEntry {
    fn new(file_path: std::path::PathBuf, size_bytes: usize, ttl: Option<Duration>) -> Self {
        let now = Utc::now();
        Self {
            file_path,
            size_bytes,
            created_at: now,
            last_accessed: now,
            expires_at: ttl.map(|ttl| now + chrono::Duration::from_std(ttl).unwrap_or_default()),
            access_count: 1,
        }
    }

    fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            Utc::now() > expires_at
        } else {
            false
        }
    }

    fn update_access(&mut self) {
        self.last_accessed = Utc::now();
        self.access_count += 1;
    }
}

impl DiskCache {
    /// 创建新的磁盘缓存
    fn new(
        max_size_bytes: usize,
        max_items: usize,
        ttl: Option<Duration>,
        eviction_policy: EvictionPolicy,
        cache_dir: std::path::PathBuf,
    ) -> Result<Self> {
        // 确保缓存目录存在
        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| Error::io_error(format!("Failed to create cache directory: {}", e)))?;

        let cache = Self {
            cache_dir,
            index: Arc::new(RwLock::new(HashMap::new())),
            max_size_bytes,
            max_items,
            ttl,
            eviction_policy,
            current_size: Arc::new(Mutex::new(0)),
            metrics: Arc::new(Mutex::new(CacheMetrics {
                hits: 0,
                misses: 0,
                writes: 0,
                deletes: 0,
                evictions: 0,
                avg_access_time_us: 0,
                hit_ratio: 0.0,
                last_updated: Utc::now(),
                tier: CacheTier::Disk,
                entries: 0,
                size_bytes: 0,
                capacity_bytes: max_size_bytes,
            })),
        };

        // 加载现有索引
        cache.load_index()?;

        Ok(cache)
    }

    /// 获取缓存项
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let start_time = Instant::now();
        
        let mut index = self.index.write()
            .map_err(|_| Error::internal("Failed to acquire write lock"))?;
        
        if let Some(entry) = index.get_mut(key) {
            if entry.is_expired() {
                // 删除过期条目和文件
                if entry.file_path.exists() {
                    let _ = std::fs::remove_file(&entry.file_path);
                }
                // 先保存entry.size_bytes，避免多次可变借用
                let size_bytes = entry.size_bytes;
                // 先移除，再处理
                index.remove(key);
                self.update_size_after_removal(size_bytes)?;
                self.update_metrics(false, start_time)?;
                return Ok(None);
            }
            
            // 读取文件内容
            let data = std::fs::read(&entry.file_path)
                .map_err(|e| Error::io_error(format!("Failed to read cache file: {}", e)))?;
            
            // 更新访问信息
            entry.update_access();
            
            self.update_metrics(true, start_time)?;
            Ok(Some(data))
        } else {
            self.update_metrics(false, start_time)?;
            Ok(None)
        }
    }

    /// 设置缓存项
    fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        let file_path = self.get_file_path(key);
        let entry_size = value.len();

        // 检查是否需要淘汰
        self.ensure_capacity(key, entry_size)?;

        // 写入文件
        std::fs::write(&file_path, value)
            .map_err(|e| Error::io_error(format!("Failed to write cache file: {}", e)))?;

        let mut index = self.index.write()
            .map_err(|_| Error::internal("Failed to acquire write lock"))?;
        
        // 如果已存在，先减去旧条目大小
        if let Some(old_entry) = index.get(key) {
            if old_entry.file_path.exists() {
                let _ = std::fs::remove_file(&old_entry.file_path);
            }
            self.update_size_after_removal(old_entry.size_bytes)?;
        }

        let entry = DiskCacheEntry::new(file_path, entry_size, self.ttl);
        index.insert(key.to_string(), entry);
        self.update_size_after_insertion(entry_size)?;
        
        // 保存索引
        self.save_index()?;
        
        // 更新写入统计
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.writes += 1;
            metrics.last_updated = Utc::now();
        }

        Ok(())
    }

    /// 删除缓存项
    fn delete(&self, key: &str) -> Result<bool> {
        let mut index = self.index.write()
            .map_err(|_| Error::internal("Failed to acquire write lock"))?;
        
        if let Some(entry) = index.remove(key) {
            // 删除文件
            if entry.file_path.exists() {
                std::fs::remove_file(&entry.file_path)
                    .map_err(|e| Error::io_error(format!("Failed to remove cache file: {}", e)))?;
            }
            
            self.update_size_after_removal(entry.size_bytes)?;
            
            // 保存索引
            drop(index);
            self.save_index()?;
            
            // 更新删除统计
            if let Ok(mut metrics) = self.metrics.lock() {
                metrics.deletes += 1;
                metrics.last_updated = Utc::now();
            }
            
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// 清空缓存
    fn clear(&self) -> Result<()> {
        let mut index = self.index.write()
            .map_err(|_| Error::internal("Failed to acquire write lock"))?;
        
        // 删除所有文件
        for entry in index.values() {
            if entry.file_path.exists() {
                let _ = std::fs::remove_file(&entry.file_path);
            }
        }
        
        index.clear();
        
        if let Ok(mut size) = self.current_size.lock() {
            *size = 0;
        }
        
        // 保存空索引
        drop(index);
        self.save_index()?;
        
        // 重置统计信息
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.size_bytes = 0;
            metrics.last_updated = Utc::now();
        }

        Ok(())
    }

    /// 清理过期项
    fn cleanup_expired(&self) -> Result<usize> {
        let mut index = self.index.write()
            .map_err(|_| Error::internal("Failed to acquire write lock"))?;
        
        let mut expired_keys = Vec::new();
        let mut total_size_freed = 0;

        for (key, entry) in index.iter() {
            if entry.is_expired() {
                expired_keys.push(key.clone());
                total_size_freed += entry.size_bytes;
            }
        }

        let count = expired_keys.len();
        for key in expired_keys {
            if let Some(entry) = index.remove(&key) {
                if entry.file_path.exists() {
                    let _ = std::fs::remove_file(&entry.file_path);
                }
            }
        }

        if total_size_freed > 0 {
            self.update_size_after_removal(total_size_freed)?;
        }

        drop(index);
        self.save_index()?;

        Ok(count)
    }

    /// 生成文件路径
    fn get_file_path(&self, key: &str) -> std::path::PathBuf {
        let hash = format!("{:x}", md5::compute(key.as_bytes()));
        self.cache_dir.join(format!("{}.cache", hash))
    }

    /// 确保有足够容量
    fn ensure_capacity(&self, new_key: &str, new_size: usize) -> Result<()> {
        let current_size = *self.current_size.lock()
            .map_err(|_| Error::internal("Failed to acquire size lock"))?;
        
        let (index_len, contains_key) = {
            let index = self.index.read()
                .map_err(|_| Error::internal("Failed to acquire read lock"))?;
            (index.len(), index.contains_key(new_key))
        };
        
        // 检查条目数量限制
        if index_len >= self.max_items && !contains_key {
            self.evict_entries(1)?;
        }

        // 检查磁盘大小限制
        if current_size + new_size > self.max_size_bytes {
            let space_to_free = (current_size + new_size) - self.max_size_bytes;
            self.evict_by_size(space_to_free)?;
        }

        Ok(())
    }

    /// 按数量淘汰条目
    fn evict_entries(&self, count: usize) -> Result<()> {
        let mut evicted = 0;
        
        while evicted < count {
            let key_to_evict = match self.eviction_policy {
                EvictionPolicy::LRU => self.find_lru_key()?,
                EvictionPolicy::LFU => self.find_lfu_key()?,
                EvictionPolicy::Random => self.find_random_key()?,
                EvictionPolicy::FIFO => self.find_fifo_key()?,
            };

            if let Some(key) = key_to_evict {
                self.delete(&key)?;
                evicted += 1;
                
                // 更新淘汰统计
                if let Ok(mut metrics) = self.metrics.lock() {
                    metrics.evictions += 1;
                }
            } else {
                break;
            }
        }

        Ok(())
    }

    /// 按大小淘汰条目
    fn evict_by_size(&self, target_size: usize) -> Result<()> {
        let mut freed_size = 0;
        
        while freed_size < target_size {
            let key_to_evict = match self.eviction_policy {
                EvictionPolicy::LRU => self.find_lru_key()?,
                EvictionPolicy::LFU => self.find_lfu_key()?,
                EvictionPolicy::Random => self.find_random_key()?,
                EvictionPolicy::FIFO => self.find_fifo_key()?,
            };

            if let Some(key) = key_to_evict {
                let index = self.index.read()
                    .map_err(|_| Error::internal("Failed to acquire read lock"))?;
                if let Some(entry) = index.get(&key) {
                    freed_size += entry.size_bytes;
                }
                drop(index);
                
                self.delete(&key)?;
                
                // 更新淘汰统计
                if let Ok(mut metrics) = self.metrics.lock() {
                    metrics.evictions += 1;
                }
            } else {
                break;
            }
        }

        Ok(())
    }

    /// 查找LRU条目
    fn find_lru_key(&self) -> Result<Option<String>> {
        let index = self.index.read()
            .map_err(|_| Error::internal("Failed to acquire read lock"))?;
        
        let mut oldest_key: Option<String> = None;
        let mut oldest_time = Utc::now();

        for (key, entry) in index.iter() {
            if entry.last_accessed < oldest_time {
                oldest_time = entry.last_accessed;
                oldest_key = Some(key.clone());
            }
        }

        Ok(oldest_key)
    }

    /// 查找LFU条目
    fn find_lfu_key(&self) -> Result<Option<String>> {
        let index = self.index.read()
            .map_err(|_| Error::internal("Failed to acquire read lock"))?;
        
        let mut least_used_key: Option<String> = None;
        let mut least_count = usize::MAX;

        for (key, entry) in index.iter() {
            if entry.access_count < least_count {
                least_count = entry.access_count;
                least_used_key = Some(key.clone());
            }
        }

        Ok(least_used_key)
    }

    /// 查找随机条目
    fn find_random_key(&self) -> Result<Option<String>> {
        let index = self.index.read()
            .map_err(|_| Error::internal("Failed to acquire read lock"))?;
        
        if index.is_empty() {
            return Ok(None);
        }

        let keys: Vec<String> = index.keys().cloned().collect();
        let random_index = chrono::Utc::now().timestamp_millis() as usize % keys.len();
        Ok(Some(keys[random_index].clone()))
    }

    /// 查找FIFO条目
    fn find_fifo_key(&self) -> Result<Option<String>> {
        let index = self.index.read()
            .map_err(|_| Error::internal("Failed to acquire read lock"))?;
        
        let mut oldest_key: Option<String> = None;
        let mut oldest_time = Utc::now();

        for (key, entry) in index.iter() {
            if entry.created_at < oldest_time {
                oldest_time = entry.created_at;
                oldest_key = Some(key.clone());
            }
        }

        Ok(oldest_key)
    }

    /// 加载索引文件
    fn load_index(&self) -> Result<()> {
        let index_path = self.cache_dir.join("index.json");
        
        if !index_path.exists() {
            return Ok(());
        }

        let index_data = std::fs::read_to_string(&index_path)
            .map_err(|e| Error::io_error(format!("Failed to read index file: {}", e)))?;
        
        let disk_index: HashMap<String, DiskCacheEntry> = serde_json::from_str(&index_data)
            .map_err(|e| Error::serialization(format!("Failed to deserialize index: {}", e)))?;

        let mut total_size = 0;
        let mut valid_entries = HashMap::new();

        // 验证索引中的文件是否存在，计算总大小
        for (key, entry) in disk_index {
            if entry.file_path.exists() {
                total_size += entry.size_bytes;
                valid_entries.insert(key, entry);
            }
        }

        // 更新索引和大小
        if let Ok(mut index) = self.index.write() {
            *index = valid_entries;
        }

        if let Ok(mut size) = self.current_size.lock() {
            *size = total_size;
        }

        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.size_bytes = total_size;
        }

        Ok(())
    }

    /// 保存索引文件
    fn save_index(&self) -> Result<()> {
        let index_path = self.cache_dir.join("index.json");
        
        let index = self.index.read()
            .map_err(|_| Error::internal("Failed to acquire read lock"))?;
        
        let index_data = serde_json::to_string_pretty(&*index)
            .map_err(|e| Error::serialization(format!("Failed to serialize index: {}", e)))?;
        
        std::fs::write(&index_path, index_data)
            .map_err(|e| Error::io_error(format!("Failed to write index file: {}", e)))?;

        Ok(())
    }

    /// 更新大小(插入后)
    fn update_size_after_insertion(&self, size: usize) -> Result<()> {
        if let Ok(mut current_size) = self.current_size.lock() {
            *current_size += size;
        }
        
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.size_bytes = *self.current_size.lock().unwrap_or(&mut 0);
        }
        
        Ok(())
    }

    /// 更新大小(删除后)
    fn update_size_after_removal(&self, size: usize) -> Result<()> {
        if let Ok(mut current_size) = self.current_size.lock() {
            *current_size = current_size.saturating_sub(size);
        }
        
        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.size_bytes = *self.current_size.lock().unwrap_or(&mut 0);
        }
        
        Ok(())
    }

    /// 更新统计信息
    fn update_metrics(&self, hit: bool, start_time: Instant) -> Result<()> {
        if let Ok(mut metrics) = self.metrics.lock() {
            if hit {
                metrics.hits += 1;
            } else {
                metrics.misses += 1;
            }
            
            let total_requests = metrics.hits + metrics.misses;
            metrics.hit_ratio = if total_requests > 0 {
                metrics.hits as f64 / total_requests as f64
            } else {
                0.0
            };
            
            let access_time_us = start_time.elapsed().as_micros() as u64;
            metrics.avg_access_time_us = (metrics.avg_access_time_us + access_time_us) / 2;
            metrics.last_updated = Utc::now();
        }
        Ok(())
    }
}

/// 分布式缓存接口
pub trait DistributedCache: Send + Sync {
    /// 获取缓存项
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    
    /// 设置缓存项
    fn set(&self, key: &str, value: &[u8]) -> Result<()>;
    
    /// 删除缓存项
    fn delete(&self, key: &str) -> Result<bool>;
    
    /// 清空缓存
    fn clear(&self) -> Result<()>;
    
    /// 获取缓存统计信息
    fn get_stats(&self) -> Result<CacheMetrics>;
}

/// 远程缓存接口
#[async_trait]
pub trait RemoteCache: Send + Sync {
    /// 获取缓存项
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    
    /// 设置缓存项
    async fn set(&self, key: &str, value: &[u8]) -> Result<()>;
    
    /// 删除缓存项
    async fn delete(&self, key: &str) -> Result<bool>;
    
    /// 清空缓存
    async fn clear(&self) -> Result<()>;
    
    /// 检查键是否存在
    async fn exists(&self, key: &str) -> Result<bool>;
    
    /// 获取缓存大小
    async fn size(&self) -> Result<usize>;
    
    /// 健康检查
    async fn health_check(&self) -> Result<bool>;
} 