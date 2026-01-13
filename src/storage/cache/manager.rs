// 存储系统缓存管理器
// 提供高效的数据和元数据缓存功能

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::{task, time};
use async_trait::async_trait;
use dashmap::DashMap;

use log::{debug, error, info, trace, warn};

use crate::Result;
use crate::error::Error;
use super::policy::{CachePolicy, CacheStats, DefaultPolicy};
use super::implementation::{Cache, CacheEntry, CacheResult};

/// 持久化存储接口
#[async_trait]
pub trait PersistentStore<K, V>: Send + Sync {
    async fn get(&self, key: &K) -> Result<Option<V>>;
    async fn put(&self, key: K, value: V) -> Result<()>;
    async fn remove(&self, key: &K) -> Result<()>;
    async fn clear(&self) -> Result<()>;
}

/// 缓存条目数据结构
#[derive(Debug, Clone)]
pub struct CachedItem<V> {
    pub value: V,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_accessed: chrono::DateTime<chrono::Utc>,
    pub access_count: u64,
    pub ttl: Option<u64>, // 生存时间（秒）
}

impl<V> CachedItem<V> {
    pub fn new(value: V, ttl: Option<u64>) -> Self {
        let now = chrono::Utc::now();
        Self {
            value,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            ttl,
        }
    }
    
    /// 检查条目是否已过期
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            let now = chrono::Utc::now();
            now > self.created_at + chrono::Duration::seconds(ttl as i64)
        } else {
            false
        }
    }
    
    /// 获取剩余生存时间（秒）
    pub fn remaining_ttl(&self) -> Option<i64> {
        if let Some(ttl) = self.ttl {
            let now = chrono::Utc::now();
            let expiry = self.created_at + chrono::Duration::seconds(ttl as i64);
            let remaining = expiry - now;
            Some(remaining.num_seconds().max(0))
        } else {
            None
        }
    }
    
    /// 更新访问时间和计数
    pub fn update_access(&mut self) {
        self.last_accessed = chrono::Utc::now();
        self.access_count += 1;
    }
}

/// 扩展的缓存配置
#[derive(Debug, Clone)]
pub struct ExtendedCacheConfig {
    /// 最大条目数
    pub max_entries: usize,
    /// 最大内存使用（字节）
    pub max_memory_usage: usize,
    /// 是否启用持久化
    pub enable_persistence: bool,
    /// 缓存目录
    pub cache_dir: Option<String>,
    /// 默认TTL（秒）
    pub default_ttl: Option<u64>,
    /// 清理间隔（秒）
    pub cleanup_interval_secs: u64,
}

impl Default for ExtendedCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            max_memory_usage: 256 * 1024 * 1024, // 256MB
            enable_persistence: false,
            cache_dir: None,
            default_ttl: Some(3600), // 1小时
            cleanup_interval_secs: 300, // 5分钟
        }
    }
}

/// 扩展的缓存指标
#[derive(Debug, Clone)]
pub struct ExtendedCacheMetrics {
    pub total_hits: u64,
    pub total_misses: u64,
    pub hit_ratio: f64,
    pub eviction_count: u64,
    pub memory_usage_bytes: usize,
    pub average_access_time_ms: f64,
    pub last_cleanup_time: chrono::DateTime<chrono::Utc>,
}

/// 缓存层级类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheTier {
    /// 内存缓存 (最快)
    Memory,
    /// 持久缓存 (较慢)
    Persistent,
}

/// 缓存管理器配置
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// 内存缓存最大大小 (字节)
    pub memory_max_size: usize,
    /// 内存缓存最大条目数
    pub memory_max_items: usize,
    /// 持久缓存最大大小 (字节)
    pub persistent_max_size: usize,
    /// 持久缓存最大条目数
    pub persistent_max_items: usize,
    /// 默认TTL (秒)
    pub default_ttl_seconds: Option<u64>,
    /// 缓存清理间隔 (秒)
    pub cleanup_interval_seconds: u64,
    /// 是否启用持久缓存
    pub enable_persistent: bool,
    /// 是否启用后台清理
    pub enable_background_cleanup: bool,
    /// 是否启用统计信息
    pub enable_stats: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            memory_max_size: 100 * 1024 * 1024, // 100MB
            memory_max_items: 10_000,
            persistent_max_size: 1024 * 1024 * 1024, // 1GB
            persistent_max_items: 100_000,
            default_ttl_seconds: Some(3600), // 1小时
            cleanup_interval_seconds: 300, // 5分钟
            enable_persistent: true,
            enable_background_cleanup: true,
            enable_stats: true,
        }
    }
}

/// 缓存统计信息
#[derive(Debug, Clone)]
pub struct CacheMetrics {
    /// 内存缓存统计
    pub memory: CacheStats,
    /// 持久缓存统计
    pub persistent: Option<CacheStats>,
    /// 缓存总命中次数
    pub total_hits: usize,
    /// 缓存总未命中次数
    pub total_misses: usize,
    /// 命中率
    pub hit_ratio: f64,
    /// 开始统计时间
    pub start_time: Instant,
    /// 最后更新时间
    pub last_updated: Instant,
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self {
            memory: CacheStats::default(),
            persistent: None,
            total_hits: 0,
            total_misses: 0,
            hit_ratio: 0.0,
            start_time: Instant::now(),
            last_updated: Instant::now(),
        }
    }
}

/// 通用缓存管理器
/// 
/// 管理内存和持久化缓存，提供统一接口
pub struct CacheManager<K: Clone + std::fmt::Debug, V: Clone> {
    /// 内存缓存
    memory_cache: Arc<RwLock<dyn Cache<K, V> + Send + Sync>>,
    /// 持久缓存
    persistent_cache: Option<Arc<RwLock<dyn Cache<K, V> + Send + Sync>>>,
    /// 缓存策略
    policy: Arc<Mutex<Box<dyn CachePolicy + Send + Sync>>>,
    /// 配置
    config: CacheConfig,
    /// 统计信息
    metrics: Arc<RwLock<CacheMetrics>>,
    /// 最后清理时间
    last_cleanup: Mutex<Instant>,
}

impl<K, V> CacheManager<K, V> 
where 
    K: Clone + std::fmt::Debug + std::hash::Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// 创建新的缓存管理器
    pub fn new(
        memory_cache: Arc<RwLock<dyn Cache<K, V> + Send + Sync>>,
        persistent_cache: Option<Arc<RwLock<dyn Cache<K, V> + Send + Sync>>>,
        config: CacheConfig,
    ) -> Self {
        // 创建默认策略
        let policy: Box<dyn CachePolicy + Send + Sync> = if let Some(ttl) = config.default_ttl_seconds {
            Box::new(DefaultPolicy::new(config.memory_max_size, ttl as i64))
        } else {
            Box::new(DefaultPolicy::new(config.memory_max_size, 3600))
        };

        let manager = Self {
            memory_cache,
            persistent_cache,
            policy: Arc::new(Mutex::new(policy)),
            config,
            metrics: Arc::new(RwLock::new(CacheMetrics::default())),
            last_cleanup: Mutex::new(Instant::now()),
        };

        // 启动后台清理任务
        if config.enable_background_cleanup {
            manager.start_background_cleanup();
        }

        manager
    }

    /// 设置缓存策略
    pub fn with_policy(mut self, policy: Box<dyn CachePolicy + Send + Sync>) -> Self {
        self.policy = Arc::new(Mutex::new(policy));
        self
    }

    /// 获取缓存项
    pub fn get(&self, key: &K) -> Result<Option<CacheEntry<V>>> {
        // 先检查内存缓存
        let result = {
            let cache = self.memory_cache.read().map_err(|_| {
                Error::from("无法获取内存缓存读锁")
            })?;
            
            cache.get(key)?
        };

        if let Some(entry) = result {
            // 记录命中
            if self.config.enable_stats {
                let mut policy = self.policy.lock().map_err(|_| {
                    Error::from("无法获取策略锁")
                })?;
                policy.access(&format!("{:?}", key))?;

                let mut metrics = self.metrics.write().map_err(|_| {
                    Error::from("无法获取指标写锁")
                })?;
                metrics.memory.hits += 1;
                metrics.total_hits += 1;
                self.update_hit_ratio(&mut metrics);
                metrics.last_updated = Instant::now();
            }
            
            return Ok(Some(entry));
        }

        // 如果内存缓存未命中，检查持久缓存
        if let Some(ref persistent_cache) = self.persistent_cache {
            let result = {
                let cache = persistent_cache.read().map_err(|_| {
                    Error::from("无法获取持久缓存读锁")
                })?;
                
                cache.get(key)?
            };

            if let Some(entry) = result.clone() {
                // 记录持久缓存命中
                if self.config.enable_stats {
                    let mut metrics = self.metrics.write().map_err(|_| {
                        Error::from("无法获取指标写锁")
                    })?;
                    
                    if let Some(ref mut persistent) = metrics.persistent {
                        persistent.hits += 1;
                    }
                    metrics.total_hits += 1;
                    self.update_hit_ratio(&mut metrics);
                    metrics.last_updated = Instant::now();
                }

                // 提升到内存缓存
                self.promote_to_memory(key, &entry)?;

                return Ok(Some(entry));
            }
        }

        // 记录未命中
        if self.config.enable_stats {
            let mut metrics = self.metrics.write().map_err(|_| {
                Error::from("无法获取指标写锁")
            })?;
            metrics.total_misses += 1;
            metrics.memory.misses += 1;
            if let Some(ref mut persistent) = metrics.persistent {
                persistent.misses += 1;
            }
            self.update_hit_ratio(&mut metrics);
            metrics.last_updated = Instant::now();
        }

        Ok(None)
    }

    /// 设置缓存项
    pub fn set(&self, key: &K, value: V, ttl: Option<Duration>) -> Result<()> {
        // 创建缓存条目
        let entry = CacheEntry::new(value, ttl);
        let entry_size = std::mem::size_of_val(&entry) + std::mem::size_of_val(&key);

        // 更新内存缓存
        {
            let mut cache = self.memory_cache.write().map_err(|_| {
                Error::from("无法获取内存缓存写锁")
            })?;
            
            cache.set(key.clone(), entry.clone())?;
        }

        // 更新持久缓存
        if let Some(ref persistent_cache) = self.persistent_cache {
            let mut cache = persistent_cache.write().map_err(|_| {
                Error::from("无法获取持久缓存写锁")
            })?;
            
            cache.set(key.clone(), entry.clone())?;
        }

        // 更新策略和统计信息
        if self.config.enable_stats {
            let mut policy = self.policy.lock().map_err(|_| {
                Error::from("无法获取策略锁")
            })?;
            policy.add(&format!("{:?}", key), entry_size)?;

            let mut metrics = self.metrics.write().map_err(|_| {
                Error::from("无法获取指标写锁")
            })?;
            metrics.memory.item_count += 1;
            metrics.memory.total_size += entry_size;
            if let Some(ref mut persistent) = metrics.persistent {
                persistent.item_count += 1;
                persistent.total_size += entry_size;
            }
            metrics.last_updated = Instant::now();
        }

        // 检查是否需要清理
        self.check_cleanup()?;

        Ok(())
    }

    /// 删除缓存项
    pub fn remove(&self, key: &K) -> Result<bool> {
        let mut removed = false;

        // 从内存缓存删除
        {
            let mut cache = self.memory_cache.write().map_err(|_| {
                Error::from("无法获取内存缓存写锁")
            })?;
            
            removed = cache.remove(key)?;
        }

        // 从持久缓存删除
        if let Some(ref persistent_cache) = self.persistent_cache {
            let mut cache = persistent_cache.write().map_err(|_| {
                Error::from("无法获取持久缓存写锁")
            })?;
            
            let persistent_removed = cache.remove(key)?;
            removed = removed || persistent_removed;
        }

        // 更新策略和统计信息
        if removed && self.config.enable_stats {
            let mut policy = self.policy.lock().map_err(|_| {
                Error::from("无法获取策略锁")
            })?;
            policy.remove(&format!("{:?}", key))?;

            let mut metrics = self.metrics.write().map_err(|_| {
                Error::from("无法获取指标写锁")
            })?;
            metrics.memory.item_count = metrics.memory.item_count.saturating_sub(1);
            if let Some(ref mut persistent) = metrics.persistent {
                persistent.item_count = persistent.item_count.saturating_sub(1);
            }
            metrics.last_updated = Instant::now();
        }

        Ok(removed)
    }

    /// 清空缓存
    pub fn clear(&self) -> Result<()> {
        // 清空内存缓存
        {
            let mut cache = self.memory_cache.write().map_err(|_| {
                Error::from("无法获取内存缓存写锁")
            })?;
            
            cache.clear()?;
        }

        // 清空持久缓存
        if let Some(ref persistent_cache) = self.persistent_cache {
            let mut cache = persistent_cache.write().map_err(|_| {
                Error::from("无法获取持久缓存写锁")
            })?;
            
            cache.clear()?;
        }

        // 清空策略和统计信息
        {
            let mut policy = self.policy.lock().map_err(|_| {
                Error::from("无法获取策略锁")
            })?;
            policy.clear()?;

            if self.config.enable_stats {
                let mut metrics = self.metrics.write().map_err(|_| {
                    Error::from("无法获取指标写锁")
                })?;
                metrics.memory = CacheStats::default();
                metrics.persistent = Some(CacheStats::default());
                metrics.last_updated = Instant::now();
            }
        }

        Ok(())
    }

    /// 获取统计信息
    pub fn get_metrics(&self) -> Result<CacheMetrics> {
        let metrics = self.metrics.read().map_err(|_| {
            Error::from("无法获取指标读锁")
        })?;
        
        Ok(metrics.clone())
    }

    /// 获取配置
    pub fn get_config(&self) -> &CacheConfig {
        &self.config
    }

    /// 检查是否需要清理过期项
    fn check_cleanup(&self) -> Result<()> {
        let mut last_cleanup = self.last_cleanup.lock().map_err(|_| {
            Error::from("无法获取最后清理时间锁")
        })?;

        let now = Instant::now();
        let cleanup_interval = Duration::from_secs(self.config.cleanup_interval_seconds);

        if now.duration_since(*last_cleanup) >= cleanup_interval {
            drop(last_cleanup); // 释放锁
            self.cleanup_expired()?;
            
            let mut last_cleanup = self.last_cleanup.lock().map_err(|_| {
                Error::from("无法获取最后清理时间锁")
            })?;
            *last_cleanup = now;
        }

        Ok(())
    }

    /// 清理过期缓存项
    pub fn cleanup_expired(&self) -> Result<usize> {
        let mut total_removed = 0;

        // 清理内存缓存
        {
            let mut cache = self.memory_cache.write().map_err(|_| {
                Error::from("无法获取内存缓存写锁")
            })?;
            
            let removed = cache.cleanup_expired()?;
            total_removed += removed;
        }

        // 清理持久缓存
        if let Some(ref persistent_cache) = self.persistent_cache {
            let mut cache = persistent_cache.write().map_err(|_| {
                Error::from("无法获取持久缓存写锁")
            })?;
            
            let removed = cache.cleanup_expired()?;
            total_removed += removed;
        }

        // 更新统计信息
        if total_removed > 0 && self.config.enable_stats {
            let mut metrics = self.metrics.write().map_err(|_| {
                Error::from("无法获取指标写锁")
            })?;
            metrics.memory.evictions += total_removed;
            metrics.last_updated = Instant::now();
        }

        Ok(total_removed)
    }

    /// 检查缓存空间并执行淘汰
    fn check_and_evict(&self) -> Result<()> {
        // 获取淘汰策略
        let policy = self.policy.lock().map_err(|_| {
            Error::from("无法获取策略锁")
        })?;

        // 检查内存缓存大小
        let mut memory_metrics = {
            let metrics = self.metrics.read().map_err(|_| {
                Error::from("无法获取指标读锁")
            })?;
            metrics.memory.clone()
        };

        if memory_metrics.total_size > self.config.memory_max_size || 
           memory_metrics.item_count > self.config.memory_max_items {
            
            // 获取淘汰候选项
            let count = std::cmp::min(
                memory_metrics.item_count / 10, // 淘汰10%的项
                50 // 最多淘汰50项
            );
            
            let candidates = policy.get_eviction_candidates(count)?;
            
            // 从内存缓存中淘汰
            let mut total_evicted = 0;
            let mut cache = self.memory_cache.write().map_err(|_| {
                Error::from("无法获取内存缓存写锁")
            })?;
            
            for key in candidates {
                if cache.remove_by_key(&key)? {
                    total_evicted += 1;
                }
            }
            
            // 更新统计信息
            if total_evicted > 0 && self.config.enable_stats {
                let mut metrics = self.metrics.write().map_err(|_| {
                    Error::from("无法获取指标写锁")
                })?;
                metrics.memory.evictions += total_evicted;
                metrics.memory.item_count -= total_evicted;
                metrics.last_updated = Instant::now();
            }
        }

        Ok(())
    }

    /// 从持久缓存提升到内存缓存
    fn promote_to_memory(&self, key: &K, entry: &CacheEntry<V>) -> Result<()> {
        let mut cache = self.memory_cache.write().map_err(|_| {
            Error::from("无法获取内存缓存写锁")
        })?;
        
        cache.set(key.clone(), entry.clone())?;
        
        Ok(())
    }

    /// 更新命中率
    fn update_hit_ratio(&self, metrics: &mut CacheMetrics) {
        let total = metrics.total_hits + metrics.total_misses;
        if total > 0 {
            metrics.hit_ratio = metrics.total_hits as f64 / total as f64;
        }
    }

    /// 启动后台清理任务
    fn start_background_cleanup(&self) {
        if !self.config.enable_background_cleanup {
            return;
        }
        
        let memory_cache = self.memory_cache.clone();
        let persistent_cache = self.persistent_cache.clone();
        let cleanup_interval = Duration::from_secs(self.config.cleanup_interval_seconds);
        let metrics = self.metrics.clone();
        
        task::spawn(async move {
            let mut interval = time::interval(cleanup_interval);
            
            loop {
                interval.tick().await;
                
                // 清理内存缓存
                {
                    let cache = memory_cache.read().unwrap();
                    let keys_to_remove = cache.keys_to_evict();
                    drop(cache);
                    
                    let mut cache = memory_cache.write().unwrap();
                    for key in keys_to_remove {
                        cache.remove(&key);
                    }
                }
                
                // 清理持久缓存
                if let Some(persistent) = &persistent_cache {
                    let cache = persistent.read().unwrap();
                    let keys_to_remove = cache.keys_to_evict();
                    drop(cache);
                    
                    let mut cache = persistent.write().unwrap();
                    for key in keys_to_remove {
                        cache.remove(&key);
                    }
                }
                
                // 更新清理时间
                {
                    let mut metrics = metrics.write().unwrap();
                    metrics.last_updated = Instant::now();
                }
                
                debug!("后台缓存清理完成");
            }
        });
        
        debug!("已启用后台缓存清理任务");
    }
}

/// 专门的二进制数据缓存管理器
pub struct BinaryCacheManager {
    cache: Arc<DashMap<String, CachedItem<Vec<u8>>>>,
    config: ExtendedCacheConfig,
    persistent_store: Option<Box<dyn PersistentStore<String, Vec<u8>>>>,
    metrics: Arc<RwLock<ExtendedCacheMetrics>>,
}

impl BinaryCacheManager {
    /// 创建二进制数据缓存管理器
    pub fn new(config: ExtendedCacheConfig) -> Result<Self> {
        // 创建二进制数据专用的内存缓存
        let memory_cache = Arc::new(DashMap::with_capacity(config.max_entries));
        
        // 创建持久化存储（可选）
        let persistent_store = if config.enable_persistence {
            Some(Self::create_persistent_store(&config)?)
        } else {
            None
        };
        
        // 创建缓存指标
        let metrics = Arc::new(RwLock::new(ExtendedCacheMetrics {
            total_hits: 0,
            total_misses: 0,
            hit_ratio: 0.0,
            eviction_count: 0,
            memory_usage_bytes: 0,
            average_access_time_ms: 0.0,
            last_cleanup_time: chrono::Utc::now(),
        }));
        
        let manager = Self {
            cache: memory_cache,
            config,
            persistent_store,
            metrics,
        };
        
        // 启动后台清理任务
        manager.start_background_cleanup();
        
        debug!("二进制数据缓存管理器初始化完成");
        Ok(manager)
    }
    
    /// 异步获取缓存项
    pub async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let start_time = Instant::now();
        
        // 首先检查内存缓存
        if let Some(mut cached_item) = self.cache.get_mut(key) {
            if !cached_item.is_expired() {
                cached_item.update_access();
                let value = cached_item.value.clone();
                
                // 更新命中统计
                self.update_hit_stats(start_time);
                return Ok(Some(value));
            } else {
                // 移除过期项
                self.cache.remove(key);
            }
        }
        
        // 检查持久化存储
        if let Some(store) = &self.persistent_store {
            if let Some(value) = store.get(&key.to_string()).await? {
                // 将数据重新加载到内存缓存
                let cached_item = CachedItem::new(value.clone(), self.config.default_ttl);
                self.cache.insert(key.to_string(), cached_item);
                
                // 更新命中统计
                self.update_hit_stats(start_time);
                return Ok(Some(value));
            }
        }
        
        // 更新未命中统计
        self.update_miss_stats(start_time);
        Ok(None)
    }
    
    /// 异步存储缓存项
    pub async fn put(&self, key: String, value: Vec<u8>) -> Result<()> {
        let start_time = Instant::now();
        
        // 检查内存使用限制
        if self.should_evict_before_insert(&value) {
            self.evict_lru_items().await?;
        }
        
        // 创建缓存项
        let cached_item = CachedItem::new(value.clone(), self.config.default_ttl);
        
        // 存储到内存缓存
        self.cache.insert(key.clone(), cached_item);
        
        // 存储到持久化存储
        if let Some(store) = &self.persistent_store {
            store.put(key, value).await?;
        }
        
        // 更新统计信息
        self.update_put_stats(start_time);
        
        Ok(())
    }
    
    /// 异步删除缓存项
    pub async fn remove(&self, key: &str) -> Result<()> {
        // 从内存缓存删除
        self.cache.remove(key);
        
        // 从持久化存储删除
        if let Some(store) = &self.persistent_store {
            store.remove(&key.to_string()).await?;
        }
        
        debug!("已删除缓存项: {}", key);
        Ok(())
    }
    
    /// 批量存储二进制数据
    pub async fn batch_put(&self, items: Vec<(String, Vec<u8>)>) -> Result<()> {
        let start_time = Instant::now();
        let mut total_size = 0;
        
        for (key, value) in items {
            total_size += value.len();
            self.put(key, value).await?;
        }
        
        // 更新性能指标
        let mut metrics = self.metrics.write().map_err(|_| Error::Internal("无法获取指标锁".to_string()))?;
        metrics.memory_usage_bytes += total_size;
        
        debug!("批量插入完成，总大小: {} 字节", total_size);
        Ok(())
    }
    
    /// 为二进制数据创建持久化存储
    fn create_persistent_store(config: &ExtendedCacheConfig) -> Result<Box<dyn PersistentStore<String, Vec<u8>>>> {
        use crate::storage::engine::rocksdb::RocksDBEngine;
        
        // 创建RocksDB配置
        let mut db_config = crate::storage::engine::config::EngineConfig::default();
        db_config.path = format!("{}/binary_cache", config.cache_dir.as_deref().unwrap_or("./cache"));
        db_config.max_memory_usage = config.max_memory_usage / 2; // 分配一半内存给持久化
        
        // 创建RocksDB存储引擎
        let engine = RocksDBEngine::new(&db_config)?;
        
        // 包装为持久化存储
        Ok(Box::new(RocksDBPersistentStore::new(engine)))
    }
    
    /// 检查是否需要在插入前驱逐项目
    fn should_evict_before_insert(&self, new_value: &[u8]) -> bool {
        let current_memory = self.estimate_memory_usage();
        let new_item_size = new_value.len() + 100; // 估计额外开销
        
        current_memory + new_item_size > self.config.max_memory_usage
    }
    
    /// 估计当前内存使用
    fn estimate_memory_usage(&self) -> usize {
        let mut total_size = 0;
        for entry in self.cache.iter() {
            let (key, value) = entry.pair();
            total_size += key.len() + value.value.len() + 64; // 估计结构开销
        }
        total_size
    }
    
    /// 驱逐LRU项目
    async fn evict_lru_items(&self) -> Result<()> {
        let mut items_to_evict = Vec::new();
        
        // 收集需要驱逐的项目（基于最少访问时间）
        for entry in self.cache.iter() {
            let (key, cached_item) = entry.pair();
            items_to_evict.push((key.clone(), cached_item.last_accessed));
        }
        
        // 按访问时间排序
        items_to_evict.sort_by_key(|item| item.1);
        
        // 驱逐最老的25%项目
        let evict_count = (items_to_evict.len() / 4).max(1);
        for i in 0..evict_count {
            self.cache.remove(&items_to_evict[i].0);
        }
        
        // 更新驱逐统计
        let mut metrics = self.metrics.write().map_err(|_| Error::Internal("无法获取指标锁".to_string()))?;
        metrics.eviction_count += evict_count as u64;
        
        debug!("驱逐了 {} 个缓存项", evict_count);
        Ok(())
    }
    
    /// 更新命中统计
    fn update_hit_stats(&self, start_time: Instant) {
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.total_hits += 1;
            metrics.average_access_time_ms = (metrics.average_access_time_ms + start_time.elapsed().as_millis() as f64) / 2.0;
            self.update_hit_ratio(&mut metrics);
        }
    }
    
    /// 更新未命中统计
    fn update_miss_stats(&self, start_time: Instant) {
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.total_misses += 1;
            metrics.average_access_time_ms = (metrics.average_access_time_ms + start_time.elapsed().as_millis() as f64) / 2.0;
            self.update_hit_ratio(&mut metrics);
        }
    }
    
    /// 更新存储统计
    fn update_put_stats(&self, start_time: Instant) {
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.average_access_time_ms = (metrics.average_access_time_ms + start_time.elapsed().as_millis() as f64) / 2.0;
        }
    }
    
    /// 更新命中率
    fn update_hit_ratio(&self, metrics: &mut ExtendedCacheMetrics) {
        let total = metrics.total_hits + metrics.total_misses;
        if total > 0 {
            metrics.hit_ratio = metrics.total_hits as f64 / total as f64;
        }
    }
    
    /// 启动后台清理任务
    fn start_background_cleanup(&self) {
        let cache_ref = Arc::downgrade(&self.cache);
        let cleanup_interval = Duration::from_secs(self.config.cleanup_interval_secs);
        
        task::spawn(async move {
            let mut interval = time::interval(cleanup_interval);
            
            loop {
                interval.tick().await;
                
                // 检查缓存引用是否仍然有效
                if let Some(cache) = cache_ref.upgrade() {
                    // 执行清理操作
                    let mut entries_to_remove = Vec::new();
                    
                    // 检查过期条目
                    for entry in cache.iter() {
                        let (key, cached_item) = entry.pair();
                        if cached_item.is_expired() {
                            entries_to_remove.push(key.clone());
                        }
                    }
                    
                    // 移除过期条目
                    for key in entries_to_remove {
                        cache.remove(&key);
                    }
                    
                    debug!("后台缓存清理完成");
                } else {
                    // 缓存已被销毁，退出清理任务
                    debug!("缓存已销毁，停止后台清理任务");
                    break;
                }
            }
        });
        
        debug!("已启用后台缓存清理任务");
    }
    
    /// 获取缓存统计信息
    pub fn get_metrics(&self) -> Result<ExtendedCacheMetrics> {
        let metrics = self.metrics.read().map_err(|_| Error::Internal("无法获取指标锁".to_string()))?;
        Ok(metrics.clone())
    }
}

/// RocksDB持久化存储实现
struct RocksDBPersistentStore {
    engine: crate::storage::engine::rocksdb::RocksDBEngine,
}

impl RocksDBPersistentStore {
    fn new(engine: crate::storage::engine::rocksdb::RocksDBEngine) -> Self {
        Self { engine }
    }
}

#[async_trait]
impl PersistentStore<String, Vec<u8>> for RocksDBPersistentStore {
    async fn get(&self, key: &String) -> Result<Option<Vec<u8>>> {
        match self.engine.get(key.as_bytes())? {
            Some(data) => Ok(Some(data)),
            None => Ok(None),
        }
    }
    
    async fn put(&self, key: String, value: Vec<u8>) -> Result<()> {
        self.engine.put(key.as_bytes(), &value)?;
        Ok(())
    }
    
    async fn remove(&self, key: &String) -> Result<()> {
        self.engine.delete(key.as_bytes())?;
        Ok(())
    }
    
    async fn clear(&self) -> Result<()> {
        // RocksDB批量删除实现
        let iterator = self.engine.iter();
        let mut keys_to_delete = Vec::new();
        
        for item in iterator {
            match item {
                Ok((key, _)) => keys_to_delete.push(key),
                Err(e) => return Err(Error::Storage(e)),
            }
        }
        
        for key in keys_to_delete {
            self.engine.delete(&key)?;
        }
        
        Ok(())
    }
}

/// 专门的字符串数据缓存管理器
pub struct StringCacheManager {
    cache: Arc<DashMap<String, CachedItem<String>>>,
    config: ExtendedCacheConfig,
    persistent_store: Option<Box<dyn PersistentStore<String, String>>>,
    metrics: Arc<RwLock<ExtendedCacheMetrics>>,
}

impl StringCacheManager {
    /// 创建字符串数据缓存管理器
    pub fn new(config: ExtendedCacheConfig) -> Result<Self> {
        // 创建字符串数据专用的内存缓存
        let memory_cache = Arc::new(DashMap::with_capacity(config.max_entries));
        
        // 创建持久化存储（可选）
        let persistent_store = if config.enable_persistence {
            Some(Self::create_string_persistent_store(&config)?)
        } else {
            None
        };
        
        // 创建缓存指标
        let metrics = Arc::new(RwLock::new(ExtendedCacheMetrics {
            total_hits: 0,
            total_misses: 0,
            hit_ratio: 0.0,
            eviction_count: 0,
            memory_usage_bytes: 0,
            average_access_time_ms: 0.0,
            last_cleanup_time: chrono::Utc::now(),
        }));
        
        let manager = Self {
            cache: memory_cache,
            config,
            persistent_store,
            metrics,
        };
        
        // 启动后台清理任务
        manager.start_background_cleanup();
        
        debug!("字符串数据缓存管理器初始化完成");
        Ok(manager)
    }
    
    /// 异步获取缓存项
    pub async fn get(&self, key: &str) -> Result<Option<String>> {
        let start_time = Instant::now();
        
        // 首先检查内存缓存
        if let Some(mut cached_item) = self.cache.get_mut(key) {
            if !cached_item.is_expired() {
                cached_item.update_access();
                let value = cached_item.value.clone();
                
                // 更新命中统计
                self.update_hit_stats(start_time);
                return Ok(Some(value));
            } else {
                // 移除过期项
                self.cache.remove(key);
            }
        }
        
        // 检查持久化存储
        if let Some(store) = &self.persistent_store {
            if let Some(value) = store.get(&key.to_string()).await? {
                // 将数据重新加载到内存缓存
                let cached_item = CachedItem::new(value.clone(), self.config.default_ttl);
                self.cache.insert(key.to_string(), cached_item);
                
                // 更新命中统计
                self.update_hit_stats(start_time);
                return Ok(Some(value));
            }
        }
        
        // 更新未命中统计
        self.update_miss_stats(start_time);
        Ok(None)
    }
    
    /// 异步存储缓存项
    pub async fn put(&self, key: String, value: String) -> Result<()> {
        let start_time = Instant::now();
        
        // 检查内存使用限制
        if self.should_evict_before_insert(&key, &value) {
            self.evict_lru_items().await?;
        }
        
        // 创建缓存项
        let cached_item = CachedItem::new(value.clone(), self.config.default_ttl);
        
        // 存储到内存缓存
        self.cache.insert(key.clone(), cached_item);
        
        // 存储到持久化存储
        if let Some(store) = &self.persistent_store {
            store.put(key, value).await?;
        }
        
        // 更新统计信息
        self.update_put_stats(start_time);
        
        Ok(())
    }
    
    /// 专门为字符串数据优化的模糊搜索
    pub async fn fuzzy_search(&self, pattern: &str, max_results: usize) -> Result<Vec<(String, String)>> {
        let mut results = Vec::new();
        let pattern_lower = pattern.to_lowercase();
        
        for entry in self.cache.iter() {
            let (key, cached_item) = entry.pair();
            
            // 检查是否过期
            if cached_item.is_expired() {
                continue;
            }
            
            // 模糊匹配
            if key.to_lowercase().contains(&pattern_lower) || 
               cached_item.value.to_lowercase().contains(&pattern_lower) {
                results.push((key.clone(), cached_item.value.clone()));
                
                if results.len() >= max_results {
                    break;
                }
            }
        }
        
        debug!("模糊搜索 '{}' 找到 {} 个结果", pattern, results.len());
        Ok(results)
    }
    
    /// 批量字符串操作
    pub async fn batch_put(&self, items: Vec<(String, String)>) -> Result<()> {
        let start_time = Instant::now();
        let mut total_size = 0;
        
        for (key, value) in items {
            total_size += key.len() + value.len();
            self.put(key, value).await?;
        }
        
        // 更新性能指标
        let mut metrics = self.metrics.write().map_err(|_| Error::Internal("无法获取指标锁".to_string()))?;
        metrics.memory_usage_bytes += total_size;
        
        debug!("批量插入完成，总大小: {} 字节", total_size);
        Ok(())
    }
    
    /// 为字符串数据创建持久化存储
    fn create_string_persistent_store(config: &ExtendedCacheConfig) -> Result<Box<dyn PersistentStore<String, String>>> {
        use crate::storage::engine::rocksdb::RocksDBEngine;
        
        // 创建RocksDB配置
        let mut db_config = crate::storage::engine::config::EngineConfig::default();
        db_config.path = format!("{}/string_cache", config.cache_dir.as_deref().unwrap_or("./cache"));
        db_config.max_memory_usage = config.max_memory_usage / 2;
        
        // 创建RocksDB存储引擎
        let engine = RocksDBEngine::new(&db_config)?;
        
        // 包装为持久化存储
        Ok(Box::new(StringPersistentStore::new(engine)))
    }
    
    /// 检查是否需要在插入前驱逐项目
    fn should_evict_before_insert(&self, key: &str, value: &str) -> bool {
        let current_memory = self.estimate_memory_usage();
        let new_item_size = key.len() + value.len() + 100; // 估计额外开销
        
        current_memory + new_item_size > self.config.max_memory_usage
    }
    
    /// 估计当前内存使用
    fn estimate_memory_usage(&self) -> usize {
        let mut total_size = 0;
        for entry in self.cache.iter() {
            let (key, cached_item) = entry.pair();
            total_size += key.len() + cached_item.value.len() + 64; // 估计结构开销
        }
        total_size
    }
    
    /// 驱逐LRU项目
    async fn evict_lru_items(&self) -> Result<()> {
        let mut items_to_evict = Vec::new();
        
        // 收集需要驱逐的项目（基于最少访问时间）
        for entry in self.cache.iter() {
            let (key, cached_item) = entry.pair();
            items_to_evict.push((key.clone(), cached_item.last_accessed));
        }
        
        // 按访问时间排序
        items_to_evict.sort_by_key(|item| item.1);
        
        // 驱逐最老的25%项目
        let evict_count = (items_to_evict.len() / 4).max(1);
        for i in 0..evict_count {
            self.cache.remove(&items_to_evict[i].0);
        }
        
        // 更新驱逐统计
        let mut metrics = self.metrics.write().map_err(|_| Error::Internal("无法获取指标锁".to_string()))?;
        metrics.eviction_count += evict_count as u64;
        
        debug!("驱逐了 {} 个缓存项", evict_count);
        Ok(())
    }
    
    /// 更新命中统计
    fn update_hit_stats(&self, start_time: Instant) {
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.total_hits += 1;
            metrics.average_access_time_ms = (metrics.average_access_time_ms + start_time.elapsed().as_millis() as f64) / 2.0;
            self.update_hit_ratio(&mut metrics);
        }
    }
    
    /// 更新未命中统计
    fn update_miss_stats(&self, start_time: Instant) {
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.total_misses += 1;
            metrics.average_access_time_ms = (metrics.average_access_time_ms + start_time.elapsed().as_millis() as f64) / 2.0;
            self.update_hit_ratio(&mut metrics);
        }
    }
    
    /// 更新存储统计
    fn update_put_stats(&self, start_time: Instant) {
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.average_access_time_ms = (metrics.average_access_time_ms + start_time.elapsed().as_millis() as f64) / 2.0;
        }
    }
    
    /// 更新命中率
    fn update_hit_ratio(&self, metrics: &mut ExtendedCacheMetrics) {
        let total = metrics.total_hits + metrics.total_misses;
        if total > 0 {
            metrics.hit_ratio = metrics.total_hits as f64 / total as f64;
        }
    }
    
    /// 启动后台清理任务
    fn start_background_cleanup(&self) {
        let cache_ref = Arc::downgrade(&self.cache);
        let cleanup_interval = Duration::from_secs(self.config.cleanup_interval_secs);
        
        task::spawn(async move {
            let mut interval = time::interval(cleanup_interval);
            
            loop {
                interval.tick().await;
                
                // 检查缓存引用是否仍然有效
                if let Some(cache) = cache_ref.upgrade() {
                    // 执行清理操作
                    let mut entries_to_remove = Vec::new();
                    
                    // 检查过期条目
                    for entry in cache.iter() {
                        let (key, cached_item) = entry.pair();
                        if cached_item.is_expired() {
                            entries_to_remove.push(key.clone());
                        }
                    }
                    
                    // 移除过期条目
                    for key in entries_to_remove {
                        cache.remove(&key);
                    }
                    
                    debug!("后台缓存清理完成");
                } else {
                    // 缓存已被销毁，退出清理任务
                    debug!("缓存已销毁，停止后台清理任务");
                    break;
                }
            }
        });
        
        debug!("已启用后台缓存清理任务");
    }
    
    /// 获取缓存统计信息
    pub fn get_metrics(&self) -> Result<ExtendedCacheMetrics> {
        let metrics = self.metrics.read().map_err(|_| Error::Internal("无法获取指标锁".to_string()))?;
        Ok(metrics.clone())
    }
}

/// 字符串持久化存储实现
struct StringPersistentStore {
    engine: crate::storage::engine::rocksdb::RocksDBEngine,
}

impl StringPersistentStore {
    fn new(engine: crate::storage::engine::rocksdb::RocksDBEngine) -> Self {
        Self { engine }
    }
}

#[async_trait]
impl PersistentStore<String, String> for StringPersistentStore {
    async fn get(&self, key: &String) -> Result<Option<String>> {
        match self.engine.get(key.as_bytes())? {
            Some(data) => {
                let value = String::from_utf8(data)
                    .map_err(|e| Error::Internal(format!("字符串解码失败: {}", e)))?;
                Ok(Some(value))
            },
            None => Ok(None),
        }
    }
    
    async fn put(&self, key: String, value: String) -> Result<()> {
        self.engine.put(key.as_bytes(), value.as_bytes())?;
        Ok(())
    }
    
    async fn remove(&self, key: &String) -> Result<()> {
        self.engine.delete(key.as_bytes())?;
        Ok(())
    }
    
    async fn clear(&self) -> Result<()> {
        // RocksDB批量删除实现
        let iterator = self.engine.iter();
        let mut keys_to_delete = Vec::new();
        
        for item in iterator {
            match item {
                Ok((key, _)) => keys_to_delete.push(key),
                Err(e) => return Err(Error::Storage(e)),
            }
        }
        
        for key in keys_to_delete {
            self.engine.delete(&key)?;
        }
        
        Ok(())
    }
} 