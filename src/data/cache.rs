use std::sync::Arc;
use tokio::sync::RwLock as AsyncRwLock;
use lru::LruCache;
use serde::{Serialize, Deserialize};
use crate::error::{Result, ErrorContext};
use crate::event::{Event, EventType, EventSystem};
use log::{debug, info, trace, warn};

/// 缓存配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// 缓存容量
    pub capacity: usize,
    /// 是否启用缓存
    pub enabled: bool,
    /// 缓存过期时间（秒）
    pub ttl_seconds: Option<u64>,
    /// 是否启用统计
    pub enable_stats: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            capacity: 1000,
            enabled: true,
            ttl_seconds: None,
            enable_stats: true,
        }
    }
}

/// 异步缓存管理器
pub struct AsyncCacheManager<K, V>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync + std::fmt::Debug + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// 内部LRU缓存，使用AsyncRwLock保护
    cache: AsyncRwLock<LruCache<K, V>>,
    /// 缓存容量
    capacity: AsyncRwLock<usize>,
    /// 缓存命中统计
    hits: AsyncRwLock<usize>,
    /// 缓存未命中统计
    misses: AsyncRwLock<usize>,
    /// 事件系统
    event_system: Option<Arc<dyn EventSystem>>,
    /// 错误上下文
    error_context: ErrorContext,
}

impl<K, V> AsyncCacheManager<K, V>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync + std::fmt::Debug + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// 创建新的异步缓存管理器
    pub fn new(capacity: usize) -> Self {
        let lru_cache = LruCache::new(std::num::NonZeroUsize::new(capacity).unwrap());
        
        Self {
            cache: AsyncRwLock::new(lru_cache),
            capacity: AsyncRwLock::new(capacity),
            hits: AsyncRwLock::new(0),
            misses: AsyncRwLock::new(0),
            event_system: None,
            error_context: "AsyncCacheManager::init".to_string(),
        }
    }

    /// 设置事件系统
    pub fn with_event_system(mut self, event_system: Arc<dyn EventSystem>) -> Self {
        self.event_system = Some(event_system);
        self
    }

    /// 获取缓存项
    pub async fn get(&self, key: &K) -> Result<Option<V>> {
        trace!("Cache get operation for key: {:?}", key);
        
        let mut cache = self.cache.write().await;
        
        if let Some(value) = cache.get(key) {
            // 缓存命中
            let mut hits = self.hits.write().await;
            *hits += 1;
            
            debug!("Cache hit for key: {:?}", key);
            
            // 发送缓存命中事件
            if let Some(event_system) = &self.event_system {
                let event = Event::new(EventType::CacheHit, "AsyncCacheManager")
                    .with_data("key", format!("{:?}", key))
                    .with_data("timestamp", chrono::Utc::now().to_rfc3339());
                
                if let Err(e) = event_system.publish(event) {
                    warn!("Failed to publish cache hit event: {}", e);
                }
            }
            
            Ok(Some(value.clone()))
        } else {
            // 缓存未命中
            let mut misses = self.misses.write().await;
            *misses += 1;
            
            debug!("Cache miss for key: {:?}", key);
            
            // 发送缓存未命中事件
            if let Some(event_system) = &self.event_system {
                let event = Event::new(EventType::CacheMiss, "AsyncCacheManager")
                    .with_data("key", format!("{:?}", key))
                    .with_data("timestamp", chrono::Utc::now().to_rfc3339());
                
                if let Err(e) = event_system.publish(event) {
                    warn!("Failed to publish cache miss event: {}", e);
                }
            }
            
            Ok(None)
        }
    }

    /// 放入缓存项
    pub async fn put(&self, key: K, value: V) -> Result<()> {
        trace!("Cache put operation for key: {:?}", key);
        
        let mut cache = self.cache.write().await;
        
        // 检查是否有被驱逐的项
        let evicted = cache.put(key.clone(), value);
        
        if evicted.is_some() {
            debug!("Cache item evicted due to capacity limit");
            
            // 发送驱逐事件
            if let Some(event_system) = &self.event_system {
                let event = Event::new(EventType::CacheEviction, "AsyncCacheManager")
                    .with_data("key", format!("{:?}", key))
                    .with_data("timestamp", chrono::Utc::now().to_rfc3339());
                
                if let Err(e) = event_system.publish(event) {
                    warn!("Failed to publish cache eviction event: {}", e);
                }
            }
        }
        
        debug!("Cache put completed for key: {:?}", key);
        Ok(())
    }

    /// 移除缓存项
    pub async fn remove(&self, key: &K) -> Result<Option<V>> {
        trace!("Cache remove operation for key: {:?}", key);
        
        let mut cache = self.cache.write().await;
        let removed = cache.pop(key);
        
        if removed.is_some() {
            debug!("Cache item removed for key: {:?}", key);
            
            // 发送移除事件
            if let Some(event_system) = &self.event_system {
                let event = Event::new(EventType::CacheRemoval, "AsyncCacheManager")
                    .with_data("key", format!("{:?}", key))
                    .with_data("timestamp", chrono::Utc::now().to_rfc3339());
                
                if let Err(e) = event_system.publish(event) {
                    warn!("Failed to publish cache removal event: {}", e);
                }
            }
        }
        
        Ok(removed)
    }

    /// 清空缓存
    pub async fn clear(&self) -> Result<()> {
        info!("Clearing cache");
        
        let mut cache = self.cache.write().await;
        let size_before = cache.len();
        cache.clear();
        
        // 发送清空事件
        if let Some(event_system) = &self.event_system {
            let event = Event::new(EventType::CacheClear, "AsyncCacheManager")
                .with_data("size_before", size_before.to_string())
                .with_data("timestamp", chrono::Utc::now().to_rfc3339());
            
            if let Err(e) = event_system.publish(event) {
                warn!("Failed to publish cache clear event: {}", e);
            }
        }
        
        // 重置统计
        let mut hits = self.hits.write().await;
        *hits = 0;
        let mut misses = self.misses.write().await;
        *misses = 0;
        
        info!("Cache cleared, removed {} items", size_before);
        Ok(())
    }

    /// 获取缓存统计
    pub async fn get_stats(&self) -> Result<CacheStats> {
        let cache = self.cache.read().await;
        let capacity = *self.capacity.read().await;
        let hits = *self.hits.read().await;
        let misses = *self.misses.read().await;
        
        let total_requests = hits + misses;
        let hit_rate = if total_requests > 0 {
            hits as f64 / total_requests as f64
        } else {
            0.0
        };
        
        Ok(CacheStats {
            capacity,
            size: cache.len(),
            hits,
            misses,
            hit_rate,
        })
    }

    /// 调整缓存大小
    pub async fn resize(&self, new_capacity: usize) -> Result<()> {
        let old_capacity = *self.capacity.read().await;
        info!("Resizing cache from {} to {}", old_capacity, new_capacity);
        
        let mut cache = self.cache.write().await;
        
        if new_capacity < cache.len() {
            // 如果新容量小于当前项数，需要驱逐一些项
            let items_to_remove = cache.len() - new_capacity;
            warn!("Need to evict {} items due to resize", items_to_remove);
            
            // LRU会自动处理驱逐，但我们需要记录
            for _ in 0..items_to_remove {
                if cache.len() <= new_capacity {
                    break;
                }
                // 驱逐最旧的项
                cache.pop_lru();
            }
        }
        
        // 创建新缓存并保留前 new_capacity 个项
        let mut new_cache = LruCache::new(std::num::NonZeroUsize::new(new_capacity).unwrap());
        
        // 将保留的项添加到新缓存中
        for (key, value) in cache.iter() {
            if new_cache.len() >= new_capacity {
                break;
            }
            new_cache.put(key.clone(), value.clone());
        }
        
        // 替换原缓存
        *cache = new_cache;
        
        // 更新容量
        *self.capacity.write().await = new_capacity;
        
        // 发送缓存调整事件
        if let Some(event_system) = &self.event_system {
            let event = Event::new(EventType::CacheResize, "AsyncCacheManager")
                .with_data("old_capacity", old_capacity.to_string())
                .with_data("new_capacity", new_capacity.to_string())
                .with_data("timestamp", chrono::Utc::now().to_rfc3339());
            
            if let Err(e) = event_system.publish(event) {
                warn!("Failed to publish cache resize event: {}", e);
            }
        }
        
        Ok(())
    }
    
    /// 预热缓存
    pub async fn warmup<F, Fut>(&self, keys: Vec<K>, loader: F) -> Result<WarmupStats>
    where
        F: Fn(K) -> Fut + Send + Sync,
        Fut: futures::Future<Output = Result<V>> + Send,
    {
        info!("开始预热缓存，键数量: {}", keys.len());
        
        // 统计
        let mut loaded = 0;
        let mut failed = 0;
        let total = keys.len();
        
        // 处理每个键
        for key in keys {
            // 检查键是否已在缓存中
            if self.get(&key).await?.is_some() {
                debug!("键已在缓存中，跳过: {:?}", key);
                continue;
            }
            
            // 加载值
            match loader(key.clone()).await {
                Ok(value) => {
                    // 添加到缓存
                    self.put(key, value).await?;
                    loaded += 1;
                },
                Err(e) => {
                    // 加载失败
                    warn!("预热缓存失败: {:?} - {}", key, e);
                    failed += 1;
                }
            }
        }
        
        // 发送缓存预热事件
        if let Some(event_system) = &self.event_system {
            let event = Event::new(EventType::CacheWarmup, "AsyncCacheManager")
                .with_data("loaded", loaded.to_string())
                .with_data("failed", failed.to_string())
                .with_data("total", total.to_string())
                .with_data("timestamp", chrono::Utc::now().to_rfc3339());
            
            if let Err(e) = event_system.publish(event) {
                warn!("Failed to publish cache warmup event: {}", e);
            }
        }
        
        // 创建统计对象
        let stats = WarmupStats {
            loaded,
            failed,
            total,
        };
        
        info!("缓存预热完成: 成功={}, 失败={}, 总计={}", loaded, failed, total);
        
        Ok(stats)
    }
}

/// 缓存统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// 缓存容量
    pub capacity: usize,
    /// 当前缓存项数量
    pub size: usize,
    /// 缓存命中次数
    pub hits: usize,
    /// 缓存未命中次数
    pub misses: usize,
    /// 缓存命中率
    pub hit_rate: f64,
}

/// 缓存预热统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupStats {
    /// 成功加载的缓存项数量
    pub loaded: usize,
    /// 加载失败的缓存项数量
    pub failed: usize,
    /// 总共尝试加载的缓存项数量
    pub total: usize,
} 