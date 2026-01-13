//! 向量数据库缓存系统模块
//!
//! 提供专门针对向量数据优化的多层次缓存机制。
//! 
//! ## 功能特性
//! 
//! - 多层缓存（内存、磁盘、分布式）
//! - 向量专用缓存优化
//! - 多种淘汰策略（LRU、FIFO等）
//! - 缓存预热和自适应优化
//! - 详细的缓存统计

pub mod backend;
pub mod disk;
pub mod distributed;
pub mod memory;
pub mod manager;
pub mod optimizer;
pub mod remote;
pub mod tiered;
pub mod vector_cache;

use std::sync::Arc;
use std::time::Duration;

pub use crate::cache_common::{EvictionPolicy, ExpirationPolicy};
pub use manager::{CacheManager, CacheTier, CacheConfig, TierConfig, CacheMetrics, 
                DistributedCache, RemoteCache, CacheResult};
pub use memory::MemoryCache;
pub use disk::DiskCache;
pub use distributed::{RedisCache, RedisConfig, MemcachedCache, MemcachedConfig, ClusteredCache, ConsistentHashRing};
pub use optimizer::{
    CacheOptimizer, OptimizationStrategy, AccessPattern, WarmupStrategy, AdaptiveStrategy,
    EvictionOptimization, PrefetchStrategy, MemoryManagement, AccessPatternAnalyzer,
};

// 向量专用缓存
pub use vector_cache::{
    VectorCache,
    VectorCacheEntry,
    VectorCacheConfig,
    VectorCacheStats,
};

use crate::error::Result;

/// 创建默认缓存管理器，使用内存和磁盘两级缓存
pub fn create_default_cache_manager() -> Result<CacheManager> {
    let config = CacheConfig::default();
    CacheManager::new(config)
}

/// 创建内存缓存管理器，只使用内存缓存
pub fn create_memory_cache_manager() -> Result<CacheManager> {
    let mut config = CacheConfig::default();
    config.disk = None;
    config.distributed = None;
    config.remote = None;
    
    CacheManager::new(config)
}

/// 创建带有Redis分布式缓存的缓存管理器
pub fn create_redis_cache_manager(redis_config: RedisConfig) -> Result<CacheManager> {
    // 创建基本配置
    let mut config = CacheConfig::default();
    
    // 创建Redis缓存
    let redis_cache = Arc::new(RedisCache::new(redis_config)?);
    
    // 创建缓存管理器
    let cache_manager = CacheManager::new(config)?
        .with_distributed_cache(redis_cache)?;
    
    Ok(cache_manager)
}

/// 创建带有集群分布式缓存的缓存管理器
pub fn create_clustered_cache_manager() -> Result<CacheManager> {
    // 创建基本配置
    let mut config = CacheConfig::default();
    
    // 创建多个Redis实例作为后端
    let redis_config1 = RedisConfig {
        host: "redis1.example.com".to_string(),
        port: 6379,
        ..RedisConfig::default()
    };
    
    let redis_config2 = RedisConfig {
        host: "redis2.example.com".to_string(),
        port: 6379,
        ..RedisConfig::default()
    };
    
    // 这里使用占位实现，避免实际创建Redis连接
    // 实际项目中应该使用真正的Redis客户端
    let backend1 = Arc::new(MemcachedCache::new(MemcachedConfig::default())?);
    let backend2 = Arc::new(MemcachedCache::new(MemcachedConfig::default())?);
    
    // 创建集群缓存
    let clustered_cache = Arc::new(ClusteredCache::new(vec![backend1, backend2])?);
    
    // 创建缓存管理器
    let cache_manager = CacheManager::new(config)?
        .with_distributed_cache(clustered_cache)?;
    
    Ok(cache_manager)
}

/// 缓存选项构建器，用于灵活创建缓存管理器
pub struct CacheBuilder {
    /// 缓存配置
    config: CacheConfig,
    /// Redis配置（如果使用）
    redis_config: Option<RedisConfig>,
    /// Memcached配置（如果使用）
    memcached_config: Option<MemcachedConfig>,
    /// 自定义分布式缓存（如果使用）
    custom_distributed_cache: Option<Arc<dyn DistributedCache>>,
    /// 自定义远程缓存（如果使用）
    custom_remote_cache: Option<Arc<dyn RemoteCache>>,
}

impl CacheBuilder {
    /// 创建新的缓存构建器
    pub fn new() -> Self {
        Self {
            config: CacheConfig::default(),
            redis_config: None,
            memcached_config: None,
            custom_distributed_cache: None,
            custom_remote_cache: None,
        }
    }
    
    /// 设置内存缓存大小
    pub fn with_memory_size(mut self, size_bytes: usize) -> Self {
        self.config.memory.max_size_bytes = size_bytes;
        self
    }
    
    /// 设置内存缓存最大项数
    pub fn with_memory_items(mut self, max_items: usize) -> Self {
        self.config.memory.max_items = max_items;
        self
    }
    
    /// 设置内存缓存TTL
    pub fn with_memory_ttl(mut self, ttl_secs: u64) -> Self {
        self.config.memory.ttl_secs = Some(ttl_secs);
        self
    }
    
    /// 设置内存缓存淘汰策略
    pub fn with_memory_eviction_policy(mut self, policy: EvictionPolicy) -> Self {
        self.config.memory.eviction_policy = policy;
        self
    }
    
    /// 启用或禁用磁盘缓存
    pub fn with_disk_cache(mut self, enabled: bool) -> Self {
        if enabled {
            if self.config.disk.is_none() {
                self.config.disk = Some(TierConfig {
                    max_size_bytes: 1024 * 1024 * 1024, // 1GB
                    max_items: 100000,
                    ttl_secs: Some(86400), // 1天
                    eviction_policy: EvictionPolicy::LRU,
                    enabled: true,
                    cache_dir: "./cache/disk".to_string(),
                });
            } else if let Some(ref mut disk) = self.config.disk {
                disk.enabled = true;
            }
        } else {
            self.config.disk = None;
        }
        self
    }
    
    /// 设置磁盘缓存大小
    pub fn with_disk_size(mut self, size_bytes: usize) -> Self {
        if self.config.disk.is_none() {
            self.config.disk = Some(TierConfig {
                max_size_bytes: size_bytes,
                max_items: 100000,
                ttl_secs: Some(86400), // 1天
                eviction_policy: EvictionPolicy::LRU,
                enabled: true,
                cache_dir: "./cache/disk".to_string(),
            });
        } else if let Some(ref mut disk) = self.config.disk {
            disk.max_size_bytes = size_bytes;
        }
        self
    }
    
    /// 设置磁盘缓存TTL
    pub fn with_disk_ttl(mut self, ttl_secs: u64) -> Self {
        if let Some(ref mut disk) = self.config.disk {
            disk.ttl_secs = Some(ttl_secs);
        }
        self
    }
    
    /// 设置磁盘缓存淘汰策略
    pub fn with_disk_eviction_policy(mut self, policy: EvictionPolicy) -> Self {
        if let Some(ref mut disk) = self.config.disk {
            disk.eviction_policy = policy;
        }
        self
    }
    
    /// 使用Redis作为分布式缓存
    pub fn with_redis(mut self, redis_config: RedisConfig) -> Self {
        if self.config.distributed.is_none() {
            self.config.distributed = Some(TierConfig {
                max_size_bytes: usize::MAX, // Redis大小由服务器决定
                max_items: usize::MAX,
                ttl_secs: redis_config.default_ttl_secs,
                eviction_policy: EvictionPolicy::LRU,
                enabled: true,
                cache_dir: "./cache/redis".to_string(),
            });
        } else if let Some(ref mut distributed) = self.config.distributed {
            distributed.enabled = true;
        }
        
        self.redis_config = Some(redis_config);
        self
    }
    
    /// 使用Memcached作为分布式缓存
    pub fn with_memcached(mut self, memcached_config: MemcachedConfig) -> Self {
        if self.config.distributed.is_none() {
            self.config.distributed = Some(TierConfig {
                max_size_bytes: usize::MAX, // Memcached大小由服务器决定
                max_items: usize::MAX,
                ttl_secs: memcached_config.default_ttl_secs,
                eviction_policy: EvictionPolicy::LRU,
                enabled: true,
                cache_dir: "./cache/memcached".to_string(),
            });
        } else if let Some(ref mut distributed) = self.config.distributed {
            distributed.enabled = true;
        }
        
        self.memcached_config = Some(memcached_config);
        self
    }
    
    /// 使用自定义分布式缓存
    pub fn with_custom_distributed_cache(mut self, cache: Arc<dyn DistributedCache>) -> Self {
        self.custom_distributed_cache = Some(cache);
        
        if self.config.distributed.is_none() {
            self.config.distributed = Some(TierConfig {
                max_size_bytes: usize::MAX,
                max_items: usize::MAX,
                ttl_secs: Some(3600), // 1小时
                eviction_policy: EvictionPolicy::LRU,
                enabled: true,
                cache_dir: "./cache/distributed".to_string(),
            });
        } else if let Some(ref mut distributed) = self.config.distributed {
            distributed.enabled = true;
        }
        
        self
    }
    
    /// 使用自定义远程缓存
    pub fn with_custom_remote_cache(mut self, cache: Arc<dyn RemoteCache>) -> Self {
        self.custom_remote_cache = Some(cache);
        
        if self.config.remote.is_none() {
            self.config.remote = Some(TierConfig {
                max_size_bytes: usize::MAX,
                max_items: usize::MAX,
                ttl_secs: Some(86400), // 1天
                eviction_policy: EvictionPolicy::LRU,
                enabled: true,
                cache_dir: "./cache/remote".to_string(),
            });
        } else if let Some(ref mut remote) = self.config.remote {
            remote.enabled = true;
        }
        
        self
    }
    
    /// 设置清理间隔
    pub fn with_cleanup_interval(mut self, interval_secs: u64) -> Self {
        self.config.cleanup_interval_secs = interval_secs;
        self
    }
    
    /// 启用或禁用后台淘汰
    pub fn with_background_eviction(mut self, enabled: bool) -> Self {
        self.config.enable_background_eviction = enabled;
        self
    }
    
    /// 启用或禁用指标收集
    pub fn with_metrics(mut self, enabled: bool) -> Self {
        self.config.enable_metrics = enabled;
        self
    }
    
    /// 启用或禁用缓存预热
    pub fn with_warmup(mut self, enabled: bool) -> Self {
        self.config.enable_warmup = enabled;
        self
    }
    
    /// 启用或禁用自适应缓存
    pub fn with_adaptive_caching(mut self, enabled: bool) -> Self {
        self.config.enable_adaptive_caching = enabled;
        self
    }
    
    /// 构建缓存管理器
    pub fn build(self) -> Result<CacheManager> {
        // 保存config的克隆，以便后续使用
        let config_clone = self.config.clone();
        
        // 创建缓存管理器
        let mut cache_manager = CacheManager::new(self.config)?;
        
        // 添加分布式缓存（如果配置）
        if config_clone.distributed.is_some() && config_clone.distributed.as_ref().unwrap().enabled {
            if let Some(custom_cache) = self.custom_distributed_cache {
                cache_manager = cache_manager.with_distributed_cache(custom_cache)?;
            } else if let Some(redis_config) = self.redis_config {
                let redis_cache = Arc::new(RedisCache::new(redis_config)?);
                cache_manager = cache_manager.with_distributed_cache(redis_cache)?;
            } else if let Some(memcached_config) = self.memcached_config {
                let memcached_cache = Arc::new(MemcachedCache::new(memcached_config)?);
                cache_manager = cache_manager.with_distributed_cache(memcached_cache)?;
            }
        }
        
        // 添加远程缓存（如果配置）
        if config_clone.remote.is_some() && config_clone.remote.as_ref().unwrap().enabled {
            if let Some(custom_cache) = self.custom_remote_cache {
                cache_manager = cache_manager.with_remote_cache(custom_cache)?;
            }
        }
        
        Ok(cache_manager)
    }
}

/// 自动缓存包装器，为任意类型实现自动缓存
pub struct AutoCache<T> {
    /// 缓存键前缀
    key_prefix: String,
    /// 缓存管理器
    cache_manager: Arc<CacheManager>,
    /// 被包装的值
    value: T,
    /// 缓存TTL
    ttl: Option<Duration>,
}

impl<T: serde::Serialize + serde::de::DeserializeOwned> AutoCache<T> {
    /// 创建新的自动缓存包装器
    pub fn new(value: T, key_prefix: &str, cache_manager: Arc<CacheManager>) -> Self {
        Self {
            key_prefix: key_prefix.to_string(),
            cache_manager,
            value,
            ttl: None,
        }
    }
    
    /// 设置缓存TTL
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }
    
    /// 获取被包装的值的只读引用
    pub fn get(&self) -> &T {
        &self.value
    }
    
    /// 获取被包装的值的可变引用
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.value
    }
    
    /// 缓存值
    pub fn cache(&self, key: &str) -> Result<()> {
        // 序列化值
        let value = serde_json::to_vec(&self.value)
            .map_err(|e| crate::error::Error::serialization(format!("序列化缓存值失败: {}", e)))?;
        
        // 构造完整键
        let full_key = format!("{}:{}", self.key_prefix, key);
        
        // 缓存值
        self.cache_manager.set(&full_key, &value)
    }
    
    /// 从缓存加载值
    pub fn load_from_cache(&mut self, key: &str) -> Result<bool> {
        // 构造完整键
        let full_key = format!("{}:{}", self.key_prefix, key);
        
        // 尝试从缓存获取
        match self.cache_manager.get(&full_key)? {
            CacheResult::Hit(value, _) => {
                // 反序列化值
                self.value = serde_json::from_slice(&value)
                    .map_err(|e| crate::error::Error::serialization(format!("反序列化缓存值失败: {}", e)))?;
                Ok(true)
            },
            CacheResult::Miss => Ok(false),
        }
    }
    
    /// 从缓存移除值
    pub fn remove_from_cache(&self, key: &str) -> Result<bool> {
        // 构造完整键
        let full_key = format!("{}:{}", self.key_prefix, key);
        
        // 从缓存删除
        self.cache_manager.delete(&full_key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_builder() {
        let builder = CacheBuilder::new()
            .with_memory_size(50 * 1024 * 1024) // 50MB
            .with_memory_ttl(1800) // 30分钟
            .with_disk_cache(true)
            .with_disk_size(500 * 1024 * 1024) // 500MB
            .with_cleanup_interval(600); // 10分钟
        
        let result = builder.build();
        assert!(result.is_ok(), "Cache builder should create a valid cache manager");
    }
    
    #[test]
    fn test_auto_cache() {
        // 此测试需要实际可工作的缓存管理器，留作集成测试
    }
} 