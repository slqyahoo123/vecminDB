use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant};
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use chrono::Utc;
use log::warn;
#[cfg(feature = "redis")]
use redis::{Client, Commands, ConnectionInfo, RedisResult};
use serde::{Serialize, Deserialize};

use crate::error::{Error, Result};
use super::manager::{CacheMetrics, DistributedCache, CacheTier};

/// Redis分布式缓存配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    /// Redis主机地址
    pub host: String,
    /// Redis端口
    pub port: u16,
    /// Redis密码
    pub password: Option<String>,
    /// Redis数据库索引
    pub db: i64,
    /// 连接池大小
    pub pool_size: u32,
    /// 键前缀
    pub key_prefix: String,
    /// 默认TTL（秒）
    pub default_ttl_secs: Option<u64>,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 6379,
            password: None,
            db: 0,
            pool_size: 10,
            key_prefix: "vecmind:cache:".to_string(),
            default_ttl_secs: Some(3600), // 1小时
        }
    }
}

/// Redis分布式缓存实现
#[cfg(feature = "redis")]
pub struct RedisCache {
    /// Redis客户端
    client: Client,
    /// 缓存配置
    config: RedisConfig,
    /// 缓存统计
    metrics: RwLock<CacheMetrics>,
    /// 最后统计更新锁
    stats_lock: Mutex<()>,
    /// 最后统计更新时间
    last_stats_update: RwLock<Instant>,
}

/// 当未启用 `redis` 特性时提供占位实现，确保类型与接口存在，默认构建可通过
#[cfg(not(feature = "redis"))]
pub struct RedisCache {
    /// 缓存配置
    config: RedisConfig,
}

#[cfg(feature = "redis")]
impl RedisCache {
    /// 创建新的Redis缓存
    pub fn new(config: RedisConfig) -> Result<Self> {
        // 构建连接信息
        let connection_info = ConnectionInfo {
            addr: redis::ConnectionAddr::Tcp(config.host.clone(), config.port),
            db: config.db,
            username: None,
            passwd: config.password.clone(),
        };
        
        // 创建Redis客户端
        let client = Client::open(connection_info)
            .map_err(|e| Error::external(format!("创建Redis客户端失败: {}", e)))?;
        
        // 测试连接
        let mut conn = client.get_connection()
            .map_err(|e| Error::external(format!("连接Redis服务器失败: {}", e)))?;
        
        // 执行PING命令测试连接
        let pong: String = redis::cmd("PING").query(&mut conn)
            .map_err(|e| Error::external(format!("Redis PING命令失败: {}", e)))?;
        
        if pong != "PONG" {
            return Err(Error::external(format!("Redis PING响应异常: {}", pong)));
        }
        
        // 创建缓存指标
        let metrics = CacheMetrics::new(CacheTier::Distributed, 0); // Redis容量无法精确获取
        
        Ok(Self {
            client,
            config,
            metrics: RwLock::new(metrics),
            stats_lock: Mutex::new(()),
            last_stats_update: RwLock::new(Instant::now()),
        })
    }
    
    /// 获取带前缀的键
    fn get_prefixed_key(&self, key: &str) -> String {
        format!("{}{}", self.config.key_prefix, key)
    }
    
    /// 更新缓存统计信息
    fn update_stats(&self) -> Result<()> {
        // 获取锁，但不阻塞
        if let Ok(_lock) = self.stats_lock.try_lock() {
            let now = Instant::now();
            let mut last_update = self.last_stats_update.write()
                .map_err(|_| Error::locks_poison("Redis缓存统计更新时间锁被污染"))?;
            
            // 每分钟最多更新一次统计信息
            if now.duration_since(*last_update) > Duration::from_secs(60) {
                let mut metrics = self.metrics.write()
                    .map_err(|_| Error::locks_poison("Redis缓存指标锁被污染"))?;
                
                // 尝试获取Redis服务器信息
                if let Ok(mut conn) = self.client.get_connection() {
                    // 使用INFO命令获取服务器信息
                    if let Ok(info) = redis::cmd("INFO").query::<String>(&mut conn) {
                        // 解析缓存大小等信息
                        for line in info.lines() {
                            if line.starts_with("used_memory:") {
                                if let Some(value) = line.split(':').nth(1) {
                                    if let Ok(size) = value.trim().parse::<usize>() {
                                        metrics.size_bytes = size;
                                    }
                                }
                            } else if line.starts_with("db") && line.contains("keys=") {
                                // 例如 "db0:keys=1,expires=0,avg_ttl=0"
                                if let Some(keys_part) = line.split(',').next() {
                                    if let Some(keys_value) = keys_part.split('=').nth(1) {
                                        if let Ok(count) = keys_value.trim().parse::<usize>() {
                                            // 只计算我们的前缀的键
                                            // 这是一个估计值，实际上需要更复杂的统计
                                            metrics.entries = count;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                
                // 更新统计时间
                *last_update = now;
                metrics.last_updated = Utc::now();
            }
        }
        
        Ok(())
    }
}

#[cfg(not(feature = "redis"))]
impl RedisCache {
    /// 创建占位实现（未启用 redis 特性）
    pub fn new(config: RedisConfig) -> Result<Self> {
        Ok(Self { config })
    }
}

#[cfg(feature = "redis")]
impl DistributedCache for RedisCache {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let start = Instant::now();
        let prefixed_key = self.get_prefixed_key(key);
        
        // 获取连接
        let mut conn = self.client.get_connection()
            .map_err(|e| Error::external(format!("获取Redis连接失败: {}", e)))?;
        
        // 尝试获取值
        let result: RedisResult<Vec<u8>> = conn.get(&prefixed_key);
        
        // 更新指标
        let mut metrics = self.metrics.write()
            .map_err(|_| Error::locks_poison("Redis缓存指标锁被污染"))?;
        
        match result {
            Ok(value) => {
                // 命中
                metrics.record_hit(start.elapsed().as_micros() as u64);
                Ok(Some(value))
            },
            Err(e) => {
                // 检查是否是键不存在错误
                if e.kind() == redis::ErrorKind::TypeError || e.kind() == redis::ErrorKind::ResponseError {
                    // 未命中
                    metrics.record_miss(start.elapsed().as_micros() as u64);
                    Ok(None)
                } else {
                    // 其他错误
                    Err(Error::external(format!("Redis获取值失败: {}", e)))
                }
            }
        }
    }
    
    fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        let prefixed_key = self.get_prefixed_key(key);
        
        // 获取连接
        let mut conn = self.client.get_connection()
            .map_err(|e| Error::external(format!("获取Redis连接失败: {}", e)))?;
        
        // 设置值
        let result: RedisResult<()> = if let Some(ttl) = self.config.default_ttl_secs {
            // 设置带过期时间的值
            conn.set_ex(&prefixed_key, value, ttl as usize)
        } else {
            // 设置永久值
            conn.set(&prefixed_key, value)
        };
        
        // 更新指标
        if result.is_ok() {
            let mut metrics = self.metrics.write()
                .map_err(|_| Error::locks_poison("Redis缓存指标锁被污染"))?;
            metrics.record_write(value.len());
        }
        
        result.map_err(|e| Error::external(format!("Redis设置值失败: {}", e)))?;
        
        // 更新统计信息
        self.update_stats()?;
        
        Ok(())
    }
    
    fn delete(&self, key: &str) -> Result<bool> {
        let prefixed_key = self.get_prefixed_key(key);
        
        // 获取连接
        let mut conn = self.client.get_connection()
            .map_err(|e| Error::external(format!("获取Redis连接失败: {}", e)))?;
        
        // 删除键
        let result: RedisResult<i64> = conn.del(&prefixed_key);
        
        match result {
            Ok(deleted) => {
                // 更新指标
                if deleted > 0 {
                    let mut metrics = self.metrics.write()
                        .map_err(|_| Error::locks_poison("Redis缓存指标锁被污染"))?;
                    metrics.record_delete(0); // 无法准确获取删除的大小
                }
                
                // 更新统计信息
                self.update_stats()?;
                
                Ok(deleted > 0)
            },
            Err(e) => {
                Err(Error::external(format!("Redis删除键失败: {}", e)))
            }
        }
    }
    
    fn clear(&self) -> Result<()> {
        // 获取连接
        let mut conn = self.client.get_connection()
            .map_err(|e| Error::external(format!("获取Redis连接失败: {}", e)))?;
        
        // 获取所有前缀匹配的键
        let pattern = format!("{}*", self.config.key_prefix);
        let keys: Vec<String> = redis::cmd("KEYS")
            .arg(&pattern)
            .query(&mut conn)
            .map_err(|e| Error::external(format!("Redis获取键列表失败: {}", e)))?;
        
        if !keys.is_empty() {
            // 删除所有键
            let result: RedisResult<i64> = conn.del(&keys);
            
            if let Ok(deleted) = result {
                // 更新指标
                let mut metrics = self.metrics.write()
                    .map_err(|_| Error::locks_poison("Redis缓存指标锁被污染"))?;
                metrics.entries = metrics.entries.saturating_sub(deleted as usize);
                metrics.last_updated = Utc::now();
            }
        }
        
        // 更新统计信息
        self.update_stats()?;
        
        Ok(())
    }
    
    fn get_stats(&self) -> Result<CacheMetrics> {
        // 先尝试更新统计信息
        self.update_stats()?;
        
        // 返回指标
        let metrics = self.metrics.read()
            .map_err(|_| Error::locks_poison("Redis缓存指标锁被污染"))?;
        
        Ok(metrics.clone())
    }
}

#[cfg(not(feature = "redis"))]
impl DistributedCache for RedisCache {
    fn get(&self, _key: &str) -> Result<Option<Vec<u8>>> {
        Err(Error::not_implemented("Redis support requires enabling the 'redis' feature"))
    }
    fn set(&self, _key: &str, _value: &[u8]) -> Result<()> {
        Err(Error::not_implemented("Redis support requires enabling the 'redis' feature"))
    }
    fn delete(&self, _key: &str) -> Result<bool> {
        Err(Error::not_implemented("Redis support requires enabling the 'redis' feature"))
    }
    fn clear(&self) -> Result<()> {
        Err(Error::not_implemented("Redis support requires enabling the 'redis' feature"))
    }
    fn get_stats(&self) -> Result<CacheMetrics> {
        Err(Error::not_implemented("Redis support requires enabling the 'redis' feature"))
    }
}

/// Memcached分布式缓存配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemcachedConfig {
    /// Memcached服务器列表
    pub servers: Vec<String>,
    /// 键前缀
    pub key_prefix: String,
    /// 默认TTL（秒）
    pub default_ttl_secs: Option<u64>,
}

impl Default for MemcachedConfig {
    fn default() -> Self {
        Self {
            servers: vec!["127.0.0.1:11211".to_string()],
            key_prefix: "vecmind:cache:".to_string(),
            default_ttl_secs: Some(3600), // 1小时
        }
    }
}

/// Memcached分布式缓存实现（特性门控）
#[cfg(feature = "memcached")]
pub struct MemcachedCache {
    /// Memcached客户端
    #[allow(dead_code)]
    client: String, // 实际实现中应该使用真正的Memcached客户端
    /// 缓存配置
    config: MemcachedConfig,
    /// 缓存统计
    metrics: RwLock<CacheMetrics>,
}

#[cfg(feature = "memcached")]
impl MemcachedCache {
    /// 创建新的Memcached缓存
    pub fn new(config: MemcachedConfig) -> Result<Self> {
        // 这是一个占位实现，实际项目中应该使用真正的Memcached客户端
        info!("创建Memcached缓存（占位实现）");
        
        // 创建缓存指标
        let metrics = CacheMetrics::new(CacheTier::Distributed, 0);
        
        Ok(Self {
            client: "memcached_client_placeholder".to_string(),
            config,
            metrics: RwLock::new(metrics),
        })
    }
}

#[cfg(feature = "memcached")]
impl DistributedCache for MemcachedCache {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let start = Instant::now();
        let _prefixed_key = format!("{}{}", self.config.key_prefix, key);
        
        // 更新指标
        let mut metrics = self.metrics.write()
            .map_err(|_| Error::locks_poison("Memcached缓存指标锁被污染"))?;
        
        // 这是占位实现，总是返回未命中
        metrics.record_miss(start.elapsed().as_micros() as u64);
        
        warn!("Memcached缓存是占位实现，总是返回未命中");
        Ok(None)
    }
    
    fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        let _prefixed_key = format!("{}{}", self.config.key_prefix, key);
        
        // 更新指标
        let mut metrics = self.metrics.write()
            .map_err(|_| Error::locks_poison("Memcached缓存指标锁被污染"))?;
        metrics.record_write(value.len());
        
        warn!("Memcached缓存是占位实现，不会实际存储值");
        Ok(())
    }
    
    fn delete(&self, key: &str) -> Result<bool> {
        let _prefixed_key = format!("{}{}", self.config.key_prefix, key);
        
        warn!("Memcached缓存是占位实现，不会实际删除键");
        Ok(false)
    }
    
    fn clear(&self) -> Result<()> {
        warn!("Memcached缓存是占位实现，不会实际清空缓存");
        Ok(())
    }
    
    fn get_stats(&self) -> Result<CacheMetrics> {
        let metrics = self.metrics.read()
            .map_err(|_| Error::locks_poison("Memcached缓存指标锁被污染"))?;
        
        Ok(metrics.clone())
    }
}

/// 当未启用 `memcached` 特性时提供类型与错误返回，避免占位实现影响构建语义
#[cfg(not(feature = "memcached"))]
pub struct MemcachedCache {
    pub(crate) config: MemcachedConfig,
}

#[cfg(not(feature = "memcached"))]
impl MemcachedCache {
    pub fn new(config: MemcachedConfig) -> Result<Self> { Ok(Self { config }) }
}

#[cfg(not(feature = "memcached"))]
impl DistributedCache for MemcachedCache {
    fn get(&self, _key: &str) -> Result<Option<Vec<u8>>> { Err(Error::not_implemented("Memcached support requires enabling the 'memcached' feature")) }
    fn set(&self, _key: &str, _value: &[u8]) -> Result<()> { Err(Error::not_implemented("Memcached support requires enabling the 'memcached' feature")) }
    fn delete(&self, _key: &str) -> Result<bool> { Err(Error::not_implemented("Memcached support requires enabling the 'memcached' feature")) }
    fn clear(&self) -> Result<()> { Err(Error::not_implemented("Memcached support requires enabling the 'memcached' feature")) }
    fn get_stats(&self) -> Result<CacheMetrics> { Err(Error::not_implemented("Memcached support requires enabling the 'memcached' feature")) }
}

/// 一致性哈希环实现
#[derive(Debug, Clone)]
pub struct ConsistentHashRing {
    /// 哈希环，键为哈希值，值为节点索引
    ring: BTreeMap<u64, usize>,
    /// 节点列表
    nodes: Vec<String>,
    /// 虚拟节点数量
    virtual_nodes: usize,
}

impl ConsistentHashRing {
    /// 创建新的一致性哈希环
    pub fn new(nodes: Vec<String>, virtual_nodes: usize) -> Self {
        let mut ring = BTreeMap::new();
        
        // 为每个节点创建虚拟节点
        for (node_index, node) in nodes.iter().enumerate() {
            for i in 0..virtual_nodes {
                let virtual_node_name = format!("{}:{}", node, i);
                let hash = Self::hash_string(&virtual_node_name);
                ring.insert(hash, node_index);
            }
        }
        
        Self {
            ring,
            nodes,
            virtual_nodes,
        }
    }
    
    /// 添加节点
    pub fn add_node(&mut self, node: String) -> usize {
        let node_index = self.nodes.len();
        self.nodes.push(node.clone());
        
        // 为新节点创建虚拟节点
        for i in 0..self.virtual_nodes {
            let virtual_node_name = format!("{}:{}", node, i);
            let hash = Self::hash_string(&virtual_node_name);
            self.ring.insert(hash, node_index);
        }
        
        node_index
    }
    
    /// 移除节点
    pub fn remove_node(&mut self, node_index: usize) -> Option<String> {
        if node_index >= self.nodes.len() {
            return None;
        }
        
        let node = self.nodes[node_index].clone();
        
        // 移除该节点的所有虚拟节点
        for i in 0..self.virtual_nodes {
            let virtual_node_name = format!("{}:{}", node, i);
            let hash = Self::hash_string(&virtual_node_name);
            self.ring.remove(&hash);
        }
        
        // 更新其他节点的索引
        let mut new_ring = BTreeMap::new();
        for (hash, old_index) in &self.ring {
            let new_index = if *old_index > node_index {
                *old_index - 1
            } else {
                *old_index
            };
            new_ring.insert(*hash, new_index);
        }
        
        self.ring = new_ring;
        self.nodes.remove(node_index);
        
        Some(node)
    }
    
    /// 获取键对应的节点索引
    pub fn get_node(&self, key: &str) -> Option<usize> {
        if self.ring.is_empty() {
            return None;
        }
        
        let hash = Self::hash_string(key);
        
        // 在环上查找第一个大于等于该哈希值的节点
        if let Some((&_, &node_index)) = self.ring.range(hash..).next() {
            Some(node_index)
        } else {
            // 如果没找到，返回环上的第一个节点（环形结构）
            self.ring.iter().next().map(|(_, &node_index)| node_index)
        }
    }
    
    /// 获取键对应的多个节点（用于复制）
    pub fn get_nodes(&self, key: &str, count: usize) -> Vec<usize> {
        if self.ring.is_empty() {
            return Vec::new();
        }
        
        let hash = Self::hash_string(key);
        let mut result = Vec::new();
        let mut seen = std::collections::HashSet::new();
        
        // 从哈希位置开始环形查找不同的节点
        let mut iter = self.ring.range(hash..).chain(self.ring.range(..hash));
        
        for (_, &node_index) in iter {
            if !seen.contains(&node_index) {
                seen.insert(node_index);
                result.push(node_index);
                if result.len() >= count {
                    break;
                }
            }
        }
        
        result
    }
    
    /// 获取所有节点
    pub fn get_all_nodes(&self) -> &[String] {
        &self.nodes
    }
    
    /// 获取节点数量
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    
    /// 哈希字符串
    fn hash_string(s: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }
}

/// 集群分布式缓存实现，使用多个缓存后端
pub struct ClusteredCache {
    /// 缓存后端列表
    backends: Vec<Arc<dyn DistributedCache>>,
    /// 一致性哈希环（用于确定键应该存储在哪个后端）
    hash_ring: RwLock<ConsistentHashRing>,
    /// 缓存统计
    metrics: RwLock<CacheMetrics>,
    /// 副本数量
    replication_factor: usize,
}

impl ClusteredCache {
    /// 创建新的集群缓存
    pub fn new(backends: Vec<Arc<dyn DistributedCache>>) -> Result<Self> {
        if backends.is_empty() {
            return Err(Error::invalid_argument("缓存后端列表不能为空"));
        }
        
        // 创建节点名称列表
        let node_names: Vec<String> = (0..backends.len())
            .map(|i| format!("backend-{}", i))
            .collect();
        
        // 创建一致性哈希环，使用160个虚拟节点以获得良好的负载均衡
        let hash_ring = ConsistentHashRing::new(node_names, 160);
        
        // 创建缓存指标
        let metrics = CacheMetrics::new(CacheTier::Distributed, 0);
        
        Ok(Self {
            backends,
            hash_ring: RwLock::new(hash_ring),
            metrics: RwLock::new(metrics),
            replication_factor: 1, // 默认不复制
        })
    }
    
    /// 创建带有副本的集群缓存
    pub fn new_with_replication(
        backends: Vec<Arc<dyn DistributedCache>>,
        replication_factor: usize,
    ) -> Result<Self> {
        let mut cache = Self::new(backends)?;
        cache.replication_factor = replication_factor.max(1).min(cache.backends.len());
        Ok(cache)
    }
    
    /// 添加缓存后端
    pub fn add_backend(&mut self, backend: Arc<dyn DistributedCache>) -> Result<()> {
        let backend_index = self.backends.len();
        self.backends.push(backend);
        
        // 更新哈希环
        let mut ring = self.hash_ring.write()
            .map_err(|_| Error::locks_poison("集群缓存哈希环锁被污染"))?;
        ring.add_node(format!("backend-{}", backend_index));
        
        Ok(())
    }
    
    /// 移除缓存后端
    pub fn remove_backend(&mut self, backend_index: usize) -> Result<()> {
        if backend_index >= self.backends.len() {
            return Err(Error::invalid_argument("无效的后端索引"));
        }
        
        // 更新哈希环
        let mut ring = self.hash_ring.write()
            .map_err(|_| Error::locks_poison("集群缓存哈希环锁被污染"))?;
        ring.remove_node(backend_index);
        
        // 移除后端
        self.backends.remove(backend_index);
        
        Ok(())
    }
    
    /// 获取键对应的后端索引
    fn get_backend_indices(&self, key: &str) -> Result<Vec<usize>> {
        let ring = self.hash_ring.read()
            .map_err(|_| Error::locks_poison("集群缓存哈希环锁被污染"))?;
        
        let indices = ring.get_nodes(key, self.replication_factor);
        Ok(indices)
    }
    
    /// 获取主要后端索引
    fn get_primary_backend_index(&self, key: &str) -> Result<usize> {
        let ring = self.hash_ring.read()
            .map_err(|_| Error::locks_poison("集群缓存哈希环锁被污染"))?;
        
        ring.get_node(key)
            .ok_or_else(|| Error::resource_exhausted("没有可用的缓存后端"))
    }
}

impl DistributedCache for ClusteredCache {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let start = Instant::now();
        
        // 获取所有副本的后端索引
        let backend_indices = self.get_backend_indices(key)?;
        
        // 尝试从主要后端获取值
        if let Some(&primary_index) = backend_indices.first() {
            if let Ok(Some(value)) = self.backends[primary_index].get(key) {
                // 主要后端成功返回值
                let mut metrics = self.metrics.write()
                    .map_err(|_| Error::locks_poison("集群缓存指标锁被污染"))?;
                metrics.record_hit(start.elapsed().as_micros() as u64);
                return Ok(Some(value));
            }
        }
        
        // 主要后端失败，尝试其他副本
        for &backend_index in backend_indices.iter().skip(1) {
            if let Ok(Some(value)) = self.backends[backend_index].get(key) {
                // 副本后端成功返回值
                let mut metrics = self.metrics.write()
                    .map_err(|_| Error::locks_poison("集群缓存指标锁被污染"))?;
                metrics.record_hit(start.elapsed().as_micros() as u64);
                
                // 异步修复主要后端（这里简化处理，实际应该在后台进行）
                if let Some(&primary_index) = backend_indices.first() {
                    let _ = self.backends[primary_index].set(key, &value);
                }
                
                return Ok(Some(value));
            }
        }
        
        // 所有后端都未命中
        let mut metrics = self.metrics.write()
            .map_err(|_| Error::locks_poison("集群缓存指标锁被污染"))?;
        metrics.record_miss(start.elapsed().as_micros() as u64);
        
        Ok(None)
    }
    
    fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        // 获取所有副本的后端索引
        let backend_indices = self.get_backend_indices(key)?;
        
        let mut success_count = 0;
        let mut last_error = None;
        
        // 写入所有副本
        for &backend_index in &backend_indices {
            match self.backends[backend_index].set(key, value) {
                Ok(()) => {
                    success_count += 1;
                },
                Err(e) => {
                    last_error = Some(e);
                    warn!("写入缓存后端 {} 失败: {:?}", backend_index, last_error);
                }
            }
        }
        
        // 需要至少一个副本写入成功
        if success_count > 0 {
            // 更新指标
            let mut metrics = self.metrics.write()
                .map_err(|_| Error::locks_poison("集群缓存指标锁被污染"))?;
            metrics.record_write(value.len());
            
            Ok(())
        } else {
            // 所有副本都写入失败
            Err(last_error.unwrap_or_else(|| {
                Error::external("所有缓存后端写入失败".to_string())
            }))
        }
    }
    
    fn delete(&self, key: &str) -> Result<bool> {
        // 获取所有副本的后端索引
        let backend_indices = self.get_backend_indices(key)?;
        
        let mut deleted = false;
        
        // 从所有副本删除
        for &backend_index in &backend_indices {
            match self.backends[backend_index].delete(key) {
                Ok(true) => {
                    deleted = true;
                },
                Ok(false) => {
                    // 键不存在，继续处理其他副本
                },
                Err(e) => {
                    warn!("从缓存后端 {} 删除键失败: {:?}", backend_index, e);
                }
            }
        }
        
        if deleted {
            // 更新指标
            let mut metrics = self.metrics.write()
                .map_err(|_| Error::locks_poison("集群缓存指标锁被污染"))?;
            metrics.record_delete(0); // 无法准确获取删除的大小
        }
        
        Ok(deleted)
    }
    
    fn clear(&self) -> Result<()> {
        // 清空所有后端
        for (index, backend) in self.backends.iter().enumerate() {
            if let Err(e) = backend.clear() {
                warn!("清空缓存后端 {} 失败: {}", index, e);
            }
        }
        
        // 重置指标
        let mut metrics = self.metrics.write()
            .map_err(|_| Error::locks_poison("集群缓存指标锁被污染"))?;
        metrics.entries = 0;
        metrics.size_bytes = 0;
        metrics.last_updated = Utc::now();
        
        Ok(())
    }
    
    fn get_stats(&self) -> Result<CacheMetrics> {
        // 聚合所有后端的统计信息
        let mut total_metrics = CacheMetrics::new(CacheTier::Distributed, 0);
        
        for (index, backend) in self.backends.iter().enumerate() {
            match backend.get_stats() {
                Ok(backend_metrics) => {
                    total_metrics.hits += backend_metrics.hits;
                    total_metrics.misses += backend_metrics.misses;
                    total_metrics.writes += backend_metrics.writes;
                    total_metrics.deletes += backend_metrics.deletes;
                    total_metrics.evictions += backend_metrics.evictions;
                    total_metrics.entries += backend_metrics.entries;
                    total_metrics.size_bytes += backend_metrics.size_bytes;
                    total_metrics.capacity_bytes += backend_metrics.capacity_bytes;
                },
                Err(e) => {
                    warn!("获取缓存后端 {} 统计信息失败: {}", index, e);
                }
            }
        }
        
        // 计算平均值（避免重复计算副本）
        if self.replication_factor > 1 {
            total_metrics.entries /= self.replication_factor;
            total_metrics.size_bytes /= self.replication_factor;
        }
        
        // 更新命中率
        let total = total_metrics.hits + total_metrics.misses;
        if total > 0 {
            total_metrics.hit_ratio = total_metrics.hits as f64 / total as f64;
        }
        
        total_metrics.last_updated = Utc::now();
        
        // 存回指标
        *self.metrics.write().map_err(|_| Error::locks_poison("集群缓存指标锁被污染"))? = total_metrics.clone();
        
        Ok(total_metrics)
    }
} 