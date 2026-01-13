//! 缓存后端模块
//!
//! 提供统一的缓存后端抽象接口，支持多种类型的缓存后端

use crate::error::Result;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use std::path::PathBuf;
use tokio::sync::{Mutex, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use log::{info, warn};

/// 缓存后端类型
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendType {
    /// 内存缓存
    Memory,
    /// 磁盘缓存
    Disk,
    /// Redis缓存
    Redis,
    /// Memcached缓存
    Memcached,
    /// 自定义缓存
    Custom(String),
}

/// 缓存后端配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// 后端类型
    pub backend_type: BackendType,
    /// 连接字符串
    pub connection_string: Option<String>,
    /// 配置参数
    pub parameters: HashMap<String, String>,
    /// 超时配置
    pub timeout: Option<Duration>,
    /// 重试次数
    pub retry_count: usize,
    /// 是否启用压缩
    pub compression_enabled: bool,
    /// 缓存目录
    pub cache_dir: PathBuf,
    /// Redis连接池大小
    pub max_connections: Option<usize>,
    /// Redis键前缀
    pub key_prefix: String,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            backend_type: BackendType::Memory,
            connection_string: None,
            parameters: HashMap::new(),
            timeout: Some(Duration::from_secs(30)),
            retry_count: 3,
            compression_enabled: false,
            cache_dir: PathBuf::from("./cache"),
            max_connections: Some(10),
            key_prefix: String::new(),
        }
    }
}

/// 缓存后端统一接口
#[async_trait]
pub trait CacheBackend: Send + Sync {
    /// 获取缓存值
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    
    /// 设置缓存值
    async fn set(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> Result<()>;
    
    /// 删除缓存值
    async fn delete(&self, key: &str) -> Result<bool>;
    
    /// 检查键是否存在
    async fn exists(&self, key: &str) -> Result<bool>;
    
    /// 设置过期时间
    async fn expire(&self, key: &str, ttl: Duration) -> Result<bool>;
    
    /// 清空所有缓存
    async fn clear(&self) -> Result<()>;
    
    /// 获取缓存大小
    async fn size(&self) -> Result<usize>;
    
    /// 获取所有键
    async fn keys(&self, pattern: Option<&str>) -> Result<Vec<String>>;
    
    /// 批量获取
    async fn mget(&self, keys: &[String]) -> Result<HashMap<String, Option<Vec<u8>>>>;
    
    /// 批量设置
    async fn mset(&self, data: HashMap<String, Vec<u8>>, ttl: Option<Duration>) -> Result<()>;
    
    /// 批量删除
    async fn mdel(&self, keys: &[String]) -> Result<usize>;
    
    /// 健康检查
    async fn health_check(&self) -> Result<bool>;
    
    /// 获取后端统计信息
    async fn stats(&self) -> Result<BackendStats>;
    
    /// 获取后端类型
    fn backend_type(&self) -> BackendType;
    
    /// 关闭连接
    async fn close(&self) -> Result<()>;
}

/// 后端统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendStats {
    /// 总键数
    pub total_keys: usize,
    /// 总内存使用量（字节）
    pub memory_usage: usize,
    /// 命中次数
    pub hits: u64,
    /// 未命中次数
    pub misses: u64,
    /// 命中率
    pub hit_ratio: f64,
    /// 连接数
    pub connections: usize,
    /// 是否在线
    pub online: bool,
    /// 延迟（毫秒）
    pub latency_ms: f64,
    /// 总操作数
    pub total_operations: u64,
    /// 成功操作数
    pub successful_operations: u64,
    /// 失败操作数
    pub failed_operations: u64,
    /// 命中率
    pub hit_rate: f64,
    /// 总运行时间（秒）
    pub uptime_seconds: u64,
    /// 最后错误
    pub last_error: Option<String>,
}

impl Default for BackendStats {
    fn default() -> Self {
        Self {
            total_keys: 0,
            memory_usage: 0,
            hits: 0,
            misses: 0,
            hit_ratio: 0.0,
            connections: 0,
            online: true,
            latency_ms: 0.0,
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            hit_rate: 0.0,
            uptime_seconds: 0,
            last_error: None,
        }
    }
}

/// 后端工厂
pub struct BackendFactory;

impl BackendFactory {
    /// 创建后端实例
    pub fn create_backend(config: BackendConfig) -> Result<Arc<dyn CacheBackend>> {
        match config.backend_type {
            BackendType::Memory => {
                Ok(Arc::new(MemoryBackend::new(config)?))
            }
            BackendType::Disk => {
                Ok(Arc::new(DiskBackend::new(config)?))
            }
            BackendType::Redis => {
                Ok(Arc::new(RedisBackend::new(config)?))
            }
            BackendType::Memcached => {
                Ok(Arc::new(MemcachedBackend::new(config)?))
            }
            BackendType::Custom(ref name) => {
                Err(crate::error::Error::Unsupported(
                    format!("暂不支持自定义后端类型: {}", name)
                ))
            }
        }
    }
}

/// 内存后端实现
pub struct MemoryBackend {
    storage: Arc<tokio::sync::RwLock<HashMap<String, (Vec<u8>, Option<tokio::time::Instant>)>>>,
    config: BackendConfig,
    stats: Arc<tokio::sync::RwLock<BackendStats>>,
}

impl MemoryBackend {
    pub fn new(config: BackendConfig) -> Result<Self> {
        Ok(Self {
            storage: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(tokio::sync::RwLock::new(BackendStats::default())),
        })
    }
    
    async fn is_expired(&self, expiry: Option<tokio::time::Instant>) -> bool {
        if let Some(exp) = expiry {
            tokio::time::Instant::now() > exp
        } else {
            false
        }
    }
    
    async fn cleanup_expired(&self) {
        let mut storage = self.storage.write().await;
        let now = tokio::time::Instant::now();
        
        storage.retain(|_, (_, expiry)| {
            if let Some(exp) = expiry {
                now <= *exp
            } else {
                true
            }
        });
    }
}

#[async_trait]
impl CacheBackend for MemoryBackend {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        self.cleanup_expired().await;
        
        let storage = self.storage.read().await;
        if let Some((value, expiry)) = storage.get(key) {
            if !self.is_expired(*expiry).await {
                let mut stats = self.stats.write().await;
                stats.hits += 1;
                return Ok(Some(value.clone()));
            }
        }
        
        let mut stats = self.stats.write().await;
        stats.misses += 1;
        Ok(None)
    }
    
    async fn set(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> Result<()> {
        let expiry = ttl.map(|t| tokio::time::Instant::now() + t);
        
        let mut storage = self.storage.write().await;
        storage.insert(key.to_string(), (value.to_vec(), expiry));
        
        let mut stats = self.stats.write().await;
        stats.total_keys = storage.len();
        stats.memory_usage += value.len();
        
        Ok(())
    }
    
    async fn delete(&self, key: &str) -> Result<bool> {
        let mut storage = self.storage.write().await;
        let removed = storage.remove(key).is_some();
        
        let mut stats = self.stats.write().await;
        stats.total_keys = storage.len();
        
        Ok(removed)
    }
    
    async fn exists(&self, key: &str) -> Result<bool> {
        let storage = self.storage.read().await;
        if let Some((_, expiry)) = storage.get(key) {
            Ok(!self.is_expired(*expiry).await)
        } else {
            Ok(false)
        }
    }
    
    async fn expire(&self, key: &str, ttl: Duration) -> Result<bool> {
        let mut storage = self.storage.write().await;
        if let Some((value, _)) = storage.get(key).cloned() {
            let expiry = Some(tokio::time::Instant::now() + ttl);
            storage.insert(key.to_string(), (value, expiry));
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    async fn clear(&self) -> Result<()> {
        let mut storage = self.storage.write().await;
        storage.clear();
        
        let mut stats = self.stats.write().await;
        stats.total_keys = 0;
        stats.memory_usage = 0;
        
        Ok(())
    }
    
    async fn size(&self) -> Result<usize> {
        self.cleanup_expired().await;
        let storage = self.storage.read().await;
        Ok(storage.len())
    }
    
    async fn keys(&self, pattern: Option<&str>) -> Result<Vec<String>> {
        self.cleanup_expired().await;
        
        let storage = self.storage.read().await;
        let keys: Vec<String> = if let Some(pat) = pattern {
            storage.keys()
                .filter(|k| k.contains(pat))
                .cloned()
                .collect()
        } else {
            storage.keys().cloned().collect()
        };
        
        Ok(keys)
    }
    
    async fn mget(&self, keys: &[String]) -> Result<HashMap<String, Option<Vec<u8>>>> {
        let mut result = HashMap::new();
        for key in keys {
            result.insert(key.clone(), self.get(key).await?);
        }
        Ok(result)
    }
    
    async fn mset(&self, data: HashMap<String, Vec<u8>>, ttl: Option<Duration>) -> Result<()> {
        for (key, value) in data {
            self.set(&key, &value, ttl).await?;
        }
        Ok(())
    }
    
    async fn mdel(&self, keys: &[String]) -> Result<usize> {
        let mut count = 0;
        for key in keys {
            if self.delete(key).await? {
                count += 1;
            }
        }
        Ok(count)
    }
    
    async fn health_check(&self) -> Result<bool> {
        Ok(true)
    }
    
    async fn stats(&self) -> Result<BackendStats> {
        let stats = self.stats.read().await;
        let mut result = stats.clone();
        
        if result.hits + result.misses > 0 {
            result.hit_ratio = result.hits as f64 / (result.hits + result.misses) as f64;
        }
        
        Ok(result)
    }
    
    fn backend_type(&self) -> BackendType {
        BackendType::Memory
    }
    
    async fn close(&self) -> Result<()> {
        self.clear().await
    }
}

/// 磁盘后端实现
pub struct DiskBackend {
    config: BackendConfig,
}

impl DiskBackend {
    pub fn new(config: BackendConfig) -> Result<Self> {
        Ok(Self { config })
    }
}

#[async_trait]
impl CacheBackend for DiskBackend {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let file_path = self.get_file_path(key);
        if !file_path.exists() {
            return Ok(None);
        }

        // 检查文件是否过期
        if let Ok(metadata) = file_path.metadata() {
            if let Ok(modified) = metadata.modified() {
                if let Ok(elapsed) = modified.elapsed() {
                    if let Some(ttl) = self.get_key_ttl(key).await? {
                        if elapsed > ttl {
                            // 文件已过期，删除并返回None
                            let _ = tokio::fs::remove_file(&file_path).await;
                            self.remove_key_ttl(key).await?;
                            return Ok(None);
                        }
                    }
                }
            }
        }

        match tokio::fs::read(&file_path).await {
            Ok(data) => Ok(Some(data)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(crate::error::Error::io(format!("读取缓存文件失败: {}", e))),
        }
    }
    
    async fn set(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> Result<()> {
        let file_path = self.get_file_path(key);
        
        // 确保目录存在
        if let Some(parent) = file_path.parent() {
            tokio::fs::create_dir_all(parent).await
                .map_err(|e| crate::error::Error::io(format!("创建缓存目录失败: {}", e)))?;
        }

        // 写入数据
        tokio::fs::write(&file_path, value).await
            .map_err(|e| crate::error::Error::io(format!("写入缓存文件失败: {}", e)))?;

        // 设置TTL
        if let Some(duration) = ttl {
            self.set_key_ttl(key, duration).await?;
        }

        Ok(())
    }
    
    async fn delete(&self, key: &str) -> Result<bool> {
        let file_path = self.get_file_path(key);
        match tokio::fs::remove_file(&file_path).await {
            Ok(_) => {
                self.remove_key_ttl(key).await?;
                Ok(true)
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(crate::error::Error::io(format!("删除缓存文件失败: {}", e))),
        }
    }
    
    async fn exists(&self, key: &str) -> Result<bool> {
        let file_path = self.get_file_path(key);
        Ok(file_path.exists())
    }
    
    async fn expire(&self, key: &str, ttl: Duration) -> Result<bool> {
        if self.exists(key).await? {
            self.set_key_ttl(key, ttl).await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    async fn clear(&self) -> Result<()> {
        let cache_dir = &self.config.cache_dir;
        if cache_dir.exists() {
            tokio::fs::remove_dir_all(cache_dir).await
                .map_err(|e| crate::error::Error::io(format!("清空缓存目录失败: {}", e)))?;
            tokio::fs::create_dir_all(cache_dir).await
                .map_err(|e| crate::error::Error::io(format!("重建缓存目录失败: {}", e)))?;
        }
        
        // 清理TTL数据
        self.clear_all_ttl().await?;
        Ok(())
    }
    
    async fn size(&self) -> Result<usize> {
        let cache_dir = &self.config.cache_dir;
        let mut count = 0;
        
        if cache_dir.exists() {
            let mut entries = tokio::fs::read_dir(cache_dir).await
                .map_err(|e| crate::error::Error::io(format!("读取缓存目录失败: {}", e)))?;
            
            while let Some(_entry) = entries.next_entry().await
                .map_err(|e| crate::error::Error::io(format!("读取目录条目失败: {}", e)))? {
                count += 1;
            }
        }
        
        Ok(count)
    }
    
    async fn keys(&self, pattern: Option<&str>) -> Result<Vec<String>> {
        let cache_dir = &self.config.cache_dir;
        let mut keys = Vec::new();
        
        if cache_dir.exists() {
            let mut entries = tokio::fs::read_dir(cache_dir).await
                .map_err(|e| crate::error::Error::io(format!("读取缓存目录失败: {}", e)))?;
            
            while let Some(entry) = entries.next_entry().await
                .map_err(|e| crate::error::Error::io(format!("读取目录条目失败: {}", e)))? {
                
                if let Some(file_name) = entry.file_name().to_str() {
                    let key = self.decode_key_from_filename(file_name);
                    
                    if let Some(pat) = pattern {
                        if self.match_pattern(&key, pat) {
                            keys.push(key);
                        }
                    } else {
                        keys.push(key);
                    }
                }
            }
        }
        
        Ok(keys)
    }
    
    async fn mget(&self, keys: &[String]) -> Result<HashMap<String, Option<Vec<u8>>>> {
        let mut results = HashMap::new();
        
        for key in keys {
            let value = self.get(key).await?;
            results.insert(key.clone(), value);
        }
        
        Ok(results)
    }
    
    async fn mset(&self, data: HashMap<String, Vec<u8>>, ttl: Option<Duration>) -> Result<()> {
        for (key, value) in data {
            self.set(&key, &value, ttl).await?;
        }
        Ok(())
    }
    
    async fn mdel(&self, keys: &[String]) -> Result<usize> {
        let mut deleted_count = 0;
        
        for key in keys {
            if self.delete(key).await? {
                deleted_count += 1;
            }
        }
        
        Ok(deleted_count)
    }
    
    async fn health_check(&self) -> Result<bool> {
        // 检查缓存目录是否可写
        let test_key = "health_check_test";
        let test_data = b"test";
        
        match self.set(test_key, test_data, None).await {
            Ok(_) => {
                let _ = self.delete(test_key).await;
                Ok(true)
            },
            Err(_) => Ok(false),
        }
    }
    
    async fn stats(&self) -> Result<BackendStats> {
        let size = self.size().await?;
        let cache_dir = &self.config.cache_dir;
        
        let mut total_size = 0u64;
        if cache_dir.exists() {
            let mut entries = tokio::fs::read_dir(cache_dir).await
                .map_err(|e| crate::error::Error::io(format!("读取缓存目录失败: {}", e)))?;
            
            while let Some(entry) = entries.next_entry().await
                .map_err(|e| crate::error::Error::io(format!("读取目录条目失败: {}", e)))? {
                
                if let Ok(metadata) = entry.metadata().await {
                    total_size += metadata.len();
                }
            }
        }
        
        Ok(BackendStats {
            hits: 0, // 磁盘缓存不跟踪命中率
            misses: 0,
            total_keys: size,
            memory_usage: total_size,
            network_bytes_in: 0,
            network_bytes_out: 0,
        })
    }
    
    fn backend_type(&self) -> BackendType {
        BackendType::Disk
    }
    
    async fn close(&self) -> Result<()> {
        // 磁盘缓存没有连接需要关闭
        Ok(())
    }
}

impl DiskBackend {
    /// 根据键生成文件路径
    fn get_file_path(&self, key: &str) -> PathBuf {
        let safe_key = self.encode_key_to_filename(key);
        self.config.cache_dir.join(safe_key)
    }
    
    /// 将键编码为安全的文件名
    fn encode_key_to_filename(&self, key: &str) -> String {
        // 使用base64编码确保文件名安全
        use base64::{Engine as _, engine::general_purpose};
        general_purpose::URL_SAFE_NO_PAD.encode(key.as_bytes())
    }
    
    /// 从文件名解码键
    fn decode_key_from_filename(&self, filename: &str) -> String {
        use base64::{Engine as _, engine::general_purpose};
        general_purpose::URL_SAFE_NO_PAD.decode(filename)
            .map(|bytes| String::from_utf8_lossy(&bytes).to_string())
            .unwrap_or_else(|_| filename.to_string())
    }
    
    /// 模式匹配
    fn match_pattern(&self, key: &str, pattern: &str) -> bool {
        // 简单的glob模式匹配
        if pattern == "*" {
            return true;
        }
        
        if pattern.contains('*') {
            let regex_pattern = pattern.replace('*', ".*");
            if let Ok(regex) = regex::Regex::new(&regex_pattern) {
                return regex.is_match(key);
            }
        }
        
        key == pattern
    }
    
    /// 设置键的TTL
    async fn set_key_ttl(&self, key: &str, ttl: Duration) -> Result<()> {
        let ttl_file = self.get_ttl_file_path(key);
        let expiry_time = std::time::SystemTime::now() + ttl;
        let timestamp = expiry_time.duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| crate::error::Error::io(format!("计算过期时间失败: {}", e)))?
            .as_secs();
        
        tokio::fs::write(&ttl_file, timestamp.to_string()).await
            .map_err(|e| crate::error::Error::io(format!("写入TTL文件失败: {}", e)))?;
        
        Ok(())
    }
    
    /// 获取键的TTL
    async fn get_key_ttl(&self, key: &str) -> Result<Option<Duration>> {
        let ttl_file = self.get_ttl_file_path(key);
        if !ttl_file.exists() {
            return Ok(None);
        }
        
        let content = tokio::fs::read_to_string(&ttl_file).await
            .map_err(|e| crate::error::Error::io(format!("读取TTL文件失败: {}", e)))?;
        
        let timestamp: u64 = content.trim().parse()
            .map_err(|e| crate::error::Error::io(format!("解析TTL时间戳失败: {}", e)))?;
        
        let expiry_time = std::time::UNIX_EPOCH + Duration::from_secs(timestamp);
        let now = std::time::SystemTime::now();
        
        if now < expiry_time {
            Ok(Some(expiry_time.duration_since(now).unwrap_or_default()))
        } else {
            Ok(Some(Duration::from_secs(0))) // 已过期
        }
    }
    
    /// 移除键的TTL
    async fn remove_key_ttl(&self, key: &str) -> Result<()> {
        let ttl_file = self.get_ttl_file_path(key);
        if ttl_file.exists() {
            tokio::fs::remove_file(&ttl_file).await
                .map_err(|e| crate::error::Error::io(format!("删除TTL文件失败: {}", e)))?;
        }
        Ok(())
    }
    
    /// 清理所有TTL数据
    async fn clear_all_ttl(&self) -> Result<()> {
        let ttl_dir = self.get_ttl_dir();
        if ttl_dir.exists() {
            tokio::fs::remove_dir_all(&ttl_dir).await
                .map_err(|e| crate::error::Error::io(format!("清空TTL目录失败: {}", e)))?;
        }
        Ok(())
    }
    
    /// 获取TTL文件路径
    fn get_ttl_file_path(&self, key: &str) -> PathBuf {
        let safe_key = self.encode_key_to_filename(key);
        self.get_ttl_dir().join(format!("{}.ttl", safe_key))
    }
    
    /// 获取TTL目录
    fn get_ttl_dir(&self) -> PathBuf {
        self.config.cache_dir.join(".ttl")
    }
}

/// Redis后端实现
pub struct RedisBackend {
    config: BackendConfig,
    // 添加Redis连接池字段（模拟）
    connection_pool: Arc<Mutex<Vec<String>>>, // 在生产环境中应使用实际的Redis连接
    metrics: Arc<RwLock<RedisMetrics>>,
    is_connected: Arc<AtomicBool>,
}

/// Redis后端统计指标
#[derive(Debug, Clone)]
pub struct RedisMetrics {
    pub total_commands: u64,
    pub successful_commands: u64,
    pub failed_commands: u64,
    pub connection_errors: u64,
    pub key_hits: u64,
    pub key_misses: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub connection_pool_size: usize,
    pub active_connections: usize,
    pub last_error: Option<String>,
    pub uptime: Instant,
}

impl Default for RedisMetrics {
    fn default() -> Self {
        Self {
            total_commands: 0,
            successful_commands: 0,
            failed_commands: 0,
            connection_errors: 0,
            key_hits: 0,
            key_misses: 0,
            bytes_read: 0,
            bytes_written: 0,
            connection_pool_size: 0,
            active_connections: 0,
            last_error: None,
            uptime: Instant::now(),
        }
    }
}

impl RedisBackend {
    pub fn new(config: BackendConfig) -> Result<Self> {
        let max_connections = config.max_connections.unwrap_or(10);
        let connection_pool = Arc::new(Mutex::new(Vec::with_capacity(max_connections)));
        
        // 初始化连接池（模拟）
        {
            let mut pool = connection_pool.lock()
                .map_err(|_| crate::error::Error::locks_poison("Redis连接池锁被污染"))?;
                
            for i in 0..max_connections {
                // 在实际实现中，这里应该创建真实的Redis连接
                pool.push(format!("redis_connection_{}", i));
            }
        }
        
        let mut metrics = RedisMetrics::default();
        metrics.connection_pool_size = max_connections;
        metrics.active_connections = max_connections;
        
        Ok(Self { 
            config,
            connection_pool,
            metrics: Arc::new(RwLock::new(metrics)),
            is_connected: Arc::new(AtomicBool::new(true)),
        })
    }
    
    /// 获取连接（模拟）
    async fn get_connection(&self) -> Result<String> {
        if !self.is_connected.load(Ordering::Relaxed) {
            {
                let mut metrics = self.metrics.write().await;
                metrics.connection_errors += 1;
                metrics.last_error = Some("Redis连接已断开".to_string());
            }
            return Err(crate::error::Error::cache("Redis连接不可用"));
        }
        
        let pool = self.connection_pool.lock().await;
        
        if pool.is_empty() {
            {
                let mut metrics = self.metrics.write().await;
                metrics.connection_errors += 1;
                metrics.last_error = Some("连接池已耗尽".to_string());
            }
            return Err(crate::error::Error::cache("连接池已耗尽"));
        }
        
        Ok(pool[0].clone())
    }
    
    /// 释放连接（模拟）
    async fn return_connection(&self, _connection: String) -> Result<()> {
        // 在实际实现中，这里应该将连接返回到池中
        Ok(())
    }
    
    /// 记录成功的操作
    async fn record_success(&self, bytes_read: usize, bytes_written: usize, hit: bool) {
        let mut metrics = self.metrics.write().await;
        metrics.total_commands += 1;
        metrics.successful_commands += 1;
        metrics.bytes_read += bytes_read as u64;
        metrics.bytes_written += bytes_written as u64;
        if hit {
            metrics.key_hits += 1;
        } else {
            metrics.key_misses += 1;
        }
    }
    
    /// 记录失败的操作
    async fn record_failure(&self, error: &str) {
        let mut metrics = self.metrics.write().await;
        metrics.total_commands += 1;
        metrics.failed_commands += 1;
        metrics.last_error = Some(error.to_string());
    }
    
    /// 构建完整的Redis键名
    fn build_key(&self, key: &str) -> String {
        format!("{}{}", self.config.key_prefix, key)
    }
    
    /// 模拟Redis命令执行
    async fn execute_redis_command(&self, command: &str, key: &str, value: Option<&[u8]>) -> Result<RedisCommandResult> {
        let connection = self.get_connection().await?;
        
        // 模拟网络延迟
        tokio::time::sleep(Duration::from_millis(1)).await;
        
        let full_key = self.build_key(key);
        
        // 模拟Redis命令执行
        let result = match command {
            "GET" => {
                // 模拟：根据键名哈希判断是否存在值
                let key_hash = full_key.chars().map(|c| c as u32).sum::<u32>();
                if key_hash % 3 == 0 {
                    // 模拟命中
                    let mock_value = format!("cached_value_for_{}", key).into_bytes();
                    RedisCommandResult::Value(Some(mock_value))
                } else {
                    RedisCommandResult::Value(None)
                }
            },
            "SET" => {
                if value.is_some() {
                    RedisCommandResult::Ok
                } else {
                    return Err(crate::error::Error::cache("SET命令需要值"));
                }
            },
            "DEL" => {
                let key_hash = full_key.chars().map(|c| c as u32).sum::<u32>();
                RedisCommandResult::Integer(if key_hash % 2 == 0 { 1 } else { 0 })
            },
            "EXISTS" => {
                let key_hash = full_key.chars().map(|c| c as u32).sum::<u32>();
                RedisCommandResult::Integer(if key_hash % 3 == 0 { 1 } else { 0 })
            },
            "EXPIRE" => {
                RedisCommandResult::Integer(1)
            },
            "FLUSHALL" => {
                RedisCommandResult::Ok
            },
            "DBSIZE" => {
                // 模拟数据库大小
                RedisCommandResult::Integer(42)
            },
            "KEYS" => {
                // 模拟返回一些键
                let mock_keys = vec![
                    format!("{}key1", self.config.key_prefix),
                    format!("{}key2", self.config.key_prefix),
                    format!("{}key3", self.config.key_prefix),
                ];
                RedisCommandResult::Array(mock_keys)
            },
            "MGET" => {
                // 模拟批量获取
                RedisCommandResult::MultiValue(HashMap::new())
            },
            "MSET" => {
                RedisCommandResult::Ok
            },
            "PING" => {
                RedisCommandResult::Status("PONG".to_string())
            },
            _ => {
                return Err(crate::error::Error::cache(&format!("不支持的Redis命令: {}", command)));
            }
        };
        
        self.return_connection(connection).await?;
        Ok(result)
    }
}

/// Redis命令执行结果
#[derive(Debug)]
enum RedisCommandResult {
    Ok,
    Value(Option<Vec<u8>>),
    Integer(i64),
    Status(String),
    Array(Vec<String>),
    MultiValue(HashMap<String, Option<Vec<u8>>>),
}

#[async_trait]
impl CacheBackend for RedisBackend {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        match self.execute_redis_command("GET", key, None).await {
            Ok(RedisCommandResult::Value(value)) => {
                let bytes_read = value.as_ref().map(|v| v.len()).unwrap_or(0);
                let hit = value.is_some();
                self.record_success(bytes_read, 0, hit).await;
                Ok(value)
            },
            Ok(_) => {
                self.record_failure("GET命令返回了意外的结果类型").await;
                Err(crate::error::Error::cache("GET命令返回了意外的结果类型"))
            },
            Err(e) => {
                self.record_failure(&e.to_string()).await;
                Err(e)
            }
        }
    }
    
    async fn set(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> Result<()> {
        // 首先执行SET命令
        match self.execute_redis_command("SET", key, Some(value)).await {
            Ok(RedisCommandResult::Ok) => {
                // 如果有TTL，设置过期时间
                if ttl.is_some() {
                    if let Err(e) = self.execute_redis_command("EXPIRE", key, None).await {
                        warn!("设置TTL失败: {}", e);
                        // 不要因为TTL失败而使整个SET操作失败
                    }
                }
                
                self.record_success(0, value.len(), false).await;
                Ok(())
            },
            Ok(_) => {
                self.record_failure("SET命令返回了意外的结果类型").await;
                Err(crate::error::Error::cache("SET命令返回了意外的结果类型"))
            },
            Err(e) => {
                self.record_failure(&e.to_string()).await;
                Err(e)
            }
        }
    }
    
    async fn delete(&self, key: &str) -> Result<bool> {
        match self.execute_redis_command("DEL", key, None).await {
            Ok(RedisCommandResult::Integer(count)) => {
                let deleted = count > 0;
                self.record_success(0, 0, false).await;
                Ok(deleted)
            },
            Ok(_) => {
                self.record_failure("DEL命令返回了意外的结果类型").await;
                Err(crate::error::Error::cache("DEL命令返回了意外的结果类型"))
            },
            Err(e) => {
                self.record_failure(&e.to_string()).await;
                Err(e)
            }
        }
    }
    
    async fn exists(&self, key: &str) -> Result<bool> {
        match self.execute_redis_command("EXISTS", key, None).await {
            Ok(RedisCommandResult::Integer(exists)) => {
                self.record_success(0, 0, false).await;
                Ok(exists > 0)
            },
            Ok(_) => {
                self.record_failure("EXISTS命令返回了意外的结果类型").await;
                Err(crate::error::Error::cache("EXISTS命令返回了意外的结果类型"))
            },
            Err(e) => {
                self.record_failure(&e.to_string()).await;
                Err(e)
            }
        }
    }
    
    async fn expire(&self, key: &str, _ttl: Duration) -> Result<bool> {
        match self.execute_redis_command("EXPIRE", key, None).await {
            Ok(RedisCommandResult::Integer(result)) => {
                self.record_success(0, 0, false).await;
                Ok(result > 0)
            },
            Ok(_) => {
                self.record_failure("EXPIRE命令返回了意外的结果类型").await;
                Err(crate::error::Error::cache("EXPIRE命令返回了意外的结果类型"))
            },
            Err(e) => {
                self.record_failure(&e.to_string()).await;
                Err(e)
            }
        }
    }
    
    async fn clear(&self) -> Result<()> {
        match self.execute_redis_command("FLUSHALL", "", None).await {
            Ok(RedisCommandResult::Ok) => {
                self.record_success(0, 0, false);
                Ok(())
            },
            Ok(_) => {
                self.record_failure("FLUSHALL命令返回了意外的结果类型");
                Err(crate::error::Error::cache("FLUSHALL命令返回了意外的结果类型"))
            },
            Err(e) => {
                self.record_failure(&e.to_string());
                Err(e)
            }
        }
    }
    
    async fn size(&self) -> Result<usize> {
        match self.execute_redis_command("DBSIZE", "", None).await {
            Ok(RedisCommandResult::Integer(size)) => {
                self.record_success(0, 0, false);
                Ok(size as usize)
            },
            Ok(_) => {
                self.record_failure("DBSIZE命令返回了意外的结果类型");
                Err(crate::error::Error::cache("DBSIZE命令返回了意外的结果类型"))
            },
            Err(e) => {
                self.record_failure(&e.to_string());
                Err(e)
            }
        }
    }
    
    async fn keys(&self, pattern: Option<&str>) -> Result<Vec<String>> {
        let search_pattern = pattern.unwrap_or("*");
        match self.execute_redis_command("KEYS", search_pattern, None).await {
            Ok(RedisCommandResult::Array(keys)) => {
                // 过滤掉不匹配前缀的键
                let filtered_keys: Vec<String> = keys.into_iter()
                    .filter(|k| k.starts_with(&self.config.key_prefix))
                    .map(|k| k.strip_prefix(&self.config.key_prefix).unwrap_or(&k).to_string())
                    .collect();
                    
                self.record_success(0, 0, false);
                Ok(filtered_keys)
            },
            Ok(_) => {
                self.record_failure("KEYS命令返回了意外的结果类型");
                Err(crate::error::Error::cache("KEYS命令返回了意外的结果类型"))
            },
            Err(e) => {
                self.record_failure(&e.to_string());
                Err(e)
            }
        }
    }
    
    async fn mget(&self, keys: &[String]) -> Result<HashMap<String, Option<Vec<u8>>>> {
        let mut result = HashMap::new();
        
        // 在生产环境中，这应该是一个真正的MGET命令
        // 这里我们逐个获取以保持一致性
        for key in keys {
            match self.get(key).await {
                Ok(value) => {
                    result.insert(key.clone(), value);
                },
                Err(_) => {
                    // 记录错误但继续处理其他键
                    result.insert(key.clone(), None);
                }
            }
        }
        
        Ok(result)
    }
    
    async fn mset(&self, data: HashMap<String, Vec<u8>>, ttl: Option<Duration>) -> Result<()> {
        // 在生产环境中，这应该是一个真正的MSET命令
        // 这里我们逐个设置以保持一致性
        for (key, value) in data {
            if let Err(e) = self.set(&key, &value, ttl).await {
                // 如果任何一个设置失败，记录错误但继续
                warn!("批量设置键 {} 失败: {}", key, e);
            }
        }
        
        Ok(())
    }
    
    async fn mdel(&self, keys: &[String]) -> Result<usize> {
        let mut deleted_count = 0;
        
        // 在生产环境中，这应该是一个真正的DEL命令支持多个键
        // 这里我们逐个删除以保持一致性
        for key in keys {
            match self.delete(key).await {
                Ok(true) => deleted_count += 1,
                Ok(false) => {}, // 键不存在，不计入删除数
                Err(_) => {
                    // 记录错误但继续处理其他键
                    warn!("删除键 {} 时出错", key);
                }
            }
        }
        
        Ok(deleted_count)
    }
    
    async fn health_check(&self) -> Result<bool> {
        match self.execute_redis_command("PING", "", None).await {
            Ok(RedisCommandResult::Status(response)) => {
                let healthy = response == "PONG";
                if !healthy {
                    self.is_connected.store(false, Ordering::Relaxed);
                }
                Ok(healthy)
            },
            Ok(_) => {
                self.record_failure("PING命令返回了意外的结果类型");
                self.is_connected.store(false, Ordering::Relaxed);
                Ok(false)
            },
            Err(e) => {
                self.record_failure(&e.to_string());
                self.is_connected.store(false, Ordering::Relaxed);
                Ok(false)
            }
        }
    }
    
    async fn stats(&self) -> Result<BackendStats> {
        let metrics = self.metrics.read()
            .map_err(|_| crate::error::Error::locks_poison("Redis指标锁被污染"))?;
            
        let hit_rate = if metrics.total_commands > 0 {
            metrics.key_hits as f64 / metrics.total_commands as f64
        } else {
            0.0
        };
        
        let uptime_duration = metrics.uptime.elapsed();
        
        Ok(BackendStats {
            total_operations: metrics.total_commands,
            successful_operations: metrics.successful_commands,
            failed_operations: metrics.failed_commands,
            hit_rate,
            total_keys: 0, // 需要调用DBSIZE获取，为了避免额外开销暂时设为0
            memory_usage: 0, // Redis内存使用信息，需要INFO命令获取
            uptime_seconds: uptime_duration.as_secs(),
            connection_count: metrics.active_connections,
            last_error: metrics.last_error.clone(),
        })
    }
    
    fn backend_type(&self) -> BackendType {
        BackendType::Redis
    }
    
    async fn close(&self) -> Result<()> {
        self.is_connected.store(false, Ordering::Relaxed);
        
        // 清空连接池
        {
            let mut pool = self.connection_pool.lock()
                .map_err(|_| crate::error::Error::locks_poison("Redis连接池锁被污染"))?;
            pool.clear();
        }
        
        // 重置指标中的连接数
        {
            let mut metrics = self.metrics.write()
                .map_err(|_| crate::error::Error::locks_poison("Redis指标锁被污染"))?;
            metrics.active_connections = 0;
        }
        
        info!("Redis后端已关闭");
        Ok(())
    }
}

/// Memcached后端实现
pub struct MemcachedBackend {
    config: BackendConfig,
    // 添加Memcached连接池字段（模拟）
    connection_pool: Arc<Mutex<Vec<String>>>, // 在生产环境中应使用实际的Memcached连接
    metrics: Arc<RwLock<MemcachedMetrics>>,
    is_connected: Arc<AtomicBool>,
}

/// Memcached后端统计指标
#[derive(Debug, Clone)]
pub struct MemcachedMetrics {
    pub total_commands: u64,
    pub successful_commands: u64,
    pub failed_commands: u64,
    pub connection_errors: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    pub connection_pool_size: usize,
    pub active_connections: usize,
    pub evictions: u64,
    pub last_error: Option<String>,
    pub uptime: Instant,
}

impl Default for MemcachedMetrics {
    fn default() -> Self {
        Self {
            total_commands: 0,
            successful_commands: 0,
            failed_commands: 0,
            connection_errors: 0,
            cache_hits: 0,
            cache_misses: 0,
            bytes_read: 0,
            bytes_written: 0,
            connection_pool_size: 0,
            active_connections: 0,
            evictions: 0,
            last_error: None,
            uptime: Instant::now(),
        }
    }
}

impl MemcachedBackend {
    pub fn new(config: BackendConfig) -> Result<Self> {
        let max_connections = config.max_connections.unwrap_or(8);
        let connection_pool = Arc::new(Mutex::new(Vec::with_capacity(max_connections)));
        
        // 初始化连接池（模拟）
        {
            let mut pool = connection_pool.lock()
                .map_err(|_| crate::error::Error::locks_poison("Memcached连接池锁被污染"))?;
                
            for i in 0..max_connections {
                // 在实际实现中，这里应该创建真实的Memcached连接
                pool.push(format!("memcached_connection_{}", i));
            }
        }
        
        let mut metrics = MemcachedMetrics::default();
        metrics.connection_pool_size = max_connections;
        metrics.active_connections = max_connections;
        
        Ok(Self { 
            config,
            connection_pool,
            metrics: Arc::new(RwLock::new(metrics)),
            is_connected: Arc::new(AtomicBool::new(true)),
        })
    }
    
    /// 获取连接（模拟）
    async fn get_connection(&self) -> Result<String> {
        if !self.is_connected.load(Ordering::Relaxed) {
            {
                let mut metrics = self.metrics.write().await;
                metrics.connection_errors += 1;
                metrics.last_error = Some("Memcached连接已断开".to_string());
            }
            return Err(crate::error::Error::cache("Memcached连接不可用"));
        }
        
        let pool = self.connection_pool.lock().await;
        
        if pool.is_empty() {
            {
                let mut metrics = self.metrics.write().await;
                metrics.connection_errors += 1;
                metrics.last_error = Some("连接池已耗尽".to_string());
            }
            return Err(crate::error::Error::cache("连接池已耗尽"));
        }
        
        Ok(pool[0].clone())
    }
    
    /// 释放连接（模拟）
    async fn return_connection(&self, _connection: String) -> Result<()> {
        // 在实际实现中，这里应该将连接返回到池中
        Ok(())
    }
    
    /// 记录成功的操作
    async fn record_success(&self, bytes_read: usize, bytes_written: usize, hit: bool) {
        let mut metrics = self.metrics.write().await;
        metrics.total_commands += 1;
        metrics.successful_commands += 1;
        metrics.bytes_read += bytes_read as u64;
        metrics.bytes_written += bytes_written as u64;
        if hit {
            metrics.cache_hits += 1;
        } else {
            metrics.cache_misses += 1;
        }
    }
    
    /// 记录失败的操作
    async fn record_failure(&self, error: &str) {
        let mut metrics = self.metrics.write().await;
        metrics.total_commands += 1;
        metrics.failed_commands += 1;
        metrics.last_error = Some(error.to_string());
    }
    
    /// 构建完整的Memcached键名
    fn build_key(&self, key: &str) -> String {
        format!("{}{}", self.config.key_prefix, key)
    }
    
    /// 模拟Memcached命令执行
    async fn execute_memcached_command(&self, command: &str, key: &str, value: Option<&[u8]>, _expiration: Option<Duration>) -> Result<MemcachedCommandResult> {
        let connection = self.get_connection().await?;
        
        // 模拟网络延迟（Memcached通常比Redis稍快）
        tokio::time::sleep(Duration::from_millis(1)).await;
        
        let full_key = self.build_key(key);
        
        // 模拟Memcached命令执行
        let result = match command {
            "get" => {
                // 模拟：根据键名哈希判断是否存在值
                let key_hash = full_key.chars().map(|c| c as u32).sum::<u32>();
                if key_hash % 4 == 0 {
                    // 模拟命中
                    let mock_value = format!("memcached_value_for_{}", key).into_bytes();
                    MemcachedCommandResult::Value(Some(mock_value))
                } else {
                    MemcachedCommandResult::Value(None)
                }
            },
            "set" => {
                if value.is_some() {
                    // 模拟可能的驱逐
                    let key_hash = full_key.chars().map(|c| c as u32).sum::<u32>();
                    if key_hash % 20 == 0 {
                        // 记录驱逐事件
                        let mut metrics = self.metrics.write().await;
                        metrics.evictions += 1;
                    }
                    MemcachedCommandResult::Stored
                } else {
                    return Err(crate::error::Error::cache("set命令需要值"));
                }
            },
            "delete" => {
                let key_hash = full_key.chars().map(|c| c as u32).sum::<u32>();
                MemcachedCommandResult::Boolean(key_hash % 2 == 0)
            },
            "touch" => {
                // Memcached的touch命令用于更新过期时间
                let key_hash = full_key.chars().map(|c| c as u32).sum::<u32>();
                MemcachedCommandResult::Boolean(key_hash % 3 == 0)
            },
            "flush_all" => {
                MemcachedCommandResult::Ok
            },
            "stats" => {
                // 模拟统计信息
                let mut stats = HashMap::new();
                stats.insert("curr_items".to_string(), "42".to_string());
                stats.insert("total_items".to_string(), "1000".to_string());
                stats.insert("bytes".to_string(), "123456".to_string());
                stats.insert("curr_connections".to_string(), "5".to_string());
                stats.insert("get_hits".to_string(), "800".to_string());
                stats.insert("get_misses".to_string(), "200".to_string());
                stats.insert("evictions".to_string(), "10".to_string());
                MemcachedCommandResult::Stats(stats)
            },
            "version" => {
                MemcachedCommandResult::Version("1.6.0".to_string())
            },
            _ => {
                return Err(crate::error::Error::cache(&format!("不支持的Memcached命令: {}", command)));
            }
        };
        
        self.return_connection(connection).await?;
        Ok(result)
    }
}

/// Memcached命令执行结果
#[derive(Debug)]
enum MemcachedCommandResult {
    Ok,
    Stored,
    NotStored,
    Value(Option<Vec<u8>>),
    Boolean(bool),
    Version(String),
    Stats(HashMap<String, String>),
}

#[async_trait]
impl CacheBackend for MemcachedBackend {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        match self.execute_memcached_command("get", key, None, None).await {
            Ok(MemcachedCommandResult::Value(value)) => {
                let bytes_read = value.as_ref().map(|v| v.len()).unwrap_or(0);
                let hit = value.is_some();
                self.record_success(bytes_read, 0, hit).await;
                Ok(value)
            },
            Ok(_) => {
                self.record_failure("get命令返回了意外的结果类型").await;
                Err(crate::error::Error::cache("get命令返回了意外的结果类型"))
            },
            Err(e) => {
                self.record_failure(&e.to_string()).await;
                Err(e)
            }
        }
    }
    
    async fn set(&self, key: &str, value: &[u8], ttl: Option<Duration>) -> Result<()> {
        match self.execute_memcached_command("set", key, Some(value), ttl).await {
            Ok(MemcachedCommandResult::Stored) => {
                self.record_success(0, value.len(), false).await;
                Ok(())
            },
            Ok(MemcachedCommandResult::NotStored) => {
                self.record_failure("Memcached存储失败（可能是内存不足）").await;
                Err(crate::error::Error::cache("Memcached存储失败"))
            },
            Ok(_) => {
                self.record_failure("set命令返回了意外的结果类型").await;
                Err(crate::error::Error::cache("set命令返回了意外的结果类型"))
            },
            Err(e) => {
                self.record_failure(&e.to_string()).await;
                Err(e)
            }
        }
    }
    
    async fn delete(&self, key: &str) -> Result<bool> {
        match self.execute_memcached_command("delete", key, None, None).await {
            Ok(MemcachedCommandResult::Boolean(deleted)) => {
                self.record_success(0, 0, false);
                Ok(deleted)
            },
            Ok(_) => {
                self.record_failure("delete命令返回了意外的结果类型");
                Err(crate::error::Error::cache("delete命令返回了意外的结果类型"))
            },
            Err(e) => {
                self.record_failure(&e.to_string());
                Err(e)
            }
        }
    }
    
    async fn exists(&self, key: &str) -> Result<bool> {
        // Memcached没有exists命令，我们通过get来模拟
        match self.get(key).await {
            Ok(Some(_)) => Ok(true),
            Ok(None) => Ok(false),
            Err(_) => Ok(false), // 出错时认为不存在
        }
    }
    
    async fn expire(&self, key: &str, ttl: Duration) -> Result<bool> {
        // Memcached使用touch命令更新过期时间
        match self.execute_memcached_command("touch", key, None, Some(ttl)).await {
            Ok(MemcachedCommandResult::Boolean(touched)) => {
                self.record_success(0, 0, false);
                Ok(touched)
            },
            Ok(_) => {
                self.record_failure("touch命令返回了意外的结果类型");
                Err(crate::error::Error::cache("touch命令返回了意外的结果类型"))
            },
            Err(e) => {
                self.record_failure(&e.to_string());
                Err(e)
            }
        }
    }
    
    async fn clear(&self) -> Result<()> {
        match self.execute_memcached_command("flush_all", "", None, None).await {
            Ok(MemcachedCommandResult::Ok) => {
                self.record_success(0, 0, false);
                Ok(())
            },
            Ok(_) => {
                self.record_failure("flush_all命令返回了意外的结果类型");
                Err(crate::error::Error::cache("flush_all命令返回了意外的结果类型"))
            },
            Err(e) => {
                self.record_failure(&e.to_string());
                Err(e)
            }
        }
    }
    
    async fn size(&self) -> Result<usize> {
        match self.execute_memcached_command("stats", "", None, None).await {
            Ok(MemcachedCommandResult::Stats(stats)) => {
                let size = stats.get("curr_items")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                self.record_success(0, 0, false);
                Ok(size)
            },
            Ok(_) => {
                self.record_failure("stats命令返回了意外的结果类型");
                Err(crate::error::Error::cache("stats命令返回了意外的结果类型"))
            },
            Err(e) => {
                self.record_failure(&e.to_string());
                Err(e)
            }
        }
    }
    
    async fn keys(&self, _pattern: Option<&str>) -> Result<Vec<String>> {
        // Memcached不支持列出所有键的功能
        // 这是一个已知的限制，返回空列表并记录警告
        warn!("Memcached不支持列出键的功能，返回空列表");
        self.record_success(0, 0, false);
        Ok(vec![])
    }
    
    async fn mget(&self, keys: &[String]) -> Result<HashMap<String, Option<Vec<u8>>>> {
        let mut result = HashMap::new();
        
        // Memcached支持真正的multi-get，但这里我们模拟逐个获取
        for key in keys {
            match self.get(key).await {
                Ok(value) => {
                    result.insert(key.clone(), value);
                },
                Err(_) => {
                    // 记录错误但继续处理其他键
                    result.insert(key.clone(), None);
                }
            }
        }
        
        Ok(result)
    }
    
    async fn mset(&self, data: HashMap<String, Vec<u8>>, ttl: Option<Duration>) -> Result<()> {
        // Memcached不支持原子的multi-set，我们逐个设置
        for (key, value) in data {
            if let Err(e) = self.set(&key, &value, ttl).await {
                // 如果任何一个设置失败，记录错误但继续
                warn!("批量设置键 {} 失败: {}", key, e);
            }
        }
        
        Ok(())
    }
    
    async fn mdel(&self, keys: &[String]) -> Result<usize> {
        let mut deleted_count = 0;
        
        // Memcached支持批量删除，但这里我们模拟逐个删除
        for key in keys {
            match self.delete(key).await {
                Ok(true) => deleted_count += 1,
                Ok(false) => {}, // 键不存在，不计入删除数
                Err(_) => {
                    // 记录错误但继续处理其他键
                    warn!("删除键 {} 时出错", key);
                }
            }
        }
        
        Ok(deleted_count)
    }
    
    async fn health_check(&self) -> Result<bool> {
        match self.execute_memcached_command("version", "", None, None).await {
            Ok(MemcachedCommandResult::Version(_)) => {
                Ok(true)
            },
            Ok(_) => {
                self.record_failure("version命令返回了意外的结果类型");
                self.is_connected.store(false, Ordering::Relaxed);
                Ok(false)
            },
            Err(e) => {
                self.record_failure(&e.to_string());
                self.is_connected.store(false, Ordering::Relaxed);
                Ok(false)
            }
        }
    }
    
    async fn stats(&self) -> Result<BackendStats> {
        let metrics = self.metrics.read()
            .map_err(|_| crate::error::Error::locks_poison("Memcached指标锁被污染"))?;
            
        let hit_rate = if metrics.total_commands > 0 {
            metrics.cache_hits as f64 / metrics.total_commands as f64
        } else {
            0.0
        };
        
        let uptime_duration = metrics.uptime.elapsed();
        
        Ok(BackendStats {
            total_operations: metrics.total_commands,
            successful_operations: metrics.successful_commands,
            failed_operations: metrics.failed_commands,
            hit_rate,
            total_keys: 0, // 需要调用stats获取，为了避免额外开销暂时设为0
            memory_usage: 0, // Memcached内存使用信息，需要stats命令获取
            uptime_seconds: uptime_duration.as_secs(),
            connection_count: metrics.active_connections,
            last_error: metrics.last_error.clone(),
        })
    }
    
    fn backend_type(&self) -> BackendType {
        BackendType::Memcached
    }
    
    async fn close(&self) -> Result<()> {
        self.is_connected.store(false, Ordering::Relaxed);
        
        // 清空连接池
        {
            let mut pool = self.connection_pool.lock()
                .map_err(|_| crate::error::Error::locks_poison("Memcached连接池锁被污染"))?;
            pool.clear();
        }
        
        // 重置指标中的连接数
        {
            let mut metrics = self.metrics.write()
                .map_err(|_| crate::error::Error::locks_poison("Memcached指标锁被污染"))?;
            metrics.active_connections = 0;
        }
        
        info!("Memcached后端已关闭");
        Ok(())
    }
} 