//! 键值存储引擎实现

use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::Duration;
use serde::{Serialize, Deserialize};
use crate::error::{Error, Result};

/// 键值存储引擎统一接口
pub trait KeyValueStorageEngine: Send + Sync {
    /// 设置键值对
    fn set(&self, key: &str, value: &[u8]) -> Result<()>;
    
    /// 获取键对应的值
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    
    /// 删除键
    fn delete(&self, key: &str) -> Result<()>;
    
    /// 检查键是否存在
    fn exists(&self, key: &str) -> Result<bool>;
    
    /// 获取所有键
    fn keys(&self) -> Result<Vec<String>>;
    
    /// 批量操作
    fn batch(&self, operations: Vec<BatchOperation>) -> Result<()>;
    
    /// 按模式获取键
    fn get_keys_with_pattern(&self, pattern: &str) -> Result<Vec<String>>;
    
    /// 获取TTL
    fn get_ttl(&self, key: &str) -> Result<Option<u64>>;
    
    /// 设置TTL
    fn set_ttl(&self, key: &str, ttl: u64) -> Result<bool>;
    
    /// 延长TTL
    fn extend_ttl(&self, key: &str, seconds: u64) -> Result<bool>;
    
    /// 创建快照
    fn create_snapshot(&self, path: &str) -> Result<()>;
    
    /// 从快照恢复
    fn restore_from_snapshot(&self, path: &str) -> Result<()>;
    
    /// 获取存储信息
    fn get_storage_info(&self) -> Result<HashMap<String, String>>;
    
    /// 清空所有数据
    fn clear(&self) -> Result<()>;
    
    /// 获取引擎名称
    fn name(&self) -> &'static str;
    
    /// 获取详细信息
    fn get_info(&self) -> Result<HashMap<String, String>>;
    
    /// 关闭存储引擎
    fn close(&self) -> Result<()>;
}

/// 序列化格式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerializationFormat {
    /// Json格式
    Json,
    /// 二进制格式
    Bincode,
    /// 消息包格式
    MessagePack,
}

/// 键值存储引擎枚举
#[derive(Debug, Clone)]
pub enum KeyValueStorageEngineEnum {
    /// 内存存储引擎
    Memory(MemoryKeyValueStorage),
    /// RocksDB存储引擎
    RocksDB {
        engine: Arc<RocksDBEngine>,
        options: RocksDBOptions,
    },
    /// Redis存储引擎
    Redis {
        client: Arc<RedisClient>,
        options: RedisOptions,
    },
}

/// RocksDB 配置选项
#[derive(Debug, Clone)]
pub struct RocksDBOptions {
    pub create_if_missing: bool,
    pub use_compression: bool,
    pub cache_size_mb: usize,
    pub max_open_files: i32,
    pub write_buffer_size: usize,
    pub max_write_buffer_number: usize,
    pub target_file_size_base: u64,
    pub max_background_jobs: i32,
}

impl Default for RocksDBOptions {
    fn default() -> Self {
        Self {
            create_if_missing: true,
            use_compression: true,
            cache_size_mb: 128,
            max_open_files: 1000,
            write_buffer_size: 64 * 1024 * 1024, // 64MB
            max_write_buffer_number: 3,
            target_file_size_base: 64 * 1024 * 1024, // 64MB
            max_background_jobs: 4,
        }
    }
}

/// Redis 配置选项
#[derive(Debug, Clone)]
pub struct RedisOptions {
    pub database: u8,
    pub connection_timeout_ms: u64,
    pub response_timeout_ms: u64,
    pub retry_max_attempts: usize,
    pub retry_delay_ms: u64,
    pub max_connections: usize,
}

impl Default for RedisOptions {
    fn default() -> Self {
        Self {
            database: 0,
            connection_timeout_ms: 5000,
            response_timeout_ms: 10000,
            retry_max_attempts: 3,
            retry_delay_ms: 100,
            max_connections: 100,
        }
    }
}

/// RocksDB存储引擎实现
#[derive(Clone)]
pub struct RocksDBEngine {
    db: Arc<Mutex<Option<rocksdb::DB>>>,
    path: PathBuf,
    options: RocksDBOptions,
}

impl std::fmt::Debug for RocksDBEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocksDBEngine")
            .field("path", &self.path)
            .field("options", &self.options)
            .finish()
    }
}

impl RocksDBEngine {
    /// 创建新的RocksDB引擎
    pub fn new(path: impl AsRef<Path>, options: RocksDBOptions) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        
        // 创建目录
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                Error::Storage(format!("创建RocksDB目录失败: {}", e))
            })?;
        }
        
        let engine = Self {
            db: Arc::new(Mutex::new(None)),
            path,
            options,
        };
        
        engine.initialize()?;
        Ok(engine)
    }

    /// 初始化数据库连接
    fn initialize(&self) -> Result<()> {
        let mut db_opts = rocksdb::Options::default();
        db_opts.create_if_missing(self.options.create_if_missing);
        db_opts.set_max_open_files(self.options.max_open_files);
        db_opts.set_write_buffer_size(self.options.write_buffer_size);
        db_opts.set_max_write_buffer_number(self.options.max_write_buffer_number as i32);
        db_opts.set_target_file_size_base(self.options.target_file_size_base);
        db_opts.set_max_background_jobs(self.options.max_background_jobs);
        
        // 设置压缩
        if self.options.use_compression {
            db_opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        }
        
        // 设置缓存
        let cache = rocksdb::Cache::new_lru_cache(self.options.cache_size_mb * 1024 * 1024);
        
        let mut block_opts = rocksdb::BlockBasedOptions::default();
        block_opts.set_block_cache(&cache);
        db_opts.set_block_based_table_factory(&block_opts);
        
        // 打开数据库
        let db = rocksdb::DB::open(&db_opts, &self.path).map_err(|e| {
            Error::Storage(format!("打开RocksDB失败: {}", e))
        })?;
        
        *self.db.lock().unwrap() = Some(db);
        Ok(())
    }
    
    /// 获取数据库引用
    fn with_db<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&rocksdb::DB) -> Result<R>,
    {
        let db_guard = self.db.lock().unwrap();
        match db_guard.as_ref() {
            Some(db) => f(db),
            None => Err(Error::Storage("RocksDB未初始化".to_string())),
        }
    }
    
    /// 获取值
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        self.with_db(|db| {
            db.get(key.as_bytes()).map_err(|e| {
                Error::Storage(format!("RocksDB获取失败: {}", e))
            })
        })
    }
    
    /// 设置值
    pub fn put(&self, key: &str, value: &[u8]) -> Result<()> {
        self.with_db(|db| {
            db.put(key.as_bytes(), value).map_err(|e| {
                Error::Storage(format!("RocksDB存储失败: {}", e))
            })
        })
    }
    
    /// 删除键
    pub fn delete(&self, key: &str) -> Result<()> {
        self.with_db(|db| {
            db.delete(key.as_bytes()).map_err(|e| {
                Error::Storage(format!("RocksDB删除失败: {}", e))
            })
        })
    }
    
    /// 检查键是否存在
    pub fn exists(&self, key: &str) -> Result<bool> {
        self.get(key).map(|opt| opt.is_some())
    }
    
    /// 获取所有键
    pub fn keys(&self) -> Result<Vec<String>> {
        self.with_db(|db| {
            let mut keys = Vec::new();
            let iter = db.iterator(rocksdb::IteratorMode::Start);
            
            for item in iter {
                match item {
                    Ok((key, _)) => {
                        let key_str = String::from_utf8_lossy(&key).to_string();
                        keys.push(key_str);
                    }
                    Err(e) => return Err(Error::Storage(format!("RocksDB迭代失败: {}", e))),
                }
            }
            
            Ok(keys)
        })
    }
    
    /// 模式匹配获取键
    pub fn keys_with_pattern(&self, pattern: &str) -> Result<Vec<String>> {
        let all_keys = self.keys()?;
        let mut matching_keys = Vec::new();
        
        for key in all_keys {
            if key.contains(pattern) || wildcard_match(pattern, &key) {
                matching_keys.push(key);
            }
        }
        
        Ok(matching_keys)
    }
    
    /// 批量操作
    pub fn batch(&self, operations: Vec<BatchOperation>) -> Result<()> {
        self.with_db(|db| {
            let mut batch = rocksdb::WriteBatch::default();
            
            for op in operations {
                match op {
                    BatchOperation::Set { key, value } => {
                        batch.put(key.as_bytes(), &value);
                    }
                    BatchOperation::Delete { key } => {
                        batch.delete(key.as_bytes());
                    }
                }
            }
            
            db.write(batch).map_err(|e| {
                Error::Storage(format!("RocksDB批量操作失败: {}", e))
            })
        })
    }
    
    /// 迭代器
    pub fn iter(&self) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.with_db(|db| {
            let mut items = Vec::new();
            let iter = db.iterator(rocksdb::IteratorMode::Start);
            
            for item in iter {
                match item {
                    Ok((key, value)) => items.push((key.to_vec(), value.to_vec())),
                    Err(e) => return Err(Error::Storage(format!("RocksDB迭代失败: {}", e))),
                }
            }
            
            Ok(items)
        })
    }
    
    /// 压缩数据库
    pub fn compact(&self) -> Result<()> {
        self.with_db(|db| {
            db.compact_range(None::<&[u8]>, None::<&[u8]>);
            Ok(())
        })
    }
    
    /// 获取数据库统计信息
    pub fn get_property(&self, property: &str) -> Result<Option<String>> {
        self.with_db(|db| {
            Ok(db.property_value(property).unwrap_or(None))
        })
    }
}

/// Redis客户端实现
#[derive(Clone)]
pub struct RedisClient {
    connection_pool: Arc<RwLock<Vec<redis::Connection>>>,
    config: RedisOptions,
    url: String,
}

impl std::fmt::Debug for RedisClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedisClient")
            .field("url", &self.url)
            .field("config", &self.config)
            .finish()
    }
}

impl RedisClient {
    /// 创建新的Redis客户端
    pub fn new(url: String, config: RedisOptions) -> Result<Self> {
        let client = Self {
            connection_pool: Arc::new(RwLock::new(Vec::new())),
            config,
            url,
        };
        
        client.initialize_pool()?;
        Ok(client)
    }
    
    /// 初始化连接池
    fn initialize_pool(&self) -> Result<()> {
        let mut pool = self.connection_pool.write().unwrap();
        
        for _ in 0..self.config.max_connections {
            let connection = self.create_connection()?;
            pool.push(connection);
        }
        
        Ok(())
    }
    
    /// 创建Redis连接
    #[cfg(feature = "redis")]
    fn create_connection(&self) -> Result<redis::Connection> {
        let client = redis::Client::open(self.url.as_str()).map_err(|e| {
            Error::Storage(format!("创建Redis客户端失败: {}", e))
        })?;
        
        let mut connection = client.get_connection_with_timeout(
            Duration::from_millis(self.config.connection_timeout_ms)
        ).map_err(|e| {
            Error::Storage(format!("连接Redis失败: {}", e))
        })?;
        
        // 选择数据库
        if self.config.database != 0 {
            redis::cmd("SELECT")
                .arg(self.config.database)
                .execute(&mut connection);
        }
        
        Ok(connection)
    }

    #[cfg(not(feature = "redis"))]
    fn create_connection(&self) -> Result<()> {
        Err(Error::feature_not_enabled("redis"))
    }

    /// 从连接池获取连接
    #[cfg(feature = "redis")]
    fn get_connection(&self) -> Result<redis::Connection> {
        let mut pool = self.connection_pool.write().unwrap();
        
        if let Some(connection) = pool.pop() {
            Ok(connection)
        } else {
            // 连接池为空，创建新连接
            self.create_connection()
        }
    }
    
    /// 归还连接到池中
    fn return_connection(&self, connection: redis::Connection) {
        let mut pool = self.connection_pool.write().unwrap();
        if pool.len() < self.config.max_connections {
            pool.push(connection);
        }
        // 如果池已满，连接会被丢弃
    }
    
    /// 带重试的操作执行
    fn execute_with_retry<F, R>(&self, mut operation: F) -> Result<R>
    where
        F: FnMut(&mut redis::Connection) -> redis::RedisResult<R>,
    {
        let mut last_error = None;
        
        for attempt in 0..self.config.retry_max_attempts {
            let mut connection = self.get_connection()?;
            
            match operation(&mut connection) {
                Ok(result) => {
                    self.return_connection(connection);
                    return Ok(result);
                }
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.config.retry_max_attempts - 1 {
                        std::thread::sleep(Duration::from_millis(self.config.retry_delay_ms));
                    }
                }
            }
        }
        
        Err(Error::Storage(format!(
            "Redis操作重试失败: {}",
            last_error.unwrap()
        )))
    }
    
    /// 获取值
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        self.execute_with_retry(|conn| {
            redis::cmd("GET").arg(key).query(conn)
        })
    }
    
    /// 设置值
    pub fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        self.execute_with_retry(|conn| {
            redis::cmd("SET").arg(key).arg(value).query::<String>(conn)
        })?;
        Ok(())
    }
    
    /// 设置带TTL的值
    pub fn setex(&self, key: &str, value: &[u8], ttl_seconds: u64) -> Result<()> {
        self.execute_with_retry(|conn| {
            redis::cmd("SETEX").arg(key).arg(ttl_seconds).arg(value).query::<String>(conn)
        })?;
        Ok(())
    }
    
    /// 删除键
    pub fn del(&self, key: &str) -> Result<()> {
        self.execute_with_retry(|conn| {
            redis::cmd("DEL").arg(key).query::<i32>(conn)
        })?;
        Ok(())
    }
    
    /// 检查键是否存在
    pub fn exists(&self, key: &str) -> Result<bool> {
        let result: i32 = self.execute_with_retry(|conn| {
            redis::cmd("EXISTS").arg(key).query(conn)
        })?;
        Ok(result > 0)
    }
    
    /// 获取所有键
    pub fn keys(&self, pattern: &str) -> Result<Vec<String>> {
        self.execute_with_retry(|conn| {
            redis::cmd("KEYS").arg(pattern).query(conn)
        })
    }
    
    /// 获取TTL
    pub fn ttl(&self, key: &str) -> Result<Option<u64>> {
        let result: i64 = self.execute_with_retry(|conn| {
            redis::cmd("TTL").arg(key).query(conn)
        })?;
        
        match result {
            -1 => Ok(None), // 键存在但没有TTL
            -2 => Ok(None), // 键不存在
            ttl if ttl > 0 => Ok(Some(ttl as u64)),
            _ => Ok(None),
        }
    }
    
    /// 设置TTL
    pub fn expire(&self, key: &str, ttl_seconds: u64) -> Result<bool> {
        let result: i32 = self.execute_with_retry(|conn| {
            redis::cmd("EXPIRE").arg(key).arg(ttl_seconds).query(conn)
        })?;
        Ok(result == 1)
    }
    
    /// 批量操作
    pub fn pipeline(&self, operations: Vec<BatchOperation>) -> Result<()> {
        self.execute_with_retry(|conn| {
            let mut pipe = redis::pipe();
            
            for op in operations {
                match op {
                    BatchOperation::Set { key, value } => {
                        pipe.cmd("SET").arg(key).arg(value);
                    }
                    BatchOperation::Delete { key } => {
                        pipe.cmd("DEL").arg(key);
                    }
                }
            }
            
            pipe.query::<()>(conn)
        })?;
        Ok(())
    }
    
    /// 获取数据库信息
    pub fn info(&self) -> Result<String> {
        self.execute_with_retry(|conn| {
            redis::cmd("INFO").query(conn)
        })
    }
}

impl KeyValueStorageEngineEnum {
    /// 创建内存存储引擎
    pub fn memory() -> Self {
        Self::Memory(MemoryKeyValueStorage::new())
    }

    /// 创建RocksDB存储引擎
    pub fn rocksdb<P: Into<PathBuf>>(path: P, options: Option<RocksDBOptions>) -> Result<Self> {
        let options = options.unwrap_or_default();
        let path_buf: PathBuf = path.into();
        let engine = Arc::new(RocksDBEngine::new(path_buf.as_path(), options.clone())?);
        Ok(Self::RocksDB { engine, options })
    }

    /// 创建Redis存储引擎
    pub fn redis(url: String, options: Option<RedisOptions>) -> Result<Self> {
        let options = options.unwrap_or_default();
        let client = Arc::new(RedisClient::new(url, options.clone())?);
        Ok(Self::Redis { client, options })
    }

    /// 存储序列化数据
    pub fn set<T: Serialize>(&self, key: &str, value: &T, format: SerializationFormat) -> Result<()> {
        let serialized = match format {
            SerializationFormat::Json => serde_json::to_vec(value).map_err(|e| {
                Error::Serialization(format!("JSON序列化失败: {}", e))
            })?,
            SerializationFormat::Bincode => bincode::serialize(value).map_err(|e| {
                Error::Serialization(format!("Bincode序列化失败: {}", e))
            })?,
            SerializationFormat::MessagePack => rmp_serde::to_vec(value).map_err(|e| {
                Error::Serialization(format!("MessagePack序列化失败: {}", e))
            })?,
        };

        KeyValueStorageEngine::set(self, key, &serialized)
    }

    /// 获取并反序列化数据
    pub fn get<T: for<'de> Deserialize<'de>>(&self, key: &str, format: SerializationFormat) -> Result<Option<T>> {
        if let Some(data) = KeyValueStorageEngine::get(self, key)? {
            let value = match format {
                SerializationFormat::Json => serde_json::from_slice(&data).map_err(|e| {
                    Error::Serialization(format!("JSON反序列化失败: {}", e))
                })?,
                SerializationFormat::Bincode => bincode::deserialize(&data).map_err(|e| {
                    Error::Serialization(format!("Bincode反序列化失败: {}", e))
                })?,
                SerializationFormat::MessagePack => rmp_serde::from_slice(&data).map_err(|e| {
                    Error::Serialization(format!("MessagePack反序列化失败: {}", e))
                })?,
            };
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    /// 获取原始字节数据
    pub fn get_raw(&self, key: &str) -> Result<Option<Vec<u8>>> {
        KeyValueStorageEngine::get(self, key)
    }

    /// 检查键是否存在
    pub fn exists(&self, key: &str) -> Result<bool> {
        KeyValueStorageEngine::exists(self, key)
    }

    /// 获取存储引擎信息
    pub fn get_engine_info(&self) -> Result<HashMap<String, String>> {
        KeyValueStorageEngine::get_info(self)
    }
}

impl KeyValueStorageEngine for KeyValueStorageEngineEnum {
    fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        match self {
            Self::Memory(engine) => engine.set(key, value),
            Self::RocksDB { engine, .. } => engine.put(key, value),
            Self::Redis { client, .. } => client.set(key, value),
        }
    }

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        match self {
            Self::Memory(engine) => engine.get(key),
            Self::RocksDB { engine, .. } => engine.get(key),
            Self::Redis { client, .. } => client.get(key),
        }
    }

    fn delete(&self, key: &str) -> Result<()> {
        match self {
            Self::Memory(engine) => engine.delete(key),
            Self::RocksDB { engine, .. } => engine.delete(key),
            Self::Redis { client, .. } => client.del(key),
        }
    }

    fn exists(&self, key: &str) -> Result<bool> {
        match self {
            Self::Memory(engine) => engine.exists(key),
            Self::RocksDB { engine, .. } => engine.exists(key),
            Self::Redis { client, .. } => client.exists(key),
        }
    }

    fn keys(&self) -> Result<Vec<String>> {
        match self {
            Self::Memory(engine) => engine.keys(),
            Self::RocksDB { engine, .. } => engine.keys(),
            Self::Redis { client, .. } => client.keys("*"),
        }
    }

    fn batch(&self, operations: Vec<BatchOperation>) -> Result<()> {
        match self {
            Self::Memory(engine) => engine.batch(operations),
            Self::RocksDB { engine, .. } => engine.batch(operations),
            Self::Redis { client, .. } => client.pipeline(operations),
        }
    }

    fn get_keys_with_pattern(&self, pattern: &str) -> Result<Vec<String>> {
        match self {
            Self::Memory(engine) => engine.get_keys_with_pattern(pattern),
            Self::RocksDB { engine, .. } => engine.keys_with_pattern(pattern),
            Self::Redis { client, .. } => client.keys(pattern),
        }
    }

    fn get_ttl(&self, key: &str) -> Result<Option<u64>> {
        match self {
            Self::Memory(engine) => engine.get_ttl(key),
            Self::RocksDB { engine, .. } => {
                // RocksDB 不支持 TTL，返回 None
                Ok(None)
            },
            Self::Redis { client, .. } => client.ttl(key),
        }
    }

    fn set_ttl(&self, key: &str, ttl: u64) -> Result<bool> {
        match self {
            Self::Memory(engine) => engine.set_ttl(key, ttl),
            Self::RocksDB { engine, .. } => {
                // RocksDB 不支持直接设置 TTL
                Err(Error::NotImplemented("RocksDB does not support TTL".to_string()))
            },
            Self::Redis { client, .. } => client.expire(key, ttl),
        }
    }

    fn extend_ttl(&self, key: &str, seconds: u64) -> Result<bool> {
        match self {
            Self::Memory(engine) => engine.extend_ttl(key, seconds),
            Self::RocksDB { engine, .. } => {
                // RocksDB 不支持直接设置 TTL
                Err(Error::NotImplemented("RocksDB does not support TTL".to_string()))
            },
            Self::Redis { client, .. } => {
                // Redis 需要先获取当前TTL再设置新的TTL
                if let Some(current_ttl) = client.ttl(key)? {
                    let new_ttl = current_ttl + seconds;
                    client.expire(key, new_ttl)
                } else {
                    Ok(false)
                }
            },
        }
    }

    fn create_snapshot(&self, path: &str) -> Result<()> {
        match self {
            Self::Memory(engine) => engine.create_snapshot(path),
            Self::RocksDB { engine, .. } => {
                // RocksDB 不支持直接创建快照到文件
                Err(Error::NotImplemented("RocksDB does not support file snapshots".to_string()))
            },
            Self::Redis { client, .. } => {
                // Redis 不支持直接创建快照
                Err(Error::NotImplemented("Redis does not support file snapshots".to_string()))
            },
        }
    }

    fn restore_from_snapshot(&self, path: &str) -> Result<()> {
        match self {
            Self::Memory(engine) => engine.restore_from_snapshot(path),
            Self::RocksDB { engine, .. } => {
                // RocksDB 不支持直接从快照恢复
                Err(Error::NotImplemented("RocksDB does not support restoring from snapshots".to_string()))
            },
            Self::Redis { client, .. } => {
                // Redis 不支持直接从快照恢复
                Err(Error::NotImplemented("Redis does not support restoring from snapshots".to_string()))
            },
        }
    }

    fn get_storage_info(&self) -> Result<HashMap<String, String>> {
        match self {
            Self::Memory(engine) => engine.get_storage_info(),
            Self::RocksDB { engine, .. } => {
                let mut info = HashMap::new();
                info.insert("type".to_string(), "rocksdb".to_string());
                if let Ok(Some(property)) = engine.get_property("rocksdb.estimate-num-keys") {
                    info.insert("estimated_keys".to_string(), property);
                }
                if let Ok(Some(property)) = engine.get_property("rocksdb.total-sst-files-size") {
                    info.insert("total_size_bytes".to_string(), property);
                }
                Ok(info)
            },
            Self::Redis { client, .. } => {
                let mut info = HashMap::new();
                info.insert("type".to_string(), "redis".to_string());
                if let Ok(info_str) = client.info() {
                    // 解析Redis INFO命令的输出
                    for line in info_str.lines() {
                        if let Some((key, value)) = line.split_once(':') {
                            info.insert(key.to_string(), value.to_string());
                        }
                    }
                }
                Ok(info)
            },
        }
    }

    fn clear(&self) -> Result<()> {
        match self {
            Self::Memory(engine) => engine.clear(),
            Self::RocksDB { engine, .. } => {
                // 获取所有键并删除
                let keys = engine.keys()?;
                for key in keys {
                    engine.delete(&key)?;
                }
                Ok(())
            },
            Self::Redis { client, .. } => {
                // Redis FLUSHDB命令清空当前数据库
                client.execute_with_retry(|conn| {
                    redis::cmd("FLUSHDB").query::<String>(conn)
                })?;
                Ok(())
            },
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Memory(engine) => engine.name(),
            Self::RocksDB { engine, .. } => "rocksdb",
            Self::Redis { client, .. } => "redis",
        }
    }

    fn get_info(&self) -> Result<HashMap<String, String>> {
        self.get_storage_info()
    }

    fn close(&self) -> Result<()> {
        match self {
            Self::Memory(engine) => engine.close(),
            Self::RocksDB { engine, .. } => {
                // RocksDB会在Drop时自动关闭
                Ok(())
            },
            Self::Redis { client, .. } => {
                // Redis连接会在Drop时自动关闭
                Ok(())
            },
        }
    }
}

/// 存储配置
#[derive(Debug, Clone)]
pub struct KeyValueStorageConfig {
    pub path: PathBuf,
    pub redis_url: Option<String>,
    pub rocksdb_options: Option<RocksDBOptions>,
    pub redis_options: Option<RedisOptions>,
}

impl Default for KeyValueStorageConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("./storage"),
            redis_url: None,
            rocksdb_options: None,
            redis_options: None,
        }
    }
}

/// 批量操作类型
#[derive(Debug, Clone)]
pub enum BatchOperation {
    /// 设置操作
    Set { key: String, value: Vec<u8> },
    /// 删除操作
    Delete { key: String },
}

/// 通配符匹配函数
fn wildcard_match(pattern: &str, text: &str) -> bool {
    let pattern_chars: Vec<char> = pattern.chars().collect();
    let text_chars: Vec<char> = text.chars().collect();
    
    fn match_recursive(
        pattern: &[char],
        text: &[char],
        p_idx: usize,
        t_idx: usize,
    ) -> bool {
        if p_idx == pattern.len() {
            return t_idx == text.len();
        }
        
        if pattern[p_idx] == '*' {
            // 尝试匹配0个字符
            if match_recursive(pattern, text, p_idx + 1, t_idx) {
                return true;
            }
            
            // 尝试匹配1个或多个字符
            for i in t_idx..text.len() {
                if match_recursive(pattern, text, p_idx + 1, i + 1) {
                    return true;
                }
            }
            
            false
        } else if pattern[p_idx] == '?' {
            if t_idx < text.len() {
                match_recursive(pattern, text, p_idx + 1, t_idx + 1)
            } else {
                false
            }
        } else {
            if t_idx < text.len() && pattern[p_idx] == text[t_idx] {
                match_recursive(pattern, text, p_idx + 1, t_idx + 1)
            } else {
                false
            }
        }
    }
    
    match_recursive(&pattern_chars, &text_chars, 0, 0)
}

/// 内存键值存储实现
pub struct MemoryKeyValueStorage {
    /// 存储数据的哈希表
    data: std::sync::RwLock<HashMap<String, Vec<u8>>>,
    /// 引擎名称
    name: String,
}

impl Clone for MemoryKeyValueStorage {
    fn clone(&self) -> Self {
        // 克隆数据
        let data = self.data.read().unwrap();
        let cloned_data = data.clone();
        Self {
            data: std::sync::RwLock::new(cloned_data),
            name: self.name.clone(),
        }
    }
}

impl std::fmt::Debug for MemoryKeyValueStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryKeyValueStorage")
            .field("name", &self.name)
            .finish()
    }
}

impl MemoryKeyValueStorage {
    /// 创建新的内存存储实例
    pub fn new() -> Self {
        Self {
            data: std::sync::RwLock::new(HashMap::new()),
            name: "memory".to_string(),
        }
    }
    
    /// 使用指定名称创建新的内存存储实例
    pub fn new_with_name(name: &str) -> Self {
        Self {
            data: std::sync::RwLock::new(HashMap::new()),
            name: name.to_string(),
        }
    }
}

impl KeyValueStorageEngine for MemoryKeyValueStorage {
    fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        let mut data = self.data.write().map_err(|e| {
            Error::Storage(format!("无法获取内存存储写锁: {}", e))
        })?;
        data.insert(key.to_string(), value.to_vec());
        Ok(())
    }

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let data = self.data.read().map_err(|e| {
            Error::Storage(format!("无法获取内存存储读锁: {}", e))
        })?;
        Ok(data.get(key).cloned())
    }

    fn delete(&self, key: &str) -> Result<()> {
        let mut data = self.data.write().map_err(|e| {
            Error::Storage(format!("无法获取内存存储写锁: {}", e))
        })?;
        data.remove(key);
        Ok(())
    }

    fn exists(&self, key: &str) -> Result<bool> {
        let data = self.data.read().map_err(|e| {
            Error::Storage(format!("无法获取内存存储读锁: {}", e))
        })?;
        Ok(data.contains_key(key))
    }

    fn keys(&self) -> Result<Vec<String>> {
        let data = self.data.read().map_err(|e| {
            Error::Storage(format!("无法获取内存存储读锁: {}", e))
        })?;
        Ok(data.keys().cloned().collect())
    }

    fn batch(&self, operations: Vec<BatchOperation>) -> Result<()> {
        let mut data = self.data.write().map_err(|e| {
            Error::Storage(format!("无法获取内存存储写锁: {}", e))
        })?;
        
        for operation in operations {
            match operation {
                BatchOperation::Set { key, value } => {
                    data.insert(key, value);
                }
                BatchOperation::Delete { key } => {
                    data.remove(&key);
                }
            }
        }
        Ok(())
    }

    fn get_keys_with_pattern(&self, pattern: &str) -> Result<Vec<String>> {
        let all_keys = self.keys()?;
        let mut matching_keys = Vec::new();
        
        for key in all_keys {
            if key.contains(pattern) || wildcard_match(pattern, &key) {
                matching_keys.push(key);
            }
        }
        
        Ok(matching_keys)
    }

    fn get_ttl(&self, _key: &str) -> Result<Option<u64>> {
        // 内存存储不支持TTL
        Ok(None)
    }

    fn set_ttl(&self, _key: &str, _ttl: u64) -> Result<bool> {
        // 内存存储不支持TTL
        Err(Error::NotImplemented("内存存储不支持TTL".to_string()))
    }

    fn extend_ttl(&self, _key: &str, _seconds: u64) -> Result<bool> {
        // 内存存储不支持TTL
        Err(Error::NotImplemented("内存存储不支持TTL".to_string()))
    }

    fn create_snapshot(&self, path: &str) -> Result<()> {
        let data = self.data.read().map_err(|e| {
            Error::Storage(format!("无法获取内存存储读锁: {}", e))
        })?;
        
        let serialized = bincode::serialize(&*data).map_err(|e| {
            Error::Storage(format!("序列化快照失败: {}", e))
        })?;
        
        std::fs::write(path, serialized).map_err(|e| {
            Error::Storage(format!("写入快照文件失败: {}", e))
        })?;
        
        Ok(())
    }

    fn restore_from_snapshot(&self, path: &str) -> Result<()> {
        let serialized = std::fs::read(path).map_err(|e| {
            Error::Storage(format!("读取快照文件失败: {}", e))
        })?;
        
        let restored_data: HashMap<String, Vec<u8>> = bincode::deserialize(&serialized).map_err(|e| {
            Error::Storage(format!("反序列化快照失败: {}", e))
        })?;
        
        let mut data = self.data.write().map_err(|e| {
            Error::Storage(format!("无法获取内存存储写锁: {}", e))
        })?;
        
        *data = restored_data;
        Ok(())
    }

    fn get_storage_info(&self) -> Result<HashMap<String, String>> {
        let mut info = HashMap::new();
        info.insert("type".to_string(), "memory".to_string());
        info.insert("name".to_string(), self.name.clone());
        
        let data = self.data.read().map_err(|e| {
            Error::Storage(format!("无法获取内存存储读锁: {}", e))
        })?;
        info.insert("key_count".to_string(), data.len().to_string());
        
        // 估算内存使用量
        let memory_usage: usize = data.iter()
            .map(|(k, v)| k.len() + v.len())
            .sum();
        info.insert("estimated_memory_bytes".to_string(), memory_usage.to_string());
        
        Ok(info)
    }

    fn clear(&self) -> Result<()> {
        let mut data = self.data.write().map_err(|e| {
            Error::Storage(format!("无法获取内存存储写锁: {}", e))
        })?;
        data.clear();
        Ok(())
    }

    fn name(&self) -> &'static str {
        "memory"
    }

    fn get_info(&self) -> Result<HashMap<String, String>> {
        self.get_storage_info()
    }

    fn close(&self) -> Result<()> {
        // 内存存储无需特殊关闭操作
        Ok(())
    }
}

/// 创建存储引擎
pub fn create_engine(engine_type: &str, config: KeyValueStorageConfig) -> Result<KeyValueStorageEngineEnum> {
    match engine_type {
        "memory" => Ok(KeyValueStorageEngineEnum::memory()),
        "rocksdb" => KeyValueStorageEngineEnum::rocksdb(config.path, config.rocksdb_options),
        "redis" => {
            let url = config.redis_url.unwrap_or_else(|| "redis://localhost:6379".to_string());
            KeyValueStorageEngineEnum::redis(url, config.redis_options)
        },
        _ => Err(Error::NotImplemented(format!("不支持的存储引擎类型: {}", engine_type))),
    }
}

/// 分布式锁管理器子模块
pub mod distributed_lock {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};
    use log::{debug, error, info, warn};
    use serde::{Serialize, Deserialize};

    use crate::error::{Error, Result};
    use crate::storage::engine::StorageEngine;

    /// 锁类型
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    pub enum LockType {
        /// 共享锁（读锁）
        Shared,
        /// 排他锁（写锁）
        Exclusive,
    }

    /// 锁状态
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct LockStatus {
        /// 锁类型
        pub lock_type: LockType,
        /// 锁持有者ID
        pub owner_id: String,
        /// 锁创建时间戳
        pub created_at: i64,
        /// 锁过期时间戳
        pub expires_at: i64,
        /// 锁版本号（用于乐观并发控制）
        pub version: u64,
    }

    /// 锁选项
    #[derive(Debug, Clone)]
    pub struct LockOptions {
        /// 锁超时时间（毫秒）
        pub timeout_ms: u64,
        /// 获取锁的重试次数
        pub retry_count: usize,
        /// 重试间隔（毫秒）
        pub retry_interval_ms: u64,
        /// 是否自动续约
        pub auto_refresh: bool,
        /// 续约间隔（毫秒）
        pub refresh_interval_ms: u64,
    }

    impl Default for LockOptions {
        fn default() -> Self {
            Self {
                timeout_ms: 30000,       // 默认30秒超时
                retry_count: 3,          // 默认重试3次
                retry_interval_ms: 100,  // 默认100毫秒重试间隔
                auto_refresh: false,     // 默认不自动续约
                refresh_interval_ms: 5000, // 默认5秒续约间隔
            }
        }
    }

    /// 分布式锁管理器
    pub struct DistributedLockManager {
        /// 存储引擎
        storage: Arc<dyn StorageEngine>,
        /// 锁前缀（用于区分锁键与其他数据）
        lock_prefix: String,
        /// 当前持有的锁（用于跟踪和自动释放）
        held_locks: Arc<Mutex<HashMap<String, (LockType, Instant, u64)>>>,
        /// 是否运行中
        running: Arc<Mutex<bool>>,
    }

    impl DistributedLockManager {
        /// 创建新的分布式锁管理器
        pub fn new(storage: Arc<dyn StorageEngine>, lock_prefix: &str) -> Self {
            Self {
                storage,
                lock_prefix: lock_prefix.to_string(),
                held_locks: Arc::new(Mutex::new(HashMap::new())),
                running: Arc::new(Mutex::new(true)),
            }
        }

        /// 生成锁键
        fn make_lock_key(&self, resource: &str) -> String {
            format!("{}:{}", self.lock_prefix, resource)
        }

        /// 获取锁
        pub fn acquire_lock(&self, resource: &str, lock_type: LockType, owner_id: &str, options: Option<LockOptions>) -> Result<bool> {
            let options = options.unwrap_or_default();
            let lock_key = self.make_lock_key(resource);
            let current_time = chrono::Utc::now().timestamp_millis();
            let expiry_time = current_time + options.timeout_ms as i64;
            
            // 尝试获取锁
            for retry in 0..=options.retry_count {
                // 先检查锁是否存在
                let lock_data = self.storage.get(lock_key.as_bytes())?;
                
                if let Some(data) = lock_data {
                    // 锁已存在，解析锁状态
                    let lock_status: LockStatus = bincode::deserialize(&data)
                        .map_err(|e| Error::data_corruption(format!("Failed to deserialize lock: {}", e)))?;
                    
                    // 检查锁是否过期
                    if current_time > lock_status.expires_at {
                        // 锁已过期，尝试获取
                        debug!("Lock for {} expired, attempting to acquire", resource);
                        if self.try_acquire_expired_lock(&lock_key, lock_type, owner_id, expiry_time, lock_status.version)? {
                            // 成功获取锁
                            self.track_lock(resource, lock_type, options.timeout_ms)?;
                            return Ok(true);
                        }
                    } else if lock_status.lock_type == LockType::Shared && lock_type == LockType::Shared {
                        // 共享锁可以被多个持有者持有
                        debug!("Acquiring shared lock for {} with existing shared lock", resource);
                        let new_lock = LockStatus {
                            lock_type,
                            owner_id: owner_id.to_string(),
                            created_at: current_time,
                            expires_at: expiry_time,
                            version: lock_status.version + 1,
                        };
                        
                        if self.try_store_lock(&lock_key, &new_lock)? {
                            self.track_lock(resource, lock_type, options.timeout_ms)?;
                            return Ok(true);
                        }
                    } else if lock_status.owner_id == owner_id {
                        // 已经持有该锁
                        debug!("Already holding lock for {}", resource);
                        // 刷新锁超时
                        self.refresh_lock(resource, options.timeout_ms)?;
                        return Ok(true);
                    } else {
                        // 锁被其他人持有且未过期
                        if retry < options.retry_count {
                            // 等待后重试
                            debug!("Lock for {} is held by {}, retry {}/{}", 
                                   resource, lock_status.owner_id, retry + 1, options.retry_count);
                            std::thread::sleep(Duration::from_millis(options.retry_interval_ms));
                            continue;
                        } else {
                            // 重试次数用尽
                            return Ok(false);
                        }
                    }
                } else {
                    // 锁不存在，尝试创建
                    let new_lock = LockStatus {
                        lock_type,
                        owner_id: owner_id.to_string(),
                        created_at: current_time,
                        expires_at: expiry_time,
                        version: 1,
                    };
                    
                    if self.try_store_lock(&lock_key, &new_lock)? {
                        self.track_lock(resource, lock_type, options.timeout_ms)?;
                        return Ok(true);
                    }
                }
            }
            
            // 无法获取锁
            Ok(false)
        }

        /// 尝试获取过期的锁
        fn try_acquire_expired_lock(&self, lock_key: &str, lock_type: LockType, owner_id: &str, 
                                  expiry_time: i64, version: u64) -> Result<bool> {
            let current_time = chrono::Utc::now().timestamp_millis();
            let new_lock = LockStatus {
                lock_type,
                owner_id: owner_id.to_string(),
                created_at: current_time,
                expires_at: expiry_time,
                version: version + 1,
            };
            
            self.try_store_lock(lock_key, &new_lock)
        }

        /// 尝试存储锁
        fn try_store_lock(&self, lock_key: &str, lock_status: &LockStatus) -> Result<bool> {
            let encoded = bincode::serialize(lock_status)
                .map_err(|e| Error::internal(format!("Failed to serialize lock: {}", e)))?;
            
            // 使用条件更新操作确保原子性
            self.storage.put(lock_key.as_bytes(), &encoded)?;
            Ok(true)
        }

        /// 跟踪锁
        fn track_lock(&self, resource: &str, lock_type: LockType, timeout_ms: u64) -> Result<()> {
            if let Ok(mut held_locks) = self.held_locks.lock() {
                held_locks.insert(
                    resource.to_string(),
                    (lock_type, Instant::now(), timeout_ms)
                );
            }
            Ok(())
        }

        /// 释放锁
        pub fn release_lock(&self, resource: &str, owner_id: &str) -> Result<bool> {
            let lock_key = self.make_lock_key(resource);
            
            // 获取当前锁状态
            let lock_data = self.storage.get(lock_key.as_bytes())?;
            
            if let Some(data) = lock_data {
                let lock_status: LockStatus = bincode::deserialize(&data)
                    .map_err(|e| Error::data_corruption(format!("Failed to deserialize lock: {}", e)))?;
                
                // 检查持有者
                if lock_status.owner_id == owner_id {
                    // 删除锁
                    self.storage.delete(lock_key.as_bytes())?;
                    
                    // 从跟踪中移除
                    if let Ok(mut held_locks) = self.held_locks.lock() {
                        held_locks.remove(resource);
                    }
                    
                    info!("Lock for {} released by {}", resource, owner_id);
                    return Ok(true);
                } else {
                    warn!("Attempt to release lock {} by non-owner {}", resource, owner_id);
                    return Ok(false);
                }
            } else {
                // 锁不存在
                debug!("Lock for {} does not exist", resource);
                return Ok(false);
            }
        }

        /// 检查锁状态
        pub fn check_lock(&self, resource: &str) -> Result<Option<LockStatus>> {
            let lock_key = self.make_lock_key(resource);
            let lock_data = self.storage.get(lock_key.as_bytes())?;
            
            if let Some(data) = lock_data {
                let lock_status: LockStatus = bincode::deserialize(&data)
                    .map_err(|e| Error::data_corruption(format!("Failed to deserialize lock: {}", e)))?;
                
                // 检查是否过期
                let current_time = chrono::Utc::now().timestamp_millis();
                if current_time > lock_status.expires_at {
                    // 锁已过期，清理
                    self.storage.delete(lock_key.as_bytes())?;
                    return Ok(None);
                }
                
                return Ok(Some(lock_status));
            }
            
            Ok(None)
        }

        /// 刷新锁
        pub fn refresh_lock(&self, resource: &str, timeout_ms: u64) -> Result<bool> {
            let lock_key = self.make_lock_key(resource);
            let lock_data = self.storage.get(lock_key.as_bytes())?;
            
            if let Some(data) = lock_data {
                let mut lock_status: LockStatus = bincode::deserialize(&data)
                    .map_err(|e| Error::data_corruption(format!("Failed to deserialize lock: {}", e)))?;
                
                // 更新过期时间
                let current_time = chrono::Utc::now().timestamp_millis();
                lock_status.expires_at = current_time + timeout_ms as i64;
                lock_status.version += 1;
                
                // 保存更新的锁
                let encoded = bincode::serialize(&lock_status)
                    .map_err(|e| Error::internal(format!("Failed to serialize lock: {}", e)))?;
                self.storage.put(lock_key.as_bytes(), &encoded)?;
                
                debug!("Lock for {} refreshed", resource);
                return Ok(true);
            }
            
            Ok(false)
        }

        /// 启动自动续约
        pub fn start_auto_refresh(&self) -> Result<()> {
            // 实现自动续约逻辑
            // 启动后台线程定期刷新持有的锁
            let held_locks = Arc::clone(&self.held_locks);
            let running = Arc::clone(&self.running);
            let manager = self.clone();
            
            std::thread::spawn(move || {
                while *running.lock().unwrap() {
                    if let Ok(locks) = held_locks.lock() {
                        for (resource, (_, acquired_at, timeout_ms)) in locks.iter() {
                            let elapsed = acquired_at.elapsed();
                            let refresh_threshold = Duration::from_millis(*timeout_ms / 2);
                            
                            if elapsed > refresh_threshold {
                                if let Err(e) = manager.refresh_lock(resource, *timeout_ms) {
                                    error!("Failed to refresh lock for {}: {}", resource, e);
                                }
                            }
                        }
                    }
                    
                    std::thread::sleep(Duration::from_millis(1000)); // 每秒检查一次
                }
            });
            
            Ok(())
        }

        /// 停止自动续约
        pub fn stop_auto_refresh(&self) -> Result<()> {
            if let Ok(mut running) = self.running.lock() {
                *running = false;
            }
            Ok(())
        }

        /// 清理过期锁
        pub fn cleanup_expired_locks(&self) -> Result<usize> {
            // 这是一个简化的实现，实际中需要遍历所有锁键
            // 由于这里使用的是通用存储引擎，无法直接获取所有锁键
            // 建议在生产环境中使用支持键模式查询的存储引擎
            info!("Cleanup expired locks called - implementation depends on storage engine capabilities");
            Ok(0)
        }
    }

    impl Clone for DistributedLockManager {
        fn clone(&self) -> Self {
            Self {
                storage: Arc::clone(&self.storage),
                lock_prefix: self.lock_prefix.clone(),
                held_locks: Arc::clone(&self.held_locks),
                running: Arc::clone(&self.running),
            }
        }
    }

    /// 锁守卫，确保锁在超出作用域时自动释放
    pub struct LockGuard<'a> {
        /// 锁管理器引用
        manager: &'a DistributedLockManager,
        /// 资源名称
        resource: String,
        /// 持有者ID
        owner_id: String,
        /// 是否仍持有锁
        holding: bool,
    }

    impl<'a> LockGuard<'a> {
        /// 创建新的锁守卫
        pub fn new(manager: &'a DistributedLockManager, resource: String, owner_id: String) -> Self {
            Self {
                manager,
                resource,
                owner_id,
                holding: true,
            }
        }

        /// 手动释放锁
        pub fn release(&mut self) -> Result<bool> {
            if self.holding {
                let result = self.manager.release_lock(&self.resource, &self.owner_id);
                self.holding = false;
                result
            } else {
                Ok(false)
            }
        }

        /// 检查是否仍持有锁
        pub fn is_holding(&self) -> bool {
            self.holding
        }
    }

    impl<'a> Drop for LockGuard<'a> {
        fn drop(&mut self) {
            if self.holding {
                if let Err(e) = self.manager.release_lock(&self.resource, &self.owner_id) {
                    error!("Failed to release lock for {} during drop: {}", self.resource, e);
                }
            }
        }
    }
}

// 重新导出分布式锁组件，保持API兼容性
pub use distributed_lock::{
    DistributedLockManager,
    LockGuard,
    LockType,
    LockOptions,
    LockStatus,
}; 