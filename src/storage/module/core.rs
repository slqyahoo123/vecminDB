// 核心存储模块
// 包含Storage结构的定义和主要实现

use std::path::Path;
use std::fs;
use std::io::{Read, Write};
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use log::warn;
use rocksdb::{
    DB, Options, WriteBatch, ColumnFamilyDescriptor,
    Cache, BlockBasedOptions, DBCompressionType,
};
use serde_json;
use serde::de::DeserializeOwned;
use chrono::Utc;
use crate::{Error, Result};
use std::cell::RefCell;
use bincode;

use super::config::StorageConfig;
use super::cache::{CachePolicy, CacheEntry};
use super::data::{CF_RAW_DATA, CF_PROCESSED_DATA};
// 模型相关的列族常量（已移除模型模块，使用兼容层）
const CF_MODEL_PARAMS: &str = "model_params";
const CF_CHECKPOINTS: &str = "checkpoints";
use super::algorithm::CF_ALGORITHMS;
use super::transaction::{CF_VERSIONS, CF_INFO};
use crate::storage::module::permissions::PermissionManager;

use async_trait::async_trait;
use crate::core::interfaces::storage_interface as core_storage_if;
use crate::core::interfaces::storage as kv_storage_if;
use once_cell::sync::Lazy;

/// 存储引擎主结构
#[derive(Debug)]
pub struct Storage {
    db: DB,
    config: StorageConfig,
    cache: RwLock<HashMap<String, CacheEntry>>,
    cache_size: RwLock<usize>,
    pub(crate) permission_manager: RefCell<PermissionManager>,
}

// Implement Send and Sync for Storage
unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

impl Storage {
    /// 获取键值（字符串键版本）
    pub async fn get_raw(&self, key: &str) -> Result<Option<Vec<u8>>> {
        // 直接调用同步方法，async函数会自动将Result包装成Future
        Ok(self.get(key.as_bytes())?)
    }
    
    /// 设置键值（字符串键版本）
    pub async fn put_raw(&self, key: &str, value: &[u8]) -> Result<()> {
        // 直接调用同步方法，async函数会自动将Result包装成Future
        Ok(self.put(key.as_bytes(), value)?)
    }
    
    /// 删除键（字符串键版本）
    pub async fn delete_raw(&self, key: &str) -> Result<()> {
        // 直接调用同步方法，async函数会自动将Result包装成Future
        Ok(self.delete(key.as_bytes())?)
    }
    
    /// 扫描前缀（字符串键版本）
    pub async fn scan_prefix_raw(&self, prefix: &str) -> Result<Vec<(String, Vec<u8>)>> {
        // 直接调用同步方法，async函数会自动将Result包装成Future
        let mut results = Vec::new();
        let iter = self.db.iterator(rocksdb::IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward));
        
        for item in iter {
            match item {
                Ok((key, value)) => {
                    if key.starts_with(prefix.as_bytes()) {
                        if let Ok(key_str) = String::from_utf8(key.to_vec()) {
                            results.push((key_str, value.to_vec()));
                        }
                    } else {
                        // 如果键不再以前缀开头，停止迭代
                        break;
                    }
                }
                Err(e) => {
                    return Err(Error::storage(format!("扫描错误: {}", e)));
                }
            }
        }
        
        Ok(results)
    }
    
    /// 存储数据内容
    pub async fn put_data(&self, dataset_id: &str, data: &[u8]) -> Result<()> {
        let key = format!("dataset_data:{}", dataset_id);
        self.put_raw(&key, data).await
    }
    
    /// 获取数据内容
    pub async fn get_data(&self, dataset_id: &str) -> Result<Option<Vec<u8>>> {
        let key = format!("dataset_data:{}", dataset_id);
        self.get_raw(&key).await
    }
    
    /// 删除数据内容
    pub async fn delete_data(&self, dataset_id: &str) -> Result<()> {
        let key = format!("dataset_data:{}", dataset_id);
        self.delete_raw(&key).await
    }
    
    /// 存储数据集元数据
    pub async fn put_dataset<T: serde::Serialize>(&self, dataset_id: &str, dataset: &T) -> Result<()> {
        let key = format!("dataset_meta:{}", dataset_id);
        let value = serde_json::to_vec(dataset)
            .map_err(|e| Error::Serialization(format!("序列化数据集元数据失败: {}", e)))?;
        self.put_raw(&key, &value).await
    }
    
    /// 删除数据集元数据
    pub async fn delete_dataset(&self, dataset_id: &str) -> Result<()> {
        let key = format!("dataset_meta:{}", dataset_id);
        self.delete_raw(&key).await
    }
    
    /// 获取数据集元数据并反序列化为指定类型
    pub async fn get_dataset<T: DeserializeOwned>(&self, dataset_id: &str) -> Result<Option<T>> {
        let key = format!("dataset_meta:{}", dataset_id);
        if let Some(bytes) = self.get_raw(&key).await? {
            let value = serde_json::from_slice(&bytes)
                .map_err(|e| Error::Serialization(format!("反序列化数据集元数据失败: {}", e)))?;
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }
    
    /// 创建内存存储实例
    pub fn new_in_memory() -> Result<Arc<Self>> {
        let config = StorageConfig {
            path: ":memory:".to_string(),
            max_background_jobs: 1,
            create_if_missing: true,
            enable_compression: false,
            compression_level: 0,
            enable_cache: true,
            max_cache_size: 1024 * 1024 * 10, // 10MB
            cache_policy: CachePolicy::LRU,
            permissions_path: ":memory:".to_string(),
            max_open_files: 100,
            keep_log_file_num: 2,
            max_total_wal_size: 1024 * 1024, // 1MB
        };
        
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);
        
        // 使用临时目录作为内存数据库
        let temp_dir = std::env::temp_dir().join(format!("vecmind_storage_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir)?;
        
        let db = DB::open(&opts, &temp_dir)?;
        
        let permission_manager = PermissionManager::new_in_memory()
            .map_err(|e| Error::StorageError(format!("Failed to initialize permission manager: {}", e)))?;
        
        let storage = Arc::new(Self {
            db,
            config,
            cache: RwLock::new(HashMap::new()),
            cache_size: RwLock::new(0),
            permission_manager: RefCell::new(permission_manager),
        });
        // 注册到全局注册表，以便在 trait 方法中获取 Arc
        // 使用 usize 作为键，因为 *const Storage 不能在线程间安全传递
        let storage_ptr = Arc::as_ptr(&storage) as usize;
        STORAGE_REGISTRY.lock().map_err(|_| Error::lock("无法获取存储注册表锁".to_string()))?.insert(storage_ptr, storage.clone());
        Ok(storage)
    }
    
    /// 打开存储
    pub fn open(config: StorageConfig) -> Result<Self> {
        let mut opts = Options::default();
        opts.create_if_missing(config.create_if_missing);
        opts.set_max_background_jobs(config.max_background_jobs);
        // RocksDB 接口使用 i32 / usize / u64，这里做安全类型转换
        opts.set_max_open_files(config.max_open_files);
        opts.set_keep_log_file_num(config.keep_log_file_num as usize);
        opts.set_max_total_wal_size(config.max_total_wal_size as u64);
        
        if config.enable_compression {
            let compression_type = match config.compression_level {
                1..=3 => DBCompressionType::Snappy,
                4..=6 => DBCompressionType::Zlib,
                7..=9 => DBCompressionType::Zstd,
                _ => DBCompressionType::None,
            };
            opts.set_compression_type(compression_type);
        }
        
        // 创建默认列族
        let cf_opts = Options::default();
        let cf_descriptors = vec![
            ColumnFamilyDescriptor::new(CF_MODEL_PARAMS, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_CHECKPOINTS, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_RAW_DATA, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_PROCESSED_DATA, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_ALGORITHMS, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_VERSIONS, cf_opts.clone()),
            ColumnFamilyDescriptor::new(CF_INFO, cf_opts.clone()),
        ];
        
        let db_path = Path::new(&config.path);
        if !db_path.exists() && config.create_if_missing {
            fs::create_dir_all(db_path)?;
        }
        
        let db = match DB::open_cf_descriptors(&opts, &config.path, cf_descriptors) {
            Ok(db) => db,
            Err(e) => {
                // 如果之前没有列族，尝试直接打开
                if e.to_string().contains("Invalid argument: Column family not found") {
                    DB::open(&opts, &config.path)?
                } else {
                    return Err(Error::StorageError(format!("Failed to open RocksDB: {}", e)));
                }
            }
        };
        
        // 初始化权限管理器
        let permission_manager = PermissionManager::new(&config.permissions_path)?;
        
        Ok(Self {
            db,
            config,
            cache: RwLock::new(HashMap::new()),
            cache_size: RwLock::new(0),
            permission_manager: RefCell::new(permission_manager),
        })
    }
    
    /// 创建默认存储
    pub fn new_default(path: &str) -> Result<Self> {
        let config = StorageConfig::new(path.to_string());
        Self::open(config)
    }
    
    /// 创建数据库选项
    fn create_db_options(config: &StorageConfig) -> Result<Options> {
        let mut opts = Options::default();
        opts.create_if_missing(config.create_if_missing);
        opts.set_max_background_jobs(config.max_background_jobs);
        
        if config.enable_cache {
            let cache = Cache::new_lru_cache(config.max_cache_size);
            let mut block_opts = BlockBasedOptions::default();
            block_opts.set_block_cache(&cache);
            opts.set_block_based_table_factory(&block_opts);
        }
        
        if config.enable_compression {
            let compression_type = match config.compression_level {
                1..=3 => DBCompressionType::Snappy,
                4..=6 => DBCompressionType::Zlib,
                7..=9 => DBCompressionType::Zstd,
                _ => DBCompressionType::None,
            };
            opts.set_compression_type(compression_type);
        }
        
        Ok(opts)
    }
    
    /// 创建新存储
    pub fn new(config: StorageConfig) -> Result<Arc<Self>> {
        let opts = Self::create_db_options(&config)?;
        
        // 创建默认列族
        let cf_names = vec![
            CF_MODEL_PARAMS,
            CF_CHECKPOINTS,
            CF_RAW_DATA,
            CF_PROCESSED_DATA,
            CF_ALGORITHMS,
            CF_VERSIONS,
            CF_INFO,
        ];
        
        let cf_descriptors: Vec<_> = cf_names
            .iter()
            .map(|name| ColumnFamilyDescriptor::new(*name, Options::default()))
            .collect();
        
        let db_path = Path::new(&config.path);
        if !db_path.exists() && config.create_if_missing {
            fs::create_dir_all(db_path)?;
        }
        
        let db = DB::open_cf_descriptors(&opts, &config.path, cf_descriptors)
            .or_else(|_| {
                // 如果列族不存在，尝试直接打开并创建列族
                let db = DB::open(&opts, &config.path)?;
                for name in cf_names {
                    if !db.cf_handle(name).is_some() {
                        db.create_cf(name, &Options::default())?;
                    }
                }
                Ok::<DB, Error>(db)
            })?;
        
        // 初始化权限管理器
        let permission_manager = PermissionManager::new(&config.permissions_path)?;
        
        let storage = Arc::new(Self {
            db,
            config,
            cache: RwLock::new(HashMap::new()),
            cache_size: RwLock::new(0),
            permission_manager: RefCell::new(permission_manager),
        });
        // 注册到全局注册表，以便在 trait 方法中获取 Arc
        // 使用 usize 作为键，因为 *const Storage 不能在线程间安全传递
        let storage_ptr = Arc::as_ptr(&storage) as usize;
        STORAGE_REGISTRY.lock().map_err(|_| Error::lock("无法获取存储注册表锁".to_string()))?.insert(storage_ptr, storage.clone());
        Ok(storage)
    }
}

// ================= Core Interfaces bridging (core::interfaces::storage_interface) =================

struct ModuleStorageTransaction {
    storage: std::sync::Arc<Storage>,
    id: String,
    state: core_storage_if::TransactionState,
    ops: Vec<ModuleTxOp>,
}

enum ModuleTxOp { Put(String, Vec<u8>), Delete(String) }

impl ModuleStorageTransaction {
    fn new(storage: std::sync::Arc<Storage>) -> Self {
        Self { storage, id: uuid::Uuid::new_v4().to_string(), state: core_storage_if::TransactionState::Active, ops: Vec::new() }
    }
}

impl core_storage_if::StorageTransaction for ModuleStorageTransaction {
    fn commit(&mut self) -> crate::Result<()> {
        if self.state != core_storage_if::TransactionState::Active {
            return Err(crate::Error::Transaction("事务非活跃状态，无法提交".to_string()));
        }
        
        // 使用 tokio::runtime::Handle::try_current() 来安全地执行异步操作
        // 如果当前在 tokio 运行时中，使用 block_on 会检测到并避免死锁
        // 如果不在运行时中，创建一个临时的运行时
        let rt_handle = tokio::runtime::Handle::try_current();
        
        for op in self.ops.drain(..) {
            match op {
                ModuleTxOp::Put(k, v) => {
                    if let Ok(handle) = rt_handle.as_ref() {
                        // 在 tokio 运行时中，使用 block_on 是安全的
                        handle.block_on(self.storage.put_raw(&k, &v))?;
                    } else {
                        // 不在运行时中，创建临时运行时
                        let rt = tokio::runtime::Runtime::new()
                            .map_err(|e| crate::Error::lock(format!("无法创建 tokio 运行时: {}", e)))?;
                        rt.block_on(self.storage.put_raw(&k, &v))?;
                    }
                },
                ModuleTxOp::Delete(k) => {
                    if let Ok(handle) = rt_handle.as_ref() {
                        handle.block_on(self.storage.delete_raw(&k))?;
                    } else {
                        let rt = tokio::runtime::Runtime::new()
                            .map_err(|e| crate::Error::lock(format!("无法创建 tokio 运行时: {}", e)))?;
                        rt.block_on(self.storage.delete_raw(&k))?;
                    }
                },
            }
        }
        self.state = core_storage_if::TransactionState::Committed;
        Ok(())
    }

    fn rollback(&mut self) -> crate::Result<()> {
        self.ops.clear();
        self.state = core_storage_if::TransactionState::RolledBack;
        Ok(())
    }

    fn store(&mut self, key: &str, value: &[u8]) -> crate::Result<()> {
        self.ops.push(ModuleTxOp::Put(key.to_string(), value.to_vec()));
        Ok(())
    }

    fn retrieve(&self, key: &str) -> crate::Result<Option<Vec<u8>>> {
        // 使用 tokio::runtime::Handle::try_current() 来安全地执行异步操作
        let rt_handle = tokio::runtime::Handle::try_current();
        if let Ok(handle) = rt_handle.as_ref() {
            handle.block_on(self.storage.get_raw(key))
        } else {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| crate::Error::lock(format!("无法创建 tokio 运行时: {}", e)))?;
            rt.block_on(self.storage.get_raw(key))
        }
    }

    fn delete(&mut self, key: &str) -> crate::Result<()> {
        self.ops.push(ModuleTxOp::Delete(key.to_string()));
        Ok(())
    }

    fn exists(&self, key: &str) -> crate::Result<bool> {
        // 使用 tokio::runtime::Handle::try_current() 来安全地执行异步操作
        let rt_handle = tokio::runtime::Handle::try_current();
        let result = if let Ok(handle) = rt_handle.as_ref() {
            handle.block_on(self.storage.get_raw(key))
        } else {
            let rt = tokio::runtime::Runtime::new()
                .map_err(|e| crate::Error::lock(format!("无法创建 tokio 运行时: {}", e)))?;
            rt.block_on(self.storage.get_raw(key))
        }?;
        Ok(result.is_some())
    }

    fn get_state(&self) -> core_storage_if::TransactionState { self.state }
    fn get_id(&self) -> &str { &self.id }
}

// 事务注册表：用于实现 kv_storage_if::TransactionalStore 的基于ID的事务管理
static MODULE_TX_REGISTRY: Lazy<Mutex<std::collections::HashMap<String, ModuleStorageTransaction>>> = Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

// 全局 Storage 注册表，用于在 trait 方法中获取 Arc<Storage>
// 使用 usize 作为键，因为 *const Storage 不能在线程间安全传递
static STORAGE_REGISTRY: Lazy<Mutex<std::collections::HashMap<usize, Arc<Storage>>>> = Lazy::new(|| Mutex::new(std::collections::HashMap::new()));

/// KeyValueStore 适配到 Storage
#[async_trait]
impl kv_storage_if::KeyValueStore for Storage {
    async fn get(&self, key: &str) -> crate::Result<Option<Vec<u8>>> {
        self.get_raw(key).await
    }

    async fn put(&self, key: &str, value: &[u8]) -> crate::Result<()> {
        self.put_raw(key, value).await
    }

    async fn delete(&self, key: &str) -> crate::Result<()> {
        self.delete_raw(key).await
    }

    async fn exists(&self, key: &str) -> crate::Result<bool> {
        Ok(self.get_raw(key).await?.is_some())
    }

    async fn list_keys(&self, prefix: &str) -> crate::Result<Vec<String>> {
        let entries = self.scan_prefix_raw(prefix).await?;
        Ok(entries.into_iter().map(|(k, _)| k).collect())
    }
}

/// TransactionalStore 适配到 Storage（基于内存注册表管理事务ID）
#[async_trait]
impl kv_storage_if::TransactionalStore for Storage {
    async fn begin_transaction(&self) -> crate::Result<String> {
        // 从全局注册表中获取 Arc<Storage>
        // 使用 usize 作为键，因为 *const Storage 不能在线程间安全传递
        let storage_ptr = self as *const Storage as usize;
        let storage_arc = STORAGE_REGISTRY.lock()
            .map_err(|_| crate::Error::lock("无法获取存储注册表锁".to_string()))?
            .get(&storage_ptr)
            .ok_or_else(|| crate::Error::lock("Storage 未在注册表中找到".to_string()))?
            .clone();
        
        let tx = ModuleStorageTransaction::new(storage_arc);
        let id = tx.id.clone();
        MODULE_TX_REGISTRY.lock().map_err(|_| crate::Error::lock("无法获取事务注册表锁".to_string()))?.insert(id.clone(), tx);
        Ok(id)
    }

    async fn commit_transaction(&self, transaction_id: &str) -> crate::Result<()> {
        let mut registry = MODULE_TX_REGISTRY.lock().map_err(|_| crate::Error::lock("无法获取事务注册表锁".to_string()))?;
        if let Some(mut tx) = registry.remove(transaction_id) {
            // 显式调用 trait 方法
            core_storage_if::StorageTransaction::commit(&mut tx)
        } else {
            Err(crate::Error::Transaction(format!("事务不存在: {}", transaction_id)))
        }
    }

    async fn rollback_transaction(&self, transaction_id: &str) -> crate::Result<()> {
        let mut registry = MODULE_TX_REGISTRY.lock().map_err(|_| crate::Error::lock("无法获取事务注册表锁".to_string()))?;
        if let Some(mut tx) = registry.remove(transaction_id) {
            // 显式调用 trait 方法
            core_storage_if::StorageTransaction::rollback(&mut tx)
        } else {
            Err(crate::Error::Transaction(format!("事务不存在: {}", transaction_id)))
        }
    }
}

fn make_object_key(bucket: &str, key: &str) -> String { format!("object:{}:{}", bucket, key) }
fn make_object_prefix(bucket: &str, prefix: &str) -> String { format!("object:{}:{}", bucket, prefix) }

/// ObjectStore 适配到 Storage（使用前缀命名空间）
#[async_trait]
impl kv_storage_if::ObjectStore for Storage {
    async fn put_object(&self, bucket: &str, key: &str, data: &[u8]) -> crate::Result<()> {
        let full = make_object_key(bucket, key);
        self.put_raw(&full, data).await
    }

    async fn get_object(&self, bucket: &str, key: &str) -> crate::Result<Option<Vec<u8>>> {
        let full = make_object_key(bucket, key);
        self.get_raw(&full).await
    }

    async fn delete_object(&self, bucket: &str, key: &str) -> crate::Result<()> {
        let full = make_object_key(bucket, key);
        self.delete_raw(&full).await
    }

    async fn list_objects(&self, bucket: &str, prefix: &str) -> crate::Result<Vec<String>> {
        let full_prefix = make_object_prefix(bucket, prefix);
        let entries = self.scan_prefix_raw(&full_prefix).await?;
        let base = format!("object:{}:", bucket);
        Ok(entries.into_iter().map(|(k, _)| k.trim_start_matches(&base).to_string()).collect())
    }
}

/// 为对象元数据提供简单的键空间：object_meta:{bucket}:{key}
fn make_object_meta_key(bucket: &str, key: &str) -> String { format!("object_meta:{}:{}", bucket, key) }

#[async_trait]
impl kv_storage_if::ObjectMetadataStore for Storage {
    async fn put_object_metadata(&self, bucket: &str, key: &str, metadata: &std::collections::HashMap<String, String>) -> crate::Result<()> {
        let meta_key = make_object_meta_key(bucket, key);
        let data = serde_json::to_vec(metadata).map_err(|e| crate::Error::serialization(e.to_string()))?;
        self.put_raw(&meta_key, &data).await
    }

    async fn get_object_metadata(&self, bucket: &str, key: &str) -> crate::Result<Option<std::collections::HashMap<String, String>>> {
        let meta_key = make_object_meta_key(bucket, key);
        match self.get_raw(&meta_key).await? {
            Some(bytes) => {
                let map = serde_json::from_slice(&bytes).map_err(|e| crate::Error::deserialization(e.to_string()))?;
                Ok(Some(map))
            },
            None => Ok(None)
        }
    }
}

#[async_trait]
impl core_storage_if::StorageService for Storage {
    async fn store(&self, key: &str, value: &[u8]) -> crate::Result<()> { self.put_raw(key, value).await }
    async fn retrieve(&self, key: &str) -> crate::Result<Option<Vec<u8>>> { self.get_raw(key).await }
    async fn delete(&self, key: &str) -> crate::Result<()> { self.delete_raw(key).await }
    async fn exists(&self, key: &str) -> crate::Result<bool> { Ok(self.get_raw(key).await?.is_some()) }
    async fn list_keys(&self, prefix: &str) -> crate::Result<Vec<String>> {
        let entries = self.scan_prefix_raw(prefix).await?;
        Ok(entries.into_iter().map(|(k, _)| k).collect())
    }
    async fn batch_store(&self, items: Vec<(String, Vec<u8>)>) -> crate::Result<()> {
        for (k, v) in items { self.put_raw(&k, &v).await?; }
        Ok(())
    }
    async fn batch_retrieve(&self, keys: &[String]) -> crate::Result<Vec<Option<Vec<u8>>>> {
        let mut out = Vec::with_capacity(keys.len());
        for k in keys { out.push(self.get_raw(k).await?); }
        Ok(out)
    }
    async fn batch_delete(&self, keys: &[String]) -> crate::Result<()> {
        for k in keys { self.delete_raw(k).await?; }
        Ok(())
    }
    fn transaction(&self) -> crate::Result<Box<dyn core_storage_if::StorageTransaction>> {
        // 从全局注册表中获取 Arc<Storage>
        // 使用 usize 作为键，因为 *const Storage 不能在线程间安全传递
        let storage_ptr = self as *const Storage as usize;
        let storage_arc = STORAGE_REGISTRY.lock()
            .map_err(|_| crate::Error::lock("无法获取存储注册表锁".to_string()))?
            .get(&storage_ptr)
            .ok_or_else(|| crate::Error::lock("Storage 未在注册表中找到".to_string()))?
            .clone();
        
        Ok(Box::new(ModuleStorageTransaction::new(storage_arc)))
    }
    fn transaction_with_isolation(&self, _isolation_level: core_storage_if::IsolationLevel) -> crate::Result<Box<dyn core_storage_if::StorageTransaction>> {
        self.transaction()
    }
    async fn get_dataset_size(&self, dataset_id: &str) -> crate::Result<usize> {
        let key = format!("dataset_data:{}", dataset_id);
        match self.get_raw(&key).await? { Some(v) => Ok(v.len()), None => Err(crate::Error::not_found(format!("数据集不存在: {}", dataset_id))) }
    }
    async fn get_dataset_chunk(&self, dataset_id: &str, start: usize, end: usize) -> crate::Result<Vec<u8>> {
        let key = format!("dataset_data:{}", dataset_id);
        let data = self.get_raw(&key).await?.ok_or_else(|| crate::Error::not_found(format!("数据集不存在: {}", dataset_id)))?;
        let len = data.len();
        let s = start.min(len); let e = end.min(len);
        if s > e { return Ok(Vec::new()); }
        Ok(data[s..e].to_vec())
    }
    
    async fn save_model_parameters(&self, model_id: &str, params: &crate::model::parameters::ModelParameters) -> crate::Result<()> {
        let key = format!("model:parameters:{}", model_id);
        let value = bincode::serialize(params)?;
        self.put_raw(&key, &value).await
    }
    
    async fn get_model_parameters(&self, model_id: &str) -> crate::Result<Option<crate::model::parameters::ModelParameters>> {
        let key = format!("model:parameters:{}", model_id);
        if let Some(data) = self.get_raw(&key).await? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }
    
    async fn save_model_architecture(&self, model_id: &str, arch: &crate::model::ModelArchitecture) -> crate::Result<()> {
        let key = format!("model:architecture:{}", model_id);
        let value = bincode::serialize(arch)?;
        self.put_raw(&key, &value).await
    }
    
    async fn get_model_architecture(&self, model_id: &str) -> crate::Result<Option<crate::model::ModelArchitecture>> {
        let key = format!("model:architecture:{}", model_id);
        if let Some(data) = self.get_raw(&key).await? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }
    
    // 训练状态和训练结果相关方法已移除 - 向量数据库系统不需要训练功能
    
    async fn list_inference_results(&self, model_id: &str) -> crate::Result<Vec<String>> {
        let prefix = format!("inference:result:{}:", model_id);
        let entries = self.scan_prefix_raw(&prefix).await?;
        Ok(entries.into_iter().map(|(k, _)| k).collect())
    }
    
    async fn get_inference_result(&self, result_id: &str) -> crate::Result<Option<crate::core::results::InferenceResult>> {
        let key = format!("inference:result:{}", result_id);
        if let Some(data) = self.get_raw(&key).await? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }
    
    async fn save_inference_result(&self, model_id: &str, result: &crate::core::results::InferenceResult) -> crate::Result<()> {
        let key = format!("inference:result:{}:{}", model_id, uuid::Uuid::new_v4());
        let value = bincode::serialize(result)?;
        self.put_raw(&key, &value).await
    }
    
    async fn save_detailed_inference_result(&self, result_id: &str, result: &crate::core::results::InferenceResult) -> crate::Result<()> {
        let key = format!("inference:detailed:{}", result_id);
        let value = bincode::serialize(result)?;
        self.put_raw(&key, &value).await
    }
    
    async fn save_model_info(&self, model_id: &str, info: &crate::core::types::ModelInfo) -> crate::Result<()> {
        let key = format!("model:info:{}", model_id);
        let value = bincode::serialize(info)?;
        self.put_raw(&key, &value).await
    }
    
    async fn get_model_info(&self, model_id: &str) -> crate::Result<Option<crate::core::types::ModelInfo>> {
        let key = format!("model:info:{}", model_id);
        if let Some(data) = self.get_raw(&key).await? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }
    
    async fn save_model_metrics(&self, model_id: &str, metrics: &crate::storage::models::implementation::ModelMetrics) -> crate::Result<()> {
        let key = format!("model:metrics:{}", model_id);
        let value = bincode::serialize(metrics)?;
        self.put_raw(&key, &value).await
    }
    
    async fn get_model_metrics(&self, model_id: &str) -> crate::Result<Option<crate::storage::models::implementation::ModelMetrics>> {
        let key = format!("model:metrics:{}", model_id);
        if let Some(data) = self.get_raw(&key).await? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }
    
    async fn get_model(&self, model_id: &str) -> crate::Result<Option<crate::model::Model>> {
        let key = format!("model:{}", model_id);
        if let Some(data) = self.get_raw(&key).await? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }
    
    async fn save_model(&self, model_id: &str, model: &crate::model::Model) -> crate::Result<()> {
        let key = format!("model:{}", model_id);
        let value = bincode::serialize(model)?;
        self.put_raw(&key, &value).await
    }
    
    async fn model_exists(&self, model_id: &str) -> crate::Result<bool> {
        let key = format!("model:{}", model_id);
        Ok(self.get_raw(&key).await?.is_some())
    }
    
    async fn has_model(&self, model_id: &str) -> crate::Result<bool> {
        self.model_exists(model_id).await
    }
    
    async fn get_dataset(&self, dataset_id: &str) -> crate::Result<Option<crate::data::loader::types::DataSchema>> {
        let key = format!("dataset:schema:{}", dataset_id);
        if let Some(data) = self.get_raw(&key).await? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }
}

// 数据集相关方法实现
impl Storage {
    /// 获取数据集ID列表
    pub async fn get_dataset_ids(&self) -> crate::Result<Vec<String>> {
        let prefix = "dataset:";
        let entries = self.scan_prefix_raw(prefix).await?;
        let mut ids = Vec::new();
        for (key, _) in entries {
            if let Some(id) = key.strip_prefix(prefix) {
                // 移除可能的后缀（如:info, :metadata等）
                if let Some(id_part) = id.split(':').next() {
                    ids.push(id_part.to_string());
                }
            }
        }
        ids.sort();
        ids.dedup();
        Ok(ids)
    }
    
    /// 获取数据集信息
    pub async fn get_dataset_info(&self, id: &str) -> crate::Result<Option<serde_json::Value>> {
        let key = format!("dataset:{}:info", id);
        if let Some(data) = self.get_raw(&key).await? {
            Ok(Some(serde_json::from_slice(&data)?))
        } else {
            Ok(None)
        }
    }
    
    /// 保存数据集信息
    pub async fn save_dataset_info(&self, dataset: &serde_json::Value) -> crate::Result<()> {
        let dataset_id = dataset.get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| crate::Error::InvalidInput("Dataset must contain 'id' field".to_string()))?;
        let key = format!("dataset:{}:info", dataset_id);
        let data = serde_json::to_vec(dataset)?;
        self.put_raw(&key, &data).await
    }
    
    /// 获取数据集特征统计
    pub async fn get_dataset_feature_stats(&self, id: &str) -> crate::Result<serde_json::Value> {
        let key = format!("dataset:{}:feature_stats", id);
        if let Some(data) = self.get_raw(&key).await? {
            Ok(serde_json::from_slice(&data)?)
        } else {
            Ok(serde_json::json!([]))
        }
    }
    
    /// 查询数据集
    pub async fn query_dataset(
        &self,
        name: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> crate::Result<Vec<serde_json::Value>> {
        // 获取数据集数据
        let data_key = format!("dataset:{}:data", name);
        if let Some(data) = self.get_raw(&data_key).await? {
            // 解析为JSON数组
            let mut rows: Vec<serde_json::Value> = serde_json::from_slice(&data)?;
            
            // 应用offset和limit
            let offset = offset.unwrap_or(0);
            let limit = limit.unwrap_or(rows.len());
            
            let start = offset.min(rows.len());
            let end = (start + limit).min(rows.len());
            
            Ok(rows.drain(start..end).collect())
        } else {
            Ok(Vec::new())
        }
    }
}

// 基本操作实现
impl Storage {
    /// 添加数据
    pub fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.db.put(key, value)?;
        
        // 如果启用缓存，更新缓存
        if self.config.enable_cache {
            let key_str = match std::str::from_utf8(key) {
                Ok(s) => s.to_string(),
                Err(_) => return Ok(()), // 如果键不是UTF-8，不缓存
            };
            
            let mut cache = self.cache.write()
                .map_err(|e| Error::lock(format!("无法获取缓存写锁: {}", e)))?;
            let mut cache_size = self.cache_size.write()
                .map_err(|e| Error::lock(format!("无法获取缓存大小写锁: {}", e)))?;
            
            // 计算大小变化
            let value_len = value.len();
            let old_len = cache.get(&key_str).map_or(0, |e| e.data.len());
            let size_delta = value_len as isize - old_len as isize;
            
            // 检查是否需要清理缓存
            if *cache_size as isize + size_delta > self.config.max_cache_size as isize {
                // 根据缓存策略清理
                match self.config.cache_policy {
                    CachePolicy::LRU => {
                        // 找到最不常用的条目删除
                        if let Some(oldest_key) = cache.iter()
                            .min_by_key(|(_, entry)| entry.timestamp)
                            .map(|(k, _)| k.clone())
                        {
                            if let Some(entry) = cache.get(&oldest_key) {
                                let removed_size = entry.data.len();
                                cache.remove(&oldest_key);
                                *cache_size = cache_size.saturating_sub(removed_size);
                            }
                        }
                    },
                    CachePolicy::MRU => {
                        // 找到最常用的条目删除
                        if let Some(newest_key) = cache.iter()
                            .max_by_key(|(_, entry)| entry.timestamp)
                            .map(|(k, _)| k.clone())
                        {
                            if let Some(entry) = cache.get(&newest_key) {
                                let removed_size = entry.data.len();
                                cache.remove(&newest_key);
                                *cache_size = cache_size.saturating_sub(removed_size);
                            }
                        }
                    },
                    CachePolicy::FIFO => {
                        // 先进先出，删除最早插入的
                        if let Some(oldest_key) = cache.iter()
                            .min_by_key(|(_, entry)| entry.timestamp)
                            .map(|(k, _)| k.clone())
                        {
                            if let Some(entry) = cache.get(&oldest_key) {
                                let removed_size = entry.data.len();
                                cache.remove(&oldest_key);
                                *cache_size = cache_size.saturating_sub(removed_size);
                            }
                        }
                    },
                }
            }
            
            // 更新缓存
            cache.insert(key_str, CacheEntry {
                data: value.to_vec(),
                timestamp: Utc::now(),
            });
            *cache_size = (*cache_size as isize + size_delta) as usize;
        }
        
        Ok(())
    }
    
    /// 获取数据
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        // 如果启用缓存，先检查缓存
        if self.config.enable_cache {
            if let Ok(key_str) = std::str::from_utf8(key) {
                let mut cache = self.cache.write()
                    .map_err(|e| Error::lock(format!("无法获取缓存读锁: {}", e)))?;
                
                if let Some(entry) = cache.get_mut(key_str) {
                    // 更新访问时间（对于LRU策略）
                    if self.config.cache_policy == CachePolicy::LRU {
                        entry.timestamp = Utc::now();
                    }
                    return Ok(Some(entry.data.clone()));
                }
            }
        }
        
        // 缓存未命中，从数据库读取
        match self.db.get(key) {
            Ok(Some(value)) => {
                // 如果启用缓存，将结果加入缓存
                if self.config.enable_cache {
                    if let Ok(key_str) = std::str::from_utf8(key) {
                        let mut cache = self.cache.write()
                            .map_err(|e| Error::lock(format!("无法获取缓存写锁: {}", e)))?;
                        let mut cache_size = self.cache_size.write()
                            .map_err(|e| Error::lock(format!("无法获取缓存大小写锁: {}", e)))?;
                        
                        // 添加到缓存
                        let value_len = value.len();
                        cache.insert(key_str.to_string(), CacheEntry {
                            data: value.clone(),
                            timestamp: Utc::now(),
                        });
                        *cache_size += value_len;
                        
                        // 如果缓存超出大小限制，删除最旧的条目
                        while *cache_size > self.config.max_cache_size && !cache.is_empty() {
                            let oldest_key = match self.config.cache_policy {
                                CachePolicy::LRU | CachePolicy::FIFO => {
                                    cache.iter()
                                        .min_by_key(|(_, entry)| entry.timestamp)
                                        .map(|(k, _)| k.clone())
                                },
                                CachePolicy::MRU => {
                                    cache.iter()
                                        .max_by_key(|(_, entry)| entry.timestamp)
                                        .map(|(k, _)| k.clone())
                                },
                            };
                            
                            if let Some(key_to_remove) = oldest_key {
                                if let Some(entry) = cache.get(&key_to_remove) {
                                    let removed_size = entry.data.len();
                                    cache.remove(&key_to_remove);
                                    *cache_size = cache_size.saturating_sub(removed_size);
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                    }
                }
                
                Ok(Some(value))
            },
            Ok(None) => Ok(None),
            Err(e) => Err(Error::StorageError(format!("RocksDB read error: {}", e))),
        }
    }
    
    /// 删除数据
    pub fn delete(&self, key: &[u8]) -> Result<()> {
        self.db.delete(key)?;
        
        // 如果启用缓存，也从缓存中删除
        if self.config.enable_cache {
            if let Ok(key_str) = std::str::from_utf8(key) {
                let mut cache = self.cache.write()
                    .map_err(|e| Error::lock(format!("无法获取缓存写锁: {}", e)))?;
                let mut cache_size = self.cache_size.write()
                    .map_err(|e| Error::lock(format!("无法获取缓存大小写锁: {}", e)))?;
                
                if let Some(entry) = cache.remove(key_str) {
                    *cache_size = cache_size.saturating_sub(entry.data.len());
                }
            }
        }
        
        Ok(())
    }
    
    /// 批量添加数据
    pub fn batch_put(&self, batch: &[(Vec<u8>, Vec<u8>)]) -> Result<()> {
        let mut write_batch = WriteBatch::default();
        
        for (key, value) in batch {
            write_batch.put(&key, &value);
        }
        
        self.db.write(write_batch)?;
        
        // 如果启用缓存，批量更新缓存
        if self.config.enable_cache {
            let mut cache = self.cache.write()
                .map_err(|e| Error::lock(format!("无法获取缓存写锁: {}", e)))?;
            let mut cache_size = self.cache_size.write()
                .map_err(|e| Error::lock(format!("无法获取缓存大小写锁: {}", e)))?;
            
            for (key, value) in batch {
                if let Ok(key_str) = std::str::from_utf8(key) {
                    let value_len = value.len();
                    let old_len = cache.get(key_str).map_or(0, |e| e.data.len());
                    let size_delta = value_len as isize - old_len as isize;
                    
                    // 添加到缓存
                    cache.insert(key_str.to_string(), CacheEntry {
                        data: value.clone(),
                        timestamp: Utc::now(),
                    });
                    *cache_size = (*cache_size as isize + size_delta).max(0) as usize;
                }
            }
            
            // 如果缓存超出大小限制，清理
            while *cache_size > self.config.max_cache_size && !cache.is_empty() {
                let oldest_key = match self.config.cache_policy {
                    CachePolicy::LRU | CachePolicy::FIFO => {
                        cache.iter()
                            .min_by_key(|(_, entry)| entry.timestamp)
                            .map(|(k, _)| k.clone())
                    },
                    CachePolicy::MRU => {
                        cache.iter()
                            .max_by_key(|(_, entry)| entry.timestamp)
                            .map(|(k, _)| k.clone())
                    },
                };
                
                if let Some(key_to_remove) = oldest_key {
                    if let Some(entry) = cache.get(&key_to_remove) {
                        let removed_size = entry.data.len();
                        cache.remove(&key_to_remove);
                        *cache_size = cache_size.saturating_sub(removed_size);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        
        Ok(())
    }
    
    /// 获取缓存大小
    pub fn get_cache_size(&self) -> Result<usize> {
        Ok(*self.cache_size.read()
            .map_err(|e| Error::lock(format!("无法获取缓存大小读锁: {}", e)))?)
    }
    
    /// 获取缓存条目数
    pub fn get_cache_count(&self) -> Result<usize> {
        Ok(self.cache.read()
            .map_err(|e| Error::lock(format!("无法获取缓存读锁: {}", e)))?
            .len())
    }
    
    /// 清空缓存
    pub fn clear_cache(&self) -> Result<()> {
        let mut cache = self.cache.write()
            .map_err(|e| Error::lock(format!("无法获取缓存写锁: {}", e)))?;
        let mut cache_size = self.cache_size.write()
            .map_err(|e| Error::lock(format!("无法获取缓存大小写锁: {}", e)))?;
        
        cache.clear();
        *cache_size = 0;
        
        Ok(())
    }
    
    /// 创建事务
    pub fn begin_transaction(&self) -> Result<WriteBatch> {
        Ok(WriteBatch::default())
    }
    
    /// 获取原始数据列族
    pub fn cf_raw_data(&self) -> Result<rocksdb::BoundColumnFamily> {
        // 注意：BoundColumnFamily 是生命周期绑定的，不能直接返回
        // 这里返回错误，提示需要使用不同的方法
        Err(Error::NotImplemented("cf_raw_data requires lifetime-bound ColumnFamily. Use db.cf_handle() directly instead.".to_string()))
    }
    
    /// 获取数据库引用
    pub fn db(&self) -> &DB {
        &self.db
    }
    
    /// 写入批量操作
    pub fn write(&self, batch: WriteBatch) -> Result<()> {
        self.db.write(batch)?;
        Ok(())
    }
    
    /// 创建写入批处理
    pub fn create_write_batch(&self) -> WriteBatch {
        WriteBatch::default()
    }
    
    /// 批量写入数据
    pub fn batch_write(&self, operations: Vec<(Vec<u8>, Option<Vec<u8>>)>) -> Result<()> {
        let mut batch = self.create_write_batch();
        for (key, value) in operations {
            match value {
                Some(data) => batch.put(&key, &data),
                None => batch.delete(&key),
            }
        }
        self.db.write(batch)?;
        Ok(())
    }
    
    /// 获取数据批次（异步包装）
    pub async fn get_data_batch(self: &Arc<Self>, batch_id: &str) -> Result<crate::data::DataBatch> {
        let storage = Arc::clone(self);
        let batch_id = batch_id.to_string();
        tokio::task::spawn_blocking(move || {
            let key = format!("data_batch:{}", batch_id);
            match storage.get(key.as_bytes())? {
                Some(data) => {
                    let batch: crate::data::DataBatch = bincode::deserialize(&data)?;
                    Ok(batch)
                },
                None => Err(Error::NotFound(format!("数据批次 {} 不存在", batch_id)))
            }
        }).await.map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }
    
    /// 获取存储统计信息
    pub async fn get_statistics(self: &Arc<Self>) -> Result<crate::storage::engine::types::StorageStatistics> {
        let storage = Arc::clone(self);
        tokio::task::spawn_blocking(move || {
            // 扫描所有键来计算统计信息
            let mut total_objects = 0u64;
            let mut total_size_bytes = 0u64;
            let iter = storage.db().iterator(rocksdb::IteratorMode::Start);
            
            for item in iter {
                match item {
                    Ok((key, value)) => {
                        total_objects += 1;
                        total_size_bytes += key.len() as u64 + value.len() as u64;
                    }
                    Err(e) => {
                        warn!("统计扫描错误: {}", e);
                    }
                }
            }
            
            Ok(crate::storage::engine::types::StorageStatistics {
                total_objects,
                total_size_bytes,
                read_operations: 0, // 需要从监控系统获取
                write_operations: 0, // 需要从监控系统获取
                cache_hit_rate: 0.0, // 需要从缓存系统获取
                average_operation_time_ms: 0.0, // 需要从监控系统获取
                peak_memory_usage_bytes: 0, // 需要从监控系统获取
                active_connections: 0, // 需要从连接池获取
                last_backup_time: 0, // 需要从备份系统获取
                database_health_score: 1.0, // 默认健康
            })
        }).await.map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }
    
    /// 获取存储指标
    pub async fn get_metrics(self: &Arc<Self>) -> Result<crate::compat::api::routes::system::types::StorageMetrics> {
        let storage = Arc::clone(self);
        tokio::task::spawn_blocking(move || {
            // 扫描所有键来计算指标
            let mut total_objects = 0u64;
            let mut total_size_bytes = 0u64;
            let iter = storage.db().iterator(rocksdb::IteratorMode::Start);
            
            for item in iter {
                match item {
                    Ok((key, value)) => {
                        total_objects += 1;
                        total_size_bytes += key.len() as u64 + value.len() as u64;
                    }
                    Err(e) => {
                        warn!("指标扫描错误: {}", e);
                    }
                }
            }
            
            Ok(crate::compat::api::routes::system::types::StorageMetrics {
                total_objects,
                total_size_bytes,
                read_operations: 0, // 需要从监控系统获取
                write_operations: 0, // 需要从监控系统获取
                operations_per_second: 0.0, // 需要从监控系统获取
            })
        }).await.map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }
    
    /// 获取数据集存储服务（返回适配器）
    pub fn dataset_storage(self: &Arc<Self>) -> DatasetStorageAdapter {
        DatasetStorageAdapter {
            storage: Arc::clone(self),
        }
    }
    
    // 训练相关方法已迁移到 training_ops.rs，通过模块导入确保可用
    // 模型相关方法已迁移到 model_ops.rs，通过模块导入确保可用
    
    /// 获取存储统计信息
    pub async fn get_stats(&self) -> crate::Result<crate::compat::api::routes::system::types::StorageStatsResponse> {
        use chrono::Utc;
        use std::collections::HashMap;
        
        // 扫描不同类型的文件
        let data_entries = self.scan_prefix_raw("dataset:").await?;
        let model_entries = self.scan_prefix_raw("model:").await?;
        let algorithm_entries = self.scan_prefix_raw("algorithm:").await?;
        let all_entries = self.scan_prefix_raw("").await?;
        
        let mut total_size_bytes = 0u64;
        let mut usage_by_type = HashMap::new();
        
        for (_, value) in &all_entries {
            total_size_bytes += value.len() as u64;
        }
        
        usage_by_type.insert("data".to_string(), data_entries.len() as u64);
        usage_by_type.insert("model".to_string(), model_entries.len() as u64);
        usage_by_type.insert("algorithm".to_string(), algorithm_entries.len() as u64);
        
        Ok(crate::compat::api::routes::system::types::StorageStatsResponse {
            timestamp: Utc::now(),
            data_files_count: data_entries.len() as u64,
            model_files_count: model_entries.len() as u64,
            algorithm_files_count: algorithm_entries.len() as u64,
            total_size_bytes,
            usage_by_type,
        })
    }
    
    /// 获取计数器值
    pub async fn get_counter(&self, name: &str) -> crate::Result<u64> {
        let key = format!("counter:{}", name);
        if let Some(data) = self.get_raw(&key).await? {
            Ok(serde_json::from_slice::<u64>(&data).unwrap_or(0))
        } else {
            Ok(0)
        }
    }
    
    /// 统计模型数量
    pub async fn count_models(&self) -> crate::Result<usize> {
        let prefix = "model:";
        let entries = self.scan_prefix_raw(prefix).await?;
        Ok(entries.len())
    }
    
    /// 获取最近的模型列表
    pub async fn get_recent_models(&self, limit: usize) -> crate::Result<Vec<String>> {
        let prefix = "model:";
        let entries = self.scan_prefix_raw(prefix).await?;
        let mut models: Vec<String> = entries.into_iter()
            .map(|(key, _)| key.strip_prefix(prefix).unwrap_or(&key).to_string())
            .collect();
        models.sort();
        models.reverse();
        models.truncate(limit);
        Ok(models)
    }
    
    /// 统计任务数量
    pub async fn count_tasks(&self) -> crate::Result<usize> {
        let prefix = "task:";
        let entries = self.scan_prefix_raw(prefix).await?;
        Ok(entries.len())
    }
    
    /// 获取最近的任务列表
    pub async fn get_recent_tasks(&self, limit: usize) -> crate::Result<Vec<String>> {
        let prefix = "task:";
        let entries = self.scan_prefix_raw(prefix).await?;
        let mut tasks: Vec<String> = entries.into_iter()
            .map(|(key, _)| key.strip_prefix(prefix).unwrap_or(&key).to_string())
            .collect();
        tasks.sort();
        tasks.reverse();
        tasks.truncate(limit);
        Ok(tasks)
    }
    
    /// 获取日志列表
    pub async fn get_logs(&self, level: &str, limit: usize) -> crate::Result<Vec<String>> {
        let prefix = format!("log:{}:", level);
        let entries = self.scan_prefix_raw(&prefix).await?;
        let mut logs: Vec<String> = entries.into_iter()
            .filter_map(|(_, value)| String::from_utf8(value).ok())
            .collect();
        logs.reverse();
        logs.truncate(limit);
        Ok(logs)
    }
    
    /// 获取配置
    pub fn get_config(&self) -> &StorageConfig {
        &self.config
    }
    
    /// 更新配置
    pub async fn update_config(&self, config: &serde_json::Value) -> crate::Result<()> {
        // 将配置保存到存储中
        let key = "storage:config";
        let data = serde_json::to_vec(config)?;
        self.put_raw(key, &data).await
    }
    
    /// 获取存储信息
    pub async fn get_info(&self) -> crate::Result<serde_json::Value> {
        Ok(serde_json::json!({
            "path": self.config.path,
            "cache_size": *self.cache_size.read().unwrap(),
            "status": "ok"
        }))
    }
    
    /// 统计活跃任务数量
    pub async fn count_active_tasks(&self) -> crate::Result<usize> {
        let prefix = "task:active:";
        let entries = self.scan_prefix_raw(prefix).await?;
        Ok(entries.len())
    }
    
    /// 获取API统计信息
    pub async fn get_api_stats(&self) -> crate::Result<std::collections::HashMap<String, u64>> {
        let key = "api:stats";
        if let Some(data) = self.get_raw(key).await? {
            Ok(serde_json::from_slice(&data)?)
        } else {
            Ok(std::collections::HashMap::new())
        }
    }
}

/// 数据集存储适配器
/// 用于适配 Storage 到 DatasetStorageInterface
pub struct DatasetStorageAdapter {
    storage: Arc<Storage>,
}

impl DatasetStorageAdapter {
    /// 获取数据集
    pub async fn get_dataset(&self, dataset_id: &str) -> Result<Option<crate::data::Dataset>> {
        let storage = Arc::clone(&self.storage);
        let dataset_id = dataset_id.to_string();
        tokio::task::spawn_blocking(move || {
            let key = format!("dataset:{}", dataset_id);
            match storage.get(key.as_bytes())? {
                Some(data) => {
                    let dataset: crate::data::Dataset = bincode::deserialize(&data)
                        .map_err(|e| Error::Serialization(format!("反序列化数据集失败: {}", e)))?;
                    Ok(Some(dataset))
                }
                None => Ok(None)
            }
        }).await.map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }
}

// Trait实现已经在拆分后的模块文件中完成，通过模块导入确保可用

// 默认实现
// 注意：Default trait 需要返回 Self，但 Storage 的创建需要配置
// 这里提供一个最小化的实现，但强烈建议使用 Storage::new() 或 Storage::new_in_memory()
impl Default for Storage {
    fn default() -> Self {
        // 使用临时目录创建最小化的存储实例
        // 注意：这不是推荐的使用方式，仅用于满足Default trait的要求
        use std::env;
        let temp_dir = env::temp_dir().join("vecmindb_default_storage");
        let config = StorageConfig {
            path: temp_dir.to_string_lossy().to_string(),
            ..Default::default()
        };
        
        // Default trait 必须返回 Self，无法返回 Result
        // 因此当创建失败时只能 panic，这是 Rust trait 系统的限制
        // 生产环境应避免使用 Default，改用 Storage::new() 或 Storage::new_in_memory()
        log::warn!("Storage::default() 被调用，这不是推荐的使用方式。请使用 Storage::new() 或 Storage::new_in_memory() 代替。");
        
        match Self::open(config) {
            Ok(storage) => storage,
            Err(e) => {
                let error_msg = format!(
                    "Storage::default() 创建失败: {}。请使用 Storage::new() 或 Storage::new_in_memory() 代替。",
                    e
                );
                log::error!("{}", error_msg);
                panic!("{}", error_msg)
            }
        }
    }
}