use crate::Result;
use crate::Error;
use crate::config::Config;
use crate::storage::models::implementation::{ModelStorage, StorageOptions};
use std::sync::Arc;
use std::collections::HashMap;

// 导入实现类 - 改为使用 use 语句而不是 mod 声明
use crate::storage::models::file_storage::FileModelStorage;
use crate::storage::models::rocksdb_storage::RocksDBModelStorage;

/// 模型存储类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStorageType {
    /// 本地文件系统
    Local,
    /// 分布式文件系统
    Distributed,
    /// 对象存储
    ObjectStorage,
    /// 数据库
    Database,
}

impl std::str::FromStr for ModelStorageType {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "file" => Ok(ModelStorageType::Local),
            "rocksdb" => Ok(ModelStorageType::Local),
            "memory" => Ok(ModelStorageType::Local),
            "local" => Ok(ModelStorageType::Local),
            "distributed" => Ok(ModelStorageType::Distributed),
            "object" => Ok(ModelStorageType::ObjectStorage),
            "database" => Ok(ModelStorageType::Database),
            _ => Err(Error::invalid_argument(format!("不支持的模型存储类型: {}", s))),
        }
    }
}

/// 模型存储工厂接口
pub trait ModelStorageFactory: Send + Sync {
    /// 创建模型存储实例
    fn create(&self, config: HashMap<String, String>) -> Result<Box<dyn ModelStorage>>;
    
    /// 获取工厂支持的存储类型
    fn get_supported_types(&self) -> Vec<ModelStorageType>;
}

/// 模型存储工厂
pub struct ModelStorageFactoryImpl;

impl ModelStorageFactory for ModelStorageFactoryImpl {
    /// 创建模型存储实例
    fn create(&self, config: HashMap<String, String>) -> Result<Box<dyn ModelStorage>> {
        // 实现创建模型存储实例的逻辑
        Err(Error::not_implemented("模型存储工厂尚未实现完整功能"))
    }
    
    /// 获取工厂支持的存储类型
    fn get_supported_types(&self) -> Vec<ModelStorageType> {
        vec![ModelStorageType::Local, ModelStorageType::Database]
    }
}

/// 模型存储实用工具
pub struct ModelStorageUtil;

impl ModelStorageUtil {
    /// 创建模型存储
    pub fn create(
        storage_type: ModelStorageType,
        options: StorageOptions,
    ) -> Result<Arc<dyn ModelStorage>> {
        match storage_type {
            ModelStorageType::Local => {
                let storage = FileModelStorage::new(options)?;
                Ok(Arc::new(storage))
            },
            ModelStorageType::Database => {
                let storage = RocksDBModelStorage::new(options)?;
                Ok(Arc::new(storage))
            },
            ModelStorageType::Distributed | ModelStorageType::ObjectStorage => {
                Err(Error::not_implemented("分布式和对象存储尚未实现"))
            },
        }
    }

    /// 根据配置创建模型存储
    pub fn create_from_config(config: &Config) -> Result<Arc<dyn ModelStorage>> {
        // 从配置中获取存储类型（LegacyConfig 结构体直接访问字段）
        let storage_type_str = "rocksdb"; // 默认值，LegacyConfig 没有 model_storage 字段
        
        let storage_type = storage_type_str.parse::<ModelStorageType>()?;
        
        // 构建存储选项
        let mut options = StorageOptions {
            path: "./data/models".to_string(), // 默认路径
            cache_size_mb: None,
            compression_level: None,
            compression_type: None, // 稍后根据配置设置
            encryption_type: None, // 稍后根据配置设置
            encryption_key: None,
            shard_size_kb: None,
            backup_frequency_hours: None,
            max_backups: None,
            create_if_missing: true,
            use_wal: true,
            enable_multithreading: true,
            max_threads: None,
            preallocation_size_mb: None,
            read_cache_size_kb: None,
            write_cache_size_kb: None,
            persistence_options: Default::default(), // 使用默认值
        };
        
        // 创建存储实例
        Self::create(storage_type, options)
    }
    
    /// 创建默认的模型存储
    pub fn create_default() -> Result<Arc<dyn ModelStorage>> {
        let options = StorageOptions {
            path: "./data/models".to_string(),
            create_if_missing: true,
            ..Default::default()
        };
        
        Self::create(ModelStorageType::Database, options)
    }
} 