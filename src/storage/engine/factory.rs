// src/storage/engine/factory.rs
//
// 存储引擎工厂
// 用于创建和管理不同类型的存储引擎

use std::collections::HashMap;
use crate::Result;
use crate::Error;
use super::interfaces::StorageEngine;
use super::core::StorageEngineImpl;
use crate::storage::config::StorageConfig;
use crate::data::schema::Schema;
use crate::data::record::Record;

/// 存储引擎工厂
/// 
/// 用于创建和管理不同类型的存储引擎实例
pub struct StorageEngineFactory;

impl StorageEngineFactory {
    /// 创建新的存储引擎工厂
    pub fn new() -> Self {
        Self {}
    }
    
    /// 创建默认存储引擎
    pub fn create_default(&self) -> Result<Box<dyn IStorageEngine>> {
        let config = StorageConfig::default();
        self.create("local", &HashMap::new())
    }
    
    /// 根据类型和选项创建存储引擎
    pub fn create(&self, engine_type: &str, options: &HashMap<String, String>) -> Result<Box<dyn IStorageEngine>> {
        match engine_type {
            "local" => {
                let config = self.create_config_from_options(options);
                let engine = StorageEngineImpl::new(config)?;
                Ok(Box::new(StorageEngineWrapper::new(engine)))
            },
            "memory" => {
                let mut config = self.create_config_from_options(options);
                config.path = std::path::PathBuf::from("memory://storage");
                let engine = StorageEngineImpl::new(config)?;
                Ok(Box::new(StorageEngineWrapper::new(engine)))
            },
            _ => Err(Error::config(format!("不支持的存储引擎类型: {}", engine_type))),
        }
    }
    
    /// 从选项创建配置
    fn create_config_from_options(&self, options: &HashMap<String, String>) -> StorageConfig {
        let mut config = StorageConfig::default();
        
        if let Some(path) = options.get("path") {
            config.path = std::path::PathBuf::from(path);
        }
        
        if let Some(create_if_missing) = options.get("create_if_missing") {
            config.create_if_missing = create_if_missing.to_lowercase() == "true";
        }
        
        if let Some(compression) = options.get("compression") {
            config.use_compression = compression.to_lowercase() == "true";
        }
        
        config
    }
}

/// 存储引擎接口
/// 定义存储引擎的通用操作
pub trait IStorageEngine: Send + Sync {
    /// 检查路径是否存在
    fn exists(&self, path: &str) -> Result<bool>;
    
    /// 创建路径
    fn create(&self, path: &str) -> Result<()>;
    
    /// 删除路径
    fn delete(&self, path: &str) -> Result<()>;
    
    /// 写入数据批次
    fn write_batch(&mut self, location: &str, schema: &Schema, records: &[Record]) -> Result<()>;
    
    /// 关闭引擎
    fn close(&self) -> Result<()>;
}

/// 存储引擎包装器
/// 将StorageEngineImpl适配为IStorageEngine接口
struct StorageEngineWrapper {
    inner: StorageEngineImpl,
}

impl StorageEngineWrapper {
    /// 创建新的包装器
    pub fn new(engine: StorageEngineImpl) -> Self {
        Self { inner: engine }
    }
}

impl IStorageEngine for StorageEngineWrapper {
    fn exists(&self, path: &str) -> Result<bool> {
        // 简单实现，检查路径是否存在
        Ok(std::path::Path::new(path).exists())
    }
    
    fn create(&self, path: &str) -> Result<()> {
        // 创建目录
        std::fs::create_dir_all(path).map_err(|e| Error::storage(format!("创建路径失败: {}", e)))?;
        Ok(())
    }
    
    fn delete(&self, path: &str) -> Result<()> {
        // 删除路径
        if std::path::Path::new(path).is_dir() {
            std::fs::remove_dir_all(path).map_err(|e| Error::storage(format!("删除目录失败: {}", e)))?;
        } else {
            std::fs::remove_file(path).map_err(|e| Error::storage(format!("删除文件失败: {}", e)))?;
        }
        Ok(())
    }
    
    fn write_batch(&mut self, _location: &str, _schema: &Schema, _records: &[Record]) -> Result<()> {
        // 批量写入功能需要完整的parquet格式化和写入实现，当前返回特征未启用错误
        // 如需使用批量写入，请集成parquet库或使用StorageEngineImpl的方法
        Err(Error::feature_not_enabled(
            "批量写入功能需要完整的parquet格式化和写入实现，当前未集成。请使用StorageEngineImpl的方法或集成parquet库。".to_string()
        ))
    }
    
    fn close(&self) -> Result<()> {
        // 关闭引擎
        self.inner.close()
    }
} 