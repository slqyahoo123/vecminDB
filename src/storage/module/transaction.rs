// 事务模块
// 提供事务支持和批量操作功能

use crate::{Error, Result};
use serde_json::{self, Value};
use rocksdb::{DB, WriteBatch};
use super::data::{DATA_RAW_PREFIX, DATA_PROCESSED_PREFIX};
use uuid::Uuid;

/// 版本相关常量
pub const VERSION_PREFIX: &str = "version:";
pub const CF_VERSIONS: &str = "versions";
pub const CF_INFO: &str = "info";

/// 事务辅助功能的trait
pub trait WriteBatchExt {
    fn store_raw_data(&mut self, key: &str, value: Vec<String>) -> Result<()>;
    fn update_raw_data(&mut self, key: &str, value: Vec<String>) -> Result<()>;
    fn commit(&mut self) -> Result<()>;
    fn rollback(&mut self) -> Result<()>;
}

/// 存储事务
pub struct StorageTransaction<'a> {
    db: &'a DB,
    batch: WriteBatch,
}

impl<'a> StorageTransaction<'a> {
    /// 创建新事务
    pub fn new(db: &'a DB) -> Self {
        Self {
            db,
            batch: WriteBatch::default(),
        }
    }
    
    /// 添加数据
    pub fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        self.batch.put(key, value);
        Ok(())
    }
    
    /// 删除数据
    pub fn delete(&mut self, key: &[u8]) -> Result<()> {
        self.batch.delete(key);
        Ok(())
    }
    
    /// 存储原始数据
    pub fn store_raw_data(&mut self, data_id: &str, data: Vec<String>) -> Result<()> {
        let key = format!("{}{}", DATA_RAW_PREFIX, data_id);
        let value = serde_json::to_string(&data)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        self.batch.put(key.as_bytes(), value.as_bytes());
        Ok(())
    }
    
    /// 更新原始数据
    pub fn update_raw_data(&mut self, data_id: &str, data: Vec<String>) -> Result<()> {
        let key = format!("{}{}", DATA_RAW_PREFIX, data_id);
        let value = serde_json::to_string(&data)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        self.batch.put(key.as_bytes(), value.as_bytes());
        Ok(())
    }
    
    /// 获取原始数据
    pub fn get_raw_data(&self, data_id: &str) -> Result<Option<Vec<String>>> {
        let key = format!("{}{}", DATA_RAW_PREFIX, data_id);
        if let Some(value) = self.db.get(key.as_bytes())? {
            let data: Vec<String> = serde_json::from_slice(&value)
                .map_err(|e| Error::Deserialization(e.to_string()))?;
            Ok(Some(data))
        } else {
            Ok(None)
        }
    }
    
    /// 存储处理后的数据
    pub fn store_processed_data(&mut self, data_id: &str, data: Vec<String>) -> Result<()> {
        let key = format!("{}{}", DATA_PROCESSED_PREFIX, data_id);
        let value = serde_json::to_string(&data)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        self.batch.put(key.as_bytes(), value.as_bytes());
        Ok(())
    }
    
    /// 获取处理后的数据
    pub fn get_processed_data(&self, data_id: &str) -> Result<Option<Vec<String>>> {
        let key = format!("{}{}", DATA_PROCESSED_PREFIX, data_id);
        if let Some(value) = self.db.get(key.as_bytes())? {
            let data: Vec<String> = serde_json::from_slice(&value)
                .map_err(|e| Error::Deserialization(e.to_string()))?;
            Ok(Some(data))
        } else {
            Ok(None)
        }
    }
    
    /// 创建版本
    pub fn create_version(&mut self, data_id: &str, parent_version: Option<&str>) -> Result<String> {
        let version_id = Uuid::new_v4().to_string();
        let version_key = format!("{}{}", VERSION_PREFIX, version_id);
        
        let mut version_data = serde_json::Map::new();
        version_data.insert("id".to_string(), serde_json::to_value(version_id.clone())?);
        version_data.insert("data_id".to_string(), serde_json::to_value(data_id)?);
        version_data.insert("created_at".to_string(), serde_json::to_value(chrono::Utc::now())?);
        
        if let Some(parent) = parent_version {
            version_data.insert("parent_version".to_string(), serde_json::to_value(parent)?);
        }
        
        let value = serde_json::to_string(&version_data)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        self.batch.put(version_key.as_bytes(), value.as_bytes());
        
        Ok(version_id)
    }
    
    /// 获取版本信息
    pub fn get_version(&self, version_id: &str) -> Result<Option<Value>> {
        let key = format!("{}{}", VERSION_PREFIX, version_id);
        if let Some(value) = self.db.get(key.as_bytes())? {
            let version_data: Value = serde_json::from_slice(&value)
                .map_err(|e| Error::Deserialization(e.to_string()))?;
            Ok(Some(version_data))
        } else {
            Ok(None)
        }
    }
    
    /// 获取版本链
    pub fn get_version_chain(&self, version_id: &str) -> Result<Vec<Value>> {
        let mut chain = Vec::new();
        let mut current_id = version_id.to_string();
        
        loop {
            if let Some(version) = self.get_version(&current_id)? {
                chain.push(version.clone());
                
                if let Some(parent) = version.get("parent_version") {
                    if let Some(parent_id) = parent.as_str() {
                        current_id = parent_id.to_string();
                        continue;
                    }
                }
            }
            break;
        }
        
        Ok(chain)
    }
    
    /// 提交事务
    pub fn commit(self) -> Result<()> {
        self.db.write(self.batch)?;
        Ok(())
    }
    
    /// 回滚事务
    pub fn rollback(self) -> Result<()> {
        // WriteBatch自动丢弃，不需要额外操作
        Ok(())
    }
}

/// 为WriteBatch实现扩展trait
impl WriteBatchExt for WriteBatch {
    fn store_raw_data(&mut self, key: &str, value: Vec<String>) -> Result<()> {
        let key = format!("{}{}", DATA_RAW_PREFIX, key);
        let value = serde_json::to_string(&value)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        self.put(key.as_bytes(), value.as_bytes());
        Ok(())
    }
    
    fn update_raw_data(&mut self, key: &str, value: Vec<String>) -> Result<()> {
        self.store_raw_data(key, value)
    }
    
    fn commit(&mut self) -> Result<()> {
        // 实际实现由Storage.write处理
        Ok(())
    }
    
    fn rollback(&mut self) -> Result<()> {
        // WriteBatch在不提交时自动丢弃
        Ok(())
    }
} 