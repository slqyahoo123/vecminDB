// 算法操作模块
// 实现AlgorithmOperations trait和算法相关方法

use std::sync::Arc;
use log::{info, warn};
use serde_json;
use rocksdb;

use crate::{Error, Result};
use super::core::Storage;
use super::algorithm::AlgorithmOperations;

impl AlgorithmOperations for Storage {
    fn store_algorithm(&self, algorithm_id: &str, code: &str) -> Result<()> {
        let key = format!("algorithm:{}", algorithm_id);
        
        // 使用当前时间戳作为版本标识
        let timestamp = chrono::Utc::now().timestamp();
        let version_id = timestamp.to_string();
        let version_key = format!("algorithm:{}:{}", algorithm_id, version_id);
        
        // 开始事务
        let mut batch = self.create_write_batch();
        
        // 存储当前算法代码
        batch.put(key.as_bytes(), code.as_bytes());
        
        // 存储版本化的算法代码
        batch.put(version_key.as_bytes(), code.as_bytes());
        
        // 更新算法版本列表
        let versions_key = format!("algorithm:versions:{}", algorithm_id);
        let mut versions = match self.get(versions_key.as_bytes())? {
            Some(data) => {
                let versions_str = String::from_utf8(data)?;
                let versions: Vec<String> = serde_json::from_str(&versions_str)?;
                versions
            },
            None => Vec::new(),
        };
        
        versions.push(version_id);
        let versions_json = serde_json::to_string(&versions)?;
        batch.put(versions_key.as_bytes(), versions_json.as_bytes());
        
        // 提交事务
        self.write(batch)?;
        
        info!("Stored algorithm with ID: {}", algorithm_id);
        Ok(())
    }
    
    fn get_algorithm(&self, algorithm_id: &str) -> Result<Option<String>> {
        let key = format!("algorithm:{}", algorithm_id);
        
        match self.get(key.as_bytes())? {
            Some(data) => {
                let code = String::from_utf8(data)?;
                Ok(Some(code))
            },
            None => {
                info!("Algorithm with ID {} not found", algorithm_id);
                Ok(None)
            }
        }
    }
    
    fn store_algorithm_metadata(&self, algorithm_id: &str, metadata: &serde_json::Value) -> Result<()> {
        let key = format!("algorithm:metadata:{}", algorithm_id);
        let value = metadata.to_string();
        
        self.put(key.as_bytes(), value.as_bytes())?;
        
        // 更新算法版本列表中的元数据
        let versions_key = format!("algorithm:versions:{}", algorithm_id);
        let versions = match self.get(versions_key.as_bytes())? {
            Some(data) => {
                let versions_str = String::from_utf8(data)?;
                let versions: Vec<String> = serde_json::from_str(&versions_str)?;
                versions
            },
            None => Vec::new(),
        };
        
        // 如果存在版本记录，为最新版本添加元数据
        if !versions.is_empty() {
            let latest_version = versions.last()
                .ok_or_else(|| Error::StorageError(format!("算法 {} 没有可用版本", algorithm_id)))?;
            let metadata_version_key = format!("algorithm:metadata:{}:{}", algorithm_id, latest_version);
            self.put(metadata_version_key.as_bytes(), value.as_bytes())?;
        }
        
        info!("Stored metadata for algorithm: {}", algorithm_id);
        Ok(())
    }
    
    fn get_algorithm_metadata(&self, algorithm_id: &str) -> Result<Option<serde_json::Value>> {
        let key = format!("algorithm:metadata:{}", algorithm_id);
        
        match self.get(key.as_bytes())? {
            Some(data) => {
                let metadata_str = String::from_utf8(data)?;
                let metadata: serde_json::Value = serde_json::from_str(&metadata_str)?;
                Ok(Some(metadata))
            },
            None => {
                info!("No metadata found for algorithm {}", algorithm_id);
                Ok(None)
            }
        }
    }
    
    fn get_algorithm_version(&self, algorithm_id: &str, version_id: &str) -> Result<Option<String>> {
        let key = format!("algorithm:{}:{}", algorithm_id, version_id);
        
        // 先检查该版本是否存在
        let versions_key = format!("algorithm:versions:{}", algorithm_id);
        let versions = match self.get(versions_key.as_bytes())? {
            Some(data) => {
                let versions_str = String::from_utf8(data)?;
                let versions: Vec<String> = serde_json::from_str(&versions_str)?;
                versions
            },
            None => Vec::new(),
        };
        
        // 如果版本不存在，提早返回
        if !versions.contains(&version_id.to_string()) {
            info!("Version {} not found for algorithm {}", version_id, algorithm_id);
            return Ok(None);
        }
        
        match self.get(key.as_bytes())? {
            Some(data) => {
                let code = String::from_utf8(data)?;
                Ok(Some(code))
            },
            None => {
                warn!("Algorithm version data not found despite version being listed");
                Ok(None)
            }
        }
    }
    
    fn delete_algorithm(&self, algorithm_id: &str) -> Result<()> {
        let key = format!("algorithm:{}", algorithm_id);
        let metadata_key = format!("algorithm:metadata:{}", algorithm_id);
        let versions_key = format!("algorithm:versions:{}", algorithm_id);
        
        // 删除算法代码
        self.delete(key.as_bytes())?;
        
        // 删除元数据
        self.delete(metadata_key.as_bytes())?;
        
        // 删除版本列表
        self.delete(versions_key.as_bytes())?;
        
        info!("Deleted algorithm with ID: {}", algorithm_id);
        Ok(())
    }
}

// 算法相关的异步方法实现
impl Storage {
    /// 列出过滤后的算法（返回算法ID列表）
    pub async fn list_algorithms_filtered(self: &Arc<Self>, category: Option<&str>, limit: Option<usize>) -> Result<Vec<String>> {
        let storage = Arc::clone(self);
        let category = category.map(|s| s.to_string());
        let limit = limit.unwrap_or(100);
        tokio::task::spawn_blocking(move || {
            let prefix = "algorithm:";
            let mut algorithms = Vec::new();
            let iter = storage.db().iterator(rocksdb::IteratorMode::From(prefix.as_bytes(), rocksdb::Direction::Forward));
            
            for item in iter {
                match item {
                    Ok((key, _)) => {
                        if let Ok(key_str) = String::from_utf8(key.to_vec()) {
                            if key_str.starts_with(prefix) && !key_str.contains(":metadata:") && !key_str.contains(":versions:") {
                                // 提取算法ID
                                if let Some(algorithm_id) = key_str.strip_prefix(prefix) {
                                    // 如果指定了类别，检查元数据
                                    if let Some(ref cat) = category {
                                        let metadata_key = format!("algorithm:metadata:{}", algorithm_id);
                                        if let Ok(Some(metadata_data)) = storage.get(metadata_key.as_bytes()) {
                                            if let Ok(metadata_str) = String::from_utf8(metadata_data) {
                                                if let Ok(metadata) = serde_json::from_str::<serde_json::Value>(&metadata_str) {
                                                    if let Some(meta_cat) = metadata.get("category").and_then(|v| v.as_str()) {
                                                        if meta_cat == cat {
                                                            algorithms.push(algorithm_id.to_string());
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        algorithms.push(algorithm_id.to_string());
                                    }
                                    
                                    if algorithms.len() >= limit {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        warn!("算法列表扫描错误: {}", e);
                    }
                }
            }
            
            Ok(algorithms)
        }).await.map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }
    
    /// 异步存储算法（包装同步方法）
    pub async fn store_algorithm_async(self: &Arc<Self>, algorithm_id: &str, code: &str) -> Result<()> {
        let storage = Arc::clone(self);
        let algorithm_id = algorithm_id.to_string();
        let code = code.to_string();
        tokio::task::spawn_blocking(move || {
            storage.store_algorithm(&algorithm_id, &code)
        }).await.map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }
    
    /// 异步加载算法（包装同步方法）
    pub async fn load_algorithm(self: &Arc<Self>, algorithm_id: &str) -> Result<Option<String>> {
        let storage = Arc::clone(self);
        let algorithm_id = algorithm_id.to_string();
        tokio::task::spawn_blocking(move || {
            storage.get_algorithm(&algorithm_id)
        }).await.map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }
    
    /// 异步删除算法（包装同步方法）
    pub async fn delete_algorithm_async(self: &Arc<Self>, algorithm_id: &str) -> Result<()> {
        let storage = Arc::clone(self);
        let algorithm_id = algorithm_id.to_string();
        tokio::task::spawn_blocking(move || {
            storage.delete_algorithm(&algorithm_id)
        }).await.map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }
}

