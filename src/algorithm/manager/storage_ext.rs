// 存储扩展模块
// 为Storage添加异步操作方法

use async_trait::async_trait;
use crate::error::{Error, Result};
use crate::storage::Storage;
use crate::algorithm::types::Algorithm;

/// 存储扩展特性，为Storage添加异步方法
#[async_trait]
pub trait StorageExt {
    /// 异步获取数据
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    
    /// 异步存储数据
    async fn put(&self, key: &str, value: &[u8]) -> Result<()>;
    
    /// 异步检查键是否存在
    async fn exists(&self, key: &str) -> Result<bool>;
    
    /// 异步删除数据
    async fn delete(&self, key: &str) -> Result<()>;
    
    /// 获取算法数据
    async fn get_algorithm(&self, algorithm_id: &str) -> Result<Option<Algorithm>>;
    
    /// 存储算法数据
    async fn put_algorithm(&self, algorithm: &Algorithm) -> Result<()>;
    
    /// 获取算法列表
    async fn list_algorithms(&self, prefix: Option<&str>, limit: Option<usize>) -> Result<Vec<Algorithm>>;
}

#[async_trait]
impl StorageExt for Storage {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        // 在实际环境中，这里可能需要使用异步运行时来包装同步调用
        // 或者实现真正的异步存储层
        tokio::task::spawn_blocking(move || {
            self.get(key)
        }).await.map_err(|e| Error::internal(format!("存储任务执行失败: {}", e)))?
    }
    
    async fn put(&self, key: &str, value: &[u8]) -> Result<()> {
        let key = key.to_string();
        let value = value.to_vec();
        tokio::task::spawn_blocking(move || {
            self.put(&key, &value)
        }).await.map_err(|e| Error::internal(format!("存储任务执行失败: {}", e)))?
    }
    
    async fn exists(&self, key: &str) -> Result<bool> {
        let key = key.to_string();
        tokio::task::spawn_blocking(move || {
            self.exists(&key)
        }).await.map_err(|e| Error::internal(format!("存储任务执行失败: {}", e)))?
    }
    
    async fn delete(&self, key: &str) -> Result<()> {
        let key = key.to_string();
        tokio::task::spawn_blocking(move || {
            self.delete(&key)
        }).await.map_err(|e| Error::internal(format!("存储任务执行失败: {}", e)))?
    }
    
    async fn get_algorithm(&self, algorithm_id: &str) -> Result<Option<Algorithm>> {
        let key = format!("algorithm:{}", algorithm_id);
        let data = self.get(&key).await?;
        
        match data {
            Some(bytes) => {
                let algorithm = serde_json::from_slice(&bytes)
                    .map_err(|e| Error::serialization(format!("无法反序列化算法数据: {}", e)))?;
                Ok(Some(algorithm))
            },
            None => Ok(None),
        }
    }
    
    async fn put_algorithm(&self, algorithm: &Algorithm) -> Result<()> {
        let key = format!("algorithm:{}", algorithm.id);
        let data = serde_json::to_vec(algorithm)
            .map_err(|e| Error::serialization(format!("无法序列化算法数据: {}", e)))?;
        
        self.put(&key, &data).await
    }
    
    async fn list_algorithms(&self, prefix: Option<&str>, limit: Option<usize>) -> Result<Vec<Algorithm>> {
        let prefix = prefix.unwrap_or("algorithm:");
        let limit = limit.unwrap_or(100);
        
        // 使用 scan_keys 方法列出所有以指定前缀开头的键
        // 该方法内部实现了键的扫描和过滤逻辑
        let prefix = prefix.to_string();
        let keys = tokio::task::spawn_blocking(move || {
            self.scan_keys(&prefix, limit)
        }).await.map_err(|e| Error::internal(format!("存储任务执行失败: {}", e)))?;
        
        let mut algorithms = Vec::new();
        for key in keys {
            if let Some(data) = self.get(&key).await? {
                let algorithm = serde_json::from_slice(&data)
                    .map_err(|e| Error::serialization(format!("无法反序列化算法数据: {}", e)))?;
                algorithms.push(algorithm);
            }
        }
        
        Ok(algorithms)
    }
} 