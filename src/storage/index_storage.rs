//! 索引存储模块
//! 
//! 提供向量索引的持久化存储功能

use crate::Result;
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use super::kv_storage::KeyValueStorageEngine;

/// 索引元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// 索引名称
    pub name: String,
    /// 索引类型（HNSW, IVF, PQ等）
    pub index_type: String,
    /// 向量维度
    pub dimension: usize,
    /// 索引参数
    pub parameters: serde_json::Value,
    /// 向量数量
    pub vector_count: usize,
    /// 创建时间
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// 最后更新时间
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl IndexMetadata {
    /// 创建新的索引元数据
    pub fn new(name: String, index_type: String, dimension: usize) -> Self {
        let now = chrono::Utc::now();
        Self {
            name,
            index_type,
            dimension,
            parameters: serde_json::json!({}),
            vector_count: 0,
            created_at: now,
            updated_at: now,
        }
    }

    /// 设置参数
    pub fn with_parameters(mut self, parameters: serde_json::Value) -> Self {
        self.parameters = parameters;
        self
    }

    /// 更新向量数量
    pub fn update_count(&mut self, count: usize) {
        self.vector_count = count;
        self.updated_at = chrono::Utc::now();
    }
}

/// 索引存储配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStorageConfig {
    /// 集合名称
    pub collection_name: String,
    /// 是否启用压缩
    pub enable_compression: bool,
}

impl Default for IndexStorageConfig {
    fn default() -> Self {
        Self {
            collection_name: "default".to_string(),
            enable_compression: true,
        }
    }
}

/// 索引存储接口
pub struct IndexStorage {
    /// 底层KV存储引擎
    storage: Arc<dyn KeyValueStorageEngine>,
    /// 配置
    config: IndexStorageConfig,
    /// 键前缀
    key_prefix: String,
}

impl IndexStorage {
    /// 创建新的索引存储
    pub fn new(storage: Arc<dyn KeyValueStorageEngine>, config: IndexStorageConfig) -> Result<Self> {
        let key_prefix = format!("index:{}:", config.collection_name);
        Ok(Self {
            storage,
            config,
            key_prefix,
        })
    }

    /// 保存索引元数据
    pub async fn save_metadata(&self, metadata: &IndexMetadata) -> Result<()> {
        let key = self.make_metadata_key(&metadata.name);
        let value = self.serialize(metadata)?;
        // KeyValueStorageEngine 是同步接口，这里直接调用同步方法
        self.storage.set(&key, &value)
    }

    /// 获取索引元数据
    pub async fn get_metadata(&self, name: &str) -> Result<Option<IndexMetadata>> {
        let key = self.make_metadata_key(name);
        match self.storage.get(&key)? {
            Some(value) => Ok(Some(self.deserialize(&value)?)),
            None => Ok(None),
        }
    }

    /// 保存索引数据
    pub async fn save_index_data(&self, index_name: &str, data: &[u8]) -> Result<()> {
        let key = self.make_data_key(index_name);
        let value = if self.config.enable_compression {
            super::compression::compress(data)?
        } else {
            data.to_vec()
        };
        self.storage.set(&key, &value)
    }

    /// 获取索引数据
    pub async fn get_index_data(&self, index_name: &str) -> Result<Option<Vec<u8>>> {
        let key = self.make_data_key(index_name);
        match self.storage.get(&key)? {
            Some(value) => {
                let data = if self.config.enable_compression {
                    super::compression::decompress(&value)?
                } else {
                    value
                };
                Ok(Some(data))
            },
            None => Ok(None),
        }
    }

    /// 删除索引
    pub async fn delete_index(&self, index_name: &str) -> Result<()> {
        // 删除元数据
        let metadata_key = self.make_metadata_key(index_name);
        self.storage.delete(&metadata_key)?;
        
        // 删除索引数据
        let data_key = self.make_data_key(index_name);
        self.storage.delete(&data_key)?;
        
        Ok(())
    }

    /// 列出所有索引
    pub async fn list_indexes(&self) -> Result<Vec<String>> {
        let prefix = format!("{}metadata:", self.key_prefix);
        let mut names = Vec::new();
        
        let pattern = format!("{}*", prefix);
        let keys = self.storage.get_keys_with_pattern(&pattern)?;
        for key in keys {
            if let Some(name) = key.strip_prefix(&prefix) {
                names.push(name.to_string());
            }
        }
        
        Ok(names)
    }

    // 内部辅助方法

    fn make_metadata_key(&self, index_name: &str) -> String {
        format!("{}metadata:{}", self.key_prefix, index_name)
    }

    fn make_data_key(&self, index_name: &str) -> String {
        format!("{}data:{}", self.key_prefix, index_name)
    }

    fn serialize<T: Serialize>(&self, data: &T) -> Result<Vec<u8>> {
        bincode::serialize(data)
            .map_err(|e| crate::Error::serialization(format!("序列化失败: {}", e)))
    }

    fn deserialize<T: for<'de> Deserialize<'de>>(&self, data: &[u8]) -> Result<T> {
        bincode::deserialize(data)
            .map_err(|e| crate::Error::serialization(format!("反序列化失败: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_metadata_creation() {
        let metadata = IndexMetadata::new(
            "test_index".to_string(),
            "HNSW".to_string(),
            128,
        );
        
        assert_eq!(metadata.name, "test_index");
        assert_eq!(metadata.index_type, "HNSW");
        assert_eq!(metadata.dimension, 128);
        assert_eq!(metadata.vector_count, 0);
    }

    #[test]
    fn test_index_metadata_with_parameters() {
        let params = serde_json::json!({
            "M": 16,
            "ef_construction": 200
        });
        
        let metadata = IndexMetadata::new(
            "test_index".to_string(),
            "HNSW".to_string(),
            128,
        ).with_parameters(params.clone());
        
        assert_eq!(metadata.parameters, params);
    }
}




