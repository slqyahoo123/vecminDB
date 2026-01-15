//! 向量存储模块
//! 
//! 提供向量数据的持久化存储功能

use crate::Result;
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use super::kv_storage::KeyValueStorageEngine;

/// 向量记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    /// 向量ID
    pub id: String,
    /// 向量数据
    pub vector: Vec<f32>,
    /// 元数据（可选）
    pub metadata: Option<serde_json::Value>,
    /// 创建时间
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// 更新时间
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl VectorRecord {
    /// 创建新的向量记录
    pub fn new(id: String, vector: Vec<f32>) -> Self {
        let now = chrono::Utc::now();
        Self {
            id,
            vector,
            metadata: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// 设置元数据
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// 获取向量维度
    pub fn dimension(&self) -> usize {
        self.vector.len()
    }
}

/// 向量存储配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStorageConfig {
    /// 集合名称
    pub collection_name: String,
    /// 向量维度
    pub dimension: usize,
    /// 是否启用压缩
    pub enable_compression: bool,
    /// 批量写入大小
    pub batch_size: usize,
}

impl Default for VectorStorageConfig {
    fn default() -> Self {
        Self {
            collection_name: "default".to_string(),
            dimension: 128,
            enable_compression: false,
            batch_size: 1000,
        }
    }
}

/// 向量存储接口
pub struct VectorStorage {
    /// 底层KV存储引擎
    storage: Arc<dyn KeyValueStorageEngine>,
    /// 配置
    config: VectorStorageConfig,
    /// 键前缀
    key_prefix: String,
}

impl VectorStorage {
    /// 创建新的向量存储
    pub fn new(storage: Arc<KeyValueStorageEngine>, config: VectorStorageConfig) -> Result<Self> {
        let key_prefix = format!("vector:{}:", config.collection_name);
        Ok(Self {
            storage,
            config,
            key_prefix,
        })
    }

    /// 插入向量
    pub async fn insert(&self, record: &VectorRecord) -> Result<()> {
        let key = self.make_key(&record.id);
        let value = self.serialize_record(record)?;
        // KeyValueStorageEngine 是同步接口，这里直接调用同步方法
        self.storage.set(&key, &value)
    }

    /// 批量插入向量
    pub async fn batch_insert(&self, records: &[VectorRecord]) -> Result<()> {
        for chunk in records.chunks(self.config.batch_size) {
            for record in chunk {
                self.insert(record).await?;
            }
        }
        Ok(())
    }

    /// 获取向量
    pub async fn get(&self, id: &str) -> Result<Option<VectorRecord>> {
        let key = self.make_key(id);
        match self.storage.get(&key)? {
            Some(value) => Ok(Some(self.deserialize_record(&value)?)),
            None => Ok(None),
        }
    }

    /// 删除向量
    pub async fn delete(&self, id: &str) -> Result<()> {
        let key = self.make_key(id);
        self.storage.delete(&key)
    }

    /// 检查向量是否存在
    pub async fn exists(&self, id: &str) -> Result<bool> {
        let key = self.make_key(id);
        self.storage.exists(&key)
    }

    /// 列出所有向量ID
    pub async fn list_ids(&self) -> Result<Vec<String>> {
        let mut ids = Vec::new();
        // 使用模式匹配获取所有以 key_prefix 开头的键
        let pattern = format!("{}*", self.key_prefix);
        let keys = self.storage.get_keys_with_pattern(&pattern)?;
        for key in keys {
            if let Some(id) = key.strip_prefix(&self.key_prefix) {
                ids.push(id.to_string());
            }
        }
        Ok(ids)
    }

    /// 统计向量数量
    pub async fn count(&self) -> Result<usize> {
        self.list_ids().await.map(|ids| ids.len())
    }

    /// 清空集合
    pub async fn clear(&self) -> Result<()> {
        let ids = self.list_ids().await?;
        for id in ids {
            self.delete(&id).await?;
        }
        Ok(())
    }

    // 内部辅助方法

    fn make_key(&self, id: &str) -> String {
        format!("{}{}", self.key_prefix, id)
    }

    fn serialize_record(&self, record: &VectorRecord) -> Result<Vec<u8>> {
        if self.config.enable_compression {
            // 使用压缩
            let serialized = bincode::serialize(record)
                .map_err(|e| crate::Error::serialization(format!("序列化失败: {}", e)))?;
            super::compression::compress(&serialized)
        } else {
            bincode::serialize(record)
                .map_err(|e| crate::Error::serialization(format!("序列化失败: {}", e)))
        }
    }

    fn deserialize_record(&self, data: &[u8]) -> Result<VectorRecord> {
        if self.config.enable_compression {
            // 解压缩
            let decompressed = super::compression::decompress(data)?;
            bincode::deserialize(&decompressed)
                .map_err(|e| crate::Error::serialization(format!("反序列化失败: {}", e)))
        } else {
            bincode::deserialize(data)
                .map_err(|e| crate::Error::serialization(format!("反序列化失败: {}", e)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_record_creation() {
        let vector = vec![0.1, 0.2, 0.3, 0.4];
        let record = VectorRecord::new("test_id".to_string(), vector.clone());
        
        assert_eq!(record.id, "test_id");
        assert_eq!(record.vector, vector);
        assert_eq!(record.dimension(), 4);
    }

    #[test]
    fn test_vector_record_with_metadata() {
        let vector = vec![0.1, 0.2, 0.3];
        let metadata = serde_json::json!({"label": "test"});
        let record = VectorRecord::new("test_id".to_string(), vector)
            .with_metadata(metadata.clone());
        
        assert_eq!(record.metadata, Some(metadata));
    }
}




