//! 元数据存储模块
//! 
//! 提供集合元数据的持久化存储功能

use crate::Result;
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use super::kv_storage::KeyValueStorageEngine;

/// 集合元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMetadata {
    /// 集合名称
    pub name: String,
    /// 向量维度
    pub dimension: usize,
    /// 相似度度量类型
    pub similarity_metric: String,
    /// 向量数量
    pub vector_count: usize,
    /// 索引类型
    pub index_type: Option<String>,
    /// 索引参数
    pub index_parameters: Option<serde_json::Value>,
    /// 自定义属性
    pub properties: serde_json::Value,
    /// 创建时间
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// 最后更新时间
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl CollectionMetadata {
    /// 创建新的集合元数据
    pub fn new(name: String, dimension: usize, similarity_metric: String) -> Self {
        let now = chrono::Utc::now();
        Self {
            name,
            dimension,
            similarity_metric,
            vector_count: 0,
            index_type: None,
            index_parameters: None,
            properties: serde_json::json!({}),
            created_at: now,
            updated_at: now,
        }
    }

    /// 设置索引类型
    pub fn with_index_type(mut self, index_type: String) -> Self {
        self.index_type = Some(index_type);
        self
    }

    /// 设置索引参数
    pub fn with_index_parameters(mut self, parameters: serde_json::Value) -> Self {
        self.index_parameters = Some(parameters);
        self
    }

    /// 设置自定义属性
    pub fn with_properties(mut self, properties: serde_json::Value) -> Self {
        self.properties = properties;
        self
    }

    /// 更新向量数量
    pub fn update_vector_count(&mut self, count: usize) {
        self.vector_count = count;
        self.updated_at = chrono::Utc::now();
    }
}

/// 元数据存储接口
pub struct MetadataStorage {
    /// 底层KV存储引擎
    storage: Arc<dyn KeyValueStorageEngine>,
    /// 键前缀
    key_prefix: String,
}

impl MetadataStorage {
    /// 创建新的元数据存储
    pub fn new(storage: Arc<KeyValueStorageEngine>) -> Result<Self> {
        Ok(Self {
            storage,
            key_prefix: "metadata:collection:".to_string(),
        })
    }

    /// 保存集合元数据
    pub async fn save_collection(&self, metadata: &CollectionMetadata) -> Result<()> {
        let key = self.make_key(&metadata.name);
        let value = self.serialize(metadata)?;
        // KeyValueStorageEngine 是同步接口，这里直接调用同步方法
        self.storage.set(&key, &value)
    }

    /// 获取集合元数据
    pub async fn get_collection(&self, name: &str) -> Result<Option<CollectionMetadata>> {
        let key = self.make_key(name);
        match self.storage.get(&key)? {
            Some(value) => Ok(Some(self.deserialize(&value)?)),
            None => Ok(None),
        }
    }

    /// 删除集合元数据
    pub async fn delete_collection(&self, name: &str) -> Result<()> {
        let key = self.make_key(name);
        self.storage.delete(&key)
    }

    /// 检查集合是否存在
    pub async fn collection_exists(&self, name: &str) -> Result<bool> {
        let key = self.make_key(name);
        self.storage.exists(&key)
    }

    /// 列出所有集合
    pub async fn list_collections(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        let pattern = format!("{}*", self.key_prefix);
        let keys = self.storage.get_keys_with_pattern(&pattern)?;
        for key in keys {
            if let Some(name) = key.strip_prefix(&self.key_prefix) {
                names.push(name.to_string());
            }
        }
        Ok(names)
    }

    /// 更新集合向量数量
    pub async fn update_vector_count(&self, collection_name: &str, count: usize) -> Result<()> {
        if let Some(mut metadata) = self.get_collection(collection_name).await? {
            metadata.update_vector_count(count);
            self.save_collection(&metadata).await?;
        }
        Ok(())
    }

    // 内部辅助方法

    fn make_key(&self, collection_name: &str) -> String {
        format!("{}{}", self.key_prefix, collection_name)
    }

    fn serialize(&self, metadata: &CollectionMetadata) -> Result<Vec<u8>> {
        bincode::serialize(metadata)
            .map_err(|e| crate::Error::serialization(format!("序列化失败: {}", e)))
    }

    fn deserialize(&self, data: &[u8]) -> Result<CollectionMetadata> {
        bincode::deserialize(data)
            .map_err(|e| crate::Error::serialization(format!("反序列化失败: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_metadata_creation() {
        let metadata = CollectionMetadata::new(
            "test_collection".to_string(),
            128,
            "cosine".to_string(),
        );
        
        assert_eq!(metadata.name, "test_collection");
        assert_eq!(metadata.dimension, 128);
        assert_eq!(metadata.similarity_metric, "cosine");
        assert_eq!(metadata.vector_count, 0);
    }

    #[test]
    fn test_collection_metadata_with_index() {
        let params = serde_json::json!({
            "M": 16,
            "ef_construction": 200
        });
        
        let metadata = CollectionMetadata::new(
            "test_collection".to_string(),
            128,
            "cosine".to_string(),
        )
        .with_index_type("HNSW".to_string())
        .with_index_parameters(params.clone());
        
        assert_eq!(metadata.index_type, Some("HNSW".to_string()));
        assert_eq!(metadata.index_parameters, Some(params));
    }

    #[test]
    fn test_update_vector_count() {
        let mut metadata = CollectionMetadata::new(
            "test_collection".to_string(),
            128,
            "cosine".to_string(),
        );
        
        metadata.update_vector_count(100);
        assert_eq!(metadata.vector_count, 100);
    }
}




