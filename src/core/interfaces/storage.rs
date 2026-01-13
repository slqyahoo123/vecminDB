use std::collections::HashMap;
use async_trait::async_trait;
use crate::error::Result;

/// 键值存储接口
#[async_trait]
pub trait KeyValueStore: Send + Sync {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    async fn put(&self, key: &str, value: &[u8]) -> Result<()>;
    async fn delete(&self, key: &str) -> Result<()>;
    async fn exists(&self, key: &str) -> Result<bool>;
    async fn list_keys(&self, prefix: &str) -> Result<Vec<String>>;
}

/// 事务存储接口
#[async_trait]
pub trait TransactionalStore: KeyValueStore {
    async fn begin_transaction(&self) -> Result<String>;
    async fn commit_transaction(&self, transaction_id: &str) -> Result<()>;
    async fn rollback_transaction(&self, transaction_id: &str) -> Result<()>;
}

/// 对象存储接口
#[async_trait]
pub trait ObjectStore: Send + Sync {
    async fn put_object(&self, bucket: &str, key: &str, data: &[u8]) -> Result<()>;
    async fn get_object(&self, bucket: &str, key: &str) -> Result<Option<Vec<u8>>>;
    async fn delete_object(&self, bucket: &str, key: &str) -> Result<()>;
    async fn list_objects(&self, bucket: &str, prefix: &str) -> Result<Vec<String>>;
}

/// 元数据存取（对象扩展）
#[async_trait]
pub trait ObjectMetadataStore: Send + Sync {
    async fn put_object_metadata(&self, bucket: &str, key: &str, metadata: &HashMap<String, String>) -> Result<()>;
    async fn get_object_metadata(&self, bucket: &str, key: &str) -> Result<Option<HashMap<String, String>>>;
}


