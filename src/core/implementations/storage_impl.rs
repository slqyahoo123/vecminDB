/// 存储层接口的完整生产级实现
/// 提供高性能、可靠的数据存储服务

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::path::Path;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use tokio::sync::Mutex;
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::{Result, Error};
use crate::core::interfaces::storage::*;
use crate::storage::kv_storage::KeyValueStorage;

/// 生产级键值存储实现
pub struct ProductionKeyValueStore {
    storage: Arc<KeyValueStorage>,
    metrics: Arc<RwLock<StorageMetrics>>,
}

impl ProductionKeyValueStore {
    pub fn new(storage_path: &str) -> Result<Self> {
        let storage = Arc::new(KeyValueStorage::new(storage_path)?);
        let metrics = Arc::new(RwLock::new(StorageMetrics::new()));
        
        Ok(Self {
            storage,
            metrics,
        })
    }

    fn update_metrics<F>(&self, operation: &str, result: &Result<()>, measure_fn: F) 
    where 
        F: FnOnce() -> u64 
    {
        let start_time = std::time::Instant::now();
        let duration = measure_fn();
        
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.total_operations += 1;
            metrics.total_duration_ms += duration;
            
            match result {
                Ok(_) => metrics.successful_operations += 1,
                Err(_) => metrics.failed_operations += 1,
            }
            
            metrics.operations_by_type
                .entry(operation.to_string())
                .and_modify(|e| *e += 1)
                .or_insert(1);
        }
    }
}

#[async_trait]
impl KeyValueStore for ProductionKeyValueStore {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let start = std::time::Instant::now();
        let result = self.storage.get(key).await;
        let duration = start.elapsed().as_millis() as u64;
        
        self.update_metrics("get", &result.as_ref().map(|_| ()), || duration);
        result
    }

    async fn put(&self, key: &str, value: &[u8]) -> Result<()> {
        let start = std::time::Instant::now();
        let result = self.storage.put(key, value).await;
        let duration = start.elapsed().as_millis() as u64;
        
        self.update_metrics("put", &result, || duration);
        result
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let start = std::time::Instant::now();
        let result = self.storage.delete(key).await;
        let duration = start.elapsed().as_millis() as u64;
        
        self.update_metrics("delete", &result, || duration);
        result
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let start = std::time::Instant::now();
        let result = self.storage.exists(key).await;
        let duration = start.elapsed().as_millis() as u64;
        
        self.update_metrics("exists", &result.as_ref().map(|_| ()), || duration);
        result
    }

    async fn list_keys(&self, prefix: &str) -> Result<Vec<String>> {
        let start = std::time::Instant::now();
        let result = self.storage.list_keys(prefix).await;
        let duration = start.elapsed().as_millis() as u64;
        
        self.update_metrics("list_keys", &result.as_ref().map(|_| ()), || duration);
        result
    }
}

/// 生产级事务存储实现
pub struct ProductionTransactionalStore {
    kv_store: Arc<ProductionKeyValueStore>,
    transactions: Arc<Mutex<HashMap<String, Transaction>>>,
}

impl ProductionTransactionalStore {
    pub fn new(storage_path: &str) -> Result<Self> {
        let kv_store = Arc::new(ProductionKeyValueStore::new(storage_path)?);
        let transactions = Arc::new(Mutex::new(HashMap::new()));
        
        Ok(Self {
            kv_store,
            transactions,
        })
    }
}

#[async_trait]
impl KeyValueStore for ProductionTransactionalStore {
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        self.kv_store.get(key).await
    }

    async fn put(&self, key: &str, value: &[u8]) -> Result<()> {
        self.kv_store.put(key, value).await
    }

    async fn delete(&self, key: &str) -> Result<()> {
        self.kv_store.delete(key).await
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        self.kv_store.exists(key).await
    }

    async fn list_keys(&self, prefix: &str) -> Result<Vec<String>> {
        self.kv_store.list_keys(prefix).await
    }
}

#[async_trait]
impl TransactionalStore for ProductionTransactionalStore {
    async fn begin_transaction(&self) -> Result<String> {
        let transaction_id = Uuid::new_v4().to_string();
        let transaction = Transaction {
            id: transaction_id.clone(),
            operations: Vec::new(),
            status: TransactionStatus::Active,
            created_at: Utc::now(),
        };
        
        let mut transactions = self.transactions.lock().await;
        transactions.insert(transaction_id.clone(), transaction);
        
        Ok(transaction_id)
    }

    async fn commit_transaction(&self, transaction_id: &str) -> Result<()> {
        let mut transactions = self.transactions.lock().await;
        
        if let Some(mut transaction) = transactions.remove(transaction_id) {
            transaction.status = TransactionStatus::Committed;
            
            // 执行所有操作
            for operation in &transaction.operations {
                match operation {
                    TransactionOperation::Put { key, value } => {
                        self.kv_store.put(key, value).await?;
                    },
                    TransactionOperation::Delete { key } => {
                        self.kv_store.delete(key).await?;
                    },
                }
            }
            
            Ok(())
        } else {
            Err(Error::InvalidInput(format!("事务未找到: {}", transaction_id)))
        }
    }

    async fn rollback_transaction(&self, transaction_id: &str) -> Result<()> {
        let mut transactions = self.transactions.lock().await;
        
        if let Some(mut transaction) = transactions.remove(transaction_id) {
            transaction.status = TransactionStatus::RolledBack;
            Ok(())
        } else {
            Err(Error::InvalidInput(format!("事务未找到: {}", transaction_id)))
        }
    }
}

/// 生产级对象存储实现
pub struct ProductionObjectStore {
    base_path: String,
    buckets: Arc<RwLock<HashMap<String, BucketInfo>>>,
}

impl ProductionObjectStore {
    pub fn new(base_path: &str) -> Result<Self> {
        std::fs::create_dir_all(base_path)?;
        
        Ok(Self {
            base_path: base_path.to_string(),
            buckets: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    fn get_object_path(&self, bucket: &str, key: &str) -> String {
        format!("{}/{}/{}", self.base_path, bucket, key)
    }

    fn ensure_bucket_exists(&self, bucket: &str) -> Result<()> {
        let bucket_path = format!("{}/{}", self.base_path, bucket);
        std::fs::create_dir_all(&bucket_path)?;
        
        let mut buckets = self.buckets.write().unwrap();
        buckets.entry(bucket.to_string()).or_insert_with(|| BucketInfo {
            name: bucket.to_string(),
            created_at: Utc::now(),
            object_count: 0,
            total_size: 0,
        });
        
        Ok(())
    }
}

#[async_trait]
impl ObjectStore for ProductionObjectStore {
    async fn put_object(&self, bucket: &str, key: &str, data: &[u8]) -> Result<()> {
        self.ensure_bucket_exists(bucket)?;
        
        let object_path = self.get_object_path(bucket, key);
        if let Some(parent) = Path::new(&object_path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        tokio::fs::write(&object_path, data).await?;
        
        // 更新桶信息
        if let Ok(mut buckets) = self.buckets.write() {
            if let Some(bucket_info) = buckets.get_mut(bucket) {
                bucket_info.object_count += 1;
                bucket_info.total_size += data.len() as u64;
            }
        }
        
        Ok(())
    }

    async fn get_object(&self, bucket: &str, key: &str) -> Result<Option<Vec<u8>>> {
        let object_path = self.get_object_path(bucket, key);
        
        match tokio::fs::read(&object_path).await {
            Ok(data) => Ok(Some(data)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(Error::Io(e)),
        }
    }

    async fn delete_object(&self, bucket: &str, key: &str) -> Result<()> {
        let object_path = self.get_object_path(bucket, key);
        
        if tokio::fs::metadata(&object_path).await.is_ok() {
            tokio::fs::remove_file(&object_path).await?;
            
            // 更新桶信息
            if let Ok(mut buckets) = self.buckets.write() {
                if let Some(bucket_info) = buckets.get_mut(bucket) {
                    bucket_info.object_count = bucket_info.object_count.saturating_sub(1);
                }
            }
        }
        
        Ok(())
    }

    async fn list_objects(&self, bucket: &str, prefix: &str) -> Result<Vec<String>> {
        let bucket_path = format!("{}/{}", self.base_path, bucket);
        let prefix_path = format!("{}/{}", bucket_path, prefix);
        
        let mut objects = Vec::new();
        
        if let Ok(mut entries) = tokio::fs::read_dir(&bucket_path).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                if let Ok(file_type) = entry.file_type().await {
                    if file_type.is_file() {
                        if let Some(file_name) = entry.file_name().to_str() {
                            if file_name.starts_with(prefix) {
                                objects.push(file_name.to_string());
                            }
                        }
                    }
                }
            }
        }
        
        Ok(objects)
    }
}

/// 存储指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageMetrics {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub total_duration_ms: u64,
    pub operations_by_type: HashMap<String, u64>,
}

impl StorageMetrics {
    pub fn new() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            total_duration_ms: 0,
            operations_by_type: HashMap::new(),
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_operations > 0 {
            self.successful_operations as f64 / self.total_operations as f64
        } else {
            0.0
        }
    }

    pub fn average_duration_ms(&self) -> f64 {
        if self.total_operations > 0 {
            self.total_duration_ms as f64 / self.total_operations as f64
        } else {
            0.0
        }
    }
}

/// 事务
#[derive(Debug, Clone)]
struct Transaction {
    id: String,
    operations: Vec<TransactionOperation>,
    status: TransactionStatus,
    created_at: DateTime<Utc>,
}

/// 事务操作
#[derive(Debug, Clone)]
enum TransactionOperation {
    Put { key: String, value: Vec<u8> },
    Delete { key: String },
}

/// 事务状态
#[derive(Debug, Clone, PartialEq)]
enum TransactionStatus {
    Active,
    Committed,
    RolledBack,
}

/// 桶信息
#[derive(Debug, Clone)]
struct BucketInfo {
    name: String,
    created_at: DateTime<Utc>,
    object_count: u64,
    total_size: u64,
} 