use crate::Result;
use crate::Error;
use crate::storage::config::StorageConfig;
use crate::storage::engine::StorageService;
use crate::compat::{ModelArchitecture, ModelParameters};
// remove unused Enhanced* imports
use crate::core::{InferenceResultDetail, InferenceResult};
// 注意：向量数据库系统不需要训练相关功能，已移除所有训练相关导入
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};
use rocksdb::{DB, Options};
use log::{warn, error};
use crate::storage::engine::interfaces::StorageEngine;
use crate::storage::models::{ModelInfo, ModelMetrics};
// remove unused alias import for Model
use crate::storage::engine::implementation::StorageOptions;
use crate::interfaces::storage::StorageTransaction;

/// RocksDB存储引擎实现
pub struct RocksDBStorage {
    config: StorageConfig,
    db: Arc<RwLock<DB>>,
}

impl RocksDBStorage {
    /// 创建新的RocksDB存储引擎
    pub fn new(config: StorageConfig) -> Result<Self> {
        // 创建目录（如果不存在）
        if !Path::new(&config.path).exists() && config.create_if_missing {
            std::fs::create_dir_all(&config.path)?;
        }
        
        // 配置RocksDB
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_max_background_jobs(4);
        
        // 打开数据库
        let db = DB::open(&opts, &config.path)?;
        
        Ok(Self {
            config,
            db: Arc::new(RwLock::new(db)),
        })
    }
}

impl StorageService for RocksDBStorage {
    fn save_model_parameters(&self, model_id: &str, params: &ModelParameters) -> Result<()> {
        let key = format!("model:{}:params", model_id);
        let value = bincode::serialize(params)?;
        self.db.write().map_err(|e| Error::lock(e.to_string()))?.put(key.as_bytes(), value.as_slice())?;
        Ok(())
    }
    
    fn get_model_parameters(&self, model_id: &str) -> Result<Option<ModelParameters>> {
        let key = format!("model:{}:params", model_id);
        if let Some(value) = self.db.read().map_err(|e| Error::lock(e.to_string()))?.get(key.as_bytes())? {
            let params = bincode::deserialize(&value)?;
            Ok(Some(params))
        } else {
            Ok(None)
        }
    }
    
    fn save_model_architecture(&self, model_id: &str, arch: &ModelArchitecture) -> Result<()> {
        let key = format!("model:{}:arch", model_id);
        let value = bincode::serialize(arch)?;
        self.db.write().map_err(|e| Error::lock(e.to_string()))?.put(key.as_bytes(), value.as_slice())?;
        Ok(())
    }
    
    fn get_model_architecture(&self, model_id: &str) -> Result<Option<ModelArchitecture>> {
        let key = format!("model:{}:arch", model_id);
        if let Some(value) = self.db.read().map_err(|e| Error::lock(e.to_string()))?.get(key.as_bytes())? {
            let arch = bincode::deserialize(&value)?;
            Ok(Some(arch))
        } else {
            Ok(None)
        }
    }
    
    // 注意：向量数据库系统不需要训练相关功能，已移除所有训练相关方法
    
    fn list_inference_results(&self, model_id: &str) -> Result<Vec<InferenceResult>> {
        let prefix = format!("inference:results:{}:", model_id);
        let mut results = Vec::new();
        
        let db = self.db.read().map_err(|e| Error::lock(e.to_string()))?;
        let iter = db.prefix_iterator(prefix.as_bytes());
        
        for item in iter {
            if let Ok((_, value)) = item {
                if let Ok(result) = bincode::deserialize::<InferenceResult>(&value) {
                    results.push(result);
                }
            }
        }
        
        Ok(results)
    }
    
    fn get_inference_result(&self, model_id: &str) -> Result<Option<crate::core::InferenceResult>> {
        let key = format!("inference:results:{}", model_id);
        if let Some(value) = self.db.read().map_err(|e| Error::lock(e.to_string()))?.get(key.as_bytes())? {
            let result = bincode::deserialize(&value)?;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }
    
    fn save_inference_result(&self, model_id: &str, result: &crate::core::InferenceResult) -> Result<()> {
        let key = format!("inference:result:{}", model_id);
        let value = bincode::serialize(result)?;
        self.db.write().map_err(|e| Error::lock(e.to_string()))?.put(key.as_bytes(), value.as_slice())?;
        Ok(())
    }
    
    async fn store(&self, key: &str, value: &[u8]) -> Result<()> {
        let db = self.db.clone();
        let key = key.to_string();
        let value = value.to_vec();
        tokio::task::spawn_blocking(move || {
            db.write().map_err(|e| Error::lock(e.to_string()))?.put(key.as_bytes(), &value)?;
            Ok::<(), Error>(())
        }).await.map_err(|e| Error::internal(format!("存储操作失败: {}", e)))??;
        Ok(())
    }
    
    async fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let db = self.db.clone();
        let key = key.to_string();
        tokio::task::spawn_blocking(move || {
            let db_guard = db.read().map_err(|e| Error::lock(e.to_string()))?;
            Ok(db_guard.get(key.as_bytes())?)
        }).await.map_err(|e| Error::internal(format!("检索操作失败: {}", e)))?
    }
    
    async fn delete(&self, key: &str) -> Result<()> {
        let db = self.db.clone();
        let key = key.to_string();
        tokio::task::spawn_blocking(move || {
            db.write().map_err(|e| Error::lock(e.to_string()))?.delete(key.as_bytes())?;
            Ok::<(), Error>(())
        }).await.map_err(|e| Error::internal(format!("删除操作失败: {}", e)))??;
        Ok(())
    }
    
    async fn exists(&self, key: &str) -> Result<bool> {
        let db = self.db.clone();
        let key = key.to_string();
        tokio::task::spawn_blocking(move || {
            Ok(db.read().map_err(|e| Error::lock(e.to_string()))?.get(key.as_bytes())?.is_some())
        }).await.map_err(|e| Error::internal(format!("检查存在性失败: {}", e)))?
    }
    
    async fn list_keys(&self, prefix: &str) -> Result<Vec<String>> {
        let db = self.db.clone();
        let prefix = prefix.to_string();
        tokio::task::spawn_blocking(move || {
            let db_guard = db.read().map_err(|e| Error::lock(e.to_string()))?;
            let iter = db_guard.prefix_iterator(prefix.as_bytes());
            let mut keys = Vec::new();
            for item in iter {
                if let Ok((key, _)) = item {
                    if let Ok(key_str) = String::from_utf8(key.to_vec()) {
                        keys.push(key_str);
                    }
                }
            }
            Ok(keys)
        }).await.map_err(|e| Error::internal(format!("列出键失败: {}", e)))?
    }
    
    async fn batch_store(&self, items: &[(String, Vec<u8>)]) -> Result<()> {
        let db = self.db.clone();
        let items = items.to_vec();
        tokio::task::spawn_blocking(move || {
            let db_guard = db.write().map_err(|e| Error::lock(e.to_string()))?;
            for (key, value) in items {
                db_guard.put(key.as_bytes(), value.as_slice())?;
            }
            Ok::<(), Error>(())
        }).await.map_err(|e| Error::internal(format!("批量存储失败: {}", e)))??;
        Ok(())
    }
    
    async fn batch_retrieve(&self, keys: &[String]) -> Result<HashMap<String, Option<Vec<u8>>>> {
        let db = self.db.clone();
        let keys = keys.to_vec();
        tokio::task::spawn_blocking(move || {
            let db_guard = db.read().map_err(|e| Error::lock(e.to_string()))?;
            let mut results = HashMap::new();
            for key in keys {
                let value = db_guard.get(key.as_bytes())?;
                results.insert(key, value);
            }
            Ok(results)
        }).await.map_err(|e| Error::internal(format!("批量检索失败: {}", e)))?
    }
    
    async fn batch_delete(&self, keys: &[String]) -> Result<()> {
        let db = self.db.clone();
        let keys = keys.to_vec();
        tokio::task::spawn_blocking(move || {
            let db_guard = db.write().map_err(|e| Error::lock(e.to_string()))?;
            for key in keys {
                db_guard.delete(key.as_bytes())?;
            }
            Ok::<(), Error>(())
        }).await.map_err(|e| Error::internal(format!("批量删除失败: {}", e)))??;
        Ok(())
    }
    
    async fn transaction(&self) -> Result<Box<dyn StorageTransaction + Send + Sync>> {
        Ok(Box::new(RocksDBTransaction::new(self.db.clone())))
    }
    
    async fn transaction_with_isolation(&self, isolation_level: crate::interfaces::storage::IsolationLevel) -> Result<Box<dyn StorageTransaction + Send + Sync>> {
        Ok(Box::new(RocksDBTransaction::new_with_isolation(self.db.clone(), isolation_level)))
    }
    
    async fn get_dataset_size(&self, dataset_id: &str) -> Result<usize> {
        let db = self.db.clone();
        let dataset_id = dataset_id.to_string();
        tokio::task::spawn_blocking(move || {
            let db_guard = db.read().map_err(|e| Error::lock(e.to_string()))?;
            let key = format!("dataset:{}:data", dataset_id);
            if let Some(value) = db_guard.get(key.as_bytes())? {
                Ok(value.len())
            } else {
                Ok(0)
            }
        }).await.map_err(|e| Error::internal(format!("获取数据集大小失败: {}", e)))?
    }
    
    async fn get_dataset_chunk(&self, dataset_id: &str, start: usize, end: usize) -> Result<Vec<u8>> {
        let db = self.db.clone();
        let dataset_id = dataset_id.to_string();
        tokio::task::spawn_blocking(move || {
            let db_guard = db.read().map_err(|e| Error::lock(e.to_string()))?;
            let key = format!("dataset:{}:data", dataset_id);
            if let Some(value) = db_guard.get(key.as_bytes())? {
                if end <= value.len() && start < end {
                    Ok(value[start..end].to_vec())
                } else {
                    Err(Error::invalid_input("Invalid chunk range"))
                }
            } else {
                Err(Error::not_found(format!("Dataset {} not found", dataset_id)))
            }
        }).await.map_err(|e| Error::internal(format!("获取数据集块失败: {}", e)))?
    }
    
    fn get_specific_inference_result(&self, model_id: &str, inference_id: &str) -> Result<Option<crate::core::InferenceResult>> {
        let key = format!("inference:results:{}:{}", model_id, inference_id);
        if let Some(value) = self.db.read().map_err(|e| Error::lock(e.to_string()))?.get(key.as_bytes())? {
            Ok(Some(bincode::deserialize(&value)?))
        } else {
            Ok(None)
        }
    }
    
    fn save_detailed_inference_result(
        &self,
        model_id: &str,
        inference_id: &str,
        result: &InferenceResult,
        detail: &InferenceResultDetail,
        processing_time: u64,
    ) -> Result<()> {
        // 保存推理结果
        let result_key = format!("inference:results:{}:{}", model_id, inference_id);
        let result_value = bincode::serialize(result)?;
        
        // 保存详细信息
        let detail_key = format!("inference:details:{}:{}", model_id, inference_id);
        let detail_value = bincode::serialize(detail)?;
        
        // 保存处理时间
        let time_key = format!("inference:time:{}:{}", model_id, inference_id);
        let time_value = bincode::serialize(&processing_time)?;
        
        let db = self.db.write().map_err(|e| Error::lock(e.to_string()))?;
        db.put(result_key.as_bytes(), result_value.as_slice())?;
        db.put(detail_key.as_bytes(), detail_value.as_slice())?;
        db.put(time_key.as_bytes(), time_value.as_slice())?;
        
        Ok(())
    }
    
    fn save_model_info(&self, model_id: &str, info: &ModelInfo) -> Result<()> {
        let key = format!("model:{}:info", model_id);
        let value = bincode::serialize(info)?;
        self.db.write().map_err(|e| Error::lock(e.to_string()))?.put(key.as_bytes(), value.as_slice())?;
        Ok(())
    }
    
    fn get_model_info(&self, model_id: &str) -> Result<Option<ModelInfo>> {
        let key = format!("model:{}:info", model_id);
        if let Some(value) = self.db.read().map_err(|e| Error::lock(e.to_string()))?.get(key.as_bytes())? {
            let info = bincode::deserialize(&value)?;
            Ok(Some(info))
        } else {
            Ok(None)
        }
    }
    
    fn save_model_metrics(&self, model_id: &str, metrics: &ModelMetrics) -> Result<()> {
        let key = format!("model:{}:metrics", model_id);
        let value = bincode::serialize(metrics)?;
        self.db.write().map_err(|e| Error::lock(e.to_string()))?.put(key.as_bytes(), value.as_slice())?;
        Ok(())
    }
    
    fn get_model_metrics(&self, model_id: &str) -> Result<Option<ModelMetrics>> {
        let key = format!("model:{}:metrics", model_id);
        if let Some(value) = self.db.read().map_err(|e| Error::lock(e.to_string()))?.get(key.as_bytes())? {
            let metrics = bincode::deserialize(&value)?;
            Ok(Some(metrics))
        } else {
            Ok(None)
        }
    }
    
    fn get_model(&self, model_id: &str) -> Result<Option<crate::model::Model>> {
        let key = format!("model:{}", model_id);
        if let Some(value) = self.db.read().map_err(|e| Error::lock(e.to_string()))?.get(key.as_bytes())? {
            let model = bincode::deserialize(&value)?;
            Ok(Some(model))
        } else {
            Ok(None)
        }
    }
    
    fn save_model(&self, model_id: &str, model: &crate::model::Model) -> Result<()> {
        let key = format!("model:{}", model_id);
        let value = bincode::serialize(model)?;
        self.db.write().map_err(|e| Error::lock(e.to_string()))?.put(key.as_bytes(), value.as_slice())?;
        Ok(())
    }
    
    fn model_exists(&self, model_id: &str) -> Result<bool> {
        let key = format!("model:{}", model_id);
        Ok(self.db.read().map_err(|e| Error::lock(e.to_string()))?.get(key.as_bytes())?.is_some())
    }
    
    fn has_model(&self, model_id: &str) -> Result<bool> {
        self.model_exists(model_id)
    }

    fn get_dataset(&self, dataset_id: &str) -> Result<Option<crate::data::DataBatch>> {
        let key = format!("dataset:{}:data", dataset_id);
        if let Some(value) = self.db.read().map_err(|e| Error::lock(e.to_string()))?.get(key.as_bytes())? {
            let dataset = bincode::deserialize(&value)?;
            Ok(Some(dataset))
        } else {
            Ok(None)
        }
    }
}

impl StorageEngine for RocksDBStorage {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.db.read().map_err(|e| Error::lock(e.to_string()))?.get(key).map_err(Into::into)
    }

    fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        self.db.write().map_err(|e| Error::lock(e.to_string()))?.put(key, value).map_err(Into::into)
    }

    fn delete(&self, key: &[u8]) -> Result<()> {
        self.db.write().map_err(|e| Error::lock(e.to_string()))?.delete(key).map_err(Into::into)
    }

    fn exists(&self, key: &[u8]) -> Result<bool> {
        Ok(self.get(key)?.is_some())
    }

    fn close(&self) -> Result<()> {
        // RocksDB automatically closes when dropped
        Ok(())
    }
    
    fn scan_prefix(&self, prefix: &[u8]) -> Box<dyn Iterator<Item = Result<(Vec<u8>, Vec<u8>)>> + '_> {
        let db = match self.db.read() {
            Ok(db) => db,
            Err(e) => {
                error!("Failed to acquire read lock on database: {}", e);
                return Box::new(std::iter::empty());
            }
        };
        let iter = db.prefix_iterator(prefix);
        Box::new(iter.map(|item| {
            item.map(|(k, v)| (k.to_vec(), v.to_vec())).map_err(Into::into)
        }))
    }
    
    fn set_options(&mut self, _options: &StorageOptions) -> Result<()> {
        // RocksDB options are set during initialization
        // For runtime changes, we'd need to reopen the database
        warn!("Runtime options changes not supported for RocksDB");
        Ok(())
    }
    
    fn dataset_exists(&self, dataset_id: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<bool>> + Send + '_>> {
        let key = format!("dataset:{}:metadata", dataset_id);
        let exists = self.db.read().unwrap().get(key.as_bytes()).unwrap_or(None).is_some();
        Box::pin(async move { Ok(exists) })
    }
    
    fn get_dataset_data(&self, dataset_id: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>>> + Send + '_>> {
        let key = format!("dataset:{}:data", dataset_id);
        let data = self.db.read().unwrap().get(key.as_bytes()).unwrap_or(None).unwrap_or_default();
        Box::pin(async move { Ok(data) })
    }
    
    fn get_dataset_metadata(&self, dataset_id: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<serde_json::Value>> + Send + '_>> {
        let key = format!("dataset:{}:metadata", dataset_id);
        let db = self.db.read().unwrap();
        let result = if let Some(value) = db.get(key.as_bytes()).unwrap_or(None) {
            serde_json::from_slice(&value).unwrap_or(serde_json::Value::Null)
        } else {
            serde_json::Value::Null
        };
        Box::pin(async move { Ok(result) })
    }
    
    fn save_dataset_data(&self, dataset_id: &str, data: &[u8]) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        let key = format!("dataset:{}:data", dataset_id);
        let result = self.db.write().unwrap().put(key.as_bytes(), data);
        Box::pin(async move { result.map_err(Into::into) })
    }
    
    fn save_dataset_metadata(&self, dataset_id: &str, metadata: &serde_json::Value) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        let key = format!("dataset:{}:metadata", dataset_id);
        let value = serde_json::to_vec(metadata).unwrap_or_default();
        let result = self.db.write().unwrap().put(key.as_bytes(), &value);
        Box::pin(async move { result.map_err(Into::into) })
    }
    
    fn get_dataset_schema(&self, dataset_id: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<serde_json::Value>> + Send + '_>> {
        let key = format!("dataset:{}:schema", dataset_id);
        let db = self.db.read().unwrap();
        let result = if let Some(value) = db.get(key.as_bytes()).unwrap_or(None) {
            serde_json::from_slice(&value).unwrap_or(serde_json::Value::Null)
        } else {
            serde_json::Value::Null
        };
        Box::pin(async move { Ok(result) })
    }
    
    fn save_dataset_schema(&self, dataset_id: &str, schema: &serde_json::Value) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        let key = format!("dataset:{}:schema", dataset_id);
        let value = serde_json::to_vec(schema).unwrap_or_default();
        let result = self.db.write().unwrap().put(key.as_bytes(), &value);
        Box::pin(async move { result.map_err(Into::into) })
    }
    
    fn delete_dataset_complete(&self, dataset_id: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        let data_key = format!("dataset:{}:data", dataset_id);
        let metadata_key = format!("dataset:{}:metadata", dataset_id);
        let schema_key = format!("dataset:{}:schema", dataset_id);
        
        let db = self.db.write().unwrap();
        let _ = db.delete(data_key.as_bytes());
        let _ = db.delete(metadata_key.as_bytes());
        let result = db.delete(schema_key.as_bytes());
        
        Box::pin(async move { result.map_err(Into::into) })
    }
    
    fn list_datasets(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<String>>> + Send + '_>> {
        let prefix = "dataset:";
        let suffix = ":metadata";
        let db = self.db.read().unwrap();
        let mut datasets = Vec::new();
        
        let iter = db.prefix_iterator(prefix.as_bytes());
        for item in iter {
            if let Ok((key, _)) = item {
                let key_str = String::from_utf8_lossy(&key);
                if key_str.starts_with(prefix) && key_str.ends_with(suffix) {
                    let dataset_id = key_str.strip_prefix(prefix).unwrap()
                        .strip_suffix(suffix).unwrap();
                    datasets.push(dataset_id.to_string());
                }
            }
        }
        
        Box::pin(async move { Ok(datasets) })
    }
    
    fn store(&self, key: &str, data: &[u8]) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        let result = self.db.write().unwrap().put(key.as_bytes(), data);
        Box::pin(async move { result.map_err(Into::into) })
    }
    
    fn list_models_with_filters(
        &self,
        _filters: std::collections::HashMap<String, String>,
        _limit: usize,
        _offset: usize,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<crate::model::Model>>> + Send + '_>> {
        // For now, return empty list - would need to implement proper filtering
        Box::pin(async move { Ok(Vec::new()) })
    }
    
    fn count_models(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<usize>> + Send + '_>> {
        let prefix = "model:";
        let db = self.db.read().unwrap();
        let iter = db.prefix_iterator(prefix.as_bytes());
        let count = iter.count();
        Box::pin(async move { Ok(count) })
    }
    
    fn get_data_batch(&self, batch_id: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<crate::data::DataBatch>> + Send + '_>> {
        let key = format!("batch:{}", batch_id);
        let db = self.db.read().unwrap();
        let result = if let Some(value) = db.get(key.as_bytes()).unwrap_or(None) {
            serde_json::from_slice(&value).unwrap_or_default()
        } else {
            crate::data::DataBatch::default()
        };
        Box::pin(async move { Ok(result) })
    }
    
    // Training-related methods removed: vector database does not need training functionality
    
    fn query_dataset(
        &self,
        name: &str,
        limit: Option<usize>,
        _offset: Option<usize>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<serde_json::Value>>> + Send + '_>> {
        let key = format!("dataset:{}:data", name);
        let db = self.db.read().unwrap();
        let mut results = Vec::new();
        
        if let Some(value) = db.get(key.as_bytes()).unwrap_or(None) {
            if let Ok(data) = serde_json::from_slice::<Vec<serde_json::Value>>(&value) {
                results = data.into_iter().take(limit.unwrap_or(1000)).collect();
            }
        }
        
        Box::pin(async move { Ok(results) })
    }

    fn get_dataset_size(&self, dataset_id: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<usize>> + Send + '_>> {
        let key = format!("dataset:{}:data", dataset_id);
        let db = self.db.read().unwrap();
        let size = if let Some(value) = db.get(key.as_bytes()).unwrap_or(None) {
            value.len()
        } else {
            0
        };
        Box::pin(async move { Ok(size) })
    }

    fn get_dataset_chunk(&self, dataset_id: &str, start: usize, end: usize) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>>> + Send + '_>> {
        let key = format!("dataset:{}:data", dataset_id);
        let db = self.db.read().unwrap();
        let chunk = if let Some(value) = db.get(key.as_bytes()).unwrap_or(None) {
            let actual_end = std::cmp::min(end, value.len());
            if start >= value.len() {
                Vec::new()
            } else {
                value[start..actual_end].to_vec()
            }
        } else {
            Vec::new()
        };
        Box::pin(async move { Ok(chunk) })
    }

    fn save_processed_data(
        &self,
        model_id: &str,
        data: &[Vec<f32>],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        let key = format!("model:{}:processed_data", model_id);
        let value = bincode::serialize(data).unwrap_or_default();
        let result = self.db.write().unwrap().put(key.as_bytes(), &value);
        Box::pin(async move { result.map_err(Into::into) })
    }
    
    fn get_batch_data(&self, batch_id: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<crate::data::DataBatch>> + Send + '_>> {
        let key = format!("batch:{}:data", batch_id);
        let db = self.db.read().unwrap();
        let result = if let Some(value) = db.get(key.as_bytes()).unwrap_or(None) {
            match bincode::deserialize::<crate::data::DataBatch>(&value) {
                Ok(batch) => Ok(batch),
                Err(_) => Err(crate::error::Error::deserialization("Failed to deserialize batch data"))
            }
        } else {
            Err(crate::error::Error::not_found("Batch data not found"))
        };
        Box::pin(async move { result })
    }
}

// RocksDB 事务实现
struct RocksDBTransaction {
    db: Arc<RwLock<DB>>,
    id: String,
    state: crate::core::interfaces::storage_interface::TransactionState,
    ops: Vec<RocksDBTxOp>,
    isolation_level: crate::interfaces::storage::IsolationLevel,
}

enum RocksDBTxOp {
    Store { key: String, value: Vec<u8> },
    Delete { key: String },
}

impl RocksDBTransaction {
    fn new(db: Arc<RwLock<DB>>) -> Self {
        Self {
            db,
            id: uuid::Uuid::new_v4().to_string(),
            state: crate::core::interfaces::storage_interface::TransactionState::Active,
            ops: Vec::new(),
            isolation_level: crate::interfaces::storage::IsolationLevel::default(),
        }
    }
    
    fn new_with_isolation(db: Arc<RwLock<DB>>, isolation_level: crate::interfaces::storage::IsolationLevel) -> Self {
        Self {
            db,
            id: uuid::Uuid::new_v4().to_string(),
            state: crate::core::interfaces::storage_interface::TransactionState::Active,
            ops: Vec::new(),
            isolation_level,
        }
    }
}

impl StorageTransaction for RocksDBTransaction {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn state(&self) -> crate::interfaces::storage::TransactionState {
        match self.state {
            crate::core::interfaces::storage_interface::TransactionState::Active => crate::interfaces::storage::TransactionState::Active,
            crate::core::interfaces::storage_interface::TransactionState::Committed => crate::interfaces::storage::TransactionState::Committed,
            crate::core::interfaces::storage_interface::TransactionState::RolledBack => crate::interfaces::storage::TransactionState::RolledBack,
        }
    }
    
    fn isolation_level(&self) -> crate::interfaces::storage::IsolationLevel {
        self.isolation_level
    }
    
    fn commit(self: Box<Self>) -> Result<()> {
        if self.state != crate::core::interfaces::storage_interface::TransactionState::Active {
            return Err(Error::Transaction("事务非活跃状态，无法提交".to_string()));
        }
        
        let db = self.db.clone();
        let ops = self.ops;
        tokio::runtime::Handle::current().block_on(async move {
            let mut db_guard = db.write()
                .map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
            for op in ops {
                match op {
                    RocksDBTxOp::Store { key, value } => {
                        db_guard.put(key.as_bytes(), &value).map_err(|e| Error::StorageError(format!("存储操作失败: {}", e)))?;
                    }
                    RocksDBTxOp::Delete { key } => {
                        db_guard.delete(key.as_bytes()).map_err(|e| Error::StorageError(format!("删除操作失败: {}", e)))?;
                    }
                }
            }
            Ok::<(), Error>(())
        })
    }
    
    fn rollback(self: Box<Self>) -> Result<()> {
        if self.state != crate::core::interfaces::storage_interface::TransactionState::Active {
            return Err(Error::Transaction("事务非活跃状态，无法回滚".to_string()));
        }
        Ok(())
    }
    
    fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        if self.state != crate::core::interfaces::storage_interface::TransactionState::Active {
            return Err(Error::Transaction("事务非活跃状态，无法执行操作".to_string()));
        }
        
        let key_str = String::from_utf8(key.to_vec()).map_err(|e| Error::InvalidInput(format!("无效的键: {}", e)))?;
        self.ops.push(RocksDBTxOp::Store {
            key: key_str,
            value: value.to_vec(),
        });
        Ok(())
    }
    
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let key_str = String::from_utf8(key.to_vec()).map_err(|e| Error::InvalidInput(format!("无效的键: {}", e)))?;
        
        // 先检查事务中的操作
        for op in self.ops.iter().rev() {
            match op {
                RocksDBTxOp::Store { key: tx_key, value } if tx_key == &key_str => {
                    return Ok(Some(value.clone()));
                }
                RocksDBTxOp::Delete { key: tx_key } if tx_key == &key_str => {
                    return Ok(None);
                }
                _ => {}
            }
        }
        
        // 如果事务中没有相关操作，从数据库读取
        let db = self.db.clone();
        let key_str = key_str;
        tokio::runtime::Handle::current().block_on(async move {
            let db_guard = db.read()
                .map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
            Ok(db_guard.get(key_str.as_bytes()).map_err(|e| Error::StorageError(format!("读取操作失败: {}", e)))?)
        })
    }
    
    fn delete(&mut self, key: &[u8]) -> Result<()> {
        if self.state != crate::core::interfaces::storage_interface::TransactionState::Active {
            return Err(Error::Transaction("事务非活跃状态，无法执行操作".to_string()));
        }
        
        let key_str = String::from_utf8(key.to_vec()).map_err(|e| Error::InvalidInput(format!("无效的键: {}", e)))?;
        self.ops.push(RocksDBTxOp::Delete {
            key: key_str,
        });
        Ok(())
    }
} 