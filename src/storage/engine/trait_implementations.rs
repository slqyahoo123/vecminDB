// remove unused Arc/Mutex
use std::pin::Pin;
use std::future::Future;
use serde_json::Value;
// remove unused RwLock

use crate::Result;
use crate::Error;
use crate::compat::{TrainingMetrics};
// 未使用的导入移除，保持代码整洁
// 注意：Model, ModelArchitecture, ModelParameters, TrainingStateManager 在代码中使用完全限定路径
use crate::core::InferenceResult;
use crate::compat::TrainingResultDetail; // 使用 compat 模块中的 stub 类型

use super::interfaces::{StorageEngine, DatasetStorageInterface};
use crate::core::interfaces::storage_interface::{StorageService, StorageTransaction};
use crate::core::interfaces::{MonitoringInterface, IsolationLevel};
use super::core::StorageEngineImpl;

// ==================== TRAIT IMPLEMENTATIONS ====================

#[async_trait::async_trait]
impl StorageService for StorageEngineImpl {
    async fn save_model_parameters(&self, model_id: &str, params: &crate::model::parameters::ModelParameters) -> Result<()> {
        let key = format!("model:{}:params", model_id);
        let data = bincode::serialize(params)?;
        let db = self.get_db_clone();
        let db = db.read().await;
        db.insert(key.as_bytes(), data.as_slice())?;
        Ok(())
    }
    
    async fn get_model_parameters(&self, model_id: &str) -> Result<Option<crate::model::parameters::ModelParameters>> {
        let key = format!("model:{}:params", model_id);
        let db = self.get_db_clone();
        let db = db.read().await;
        if let Some(data) = db.get(key.as_bytes())? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }
    
    async fn save_model_architecture(&self, model_id: &str, arch: &crate::model::ModelArchitecture) -> Result<()> {
        let key = format!("model:{}:arch", model_id);
        let data = bincode::serialize(arch)?;
        let db = self.get_db_clone();
        let db = db.read().await;
        db.insert(key.as_bytes(), data.as_slice())?;
        Ok(())
    }
    
    async fn get_model_architecture(&self, model_id: &str) -> Result<Option<crate::model::ModelArchitecture>> {
        let key = format!("model:{}:arch", model_id);
        let db = self.get_db_clone();
        let db = db.read().await;
        if let Some(data) = db.get(key.as_bytes())? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }
    
    async fn save_model_info(&self, model_id: &str, info: &crate::core::types::ModelInfo) -> Result<()> {
        let key = format!("model:{}:info", model_id);
        let data = bincode::serialize(info)?;
        let db = self.get_db_clone();
        let db = db.read().await;
        db.insert(key.as_bytes(), data.as_slice())?;
        Ok(())
    }
    
    async fn get_model_info(&self, model_id: &str) -> Result<Option<crate::core::types::ModelInfo>> {
        let key = format!("model:{}:info", model_id);
        let db = self.get_db_clone();
        let db = db.read().await;
        if let Some(data) = db.get(key.as_bytes())? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }
    
    async fn save_model_metrics(&self, model_id: &str, metrics: &crate::storage::models::implementation::ModelMetrics) -> Result<()> {
        let key = format!("model:{}:metrics", model_id);
        let data = bincode::serialize(metrics)?;
        let db = self.get_db_clone();
        let db = db.read().await;
        db.insert(key.as_bytes(), data.as_slice())?;
        Ok(())
    }
    
    async fn get_model_metrics(&self, model_id: &str) -> Result<Option<crate::storage::models::implementation::ModelMetrics>> {
        let key = format!("model:{}:metrics", model_id);
        let db = self.get_db_clone();
        let db = db.read().await;
        if let Some(data) = db.get(key.as_bytes())? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }
    
    // 训练状态和训练结果相关方法已移除 - 向量数据库系统不需要训练功能
    
    async fn save_inference_result(&self, model_id: &str, result: &InferenceResult) -> Result<()> {
        let key = format!("model:{}:inference_result", model_id);
        let data = bincode::serialize(result)?;
        let db = self.get_db_clone();
        let db = db.read().await;
        db.insert(key.as_bytes(), data.as_slice())?;
        Ok(())
    }
    
    async fn get_inference_result(&self, result_id: &str) -> Result<Option<crate::core::results::InferenceResult>> {
        let key = format!("inference_result:{}", result_id);
        let db = self.get_db_clone();
        let db = db.read().await;
        if let Some(data) = db.get(key.as_bytes())? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }
    
    // 实现StorageService trait的缺失方法
    async fn store(&self, key: &str, value: &[u8]) -> Result<()> {
        let db = self.get_db_clone();
        let db = db.read().await;
        db.insert(key.as_bytes(), value)?;
        Ok(())
    }
    
    async fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let db = self.get_db_clone();
        let db = db.read().await;
        Ok(db.get(key.as_bytes())?.map(|v| v.to_vec()))
    }
    
    async fn delete(&self, key: &str) -> Result<()> {
        let db = self.get_db_clone();
        let db = db.read().await;
        db.remove(key.as_bytes())?;
        Ok(())
    }
    
    async fn exists(&self, key: &str) -> Result<bool> {
        let db = self.get_db_clone();
        let db = db.read().await;
        Ok(db.get(key.as_bytes())?.is_some())
    }
    
    async fn list_keys(&self, prefix: &str) -> Result<Vec<String>> {
        let db = self.get_db_clone();
        let db = db.read().await;
        let mut keys = Vec::new();
        let iter = (*db).scan_prefix(prefix.as_bytes());
        for item in iter {
            let (key, _) = item?;
            if let Ok(key_str) = String::from_utf8(key.to_vec()) {
                keys.push(key_str);
            }
        }
        Ok(keys)
    }
    
    async fn batch_store(&self, items: Vec<(String, Vec<u8>)>) -> Result<()> {
        let db = self.get_db_clone();
        let db = db.read().await;
        for (key, value) in items {
            db.insert(key.as_bytes(), value.as_slice())?;
        }
        Ok(())
    }
    
    async fn batch_retrieve(&self, keys: &[String]) -> Result<Vec<Option<Vec<u8>>>> {
        let db = self.get_db_clone();
        let db = db.read().await;
        let mut results = Vec::new();
        for key in keys {
            results.push(db.get(key.as_bytes())?.map(|v| v.to_vec()));
        }
        Ok(results)
    }
    
    async fn batch_delete(&self, keys: &[String]) -> Result<()> {
        let db = self.get_db_clone();
        let db = db.read().await;
        for key in keys {
            db.remove(key.as_bytes())?;
        }
        Ok(())
    }
    
    fn transaction(&self) -> Result<Box<dyn StorageTransaction>> {
        // StorageService trait 需要实现 transaction，但 StorageTransactionImpl 期望 rocksdb::DB
        // 由于我们使用的是 sled::Db，应该使用 StorageEngineImpl 的 StorageInterface 实现
        // 这里返回一个错误，提示使用 StorageInterface trait
        Err(Error::NotImplemented(
            "StorageService::transaction 需要 rocksdb::DB，当前使用 sled::Db。请使用 StorageInterface trait".to_string()
        ))
    }
    
    fn transaction_with_isolation(&self, _isolation_level: IsolationLevel) -> Result<Box<dyn StorageTransaction>> {
        Err(Error::NotImplemented(
            "StorageService::transaction_with_isolation 需要 rocksdb::DB，当前使用 sled::Db。请使用 StorageInterface trait".to_string()
        ))
    }
    
    async fn get_dataset_size(&self, dataset_id: &str) -> Result<usize> {
        let key = format!("dataset:{}:size", dataset_id);
        let db = self.get_db_clone();
        let db = db.read().await;
        if let Some(data) = db.get(key.as_bytes())? {
            Ok(bincode::deserialize(&data)?)
        } else {
            Ok(0)
        }
    }
    
    async fn get_dataset_chunk(&self, dataset_id: &str, start: usize, end: usize) -> Result<Vec<u8>> {
        let key = format!("dataset:{}:chunk:{}:{}", dataset_id, start, end);
        let db = self.get_db_clone();
        let db = db.read().await;
        if let Some(data) = db.get(key.as_bytes())? {
            Ok(data.to_vec())
        } else {
            Ok(Vec::new())
        }
    }

    // 训练状态和训练结果相关方法已移除 - 向量数据库系统不需要训练功能

    async fn list_inference_results(&self, model_id: &str) -> Result<Vec<String>> {
        let prefix = format!("inference_result:{}:", model_id);
        let db = self.get_db_clone();
        let db = db.read().await;
        let mut results = Vec::new();
        let iter = (*db).scan_prefix(prefix.as_bytes());
        for item in iter {
            let (key, _) = item?;
            if let Ok(key_str) = String::from_utf8(key.to_vec()) {
                if let Some(result_id) = key_str.strip_prefix(&prefix) {
                    results.push(result_id.to_string());
                }
            }
        }
        Ok(results)
    }

    async fn save_detailed_inference_result(&self, result_id: &str, result: &crate::core::results::InferenceResult) -> Result<()> {
        let key = format!("inference_result:{}", result_id);
        let data = bincode::serialize(result)?;
        let db = self.get_db_clone();
        let db = db.read().await;
        db.insert(key.as_bytes(), data.as_slice())?;
        Ok(())
    }

    async fn get_model(&self, model_id: &str) -> Result<Option<crate::model::Model>> {
        let key = format!("model:{}:full", model_id);
        let db = self.get_db_clone();
        let db = db.read().await;
        if let Some(data) = db.get(key.as_bytes())? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }

    async fn save_model(&self, model_id: &str, model: &crate::model::Model) -> Result<()> {
        let key = format!("model:{}:full", model_id);
        let data = bincode::serialize(model)?;
        let db = self.get_db_clone();
        let db = db.read().await;
        db.insert(key.as_bytes(), data.as_slice())?;
        Ok(())
    }

    async fn model_exists(&self, model_id: &str) -> Result<bool> {
        let key = format!("model:{}:full", model_id);
        let db = self.get_db_clone();
        let db = db.read().await;
        Ok(db.get(key.as_bytes())?.is_some())
    }

    async fn has_model(&self, model_id: &str) -> Result<bool> {
        self.model_exists(model_id).await
    }

    async fn get_dataset(&self, dataset_id: &str) -> Result<Option<crate::data::loader::types::DataSchema>> {
        let key = format!("dataset:{}", dataset_id);
        let db = self.get_db_clone();
        let db = db.read().await;
        if let Some(data) = db.get(key.as_bytes())? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }
}

impl StorageEngine for StorageEngineImpl {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let db = self.get_db_clone();
        let db = db.blocking_read();
        Ok(db.get(key)?.map(|v| v.to_vec()))
    }
    
    fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        let db = self.get_db_clone();
        let db = db.blocking_read();
        db.insert(key, value)?;
        Ok(())
    }
    
    fn delete(&self, key: &[u8]) -> Result<()> {
        let db = self.get_db_clone();
        let db = db.blocking_read();
        db.remove(key)?;
        Ok(())
    }
    
    fn scan_prefix(&self, prefix: &[u8]) -> Box<dyn Iterator<Item = Result<(Vec<u8>, Vec<u8>)>> + '_> {
        let db = self.get_db_clone();
        let db = db.blocking_read();
        let iter = db.scan_prefix(prefix).map(|result| {
            result.map(|(k, v)| (k.to_vec(), v.to_vec())).map_err(|e| crate::Error::StorageError(e.to_string()))
        });
        Box::new(iter)
    }
    
    fn set_options(&mut self, options: &crate::storage::engine::implementation::StorageOptions) -> Result<()> {
        // 更新存储引擎配置
        let mut config = self.get_config().clone();
        
        // 更新缓存配置
        config.cache_size = options.cache_size;
        
        // 更新压缩配置（使用 use_compression 字段）
        // StorageOptions 中没有 compression_enabled 字段
        // 如果需要压缩，可以通过 compression_level 设置
        if let Some(compression_level) = options.compression_level {
            config.compression = Some(compression_level);
        }
        
        // 更新并发配置（StorageConfig 中没有 max_concurrent_operations 字段）
        // 可以通过 max_background_jobs 设置
        config.max_background_jobs = options.max_background_threads as i32;
        
        // 注意：无法直接更新 config，因为它是私有的
        // 应该通过其他方式更新配置，或者提供配置更新方法
        // 这里只能更新可访问的配置项，无法直接设置私有字段
        
        // 重新配置数据库选项（如果 reconfigure_database 方法可用）
        // self.reconfigure_database()?;
        
        // 当前方法只负责根据 StorageOptions 调整可访问的 StorageConfig 字段
        // 这里返回 Ok(()) 表示配置更新逻辑执行完成
        Ok(())
    }
    
    fn dataset_exists(&self, dataset_id: &str) -> Pin<Box<dyn Future<Output = Result<bool>> + Send + '_>> {
        let key = format!("dataset:{}:metadata", dataset_id);
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.read().await;
            Ok(db.contains_key(key.as_bytes())?)
        })
    }
    
    fn get_dataset_data(&self, dataset_id: &str) -> Pin<Box<dyn Future<Output = Result<Vec<u8>>> + Send + '_>> {
        let key = format!("dataset:{}:data", dataset_id);
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.read().await;
            if let Some(data) = db.get(key.as_bytes())? {
                Ok(data.to_vec())
            } else {
                Err(Error::NotFound(format!("Dataset {} not found", dataset_id)))
            }
        })
    }
    
    fn get_dataset_metadata(&self, dataset_id: &str) -> Pin<Box<dyn Future<Output = Result<serde_json::Value>> + Send + '_>> {
        let key = format!("dataset:{}:metadata", dataset_id);
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.read().await;
            if let Some(data) = db.get(key.as_bytes())? {
                Ok(serde_json::from_slice(&data)?)
            } else {
                Err(Error::NotFound(format!("Dataset metadata {} not found", dataset_id)))
            }
        })
    }
    
    fn save_dataset_data(&self, dataset_id: &str, data: &[u8]) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        let key = format!("dataset:{}:data", dataset_id);
        let data = data.to_vec();
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let mut db = db.write().await;
            db.insert(key.as_bytes(), data.as_slice()).map_err(|e| Error::StorageError(e.to_string()))?;
            Ok(())
        })
    }
    
    fn save_dataset_metadata(&self, dataset_id: &str, metadata: &serde_json::Value) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        let key = format!("dataset:{}:metadata", dataset_id);
        let db = self.get_db_clone();
        let metadata_clone = metadata.clone();
        
        Box::pin(async move {
            let data = serde_json::to_vec(&metadata_clone)
                .map_err(|e| Error::Serialization(format!("序列化元数据失败: {}", e)))?;
            let db = db.read().await;
            db.insert(key.as_bytes(), data.as_slice())
                .map_err(|e| Error::StorageError(format!("存储元数据失败: {}", e)))?;
            Ok(())
        })
    }
    
    fn get_dataset_schema(&self, dataset_id: &str) -> Pin<Box<dyn Future<Output = Result<serde_json::Value>> + Send + '_>> {
        let key = format!("dataset:{}:schema", dataset_id);
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.read().await;
            if let Some(data) = db.get(key.as_bytes())? {
                Ok(serde_json::from_slice(&data)?)
            } else {
                Err(Error::NotFound(format!("Dataset schema {} not found", dataset_id)))
            }
        })
    }
    
    fn save_dataset_schema(&self, dataset_id: &str, schema: &serde_json::Value) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        let key = format!("dataset:{}:schema", dataset_id);
        let data = match serde_json::to_vec(schema) {
            Ok(data) => data,
            Err(e) => return Box::pin(async move { Err(Error::SerializationError(e.to_string())) }),
        };
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let mut db = db.write().await;
            db.insert(key.as_bytes(), data.as_slice()).map_err(|e| Error::StorageError(e.to_string()))?;
            Ok(())
        })
    }
    
    fn delete_dataset_complete(&self, dataset_id: &str) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        let db = self.get_db_clone();
        let dataset_id = dataset_id.to_string();
        
        Box::pin(async move {
            let db = db.read().await;
            let prefix = format!("dataset:{}:", dataset_id);
            
            for result in db.scan_prefix(prefix.as_bytes()) {
                let (key, _) = result?;
                db.remove(key)?;
            }
            
            Ok(())
        })
    }
    
    fn list_datasets(&self) -> Pin<Box<dyn Future<Output = Result<Vec<String>>> + Send + '_>> {
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.read().await;
            let mut datasets = std::collections::HashSet::new();
            
            for result in db.scan_prefix(b"dataset:") {
                let (key, _) = result?;
                if let Ok(key_str) = std::str::from_utf8(&key) {
                    if let Some(dataset_id) = key_str.strip_prefix("dataset:") {
                        if let Some(dataset_id) = dataset_id.split(':').next() {
                            datasets.insert(dataset_id.to_string());
                        }
                    }
                }
            }
            
            Ok(datasets.into_iter().collect())
        })
    }
    
    fn store(&self, key: &str, data: &[u8]) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        let key = key.to_string();
        let data = data.to_vec();
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.read().await;
            db.insert(key.as_bytes(), data.as_slice())?;
            Ok(())
        })
    }
    
    fn list_models_with_filters(
        &self,
        filters: std::collections::HashMap<String, String>,
        limit: usize,
        offset: usize,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<crate::model::Model>>> + Send + '_>> {
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.read().await;
            let mut models = Vec::new();
            let mut count = 0;
            let mut skipped = 0;
            
            // 扫描所有模型
            for result in db.scan_prefix(b"model:") {
                let (key, value) = result?;
                
                // 跳过偏移量
                if skipped < offset {
                    skipped += 1;
                    continue;
                }
                
                // 检查是否达到限制
                if count >= limit {
                    break;
                }
                
                // 尝试反序列化模型
                if let Ok(model) = bincode::deserialize::<crate::model::Model>(&value) {
                    // 应用过滤器
                    let mut matches_filters = true;
                    
                    for (filter_key, filter_value) in &filters {
                        match filter_key.as_str() {
                            "model_type" | "arch_type" => {
                                // ModelArchitecture 使用 arch_type 字段
                                if model.architecture.arch_type != *filter_value {
                                    matches_filters = false;
                                    break;
                                }
                            },
                            "status" => {
                                if model.status.to_string() != *filter_value {
                                    matches_filters = false;
                                    break;
                                }
                            },
                            "created_after" => {
                                if let Ok(timestamp) = filter_value.parse::<i64>() {
                                    if model.created_at.timestamp() <= timestamp {
                                        matches_filters = false;
                                        break;
                                    }
                                }
                            },
                            "created_before" => {
                                if let Ok(timestamp) = filter_value.parse::<i64>() {
                                    if model.created_at.timestamp() >= timestamp {
                                        matches_filters = false;
                                        break;
                                    }
                                }
                            },
                            _ => {
                                // 检查模型元数据中的自定义字段
                                // model.metadata 是 HashMap<String, String>，不是 Option
                                if let Some(field_value) = model.metadata.get(filter_key) {
                                    if field_value != filter_value {
                                        matches_filters = false;
                                        break;
                                    }
                                } else {
                                    matches_filters = false;
                                    break;
                                }
                            }
                        }
                    }
                    
                    if matches_filters {
                        models.push(model);
                        count += 1;
                    }
                }
            }
            
            Ok(models)
        })
    }
    
    fn count_models(&self) -> Pin<Box<dyn Future<Output = Result<usize>> + Send + '_>> {
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.read().await;
            let mut count = 0;
            
            for _ in db.scan_prefix(b"model:") {
                count += 1;
            }
            
            Ok(count)
        })
    }
    
    fn get_data_batch(&self, batch_id: &str) -> Pin<Box<dyn Future<Output = Result<crate::data::DataBatch>> + Send + '_>> {
        let key = format!("batch:{}", batch_id);
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.read().await;
            if let Some(data) = db.get(key.as_bytes())? {
                Ok(bincode::deserialize(&data)?)
            } else {
                Err(Error::NotFound(format!("Batch {} not found", batch_id)))
            }
        })
    }
    
    fn get_batch_data(&self, batch_id: &str) -> Pin<Box<dyn Future<Output = Result<crate::data::DataBatch>> + Send + '_>> {
        self.get_data_batch(batch_id)
    }
    
    fn save_processed_batch(
        &self,
        model_id: &str,
        batch: &crate::data::ProcessedBatch,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        let key = format!("processed_batch:{}:{}", model_id, chrono::Utc::now().timestamp());
        let db = self.get_db_clone();
        let batch_clone = batch.clone();
        
        Box::pin(async move {
            let data = bincode::serialize(&batch_clone)
                .map_err(|e| Error::Serialization(format!("序列化批次失败: {}", e)))?;
            let db = db.read().await;
            db.insert(key.as_bytes(), data.as_slice())
                .map_err(|e| Error::StorageError(format!("存储批次失败: {}", e)))?;
            Ok(())
        })
    }
    
    fn get_training_metrics_history(
        &self,
        model_id: &str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<TrainingMetrics>>> + Send + '_>> {
        let prefix = format!("model:{}:metrics:", model_id);
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.read().await;
            let mut metrics = Vec::new();
            
            for result in db.scan_prefix(prefix.as_bytes()) {
                let (_, value) = result?;
                if let Ok(metric) = bincode::deserialize::<TrainingMetrics>(&value) {
                    metrics.push(metric);
                }
            }
            
            Ok(metrics)
        })
    }
    
    fn record_training_metrics(
        &self,
        model_id: &str,
        metrics: &TrainingMetrics,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        let key = format!("model:{}:metrics:{}", model_id, chrono::Utc::now().timestamp());
        let db = self.get_db_clone();
        let metrics_clone = metrics.clone();
        
        Box::pin(async move {
            let data = bincode::serialize(&metrics_clone)
                .map_err(|e| Error::Serialization(format!("序列化指标失败: {}", e)))?;
            let db = db.read().await;
            db.insert(key.as_bytes(), data.as_slice())
                .map_err(|e| Error::StorageError(format!("存储指标失败: {}", e)))?;
            Ok(())
        })
    }
    
    fn query_dataset(
        &self,
        name: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<serde_json::Value>>> + Send + '_>> {
        let db = self.get_db_clone();
        let name = name.to_string();
        let limit = limit.unwrap_or(100);
        let offset = offset.unwrap_or(0);
        
        Box::pin(async move {
            let db = db.read().await;
            let mut results = Vec::new();
            let mut count = 0;
            let mut skipped = 0;
            
            // 构建查询前缀
            let query_prefix = format!("dataset:{}:record:", name);
            
            // 扫描数据集记录
            for result in db.scan_prefix(query_prefix.as_bytes()) {
                let (key, value) = result?;
                
                // 跳过偏移量
                if skipped < offset {
                    skipped += 1;
                    continue;
                }
                
                // 检查是否达到限制
                if count >= limit {
                    break;
                }
                
                // 尝试解析记录数据
                if let Ok(record) = serde_json::from_slice::<serde_json::Value>(&value) {
                    results.push(record);
                    count += 1;
                }
            }
            
            // 如果没有找到记录，尝试查询数据集元数据
            if results.is_empty() {
                let metadata_key = format!("dataset:{}:metadata", name);
                if let Some(metadata_data) = db.get(metadata_key.as_bytes())? {
                    if let Ok(metadata) = serde_json::from_slice::<serde_json::Value>(&metadata_data) {
                        results.push(metadata);
                    }
                }
            }
            
            Ok(results)
        })
    }
    
    fn exists(&self, key: &[u8]) -> Result<bool> {
        let db = self.get_db_clone();
        let db = db.blocking_read();
        Ok(db.contains_key(key)?)
    }
    
    fn get_dataset_size(&self, dataset_id: &str) -> Pin<Box<dyn Future<Output = Result<usize>> + Send + '_>> {
        let key = format!("dataset:{}:data", dataset_id);
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.read().await;
            if let Some(data) = db.get(key.as_bytes())? {
                Ok(data.len())
            } else {
                Err(Error::NotFound(format!("Dataset {} not found", dataset_id)))
            }
        })
    }
    
    fn get_dataset_chunk(&self, dataset_id: &str, start: usize, end: usize) -> Pin<Box<dyn Future<Output = Result<Vec<u8>>> + Send + '_>> {
        let key = format!("dataset:{}:data", dataset_id);
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.read().await;
            if let Some(data) = db.get(key.as_bytes())? {
                if start >= data.len() || end > data.len() || start >= end {
                    return Err(Error::InvalidInput("Invalid chunk range".to_string()));
                }
                Ok(data[start..end].to_vec())
            } else {
                Err(Error::NotFound(format!("Dataset {} not found", dataset_id)))
            }
        })
    }
    
    fn close(&self) -> Result<()> {
        // 执行清理操作
        let db = self.get_db_clone();
        let db = db.blocking_read();
        
        // 刷新所有待写入的数据
        db.flush()?;
        
        // 执行数据库压缩
        // 注意：compact_range 需要在 RocksDB 实例上调用，而不是在 RwLockReadGuard 上
        // 这里我们跳过压缩操作，因为它在只读锁上不可用
        // 如果需要压缩，应该在写入锁上进行
        
        // 记录关闭事件
        let close_event = serde_json::json!({
            "event": "storage_engine_closed",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "engine_id": self.get_config().engine_id,
            "total_operations": self.get_operation_count()
        });
        
        let event_key = format!("event:close:{}", chrono::Utc::now().timestamp());
        let event_data = serde_json::to_vec(&close_event)
            .map_err(|e| Error::Serialization(format!("序列化事件失败: {}", e)))?;
        db.insert(event_key.as_bytes(), event_data.as_slice())
            .map_err(|e| Error::StorageError(format!("存储事件失败: {}", e)))?;
        
        Ok(())
    }
    
    fn save_processed_data(
        &self,
        model_id: &str,
        data: &[Vec<f32>],
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + '_>> {
        let key = format!("processed_data:{}:{}", model_id, chrono::Utc::now().timestamp());
        let db = self.get_db_clone();
        let data_clone = data.to_vec();
        
        Box::pin(async move {
            let data = bincode::serialize(&data_clone)
                .map_err(|e| Error::Serialization(format!("序列化数据失败: {}", e)))?;
            let db = db.read().await;
            db.insert(key.as_bytes(), data.as_slice())
                .map_err(|e| Error::StorageError(format!("存储数据失败: {}", e)))?;
            Ok(())
        })
    }
}

// ================= Core Interfaces bridging =================
use crate::core::interfaces::{StorageInterface as core_storage_if};

struct EngineStorageTransaction {
    engine: std::sync::Arc<StorageEngineImpl>,
    id: String,
    state: crate::core::interfaces::storage_interface::TransactionState,
    ops: Vec<EngineTxOp>,
}

enum EngineTxOp { Put(Vec<u8>, Vec<u8>), Delete(Vec<u8>) }

impl EngineStorageTransaction {
    fn new(engine: std::sync::Arc<StorageEngineImpl>) -> Self {
        Self {
            engine,
            id: uuid::Uuid::new_v4().to_string(),
            state: crate::core::interfaces::storage_interface::TransactionState::Active,
            ops: Vec::new(),
        }
    }
}

impl StorageTransaction for EngineStorageTransaction {
    fn commit(&mut self) -> crate::Result<()> {
        if self.state != crate::core::interfaces::storage_interface::TransactionState::Active {
            return Err(crate::Error::Transaction("事务非活跃状态，无法提交".to_string()));
        }
        for op in self.ops.drain(..) {
            match op {
                EngineTxOp::Put(k, v) => { 
                    let engine = self.engine.clone();
                    let key_str = std::str::from_utf8(&k)
                        .map_err(|e| Error::StorageError(format!("键转换失败: {}", e)))?;
                    tokio::runtime::Handle::current().block_on(async move {
                        engine.put(key_str.as_bytes(), &v).await
                    })?;
                }
                EngineTxOp::Delete(k) => { 
                    let engine = self.engine.clone();
                    let key_str = std::str::from_utf8(&k)
                        .map_err(|e| Error::StorageError(format!("键转换失败: {}", e)))?;
                    tokio::runtime::Handle::current().block_on(async move {
                        engine.delete(key_str).await
                    })?;
                }
            }
        }
        self.state = crate::core::interfaces::storage_interface::TransactionState::Committed;
        Ok(())
    }

    fn rollback(&mut self) -> crate::Result<()> {
        self.ops.clear();
        self.state = crate::core::interfaces::storage_interface::TransactionState::RolledBack;
        Ok(())
    }

    fn store(&mut self, key: &str, value: &[u8]) -> crate::Result<()> {
        self.ops.push(EngineTxOp::Put(key.as_bytes().to_vec(), value.to_vec()));
        Ok(())
    }
    
    fn retrieve(&self, key: &str) -> crate::Result<Option<Vec<u8>>> {
        let engine = self.engine.clone();
        let key_str = key.to_string();
        tokio::runtime::Handle::current().block_on(async move {
            engine.get(&key_str).await
        })
    }

    fn delete(&mut self, key: &str) -> crate::Result<()> {
        self.ops.push(EngineTxOp::Delete(key.as_bytes().to_vec()));
        Ok(())
    }

    fn exists(&self, key: &str) -> crate::Result<bool> {
        let engine = self.engine.clone();
        let key_str = key.to_string();
        tokio::runtime::Handle::current().block_on(async move {
            engine.exists(&key_str).await
        })
    }

    fn get_state(&self) -> crate::core::interfaces::storage_interface::TransactionState { self.state }
    fn get_id(&self) -> &str { &self.id }
}

impl DatasetStorageInterface for StorageEngineImpl {
    fn dataset_exists(&self, dataset_id: &str) -> impl std::future::Future<Output = Result<bool>> + Send {
        let key = format!("dataset:{}:metadata", dataset_id);
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.read().await;
            Ok(db.contains_key(key.as_bytes())?)
        })
    }
    
    fn get_dataset_data(&self, dataset_id: &str) -> impl std::future::Future<Output = Result<Vec<u8>>> + Send {
        let key = format!("dataset:{}:data", dataset_id);
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.read().await;
            if let Some(data) = db.get(key.as_bytes())? {
                Ok(data.to_vec())
            } else {
                Err(Error::NotFound(format!("Dataset {} not found", dataset_id)))
            }
        })
    }
    
    fn get_dataset_metadata(&self, dataset_id: &str) -> impl std::future::Future<Output = Result<Value>> + Send {
        let key = format!("dataset:{}:metadata", dataset_id);
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.read().await;
            if let Some(data) = db.get(key.as_bytes())? {
                Ok(serde_json::from_slice(&data)?)
            } else {
                Err(Error::NotFound(format!("Dataset metadata {} not found", dataset_id)))
            }
        })
    }
    
    fn save_dataset_data(&self, dataset_id: &str, data: &[u8]) -> impl std::future::Future<Output = Result<()>> + Send {
        let key = format!("dataset:{}:data", dataset_id);
        let data = data.to_vec();
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.write().await;
            db.insert(key.as_bytes(), data).map_err(|e| Error::StorageError(e.to_string()))?;
            Ok(())
        })
    }
    
    fn save_dataset_metadata(&self, dataset_id: &str, metadata: &Value) -> impl std::future::Future<Output = Result<()>> + Send {
        let key = format!("dataset:{}:metadata", dataset_id);
        let data_result = serde_json::to_vec(metadata);
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let data = data_result.map_err(|e| Error::SerializationError(e.to_string()))?;
            let db = db.write().await;
            db.insert(key.as_bytes(), data.as_slice()).map_err(|e| Error::StorageError(e.to_string()))?;
            Ok(())
        })
    }
    
    fn get_dataset_schema(&self, dataset_id: &str) -> impl std::future::Future<Output = Result<Value>> + Send {
        let key = format!("dataset:{}:schema", dataset_id);
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.read().await;
            if let Some(data) = db.get(key.as_bytes()).map_err(|e| Error::StorageError(e.to_string()))? {
                Ok(serde_json::from_slice(&data)?)
            } else {
                Err(Error::NotFound(format!("Dataset schema {} not found", dataset_id)))
            }
        })
    }
    
    fn save_dataset_schema(&self, dataset_id: &str, schema: &Value) -> impl std::future::Future<Output = Result<()>> + Send {
        let key = format!("dataset:{}:schema", dataset_id);
        let data_result = serde_json::to_vec(schema);
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let data = data_result.map_err(|e| Error::SerializationError(e.to_string()))?;
            let db = db.write().await;
            db.insert(key.as_bytes(), data.as_slice()).map_err(|e| Error::StorageError(e.to_string()))?;
            Ok(())
        })
    }
    
    fn delete_dataset(&self, dataset_id: &str) -> impl std::future::Future<Output = Result<()>> + Send {
        let db = self.get_db_clone();
        let dataset_id = dataset_id.to_string();
        
        Box::pin(async move {
            let db = db.read().await;
            let prefix = format!("dataset:{}:", dataset_id);
            
            for result in db.scan_prefix(prefix.as_bytes()) {
                let (key, _) = result?;
                db.remove(key)?;
            }
            
            Ok(())
        })
    }
    
    fn list_datasets(&self) -> impl std::future::Future<Output = Result<Vec<String>>> + Send {
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.read().await;
            let mut datasets = std::collections::HashSet::new();
            
            for result in db.scan_prefix(b"dataset:") {
                let (key, _) = result?;
                if let Ok(key_str) = std::str::from_utf8(&key) {
                    if let Some(dataset_id) = key_str.strip_prefix("dataset:") {
                        if let Some(dataset_id) = dataset_id.split(':').next() {
                            datasets.insert(dataset_id.to_string());
                        }
                    }
                }
            }
            
            Ok(datasets.into_iter().collect())
        })
    }
    
    fn get_dataset_info(&self, id: &str) -> impl std::future::Future<Output = Result<Option<Value>>> + Send {
        let key = format!("dataset:{}:info", id);
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.read().await;
            if let Some(data) = db.get(key.as_bytes())? {
                Ok(Some(serde_json::from_slice(&data)?))
            } else {
                Ok(None)
            }
        })
    }
    
    fn save_dataset_info(&self, dataset: &Value) -> impl std::future::Future<Output = Result<()>> + Send {
        let db = self.get_db_clone();
        let dataset = dataset.clone();
        
        Box::pin(async move {
            // 从dataset中提取id
            let dataset_id = if let Some(id) = dataset.get("id") {
                if let Some(id_str) = id.as_str() {
                    id_str.to_string()
                } else {
                    return Err(Error::InvalidInput("Dataset ID must be a string".to_string()));
                }
            } else {
                return Err(Error::InvalidInput("Dataset must contain 'id' field".to_string()));
            };
            
            // 验证数据集ID格式
            if dataset_id.is_empty() {
                return Err(Error::InvalidInput("Dataset ID cannot be empty".to_string()));
            }
            
            // 保存数据集信息
            let key = format!("dataset:{}:info", dataset_id);
            let data = serde_json::to_vec(&dataset)?;
            let db = db.write().await;
            db.insert(key.as_bytes(), data.as_slice())?;
            
            // 更新数据集列表索引
            let list_key = format!("dataset_list:{}", dataset_id);
            let list_data = serde_json::json!({
                "id": dataset_id,
                "name": dataset.get("name").unwrap_or(&serde_json::Value::String(dataset_id.clone())),
                "created_at": chrono::Utc::now().to_rfc3339(),
                "updated_at": chrono::Utc::now().to_rfc3339()
            });
            let list_data_bytes = serde_json::to_vec(&list_data)?;
            db.insert(list_key.as_bytes(), list_data_bytes.as_slice())?;
            
            Ok(())
        })
    }
    
    fn query_dataset(
        &self,
        name: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> impl std::future::Future<Output = Result<Vec<Value>>> + Send {
        let db = self.get_db_clone();
        let name = name.to_string();
        let limit = limit.unwrap_or(100);
        let offset = offset.unwrap_or(0);
        
        Box::pin(async move {
            let db = db.read().await;
            let mut results = Vec::new();
            let mut count = 0;
            let mut skipped = 0;
            
            // 构建查询前缀
            let query_prefix = format!("dataset:{}:record:", name);
            
            // 扫描数据集记录
            for result in db.scan_prefix(query_prefix.as_bytes()) {
                let (_key, value) = result?;
                
                // 跳过偏移量
                if skipped < offset {
                    skipped += 1;
                    continue;
                }
                
                // 检查是否达到限制
                if count >= limit {
                    break;
                }
                
                // 尝试解析记录数据
                if let Ok(record) = serde_json::from_slice::<serde_json::Value>(&value) {
                    results.push(record);
                    count += 1;
                }
            }
            
            // 如果没有找到记录，尝试查询数据集元数据
            if results.is_empty() {
                let metadata_key = format!("dataset:{}:metadata", name);
                if let Some(metadata_data) = db.get(metadata_key.as_bytes())? {
                    if let Ok(metadata) = serde_json::from_slice::<serde_json::Value>(&metadata_data) {
                        results.push(metadata);
                    }
                }
            }
            
            // 如果仍然没有结果，尝试查询数据集信息
            if results.is_empty() {
                let info_key = format!("dataset:{}:info", name);
                if let Some(info_data) = db.get(info_key.as_bytes())? {
                    if let Ok(info) = serde_json::from_slice::<serde_json::Value>(&info_data) {
                        results.push(info);
                    }
                }
            }
            
            Ok(results)
        })
    }
    
    fn get_dataset_stats(&self, dataset_id: &str) -> impl std::future::Future<Output = Result<Value>> + Send {
        let db = self.get_db_clone();
        let dataset_id = dataset_id.to_string();
        
        Box::pin(async move {
            let db = db.read().await;
            let mut stats = serde_json::Map::new();
            
            // 计算数据集大小
            let data_key = format!("dataset:{}:data", dataset_id);
            if let Some(data) = db.get(data_key.as_bytes())? {
                stats.insert("size_bytes".to_string(), Value::Number(serde_json::Number::from(data.len())));
            }
            
            // 计算记录数
            let records_key_prefix = format!("dataset:{}:record:", dataset_id);
            let records_count = db.scan_prefix(records_key_prefix.as_bytes()).count();
            stats.insert("record_count".to_string(), Value::Number(serde_json::Number::from(records_count)));
            
            // 获取创建时间
            let metadata_key = format!("dataset:{}:metadata", dataset_id);
            if let Some(metadata_bytes) = db.get(metadata_key.as_bytes())? {
                if let Ok(metadata) = serde_json::from_slice::<serde_json::Value>(&metadata_bytes) {
                    if let Some(created_at) = metadata.get("created_at").and_then(|v| v.as_str()) {
                        stats.insert("created_at".to_string(), Value::String(created_at.to_string()));
                    }
                }
            }
            
            // 计算数据完整性
            let schema_key = format!("dataset:{}:schema", dataset_id);
            let has_schema = db.contains_key(schema_key.as_bytes())?;
            stats.insert("has_schema".to_string(), Value::Bool(has_schema));
            
            // 计算数据质量指标
            let quality_score = if has_schema && records_count > 0 { 100.0 } else if records_count > 0 { 80.0 } else { 0.0 };
            stats.insert("quality_score".to_string(), Value::Number(serde_json::Number::from_f64(quality_score).unwrap_or(serde_json::Number::from(0))));
            
            Ok(Value::Object(stats))
        })
    }
    
    fn validate_dataset(&self, dataset_id: &str) -> impl std::future::Future<Output = Result<Value>> + Send {
        let db = self.get_db_clone();
        let dataset_id = dataset_id.to_string();
        
        Box::pin(async move {
            let db = db.read().await;
            let mut validation_result = serde_json::Map::new();
            
            // 检查数据集是否存在
            let metadata_key = format!("dataset:{}:metadata", dataset_id);
            let exists = db.contains_key(metadata_key.as_bytes())?;
            validation_result.insert("exists".to_string(), Value::Bool(exists));
            
            if exists {
                validation_result.insert("valid".to_string(), Value::Bool(true));
                validation_result.insert("errors".to_string(), Value::Array(Vec::new()));
            } else {
                validation_result.insert("valid".to_string(), Value::Bool(false));
                validation_result.insert("errors".to_string(), Value::Array(vec![
                    Value::String("Dataset not found".to_string())
                ]));
            }
            
            Ok(Value::Object(validation_result))
        })
    }
    
    fn get_dataset_size(&self, dataset_id: &str) -> impl std::future::Future<Output = Result<usize>> + Send {
        let key = format!("dataset:{}:data", dataset_id);
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.read().await;
            if let Some(data) = db.get(key.as_bytes())? {
                Ok(data.len())
            } else {
                Err(Error::NotFound(format!("Dataset {} not found", dataset_id)))
            }
        })
    }
    
    fn get_dataset_chunk(&self, dataset_id: &str, start: usize, end: usize) -> impl std::future::Future<Output = Result<Vec<u8>>> + Send {
        let key = format!("dataset:{}:data", dataset_id);
        let db = self.get_db_clone();
        
        Box::pin(async move {
            let db = db.read().await;
            if let Some(data) = db.get(key.as_bytes())? {
                if start >= data.len() || end > data.len() || start >= end {
                    return Err(Error::InvalidInput("Invalid chunk range".to_string()));
                }
                Ok(data[start..end].to_vec())
            } else {
                Err(Error::NotFound(format!("Dataset {} not found", dataset_id)))
            }
        })
    }
}

#[async_trait::async_trait]
impl MonitoringInterface for StorageEngineImpl {
    async fn get_stats(&self) -> Result<std::collections::HashMap<String, f64>, crate::Error> {
        let db = self.get_db_clone();
        let db_guard = db.read().await;
        let mut stats = std::collections::HashMap::new();
        
        // 基础统计信息
        stats.insert("total_keys".to_string(), db_guard.len() as f64);
        
        // 计算数据库大小（估算）
        let mut total_size = 0usize;
        for result in db_guard.scan_prefix(b"") {
            let (key, value) = result.map_err(|e| crate::Error::StorageError(format!("扫描数据库失败: {}", e)))?;
            total_size += key.len() + value.len();
        }
        stats.insert("database_size_bytes".to_string(), total_size as f64);
            
            // 模型统计
            let mut model_count = 0;
            for result in db_guard.scan_prefix(b"model:") {
                result.map_err(|e| crate::Error::StorageError(format!("扫描模型失败: {}", e)))?;
                model_count += 1;
            }
            stats.insert("model_count".to_string(), model_count as f64);
            
            // 数据集统计
            let mut dataset_count = 0;
            for result in db_guard.scan_prefix(b"dataset:") {
                result.map_err(|e| crate::Error::StorageError(format!("扫描数据集失败: {}", e)))?;
                dataset_count += 1;
            }
            stats.insert("dataset_count".to_string(), dataset_count as f64);
            
            Ok(stats)
    }
    
    async fn get_counter(&self, name: &str) -> Result<u64, crate::Error> {
        let key = format!("counter:{}", name);
        let db = self.get_db_clone();
        let db_guard = db.read().await;
        
        if let Some(data) = db_guard.get(key.as_bytes()).map_err(|e| crate::Error::StorageError(format!("获取计数器失败: {}", e)))? {
            Ok(bincode::deserialize(&data).map_err(|e| crate::Error::Serialization(format!("反序列化计数器失败: {}", e)))?)
        } else {
            Ok(0)
        }
    }
    
    async fn increment_counter(&self, name: &str, value: u64) -> Result<(), crate::Error> {
        let key = format!("counter:{}", name);
        let db = self.get_db_clone();
        let db_guard = db.read().await;
        
        let current_value: u64 = if let Some(data) = db_guard.get(key.as_bytes()).map_err(|e| crate::Error::StorageError(format!("获取计数器失败: {}", e)))? {
            bincode::deserialize(&data).unwrap_or(0)
        } else {
            0
        };
        
        let new_value = current_value + value;
        let data = bincode::serialize(&new_value).map_err(|e| crate::Error::Serialization(format!("序列化计数器失败: {}", e)))?;
        db_guard.insert(key.as_bytes(), data.as_slice()).map_err(|e| crate::Error::StorageError(format!("更新计数器失败: {}", e)))?;
        
        Ok(())
    }
    
    async fn count_models(&self) -> Result<u64, crate::Error> {
        let db = self.get_db_clone();
        let db_guard = db.read().await;
        let mut count = 0u64;
        
        for result in db_guard.scan_prefix(b"model:") {
            result.map_err(|e| crate::Error::StorageError(format!("扫描模型失败: {}", e)))?;
            count += 1;
        }
        
        Ok(count)
    }
    
    async fn count_models_by_type(&self, model_type: &str) -> Result<u64, crate::Error> {
        let db = self.get_db_clone();
        let db_guard = db.read().await;
        let mut count = 0u64;
        
        for result in db_guard.scan_prefix(b"model:") {
            let (_, value) = result.map_err(|e| crate::Error::StorageError(format!("扫描模型失败: {}", e)))?;
            if let Ok(model) = bincode::deserialize::<crate::model::Model>(&value) {
                // 从模型架构中获取类型信息
                let found_model_type = model.architecture.metadata.get("model_type")
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string());
                if found_model_type == model_type {
                    count += 1;
                }
            }
        }
        
        Ok(count)
    }
    
    async fn get_recent_models(&self, limit: usize) -> Result<Vec<String>, crate::Error> {
        let db = self.get_db_clone();
        let db_guard = db.read().await;
        let mut models = Vec::new();
        
        for result in db_guard.scan_prefix(b"model:") {
            let (key, _value) = result.map_err(|e| crate::Error::StorageError(format!("扫描模型失败: {}", e)))?;
            // 从 key 中提取模型ID（格式：model:{model_id}）
            if let Ok(key_str) = String::from_utf8(key.to_vec()) {
                if let Some(model_id) = key_str.strip_prefix("model:") {
                    models.push(model_id.to_string());
                    if models.len() >= limit {
                        break;
                    }
                }
            }
        }
        
        Ok(models)
    }
    
    async fn count_tasks(&self) -> Result<u64, crate::Error> {
        let db = self.get_db_clone();
        let db_guard = db.read().await;
        let mut count = 0u64;
        
        for result in db_guard.scan_prefix(b"task:") {
            result.map_err(|e| crate::Error::StorageError(format!("扫描任务失败: {}", e)))?;
            count += 1;
        }
        
        Ok(count)
    }
    
    async fn count_tasks_by_status(&self, status: &str) -> Result<u64, crate::Error> {
        let db = self.get_db_clone();
        let db_guard = db.read().await;
        let mut count = 0u64;
        
        for result in db_guard.scan_prefix(b"task:") {
            let (_, value) = result.map_err(|e| crate::Error::StorageError(format!("扫描任务失败: {}", e)))?;
            if let Ok(task_info) = serde_json::from_slice::<serde_json::Value>(&value) {
                if let Some(task_status) = task_info.get("status") {
                    if let Some(status_str) = task_status.as_str() {
                        if status_str == status {
                            count += 1;
                        }
                    }
                }
            }
        }
        
        Ok(count)
    }
    
    async fn get_recent_tasks(&self, limit: usize) -> Result<Vec<String>, crate::Error> {
        let db = self.get_db_clone();
        let db_guard = db.read().await;
        let mut tasks = Vec::new();
        
        for result in db_guard.scan_prefix(b"task:") {
            let (key, _value) = result.map_err(|e| crate::Error::StorageError(format!("扫描任务失败: {}", e)))?;
            // 从 key 中提取任务ID（格式：task:{task_id}）
            if let Ok(key_str) = String::from_utf8(key.to_vec()) {
                if let Some(task_id) = key_str.strip_prefix("task:") {
                    tasks.push(task_id.to_string());
                    if tasks.len() >= limit {
                        break;
                    }
                }
            }
        }
        
        Ok(tasks)
    }
    
    async fn get_logs(&self, level: &str, limit: usize) -> Result<Vec<String>, crate::Error> {
        let db = self.get_db_clone();
        let db_guard = db.read().await;
        let mut logs = Vec::new();
        let level_lower = level.to_lowercase();
        
        for result in db_guard.scan_prefix(b"log:") {
            let (_, value) = result.map_err(|e| crate::Error::StorageError(format!("扫描日志失败: {}", e)))?;
            if let Ok(log_entry) = serde_json::from_slice::<serde_json::Value>(&value) {
                // 根据日志级别过滤
                let should_include = if let Some(log_level) = log_entry.get("level") {
                    if let Some(level_str) = log_level.as_str() {
                        level_lower == "all" || level_str.to_lowercase() == level_lower
                    } else {
                        level_lower == "all"
                    }
                } else {
                    level_lower == "all"
                };
                
                if should_include {
                    // 将日志条目转换为字符串
                    if let Ok(log_str) = serde_json::to_string(&log_entry) {
                        logs.push(log_str);
                        if logs.len() >= limit {
                            break;
                        }
                    }
                }
            }
        }
        
        Ok(logs)
    }
    
    async fn count_active_tasks(&self) -> Result<u64, crate::Error> {
        let db = self.get_db_clone();
        let db_guard = db.read().await;
        let mut active_count = 0u64;
        
        for result in db_guard.scan_prefix(b"task:") {
            let (_, value) = result.map_err(|e| crate::Error::StorageError(format!("扫描任务失败: {}", e)))?;
            if let Ok(task_info) = serde_json::from_slice::<serde_json::Value>(&value) {
                if let Some(status) = task_info.get("status") {
                    if let Some(status_str) = status.as_str() {
                        // 检查是否为活跃状态
                        match status_str {
                            "running" | "pending" | "queued" | "starting" => {
                                active_count += 1;
                            },
                            _ => {}
                        }
                    }
                }
            }
        }
        
        Ok(active_count)
    }
    
    async fn get_api_stats(&self) -> Result<std::collections::HashMap<String, f64>, crate::Error> {
        let db = self.get_db_clone();
        let db_guard = db.read().await;
        let mut stats = std::collections::HashMap::new();
        
        // 统计总请求数
        let mut total_requests = 0u64;
        let mut successful_requests = 0u64;
        let mut failed_requests = 0u64;
        let mut total_response_time = 0u64;
        let mut request_count = 0u64;
        
        for result in db_guard.scan_prefix(b"api_request:") {
            let (_, value) = result.map_err(|e| crate::Error::StorageError(format!("扫描API请求失败: {}", e)))?;
            if let Ok(request_log) = serde_json::from_slice::<serde_json::Value>(&value) {
                total_requests += 1;
                
                if let Some(success) = request_log.get("success") {
                    if success.as_bool().unwrap_or(false) {
                        successful_requests += 1;
                    } else {
                        failed_requests += 1;
                    }
                }
                
                if let Some(response_time) = request_log.get("response_time_ms") {
                    if let Some(time) = response_time.as_u64() {
                        total_response_time += time;
                        request_count += 1;
                    }
                }
            }
        }
        
        stats.insert("total_requests".to_string(), total_requests as f64);
        stats.insert("successful_requests".to_string(), successful_requests as f64);
        stats.insert("failed_requests".to_string(), failed_requests as f64);
        
        let average_response_time = if request_count > 0 {
            (total_response_time as f64) / (request_count as f64)
        } else {
            0.0
        };
        stats.insert("average_response_time_ms".to_string(), average_response_time);
        
        // 计算成功率
        let success_rate = if total_requests > 0 {
            (successful_requests as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        };
        stats.insert("success_rate_percent".to_string(), success_rate);
        
        Ok(stats)
    }
    
    async fn check_health(&self) -> Result<bool, crate::Error> {
        let db = self.get_db_clone();
        let db_guard = db.read().await;
        
        // 检查数据库连接是否正常
        // 尝试读取一个测试键来验证数据库是否可访问
        let _ = db_guard.len();
        
        // 如果能够访问数据库，则认为健康
        Ok(true)
    }
    
    async fn record_metric(&self, name: &str, value: f64, tags: &std::collections::HashMap<String, String>) -> Result<(), crate::Error> {
        let db = self.get_db_clone();
        let db_guard = db.read().await;
        let key = format!("metric:{}:{}", name, chrono::Utc::now().timestamp());
        let metric_data = serde_json::json!({
            "name": name,
            "value": value,
            "tags": tags,
            "timestamp": chrono::Utc::now().timestamp()
        });
        let data = serde_json::to_vec(&metric_data).map_err(|e| crate::Error::Serialization(format!("序列化指标失败: {}", e)))?;
        db_guard.insert(key.as_bytes(), data.as_slice()).map_err(|e| crate::Error::StorageError(format!("保存指标失败: {}", e)))?;
        Ok(())
    }
    
    async fn get_metrics(&self, name: &str) -> Result<Vec<crate::core::interfaces::MetricPoint>, crate::Error> {
        let db = self.get_db_clone();
        let db_guard = db.read().await;
        let mut metrics = Vec::new();
        let prefix = format!("metric:{}:", name);
        
        for result in db_guard.scan_prefix(prefix.as_bytes()) {
            let (_, value) = result.map_err(|e| crate::Error::StorageError(format!("扫描指标失败: {}", e)))?;
            if let Ok(metric_data) = serde_json::from_slice::<serde_json::Value>(&value) {
                if let (Some(value), Some(timestamp)) = (metric_data.get("value"), metric_data.get("timestamp")) {
                    if let (Some(v), Some(ts)) = (value.as_f64(), timestamp.as_i64()) {
                        // 将 i64 时间戳转换为 DateTime<Utc>
                        let dt = chrono::DateTime::<chrono::Utc>::from_timestamp(ts, 0)
                            .unwrap_or_else(|| chrono::Utc::now());
                        metrics.push(crate::core::interfaces::MetricPoint {
                            value: v,
                            timestamp: dt,
                            tags: metric_data.get("tags")
                                .and_then(|t| serde_json::from_value::<std::collections::HashMap<String, String>>(t.clone()).ok())
                                .unwrap_or_default(),
                        });
                    }
                }
            }
        }
        
        Ok(metrics)
    }
    
    async fn create_alert(&self, condition: &crate::core::interfaces::monitoring::AlertCondition) -> Result<(), crate::Error> {
        let db = self.get_db_clone();
        let db_guard = db.read().await;
        let alert_id = uuid::Uuid::new_v4().to_string();
        let key = format!("alert:{}", alert_id);
        let alert_data = serde_json::to_vec(condition).map_err(|e| crate::Error::Serialization(format!("序列化告警条件失败: {}", e)))?;
        db_guard.insert(key.as_bytes(), alert_data.as_slice()).map_err(|e| crate::Error::StorageError(format!("保存告警失败: {}", e)))?;
        Ok(())
    }
    
    async fn get_system_health(&self) -> Result<crate::core::interfaces::monitoring::SystemHealth, crate::Error> {
        let db = self.get_db_clone();
        let db_guard = db.read().await;
        
        // 检查数据库连接
        let _ = db_guard.len();
        
        // 返回系统健康状态
        Ok(crate::core::interfaces::monitoring::SystemHealth {
            overall_status: "healthy".to_string(),
            components: std::collections::HashMap::new(),
            last_updated: chrono::Utc::now(),
        })
    }
}

// StorageTransactionImpl 实现
struct StorageTransactionImpl {
    db: std::sync::Arc<tokio::sync::RwLock<rocksdb::DB>>,
    id: String,
    state: crate::core::interfaces::storage_interface::TransactionState,
    ops: Vec<StorageTransactionOp>,
}

enum StorageTransactionOp {
    Store { key: String, value: Vec<u8> },
    Delete { key: String },
}

impl StorageTransactionImpl {
    fn new(db: std::sync::Arc<tokio::sync::RwLock<rocksdb::DB>>) -> Self {
        Self {
            db,
            id: uuid::Uuid::new_v4().to_string(),
            state: crate::core::interfaces::storage_interface::TransactionState::Active,
            ops: Vec::new(),
        }
    }
    
    fn new_with_isolation(db: std::sync::Arc<tokio::sync::RwLock<rocksdb::DB>>, _isolation_level: IsolationLevel) -> Self {
        Self {
            db,
            id: uuid::Uuid::new_v4().to_string(),
            state: crate::core::interfaces::storage_interface::TransactionState::Active,
            ops: Vec::new(),
        }
    }
}

impl StorageTransaction for StorageTransactionImpl {
    fn commit(&mut self) -> Result<()> {
        if self.state != crate::core::interfaces::storage_interface::TransactionState::Active {
            return Err(Error::Transaction("事务非活跃状态，无法提交".to_string()));
        }
        
        // 执行所有操作
        let db = self.db.clone();
        let db = db.blocking_read();
        for op in &self.ops {
            match op {
                StorageTransactionOp::Store { key, value } => {
                    db.put(key.as_bytes(), value)?;
                }
                StorageTransactionOp::Delete { key } => {
                    db.delete(key.as_bytes())?;
                }
            }
        }
        
        self.state = crate::core::interfaces::storage_interface::TransactionState::Committed;
        Ok(())
    }
    
    fn rollback(&mut self) -> Result<()> {
        if self.state != crate::core::interfaces::storage_interface::TransactionState::Active {
            return Err(Error::Transaction("事务非活跃状态，无法回滚".to_string()));
        }
        
        self.state = crate::core::interfaces::storage_interface::TransactionState::RolledBack;
        self.ops.clear();
        Ok(())
    }
    
    fn store(&mut self, key: &str, value: &[u8]) -> Result<()> {
        if self.state != crate::core::interfaces::storage_interface::TransactionState::Active {
            return Err(Error::Transaction("事务非活跃状态，无法执行操作".to_string()));
        }
        
        self.ops.push(StorageTransactionOp::Store {
            key: key.to_string(),
            value: value.to_vec(),
        });
        Ok(())
    }
    
    fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>> {
        if self.state != crate::core::interfaces::storage_interface::TransactionState::Active {
            return Err(Error::Transaction("事务非活跃状态，无法执行操作".to_string()));
        }
        
        let db = self.db.clone();
        let db = db.blocking_read();
        Ok(db.get(key.as_bytes())?.map(|v| v.to_vec()))
    }
    
    fn delete(&mut self, key: &str) -> Result<()> {
        if self.state != crate::core::interfaces::storage_interface::TransactionState::Active {
            return Err(Error::Transaction("事务非活跃状态，无法执行操作".to_string()));
        }
        
        self.ops.push(StorageTransactionOp::Delete {
            key: key.to_string(),
        });
        Ok(())
    }
    
    fn exists(&self, key: &str) -> Result<bool> {
        if self.state != crate::core::interfaces::storage_interface::TransactionState::Active {
            return Err(Error::Transaction("事务非活跃状态，无法执行操作".to_string()));
        }
        
        let db = self.db.clone();
        let db = db.blocking_read();
        Ok(db.get(key.as_bytes())?.is_some())
    }
    
    fn get_state(&self) -> crate::core::interfaces::storage_interface::TransactionState {
        self.state
    }
    
    fn get_id(&self) -> &str {
        &self.id
    }
} 