use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::path::PathBuf;
// use std::collections::BTreeMap;
use serde::{Serialize, Deserialize};
use serde_json::Value;
// use tokio::sync::RwLock;
// use sled::Db;
use bincode;
// use uuid::Uuid;
use log::{warn, debug};
// use chrono::{DateTime, Utc};

use crate::error::{Error, Result};
// use crate::storage::engine::types::{DistributedTaskInfo, TaskStatus};
use crate::storage::config::{StorageConfig, StorageConfigUpdate};
// use crate::storage::{StorageOptions};
// use crate::model::{Model, ModelArchitecture, ModelStatus};
// use crate::model::parameters::{ModelParameters, TrainingState, TrainingStateManager};
// use crate::storage::models::{ModelInfo, ModelMetrics, StorageFormat, ItemInfo, MetadataInfo};
// use crate::core::{TrainingResultDetail, InferenceResultDetail, InferenceResult};
// use crate::data::{DataBatch, ProcessedBatch};

use super::transaction::{TransactionState, TransactionManager};
use super::interfaces::StorageEngine;
// alias to authoritative core interface to avoid local name conflicts
use crate::core::interfaces::StorageInterface as CoreStorageService;
use super::models::ModelStorageService;
use super::datasets::DatasetStorageService;
use super::monitoring::MonitoringStorageService;
use super::types::StorageStatistics;
use super::monitoring_integration::MonitoringIntegration;
use super::dataset_integration::DatasetIntegration;
use super::transaction_manager::TransactionManagerService;
use super::advanced_operations::AdvancedOperationsService;
use super::config_manager::ConfigManagerService;
// use super::model_manager::ModelManager;
use super::algorithm_manager::AlgorithmManager;
use super::distributed_manager::DistributedManager;

/// 存储引擎核心实现
#[derive(Clone)]
pub struct StorageEngineImpl {
    /// 存储配置
    config: StorageConfig,
    /// 底层数据库连接
    db: Arc<tokio::sync::RwLock<sled::Db>>,
    /// 事务管理器
    transaction_manager: Arc<Mutex<TransactionManager>>,
    /// 模型存储服务
    model_storage: ModelStorageService,
    /// 数据集存储服务
    dataset_storage: DatasetStorageService,
    /// 监控存储服务
    monitoring_storage: MonitoringStorageService,
    /// 模型信息缓存
    models: Option<Arc<Mutex<HashMap<String, Vec<u8>>>>>,
    /// 监控集成服务
    monitoring_integration: MonitoringIntegration,
    /// 数据集集成服务
    dataset_integration: DatasetIntegration,
    /// 事务管理器服务
    transaction_manager_service: TransactionManagerService,
    /// 高级操作服务
    advanced_operations_service: AdvancedOperationsService,
    /// 配置管理器服务
    config_manager_service: ConfigManagerService,
    /// 模型管理器
    model_manager: ModelManager,
    /// 算法管理器
    algorithm_manager: AlgorithmManager,
    /// 分布式管理器
    distributed_manager: DistributedManager,
}

impl StorageEngineImpl {
    /// 存储数据
    pub async fn store_data(&self, key: &str, data: &[u8]) -> Result<()> {
        let db = self.db.read().await;
        db.insert(key, data)
            .map_err(|e| Error::StorageError(format!("存储数据失败: {}", e)))?;
        Ok(())
    }

    // 注意：保留异步版本的统一实现，避免重复签名

    /// 存储数据（兼容put接口）
    pub async fn put(&self, key: &[u8], data: &[u8]) -> Result<()> {
        let key_str = std::str::from_utf8(key)
            .map_err(|e| Error::StorageError(format!("键转换失败: {}", e)))?;
        self.store_data(key_str, data).await
    }


    /// 创建新的存储引擎实例
    pub fn new(config: StorageConfig) -> Result<Self> {
        let db = sled::open(&config.path)
            .map_err(|e| Error::StorageError(format!("Failed to open database: {}", e)))?;

        let db_arc = Arc::new(tokio::sync::RwLock::new(db));
        let transaction_manager = Arc::new(Mutex::new(TransactionManager::new(100)));
        let model_storage = ModelStorageService::new(db_arc.clone());
        let dataset_storage = DatasetStorageService::new(db_arc.clone());
        let monitoring_storage = MonitoringStorageService::new(db_arc.clone());
        
        let monitoring_integration = MonitoringIntegration::new(
            config.clone(),
            db_arc.clone(),
            transaction_manager.clone(),
            model_storage.clone(),
            dataset_storage.clone(),
            monitoring_storage.clone(),
        );
        
        let dataset_integration = DatasetIntegration::new(
            config.clone(),
            db_arc.clone(),
            transaction_manager.clone(),
            model_storage.clone(),
            dataset_storage.clone(),
            monitoring_storage.clone(),
        );

        let transaction_manager_service = TransactionManagerService::new(transaction_manager.clone());
        let advanced_operations_service = AdvancedOperationsService::new(db_arc.clone());
        let config_manager_service = ConfigManagerService::new(config.clone(), db_arc.clone());

        let model_manager = ModelManager::new(
            db_arc.clone(),
            transaction_manager.clone(),
            model_storage.clone(),
        );

        let algorithm_manager = AlgorithmManager::new(
            db_arc.clone(),
            transaction_manager.clone(),
        );

        let distributed_manager = DistributedManager::new(
            db_arc.clone(),
            transaction_manager.clone(),
        );

        Ok(Self {
            config,
            db: db_arc,
            transaction_manager,
            model_storage,
            dataset_storage,
            monitoring_storage,
            models: None,
            monitoring_integration,
            dataset_integration,
            transaction_manager_service,
            advanced_operations_service,
            config_manager_service,
            model_manager,
            algorithm_manager,
            distributed_manager,
        })
    }
    
    /// 创建内存存储引擎实例
    pub fn new_in_memory() -> Result<Self> {
        let mut config = StorageConfig::default();
        config.path = PathBuf::from(":memory:");
        Self::new(config)
    }

    /// 获取数据库实例
    pub fn get_db(&self) -> &Arc<tokio::sync::RwLock<sled::Db>> {
        &self.db
    }

    /// 获取数据库连接的克隆
    pub fn get_db_clone(&self) -> Arc<tokio::sync::RwLock<sled::Db>> {
        self.db.clone()
    }

    /// 获取数据库读取锁
    pub async fn get_db_read(&self) -> Result<tokio::sync::RwLockReadGuard<'_, sled::Db>> {
        Ok(self.db.read().await)
    }
    
    /// 获取存储配置
    pub fn get_config(&self) -> &StorageConfig {
        &self.config
    }
    
    /// 获取模型存储服务
    pub fn model_storage(&self) -> &ModelStorageService {
        &self.model_storage
    }
    
    /// 获取数据集存储服务
    pub fn dataset_storage(&self) -> &DatasetStorageService {
        &self.dataset_storage
    }
    
    /// 获取监控存储服务
    pub fn monitoring_storage(&self) -> &MonitoringStorageService {
        &self.monitoring_storage
    }
    
    // ==================== 高级监控集成 ====================

    /// 获取系统健康状态
    pub async fn get_system_health(&self) -> Result<serde_json::Value> {
        let mut health = self.monitoring_integration.get_system_health().await?;

        // Use authoritative core storage interface to derive a simple readiness signal
        // This exercises the trait methods intentionally (avoid unused import and validate wiring)
        let dataset_keys = self.list_keys("dataset:").await.unwrap_or_default();

        // Attach extra diagnostics
        if let Some(obj) = health.as_object_mut() {
            obj.insert(
                "dataset_key_count".to_string(),
                serde_json::json!(dataset_keys.len()),
            );
        }

        Ok(health)
    }



    /// 获取详细性能指标
    pub async fn get_performance_metrics(&self) -> Result<serde_json::Value> {
        self.monitoring_integration.get_performance_metrics().await
    }


    
    // ==================== 高级数据集管理集成 ====================

    /// 获取数据集统计信息
    pub async fn get_dataset_statistics(&self) -> Result<serde_json::Value> {
        self.dataset_integration.get_dataset_statistics().await
    }

    /// 就绪探针：行使核心契约与关键子系统，确保依赖正确接线
    pub async fn readiness_probe(&self) -> Result<()> {
        // 1) 行使核心存储契约：列举数据集相关键，验证基础KV接口
        let _keys = self.list_keys("dataset:").await?;

        // 2) 行使数据集与监控子系统：统计与性能指标
        let _stats = self.get_dataset_statistics().await?;
        let _perf = self.get_performance_metrics().await?;

        // 3) 行使分布式/健康路径的轻量检查（如果实现可能返回Ok）
        let _ = self.distributed_manager.check_connection().await;

        Ok(())
    }

    /// 验证数据集完整性
    pub async fn validate_dataset_integrity(&self, dataset_id: &str) -> Result<serde_json::Value> {
        self.dataset_integration.validate_dataset_integrity(dataset_id).await
    }
    
    // 事务管理方法
    pub fn begin_transaction(&self) -> Result<String> {
        self.transaction_manager_service.begin_transaction()
    }
    
    pub fn transaction_put(&self, transaction_id: &str, key: &[u8], value: &[u8]) -> Result<()> {
        self.transaction_manager_service.transaction_put(transaction_id, key, value)
    }

    pub fn transaction_delete(&self, transaction_id: &str, key: &[u8]) -> Result<()> {
        self.transaction_manager_service.transaction_delete(transaction_id, key)
    }

    pub fn commit_transaction(&self, transaction_id: &str) -> Result<()> {
        self.transaction_manager_service.commit_transaction(transaction_id)
    }
    
    pub fn rollback_transaction(&self, transaction_id: &str) -> Result<()> {
        self.transaction_manager_service.rollback_transaction(transaction_id)
    }
    
    pub fn get_transaction_state(&self, transaction_id: &str) -> Result<TransactionState> {
        self.transaction_manager_service.get_transaction_state(transaction_id)
    }

    pub fn cleanup_transactions(&self) -> Result<usize> {
        self.transaction_manager_service.cleanup_transactions()
    }
    
    pub fn get_active_transaction_count(&self) -> Result<usize> {
        self.transaction_manager_service.get_active_transaction_count()
    }
    
    // 基础存储操作方法
    pub async fn get_raw(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let db = self.db.read().await;
        Ok(db.get(key)?.map(|v| v.to_vec()))
    }

    pub async fn put_raw(&self, key: &[u8], value: &[u8]) -> Result<()> {
        let db = self.db.write().await;
        db.insert(key, value)?;
        Ok(())
    }
    
    pub async fn delete_raw(&self, key: &[u8]) -> Result<()> {
        let db = self.db.write().await;
        db.remove(key)?;
        Ok(())
    }
    
    pub async fn exists_raw(&self, key: &[u8]) -> Result<bool> {
        let db = self.db.read().await;
        Ok(db.contains_key(key)?)
    }

    /// 批量写入操作
    pub fn batch_write(&self, operations: Vec<(Vec<u8>, Option<Vec<u8>>)>) -> Result<()> {
        self.advanced_operations_service.batch_write(operations)
    }

    /// 高级批量写入操作（带事务支持）
    pub async fn advanced_batch_write(&self, operations: Vec<(Vec<u8>, Option<Vec<u8>>)>) -> Result<()> {
        // 开始事务
        let transaction_id = self.transaction_manager_service.begin_transaction()?;
        
        // 在事务中执行批量操作
        for (key, value) in &operations {
            match value {
                Some(val) => {
                    self.transaction_manager_service.transaction_put(&transaction_id, key, val)?;
                },
                None => {
                    self.transaction_manager_service.transaction_delete(&transaction_id, key)?;
                }
            }
        }
        
        // 提交事务
        self.transaction_manager_service.commit_transaction(&transaction_id)?;
        
        debug!("高级批量写入完成，操作数量: {}", operations.len());
        Ok(())
    }
    
    /// 高性能前缀扫描
    pub fn scan_prefix_optimized(&self, prefix: &[u8], limit: Option<usize>) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.advanced_operations_service.scan_prefix_optimized(prefix, limit)
    }

    /// 异步前缀扫描
    pub async fn scan_prefix_async(&self, prefix: &[u8], limit: Option<usize>) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.advanced_operations_service.scan_prefix_async(prefix, limit).await
    }

    /// 高性能刷新操作
    pub fn flush_optimized(&self) -> Result<()> {
        self.advanced_operations_service.flush_optimized()
    }

    /// 异步刷新操作
    pub async fn flush_async(&self) -> Result<()> {
        self.advanced_operations_service.flush_async().await
    }
    
    /// 优雅关闭
    pub async fn graceful_shutdown(&self) -> Result<()> {
        self.advanced_operations_service.graceful_shutdown().await
    }
    
    /// 获取存储引擎状态
    pub async fn get_storage_engine_status(&self) -> Result<serde_json::Value> {
        self.config_manager_service.get_storage_engine_status().await
    }
    
    // 存储统计方法
    pub fn size(&self) -> Result<u64> {
        self.config_manager_service.size()
    }
    
    pub fn len(&self) -> Result<usize> {
        self.config_manager_service.len()
    }
    
    pub fn is_empty(&self) -> Result<bool> {
        self.config_manager_service.is_empty()
    }
    
    pub fn flush(&self) -> Result<()> {
        self.config_manager_service.flush()
    }
    
    pub fn close(&self) -> Result<()> {
        self.config_manager_service.close()
    }
    
    // 高级存储操作方法
    pub async fn store(&self, key: &str, data: &[u8]) -> Result<()> {
        self.put_raw(key.as_bytes(), data).await
    }
    
    pub async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        self.get_raw(key.as_bytes()).await
    }
    
    pub async fn delete(&self, key: &str) -> Result<()> {
        self.delete_raw(key.as_bytes()).await
    }
    
    pub async fn batch_store(&self, items: Vec<(String, Vec<u8>)>) -> Result<()> {
        let operations: Vec<(Vec<u8>, Option<Vec<u8>>)> = items
            .into_iter()
            .map(|(key, value)| (key.into_bytes(), Some(value)))
            .collect();
        
        self.advanced_operations_service.batch_write(operations)
    }
    
    pub async fn exists(&self, key: &str) -> Result<bool> {
        self.exists_raw(key.as_bytes()).await
    }
    
    pub async fn list_keys(&self, prefix: &str) -> Result<Vec<String>> {
        self.advanced_operations_service.list_keys(prefix).await
    }

    pub async fn get_keys_with_prefix(&self, prefix: &str) -> Result<Vec<String>> {
        self.advanced_operations_service.get_keys_with_prefix(prefix).await
    }

    // 配置更新方法
    pub async fn update_config(&self, new_config: &serde_json::Value) -> Result<()> {
        self.config_manager_service.update_config(new_config).await
    }

    pub async fn get_info(&self) -> Result<serde_json::Value> {
        self.config_manager_service.get_info().await
    }

    fn validate_config_update(&self, update: &StorageConfigUpdate) -> Result<()> {
        // 验证配置更新的合法性
        if let Some(write_buffer_size) = update.write_buffer_size {
            if write_buffer_size < 1024 * 1024 { // 最小1MB
                return Err(Error::InvalidInput("Write buffer size must be at least 1MB".to_string()));
            }
        }
        
        if let Some(cache_size_mb) = update.cache_size_mb {
            if cache_size_mb < 1 { // 最小1MB
                return Err(Error::InvalidInput("Cache size must be at least 1MB".to_string()));
            }
        }
        
        Ok(())
    }

    // 统计信息方法
    pub async fn get_statistics(&self) -> Result<StorageStatistics> {
        self.config_manager_service.get_statistics().await
    }

    async fn calculate_database_health_score(&self) -> Result<f64> {
        let mut score = 100.0;
        
        // 检查数据库大小
        let size = self.config_manager_service.size()?;
        if size > 1024 * 1024 * 1024 * 10 { // 超过10GB
            score -= 20.0;
        }
        
        // 检查活跃事务数
        let active_transactions = self.transaction_manager_service.get_active_transaction_count()?;
        if active_transactions > 100 {
            score -= 15.0;
        }
        
        // 检查是否为空
        if self.config_manager_service.is_empty()? {
            score -= 10.0;
        }
        
        // 确保分数在0-100范围内
        let final_score = (score as f64).max(0.0_f64).min(100.0_f64);
        Ok(final_score)
    }

    /// 存储原始数据
    pub async fn put_data(&self, dataset_id: &str, data: &[u8]) -> Result<()> {
        let key = format!("dataset:{}:data", dataset_id);
        self.put_raw(key.as_bytes(), data).await
    }

    /// 获取原始数据
    pub async fn get_data(&self, dataset_id: &str) -> Result<Option<Vec<u8>>> {
        let key = format!("dataset:{}:data", dataset_id);
        self.get_raw(key.as_bytes()).await
    }

    /// 删除原始数据
    pub async fn delete_data(&self, dataset_id: &str) -> Result<()> {
        let key = format!("dataset:{}:data", dataset_id);
        self.delete_raw(key.as_bytes()).await
    }

    /// 存储数据集元数据
    pub async fn put_dataset(&self, dataset_id: &str, dataset: &crate::data::manager::ManagerDataset) -> Result<()> {
        let key = format!("dataset:{}:metadata", dataset_id);
        let value = bincode::serialize(dataset)?;
        self.put_raw(key.as_bytes(), &value).await
    }

    /// 获取数据集元数据
    pub async fn get_dataset(&self, dataset_id: &str) -> Result<Option<crate::data::manager::ManagerDataset>> {
        let key = format!("dataset:{}:metadata", dataset_id);
        if let Some(data) = self.get_raw(key.as_bytes()).await? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }

    /// 删除数据集元数据
    pub async fn delete_dataset(&self, dataset_id: &str) -> Result<()> {
        let key = format!("dataset:{}:metadata", dataset_id);
        self.delete_raw(key.as_bytes()).await
    }

    /// 存储算法
    pub async fn store_algorithm(&self, algorithm_id: &str, algorithm: &crate::algorithm::algorithm::Algorithm) -> Result<()> {
        let key = format!("algorithm:{}", algorithm_id);
        let value = bincode::serialize(algorithm)?;
        self.put_raw(key.as_bytes(), &value).await
    }
    
    /// 加载算法
    pub async fn load_algorithm(&self, algorithm_id: &str) -> Result<Option<crate::algorithm::algorithm::Algorithm>> {
        let key = format!("algorithm:{}", algorithm_id);
        if let Some(data) = self.get_raw(key.as_bytes()).await? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }
    
    /// 删除算法
    pub async fn delete_algorithm(&self, algorithm_id: &str) -> Result<()> {
        let key = format!("algorithm:{}", algorithm_id);
        self.delete_raw(key.as_bytes()).await
    }

    /// 存储统一模型
    pub async fn put_unified_model(&self, model_id: &str, model: &Arc<dyn crate::model::unified::UnifiedModel>) -> Result<()> {
        self.model_storage.put_unified_model(model_id, model).await
    }

    /// 获取统一模型
    pub async fn get_unified_model(&self, model_id: &str) -> Result<Option<Arc<dyn crate::model::unified::UnifiedModel>>> {
        self.model_storage.get_unified_model(model_id).await
    }

    /// 获取模型
    pub async fn get_model(&self, model_id: &str) -> Result<Option<crate::model::Model>> {
        let key = format!("model:{}", model_id);
        if let Some(data) = self.get_raw(key.as_bytes()).await? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }

    /// 保存模型
    pub async fn save_model(&self, model: &crate::model::Model) -> Result<()> {
        let key = format!("model:{}", model.id);
        let value = bincode::serialize(model)?;
        self.put_raw(key.as_bytes(), &value).await
    }

    /// 根据数据集ID获取数据集（别名方法）
    pub fn get_dataset_by_id(&self, dataset_id: &str) -> Result<Option<crate::data::manager::ManagerDataset>> {
        // 使用同步方式获取数据集
        let db = self.db.blocking_read();
        let key = format!("dataset:{}", dataset_id);
        
        match db.get(key.as_bytes()) {
            Ok(Some(value)) => {
                match bincode::deserialize::<crate::data::manager::ManagerDataset>(&value) {
                    Ok(dataset) => Ok(Some(dataset)),
                    Err(e) => Err(Error::DeserializationError(format!("反序列化数据集失败: {}", e))),
                }
            },
            Ok(None) => Ok(None),
            Err(e) => Err(Error::StorageError(format!("获取数据集失败: {}", e))),
        }
    }

    /// 获取模型架构
    pub async fn get_model_architecture(&self, model_id: &str) -> Result<Option<crate::model::ModelArchitecture>> {
        let model_storage = self.model_storage.clone();
        let model_id = model_id.to_string();
        tokio::task::spawn_blocking(move || {
            model_storage.get_model_architecture(&model_id)
        }).await.map_err(|e| Error::Internal(format!("任务执行失败: {}", e)))?
    }

    /// 获取模型信息
    pub async fn get_model_info(&self, model_id: &str) -> Result<Option<crate::storage::models::ModelInfo>> {
        let model_storage = self.model_storage.clone();
        let model_id = model_id.to_string();
        tokio::task::spawn_blocking(move || {
            model_storage.get_model_info(&model_id)
        }).await.map_err(|e| Error::Internal(format!("任务执行失败: {}", e)))?
    }

    /// 获取模型参数
    pub async fn get_model_parameters(&self, model_id: &str) -> Result<Option<crate::model::parameters::ModelParameters>> {
        let model_storage = self.model_storage.clone();
        let model_id = model_id.to_string();
        tokio::task::spawn_blocking(move || {
            model_storage.get_model_parameters(&model_id)
        }).await?
    }

    /// 获取训练指标
    pub async fn get_training_metrics(&self, task_id: &str, start_epoch: Option<u32>, end_epoch: Option<u32>) -> Result<(Vec<crate::training::types::TrainingMetrics>, u32, u32)> {
        // 从分布式管理器获取训练指标
        let (metrics, current_epoch_opt, total_epochs_opt) = self.distributed_manager.get_training_metrics(task_id, start_epoch, end_epoch).await?;
        let current_epoch = current_epoch_opt.unwrap_or(0);
        let total_epochs = total_epochs_opt.unwrap_or(0);
        Ok((metrics, current_epoch, total_epochs))
    }

    /// 记录模型指标
    pub async fn record_model_metrics(&self, model_id: &str, metrics: crate::training::types::TrainingMetrics) -> Result<()> {
        // 直接将训练指标记录到分布式管理器
        self.distributed_manager.record_model_metrics(model_id, metrics).await
    }

    /// 更新训练配置
    pub async fn update_training_config(&self, model_id: &str, config: &crate::training::config::TrainingConfig) -> Result<()> {
        self.distributed_manager.update_training_config(model_id, config).await
    }

    /// 扫描前缀（原始字节）
    pub async fn scan_prefix_raw(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let db = self.db.read().await;
        let mut results = Vec::new();
        
        for result in db.scan_prefix(prefix) {
            match result {
                Ok((key, value)) => {
                    results.push((key.to_vec(), value.to_vec()));
                },
                Err(e) => {
                    warn!("扫描前缀时出错: {}", e);
                    continue;
                }
            }
        }
        
        debug!("扫描前缀完成，前缀: {:?}, 结果数量: {}", prefix, results.len());
        Ok(results)
    }

    /// 列出过滤的算法
    pub async fn list_algorithms_filtered(&self, category: Option<&str>, limit: Option<usize>) -> Result<Vec<crate::algorithm::types::Algorithm>> {
        self.algorithm_manager.list_algorithms_filtered(category, limit).await
    }

    /// 保存训练结果
    pub async fn save_training_result(&self, model_id: &str, result: &HashMap<String, serde_json::Value>) -> Result<()> {
        self.model_manager.save_training_result(model_id, result).await
    }

    /// 保存数据分区
    pub async fn save_data_partition(&self, partition_path: &str, partition: &crate::data::batch::DataBatch) -> Result<()> {
        self.distributed_manager.save_data_partition(partition_path, partition).await
    }

    /// 保存分布式任务信息
    pub async fn save_distributed_task_info(&self, task_id: &str, task_info: &crate::storage::engine::types::DistributedTaskInfo) -> Result<()> {
        self.distributed_manager.save_distributed_task_info(task_id, task_info).await
    }

    /// 获取分布式任务信息
    pub async fn get_distributed_task_info(&self, task_id: &str) -> Result<Option<crate::storage::engine::types::DistributedTaskInfo>> {
        self.distributed_manager.get_distributed_task_info(task_id).await
    }

    /// 获取节点任务状态
    pub async fn get_node_task_status(&self, task_id: &str, node_id: &str) -> Result<Option<crate::storage::engine::types::TaskStatus>> {
        self.distributed_manager.get_node_task_status(task_id, node_id).await
    }

    /// 获取模型参数
    pub async fn save_model_parameters(&self, model_id: &str, parameters: &crate::model::parameters::ModelParameters) -> Result<()> {
        self.model_manager.save_model_parameters(model_id, parameters).await
    }

    /// 获取训练历史
    pub async fn get_training_history(&self, model_id: &str) -> Result<Option<Vec<HashMap<String, serde_json::Value>>>> {
        self.model_manager.get_training_history(model_id).await
    }

    /// 保存训练历史
    pub async fn save_training_history(&self, model_id: &str, history: &[HashMap<String, serde_json::Value>]) -> Result<()> {
        self.model_manager.save_training_history(model_id, history).await
    }

    /// 获取模型元数据
    pub async fn get_model_metadata(&self, model_id: &str) -> Result<Option<HashMap<String, String>>> {
        self.model_manager.get_model_metadata(model_id).await
    }

    /// 保存模型元数据
    pub async fn save_model_metadata(&self, model_id: &str, metadata: &HashMap<String, String>) -> Result<()> {
        self.model_manager.save_model_metadata(model_id, metadata).await
    }

    /// 检查模型是否存在
    pub async fn model_exists(&self, model_id: &str) -> Result<bool> {
        self.model_manager.model_exists(model_id).await
    }

    /// 获取模型架构（JSON格式）
    pub async fn get_model_architecture_json(&self, model_id: &str) -> Result<Option<HashMap<String, serde_json::Value>>> {
        // 获取模型架构并转换为JSON格式
        if let Some(architecture) = self.model_manager.get_model_architecture(model_id).await? {
            let json_value = serde_json::to_value(architecture)?;
            if let serde_json::Value::Object(map) = json_value {
                // 将Map转换为HashMap
                let hash_map: HashMap<String, serde_json::Value> = map.into_iter().collect();
                Ok(Some(hash_map))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// 获取模型训练配置
    pub async fn get_model_training_config(&self, model_id: &str) -> Result<Option<HashMap<String, serde_json::Value>>> {
        self.model_manager.get_model_training_config(model_id).await
    }

    /// 检查连接状态
    pub async fn check_connection(&self) -> Result<()> {
        self.distributed_manager.check_connection().await
    }

    /// 获取数据集ID列表
    pub async fn get_dataset_ids(&self) -> Result<Vec<String>> {
        self.dataset_storage.get_dataset_ids().await
    }

    /// 获取数据集特征统计
    pub async fn get_dataset_feature_stats(&self, id: &str) -> Result<Value> {
        self.dataset_storage.get_dataset_feature_stats(id).await
    }

    /// 获取模型数据
    pub async fn get_model_data(&self, model_id: &str) -> Result<Option<serde_json::Value>> {
        self.model_manager.get_model_data(model_id).await
    }

    /// 保存模型训练数据
    pub async fn save_model_training_data(&self, model_id: &str, data: &crate::data::ProcessedBatch) -> Result<()> {
        self.model_manager.save_model_training_data(model_id, data).await
    }

    /// 列出带过滤条件的模型
    pub async fn list_models_with_filters(&self, filters: HashMap<String, String>, limit: usize, offset: usize) -> Result<Vec<crate::model::Model>> {
        self.model_manager.list_models_with_filters(filters, limit, offset).await
    }

    /// 删除模型
    pub async fn delete_model(&self, model_id: &str) -> Result<()> {
        self.model_manager.delete_model(model_id).await
    }

    /// 获取模型指标历史
    pub async fn get_model_metrics_history(&self, model_id: &str) -> Result<Option<Vec<crate::training::types::TrainingMetrics>>> {
        self.model_manager.get_model_metrics_history(model_id).await
    }

    /// 记录训练指标
    pub async fn record_training_metrics(&self, model_id: &str, metrics: &crate::training::types::TrainingMetrics) -> Result<()> {
        self.model_manager.record_training_metrics(model_id, metrics).await
    }

    /// 创建新模型
    pub async fn create_model(&self, model_id: &str, architecture: &crate::model::ModelArchitecture) -> Result<()> {
        // 创建新的模型对象
        let model = crate::model::Model {
            id: model_id.to_string(),
            name: model_id.to_string(), // 使用ID作为默认名称
            description: None,
            version: "1.0.0".to_string(),
            model_type: "custom".to_string(),
            smart_parameters: crate::model::SmartModelParameters::default(),
            architecture: architecture.clone(),
            status: crate::model::ModelStatus::Created,
            metrics: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            parent_id: None,
            metadata: HashMap::new(),
            input_shape: architecture.input_shape.clone(),
            output_shape: architecture.output_shape.clone(),
            import_source: None,
            memory_monitor: Arc::new(Mutex::new(crate::model::ModelMemoryMonitor::new())),
        };
        
        self.model_storage.save_model(model_id, &model)
    }

    /// 获取模型指标
    pub async fn get_model_metrics(&self, model_id: &str) -> Result<Option<crate::storage::models::ModelMetrics>> {
        let model_storage = self.model_storage.clone();
        let model_id = model_id.to_string();
        let result = tokio::task::spawn_blocking(move || {
            model_storage.get_model_metrics(&model_id)
        }).await;
        
        match result {
            Ok(inner_result) => inner_result,
            Err(e) => Err(Error::Internal(format!("任务执行失败: {}", e))),
        }
    }

    /// 保存模型指标
    pub async fn save_model_metrics(&self, model_id: &str, metrics: &crate::storage::models::ModelMetrics) -> Result<()> {
        let model_storage = self.model_storage.clone();
        let model_id = model_id.to_string();
        let metrics = metrics.clone();
        let result = tokio::task::spawn_blocking(move || {
            model_storage.save_model_metrics(&model_id, &metrics)
        }).await;
        
        match result {
            Ok(inner_result) => inner_result,
            Err(e) => Err(Error::Internal(format!("任务执行失败: {}", e))),
        }
    }

    /// 获取任务指标
    pub async fn get_task_metrics(&self, task_id: &str) -> Result<Option<HashMap<String, f32>>> {
        let key = format!("task:{}:metrics", task_id);
        let db = self.db.read().await;
        if let Some(data) = db.get(key.as_bytes())? {
            let metrics: HashMap<String, f32> = bincode::deserialize(data.as_ref())?;
            Ok(Some(metrics))
        } else {
            Ok(None)
        }
    }

    /// 删除训练任务
    pub async fn delete_training_task(&self, task_id: &str) -> Result<()> {
        let db = self.db.write().await;
        
        // 删除任务相关的所有数据
        let task_prefix = format!("task:{}", task_id);
        let task_keys: Vec<_> = db.scan_prefix(task_prefix.as_bytes())
            .filter_map(|result| result.ok())
            .collect();
        
        for (key, _) in task_keys {
            db.remove(&key)?;
        }
        
        Ok(())
    }

    /// 更新模型访问时间
    pub async fn update_model_access_time(&self, model_id: &str, access_time: chrono::DateTime<chrono::Utc>) -> Result<()> {
        self.model_manager.update_model_access_time(model_id, access_time).await
    }

    /// 检查模型是否已训练
    pub async fn is_model_trained(&self, model_id: &str) -> Result<bool> {
        self.model_manager.is_model_trained(model_id).await
    }

    /// 检查模型是否可部署
    pub async fn is_model_deployable(&self, model_id: &str) -> Result<bool> {
        self.model_manager.is_model_deployable(model_id).await
    }

    /// 列出模型
    pub async fn list_models(&self, filters: HashMap<String, String>, limit: usize) -> Result<Vec<crate::model::Model>> {
        self.model_manager.list_models(filters, limit).await
    }

    /// 导出模型
    pub async fn export_model(&self, model_id: &str, format: &str, options: Option<HashMap<String, String>>) -> Result<crate::storage::engine::types::ExportInfo> {
        self.model_manager.export_model(model_id, format, options).await
    }

    /// 部署模型
    pub async fn deploy_model(&self, model_id: &str, options: HashMap<String, String>) -> Result<crate::storage::engine::types::DeploymentInfo> {
        self.model_manager.deploy_model(model_id, options).await
    }

    /// 创建数据批次
    pub async fn create_data_batch(&self, model_id: &str, config: &crate::data::DataConfig) -> Result<String> {
        let batch_id = uuid::Uuid::new_v4().to_string();
        let key = format!("data_batch:{}:{}", model_id, batch_id);
        let batch_info = serde_json::json!({
            "batch_id": batch_id,
            "model_id": model_id,
            "config": config,
            "created_at": chrono::Utc::now().timestamp(),
        });
        let value = serde_json::to_vec(&batch_info)?;
        self.put_raw(key.as_bytes(), &value).await?;
        Ok(batch_id)
    }

    /// 列出模型版本
    pub async fn list_model_versions(&self, model_id: &str) -> Result<Vec<crate::storage::engine::types::ModelVersionInfo>> {
        let key_prefix = format!("model_version:{}", model_id);
        let mut versions = Vec::new();
        
        let db = self.db.read().await;
        for result in db.scan_prefix(key_prefix.as_bytes()) {
            let (_key, value) = result?;
            if let Ok(version_info) = serde_json::from_slice::<crate::storage::engine::types::ModelVersionInfo>(value.as_ref()) {
                versions.push(version_info);
            }
        }
        
        Ok(versions)
    }
    
    /// 获取操作计数
    pub fn get_operation_count(&self) -> u64 {
        // 从数据库中读取操作计数器
        let db = self.db.blocking_read();
        if let Ok(Some(data)) = db.get(b"operation_counter") {
            if let Ok(count) = bincode::deserialize::<u64>(&data) {
                return count;
            }
        }
        0
    }
    
    /// 重新配置数据库
    pub fn reconfigure_database(&self) -> Result<()> {
        let db = self.db.blocking_read();
        
        // 应用新的配置选项
        let mut opts = sled::Config::default();
        opts.path = self.config.path.clone();
        
        // 设置缓存大小
        opts.cache_capacity = self.config.cache_size as u64;
        
        // 设置压缩（使用 use_compression 字段）
        if self.config.use_compression {
            // sled 可能不支持直接的 Compression 枚举，使用配置方式
            // opts.compression = Some(sled::Compression::Lz4);
        }
        
        // 记录配置更新
        let config_update = serde_json::json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "cache_size": self.config.cache_size,
            "use_compression": self.config.use_compression
        });
        
        let config_data = serde_json::to_vec(&config_update)?;
        db.insert(b"config_update", config_data)?;
        
        Ok(())
    }

    /// 获取存储引擎指标
    pub async fn get_metrics(&self) -> Result<StorageMetrics> {
        let db = self.db.read().await;
        
        // 计算总对象数
        let total_objects = db.len() as u64;
        
        // 计算总大小（字节）
        let total_size_bytes = total_objects * 100; // 估算平均对象大小
        
        // 获取读写操作统计（暂时设为0，后续可以实现）
        let read_operations = 0u64;
        let write_operations = 0u64;
        
        // 计算平均对象大小
        let avg_object_size = if total_objects > 0 {
            total_size_bytes / total_objects
        } else {
            0
        };
        
        Ok(StorageMetrics {
            total_objects,
            total_size_bytes,
            read_operations,
            write_operations,
            avg_object_size,
            cache_hit_rate: 0.0, // 暂时设为0，后续可以实现
            compression_ratio: 1.0, // 暂时设为1.0，后续可以实现
        })
    }
}

/// 存储引擎指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageMetrics {
    /// 总对象数
    pub total_objects: u64,
    /// 总大小（字节）
    pub total_size_bytes: u64,
    /// 读操作数
    pub read_operations: u64,
    /// 写操作数
    pub write_operations: u64,
    /// 平均对象大小
    pub avg_object_size: u64,
    /// 缓存命中率
    pub cache_hit_rate: f64,
    /// 压缩比
    pub compression_ratio: f64,
}

// 类型别名
pub type Storage = StorageEngineImpl;

// 为StorageEngineImpl实现StorageInterface trait
/// 存储事务包装器
struct StorageTransactionWrapper {
    transaction_id: String,
    engine: StorageEngineImpl,
}

impl crate::core::interfaces::StorageTransaction for StorageTransactionWrapper {
    fn commit(&mut self) -> Result<()> {
        let mut manager = self.engine.transaction_manager.lock()
            .map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
        
        // 获取事务并检查状态
        let transaction = manager.get_transaction(&self.transaction_id)
            .ok_or_else(|| Error::Transaction(format!("事务 {} 不存在", self.transaction_id)))?;
        
        // 检查事务状态
        if transaction.state != super::transaction::TransactionState::Active {
            return Err(Error::Transaction(format!(
                "事务 {} 不处于活跃状态，无法提交。当前状态: {:?}",
                self.transaction_id, transaction.state
            )));
        }
        
        // 先执行所有操作，确保原子性
        let operations = transaction.operations.clone();
        drop(manager); // 释放锁，避免在执行操作时持有锁
        
        // 执行操作，如果任何操作失败，整个事务回滚
        let mut executed_operations = Vec::new();
        for operation in operations {
            match operation {
                super::transaction::TransactionOperation::Put { ref key, ref value } => {
                    // 执行 Put 操作
                    tokio::runtime::Handle::try_current()
                        .map(|handle| handle.block_on(self.engine.put(key, value)))
                        .unwrap_or_else(|_| {
                            tokio::runtime::Runtime::new()
                                .map_err(|e| Error::StorageError(format!("创建运行时失败: {}", e)))?
                                .block_on(self.engine.put(key, value))
                        })
                        .map_err(|e| {
                            // 如果操作失败，回滚已执行的操作
                            for executed_op in executed_operations.iter().rev() {
                                match executed_op {
                                    super::transaction::TransactionOperation::Put { key, .. } => {
                                        let _ = tokio::runtime::Handle::try_current()
                                            .map(|handle| handle.block_on(self.engine.delete(
                                                std::str::from_utf8(key).unwrap_or("")
                                            )));
                                    }
                                    super::transaction::TransactionOperation::Delete { key } => {
                                        // 对于删除操作，无法完全回滚，只能记录错误
                                        log::warn!("无法回滚删除操作: {:?}", key);
                                    }
                                }
                            }
                            Error::Transaction(format!("执行事务操作失败: {}", e))
                        })?;
                    executed_operations.push(operation);
                }
                super::transaction::TransactionOperation::Delete { ref key } => {
                    let key_str = std::str::from_utf8(key)
                        .map_err(|e| Error::StorageError(format!("键转换失败: {}", e)))?;
                    // 执行 Delete 操作
                    tokio::runtime::Handle::try_current()
                        .map(|handle| handle.block_on(self.engine.delete(key_str)))
                        .unwrap_or_else(|_| {
                            tokio::runtime::Runtime::new()
                                .map_err(|e| Error::StorageError(format!("创建运行时失败: {}", e)))?
                                .block_on(self.engine.delete(key_str))
                        })
                        .map_err(|e| {
                            // 如果操作失败，回滚已执行的操作
                            for executed_op in executed_operations.iter().rev() {
                                match executed_op {
                                    super::transaction::TransactionOperation::Put { key, .. } => {
                                        let _ = tokio::runtime::Handle::try_current()
                                            .map(|handle| handle.block_on(self.engine.delete(
                                                std::str::from_utf8(key).unwrap_or("")
                                            )));
                                    }
                                    super::transaction::TransactionOperation::Delete { .. } => {
                                        // 对于删除操作，无法完全回滚
                                        log::warn!("无法回滚删除操作");
                                    }
                                }
                            }
                            Error::Transaction(format!("执行事务操作失败: {}", e))
                        })?;
                    executed_operations.push(operation);
                }
            }
        }
        
        // 所有操作成功执行后，标记事务为已提交
        let mut manager = self.engine.transaction_manager.lock()
            .map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
        manager.commit_transaction(&self.transaction_id)?;
        
        Ok(())
    }
    
    fn rollback(&mut self) -> Result<()> {
        self.engine.transaction_manager.lock()
            .map_err(|e| Error::LockError(format!("锁错误: {}", e)))?
            .rollback_transaction(&self.transaction_id)
    }
    
    fn store(&mut self, key: &str, value: &[u8]) -> Result<()> {
        // 将操作添加到事务中，而不是直接执行
        let mut manager = self.engine.transaction_manager.lock()
            .map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
        
        let transaction = manager.get_transaction_mut(&self.transaction_id)
            .ok_or_else(|| Error::Transaction(format!("事务 {} 不存在", self.transaction_id)))?;
        
        transaction.add_operation(super::transaction::TransactionOperation::put(
            key.as_bytes().to_vec(),
            value.to_vec(),
        ))?;
        
        Ok(())
    }
    
    fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>> {
        // 先检查事务中是否有该键的操作
        let manager = self.engine.transaction_manager.lock()
            .map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
        
        if let Some(transaction) = manager.get_transaction(&self.transaction_id) {
            // 查找该键的最后操作
            if let Some(operation) = transaction.get_last_operation_for_key(key.as_bytes()) {
                match operation {
                    super::transaction::TransactionOperation::Put { value, .. } => {
                        return Ok(Some(value.clone()));
                    }
                    super::transaction::TransactionOperation::Delete { .. } => {
                        return Ok(None);
                    }
                }
            }
        }
        
        drop(manager);
        
        // 如果事务中没有该键的操作，从数据库读取
        tokio::runtime::Handle::try_current()
            .map(|handle| handle.block_on(self.engine.get(key)))
            .unwrap_or_else(|_| {
                tokio::runtime::Runtime::new()
                    .map_err(|e| Error::StorageError(format!("创建运行时失败: {}", e)))?
                    .block_on(self.engine.get(key))
            })
    }
    
    fn delete(&mut self, key: &str) -> Result<()> {
        // 将删除操作添加到事务中
        let mut manager = self.engine.transaction_manager.lock()
            .map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
        
        let transaction = manager.get_transaction_mut(&self.transaction_id)
            .ok_or_else(|| Error::Transaction(format!("事务 {} 不存在", self.transaction_id)))?;
        
        transaction.add_operation(super::transaction::TransactionOperation::delete(
            key.as_bytes().to_vec(),
        ))?;
        
        Ok(())
    }
    
    fn exists(&self, key: &str) -> Result<bool> {
        self.retrieve(key).map(|v| v.is_some())
    }
    
    fn get_state(&self) -> crate::core::interfaces::TransactionState {
        self.engine.transaction_manager.lock()
            .ok()
            .and_then(|manager| {
                manager.get_transaction(&self.transaction_id)
                    .map(|t| match t.state {
                        super::transaction::TransactionState::Active => crate::core::interfaces::TransactionState::Active,
                        super::transaction::TransactionState::Committed => crate::core::interfaces::TransactionState::Committed,
                        super::transaction::TransactionState::Aborted => crate::core::interfaces::TransactionState::RolledBack,
                    })
            })
            .unwrap_or(crate::core::interfaces::TransactionState::RolledBack)
    }
    
    fn get_id(&self) -> &str {
        &self.transaction_id
    }
}

impl crate::core::interfaces::StorageInterface for StorageEngineImpl {
    fn transaction(&self) -> Result<Box<dyn crate::core::interfaces::StorageTransaction>> {
        // 创建事务
        let transaction_id = self.transaction_manager.lock().unwrap().begin_transaction()?;
        // 返回一个包装的事务对象
        Ok(Box::new(StorageTransactionWrapper {
            transaction_id,
            engine: self.clone(),
        }))
    }
    
    fn transaction_with_isolation(&self, _isolation_level: crate::core::interfaces::IsolationLevel) -> Result<Box<dyn crate::core::interfaces::StorageTransaction>> {
        // 暂时使用默认事务，隔离级别功能待实现
        self.transaction()
    }
    
    fn store(&self, key: &str, value: &[u8]) -> Result<()> {
        tokio::runtime::Handle::try_current()
            .map(|handle| handle.block_on(self.put(key.as_bytes(), value)))
            .unwrap_or_else(|_| {
                // 如果没有运行时，创建一个新的
                tokio::runtime::Runtime::new()
                    .map_err(|e| Error::StorageError(format!("创建运行时失败: {}", e)))?
                    .block_on(self.put(key.as_bytes(), value))
            })
    }
    
    fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>> {
        tokio::runtime::Handle::try_current()
            .map(|handle| handle.block_on(self.get(key)))
            .unwrap_or_else(|_| {
                tokio::runtime::Runtime::new()
                    .map_err(|e| Error::StorageError(format!("创建运行时失败: {}", e)))?
                    .block_on(self.get(key))
            })
    }
    
    fn delete(&self, key: &str) -> Result<()> {
        tokio::runtime::Handle::try_current()
            .map(|handle| handle.block_on(self.delete(key)))
            .unwrap_or_else(|_| {
                tokio::runtime::Runtime::new()
                    .map_err(|e| Error::StorageError(format!("创建运行时失败: {}", e)))?
                    .block_on(self.delete(key))
            })
    }
    
    fn exists(&self, key: &str) -> Result<bool> {
        self.retrieve(key).map(|v| v.is_some())
    }
    
    fn list_keys(&self, prefix: &str) -> Result<Vec<String>> {
        tokio::runtime::Handle::try_current()
            .map(|handle| handle.block_on(self.get_keys_with_prefix(prefix)))
            .unwrap_or_else(|_| {
                tokio::runtime::Runtime::new()
                    .map_err(|e| Error::StorageError(format!("创建运行时失败: {}", e)))?
                    .block_on(self.get_keys_with_prefix(prefix))
            })
    }
    
    fn check_disk_space(&self) -> Result<crate::core::interfaces::DiskSpaceInfo> {
        use std::fs;
        use std::path::Path;
        
        let path = &self.config.path;
        
        // 获取路径的父目录或路径本身
        let check_path = if path.is_dir() {
            path.as_path()
        } else if let Some(parent) = path.parent() {
            parent
        } else {
            Path::new(".")
        };
        
        // 使用平台特定的磁盘空间检查
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let metadata = fs::metadata(check_path)
                .map_err(|e| Error::StorageError(format!("获取路径元数据失败: {}", e)))?;
            
            // 在 Unix 系统上，可以使用 statvfs
            // 这里使用简化实现，实际应该使用 libc::statvfs
            // 为了跨平台，我们使用一个通用的方法
            let total_space = 1024 * 1024 * 1024 * 100; // 100GB 默认值
            let used_space = metadata.len();
            let available_space: u64 = total_space.saturating_sub(used_space as u64);
            let usage_percentage = if total_space > 0 {
                (used_space as f64 / total_space as f64 * 100.0) as f32
            } else {
                0.0
            };
            
            Ok(crate::core::interfaces::DiskSpaceInfo {
                total_space,
                used_space,
                available_space,
                usage_percentage,
            })
        }
        
        #[cfg(windows)]
        {
            let metadata = fs::metadata(check_path)
                .map_err(|e| Error::StorageError(format!("获取路径元数据失败: {}", e)))?;
            
            // Windows 上可以使用 GetDiskFreeSpaceEx
            // 这里使用简化实现
            let total_space: u64 = 1024 * 1024 * 1024 * 100; // 100GB 默认值
            let used_space: u64 = metadata.len();
            let available_space: u64 = total_space.saturating_sub(used_space);
            let usage_percentage = if total_space > 0 {
                (used_space as f64 / total_space as f64 * 100.0) as f32
            } else {
                0.0
            };
            
            Ok(crate::core::interfaces::DiskSpaceInfo {
                total_space,
                used_space,
                available_space,
                usage_percentage,
            })
        }
        
        #[cfg(not(any(unix, windows)))]
        {
            // 其他平台使用简化实现
            let metadata = fs::metadata(check_path)
                .map_err(|e| Error::StorageError(format!("获取路径元数据失败: {}", e)))?;
            
            let total_space = 1024 * 1024 * 1024 * 100; // 100GB 默认值
            let used_space = metadata.len();
            let available_space: u64 = total_space.saturating_sub(used_space as u64);
            let usage_percentage = if total_space > 0 {
                (used_space as f64 / total_space as f64 * 100.0) as f32
            } else {
                0.0
            };
            
            Ok(crate::core::interfaces::DiskSpaceInfo {
                total_space,
                used_space,
                available_space,
                usage_percentage,
            })
        }
    }
    
    fn get_active_connections_count(&self) -> Result<usize> {
        // 返回活跃事务数作为连接数的近似值
        // 在实际实现中，应该维护一个连接池并跟踪活跃连接
        let manager = self.transaction_manager.lock()
            .map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
        Ok(manager.get_active_transaction_count())
    }
}