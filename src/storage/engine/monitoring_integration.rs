use std::sync::{Arc, Mutex};
use log::info;
use tokio::sync::RwLock;

use crate::Result;
use crate::Error;
use crate::storage::config::StorageConfig;
use super::transaction::TransactionManager;
use super::models::ModelStorageService;
use super::datasets::DatasetStorageService;
use super::monitoring::MonitoringStorageService;

/// 监控集成服务
/// 
/// 提供系统健康状态监控、性能指标收集等高级监控功能
#[derive(Clone)]
pub struct MonitoringIntegration {
    config: StorageConfig,
    db: Arc<RwLock<sled::Db>>,
    transaction_manager: Arc<Mutex<TransactionManager>>,
    model_storage: ModelStorageService,
    dataset_storage: DatasetStorageService,
    monitoring_storage: MonitoringStorageService,
}

impl MonitoringIntegration {
    pub fn new(
        config: StorageConfig,
        db: Arc<RwLock<sled::Db>>,
        transaction_manager: Arc<Mutex<TransactionManager>>,
        model_storage: ModelStorageService,
        dataset_storage: DatasetStorageService,
        monitoring_storage: MonitoringStorageService,
    ) -> Self {
        Self {
            config,
            db,
            transaction_manager,
            model_storage,
            dataset_storage,
            monitoring_storage,
        }
    }

    /// 获取系统健康状态
    pub async fn get_system_health(&self) -> Result<serde_json::Value> {
        let mut health = serde_json::Map::new();
        info!("collecting system health metrics");
        
        // 数据库健康状态
        let db_health = self.check_database_health().await?;
        health.insert("database_health".to_string(), serde_json::Value::Object(db_health));
        
        // 存储服务健康状态
        let storage_health = self.check_storage_services_health().await?;
        health.insert("storage_services_health".to_string(), serde_json::Value::Object(storage_health));
        
        // 事务健康状态
        let tx_health = self.check_transaction_health().await?;
        health.insert("transaction_health".to_string(), serde_json::Value::Object(tx_health));
        
        // 计算整体健康评分
        let overall_score = self.calculate_overall_health_score(&health).await?;
        health.insert("overall_health_score".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(overall_score).unwrap()));
        info!("system health score computed: {}", overall_score);
        
        Ok(serde_json::Value::Object(health))
    }

    /// 检查数据库健康状态
    async fn check_database_health(&self) -> Result<serde_json::Map<String, serde_json::Value>> {
        let mut health = serde_json::Map::new();
        
        let db = self.db.read().await;
        let total_keys = db.len();
        let is_empty = db.is_empty();
        
        health.insert("total_keys".to_string(), serde_json::Value::Number(serde_json::Number::from(total_keys)));
        health.insert("is_empty".to_string(), serde_json::Value::Bool(is_empty));
        health.insert("path".to_string(), serde_json::Value::String(self.config.path.to_string_lossy().to_string()));
        
        // 检查数据库大小
        if let Ok(size) = db.size_on_disk() {
            health.insert("size_bytes".to_string(), serde_json::Value::Number(serde_json::Number::from(size)));
        }
        
        Ok(health)
    }

    /// 检查存储服务健康状态
    async fn check_storage_services_health(&self) -> Result<serde_json::Map<String, serde_json::Value>> {
        let mut health = serde_json::Map::new();
        
        // 模型存储服务状态
        let model_count = self.model_storage.count_models()?;
        health.insert("model_storage_models".to_string(), serde_json::Value::Number(serde_json::Number::from(model_count)));
        
        // 数据集存储服务状态
        let dataset_count = self.dataset_storage.list_datasets().await.unwrap_or_default().len();
        health.insert("dataset_storage_datasets".to_string(), serde_json::Value::Number(serde_json::Number::from(dataset_count)));
        
        // 监控存储服务状态
        let monitoring_stats = self.monitoring_storage.get_stats().await.unwrap_or_default();
        health.insert("monitoring_stats".to_string(), monitoring_stats);
        
        Ok(health)
    }

    /// 检查事务健康状态
    async fn check_transaction_health(&self) -> Result<serde_json::Map<String, serde_json::Value>> {
        let mut health = serde_json::Map::new();
        
        // 活跃事务数量
        let active_count = self.get_active_transaction_count()?;
        health.insert("active_transactions".to_string(), serde_json::Value::Number(serde_json::Number::from(active_count)));
        
        // 事务管理器状态
        health.insert("transaction_manager_healthy".to_string(), serde_json::Value::Bool(active_count < 1000));
        
        // 事务清理状态
        let cleaned_count = self.cleanup_transactions()?;
        health.insert("cleaned_transactions".to_string(), serde_json::Value::Number(serde_json::Number::from(cleaned_count)));
        
        Ok(health)
    }

    /// 计算整体健康评分
    async fn calculate_overall_health_score(&self, health_data: &serde_json::Map<String, serde_json::Value>) -> Result<f64> {
        let mut score: f64 = 100.0;
        
        // 数据库健康评分
        if let Some(db_health) = health_data.get("database_health") {
            if let Some(db_map) = db_health.as_object() {
                if let Some(is_empty) = db_map.get("is_empty") {
                    if is_empty.as_bool().unwrap_or(false) {
                        score -= 10.0; // 空数据库扣分
                    }
                }
            }
        }
        
        // 事务健康评分
        if let Some(tx_health) = health_data.get("transaction_health") {
            if let Some(tx_map) = tx_health.as_object() {
                if let Some(active_tx) = tx_map.get("active_transactions") {
                    if let Some(count) = active_tx.as_u64() {
                        if count > 100 {
                            score -= 15.0; // 活跃事务过多扣分
                        }
                    }
                }
            }
        }
        
        // 存储服务健康评分
        if let Some(storage_health) = health_data.get("storage_services_health") {
            if let Some(storage_map) = storage_health.as_object() {
                if let Some(model_count) = storage_map.get("model_storage_models") {
                    if let Some(count) = model_count.as_u64() {
                        if count == 0 {
                            score -= 5.0; // 没有模型扣分
                        }
                    }
                }
            }
        }
        
        Ok(score.max(0.0_f64).min(100.0_f64))
    }

    /// 获取详细性能指标
    pub async fn get_performance_metrics(&self) -> Result<serde_json::Value> {
        let mut metrics = serde_json::Map::new();
        
        // 存储性能指标
        let storage_metrics = self.get_storage_performance_metrics().await?;
        metrics.insert("storage_metrics".to_string(), serde_json::Value::Object(storage_metrics));
        
        // 事务性能指标
        let transaction_metrics = self.get_transaction_performance_metrics().await?;
        metrics.insert("transaction_metrics".to_string(), serde_json::Value::Object(transaction_metrics));
        
        // 服务性能指标
        let service_metrics = self.get_service_performance_metrics().await?;
        metrics.insert("service_metrics".to_string(), serde_json::Value::Object(service_metrics));
        
        Ok(serde_json::Value::Object(metrics))
    }

    /// 获取存储性能指标
    async fn get_storage_performance_metrics(&self) -> Result<serde_json::Map<String, serde_json::Value>> {
        let mut metrics = serde_json::Map::new();
        
        let db = self.db.read().await;
        metrics.insert("total_keys".to_string(), serde_json::Value::Number(serde_json::Number::from(db.len())));
        
        if let Ok(size) = db.size_on_disk() {
            metrics.insert("size_bytes".to_string(), serde_json::Value::Number(serde_json::Number::from(size)));
        }
        
        metrics.insert("is_empty".to_string(), serde_json::Value::Bool(db.is_empty()));
        
        Ok(metrics)
    }

    /// 获取事务性能指标
    async fn get_transaction_performance_metrics(&self) -> Result<serde_json::Map<String, serde_json::Value>> {
        let mut metrics = serde_json::Map::new();
        
        // 活跃事务数量
        let active_count = self.get_active_transaction_count()?;
        metrics.insert("active_transactions".to_string(), serde_json::Value::Number(serde_json::Number::from(active_count)));
        
        // 事务清理数量
        let cleaned_count = self.cleanup_transactions()?;
        metrics.insert("cleaned_transactions".to_string(), serde_json::Value::Number(serde_json::Number::from(cleaned_count)));
        
        Ok(metrics)
    }

    /// 获取服务性能指标
    async fn get_service_performance_metrics(&self) -> Result<serde_json::Map<String, serde_json::Value>> {
        let mut metrics = serde_json::Map::new();
        
        // 模型存储服务指标
        let model_count = self.model_storage.count_models()?;
        metrics.insert("model_count".to_string(), serde_json::Value::Number(serde_json::Number::from(model_count)));
        
        // 数据集存储服务指标
        let dataset_count = self.dataset_storage.list_datasets().await.unwrap_or_default().len();
        metrics.insert("dataset_count".to_string(), serde_json::Value::Number(serde_json::Number::from(dataset_count)));
        
        // 监控存储服务指标
        let monitoring_stats = self.monitoring_storage.get_stats().await.unwrap_or_default();
        metrics.insert("monitoring_stats".to_string(), monitoring_stats);
        
        Ok(metrics)
    }

    /// 获取活跃事务数量
    fn get_active_transaction_count(&self) -> Result<usize> {
        let manager = self.transaction_manager.lock()
            .map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
        Ok(manager.get_active_transaction_count())
    }

    /// 清理过期事务
    fn cleanup_transactions(&self) -> Result<usize> {
        let mut manager = self.transaction_manager.lock()
            .map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
        Ok(manager.cleanup_transactions())
    }
} 