use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use tokio::sync::RwLock;

use crate::Result;
use crate::Error;
use crate::compat::TrainingMetrics;
use crate::data::DataBatch;

use super::transaction::TransactionManager;
use super::types::{DistributedTaskInfo, DistributedTaskStatus, TaskStatus};

/// 分布式任务管理服务
#[derive(Clone)]
pub struct DistributedManager {
    storage: Arc<RwLock<sled::Db>>,
    transaction_manager: Arc<Mutex<TransactionManager>>,
}

impl DistributedManager {
    pub fn new(
        storage: Arc<RwLock<sled::Db>>,
        transaction_manager: Arc<Mutex<TransactionManager>>,
    ) -> Self {
        Self {
            storage,
            transaction_manager,
        }
    }

    /// 保存分布式任务信息
    pub async fn save_distributed_task_info(&self, task_id: &str, task_info: &DistributedTaskInfo) -> Result<()> {
        let key = format!("distributed_task:{}", task_id);
        let data = bincode::serialize(task_info)?;
        self.put_raw(key.as_bytes(), &data).await?;

        // 保存任务元数据
        let metadata = HashMap::from([
            ("task_id".to_string(), task_id.to_string()),
            ("model_id".to_string(), task_info.model_id.clone()),
            ("status".to_string(), format!("{:?}", task_info.status)),
            ("created_at".to_string(), task_info.created_at.to_rfc3339()),
            ("node_count".to_string(), task_info.nodes.len().to_string()),
        ]);

        let metadata_key = format!("distributed_task:{}:metadata", task_id);
        let metadata_data = serde_json::to_vec(&metadata)?;
        self.put_raw(metadata_key.as_bytes(), &metadata_data).await
    }

    /// 获取分布式任务信息
    pub async fn get_distributed_task_info(&self, task_id: &str) -> Result<Option<DistributedTaskInfo>> {
        let key = format!("distributed_task:{}", task_id);
        if let Some(data) = self.get_raw(&key).await? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }

    /// 获取节点任务状态
    pub async fn get_node_task_status(&self, task_id: &str, node_id: &str) -> Result<Option<TaskStatus>> {
        let key = format!("distributed_task:{}:node:{}:status", task_id, node_id);
        if let Some(data) = self.get_raw(&key).await? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }

    /// 更新节点任务状态
    pub async fn update_node_task_status(&self, task_id: &str, node_id: &str, status: TaskStatus) -> Result<()> {
        let key = format!("distributed_task:{}:node:{}:status", task_id, node_id);
        let data = bincode::serialize(&status)?;
        self.put_raw(key.as_bytes(), &data).await?;

        // 更新任务整体状态
        if let Some(mut task_info) = self.get_distributed_task_info(task_id).await? {
            // 检查所有节点状态
            let mut all_completed = true;
            let mut any_failed = false;

            for node in &task_info.nodes {
                if let Some(node_status) = self.get_node_task_status(task_id, &node.node_id).await? {
                    match node_status {
                        TaskStatus::Completed => {}
                        TaskStatus::Failed { .. } => {
                            any_failed = true;
                            break;
                        }
                        _ => {
                            all_completed = false;
                        }
                    }
                } else {
                    all_completed = false;
                }
            }

            // 更新任务状态
            if any_failed {
                task_info.status = DistributedTaskStatus::Failed { error: "One or more nodes failed".to_string() };
            } else if all_completed {
                task_info.status = DistributedTaskStatus::Completed;
            }

            self.save_distributed_task_info(task_id, &task_info).await
        } else {
            Ok(())
        }
    }

    /// 删除分布式任务信息
    pub async fn delete_distributed_task_info(&self, task_id: &str) -> Result<()> {
        let prefixes = vec![
            format!("distributed_task:{}", task_id),
            format!("distributed_task:{}:metadata", task_id),
            format!("distributed_task:{}:node:", task_id),
        ];

        for prefix in prefixes {
            let entries = self.scan_prefix_raw(prefix.as_bytes())?;
            for (key, _) in entries {
                self.delete_raw(&key).await?;
            }
        }

        Ok(())
    }

    /// 检查连接状态
    pub async fn check_connection(&self) -> Result<()> {
        // 简单的连接检查
        let test_key = "connection_test";
        let test_value = b"test";
        
        self.put_raw(test_key.as_bytes(), test_value).await?;
        
        if let Some(value) = self.get_raw(test_key).await? {
            if value == test_value {
                self.delete_raw(test_key.as_bytes()).await?;
                Ok(())
            } else {
                Err(Error::StorageError("Connection test failed".to_string()))
            }
        } else {
            Err(Error::StorageError("Connection test failed".to_string()))
        }
    }

    /// 获取节点任务状态
    pub async fn get_node_task_status_by_node(&self, node_id: &str) -> Result<TaskStatus> {
        let prefix = "distributed_task:";
        let entries = self.scan_prefix_raw(prefix.as_bytes())?;
        
        for (key, value) in entries {
            if let Ok(key_str) = String::from_utf8(key) {
                if key_str.contains(&format!(":node:{}:status", node_id)) {
                    if let Ok(status) = bincode::deserialize::<TaskStatus>(&value) {
                        return Ok(status);
                    }
                }
            }
        }
        
        Err(Error::NotFound(format!("No task status found for node {}", node_id)))
    }

    /// 保存数据分区
    pub async fn save_data_partition(&self, partition_path: &str, partition: &DataBatch) -> Result<()> {
        let key = format!("data_partition:{}", partition_path);
        let data = bincode::serialize(partition)?;
        self.put_raw(key.as_bytes(), &data).await
    }

    /// 加载数据分区
    pub async fn load_data_partition(&self, partition_path: &str) -> Result<Option<DataBatch>> {
        let key = format!("data_partition:{}", partition_path);
        if let Some(data) = self.get_raw(&key).await? {
            Ok(Some(bincode::deserialize(&data)?))
        } else {
            Ok(None)
        }
    }

    /// 获取任务指标
    pub async fn get_task_metrics(&self, task_id: &str) -> Result<HashMap<String, f32>> {
        let key = format!("distributed_task:{}:metrics", task_id);
        if let Some(data) = self.get_raw(&key).await? {
            Ok(serde_json::from_slice(&data)?)
        } else {
            Ok(HashMap::new())
        }
    }

    /// 保存任务指标
    pub async fn save_task_metrics(&self, task_id: &str, metrics: HashMap<String, f32>) -> Result<()> {
        let key = format!("distributed_task:{}:metrics", task_id);
        let data = serde_json::to_vec(&metrics)?;
        self.put_raw(key.as_bytes(), &data).await
    }

    /// 获取训练指标
    pub async fn get_training_metrics(
        &self,
        task_id: &str,
        start_epoch: Option<u32>,
        end_epoch: Option<u32>,
    ) -> Result<(Vec<TrainingMetrics>, Option<u32>, Option<u32>)> {
        let mut metrics = Vec::new();
        let prefix = format!("distributed_task:{}:training_metrics:", task_id);
        
        let entries = self.scan_prefix_raw(prefix.as_bytes())?;
        for (_key, value) in entries {
            if let Ok(metric) = bincode::deserialize::<TrainingMetrics>(&value) {
                let epoch = metric.epoch as u32;
                match (start_epoch, end_epoch) {
                    (Some(s), Some(e)) if epoch >= s && epoch <= e => metrics.push(metric),
                    (Some(s), None) if epoch >= s => metrics.push(metric),
                    (None, Some(e)) if epoch <= e => metrics.push(metric),
                    (None, None) => metrics.push(metric),
                    _ => {}
                }
            }
        }
        
        // 按epoch排序
        metrics.sort_by(|a, b| a.epoch.cmp(&b.epoch));
        
        let total_epochs = if metrics.is_empty() { None } else { Some(metrics.len() as u32) };
        let current_epoch = metrics.last().map(|m| m.epoch as u32);
        Ok((metrics, current_epoch, total_epochs))
    }

    /// 记录模型指标
    pub async fn record_model_metrics(&self, model_id: &str, metrics: TrainingMetrics) -> Result<()> {
        let key = format!("model:{}:metrics:{}", model_id, metrics.epoch);
        let data = bincode::serialize(&metrics)?;
        self.put_raw(key.as_bytes(), &data).await?;

        // 更新训练历史
        let history_key = format!("model:{}:metrics:history", model_id);
        let mut history = if let Some(data) = self.get_raw(&history_key).await? {
            serde_json::from_slice::<Vec<TrainingMetrics>>(&data).unwrap_or_default()
        } else {
            Vec::new()
        };

        history.push(metrics);
        let history_data = serde_json::to_vec(&history)?;
        self.put_raw(history_key.as_bytes(), &history_data).await
    }

    /// 更新训练配置
    pub async fn update_training_config(&self, model_id: &str, config: &crate::training::config::TrainingConfig) -> Result<()> {
        let key = format!("model:{}:training_config", model_id);
        let data = bincode::serialize(config)?;
        self.put_raw(key.as_bytes(), &data).await
    }

    // 训练任务相关方法已移除 - 向量数据库系统不需要训练功能

    /// 列出所有分布式任务
    pub async fn list_distributed_tasks(&self) -> Result<Vec<DistributedTaskInfo>> {
        let mut tasks = Vec::new();
        let prefix = "distributed_task:";
        
        let entries = self.scan_prefix_raw(prefix.as_bytes())?;
        for (key, value) in entries {
            if let Ok(key_str) = String::from_utf8(key) {
                if !key_str.contains(":metadata") && !key_str.contains(":node:") && !key_str.contains(":metrics") {
                    if let Ok(task) = bincode::deserialize::<DistributedTaskInfo>(&value) {
                        tasks.push(task);
                    }
                }
            }
        }
        
        Ok(tasks)
    }

    /// 获取任务统计信息
    pub async fn get_task_statistics(&self) -> Result<HashMap<String, usize>> {
        let mut stats = HashMap::new();
        let tasks = self.list_distributed_tasks().await?;
        
        for task in tasks {
            let count = stats.entry(format!("{:?}", task.status)).or_insert(0);
            *count += 1;
        }
        
        Ok(stats)
    }

    /// 清理过期任务
    pub async fn cleanup_expired_tasks(&self, max_age_hours: u64) -> Result<Vec<String>> {
        let mut expired_tasks = Vec::new();
        let tasks = self.list_distributed_tasks().await?;
        let now = chrono::Utc::now();
        
        for task in tasks {
            let age = now.signed_duration_since(task.created_at);
            if age.num_hours() > max_age_hours as i64 {
                let task_id = task.task_id.clone();
                expired_tasks.push(task_id.clone());
                self.delete_distributed_task_info(&task_id).await?;
            }
        }
        
        Ok(expired_tasks)
    }

    // 私有方法：底层存储操作
    async fn get_raw(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let db = self.storage.read().await;
        Ok(db.get(key.as_bytes())?.map(|v| v.to_vec()))
    }

    async fn put_raw(&self, key: &[u8], value: &[u8]) -> Result<()> {
        let db = self.storage.write().await;
        db.insert(key, value)?;
        Ok(())
    }

    async fn delete_raw(&self, key: &[u8]) -> Result<()> {
        let db = self.storage.write().await;
        db.remove(key)?;
        Ok(())
    }

    fn exists_raw(&self, key: &[u8]) -> Result<bool> {
        let db = self.storage.try_read().map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
        Ok(db.contains_key(key)?)
    }

    fn scan_prefix_raw(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let db = self.storage.try_read().map_err(|e| Error::LockError(format!("锁错误: {}", e)))?;
        let mut results = Vec::new();
        
        for result in db.scan_prefix(prefix) {
            let (key, value) = result?;
            results.push((key.to_vec(), value.to_vec()));
        }
        
        Ok(results)
    }
} 