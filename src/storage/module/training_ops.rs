// 训练操作模块
// 实现训练相关方法

use std::sync::Arc;
use std::collections::HashMap;
use log::warn;
use bincode;
use serde_json;

use crate::{Error, Result};
use super::core::Storage;

impl Storage {
    /// 获取任务指标
    pub async fn get_task_metrics(self: &Arc<Self>, task_id: &str) -> Result<Option<HashMap<String, f32>>> {
        let storage = Arc::clone(self);
        let task_id = task_id.to_string(); // 拥有所有权，可以在闭包中使用
        tokio::task::spawn_blocking(move || {
            let key = format!("task:{}:metrics", task_id);
            match storage.db().get(key.as_bytes())
                .map_err(|e| Error::Storage(format!("获取任务指标失败: {}", e)))? {
                Some(data) => {
                    let metrics: HashMap<String, f32> = bincode::deserialize(&data)
                        .map_err(|e| Error::Serialization(format!("反序列化任务指标失败: {}", e)))?;
                    Ok::<Option<HashMap<String, f32>>, Error>(Some(metrics))
                }
                None => Ok(None)
            }
        }).await.map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }

    /// 删除训练任务
    pub async fn delete_training_task(self: &Arc<Self>, task_id: &str) -> Result<()> {
        let storage = Arc::clone(self);
        let task_id = task_id.to_string();
        tokio::task::spawn_blocking(move || {
            // 删除任务相关的所有数据
            let mut deleted_count = 0;
            
            // 删除任务主键
            let task_key = format!("training_task:{}", task_id);
                match storage.db().delete(task_key.as_bytes()) {
                Ok(_) => deleted_count += 1,
                Err(e) => {
                    if !e.to_string().contains("NotFound") {
                        return Err(Error::Storage(format!("删除训练任务键失败: {}", e)));
                    }
                }
            }
            
            // 删除任务指标
            let metrics_key = format!("task:{}:metrics", task_id);
                match storage.db().delete(metrics_key.as_bytes()) {
                Ok(_) => deleted_count += 1,
                Err(e) => {
                    if !e.to_string().contains("NotFound") {
                        return Err(Error::Storage(format!("删除任务指标键失败: {}", e)));
                    }
                }
            }
            
            // 删除任务相关的所有训练指标
            // 由于RocksDB不支持通配符查询，我们使用迭代器扫描
            let iter = storage.db().iterator(rocksdb::IteratorMode::Start);
            for item in iter {
                match item {
                    Ok((key, _)) => {
                        let key_str = String::from_utf8_lossy(&key);
                        if key_str.contains(&format!("task:{}", task_id)) {
                            if let Err(e) = storage.db().delete(&key) {
                                warn!("删除任务相关键失败: {} - {}", key_str, e);
                            } else {
                                deleted_count += 1;
                            }
                        }
                    }
                    Err(e) => {
                        warn!("迭代任务键时出错: {}", e);
                    }
                }
            }
            
            // 确保至少删除了主任务键
            if deleted_count == 0 {
                return Err(Error::NotFound(format!("训练任务 {} 不存在", task_id)));
            }
            
            Ok::<(), Error>(())
        }).await.map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }

    /// 获取训练指标
    pub async fn get_training_metrics(self: &Arc<Self>, task_id: &str, start_epoch: Option<u32>, end_epoch: Option<u32>) -> Result<(Vec<crate::training::types::TrainingMetrics>, u32, u32)> {
        let storage = Arc::clone(self);
        let task_id = task_id.to_string(); // 用于过滤特定任务的指标
        tokio::task::spawn_blocking(move || {
            // 从训练历史中获取指标
            let mut all_metrics = Vec::new();
            let mut current_epoch = 0u32;
            let mut total_epochs = 0u32;
            
            // 首先尝试从训练历史中获取（更高效的方式）
            let history_prefix = format!("model:");
            let iter = storage.db().iterator(rocksdb::IteratorMode::From(history_prefix.as_bytes(), rocksdb::Direction::Forward));
            for item in iter {
                match item {
                    Ok((key, value)) => {
                        let key_str = String::from_utf8_lossy(&key);
                        // 检查是否是训练指标键（格式：model:{model_id}:metrics:{epoch} 或 model:{model_id}:metrics:history）
                        // 如果提供了task_id，尝试从task键中查找对应的model_id来过滤
                        if key_str.starts_with("model:") && key_str.contains(":metrics:") {
                            // 如果提供了task_id，尝试通过task键查找对应的model_id来过滤指标
                            if !task_id.is_empty() && !key_str.contains(&task_id) {
                                // 尝试通过task键查找对应的model_id
                                let task_key = format!("training_task:{}", task_id);
                                if let Ok(Some(task_data)) = storage.db().get(task_key.as_bytes()) {
                                    if let Ok(task_info) = serde_json::from_slice::<serde_json::Value>(&task_data) {
                                        if let Some(model_id) = task_info.get("model_id").and_then(|v| v.as_str()) {
                                            // 只处理匹配model_id的指标
                                            if !key_str.contains(model_id) {
                                                continue; // 跳过不匹配的指标
                                            }
                                        } else {
                                            continue; // 无法获取model_id，跳过
                                        }
                                    } else {
                                        continue; // 无法解析task信息，跳过
                                    }
                                } else {
                                    // 如果找不到task键，尝试直接匹配task_id（可能task_id就是model_id）
                                    continue; // 跳过不匹配的指标
                                }
                            }
                            // 尝试从历史记录中获取
                            if key_str.ends_with(":history") {
                                if let Ok(history) = serde_json::from_slice::<Vec<crate::training::types::TrainingMetrics>>(&value) {
                                    for metrics in history {
                                        // 应用epoch过滤
                                        let epoch = metrics.epoch as u32;
                                        let include = match (start_epoch, end_epoch) {
                                            (Some(start), Some(end)) => epoch >= start && epoch <= end,
                                            (Some(start), None) => epoch >= start,
                                            (None, Some(end)) => epoch <= end,
                                            (None, None) => true,
                                        };
                                        
                                        if include {
                                            all_metrics.push(metrics.clone());
                                            if epoch > current_epoch {
                                                current_epoch = epoch;
                                            }
                                            if metrics.total_epochs > total_epochs as usize {
                                                total_epochs = metrics.total_epochs as u32;
                                            }
                                        }
                                    }
                                }
                            } else if let Ok(metrics) = bincode::deserialize::<crate::training::types::TrainingMetrics>(&value) {
                                // 应用epoch过滤
                                let epoch = metrics.epoch as u32;
                                let include = match (start_epoch, end_epoch) {
                                    (Some(start), Some(end)) => epoch >= start && epoch <= end,
                                    (Some(start), None) => epoch >= start,
                                    (None, Some(end)) => epoch <= end,
                                    (None, None) => true,
                                };
                                
                                if include {
                                    all_metrics.push(metrics.clone());
                                    if epoch > current_epoch {
                                        current_epoch = epoch;
                                    }
                                    if metrics.total_epochs > total_epochs as usize {
                                        total_epochs = metrics.total_epochs as u32;
                                    }
                                }
                            }
                        } else if !key_str.starts_with("model:") {
                            // 如果不再以model:开头，停止迭代
                            break;
                        }
                    }
                    Err(e) => {
                        warn!("迭代训练指标时出错: {}", e);
                    }
                }
            }
            
            // 按epoch排序
            all_metrics.sort_by_key(|m| m.epoch);
            
            Ok::<(Vec<crate::training::types::TrainingMetrics>, u32, u32), Error>((all_metrics, current_epoch, total_epochs))
        }).await.map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }

    /// 记录模型指标
    pub async fn record_model_metrics(self: &Arc<Self>, model_id: &str, metrics: crate::training::types::TrainingMetrics) -> Result<()> {
        let storage = Arc::clone(self);
        let model_id = model_id.to_string();
        let metrics_clone = metrics.clone();
        tokio::task::spawn_blocking(move || {
            // 存储单个epoch的指标
            let key = format!("model:{}:metrics:{}", model_id, metrics_clone.epoch);
            let data = bincode::serialize(&metrics_clone)
                .map_err(|e| Error::Serialization(format!("序列化训练指标失败: {}", e)))?;
            storage.db().put(key.as_bytes(), &data)
                .map_err(|e| Error::Storage(format!("保存训练指标失败: {}", e)))?;

            // 更新训练历史
            let history_key = format!("model:{}:metrics:history", model_id);
            let mut history = match storage.db().get(history_key.as_bytes())
                .map_err(|e| Error::Storage(format!("获取训练历史失败: {}", e)))? {
                Some(data) => {
                    match serde_json::from_slice::<Vec<crate::training::types::TrainingMetrics>>(&data) {
                        Ok(hist) => hist,
                        Err(e) => {
                            warn!("反序列化训练历史失败，使用空历史: {}", e);
                            Vec::new()
                        }
                    }
                }
                None => Vec::new()
            };

            history.push(metrics_clone);
            let history_data = serde_json::to_vec(&history)
                .map_err(|e| Error::Serialization(format!("序列化训练历史失败: {}", e)))?;
            storage.db().put(history_key.as_bytes(), &history_data)
                .map_err(|e| Error::Storage(format!("保存训练历史失败: {}", e)))?;
            
            Ok::<(), Error>(())
        }).await.map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }

    /// 更新训练配置
    pub async fn update_training_config(self: &Arc<Self>, model_id: &str, config: &crate::training::config::TrainingConfig) -> Result<()> {
        let storage = Arc::clone(self);
        let model_id = model_id.to_string();
        let config_clone = config.clone();
        tokio::task::spawn_blocking(move || {
            let key = format!("model:{}:training_config", model_id);
            let data = bincode::serialize(&config_clone)
                .map_err(|e| Error::Serialization(format!("序列化训练配置失败: {}", e)))?;
            storage.db().put(key.as_bytes(), &data)
                .map_err(|e| Error::Storage(format!("保存训练配置失败: {}", e)))?;
            Ok::<(), Error>(())
        }).await.map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }

    /// 保存训练结果
    pub async fn save_training_result(self: &Arc<Self>, model_id: &str, result: &HashMap<String, serde_json::Value>) -> Result<()> {
        let storage = Arc::clone(self);
        let model_id = model_id.to_string();
        let result_data = serde_json::to_vec(result)
            .map_err(|e| Error::Serialization(format!("序列化训练结果失败: {}", e)))?;
        tokio::task::spawn_blocking(move || {
            let key = format!("training_result:{}", model_id);
            storage.put(key.as_bytes(), &result_data)
        }).await.map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }
    
    /// 保存数据分区
    pub async fn save_data_partition(self: &Arc<Self>, partition_path: &str, partition: &crate::data::DataBatch) -> Result<()> {
        let storage = Arc::clone(self);
        let partition_path = partition_path.to_string();
        let partition_data = bincode::serialize(partition)
            .map_err(|e| Error::Serialization(format!("序列化数据分区失败: {}", e)))?;
        tokio::task::spawn_blocking(move || {
            let key = format!("data_partition:{}", partition_path);
            storage.put(key.as_bytes(), &partition_data)
        }).await.map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }
    
    /// 保存分布式任务信息
    pub async fn save_distributed_task_info(self: &Arc<Self>, task_id: &str, info: &crate::storage::engine::types::DistributedTaskInfo) -> Result<()> {
        let storage = Arc::clone(self);
        let task_id = task_id.to_string();
        let info_data = bincode::serialize(info)
            .map_err(|e| Error::Serialization(format!("序列化分布式任务信息失败: {}", e)))?;
        tokio::task::spawn_blocking(move || {
            let key = format!("distributed_task:{}", task_id);
            storage.put(key.as_bytes(), &info_data)
        }).await.map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }
    
    /// 获取分布式任务信息
    pub async fn get_distributed_task_info(self: &Arc<Self>, task_id: &str) -> Result<Option<crate::storage::engine::types::DistributedTaskInfo>> {
        let storage = Arc::clone(self);
        let task_id = task_id.to_string();
        tokio::task::spawn_blocking(move || {
            let key = format!("distributed_task:{}", task_id);
            match storage.get(key.as_bytes())? {
                Some(data) => {
                    let info: crate::storage::engine::types::DistributedTaskInfo = bincode::deserialize(&data)
                        .map_err(|e| Error::Serialization(format!("反序列化分布式任务信息失败: {}", e)))?;
                    Ok(Some(info))
                }
                None => Ok(None)
            }
        }).await.map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }
    
    /// 获取节点任务状态
    pub async fn get_node_task_status(self: &Arc<Self>, task_id: &str, node_id: &str) -> Result<Option<crate::storage::engine::types::TaskStatus>> {
        let storage = Arc::clone(self);
        let task_id = task_id.to_string();
        let node_id = node_id.to_string();
        tokio::task::spawn_blocking(move || {
            let key = format!("node_task_status:{}:{}", task_id, node_id);
            match storage.get(key.as_bytes())? {
                Some(data) => {
                    let status: crate::storage::engine::types::TaskStatus = bincode::deserialize(&data)
                        .map_err(|e| Error::Serialization(format!("反序列化节点任务状态失败: {}", e)))?;
                    Ok(Some(status))
                }
                None => Ok(None)
            }
        }).await.map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }
    
    /// 获取训练历史
    pub async fn get_training_history(self: &Arc<Self>, model_id: &str) -> Result<Option<Vec<HashMap<String, serde_json::Value>>>> {
        let storage = Arc::clone(self);
        let model_id = model_id.to_string();
        tokio::task::spawn_blocking(move || {
            let key = format!("training_history:{}", model_id);
            match storage.get(key.as_bytes())? {
                Some(data) => {
                    let history: Vec<HashMap<String, serde_json::Value>> = bincode::deserialize(&data)
                        .map_err(|e| Error::Serialization(format!("反序列化训练历史失败: {}", e)))?;
                    Ok(Some(history))
                }
                None => Ok(None)
            }
        }).await.map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }
    
    /// 保存训练历史
    pub async fn save_training_history(self: &Arc<Self>, model_id: &str, history: &Vec<HashMap<String, serde_json::Value>>) -> Result<()> {
        let storage = Arc::clone(self);
        let model_id = model_id.to_string();
        let history_data = bincode::serialize(history)
            .map_err(|e| Error::Serialization(format!("序列化训练历史失败: {}", e)))?;
        tokio::task::spawn_blocking(move || {
            let key = format!("training_history:{}", model_id);
            storage.put(key.as_bytes(), &history_data)
        }).await.map_err(|e| Error::Internal(format!("Task join error: {}", e)))?
    }
}

