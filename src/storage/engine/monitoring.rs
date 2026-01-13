use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
// serde derives not used directly here
use serde_json::Value;
use log::{debug, info, warn};

use crate::Result;
use crate::Error;
use crate::compat::TrainingMetrics;
use crate::storage::engine::types::DistributedTaskInfo;

/// 监控存储服务
/// 
/// 提供系统监控、指标统计、健康检查等功能的存储和检索
#[derive(Clone)]
pub struct MonitoringStorageService {
    /// 底层数据库连接
    db: Arc<RwLock<sled::Db>>,
}

impl MonitoringStorageService {
    /// 创建新的监控存储服务
    /// 
    /// # 参数
    /// - `db`: Sled数据库实例
    pub fn new(db: Arc<RwLock<sled::Db>>) -> Self {
        Self { db }
    }
    
    /// 获取训练指标历史
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 异步返回训练指标历史列表
    pub async fn get_training_metrics_history(
        &self,
        model_id: &str,
    ) -> Result<Vec<TrainingMetrics>> {
        let prefix = format!("model:{}:training_metrics:", model_id);
        let db = self.db.read().await;
        
        let mut metrics_list = Vec::new();
        for item in db.scan_prefix(prefix.as_bytes()) {
            match item {
                Ok((_, value)) => {
                    if let Ok(metrics) = bincode::deserialize::<TrainingMetrics>(&value) {
                        metrics_list.push(metrics);
                    }
                },
                Err(e) => {
                    warn!("扫描训练指标历史时出错: {}", e);
                }
            }
        }
        
        // 按时间戳排序
        metrics_list.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        
        Ok(metrics_list)
    }
    
    /// 记录训练指标
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `metrics`: 训练指标对象
    pub async fn record_training_metrics(
        &self,
        model_id: &str,
        metrics: &TrainingMetrics,
    ) -> Result<()> {
        let key = format!("model:{}:training_metrics:{}", model_id, metrics.timestamp);
        let value = bincode::serialize(metrics)
            .map_err(|e| Error::serialization(format!("序列化训练指标失败: {}", e)))?;
        
        let db = self.db.read().await;
        db.insert(key.as_bytes(), value)
            .map_err(|e| Error::storage(format!("记录训练指标失败: {}", e)))?;
        
        debug!("已记录模型 {} 的训练指标，时间戳: {}", model_id, metrics.timestamp);
        Ok(())
    }
    
    /// 保存分布式任务信息
    /// 
    /// # 参数
    /// - `task_id`: 任务唯一标识符
    /// - `info`: 分布式任务信息
    pub async fn save_distributed_task_info(&self, task_id: &str, info: &DistributedTaskInfo) -> Result<()> {
        let key = format!("distributed_task:{}", task_id);
        let value = bincode::serialize(info)
            .map_err(|e| Error::serialization(format!("序列化分布式任务信息失败: {}", e)))?;
        
        let db = self.db.read().await;
        db.insert(key.as_bytes(), value)
            .map_err(|e| Error::storage(format!("保存分布式任务信息失败: {}", e)))?;
        
        info!("已保存分布式任务 {} 的信息", task_id);
        Ok(())
    }
    
    /// 获取分布式任务信息
    /// 
    /// # 参数
    /// - `task_id`: 任务唯一标识符
    /// 
    /// # 返回值
    /// 异步返回分布式任务信息，如果不存在则返回None
    pub async fn get_distributed_task_info(&self, task_id: &str) -> Result<Option<DistributedTaskInfo>> {
        let key = format!("distributed_task:{}", task_id);
        let db = self.db.read().await;
        
        if let Some(value) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取分布式任务信息失败: {}", e)))? {
            
            let info = bincode::deserialize(&value)
                .map_err(|e| Error::serialization(format!("反序列化分布式任务信息失败: {}", e)))?;
            Ok(Some(info))
        } else {
            Ok(None)
        }
    }
    
    /// 获取节点任务状态
    /// 
    /// # 参数
    /// - `task_id`: 任务唯一标识符
    /// 
    /// # 返回值
    /// 异步返回任务状态，如果不存在则返回None
    pub async fn get_node_task_status(&self, task_id: &str) -> Result<Option<TaskStatus>> {
        let key = format!("node_task_status:{}", task_id);
        let db = self.db.read().await;
        
        if let Some(value) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取节点任务状态失败: {}", e)))? {
            
            let status = bincode::deserialize(&value)
                .map_err(|e| Error::serialization(format!("反序列化节点任务状态失败: {}", e)))?;
            Ok(Some(status))
        } else {
            Ok(None)
        }
    }
    
    /// 保存节点任务状态
    /// 
    /// # 参数
    /// - `task_id`: 任务唯一标识符
    /// - `status`: 任务状态
    pub async fn save_node_task_status(&self, task_id: &str, status: &TaskStatus) -> Result<()> {
        let key = format!("node_task_status:{}", task_id);
        let value = bincode::serialize(status)
            .map_err(|e| Error::serialization(format!("序列化节点任务状态失败: {}", e)))?;
        
        let db = self.db.read().await;
        db.insert(key.as_bytes(), value)
            .map_err(|e| Error::storage(format!("保存节点任务状态失败: {}", e)))?;
        
        debug!("已保存任务 {} 的节点状态", task_id);
        Ok(())
    }
    
    /// 获取系统统计信息
    /// 
    /// # 返回值
    /// 异步返回系统统计信息JSON对象
    pub async fn get_stats(&self) -> Result<Value> {
        let db = self.db.read().await;
        
        let mut stats = serde_json::Map::new();
        
        // 计算各种统计信息
        let mut model_count = 0;
        let mut dataset_count = 0;
        let mut task_count = 0;
        let mut training_count = 0;
        
        for item in db.iter() {
            if let Ok((key, _)) = item {
                let key_str = String::from_utf8_lossy(&key);
                
                if key_str.starts_with("model:") && !key_str.contains(':') {
                    model_count += 1;
                } else if key_str.starts_with("dataset:") && key_str.ends_with(":metadata") {
                    dataset_count += 1;
                } else if key_str.starts_with("distributed_task:") {
                    task_count += 1;
                } else if key_str.contains(":training_metrics:") {
                    training_count += 1;
                }
            }
        }
        
        stats.insert("model_count".to_string(), Value::Number(model_count.into()));
        stats.insert("dataset_count".to_string(), Value::Number(dataset_count.into()));
        stats.insert("task_count".to_string(), Value::Number(task_count.into()));
        stats.insert("training_records_count".to_string(), Value::Number(training_count.into()));
        
        // 添加数据库统计信息
        let db_size = db.size_on_disk().unwrap_or(0);
        stats.insert("database_size_bytes".to_string(), Value::Number(db_size.into()));
        
        let db_len = db.len();
        stats.insert("total_keys".to_string(), Value::Number(db_len.into()));
        
        // 添加时间戳
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        stats.insert("timestamp".to_string(), Value::Number(timestamp.into()));
        
        Ok(Value::Object(stats))
    }
    
    /// 获取计数器值
    /// 
    /// # 参数
    /// - `counter_name`: 计数器名称
    /// 
    /// # 返回值
    /// 异步返回计数器值
    pub async fn get_counter(&self, counter_name: &str) -> Result<u64> {
        let key = format!("counter:{}", counter_name);
        let db = self.db.read().await;
        
        if let Some(value) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取计数器失败: {}", e)))? {
            
            let counter: u64 = bincode::deserialize(&value)
                .map_err(|e| Error::serialization(format!("反序列化计数器失败: {}", e)))?;
            Ok(counter)
        } else {
            Ok(0)
        }
    }
    
    /// 递增计数器
    /// 
    /// # 参数
    /// - `counter_name`: 计数器名称
    /// 
    /// # 返回值
    /// 异步返回递增后的计数器值
    pub async fn increment_counter(&self, counter_name: &str) -> Result<u64> {
        let key = format!("counter:{}", counter_name);
        let db = self.db.read().await;
        
        let current_value = if let Some(value) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取计数器失败: {}", e)))? {
            
            bincode::deserialize::<u64>(&value)
                .map_err(|e| Error::serialization(format!("反序列化计数器失败: {}", e)))?
        } else {
            0
        };
        
        let new_value = current_value + 1;
        let serialized = bincode::serialize(&new_value)
            .map_err(|e| Error::serialization(format!("序列化计数器失败: {}", e)))?;
        
        db.insert(key.as_bytes(), serialized)
            .map_err(|e| Error::storage(format!("保存计数器失败: {}", e)))?;
        
        debug!("计数器 {} 递增到 {}", counter_name, new_value);
        Ok(new_value)
    }
    
    /// 统计模型数量
    /// 
    /// # 返回值
    /// 异步返回模型总数
    pub async fn count_models(&self) -> Result<usize> {
        let db = self.db.read().await;
        
        let mut count = 0;
        for item in db.scan_prefix("model:".as_bytes()) {
            if let Ok((key, _)) = item {
                let key_str = String::from_utf8_lossy(&key);
                // 只计算主模型键，不计算子键（如参数、架构等）
                if !key_str[6..].contains(':') {
                    count += 1;
                }
            }
        }
        
        Ok(count)
    }
    
    /// 按类型统计模型数量
    /// 
    /// # 返回值
    /// 异步返回按模型类型分组的数量映射
    pub async fn count_models_by_type(&self) -> Result<HashMap<String, usize>> {
        let db = self.db.read().await;
        
        let mut type_counts = HashMap::new();
        
        for item in db.scan_prefix("model:".as_bytes()) {
            if let Ok((key, value)) = item {
                let key_str = String::from_utf8_lossy(&key);
                if key_str.ends_with(":info") {
                    if let Ok(info) = serde_json::from_slice::<Value>(&value) {
                        if let Some(model_type) = info.get("type").and_then(|v| v.as_str()) {
                            *type_counts.entry(model_type.to_string()).or_insert(0) += 1;
                        } else {
                            *type_counts.entry("unknown".to_string()).or_insert(0) += 1;
                        }
                    }
                }
            }
        }
        
        Ok(type_counts)
    }
    
    /// 获取最近的模型
    /// 
    /// # 参数
    /// - `limit`: 限制返回数量
    /// 
    /// # 返回值
    /// 异步返回最近创建的模型信息列表
    pub async fn get_recent_models(&self, limit: usize) -> Result<Vec<Value>> {
        let db = self.db.read().await;
        
        let mut models = Vec::new();
        
        for item in db.scan_prefix("model:".as_bytes()) {
            if let Ok((key, value)) = item {
                let key_str = String::from_utf8_lossy(&key);
                if key_str.ends_with(":info") {
                    if let Ok(info) = serde_json::from_slice::<Value>(&value) {
                        models.push(info);
                    }
                }
            }
        }
        
        // 按创建时间排序（如果有的话）
        models.sort_by(|a, b| {
            let a_time = a.get("created_at").and_then(|v| v.as_u64()).unwrap_or(0);
            let b_time = b.get("created_at").and_then(|v| v.as_u64()).unwrap_or(0);
            b_time.cmp(&a_time) // 降序排列，最新的在前
        });
        
        models.truncate(limit);
        Ok(models)
    }
    
    /// 统计任务数量
    /// 
    /// # 返回值
    /// 异步返回任务总数
    pub async fn count_tasks(&self) -> Result<usize> {
        let db = self.db.read().await;
        
        let mut count = 0;
        for item in db.scan_prefix("distributed_task:".as_bytes()) {
            if item.is_ok() {
                count += 1;
            }
        }
        
        Ok(count)
    }
    
    /// 按状态统计任务数量
    /// 
    /// # 返回值
    /// 异步返回按任务状态分组的数量映射
    pub async fn count_tasks_by_status(&self) -> Result<HashMap<String, usize>> {
        let db = self.db.read().await;
        
        let mut status_counts = HashMap::new();
        
        for item in db.scan_prefix("node_task_status:".as_bytes()) {
            if let Ok((_, value)) = item {
                if let Ok(status) = bincode::deserialize::<TaskStatus>(&value) {
                    let status_str = format!("{:?}", status);
                    *status_counts.entry(status_str).or_insert(0) += 1;
                }
            }
        }
        
        Ok(status_counts)
    }
    
    /// 获取最近的任务
    /// 
    /// # 参数
    /// - `limit`: 限制返回数量
    /// 
    /// # 返回值
    /// 异步返回最近创建的任务信息列表
    pub async fn get_recent_tasks(&self, limit: usize) -> Result<Vec<Value>> {
        let db = self.db.read().await;
        
        let mut tasks = Vec::new();
        
        for item in db.scan_prefix("distributed_task:".as_bytes()) {
            if let Ok((_, value)) = item {
                if let Ok(task_info) = bincode::deserialize::<DistributedTaskInfo>(&value) {
                    // 将任务信息转换为JSON
                    if let Ok(task_json) = serde_json::to_value(&task_info) {
                        tasks.push(task_json);
                    }
                }
            }
        }
        
        // 按创建时间排序（如果有的话）
        tasks.sort_by(|a, b| {
            let a_time = a.get("created_at").and_then(|v| v.as_u64()).unwrap_or(0);
            let b_time = b.get("created_at").and_then(|v| v.as_u64()).unwrap_or(0);
            b_time.cmp(&a_time) // 降序排列，最新的在前
        });
        
        tasks.truncate(limit);
        Ok(tasks)
    }
    
    /// 获取日志
    /// 
    /// # 参数
    /// - `level`: 日志级别
    /// - `limit`: 限制返回数量
    /// 
    /// # 返回值
    /// 异步返回日志条目列表
    pub async fn get_logs(&self, level: &str, limit: usize) -> Result<Vec<Value>> {
        let db = self.db.read().await;
        
        let mut logs = Vec::new();
        let prefix = if level == "all" {
            "log:".to_string()
        } else {
            format!("log:{}:", level)
        };
        
        for item in db.scan_prefix(prefix.as_bytes()) {
            if let Ok((_, value)) = item {
                if let Ok(log_entry) = serde_json::from_slice::<Value>(&value) {
                    logs.push(log_entry);
                    if logs.len() >= limit {
                        break;
                    }
                }
            }
        }
        
        // 按时间戳排序（如果有的话）
        logs.sort_by(|a, b| {
            let a_time = a.get("timestamp").and_then(|v| v.as_u64()).unwrap_or(0);
            let b_time = b.get("timestamp").and_then(|v| v.as_u64()).unwrap_or(0);
            b_time.cmp(&a_time) // 降序排列，最新的在前
        });
        
        Ok(logs)
    }
    
    /// 统计活跃任务数量
    /// 
    /// # 返回值
    /// 异步返回活跃任务数量
    pub async fn count_active_tasks(&self) -> Result<usize> {
        let db = self.db.read().await;
        
        let mut count = 0;
        for item in db.scan_prefix("node_task_status:".as_bytes()) {
            if let Ok((_, value)) = item {
                if let Ok(status) = bincode::deserialize::<TaskStatus>(&value) {
                    if matches!(status, TaskStatus::Running { .. } | TaskStatus::Pending) {
                        count += 1;
                    }
                }
            }
        }
        
        Ok(count)
    }
    
    /// 获取API统计信息
    /// 
    /// # 返回值
    /// 异步返回API统计信息JSON对象
    pub async fn get_api_stats(&self) -> Result<Value> {
        let db = self.db.read().await;
        
        let mut stats = serde_json::Map::new();
        
        // 统计API调用次数
        let mut api_calls = HashMap::new();
        for item in db.scan_prefix("api_call:".as_bytes()) {
            if let Ok((key, _)) = item {
                let key_str = String::from_utf8_lossy(&key);
                if let Some(endpoint) = key_str.strip_prefix("api_call:") {
                    let endpoint = endpoint.split(':').next().unwrap_or("unknown");
                    *api_calls.entry(endpoint.to_string()).or_insert(0) += 1;
                }
            }
        }
        
        stats.insert("api_calls".to_string(), serde_json::to_value(api_calls).unwrap());
        
        // 统计错误次数
        let error_count = self.get_counter("api_errors").await.unwrap_or(0);
        stats.insert("error_count".to_string(), Value::Number(error_count.into()));
        
        // 统计请求总数
        let total_requests = self.get_counter("total_requests").await.unwrap_or(0);
        stats.insert("total_requests".to_string(), Value::Number(total_requests.into()));
        
        // 计算成功率
        let success_rate = if total_requests > 0 {
            (total_requests - error_count) as f64 / total_requests as f64 * 100.0
        } else {
            100.0
        };
        stats.insert("success_rate_percent".to_string(), 
                    Value::Number(serde_json::Number::from_f64(success_rate)
                        .unwrap_or_else(|| serde_json::Number::from(0))));
        
        Ok(Value::Object(stats))
    }
    
    /// 统计活跃事务数量
    /// 
    /// # 返回值
    /// 异步返回活跃事务数量
    pub async fn count_active_transactions(&self) -> Result<usize> {
        let db = self.db.read().await;
        
        let mut count = 0;
        for item in db.scan_prefix("transaction:".as_bytes()) {
            if let Ok((key, value)) = item {
                let key_str = String::from_utf8_lossy(&key);
                if key_str.ends_with(":status") {
                    if let Ok(status) = String::from_utf8(value.to_vec()) {
                        if status == "active" {
                            count += 1;
                        }
                    }
                }
            }
        }
        
        Ok(count)
    }
    
    /// 获取分析报告
    /// 
    /// # 参数
    /// - `limit`: 限制返回数量
    /// 
    /// # 返回值
    /// 异步返回分析报告列表
    pub async fn get_analysis_reports(&self, limit: Option<usize>) -> Result<Vec<Value>> {
        let db = self.db.read().await;
        
        let mut reports = Vec::new();
        let limit = limit.unwrap_or(10);
        
        for item in db.scan_prefix("analysis_report:".as_bytes()) {
            if let Ok((_, value)) = item {
                if let Ok(report) = serde_json::from_slice::<Value>(&value) {
                    reports.push(report);
                    if reports.len() >= limit {
                        break;
                    }
                }
            }
        }
        
        // 按创建时间排序
        reports.sort_by(|a, b| {
            let a_time = a.get("created_at").and_then(|v| v.as_u64()).unwrap_or(0);
            let b_time = b.get("created_at").and_then(|v| v.as_u64()).unwrap_or(0);
            b_time.cmp(&a_time) // 降序排列，最新的在前
        });
        
        Ok(reports)
    }
    
    /// 检查系统健康状态
    /// 
    /// # 返回值
    /// 异步返回健康检查结果JSON对象
    pub async fn check_health(&self) -> Result<Value> {
        let db = self.db.read().await;
        
        let mut health = serde_json::Map::new();
        
        // 检查数据库连接
        health.insert("database_connected".to_string(), Value::Bool(true));
        
        // 检查数据库大小
        let db_size = db.size_on_disk().unwrap_or(0);
        health.insert("database_size_bytes".to_string(), Value::Number(db_size.into()));
        
        // 检查键数量
        let key_count = db.len();
        health.insert("total_keys".to_string(), Value::Number(key_count.into()));
        
        // 检查活跃任务数量
        let active_tasks = self.count_active_tasks().await.unwrap_or(0);
        health.insert("active_tasks".to_string(), Value::Number(active_tasks.into()));
        
        // 计算整体健康状态
        let is_healthy = db_size > 0 && key_count > 0;
        health.insert("status".to_string(), Value::String(if is_healthy { "healthy".to_string() } else { "unhealthy".to_string() }));
        
        // 添加时间戳
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        health.insert("timestamp".to_string(), Value::Number(timestamp.into()));
        
        Ok(Value::Object(health))
    }
    
    /// 获取存储指标
    /// 
    /// # 返回值
    /// 异步返回存储相关的性能指标
    pub async fn get_storage_metrics(&self) -> Result<std::collections::HashMap<String, f64>> {
        let db = self.db.read().await;
        
        let mut metrics = std::collections::HashMap::new();
        
        // 读取操作计数
        let read_ops_key = "storage_metrics:read_operations";
        let read_ops = if let Ok(Some(data)) = db.get(read_ops_key.as_bytes()) {
            bincode::deserialize::<u64>(&data).unwrap_or(0) as f64
        } else {
            0.0
        };
        metrics.insert("read_operations".to_string(), read_ops);
        
        // 写入操作计数
        let write_ops_key = "storage_metrics:write_operations";
        let write_ops = if let Ok(Some(data)) = db.get(write_ops_key.as_bytes()) {
            bincode::deserialize::<u64>(&data).unwrap_or(0) as f64
        } else {
            0.0
        };
        metrics.insert("write_operations".to_string(), write_ops);
        
        // 缓存命中率
        let cache_hit_key = "storage_metrics:cache_hits";
        let cache_miss_key = "storage_metrics:cache_misses";
        
        let cache_hits = if let Ok(Some(data)) = db.get(cache_hit_key.as_bytes()) {
            bincode::deserialize::<u64>(&data).unwrap_or(0) as f64
        } else {
            0.0
        };
        
        let cache_misses = if let Ok(Some(data)) = db.get(cache_miss_key.as_bytes()) {
            bincode::deserialize::<u64>(&data).unwrap_or(0) as f64
        } else {
            0.0
        };
        
        let cache_hit_rate = if cache_hits + cache_misses > 0.0 {
            cache_hits / (cache_hits + cache_misses)
        } else {
            0.0
        };
        metrics.insert("cache_hit_rate".to_string(), cache_hit_rate);
        
        // 数据库大小（以MB为单位）
        let db_size = db.size_on_disk().unwrap_or(0) as f64 / (1024.0 * 1024.0);
        metrics.insert("database_size_mb".to_string(), db_size);
        
        // 键的数量
        let key_count = db.len() as f64;
        metrics.insert("total_keys".to_string(), key_count);
        
        // 平均响应时间（毫秒）- 从最近的操作记录中计算
        let avg_response_time = self.calculate_average_response_time().await.unwrap_or(0.0);
        metrics.insert("avg_response_time_ms".to_string(), avg_response_time);
        
        Ok(metrics)
    }
    
    /// 计算平均响应时间
    async fn calculate_average_response_time(&self) -> Result<f64> {
        let db = self.db.read().await;
        
        let prefix = "response_time:";
        let mut response_times = Vec::new();
        
        for item in db.scan_prefix(prefix.as_bytes()) {
            if let Ok((_, value)) = item {
                if let Ok(time) = bincode::deserialize::<f64>(&value) {
                    response_times.push(time);
                }
            }
        }
        
        if response_times.is_empty() {
            Ok(0.0)
        } else {
            let avg = response_times.iter().sum::<f64>() / response_times.len() as f64;
            Ok(avg)
        }
    }
} 