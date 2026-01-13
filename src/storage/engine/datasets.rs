use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use serde_json::Value;
use log::{debug, info, warn};
// use std::future::Future;
// use std::pin::Pin;

use crate::Result;
use crate::Error;
use crate::storage::models::{DataInfo};
use crate::data::{DataBatch, ProcessedBatch};

/// 数据集存储服务
/// 
/// 提供数据集相关数据的存储和检索功能，包括数据集数据、元数据、模式等
#[derive(Clone)]
pub struct DatasetStorageService {
    /// 底层数据库连接
    db: Arc<RwLock<sled::Db>>,
}

impl DatasetStorageService {
    /// 创建新的数据集存储服务
    /// 
    /// # 参数
    /// - `db`: Sled数据库实例
    pub fn new(db: Arc<RwLock<sled::Db>>) -> Self {
        Self { db }
    }
    
    /// 检查数据集是否存在
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    /// 
    /// # 返回值
    /// 异步返回数据集是否存在
    pub async fn dataset_exists(&self, dataset_id: &str) -> Result<bool> {
        let key = format!("dataset:{}:metadata", dataset_id);
        let db = self.db.read().await;
        
        Ok(db.contains_key(key.as_bytes())
            .map_err(|e| Error::storage(format!("检查数据集存在失败: {}", e)))?)
    }
    
    /// 获取数据集数据
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    /// 
    /// # 返回值
    /// 异步返回数据集的原始字节数据
    pub async fn get_dataset_data(&self, dataset_id: &str) -> Result<Vec<u8>> {
        let key = format!("dataset:{}:data", dataset_id);
        let db = self.db.read().await;
        
        if let Some(data) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取数据集数据失败: {}", e)))? {
            Ok(data.to_vec())
        } else {
            Err(Error::not_found(format!("数据集 {} 的数据不存在", dataset_id)))
        }
    }
    
    /// 获取数据集元数据
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    /// 
    /// # 返回值
    /// 异步返回数据集的元数据JSON对象
    pub async fn get_dataset_metadata(&self, dataset_id: &str) -> Result<Value> {
        let key = format!("dataset:{}:metadata", dataset_id);
        let db = self.db.read().await;
        
        if let Some(data) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取数据集元数据失败: {}", e)))? {
            let metadata: Value = serde_json::from_slice(&data)
                .map_err(|e| Error::serialization(format!("反序列化数据集元数据失败: {}", e)))?;
            Ok(metadata)
        } else {
            Err(Error::not_found(format!("数据集 {} 的元数据不存在", dataset_id)))
        }
    }
    
    /// 保存数据集数据
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    /// - `data`: 数据集的原始字节数据
    pub async fn save_dataset_data(&self, dataset_id: &str, data: &[u8]) -> Result<()> {
        let key = format!("dataset:{}:data", dataset_id);
        let db = self.db.read().await;
        
        db.insert(key.as_bytes(), data)
            .map_err(|e| Error::storage(format!("保存数据集数据失败: {}", e)))?;
        
        info!("已保存数据集 {} 的数据，大小: {} 字节", dataset_id, data.len());
        Ok(())
    }
    
    /// 保存数据集元数据
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    /// - `metadata`: 数据集的元数据JSON对象
    pub async fn save_dataset_metadata(&self, dataset_id: &str, metadata: &Value) -> Result<()> {
        let key = format!("dataset:{}:metadata", dataset_id);
        let data = serde_json::to_vec(metadata)
            .map_err(|e| Error::serialization(format!("序列化数据集元数据失败: {}", e)))?;
        
        let db = self.db.read().await;
        db.insert(key.as_bytes(), data)
            .map_err(|e| Error::storage(format!("保存数据集元数据失败: {}", e)))?;
        
        debug!("已保存数据集 {} 的元数据", dataset_id);
        Ok(())
    }
    
    /// 获取数据集模式
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    /// 
    /// # 返回值
    /// 异步返回数据集的模式定义JSON对象
    pub async fn get_dataset_schema(&self, dataset_id: &str) -> Result<Value> {
        let key = format!("dataset:{}:schema", dataset_id);
        let db = self.db.read().await;
        
        if let Some(data) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取数据集模式失败: {}", e)))? {
            let schema: Value = serde_json::from_slice(&data)
                .map_err(|e| Error::serialization(format!("反序列化数据集模式失败: {}", e)))?;
            Ok(schema)
        } else {
            // 如果没有明确的模式定义，返回一个默认的模式
            Ok(serde_json::json!({
                "type": "object",
                "properties": {},
                "description": "未定义的数据集模式"
            }))
        }
    }
    
    /// 保存数据集模式
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    /// - `schema`: 数据集的模式定义JSON对象
    pub async fn save_dataset_schema(&self, dataset_id: &str, schema: &Value) -> Result<()> {
        let key = format!("dataset:{}:schema", dataset_id);
        let data = serde_json::to_vec(schema)
            .map_err(|e| Error::serialization(format!("序列化数据集模式失败: {}", e)))?;
        
        let db = self.db.read().await;
        db.insert(key.as_bytes(), data)
            .map_err(|e| Error::storage(format!("保存数据集模式失败: {}", e)))?;
        
        debug!("已保存数据集 {} 的模式", dataset_id);
        Ok(())
    }
    
    /// 删除完整数据集
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    pub async fn delete_dataset_complete(&self, dataset_id: &str) -> Result<()> {
        let db = self.db.read().await;
        
        // 删除数据集的所有相关键
        let keys_to_delete = vec![
            format!("dataset:{}:data", dataset_id),
            format!("dataset:{}:metadata", dataset_id),
            format!("dataset:{}:schema", dataset_id),
            format!("dataset:{}:info", dataset_id),
        ];
        
        for key in keys_to_delete {
            if let Err(e) = db.remove(key.as_bytes()) {
                warn!("删除数据集键 {} 失败: {}", key, e);
            }
        }
        
        // 删除所有以数据集ID为前缀的键
        let prefix = format!("dataset:{}:", dataset_id);
        let mut keys_to_remove = Vec::new();
        
        for item in db.scan_prefix(prefix.as_bytes()) {
            match item {
                Ok((key, _)) => keys_to_remove.push(key),
                Err(e) => warn!("扫描数据集前缀时出错: {}", e),
            }
        }
        
        for key in keys_to_remove {
            if let Err(e) = db.remove(&key) {
                warn!("删除数据集键失败: {}", e);
            }
        }
        
        info!("已完全删除数据集 {}", dataset_id);
        Ok(())
    }
    
    /// 列出所有数据集
    /// 
    /// # 返回值
    /// 异步返回所有数据集的ID列表
    pub async fn list_datasets(&self) -> Result<Vec<String>> {
        let prefix = "dataset:";
        let db = self.db.read().await;
        
        let mut dataset_ids = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();
        
        for item in db.scan_prefix(prefix.as_bytes()) {
            match item {
                Ok((key, _)) => {
                    let key_str = String::from_utf8_lossy(&key);
                    if let Some(parts) = key_str.strip_prefix("dataset:") {
                        if let Some(dataset_id) = parts.split(':').next() {
                            if seen_ids.insert(dataset_id.to_string()) {
                                dataset_ids.push(dataset_id.to_string());
                            }
                        }
                    }
                },
                Err(e) => {
                    warn!("扫描数据集时出错: {}", e);
                }
            }
        }
        
        dataset_ids.sort();
        Ok(dataset_ids)
    }
    
    /// 获取数据集ID列表
    /// 
    /// # 返回值
    /// 异步返回所有数据集的ID列表
    pub async fn get_dataset_ids(&self) -> Result<Vec<String>> {
        self.list_datasets().await
    }
    
    /// 获取数据集信息
    /// 
    /// # 参数
    /// - `id`: 数据集唯一标识符
    /// 
    /// # 返回值
    /// 异步返回数据集信息JSON对象，如果不存在则返回None
    pub async fn get_dataset_info(&self, id: &str) -> Result<Option<Value>> {
        let key = format!("dataset:{}:info", id);
        let db = self.db.read().await;
        
        if let Some(data) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取数据集信息失败: {}", e)))? {
            let info: Value = serde_json::from_slice(&data)
                .map_err(|e| Error::serialization(format!("反序列化数据集信息失败: {}", e)))?;
            Ok(Some(info))
        } else {
            Ok(None)
        }
    }
    
    /// 获取数据集
    /// 
    /// # 参数
    /// - `dataset_id`: 数据集唯一标识符
    /// 
    /// # 返回值
    /// 异步返回数据集对象，如果不存在则返回None
    pub async fn get_dataset(&self, dataset_id: &str) -> Result<Option<crate::data::Dataset>> {
        // 获取数据集信息
        let info = self.get_dataset_info(dataset_id).await?;
        if info.is_none() {
            return Ok(None);
        }
        
        // 获取数据集数据
        let data = self.get_dataset_data(dataset_id).await?;
        let metadata = self.get_dataset_metadata(dataset_id).await?;
        let schema_value = self.get_dataset_schema(dataset_id).await.ok();
        
        // 将Value转换为DataSchema
        let schema = if let Some(schema_json) = schema_value {
            if let Ok(schema_str) = serde_json::to_string(&schema_json) {
                crate::data::schema::DataSchema::from_json(&schema_str).ok()
            } else {
                None
            }
        } else {
            None
        };
        
        // 构建数据集元数据
        let dataset_metadata = crate::data::DatasetMetadata {
            id: dataset_id.to_string(),
            name: info.as_ref()
                .and_then(|i| i.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or(dataset_id)
                .to_string(),
            description: info.as_ref()
                .and_then(|i| i.get("description"))
                .and_then(|d| d.as_str())
                .map(|s| s.to_string()),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            version: "1.0".to_string(),
            owner: "system".to_string(),
            schema: schema.as_ref().and_then(|s| s.to_json().ok()),
            properties: {
                let mut props = HashMap::new();
                if let Some(meta_obj) = metadata.as_object() {
                    for (key, value) in meta_obj {
                        if let Some(str_val) = value.as_str() {
                            props.insert(key.clone(), str_val.to_string());
                        }
                    }
                }
                props
            },
            tags: Vec::new(),
            records_count: data.len(),
            size_bytes: data.len() as u64,
        };
        
        // 构建数据集对象
        let dataset = crate::data::Dataset {
            id: dataset_id.to_string(),
            name: dataset_metadata.name.clone(),
            description: dataset_metadata.description.clone(),
            format: crate::data::DataFormat::Json, // 默认格式
            size: data.len(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            metadata: dataset_metadata,
            path: format!("/datasets/{}", dataset_id),
            processed: true,
            loader: std::sync::Arc::new(crate::data::loader::memory::MemoryDataLoader::new()),
            batch_size: 32,
            schema: schema,
            batches: Vec::new(), // 添加缺失的batches字段
        };
        
        Ok(Some(dataset))
    }

    /// 删除数据集
    /// 
    /// # 参数
    /// - `id`: 数据集唯一标识符
    pub async fn delete_dataset(&self, id: &str) -> Result<()> {
        self.delete_dataset_complete(id).await
    }
    
    /// 保存数据集信息
    /// 
    /// # 参数
    /// - `dataset`: 数据集信息JSON对象
    pub async fn save_dataset_info(&self, dataset: &Value) -> Result<()> {
        // 从dataset对象中提取ID
        let dataset_id = dataset.get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::validation("数据集信息缺少ID字段".to_string()))?;
        
        let key = format!("dataset:{}:info", dataset_id);
        let data = serde_json::to_vec(dataset)
            .map_err(|e| Error::serialization(format!("序列化数据集信息失败: {}", e)))?;
        
        let db = self.db.read().await;
        db.insert(key.as_bytes(), data)
            .map_err(|e| Error::storage(format!("保存数据集信息失败: {}", e)))?;
        
        info!("已保存数据集 {} 的信息", dataset_id);
        Ok(())
    }
    
    /// 获取数据集特征统计
    /// 
    /// # 参数
    /// - `id`: 数据集唯一标识符
    /// 
    /// # 返回值
    /// 异步返回数据集特征统计JSON对象
    pub async fn get_dataset_feature_stats(&self, id: &str) -> Result<Value> {
        let key = format!("dataset:{}:feature_stats", id);
        let db = self.db.read().await;
        
        if let Some(data) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取数据集特征统计失败: {}", e)))? {
            let stats: Value = serde_json::from_slice(&data)
                .map_err(|e| Error::serialization(format!("反序列化数据集特征统计失败: {}", e)))?;
            Ok(stats)
        } else {
            // 如果没有预计算的统计信息，返回空的统计对象
            Ok(serde_json::json!({
                "feature_count": 0,
                "record_count": 0,
                "features": {}
            }))
        }
    }
    
    /// 保存数据集特征统计
    /// 
    /// # 参数
    /// - `id`: 数据集唯一标识符
    /// - `stats`: 特征统计JSON对象
    pub async fn save_dataset_feature_stats(&self, id: &str, stats: &Value) -> Result<()> {
        let key = format!("dataset:{}:feature_stats", id);
        let data = serde_json::to_vec(stats)
            .map_err(|e| Error::serialization(format!("序列化数据集特征统计失败: {}", e)))?;
        
        let db = self.db.read().await;
        db.insert(key.as_bytes(), data)
            .map_err(|e| Error::storage(format!("保存数据集特征统计失败: {}", e)))?;
        
        debug!("已保存数据集 {} 的特征统计", id);
        Ok(())
    }
    
    /// 查询数据集
    /// 
    /// # 参数
    /// - `name`: 数据集名称
    /// - `limit`: 限制返回数量
    /// - `offset`: 偏移量
    /// 
    /// # 返回值
    /// 异步返回查询结果JSON数组
    pub async fn query_dataset(
        &self,
        name: &str,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<Vec<Value>> {
        let db = self.db.read().await;
        
        let mut results = Vec::new();
        let mut count = 0;
        let offset = offset.unwrap_or(0);
        let limit = limit.unwrap_or(100);
        
        // 扫描所有数据集信息
        for item in db.scan_prefix("dataset:".as_bytes()) {
            match item {
                Ok((key, value)) => {
                    let key_str = String::from_utf8_lossy(&key);
                    if key_str.ends_with(":info") {
                        if count >= offset {
                            if let Ok(info) = serde_json::from_slice::<Value>(&value) {
                                // 检查数据集名称是否匹配
                                if let Some(dataset_name) = info.get("name").and_then(|v| v.as_str()) {
                                    if dataset_name.contains(name) || name.is_empty() {
                                        results.push(info);
                                        if results.len() >= limit {
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        count += 1;
                    }
                },
                Err(e) => {
                    warn!("扫描数据集查询时出错: {}", e);
                }
            }
        }
        
        Ok(results)
    }
    
    /// 获取数据信息（同步方法）
    /// 
    /// # 参数
    /// - `data_id`: 数据唯一标识符
    /// 
    /// # 返回值
    /// 返回数据信息，如果不存在则返回None
    pub fn get_data_info_sync(&self, data_id: &str) -> Result<Option<DataInfo>> {
        let key = format!("data:{}:info", data_id);
        let db = self.db.blocking_read();
        
        if let Some(data) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取数据信息失败: {}", e)))? {
            let info = bincode::deserialize(&data)
                .map_err(|e| Error::serialization(format!("反序列化数据信息失败: {}", e)))?;
            Ok(Some(info))
        } else {
            Ok(None)
        }
    }
    
    /// 保存数据信息
    /// 
    /// # 参数
    /// - `data_id`: 数据唯一标识符
    /// - `info`: 数据信息对象
    pub async fn save_data_info(&self, data_id: &str, info: &DataInfo) -> Result<()> {
        let key = format!("data:{}:info", data_id);
        let value = bincode::serialize(info)
            .map_err(|e| Error::serialization(format!("序列化数据信息失败: {}", e)))?;
        
        let db = self.db.read().await;
        db.insert(key.as_bytes(), value)
            .map_err(|e| Error::storage(format!("保存数据信息失败: {}", e)))?;
        
        debug!("已保存数据 {} 的信息", data_id);
        Ok(())
    }
    
    /// 删除数据
    /// 
    /// # 参数
    /// - `data_id`: 数据唯一标识符
    pub async fn delete_data(&self, data_id: &str) -> Result<()> {
        let db = self.db.read().await;
        
        // 删除数据的所有相关键
        let keys_to_delete = vec![
            format!("data:{}:info", data_id),
            format!("data:{}:content", data_id),
            format!("data:{}", data_id),
        ];
        
        for key in keys_to_delete {
            if let Err(e) = db.remove(key.as_bytes()) {
                debug!("删除数据键 {} 失败（可能不存在）: {}", key, e);
            }
        }
        
        info!("已删除数据 {}", data_id);
        Ok(())
    }
    
    /// 获取数据批次
    /// 
    /// # 参数
    /// - `batch_id`: 批次唯一标识符
    /// 
    /// # 返回值
    /// 异步返回数据批次对象
    pub async fn get_data_batch(&self, batch_id: &str) -> Result<DataBatch> {
        let key = format!("batch:{}", batch_id);
        let db = self.db.read().await;
        
        if let Some(data) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取数据批次失败: {}", e)))? {
            let batch = bincode::deserialize(&data)
                .map_err(|e| Error::serialization(format!("反序列化数据批次失败: {}", e)))?;
            Ok(batch)
        } else {
            Err(Error::not_found(format!("数据批次 {} 不存在", batch_id)))
        }
    }
    
    /// 保存处理后的批次
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `batch`: 处理后的数据批次
    pub async fn save_processed_batch(
        &self,
        model_id: &str,
        batch: &ProcessedBatch,
    ) -> Result<()> {
        let key = format!("model:{}:processed_batch:{}", model_id, batch.id);
        let value = bincode::serialize(batch)
            .map_err(|e| Error::serialization(format!("序列化处理后批次失败: {}", e)))?;
        
        let db = self.db.read().await;
        db.insert(key.as_bytes(), value)
            .map_err(|e| Error::storage(format!("保存处理后批次失败: {}", e)))?;
        
        info!("已保存模型 {} 的处理后批次 {}", model_id, batch.id);
        Ok(())
    }
} 