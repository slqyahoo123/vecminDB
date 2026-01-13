use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use log::info;
use tokio::sync::RwLock;

use crate::Result;
use crate::storage::config::StorageConfig;
use super::transaction::TransactionManager;
use super::models::ModelStorageService;
use super::datasets::DatasetStorageService;
use super::monitoring::MonitoringStorageService;

/// 数据集集成服务
/// 
/// 提供数据集统计信息、完整性验证、分析报告等高级数据集管理功能
#[derive(Clone)]
pub struct DatasetIntegration {
    config: StorageConfig,
    db: Arc<RwLock<sled::Db>>,
    transaction_manager: Arc<Mutex<TransactionManager>>,
    model_storage: ModelStorageService,
    dataset_storage: DatasetStorageService,
    monitoring_storage: MonitoringStorageService,
}

impl DatasetIntegration {
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

    /// 获取数据集统计信息
    pub async fn get_dataset_statistics(&self) -> Result<serde_json::Value> {
        let mut stats = serde_json::Map::new();
        info!("collecting dataset statistics");
        
        // 获取所有数据集
        let datasets = self.dataset_storage.list_datasets().await.unwrap_or_default();
        stats.insert("total_datasets".to_string(), serde_json::Value::Number(serde_json::Number::from(datasets.len())));
        
        // 数据集类型统计（生产级实现：基于实际元数据）
        let mut type_stats = HashMap::new();
        let mut total_size_bytes = 0u64;
        let mut size_distribution = HashMap::new();
        let mut recent_access_times = Vec::new();
        
        for dataset_id in &datasets {
            // 获取数据集的实际元数据
            let metadata_key = format!("dataset:{}:metadata", dataset_id);
            if let Ok(metadata_bytes) = self.db.read().await.get(metadata_key.as_bytes()) {
                if let Some(bytes) = metadata_bytes {
                    // 尝试解析元数据
                    if let Ok(metadata) = serde_json::from_slice::<serde_json::Value>(&bytes) {
                        // 统计数据集类型
                        let dataset_type = metadata.get("type")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string();
                        *type_stats.entry(dataset_type).or_insert(0) += 1;
                        
                        // 统计数据集大小
                        if let Some(size) = metadata.get("size_bytes").and_then(|v| v.as_u64()) {
                            total_size_bytes += size;
                            
                            // 按大小分类：small(<10MB), medium(10MB-100MB), large(>100MB)
                            let size_category = if size < 10 * 1024 * 1024 {
                                "small"
                            } else if size < 100 * 1024 * 1024 {
                                "medium"
                            } else {
                                "large"
                            };
                            *size_distribution.entry(size_category.to_string()).or_insert(0) += 1;
                        } else {
                            // 如果没有大小信息，尝试从数据键获取实际大小
                            let data_key = format!("dataset:{}:data", dataset_id);
                            if let Ok(Some(data)) = self.db.read().await.get(data_key.as_bytes()) {
                                let size = data.len() as u64;
                                total_size_bytes += size;
                                
                                let size_category = if size < 10 * 1024 * 1024 {
                                    "small"
                                } else if size < 100 * 1024 * 1024 {
                                    "medium"
                                } else {
                                    "large"
                                };
                                *size_distribution.entry(size_category.to_string()).or_insert(0) += 1;
                            } else {
                                // 无法获取大小，归类为unknown
                                *size_distribution.entry("unknown".to_string()).or_insert(0) += 1;
                            }
                        }
                        
                        // 统计最近访问时间
                        if let Some(last_access) = metadata.get("last_accessed_at").and_then(|v| v.as_str()) {
                            recent_access_times.push(serde_json::Value::String(last_access.to_string()));
                        } else {
                            // 如果没有访问时间，使用创建时间
                            let created_at = metadata.get("created_at")
                                .and_then(|v| v.as_str())
                                .unwrap_or_else(|| "unknown");
                            recent_access_times.push(serde_json::Value::String(created_at.to_string()));
                        }
                    } else {
                        // 元数据解析失败，使用默认值
                        *type_stats.entry("unknown".to_string()).or_insert(0) += 1;
                        *size_distribution.entry("unknown".to_string()).or_insert(0) += 1;
                        recent_access_times.push(serde_json::Value::String("unknown".to_string()));
                    }
                } else {
                    // 元数据不存在，使用默认值
                    *type_stats.entry("no_metadata".to_string()).or_insert(0) += 1;
                    *size_distribution.entry("unknown".to_string()).or_insert(0) += 1;
                    recent_access_times.push(serde_json::Value::String("no_metadata".to_string()));
                }
            } else {
                // 数据库访问失败，使用默认值
                *type_stats.entry("error".to_string()).or_insert(0) += 1;
                *size_distribution.entry("error".to_string()).or_insert(0) += 1;
                recent_access_times.push(serde_json::Value::String("error".to_string()));
            }
        }
        
        // 添加统计数据到结果
        stats.insert("dataset_types".to_string(), serde_json::to_value(type_stats)?);
        stats.insert("size_distribution".to_string(), serde_json::to_value(size_distribution)?);
        stats.insert("total_size_bytes".to_string(), serde_json::Value::Number(serde_json::Number::from(total_size_bytes)));
        stats.insert("average_size_bytes".to_string(), serde_json::Value::Number(serde_json::Number::from(
            if datasets.is_empty() { 0 } else { total_size_bytes / datasets.len() as u64 }
        )));
        stats.insert("recent_accesses".to_string(), serde_json::Value::Array(recent_access_times));
        
        info!("dataset statistics collected (total: {}, total_size: {} bytes)", 
              datasets.len(), total_size_bytes);
        
        Ok(serde_json::Value::Object(stats))
    }

    /// 验证数据集完整性
    pub async fn validate_dataset_integrity(&self, dataset_id: &str) -> Result<serde_json::Value> {
        let mut integrity = serde_json::Map::new();
        
        // 检查数据集是否存在
        let exists = self.dataset_storage.dataset_exists(dataset_id).await?;
        integrity.insert("dataset_exists".to_string(), serde_json::Value::Bool(exists));
        
        if exists {
            // 数据完整性检查
            let data_integrity = self.check_dataset_data_integrity(dataset_id).await?;
            integrity.insert("data_integrity".to_string(), serde_json::Value::Object(data_integrity));
            
            // 元数据完整性检查
            let metadata_integrity = self.check_dataset_metadata_integrity(dataset_id).await?;
            integrity.insert("metadata_integrity".to_string(), serde_json::Value::Object(metadata_integrity));
            
            // 模式完整性检查
            let schema_integrity = self.check_dataset_schema_integrity(dataset_id).await?;
            integrity.insert("schema_integrity".to_string(), serde_json::Value::Object(schema_integrity));
            
            // 计算整体完整性评分
            let overall_score = self.calculate_integrity_score(&integrity)?;
            integrity.insert("overall_integrity_score".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(overall_score).unwrap()));
        }
        
        Ok(serde_json::Value::Object(integrity))
    }

    /// 检查数据集数据完整性
    async fn check_dataset_data_integrity(&self, dataset_id: &str) -> Result<serde_json::Map<String, serde_json::Value>> {
        let mut integrity = serde_json::Map::new();
        
        // 检查数据文件是否存在
        let data_key = format!("dataset:{}:data", dataset_id);
        let data_exists = self.db.read().await.contains_key(data_key.as_bytes())?;
        integrity.insert("data_file_exists".to_string(), serde_json::Value::Bool(data_exists));
        
        if data_exists {
            // 检查数据大小
            if let Some(data) = self.db.read().await.get(data_key.as_bytes())? {
                let data_size = data.len();
                integrity.insert("data_size_bytes".to_string(), serde_json::Value::Number(serde_json::Number::from(data_size)));
                integrity.insert("data_size_valid".to_string(), serde_json::Value::Bool(data_size > 0));
            }
            
            // 检查数据格式（生产级实现：实际验证数据格式）
            let format_valid = if let Some(data) = self.db.read().await.get(data_key.as_bytes())? {
                // 尝试验证数据格式
                match self.validate_data_format(&data, dataset_id).await {
                    Ok(is_valid) => is_valid,
                    Err(e) => {
                        log::warn!("数据格式验证失败: {}", e);
                        false
                    }
                }
            } else {
                false
            };
            integrity.insert("data_format_valid".to_string(), serde_json::Value::Bool(format_valid));
        }
        
        Ok(integrity)
    }

    /// 检查数据集元数据完整性
    async fn check_dataset_metadata_integrity(&self, dataset_id: &str) -> Result<serde_json::Map<String, serde_json::Value>> {
        let mut integrity = serde_json::Map::new();
        
        // 检查元数据文件是否存在
        let metadata_key = format!("dataset:{}:metadata", dataset_id);
        let metadata_exists = self.db.read().await.contains_key(metadata_key.as_bytes())?;
        integrity.insert("metadata_file_exists".to_string(), serde_json::Value::Bool(metadata_exists));
        
        if metadata_exists {
            // 检查元数据内容
            if let Some(metadata) = self.db.read().await.get(metadata_key.as_bytes())? {
                if let Ok(metadata_value) = serde_json::from_slice::<serde_json::Value>(&metadata) {
                    if let Some(obj) = metadata_value.as_object() {
                        integrity.insert("has_name".to_string(), serde_json::Value::Bool(obj.contains_key("name")));
                        integrity.insert("has_description".to_string(), serde_json::Value::Bool(obj.contains_key("description")));
                        integrity.insert("has_type".to_string(), serde_json::Value::Bool(obj.contains_key("type")));
                        integrity.insert("has_created_at".to_string(), serde_json::Value::Bool(obj.contains_key("created_at")));
                    }
                }
            }
        }
        
        Ok(integrity)
    }

    /// 检查数据集模式完整性
    async fn check_dataset_schema_integrity(&self, dataset_id: &str) -> Result<serde_json::Map<String, serde_json::Value>> {
        let mut integrity = serde_json::Map::new();
        
        // 检查模式文件是否存在
        let schema_key = format!("dataset:{}:schema", dataset_id);
        let schema_exists = self.db.read().await.contains_key(schema_key.as_bytes())?;
        integrity.insert("schema_file_exists".to_string(), serde_json::Value::Bool(schema_exists));
        
        if schema_exists {
            // 检查模式内容
            if let Some(schema) = self.db.read().await.get(schema_key.as_bytes())? {
                if let Ok(schema_value) = serde_json::from_slice::<serde_json::Value>(&schema) {
                    if let Some(obj) = schema_value.as_object() {
                        integrity.insert("has_fields".to_string(), serde_json::Value::Bool(obj.contains_key("fields")));
                        integrity.insert("has_types".to_string(), serde_json::Value::Bool(obj.contains_key("types")));
                        integrity.insert("has_constraints".to_string(), serde_json::Value::Bool(obj.contains_key("constraints")));
                    }
                }
            }
        }
        
        Ok(integrity)
    }

    /// 验证数据格式
    async fn validate_data_format(&self, data: &[u8], dataset_id: &str) -> Result<bool> {
        // 获取数据集的元数据以确定预期格式
        let metadata_key = format!("dataset:{}:metadata", dataset_id);
        let expected_format = if let Some(metadata) = self.db.read().await.get(metadata_key.as_bytes())? {
            if let Ok(metadata_value) = serde_json::from_slice::<serde_json::Value>(&metadata) {
                metadata_value.get("format")
                    .and_then(|v| v.as_str())
                    .unwrap_or("json")
                    .to_string()
            } else {
                "json".to_string()
            }
        } else {
            "json".to_string()
        };
        
        // 根据预期格式验证数据
        let is_valid = match expected_format.as_str() {
            "json" => {
                // 验证JSON格式
                serde_json::from_slice::<serde_json::Value>(data).is_ok()
            },
            "bincode" => {
                // 验证Bincode格式（尝试反序列化为通用值）
                bincode::deserialize::<serde_json::Value>(data).is_ok()
            },
            "msgpack" | "messagepack" => {
                // 验证MessagePack格式
                rmp_serde::from_slice::<serde_json::Value>(data).is_ok()
            },
            "csv" => {
                // 验证CSV格式（检查是否包含有效的行）
                if data.is_empty() {
                    return Ok(false);
                }
                // 简单验证：检查是否包含换行符和可打印字符
                let data_str = String::from_utf8_lossy(data);
                !data_str.is_empty() && data_str.chars().any(|c| c == '\n' || c == '\r')
            },
            "parquet" => {
                // Parquet格式验证（检查文件头）
                if data.len() < 4 {
                    return Ok(false);
                }
                // Parquet文件以"PAR1"开头
                data.starts_with(b"PAR1")
            },
            "binary" | "raw" => {
                // 二进制格式：只要有数据就认为有效
                !data.is_empty()
            },
            _ => {
                // 未知格式：尝试JSON验证作为后备
                log::warn!("未知的数据格式: {}, 使用JSON验证作为后备", expected_format);
                serde_json::from_slice::<serde_json::Value>(data).is_ok()
            }
        };
        
        if !is_valid {
            log::warn!("数据格式验证失败：数据集 {}, 预期格式 {}", dataset_id, expected_format);
        }
        
        Ok(is_valid)
    }

    /// 计算完整性评分
    fn calculate_integrity_score(&self, integrity: &serde_json::Map<String, serde_json::Value>) -> Result<f64> {
        let mut score: f64 = 100.0;
        
        // 检查数据完整性
        if let Some(data_integrity) = integrity.get("data_integrity") {
            if let Some(data_map) = data_integrity.as_object() {
                if let Some(exists) = data_map.get("data_file_exists") {
                    if !exists.as_bool().unwrap_or(false) {
                        score -= 30.0; // 数据文件不存在扣分
                    }
                }
                if let Some(valid) = data_map.get("data_size_valid") {
                    if !valid.as_bool().unwrap_or(false) {
                        score -= 20.0; // 数据大小无效扣分
                    }
                }
            }
        }
        
        // 检查元数据完整性
        if let Some(metadata_integrity) = integrity.get("metadata_integrity") {
            if let Some(metadata_map) = metadata_integrity.as_object() {
                if let Some(exists) = metadata_map.get("metadata_file_exists") {
                    if !exists.as_bool().unwrap_or(false) {
                        score -= 25.0; // 元数据文件不存在扣分
                    }
                }
            }
        }
        
        // 检查模式完整性
        if let Some(schema_integrity) = integrity.get("schema_integrity") {
            if let Some(schema_map) = schema_integrity.as_object() {
                if let Some(exists) = schema_map.get("schema_file_exists") {
                    if !exists.as_bool().unwrap_or(false) {
                        score -= 25.0; // 模式文件不存在扣分
                    }
                }
            }
        }
        
        Ok(score.max(0.0_f64).min(100.0_f64))
    }

    /// 获取数据集分析报告
    pub async fn get_dataset_analysis_report(&self, dataset_id: &str) -> Result<serde_json::Value> {
        let mut report = serde_json::Map::new();
        
        // 基础信息
        let exists = self.dataset_storage.dataset_exists(dataset_id).await?;
        report.insert("dataset_id".to_string(), serde_json::Value::String(dataset_id.to_string()));
        report.insert("exists".to_string(), serde_json::Value::Bool(exists));
        
        if exists {
            // 数据统计
            let data_stats = self.get_dataset_data_statistics(dataset_id).await?;
            report.insert("data_statistics".to_string(), serde_json::Value::Object(data_stats));
            
            // 元数据分析
            let metadata_analysis = self.get_dataset_metadata_analysis(dataset_id).await?;
            report.insert("metadata_analysis".to_string(), serde_json::Value::Object(metadata_analysis));
            
            // 模式分析
            let schema_analysis = self.get_dataset_schema_analysis(dataset_id).await?;
            report.insert("schema_analysis".to_string(), serde_json::Value::Object(schema_analysis));
            
            // 完整性检查
            let integrity_check = self.validate_dataset_integrity(dataset_id).await?;
            report.insert("integrity_check".to_string(), integrity_check);
        }
        
        Ok(serde_json::Value::Object(report))
    }

    /// 获取数据集数据统计
    async fn get_dataset_data_statistics(&self, dataset_id: &str) -> Result<serde_json::Map<String, serde_json::Value>> {
        let mut stats = serde_json::Map::new();
        
        let data_key = format!("dataset:{}:data", dataset_id);
        if let Some(data) = self.db.read().await.get(data_key.as_bytes())? {
            stats.insert("data_size_bytes".to_string(), serde_json::Value::Number(serde_json::Number::from(data.len())));
            stats.insert("data_available".to_string(), serde_json::Value::Bool(true));
            
            // 尝试解析数据格式
            if let Ok(json_data) = serde_json::from_slice::<serde_json::Value>(&data) {
                if let Some(array) = json_data.as_array() {
                    stats.insert("record_count".to_string(), serde_json::Value::Number(serde_json::Number::from(array.len())));
                }
            }
        } else {
            stats.insert("data_available".to_string(), serde_json::Value::Bool(false));
        }
        
        Ok(stats)
    }

    /// 获取数据集元数据分析
    async fn get_dataset_metadata_analysis(&self, dataset_id: &str) -> Result<serde_json::Map<String, serde_json::Value>> {
        let mut analysis = serde_json::Map::new();
        
        let metadata_key = format!("dataset:{}:metadata", dataset_id);
        if let Some(metadata) = self.db.read().await.get(metadata_key.as_bytes())? {
            if let Ok(metadata_value) = serde_json::from_slice::<serde_json::Value>(&metadata) {
                if let Some(obj) = metadata_value.as_object() {
                    analysis.insert("metadata_available".to_string(), serde_json::Value::Bool(true));
                    analysis.insert("field_count".to_string(), serde_json::Value::Number(serde_json::Number::from(obj.len())));
                    
                    // 分析关键字段
                    for (key, value) in obj {
                        analysis.insert(format!("has_{}", key), serde_json::Value::Bool(!value.is_null()));
                    }
                }
            }
        } else {
            analysis.insert("metadata_available".to_string(), serde_json::Value::Bool(false));
        }
        
        Ok(analysis)
    }

    /// 获取数据集模式分析
    async fn get_dataset_schema_analysis(&self, dataset_id: &str) -> Result<serde_json::Map<String, serde_json::Value>> {
        let mut analysis = serde_json::Map::new();
        
        let schema_key = format!("dataset:{}:schema", dataset_id);
        if let Some(schema) = self.db.read().await.get(schema_key.as_bytes())? {
            if let Ok(schema_value) = serde_json::from_slice::<serde_json::Value>(&schema) {
                if let Some(obj) = schema_value.as_object() {
                    analysis.insert("schema_available".to_string(), serde_json::Value::Bool(true));
                    analysis.insert("field_count".to_string(), serde_json::Value::Number(serde_json::Number::from(obj.len())));
                    
                    // 分析模式字段
                    for (key, value) in obj {
                        analysis.insert(format!("has_{}", key), serde_json::Value::Bool(!value.is_null()));
                    }
                }
            }
        } else {
            analysis.insert("schema_available".to_string(), serde_json::Value::Bool(false));
        }
        
        Ok(analysis)
    }
} 