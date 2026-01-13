use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use crate::core::types::CoreTensorData;
use crate::error::Result;
use async_trait::async_trait;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataBatchInterface {
    pub id: String,
    pub features: Vec<CoreTensorData>,
    pub labels: Option<Vec<CoreTensorData>>,
    pub metadata: HashMap<String, String>,
    pub batch_size: usize,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInterface {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub data_type: String,
    pub schema: DataSchemaInterface,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSchemaInterface {
    pub fields: Vec<FieldDefinitionInterface>,
    pub version: String,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinitionInterface {
    pub name: String,
    pub data_type: String,
    pub required: bool,
    pub constraints: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// 数据处理服务接口
#[async_trait]
pub trait DataProcessingService: Send + Sync {
    async fn process_batch(&self, batch: DataBatchInterface) -> Result<DataBatchInterface>;
    async fn get_dataset(&self, dataset_id: &str) -> Result<Option<DatasetInterface>>;
    async fn create_dataset(&self, dataset: DatasetInterface) -> Result<String>;
    async fn validate_data(&self, data: &DataBatchInterface, schema: &DataSchemaInterface) -> Result<ValidationResult>;
}

/// 数据处理器接口
/// 提供统一的数据处理能力，支持多种数据格式和转换操作
#[async_trait]
pub trait DataProcessorInterface: Send + Sync {
    /// 处理数据批次
    async fn process_batch(&self, batch: &crate::core::types::CoreDataBatch) -> Result<crate::core::types::CoreDataBatch>;
    
    /// 验证数据格式
    async fn validate_data(&self, data: &crate::core::types::CoreDataBatch) -> Result<ValidationResult>;
    
    /// 转换数据格式
    async fn convert_data(&self, data: &crate::core::types::CoreDataBatch, target_format: &str) -> Result<crate::core::types::CoreDataBatch>;
    
    /// 获取数据统计信息
    async fn get_data_statistics(&self, data: &crate::core::types::CoreDataBatch) -> Result<HashMap<String, f64>>;
    
    /// 清理数据
    async fn clean_data(&self, data: &crate::core::types::CoreDataBatch) -> Result<crate::core::types::CoreDataBatch>;
    
    /// 标准化数据
    async fn normalize_data(&self, data: &crate::core::types::CoreDataBatch) -> Result<crate::core::types::CoreDataBatch>;
    
    /// 获取处理器配置
    async fn get_processor_config(&self) -> Result<HashMap<String, String>>;
    
    /// 更新处理器配置
    async fn update_processor_config(&self, config: HashMap<String, String>) -> Result<()>;
    
    /// 处理数据
    async fn process_data(&self, data: &crate::core::types::CoreDataBatch) -> Result<crate::core::types::ProcessedData>;
    
    /// 转换为张量
    async fn convert_to_tensors(&self, data: &crate::core::types::ProcessedData) -> Result<Vec<crate::core::types::CoreTensorData>>;
    
    /// 验证数据模式
    async fn validate_data_schema(&self, data: &crate::core::types::CoreDataBatch, schema: &crate::data::loader::types::DataSchema) -> Result<ValidationResult>;
    
    /// 预处理数据
    async fn preprocess_data(&self, data: &crate::core::types::CoreDataBatch) -> Result<crate::core::types::ProcessedData>;
    
    /// 分割数据
    async fn split_data(&self, data: &crate::core::types::CoreDataBatch, train_ratio: f32, val_ratio: f32, test_ratio: f32) -> Result<(crate::core::types::CoreDataBatch, crate::core::types::CoreDataBatch, crate::core::types::CoreDataBatch)>;
}

// ValidationResult 现在已在 mod.rs 中定义，通过 super 导入
use super::ValidationResult;


