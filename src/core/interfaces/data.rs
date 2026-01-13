use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use crate::error::Result;
use crate::core::types::{CoreDataBatch, CoreTensorData, ProcessedData};
use crate::data::schema::DataSchema;

/// 数据加载器接口
#[async_trait]
pub trait DataLoader: Send + Sync {
    async fn load_batch(&self, batch_size: usize, shuffle: bool) -> Result<CoreDataBatch>;
    async fn get_dataset_size(&self) -> Result<usize>;
    async fn reset(&self) -> Result<()>;
    async fn get_schema(&self) -> Result<DataSchema>;
}

/// 数据预处理接口
#[async_trait]
pub trait DataPreprocessor: Send + Sync {
    async fn preprocess(&self, data: &CoreDataBatch) -> Result<ProcessedData>;
    async fn fit(&self, data: &[CoreDataBatch]) -> Result<()>;
    async fn transform(&self, data: &CoreDataBatch) -> Result<CoreDataBatch>;
    async fn inverse_transform(&self, data: &ProcessedData) -> Result<CoreDataBatch>;
}

/// 数据验证接口
#[async_trait]
pub trait DataValidator: Send + Sync {
    async fn validate(&self, data: &CoreDataBatch, schema: &DataSchema) -> Result<ValidationReport>;
    async fn validate_schema(&self, schema: &DataSchema) -> Result<bool>;
    async fn suggest_schema(&self, data: &[CoreDataBatch]) -> Result<DataSchema>;
}

/// 特征工程接口
#[async_trait]
pub trait FeatureEngineer: Send + Sync {
    async fn extract_features(&self, data: &CoreDataBatch) -> Result<FeatureSet>;
    async fn select_features(&self, features: &FeatureSet, selection_method: &str) -> Result<FeatureSet>;
    async fn transform_features(&self, features: &FeatureSet) -> Result<Vec<CoreTensorData>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub statistics: DataStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub field: String,
    pub error_type: String,
    pub message: String,
    pub severity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    pub field: String,
    pub warning_type: String,
    pub message: String,
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStatistics {
    pub total_records: usize,
    pub valid_records: usize,
    pub invalid_records: usize,
    pub field_statistics: HashMap<String, FieldStatistics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldStatistics {
    pub field_name: String,
    pub data_type: String,
    pub null_count: usize,
    pub unique_count: usize,
    pub min_value: Option<String>,
    pub max_value: Option<String>,
    pub mean_value: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSet {
    pub features: HashMap<String, Vec<f32>>,
    pub feature_names: Vec<String>,
    pub feature_types: HashMap<String, String>,
    pub metadata: HashMap<String, String>,
}


