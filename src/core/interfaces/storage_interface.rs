use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use crate::error::Result;

/// 存储接口
#[async_trait]
pub trait StorageService: Send + Sync {
    async fn store(&self, key: &str, value: &[u8]) -> Result<()>;
    async fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>>;
    async fn delete(&self, key: &str) -> Result<()>;
    async fn exists(&self, key: &str) -> Result<bool>;
    async fn list_keys(&self, prefix: &str) -> Result<Vec<String>>;
    async fn batch_store(&self, items: Vec<(String, Vec<u8>)>) -> Result<()>;
    async fn batch_retrieve(&self, keys: &[String]) -> Result<Vec<Option<Vec<u8>>>>;
    async fn batch_delete(&self, keys: &[String]) -> Result<()>;
    fn transaction(&self) -> Result<Box<dyn StorageTransaction>>;
    fn transaction_with_isolation(&self, isolation_level: IsolationLevel) -> Result<Box<dyn StorageTransaction>>;
    async fn get_dataset_size(&self, dataset_id: &str) -> Result<usize>;
    async fn get_dataset_chunk(&self, dataset_id: &str, start: usize, end: usize) -> Result<Vec<u8>>;
    
    /// 保存模型参数
    async fn save_model_parameters(&self, model_id: &str, params: &crate::model::parameters::ModelParameters) -> Result<()>;
    
    /// 获取模型参数
    async fn get_model_parameters(&self, model_id: &str) -> Result<Option<crate::model::parameters::ModelParameters>>;
    
    /// 保存模型架构
    async fn save_model_architecture(&self, model_id: &str, arch: &crate::model::ModelArchitecture) -> Result<()>;
    
    /// 获取模型架构
    async fn get_model_architecture(&self, model_id: &str) -> Result<Option<crate::model::ModelArchitecture>>;
    
    // 训练状态相关方法已移除 - 向量数据库系统不需要训练功能
    // async fn get_training_state(&self, model_id: &str) -> Result<Option<crate::model::state::ModelState>>;
    // async fn save_training_state(&self, model_id: &str, state: &crate::model::parameters::TrainingState) -> Result<()>;
    // async fn get_training_state_manager(&self, model_id: &str) -> Result<Option<crate::model::parameters::TrainingStateManager>>;
    // async fn save_training_state_manager(&self, model_id: &str, manager: &crate::model::parameters::TrainingStateManager) -> Result<()>;
    // async fn update_training_state(&self, model_id: &str, state: &crate::model::state::ModelState) -> Result<()>;
    
    // 训练结果相关方法已移除 - 向量数据库系统不需要训练功能
    // async fn list_training_results(&self, model_id: &str) -> Result<Vec<String>>;
    // async fn get_training_result(&self, result_id: &str) -> Result<Option<crate::training::types::TrainingResult>>;
    // async fn save_training_result(&self, model_id: &str, result: &crate::core::results::TrainingResultDetail) -> Result<()>;
    // async fn save_detailed_training_result(&self, result_id: &str, result: &crate::training::types::TrainingResult) -> Result<()>;
    
    /// 列出推理结果
    async fn list_inference_results(&self, model_id: &str) -> Result<Vec<String>>;
    
    /// 获取推理结果
    async fn get_inference_result(&self, result_id: &str) -> Result<Option<crate::core::results::InferenceResult>>;
    
    /// 保存推理结果
    async fn save_inference_result(&self, model_id: &str, result: &crate::core::results::InferenceResult) -> Result<()>;
    
    /// 保存详细推理结果
    async fn save_detailed_inference_result(&self, result_id: &str, result: &crate::core::results::InferenceResult) -> Result<()>;
    
    /// 保存模型信息
    async fn save_model_info(&self, model_id: &str, info: &crate::core::types::ModelInfo) -> Result<()>;
    
    /// 获取模型信息
    async fn get_model_info(&self, model_id: &str) -> Result<Option<crate::core::types::ModelInfo>>;
    
    /// 保存模型指标
    async fn save_model_metrics(&self, model_id: &str, metrics: &crate::storage::models::implementation::ModelMetrics) -> Result<()>;
    
    /// 获取模型指标
    async fn get_model_metrics(&self, model_id: &str) -> Result<Option<crate::storage::models::implementation::ModelMetrics>>;
    
    /// 获取模型
    async fn get_model(&self, model_id: &str) -> Result<Option<crate::model::Model>>;
    
    /// 保存模型
    async fn save_model(&self, model_id: &str, model: &crate::model::Model) -> Result<()>;
    
    /// 检查模型是否存在
    async fn model_exists(&self, model_id: &str) -> Result<bool>;
    
    /// 检查是否有模型
    async fn has_model(&self, model_id: &str) -> Result<bool>;
    
    /// 获取数据集
    async fn get_dataset(&self, dataset_id: &str) -> Result<Option<crate::data::loader::types::DataSchema>>;
}

/// 对象存储接口
#[async_trait]
pub trait ObjectStorageService: Send + Sync {
    async fn store_object<T: Serialize + Send>(&self, key: &str, object: &T) -> Result<()>;
    async fn retrieve_object<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Result<Option<T>>;
    async fn delete_object(&self, key: &str) -> Result<()>;
    async fn list_objects(&self, prefix: &str) -> Result<Vec<ObjectMetadata>>;
}

/// 存储事务
pub trait StorageTransaction: Send + Sync {
    fn commit(&mut self) -> Result<()>;
    fn rollback(&mut self) -> Result<()>;
    fn store(&mut self, key: &str, value: &[u8]) -> Result<()>;
    fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>>;
    fn delete(&mut self, key: &str) -> Result<()>;
    fn exists(&self, key: &str) -> Result<bool>;
    fn get_state(&self) -> TransactionState;
    fn get_id(&self) -> &str;
}

pub trait StorageInterface: Send + Sync {
    fn transaction(&self) -> Result<Box<dyn StorageTransaction>>;
    fn transaction_with_isolation(&self, isolation_level: IsolationLevel) -> Result<Box<dyn StorageTransaction>>;
    fn store(&self, key: &str, value: &[u8]) -> Result<()>;
    fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>>;
    fn delete(&self, key: &str) -> Result<()>;
    fn exists(&self, key: &str) -> Result<bool>;
    fn list_keys(&self, prefix: &str) -> Result<Vec<String>>;
    fn check_disk_space(&self) -> Result<DiskSpaceInfo>;
    fn get_active_connections_count(&self) -> Result<usize>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
}

impl Default for IsolationLevel {
    fn default() -> Self { IsolationLevel::ReadCommitted }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionState {
    Active,
    Committed,
    RolledBack,
    PartiallyFailed,
    Prepared,
}

impl Default for TransactionState {
    fn default() -> Self { TransactionState::Active }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectMetadata {
    pub key: String,
    pub size: usize,
    pub content_type: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskSpaceInfo {
    pub total_space: u64,
    pub available_space: u64,
    pub used_space: u64,
    pub usage_percentage: f32,
}

impl ObjectMetadata {
    pub fn new(key: String, size: usize) -> Self {
        let now = chrono::Utc::now();
        Self { key, size, content_type: None, created_at: now, updated_at: now, metadata: HashMap::new() }
    }
}


