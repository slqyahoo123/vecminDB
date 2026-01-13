use crate::Result;
use crate::Error;
use crate::storage::config::StorageConfig;
use crate::storage::engine::StorageService;
use crate::compat::{ModelArchitecture, ModelParameters};
// remove unused Enhanced* imports
use crate::core::{InferenceResultDetail, InferenceResult};
use crate::compat::TrainingResultDetail; // 使用 compat 模块中的 stub 类型
use crate::interfaces::storage::StorageTransaction;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde_json::Value;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// 内存存储引擎实现
pub struct MemoryStorage {
    config: StorageConfig,
    model_params: Arc<RwLock<HashMap<String, ModelParameters>>>,
    model_arch: Arc<RwLock<HashMap<String, ModelArchitecture>>>,
    training_state: Arc<RwLock<HashMap<String, TrainingStateManager>>>,
    training_results: Arc<RwLock<HashMap<String, HashMap<String, Value>>>>,
    inference_results: Arc<RwLock<HashMap<String, InferenceResult>>>,
    model_info: Arc<RwLock<HashMap<String, crate::storage::models::implementation::ModelInfo>>>,
    model_metrics: Arc<RwLock<HashMap<String, crate::storage::models::implementation::ModelMetrics>>>,
    models: Arc<RwLock<HashMap<String, crate::model::Model>>>,
    // 索引结构用于快速查询
    model_index: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
    inference_index: Arc<RwLock<HashMap<String, Vec<String>>>>, // model_id -> Vec<inference_id>
    // 通用键值存储
    storage: Arc<tokio::sync::RwLock<HashMap<String, Vec<u8>>>>,
}

impl MemoryStorage {
    /// 创建新的内存存储引擎
    pub fn new(config: StorageConfig) -> Result<Self> {
        Ok(Self {
            config,
            model_params: Arc::new(RwLock::new(HashMap::new())),
            model_arch: Arc::new(RwLock::new(HashMap::new())),
            training_state: Arc::new(RwLock::new(HashMap::new())),
            training_results: Arc::new(RwLock::new(HashMap::new())),
            inference_results: Arc::new(RwLock::new(HashMap::new())),
            model_info: Arc::new(RwLock::new(HashMap::new())),
            model_metrics: Arc::new(RwLock::new(HashMap::new())),
            models: Arc::new(RwLock::new(HashMap::new())),
            model_index: Arc::new(RwLock::new(HashMap::new())),
            inference_index: Arc::new(RwLock::new(HashMap::new())),
            storage: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
        })
    }
    
    /// 生成推理ID
    fn generate_inference_id(&self) -> String {
        Uuid::new_v4().to_string()
    }
    
    /// 添加推理索引
    fn add_inference_index(&self, model_id: &str, inference_id: &str) -> Result<()> {
        let mut index = self.inference_index.write().map_err(|e| Error::lock(e.to_string()))?;
        index.entry(model_id.to_string())
            .or_insert_with(Vec::new)
            .push(inference_id.to_string());
        Ok(())
    }
    
    /// 验证模型ID
    fn validate_model_id(&self, model_id: &str) -> Result<()> {
        if model_id.is_empty() {
            return Err(Error::InvalidArgument("模型ID不能为空".to_string()));
        }
        Ok(())
    }
}

impl StorageService for MemoryStorage {
    fn save_model_parameters(&self, model_id: &str, params: &ModelParameters) -> Result<()> {
        self.validate_model_id(model_id)?;
        
        let mut param_storage = self.model_params.write().map_err(|e| Error::lock(e.to_string()))?;
        param_storage.insert(model_id.to_string(), params.clone());
        
        // 更新模型索引
        let mut index = self.model_index.write().map_err(|e| Error::lock(e.to_string()))?;
        index.insert(model_id.to_string(), Utc::now());
        
        Ok(())
    }
    
    fn get_model_parameters(&self, model_id: &str) -> Result<Option<ModelParameters>> {
        self.validate_model_id(model_id)?;
        
        let param_storage = self.model_params.read().map_err(|e| Error::lock(e.to_string()))?;
        Ok(param_storage.get(model_id).cloned())
    }
    
    fn save_model_architecture(&self, model_id: &str, arch: &ModelArchitecture) -> Result<()> {
        self.validate_model_id(model_id)?;
        
        let mut arch_storage = self.model_arch.write().map_err(|e| Error::lock(e.to_string()))?;
        arch_storage.insert(model_id.to_string(), arch.clone());
        
        // 更新模型索引
        let mut index = self.model_index.write().map_err(|e| Error::lock(e.to_string()))?;
        index.insert(model_id.to_string(), Utc::now());
        
        Ok(())
    }
    
    fn get_model_architecture(&self, model_id: &str) -> Result<Option<ModelArchitecture>> {
        self.validate_model_id(model_id)?;
        
        let arch_storage = self.model_arch.read().map_err(|e| Error::lock(e.to_string()))?;
        Ok(arch_storage.get(model_id).cloned())
    }
    
    fn get_training_state(&self, model_id: &str) -> Result<Option<TrainingState>> {
        self.validate_model_id(model_id)?;
        
        let training_storage = self.training_state.read().map_err(|e| Error::lock(e.to_string()))?;
        if let Some(manager) = training_storage.get(model_id) {
            // TrainingStateManager 使用 current_state 字段
            Ok(Some(manager.current_state.clone()))
        } else {
            Ok(None)
        }
    }

    fn get_training_state_manager(&self, model_id: &str) -> Result<Option<TrainingStateManager>> {
        self.validate_model_id(model_id)?;
        
        let training_storage = self.training_state.read().map_err(|e| Error::lock(e.to_string()))?;
        Ok(training_storage.get(model_id).cloned())
    }

    fn save_training_state_manager(&self, model_id: &str, state_manager: &TrainingStateManager) -> Result<()> {
        self.validate_model_id(model_id)?;
        
        let mut training_storage = self.training_state.write().map_err(|e| Error::lock(e.to_string()))?;
        training_storage.insert(model_id.to_string(), state_manager.clone());
        
        Ok(())
    }

    fn update_training_state(&self, model_id: &str, state: &TrainingState) -> Result<()> {
        self.validate_model_id(model_id)?;
        
        let mut training_storage = self.training_state.write().map_err(|e| Error::lock(e.to_string()))?;
        if let Some(manager) = training_storage.get_mut(model_id) {
            // TrainingStateManager 使用 current_state 字段
            manager.current_state = state.clone();
            Ok(())
        } else {
            Err(Error::NotFound(format!("模型 {} 的训练状态管理器未找到", model_id)))
        }
    }

    fn list_training_results(&self, model_id: &str) -> Result<Vec<HashMap<String, Value>>> {
        self.validate_model_id(model_id)?;
        
        let prefix = format!("{}:", model_id);
        let results = self.training_results.read().map_err(|e| Error::lock(e.to_string()))?;
        
        let mut result_list = Vec::new();
        for (key, value) in results.iter() {
            if key.starts_with(&prefix) {
                result_list.push(value.clone());
            }
        }
        
        // 按时间戳排序（如果存在）
        result_list.sort_by(|a, b| {
            let timestamp_a = a.get("timestamp").and_then(|v| v.as_str()).unwrap_or("");
            let timestamp_b = b.get("timestamp").and_then(|v| v.as_str()).unwrap_or("");
            timestamp_b.cmp(timestamp_a) // 降序排列，最新的在前
        });
        
        Ok(result_list)
    }

    fn get_training_result(&self, model_id: &str, training_id: &str) -> Result<Option<HashMap<String, Value>>> {
        self.validate_model_id(model_id)?;
        
        if training_id.is_empty() {
            return Err(Error::InvalidArgument("训练ID不能为空".to_string()));
        }
        
        let key = format!("{}:{}", model_id, training_id);
        let results = self.training_results.read().map_err(|e| Error::lock(e.to_string()))?;
        Ok(results.get(&key).cloned())
    }

    fn save_detailed_training_result(
        &self,
        model_id: &str,
        training_id: &str,
        result: &HashMap<String, Value>,
        detail: &TrainingResultDetail,
    ) -> Result<()> {
        self.validate_model_id(model_id)?;
        
        if training_id.is_empty() {
            return Err(Error::InvalidArgument("训练ID不能为空".to_string()));
        }
        
        let key = format!("{}:{}", model_id, training_id);
        
        // 增强结果信息
        let mut enhanced_result = result.clone();
        enhanced_result.insert("model_id".to_string(), Value::String(model_id.to_string()));
        enhanced_result.insert("training_id".to_string(), Value::String(training_id.to_string()));
        enhanced_result.insert("timestamp".to_string(), Value::String(Utc::now().to_rfc3339()));
        enhanced_result.insert("detail_epochs".to_string(), Value::Number(detail.total_epochs.into()));
        // TrainingResultDetail 没有 batch_size 字段，使用 config 中的信息或默认值
        // enhanced_result.insert("detail_batch_size".to_string(), Value::Number(detail.batch_size.into()));
        
        // TrainingResultDetail 没有 final_loss 字段，使用 loss_history 的最后一个值
        if let Some(&final_loss) = detail.loss_history.last() {
            enhanced_result.insert("final_loss".to_string(), Value::Number(serde_json::Number::from_f64(final_loss).unwrap_or_else(|| serde_json::Number::from(0))));
        }
        
        let mut results = self.training_results.write().map_err(|e| Error::lock(e.to_string()))?;
        results.insert(key, enhanced_result);
        
        Ok(())
    }

    fn list_inference_results(&self, model_id: &str) -> Result<Vec<InferenceResult>> {
        self.validate_model_id(model_id)?;
        
        let index = self.inference_index.read().map_err(|e| Error::lock(e.to_string()))?;
        let results = self.inference_results.read().map_err(|e| Error::lock(e.to_string()))?;
        
        let mut inference_list = Vec::new();
        
        if let Some(inference_ids) = index.get(model_id) {
            for inference_id in inference_ids {
                if let Some(result) = results.get(inference_id) {
                    inference_list.push(result.clone());
                }
            }
        }
        
        // 按创建时间排序（InferenceResult 使用 created_at 字段）
        inference_list.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        
        Ok(inference_list)
    }

    fn get_inference_result(&self, model_id: &str) -> Result<Option<crate::core::InferenceResult>> {
        self.validate_model_id(model_id)?;
        
        let results = self.inference_results.read().map_err(|e| Error::lock(e.to_string()))?;
        
        // 查找该模型的第一个推理结果
        for result in results.values() {
            if result.model_id == model_id {
                return Ok(Some(result.clone()));
            }
        }
        
        Ok(None)
    }
    
    fn save_training_state(&self, model_id: &str, state: &TrainingState) -> Result<()> {
        self.validate_model_id(model_id)?;
        // MemoryStorage 使用 training_state 字段，不是 training_states
        let mut training_storage = self.training_state.write().map_err(|e| Error::lock(e.to_string()))?;
        // 需要创建或更新 TrainingStateManager
        if let Some(manager) = training_storage.get_mut(model_id) {
            manager.current_state = state.clone();
        } else {
            // 如果不存在，创建一个新的 TrainingStateManager
            let mut manager = TrainingStateManager::new(model_id.to_string());
            manager.current_state = state.clone();
            training_storage.insert(model_id.to_string(), manager);
        }
        Ok(())
    }
    
    fn save_training_result(&self, model_id: &str, result: &crate::core::TrainingResultDetail) -> Result<()> {
        self.validate_model_id(model_id)?;
        let mut results = self.training_results.write().map_err(|e| Error::lock(e.to_string()))?;
        // TrainingResultDetail 需要序列化为 Value，然后包装在 HashMap 中
        let result_value = serde_json::to_value(result)
            .map_err(|e| Error::Serialization(format!("序列化训练结果失败: {}", e)))?;
        // 如果 result_value 是 Object，直接使用；否则包装在 HashMap 中
        if let Value::Object(obj) = result_value {
            let mut map = HashMap::new();
            for (k, v) in obj {
                map.insert(k, v);
            }
            results.insert(model_id.to_string(), map);
        } else {
            // 如果不是 Object，创建一个包含 result 的 HashMap
            let mut map = HashMap::new();
            map.insert("result".to_string(), result_value);
            results.insert(model_id.to_string(), map);
        }
        Ok(())
    }
    
    fn save_inference_result(&self, model_id: &str, result: &crate::core::InferenceResult) -> Result<()> {
        self.validate_model_id(model_id)?;
        let mut results = self.inference_results.write().map_err(|e| Error::lock(e.to_string()))?;
        let inference_id = uuid::Uuid::new_v4().to_string();
        results.insert(inference_id, result.clone());
        Ok(())
    }
    
    async fn store(&self, key: &str, value: &[u8]) -> Result<()> {
        let mut storage = self.storage.write().await;
        storage.insert(key.to_string(), value.to_vec());
        Ok(())
    }
    
    async fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let storage = self.storage.read().await;
        Ok(storage.get(key).cloned())
    }
    
    async fn delete(&self, key: &str) -> Result<()> {
        let mut storage = self.storage.write().await;
        storage.remove(key);
        Ok(())
    }
    
    async fn exists(&self, key: &str) -> Result<bool> {
        let storage = self.storage.read().await;
        Ok(storage.contains_key(key))
    }
    
    async fn list_keys(&self, prefix: &str) -> Result<Vec<String>> {
        let storage = self.storage.read().await;
        Ok(storage.keys().filter(|k| k.starts_with(prefix)).cloned().collect())
    }
    
    async fn batch_store(&self, items: &[(String, Vec<u8>)]) -> Result<()> {
        let mut storage = self.storage.write().await;
        for (key, value) in items {
            storage.insert(key.clone(), value.clone());
        }
        Ok(())
    }
    
    async fn batch_retrieve(&self, keys: &[String]) -> Result<HashMap<String, Option<Vec<u8>>>> {
        let storage = self.storage.read().await;
        let mut results = HashMap::new();
        for key in keys {
            results.insert(key.clone(), storage.get(key).cloned());
        }
        Ok(results)
    }
    
    async fn batch_delete(&self, keys: &[String]) -> Result<()> {
        let mut storage = self.storage.write().await;
        for key in keys {
            storage.remove(key);
        }
        Ok(())
    }
    
    async fn transaction(&self) -> Result<Box<dyn StorageTransaction + Send + Sync>> {
        Ok(Box::new(MemoryTransaction::new(self.storage.clone())))
    }
    
    async fn transaction_with_isolation(&self, isolation_level: crate::interfaces::storage::IsolationLevel) -> Result<Box<dyn StorageTransaction + Send + Sync>> {
        // 转换隔离级别类型
        let core_isolation = match isolation_level {
            crate::interfaces::storage::IsolationLevel::ReadUncommitted => crate::core::interfaces::IsolationLevel::ReadUncommitted,
            crate::interfaces::storage::IsolationLevel::ReadCommitted => crate::core::interfaces::IsolationLevel::ReadCommitted,
            crate::interfaces::storage::IsolationLevel::RepeatableRead => crate::core::interfaces::IsolationLevel::RepeatableRead,
            crate::interfaces::storage::IsolationLevel::Serializable => crate::core::interfaces::IsolationLevel::Serializable,
        };
        Ok(Box::new(MemoryTransaction::new_with_isolation(self.storage.clone(), core_isolation)))
    }
    
    async fn get_dataset_size(&self, _dataset_id: &str) -> Result<usize> {
        Ok(0)
    }
    
    async fn get_dataset_chunk(&self, _dataset_id: &str, _start: usize, _end: usize) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }
    
    fn get_specific_inference_result(&self, model_id: &str, inference_id: &str) -> Result<Option<crate::core::InferenceResult>> {
        self.validate_model_id(model_id)?;
        let results = self.inference_results.read().map_err(|e| Error::lock(e.to_string()))?;
        if let Some(result) = results.get(inference_id) {
            if result.model_id == model_id {
                Ok(Some(result.clone()))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    fn save_detailed_inference_result(
        &self,
        model_id: &str,
        inference_id: &str,
        result: &InferenceResult,
        detail: &InferenceResultDetail,
        processing_time: u64,
    ) -> Result<()> {
        self.validate_model_id(model_id)?;
        
        if inference_id.is_empty() {
            return Err(Error::InvalidArgument("推理ID不能为空".to_string()));
        }
        
        // 创建增强的推理结果
        // InferenceResult 没有 processing_time 和 metadata 字段，使用 created_at 和 processing_time_ms
        // InferenceResultDetail 没有 input_size 和 output_size 字段
        let enhanced_result = result.clone();
        
        // 保存结果
        let mut results = self.inference_results.write().map_err(|e| Error::lock(e.to_string()))?;
        results.insert(inference_id.to_string(), enhanced_result);
        
        // 更新索引
        self.add_inference_index(model_id, inference_id)?;
        
        Ok(())
    }

    fn save_model_info(&self, model_id: &str, info: &crate::storage::models::implementation::ModelInfo) -> Result<()> {
        self.validate_model_id(model_id)?;
        
        let mut info_storage = self.model_info.write().map_err(|e| Error::lock(e.to_string()))?;
        info_storage.insert(model_id.to_string(), info.clone());
        
        // 更新模型索引
        let mut index = self.model_index.write().map_err(|e| Error::lock(e.to_string()))?;
        index.insert(model_id.to_string(), Utc::now());
        
        Ok(())
    }

    fn get_model_info(&self, model_id: &str) -> Result<Option<crate::storage::models::implementation::ModelInfo>> {
        self.validate_model_id(model_id)?;
        
        let info_storage = self.model_info.read().map_err(|e| Error::lock(e.to_string()))?;
        Ok(info_storage.get(model_id).cloned())
    }

    fn save_model_metrics(&self, model_id: &str, metrics: &crate::storage::models::implementation::ModelMetrics) -> Result<()> {
        self.validate_model_id(model_id)?;
        
        let mut metrics_storage = self.model_metrics.write().map_err(|e| Error::lock(e.to_string()))?;
        metrics_storage.insert(model_id.to_string(), metrics.clone());
        
        Ok(())
    }

    fn get_model_metrics(&self, model_id: &str) -> Result<Option<crate::storage::models::implementation::ModelMetrics>> {
        self.validate_model_id(model_id)?;
        
        let metrics_storage = self.model_metrics.read().map_err(|e| Error::lock(e.to_string()))?;
        Ok(metrics_storage.get(model_id).cloned())
    }

    fn get_model(&self, model_id: &str) -> Result<Option<crate::model::Model>> {
        self.validate_model_id(model_id)?;
        
        let models = self.models.read().map_err(|e| Error::lock(e.to_string()))?;
        Ok(models.get(model_id).cloned())
    }

    fn save_model(&self, model_id: &str, model: &crate::model::Model) -> Result<()> {
        self.validate_model_id(model_id)?;
        
        let mut models = self.models.write().map_err(|e| Error::lock(e.to_string()))?;
        models.insert(model_id.to_string(), model.clone());
        
        // 更新模型索引
        let mut index = self.model_index.write().map_err(|e| Error::lock(e.to_string()))?;
        index.insert(model_id.to_string(), Utc::now());
        
        Ok(())
    }

    fn model_exists(&self, model_id: &str) -> Result<bool> {
        self.validate_model_id(model_id)?;
        
        let models = self.models.read().map_err(|e| Error::lock(e.to_string()))?;
        let model_params = self.model_params.read().map_err(|e| Error::lock(e.to_string()))?;
        let model_arch = self.model_arch.read().map_err(|e| Error::lock(e.to_string()))?;
        
        // 检查是否在任何一个存储中存在
        let exists = models.contains_key(model_id) || 
                    model_params.contains_key(model_id) || 
                    model_arch.contains_key(model_id);
        
        Ok(exists)
    }

    fn has_model(&self, model_id: &str) -> Result<bool> {
        self.model_exists(model_id)
    }

    fn get_dataset(&self, dataset_id: &str) -> Result<Option<crate::data::DataBatch>> {
        // 内存存储中暂时返回None，因为内存存储主要用于模型数据
        // 数据集数据通常存储在持久化存储中
        Ok(None)
    }
}

// Memory 事务实现
struct MemoryTransaction {
    storage: Arc<tokio::sync::RwLock<HashMap<String, Vec<u8>>>>,
    id: String,
    state: crate::core::interfaces::storage_interface::TransactionState,
    ops: Vec<MemoryTxOp>,
    isolation_level: crate::interfaces::storage::IsolationLevel,
}

enum MemoryTxOp {
    Store { key: String, value: Vec<u8> },
    Delete { key: String },
}

impl MemoryTransaction {
    fn new(storage: Arc<tokio::sync::RwLock<HashMap<String, Vec<u8>>>>) -> Self {
        Self {
            storage,
            id: Uuid::new_v4().to_string(),
            state: crate::core::interfaces::storage_interface::TransactionState::Active,
            ops: Vec::new(),
            isolation_level: crate::interfaces::storage::IsolationLevel::default(),
        }
    }
    
    fn new_with_isolation(storage: Arc<tokio::sync::RwLock<HashMap<String, Vec<u8>>>>, isolation_level: crate::core::interfaces::IsolationLevel) -> Self {
        // 转换隔离级别类型
        let isolation = match isolation_level {
            crate::core::interfaces::IsolationLevel::ReadUncommitted => crate::interfaces::storage::IsolationLevel::ReadUncommitted,
            crate::core::interfaces::IsolationLevel::ReadCommitted => crate::interfaces::storage::IsolationLevel::ReadCommitted,
            crate::core::interfaces::IsolationLevel::RepeatableRead => crate::interfaces::storage::IsolationLevel::RepeatableRead,
            crate::core::interfaces::IsolationLevel::Serializable => crate::interfaces::storage::IsolationLevel::Serializable,
        };
        Self {
            storage,
            id: Uuid::new_v4().to_string(),
            state: crate::core::interfaces::storage_interface::TransactionState::Active,
            ops: Vec::new(),
            isolation_level: isolation,
        }
    }
}

impl StorageTransaction for MemoryTransaction {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn state(&self) -> crate::interfaces::storage::TransactionState {
        match self.state {
            crate::core::interfaces::storage_interface::TransactionState::Active => crate::interfaces::storage::TransactionState::Active,
            crate::core::interfaces::storage_interface::TransactionState::Committed => crate::interfaces::storage::TransactionState::Committed,
            crate::core::interfaces::storage_interface::TransactionState::RolledBack => crate::interfaces::storage::TransactionState::RolledBack,
        }
    }
    
    fn isolation_level(&self) -> crate::interfaces::storage::IsolationLevel {
        self.isolation_level
    }
    
    fn commit(self: Box<Self>) -> Result<()> {
        if self.state != crate::core::interfaces::storage_interface::TransactionState::Active {
            return Err(Error::Transaction("事务非活跃状态，无法提交".to_string()));
        }
        
        let storage = self.storage.clone();
        let ops = self.ops;
        tokio::runtime::Handle::current().block_on(async move {
            let mut storage_guard = storage.write().await;
            for op in ops {
                match op {
                    MemoryTxOp::Store { key, value } => {
                        storage_guard.insert(key, value);
                    }
                    MemoryTxOp::Delete { key } => {
                        storage_guard.remove(&key);
                    }
                }
            }
            Ok::<(), Error>(())
        })
    }
    
    fn rollback(self: Box<Self>) -> Result<()> {
        if self.state != crate::core::interfaces::storage_interface::TransactionState::Active {
            return Err(Error::Transaction("事务非活跃状态，无法回滚".to_string()));
        }
        Ok(())
    }
    
    fn put(&mut self, key: &[u8], value: &[u8]) -> Result<()> {
        if self.state != crate::core::interfaces::storage_interface::TransactionState::Active {
            return Err(Error::Transaction("事务非活跃状态，无法执行操作".to_string()));
        }
        
        let key_str = String::from_utf8(key.to_vec()).map_err(|e| Error::InvalidInput(format!("无效的键: {}", e)))?;
        self.ops.push(MemoryTxOp::Store {
            key: key_str,
            value: value.to_vec(),
        });
        Ok(())
    }
    
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let key_str = String::from_utf8(key.to_vec()).map_err(|e| Error::InvalidInput(format!("无效的键: {}", e)))?;
        
        // 先检查事务中的操作
        for op in self.ops.iter().rev() {
            match op {
                MemoryTxOp::Store { key: tx_key, value } if tx_key == &key_str => {
                    return Ok(Some(value.clone()));
                }
                MemoryTxOp::Delete { key: tx_key } if tx_key == &key_str => {
                    return Ok(None);
                }
                _ => {}
            }
        }
        
        // 如果事务中没有相关操作，从存储读取
        let storage = self.storage.clone();
        let key_str = key_str;
        tokio::runtime::Handle::current().block_on(async move {
            let storage_guard = storage.read().await;
            Ok(storage_guard.get(&key_str).cloned())
        })
    }
    
    fn delete(&mut self, key: &[u8]) -> Result<()> {
        if self.state != crate::core::interfaces::storage_interface::TransactionState::Active {
            return Err(Error::Transaction("事务非活跃状态，无法执行操作".to_string()));
        }
        
        let key_str = String::from_utf8(key.to_vec()).map_err(|e| Error::InvalidInput(format!("无效的键: {}", e)))?;
        self.ops.push(MemoryTxOp::Delete {
            key: key_str,
        });
        Ok(())
    }
} 