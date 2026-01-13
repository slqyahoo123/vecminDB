use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use serde_json::Value;
use log::{debug, info};

use crate::Result;
use crate::Error;
use crate::compat::{Model, ModelArchitecture, ModelParameters};
use crate::storage::models::{ModelInfo, ModelMetrics};
use crate::core::{InferenceResultDetail, InferenceResult};
use crate::compat::TrainingResultDetail; // 使用 compat 模块中的 stub 类型

/// 增强的训练结果
/// 
/// 包含基础训练结果和详细指标的完整训练结果信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedTrainingResult {
    /// 基础训练结果数据
    pub base_result: HashMap<String, Value>,
    /// 详细训练指标
    pub detailed_metrics: TrainingResultDetail,
    /// 创建时间戳
    pub creation_time: u64,
    /// 最后更新时间戳
    pub last_updated: u64,
}

impl EnhancedTrainingResult {
    /// 创建新的增强训练结果
    pub fn new(
        base_result: HashMap<String, Value>,
        detailed_metrics: TrainingResultDetail,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
            
        Self {
            base_result,
            detailed_metrics,
            creation_time: now,
            last_updated: now,
        }
    }
    
    /// 更新训练结果
    pub fn update(&mut self, base_result: HashMap<String, Value>, detailed_metrics: TrainingResultDetail) {
        self.base_result = base_result;
        self.detailed_metrics = detailed_metrics;
        self.last_updated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
    }
}

/// 增强的推理结果
/// 
/// 包含基础推理结果和详细指标的完整推理结果信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedInferenceResult {
    /// 基础推理结果
    pub base_result: InferenceResult,
    /// 详细推理指标
    pub detailed_metrics: InferenceResultDetail,
    /// 创建时间戳
    pub creation_time: u64,
    /// 处理时间（毫秒）
    pub processing_time: u64,
}

impl EnhancedInferenceResult {
    /// 创建新的增强推理结果
    pub fn new(
        base_result: InferenceResult,
        detailed_metrics: InferenceResultDetail,
        processing_time: u64,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
            
        Self {
            base_result,
            detailed_metrics,
            creation_time: now,
            processing_time,
        }
    }
}

/// 模型存储服务
/// 
/// 提供模型相关数据的存储和检索功能，包括模型参数、架构、训练状态等
#[derive(Clone)]
pub struct ModelStorageService {
    /// 底层数据库连接
    db: Arc<RwLock<sled::Db>>,
}

impl ModelStorageService {
    /// 创建新的模型存储服务
    /// 
    /// # 参数
    /// - `db`: Sled数据库实例
    pub fn new(db: Arc<RwLock<sled::Db>>) -> Self {
        Self { db }
    }
    
    /// 保存模型参数
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `params`: 模型参数对象
    pub fn save_model_parameters(&self, model_id: &str, params: &ModelParameters) -> Result<()> {
        let key = format!("model:{}:parameters", model_id);
        let value = bincode::serialize(params)
            .map_err(|e| Error::serialization(format!("序列化模型参数失败: {}", e)))?;
            
        let db = self.db.blocking_read();
        db.insert(key.as_bytes(), value)
            .map_err(|e| Error::storage(format!("保存模型参数失败: {}", e)))?;
            
        debug!("已保存模型 {} 的参数", model_id);
        Ok(())
    }
    
    /// 获取模型参数
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回模型参数，如果不存在则返回None
    pub fn get_model_parameters(&self, model_id: &str) -> Result<Option<ModelParameters>> {
        let key = format!("model:{}:parameters", model_id);
        let db = self.db.blocking_read();
        
        if let Some(value) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取模型参数失败: {}", e)))? {
            
            let params = bincode::deserialize(&value)
                .map_err(|e| Error::serialization(format!("反序列化模型参数失败: {}", e)))?;
            Ok(Some(params))
        } else {
            Ok(None)
        }
    }
    
    /// 保存模型架构
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `arch`: 模型架构对象
    pub fn save_model_architecture(&self, model_id: &str, arch: &ModelArchitecture) -> Result<()> {
        let key = format!("model:{}:architecture", model_id);
        let value = bincode::serialize(arch)
            .map_err(|e| Error::serialization(format!("序列化模型架构失败: {}", e)))?;
            
        let db = self.db.blocking_read();
        db.insert(key.as_bytes(), value)
            .map_err(|e| Error::storage(format!("保存模型架构失败: {}", e)))?;
            
        debug!("已保存模型 {} 的架构", model_id);
        Ok(())
    }
    
    /// 获取模型架构
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回模型架构，如果不存在则返回None
    pub fn get_model_architecture(&self, model_id: &str) -> Result<Option<ModelArchitecture>> {
        let key = format!("model:{}:architecture", model_id);
        let db = self.db.blocking_read();
        
        if let Some(value) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取模型架构失败: {}", e)))? {
            
            let arch = bincode::deserialize(&value)
                .map_err(|e| Error::serialization(format!("反序列化模型架构失败: {}", e)))?;
            Ok(Some(arch))
        } else {
            Ok(None)
        }
    }
    
    /// 获取训练状态
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回训练状态，如果不存在则返回None
    pub fn get_training_state(&self, model_id: &str) -> Result<Option<TrainingState>> {
        let key = format!("model:{}:training_state", model_id);
        let db = self.db.blocking_read();
        
        if let Some(value) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取训练状态失败: {}", e)))? {
            
            let state = bincode::deserialize(&value)
                .map_err(|e| Error::serialization(format!("反序列化训练状态失败: {}", e)))?;
            Ok(Some(state))
        } else {
            Ok(None)
        }
    }
    
    /// 获取训练状态管理器
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回训练状态管理器，如果不存在则返回None
    pub fn get_training_state_manager(&self, model_id: &str) -> Result<Option<TrainingStateManager>> {
        let key = format!("model:{}:training_state_manager", model_id);
        let db = self.db.blocking_read();
        
        if let Some(value) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取训练状态管理器失败: {}", e)))? {
            
            let manager = bincode::deserialize(&value)
                .map_err(|e| Error::serialization(format!("反序列化训练状态管理器失败: {}", e)))?;
            Ok(Some(manager))
        } else {
            Ok(None)
        }
    }
    
    /// 保存训练状态管理器
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `state_manager`: 训练状态管理器对象
    pub fn save_training_state_manager(&self, model_id: &str, state_manager: &TrainingStateManager) -> Result<()> {
        let key = format!("model:{}:training_state_manager", model_id);
        let value = bincode::serialize(state_manager)
            .map_err(|e| Error::serialization(format!("序列化训练状态管理器失败: {}", e)))?;
            
        let db = self.db.blocking_read();
        db.insert(key.as_bytes(), value)
            .map_err(|e| Error::storage(format!("保存训练状态管理器失败: {}", e)))?;
            
        debug!("已保存模型 {} 的训练状态管理器", model_id);
        Ok(())
    }
    
    /// 更新训练状态
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `state`: 新的训练状态
    pub fn update_training_state(&self, model_id: &str, state: &TrainingState) -> Result<()> {
        let key = format!("model:{}:training_state", model_id);
        let value = bincode::serialize(state)
            .map_err(|e| Error::serialization(format!("序列化训练状态失败: {}", e)))?;
            
        let db = self.db.blocking_read();
        db.insert(key.as_bytes(), value)
            .map_err(|e| Error::storage(format!("更新训练状态失败: {}", e)))?;
            
        debug!("已更新模型 {} 的训练状态", model_id);
        Ok(())
    }
    
    /// 列出训练结果
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回训练结果列表
    pub fn list_training_results(&self, model_id: &str) -> Result<Vec<HashMap<String, Value>>> {
        let prefix = format!("model:{}:training_results:", model_id);
        let db = self.db.blocking_read();
        
        let mut results = Vec::new();
        for item in db.scan_prefix(prefix.as_bytes()) {
            let (_, value) = item.map_err(|e| Error::storage(format!("扫描训练结果失败: {}", e)))?;
            let result: HashMap<String, Value> = bincode::deserialize(&value)
                .map_err(|e| Error::serialization(format!("反序列化训练结果失败: {}", e)))?;
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// 获取训练结果
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `training_id`: 训练ID
    /// 
    /// # 返回值
    /// 返回指定的训练结果，如果不存在则返回None
    pub fn get_training_result(&self, model_id: &str, training_id: &str) -> Result<Option<HashMap<String, Value>>> {
        let key = format!("model:{}:training_results:{}", model_id, training_id);
        let db = self.db.blocking_read();
        
        if let Some(value) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取训练结果失败: {}", e)))? {
            
            let result = bincode::deserialize(&value)
                .map_err(|e| Error::serialization(format!("反序列化训练结果失败: {}", e)))?;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }
    
    /// 保存详细训练结果
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `training_id`: 训练ID
    /// - `result`: 基础训练结果
    /// - `detail`: 详细训练指标
    pub fn save_detailed_training_result(
        &self,
        model_id: &str,
        training_id: &str,
        result: &HashMap<String, Value>,
        detail: &TrainingResultDetail,
    ) -> Result<()> {
        let enhanced_result = EnhancedTrainingResult::new(result.clone(), detail.clone());
        
        // 保存基础结果
        let base_key = format!("model:{}:training_results:{}", model_id, training_id);
        let base_value = bincode::serialize(result)
            .map_err(|e| Error::serialization(format!("序列化训练结果失败: {}", e)))?;
        
        // 保存增强结果
        let enhanced_key = format!("model:{}:enhanced_training_results:{}", model_id, training_id);
        let enhanced_value = bincode::serialize(&enhanced_result)
            .map_err(|e| Error::serialization(format!("序列化增强训练结果失败: {}", e)))?;
        
        let db = self.db.blocking_read();
        
        db.insert(base_key.as_bytes(), base_value)
            .map_err(|e| Error::storage(format!("保存训练结果失败: {}", e)))?;
        db.insert(enhanced_key.as_bytes(), enhanced_value)
            .map_err(|e| Error::storage(format!("保存增强训练结果失败: {}", e)))?;
        
        info!("已保存模型 {} 的训练结果 {}", model_id, training_id);
        Ok(())
    }
    
    /// 获取详细训练结果
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `training_id`: 训练ID
    /// 
    /// # 返回值
    /// 返回增强的训练结果，如果不存在则返回None
    pub fn get_detailed_training_result(&self, model_id: &str, training_id: &str) -> Result<Option<EnhancedTrainingResult>> {
        let key = format!("model:{}:enhanced_training_results:{}", model_id, training_id);
        let db = self.db.blocking_read();
        
        if let Some(value) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取详细训练结果失败: {}", e)))? {
            
            let result = bincode::deserialize(&value)
                .map_err(|e| Error::serialization(format!("反序列化详细训练结果失败: {}", e)))?;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }
    
    /// 列出推理结果
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回推理结果列表
    pub fn list_inference_results(&self, model_id: &str) -> Result<Vec<InferenceResult>> {
        let prefix = format!("model:{}:inference_results:", model_id);
        let db = self.db.blocking_read();
        
        let mut results = Vec::new();
        for item in db.scan_prefix(prefix.as_bytes()) {
            let (_, value) = item.map_err(|e| Error::storage(format!("扫描推理结果失败: {}", e)))?;
            let result: InferenceResult = bincode::deserialize(&value)
                .map_err(|e| Error::serialization(format!("反序列化推理结果失败: {}", e)))?;
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// 获取推理结果
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `inference_id`: 推理ID
    /// 
    /// # 返回值
    /// 返回指定的推理结果，如果不存在则返回None
    pub fn get_inference_result(&self, model_id: &str, inference_id: &str) -> Result<Option<InferenceResult>> {
        let key = format!("model:{}:inference_results:{}", model_id, inference_id);
        let db = self.db.blocking_read();
        
        if let Some(value) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取推理结果失败: {}", e)))? {
            
            let result = bincode::deserialize(&value)
                .map_err(|e| Error::serialization(format!("反序列化推理结果失败: {}", e)))?;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }
    
    /// 保存详细推理结果
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `inference_id`: 推理ID
    /// - `result`: 基础推理结果
    /// - `detail`: 详细推理指标
    /// - `processing_time`: 处理时间（毫秒）
    pub fn save_detailed_inference_result(
        &self,
        model_id: &str,
        inference_id: &str,
        result: &InferenceResult,
        detail: &InferenceResultDetail,
        processing_time: u64,
    ) -> Result<()> {
        let enhanced_result = EnhancedInferenceResult::new(result.clone(), detail.clone(), processing_time);
        
        // 保存基础结果
        let base_key = format!("model:{}:inference_results:{}", model_id, inference_id);
        let base_value = bincode::serialize(result)
            .map_err(|e| Error::serialization(format!("序列化推理结果失败: {}", e)))?;
        
        // 保存增强结果
        let enhanced_key = format!("model:{}:enhanced_inference_results:{}", model_id, inference_id);
        let enhanced_value = bincode::serialize(&enhanced_result)
            .map_err(|e| Error::serialization(format!("序列化增强推理结果失败: {}", e)))?;
        
        let db = self.db.blocking_read();
        
        db.insert(base_key.as_bytes(), base_value)
            .map_err(|e| Error::storage(format!("保存推理结果失败: {}", e)))?;
        db.insert(enhanced_key.as_bytes(), enhanced_value)
            .map_err(|e| Error::storage(format!("保存增强推理结果失败: {}", e)))?;
        
        info!("已保存模型 {} 的推理结果 {}", model_id, inference_id);
        Ok(())
    }
    
    /// 获取详细推理结果
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `inference_id`: 推理ID
    /// 
    /// # 返回值
    /// 返回增强的推理结果，如果不存在则返回None
    pub fn get_detailed_inference_result(&self, model_id: &str, inference_id: &str) -> Result<Option<EnhancedInferenceResult>> {
        let key = format!("model:{}:enhanced_inference_results:{}", model_id, inference_id);
        let db = self.db.blocking_read();
        
        if let Some(value) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取详细推理结果失败: {}", e)))? {
            
            let result = bincode::deserialize(&value)
                .map_err(|e| Error::serialization(format!("反序列化详细推理结果失败: {}", e)))?;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }
    
    /// 保存模型信息
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `info`: 模型信息对象
    pub fn save_model_info(&self, model_id: &str, info: &ModelInfo) -> Result<()> {
        let key = format!("model:{}:info", model_id);
        let value = bincode::serialize(info)
            .map_err(|e| Error::serialization(format!("序列化模型信息失败: {}", e)))?;
            
        let db = self.db.blocking_read();
        db.insert(key.as_bytes(), value)
            .map_err(|e| Error::storage(format!("保存模型信息失败: {}", e)))?;
            
        debug!("已保存模型 {} 的信息", model_id);
        Ok(())
    }
    
    /// 获取模型信息
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回模型信息，如果不存在则返回None
    pub fn get_model_info(&self, model_id: &str) -> Result<Option<ModelInfo>> {
        let key = format!("model:{}:info", model_id);
        let db = self.db.blocking_read();
        
        if let Some(value) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取模型信息失败: {}", e)))? {
            
            let info = bincode::deserialize(&value)
                .map_err(|e| Error::serialization(format!("反序列化模型信息失败: {}", e)))?;
            Ok(Some(info))
        } else {
            Ok(None)
        }
    }
    
    /// 保存模型指标
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `metrics`: 模型指标对象
    pub fn save_model_metrics(&self, model_id: &str, metrics: &ModelMetrics) -> Result<()> {
        let key = format!("model:{}:metrics", model_id);
        let value = bincode::serialize(metrics)
            .map_err(|e| Error::serialization(format!("序列化模型指标失败: {}", e)))?;
            
        let db = self.db.blocking_read();
        db.insert(key.as_bytes(), value)
            .map_err(|e| Error::storage(format!("保存模型指标失败: {}", e)))?;
            
        debug!("已保存模型 {} 的指标", model_id);
        Ok(())
    }
    
    /// 获取模型指标
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回模型指标，如果不存在则返回None
    pub fn get_model_metrics(&self, model_id: &str) -> Result<Option<ModelMetrics>> {
        let key = format!("model:{}:metrics", model_id);
        let db = self.db.blocking_read();
        
        if let Some(value) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取模型指标失败: {}", e)))? {
            
            let metrics = bincode::deserialize(&value)
                .map_err(|e| Error::serialization(format!("反序列化模型指标失败: {}", e)))?;
            Ok(Some(metrics))
        } else {
            Ok(None)
        }
    }
    
    /// 获取模型
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回完整的模型对象，如果不存在则返回None
    pub fn get_model(&self, model_id: &str) -> Result<Option<Model>> {
        let key = format!("model:{}", model_id);
        let db = self.db.blocking_read();
        
        if let Some(value) = db.get(key.as_bytes())
            .map_err(|e| Error::storage(format!("获取模型失败: {}", e)))? {
            
            let model = bincode::deserialize(&value)
                .map_err(|e| Error::serialization(format!("反序列化模型失败: {}", e)))?;
            Ok(Some(model))
        } else {
            Ok(None)
        }
    }
    
    /// 保存模型
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `model`: 模型对象
    pub fn save_model(&self, model_id: &str, model: &Model) -> Result<()> {
        let key = format!("model:{}", model_id);
        let value = bincode::serialize(model)
            .map_err(|e| Error::serialization(format!("序列化模型失败: {}", e)))?;
            
        let db = self.db.blocking_read();
        db.insert(key.as_bytes(), value)
            .map_err(|e| Error::storage(format!("保存模型失败: {}", e)))?;
            
        info!("已保存模型 {}", model_id);
        Ok(())
    }
    
    /// 检查模型是否存在
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回模型是否存在
    pub fn model_exists(&self, model_id: &str) -> Result<bool> {
        let key = format!("model:{}", model_id);
        let db = self.db.blocking_read();
        
        Ok(db.contains_key(key.as_bytes())
            .map_err(|e| Error::storage(format!("检查模型存在失败: {}", e)))?)
    }
    
    /// 检查是否有模型
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回是否拥有该模型
    pub fn has_model(&self, model_id: &str) -> Result<bool> {
        self.model_exists(model_id)
    }

    /// 统计模型数量
    /// 
    /// # 返回值
    /// 返回存储的模型总数
    pub fn count_models(&self) -> Result<usize> {
        let db = self.db.blocking_read();
        let count = db.scan_prefix("model:").count();
        Ok(count)
    }

    /// 获取所有模型列表
    /// 
    /// # 返回值
    /// 返回所有模型的列表
    pub async fn list_models(&self) -> Result<Vec<crate::model::Model>> {
        let db = self.db.read().await;
        
        let mut models = Vec::new();
        let prefix = "model:";
        
        for result in db.scan_prefix(prefix.as_bytes()) {
            let (key, _value) = result.map_err(|e| Error::storage(format!("扫描模型数据失败: {}", e)))?;
            let key_str = String::from_utf8_lossy(&key);
            
            if key_str.contains(":parameters") {
                // 提取模型ID
                if let Some(model_id) = key_str.strip_prefix("model:").and_then(|s| s.strip_suffix(":parameters")) {
                    // 尝试获取完整的模型信息
                    if let Ok(Some(model)) = self.get_model(model_id) {
                        models.push(model);
                    }
                }
            }
        }
        
        Ok(models)
    }

    /// 获取训练指标历史
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// 
    /// # 返回值
    /// 返回训练指标历史列表
    pub async fn get_training_metrics_history(&self, model_id: &str) -> Result<Vec<crate::training::types::TrainingMetrics>> {
        let db = self.db.read().await;
        
        let mut metrics_history = Vec::new();
        let prefix = format!("metrics:{}:", model_id);
        
        for result in db.scan_prefix(prefix.as_bytes()) {
            let (_key, value) = result.map_err(|e| Error::storage(format!("扫描训练指标失败: {}", e)))?;
            
            if let Ok(metrics) = bincode::deserialize::<crate::training::types::TrainingMetrics>(&value) {
                metrics_history.push(metrics);
            }
        }
        
        // 按时间戳排序
        metrics_history.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        
        Ok(metrics_history)
    }

    /// 记录训练指标
    /// 
    /// # 参数
    /// - `model_id`: 模型唯一标识符
    /// - `metrics`: 训练指标
    /// 
    /// # 返回值
    /// 返回操作结果
    pub async fn record_training_metrics(&self, model_id: &str, metrics: &crate::training::types::TrainingMetrics) -> Result<()> {
        let key = format!("model:{}:metrics:{}", model_id, chrono::Utc::now().timestamp());
        let data = bincode::serialize(metrics)?;
        self.put_raw(key.as_bytes(), &data).await
    }

    /// 存储统一模型
    pub async fn put_unified_model(&self, model_id: &str, _model: &Arc<dyn crate::model::unified::UnifiedModel>) -> Result<()> {
        let key = format!("unified_model:{}", model_id);
        // 由于UnifiedModel是trait对象，需要先转换为可序列化的格式
        // 这里使用模型的基本信息进行序列化
        let model_info = serde_json::json!({
            "model_id": model_id,
            "model_type": "unified",
            "timestamp": chrono::Utc::now().timestamp(),
        });
        let value = serde_json::to_vec(&model_info)?;
        self.put_raw(key.as_bytes(), &value).await
    }

    /// 获取统一模型
    /// 
    /// 从存储中恢复统一模型，包括模型参数、架构和元数据。
    /// 注意：此方法需要模型已在模型注册表中注册，否则无法创建ModelInterface。
    pub async fn get_unified_model(&self, model_id: &str) -> Result<Option<Arc<dyn crate::model::unified::UnifiedModel>>> {
        use crate::compat::{ModelArchitecture as UnifiedModelArchitecture};
        use std::collections::HashMap;
        
        info!("尝试从存储中恢复统一模型: {}", model_id);
        
        // 检查统一模型元数据是否存在
        let key = format!("unified_model:{}", model_id);
        let model_metadata = match self.get_raw(&key).await? {
            Some(data) => {
                serde_json::from_slice::<serde_json::Value>(&data)
                    .map_err(|e| Error::serialization(format!("反序列化统一模型元数据失败: {}", e)))?
            }
            None => {
                debug!("统一模型 {} 的元数据不存在", model_id);
                return Ok(None);
            }
        };
        
        // 获取模型参数
        let parameters = match self.get_model_parameters(model_id) {
            Ok(Some(params)) => Some(params),
            Ok(None) => {
                debug!("模型 {} 的参数不存在，使用默认参数", model_id);
                None
            }
            Err(e) => {
                log::warn!("获取模型 {} 的参数失败: {}，使用默认参数", model_id, e);
                None
            }
        };
        
        // 获取模型架构
        let architecture = match self.get_model_architecture(model_id) {
            Ok(Some(arch)) => {
                Some(UnifiedModelArchitecture::from_core_types(arch))
            }
            Ok(None) => {
                debug!("模型 {} 的架构不存在", model_id);
                None
            }
            Err(e) => {
                log::warn!("获取模型 {} 的架构失败: {}，跳过架构", model_id, e);
                None
            }
        };
        
        // 获取模型信息用于元数据
        let metadata = match self.get_model_info(model_id) {
            Ok(Some(info)) => {
                let mut meta = HashMap::new();
                meta.insert("name".to_string(), info.name);
                if let Some(desc) = info.description {
                    meta.insert("description".to_string(), desc);
                }
                meta.insert("version".to_string(), info.version);
                meta.insert("model_type".to_string(), info.model_type);
                Some(meta)
            }
            Ok(None) | Err(_) => {
                // 从存储的元数据中提取信息
                let mut meta = HashMap::new();
                if let Some(model_id_val) = model_metadata.get("model_id").and_then(|v| v.as_str()) {
                    meta.insert("model_id".to_string(), model_id_val.to_string());
                }
                if let Some(model_type_val) = model_metadata.get("model_type").and_then(|v| v.as_str()) {
                    meta.insert("model_type".to_string(), model_type_val.to_string());
                }
                Some(meta)
            }
        };
        
        // 模型注册表功能已移除 - 向量数据库系统不需要此功能
        log::warn!("模型注册表功能已移除，无法恢复统一模型: {}", model_id);
        return Ok(None);
        
        /* 原代码已注释
        let model_interface: Arc<dyn crate::model::interface::ModelInterface> = {
            let registry = crate::model::registry::ModelRegistry::new();
            match registry.get_model(model_id) {
                Ok(model_arc) => {
                    info!("从模型注册表获取模型实例: {}", model_id);
                    Arc::new(ModelInterfaceAdapter::from_std_mutex(model_arc))
                }
                Err(_) => {
                    log::warn!("模型 {} 未在注册表中注册，无法恢复统一模型", model_id);
                    return Ok(None);
                }
            }
        };
        
        let unified_model = UnifiedModelAdapter::new(
            model_id.to_string(),
            model_interface,
            architecture,
            parameters,
            metadata,
        );
        
        info!("成功从存储中恢复统一模型: {}", model_id);
        Ok(Some(Arc::new(unified_model)))
        */
    }

    /// 基础数据存储方法
    async fn put_raw(&self, key: &[u8], value: &[u8]) -> Result<()> {
        let db = self.db.write().await;
        db.insert(key, value)
            .map_err(|e| Error::storage(format!("存储数据失败: {}", e)))?;
        Ok(())
    }

    /// 基础数据检索方法
    async fn get_raw(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let db = self.db.read().await;
        Ok(db.get(key.as_bytes())?.map(|v| v.to_vec()))
    }
} 