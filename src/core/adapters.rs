/// 接口适配器层
/// 
/// 提供新旧接口之间的兼容性适配，确保现有模块可以无缝迁移到统一系统
/// 解决接口抽象不足问题的桥接实现

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use async_trait::async_trait;
use serde::Serialize;
use uuid::Uuid;
use chrono::Utc;
use log::{info, warn, debug};
use std::convert::TryFrom;

use crate::error::{Error, Result};
use crate::core::unified_system::*;

// 导入现有模块类型
 

// ============================================================================
// 数据值适配器
// ============================================================================

/// 数据值适配器 - 处理不同数据格式的转换
pub struct DataValueAdapter;

impl DataValueAdapter {
    /// 从任意数据类型转换为统一数据值
    pub fn from_any<T: Serialize>(value: T) -> Result<UnifiedDataValue> {
        // 序列化为JSON值进行转换
        let json_value = serde_json::to_value(value)
            .map_err(|e| Error::serialization(e.to_string()))?;
        
        warn!("DataValueAdapter::from_any used for conversion");
        Self::from_json_value(json_value)
    }
    
    /// 从JSON值转换
    pub fn from_json_value(value: serde_json::Value) -> Result<UnifiedDataValue> {
        debug!("DataValueAdapter::from_json_value invoked");
        match value {
            serde_json::Value::Null => Ok(UnifiedDataValue::Null),
            serde_json::Value::Bool(b) => Ok(UnifiedDataValue::Scalar(UnifiedScalar::Bool(b))),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(UnifiedDataValue::Scalar(UnifiedScalar::Int64(i)))
                } else if let Some(f) = n.as_f64() {
                    Ok(UnifiedDataValue::Scalar(UnifiedScalar::Float64(f)))
                } else {
                    Err(Error::invalid_data("无效的数字类型"))
                }
            }
            serde_json::Value::String(s) => Ok(UnifiedDataValue::Text(s)),
            serde_json::Value::Array(arr) => {
                let converted: Result<Vec<_>> = arr.into_iter()
                    .map(Self::from_json_value)
                    .collect();
                Ok(UnifiedDataValue::Array(converted?))
            }
            serde_json::Value::Object(obj) => {
                let mut converted = HashMap::new();
                for (k, v) in obj {
                    converted.insert(k, Self::from_json_value(v)?);
                }
                Ok(UnifiedDataValue::Composite(converted))
            }
        }
    }
    
    /// 转换为JSON值
    pub fn to_json_value(unified_value: UnifiedDataValue) -> Result<serde_json::Value> {
        info!("DataValueAdapter::to_json_value invoked");
        match unified_value {
            UnifiedDataValue::Null => Ok(serde_json::Value::Null),
            UnifiedDataValue::Scalar(s) => match s {
                UnifiedScalar::Bool(b) => Ok(serde_json::Value::Bool(b)),
                UnifiedScalar::Int8(i) => Ok(serde_json::Value::Number(serde_json::Number::from(i))),
                UnifiedScalar::Int16(i) => Ok(serde_json::Value::Number(serde_json::Number::from(i))),
                UnifiedScalar::Int32(i) => Ok(serde_json::Value::Number(serde_json::Number::from(i))),
                UnifiedScalar::Int64(i) => Ok(serde_json::Value::Number(serde_json::Number::from(i))),
                UnifiedScalar::UInt8(i) => Ok(serde_json::Value::Number(serde_json::Number::from(i))),
                UnifiedScalar::UInt16(i) => Ok(serde_json::Value::Number(serde_json::Number::from(i))),
                UnifiedScalar::UInt32(i) => Ok(serde_json::Value::Number(serde_json::Number::from(i))),
                UnifiedScalar::UInt64(i) => Ok(serde_json::Value::Number(serde_json::Number::from(i))),
                UnifiedScalar::Float32(f) => {
                    if let Some(n) = serde_json::Number::from_f64(f as f64) {
                        Ok(serde_json::Value::Number(n))
                    } else {
                        Err(Error::invalid_data("无效的浮点数"))
                    }
                }
                UnifiedScalar::Float64(f) => {
                    if let Some(n) = serde_json::Number::from_f64(f) {
                        Ok(serde_json::Value::Number(n))
                    } else {
                        Err(Error::invalid_data("无效的浮点数"))
                    }
                }
            },
            UnifiedDataValue::Text(s) => Ok(serde_json::Value::String(s)),
            UnifiedDataValue::Binary(_) => Ok(serde_json::Value::String("binary_data".to_string())),
            UnifiedDataValue::Array(arr) => {
                let converted: Result<Vec<_>> = arr.into_iter()
                    .map(Self::to_json_value)
                    .collect();
                Ok(serde_json::Value::Array(converted?))
            }
            UnifiedDataValue::Composite(comp) => {
                let mut converted = serde_json::Map::new();
                for (k, v) in comp {
                    converted.insert(k, Self::to_json_value(v)?);
                }
                Ok(serde_json::Value::Object(converted))
            }
            UnifiedDataValue::Timestamp(dt) => {
                Ok(serde_json::Value::String(dt.to_rfc3339()))
            }
            UnifiedDataValue::Vector(v) => {
                let values: Vec<serde_json::Value> = v.data.into_iter()
                    .map(|f| serde_json::Number::from_f64(f as f64)
                        .map(serde_json::Value::Number)
                        .unwrap_or(serde_json::Value::Null))
                    .collect();
                Ok(serde_json::Value::Array(values))
            }
            UnifiedDataValue::Matrix(m) => {
                let mut rows = Vec::new();
                for i in 0..m.rows {
                    let start = i * m.cols;
                    let end = start + m.cols;
                    let row: Vec<serde_json::Value> = m.data[start..end].iter()
                        .map(|&f| serde_json::Number::from_f64(f as f64)
                            .map(serde_json::Value::Number)
                            .unwrap_or(serde_json::Value::Null))
                        .collect();
                    rows.push(serde_json::Value::Array(row));
                }
                Ok(serde_json::Value::Array(rows))
            }
            UnifiedDataValue::Tensor(t) => {
                let mut tensor_obj = serde_json::Map::new();
                tensor_obj.insert("shape".to_string(), 
                    serde_json::Value::Array(
                        t.shape.into_iter()
                            .map(|s| serde_json::Value::Number(serde_json::Number::from(s)))
                            .collect()
                    )
                );
                tensor_obj.insert("data".to_string(),
                    serde_json::Value::Array(
                        t.data.into_iter()
                            .map(|f| serde_json::Number::from_f64(f as f64)
                                .map(serde_json::Value::Number)
                                .unwrap_or(serde_json::Value::Null))
                            .collect()
                    )
                );
                Ok(serde_json::Value::Object(tensor_obj))
            }
        }
    }
}

// =============================================================================
// UnifiedDataValue <-> serde_json::Value 双向转换实现（TryFrom/From）
// =============================================================================

impl TryFrom<serde_json::Value> for UnifiedDataValue {
    type Error = crate::error::Error;

    fn try_from(value: serde_json::Value) -> std::result::Result<Self, Self::Error> {
        DataValueAdapter::from_json_value(value)
    }
}

impl TryFrom<&serde_json::Value> for UnifiedDataValue {
    type Error = crate::error::Error;

    fn try_from(value: &serde_json::Value) -> std::result::Result<Self, Self::Error> {
        DataValueAdapter::from_json_value(value.clone())
    }
}

impl TryFrom<UnifiedDataValue> for serde_json::Value {
    type Error = crate::error::Error;

    fn try_from(value: UnifiedDataValue) -> std::result::Result<Self, Self::Error> {
        DataValueAdapter::to_json_value(value)
    }
}

impl TryFrom<&UnifiedDataValue> for serde_json::Value {
    type Error = crate::error::Error;

    fn try_from(value: &UnifiedDataValue) -> std::result::Result<Self, Self::Error> {
        DataValueAdapter::to_json_value(value.clone())
    }
}

// ============================================================================
// 数据处理服务适配器
// ============================================================================

/// 数据处理服务适配器
pub struct DataProcessingServiceAdapter {
    name: String,
}

impl DataProcessingServiceAdapter {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

#[async_trait]
impl DataProcessingService for DataProcessingServiceAdapter {
    async fn process_data(&self, data: UnifiedDataValue) -> Result<UnifiedDataValue> {
        // 简单的数据处理逻辑
        match data {
            UnifiedDataValue::Vector(mut v) => {
                // 标准化向量
                if !v.data.is_empty() {
                    let sum: f32 = v.data.iter().sum();
                    let mean = sum / v.data.len() as f32;
                    for val in &mut v.data {
                        *val = *val - mean;
                    }
                }
                Ok(UnifiedDataValue::Vector(v))
            }
            UnifiedDataValue::Text(s) => {
                // 文本处理 - 转换为小写
                Ok(UnifiedDataValue::Text(s.to_lowercase()))
            }
            _ => Ok(data),
        }
    }
    
    async fn validate_data(&self, data: &UnifiedDataValue) -> Result<bool> {
        match data {
            UnifiedDataValue::Null => Ok(false),
            UnifiedDataValue::Vector(v) => Ok(!v.data.is_empty()),
            UnifiedDataValue::Matrix(m) => Ok(!m.data.is_empty() && m.rows > 0 && m.cols > 0),
            UnifiedDataValue::Tensor(t) => Ok(!t.data.is_empty() && !t.shape.is_empty()),
            UnifiedDataValue::Text(s) => Ok(!s.is_empty()),
            _ => Ok(true),
        }
    }
    
    async fn transform_data(&self, data: UnifiedDataValue, transform_type: &str) -> Result<UnifiedDataValue> {
        match transform_type {
            "normalize" => {
                match data {
                    UnifiedDataValue::Vector(mut v) => {
                        if !v.data.is_empty() {
                            let max_val = v.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                            let min_val = v.data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                            let range = max_val - min_val;
                            
                            if range > 0.0 {
                                for val in &mut v.data {
                                    *val = (*val - min_val) / range;
                                }
                            }
                        }
                        Ok(UnifiedDataValue::Vector(v))
                    }
                    _ => Ok(data),
                }
            }
            "scale" => {
                match data {
                    UnifiedDataValue::Vector(mut v) => {
                        for val in &mut v.data {
                            *val *= 0.1;
                        }
                        Ok(UnifiedDataValue::Vector(v))
                    }
                    _ => Ok(data),
                }
            }
            _ => Ok(data),
        }
    }
}

// ============================================================================
// 模型管理服务适配器
// ============================================================================

/// 模型管理服务适配器
pub struct ModelManagementServiceAdapter {
    model_storage: Arc<RwLock<HashMap<String, ModelInfo>>>,
}

impl ModelManagementServiceAdapter {
    pub fn new() -> Self {
        Self {
            model_storage: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl ModelManagementService for ModelManagementServiceAdapter {
    async fn create_model(&self, config: ModelConfig) -> Result<String> {
        let model_id = Uuid::new_v4().to_string();
        let now = Utc::now();
        
        let model_info = ModelInfo {
            id: model_id.clone(),
            name: config.name,
            model_type: config.model_type,
            status: ModelStatus::Created,
            version: "1.0.0".to_string(),
            created_at: now,
            updated_at: now,
            metadata: config.metadata,
        };
        
        if let Ok(mut storage) = self.model_storage.write() {
            storage.insert(model_id.clone(), model_info);
        }
        
        Ok(model_id)
    }
    
    async fn get_model(&self, model_id: &str) -> Result<Option<ModelInfo>> {
        if let Ok(storage) = self.model_storage.read() {
            Ok(storage.get(model_id).cloned())
        } else {
            Err(Error::storage("无法读取模型存储"))
        }
    }
    
    async fn update_model(&self, model_id: &str, updates: ModelUpdates) -> Result<()> {
        if let Ok(mut storage) = self.model_storage.write() {
            if let Some(model) = storage.get_mut(model_id) {
                if let Some(name) = updates.name {
                    model.name = name;
                }
                if let Some(status) = updates.status {
                    model.status = status;
                }
                if let Some(metadata) = updates.metadata {
                    model.metadata = metadata;
                }
                model.updated_at = Utc::now();
                Ok(())
            } else {
                Err(Error::not_found(format!("模型 {} 不存在", model_id)))
            }
        } else {
            Err(Error::storage("无法写入模型存储"))
        }
    }
    
    async fn delete_model(&self, model_id: &str) -> Result<()> {
        if let Ok(mut storage) = self.model_storage.write() {
            if storage.remove(model_id).is_some() {
                Ok(())
            } else {
                Err(Error::not_found(format!("模型 {} 不存在", model_id)))
            }
        } else {
            Err(Error::storage("无法写入模型存储"))
        }
    }
    
    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        if let Ok(storage) = self.model_storage.read() {
            Ok(storage.values().cloned().collect())
        } else {
            Err(Error::storage("无法读取模型存储"))
        }
    }
}

// Training service adapter removed: vector database does not need training functionality

// ============================================================================
// 算法执行服务适配器
// ============================================================================

/// 算法执行服务适配器
pub struct AlgorithmExecutionServiceAdapter {
    algorithm_storage: Arc<RwLock<HashMap<String, AlgorithmDefinition>>>,
}

impl AlgorithmExecutionServiceAdapter {
    pub fn new() -> Self {
        Self {
            algorithm_storage: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl AlgorithmExecutionService for AlgorithmExecutionServiceAdapter {
    async fn execute_algorithm(&self, algorithm_id: &str, inputs: Vec<UnifiedDataValue>) -> Result<ExecutionResult> {
        let start_time = std::time::Instant::now();
        
        // 简单的算法执行逻辑
        let outputs = match algorithm_id {
            "sum" => {
                // 求和算法
                let mut sum = 0.0;
                for input in &inputs {
                    if let UnifiedDataValue::Vector(v) = input {
                        sum += v.data.iter().sum::<f32>();
                    }
                }
                vec![UnifiedDataValue::Scalar(UnifiedScalar::Float32(sum))]
            }
            "normalize" => {
                // 标准化算法
                let mut outputs = Vec::new();
                for input in inputs {
                    if let UnifiedDataValue::Vector(mut v) = input {
                        if !v.data.is_empty() {
                            let max_val = v.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                            let min_val = v.data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                            let range = max_val - min_val;
                            
                            if range > 0.0 {
                                for val in &mut v.data {
                                    *val = (*val - min_val) / range;
                                }
                            }
                        }
                        outputs.push(UnifiedDataValue::Vector(v));
                    } else {
                        outputs.push(input);
                    }
                }
                outputs
            }
            _ => inputs, // 默认直接返回输入
        };
        
        let execution_time = start_time.elapsed().as_millis() as u64;
        
        Ok(ExecutionResult {
            success: true,
            outputs,
            execution_time_ms: execution_time,
            resource_usage: ResourceUsage {
                memory_used_mb: 10, // 模拟值
                cpu_percent: 5.0,
                execution_time_ms: execution_time,
            },
            error_message: None,
        })
    }
    
    async fn validate_algorithm(&self, algorithm_code: &str) -> Result<ValidationResult> {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        
        // 简单的代码验证
        if algorithm_code.is_empty() {
            errors.push("算法代码不能为空".to_string());
        }
        
        if algorithm_code.len() > 10000 {
            warnings.push("算法代码过长，可能影响性能".to_string());
        }
        
        if !algorithm_code.contains("fn") && !algorithm_code.contains("function") {
            warnings.push("未找到函数定义".to_string());
        }
        
        let mut vr = ValidationResult::success();
        vr.is_valid = errors.is_empty();
        vr.errors = errors;
        vr.warnings = warnings;
        Ok(vr)
    }
    
    async fn register_algorithm(&self, algorithm: AlgorithmDefinition) -> Result<String> {
        let algorithm_id = algorithm.id.clone();
        
        if let Ok(mut storage) = self.algorithm_storage.write() {
            storage.insert(algorithm_id.clone(), algorithm);
        }
        
        Ok(algorithm_id)
    }
}

// ============================================================================
// 存储服务适配器
// ============================================================================

/// 存储服务适配器
pub struct StorageServiceAdapter {
    storage: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

impl StorageServiceAdapter {
    pub fn new() -> Self {
        Self {
            storage: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl StorageService for StorageServiceAdapter {
    async fn store(&self, key: &str, value: &[u8]) -> Result<()> {
        if let Ok(mut storage) = self.storage.write() {
            storage.insert(key.to_string(), value.to_vec());
            Ok(())
        } else {
            Err(Error::storage("无法写入存储"))
        }
    }
    
    async fn retrieve(&self, key: &str) -> Result<Option<Vec<u8>>> {
        if let Ok(storage) = self.storage.read() {
            Ok(storage.get(key).cloned())
        } else {
            Err(Error::storage("无法读取存储"))
        }
    }
    
    async fn delete(&self, key: &str) -> Result<()> {
        if let Ok(mut storage) = self.storage.write() {
            storage.remove(key);
            Ok(())
        } else {
            Err(Error::storage("无法写入存储"))
        }
    }
    
    async fn list_keys(&self, prefix: &str) -> Result<Vec<String>> {
        if let Ok(storage) = self.storage.read() {
            let keys: Vec<String> = storage.keys()
                .filter(|key| key.starts_with(prefix))
                .cloned()
                .collect();
            Ok(keys)
        } else {
            Err(Error::storage("无法读取存储"))
        }
    }
    
    async fn exists(&self, key: &str) -> Result<bool> {
        if let Ok(storage) = self.storage.read() {
            Ok(storage.contains_key(key))
        } else {
            Err(Error::storage("无法读取存储"))
        }
    }
}

// ============================================================================
// 适配器工厂
// ============================================================================

/// 适配器工厂
pub struct AdapterFactory;

impl AdapterFactory {
    /// 创建数据处理服务适配器
    pub fn create_data_processing_adapter(name: String) -> Arc<dyn DataProcessingService> {
        Arc::new(DataProcessingServiceAdapter::new(name))
    }
    
    /// 创建模型管理服务适配器
    pub fn create_model_management_adapter() -> Arc<dyn ModelManagementService> {
        Arc::new(ModelManagementServiceAdapter::new())
    }
    
    // Training service adapter creation removed: vector database does not need training functionality
    
    /// 创建算法执行服务适配器
    pub fn create_algorithm_service_adapter() -> Arc<dyn AlgorithmExecutionService> {
        Arc::new(AlgorithmExecutionServiceAdapter::new())
    }
    
    /// 创建存储服务适配器
    pub fn create_storage_service_adapter() -> Arc<dyn StorageService> {
        Arc::new(StorageServiceAdapter::new())
    }
    
    /// 创建完整的服务注册表
    pub fn create_unified_registry() -> UnifiedServiceRegistry {
        let mut registry = UnifiedServiceRegistry::new();
        
        registry.register_data_service(Self::create_data_processing_adapter("default".to_string()));
        registry.register_model_service(Self::create_model_management_adapter());
        // Training service registration removed: vector database does not need training functionality
        registry.register_algorithm_service(Self::create_algorithm_service_adapter());
        registry.register_storage_service(Self::create_storage_service_adapter());
        
        registry
    }
}

// ============================================================================
// 默认实现
// ============================================================================

impl Default for ModelManagementServiceAdapter {
    fn default() -> Self {
        Self::new()
    }
}

// TrainingServiceAdapter Default implementation removed: vector database does not need training functionality

impl Default for AlgorithmExecutionServiceAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for StorageServiceAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_value_adapter() {
        let json_bool = serde_json::Value::Bool(true);
        let unified = DataValueAdapter::from_json_value(json_bool).unwrap();
        
        match unified {
            UnifiedDataValue::Scalar(UnifiedScalar::Bool(true)) => {},
            _ => panic!("转换失败"),
        }
        
        let back_to_json = DataValueAdapter::to_json_value(unified).unwrap();
        match back_to_json {
            serde_json::Value::Bool(true) => {},
            _ => panic!("反向转换失败"),
        }
    }

    #[tokio::test]
    async fn test_model_management_adapter() {
        let adapter = ModelManagementServiceAdapter::new();
        
        let config = ModelConfig {
            name: "test_model".to_string(),
            model_type: "neural_network".to_string(),
            architecture: ModelArchitecture {
                layers: vec![],
                connections: vec![],
                input_shape: vec![784],
                output_shape: vec![10],
            },
            hyperparameters: HashMap::new(),
            metadata: HashMap::new(),
        };
        
        let model_id = adapter.create_model(config).await.unwrap();
        let retrieved_model = adapter.get_model(&model_id).await.unwrap();
        
        assert!(retrieved_model.is_some());
        assert_eq!(retrieved_model.unwrap().name, "test_model");
    }

    // Training service adapter test removed: vector database does not need training functionality

    #[tokio::test]
    async fn test_algorithm_execution_adapter() {
        let adapter = AlgorithmExecutionServiceAdapter::new();
        
        let inputs = vec![
            UnifiedDataValue::Vector(UnifiedVector {
                data: vec![1.0, 2.0, 3.0],
                dtype: UnifiedDataType::Float32,
                metadata: HashMap::new(),
            })
        ];
        
        let result = adapter.execute_algorithm("sum", inputs).await.unwrap();
        assert!(result.success);
        assert_eq!(result.outputs.len(), 1);
        
        match &result.outputs[0] {
            UnifiedDataValue::Scalar(UnifiedScalar::Float32(sum)) => {
                assert_eq!(*sum, 6.0);
            }
            _ => panic!("期望标量结果"),
        }
    }

    #[tokio::test]
    async fn test_storage_service_adapter() {
        let adapter = StorageServiceAdapter::new();
        
        let key = "test_key";
        let value = b"test_value";
        
        adapter.store(key, value).await.unwrap();
        
        let retrieved = adapter.retrieve(key).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), value);
        
        let exists = adapter.exists(key).await.unwrap();
        assert!(exists);
        
        adapter.delete(key).await.unwrap();
        
        let exists_after_delete = adapter.exists(key).await.unwrap();
        assert!(!exists_after_delete);
    }
} 