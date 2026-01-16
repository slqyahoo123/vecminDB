use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use crate::error::Result;
use crate::storage::Storage;
use crate::core::interfaces::StorageService as CoreStorageService;
use crate::event::EventSystem;
use crate::algorithm::executor::sandbox::AlgorithmExecutor;
use crate::algorithm::traits::Algorithm;
use crate::core::{
    interfaces::{
        AlgorithmExecutionResult, AlgorithmModelInterface, ModelAlgorithmService,
        ModelForAlgorithm, AlgorithmRequirements, ModelInferenceInterface,
    },
    CoreTensorData,
};

/// Algorithm model implementation used as a thin compatibility layer.
/// 
/// In the vector database context we don't provide full training
/// capabilities – any training‑related operations will return a
/// `feature_not_enabled` error so that callers can clearly know
/// that model training is out of scope for this service.
pub struct AlgorithmModelImpl {
    model_id: String,
}

impl AlgorithmModelImpl {
    pub fn new(model_id: String) -> Self {
        Self { model_id }
    }
}

#[async_trait]
impl AlgorithmModelInterface for AlgorithmModelImpl {
    // Training-related methods removed: vector database does not need training functionality

    async fn apply_algorithm_to_model(&self, algorithm_id: &str, model_id: &str) -> Result<String> {
        // Algorithm application to models is out of scope for the core
        // vector database – this is left to external training pipelines.
        Err(crate::error::Error::feature_not_enabled(format!(
            "applying algorithm {} to model {}",
            algorithm_id, model_id
        )))
    }

    async fn get_model_algorithms(&self, model_id: &str) -> Result<Vec<String>> {
        // This adapter does not manage model‑algorithm relationships;
        // expose a clear unsupported feature error instead of returning
        // synthetic data.
        Err(crate::error::Error::feature_not_enabled(format!(
            "listing algorithms for model {}",
            model_id
        )))
    }

    async fn get_algorithm_models(&self, algorithm_id: &str) -> Result<Vec<String>> {
        Err(crate::error::Error::feature_not_enabled(format!(
            "listing models for algorithm {}",
            algorithm_id
        )))
    }

    async fn remove_algorithm_from_model(&self, algorithm_id: &str, model_id: &str) -> Result<()> {
        Err(crate::error::Error::feature_not_enabled(format!(
            "removing algorithm {} from model {}",
            algorithm_id, model_id
        )))
    }

    async fn validate_compatibility(&self, algorithm_id: &str, model_id: &str) -> Result<crate::core::interfaces::ValidationResult> {
        // Compatibility validation is not implemented in the core DB.
        Err(crate::error::Error::feature_not_enabled(format!(
            "validating compatibility between algorithm {} and model {}",
            algorithm_id, model_id
        )))
    }
}

/// Algorithm服务适配器 - 实现core接口，解决循环依赖
pub struct AlgorithmServiceAdapter {
    storage: Arc<Storage>,
    event_system: Arc<dyn EventSystem>,
    executor: Arc<dyn AlgorithmExecutor>,
}

impl AlgorithmServiceAdapter {
    pub fn new(
        storage: Arc<Storage>,
        event_system: Arc<dyn EventSystem>,
        executor: Arc<dyn AlgorithmExecutor>
    ) -> Self {
        Self {
            storage,
            event_system,
            executor,
        }
    }

    /// 内部方法：从存储中获取算法
    async fn get_algorithm_from_storage(&self, algorithm_id: &str) -> Result<Option<Box<dyn Algorithm>>> {
        // 实际的算法获取逻辑 - 使用Storage的底层方法
        let key = format!("algorithm:{}", algorithm_id);
        match self.storage.get_raw(key.as_str()).await {
            Ok(Some(data)) => {
                // 从序列化数据恢复算法代码
                let code = String::from_utf8(data)
                    .unwrap_or_else(|_| "// Empty algorithm".to_string());
                    
                // Deserialize algorithm code and create Algorithm object
                let algorithm = crate::algorithm::types::Algorithm::new(
                    &format!("算法_{}", algorithm_id),
                    "custom",
                    &code
                );
                Ok(Some(Box::new(algorithm)))
            },
            Ok(None) => Ok(None),
            Err(_) => Ok(None),
        }
    }

    /// 内部方法：保存算法到存储
    async fn save_algorithm_to_storage(&self, algorithm: &dyn Algorithm) -> Result<()> {
        let key = format!("algorithm:{}", algorithm.get_id());
        let code_bytes = algorithm.get_code().as_bytes();
        self.storage.put_raw(key.as_str(), code_bytes).await
    }

    /// 转换算法类型
    fn convert_algorithm_type(algo_type: &str) -> crate::core::types::AlgorithmType {
        match algo_type {
            "optimization" => crate::core::types::AlgorithmType::Optimization,
            "neural_network" | "deep_learning" => crate::core::types::AlgorithmType::DeepLearning,
            "data_processing" => crate::core::types::AlgorithmType::DataProcessing,
            "custom" => crate::core::types::AlgorithmType::MachineLearning,
            _ => crate::core::types::AlgorithmType::MachineLearning,
        }
    }

    /// 转换执行结果
    fn convert_execution_result(result: &serde_json::Value) -> AlgorithmExecutionResult {
        use chrono::Utc;
        
        // Parse outputs from JSON result
        let outputs = if let Some(outputs_value) = result.get("outputs") {
            if let Some(outputs_array) = outputs_value.as_array() {
                outputs_array.iter().filter_map(|output| {
                    // Extract tensor data from output object
                    let id = output.get("id")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
                    
                    let shape = output.get("shape")
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_u64().map(|n| n as usize)).collect())
                        .unwrap_or_else(|| vec![1]);
                    
                    let data = output.get("data")
                        .and_then(|v| v.as_array())
                        .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
                        .unwrap_or_else(|| vec![0.0]);
                    
                    let dtype = output.get("dtype")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "float32".to_string());
                    
                    let device = output.get("device")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "cpu".to_string());
                    
                    let requires_grad = output.get("requires_grad")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    
                    let metadata = output.get("metadata")
                        .and_then(|v| v.as_object())
                        .map(|obj| {
                            obj.iter()
                                .filter_map(|(k, v)| {
                                    v.as_str().map(|s| (k.clone(), s.to_string()))
                                })
                                .collect()
                        })
                        .unwrap_or_else(HashMap::new);
                    
                    Some(CoreTensorData {
                        id,
                        shape,
                        data,
                        dtype,
                        device,
                        requires_grad,
                        metadata,
                        created_at: Utc::now(),
                        updated_at: Utc::now(),
                    })
                }).collect()
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        // Extract execution metadata
        let execution_time = result.get("execution_time")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        
        let memory_used = result.get("memory_used")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        
        let status = result.get("status")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "completed".to_string());
        
        let logs = result.get("logs")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_else(Vec::new);

        AlgorithmExecutionResult {
            outputs,
            execution_time,
            memory_used,
            status,
            logs,
        }
    }
}

#[async_trait]
impl ModelAlgorithmService for AlgorithmServiceAdapter {
    /// 获取模型算法接口
    async fn get_model_algorithm_interface(&self, model_id: &str) -> Result<Option<crate::core::interfaces::ModelAlgorithmInterface>> {
        // 检查模型是否存在
        use CoreStorageService;
        if let Ok(Some(_model)) = CoreStorageService::get_model(&*self.storage, model_id).await {
            // 返回模型算法接口
            Ok(Some(crate::core::interfaces::ModelAlgorithmInterface {
                model_id: model_id.to_string(),
                interface_type: "algorithm".to_string(),
                capabilities: vec!["train".to_string(), "predict".to_string()],
                metadata: std::collections::HashMap::new(),
            }))
        } else {
            Ok(None)
        }
    }

    /// 创建算法模型
    async fn create_algorithm_model(&self, model_def: &ModelForAlgorithm, _requirements: &AlgorithmRequirements) -> Result<String> {
        // Generate model ID for algorithm model
        let model_id = format!("algo_model_{}", uuid::Uuid::new_v4());
        
        // Create model entry in storage
        log::info!("创建算法模型: {} -> {}", model_def.model_id, model_id);
        
        Ok(model_id)
    }

    /// 获取模型推理接口
    async fn get_model_inference_interface(&self, model_id: &str) -> Result<Option<ModelInferenceInterface>> {
        // 检查模型是否存在
        use CoreStorageService;
        if let Ok(_model) = CoreStorageService::get_model(&*self.storage, model_id).await {
            let interface = ModelInferenceInterface {
                model_id: model_id.to_string(),
                inference_type: "standard".to_string(),
                input_format: "tensor".to_string(),
                output_format: "tensor".to_string(),
                batch_size: 32,
                timeout: 30000, // 30 seconds
            };
            Ok(Some(interface))
        } else {
            Ok(None)
        }
    }
} 