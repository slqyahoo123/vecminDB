use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::core::types::CoreTensorData;
use crate::error::Result;
use async_trait::async_trait;

/// 模型算法服务
#[async_trait]
pub trait ModelAlgorithmService: Send + Sync {
    async fn get_model_algorithm_interface(&self, model_id: &str) -> Result<Option<ModelAlgorithmInterface>>;
    async fn create_algorithm_model(&self, model_def: &ModelForAlgorithm, requirements: &AlgorithmRequirements) -> Result<String>;
    async fn get_model_inference_interface(&self, model_id: &str) -> Result<Option<ModelInferenceInterface>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAlgorithmInterface {
    pub model_id: String,
    pub interface_type: String,
    pub capabilities: Vec<String>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmExecutionResult {
    pub outputs: Vec<CoreTensorData>,
    pub execution_time: u64,
    pub memory_used: usize,
    pub status: String,
    pub logs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelForAlgorithm {
    pub model_id: String,
    pub model_type: String,
    pub architecture: String,
    pub parameters: HashMap<String, String>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmRequirements {
    pub memory_limit: usize,
    pub cpu_limit: f32,
    pub gpu_required: bool,
    pub execution_timeout: u64,
    pub security_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInferenceInterface {
    pub model_id: String,
    pub inference_type: String,
    pub input_format: String,
    pub output_format: String,
    pub batch_size: usize,
    pub timeout: u64,
}


