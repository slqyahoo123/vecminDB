use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// 推理结果详情（核心层DTO）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResultDetail {
    pub id: String,
    pub model_id: String,
    pub model_name: String,
    pub input_data: serde_json::Value,
    pub output_data: serde_json::Value,
    pub confidence: Option<f64>,
    pub processing_time_ms: u64,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// 推理结果（核心层DTO）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub id: String,
    pub model_id: String,
    pub input: serde_json::Value,
    pub output: serde_json::Value,
    pub confidence: Option<f64>,
    pub processing_time_ms: u64,
    pub created_at: DateTime<Utc>,
}


