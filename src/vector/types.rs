use std::collections::HashMap;
use serde::{Deserialize, Serialize};
// no direct chrono imports needed; fully-qualified paths are used
// no direct error Result used in this module
use crate::vector::operations::SimilarityMetric;
use crate::vector::core::operations::VectorOps;

/// 向量数据结构
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vector {
    pub id: String,
    pub data: Vec<f32>,
    pub metadata: Option<super::search::VectorMetadata>,
}

impl Vector {
    /// 计算与另一个向量的相似度
    pub fn similarity_to(&self, other: &Vector, metric: SimilarityMetric) -> f32 {
        VectorOps::compute_vector_similarity(self, other, metric)
    }
}

/// 向量操作结果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorOperationResult {
    pub success: bool,
    pub message: Option<String>,
    pub vector_id: Option<String>,
}

/// 向量批量操作结果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorBatchResult {
    pub success_count: usize,
    pub failed_count: usize,
    pub failed_ids: Vec<String>,
    pub error_messages: HashMap<String, String>,
}

/// 向量统计信息
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorStats {
    pub count: usize,
    pub dimension: usize,
    pub index_type: String,
    pub memory_usage: Option<usize>,
}

/// 向量列表
pub type VectorList = Vec<Vector>;

/// 向量ID类型
pub type VectorId = String;

/// 向量条目
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorEntry {
    pub id: VectorId,
    pub vector: Vector,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub tags: Vec<String>,
}

impl VectorEntry {
    /// 创建新的向量条目
    pub fn new(id: VectorId, vector: Vector) -> Self {
        Self {
            id,
            vector,
            timestamp: chrono::Utc::now(),
            tags: Vec::new(),
        }
    }
    
    /// 添加标签
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

/// 距离函数类型
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DistanceFunction {
    /// 欧几里得距离
    Euclidean,
    /// 曼哈顿距离
    Manhattan,
    /// 余弦距离
    Cosine,
    /// 点积
    DotProduct,
    /// 汉明距离
    Hamming,
    /// 自定义距离函数
    Custom(String),
}

impl Default for DistanceFunction {
    fn default() -> Self {
        DistanceFunction::Euclidean
    }
}

/// 创建向量请求
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateVectorRequest {
    /// 向量ID
    pub id: String,
    /// 向量值
    pub values: Vec<f32>,
    /// 元数据
    pub metadata: Option<serde_json::Value>,
}

/// 向量搜索结果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorSearchResult {
    /// 向量ID
    pub id: String,
    /// 相似度分数
    pub score: f32,
    /// 向量数据
    pub vector: Option<Vector>,
    /// 元数据
    pub metadata: Option<serde_json::Value>,
}

// VectorMetadata 已在 vector/search.rs 中定义，这里重新导出
pub use super::search::VectorMetadata;

/// 搜索向量请求
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchVectorsRequest {
    /// 查询向量
    pub query_vector: Vec<f32>,
    /// 返回结果数量
    pub top_k: usize,
    /// 相似度度量方法
    pub metric: String,
    /// 过滤条件
    pub filter: Option<serde_json::Value>,
} 