//! 向量API类型定义

use serde::{Deserialize, Serialize};

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

/// 批量创建向量请求
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchCreateVectorsRequest {
    /// 向量列表
    pub vectors: Vec<CreateVectorRequest>,
}

/// 向量信息
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vector {
    /// 向量ID
    pub id: String,
    /// 向量值
    pub values: Vec<f32>,
    /// 元数据
    pub metadata: Option<serde_json::Value>,
    /// 创建时间
    pub created_at: String,
    /// 更新时间
    pub updated_at: String,
}

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

/// 批量删除向量请求
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchDeleteVectorsRequest {
    /// 要删除的向量ID列表
    pub ids: Vec<String>,
}

/// 向量统计信息
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorStats {
    /// 总向量数量
    pub total_count: usize,
    /// 平均维度
    pub avg_dimension: f32,
    /// 最大维度
    pub max_dimension: usize,
    /// 最小维度
    pub min_dimension: usize,
    /// 创建时间范围
    pub created_at_range: Option<(String, String)>,
}

/// 向量操作结果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorOperationResult {
    /// 操作是否成功
    pub success: bool,
    /// 操作消息
    pub message: String,
    /// 影响的向量数量
    pub affected_count: usize,
    /// 错误列表
    pub errors: Vec<String>,
}

/// 向量批处理结果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorBatchResult {
    /// 成功处理的向量数量
    pub success_count: usize,
    /// 失败的向量数量
    pub failure_count: usize,
    /// 成功的结果
    pub successes: Vec<Vector>,
    /// 失败的结果
    pub failures: Vec<String>,
}

/// 距离函数类型
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DistanceFunction {
    /// 欧几里得距离
    Euclidean,
    /// 余弦相似度
    Cosine,
    /// 曼哈顿距离
    Manhattan,
    /// 切比雪夫距离
    Chebyshev,
}

impl Default for DistanceFunction {
    fn default() -> Self {
        DistanceFunction::Euclidean
    }
}

