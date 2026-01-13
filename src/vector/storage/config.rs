use serde::{Deserialize, Serialize};
use crate::vector::indexes::IndexType;
use crate::vector::SimilarityMetric;

/// 向量存储配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStorageConfig {
    /// 存储路径
    pub path: String,
    /// 索引类型
    pub index_type: IndexType,
    /// 相似度计算方式
    pub metric: SimilarityMetric,
    /// 向量维度
    pub dimension: usize,
    /// 是否自动重建索引
    pub auto_rebuild: bool,
    /// 批处理大小
    pub batch_size: usize,
}

impl Default for VectorStorageConfig {
    fn default() -> Self {
        Self {
            path: "data/vector_storage".to_string(),
            index_type: IndexType::HNSW,
            metric: SimilarityMetric::Cosine,
            dimension: 128,
            auto_rebuild: false,
            batch_size: 1000,
        }
    }
}

/// 存储统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    /// 向量总数
    pub vector_count: usize,
    /// 存储大小(字节)
    pub storage_size: u64,
    /// 索引类型
    pub index_type: IndexType,
    /// 向量维度
    pub dimension: usize,
    /// 创建时间
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// 最后更新时间
    pub last_updated_at: chrono::DateTime<chrono::Utc>,
} 