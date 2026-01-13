//! 集合API类型定义

use serde::{Deserialize, Serialize};
use crate::vector::index::IndexType;

/// 创建集合请求
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreateCollectionRequest {
    /// 集合名称
    pub name: String,
    /// 向量维度
    pub dimension: usize,
    /// 索引类型
    pub index_type: IndexType,
    /// 元数据模式（可选）
    pub metadata_schema: Option<std::collections::HashMap<String, String>>,
}

/// 集合信息
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CollectionInfo {
    /// 集合名称
    pub name: String,
    /// 向量维度
    pub dimension: usize,
    /// 索引类型
    pub index_type: String,
    /// 向量数量
    pub vector_count: usize,
    /// 创建时间
    pub created_at: String,
}

/// 集合统计信息
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CollectionStats {
    /// 集合名称
    pub name: String,
    /// 向量数量
    pub vector_count: usize,
    /// 索引类型
    pub index_type: String,
    /// 内存使用（字节）
    pub memory_usage: Option<usize>,
    /// 索引状态
    pub index_status: String,
}

