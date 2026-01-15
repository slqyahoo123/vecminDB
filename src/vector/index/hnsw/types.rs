//! HNSW索引的基础数据类型
//!
//! 本模块定义了HNSW索引所需的基础数据类型，包括距离类型、
//! 搜索配置、节点连接等核心结构体。

use std::cmp::Ordering;
use std::sync::Arc;
use std::collections::HashMap;
use std::time::Duration;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
// use crate::vector::index::common::HNSWNode;
// use crate::vector::index::types::SearchResult as IndexSearchResult;
use uuid::Uuid;

/// 搜索结果结构体
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HNSWSearchResult {
    /// 向量ID
    pub vector_id: Uuid,
    /// 相似度分数
    pub score: f32,
    /// 可选的向量数据
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    /// 可选的元数据
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// 浮点比较包装器，用于优先级队列
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct FloatComparison(pub f32);

impl PartialOrd for FloatComparison {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.0.partial_cmp(&self.0)
    }
}

impl Eq for FloatComparison {}

impl Ord for FloatComparison {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// 反向浮点比较包装器，用于优先级队列（倒序）
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ReverseFloatComparison(pub f32);

impl PartialOrd for ReverseFloatComparison {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Eq for ReverseFloatComparison {}

impl Ord for ReverseFloatComparison {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// 向量id类型
pub type VectorId = u64;

/// 节点索引类型
pub type NodeIndex = usize;

/// 距离值类型
pub type Distance = f32;

/// 向量数据类型
pub type Vector = Vec<f32>;

/// 向量维度类型
pub type Dimension = usize;

/// 图层级类型
pub type Level = usize;

/// 距离计算类型
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistanceType {
    /// 欧氏距离
    Euclidean,
    /// 余弦距离
    Cosine,
    /// 点积距离
    DotProduct,
    /// 曼哈顿距离
    Manhattan,
}

impl Default for DistanceType {
    fn default() -> Self {
        DistanceType::Euclidean
    }
}

impl DistanceType {
    /// 从SimilarityMetric转换为DistanceType
    pub fn from_similarity_metric(metric: crate::vector::core::operations::SimilarityMetric) -> Self {
        use crate::vector::core::operations::SimilarityMetric;
        match metric {
            SimilarityMetric::Cosine => DistanceType::Cosine,
            SimilarityMetric::Euclidean => DistanceType::Euclidean,
            SimilarityMetric::DotProduct => DistanceType::DotProduct,
            SimilarityMetric::Manhattan => DistanceType::Manhattan,
            SimilarityMetric::Jaccard => DistanceType::Euclidean, // Jaccard映射到Euclidean
        }
    }
}

/// 图节点连接（简单元组形式）
pub type NodeConnectionTuple = (NodeIndex, Distance);

/// 单层连接列表
pub type LayerConnections = Vec<NodeConnectionTuple>;

/// 多层连接数据结构
pub type NodeConnections = Vec<LayerConnections>;

/// 元数据映射类型
pub type Metadata = HashMap<String, String>;

/// HNSW索引配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWConfig {
    /// 向量维度
    pub dimension: Dimension,
    /// 最大层数
    pub max_level: usize,
    /// 每层最大连接数
    pub max_connections: usize,
    /// 构建过程中每层搜索的候选数
    pub ef_construction: usize,
    /// 搜索过程中访问的候选数
    pub ef_search: usize,
    /// 距离计算类型
    pub distance_type: DistanceType,
    /// 是否使用查询缓存
    pub use_cache: bool,
    /// 是否使用并行搜索
    pub parallel_search: bool,
}

impl Default for HNSWConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            max_level: 16,
            max_connections: 16,
            ef_construction: 128,
            ef_search: 64,
            distance_type: DistanceType::Euclidean,
            use_cache: true,
            parallel_search: true,
        }
    }
}

/// 搜索配置
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// 搜索类型
    pub strategy: SearchStrategy,
    /// 最大结果数
    pub limit: usize,
    /// 是否包含向量
    pub include_vectors: bool,
    /// 是否包含元数据
    pub include_metadata: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            strategy: SearchStrategy::default(),
            limit: 10,
            include_vectors: false,
            include_metadata: false,
        }
    }
}

/// 搜索策略
#[derive(Clone)]
pub enum SearchStrategy {
    /// 标准KNN搜索，ef_search参数
    Standard { ef_search: usize },
    /// 范围搜索，给定距离阈值
    Range { radius: f32, max_elements: usize },
    /// 混合搜索，结合KNN与布尔过滤
    Hybrid { ef_search: usize, filter: Option<FilterFunction> },
}

impl std::fmt::Debug for SearchStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchStrategy::Standard { ef_search } => {
                f.debug_struct("Standard")
                    .field("ef_search", ef_search)
                    .finish()
            },
            SearchStrategy::Range { radius, max_elements } => {
                f.debug_struct("Range")
                    .field("radius", radius)
                    .field("max_elements", max_elements)
                    .finish()
            },
            SearchStrategy::Hybrid { ef_search, filter } => {
                f.debug_struct("Hybrid")
                    .field("ef_search", ef_search)
                    .field("filter", &filter.as_ref().map(|_| "<function>"))
                    .finish()
            },
        }
    }
}

impl Default for SearchStrategy {
    fn default() -> Self {
        SearchStrategy::Standard { ef_search: 64 }
    }
}

/// 过滤函数类型
pub type FilterFunction = Arc<dyn Fn(&VectorId, Option<&Metadata>) -> bool + Send + Sync>;

/// 搜索结果项
#[derive(Debug, Clone)]
pub struct SearchResultItem {
    /// 向量ID
    pub id: VectorId,
    /// 距离分数
    pub distance: Distance,
    /// 向量数据(可选)
    pub vector: Option<Vector>,
    /// 元数据(可选)
    pub metadata: Option<Metadata>,
}

impl SearchResultItem {
    /// 创建基本搜索结果项
    pub fn new(id: VectorId, distance: Distance) -> Self {
        Self {
            id,
            distance,
            vector: None,
            metadata: None,
        }
    }

    /// 创建包含数据的搜索结果项
    pub fn with_data(
        id: VectorId,
        distance: Distance,
        vector: Option<Vector>,
        metadata: Option<Metadata>,
    ) -> Self {
        Self {
            id,
            distance,
            vector,
            metadata,
        }
    }
}

/// 搜索结果集
pub type SearchResults = Vec<SearchResultItem>;

/// 统计信息
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    /// 总向量数
    pub num_vectors: usize,
    /// 已删除向量数
    pub num_deleted: usize,
    /// 维度
    pub dimension: usize,
    /// 最大层级
    pub max_level: usize,
    /// 当前入口节点
    pub entry_point: Option<NodeIndex>,
    /// 每层节点数
    pub nodes_per_level: HashMap<Level, usize>,
    /// 每个节点的平均连接数
    pub avg_connections_per_node: f32,
    /// 最小连接数
    pub min_connections: usize,
    /// 最大连接数
    pub max_connections: usize,
    /// 链接均匀度(0-1)，越接近1表示越均匀
    pub connections_uniformity: f32,
    /// 估算内存使用(字节)
    pub estimated_memory_usage: usize,
    /// 查询缓存统计
    pub cache_stats: Option<CacheStats>,
}

/// 缓存统计
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// 缓存大小(条目数)
    pub size: usize,
    /// 命中次数
    pub hits: usize,
    /// 未命中次数
    pub misses: usize,
    /// 缓存驱逐次数
    pub evictions: usize,
    /// 命中率
    pub hit_rate: f32,
}

/// 图数据结构状态
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GraphState {
    /// 节点总数
    pub node_count: usize,
    /// 边总数
    pub edge_count: usize,
    /// 图直径(最长路径)
    pub diameter: Option<usize>,
    /// 平均路径长度
    pub avg_path_length: Option<f32>,
    /// 图密度
    pub density: f32,
}

/// 坐标点，用于节点可视化
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub id: Uuid,
    pub layer: usize,
}

/// 节点连接，用于可视化
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Connection {
    pub from: Uuid,
    pub to: Uuid,
    pub weight: f32,
}

/// 可视化数据
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisualizationData {
    pub points: Vec<Point>,
    pub connections: Vec<Connection>,
}

/// 布尔过滤选项
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FilterOptions {
    /// 过滤表达式
    pub expression: String,
    /// 字段名
    pub field: String,
    /// 比较值
    pub value: String,
}

/// 范围搜索配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeSearchConfig {
    /// 距离阈值
    pub threshold: f32,
    /// 是否包含向量数据
    pub include_vectors: bool,
    /// 是否包含元数据
    pub include_metadata: bool,
    /// 最大结果数量限制
    pub limit: usize,
}

/// 索引构建进度
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildProgress {
    /// 总向量数
    pub total: usize,
    /// 已处理数量
    pub processed: usize,
    /// 开始时间（Unix时间戳）
    pub start_time: u64,
    /// 当前速度（项/秒）
    pub current_speed: f32,
}

/// HNSW索引参数
#[derive(Debug, Clone)]
pub struct HNSWParams {
    // 索引参数
}

/// 索引健康状况
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexHealth {
    /// 索引是否健康
    pub is_healthy: bool,
    /// 详细健康指标
    pub metrics: Vec<HealthMetric>,
    /// 健康检查时间
    pub checked_at: DateTime<Utc>,
}

/// 健康指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetric {
    /// 指标名称
    pub name: String,
    /// 是否通过
    pub passed: bool,
    /// 指标值
    pub value: String,
    /// 健康阈值
    pub threshold: String,
    /// 详细描述
    pub description: Option<String>,
}

/// 索引元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// 索引类型
    pub index_type: String,
    /// 向量维度
    pub dimension: usize,
    /// 距离计算方式
    pub metric: String,
    /// 索引参数
    pub parameters: IndexParameters,
    /// 向量数量
    pub size: usize,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 最后更新时间
    pub updated_at: DateTime<Utc>,
}

/// 索引参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexParameters {
    /// 构建时ef参数
    pub ef_construction: usize,
    /// 搜索时ef参数
    pub ef_search: usize,
    /// 最大层数
    pub max_layers: usize,
    /// 每个节点的最大连接数
    pub m: usize,
}

/// 压缩统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    /// 压缩前内存使用
    pub before_memory: usize,
    /// 压缩后内存使用
    pub after_memory: usize,
    /// 移除的连接数
    pub removed_links: usize,
    /// 移除的节点数
    pub removed_nodes: usize,
    /// 压缩耗时
    pub duration: Duration,
}

/// 节点链接
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeLinks {
    /// 每一层的连接列表，索引0是最底层
    pub links: Vec<Vec<usize>>,
}

/// 距离度量
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Metric {
    /// 欧几里得距离
    Euclidean,
    /// 余弦距离
    Cosine,
    /// 点积距离
    DotProduct,
}

/// 距离和ID对，用于搜索结果排序
#[derive(Debug, Clone, PartialEq)]
pub struct DistanceIdPair {
    /// 向量ID
    pub id: Uuid,
    /// 距离值
    pub distance: f32,
}

impl DistanceIdPair {
    /// 创建新的距离ID对
    pub fn new(id: Uuid, distance: f32) -> Self {
        Self { id, distance }
    }
}

impl PartialOrd for DistanceIdPair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Eq for DistanceIdPair {}

impl Ord for DistanceIdPair {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
    }
}

/// 节点连接信息
#[derive(Debug, Clone, PartialEq)]
pub struct NodeConnection {
    /// 节点ID
    pub node_id: usize,
    /// 距离
    pub distance: f32,
}

impl NodeConnection {
    /// 创建新的节点连接
    pub fn new(node_id: usize, distance: f32) -> Self {
        Self { node_id, distance }
    }
}

/// 向量添加或更新的结果
#[derive(Debug, PartialEq, Eq)]
pub enum VectorUpdateResult {
    /// 新向量添加成功
    Added,
    /// 已存在的向量被替换
    Replaced,
    /// 向量添加失败
    Failed,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_id_pair() {
        let id = Uuid::new_v4();
        let pair = DistanceIdPair::new(id, 0.5);
        
        assert_eq!(pair.id, id);
        assert_eq!(pair.distance, 0.5);
    }

    #[test]
    fn test_node_connection() {
        let conn = NodeConnection::new(5, 0.75);
        
        assert_eq!(conn.node_id, 5);
        assert_eq!(conn.distance, 0.75);
    }

    #[test]
    fn test_search_result_item() {
        let id = Uuid::new_v4();
        let basic = SearchResultItem::new(id, 0.25);
        
        assert_eq!(basic.id, id);
        assert_eq!(basic.distance, 0.25);
        assert!(basic.vector.is_none());
        assert!(basic.metadata.is_none());
        
        let vector = vec![1.0, 2.0, 3.0];
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), serde_json::json!("value"));
        
        let full = SearchResultItem::with_data(
            id, 
            0.25, 
            Some(vector.clone()), 
            Some(metadata.clone())
        );
        
        assert_eq!(full.id, id);
        assert_eq!(full.distance, 0.25);
        assert_eq!(full.vector, Some(vector));
        assert_eq!(full.metadata.unwrap().get("key").unwrap(), &serde_json::json!("value"));
    }

    #[test]
    fn test_distance_type_default() {
        let dt = DistanceType::default();
        assert_eq!(dt, DistanceType::Euclidean);
    }
    
    #[test]
    fn test_hnsw_config_default() {
        let config = HNSWConfig::default();
        assert_eq!(config.dimension, 128);
        assert_eq!(config.max_level, 16);
        assert_eq!(config.max_connections, 16);
        assert_eq!(config.ef_construction, 128);
        assert_eq!(config.ef_search, 64);
        assert_eq!(config.distance_type, DistanceType::Euclidean);
        assert!(config.use_cache);
        assert!(config.parallel_search);
    }
    
    #[test]
    fn test_search_config_default() {
        let config = SearchConfig::default();
        assert_eq!(config.limit, 10);
        assert!(!config.include_vectors);
        assert!(!config.include_metadata);
        
        match config.strategy {
            SearchStrategy::Standard { ef_search } => {
                assert_eq!(ef_search, 64);
            },
            other => {
                panic!("Default strategy should be Standard, got {:?}", other);
            }
        }
    }
}
