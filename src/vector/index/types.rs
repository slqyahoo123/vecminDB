// 向量索引类型定义
// 包含索引类型枚举、配置结构和搜索结果类型

use std::cmp::Ordering;
use serde::{Serialize, Deserialize};
use crate::vector::operations::SimilarityMetric;
use std::collections::HashMap;

/// 向量搜索结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub metadata: Option<serde_json::Value>,
}

impl SearchResult {
    /// 获取相似度分数，作为distance的别名
    /// 这是为了保持与VectorSearchResult中的score字段一致性
    pub fn score(&self) -> f32 {
        self.distance
    }
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.distance.eq(&other.distance)
    }
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// 向量索引类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexType {
    /// 暴力搜索（无索引）
    Flat,
    /// 分层导航小世界图索引
    HNSW,
    /// 倒排文件索引
    IVF,
    /// 乘积量化索引
    PQ,
    /// IVF与PQ组合索引
    IVFPQ,
    /// IVF与HNSW组合索引
    IVFHNSW,
    /// 局部敏感哈希索引
    LSH,
    /// 随机投影树索引
    ANNOY,
    /// 邻域图索引
    NGT,
    /// 分层聚类索引
    HierarchicalClustering,
    /// 基于图的索引
    GraphIndex,
    /// VP-Tree索引
    VPTree,
}

impl Default for IndexType {
    fn default() -> Self {
        IndexType::Flat
    }
}

/// 向量索引配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexConfig {
    pub index_type: IndexType,
    pub metric: SimilarityMetric,
    /// 为兼容历史代码，保留同义字段。与 metric 保持一致
    pub similarity_metric: SimilarityMetric,
    pub dimension: usize,
    /// 通用搜索候选数量（部分索引会使用）
    pub search_k: usize,
    /// NGT专用边大小别名（兼容历史）
    pub ngt_edge_size: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_m: usize,
    pub hnsw_ef_search: usize,
    pub ivf_nlist: usize,
    pub ivf_nprobe: usize,
    pub ivf_centers: usize,
    pub pq_subvector_count: usize,
    pub pq_subvector_bits: usize,
    pub pq_m: usize,
    pub pq_nbits: usize,
    pub pq_subvectors: usize,
    pub lsh_hash_count: usize,
    pub lsh_hash_length: usize,
    pub lsh_ntables: usize,
    pub lsh_multi_probe: usize,
    pub annoy_tree_count: usize,
    pub graph_degree: usize,
    /// 层次聚类相关配置（兼容历史字段）
    pub cluster_levels: usize,
    pub clusters_per_level: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub max_layers: usize,
    pub expected_elements: usize,
    pub ivfhnsw_nlist: usize,
    pub ivfhnsw_nprobe: usize,
    pub ivfhnsw_m: usize,
    pub ivfhnsw_ef_construction: usize,
    pub ivfhnsw_ef_search: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        IndexConfig {
            index_type: IndexType::default(),
            metric: SimilarityMetric::default(),
            similarity_metric: SimilarityMetric::default(),
            dimension: 128,
            search_k: 0,
            ngt_edge_size: 0,
            hnsw_ef_construction: 200,
            hnsw_m: 16,
            hnsw_ef_search: 50,
            ivf_nlist: 100,
            ivf_nprobe: 0,
            ivf_centers: 0,
            pq_subvector_count: 8,
            pq_subvector_bits: 8,
            pq_m: 0,
            pq_nbits: 0,
            pq_subvectors: 0,
            lsh_hash_count: 10,
            lsh_hash_length: 32,
            lsh_ntables: 0,
            lsh_multi_probe: 0,
            annoy_tree_count: 10,
            graph_degree: 16,
            cluster_levels: 0,
            clusters_per_level: 0,
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            max_layers: 16,
            expected_elements: 0,
            ivfhnsw_nlist: 0,
            ivfhnsw_nprobe: 0,
            ivfhnsw_m: 0,
            ivfhnsw_ef_construction: 0,
            ivfhnsw_ef_search: 0,
        }
    }
}

impl IndexConfig {
    /// 设置索引参数
    pub fn set_param(&mut self, param_name: &str, value: usize) {
        match param_name {
            "dimension" => self.dimension = value,
            "ef_construction" => {
                self.ef_construction = value;
                self.hnsw_ef_construction = value;
            }
            "m" => {
                self.m = value;
                self.hnsw_m = value;
            }
            "search_k" => self.search_k = value,
            "ngt_edge_size" => self.ngt_edge_size = value,
            "ef" => self.hnsw_ef_search = value,
            "nlist" => self.ivf_nlist = value,
            "nprobe" => self.ivf_nprobe = value,
            "pq_m" => self.pq_m = value,
            "nbits" => self.pq_nbits = value,
            "ntables" => self.lsh_ntables = value,
            "multi_probe" => self.lsh_multi_probe = value,
            "subvector_count" => self.pq_subvector_count = value,
            "subvector_bits" => self.pq_subvector_bits = value,
            "hash_count" => self.lsh_hash_count = value,
            "hash_length" => self.lsh_hash_length = value,
            "tree_count" => self.annoy_tree_count = value,
            "degree" => self.graph_degree = value,
            "cluster_levels" => self.cluster_levels = value,
            "clusters_per_level" => self.clusters_per_level = value,
            "max_layers" => self.max_layers = value,
            "expected_elements" => self.expected_elements = value,
            "ivfhnsw_nlist" => self.ivfhnsw_nlist = value,
            "ivfhnsw_nprobe" => self.ivfhnsw_nprobe = value,
            "ivfhnsw_m" => self.ivfhnsw_m = value,
            "ivfhnsw_ef_construction" => self.ivfhnsw_ef_construction = value,
            "ivfhnsw_ef_search" => self.ivfhnsw_ef_search = value,
            _ => {}
        }
    }
}

/// 索引性能指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetrics {
    /// 索引大小（向量数量）
    pub size: usize,
    /// 索引维度
    pub dimension: usize,
    /// 内存使用量（字节）
    pub memory_usage: usize,
    /// 索引构建时间（毫秒）
    pub build_time_ms: u64,
    /// 平均搜索时间（微秒）
    pub avg_search_time_us: f64,
    /// 索引类型
    pub index_type: IndexType,
    /// 相似度度量
    pub metric: SimilarityMetric,
    /// 召回率（如果有基准测试）
    pub recall: Option<f64>,
    /// QPS（每秒查询数）
    pub qps: Option<f64>,
    /// 最后更新时间
    pub last_updated: chrono::DateTime<chrono::Utc>,
    /// 额外指标
    pub additional_metrics: HashMap<String, f64>,
}

impl Default for IndexMetrics {
    fn default() -> Self {
        Self {
            size: 0,
            dimension: 0,
            memory_usage: 0,
            build_time_ms: 0,
            avg_search_time_us: 0.0,
            index_type: IndexType::Flat,
            metric: SimilarityMetric::Cosine,
            recall: None,
            qps: None,
            last_updated: chrono::Utc::now(),
            additional_metrics: HashMap::new(),
        }
    }
}

impl IndexMetrics {
    /// 创建新的索引指标
    pub fn new(index_type: IndexType, metric: SimilarityMetric, dimension: usize) -> Self {
        Self {
            index_type,
            metric,
            dimension,
            last_updated: chrono::Utc::now(),
            ..Default::default()
        }
    }
    
    /// 更新索引大小
    pub fn update_size(&mut self, size: usize) {
        self.size = size;
        self.last_updated = chrono::Utc::now();
    }
    
    /// 更新内存使用量
    pub fn update_memory_usage(&mut self, memory_bytes: usize) {
        self.memory_usage = memory_bytes;
        self.last_updated = chrono::Utc::now();
    }
    
    /// 更新搜索时间
    pub fn update_search_time(&mut self, time_us: f64) {
        self.avg_search_time_us = time_us;
        self.last_updated = chrono::Utc::now();
    }
    
    /// 设置QPS
    pub fn set_qps(&mut self, qps: f64) {
        self.qps = Some(qps);
        self.last_updated = chrono::Utc::now();
    }
    
    /// 设置召回率
    pub fn set_recall(&mut self, recall: f64) {
        self.recall = Some(recall);
        self.last_updated = chrono::Utc::now();
    }
    
    /// 添加自定义指标
    pub fn add_metric(&mut self, name: &str, value: f64) {
        self.additional_metrics.insert(name.to_string(), value);
        self.last_updated = chrono::Utc::now();
    }
} 