//! HNSW索引的统计信息收集模块
//!
//! 本模块实现了HNSW索引的统计信息收集和计算功能，包括节点分布、
//! 连接统计、内存使用估算等功能。

use std::collections::HashMap;
use std::sync::Arc;
use rand::{thread_rng, Rng};
// 该模块仅在调试时使用日志，避免未使用警告

use crate::vector::index::hnsw::node::SharedNode;
use crate::vector::index::hnsw::types::{Distance, DistanceType, NodeIndex};
use crate::vector::index::hnsw::distance;
use crate::core::interfaces::monitoring::PerformanceMonitor;
use crate::Result;

/// 搜索配置结构体
#[derive(Clone)]
pub struct SearchConfig {
    /// 搜索策略
    pub strategy: SearchStrategy,
    /// 返回结果限制
    pub limit: usize,
    /// 搜索效率参数，控制搜索广度
    pub ef_search: Option<usize>,
    /// 向量过滤函数
    pub filter: Option<Arc<dyn Fn(&uuid::Uuid, Option<&HashMap<String, String>>) -> bool + Send + Sync>>,
    /// 是否包含向量数据
    pub with_vectors: bool,
    /// 是否包含元数据
    pub with_metadata: bool,
}

impl std::fmt::Debug for SearchConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchConfig")
            .field("strategy", &self.strategy)
            .field("limit", &self.limit)
            .field("ef_search", &self.ef_search)
            .field("filter", &self.filter.as_ref().map(|_| "<function>"))
            .field("with_vectors", &self.with_vectors)
            .field("with_metadata", &self.with_metadata)
            .finish()
    }
}

impl SearchConfig {
    /// 创建新的搜索配置
    pub fn new() -> Self {
        Self {
            strategy: SearchStrategy::Standard,
            limit: 10,
            ef_search: None,
            filter: None,
            with_vectors: false,
            with_metadata: false,
        }
    }

    /// 设置结果数量限制
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// 设置搜索效率参数
    pub fn with_ef(mut self, ef: usize) -> Self {
        self.ef_search = Some(ef);
        self
    }

    /// 设置搜索策略
    pub fn with_strategy(mut self, strategy: SearchStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// 设置过滤函数
    pub fn with_filter<F>(mut self, filter: F) -> Self
    where
        F: Fn(&uuid::Uuid, Option<&HashMap<String, String>>) -> bool + Send + Sync + 'static,
    {
        self.filter = Some(Arc::new(filter));
        self
    }

    /// 设置是否包含向量数据
    pub fn with_vectors(mut self, include: bool) -> Self {
        self.with_vectors = include;
        self
    }

    /// 设置是否包含元数据
    pub fn with_metadata(mut self, include: bool) -> Self {
        self.with_metadata = include;
        self
    }
}

/// 搜索策略枚举
#[derive(Clone, Debug)]
pub enum SearchStrategy {
    /// 标准搜索策略
    Standard,
    /// 启发式搜索策略
    Heuristic,
}

/// 搜索结果结构体
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// 向量ID
    pub id: uuid::Uuid,
    /// 与查询向量的距离
    pub distance: Distance,
    /// 可选的向量数据
    pub vector: Option<Vec<f32>>,
    /// 可选的元数据
    pub metadata: Option<HashMap<String, String>>,
}

/// 将统计结果上报到统一性能监控接口
/// 通过抽象接口解除对 training 模块的直接依赖
pub async fn publish_hnsw_stats<M: PerformanceMonitor + ?Sized>(
    monitor: &M,
    stats: &HashMap<String, String>,
    index_name: &str,
) -> Result<()> {
    let mut tags = HashMap::new();
    tags.insert("component".to_string(), "vector_hnsw".to_string());
    tags.insert("index".to_string(), index_name.to_string());

    // 选取核心统计指标进行上报
    for key in [
        "total_nodes",
        "active_nodes",
        "deleted_nodes",
        "min_links",
        "max_links",
        "avg_links",
        "total_links",
        "uniformity",
        "estimated_memory_bytes",
    ] {
        if let Some(value_str) = stats.get(key) {
            if let Ok(value) = value_str.parse::<f64>() {
                monitor.record_metric(key, value, &tags).await?;
            } else if let Ok(value_u) = value_str.parse::<u64>() {
                monitor.record_metric(key, value_u as f64, &tags).await?;
            } else if let Ok(value_usize) = value_str.parse::<usize>() {
                monitor.record_metric(key, value_usize as f64, &tags).await?;
            } else if let Ok(value_f) = value_str.parse::<f32>() {
                monitor.record_metric(key, value_f as f64, &tags).await?;
            }
        }
    }

    // 逐层节点数量（level_X_nodes）
    for (k, v) in stats.iter() {
        if k.starts_with("level_") && k.ends_with("_nodes") {
            if let Ok(value) = v.parse::<f64>() {
                monitor.record_metric(k, value, &tags).await?;
            }
        }
    }

    Ok(())
}

/// 强类型化的 HNSW 统计结果，便于后续处理
#[derive(Clone, Debug, Default)]
pub struct HnswStats {
    pub total_nodes: usize,
    pub active_nodes: usize,
    pub deleted_nodes: usize,
    pub max_level: usize,
    pub min_links: usize,
    pub max_links: usize,
    pub avg_links: f32,
    pub total_links: usize,
    pub uniformity: f32,
    pub estimated_memory_bytes: usize,
}

impl HnswStats {
    pub fn from_map(stats: &HashMap<String, String>) -> Option<Self> {
        Some(Self {
            total_nodes: stats.get("total_nodes")?.parse().ok()?,
            active_nodes: stats.get("active_nodes")?.parse().ok()?,
            deleted_nodes: stats.get("deleted_nodes")?.parse().ok()?,
            max_level: stats.get("max_level")?.parse().ok()?,
            min_links: stats.get("min_links")?.parse().ok()?,
            max_links: stats.get("max_links")?.parse().ok()?,
            avg_links: stats.get("avg_links")?.parse().ok()?,
            total_links: stats.get("total_links")?.parse().ok()?,
            uniformity: stats.get("uniformity")?.parse().ok()?,
            estimated_memory_bytes: stats.get("estimated_memory_bytes")?.parse().ok()?,
        })
    }
}

/// 计算索引统计信息
pub fn get_stats(nodes: &[SharedNode], max_level: usize, entry_point: Option<NodeIndex>) -> HashMap<String, String> {
    let mut stats = HashMap::new();
    
    // 基本统计信息
    let total_nodes = nodes.len();
    let mut active_nodes = 0;
    let mut deleted_nodes = 0;
    
    // 连接统计
    let mut min_links = usize::MAX;
    let mut max_links = 0;
    let mut total_links = 0;
    
    // 每层节点计数
    let mut nodes_per_level = vec![0; max_level + 1];
    
    // 收集节点统计信息
    for node in nodes {
        let node_read = node.read();
        
        if node_read.marked_deleted {
            deleted_nodes += 1;
            continue;
        }
        
        active_nodes += 1;
        
        // 计算节点连接数
        let mut node_links = 0;
        for level in 0..=node_read.level {
            if let Some(connections) = node_read.get_connections(level) {
                node_links += connections.len();
            }
        }
        
        total_links += node_links;
        min_links = min_links.min(node_links);
        max_links = max_links.max(node_links);
        
        // 更新层级统计
        nodes_per_level[node_read.level] += 1;
    }
    
    // 计算平均连接数
    let avg_links = if active_nodes > 0 {
        total_links as f32 / active_nodes as f32
    } else {
        0.0
    };
    
    // 计算连接分布均匀性
    let mut uniformity = 1.0;
    if active_nodes > 1 && max_links > min_links {
        let link_variance = calculate_link_variance(nodes, avg_links);
        let max_variance = (max_links as f32 - avg_links).powi(2);
        uniformity = 1.0 - (link_variance / max_variance).sqrt();
    }
    
    // 估算内存使用
    let memory_usage = estimate_memory_usage(nodes);
    
    // 填充统计结果
    stats.insert("total_nodes".to_string(), total_nodes.to_string());
    stats.insert("active_nodes".to_string(), active_nodes.to_string());
    stats.insert("deleted_nodes".to_string(), deleted_nodes.to_string());
    stats.insert("max_level".to_string(), max_level.to_string());
    
    if let Some(ep) = entry_point {
        stats.insert("entry_point".to_string(), ep.to_string());
    }
    
    stats.insert("min_links".to_string(), min_links.to_string());
    stats.insert("max_links".to_string(), max_links.to_string());
    stats.insert("avg_links".to_string(), format!("{:.2}", avg_links));
    stats.insert("total_links".to_string(), total_links.to_string());
    stats.insert("uniformity".to_string(), format!("{:.4}", uniformity));
    stats.insert("estimated_memory_bytes".to_string(), memory_usage.to_string());
    
    // 添加每层节点数统计
    for (level, count) in nodes_per_level.iter().enumerate() {
        stats.insert(format!("level_{}_nodes", level), count.to_string());
    }
    
    stats
}

/// 计算连接数的方差
fn calculate_link_variance(nodes: &[SharedNode], avg_links: f32) -> f32 {
    let mut sum_squared_diff = 0.0;
    let mut valid_nodes = 0;
    
    for node in nodes {
        let node_read = node.read();
        
        if node_read.marked_deleted {
            continue;
        }
        
        valid_nodes += 1;
        
        // 计算节点连接数
        let mut node_links = 0;
        for level in 0..=node_read.level {
            if let Some(connections) = node_read.get_connections(level) {
                node_links += connections.len();
            }
        }
        
        let diff = node_links as f32 - avg_links;
        sum_squared_diff += diff * diff;
    }
    
    if valid_nodes > 1 {
        sum_squared_diff / valid_nodes as f32
    } else {
        0.0
    }
}

/// 估算索引内存使用
fn estimate_memory_usage(nodes: &[SharedNode]) -> usize {
    let mut total_memory = 0;
    
    // 基础结构大小
    let base_structure_size = std::mem::size_of::<HashMap<uuid::Uuid, NodeIndex>>();
    total_memory += base_structure_size;
    
    // 节点内存
    for node in nodes {
        let node_read = node.read();
        // 节点基础大小
        let node_size = std::mem::size_of::<uuid::Uuid>() + 
                        std::mem::size_of::<usize>() * 3 +
                        std::mem::size_of::<bool>();
                        
        // 向量数据大小
        let vector_size = node_read.vector.len() * std::mem::size_of::<f32>();
        
        // 连接数据大小
        let mut connections_size = 0;
        for level in 0..=node_read.level {
            if let Some(connections) = node_read.get_connections(level) {
                connections_size += connections.len() * 
                    (std::mem::size_of::<NodeIndex>() + std::mem::size_of::<Distance>());
            }
        }
        
        total_memory += node_size + vector_size + connections_size;
    }
    
    total_memory
}

/// 计算索引中向量的平均距离
pub fn calculate_average_distance(
    nodes: &[SharedNode], 
    distance_type: DistanceType,
    sample_size: Option<usize>
) -> Option<f32> {
    let active_nodes: Vec<_> = nodes.iter()
        .filter_map(|n| {
            let node = n.read();
            if !node.marked_deleted {
                Some(node.node_index)
            } else {
                None
            }
        })
        .collect();
        
    let node_count = active_nodes.len();
    
    if node_count < 2 {
        return None;
    }
    
    // 确定采样大小
    let samples = sample_size.unwrap_or_else(|| {
        let max_pairs = 10000;
        let actual_pairs = node_count * (node_count - 1) / 2;
        if actual_pairs <= max_pairs {
            actual_pairs
        } else {
            max_pairs
        }
    });
    
    // 随机采样节点对
    let mut rng = thread_rng();
    let mut total_distance = 0.0;
    let mut sample_count = 0;
    
    for _ in 0..samples {
        // 随机选择两个不同的节点
        let idx1 = rng.gen_range(0..node_count);
        let mut idx2 = rng.gen_range(0..node_count);
        while idx2 == idx1 {
            idx2 = rng.gen_range(0..node_count);
        }
        
        let node_idx1 = active_nodes[idx1];
        let node_idx2 = active_nodes[idx2];
        
        // 获取节点向量并计算距离
        let node1 = nodes[node_idx1].read();
        let node2 = nodes[node_idx2].read();
        let dist = distance::calculate_distance(&node1.vector, &node2.vector, distance_type);
        total_distance += dist;
        sample_count += 1;
    }
    
    if sample_count > 0 {
        Some(total_distance / sample_count as f32)
    } else {
        None
    }
}
