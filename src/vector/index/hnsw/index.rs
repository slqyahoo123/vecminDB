//! HNSW索引核心实现
//!
//! 本模块包含HNSWIndex核心结构体及其基本操作方法。

use crate::vector::index::hnsw::types::*;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use rand::rngs::ThreadRng;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::mem::size_of;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, trace, warn};
use uuid::Uuid;

/// HNSW索引节点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWNode {
    /// 节点ID
    pub id: Uuid,
    
    /// 节点所在层级
    pub level: usize,
    
    /// 每层连接（邻居节点）
    pub connections: Vec<Vec<(Uuid, f32)>>,
    
    /// 节点对应的向量
    pub vector: Vec<f32>,
    
    /// 节点关联的元数据
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    
    /// 删除标记
    pub deleted: bool,
}

impl HNSWNode {
    /// 创建新节点
    pub fn new(id: Uuid, vector: Vec<f32>, level: usize, metadata: Option<serde_json::Value>) -> Self {
        let mut connections = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            connections.push(Vec::new());
        }
        
        Self {
            id,
            level,
            connections,
            vector,
            metadata,
            deleted: false,
        }
    }
    
    /// 估算该节点的内存占用（字节）
    pub fn estimate_memory_usage(&self) -> usize {
        // 基本结构体大小
        let mut size = size_of::<Self>();
        
        // 向量大小
        size += self.vector.capacity() * size_of::<f32>();
        
        // 连接大小
        for conn_layer in &self.connections {
            size += conn_layer.capacity() * size_of::<(Uuid, f32)>();
        }
        
        // 元数据大小（粗略估计）
        if let Some(metadata) = &self.metadata {
            // 元数据序列化后的大小作为估计
            if let Ok(json) = serde_json::to_string(metadata) {
                size += json.len();
            }
        }
        
        size
    }
    
    /// 添加连接到指定层
    pub fn add_connection(&mut self, to_id: Uuid, distance: f32, layer: usize) {
        if layer <= self.level {
            self.connections[layer].push((to_id, distance));
        }
    }
    
    /// 获取指定层的所有连接
    pub fn get_connections(&self, layer: usize) -> Option<&Vec<(Uuid, f32)>> {
        if layer <= self.level {
            Some(&self.connections[layer])
        } else {
            None
        }
    }
    
    /// 获取指定层的连接数量
    pub fn connection_count(&self, layer: usize) -> usize {
        if layer <= self.level {
            self.connections[layer].len()
        } else {
            0
        }
    }
    
    /// 设置连接列表（替换现有连接）
    pub fn set_connections(&mut self, connections: Vec<(Uuid, f32)>, layer: usize) {
        if layer <= self.level {
            self.connections[layer] = connections;
        }
    }
}

/// HNSW索引实现
#[derive(Debug, Serialize, Deserialize)]
pub struct HNSWIndex {
    /// 索引配置
    pub config: HNSWConfig,
    
    /// 节点集合
    #[serde(skip)]
    nodes: Arc<RwLock<HashMap<Uuid, HNSWNode>>>,
    
    /// 持久化的节点数据（用于序列化/反序列化）
    #[serde(skip_serializing_if = "Option::is_none")]
    persisted_nodes: Option<Vec<HNSWNode>>,
    
    /// 入口节点ID（每层首节点）
    pub entry_point: Option<Uuid>,
    
    /// 最大层级
    pub max_level: usize,
    
    /// 向量维度
    pub dimensions: usize,
    
    /// 向量数量
    pub vector_count: usize,
    
    /// 已删除向量数量
    pub deleted_count: usize,
    
    /// 是否需要重建
    pub need_rebuild: bool,
    
    /// 搜索计数（统计用）
    #[serde(skip)]
    search_count: Arc<RwLock<usize>>,
    
    /// 总搜索时间（统计用，毫秒）
    #[serde(skip)]
    total_search_time_ms: Arc<RwLock<f64>>,
    
    /// 最后重建时间
    #[serde(skip)]
    last_rebuild_time: Arc<RwLock<Option<Instant>>>,
    
    /// 最后重建耗时（毫秒）
    pub last_rebuild_duration_ms: f64,
}

impl HNSWIndex {
    /// 创建新的HNSW索引
    pub fn new(dimensions: usize, config: HNSWConfig) -> Self {
        let hnsw = Self {
            config,
            nodes: Arc::new(RwLock::new(HashMap::new())),
            persisted_nodes: None,
            entry_point: None,
            max_level: 0,
            dimensions,
            vector_count: 0,
            deleted_count: 0,
            need_rebuild: false,
            search_count: Arc::new(RwLock::new(0)),
            total_search_time_ms: Arc::new(RwLock::new(0.0)),
            last_rebuild_time: Arc::new(RwLock::new(None)),
            last_rebuild_duration_ms: 0.0,
        };
        
        info!("Created new HNSW index with dimensions={}, max_connections={}, ef_construction={}",
            dimensions, hnsw.config.max_connections, hnsw.config.ef_construction);
        
        hnsw
    }
    
    /// 获取节点读锁
    pub fn read_nodes(&self) -> RwLockReadGuard<HashMap<Uuid, HNSWNode>> {
        self.nodes.read()
    }
    
    /// 获取节点写锁
    pub fn write_nodes(&self) -> RwLockWriteGuard<HashMap<Uuid, HNSWNode>> {
        self.nodes.write()
    }
    
    /// 随机生成节点层级
    fn get_random_level(&self, rng: &mut ThreadRng) -> usize {
        // 以1/e的概率上升一层，最大层数为32
        const LEVEL_PROBABILITY: f64 = 1.0 / std::f64::consts::E;
        let mut level = 0;
        while rng.gen::<f64>() < LEVEL_PROBABILITY && level < 32 {
            level += 1;
        }
        level
    }
    
    /// 添加向量到索引
    pub fn add(&mut self, vector_id: Uuid, vector: Vec<f32>, metadata: Option<serde_json::Value>) -> Result<(), String> {
        if vector.len() != self.dimensions {
            return Err(format!("Vector dimension mismatch: expected {}, got {}", 
                self.dimensions, vector.len()));
        }
        
        let mut rng = rand::thread_rng();
        let level = self.get_random_level(&mut rng);
        
        // 创建新节点
        let new_node = HNSWNode::new(vector_id, vector, level, metadata);
        
        let mut nodes = self.write_nodes();
        
        // 检查是否已存在
        if nodes.contains_key(&vector_id) {
            if self.config.allow_replace {
                // 如果允许替换，删除旧节点的所有连接
                self.remove_connections(&mut nodes, &vector_id);
                nodes.remove(&vector_id);
            } else {
                return Err(format!("Vector with ID {} already exists", vector_id));
            }
        }
        
        // 更新最大层级
        if level > self.max_level {
            self.max_level = level;
        }
        
        // 如果是第一个节点，设为入口点
        if nodes.is_empty() {
            nodes.insert(vector_id, new_node);
            self.entry_point = Some(vector_id);
            self.vector_count = 1;
            return Ok(());
        }
        
        // 插入新节点
        nodes.insert(vector_id, new_node);
        
        // 更新计数
        self.vector_count += 1;
        
        // 设置重建标志
        if self.vector_count % 1000 == 0 {
            self.need_rebuild = true;
        }
        
        Ok(())
    }
    
    /// 从索引中移除连接引用
    fn remove_connections(&self, nodes: &mut HashMap<Uuid, HNSWNode>, vector_id: &Uuid) {
        // 遍历所有节点，移除对该向量的连接
        for node in nodes.values_mut() {
            for layer_connections in &mut node.connections {
                layer_connections.retain(|(id, _)| id != vector_id);
            }
        }
    }
    
    /// 从索引中移除向量
    pub fn remove(&mut self, vector_id: &Uuid) -> bool {
        let mut nodes = self.write_nodes();
        
        // 检查向量是否存在
        if !nodes.contains_key(vector_id) {
            return false;
        }
        
        // 获取节点
        if let Some(node) = nodes.get_mut(vector_id) {
            // 标记为删除
            node.deleted = true;
            self.deleted_count += 1;
            
            // 如果删除比例超过20%，标记需要重建
            if self.deleted_count as f32 / self.vector_count as f32 > 0.2 {
                self.need_rebuild = true;
            }
            
            true
        } else {
            false
        }
    }
    
    /// 重置索引
    pub fn reset(&mut self) {
        let mut nodes = self.write_nodes();
        nodes.clear();
        drop(nodes);
        
        self.entry_point = None;
        self.max_level = 0;
        self.vector_count = 0;
        self.deleted_count = 0;
        self.need_rebuild = false;
        
        *self.search_count.write() = 0;
        *self.total_search_time_ms.write() = 0.0;
        *self.last_rebuild_time.write() = None;
        
        info!("HNSW index reset");
    }
    
    /// 获取索引统计信息
    pub fn get_stats(&self) -> HNSWStats {
        let nodes = self.read_nodes();
        let active_count = self.vector_count - self.deleted_count;
        
        // 统计连接信息
        let mut total_connections = 0;
        let mut min_connections = usize::MAX;
        let mut max_connections = 0;
        let mut memory_usage = 0;
        
        for node in nodes.values() {
            if node.deleted {
                continue;
            }
            
            let node_connections = node.connections.iter()
                .map(|layer| layer.len())
                .sum::<usize>();
                
            total_connections += node_connections;
            min_connections = min_connections.min(node_connections);
            max_connections = max_connections.max(node_connections);
            
            // 估算内存使用
            memory_usage += node.estimate_memory_usage();
        }
        
        // 计算平均连接数
        let avg_connections = if active_count > 0 {
            total_connections as f32 / active_count as f32
        } else {
            0.0
        };
        
        // 计算平均搜索时间
        let search_count = *self.search_count.read();
        let avg_search_time_ms = if search_count > 0 {
            *self.total_search_time_ms.read() / search_count as f64
        } else {
            0.0
        };
        
        // 计算索引健康度分数（简单实现）
        let delete_ratio = if self.vector_count > 0 {
            self.deleted_count as f32 / self.vector_count as f32
        } else {
            0.0
        };
        
        // 健康度评分（简化版）：删除率越低，分数越高
        let health_score = 1.0 - delete_ratio;
        
        HNSWStats {
            vector_count: active_count,
            dimensions: self.dimensions,
            avg_connections,
            min_connections: min_connections.min(usize::MAX - 1),
            max_connections,
            layers: self.max_level + 1,
            memory_usage,
            cache_hit_rate: 0.0, // 暂不实现缓存统计
            health_score,
            avg_search_time_ms: avg_search_time_ms as f32,
            recent_build_time_ms: self.last_rebuild_duration_ms as f32,
        }
    }
    
    /// 检查索引是否需要重建
    pub fn needs_rebuild(&self) -> bool {
        self.need_rebuild
    }
    
    /// 计算两个向量间的距离
    pub fn calculate_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.distance_type {
            DistanceType::Euclidean => {
                // 欧几里得距离
                a.iter().zip(b.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt()
            },
            DistanceType::DotProduct => {
                // 点积距离（1 - 点积，使得越小越相似）
                1.0 - a.iter().zip(b.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
            },
            DistanceType::Cosine => {
                // 余弦距离
                let dot_product = a.iter().zip(b.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
                    
                let a_norm = a.iter()
                    .map(|a| a.powi(2))
                    .sum::<f32>()
                    .sqrt();
                    
                let b_norm = b.iter()
                    .map(|b| b.powi(2))
                    .sum::<f32>()
                    .sqrt();
                    
                1.0 - (dot_product / (a_norm * b_norm).max(1e-10))
            },
            DistanceType::Manhattan => {
                // 曼哈顿距离
                a.iter().zip(b.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum()
            }
        }
    }
    
    /// 将距离转换为相似度分数
    pub fn distance_to_score(&self, distance: f32) -> f32 {
        match self.config.distance_type {
            DistanceType::Euclidean => {
                // 欧几里得距离转换为相似度
                1.0 / (1.0 + distance)
            },
            DistanceType::DotProduct => {
                // 点积距离转换为相似度
                1.0 - distance
            },
            DistanceType::Cosine => {
                // 余弦距离转换为相似度
                1.0 - distance
            },
            DistanceType::Manhattan => {
                // 曼哈顿距离转换为相似度
                1.0 / (1.0 + distance)
            }
        }
    }
    
    /// 记录搜索统计信息
    pub fn record_search_time(&self, duration: Duration) {
        let mut search_count = self.search_count.write();
        *search_count += 1;
        
        let mut total_time = self.total_search_time_ms.write();
        *total_time += duration.as_secs_f64() * 1000.0;
    }
    
    /// 获取向量的维度
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
    
    /// 获取索引中的向量数量
    pub fn size(&self) -> usize {
        self.vector_count - self.deleted_count
    }
    
    /// 获取指定ID的向量
    pub fn get_vector(&self, id: &Uuid) -> Option<Vec<f32>> {
        let nodes = self.read_nodes();
        nodes.get(id).map(|node| node.vector.clone())
    }
    
    /// 获取指定ID的元数据
    pub fn get_metadata(&self, id: &Uuid) -> Option<serde_json::Value> {
        let nodes = self.read_nodes();
        nodes.get(id).and_then(|node| node.metadata.clone())
    }
    
    /// 检查向量是否存在于索引中
    pub fn contains(&self, id: &Uuid) -> bool {
        let nodes = self.read_nodes();
        if let Some(node) = nodes.get(id) {
            !node.deleted
        } else {
            false
        }
    }
    
    /// 更新向量的元数据
    pub fn update_metadata(&mut self, id: &Uuid, metadata: serde_json::Value) -> bool {
        let mut nodes = self.write_nodes();
        if let Some(node) = nodes.get_mut(id) {
            if !node.deleted {
                node.metadata = Some(metadata);
                return true;
            }
        }
        false
    }
}
