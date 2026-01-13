//! HNSW索引构建与管理模块
//!
//! 本模块实现了HNSW索引的构建、维护和基本操作，包括:
//! - 索引构建器配置
//! - 向量添加和删除
//! - 索引持久化
//! - 资源管理

use std::collections::HashMap;
use std::sync::Arc;
use rand::{SeedableRng, Rng};
use rand::rngs::StdRng;
use rand_distr::Distribution;
use parking_lot::RwLock;
// use tracing::{debug, trace, warn};
// use serde::{Serialize, Deserialize};
// use ndarray::{Array, Array1, Array2, ArrayView1, Axis};
// use std::time::Instant;
// 本模块使用本地宏别名，以保持与历史日志调用兼容
use log as _log;
macro_rules! log_debug { ($($arg:tt)*) => { _log::debug!($($arg)*) }; }

use uuid::Uuid;
use crate::vector::index::hnsw::distance;
use crate::vector::index::hnsw::node::{HNSWNode, SharedNode};
use crate::vector::index::hnsw::types::{
    Distance, DistanceType, NodeIndex, 
    Vector, VectorId
};
// use crate::vector::index::utils::MaxHeap;

/// HNSW索引构建器
pub struct HNSWBuilder {
    /// 向量维度
    dimension: usize,
    /// 每层最大连接数
    m: usize,
    /// 构建时每层扩展因子
    ef_construction: usize,
    /// 距离计算类型
    distance_type: DistanceType,
    /// 层级分布参数
    ml: f32,
    /// 最大层数限制
    max_level_limit: usize,
    /// 索引ID
    index_id: String,
}

impl HNSWBuilder {
    /// 创建新的索引构建器
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            m: 16,                   // 默认连接数
            ef_construction: 200,    // 默认构建扩展因子
            distance_type: DistanceType::Euclidean,
            ml: 1.0 / 2.0_f32.ln(),      // 默认层级分布参数
            max_level_limit: 16,     // 默认最大层数
            index_id: "hnsw_default".to_string(),
        }
    }

    /// 设置每层最大连接数
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// 设置构建时扩展因子
    pub fn with_ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }

    /// 设置距离计算类型
    pub fn with_distance_type(mut self, distance_type: DistanceType) -> Self {
        self.distance_type = distance_type;
        self
    }

    /// 设置层级分布参数
    pub fn with_ml(mut self, ml: f32) -> Self {
        self.ml = ml;
        self
    }

    /// 设置最大层数限制
    pub fn with_max_level_limit(mut self, max_level_limit: usize) -> Self {
        self.max_level_limit = max_level_limit;
        self
    }

    /// 设置索引ID
    pub fn with_index_id(mut self, index_id: String) -> Self {
        self.index_id = index_id;
        self
    }

    /// 构建空索引
    pub fn build(self) -> HNSWIndex {
        HNSWIndex::new(
            self.dimension,
            self.m,
            self.ef_construction,
            self.distance_type,
            self.ml,
            self.max_level_limit,
            self.index_id,
        )
    }
}

/// HNSW索引实现
#[derive(Debug)]
pub struct HNSWIndex {
    /// 向量维度
    dimension: usize,
    /// 每层最大连接数
    m: usize,
    /// 实际最大每层连接数 (2*m)
    m_max: usize,
    /// 构建时扩展因子
    ef_construction: usize,
    /// 距离计算类型
    distance_type: DistanceType,
    /// 层级分布参数
    ml: f32,
    /// 最大层数限制
    max_level_limit: usize,
    /// 当前最大层级
    max_level: usize,
    /// 索引ID
    index_id: String,
    /// 入口点节点索引
    entry_point: Option<NodeIndex>,
    /// 所有节点存储
    nodes: Vec<SharedNode>,
    /// 向量ID到节点索引的映射
    id_to_index: HashMap<VectorId, NodeIndex>,
    /// 元数据存储
    metadata: HashMap<VectorId, HashMap<String, String>>,
    /// 已删除节点数量
    deleted_count: usize,
    /// 随机数生成器
    rng: StdRng,
}

impl HNSWIndex {
    /// 创建新的HNSW索引
    pub fn new(
        dimension: usize,
        m: usize,
        ef_construction: usize,
        distance_type: DistanceType,
        ml: f32,
        max_level_limit: usize,
        index_id: String,
    ) -> Self {
        let m_max = 2 * m; // 实际最大连接数为配置值的两倍

        Self {
            dimension,
            m,
            m_max,
            ef_construction,
            distance_type,
            ml,
            max_level_limit,
            max_level: 0,
            index_id,
            entry_point: None,
            nodes: Vec::new(),
            id_to_index: HashMap::new(),
            metadata: HashMap::new(),
            deleted_count: 0,
            rng: StdRng::from_entropy(),
        }
    }

    /// 获取向量维度
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// 获取距离计算类型
    pub fn distance_type(&self) -> DistanceType {
        self.distance_type
    }

    /// 获取当前最大层级
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// 获取索引ID
    pub fn index_id(&self) -> &str {
        &self.index_id
    }

    /// 获取每层最大连接数
    pub fn m(&self) -> usize {
        self.m
    }

    /// 获取实际最大每层连接数
    pub fn m_max(&self) -> usize {
        self.m_max
    }

    /// 获取构建时扩展因子
    pub fn ef_construction(&self) -> usize {
        self.ef_construction
    }

    /// 获取最大层数限制
    pub fn max_level_limit(&self) -> usize {
        self.max_level_limit
    }

    /// 获取入口点节点索引
    pub fn entry_point(&self) -> Option<NodeIndex> {
        self.entry_point
    }

    /// 获取所有节点存储
    pub fn nodes(&self) -> &Vec<SharedNode> {
        &self.nodes
    }

    /// 设置当前最大层级（用于反序列化）
    pub(crate) fn set_max_level(&mut self, max_level: usize) {
        self.max_level = max_level;
    }

    /// 设置入口点（用于反序列化）
    pub(crate) fn set_entry_point(&mut self, entry_point: Option<NodeIndex>) {
        self.entry_point = entry_point;
    }

    /// 设置ID到索引的映射（用于反序列化）
    pub(crate) fn set_id_to_index(&mut self, id_to_index: HashMap<VectorId, NodeIndex>) {
        self.id_to_index = id_to_index;
    }

    /// 添加节点（用于反序列化）
    pub(crate) fn add_node(&mut self, node: SharedNode) {
        self.nodes.push(node);
    }

    /// 获取索引大小（向量数量）
    pub fn size(&self) -> usize {
        self.nodes.len() - self.deleted_count
    }

    /// 检查索引是否为空
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// 获取入口点
    pub fn get_entry_point(&self) -> NodeIndex {
        self.entry_point.unwrap_or(0)
    }

    /// 随机生成节点层级
    fn random_level(&mut self) -> usize {
        let mut level = 0;
        let mut rnd: f32 = self.rng.gen();
        
        // 随机确定层级，最大不超过max_level + 1和max_level_limit
        while rnd < self.ml && level < self.max_level + 1 && level < self.max_level_limit {
            level += 1;
            rnd = self.rng.gen();
        }
        
        level
    }

    /// 添加新向量到索引
    pub fn add(&mut self, id: VectorId, vector: Vector, metadata: Option<HashMap<String, String>>) -> Result<(), String> {
        // 检查维度是否匹配
        if vector.len() != self.dimension {
            return Err(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension, vector.len()
            ));
        }

        // 检查ID是否已存在
        if self.id_to_index.contains_key(&id) {
            return Err(format!("Vector with ID {} already exists", id));
        }

        // 随机选择层级
        let level = self.random_level();
        
        // 更新全局最大层级
        if level > self.max_level {
            self.max_level = level;
        }

        // 创建新节点
        let node_index = self.nodes.len();
        // 将 VectorId (u64) 转换为 Uuid
        // 使用 from_u128 将 u64 转换为 Uuid（将 u64 作为低64位，高64位为0）
        let uuid_id = Uuid::from_u128(id as u128);
        let node = HNSWNode::new(node_index, uuid_id, vector, level);
        let shared_node = Arc::new(RwLock::new(node));
        self.nodes.push(shared_node.clone());
        self.id_to_index.insert(id, node_index);

        // 存储元数据
        if let Some(meta) = metadata {
            self.metadata.insert(id, meta);
        }

        // 如果是第一个节点，设为入口点并返回
        if self.nodes.len() == 1 {
            self.entry_point = Some(node_index);
            log_debug!("Added first vector with ID {} as entry point", id);
            return Ok(());
        }

        // 处理其他节点的插入
        self.insert_node(node_index, shared_node)?;
        
        log_debug!("Added vector with ID {} at level {}", id, level);
        Ok(())
    }

    /// 将节点插入到索引中
    fn insert_node(&mut self, node_index: NodeIndex, node: SharedNode) -> Result<(), String> {
        // 获取入口点
        let mut entry_point = self.entry_point.unwrap();
        let mut entry_point_node = self.get_node_by_index(entry_point).map_err(|e| e.to_string())?;
        
        let node_read = node.read();
        let node_level = node_read.level;
        let node_vector = node_read.vector.clone();
        drop(node_read); // 释放锁
        
        // 计算与入口点的距离
        let entry_point_distance = self.calculate_distance(entry_point, &node_vector);
        
        // 从最高层向下搜索最近邻
        let mut current_level = self.max_level;
        let mut current_nearest = entry_point;
        let mut current_distance = entry_point_distance;
        
        // 对每一层，找到最近的节点
        while current_level > node_level {
            let changed = self.search_nearest_neighbor(
                current_level,
                &node_vector,
                &mut current_nearest,
                &mut current_distance,
            )?;
            
            if !changed {
                // 如果未找到更近的节点，下降一层
                current_level -= 1;
            }
        }
        
        // 对节点所在的每一层，执行插入
        for level in (0..=node_level).rev() {
            // 在当前层找到ef_construction个最近邻
            let neighbors = self.search_nearest_neighbors(
                level,
                &node_vector,
                self.ef_construction,
                current_nearest,
                current_distance,
            )?;
            
            // 在当前层连接节点
            self.connect_new_node(node_index, neighbors, level)?;
            
            // 更新当前最近邻为下一层搜索起点
            if !neighbors.is_empty() {
                current_nearest = neighbors[0].0;
                current_distance = neighbors[0].1;
            }
        }
        
        // 更新入口点
        if node_level > self.get_node_level(entry_point)? {
            self.entry_point = Some(node_index);
        }
        
        Ok(())
    }

    /// 在指定层搜索最近邻节点
    fn search_nearest_neighbor(
        &self,
        level: usize,
        query: &Vector,
        current_nearest: &mut NodeIndex,
        current_distance: &mut Distance,
    ) -> Result<bool, String> {
        let mut changed = false;
        let mut visited = std::collections::HashSet::new();
        
        visited.insert(*current_nearest);
        
        // 贪婪搜索
        loop {
            let neighbors = self.get_node_connections(*current_nearest, level)?;
            let mut found_closer = false;
            
            for &neighbor_idx in &neighbors {
                if visited.contains(&neighbor_idx) {
                    continue;
                }
                
                visited.insert(neighbor_idx);
                
                let distance = self.calculate_distance(neighbor_idx, query);
                
                if distance < *current_distance {
                    *current_nearest = neighbor_idx;
                    *current_distance = distance;
                    found_closer = true;
                    changed = true;
                }
            }
            
            if !found_closer {
                break;
            }
        }
        
        Ok(changed)
    }

    /// 搜索指定数量的最近邻节点
    fn search_nearest_neighbors(
        &self,
        level: usize,
        query: &Vector,
        ef: usize,
        start_node: NodeIndex,
        start_distance: Distance,
    ) -> Result<Vec<(NodeIndex, Distance)>, String> {
        // 如果目标ef为0，返回空结果
        if ef == 0 {
            return Ok(Vec::new());
        }
        
        // 初始化候选集和结果集
        // 使用 Vec 和手动排序，因为 f32 不实现 Ord
        let mut candidates: Vec<(f32, usize)> = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut results: Vec<(f32, usize)> = Vec::new();
        
        // 添加起始节点
        candidates.push((start_distance, start_node));
        visited.insert(start_node);
        results.push((start_distance, start_node));
        
        // 对候选集进行排序（按距离升序）
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        // 广度优先搜索
        while !candidates.is_empty() {
            // 取出距离最近的候选（Vec 已按距离升序排序）
            let current = candidates.remove(0);
            
            // 如果当前候选比结果集最远元素更远，结束搜索
            if let Some(farthest) = results.last() {
                if current.0 > farthest.0 {
                    break;
                }
            }
            
            // 获取当前节点的邻居
            let neighbors = self.get_node_connections(current.1, level)?;
            
            for &neighbor_idx in &neighbors {
                if visited.contains(&neighbor_idx) {
                    continue;
                }
                
                visited.insert(neighbor_idx);
                
                let distance = self.calculate_distance(neighbor_idx, query);
                
                // 如果结果集未满或邻居比最远结果更近，添加到结果
                if results.len() < ef || distance < results.last().unwrap().0 {
                    candidates.push((distance, neighbor_idx));
                    results.push((distance, neighbor_idx));
                    
                    // 维持结果集大小（按距离升序排序，移除最远的）
                    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                    if results.len() > ef {
                        results.pop();
                    }
                }
                
                // 重新排序候选集（按距离升序）
                candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
        
        // 转换结果为向量（结果集已按距离升序排序）
        let result_vec: Vec<_> = results
            .into_iter()
            .map(|pair| (pair.1, pair.0))
            .collect();
            
        Ok(result_vec)
    }

    /// 连接新节点到其最近邻
    fn connect_new_node(
        &mut self,
        node_index: NodeIndex,
        neighbors: Vec<(NodeIndex, Distance)>,
        level: usize,
    ) -> Result<(), String> {
        // 获取节点的写锁
        let node = self.get_node_by_index(node_index).map_err(|e| e.to_string())?;
        let mut node_write = node.write();
        
        // 最多选择m个最近邻
        let m = std::cmp::min(self.m, neighbors.len());
        
        // 添加连接到新节点
        for i in 0..m {
            let (neighbor_idx, distance) = neighbors[i];
            
            // 添加连接（使用元组 (NodeIndex, Distance)）
            if level < node_write.connections.len() {
                node_write.connections[level].push((neighbor_idx, distance));
            } else {
                // 如果层级不足，添加新层
                let mut new_layer = Vec::new();
                new_layer.push((neighbor_idx, distance));
                node_write.connections.push(new_layer);
            }
        }
        
        drop(node_write); // 释放锁
        
        // 更新已选邻居的连接
        for i in 0..m {
            let (neighbor_idx, distance) = neighbors[i];
            
            // 添加相反方向的连接
            self.add_connection(neighbor_idx, node_index, distance, level)?;
        }
        
        Ok(())
    }

    /// 添加连接并根据需要裁剪连接
    fn add_connection(
        &mut self,
        from_idx: NodeIndex,
        to_idx: NodeIndex,
        distance: Distance,
        level: usize,
    ) -> Result<(), String> {
        // 获取起始节点的写锁
        let from_node = self.get_node_by_index(from_idx).map_err(|e| e.to_string())?;
        let mut from_write = from_node.write();
        
        // 确保有足够的层
        while from_write.connections.len() <= level {
            from_write.connections.push(Vec::new());
        }
        
        // 检查是否已有连接
        let already_connected = from_write.connections[level]
            .iter()
            .any(|c| c.0 == to_idx);
            
        if already_connected {
            return Ok(());
        }
        
        // 添加新连接（使用元组 (NodeIndex, Distance)）
        from_write.connections[level].push((to_idx, distance));
        
        // 如果连接数超过限制，进行裁剪
        if from_write.connections[level].len() > self.m_max {
            self.prune_connections(&mut from_write, level)?;
        }
        
        Ok(())
    }

    /// 裁剪连接，保留最好的m个连接
    fn prune_connections(&self, node: &mut HNSWNode, level: usize) -> Result<(), String> {
        if level >= node.connections.len() {
            return Ok(());
        }
        
        let connections = &mut node.connections[level];
        if connections.len() <= self.m {
            return Ok(());
        }
        
        // 排序连接（按距离升序，元组的第二个元素是距离）
        connections.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // 保留最近的m个连接
        connections.truncate(self.m);
        
        Ok(())
    }

    /// 计算节点与查询向量的距离
    fn calculate_distance(&self, node_idx: NodeIndex, query: &Vector) -> Distance {
        if let Ok(node) = self.get_node_by_index(node_idx) {
            let node_read = node.read();
            return distance::calculate_distance(&node_read.vector, query, self.distance_type);
        }
        f32::MAX
    }

    /// 获取节点的层级
    fn get_node_level(&self, node_idx: NodeIndex) -> Result<usize, String> {
        let node = self.get_node_by_index(node_idx).map_err(|e| e.to_string())?;
        let node_read = node.read();
        Ok(node_read.level)
    }

    /// 获取节点的连接
    fn get_node_connections(&self, node_idx: NodeIndex, level: usize) -> Result<Vec<NodeIndex>, String> {
        let node = self.get_node_by_index(node_idx).map_err(|e| e.to_string())?;
        let node_read = node.read();
        
        if level >= node_read.connections.len() {
            return Ok(Vec::new());
        }
        
        let indices: Vec<_> = node_read.connections[level]
            .iter()
            .map(|conn| conn.0)  // 元组的第一个元素是 NodeIndex
            .collect();
            
        Ok(indices)
    }

    /// 通过索引获取节点
    pub fn get_node_by_index(&self, node_idx: NodeIndex) -> Result<SharedNode, String> {
        if node_idx >= self.nodes.len() {
            return Err(format!("Node index {} out of bounds", node_idx));
        }
        Ok(self.nodes[node_idx].clone())
    }

    /// 通过ID获取节点
    pub fn get_node_by_id(&self, id: VectorId) -> Result<SharedNode, String> {
        if let Some(&node_idx) = self.id_to_index.get(&id) {
            return self.get_node_by_index(node_idx);
        }
        Err(format!("Vector with ID {} not found", id))
    }

    /// 获取向量ID对应的元数据
    pub fn get_metadata(&self, id: VectorId) -> Option<&HashMap<String, String>> {
        self.metadata.get(&id)
    }

    /// 删除向量
    pub fn delete(&mut self, id: VectorId) -> Result<(), String> {
        if let Some(&node_idx) = self.id_to_index.get(&id) {
            let node = self.get_node_by_index(node_idx).map_err(|e| e.to_string())?;
            let mut node_write = node.write();
            
            // 标记为已删除
            node_write.marked_deleted = true;
            drop(node_write);
            
            // 移除ID映射和元数据
            self.id_to_index.remove(&id);
            self.metadata.remove(&id);
            self.deleted_count += 1;
            
            // 如果是入口点，需要更新
            if self.entry_point == Some(node_idx) {
                self.update_entry_point()?;
            }
            
            log_debug!("Deleted vector with ID {}", id);
            return Ok(());
        }
        
        Err(format!("Vector with ID {} not found", id))
    }

    /// 更新入口点（当当前入口点被删除时）
    fn update_entry_point(&mut self) -> Result<(), String> {
        // 如果所有节点都被删除，清空入口点
        if self.size() == 0 {
            self.entry_point = None;
            return Ok(());
        }
        
        // 找到最高层级的非删除节点
        let mut max_level = 0;
        let mut new_entry_point = None;
        
        for (idx, node) in self.nodes.iter().enumerate() {
            let node_read = node.read();
            if !node_read.marked_deleted && node_read.level >= max_level {
                max_level = node_read.level;
                new_entry_point = Some(idx);
            }
        }
        
        if let Some(ep) = new_entry_point {
            self.entry_point = Some(ep);
            self.max_level = max_level;
            log_debug!("Updated entry point to node index {}", ep);
            return Ok(());
        }
        
        Err("Failed to find new entry point".to_string())
    }

    /// 获取索引统计信息
    pub fn get_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        
        stats.insert("index_id".to_string(), self.index_id.clone());
        stats.insert("dimension".to_string(), self.dimension.to_string());
        stats.insert("total_vectors".to_string(), (self.nodes.len() - self.deleted_count).to_string());
        stats.insert("max_level".to_string(), self.max_level.to_string());
        stats.insert("m".to_string(), self.m.to_string());
        stats.insert("ef_construction".to_string(), self.ef_construction.to_string());
        
        // 计算内存占用估计
        let mut total_memory = 0;
        let mut total_connections = 0;
        let mut min_connections = usize::MAX;
        let mut max_connections = 0;
        
        for node in &self.nodes {
            let node_read = node.read();
            if !node_read.marked_deleted {
                let node_memory = node_read.memory_usage();
                total_memory += node_memory;
                
                let node_connections: usize = node_read.connections.iter().map(|layer| layer.len()).sum();
                total_connections += node_connections;
                
                min_connections = min_connections.min(node_connections);
                max_connections = max_connections.max(node_connections);
            }
        }
        
        let active_nodes = self.nodes.len() - self.deleted_count;
        let avg_connections = if active_nodes > 0 {
            total_connections as f32 / active_nodes as f32
        } else {
            0.0
        };
        
        stats.insert("total_memory_bytes".to_string(), total_memory.to_string());
        stats.insert("min_connections".to_string(), min_connections.to_string());
        stats.insert("max_connections".to_string(), max_connections.to_string());
        stats.insert("avg_connections".to_string(), format!("{:.2}", avg_connections));
        
        stats
    }
}

// 用于优先队列的有序对（按距离升序）
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedPair(Distance, NodeIndex);

impl Eq for OrderedPair {}

impl PartialOrd for OrderedPair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.0.partial_cmp(&self.0)
    }
}

impl Ord for OrderedPair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// 用于优先队列的有序对（按距离降序）
#[derive(Debug, Clone, Copy, PartialEq)]
struct ReverseOrderedPair(Distance, NodeIndex);

impl Eq for ReverseOrderedPair {}

impl PartialOrd for ReverseOrderedPair {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for ReverseOrderedPair {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl Clone for HNSWIndex {
    fn clone(&self) -> Self {
        // 创建一个新的随机数生成器，因为 StdRng 不能直接克隆
        let mut rng = StdRng::from_entropy();
        Self {
            dimension: self.dimension,
            m: self.m,
            m_max: self.m_max,
            ef_construction: self.ef_construction,
            distance_type: self.distance_type,
            ml: self.ml,
            max_level_limit: self.max_level_limit,
            max_level: self.max_level,
            index_id: self.index_id.clone(),
            entry_point: self.entry_point,
            nodes: self.nodes.clone(),
            id_to_index: self.id_to_index.clone(),
            metadata: self.metadata.clone(),
            deleted_count: self.deleted_count,
            rng,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder() {
        let builder = HNSWBuilder::new(10)
            .with_m(20)
            .with_ef_construction(100)
            .with_distance_type(DistanceType::Cosine)
            .with_ml(1.5)
            .with_max_level_limit(12)
            .with_index_id("test_index".to_string());
            
        let index = builder.build();
        
        assert_eq!(index.dimension(), 10);
        assert_eq!(index.distance_type(), DistanceType::Cosine);
        assert_eq!(index.index_id(), "test_index");
        assert_eq!(index.max_level(), 0);
        assert!(index.is_empty());
    }
    
    #[test]
    fn test_add_vector() {
        let mut index = HNSWBuilder::new(3).build();
        
        assert!(index.add(1, vec![1.0, 2.0, 3.0], None).is_ok());
        assert!(index.add(2, vec![4.0, 5.0, 6.0], None).is_ok());
        
        assert_eq!(index.size(), 2);
        assert!(!index.is_empty());
        
        // 测试添加错误维度向量
        assert!(index.add(3, vec![1.0, 2.0], None).is_err());
        
        // 测试添加重复ID
        assert!(index.add(1, vec![7.0, 8.0, 9.0], None).is_err());
    }
    
    #[test]
    fn test_get_node() {
        let mut index = HNSWBuilder::new(3).build();
        
        index.add(1, vec![1.0, 2.0, 3.0], None).unwrap();
        
        // 通过索引获取
        let node = index.get_node_by_index(0).unwrap();
        let node_read = node.read().unwrap();
        assert_eq!(node_read.id, 1);
        assert_eq!(node_read.vector, vec![1.0, 2.0, 3.0]);
        
        // 通过ID获取
        let node = index.get_node_by_id(1).unwrap();
        let node_read = node.read().unwrap();
        assert_eq!(node_read.id, 1);
        
        // 测试不存在的ID
        assert!(index.get_node_by_id(99).is_err());
    }
    
    #[test]
    fn test_delete_vector() {
        let mut index = HNSWBuilder::new(3).build();
        
        index.add(1, vec![1.0, 2.0, 3.0], None).unwrap();
        index.add(2, vec![4.0, 5.0, 6.0], None).unwrap();
        
        assert_eq!(index.size(), 2);
        
        // 测试删除
        assert!(index.delete(1).is_ok());
        assert_eq!(index.size(), 1);
        assert!(index.get_node_by_id(1).is_err());
        
        // 测试删除不存在的ID
        assert!(index.delete(99).is_err());
    }
    
    #[test]
    fn test_with_metadata() {
        let mut index = HNSWBuilder::new(3).build();
        
        let mut metadata = HashMap::new();
        metadata.insert("key1".to_string(), "value1".to_string());
        metadata.insert("key2".to_string(), "value2".to_string());
        
        index.add(1, vec![1.0, 2.0, 3.0], Some(metadata)).unwrap();
        
        let retrieved_metadata = index.get_metadata(1).unwrap();
        assert_eq!(retrieved_metadata.get("key1").unwrap(), "value1");
        assert_eq!(retrieved_metadata.get("key2").unwrap(), "value2");
    }
}
