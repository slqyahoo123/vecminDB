//! HNSW索引的搜索算法实现
//!
//! 本模块实现了HNSW索引的k最近邻(KNN)搜索算法、范围搜索和混合搜索等核心功能。
//! 搜索过程利用HNSW图的多层结构特性，通过从顶层开始依次缩小搜索范围来实现高效查找。

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, RwLock};
use tracing::{debug, trace};

use crate::vector::index::hnsw::distance;
use crate::vector::index::hnsw::node::SharedNode;
use crate::vector::index::hnsw::HNSWNode;
use crate::vector::index::hnsw::types::{
    Distance, DistanceType, NodeIndex, SearchConfig, 
    SearchStrategy, VectorId, FilterFunction, SearchResults, SearchResultItem
};

/// 搜索结果条目
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// 向量ID
    pub id: VectorId,
    /// 到查询向量的距离
    pub distance: Distance,
    /// 可选的向量数据
    pub vector: Option<Vec<f32>>,
    /// 可选的元数据
    pub metadata: Option<HashMap<String, String>>,
}

/// 并发安全的HNSW搜索器
pub struct ConcurrentHNSWSearcher {
    /// 搜索配置
    config: SearchConfig,
    /// 并发控制
    semaphore: Arc<tokio::sync::Semaphore>,
    /// 搜索统计
    stats: Arc<RwLock<SearchStats>>,
}

/// 搜索统计信息
#[derive(Debug, Clone)]
pub struct SearchStats {
    /// 总搜索次数
    pub total_searches: usize,
    /// 平均搜索时间
    pub average_search_time: f64,
    /// 搜索成功率
    pub success_rate: f64,
    /// 并发搜索次数
    pub concurrent_searches: usize,
}

impl ConcurrentHNSWSearcher {
    /// 创建新的并发安全搜索器
    pub fn new(config: SearchConfig, max_concurrent: usize) -> Self {
        Self {
            config,
            semaphore: Arc::new(tokio::sync::Semaphore::new(max_concurrent)),
            stats: Arc::new(RwLock::new(SearchStats {
                total_searches: 0,
                average_search_time: 0.0,
                success_rate: 0.0,
                concurrent_searches: 0,
            })),
        }
    }
    
    /// 并发安全的搜索方法
    pub async fn search_concurrent(
        &self,
        query_vector: &[f32],
        k: usize,
        filter: Option<FilterFunction>,
    ) -> Result<SearchResults, Box<dyn std::error::Error>> {
        let _permit = self.semaphore.acquire().await?;
        
        let start_time = std::time::Instant::now();
        
        // 执行搜索
        let results = self.perform_search(query_vector, k, filter.clone()).await?;
        
        let search_time = start_time.elapsed().as_millis() as f64;
        
        // 更新统计信息
        self.update_stats(search_time, !results.is_empty()).await;
        
        Ok(results)
    }
    
    /// 执行实际搜索
    async fn perform_search(
        &self,
        query_vector: &[f32],
        k: usize,
        filter: Option<FilterFunction>,
    ) -> Result<SearchResults, Box<dyn std::error::Error>> {
        // 这里实现具体的搜索逻辑
        // 由于这是示例代码，我们返回一个模拟结果
        let mut results = Vec::new();
        
        for i in 0..k {
            let item_id = i as u64;
            
            // 应用过滤器（如果提供）
            if let Some(ref filter_fn) = filter {
                if !filter_fn(&item_id, None) {
                    continue; // 跳过不满足过滤条件的项
                }
            }
            
            results.push(SearchResultItem {
                id: item_id,
                distance: i as f32,
                vector: Some(query_vector.to_vec()),
                metadata: Some(HashMap::new()),
            });
        }
        
        Ok(results)
    }
    
    /// 更新搜索统计
    async fn update_stats(&self, search_time: f64, success: bool) {
        let mut stats = self.stats.write().unwrap();
        stats.total_searches += 1;
        
        // 更新平均搜索时间
        let total_time = stats.average_search_time * (stats.total_searches - 1) as f64 + search_time;
        stats.average_search_time = total_time / stats.total_searches as f64;
        
        // 更新成功率
        if success {
            let success_count = (stats.success_rate * (stats.total_searches - 1) as f64) as usize + 1;
            stats.success_rate = success_count as f64 / stats.total_searches as f64;
        } else {
            let success_count = (stats.success_rate * (stats.total_searches - 1) as f64) as usize;
            stats.success_rate = success_count as f64 / stats.total_searches as f64;
        }
    }
    
    /// 获取搜索统计
    pub async fn get_stats(&self) -> SearchStats {
        self.stats.read().unwrap().clone()
    }
    
    /// 重置搜索统计
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.write().unwrap();
        *stats = SearchStats {
            total_searches: 0,
            average_search_time: 0.0,
            success_rate: 0.0,
            concurrent_searches: 0,
        };
    }
}

/// 并发安全的HNSW节点操作
pub struct ConcurrentHNSWNode {
    /// 节点数据
    node: Arc<RwLock<HNSWNode>>,
    /// 访问控制
    access_control: Arc<tokio::sync::RwLock<AccessControl>>,
}

/// 访问控制
#[derive(Debug)]
pub struct AccessControl {
    /// 读锁数量
    read_count: usize,
    /// 写锁状态
    write_locked: bool,
    /// 等待队列
    wait_queue: Vec<tokio::sync::oneshot::Sender<()>>,
}

impl ConcurrentHNSWNode {
    /// 创建新的并发安全节点
    pub fn new(node: HNSWNode) -> Self {
        Self {
            node: Arc::new(RwLock::new(node)),
            access_control: Arc::new(tokio::sync::RwLock::new(AccessControl {
                read_count: 0,
                write_locked: false,
                wait_queue: Vec::new(),
            })),
        }
    }
    
    /// 并发安全的读取操作
    pub async fn read<F, R>(&self, f: F) -> Result<R, Box<dyn std::error::Error>>
    where
        F: FnOnce(&HNSWNode) -> R,
    {
        let node = self.node.read().unwrap();
        Ok(f(&node))
    }
    
    /// 并发安全的写入操作
    pub async fn write<F, R>(&self, f: F) -> Result<R, Box<dyn std::error::Error>>
    where
        F: FnOnce(&mut HNSWNode) -> R,
    {
        let mut node = self.node.write().unwrap();
        Ok(f(&mut node))
    }
    
    /// 获取节点引用
    pub fn get_node(&self) -> Arc<RwLock<HNSWNode>> {
        self.node.clone()
    }
}

/// 搜索结果条目的内部表示，用于优先队列
#[derive(Clone, Debug, PartialEq)]
struct SearchCandidate {
    /// 节点索引
    pub node_idx: NodeIndex,
    /// 到查询向量的距离
    pub distance: Distance,
}

impl Eq for SearchCandidate {}

/// 优先队列排序实现 - 最小堆（距离越小优先级越高）
impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // 反向比较距离，将最小堆变成最大堆
        other.distance.partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// 优先队列排序实现 - 最大堆（距离越大优先级越高）
#[derive(Clone, Debug, PartialEq, Eq)]
struct FurthestFirst(SearchCandidate);

impl Ord for FurthestFirst {
    fn cmp(&self, other: &Self) -> Ordering {
        // 直接比较距离，越大优先级越高
        self.0.distance.partial_cmp(&other.0.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for FurthestFirst {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// 用于优先队列的候选节点，按距离排序
#[derive(Debug, Clone)]
struct Candidate {
    distance: Distance,
    node_index: NodeIndex,
}

impl Eq for Candidate {}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.node_index == other.node_index
    }
}

/// 让候选者按照距离排序，对于最近优先队列：
/// 距离越小，优先级越高（最小堆）
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // 使用反向比较，越小的距离优先级越高
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// 对于最远优先队列：距离越大，优先级越高（最大堆）
#[derive(Debug, Clone)]
struct FurthestFirstCandidate(Candidate);

impl Eq for FurthestFirstCandidate {}

impl PartialEq for FurthestFirstCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Ord for FurthestFirstCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // 直接比较，越大的距离优先级越高
        self.0.distance.partial_cmp(&other.0.distance).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for FurthestFirstCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// 实现HNSW索引的搜索功能
pub struct HNSWSearch<'a> {
    nodes: &'a [SharedNode],
    entry_point: Option<NodeIndex>,
    distance_type: DistanceType,
}

impl<'a> HNSWSearch<'a> {
    /// 创建新的搜索实例
    pub fn new(
        nodes: &'a [SharedNode],
        entry_point: Option<NodeIndex>,
        distance_type: DistanceType,
    ) -> Self {
        Self {
            nodes,
            entry_point,
            distance_type,
        }
    }

    /// 使用指定配置搜索向量最近邻
    pub fn search(&self, query: &[f32], config: SearchConfig) -> Vec<SearchResult> {
        if self.nodes.is_empty() || self.entry_point.is_none() {
            return Vec::new();
        }

        // 从 SearchStrategy 中提取 ef_search 参数
        let ef = match &config.strategy {
            SearchStrategy::Standard { ef_search } => *ef_search,
            SearchStrategy::Range { .. } => 50, // 范围搜索使用默认值
            SearchStrategy::Hybrid { ef_search, .. } => *ef_search,
        };
        
        let nearest_neighbors = match &config.strategy {
            SearchStrategy::Standard { .. } => self.search_standard(query, ef),
            SearchStrategy::Range { radius, max_elements } => {
                // 范围搜索需要特殊处理，这里先使用标准搜索
                self.search_standard(query, ef)
            },
            SearchStrategy::Hybrid { .. } => self.search_standard(query, ef),
        };

        // 应用过滤器和限制
        let filter_fn = match &config.strategy {
            SearchStrategy::Hybrid { filter, .. } => filter.as_ref(),
            _ => None,
        };
        
        let mut results: Vec<SearchResult> = nearest_neighbors
            .into_iter()
            .filter(|candidate| {
                let node = self.nodes[candidate.node_index].read();
                if node.is_deleted() {
                    return false;
                }
                
                // 如果有过滤器，应用过滤器
                if let Some(filter) = filter_fn {
                    filter(&(node.id.as_u128() as u64), None)
                } else {
                    true
                }
            })
            .map(|candidate| {
                let node = self.nodes[candidate.node_index].read();
                let mut result = SearchResult {
                    id: node.id.as_u128() as u64,  // 将 Uuid 转换为 u64
                    distance: candidate.distance,
                    vector: None,
                    metadata: None,
                };

                // 根据配置添加向量和元数据
                if config.include_vectors {
                    result.vector = Some(node.vector.clone());
                }
                // HNSWNode 没有 metadata 字段，如果需要可以从外部存储获取
                // if config.include_metadata {
                //     result.metadata = node.metadata.clone();
                // }

                result
            })
            .collect();

        // 限制结果数量
        if results.len() > config.limit {
            results.truncate(config.limit);
        }

        results
    }

    /// 标准HNSW搜索算法
    fn search_standard(&self, query: &[f32], ef: usize) -> BinaryHeap<Candidate> {
        let entry_point = self.entry_point.unwrap();
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        // 计算入口点的距离
        let entry_node = self.nodes[entry_point].read();
        let entry_dist = self.calculate_distance(query, &entry_node.vector);
        
        // 初始化候选和结果集
        candidates.push(Candidate {
            distance: entry_dist,
            node_index: entry_point,
        });
        
        results.push(FurthestFirstCandidate(Candidate {
            distance: entry_dist,
            node_index: entry_point,
        }));
        
        visited.insert(entry_point);

        // 主循环：检查所有候选点，更新结果集
        while let Some(current) = candidates.pop() {
            // 检查当前最远结果的距离
            if let Some(furthest) = results.peek() {
                if current.distance > furthest.0.distance && results.len() >= ef {
                    // 如果当前点比结果集中最远的点还远，且结果集已满，跳过
                    continue;
                }
            }

            // 获取当前节点
            let current_node = self.nodes[current.node_index].read();

            // 检查底层的所有邻居
            if let Some(connections) = current_node.get_connections(0) {
                for connection in connections {
                    let neighbor_index = connection.0;  // 元组的第一个元素是 node_index
                    
                    // 避免重复访问
                    if visited.contains(&neighbor_index) {
                        continue;
                    }
                    visited.insert(neighbor_index);

                    // 读取邻居节点
                    let neighbor_node = self.nodes[neighbor_index].read();

                    // 计算距离
                    let distance = self.calculate_distance(query, &neighbor_node.vector);
                    
                    // 检查是否需要添加到结果集
                    if results.len() < ef || distance < results.peek().unwrap().0.distance {
                        candidates.push(Candidate {
                            distance,
                            node_index: neighbor_index,
                        });
                        
                        results.push(FurthestFirstCandidate(Candidate {
                            distance,
                            node_index: neighbor_index,
                        }));
                        
                        // 保持结果集大小为ef
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // 转换结果格式
        let mut nearest_neighbors = BinaryHeap::new();
        while let Some(furthest) = results.pop() {
            nearest_neighbors.push(furthest.0);
        }
        
        nearest_neighbors
    }

    /// 启发式搜索算法（使用贪婪算法优化）
    fn search_heuristic(&self, query: &[f32], ef: usize) -> BinaryHeap<Candidate> {
        // 启发式搜索基于标准搜索，但添加了贪婪优化
        // 这里实现类似标准搜索，但在选择邻居时更加贪婪，优先选择更好的候选项
        
        let entry_point = self.entry_point.unwrap();
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        // 计算入口点的距离
        let entry_node = self.nodes[entry_point].read();
        let entry_dist = self.calculate_distance(query, &entry_node.vector);
        
        // 初始化候选和结果集
        candidates.push(Candidate {
            distance: entry_dist,
            node_index: entry_point,
        });
        
        results.push(FurthestFirstCandidate(Candidate {
            distance: entry_dist,
            node_index: entry_point,
        }));
        
        visited.insert(entry_point);

        // 主循环：检查所有候选点，更新结果集
        while let Some(current) = candidates.pop() {
            // 贪婪搜索策略：如果当前候选的距离大于结果集中最好的距离，跳过
            if !results.is_empty() && current.distance > results.peek().unwrap().0.distance {
                continue;
            }

            // 获取当前节点
            let current_node = self.nodes[current.node_index].read();

            // 检查底层的所有邻居
            if let Some(connections) = current_node.get_connections(0) {
                for connection in connections {
                    let neighbor_index = connection.0;  // 元组的第一个元素是 node_index
                    
                    // 避免重复访问
                    if visited.contains(&neighbor_index) {
                        continue;
                    }
                    visited.insert(neighbor_index);

                    // 读取邻居节点
                    let neighbor_node = self.nodes[neighbor_index].read();

                    // 计算距离
                    let distance = self.calculate_distance(query, &neighbor_node.vector);
                    
                    // 启发式策略：更早地跳过不太可能的候选者
                    if results.len() >= ef && distance > results.peek().unwrap().0.distance {
                        continue;
                    }
                    
                    candidates.push(Candidate {
                        distance,
                        node_index: neighbor_index,
                    });
                    
                    results.push(FurthestFirstCandidate(Candidate {
                        distance,
                        node_index: neighbor_index,
                    }));
                    
                    // 保持结果集大小为ef
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        // 转换结果格式
        let mut nearest_neighbors = BinaryHeap::new();
        while let Some(furthest) = results.pop() {
            nearest_neighbors.push(furthest.0);
        }
        
        nearest_neighbors
    }

    /// 计算两个向量之间的距离
    fn calculate_distance(&self, vec1: &[f32], vec2: &[f32]) -> Distance {
        distance::calculate_distance(vec1, vec2, self.distance_type)
    }
}

/// 执行K最近邻(KNN)搜索
pub fn search_knn(
    query: &[f32],
    k: usize,
    ef_search: usize,
    entry_point: NodeIndex,
    max_level: usize,
    nodes: &HashMap<NodeIndex, SharedNode>,
    distance_type: DistanceType,
    filter: Option<&FilterFunction>,
) -> SearchResults {
    if nodes.is_empty() || k == 0 {
        return Vec::new();
    }

    // 有效的搜索范围，至少为k
    let ef = ef_search.max(k);
    
    debug!("Starting KNN search: k={}, ef={}, max_level={}", k, ef, max_level);
    
    // 从最高层开始，找到入口点
    let mut current_level = max_level;
    let mut ep = entry_point;
    
    // 在每一层找到最近的节点，用作下一层的入口点
    while current_level > 0 {
        trace!("Searching layer {}", current_level);
        ep = search_layer(
            query,
            ep,
            1,
            current_level,
            nodes,
            distance_type,
            None
        ).get(0).map(|c| c.node_idx).unwrap_or(ep);
        
        current_level -= 1;
    }
    
    // 在底层进行完整搜索
    trace!("Performing full search on base layer");
    let candidates = search_layer(
        query,
        ep,
        ef,
        0,
        nodes,
        distance_type,
        filter
    );
    
    // 将结果转换为SearchResultItem格式
    candidates.into_iter()
        .take(k)
        .map(|c| {
            let node_lock = nodes.get(&c.node_idx).unwrap().read();
            SearchResultItem {
                id: node_lock.id.as_u128() as u64,  // 将 Uuid 转换为 u64
                distance: c.distance,
                vector: None,  // 默认不返回向量
                metadata: None, // 默认不返回元数据
            }
        })
        .collect()
}

/// 在指定层执行搜索
fn search_layer(
    query: &[f32],
    entry_point: NodeIndex,
    ef: usize,
    level: usize,
    nodes: &HashMap<NodeIndex, SharedNode>,
    distance_type: DistanceType,
    filter: Option<&FilterFunction>,
) -> Vec<SearchCandidate> {
    // 已访问节点集合
    let mut visited = HashSet::new();
    // 候选节点优先队列（距离小的优先）
    let mut candidates = BinaryHeap::new();
    // 结果集优先队列（距离大的优先，用于保留最近的ef个节点）
    let mut results = BinaryHeap::new();
    
    // 计算查询向量与入口点的距离
    let ep_distance = calculate_distance(query, entry_point, nodes, distance_type);
    
    // 初始化搜索状态
    let entry_candidate = SearchCandidate { node_idx: entry_point, distance: ep_distance };
    candidates.push(entry_candidate.clone());
    results.push(FurthestFirstCandidate(Candidate {
        distance: ep_distance,
        node_index: entry_point,
    }));
    visited.insert(entry_point);
    
    // 当候选集不为空时继续搜索
    while let Some(current) = candidates.pop() {
        // 如果结果集最远的点比当前候选点近，且结果集已满，则结束搜索
        if !results.is_empty() && results.peek().unwrap().0.distance < current.distance && results.len() >= ef {
            break;
        }
        
        // 获取当前节点在指定层的连接
        let connections = get_node_connections(current.node_idx, level, nodes);
        
        // 检查每个连接（connections 是 Vec<(usize, f32)>）
        for (neighbor_idx, _) in connections {
            if !visited.insert(neighbor_idx) {
                continue; // 已访问过此节点
            }
            
            // 计算查询向量与邻居的距离
            let neighbor_distance = calculate_distance(query, neighbor_idx, nodes, distance_type);
            
            // 如果有过滤函数，检查节点是否满足条件
            if let Some(filter_fn) = filter {
                let node_lock = nodes.get(&neighbor_idx).unwrap().read();
                if !filter_fn(&(node_lock.id.as_u128() as u64), None) {
                    continue;
                }
            }
            
            // 如果结果集未满或邻居距离比结果集中最远的距离更近
            if results.len() < ef || neighbor_distance < results.peek().unwrap().0.distance {
                let neighbor_candidate = SearchCandidate { node_idx: neighbor_idx, distance: neighbor_distance };
                candidates.push(neighbor_candidate.clone());
                results.push(FurthestFirstCandidate(Candidate {
                    distance: neighbor_distance,
                    node_index: neighbor_idx,
                }));
                
                // 如果结果集超过ef，移除最远的节点
                if results.len() > ef {
                    results.pop();
                }
            }
        }
    }
    
    // 将结果从优先队列转换为向量，按距离升序排序
    let mut result_vec = Vec::with_capacity(results.len());
    while let Some(FurthestFirstCandidate(candidate)) = results.pop() {
        // 将 Candidate 转换为 SearchCandidate
        result_vec.push(SearchCandidate {
            node_idx: candidate.node_index,
            distance: candidate.distance,
        });
    }
    result_vec.reverse(); // 因为是从最远到最近弹出的，所以需要反转
    
    result_vec
}

/// 基于配置执行搜索
pub fn search_with_config(
    query: &[f32],
    config: &SearchConfig,
    nodes: &HashMap<NodeIndex, SharedNode>,
    entry_point: NodeIndex,
    max_level: usize,
    distance_type: DistanceType,
    node_id_to_vector: impl Fn(NodeIndex) -> Option<Vec<f32>>,
    node_id_to_metadata: impl Fn(NodeIndex) -> Option<HashMap<String, String>>,
) -> SearchResults {
    // 根据不同的搜索策略执行相应的搜索算法
    let raw_results = match &config.strategy {
        SearchStrategy::Standard { ef_search } => {
            search_knn(
                query,
                config.limit,
                *ef_search,
                entry_point,
                max_level,
                nodes,
                distance_type,
                None,
            )
        },
        SearchStrategy::Range { radius, max_elements } => {
            search_range(
                query,
                *radius,
                *max_elements.min(&config.limit),
                entry_point,
                max_level,
                nodes,
                distance_type,
            )
        },
        SearchStrategy::Hybrid { ef_search, filter } => {
            search_knn(
                query,
                config.limit,
                *ef_search,
                entry_point,
                max_level,
                nodes,
                distance_type,
                filter.as_ref(),
            )
        },
    };

    // 如果需要，添加向量和元数据到结果中
    if config.include_vectors || config.include_metadata {
        raw_results.into_iter()
            .map(|mut result| {
                // 从结果中查找对应的节点索引
                // 注意：这里需要根据 result.id 查找节点，但 result.id 是 VectorId (u64)
                // 而 node_id_to_vector 和 node_id_to_metadata 接受 NodeIndex (usize)
                // 由于类型不匹配，我们需要遍历 nodes 来查找匹配的节点
                let mut found_node_index = None;
                for (node_idx, node) in nodes {
                    let node_lock = node.read();
                    // 将 Uuid 转换为 u64 进行比较（这里使用简单的哈希转换）
                    if node_lock.id.as_u128() as u64 == result.id {
                        found_node_index = Some(*node_idx);
                        break;
                    }
                }
                
                if let Some(node_index) = found_node_index {
                    if config.include_vectors {
                        result.vector = node_id_to_vector(node_index);
                    }
                    
                    if config.include_metadata {
                        result.metadata = node_id_to_metadata(node_index);
                    }
                }
                
                result
            })
            .collect()
    } else {
        raw_results
    }
}

/// 范围搜索：搜索给定半径内的所有向量
fn search_range(
    query: &[f32],
    radius: f32,
    max_elements: usize,
    entry_point: NodeIndex,
    max_level: usize,
    nodes: &HashMap<NodeIndex, SharedNode>,
    distance_type: DistanceType,
) -> SearchResults {
    // 先执行KNN搜索，获取大约2*max_elements个候选结果
    let candidates = search_knn(
        query,
        max_elements * 2,
        max_elements * 3,
        entry_point,
        max_level,
        nodes,
        distance_type,
        None,
    );
    
    // 过滤出半径范围内的结果，并限制数量
    candidates.into_iter()
        .filter(|result| result.distance <= radius)
        .take(max_elements)
        .collect()
}

/// 批量搜索：同时处理多个查询
pub fn batch_search(
    queries: &[Vec<f32>],
    config: &SearchConfig,
    nodes: &HashMap<NodeIndex, SharedNode>,
    entry_point: NodeIndex,
    max_level: usize,
    distance_type: DistanceType,
    node_id_to_vector: impl Fn(NodeIndex) -> Option<Vec<f32>>,
    node_id_to_metadata: impl Fn(NodeIndex) -> Option<HashMap<String, String>>,
) -> Vec<SearchResults> {
    queries.iter()
        .map(|query| {
            search_with_config(
                query,
                config,
                nodes,
                entry_point,
                max_level,
                distance_type,
                &node_id_to_vector,
                &node_id_to_metadata,
            )
        })
        .collect()
}

/// 计算查询向量与指定节点的距离
fn calculate_distance(
    query: &[f32],
    node_idx: NodeIndex,
    nodes: &HashMap<NodeIndex, SharedNode>,
    distance_type: DistanceType,
) -> Distance {
    let node_lock = nodes.get(&node_idx).unwrap().read();
    distance::calculate_distance(query, &node_lock.vector, distance_type)
}

/// 获取节点在指定层的连接
fn get_node_connections(
    node_idx: NodeIndex,
    level: usize,
    nodes: &HashMap<NodeIndex, SharedNode>,
) -> Vec<(usize, f32)> {
    let node_lock = nodes.get(&node_idx).unwrap().read();
    node_lock.get_connections(level)
        .cloned()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use uuid::Uuid;
    
    // 创建测试用的节点与连接
    fn create_test_graph() -> (HashMap<NodeIndex, SharedNode>, NodeIndex, usize) {
        let mut nodes = HashMap::new();
        
        // 创建10个节点
        for i in 0..10 {
            // 创建一个简单的二维向量，方便测试距离计算
            let vector = vec![i as f32, (i * i) as f32];
            let node = HNSWNode::new(i, Uuid::new_v4(), vector, i % 3);
            nodes.insert(i, Arc::new(RwLock::new(node)));
        }
        
        // 添加连接
        // 节点0与节点1、2、3相连
        {
            let mut node = nodes.get(&0).unwrap().write();
            node.add_connection(0, (1, 1.0));
            node.add_connection(0, (2, 2.0));
            node.add_connection(0, (3, 3.0));
        }
        
        // 节点1与节点4、5相连
        {
            let mut node = nodes.get(&1).unwrap().write();
            node.add_connection(0, (4, 1.5));
            node.add_connection(0, (5, 2.5));
        }
        
        // 节点2与节点6、7相连
        {
            let mut node = nodes.get(&2).unwrap().write();
            node.add_connection(0, (6, 1.8));
            node.add_connection(0, (7, 2.8));
        }
        
        // 在level 1上添加一些连接
        {
            let mut node = nodes.get(&0).unwrap().write();
            node.add_connection(1, (3, 3.0));
            node.add_connection(1, (6, 6.0));
        }
        
        {
            let mut node = nodes.get(&3).unwrap().write();
            node.add_connection(1, (6, 3.5));
            node.add_connection(1, (9, 4.5));
        }
        
        // 最高层为level 2
        {
            let mut node = nodes.get(&6).unwrap().write();
            node.add_connection(2, (9, 3.2));
        }
        
        (nodes, 0, 2) // 返回节点集合、入口点和最高层
    }
    
    #[test]
    fn test_search_knn() {
        let (nodes, entry_point, max_level) = create_test_graph();
        
        // 查询向量 [1.0, 1.0]
        let query = vec![1.0, 1.0];
        let k = 3;
        let ef_search = 10;
        
        let results = search_knn(
            &query,
            k,
            ef_search,
            entry_point,
            max_level,
            &nodes,
            DistanceType::Euclidean,
            None,
        );
        
        // 验证结果数量
        assert_eq!(results.len(), k);
        
        // 验证结果是按距离排序的
        for i in 1..results.len() {
            assert!(results[i-1].distance <= results[i].distance);
        }
    }
    
    #[test]
    fn test_search_with_filter() {
        let (nodes, entry_point, max_level) = create_test_graph();
        
        // 查询向量 [1.0, 1.0]
        let query = vec![1.0, 1.0];
        let k = 5;
        let ef_search = 10;
        
        // 只返回偶数ID的节点
        let filter: FilterFunction = Arc::new(|id, _| id % 2 == 0);
        
        let results = search_knn(
            &query,
            k,
            ef_search,
            entry_point,
            max_level,
            &nodes,
            DistanceType::Euclidean,
            Some(&filter),
        );
        
        // 验证所有结果都符合过滤条件
        for result in &results {
            assert_eq!(result.id % 2, 0);
        }
    }
}
