use crate::vector::index::interfaces::VectorIndex;
use crate::vector::index::types::{IndexConfig, SearchResult};
use crate::vector::{Vector, SimilarityMetric};
use crate::Result;
use crate::Error;
use std::collections::{HashMap, BinaryHeap, HashSet};
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};

/// 通用图索引
/// 
/// 基于图结构的向量索引，不同于HNSW的另一种图索引实现
/// 使用邻接表表示图结构，支持多种图构建策略和搜索算法
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphIndex {
    /// 配置参数
    config: IndexConfig,
    /// 向量存储
    vectors: Vec<Vector>,
    /// 邻接图：节点ID到邻居列表的映射
    adjacency_list: HashMap<usize, Vec<GraphEdge>>,
    /// 图节点信息
    nodes: HashMap<usize, GraphNode>,
    /// 向量维度
    dimension: usize,
    /// 是否已构建索引
    is_built: bool,
    /// 图的度数（每个节点的最大邻居数）
    graph_degree: usize,
    /// 搜索时的候选数量
    search_candidates: usize,
}

/// 图边
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GraphEdge {
    /// 目标节点ID
    target: usize,
    /// 边权重（距离）
    weight: f32,
    /// 边类型
    edge_type: EdgeType,
}

/// 边类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
enum EdgeType {
    /// 强连接边（双向保证）
    Strong,
    /// 弱连接边（单向可能）
    Weak,
    /// 临时边（搜索时使用）
    Temporary,
}

/// 图节点
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GraphNode {
    /// 节点ID
    id: usize,
    /// 节点层级（用于分层图）
    level: usize,
    /// 节点度数
    degree: usize,
    /// 节点状态
    status: NodeStatus,
}

/// 节点状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
enum NodeStatus {
    /// 活跃节点
    Active,
    /// 已删除节点
    Deleted,
    /// 临时节点
    Temporary,
}

/// 搜索候选项
#[derive(Debug, Clone)]
struct SearchCandidate {
    id: usize,
    distance: f32,
    visited: bool,
}

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for SearchCandidate {}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl GraphIndex {
    /// 创建新的图索引
    pub fn new(config: IndexConfig) -> Result<Self> {
        let graph_degree = if config.graph_degree > 0 {
            config.graph_degree
        } else {
            // 根据维度动态调整图的度数
            ((config.dimension as f32).sqrt() * 1.5).round() as usize
        };

        let search_candidates = if config.search_k > 0 {
            config.search_k
        } else {
            graph_degree * 3
        };

        Ok(Self {
            dimension: config.dimension,
            graph_degree: graph_degree.max(4).min(64),
            search_candidates: search_candidates.max(10).min(1000),
            config,
            vectors: Vec::new(),
            adjacency_list: HashMap::new(),
            nodes: HashMap::new(),
            is_built: false,
        })
    }

    /// 构建图索引
    fn build_graph(&mut self) -> Result<()> {
        if self.vectors.is_empty() {
            return Ok(());
        }

        // 1. 初始化节点
        self.initialize_nodes()?;

        // 2. 构建连接（使用k-NN图方法）
        self.build_knn_graph()?;

        // 3. 优化图结构
        self.optimize_graph()?;

        Ok(())
    }

    /// 初始化图节点
    fn initialize_nodes(&mut self) -> Result<()> {
        self.nodes.clear();

        for i in 0..self.vectors.len() {
            if self.vectors[i].data.is_empty() {
                continue; // 跳过已删除的向量
            }

            // 生产级层级分配：基于节点重要性和图的全局结构
            let level = self.calculate_node_level(i)?;

            self.nodes.insert(i, GraphNode {
                id: i,
                level,
                degree: 0,
                status: NodeStatus::Active,
            });
        }

        Ok(())
    }
    
    /// 生产级节点层级计算
    fn calculate_node_level(&self, node_id: usize) -> Result<usize> {
        // 基于多个因素计算节点层级：
        // 1. 向量的局部密度
        // 2. 与其他向量的距离分布
        // 3. 随机因子（避免确定性偏差）
        
        let node_vector = &self.vectors[node_id];
        let mut local_density = 0.0;
        let mut distance_variance = 0.0;
        let sample_size = (self.vectors.len() / 10).max(10).min(100); // 采样计算
        
        // 计算局部密度
        let mut distances = Vec::new();
        let mut sample_count = 0;
        
        for (i, vector) in self.vectors.iter().enumerate() {
            if i != node_id && !vector.data.is_empty() && sample_count < sample_size {
                let distance = self.calculate_distance(node_vector, vector)?;
                distances.push(distance);
                sample_count += 1;
            }
        }
        
        if !distances.is_empty() {
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            
            // 计算k近邻平均距离作为局部密度指标
            let k = (distances.len() / 4).max(1).min(10);
            local_density = distances.iter().take(k).sum::<f32>() / k as f32;
            
            // 计算距离方差作为分布指标
            let mean_distance = distances.iter().sum::<f32>() / distances.len() as f32;
            distance_variance = distances.iter()
                .map(|&d| (d - mean_distance).powi(2))
                .sum::<f32>() / distances.len() as f32;
        }
        
        // 基于密度和方差计算层级概率
        let density_factor = 1.0 / (1.0 + local_density); // 密度越高，层级越高的概率越大
        let variance_factor = distance_variance.sqrt() / (distance_variance.sqrt() + 1.0); // 方差适中最好
        
        // 结合随机因子
        let random_factor = (node_id as f32 * 0.618033988749) % 1.0; // 使用黄金比例的小数部分作为伪随机
        
        // 综合计算层级
        let level_probability = density_factor * 0.4 + variance_factor * 0.3 + random_factor * 0.3;
        
        // 根据概率分配层级（分层图中，大多数节点在低层）
        let max_level = (self.vectors.len() as f32).log2().ceil() as usize;
        let level = if level_probability > 0.8 {
            max_level.min(3)
        } else if level_probability > 0.6 {
            max_level.min(2)
        } else if level_probability > 0.4 {
            max_level.min(1)
        } else {
            0
        };
        
        Ok(level)
    }

    /// 构建k-NN图
    fn build_knn_graph(&mut self) -> Result<()> {
        self.adjacency_list.clear();

        // 为每个节点找到k个最近邻居
        // 先收集所有节点ID，避免在迭代keys的同时对self.nodes进行可变借用
        let node_ids: Vec<usize> = self.nodes.keys().cloned().collect();
        for node_id in node_ids {
            let mut neighbors = self.find_knn_neighbors(node_id, self.graph_degree)?;
            
            // 创建边
            let mut edges = Vec::new();
            for (neighbor_id, distance) in neighbors {
                if neighbor_id != node_id {
                    edges.push(GraphEdge {
                        target: neighbor_id,
                        weight: distance,
                        edge_type: EdgeType::Strong,
                    });
                }
            }

            self.adjacency_list.insert(node_id, edges);
            
            // 更新节点度数
            if let Some(node) = self.nodes.get_mut(&node_id) {
                node.degree = self.adjacency_list.get(&node_id).map_or(0, |e| e.len());
            }
        }

        // 确保图的双向连接性
        self.ensure_bidirectional_connections()?;

        Ok(())
    }

    /// 找到k个最近邻居
    fn find_knn_neighbors(&self, node_id: usize, k: usize) -> Result<Vec<(usize, f32)>> {
        let mut distances = Vec::new();
        let query_vector = &self.vectors[node_id];

        for &other_id in self.nodes.keys() {
            if other_id != node_id {
                let distance = self.calculate_distance(query_vector, &self.vectors[other_id])?;
                distances.push((other_id, distance));
            }
        }

        // 排序并取前k个
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        distances.truncate(k);

        Ok(distances)
    }

    /// 确保双向连接
    fn ensure_bidirectional_connections(&mut self) -> Result<()> {
        let mut additional_edges = HashMap::new();

        // 收集需要添加的反向边
        for (source_id, edges) in &self.adjacency_list {
            for edge in edges {
                let target_id = edge.target;
                
                // 检查反向边是否存在
                let has_reverse = self.adjacency_list
                    .get(&target_id)
                    .map_or(false, |target_edges| {
                        target_edges.iter().any(|e| e.target == *source_id)
                    });

                if !has_reverse {
                    // 添加反向边
                    additional_edges
                        .entry(target_id)
                        .or_insert_with(Vec::new)
                        .push(GraphEdge {
                            target: *source_id,
                            weight: edge.weight,
                            edge_type: EdgeType::Weak,
                        });
                }
            }
        }

        // 应用额外的边
        for (node_id, edges) in additional_edges {
            self.adjacency_list
                .entry(node_id)
                .or_insert_with(Vec::new)
                .extend(edges);
        }

        Ok(())
    }

    /// 优化图结构
    fn optimize_graph(&mut self) -> Result<()> {
        // 移除冗余边
        self.remove_redundant_edges()?;
        
        // 平衡节点度数
        self.balance_node_degrees()?;

        Ok(())
    }

    /// 移除冗余边
    fn remove_redundant_edges(&mut self) -> Result<()> {
        for edges in self.adjacency_list.values_mut() {
            // 保留最短的边，移除距离过大的边
            edges.sort_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap_or(Ordering::Equal));
            
            if edges.len() > self.graph_degree {
                edges.truncate(self.graph_degree);
            }
        }

        Ok(())
    }

    /// 平衡节点度数
    fn balance_node_degrees(&mut self) -> Result<()> {
        let max_degree = self.graph_degree;
        
        for (node_id, edges) in self.adjacency_list.iter_mut() {
            if edges.len() > max_degree {
                // 保留权重最小的边
                edges.sort_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap_or(Ordering::Equal));
                edges.truncate(max_degree);
            }

            // 更新节点度数
            if let Some(node) = self.nodes.get_mut(node_id) {
                node.degree = edges.len();
            }
        }

        Ok(())
    }

    /// 计算两个向量间的距离
    fn calculate_distance(&self, v1: &Vector, v2: &Vector) -> Result<f32> {
        if v1.data.len() != v2.data.len() {
            return Err(Error::vector("向量维度不匹配".to_string()));
        }

        match self.config.similarity_metric {
            SimilarityMetric::Euclidean => {
                let distance = v1.data.iter()
                    .zip(v2.data.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                Ok(distance)
            },
            SimilarityMetric::Cosine => {
                let dot_product: f32 = v1.data.iter()
                    .zip(v2.data.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                
                let norm1: f32 = v1.data.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                let norm2: f32 = v2.data.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                
                if norm1 == 0.0 || norm2 == 0.0 {
                    Ok(1.0)
                } else {
                    Ok(1.0 - (dot_product / (norm1 * norm2)))
                }
            },
            SimilarityMetric::DotProduct => {
                let dot_product: f32 = v1.data.iter()
                    .zip(v2.data.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                Ok(-dot_product)
            },
            SimilarityMetric::Manhattan => {
                let distance = v1.data.iter()
                    .zip(v2.data.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>();
                Ok(distance)
            },
        }
    }

    /// 计算向量与查询的距离
    fn calculate_distance_to_slice(&self, vector: &Vector, query: &[f32]) -> Result<f32> {
        if vector.data.len() != query.len() {
            return Err(Error::vector("向量维度不匹配".to_string()));
        }

        match self.config.similarity_metric {
            SimilarityMetric::Euclidean => {
                let distance = vector.data.iter()
                    .zip(query.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                Ok(distance)
            },
            SimilarityMetric::Cosine => {
                let dot_product: f32 = vector.data.iter()
                    .zip(query.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                
                let norm1: f32 = vector.data.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                let norm2: f32 = query.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
                
                if norm1 == 0.0 || norm2 == 0.0 {
                    Ok(1.0)
                } else {
                    Ok(1.0 - (dot_product / (norm1 * norm2)))
                }
            },
            SimilarityMetric::DotProduct => {
                let dot_product: f32 = vector.data.iter()
                    .zip(query.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                Ok(-dot_product)
            },
            SimilarityMetric::Manhattan => {
                let distance = vector.data.iter()
                    .zip(query.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>();
                Ok(distance)
            },
        }
    }

    /// 图搜索
    fn graph_search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }

        // 1. 找到入口点（随机选择或使用启发式）
        let entry_point = self.find_entry_point(query)?;
        
        // 2. 使用贪婪搜索
        let results = self.greedy_search(query, entry_point, k)?;

        Ok(results)
    }

    /// 找到搜索入口点
    fn find_entry_point(&self, query: &[f32]) -> Result<usize> {
        // 生产级入口点选择策略：基于查询向量特征和图结构
        
        // 1. 优先选择高层级节点作为入口点（如果是分层图）
        let mut high_level_candidates = Vec::new();
        let mut max_level = 0;
        
        for (&node_id, node) in &self.nodes {
            if node.status == NodeStatus::Active {
                if node.level > max_level {
                    max_level = node.level;
                    high_level_candidates.clear();
                    high_level_candidates.push(node_id);
                } else if node.level == max_level {
                    high_level_candidates.push(node_id);
                }
            }
        }
        
        // 2. 如果有高层级节点，从中选择最佳入口点
        if !high_level_candidates.is_empty() {
            return self.select_best_entry_from_candidates(query, &high_level_candidates);
        }
        
        // 3. 如果没有分层信息，使用启发式方法选择入口点
        let mut candidates = Vec::new();
        for (&node_id, node) in &self.nodes {
            if node.status == NodeStatus::Active {
                candidates.push(node_id);
            }
        }
        
        if candidates.is_empty() {
            return Err(Error::vector("没有可用的入口点".to_string()));
        }
        
        // 4. 使用多样化采样策略选择入口点
        self.select_diversified_entry_point(query, &candidates)
    }
    
    /// 从候选节点中选择最佳入口点
    fn select_best_entry_from_candidates(&self, query: &[f32], candidates: &[usize]) -> Result<usize> {
        if candidates.is_empty() {
            return Err(Error::vector("候选入口点列表为空".to_string()));
        }
        
        if candidates.len() == 1 {
            return Ok(candidates[0]);
        }
        
        // 计算每个候选点与查询的距离，选择最近的
        let mut best_candidate = candidates[0];
        let mut best_distance = f32::INFINITY;
        
        for &candidate_id in candidates {
            if candidate_id < self.vectors.len() && !self.vectors[candidate_id].data.is_empty() {
                match self.calculate_distance_to_slice(&self.vectors[candidate_id], query) {
                    Ok(distance) => {
                        if distance < best_distance {
                            best_distance = distance;
                            best_candidate = candidate_id;
                        }
                    }
                    Err(_) => continue,
                }
            }
        }
        
        Ok(best_candidate)
    }
    
    /// 使用多样化策略选择入口点
    fn select_diversified_entry_point(&self, query: &[f32], candidates: &[usize]) -> Result<usize> {
        if candidates.is_empty() {
            return Err(Error::vector("候选入口点列表为空".to_string()));
        }
        
        // 如果候选点很少，直接选择最近的
        if candidates.len() <= 3 {
            return self.select_best_entry_from_candidates(query, candidates);
        }
        
        // 对于大量候选点，使用多样化采样
        let sample_size = (candidates.len() / 10).max(5).min(20);
        let mut sampled_candidates = Vec::new();
        
        // 使用确定性采样（基于查询向量的哈希）
        let query_hash = self.hash_query(query);
        for i in 0..sample_size {
            let index = (query_hash as usize + i * 17) % candidates.len();
            sampled_candidates.push(candidates[index]);
        }
        
        // 从采样候选点中选择最佳的
        self.select_best_entry_from_candidates(query, &sampled_candidates)
    }
    
    /// 计算查询向量的哈希值（用于确定性采样）
    fn hash_query(&self, query: &[f32]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // 将浮点数转换为可哈希的形式
        for &value in query.iter().take(16) { // 只使用前16维避免计算过重
            let bits = value.to_bits();
            bits.hash(&mut hasher);
        }
        
        hasher.finish()
    }

    /// 贪婪搜索
    fn greedy_search(&self, query: &[f32], entry_point: usize, k: usize) -> Result<Vec<SearchResult>> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut best_results = BinaryHeap::new();

        // 初始化候选队列
        let entry_distance = self.calculate_distance_to_slice(&self.vectors[entry_point], query)?;
        candidates.push(SearchCandidate {
            id: entry_point,
            distance: entry_distance,
            visited: false,
        });

        while let Some(current) = candidates.pop() {
            if visited.contains(&current.id) {
                continue;
            }

            visited.insert(current.id);

            // 添加到结果集
            let metadata = self.vectors[current.id].metadata.as_ref()
                .map(|m| serde_json::to_value(m).unwrap_or(serde_json::Value::Null));
            best_results.push(SearchResult {
                id: current.id.to_string(),
                distance: current.distance,
                metadata: metadata,
            });

            // 如果结果集过大，移除最差的结果
            if best_results.len() > k {
                best_results.pop();
            }

            // 探索邻居
            if let Some(edges) = self.adjacency_list.get(&current.id) {
                for edge in edges {
                    if !visited.contains(&edge.target) {
                        let distance = self.calculate_distance_to_slice(&self.vectors[edge.target], query)?;
                        candidates.push(SearchCandidate {
                            id: edge.target,
                            distance,
                            visited: false,
                        });
                    }
                }
            }

            // 限制搜索范围
            if visited.len() > self.search_candidates {
                break;
            }
        }

        // 转换结果
        let mut final_results: Vec<SearchResult> = best_results.into_sorted_vec();
        final_results.truncate(k);
        final_results.reverse(); // 最高相似度在前

        Ok(final_results)
    }

    /// 动态添加节点到图中
    fn add_node_to_graph(&mut self, node_id: usize) -> Result<()> {
        // 为新节点找到邻居
        let neighbors = self.find_knn_neighbors(node_id, self.graph_degree)?;
        
        let mut edges = Vec::new();
        for (neighbor_id, distance) in neighbors {
            edges.push(GraphEdge {
                target: neighbor_id,
                weight: distance,
                edge_type: EdgeType::Strong,
            });

            // 为邻居添加反向边
            self.adjacency_list
                .entry(neighbor_id)
                .or_insert_with(Vec::new)
                .push(GraphEdge {
                    target: node_id,
                    weight: distance,
                    edge_type: EdgeType::Weak,
                });
        }

        self.adjacency_list.insert(node_id, edges);

        // 添加节点信息
        self.nodes.insert(node_id, GraphNode {
            id: node_id,
            level: 0,
            degree: self.adjacency_list.get(&node_id).map_or(0, |e| e.len()),
            status: NodeStatus::Active,
        });

        Ok(())
    }

    /// 从图中移除节点
    fn remove_node_from_graph(&mut self, node_id: usize) -> Result<()> {
        // 标记节点为已删除
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.status = NodeStatus::Deleted;
        }

        // 移除所有指向该节点的边
        for edges in self.adjacency_list.values_mut() {
            edges.retain(|edge| edge.target != node_id);
        }

        // 移除节点的邻接列表
        self.adjacency_list.remove(&node_id);

        Ok(())
    }

    /// 构建索引
    pub fn build(&mut self) -> Result<()> {
        if self.vectors.is_empty() {
            return Err(Error::vector("无法为空向量集构建索引".to_string()));
        }

        self.build_graph()?;
        self.is_built = true;
        Ok(())
    }

    /// 添加向量（返回ID的版本）
    pub fn add_vector(&mut self, vector: Vector) -> Result<usize> {
        let id = self.vectors.len();
        
        if vector.data.len() != self.dimension {
            return Err(Error::vector(format!(
                "向量维度不匹配: 期望 {}, 实际 {}",
                self.dimension,
                vector.data.len()
            )));
        }

        self.vectors.push(vector);

        if self.is_built {
            // 如果索引已构建，动态添加节点
            self.add_node_to_graph(id)?;
        } else {
            self.is_built = false; // 需要重新构建索引
        }
        
        Ok(id)
    }

    /// 移除向量
    pub fn remove_vector(&mut self, id: usize) -> Result<()> {
        if id >= self.vectors.len() {
            return Err(Error::vector(format!("向量ID {} 不存在", id)));
        }

        // 标记向量为已删除
        self.vectors[id].data.clear();

        if self.is_built {
            // 如果索引已构建，动态移除节点
            self.remove_node_from_graph(id)?;
        } else {
            self.is_built = false;
        }
        
        Ok(())
    }

    /// 更新向量
    pub fn update_vector(&mut self, id: usize, vector: Vector) -> Result<()> {
        if id >= self.vectors.len() {
            return Err(Error::vector(format!("向量ID {} 不存在", id)));
        }

        if vector.data.len() != self.dimension {
            return Err(Error::vector(format!(
                "向量维度不匹配: 期望 {}, 实际 {}",
                self.dimension,
                vector.data.len()
            )));
        }

        self.vectors[id] = vector;

        if self.is_built {
            // 重新计算该节点的连接
            self.remove_node_from_graph(id)?;
            self.add_node_to_graph(id)?;
        } else {
            self.is_built = false;
        }
        
        Ok(())
    }

    /// 获取向量
    pub fn get_vector(&self, id: usize) -> Result<Option<Vector>> {
        if id < self.vectors.len() && !self.vectors[id].data.is_empty() {
            Ok(Some(self.vectors[id].clone()))
        } else {
            Ok(None)
        }
    }
}

impl VectorIndex for GraphIndex {
    fn add(&mut self, vector: Vector) -> Result<()> {
        if vector.data.len() != self.dimension {
            return Err(Error::vector(format!(
                "向量维度不匹配: 期望 {}, 实际 {}",
                self.dimension,
                vector.data.len()
            )));
        }

        let id = self.vectors.len();
        self.vectors.push(vector);

        if self.is_built {
            // 如果索引已构建，动态添加节点
            self.add_node_to_graph(id)?;
        } else {
            self.is_built = false; // 需要重新构建索引
        }
        
        Ok(())
    }

    fn search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        if !self.is_built {
            return Err(Error::vector("索引未构建".to_string()));
        }

        if query.len() != self.dimension {
            return Err(Error::vector(format!(
                "查询向量维度不匹配: 期望 {}, 实际 {}",
                self.dimension,
                query.len()
            )));
        }

        self.graph_search(query, limit)
    }

    fn delete(&mut self, id: &str) -> Result<bool> {
        let vector_id: usize = id.parse().map_err(|_| Error::vector("无效的向量ID".to_string()))?;
        
        if vector_id >= self.vectors.len() {
            return Ok(false);
        }

        // 标记向量为已删除
        self.vectors[vector_id].data.clear();

        if self.is_built {
            // 如果索引已构建，动态移除节点
            self.remove_node_from_graph(vector_id)?;
        } else {
            self.is_built = false;
        }
        
        Ok(true)
    }

    fn size(&self) -> usize {
        self.vectors.len()
    }

    fn contains(&self, id: &str) -> bool {
        if let Ok(vector_id) = id.parse::<usize>() {
            vector_id < self.vectors.len() && !self.vectors[vector_id].data.is_empty()
        } else {
            false
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn get_config(&self) -> IndexConfig {
        self.config.clone()
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| Error::vector(format!("序列化失败: {}", e)))
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let deserialized: GraphIndex = bincode::deserialize(data)
            .map_err(|e| Error::vector(format!("反序列化失败: {}", e)))?;
        
        *self = deserialized;
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn VectorIndex + Send + Sync> {
        Box::new(self.clone())
    }

    fn deserialize_box(data: &[u8]) -> Result<Box<dyn VectorIndex + Send + Sync>> where Self: Sized {
        let index: GraphIndex = bincode::deserialize(data)
            .map_err(|e| Error::vector(format!("反序列化失败: {}", e)))?;
        Ok(Box::new(index))
    }

    fn get(&self, id: &str) -> Result<Option<Vector>> {
        if let Ok(vector_id) = id.parse::<usize>() {
            if vector_id < self.vectors.len() && !self.vectors[vector_id].data.is_empty() {
                Ok(Some(self.vectors[vector_id].clone()))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    fn clear(&mut self) -> Result<()> {
        self.vectors.clear();
        self.adjacency_list.clear();
        self.nodes.clear();
        self.is_built = false;
        Ok(())
    }

    fn remove(&mut self, vector_id: u64) -> Result<()> {
        let id = vector_id as usize;
        if id >= self.vectors.len() {
            return Err(Error::vector(format!("向量ID {} 不存在", id)));
        }

        // 标记向量为已删除
        self.vectors[id].data.clear();

        if self.is_built {
            // 如果索引已构建，动态移除节点
            self.remove_node_from_graph(id)?;
        } else {
            self.is_built = false;
        }
        
        Ok(())
    }
} 