use crate::vector::index::interfaces::VectorIndex;
use crate::vector::index::types::{IndexConfig, SearchResult};
use crate::vector::{Vector, SimilarityMetric};
use crate::Result;
use crate::Error;
use std::collections::{HashMap, BinaryHeap, HashSet};
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};
use base64::{Engine as _, engine::general_purpose};
// use std::sync::{Arc, RwLock};
// use std::str::FromStr;
// use std::hash::{Hash, Hasher};
// use std::collections::hash_map::DefaultHasher;

/// NGT (Neighborhood Graph and Tree) 索引
/// 
/// NGT是一种高性能的近似最近邻搜索算法，结合了图索引和树索引的优点
/// 通过构建邻域图和搜索树来实现快速的相似性搜索
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NGTIndex {
    /// 配置参数
    config: IndexConfig,
    /// 向量存储
    vectors: Vec<Vector>,
    /// 邻域图：每个节点到其邻居的映射
    graph: HashMap<usize, Vec<usize>>,
    /// 搜索树：用于初始搜索的树结构
    search_tree: Vec<TreeNode>,
    /// 向量维度
    dimension: usize,
    /// 是否已构建索引
    is_built: bool,
    /// 边的大小（每个节点的最大邻居数）
    edge_size: usize,
    /// 搜索时的候选数量
    search_k: usize,
}

/// 搜索树节点
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TreeNode {
    /// 节点ID
    id: usize,
    /// 向量数据
    vector: Vector,
    /// 左子树
    left: Option<Box<TreeNode>>,
    /// 右子树
    right: Option<Box<TreeNode>>,
    /// 分割维度
    split_dim: usize,
    /// 分割值
    split_value: f32,
}

/// 搜索候选项
#[derive(Debug, Clone)]
struct SearchCandidate {
    id: usize,
    distance: f32,
    checked: bool,
}

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for SearchCandidate {}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance.partial_cmp(&self.distance) // 最小堆
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl NGTIndex {
    /// 创建新的NGT索引
    pub fn new(config: IndexConfig) -> Result<Self> {
        let edge_size = if config.ngt_edge_size > 0 {
            config.ngt_edge_size
        } else {
            // 根据维度动态调整边数
            ((config.dimension as f32).sqrt() * 2.0).round() as usize
        };
        
        let search_k = if config.search_k > 0 {
            config.search_k
        } else {
            edge_size * 4
        };

        Ok(Self {
            dimension: config.dimension,
            edge_size,
            search_k,
            config,
            vectors: Vec::new(),
            graph: HashMap::new(),
            search_tree: Vec::new(),
            is_built: false,
        })
    }

    /// 构建搜索树
    fn build_search_tree(&mut self) -> Result<()> {
        if self.vectors.is_empty() {
            return Ok(());
        }

        // 选择根节点（随机选择或基于启发式）
        let root_id = 0;
        let root_vector = self.vectors[root_id].clone();
        
        // 构建二叉搜索树
        let mut indices: Vec<usize> = (0..self.vectors.len()).collect();
        self.search_tree = vec![self.build_tree_node(&mut indices, 0)?];
        
        Ok(())
    }

    /// 递归构建树节点
    fn build_tree_node(&self, indices: &mut [usize], depth: usize) -> Result<TreeNode> {
        if indices.is_empty() {
            return Err(Error::vector("无法从空索引创建树节点".to_string()));
        }

        if indices.len() == 1 {
            let id = indices[0];
            return Ok(TreeNode {
                id,
                vector: self.vectors[id].clone(),
                left: None,
                right: None,
                split_dim: 0,
                split_value: 0.0,
            });
        }

        // 选择分割维度（轮流或基于方差）
        let split_dim = depth % self.dimension;
        
        // 计算分割值（中位数）
        let mut values: Vec<f32> = indices.iter()
            .map(|&i| self.vectors[i].data[split_dim])
            .collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let split_value = values[values.len() / 2];

        // 分割索引
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        
        for &idx in indices.iter() {
            if self.vectors[idx].data[split_dim] <= split_value {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        // 确保两边都有节点
        if left_indices.is_empty() {
            left_indices.push(right_indices.pop().unwrap());
        } else if right_indices.is_empty() {
            right_indices.push(left_indices.pop().unwrap());
        }

        // 选择当前节点
        let current_id = left_indices[0];
        
        // 递归构建子树
        let left = if left_indices.len() > 1 {
            Some(Box::new(self.build_tree_node(&mut left_indices[1..], depth + 1)?))
        } else {
            None
        };
        
        let right = if !right_indices.is_empty() {
            Some(Box::new(self.build_tree_node(&mut right_indices, depth + 1)?))
        } else {
            None
        };

        Ok(TreeNode {
            id: current_id,
            vector: self.vectors[current_id].clone(),
            left,
            right,
            split_dim,
            split_value,
        })
    }

    /// 构建邻域图
    fn build_graph(&mut self) -> Result<()> {
        for i in 0..self.vectors.len() {
            let mut neighbors = Vec::new();
            let mut distances: Vec<(usize, f32)> = Vec::new();
            
            // 计算到所有其他向量的距离
            for j in 0..self.vectors.len() {
                if i != j {
                    let distance = self.calculate_distance(&self.vectors[i], &self.vectors[j])?;
                    distances.push((j, distance));
                }
            }
            
            // 排序并选择最近的邻居
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            
            for (neighbor_id, _) in distances.iter().take(self.edge_size) {
                neighbors.push(*neighbor_id);
            }
            
            self.graph.insert(i, neighbors);
        }
        
        Ok(())
    }

    /// 计算两个向量间的距离
    fn calculate_distance(&self, v1: &Vector, v2: &Vector) -> Result<f32> {
        if v1.data.len() != v2.data.len() {
            return Err(Error::vector("向量维度不匹配".to_string()));
        }

        match self.config.metric {
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
                    Ok(1.0) // 最大距离
                } else {
                    Ok(1.0 - (dot_product / (norm1 * norm2)))
                }
            },
            SimilarityMetric::DotProduct => {
                let dot_product: f32 = v1.data.iter()
                    .zip(v2.data.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                Ok(-dot_product) // 转换为距离（距离越小越相似）
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

    /// 在搜索树中查找初始候选
    fn search_tree(&self, query: &[f32], k: usize) -> Result<Vec<usize>> {
        if self.search_tree.is_empty() {
            return Ok(Vec::new());
        }

        let mut candidates = Vec::new();
        self.tree_search_recursive(&self.search_tree[0], query, &mut candidates)?;
        
        // 计算距离并排序
        let mut candidate_distances: Vec<(usize, f32)> = Vec::new();
        for &candidate_id in &candidates {
            let distance = self.calculate_distance_to_slice(&self.vectors[candidate_id], query)?;
            candidate_distances.push((candidate_id, distance));
        }
        
        candidate_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        
        Ok(candidate_distances.iter().take(k).map(|(id, _)| *id).collect())
    }

    /// 递归搜索树
    fn tree_search_recursive(&self, node: &TreeNode, query: &[f32], candidates: &mut Vec<usize>) -> Result<()> {
        candidates.push(node.id);
        
        if query[node.split_dim] <= node.split_value {
            if let Some(ref left) = node.left {
                self.tree_search_recursive(left, query, candidates)?;
            }
        } else {
            if let Some(ref right) = node.right {
                self.tree_search_recursive(right, query, candidates)?;
            }
        }
        
        Ok(())
    }

    /// 计算向量与查询切片的距离
    fn calculate_distance_to_slice(&self, vector: &Vector, query: &[f32]) -> Result<f32> {
        if vector.data.len() != query.len() {
            return Err(Error::vector("向量维度不匹配".to_string()));
        }

        match self.config.metric {
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
    fn graph_search(&self, query: &[f32], initial_candidates: Vec<usize>, k: usize) -> Result<Vec<SearchResult>> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        // 初始化候选队列
        for candidate_id in initial_candidates {
            let distance = self.calculate_distance_to_slice(&self.vectors[candidate_id], query)?;
            candidates.push(SearchCandidate {
                id: candidate_id,
                distance,
                checked: false,
            });
        }

        while let Some(current) = candidates.pop() {
            if current.checked || visited.contains(&current.id) {
                continue;
            }

            visited.insert(current.id);

            // 添加到结果集
            let metadata = self.vectors[current.id].metadata.as_ref()
                .map(|m| serde_json::to_value(m).unwrap_or(serde_json::Value::Null));
            results.push(SearchResult {
                id: current.id.to_string(),
                distance: current.distance,
                metadata: metadata,
            });

            // 如果结果集过大，移除最差的结果
            if results.len() > k {
                results.pop();
            }

            // 探索邻居
            if let Some(neighbors) = self.graph.get(&current.id) {
                for &neighbor_id in neighbors {
                    if !visited.contains(&neighbor_id) {
                        let distance = self.calculate_distance_to_slice(&self.vectors[neighbor_id], query)?;
                        candidates.push(SearchCandidate {
                            id: neighbor_id,
                            distance,
                            checked: false,
                        });
                    }
                }
            }

            // 限制搜索范围
            if visited.len() > self.search_k {
                break;
            }
        }

        // 转换结果
        let mut final_results: Vec<SearchResult> = results.into_sorted_vec();
        final_results.truncate(k);
        final_results.reverse(); // 最高相似度在前

        Ok(final_results)
    }

    /// 构建索引
    pub fn build(&mut self) -> Result<()> {
        if self.vectors.is_empty() {
            return Err(Error::vector("无法为空向量集构建索引".to_string()));
        }

        // 构建搜索树
        self.build_search_tree()?;
        
        // 构建邻域图
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
        self.is_built = false; // 需要重新构建索引
        
        Ok(id)
    }

    /// 移除向量
    pub fn remove_vector(&mut self, id: usize) -> Result<()> {
        if id >= self.vectors.len() {
            return Err(Error::vector(format!("向量ID {} 不存在", id)));
        }

        // 标记为已删除（实际项目中可能需要更复杂的删除策略）
        self.vectors[id].data.clear();
        self.is_built = false;
        
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
        self.is_built = false;
        
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



impl NGTIndex {
    /// 生产级ID解析器（私有方法）
    fn parse_vector_id(&self, id: &str) -> Result<usize> {
        // 支持多种ID格式：纯数字、UUID、自定义格式等
        if let Ok(numeric_id) = id.parse::<usize>() {
            return Ok(numeric_id);
        }
        
        // 支持十六进制ID
        if id.starts_with("0x") || id.starts_with("0X") {
            if let Ok(hex_id) = usize::from_str_radix(&id[2..], 16) {
                return Ok(hex_id);
            }
        }
        
        // 支持Base64编码的ID
        if let Ok(base64_bytes) = general_purpose::STANDARD.decode(id) {
            if base64_bytes.len() == std::mem::size_of::<usize>() {
                let id_bytes: [u8; std::mem::size_of::<usize>()] = 
                    base64_bytes.try_into().map_err(|_| Error::vector("Base64 ID转换失败".to_string()))?;
                return Ok(usize::from_le_bytes(id_bytes));
            }
        }
        
        // 支持UUID格式（简化为哈希映射）
        if id.len() == 36 && id.chars().filter(|&c| c == '-').count() == 4 {
            // 使用UUID的哈希值作为数字ID
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            
            let mut hasher = DefaultHasher::new();
            id.hash(&mut hasher);
            let hash_id = hasher.finish() as usize;
            return Ok(hash_id % (usize::MAX / 2)); // 避免溢出
        }
        
        // 支持自定义前缀格式（如：vec_123456）
        if id.contains('_') {
            let parts: Vec<&str> = id.split('_').collect();
            if parts.len() >= 2 {
                if let Ok(suffix_id) = parts.last().unwrap().parse::<usize>() {
                    return Ok(suffix_id);
                }
            }
        }
        
        Err(Error::vector(format!("无法解析向量ID: {}", id)))
    }
    
    /// 从图中删除向量节点（私有方法）
    fn delete_vector_from_graph(&mut self, vector_id: usize) -> Result<()> {
        // 1. 移除该节点的所有边
        self.graph.remove(&vector_id);
        
        // 2. 从其他节点的邻接列表中移除该节点
        for (_, neighbors) in self.graph.iter_mut() {
            neighbors.retain(|&neighbor_id| neighbor_id != vector_id);
        }
        
        // 3. 重新构建受影响节点的邻接关系
        self.rebuild_local_connections(vector_id)?;
        
        Ok(())
    }
    
    /// 重建局部连接（删除节点后修复图的连通性）（私有方法）
    fn rebuild_local_connections(&mut self, deleted_id: usize) -> Result<()> {
        // 获取被删除节点的邻居
        let affected_neighbors: Vec<usize> = self.graph.values()
            .flatten()
            .filter(|&&id| id != deleted_id)
            .copied()
            .collect();
        
        // 为每个受影响的邻居重新计算最佳连接
        for neighbor_id in affected_neighbors {
            if neighbor_id >= self.vectors.len() || self.vectors[neighbor_id].data.is_empty() {
                continue;
            }
            
            // 找到该邻居的新最佳连接
            let new_connections = self.find_best_neighbors(neighbor_id, self.edge_size)?;
            
            // 更新邻接列表
            if let Some(neighbor_edges) = self.graph.get_mut(&neighbor_id) {
                neighbor_edges.clear();
                neighbor_edges.extend(new_connections);
            }
        }
        
        Ok(())
    }
    
    /// 为指定节点找到最佳邻居（私有方法）
    fn find_best_neighbors(&self, node_id: usize, max_connections: usize) -> Result<Vec<usize>> {
        if node_id >= self.vectors.len() || self.vectors[node_id].data.is_empty() {
            return Ok(Vec::new());
        }
        
        let node_vector = &self.vectors[node_id];
        let mut candidates = Vec::new();
        
        // 计算与所有其他节点的距离
        for (candidate_id, candidate_vector) in self.vectors.iter().enumerate() {
            if candidate_id == node_id || candidate_vector.data.is_empty() {
                continue;
            }
            
            let distance = self.calculate_distance(node_vector, candidate_vector)?;
            candidates.push((candidate_id, distance));
        }
        
        // 按距离排序并选择最近的邻居
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(candidates.into_iter()
            .take(max_connections)
            .map(|(id, _)| id)
            .collect())
    }
}

impl VectorIndex for NGTIndex {
    fn add(&mut self, vector: Vector) -> Result<()> {
        if vector.data.len() != self.dimension {
            return Err(Error::vector(format!(
                "向量维度不匹配: 期望 {}, 实际 {}",
                self.dimension,
                vector.data.len()
            )));
        }

        self.vectors.push(vector);
        self.is_built = false; // 需要重新构建索引
        
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

        // 1. 从搜索树获取初始候选
        let initial_candidates = self.search_tree(query, self.search_k)?;
        
        // 2. 在图中搜索
        self.graph_search(query, initial_candidates, limit)
    }

    fn delete(&mut self, id: &str) -> Result<bool> {
        // 生产级ID处理：支持多种ID格式
        let vector_id: usize = self.parse_vector_id(id)?;
        
        if vector_id >= self.vectors.len() {
            return Ok(false);
        }

        // 检查向量是否已被删除
        if self.vectors[vector_id].data.is_empty() {
            return Ok(false);
        }

        // 执行删除操作
        self.delete_vector_from_graph(vector_id)?;
        self.vectors[vector_id].data.clear();
        self.is_built = false;
        
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
        let deserialized: NGTIndex = bincode::deserialize(data)
            .map_err(|e| Error::vector(format!("反序列化失败: {}", e)))?;
        
        *self = deserialized;
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn VectorIndex + Send + Sync> {
        Box::new(self.clone())
    }

    fn deserialize_box(data: &[u8]) -> Result<Box<dyn VectorIndex + Send + Sync>> where Self: Sized {
        let index: NGTIndex = bincode::deserialize(data)
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
        self.graph.clear();
        self.search_tree.clear();
        self.is_built = false;
        Ok(())
    }

    fn remove(&mut self, vector_id: u64) -> Result<()> {
        let id = vector_id as usize;
        if id >= self.vectors.len() {
            return Err(Error::vector(format!("向量ID {} 不存在", id)));
        }

        self.vectors[id].data.clear();
        self.is_built = false;
        Ok(())
    }
} 