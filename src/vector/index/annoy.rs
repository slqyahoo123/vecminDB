// ANNOY (Approximate Nearest Neighbors Oh Yeah) 索引实现
// 基于随机投影树的近似最近邻搜索

use std::collections::{HashMap, HashSet};
use rand::{thread_rng, Rng, seq::SliceRandom};
use serde::{Serialize, Deserialize};
use rayon::prelude::*;

use crate::{Error, Result, vector::{Vector, operations::SimilarityMetric, utils::normalize_vector}};
use super::types::{IndexConfig, SearchResult};
use super::interfaces::VectorIndex;
use super::distance::{compute_distance_raw, Distance};

/// ANNOY索引：使用多棵随机投影树实现高效的近似最近邻搜索
/// 适用于大规模数据集，能够平衡查询速度和准确性
pub struct ANNOYIndex {
    /// 索引配置
    config: IndexConfig,
    
    /// 距离计算器
    distance: Box<dyn Distance + Send + Sync>,
    
    /// 随机投影树森林
    forest: Vec<Tree>,
    
    /// 所有向量的存储
    vectors: HashMap<u64, Vec<f32>>,
    
    /// 向量元数据
    vector_metadata: HashMap<u64, HashMap<String, String>>,
    
    /// 向量数量
    vector_count: usize,
    
    /// 是否已构建
    is_built: bool,
}

impl std::fmt::Debug for ANNOYIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ANNOYIndex")
            .field("config", &self.config)
            .field("vector_count", &self.vector_count)
            .finish()
    }
}

/// 树节点类型
#[derive(Clone, Serialize, Deserialize)]
enum Node {
    /// 内部节点，包含分割平面和子节点引用
    Internal {
        /// 左子节点索引
        left: usize,
        /// 右子节点索引
        right: usize,
        /// 分割平面的法向量
        normal: Vec<f32>,
        /// 分割点位置(投影到法向量上的阈值)
        threshold: f32,
    },
    /// 叶子节点，包含向量ID列表
    Leaf {
        /// 叶子中的向量ID集合
        vector_ids: Vec<u64>,
    },
}

/// 树结构
#[derive(Clone, Serialize, Deserialize)]
struct Tree {
    /// 树的所有节点
    nodes: Vec<Node>,
    /// 根节点索引
    root: usize,
}

impl ANNOYIndex {
    /// Convert String ID to u64 using hash
    fn string_id_to_u64(id: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        id.hash(&mut hasher);
        hasher.finish()
    }
    
    /// 创建新的ANNOY索引
    pub fn new(config: IndexConfig) -> Result<Self> {
        // 验证配置
        if config.dimension == 0 {
            return Err(Error::vector("Dimension must be greater than 0"));
        }
        
        let mut annoy_trees = config.annoy_tree_count;
        if annoy_trees == 0 {
            // 设置默认树的数量，通常与数据集大小和期望精度有关
            let base_trees = 10;
            let scale_factor = (config.expected_elements as f32).log10();
            annoy_trees = (base_trees as f32 * scale_factor).round() as usize;
            annoy_trees = annoy_trees.max(10).min(100);
        }
        
        // Create distance calculator based on metric
        let distance: Box<dyn Distance + Send + Sync> = match config.metric {
            SimilarityMetric::Euclidean => Box::new(super::distance::EuclideanDistance),
            SimilarityMetric::Cosine => Box::new(super::distance::CosineDistance),
            SimilarityMetric::DotProduct => Box::new(super::distance::DotProductDistance),
            SimilarityMetric::Manhattan => Box::new(super::distance::ManhattanDistance),
            SimilarityMetric::Jaccard => Box::new(super::distance::JaccardDistance),
            _ => Box::new(super::distance::EuclideanDistance),
        };
        
        Ok(Self {
            config,
            distance,
            forest: Vec::new(),
            vectors: HashMap::new(),
            vector_metadata: HashMap::new(),
            vector_count: 0,
            is_built: false,
        })
    }
    
    /// 构建索引
    fn build(&mut self) -> Result<()> {
        if self.is_built {
            return Ok(());
        }
        
        // 确保有足够的向量
        if self.vector_count < 2 {
            return Err(Error::vector(format!(
                "Not enough vectors ({}) to build the index", self.vector_count
            )));
        }
        
        // 获取所有向量ID
        let vector_ids: Vec<u64> = self.vectors.keys().cloned().collect();
        
        // 构建多棵树
        self.forest = Vec::with_capacity(self.config.annoy_tree_count);
        
        for _ in 0..self.config.annoy_tree_count {
            // 为每棵树构建一个随机分割平面的树
            let tree = self.build_tree(vector_ids.clone())?;
            self.forest.push(tree);
        }
        
        self.is_built = true;
        Ok(())
    }
    
    /// 构建单棵树
    fn build_tree(&self, vector_ids: Vec<u64>) -> Result<Tree> {
        let mut nodes = Vec::new();
        
        // 递归构建树，返回根节点索引
        let root = self.build_tree_recursive(&mut nodes, vector_ids)?;
        
        Ok(Tree { nodes, root })
    }
    
    /// 递归构建树
    fn build_tree_recursive(&self, nodes: &mut Vec<Node>, vector_ids: Vec<u64>) -> Result<usize> {
        // 如果向量数量小于等于叶子大小阈值，创建叶子节点
        // Use default leaf size if not specified
        let leaf_size = 10; // Default leaf size
        if vector_ids.len() <= leaf_size || vector_ids.len() <= 1 {
            let node_index = nodes.len();
            nodes.push(Node::Leaf { vector_ids });
            return Ok(node_index);
        }
        
        // 选择两个随机向量，确定分割平面
        let (normal, threshold) = self.choose_splitting_plane(&vector_ids)?;
        
        // 根据分割平面将向量分为左右两部分
        let (left_ids, right_ids) = self.split_vectors(&vector_ids, &normal, threshold)?;
        
        // 处理特殊情况：如果分割后一侧为空，创建叶子节点
        if left_ids.is_empty() || right_ids.is_empty() {
            let node_index = nodes.len();
            nodes.push(Node::Leaf { vector_ids });
            return Ok(node_index);
        }
        
        // 创建当前节点
        let node_index = nodes.len();
        
        // 创建叶子节点（初始为空，将在递归构建过程中填充）
        nodes.push(Node::Leaf { vector_ids: Vec::new() });
        
        // 递归构建左右子树
        let left = self.build_tree_recursive(nodes, left_ids)?;
        let right = self.build_tree_recursive(nodes, right_ids)?;
        
        // 更新当前节点
        nodes[node_index] = Node::Internal {
            left,
            right,
            normal,
            threshold,
        };
        
        Ok(node_index)
    }
    
    /// 选择分割平面
    fn choose_splitting_plane(&self, vector_ids: &[u64]) -> Result<(Vec<f32>, f32)> {
        let mut rng = thread_rng();
        
        // 随机选择两个不同的向量
        if vector_ids.len() < 2 {
            return Err(Error::vector("Need at least 2 vectors to choose a splitting plane"));
        }
        
        let indices: Vec<usize> = (0..vector_ids.len()).collect();
        let mut selected_indices = indices;
        selected_indices.shuffle(&mut rng);
        
        let a_id = vector_ids[selected_indices[0]];
        let b_id = vector_ids[selected_indices[1]];
        
        // 获取向量数据
        let a = self.vectors.get(&a_id).ok_or_else(|| 
            Error::vector(format!("Vector with ID {} not found", a_id)))?;
        let b = self.vectors.get(&b_id).ok_or_else(|| 
            Error::vector(format!("Vector with ID {} not found", b_id)))?;
        
        // 计算法向量 (a-b)
        let mut normal = Vec::with_capacity(self.config.dimension);
        for i in 0..self.config.dimension {
            normal.push(a[i] - b[i]);
        }
        
        // 规范化法向量
        let norm = (normal.iter().map(|&x| x*x).sum::<f32>()).sqrt();
        if norm > 1e-10 {
            for i in 0..normal.len() {
                normal[i] /= norm;
            }
        } else {
            // 如果两点太近，生成随机分割平面
            for i in 0..normal.len() {
                normal[i] = rng.gen_range(-1.0..1.0);
            }
            let random_norm = (normal.iter().map(|&x| x*x).sum::<f32>()).sqrt();
            for i in 0..normal.len() {
                normal[i] /= random_norm;
            }
        }
        
        // 计算所有向量在法向量上的投影，取中位数作为阈值
        let mut projections = Vec::with_capacity(vector_ids.len());
        
        for &id in vector_ids {
            if let Some(vector) = self.vectors.get(&id) {
                let projection = vector.iter().zip(normal.iter()).map(|(&v, &n)| v * n).sum::<f32>();
                projections.push(projection);
            }
        }
        
        projections.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        // 取中位数作为分割阈值
        let threshold = if !projections.is_empty() {
            let mid = projections.len() / 2;
            if projections.len() % 2 == 0 {
                (projections[mid - 1] + projections[mid]) / 2.0
            } else {
                projections[mid]
            }
        } else {
            0.0
        };
        
        Ok((normal, threshold))
    }
    
    /// 根据分割平面将向量分为左右两部分
    fn split_vectors(&self, vector_ids: &[u64], normal: &[f32], threshold: f32) -> Result<(Vec<u64>, Vec<u64>)> {
        let mut left_ids = Vec::new();
        let mut right_ids = Vec::new();
        
        for &id in vector_ids {
            if let Some(vector) = self.vectors.get(&id) {
                let projection = vector.iter().zip(normal.iter()).map(|(&v, &n)| v * n).sum::<f32>();
                
                if projection <= threshold {
                    left_ids.push(id);
                } else {
                    right_ids.push(id);
                }
            }
        }
        
        Ok((left_ids, right_ids))
    }
    
    /// 在树中查找
    fn search_tree(&self, tree: &Tree, query: &[f32], result_set: &mut HashSet<u64>, n_nodes: &mut usize) -> Result<()> {
        // 从根节点开始搜索
        let mut node_index = tree.root;
        let mut stack = Vec::new();
        
        // 递归搜索树，收集叶子节点中的向量
        loop {
            *n_nodes += 1;
            
            match &tree.nodes[node_index] {
                Node::Leaf { vector_ids } => {
                    // 将叶子节点中的所有向量ID添加到结果集
                    for &id in vector_ids {
                        result_set.insert(id);
                    }
                    
                    // 如果栈为空，搜索完成
                    if stack.is_empty() {
                        break;
                    }
                    
                    // 否则，从栈中弹出下一个节点继续搜索
                    node_index = stack.pop().unwrap();
                },
                Node::Internal { left, right, normal, threshold } => {
                    // 计算查询向量在分割平面上的投影
                    let projection = query.iter().zip(normal.iter()).map(|(&q, &n)| q * n).sum::<f32>();
                    
                    // 确定优先搜索的方向
                    let (priority_node, alternate_node) = if projection <= *threshold {
                        (*left, *right)
                    } else {
                        (*right, *left)
                    };
                    
                    // 将备选方向压入栈
                    stack.push(alternate_node);
                    
                    // 继续沿优先方向搜索
                    node_index = priority_node;
                }
            }
        }
        
        Ok(())
    }
    
    /// 在森林中搜索
    fn search_forest(&self, query: &[f32], top_k: usize, search_k: usize) -> Result<Vec<SearchResult>> {
        // 初始化结果集和优先队列
        let mut candidate_set = HashSet::new();
        let mut n_nodes_visited = 0;
        
        // 在每棵树中搜索
        for tree in &self.forest {
            self.search_tree(tree, query, &mut candidate_set, &mut n_nodes_visited)?;
        }
        
        // 如果search_k小于candidate_set的大小，随机选择一部分
        let mut candidates: Vec<u64> = candidate_set.into_iter().collect();
        if search_k > 0 && search_k < candidates.len() {
            candidates.shuffle(&mut thread_rng());
            candidates.truncate(search_k);
        }
        
        // 计算距离并排序
        let mut results = Vec::new();
        for id in candidates {
            if let Some(vector) = self.vectors.get(&id) {
                let distance = compute_distance_raw(query, vector, &self.distance);
                let metadata = self.vector_metadata.get(&id).map(|m| {
                    let mut props = serde_json::Map::new();
                    for (k, v) in m {
                        props.insert(k.clone(), serde_json::Value::String(v.clone()));
                    }
                    serde_json::Value::Object(props)
                });
                
                // Convert u64 ID to String (using hex representation)
                results.push(SearchResult {
                    id: format!("{:x}", id),
                    distance,
                    metadata,
                });
            }
        }
        
        // 根据相似度度量排序
        match self.config.metric {
            SimilarityMetric::Cosine | SimilarityMetric::DotProduct => {
                // 对于余弦和内积，距离越大越好（相似度越高）
                results.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap_or(std::cmp::Ordering::Equal));
            }
            _ => {
                // 对于欧氏距离等，距离越小越好
                results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
        
        // 截取top_k个结果
        results.truncate(top_k);
        
        Ok(results)
    }
}

impl VectorIndex for ANNOYIndex {
    /// 添加向量到索引
    fn add(&mut self, vector: Vector) -> Result<()> {
        // 如果索引已构建，不能添加新向量（需要重建）
        if self.is_built {
            return Err(Error::vector("Cannot add vectors after index is built. Rebuild the index."));
        }
        
        // 验证向量维度
        if vector.data.len() != self.config.dimension {
            return Err(Error::vector(format!(
                "Vector dimension ({}) does not match index dimension ({})",
                vector.data.len(), self.config.dimension
            )));
        }
        
        // Convert String ID to u64
        let vector_id = Self::string_id_to_u64(&vector.id);
        
        // 规范化向量（对于余弦距离，需要规范化）
        let normalized_data = if matches!(self.config.metric, SimilarityMetric::Cosine) {
            let mut data = vector.data.clone();
            normalize_vector(&mut data)?;
            data
        } else {
            vector.data.clone()
        };
        
        // 存储向量
        self.vectors.insert(vector_id, normalized_data);
        
        // 存储元数据
        if let Some(metadata) = &vector.metadata {
            // Convert VectorMetadata to HashMap<String, String>
            let mut metadata_map = HashMap::new();
            for (k, v) in &metadata.properties {
                if let Some(s) = v.as_str() {
                    metadata_map.insert(k.clone(), s.to_string());
                } else {
                    metadata_map.insert(k.clone(), v.to_string());
                }
            }
            self.vector_metadata.insert(vector_id, metadata_map);
        }
        
        self.vector_count += 1;
        
        // 判断是否需要自动构建索引（当向量数量达到预期元素数量时）
        if !self.is_built && self.vector_count >= self.config.expected_elements.max(100) {
            self.build()?;
        }
        
        Ok(())
    }
    
    /// 在索引中搜索最近的向量
    fn search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        if !self.is_built {
            return Err(Error::vector("Index has not been built yet"));
        }
        
        if query.len() != self.config.dimension {
            return Err(Error::vector(format!(
                "Query dimension ({}) does not match index dimension ({})",
                query.len(), self.config.dimension
            )));
        }
        
        // 规范化查询向量（对于余弦距离，需要规范化）
        let normalized_query = if matches!(self.config.metric, SimilarityMetric::Cosine) {
            let mut q = query.to_vec();
            normalize_vector(&mut q)?;
            q
        } else {
            query.to_vec()
        };
        
        // 使用默认的search_k参数
        let search_k = limit * self.config.annoy_tree_count;
        
        // 在森林中搜索
        self.search_forest(&normalized_query, limit, search_k)
    }

    /// 从索引中删除向量
    fn delete(&mut self, id: &str) -> Result<bool> {
        let vector_id = id.parse::<u64>()
            .map_err(|_| Error::vector("Invalid vector ID format"))?;
            
        // ANNOY索引不支持高效的单向量删除，需要重建索引
        if self.is_built {
            return Err(Error::vector("Cannot remove vectors from built ANNOY index. Rebuild the index."));
        }
        
        // 从存储中删除向量
        if self.vectors.remove(&vector_id).is_some() {
            // 删除元数据
            self.vector_metadata.remove(&vector_id);
            self.vector_count -= 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// 检查索引中是否包含指定ID的向量
    fn contains(&self, id: &str) -> bool {
        if let Ok(vector_id) = id.parse::<u64>() {
            self.vectors.contains_key(&vector_id)
        } else {
            false
        }
    }

    /// 获取索引中的向量数量
    fn size(&self) -> usize {
        self.vector_count
    }

    /// 返回索引向量的维度
    fn dimension(&self) -> usize {
        self.config.dimension
    }

    /// 获取索引配置
    fn get_config(&self) -> IndexConfig {
        self.config.clone()
    }

    /// 序列化索引
    fn serialize(&self) -> Result<Vec<u8>> {
        // Create a serializable version without the distance field
        #[derive(Serialize)]
        struct SerializableANNOYIndex {
            config: IndexConfig,
            forest: Vec<Tree>,
            vectors: HashMap<u64, Vec<f32>>,
            vector_metadata: HashMap<u64, HashMap<String, String>>,
            vector_count: usize,
            is_built: bool,
        }
        
        let serializable = SerializableANNOYIndex {
            config: self.config.clone(),
            forest: self.forest.clone(),
            vectors: self.vectors.clone(),
            vector_metadata: self.vector_metadata.clone(),
            vector_count: self.vector_count,
            is_built: self.is_built,
        };
        
        bincode::serialize(&serializable)
            .map_err(|e| Error::vector(format!("Failed to serialize index: {}", e)))
    }
    
    /// 反序列化索引
    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        // Deserialize without the distance field
        #[derive(Deserialize)]
        struct DeserializableANNOYIndex {
            config: IndexConfig,
            forest: Vec<Tree>,
            vectors: HashMap<u64, Vec<f32>>,
            vector_metadata: HashMap<u64, HashMap<String, String>>,
            vector_count: usize,
            is_built: bool,
        }
        
        let deserialized: DeserializableANNOYIndex = bincode::deserialize(data)
            .map_err(|e| Error::vector(format!("Failed to deserialize index: {}", e)))?;
        
        // Recreate distance calculator based on config.metric
        let distance: Box<dyn Distance + Send + Sync> = match deserialized.config.metric {
            SimilarityMetric::Euclidean => Box::new(super::distance::EuclideanDistance),
            SimilarityMetric::Cosine => Box::new(super::distance::CosineDistance),
            SimilarityMetric::DotProduct => Box::new(super::distance::DotProductDistance),
            SimilarityMetric::Manhattan => Box::new(super::distance::ManhattanDistance),
            SimilarityMetric::Jaccard => Box::new(super::distance::JaccardDistance),
            _ => Box::new(super::distance::EuclideanDistance),
        };
        
        self.config = deserialized.config;
        self.distance = distance;
        self.forest = deserialized.forest;
        self.vectors = deserialized.vectors;
        self.vector_metadata = deserialized.vector_metadata;
        self.vector_count = deserialized.vector_count;
        self.is_built = deserialized.is_built;
        
        Ok(())
    }

    /// 创建索引的深拷贝并装箱
    fn clone_box(&self) -> Box<dyn VectorIndex + Send + Sync> {
        Box::new(self.clone())
    }

    /// 从字节数组创建索引并装箱
    fn deserialize_box(data: &[u8]) -> Result<Box<dyn VectorIndex + Send + Sync>> where Self: Sized {
        // Deserialize without the distance field
        #[derive(Deserialize)]
        struct DeserializableANNOYIndex {
            config: IndexConfig,
            forest: Vec<Tree>,
            vectors: HashMap<u64, Vec<f32>>,
            vector_metadata: HashMap<u64, HashMap<String, String>>,
            vector_count: usize,
            is_built: bool,
        }
        
        let deserialized: DeserializableANNOYIndex = bincode::deserialize(data)
            .map_err(|e| Error::vector(format!("Failed to deserialize index: {}", e)))?;
        
        // Recreate distance calculator based on config.metric
        let distance: Box<dyn Distance + Send + Sync> = match deserialized.config.metric {
            SimilarityMetric::Euclidean => Box::new(super::distance::EuclideanDistance),
            SimilarityMetric::Cosine => Box::new(super::distance::CosineDistance),
            SimilarityMetric::DotProduct => Box::new(super::distance::DotProductDistance),
            SimilarityMetric::Manhattan => Box::new(super::distance::ManhattanDistance),
            SimilarityMetric::Jaccard => Box::new(super::distance::JaccardDistance),
            _ => Box::new(super::distance::EuclideanDistance),
        };
        
        let index = ANNOYIndex {
            config: deserialized.config,
            distance,
            forest: deserialized.forest,
            vectors: deserialized.vectors,
            vector_metadata: deserialized.vector_metadata,
            vector_count: deserialized.vector_count,
            is_built: deserialized.is_built,
        };
        
        Ok(Box::new(index))
    }

    /// 清空索引
    fn clear(&mut self) -> Result<()> {
        self.forest.clear();
        self.vectors.clear();
        self.vector_metadata.clear();
        self.vector_count = 0;
        self.is_built = false;
        
        Ok(())
    }

    /// 从索引中删除向量（按ID）
    fn remove(&mut self, vector_id: u64) -> Result<()> {
        // ANNOY索引不支持高效的单向量删除，需要重建索引
        if self.is_built {
            return Err(Error::vector("Cannot remove vectors from built ANNOY index. Rebuild the index."));
        }
        
        // 从存储中删除向量
        if self.vectors.remove(&vector_id).is_some() {
            // 删除元数据
            self.vector_metadata.remove(&vector_id);
            self.vector_count -= 1;
            Ok(())
        } else {
            Err(Error::vector(format!("Vector with ID {} not found", vector_id)))
        }
    }

    /// 获取所有向量（用于线性搜索和调试）
    fn get_all_vectors(&self) -> Result<Vec<crate::vector::index::interfaces::VectorData>> {
        let mut all_vectors = Vec::new();
        
        for (vector_id, vector_data) in &self.vectors {
            let metadata = self.vector_metadata.get(vector_id).cloned();
            
            all_vectors.push(crate::vector::index::interfaces::VectorData {
                id: *vector_id,
                vector: vector_data.clone(),
                metadata,
            });
        }
        
        Ok(all_vectors)
    }
}

impl Clone for ANNOYIndex {
    fn clone(&self) -> Self {
        // Recreate distance calculator based on config.metric
        let distance: Box<dyn Distance + Send + Sync> = match self.config.metric {
            SimilarityMetric::Euclidean => Box::new(super::distance::EuclideanDistance),
            SimilarityMetric::Cosine => Box::new(super::distance::CosineDistance),
            SimilarityMetric::DotProduct => Box::new(super::distance::DotProductDistance),
            SimilarityMetric::Manhattan => Box::new(super::distance::ManhattanDistance),
            SimilarityMetric::Jaccard => Box::new(super::distance::JaccardDistance),
            _ => Box::new(super::distance::EuclideanDistance),
        };
        
        Self {
            config: self.config.clone(),
            distance,
            forest: self.forest.clone(),
            vectors: self.vectors.clone(),
            vector_metadata: self.vector_metadata.clone(),
            vector_count: self.vector_count,
            is_built: self.is_built,
        }
    }
} 