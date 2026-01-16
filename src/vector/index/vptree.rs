/// VP-Tree (Vantage Point Tree) 索引实现
/// 
/// VP-Tree 是一种基于空间分割的树形数据结构,用于高效地进行最近邻搜索。
/// 每个节点都选择一个支点(vantage point),并根据到该支点的距离将其他点分为内部和外部两部分。

use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};

use crate::{Error, Result, vector::Vector};
// use crate::vector::ops::VectorOps;
use crate::vector::operations::SimilarityMetric;
use crate::vector::index::interfaces::VectorIndex;
use crate::vector::index::types::{IndexConfig, SearchResult};
use crate::vector::distance::{Distance, EuclideanDistance, CosineDistance, DotProductDistance, ManhattanDistance};
use crate::vector::index::distance::JaccardDistance;

/// VP树节点结构
/// 
/// 每个节点包含:
/// - 一个支点向量及其相关信息
/// - 一个半径值,用于分割空间
/// - 左右子树,分别包含距离小于等于半径和大于半径的点
#[derive(Debug, Clone, Serialize, Deserialize)]
struct VPTreeNode {
    /// 向量的唯一标识符
    id: String,
    /// 向量数据
    vector: Vec<f32>,
    /// 可选的元数据
    metadata: Option<serde_json::Value>,
    /// 分割半径,用于确定点是在内部还是外部
    radius: f32,
    /// 内部子树(距离 <= radius 的点)
    left: Option<Box<VPTreeNode>>,
    /// 外部子树(距离 > radius 的点)
    right: Option<Box<VPTreeNode>>,
}

/// VP-Tree索引实现
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VPTreeIndex {
    /// 树的根节点
    root: Option<Box<VPTreeNode>>,
    /// 存储所有向量的哈希表,用于快速访问
    vectors: HashMap<String, Vector>,
    /// 索引配置,包含维度和距离度量方式
    config: IndexConfig,
    /// 向量维度
    dimensions: usize,
}

impl VPTreeIndex {
    /// 创建新的VP-Tree索引
    pub fn new(config: IndexConfig) -> Result<Self> {
        // 检查配置中的度量方式是否支持
        if config.metric == SimilarityMetric::Jaccard {
            return Err(Error::vector("Jaccard similarity not supported for VPTree".to_string()));
        }
        
        let dimensions = config.dimension;
        
        Ok(Self {
            root: None,
            vectors: HashMap::new(),
            config,
            dimensions,
        })
    }
    
    /// 将 Vector 的元数据转换为 serde_json::Value
    fn convert_metadata(vector: &Vector) -> Option<serde_json::Value> {
        vector.metadata.as_ref().map(|m| {
            let mut map = serde_json::Map::new();
            for (k, v) in &m.properties {
                map.insert(k.clone(), v.clone());
            }
            serde_json::Value::Object(map)
        })
    }
    
    /// 获取当前配置的距离度量方式
    fn get_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.metric {
            SimilarityMetric::Euclidean => EuclideanDistance.calculate(a, b),
            SimilarityMetric::Cosine => CosineDistance.calculate(a, b),
            SimilarityMetric::DotProduct => DotProductDistance.calculate(a, b),
            SimilarityMetric::Manhattan => ManhattanDistance.calculate(a, b),
            SimilarityMetric::Jaccard => {
                use crate::vector::index::distance::Distance;
                JaccardDistance.calculate(a, b)
            },
            _ => EuclideanDistance.calculate(a, b),
        }
    }
    
    /// 选择支点(pivot)
    /// 
    /// 实现多种支点选择策略:
    /// 1. 固定选择: 始终选择第一个向量
    /// 2. 随机选择: 随机选择一个向量
    /// 3. 采样选择: 从随机采样的向量中选择最优支点
    /// 
    /// 最优支点定义为: 到其他采样点的距离方差最大的点
    /// 这有助于创建更平衡的树结构
    fn select_pivot(&self, vectors: &[Vector]) -> Result<usize> {
        // 如果向量数量很少，直接返回第一个
        if vectors.len() <= 2 {
            return Ok(0);
        }
        
        // 根据配置选择不同的策略
        // Note: IndexConfig doesn't have metadata field, use default strategy
        let pivot_strategy = "random"; // Default strategy
        match pivot_strategy {
            // 随机选择策略
            "random" => {
                use rand::{thread_rng, Rng};
                let mut rng = thread_rng();
                Ok(rng.gen_range(0..vectors.len()))
            },
            
            // 采样选择策略 - 选择方差最大的点
            "sample" => {
                use rand::{thread_rng, seq::SliceRandom};
                let mut rng = thread_rng();
                
                // 确定采样数量，最多30个点
                let sample_size = std::cmp::min(30, vectors.len());
                
                // 随机采样索引
                let mut indices: Vec<usize> = (0..vectors.len()).collect();
                indices.shuffle(&mut rng);
                let sample_indices = &indices[0..sample_size];
                
                // 为每个采样点计算到其他点的距离方差
                let mut best_idx = 0;
                let mut max_variance = f32::NEG_INFINITY;
                
                for &idx in sample_indices {
                    let mut distances = Vec::with_capacity(sample_size - 1);
                    
                    // 计算到其他采样点的距离
                    for &other_idx in sample_indices {
                        if idx != other_idx {
                            let distance = self.get_distance(
                                &vectors[idx].data,
                                &vectors[other_idx].data
                            );
                            distances.push(distance);
                        }
                    }
                    
                    // 计算距离的方差
                    if !distances.is_empty() {
                        let mean = distances.iter().sum::<f32>() / distances.len() as f32;
                        let variance = distances.iter()
                            .map(|&d| (d - mean).powi(2))
                            .sum::<f32>() / distances.len() as f32;
                            
                        // 更新最大方差
                        if variance > max_variance {
                            max_variance = variance;
                            best_idx = idx;
                        }
                    }
                }
                
                Ok(best_idx)
            },
            
            // 默认策略: 使用第一个点
            _ => Ok(0),
        }
    }
    
    /// 构建VP-Tree
    fn build_tree(&mut self, vectors: Vec<Vector>) -> Result<Option<Box<VPTreeNode>>> {
        if vectors.is_empty() {
            return Ok(None);
        }
        
        // 选择一个向量作为支点（pivot）
        // 使用改进的支点选择策略
        let pivot_idx = self.select_pivot(&vectors)?;
        let pivot = &vectors[pivot_idx];
        
        // 如果只有一个向量，直接返回叶节点
        if vectors.len() == 1 {
            return Ok(Some(Box::new(VPTreeNode {
                id: pivot.id.clone(),
                vector: pivot.data.clone(),
                metadata: Self::convert_metadata(pivot),
                radius: 0.0,
                left: None,
                right: None,
            })));
        }
        
        // 计算其他向量到支点的距离
        let mut distances = Vec::with_capacity(vectors.len() - 1);
        for (i, vector) in vectors.iter().enumerate() {
            if i == pivot_idx {
                continue;
            }
            
            let distance = self.get_distance(&pivot.data, &vector.data);
            distances.push((i, distance));
        }
        
        // 按距离排序
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        
        // 确定中位数作为半径
        let median_idx = distances.len() / 2;
        let radius = distances[median_idx].1;
        
        // 预分配内部和外部向量的容量
        let mut inner_vectors = Vec::with_capacity(median_idx + 1);
        let mut outer_vectors = Vec::with_capacity(distances.len() - median_idx);
        
        // 分离内部和外部向量
        for (i, distance) in distances {
            let vector = vectors[i].clone();
            if distance <= radius {
                inner_vectors.push(vector);
            } else {
                outer_vectors.push(vector);
            }
        }
        
        // 递归构建左右子树
        let left = self.build_tree(inner_vectors)?;
        let right = self.build_tree(outer_vectors)?;
        
        // 构建节点
        Ok(Some(Box::new(VPTreeNode {
            id: pivot.id.clone(),
            vector: pivot.data.clone(),
            metadata: Self::convert_metadata(pivot),
            radius,
            left,
            right,
        })))
    }
    
    /// 在树中搜索最近的K个向量
    /// 
    /// # 参数
    /// - `node`: 当前搜索的树节点
    /// - `query`: 查询向量,维度必须与索引维度相同
    /// - `k`: 返回的最近邻数量
    /// - `results`: 用于存储结果的最小堆
    /// 
    /// # 返回
    /// - `Ok(())`: 搜索成功
    /// - `Err(Error)`: 发生错误,例如维度不匹配或距离计算失败
    fn search_recursive(&self, node: &VPTreeNode, query: &[f32], k: usize, results: &mut BinaryHeap<SearchResult>) -> Result<()> {
        // 计算查询向量到当前节点的距离
        let distance = self.get_distance(query, &node.vector);
        
        // 将结果添加到结果集
        results.push(SearchResult {
            id: node.id.clone(),
            distance,
            metadata: node.metadata.clone(),
        });
        
        if results.len() > k {
            results.pop();
        }
        
        // 确定搜索顺序（先内部还是先外部）
        let (first, second) = if distance <= node.radius {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };
        
        // 先搜索更可能包含结果的子树
        if let Some(first_node) = first {
            self.search_recursive(first_node, query, k, results)?;
        }
        
        // 检查是否需要搜索另一个子树
        let min_distance = if results.len() == k {
            results.peek().unwrap().distance
        } else {
            0.0
        };
        
        // 如果到边界的距离比最差结果好，可能在另一半有更好的结果
        if (distance - node.radius).abs() < min_distance {
            if let Some(second_node) = second {
                self.search_recursive(second_node, query, k, results)?;
            }
        }
        
        Ok(())
    }
    
    /// 序列化索引为字节数组
    /// 
    /// 使用 bincode 将整个索引序列化为字节数组,包括:
    /// - 所有节点的结构和数据
    /// - 向量集合
    /// - 配置信息
    /// 
    /// # 返回
    /// - `Ok(Vec<u8>)`: 序列化后的字节数组
    /// - `Err(Error)`: 序列化失败的错误
    pub fn serialize(&self) -> Result<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| Error::serialization(e.to_string()))
    }
    
    /// 从字节数组反序列化索引
    /// 
    /// # 参数
    /// - `data`: 包含序列化索引数据的字节数组
    /// 
    /// # 返回
    /// - `Ok(Self)`: 反序列化成功的索引实例
    /// - `Err(Error)`: 反序列化失败的错误,可能的原因:
    ///   - 数据格式不正确
    ///   - 版本不兼容
    ///   - 数据损坏
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data)
            .map_err(|e| Error::serialization(e.to_string()))
    }
}

impl VectorIndex for VPTreeIndex {
    fn add(&mut self, vector: Vector) -> Result<()> {
        if vector.data.len() != self.config.dimension {
            return Err(Error::vector(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.config.dimension, vector.data.len()
            )));
        }
        
        // 存储向量
        self.vectors.insert(vector.id.clone(), vector.clone());
        
        // 重建树
        let vectors: Vec<Vector> = self.vectors.values().cloned().collect();
        self.root = self.build_tree(vectors)?;
        
        Ok(())
    }

    /// 搜索最近的K个向量
    /// 
    /// # 参数
    /// - `query`: 查询向量,维度必须与索引维度相同
    /// - `k`: 返回的最近邻数量
    /// 
    /// # 返回
    /// - `Ok(Vec<SearchResult>)`: 按距离升序排序的K个最近邻结果
    /// - `Err(Error)`: 发生错误,例如维度不匹配或距离计算失败
    /// 
    /// # 示例
    /// ```
    /// use vecmind::vector::index::{VectorIndex, VPTreeIndex};
    /// use vecmind::vector::types::IndexConfig;
    /// 
    /// let index = VPTreeIndex::new(IndexConfig::default());
    /// let query = vec![1.0, 2.0, 3.0];
    /// let k = 5;
    /// let results = index.search(&query, k).unwrap();
    /// assert!(results.len() <= k);
    /// ```
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        // 创建最小堆来跟踪K个最近的向量
        let mut results = BinaryHeap::new();
        
        // 搜索最近的K个向量
        if let Some(root) = &self.root {
            self.search_recursive(root, query, k, &mut results)?;
        }
        
        // 将堆中的结果转换为有序的结果列表
        let mut result_list = Vec::with_capacity(results.len());
        for result in results.into_iter().rev() {
            result_list.push(result);
        }
        
        Ok(result_list)
    }
    
    fn delete(&mut self, id: &str) -> Result<bool> {
        // 从向量集合中删除
        if self.vectors.remove(id).is_none() {
            return Ok(false);
        }
        
        // 如果向量集合为空,直接设置 root 为 None
        if self.vectors.is_empty() {
            self.root = None;
            return Ok(true);
        }
        
        // 重建树
        let vectors: Vec<Vector> = self.vectors.values().cloned().collect();
        self.root = self.build_tree(vectors)?;
        
        Ok(true)
    }
    
    fn contains(&self, id: &str) -> bool {
        self.vectors.contains_key(id)
    }
    
    fn dimension(&self) -> usize {
        self.dimensions
    }
    
    fn get_config(&self) -> IndexConfig {
        self.config.clone()
    }
    
    fn size(&self) -> usize {
        self.vectors.len()
    }
    
    /// 序列化索引为字节数组
    /// 
    /// 这是 VectorIndex trait 要求的方法,内部调用 VPTreeIndex::serialize
    fn serialize(&self) -> Result<Vec<u8>> {
        VPTreeIndex::serialize(self)
    }
    
    /// 从字节数组更新索引
    /// 
    /// # 参数
    /// - `data`: 包含序列化索引数据的字节数组
    /// 
    /// # 返回
    /// - `Ok(())`: 更新成功
    /// - `Err(Error)`: 反序列化失败
    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let new_index = VPTreeIndex::deserialize(data)?;
        *self = new_index;
        Ok(())
    }
    
    /// 创建索引的克隆作为 trait 对象
    /// 
    /// 这个方法用于在需要 trait 对象时创建索引的克隆
    fn clone_box(&self) -> Box<dyn VectorIndex + Send + Sync> {
        Box::new(self.clone())
    }
    
    /// 从字节数组创建索引的 trait 对象
    /// 
    /// 这个方法用于直接从序列化数据创建 trait 对象,
    /// 避免了先创建具体类型再转换为 trait 对象的步骤
    /// 
    /// # 参数
    /// - `data`: 包含序列化索引数据的字节数组
    /// 
    /// # 返回
    /// - `Ok(Box<dyn VectorIndex>)`: 反序列化成功的索引 trait 对象
    /// - `Err(Error)`: 反序列化失败
    fn deserialize_box(data: &[u8]) -> Result<Box<dyn VectorIndex + Send + Sync>> {
        let index = VPTreeIndex::deserialize(data)?;
        Ok(Box::new(index))
    }
} 