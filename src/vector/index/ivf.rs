// IVF索引实现
// 倒排文件（Inverted File）索引实现

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use crate::{Error, Result, vector::Vector};
use super::interfaces::VectorIndex;
use super::types::SearchResult;
use super::distance::Distance;
use super::distance::ManhattanDistance;
use crate::vector::index::distance::JaccardDistance;

/// 倒排文件索引的簇
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IVFCluster {
    centroid: Vec<f32>,
    vectors: Vec<(String, Vec<f32>)>,
}

/// 倒排文件索引
pub struct IVFIndex {
    clusters: HashMap<usize, IVFCluster>,
    config: super::types::IndexConfig,
    vector_to_cluster: HashMap<String, usize>,
    total_vectors: usize,
    distance: Box<dyn Distance + Send + Sync>,
}

impl IVFIndex {
    /// 创建新的IVF索引
    pub fn new(config: super::types::IndexConfig) -> Result<Self> {
        let distance: Box<dyn Distance + Send + Sync> = match config.metric {
            crate::vector::operations::SimilarityMetric::Euclidean => Box::new(super::distance::EuclideanDistance),
            crate::vector::operations::SimilarityMetric::Cosine => Box::new(super::distance::CosineDistance),
            crate::vector::operations::SimilarityMetric::DotProduct => Box::new(super::distance::DotProductDistance),
            crate::vector::operations::SimilarityMetric::Manhattan => Box::new(ManhattanDistance),
            crate::vector::operations::SimilarityMetric::Jaccard => Box::new(JaccardDistance),
            _ => Box::new(super::distance::EuclideanDistance),
        };
        
        Ok(Self {
            clusters: HashMap::new(),
            config,
            vector_to_cluster: HashMap::new(),
            total_vectors: 0,
            distance,
        })
    }
    
    /// 查找最近的聚类中心
    pub fn find_nearest_center(&self, vector: &[f32]) -> Result<usize> {
        if self.clusters.is_empty() {
            return Ok(0);
        }
        
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;
        
        for (&idx, cluster) in &self.clusters {
            // 计算与中心点的距离
            let dist = self.calculate_distance(vector, &cluster.centroid);
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }
        
        Ok(best_idx)
    }
    
    /// 计算两个向量之间的距离
    fn calculate_distance(&self, v1: &[f32], v2: &[f32]) -> f32 {
        self.distance.calculate(v1, v2)
    }
}

// 在这里实现VectorIndex trait，具体实现内容略（根据需要添加）

// 实现VectorIndex trait
impl VectorIndex for IVFIndex {
    fn add(&mut self, vector: Vector) -> Result<()> {
        let id = &vector.id;
        let vector_data = &vector.data;
        
        // 寻找最近的中心点，或创建新的中心点
        let center_id = if self.clusters.is_empty() {
            // 如果没有簇，创建第一个
            let new_id = 0;
            let cluster = IVFCluster {
                centroid: vector_data.clone(),
                vectors: vec![(id.clone(), vector_data.clone())],
            };
            self.clusters.insert(new_id, cluster);
            new_id
        } else {
            // 找到最近的簇
            self.find_nearest_center(vector_data)?
        };
        
        // 将向量添加到对应的簇
        if let Some(cluster) = self.clusters.get_mut(&center_id) {
            cluster.vectors.push((id.clone(), vector_data.clone()));
        } else {
            // 创建新的簇
            let cluster = IVFCluster {
                centroid: vector_data.clone(),
                vectors: vec![(id.clone(), vector_data.clone())],
            };
            self.clusters.insert(center_id, cluster);
        }
        
        // 记录向量到簇的映射
        self.vector_to_cluster.insert(id.clone(), center_id);
        self.total_vectors += 1;
        
        Ok(())
    }
    
    fn search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        // 找到最近的簇
        let center_id = self.find_nearest_center(query)?;
        
        // 获取簇内的向量
        let cluster = match self.clusters.get(&center_id) {
            Some(c) => c,
            None => return Ok(vec![])
        };
        
        // 计算每个向量与查询向量的距离
        let mut results = Vec::new();
        for (vector_id, vector_data) in &cluster.vectors {
            let distance = self.calculate_distance(query, vector_data);
            results.push(SearchResult {
                id: vector_id.clone(),
                distance,
                metadata: None,
            });
        }
        
        // 排序并限制结果数量
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        if results.len() > limit {
            results.truncate(limit);
        }
        
        Ok(results)
    }
    
    fn delete(&mut self, id: &str) -> Result<bool> {
        // 获取向量所在的簇
        let cluster_id = match self.vector_to_cluster.remove(id) {
            Some(id) => id,
            None => return Ok(false)
        };
        
        // 从簇中删除向量
        if let Some(cluster) = self.clusters.get_mut(&cluster_id) {
            let index = cluster.vectors.iter().position(|(vid, _)| vid == id);
            if let Some(idx) = index {
                cluster.vectors.swap_remove(idx);
                self.total_vectors -= 1;
                return Ok(true);
            }
        }
        
        Ok(false)
    }
    
    fn contains(&self, id: &str) -> bool {
        self.vector_to_cluster.contains_key(id)
    }
    
    fn dimension(&self) -> usize {
        self.config.dimension
    }
    
    fn get_config(&self) -> super::types::IndexConfig {
        self.config.clone()
    }
    
    fn size(&self) -> usize {
        self.total_vectors
    }
    
    fn serialize(&self) -> Result<Vec<u8>> {
        // 创建可序列化的结构
        #[derive(Serialize)]
        struct SerializableIVFIndex {
            clusters: HashMap<usize, IVFCluster>,
            config: super::types::IndexConfig,
            vector_to_cluster: HashMap<String, usize>,
            total_vectors: usize,
        }
        
        let serializable = SerializableIVFIndex {
            clusters: self.clusters.clone(),
            config: self.config.clone(),
            vector_to_cluster: self.vector_to_cluster.clone(),
            total_vectors: self.total_vectors,
        };
        
        bincode::serialize(&serializable).map_err(|e| Error::serialization(e.to_string()))
    }
    
    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        // 反序列化
        #[derive(Deserialize)]
        struct SerializableIVFIndex {
            clusters: HashMap<usize, IVFCluster>,
            config: super::types::IndexConfig,
            vector_to_cluster: HashMap<String, usize>,
            total_vectors: usize,
        }
        
        let serialized: SerializableIVFIndex = bincode::deserialize(data)
            .map_err(|e| Error::deserialization(e.to_string()))?;
        
        // 更新现有实例
        self.clusters = serialized.clusters;
        self.vector_to_cluster = serialized.vector_to_cluster;
        self.total_vectors = serialized.total_vectors;
        self.config = serialized.config;
        
        // 重新创建距离函数
        self.distance = match self.config.metric {
            crate::vector::operations::SimilarityMetric::Cosine => Box::new(super::distance::CosineDistance) as Box<dyn Distance + Send + Sync>,
            crate::vector::operations::SimilarityMetric::Euclidean => Box::new(super::distance::EuclideanDistance) as Box<dyn Distance + Send + Sync>,
            crate::vector::operations::SimilarityMetric::DotProduct => Box::new(super::distance::DotProductDistance) as Box<dyn Distance + Send + Sync>,
            crate::vector::operations::SimilarityMetric::Manhattan => Box::new(super::distance::ManhattanDistance) as Box<dyn Distance + Send + Sync>,
            crate::vector::operations::SimilarityMetric::Jaccard => Box::new(super::distance::JaccardDistance) as Box<dyn Distance + Send + Sync>,
        };
        
        Ok(())
    }
    
    fn clone_box(&self) -> Box<dyn VectorIndex + Send + Sync> {
        Box::new(self.clone())
    }
    
    fn deserialize_box(data: &[u8]) -> Result<Box<dyn VectorIndex + Send + Sync>> {
        // 反序列化
        #[derive(Deserialize)]
        struct SerializableIVFIndex {
            clusters: HashMap<usize, IVFCluster>,
            config: super::types::IndexConfig,
            vector_to_cluster: HashMap<String, usize>,
            total_vectors: usize,
        }
        
        let serialized: SerializableIVFIndex = bincode::deserialize(data)
            .map_err(|e| Error::deserialization(e.to_string()))?;
        
        // 创建实例并重新创建不可序列化的成员
        let mut index = Self::new(serialized.config)?;
        index.clusters = serialized.clusters;
        index.vector_to_cluster = serialized.vector_to_cluster;
        index.total_vectors = serialized.total_vectors;
        
        Ok(Box::new(index))
    }
}

impl std::fmt::Debug for IVFIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IVFIndex")
            .field("clusters", &self.clusters)
            .field("config", &self.config)
            .field("vector_to_cluster", &self.vector_to_cluster)
            .field("total_vectors", &self.total_vectors)
            .field("distance", &format!("<Distance function>"))
            .finish()
    }
}

impl Clone for IVFIndex {
    fn clone(&self) -> Self {
        // 为克隆创建新的distance实例
        let distance: Box<dyn Distance + Send + Sync> = match self.config.metric {
            crate::vector::operations::SimilarityMetric::Euclidean => Box::new(super::distance::EuclideanDistance),
            crate::vector::operations::SimilarityMetric::Cosine => Box::new(super::distance::CosineDistance),
            crate::vector::operations::SimilarityMetric::DotProduct => Box::new(super::distance::DotProductDistance),
            crate::vector::operations::SimilarityMetric::Manhattan => Box::new(super::distance::ManhattanDistance),
            crate::vector::operations::SimilarityMetric::Jaccard => Box::new(super::distance::JaccardDistance),
        };
        
        Self {
            clusters: self.clusters.clone(),
            config: self.config.clone(),
            vector_to_cluster: self.vector_to_cluster.clone(),
            total_vectors: self.total_vectors,
            distance,
        }
    }
} 