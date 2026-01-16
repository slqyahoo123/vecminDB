// Flat索引实现
// 暴力搜索的向量索引实现

use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
// no sync primitives needed in this implementation

use crate::error::{Error, Result};
use crate::vector::types::Vector;
// use crate::vector::VectorMetadata; // metadata is converted locally to serde_json::Value
use crate::vector::index::types::{SearchResult, IndexConfig};
use crate::vector::index::VectorIndex;
use crate::vector::operations::SimilarityMetric;
use crate::vector::core::operations::VectorOps;
// use super::distance::Distance;

/// 暴力搜索索引（无索引，线性扫描）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatIndex {
    vectors: HashMap<String, Vector>,
    config: IndexConfig,
}

impl FlatIndex {
    pub fn new(config: IndexConfig) -> Self {
        FlatIndex {
            vectors: HashMap::new(),
            config,
        }
    }
    
    /// 序列化索引
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let serialized = bincode::serialize(self)
            .map_err(|e| Error::serialization(format!("Failed to serialize FlatIndex: {}", e)))?;
        Ok(serialized)
    }
    
    /// 反序列化索引
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        let index: FlatIndex = bincode::deserialize(data)
            .map_err(|e| Error::serialization(format!("Failed to deserialize FlatIndex: {}", e)))?;
        Ok(index)
    }
    
    /// 检查索引是否包含指定ID的向量
    pub fn contains(&self, id: &str) -> bool {
        self.vectors.contains_key(id)
    }
    
    /// 获取索引的维度
    pub fn dimension(&self) -> usize {
        self.config.dimension
    }
    
    /// 获取索引的配置
    pub fn get_config(&self) -> IndexConfig {
        self.config.clone()
    }
}

impl VectorIndex for FlatIndex {
    fn add(&mut self, vector: Vector) -> Result<()> {
        if vector.data.len() != self.config.dimension {
            return Err(Error::vector(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.config.dimension, vector.data.len()
            )));
        }
        
        self.vectors.insert(vector.id.clone(), vector);
        Ok(())
    }

    fn search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.config.dimension {
            return Err(Error::vector(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.config.dimension, query.len()
            )));
        }

        let mut results = BinaryHeap::new();
        
        // 并行处理向量比较
        let distances: Vec<(String, f32, Option<serde_json::Value>)> = self.vectors
            .par_iter()
            .map(|(id, vector)| {
                let distance = match self.config.metric {
                    SimilarityMetric::Euclidean => {
                        -VectorOps::euclidean_distance(query, &vector.data)
                    },
                    SimilarityMetric::Manhattan => {
                        -VectorOps::manhattan_distance(query, &vector.data)
                    },
                    _ => {
                        // 对于余弦相似度和点积，我们用1-相似度作为距离
                        // 注意：这里较低的值表示较高的相似度
                        let sim = VectorOps::compute_similarity(
                            query, 
                            &vector.data,
                            self.config.metric
                        );
                        1.0 - sim
                    }
                };
                
                // 将VectorMetadata转换为serde_json::Value
                let metadata_value = vector.metadata.as_ref().map(|m| {
                    let mut map = serde_json::Map::new();
                    for (k, v) in &m.properties {
                        map.insert(k.clone(), v.clone());
                    }
                    serde_json::Value::Object(map)
                });
                
                (id.clone(), distance, metadata_value)
            })
            .collect();
        
        // 构建结果堆
        for (id, distance, metadata) in distances {
            results.push(SearchResult {
                id,
                distance,
                metadata,
            });
            
            if results.len() > limit {
                results.pop();
            }
        }
        
        // 转换为有序结果
        let mut sorted_results = Vec::with_capacity(results.len());
        while let Some(result) = results.pop() {
            sorted_results.push(result);
        }
        
        // 根据距离排序 (对于欧几里得距离，越小越好；对于余弦相似度和点积，值已转换为距离)
        if self.config.metric == SimilarityMetric::Euclidean {
            // 欧几里得距离：从小到大
            sorted_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        } else {
            // 相似度已转换为距离(1-相似度)：从小到大
            sorted_results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        }
        
        Ok(sorted_results)
    }

    fn delete(&mut self, id: &str) -> Result<bool> {
        Ok(self.vectors.remove(id).is_some())
    }

    fn contains(&self, id: &str) -> bool {
        self.vectors.contains_key(id)
    }
    
    fn dimension(&self) -> usize {
        self.config.dimension
    }
    
    fn get_config(&self) -> IndexConfig {
        self.config.clone()
    }

    fn size(&self) -> usize {
        self.vectors.len()
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        FlatIndex::serialize(self)
    }
    
    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let new_index = FlatIndex::deserialize(data)?;
        *self = new_index;
        Ok(())
    }
    
    fn clone_box(&self) -> Box<dyn VectorIndex + Send + Sync> {
        Box::new(self.clone())
    }
    
    fn deserialize_box(data: &[u8]) -> Result<Box<dyn VectorIndex + Send + Sync>> {
        let index = FlatIndex::deserialize(data)?;
        Ok(Box::new(index))
    }
} 