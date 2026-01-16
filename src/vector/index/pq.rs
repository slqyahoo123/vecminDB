// PQ索引实现
// 乘积量化（Product Quantization）索引实现

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use rayon::prelude::*;

use crate::{Error, Result, vector::Vector};
use super::interfaces::VectorIndex;
use super::types::{IndexConfig, SearchResult};
// use crate::vector::VectorMetadata; // metadata conversion handled locally when needed
use super::kmeans; // 导入kmeans模块

/// PQ索引的码本
#[derive(Clone, Serialize, Deserialize, Debug)]
struct PQCodebook {
    subvector_size: usize,
    centroids: Vec<Vec<Vec<f32>>>,  // [subvector_idx][centroid_idx][value]
}

/// 乘积量化索引
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct PQIndex {
    codebook: PQCodebook,
    codes: HashMap<String, Vec<u8>>,
    metadata: HashMap<String, Option<serde_json::Value>>,
    config: IndexConfig,
    dimension: usize,
    subvector_count: usize,
    subvector_dim: usize,
    codebook_size: usize,
    vectors: HashMap<String, Vec<f32>>,
}

impl PQIndex {
    /// 创建新的PQ索引
    pub fn new(config: IndexConfig) -> Result<Self> {
        let subvector_count = if config.pq_subvector_count > 0 {
            config.pq_subvector_count
        } else {
            8
        };
        
        let codebook_size = 256; // 每个子空间的聚类中心数量
        
        Ok(Self {
            codebook: PQCodebook {
                subvector_size: config.dimension / subvector_count,
                centroids: Vec::new(),
            },
            codes: HashMap::new(),
            metadata: HashMap::new(),
            config: config.clone(),
            dimension: config.dimension,
            subvector_count,
            subvector_dim: config.dimension / subvector_count,
            codebook_size,
            vectors: HashMap::new(),
        })
    }
    
    /// 训练产品量化编码本
    fn train_codebook(&mut self, data: &[Vec<f32>]) -> Result<()> {
        if data.is_empty() {
            return Err(Error::invalid_operation("训练数据不能为空"));
        }
        
        // 确保维度匹配
        if data[0].len() != self.dimension {
            return Err(Error::invalid_operation(
                format!("向量维度不匹配，期望 {}，实际 {}", self.dimension, data[0].len())
            ));
        }
        
        // 计算真实子向量大小
        let actual_subvector_dim = self.dimension / self.subvector_count;
        let mut remaining_dims = self.dimension % self.subvector_count;
        
        // 初始化码本
        let mut codebook = Vec::with_capacity(self.subvector_count);
        
        // 为每个子空间训练聚类中心
        let mut start_idx = 0;
        for subvector_idx in 0..self.subvector_count {
            // 计算当前子向量的维度
            let current_dim = if remaining_dims > 0 {
                remaining_dims -= 1;
                actual_subvector_dim + 1
            } else {
                actual_subvector_dim
            };
            
            let end_idx = start_idx + current_dim;
            
            // 提取所有向量的当前子向量部分
            let mut subvectors = Vec::with_capacity(data.len());
            for vec in data {
                let subvec = vec[start_idx..end_idx].to_vec();
                subvectors.push(subvec);
            }
            
            // 对子向量进行k-means聚类
            let centroids = self.kmeans_clustering(&subvectors, self.codebook_size)?;
            codebook.push(centroids);
            
            start_idx = end_idx;
        }
        
        // 更新码本
        self.codebook = PQCodebook {
            subvector_size: actual_subvector_dim,
            centroids: codebook,
        };
        
        Ok(())
    }
    
    /// K-means聚类
    fn kmeans_clustering(&self, data: &[Vec<f32>], k: usize) -> Result<Vec<Vec<f32>>> {
        // 使用共享的kmeans模块
        kmeans::kmeans_clustering(data, k, 100)
    }
    
    /// 计算欧氏距离
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        // 使用共享kmeans模块的distance方法
        kmeans::euclidean_distance(a, b) as f32
    }
    
    /// 对向量进行PQ编码
    fn encode_vector(&self, vector: &[f32]) -> Result<Vec<u8>> {
        if vector.len() != self.dimension {
            return Err(Error::invalid_operation(
                format!("向量维度不匹配，期望 {}，实际 {}", self.dimension, vector.len())
            ));
        }
        
        // 如果码本为空，返回错误
        if self.codebook.centroids.is_empty() {
            return Err(Error::invalid_operation("编码本未训练，请先调用train_codebook"));
        }
        
        let mut codes = Vec::with_capacity(self.subvector_count);
        let mut start_idx = 0;
        
        // 对每个子向量进行编码
        for subvector_idx in 0..self.subvector_count {
            // 获取当前子向量的维度和当前子空间的聚类中心集合
            let subvector_dim = if subvector_idx < self.subvector_count - 1 {
                self.subvector_dim
            } else {
                self.dimension - start_idx
            };
            
            let end_idx = start_idx + subvector_dim;
            let centroids = &self.codebook.centroids[subvector_idx];
            
            // 提取当前子向量
            let subvector = &vector[start_idx..end_idx];
            
            // 寻找最近的聚类中心
            let mut best_distance = f32::MAX;
            let mut best_centroid_idx = 0;
            
            for (i, centroid) in centroids.iter().enumerate() {
                let distance = self.euclidean_distance(subvector, centroid);
                if distance < best_distance {
                    best_distance = distance;
                    best_centroid_idx = i;
                }
            }
            
            // 存储最近中心的索引作为编码
            codes.push(best_centroid_idx as u8);
            start_idx = end_idx;
        }
        
        Ok(codes)
    }
    
    /// 计算查询向量与编码本中心点的距离表
    fn compute_distance_tables(&self, query: &[f32]) -> Result<Vec<Vec<f32>>> {
        if query.len() != self.dimension {
            return Err(Error::invalid_operation(
                format!("查询向量维度不匹配，期望 {}，实际 {}", self.dimension, query.len())
            ));
        }
        
        // 如果码本为空，返回错误
        if self.codebook.centroids.is_empty() {
            return Err(Error::invalid_operation("编码本未训练，请先调用train_codebook"));
        }
        
        let mut distance_tables = Vec::with_capacity(self.subvector_count);
        let mut start_idx = 0;
        
        // 计算每个子空间的距离表
        for subvector_idx in 0..self.subvector_count {
            // 获取子向量维度
            let subvector_dim = if subvector_idx < self.subvector_count - 1 {
                self.subvector_dim
            } else {
                self.dimension - start_idx
            };
            
            let end_idx = start_idx + subvector_dim;
            let centroids = &self.codebook.centroids[subvector_idx];
            
            // 提取查询向量的子向量
            let subquery = &query[start_idx..end_idx];
            
            // 计算子查询与每个中心点的距离
            let mut distances = Vec::with_capacity(centroids.len());
            for centroid in centroids {
                let distance = self.euclidean_distance(subquery, centroid);
                distances.push(distance);
            }
            
            distance_tables.push(distances);
            start_idx = end_idx;
        }
        
        Ok(distance_tables)
    }
    
    /// 计算非对称距离（查询向量与编码向量的近似距离）
    fn asymmetric_distance(&self, query_distance_tables: &[Vec<f32>], codes: &[u8]) -> f32 {
        let mut distance = 0.0;
        
        for (subvector_idx, &code) in codes.iter().enumerate() {
            let code_idx = code as usize;
            distance += query_distance_tables[subvector_idx][code_idx];
        }
        
        distance
    }

    /// 估计向量的近似重建
    fn estimate_vector(&self, code: &[u8]) -> Result<Vec<f32>> {
        if code.len() != self.subvector_count {
            return Err(Error::invalid_operation(
                format!("编码长度不匹配，期望 {}，实际 {}", self.subvector_count, code.len())
            ));
        }
        
        // 如果码本为空，返回错误
        if self.codebook.centroids.is_empty() {
            return Err(Error::invalid_operation("编码本未训练，无法重建向量"));
        }
        
        let mut reconstructed = Vec::with_capacity(self.dimension);
        
        // 对每个子向量进行解码
        for (subvector_idx, &code_idx) in code.iter().enumerate() {
            if subvector_idx >= self.codebook.centroids.len() {
                return Err(Error::invalid_operation("编码本维度不足，无法完全解码"));
            }
            
            let centroids = &self.codebook.centroids[subvector_idx];
            let code_idx_usize = code_idx as usize;
            
            if code_idx_usize >= centroids.len() {
                return Err(Error::invalid_operation(
                    format!("编码索引超出范围: {} >= {}", code_idx_usize, centroids.len())
                ));
            }
            
            // 获取对应的中心点并添加到重建向量
            let centroid = &centroids[code_idx_usize];
            reconstructed.extend_from_slice(centroid);
        }
        
        // 确保重建的向量维度正确
        if reconstructed.len() != self.dimension {
            log::warn!(
                "重建向量维度不符合预期: {} vs {}",
                reconstructed.len(),
                self.dimension
            );
            
            // 修正向量维度
            if reconstructed.len() < self.dimension {
                // 如果不足，补0
                reconstructed.resize(self.dimension, 0.0);
            } else {
                // 如果超出，裁剪
                reconstructed.truncate(self.dimension);
            }
        }
        
        Ok(reconstructed)
    }
}

// 实现VectorIndex trait
impl VectorIndex for PQIndex {
    /// 添加向量到索引
    fn add(&mut self, vector: Vector) -> Result<()> {
        // 使用单个向量调用批量添加
        self.batch_add(&[vector])
    }
    
    /// 批量添加向量（内部方法）
    fn batch_add(&mut self, vectors: &[Vector]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }
        
        let mut raw_vectors = Vec::with_capacity(vectors.len());
        for vector in vectors {
            raw_vectors.push(vector.data.clone());
        }
        
        // 如果编码本未训练且有足够的数据，则训练编码本
        if self.codebook.centroids.is_empty() && raw_vectors.len() >= self.codebook_size {
            self.train_codebook(&raw_vectors)?;
        }
        
        // 编码并存储每个向量
        for vector in vectors {
            // 跳过已存在的向量
            if self.codes.contains_key(&vector.id) {
                continue;
            }
            
            let code = self.encode_vector(&vector.data)?;
            self.codes.insert(vector.id.clone(), code);
            
            // 存储原始向量数据
            self.vectors.insert(vector.id.clone(), vector.data.clone());
            
            // 存储元数据（如果有）
            if let Some(meta) = &vector.metadata {
                // 将 VectorMetadata 转换为 serde_json::Value
                let meta_value = serde_json::to_value(meta)
                    .unwrap_or_else(|_| serde_json::Value::Null);
                self.metadata.insert(vector.id.clone(), Some(meta_value));
            }
        }
        
        Ok(())
    }
    
    /// 搜索向量
    fn search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(Error::invalid_operation(
                format!("查询向量维度不匹配，期望 {}，实际 {}", self.dimension, query.len())
            ));
        }
        
        // 如果索引为空，返回空结果
        if self.codes.is_empty() || self.codebook.centroids.is_empty() {
            return Ok(Vec::new());
        }
        
        // 计算查询向量与编码本中心点的距离表
        let distance_tables = self.compute_distance_tables(query)?;
        
        // 计算所有编码向量与查询向量的近似距离
        let mut results = Vec::with_capacity(self.codes.len());
        
        for (id, code) in &self.codes {
            let distance = self.asymmetric_distance(&distance_tables, code);
            
            results.push(SearchResult {
                id: id.clone(),
                distance: -distance, // 负距离作为相似度分数（越高越好）
                metadata: self.metadata.get(id).cloned().flatten(),
            });
        }
        
        // 排序并截取前limit个结果
        results.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap_or(std::cmp::Ordering::Equal));
        
        if results.len() > limit {
            results.truncate(limit);
        }
        
        Ok(results)
    }
    
    /// 获取向量
    fn get(&self, id: &str) -> Result<Option<Vector>> {
        // 先检查是否存在这个ID
        if !self.codes.contains_key(id) {
            return Ok(None);
        }

        // 优先从存储的原始向量中获取
        if let Some(vector_data) = self.vectors.get(id) {
            let metadata = self.metadata.get(id)
                .and_then(|opt| opt.as_ref())
                .and_then(|v| {
                    // 将 serde_json::Value 转换为 VectorMetadata
                    if let Some(obj) = v.as_object() {
                        let mut properties = std::collections::HashMap::new();
                        for (k, v) in obj {
                            properties.insert(k.clone(), v.clone());
                        }
                        Some(crate::vector::search::VectorMetadata { properties })
                    } else {
                        None
                    }
                });
            return Ok(Some(Vector {
                id: id.to_string(),
                data: vector_data.clone(),
                metadata,
            }));
        }
        
        // 如果没有存储原始向量，则尝试从编码估计重建
        if let Some(code) = self.codes.get(id) {
            match self.estimate_vector(code) {
                Ok(reconstructed) => {
                    let metadata = self.metadata.get(id)
                        .and_then(|opt| opt.as_ref())
                        .and_then(|v| {
                            if let Some(obj) = v.as_object() {
                                let mut properties = std::collections::HashMap::new();
                                for (k, v) in obj {
                                    properties.insert(k.clone(), v.clone());
                                }
                                Some(crate::vector::search::VectorMetadata { properties })
                            } else {
                                None
                            }
                        });
                    return Ok(Some(Vector {
                        id: id.to_string(),
                        data: reconstructed,
                        metadata,
                    }));
                },
                Err(e) => {
                    log::warn!("无法重建向量 {}: {}", id, e);
                    // 失败时返回空向量但保留ID和元数据
                    let metadata = self.metadata.get(id)
                        .and_then(|opt| opt.as_ref())
                        .and_then(|v| {
                            if let Some(obj) = v.as_object() {
                                let mut properties = std::collections::HashMap::new();
                                for (k, v) in obj {
                                    properties.insert(k.clone(), v.clone());
                                }
                                Some(crate::vector::search::VectorMetadata { properties })
                            } else {
                                None
                            }
                        });
                    return Ok(Some(Vector {
                        id: id.to_string(),
                        data: vec![0.0; self.dimension],
                        metadata,
                    }));
                }
            }
        }
        
        Ok(None)
    }

    /// 删除向量
    fn delete(&mut self, id: &str) -> Result<bool> {
        let removed = self.codes.remove(id).is_some();
        if removed {
            self.metadata.remove(id);
            // 同时删除原始向量
            self.vectors.remove(id);
        }
        Ok(removed)
    }

    /// 批量删除向量（内部方法）
    fn batch_delete(&mut self, ids: &[String]) -> Result<usize> {
        let mut deleted = 0;
        
        for id in ids {
            if self.delete(id)? {
                deleted += 1;
            }
        }
        
        Ok(deleted)
    }

    /// 获取索引中的向量数量
    fn size(&self) -> usize {
        self.codes.len()
    }
    
    /// 获取索引配置
    fn get_config(&self) -> IndexConfig {
        self.config.clone()
    }

    /// 清空索引
    fn clear(&mut self) -> Result<()> {
        self.codes.clear();
        self.metadata.clear();
        // 同时清空原始向量存储
        self.vectors.clear();
        Ok(())
    }
    
    /// 序列化索引
    fn serialize(&self) -> Result<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| Error::serialization(format!("无法序列化PQ索引: {}", e)))
    }
    
    /// 反序列化索引
    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let index: PQIndex = bincode::deserialize(data)
            .map_err(|e| Error::serialization(format!("无法反序列化PQ索引: {}", e)))?;
        *self = index;
        Ok(())
    }
    
    /// 克隆索引
    fn clone_box(&self) -> Box<dyn VectorIndex + Send + Sync> {
        Box::new(self.clone())
    }
    
    /// 从字节数组创建索引并装箱
    fn deserialize_box(data: &[u8]) -> Result<Box<dyn VectorIndex + Send + Sync>> {
        let index: PQIndex = bincode::deserialize(data)
            .map_err(|e| Error::serialization(format!("无法反序列化PQ索引: {}", e)))?;
        Ok(Box::new(index))
    }
    
    /// 检查索引中是否包含指定ID的向量
    fn contains(&self, id: &str) -> bool {
        self.codes.contains_key(id)
    }
    
    /// 返回索引向量的维度
    fn dimension(&self) -> usize {
        self.dimension
    }
}

// 在这里实现VectorIndex trait，具体实现内容略（根据需要添加） 