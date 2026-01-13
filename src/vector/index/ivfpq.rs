// IVF-PQ (Inverted File System with Product Quantization) 索引实现
// 结合了IVF的快速搜索和PQ的压缩效果

use std::collections::HashMap;
// use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};
use rayon::prelude::*;
// use uuid::Uuid; // vector ids use u64 in this module

use crate::{Error, Result, vector::{Vector, operations::SimilarityMetric}};
use super::types::{IndexConfig, SearchResult};
use super::interfaces::VectorIndex;
use super::distance::{compute_distance_raw, Distance};
use super::kmeans::{KMeans, KMeansConfig}; // 使用共享kmeans模块

/// IVFPQ索引：结合了IVF (倒排文件) 和 PQ (乘积量化) 的索引结构
/// 用于大规模向量检索，具有高效存储和查询特性
#[derive(Debug)]
pub struct IVFPQIndex {
    /// 索引配置
    config: IndexConfig,
    
    /// 距离计算器
    distance: Box<dyn Distance + Send + Sync>,
    
    /// 聚类中心 (IVF部分)
    centroids: Vec<Vec<f32>>,
    
    /// 向量ID到聚类中心的映射
    vector_to_cluster: HashMap<u64, usize>,
    
    /// 聚类到向量的倒排映射
    clusters: Vec<Vec<(u64, Vec<u8>)>>,
    
    /// 每个子空间的编码本 (PQ部分)
    codebooks: Vec<Vec<Vec<f32>>>,
    
    /// 向量元数据
    vector_metadata: HashMap<u64, HashMap<String, String>>,
    
    /// 原始向量存储（用于构建索引时）
    raw_vectors: HashMap<u64, Vec<f32>>,
    
    /// 向量数量
    vector_count: usize,
    
    /// 是否已构建
    is_built: bool,
}

impl IVFPQIndex {
    /// Convert String ID to u64 using hash
    fn string_id_to_u64(id: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        id.hash(&mut hasher);
        hasher.finish()
    }
    
    /// 创建新的IVFPQ索引
    pub fn new(config: IndexConfig) -> Result<Self> {
        // 验证配置
        if config.dimension == 0 {
            return Err(Error::vector("Dimension must be greater than 0"));
        }
        
        let mut ivf_centers = config.ivf_centers;
        if ivf_centers == 0 {
            // 设置默认聚类中心数量 (通常是数据大小的平方根或预期元素的一部分)
            ivf_centers = (config.expected_elements as f32 / 50.0).round() as usize;
            ivf_centers = ivf_centers.max(8).min(config.expected_elements);
        }
        
        let mut pq_subvectors = config.pq_subvectors;
        if pq_subvectors == 0 {
            // 子向量数量通常是维度的1/4到1/8
            pq_subvectors = (config.dimension / 4).max(1).min(config.dimension);
        }
        
        let mut pq_bits = config.pq_nbits;
        if pq_bits == 0 {
            pq_bits = 8; // 默认用8位编码
        }
        
        // 确保子向量数能整除维度
        if config.dimension % pq_subvectors != 0 {
            return Err(Error::vector(format!(
                "Dimension ({}) must be divisible by the number of subvectors ({})",
                config.dimension, pq_subvectors
            )));
        }
        
        // 初始化空的聚类中心和编码本
        let centroids = Vec::new();
        let codebooks = Vec::new();
        
        // 创建距离计算器
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
            centroids,
            vector_to_cluster: HashMap::new(),
            clusters: Vec::new(),
            codebooks,
            vector_metadata: HashMap::new(),
            raw_vectors: HashMap::new(),
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
        if self.vector_count < self.config.ivf_centers {
            return Err(Error::vector(format!(
                "Not enough vectors ({}) to create the requested number of clusters ({})",
                self.vector_count, self.config.ivf_centers
            )));
        }
        
        // 1. 构建IVF部分 - 使用K-means聚类
        self.build_ivf()?;
        
        // 2. 构建PQ部分 - 为每个子空间创建编码本
        self.build_pq()?;
        
        self.is_built = true;
        Ok(())
    }
    
    /// 构建IVF (倒排文件) 部分
    fn build_ivf(&mut self) -> Result<()> {
        // 收集所有原始向量
        let mut all_vectors = Vec::new();
        let mut vector_ids = Vec::new();
        
        for (vector_id, cluster) in &self.vector_to_cluster {
            // 这里需要从临时存储中获取原始向量
            if let Some(vector_data) = self.get_raw_vector(*vector_id) {
                all_vectors.push(vector_data);
                vector_ids.push(*vector_id);
            }
        }
        
        if all_vectors.is_empty() {
            return Err(Error::vector("No vectors available for building IVF"));
        }
        
        // 运行K-means聚类
        let k_means_config = KMeansConfig {
            k: self.config.ivf_centers,
            max_iterations: 100,
            convergence_threshold: 0.001,
            ..Default::default()
        };
        
        let k_means = KMeans::new(k_means_config);
        let (centroids, assignments) = k_means.fit(&all_vectors)?;
        
        // 保存聚类中心
        self.centroids = centroids;
        
        // 初始化聚类
        self.clusters = vec![Vec::new(); self.config.ivf_centers];
        
        // 更新向量到聚类的映射
        for (i, &vector_id) in vector_ids.iter().enumerate() {
            let cluster_id = assignments[i];
            self.vector_to_cluster.insert(vector_id, cluster_id);
        }
        
        Ok(())
    }
    
    /// 构建PQ (乘积量化) 部分
    fn build_pq(&mut self) -> Result<()> {
        let sub_dim = self.config.dimension / self.config.pq_subvectors;
        let n_clusters = 1 << self.config.pq_nbits; // 2^bits
        
        // 初始化编码本
        self.codebooks = vec![vec![vec![0.0; sub_dim]; n_clusters]; self.config.pq_subvectors];
        
        // 收集所有向量并分割为子向量
        let mut subvectors = vec![Vec::new(); self.config.pq_subvectors];
        
        // 为每个聚类收集向量
        for cluster_id in 0..self.config.ivf_centers {
            let vectors_in_cluster = self.collect_cluster_vectors(cluster_id)?;
            
            // 跳过空聚类
            if vectors_in_cluster.is_empty() {
                continue;
            }
            
            // 将向量分割为子向量
            for vector in &vectors_in_cluster {
                for i in 0..self.config.pq_subvectors {
                    let start = i * sub_dim;
                    let end = start + sub_dim;
                    let subvector = vector[start..end].to_vec();
                    subvectors[i].push(subvector);
                }
            }
            
            // 对每个向量进行PQ编码
            for (vector_id, vector) in self.collect_cluster_vector_ids(cluster_id)?.iter().zip(vectors_in_cluster.iter()) {
                let codes = self.encode_vector(vector)?;
                
                // 将编码添加到对应的聚类中
                self.clusters[cluster_id].push((*vector_id, codes));
            }
        }
        
        // 对每个子空间运行K-means聚类以创建编码本
        let k_means_config = KMeansConfig {
            k: n_clusters,
            max_iterations: 100,
            convergence_threshold: 0.001,
            ..Default::default()
        };
        
        for i in 0..self.config.pq_subvectors {
            if subvectors[i].is_empty() {
                continue;
            }
            
            let k_means = KMeans::new(k_means_config.clone());
            let (codebook, _) = k_means.fit(&subvectors[i])?;
            self.codebooks[i] = codebook;
        }
        
        Ok(())
    }
    
    /// 编码向量
    fn encode_vector(&self, vector: &[f32]) -> Result<Vec<u8>> {
        let sub_dim = self.config.dimension / self.config.pq_subvectors;
        let mut codes = Vec::with_capacity(self.config.pq_subvectors);
        
        for i in 0..self.config.pq_subvectors {
            let start = i * sub_dim;
            let end = start + sub_dim;
            let subvector = &vector[start..end];
            
            // 找到最近的编码本项
            let mut min_dist = f32::MAX;
            let mut min_code = 0;
            
            for (code, centroid) in self.codebooks[i].iter().enumerate() {
                let dist = compute_distance_raw(subvector, centroid, &self.distance);
                if dist < min_dist {
                    min_dist = dist;
                    min_code = code;
                }
            }
            
            codes.push(min_code as u8);
        }
        
        Ok(codes)
    }
    
    /// 解码向量
    fn decode_vector(&self, codes: &[u8]) -> Result<Vec<f32>> {
        if codes.len() != self.config.pq_subvectors {
            return Err(Error::vector(format!(
                "Invalid code length: {} (expected {})",
                codes.len(), self.config.pq_subvectors
            )));
        }
        
        let sub_dim = self.config.dimension / self.config.pq_subvectors;
        let mut decoded = Vec::with_capacity(self.config.dimension);
        
        for i in 0..self.config.pq_subvectors {
            let code = codes[i] as usize;
            if code >= self.codebooks[i].len() {
                return Err(Error::vector(format!(
                    "Invalid code {} for subvector {}", code, i
                )));
            }
            
            decoded.extend_from_slice(&self.codebooks[i][code]);
        }
        
        Ok(decoded)
    }
    
    /// 获取原始向量
    fn get_raw_vector(&self, vector_id: u64) -> Option<Vec<f32>> {
        self.raw_vectors.get(&vector_id).cloned()
    }
    
    /// 收集聚类中的向量
    fn collect_cluster_vectors(&self, cluster_id: usize) -> Result<Vec<Vec<f32>>> {
        let mut vectors = Vec::new();
        
        // 获取该聚类中的所有向量ID
        let vector_ids = self.collect_cluster_vector_ids(cluster_id)?;
        
        // 收集向量
        for vector_id in vector_ids {
            if let Some(vector) = self.get_raw_vector(vector_id) {
                vectors.push(vector);
            }
        }
        
        Ok(vectors)
    }
    
    /// 收集聚类中的向量ID
    fn collect_cluster_vector_ids(&self, cluster_id: usize) -> Result<Vec<u64>> {
        if cluster_id >= self.clusters.len() {
            return Err(Error::vector(format!("Invalid cluster ID: {}", cluster_id)));
        }
        
        let mut vector_ids = Vec::new();
        
        for (vector_id, _) in &self.clusters[cluster_id] {
            vector_ids.push(*vector_id);
        }
        
        Ok(vector_ids)
    }
    
    /// 查找向量所属的聚类
    fn find_closest_centroids(&self, vector: &[f32], nprobe: usize) -> Vec<(usize, f32)> {
        let mut distances = Vec::with_capacity(self.centroids.len());
        
        // 计算向量到所有聚类中心的距离
        for (i, centroid) in self.centroids.iter().enumerate() {
            let dist = compute_distance_raw(vector, centroid, &self.distance);
            distances.push((i, dist));
        }
        
        // 按距离排序
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // 返回最近的nprobe个聚类
        distances.truncate(nprobe);
        distances
    }
    
    /// 使用预计算的查找表计算向量到PQ编码向量的距离
    fn compute_distance_with_tables(&self, vector: &[f32], codes: &[u8], tables: &Vec<Vec<f32>>) -> f32 {
        let mut distance = 0.0;
        
        for i in 0..self.config.pq_subvectors {
            let code = codes[i] as usize;
            distance += tables[i][code];
        }
        
        distance
    }
    
    /// 计算距离查找表
    fn compute_distance_tables(&self, vector: &[f32]) -> Vec<Vec<f32>> {
        let sub_dim = self.config.dimension / self.config.pq_subvectors;
        let mut tables = Vec::with_capacity(self.config.pq_subvectors);
        
        for i in 0..self.config.pq_subvectors {
            let start = i * sub_dim;
            let end = start + sub_dim;
            let subvector = &vector[start..end];
            
            let mut subtable = Vec::with_capacity(self.codebooks[i].len());
            
            for centroid in &self.codebooks[i] {
                let dist = compute_distance_raw(subvector, centroid, &self.distance);
                subtable.push(dist);
            }
            
            tables.push(subtable);
        }
        
        tables
    }
}

impl VectorIndex for IVFPQIndex {
    /// 添加向量到索引
    fn add(&mut self, vector: Vector) -> Result<()> {
        // 如果索引已构建，需要为新向量找到最近的聚类中心
        if self.is_built {
            // 找到最近的聚类中心
            let closest = self.find_closest_centroids(&vector.data, 1);
            if let Some((cluster_id, _)) = closest.first() {
                // 编码向量
                let codes = self.encode_vector(&vector.data)?;
                
                // Convert String ID to u64
                let vector_id = Self::string_id_to_u64(&vector.id);
                
                // 添加到对应的聚类
                self.clusters[*cluster_id].push((vector_id, codes));
                self.vector_to_cluster.insert(vector_id, *cluster_id);
            } else {
                return Err(Error::vector("Failed to find closest centroid"));
            }
        } else {
            // 索引尚未构建，存储原始向量以便后续构建索引
            // Convert String ID to u64
            let vector_id = Self::string_id_to_u64(&vector.id);
            self.raw_vectors.insert(vector_id, vector.data.clone());
            self.vector_to_cluster.insert(vector_id, 0); // 临时分配到第一个聚类
        }
        
        // 保存向量元数据
        if let Some(metadata) = &vector.metadata {
            if !metadata.properties.is_empty() {
                // Convert String ID to u64
                let vector_id = Self::string_id_to_u64(&vector.id);
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
            normalize_vector(&mut q);
            q
        } else {
            query.to_vec()
        };
        
        // 使用默认的nprobe设置
        let nprobe = self.config.ivf_nprobe.max(1);
        
        // 找到最近的聚类中心
        let closest_centroids = self.find_closest_centroids(&normalized_query, nprobe);
        
        // 预计算距离查找表
        let distance_tables = self.compute_distance_tables(&normalized_query);
        
        // 在选定的聚类中搜索
        let mut results = Vec::new();
        
        for (cluster_id, _) in closest_centroids {
            if cluster_id >= self.clusters.len() {
                continue;
            }
            
            for (vector_id, codes) in &self.clusters[cluster_id] {
                // 计算距离
                let distance = self.compute_distance_with_tables(&normalized_query, codes, &distance_tables);
                
                // 获取元数据并转换
                let metadata = self.vector_metadata.get(vector_id).map(|m| {
                    let mut props = serde_json::Map::new();
                    for (k, v) in m {
                        props.insert(k.clone(), serde_json::Value::String(v.clone()));
                    }
                    serde_json::Value::Object(props)
                });
                
                // 添加到结果
                // Convert u64 ID back to String (using hex representation)
                results.push(SearchResult {
                    id: format!("{:x}", vector_id),
                    distance,
                    metadata,
                });
            }
        }
        
        // 根据距离排序
        match self.config.metric {
            SimilarityMetric::Cosine | SimilarityMetric::DotProduct => {
                // 对于余弦和内积，分数越大越好
                results.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap_or(std::cmp::Ordering::Equal));
            }
            _ => {
                // 对于欧氏距离等，分数越小越好
                results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
        
        // 截取limit个结果
        results.truncate(limit);
        
        Ok(results)
    }

    /// 从索引中删除向量
    fn delete(&mut self, id: &str) -> Result<bool> {
        let vector_id = id.parse::<u64>()
            .map_err(|_| Error::vector("Invalid vector ID format"))?;
            
        // 查找向量所在的聚类
        if let Some(&cluster_id) = self.vector_to_cluster.get(&vector_id) {
            // 从聚类中删除向量
            if cluster_id < self.clusters.len() {
                self.clusters[cluster_id].retain(|(vid, _)| *vid != vector_id);
            }
            
            // 从映射中删除
            self.vector_to_cluster.remove(&vector_id);
            
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
            self.vector_to_cluster.contains_key(&vector_id)
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
        // 创建可序列化的结构（排除 distance 字段）
        #[derive(Serialize)]
        struct SerializableIVFPQIndex {
            config: IndexConfig,
            centroids: Vec<Vec<f32>>,
            vector_to_cluster: HashMap<u64, usize>,
            clusters: Vec<Vec<(u64, Vec<u8>)>>,
            codebooks: Vec<Vec<Vec<f32>>>,
            vector_metadata: HashMap<u64, HashMap<String, String>>,
            raw_vectors: HashMap<u64, Vec<f32>>,
            vector_count: usize,
            is_built: bool,
        }
        
        let serializable = SerializableIVFPQIndex {
            config: self.config.clone(),
            centroids: self.centroids.clone(),
            vector_to_cluster: self.vector_to_cluster.clone(),
            clusters: self.clusters.clone(),
            codebooks: self.codebooks.clone(),
            vector_metadata: self.vector_metadata.clone(),
            raw_vectors: self.raw_vectors.clone(),
            vector_count: self.vector_count,
            is_built: self.is_built,
        };
        
        bincode::serialize(&serializable)
            .map_err(|e| Error::vector(format!("Failed to serialize index: {}", e)))
    }
    
    /// 反序列化索引
    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        // 反序列化
        #[derive(Deserialize)]
        struct SerializableIVFPQIndex {
            config: IndexConfig,
            centroids: Vec<Vec<f32>>,
            vector_to_cluster: HashMap<u64, usize>,
            clusters: Vec<Vec<(u64, Vec<u8>)>>,
            codebooks: Vec<Vec<Vec<f32>>>,
            vector_metadata: HashMap<u64, HashMap<String, String>>,
            raw_vectors: HashMap<u64, Vec<f32>>,
            vector_count: usize,
            is_built: bool,
        }
        
        let serialized: SerializableIVFPQIndex = bincode::deserialize(data)
            .map_err(|e| Error::vector(format!("Failed to deserialize index: {}", e)))?;
        
        // 更新现有实例
        self.config = serialized.config.clone();
        self.centroids = serialized.centroids;
        self.vector_to_cluster = serialized.vector_to_cluster;
        self.clusters = serialized.clusters;
        self.codebooks = serialized.codebooks;
        self.vector_metadata = serialized.vector_metadata;
        self.raw_vectors = serialized.raw_vectors;
        self.vector_count = serialized.vector_count;
        self.is_built = serialized.is_built;
        
        // 重新创建距离函数
        self.distance = match self.config.metric {
            SimilarityMetric::Euclidean => Box::new(super::distance::EuclideanDistance),
            SimilarityMetric::Cosine => Box::new(super::distance::CosineDistance),
            SimilarityMetric::DotProduct => Box::new(super::distance::DotProductDistance),
            SimilarityMetric::Manhattan => Box::new(super::distance::ManhattanDistance),
            SimilarityMetric::Jaccard => Box::new(super::distance::JaccardDistance),
            _ => Box::new(super::distance::EuclideanDistance),
        };
        
        Ok(())
    }

    /// 创建索引的深拷贝并装箱
    fn clone_box(&self) -> Box<dyn VectorIndex + Send + Sync> {
        Box::new(self.clone())
    }

    /// 从字节数组创建索引并装箱
    fn deserialize_box(data: &[u8]) -> Result<Box<dyn VectorIndex + Send + Sync>> where Self: Sized {
        // 反序列化
        #[derive(Deserialize)]
        struct SerializableIVFPQIndex {
            config: IndexConfig,
            centroids: Vec<Vec<f32>>,
            vector_to_cluster: HashMap<u64, usize>,
            clusters: Vec<Vec<(u64, Vec<u8>)>>,
            codebooks: Vec<Vec<Vec<f32>>>,
            vector_metadata: HashMap<u64, HashMap<String, String>>,
            raw_vectors: HashMap<u64, Vec<f32>>,
            vector_count: usize,
            is_built: bool,
        }
        
        let serialized: SerializableIVFPQIndex = bincode::deserialize(data)
            .map_err(|e| Error::vector(format!("Failed to deserialize index: {}", e)))?;
        
        // 创建实例并重新创建不可序列化的成员
        let mut index = Self::new(serialized.config)?;
        index.centroids = serialized.centroids;
        index.vector_to_cluster = serialized.vector_to_cluster;
        index.clusters = serialized.clusters;
        index.codebooks = serialized.codebooks;
        index.vector_metadata = serialized.vector_metadata;
        index.raw_vectors = serialized.raw_vectors;
        index.vector_count = serialized.vector_count;
        index.is_built = serialized.is_built;
        
        Ok(Box::new(index))
    }

    /// 清空索引
    fn clear(&mut self) -> Result<()> {
        self.centroids.clear();
        self.vector_to_cluster.clear();
        self.clusters.clear();
        self.codebooks.clear();
        self.vector_metadata.clear();
        self.vector_count = 0;
        self.is_built = false;
        
        Ok(())
    }

    /// 从索引中删除向量（按ID）
    fn remove(&mut self, vector_id: u64) -> Result<()> {
        // 查找向量所在的聚类
        if let Some(&cluster_id) = self.vector_to_cluster.get(&vector_id) {
            // 从聚类中删除向量
            if cluster_id < self.clusters.len() {
                self.clusters[cluster_id].retain(|(id, _)| *id != vector_id);
            }
            
            // 从映射中删除
            self.vector_to_cluster.remove(&vector_id);
            
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
        
        for (cluster_id, cluster) in self.clusters.iter().enumerate() {
            for (vector_id, codes) in cluster {
                // 尝试解码向量
                if let Ok(decoded_vector) = self.decode_vector(codes) {
                    let metadata = self.vector_metadata.get(vector_id).cloned();
                    
                    // VectorData uses u64 ID directly
                    all_vectors.push(crate::vector::index::interfaces::VectorData {
                        id: *vector_id,
                        vector: decoded_vector,
                        metadata: metadata,
                    });
                }
            }
        }
        
        Ok(all_vectors)
    }
}

/// 向量标准化函数
impl Clone for IVFPQIndex {
    fn clone(&self) -> Self {
        // Recreate distance based on config metric
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
            centroids: self.centroids.clone(),
            vector_to_cluster: self.vector_to_cluster.clone(),
            clusters: self.clusters.clone(),
            codebooks: self.codebooks.clone(),
            vector_metadata: self.vector_metadata.clone(),
            raw_vectors: self.raw_vectors.clone(),
            vector_count: self.vector_count,
            is_built: self.is_built,
        }
    }
}

pub fn normalize_vector(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vector.iter_mut() {
            *x /= norm;
        }
    }
} 