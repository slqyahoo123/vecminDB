// use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::{Result, Error};
use crate::vector::types::Vector;
use crate::vector::index::types::{IndexConfig, SearchResult};
use crate::vector::index::interfaces::VectorIndex;
// use crate::vector::search::VectorMetadata;

/// K-means聚类配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KMeansConfig {
    /// 聚类中心数量
    pub k: usize,
    
    /// 最大迭代次数
    pub max_iterations: usize,
    
    /// 收敛阈值
    pub convergence_threshold: f64,
    
    /// 初始化方法 (0: 随机, 1: K-means++)
    pub init_method: usize,
    
    /// 并行计算
    pub parallel: bool,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 10,
            max_iterations: 100,
            convergence_threshold: 0.001,
            init_method: 1,
            parallel: true,
        }
    }
}

/// K-means聚类算法实现
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KMeans {
    config: KMeansConfig,
}

impl KMeans {
    /// 创建新的K-means实例
    pub fn new(config: KMeansConfig) -> Self {
        Self { config }
    }
    
    /// 执行K-means聚类
    pub fn fit(&self, data: &[Vec<f32>]) -> Result<(Vec<Vec<f32>>, Vec<usize>)> {
        if data.is_empty() {
            return Err(Error::vector("Input data is empty"));
        }
        
        if data.len() < self.config.k {
            return Err(Error::vector("Number of data points is less than k"));
        }
        
        let dimension = data[0].len();
        
        // 初始化聚类中心
        let mut centroids = match self.config.init_method {
            0 => self.random_init(data)?,
            1 => self.kmeans_plus_plus_init(data)?,
            _ => self.random_init(data)?,
        };
        
        let mut assignments = vec![0; data.len()];
        let mut converged = false;
        
        for iteration in 0..self.config.max_iterations {
            // 分配数据点到聚类
            let (new_assignments, distances, changed_count) = self.parallel_assign(data, &centroids);
            assignments = new_assignments;
            
            // 检查收敛
            if changed_count == 0 {
                converged = true;
                break;
            }
            
            // 更新聚类中心
            let new_centroids = self.update_centroids(data, &assignments)?;
            
            // 计算中心移动距离
            let centroid_movement = centroids.iter()
                .zip(new_centroids.iter())
                .map(|(old, new)| euclidean_distance(old, new))
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);
            
            centroids = new_centroids;
            
            // 检查收敛
            if centroid_movement < self.config.convergence_threshold {
                converged = true;
                break;
            }
        }
        
        if !converged {
            log::warn!("K-means did not converge within {} iterations", self.config.max_iterations);
        }
        
        Ok((centroids, assignments))
    }
    
    /// 随机初始化聚类中心
    fn random_init(&self, data: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        
        let mut rng = thread_rng();
        let mut centroids = Vec::with_capacity(self.config.k);
        
        // 随机选择k个数据点作为初始聚类中心
        let mut indices: Vec<usize> = (0..data.len()).collect();
        indices.shuffle(&mut rng);
        
        for i in 0..self.config.k {
            centroids.push(data[indices[i]].clone());
        }
        
        Ok(centroids)
    }
    
    /// K-means++初始化
    fn kmeans_plus_plus_init(&self, data: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        use rand::{thread_rng, Rng};
        use rand::distributions::WeightedIndex;
        
        let mut rng = thread_rng();
        let mut centroids = Vec::with_capacity(self.config.k);
        
        // 随机选择第一个聚类中心
        let first_idx = rng.gen_range(0..data.len());
        centroids.push(data[first_idx].clone());
        
        // 选择剩余的聚类中心
        for _ in 1..self.config.k {
            let mut distances = Vec::with_capacity(data.len());
            
            // 计算每个点到最近聚类中心的距离
            for point in data {
                let min_distance = centroids.iter()
                    .map(|centroid| euclidean_distance(point, centroid))
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0);
                distances.push(min_distance * min_distance); // 距离的平方
            }
            
            // 根据距离权重选择下一个聚类中心
            let weighted_dist = WeightedIndex::new(&distances)
                .map_err(|e| Error::vector(format!("Failed to create weighted distribution: {}", e)))?;
            use rand_distr::Distribution;
            let selected_idx = weighted_dist.sample(&mut rng);
            centroids.push(data[selected_idx].clone());
        }
        
        Ok(centroids)
    }
    
    /// 分配数据点到聚类
    fn find_nearest_centroid(&self, vector: &[f32], centroids: &[Vec<f32>]) -> (usize, f64) {
        let mut min_distance = f64::MAX;
        let mut nearest_centroid = 0;
        
        for (i, centroid) in centroids.iter().enumerate() {
            let distance = euclidean_distance(vector, centroid);
            if distance < min_distance {
                min_distance = distance;
                nearest_centroid = i;
            }
        }
        
        (nearest_centroid, min_distance)
    }
    
    /// 并行分配数据点到聚类
    fn parallel_assign(&self, data: &[Vec<f32>], centroids: &[Vec<f32>]) -> (Vec<usize>, Vec<f64>, usize) {
        use rayon::prelude::*;
        
        let results: Vec<(usize, f64)> = if self.config.parallel {
            data.par_iter()
                .map(|point| self.find_nearest_centroid(point, centroids))
                .collect()
        } else {
            data.iter()
                .map(|point| self.find_nearest_centroid(point, centroids))
                .collect()
        };
        
        let assignments: Vec<usize> = results.iter().map(|(idx, _)| *idx).collect();
        let distances: Vec<f64> = results.iter().map(|(_, dist)| *dist).collect();
        
        // 这里简化处理，实际应该与之前的分配比较
        let changed_count = assignments.len(); // 假设都变化了，第一次迭代
        
        (assignments, distances, changed_count)
    }
    
    /// 更新聚类中心
    fn update_centroids(&self, data: &[Vec<f32>], assignments: &[usize]) -> Result<Vec<Vec<f32>>> {
        let dimension = data[0].len();
        let mut new_centroids = vec![vec![0.0; dimension]; self.config.k];
        let mut counts = vec![0; self.config.k];
        
        // 累加每个聚类的点
        for (point, &cluster_id) in data.iter().zip(assignments.iter()) {
            if cluster_id < self.config.k {
                for (i, &value) in point.iter().enumerate() {
                    new_centroids[cluster_id][i] += value;
                }
                counts[cluster_id] += 1;
            }
        }
        
        // 计算平均值
        for (cluster_id, count) in counts.iter().enumerate() {
            if *count > 0 {
                for value in new_centroids[cluster_id].iter_mut() {
                    *value /= *count as f32;
                }
            }
        }
        
        Ok(new_centroids)
    }
}

/// 计算欧几里得距离
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return f64::MAX;
    }
    
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = (x - y) as f64;
            diff * diff
        })
        .sum::<f64>()
        .sqrt()
}

/// K-means聚类执行函数
pub fn kmeans_clustering(data: &[Vec<f32>], k: usize, max_iterations: usize) -> Result<Vec<Vec<f32>>> {
    let config = KMeansConfig {
        k,
        max_iterations,
        convergence_threshold: 0.001,
        init_method: 1,
        parallel: true,
    };
    
    let kmeans = KMeans::new(config);
    let (centroids, _) = kmeans.fit(data)?;
    
    Ok(centroids)
}

/// K-means向量索引
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KMeansIndex {
    /// 向量维度
    dimension: usize,
    /// 索引配置
    config: IndexConfig,
    /// 向量数据
    vectors: Vec<Vector>,
    /// 聚类中心
    centroids: Vec<Vec<f32>>,
    /// 向量分配到的聚类
    assignments: Vec<usize>,
    /// 聚类算法实例
    kmeans: KMeans,
}

impl KMeansIndex {
    /// 创建新的K-means索引
    pub fn new(config: IndexConfig) -> Result<Self> {
        let k = config.ivf_centers.max(100);
        let max_iterations = 100; // Default max iterations
        
        let kmeans_config = KMeansConfig {
            k,
            max_iterations,
            convergence_threshold: 0.001,
            init_method: 1,
            parallel: true,
        };
        
        Ok(Self {
            dimension: config.dimension,
            config,
            vectors: Vec::new(),
            centroids: Vec::new(),
            assignments: Vec::new(),
            kmeans: KMeans::new(kmeans_config),
        })
    }
    
    /// 重新构建聚类
    fn rebuild_clusters(&mut self) -> Result<()> {
        if self.vectors.is_empty() {
            return Ok(());
        }
        
        // 提取向量数据
        let data: Vec<Vec<f32>> = self.vectors
            .iter()
            .map(|v| v.data.clone())
            .collect();
        
        // 执行聚类
        let (centroids, assignments) = self.kmeans.fit(&data)?;
        
        self.centroids = centroids;
        self.assignments = assignments;
        
        Ok(())
    }
}

impl VectorIndex for KMeansIndex {
    fn add(&mut self, vector: Vector) -> Result<()> {
        // 验证向量维度
        if vector.data.len() != self.dimension {
            return Err(Error::vector(
                format!("Vector dimension mismatch: expected {}, got {}", 
                        self.dimension, vector.data.len())
            ));
        }
        
        // 添加向量
        self.vectors.push(vector);
        
        // 如果向量数量达到一定阈值，重建聚类
        if self.vectors.len() % 1000 == 0 {
            self.rebuild_clusters()?;
        }
        
        Ok(())
    }
    
    fn search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(Error::vector(
                format!("Query dimension mismatch: expected {}, got {}", 
                        self.dimension, query.len())
            ));
        }
        
        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }
        
        if self.centroids.is_empty() {
            // 如果尚未构建聚类，则执行线性搜索
            return self.linear_search(query, limit);
        }
        
        // 找到最近的聚类中心
        let mut nearest_centroids = Vec::new();
        for (i, centroid) in self.centroids.iter().enumerate() {
            let distance = euclidean_distance(query, centroid);
            nearest_centroids.push((i, distance));
        }
        
        // 按距离排序
        nearest_centroids.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // 搜索最近聚类中的向量
        let mut results = Vec::new();
        for &(cluster_idx, _) in nearest_centroids.iter().take(self.kmeans.config.k.min(5)) {
            // 获取该聚类中的所有向量
            let cluster_vectors: Vec<&Vector> = self.vectors
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < self.assignments.len() && self.assignments[*i] == cluster_idx)
                .map(|(_, v)| v)
                .collect();
            
            // 计算与查询向量的距离
            for vector in cluster_vectors {
                let distance = euclidean_distance(query, &vector.data);
                let metadata = vector.metadata.as_ref().map(|m| {
                    let mut props = serde_json::Map::new();
                    for (k, v) in &m.properties {
                        props.insert(k.clone(), v.clone());
                    }
                    serde_json::Value::Object(props)
                });
                
                results.push(SearchResult {
                    id: vector.id.clone(),
                    distance: -distance as f32, // Negative distance as score, larger is more similar
                    metadata,
                });
            }
        }
        
        // 排序并限制结果数量
        results.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap_or(std::cmp::Ordering::Equal));
        if results.len() > limit {
            results.truncate(limit);
        }
        
        Ok(results)
    }
    
    fn delete(&mut self, id: &str) -> Result<bool> {
        let initial_len = self.vectors.len();
        self.vectors.retain(|v| v.id != id);
        
        if self.vectors.len() != initial_len {
            // 如果删除了向量，重建聚类
            self.rebuild_clusters()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    fn contains(&self, id: &str) -> bool {
        self.vectors.iter().any(|v| v.id == id)
    }
    
    fn size(&self) -> usize {
        self.vectors.len()
    }
    
    fn dimension(&self) -> usize {
        self.dimension
    }
    
    fn get_config(&self) -> IndexConfig {
        self.config.clone()
    }
    
    fn serialize(&self) -> Result<Vec<u8>> {
        bincode::serialize(self)
            .map_err(|e| Error::vector(format!("Failed to serialize KMeansIndex: {}", e)))
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let deserialized: KMeansIndex = bincode::deserialize(data)
            .map_err(|e| Error::vector(format!("Failed to deserialize KMeansIndex: {}", e)))?;
        
        self.dimension = deserialized.dimension;
        self.config = deserialized.config;
        self.vectors = deserialized.vectors;
        self.centroids = deserialized.centroids;
        self.assignments = deserialized.assignments;
        self.kmeans = deserialized.kmeans;
        
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn VectorIndex + Send + Sync> {
        Box::new(self.clone())
    }

    fn deserialize_box(data: &[u8]) -> Result<Box<dyn VectorIndex + Send + Sync>> where Self: Sized {
        let index: KMeansIndex = bincode::deserialize(data)
            .map_err(|e| Error::vector(format!("Failed to deserialize KMeansIndex: {}", e)))?;
        Ok(Box::new(index))
    }

    fn get_all_vectors(&self) -> Result<Vec<crate::vector::index::interfaces::VectorData>> {
        Ok(self.vectors.iter().map(|v| {
            // Convert String ID to u64 (using hash)
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            v.id.hash(&mut hasher);
            let id_u64 = hasher.finish();
            
            // Convert VectorMetadata to HashMap<String, String>
            let metadata = v.metadata.as_ref().map(|m| {
                let mut map = std::collections::HashMap::new();
                for (k, v) in &m.properties {
                    if let Some(s) = v.as_str() {
                        map.insert(k.clone(), s.to_string());
                    } else {
                        map.insert(k.clone(), v.to_string());
                    }
                }
                map
            });
            
            crate::vector::index::interfaces::VectorData {
                id: id_u64,
                vector: v.data.clone(),
                metadata,
            }
        }).collect())
    }
    
    /// 线性搜索 - 当聚类未构建时使用
    fn linear_search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        let mut results = Vec::new();
        
        for vector in &self.vectors {
            let distance = euclidean_distance(query, &vector.data);
            let metadata = vector.metadata.as_ref().map(|m| {
                let mut props = serde_json::Map::new();
                for (k, v) in &m.properties {
                    props.insert(k.clone(), v.clone());
                }
                serde_json::Value::Object(props)
            });
            
            results.push(SearchResult {
                id: vector.id.clone(),
                distance: -distance as f32, // Negative distance as score, larger is more similar
                metadata,
            });
        }
        
        // 排序并限制结果数量
        results.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap_or(std::cmp::Ordering::Equal));
        if results.len() > limit {
            results.truncate(limit);
        }
        
        Ok(results)
    }
    
    /// 获取内存使用情况
    fn get_memory_usage(&self) -> Result<usize> {
        // 粗略估计内存使用
        let vectors_size = self.vectors.len() * self.dimension * std::mem::size_of::<f32>();
        let centroids_size = self.centroids.len() * self.dimension * std::mem::size_of::<f32>();
        let assignments_size = self.assignments.len() * std::mem::size_of::<usize>();
        
        Ok(vectors_size + centroids_size + assignments_size)
    }
    
    /// 批量插入向量
    fn batch_insert(&mut self, vectors: &[Vector]) -> Result<()> {
        for vector in vectors {
            // 验证向量维度
            if vector.data.len() != self.dimension {
                return Err(Error::vector(
                    format!("Vector dimension mismatch: expected {}, got {}", 
                            self.dimension, vector.data.len())
                ));
            }
            
            // 添加向量
            self.vectors.push(vector.clone());
        }
        
        // 重建聚类
        self.rebuild_clusters()?;
        
        Ok(())
    }
} 