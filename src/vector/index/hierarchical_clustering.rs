use crate::vector::index::interfaces::VectorIndex;
use crate::vector::index::types::{IndexConfig, SearchResult};
use crate::vector::{Vector, SimilarityMetric};
use crate::Result;
use crate::Error;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};
use uuid::Uuid;

/// 层次聚类索引
/// 
/// 使用层次聚类算法构建多层次的向量索引结构
/// 通过自底向上的聚类方式组织向量，提供高效的近似搜索
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalClusteringIndex {
    /// 配置参数
    config: IndexConfig,
    /// 向量存储
    vectors: Vec<Vector>,
    /// 聚类层次结构
    hierarchy: Vec<ClusterLevel>,
    /// 向量维度
    dimension: usize,
    /// 是否已构建索引
    is_built: bool,
    /// 聚类层数
    levels: usize,
    /// 每层的聚类数量
    clusters_per_level: usize,
}

/// 聚类层级
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClusterLevel {
    /// 当前层级
    level: usize,
    /// 聚类列表
    clusters: Vec<Cluster>,
}

/// 聚类结构
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Cluster {
    /// 聚类ID
    id: usize,
    /// 聚类中心
    centroid: Vector,
    /// 包含的向量ID
    vector_ids: Vec<usize>,
    /// 子聚类ID（下一层）
    child_clusters: Vec<usize>,
    /// 父聚类ID（上一层）
    parent_cluster: Option<usize>,
    /// 聚类半径
    radius: f32,
}

/// 搜索候选项
#[derive(Debug, Clone)]
struct SearchCandidate {
    cluster_id: usize,
    level: usize,
    distance_to_centroid: f32,
}

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance_to_centroid == other.distance_to_centroid
    }
}

impl Eq for SearchCandidate {}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance_to_centroid.partial_cmp(&other.distance_to_centroid)
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl HierarchicalClusteringIndex {
    /// 创建新的层次聚类索引
    pub fn new(config: IndexConfig) -> Result<Self> {
        let levels = if config.cluster_levels > 0 {
            config.cluster_levels
        } else {
            // 根据期望的向量数量动态调整层数
            let base_levels = 3;
            let scale_factor = (config.expected_elements as f32).log10();
            (base_levels as f32 * scale_factor / 2.0).round() as usize
        };

        let clusters_per_level = if config.clusters_per_level > 0 {
            config.clusters_per_level
        } else {
            // 根据维度和期望元素数量计算
            ((config.expected_elements as f32).sqrt() / levels as f32).round() as usize
        };

        Ok(Self {
            dimension: config.dimension,
            levels: levels.max(2).min(10),
            clusters_per_level: clusters_per_level.max(2).min(1000),
            config,
            vectors: Vec::new(),
            hierarchy: Vec::new(),
            is_built: false,
        })
    }

    /// 构建层次聚类
    fn build_hierarchy(&mut self) -> Result<()> {
        if self.vectors.is_empty() {
            return Ok(());
        }

        self.hierarchy.clear();

        // 第0层：每个向量作为单独的聚类
        let mut current_clusters = self.create_initial_clusters()?;
        self.hierarchy.push(ClusterLevel {
            level: 0,
            clusters: current_clusters.clone(),
        });

        // 逐层聚类
        for level in 1..self.levels {
            if current_clusters.len() <= 1 {
                break;
            }

            let prev_len = current_clusters.len();
            let new_clusters = self.merge_clusters(current_clusters, level)?;

            if new_clusters.len() == prev_len {
                // 无法进一步聚类，停止
                break;
            }

            self.hierarchy.push(ClusterLevel {
                level,
                clusters: new_clusters.clone(),
            });

            current_clusters = new_clusters;
        }

        Ok(())
    }

    /// 创建初始聚类（每个向量一个聚类）
    fn create_initial_clusters(&self) -> Result<Vec<Cluster>> {
        let mut clusters = Vec::new();
        
        for (i, vector) in self.vectors.iter().enumerate() {
            if vector.data.is_empty() {
                continue; // 跳过已删除的向量
            }

            clusters.push(Cluster {
                id: i,
                centroid: vector.clone(),
                vector_ids: vec![i],
                child_clusters: Vec::new(),
                parent_cluster: None,
                radius: 0.0,
            });
        }

        Ok(clusters)
    }

    /// 合并聚类到下一层
    fn merge_clusters(&self, current_clusters: Vec<Cluster>, level: usize) -> Result<Vec<Cluster>> {
        let target_count = (current_clusters.len() as f32 / 2.0).ceil() as usize;
        let target_count = target_count.max(1).min(self.clusters_per_level);

        // 使用k-means聚类算法进行聚类
        let centroids = self.initialize_centroids(&current_clusters, target_count)?;
        let cluster_assignments = self.assign_clusters_to_centroids(&current_clusters, &centroids)?;
        
        let mut new_clusters = Vec::new();
        
        for (cluster_id, assigned_clusters) in cluster_assignments.iter().enumerate() {
            if assigned_clusters.is_empty() {
                continue;
            }

            // 计算新的质心
            let new_centroid = self.calculate_merged_centroid(assigned_clusters)?;
            
            // 收集所有向量ID
            let mut all_vector_ids = Vec::new();
            let mut child_cluster_ids = Vec::new();
            
            for &old_cluster_idx in assigned_clusters {
                all_vector_ids.extend(&current_clusters[old_cluster_idx].vector_ids);
                child_cluster_ids.push(current_clusters[old_cluster_idx].id);
            }

            // 计算聚类半径
            let radius = self.calculate_cluster_radius(&new_centroid, &all_vector_ids)?;

            new_clusters.push(Cluster {
                id: cluster_id,
                centroid: new_centroid,
                vector_ids: all_vector_ids,
                child_clusters: child_cluster_ids,
                parent_cluster: None,
                radius,
            });
        }

        // 设置父子关系
        for (parent_id, assigned_clusters) in cluster_assignments.iter().enumerate() {
            for &child_idx in assigned_clusters {
                // 这里需要更新原聚类的父聚类信息，但由于我们返回新的聚类，
                // 父子关系信息主要用于搜索时的导航
            }
        }

        Ok(new_clusters)
    }

    /// 初始化质心
    fn initialize_centroids(&self, clusters: &[Cluster], k: usize) -> Result<Vec<Vector>> {
        if clusters.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let mut centroids = Vec::new();
        
        // 使用k-means++初始化方法
        // 1. 随机选择第一个质心
        centroids.push(clusters[0].centroid.clone());
        
        // 2. 依次选择距离已有质心最远的点作为新质心
        while centroids.len() < k && centroids.len() < clusters.len() {
            let mut max_min_distance = 0.0;
            let mut best_candidate = 0;
            
            for (i, cluster) in clusters.iter().enumerate() {
                // 计算到最近质心的距离
                let mut min_distance = f32::MAX;
                for centroid in &centroids {
                    let distance = self.calculate_distance(&cluster.centroid, centroid)?;
                    min_distance = min_distance.min(distance);
                }
                
                if min_distance > max_min_distance {
                    max_min_distance = min_distance;
                    best_candidate = i;
                }
            }
            
            centroids.push(clusters[best_candidate].centroid.clone());
        }

        Ok(centroids)
    }

    /// 将聚类分配给质心
    fn assign_clusters_to_centroids(&self, clusters: &[Cluster], centroids: &[Vector]) -> Result<Vec<Vec<usize>>> {
        let mut assignments = vec![Vec::new(); centroids.len()];
        
        for (cluster_idx, cluster) in clusters.iter().enumerate() {
            let mut best_centroid = 0;
            let mut min_distance = f32::MAX;
            
            for (centroid_idx, centroid) in centroids.iter().enumerate() {
                let distance = self.calculate_distance(&cluster.centroid, centroid)?;
                if distance < min_distance {
                    min_distance = distance;
                    best_centroid = centroid_idx;
                }
            }
            
            assignments[best_centroid].push(cluster_idx);
        }

        Ok(assignments)
    }

    /// 计算合并后的质心
    fn calculate_merged_centroid(&self, clusters: &[usize]) -> Result<Vector> {
        if clusters.is_empty() {
            return Err(Error::vector("无法计算空聚类的质心".to_string()));
        }

        let first_cluster = &self.hierarchy.last().unwrap().clusters[clusters[0]];
        let mut centroid_data = vec![0.0; self.dimension];
        let mut total_weight = 0.0;

        for &cluster_idx in clusters {
            let cluster = &self.hierarchy.last().unwrap().clusters[cluster_idx];
            let weight = cluster.vector_ids.len() as f32;
            
            for (i, &value) in cluster.centroid.data.iter().enumerate() {
                centroid_data[i] += value * weight;
            }
            total_weight += weight;
        }

        // 归一化
        for value in &mut centroid_data {
            *value /= total_weight;
        }

        Ok(Vector {
            id: format!("centroid_{}", Uuid::new_v4()), // 质心使用生成的ID
            data: centroid_data,
            metadata: None,
        })
    }

    /// 计算聚类半径
    fn calculate_cluster_radius(&self, centroid: &Vector, vector_ids: &[usize]) -> Result<f32> {
        let mut max_distance: f32 = 0.0;
        
        for &vector_id in vector_ids {
            if vector_id < self.vectors.len() {
                let distance = self.calculate_distance(centroid, &self.vectors[vector_id])?;
                max_distance = max_distance.max(distance);
            }
        }

        Ok(max_distance)
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
            Manhattan => {
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
            Manhattan => {
                let distance = vector.data.iter()
                    .zip(query.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>();
                Ok(distance)
            },
        }
    }

    /// 层次搜索
    fn hierarchical_search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if self.hierarchy.is_empty() {
            return Ok(Vec::new());
        }

        // 从最顶层开始搜索
        let top_level = self.hierarchy.len() - 1;
        let mut candidates = BinaryHeap::new();

        // 在顶层找到最相似的聚类
        for cluster in &self.hierarchy[top_level].clusters {
            let distance = self.calculate_distance_to_slice(&cluster.centroid, query)?;
            candidates.push(SearchCandidate {
                cluster_id: cluster.id,
                level: top_level,
                distance_to_centroid: distance,
            });
        }

        // 向下层层搜索
        let mut final_candidates = Vec::new();
        let search_width = (k * 2).max(10); // 搜索宽度

        for level in (0..self.hierarchy.len()).rev() {
            let mut next_candidates = BinaryHeap::new();
            let mut processed = 0;

            while let Some(candidate) = candidates.pop() {
                if candidate.level != level {
                    continue;
                }

                processed += 1;
                if processed > search_width {
                    break;
                }

                if level == 0 {
                    // 最底层，收集实际向量
                    let cluster = &self.hierarchy[level].clusters
                        .iter()
                        .find(|c| c.id == candidate.cluster_id)
                        .ok_or_else(|| Error::vector("聚类不存在".to_string()))?;

                    for &vector_id in &cluster.vector_ids {
                        if vector_id < self.vectors.len() && !self.vectors[vector_id].data.is_empty() {
                            let distance = self.calculate_distance_to_slice(&self.vectors[vector_id], query)?;
                            final_candidates.push((vector_id, distance));
                        }
                    }
                } else {
                    // 中间层，搜索子聚类
                    let cluster = &self.hierarchy[level].clusters
                        .iter()
                        .find(|c| c.id == candidate.cluster_id)
                        .ok_or_else(|| Error::vector("聚类不存在".to_string()))?;

                    // 在下一层找到相关的聚类
                    if level > 0 {
                        for child_cluster_id in &cluster.child_clusters {
                            if let Some(child_cluster) = self.hierarchy[level - 1].clusters
                                .iter()
                                .find(|c| c.id == *child_cluster_id) {
                                let distance = self.calculate_distance_to_slice(&child_cluster.centroid, query)?;
                                next_candidates.push(SearchCandidate {
                                    cluster_id: child_cluster.id,
                                    level: level - 1,
                                    distance_to_centroid: distance,
                                });
                            }
                        }
                    }
                }
            }

            candidates = next_candidates;
        }

        // 排序并返回最佳结果
        final_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        
        let mut results = Vec::new();
        for (vector_id, distance) in final_candidates.into_iter().take(k) {
            let metadata = self.vectors[vector_id].metadata.as_ref()
                .map(|m| serde_json::to_value(m).unwrap_or(serde_json::Value::Null));
            results.push(SearchResult {
                id: vector_id.to_string(),
                distance,
                metadata: metadata,
            });
        }

        Ok(results)
    }

    /// 构建索引
    pub fn build(&mut self) -> Result<()> {
        if self.vectors.is_empty() {
            return Err(Error::vector("无法为空向量集构建索引".to_string()));
        }

        self.build_hierarchy()?;
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

        // 标记为已删除
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

impl VectorIndex for HierarchicalClusteringIndex {
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

        self.hierarchical_search(query, limit)
    }

    fn delete(&mut self, id: &str) -> Result<bool> {
        let vector_id: usize = id.parse().map_err(|_| Error::vector("无效的向量ID".to_string()))?;
        
        if vector_id >= self.vectors.len() {
            return Ok(false);
        }

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
        let deserialized: HierarchicalClusteringIndex = bincode::deserialize(data)
            .map_err(|e| Error::vector(format!("反序列化失败: {}", e)))?;
        
        *self = deserialized;
        Ok(())
    }

    fn clone_box(&self) -> Box<dyn VectorIndex + Send + Sync> {
        Box::new(self.clone())
    }

    fn deserialize_box(data: &[u8]) -> Result<Box<dyn VectorIndex + Send + Sync>> where Self: Sized {
        let index: HierarchicalClusteringIndex = bincode::deserialize(data)
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
        self.hierarchy.clear();
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