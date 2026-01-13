// 向量并行处理模块
//
// 提供高效的向量并行处理操作，用于大规模向量计算

use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use rayon::prelude::*;
use crate::Result;
use crate::Error;
use crate::vector::types::Vector;
use crate::vector::search::VectorSearchResult;
use crate::vector::core::operations::VectorOps;
use crate::vector::operations::SimilarityMetric;

/// 并行向量处理工具
pub struct ParallelVectorOps;

impl ParallelVectorOps {
    /// 并行计算多个查询向量与目标向量之间的相似度
    /// 
    /// 对于大量的查询向量，这个方法可以显著提高性能
    /// 
    /// # 参数
    /// * `queries` - 查询向量列表
    /// * `target` - 目标向量
    /// * `metric` - 相似度计算方法
    /// 
    /// # 返回
    /// * 相似度列表，与查询向量顺序对应
    pub fn batch_similarity(
        queries: &[Vec<f32>], 
        target: &[f32], 
        metric: SimilarityMetric
    ) -> Result<Vec<f32>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        
        // 检查维度
        for (i, query) in queries.iter().enumerate() {
            if query.len() != target.len() {
                return Err(Error::vector(format!(
                    "Query vector at index {} has incorrect dimension: expected {}, got {}",
                    i, target.len(), query.len()
                )));
            }
        }
        
        // 使用rayon并行计算相似度
        let similarities: Vec<f32> = queries.par_iter()
            .map(|query| VectorOps::compute_similarity(query, target, metric))
            .collect();
        
        Ok(similarities)
    }
    
    /// 并行计算多个向量对之间的相似度
    /// 
    /// # 参数
    /// * `vector_pairs` - 向量对列表，每个元素是(向量1，向量2)
    /// * `metric` - 相似度计算方法
    /// 
    /// # 返回
    /// * 相似度列表，与向量对顺序对应
    pub fn batch_pairwise_similarity(
        vector_pairs: &[(&[f32], &[f32])], 
        metric: SimilarityMetric
    ) -> Result<Vec<f32>> {
        if vector_pairs.is_empty() {
            return Ok(Vec::new());
        }
        
        // 检查维度
        for (i, (vec1, vec2)) in vector_pairs.iter().enumerate() {
            if vec1.len() != vec2.len() {
                return Err(Error::vector(format!(
                    "Vector pair at index {} has mismatched dimensions: {} vs {}",
                    i, vec1.len(), vec2.len()
                )));
            }
        }
        
        // 使用rayon并行计算相似度
        let similarities: Vec<f32> = vector_pairs.par_iter()
            .map(|(vec1, vec2)| VectorOps::compute_similarity(vec1, vec2, metric))
            .collect();
        
        Ok(similarities)
    }
    
    /// 并行归一化多个向量
    /// 
    /// # 参数
    /// * `vectors` - 需要归一化的向量列表
    /// 
    /// # 返回
    /// * 归一化后的向量列表
    pub fn batch_normalize(vectors: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
        
        let result: Vec<Vec<f32>> = vectors.par_iter()
            .map(|vec| {
                let mut vec_copy = vec.clone();
                
                // 计算范数
                let norm: f32 = vec_copy.iter().map(|x| x * x).sum::<f32>().sqrt();
                
                if norm > 1e-6 {
                    // 归一化
                    for val in &mut vec_copy {
                        *val /= norm;
                    }
                }
                
                vec_copy
            })
            .collect();
        
        Ok(result)
    }
    
    /// 并行批量向量检索
    /// 
    /// 对一组查询向量同时执行检索，适合大批量查询场景
    /// 
    /// # 参数
    /// * `queries` - 查询向量列表
    /// * `vectors` - 目标向量集合
    /// * `k` - 每个查询返回的最近邻数量
    /// * `metric` - 相似度计算方法
    /// 
    /// # 返回
    /// * 检索结果列表，每个查询对应一个结果列表
    pub fn batch_search(
        queries: &[Vec<f32>],
        vectors: &HashMap<String, Vector>,
        k: usize,
        metric: SimilarityMetric
    ) -> Result<Vec<Vec<VectorSearchResult>>> {
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        
        // 使用线程池并行处理多个查询
        let results: Vec<Vec<VectorSearchResult>> = queries.par_iter()
            .map(|query| {
                // 为每个查询计算与所有向量的相似度
                let mut search_results: Vec<VectorSearchResult> = vectors.par_iter()
                    .map(|(id, vector)| {
                        let score = VectorOps::compute_similarity(query, &vector.data, metric);
                        
                        VectorSearchResult {
                            id: id.clone(),
                            score,
                            metadata: None,
                            vector: None,
                        }
                    })
                    .collect();
                
                // 根据相似度排序
                search_results.par_sort_by(|a, b| {
                    b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
                });
                
                // 只保留前k个结果
                if search_results.len() > k {
                    search_results.truncate(k);
                }
                
                search_results
            })
            .collect();
        
        Ok(results)
    }
    
    /// 多线程并行向量处理
    /// 
    /// 将输入向量集合分成多个分块，并行处理后再合并结果
    /// 
    /// # 参数
    /// * `vectors` - 输入向量列表
    /// * `chunk_size` - 每个线程处理的向量数量
    /// * `processor` - 向量处理函数
    /// 
    /// # 返回
    /// * 处理后的向量列表
    pub fn parallel_process<F, R>(
        vectors: &[Vec<f32>], 
        chunk_size: usize,
        processor: F
    ) -> Result<Vec<R>> 
    where 
        F: Fn(&[f32]) -> Result<R> + Send + Sync,
        R: Send + 'static,
    {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
        
        // 每个线程至少处理一个向量
        let actual_chunk_size = chunk_size.max(1);
        
        // 使用并行迭代器处理
        let results: Result<Vec<R>> = vectors.par_chunks(actual_chunk_size)
            .map(|chunk| {
                // 处理这个块中的每个向量
                chunk.iter().map(|vec| processor(vec)).collect::<Result<Vec<R>>>()
            })
            .reduce(
                || Ok(Vec::new()),
                |acc, chunk_result| {
                    let mut acc_vec = acc?;
                    let mut chunk_vec = chunk_result?;
                    acc_vec.append(&mut chunk_vec);
                    Ok(acc_vec)
                }
            );
        
        results
    }
    
    /// 并行计算向量的统计信息
    /// 
    /// 对大量向量计算统计特征，如平均值、标准差等
    /// 
    /// # 参数
    /// * `vectors` - 输入向量列表
    /// 
    /// # 返回
    /// * 向量统计信息（均值向量和标准差向量）
    pub fn compute_statistics(vectors: &[Vec<f32>]) -> Result<(Vec<f32>, Vec<f32>)> {
        if vectors.is_empty() {
            return Err(Error::vector("Cannot compute statistics for empty vector list".to_string()));
        }
        
        let dimension = vectors[0].len();
        
        // 检查所有向量维度是否一致
        for (i, vec) in vectors.iter().enumerate().skip(1) {
            if vec.len() != dimension {
                return Err(Error::vector(format!(
                    "Vector at index {} has incorrect dimension: expected {}, got {}",
                    i, dimension, vec.len()
                )));
            }
        }
        
        // 初始化结果
        let mut mean = vec![0.0; dimension];
        let mut variance = vec![0.0; dimension];
        
        // 计算均值
        let count = vectors.len() as f32;
        
        for vec in vectors {
            for (i, &val) in vec.iter().enumerate() {
                mean[i] += val / count;
            }
        }
        
        // 计算方差
        for vec in vectors {
            for (i, &val) in vec.iter().enumerate() {
                let diff = val - mean[i];
                variance[i] += (diff * diff) / count;
            }
        }
        
        // 计算标准差
        let std_dev: Vec<f32> = variance.iter().map(|&var| var.sqrt()).collect();
        
        Ok((mean, std_dev))
    }

    /// 并行K-means向量聚类
    /// 
    /// 使用并行化的K-means算法对向量集合进行聚类
    /// 
    /// # 参数
    /// * `vectors` - 输入向量集合
    /// * `k` - 聚类数量
    /// * `max_iterations` - 最大迭代次数
    /// * `metric` - 相似度度量方式
    /// * `convergence_threshold` - 收敛阈值
    /// 
    /// # 返回
    /// * 聚类结果: (类别中心点列表, 每个向量的类别)
    pub fn kmeans_clustering(
        vectors: &[Vec<f32>],
        k: usize,
        max_iterations: usize,
        metric: SimilarityMetric,
        convergence_threshold: f32
    ) -> Result<(Vec<Vec<f32>>, Vec<usize>)> {
        if vectors.is_empty() {
            return Err(Error::vector("Cannot cluster empty vector list".to_string()));
        }
        
        let num_vectors = vectors.len();
        if k == 0 || k > num_vectors {
            return Err(Error::vector(format!(
                "Invalid cluster count k={}, should be between 1 and {}", k, num_vectors
            )));
        }
        
        let dim = vectors[0].len();
        
        // 检查所有向量维度是否一致
        for (i, vec) in vectors.iter().enumerate().skip(1) {
            if vec.len() != dim {
                return Err(Error::vector(format!(
                    "Vector at index {} has incorrect dimension: expected {}, got {}",
                    i, dim, vec.len()
                )));
            }
        }
        
        // 随机初始化中心点
        let mut centroids = initialize_centroids(vectors, k)?;
        let mut assignments = vec![0; num_vectors];
        let mut prev_assignments = vec![0; num_vectors];
        let mut converged = false;
        let mut iteration = 0;
        
        // 主要迭代循环
        while !converged && iteration < max_iterations {
            // 记录上一次的分配结果
            prev_assignments.copy_from_slice(&assignments);
            
            // 1. 分配阶段: 并行分配每个向量到最近的中心点
            let assignments_arc = Arc::new(Mutex::new(assignments.clone()));
            
            vectors.par_iter().enumerate().for_each(|(i, vec)| {
                let mut best_cluster = 0;
                let mut best_similarity = f32::NEG_INFINITY;
                
                // 找到最相似的中心点
                for (j, centroid) in centroids.iter().enumerate() {
                    let similarity = VectorOps::compute_similarity(vec, centroid, metric);
                    if similarity > best_similarity {
                        best_similarity = similarity;
                        best_cluster = j;
                    }
                }
                
                // 更新分配
                if let Ok(mut assignments) = assignments_arc.lock() {
                    assignments[i] = best_cluster;
                }
            });
            
            // 获取更新后的分配
            assignments = Arc::try_unwrap(assignments_arc)
                .map_err(|_| Error::vector("Failed to unwrap assignments".to_string()))? 
                .into_inner()
                .map_err(|_| Error::vector("Failed to get inner value".to_string()))?;
            
            // 2. 更新阶段: 计算新的中心点
            let mut new_centroids = vec![vec![0.0; dim]; k];
            let mut cluster_sizes = vec![0; k];
            
            // 累加每个集群中的向量
            for (i, &cluster) in assignments.iter().enumerate() {
                cluster_sizes[cluster] += 1;
                for (j, &val) in vectors[i].iter().enumerate() {
                    new_centroids[cluster][j] += val;
                }
            }
            
            // 计算新的中心点
            let mut max_centroid_shift: f32 = 0.0;
            
            for (i, centroid) in new_centroids.iter_mut().enumerate() {
                let size = cluster_sizes[i];
                
                if size > 0 {
                    // 更新中心点为平均值
                    for val in centroid.iter_mut() {
                        *val /= size as f32;
                    }
                    
                    // 计算中心点移动距离
                    let similarity = VectorOps::compute_similarity(centroid, &centroids[i], metric);
                    let shift = 1.0 - similarity; // 相似度转为距离
                    max_centroid_shift = max_centroid_shift.max(shift);
                } else {
                    // 如果集群为空，选择一个随机向量作为新中心点
                    let random_index = rand::random::<usize>() % num_vectors;
                    centroid.copy_from_slice(&vectors[random_index]);
                }
            }
            
            // 更新中心点
            centroids = new_centroids;
            
            // 检查收敛：如果分配不变或中心点移动量低于阈值
            let assignments_changed = assignments != prev_assignments;
            converged = !assignments_changed || max_centroid_shift < convergence_threshold;
            
            iteration += 1;
        }
        
        Ok((centroids, assignments))
    }
    
    /// 并行计算相似度矩阵
    /// 
    /// 计算向量集合中所有向量对之间的相似度矩阵
    /// 
    /// # 参数
    /// * `vectors` - 输入向量列表
    /// * `metric` - 相似度计算方法
    /// 
    /// # 返回
    /// * 相似度矩阵，矩阵[i][j]表示向量i和j之间的相似度
    pub fn similarity_matrix(
        vectors: &[Vec<f32>],
        metric: SimilarityMetric
    ) -> Result<Vec<Vec<f32>>> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
        
        let n = vectors.len();
        let dim = vectors[0].len();
        
        // 检查所有向量维度是否一致
        for (i, vec) in vectors.iter().enumerate().skip(1) {
            if vec.len() != dim {
                return Err(Error::vector(format!(
                    "Vector at index {} has incorrect dimension: expected {}, got {}",
                    i, dim, vec.len()
                )));
            }
        }
        
        // 创建结果矩阵
        let matrix = Arc::new(Mutex::new(vec![vec![0.0; n]; n]));
        
        // 并行计算相似度矩阵的上三角部分
        (0..n).into_par_iter().for_each(|i| {
            let vec_i = &vectors[i];
            
            // 只计算上三角矩阵，包括对角线
            for j in i..n {
                let vec_j = &vectors[j];
                let similarity = VectorOps::compute_similarity(vec_i, vec_j, metric);
                
                // 更新矩阵
                if let Ok(mut matrix_guard) = matrix.lock() {
                    matrix_guard[i][j] = similarity;
                    if i != j {
                        matrix_guard[j][i] = similarity; // 矩阵是对称的
                    }
                }
            }
        });
        
        // 提取结果
        let result = Arc::try_unwrap(matrix)
            .map_err(|_| Error::vector("Failed to unwrap similarity matrix".to_string()))?
            .into_inner()
            .map_err(|_| Error::vector("Failed to get inner value".to_string()))?;
        
        Ok(result)
    }
    
    /// 并行层次聚类
    /// 
    /// 使用层次聚类算法对向量进行分组
    /// 
    /// # 参数
    /// * `vectors` - 输入向量列表
    /// * `num_clusters` - 目标聚类数量
    /// * `metric` - 相似度计算方法
    /// * `linkage` - 链接方法
    /// 
    /// # 返回
    /// * 聚类结果，每个向量的类别
    pub fn hierarchical_clustering(
        vectors: &[Vec<f32>],
        num_clusters: usize,
        metric: SimilarityMetric,
        linkage: LinkageMethod
    ) -> Result<Vec<usize>> {
        if vectors.is_empty() {
            return Err(Error::vector("Cannot cluster empty vector list".to_string()));
        }
        
        if num_clusters == 0 || num_clusters > vectors.len() {
            return Err(Error::vector(format!(
                "Invalid cluster count: {}, should be between 1 and {}", 
                num_clusters, vectors.len()
            )));
        }
        
        // 1. 计算相似度矩阵
        let sim_matrix = Self::similarity_matrix(vectors, metric)?;
        
        // 2. 初始化集群，每个向量都是一个集群
        let n = vectors.len();
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut assignments = (0..n).collect::<Vec<_>>();
        
        // 3. 执行凝聚层次聚类
        while clusters.len() > num_clusters {
            // 找到最相似的两个集群
            let (cluster1, cluster2, _) = find_closest_clusters(&clusters, &sim_matrix, linkage)?;
            
            // 合并集群
            merge_clusters(&mut clusters, &mut assignments, cluster1, cluster2);
        }
        
        Ok(assignments)
    }
}

/// 层次聚类的链接方法
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LinkageMethod {
    /// 单链接 - 两个集群中最相似的一对点的相似度
    Single,
    /// 完全链接 - 两个集群中最不相似的一对点的相似度
    Complete,
    /// 平均链接 - 两个集群所有点对相似度的平均
    Average,
    /// 重心链接 - 两个集群重心之间的相似度
    Centroid,
}

// 初始化K-means中心点
fn initialize_centroids(vectors: &[Vec<f32>], k: usize) -> Result<Vec<Vec<f32>>> {
    use rand::seq::SliceRandom;
    
    let mut rng = rand::thread_rng();
    let n = vectors.len();
    
    // 使用k-means++算法选择初始点
    let mut centroids = Vec::with_capacity(k);
    let mut indices = Vec::with_capacity(k);
    
    // 随机选择第一个中心点
    let first_idx = rand::random::<usize>() % n;
    centroids.push(vectors[first_idx].clone());
    indices.push(first_idx);
    
    // 选择剩余的中心点
    for _ in 1..k {
        // 计算每个点到最近中心点的距离
        let distances: Vec<f32> = vectors.par_iter().enumerate()
            .map(|(i, vec)| {
                if indices.contains(&i) {
                    return 0.0; // 已经是中心点
                }
                
                // 计算到最近中心点的距离
                let mut min_dist = f32::MAX;
                for centroid in &centroids {
                    // 使用欧氏距离平方
                    let dist = vec.iter().zip(centroid.iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f32>();
                    min_dist = min_dist.min(dist);
                }
                min_dist
            })
            .collect();
        
        // 按距离加权选择下一个中心点
        let total_dist: f32 = distances.iter().sum();
        if total_dist > 0.0 {
            let threshold = rand::random::<f32>() * total_dist;
            let mut cumsum = 0.0;
            let mut selected_idx = 0;
            
            for (i, &dist) in distances.iter().enumerate() {
                if indices.contains(&i) {
                    continue;
                }
                cumsum += dist;
                if cumsum >= threshold {
                    selected_idx = i;
                    break;
                }
            }
            
            centroids.push(vectors[selected_idx].clone());
            indices.push(selected_idx);
        } else {
            // 如果所有点都太近，随机选择
            let remaining: Vec<usize> = (0..n).filter(|i| !indices.contains(i)).collect();
            if !remaining.is_empty() {
                let idx = *remaining.choose(&mut rng).unwrap();
                centroids.push(vectors[idx].clone());
                indices.push(idx);
            } else {
                // 如果没有剩余点，随机扰动已有中心点
                let idx = rand::random::<usize>() % centroids.len();
                let mut new_centroid = centroids[idx].clone();
                for val in &mut new_centroid {
                    // 添加小噪声
                    *val += (rand::random::<f32>() - 0.5) * 0.01;
                }
                centroids.push(new_centroid);
                indices.push(idx); // 注意这不是向量索引，只是为了占位
            }
        }
    }
    
    Ok(centroids)
}

// 寻找最相似的两个集群
fn find_closest_clusters(
    clusters: &[Vec<usize>],
    sim_matrix: &[Vec<f32>],
    linkage: LinkageMethod
) -> Result<(usize, usize, f32)> {
    let n = clusters.len();
    let mut best_i = 0;
    let mut best_j = 0;
    let mut best_similarity = f32::NEG_INFINITY;
    
    for i in 0..n-1 {
        for j in i+1..n {
            let similarity = calculate_cluster_similarity(
                &clusters[i], &clusters[j], sim_matrix, linkage);
            
            if similarity > best_similarity {
                best_similarity = similarity;
                best_i = i;
                best_j = j;
            }
        }
    }
    
    Ok((best_i, best_j, best_similarity))
}

// 计算两个集群之间的相似度
fn calculate_cluster_similarity(
    cluster1: &[usize],
    cluster2: &[usize],
    sim_matrix: &[Vec<f32>],
    linkage: LinkageMethod
) -> f32 {
    match linkage {
        LinkageMethod::Single => {
            // 单链接：最大相似度
            let mut max_sim = f32::NEG_INFINITY;
            for &i in cluster1 {
                for &j in cluster2 {
                    max_sim = max_sim.max(sim_matrix[i][j]);
                }
            }
            max_sim
        },
        LinkageMethod::Complete => {
            // 完全链接：最小相似度
            let mut min_sim = f32::INFINITY;
            for &i in cluster1 {
                for &j in cluster2 {
                    min_sim = min_sim.min(sim_matrix[i][j]);
                }
            }
            min_sim
        },
        LinkageMethod::Average => {
            // 平均链接：平均相似度
            let mut sum_sim = 0.0;
            let mut count = 0;
            for &i in cluster1 {
                for &j in cluster2 {
                    sum_sim += sim_matrix[i][j];
                    count += 1;
                }
            }
            if count > 0 {
                sum_sim / count as f32
            } else {
                f32::NEG_INFINITY
            }
        },
        LinkageMethod::Centroid => {
            // 重心链接：所有点对的平均相似度
            let mut sum_sim = 0.0;
            let mut count = 0;
            for &i in cluster1 {
                for &j in cluster2 {
                    sum_sim += sim_matrix[i][j];
                    count += 1;
                }
            }
            if count > 0 {
                sum_sim / count as f32
            } else {
                f32::NEG_INFINITY
            }
        },
    }
}

// 合并两个集群
fn merge_clusters(
    clusters: &mut Vec<Vec<usize>>,
    assignments: &mut Vec<usize>,
    i: usize,
    j: usize
) {
    // 获取要合并的集群
    let cluster_j = clusters.remove(j);
    let cluster_i = &mut clusters[i];
    
    // 更新分配
    for &idx in &cluster_j {
        assignments[idx] = i;
    }
    
    // 合并集群
    cluster_i.extend(cluster_j);
    
    // 更新大于j的集群索引
    for idx in 0..assignments.len() {
        if assignments[idx] > j {
            assignments[idx] -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_batch_similarity() {
        let queries = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let target = vec![1.0, 1.0, 1.0];
        
        let similarities = ParallelVectorOps::batch_similarity(&queries, &target, SimilarityMetric::Cosine).unwrap();
        
        assert_eq!(similarities.len(), 3);
        
        // 根据余弦相似度公式，这些向量与[1,1,1]的相似度应该都是1/√3
        let expected_similarity = 1.0 / 3.0_f32.sqrt();
        
        for sim in similarities {
            assert!((sim - expected_similarity).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_batch_normalize() {
        let vectors = vec![
            vec![3.0, 4.0],
            vec![5.0, 12.0],
            vec![1.0, 1.0],
        ];
        
        let normalized = ParallelVectorOps::batch_normalize(&vectors).unwrap();
        
        assert_eq!(normalized.len(), 3);
        
        // 检查第一个向量 [3,4] -> [3/5, 4/5]
        assert!((normalized[0][0] - 0.6).abs() < 1e-6);
        assert!((normalized[0][1] - 0.8).abs() < 1e-6);
        
        // 检查第二个向量 [5,12] -> [5/13, 12/13]
        assert!((normalized[1][0] - 5.0/13.0).abs() < 1e-6);
        assert!((normalized[1][1] - 12.0/13.0).abs() < 1e-6);
        
        // 检查第三个向量 [1,1] -> [1/√2, 1/√2]
        assert!((normalized[2][0] - 1.0/2.0_f32.sqrt()).abs() < 1e-6);
        assert!((normalized[2][1] - 1.0/2.0_f32.sqrt()).abs() < 1e-6);
    }
    
    #[test]
    fn test_kmeans_clustering() {
        // 创建测试数据：三个明显的集群
        let vectors = vec![
            // 集群1
            vec![1.0, 1.0],
            vec![1.2, 0.8],
            vec![0.9, 1.1],
            vec![1.1, 0.9],
            
            // 集群2
            vec![4.0, 4.0],
            vec![4.1, 3.9],
            vec![3.9, 4.1],
            vec![4.2, 3.8],
            
            // 集群3
            vec![1.0, 4.0],
            vec![0.9, 4.1],
            vec![1.1, 3.9],
            vec![1.0, 4.2],
        ];
        
        let (centroids, assignments) = ParallelVectorOps::kmeans_clustering(
            &vectors,
            3,  // 3个集群
            100, // 最大迭代次数
            SimilarityMetric::Cosine,
            0.001 // 收敛阈值
        ).unwrap();
        
        // 检查结果
        assert_eq!(centroids.len(), 3);
        assert_eq!(assignments.len(), vectors.len());
        
        // 验证集群1中的向量都被分到同一组
        let cluster_0 = assignments[0];
        for i in 1..4 {
            assert_eq!(assignments[i], cluster_0);
        }
        
        // 验证集群2中的向量都被分到同一组
        let cluster_1 = assignments[4];
        for i in 5..8 {
            assert_eq!(assignments[i], cluster_1);
        }
        
        // 验证集群3中的向量都被分到同一组
        let cluster_2 = assignments[8];
        for i in 9..12 {
            assert_eq!(assignments[i], cluster_2);
        }
        
        // 验证三个集群是不同的
        assert!(cluster_0 != cluster_1);
        assert!(cluster_0 != cluster_2);
        assert!(cluster_1 != cluster_2);
    }
    
    #[test]
    fn test_similarity_matrix() {
        let vectors = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        
        let matrix = ParallelVectorOps::similarity_matrix(&vectors, SimilarityMetric::Cosine).unwrap();
        
        // 矩阵大小检查
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);
        
        // 对角线上的元素应该是1.0（与自己的相似度）
        assert!((matrix[0][0] - 1.0).abs() < 1e-6);
        assert!((matrix[1][1] - 1.0).abs() < 1e-6);
        assert!((matrix[2][2] - 1.0).abs() < 1e-6);
        
        // 检查余弦相似度
        assert!((matrix[0][1] - 0.0).abs() < 1e-6); // [1,0]和[0,1]正交
        assert!((matrix[0][2] - 1.0/2.0_f32.sqrt()).abs() < 1e-6); // [1,0]和[1,1]
        assert!((matrix[1][2] - 1.0/2.0_f32.sqrt()).abs() < 1e-6); // [0,1]和[1,1]
        
        // 验证矩阵对称性
        assert!((matrix[1][0] - matrix[0][1]).abs() < 1e-6);
        assert!((matrix[2][0] - matrix[0][2]).abs() < 1e-6);
        assert!((matrix[2][1] - matrix[1][2]).abs() < 1e-6);
    }
    
    #[test]
    fn test_hierarchical_clustering() {
        // 创建测试数据：两个明显的集群
        let vectors = vec![
            // 集群1
            vec![1.0, 1.0],
            vec![1.2, 0.8],
            vec![0.9, 1.1],
            
            // 集群2
            vec![4.0, 4.0],
            vec![4.1, 3.9],
            vec![3.9, 4.1],
        ];
        
        let assignments = ParallelVectorOps::hierarchical_clustering(
            &vectors,
            2, // 2个集群
            SimilarityMetric::Cosine,
            LinkageMethod::Average
        ).unwrap();
        
        // 检查结果
        assert_eq!(assignments.len(), vectors.len());
        
        // 验证集群1中的向量都被分到同一组
        let cluster_0 = assignments[0];
        assert_eq!(assignments[1], cluster_0);
        assert_eq!(assignments[2], cluster_0);
        
        // 验证集群2中的向量都被分到同一组
        let cluster_1 = assignments[3];
        assert_eq!(assignments[4], cluster_1);
        assert_eq!(assignments[5], cluster_1);
        
        // 验证两个集群是不同的
        assert!(cluster_0 != cluster_1);
    }
} 