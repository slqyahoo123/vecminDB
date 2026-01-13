// 向量批处理模块
//
// 提供用于高效批量处理向量数据的功能

use std::sync::{Arc, Mutex, RwLock};
use crossbeam_channel::bounded;
use crate::Result;
use crate::Error;
use crate::vector::operations::SimilarityMetric;
use super::{WorkerPool, ParallelConfig};

/// 向量批处理器
pub struct BatchProcessor {
    /// 工作线程池
    workers: Arc<RwLock<WorkerPool>>,
    /// 配置
    config: ParallelConfig,
}

/// 批处理任务状态
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BatchTaskState {
    /// 等待执行
    Pending,
    /// 正在执行
    Running,
    /// 已完成
    Completed,
    /// 执行失败
    Failed,
    /// 已取消
    Cancelled,
}

/// 批处理任务
pub struct BatchTask<T, R> {
    /// 任务ID
    pub id: String,
    /// 输入数据
    pub input: T,
    /// 输出数据
    pub output: Option<R>,
    /// 任务状态
    pub state: BatchTaskState,
    /// 错误信息
    pub error: Option<String>,
}

impl BatchProcessor {
    /// 创建新的批处理器
    pub fn new(workers: Arc<RwLock<WorkerPool>>, config: ParallelConfig) -> Self {
        Self { workers, config }
    }
    
    /// 批量计算向量相似度
    pub fn batch_similarity(
        &self,
        vectors: &[Vec<f32>],
        queries: &[Vec<f32>],
        metric: SimilarityMetric,
    ) -> Result<Vec<Vec<f32>>> {
        if vectors.is_empty() || queries.is_empty() {
            return Ok(Vec::new());
        }
        
        let vector_dimension = vectors[0].len();
        
        // 验证所有向量维度是否一致
        for (i, vec) in vectors.iter().enumerate() {
            if vec.len() != vector_dimension {
                return Err(Error::vector(format!(
                    "Vector at index {} has incorrect dimension", i
                )));
            }
        }
        
        // 验证所有查询向量维度是否一致
        for (i, query) in queries.iter().enumerate() {
            if query.len() != vector_dimension {
                return Err(Error::vector(format!(
                    "Query vector at index {} has incorrect dimension", i
                )));
            }
        }
        
        let num_vectors = vectors.len();
        let num_queries = queries.len();
        let batch_size = self.config.batch_size;
        
        // 创建结果矩阵
        let results = Arc::new(Mutex::new(vec![vec![0.0; num_vectors]; num_queries]));
        
        // 创建完成信号通道
        let total_tasks = (num_queries + batch_size - 1) / batch_size;
        let (tx, rx) = bounded(total_tasks);
        
        // 分批提交查询
        for query_batch_start in (0..num_queries).step_by(batch_size) {
            let query_batch_end = (query_batch_start + batch_size).min(num_queries);
            let query_batch: Vec<Vec<f32>> = queries[query_batch_start..query_batch_end]
                .iter()
                .cloned()
                .collect();
                
            let vectors_clone = vectors.to_vec();
            let results_clone = Arc::clone(&results);
            let tx_clone = tx.clone();
            
            let workers = self.workers.read().map_err(|_| 
                Error::vector("Failed to acquire read lock on worker pool".to_string()))?;
            
            workers.submit(move || {
                // 为这个批次中的所有查询计算相似度
                for (batch_idx, query) in query_batch.iter().enumerate() {
                    let query_idx = query_batch_start + batch_idx;
                    
                    // 计算这个查询与所有向量的相似度
                    for (vec_idx, vector) in vectors_clone.iter().enumerate() {
                        let similarity = super::super::core::operations::VectorOps::compute_similarity(
                            query, vector, metric
                        );
                        
                        // 存储结果
                        if let Ok(mut results_guard) = results_clone.lock() {
                            results_guard[query_idx][vec_idx] = similarity;
                        }
                    }
                }
                
                // 发送完成信号
                let _ = tx_clone.send(());
            })?;
        }
        
        // 等待所有计算完成
        for _ in 0..total_tasks {
            rx.recv().map_err(|_| Error::vector("Channel error while waiting for tasks to complete".to_string()))?;
        }
        
        // 获取结果
        let result_matrix = results.lock().map_err(|_| 
            Error::vector("Failed to acquire lock on results".to_string()))?;
        
        Ok(result_matrix.clone())
    }
    
    /// 批量归一化向量
    pub fn batch_normalize(
        &self,
        vectors: &mut [Vec<f32>],
    ) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }
        
        let total_vectors = vectors.len();
        let batch_size = self.config.batch_size;
        
        // 创建完成信号通道
        let total_batches = (total_vectors + batch_size - 1) / batch_size;
        let (tx, rx) = bounded(total_batches);
        
        // 分批处理向量
        for batch_start in (0..total_vectors).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(total_vectors);
            
            let tx_clone = tx.clone();
            
            // 获取这个批次的向量切片
            let batch_vectors = unsafe {
                // 安全性：我们确保不同批次不会重叠
                let ptr = vectors.as_mut_ptr().add(batch_start);
                std::slice::from_raw_parts_mut(ptr, batch_end - batch_start)
            };
            
            let workers = self.workers.read().map_err(|_| 
                Error::vector("Failed to acquire read lock on worker pool".to_string()))?;
            
            workers.submit(move || {
                // 归一化这个批次中的所有向量
                for vec in batch_vectors.iter_mut() {
                    let norm = super::super::utils::vector::normalize_vector(vec);
                    let _ = norm; // 忽略归一化结果
                }
                
                // 发送完成信号
                let _ = tx_clone.send(());
            })?;
        }
        
        // 等待所有批次完成
        for _ in 0..total_batches {
            rx.recv().map_err(|_| Error::vector("Channel error while waiting for tasks to complete".to_string()))?;
        }
        
        Ok(())
    }
    
    /// 批量计算向量距离矩阵
    pub fn batch_distance_matrix(
        &self,
        vectors_a: &[Vec<f32>],
        vectors_b: &[Vec<f32>],
        metric: SimilarityMetric,
    ) -> Result<Vec<Vec<f32>>> {
        if vectors_a.is_empty() || vectors_b.is_empty() {
            return Ok(Vec::new());
        }
        
        let dim_a = vectors_a[0].len();
        let dim_b = vectors_b[0].len();
        
        if dim_a != dim_b {
            return Err(Error::vector(format!(
                "Vector dimension mismatch: {} vs {}", dim_a, dim_b
            )));
        }
        
        // 验证所有向量维度是否一致
        for (i, vec) in vectors_a.iter().enumerate() {
            if vec.len() != dim_a {
                return Err(Error::vector(format!(
                    "Vector A at index {} has incorrect dimension", i
                )));
            }
        }
        
        for (i, vec) in vectors_b.iter().enumerate() {
            if vec.len() != dim_b {
                return Err(Error::vector(format!(
                    "Vector B at index {} has incorrect dimension", i
                )));
            }
        }
        
        let a_len = vectors_a.len();
        let b_len = vectors_b.len();
        let batch_size = self.config.batch_size;
        
        // 创建结果矩阵
        let results = Arc::new(Mutex::new(vec![vec![0.0; b_len]; a_len]));
        
        // 创建完成信号通道
        let total_tasks = (a_len * b_len + batch_size * batch_size - 1) / (batch_size * batch_size);
        let (tx, rx) = bounded(total_tasks);
        
        // 分块处理距离矩阵
        for a_start in (0..a_len).step_by(batch_size) {
            let a_end = (a_start + batch_size).min(a_len);
            
            for b_start in (0..b_len).step_by(batch_size) {
                let b_end = (b_start + batch_size).min(b_len);
                
                // 复制这个块需要的数据
                let a_batch: Vec<Vec<f32>> = vectors_a[a_start..a_end].to_vec();
                let b_batch: Vec<Vec<f32>> = vectors_b[b_start..b_end].to_vec();
                
                let results_clone = Arc::clone(&results);
                let tx_clone = tx.clone();
                
                let workers = self.workers.read().map_err(|_| 
                    Error::vector("Failed to acquire read lock on worker pool"))?;
                
                workers.submit(move || {
                    // 计算这个块中所有向量对的距离
                    for (batch_a_idx, vec_a) in a_batch.iter().enumerate() {
                        let a_idx = a_start + batch_a_idx;
                        
                        for (batch_b_idx, vec_b) in b_batch.iter().enumerate() {
                            let b_idx = b_start + batch_b_idx;
                            
                            let similarity = super::super::core::operations::VectorOps::compute_similarity(
                                vec_a, vec_b, metric
                            );
                            
                            // 存储结果
                            if let Ok(mut results_guard) = results_clone.lock() {
                                results_guard[a_idx][b_idx] = similarity;
                            }
                        }
                    }
                    
                    // 发送完成信号
                    let _ = tx_clone.send(());
                })?;
            }
        }
        
        // 等待所有计算完成
        for _ in 0..total_tasks {
            rx.recv().map_err(|_| Error::vector("Channel error while waiting for tasks to complete".to_string()))?;
        }
        
        // 获取结果
        let distance_matrix = results.lock().map_err(|_| 
            Error::vector("Failed to acquire lock on results".to_string()))?;
        
        Ok(distance_matrix.clone())
    }
    
    /// 批量向量转换
    pub fn batch_transform<F>(
        &self,
        vectors: &[Vec<f32>],
        transform_fn: F,
    ) -> Result<Vec<Vec<f32>>> 
    where 
        F: Fn(&[f32]) -> Result<Vec<f32>> + Send + Sync + 'static + Clone,
    {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
        
        let total_vectors = vectors.len();
        let batch_size = self.config.batch_size;
        
        // 创建结果容器
        let results = Arc::new(Mutex::new(vec![Vec::new(); total_vectors]));
        
        // 创建完成信号通道
        let total_batches = (total_vectors + batch_size - 1) / batch_size;
        let (tx, rx) = bounded(total_batches);
        
        // 分批处理向量
        for batch_start in (0..total_vectors).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(total_vectors);
            let batch_vectors = vectors[batch_start..batch_end].to_vec();
            
            let results_clone = Arc::clone(&results);
            let tx_clone = tx.clone();
            let transform_fn = Arc::new(transform_fn.clone());
            
            let workers = self.workers.read().map_err(|_| 
                Error::vector("Failed to acquire read lock on worker pool".to_string()))?;
            
            workers.submit(move || {
                // 转换这个批次中的所有向量
                for (batch_idx, vec) in batch_vectors.iter().enumerate() {
                    let idx = batch_start + batch_idx;
                    
                    match transform_fn(vec) {
                        Ok(transformed) => {
                            // 存储结果
                            if let Ok(mut results_guard) = results_clone.lock() {
                                results_guard[idx] = transformed;
                            }
                        },
                        Err(e) => {
                            // 记录错误但继续处理
                            log::error!("Error transforming vector at index {}: {:?}", idx, e);
                        }
                    }
                }
                
                // 发送完成信号
                let _ = tx_clone.send(());
            })?;
        }
        
        // 等待所有批次完成
        for _ in 0..total_batches {
            rx.recv().map_err(|_| Error::vector("Channel error while waiting for tasks to complete".to_string()))?;
        }
        
        // 获取结果
        let transformed_vectors = results.lock().map_err(|_| 
            Error::vector("Failed to acquire lock on results".to_string()))?;
        
        Ok(transformed_vectors.clone())
    }
    
    /// 批量向量聚类
    pub fn batch_cluster(
        &self,
        vectors: &[Vec<f32>],
        k: usize,
        max_iterations: usize,
        metric: SimilarityMetric,
    ) -> Result<Vec<usize>> {
        // 实现k-means聚类算法
        if vectors.is_empty() {
            return Ok(Vec::new());
        }
        
        if k == 0 || k > vectors.len() {
            return Err(Error::vector(format!(
                "Invalid cluster count: {}, should be between 1 and {}", 
                k, vectors.len()
            )));
        }
        
        let dim = vectors[0].len();
        
        // 验证所有向量维度是否一致
        for (i, vec) in vectors.iter().enumerate() {
            if vec.len() != dim {
                return Err(Error::vector(format!(
                    "Vector at index {} has incorrect dimension", i
                )));
            }
        }
        
        // 随机选择初始聚类中心
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut centroids = Vec::with_capacity(k);
        
        // 保证初始中心点不重复
        let mut indices: Vec<usize> = (0..vectors.len()).collect();
        indices.shuffle(&mut rng);
        
        for &idx in indices.iter().take(k) {
            centroids.push(vectors[idx].clone());
        }
        
        // 初始化分类结果
        let mut assignments = vec![0; vectors.len()];
        let mut prev_assignments = vec![usize::MAX; vectors.len()];
        
        // 迭代直到收敛或达到最大迭代次数
        for iteration in 0..max_iterations {
            // 如果分配不变，则收敛
            if assignments == prev_assignments {
                log::info!("K-means converged after {} iterations", iteration);
                break;
            }
            
            prev_assignments = assignments.clone();
            
            // 第一步：分配向量到最近的中心点
            {
                let centroids_clone = centroids.clone();
                let assignments_arc = Arc::new(Mutex::new(assignments.clone()));
                
                let batch_size = self.config.batch_size;
                let total_vectors = vectors.len();
                let total_batches = (total_vectors + batch_size - 1) / batch_size;
                
                // 创建完成信号通道
                let (tx, rx) = bounded(total_batches);
                
                // 分批处理向量分配
                for batch_start in (0..total_vectors).step_by(batch_size) {
                    let batch_end = (batch_start + batch_size).min(total_vectors);
                    let batch_vectors = vectors[batch_start..batch_end].to_vec();
                    
                    let centroids_clone = centroids_clone.clone();
                    let assignments_clone = Arc::clone(&assignments_arc);
                    let tx_clone = tx.clone();
                    
                    let workers = self.workers.read().map_err(|_| 
                        Error::vector("Failed to acquire read lock on worker pool"))?;
                    
                    workers.submit(move || {
                        // 为这个批次中的每个向量找到最近的中心点
                        let mut batch_assignments = Vec::with_capacity(batch_vectors.len());
                        
                        for vec in &batch_vectors {
                            let mut best_cluster = 0;
                            let mut best_similarity = f32::NEG_INFINITY;
                            
                            for (cluster, centroid) in centroids_clone.iter().enumerate() {
                                let similarity = super::super::core::operations::VectorOps::compute_similarity(
                                    vec, centroid, metric
                                );
                                
                                if similarity > best_similarity {
                                    best_similarity = similarity;
                                    best_cluster = cluster;
                                }
                            }
                            
                            batch_assignments.push(best_cluster);
                        }
                        
                        // 更新全局分配
                        if let Ok(mut assignments) = assignments_clone.lock() {
                            for (i, cluster) in batch_assignments.iter().enumerate() {
                                assignments[batch_start + i] = *cluster;
                            }
                        }
                        
                        // 发送完成信号
                        let _ = tx_clone.send(());
                    })?;
                }
                
                // 等待所有批次完成
                for _ in 0..total_batches {
                    rx.recv().map_err(|_| Error::vector("Channel error while waiting for tasks to complete".to_string()))?;
                }
                
                // 获取更新后的分配
                assignments = assignments_arc.lock().map_err(|_| 
                    Error::vector("Failed to acquire lock on assignments".to_string()))?.clone();
            }
            
            // 第二步：更新中心点
            {
                let mut new_centroids = vec![vec![0.0; dim]; k];
                let mut counts = vec![0; k];
                
                for (i, &cluster) in assignments.iter().enumerate() {
                    counts[cluster] += 1;
                    
                    for j in 0..dim {
                        new_centroids[cluster][j] += vectors[i][j];
                    }
                }
                
                // 计算新的中心点
                for (cluster, centroid) in new_centroids.iter_mut().enumerate() {
                    if counts[cluster] > 0 {
                        for j in 0..dim {
                            centroid[j] /= counts[cluster] as f32;
                        }
                    } else {
                        // 如果没有向量被分配到这个聚类，保持原中心点
                        *centroid = centroids[cluster].clone();
                    }
                }
                
                centroids = new_centroids;
            }
        }
        
        Ok(assignments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_batch_processor() {
        // 创建工作线程池和批处理器
        let worker_pool = Arc::new(RwLock::new(WorkerPool::new(4, 100)));
        {
            let mut pool = worker_pool.write().unwrap();
            pool.start().unwrap();
        }
        
        let config = ParallelConfig {
            num_threads: 4,
            batch_size: 10,
            ..Default::default()
        };
        
        let processor = BatchProcessor::new(worker_pool.clone(), config);
        
        // 测试批量相似度计算
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        
        let queries = vec![
            vec![0.5, 0.5, 0.0],
            vec![0.0, 0.5, 0.5],
        ];
        
        let similarities = processor.batch_similarity(&vectors, &queries, SimilarityMetric::Cosine).unwrap();
        
        assert_eq!(similarities.len(), 2); // 2个查询
        assert_eq!(similarities[0].len(), 3); // 3个目标向量
        
        // 停止工作线程池
        {
            let mut pool = worker_pool.write().unwrap();
            pool.stop().unwrap();
        }
    }
} 