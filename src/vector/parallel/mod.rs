// 向量并行处理模块
//
// 提供高效的多线程并行向量处理和批量操作功能

use std::sync::{Arc, Mutex, RwLock};
use std::collections::{HashMap};
use std::cmp::Ordering;
use rayon::prelude::*;
use crossbeam_channel::{bounded, unbounded};
use num_cpus;

use crate::Result;
use crate::Error;
use crate::vector::types::Vector;
use crate::vector::index::SearchResult;
use crate::vector::operations::SimilarityMetric;
use crate::vector::core::operations::VectorOps;

pub mod batch;
pub mod pipeline;
pub mod worker;

pub use batch::BatchProcessor;
pub use pipeline::VectorPipeline;
pub use worker::WorkerPool;

/// 并行任务类型
#[derive(Debug, Clone, Copy)]
pub enum TaskType {
    /// 相似度计算
    Similarity,
    /// 向量归一化
    Normalization,
    /// 向量检索
    Search,
    /// 向量聚类
    Clustering,
    /// 特征提取
    FeatureExtraction,
    /// 维度约简
    DimensionReduction,
    /// 自定义处理
    Custom,
}

/// 并行执行配置
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// 线程数量
    pub num_threads: usize,
    /// 最大队列大小
    pub queue_size: usize,
    /// 批处理大小
    pub batch_size: usize,
    /// 是否提前结束（找到结果就返回）
    pub early_termination: bool,
    /// 是否需要保持向量顺序
    pub preserve_order: bool,
    /// 使用的CPU核心数比例 (0.0-1.0)
    pub cpu_usage_ratio: f32,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            queue_size: 1000,
            batch_size: 100,
            early_termination: false,
            preserve_order: true,
            cpu_usage_ratio: 0.75,
        }
    }
}

/// 向量并行处理管理器
pub struct ParallelVectorManager {
    /// 配置
    config: ParallelConfig,
    /// 工作线程池
    workers: Arc<RwLock<WorkerPool>>,
    /// 是否已启动
    started: bool,
}

impl ParallelVectorManager {
    /// 创建新的并行处理管理器
    pub fn new(config: ParallelConfig) -> Self {
        let worker_pool = WorkerPool::new(config.num_threads, config.queue_size);
        
        Self {
            config,
            workers: Arc::new(RwLock::new(worker_pool)),
            started: false,
        }
    }
    
    /// 启动并行处理系统
    pub fn start(&mut self) -> Result<()> {
        if self.started {
            return Ok(());
        }
        
        let mut workers = self.workers.write().map_err(|_| 
            Error::vector("Failed to acquire write lock on worker pool".to_string()))?;
            
        workers.start()?;
        self.started = true;
        
        Ok(())
    }
    
    /// 停止并行处理系统
    pub fn stop(&mut self) -> Result<()> {
        if !self.started {
            return Ok(());
        }
        
        let mut workers = self.workers.write().map_err(|_| 
            Error::vector("Failed to acquire write lock on worker pool".to_string()))?;
            
        workers.stop()?;
        self.started = false;
        
        Ok(())
    }
    
    /// 并行执行向量相似度计算
    pub fn parallel_similarity(
        &self,
        vectors: &[Vec<f32>],
        query: &[f32],
        metric: SimilarityMetric,
    ) -> Result<Vec<(usize, f32)>> {
        if !self.started {
            return Err(Error::vector("Parallel manager not started".to_string()));
        }
        
        let total_vectors = vectors.len();
        let query = query.to_vec();
        let batch_size = self.config.batch_size;
        
        // 创建结果容器
        let results = Arc::new(Mutex::new(Vec::with_capacity(total_vectors)));
        
        // 创建信号通道来跟踪完成情况
        let (tx, rx) = bounded(total_vectors);
        
        // 分批次提交任务
        for chunk_start in (0..total_vectors).step_by(batch_size) {
            let chunk_end = (chunk_start + batch_size).min(total_vectors);
            let chunk_vectors: Vec<Vec<f32>> = vectors[chunk_start..chunk_end]
                .iter()
                .map(|v| v.clone())
                .collect();
            
            let query_clone = query.clone();
            let results_clone = Arc::clone(&results);
            let tx_clone = tx.clone();
            let job_start_index = chunk_start;
            
            let workers = self.workers.read().map_err(|_| 
                Error::vector("Failed to acquire read lock on worker pool".to_string()))?;
            
            workers.submit(move || {
                // 计算这个批次中所有向量的相似度
                let chunk_results: Vec<(usize, f32)> = chunk_vectors.iter().enumerate()
                    .map(|(i, vec)| {
                        let similarity = VectorOps::compute_similarity(vec, &query_clone, metric);
                        (job_start_index + i, similarity)
                    })
                    .collect();
                
                // 添加到总结果
                if let Ok(mut results_guard) = results_clone.lock() {
                    results_guard.extend(chunk_results);
                }
                
                // 发送完成信号
                for _ in job_start_index..chunk_end {
                    let _ = tx_clone.send(());
                }
            })?;
        }
        
        // 等待所有计算完成
        for _ in 0..total_vectors {
            rx.recv().map_err(|_| Error::vector("Channel error while waiting for tasks to complete".to_string()))?;
        }
        
        // 获取结果
        let final_results = results.lock().map_err(|_| 
            Error::vector("Failed to acquire lock on results".to_string()))?;
            
        let mut similarity_results = final_results.clone();
        
        // 根据索引排序
        if self.config.preserve_order {
            similarity_results.sort_by_key(|(idx, _)| *idx);
        } else {
            // 根据相似度排序（降序）
            similarity_results.sort_by(|(_, a), (_, b)| {
                b.partial_cmp(a).unwrap_or(Ordering::Equal)
            });
        }
        
        Ok(similarity_results)
    }
    
    /// 并行执行向量搜索
    pub fn parallel_search(
        &self,
        vectors: &HashMap<String, Vector>,
        query: &[f32],
        limit: usize,
        metric: SimilarityMetric,
    ) -> Result<Vec<SearchResult>> {
        if !self.started {
            return Err(Error::vector("Parallel manager not started".to_string()));
        }
        
        let total_vectors = vectors.len();
        let query = query.to_vec();
        let batch_size = self.config.batch_size;
        
        // 创建局部结果集的通道
        let (tx, rx) = unbounded();
        
        // 将向量分成批次
        let vector_batches: Vec<Vec<(String, Vector)>> = vectors
            .iter()
            .map(|(id, vec)| (id.clone(), vec.clone()))
            .collect::<Vec<_>>()
            .chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        // 提交搜索任务
        for batch in vector_batches {
            let query_clone = query.clone();
            let tx_clone = tx.clone();
            
            let workers = self.workers.read().map_err(|_| 
                Error::vector("Failed to acquire read lock on worker pool".to_string()))?;
            
            workers.submit(move || {
                // 计算这个批次中所有向量的相似度
                let mut batch_results = Vec::with_capacity(batch.len());
                
                for (id, vector) in batch {
                    let similarity = VectorOps::compute_similarity(&vector.data, &query_clone, metric);
                    
                    let metadata = vector.metadata.as_ref().map(|m| {
                        let mut map = serde_json::Map::new();
                        for (k, v) in &m.properties {
                            map.insert(k.clone(), v.clone());
                        }
                        serde_json::Value::Object(map)
                    });
                    
                    batch_results.push(SearchResult {
                        id,
                        distance: similarity,
                        metadata,
                    });
                }
                
                // 发送这个批次的结果
                let _ = tx_clone.send(batch_results);
            })?;
        }
        
        // 关闭发送通道
        drop(tx);
        
        // 收集和合并所有批次的结果
        let mut all_results = Vec::with_capacity(total_vectors);
        while let Ok(batch_results) = rx.recv() {
            all_results.extend(batch_results);
        }
        
        // 排序并截断结果
        all_results.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap_or(Ordering::Equal));
        if all_results.len() > limit {
            all_results.truncate(limit);
        }
        
        Ok(all_results)
    }
    
    /// 并行执行向量批量归一化
    pub fn parallel_normalize(&self, vectors: &mut [Vec<f32>]) -> Result<()> {
        if !self.started {
            return Err(Error::vector("Parallel manager not started".to_string()));
        }
        
        let total_vectors = vectors.len();
        let batch_size = self.config.batch_size;
        
        // 创建完成信号通道
        let (tx, rx) = bounded(total_vectors);
        
        // 分批次提交归一化任务
        for chunk_start in (0..total_vectors).step_by(batch_size) {
            let chunk_end = (chunk_start + batch_size).min(total_vectors);
            
            // 获取对这个批次中向量的可变引用
            let chunk_vectors = unsafe {
                // 安全性：我们确保不同批次访问不同的数组元素
                let ptr = vectors.as_mut_ptr().add(chunk_start);
                std::slice::from_raw_parts_mut(ptr, chunk_end - chunk_start)
            };
            
            let tx_clone = tx.clone();
            
            let workers = self.workers.read().map_err(|_| 
                Error::vector("Failed to acquire read lock on worker pool".to_string()))?;
            
            workers.submit(move || {
                // 归一化这个批次中的所有向量
                for vec in chunk_vectors {
                    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 1e-6 {
                        for val in vec.iter_mut() {
                            *val /= norm;
                        }
                    }
                }
                
                // 发送完成信号
                for _ in chunk_start..chunk_end {
                    let _ = tx_clone.send(());
                }
            })?;
        }
        
        // 等待所有归一化任务完成
        for _ in 0..total_vectors {
            rx.recv().map_err(|_| Error::vector("Channel error while waiting for tasks to complete".to_string()))?;
        }
        
        Ok(())
    }
    
    /// 创建向量处理管道
    pub fn create_pipeline(&self) -> VectorPipeline {
        VectorPipeline::new(Arc::clone(&self.workers), self.config.clone())
    }
    
    /// 创建批处理器
    pub fn create_batch_processor(&self) -> BatchProcessor {
        BatchProcessor::new(Arc::clone(&self.workers), self.config.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parallel_manager() {
        // 创建配置和管理器
        let config = ParallelConfig {
            num_threads: 4,
            batch_size: 10,
            ..Default::default()
        };
        
        let mut manager = ParallelVectorManager::new(config);
        manager.start().unwrap();
        
        // 准备测试数据
        let mut vectors = vec![
            vec![3.0, 4.0],
            vec![5.0, 12.0],
            vec![1.0, 1.0],
            vec![7.0, 24.0],
            vec![8.0, 15.0],
        ];
        
        let query = vec![1.0, 0.0];
        
        // 测试并行相似度计算
        let similarities = manager.parallel_similarity(&vectors, &query, SimilarityMetric::Cosine).unwrap();
        assert_eq!(similarities.len(), 5);
        
        // 对于[3,4]向量，与[1,0]的余弦相似度应该是3/5
        assert!((similarities[0].1 - 3.0/5.0).abs() < 1e-6);
        
        // 测试并行归一化
        manager.parallel_normalize(&mut vectors).unwrap();
        
        // 第一个向量[3,4]应该被归一化为[3/5, 4/5]
        assert!((vectors[0][0] - 3.0/5.0).abs() < 1e-6);
        assert!((vectors[0][1] - 4.0/5.0).abs() < 1e-6);
        
        // 停止管理器
        manager.stop().unwrap();
    }
} 