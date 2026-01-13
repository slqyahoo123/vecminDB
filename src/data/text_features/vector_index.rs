// 文本特征向量索引管理器
// 提供文本特征向量的索引构建和优化功能

use crate::Result;
use crate::Error;
use crate::data::text_features::vector_store::TextFeatureVectorStore;
use crate::vector::{Vector, index::{IndexType, IndexConfig, VectorIndexFactory}};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// 索引优化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexOptimizationConfig {
    /// 选择的索引类型
    pub index_type: Option<IndexType>,
    /// 自动选择最佳索引
    pub auto_select_best: bool,
    /// 自动优化间隔(秒)
    pub auto_optimize_interval: u64,
    /// 性能评估样本数
    pub benchmark_sample_size: usize,
    /// 性能评估查询数
    pub benchmark_query_count: usize,
    /// 定期重建索引
    pub periodic_rebuild: bool,
    /// 重建索引间隔(秒)
    pub rebuild_interval: u64,
    /// 索引参数
    pub index_params: HashMap<String, String>,
}

impl Default for IndexOptimizationConfig {
    fn default() -> Self {
        Self {
            index_type: None,
            auto_select_best: true,
            auto_optimize_interval: 3600,
            benchmark_sample_size: 1000,
            benchmark_query_count: 10,
            periodic_rebuild: false,
            rebuild_interval: 86400,
            index_params: HashMap::new(),
        }
    }
}

/// 索引优化结果
#[derive(Debug, Clone)]
pub struct IndexOptimizationResult {
    /// 选择的索引类型
    pub selected_index: IndexType,
    /// 构建时间
    pub build_time: Duration,
    /// 查询时间
    pub query_time: Duration,
    /// 内存使用
    pub memory_usage: usize,
    /// 召回率
    pub recall: f32,
    /// 准确率
    pub precision: f32,
}

/// 向量索引管理器
pub struct VectorIndexManager {
    /// 向量存储
    store: TextFeatureVectorStore,
    /// 优化配置
    config: IndexOptimizationConfig,
    /// 最后优化时间
    last_optimize_time: Option<Instant>,
    /// 最后重建时间
    last_rebuild_time: Option<Instant>,
    /// 优化历史
    optimization_history: Vec<IndexOptimizationResult>,
}

impl VectorIndexManager {
    /// 创建新的索引管理器
    pub fn new(store: TextFeatureVectorStore, config: IndexOptimizationConfig) -> Self {
        Self {
            store,
            config,
            last_optimize_time: None,
            last_rebuild_time: None,
            optimization_history: Vec::new(),
        }
    }
    
    /// 构建索引
    pub fn build_index(&mut self, index_type: Option<IndexType>) -> Result<IndexOptimizationResult> {
        let index_type = index_type.unwrap_or_else(|| {
            self.config.index_type.unwrap_or(IndexType::Flat)
        });
        
        // 获取维度
        let dimension = self.store.get_config().dimension;
        
        // 创建索引配置
        let mut index_config = IndexConfig {
            dimension,
            index_type,
            ef_construction: 200,
            m: 16,
            ef_search: 50,
            ..Default::default()
        };
        
        // 应用自定义参数
        for (key, value) in &self.config.index_params {
            match key.as_str() {
                "ef_construction" | "hnsw_ef_construction" => {
                    if let Ok(v) = value.parse::<usize>() {
                        index_config.hnsw_ef_construction = v;
                        index_config.ef_construction = v; // 兼容历史字段
                    }
                },
                "m" | "hnsw_m" => {
                    if let Ok(v) = value.parse::<usize>() {
                        index_config.hnsw_m = v;
                        index_config.m = v; // 兼容历史字段
                    }
                },
                "ef_search" | "hnsw_ef_search" => {
                    if let Ok(v) = value.parse::<usize>() {
                        index_config.hnsw_ef_search = v;
                        index_config.ef_search = v; // 兼容历史字段
                    }
                },
                "nlist" | "ivf_nlist" => {
                    if let Ok(v) = value.parse::<usize>() {
                        index_config.ivf_nlist = v;
                    }
                },
                "nprobe" | "ivf_nprobe" => {
                    if let Ok(v) = value.parse::<usize>() {
                        index_config.ivf_nprobe = v;
                    }
                },
                "m_pq" | "pq_m" => {
                    if let Ok(v) = value.parse::<usize>() {
                        index_config.pq_m = v;
                    }
                },
                "nbits" | "pq_nbits" => {
                    if let Ok(v) = value.parse::<usize>() {
                        index_config.pq_nbits = v;
                    }
                },
                "hash_bits" | "lsh_hash_length" => {
                    if let Ok(v) = value.parse::<usize>() {
                        index_config.lsh_hash_length = v;
                    }
                },
                "n_tables" | "lsh_ntables" => {
                    if let Ok(v) = value.parse::<usize>() {
                        index_config.lsh_ntables = v;
                    }
                },
                _ => {}
            }
        }
        
        let start_time = Instant::now();
        
        // 注意：VectorCollection 没有 set_index 方法
        // 索引配置在创建 VectorCollection 时设置，这里只记录配置
        // 如果需要重建索引，需要重新创建 VectorCollection
        
        let build_time = start_time.elapsed();
        
        // 执行基准测试
        let benchmark_result = self.benchmark_index()?;
        
        let result = IndexOptimizationResult {
            selected_index: index_type,
            build_time,
            query_time: benchmark_result.query_time,
            memory_usage: benchmark_result.memory_usage,
            recall: benchmark_result.recall,
            precision: benchmark_result.precision,
        };
        
        // 更新状态
        self.last_rebuild_time = Some(Instant::now());
        self.optimization_history.push(result.clone());
        
        Ok(result)
    }
    
    /// 自动选择最佳索引
    pub fn auto_select_best_index(&mut self) -> Result<IndexOptimizationResult> {
        // 准备测试数据
        let vector_count = self.store.count()?;
        let sample_size = std::cmp::min(self.config.benchmark_sample_size, vector_count);
        
        if sample_size < 10 {
            return Err(Error::InvalidOperation("向量数量不足，无法执行基准测试".to_string()));
        }
        
        // 获取维度
        let dimension = self.store.get_config().dimension;
        
        // 收集向量样本
        let mut vectors = Vec::with_capacity(sample_size);
        {
            let collection = self.store.get_collection()?;
            let all_vectors: Vec<&Vector> = collection.vectors.values().collect();
            
            for (i, vector) in all_vectors.iter().enumerate() {
                if i >= sample_size {
                    break;
                }
                vectors.push((*vector).clone());
            }
        }
        
        // 准备查询向量
        let mut query_vectors = Vec::with_capacity(self.config.benchmark_query_count);
        for _ in 0..self.config.benchmark_query_count {
            if let Some(vector) = vectors.get(rand::random::<usize>() % vectors.len()) {
                query_vectors.push(vector.data.clone());
            }
        }
        
        // 执行索引基准测试
        // benchmark_indexes 期望 &[Vector]，需要传入完整的 Vector 对象
        let benchmark_results = VectorIndexFactory::benchmark_indexes(
            &vectors,
            &query_vectors,
            dimension,
            10
        )?;
        
        // 选择最佳索引（基于准确率和查询时间）
        let best_result = benchmark_results.iter()
            .max_by(|a, b| {
                // 优先考虑准确率，然后考虑查询时间
                a.accuracy.partial_cmp(&b.accuracy)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        b.avg_query_time_ms.partial_cmp(&a.avg_query_time_ms)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            })
            .ok_or_else(|| Error::Internal("基准测试结果为空".to_string()))?;
        
        // 构建最佳索引
        self.build_index(Some(best_result.index_type))
    }
    
    /// 优化索引
    /// 注意：VectorCollection 目前没有 optimize 方法，此方法暂时为空实现
    pub fn optimize_index(&mut self) -> Result<()> {
        // VectorCollection 目前不支持优化操作
        // 如果需要优化，可以在这里实现索引重建等逻辑
        self.last_optimize_time = Some(Instant::now());
        
        Ok(())
    }
    
    /// 检查是否需要优化
    pub fn check_need_optimize(&self) -> bool {
        if let Some(last_time) = self.last_optimize_time {
            last_time.elapsed().as_secs() >= self.config.auto_optimize_interval
        } else {
            true
        }
    }
    
    /// 检查是否需要重建
    pub fn check_need_rebuild(&self) -> bool {
        if let Some(last_time) = self.last_rebuild_time {
            last_time.elapsed().as_secs() >= self.config.rebuild_interval
        } else {
            true
        }
    }
    
    /// 执行自动维护
    pub fn perform_auto_maintenance(&mut self) -> Result<Option<IndexOptimizationResult>> {
        let mut result = None;
        
        // 检查是否需要重建索引
        if self.config.periodic_rebuild && self.check_need_rebuild() {
            if self.config.auto_select_best {
                result = Some(self.auto_select_best_index()?);
            } else if let Some(index_type) = self.config.index_type {
                result = Some(self.build_index(Some(index_type))?);
            }
        }
        // 检查是否需要优化索引
        else if self.check_need_optimize() {
            self.optimize_index()?;
        }
        
        Ok(result)
    }
    
    /// 基准测试当前索引
    pub fn benchmark_index(&self) -> Result<IndexOptimizationResult> {
        // 获取维度和当前索引类型
        let dimension = self.store.get_config().dimension;
        // 从配置中获取索引类型，如果没有则使用默认值
        let index_type = self.config.index_type.unwrap_or(IndexType::HNSW);
        
        // 准备测试数据
        let vector_count = self.store.count()?;
        let sample_size = std::cmp::min(self.config.benchmark_sample_size, vector_count);
        
        if sample_size < 10 {
            return Err(Error::InvalidOperation("向量数量不足，无法执行基准测试".to_string()));
        }
        
        // 收集向量样本
        let mut vectors = Vec::with_capacity(sample_size);
        {
            let collection = self.store.get_collection()?;
            let all_vectors: Vec<&Vector> = collection.vectors.values().collect();
            
            for (i, vector) in all_vectors.iter().enumerate() {
                if i >= sample_size {
                    break;
                }
                vectors.push((*vector).clone());
            }
        }
        
        // 准备查询向量和参考结果
        let mut query_vectors = Vec::with_capacity(self.config.benchmark_query_count);
        let mut reference_results = Vec::with_capacity(self.config.benchmark_query_count);
        
        for _ in 0..self.config.benchmark_query_count {
            if let Some(vector) = vectors.get(rand::random::<usize>() % vectors.len()) {
                query_vectors.push(vector.data.clone());
                
                // 计算与所有向量的距离
                let mut distances = Vec::with_capacity(vectors.len());
                for other in &vectors {
                    let dist = calculate_euclidean_distance(&vector.data, &other.data);
                    distances.push((other.id.clone(), dist));
                }
                
                // 按距离排序
                distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                
                // 取前10个作为参考结果
                let top_10 = distances.into_iter().take(10).collect::<Vec<_>>();
                reference_results.push(top_10);
            }
        }
        
        // 执行查询性能测试
        let start_time = Instant::now();
        let mut query_results = Vec::with_capacity(query_vectors.len());
        
        for query in &query_vectors {
            let results = {
                let collection = self.store.get_collection()?;
                let query_obj = crate::vector::VectorQuery {
                    vector: query.clone(),
                    filter: None,
                    top_k: 10,
                    include_metadata: false,
                    include_vectors: false,
                };
                collection.search(&query_obj)?
            };
            query_results.push(results);
        }
        
        let query_time = start_time.elapsed();
        
        // 计算召回率和准确率
        let mut total_recall = 0.0;
        let mut total_precision = 0.0;
        
        for (i, result) in query_results.iter().enumerate() {
            let reference = &reference_results[i];
            
            // 计算结果中有多少在参考结果中
            let mut hits = 0;
            let result_ids: Vec<String> = result.iter().map(|r| r.id.clone()).collect();
            
            for (ref_id, _) in reference {
                if result_ids.contains(ref_id) {
                    hits += 1;
                }
            }
            
            let recall = hits as f32 / reference.len() as f32;
            let precision = hits as f32 / result.len() as f32;
            
            total_recall += recall;
            total_precision += precision;
        }
        
        let avg_recall = total_recall / query_results.len() as f32;
        let avg_precision = total_precision / query_results.len() as f32;
        
        // 获取内存使用估计
        // 注意：VectorCollection 目前没有 estimate_memory_usage 方法
        // 这里使用一个简单的估算：向量数量 * 向量维度 * 4字节（f32）
        let memory_usage = {
            let collection = self.store.get_collection()?;
            let vector_count = collection.vectors.len();
            let dimension = collection.config.dimension;
            vector_count * dimension * 4 // 每个 f32 占 4 字节
        };
        
        Ok(IndexOptimizationResult {
            selected_index: index_type,
            build_time: Duration::from_secs(0), // 未测量构建时间
            query_time,
            memory_usage,
            recall: avg_recall,
            precision: avg_precision,
        })
    }
    
    /// 获取优化历史
    pub fn get_optimization_history(&self) -> &[IndexOptimizationResult] {
        &self.optimization_history
    }
    
    /// 获取向量存储
    pub fn get_store(&self) -> &TextFeatureVectorStore {
        &self.store
    }
    
    /// 获取可变向量存储
    pub fn get_store_mut(&mut self) -> &mut TextFeatureVectorStore {
        &mut self.store
    }
    
    /// 获取配置
    pub fn get_config(&self) -> &IndexOptimizationConfig {
        &self.config
    }
    
    /// 设置配置
    pub fn set_config(&mut self, config: IndexOptimizationConfig) {
        self.config = config;
    }
}

/// 计算欧氏距离
fn calculate_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::MAX;
    }
    
    let mut sum = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    
    sum.sqrt()
} 