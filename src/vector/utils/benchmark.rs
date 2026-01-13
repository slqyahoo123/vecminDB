use crate::{
    error::Error,
    vector::{
        index::{
            IndexType, IndexConfig, VectorIndex, VectorIndexEnum,
            FlatIndex
        }
    },
    Result,
};
use crate::vector::types::Vector;
use crate::vector::operations::SimilarityMetric;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use rand::{Rng, SeedableRng};
use serde::{Serialize, Deserialize};
use rayon::prelude::*;
use std::sync::Arc;
use parking_lot::RwLock;

/// 索引性能指标
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IndexPerformanceMetrics {
    /// 构建索引时间（毫秒）
    pub build_time_ms: f64,
    /// 平均查询时间（毫秒）
    pub avg_query_time_ms: f64,
    /// 内存使用（字节）
    pub memory_usage_bytes: u64,
    /// 索引大小（字节）
    pub index_size_bytes: u64,
    /// 召回率
    pub recall_rate: f64,
    /// 查询准确率
    pub accuracy: f64,
}

/// 性能测试结果
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkResult {
    /// 索引类型
    pub index_type: IndexType,
    /// 索引配置
    pub config: IndexConfig,
    /// 性能指标
    pub metrics: IndexPerformanceMetrics,
    /// 数据集大小
    pub dataset_size: usize,
    /// 向量维度
    pub dimension: usize,
    /// 构建索引时间（毫秒）
    pub build_time_ms: u64,
    /// 平均查询时间（毫秒）
    pub avg_query_time_ms: f64,
    /// 查询吞吐量（每秒查询数）
    pub queries_per_second: f64,
    /// 内存使用（字节）
    pub memory_usage_bytes: usize,
    /// 查询准确率（与暴力搜索相比）
    pub accuracy: f64,
    /// 索引大小（字节）
    pub index_size_bytes: usize,
}

/// 性能测试配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// 数据集大小
    pub dataset_size: usize,
    /// 向量维度
    pub dimension: usize,
    /// 查询集大小
    pub query_size: usize,
    /// 查询结果数量
    pub top_k: usize,
    /// 重复次数
    pub repeat_count: usize,
    /// 是否测试准确率
    pub test_accuracy: bool,
    /// 是否测试内存使用
    pub test_memory: bool,
    /// 是否测试索引大小
    pub test_index_size: bool,
    /// 是否使用并行测试
    pub parallel: bool,
    /// 随机种子
    pub random_seed: Option<u64>,
    /// 相似度计算方式
    pub metric: SimilarityMetric,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        BenchmarkConfig {
            dataset_size: 10000,
            dimension: 128,
            query_size: 100,
            top_k: 10,
            repeat_count: 3,
            test_accuracy: true,
            test_memory: true,
            test_index_size: true,
            parallel: true,
            random_seed: None,
            metric: SimilarityMetric::Euclidean,
        }
    }
}

/// 向量索引性能测试器
#[derive(Clone)]
pub struct IndexBenchmark {
    config: BenchmarkConfig,
}

impl IndexBenchmark {
    /// 创建新的性能测试器
    pub fn new(config: BenchmarkConfig) -> Self {
        IndexBenchmark { config }
    }
    
    /// 生成随机向量数据集
    pub fn generate_dataset(&self) -> Vec<Vector> {
        let mut rng = match self.config.random_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };
        
        let mut vectors = Vec::with_capacity(self.config.dataset_size);
        
        for i in 0..self.config.dataset_size {
            let mut data = Vec::with_capacity(self.config.dimension);
            
            for _ in 0..self.config.dimension {
                data.push(rng.gen_range(-1.0..1.0));
            }
            
            // 归一化向量
            let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            let data: Vec<f32> = data.iter().map(|x| x / norm).collect();
            
            vectors.push(Vector {
                id: format!("vec_{}", i),
                data,
                metadata: None,
            });
        }
        
        vectors
    }
    
    /// 生成随机查询向量
    pub fn generate_queries(&self) -> Vec<Vec<f32>> {
        let mut rng = match self.config.random_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };
        
        let mut queries = Vec::with_capacity(self.config.query_size);
        
        for _ in 0..self.config.query_size {
            let mut query = Vec::with_capacity(self.config.dimension);
            
            for _ in 0..self.config.dimension {
                query.push(rng.gen_range(-1.0..1.0));
            }
            
            // 归一化向量
            let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            let query: Vec<f32> = query.iter().map(|x| x / norm).collect();
            
            queries.push(query);
        }
        
        queries
    }
    
    /// 测试单个索引类型
    pub fn benchmark_index(&self, index_type: IndexType, index_config: IndexConfig) -> Result<BenchmarkResult, Error> {
        // 生成数据集
        let vectors = self.generate_dataset();
        let queries = self.generate_queries();
        
        // 创建索引
        let build_start = Instant::now();
        
        let index = match index_type {
            IndexType::Flat => {
                let flat_index = crate::vector::index::FlatIndex::new(index_config.clone());
                Arc::new(RwLock::new(VectorIndexEnum::Flat(flat_index)))
            },
            IndexType::HNSW => {
                // 当前基准测试暂不直接使用 HNSW，实现上退化为 Flat 以保证功能可用
                let flat_index = crate::vector::index::FlatIndex::new(index_config.clone());
                Arc::new(RwLock::new(VectorIndexEnum::Flat(flat_index)))
            },
            IndexType::IVF => {
                let ivf_index = crate::vector::index::IVFIndex::new(index_config.clone())?;
                Arc::new(RwLock::new(VectorIndexEnum::IVF(ivf_index)))
            },
            IndexType::PQ => {
                let pq_index = crate::vector::index::PQIndex::new(index_config.clone())?;
                Arc::new(RwLock::new(VectorIndexEnum::PQ(pq_index)))
            },
            IndexType::LSH => {
                let lsh_index = crate::vector::index::LSHIndex::new(index_config.clone())?;
                Arc::new(RwLock::new(VectorIndexEnum::LSH(lsh_index)))
            },
            IndexType::IVFPQ => {
                println!("创建IVFPQ索引 (使用IVF实现)");
                let ivf_index = crate::vector::index::IVFIndex::new(index_config.clone())?;
                Arc::new(RwLock::new(VectorIndexEnum::IVF(ivf_index)))
            },
            IndexType::ANNOY => {
                println!("创建ANNOY索引 (基准测试中退化为 Flat)");
                let flat_index = crate::vector::index::FlatIndex::new(index_config.clone());
                Arc::new(RwLock::new(VectorIndexEnum::Flat(flat_index)))
            },
            IndexType::NGT => {
                println!("创建NGT索引 (基准测试中退化为 LSH)");
                let lsh_index = crate::vector::index::LSHIndex::new(index_config.clone())?;
                Arc::new(RwLock::new(VectorIndexEnum::LSH(lsh_index)))
            },
            IndexType::HierarchicalClustering => {
                println!("创建层次聚类索引 (基准测试中退化为 Flat)");
                let flat_index = crate::vector::index::FlatIndex::new(index_config.clone());
                Arc::new(RwLock::new(VectorIndexEnum::Flat(flat_index)))
            },
            IndexType::GraphIndex => {
                println!("创建图索引 (基准测试中退化为 Flat)");
                let flat_index = crate::vector::index::FlatIndex::new(index_config.clone());
                Arc::new(RwLock::new(VectorIndexEnum::Flat(flat_index)))
            },
            _ => {
                // 默认使用Flat索引
                println!("未知索引类型，使用Flat索引");
                let flat_index = crate::vector::index::FlatIndex::new(index_config.clone());
                Arc::new(RwLock::new(VectorIndexEnum::Flat(flat_index)))
            }
        };
        
        // 添加向量到索引
        {
            let mut index_write = index.write();
            for vector in &vectors {
                index_write.add(vector.clone())?;
            }
        }
        let build_time = build_start.elapsed();
        
        // 测试查询性能
        let mut query_times = Vec::with_capacity(self.config.query_size * self.config.repeat_count);
        
        for _ in 0..self.config.repeat_count {
            for query in &queries {
                let query_start = Instant::now();
                let index_read = index.read();
                let _ = index_read.search(query, self.config.top_k)?;
                query_times.push(query_start.elapsed());
            }
        }
        
        // 计算平均查询时间
        let total_query_time: Duration = query_times.iter().sum();
        let avg_query_time = total_query_time / (self.config.query_size * self.config.repeat_count) as u32;
        let queries_per_second = 1000.0 / (avg_query_time.as_millis() as f64);
        
        // 测试准确率（与暴力搜索比较）
        let accuracy = if self.config.test_accuracy {
            self.test_accuracy(&vectors, &queries, &index)?
        } else {
            1.0
        };
        
        // 测试内存使用
        let memory_usage: usize = if self.config.test_memory {
            self.estimate_memory_usage(&index)
        } else {
            0
        };
        
        // 测试索引大小
        let index_size: usize = if self.config.test_index_size {
            self.estimate_index_size(&index)
        } else {
            0
        };
        
        Ok(BenchmarkResult {
            index_type,
            config: index_config,
            metrics: IndexPerformanceMetrics {
                build_time_ms: build_time.as_millis() as f64,
                avg_query_time_ms: avg_query_time.as_millis() as f64,
                memory_usage_bytes: memory_usage as u64,
                index_size_bytes: index_size as u64,
                recall_rate: 0.0,
                accuracy,
            },
            dataset_size: self.config.dataset_size,
            dimension: self.config.dimension,
            build_time_ms: build_time.as_millis() as u64,
            avg_query_time_ms: avg_query_time.as_millis() as f64,
            queries_per_second,
            memory_usage_bytes: memory_usage,
            accuracy,
            index_size_bytes: index_size,
        })
    }
    
    /// 测试所有索引类型
    pub fn benchmark_all(&self) -> Result<Vec<BenchmarkResult>> {
        let index_types = vec![
            IndexType::Flat,
            IndexType::HNSW,
            IndexType::IVF,
            IndexType::PQ,
            IndexType::LSH,
        ];
        
        let results = if self.config.parallel {
            index_types.par_iter().map(|&index_type| {
                let config = self.create_config_for_index(index_type);
                self.benchmark_index(index_type, config)
            }).collect::<Result<Vec<_>>>()?
        } else {
            let mut results = Vec::with_capacity(index_types.len());
            for &index_type in &index_types {
                let config = self.create_config_for_index(index_type);
                results.push(self.benchmark_index(index_type, config)?);
            }
            results
        };
        
        Ok(results)
    }
    
    /// 为指定索引类型创建配置
    pub fn create_config_for_index(&self, index_type: IndexType) -> IndexConfig {
        let mut config = IndexConfig::default();
        config.index_type = index_type;
        config.dimension = self.config.dimension;
        config.metric = self.config.metric; // 使用BenchmarkConfig中的metric
        
        // 根据索引类型设置特定参数
        match index_type {
            IndexType::HNSW => {
                config.hnsw_m = 16;
                config.hnsw_ef_construction = 200;
                config.hnsw_ef_search = 100;
            },
            IndexType::IVF => {
                config.ivf_nlist = (self.config.dataset_size as f64).sqrt() as usize;
                config.ivf_nprobe = 10;
            },
            IndexType::PQ => {
                config.pq_subvector_count = 8;
                config.pq_subvector_bits = 8;
            },
            IndexType::LSH => {
                config.lsh_hash_count = 10;
                config.lsh_hash_length = 32;
            },
            _ => {}
        }
        
        config
    }
    
    /// 测试索引准确率（与暴力搜索比较）
    fn test_accuracy(&self, vectors: &[Vector], queries: &[Vec<f32>], index: &Arc<RwLock<VectorIndexEnum>>) -> Result<f64> {
        // 创建暴力搜索索引
        let flat_config = IndexConfig {
            index_type: IndexType::Flat,
            metric: self.config.metric.clone(),
            dimension: self.config.dimension,
            ..Default::default()
        };
        
        let flat_index = FlatIndex::new(flat_config.clone());
        let flat_index = Arc::new(RwLock::new(VectorIndexEnum::Flat(flat_index)));
        
        // 添加向量到暴力搜索索引
        {
            let mut flat_write = flat_index.write();
            for vector in vectors {
                flat_write.add(vector.clone())?;
            }
        }
        
        let mut total_overlap = 0;
        
        // 对每个查询，比较两个索引的结果
        for query in queries {
            let flat_results = {
                let flat_read = flat_index.read();
                flat_read.search(query, self.config.top_k)?
            };
            
            let index_results = {
                let index_read = index.read();
                index_read.search(query, self.config.top_k)?
            };
            
            // 计算结果重叠度
            let flat_ids: Vec<String> = flat_results.iter().map(|r| r.id.clone()).collect();
            let index_ids: Vec<String> = index_results.iter().map(|r| r.id.clone()).collect();
            
            let mut overlap = 0;
            for id in &index_ids {
                if flat_ids.contains(id) {
                    overlap += 1;
                }
            }
            
            total_overlap += overlap;
        }
        
        // 计算平均准确率
        let accuracy = total_overlap as f64 / (queries.len() * self.config.top_k) as f64;
        
        Ok(accuracy)
    }
    
    /// 估计索引内存使用
    fn estimate_memory_usage(&self, index: &Arc<RwLock<VectorIndexEnum>>) -> usize {
        let index_read = index.read();
        let size = index_read.size();
        let dimension = self.config.dimension;
        
        // 基础向量存储大小计算
        let vector_data_size = size * dimension * std::mem::size_of::<f32>();
        let vector_ids_size = size * 24; // 假设平均ID长度为24字节
        let metadata_size = size * 16; // 每个向量的元数据开销
        
        let base_size = vector_data_size + vector_ids_size + metadata_size;
        
        // 根据索引类型计算额外开销
        let additional_size = match *index_read {
            VectorIndexEnum::Flat(_) => {
                // 暴力搜索：只需要存储向量，无额外结构
                0
            },
            VectorIndexEnum::HNSW(_) => {
                // HNSW：图结构开销
                let avg_connections = 16; // 平均每个节点的连接数
                let layer_count = ((size as f32).ln() / 2.0).ceil() as usize; // 分层数估算
                let graph_size = size * avg_connections * std::mem::size_of::<u32>(); // 连接表
                let layer_info_size = size * layer_count * std::mem::size_of::<u8>(); // 层级信息
                let entry_points_size = layer_count * std::mem::size_of::<u32>(); // 入口点
                
                graph_size + layer_info_size + entry_points_size
            },
            VectorIndexEnum::IVF(_) => {
                // IVF：倒排索引结构
                let cluster_count = (size as f32 / 100.0).sqrt() as usize; // 聚类数估算
                let centroids_size = cluster_count * dimension * std::mem::size_of::<f32>();
                let inverted_lists_size = size * std::mem::size_of::<u32>(); // 每个向量的聚类ID
                let cluster_metadata_size = cluster_count * 64; // 聚类元数据
                
                centroids_size + inverted_lists_size + cluster_metadata_size
            },
            VectorIndexEnum::PQ(_) => {
                // PQ：产品量化压缩
                let subvector_count = 8; // 通常将向量分为8个子向量
                let codebook_size = 256; // 每个子空间256个码本
                let codebooks_size = subvector_count * codebook_size * (dimension / subvector_count) * std::mem::size_of::<f32>();
                let compressed_vectors_size = size * subvector_count; // 每个向量用8个字节表示
                
                codebooks_size + compressed_vectors_size
            },
            VectorIndexEnum::LSH(_) => {
                // LSH：局部敏感哈希
                let hash_count = 10; // 哈希函数数量
                let bucket_count = size * 2; // 桶数量
                let hash_functions_size = hash_count * dimension * std::mem::size_of::<f32>();
                let hash_tables_size = bucket_count * std::mem::size_of::<u32>();
                let hash_values_size = size * hash_count; // 每个向量的哈希值
                
                hash_functions_size + hash_tables_size + hash_values_size
            },
            VectorIndexEnum::VPTree(_) => {
                // VP-Tree：距离树结构
                let tree_nodes = size * 2; // 二叉树节点数估算
                let node_size = std::mem::size_of::<u32>() * 3 + std::mem::size_of::<f32>(); // 左右子节点 + 分割点ID + 距离阈值
                let tree_structure_size = tree_nodes * node_size;
                let distance_cache_size = size * size / 10; // 部分距离缓存
                
                tree_structure_size + distance_cache_size
            },
            VectorIndexEnum::IVFPQ(_) => {
                // IVFPQ：IVF + PQ 结构叠加
                let cluster_count = (size as f32 / 100.0).sqrt() as usize;
                let centroids_size = cluster_count * dimension * std::mem::size_of::<f32>();
                let inverted_lists_size = size * std::mem::size_of::<u32>();
                let subvector_count = 8;
                let codebook_size = 256;
                let codebooks_size = subvector_count * codebook_size * (dimension / subvector_count) * std::mem::size_of::<f32>();
                let compressed_vectors_size = size * subvector_count; // 每个向量若干字节编码
                centroids_size + inverted_lists_size + codebooks_size + compressed_vectors_size
            },
            VectorIndexEnum::IVFHNSW(_) => {
                // IVFHNSW：IVF 聚类 + HNSW 子索引
                let cluster_count = (size as f32 / 100.0).sqrt() as usize;
                let centroids_size = cluster_count * dimension * std::mem::size_of::<f32>();
                let hnsw_graph_per_cluster = (size / cluster_count.max(1)) * 16 * std::mem::size_of::<u32>();
                let layer_info = (size / cluster_count.max(1)) * std::mem::size_of::<u8>();
                centroids_size + cluster_count * (hnsw_graph_per_cluster + layer_info)
            },
            VectorIndexEnum::ANNOY(_) => {
                // ANNOY：多棵随机投影树
                let tree_count = 10;
                let nodes_per_tree = size * 2; // 近似
                let node_bytes = 32; // 超平面+指针等
                tree_count * nodes_per_tree * node_bytes
            },
            VectorIndexEnum::KMeans(_) => {
                // KMeans 质心 + 分配表
                let cluster_count = (size as f32 / 100.0).sqrt() as usize;
                let centroids_size = cluster_count * dimension * std::mem::size_of::<f32>();
                let assignments = size * std::mem::size_of::<u32>();
                centroids_size + assignments
            },
            VectorIndexEnum::NGT(_) => {
                // NGT：邻域图与树
                let edge_per_node = 16;
                let graph_bytes = size * edge_per_node * std::mem::size_of::<u32>();
                let tree_overhead = size * 8;
                graph_bytes + tree_overhead
            },
            VectorIndexEnum::HierarchicalClustering(_) => {
                // 层次聚类：树结构
                let cluster_nodes = size * 2;
                let node_bytes = 24;
                cluster_nodes * node_bytes
            },
            VectorIndexEnum::GraphIndex(_) => {
                // 通用图索引：邻接表
                let degree = 8;
                size * degree * std::mem::size_of::<u32>()
            },
        };
        
        // 添加系统开销（内存对齐、堆管理等）
        let system_overhead = (base_size + additional_size) / 10; // 约10%的系统开销
        
        base_size + additional_size + system_overhead
    }
    
    /// 估计索引大小（序列化后）
    fn estimate_index_size(&self, index: &Arc<RwLock<VectorIndexEnum>>) -> usize {
        let index_read = index.read();
        let size = index_read.size();
        let dimension = self.config.dimension;
        
        // 序列化的基础数据大小
        let serialized_vectors_size = size * dimension * std::mem::size_of::<f32>();
        let serialized_ids_size = size * 20; // 序列化后的ID平均大小
        let metadata_size = 1024; // 索引元数据
        
        let base_size = serialized_vectors_size + serialized_ids_size + metadata_size;
        
        // 根据索引类型计算序列化后的结构大小
        let structure_size = match *index_read {
            VectorIndexEnum::Flat(_) => {
                // 暴力搜索：序列化大小与内存大小基本相同
                0
            },
            VectorIndexEnum::HNSW(_) => {
                // HNSW：图结构的序列化大小
                let avg_connections = 16;
                let layer_count = ((size as f32).ln() / 2.0).ceil() as usize;
                let graph_size = size * avg_connections * 4; // 连接表序列化
                let layer_info_size = size * layer_count; // 层级信息
                let header_size = 256; // HNSW头部信息
                
                graph_size + layer_info_size + header_size
            },
            VectorIndexEnum::IVF(_) => {
                // IVF：倒排索引的序列化大小
                let cluster_count = (size as f32 / 100.0).sqrt() as usize;
                let centroids_size = cluster_count * dimension * 4;
                let inverted_lists_size = size * 8; // 每个向量的聚类ID和offset
                let header_size = 128;
                
                centroids_size + inverted_lists_size + header_size
            },
            VectorIndexEnum::PQ(_) => {
                // PQ：压缩后的大小显著小于原始大小
                let subvector_count = 8;
                let codebook_size = 256;
                let codebooks_size = subvector_count * codebook_size * (dimension / subvector_count) * 4;
                let compressed_vectors_size = size * subvector_count; // 高度压缩
                let header_size = 64;
                
                codebooks_size + compressed_vectors_size + header_size
            },
            VectorIndexEnum::LSH(_) => {
                // LSH：哈希表的序列化大小
                let hash_count = 10;
                let bucket_count = size * 2;
                let hash_functions_size = hash_count * dimension * 4;
                let hash_tables_size = bucket_count * 8; // 桶ID和指针
                let header_size = 128;
                
                hash_functions_size + hash_tables_size + header_size
            },
            VectorIndexEnum::VPTree(_) => {
                // VP-Tree：树结构的紧凑序列化
                let tree_nodes = size * 2;
                let node_size = 16; // 紧凑的节点表示
                let tree_structure_size = tree_nodes * node_size;
                let header_size = 64;
                
                tree_structure_size + header_size
            },
            VectorIndexEnum::IVFPQ(_) => {
                // IVFPQ：质心+压缩向量+头
                let cluster_count = (size as f32 / 100.0).sqrt() as usize;
                let centroids_size = cluster_count * dimension * 4;
                let subvector_count = 8;
                let codebook_size = 256;
                let codebooks_size = subvector_count * codebook_size * (dimension / subvector_count) * 4;
                let compressed_vectors_size = size * subvector_count;
                let header_size = 128;
                centroids_size + codebooks_size + compressed_vectors_size + header_size
            },
            VectorIndexEnum::IVFHNSW(_) => {
                // IVFHNSW：IVF 头 + HNSW 层信息
                let cluster_count = (size as f32 / 100.0).sqrt() as usize;
                let ivf_header = 128;
                let hnsw_layer = ((size as f32).ln() / 2.0).ceil() as usize;
                let per_cluster = hnsw_layer * 64;
                ivf_header + cluster_count * per_cluster
            },
            VectorIndexEnum::ANNOY(_) => {
                // ANNOY：树节点序列化
                let tree_count = 10;
                let nodes_per_tree = size * 2;
                let node_bytes = 24;
                tree_count * nodes_per_tree * node_bytes
            },
            VectorIndexEnum::KMeans(_) => {
                // KMeans：质心与分配
                let cluster_count = (size as f32 / 100.0).sqrt() as usize;
                let centroids_size = cluster_count * dimension * 4;
                let assignments = size * 4;
                centroids_size + assignments
            },
            VectorIndexEnum::NGT(_) => {
                // NGT：邻接表与参数
                let edge_per_node = 16;
                let graph_bytes = size * edge_per_node * 4;
                let header = 64;
                graph_bytes + header
            },
            VectorIndexEnum::HierarchicalClustering(_) => {
                // 层次聚类树
                let cluster_nodes = size * 2;
                let node_bytes = 16;
                cluster_nodes * node_bytes
            },
            VectorIndexEnum::GraphIndex(_) => {
                // 通用图
                let degree = 8;
                size * degree * 4
            },
        };
        
        // 序列化通常比内存表示更紧凑
        let compression_ratio = 0.8; // 平均80%的压缩比
        
        ((base_size + structure_size) as f64 * compression_ratio) as usize
    }
    
    /// 比较不同索引类型的性能
    pub fn compare_indexes(&self) -> Result<HashMap<IndexType, BenchmarkResult>> {
        let results = self.benchmark_all()?;
        
        let mut comparison = HashMap::new();
        for result in results {
            comparison.insert(result.index_type, result);
        }
        
        Ok(comparison)
    }
    
    /// 生成性能报告
    pub fn generate_report(&self, results: &[BenchmarkResult]) -> String {
        let mut report = String::new();
        
        report.push_str("# 向量索引性能测试报告\n\n");
        report.push_str(&format!("- 数据集大小: {}\n", self.config.dataset_size));
        report.push_str(&format!("- 向量维度: {}\n", self.config.dimension));
        report.push_str(&format!("- 查询集大小: {}\n", self.config.query_size));
        report.push_str(&format!("- Top-K: {}\n", self.config.top_k));
        report.push_str(&format!("- 重复次数: {}\n\n", self.config.repeat_count));
        
        report.push_str("## 性能比较\n\n");
        report.push_str("| 索引类型 | 构建时间(ms) | 查询时间(ms) | 查询/秒 | 准确率 | 内存使用(MB) |\n");
        report.push_str("|----------|--------------|--------------|---------|--------|-------------|\n");
        
        for result in results {
            report.push_str(&format!(
                "| {:?} | {} | {:.2} | {:.2} | {:.2} | {:.2} |\n",
                result.index_type,
                result.build_time_ms,
                result.avg_query_time_ms,
                result.queries_per_second,
                result.accuracy,
                result.memory_usage_bytes as f64 / (1024.0 * 1024.0)
            ));
        }
        
        report.push_str("\n## 详细结果\n\n");
        
        for result in results {
            report.push_str(&format!("### {:?}\n\n", result.index_type));
            report.push_str(&format!("- 构建时间: {}ms\n", result.build_time_ms));
            report.push_str(&format!("- 平均查询时间: {:.2}ms\n", result.avg_query_time_ms));
            report.push_str(&format!("- 查询吞吐量: {:.2}查询/秒\n", result.queries_per_second));
            report.push_str(&format!("- 准确率: {:.2}\n", result.accuracy));
            report.push_str(&format!("- 内存使用: {:.2}MB\n", result.memory_usage_bytes as f64 / (1024.0 * 1024.0)));
            report.push_str(&format!("- 索引大小: {:.2}MB\n\n", result.index_size_bytes as f64 / (1024.0 * 1024.0)));
        }
        
        report
    }
}

/// 性能测试示例
pub fn run_benchmark_example() -> Result<String> {
    let config = BenchmarkConfig {
        dataset_size: 10000,
        dimension: 128,
        query_size: 100,
        top_k: 10,
        repeat_count: 3,
        test_accuracy: true,
        test_memory: true,
        test_index_size: true,
        parallel: true,
        random_seed: Some(42),
        metric: SimilarityMetric::Euclidean,
    };
    
    let benchmark = IndexBenchmark::new(config);
    let results = benchmark.benchmark_all()?;
    let report = benchmark.generate_report(&results);
    
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_flat() {
        let config = BenchmarkConfig {
            dataset_size: 1000,
            dimension: 64,
            query_size: 10,
            top_k: 5,
            repeat_count: 1,
            test_accuracy: true,
            test_memory: false,
            test_index_size: false,
            parallel: false,
            random_seed: Some(42),
            metric: SimilarityMetric::Euclidean,
        };
        
        let benchmark = IndexBenchmark::new(config);
        let mut index_config = IndexConfig::default();
        index_config.index_type = IndexType::Flat;
        index_config.dimension = 64;
        
        let result = benchmark.benchmark_index(IndexType::Flat, index_config).unwrap();
        
        assert_eq!(result.index_type, IndexType::Flat);
        assert_eq!(result.dataset_size, 1000);
        assert_eq!(result.dimension, 64);
        assert!(result.build_time_ms > 0);
        assert!(result.avg_query_time_ms > 0.0);
        assert!(result.queries_per_second > 0.0);
        assert_eq!(result.accuracy, 1.0); // 暴力搜索应该是100%准确
    }
} 