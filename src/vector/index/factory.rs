// 向量索引工厂
// 提供创建各种索引类型的工厂方法

use std::sync::{Arc, RwLock};
use bincode;
use std::collections::HashSet;
use std::collections::HashMap;
use rand::seq::SliceRandom;
use rand::seq::IteratorRandom;

use crate::{Error, Result, vector::{Vector, operations::SimilarityMetric}};
use super::types::{IndexType, IndexConfig};
use super::{VectorIndexEnum, VectorIndex};
use super::flat::FlatIndex;
use super::hnsw::HNSWIndex;
use super::ivf::IVFIndex;
use super::pq::PQIndex;
use super::lsh::LSHIndex;
use super::vptree::VPTreeIndex;
use super::ivfpq::IVFPQIndex;
use super::annoy::ANNOYIndex;
// use super::ngt::NGTIndex;
// use super::hierarchical_clustering::HierarchicalClusteringIndex;
// use super::graph::GraphIndex;
use super::ivfhnsw::IVFHNSWIndex;
use crate::vector::optimizer::{AutoTuneConfig, VectorIndexAutoTuner, OptimizationResult};
use crate::vector::utils::benchmark::BenchmarkResult;

/// 向量索引工厂
pub struct VectorIndexFactory;

impl VectorIndexFactory {
    /// 创建指定类型的索引
    pub fn create_index(config: IndexConfig) -> Result<VectorIndexEnum> {
        match config.index_type {
            IndexType::Flat => Ok(VectorIndexEnum::Flat(FlatIndex::new(config))),
            IndexType::HNSW => {
                use super::hnsw::types::DistanceType;
                // Convert SimilarityMetric to DistanceType
                let distance_type = match config.metric {
                    SimilarityMetric::Euclidean => DistanceType::Euclidean,
                    SimilarityMetric::Cosine => DistanceType::Cosine,
                    SimilarityMetric::DotProduct => DistanceType::DotProduct,
                    SimilarityMetric::Manhattan => DistanceType::Manhattan,
                    _ => DistanceType::Euclidean, // Jaccard and others default to Euclidean
                };
                Ok(VectorIndexEnum::HNSW(HNSWIndex::new(
                    config.dimension,
                    config.hnsw_m.max(4),
                    config.hnsw_ef_construction.max(16),
                    distance_type,
                    1.0 / (config.hnsw_m as f32).ln(), // ml
                    config.max_layers.max(16),
                    format!("hnsw_{}", uuid::Uuid::new_v4()),
                )))
            },
            IndexType::IVF => Ok(VectorIndexEnum::IVF(IVFIndex::new(config)?)),
            IndexType::PQ => Ok(VectorIndexEnum::PQ(PQIndex::new(config)?)),
            IndexType::IVFPQ => {
                // 实现IVFPQ索引，结合IVF和PQ
                let mut ivfpq_config = config.clone();
                
                // 优化IVFPQ参数
                if ivfpq_config.ivf_centers == 0 {
                    // 根据数据集大小动态调整聚类中心数量
                    let base_centers = 8;
                    let scale_factor = (ivfpq_config.expected_elements as f32).log10();
                    ivfpq_config.ivf_centers = (base_centers as f32 * scale_factor).round() as usize;
                    ivfpq_config.ivf_centers = ivfpq_config.ivf_centers.max(8).min(ivfpq_config.expected_elements);
                }
                
                if ivfpq_config.pq_subvectors == 0 {
                    // 根据维度动态调整子向量数量
                    let base_subvectors = 4;
                    let dimension_factor = (ivfpq_config.dimension as f32).sqrt();
                    ivfpq_config.pq_subvectors = (base_subvectors as f32 * dimension_factor).round() as usize;
                    ivfpq_config.pq_subvectors = ivfpq_config.pq_subvectors.max(1).min(ivfpq_config.dimension);
                }
                
                if ivfpq_config.pq_nbits == 0 {
                    // 根据数据集大小动态调整量化位数
                    let default_bits = if config.expected_elements > 1000000 {
                        12 // 大数据集使用高精度
                    } else if config.expected_elements > 100000 {
                        10 // 中等数据集
                    } else {
                        8 // 小数据集
                    };
                    ivfpq_config.pq_nbits = default_bits;
                }
                
                // 创建IVFPQ索引实例
                let ivfpq_index = IVFPQIndex::new(ivfpq_config)?;
                Ok(VectorIndexEnum::IVFPQ(ivfpq_index))
            },
            IndexType::IVFHNSW => {
                // 实现IVFHNSW索引，结合IVF和HNSW
                let mut ivfhnsw_config = config.clone();
                
                // 优化IVFHNSW参数
                if ivfhnsw_config.ivf_centers == 0 {
                    // 根据数据集大小动态调整聚类中心数量
                    let base_centers = 8;
                    let scale_factor = (ivfhnsw_config.expected_elements as f32).log10();
                    ivfhnsw_config.ivf_centers = (base_centers as f32 * scale_factor).round() as usize;
                    ivfhnsw_config.ivf_centers = ivfhnsw_config.ivf_centers.max(8).min(ivfhnsw_config.expected_elements);
                }
                
                // 优化HNSW参数
                if ivfhnsw_config.hnsw_m == 0 {
                    // 根据维度调整HNSW的M参数
                    ivfhnsw_config.hnsw_m = 16;  // 默认值
                }
                
                if ivfhnsw_config.hnsw_ef_construction == 0 {
                    // 设置合理的HNSW构建参数
                    ivfhnsw_config.hnsw_ef_construction = 200;  // 默认值
                }
                
                let ivfhnsw_index = IVFHNSWIndex::new(ivfhnsw_config)?;
                Ok(VectorIndexEnum::IVFHNSW(ivfhnsw_index))
            },
            IndexType::LSH => Ok(VectorIndexEnum::LSH(LSHIndex::new(config)?)),
            IndexType::ANNOY => {
                // 实现ANNOY (Approximate Nearest Neighbors Oh Yeah) 索引
                let mut annoy_config = config.clone();
                
                // 优化ANNOY参数
                if annoy_config.annoy_tree_count == 0 {
                    // 根据数据集大小动态调整树的数量
                    let base_trees = 10;
                    let scale_factor = (annoy_config.expected_elements as f32).log10();
                    let accuracy_factor = 1.5; // Default accuracy factor
                    
                    annoy_config.annoy_tree_count = (base_trees as f32 * scale_factor * accuracy_factor).round() as usize;
                    annoy_config.annoy_tree_count = annoy_config.annoy_tree_count.max(10).min(100);
                }
                
                let annoy_index = ANNOYIndex::new(annoy_config)?;
                Ok(VectorIndexEnum::ANNOY(annoy_index))
            },
            IndexType::NGT => {
                // 创建NGT (Neighborhood Graph and Tree) 索引
                use crate::vector::index::NGTIndex;
                let mut ngt_config = config.clone();
                
                // 优化NGT参数
                if ngt_config.ngt_edge_size == 0 {
                    // 根据维度动态调整边数
                    ngt_config.ngt_edge_size = (ngt_config.dimension as f32).sqrt().round() as usize;
                    ngt_config.ngt_edge_size = ngt_config.ngt_edge_size.max(10).min(100);
                }
                
                let ngt_index = NGTIndex::new(ngt_config)?;
                Ok(VectorIndexEnum::NGT(ngt_index))
            },
            IndexType::HierarchicalClustering => {
                // 创建层次聚类索引
                use crate::vector::index::HierarchicalClusteringIndex;
                let mut hc_config = config.clone();
                
                // 优化层次聚类参数
                if hc_config.cluster_levels == 0 {
                    // 根据数据集大小动态调整聚类层数
                    let base_levels = 3;
                    let scale_factor = (hc_config.expected_elements as f32).log10();
                    hc_config.cluster_levels = (base_levels as f32 * scale_factor).round() as usize;
                    hc_config.cluster_levels = hc_config.cluster_levels.max(2).min(10);
                }
                
                let hc_index = HierarchicalClusteringIndex::new(hc_config)?;
                Ok(VectorIndexEnum::HierarchicalClustering(hc_index))
            },
            IndexType::GraphIndex => {
                // 通用图索引，区别于HNSW的另一种图索引实现
                use crate::vector::index::GraphIndex;
                let mut graph_config = config.clone();
                
                // 优化图索引参数
                if graph_config.graph_degree == 0 {
                    // 根据维度动态调整图的度数
                    graph_config.graph_degree = (graph_config.dimension as f32).sqrt().round() as usize;
                    graph_config.graph_degree = graph_config.graph_degree.max(4).min(32);
                }
                
                let graph_index = GraphIndex::new(graph_config)?;
                Ok(VectorIndexEnum::GraphIndex(graph_index))
            },
            IndexType::VPTree => Ok(VectorIndexEnum::VPTree(VPTreeIndex::new(config)?)),
            _ => Err(Error::vector(format!("Unsupported index type: {:?}", config.index_type))),
        }
    }
    
    /// 从序列化数据创建索引
    pub fn from_serialized(data: &[u8], config: IndexConfig) -> Result<Arc<RwLock<VectorIndexEnum>>> {
        let index_type = bincode::deserialize::<IndexType>(&data[0..std::mem::size_of::<IndexType>()])
            .map_err(|e| Error::vector(format!("Failed to deserialize index type: {}", e)))?;
        
        let index_data = &data[std::mem::size_of::<IndexType>()..];
        
        let index = match index_type {
            IndexType::Flat => {
                let flat_index = FlatIndex::new(config);
                let mut index = VectorIndexEnum::Flat(flat_index);
                index.deserialize(index_data)?;
                index
            },
            IndexType::HNSW => {
                use super::hnsw::types::DistanceType;
                // Convert SimilarityMetric to DistanceType
                let distance_type = match config.metric {
                    SimilarityMetric::Euclidean => DistanceType::Euclidean,
                    SimilarityMetric::Cosine => DistanceType::Cosine,
                    SimilarityMetric::DotProduct => DistanceType::DotProduct,
                    SimilarityMetric::Manhattan => DistanceType::Manhattan,
                    _ => DistanceType::Euclidean,
                };
                let hnsw_index = HNSWIndex::new(
                    config.dimension,
                    config.hnsw_m.max(4),
                    config.hnsw_ef_construction.max(16),
                    distance_type,
                    1.0 / (config.hnsw_m as f32).ln(), // ml
                    config.max_layers.max(16),
                    format!("hnsw_{}", uuid::Uuid::new_v4()),
                );
                let mut index = VectorIndexEnum::HNSW(hnsw_index);
                index.deserialize(index_data)?;
                index
            },
            IndexType::IVF => {
                let ivf_index = IVFIndex::new(config)?;
                let mut index = VectorIndexEnum::IVF(ivf_index);
                index.deserialize(index_data)?;
                index
            },
            IndexType::PQ => {
                let pq_index = PQIndex::new(config)?;
                let mut index = VectorIndexEnum::PQ(pq_index);
                index.deserialize(index_data)?;
                index
            },
            IndexType::IVFPQ => {
                let ivfpq_index = IVFPQIndex::new(config)?;
                let mut index = VectorIndexEnum::IVFPQ(ivfpq_index);
                index.deserialize(index_data)?;
                index
            },
            IndexType::IVFHNSW => {
                let ivfhnsw_index = IVFHNSWIndex::new(config)?;
                let mut index = VectorIndexEnum::IVFHNSW(ivfhnsw_index);
                index.deserialize(index_data)?;
                index
            },
            IndexType::LSH => {
                let lsh_index = LSHIndex::new(config)?;
                let mut index = VectorIndexEnum::LSH(lsh_index);
                index.deserialize(index_data)?;
                index
            },
            IndexType::ANNOY => {
                let annoy_index = ANNOYIndex::new(config)?;
                let mut index = VectorIndexEnum::ANNOY(annoy_index);
                index.deserialize(index_data)?;
                index
            },
            IndexType::NGT => {
                use crate::vector::index::NGTIndex;
                let ngt_index = NGTIndex::new(config)?;
                let mut index = VectorIndexEnum::NGT(ngt_index);
                index.deserialize(index_data)?;
                index
            },
            IndexType::HierarchicalClustering => {
                use crate::vector::index::HierarchicalClusteringIndex;
                let hc_index = HierarchicalClusteringIndex::new(config)?;
                let mut index = VectorIndexEnum::HierarchicalClustering(hc_index);
                index.deserialize(index_data)?;
                index
            },
            IndexType::GraphIndex => {
                use crate::vector::index::GraphIndex;
                let graph_index = GraphIndex::new(config)?;
                let mut index = VectorIndexEnum::GraphIndex(graph_index);
                index.deserialize(index_data)?;
                index
            },
            IndexType::VPTree => {
                let vptree_index = VPTreeIndex::new(config)?;
                let mut index = VectorIndexEnum::VPTree(vptree_index);
                index.deserialize(index_data)?;
                index
            },
            _ => return Err(Error::vector(format!("Unsupported index type for deserialization: {:?}", index_type))),
        };
        
        Ok(Arc::new(RwLock::new(index)))
    }
    
    /// 序列化索引
    pub fn serialize_index(index: &Arc<RwLock<VectorIndexEnum>>, index_type: IndexType) -> Result<Vec<u8>> {
        let index_read = index.read().unwrap();
        let index_data = index_read.serialize()?;
        
        let mut data = Vec::with_capacity(std::mem::size_of::<IndexType>() + index_data.len());
        let index_type_data = bincode::serialize(&index_type)
            .map_err(|e| Error::vector(format!("Failed to serialize index type: {}", e)))?;
        
        data.extend_from_slice(&index_type_data);
        data.extend_from_slice(&index_data);
        
        Ok(data)
    }
    
    /// 创建最适合给定数据集的索引
    pub fn create_optimal_index(
        vectors: &[Vector], 
        dimension: usize, 
        metric: SimilarityMetric,
        optimization_target: crate::vector::optimizer::OptimizationTarget
    ) -> Result<(Arc<RwLock<VectorIndexEnum>>, IndexType, IndexConfig)> {
        use crate::vector::optimizer::{IndexOptimizer, OptimizerConfig};
        use crate::vector::benchmark::BenchmarkConfig;
        
        // 检查输入有效性
        if vectors.is_empty() {
            return Err(Error::vector("无法为空向量集创建最优索引".to_string()));
        }
        
        if dimension == 0 {
            return Err(Error::vector("维度必须大于0".to_string()));
        }
        
        // 分析数据集特征
        let dataset_size = vectors.len();
        let is_large_dataset = dataset_size > 1_000_000;
        let is_medium_dataset = dataset_size > 10_000 && dataset_size <= 1_000_000;
        let is_small_dataset = dataset_size <= 10_000;
        
        let is_high_dim = dimension > 100;
        let is_medium_dim = dimension > 10 && dimension <= 100;
        let is_low_dim = dimension <= 10;
        
        // 根据数据特征和优化目标预先筛选可能的索引类型
        let candidate_index_types = match (optimization_target, is_large_dataset, is_medium_dataset, is_small_dataset, is_high_dim, is_medium_dim, is_low_dim) {
            // 优化查询速度
            (crate::vector::optimizer::OptimizationTarget::QuerySpeed, true, _, _, true, _, _) => 
                // 大数据集+高维 => IVFHNSW, HNSW, IVFPQ
                vec![IndexType::IVFHNSW, IndexType::HNSW, IndexType::IVFPQ],
            (crate::vector::optimizer::OptimizationTarget::QuerySpeed, true, _, _, _, true, _) => 
                // 大数据集+中维 => IVFHNSW, HNSW, IVFPQ
                vec![IndexType::IVFHNSW, IndexType::HNSW, IndexType::IVFPQ],
            (crate::vector::optimizer::OptimizationTarget::QuerySpeed, true, _, _, _, _, true) => 
                // 大数据集+低维 => IVFPQ, IVF, HNSW
                vec![IndexType::IVFPQ, IndexType::IVF, IndexType::HNSW],
            
            (crate::vector::optimizer::OptimizationTarget::QuerySpeed, _, true, _, _, _, _) => 
                // 中等数据集 => HNSW, IVFHNSW, ANNOY
                vec![IndexType::HNSW, IndexType::IVFHNSW, IndexType::ANNOY],
            
            (crate::vector::optimizer::OptimizationTarget::QuerySpeed, _, _, true, _, _, _) => 
                // 小数据集 => HNSW, ANNOY, Flat
                vec![IndexType::HNSW, IndexType::ANNOY, IndexType::Flat],
            
            // 优化内存使用
            (crate::vector::optimizer::OptimizationTarget::MemoryUsage, true, _, _, _, _, _) => 
                // 大数据集 => IVFPQ, LSH, IVF
                vec![IndexType::IVFPQ, IndexType::LSH, IndexType::IVF],
            
            (crate::vector::optimizer::OptimizationTarget::MemoryUsage, _, true, _, _, _, _) => 
                // 中等数据集 => IVFPQ, LSH, PQ
                vec![IndexType::IVFPQ, IndexType::LSH, IndexType::PQ],
            
            (crate::vector::optimizer::OptimizationTarget::MemoryUsage, _, _, true, _, _, _) => 
                // 小数据集 => LSH, PQ, VPTree
                vec![IndexType::LSH, IndexType::PQ, IndexType::VPTree],
            
            // 优化准确率
            (crate::vector::optimizer::OptimizationTarget::Accuracy, true, _, _, _, _, _) => 
                // 大数据集 => HNSW, IVFHNSW, Flat(如果内存允许)
                vec![IndexType::HNSW, IndexType::IVFHNSW, IndexType::IVF],
            
            (crate::vector::optimizer::OptimizationTarget::Accuracy, _, true, _, _, _, _) => 
                // 中等数据集 => HNSW, Flat, IVFHNSW
                vec![IndexType::HNSW, IndexType::Flat, IndexType::IVFHNSW],
            
            (crate::vector::optimizer::OptimizationTarget::Accuracy, _, _, true, _, _, _) => 
                // 小数据集 => Flat, HNSW, VPTree
                vec![IndexType::Flat, IndexType::HNSW, IndexType::VPTree],
            
            // 平衡性能
            (crate::vector::optimizer::OptimizationTarget::BalancedPerformance, true, _, _, _, _, _) => 
                // 大数据集 => IVFHNSW, HNSW, IVF
                vec![IndexType::IVFHNSW, IndexType::HNSW, IndexType::IVF],
            
            (crate::vector::optimizer::OptimizationTarget::BalancedPerformance, _, true, _, _, _, _) => 
                // 中等数据集 => HNSW, IVFHNSW, ANNOY
                vec![IndexType::HNSW, IndexType::IVFHNSW, IndexType::ANNOY],
            
            (crate::vector::optimizer::OptimizationTarget::BalancedPerformance, _, _, true, _, _, _) => 
                // 小数据集 => HNSW, Flat, ANNOY
                vec![IndexType::HNSW, IndexType::Flat, IndexType::ANNOY],
            
            // 优化构建速度
            (crate::vector::optimizer::OptimizationTarget::BuildSpeed, true, _, _, _, _, _) => 
                // 大数据集 => IVF, LSH, ANNOY
                vec![IndexType::IVF, IndexType::LSH, IndexType::ANNOY],
            
            (crate::vector::optimizer::OptimizationTarget::BuildSpeed, _, true, _, _, _, _) => 
                // 中等数据集 => LSH, ANNOY, IVF
                vec![IndexType::LSH, IndexType::ANNOY, IndexType::IVF],
            
            (crate::vector::optimizer::OptimizationTarget::BuildSpeed, _, _, true, _, _, _) => 
                // 小数据集 => Flat, LSH, ANNOY
                vec![IndexType::Flat, IndexType::LSH, IndexType::ANNOY],
            
            // 默认返回全部索引类型以进行全面优化
            _ => vec![
                IndexType::HNSW, 
                IndexType::IVFHNSW, 
                IndexType::IVF, 
                IndexType::IVFPQ,
                IndexType::Flat,
                IndexType::ANNOY,
                IndexType::LSH,
                IndexType::VPTree
            ],
        };

        // 生成查询样本 - 从向量数据集随机抽取部分作为查询样本
        let mut rng = rand::thread_rng();
        let query_count = 10.min(dataset_size / 10); // 至少10个查询，最多是数据集的1/10
        let mut queries = Vec::with_capacity(query_count);
        
        // 从数据集中随机抽取查询样本
        let sample_indices: Vec<usize> = (0..dataset_size)
            .choose_multiple(&mut rng, query_count);
            
        for &idx in &sample_indices {
            queries.push(vectors[idx].data.clone());
        }
        
        // 创建优化器配置
        let optimizer_config = OptimizerConfig {
            benchmark_config: BenchmarkConfig {
                dataset_size,
                dimension,
                metric,
                top_k: 10, // 默认查询返回10个结果
                query_size: queries.len(),
                repeat_count: 3,
                test_accuracy: true,
                test_memory: true,
                test_index_size: true,
                parallel: true,
                random_seed: None,
            },
            target: optimization_target,
            max_iterations: 5, // 设置较小的迭代次数以控制优化时间
            parallel: true,
            random_seed: None,
            use_bayesian: false,
            use_grid_search: true, // 对预选的索引类型使用网格搜索
            use_random_search: false,
            use_genetic_algorithm: false,
            dataset: None,
        };
        
        // 创建优化器
        let optimizer: IndexOptimizer = IndexOptimizer::new(optimizer_config);
        
        // 优化每种候选索引类型
        let mut optimization_results = HashMap::new();
        for index_type in candidate_index_types {
            println!("正在优化索引类型: {:?}", index_type);
            match optimizer.optimize(index_type) {
                Ok(result) => {
                    optimization_results.insert(index_type, result);
                },
                Err(e) => {
                    println!("优化 {:?} 失败: {}", index_type, e);
                }
            }
        }
        
        // 如果没有获得任何结果，则使用默认索引
        if optimization_results.is_empty() {
            let default_config = IndexConfig {
                index_type: IndexType::HNSW, // 默认使用HNSW
                dimension,
                metric,
                hnsw_m: 16, // 合理的默认值
                hnsw_ef_construction: 200,
                hnsw_ef_search: 50,
                ..Default::default()
            };
            
            let index = Self::create_index(default_config.clone())?;
            return Ok((Arc::new(RwLock::new(index)), IndexType::HNSW, default_config));
        }
        
        // 找出最佳的索引类型和配置
        let mut best_index_type = optimization_results.keys().next().unwrap().clone();
        let mut best_score = optimization_results[&best_index_type].score;
        
        for (index_type, result) in &optimization_results {
            if result.score > best_score {
                best_score = result.score;
                best_index_type = *index_type;
            }
        }
        
        // 获取最佳配置
        let best_config = optimization_results[&best_index_type].best_config.clone();
        
        // 创建最优索引
        let index = Self::create_index(best_config.clone())?;
        
        // 打印最佳索引性能数据
        let performance = &optimization_results[&best_index_type].performance;
        println!("最佳索引类型: {:?}", best_index_type);
        println!("查询时间: {:.3} ms", performance.avg_query_time_ms);
        println!("构建时间: {:.3} ms", performance.build_time_ms);
        println!("内存使用: {:.2} MB", performance.memory_usage_bytes as f64 / 1024.0 / 1024.0);
        println!("召回率: {:.2}%", performance.metrics.recall_rate * 100.0);
        
        Ok((Arc::new(RwLock::new(index)), best_index_type, best_config))
    }
    
    /// 运行基准测试并返回所有索引类型的性能结果
    pub fn benchmark_indexes(
        vectors: &[Vector], 
        queries: &[Vec<f32>], 
        dimension: usize,
        top_k: usize
    ) -> Result<Vec<crate::vector::benchmark::BenchmarkResult>> {
        use crate::vector::benchmark::BenchmarkResult;
        use std::time::{Instant, Duration};
        
        // 检查输入
        if vectors.is_empty() {
            return Err(Error::vector("Empty vector set for benchmark".to_string()));
        }
        
        if queries.is_empty() {
            return Err(Error::vector("Empty query set for benchmark".to_string()));
        }
        
        // 构建参考结果（暴力搜索作为基准）
        println!("构建精确搜索参考结果...");
        let flat_config = IndexConfig {
            index_type: IndexType::Flat,
            dimension,
            ..Default::default()
        };
        
        let mut reference_index = FlatIndex::new(flat_config);
        
        // 添加向量到索引
        let start_time = Instant::now();
        for vector in vectors {
            reference_index.add(vector.clone())?;
        }
        let build_duration = start_time.elapsed();
        
        // 执行参考查询
        let mut reference_results = Vec::new();
        let mut total_query_time = Duration::new(0, 0);
        
        for query in queries {
            let query_start = Instant::now();
            let results = reference_index.search(query, top_k)?;
            total_query_time += query_start.elapsed();
            
            reference_results.push(results);
        }
        
        let avg_query_time = total_query_time.as_secs_f64() / queries.len() as f64;
        
        // 添加参考基准结果
        let mut benchmark_results = Vec::new();
        let memory_usage = reference_index.get_memory_usage()?;
        let queries_per_second = if avg_query_time > 0.0 { 1.0 / avg_query_time } else { 0.0 };
        let dataset_size = vectors.len();
        benchmark_results.push(BenchmarkResult {
            index_type: IndexType::Flat,
            config: flat_config.clone(),
            metrics: crate::vector::utils::benchmark::IndexPerformanceMetrics {
                build_time_ms: build_duration.as_millis() as f64,
                avg_query_time_ms: avg_query_time * 1000.0,
                memory_usage_bytes: memory_usage as u64,
                index_size_bytes: 0, // 内存索引无持久化大小
                recall_rate: 1.0, // 作为基准，召回率是1.0
                accuracy: 1.0,    // 作为基准，准确率是1.0
            },
            dataset_size: dataset_size,
            dimension: dimension,
            build_time_ms: build_duration.as_millis() as u64,
            avg_query_time_ms: avg_query_time * 1000.0,
            queries_per_second: queries_per_second,
            memory_usage_bytes: memory_usage,
            accuracy: 1.0,
            index_size_bytes: 0,
        });
        
        // 要测试的索引类型
        let index_types = vec![
            IndexType::HNSW,
            IndexType::IVF,
            IndexType::IVFHNSW,
            IndexType::IVFPQ,
            IndexType::LSH,
            IndexType::ANNOY,
            IndexType::VPTree,
            // 可以添加更多索引类型
        ];
        
        // 对每种索引类型进行基准测试
        for index_type in index_types {
            println!("测试索引类型: {:?}", index_type);
            
            // 生成索引配置
            let mut config = IndexConfig {
                index_type,
                dimension,
                ..Default::default()
            };
            
            // 根据索引类型设置合理的默认参数
            match index_type {
                IndexType::HNSW => {
                    config.hnsw_m = 16;
                    config.hnsw_ef_construction = 200;
                    config.hnsw_ef_search = 50;
                },
                IndexType::IVF => {
                    config.ivf_nlist = (vectors.len() as f32).sqrt().round() as usize;
                    config.ivf_nlist = config.ivf_nlist.max(10).min(100);
                    config.ivf_nprobe = (config.ivf_nlist as f32 * 0.1).ceil() as usize;
                },
                IndexType::IVFHNSW => {
                    config.ivfhnsw_nlist = (vectors.len() as f32).sqrt().round() as usize;
                    config.ivfhnsw_nlist = config.ivfhnsw_nlist.max(10).min(100);
                    config.ivfhnsw_nprobe = (config.ivfhnsw_nlist as f32 * 0.1).ceil() as usize;
                    config.ivfhnsw_m = 16;
                    config.ivfhnsw_ef_construction = 200;
                    config.ivfhnsw_ef_search = 50;
                },
                IndexType::IVFPQ => {
                    config.ivf_centers = (vectors.len() as f32).sqrt().round() as usize;
                    config.ivf_centers = config.ivf_centers.max(10).min(100);
                    config.pq_subvectors = dimension / 4;
                    config.pq_subvector_bits = 8;
                },
                IndexType::LSH => {
                    config.lsh_hash_count = dimension / 8;
                    config.lsh_hash_count = config.lsh_hash_count.max(4).min(32);
                    config.lsh_hash_length = 16;
                },
                IndexType::ANNOY => {
                    config.annoy_tree_count = 10 + (dimension / 10);
                    config.annoy_tree_count = config.annoy_tree_count.max(10).min(50);
                },
                _ => {}
            }
            
            // 尝试创建并测试索引
            let create_result = (|| {
                let index_result = Self::create_index(config.clone());
                let mut index = match index_result {
                    Ok(idx) => idx,
                    Err(e) => {
                        println!("创建索引 {:?} 失败: {}", index_type, e);
                        return Err(e);
                    }
                };
                
                // 构建索引
                let build_start = Instant::now();
                for vector in vectors {
                    if let Err(e) = index.add(vector.clone()) {
                        println!("添加向量到 {:?} 索引失败: {}", index_type, e);
                        return Err(e);
                    }
                }
                let build_time = build_start.elapsed();
                
                // 内存使用
                let memory_usage = index.get_memory_usage()?;
                
                // 查询性能
                let mut test_results = Vec::new();
                let mut total_time = Duration::new(0, 0);
                
                for query in queries {
                    let query_start = Instant::now();
                    let results = index.search(query, top_k)?;
                    total_time += query_start.elapsed();
                    test_results.push(results);
                }
                
                let avg_time = total_time.as_secs_f64() / queries.len() as f64;
                
                // 计算召回率
                let mut total_recall = 0.0;
                for (i, reference) in reference_results.iter().enumerate() {
                    let test = &test_results[i];
                    
                    // 创建参考结果的ID集合
                    let reference_ids: HashSet<String> = reference.iter()
                        .map(|r| r.id.clone())
                        .collect();
                    
                    // 计算共同结果数量
                    let common_count = test.iter()
                        .filter(|r| reference_ids.contains(&r.id))
                        .count();
                    
                    // 计算这个查询的召回率
                    let recall = if reference.is_empty() {
                        1.0 // 如果参考结果为空，则召回率为1
                    } else {
                        common_count as f64 / reference.len().min(top_k) as f64
                    };
                    
                    total_recall += recall;
                }
                
                let avg_recall = total_recall / queries.len() as f64;
                let queries_per_second = if avg_time > 0.0 { 1.0 / avg_time } else { 0.0 };
                
                // 保存基准测试结果
                Ok(BenchmarkResult {
                    index_type,
                    config: config.clone(),
                    metrics: crate::vector::utils::benchmark::IndexPerformanceMetrics {
                        build_time_ms: build_time.as_millis() as f64,
                        avg_query_time_ms: avg_time * 1000.0,
                        memory_usage_bytes: memory_usage as u64,
                        index_size_bytes: 0, // 内存索引无持久化大小
                        recall_rate: avg_recall,
                        accuracy: avg_recall, // 简化：使用召回率作为准确率近似值
                    },
                    dataset_size: vectors.len(),
                    dimension: dimension,
                    build_time_ms: build_time.as_millis() as u64,
                    avg_query_time_ms: avg_time * 1000.0,
                    queries_per_second: queries_per_second,
                    memory_usage_bytes: memory_usage,
                    accuracy: avg_recall,
                    index_size_bytes: 0,
                })
            })();
            
            // 处理结果
            match create_result {
                Ok(result) => benchmark_results.push(result),
                Err(e) => println!("索引 {:?} 基准测试失败: {}", index_type, e),
            }
        }
        
        Ok(benchmark_results)
    }
    
    /// 根据给定的索引类型和配置创建索引，并添加向量
    pub fn build_index(
        vectors: &[Vector], 
        index_type: IndexType, 
        config: Option<IndexConfig>
    ) -> Result<Arc<RwLock<VectorIndexEnum>>> {
        if vectors.is_empty() {
            return Err(Error::vector("Cannot build index with empty vectors".to_string()));
        }
        
        let dimension = vectors[0].data.len();
        
        // 使用提供的配置或创建默认配置
        let mut actual_config = config.unwrap_or_else(|| {
            let mut default_config = IndexConfig::default();
            default_config.index_type = index_type;
            default_config.dimension = dimension;
            default_config
        });
        
        // 确保配置中的维度正确
        actual_config.dimension = dimension;
        
        // 创建索引
        let index = Self::create_index(actual_config)?;
        let index = Arc::new(RwLock::new(index));
        
        // 添加向量到索引
        {
            let mut index_write = index.write().unwrap();
            for vector in vectors {
                index_write.add(vector.clone())?;
            }
        }
        
        Ok(index)
    }
    
    /// 比较不同索引类型的性能并返回详细报告
    pub fn compare_indexes(vectors: &[Vector], dimension: usize) -> Result<String> {
        use crate::vector::benchmark::{IndexBenchmark, BenchmarkConfig};
        
        // 创建基准测试配置
        let benchmark_config = BenchmarkConfig {
            dataset_size: vectors.len(),
            dimension,
            ..Default::default()
        };
        
        // 创建基准测试器
        let benchmark = IndexBenchmark::new(benchmark_config);
        
        // 运行所有索引类型的基准测试
        let results = benchmark.benchmark_all()?;
        
        // 生成比较报告
        let report = benchmark.generate_report(&results);
        
        Ok(report)
    }

    /// 轻量级就绪检查：尝试为所有受支持的索引类型创建最小实例
    /// 不引入数据，也不执行构建，确保工厂与实现接线正常
    pub fn readiness_probe() -> Result<()> {
        let supported = vec![
            IndexType::Flat,
            IndexType::HNSW,
            IndexType::IVF,
            IndexType::PQ,
            IndexType::IVFPQ,
            IndexType::IVFHNSW,
            IndexType::LSH,
            IndexType::ANNOY,
            IndexType::NGT,
            IndexType::HierarchicalClustering,
            IndexType::GraphIndex,
            IndexType::VPTree,
        ];

        for idx in supported {
            let mut cfg = IndexConfig::default();
            cfg.index_type = idx;
            cfg.dimension = 2;
            cfg.expected_elements = 1;
            // 某些类型需要基本参数，给出安全的最小值
            cfg.hnsw_m = cfg.hnsw_m.max(4);
            cfg.hnsw_ef_construction = cfg.hnsw_ef_construction.max(32);
            cfg.ivf_nlist = cfg.ivf_nlist.max(10);
            cfg.lsh_hash_count = cfg.lsh_hash_count.max(4);
            cfg.annoy_tree_count = cfg.annoy_tree_count.max(10);

            let _ = Self::create_index(cfg)?;
        }
        Ok(())
    }

    /// 基于自动基准与自动优化选择最佳索引并创建实例
    /// 返回 (索引实例, 最佳配置, 基准结果)
    pub fn auto_select_and_create_index(auto_cfg: AutoTuneConfig) -> Result<(VectorIndexEnum, IndexConfig, BenchmarkResult)> {
        // 运行自动调优，获取各索引类型的优化结果
        let mut tuner = VectorIndexAutoTuner::new(auto_cfg);
        let results = tuner.auto_tune()?;

        // 选择分数最高的索引类型
        let mut best_type: Option<super::types::IndexType> = None;
        let mut best_res: Option<&OptimizationResult> = None;
        for (idx_ty, res) in &results {
            if let Some(cur) = best_res {
                if res.score > cur.score {
                    best_type = Some(*idx_ty);
                    best_res = Some(res);
                }
            } else {
                best_type = Some(*idx_ty);
                best_res = Some(res);
            }
        }

        let chosen = best_res.ok_or_else(|| Error::vector("Auto tuning did not produce any result".to_string()))?;
        let best_config = chosen.best_config.clone();
        let bench = chosen.performance.clone();

        // 创建最佳索引实例
        let index = Self::create_index(best_config.clone())?;
        Ok((index, best_config, bench))
    }
} 