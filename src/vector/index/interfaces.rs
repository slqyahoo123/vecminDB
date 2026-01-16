// 向量索引接口定义
// 定义所有向量索引实现必须遵循的接口

use crate::{Result, vector::Vector};
use super::types::{IndexConfig, SearchResult};
use serde_json;

/// 向量索引接口
pub trait VectorIndex: Send + Sync {
    /// 添加一个向量到索引
    fn add(&mut self, vector: Vector) -> Result<()>;
    
    /// 在索引中搜索与查询向量最相似的向量
    fn search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchResult>>;
    
    /// 在索引中搜索指定半径范围内的所有向量
    /// 
    /// # 参数
    /// * `query` - 查询向量
    /// * `radius` - 搜索半径，满足 distance(query, vector) <= radius 的向量将被返回
    /// * `with_vectors` - 是否在结果中包含向量数据
    /// * `with_metadata` - 是否在结果中包含元数据
    /// * `max_elements` - 最多返回的结果数量，为 None 时返回所有满足条件的结果
    /// * `dynamic_ef` - 是否动态调整搜索参数以提高质量，默认为 false
    /// 
    /// # 返回值
    /// 返回搜索结果的列表，按距离升序排序
    fn range_search(&self, 
                   query: &[f32], 
                   radius: f32, 
                   with_vectors: bool, 
                   with_metadata: bool,
                   max_elements: Option<usize>, 
                   dynamic_ef: bool) -> Result<Vec<SearchResult>> {
        use log::{debug, warn};
        use std::cmp::Ordering;
        
        debug!("执行范围搜索: radius={}, max_elements={:?}, dynamic_ef={}", 
               radius, max_elements, dynamic_ef);
        if let Some(max) = max_elements {
            if max == 0 {
                warn!("范围搜索的 max_elements=0，将返回空结果");
            } else if max > 100_000 {
                warn!("范围搜索的 max_elements 非常大 ({}): 可能影响性能", max);
            }
        }
        
        // 参数验证
        if query.is_empty() {
            return Err(crate::Error::InvalidInput("查询向量不能为空".to_string()));
        }
        
        if radius < 0.0 {
            return Err(crate::Error::InvalidInput("搜索半径不能为负数".to_string()));
        }
        
        if radius == 0.0 {
            return Ok(Vec::new()); // 零半径返回空结果
        }
        
        // 执行近似最近邻搜索，用更大的k值来捕获更多候选
        let initial_k = max_elements.unwrap_or(10000).max(1000);
        let candidates = self.search(query, initial_k)?;
        
        // 过滤距离在半径内的结果
        let mut filtered_results = Vec::new();
        for candidate in candidates {
            if candidate.distance <= radius {
                let result = SearchResult {
                    id: candidate.id,
                    distance: candidate.distance,
                    metadata: if with_metadata { candidate.metadata } else { None },
                };
                
                filtered_results.push(result);
                
                // 检查是否达到最大元素数量限制
                if let Some(max) = max_elements {
                    if filtered_results.len() >= max {
                        break;
                    }
                }
            }
        }
        
        // 按距离排序
        filtered_results.sort_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal)
        });
        
        // 动态ef调整：如果结果太少且dynamic_ef为true，尝试扩大搜索范围
        if dynamic_ef && filtered_results.len() < max_elements.unwrap_or(10).min(10) {
            warn!("结果数量较少，启用动态ef调整以尝试提升召回");
            
            // 增加搜索范围并重试
            let expanded_k = initial_k * 2;
            let expanded_candidates = self.search(query, expanded_k)?;
            
            filtered_results.clear();
            for candidate in expanded_candidates {
                if candidate.distance <= radius {
                    let result = SearchResult {
                        id: candidate.id,
                        distance: candidate.distance,
                        metadata: if with_metadata { candidate.metadata } else { None },
                    };
                    
                    filtered_results.push(result);
                    
                    if let Some(max) = max_elements {
                        if filtered_results.len() >= max {
                            break;
                        }
                    }
                }
            }
            
            filtered_results.sort_by(|a, b| {
                a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal)
            });
        }
        
        debug!("范围搜索完成，返回{}个结果", filtered_results.len());
        Ok(filtered_results)
    }
    
    /// 在索引中执行线性搜索，适用于小型数据集或作为基准测试
    /// 
    /// # 参数
    /// * `query` - 查询向量
    /// * `limit` - 最多返回的结果数量
    /// 
    /// # 返回值
    /// 返回搜索结果的列表，按距离升序排序
    fn linear_search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        use log::{debug, info, warn};
        use std::collections::BinaryHeap;
        use std::cmp::Reverse;
        
        debug!("执行线性搜索: limit={}", limit);
        
        // 参数验证
        if query.is_empty() {
            warn!("线性搜索收到空查询向量");
            return Err(crate::Error::InvalidInput("查询向量不能为空".to_string()));
        }
        
        if limit == 0 {
            warn!("线性搜索的 limit=0，将返回空结果");
            return Ok(Vec::new());
        }
        
        // 获取所有向量（这需要索引支持完整遍历）
        let all_vectors = self.get_all_vectors()?;
        
        if all_vectors.is_empty() {
            debug!("索引为空，返回空结果");
            return Ok(Vec::new());
        }
        
        if all_vectors.len() > 1_000_000 {
            warn!("线性搜索将在非常大的索引上运行 ({} 向量): 建议使用近似索引", all_vectors.len());
        }
        info!("对{}个向量执行线性搜索", all_vectors.len());
        
        // 使用最小堆维护top-k结果
        let mut heap = BinaryHeap::new();
        
        for (i, vector_data) in all_vectors.iter().enumerate() {
            // 计算欧几里得距离
            let distance = calculate_euclidean_distance(query, &vector_data.vector)?;
            
            let result = SearchResult {
                id: vector_data.id.to_string(),
                distance,
                metadata: vector_data.metadata.as_ref().map(|m| {
                    serde_json::to_value(m).unwrap_or(serde_json::Value::Null)
                }),
            };
            
            if heap.len() < limit {
                heap.push(Reverse(OrderedSearchResult(result)));
            } else if let Some(Reverse(OrderedSearchResult(worst))) = heap.peek() {
                if distance < worst.distance {
                    heap.pop();
                    heap.push(Reverse(OrderedSearchResult(result)));
                }
            }
            
            // 每处理1000个向量记录一次进度
            if (i + 1) % 1000 == 0 {
                debug!("已处理 {}/{} 个向量", i + 1, all_vectors.len());
            }
        }
        
        // 从堆中提取结果并排序
        let mut results: Vec<SearchResult> = heap
            .into_iter()
            .map(|Reverse(OrderedSearchResult(result))| result)
            .collect();
        
        results.sort_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        info!("线性搜索完成，返回{}个结果", results.len());
        Ok(results)
    }
    
    /// 从索引中删除向量
    /// 
    /// 返回一个布尔值，表示是否成功删除了向量
    /// - true: 向量存在并被删除
    /// - false: 向量不存在
    fn delete(&mut self, id: &str) -> Result<bool>;
    
    /// 返回索引中向量的数量
    fn size(&self) -> usize;
    
    /// 检查索引中是否包含指定ID的向量
    fn contains(&self, id: &str) -> bool;
    
    /// 返回索引向量的维度
    fn dimension(&self) -> usize;
    
    /// 获取索引配置
    fn get_config(&self) -> IndexConfig;
    
    /// 批量插入向量
    fn batch_insert(&mut self, vectors: &[Vector]) -> Result<()> {
        for vector in vectors {
            self.add(vector.clone())?;
        }
        Ok(())
    }
    
    /// 批量添加向量（别名方法）
    fn batch_add(&mut self, vectors: &[Vector]) -> Result<()> {
        self.batch_insert(vectors)
    }
    
    /// 批量删除向量
    /// 
    /// 返回成功删除的向量数量
    fn batch_delete(&mut self, ids: &[String]) -> Result<usize> {
        let mut deleted_count = 0;
        for id in ids {
            if self.delete(id)? {
                deleted_count += 1;
            }
        }
        Ok(deleted_count)
    }
    
    /// 批量更新向量
    /// 
    /// 用于高效地更新多个向量，比单独更新每个向量更高效
    /// 
    /// # 参数
    /// * `vectors` - 要更新的向量列表
    /// 
    /// # 返回值
    /// * `Result<usize>` - 成功更新的向量数量
    fn batch_update(&mut self, vectors: &[Vector]) -> Result<usize> {
        let mut updated_count = 0;
        
        // 默认实现是先删除再添加
        for vector in vectors {
            if self.delete(&vector.id).unwrap_or(false) {
                self.add(vector.clone())?;
                updated_count += 1;
            }
        }
        
        Ok(updated_count)
    }
    
    /// 批量执行范围搜索，处理多个查询向量
    /// 
    /// # 参数
    /// * `queries` - 查询向量列表
    /// * `radius` - 搜索半径，满足 distance(query, vector) <= radius 的向量将被返回
    /// * `with_vectors` - 是否在结果中包含向量数据
    /// * `with_metadata` - 是否在结果中包含元数据
    /// * `max_elements` - 最多返回的结果数量，为 None 时返回所有满足条件的结果
    /// * `dynamic_ef` - 是否动态调整搜索参数以提高质量
    /// 
    /// # 返回值
    /// 返回每个查询向量对应的搜索结果列表
    fn batch_range_search(&self, 
                         queries: &[Vec<f32>], 
                         radius: f32, 
                         with_vectors: bool, 
                         with_metadata: bool,
                         max_elements: Option<usize>, 
                         dynamic_ef: bool) -> Result<Vec<Vec<SearchResult>>> {
        // 默认实现是按序执行每个查询
        let mut results = Vec::with_capacity(queries.len());
        for query in queries {
            results.push(self.range_search(query, radius, with_vectors, with_metadata, max_elements, dynamic_ef)?);
        }
        Ok(results)
    }
    
    /// 获取内存使用量
    fn get_memory_usage(&self) -> Result<usize> {
        Ok(std::mem::size_of_val(self))
    }
    
    /// 序列化索引
    fn serialize(&self) -> Result<Vec<u8>>;
    
    /// 反序列化索引
    fn deserialize(&mut self, data: &[u8]) -> Result<()>;
    
    /// 创建索引的深拷贝并装箱
    fn clone_box(&self) -> Box<dyn VectorIndex + Send + Sync>;
    
    /// 从字节数组创建索引并装箱（静态方法）
    fn deserialize_box(data: &[u8]) -> Result<Box<dyn VectorIndex + Send + Sync>> where Self: Sized;
    
    /// 获取向量
    fn get(&self, _id: &str) -> Result<Option<Vector>> {
        Err(crate::Error::not_implemented("该索引类型不支持get操作"))
    }
    
    /// 清空索引
    fn clear(&mut self) -> Result<()> {
        Err(crate::Error::not_implemented("该索引类型不支持clear操作"))
    }
    
    /// 从索引中删除向量（按ID）
    fn remove(&mut self, _vector_id: u64) -> Result<()> {
        Err(crate::Error::not_implemented("该索引类型不支持remove操作"))
    }
    
    /// 获取统计信息
    fn get_statistics(&self) -> Result<IndexStatistics> {
        Ok(IndexStatistics {
            total_vectors: 0,
            memory_usage: 0,
            index_type: "unknown".to_string(),
            build_time: std::time::Duration::from_secs(0),
            last_updated: std::time::SystemTime::now(),
        })
    }
    
    /// 获取所有向量（用于线性搜索和调试）
    fn get_all_vectors(&self) -> Result<Vec<VectorData>> {
        Err(crate::Error::not_implemented("该索引类型不支持get_all_vectors操作"))
    }
    
    /// 检查索引健康状态
    fn health_check(&self) -> Result<IndexHealth> {
        Ok(IndexHealth {
            is_healthy: true,
            total_vectors: 0,
            corrupted_vectors: 0,
            last_check: std::time::SystemTime::now(),
            errors: Vec::new(),
        })
    }
}

/// 从字节数组创建索引并装箱的工厂函数
pub fn deserialize_index_box(data: &[u8]) -> Result<Box<dyn VectorIndex + Send + Sync>> {
    // 序列化数据的首字节用于表示索引类型
    if data.is_empty() {
        return Err(crate::Error::invalid_input("无效的序列化数据：数据为空".to_string()));
    }
    
    let index_type = data[0];
    let index_data = &data[1..];
    
    match index_type {
        0 => {
            crate::vector::index::flat::FlatIndex::deserialize_box(index_data)
        },
        1 => {
            use std::io::Cursor;
            let mut cursor = Cursor::new(index_data);
            let index = crate::vector::index::hnsw::HNSWIndex::deserialize(&mut cursor)
                .map_err(|e| crate::Error::serialization(format!("Failed to deserialize HNSWIndex: {}", e)))?;
            // HNSWIndex 通过 VectorIndexEnum 包装后实现 VectorIndex
            Ok(Box::new(crate::vector::index::VectorIndexEnum::HNSW(index)) as Box<dyn VectorIndex + Send + Sync>)
        },
        2 => {
            crate::vector::index::ivf::IVFIndex::deserialize_box(index_data)
        },
        3 => {
            crate::vector::index::pq::PQIndex::deserialize_box(index_data)
        },
        4 => {
            crate::vector::index::lsh::LSHIndex::deserialize_box(index_data)
        },
        5 => {
            crate::vector::index::vptree::VPTreeIndex::deserialize_box(index_data)
        },
        6 => {
            crate::vector::index::ivfpq::IVFPQIndex::deserialize_box(index_data)
        },
        7 => {
            crate::vector::index::ivfhnsw::IVFHNSWIndex::deserialize_box(index_data)
        },
        8 => {
            crate::vector::index::annoy::ANNOYIndex::deserialize_box(index_data)
        },
        9 => {
            crate::vector::index::kmeans::KMeansIndex::deserialize_box(index_data)
        },
        10 => {
            crate::vector::index::ngt::NGTIndex::deserialize_box(index_data)
        },
        11 => {
            crate::vector::index::hierarchical_clustering::HierarchicalClusteringIndex::deserialize_box(index_data)
        },
        12 => {
            crate::vector::index::graph_index::GraphIndex::deserialize_box(index_data)
        },
        _ => Err(crate::Error::invalid_input(format!("未知的索引类型: {}", index_type))),
    }
}

/// 向量数据结构，用于线性搜索
#[derive(Debug, Clone)]
pub struct VectorData {
    pub id: u64,
    pub vector: Vec<f32>,
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

/// 用于堆排序的包装结构
#[derive(Debug, Clone)]
struct OrderedSearchResult(SearchResult);

impl PartialEq for OrderedSearchResult {
    fn eq(&self, other: &Self) -> bool {
        self.0.distance == other.0.distance
    }
}

impl Eq for OrderedSearchResult {}

impl PartialOrd for OrderedSearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.distance.partial_cmp(&other.0.distance)
    }
}

impl Ord for OrderedSearchResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// 索引统计信息
#[derive(Debug, Clone)]
pub struct IndexStatistics {
    pub total_vectors: usize,
    pub memory_usage: usize,
    pub index_type: String,
    pub build_time: std::time::Duration,
    pub last_updated: std::time::SystemTime,
}

/// 索引健康状态
#[derive(Debug, Clone)]
pub struct IndexHealth {
    pub is_healthy: bool,
    pub total_vectors: usize,
    pub corrupted_vectors: usize,
    pub last_check: std::time::SystemTime,
    pub errors: Vec<String>,
}

/// 计算欧几里得距离
pub fn calculate_euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(crate::Error::InvalidInput(format!(
            "向量维度不匹配: {} vs {}", 
            a.len(), 
            b.len()
        )));
    }
    
    if a.is_empty() {
        return Ok(0.0);
    }
    
    let sum_squared: f32 = a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();
    
    Ok(sum_squared.sqrt())
}

/// 计算余弦距离
pub fn calculate_cosine_distance(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(crate::Error::InvalidInput(format!(
            "向量维度不匹配: {} vs {}", 
            a.len(), 
            b.len()
        )));
    }
    
    if a.is_empty() {
        return Ok(0.0);
    }
    
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(1.0); // 如果任一向量为零向量，余弦距离为1
    }
    
    let cosine_similarity = dot_product / (norm_a * norm_b);
    Ok(1.0 - cosine_similarity.max(-1.0).min(1.0))
}

/// 计算曼哈顿距离（L1距离）
pub fn calculate_manhattan_distance(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(crate::Error::InvalidInput(format!(
            "向量维度不匹配: {} vs {}", 
            a.len(), 
            b.len()
        )));
    }
    
    if a.is_empty() {
        return Ok(0.0);
    }
    
    let distance: f32 = a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .sum();
    
    Ok(distance)
}

/// 计算汉明距离（用于二进制向量）
pub fn calculate_hamming_distance(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(crate::Error::InvalidInput(format!(
            "向量维度不匹配: {} vs {}", 
            a.len(), 
            b.len()
        )));
    }
    
    if a.is_empty() {
        return Ok(0.0);
    }
    
    let distance = a.iter()
        .zip(b.iter())
        .filter(|&(x, y)| (x > &0.5) != (y > &0.5)) // 将浮点数转换为二进制
        .count() as f32;
    
    Ok(distance)
}

/// 根据距离类型计算距离
pub fn calculate_distance(a: &[f32], b: &[f32], distance_type: DistanceType) -> Result<f32> {
    match distance_type {
        DistanceType::Euclidean => calculate_euclidean_distance(a, b),
        DistanceType::Cosine => calculate_cosine_distance(a, b),
        DistanceType::Manhattan => calculate_manhattan_distance(a, b),
        DistanceType::Hamming => calculate_hamming_distance(a, b),
    }
}

/// 距离计算类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceType {
    /// 欧几里得距离（L2距离）
    Euclidean,
    /// 余弦距离
    Cosine,
    /// 曼哈顿距离（L1距离）
    Manhattan,
    /// 汉明距离
    Hamming,
}

/// 向量搜索性能统计
#[derive(Debug, Clone)]
pub struct SearchPerformanceStats {
    pub total_searches: usize,
    pub average_search_time_ms: f64,
    pub total_vectors_processed: usize,
    pub cache_hit_rate: f64,
    pub last_reset: std::time::SystemTime,
}

impl Default for SearchPerformanceStats {
    fn default() -> Self {
        Self {
            total_searches: 0,
            average_search_time_ms: 0.0,
            total_vectors_processed: 0,
            cache_hit_rate: 0.0,
            last_reset: std::time::SystemTime::now(),
        }
    }
}

impl SearchPerformanceStats {
    /// 创建新的性能统计
    pub fn new() -> Self {
        Self {
            total_searches: 0,
            average_search_time_ms: 0.0,
            total_vectors_processed: 0,
            cache_hit_rate: 0.0,
            last_reset: std::time::SystemTime::now(),
        }
    }
    
    /// 记录一次搜索
    pub fn record_search(&mut self, duration_ms: f64, vectors_processed: usize) {
        self.total_searches += 1;
        self.total_vectors_processed += vectors_processed;
        
        // 计算移动平均
        let alpha = 0.1; // 平滑因子
        self.average_search_time_ms = 
            alpha * duration_ms + (1.0 - alpha) * self.average_search_time_ms;
    }
    
    /// 重置统计信息
    pub fn reset(&mut self) {
        *self = Self::new();
    }
} 