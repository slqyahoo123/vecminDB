// 向量索引模块
// 提供多种向量索引的实现和工厂函数

// 导入基础依赖
use crate::{Error, Result, vector::Vector};

// 导出子模块
pub mod types;
pub mod interfaces;
pub mod common;
pub mod search_config;
pub mod distance;
pub mod utils; // 导出工具模块
pub mod flat;
pub mod hnsw;
pub mod ivf;
pub mod pq;
pub mod ivfpq;
pub mod ivfhnsw;
pub mod lsh;
pub mod kmeans;
pub mod vptree;
pub mod annoy;
pub mod factory;
pub mod ngt;
pub mod hierarchical_clustering;
pub mod graph_index;

// 从子模块重新导出公共接口
pub use types::{SearchResult, IndexType, IndexConfig};
pub use interfaces::VectorIndex;
pub use flat::FlatIndex;
pub use hnsw::HNSWIndex;
pub use ivf::IVFIndex;
pub use pq::PQIndex;
pub use lsh::LSHIndex;
pub use vptree::VPTreeIndex;
pub use factory::VectorIndexFactory;
pub use ivfpq::IVFPQIndex;
pub use ivfhnsw::IVFHNSWIndex;
pub use annoy::ANNOYIndex;
pub use kmeans::KMeansIndex;
pub use ngt::NGTIndex;
pub use hierarchical_clustering::HierarchicalClusteringIndex;
pub use graph_index::GraphIndex;
pub use common::HNSWNode;

/// 索引参数
pub type IndexParams = IndexConfig;

/// 索引状态
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexStatus {
    /// 未初始化
    Uninitialized,
    /// 正在构建
    Building,
    /// 已就绪
    Ready,
    /// 错误状态
    Error(String),
    /// 正在更新
    Updating,
    /// 已损坏
    Corrupted,
}

/// 向量索引枚举
#[derive(Clone, Debug)]
pub enum VectorIndexEnum {
    Flat(FlatIndex),
    HNSW(HNSWIndex),
    IVF(IVFIndex),
    PQ(PQIndex),
    LSH(LSHIndex),
    VPTree(VPTreeIndex),
    IVFPQ(IVFPQIndex),
    IVFHNSW(IVFHNSWIndex),
    ANNOY(ANNOYIndex),
    KMeans(KMeansIndex),
    NGT(NGTIndex),
    HierarchicalClustering(HierarchicalClusteringIndex),
    GraphIndex(GraphIndex),
}

impl VectorIndex for VectorIndexEnum {
    fn add(&mut self, vector: Vector) -> Result<()> {
        match self {
            Self::Flat(index) => index.add(vector),
            // HNSW 当前通过专用 API 使用，尚未接入通用 VectorIndex 接口
            Self::HNSW(_) => Err(Error::vector("HNSW index is not integrated with VectorIndex::add")),
            Self::IVF(index) => index.add(vector),
            Self::PQ(index) => index.add(vector),
            Self::LSH(index) => index.add(vector),
            Self::VPTree(index) => index.add(vector),
            Self::IVFPQ(index) => index.add(vector),
            Self::IVFHNSW(index) => index.add(vector),
            Self::ANNOY(index) => index.add(vector),
            Self::KMeans(index) => index.add(vector),
            Self::NGT(index) => index.add(vector),
            Self::HierarchicalClustering(index) => index.add(vector),
            Self::GraphIndex(index) => index.add(vector),
        }
    }
    
    fn search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        match self {
            Self::Flat(index) => index.search(query, limit),
            // HNSW 搜索需使用专用配置与 API
            Self::HNSW(_) => Err(Error::vector("HNSW index is not integrated with VectorIndex::search")),
            Self::IVF(index) => index.search(query, limit),
            Self::PQ(index) => index.search(query, limit),
            Self::LSH(index) => index.search(query, limit),
            Self::VPTree(index) => index.search(query, limit),
            Self::IVFPQ(index) => index.search(query, limit),
            Self::IVFHNSW(index) => index.search(query, limit),
            Self::ANNOY(index) => index.search(query, limit),
            Self::KMeans(index) => index.search(query, limit),
            Self::NGT(index) => index.search(query, limit),
            Self::HierarchicalClustering(index) => index.search(query, limit),
            Self::GraphIndex(index) => index.search(query, limit),
        }
    }
    
    fn delete(&mut self, id: &str) -> Result<bool> {
        match self {
            Self::Flat(index) => index.delete(id),
            Self::HNSW(_) => Err(Error::vector("HNSW index is not integrated with VectorIndex::delete")),
            Self::IVF(index) => index.delete(id),
            Self::PQ(index) => index.delete(id),
            Self::LSH(index) => index.delete(id),
            Self::VPTree(index) => index.delete(id),
            Self::IVFPQ(index) => index.delete(id),
            Self::IVFHNSW(index) => index.delete(id),
            Self::ANNOY(index) => index.delete(id),
            Self::KMeans(index) => index.delete(id),
            Self::NGT(index) => index.delete(id),
            Self::HierarchicalClustering(index) => index.delete(id),
            Self::GraphIndex(index) => index.delete(id),
        }
    }
    
    fn contains(&self, id: &str) -> bool {
        match self {
            Self::Flat(index) => index.contains(id),
            Self::HNSW(_) => false,
            Self::IVF(index) => index.contains(id),
            Self::PQ(index) => index.contains(id),
            Self::LSH(index) => index.contains(id),
            Self::VPTree(index) => index.contains(id),
            Self::IVFPQ(index) => index.contains(id),
            Self::IVFHNSW(index) => index.contains(id),
            Self::ANNOY(index) => index.contains(id),
            Self::KMeans(index) => index.contains(id),
            Self::NGT(index) => index.contains(id),
            Self::HierarchicalClustering(index) => index.contains(id),
            Self::GraphIndex(index) => index.contains(id),
        }
    }
    
    fn size(&self) -> usize {
        match self {
            Self::Flat(index) => index.size(),
            Self::HNSW(index) => index.size(),
            Self::IVF(index) => index.size(),
            Self::PQ(index) => index.size(),
            Self::LSH(index) => index.size(),
            Self::VPTree(index) => index.size(),
            Self::IVFPQ(index) => index.size(),
            Self::IVFHNSW(index) => index.size(),
            Self::ANNOY(index) => index.size(),
            Self::KMeans(index) => index.size(),
            Self::NGT(index) => index.size(),
            Self::HierarchicalClustering(index) => index.size(),
            Self::GraphIndex(index) => index.size(),
        }
    }
    
    fn dimension(&self) -> usize {
        match self {
            Self::Flat(index) => index.dimension(),
            Self::HNSW(index) => index.dimension(),
            Self::IVF(index) => index.dimension(),
            Self::PQ(index) => index.dimension(),
            Self::LSH(index) => index.dimension(),
            Self::VPTree(index) => index.dimension(),
            Self::IVFPQ(index) => index.dimension(),
            Self::IVFHNSW(index) => index.dimension(),
            Self::ANNOY(index) => index.dimension(),
            Self::KMeans(index) => index.dimension(),
            Self::NGT(index) => index.dimension(),
            Self::HierarchicalClustering(index) => index.dimension(),
            Self::GraphIndex(index) => index.dimension(),
        }
    }
    
    fn get_config(&self) -> IndexConfig {
        match self {
            Self::Flat(index) => index.get_config(),
            // 为保持接口稳定，这里返回一个基于默认值的占位配置
            Self::HNSW(index) => {
                let mut cfg = IndexConfig::default();
                cfg.index_type = IndexType::HNSW;
                cfg.dimension = index.dimension();
                cfg
            }
            Self::IVF(index) => index.get_config(),
            Self::PQ(index) => index.get_config(),
            Self::LSH(index) => index.get_config(),
            Self::VPTree(index) => index.get_config(),
            Self::IVFPQ(index) => index.get_config(),
            Self::IVFHNSW(index) => index.get_config(),
            Self::ANNOY(index) => index.get_config(),
            Self::KMeans(index) => index.get_config(),
            Self::NGT(index) => index.get_config(),
            Self::HierarchicalClustering(index) => index.get_config(),
            Self::GraphIndex(index) => index.get_config(),
        }
    }
    
    fn serialize(&self) -> Result<Vec<u8>> {
        match self {
            Self::Flat(index) => index.serialize(),
            Self::HNSW(_) => Err(Error::vector("HNSW index serialization via VectorIndex is not implemented")),
            Self::IVF(index) => index.serialize(),
            Self::PQ(index) => index.serialize(),
            Self::LSH(index) => index.serialize(),
            Self::VPTree(index) => index.serialize(),
            Self::IVFPQ(index) => index.serialize(),
            Self::IVFHNSW(index) => index.serialize(),
            Self::ANNOY(index) => index.serialize(),
            Self::KMeans(index) => index.serialize(),
            Self::NGT(index) => index.serialize(),
            Self::HierarchicalClustering(index) => index.serialize(),
            Self::GraphIndex(index) => index.serialize(),
        }
    }
    
    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        match self {
            Self::Flat(index) => index.deserialize(data),
            Self::HNSW(_) => Err(Error::vector("HNSW index deserialization via VectorIndex is not implemented")),
            Self::IVF(index) => index.deserialize(data),
            Self::PQ(index) => index.deserialize(data),
            Self::LSH(index) => index.deserialize(data),
            Self::VPTree(index) => index.deserialize(data),
            Self::IVFPQ(index) => index.deserialize(data),
            Self::IVFHNSW(index) => index.deserialize(data),
            Self::ANNOY(index) => index.deserialize(data),
            Self::KMeans(index) => index.deserialize(data),
            Self::NGT(index) => index.deserialize(data),
            Self::HierarchicalClustering(index) => index.deserialize(data),
            Self::GraphIndex(index) => index.deserialize(data),
        }
    }
    
    fn batch_insert(&mut self, vectors: &[Vector]) -> Result<()> {
        match self {
            Self::Flat(index) => index.batch_insert(vectors),
            Self::HNSW(_) => Err(Error::vector("HNSW index is not integrated with VectorIndex::batch_insert")),
            Self::IVF(index) => index.batch_insert(vectors),
            Self::PQ(index) => index.batch_insert(vectors),
            Self::LSH(index) => index.batch_insert(vectors),
            Self::VPTree(index) => index.batch_insert(vectors),
            Self::IVFPQ(index) => index.batch_insert(vectors),
            Self::IVFHNSW(index) => index.batch_insert(vectors),
            Self::ANNOY(index) => index.batch_insert(vectors),
            Self::KMeans(index) => index.batch_insert(vectors),
            Self::NGT(index) => index.batch_insert(vectors),
            Self::HierarchicalClustering(index) => index.batch_insert(vectors),
            Self::GraphIndex(index) => index.batch_insert(vectors),
        }
    }
    
    fn batch_delete(&mut self, ids: &[String]) -> Result<usize> {
        match self {
            Self::Flat(index) => index.batch_delete(ids),
            Self::HNSW(_) => Err(Error::vector("HNSW index is not integrated with VectorIndex::batch_delete")),
            Self::IVF(index) => index.batch_delete(ids),
            Self::PQ(index) => index.batch_delete(ids),
            Self::LSH(index) => index.batch_delete(ids),
            Self::VPTree(index) => index.batch_delete(ids),
            Self::IVFPQ(index) => index.batch_delete(ids),
            Self::IVFHNSW(index) => index.batch_delete(ids),
            Self::ANNOY(index) => index.batch_delete(ids),
            Self::KMeans(index) => index.batch_delete(ids),
            Self::NGT(index) => index.batch_delete(ids),
            Self::HierarchicalClustering(index) => index.batch_delete(ids),
            Self::GraphIndex(index) => index.batch_delete(ids),
        }
    }
    
    fn batch_update(&mut self, vectors: &[Vector]) -> Result<usize> {
        match self {
            Self::Flat(index) => index.batch_update(vectors),
            Self::HNSW(_) => Err(Error::vector("HNSW index is not integrated with VectorIndex::batch_update")),
            Self::IVF(index) => index.batch_update(vectors),
            Self::PQ(index) => index.batch_update(vectors),
            Self::LSH(index) => index.batch_update(vectors),
            Self::VPTree(index) => index.batch_update(vectors),
            Self::IVFPQ(index) => index.batch_update(vectors),
            Self::IVFHNSW(index) => index.batch_update(vectors),
            Self::ANNOY(index) => index.batch_update(vectors),
            Self::KMeans(index) => index.batch_update(vectors),
            Self::NGT(index) => index.batch_update(vectors),
            Self::HierarchicalClustering(index) => index.batch_update(vectors),
            Self::GraphIndex(index) => index.batch_update(vectors),
        }
    }
    
    fn get_memory_usage(&self) -> Result<usize> {
        match self {
            Self::Flat(index) => index.get_memory_usage(),
            Self::HNSW(_) => Err(Error::vector("HNSW index memory usage via VectorIndex is not implemented")),
            Self::IVF(index) => index.get_memory_usage(),
            Self::PQ(index) => index.get_memory_usage(),
            Self::LSH(index) => index.get_memory_usage(),
            Self::VPTree(index) => index.get_memory_usage(),
            Self::IVFPQ(index) => index.get_memory_usage(),
            Self::IVFHNSW(index) => index.get_memory_usage(),
            Self::ANNOY(index) => index.get_memory_usage(),
            Self::KMeans(index) => index.get_memory_usage(),
            Self::NGT(index) => index.get_memory_usage(),
            Self::HierarchicalClustering(index) => index.get_memory_usage(),
            Self::GraphIndex(index) => index.get_memory_usage(),
        }
    }

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
        match self {
            Self::Flat(index) => index.range_search(query, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::HNSW(_) => Err(Error::vector("HNSW index range_search via VectorIndex is not implemented")),
            Self::IVF(index) => index.range_search(query, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::PQ(index) => index.range_search(query, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::LSH(index) => index.range_search(query, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::VPTree(index) => index.range_search(query, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::IVFPQ(index) => index.range_search(query, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::IVFHNSW(index) => index.range_search(query, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::ANNOY(index) => index.range_search(query, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::KMeans(index) => index.range_search(query, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::NGT(index) => index.range_search(query, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::HierarchicalClustering(index) => index.range_search(query, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::GraphIndex(index) => index.range_search(query, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
        }
    }
    
    /// 批量执行范围搜索，处理多个查询向量
    fn batch_range_search(&self, 
                         queries: &[Vec<f32>], 
                         radius: f32, 
                         with_vectors: bool, 
                         with_metadata: bool,
                         max_elements: Option<usize>, 
                         dynamic_ef: bool) -> Result<Vec<Vec<SearchResult>>> {
        match self {
            Self::Flat(index) => index.batch_range_search(queries, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::HNSW(_) => Err(Error::vector("HNSW index batch_range_search via VectorIndex is not implemented")),
            Self::IVF(index) => index.batch_range_search(queries, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::PQ(index) => index.batch_range_search(queries, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::LSH(index) => index.batch_range_search(queries, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::VPTree(index) => index.batch_range_search(queries, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::IVFPQ(index) => index.batch_range_search(queries, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::IVFHNSW(index) => index.batch_range_search(queries, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::ANNOY(index) => index.batch_range_search(queries, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::KMeans(index) => index.batch_range_search(queries, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::NGT(index) => index.batch_range_search(queries, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::HierarchicalClustering(index) => index.batch_range_search(queries, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
            Self::GraphIndex(index) => index.batch_range_search(queries, radius, with_vectors, with_metadata, max_elements, dynamic_ef),
        }
    }
    
    /// 在索引中执行线性搜索，适用于小型数据集或作为基准测试
    fn linear_search(&self, query: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
        match self {
            Self::Flat(index) => index.linear_search(query, limit),
            Self::HNSW(_) => Err(Error::vector("HNSW index linear_search via VectorIndex is not implemented")),
            Self::IVF(index) => index.linear_search(query, limit),
            Self::PQ(index) => index.linear_search(query, limit),
            Self::LSH(index) => index.linear_search(query, limit),
            Self::VPTree(index) => index.linear_search(query, limit),
            Self::IVFPQ(index) => index.linear_search(query, limit),
            Self::IVFHNSW(index) => index.linear_search(query, limit),
            Self::ANNOY(index) => index.linear_search(query, limit),
            Self::KMeans(index) => index.linear_search(query, limit),
            Self::NGT(index) => index.linear_search(query, limit),
            Self::HierarchicalClustering(index) => index.linear_search(query, limit),
            Self::GraphIndex(index) => index.linear_search(query, limit),
        }
    }
    
    /// 获取向量
    fn get(&self, id: &str) -> Result<Option<Vector>> {
        match self {
            Self::Flat(index) => index.get(id),
            Self::HNSW(_) => Err(Error::vector("HNSW index get via VectorIndex is not implemented")),
            Self::IVF(index) => index.get(id),
            Self::PQ(index) => index.get(id),
            Self::LSH(index) => index.get(id),
            Self::VPTree(index) => index.get(id),
            Self::IVFPQ(index) => index.get(id),
            Self::IVFHNSW(index) => index.get(id),
            Self::ANNOY(index) => index.get(id),
            Self::KMeans(index) => index.get(id),
            Self::NGT(index) => index.get(id),
            Self::HierarchicalClustering(index) => index.get(id),
            Self::GraphIndex(index) => index.get(id),
        }
    }
    
    /// 清空索引
    fn clear(&mut self) -> Result<()> {
        match self {
            Self::Flat(index) => index.clear(),
            Self::HNSW(_) => Err(Error::vector("HNSW index clear via VectorIndex is not implemented")),
            Self::IVF(index) => index.clear(),
            Self::PQ(index) => index.clear(),
            Self::LSH(index) => index.clear(),
            Self::VPTree(index) => index.clear(),
            Self::IVFPQ(index) => index.clear(),
            Self::IVFHNSW(index) => index.clear(),
            Self::ANNOY(index) => index.clear(),
            Self::KMeans(index) => index.clear(),
            Self::NGT(index) => index.clear(),
            Self::HierarchicalClustering(index) => index.clear(),
            Self::GraphIndex(index) => index.clear(),
        }
    }
    
    /// 从索引中删除向量（按ID）
    fn remove(&mut self, vector_id: u64) -> Result<()> {
        match self {
            Self::Flat(index) => index.remove(vector_id),
            Self::HNSW(_) => Err(Error::vector("HNSW index remove via VectorIndex is not implemented")),
            Self::IVF(index) => index.remove(vector_id),
            Self::PQ(index) => index.remove(vector_id),
            Self::LSH(index) => index.remove(vector_id),
            Self::VPTree(index) => index.remove(vector_id),
            Self::IVFPQ(index) => index.remove(vector_id),
            Self::IVFHNSW(index) => index.remove(vector_id),
            Self::ANNOY(index) => index.remove(vector_id),
            Self::KMeans(index) => index.remove(vector_id),
            Self::NGT(index) => index.remove(vector_id),
            Self::HierarchicalClustering(index) => index.remove(vector_id),
            Self::GraphIndex(index) => index.remove(vector_id),
        }
    }
    
    /// 创建索引的深拷贝并装箱
    fn clone_box(&self) -> Box<dyn VectorIndex + Send + Sync> {
        match self {
            Self::Flat(index) => index.clone_box(),
            // HNSW 当前未完全接入通用 VectorIndex 体系，这里不支持通过 clone_box 复制
            Self::HNSW(_) => panic!("HNSW index clone_box via VectorIndexEnum is not supported"),
            Self::IVF(index) => index.clone_box(),
            Self::PQ(index) => index.clone_box(),
            Self::LSH(index) => index.clone_box(),
            Self::VPTree(index) => index.clone_box(),
            Self::IVFPQ(index) => index.clone_box(),
            Self::IVFHNSW(index) => index.clone_box(),
            Self::ANNOY(index) => index.clone_box(),
            Self::KMeans(index) => index.clone_box(),
            Self::NGT(index) => index.clone_box(),
            Self::HierarchicalClustering(index) => index.clone_box(),
            Self::GraphIndex(index) => index.clone_box(),
        }
    }
    
    fn deserialize_box(data: &[u8]) -> Result<Box<dyn VectorIndex + Send + Sync>> {
        VectorIndexEnum::deserialize_box(data)
    }
}

impl VectorIndexEnum {
    /// 从字节数组反序列化索引
    pub fn deserialize_box(data: &[u8]) -> Result<Box<dyn VectorIndex + Send + Sync>> {
        // 序列化数据的首字节用于表示索引类型
        if data.is_empty() {
            return Err(Error::invalid_input("无效的序列化数据：数据为空".to_string()));
        }
        
        let index_type = data[0];
        let index_data = &data[1..];
        
        match index_type {
            0 => FlatIndex::deserialize_box(index_data),
            // HNSW 使用专用序列化/反序列化 API，这里不通过通用工厂恢复
            1 => Err(Error::vector("HNSW index deserialization via VectorIndexEnum is not implemented")),
            2 => IVFIndex::deserialize_box(index_data),
            3 => PQIndex::deserialize_box(index_data),
            4 => LSHIndex::deserialize_box(index_data),
            5 => VPTreeIndex::deserialize_box(index_data),
            6 => IVFPQIndex::deserialize_box(index_data),
            7 => IVFHNSWIndex::deserialize_box(index_data),
            8 => ANNOYIndex::deserialize_box(index_data),
            9 => KMeansIndex::deserialize_box(index_data),
            10 => NGTIndex::deserialize_box(index_data),
            11 => HierarchicalClusteringIndex::deserialize_box(index_data),
            12 => GraphIndex::deserialize_box(index_data),
            _ => Err(Error::invalid_input(format!("未知的索引类型: {}", index_type))),
        }
    }
}