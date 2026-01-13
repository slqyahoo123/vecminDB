//! HNSW向量索引模块
//!
//! 本模块实现了层次可导航小世界图(Hierarchical Navigable Small World)索引算法，
//! 用于高效的向量相似性搜索。
//!
//! HNSW是一种图索引结构，通过构建多层次的近似最近邻图实现对高维向量的快速搜索。
//! 该算法在保持高查询精度的同时，显著减少搜索空间，实现亚线性的搜索复杂度。

pub mod build;
pub mod distance;
pub mod node;
pub mod search;
pub mod stats;
pub mod types;
pub mod serialization;

// use std::collections::HashMap; // not used directly in this module

// 重新导出主要类型和方法
pub use build::HNSWBuilder;
pub use build::HNSWIndex;
pub use distance::calculate_distance;
pub use node::{HNSWNode, SharedNode};
pub use stats::{SearchConfig, SearchResult};
pub use types::{Distance, DistanceType, NodeIndex, Vector, VectorId};

/// 创建一个新的HNSW索引
pub fn create_index(dimension: usize) -> HNSWBuilder {
    HNSWBuilder::new(dimension)
}

/// 构建统一的HNSW索引配置
pub struct HNSWConfig {
    /// 向量维度
    pub dimension: usize,
    /// 每层最大连接数
    pub m: usize,
    /// 构建时查找范围
    pub ef_construction: usize,
    /// 搜索时查找范围
    pub ef_search: usize,
    /// 距离函数类型
    pub distance_type: DistanceType,
    /// 最大层数限制
    pub max_level_limit: usize,
    /// 索引ID
    pub index_id: String,
}

impl Default for HNSWConfig {
    fn default() -> Self {
        Self {
            dimension: 128,
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            distance_type: DistanceType::Euclidean,
            max_level_limit: 16,
            index_id: "hnsw_default".to_string(),
        }
    }
}

impl HNSWConfig {
    /// 创建新的配置
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            ..Default::default()
        }
    }
    
    /// 从配置创建索引
    pub fn create_index(&self) -> HNSWIndex {
        HNSWBuilder::new(self.dimension)
            .with_m(self.m)
            .with_ef_construction(self.ef_construction)
            .with_distance_type(self.distance_type)
            .with_max_level_limit(self.max_level_limit)
            .with_index_id(self.index_id.clone())
            .build()
    }
    
    /// 设置每层最大连接数
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }
    
    /// 设置构建时查找范围
    pub fn with_ef_construction(mut self, ef_construction: usize) -> Self {
        self.ef_construction = ef_construction;
        self
    }
    
    /// 设置搜索时查找范围
    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = ef_search;
        self
    }
    
    /// 设置距离函数类型
    pub fn with_distance_type(mut self, distance_type: DistanceType) -> Self {
        self.distance_type = distance_type;
        self
    }
    
    /// 设置最大层数限制
    pub fn with_max_level_limit(mut self, max_level_limit: usize) -> Self {
        self.max_level_limit = max_level_limit;
        self
    }
    
    /// 设置索引ID
    pub fn with_index_id(mut self, index_id: String) -> Self {
        self.index_id = index_id;
        self
    }
    
    /// 从配置信息中创建搜索配置
    pub fn create_search_config(&self) -> SearchConfig {
        SearchConfig::new()
            .with_limit(50)
            .with_ef(self.ef_search)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_index() {
        let index = create_index(128);
        assert_eq!(index.dimension, 128);
    }
    
    #[test]
    fn test_hnsw_config() {
        let config = HNSWConfig::new(256)
            .with_m(24)
            .with_ef_construction(300)
            .with_ef_search(100)
            .with_distance_type(DistanceType::Cosine)
            .with_max_level_limit(20)
            .with_index_id("test_index".to_string());
            
        assert_eq!(config.dimension, 256);
        assert_eq!(config.m, 24);
        assert_eq!(config.ef_construction, 300);
        assert_eq!(config.ef_search, 100);
        assert_eq!(config.distance_type, DistanceType::Cosine);
        assert_eq!(config.max_level_limit, 20);
        assert_eq!(config.index_id, "test_index");
        
        let index = config.create_index();
        assert_eq!(index.dimension(), 256);
        assert_eq!(index.index_id(), "test_index");
    }
}
