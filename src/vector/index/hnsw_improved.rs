// HNSW索引实现
// 分层导航小世界图（Hierarchical Navigable Small World）索引实现

use serde::{Serialize, Deserialize};
use crate::{Result, Error, vector::Vector};
use super::interfaces::VectorIndex;
use super::types::{IndexConfig, SearchResult};
use super::distance::Distance;
use super::common::HNSWNode;
use super::search_config::SearchConfig;
use std::sync::{Arc, RwLock};
use std::collections::{HashMap, BinaryHeap, HashSet};
use std::cmp::Reverse;
use std::time::{Instant, Duration};
use crate::vector::distance::EuclideanDistance;
use rand::prelude::*;
use log::{info, debug, warn};
use chrono::{Utc, DateTime};
use std::cmp::Ordering;

/// 用于浮点数比较的辅助结构体，用于构建最大堆
#[derive(Debug, PartialEq)]
struct FloatComparison(f32, usize);

impl Eq for FloatComparison {}

impl PartialOrd for FloatComparison {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Ord for FloatComparison {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// 用于浮点数比较的辅助结构体，用于构建最小堆
#[derive(Debug, PartialEq)]
struct ReverseFloatComparison(f32, usize);

impl Eq for ReverseFloatComparison {}

impl PartialOrd for ReverseFloatComparison {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.0.partial_cmp(&self.0)
    }
}

impl Ord for ReverseFloatComparison {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// HNSW索引参数
#[derive(Debug, Clone)]
pub struct HNSWParams {
    // ... existing code ...
}

/// 查询缓存，用于存储最近的查询结果
#[derive(Debug)]
pub struct QueryCache {
    cache: RwLock<HashMap<Vec<f32>, (Vec<SearchResult>, Instant)>>,
    capacity: usize,
    hits: RwLock<usize>,
    misses: RwLock<usize>,
}

// 其他代码保持不变...

/// 使用搜索配置进行搜索
pub fn search_with_config(&self, query: &[f32], config: SearchConfig<HNSWNode>) -> Result<Vec<SearchResult>> {
    // 先执行基本搜索
    let mut results = self.search_with_strategy(query, config.limit, &config.strategy)?;
    
    // 如果有过滤器，应用过滤
    if let Some(filter) = &config.filter {
        let nodes = self.nodes.read().map_err(|e| Error::lock(e.to_string()))?;
        
        // 过滤结果
        results = results
            .into_iter()
            .filter(|result| {
                if let Some(node) = nodes.get(&result.id) {
                    filter(node)
                } else {
                    false
                }
            })
            .collect();
    }
    
    // 如果需要，添加向量数据
    if config.include_vectors {
        let vectors = self.vectors.read().map_err(|e| Error::lock(e.to_string()))?;
        
        for result in &mut results {
            if let Some(vector) = vectors.get(&result.id) {
                result.vector = Some(vector.clone());
            }
        }
    }
    
    // 如果需要，添加元数据
    if config.include_metadata {
        let nodes = self.nodes.read().map_err(|e| Error::lock(e.to_string()))?;
        
        for result in &mut results {
            if let Some(node) = nodes.get(&result.id) {
                result.metadata = node.metadata.clone();
            }
        }
    }
    
    Ok(results)
} 