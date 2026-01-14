// HNSW缓存实现
// 实现查询缓存功能，提高搜索性能

use std::collections::{HashMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use std::sync::RwLock;
use std::time::Instant;
use crate::vector::index::types::SearchResult;

/// 查询缓存，用于存储最近的查询结果
#[derive(Debug)]
pub struct QueryCache {
    cache: RwLock<HashMap<Vec<f32>, (Vec<SearchResult>, Instant)>>,
    capacity: usize,
    hits: RwLock<usize>,
    misses: RwLock<usize>,
}

impl QueryCache {
    /// 创建一个新的查询缓存
    pub fn new(capacity: usize) -> Self {
        QueryCache {
            cache: RwLock::new(HashMap::with_capacity(capacity)),
            capacity,
            hits: RwLock::new(0),
            misses: RwLock::new(0),
        }
    }

    /// 获取缓存中的查询结果
    pub fn get(&self, query: &[f32]) -> Option<Vec<SearchResult>> {
        let query_vec = query.to_vec();
        if let Ok(cache) = self.cache.read() {
            if let Some((results, timestamp)) = cache.get(&query_vec) {
                if let Ok(mut hits) = self.hits.write() {
                    *hits += 1;
                }
                return Some(results.clone());
            }
        }
        
        if let Ok(mut misses) = self.misses.write() {
            *misses += 1;
        }
        None
    }

    /// 将查询结果存入缓存
    pub fn put(&self, query: &[f32], results: Vec<SearchResult>) {
        let query_vec = query.to_vec();
        if let Ok(mut cache) = self.cache.write() {
            // 如果缓存已满，删除最旧的条目
            if cache.len() >= self.capacity && !cache.contains_key(&query_vec) {
                let oldest_key = cache
                    .iter()
                    .min_by_key(|(_, (_, time))| time)
                    .map(|(k, _)| k.clone());
                
                if let Some(key) = oldest_key {
                    cache.remove(&key);
                }
            }
            
            cache.insert(query_vec, (results, Instant::now()));
        }
    }

    /// 清空缓存
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
        if let Ok(mut hits) = self.hits.write() {
            *hits = 0;
        }
        if let Ok(mut misses) = self.misses.write() {
            *misses = 0;
        }
    }

    /// 获取缓存统计信息
    pub fn stats(&self) -> (usize, usize, f32) {
        // 使用默认值处理锁获取失败的情况，避免panic
        let hits = self.hits.read()
            .map(|guard| *guard)
            .unwrap_or(0);
        let misses = self.misses.read()
            .map(|guard| *guard)
            .unwrap_or(0);
        let total = hits + misses;
        let hit_rate = if total > 0 { hits as f32 / total as f32 } else { 0.0 };
        (hits, misses, hit_rate)
    }

    /// 获取缓存大小
    pub fn size(&self) -> usize {
        if let Ok(cache) = self.cache.read() {
            cache.len()
        } else {
            0
        }
    }

    /// 使用自定义键获取缓存中的查询结果
    pub fn get_query_result_with_key(&self, query: &[f32], key: &str, ttl_ms: u64) -> Option<Vec<SearchResult>> {
        let query_vec = query.to_vec();
        let cache_key = format!("{}:{}", self.hash_vector(&query_vec), key);
        
        if let Ok(cache) = self.cache.read() {
            if let Some((results, timestamp)) = cache.get(&query_vec) {
                // 检查结果是否过期
                if timestamp.elapsed().as_millis() <= ttl_ms as u128 {
                    if let Ok(mut hits) = self.hits.write() {
                        *hits += 1;
                    }
                    return Some(results.clone());
                }
            }
        }
        
        if let Ok(mut misses) = self.misses.write() {
            *misses += 1;
        }
        None
    }

    /// 使用自定义键存储查询结果
    pub fn put_query_result_with_key(&self, query: &[f32], key: &str, results: Vec<SearchResult>) {
        let query_vec = query.to_vec();
        let cache_key = format!("{}:{}", self.hash_vector(&query_vec), key);
        
        if let Ok(mut cache) = self.cache.write() {
            // 如果缓存已满，删除最旧的条目
            if cache.len() >= self.capacity && !cache.contains_key(&cache_key) {
                let oldest_key = cache
                    .iter()
                    .min_by_key(|(_, (_, time))| time)
                    .map(|(k, _)| k.clone());
                
                if let Some(key) = oldest_key {
                    cache.remove(&key);
                }
            }
            
            cache.insert(query_vec, (results, Instant::now()));
        }
    }
    
    /// 计算向量的哈希值(用于缓存键)
    fn hash_vector(&self, vector: &[f32]) -> String {
        let mut hasher = DefaultHasher::new();
        for val in vector {
            val.to_bits().hash(&mut hasher);
        }
        format!("{:x}", hasher.finish())
    }

    /// 获取缓存容量
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// 获取命中率
    pub fn hit_rate(&self) -> f64 {
        // 使用默认值处理锁获取失败的情况，避免panic
        let hits = self.hits.read()
            .map(|guard| *guard)
            .unwrap_or(0);
        let misses = self.misses.read()
            .map(|guard| *guard)
            .unwrap_or(0);
        let total = hits + misses;
        if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        }
    }

    /// 估算内存使用
    pub fn memory_usage(&self) -> usize {
        let mut total = std::mem::size_of::<Self>();
        
        if let Ok(cache) = self.cache.read() {
            for (k, (v, _)) in cache.iter() {
                // 键的内存：向量的长度 * 每个f32元素的大小
                total += k.len() * std::mem::size_of::<f32>();
                
                // 值的内存：搜索结果数量 * 每个结果的估计大小 + 时间戳大小
                total += v.len() * (std::mem::size_of::<SearchResult>() + 64); // 64是元数据的粗略估计
                total += std::mem::size_of::<Instant>();
            }
        }
        
        total
    }
}
