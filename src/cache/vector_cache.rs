//! 向量缓存模块
//! 
//! 提供专门针对向量数据优化的缓存功能

use crate::Result;
use super::{CacheManager, CacheResult};
use std::sync::Arc;
use serde::{Serialize, Deserialize};

/// 向量缓存条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorCacheEntry {
    /// 向量ID
    pub id: String,
    /// 向量数据
    pub vector: Vec<f32>,
    /// 元数据
    pub metadata: Option<serde_json::Value>,
    /// 访问次数
    pub access_count: u64,
    /// 最后访问时间
    pub last_accessed: chrono::DateTime<chrono::Utc>,
}

impl VectorCacheEntry {
    /// 创建新的缓存条目
    pub fn new(id: String, vector: Vec<f32>) -> Self {
        Self {
            id,
            vector,
            metadata: None,
            access_count: 0,
            last_accessed: chrono::Utc::now(),
        }
    }

    /// 设置元数据
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// 记录访问
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = chrono::Utc::now();
    }

    /// 获取向量维度
    pub fn dimension(&self) -> usize {
        self.vector.len()
    }
}

/// 向量缓存配置
#[derive(Debug, Clone)]
pub struct VectorCacheConfig {
    /// 最大缓存向量数
    pub max_vectors: usize,
    /// 最大缓存大小（字节）
    pub max_size_bytes: usize,
    /// TTL（秒）
    pub ttl_seconds: Option<u64>,
    /// 是否启用访问统计
    pub enable_stats: bool,
}

impl Default for VectorCacheConfig {
    fn default() -> Self {
        Self {
            max_vectors: 10000,
            max_size_bytes: 1024 * 1024 * 1024, // 1GB
            ttl_seconds: Some(3600), // 1小时
            enable_stats: true,
        }
    }
}

/// 向量缓存
pub struct VectorCache {
    /// 底层缓存管理器
    cache_manager: Arc<CacheManager>,
    /// 配置
    config: VectorCacheConfig,
    /// 键前缀
    key_prefix: String,
}

impl VectorCache {
    /// 创建新的向量缓存
    pub fn new(cache_manager: Arc<CacheManager>, config: VectorCacheConfig) -> Self {
        Self {
            cache_manager,
            config,
            key_prefix: "vector:".to_string(),
        }
    }

    /// 创建默认向量缓存
    pub fn with_default_config(cache_manager: Arc<CacheManager>) -> Self {
        Self::new(cache_manager, VectorCacheConfig::default())
    }

    /// 设置键前缀
    pub fn with_prefix(mut self, prefix: String) -> Self {
        self.key_prefix = prefix;
        self
    }

    /// 缓存向量
    pub fn put(&self, entry: &VectorCacheEntry) -> Result<()> {
        let key = self.make_key(&entry.id);
        let value = self.serialize_entry(entry)?;
        self.cache_manager.set(&key, &value)
    }

    /// 获取向量
    pub fn get(&self, id: &str) -> Result<Option<VectorCacheEntry>> {
        let key = self.make_key(id);
        match self.cache_manager.get(&key)? {
            CacheResult::Hit(value, _tier) => {
                let mut entry = self.deserialize_entry(&value)?;
                if self.config.enable_stats {
                    entry.record_access();
                    // 更新缓存中的访问统计
                    let _ = self.put(&entry);
                }
                Ok(Some(entry))
            },
            CacheResult::Miss => Ok(None),
        }
    }

    /// 删除向量
    pub fn delete(&self, id: &str) -> Result<bool> {
        let key = self.make_key(id);
        self.cache_manager.delete(&key)
    }

    /// 检查向量是否在缓存中
    pub fn contains(&self, id: &str) -> Result<bool> {
        let key = self.make_key(id);
        match self.cache_manager.get(&key)? {
            CacheResult::Hit(_, _) => Ok(true),
            CacheResult::Miss => Ok(false),
        }
    }

    /// 批量缓存向量
    pub fn batch_put(&self, entries: &[VectorCacheEntry]) -> Result<()> {
        for entry in entries {
            self.put(entry)?;
        }
        Ok(())
    }

    /// 批量获取向量
    pub fn batch_get(&self, ids: &[String]) -> Result<Vec<Option<VectorCacheEntry>>> {
        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            results.push(self.get(id)?);
        }
        Ok(results)
    }

    /// 清空缓存
    pub fn clear(&self) -> Result<()> {
        self.cache_manager.clear()
    }

    /// 获取缓存统计
    pub fn get_stats(&self) -> Result<VectorCacheStats> {
        let metrics = self.cache_manager.get_metrics()?;
        Ok(VectorCacheStats {
            total_hits: metrics.total_hits,
            total_misses: metrics.total_misses,
            hit_rate: metrics.hit_rate(),
            total_size_bytes: metrics.total_size_bytes,
            item_count: metrics.item_count,
        })
    }

    // 内部辅助方法

    fn make_key(&self, id: &str) -> String {
        format!("{}{}", self.key_prefix, id)
    }

    fn serialize_entry(&self, entry: &VectorCacheEntry) -> Result<Vec<u8>> {
        bincode::serialize(entry)
            .map_err(|e| crate::Error::serialization(format!("序列化失败: {}", e)))
    }

    fn deserialize_entry(&self, data: &[u8]) -> Result<VectorCacheEntry> {
        bincode::deserialize(data)
            .map_err(|e| crate::Error::serialization(format!("反序列化失败: {}", e)))
    }
}

/// 向量缓存统计
#[derive(Debug, Clone)]
pub struct VectorCacheStats {
    /// 总命中次数
    pub total_hits: u64,
    /// 总未命中次数
    pub total_misses: u64,
    /// 命中率
    pub hit_rate: f64,
    /// 总缓存大小（字节）
    pub total_size_bytes: usize,
    /// 缓存项数量
    pub item_count: usize,
}

impl VectorCacheStats {
    /// 创建空统计
    pub fn empty() -> Self {
        Self {
            total_hits: 0,
            total_misses: 0,
            hit_rate: 0.0,
            total_size_bytes: 0,
            item_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::create_memory_cache_manager;

    #[test]
    fn test_vector_cache_entry() {
        let vector = vec![1.0, 2.0, 3.0];
        let mut entry = VectorCacheEntry::new("test_id".to_string(), vector.clone());
        
        assert_eq!(entry.id, "test_id");
        assert_eq!(entry.vector, vector);
        assert_eq!(entry.dimension(), 3);
        assert_eq!(entry.access_count, 0);
        
        entry.record_access();
        assert_eq!(entry.access_count, 1);
    }

    #[test]
    fn test_vector_cache_config() {
        let config = VectorCacheConfig::default();
        assert_eq!(config.max_vectors, 10000);
        assert!(config.enable_stats);
    }

    #[tokio::test]
    async fn test_vector_cache_operations() {
        let cache_manager = Arc::new(create_memory_cache_manager().unwrap());
        let vector_cache = VectorCache::with_default_config(cache_manager);
        
        // 创建测试向量
        let vector = vec![1.0, 2.0, 3.0, 4.0];
        let entry = VectorCacheEntry::new("test_vec".to_string(), vector.clone());
        
        // 缓存向量
        vector_cache.put(&entry).unwrap();
        
        // 获取向量
        let cached = vector_cache.get("test_vec").unwrap();
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().vector, vector);
        
        // 检查存在性
        assert!(vector_cache.contains("test_vec").unwrap());
        
        // 删除向量
        assert!(vector_cache.delete("test_vec").unwrap());
        assert!(!vector_cache.contains("test_vec").unwrap());
    }
}



