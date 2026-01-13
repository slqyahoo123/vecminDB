// 缓存模块
// 提供特征提取结果的缓存功能

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use serde::{Serialize, Deserialize};

/// 缓存配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// 是否启用缓存
    pub enabled: bool,
    /// 最大缓存条目数
    pub max_entries: usize,
    /// 缓存过期时间(秒)
    pub ttl: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_entries: 1000,
            ttl: 3600,
        }
    }
}

/// 特征缓存
#[derive(Debug)]
pub struct FeatureCache {
    cache: Arc<Mutex<HashMap<String, CacheEntry>>>,
    config: CacheConfig,
}

/// 缓存条目
#[derive(Debug, Clone)]
struct CacheEntry {
    features: Vec<f32>,
    timestamp: u64,
}

impl FeatureCache {
    /// 创建新的缓存
    pub fn new(config: CacheConfig) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            config,
        }
    }
    
    /// 获取缓存的特征
    pub fn get(&self, key: &str) -> Option<Vec<f32>> {
        if !self.config.enabled {
            return None;
        }
        
        let cache = self.cache.lock().unwrap();
        cache.get(key).map(|entry| entry.features.clone())
    }
    
    /// 设置缓存
    pub fn set(&self, key: &str, features: Vec<f32>) {
        if !self.config.enabled {
            return;
        }
        
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
            
        let entry = CacheEntry {
            features,
            timestamp: now,
        };
        
        let mut cache = self.cache.lock().unwrap();
        
        // 检查缓存大小
        if cache.len() >= self.config.max_entries {
            // 清理过期条目
            self.cleanup(&mut cache);
        }
        
        cache.insert(key.to_string(), entry);
    }
    
    /// 清理过期缓存条目
    fn cleanup(&self, cache: &mut HashMap<String, CacheEntry>) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
            
        let expired_keys: Vec<String> = cache
            .iter()
            .filter(|(_, entry)| now - entry.timestamp > self.config.ttl)
            .map(|(key, _)| key.clone())
            .collect();
            
        for key in expired_keys {
            cache.remove(&key);
        }
    }
} 