// 缓存策略模块
// 提供缓存管理策略

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use chrono::{DateTime, Duration, Utc};

use crate::Result;

/// 缓存策略接口
pub trait CachePolicy: Send + Sync {
    /// 添加缓存项
    fn add(&mut self, key: &str, size: usize) -> Result<()>;
    
    /// 访问缓存项
    fn access(&mut self, key: &str) -> Result<()>;
    
    /// 移除缓存项
    fn remove(&mut self, key: &str) -> Result<()>;
    
    /// 获取要淘汰的键
    fn get_eviction_candidates(&self, count: usize) -> Result<Vec<String>>;
    
    /// 清理缓存
    fn clear(&mut self) -> Result<()>;
}

/// 缓存管理器
#[derive(Clone)]
pub struct CacheManager {
    policy: Arc<Mutex<Box<dyn CachePolicy>>>,
    stats: Arc<Mutex<CacheStats>>,
}

/// 缓存统计信息
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// 缓存命中次数
    pub hits: usize,
    /// 缓存未命中次数
    pub misses: usize,
    /// 缓存项数量
    pub item_count: usize,
    /// 缓存总大小(字节)
    pub total_size: usize,
    /// 缓存淘汰次数
    pub evictions: usize,
}

impl CacheManager {
    /// 创建新的缓存管理器
    pub fn new(policy: Box<dyn CachePolicy>) -> Self {
        Self {
            policy: Arc::new(Mutex::new(policy)),
            stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }
    
    /// 添加缓存项
    pub fn add(&self, key: &str, size: usize) -> Result<()> {
        let mut policy = self.policy.lock().unwrap();
        policy.add(key, size)?;
        
        let mut stats = self.stats.lock().unwrap();
        stats.item_count += 1;
        stats.total_size += size;
        
        Ok(())
    }
    
    /// 访问缓存项
    pub fn access(&self, key: &str, hit: bool) -> Result<()> {
        let mut policy = self.policy.lock().unwrap();
        policy.access(key)?;
        
        let mut stats = self.stats.lock().unwrap();
        if hit {
            stats.hits += 1;
        } else {
            stats.misses += 1;
        }
        
        Ok(())
    }
    
    /// 移除缓存项
    pub fn remove(&self, key: &str, size: usize) -> Result<()> {
        let mut policy = self.policy.lock().unwrap();
        policy.remove(key)?;
        
        let mut stats = self.stats.lock().unwrap();
        stats.item_count -= 1;
        stats.total_size -= size;
        
        Ok(())
    }
    
    /// 获取淘汰候选项
    pub fn get_eviction_candidates(&self, count: usize) -> Result<Vec<String>> {
        let policy = self.policy.lock().unwrap();
        policy.get_eviction_candidates(count)
    }
    
    /// 获取缓存统计信息
    pub fn get_stats(&self) -> Result<CacheStats> {
        let stats = self.stats.lock().unwrap();
        Ok(stats.clone())
    }
}

/// LRU缓存策略
pub struct LRUPolicy {
    items: HashMap<String, usize>, // 键 -> 大小
    queue: VecDeque<String>,       // LRU队列
    max_size: usize,               // 最大缓存大小
    current_size: usize,           // 当前缓存大小
}

impl LRUPolicy {
    /// 创建新的LRU策略
    pub fn new(max_size: usize) -> Self {
        Self {
            items: HashMap::new(),
            queue: VecDeque::new(),
            max_size,
            current_size: 0,
        }
    }
}

impl CachePolicy for LRUPolicy {
    fn add(&mut self, key: &str, size: usize) -> Result<()> {
        // 如果键已存在,先移除
        if let Some(old_size) = self.items.remove(key) {
            self.current_size -= old_size;
            // 从队列中移除（实际实现应该更高效）
            self.queue.retain(|k| k != key);
        }
        
        // 添加新项
        self.items.insert(key.to_string(), size);
        self.queue.push_back(key.to_string());
        self.current_size += size;
        
        Ok(())
    }
    
    fn access(&mut self, key: &str) -> Result<()> {
        if self.items.contains_key(key) {
            // 将键移到队列末尾
            self.queue.retain(|k| k != key);
            self.queue.push_back(key.to_string());
        }
        Ok(())
    }
    
    fn remove(&mut self, key: &str) -> Result<()> {
        if let Some(size) = self.items.remove(key) {
            self.current_size -= size;
            // 从队列中移除
            self.queue.retain(|k| k != key);
        }
        Ok(())
    }
    
    fn get_eviction_candidates(&self, count: usize) -> Result<Vec<String>> {
        // 返回队列前面的项(最久未使用)
        let mut candidates = Vec::new();
        let mut i = 0;
        
        for key in &self.queue {
            if i >= count {
                break;
            }
            candidates.push(key.clone());
            i += 1;
        }
        
        Ok(candidates)
    }
    
    fn clear(&mut self) -> Result<()> {
        self.items.clear();
        self.queue.clear();
        self.current_size = 0;
        Ok(())
    }
}

/// TTL缓存策略
pub struct TTLPolicy {
    items: HashMap<String, (usize, DateTime<Utc>)>, // 键 -> (大小, 过期时间)
    ttl: Duration,                                 // 生存时间
}

impl TTLPolicy {
    /// 创建新的TTL策略
    pub fn new(ttl_seconds: i64) -> Self {
        Self {
            items: HashMap::new(),
            ttl: Duration::seconds(ttl_seconds),
        }
    }
}

impl CachePolicy for TTLPolicy {
    fn add(&mut self, key: &str, size: usize) -> Result<()> {
        let expires_at = Utc::now() + self.ttl;
        self.items.insert(key.to_string(), (size, expires_at));
        Ok(())
    }
    
    fn access(&mut self, key: &str) -> Result<()> {
        // TTL策略访问不更新过期时间
        Ok(())
    }
    
    fn remove(&mut self, key: &str) -> Result<()> {
        self.items.remove(key);
        Ok(())
    }
    
    fn get_eviction_candidates(&self, count: usize) -> Result<Vec<String>> {
        let now = Utc::now();
        
        // 找出过期项
        let mut expired: Vec<String> = self.items
            .iter()
            .filter(|(_, (_, expires_at))| *expires_at <= now)
            .map(|(key, _)| key.clone())
            .collect();
            
        // 如果过期项不足,按过期时间排序添加即将过期的项
        if expired.len() < count {
            let mut candidates: Vec<_> = self.items
                .iter()
                .filter(|(key, _)| !expired.contains(key))
                .collect();
                
            candidates.sort_by(|a, b| a.1.1.cmp(&b.1.1));
            
            for (key, _) in candidates {
                if expired.len() >= count {
                    break;
                }
                expired.push(key.clone());
            }
        }
        
        Ok(expired)
    }
    
    fn clear(&mut self) -> Result<()> {
        self.items.clear();
        Ok(())
    }
}

/// 默认缓存策略(结合LRU和TTL)
pub struct DefaultPolicy {
    lru: LRUPolicy,
    ttl: TTLPolicy,
}

impl DefaultPolicy {
    /// 创建新的默认策略
    pub fn new(max_size: usize, ttl_seconds: i64) -> Self {
        Self {
            lru: LRUPolicy::new(max_size),
            ttl: TTLPolicy::new(ttl_seconds),
        }
    }
}

impl CachePolicy for DefaultPolicy {
    fn add(&mut self, key: &str, size: usize) -> Result<()> {
        self.lru.add(key, size)?;
        self.ttl.add(key, size)?;
        Ok(())
    }
    
    fn access(&mut self, key: &str) -> Result<()> {
        self.lru.access(key)?;
        self.ttl.access(key)?;
        Ok(())
    }
    
    fn remove(&mut self, key: &str) -> Result<()> {
        self.lru.remove(key)?;
        self.ttl.remove(key)?;
        Ok(())
    }
    
    fn get_eviction_candidates(&self, count: usize) -> Result<Vec<String>> {
        // 先检查TTL过期项
        let ttl_candidates = self.ttl.get_eviction_candidates(count)?;
        
        if ttl_candidates.len() >= count {
            return Ok(ttl_candidates);
        }
        
        // 不足则从LRU获取
        let mut candidates = ttl_candidates;
        let lru_candidates = self.lru.get_eviction_candidates(count - candidates.len())?;
        
        // 合并结果,确保没有重复
        for key in lru_candidates {
            if !candidates.contains(&key) {
                candidates.push(key);
                if candidates.len() >= count {
                    break;
                }
            }
        }
        
        Ok(candidates)
    }
    
    fn clear(&mut self) -> Result<()> {
        self.lru.clear()?;
        self.ttl.clear()?;
        Ok(())
    }
} 