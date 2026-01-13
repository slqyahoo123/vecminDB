use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use crate::Result;
use crate::Error;
use super::policy::{CachePolicy, LRUPolicy};

/// 缓存结果类型
pub enum CacheResult<T> {
    /// 缓存命中
    Hit(T),
    /// 缓存未命中
    Miss,
}

/// 缓存项
pub struct CacheEntry<T> {
    /// 数据
    pub value: T,
    /// 大小（字节）
    pub size: usize,
    /// 创建时间
    pub created_at: Instant,
    /// 上次访问时间
    pub last_accessed: Instant,
    /// 访问次数
    pub access_count: usize,
}

impl<T> CacheEntry<T> {
    /// 创建新的缓存项
    pub fn new(value: T, size: usize) -> Self {
        let now = Instant::now();
        Self {
            value,
            size,
            created_at: now,
            last_accessed: now,
            access_count: 0,
        }
    }
    
    /// 访问缓存项
    pub fn access(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

/// 缓存接口
pub trait Cache<K, V> {
    /// 获取缓存项
    fn get(&self, key: &K) -> Result<CacheResult<V>>;
    
    /// 设置缓存项
    fn set(&self, key: K, value: V, size: usize) -> Result<()>;
    
    /// 删除缓存项
    fn remove(&self, key: &K) -> Result<()>;
    
    /// 清空缓存
    fn clear(&self) -> Result<()>;
    
    /// 获取缓存中的项数
    fn len(&self) -> usize;
    
    /// 检查缓存是否为空
    fn is_empty(&self) -> bool;
}

/// 内存缓存实现
pub struct MemoryCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    data: RwLock<HashMap<K, CacheEntry<V>>>,
    policy: Arc<Mutex<Box<dyn CachePolicy>>>,
    max_size: usize,
    current_size: Arc<Mutex<usize>>,
    ttl: Option<Duration>,
}

impl<K, V> MemoryCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// 创建新的内存缓存
    pub fn new(max_size: usize) -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
            policy: Arc::new(Mutex::new(Box::new(LRUPolicy::new(max_size)))),
            max_size,
            current_size: Arc::new(Mutex::new(0)),
            ttl: None,
        }
    }
    
    /// 创建带TTL的内存缓存
    pub fn with_ttl(max_size: usize, ttl: Duration) -> Self {
        let mut cache = Self::new(max_size);
        cache.ttl = Some(ttl);
        cache
    }
    
    /// 设置缓存策略
    pub fn with_policy(mut self, policy: Box<dyn CachePolicy>) -> Self {
        self.policy = Arc::new(Mutex::new(policy));
        self
    }
    
    /// 检查并淘汰过期项
    fn evict_expired(&self) -> Result<()> 
    where
        Self: Cache<K, V>,
    {
        if let Some(ttl) = self.ttl {
            let now = Instant::now();
            let mut to_remove = Vec::new();
            
            // 查找过期项
            {
                let data = self.data.read().unwrap();
                for (key, entry) in data.iter() {
                    if now.duration_since(entry.created_at) > ttl {
                        to_remove.push(key.clone());
                    }
                }
            }
            
            // 移除过期项
            for key in to_remove {
                <Self as Cache<K, V>>::remove(self, &key)?;
            }
        }
        
        Ok(())
    }
    
    /// 根据缓存策略淘汰项
    fn evict_by_policy(&self, required_size: usize) -> Result<()> 
    where
        Self: Cache<K, V>,
        K: serde::de::DeserializeOwned,
    {
        let mut current_size = self.current_size.lock().unwrap();
        
        // 如果空间足够，不需要淘汰
        if *current_size + required_size <= self.max_size {
            return Ok(());
        }
        
        // 计算需要淘汰的大小
        let to_evict = (*current_size + required_size) - self.max_size;
        
        // 获取淘汰候选项
        let mut policy = self.policy.lock().unwrap();
        // 获取足够的候选项，确保能释放足够空间
        let cache_len = <Self as Cache<K, V>>::len(self);
        let candidates_count = (to_evict as f64 / (*current_size as f64 / cache_len as f64)).ceil() as usize;
        let candidates = policy.get_eviction_candidates(candidates_count.max(1))?;
        
        if candidates.is_empty() {
            return Err(Error::cache("无法找到合适的淘汰项".to_string()));
        }
        
        let mut evicted_size = 0;
        let mut to_remove = Vec::new();
        
        // 计算要移除的项
        {
            let data = self.data.read().unwrap();
            for key_str in candidates {
                // 尝试将JSON字符串转换回原始键类型
                match deserialize_key::<K>(&key_str) {
                    Ok(key) => {
                        if let Some(entry) = data.get(&key) {
                            evicted_size += entry.size;
                            to_remove.push(key);
                            
                            if evicted_size >= to_evict {
                                break;
                            }
                        }
                    },
                    Err(e) => {
                        log::warn!("淘汰缓存时无法反序列化键 {}: {}", key_str, e);
                        // 继续处理其他键
                    }
                }
            }
        }
        
        // 如果没有找到足够的项来淘汰，清空缓存
        if evicted_size < to_evict && !to_remove.is_empty() {
            log::warn!("无法淘汰足够的缓存项，将清空所有缓存");
            let result = <Self as Cache<K, V>>::clear(self);
            *current_size = 0;
            return result;
        }
        
        // 移除选定的项
        for key in to_remove {
            let size_removed = match <Self as Cache<K, V>>::remove(self, &key) {
                Ok(_) => {
                    if let Some(entry_size) = self.data.read().unwrap().get(&key).map(|e| e.size) {
                        entry_size
                    } else {
                        0 // 键已经被移除
                    }
                },
                Err(e) => {
                    log::error!("移除缓存项时出错: {}", e);
                    0 // 出错时假设没有移除任何大小
                }
            };
            evicted_size += size_removed;
        }
        
        // 更新当前大小
        *current_size = if evicted_size <= *current_size {
            *current_size - evicted_size
        } else {
            0 // 防止下溢
        };
        
        Ok(())
    }
}

impl<K, V> Cache<K, V> for MemoryCache<K, V>
where
    K: Eq + Hash + Clone + serde::Serialize,
    V: Clone,
{
    fn get(&self, key: &K) -> Result<CacheResult<V>> {
        // 先检查并淘汰过期项
        self.evict_expired()?;
        
        // 尝试获取缓存项
        let mut data = self.data.write().unwrap();
        
        if let Some(entry) = data.get_mut(key) {
            // 更新访问信息
            entry.access();
            
            // 更新策略
            let key_str = serde_json::to_string(key)?;
            let mut policy = self.policy.lock().unwrap();
            policy.access(&key_str)?;
            
            // 返回值
            Ok(CacheResult::Hit(entry.value.clone()))
        } else {
            Ok(CacheResult::Miss)
        }
    }
    
    fn set(&self, key: K, value: V, size: usize) -> Result<()> {
        // 确保有足够空间（evict_by_policy 需要 K: DeserializeOwned，但这里不满足）
        // 先尝试清理过期项
        self.evict_expired()?;
        
        // 如果空间仍然不足，使用简单的清理策略
        {
            let current_size = self.current_size.lock().unwrap();
            if *current_size + size > self.max_size {
                // 简单的清理：移除最旧的项
                let mut data = self.data.write().unwrap();
                if !data.is_empty() {
                    // 找到最旧的项并移除
                    let oldest_key = data.iter()
                        .min_by_key(|(_, entry)| entry.created_at)
                        .map(|(k, _)| k.clone());
                    if let Some(key) = oldest_key {
                        data.remove(&key);
                    }
                }
            }
        }
        
        // 如果键已存在，先移除
        self.remove(&key)?;
        
        // 创建新缓存项
        let entry = CacheEntry::new(value, size);
        
        // 添加到缓存
        {
            let mut data = self.data.write().unwrap();
            data.insert(key.clone(), entry);
        }
        
        // 更新策略
        let key_str = serde_json::to_string(&key)?;
        let mut policy = self.policy.lock().unwrap();
        policy.add(&key_str, size)?;
        
        // 更新当前大小
        let mut current_size = self.current_size.lock().unwrap();
        *current_size += size;
        
        Ok(())
    }
    
    fn remove(&self, key: &K) -> Result<()> {
        let mut data = self.data.write().unwrap();
        
        if let Some(entry) = data.remove(key) {
            // 更新策略
            let key_str = serde_json::to_string(key)?;
            let mut policy = self.policy.lock().unwrap();
            policy.remove(&key_str)?;
            
            // 更新当前大小
            let mut current_size = self.current_size.lock().unwrap();
            *current_size -= entry.size;
        }
        
        Ok(())
    }
    
    fn clear(&self) -> Result<()> {
        // 清空缓存
        {
            let mut data = self.data.write().unwrap();
            data.clear();
        }
        
        // 清空策略
        {
            let mut policy = self.policy.lock().unwrap();
            policy.clear()?;
        }
        
        // 重置当前大小
        let mut current_size = self.current_size.lock().unwrap();
        *current_size = 0;
        
        Ok(())
    }
    
    fn len(&self) -> usize {
        let data = self.data.read().unwrap();
        data.len()
    }
    
    fn is_empty(&self) -> bool {
        let data = self.data.read().unwrap();
        data.is_empty()
    }
}

/// 辅助函数：从字符串反序列化键
fn deserialize_key<K: serde::de::DeserializeOwned>(key_str: &str) -> Result<K> {
    serde_json::from_str(key_str)
        .map_err(|e| Error::serialization(format!("反序列化缓存键失败: {}", e)))
}
