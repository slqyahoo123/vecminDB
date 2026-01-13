use std::collections::{HashMap, VecDeque, HashSet};
use std::time::{Duration, Instant};
use std::sync::{Arc, RwLock, Mutex};
use std::io::{Read, Write};
use chrono::Utc;

use crate::error::{Error, Result};
use crate::cache_common::EvictionPolicy;
use super::manager::CacheMetrics;

// 集成内存管理系统
use crate::memory::{
    MemoryManager, ObjectType, ManagedPtr, MemoryConfig, get_global_memory_manager,
    allocate, deallocate, get_memory_stats, memory_health_check
};
use crate::memory::pool::MemoryPoolManager;
use crate::memory::cache_optimizer::CacheOptimizer;
use crate::compat::{ManagedTensorData, TensorDataType};

/// 智能内存缓存项 - 使用内存管理系统
struct SmartMemoryCacheItem {
    /// 键
    key: String,
    /// 智能管理的值（支持Tensor数据）
    value: CacheValue,
    /// 创建时间
    created_at: Instant,
    /// 最后访问时间
    last_accessed: Instant,
    /// 访问次数
    access_count: u64,
    /// 过期时间（如果有）
    expires_at: Option<Instant>,
    /// 值大小（字节）
    size: usize,
    /// 压缩状态
    is_compressed: bool,
    /// 原始大小（压缩前）
    original_size: usize,
    /// 访问频率权重
    frequency_weight: f64,
}

/// 缓存值类型
#[derive(Debug, Clone)]
pub enum CacheValue {
    /// 原始字节数据
    Raw(Arc<ManagedPtr>),
    /// 张量数据
    Tensor(ManagedTensorData),
    /// 压缩数据
    Compressed {
        data: Arc<ManagedPtr>,
        original_size: usize,
        compression_ratio: f64,
    },
    /// 类型化张量数据
    TypedTensor {
        data: Arc<ManagedPtr>,
        data_type: TensorDataType,
        shape: Vec<usize>,
    },
}

impl CacheValue {
    /// 获取数据大小
    pub fn size(&self) -> usize {
        match self {
            Self::Raw(ptr) => ptr.size,
            Self::Tensor(tensor) => tensor.total_bytes(),
            Self::Compressed { data, .. } => data.size,
            Self::TypedTensor { data, .. } => data.size,
        }
    }
    
    /// 获取原始数据
    pub fn get_raw_data(&self) -> Result<Vec<u8>> {
        match self {
            Self::Raw(ptr) => {
                ptr.mark_accessed();
                unsafe {
                    Ok(std::slice::from_raw_parts(ptr.ptr.as_ptr(), ptr.size).to_vec())
                }
            }
            Self::Tensor(tensor) => {
                let slice = tensor.as_slice()?;
                Ok(slice.to_vec())
            }
            Self::Compressed { data, original_size, .. } => {
                // 解压缩数据
                data.mark_accessed();
                let compressed_data = unsafe {
                    std::slice::from_raw_parts(data.ptr.as_ptr(), data.size)
                };
                
                // 使用Deflate解压缩
                let mut decompressed = Vec::new();
                let mut decoder = flate2::read::DeflateDecoder::new(compressed_data);
                std::io::Read::read_to_end(&mut decoder, &mut decompressed)
                    .map_err(|e| Error::compression(format!("Failed to decompress data: {}", e)))?;
                
                if decompressed.len() != *original_size {
                    return Err(Error::compression(format!(
                        "Decompressed size mismatch: expected {}, got {}", 
                        original_size, decompressed.len()
                    )));
                }
                
                Ok(decompressed)
            }
            Self::TypedTensor { data, .. } => {
                data.mark_accessed();
                let slice = unsafe {
                    std::slice::from_raw_parts(data.ptr.as_ptr(), data.size)
                };
                Ok(slice.to_vec())
            }
        }
    }
    
    /// 从字节数据创建缓存值
    pub fn from_bytes(data: &[u8], compress_threshold: usize) -> Result<Self> {
        if data.len() > compress_threshold {
            // 使用Deflate压缩大数据
            let mut encoder = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
            std::io::Write::write_all(&mut encoder, data)
                .map_err(|e| Error::compression(format!("Failed to compress data: {}", e)))?;
            let compressed = encoder.finish()
                .map_err(|e| Error::compression(format!("Failed to finish compression: {}", e)))?;
            
            if compressed.len() < data.len() {
                let managed_ptr = Arc::new(allocate(compressed.len(), ObjectType::Cache)?);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        compressed.as_ptr(),
                        managed_ptr.ptr.as_ptr(),
                        compressed.len()
                    );
                }
                
                let compression_ratio = compressed.len() as f64 / data.len() as f64;
                
                return Ok(Self::Compressed {
                    data: managed_ptr,
                    original_size: data.len(),
                    compression_ratio,
                });
            }
        }
        
        // 不压缩或压缩无效果时使用原始数据
        let managed_ptr = Arc::new(allocate(data.len(), ObjectType::Cache)?);
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                managed_ptr.ptr.as_ptr(),
                data.len()
            );
        }
        
        Ok(Self::Raw(managed_ptr))
    }
    
    /// 从张量创建缓存值
    pub fn from_tensor(tensor: ManagedTensorData) -> Self {
        Self::Tensor(tensor)
    }
}

impl SmartMemoryCacheItem {
    /// 创建新的智能缓存项
    fn new(key: String, value: CacheValue, ttl: Option<Duration>) -> Self {
        let now = Instant::now();
        let expires_at = ttl.map(|t| now + t);
        let size = value.size();
        
        let (is_compressed, original_size) = match &value {
            CacheValue::Compressed { original_size, .. } => (true, *original_size),
            _ => (false, size),
        };
        
        Self {
            key,
            value,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            expires_at,
            size,
            is_compressed,
            original_size,
            frequency_weight: 1.0,
        }
    }
    
    /// 检查是否已过期
    fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            Instant::now() > expires_at
        } else {
            false
        }
    }
    
    /// 更新访问信息
    fn mark_accessed(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
        
        // 更新频率权重（时间衰减）
        let time_factor = 1.0 - (self.last_accessed.duration_since(self.created_at).as_secs_f64() / 3600.0).min(0.9);
        self.frequency_weight = (self.access_count as f64).log2() * time_factor;
    }
    
    /// 获取内存效率分数
    fn get_memory_efficiency_score(&self) -> f64 {
        if self.is_compressed {
            let compression_savings = 1.0 - (self.size as f64 / self.original_size as f64);
            compression_savings * 0.7 + self.frequency_weight * 0.3
        } else {
            self.frequency_weight * 0.8 + 0.2 // 基础分数
        }
    }
}

/// 淘汰策略实现
trait EvictionStrategy: Send + Sync {
    /// 添加项
    fn add_item(&mut self, key: &str);
    
    /// 更新项访问
    fn update_item(&mut self, key: &str);
    
    /// 移除项
    fn remove_item(&mut self, key: &str);
    
    /// 获取要淘汰的键
    fn get_eviction_candidates(&self, count: usize) -> Vec<String>;
    
    /// 清除所有项
    fn clear(&mut self);
}

/// LRU淘汰策略
struct LRUStrategy {
    /// 访问顺序
    access_order: VecDeque<String>,
    /// 键到位置的映射
    key_positions: HashMap<String, usize>,
    /// 下一个位置索引
    next_position: usize,
}

impl LRUStrategy {
    /// 创建新的LRU策略
    fn new() -> Self {
        Self {
            access_order: VecDeque::new(),
            key_positions: HashMap::new(),
            next_position: 0,
        }
    }
}

impl EvictionStrategy for LRUStrategy {
    fn add_item(&mut self, key: &str) {
        // 如果键已存在，先移除
        if self.key_positions.contains_key(key) {
            self.remove_item(key);
        }
        
        // 添加到访问顺序末尾
        self.access_order.push_back(key.to_string());
        self.key_positions.insert(key.to_string(), self.next_position);
        self.next_position += 1;
    }
    
    fn update_item(&mut self, key: &str) {
        // 移除旧位置
        self.remove_item(key);
        
        // 添加到访问顺序末尾
        self.access_order.push_back(key.to_string());
        self.key_positions.insert(key.to_string(), self.next_position);
        self.next_position += 1;
    }
    
    fn remove_item(&mut self, key: &str) {
        if let Some(_) = self.key_positions.remove(key) {
            // 从访问顺序中移除（比较低效，但实际应用中数据量不大）
            self.access_order.retain(|k| k != key);
        }
    }
    
    fn get_eviction_candidates(&self, count: usize) -> Vec<String> {
        let mut candidates = Vec::with_capacity(count);
        
        // 从最早的访问项开始淘汰
        for (i, key) in self.access_order.iter().enumerate() {
            if i >= count {
                break;
            }
            candidates.push(key.clone());
        }
        
        candidates
    }
    
    fn clear(&mut self) {
        self.access_order.clear();
        self.key_positions.clear();
        self.next_position = 0;
    }
}

/// LFU淘汰策略
struct LFUStrategy {
    /// 访问频率计数
    frequency: HashMap<String, u64>,
    /// 频率到键的映射
    freq_to_keys: HashMap<u64, HashSet<String>>,
    /// 最小频率
    min_frequency: u64,
}

impl LFUStrategy {
    /// 创建新的LFU策略
    fn new() -> Self {
        Self {
            frequency: HashMap::new(),
            freq_to_keys: HashMap::new(),
            min_frequency: 0,
        }
    }
}

impl EvictionStrategy for LFUStrategy {
    fn add_item(&mut self, key: &str) {
        // 新项始终从频率1开始
        self.frequency.insert(key.to_string(), 1);
        self.freq_to_keys.entry(1).or_insert_with(HashSet::new).insert(key.to_string());
        self.min_frequency = 1;
    }
    
    fn update_item(&mut self, key: &str) {
        // 获取当前频率
        let freq = if let Some(freq) = self.frequency.get(key) {
            *freq
        } else {
            // 如果键不存在，添加它
            self.add_item(key);
            return;
        };
        
        // 从当前频率集合中移除
        if let Some(keys) = self.freq_to_keys.get_mut(&freq) {
            keys.remove(key);
            
            // 如果这是最小频率且集合为空，更新最小频率
            if freq == self.min_frequency && keys.is_empty() {
                self.min_frequency = freq + 1;
            }
        }
        
        // 增加频率并添加到新频率集合
        let new_freq = freq + 1;
        self.frequency.insert(key.to_string(), new_freq);
        self.freq_to_keys.entry(new_freq).or_insert_with(HashSet::new).insert(key.to_string());
    }
    
    fn remove_item(&mut self, key: &str) {
        // 获取当前频率
        if let Some(freq) = self.frequency.remove(key) {
            // 从频率集合中移除
            if let Some(keys) = self.freq_to_keys.get_mut(&freq) {
                keys.remove(key);
                
                // 如果这是最小频率且集合为空，更新最小频率
                if freq == self.min_frequency && keys.is_empty() {
                    // 找到新的最小频率
                    self.min_frequency = self.freq_to_keys.keys()
                        .filter(|&f| *f > freq && !self.freq_to_keys[f].is_empty())
                        .min()
                        .cloned()
                        .unwrap_or(0);
                }
            }
        }
    }
    
    fn get_eviction_candidates(&self, count: usize) -> Vec<String> {
        let mut candidates = Vec::with_capacity(count);
        let mut current_freq = self.min_frequency;
        
        // 从最低频率开始收集候选项
        while candidates.len() < count && current_freq <= self.frequency.values().max().unwrap_or(&0) + 1 {
            if let Some(keys) = self.freq_to_keys.get(&current_freq) {
                for key in keys {
                    if candidates.len() >= count {
                        break;
                    }
                    candidates.push(key.clone());
                }
            }
            current_freq += 1;
        }
        
        candidates
    }
    
    fn clear(&mut self) {
        self.frequency.clear();
        self.freq_to_keys.clear();
        self.min_frequency = 0;
    }
}

/// FIFO淘汰策略
struct FIFOStrategy {
    /// 插入顺序
    insertion_order: VecDeque<String>,
}

impl FIFOStrategy {
    /// 创建新的FIFO策略
    fn new() -> Self {
        Self {
            insertion_order: VecDeque::new(),
        }
    }
}

impl EvictionStrategy for FIFOStrategy {
    fn add_item(&mut self, key: &str) {
        // 如果键已存在，先移除
        self.remove_item(key);
        
        // 添加到插入顺序末尾
        self.insertion_order.push_back(key.to_string());
    }
    
    fn update_item(&mut self, key: &str) {
        // FIFO不需要更新访问顺序
    }
    
    fn remove_item(&mut self, key: &str) {
        // 从插入顺序中移除
        self.insertion_order.retain(|k| k != key);
    }
    
    fn get_eviction_candidates(&self, count: usize) -> Vec<String> {
        let mut candidates = Vec::with_capacity(count);
        
        // 从最早插入的项开始淘汰
        for (i, key) in self.insertion_order.iter().enumerate() {
            if i >= count {
                break;
            }
            candidates.push(key.clone());
        }
        
        candidates
    }
    
    fn clear(&mut self) {
        self.insertion_order.clear();
    }
}

/// MRU淘汰策略 - 最近最常使用
struct MRUStrategy {
    /// 访问顺序（最后访问的在前面）
    access_order: VecDeque<String>,
    /// 键到位置的映射
    key_positions: HashMap<String, usize>,
    /// 下一个位置索引
    next_position: usize,
}

impl MRUStrategy {
    /// 创建新的MRU策略
    fn new() -> Self {
        Self {
            access_order: VecDeque::new(),
            key_positions: HashMap::new(),
            next_position: 0,
        }
    }
}

impl EvictionStrategy for MRUStrategy {
    fn add_item(&mut self, key: &str) {
        // 如果键已存在，先移除
        self.remove_item(key);
        
        // 添加到前面（MRU位置）
        self.access_order.push_front(key.to_string());
        self.key_positions.insert(key.to_string(), self.next_position);
        self.next_position += 1;
    }
    
    fn update_item(&mut self, key: &str) {
        if self.key_positions.contains_key(key) {
            // 移除旧位置
            self.access_order.retain(|k| k != key);
            
            // 添加到前面（MRU位置）
            self.access_order.push_front(key.to_string());
            self.key_positions.insert(key.to_string(), self.next_position);
            self.next_position += 1;
        }
    }
    
    fn remove_item(&mut self, key: &str) {
        if self.key_positions.remove(key).is_some() {
            self.access_order.retain(|k| k != key);
        }
    }
    
    fn get_eviction_candidates(&self, count: usize) -> Vec<String> {
        // MRU策略：返回最近访问的项（从前面开始）
        self.access_order
            .iter()
            .take(count)
            .cloned()
            .collect()
    }
    
    fn clear(&mut self) {
        self.access_order.clear();
        self.key_positions.clear();
        self.next_position = 0;
    }
}

/// 内存缓存实现
pub struct MemoryCache {
    /// 缓存数据
    data: RwLock<HashMap<String, SmartMemoryCacheItem>>,
    /// 淘汰策略
    eviction_strategy: Mutex<Box<dyn EvictionStrategy>>,
    /// 最大大小（字节）
    max_size_bytes: usize,
    /// 最大项数
    max_items: usize,
    /// 当前大小（字节）
    current_size: RwLock<usize>,
    /// 默认TTL
    default_ttl: Option<Duration>,
    /// 缓存统计
    metrics: RwLock<CacheMetrics>,
    /// 内存管理器配置
    memory_config: MemoryConfig,
    /// 内存管理器引用
    memory_manager: Arc<MemoryManager>,
    /// 内存健康检查间隔
    health_check_interval: Duration,
    /// 上次健康检查时间
    last_health_check: RwLock<Instant>,
    /// 内存池管理器
    memory_pool_manager: Arc<MemoryPoolManager>,
    /// 缓存优化器
    cache_optimizer: Arc<CacheOptimizer>,
}

impl MemoryCache {
    /// 创建新的内存缓存
    pub fn new(
        max_size_bytes: usize,
        max_items: usize,
        ttl: Option<Duration>,
        eviction_policy: EvictionPolicy,
    ) -> Result<Self> {
        // 创建淘汰策略
        let strategy: Box<dyn EvictionStrategy> = match eviction_policy {
            EvictionPolicy::LRU => Box::new(LRUStrategy::new()),
            EvictionPolicy::MRU => Box::new(MRUStrategy::new()),
            EvictionPolicy::LFU => Box::new(LFUStrategy::new()),
            EvictionPolicy::FIFO => Box::new(FIFOStrategy::new()),
        };
        
        // 获取全局内存管理器
        let memory_manager = get_global_memory_manager();
        let memory_config = memory_manager.get_config();
        
        // 创建内存池管理器
        let memory_pool_manager = Arc::new(MemoryPoolManager::new());
        
        // 创建缓存优化器
        let cache_optimizer = Arc::new(CacheOptimizer::new());
        
        Ok(Self {
            data: RwLock::new(HashMap::new()),
            eviction_strategy: Mutex::new(strategy),
            max_size_bytes,
            max_items,
            current_size: RwLock::new(0),
            default_ttl: ttl,
            metrics: RwLock::new(CacheMetrics {
                hits: 0,
                misses: 0,
                writes: 0,
                deletes: 0,
                evictions: 0,
                avg_access_time_us: 0,
                hit_ratio: 0.0,
                last_updated: Utc::now(),
                tier: super::manager::CacheTier::Memory,
                entries: 0,
                size_bytes: 0,
                capacity_bytes: max_size_bytes,
            }),
            memory_config,
            memory_manager: Arc::new(memory_manager),
            health_check_interval: Duration::from_secs(300), // 5分钟
            last_health_check: RwLock::new(Instant::now()),
            memory_pool_manager,
            cache_optimizer,
        })
    }
    
    /// 从缓存获取值
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let start = Instant::now();
        let mut metrics = self.metrics.write().map_err(|_| Error::locks_poison("内存缓存指标锁被污染"))?;
        
        // 读取缓存
        let data = self.data.read().map_err(|_| Error::locks_poison("内存缓存数据锁被污染"))?;
        
        if let Some(item) = data.get(key) {
            // 检查是否过期
            if item.is_expired() {
                // 项已过期，记为未命中
                metrics.record_miss(start.elapsed().as_micros() as u64);
                drop(data);
                
                // 稍后删除过期项
                let _ = self.delete(key);
                
                return Ok(None);
            }
            
            // 项有效，记为命中
            metrics.record_hit(start.elapsed().as_micros() as u64);
            
            // 克隆值以返回
            let value = item.value.get_raw_data()?;
            
            // 更新访问信息（需要在读取锁之外进行写入操作）
            drop(data);
            drop(metrics);
            
            {
                let mut data = self.data.write().map_err(|_| Error::locks_poison("内存缓存数据锁被污染"))?;
                if let Some(item) = data.get_mut(key) {
                    item.mark_accessed();
                }
                
                // 更新淘汰策略
                let mut strategy = self.eviction_strategy.lock().map_err(|_| Error::locks_poison("内存缓存策略锁被污染"))?;
                strategy.update_item(key);
            }
            
            Ok(Some(value))
        } else {
            // 未命中
            metrics.record_miss(start.elapsed().as_micros() as u64);
            Ok(None)
        }
    }
    
    /// 设置缓存值
    pub fn set(&self, key: &str, value: &[u8]) -> Result<()> {
        let value_len = value.len();
        
        // 检查值大小是否超过最大限制
        if value_len > self.max_size_bytes {
            return Err(Error::resource_exceeded(format!(
                "缓存值大小超过最大限制: {} > {}", 
                value_len, self.max_size_bytes
            )));
        }
        
        let mut metrics = self.metrics.write().map_err(|_| Error::locks_poison("内存缓存指标锁被污染"))?;
        
        // 创建新的缓存项
        let new_item = SmartMemoryCacheItem::new(
            key.to_string(),
            CacheValue::from_bytes(value, 1024)?,
            self.default_ttl,
        );
        
        // 获取现有项大小（如果有）
        let mut existing_size = 0;
        {
            let data = self.data.read().map_err(|_| Error::locks_poison("内存缓存数据锁被污染"))?;
            if let Some(existing_item) = data.get(key) {
                existing_size = existing_item.size;
            }
        }
        
        // 计算大小变化
        let size_delta = value_len as isize - existing_size as isize;
        
        // 检查是否需要淘汰
        if size_delta > 0 {
            let mut current_size = self.current_size.write().map_err(|_| Error::locks_poison("内存缓存大小锁被污染"))?;
            let new_size = (*current_size as isize + size_delta) as usize;
            
            // 如果新大小超过最大限制，需要淘汰
            if new_size > self.max_size_bytes {
                drop(current_size); // 释放锁
                // 执行淘汰
                self.evict(size_delta as usize)?;
                // 重新获取锁
                current_size = self.current_size.write().map_err(|_| Error::locks_poison("内存缓存大小锁被污染"))?;
            }
            
            // 更新大小
            *current_size = (*current_size as isize + size_delta) as usize;
        }
        
        // 更新缓存
        {
            let mut data = self.data.write().map_err(|_| Error::locks_poison("内存缓存数据锁被污染"))?;
            
            // 更新指标
            if data.contains_key(key) {
                metrics.entries = metrics.entries.saturating_sub(1);
                metrics.size_bytes = metrics.size_bytes.saturating_sub(existing_size);
            }
            
            // 添加或更新缓存项
            data.insert(key.to_string(), new_item);
            
            metrics.record_write(value_len);
        }
        
        // 更新淘汰策略
        {
            let mut strategy = self.eviction_strategy.lock().map_err(|_| Error::locks_poison("内存缓存策略锁被污染"))?;
            strategy.add_item(key);
        }
        
        // 检查是否超过最大项数
        {
            let data = self.data.read().map_err(|_| Error::locks_poison("内存缓存数据锁被污染"))?;
            if data.len() > self.max_items {
                drop(data); // 释放锁
                // 淘汰超出的项
                self.evict_by_count(data.len() - self.max_items)?;
            }
        }
        
        Ok(())
    }
    
    /// 删除缓存值
    pub fn delete(&self, key: &str) -> Result<bool> {
        let mut metrics = self.metrics.write().map_err(|_| Error::locks_poison("内存缓存指标锁被污染"))?;
        
        let mut deleted = false;
        let mut size_removed = 0;
        
        // 从缓存中删除
        {
            let mut data = self.data.write().map_err(|_| Error::locks_poison("内存缓存数据锁被污染"))?;
            if let Some(item) = data.remove(key) {
                deleted = true;
                size_removed = item.size;
                
                // 更新大小
                if size_removed > 0 {
                    let mut current_size = self.current_size.write().map_err(|_| Error::locks_poison("内存缓存大小锁被污染"))?;
                    *current_size = current_size.saturating_sub(size_removed);
                }
                
                // 更新指标
                metrics.record_delete(size_removed);
            }
        }
        
        // 更新淘汰策略
        if deleted {
            let mut strategy = self.eviction_strategy.lock().map_err(|_| Error::locks_poison("内存缓存策略锁被污染"))?;
            strategy.remove_item(key);
        }
        
        Ok(deleted)
    }
    
    /// 清空缓存
    pub fn clear(&self) -> Result<()> {
        // 清空数据
        {
            let mut data = self.data.write().map_err(|_| Error::locks_poison("内存缓存数据锁被污染"))?;
            data.clear();
        }
        
        // 重置大小
        {
            let mut current_size = self.current_size.write().map_err(|_| Error::locks_poison("内存缓存大小锁被污染"))?;
            *current_size = 0;
        }
        
        // 清空淘汰策略
        {
            let mut strategy = self.eviction_strategy.lock().map_err(|_| Error::locks_poison("内存缓存策略锁被污染"))?;
            strategy.clear();
        }
        
        // 重置指标（但保留历史统计）
        {
            let mut metrics = self.metrics.write().map_err(|_| Error::locks_poison("内存缓存指标锁被污染"))?;
            metrics.entries = 0;
            metrics.size_bytes = 0;
            metrics.last_updated = Utc::now();
        }
        
        Ok(())
    }
    
    /// 清理过期项
    pub fn cleanup_expired(&self) -> Result<usize> {
        let mut cleaned_count = 0;
        let mut keys_to_remove = Vec::new();
        
        // 查找过期项
        {
            let data = self.data.read().map_err(|_| Error::locks_poison("内存缓存数据锁被污染"))?;
            for (key, item) in data.iter() {
                if item.is_expired() {
                    keys_to_remove.push(key.clone());
                }
            }
        }
        
        // 删除过期项
        for key in keys_to_remove {
            if let Ok(true) = self.delete(&key) {
                cleaned_count += 1;
            }
        }
        
        Ok(cleaned_count)
    }
    
    /// 根据大小淘汰缓存项
    fn evict(&self, required_space: usize) -> Result<usize> {
        let mut evicted_count = 0;
        let mut evicted_size = 0;
        
        // 估计需要淘汰的项数
        let avg_item_size = {
            let data = self.data.read().map_err(|_| Error::locks_poison("内存缓存数据锁被污染"))?;
            if data.is_empty() {
                return Ok(0); // 没有项可淘汰
            }
            
            let current_size = *self.current_size.read().map_err(|_| Error::locks_poison("内存缓存大小锁被污染"))?;
            current_size / data.len()
        };
        
        // 计算需要淘汰的项数（加上一个余量）
        let items_to_evict = (required_space / avg_item_size).max(1) + 2;
        
        // 获取淘汰候选项
        let candidates = {
            let strategy = self.eviction_strategy.lock().map_err(|_| Error::locks_poison("内存缓存策略锁被污染"))?;
            strategy.get_eviction_candidates(items_to_evict)
        };
        
        // 淘汰候选项
        for key in candidates {
            if evicted_size >= required_space {
                break;
            }
            
            if let Ok(true) = self.delete(&key) {
                evicted_count += 1;
                
                // 获取项大小（已经在delete中减去）
                let data = self.data.read().map_err(|_| Error::locks_poison("内存缓存数据锁被污染"))?;
                if let Some(item) = data.get(&key) {
                    evicted_size += item.size;
                }
            }
        }
        
        // 更新指标
        if evicted_count > 0 {
            let mut metrics = self.metrics.write().map_err(|_| Error::locks_poison("内存缓存指标锁被污染"))?;
            metrics.record_eviction(evicted_count, evicted_size);
        }
        
        Ok(evicted_count)
    }
    
    /// 根据数量淘汰缓存项
    fn evict_by_count(&self, count: usize) -> Result<usize> {
        let mut evicted_count = 0;
        let mut evicted_size = 0;
        
        // 获取淘汰候选项
        let candidates = {
            let strategy = self.eviction_strategy.lock().map_err(|_| Error::locks_poison("内存缓存策略锁被污染"))?;
            strategy.get_eviction_candidates(count)
        };
        
        // 淘汰候选项
        for key in candidates {
            if evicted_count >= count {
                break;
            }
            
            if let Ok(true) = self.delete(&key) {
                evicted_count += 1;
                
                // 获取项大小（已经在delete中减去）
                let data = self.data.read().map_err(|_| Error::locks_poison("内存缓存数据锁被污染"))?;
                if let Some(item) = data.get(&key) {
                    evicted_size += item.size;
                }
            }
        }
        
        // 更新指标
        if evicted_count > 0 {
            let mut metrics = self.metrics.write().map_err(|_| Error::locks_poison("内存缓存指标锁被污染"))?;
            metrics.record_eviction(evicted_count, evicted_size);
        }
        
        Ok(evicted_count)
    }
    
    /// 强制淘汰指定数量的缓存项
    pub fn evict_entries(&self, count: usize) -> Result<usize> {
        if count == 0 {
            return Ok(0);
        }
        
        self.evict_by_count(count)
    }

    /// 获取缓存项数量
    pub fn len(&self) -> usize {
        self.data.read().map(|data| data.len()).unwrap_or(0)
    }

    /// 检查缓存是否为空
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 获取缓存配置
    pub fn max_size(&self) -> usize {
        self.max_size_bytes
    }

    /// 获取当前使用的内存
    pub fn current_size(&self) -> usize {
        self.current_size.read().map(|size| *size).unwrap_or(0)
    }

    /// 获取缓存统计信息
    pub fn get_metrics(&self) -> Result<CacheMetrics> {
        let metrics = self.metrics.read().map_err(|_| Error::locks_poison("内存缓存指标锁被污染"))?;
        Ok(metrics.clone())
    }
    
    /// 获取内存配置
    pub fn get_memory_config(&self) -> &MemoryConfig {
        &self.memory_config
    }
    
    /// 获取内存统计信息
    pub fn get_memory_stats(&self) -> Result<crate::memory::MemoryStats> {
        Ok(get_memory_stats())
    }
    
    /// 执行内存健康检查
    pub fn perform_memory_health_check(&self) -> Result<bool> {
        let now = Instant::now();
        let last_check = *self.last_health_check.read().map_err(|_| Error::locks_poison("健康检查时间锁被污染"))?;
        
        // 检查是否需要执行健康检查
        if now.duration_since(last_check) < self.health_check_interval {
            return Ok(true); // 健康检查间隔未到
        }
        
        // 执行健康检查
        let is_healthy = memory_health_check();
        
        // 更新上次检查时间
        if let Ok(mut last_check) = self.last_health_check.write() {
            *last_check = now;
        }
        
        Ok(is_healthy)
    }
    
    /// 释放指定大小的内存
    pub fn deallocate_memory(&self, size: usize) -> Result<()> {
        // 使用内存管理器的释放功能
        deallocate(size);
        Ok(())
    }
    
    /// 获取内存管理器引用
    pub fn get_memory_manager(&self) -> &Arc<MemoryManager> {
        &self.memory_manager
    }
    
    /// 更新内存配置
    pub fn update_memory_config(&mut self, new_config: MemoryConfig) -> Result<()> {
        self.memory_config = new_config;
        Ok(())
    }
    
    /// 获取内存池管理器
    pub fn get_memory_pool_manager(&self) -> &Arc<MemoryPoolManager> {
        &self.memory_pool_manager
    }
    
    /// 从内存池分配指定大小的内存
    pub fn allocate_from_pool(&self, size: usize) -> Result<Arc<ManagedPtr>> {
        let size_class = self.memory_pool_manager.get_size_class(size);
        let ptr = self.memory_pool_manager.allocate(size_class)?;
        Ok(ptr)
    }
    
    /// 将内存返回到内存池
    pub fn return_to_pool(&self, ptr: Arc<ManagedPtr>) -> Result<()> {
        self.memory_pool_manager.deallocate(ptr)?;
        Ok(())
    }
    
    /// 获取缓存优化器
    pub fn get_cache_optimizer(&self) -> &Arc<CacheOptimizer> {
        &self.cache_optimizer
    }
    
    /// 执行缓存优化
    pub fn optimize_cache(&self) -> Result<()> {
        self.cache_optimizer.optimize(self)?;
        Ok(())
    }
    
    /// 获取内存池统计信息
    pub fn get_pool_stats(&self) -> Result<crate::memory::pool::PoolStats> {
        Ok(self.memory_pool_manager.get_stats())
    }
    
    /// 创建类型化张量缓存值
    pub fn create_typed_tensor_value(&self, data: Vec<u8>, data_type: TensorDataType, shape: Vec<usize>) -> Result<CacheValue> {
        let size = data.len();
        let managed_ptr = Arc::new(allocate(size, ObjectType::Cache)?);
        
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                managed_ptr.ptr.as_ptr(),
                size
            );
        }
        
        Ok(CacheValue::TypedTensor {
            data: managed_ptr,
            data_type,
            shape,
        })
    }
    
    /// 获取张量数据类型信息
    pub fn get_tensor_data_type_info(&self, data_type: &TensorDataType) -> String {
        match data_type {
            TensorDataType::Float32 => "f32".to_string(),
            TensorDataType::Float64 => "f64".to_string(),
            TensorDataType::Int32 => "i32".to_string(),
            TensorDataType::Int64 => "i64".to_string(),
            TensorDataType::UInt8 => "u8".to_string(),
            TensorDataType::Bool => "bool".to_string(),
        }
    }
} 