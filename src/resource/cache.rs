use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use log::{debug, info, error};

use crate::error::{Error, Result};
use crate::cache_common::{CacheItem, EvictionPolicy};
use crate::resource::manager::{CacheLevel, MultiLevelCacheConfig};

/// 缓存访问统计
#[derive(Debug, Clone)]
struct AccessStatistics {
    /// 访问次数
    pub access_count: u64,
    /// 最后访问时间
    pub last_access: Instant,
    /// 访问间隔统计(毫秒)
    pub access_intervals: Vec<u64>,
    /// 热度得分(0.0-1.0)
    pub hotness_score: f64,
    /// 访问趋势(递增=1.0, 稳定=0.0, 递减=-1.0)
    pub access_trend: f64,
    /// 缓存级别
    pub current_level: CacheLevel,
}

impl AccessStatistics {
    /// 创建新的访问统计
    pub fn new(level: CacheLevel) -> Self {
        Self {
            access_count: 0,
            last_access: Instant::now(),
            access_intervals: Vec::with_capacity(10),
            hotness_score: 0.0,
            access_trend: 0.0,
            current_level: level,
        }
    }
    
    /// 记录访问
    pub fn record_access(&mut self) {
        let now = Instant::now();
        let interval = now.duration_since(self.last_access).as_millis() as u64;
        
        // 记录访问间隔
        if self.access_count > 0 {
            if self.access_intervals.len() >= 10 {
                self.access_intervals.remove(0);
            }
            self.access_intervals.push(interval);
        }
        
        // 更新访问计数和时间
        self.access_count += 1;
        self.last_access = now;
        
        // 更新热度得分
        self.update_hotness_score();
        
        // 更新访问趋势
        self.update_access_trend();
    }
    
    /// 更新热度得分
    fn update_hotness_score(&mut self) {
        // 基于访问频率和间隔计算热度
        // 1. 访问次数越多，热度越高
        // 2. 访问间隔越短，热度越高
        
        // 访问次数的影响 (0.0-0.7)
        let count_factor = (self.access_count as f64).min(100.0) / 100.0 * 0.7;
        
        // 访问间隔的影响 (0.0-0.3)
        let interval_factor = if !self.access_intervals.is_empty() {
            let avg_interval = self.access_intervals.iter().sum::<u64>() as f64 / self.access_intervals.len() as f64;
            // 间隔越短，得分越高 (最高0.3)
            (1.0 - (avg_interval.min(60000.0) / 60000.0)) * 0.3
        } else {
            0.0
        };
        
        // 总热度得分
        self.hotness_score = count_factor + interval_factor;
    }
    
    /// 更新访问趋势
    fn update_access_trend(&mut self) {
        if self.access_intervals.len() < 3 {
            self.access_trend = 0.0; // 数据不足，趋势不明显
            return;
        }
        
        // 计算最近3次访问间隔的变化趋势
        let recent = &self.access_intervals[self.access_intervals.len() - 3..];
        
        // 如果间隔在减小，表示访问频率增加
        if recent[2] < recent[1] && recent[1] < recent[0] {
            self.access_trend = 1.0;
        }
        // 如果间隔在增大，表示访问频率减少
        else if recent[2] > recent[1] && recent[1] > recent[0] {
            self.access_trend = -1.0;
        }
        // 访问频率相对稳定
        else {
            self.access_trend = 0.0;
        }
    }
    
    /// 判断是否应该升级到更高级缓存
    pub fn should_promote(&self, promotion_threshold: u64) -> bool {
        self.access_count >= promotion_threshold && 
        self.hotness_score > 0.7 && 
        self.access_trend >= 0.0
    }
    
    /// 判断是否应该降级到更低级缓存
    pub fn should_demote(&self, demotion_threshold: u64) -> bool {
        let now = Instant::now();
        let idle_time = now.duration_since(self.last_access).as_secs();
        
        idle_time > demotion_threshold || 
        (self.hotness_score < 0.3 && self.access_trend < 0.0)
    }
}

/// 多级缓存项
struct MultiLevelCacheEntry {
    /// 缓存内容
    item: CacheItem,
    /// 访问统计
    stats: AccessStatistics,
}

/// 多级缓存系统
pub struct MultiLevelCache {
    /// 配置
    config: MultiLevelCacheConfig,
    /// 各级缓存
    caches: HashMap<CacheLevel, RwLock<HashMap<String, MultiLevelCacheEntry>>>,
    /// 各级缓存大小(字节)
    sizes: RwLock<HashMap<CacheLevel, usize>>,
    /// 访问统计
    stats: RwLock<HashMap<String, AccessStatistics>>,
    /// 预取队列
    prefetch_queue: Mutex<VecDeque<String>>,
    /// 预取线程运行标志
    prefetch_running: Arc<Mutex<bool>>,
}

impl MultiLevelCache {
    /// 创建新的多级缓存
    pub fn new(config: MultiLevelCacheConfig) -> Self {
        let mut caches = HashMap::new();
        let mut sizes = HashMap::new();
        
        // 初始化各级缓存
        caches.insert(CacheLevel::L1, RwLock::new(HashMap::new()));
        caches.insert(CacheLevel::L2, RwLock::new(HashMap::new()));
        caches.insert(CacheLevel::L3, RwLock::new(HashMap::new()));
        
        // 初始化各级大小
        sizes.insert(CacheLevel::L1, 0);
        sizes.insert(CacheLevel::L2, 0);
        sizes.insert(CacheLevel::L3, 0);
        
        Self {
            config,
            caches,
            sizes: RwLock::new(sizes),
            stats: RwLock::new(HashMap::new()),
            prefetch_queue: Mutex::new(VecDeque::new()),
            prefetch_running: Arc::new(Mutex::new(false)),
        }
    }
    
    /// 获取缓存项
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        // 按照缓存级别从高到低查找
        for level in &[CacheLevel::L1, CacheLevel::L2, CacheLevel::L3] {
            let cache = self.caches.get(level).ok_or_else(|| Error::Internal("缓存级别未初始化".into()))?;
            let cache_read = cache.read().unwrap();
            
            if let Some(entry) = cache_read.get(key) {
                // 检查是否过期
                if entry.item.is_expired() {
                    drop(cache_read);
                    self.remove(key, *level)?;
                    continue;
                }
                
                // 获取数据
                let value = entry.item.value.clone();
                
                // 更新访问统计
                drop(cache_read);
                self.record_access(key, *level)?;
                
                // 考虑升级缓存级别
                self.consider_promotion(key)?;
                
                return Ok(Some(value));
            }
        }
        
        // 未找到缓存项
        Ok(None)
    }
    
    /// 设置缓存项
    pub fn put(&self, key: &str, value: Vec<u8>, ttl: Option<Duration>) -> Result<()> {
        // 创建缓存项
        let item = CacheItem::new(key, value, ttl);
        let size = item.size_bytes;
        
        // 默认放入L3缓存
        let level = CacheLevel::L3;
        let entry = MultiLevelCacheEntry {
            item,
            stats: AccessStatistics::new(level),
        };
        
        // 检查并确保有足够空间
        self.ensure_space(level, size)?;
        
        // 添加到缓存
        let cache = self.caches.get(&level).ok_or_else(|| Error::Internal("缓存级别未初始化".into()))?;
        let mut cache_write = cache.write().unwrap();
        
        // 记录大小
        let mut sizes = self.sizes.write().unwrap();
        let current_size = sizes.get_mut(&level).unwrap();
        *current_size += size;
        
        // 添加条目
        cache_write.insert(key.to_string(), entry);
        
        // 添加访问统计
        drop(cache_write);
        drop(sizes);
        
        let mut stats = self.stats.write().unwrap();
        stats.insert(key.to_string(), AccessStatistics::new(level));
        
        Ok(())
    }
    
    /// 移除缓存项
    pub fn remove(&self, key: &str, level: CacheLevel) -> Result<()> {
        let cache = self.caches.get(&level).ok_or_else(|| Error::Internal("缓存级别未初始化".into()))?;
        let mut cache_write = cache.write().unwrap();
        
        if let Some(entry) = cache_write.remove(key) {
            // 更新大小
            let mut sizes = self.sizes.write().unwrap();
            let current_size = sizes.get_mut(&level).unwrap();
            *current_size = current_size.saturating_sub(entry.item.size_bytes);
            
            // 更新统计
            drop(cache_write);
            drop(sizes);
            
            let mut stats = self.stats.write().unwrap();
            stats.remove(key);
        }
        
        Ok(())
    }
    
    /// 清空缓存
    pub fn clear(&self) -> Result<()> {
        for level in &[CacheLevel::L1, CacheLevel::L2, CacheLevel::L3] {
            let cache = self.caches.get(level).ok_or_else(|| Error::Internal("缓存级别未初始化".into()))?;
            let mut cache_write = cache.write().unwrap();
            cache_write.clear();
            
            let mut sizes = self.sizes.write().unwrap();
            let current_size = sizes.get_mut(level).unwrap();
            *current_size = 0;
        }
        
        let mut stats = self.stats.write().unwrap();
        stats.clear();
        
        Ok(())
    }
    
    /// 记录访问
    fn record_access(&self, key: &str, level: CacheLevel) -> Result<()> {
        // 更新缓存项的访问信息
        let cache = self.caches.get(&level).ok_or_else(|| Error::Internal("缓存级别未初始化".into()))?;
        let mut cache_write = cache.write().unwrap();
        
        if let Some(entry) = cache_write.get_mut(key) {
            entry.item.mark_accessed();
            entry.stats.record_access();
        }
        
        // 更新全局访问统计
        drop(cache_write);
        
        let mut stats = self.stats.write().unwrap();
        if let Some(stat) = stats.get_mut(key) {
            stat.record_access();
        } else {
            let mut new_stat = AccessStatistics::new(level);
            new_stat.record_access();
            stats.insert(key.to_string(), new_stat);
        }
        
        Ok(())
    }
    
    /// 考虑是否将缓存项升级到更高级别
    fn consider_promotion(&self, key: &str) -> Result<()> {
        let stats = self.stats.read().unwrap();
        
        if let Some(stat) = stats.get(key) {
            // 检查是否应该升级
            if stat.should_promote(self.config.promotion_threshold) {
                // 确定当前级别和目标级别
                let current_level = stat.current_level;
                let target_level = match current_level {
                    CacheLevel::L3 => CacheLevel::L2,
                    CacheLevel::L2 => CacheLevel::L1,
                    CacheLevel::L1 => return Ok(()), // 已经是最高级别
                };
                
                // 执行升级
                drop(stats);
                self.promote_item(key, current_level, target_level)?;
            }
        }
        
        Ok(())
    }
    
    /// 考虑是否将缓存项降级到更低级别
    fn consider_demotion(&self) -> Result<()> {
        let stats = self.stats.read().unwrap();
        let threshold = self.config.demotion_threshold;
        
        // 收集需要降级的项
        let mut to_demote = Vec::new();
        
        for (key, stat) in stats.iter() {
            if stat.should_demote(threshold) {
                // 确定当前级别和目标级别
                let current_level = stat.current_level;
                let target_level = match current_level {
                    CacheLevel::L1 => CacheLevel::L2,
                    CacheLevel::L2 => CacheLevel::L3,
                    CacheLevel::L3 => continue, // 已经是最低级别
                };
                
                to_demote.push((key.clone(), current_level, target_level));
            }
        }
        
        // 执行降级
        drop(stats);
        
        for (key, current, target) in to_demote {
            self.demote_item(&key, current, target)?;
        }
        
        Ok(())
    }
    
    /// 升级缓存项
    fn promote_item(&self, key: &str, from_level: CacheLevel, to_level: CacheLevel) -> Result<()> {
        // 从源缓存获取项
        let from_cache = self.caches.get(&from_level).ok_or_else(|| Error::Internal("源缓存级别未初始化".into()))?;
        let mut from_write = from_cache.write().unwrap();
        
        if let Some(entry) = from_write.remove(key) {
            let size = entry.item.size_bytes;
            
            // 确保目标缓存有足够空间
            drop(from_write);
            self.ensure_space(to_level, size)?;
            
            // 添加到目标缓存
            let to_cache = self.caches.get(&to_level).ok_or_else(|| Error::Internal("目标缓存级别未初始化".into()))?;
            let mut to_write = to_cache.write().unwrap();
            
            // 更新大小
            let mut sizes = self.sizes.write().unwrap();
            
            // 减少源缓存大小
            let from_size = sizes.get_mut(&from_level).unwrap();
            *from_size = from_size.saturating_sub(size);
            
            // 增加目标缓存大小
            let to_size = sizes.get_mut(&to_level).unwrap();
            *to_size += size;
            
            // 更新条目级别
            let mut updated_entry = entry;
            updated_entry.stats.current_level = to_level;
            
            // 添加到目标缓存
            to_write.insert(key.to_string(), updated_entry);
            
            // 更新统计
            drop(to_write);
            drop(sizes);
            
            let mut stats = self.stats.write().unwrap();
            if let Some(stat) = stats.get_mut(key) {
                stat.current_level = to_level;
            }
            
            debug!("缓存项升级: {} 从 {:?} 到 {:?}", key, from_level, to_level);
        }
        
        Ok(())
    }
    
    /// 降级缓存项
    fn demote_item(&self, key: &str, from_level: CacheLevel, to_level: CacheLevel) -> Result<()> {
        // 从源缓存获取项
        let from_cache = self.caches.get(&from_level).ok_or_else(|| Error::Internal("源缓存级别未初始化".into()))?;
        let mut from_write = from_cache.write().unwrap();
        
        if let Some(entry) = from_write.remove(key) {
            let size = entry.item.size_bytes;
            
            // 确保目标缓存有足够空间
            drop(from_write);
            self.ensure_space(to_level, size)?;
            
            // 添加到目标缓存
            let to_cache = self.caches.get(&to_level).ok_or_else(|| Error::Internal("目标缓存级别未初始化".into()))?;
            let mut to_write = to_cache.write().unwrap();
            
            // 更新大小
            let mut sizes = self.sizes.write().unwrap();
            
            // 减少源缓存大小
            let from_size = sizes.get_mut(&from_level).unwrap();
            *from_size = from_size.saturating_sub(size);
            
            // 增加目标缓存大小
            let to_size = sizes.get_mut(&to_level).unwrap();
            *to_size += size;
            
            // 更新条目级别
            let mut updated_entry = entry;
            updated_entry.stats.current_level = to_level;
            
            // 添加到目标缓存
            to_write.insert(key.to_string(), updated_entry);
            
            // 更新统计
            drop(to_write);
            drop(sizes);
            
            let mut stats = self.stats.write().unwrap();
            if let Some(stat) = stats.get_mut(key) {
                stat.current_level = to_level;
            }
            
            debug!("缓存项降级: {} 从 {:?} 到 {:?}", key, from_level, to_level);
        }
        
        Ok(())
    }
    
    /// 确保缓存有足够空间
    fn ensure_space(&self, level: CacheLevel, needed_space: usize) -> Result<()> {
        let mut sizes = self.sizes.write().unwrap();
        let current_size = *sizes.get(&level).unwrap();
        let max_size = *self.config.sizes.get(&level).unwrap_or(&0);
        
        // 如果空间足够，直接返回
        if current_size + needed_space <= max_size {
            return Ok(());
        }
        
        // 需要淘汰一些项目
        drop(sizes);
        
        let policy = *self.config.policies.get(&level).unwrap_or(&EvictionPolicy::LRU);
        self.evict(level, needed_space, policy)?;
        
        Ok(())
    }
    
    /// 淘汰缓存项
    fn evict(&self, level: CacheLevel, needed_space: usize, policy: EvictionPolicy) -> Result<()> {
        let cache = self.caches.get(&level).ok_or_else(|| Error::Internal("缓存级别未初始化".into()))?;
        let mut cache_write = cache.write().unwrap();
        
        // 当前缓存大小
        let mut sizes = self.sizes.write().unwrap();
        let current_size = *sizes.get(&level).unwrap();
        let max_size = *self.config.sizes.get(&level).unwrap_or(&0);
        
        // 需要释放的空间
        let to_free = (current_size + needed_space).saturating_sub(max_size);
        
        if to_free == 0 {
            return Ok(());
        }
        
        // 收集可能的淘汰候选项
        let mut candidates: Vec<_> = cache_write.iter().collect();
        
        // 根据策略排序
        match policy {
            EvictionPolicy::LRU => {
                // 按最后访问时间排序，最早的先淘汰
                candidates.sort_by(|(_, a), (_, b)| a.item.accessed_at.cmp(&b.item.accessed_at));
            },
            EvictionPolicy::MRU => {
                // 按最后访问时间排序，最近的先淘汰
                candidates.sort_by(|(_, a), (_, b)| b.item.accessed_at.cmp(&a.item.accessed_at));
            },
            EvictionPolicy::LFU => {
                // 按访问次数排序，最少的先淘汰
                candidates.sort_by(|(_, a), (_, b)| a.item.access_count.cmp(&b.item.access_count));
            },
            EvictionPolicy::FIFO => {
                // 按创建时间排序，最早的先淘汰
                candidates.sort_by(|(_, a), (_, b)| a.item.created_at.cmp(&b.item.created_at));
            },
        }
        
        // 执行淘汰
        let mut freed = 0;
        let mut to_remove = Vec::new();
        
        for (key, entry) in candidates {
            freed += entry.item.size_bytes;
            to_remove.push(key.clone());
            
            if freed >= to_free {
                break;
            }
        }
        
        // 实际移除
        let mut removed_size = 0;
        for key in to_remove {
            if let Some(entry) = cache_write.remove(&key) {
                removed_size += entry.item.size_bytes;
                
                // 更新统计
                let mut stats = self.stats.write().unwrap();
                stats.remove(&key);
                
                debug!("缓存项淘汰: {} 从 {:?} 级别", key, level);
            }
        }
        
        // 更新大小
        let current_size = sizes.get_mut(&level).unwrap();
        *current_size = current_size.saturating_sub(removed_size);
        
        Ok(())
    }
    
    /// 启动预取线程
    pub fn start_prefetch(&self) -> Result<()> {
        let mut running_guard = self.prefetch_running.lock().map_err(|_| Error::Internal("无法锁定预取运行标志".into()))?;
        if *running_guard {
            return Err(Error::Internal("预取线程已在运行".into()));
        }
        *running_guard = true;

        let prefetch_running = Arc::clone(&self.prefetch_running);
        let config = self.config.clone();
        let stats = Arc::clone(&self.stats);
        let caches = self.caches.clone();
        let prefetch_queue = Arc::clone(&self.prefetch_queue);

        std::thread::spawn(move || {
            info!("预取线程已启动");
            while *prefetch_running.lock().unwrap() {
                std::thread::sleep(Duration::from_secs(60)); // 使用固定的预取间隔
                
                if !config.prefetch_enabled {
                    continue;
                }
                
                let mut to_prefetch = Vec::new();
                {
                    let stats_read = stats.read().unwrap();
                    for (key, stat) in stats_read.iter() {
                        if stat.hotness_score >= config.prefetch_threshold && stat.access_trend > 0.0 {
                            to_prefetch.push(key.clone());
                        }
                    }
                }
                
                if !to_prefetch.is_empty() {
                    let mut queue = prefetch_queue.lock().unwrap();
                    for key in to_prefetch {
                        if !queue.contains(&key) {
                            queue.push_back(key);
                        }
                    }
                }
                
                let mut processed = 0;
                while processed < 5 { // Limit processing per cycle
                    let key = {
                        let mut queue = prefetch_queue.lock().unwrap();
                        queue.pop_front()
                    };
                    
                    if let Some(key) = key {
                        let current_level = {
                            let stats_read = stats.read().unwrap();
                            stats_read.get(&key).map(|s| s.current_level)
                        };
                        
                        if let Some(level) = current_level {
                            let target_level = match level {
                                CacheLevel::L3 => Some(CacheLevel::L2),
                                CacheLevel::L2 => Some(CacheLevel::L1),
                                CacheLevel::L1 => None,
                            };
                            
                            if let Some(target) = target_level {
                                let _ = Self::prefetch_item(&key, level, target, &caches);
                            }
                        }
                        processed += 1;
                    } else {
                        break;
                    }
                }
            }
            info!("预取线程已停止");
        });

        Ok(())
    }
    
    /// 预取缓存项
    fn prefetch_item(key: &str, from_level: CacheLevel, to_level: CacheLevel, caches: &HashMap<CacheLevel, RwLock<HashMap<String, MultiLevelCacheEntry>>>) -> Result<()> {
        // 从源缓存获取项
        if let Some(from_cache) = caches.get(&from_level) {
            let from_read = from_cache.read().unwrap();
            
            if let Some(entry) = from_read.get(key) {
                // 复制条目到更高级别
                if let Some(to_cache) = caches.get(&to_level) {
                    let mut to_write = to_cache.write().unwrap();
                    
                    // 只复制，不从源缓存移除
                    let mut new_entry = MultiLevelCacheEntry {
                        item: entry.item.clone(),
                        stats: entry.stats.clone(),
                    };
                    
                    // 更新级别
                    new_entry.stats.current_level = to_level;
                    
                    // 添加到目标缓存
                    to_write.insert(key.to_string(), new_entry);
                    
                    debug!("缓存项预取: {} 从 {:?} 到 {:?}", key, from_level, to_level);
                }
            }
        }
        
        Ok(())
    }
    
    /// 停止预取
    pub fn stop_prefetch(&self) -> Result<()> {
        let mut running_guard = self.prefetch_running.lock().map_err(|_| Error::Internal("无法锁定预取运行标志".into()))?;
        if !*running_guard {
            return Err(Error::Internal("预取线程未在运行".into()));
        }
        *running_guard = false;
        Ok(())
    }
    
    /// 定期维护
    pub fn maintenance(&self) -> Result<()> {
        // 检查过期项
        self.check_expirations()?;
        
        // 考虑降级
        self.consider_demotion()?;
        
        Ok(())
    }
    
    /// 检查过期项
    fn check_expirations(&self) -> Result<()> {
        for level in &[CacheLevel::L1, CacheLevel::L2, CacheLevel::L3] {
            let cache = self.caches.get(level).ok_or_else(|| Error::Internal("缓存级别未初始化".into()))?;
            let cache_read = cache.read().unwrap();
            
            // 收集过期项
            let mut expired = Vec::new();
            for (key, entry) in cache_read.iter() {
                if entry.item.is_expired() {
                    expired.push(key.clone());
                }
            }
            
            // 移除过期项
            drop(cache_read);
            for key in expired {
                self.remove(&key, *level)?;
            }
        }
        
        Ok(())
    }
    
    /// 获取缓存统计信息
    pub fn get_statistics(&self) -> Result<HashMap<CacheLevel, (usize, usize, usize)>> {
        let mut result = HashMap::new();
        let sizes = self.sizes.read().unwrap();
        
        for level in &[CacheLevel::L1, CacheLevel::L2, CacheLevel::L3] {
            let cache = self.caches.get(level).ok_or_else(|| Error::Internal("缓存级别未初始化".into()))?;
            let cache_read = cache.read().unwrap();
            
            let count = cache_read.len();
            let size = *sizes.get(level).unwrap_or(&0);
            let capacity = *self.config.sizes.get(level).unwrap_or(&0);
            
            result.insert(*level, (count, size, capacity));
        }
        
        Ok(result)
    }
}

/// 缓存管理器
pub struct CacheManager {
    /// 多级缓存
    cache: Arc<MultiLevelCache>,
    /// 维护任务运行标志
    maintenance_running: Arc<Mutex<bool>>,
}

impl CacheManager {
    /// 创建新的缓存管理器
    pub fn new(config: MultiLevelCacheConfig) -> Self {
        let cache = Arc::new(MultiLevelCache::new(config));
        
        Self {
            cache,
            maintenance_running: Arc::new(Mutex::new(false)),
        }
    }
    
    /// 启动缓存管理器
    pub fn start(&self) -> Result<()> {
        // 启动预取
        self.cache.start_prefetch()?;
        
        // 启动维护任务
        let cache = self.cache.clone();
        let maintenance_running = self.maintenance_running.clone();
        
        // 设置运行标志
        {
            let mut running = maintenance_running.lock().unwrap();
            *running = true;
        }
        
        // 启动维护线程
        std::thread::spawn(move || {
            // 维护循环
            while {
                let running = maintenance_running.lock().unwrap();
                *running
            } {
                // 休眠一段时间
                std::thread::sleep(Duration::from_secs(60));
                
                // 执行维护
                if let Err(e) = cache.maintenance() {
                    error!("缓存维护错误: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// 停止缓存管理器
    pub fn stop(&self) -> Result<()> {
        // 停止预取
        self.cache.stop_prefetch()?;
        
        // 停止维护任务
        {
            let mut running = self.maintenance_running.lock().unwrap();
            *running = false;
        }
        
        Ok(())
    }
    
    /// 获取缓存项
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        self.cache.get(key)
    }
    
    /// 设置缓存项
    pub fn put(&self, key: &str, value: Vec<u8>, ttl: Option<Duration>) -> Result<()> {
        self.cache.put(key, value, ttl)
    }
    
    /// 移除缓存项
    pub fn remove(&self, key: &str) -> Result<()> {
        // 尝试从所有级别移除
        for level in &[CacheLevel::L1, CacheLevel::L2, CacheLevel::L3] {
            self.cache.remove(key, *level)?;
        }
        
        Ok(())
    }
    
    /// 清空缓存
    pub fn clear(&self) -> Result<()> {
        self.cache.clear()
    }
    
    /// 获取缓存统计信息
    pub fn get_statistics(&self) -> Result<HashMap<CacheLevel, (usize, usize, usize)>> {
        self.cache.get_statistics()
    }
}

/// 资源缓存（CacheManager的别名）
pub type ResourceCache = CacheManager;

/// 缓存配置（MultiLevelCacheConfig的别名）
pub type CacheConfig = MultiLevelCacheConfig; 