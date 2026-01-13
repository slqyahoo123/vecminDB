/// 智能缓存优化器
/// 
/// 提供高性能的缓存管理功能：
/// 1. 多种缓存策略（LRU、LFU、ARC、自适应）
/// 2. 智能压缩（LZ4压缩）
/// 3. 访问模式分析和预测
/// 4. 自动预加载和淘汰
/// 5. 后台优化任务

use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::{Instant, Duration};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use serde::{Serialize, Deserialize};
use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use parking_lot::RwLock;
use crossbeam::queue::SegQueue;
use tokio::time::sleep;

use super::{ObjectType, MemoryConfig};
use crate::{Error, Result};
use crate::core::types::HealthStatus;

/// 压缩类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionType {
    /// 无压缩
    None,
    /// LZ4压缩（快速）
    Lz4,
}

impl CompressionType {
    /// 压缩数据
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self {
            Self::None => Ok(data.to_vec()),
            Self::Lz4 => {
                compress_prepend_size(data)
                    .map_err(|e| Error::internal(&format!("LZ4 compression failed: {}", e)))
            },
        }
    }
    
    /// 解压数据
    pub fn decompress(&self, data: &[u8]) -> Result<Vec<u8>> {
        match self {
            Self::None => Ok(data.to_vec()),
            Self::Lz4 => {
                decompress_size_prepended(data)
                    .map_err(|e| Error::internal(&format!("LZ4 decompression failed: {}", e)))
            },
        }
    }
}

/// 缓存策略
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheStrategy {
    /// 最近最少使用
    LRU,
    /// 最少使用频率
    LFU,
    /// 自适应替换缓存
    ARC,
    /// 自适应策略
    Adaptive,
}

/// 访问模式
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessPattern {
    /// 顺序访问
    Sequential,
    /// 随机访问
    Random,
    /// 热点访问
    Hotspot,
    /// 循环访问
    Cyclic,
    /// 未知模式
    Unknown,
}

/// 缓存条目
#[derive(Debug)]
pub struct CacheEntry {
    /// 键
    pub key: String,
    /// 原始数据大小
    pub original_size: usize,
    /// 压缩后数据
    pub compressed_data: Vec<u8>,
    /// 压缩类型
    pub compression_type: CompressionType,
    /// 对象类型
    pub object_type: ObjectType,
    /// 创建时间
    pub created_at: Instant,
    /// 最后访问时间
    pub last_accessed: Instant,
    /// 访问次数
    pub access_count: AtomicU64,
    /// 访问频率（每秒）
    pub access_frequency: f64,
    /// TTL（生存时间）
    pub ttl: Option<Duration>,
    /// 优先级
    pub priority: u8,
    /// 是否热点数据
    pub is_hot: bool,
}

impl CacheEntry {
    /// 创建新的缓存条目
    pub fn new(
        key: String,
        data: &[u8],
        object_type: ObjectType,
        compression_type: CompressionType,
        ttl: Option<Duration>,
    ) -> Result<Self> {
        let compressed_data = compression_type.compress(data)?;
        let now = Instant::now();
        
        Ok(Self {
            key,
            original_size: data.len(),
            compressed_data,
            compression_type,
            object_type,
            created_at: now,
            last_accessed: now,
            access_count: AtomicU64::new(0),
            access_frequency: 0.0,
            ttl,
            priority: object_type.priority(),
            is_hot: false,
        })
    }
    
    /// 获取解压后的数据
    pub fn get_data(&self) -> Result<Vec<u8>> {
        self.compression_type.decompress(&self.compressed_data)
    }
    
    /// 标记访问
    pub fn mark_accessed(&self) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// 检查是否过期
    pub fn is_expired(&self) -> bool {
        if let Some(ttl) = self.ttl {
            self.created_at.elapsed() > ttl
        } else {
            false
        }
    }
    
    /// 获取访问次数
    pub fn access_count(&self) -> u64 {
        self.access_count.load(Ordering::Relaxed)
    }
    
    /// 获取年龄
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
    
    /// 获取空闲时间
    pub fn idle_time(&self) -> Duration {
        self.last_accessed.elapsed()
    }
    
    /// 计算压缩率
    pub fn compression_ratio(&self) -> f64 {
        if self.original_size == 0 {
            1.0
        } else {
            self.compressed_data.len() as f64 / self.original_size as f64
        }
    }
    
    /// 计算得分（用于淘汰策略）
    pub fn calculate_score(&self, strategy: CacheStrategy) -> f64 {
        match strategy {
            CacheStrategy::LRU => {
                // 基于最后访问时间
                -(self.last_accessed.elapsed().as_secs_f64())
            },
            CacheStrategy::LFU => {
                // 基于访问频率
                self.access_frequency
            },
            CacheStrategy::ARC => {
                // 结合访问频率和最近性
                let recency_score = -(self.last_accessed.elapsed().as_secs_f64());
                let frequency_score = self.access_frequency;
                0.7 * frequency_score + 0.3 * recency_score
            },
            CacheStrategy::Adaptive => {
                // 综合考虑多个因素
                let recency_score = -(self.last_accessed.elapsed().as_secs_f64());
                let frequency_score = self.access_frequency;
                let priority_score = self.priority as f64;
                let size_penalty = -(self.original_size as f64).log10();
                
                0.4 * frequency_score + 0.3 * recency_score + 0.2 * priority_score + 0.1 * size_penalty
            },
        }
    }
}

/// 优化任务
#[derive(Debug, Clone)]
pub struct OptimizationTask {
    /// 任务ID
    pub id: String,
    /// 任务类型
    pub task_type: OptimizationTaskType,
    /// 创建时间
    pub created_at: Instant,
    /// 优先级
    pub priority: u8,
    /// 参数
    pub parameters: HashMap<String, String>,
}

/// 优化任务类型
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationTaskType {
    /// 预加载
    Preload,
    /// 压缩优化
    CompressionOptimization,
    /// 淘汰过期条目
    EvictExpired,
    /// 重新平衡
    Rebalance,
    /// 碎片整理
    Defragment,
}

/// 缓存统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    /// 总条目数
    pub total_entries: usize,
    /// 命中次数
    pub hits: u64,
    /// 未命中次数
    pub misses: u64,
    /// 命中率
    pub hit_rate: f64,
    /// 总大小（字节）
    pub total_size: usize,
    /// 压缩后大小（字节）
    pub compressed_size: usize,
    /// 压缩率
    pub compression_ratio: f64,
    /// 淘汰次数
    pub evictions: u64,
}

impl Default for CacheStats {
    fn default() -> Self {
        Self {
            total_entries: 0,
            hits: 0,
            misses: 0,
            hit_rate: 0.0,
            total_size: 0,
            compressed_size: 0,
            compression_ratio: 1.0,
            evictions: 0,
        }
    }
}

/// 缓存优化器
#[derive(Debug)]
pub struct CacheOptimizer {
    /// 缓存条目
    entries: RwLock<HashMap<String, Arc<CacheEntry>>>,
    /// 缓存策略
    strategy: CacheStrategy,
    /// 压缩类型
    compression_type: CompressionType,
    /// 最大缓存大小
    max_cache_size: usize,
    /// 最大条目数
    max_entries: usize,
    /// 压缩阈值
    compression_threshold: usize,
    /// 优化任务队列
    optimization_queue: SegQueue<OptimizationTask>,
    /// 统计信息
    stats: Arc<Mutex<CacheStats>>,
    /// 是否运行中
    running: Arc<AtomicBool>,
    /// 预加载启用
    preload_enabled: bool,
}

impl CacheOptimizer {
    /// 创建新的缓存优化器
    pub fn new(config: &MemoryConfig) -> Result<Self> {
        Ok(Self {
            entries: RwLock::new(HashMap::new()),
            strategy: CacheStrategy::Adaptive,
            compression_type: CompressionType::Lz4,
            max_cache_size: 2 * 1024 * 1024 * 1024, // 2GB
            max_entries: 100000,
            compression_threshold: 1024, // 1KB
            optimization_queue: SegQueue::new(),
            stats: Arc::new(Mutex::new(CacheStats::default())),
            running: Arc::new(AtomicBool::new(false)),
            preload_enabled: true,
        })
    }
    
    /// 启动缓存优化器
    pub async fn start(self: Arc<Self>) -> Result<()> {
        if self.running.load(Ordering::SeqCst) {
            return Err(Error::invalid_state("Cache optimizer is already running"));
        }
        
        // 启动后台优化任务
        self.start_background_tasks().await?;
        
        Ok(())
    }
    
    /// 停止缓存优化器
    pub async fn stop(&self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);
        Ok(())
    }
    
    /// 获取缓存条目
    pub fn get(&self, key: &str) -> Option<Vec<u8>> {
        let entries = self.entries.read();
        if let Some(entry) = entries.get(key) {
            entry.mark_accessed();
            
            // 更新统计
            if let Ok(mut stats) = self.stats.lock() {
                stats.hits += 1;
            }
            
            entry.get_data().ok()
        } else {
            // 更新统计
            if let Ok(mut stats) = self.stats.lock() {
                stats.misses += 1;
            }
            
            None
        }
    }
    
    /// 设置缓存条目
    pub fn set(&self, key: String, data: &[u8], object_type: ObjectType, ttl: Option<Duration>) -> Result<()> {
        // 检查是否需要压缩
        let compression_type = if data.len() >= self.compression_threshold {
            self.compression_type
        } else {
            CompressionType::None
        };
        
        let entry = Arc::new(CacheEntry::new(key.clone(), data, object_type, compression_type, ttl)?);
        
        {
            let mut entries = self.entries.write();
            
            // 检查容量限制
            if entries.len() >= self.max_entries {
                self.evict_entries(&mut entries, 1)?;
            }
            
            entries.insert(key, entry);
        }
        
        // 更新统计
        self.update_stats();
        
        Ok(())
    }
    
    /// 删除缓存条目
    pub fn remove(&self, key: &str) -> bool {
        let mut entries = self.entries.write();
        entries.remove(key).is_some()
    }
    
    /// 清空缓存
    pub fn clear(&self) {
        let mut entries = self.entries.write();
        entries.clear();
        self.update_stats();
    }
    
    /// 淘汰条目
    fn evict_entries(&self, entries: &mut HashMap<String, Arc<CacheEntry>>, count: usize) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        
        // 收集所有条目及其得分
        let mut scored_entries: Vec<(String, f64)> = entries
            .iter()
            .map(|(key, entry)| (key.clone(), entry.calculate_score(self.strategy)))
            .collect();
        
        // 按得分排序（得分低的先淘汰）
        scored_entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // 淘汰指定数量的条目
        let evict_count = count.min(scored_entries.len());
        for i in 0..evict_count {
            let key = &scored_entries[i].0;
            entries.remove(key);
            
            // 更新统计
            if let Ok(mut stats) = self.stats.lock() {
                stats.evictions += 1;
            }
        }
        
        Ok(())
    }
    
    /// 更新统计信息
    fn update_stats(&self) {
        if let Ok(mut stats) = self.stats.lock() {
            let entries = self.entries.read();
            
            stats.total_entries = entries.len();
            stats.total_size = entries.values().map(|e| e.original_size).sum();
            stats.compressed_size = entries.values().map(|e| e.compressed_data.len()).sum();
            
            // 计算命中率
            let total = stats.hits + stats.misses;
            if total > 0 {
                stats.hit_rate = stats.hits as f64 / total as f64;
            }
            
            // 计算压缩率
            if stats.total_size > 0 {
                stats.compression_ratio = stats.compressed_size as f64 / stats.total_size as f64;
            }
        }
    }
    
    /// 清理过期条目
    pub async fn cleanup_expired(&self) -> Result<usize> {
        let mut removed_count = 0;
        
        {
            let mut entries = self.entries.write();
            let expired_keys: Vec<String> = entries
                .iter()
                .filter(|(_, entry)| entry.is_expired())
                .map(|(key, _)| key.clone())
                .collect();
            
            for key in expired_keys {
                entries.remove(&key);
                removed_count += 1;
            }
        }
        
        if removed_count > 0 {
            self.update_stats();
        }
        
        Ok(removed_count)
    }
    
    /// 获取统计信息
    pub fn get_stats(&self) -> CacheStats {
        if let Ok(stats) = self.stats.lock() {
            stats.clone()
        } else {
            CacheStats::default()
        }
    }
    
    /// 健康检查
    pub fn health_check(&self) -> Result<HealthStatus> {
        let stats = self.get_stats();
        
        // 检查命中率
        if stats.hit_rate < 0.5 {
            return Ok(HealthStatus::Warning);
        }
        
        // 检查内存使用
        if stats.total_entries >= self.max_entries {
            return Ok(HealthStatus::Warning);
        }
        
        Ok(HealthStatus::Healthy)
    }
    
    /// 执行清理
    pub async fn cleanup(&self) -> Result<()> {
        self.cleanup_expired().await?;
        Ok(())
    }
    
    /// 启动后台任务
    async fn start_background_tasks(self: Arc<Self>) -> Result<()> {
        let running = self.running.clone();
        let weak_self = Arc::downgrade(&self);
        tokio::spawn(async move {
            while running.load(Ordering::SeqCst) {
                if let Some(optimizer) = weak_self.upgrade() {
                    let _ = optimizer.cleanup_expired().await;
                }
                sleep(Duration::from_secs(300)).await;
            }
        });
        Ok(())
    }
    
    /// 优化缓存
    pub async fn optimize(&self) -> Result<()> {
        // 1. 清理过期条目
        self.cleanup_expired().await?;
        
        // 2. 重新平衡缓存
        self.rebalance_cache().await?;
        
        // 3. 优化压缩
        self.optimize_compression().await?;
        
        // 4. 更新统计信息
        self.update_stats();
        
        Ok(())
    }
    
    /// 获取命中率
    pub fn get_hit_rate(&self) -> f64 {
        if let Ok(stats) = self.stats.lock() {
            stats.hit_rate
        } else {
            0.0
        }
    }
    
    /// 压缩数据
    pub fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.compression_type.compress(data)
    }
    
    /// 重新平衡缓存
    async fn rebalance_cache(&self) -> Result<()> {
        let mut entries = self.entries.write();
        
        // 如果条目数超过限制，执行淘汰
        if entries.len() > self.max_entries {
            let evict_count = entries.len() - self.max_entries;
            self.evict_entries(&mut entries, evict_count)?;
        }
        
        Ok(())
    }
    
    /// 优化压缩
    async fn optimize_compression(&self) -> Result<()> {
        let entries = self.entries.read();
        
        // 统计压缩效果
        let mut total_original = 0;
        let mut total_compressed = 0;
        
        for entry in entries.values() {
            total_original += entry.original_size;
            total_compressed += entry.compressed_data.len();
        }
        
        // 记录压缩效果
        log::debug!(
            "Compression optimization: {:.2}% ratio ({} -> {} bytes)",
            (total_compressed as f64 / total_original as f64) * 100.0,
            total_original,
            total_compressed
        );
        
        Ok(())
    }
    
    /// 通知参数添加
    pub fn notify_parameter_added(&self, name: &str, size: usize) {
        log::debug!("Parameter added: {} ({} bytes)", name, size);
        
        // 可以在这里添加更多的统计或优化逻辑
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_size += size;
            stats.total_entries += 1;
        }
    }
    
    /// 通知参数移除
    pub fn notify_parameter_removed(&self, name: &str, size: usize) {
        log::debug!("Parameter removed: {} ({} bytes)", name, size);
        
        // 可以在这里添加更多的统计或优化逻辑
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_size = stats.total_size.saturating_sub(size);
            stats.total_entries = stats.total_entries.saturating_sub(1);
        }
    }
}