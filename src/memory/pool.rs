/// 高性能内存池管理器
/// 
/// 提供分层内存池管理，根据对象类型和大小进行分类管理：
/// 1. 小对象池（<1KB）- 快速分配
/// 2. 中等对象池（1KB-1MB）- 平衡性能和内存使用
/// 3. 大对象池（>1MB）- 分片管理
/// 4. 类型特化池 - 针对Tensor、参数等优化

use std::sync::{Arc, Mutex};
use std::collections::{HashMap, BTreeMap};
use std::alloc::{Layout, alloc, dealloc};
use std::ptr::NonNull;
use std::time::{Instant, Duration};
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use parking_lot::RwLock;
use crossbeam::queue::SegQueue;
use serde::{Serialize, Deserialize};

use super::{ObjectType, ManagedPtr, MemoryConfig, HealthStatus};
use crate::{Error, Result};

/// 内存块大小类别
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SizeClass {
    /// 小对象 (0-1KB)
    Small(usize),
    /// 中等对象 (1KB-64KB) 
    Medium(usize),
    /// 大对象 (64KB-1MB)
    Large(usize),
    /// 超大对象 (>1MB)
    XLarge(usize),
}

impl SizeClass {
    /// 根据大小确定类别
    pub fn from_size(size: usize) -> Self {
        if size <= 1024 {
            Self::Small(Self::round_up_small(size))
        } else if size <= 64 * 1024 {
            Self::Medium(Self::round_up_medium(size))
        } else if size <= 1024 * 1024 {
            Self::Large(Self::round_up_large(size))
        } else {
            Self::XLarge(size)
        }
    }
    
    /// 获取实际大小
    pub fn actual_size(&self) -> usize {
        match self {
            Self::Small(size) | Self::Medium(size) | Self::Large(size) | Self::XLarge(size) => *size,
        }
    }
    
    /// 小对象大小对齐 (8字节对齐)
    fn round_up_small(size: usize) -> usize {
        (size + 7) & !7
    }
    
    /// 中等对象大小对齐 (64字节对齐)
    fn round_up_medium(size: usize) -> usize {
        (size + 63) & !63
    }
    
    /// 大对象大小对齐 (4KB对齐)
    fn round_up_large(size: usize) -> usize {
        (size + 4095) & !4095
    }
}

/// 内存块
pub struct MemoryBlock {
    /// 内存指针
    pub ptr: NonNull<u8>,
    /// 块大小
    pub size: usize,
    /// 分配时间
    pub allocated_at: Instant,
    /// 最后使用时间
    pub last_used: Instant,
    /// 使用次数
    pub use_count: u64,
    /// 对象类型
    pub object_type: ObjectType,
    /// 内存布局
    pub layout: Layout,
}

// 手动实现Send和Sync，因为NonNull<u8>默认不是Send+Sync
// 但在我们的使用场景中，内存块的所有权是明确的，可以安全地在线程间传递
unsafe impl Send for MemoryBlock {}
unsafe impl Sync for MemoryBlock {}

impl MemoryBlock {
    /// 创建新的内存块
    pub fn new(size: usize, object_type: ObjectType) -> Result<Self> {
        let layout = Layout::from_size_align(size, std::mem::align_of::<u8>())
            .map_err(|e| Error::invalid_input(&format!("Invalid layout: {}", e)))?;
        
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            return Err(Error::resource("Failed to allocate memory block"));
        }
        
        let now = Instant::now();
        Ok(Self {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            size,
            allocated_at: now,
            last_used: now,
            use_count: 0,
            object_type,
            layout,
        })
    }
    
    /// 标记使用
    pub fn mark_used(&mut self) {
        self.last_used = Instant::now();
        self.use_count += 1;
    }
    
    /// 检查是否可以回收
    pub fn is_recyclable(&self, max_idle_time: Duration) -> bool {
        self.last_used.elapsed() > max_idle_time
    }
    
    /// 重置块状态
    pub fn reset(&mut self) {
        self.last_used = Instant::now();
        self.use_count = 0;
        // 清零内存内容（可选，用于安全）
        unsafe {
            std::ptr::write_bytes(self.ptr.as_ptr(), 0, self.size);
        }
    }
}

impl Drop for MemoryBlock {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

/// 单个内存池
pub struct MemoryPool {
    /// 池ID
    pub id: String,
    /// 大小类别
    pub size_class: SizeClass,
    /// 对象类型
    pub object_type: ObjectType,
    /// 可用块队列
    available_blocks: SegQueue<MemoryBlock>,
    /// 使用中的块 - 使用usize地址作为key以确保线程安全
    used_blocks: RwLock<HashMap<usize, MemoryBlock>>,
    /// 池统计
    stats: PoolStats,
    /// 配置
    config: PoolConfig,
    /// 最大容量
    max_capacity: usize,
    /// 当前容量
    current_capacity: AtomicUsize,
}

impl MemoryPool {
    /// 创建新的内存池
    pub fn new(
        id: String,
        size_class: SizeClass,
        object_type: ObjectType,
        config: PoolConfig,
    ) -> Self {
        Self {
            id,
            size_class,
            object_type,
            available_blocks: SegQueue::new(),
            used_blocks: RwLock::new(HashMap::new()),
            stats: PoolStats::new(),
            max_capacity: config.max_capacity,
            current_capacity: AtomicUsize::new(0),
            config,
        }
    }
    
    /// 从池中分配内存
    pub fn allocate(&self) -> Result<ManagedPtr> {
        // 尝试从可用队列获取
        if let Some(mut block) = self.available_blocks.pop() {
            block.mark_used();
            let ptr = block.ptr;
            let size = block.size;
            let layout = block.layout;
            
            // 移动到使用中队列
            {
                let mut used = self.used_blocks.write();
                used.insert(ptr.as_ptr() as usize, block);
            }
            
            self.stats.increment_hit();
            
            return Ok(ManagedPtr::new(ptr, size, self.object_type, layout));
        }
        
        // 检查是否可以创建新块
        if self.current_capacity.load(Ordering::Relaxed) >= self.max_capacity {
            self.stats.increment_miss();
            return Err(Error::resource("Memory pool at capacity"));
        }
        
        // 创建新块
        let mut block = MemoryBlock::new(self.size_class.actual_size(), self.object_type)?;
        block.mark_used();
        
        let ptr = block.ptr;
        let size = block.size;
        let layout = block.layout;
        
        // 添加到使用中队列
        {
            let mut used = self.used_blocks.write();
            used.insert(ptr.as_ptr() as usize, block);
        }
        
        self.current_capacity.fetch_add(1, Ordering::Relaxed);
        self.stats.increment_allocation();
        
        Ok(ManagedPtr::new(ptr, size, self.object_type, layout))
    }
    
    /// 释放内存回池
    pub fn deallocate(&self, ptr: ManagedPtr) -> Result<()> {
        let raw_ptr = ptr.ptr.as_ptr() as usize;
        
        // 从使用中队列移除
        let block = {
            let mut used = self.used_blocks.write();
            used.remove(&raw_ptr)
        };
        
        if let Some(mut block) = block {
            // 检查是否应该回收到池中
            if self.should_recycle(&block) {
                block.reset();
                self.available_blocks.push(block);
                self.stats.increment_recycle();
            } else {
                // 直接释放
                drop(block);
                self.current_capacity.fetch_sub(1, Ordering::Relaxed);
                self.stats.increment_deallocation();
            }
            
            Ok(())
        } else {
            Err(Error::invalid_input("Invalid pointer for this pool"))
        }
    }
    
    /// 预热池（预分配一些块）
    pub fn warmup(&self, count: usize) -> Result<()> {
        let warmup_count = count.min(self.max_capacity);
        
        for _ in 0..warmup_count {
            if self.current_capacity.load(Ordering::Relaxed) >= self.max_capacity {
                break;
            }
            
            let block = MemoryBlock::new(self.size_class.actual_size(), self.object_type)?;
            self.available_blocks.push(block);
            self.current_capacity.fetch_add(1, Ordering::Relaxed);
        }
        
        Ok(())
    }
    
    /// 清理过期的块
    pub fn cleanup_expired(&self) -> Result<usize> {
        let max_idle_time = Duration::from_secs(self.config.max_idle_seconds);
        let mut cleaned_count = 0;
        
        // 收集过期的块
        let mut expired_blocks = Vec::new();
        while let Some(block) = self.available_blocks.pop() {
            if block.is_recyclable(max_idle_time) {
                expired_blocks.push(block);
            } else {
                // 如果不过期，放回队列
                self.available_blocks.push(block);
                break;
            }
        }
        
        // 释放过期的块
        cleaned_count = expired_blocks.len();
        drop(expired_blocks);
        
        self.current_capacity.fetch_sub(cleaned_count, Ordering::Relaxed);
        self.stats.add_cleaned(cleaned_count);
        
        Ok(cleaned_count)
    }
    
    /// 获取池统计信息
    pub fn get_stats(&self) -> PoolStats {
        let mut stats = self.stats.clone();
        stats.current_capacity = self.current_capacity.load(Ordering::Relaxed);
        stats.available_count = self.available_blocks.len();
        stats.used_count = self.used_blocks.read().len();
        stats
    }
    
    /// 自适应调整池大小
    pub fn adaptive_resize(&self) -> Result<()> {
        let stats = self.get_stats();
        let hit_rate = stats.hit_rate();
        
        // 如果命中率过低，预分配更多块
        if hit_rate < 0.8 && stats.current_capacity < self.max_capacity {
            let additional = ((self.max_capacity - stats.current_capacity) / 4).max(1);
            self.warmup(additional)?;
        }
        
        // 如果可用块过多且命中率高，清理一些块
        if hit_rate > 0.95 && stats.available_count > self.config.max_recycled * 2 {
            self.cleanup_expired()?;
        }
        
        Ok(())
    }
    
    /// 内存碎片整理
    pub async fn defragment(&self) -> Result<()> {
        // 获取所有可用块
        let mut blocks = Vec::new();
        while let Some(block) = self.available_blocks.pop() {
            blocks.push(block);
        }
        
        // 按大小排序以减少碎片
        blocks.sort_by_key(|block| block.size);
        
        // 重新添加到队列
        for block in blocks {
            self.available_blocks.push(block);
        }
        
        Ok(())
    }

    /// 检查是否应该回收
    fn should_recycle(&self, block: &MemoryBlock) -> bool {
        // 如果池已满，不回收
        if self.available_blocks.len() >= self.config.max_recycled {
            return false;
        }
        
        // 如果块使用次数太少，不回收（可能是一次性使用）
        if block.use_count < self.config.min_use_count {
            return false;
        }
        
        true
    }
}

/// 池配置
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PoolConfig {
    /// 最大容量
    pub max_capacity: usize,
    /// 最小容量
    pub min_capacity: usize,
    /// 最大回收数量
    pub max_recycled: usize,
    /// 最大空闲时间（秒）
    pub max_idle_seconds: u64,
    /// 最小使用次数
    pub min_use_count: u64,
    /// 预热百分比
    pub warmup_percentage: f64,
    /// 自适应调整间隔（秒）
    pub adaptive_resize_interval: u64,
    /// 碎片整理间隔（秒）
    pub defrag_interval: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_capacity: 1000,
            min_capacity: 10,
            max_recycled: 100,
            max_idle_seconds: 300, // 5分钟
            min_use_count: 2,
            warmup_percentage: 0.2, // 20%预热
            adaptive_resize_interval: 60, // 1分钟
            defrag_interval: 600, // 10分钟
        }
    }
}

/// 池统计信息
#[derive(Debug, Serialize, Deserialize)]
pub struct PoolStats {
    /// 命中次数
    pub hits: AtomicU64,
    /// 未命中次数
    pub misses: AtomicU64,
    /// 分配次数
    pub allocations: AtomicU64,
    /// 释放次数
    pub deallocations: AtomicU64,
    /// 回收次数
    pub recycles: AtomicU64,
    /// 清理次数
    pub cleaned: AtomicU64,
    /// 当前容量
    pub current_capacity: usize,
    /// 可用块数量
    pub available_count: usize,
    /// 使用中块数量
    pub used_count: usize,
    /// 创建时间
    #[serde(skip)]
    pub created_at: Instant,
    /// 平均分配时间（纳秒）
    pub avg_allocation_time_ns: AtomicU64,
    /// 平均释放时间（纳秒）
    pub avg_deallocation_time_ns: AtomicU64,
}

impl Clone for PoolStats {
    fn clone(&self) -> Self {
        Self {
            hits: AtomicU64::new(self.hits.load(Ordering::Relaxed)),
            misses: AtomicU64::new(self.misses.load(Ordering::Relaxed)),
            allocations: AtomicU64::new(self.allocations.load(Ordering::Relaxed)),
            deallocations: AtomicU64::new(self.deallocations.load(Ordering::Relaxed)),
            recycles: AtomicU64::new(self.recycles.load(Ordering::Relaxed)),
            cleaned: AtomicU64::new(self.cleaned.load(Ordering::Relaxed)),
            current_capacity: self.current_capacity,
            available_count: self.available_count,
            used_count: self.used_count,
            created_at: self.created_at,
            avg_allocation_time_ns: AtomicU64::new(self.avg_allocation_time_ns.load(Ordering::Relaxed)),
            avg_deallocation_time_ns: AtomicU64::new(self.avg_deallocation_time_ns.load(Ordering::Relaxed)),
        }
    }
}

impl PoolStats {
    pub fn new() -> Self {
        Self {
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            allocations: AtomicU64::new(0),
            deallocations: AtomicU64::new(0),
            recycles: AtomicU64::new(0),
            cleaned: AtomicU64::new(0),
            current_capacity: 0,
            available_count: 0,
            used_count: 0,
            created_at: Instant::now(),
            avg_allocation_time_ns: AtomicU64::new(0),
            avg_deallocation_time_ns: AtomicU64::new(0),
        }
    }
    
    pub fn increment_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn increment_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn increment_allocation(&self) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn increment_deallocation(&self) {
        self.deallocations.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn increment_recycle(&self) {
        self.recycles.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn add_cleaned(&self, count: usize) {
        self.cleaned.fetch_add(count as u64, Ordering::Relaxed);
    }
    
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let total = hits + self.misses.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
    
    /// 更新分配时间统计
    pub fn update_allocation_time(&self, time_ns: u64) {
        let current = self.avg_allocation_time_ns.load(Ordering::Relaxed);
        let count = self.allocations.load(Ordering::Relaxed);
        if count > 0 {
            let new_avg = (current * (count - 1) + time_ns) / count;
            self.avg_allocation_time_ns.store(new_avg, Ordering::Relaxed);
        }
    }
    
    /// 更新释放时间统计
    pub fn update_deallocation_time(&self, time_ns: u64) {
        let current = self.avg_deallocation_time_ns.load(Ordering::Relaxed);
        let count = self.deallocations.load(Ordering::Relaxed);
        if count > 0 {
            let new_avg = (current * (count - 1) + time_ns) / count;
            self.avg_deallocation_time_ns.store(new_avg, Ordering::Relaxed);
        }
    }
    
    /// 获取效率指标
    pub fn efficiency_metrics(&self) -> EfficiencyMetrics {
        EfficiencyMetrics {
            hit_rate: self.hit_rate(),
            avg_allocation_time_ns: self.avg_allocation_time_ns.load(Ordering::Relaxed),
            avg_deallocation_time_ns: self.avg_deallocation_time_ns.load(Ordering::Relaxed),
            utilization_rate: if self.current_capacity > 0 {
                self.used_count as f64 / self.current_capacity as f64
            } else {
                0.0
            },
            fragmentation_ratio: self.calculate_fragmentation(),
        }
    }
    
    /// 计算碎片比率
    fn calculate_fragmentation(&self) -> f64 {
        // 简化的碎片计算，实际实现中可以更复杂
        if self.available_count == 0 {
            0.0
        } else {
            let theoretical_optimal = self.available_count;
            let actual_usage = self.used_count;
            if theoretical_optimal > 0 {
                1.0 - (actual_usage as f64 / theoretical_optimal as f64)
            } else {
                0.0
            }
        }
    }
}

/// 效率指标
#[derive(Debug, Clone)]
pub struct EfficiencyMetrics {
    /// 命中率
    pub hit_rate: f64,
    /// 平均分配时间（纳秒）
    pub avg_allocation_time_ns: u64,
    /// 平均释放时间（纳秒）
    pub avg_deallocation_time_ns: u64,
    /// 利用率
    pub utilization_rate: f64,
    /// 碎片比率
    pub fragmentation_ratio: f64,
}

/// 内存池管理器
pub struct MemoryPoolManager {
    /// 按类型和大小分组的池
    pools: Arc<RwLock<BTreeMap<(ObjectType, SizeClass), Arc<MemoryPool>>>>,
    /// 全局统计
    global_stats: Arc<Mutex<GlobalPoolStats>>,
    /// 配置
    config: MemoryConfig,
    /// 优化器
    optimizer: Arc<PoolOptimizer>,
    /// 监控器
    monitor: Arc<PoolMonitor>,
}

impl MemoryPoolManager {
    /// 创建新的池管理器
    pub fn new(config: &MemoryConfig) -> Result<Self> {
        let optimizer = Arc::new(PoolOptimizer::new());
        let monitor = Arc::new(PoolMonitor::new());
        
        Ok(Self {
            pools: Arc::new(RwLock::new(BTreeMap::new())),
            global_stats: Arc::new(Mutex::new(GlobalPoolStats::new())),
            config: config.clone(),
            optimizer,
            monitor,
        })
    }
    
    /// 分配内存
    pub fn allocate(&self, size: usize, object_type: ObjectType) -> Result<ManagedPtr> {
        let start_time = Instant::now();
        let size_class = SizeClass::from_size(size);
        let pool = self.get_or_create_pool(object_type, size_class)?;
        
        let result = pool.allocate();
        
        // 记录性能指标
        let allocation_time = start_time.elapsed().as_nanos() as u64;
        pool.get_stats().update_allocation_time(allocation_time);
        
        // 更新全局统计
        {
            let mut stats = self.global_stats.lock().unwrap();
            match &result {
                Ok(_) => {
                    stats.total_allocations += 1;
                    stats.total_allocated_bytes += size;
                }
                Err(_) => stats.failed_allocations += 1,
            }
        }
        
        result
    }
    
    /// 释放内存
    pub fn deallocate(&self, ptr: ManagedPtr) -> Result<()> {
        let start_time = Instant::now();
        let size_class = SizeClass::from_size(ptr.size);
        let object_type = ptr.object_type;
        
        let result = if let Some(pool) = self.get_pool(object_type, size_class) {
            pool.deallocate(ptr)
        } else {
            Err(Error::invalid_input("No pool found for pointer"))
        };
        
        // 记录性能指标
        if let Some(pool) = self.get_pool(object_type, size_class) {
            let deallocation_time = start_time.elapsed().as_nanos() as u64;
            pool.get_stats().update_deallocation_time(deallocation_time);
        }
        
        // 更新全局统计
        {
            let mut stats = self.global_stats.lock().unwrap();
            match &result {
                Ok(_) => stats.total_deallocations += 1,
                Err(_) => stats.failed_deallocations += 1,
            }
        }
        
        result
    }
    
    /// 预热所有池
    pub async fn warmup_all(&self) -> Result<()> {
        let pools = self.pools.read().values().cloned().collect::<Vec<_>>();
        
        for pool in pools {
            let warmup_count = (pool.max_capacity as f64 * pool.config.warmup_percentage) as usize;
            pool.warmup(warmup_count)?;
        }
        
        Ok(())
    }
    
    /// 启动自动优化
    pub async fn start_auto_optimization(&self) -> Result<()> {
        let pools_for_opt = Arc::clone(&self.pools);
        let _optimizer_for_opt = Arc::clone(&self.optimizer);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                // 避免持锁跨await：读取键集合后逐个处理
                let keys: Vec<(ObjectType, SizeClass)> = {
                    let pools_read = pools_for_opt.read();
                    pools_read.keys().cloned().collect()
                };
                for key in keys {
                    if let Some(pool) = pools_for_opt.read().get(&key).cloned() {
                        if let Err(e) = pool.adaptive_resize() {
                            log::warn!("Failed to resize pool: {}", e);
                        }
                    }
                }
            }
        });
        // 碎片整理任务用独立变量
        let pools_for_defrag = Arc::clone(&self.pools);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(600));
            loop {
                interval.tick().await;
                // 避免持锁跨await
                let pools: Vec<Arc<MemoryPool>> = {
                    let read = pools_for_defrag.read();
                    read.values().cloned().collect()
                };
                for pool in pools {
                    if let Err(e) = pool.defragment().await {
                        log::warn!("Failed to defragment pool: {}", e);
                    }
                }
            }
        });
        Ok(())
    }
    
    /// 健康检查
    pub fn health_check(&self) -> Result<HealthStatus> {
        let pools = self.pools.read();
        let mut total_hit_rate = 0.0;
        let mut pool_count = 0;
        
        for pool in pools.values() {
            let stats = pool.get_stats();
            total_hit_rate += stats.hit_rate();
            pool_count += 1;
        }
        
        let avg_hit_rate = if pool_count > 0 {
            total_hit_rate / pool_count as f64
        } else {
            1.0
        };
        
        if avg_hit_rate >= 0.9 {
            Ok(HealthStatus::Healthy)
        } else if avg_hit_rate >= 0.7 {
            Ok(HealthStatus::Degraded)
        } else {
            Ok(HealthStatus::Unhealthy)
        }
    }
    
    /// 清理过期资源
    pub fn cleanup_expired(&self) -> Result<()> {
        let pools = self.pools.read();
        let mut total_cleaned = 0;
        
        for pool in pools.values() {
            total_cleaned += pool.cleanup_expired()?;
        }
        
        {
            let mut stats = self.global_stats.lock().unwrap();
            stats.total_cleaned += total_cleaned;
        }
        
        Ok(())
    }
    
    /// 获取全局统计
    pub fn get_global_stats(&self) -> GlobalPoolStats {
        self.global_stats.lock().unwrap().clone()
    }
    
    /// 检查是否可以回收
    pub fn can_recycle(&self, ptr: &ManagedPtr) -> bool {
        let size_class = SizeClass::from_size(ptr.size);
        let object_type = ptr.object_type;
        
        self.get_pool(object_type, size_class).is_some()
    }

    /// 获取或创建池
    fn get_or_create_pool(&self, object_type: ObjectType, size_class: SizeClass) -> Result<Arc<MemoryPool>> {
        let key = (object_type, size_class);
        
        // 先尝试读取
        {
            let pools = self.pools.read();
            if let Some(pool) = pools.get(&key) {
                return Ok(pool.clone());
            }
        }
        
        // 需要创建新池
        let mut pools = self.pools.write();
        
        // 双重检查
        if let Some(pool) = pools.get(&key) {
            return Ok(pool.clone());
        }
        
        // 创建新池
        let pool_id = format!("{:?}_{:?}", object_type, size_class);
        let pool_config = self.create_pool_config(&size_class);
        let pool = Arc::new(MemoryPool::new(pool_id, size_class, object_type, pool_config));
        
        pools.insert(key, pool.clone());
        
        // 更新全局统计
        {
            let mut stats = self.global_stats.lock().unwrap();
            stats.total_pools += 1;
        }
        
        Ok(pool)
    }
    
    /// 获取现有池
    fn get_pool(&self, object_type: ObjectType, size_class: SizeClass) -> Option<Arc<MemoryPool>> {
        let pools = self.pools.read();
        pools.get(&(object_type, size_class)).cloned()
    }
    
    /// 根据大小类别创建池配置
    fn create_pool_config(&self, size_class: &SizeClass) -> PoolConfig {
        match size_class {
            SizeClass::Small(_) => PoolConfig {
                max_capacity: 5000,
                min_capacity: 50,
                max_recycled: 500,
                max_idle_seconds: 600,
                min_use_count: 1,
                warmup_percentage: 0.3,
                adaptive_resize_interval: 30,
                defrag_interval: 300,
            },
            SizeClass::Medium(_) => PoolConfig {
                max_capacity: 1000,
                min_capacity: 20,
                max_recycled: 100,
                max_idle_seconds: 300,
                min_use_count: 2,
                warmup_percentage: 0.2,
                adaptive_resize_interval: 60,
                defrag_interval: 600,
            },
            SizeClass::Large(_) => PoolConfig {
                max_capacity: 100,
                min_capacity: 5,
                max_recycled: 20,
                max_idle_seconds: 180,
                min_use_count: 3,
                warmup_percentage: 0.1,
                adaptive_resize_interval: 120,
                defrag_interval: 1200,
            },
            SizeClass::XLarge(_) => PoolConfig {
                max_capacity: 10,
                min_capacity: 1,
                max_recycled: 2,
                max_idle_seconds: 60,
                min_use_count: 5,
                warmup_percentage: 0.05,
                adaptive_resize_interval: 300,
                defrag_interval: 1800,
            },
        }
    }
}

/// 池优化器
pub struct PoolOptimizer {
    /// 优化策略
    strategies: Vec<Box<dyn OptimizationStrategy>>,
}

impl PoolOptimizer {
    pub fn new() -> Self {
        Self {
            strategies: vec![
                Box::new(HitRateOptimizer::new()),
                Box::new(MemoryUsageOptimizer::new()),
                Box::new(FragmentationOptimizer::new()),
            ],
        }
    }
}

/// 池监控器
pub struct PoolMonitor {
    /// 监控指标
    metrics: Arc<Mutex<HashMap<String, PoolMetrics>>>,
}

impl PoolMonitor {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

/// 池指标
#[derive(Debug, Clone)]
pub struct PoolMetrics {
    pub pool_id: String,
    pub hit_rate: f64,
    pub utilization: f64,
    pub fragmentation: f64,
    pub avg_allocation_time: Duration,
    pub avg_deallocation_time: Duration,
}

/// 优化策略特质
pub trait OptimizationStrategy: Send + Sync {
    fn optimize(&self, pool: &Arc<MemoryPool>) -> Result<()>;
    fn name(&self) -> &str;
}

/// 命中率优化器
pub struct HitRateOptimizer;

impl HitRateOptimizer {
    pub fn new() -> Self {
        Self
    }
}

impl OptimizationStrategy for HitRateOptimizer {
    fn optimize(&self, pool: &Arc<MemoryPool>) -> Result<()> {
        let stats = pool.get_stats();
        if stats.hit_rate() < 0.8 {
            // 如果命中率低，增加预热
            let additional = pool.max_capacity / 10;
            pool.warmup(additional)?;
        }
        Ok(())
    }
    
    fn name(&self) -> &str {
        "HitRateOptimizer"
    }
}

/// 内存使用优化器
pub struct MemoryUsageOptimizer;

impl MemoryUsageOptimizer {
    pub fn new() -> Self {
        Self
    }
}

impl OptimizationStrategy for MemoryUsageOptimizer {
    fn optimize(&self, pool: &Arc<MemoryPool>) -> Result<()> {
        let stats = pool.get_stats();
        let utilization = stats.used_count as f64 / stats.current_capacity as f64;
        
        if utilization < 0.3 && stats.current_capacity > pool.config.min_capacity {
            // 利用率低，清理一些块
            pool.cleanup_expired()?;
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "MemoryUsageOptimizer"
    }
}

/// 碎片优化器
pub struct FragmentationOptimizer;

impl FragmentationOptimizer {
    pub fn new() -> Self {
        Self
    }
}

impl OptimizationStrategy for FragmentationOptimizer {
    fn optimize(&self, pool: &Arc<MemoryPool>) -> Result<()> {
        let stats = pool.get_stats();
        let metrics = stats.efficiency_metrics();
        
        if metrics.fragmentation_ratio > 0.5 {
            // 碎片比率高，进行整理
            let pool_clone = Arc::clone(pool);
            tokio::spawn(async move {
                if let Err(e) = pool_clone.defragment().await {
                    log::warn!("Defragmentation failed: {}", e);
                }
            });
        }
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "FragmentationOptimizer"
    }
}

/// 全局池统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalPoolStats {
    /// 总池数量
    pub total_pools: usize,
    /// 总分配次数
    pub total_allocations: u64,
    /// 总释放次数
    pub total_deallocations: u64,
    /// 失败分配次数
    pub failed_allocations: u64,
    /// 失败释放次数
    pub failed_deallocations: u64,
    /// 总清理次数
    pub total_cleaned: usize,
    /// 总分配字节数
    pub total_allocated_bytes: usize,
    /// 创建时间
    #[serde(skip)]
    pub created_at: Instant,
}

impl GlobalPoolStats {
    pub fn new() -> Self {
        Self {
            total_pools: 0,
            total_allocations: 0,
            total_deallocations: 0,
            failed_allocations: 0,
            failed_deallocations: 0,
            total_cleaned: 0,
            total_allocated_bytes: 0,
            created_at: Instant::now(),
        }
    }
    
    /// 获取整体成功率
    pub fn success_rate(&self) -> f64 {
        let total_operations = self.total_allocations + self.total_deallocations;
        let failed_operations = self.failed_allocations + self.failed_deallocations;
        
        if total_operations == 0 {
            1.0
        } else {
            (total_operations - failed_operations) as f64 / total_operations as f64
        }
    }
    
    /// 获取平均池大小
    pub fn avg_pool_utilization(&self) -> f64 {
        if self.total_pools == 0 {
            0.0
        } else {
            self.total_allocated_bytes as f64 / self.total_pools as f64
        }
    }
} 