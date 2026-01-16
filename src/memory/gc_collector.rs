/// 垃圾回收器
/// 
/// 提供智能垃圾回收功能：
/// 1. 分代垃圾回收
/// 2. 增量回收
/// 3. 并发回收
/// 4. 自适应回收策略
/// 5. 内存碎片整理

use std::sync::{Arc, Mutex};
use std::collections::{HashMap, VecDeque, HashSet};
use std::time::{Instant, Duration};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use serde::{Serialize, Deserialize};
use parking_lot::RwLock;
use tokio::sync::{Semaphore, RwLock as AsyncRwLock};

use super::{ObjectType, MemoryConfig};
use crate::{Error, Result};
use crate::core::types::HealthStatus;

/// GC配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCConfig {
    /// 启用分代回收
    pub generational_enabled: bool,
    /// 年轻代大小阈值（字节）
    pub young_generation_threshold: usize,
    /// 老年代大小阈值（字节）
    pub old_generation_threshold: usize,
    /// 回收间隔（秒）
    pub collection_interval_seconds: u64,
    /// 增量回收启用
    pub incremental_enabled: bool,
    /// 并发回收启用
    pub concurrent_enabled: bool,
    /// 内存压力阈值
    pub memory_pressure_threshold: f64,
    /// 最大暂停时间（毫秒）
    pub max_pause_time_ms: u64,
    /// 复制区域大小比例
    pub copy_region_ratio: f64,
    /// 内存碎片整理阈值
    pub compaction_threshold: f64,
}

impl Default for GCConfig {
    fn default() -> Self {
        Self {
            generational_enabled: true,
            young_generation_threshold: 64 * 1024 * 1024, // 64MB
            old_generation_threshold: 512 * 1024 * 1024, // 512MB
            collection_interval_seconds: 300, // 5分钟
            incremental_enabled: true,
            concurrent_enabled: true,
            memory_pressure_threshold: 0.8, // 80%
            max_pause_time_ms: 100, // 100ms
            copy_region_ratio: 0.5, // 50%
            compaction_threshold: 0.3, // 30%碎片率
        }
    }
}

/// 回收策略
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CollectionStrategy {
    /// 标记清除
    MarkAndSweep,
    /// 复制回收
    Copying,
    /// 标记整理
    MarkAndCompact,
    /// 增量回收
    Incremental,
    /// 并发回收
    Concurrent,
}

/// 内存区域
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    /// 区域ID
    pub id: String,
    /// 起始地址
    pub start_address: usize,
    /// 结束地址
    pub end_address: usize,
    /// 当前分配指针
    pub allocation_pointer: AtomicUsize,
    /// 区域类型
    pub region_type: RegionType,
    /// 是否活跃
    pub active: AtomicBool,
}

/// 区域类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionType {
    /// Eden区（年轻代）
    Eden,
    /// 幸存者区1
    Survivor1,
    /// 幸存者区2
    Survivor2,
    /// 老年代
    OldGeneration,
    /// 大对象区
    HugeObject,
}

/// 对象元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectMetadata {
    /// 对象ID
    pub object_id: String,
    /// 对象类型
    pub object_type: ObjectType,
    /// 大小
    pub size: usize,
    /// 内存地址
    pub address: usize,
    /// 分配时间
    #[serde(skip)]
    pub allocated_at: Instant,
    /// 最后访问时间
    #[serde(skip)]
    pub last_accessed: Instant,
    /// 访问次数
    pub access_count: u64,
    /// 引用计数
    pub ref_count: usize,
    /// 代数（分代GC）
    pub generation: u8,
    /// 是否标记
    pub marked: bool,
    /// 是否可达
    pub reachable: bool,
    /// 转发指针（复制回收使用）
    pub forwarding_pointer: Option<usize>,
    /// 所在区域
    pub region_id: String,
}

impl ObjectMetadata {
    /// 创建新的对象元数据
    pub fn new(object_id: String, object_type: ObjectType, size: usize, address: usize, region_id: String) -> Self {
        let now = Instant::now();
        Self {
            object_id,
            object_type,
            size,
            address,
            allocated_at: now,
            last_accessed: now,
            access_count: 0,
            ref_count: 1,
            generation: 0,
            marked: false,
            reachable: true,
            forwarding_pointer: None,
            region_id,
        }
    }
    
    /// 更新访问信息
    pub fn update_access(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
    
    /// 检查是否应该晋升到下一代
    pub fn should_promote(&self) -> bool {
        self.access_count > 10 && self.allocated_at.elapsed() > Duration::from_secs(3600)
    }
    
    /// 晋升到下一代
    pub fn promote(&mut self) {
        if self.generation < 2 {
            self.generation += 1;
        }
    }
    
    /// 重置标记状态
    pub fn reset_mark(&mut self) {
        self.marked = false;
        self.reachable = false;
        self.forwarding_pointer = None;
    }
}

/// GC统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCStats {
    /// 总回收次数
    pub total_collections: u64,
    /// 年轻代回收次数
    pub young_collections: u64,
    /// 老年代回收次数
    pub old_collections: u64,
    /// 回收的对象数量
    pub collected_objects: u64,
    /// 回收的内存大小（字节）
    pub collected_memory: usize,
    /// 平均暂停时间（毫秒）
    pub avg_pause_time_ms: f64,
    /// 最大暂停时间（毫秒）
    pub max_pause_time_ms: u64,
    /// 最后回收时间
    #[serde(skip)]
    pub last_collection: Option<Instant>,
    /// 内存碎片率
    pub fragmentation_rate: f64,
    /// 复制回收次数
    pub copy_collections: u64,
    /// 整理回收次数
    pub compact_collections: u64,
    /// 增量回收次数
    pub incremental_collections: u64,
    /// 并发回收次数
    pub concurrent_collections: u64,
    /// 创建时间
    #[serde(skip)]
    pub created_at: Instant,
}

impl GCStats {
    /// 创建新的统计
    pub fn new() -> Self {
        Self {
            total_collections: 0,
            young_collections: 0,
            old_collections: 0,
            collected_objects: 0,
            collected_memory: 0,
            avg_pause_time_ms: 0.0,
            max_pause_time_ms: 0,
            last_collection: None,
            fragmentation_rate: 0.0,
            copy_collections: 0,
            compact_collections: 0,
            incremental_collections: 0,
            concurrent_collections: 0,
            created_at: Instant::now(),
        }
    }
    
    /// 更新暂停时间统计
    pub fn update_pause_time(&mut self, pause_time_ms: u64) {
        if pause_time_ms > self.max_pause_time_ms {
            self.max_pause_time_ms = pause_time_ms;
        }
        
        // 计算平均暂停时间
        let total_pause_time = self.avg_pause_time_ms * self.total_collections as f64;
        self.avg_pause_time_ms = (total_pause_time + pause_time_ms as f64) / (self.total_collections + 1) as f64;
    }
    
    /// 更新回收策略统计
    pub fn update_strategy_stats(&mut self, strategy: CollectionStrategy) {
        match strategy {
            CollectionStrategy::Copying => self.copy_collections += 1,
            CollectionStrategy::MarkAndCompact => self.compact_collections += 1,
            CollectionStrategy::Incremental => self.incremental_collections += 1,
            CollectionStrategy::Concurrent => self.concurrent_collections += 1,
            _ => {}
        }
    }
}

/// 根对象追踪器
#[derive(Debug)]
pub struct RootTracker {
    /// 根对象集合
    roots: RwLock<HashSet<String>>,
    /// 弱引用映射
    weak_references: RwLock<HashMap<String, HashSet<String>>>,
}

impl RootTracker {
    pub fn new() -> Self {
        Self {
            roots: RwLock::new(HashSet::new()),
            weak_references: RwLock::new(HashMap::new()),
        }
    }
    
    /// 添加根对象
    pub fn add_root(&self, object_id: String) {
        let mut roots = self.roots.write();
        roots.insert(object_id);
    }
    
    /// 移除根对象
    pub fn remove_root(&self, object_id: &str) {
        let mut roots = self.roots.write();
        roots.remove(object_id);
    }
    
    /// 获取所有根对象
    pub fn get_roots(&self) -> HashSet<String> {
        let roots = self.roots.read();
        roots.clone()
    }
    
    /// 追踪可达对象
    pub fn trace_reachable(&self, objects: &HashMap<String, ObjectMetadata>) -> HashSet<String> {
        let mut reachable = HashSet::new();
        let mut stack = Vec::new();
        let roots = self.get_roots();
        
        // 从根对象开始
        for root in roots {
            if objects.contains_key(&root) {
                stack.push(root);
            }
        }
        
        // 深度优先搜索
        while let Some(object_id) = stack.pop() {
            if reachable.contains(&object_id) {
                continue;
            }
            
            reachable.insert(object_id.clone());
            
            // 查找引用的对象
            if let Some(refs) = self.weak_references.read().get(&object_id) {
                for referenced in refs {
                    if objects.contains_key(referenced) && !reachable.contains(referenced) {
                        stack.push(referenced.clone());
                    }
                }
            }
        }
        
        reachable
    }
}

/// 垃圾回收器
pub struct GarbageCollector {
    /// 对象元数据
    objects: AsyncRwLock<HashMap<String, ObjectMetadata>>,
    /// 内存区域
    regions: RwLock<HashMap<String, MemoryRegion>>,
    /// 根对象追踪器
    root_tracker: Arc<RootTracker>,
    /// 配置
    config: GCConfig,
    /// 统计信息
    stats: Arc<Mutex<GCStats>>,
    /// 是否运行中
    running: Arc<AtomicBool>,
    /// 是否正在回收
    collecting: AtomicBool,
    /// 回收策略
    strategy: CollectionStrategy,
    /// 并发控制信号量
    concurrent_semaphore: Arc<Semaphore>,
    /// 增量回收状态
    incremental_state: RwLock<IncrementalState>,
}

/// 增量回收状态
#[derive(Debug, Clone)]
pub struct IncrementalState {
    /// 当前阶段
    pub phase: IncrementalPhase,
    /// 待处理对象队列
    pub pending_objects: VecDeque<String>,
    /// 每次处理的批次大小
    pub batch_size: usize,
    /// 已处理对象数
    pub processed_count: usize,
}

/// 增量回收阶段
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncrementalPhase {
    /// 标记阶段
    Marking,
    /// 清除阶段
    Sweeping,
    /// 整理阶段
    Compacting,
    /// 完成
    Completed,
}

impl GarbageCollector {
    /// 创建新的垃圾回收器
    pub fn new(_config: &MemoryConfig) -> Result<Self> {
        let gc_config = GCConfig::default();
        let mut regions = HashMap::new();
        
        // 创建内存区域
        Self::create_memory_regions(&mut regions, &gc_config)?;
        
        Ok(Self {
            objects: AsyncRwLock::new(HashMap::new()),
            regions: RwLock::new(regions),
            root_tracker: Arc::new(RootTracker::new()),
            config: gc_config,
            stats: Arc::new(Mutex::new(GCStats::new())),
            running: Arc::new(AtomicBool::new(false)),
            collecting: AtomicBool::new(false),
            strategy: CollectionStrategy::MarkAndSweep,
            concurrent_semaphore: Arc::new(Semaphore::new(4)),
            incremental_state: RwLock::new(IncrementalState {
                phase: IncrementalPhase::Completed,
                pending_objects: VecDeque::new(),
                batch_size: 100,
                processed_count: 0,
            }),
        })
    }
    
    /// 创建内存区域
    fn create_memory_regions(regions: &mut HashMap<String, MemoryRegion>, config: &GCConfig) -> Result<()> {
        let base_address = 0x1000000; // 模拟基地址
        let region_size = config.young_generation_threshold / 3;
        
        // Eden区
        regions.insert("eden".to_string(), MemoryRegion {
            id: "eden".to_string(),
            start_address: base_address,
            end_address: base_address + region_size,
            allocation_pointer: AtomicUsize::new(base_address),
            region_type: RegionType::Eden,
            active: AtomicBool::new(true),
        });
        
        // 幸存者区1
        regions.insert("survivor1".to_string(), MemoryRegion {
            id: "survivor1".to_string(),
            start_address: base_address + region_size,
            end_address: base_address + region_size * 2,
            allocation_pointer: AtomicUsize::new(base_address + region_size),
            region_type: RegionType::Survivor1,
            active: AtomicBool::new(false),
        });
        
        // 幸存者区2
        regions.insert("survivor2".to_string(), MemoryRegion {
            id: "survivor2".to_string(),
            start_address: base_address + region_size * 2,
            end_address: base_address + region_size * 3,
            allocation_pointer: AtomicUsize::new(base_address + region_size * 2),
            region_type: RegionType::Survivor2,
            active: AtomicBool::new(false),
        });
        
        // 老年代
        regions.insert("old".to_string(), MemoryRegion {
            id: "old".to_string(),
            start_address: base_address + region_size * 3,
            end_address: base_address + region_size * 3 + config.old_generation_threshold,
            allocation_pointer: AtomicUsize::new(base_address + region_size * 3),
            region_type: RegionType::OldGeneration,
            active: AtomicBool::new(true),
        });
        
        Ok(())
    }
    
    /// 启动垃圾回收器
    pub async fn start(self: Arc<Self>) -> Result<()> {
        if self.running.load(Ordering::SeqCst) {
            return Err(Error::invalid_state("Garbage collector is already running"));
        }
        
        // 启动后台回收任务
        self.start_background_tasks().await?;
        
        Ok(())
    }
    
    /// 停止垃圾回收器
    pub async fn stop(&self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);
        
        // 等待当前回收完成
        while self.collecting.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        
        Ok(())
    }
    
    /// 注册对象
    pub async fn register_object(&self, object_id: String, object_type: ObjectType, size: usize) -> Result<()> {
        let region_id = self.select_allocation_region(size, object_type).await?;
        let address = self.allocate_in_region(&region_id, size).await?;
        
        let metadata = ObjectMetadata::new(object_id.clone(), object_type, size, address, region_id);
        
        let mut objects = self.objects.write().await;
        objects.insert(object_id, metadata);
        
        Ok(())
    }
    
    /// 选择分配区域
    async fn select_allocation_region(&self, size: usize, object_type: ObjectType) -> Result<String> {
        // 大对象直接分配到老年代
        if size > self.config.young_generation_threshold / 4 {
            return Ok("old".to_string());
        }
        
        // 根据对象类型选择区域
        match object_type {
            ObjectType::System | ObjectType::ModelParameters => Ok("old".to_string()),
            _ => Ok("eden".to_string()),
        }
    }
    
    /// 在区域中分配内存
    async fn allocate_in_region(&self, region_id: &str, size: usize) -> Result<usize> {
        let regions = self.regions.read();
        if let Some(region) = regions.get(region_id) {
            let current_ptr = region.allocation_pointer.load(Ordering::SeqCst);
            let new_ptr = current_ptr + size;
            
            if new_ptr <= region.end_address {
                region.allocation_pointer.store(new_ptr, Ordering::SeqCst);
                Ok(current_ptr)
            } else {
                Err(Error::resource("Region is full"))
            }
        } else {
            Err(Error::invalid_argument("Invalid region ID"))
        }
    }
    
    /// 取消注册对象
    pub async fn unregister_object(&self, object_id: &str) -> Result<()> {
        let mut objects = self.objects.write().await;
        objects.remove(object_id);
        Ok(())
    }
    
    /// 更新对象访问信息
    pub async fn update_object_access(&self, object_id: &str, ref_count: usize) -> Result<()> {
        let mut objects = self.objects.write().await;
        if let Some(metadata) = objects.get_mut(object_id) {
            metadata.update_access();
            metadata.ref_count = ref_count;
        }
        Ok(())
    }
    
    /// 执行垃圾回收
    pub async fn collect(&self) -> Result<usize> {
        if self.collecting.swap(true, Ordering::SeqCst) {
            return Ok(0); // 已经在回收中
        }
        
        let start_time = Instant::now();
        let collected_count = match self.strategy {
            CollectionStrategy::MarkAndSweep => self.mark_and_sweep().await?,
            CollectionStrategy::Copying => self.copying_collect().await?,
            CollectionStrategy::MarkAndCompact => self.mark_and_compact().await?,
            CollectionStrategy::Incremental => self.incremental_collect().await?,
            CollectionStrategy::Concurrent => self.concurrent_collect().await?,
        };
        
        let pause_time_ms = start_time.elapsed().as_millis() as u64;
        
        // 更新统计
        if let Ok(mut stats) = self.stats.lock() {
            stats.total_collections += 1;
            stats.collected_objects += collected_count as u64;
            stats.last_collection = Some(Instant::now());
            stats.update_pause_time(pause_time_ms);
            stats.update_strategy_stats(self.strategy);
        }
        
        self.collecting.store(false, Ordering::SeqCst);
        Ok(collected_count)
    }
    
    /// 标记清除算法
    async fn mark_and_sweep(&self) -> Result<usize> {
        let mut collected_count = 0;
        
        // 标记阶段
        {
            let objects = self.objects.read().await;
            let reachable_objects = self.root_tracker.trace_reachable(&objects);
            
            // 更新可达性标记
            drop(objects);
            let mut objects = self.objects.write().await;
            for (object_id, metadata) in objects.iter_mut() {
                metadata.marked = reachable_objects.contains(object_id);
                metadata.reachable = metadata.marked;
            }
        }
        
        // 清除阶段
        {
            let mut objects = self.objects.write().await;
            let mut to_remove = Vec::new();
            
            for (object_id, metadata) in objects.iter() {
                if !metadata.marked {
                    to_remove.push(object_id.clone());
                    collected_count += 1;
                    
                    // 更新统计
                    if let Ok(mut stats) = self.stats.lock() {
                        stats.collected_memory += metadata.size;
                    }
                }
            }
            
            // 移除未标记的对象
            for object_id in to_remove {
                objects.remove(&object_id);
            }
        }
        
        Ok(collected_count)
    }
    
    /// 复制回收算法
    async fn copying_collect(&self) -> Result<usize> {
        let mut collected_count = 0;
        let mut moved_objects = HashMap::new();
        
        // 选择源区域和目标区域
        let (source_region, target_region) = self.select_copy_regions().await?;
        
        {
            let objects = self.objects.read().await;
            let reachable_objects = self.root_tracker.trace_reachable(&objects);
            
            // 复制可达对象到目标区域
            for object_id in reachable_objects {
                if let Some(metadata) = objects.get(&object_id) {
                    if metadata.region_id == source_region {
                        // 在目标区域分配新地址
                        if let Ok(new_address) = self.allocate_in_region(&target_region, metadata.size).await {
                            moved_objects.insert(object_id.clone(), new_address);
                        }
                    }
                }
            }
        }
        
        // 更新对象元数据
        {
            let mut objects = self.objects.write().await;
            let mut to_remove = Vec::new();
            
            for (object_id, metadata) in objects.iter_mut() {
                if metadata.region_id == source_region {
                    if let Some(&new_address) = moved_objects.get(object_id) {
                        // 更新对象位置
                        metadata.address = new_address;
                        metadata.region_id = target_region.clone();
                        metadata.forwarding_pointer = Some(new_address);
                    } else {
                        // 标记为待删除
                        to_remove.push(object_id.clone());
                        collected_count += 1;
                        
                        // 更新统计
                        if let Ok(mut stats) = self.stats.lock() {
                            stats.collected_memory += metadata.size;
                        }
                    }
                }
            }
            
            // 移除未复制的对象
            for object_id in to_remove {
                objects.remove(&object_id);
            }
        }
        
        // 重置源区域
        self.reset_region(&source_region).await?;
        
        Ok(collected_count)
    }
    
    /// 选择复制区域
    async fn select_copy_regions(&self) -> Result<(String, String)> {
        let regions = self.regions.read();
        
        // 年轻代复制：Eden -> Survivor
        if let Some(eden) = regions.get("eden") {
            if eden.active.load(Ordering::SeqCst) {
                let survivor1_active = regions.get("survivor1").unwrap().active.load(Ordering::SeqCst);
                let target = if survivor1_active { "survivor2" } else { "survivor1" };
                return Ok(("eden".to_string(), target.to_string()));
            }
        }
        
        // Survivor之间的复制
        let survivor1_active = regions.get("survivor1").unwrap().active.load(Ordering::SeqCst);
        if survivor1_active {
            Ok(("survivor1".to_string(), "survivor2".to_string()))
        } else {
            Ok(("survivor2".to_string(), "survivor1".to_string()))
        }
    }
    
    /// 重置区域
    async fn reset_region(&self, region_id: &str) -> Result<()> {
        let regions = self.regions.read();
        if let Some(region) = regions.get(region_id) {
            region.allocation_pointer.store(region.start_address, Ordering::SeqCst);
            region.active.store(false, Ordering::SeqCst);
        }
        Ok(())
    }
    
    /// 标记整理算法
    async fn mark_and_compact(&self) -> Result<usize> {
        // 先执行标记清除
        let collected_count = self.mark_and_sweep().await?;
        
        // 执行内存整理
        self.compact_memory().await?;
        
        // 更新统计
        if let Ok(mut stats) = self.stats.lock() {
            stats.compact_collections += 1;
        }
        
        Ok(collected_count)
    }
    
    /// 内存整理
    async fn compact_memory(&self) -> Result<()> {
        let mut compaction_plan = Vec::new();
        
        // 分析内存布局，制定整理计划
        {
            let objects = self.objects.read().await;
            let mut region_objects: HashMap<String, Vec<&ObjectMetadata>> = HashMap::new();
            
            // 按区域分组对象
            for metadata in objects.values() {
                region_objects.entry(metadata.region_id.clone())
                    .or_insert_with(Vec::new)
                    .push(metadata);
            }
            
            // 为每个区域制定整理计划
            for (region_id, mut objects_in_region) in region_objects {
                // 按地址排序
                objects_in_region.sort_by_key(|obj| obj.address);
                
                let regions = self.regions.read();
                if let Some(region) = regions.get(&region_id) {
                    let mut current_address = region.start_address;
                    
                    for obj in objects_in_region {
                        if obj.address != current_address {
                            // 需要移动对象
                            compaction_plan.push((obj.object_id.clone(), obj.address, current_address));
                        }
                        current_address += obj.size;
                        // 对齐到8字节边界
                        current_address = (current_address + 7) & !7;
                    }
                }
            }
        }
        
        // 执行内存整理
        {
            let mut objects = self.objects.write().await;
            
            for (object_id, _old_address, new_address) in compaction_plan {
                if let Some(metadata) = objects.get_mut(&object_id) {
                    // 模拟内存移动
                    metadata.address = new_address;
                    metadata.forwarding_pointer = Some(new_address);
                }
            }
        }
        
        // 更新碎片率统计
        self.update_fragmentation_stats().await?;
        
        Ok(())
    }
    
    /// 更新碎片率统计
    async fn update_fragmentation_stats(&self) -> Result<()> {
        let _objects = self.objects.read().await;
        let regions = self.regions.read();
        
        let mut total_allocated = 0;
        let mut total_used = 0;
        
        for region in regions.values() {
            let region_size = region.end_address - region.start_address;
            total_allocated += region_size;
            
            let current_ptr = region.allocation_pointer.load(Ordering::SeqCst);
            let used_size = current_ptr - region.start_address;
            total_used += used_size;
        }
        
        let fragmentation_rate = if total_allocated > 0 {
            1.0 - (total_used as f64 / total_allocated as f64)
        } else {
            0.0
        };
        
        if let Ok(mut stats) = self.stats.lock() {
            stats.fragmentation_rate = fragmentation_rate;
        }
        
        Ok(())
    }
    
    /// 增量回收算法
    async fn incremental_collect(&self) -> Result<usize> {
        let mut collected_count = 0;
        let max_time = Duration::from_millis(self.config.max_pause_time_ms);
        let start_time = Instant::now();
        
        loop {
            // 检查时间限制
            if start_time.elapsed() >= max_time {
                break;
            }
            
            let current_phase = {
                let state = self.incremental_state.read();
                state.phase
            };
            
            match current_phase {
                IncrementalPhase::Marking => {
                    let processed = self.incremental_mark_step().await?;
                    if processed == 0 {
                        // 标记阶段完成，进入清除阶段
                        let mut state = self.incremental_state.write();
                        state.phase = IncrementalPhase::Sweeping;
                        state.processed_count = 0;
                    }
                }
                IncrementalPhase::Sweeping => {
                    let swept = self.incremental_sweep_step().await?;
                    collected_count += swept;
                    if swept == 0 {
                        // 清除阶段完成
                        let mut state = self.incremental_state.write();
                        state.phase = IncrementalPhase::Completed;
                    }
                }
                IncrementalPhase::Compacting => {
                    let compacted = self.incremental_compact_step().await?;
                    if compacted == 0 {
                        // 整理阶段完成
                        let mut state = self.incremental_state.write();
                        state.phase = IncrementalPhase::Completed;
                    }
                }
                IncrementalPhase::Completed => {
                    // 重新开始标记阶段
                    self.start_incremental_collection().await?;
                }
            }
            
            // 让出控制权
            tokio::task::yield_now().await;
        }
        
        // 更新统计
        if let Ok(mut stats) = self.stats.lock() {
            stats.incremental_collections += 1;
        }
        
        Ok(collected_count)
    }
    
    /// 开始增量回收
    async fn start_incremental_collection(&self) -> Result<()> {
        let objects = self.objects.read().await;
        let object_ids: Vec<String> = objects.keys().cloned().collect();
        
        let mut state = self.incremental_state.write();
        state.phase = IncrementalPhase::Marking;
        state.pending_objects = object_ids.into();
        state.processed_count = 0;
        
        Ok(())
    }
    
    /// 增量标记步骤
    async fn incremental_mark_step(&self) -> Result<usize> {
        let batch_size = {
            let state = self.incremental_state.read();
            state.batch_size
        };
        
        let mut processed = 0;
        let reachable_objects = {
            let objects = self.objects.read().await;
            self.root_tracker.trace_reachable(&objects)
        };
        
        {
            let mut state = self.incremental_state.write();
            
            for _ in 0..batch_size {
                if let Some(object_id) = state.pending_objects.pop_front() {
                    // 标记对象
                    if let Ok(mut objects) = self.objects.try_write() {
                        if let Some(metadata) = objects.get_mut(&object_id) {
                            metadata.marked = reachable_objects.contains(&object_id);
                        }
                    }
                    processed += 1;
                } else {
                    break;
                }
            }
            
            state.processed_count += processed;
        }
        
        Ok(processed)
    }
    
    /// 增量清除步骤
    async fn incremental_sweep_step(&self) -> Result<usize> {
        let batch_size = {
            let state = self.incremental_state.read();
            state.batch_size
        };
        
        let mut collected = 0;
        
        {
            let mut objects = self.objects.write().await;
            let mut to_remove = Vec::new();
            
            let mut count = 0;
            for (object_id, metadata) in objects.iter() {
                if count >= batch_size {
                    break;
                }
                
                if !metadata.marked {
                    to_remove.push(object_id.clone());
                    collected += 1;
                    
                    // 更新统计
                    if let Ok(mut stats) = self.stats.lock() {
                        stats.collected_memory += metadata.size;
                    }
                }
                count += 1;
            }
            
            // 移除未标记的对象
            for object_id in to_remove {
                objects.remove(&object_id);
            }
        }
        
        Ok(collected)
    }
    
    /// 增量整理步骤
    /// 
    /// 执行增量式的内存整理，每次只处理一小批对象，避免长时间暂停
    async fn incremental_compact_step(&self) -> Result<usize> {
        let batch_size = {
            let state = self.incremental_state.read();
            state.batch_size / 10 // 整理比较耗时，减少批次大小
        };
        
        let mut compacted_count = 0;
        let mut compaction_plan = Vec::new();
        
        // 获取当前增量状态
        let current_phase = {
            let state = self.incremental_state.read();
            state.phase
        };
        
        // 如果不在整理阶段，初始化整理计划
        if current_phase != IncrementalPhase::Compacting {
            // 分析内存布局，制定整理计划
            {
                let objects = self.objects.read().await;
                let mut region_objects: HashMap<String, Vec<(&String, &ObjectMetadata)>> = HashMap::new();
                
                // 按区域分组对象
                for (object_id, metadata) in objects.iter() {
                    if metadata.marked && metadata.reachable {
                        region_objects.entry(metadata.region_id.clone())
                            .or_insert_with(Vec::new)
                            .push((object_id, metadata));
                    }
                }
                
                // 为每个区域制定整理计划
                let regions = self.regions.read();
                for (region_id, mut objects_in_region) in region_objects {
                    if let Some(region) = regions.get(&region_id) {
                        // 按地址排序
                        objects_in_region.sort_by_key(|(_, obj)| obj.address);
                        
                        let mut current_address = region.start_address;
                        
                        for (object_id, obj) in objects_in_region {
                            if obj.address != current_address {
                                // 需要移动对象
                                compaction_plan.push((object_id.clone(), obj.address, current_address));
                            }
                            current_address += obj.size;
                            // 对齐到8字节边界
                            current_address = (current_address + 7) & !7;
                        }
                    }
                }
            }
            
            // 更新增量状态，进入整理阶段
            {
                let mut state = self.incremental_state.write();
                state.phase = IncrementalPhase::Compacting;
                state.pending_objects = compaction_plan.iter()
                    .map(|(object_id, _, _)| object_id.clone())
                    .collect();
                state.processed_count = 0;
            }
        }
        
        // 执行增量整理：每次只处理一小批对象
        {
            let mut state = self.incremental_state.write();
            let mut objects = self.objects.write().await;
            
            let mut processed = 0;
            let mut to_process = Vec::new();
            
            // 从待处理队列中取出一批对象
            for _ in 0..batch_size.min(state.pending_objects.len()) {
                if let Some(object_id) = state.pending_objects.pop_front() {
                    to_process.push(object_id);
                } else {
                    break;
                }
            }
            
            // 处理这批对象
            for object_id in to_process {
                // 查找整理计划中对应的移动信息
                if let Some((_, old_address, new_address)) = compaction_plan.iter()
                    .find(|(id, _, _)| id == &object_id)
                    .cloned()
                {
                    if let Some(metadata) = objects.get_mut(&object_id) {
                        // 检查对象地址是否匹配
                        if metadata.address == old_address {
                            // 执行内存移动（在实际系统中，这里需要真正的内存拷贝）
                            metadata.address = new_address;
                            metadata.forwarding_pointer = Some(new_address);
                            compacted_count += 1;
                            processed += 1;
                        }
                    }
                }
            }
            
            state.processed_count += processed;
            
            // 如果所有对象都已处理，更新碎片率统计
            if state.pending_objects.is_empty() {
                drop(objects);
                drop(state);
                self.update_fragmentation_stats().await?;
                
                // 标记整理完成
                let mut state = self.incremental_state.write();
                state.phase = IncrementalPhase::Completed;
            }
        }
        
        Ok(compacted_count)
    }
    
    /// 并发回收算法
    async fn concurrent_collect(&self) -> Result<usize> {
        let _permit = self.concurrent_semaphore.acquire().await.map_err(|_| Error::internal("Failed to acquire semaphore"))?;
        
        let mut collected_count = 0;
        
        // 并发标记阶段
        let reachable_objects = {
            let objects = self.objects.read().await;
            self.root_tracker.trace_reachable(&objects)
        };
        
        // 并发清除阶段
        {
            let mut objects = self.objects.write().await;
            let mut to_remove = Vec::new();
            
            for (object_id, metadata) in objects.iter() {
                if !reachable_objects.contains(object_id) {
                    to_remove.push(object_id.clone());
                    collected_count += 1;
                    
                    // 更新统计
                    if let Ok(mut stats) = self.stats.lock() {
                        stats.collected_memory += metadata.size;
                    }
                }
            }
            
            // 移除未标记的对象
            for object_id in to_remove {
                objects.remove(&object_id);
            }
        }
        
        // 更新统计
        if let Ok(mut stats) = self.stats.lock() {
            stats.concurrent_collections += 1;
        }
        
        Ok(collected_count)
    }
    
    /// 分代回收
    pub async fn generational_collect(&mut self) -> Result<(usize, usize)> {
        if !self.config.generational_enabled {
            let total = self.collect().await?;
            return Ok((total, 0));
        }
        
        // 年轻代回收
        let young_collected = self.collect_young_generation().await?;
        let mut old_collected = 0;
        
        // 检查是否需要老年代回收
        if self.should_collect_old_generation().await {
            old_collected = self.collect_old_generation().await?;
        }
        
        // 更新统计
        if let Ok(mut stats) = self.stats.lock() {
            if young_collected > 0 {
                stats.young_collections += 1;
            }
            if old_collected > 0 {
                stats.old_collections += 1;
            }
            stats.total_collections += 1;
            stats.collected_objects += (young_collected + old_collected) as u64;
            stats.last_collection = Some(Instant::now());
        }
        
        Ok((young_collected, old_collected))
    }
    
    /// 收集年轻代
    async fn collect_young_generation(&mut self) -> Result<usize> {
        // 使用复制回收算法收集年轻代
        let original_strategy = self.strategy;
        self.strategy = CollectionStrategy::Copying;
        
        let collected = self.copying_collect().await?;
        
        // 恢复原始策略
        self.strategy = original_strategy;
        
        Ok(collected)
    }
    
    /// 收集老年代
    async fn collect_old_generation(&mut self) -> Result<usize> {
        // 使用标记整理算法收集老年代
        let original_strategy = self.strategy;
        self.strategy = CollectionStrategy::MarkAndCompact;
        
        let collected = self.mark_and_compact().await?;
        
        // 恢复原始策略
        self.strategy = original_strategy;
        
        Ok(collected)
    }
    
    /// 检查是否应该进行老年代回收
    async fn should_collect_old_generation(&self) -> bool {
        // 检查老年代使用率
        let old_usage = self.get_region_usage("old").await;
        if old_usage > self.config.memory_pressure_threshold {
            return true;
        }
        
        // 检查时间间隔
        if let Ok(stats) = self.stats.lock() {
            if let Some(last_collection) = stats.last_collection {
                last_collection.elapsed() > Duration::from_secs(3600) // 1小时
            } else {
                true
            }
        } else {
            false
        }
    }
    
    /// 获取区域使用率
    async fn get_region_usage(&self, region_id: &str) -> f64 {
        let regions = self.regions.read();
        if let Some(region) = regions.get(region_id) {
            let total_size = region.end_address - region.start_address;
            let used_size = region.allocation_pointer.load(Ordering::SeqCst) - region.start_address;
            used_size as f64 / total_size as f64
        } else {
            0.0
        }
    }
    
    /// 获取统计信息
    pub fn get_stats(&self) -> GCStats {
        if let Ok(stats) = self.stats.lock() {
            stats.clone()
        } else {
            GCStats::new()
        }
    }
    
    /// 健康检查
    pub fn health_check(&self) -> Result<HealthStatus> {
        let stats = self.get_stats();
        
        // 检查暂停时间
        if stats.max_pause_time_ms > self.config.max_pause_time_ms * 2 {
            return Ok(HealthStatus::Warning);
        }
        
        // 检查内存碎片率
        if stats.fragmentation_rate > self.config.compaction_threshold {
            return Ok(HealthStatus::Warning);
        }
        
        // 检查回收效率
        let collection_rate = if stats.total_collections > 0 {
            stats.collected_objects as f64 / stats.total_collections as f64
        } else {
            0.0
        };
        
        if collection_rate < 1.0 {
            return Ok(HealthStatus::Warning);
        }
        
        Ok(HealthStatus::Healthy)
    }
    
    /// 启动后台任务
    async fn start_background_tasks(self: Arc<Self>) -> Result<()> {
        let running = self.running.clone();
        let weak_self = Arc::downgrade(&self);
        tokio::spawn(async move {
            while running.load(Ordering::SeqCst) {
                if let Some(collector) = weak_self.upgrade() {
                    let _ = collector.collect().await;
                }
                tokio::time::sleep(Duration::from_secs(300)).await;
            }
        });
        // 增量回收任务
        if self.config.incremental_enabled {
            let running = self.running.clone();
            let weak_self = Arc::downgrade(&self);
            tokio::spawn(async move {
                while running.load(Ordering::SeqCst) {
                    if let Some(collector) = weak_self.upgrade() {
                        let _ = collector.incremental_collect().await;
                    }
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            });
        }
        Ok(())
    }
} 