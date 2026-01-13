/// 高性能内存管理模块
/// 
/// 提供全面的内存管理功能：
/// 1. 分层内存池管理
/// 2. 智能缓存优化
/// 3. 分片管理器
/// 4. 内存泄漏检测
/// 5. 垃圾回收
/// 6. 引用优化

use std::sync::{Arc, Mutex};
use std::alloc::Layout;
use std::ptr::NonNull;
use std::time::{Instant, Duration};
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use serde::{Serialize, Deserialize};
use log;

use crate::{Error, Result};
use crate::core::types::HealthStatus;

// 模块导出
pub mod pool;
pub mod cache_optimizer;
pub mod shard_manager;
pub mod leak_detector;
pub mod gc_collector;
pub mod reference_optimizer;

// 重新导出主要类型
pub use pool::{MemoryPoolManager, MemoryPool, SizeClass, MemoryBlock, PoolConfig, PoolStats, GlobalPoolStats};

/// 内存对象类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectType {
    /// 张量数据
    Tensor,
    /// 模型参数
    ModelParameters,
    /// 梯度数据
    Gradients,
    /// 缓存数据
    Cache,
    /// 临时数据
    Temporary,
    /// 算法数据
    Algorithm,
    /// 用户数据
    UserData,
    /// 系统数据
    System,
}

impl ObjectType {
    /// 获取对象类型的优先级
    pub fn priority(&self) -> u8 {
        match self {
            Self::System => 255,
            Self::ModelParameters => 200,
            Self::Gradients => 180,
            Self::Tensor => 150,
            Self::Algorithm => 120,
            Self::Cache => 100,
            Self::UserData => 80,
            Self::Temporary => 50,
        }
    }
    
    /// 获取默认生存时间（秒）
    pub fn default_ttl(&self) -> Option<Duration> {
        match self {
            Self::System => None, // 永不过期
            Self::ModelParameters => None, // 永不过期
            Self::Gradients => Some(Duration::from_secs(3600)), // 1小时
            Self::Tensor => Some(Duration::from_secs(1800)), // 30分钟
            Self::Algorithm => Some(Duration::from_secs(7200)), // 2小时
            Self::Cache => Some(Duration::from_secs(900)), // 15分钟
            Self::UserData => Some(Duration::from_secs(3600)), // 1小时
            Self::Temporary => Some(Duration::from_secs(300)), // 5分钟
        }
    }
}

impl PartialOrd for ObjectType {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ObjectType {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority().cmp(&other.priority())
    }
}

/// 托管指针 - 带有元数据的智能指针
#[derive(Debug)]
pub struct ManagedPtr {
    /// 内存指针
    pub ptr: NonNull<u8>,
    /// 内存大小
    pub size: usize,
    /// 对象类型
    pub object_type: ObjectType,
    /// 内存布局
    pub layout: Layout,
    /// 分配时间
    pub allocated_at: Instant,
    /// 最后访问时间
    pub last_accessed: Instant,
    /// 访问次数
    pub access_count: AtomicU64,
    /// 引用计数
    pub ref_count: AtomicUsize,
    /// 是否可回收
    pub recyclable: bool,
}

impl ManagedPtr {
    /// 创建新的托管指针
    pub fn new(ptr: NonNull<u8>, size: usize, object_type: ObjectType, layout: Layout) -> Self {
        let now = Instant::now();
        Self {
            ptr,
            size,
            object_type,
            layout,
            allocated_at: now,
            last_accessed: now,
            access_count: AtomicU64::new(0),
            ref_count: AtomicUsize::new(1),
            recyclable: true,
        }
    }
    
    /// 获取原始指针
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }
    
    /// 获取大小
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// 获取对象类型
    pub fn object_type(&self) -> ObjectType {
        self.object_type
    }
    
    /// 标记访问
    pub fn mark_accessed(&self) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
    }
    
    /// 增加引用计数
    pub fn add_ref(&self) -> usize {
        self.ref_count.fetch_add(1, Ordering::SeqCst)
    }
    
    /// 减少引用计数
    pub fn release(&self) -> usize {
        self.ref_count.fetch_sub(1, Ordering::SeqCst)
    }
    
    /// 获取引用计数
    pub fn ref_count(&self) -> usize {
        self.ref_count.load(Ordering::SeqCst)
    }
    
    /// 获取访问次数
    pub fn access_count(&self) -> u64 {
        self.access_count.load(Ordering::Relaxed)
    }
    
    /// 检查是否可以回收
    pub fn can_recycle(&self) -> bool {
        self.recyclable && self.ref_count() <= 1
    }
    
    /// 获取年龄（分配后经过的时间）
    pub fn age(&self) -> Duration {
        self.allocated_at.elapsed()
    }
    
    /// 获取空闲时间（最后访问后经过的时间）
    pub fn idle_time(&self) -> Duration {
        self.last_accessed.elapsed()
    }
}

unsafe impl Send for ManagedPtr {}
unsafe impl Sync for ManagedPtr {}

/// 内存配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// 最大内存使用量（字节）
    pub max_memory_bytes: usize,
    /// 内存池配置
    pub pool_config: PoolConfig,
    /// 监控配置
    pub monitoring_config: MonitoringConfig,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8GB
            pool_config: PoolConfig::default(),
            monitoring_config: MonitoringConfig::default(),
        }
    }
}

/// 监控配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// 监控间隔（秒）
    pub monitoring_interval_seconds: u64,
    /// 健康检查间隔（秒）
    pub health_check_interval_seconds: u64,
    /// 指标收集启用
    pub metrics_enabled: bool,
    /// 详细日志启用
    pub verbose_logging: bool,
    /// 性能分析启用
    pub profiling_enabled: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            monitoring_interval_seconds: 60, // 1分钟
            health_check_interval_seconds: 30, // 30秒
            metrics_enabled: true,
            verbose_logging: false,
            profiling_enabled: false,
        }
    }
}

/// 内存统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// 总内存使用量（字节）
    pub total_memory: usize,
    /// 可用内存（字节）  
    pub available_memory: usize,
    /// 已使用内存（字节）
    pub used_memory: usize,
    /// 内存使用率
    pub usage_rate: f64,
    /// 碎片率
    pub fragmentation_rate: f64,
    /// 缓存命中率
    pub cache_hit_rate: f64,
    /// 垃圾回收次数
    pub gc_count: u64,
    /// 内存泄漏检测数量
    pub leak_count: u64,
    /// 最后更新时间
    pub last_updated: chrono::DateTime<chrono::Utc>,
    /// 内存池全局统计（补全pool_stats字段，便于外部统计调用）
    pub pool_stats: Option<crate::memory::pool::GlobalPoolStats>,
    /// 更新时间戳（补全updated_at字段，便于外部调用）
    #[serde(skip)]
    pub updated_at: Option<std::time::Instant>,
}

/// 内存管理器 - 统一的内存管理接口
pub struct MemoryManager {
    /// 内存池管理器
    pool_manager: Arc<MemoryPoolManager>,
    /// 配置
    config: MemoryConfig,
    /// 统计信息
    stats: Arc<Mutex<MemoryStats>>,
    /// 是否运行中
    running: Arc<std::sync::atomic::AtomicBool>,
}

impl MemoryManager {
    /// 创建新的内存管理器
    pub fn new(config: MemoryConfig) -> Result<Self> {
        let pool_manager = Arc::new(MemoryPoolManager::new(&config)?);
        
        let stats = Arc::new(Mutex::new(MemoryStats {
            total_memory: 0,
            available_memory: config.max_memory_bytes,
            used_memory: 0,
            usage_rate: 0.0,
            fragmentation_rate: 0.0,
            cache_hit_rate: 0.0,
            gc_count: 0,
            leak_count: 0,
            last_updated: chrono::Utc::now(),
            pool_stats: None,
            updated_at: None,
        }));
        
        Ok(Self {
            pool_manager,
            config,
            stats,
            running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
    }
    
    /// 启动内存管理器
    pub async fn start(&self) -> Result<()> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Err(Error::invalid_state("Memory manager is already running"));
        }
        
        // 启动监控任务
        self.start_monitoring_task().await?;
        
        Ok(())
    }
    
    /// 停止内存管理器
    pub async fn stop(&self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);
        Ok(())
    }
    
    /// 分配内存
    pub fn allocate(&self, size: usize, object_type: ObjectType) -> Result<ManagedPtr> {
        self.pool_manager.allocate(size, object_type)
    }
    
    /// 释放内存
    pub fn deallocate(&self, ptr: ManagedPtr) -> Result<()> {
        self.pool_manager.deallocate(ptr)
    }
    
    /// 获取统计信息
    pub fn get_stats(&self) -> Result<MemoryStats> {
        let stats = self.stats.lock().map_err(|_| Error::internal("Failed to lock stats"))?;
        Ok(stats.clone())
    }
    
    /// 健康检查
    pub fn health_check(&self) -> Result<HealthStatus> {
        self.pool_manager.health_check()
    }
    
    /// 执行清理
    pub async fn cleanup(&self) -> Result<()> {
        self.pool_manager.cleanup_expired()?;
        Ok(())
    }
    
    /// 启动监控任务
    async fn start_monitoring_task(&self) -> Result<()> {
        let stats = Arc::clone(&self.stats);
        let pool_manager = Arc::clone(&self.pool_manager);
        let running = Arc::clone(&self.running);
        let interval = Duration::from_secs(self.config.monitoring_config.monitoring_interval_seconds);
        
        tokio::spawn(async move {
            loop {
                if !running.load(Ordering::SeqCst) { break; }

                // 先快照统计数据，避免在 await 期间持锁
                let snapshot = pool_manager.get_global_stats();
                if let Ok(mut stats_guard) = stats.lock() {
                    stats_guard.pool_stats = Some(snapshot);
                    stats_guard.updated_at = Some(Instant::now());
                }
                tokio::time::sleep(interval).await;
            }
        });
        
        Ok(())
    }

    /// 启动后台任务
    pub async fn start_background_tasks(self: Arc<Self>) -> Result<()> {
        // 启动监控任务
        let manager = Arc::clone(&self);
        tokio::spawn(async move {
            manager.start_monitoring_task().await.unwrap_or_else(|e| {
                log::error!("内存监控任务失败: {}", e);
            });
        });

        // 启动垃圾回收任务
        let manager = Arc::clone(&self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                if let Err(e) = manager.cleanup().await {
                    log::error!("内存清理任务失败: {}", e);
                }
            }
        });

        Ok(())
    }
}

/// 全局内存管理器实例
static GLOBAL_MEMORY_MANAGER: std::sync::OnceLock<Arc<MemoryManager>> = std::sync::OnceLock::new();

/// 初始化全局内存管理器
pub fn init_global_memory_manager(config: MemoryConfig) -> Result<()> {
    let manager = Arc::new(MemoryManager::new(config)?);
    GLOBAL_MEMORY_MANAGER.set(manager)
        .map_err(|_| Error::invalid_state("Global memory manager already initialized"))?;
    Ok(())
}

/// 获取全局内存管理器
pub fn get_global_memory_manager() -> Result<Arc<MemoryManager>> {
    GLOBAL_MEMORY_MANAGER.get()
        .cloned()
        .ok_or_else(|| Error::invalid_state("Global memory manager not initialized"))
}

/// 便捷函数：分配内存
pub fn allocate(size: usize, object_type: ObjectType) -> Result<ManagedPtr> {
    get_global_memory_manager()?.allocate(size, object_type)
}

/// 便捷函数：释放内存
pub fn deallocate(ptr: ManagedPtr) -> Result<()> {
    get_global_memory_manager()?.deallocate(ptr)
}

/// 便捷函数：获取内存统计
pub fn get_memory_stats() -> Result<MemoryStats> {
    get_global_memory_manager()?.get_stats()
}

/// 便捷函数：内存健康检查
pub fn memory_health_check() -> Result<HealthStatus> {
    get_global_memory_manager()?.health_check()
} 