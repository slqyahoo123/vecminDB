 /// 分片管理器
/// 
/// 专门处理大模型参数的分片管理：
/// 1. 智能分片策略
/// 2. 内存映射文件支持
/// 3. 跨节点分片协调
/// 4. 分片生命周期管理
/// 5. 压缩和校验

use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Instant, Duration};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use serde::{Serialize, Deserialize};
use parking_lot::RwLock;
use sha2::{Sha256, Digest};
use lz4_flex::{compress_prepend_size, decompress_size_prepended};

use super::{ObjectType, MemoryConfig};
use crate::{Error, Result};
use crate::core::types::HealthStatus;

/// 分片配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardConfig {
    /// 最大分片大小（字节）
    pub max_shard_size: usize,
    /// 最小分片大小（字节）
    pub min_shard_size: usize,
    /// 分片目录
    pub shard_directory: PathBuf,
    /// 启用压缩
    pub compression_enabled: bool,
    /// 启用校验
    pub checksum_enabled: bool,
    /// 最大内存映射数量
    pub max_memory_maps: usize,
    /// 分片复制因子
    pub replication_factor: usize,
    /// 自动清理间隔（秒）
    pub cleanup_interval_seconds: u64,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            max_shard_size: 1024 * 1024 * 1024, // 1GB
            min_shard_size: 64 * 1024 * 1024,   // 64MB
            shard_directory: PathBuf::from("./shards"),
            compression_enabled: true,
            checksum_enabled: true,
            max_memory_maps: 1000,
            replication_factor: 1,
            cleanup_interval_seconds: 3600, // 1小时
        }
    }
}

/// 分片状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardStatus {
    /// 创建中
    Creating,
    /// 活跃
    Active,
    /// 只读
    ReadOnly,
    /// 迁移中
    Migrating,
    /// 已损坏
    Corrupted,
    /// 已删除
    Deleted,
}

/// 分片信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardInfo {
    /// 分片ID
    pub id: String,
    /// 分片路径
    pub path: PathBuf,
    /// 分片大小（字节）
    pub size: usize,
    /// 压缩后大小（字节）
    pub compressed_size: Option<usize>,
    /// 校验和
    pub checksum: Option<String>,
    /// 对象类型
    pub object_type: ObjectType,
    /// 状态
    pub status: ShardStatus,
    /// 创建时间
    pub created_at: Instant,
    /// 最后访问时间
    pub last_accessed: Instant,
    /// 访问次数
    pub access_count: u64,
    /// 引用计数
    pub ref_count: usize,
    /// 是否启用压缩
    pub compressed: bool,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

impl ShardInfo {
    /// 创建新的分片信息
    pub fn new(id: String, path: PathBuf, size: usize, object_type: ObjectType) -> Self {
        let now = Instant::now();
        Self {
            id,
            path,
            size,
            compressed_size: None,
            checksum: None,
            object_type,
            status: ShardStatus::Creating,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            ref_count: 0,
            compressed: false,
            metadata: HashMap::new(),
        }
    }
    
    /// 标记访问
    pub fn mark_accessed(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
    
    /// 检查是否可以删除
    pub fn can_delete(&self) -> bool {
        self.ref_count == 0 && self.status != ShardStatus::Active
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
        if let Some(compressed_size) = self.compressed_size {
            if self.size > 0 {
                compressed_size as f64 / self.size as f64
            } else {
                1.0
            }
        } else {
            1.0
        }
    }
}

/// 分片统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardMetrics {
    /// 总分片数
    pub total_shards: usize,
    /// 活跃分片数
    pub active_shards: usize,
    /// 总大小（字节）
    pub total_size: usize,
    /// 压缩后总大小（字节）
    pub compressed_size: usize,
    /// 平均压缩率
    pub avg_compression_ratio: f64,
    /// 总访问次数
    pub total_accesses: u64,
    /// 内存映射数量
    pub memory_maps: usize,
    /// 损坏分片数
    pub corrupted_shards: usize,
    /// 创建时间
    pub created_at: Instant,
}

impl ShardMetrics {
    /// 创建新的统计
    pub fn new() -> Self {
        Self {
            total_shards: 0,
            active_shards: 0,
            total_size: 0,
            compressed_size: 0,
            avg_compression_ratio: 1.0,
            total_accesses: 0,
            memory_maps: 0,
            corrupted_shards: 0,
            created_at: Instant::now(),
        }
    }
}

/// 分片管理器
pub struct ShardManager {
    /// 分片信息映射
    shards: RwLock<HashMap<String, ShardInfo>>,
    /// 分片配置
    config: ShardConfig,
    /// 统计信息
    metrics: Arc<Mutex<ShardMetrics>>,
    /// 是否运行中
    running: Arc<AtomicBool>,
    /// 下一个分片ID
    next_shard_id: AtomicU64,
}

impl ShardManager {
    /// 创建新的分片管理器
    pub fn new(config: &MemoryConfig) -> Result<Self> {
        let shard_config = ShardConfig::default();
        
        // 确保分片目录存在
        std::fs::create_dir_all(&shard_config.shard_directory)?;
        
        Ok(Self {
            shards: RwLock::new(HashMap::new()),
            config: shard_config,
            metrics: Arc::new(Mutex::new(ShardMetrics::new())),
            running: Arc::new(AtomicBool::new(false)),
            next_shard_id: AtomicU64::new(1),
        })
    }
    
    /// 启动分片管理器
    pub async fn start(self: Arc<Self>) -> Result<()> {
        if self.running.swap(true, Ordering::SeqCst) {
            return Err(Error::invalid_state("Shard manager is already running"));
        }
        
        // 加载现有分片
        self.load_existing_shards().await?;
        
        // 启动后台任务
        self.start_background_tasks().await?;
        
        Ok(())
    }
    
    /// 停止分片管理器
    pub async fn stop(&self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);
        Ok(())
    }
    
    /// 创建分片
    pub async fn create_shard(&self, data: &[u8], object_type: ObjectType) -> Result<String> {
        let shard_id = self.generate_shard_id();
        let shard_path = self.config.shard_directory.join(format!("{}.shard", shard_id));
        
        // 创建分片信息
        let mut shard_info = ShardInfo::new(shard_id.clone(), shard_path.clone(), data.len(), object_type);
        
        // 写入数据到文件
        let processed_data = if self.config.compression_enabled && data.len() > 1024 {
            let compressed = compress_prepend_size(data);
            shard_info.compressed = true;
            shard_info.compressed_size = Some(compressed.len());
            compressed
        } else {
            data.to_vec()
        };
        
        std::fs::write(&shard_path, &processed_data)?;
        
        // 计算校验和
        if self.config.checksum_enabled {
            let mut hasher = Sha256::new();
            hasher.update(&processed_data);
            let checksum = format!("{:x}", hasher.finalize());
            shard_info.checksum = Some(checksum);
        }
        
        shard_info.status = ShardStatus::Active;
        
        // 添加到分片映射
        {
            let mut shards = self.shards.write();
            shards.insert(shard_id.clone(), shard_info);
        }
        
        // 更新统计
        self.update_metrics();
        
        Ok(shard_id)
    }
    
    /// 读取分片
    pub async fn read_shard(&self, shard_id: &str) -> Result<Vec<u8>> {
        // 获取分片信息
        let shard_info = {
            let mut shards = self.shards.write();
            if let Some(info) = shards.get_mut(shard_id) {
                info.mark_accessed();
                info.clone()
            } else {
                return Err(Error::not_found(&format!("Shard not found: {}", shard_id)));
            }
        };
        
        // 检查状态
        if shard_info.status == ShardStatus::Corrupted {
            return Err(Error::data_corruption(&format!("Shard is corrupted: {}", shard_id)));
        }
        
        // 从文件读取
        let data = std::fs::read(&shard_info.path)?;
        
        // 解压缩（如果需要）
        let result = if shard_info.compressed {
            decompress_size_prepended(&data)
                .map_err(|e| Error::internal(&format!("Decompression failed: {}", e)))?
        } else {
            data
        };
        
        Ok(result)
    }
    
    /// 删除分片
    pub async fn delete_shard(&self, shard_id: &str) -> Result<()> {
        // 获取分片信息
        let shard_info = {
            let mut shards = self.shards.write();
            if let Some(info) = shards.remove(shard_id) {
                info
            } else {
                return Err(Error::not_found(&format!("Shard not found: {}", shard_id)));
            }
        };
        
        // 检查是否可以删除
        if !shard_info.can_delete() {
            return Err(Error::invalid_state(&format!("Shard cannot be deleted: {}", shard_id)));
        }
        
        // 删除文件
        if shard_info.path.exists() {
            std::fs::remove_file(&shard_info.path)?;
        }
        
        // 更新统计
        self.update_metrics();
        
        Ok(())
    }
    
    /// 加载现有分片
    async fn load_existing_shards(&self) -> Result<()> {
        if !self.config.shard_directory.exists() {
            return Ok(());
        }
        
        let entries = std::fs::read_dir(&self.config.shard_directory)?;
        
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().and_then(|s| s.to_str()) == Some("shard") {
                if let Some(file_stem) = path.file_stem().and_then(|s| s.to_str()) {
                    // 尝试解析分片ID
                    if let Ok(shard_id) = u64::from_str_radix(file_stem, 16) {
                        // 更新下一个分片ID
                        let current = self.next_shard_id.load(Ordering::SeqCst);
                        if shard_id >= current {
                            self.next_shard_id.store(shard_id + 1, Ordering::SeqCst);
                        }
                        
                        // 创建分片信息（简化版本）
                        let metadata = std::fs::metadata(&path)?;
                        let mut shard_info = ShardInfo::new(
                            file_stem.to_string(),
                            path.clone(),
                            metadata.len() as usize,
                            ObjectType::UserData, // 默认类型
                        );
                        shard_info.status = ShardStatus::Active;
                        
                        let mut shards = self.shards.write();
                        shards.insert(file_stem.to_string(), shard_info);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// 生成分片ID
    fn generate_shard_id(&self) -> String {
        let id = self.next_shard_id.fetch_add(1, Ordering::SeqCst);
        format!("{:016x}", id)
    }
    
    /// 更新统计信息
    fn update_metrics(&self) {
        if let Ok(mut metrics) = self.metrics.lock() {
            let shards = self.shards.read();
            
            metrics.total_shards = shards.len();
            metrics.active_shards = shards.values().filter(|s| s.status == ShardStatus::Active).count();
            metrics.total_size = shards.values().map(|s| s.size).sum();
            metrics.compressed_size = shards.values()
                .filter_map(|s| s.compressed_size)
                .sum();
            metrics.total_accesses = shards.values().map(|s| s.access_count).sum();
            metrics.corrupted_shards = shards.values().filter(|s| s.status == ShardStatus::Corrupted).count();
            
            // 计算平均压缩率
            let compressed_shards: Vec<_> = shards.values().filter(|s| s.compressed).collect();
            if !compressed_shards.is_empty() {
                metrics.avg_compression_ratio = compressed_shards.iter()
                    .map(|s| s.compression_ratio())
                    .sum::<f64>() / compressed_shards.len() as f64;
            }
        }
    }
    
    /// 获取统计信息
    pub fn get_metrics(&self) -> ShardMetrics {
        if let Ok(metrics) = self.metrics.lock() {
            metrics.clone()
        } else {
            ShardMetrics::new()
        }
    }
    
    /// 健康检查
    pub fn health_check(&self) -> Result<HealthStatus> {
        let metrics = self.get_metrics();
        
        // 检查损坏分片
        if metrics.corrupted_shards > 0 {
            return Ok(HealthStatus::Critical);
        }
        
        Ok(HealthStatus::Healthy)
    }
    
    /// 启动后台任务
    async fn start_background_tasks(self: Arc<Self>) -> Result<()> {
        let running = self.running.clone();
        let weak_self = Arc::downgrade(&self);
        tokio::spawn(async move {
            while running.load(Ordering::SeqCst) {
                if let Some(manager) = weak_self.upgrade() {
                    manager.update_metrics();
                }
                tokio::time::sleep(Duration::from_secs(300)).await;
            }
        });
        Ok(())
    }
}