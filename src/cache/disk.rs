use std::collections::HashMap;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{RwLock, Mutex};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use log::warn;
use sha2::{Sha256, Digest};
use serde::{Serialize, Deserialize};

use crate::error::{Error, Result};
use crate::cache_common::EvictionPolicy;
use super::manager::CacheMetrics;

/// 磁盘缓存项元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiskCacheItemMeta {
    /// 键
    key: String,
    /// 文件路径（相对于缓存根目录）
    file_path: String,
    /// 数据大小（字节）
    size: usize,
    /// 创建时间
    created_at: DateTime<Utc>,
    /// 最后访问时间
    last_accessed: DateTime<Utc>,
    /// 访问次数
    access_count: u64,
    /// 过期时间（如果有）
    expires_at: Option<DateTime<Utc>>,
    /// 校验和
    checksum: String,
}

impl DiskCacheItemMeta {
    /// 创建新的磁盘缓存项元数据
    fn new(key: &str, file_path: &str, size: usize, checksum: &str, ttl: Option<Duration>) -> Self {
        let now = Utc::now();
        let expires_at = ttl.map(|t| now + chrono::Duration::from_std(t).unwrap_or_default());
        
        Self {
            key: key.to_string(),
            file_path: file_path.to_string(),
            size,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            expires_at,
            checksum: checksum.to_string(),
        }
    }
    
    /// 检查是否已过期
    fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            expires_at < Utc::now()
        } else {
            false
        }
    }
    
    /// 更新访问信息
    fn mark_accessed(&mut self) {
        self.last_accessed = Utc::now();
        self.access_count += 1;
    }
}

/// 磁盘缓存清单，用于持久化缓存元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiskCacheManifest {
    /// 缓存项元数据
    items: HashMap<String, DiskCacheItemMeta>,
    /// 最大缓存大小（字节）
    max_size_bytes: usize,
    /// 最大缓存项数
    max_items: usize,
    /// 当前缓存大小（字节）
    current_size_bytes: usize,
    /// 上次更新时间
    last_updated: DateTime<Utc>,
    /// 命中次数
    hits: usize,
    /// 未命中次数
    misses: usize,
}

impl DiskCacheManifest {
    /// 创建新的磁盘缓存清单
    fn new(max_size_bytes: usize, max_items: usize) -> Self {
        Self {
            items: HashMap::new(),
            max_size_bytes,
            max_items,
            current_size_bytes: 0,
            last_updated: Utc::now(),
            hits: 0,
            misses: 0,
        }
    }
    
    /// 从JSON加载缓存清单
    fn from_json(json: &str) -> Result<Self> {
        match serde_json::from_str(json) {
            Ok(manifest) => Ok(manifest),
            Err(e) => Err(Error::serialization(format!("解析缓存清单JSON失败: {}", e))),
        }
    }
    
    /// 转换为JSON
    fn to_json(&self) -> Result<String> {
        match serde_json::to_string_pretty(self) {
            Ok(json) => Ok(json),
            Err(e) => Err(Error::serialization(format!("序列化缓存清单失败: {}", e))),
        }
    }
    
    /// 添加缓存项
    fn add_item(&mut self, meta: DiskCacheItemMeta) {
        // 如果键已存在，先移除旧项
        if let Some(old_meta) = self.items.remove(&meta.key) {
            self.current_size_bytes = self.current_size_bytes.saturating_sub(old_meta.size);
        }
        
        // 添加新项
        self.current_size_bytes += meta.size;
        self.items.insert(meta.key.clone(), meta);
        self.last_updated = Utc::now();
    }
    
    /// 获取缓存项
    fn get_item(&self, key: &str) -> Option<&DiskCacheItemMeta> {
        self.items.get(key)
    }
    
    /// 获取缓存项可变引用
    fn get_item_mut(&mut self, key: &str) -> Option<&mut DiskCacheItemMeta> {
        self.items.get_mut(key)
    }
    
    /// 移除缓存项
    fn remove_item(&mut self, key: &str) -> Option<DiskCacheItemMeta> {
        let item = self.items.remove(key);
        
        if let Some(ref meta) = item {
            self.current_size_bytes = self.current_size_bytes.saturating_sub(meta.size);
            self.last_updated = Utc::now();
        }
        
        item
    }
    
    /// 清空缓存
    fn clear(&mut self) {
        self.items.clear();
        self.current_size_bytes = 0;
        self.last_updated = Utc::now();
    }
}

/// 磁盘缓存实现
pub struct DiskCache {
    /// 缓存根目录
    root_dir: PathBuf,
    /// 缓存清单（内存中保存的缓存元数据）
    manifest: RwLock<DiskCacheManifest>,
    /// 缓存索引（为了兼容性）
    pub index: RwLock<HashMap<String, DiskCacheItemMeta>>,
    /// 清单文件锁（防止并发写入）
    manifest_lock: Mutex<()>,
    /// 最大缓存大小（字节）
    pub max_size_bytes: usize,
    /// 最大缓存项数
    max_items: usize,
    /// 默认TTL
    default_ttl: Option<Duration>,
    /// 淘汰策略
    eviction_policy: EvictionPolicy,
    /// 缓存统计
    metrics: RwLock<CacheMetrics>,
}

impl DiskCache {
    /// 创建新的磁盘缓存
    pub fn new(
        max_size_bytes: usize,
        max_items: usize,
        ttl: Option<Duration>,
        eviction_policy: EvictionPolicy,
    ) -> Result<Self> {
        // 使用系统临时目录作为缓存根目录
        let mut root_dir = std::env::temp_dir();
        root_dir.push("vecmind_cache");
        
        Self::with_root_dir(root_dir, max_size_bytes, max_items, ttl, eviction_policy)
    }
    
    /// 使用指定根目录创建磁盘缓存
    pub fn with_root_dir(
        root_dir: PathBuf,
        max_size_bytes: usize,
        max_items: usize,
        ttl: Option<Duration>,
        eviction_policy: EvictionPolicy,
    ) -> Result<Self> {
        // 确保缓存目录存在
        if !root_dir.exists() {
            fs::create_dir_all(&root_dir)
                .map_err(|e| Error::io(format!("创建缓存目录失败: {}", e)))?;
        }
        
        // 尝试加载现有清单
        let manifest = if let Ok(manifest) = Self::load_manifest(&root_dir) {
            manifest
        } else {
            // 无法加载或不存在，创建新的清单
            let manifest = DiskCacheManifest::new(max_size_bytes, max_items);
            // 保存新清单
            if let Err(e) = Self::save_manifest(&root_dir, &manifest) {
                warn!("保存缓存清单失败: {}", e);
            }
            manifest
        };
        
        // 创建缓存
        let cache = Self {
            root_dir,
            manifest: RwLock::new(manifest),
            index: RwLock::new(HashMap::new()),
            manifest_lock: Mutex::new(()),
            max_size_bytes,
            max_items,
            default_ttl: ttl,
            eviction_policy,
            metrics: RwLock::new(CacheMetrics::new()),
        };
        
        // 同步索引
        cache.sync_index()?;
        
        Ok(cache)
    }
    
    /// 从缓存获取值
    pub fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let start = Instant::now();
        let mut metrics = self.metrics.write().map_err(|_| Error::locks_poison("磁盘缓存指标锁被污染"))?;
        
        // 从清单中获取元数据
        let mut manifest = self.manifest.write().map_err(|_| Error::locks_poison("磁盘缓存清单锁被污染"))?;
        
        if let Some(meta) = manifest.get_item(key) {
            // 检查是否过期
            if meta.is_expired() {
                // 项已过期，记为未命中
                manifest.misses += 1;
                metrics.record_miss(start.elapsed().as_micros() as u64);
                drop(manifest);
                
                // 稍后删除过期项
                let _ = self.delete(key);
                
                return Ok(None);
            }
            
            // 尝试读取文件
            let file_path = self.get_item_path(&meta.file_path);
            match fs::read(&file_path) {
                Ok(data) => {
                    // 验证校验和
                    let checksum = Self::calculate_checksum(&data);
                    if checksum != meta.checksum {
                        warn!("缓存项校验和不匹配，可能已损坏: {}", key);
                        manifest.misses += 1;
                        metrics.record_miss(start.elapsed().as_micros() as u64);
                        drop(manifest);
                        
                        // 删除损坏的项
                        let _ = self.delete(key);
                        
                        return Ok(None);
                    }
                    
                    // 更新访问信息
                    if let Some(meta) = manifest.get_item_mut(key) {
                        meta.mark_accessed();
                    }
                    
                    // 更新统计信息
                    manifest.hits += 1;
                    metrics.record_hit(start.elapsed().as_micros() as u64);
                    
                    // 保存更新后的清单
                    drop(manifest);
                    let _ = self.save_manifest();
                    
                    Ok(Some(data))
                },
                Err(e) => {
                    warn!("读取缓存文件失败: {}, {}", file_path.display(), e);
                    // 文件不存在或读取失败，记为未命中
                    manifest.misses += 1;
                    metrics.record_miss(start.elapsed().as_micros() as u64);
                    drop(manifest);
                    
                    // 删除无效的缓存项
                    let _ = self.delete(key);
                    
                    Ok(None)
                }
            }
        } else {
            // 未命中
            manifest.misses += 1;
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
        
        let mut metrics = self.metrics.write().map_err(|_| Error::locks_poison("磁盘缓存指标锁被污染"))?;
        
        // 计算校验和
        let checksum = Self::calculate_checksum(value);
        
        // 生成文件路径
        let file_path = self.generate_file_path(key, &checksum);
        let full_path = self.get_item_path(&file_path);
        
        // 确保父目录存在
        if let Some(parent) = full_path.parent() {
            if !parent.exists() {
                fs::create_dir_all(parent)
                    .map_err(|e| Error::io(format!("创建缓存目录失败: {}", e)))?;
            }
        }
        
        // 写入文件
        fs::write(&full_path, value)
            .map_err(|e| Error::io(format!("写入缓存文件失败: {}", e)))?;
        
        // 更新清单
        let mut manifest = self.manifest.write().map_err(|_| Error::locks_poison("磁盘缓存清单锁被污染"))?;
        
        // 检查当前大小
        let old_size = manifest.get_item(key).map(|m| m.size).unwrap_or(0);
        let size_delta = value_len as isize - old_size as isize;
        
        if size_delta > 0 && manifest.current_size_bytes as isize + size_delta > self.max_size_bytes as isize {
            // 需要淘汰一些项目
            drop(manifest);
            self.evict(size_delta as usize)?;
            manifest = self.manifest.write().map_err(|_| Error::locks_poison("磁盘缓存清单锁被污染"))?;
        }
        
        // 创建元数据
        let meta = DiskCacheItemMeta::new(
            key,
            &file_path,
            value_len,
            &checksum,
            self.default_ttl,
        );
        
        // 添加到清单
        manifest.add_item(meta.clone());
        
        // 同步索引
        {
            let mut index = self.index.write().map_err(|_| Error::locks_poison("磁盘缓存索引锁被污染"))?;
            index.insert(key.to_string(), meta);
        }
        
        // 更新指标
        metrics.record_write(value_len);
        
        // 检查条目数量限制
        if manifest.items.len() > self.max_items {
            drop(manifest);
            self.evict_by_count(manifest.items.len() - self.max_items)?;
        } else {
            // 保存清单
            drop(manifest);
            self.save_manifest()?;
        }
        
        Ok(())
    }
    
    /// 删除缓存值
    pub fn delete(&self, key: &str) -> Result<bool> {
        let mut metrics = self.metrics.write().map_err(|_| Error::locks_poison("磁盘缓存指标锁被污染"))?;
        
        let mut manifest = self.manifest.write().map_err(|_| Error::locks_poison("磁盘缓存清单锁被污染"))?;
        
        if let Some(meta) = manifest.remove_item(key) {
            // 同步索引
            {
                let mut index = self.index.write().map_err(|_| Error::locks_poison("磁盘缓存索引锁被污染"))?;
                index.remove(key);
            }
            
            // 删除文件
            let file_path = self.get_item_path(&meta.file_path);
            if let Err(e) = fs::remove_file(&file_path) {
                // 如果文件不存在，不视为错误
                if e.kind() != io::ErrorKind::NotFound {
                    warn!("删除缓存文件失败: {}, {}", file_path.display(), e);
                }
            }
            
            // 尝试删除空目录
            if let Some(parent) = file_path.parent() {
                if parent != self.root_dir {
                    // 忽略错误，可能目录不为空
                    let _ = fs::remove_dir(parent);
                }
            }
            
            // 更新指标
            metrics.record_delete(meta.size);
            
            // 保存清单
            drop(manifest);
            self.save_manifest()?;
            
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    /// 清空缓存
    pub fn clear(&self) -> Result<()> {
        let mut metrics = self.metrics.write().map_err(|_| Error::locks_poison("磁盘缓存指标锁被污染"))?;
        
        // 获取缓存项元数据
        let mut manifest = self.manifest.write().map_err(|_| Error::locks_poison("磁盘缓存清单锁被污染"))?;
        let items: Vec<DiskCacheItemMeta> = manifest.items.values().cloned().collect();
        
        // 清空清单
        manifest.clear();
        drop(manifest);
        
        // 同步索引
        {
            let mut index = self.index.write().map_err(|_| Error::locks_poison("磁盘缓存索引锁被污染"))?;
            index.clear();
        }
        
        // 删除所有缓存文件
        for meta in items {
            let file_path = self.get_item_path(&meta.file_path);
            let _ = fs::remove_file(file_path);
        }
        
        // 尝试重新创建缓存目录结构
        let _ = fs::create_dir_all(&self.root_dir);
        
        // 保存清单
        self.save_manifest()?;
        
        // 重置指标（但保留历史统计）
        metrics.entries = 0;
        metrics.size_bytes = 0;
        metrics.last_updated = Utc::now();
        
        Ok(())
    }
    
    /// 清理过期项
    pub fn cleanup_expired(&self) -> Result<usize> {
        let mut cleaned_count = 0;
        
        // 获取过期项
        let mut manifest = self.manifest.write().map_err(|_| Error::locks_poison("磁盘缓存清单锁被污染"))?;
        let expired_keys: Vec<String> = manifest.items.values()
            .filter(|meta| meta.is_expired())
            .map(|meta| meta.key.clone())
            .collect();
        
        drop(manifest);
        
        // 删除过期项
        for key in expired_keys {
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
        
        // 获取淘汰候选项
        let candidates = match self.eviction_policy {
            EvictionPolicy::LRU => self.get_lru_candidates(required_space),
            EvictionPolicy::MRU => self.get_mru_candidates(required_space),
            EvictionPolicy::LFU => self.get_lfu_candidates(required_space),
            EvictionPolicy::FIFO => self.get_fifo_candidates(required_space),
        }?;
        
        // 淘汰候选项
        for key in candidates {
            if evicted_size >= required_space {
                break;
            }
            
            let manifest = self.manifest.read().map_err(|_| Error::locks_poison("磁盘缓存清单锁被污染"))?;
            let size = manifest.get_item(&key).map(|m| m.size).unwrap_or(0);
            drop(manifest);
            
            if let Ok(true) = self.delete(&key) {
                evicted_count += 1;
                evicted_size += size;
            }
        }
        
        // 更新指标
        if evicted_count > 0 {
            let mut metrics = self.metrics.write().map_err(|_| Error::locks_poison("磁盘缓存指标锁被污染"))?;
            metrics.record_eviction(evicted_count, evicted_size);
        }
        
        Ok(evicted_count)
    }
    
    /// 根据数量淘汰缓存项
    fn evict_by_count(&self, count: usize) -> Result<usize> {
        let mut evicted_count = 0;
        let mut evicted_size = 0;
        
        // 获取淘汰候选项
        let candidates = match self.eviction_policy {
            EvictionPolicy::LRU => self.get_lru_candidates(usize::MAX)?,
            EvictionPolicy::MRU => self.get_mru_candidates(usize::MAX)?,
            EvictionPolicy::LFU => self.get_lfu_candidates(usize::MAX)?,
            EvictionPolicy::FIFO => self.get_fifo_candidates(usize::MAX)?,
        };
        
        // 限制候选项数量
        let candidates = candidates.into_iter().take(count).collect::<Vec<_>>();
        
        // 淘汰候选项
        for key in candidates {
            if evicted_count >= count {
                break;
            }
            
            let manifest = self.manifest.read().map_err(|_| Error::locks_poison("磁盘缓存清单锁被污染"))?;
            let size = manifest.get_item(&key).map(|m| m.size).unwrap_or(0);
            drop(manifest);
            
            if let Ok(true) = self.delete(&key) {
                evicted_count += 1;
                evicted_size += size;
            }
        }
        
        // 更新指标
        if evicted_count > 0 {
            let mut metrics = self.metrics.write().map_err(|_| Error::locks_poison("磁盘缓存指标锁被污染"))?;
            metrics.record_eviction(evicted_count, evicted_size);
        }
        
        Ok(evicted_count)
    }
    
    /// 获取LRU淘汰候选项
    fn get_lru_candidates(&self, required_space: usize) -> Result<Vec<String>> {
        let manifest = self.manifest.read().map_err(|_| Error::locks_poison("磁盘缓存清单锁被污染"))?;
        
        // 按最后访问时间排序
        let mut items: Vec<(&String, &DiskCacheItemMeta)> = manifest.items.iter().collect();
        items.sort_by(|a, b| a.1.last_accessed.cmp(&b.1.last_accessed));
        
        // 估计需要淘汰的项数
        let avg_size = if !manifest.items.is_empty() {
            manifest.current_size_bytes / manifest.items.len()
        } else {
            return Ok(Vec::new());
        };
        
        let candidate_count = (required_space / avg_size).max(1) + 2; // 加上余量
        
        // 返回候选项
        let candidates = items.into_iter()
            .take(candidate_count)
            .map(|(k, _)| k.clone())
            .collect();
        
        Ok(candidates)
    }
    
    /// 获取MRU淘汰候选项
    fn get_mru_candidates(&self, required_space: usize) -> Result<Vec<String>> {
        let manifest = self.manifest.read().map_err(|_| Error::locks_poison("磁盘缓存清单锁被污染"))?;
        
        // 按最后访问时间排序（逆序）
        let mut items: Vec<(&String, &DiskCacheItemMeta)> = manifest.items.iter().collect();
        items.sort_by(|a, b| b.1.last_accessed.cmp(&a.1.last_accessed));
        
        // 估计需要淘汰的项数
        let avg_size = if !manifest.items.is_empty() {
            manifest.current_size_bytes / manifest.items.len()
        } else {
            return Ok(Vec::new());
        };
        
        let candidate_count = (required_space / avg_size).max(1) + 2; // 加上余量
        
        // 返回候选项
        let candidates = items.into_iter()
            .take(candidate_count)
            .map(|(k, _)| k.clone())
            .collect();
        
        Ok(candidates)
    }
    
    /// 获取LFU淘汰候选项
    fn get_lfu_candidates(&self, required_space: usize) -> Result<Vec<String>> {
        let manifest = self.manifest.read().map_err(|_| Error::locks_poison("磁盘缓存清单锁被污染"))?;
        
        // 按访问次数排序
        let mut items: Vec<(&String, &DiskCacheItemMeta)> = manifest.items.iter().collect();
        items.sort_by(|a, b| a.1.access_count.cmp(&b.1.access_count));
        
        // 估计需要淘汰的项数
        let avg_size = if !manifest.items.is_empty() {
            manifest.current_size_bytes / manifest.items.len()
        } else {
            return Ok(Vec::new());
        };
        
        let candidate_count = (required_space / avg_size).max(1) + 2; // 加上余量
        
        // 返回候选项
        let candidates = items.into_iter()
            .take(candidate_count)
            .map(|(k, _)| k.clone())
            .collect();
        
        Ok(candidates)
    }
    
    /// 获取FIFO淘汰候选项
    fn get_fifo_candidates(&self, required_space: usize) -> Result<Vec<String>> {
        let manifest = self.manifest.read().map_err(|_| Error::locks_poison("磁盘缓存清单锁被污染"))?;
        
        // 按创建时间排序
        let mut items: Vec<(&String, &DiskCacheItemMeta)> = manifest.items.iter().collect();
        items.sort_by(|a, b| a.1.created_at.cmp(&b.1.created_at));
        
        // 估计需要淘汰的项数
        let avg_size = if !manifest.items.is_empty() {
            manifest.current_size_bytes / manifest.items.len()
        } else {
            return Ok(Vec::new());
        };
        
        let candidate_count = (required_space / avg_size).max(1) + 2; // 加上余量
        
        // 返回候选项
        let candidates = items.into_iter()
            .take(candidate_count)
            .map(|(k, _)| k.clone())
            .collect();
        
        Ok(candidates)
    }
    
    /// 强制淘汰指定数量的缓存项
    pub fn evict_entries(&self, count: usize) -> Result<usize> {
        if count == 0 {
            return Ok(0);
        }
        
        self.evict_by_count(count)
    }
    
    /// 同步索引和清单
    fn sync_index(&self) -> Result<()> {
        let manifest = self.manifest.read().map_err(|_| Error::locks_poison("磁盘缓存清单锁被污染"))?;
        let mut index = self.index.write().map_err(|_| Error::locks_poison("磁盘缓存索引锁被污染"))?;
        
        index.clear();
        for (key, meta) in &manifest.items {
            index.insert(key.clone(), meta.clone());
        }
        
        Ok(())
    }
    
    /// 获取缓存统计信息
    pub fn get_metrics(&self) -> Result<CacheMetrics> {
        let metrics = self.metrics.read().map_err(|_| Error::locks_poison("磁盘缓存指标锁被污染"))?;
        Ok(metrics.clone())
    }
    
    /// 获取缓存项完整路径
    fn get_item_path(&self, relative_path: &str) -> PathBuf {
        let mut path = self.root_dir.clone();
        path.push(relative_path);
        path
    }
    
    /// 生成文件路径
    fn generate_file_path(&self, key: &str, checksum: &str) -> String {
        // 使用键的哈希作为目录名，避免文件名过长和特殊字符问题
        let key_hash = Self::hash_string(key);
        
        // 使用前两个字符作为子目录，避免目录过大
        let subdir = &key_hash[0..2];
        
        format!("{}/{}-{}", subdir, key_hash, checksum)
    }
    
    /// 计算数据校验和
    fn calculate_checksum(data: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        
        // 使用短校验和（前8字节）
        format!("{:x}", &result[0..8].iter().fold(0u32, |acc, &x| (acc << 8) | x as u32))
    }
    
    /// 对字符串进行哈希
    fn hash_string(s: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(s.as_bytes());
        let result = hasher.finalize();
        
        // 使用短哈希（前16字节）
        format!("{:x}", &result[0..16].iter().fold(0u64, |acc, &x| (acc << 8) | x as u64))
    }
    
    /// 加载清单
    fn load_manifest(root_dir: &Path) -> Result<DiskCacheManifest> {
        let manifest_path = root_dir.join("manifest.json");
        
        if manifest_path.exists() {
            match fs::read_to_string(&manifest_path) {
                Ok(json) => DiskCacheManifest::from_json(&json),
                Err(e) => Err(Error::io(format!("读取缓存清单失败: {}", e))),
            }
        } else {
            Err(Error::not_found("缓存清单文件不存在"))
        }
    }
    
    /// 保存清单
    fn save_manifest(&self) -> Result<()> {
        // 获取锁，防止并发写入
        let _lock = self.manifest_lock.lock().map_err(|_| Error::locks_poison("磁盘缓存清单文件锁被污染"))?;
        
        let manifest_path = self.root_dir.join("manifest.json");
        let manifest = self.manifest.read().map_err(|_| Error::locks_poison("磁盘缓存清单锁被污染"))?;
        
        // 先写入临时文件
        let temp_path = manifest_path.with_extension("json.tmp");
        let json = manifest.to_json()?;
        
        fs::write(&temp_path, json)
            .map_err(|e| Error::io(format!("写入临时缓存清单失败: {}", e)))?;
        
        // 重命名为正式文件
        fs::rename(&temp_path, &manifest_path)
            .map_err(|e| Error::io(format!("重命名缓存清单失败: {}", e)))?;
        
        Ok(())
    }
    

} 