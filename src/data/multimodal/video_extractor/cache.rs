//! 视频特征提取器缓存模块
//!
//! 本模块提供了特征结果缓存管理功能，用于提高性能

use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};
use super::types::{VideoFeatureResult, VideoFeatureType};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use log;
use bincode;
use log::{info};
// use std::time::Instant; // not used in cache logic here
use std::path::Path;
use crate::data::multimodal::video_extractor::{VideoExtractionError, VideoMetadata};
use crate::data::multimodal::video_extractor::extractor::VideoFeatureExtractor as ExtractorTrait;

// 定义Result类型别名
type Result<T> = std::result::Result<T, VideoExtractionError>;

/// 特征缓存
#[derive(Debug)]
pub struct FeatureCache {
    /// 缓存映射
    cache: HashMap<String, CacheEntry>,
    /// 缓存队列(LRU)
    queue: VecDeque<String>,
    /// 最大缓存条目数
    max_entries: usize,
    /// 命中次数
    hit_count: usize,
    /// 未命中次数
    miss_count: usize,
    /// 是否启用
    enabled: bool,
}

/// 缓存条目
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct CacheEntry {
    /// 缓存键
    key: String,
    /// 特征结果
    result: VideoFeatureResult,
    /// 创建时间戳
    created_at: u64,
    /// 上次访问时间戳
    last_accessed: u64,
    /// 访问次数
    access_count: usize,
}

impl FeatureCache {
    /// 创建新的特征缓存
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_entries),
            queue: VecDeque::with_capacity(max_entries),
            max_entries,
            hit_count: 0,
            miss_count: 0,
            enabled: true,
        }
    }
    
    /// 从缓存获取特征
    pub fn get(&mut self, key: &str) -> Option<VideoFeatureResult> {
        if !self.enabled {
            return None;
        }
        
        if let Some(entry) = self.cache.get_mut(key) {
            // 更新访问时间和计数
            entry.last_accessed = current_timestamp();
            entry.access_count += 1;
            
            // 更新LRU队列（移动到队尾）
            if let Some(pos) = self.queue.iter().position(|k| k == key) {
                self.queue.remove(pos);
                self.queue.push_back(key.to_string());
            }
            
            self.hit_count += 1;
            Some(entry.result.clone())
        } else {
            self.miss_count += 1;
            None
        }
    }
    
    /// 添加特征到缓存
    pub fn put(&mut self, key: &str, result: VideoFeatureResult) {
        if !self.enabled {
            return;
        }
        
        let now = current_timestamp();
        
        // 如果已存在，更新它
        if let Some(entry) = self.cache.get_mut(key) {
            entry.result = result;
            entry.last_accessed = now;
            entry.access_count += 1;
            
            // 更新LRU队列
            if let Some(pos) = self.queue.iter().position(|k| k == key) {
                self.queue.remove(pos);
                self.queue.push_back(key.to_string());
            }
            
            return;
        }
        
        // 如果缓存已满，移除最久未使用的条目
        if self.cache.len() >= self.max_entries {
            if let Some(old_key) = self.queue.pop_front() {
                self.cache.remove(&old_key);
            }
        }
        
        // 添加新条目
        let entry = CacheEntry {
            key: key.to_string(),
            result,
            created_at: now,
            last_accessed: now,
            access_count: 1,
        };
        
        self.cache.insert(key.to_string(), entry);
        self.queue.push_back(key.to_string());
    }
    
    /// 清空缓存
    pub fn clear(&mut self) {
        self.cache.clear();
        self.queue.clear();
    }
    
    /// 获取缓存统计信息
    pub fn get_stats(&self) -> CacheStats {
        let total_requests = self.hit_count + self.miss_count;
        let hit_rate = if total_requests > 0 {
            self.hit_count as f64 / total_requests as f64
        } else {
            0.0
        };
        
        CacheStats {
            size: self.cache.len(),
            capacity: self.max_entries,
            hit_count: self.hit_count,
            miss_count: self.miss_count,
            hit_rate,
            enabled: self.enabled,
        }
    }
    
    /// 启用缓存
    pub fn enable(&mut self) {
        self.enabled = true;
    }
    
    /// 禁用缓存
    pub fn disable(&mut self) {
        self.enabled = false;
    }
    
    /// 获取按特征类型过滤的结果
    pub fn get_by_feature_type(&self, feature_type: &VideoFeatureType) -> Vec<VideoFeatureResult> {
        self.cache.values()
            .filter(|entry| entry.result.feature_type == *feature_type)
            .map(|entry| entry.result.clone())
            .collect()
    }
    
    /// 移除过期条目
    pub fn remove_expired(&mut self, max_age_seconds: u64) {
        let now = current_timestamp();
        let expired_keys: Vec<String> = self.cache.values()
            .filter(|entry| now - entry.created_at > max_age_seconds)
            .map(|entry| entry.key.clone())
            .collect();
        
        for key in &expired_keys {
            self.cache.remove(key);
            if let Some(pos) = self.queue.iter().position(|k| k == key) {
                self.queue.remove(pos);
            }
        }
    }
    
    /// 缓存预热
    pub fn warmup<P: AsRef<Path>>(&mut self, video_paths: &[P], extractor: &mut dyn ExtractorTrait, config: &super::config::VideoFeatureConfig) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        
        info!("开始缓存预热，视频数量: {}", video_paths.len());
        let start_time = std::time::Instant::now();
        let mut success_count = 0;
        let mut failed_count = 0;
        
        // 获取预热的特征类型
        let feature_types = config.feature_types.clone();
        
        // 创建线程池
        let thread_count = std::cmp::min(
            config.parallel_threads,
            video_paths.len()
        );
        let pool = ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build()
            .map_err(|e| VideoExtractionError::SystemError(
                format!("无法创建线程池: {}", e)
            ))?;
        
        // 并行预热
        let config_clone = config.clone();
        // 将路径转换为字符串向量以便并行处理
        let path_strings: Vec<String> = video_paths.iter()
            .map(|p| p.as_ref().to_string_lossy().to_string())
            .collect();
        
        let results: Vec<(String, std::result::Result<VideoFeatureResult, VideoExtractionError>)> = pool.install(|| {
            path_strings.par_iter()
                .map(|path_str| {
                    let path = std::path::Path::new(path_str);
                    let key = generate_cache_key(path, &feature_types[0]);
                    
                    // 检查缓存中是否已存在
                    if self.get(&key).is_some() {
                        return (key, Ok(VideoFeatureResult::default())); // 已在缓存中
                    }
                    
                    // 提取特征并添加到缓存
                    match extractor.extract_features(path, &config_clone) {
                        Ok(feature) => {
                            // 将VideoFeature转换为VideoFeatureResult
                            let result = VideoFeatureResult {
                                feature_type: feature.feature_type,
                                features: feature.features.clone(),
                                metadata: feature.metadata.clone(),
                                processing_info: None,
                                dimensions: feature.dimensions,
                                timestamp: feature.timestamp,
                            };
                            (key, Ok(result))
                        },
                        Err(err) => {
                            (key, Err(err))
                        }
                    }
                })
                .collect()
        });
        
        // 统计结果
        for (_key, result) in results {
            match result {
                Ok(_) => success_count += 1,
                Err(_) => failed_count += 1,
            }
        }
        
        let elapsed = start_time.elapsed();
        info!("缓存预热完成: 成功 {} 个, 失败 {} 个, 耗时 {:.2}s",
            success_count, failed_count, elapsed.as_secs_f64());
        
        Ok(())
    }
    
    /// 设置自适应缓存策略
    pub fn set_adaptive_strategy(&mut self, strategy: AdaptiveCacheStrategy) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        
        info!("设置自适应缓存策略: {:?}", strategy);
        
        // 根据策略重新组织缓存
        match strategy {
            AdaptiveCacheStrategy::Frequency => {
                // 按访问频率重新排序
                let mut entries: Vec<_> = self.cache.values().cloned().collect();
                entries.sort_by(|a, b| b.access_count.cmp(&a.access_count));
                
                // 清空当前缓存和队列
                self.queue.clear();
                
                // 保留访问频率最高的条目
                for entry in entries.iter().take(self.max_entries) {
                    self.queue.push_back(entry.key.clone());
                }
                
                // 移除不在队列中的条目
                let keys_to_keep: std::collections::HashSet<_> = self.queue.iter().cloned().collect();
                self.cache.retain(|k, _| keys_to_keep.contains(k));
                
                info!("已按访问频率优化缓存，保留 {} 个条目", self.cache.len());
            },
            AdaptiveCacheStrategy::Recency => {
                // 已经是LRU策略，不需要额外操作
                info!("使用最近访问时间(LRU)策略，当前缓存条目数: {}", self.cache.len());
            },
            AdaptiveCacheStrategy::Size => {
                // 按特征大小重新排序（保留小特征）
                let mut entries: Vec<_> = self.cache.values().cloned().collect();
                entries.sort_by(|a, b| a.result.features.len().cmp(&b.result.features.len()));
                
                // 清空当前队列
                self.queue.clear();
                
                // 保留特征大小最小的条目
                for entry in entries.iter().take(self.max_entries) {
                    self.queue.push_back(entry.key.clone());
                }
                
                // 移除不在队列中的条目
                let keys_to_keep: std::collections::HashSet<_> = self.queue.iter().cloned().collect();
                self.cache.retain(|k, _| keys_to_keep.contains(k));
                
                info!("已按特征大小优化缓存，保留 {} 个条目", self.cache.len());
            },
            AdaptiveCacheStrategy::Type(feature_type) => {
                // 优先保留指定特征类型的条目
                let mut entries: Vec<_> = self.cache.values().cloned().collect();
                entries.sort_by(|a, b| {
                    // 首先按特征类型排序
                    let a_match = a.result.feature_type == feature_type;
                    let b_match = b.result.feature_type == feature_type;
                    
                    if a_match && !b_match {
                        std::cmp::Ordering::Less
                    } else if !a_match && b_match {
                        std::cmp::Ordering::Greater
                    } else {
                        // 在特征类型相同的情况下，按访问次数排序
                        b.access_count.cmp(&a.access_count)
                    }
                });
                
                // 清空当前队列
                self.queue.clear();
                
                // 保留排序后的条目
                for entry in entries.iter().take(self.max_entries) {
                    self.queue.push_back(entry.key.clone());
                }
                
                // 移除不在队列中的条目
                let keys_to_keep: std::collections::HashSet<_> = self.queue.iter().cloned().collect();
                self.cache.retain(|k, _| keys_to_keep.contains(k));
                
                info!("已按特征类型 {:?} 优化缓存，保留 {} 个条目", feature_type, self.cache.len());
            },
            AdaptiveCacheStrategy::Hybrid => {
                // 混合策略: 结合访问频率、最近使用时间和特征大小
                let now = current_timestamp();
                let mut entries: Vec<_> = self.cache.values().cloned().collect();
                
                // 计算混合得分: 0.5 * 访问频率 + 0.3 * 最近度 + 0.2 * 逆大小因子
                entries.sort_by(|a, b| {
                    // 访问频率分数
                    let a_freq_score = a.access_count as f64;
                    let b_freq_score = b.access_count as f64;
                    
                    // 最近度分数 (反比于上次访问的时间差)
                    let a_recency_score = 1.0 / (now - a.last_accessed + 1) as f64;
                    let b_recency_score = 1.0 / (now - b.last_accessed + 1) as f64;
                    
                    // 大小因子 (反比于特征大小)
                    let a_size_score = 1.0 / (a.result.features.len() as f64 + 1.0);
                    let b_size_score = 1.0 / (b.result.features.len() as f64 + 1.0);
                    
                    // 混合得分
                    let a_score = 0.5 * a_freq_score + 0.3 * a_recency_score + 0.2 * a_size_score;
                    let b_score = 0.5 * b_freq_score + 0.3 * b_recency_score + 0.2 * b_size_score;
                    
                    // 按得分降序排序
                    b_score.partial_cmp(&a_score).unwrap_or(std::cmp::Ordering::Equal)
                });
                
                // 清空当前队列
                self.queue.clear();
                
                // 保留得分最高的条目
                for entry in entries.iter().take(self.max_entries) {
                    self.queue.push_back(entry.key.clone());
                }
                
                // 移除不在队列中的条目
                let keys_to_keep: std::collections::HashSet<_> = self.queue.iter().cloned().collect();
                self.cache.retain(|k, _| keys_to_keep.contains(k));
                
                info!("已使用混合策略优化缓存，保留 {} 个条目", self.cache.len());
            }
        }
        
        Ok(())
    }
    
    /// 按特征类型批量清除
    pub fn clear_by_type(&mut self, feature_type: &VideoFeatureType) -> usize {
        if !self.enabled {
            return 0;
        }
        
        let keys_to_remove: Vec<_> = self.cache.iter()
            .filter(|(_, v)| v.result.feature_type == *feature_type)
            .map(|(k, _)| k.clone())
            .collect();
        
        let count = keys_to_remove.len();
        
        for key in keys_to_remove {
            self.cache.remove(&key);
            if let Some(pos) = self.queue.iter().position(|k| k == &key) {
                self.queue.remove(pos);
            }
        }
        
        count
    }
    
    /// 获取缓存使用情况
    pub fn get_detailed_stats(&self) -> DetailedCacheStats {
        let mut type_counts = HashMap::new();
        let mut avg_hit_per_entry = 0.0;
        let mut avg_size = 0.0;
        let mut min_size = usize::MAX;
        let mut max_size = 0;
        
        if !self.cache.is_empty() {
            for entry in self.cache.values() {
                // 统计特征类型
                let count = type_counts.entry(entry.result.feature_type.clone())
                    .or_insert(0);
                *count += 1;
                
                // 计算特征大小
                let size = entry.result.features.len();
                avg_size += size as f64;
                min_size = min_size.min(size);
                max_size = max_size.max(size);
                
                // 统计访问次数
                avg_hit_per_entry += entry.access_count as f64;
            }
            
            avg_size /= self.cache.len() as f64;
            avg_hit_per_entry /= self.cache.len() as f64;
        } else {
            min_size = 0;
        }
        
        let total_memory_usage = self.estimate_memory_usage();
        
        DetailedCacheStats {
            basic_stats: self.get_stats(),
            type_distribution: type_counts,
            avg_hit_per_entry,
            avg_feature_size: avg_size as usize,
            min_feature_size: min_size,
            max_feature_size: max_size,
            memory_usage_bytes: total_memory_usage,
        }
    }
    
    /// 估计缓存占用内存
    pub fn estimate_memory_usage(&self) -> usize {
        let mut total_bytes = 0;
        
        // 缓存项内存
        for entry in self.cache.values() {
            // 基础结构大小
            let entry_base_size = std::mem::size_of::<CacheEntry>();
            
            // 键大小
            let key_size = entry.key.capacity();
            
            // 结果大小
            let result_size = std::mem::size_of::<VideoFeatureResult>();
            
            // 特征向量大小
            let features_size = entry.result.features.capacity() * std::mem::size_of::<f32>();
            
            // 元数据大小（如果有）
            let metadata_size = if let Some(ref metadata) = entry.result.metadata {
                std::mem::size_of::<VideoMetadata>() + 
                metadata.file_path.capacity() + 
                metadata.codec.capacity() + 
                if let Some(ref custom) = metadata.custom_metadata {
                    custom.iter().map(|(k, v)| k.capacity() + v.capacity()).sum::<usize>()
                } else {
                    0
                }
            } else {
                0
            };
            
            // 合计
            total_bytes += entry_base_size + key_size + result_size + features_size + metadata_size;
        }
        
        // 队列内存
        let queue_capacity = self.queue.capacity() * std::mem::size_of::<String>();
        let queue_content = self.queue.iter().map(|k| k.capacity()).sum::<usize>();
        
        total_bytes += queue_capacity + queue_content;
        
        // HashMap开销（估计）
        let hashmap_overhead = self.cache.capacity() * std::mem::size_of::<(String, CacheEntry)>();
        total_bytes += hashmap_overhead;
        
        total_bytes
    }
    
    /// 从文件加载缓存
    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<usize> {
        use std::io::BufReader;
        
        if !self.enabled {
            return Ok(0);
        }
        
        if !path.as_ref().exists() {
            return Err(VideoExtractionError::FileError(
                format!("缓存文件不存在: {}", path.as_ref().display())
            ));
        }
        
        let file = std::fs::File::open(path.as_ref())
            .map_err(|e| VideoExtractionError::FileError(
                format!("无法打开缓存文件: {}", e)
            ))?;
        
        let reader = BufReader::new(file);
        let entries: Vec<(String, CacheEntry)> = bincode::deserialize_from(reader)
            .map_err(|e| VideoExtractionError::CacheError(
                format!("无法反序列化缓存: {}", e)
            ))?;
        
        if entries.is_empty() {
            info!("加载的缓存文件为空");
            return Ok(0);
        }
        
        // 清空当前缓存
        self.cache.clear();
        self.queue.clear();
        
        // 加载新的缓存条目
        let mut loaded_count = 0;
        for (key, entry) in entries {
            if self.cache.len() >= self.max_entries {
                break;
            }
            
            self.cache.insert(key.clone(), entry);
            self.queue.push_back(key);
            loaded_count += 1;
        }
        
        info!("从文件加载了 {} 条缓存条目", loaded_count);
        Ok(loaded_count)
    }
    
    /// 缓存保存
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<usize> {
        if !self.enabled || self.cache.is_empty() {
            return Ok(0);
        }
        
        let entries: Vec<(String, CacheEntry)> = self.cache.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        
        let file = std::fs::File::create(path.as_ref())
            .map_err(|e| VideoExtractionError::FileError(
                format!("无法创建缓存文件: {}", e)
            ))?;
        
        let writer = std::io::BufWriter::new(file);
        bincode::serialize_into(writer, &entries)
            .map_err(|e| VideoExtractionError::CacheError(
                format!("无法序列化缓存: {}", e)
            ))?;
        
        info!("保存了 {} 条缓存条目到文件", entries.len());
        Ok(entries.len())
    }
}

/// 缓存统计信息
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// 当前大小
    pub size: usize,
    /// 容量
    pub capacity: usize,
    /// 命中次数
    pub hit_count: usize,
    /// 未命中次数
    pub miss_count: usize,
    /// 命中率
    pub hit_rate: f64,
    /// 是否启用
    pub enabled: bool,
}

/// 获取当前时间戳
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 生成缓存键
fn generate_cache_key<P: AsRef<Path>>(path: P, feature_type: &VideoFeatureType) -> String {
    let path_str = path.as_ref().to_string_lossy();
    let metadata = std::fs::metadata(path.as_ref()).ok();
    
    let mod_time = if let Some(meta) = metadata {
        meta.modified().ok().map(|time| {
            time.duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        }).unwrap_or(0)
    } else {
        0
    };
    
    format!("{}:{}:{:?}", path_str, mod_time, feature_type)
}

/// 详细缓存统计
#[derive(Debug, Clone)]
pub struct DetailedCacheStats {
    /// 基本统计信息
    pub basic_stats: CacheStats,
    /// 按特征类型分布
    pub type_distribution: HashMap<VideoFeatureType, usize>,
    /// 平均每条目命中次数
    pub avg_hit_per_entry: f64,
    /// 平均特征大小
    pub avg_feature_size: usize,
    /// 最小特征大小
    pub min_feature_size: usize,
    /// 最大特征大小
    pub max_feature_size: usize,
    /// 估计内存占用(字节)
    pub memory_usage_bytes: usize,
}

/// 适应性缓存策略
#[derive(Debug, Clone)]
pub enum AdaptiveCacheStrategy {
    /// 按访问频率
    Frequency,
    /// 按最近使用
    Recency,
    /// 按大小
    Size,
    /// 按特征类型
    Type(VideoFeatureType),
    /// 混合策略
    Hybrid,
} 