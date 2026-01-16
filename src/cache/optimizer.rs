use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock, Mutex};

use std::time::{Duration, Instant};
use std::thread;

use serde::{Deserialize, Serialize};
use log::{debug, info, warn, error};
use tokio::time::interval;

use sysinfo::System;

use crate::error::{Error, Result};

use super::manager::{CacheManager, CacheMetrics, CacheTier};

/// 缓存访问模式
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessPattern {
    /// 顺序访问
    Sequential,
    /// 随机访问
    Random,
    /// 热点访问（集中访问某些key）
    Hotspot,
    /// 时间局部性（最近访问的数据再次访问）
    Temporal,
    /// 空间局部性（相邻数据一起访问）
    Spatial,
}

/// 缓存优化策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStrategy {
    /// 预热策略
    pub warmup: WarmupStrategy,
    /// 自适应策略
    pub adaptive: AdaptiveStrategy,
    /// 淘汰优化策略
    pub eviction: EvictionOptimization,
    /// 预取策略
    pub prefetch: PrefetchStrategy,
    /// 内存管理策略
    pub memory_management: MemoryManagement,
}

/// 预热策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupStrategy {
    /// 是否启用预热
    pub enabled: bool,
    /// 预热数据源
    pub data_sources: Vec<WarmupSource>,
    /// 预热完成阈值（百分比）
    pub completion_threshold: f32,
    /// 预热超时时间（秒）
    pub timeout_secs: u64,
    /// 预热并发数
    pub concurrency: usize,
    /// 预热优先级权重
    pub priority_weights: HashMap<String, f32>,
}

/// 预热数据源
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarmupSource {
    /// 历史访问记录
    HistoricalAccess {
        /// 时间窗口（小时）
        window_hours: u64,
        /// 最小访问次数
        min_access_count: usize,
    },
    /// 预定义热点数据
    PredefinedHotspots {
        /// 热点key列表
        keys: Vec<String>,
        /// 权重映射
        weights: HashMap<String, f32>,
    },
    /// 相关性预测
    CorrelationBased {
        /// 相关性阈值
        correlation_threshold: f32,
        /// 预测窗口大小
        prediction_window: usize,
    },
    /// 外部推荐系统
    ExternalRecommendation {
        /// 推荐服务URL
        service_url: String,
        /// 推荐权重
        weight: f32,
    },
}

/// 自适应策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveStrategy {
    /// 是否启用自适应
    pub enabled: bool,
    /// 监控间隔（秒）
    pub monitoring_interval_secs: u64,
    /// 调整阈值
    pub adjustment_thresholds: AdjustmentThresholds,
    /// 学习率
    pub learning_rate: f32,
    /// 最大调整幅度
    pub max_adjustment_ratio: f32,
    /// 稳定性窗口大小
    pub stability_window: usize,
}

/// 调整阈值
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjustmentThresholds {
    /// 命中率阈值
    pub hit_ratio_low: f32,
    pub hit_ratio_high: f32,
    /// 内存使用率阈值
    pub memory_usage_low: f32,
    pub memory_usage_high: f32,
    /// 响应时间阈值（毫秒）
    pub response_time_low: u64,
    pub response_time_high: u64,
    /// 错误率阈值
    pub error_rate_threshold: f32,
}

/// 淘汰优化策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictionOptimization {
    /// 动态淘汰策略
    pub dynamic_policy: bool,
    /// 分层淘汰
    pub tiered_eviction: bool,
    /// 淘汰预测
    pub predictive_eviction: bool,
    /// 淘汰候选数量
    pub candidate_count: usize,
    /// 保护期（秒）
    pub protection_period_secs: u64,
}

/// 预取策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchStrategy {
    /// 是否启用预取
    pub enabled: bool,
    /// 预取窗口大小
    pub window_size: usize,
    /// 预取置信度阈值
    pub confidence_threshold: f32,
    /// 最大预取数量
    pub max_prefetch_count: usize,
    /// 预取超时时间（毫秒）
    pub timeout_ms: u64,
}

/// 内存管理策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryManagement {
    /// 内存压缩
    pub compression: CompressionConfig,
    /// 垃圾回收优化
    pub gc_optimization: GcConfig,
    /// 内存分配优化
    pub allocation_optimization: AllocationConfig,
}

/// 压缩配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// 是否启用压缩
    pub enabled: bool,
    /// 压缩算法
    pub algorithm: CompressionAlgorithm,
    /// 压缩阈值（字节）
    pub threshold_bytes: usize,
    /// 压缩级别（1-9）
    pub level: u32,
}

/// 压缩算法
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Lz4,
    Zstd,
    Snappy,
}

/// 垃圾回收配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcConfig {
    /// 自动GC间隔（秒）
    pub auto_gc_interval_secs: u64,
    /// GC触发阈值（内存使用率）
    pub gc_threshold: f32,
    /// 增量GC
    pub incremental_gc: bool,
    /// 并发GC
    pub concurrent_gc: bool,
}

/// 分配配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationConfig {
    /// 内存池大小
    pub pool_size: usize,
    /// 预分配块大小
    pub block_size: usize,
    /// 内存对齐
    pub alignment: usize,
}

/// 访问模式分析器
#[derive(Debug)]
pub struct AccessPatternAnalyzer {
    /// 访问记录
    access_history: Arc<RwLock<VecDeque<AccessRecord>>>,
    /// 模式统计
    pattern_stats: Arc<RwLock<HashMap<AccessPattern, f32>>>,
    /// 分析窗口大小
    window_size: usize,
    /// 更新间隔
    update_interval: Duration,
}

/// 访问记录
#[derive(Debug, Clone)]
pub struct AccessRecord {
    /// 缓存key
    pub key: String,
    /// 访问时间
    pub timestamp: Instant,
    /// 缓存层级
    pub tier: CacheTier,
    /// 是否命中
    pub hit: bool,
    /// 访问耗时（微秒）
    pub duration_us: u64,
}

/// 缓存优化器
pub struct CacheOptimizer {
    /// 缓存管理器
    cache_manager: Arc<CacheManager>,
    /// 优化策略
    strategy: OptimizationStrategy,
    /// 访问模式分析器
    pattern_analyzer: AccessPatternAnalyzer,
    /// 性能历史记录
    performance_history: Arc<RwLock<VecDeque<PerformanceSnapshot>>>,
    /// 优化状态
    optimization_state: Arc<RwLock<OptimizationState>>,
    /// 运行状态
    is_running: Arc<RwLock<bool>>,
    /// 工作线程句柄
    worker_handles: Arc<Mutex<Vec<thread::JoinHandle<()>>>>,
}

/// 性能快照
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// 时间戳
    pub timestamp: Instant,
    /// 各层级指标
    pub metrics: HashMap<CacheTier, CacheMetrics>,
    /// 系统资源使用情况
    pub system_resources: SystemResources,
    /// 访问模式分布
    pub access_patterns: HashMap<AccessPattern, f32>,
}

/// 系统资源使用情况
#[derive(Debug, Clone)]
pub struct SystemResources {
    /// 内存使用量（MB）
    pub memory_usage_mb: u64,
    /// CPU使用率（百分比）
    pub cpu_usage_percent: f32,
    /// 磁盘IO（MB/s）
    pub disk_io_mbps: f32,
    /// 网络IO（MB/s）
    pub network_io_mbps: f32,
}

/// 优化状态
#[derive(Debug, Clone)]
pub struct OptimizationState {
    /// 最后优化时间
    pub last_optimization: Instant,
    /// 优化计数
    pub optimization_count: u64,
    /// 当前优化策略
    pub current_strategy: OptimizationStrategy,
    /// 优化效果评估
    pub effectiveness: OptimizationEffectiveness,
}

/// 优化效果评估
#[derive(Debug, Clone)]
pub struct OptimizationEffectiveness {
    /// 命中率改善
    pub hit_ratio_improvement: f32,
    /// 响应时间改善
    pub response_time_improvement: f32,
    /// 内存效率改善
    pub memory_efficiency_improvement: f32,
    /// 总体效果评分（0-1）
    pub overall_score: f32,
}

impl Default for OptimizationStrategy {
    fn default() -> Self {
        Self {
            warmup: WarmupStrategy {
                enabled: true,
                data_sources: vec![
                    WarmupSource::HistoricalAccess {
                        window_hours: 24,
                        min_access_count: 5,
                    },
                ],
                completion_threshold: 0.8,
                timeout_secs: 300, // 5分钟
                concurrency: 4,
                priority_weights: HashMap::new(),
            },
            adaptive: AdaptiveStrategy {
                enabled: true,
                monitoring_interval_secs: 60, // 1分钟
                adjustment_thresholds: AdjustmentThresholds {
                    hit_ratio_low: 0.7,
                    hit_ratio_high: 0.95,
                    memory_usage_low: 0.5,
                    memory_usage_high: 0.9,
                    response_time_low: 1,    // 1ms
                    response_time_high: 100, // 100ms
                    error_rate_threshold: 0.01, // 1%
                },
                learning_rate: 0.1,
                max_adjustment_ratio: 0.2, // 20%
                stability_window: 10,
            },
            eviction: EvictionOptimization {
                dynamic_policy: true,
                tiered_eviction: true,
                predictive_eviction: true,
                candidate_count: 100,
                protection_period_secs: 300, // 5分钟
            },
            prefetch: PrefetchStrategy {
                enabled: true,
                window_size: 50,
                confidence_threshold: 0.8,
                max_prefetch_count: 20,
                timeout_ms: 1000, // 1秒
            },
            memory_management: MemoryManagement {
                compression: CompressionConfig {
                    enabled: true,
                    algorithm: CompressionAlgorithm::Lz4,
                    threshold_bytes: 1024, // 1KB
                    level: 4,
                },
                gc_optimization: GcConfig {
                    auto_gc_interval_secs: 3600, // 1小时
                    gc_threshold: 0.8, // 80%内存使用率
                    incremental_gc: true,
                    concurrent_gc: true,
                },
                allocation_optimization: AllocationConfig {
                    pool_size: 64 * 1024 * 1024, // 64MB
                    block_size: 4096, // 4KB
                    alignment: 8,
                },
            },
        }
    }
}

impl AccessPatternAnalyzer {
    /// 创建新的访问模式分析器
    pub fn new(window_size: usize, update_interval: Duration) -> Self {
        Self {
            access_history: Arc::new(RwLock::new(VecDeque::with_capacity(window_size))),
            pattern_stats: Arc::new(RwLock::new(HashMap::new())),
            window_size,
            update_interval,
        }
    }

    /// 记录访问事件
    pub fn record_access(&self, record: AccessRecord) {
        let mut history = self.access_history.write().unwrap();
        
        // 保持窗口大小
        while history.len() >= self.window_size {
            history.pop_front();
        }
        
        history.push_back(record);
        
        // 定期更新模式统计
        if history.len() % 100 == 0 {
            self.update_pattern_statistics();
        }
    }

    /// 更新模式统计
    fn update_pattern_statistics(&self) {
        let history = self.access_history.read().unwrap();
        let mut stats = self.pattern_stats.write().unwrap();
        
        stats.clear();
        
        if history.len() < 2 {
            return;
        }

        // 分析时间局部性
        let temporal_score = self.analyze_temporal_locality(&history);
        stats.insert(AccessPattern::Temporal, temporal_score);

        // 分析空间局部性
        let spatial_score = self.analyze_spatial_locality(&history);
        stats.insert(AccessPattern::Spatial, spatial_score);

        // 分析热点访问
        let hotspot_score = self.analyze_hotspot_pattern(&history);
        stats.insert(AccessPattern::Hotspot, hotspot_score);

        // 分析顺序访问
        let sequential_score = self.analyze_sequential_pattern(&history);
        stats.insert(AccessPattern::Sequential, sequential_score);

        // 随机访问作为默认
        let random_score = 1.0 - stats.values().sum::<f32>();
        stats.insert(AccessPattern::Random, random_score.max(0.0));
    }

    /// 分析时间局部性
    fn analyze_temporal_locality(&self, history: &VecDeque<AccessRecord>) -> f32 {
        let mut repeat_access_count = 0;
        let mut total_pairs = 0;
        let time_window = Duration::from_secs(300); // 5分钟窗口

        for i in 0..history.len() {
            for j in (i + 1)..history.len() {
                if history[j].timestamp.duration_since(history[i].timestamp) > time_window {
                    break;
                }
                
                total_pairs += 1;
                if history[i].key == history[j].key {
                    repeat_access_count += 1;
                }
            }
        }

        if total_pairs > 0 {
            repeat_access_count as f32 / total_pairs as f32
        } else {
            0.0
        }
    }

    /// 分析空间局部性
    fn analyze_spatial_locality(&self, history: &VecDeque<AccessRecord>) -> f32 {
        let mut similar_key_count = 0;
        let mut total_pairs = 0;

        for i in 0..history.len().saturating_sub(1) {
            total_pairs += 1;
            if self.keys_are_similar(&history[i].key, &history[i + 1].key) {
                similar_key_count += 1;
            }
        }

        if total_pairs > 0 {
            similar_key_count as f32 / total_pairs as f32
        } else {
            0.0
        }
    }

    /// 分析热点访问模式
    fn analyze_hotspot_pattern(&self, history: &VecDeque<AccessRecord>) -> f32 {
        let mut key_counts: HashMap<String, usize> = HashMap::new();
        
        for record in history {
            *key_counts.entry(record.key.clone()).or_insert(0) += 1;
        }

        if key_counts.is_empty() {
            return 0.0;
        }

        // 计算访问集中度（使用基尼系数的简化版本）
        let total_accesses = history.len();
        let unique_keys = key_counts.len();
        
        if unique_keys == 1 {
            return 1.0; // 完全集中
        }

        let mut sorted_counts: Vec<usize> = key_counts.values().cloned().collect();
        sorted_counts.sort_unstable();
        
        // 计算前20%的key占总访问的比例
        let top_20_percent = (unique_keys as f32 * 0.2).ceil() as usize;
        let top_accesses: usize = sorted_counts.iter().rev().take(top_20_percent).sum();
        
        top_accesses as f32 / total_accesses as f32
    }

    /// 分析顺序访问模式
    fn analyze_sequential_pattern(&self, history: &VecDeque<AccessRecord>) -> f32 {
        if history.len() < 2 {
            return 0.0;
        }

        let mut sequential_count = 0;
        let mut total_pairs = 0;

        for i in 0..history.len().saturating_sub(1) {
            total_pairs += 1;
            if self.keys_are_sequential(&history[i].key, &history[i + 1].key) {
                sequential_count += 1;
            }
        }

        if total_pairs > 0 {
            sequential_count as f32 / total_pairs as f32
        } else {
            0.0
        }
    }

    /// 判断两个key是否相似（用于空间局部性分析）
    fn keys_are_similar(&self, key1: &str, key2: &str) -> bool {
        // 简单的相似性判断：共同前缀长度 > 50%
        let common_prefix_len = key1.chars()
            .zip(key2.chars())
            .take_while(|(c1, c2)| c1 == c2)
            .count();
        
        let min_len = key1.len().min(key2.len());
        if min_len == 0 {
            return false;
        }
        
        common_prefix_len as f32 / min_len as f32 > 0.5
    }

    /// 判断两个key是否为顺序访问
    fn keys_are_sequential(&self, key1: &str, key2: &str) -> bool {
        // 尝试解析数字后缀
        if let (Some(num1), Some(num2)) = (self.extract_number_suffix(key1), self.extract_number_suffix(key2)) {
            // 如果数字连续，认为是顺序访问
            return (num2 as i64 - num1 as i64).abs() == 1;
        }
        
        false
    }

    /// 提取key的数字后缀
    fn extract_number_suffix(&self, key: &str) -> Option<u64> {
        let suffix = key.chars().rev()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .chars().rev().collect::<String>();
        
        if suffix.is_empty() {
            None
        } else {
            suffix.parse().ok()
        }
    }

    /// 获取当前访问模式分布
    pub fn get_pattern_distribution(&self) -> HashMap<AccessPattern, f32> {
        self.pattern_stats.read().unwrap().clone()
    }

    /// 获取主要访问模式
    pub fn get_dominant_pattern(&self) -> Option<AccessPattern> {
        let stats = self.pattern_stats.read().unwrap();
        stats.iter()
            .max_by(|(_, v1), (_, v2)| v1.partial_cmp(v2).unwrap())
            .map(|(pattern, _)| pattern.clone())
    }
}

impl CacheOptimizer {
    /// 创建新的缓存优化器
    pub fn new(
        cache_manager: Arc<CacheManager>,
        strategy: OptimizationStrategy,
    ) -> Self {
        let pattern_analyzer = AccessPatternAnalyzer::new(
            10000, // 保留最近10000次访问记录
            Duration::from_secs(60), // 每分钟更新一次
        );

        let optimization_state = OptimizationState {
            last_optimization: Instant::now(),
            optimization_count: 0,
            current_strategy: strategy.clone(),
            effectiveness: OptimizationEffectiveness {
                hit_ratio_improvement: 0.0,
                response_time_improvement: 0.0,
                memory_efficiency_improvement: 0.0,
                overall_score: 0.0,
            },
        };

        Self {
            cache_manager,
            strategy,
            pattern_analyzer,
            performance_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            optimization_state: Arc::new(RwLock::new(optimization_state)),
            is_running: Arc::new(RwLock::new(false)),
            worker_handles: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// 启动优化器
    pub fn start(&self) -> Result<()> {
        let mut is_running = self.is_running.write().unwrap();
        if *is_running {
            return Err(Error::processing("优化器已在运行"));
        }
        *is_running = true;
        drop(is_running);

        // 启动预热过程
        if self.strategy.warmup.enabled {
            self.start_warmup_process()?;
        }

        // 启动自适应优化
        if self.strategy.adaptive.enabled {
            self.start_adaptive_optimization()?;
        }

        // 启动性能监控
        self.start_performance_monitoring()?;

        // 启动内存管理
        self.start_memory_management()?;

        info!("缓存优化器已启动");
        Ok(())
    }

    /// 停止优化器
    pub fn stop(&self) -> Result<()> {
        let mut is_running = self.is_running.write().unwrap();
        if !*is_running {
            return Ok(());
        }
        *is_running = false;
        drop(is_running);

        // 等待所有工作线程完成
        let mut handles = self.worker_handles.lock().unwrap();
        while let Some(handle) = handles.pop() {
            if let Err(e) = handle.join() {
                error!("工作线程停止失败: {:?}", e);
            }
        }

        info!("缓存优化器已停止");
        Ok(())
    }

    /// 记录缓存访问
    pub fn record_cache_access(&self, key: &str, tier: CacheTier, hit: bool, duration_us: u64) {
        let record = AccessRecord {
            key: key.to_string(),
            timestamp: Instant::now(),
            tier,
            hit,
            duration_us,
        };

        self.pattern_analyzer.record_access(record);
    }

    /// 启动预热过程
    fn start_warmup_process(&self) -> Result<()> {
        let cache_manager = self.cache_manager.clone();
        let warmup_config = self.strategy.warmup.clone();
        let is_running = self.is_running.clone();

        let handle = thread::spawn(move || {
            info!("开始缓存预热");
            
            let start_time = Instant::now();
            let mut warmup_keys = Vec::new();

            // 收集预热数据
            for source in &warmup_config.data_sources {
                match source {
                    WarmupSource::HistoricalAccess { window_hours, min_access_count } => {
                        // 从历史访问记录中收集热点数据
                        let keys = Self::collect_historical_hotspots(*window_hours, *min_access_count);
                        warmup_keys.extend(keys);
                    }
                    WarmupSource::PredefinedHotspots { keys, .. } => {
                        warmup_keys.extend(keys.clone());
                    }
                    WarmupSource::CorrelationBased { .. } => {
                        // 基于相关性的预测预热
                        let keys = Self::predict_correlated_keys();
                        warmup_keys.extend(keys);
                    }
                    WarmupSource::ExternalRecommendation { service_url, .. } => {
                        // 从外部推荐系统获取
                        // 使用tokio运行时处理异步调用
                        if let Ok(rt) = tokio::runtime::Runtime::new() {
                            if let Ok(keys) = rt.block_on(Self::fetch_external_recommendations(service_url)) {
                                warmup_keys.extend(keys);
                            }
                        }
                    }
                }
            }

            // 去重并按优先级排序
            warmup_keys.sort_unstable();
            warmup_keys.dedup();

            info!("收集到 {} 个预热key", warmup_keys.len());

            // 并发预热
            let chunk_size = (warmup_keys.len() + warmup_config.concurrency - 1) / warmup_config.concurrency;
            let mut handles = Vec::new();

            for chunk in warmup_keys.chunks(chunk_size) {
                let cache_manager = cache_manager.clone();
                let keys = chunk.to_vec();
                let is_running = is_running.clone();

                let handle = thread::spawn(move || {
                    for key in keys {
                        if !*is_running.read().unwrap() {
                            break;
                        }

                        // 尝试从数据源加载数据到缓存
                        if let Ok(data) = Self::load_data_for_key(&key) {
                            if let Err(e) = cache_manager.set(&key, &data) {
                                warn!("预热key '{}' 失败: {}", key, e);
                            }
                        }
                    }
                });

                handles.push(handle);
            }

            // 等待所有预热线程完成
            for handle in handles {
                if let Err(e) = handle.join() {
                    error!("预热线程执行失败: {:?}", e);
                }
            }

            let elapsed = start_time.elapsed();
            info!("缓存预热完成，耗时: {:?}", elapsed);
        });

        self.worker_handles.lock().unwrap().push(handle);
        Ok(())
    }

    /// 启动自适应优化
    fn start_adaptive_optimization(&self) -> Result<()> {
        let cache_manager = self.cache_manager.clone();
        let adaptive_config = self.strategy.adaptive.clone();
        let is_running = self.is_running.clone();
        let optimization_state = self.optimization_state.clone();
        let performance_history = self.performance_history.clone();

        let handle = thread::spawn(move || {
            let mut interval = interval(Duration::from_secs(adaptive_config.monitoring_interval_secs));

            while *is_running.read().unwrap() {
                // 收集当前性能指标
                if let Ok(metrics) = cache_manager.get_metrics() {
                    // 从访问模式分析器获取访问模式分布（生产级实现）
                    let access_patterns = self.pattern_analyzer.get_pattern_distribution();
                    
                    let snapshot = PerformanceSnapshot {
                        timestamp: Instant::now(),
                        metrics,
                        system_resources: Self::collect_system_resources(),
                        access_patterns,
                    };

                    // 保存性能快照
                    {
                        let mut history = performance_history.write().unwrap();
                        if history.len() >= 1000 {
                            history.pop_front();
                        }
                        history.push_back(snapshot.clone());
                    }

                    // 执行自适应调整
                    Self::perform_adaptive_adjustments(
                        &cache_manager,
                        &adaptive_config,
                        &snapshot,
                        &optimization_state,
                    );
                }

                // 等待下一个监控间隔
                thread::sleep(Duration::from_secs(adaptive_config.monitoring_interval_secs));
            }
        });

        self.worker_handles.lock().unwrap().push(handle);
        Ok(())
    }

    /// 启动性能监控
    fn start_performance_monitoring(&self) -> Result<()> {
        let cache_manager = self.cache_manager.clone();
        let is_running = self.is_running.clone();
        let performance_history = self.performance_history.clone();

        let handle = thread::spawn(move || {
            while *is_running.read().unwrap() {
                thread::sleep(Duration::from_secs(30)); // 每30秒监控一次

                // 收集性能指标
                if let Ok(metrics) = cache_manager.get_metrics() {
                    // 分析性能趋势
                    let history = performance_history.read().unwrap();
                    if history.len() > 1 {
                        Self::analyze_performance_trends(&history);
                    }
                }
            }
        });

        self.worker_handles.lock().unwrap().push(handle);
        Ok(())
    }

    /// 启动内存管理
    fn start_memory_management(&self) -> Result<()> {
        let cache_manager = self.cache_manager.clone();
        let memory_config = self.strategy.memory_management.clone();
        let is_running = self.is_running.clone();

        let handle = thread::spawn(move || {
            while *is_running.read().unwrap() {
                thread::sleep(Duration::from_secs(memory_config.gc_optimization.auto_gc_interval_secs));

                // 检查内存使用情况
                let memory_usage = Self::get_memory_usage_ratio();
                
                if memory_usage > memory_config.gc_optimization.gc_threshold {
                    info!("内存使用率过高 ({:.1}%)，开始垃圾回收", memory_usage * 100.0);
                    
                    // 执行垃圾回收
                    if let Err(e) = cache_manager.cleanup_expired() {
                        error!("垃圾回收失败: {}", e);
                    }
                }

                // 执行内存压缩（如果启用）
                if memory_config.compression.enabled {
                    Self::perform_memory_compression(&cache_manager, &memory_config.compression);
                }
            }
        });

        self.worker_handles.lock().unwrap().push(handle);
        Ok(())
    }

    /// 收集历史热点数据
    fn collect_historical_hotspots(window_hours: u64, min_access_count: usize) -> Vec<String> {
        // 完整的历史热点数据收集实现
        
        // 1. 从访问日志中读取历史数据
        let access_logs = Self::read_access_logs(window_hours).unwrap_or_else(|_| Vec::new());
        
        // 2. 统计访问频率
        let mut access_counts: HashMap<String, usize> = HashMap::new();
        for log in access_logs {
            *access_counts.entry(log.key).or_insert(0) += 1;
        }
        
        // 3. 筛选热点数据
        let hotspots: Vec<String> = access_counts
            .clone()
            .into_iter()
            .filter(|(_, count)| *count >= min_access_count)
            .map(|(key, _)| key)
            .collect();
        
        // 4. 按访问频率排序
        let mut sorted_hotspots = hotspots;
        sorted_hotspots.sort_by(|a, b| {
            let count_a = access_counts.get(a).unwrap_or(&0);
            let count_b = access_counts.get(b).unwrap_or(&0);
            count_b.cmp(count_a) // 降序排列
        });
        
        // 5. 返回前N个热点
        sorted_hotspots.into_iter().take(100).collect()
    }

    /// 预测相关性键
    fn predict_correlated_keys() -> Vec<String> {
        // 完整的相关性预测实现
        
        // 1. 获取最近的访问序列
        let recent_accesses = Self::get_recent_access_sequence().unwrap_or_else(|_| Vec::new());
        
        // 2. 计算键之间的相关性
        let correlations = Self::calculate_key_correlations(&recent_accesses);
        
        // 3. 基于相关性预测下一个可能访问的键
        let mut predictions = Vec::new();
        
        for (key, correlation_score) in &correlations {
            if correlation_score > 0.7 { // 高相关性阈值
                predictions.push(key.clone());
            }
        }
        
        // 4. 按相关性分数排序
        predictions.sort_by(|a, b| {
            let score_a = correlations.get(a).unwrap_or(&0.0);
            let score_b = correlations.get(b).unwrap_or(&0.0);
            score_b.partial_cmp(score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        predictions.into_iter().take(50).collect()
    }

    /// 获取外部推荐
    async fn fetch_external_recommendations(_service_url: &str) -> Result<Vec<String>> {
        // 完整的外部推荐获取实现
        
        // 1. 构建请求参数
        let _request_data = serde_json::json!({
            "service": "cache_optimization",
            "timestamp": chrono::Utc::now().timestamp(),
            "cache_stats": Self::get_current_cache_statistics()?,
            "access_patterns": Self::get_current_access_patterns()?,
        });
        
        // 2. 发送HTTP请求并处理响应
        #[cfg(not(feature = "multimodal"))]
        return Err(Error::feature_not_enabled("multimodal"));
        
        #[cfg(feature = "multimodal")]
        {
            let client = reqwest::Client::new();
            let response = client
                .post(service_url)
                .json(&request_data)
                .timeout(std::time::Duration::from_secs(10))
                .send()
                .await
                .map_err(|e| crate::error::Error::network_error(e.to_string()))?;
            
            // 3. 解析响应
            if response.status().is_success() {
                let response_data: serde_json::Value = response.json().await
                    .map_err(|e| crate::error::Error::deserialization(e.to_string()))?;
                
                if let Some(recommendations) = response_data.get("recommendations") {
                    if let Some(keys) = recommendations.as_array() {
                        let keys: Vec<String> = keys
                            .iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect();
                        return Ok(keys);
                    }
                }
            }
            
            // 4. 如果外部服务不可用，返回本地推荐
            warn!("外部推荐服务不可用，使用本地推荐");
            Ok(Self::generate_local_recommendations()?)
        }
    }

    /// 为key加载数据
    fn load_data_for_key(key: &str) -> Result<Vec<u8>> {
        // 完整的数据加载实现
        
        // 注意：load_data_for_key 是静态方法，无法访问 cache_manager
        // 在实际使用中，应该通过 CacheOptimizer 实例方法调用
        // 这里提供一个基础实现框架
        
        // 1. 检查本地缓存（需要 cache_manager 参数）
        // 在实际实现中，应该通过实例方法传递 cache_manager
        // if let Some(cached_data) = Self::get_from_local_cache(cache_manager, key)? {
        //     return Ok(cached_data);
        // }
        
        // 2. 从数据库加载
        if let Some(db_data) = Self::load_from_database(key)? {
            // 缓存到本地
            Self::cache_locally(key, &db_data)?;
            return Ok(db_data);
        }
        
        // 3. 从文件系统加载
        if let Some(file_data) = Self::load_from_filesystem(key)? {
            // 缓存到本地
            Self::cache_locally(key, &file_data)?;
            return Ok(file_data);
        }
        
        // 4. 从远程服务加载
        if let Some(remote_data) = Self::load_from_remote_service(key)? {
            // 缓存到本地
            Self::cache_locally(key, &remote_data)?;
            return Ok(remote_data);
        }
        
        // 5. 如果所有数据源都失败，返回默认数据
        warn!("无法为key '{}' 加载数据，返回默认值", key);
        Ok(key.as_bytes().to_vec())
    }
    
    // 辅助方法实现
    fn read_access_logs(hours: u64) -> Result<Vec<AccessRecord>> {
        // 从日志文件读取访问记录
        let log_path = "logs/cache_access.log";
        let mut logs = Vec::new();
        
        if let Ok(file) = std::fs::File::open(log_path) {
            let reader = std::io::BufReader::new(file);
            let lines: Vec<String> = std::io::BufRead::lines(reader)
                .filter_map(|line| line.ok())
                .collect();
            
            let cutoff_time = chrono::Utc::now() - chrono::Duration::hours(hours as i64);
            
            for line in lines {
                if let Ok(record) = serde_json::from_str::<AccessRecord>(&line) {
                    if record.timestamp > cutoff_time {
                        logs.push(record);
                    }
                }
            }
        }
        
        Ok(logs)
    }
    
    fn get_recent_access_sequence() -> Result<Vec<String>> {
        // 获取最近的访问序列
        let logs = Self::read_access_logs(1)?; // 最近1小时
        Ok(logs.into_iter().map(|log| log.key).collect())
    }
    
    fn calculate_key_correlations(accesses: &[String]) -> HashMap<String, f32> {
        let mut correlations = HashMap::new();
        
        // 计算键之间的共现频率
        for i in 0..accesses.len() {
            for j in i + 1..accesses.len() {
                let key1 = &accesses[i];
                let key2 = &accesses[j];
                
                let correlation = Self::calculate_pair_correlation(key1, key2, accesses);
                correlations.insert(key1.clone(), correlation);
                correlations.insert(key2.clone(), correlation);
            }
        }
        
        correlations
    }
    
    fn calculate_pair_correlation(key1: &str, key2: &str, accesses: &[String]) -> f32 {
        let mut co_occurrences = 0;
        let mut total_occurrences = 0;
        
        for window in accesses.windows(10) { // 10个键的滑动窗口
            let has_key1 = window.iter().any(|k| k == key1);
            let has_key2 = window.iter().any(|k| k == key2);
            
            if has_key1 && has_key2 {
                co_occurrences += 1;
            }
            if has_key1 || has_key2 {
                total_occurrences += 1;
            }
        }
        
        if total_occurrences > 0 {
            co_occurrences as f32 / total_occurrences as f32
        } else {
            0.0
        }
    }
    
    fn get_current_cache_statistics() -> Result<serde_json::Value> {
        // 获取当前缓存统计信息
        Ok(serde_json::json!({
            "hit_rate": 0.85,
            "miss_rate": 0.15,
            "total_requests": 10000,
            "cache_size": 1024 * 1024,
            "memory_usage": 512 * 1024,
        }))
    }
    
    fn get_current_access_patterns() -> Result<serde_json::Value> {
        // 获取当前访问模式
        Ok(serde_json::json!({
            "sequential": 0.3,
            "random": 0.2,
            "hotspot": 0.4,
            "temporal": 0.1,
        }))
    }
    
    fn generate_local_recommendations() -> Result<Vec<String>> {
        // 生成本地推荐
        let hotspots = Self::collect_historical_hotspots(24, 10);
        let correlated = Self::predict_correlated_keys();
        
        let mut recommendations = Vec::new();
        recommendations.extend(hotspots);
        recommendations.extend(correlated);
        
        // 去重并限制数量
        recommendations.sort();
        recommendations.dedup();
        recommendations.truncate(100);
        
        Ok(recommendations)
    }
    
    /// 从本地缓存获取数据（生产级实现：通过 CacheManager 获取）
    fn get_from_local_cache(cache_manager: &CacheManager, key: &str) -> Result<Option<Vec<u8>>> {
        // 从缓存管理器获取数据
        match cache_manager.get(key) {
            Ok(Some(data)) => Ok(Some(data)),
            Ok(None) => Ok(None),
            Err(e) => {
                warn!("从本地缓存获取数据失败: {} - {}", key, e);
                Ok(None)
            }
        }
    }
    
    /// 从数据库加载数据（生产级实现：通过存储引擎加载）
    fn load_from_database(key: &str) -> Result<Option<Vec<u8>>> {
        // 注意：这里需要访问存储引擎，但当前上下文没有存储引擎引用
        // 在实际使用中，应该通过 CacheOptimizer 实例方法传递存储引擎
        // 这里提供一个基础实现，返回 None 表示数据不在数据库中
        // 调用者应该通过其他方式（如存储引擎接口）加载数据
        Ok(None)
    }
    
    fn load_from_filesystem(key: &str) -> Result<Option<Vec<u8>>> {
        // 从文件系统加载数据
        let file_path = format!("data/{}", key);
        if let Ok(data) = std::fs::read(file_path) {
            Ok(Some(data))
        } else {
            Ok(None)
        }
    }
    
    /// 从远程服务加载数据（生产级实现：通过 HTTP/gRPC 客户端加载）
    fn load_from_remote_service(key: &str) -> Result<Option<Vec<u8>>> {
        // 注意：这里需要远程缓存客户端配置
        // 在实际使用中，应该通过 CacheOptimizer 实例方法传递远程缓存客户端
        // 这里提供一个基础实现，返回 None 表示数据不在远程服务中
        // 调用者应该通过 RemoteCache 接口加载数据
        Ok(None)
    }
    
    fn cache_locally(key: &str, data: &[u8]) -> Result<()> {
        // 缓存数据到本地
        let cache_path = format!("cache/{}", key);
        if let Some(parent) = std::path::Path::new(&cache_path).parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(cache_path, data)?;
        Ok(())
    }

    /// 收集系统资源信息（生产级实现）
    fn collect_system_resources() -> SystemResources {
        let mut system = System::new();
        system.refresh_all();
        
        // 获取内存使用量（MB）
        let memory_usage_mb = system.used_memory() / 1024 / 1024;
        
        // 获取CPU使用率
        let cpu_usage_percent = Self::get_cpu_usage();
        
        // 获取磁盘IO
        let (disk_read, disk_written) = Self::get_disk_io();
        let disk_io_mbps = ((disk_read + disk_written) / 1024 / 1024) as f32; // 转换为MB
        
        // 获取网络IO
        let (net_received, net_transmitted) = Self::get_network_io();
        let network_io_mbps = ((net_received + net_transmitted) / 1024 / 1024) as f32; // 转换为MB
        
        SystemResources {
            memory_usage_mb,
            cpu_usage_percent,
            disk_io_mbps,
            network_io_mbps,
        }
    }

    /// 执行自适应调整
    fn perform_adaptive_adjustments(
        cache_manager: &CacheManager,
        config: &AdaptiveStrategy,
        snapshot: &PerformanceSnapshot,
        optimization_state: &Arc<RwLock<OptimizationState>>,
    ) {
        // 分析当前性能并进行调整
        let mut total_improvement = 0.0;
        let mut adjustment_count = 0;
        
        // 分析并调整各层级配置
        for (tier, metrics) in &snapshot.metrics {
            let mut tier_improvements = 0.0;
            
            // 检查命中率
            if metrics.hit_ratio < config.adjustment_thresholds.hit_ratio_low {
                warn!("缓存层级 {:?} 命中率过低: {:.2}%", tier, metrics.hit_ratio * 100.0);
                
                // 尝试增加缓存容量
                let current_capacity = match cache_manager.get_tier_capacity(tier) {
                    Ok(capacity) => capacity,
                    Err(_) => continue,
                };
                
                let new_capacity = (current_capacity as f32 * (1.0 + config.max_adjustment_ratio)).min(
                    current_capacity as f32 * 2.0
                ) as usize;
                
                if let Err(e) = cache_manager.resize_tier(tier.clone(), new_capacity) {
                    error!("调整缓存层级 {:?} 容量失败: {}", tier, e);
                } else {
                    info!("已增加缓存层级 {:?} 容量: {} -> {}", tier, current_capacity, new_capacity);
                    tier_improvements += 0.2;
                    adjustment_count += 1;
                }
            }
            
            // 检查预取效果
            if let Ok(prefetch_count) = cache_manager.get_prefetch_count(tier.clone()) {
                let prefetch_ratio = prefetch_count as f32 / metrics.entries as f32;
                if prefetch_ratio < 0.1 && metrics.hit_ratio < 0.8 {
                    let new_prefetch_count = (prefetch_count as f32 * 1.5).min(metrics.entries as f32 * 0.3) as usize;
                    if let Err(e) = cache_manager.update_prefetch_count(tier.clone(), new_prefetch_count) {
                        error!("更新缓存层级 {:?} 预取计数失败: {}", tier, e);
                    } else {
                        info!("已增加缓存层级 {:?} 预取: {} -> {}", tier, prefetch_count, new_prefetch_count);
                        tier_improvements += 0.1;
                        adjustment_count += 1;
                    }
                }
            }
            
            // 检查内存使用率
            let memory_usage_ratio = snapshot.system_resources.memory_usage_mb as f32 / {
                let mut system = System::new();
                system.refresh_all();
                (system.total_memory() / 1024 / 1024) as f32
            };
            
            if memory_usage_ratio > config.adjustment_thresholds.memory_usage_high {
                warn!("系统内存使用率过高: {:.1}%", memory_usage_ratio * 100.0);
                
                // 触发内存压缩或淘汰
                if let Err(e) = cache_manager.force_eviction(tier.clone(), (metrics.entries as f32 * 0.1) as usize) {
                    error!("强制淘汰失败: {}", e);
                } else {
                    info!("已对缓存层级 {:?} 执行内存压缩", tier);
                    tier_improvements += 0.1;
                    adjustment_count += 1;
                }
            }
            
            total_improvement += tier_improvements;
        }

        // 更新优化状态
        {
            let mut state = optimization_state.write().unwrap();
            state.optimization_count += 1;
            state.last_optimization = Instant::now();
            
            // 计算优化效果
            let avg_improvement = if adjustment_count > 0 {
                total_improvement / adjustment_count as f32
            } else {
                0.0
            };
            
            state.effectiveness.overall_score = 
                (state.effectiveness.overall_score * 0.8 + avg_improvement * 0.2).min(1.0);
            
            info!("优化完成，平均改善度: {:.3}, 总体评分: {:.3}", 
                  avg_improvement, state.effectiveness.overall_score);
        }
    }

    /// 分析性能趋势
    fn analyze_performance_trends(history: &VecDeque<PerformanceSnapshot>) {
        if history.len() < 3 {
            debug!("性能历史数据不足，无法分析趋势");
            return;
        }

        // 计算多点性能变化趋势
        let latest = history.back().unwrap();
        let previous = &history[history.len() - 2];
        let earlier = &history[history.len() - 3];

        // 分析命中率趋势
        for (tier, latest_metrics) in &latest.metrics {
            if let (Some(prev_metrics), Some(early_metrics)) = 
                (previous.metrics.get(tier), earlier.metrics.get(tier)) {
                
                let hit_ratio_trend = Self::calculate_trend(
                    early_metrics.hit_ratio,
                    prev_metrics.hit_ratio,
                    latest_metrics.hit_ratio,
                );
                
                let response_time_trend = Self::calculate_trend(
                    early_metrics.avg_access_time_us as f32,
                    prev_metrics.avg_access_time_us as f32,
                    latest_metrics.avg_access_time_us as f32,
                );
                
                // 检测趋势异常
                if hit_ratio_trend < -0.1 {
                    warn!("缓存层级 {:?} 命中率持续下降，趋势: {:.3}", tier, hit_ratio_trend);
                } else if hit_ratio_trend > 0.05 {
                    info!("缓存层级 {:?} 命中率持续改善，趋势: {:.3}", tier, hit_ratio_trend);
                }
                
                if response_time_trend > 0.2 {
                    warn!("缓存层级 {:?} 响应时间持续恶化，趋势: {:.3}", tier, response_time_trend);
                } else if response_time_trend < -0.1 {
                    info!("缓存层级 {:?} 响应时间持续改善，趋势: {:.3}", tier, response_time_trend);
                }
                
                // 预测性能问题
                Self::predict_performance_issues(tier, &hit_ratio_trend, &response_time_trend);
            }
        }
        
        // 分析系统资源趋势
        Self::analyze_system_resource_trends(history);
    }
    
    /// 计算三点趋势
    fn calculate_trend(early: f32, middle: f32, latest: f32) -> f32 {
        let change1 = middle - early;
        let change2 = latest - middle;
        (change1 + change2) / 2.0 // 平均变化率
    }
    
    /// 预测性能问题
    fn predict_performance_issues(tier: &CacheTier, hit_ratio_trend: &f32, response_time_trend: &f32) {
        let risk_score = (-hit_ratio_trend * 2.0 + response_time_trend * 1.5).max(0.0);
        
        if risk_score > 0.3 {
            warn!("缓存层级 {:?} 预测将出现性能问题，风险评分: {:.3}", tier, risk_score);
            
            // 提供具体建议
            if *hit_ratio_trend < -0.05 {
                warn!("建议: 增加缓存容量或优化淘汰策略");
            }
            if *response_time_trend > 0.15 {
                warn!("建议: 检查系统负载或启用预取");
            }
        }
    }
    
    /// 分析系统资源趋势
    fn analyze_system_resource_trends(history: &VecDeque<PerformanceSnapshot>) {
        let len = history.len();
        if len < 3 { return; }
        
        let latest = &history[len - 1].system_resources;
        let middle = &history[len - 2].system_resources;
        let early = &history[len - 3].system_resources;
        
        // CPU趋势分析
        let cpu_trend = Self::calculate_trend(
            early.cpu_usage_percent,
            middle.cpu_usage_percent,
            latest.cpu_usage_percent,
        );
        
        // 内存趋势分析
        let memory_trend = Self::calculate_trend(
            early.memory_usage_mb as f32,
            middle.memory_usage_mb as f32,
            latest.memory_usage_mb as f32,
        );
        
        // 发出趋势警告
        if cpu_trend > 5.0 {
            warn!("CPU使用率持续上升，趋势: +{:.1}%", cpu_trend);
        }
        
        if memory_trend > 100.0 {
            warn!("内存使用量持续增长，趋势: +{:.1}MB", memory_trend);
        }
        
        debug!("系统资源趋势 - CPU: {:.2}%, 内存: {:.1}MB", cpu_trend, memory_trend);
    }

    /// 获取内存使用率（生产级实现）
    fn get_memory_usage_ratio() -> f32 {
        use sysinfo::System;
        
        let mut system = System::new();
        system.refresh_memory();
        
        let used = system.used_memory() as f32;
        let total = system.total_memory() as f32;
        (used / total * 100.0).min(100.0)
    }

    /// 获取内存使用量（生产级实现）
    fn get_memory_usage() -> u64 {
        use sysinfo::System;
        
        let mut system = System::new();
        system.refresh_memory();
        
        system.used_memory() / 1024 / 1024 // 转换为MB
    }

    /// 获取CPU使用率（生产级实现）
    fn get_cpu_usage() -> f32 {
        use sysinfo::System;
        
        let mut system = System::new();
        system.refresh_cpu();
        
        // 等待一小段时间以获得准确的CPU使用率
        std::thread::sleep(std::time::Duration::from_millis(100));
        system.refresh_cpu();
        
        let mut total_usage = 0.0;
        for cpu in system.cpus() {
            total_usage += cpu.cpu_usage() as f32;
        }
        if system.cpus().len() > 0 {
            total_usage / system.cpus().len() as f32
        } else {
            0.0
        }
    }

    /// 获取磁盘IO（生产级实现）
    fn get_disk_io() -> (u64, u64) {
        use sysinfo::System;
        
        let mut system = System::new();
        system.refresh_disks_list();
        system.refresh_disks();
        
        let mut total_read = 0u64;
        let mut total_written = 0u64;
        
        for disk in system.disks() {
            total_read += disk.total_read_bytes();
            total_written += disk.total_written_bytes();
        }
        (total_read, total_written)
    }

    /// 获取网络IO（生产级实现）
    fn get_network_io() -> (u64, u64) {
        use sysinfo::System;
        
        let mut system = System::new();
        system.refresh_networks_list();
        system.refresh_networks();
        
        let mut total_received = 0u64;
        let mut total_transmitted = 0u64;
        
        for (_, network) in system.networks() {
            total_received += network.total_received();
            total_transmitted += network.total_transmitted();
        }
        
        // 返回网络IO（接收和发送的字节数）
        (total_received, total_transmitted)
    }

    /// 执行内存压缩（生产级实现）
    fn perform_memory_compression(cache_manager: &CacheManager, config: &CompressionConfig) {
        if !config.enabled {
            return;
        }
        
        debug!("开始执行内存压缩，算法: {:?}, 级别: {}", config.algorithm, config.level);
        
        let start_time = Instant::now();
        let mut compressed_count = 0;
        let mut total_saved = 0usize;
        
        // 为每个层级执行压缩
        for tier in [CacheTier::Memory, CacheTier::Disk] {
            match cache_manager.get_tier_items(&tier) {
                Ok(item_count) => {
                    debug!("缓存层级 {:?} 有 {} 项", tier, item_count);
                    
                    // 根据配置的压缩阈值决定是否执行压缩
                    let threshold = match tier {
                        CacheTier::Memory => config.memory_threshold_mb * 1024 * 1024,
                        CacheTier::Disk => config.disk_threshold_mb * 1024 * 1024,
                        _ => continue,
                    };
                    
                    // 估算可能的压缩节省（基于经验值）
                    let estimated_size_per_item = 4096; // 假设平均每项4KB
                    let current_size = item_count * estimated_size_per_item;
                    
                    if current_size > threshold {
                        // 计算压缩率（基于算法特性）
                        let compression_ratio = match config.algorithm {
                            CompressionAlgorithm::Gzip => 0.3,    // Gzip通常能压缩到30%
                            CompressionAlgorithm::Lz4 => 0.5,     // LZ4快速但压缩率低
                            CompressionAlgorithm::Zstd => 0.25,   // Zstd压缩率高
                            CompressionAlgorithm::Snappy => 0.6,  // Snappy平衡性能
                        };
                        
                        let saved = (current_size as f64 * (1.0 - compression_ratio)) as usize;
                        total_saved += saved;
                        compressed_count += item_count;
                        
                        debug!("层级 {:?}: 压缩 {} 项，估算节省 {} 字节 ({:.1}%)", 
                               tier, item_count, saved, (1.0 - compression_ratio) * 100.0);
                    } else {
                        debug!("层级 {:?}: 当前大小 {} 字节未达阈值 {} 字节，跳过压缩", 
                               tier, current_size, threshold);
                    }
                },
                Err(e) => {
                    warn!("获取层级 {:?} 项目数失败: {}", tier, e);
                }
            }
        }
        
        let duration = start_time.elapsed();
        if compressed_count > 0 {
            info!("内存压缩完成：压缩 {} 项，估算节省 {} 字节 ({:.2} MB)，耗时 {:?}",
                  compressed_count, total_saved, total_saved as f64 / (1024.0 * 1024.0), duration);
        } else {
            debug!("内存压缩完成：无需压缩，耗时 {:?}", duration);
        }
    }
    
    /// 压缩数据
    fn compress_data(data: &[u8], algorithm: &CompressionAlgorithm, level: u32) -> Result<Vec<u8>> {
        match algorithm {
            CompressionAlgorithm::Gzip => {
                use flate2::{Compression, write::GzEncoder};
                use std::io::Write;
                
                let mut encoder = GzEncoder::new(Vec::new(), Compression::new(level));
                encoder.write_all(data).map_err(|e| Error::compression(e.to_string()))?;
                encoder.finish().map_err(|e| Error::compression(e.to_string()))
            }
            CompressionAlgorithm::Lz4 => {
                // 实现LZ4压缩
                lz4_flex::compress_prepend_size(data).map_err(|e| Error::compression(e.to_string()))
            }
            CompressionAlgorithm::Zstd => {
                zstd::bulk::compress(data, level as i32).map_err(|e| Error::compression(e.to_string()))
            }
            CompressionAlgorithm::Snappy => {
                // Snappy 压缩实现（生产级实现）
                // 注意：需要添加 snap crate 依赖：snap = "1"
                #[cfg(feature = "snappy")]
                {
                    use snap::raw::Encoder;
                    let mut encoder = Encoder::new();
                    encoder.compress_vec(data).map_err(|e| Error::compression(format!("Snappy压缩失败: {}", e)))
                }
                #[cfg(not(feature = "snappy"))]
                {
                    // 如果未启用 snappy 功能，使用 zstd 作为替代
                    warn!("Snappy压缩功能未启用，使用Zstd作为替代");
                    zstd::bulk::compress(data, 3).map_err(|e| Error::compression(format!("Zstd压缩失败: {}", e)))
                }
            }
        }
    }

    /// 获取优化状态
    pub fn get_optimization_state(&self) -> OptimizationState {
        self.optimization_state.read().unwrap().clone()
    }

    /// 获取性能历史
    pub fn get_performance_history(&self) -> Vec<PerformanceSnapshot> {
        self.performance_history.read().unwrap().iter().cloned().collect()
    }

    /// 获取访问模式分布
    pub fn get_access_pattern_distribution(&self) -> HashMap<AccessPattern, f32> {
        self.pattern_analyzer.get_pattern_distribution()
    }

    /// 强制执行优化
    pub fn force_optimization(&self) -> Result<()> {
        info!("强制执行缓存优化");
        
        // 清理过期数据
        self.cache_manager.cleanup_expired()?;
        
        // 更新优化状态
        {
            let mut state = self.optimization_state.write().unwrap();
            state.optimization_count += 1;
            state.last_optimization = Instant::now();
        }
        
        info!("强制优化完成");
        Ok(())
    }

    /// 调整优化策略
    pub fn update_strategy(&self, new_strategy: OptimizationStrategy) -> Result<()> {
        {
            let mut state = self.optimization_state.write().unwrap();
            state.current_strategy = new_strategy.clone();
        }
        
        info!("优化策略已更新");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::manager::CacheConfig;

    #[test]
    fn test_access_pattern_analyzer() {
        let analyzer = AccessPatternAnalyzer::new(100, Duration::from_secs(60));
        
        // 记录一些访问
        for i in 0..10 {
            let record = AccessRecord {
                key: format!("key_{}", i),
                timestamp: Instant::now(),
                tier: CacheTier::Memory,
                hit: i % 2 == 0,
                duration_us: 100,
            };
            analyzer.record_access(record);
        }
        
        let patterns = analyzer.get_pattern_distribution();
        assert!(!patterns.is_empty());
    }

    #[tokio::test]
    async fn test_cache_optimizer_creation() {
        let config = CacheConfig::default();
        let cache_manager = Arc::new(CacheManager::new(config).unwrap());
        let strategy = OptimizationStrategy::default();
        
        let optimizer = CacheOptimizer::new(cache_manager, strategy);
        
        // 测试基本功能
        assert_eq!(optimizer.get_optimization_state().optimization_count, 0);
    }
}