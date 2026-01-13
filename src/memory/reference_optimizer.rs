/// 引用优化器
/// 
/// 专门解决Arc/Mutex过度使用问题：
/// 1. 智能引用管理 - 根据使用模式选择最优的引用类型
/// 2. 引用计数优化 - 减少不必要的引用计数开销
/// 3. 锁竞争检测 - 识别和优化锁竞争热点
/// 4. 引用生命周期分析 - 自动优化引用生命周期

use std::sync::{Arc, Weak, Mutex, RwLock};
use std::collections::{VecDeque, HashSet};
use std::time::{Instant, Duration};
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::thread::{self, ThreadId};
use dashmap::DashMap;

use super::{MemoryConfig, HealthStatus};
use crate::Result;

/// 引用类型建议
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceTypeRecommendation {
    /// 使用普通引用
    PlainReference,
    /// 使用Box
    BoxedReference,
    /// 使用Arc（读多写少）
    ArcReference,
    /// 使用Arc + RwLock（读多写少，需要内部可变性）
    ArcRwLock,
    /// 使用Arc + Mutex（写多或复杂同步）
    ArcMutex,
    /// 使用Weak引用
    WeakReference,
    /// 使用线程本地存储
    ThreadLocal,
    /// 使用原子类型
    Atomic,
}

/// 锁使用模式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LockUsagePattern {
    /// 读多写少
    ReadHeavy,
    /// 写多读少
    WriteHeavy,
    /// 读写均衡
    Balanced,
    /// 短期持有
    ShortLived,
    /// 长期持有
    LongLived,
    /// 竞争激烈
    HighContention,
    /// 竞争较少
    LowContention,
}

/// 引用使用统计
#[derive(Debug, Clone)]
pub struct ReferenceUsageStats {
    /// 引用ID
    pub reference_id: String,
    /// 当前引用类型
    pub current_type: String,
    /// 引用计数
    pub ref_count: usize,
    /// 创建时间
    pub created_at: Instant,
    /// 总访问次数
    pub total_accesses: u64,
    /// 读取次数
    pub read_accesses: u64,
    /// 写入次数
    pub write_accesses: u64,
    /// 锁等待总时间
    pub total_wait_time: Duration,
    /// 平均锁持有时间
    pub avg_hold_time: Duration,
    /// 最大锁持有时间
    pub max_hold_time: Duration,
    /// 竞争事件数
    pub contention_events: u64,
    /// 涉及的线程数
    pub thread_count: usize,
    /// 最后访问时间
    pub last_access: Instant,
}

/// 引用优化器统计信息
#[derive(Debug)]
pub struct OptimizerStats {
    /// 总优化次数
    pub total_optimizations: AtomicUsize,
    /// 成功优化次数
    pub successful_optimizations: AtomicUsize,
    /// 失败优化次数
    pub failed_optimizations: AtomicUsize,
    /// 当前监控的引用数量
    pub monitored_references: AtomicUsize,
    /// 活跃线程数
    pub active_threads: AtomicUsize,
}

impl OptimizerStats {
    /// 创建新的统计对象
    pub fn new() -> Self {
        Self {
            total_optimizations: AtomicUsize::new(0),
            successful_optimizations: AtomicUsize::new(0),
            failed_optimizations: AtomicUsize::new(0),
            monitored_references: AtomicUsize::new(0),
            active_threads: AtomicUsize::new(0),
        }
    }

    /// 增加总优化次数
    pub fn increment_total_optimizations(&self) {
        self.total_optimizations.fetch_add(1, Ordering::Relaxed);
    }

    /// 增加成功优化次数
    pub fn increment_successful_optimizations(&self) {
        self.successful_optimizations.fetch_add(1, Ordering::Relaxed);
    }

    /// 增加失败优化次数
    pub fn increment_failed_optimizations(&self) {
        self.failed_optimizations.fetch_add(1, Ordering::Relaxed);
    }

    /// 更新监控的引用数量
    pub fn update_monitored_references(&self, count: usize) {
        self.monitored_references.store(count, Ordering::Relaxed);
    }

    /// 更新活跃线程数
    pub fn update_active_threads(&self, count: usize) {
        self.active_threads.store(count, Ordering::Relaxed);
    }

    /// 获取统计信息
    pub fn get_stats(&self) -> (usize, usize, usize, usize, usize) {
        (
            self.total_optimizations.load(Ordering::Relaxed),
            self.successful_optimizations.load(Ordering::Relaxed),
            self.failed_optimizations.load(Ordering::Relaxed),
            self.monitored_references.load(Ordering::Relaxed),
            self.active_threads.load(Ordering::Relaxed),
        )
    }
}

impl ReferenceUsageStats {
    /// 创建新的统计对象
    pub fn new(reference_id: String, current_type: String) -> Self {
        let now = Instant::now();
        Self {
            reference_id,
            current_type,
            ref_count: 0,
            created_at: now,
            total_accesses: 0,
            read_accesses: 0,
            write_accesses: 0,
            total_wait_time: Duration::ZERO,
            avg_hold_time: Duration::ZERO,
            max_hold_time: Duration::ZERO,
            contention_events: 0,
            thread_count: 0,
            last_access: now,
        }
    }
    
    /// 记录读取访问
    pub fn record_read_access(&mut self, wait_time: Duration, hold_time: Duration) {
        self.total_accesses += 1;
        self.read_accesses += 1;
        self.total_wait_time += wait_time;
        self.last_access = Instant::now();
        
        // 更新平均持有时间
        let total_operations = self.total_accesses;
        self.avg_hold_time = (self.avg_hold_time * (total_operations - 1) + hold_time) / total_operations as u64;
        
        if hold_time > self.max_hold_time {
            self.max_hold_time = hold_time;
        }
        
        if wait_time > Duration::from_millis(1) {
            self.contention_events += 1;
        }
    }
    
    /// 记录写入访问
    pub fn record_write_access(&mut self, wait_time: Duration, hold_time: Duration) {
        self.total_accesses += 1;
        self.write_accesses += 1;
        self.total_wait_time += wait_time;
        self.last_access = Instant::now();
        
        // 更新平均持有时间
        let total_operations = self.total_accesses;
        self.avg_hold_time = (self.avg_hold_time * (total_operations - 1) + hold_time) / total_operations as u64;
        
        if hold_time > self.max_hold_time {
            self.max_hold_time = hold_time;
        }
        
        if wait_time > Duration::from_millis(1) {
            self.contention_events += 1;
        }
    }
    
    /// 计算读写比例
    pub fn read_write_ratio(&self) -> f64 {
        if self.write_accesses == 0 {
            f64::INFINITY
        } else {
            self.read_accesses as f64 / self.write_accesses as f64
        }
    }
    
    /// 计算竞争率
    pub fn contention_rate(&self) -> f64 {
        if self.total_accesses == 0 {
            0.0
        } else {
            self.contention_events as f64 / self.total_accesses as f64
        }
    }
    
    /// 计算访问频率
    pub fn access_frequency(&self) -> f64 {
        let elapsed = self.created_at.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.total_accesses as f64 / elapsed
        } else {
            0.0
        }
    }
    
    /// 分析锁使用模式
    pub fn analyze_lock_pattern(&self) -> LockUsagePattern {
        let read_ratio = self.read_write_ratio();
        let contention_rate = self.contention_rate();
        let avg_hold_time_ms = self.avg_hold_time.as_millis() as f64;
        
        if contention_rate > 0.3 {
            LockUsagePattern::HighContention
        } else if contention_rate < 0.05 {
            LockUsagePattern::LowContention
        } else if read_ratio > 10.0 {
            LockUsagePattern::ReadHeavy
        } else if read_ratio < 0.1 {
            LockUsagePattern::WriteHeavy
        } else if avg_hold_time_ms > 100.0 {
            LockUsagePattern::LongLived
        } else if avg_hold_time_ms < 1.0 {
            LockUsagePattern::ShortLived
        } else {
            LockUsagePattern::Balanced
        }
    }
}

/// 锁竞争事件
#[derive(Debug, Clone)]
pub struct ContentionEvent {
    /// 引用ID
    pub reference_id: String,
    /// 线程ID
    pub thread_id: ThreadId,
    /// 等待开始时间
    pub wait_start: Instant,
    /// 等待结束时间
    pub wait_end: Instant,
    /// 操作类型
    pub operation_type: String,
    /// 获取锁后的持有时间
    pub hold_duration: Duration,
}

impl ContentionEvent {
    /// 计算等待时间
    pub fn wait_duration(&self) -> Duration {
        self.wait_end.duration_since(self.wait_start)
    }
}

/// 引用优化建议
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// 引用ID
    pub reference_id: String,
    /// 当前类型
    pub current_type: String,
    /// 推荐类型
    pub recommended_type: ReferenceTypeRecommendation,
    /// 推荐原因
    pub reason: String,
    /// 预期性能提升
    pub expected_improvement: f64,
    /// 实施难度
    pub implementation_difficulty: u8, // 1-10
    /// 优先级
    pub priority: u8, // 1-10
}

impl OptimizationRecommendation {
    /// 创建新的优化建议
    pub fn new(
        reference_id: String,
        current_type: String,
        recommended_type: ReferenceTypeRecommendation,
        reason: String,
        expected_improvement: f64,
    ) -> Self {
        let difficulty = match recommended_type {
            ReferenceTypeRecommendation::PlainReference => 2,
            ReferenceTypeRecommendation::BoxedReference => 3,
            ReferenceTypeRecommendation::ArcReference => 4,
            ReferenceTypeRecommendation::WeakReference => 5,
            ReferenceTypeRecommendation::Atomic => 6,
            ReferenceTypeRecommendation::ThreadLocal => 7,
            ReferenceTypeRecommendation::ArcRwLock => 5,
            ReferenceTypeRecommendation::ArcMutex => 6,
        };
        
        let priority = ((expected_improvement * 10.0) as u8).min(10);
        
        Self {
            reference_id,
            current_type,
            recommended_type,
            reason,
            expected_improvement,
            implementation_difficulty: difficulty,
            priority,
        }
    }
}

/// 引用优化器统计
#[derive(Debug, Clone)]
pub struct ReferenceOptimizerStats {
    /// 跟踪的引用数量
    pub tracked_references: usize,
    /// 检测到的竞争事件
    pub contention_events: u64,
    /// 生成的建议数量
    pub recommendations: usize,
    /// 实施的优化数量
    pub applied_optimizations: usize,
    /// 平均引用计数
    pub avg_ref_count: f64,
    /// 高竞争引用数量
    pub high_contention_refs: usize,
    /// 未使用引用数量
    pub unused_refs: usize,
    /// 总体性能改进
    pub overall_improvement: f64,
}

impl Default for ReferenceOptimizerStats {
    fn default() -> Self {
        Self {
            tracked_references: 0,
            contention_events: 0,
            recommendations: 0,
            applied_optimizations: 0,
            avg_ref_count: 0.0,
            high_contention_refs: 0,
            unused_refs: 0,
            overall_improvement: 0.0,
        }
    }
}

/// 引用优化器
#[derive(Clone)]
#[derive(Debug)]
pub struct ReferenceOptimizer {
    /// 配置
    config: MemoryConfig,
    /// 引用使用统计
    reference_stats: Arc<DashMap<String, ReferenceUsageStats>>,
    /// 竞争事件历史
    contention_history: Arc<RwLock<VecDeque<ContentionEvent>>>,
    /// 优化建议
    recommendations: Arc<RwLock<Vec<OptimizationRecommendation>>>,
    /// 线程访问模式
    thread_patterns: Arc<DashMap<ThreadId, HashSet<String>>>,
    /// 统计信息
    stats: Arc<RwLock<ReferenceOptimizerStats>>,
    /// 分析间隔
    analysis_interval: Duration,
    /// 后台任务
    background_tasks: Mutex<Vec<std::thread::JoinHandle<()>>>,
    /// 停止信号
    stop_signal: Arc<AtomicBool>,
}

impl ReferenceOptimizer {
    /// 创建新的引用优化器
    pub fn new(config: &MemoryConfig) -> Result<Self> {
        let optimizer = Self {
            config: config.clone(),
            reference_stats: Arc::new(DashMap::new()),
            contention_history: Arc::new(RwLock::new(VecDeque::new())),
            recommendations: Arc::new(RwLock::new(Vec::new())),
            thread_patterns: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(ReferenceOptimizerStats::default())),
            analysis_interval: Duration::from_secs(300), // 5分钟
            background_tasks: Mutex::new(Vec::new()),
            stop_signal: Arc::new(AtomicBool::new(false)),
        };
        
        // 启动后台分析任务
        optimizer.start_background_analysis()?;
        
        Ok(optimizer)
    }
    
    /// 注册引用
    pub fn register_reference(&self, reference_id: String, reference_type: String) -> Result<()> {
        let stats = ReferenceUsageStats::new(reference_id.clone(), reference_type);
        self.reference_stats.insert(reference_id, stats);
        
        // 更新统计
        {
            let mut global_stats = self.stats.write().unwrap();
            global_stats.tracked_references = self.reference_stats.len();
        }
        
        Ok(())
    }
    
    /// 注销引用
    pub fn unregister_reference(&self, reference_id: &str) -> Result<()> {
        self.reference_stats.remove(reference_id);
        
        // 清理相关的竞争事件
        {
            let mut history = self.contention_history.write().unwrap();
            history.retain(|event| event.reference_id != reference_id);
        }
        
        // 更新统计
        {
            let mut global_stats = self.stats.write().unwrap();
            global_stats.tracked_references = self.reference_stats.len();
        }
        
        Ok(())
    }
    
    /// 记录引用访问
    pub fn record_reference_access(
        &self,
        reference_id: &str,
        operation_type: &str,
        wait_time: Duration,
        hold_time: Duration,
    ) -> Result<()> {
        let thread_id = thread::current().id();
        
        // 记录线程访问模式
        {
            let mut patterns = self.thread_patterns.entry(thread_id).or_insert_with(HashSet::new);
            patterns.insert(reference_id.to_string());
        }
        
        // 更新引用统计
        if let Some(mut stats) = self.reference_stats.get_mut(reference_id) {
            match operation_type {
                "read" | "read_lock" => {
                    stats.record_read_access(wait_time, hold_time);
                },
                "write" | "write_lock" | "lock" => {
                    stats.record_write_access(wait_time, hold_time);
                },
                _ => {
                    // 默认作为读取处理
                    stats.record_read_access(wait_time, hold_time);
                }
            }
        }
        
        // 如果有显著等待时间，记录竞争事件
        if wait_time > Duration::from_millis(1) {
            self.record_contention_event(reference_id, thread_id, wait_time, hold_time, operation_type)?;
        }
        
        Ok(())
    }
    
    /// 记录竞争事件
    fn record_contention_event(
        &self,
        reference_id: &str,
        thread_id: ThreadId,
        wait_time: Duration,
        hold_time: Duration,
        operation_type: &str,
    ) -> Result<()> {
        let now = Instant::now();
        let event = ContentionEvent {
            reference_id: reference_id.to_string(),
            thread_id,
            wait_start: now - wait_time,
            wait_end: now,
            operation_type: operation_type.to_string(),
            hold_duration: hold_time,
        };
        
        // 添加到历史记录
        {
            let mut history = self.contention_history.write().unwrap();
            if history.len() >= 10000 {
                history.pop_front();
            }
            history.push_back(event);
        }
        
        // 更新统计
        {
            let mut stats = self.stats.write().unwrap();
            stats.contention_events += 1;
        }
        
        Ok(())
    }
    
    /// 分析引用使用模式
    pub fn analyze_reference_patterns(&self) -> Result<()> {
        // 分析每个引用的使用模式
        for entry in self.reference_stats.iter() {
            let reference_id = entry.key();
            let stats = entry.value();
            
            // 生成优化建议
            if let Some(recommendation) = self.generate_recommendation(reference_id, stats) {
                let mut recommendations = self.recommendations.write().unwrap();
                
                // 检查是否已存在相同建议
                if !recommendations.iter().any(|r| r.reference_id == recommendation.reference_id) {
                    recommendations.push(recommendation);
                }
            }
        }
        
        // 更新全局统计
        self.update_global_stats()?;
        
        Ok(())
    }
    
    /// 生成优化建议
    fn generate_recommendation(
        &self,
        reference_id: &str,
        stats: &ReferenceUsageStats,
    ) -> Option<OptimizationRecommendation> {
        let pattern = stats.analyze_lock_pattern();
        let read_ratio = stats.read_write_ratio();
        let contention_rate = stats.contention_rate();
        let access_freq = stats.access_frequency();
        
        // 分析当前类型的问题
        let (recommended_type, reason, improvement) = match pattern {
            LockUsagePattern::ReadHeavy if read_ratio > 20.0 => {
                (
                    ReferenceTypeRecommendation::ArcRwLock,
                    "读多写少的场景，RwLock可以提高并发性能".to_string(),
                    0.7,
                )
            },
            LockUsagePattern::HighContention if contention_rate > 0.5 => {
                (
                    ReferenceTypeRecommendation::ThreadLocal,
                    "高竞争场景，考虑使用线程本地存储".to_string(),
                    0.8,
                )
            },
            LockUsagePattern::ShortLived if stats.avg_hold_time.as_millis() < 1 => {
                (
                    ReferenceTypeRecommendation::Atomic,
                    "短期锁持有，原子操作可能更高效".to_string(),
                    0.6,
                )
            },
            LockUsagePattern::LowContention if contention_rate < 0.01 => {
                (
                    ReferenceTypeRecommendation::PlainReference,
                    "低竞争场景，可以考虑去除同步原语".to_string(),
                    0.5,
                )
            },
            _ => return None,
        };
        
        // 检查是否值得优化
        if improvement < 0.3 || access_freq < 0.1 {
            return None;
        }
        
        Some(OptimizationRecommendation::new(
            reference_id.to_string(),
            stats.current_type.clone(),
            recommended_type,
            reason,
            improvement,
        ))
    }
    
    /// 更新全局统计
    fn update_global_stats(&self) -> Result<()> {
        let total_refs = self.reference_stats.len();
        let mut total_ref_count = 0;
        let mut high_contention_count = 0;
        let mut unused_count = 0;
        
        for entry in self.reference_stats.iter() {
            let stats = entry.value();
            total_ref_count += stats.ref_count;
            
            if stats.contention_rate() > 0.3 {
                high_contention_count += 1;
            }
            
            if stats.total_accesses == 0 {
                unused_count += 1;
            }
        }
        
        let avg_ref_count = if total_refs > 0 {
            total_ref_count as f64 / total_refs as f64
        } else {
            0.0
        };
        
        let recommendations_count = self.recommendations.read().unwrap().len();
        
        {
            let mut global_stats = self.stats.write().unwrap();
            global_stats.tracked_references = total_refs;
            global_stats.avg_ref_count = avg_ref_count;
            global_stats.high_contention_refs = high_contention_count;
            global_stats.unused_refs = unused_count;
            global_stats.recommendations = recommendations_count;
        }
        
        Ok(())
    }
    
    /// 获取优化建议
    pub fn get_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = self.recommendations.read().unwrap().clone();
        
        // 按优先级排序
        recommendations.sort_by(|a, b| b.priority.cmp(&a.priority));
        
        recommendations
    }
    
    /// 获取高竞争引用
    pub fn get_high_contention_references(&self) -> Vec<String> {
        self.reference_stats
            .iter()
            .filter(|entry| entry.value().contention_rate() > 0.3)
            .map(|entry| entry.key().clone())
            .collect()
    }
    
    /// 获取未使用的引用
    pub fn get_unused_references(&self) -> Vec<String> {
        let stale_threshold = Duration::from_secs(3600); // 1小时
        
        self.reference_stats
            .iter()
            .filter(|entry| {
                let stats = entry.value();
                stats.total_accesses == 0 || stats.last_access.elapsed() > stale_threshold
            })
            .map(|entry| entry.key().clone())
            .collect()
    }
    
    /// 优化引用使用
    pub fn optimize(&self) -> Result<()> {
        // 分析当前模式
        self.analyze_reference_patterns()?;
        
        // 清理未使用的引用
        self.cleanup_unused_references()?;
        
        // 生成优化报告
        self.generate_optimization_report()?;
        
        Ok(())
    }
    
    /// 清理未使用的引用
    fn cleanup_unused_references(&self) -> Result<usize> {
        let unused_refs = self.get_unused_references();
        let count = unused_refs.len();
        
        for ref_id in unused_refs {
            self.reference_stats.remove(&ref_id);
        }
        
        Ok(count)
    }
    
    /// 生成优化报告
    fn generate_optimization_report(&self) -> Result<()> {
        let stats = self.stats.read().unwrap().clone();
        let recommendations = self.get_recommendations();
        
        println!("=== 引用优化报告 ===");
        println!("跟踪的引用数量: {}", stats.tracked_references);
        println!("竞争事件总数: {}", stats.contention_events);
        println!("高竞争引用数量: {}", stats.high_contention_refs);
        println!("未使用引用数量: {}", stats.unused_refs);
        println!("平均引用计数: {:.2}", stats.avg_ref_count);
        println!("生成的建议数量: {}", recommendations.len());
        
        if !recommendations.is_empty() {
            println!("\n=== 优化建议（按优先级排序）===");
            for (i, rec) in recommendations.iter().take(10).enumerate() {
                println!("{}. {} -> {:?}", i + 1, rec.reference_id, rec.recommended_type);
                println!("   原因: {}", rec.reason);
                println!("   预期改进: {:.1}%", rec.expected_improvement * 100.0);
                println!("   实施难度: {}/10", rec.implementation_difficulty);
                println!();
            }
        }
        
        Ok(())
    }
    
    /// 获取统计信息
    pub fn get_stats(&self) -> ReferenceOptimizerStats {
        self.stats.read().unwrap().clone()
    }
    
    /// 健康检查
    pub fn health_check(&self) -> Result<HealthStatus> {
        let stats = self.get_stats();
        
        let contention_ratio = if stats.tracked_references > 0 {
            stats.high_contention_refs as f64 / stats.tracked_references as f64
        } else {
            0.0
        };
        
        let unused_ratio = if stats.tracked_references > 0 {
            stats.unused_refs as f64 / stats.tracked_references as f64
        } else {
            0.0
        };
        
        if contention_ratio > 0.3 || unused_ratio > 0.5 || stats.avg_ref_count > 10.0 {
            Ok(HealthStatus::Critical)
        } else if contention_ratio > 0.1 || unused_ratio > 0.2 || stats.avg_ref_count > 5.0 {
            Ok(HealthStatus::Warning)
        } else {
            Ok(HealthStatus::Healthy)
        }
    }
    
    /// 启动后台分析任务
    fn start_background_analysis(&self) -> Result<()> {
        let optimizer_arc = Arc::new(self.clone());
        let stop_signal = self.stop_signal.clone();
        let interval = self.analysis_interval;
        
        let handle = thread::spawn(move || {
            while !stop_signal.load(Ordering::Relaxed) {
                if let Err(e) = optimizer_arc.analyze_reference_patterns() {
                    eprintln!("Reference analysis error: {}", e);
                }
                
                thread::sleep(interval);
            }
        });
        
        self.background_tasks.lock().unwrap().push(handle);
        
        Ok(())
    }
}

impl Drop for ReferenceOptimizer {
    fn drop(&mut self) {
        // 停止后台任务
        self.stop_signal.store(true, Ordering::Relaxed);
        
        // 等待后台任务结束
        let mut tasks = self.background_tasks.lock().unwrap();
        for handle in tasks.drain(..) {
            let _ = handle.join();
        }
    }
}

/// 智能引用包装器
pub struct SmartRef<T> {
    data: T,
    reference_id: String,
    optimizer: Weak<ReferenceOptimizer>,
}

impl<T> SmartRef<T> {
    /// 创建智能引用
    pub fn new(data: T, reference_id: String, optimizer: Weak<ReferenceOptimizer>) -> Self {
        if let Some(opt) = optimizer.upgrade() {
            let _ = opt.register_reference(reference_id.clone(), "SmartRef".to_string());
        }
        
        Self {
            data,
            reference_id,
            optimizer,
        }
    }
    
    /// 获取数据
    pub fn get(&self) -> &T {
        let start_time = Instant::now();
        
        // 记录访问
        if let Some(opt) = self.optimizer.upgrade() {
            let access_time = start_time.elapsed();
            let _ = opt.record_reference_access(
                &self.reference_id,
                "read",
                Duration::ZERO,
                access_time,
            );
        }
        
        &self.data
    }
    
    /// 获取可变数据
    pub fn get_mut(&mut self) -> &mut T {
        let start_time = Instant::now();
        
        // 记录写入访问
        if let Some(opt) = self.optimizer.upgrade() {
            let access_time = start_time.elapsed();
            let _ = opt.record_reference_access(
                &self.reference_id,
                "write",
                Duration::ZERO,
                access_time,
            );
        }
        
        &mut self.data
    }
}

impl<T> Drop for SmartRef<T> {
    fn drop(&mut self) {
        if let Some(opt) = self.optimizer.upgrade() {
            let _ = opt.unregister_reference(&self.reference_id);
        }
    }
}

/// 便捷宏用于创建智能引用
#[macro_export]
macro_rules! smart_ref {
    ($data:expr, $id:expr, $optimizer:expr) => {
        $crate::memory::reference_optimizer::SmartRef::new($data, $id.to_string(), $optimizer)
    };
} 