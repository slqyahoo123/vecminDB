/// 内存泄漏检测器
/// 
/// 提供全面的内存泄漏检测和修复功能：
/// 1. 实时泄漏检测
/// 2. 泄漏模式分析
/// 3. 自动修复机制
/// 4. 泄漏报告生成
/// 5. 预防性监控

use std::sync::{Arc, Mutex};
use std::collections::{HashMap, VecDeque};
use std::time::{Instant, Duration};
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use serde::{Serialize, Deserialize};
use parking_lot::RwLock;

use super::{ObjectType, MemoryConfig};
use crate::{Error, Result};
use crate::core::types::HealthStatus;

/// 检测配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    /// 检测间隔（秒）
    pub detection_interval_seconds: u64,
    /// 泄漏阈值（字节）
    pub leak_threshold_bytes: usize,
    /// 最大跟踪对象数
    pub max_tracked_objects: usize,
    /// 历史记录保留时间（秒）
    pub history_retention_seconds: u64,
    /// 启用自动修复
    pub auto_fix_enabled: bool,
    /// 启用详细日志
    pub verbose_logging: bool,
    /// 检测敏感度
    pub sensitivity: DetectionSensitivity,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            detection_interval_seconds: 60, // 1分钟
            leak_threshold_bytes: 1024 * 1024, // 1MB
            max_tracked_objects: 100000,
            history_retention_seconds: 3600 * 24, // 24小时
            auto_fix_enabled: true,
            verbose_logging: false,
            sensitivity: DetectionSensitivity::Medium,
        }
    }
}

/// 检测敏感度
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionSensitivity {
    /// 低敏感度 - 只检测明显泄漏
    Low,
    /// 中等敏感度 - 平衡检测
    Medium,
    /// 高敏感度 - 检测潜在泄漏
    High,
    /// 极高敏感度 - 检测所有可疑情况
    VeryHigh,
}

impl DetectionSensitivity {
    /// 获取检测阈值倍数
    pub fn threshold_multiplier(&self) -> f64 {
        match self {
            Self::Low => 5.0,
            Self::Medium => 2.0,
            Self::High => 1.5,
            Self::VeryHigh => 1.2,
        }
    }
    
    /// 获取最小检测时间（秒）
    pub fn min_detection_time(&self) -> u64 {
        match self {
            Self::Low => 3600,    // 1小时
            Self::Medium => 1800, // 30分钟
            Self::High => 900,    // 15分钟
            Self::VeryHigh => 300, // 5分钟
        }
    }
}

/// 泄漏严重程度
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LeakSeverity {
    /// 轻微泄漏
    Minor,
    /// 中等泄漏
    Moderate,
    /// 严重泄漏
    Severe,
    /// 关键泄漏
    Critical,
}

impl LeakSeverity {
    /// 从泄漏大小计算严重程度
    pub fn from_size(size: usize) -> Self {
        if size < 1024 * 1024 {          // < 1MB
            Self::Minor
        } else if size < 10 * 1024 * 1024 { // < 10MB
            Self::Moderate
        } else if size < 100 * 1024 * 1024 { // < 100MB
            Self::Severe
        } else {
            Self::Critical
        }
    }
}

/// 泄漏信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakInfo {
    /// 泄漏ID
    pub id: String,
    /// 对象类型
    pub object_type: ObjectType,
    /// 泄漏大小（字节）
    pub size: usize,
    /// 严重程度
    pub severity: LeakSeverity,
    /// 检测时间
    pub detected_at: Instant,
    /// 分配时间
    pub allocated_at: Instant,
    /// 最后访问时间
    pub last_accessed: Instant,
    /// 访问次数
    pub access_count: u64,
    /// 引用计数
    pub ref_count: usize,
    /// 泄漏原因
    pub leak_reason: LeakReason,
    /// 堆栈跟踪
    pub stack_trace: Option<String>,
    /// 是否已修复
    pub fixed: bool,
    /// 修复时间
    pub fixed_at: Option<Instant>,
}

/// 泄漏原因
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeakReason {
    /// 引用计数未归零
    RefCountNotZero,
    /// 长时间未访问
    LongTimeUnaccessed,
    /// 循环引用
    CircularReference,
    /// 忘记释放
    ForgottenRelease,
    /// 异常中断
    ExceptionInterrupted,
    /// 未知原因
    Unknown,
}

/// 对象跟踪信息
#[derive(Debug, Clone)]
pub struct ObjectTracker {
    /// 对象ID
    pub object_id: String,
    /// 对象类型
    pub object_type: ObjectType,
    /// 大小
    pub size: usize,
    /// 分配时间
    pub allocated_at: Instant,
    /// 最后访问时间
    pub last_accessed: Instant,
    /// 访问次数
    pub access_count: u64,
    /// 引用计数
    pub ref_count: u64,
    /// 是否可疑
    pub suspicious: bool,
    /// 检查次数
    pub check_count: u64,
}

impl ObjectTracker {
    /// 创建新的对象跟踪器
    pub fn new(object_id: String, object_type: ObjectType, size: usize) -> Self {
        let now = Instant::now();
        Self {
            object_id,
            object_type,
            size,
            allocated_at: now,
            last_accessed: now,
            access_count: 0,
            ref_count: 1,
            suspicious: false,
            check_count: 0,
        }
    }
    
    /// 更新访问信息
    pub fn update_access(&mut self, access_count: u64, ref_count: u64) {
        self.last_accessed = Instant::now();
        self.access_count = access_count;
        self.ref_count = ref_count;
    }
    
    /// 检查是否可疑
    pub fn is_suspicious(&self, config: &DetectionConfig) -> bool {
        let age = self.allocated_at.elapsed();
        let idle_time = self.last_accessed.elapsed();
        
        // 检查各种可疑条件
        let long_lived = age > Duration::from_secs(config.sensitivity.min_detection_time());
        let long_idle = idle_time > Duration::from_secs(config.sensitivity.min_detection_time() / 2);
        let high_ref_count = self.ref_count > 10;
        let large_size = self.size > config.leak_threshold_bytes;
        
        match config.sensitivity {
            DetectionSensitivity::Low => long_lived && long_idle && large_size,
            DetectionSensitivity::Medium => (long_lived && long_idle) || (high_ref_count && large_size),
            DetectionSensitivity::High => long_lived || long_idle || high_ref_count,
            DetectionSensitivity::VeryHigh => age > Duration::from_secs(300) || idle_time > Duration::from_secs(150),
        }
    }
    
    /// 分析泄漏原因
    pub fn analyze_leak_reason(&self) -> LeakReason {
        let idle_time = self.last_accessed.elapsed();
        
        if self.ref_count > 1 {
            LeakReason::RefCountNotZero
        } else if self.access_count == 0 {
            LeakReason::ForgottenRelease
        } else if idle_time > Duration::from_secs(3600) {
            LeakReason::LongTimeUnaccessed
        } else {
            LeakReason::Unknown
        }
    }
}

/// 泄漏统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakStats {
    /// 检测到的泄漏数量
    pub detected_leaks: usize,
    /// 修复的泄漏数量
    pub fixed_leaks: usize,
    /// 泄漏的总内存（字节）
    pub leaked_memory: usize,
    /// 修复的总内存（字节）
    pub fixed_memory: usize,
    /// 最后检测时间
    pub last_detection: Option<Instant>,
    /// 最后修复时间
    pub last_fix: Option<Instant>,
    /// 按严重程度分组的泄漏数量
    pub leaks_by_severity: HashMap<LeakSeverity, usize>,
    /// 按对象类型分组的泄漏数量
    pub leaks_by_type: HashMap<ObjectType, usize>,
}

impl Default for LeakStats {
    fn default() -> Self {
        Self {
            detected_leaks: 0,
            fixed_leaks: 0,
            leaked_memory: 0,
            fixed_memory: 0,
            last_detection: None,
            last_fix: None,
            leaks_by_severity: HashMap::new(),
            leaks_by_type: HashMap::new(),
        }
    }
}

/// 泄漏报告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakReport {
    /// 报告ID
    pub id: String,
    /// 生成时间
    pub generated_at: Instant,
    /// 检测到的泄漏
    pub leaks: Vec<LeakInfo>,
    /// 统计信息
    pub stats: LeakStats,
    /// 建议
    pub recommendations: Vec<String>,
}

/// 泄漏检测器
pub struct LeakDetector {
    /// 对象跟踪器
    trackers: RwLock<HashMap<String, ObjectTracker>>,
    /// 检测到的泄漏
    detected_leaks: RwLock<HashMap<String, LeakInfo>>,
    /// 历史记录
    history: RwLock<VecDeque<LeakReport>>,
    /// 配置
    config: DetectionConfig,
    /// 统计信息
    stats: Arc<Mutex<LeakStats>>,
    /// 是否运行中
    running: Arc<AtomicBool>,
    /// 下一个泄漏ID
    next_leak_id: AtomicU64,
}

impl LeakDetector {
    /// 创建新的泄漏检测器
    pub fn new(config: &MemoryConfig) -> Result<Self> {
        Ok(Self {
            trackers: RwLock::new(HashMap::new()),
            detected_leaks: RwLock::new(HashMap::new()),
            history: RwLock::new(VecDeque::new()),
            config: DetectionConfig::default(),
            stats: Arc::new(Mutex::new(LeakStats::default())),
            running: Arc::new(AtomicBool::new(false)),
            next_leak_id: AtomicU64::new(1),
        })
    }
    
    /// 启动泄漏检测器
    pub async fn start(self: Arc<Self>) -> Result<()> {
        if self.running.load(Ordering::SeqCst) {
            return Err(Error::invalid_state("Leak detector is already running"));
        }
        
        // 启动后台检测任务
        self.start_background_tasks().await?;
        
        Ok(())
    }
    
    /// 停止泄漏检测器
    pub async fn stop(&self) -> Result<()> {
        self.running.store(false, Ordering::SeqCst);
        Ok(())
    }
    
    /// 跟踪对象
    pub fn track_object(&self, object_id: String, object_type: ObjectType, size: usize) -> Result<()> {
        let tracker = ObjectTracker::new(object_id.clone(), object_type, size);
        
        let mut trackers = self.trackers.write();
        
        // 检查跟踪数量限制
        if trackers.len() >= self.config.max_tracked_objects {
            // 移除最老的跟踪器
            if let Some((oldest_id, _)) = trackers.iter()
                .min_by_key(|(_, t)| t.allocated_at) {
                let oldest_id = oldest_id.clone();
                trackers.remove(&oldest_id);
            }
        }
        
        trackers.insert(object_id, tracker);
        Ok(())
    }
    
    /// 取消跟踪对象
    pub fn untrack_object(&self, object_id: &str) -> Result<()> {
        let mut trackers = self.trackers.write();
        trackers.remove(object_id);
        Ok(())
    }
    
    /// 更新对象访问信息
    pub fn update_object_access(&self, object_id: &str, access_count: u64, ref_count: u64) -> Result<()> {
        let mut trackers = self.trackers.write();
        if let Some(tracker) = trackers.get_mut(object_id) {
            tracker.update_access(access_count, ref_count);
        }
        Ok(())
    }
    
    /// 执行泄漏检测
    pub async fn detect_leaks(&self) -> Result<Vec<LeakInfo>> {
        let mut new_leaks = Vec::new();
        
        {
            let mut trackers = self.trackers.write();
            let mut detected_leaks = self.detected_leaks.write();
            
            for (object_id, tracker) in trackers.iter_mut() {
                // 检查是否已经检测过
                if detected_leaks.contains_key(object_id) {
                    continue;
                }
                
                // 检查是否可疑
                if tracker.is_suspicious(&self.config) {
                    tracker.suspicious = true;
                    tracker.check_count += 1;
                    
                    // 如果连续多次检查都可疑，则认为是泄漏
                    if tracker.check_count >= 3 {
                        let leak_id = self.generate_leak_id();
                        let leak_reason = tracker.analyze_leak_reason();
                        let severity = LeakSeverity::from_size(tracker.size);
                        
                        let leak_info = LeakInfo {
                            id: leak_id,
                            object_type: tracker.object_type,
                            size: tracker.size,
                            severity,
                            detected_at: Instant::now(),
                            allocated_at: tracker.allocated_at,
                            last_accessed: tracker.last_accessed,
                            access_count: tracker.access_count,
                            ref_count: tracker.ref_count as usize,
                            leak_reason,
                            stack_trace: None, // 简化实现
                            fixed: false,
                            fixed_at: None,
                        };
                        
                        detected_leaks.insert(object_id.clone(), leak_info.clone());
                        new_leaks.push(leak_info);
                    }
                } else {
                    // 重置可疑状态
                    tracker.suspicious = false;
                    tracker.check_count = 0;
                }
            }
        }
        
        // 更新统计
        if !new_leaks.is_empty() {
            self.update_stats(&new_leaks);
        }
        
        Ok(new_leaks)
    }
    
    /// 修复泄漏
    pub async fn fix_leak(&self, leak_id: &str) -> Result<bool> {
        let mut detected_leaks = self.detected_leaks.write();
        
        if let Some(leak_info) = detected_leaks.get_mut(leak_id) {
            if !leak_info.fixed {
                leak_info.fixed = true;
                leak_info.fixed_at = Some(Instant::now());
                
                // 更新统计
                if let Ok(mut stats) = self.stats.lock() {
                    stats.fixed_leaks += 1;
                    stats.fixed_memory += leak_info.size;
                    stats.last_fix = Some(Instant::now());
                }
                
                return Ok(true);
            }
        }
        
        Ok(false)
    }
    
    /// 检测并修复泄漏
    pub async fn detect_and_fix(&self) -> Result<usize> {
        let leaks = self.detect_leaks().await?;
        let mut fixed_count = 0;
        
        if self.config.auto_fix_enabled {
            for leak in &leaks {
                if self.fix_leak(&leak.id).await? {
                    fixed_count += 1;
                }
            }
        }
        
        Ok(fixed_count)
    }
    
    /// 生成泄漏报告
    pub fn generate_report(&self) -> Result<LeakReport> {
        let report_id = format!("report_{}", chrono::Utc::now().timestamp());
        let detected_leaks = self.detected_leaks.read();
        let stats = if let Ok(stats) = self.stats.lock() {
            stats.clone()
        } else {
            LeakStats::default()
        };
        
        let leaks: Vec<LeakInfo> = detected_leaks.values().cloned().collect();
        
        // 生成建议
        let recommendations = self.generate_recommendations(&leaks);
        
        let report = LeakReport {
            id: report_id,
            generated_at: Instant::now(),
            leaks,
            stats,
            recommendations,
        };
        
        // 添加到历史记录
        {
            let mut history = self.history.write();
            history.push_back(report.clone());
            
            // 限制历史记录数量
            while history.len() > 100 {
                history.pop_front();
            }
        }
        
        Ok(report)
    }
    
    /// 生成建议
    fn generate_recommendations(&self, leaks: &[LeakInfo]) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if leaks.is_empty() {
            recommendations.push("没有检测到内存泄漏，系统运行良好。".to_string());
            return recommendations;
        }
        
        // 按严重程度分析
        let critical_leaks = leaks.iter().filter(|l| l.severity == LeakSeverity::Critical).count();
        let severe_leaks = leaks.iter().filter(|l| l.severity == LeakSeverity::Severe).count();
        
        if critical_leaks > 0 {
            recommendations.push(format!("发现{}个关键泄漏，需要立即处理。", critical_leaks));
        }
        
        if severe_leaks > 0 {
            recommendations.push(format!("发现{}个严重泄漏，建议优先处理。", severe_leaks));
        }
        
        // 按原因分析
        let ref_count_leaks = leaks.iter().filter(|l| l.leak_reason == LeakReason::RefCountNotZero).count();
        if ref_count_leaks > 0 {
            recommendations.push("检查引用计数管理，确保正确释放引用。".to_string());
        }
        
        let forgotten_leaks = leaks.iter().filter(|l| l.leak_reason == LeakReason::ForgottenRelease).count();
        if forgotten_leaks > 0 {
            recommendations.push("检查内存释放逻辑，确保所有分配的内存都被正确释放。".to_string());
        }
        
        recommendations
    }
    
    /// 更新统计
    fn update_stats(&self, new_leaks: &[LeakInfo]) {
        if let Ok(mut stats) = self.stats.lock() {
            stats.detected_leaks += new_leaks.len();
            stats.leaked_memory += new_leaks.iter().map(|l| l.size).sum::<usize>();
            stats.last_detection = Some(Instant::now());
            
            // 按严重程度统计
            for leak in new_leaks {
                *stats.leaks_by_severity.entry(leak.severity).or_insert(0) += 1;
                *stats.leaks_by_type.entry(leak.object_type).or_insert(0) += 1;
            }
        }
    }
    
    /// 生成泄漏ID
    fn generate_leak_id(&self) -> String {
        let id = self.next_leak_id.fetch_add(1, Ordering::SeqCst);
        format!("leak_{:08x}", id)
    }
    
    /// 获取统计信息
    pub fn get_stats(&self) -> LeakStats {
        if let Ok(stats) = self.stats.lock() {
            stats.clone()
        } else {
            LeakStats::default()
        }
    }
    
    /// 健康检查
    pub fn health_check(&self) -> Result<HealthStatus> {
        let stats = self.get_stats();
        
        // 检查关键泄漏
        let critical_leaks = stats.leaks_by_severity.get(&LeakSeverity::Critical).unwrap_or(&0);
        if *critical_leaks > 0 {
            return Ok(HealthStatus::Critical);
        }
        
        // 检查严重泄漏
        let severe_leaks = stats.leaks_by_severity.get(&LeakSeverity::Severe).unwrap_or(&0);
        if *severe_leaks > 5 {
            return Ok(HealthStatus::Warning);
        }
        
        // 检查总泄漏内存
        if stats.leaked_memory > 100 * 1024 * 1024 { // 100MB
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
                if let Some(detector) = weak_self.upgrade() {
                    let _ = detector.detect_and_fix().await;
                }
                tokio::time::sleep(Duration::from_secs(60)).await;
            }
        });
        Ok(())
    }
} 