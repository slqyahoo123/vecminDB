use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use log::{debug, info};

/// 进度追踪器
pub struct ProgressTracker {
    /// 总记录数
    total_records: AtomicUsize,
    /// 已处理记录数
    processed_records: AtomicUsize,
    /// 开始时间
    start_time: Instant,
    /// 进度回调
    progress_callback: Option<Box<dyn Fn(usize, usize) + Send + Sync>>,
}

impl ProgressTracker {
    /// 创建新的进度追踪器
    pub fn new() -> Self {
        Self {
            total_records: AtomicUsize::new(0),
            processed_records: AtomicUsize::new(0),
            start_time: Instant::now(),
            progress_callback: None,
        }
    }
    
    /// 设置总记录数
    pub fn set_total(&self, total: usize) {
        self.total_records.store(total, Ordering::SeqCst);
    }
    
    /// 更新已处理记录数
    pub fn update(&self, processed: usize) {
        self.processed_records.store(processed, Ordering::SeqCst);
        
        // 调用进度回调
        if let Some(callback) = &self.progress_callback {
            callback(
                self.processed_records.load(Ordering::SeqCst),
                self.total_records.load(Ordering::SeqCst)
            );
        }
    }
    
    /// 增加已处理记录数
    pub fn increment(&self, count: usize) {
        let processed = self.processed_records.fetch_add(count, Ordering::SeqCst) + count;
        
        // 调用进度回调
        if let Some(callback) = &self.progress_callback {
            callback(
                processed,
                self.total_records.load(Ordering::SeqCst)
            );
        }
    }
    
    /// 获取进度百分比
    pub fn percentage(&self) -> f32 {
        let processed = self.processed_records.load(Ordering::SeqCst);
        let total = self.total_records.load(Ordering::SeqCst);
        
        if total == 0 {
            return 0.0;
        }
        
        (processed as f32 / total as f32) * 100.0
    }
    
    /// 获取已用时间
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// 获取已处理的记录数
    pub fn processed(&self) -> usize {
        self.processed_records.load(Ordering::SeqCst)
    }

    /// 获取总记录数
    pub fn total(&self) -> usize {
        self.total_records.load(Ordering::SeqCst)
    }
    
    /// 获取估计剩余时间
    pub fn estimated_time_remaining(&self) -> Option<Duration> {
        let processed = self.processed_records.load(Ordering::SeqCst);
        let total = self.total_records.load(Ordering::SeqCst);
        let elapsed = self.elapsed();
        
        if processed == 0 || total == 0 {
            return None;
        }
        
        let elapsed_secs = elapsed.as_secs_f64();
        let secs_per_record = elapsed_secs / processed as f64;
        let remaining_records = total.saturating_sub(processed);
        let remaining_secs = secs_per_record * remaining_records as f64;
        
        Some(Duration::from_secs_f64(remaining_secs))
    }
    
    /// 设置进度回调
    pub fn set_progress_callback<F>(&mut self, callback: F)
    where
        F: Fn(usize, usize) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
    }

    /// 记录当前进度信息到日志
    pub fn log_progress(&self, prefix: &str) {
        let percentage = self.percentage();
        let elapsed = self.elapsed();
        let processed = self.processed();
        let total = self.total();
        
        info!("{}: 进度 {:.1}% ({}/{}), 已用时间: {:?}", 
            prefix, percentage, processed, total, elapsed);
        
        if let Some(remaining) = self.estimated_time_remaining() {
            debug!("{}: 预计剩余时间: {:?}", prefix, remaining);
        }
    }

    /// 重置进度追踪器
    pub fn reset(&mut self) {
        self.total_records.store(0, Ordering::SeqCst);
        self.processed_records.store(0, Ordering::SeqCst);
        self.start_time = Instant::now();
    }
}

/// 进度报告器，用于定期报告进度
pub struct ProgressReporter {
    tracker: ProgressTracker,
    report_interval: Duration,
    last_report_time: Instant,
    prefix: String,
}

impl ProgressReporter {
    /// 创建新的进度报告器
    pub fn new(prefix: &str, report_interval_secs: u64) -> Self {
        Self {
            tracker: ProgressTracker::new(),
            report_interval: Duration::from_secs(report_interval_secs),
            last_report_time: Instant::now(),
            prefix: prefix.to_string(),
        }
    }

    /// 获取进度追踪器引用
    pub fn tracker(&self) -> &ProgressTracker {
        &self.tracker
    }

    /// 获取可变进度追踪器引用
    pub fn tracker_mut(&mut self) -> &mut ProgressTracker {
        &mut self.tracker
    }

    /// 更新进度并根据需要报告
    pub fn update(&mut self, processed: usize) {
        self.tracker.update(processed);
        self.report_if_needed();
    }

    /// 增加进度并根据需要报告
    pub fn increment(&mut self, count: usize) {
        self.tracker.increment(count);
        self.report_if_needed();
    }

    /// 设置总记录数并根据需要报告
    pub fn set_total(&mut self, total: usize) {
        self.tracker.set_total(total);
        self.report_if_needed();
    }

    /// 如果达到报告间隔，报告进度
    fn report_if_needed(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_report_time) >= self.report_interval {
            self.tracker.log_progress(&self.prefix);
            self.last_report_time = now;
        }
    }

    /// 强制报告当前进度
    pub fn force_report(&mut self) {
        self.tracker.log_progress(&self.prefix);
        self.last_report_time = Instant::now();
    }
} 