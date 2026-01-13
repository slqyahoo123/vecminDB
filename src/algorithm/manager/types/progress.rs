use std::collections::HashMap;
use std::time::{SystemTime, Duration};
use serde::{Serialize, Deserialize};

/// 任务进度信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskProgress {
    pub task_id: String,
    pub current_step: usize,
    pub total_steps: usize,
    pub percentage: f64,
    pub phase: ProgressPhase,
    pub status: ProgressStatus,
    pub metrics: HashMap<String, f64>,
    pub start_time: SystemTime,
    pub current_time: SystemTime,
    pub estimated_completion: Option<SystemTime>,
    pub error_message: Option<String>,
}

/// 进度阶段
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProgressPhase {
    Initialization,
    DataPreparation,
    ModelTraining,
    Validation,
    Testing,
    Deployment,
    Completed,
    Failed,
    Cancelled,
}

/// 进度状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProgressStatus {
    NotStarted,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
    Warning,
}

/// 详细进度信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressInfo {
    pub progress: TaskProgress,
    pub sub_progress: Vec<SubProgress>,
    pub logs: Vec<ProgressLog>,
    pub resource_usage: ResourceUsageInfo,
    pub performance_metrics: PerformanceMetrics,
}

/// 子进度信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubProgress {
    pub name: String,
    pub current_step: usize,
    pub total_steps: usize,
    pub percentage: f64,
    pub status: ProgressStatus,
    pub start_time: SystemTime,
    pub end_time: Option<SystemTime>,
}

/// 进度日志
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressLog {
    pub timestamp: SystemTime,
    pub level: LogLevel,
    pub message: String,
    pub category: String,
    pub metadata: HashMap<String, String>,
}

/// 日志级别
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

/// 资源使用信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageInfo {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub gpu_usage: Option<f64>,
    pub disk_usage: f64,
    pub network_usage: f64,
    pub timestamp: SystemTime,
}

/// 性能指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub throughput: f64,
    pub latency: Duration,
    pub accuracy: Option<f64>,
    pub loss: Option<f64>,
    pub custom_metrics: HashMap<String, f64>,
    pub timestamp: SystemTime,
}

/// 进度回调函数类型
pub type ProgressCallback = Box<dyn Fn(&ProgressInfo) + Send + Sync>;

/// 进度跟踪器
#[derive(Debug)]
pub struct ProgressTracker {
    pub task_id: String,
    pub progress: TaskProgress,
    pub callbacks: Vec<ProgressCallback>,
    pub auto_update_interval: Duration,
    pub last_update: SystemTime,
}

impl TaskProgress {
    /// 创建新的任务进度
    pub fn new(task_id: String, total_steps: usize) -> Self {
        let now = SystemTime::now();
        Self {
            task_id,
            current_step: 0,
            total_steps,
            percentage: 0.0,
            phase: ProgressPhase::Initialization,
            status: ProgressStatus::NotStarted,
            metrics: HashMap::new(),
            start_time: now,
            current_time: now,
            estimated_completion: None,
            error_message: None,
        }
    }

    /// 更新进度
    pub fn update_progress(&mut self, current_step: usize) {
        self.current_step = current_step;
        self.percentage = if self.total_steps > 0 {
            (current_step as f64 / self.total_steps as f64) * 100.0
        } else {
            0.0
        };
        self.current_time = SystemTime::now();
        
        // 估算完成时间
        if current_step > 0 && self.percentage > 0.0 {
            let elapsed = self.current_time.duration_since(self.start_time).unwrap_or(Duration::ZERO);
            let estimated_total = Duration::from_secs((elapsed.as_secs() as f64 / (self.percentage / 100.0)) as u64);
            self.estimated_completion = Some(self.start_time + estimated_total);
        }
    }

    /// 设置阶段
    pub fn set_phase(&mut self, phase: ProgressPhase) {
        self.phase = phase;
        self.current_time = SystemTime::now();
    }

    /// 设置状态
    pub fn set_status(&mut self, status: ProgressStatus) {
        self.status = status;
        self.current_time = SystemTime::now();
    }

    /// 添加指标
    pub fn add_metric(&mut self, name: String, value: f64) {
        self.metrics.insert(name, value);
        self.current_time = SystemTime::now();
    }

    /// 设置错误
    pub fn set_error(&mut self, error_message: String) {
        self.error_message = Some(error_message);
        self.status = ProgressStatus::Failed;
        self.current_time = SystemTime::now();
    }

    /// 完成任务
    pub fn complete(&mut self) {
        self.current_step = self.total_steps;
        self.percentage = 100.0;
        self.status = ProgressStatus::Completed;
        self.phase = ProgressPhase::Completed;
        self.current_time = SystemTime::now();
        self.estimated_completion = Some(self.current_time);
    }

    /// 计算剩余时间
    pub fn remaining_time(&self) -> Option<Duration> {
        if let Some(estimated_completion) = self.estimated_completion {
            estimated_completion.duration_since(self.current_time).ok()
        } else {
            None
        }
    }

    /// 获取运行时间
    pub fn elapsed_time(&self) -> Duration {
        self.current_time.duration_since(self.start_time).unwrap_or(Duration::ZERO)
    }

    /// 检查是否完成
    pub fn is_completed(&self) -> bool {
        matches!(self.status, ProgressStatus::Completed)
    }

    /// 检查是否失败
    pub fn is_failed(&self) -> bool {
        matches!(self.status, ProgressStatus::Failed)
    }

    /// 检查是否运行中
    pub fn is_running(&self) -> bool {
        matches!(self.status, ProgressStatus::Running)
    }
}

impl ProgressTracker {
    /// 创建新的进度跟踪器
    pub fn new(task_id: String, total_steps: usize) -> Self {
        let progress = TaskProgress::new(task_id.clone(), total_steps);
        Self {
            task_id,
            progress,
            callbacks: Vec::new(),
            auto_update_interval: Duration::from_secs(1),
            last_update: SystemTime::now(),
        }
    }

    /// 添加回调函数
    pub fn add_callback(&mut self, callback: ProgressCallback) {
        self.callbacks.push(callback);
    }

    /// 更新进度并触发回调
    pub fn update(&mut self, current_step: usize) {
        self.progress.update_progress(current_step);
        self.trigger_callbacks();
    }

    /// 设置阶段
    pub fn set_phase(&mut self, phase: ProgressPhase) {
        self.progress.set_phase(phase);
        self.trigger_callbacks();
    }

    /// 设置状态
    pub fn set_status(&mut self, status: ProgressStatus) {
        self.progress.set_status(status);
        self.trigger_callbacks();
    }

    /// 添加指标
    pub fn add_metric(&mut self, name: String, value: f64) {
        self.progress.add_metric(name, value);
        self.trigger_callbacks_if_interval_passed();
    }

    /// 设置错误
    pub fn set_error(&mut self, error_message: String) {
        self.progress.set_error(error_message);
        self.trigger_callbacks();
    }

    /// 完成任务
    pub fn complete(&mut self) {
        self.progress.complete();
        self.trigger_callbacks();
    }

    /// 触发所有回调函数
    fn trigger_callbacks(&mut self) {
        let progress_info = ProgressInfo {
            progress: self.progress.clone(),
            sub_progress: Vec::new(),
            logs: Vec::new(),
            resource_usage: ResourceUsageInfo {
                cpu_usage: 0.0,
                memory_usage: 0.0,
                gpu_usage: None,
                disk_usage: 0.0,
                network_usage: 0.0,
                timestamp: SystemTime::now(),
            },
            performance_metrics: PerformanceMetrics {
                throughput: 0.0,
                latency: Duration::ZERO,
                accuracy: None,
                loss: None,
                custom_metrics: HashMap::new(),
                timestamp: SystemTime::now(),
            },
        };

        for callback in &self.callbacks {
            callback(&progress_info);
        }
        self.last_update = SystemTime::now();
    }

    /// 如果间隔时间已过则触发回调
    fn trigger_callbacks_if_interval_passed(&mut self) {
        let now = SystemTime::now();
        if now.duration_since(self.last_update).unwrap_or(Duration::ZERO) >= self.auto_update_interval {
            self.trigger_callbacks();
        }
    }
}

impl ProgressInfo {
    /// 创建新的进度信息
    pub fn new(progress: TaskProgress) -> Self {
        Self {
            progress,
            sub_progress: Vec::new(),
            logs: Vec::new(),
            resource_usage: ResourceUsageInfo {
                cpu_usage: 0.0,
                memory_usage: 0.0,
                gpu_usage: None,
                disk_usage: 0.0,
                network_usage: 0.0,
                timestamp: SystemTime::now(),
            },
            performance_metrics: PerformanceMetrics {
                throughput: 0.0,
                latency: Duration::ZERO,
                accuracy: None,
                loss: None,
                custom_metrics: HashMap::new(),
                timestamp: SystemTime::now(),
            },
        }
    }

    /// 添加子进度
    pub fn add_sub_progress(&mut self, sub_progress: SubProgress) {
        self.sub_progress.push(sub_progress);
    }

    /// 添加日志
    pub fn add_log(&mut self, log: ProgressLog) {
        self.logs.push(log);
    }

    /// 更新资源使用情况
    pub fn update_resource_usage(&mut self, resource_usage: ResourceUsageInfo) {
        self.resource_usage = resource_usage;
    }

    /// 更新性能指标
    pub fn update_performance_metrics(&mut self, performance_metrics: PerformanceMetrics) {
        self.performance_metrics = performance_metrics;
    }

    /// 获取总体进度百分比
    pub fn overall_percentage(&self) -> f64 {
        if self.sub_progress.is_empty() {
            self.progress.percentage
        } else {
            let total: f64 = self.sub_progress.iter().map(|sp| sp.percentage).sum();
            total / self.sub_progress.len() as f64
        }
    }
}

impl ProgressLog {
    /// 创建新的进度日志
    pub fn new(level: LogLevel, message: String, category: String) -> Self {
        Self {
            timestamp: SystemTime::now(),
            level,
            message,
            category,
            metadata: HashMap::new(),
        }
    }

    /// 添加元数据
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
} 