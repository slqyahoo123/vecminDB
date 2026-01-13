use std::collections::HashMap;
use std::time::SystemTime;
use crate::algorithm::base_types::{AlgorithmStatus, LogLevel, StatusLog};
use crate::algorithm::resource::AlgorithmResourceUsage;

/// 算法状态跟踪器
pub struct AlgorithmStatusTracker {
    /// 当前状态
    current_status: AlgorithmStatus,
    /// 完成百分比
    progress: f32,
    /// 开始时间
    start_time: Option<SystemTime>,
    /// 结束时间
    end_time: Option<SystemTime>,
    /// 运行详情记录
    logs: Vec<StatusLog>,
    /// 错误记录
    errors: Vec<String>,
    /// 状态更新历史
    status_history: Vec<(AlgorithmStatus, SystemTime)>,
    /// 性能指标
    metrics: HashMap<String, f32>,
    /// 资源使用情况
    resource_usage: Option<AlgorithmResourceUsage>,
}

impl AlgorithmStatusTracker {
    /// 创建新的状态跟踪器
    pub fn new() -> Self {
        Self {
            current_status: AlgorithmStatus::Pending,
            progress: 0.0,
            start_time: None,
            end_time: None,
            logs: Vec::new(),
            errors: Vec::new(),
            status_history: Vec::new(),
            metrics: HashMap::new(),
            resource_usage: None,
        }
    }
    
    /// 更新状态
    pub fn update_status(&mut self, status: AlgorithmStatus) {
        let now = SystemTime::now();
        
        // 记录状态变化
        self.status_history.push((self.current_status, now));
        self.current_status = status;
        
        // 更新时间戳
        match status {
            AlgorithmStatus::Running if self.start_time.is_none() => {
                self.start_time = Some(now);
            }
            AlgorithmStatus::Completed | AlgorithmStatus::Failed | AlgorithmStatus::Canceled => {
                self.end_time = Some(now);
            }
            _ => {}
        }
        
        // 记录状态变化日志
        let status_msg = format!("状态变更: {:?}", status);
        self.add_log(LogLevel::Info, &status_msg, "status_tracker", None);
    }
    
    /// 更新进度
    pub fn update_progress(&mut self, progress: f32) {
        self.progress = progress.clamp(0.0, 100.0);
        
        // 记录进度日志
        if progress % 10.0 < 1.0 || progress >= 100.0 {
            let progress_msg = format!("进度更新: {:.1}%", progress);
            self.add_log(LogLevel::Info, &progress_msg, "progress_tracker", None);
        }
    }
    
    /// 添加日志
    pub fn add_log(&mut self, level: LogLevel, message: &str, component: &str, context: Option<HashMap<String, String>>) {
        let log = StatusLog {
            timestamp: SystemTime::now(),
            level,
            message: message.to_string(),
            component: component.to_string(),
            context,
        };
        
        self.logs.push(log);
        
        // 如果是错误日志，同时添加到错误记录
        if matches!(level, LogLevel::Error) {
            self.errors.push(message.to_string());
        }
    }
    
    /// 添加性能指标
    pub fn add_metric(&mut self, name: &str, value: f32) {
        self.metrics.insert(name.to_string(), value);
    }
    
    /// 更新资源使用情况
    pub fn update_resource_usage(&mut self, usage: AlgorithmResourceUsage) {
        self.resource_usage = Some(usage);
    }
    
    /// 获取当前状态
    pub fn get_status(&self) -> AlgorithmStatus {
        self.current_status
    }
    
    /// 获取当前进度
    pub fn get_progress(&self) -> f32 {
        self.progress
    }
    
    /// 获取运行时间（秒）
    pub fn get_runtime(&self) -> Option<f64> {
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => {
                end.duration_since(start).ok().map(|d| d.as_secs_f64())
            }
            (Some(start), None) => {
                SystemTime::now().duration_since(start).ok().map(|d| d.as_secs_f64())
            }
            _ => None,
        }
    }
    
    /// 获取所有日志
    pub fn get_logs(&self) -> &[StatusLog] {
        &self.logs
    }
    
    /// 获取过滤后的日志
    pub fn get_filtered_logs(&self, level: Option<LogLevel>, component: Option<&str>) -> Vec<&StatusLog> {
        self.logs.iter().filter(|log| {
            let level_match = level.map_or(true, |l| log.level == l);
            let component_match = component.map_or(true, |c| log.component == c);
            level_match && component_match
        }).collect()
    }
    
    /// 获取错误记录
    pub fn get_errors(&self) -> &[String] {
        &self.errors
    }
    
    /// 获取状态历史
    pub fn get_status_history(&self) -> &[(AlgorithmStatus, SystemTime)] {
        &self.status_history
    }
    
    /// 获取性能指标
    pub fn get_metrics(&self) -> &HashMap<String, f32> {
        &self.metrics
    }
    
    /// 获取资源使用情况
    pub fn get_resource_usage(&self) -> Option<&AlgorithmResourceUsage> {
        self.resource_usage.as_ref()
    }
    
    /// 重置跟踪器
    pub fn reset(&mut self) {
        *self = Self::new();
    }
    
    /// 生成状态报告
    pub fn generate_report(&self) -> AlgorithmStatusReport {
        AlgorithmStatusReport {
            current_status: self.current_status,
            progress: self.progress,
            start_time: self.start_time,
            end_time: self.end_time,
            runtime: self.get_runtime(),
            error_count: self.errors.len(),
            latest_error: self.errors.last().cloned(),
            metrics: self.metrics.clone(),
            resource_summary: self.resource_usage.clone(),
        }
    }
}

impl Default for AlgorithmStatusTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// 算法状态报告
#[derive(Debug, Clone)]
pub struct AlgorithmStatusReport {
    /// 当前状态
    pub current_status: AlgorithmStatus,
    /// 进度
    pub progress: f32,
    /// 开始时间
    pub start_time: Option<SystemTime>,
    /// 结束时间
    pub end_time: Option<SystemTime>,
    /// 运行时间
    pub runtime: Option<f64>,
    /// 错误数量
    pub error_count: usize,
    /// 最新错误
    pub latest_error: Option<String>,
    /// 性能指标
    pub metrics: HashMap<String, f32>,
    /// 资源使用概览
    pub resource_summary: Option<AlgorithmResourceUsage>,
} 