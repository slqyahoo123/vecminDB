// src/data/pipeline/monitor/mod.rs
//
// 监控模块入口
// 提供流水线执行过程中的监控功能

use std::collections::HashMap;
use std::time::{Duration, Instant};
use log::debug;

// 导出性能监控模块组件
pub mod performance_monitor;
pub use performance_monitor::AdvancedPerformanceMonitorStage;
pub use performance_monitor::ResourceMetrics;
pub use performance_monitor::MetricType;

// 导出管道监控组件
mod pipeline_monitor;
pub use pipeline_monitor::PipelineMonitor;
pub use pipeline_monitor::PipelineEvent;
pub use pipeline_monitor::PipelineEventType;
pub use pipeline_monitor::ExecutionMetrics;
pub use pipeline_monitor::MonitoredPipeline;
pub use pipeline_monitor::MonitoredStage;

/// 测量函数执行时间的工具函数
pub fn measure_time<F, T>(name: &str, func: F) -> (T, Duration)
where
    F: FnOnce() -> T,
{
    let start = Instant::now();
    let result = func();
    let duration = start.elapsed();
    debug!("{} 执行用时: {:?}", name, duration);
    (result, duration)
}

/// 监控工具结构体
pub struct MonitoringTools {
    /// 计时器映射
    timers: HashMap<String, Instant>,
    /// 计时结果
    durations: HashMap<String, Duration>,
    /// 计数器
    counters: HashMap<String, usize>,
}

impl MonitoringTools {
    /// 创建新的监控工具
    pub fn new() -> Self {
        Self {
            timers: HashMap::new(),
            durations: HashMap::new(),
            counters: HashMap::new(),
        }
    }

    /// 开始计时
    pub fn start_timer(&mut self, name: &str) {
        self.timers.insert(name.to_string(), Instant::now());
        debug!("开始计时: {}", name);
    }

    /// 停止计时
    pub fn stop_timer(&mut self, name: &str) -> Option<Duration> {
        if let Some(start) = self.timers.remove(name) {
            let duration = start.elapsed();
            self.durations.insert(name.to_string(), duration);
            debug!("停止计时: {}, 耗时: {:?}", name, duration);
            Some(duration)
        } else {
            debug!("无法停止计时 {}: 未找到开始时间", name);
            None
        }
    }

    /// 获取所有计时结果
    pub fn get_durations(&self) -> &HashMap<String, Duration> {
        &self.durations
    }

    /// 获取特定计时结果
    pub fn get_duration(&self, name: &str) -> Option<&Duration> {
        self.durations.get(name)
    }

    /// 记录指标
    pub fn record_metric(&mut self, name: &str, value: f64) {
        debug!("记录指标: {} = {}", name, value);
        // 这里可以扩展为存储指标数据
    }

    /// 生成性能报告
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("执行性能报告:\n");

        // 按耗时排序
        let mut durations: Vec<(&String, &Duration)> = self.durations.iter().collect();
        durations.sort_by(|a, b| b.1.cmp(a.1));

        // 计算总耗时
        let total_duration: Duration = self.durations.values().sum();
        
        report.push_str(&format!("总耗时: {:?}\n", total_duration));
        report.push_str("各阶段耗时:\n");

        for (name, duration) in durations {
            let percentage = if total_duration.as_nanos() > 0 {
                (duration.as_nanos() as f64 / total_duration.as_nanos() as f64) * 100.0
            } else {
                0.0
            };
            
            report.push_str(&format!("  {}: {:?} ({:.1}%)\n", name, duration, percentage));
        }

        report
    }
}

impl Default for MonitoringTools {
    fn default() -> Self {
        Self::new()
    }
}

/// 监控工具工厂方法
pub fn create_monitoring_tools() -> MonitoringTools {
    MonitoringTools::new()
}

/// 创建默认的性能监控阶段
pub fn create_performance_monitor(name: &str) -> AdvancedPerformanceMonitorStage {
    AdvancedPerformanceMonitorStage::new(name)
} 