use std::time::{Instant, Duration, SystemTime, UNIX_EPOCH};
use log::{info, debug};
use std::collections::HashMap;

use crate::data::pipeline::pipeline::{PipelineStage, PipelineContext};
use crate::Error;

/// 管道性能指标结构
#[derive(Debug, Clone)]
pub struct PipelinePerformanceMetrics {
    /// 阶段执行时间
    pub stage_durations: HashMap<String, Duration>,
    /// 总执行时间
    pub total_duration: Duration,
    /// 开始时间
    pub start_time: Instant,
    /// 结束时间
    pub end_time: Option<Instant>,
    /// 处理的记录数
    pub processed_records: usize,
    /// 每秒处理记录数
    pub records_per_second: f64,
}

impl PipelinePerformanceMetrics {
    /// 创建新的性能指标
    pub fn new() -> Self {
        PipelinePerformanceMetrics {
            stage_durations: HashMap::new(),
            total_duration: Duration::new(0, 0),
            start_time: Instant::now(),
            end_time: None,
            processed_records: 0,
            records_per_second: 0.0,
        }
    }
    
    /// 添加阶段执行时间
    pub fn add_stage_duration(&mut self, stage_name: String, duration: Duration) {
        self.stage_durations.insert(stage_name, duration);
    }
    
    /// 记录处理完成
    pub fn record_completion(&mut self, processed_records: usize) {
        self.end_time = Some(Instant::now());
        self.total_duration = self.end_time.unwrap().duration_since(self.start_time);
        self.processed_records = processed_records;
        
        // 计算每秒处理记录数
        let seconds = self.total_duration.as_secs_f64();
        if seconds > 0.0 && processed_records > 0 {
            self.records_per_second = processed_records as f64 / seconds;
        }
    }
    
    /// 获取性能报告
    pub fn get_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str(&format!("性能指标报告:\n"));
        report.push_str(&format!("总执行时间: {:.2} 秒\n", self.total_duration.as_secs_f64()));
        report.push_str(&format!("处理记录数: {}\n", self.processed_records));
        report.push_str(&format!("每秒处理记录: {:.2}\n", self.records_per_second));
        report.push_str("阶段执行时间:\n");
        
        // 按执行时间排序阶段
        let mut stages: Vec<(&String, &Duration)> = self.stage_durations.iter().collect();
        stages.sort_by(|a, b| b.1.cmp(a.1)); // 从高到低排序
        
        for (stage, duration) in stages {
            let percentage = if self.total_duration.as_nanos() > 0 {
                (duration.as_nanos() as f64 / self.total_duration.as_nanos() as f64) * 100.0
            } else {
                0.0
            };
            
            report.push_str(&format!("  {}: {:.2} 秒 ({:.1}%)\n", 
                stage, duration.as_secs_f64(), percentage));
        }
        
        report
    }
}

/// 性能监控阶段
#[derive(Clone, Default)]
pub struct PerformanceMonitorStage {
    /// 阶段名称
    name: String,
    /// 启动时间
    start_time: Option<Instant>,
    /// 结束时间
    end_time: Option<Instant>,
    /// 性能指标
    metrics: HashMap<String, String>,
}

impl PerformanceMonitorStage {
    /// 创建新的性能监控阶段
    pub fn new(name: &str) -> Self {
        PerformanceMonitorStage {
            name: name.to_string(),
            start_time: None,
            end_time: None,
            metrics: HashMap::new(),
        }
    }
    
    /// 设置指标
    pub fn with_metric(mut self, key: &str, value: &str) -> Self {
        self.metrics.insert(key.to_string(), value.to_string());
        self
    }
    
    /// 开始监控
    fn start_monitoring(&mut self) {
        self.start_time = Some(Instant::now());
        debug!("开始监控阶段: {}", self.name);
    }
    
    /// 结束监控
    fn end_monitoring(&mut self) -> Duration {
        let end = Instant::now();
        self.end_time = Some(end);
        
        let duration = match self.start_time {
            Some(start) => end.duration_since(start),
            None => Duration::from_secs(0),
        };
        
        debug!("阶段 {} 执行时间: {:?}", self.name, duration);
        
        duration
    }
    
    /// 计算性能指标
    fn calculate_metrics(&self, duration: Duration) -> HashMap<String, String> {
        let mut metrics = self.metrics.clone();
        
        // 添加耗时指标
        metrics.insert("duration_ms".to_string(), duration.as_millis().to_string());
        metrics.insert("duration_sec".to_string(), duration.as_secs_f64().to_string());
        
        metrics
    }
}

impl PipelineStage for PerformanceMonitorStage {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> Option<&str> {
        Some("监控管道阶段性能")
    }
    
    fn process(&self, ctx: &mut PipelineContext) -> Result<(), Error> {
        info!("执行性能监控阶段: {}", self.name);
        
        // 克隆当前实例以便可以修改
        let mut stage = self.clone();
        
        // 开始监控
        stage.start_monitoring();
        
        // 获取管道名称
        let pipeline_name = ctx.get_string("pipeline_name")
            .unwrap_or_else(|_| "未知管道".to_string());
        
        // 记录管道当前阶段
        let _ = ctx.add_data("current_stage", stage.name.clone());
        // 将 std::time::Instant 转换为 SystemTime 的秒数（可序列化）
        let now = std::time::Instant::now();
        let start_time_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let _ = ctx.add_data("stage_start_time", start_time_secs);
        
        debug!("管道 '{}' 执行阶段 '{}'", pipeline_name, stage.name);
        
        // 记录开始监控时间
        if let Some(_start_time) = stage.start_time {
            // 将 std::time::Instant 转换为可序列化的时间戳
            // 由于 Instant 是相对于系统启动的时间，我们使用当前 SystemTime 作为近似值
            let monitor_start_secs = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let _ = ctx.add_data("monitor_start_time", monitor_start_secs);
        }
        
        // 结束监控
        let duration = stage.end_monitoring();
        
        // 计算性能指标
        let metrics = stage.calculate_metrics(duration);
        
        // 更新上下文
        // 将 Duration 转换为秒数（可序列化）
        let duration_secs = duration.as_secs_f64();
        let _ = ctx.add_data("stage_duration", duration_secs);
        let _ = ctx.add_data("stage_metrics", metrics.clone());
        
        // 累积性能指标
        let mut all_metrics: HashMap<String, HashMap<String, String>> = if let Ok(existing_metrics) = ctx.get_data::<HashMap<String, HashMap<String, String>>>("performance_metrics") {
            existing_metrics
        } else {
            HashMap::new()
        };
        all_metrics.insert(stage.name.clone(), metrics);
        let _ = ctx.add_data("performance_metrics", all_metrics);
        
        info!("阶段 '{}' 执行完成，耗时: {:?}", stage.name, duration);
        
        Ok(())
    }
    
    fn can_process(&self, _context: &PipelineContext) -> bool {
        // 性能监控阶段可以处理任意上下文
        true
    }
    
    fn metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("name".to_string(), self.name.clone());
        
        if let (Some(start), Some(end)) = (self.start_time, self.end_time) {
            let duration = end.duration_since(start);
            metadata.insert("duration_ms".to_string(), duration.as_millis().to_string());
        }
        
        metadata
    }
} 