// src/data/pipeline/monitor/pipeline_monitor.rs
//
// 管道监控模块实现
// 提供对管道执行过程的监控和事件跟踪功能

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime};
use log::{info, error};
use uuid;

use crate::data::pipeline::traits::{PipelineStage, PipelineContext};
use crate::data::DataBatch;
// use crate::error::Error; // reserved for future error enrichment
use crate::Result;

/// 管道执行事件类型
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineEventType {
    /// 管道启动
    PipelineStarted,
    /// 管道完成
    PipelineCompleted,
    /// 管道失败
    PipelineFailed,
    /// 阶段开始
    StageStarted,
    /// 阶段完成
    StageCompleted,
    /// 阶段失败
    StageFailed,
    /// 自定义事件
    Custom(String),
}

/// 管道执行事件
#[derive(Debug, Clone)]
pub struct PipelineEvent {
    /// 事件类型
    pub event_type: PipelineEventType,
    /// 事件时间戳（从管道开始的毫秒数）
    pub timestamp_ms: u64,
    /// 事件名称
    pub name: String,
    /// 关联的阶段名称（如果有）
    pub stage_name: Option<String>,
    /// 事件详情
    pub details: Option<String>,
    /// 事件元数据
    pub metadata: HashMap<String, String>,
}

/// 执行指标
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    /// 管道ID
    pub pipeline_id: String,
    /// 总执行时间（毫秒）
    pub total_execution_time_ms: u64,
    /// 阶段执行时间
    pub stage_execution_times: HashMap<String, u64>,
    /// 阶段执行状态
    pub stage_statuses: HashMap<String, bool>,
    /// 处理的记录数
    pub processed_records: u64,
    /// 失败的记录数
    pub failed_records: u64,
    /// 开始时间
    pub start_time: SystemTime,
    /// 结束时间
    pub end_time: Option<SystemTime>,
    /// 是否成功完成
    pub successful: bool,
    /// 错误信息（如果失败）
    pub error: Option<String>,
}

impl Default for ExecutionMetrics {
    fn default() -> Self {
        Self {
            pipeline_id: String::new(),
            total_execution_time_ms: 0,
            stage_execution_times: HashMap::new(),
            stage_statuses: HashMap::new(),
            processed_records: 0,
            failed_records: 0,
            start_time: SystemTime::now(),
            end_time: None,
            successful: false,
            error: None,
        }
    }
}

impl ExecutionMetrics {
    /// 创建新的执行指标
    pub fn new(pipeline_id: &str) -> Self {
        Self {
            pipeline_id: pipeline_id.to_string(),
            start_time: SystemTime::now(),
            ..Default::default()
        }
    }
    
    /// 获取总阶段数
    pub fn total_stages(&self) -> usize {
        self.stage_statuses.len()
    }
    
    /// 获取成功阶段数
    pub fn succeeded_stages(&self) -> usize {
        self.stage_statuses.values().filter(|&&v| v).count()
    }
    
    /// 获取失败阶段数
    pub fn failed_stages(&self) -> usize {
        self.stage_statuses.values().filter(|&&v| !v).count()
    }
    
    /// 获取阶段执行时间
    pub fn stage_execution_time(&self, stage_name: &str) -> Option<u64> {
        self.stage_execution_times.get(stage_name).copied()
    }
}

/// 管道监控器
///
/// 用于收集和报告管道执行过程中的指标和事件
#[derive(Debug, Default)]
pub struct PipelineMonitor {
    /// 记录的事件
    events: Vec<PipelineEvent>,
    /// 管道执行指标
    metrics: HashMap<String, ExecutionMetrics>,
    /// 开始时间
    start_time: Option<Instant>,
}

impl PipelineMonitor {
    /// 创建新的管道监控器
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            metrics: HashMap::new(),
            start_time: Some(Instant::now()),
        }
    }
    
    /// 获取当前时间戳（相对于开始时间）
    fn current_timestamp_ms(&self) -> u64 {
        if let Some(start_time) = self.start_time {
            let elapsed = start_time.elapsed();
            (elapsed.as_secs() * 1000 + elapsed.subsec_millis() as u64) as u64
        } else {
            0
        }
    }
    
    /// 记录事件
    fn record_event(&mut self, event_type: PipelineEventType, name: &str, stage_name: Option<&str>, details: Option<&str>) {
        let timestamp = self.current_timestamp_ms();
        let event = PipelineEvent {
            event_type,
            timestamp_ms: timestamp,
            name: name.to_string(),
            stage_name: stage_name.map(String::from),
            details: details.map(String::from),
            metadata: HashMap::new(),
        };
        self.events.push(event);
    }
    
    /// 记录管道开始
    pub fn record_pipeline_start(&mut self, pipeline_id: &str) {
        // 记录事件
        self.record_event(PipelineEventType::PipelineStarted, pipeline_id, None, None);
        
        // 创建或重置指标
        let metrics = ExecutionMetrics::new(pipeline_id);
        self.metrics.insert(pipeline_id.to_string(), metrics);
        
        // 重置开始时间
        self.start_time = Some(Instant::now());
    }
    
    /// 记录管道完成
    pub fn record_pipeline_complete(&mut self, pipeline_id: &str) {
        // 记录事件
        self.record_event(PipelineEventType::PipelineCompleted, pipeline_id, None, None);
        
        // 更新指标
        if let Some(metrics) = self.metrics.get_mut(pipeline_id) {
            metrics.end_time = Some(SystemTime::now());
            metrics.total_execution_time_ms = self.current_timestamp_ms();
            metrics.successful = true;
        }
    }
    
    /// 记录管道失败
    pub fn record_pipeline_failed(&mut self, pipeline_id: &str, error: &str) {
        // 记录事件
        self.record_event(PipelineEventType::PipelineFailed, pipeline_id, None, Some(error));
        
        // 更新指标
        if let Some(metrics) = self.metrics.get_mut(pipeline_id) {
            metrics.end_time = Some(SystemTime::now());
            metrics.total_execution_time_ms = self.current_timestamp_ms();
            metrics.successful = false;
            metrics.error = Some(error.to_string());
        }
    }
    
    /// 记录阶段开始
    pub fn record_stage_start(&mut self, pipeline_id: &str, stage_name: &str) {
        self.record_event(PipelineEventType::StageStarted, pipeline_id, Some(stage_name), None);
    }
    
    /// 记录阶段完成
    pub fn record_stage_complete(&mut self, pipeline_id: &str, stage_name: &str, duration_ms: u64) {
        // 记录事件
        self.record_event(PipelineEventType::StageCompleted, pipeline_id, Some(stage_name), None);
        
        // 更新指标
        if let Some(metrics) = self.metrics.get_mut(pipeline_id) {
            metrics.stage_execution_times.insert(stage_name.to_string(), duration_ms);
            metrics.stage_statuses.insert(stage_name.to_string(), true);
        }
    }
    
    /// 记录阶段失败
    pub fn record_stage_failed(&mut self, pipeline_id: &str, stage_name: &str, error: &str, duration_ms: u64) {
        // 记录事件
        self.record_event(PipelineEventType::StageFailed, pipeline_id, Some(stage_name), Some(error));
        
        // 更新指标
        if let Some(metrics) = self.metrics.get_mut(pipeline_id) {
            metrics.stage_execution_times.insert(stage_name.to_string(), duration_ms);
            metrics.stage_statuses.insert(stage_name.to_string(), false);
            metrics.failed_records += 1;
        }
    }
    
    /// 获取所有事件
    pub fn get_events(&self) -> Vec<PipelineEvent> {
        self.events.clone()
    }
    
    /// 获取指定管道的指标
    pub fn get_metrics(&self, pipeline_id: &str) -> Option<ExecutionMetrics> {
        self.metrics.get(pipeline_id).cloned()
    }
    
    /// 获取所有管道的指标
    pub fn get_all_metrics(&self) -> HashMap<String, ExecutionMetrics> {
        self.metrics.clone()
    }
}

/// 带监控的异步管道
pub struct MonitoredPipeline<T> {
    /// 内部管道
    pipeline: T,
    /// 监控器
    monitor: Arc<Mutex<PipelineMonitor>>,
    /// 管道状态
    status: PipelineStageStatus,
}

// 为了满足PipelineStage trait，添加这个状态枚举
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PipelineStageStatus {
    NotInitialized,
    Initialized,
    Processing,
    Completed,
    Failed,
    Cleaned,
}

impl<T> MonitoredPipeline<T> {
    /// 创建新的带监控的管道
    pub fn new(pipeline: T, monitor: Arc<Mutex<PipelineMonitor>>) -> Self {
        Self {
            pipeline,
            monitor,
            status: PipelineStageStatus::NotInitialized,
        }
    }
    
    /// 构造带富集信息的错误消息，便于下游告警/排障
    fn enrich_error_message(
        pipeline_id: &str,
        stage_name: &str,
        error_message: &str,
        metadata: &HashMap<String, String>,
    ) -> String {
        let mut meta_pairs: Vec<String> = Vec::new();
        for (k, v) in metadata.iter() {
            meta_pairs.push(format!("\"{}\":\"{}\"", k, v));
        }
        let meta_str = meta_pairs.join(",");
        format!(
            "{{\"pipeline_id\":\"{}\",\"stage\":\"{}\",\"error\":\"{}\",\"metadata\":{{{}}}}}",
            pipeline_id,
            stage_name,
            error_message.replace('"', "'"),
            meta_str
        )
    }
    
    /// 获取内部管道
    pub fn pipeline(&self) -> &T {
        &self.pipeline
    }
    
    /// 获取监控器
    pub fn monitor(&self) -> Arc<Mutex<PipelineMonitor>> {
        self.monitor.clone()
    }
}

/// 为异步处理管道实现特殊接口
impl<T: AsyncPipelineStage> MonitoredPipeline<T> {
    /// 异步处理数据批次
    pub async fn process_batch(&self, input: DataBatch) -> Result<DataBatch> {
        // 获取管道ID和阶段名称
        let pipeline_id = format!("pipeline-{}", uuid::Uuid::new_v4());
        let stage_name = self.pipeline.name().to_string();
        
        // 记录开始
        {
            let mut monitor = self.monitor.lock().unwrap();
            monitor.record_pipeline_start(&pipeline_id);
            monitor.record_stage_start(&pipeline_id, &stage_name);
        }
        
        let start = Instant::now();
        
        // 执行处理
        let result = match self.pipeline.process_async(input).await {
            Ok(output) => {
                // 记录成功
                let duration = start.elapsed();
                let duration_ms = duration.as_secs() * 1000 + duration.subsec_millis() as u64;
                
                let mut monitor = self.monitor.lock().unwrap();
                monitor.record_stage_complete(&pipeline_id, &stage_name, duration_ms);
                monitor.record_pipeline_complete(&pipeline_id);
                
                Ok(output)
            },
            Err(e) => {
                // 记录失败
                let duration = start.elapsed();
                let duration_ms = duration.as_secs() * 1000 + duration.subsec_millis() as u64;
                
                let mut monitor = self.monitor.lock().unwrap();
                let enriched = Self::enrich_error_message(
                    &pipeline_id,
                    &stage_name,
                    &e.to_string(),
                    &self.pipeline.metadata(),
                );
                monitor.record_stage_failed(&pipeline_id, &stage_name, &enriched, duration_ms);
                monitor.record_pipeline_failed(&pipeline_id, &enriched);
                
                Err(e)
            }
        };
        
        result
    }
}

/// 异步管道阶段特征
pub trait AsyncPipelineStage: Send + Sync {
    /// 处理数据批次
    async fn process_async(&self, input: DataBatch) -> Result<DataBatch>;
    
    /// 获取阶段名称
    fn name(&self) -> &str;
    
    /// 获取阶段元数据
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

/// 为标准PipelineStage实现PipelineStage特征
impl<T: PipelineStage> PipelineStage for MonitoredPipeline<T> {
    /// 获取阶段名称
    fn name(&self) -> &str {
        self.pipeline.name()
    }
    
    /// 初始化阶段
    fn init(&mut self, context: &mut PipelineContext) -> Result<(), std::boxed::Box<dyn std::error::Error>> {
        self.status = PipelineStageStatus::Initialized;
        self.pipeline.init(context)
    }
    
    /// 处理数据
    fn process(&mut self, context: &mut PipelineContext) -> Result<(), std::boxed::Box<dyn std::error::Error>> {
        // 获取管道ID和阶段名称
        // PipelineContext 没有 pipeline_id 字段，从 metadata 或生成新的
        let pipeline_id = context.metadata.get("pipeline_id")
            .cloned()
            .unwrap_or_else(|| format!("pipeline-{}", uuid::Uuid::new_v4()));
        let stage_name = self.name().to_string();
        
        // 记录开始
        {
            let mut monitor = self.monitor.lock().unwrap();
            monitor.record_pipeline_start(&pipeline_id);
            monitor.record_stage_start(&pipeline_id, &stage_name);
        }
        
        self.status = PipelineStageStatus::Processing;
        let start = Instant::now();
        
        // 执行处理
        let result = self.pipeline.process(context);
        
        // 计算时间
        let duration = start.elapsed();
        let duration_ms = duration.as_secs() * 1000 + duration.subsec_millis() as u64;
        
        // 记录结果
        match &result {
            Ok(_) => {
                self.status = PipelineStageStatus::Completed;
                let mut monitor = self.monitor.lock().unwrap();
                monitor.record_stage_complete(&pipeline_id, &stage_name, duration_ms);
                monitor.record_pipeline_complete(&pipeline_id);
            }
            Err(e) => {
                self.status = PipelineStageStatus::Failed;
                let mut monitor = self.monitor.lock().unwrap();
                let mut meta_pairs: Vec<String> = Vec::new();
                for (k, v) in self.pipeline.metadata().iter() {
                    meta_pairs.push(format!("\"{}\":\"{}\"", k, v));
                }
                let meta_str = meta_pairs.join(",");
                let enriched = format!(
                    "{{\"pipeline_id\":\"{}\",\"stage\":\"{}\",\"error\":\"{}\",\"metadata\":{{{}}}}}",
                    pipeline_id,
                    stage_name,
                    e.to_string().replace('"', "'"),
                    meta_str
                );
                monitor.record_stage_failed(&pipeline_id, &stage_name, &enriched, duration_ms);
                monitor.record_pipeline_failed(&pipeline_id, &enriched);
            }
        }
        
        result
    }
    
    /// 清理阶段
    fn cleanup(&mut self, context: &mut PipelineContext) -> Result<(), std::boxed::Box<dyn std::error::Error>> {
        let result = self.pipeline.cleanup(context);
        if result.is_ok() {
            self.status = PipelineStageStatus::Cleaned;
        }
        result
    }
    
    /// 获取阶段状态
    fn status(&self) -> crate::data::pipeline::traits::PipelineStageStatus {
        match self.status {
            PipelineStageStatus::NotInitialized => crate::data::pipeline::traits::PipelineStageStatus::NotInitialized,
            PipelineStageStatus::Initialized => crate::data::pipeline::traits::PipelineStageStatus::Initialized,
            PipelineStageStatus::Processing => crate::data::pipeline::traits::PipelineStageStatus::Processing,
            PipelineStageStatus::Completed => crate::data::pipeline::traits::PipelineStageStatus::Completed,
            PipelineStageStatus::Failed => crate::data::pipeline::traits::PipelineStageStatus::Failed,
            PipelineStageStatus::Cleaned => crate::data::pipeline::traits::PipelineStageStatus::Cleaned,
        }
    }
    
    /// 设置阶段状态
    fn set_status(&mut self, status: crate::data::pipeline::traits::PipelineStageStatus) {
        self.status = match status {
            crate::data::pipeline::traits::PipelineStageStatus::NotInitialized => PipelineStageStatus::NotInitialized,
            crate::data::pipeline::traits::PipelineStageStatus::Initialized => PipelineStageStatus::Initialized,
            crate::data::pipeline::traits::PipelineStageStatus::Processing => PipelineStageStatus::Processing,
            crate::data::pipeline::traits::PipelineStageStatus::Completed => PipelineStageStatus::Completed,
            crate::data::pipeline::traits::PipelineStageStatus::Failed => PipelineStageStatus::Failed,
            crate::data::pipeline::traits::PipelineStageStatus::Cleaned => PipelineStageStatus::Cleaned,
        };
    }
    
    /// 获取阶段元数据
    fn metadata(&self) -> HashMap<String, String> {
        self.pipeline.metadata()
    }
    
    /// 获取依赖的阶段
    fn dependencies(&self) -> Vec<&str> {
        self.pipeline.dependencies()
    }
    
    /// 转换为Any类型
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    /// 转换为可变Any类型
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// 带监控的阶段
pub struct MonitoredStage<T: PipelineStage> {
    /// 被包装的阶段
    inner: T,
    /// 监控器
    monitor: PipelineMonitor,
    /// 阶段状态
    status: crate::data::pipeline::traits::PipelineStageStatus,
}

impl<T: PipelineStage> MonitoredStage<T> {
    /// 创建新的被监控的阶段
    pub fn new(stage: T, monitor: PipelineMonitor) -> Self {
        Self {
            inner: stage,
            monitor,
            status: crate::data::pipeline::traits::PipelineStageStatus::NotInitialized,
        }
    }
    
    /// 创建新的被监控的阶段，使用默认监控器
    pub fn with_default_monitor(stage: T) -> Self {
        Self {
            inner: stage,
            monitor: PipelineMonitor::new(),
            status: crate::data::pipeline::traits::PipelineStageStatus::NotInitialized,
        }
    }
    
    /// 获取内部阶段
    pub fn inner(&self) -> &T {
        &self.inner
    }
    
    /// 获取监控器
    pub fn monitor(&self) -> &PipelineMonitor {
        &self.monitor
    }
}

impl<T: PipelineStage> PipelineStage for MonitoredStage<T> {
    fn name(&self) -> &str {
        self.inner.name()
    }
    
    fn init(&mut self, context: &mut PipelineContext) -> Result<(), std::boxed::Box<dyn std::error::Error>> {
        self.status = crate::data::pipeline::traits::PipelineStageStatus::Initialized;
        self.inner.init(context)
    }
    
    fn process(&mut self, context: &mut PipelineContext) -> Result<(), std::boxed::Box<dyn std::error::Error>> {
        // 记录开始事件
        // PipelineContext 没有 pipeline_id 字段，从 metadata 或使用默认值
        let pipeline_id = context.metadata.get("pipeline_id")
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());
        let stage_name = self.name().to_string();
        
        self.monitor.record_stage_start(&pipeline_id, &stage_name);
        self.status = crate::data::pipeline::traits::PipelineStageStatus::Processing;
        
        let start = Instant::now();
        
        // 执行处理
        let result = self.inner.process(context);
        
        // 计算时间
        let duration = start.elapsed();
        let duration_ms = duration.as_secs() * 1000 + duration.subsec_millis() as u64;
        
        // 记录结果
        match &result {
            Ok(_) => {
                self.status = crate::data::pipeline::traits::PipelineStageStatus::Completed;
                self.monitor.record_stage_complete(&pipeline_id, &stage_name, duration_ms);
            }
            Err(e) => {
                self.status = crate::data::pipeline::traits::PipelineStageStatus::Failed;
                self.monitor.record_stage_failed(&pipeline_id, &stage_name, &e.to_string(), duration_ms);
            }
        }
        
        result
    }
    
    fn cleanup(&mut self, context: &mut PipelineContext) -> Result<(), std::boxed::Box<dyn std::error::Error>> {
        let result = self.inner.cleanup(context);
        if result.is_ok() {
            self.status = crate::data::pipeline::traits::PipelineStageStatus::Cleaned;
        }
        result
    }
    
    fn status(&self) -> crate::data::pipeline::traits::PipelineStageStatus {
        self.status
    }
    
    fn set_status(&mut self, status: crate::data::pipeline::traits::PipelineStageStatus) {
        self.status = status;
    }
    
    fn metadata(&self) -> HashMap<String, String> {
        self.inner.metadata()
    }
    
    fn dependencies(&self) -> Vec<&str> {
        self.inner.dependencies()
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// 打印执行指标摘要
pub fn print_summary(metrics: &ExecutionMetrics) {
    info!("=== 管道执行摘要 ===");
    info!("管道ID: {}", metrics.pipeline_id);
    info!("总执行时间: {}ms", metrics.total_execution_time_ms);
    info!("成功状态: {}", metrics.successful);
    
    info!("阶段执行情况:");
    for (stage, time) in &metrics.stage_execution_times {
        let status = if *metrics.stage_statuses.get(stage).unwrap_or(&false) {
            "成功"
        } else {
            "失败"
        };
        info!("  {} - {}ms ({})", stage, time, status);
    }
    
    info!("记录处理情况:");
    info!("  处理记录: {}", metrics.processed_records);
    info!("  失败记录: {}", metrics.failed_records);
    
    if let Some(error) = &metrics.error {
        error!("失败原因: {}", error);
    }
} 