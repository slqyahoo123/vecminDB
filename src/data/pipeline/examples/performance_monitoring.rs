// src/data/pipeline/examples/performance_monitoring.rs
//
// 性能监控使用示例
// 展示如何使用性能监控组件进行流水线执行过程的性能分析

use std::error::Error;
use std::time::{Duration, Instant};
use std::thread;
use log::{info, debug, error};

use crate::data::pipeline::{
    PipelineStageStatus,
    PipelineContext,
    AdvancedPerformanceMonitorStage,
    MonitoringTools,
    measure_time,
    create_monitoring_tools
};

use crate::data::pipeline::traits::{
    PipelineStage,
    PipelineMonitor
};

use crate::data::record::{Record, RecordBatch};

/// 示例阶段 - 处理数据
struct DataProcessingStage {
    name: String,
    status: PipelineStageStatus,
    processing_time_ms: u64,
}

impl DataProcessingStage {
    pub fn new(name: &str, processing_time_ms: u64) -> Self {
        Self {
            name: name.to_string(),
            status: PipelineStageStatus::NotInitialized,
            processing_time_ms,
        }
    }
}

impl PipelineStage for DataProcessingStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn init(&mut self, _context: &mut PipelineContext) -> Result<(), Box<dyn Error>> {
        self.status = PipelineStageStatus::Initialized;
        Ok(())
    }

    fn process(&mut self, context: &mut PipelineContext) -> Result<(), Box<dyn Error>> {
        info!("执行数据处理阶段: {}", self.name);
        self.status = PipelineStageStatus::Processing;

        // 模拟处理时间
        thread::sleep(Duration::from_millis(self.processing_time_ms));

        // 向上下文添加一些模拟数据
        let mut records = Vec::new();
        for i in 0..100 {
            let mut record = Record::new();
            record.add_field("value", i);
            records.push(record);
        }
        
        context.records = records;
        self.status = PipelineStageStatus::Completed;
        Ok(())
    }

    fn cleanup(&mut self, _context: &mut PipelineContext) -> Result<(), Box<dyn Error>> {
        self.status = PipelineStageStatus::Cleaned;
        Ok(())
    }

    fn status(&self) -> PipelineStageStatus {
        self.status
    }

    fn set_status(&mut self, status: PipelineStageStatus) {
        self.status = status;
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// 运行示例
pub fn run_monitoring_example() -> Result<(), Box<dyn Error>> {
    info!("启动性能监控示例");

    // 创建监控工具
    let mut tools = create_monitoring_tools();

    // 创建处理上下文
    let mut context = PipelineContext::new();

    // 创建各阶段
    let mut stage1 = DataProcessingStage::new("数据加载", 500);
    let mut stage2 = DataProcessingStage::new("数据转换", 300);
    let mut stage3 = DataProcessingStage::new("数据处理", 700);
    let mut monitor_stage = AdvancedPerformanceMonitorStage::new("性能监控")
        .with_monitor_interval(100) // 设置监控间隔为100毫秒
        .with_cpu_monitoring(true)
        .with_memory_monitoring(true);

    // 初始化阶段
    tools.start_timer("初始化");
    stage1.init(&mut context)?;
    stage2.init(&mut context)?;
    stage3.init(&mut context)?;
    monitor_stage.init(&mut context)?;
    tools.stop_timer("初始化");

    // 处理数据 - 使用计时工具
    tools.start_timer("阶段1：数据加载");
    stage1.process(&mut context)?;
    tools.stop_timer("阶段1：数据加载");

    tools.start_timer("阶段2：数据转换");
    stage2.process(&mut context)?;
    tools.stop_timer("阶段2：数据转换");

    tools.start_timer("阶段3：数据处理");
    stage3.process(&mut context)?;
    tools.stop_timer("阶段3：数据处理");

    // 处理性能监控
    monitor_stage.process(&mut context)?;

    // 清理阶段
    tools.start_timer("清理");
    stage1.cleanup(&mut context)?;
    stage2.cleanup(&mut context)?;
    stage3.cleanup(&mut context)?;
    monitor_stage.cleanup(&mut context)?;
    tools.stop_timer("清理");

    // 打印监控结果
    info!("性能监控结果：");
    info!("{}", tools.generate_report());

    // 获取资源指标
    if let Some(metrics) = context.shared_data.get("resource_metrics") {
        if let Ok(guard) = metrics.lock() {
            if let Some(resource_metrics) = guard.downcast_ref::<Vec<crate::data::pipeline::monitor::ResourceMetrics>>() {
                info!("收集到 {} 个资源指标样本", resource_metrics.len());
                if !resource_metrics.is_empty() {
                    let last = &resource_metrics[resource_metrics.len() - 1];
                    info!("最新的资源使用情况:");
                    info!("  CPU 使用率: {:.2}%", last.cpu_usage);
                    info!("  内存使用: {:.2} MB", last.memory_usage_mb);
                    info!("  磁盘读取: {:.2} KB/s", last.disk_read_kbps);
                    info!("  磁盘写入: {:.2} KB/s", last.disk_write_kbps);
                }
            }
        }
    }

    // 打印元数据
    info!("管道元数据:");
    for (key, value) in context.metadata.iter() {
        info!("  {}: {}", key, value);
    }

    info!("性能监控示例完成");
    Ok(())
}

/// 使用measure_time函数的示例
pub fn run_measure_time_example() -> Result<(), Box<dyn Error>> {
    info!("启动measure_time示例");

    // 使用measure_time包装函数
    let (result, duration) = measure_time("复杂计算", || {
        // 模拟复杂计算
        let mut sum = 0;
        for i in 0..1000000 {
            sum += i;
        }
        sum
    });

    info!("计算结果: {}, 耗时: {:?}", result, duration);

    // 使用measure_time包装可能失败的函数
    let result = measure_time("读取文件操作", || -> Result<String, Box<dyn Error>> {
        // 模拟读取文件，这里使用固定的字符串替代
        Ok("文件内容".to_string())
    });

    match result {
        (Ok(content), duration) => {
            info!("文件内容: {}, 读取耗时: {:?}", content, duration);
        }
        (Err(e), duration) => {
            error!("读取失败: {}, 耗时: {:?}", e, duration);
        }
    }

    info!("measure_time示例完成");
    Ok(())
}

/// 主示例函数
pub fn run_examples() -> Result<(), Box<dyn Error>> {
    run_monitoring_example()?;
    run_measure_time_example()?;
    Ok(())
} 