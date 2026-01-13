use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::data::pipeline::pipeline::{
    Pipeline, PipelineStage, PipelineContext,
    BasicPipeline
};
use crate::data::pipeline::monitor::{
    PipelineMonitor, PipelineEventType
};
use crate::Error;

/// 测试用的简单处理阶段
struct TestStage {
    name: String,
    description: Option<String>,
    should_fail: bool,
}

impl TestStage {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: None,
            should_fail: false,
        }
    }

    fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    fn with_failure(mut self) -> Self {
        self.should_fail = true;
        self
    }
}

impl PipelineStage for TestStage {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    fn process(&self, context: &mut PipelineContext) -> Result<(), Error> {
        // 记录此阶段已运行
        if let Ok(mut processed_stages) = context.get_temp::<Vec<String>>("processed_stages") {
            processed_stages.push(self.name.clone());
            context.add_temp("processed_stages", processed_stages)?;
        } else {
            // 第一次运行，初始化记录
            context.add_temp("processed_stages", vec![self.name.clone()])?;
        }
        
        if self.should_fail {
            return Err(Error::custom(&format!("阶段 {} 故意失败", self.name)));
        }
        
        Ok(())
    }

    fn can_process(&self, context: &PipelineContext) -> bool {
        // 检查是否应该跳过
        if let Ok(skip_stages) = context.get_temp::<Vec<String>>("skip_stages") {
            if skip_stages.contains(&self.name) {
                return false;
            }
        }
        
        true
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("stage_type".to_string(), "test".to_string());
        metadata
    }
}

#[test]
fn test_pipeline_basic() {
    let mut pipeline = BasicPipeline::new("test_pipeline")
        .with_description("测试管道");
    
    let stage1 = Arc::new(TestStage::new("stage1").with_description("第一阶段"));
    let stage2 = Arc::new(TestStage::new("stage2").with_description("第二阶段"));
    let stage3 = Arc::new(TestStage::new("stage3").with_description("第三阶段"));
    
    // 添加阶段
    pipeline.add_stage(stage1).unwrap();
    pipeline.add_stage(stage2).unwrap();
    pipeline.add_stage(stage3).unwrap();
    
    // 验证阶段数量
    assert_eq!(pipeline.stages().len(), 3);
    
    // 执行管道
    let mut context = PipelineContext::new();
    context.add_temp("test_data", "测试数据").unwrap();
    
    let result = pipeline.execute(&mut context);
    assert!(result.is_ok(), "管道执行应该成功");
    
    // 验证所有阶段都已执行
    let processed_stages = context.get_temp::<Vec<String>>("processed_stages").unwrap();
    assert_eq!(processed_stages.len(), 3);
    assert_eq!(processed_stages[0], "stage1");
    assert_eq!(processed_stages[1], "stage2");
    assert_eq!(processed_stages[2], "stage3");
}

#[test]
fn test_pipeline_with_failing_stage() {
    let mut pipeline = BasicPipeline::new("failing_pipeline");
    
    let stage1 = Arc::new(TestStage::new("stage1"));
    let stage2 = Arc::new(TestStage::new("stage2").with_failure());
    let stage3 = Arc::new(TestStage::new("stage3"));
    
    pipeline.add_stage(stage1).unwrap();
    pipeline.add_stage(stage2).unwrap();
    pipeline.add_stage(stage3).unwrap();
    
    let mut context = PipelineContext::new();
    let result = pipeline.execute(&mut context);
    
    // 验证管道执行失败
    assert!(result.is_err(), "管道执行应该失败");
    
    // 验证只有第一个阶段执行了
    let processed_stages = context.get_temp::<Vec<String>>("processed_stages").unwrap();
    assert_eq!(processed_stages.len(), 1);
    assert_eq!(processed_stages[0], "stage1");
}

#[test]
fn test_pipeline_stage_skipping() {
    let mut pipeline = BasicPipeline::new("skipping_pipeline");
    
    let stage1 = Arc::new(TestStage::new("stage1"));
    let stage2 = Arc::new(TestStage::new("stage2"));
    let stage3 = Arc::new(TestStage::new("stage3"));
    
    pipeline.add_stage(stage1).unwrap();
    pipeline.add_stage(stage2).unwrap();
    pipeline.add_stage(stage3).unwrap();
    
    let mut context = PipelineContext::new();
    
    // 设置跳过stage2
    context.add_temp("skip_stages", vec!["stage2".to_string()]).unwrap();
    
    let result = pipeline.execute(&mut context);
    assert!(result.is_ok(), "管道执行应该成功");
    
    // 验证stage2被跳过
    let processed_stages = context.get_temp::<Vec<String>>("processed_stages").unwrap();
    assert_eq!(processed_stages.len(), 2);
    assert_eq!(processed_stages[0], "stage1");
    assert_eq!(processed_stages[1], "stage3");
}

#[test]
fn test_pipeline_monitor() {
    let stage1 = Arc::new(TestStage::new("stage1"));
    let stage2 = Arc::new(TestStage::new("stage2"));
    
    let mut pipeline = BasicPipeline::new("monitored_pipeline");
    pipeline.add_stage(stage1).unwrap();
    pipeline.add_stage(stage2).unwrap();
    
    let mut monitor = PipelineMonitor::new();
    let mut context = PipelineContext::new();
    
    // 记录管道开始
    monitor.record_pipeline_start("monitored_pipeline");
    
    // 记录第一个阶段开始
    let stage1_start = Instant::now();
    monitor.record_stage_start("monitored_pipeline", "stage1");
    
    // 执行第一个阶段
    if let Some(stage) = pipeline.stages().get(0) {
        stage.process(&mut context).unwrap();
    }
    
    // 记录第一个阶段完成
    let stage1_duration = stage1_start.elapsed().as_millis() as u64;
    monitor.record_stage_complete("monitored_pipeline", "stage1", stage1_duration);
    
    // 记录第二个阶段开始
    let stage2_start = Instant::now();
    monitor.record_stage_start("monitored_pipeline", "stage2");
    
    // 执行第二个阶段
    if let Some(stage) = pipeline.stages().get(1) {
        stage.process(&mut context).unwrap();
    }
    
    // 记录第二个阶段完成
    let stage2_duration = stage2_start.elapsed().as_millis() as u64;
    monitor.record_stage_complete("monitored_pipeline", "stage2", stage2_duration);
    
    // 记录管道完成
    monitor.record_pipeline_complete("monitored_pipeline");
    
    // 验证监控数据
    let events = monitor.get_events();
    assert!(events.len() >= 6); // 至少应该有6个事件（管道开始，2个阶段开始，2个阶段完成，管道完成）
    
    // 验证事件类型
    let event_types: Vec<_> = events.iter().map(|e| &e.event_type).collect();
    assert!(event_types.contains(&&PipelineEventType::PipelineStarted));
    assert!(event_types.contains(&&PipelineEventType::PipelineCompleted));
    assert!(event_types.contains(&&PipelineEventType::StageStarted));
    assert!(event_types.contains(&&PipelineEventType::StageCompleted));
    
    // 验证处理的阶段
    let processed_stages = context.get_temp::<Vec<String>>("processed_stages").unwrap();
    assert_eq!(processed_stages.len(), 2);
    assert_eq!(processed_stages[0], "stage1");
    assert_eq!(processed_stages[1], "stage2");
    
    // 验证指标数据
    let metrics = monitor.get_metrics("monitored_pipeline").unwrap();
    assert_eq!(metrics.pipeline_id, "monitored_pipeline");
    assert!(metrics.successful);
    assert_eq!(metrics.total_stages(), 2);
    assert_eq!(metrics.succeeded_stages(), 2);
    assert_eq!(metrics.failed_stages(), 0);
}

#[test]
fn test_pipeline_context() {
    let mut context = PipelineContext::new();
    
    // 测试添加和获取参数
    context.add_param("int_param", 123).unwrap();
    context.add_param("string_param", "test_string").unwrap();
    
    let int_val: i32 = context.get_param("int_param").unwrap();
    let string_val: String = context.get_param("string_param").unwrap();
    
    assert_eq!(int_val, 123);
    assert_eq!(string_val, "test_string");
    
    // 测试添加和获取数据
    context.add_data("data1", vec![1, 2, 3]).unwrap();
    let data: Vec<i32> = context.get_data("data1").unwrap();
    assert_eq!(data, vec![1, 2, 3]);
    
    // 测试添加和获取临时数据
    context.add_temp("temp1", "临时数据").unwrap();
    let temp: String = context.get_temp("temp1").unwrap();
    assert_eq!(temp, "临时数据");
    
    // 测试设置和获取状态
    context.set_state("status", "运行中");
    assert_eq!(context.get_state("status").unwrap(), "运行中");
    
    // 测试不存在的数据
    let result: Result<String, _> = context.get_param("non_existent");
    assert!(result.is_err());
} 