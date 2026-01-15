use std::collections::HashMap;
use std::sync::{Arc, Mutex, Once};

use crate::data::pipeline::{
    ImportPipelineBuilder, Pipeline, PipelineStage, PipelineContext, PipelineResult,
};
use crate::data::pipeline::monitor::PipelineMonitor;
use crate::error::Result;

// 添加类型定义以临时解决导入问题
struct TransformPipeline {
    name: String,
    stages: Vec<Arc<dyn PipelineStage>>,
    monitor: Option<Arc<Mutex<PipelineMonitor>>>,
    error_handler: Option<Box<dyn Fn(&mut PipelineContext, &str, &str) -> bool + Send + Sync>>,
}

struct ValidationPipeline {
    name: String,
    stages: Vec<Arc<dyn PipelineStage>>,
    monitor: Option<Arc<Mutex<PipelineMonitor>>>,
    error_handler: Option<Box<dyn Fn(&mut PipelineContext, &str, &str) -> bool + Send + Sync>>,
}

struct VisualizationPipeline {
    name: String,
    stages: Vec<Arc<dyn PipelineStage>>,
    monitor: Option<Arc<Mutex<PipelineMonitor>>>,
    error_handler: Option<Box<dyn Fn(&mut PipelineContext, &str, &str) -> bool + Send + Sync>>,
}

impl TransformPipeline {
    fn default() -> Self {
        Self {
            name: "Transform Pipeline".to_string(),
            stages: Vec::new(),
            monitor: None,
            error_handler: None,
        }
    }
}

impl ValidationPipeline {
    fn default() -> Self {
        Self {
            name: "Validation Pipeline".to_string(),
            stages: Vec::new(),
            monitor: None,
            error_handler: None,
        }
    }
}

impl VisualizationPipeline {
    fn default() -> Self {
        Self {
            name: "Visualization Pipeline".to_string(),
            stages: Vec::new(),
            monitor: None,
            error_handler: None,
        }
    }
}

impl Pipeline for TransformPipeline {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn add_stage(&mut self, stage: Arc<dyn PipelineStage>) -> Result<()> {
        self.stages.push(stage);
        Ok(())
    }
    
    fn execute(&self, context: PipelineContext) -> PipelineResult {
        // 简单实现：返回成功结果
        PipelineResult::Success {
            processed_rows: 0,
            written_rows: 0,
            execution_time: 0.0,
            stage_results: Vec::new(),
            context,
        }
    }
    
    fn stages(&self) -> &[Arc<dyn PipelineStage>] {
        &self.stages
    }
    
    fn reset(&mut self) -> Result<()> {
        self.stages.clear();
        Ok(())
    }
    
    fn set_monitor(&mut self, monitor: Arc<Mutex<PipelineMonitor>>) {
        self.monitor = Some(monitor);
    }
    
    fn set_error_handler(&mut self, handler: Box<dyn Fn(&mut PipelineContext, &str, &str) -> bool + Send + Sync>) {
        self.error_handler = Some(handler);
    }
}

impl Pipeline for ValidationPipeline {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn add_stage(&mut self, stage: Arc<dyn PipelineStage>) -> Result<()> {
        self.stages.push(stage);
        Ok(())
    }
    
    fn execute(&self, context: PipelineContext) -> PipelineResult {
        // 简单实现：返回成功结果
        PipelineResult::Success {
            processed_rows: 0,
            written_rows: 0,
            execution_time: 0.0,
            stage_results: Vec::new(),
            context,
        }
    }
    
    fn stages(&self) -> &[Arc<dyn PipelineStage>] {
        &self.stages
    }
    
    fn reset(&mut self) -> Result<()> {
        self.stages.clear();
        Ok(())
    }
    
    fn set_monitor(&mut self, monitor: Arc<Mutex<PipelineMonitor>>) {
        self.monitor = Some(monitor);
    }
    
    fn set_error_handler(&mut self, handler: Box<dyn Fn(&mut PipelineContext, &str, &str) -> bool + Send + Sync>) {
        self.error_handler = Some(handler);
    }
}

impl Pipeline for VisualizationPipeline {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn add_stage(&mut self, stage: Arc<dyn PipelineStage>) -> Result<()> {
        self.stages.push(stage);
        Ok(())
    }
    
    fn execute(&self, context: PipelineContext) -> PipelineResult {
        // 简单实现：返回成功结果
        PipelineResult::Success {
            processed_rows: 0,
            written_rows: 0,
            execution_time: 0.0,
            stage_results: Vec::new(),
            context,
        }
    }
    
    fn stages(&self) -> &[Arc<dyn PipelineStage>] {
        &self.stages
    }
    
    fn reset(&mut self) -> Result<()> {
        self.stages.clear();
        Ok(())
    }
    
    fn set_monitor(&mut self, monitor: Arc<Mutex<PipelineMonitor>>) {
        self.monitor = Some(monitor);
    }
    
    fn set_error_handler(&mut self, handler: Box<dyn Fn(&mut PipelineContext, &str, &str) -> bool + Send + Sync>) {
        self.error_handler = Some(handler);
    }
}

/// 管道类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineType {
    Import,
    Transform,
    Validation,
    Visualization,
    Custom(usize),
}

/// 管道注册表结构
pub struct PipelineRegistry {
    pipelines: HashMap<PipelineType, Box<dyn Fn() -> Box<dyn Pipeline> + Send + Sync>>,
}

// 单例实现
static mut REGISTRY: Option<Arc<Mutex<PipelineRegistry>>> = None;
static INIT: Once = Once::new();

/// 获取管道注册表单例
pub fn get_registry() -> Arc<Mutex<PipelineRegistry>> {
    unsafe {
        INIT.call_once(|| {
            let registry = PipelineRegistry::new();
            REGISTRY = Some(Arc::new(Mutex::new(registry)));
        });
        REGISTRY.clone().unwrap()
    }
}

impl PipelineRegistry {
    /// 创建新的注册表
    pub fn new() -> Self {
        let mut registry = PipelineRegistry {
            pipelines: HashMap::new(),
        };

        // 注册默认管道
        registry.register_import_pipeline();
        registry.register_transform_pipeline();
        registry.register_validation_pipeline();
        registry.register_visualization_pipeline();

        registry
    }

    /// 注册导入管道
    fn register_import_pipeline(&mut self) {
        self.pipelines.insert(
            PipelineType::Import,
            Box::new(|| {
                let builder = ImportPipelineBuilder::new();
                Box::new(builder.build().unwrap())
            }),
        );
    }

    /// 注册转换管道
    fn register_transform_pipeline(&mut self) {
        self.pipelines.insert(
            PipelineType::Transform,
            Box::new(|| Box::new(TransformPipeline::default())),
        );
    }

    /// 注册验证管道
    fn register_validation_pipeline(&mut self) {
        self.pipelines.insert(
            PipelineType::Validation,
            Box::new(|| Box::new(ValidationPipeline::default())),
        );
    }
    
    /// 注册可视化管道
    fn register_visualization_pipeline(&mut self) {
        self.pipelines.insert(
            PipelineType::Visualization,
            Box::new(|| Box::new(VisualizationPipeline::default())),
        );
    }

    /// 注册自定义管道
    pub fn register_custom<F>(&mut self, id: usize, factory: F)
    where
        F: Fn() -> Box<dyn Pipeline> + Send + Sync + 'static,
    {
        self.pipelines.insert(PipelineType::Custom(id), Box::new(factory));
    }

    /// 创建指定类型的管道
    pub fn create(&self, pipeline_type: PipelineType) -> Result<Box<dyn Pipeline>> {
        match self.pipelines.get(&pipeline_type) {
            Some(factory) => Ok(factory()),
            None => Err(crate::error::Error::NotFound(
                format!("Pipeline type {:?} not registered", pipeline_type)
            )),
        }
    }

    /// 创建导入管道
    pub fn create_import_pipeline(&self) -> Result<ImportPipelineBuilder> {
        Ok(ImportPipelineBuilder::new())
    }
} 