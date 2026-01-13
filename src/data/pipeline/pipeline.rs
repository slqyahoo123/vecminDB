use crate::data::{DataBatch};
use crate::Result;
use std::collections::HashMap;
use std::sync::Arc;
use crate::error::Error;
use crate::data::record::{Record, RecordSet};
use serde_json;
use std::sync::Mutex;
use crate::data::pipeline::monitor::PipelineMonitor;

pub struct DataPipeline {
    stages: Vec<Box<dyn DataStage>>,
    validators: Vec<Box<dyn DataValidator>>,
    config: PipelineConfig,
}

pub trait DataStage: Send + Sync {
    fn process(&self, data: &mut DataBatch) -> Result<()>;
    fn requirements(&self) -> Vec<DataRequirement>;
}

pub trait DataValidator: Send + Sync {
    fn validate(&self, data: &DataBatch) -> Result<()>;
    fn name(&self) -> &str;
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub batch_size: usize,
    pub shuffle: bool,
    pub validation_split: f32,
    pub feature_columns: Vec<String>,
    pub target_column: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DataRequirement {
    pub name: String,
    pub required: bool,
}

/// 管道结果
#[derive(Debug)]
pub enum PipelineResult {
    /// 成功处理，包含详细信息
    Success {
        /// 处理的记录数
        processed_rows: usize,
        /// 写入的记录数
        written_rows: usize,
        /// 执行时间(秒)
        execution_time: f64,
        /// 各阶段的执行结果
        stage_results: Vec<(String, String)>,
        /// 执行上下文
        context: PipelineContext,
    },
    
    /// 成功处理，返回单条记录
    Record {
        /// 记录数据
        record: Record,
        /// 执行上下文
        context: PipelineContext,
    },
    
    /// 成功处理，返回记录集
    RecordSet {
        /// 记录集数据
        records: RecordSet,
        /// 执行上下文
        context: PipelineContext,
    },
    
    /// 成功处理，返回JSON值
    Value {
        /// JSON数据
        value: serde_json::Value,
        /// 执行上下文
        context: PipelineContext,
    },
    
    /// 需要额外处理
    NeedAction {
        /// 动作类型
        action: String,
        /// 动作参数
        params: serde_json::Value,
        /// 执行上下文
        context: PipelineContext,
    },
    
    /// 处理出错
    Error(String),
    
    /// 处理出错，包含详细信息
    ErrorWithContext {
        /// 错误消息
        message: String,
        /// 错误代码
        code: Option<String>,
        /// 错误详情
        details: Option<serde_json::Value>,
        /// 执行上下文
        context: PipelineContext,
    },
}

impl PipelineResult {
    /// 检查结果是否成功
    pub fn is_success(&self) -> bool {
        matches!(
            self,
            PipelineResult::Success { .. } |
            PipelineResult::Record { .. } |
            PipelineResult::RecordSet { .. } |
            PipelineResult::Value { .. }
        )
    }
    
    /// 检查是否需要额外处理
    pub fn needs_action(&self) -> bool {
        matches!(self, PipelineResult::NeedAction { .. })
    }
    
    /// 检查是否出错
    pub fn is_error(&self) -> bool {
        matches!(
            self,
            PipelineResult::Error(_) |
            PipelineResult::ErrorWithContext { .. }
        )
    }
    
    /// 获取执行上下文
    pub fn context(&self) -> Option<&PipelineContext> {
        match self {
            PipelineResult::Success { context, .. } => Some(context),
            PipelineResult::Record { context, .. } => Some(context),
            PipelineResult::RecordSet { context, .. } => Some(context),
            PipelineResult::Value { context, .. } => Some(context),
            PipelineResult::NeedAction { context, .. } => Some(context),
            PipelineResult::ErrorWithContext { context, .. } => Some(context),
            PipelineResult::Error(_) => None,
        }
    }
    
    /// 获取错误消息
    pub fn error_message(&self) -> Option<&str> {
        match self {
            PipelineResult::Error(msg) => Some(msg),
            PipelineResult::ErrorWithContext { message, .. } => Some(message),
            _ => None,
        }
    }
    
    /// 获取处理的记录数
    pub fn processed_rows(&self) -> Option<usize> {
        match self {
            PipelineResult::Success { processed_rows, .. } => Some(*processed_rows),
            _ => None,
        }
    }
    
    /// 获取写入的记录数
    pub fn written_rows(&self) -> Option<usize> {
        match self {
            PipelineResult::Success { written_rows, .. } => Some(*written_rows),
            _ => None,
        }
    }
    
    /// 获取执行时间
    pub fn execution_time(&self) -> Option<f64> {
        match self {
            PipelineResult::Success { execution_time, .. } => Some(*execution_time),
            _ => None,
        }
    }
    
    /// 获取阶段结果
    pub fn stage_results(&self) -> Option<&[(String, String)]> {
        match self {
            PipelineResult::Success { stage_results, .. } => Some(stage_results),
            _ => None,
        }
    }
}

/// 管道上下文
#[derive(Debug, Default)]
pub struct PipelineContext {
    /// 上下文参数
    pub params: HashMap<String, serde_json::Value>,
    /// 上下文数据
    pub data: HashMap<String, serde_json::Value>,
    /// 临时存储
    pub temp: HashMap<String, serde_json::Value>,
    /// 状态信息
    pub state: HashMap<String, String>,
    /// 阶段结果
    pub results: Vec<PipelineResult>,
}

impl PipelineContext {
    /// 创建新的管道上下文
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
            data: HashMap::new(),
            temp: HashMap::new(),
            state: HashMap::new(),
            results: Vec::new(),
        }
    }

    /// 添加参数
    pub fn add_param<T>(&mut self, key: &str, value: T) -> Result<(), Error> 
    where
        T: serde::Serialize,
    {
        let value = serde_json::to_value(value)
            .map_err(|e| Error::serialization(&format!("无法序列化参数 {}: {}", key, e)))?;
        self.params.insert(key.to_string(), value);
        Ok(())
    }

    /// 获取参数
    pub fn get_param<T>(&self, key: &str) -> Result<T, Error> 
    where
        T: serde::de::DeserializeOwned,
    {
        let value = self.params.get(key)
            .ok_or_else(|| Error::not_found(&format!("参数未找到: {}", key)))?;
            
        serde_json::from_value(value.clone())
            .map_err(|e| Error::deserialization(&format!("无法反序列化参数 {}: {}", key, e)))
    }

    /// 添加数据
    pub fn add_data<T>(&mut self, key: &str, value: T) -> Result<(), Error> 
    where
        T: serde::Serialize,
    {
        let value = serde_json::to_value(value)
            .map_err(|e| Error::serialization(&format!("无法序列化数据 {}: {}", key, e)))?;
        self.data.insert(key.to_string(), value);
        Ok(())
    }

    /// 获取数据
    pub fn get_data<T>(&self, key: &str) -> Result<T, Error> 
    where
        T: serde::de::DeserializeOwned,
    {
        let value = self.data.get(key)
            .ok_or_else(|| Error::not_found(&format!("数据未找到: {}", key)))?;
            
        serde_json::from_value(value.clone())
            .map_err(|e| Error::deserialization(&format!("无法反序列化数据 {}: {}", key, e)))
    }

    /// 添加临时数据
    pub fn add_temp<T>(&mut self, key: &str, value: T) -> Result<(), Error> 
    where
        T: serde::Serialize,
    {
        let value = serde_json::to_value(value)
            .map_err(|e| Error::serialization(&format!("无法序列化临时数据 {}: {}", key, e)))?;
        self.temp.insert(key.to_string(), value);
        Ok(())
    }

    /// 获取临时数据
    pub fn get_temp<T>(&self, key: &str) -> Result<T, Error> 
    where
        T: serde::de::DeserializeOwned,
    {
        let value = self.temp.get(key)
            .ok_or_else(|| Error::not_found(&format!("临时数据未找到: {}", key)))?;
            
        serde_json::from_value(value.clone())
            .map_err(|e| Error::deserialization(&format!("无法反序列化临时数据 {}: {}", key, e)))
    }

    /// 设置状态
    pub fn set_state(&mut self, key: &str, value: &str) {
        self.state.insert(key.to_string(), value.to_string());
    }

    /// 获取状态
    pub fn get_state(&self, key: &str) -> Option<&String> {
        self.state.get(key)
    }

    /// 获取字符串值
    pub fn get_string(&self, key: &str) -> Result<String, Error> {
        // 尝试从不同的存储获取值
        if let Some(value) = self.params.get(key) {
            return serde_json::from_value(value.clone())
                .map_err(|e| Error::deserialization(&format!("无法反序列化参数 {}: {}", key, e)));
        }
        
        if let Some(value) = self.data.get(key) {
            return serde_json::from_value(value.clone())
                .map_err(|e| Error::deserialization(&format!("无法反序列化数据 {}: {}", key, e)));
        }
        
        if let Some(value) = self.temp.get(key) {
            return serde_json::from_value(value.clone())
                .map_err(|e| Error::deserialization(&format!("无法反序列化临时数据 {}: {}", key, e)));
        }
        
        if let Some(value) = self.state.get(key) {
            return Ok(value.clone());
        }
        
        Err(Error::not_found(&format!("键不存在: {}", key)))
    }

    /// 设置字符串值
    pub fn set_string(&mut self, key: &str, value: String) {
        self.temp.insert(key.to_string(), serde_json::Value::String(value));
    }

    /// 检查键是否存在
    pub fn has_key(&self, key: &str) -> bool {
        self.params.contains_key(key) || 
        self.data.contains_key(key) || 
        self.temp.contains_key(key) || 
        self.state.contains_key(key)
    }

    /// 添加结果
    pub fn add_result(&mut self, result: PipelineResult) {
        self.results.push(result);
    }

    /// 获取最后一个结果
    pub fn last_result(&self) -> Option<&PipelineResult> {
        self.results.last()
    }

    /// 获取转换列表 - 生产级实现
    pub fn get_transformations(&self) -> Option<Vec<crate::data::DataTransformation>> {
        // 首先尝试从数据中获取明确的转换列表
        if let Ok(transformations) = self.get_data::<Vec<crate::data::DataTransformation>>("transformations") {
            return Some(transformations);
        }
        
        // 尝试从临时存储中获取
        if let Ok(transformations) = self.get_temp::<Vec<crate::data::DataTransformation>>("transformations") {
            return Some(transformations);
        }
        
        // 尝试从参数中获取
        if let Ok(transformations) = self.get_param::<Vec<crate::data::DataTransformation>>("transformations") {
            return Some(transformations);
        }
        
        // 如果没有找到明确的转换列表，尝试从状态中重建
        let mut transformations = Vec::new();
        
        // 检查状态中的转换相关信息
        for (key, value) in &self.state {
            if key.starts_with("transformation_") {
                if let Ok(transformation) = serde_json::from_str::<crate::data::DataTransformation>(value) {
                    transformations.push(transformation);
                }
            }
        }
        
        // 如果找到转换，按序号排序后返回
        if !transformations.is_empty() {
            transformations.sort_by(|a, b| a.name.cmp(&b.name));
            Some(transformations)
        } else {
            None
        }
    }
}

/// 管道阶段特性
pub trait PipelineStage: Send + Sync {
    /// 阶段名称
    fn name(&self) -> &str;
    
    /// 阶段描述
    fn description(&self) -> Option<&str> {
        None
    }
    
    /// 处理阶段
    fn process(&self, context: &mut PipelineContext) -> Result<(), Error>;
    
    /// 检查阶段是否可以处理
    fn can_process(&self, context: &PipelineContext) -> bool {
        true
    }
    
    /// 获取阶段元数据
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

/// 管道接口
pub trait Pipeline: Send + Sync {
    /// 管道名称
    fn name(&self) -> &str;
    
    /// 管道描述
    fn description(&self) -> Option<&str> {
        None
    }
    
    /// 添加阶段
    fn add_stage(&mut self, stage: Arc<dyn PipelineStage>) -> Result<(), Error>;
    
    /// 执行管道
    fn execute(&self, context: PipelineContext) -> PipelineResult;
    
    /// 获取阶段列表
    fn stages(&self) -> &[Arc<dyn PipelineStage>];
    
    /// 重置管道
    fn reset(&mut self) -> Result<()>;
    
    /// 获取管道元数据
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    /// 添加监控器
    fn set_monitor(&mut self, monitor: Arc<Mutex<PipelineMonitor>>);
    
    /// 添加错误处理器
    fn set_error_handler(&mut self, handler: Box<dyn Fn(&mut PipelineContext, &str, &str) -> bool + Send + Sync>);
}

/// 基本管道实现
pub struct BasicPipeline {
    name: String,
    description: Option<String>,
    stages: Vec<Arc<dyn PipelineStage>>,
    metadata: HashMap<String, String>,
    error_handler: Option<Box<dyn Fn(&mut PipelineContext, &str, &str) -> bool + Send + Sync>>,
    monitor: Option<Arc<Mutex<PipelineMonitor>>>,
}

impl BasicPipeline {
    /// 创建新的基本管道
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: None,
            stages: Vec::new(),
            metadata: HashMap::new(),
            error_handler: None,
            monitor: None,
        }
    }

    /// 设置描述
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    /// 添加元数据
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

impl Pipeline for BasicPipeline {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
    
    fn add_stage(&mut self, stage: Arc<dyn PipelineStage>) -> Result<(), Error> {
        self.stages.push(stage);
        Ok(())
    }
    
    fn execute(&self, mut context: PipelineContext) -> PipelineResult {
        // 记录管道开始
        if let Some(monitor) = &self.monitor {
            if let Ok(mut monitor) = monitor.lock() {
                monitor.record_pipeline_start(&self.name);
            }
        }

        // 依次执行每个阶段
        for stage in &self.stages {
            // 检查是否可以执行
            if !stage.can_process(&context) {
                continue;
            }

            // 记录阶段开始
            let start_time = std::time::Instant::now();
            if let Some(monitor) = &self.monitor {
                if let Ok(mut monitor) = monitor.lock() {
                    monitor.record_stage_start(&self.name, stage.name());
                }
            }

            // 执行阶段
            let result = stage.process(&mut context);

            // 记录阶段结果
            let duration = start_time.elapsed().as_millis() as u64;
            if let Some(monitor) = &self.monitor {
                if let Ok(mut monitor) = monitor.lock() {
                    match &result {
                        Ok(_) => {
                            monitor.record_stage_complete(&self.name, stage.name(), duration);
                        }
                        Err(e) => {
                            monitor.record_stage_failed(&self.name, stage.name(), &e.to_string(), duration);
                        }
                    }
                }
            }

            // 处理阶段结果
            match result {
                Ok(_) => {
                    // 阶段成功，继续执行
                }
                Err(e) => {
                    // 阶段失败
                    let error_message = e.to_string();
                    
                    // 检查是否有错误处理器
                    if let Some(handler) = &self.error_handler {
                        // 调用错误处理器
                        let continue_execution = handler(&mut context, &error_message, stage.name());
                        
                        // 如果错误处理器返回false，中断执行
                        if !continue_execution {
                            // 记录管道失败
                            if let Some(monitor) = &self.monitor {
                                if let Ok(mut monitor) = monitor.lock() {
                                    monitor.record_pipeline_failed(&self.name, &error_message);
                                }
                            }
                            
                            return PipelineResult::Error(error_message);
                        }
                        
                        // 否则继续执行下一个阶段
                    } else {
                        // 没有错误处理器，中断执行
                        // 记录管道失败
                        if let Some(monitor) = &self.monitor {
                            if let Ok(mut monitor) = monitor.lock() {
                                monitor.record_pipeline_failed(&self.name, &error_message);
                            }
                        }
                        
                        return PipelineResult::Error(error_message);
                    }
                }
            }
        }

        // 记录管道完成
        if let Some(monitor) = &self.monitor {
            if let Ok(mut monitor) = monitor.lock() {
                monitor.record_pipeline_complete(&self.name);
            }
        }

        // 返回成功结果
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
    
    fn metadata(&self) -> HashMap<String, String> {
        self.metadata.clone()
    }

    fn set_monitor(&mut self, monitor: Arc<Mutex<PipelineMonitor>>) {
        self.monitor = Some(monitor);
    }

    fn set_error_handler(&mut self, handler: Box<dyn Fn(&mut PipelineContext, &str, &str) -> bool + Send + Sync>) {
        self.error_handler = Some(handler);
    }
}

impl DataPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            stages: Vec::new(),
            validators: Vec::new(),
            config,
        }
    }

    pub fn add_stage(&mut self, stage: Box<dyn DataStage>) {
        self.stages.push(stage);
    }

    pub fn add_validator(&mut self, validator: Box<dyn DataValidator>) {
        self.validators.push(validator);
    }

    pub fn process_batch(&mut self, mut batch: DataBatch) -> Result<DataBatch> {
        // 1. Validate input
        self.validate_input(&batch)?;
        
        // 2. Process through stages
        for stage in &self.stages {
            stage.process(&mut batch)?;
        }
        
        // 3. Validate output
        self.validate_output(&batch)?;
        
        Ok(batch)
    }

    fn validate_input(&self, batch: &DataBatch) -> Result<()> {
        for validator in &self.validators {
            validator.validate(batch)?;
        }
        Ok(())
    }

    fn validate_output(&self, batch: &DataBatch) -> Result<()> {
        for validator in &self.validators {
            validator.validate(batch)?;
        }
        Ok(())
    }
} 