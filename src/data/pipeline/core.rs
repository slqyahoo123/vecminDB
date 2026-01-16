use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::data::{DataSchema, DataBatch, FieldType, SchemaMetadata};
use crate::data::record::{DataRecord, DataField};
use log::{debug, info, warn};

/// 数据处理管道接口
pub trait Pipeline: Send + Sync {
    /// 获取管道名称
    fn name(&self) -> &str;
    
    /// 获取管道ID
    fn id(&self) -> &str;
    
    /// 处理数据批次
    fn process(&self, batch: DataBatch) -> Result<DataBatch>;
    
    /// 获取管道配置
    fn config(&self) -> &PipelineConfig;
    
    /// 获取输入数据模式
    fn input_schema(&self) -> Option<&DataSchema>;
    
    /// 获取输出数据模式
    fn output_schema(&self) -> Result<DataSchema>;
    
    /// 验证输入数据
    fn validate_input(&self, batch: &DataBatch) -> Result<()>;
    
    /// 验证输出数据
    fn validate_output(&self, batch: &DataBatch) -> Result<()>;
}

/// 数据处理阶段接口
pub trait DataStage: Send + Sync {
    /// 获取阶段名称
    fn name(&self) -> &str;
    
    /// 处理数据批次
    fn process(&self, batch: &mut DataBatch) -> Result<()>;
    
    /// 获取处理结果模式
    fn output_schema(&self, input_schema: &DataSchema) -> Result<DataSchema>;
}

/// 数据验证器接口
pub trait DataValidator: Send + Sync {
    /// 获取验证器名称
    fn name(&self) -> &str;
    
    /// 验证数据批次
    fn validate(&self, batch: &DataBatch, schema: &DataSchema) -> Result<ValidationResult>;
}

use crate::core::interfaces::ValidationResult;

/// 验证错误
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// 错误字段
    pub field: Option<String>,
    
    /// 错误消息
    pub message: String,
    
    /// 错误记录索引
    pub record_index: Option<usize>,
}

/// 验证警告
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// 警告字段
    pub field: Option<String>,
    
    /// 警告消息
    pub message: String,
    
    /// 警告记录索引
    pub record_index: Option<usize>,
}

// Display 实现在 `core::interfaces::ValidationResult` 处统一提供，避免重复实现和字段不一致

/// 读取基于 details 的统计辅助方法，减少散落直取
pub trait ValidationResultStatsExt {
    fn total_records(&self) -> Option<usize>;
    fn valid_records(&self) -> Option<usize>;
    fn invalid_records(&self) -> Option<usize>;
    fn last_error_field(&self) -> Option<String>;
}

impl ValidationResultStatsExt for ValidationResult {
    fn total_records(&self) -> Option<usize> {
        self.metadata
            .get("total_records")
            .and_then(|s| s.parse::<usize>().ok())
    }

    fn valid_records(&self) -> Option<usize> {
        self.metadata
            .get("valid_records")
            .and_then(|s| s.parse::<usize>().ok())
    }

    fn invalid_records(&self) -> Option<usize> {
        self.metadata
            .get("invalid_records")
            .and_then(|s| s.parse::<usize>().ok())
    }

    fn last_error_field(&self) -> Option<String> {
        self.metadata.get("last_error_field").cloned()
    }
}

/// 从数据字段构建验证错误
impl ValidationError {
    /// 从数据字段创建验证错误
    pub fn from_field(field: &DataField, message: String, record_index: Option<usize>) -> Self {
        Self {
            field: Some(field.name.clone()),
            message,
            record_index,
        }
    }
}

/// 数据转换器接口
pub trait DataTransformer: Send + Sync {
    /// 获取转换器名称
    fn name(&self) -> &str;
    
    /// 转换数据批次
    fn transform(&self, batch: &DataBatch) -> Result<DataBatch>;
    
    /// 获取转换后的模式
    fn transform_schema(&self, schema: &DataSchema) -> Result<DataSchema>;
}

/// 数据管道配置
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// 管道ID
    pub id: String,
    /// 管道名称
    pub name: String,
    /// 管道描述
    pub description: Option<String>,
    /// 管道阶段配置
    pub stages: Vec<StageConfig>,
    /// 自定义配置
    pub custom_config: HashMap<String, String>,
}

/// 管道阶段配置
#[derive(Debug, Clone)]
pub struct StageConfig {
    /// 阶段ID
    pub id: String,
    /// 阶段类型
    pub stage_type: String,
    /// 阶段名称
    pub name: String,
    /// 阶段描述
    pub description: Option<String>,
    /// 阶段顺序
    pub order: usize,
    /// 阶段配置
    pub config: HashMap<String, String>,
    /// 自定义配置
    pub custom_config: HashMap<String, String>,
}

/// 管道执行上下文
pub struct PipelineContext {
    /// 上下文ID
    pub id: String,
    /// 管道实例ID
    pub pipeline_instance_id: String,
    /// 开始时间
    pub start_time: Option<std::time::Instant>,
    /// 结束时间
    pub end_time: Option<std::time::Instant>,
    /// 执行状态
    pub status: PipelineStatus,
    /// 错误信息
    pub errors: Vec<String>,
    /// 警告信息
    pub warnings: Vec<String>,
    /// 阶段执行上下文
    pub stage_contexts: HashMap<String, StageExecutionContext>,
    /// 元数据
    pub metadata: HashMap<String, String>,
    /// 统计信息
    pub statistics: HashMap<String, i64>,
    /// 模式元数据，用于存储数据字段的相关信息
    pub schema_metadata: Option<SchemaMetadata>,
}

/// 阶段执行上下文
pub struct StageExecutionContext {
    /// 阶段ID
    pub stage_id: String,
    /// 开始时间
    pub start_time: Option<std::time::Instant>,
    /// 结束时间
    pub end_time: Option<std::time::Instant>,
    /// 执行状态
    pub status: StageStatus,
    /// 错误信息
    pub errors: Vec<String>,
    /// 警告信息
    pub warnings: Vec<String>,
    /// 元数据
    pub metadata: HashMap<String, String>,
    /// 统计信息
    pub statistics: HashMap<String, i64>,
}

/// 管道状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStatus {
    /// 未开始
    NotStarted,
    /// 运行中
    Running,
    /// 已完成
    Completed,
    /// 失败
    Failed,
    /// 已取消
    Cancelled,
}

/// 阶段状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StageStatus {
    /// 未开始
    NotStarted,
    /// 运行中
    Running,
    /// 已完成
    Completed,
    /// 失败
    Failed,
    /// 已跳过
    Skipped,
}

impl PipelineContext {
    /// 创建新的管道上下文
    pub fn new(id: &str, pipeline_instance_id: &str) -> Self {
        Self {
            id: id.to_string(),
            pipeline_instance_id: pipeline_instance_id.to_string(),
            start_time: None,
            end_time: None,
            status: PipelineStatus::NotStarted,
            errors: Vec::new(),
            warnings: Vec::new(),
            stage_contexts: HashMap::new(),
            metadata: HashMap::new(),
            statistics: HashMap::new(),
            schema_metadata: None,
        }
    }
    
    /// 开始执行
    pub fn start(&mut self) {
        self.start_time = Some(std::time::Instant::now());
        self.status = PipelineStatus::Running;
    }
    
    /// 完成执行
    pub fn complete(&mut self) {
        self.end_time = Some(std::time::Instant::now());
        self.status = PipelineStatus::Completed;
    }
    
    /// 标记失败
    pub fn fail(&mut self, error: &str) {
        self.end_time = Some(std::time::Instant::now());
        self.status = PipelineStatus::Failed;
        self.errors.push(error.to_string());
    }
    
    /// 取消执行
    pub fn cancel(&mut self) {
        self.end_time = Some(std::time::Instant::now());
        self.status = PipelineStatus::Cancelled;
    }
    
    /// 获取执行时间
    pub fn execution_time(&self) -> Option<std::time::Duration> {
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            _ => None,
        }
    }
    
    /// 添加阶段上下文
    pub fn add_stage_context(&mut self, stage_id: &str) -> &mut StageExecutionContext {
        let stage_context = StageExecutionContext::new(stage_id);
        self.stage_contexts.insert(stage_id.to_string(), stage_context);
        self.stage_contexts.get_mut(stage_id).unwrap()
    }
    
    /// 获取阶段上下文
    pub fn get_stage_context(&self, stage_id: &str) -> Option<&StageExecutionContext> {
        self.stage_contexts.get(stage_id)
    }
    
    /// 获取可变阶段上下文
    pub fn get_stage_context_mut(&mut self, stage_id: &str) -> Option<&mut StageExecutionContext> {
        self.stage_contexts.get_mut(stage_id)
    }
    
    /// 设置模式元数据
    pub fn set_schema_metadata(&mut self, metadata: SchemaMetadata) {
        self.schema_metadata = Some(metadata);
    }
    
    /// 获取模式元数据
    pub fn get_schema_metadata(&self) -> Option<&SchemaMetadata> {
        self.schema_metadata.as_ref()
    }
    
    /// 更新模式元数据中的字段类型
    pub fn update_field_type(&mut self, field_name: &str, field_type: FieldType) -> Result<()> {
        if let Some(metadata) = &mut self.schema_metadata {
            // 将字段类型写入 properties 以保留信息
            metadata
                .properties
                .insert(format!("field_type:{}", field_name), format!("{:?}", field_type));
            Ok(())
        } else {
            Err(Error::invalid_state("模式元数据未初始化"))
        }
    }
    
    /// 添加统计信息
    pub fn add_statistic(&mut self, key: &str, value: i64) {
        self.statistics.insert(key.to_string(), value);
    }
    
    /// 增加统计计数
    pub fn increment_statistic(&mut self, key: &str, increment: i64) {
        let current = self.statistics.get(key).copied().unwrap_or(0);
        self.statistics.insert(key.to_string(), current + increment);
    }
    
    /// 添加元数据
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }
}

impl StageExecutionContext {
    /// 创建新的阶段执行上下文
    pub fn new(stage_id: &str) -> Self {
        Self {
            stage_id: stage_id.to_string(),
            start_time: None,
            end_time: None,
            status: StageStatus::NotStarted,
            errors: Vec::new(),
            warnings: Vec::new(),
            metadata: HashMap::new(),
            statistics: HashMap::new(),
        }
    }
    
    /// 开始执行
    pub fn start(&mut self) {
        self.start_time = Some(std::time::Instant::now());
        self.status = StageStatus::Running;
    }
    
    /// 完成执行
    pub fn complete(&mut self) {
        self.end_time = Some(std::time::Instant::now());
        self.status = StageStatus::Completed;
    }
    
    /// 标记失败
    pub fn fail(&mut self, error: &str) {
        self.end_time = Some(std::time::Instant::now());
        self.status = StageStatus::Failed;
        self.errors.push(error.to_string());
    }
    
    /// 标记跳过
    pub fn skip(&mut self) {
        self.status = StageStatus::Skipped;
    }
    
    /// 获取执行时间
    pub fn execution_time(&self) -> Option<std::time::Duration> {
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            _ => None,
        }
    }
    
    /// 添加统计信息
    pub fn add_statistic(&mut self, key: &str, value: i64) {
        self.statistics.insert(key.to_string(), value);
    }
    
    /// 增加统计计数
    pub fn increment_statistic(&mut self, key: &str, increment: i64) {
        let current = self.statistics.get(key).copied().unwrap_or(0);
        self.statistics.insert(key.to_string(), current + increment);
    }
    
    /// 添加元数据
    pub fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }
}

/// 管道阶段特质
pub trait PipelineStage: Send + Sync {
    /// 获取阶段名称
    fn name(&self) -> &str;
    
    /// 获取阶段配置
    fn config(&self) -> &StageConfig;
    
    /// 执行阶段处理
    fn execute(&self, batch: DataBatch, context: &mut StageExecutionContext) -> Result<DataBatch>;
}

/// 数据处理管道
pub struct DataPipeline {
    /// 管道配置
    config: PipelineConfig,
    
    /// 处理阶段
    stages: Vec<Box<dyn DataStage>>,
    
    /// 验证器
    validators: Vec<Box<dyn DataValidator>>,
    
    /// 输入模式
    input_schema: Option<Arc<DataSchema>>,
    
    /// 输出模式
    output_schema: Option<Arc<DataSchema>>,
}

impl DataPipeline {
    /// 创建新的数据管道
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            stages: Vec::new(),
            validators: Vec::new(),
            input_schema: None,
            output_schema: None,
        }
    }
    
    /// 添加处理阶段
    pub fn add_stage<S: DataStage + 'static>(&mut self, stage: S) -> &mut Self {
        info!("添加处理阶段: {}", stage.name());
        self.stages.push(Box::new(stage));
        self.output_schema = None; // 重置输出模式,需要重新计算
        self
    }
    
    /// 添加验证器
    pub fn add_validator<V: DataValidator + 'static>(&mut self, validator: V) -> &mut Self {
        if self.is_validation_enabled() {
            info!("添加数据验证器: {}", validator.name());
            self.validators.push(Box::new(validator));
        }
        self
    }
    
    /// 设置输入模式
    pub fn with_input_schema(&mut self, schema: DataSchema) -> &mut Self {
        self.input_schema = Some(Arc::new(schema));
        self.output_schema = None; // 重置输出模式,需要重新计算
        self
    }
    
    /// 计算输出模式
    pub fn compute_output_schema(&mut self) -> Result<DataSchema> {
        let input_schema = match &self.input_schema {
            Some(schema) => schema.as_ref().clone(),
            None => return Err(Error::invalid_state("未设置输入模式".to_string())),
        };
        
        let mut current_schema = input_schema;
        
        for stage in &self.stages {
            debug!("计算阶段 '{}' 的输出模式", stage.name());
            current_schema = stage.output_schema(&current_schema)?;
        }
        
        self.output_schema = Some(Arc::new(current_schema.clone()));
        Ok(current_schema)
    }
    
    /// 获取输出模式
    pub fn output_schema(&mut self) -> Result<DataSchema> {
        if let Some(schema) = &self.output_schema {
            return Ok(schema.as_ref().clone());
        }
        
        self.compute_output_schema()
    }

    /// 是否启用验证（从自定义配置读取，默认关闭）
    fn is_validation_enabled(&self) -> bool {
        self.config
            .custom_config
            .get("enable_validation")
            .map(|v| v == "true")
            .unwrap_or(false)
    }
    
    /// 处理数据批次
    pub fn process_batch(&self, mut batch: DataBatch) -> Result<DataBatch> {
        // 验证输入
        if self.is_validation_enabled() {
            self.validate_input(&batch)?;
        }
        
        // 依次执行各个处理阶段
        for stage in &self.stages {
            debug!("执行处理阶段: {}", stage.name());
            stage.process(&mut batch)?;
        }
        
        // 验证输出
        if self.is_validation_enabled() {
            self.validate_output(&batch)?;
        }
        
        Ok(batch)
    }
    
    /// 验证输入数据
    fn validate_input(&self, batch: &DataBatch) -> Result<()> {
        if !self.is_validation_enabled() {
            return Ok(());
        }
        
        let input_schema = match &self.input_schema {
            Some(schema) => schema.as_ref(),
            None => return Ok(()), // 无输入模式,跳过验证
        };
        
        debug!("验证输入数据");
        self.validate_batch(batch, input_schema)
    }
    
    /// 验证输出数据
    fn validate_output(&self, batch: &DataBatch) -> Result<()> {
        if !self.is_validation_enabled() {
            return Ok(());
        }
        
        let output_schema = match &self.output_schema {
            Some(schema) => schema.as_ref(),
            None => return Ok(()), // 无输出模式,跳过验证
        };
        
        debug!("验证输出数据");
        self.validate_batch(batch, output_schema)
    }
    
    /// 使用验证器验证数据批次
    fn validate_batch(&self, batch: &DataBatch, schema: &DataSchema) -> Result<()> {
        let mut errors = Vec::new();
        
        for validator in &self.validators {
            debug!("使用验证器 '{}' 验证数据", validator.name());
            let result = validator.validate(batch, schema)?;
            
            if !result.is_valid {
                for error in &result.errors {
                    errors.push(format!(
                        "验证器 '{}' 发现错误: {}", 
                        validator.name(),
                            error
                    ));
                }
                
                for warning in &result.warnings {
                    warn!(
                        "验证器 '{}' 发出警告: {}", 
                        validator.name(),
                            warning
                    );
                }
            }
        }
        
        if !errors.is_empty() {
            return Err(Error::validation(errors.join("; ")));
        }
        
        Ok(())
    }
    
    /// 将DataRecord转换为内部格式并处理
    pub fn process_records(&self, records: Vec<DataRecord>, schema: DataSchema) -> Result<Vec<DataRecord>> {
        debug!("开始处理记录集，记录数: {}", records.len());
        
        // 将记录集合转换为DataBatch
        let batch = DataBatch::from_data_records(records, Some(schema))?;
        
        // 处理批次
        let processed_batch = self.process_batch(batch)?;
        
        // 转换回记录集合
        let processed_records = processed_batch.to_records();
        
        debug!("完成记录处理，处理后记录数: {}", processed_records.len());
        
        Ok(processed_records)
    }
    
    /// 处理单条记录
    pub fn process_record(&self, record: DataRecord, schema: &DataSchema) -> Result<DataRecord> {
        // 创建只含一条记录的批次
        let batch = DataBatch::from_data_records(vec![record], Some(schema.clone()))?;
        
        // 处理批次
        let processed_batch = self.process_batch(batch)?;
        
        // 获取处理后的记录
        let processed_records = processed_batch.to_records();
        
        // 如果处理后没有记录或有多条记录，返回错误
        if processed_records.is_empty() {
            return Err(Error::processing("处理后记录为空"));
        }
        
        if processed_records.len() > 1 {
            warn!("处理单条记录产生了多条结果，只返回第一条");
        }
        
        Ok(processed_records[0].clone())
    }
    
    /// 根据字段名过滤记录
    pub fn filter_records_by_field(&self, records: Vec<DataRecord>, _field_name: &str, _field_value: &DataField) -> Result<Vec<DataRecord>> {
        // 字段值比较功能需要完整的类型比较实现，当前返回特征未启用错误
        // 如需使用字段过滤，请实现完整的类型比较逻辑或使用其他过滤方法
        Err(Error::feature_not_enabled(
            "字段值比较过滤需要完整的类型比较实现，当前未实现。请使用其他过滤方法或实现完整的类型比较逻辑。".to_string()
        ))
    }
    
    /// 提取记录中的指定字段
    pub fn extract_fields(&self, records: &[DataRecord], field_names: &[&str]) -> Result<Vec<HashMap<String, DataField>>> {
        let mut result = Vec::with_capacity(records.len());
        
        for record in records {
            let mut field_map = HashMap::new();
            
            for field_name in field_names {
                if let Some(value) = record.fields.get(*field_name) {
                    // 创建临时DataField用于兼容
                    let temp_field = crate::data::record::DataField::simple(
                        field_name.to_string(),
                        crate::data::schema::FieldType::String  // 临时使用String类型
                    );
                    field_map.insert(field_name.to_string(), temp_field);
                }
            }
            
            result.push(field_map);
        }
        
        Ok(result)
    }
    
    /// 将记录转换为指定模式
    pub fn convert_records_to_schema(&self, records: Vec<DataRecord>, target_schema: &DataSchema) -> Result<Vec<DataRecord>> {
        let mut converted_records = Vec::with_capacity(records.len());
        
        for record in records {
            // 创建新记录
            let mut new_record = DataRecord::new();
            
            // 为目标模式中的每个字段添加值
            for schema_field in target_schema.fields() {
                // 查找原记录中的对应字段
                let value = record
                    .fields
                    .get(schema_field.name())
                    .cloned()
                    .unwrap_or_else(|| {
                        // 使用空字符串作为默认值占位
                        crate::data::record::Value::Data(
                            crate::data::DataValue::String(String::new())
                        )
                    });
                
                // 添加到新记录
                new_record.add_field(schema_field.name(), value);
            }
            
            converted_records.push(new_record);
        }
        
        Ok(converted_records)
    }
    
    /// 验证记录是否符合模式
    pub fn validate_records(&self, records: &[DataRecord], schema: &DataSchema) -> Result<ValidationResult> {
        // 转换为DataBatch以使用现有验证功能
        let batch = DataBatch::from_data_records(records.to_vec(), Some(schema.clone()))?;
        
        // 调用已有的验证功能
        let mut all_results = ValidationResult::success();
        all_results.metadata.insert("scope".into(), "validate_records".into());
        
        // 使用所有验证器进行验证
        for validator in &self.validators {
            let result = validator.validate(&batch, schema)?;
            
            // 合并验证结果
            all_results.is_valid = all_results.is_valid && result.is_valid;
            all_results.errors.extend(result.errors);
            all_results.warnings.extend(result.warnings);
        }
        
        Ok(all_results)
    }
    
    /// 基于记录集计算聚合数据
    pub fn aggregate_records(&self, records: &[DataRecord], group_by_field: &str, aggregate_field: &str, operation: AggregateOperation) -> Result<HashMap<String, f64>> {
        let mut result = HashMap::new();
        
        // 按分组字段值对记录进行分组
        let mut grouped_records: HashMap<String, Vec<&DataRecord>> = HashMap::new();
        
        for record in records {
            if let Some(group_value_obj) = record.fields.get(group_by_field) {
                if let Ok(group_data_value) = group_value_obj.to_data_value() {
                    let group_value = group_data_value.to_string();
                    grouped_records.entry(group_value).or_default().push(record);
                }
            }
        }
        
        // 为每个分组计算聚合值
        for (group_value, group_records) in grouped_records {
            // 提取要聚合的字段值
            let values: Vec<f64> = group_records.iter()
                .filter_map(|record| {
                    record.fields.get(aggregate_field)
                        .and_then(|field_value| {
                            if let Ok(data_value) = field_value.to_data_value() {
                                match data_value {
                                    crate::data::value::DataValue::Float(f) => Some(f),
                                    crate::data::value::DataValue::Integer(i) => Some(i as f64),
                                    _ => None,
                                }
                            } else {
                                None
                            }
                        })
                })
                .collect();
                
            // 执行聚合操作
            let aggregate_value = match operation {
                AggregateOperation::Sum => values.iter().sum(),
                AggregateOperation::Average => {
                    if values.is_empty() {
                        0.0
                    } else {
                        values.iter().sum::<f64>() / values.len() as f64
                    }
                },
                AggregateOperation::Min => values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                AggregateOperation::Max => values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                AggregateOperation::Count => values.len() as f64,
            };
            
            result.insert(group_value, aggregate_value);
        }
        
        Ok(result)
    }
}

/// 聚合操作类型
#[derive(Debug, Clone, Copy)]
pub enum AggregateOperation {
    /// 求和
    Sum,
    /// 平均值
    Average,
    /// 最小值
    Min,
    /// 最大值
    Max,
    /// 计数
    Count,
}

/// 数据处理器
pub struct DataProcessor {
    /// 数据管道
    pipeline: DataPipeline,
}

impl DataProcessor {
    /// 创建新的数据处理器
    pub fn new(pipeline: DataPipeline) -> Self {
        Self { pipeline }
    }
    
    /// 处理数据批次
    pub fn process(&self, batch: DataBatch) -> Result<DataBatch> {
        self.pipeline.process_batch(batch)
    }
    
    /// 获取输出模式
    pub fn output_schema(&mut self) -> Result<DataSchema> {
        self.pipeline.output_schema()
    }
} 