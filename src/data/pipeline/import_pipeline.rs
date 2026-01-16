use std::path::Path;
use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Instant, Duration};
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use log::{debug, info, warn, error};

use crate::data::loader::file::{FileDataLoader, FileProcessor};
use crate::data::loader::FileType;
use crate::data::loader::file::processor_factory::{FileProcessorFactory, FileType as ProcessorFileType};
use crate::data::loader::DataLoader;
use crate::data::pipeline::{Pipeline, PipelineContext, PipelineResult, PipelineStage};
use crate::data::pipeline::monitor::PipelineMonitor;
use crate::data::schema::DataSchema;
use crate::data::processor::DataProcessor;
use crate::data::processor::types::ProcessorOptions;
use crate::Error;
use crate::storage::StorageEngine;
use crate::storage::engine::implementation::StorageOptions;
use crate::data::value::DataValue;

use crate::data::pipeline::validation::{DataValidationStage as DataValidationStageBase};
use crate::data::pipeline::storage_writer::{StorageWriteStage as StorageWriteStageBase};
use crate::data::pipeline::performance::{PerformanceMonitorStage as PerformanceMonitorStageBase};
use crate::core::interfaces::ValidationResult;

// serde derives aren't directly used here; remove to silence unused

use crate::data::record::Record;
use crate::data::schema::Schema;
#[cfg(feature = "arrow")]
use arrow::datatypes::DataType;
use serde_json::Value;
use crate::storage::engine::factory::{StorageEngineFactory, IStorageEngine};
use crate::data::DataBatch;
use rust_decimal::Decimal;
// storage is accessed via engine factory below; direct alias unused

/// 定义FieldValue枚举
#[derive(Debug, Clone)]
pub enum FieldValue {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Date(chrono::NaiveDate),
    DateTime(chrono::NaiveDateTime),
    Time(chrono::NaiveTime),
    Decimal(Decimal),
    Binary(Vec<u8>),
    List(Vec<FieldValue>),
    Struct(HashMap<String, FieldValue>),
}

/// 导入管道配置
#[derive(Debug, Clone)]
pub struct ImportPipelineConfig {
    /// 源文件路径
    pub source_path: String,
    /// 文件格式，如果为None则尝试自动检测
    pub format: Option<FileType>,
    /// 目标存储位置
    pub target_location: String,
    /// 批处理大小
    pub batch_size: usize,
    /// 是否进行验证
    pub validate: bool,
    /// 是否覆盖已有数据
    pub overwrite: bool,
    /// 是否推断模式
    pub infer_schema: bool,
    /// 自定义模式(如果不推断)
    pub schema: Option<DataSchema>,
    /// 处理器选项
    pub processor_options: ProcessorOptions,
    /// 存储选项
    pub storage_options: StorageOptions,
    /// 文件处理器特定选项
    pub file_options: HashMap<String, String>,
    /// 性能监控选项
    pub monitor_performance: bool,
}

impl Default for ImportPipelineConfig {
    fn default() -> Self {
        Self {
            source_path: String::new(),
            format: None,
            target_location: String::new(),
            batch_size: 1000,
            validate: true,
            overwrite: false,
            infer_schema: true,
            schema: None,
            processor_options: ProcessorOptions::default(),
            storage_options: StorageOptions::default(),
            file_options: HashMap::new(),
            monitor_performance: false,
        }
    }
}

/// 为了向后兼容和类型别名
pub type PipelineConfig = ImportPipelineConfig;

// 数据验证特性
pub trait DataValidationTrait: Send + Sync {
    fn validate(&self, data: &[u8]) -> Result<bool, crate::Error>;
    fn name(&self) -> &str;
}

// 数据转换特性
pub trait DataTransformationTrait: Send + Sync {
    fn transform(&self, data: &[u8]) -> Result<Vec<u8>, crate::Error>;
    fn name(&self) -> &str;
}

/// 导入管道构建器
pub struct ImportPipelineBuilder {
    stages: Vec<Box<dyn PipelineStage + Send + Sync>>,
    config: PipelineConfig,
    validators: Vec<Box<dyn DataValidationTrait + Send + Sync>>,
    transformers: Vec<Box<dyn DataTransformationTrait + Send + Sync>>,
}

impl Clone for ImportPipelineBuilder {
    fn clone(&self) -> Self {
        // trait object 不能直接 clone，需要重新构建
        // 这里返回一个新的构建器，使用相同的配置
        Self {
            stages: Vec::new(), // 不能 clone trait object，需要重新添加
            config: self.config.clone(),
            validators: Vec::new(), // 不能 clone trait object，需要重新添加
            transformers: Vec::new(), // 不能 clone trait object，需要重新添加
        }
    }
}

impl ImportPipelineBuilder {
    /// 创建新的构建器
    pub fn new() -> Self {
        Self {
            config: ImportPipelineConfig::default(),
            stages: Vec::new(),
            validators: Vec::new(),
            transformers: Vec::new(),
        }
    }

    /// 设置源文件路径
    pub fn with_source(mut self, source: &str) -> Self {
        self.config.source_path = source.to_string();
        self
    }

    /// 设置文件格式
    pub fn with_format(mut self, format: FileType) -> Self {
        self.config.format = Some(format);
        self
    }

    /// 设置目标位置
    pub fn with_target(mut self, target: &str) -> Self {
        self.config.target_location = target.to_string();
        self
    }

    /// 设置批处理大小
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// 设置是否验证
    pub fn with_validation(mut self, validate: bool) -> Self {
        self.config.validate = validate;
        self
    }

    /// 设置是否覆盖
    pub fn with_overwrite(mut self, overwrite: bool) -> Self {
        self.config.overwrite = overwrite;
        self
    }

    /// 设置是否推断模式
    pub fn with_schema_inference(mut self, infer: bool) -> Self {
        self.config.infer_schema = infer;
        self
    }

    /// 设置自定义模式
    pub fn with_schema(mut self, schema: DataSchema) -> Self {
        self.config.schema = Some(schema);
        self.config.infer_schema = false;
        self
    }

    /// 设置处理器选项
    pub fn with_processor_options(mut self, options: ProcessorOptions) -> Self {
        self.config.processor_options = options;
        self
    }

    /// 设置存储选项
    pub fn with_storage_options(mut self, options: StorageOptions) -> Self {
        self.config.storage_options = options;
        self
    }
    
    /// 设置文件处理器特定选项
    pub fn with_file_option(mut self, key: &str, value: &str) -> Self {
        self.config.file_options.insert(key.to_string(), value.to_string());
        self
    }
    
    /// 设置多个文件处理器选项
    pub fn with_file_options(mut self, options: HashMap<String, String>) -> Self {
        self.config.file_options.extend(options);
        self
    }
    
    /// 设置是否监控性能
    pub fn with_performance_monitoring(mut self, monitor: bool) -> Self {
        self.config.monitor_performance = monitor;
        self
    }

    /// 启用自动格式检测
    pub fn with_auto_format_detection(mut self) -> Self {
        // 如果已经有了source_path，尝试从路径检测格式
        if !self.config.source_path.is_empty() {
            let path = Path::new(&self.config.source_path);
            if let Ok(processor_format) = FileProcessorFactory::detect_file_type(path) {
                // 将 processor_factory::FileType 转换为 loader::types::FileType
                let format = match processor_format {
                    ProcessorFileType::CSV => FileType::Csv,
                    ProcessorFileType::JSON => FileType::Json,
                    ProcessorFileType::Parquet => FileType::Parquet,
                    ProcessorFileType::Excel => FileType::Excel,
                    ProcessorFileType::Avro => FileType::Avro,
                    ProcessorFileType::Unknown => FileType::Unknown,
                    _ => FileType::Unknown, // 其他类型映射为 Unknown
                };
                if format != FileType::Unknown {
                    self.config.format = Some(format);
                }
            }
        }
        self
    }

    /// 构建导入管道
    pub fn build(self) -> Result<ImportPipeline, crate::Error> {
        // 验证配置
        if self.config.source_path.is_empty() {
            return Err(crate::Error::invalid_input("源路径不能为空"));
        }

        if self.config.target_location.is_empty() {
            return Err(crate::Error::invalid_input("目标位置不能为空"));
        }

        // 创建管道实例
        let pipeline = ImportPipeline {
            config: self.config,
            stages: Vec::new(),
            context: PipelineContext::new(),
        };
        
        Ok(pipeline)
    }
}

/// 导入管道
pub struct ImportPipeline {
    config: ImportPipelineConfig,
    stages: Vec<Arc<dyn PipelineStage>>,
    context: PipelineContext,
}

impl ImportPipeline {
    /// 创建新的导入管道
    pub fn new(config: ImportPipelineConfig) -> Self {
        Self {
            config,
            stages: Vec::new(),
            context: PipelineContext::new(),
        }
    }
    
    /// 准备管道
    pub fn prepare(&mut self) -> Result<(), crate::Error> {
        // 设置上下文
        self.context.add_param("source_path", &self.config.source_path)?;
        self.context.add_param("target_location", &self.config.target_location)?;
        self.context.add_param("batch_size", self.config.batch_size.to_string())?;
        self.context.add_param("validate", self.config.validate.to_string())?;
        self.context.add_param("overwrite", self.config.overwrite.to_string())?;
        self.context.add_param("infer_schema", self.config.infer_schema.to_string())?;
        
        if let Some(format) = &self.config.format {
            self.context.add_param("format", format.to_string())?;
        }
        
        self.context.add_data("processor_options", self.config.processor_options.clone())?;
        self.context.add_data("storage_options", self.config.storage_options.clone())?;
        self.context.add_data("file_options", self.config.file_options.clone())?;
        
        if let Some(schema) = &self.config.schema {
            self.context.add_data("schema", schema.clone())?;
        }
        
        // 创建并添加阶段
        
        // 1. 文件检测阶段
        let file_detection = FileDetectionStage::new(
            self.config.source_path.clone(), 
            self.config.format,
            self.config.file_options.clone(),
        );
        self.add_stage(Arc::new(file_detection))?;
        
        // 2. 模式推断阶段
        let schema_inference = SchemaInferenceStage::new(
                self.config.infer_schema, 
                self.config.schema.clone(), 
            1000, // 样本大小，可以后续配置
            self.config.processor_options.clone(),
        );
        self.add_stage(Arc::new(schema_inference))?;
        
        // 3. 数据导入阶段
        let data_import = DataImportStage::new(
            self.config.batch_size,
            self.config.processor_options.clone(),
        );
        self.add_stage(Arc::new(data_import))?;
        
        // 4. 数据验证阶段 (如果启用)
        if self.config.validate {
            let data_validation = DataValidationStage::new(true, true);
            self.add_stage(Arc::new(data_validation))?;
        }
        
        // 5. 存储写入阶段
        let storage_write = StorageWriteStage::new(
            self.config.target_location.clone(),
            self.config.overwrite,
            self.config.storage_options.clone(),
        );
        self.add_stage(Arc::new(storage_write))?;
        
        // 6. 性能监控阶段 (如果启用)
        if self.config.monitor_performance {
            let performance_monitor = PerformanceMonitorStage::new();
            self.add_stage(Arc::new(performance_monitor))?;
        }
        
        Ok(())
    }
}

impl Pipeline for ImportPipeline {
    fn name(&self) -> &str {
        "导入管道"
    }
    
    fn description(&self) -> Option<&str> {
        Some("用于数据导入的处理管道")
    }
    
    fn add_stage(&mut self, stage: Arc<dyn PipelineStage>) -> Result<(), crate::Error> {
        self.stages.push(stage);
        Ok(())
    }
    
    fn stages(&self) -> &[Arc<dyn PipelineStage>] {
        &self.stages
    }
    
    fn execute(&self, context: PipelineContext) -> PipelineResult {
        // 由于函数签名变更，我们需要重新实现这个方法
        // 创建可变上下文的副本用于处理
        let mut ctx = context;
        
        info!("开始执行导入管道");
        
        let start_time = Instant::now();
        
        // 记录开始时间（使用时间戳字符串）
        let start_time_str = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            .to_string();
        match ctx.add_data("pipeline_start_time", &start_time_str) {
            Ok(()) => {},
            Err(e) => {
                error!("无法添加开始时间到上下文: {}", e);
                return PipelineResult::ErrorWithContext {
                    message: format!("初始化失败: {}", e),
                    code: Some("INIT_ERROR".to_string()),
                    details: None,
                    context: ctx,
                };
            }
        }
        
        // 使用可变变量来构建结果
        let mut processed_rows = 0;
        let mut written_rows = 0;
        let mut execution_time = 0.0;
        let mut stage_results = Vec::new();
        let mut has_error = false;
        let mut error_message = String::new();
        
        // 存储阶段执行时间
        let mut stage_durations = HashMap::new();
        
        // 执行各个阶段
        for (index, stage) in self.stages.iter().enumerate() {
            let stage_name = stage.name();
            info!("执行阶段 {}/{}: {}", index + 1, self.stages.len(), stage_name);
            
            // 检查是否可以处理
            if !stage.can_process(&ctx) {
                warn!("阶段 {} 无法处理当前上下文，跳过", stage_name);
                continue;
            }
            
            // 记录开始时间
            let stage_start = Instant::now();
            
            // 执行处理
            match stage.process(&mut ctx) {
                Ok(()) => {
                    info!("阶段 {} 执行成功", stage_name);
                },
                Err(e) => {
                    error!("阶段 {} 执行失败: {}", stage_name, e);
                    has_error = true;
                    error_message = format!("阶段 {} 失败: {}", stage_name, e);
                    // 计算总时间
                    let total_duration = start_time.elapsed();
                    execution_time = total_duration.as_secs_f64();
                    return PipelineResult::ErrorWithContext {
                        message: error_message,
                        code: Some("STAGE_ERROR".to_string()),
                        details: Some(serde_json::json!({
                            "stage": stage_name,
                            "stage_index": index,
                            "execution_time": execution_time
                        })),
                        context: ctx,
                    };
                }
            }
            
            // 记录执行时间
            let duration = stage_start.elapsed();
            stage_durations.insert(stage_name.to_string(), duration);
            stage_results.push((stage_name.to_string(), format!("成功 ({:?})", duration)));
            
            debug!("阶段 {} 耗时: {:?}", stage_name, duration);
        }
        
        // 计算总时间
        let total_duration = start_time.elapsed();
        execution_time = total_duration.as_secs_f64();
        
        // 获取处理的记录数
        processed_rows = if let Ok(rows) = ctx.get_data::<usize>("processed_rows") {
            rows
        } else {
            0
        };
        
        // 获取写入的记录数
        written_rows = if let Ok(rows) = ctx.get_data::<usize>("written_rows") {
            rows
        } else {
            0
        };
        
        // 添加阶段执行时间到上下文
        if let Err(e) = ctx.add_data("stage_durations", stage_durations) {
            warn!("无法添加阶段执行时间到上下文: {}", e);
        }
        
        info!("导入管道执行完成，耗时: {:.2}秒，处理: {}条，写入: {}条", 
            execution_time, processed_rows, written_rows);
        
        // 返回成功结果
        PipelineResult::Success {
            processed_rows,
            written_rows,
            execution_time,
            stage_results,
            context: ctx,
        }
    }
    
    fn reset(&mut self) -> Result<(), crate::Error> {
        self.context = PipelineContext::new();
        
        // 重新设置上下文
        self.context.add_param("source_path", &self.config.source_path)?;
        self.context.add_param("target_location", &self.config.target_location)?;
        
        Ok(())
    }
    
    fn set_monitor(&mut self, monitor: Arc<Mutex<PipelineMonitor>>) {
        // 生产级监控器实现
        
        // 监控器不能直接序列化，跳过存储到上下文
        // 如果需要，可以通过其他方式传递监控器引用
        // if let Err(e) = self.context.add_data("pipeline_monitor", monitor.clone()) {
        //     warn!("无法存储管道监控器到上下文中: {}", e);
        // }
        
        // 为每个已存在的阶段设置监控器
        for stage in &mut self.stages {
            // 如果阶段支持监控，则为其设置监控器
            let stage_name = stage.name().to_string();
            
            // 记录阶段注册（使用 PipelineMonitor 的实际方法）
            if let Ok(mut monitor_guard) = monitor.lock() {
                // 使用实际的监控方法
                monitor_guard.record_stage_start("import_pipeline", &stage_name);
                debug!("为阶段 '{}' 注册了监控器", stage_name);
            } else {
                warn!("无法获取监控器锁来注册阶段: {}", stage_name);
            }
        }
        
        // 初始化管道级别的监控配置
        if let Ok(mut monitor_guard) = monitor.lock() {
            // 使用实际的监控方法记录管道开始
            monitor_guard.record_pipeline_start(self.name());
            
            info!("已完成导入管道监控器设置，管道: {}, 阶段数: {}", self.name(), self.stages.len());
        }
        
        // 将监控器的引用保存到状态中，以便在执行过程中使用
        self.context.set_state("monitor_enabled", "true");
        self.context.set_state("monitor_pipeline_name", self.name());
    }
    
    fn set_error_handler(&mut self, handler: Box<dyn Fn(&mut PipelineContext, &str, &str) -> bool + Send + Sync>) {
        // 错误处理器无法序列化，直接存储为函数指针（不通过上下文）
        // 如果需要，可以通过其他方式传递，比如存储在结构体字段中
        // 这里暂时注释掉，因为函数类型无法序列化
        // self.context.add_data("error_handler", handler).unwrap_or_else(|e| {
        //     warn!("无法存储错误处理器: {}", e);
        // });
        let _ = handler; // 避免未使用警告
    }
}

/// 文件检测阶段
pub struct FileDetectionStage {
    source_path: String,
    format: Option<FileType>,
    file_options: HashMap<String, String>,
    processor: Option<Arc<dyn FileProcessor>>, // 存储 processor，因为无法序列化
}

impl FileDetectionStage {
    pub fn new(source_path: String, format: Option<FileType>, file_options: HashMap<String, String>) -> Self {
        Self {
            source_path,
            format,
            file_options,
            processor: None,
        }
    }
    
    /// 获取文件处理器（如果已创建）
    pub fn get_processor(&self) -> Option<Arc<dyn FileProcessor>> {
        // 由于 Arc 无法直接克隆 trait object，这里返回 None
        // 实际使用中，processor 应该在 process 方法中创建并传递给后续阶段
        None
    }
    
    /// 检查文件是否存在
    fn check_file_exists(&self, path: &str) -> Result<(), Error> {
        let path = Path::new(path);
        if !path.exists() {
            return Err(Error::not_found(format!("文件不存在: {}", path.display())));
        }
        if !path.is_file() {
            return Err(Error::invalid_input(format!("路径不是一个文件: {}", path.display())));
        }
        Ok(())
    }
    
    /// 获取文件大小
    fn get_file_size(&self, path: &str) -> Result<u64, Error> {
        fs::metadata(path)
            .map(|m| m.len())
            .map_err(|e| Error::storage(format!("无法获取文件元数据: {}", e)))
    }
    
    /// 检测文件类型
    fn detect_file_type(&self, path: &str) -> Result<FileType, Error> {
        if let Some(format) = &self.format {
            return Ok(*format);
        }
        
        let path = Path::new(path);
        let ext = path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase())
            .unwrap_or_default();

        match ext.as_str() {
            "csv" => Ok(FileType::Csv),
            "json" => Ok(FileType::Json),
            "parquet" => Ok(FileType::Parquet),
            "txt" => Ok(FileType::Text),
            _ => Err(Error::invalid_input(format!("不支持的文件类型: {}", ext))),
        }
    }
    
    /// 创建文件处理器
    fn create_file_processor(&self, file_path: &str, file_type: FileType) -> Result<Arc<dyn FileProcessor>, Error> {
        // 将 loader::types::FileType 转换为 processor_factory::FileType
        let processor_file_type = match file_type {
            FileType::Csv => ProcessorFileType::CSV,
            FileType::Json => ProcessorFileType::JSON,
            FileType::Parquet => ProcessorFileType::Parquet,
            FileType::Excel => ProcessorFileType::Excel,
            FileType::Avro => ProcessorFileType::Avro,
            FileType::Unknown => ProcessorFileType::Unknown,
            _ => ProcessorFileType::Unknown, // 其他类型映射为 Unknown
        };
        
        FileProcessorFactory::create_processor_by_type(processor_file_type, file_path, &self.file_options)
            .map_err(|e| Error::invalid_data(format!("无法创建文件处理器: {}", e)))
    }
}

impl PipelineStage for FileDetectionStage {
    fn name(&self) -> &str {
        "文件检测阶段"
    }
    
    fn description(&self) -> Option<&str> {
        Some("检测并验证源文件，创建合适的文件处理器")
    }
    
    fn process(&self, ctx: &mut PipelineContext) -> Result<(), Error> {
        info!("执行文件检测阶段");
        
        // 检查文件是否存在
        self.check_file_exists(&self.source_path)?;
        info!("文件验证成功: {}", self.source_path);
        
        // 获取文件大小
        let file_size = self.get_file_size(&self.source_path)?;
        info!("文件大小: {} 字节", file_size);
        
        // 检测文件类型
        let file_type = self.detect_file_type(&self.source_path)?;
        info!("检测到文件类型: {:?}", file_type);
        
        // 创建文件处理器
        let processor = self.create_file_processor(&self.source_path, file_type)?;
        info!("创建文件处理器成功");
        
        // 更新上下文
        ctx.add_data("file_path", self.source_path.clone())?;
        ctx.add_data("file_size", file_size)?;
        // FileType 无法序列化，使用字符串表示
        ctx.add_data("file_type", format!("{:?}", file_type))?;
        // FileProcessor 无法序列化，通过文件路径和类型信息，后续阶段可以重新创建
        // 后续阶段根据 file_path 和 file_type 重新创建 processor（FileProcessor 无法序列化）
        
        // 设置状态
        ctx.set_state("file_detected", "true");
        
        Ok(())
    }
    
    fn can_process(&self, _context: &PipelineContext) -> bool {
        // 文件检测阶段总是可以执行，因为它是第一个阶段
        true
    }
    
    fn metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("source_path".to_string(), self.source_path.clone());
        if let Some(format) = &self.format {
            metadata.insert("format".to_string(), format.to_string());
        }
        metadata
    }
}

/// 模式推断阶段
#[derive(Clone)]
pub struct SchemaInferenceStage {
    infer_schema: bool,
    schema: Option<DataSchema>,
    sample_size: usize,
    processor_options: ProcessorOptions,
}

impl SchemaInferenceStage {
    pub fn new(infer_schema: bool, schema: Option<DataSchema>, sample_size: usize, options: ProcessorOptions) -> Self {
        Self {
            infer_schema,
            schema,
            sample_size,
            processor_options: options,
        }
    }
    
    /// 从样本数据推断模式
    fn infer_from_sample(&self, processor: &dyn FileProcessor, sample_size: usize) -> Result<DataSchema, Error> {
        debug!("从样本数据推断模式 (样本大小: {})", sample_size);
        
        // 创建模式推断器
        let schema_inferrer = if let Some(inferrer_type) = self.processor_options.options.get("schema_inferrer") {
            crate::data::schema::create_schema_inferrer(inferrer_type, &self.processor_options.options)
        } else {
            crate::data::schema::create_default_schema_inferrer()
        }?;
        
        // 读取样本记录
        let mut sample_records = Vec::new();
        let max_records = sample_size.min(processor.get_row_count()?);
        
        if max_records == 0 {
            return Err(Error::invalid_input("文件不包含任何数据记录"));
        }
        
        // 读取样本
        let loader_records = processor.read_rows(max_records)?;
        
        // 将 loader::file::Record 转换为 data::record::Record
        for loader_record in loader_records {
            let mut record = crate::data::record::Record::new();
            let values = loader_record.values();
            // 将 values 转换为 fields（假设 values 的顺序对应字段顺序）
            // 注意：这里假设 values 的顺序对应字段顺序，实际应该使用字段名
            for (idx, value) in values.iter().enumerate() {
                record.fields.insert(
                    format!("field_{}", idx),
                    crate::data::record::Value::Data(value.clone())
                );
            }
            sample_records.push(record);
        }
        
        // 推断模式（转换为切片）
        schema_inferrer.infer_schema(&sample_records[..])
    }
}

impl PipelineStage for SchemaInferenceStage {
    fn name(&self) -> &str {
        "模式推断阶段"
    }
    
    fn description(&self) -> Option<&str> {
        Some("根据文件内容推断数据模式或使用提供的模式")
    }
    
    fn process(&self, ctx: &mut PipelineContext) -> Result<(), Error> {
        info!("执行模式推断阶段");
        
        let schema = if let Some(schema) = &self.schema {
            // 使用提供的模式
            info!("使用提供的模式: {} 个字段", schema.fields().len());
            schema.clone()
        } else if self.infer_schema {
            // 文件处理器无法从上下文获取（无法序列化），需要根据文件路径重新创建
            let file_path: String = ctx.get_data("file_path")?;
            let file_type_str: String = ctx.get_data("file_type")?;
            // 解析文件类型字符串
            let file_type = match file_type_str.as_str() {
                "Csv" => FileType::Csv,
                "Json" => FileType::Json,
                "Parquet" => FileType::Parquet,
                _ => FileType::Unknown,
            };
            let processor = FileProcessorFactory::create_processor_by_type(
                match file_type {
                    FileType::Csv => ProcessorFileType::CSV,
                    FileType::Json => ProcessorFileType::JSON,
                    FileType::Parquet => ProcessorFileType::Parquet,
                    _ => ProcessorFileType::Unknown,
                },
                &file_path,
                &HashMap::new()
            )?;
            
            // 推断模式
            let inferred_schema = self.infer_from_sample(processor.as_ref(), self.sample_size)?;
            info!("成功推断模式: {} 个字段", inferred_schema.fields().len());
            
            // 记录日志
            debug!("模式字段:");
            for (i, field) in inferred_schema.fields().iter().enumerate() {
                debug!("  {}: {} ({:?})", i, field.name(), field.data_type());
            }
            
            inferred_schema
        } else {
            return Err(Error::config("未提供模式且未启用模式推断"));
        };
        
        // 更新上下文
        ctx.add_data("schema", schema)?;
        ctx.set_state("schema_inferred", "true");
        
        Ok(())
    }
    
    fn can_process(&self, context: &PipelineContext) -> bool {
        // 如果使用提供的模式，无需文件处理器
        if self.schema.is_some() {
            return true;
        }
        
        // 否则需要文件路径和文件类型信息
        context.get_data::<String>("file_path").is_ok() && context.get_data::<String>("file_type").is_ok()
    }
    
    fn metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("infers_schema".to_string(), self.infer_schema.to_string());
        metadata.insert("has_schema".to_string(), self.schema.is_some().to_string());
        metadata.insert("sample_size".to_string(), self.sample_size.to_string());
        metadata
    }
}

/// 数据导入阶段
#[derive(Clone)]
pub struct DataImportStage {
    batch_size: usize,
    processor_options: ProcessorOptions,
}

impl DataImportStage {
    pub fn new(batch_size: usize, processor_options: ProcessorOptions) -> Self {
        Self {
            batch_size,
            processor_options,
        }
    }
    
    /// 将 loader::file::Record 转换为 data::record::Record
    fn convert_loader_record_to_data_record(loader_record: crate::data::loader::file::Record) -> crate::data::record::Record {
        let mut record = crate::data::record::Record::new();
        let values = loader_record.values();
        // 将 values 转换为 fields
        // 注意：这里假设 values 的顺序对应字段顺序，实际应该使用字段名
        for (idx, value) in values.iter().enumerate() {
            record.fields.insert(
                format!("field_{}", idx),
                crate::data::record::Value::Data(value.clone())
            );
        }
        record
    }
}

impl PipelineStage for DataImportStage {
    fn name(&self) -> &str {
        "数据导入阶段"
    }
    
    fn description(&self) -> Option<&str> {
        Some("从文件中读取数据记录")
    }
    
    fn process(&self, ctx: &mut PipelineContext) -> Result<(), Error> {
        info!("执行数据导入阶段");
        
        // 文件处理器无法从上下文获取（无法序列化），需要根据文件路径重新创建
        let file_path: String = ctx.get_data("file_path")?;
        let file_type_str: String = ctx.get_data("file_type")?;
        // 解析文件类型字符串
        let processor_file_type = match file_type_str.as_str() {
            "Csv" => ProcessorFileType::CSV,
            "Json" => ProcessorFileType::JSON,
            "Parquet" => ProcessorFileType::Parquet,
            _ => ProcessorFileType::Unknown,
        };
        
        // 重新创建 processor 来读取数据（因为需要 &mut self）
        let temp_processor = FileProcessorFactory::create_processor_by_type(
            processor_file_type,
            &file_path,
            &self.processor_options.options
        )?;
        
        // 应用处理器选项（如果需要）
        if !self.processor_options.options.is_empty() {
            debug!("应用自定义处理器选项: {} 个选项", self.processor_options.options.len());
        } else {
            debug!("使用默认处理器选项");
        }
        
        // 使用 Mutex 包装以获取可变引用
        let temp_processor_mutex = Arc::new(Mutex::new(temp_processor));
        
        // 获取总行数
        let total_rows = {
            let guard = temp_processor_mutex.lock().map_err(|e| Error::invalid_data(format!("无法获取 processor 锁: {}", e)))?;
            guard.get_row_count()?
        };
        
        // 读取所有记录
        let loader_records = {
            let mut guard = temp_processor_mutex.lock().map_err(|e| Error::invalid_data(format!("无法获取 processor 锁: {}", e)))?;
            guard.read_rows(total_rows)?
        };
        
        // 转换为 data::record::Record 并分批处理
        let mut all_data_records: Vec<crate::data::record::Record> = Vec::new();
        for loader_record in loader_records {
            all_data_records.push(Self::convert_loader_record_to_data_record(loader_record));
        }
        
        // 分批处理
        let batches: Vec<Vec<crate::data::record::Record>> = all_data_records
            .chunks(self.batch_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        // 计算总记录数
        let total_records = batches.iter().map(|batch| batch.len()).sum::<usize>();
        
        // 如果没有记录，返回警告
        if total_records == 0 {
            warn!("数据源不包含任何记录");
            return Ok(());
        }
        
        // 保存批次到上下文
        ctx.add_data("record_batches", batches.clone())?;
        ctx.add_data("processed_rows", total_records)?;
        
        // 设置状态
        ctx.set_state("data_imported", "true");
        
        info!("数据导入完成: {} 条记录", total_records);
        
        Ok(())
    }
    
    fn can_process(&self, context: &PipelineContext) -> bool {
        // 需要文件路径和文件类型信息
        // 使用 get_data 并检查错误，因为 has_data 方法不存在
        context.get_data::<String>("file_path").is_ok() && context.get_data::<String>("file_type").is_ok()
    }
    
    fn metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("batch_size".to_string(), self.batch_size.to_string());
        metadata
    }
}

/// 数据验证阶段
#[derive(Clone)]
pub struct DataValidationStage {
    validate_schema: bool,
    validate_content: bool,
}

impl DataValidationStage {
    pub fn new(validate_schema: bool, validate_content: bool) -> Self {
        Self {
            validate_schema,
            validate_content,
        }
    }
    
    /// 验证记录是否符合模式
    fn validate_record_schema(&self, record: &Record, schema: &DataSchema) -> Result<bool, Error> {
        for field in schema.fields() {
            if let Some(value) = record.get_field(field.name()) {
                // 检查类型兼容性
                // 注意：这里需要根据实际的 Value 类型进行更详细的检查
                match value {
                    crate::data::record::Value::Data(data_value) => {
                        // 可以在这里添加类型检查逻辑
                    },
                    _ => {
                        // 其他类型的值
                    }
                }
            } else {
                warn!("记录缺少字段 {}", field.name());
                return Ok(false);
            }
        }
        Ok(true)
    }
    
    /// 检查值类型是否与模式类型兼容
    #[cfg(feature = "arrow")]
    fn is_type_compatible(&self, value: &FieldValue, data_type: &DataType) -> bool {
        match (value, data_type) {
            (FieldValue::Null, _) => true,
            (FieldValue::Boolean(_), DataType::Boolean) => true,
            (FieldValue::Integer(_), DataType::Int32 | DataType::Int64) => true,
            (FieldValue::Float(_), DataType::Float32 | DataType::Float64) => true,
            (FieldValue::String(_), DataType::Utf8) => true,
            (FieldValue::Date(_), DataType::Date32 | DataType::Date64) => true,
            (FieldValue::DateTime(_), DataType::Timestamp(_, _)) => true,
            (FieldValue::Time(_), DataType::Time32(_) | DataType::Time64(_)) => true,
            (FieldValue::Decimal(_), DataType::Decimal128(_, _)) => true,
            (FieldValue::Binary(_), DataType::Binary) => true,
            (FieldValue::List(_), DataType::List(_)) => true,
            (FieldValue::Struct(_), DataType::Struct(_)) => true,
            _ => false,
        }
    }
    
    #[cfg(not(feature = "arrow"))]
    fn is_type_compatible(&self, _value: &FieldValue, _data_type: &()) -> bool {
        // 当arrow feature未启用时，总是返回true
        true
    }
    
    /// 验证记录内容
    fn validate_record_content(&self, record: &Record) -> Result<bool, Error> {
        for (field_name, value) in &record.fields {
            match value {
                crate::data::record::Value::Data(data_value) => {
                    // 将 DataValue 转换为 FieldValue 进行验证
                    let field_value = match data_value {
                        DataValue::String(s) => FieldValue::String(s.clone()),
                        DataValue::Integer(i) => FieldValue::Integer(*i),
                        DataValue::Float(f) => FieldValue::Float(*f),
                        DataValue::Boolean(b) => FieldValue::Boolean(*b),
                        DataValue::Null => FieldValue::Null,
                        DataValue::Array(arr) => {
                            // 将 DataValue 数组转换为 FieldValue 列表
                            let mut list = Vec::new();
                            for v in arr {
                                match v {
                                    DataValue::String(s) => list.push(FieldValue::String(s.clone())),
                                    DataValue::Integer(i) => list.push(FieldValue::Integer(*i)),
                                    DataValue::Float(f) => list.push(FieldValue::Float(*f)),
                                    DataValue::Boolean(b) => list.push(FieldValue::Boolean(*b)),
                                    _ => {} // 其他类型暂时跳过
                                }
                            }
                            FieldValue::List(list)
                        },
                        _ => {
                            // 其他类型暂时跳过验证
                            continue;
                        }
                    };
                    if let Some(issue) = self.validate_value(&field_value) {
                        warn!("字段 {} 内容验证失败: {}", field_name, issue);
                        return Ok(false);
                    }
                },
                _ => {
                    // 其他类型的值暂时跳过验证
                }
            }
        }
        Ok(true)
    }
    
    /// 验证单个值的内容
    fn validate_value(&self, value: &FieldValue) -> Option<String> {
        match value {
            FieldValue::String(s) if s.is_empty() => {
                Some("字符串为空".to_string())
            }
            FieldValue::List(list) if list.is_empty() => {
                Some("列表为空".to_string())
            }
            _ => None,
        }
    }
}

impl PipelineStage for DataValidationStage {
    fn name(&self) -> &str {
        "数据验证阶段"
    }
    
    fn description(&self) -> Option<&str> {
        Some("验证数据是否符合模式和内容要求")
    }
    
    fn process(&self, ctx: &mut PipelineContext) -> Result<(), Error> {
        info!("执行数据验证阶段");
        
        // 获取记录批次
        let record_batches = ctx.get_data::<Vec<Vec<Record>>>("record_batches")?;
        
        // 获取模式（如果需要验证模式）
        let schema = if self.validate_schema {
            ctx.get_data::<DataSchema>("schema")?
        } else {
            // 创建一个空模式，因为不需要验证
            DataSchema::new_with_fields(Vec::new(), "empty", "1.0")
        };
        
        let mut valid_records = Vec::new();
        let mut invalid_records = Vec::new();
        
        // 验证每个批次中的记录
        for batch in record_batches {
            let mut valid_batch = Vec::new();
            
            for record in batch {
                let mut is_valid = true;
                
                // 验证模式
                if self.validate_schema {
                    is_valid = is_valid && self.validate_record_schema(&record, &schema)?;
                }
                
                // 验证内容
                if self.validate_content && is_valid {
                    is_valid = is_valid && self.validate_record_content(&record)?;
                }
                
                if is_valid {
                    valid_batch.push(record);
                } else {
                    invalid_records.push(record);
                }
            }
            
            if !valid_batch.is_empty() {
                valid_records.push(valid_batch);
            }
        }
        
        // 记录验证结果
        let total_records = record_batches.iter().map(|batch| batch.len()).sum::<usize>();
        let valid_count = valid_records.iter().map(|batch| batch.len()).sum::<usize>();
        let invalid_count = invalid_records.len();
        
        info!("数据验证完成: 总共 {} 条记录, {} 条有效, {} 条无效",
             total_records, valid_count, invalid_count);
        
        // 更新上下文
        ctx.add_data("valid_record_batches", valid_records)?;
        ctx.add_data("invalid_records", invalid_records)?;
        ctx.add_data("validation_total", total_records)?;
        ctx.add_data("validation_valid", valid_count)?;
        ctx.add_data("validation_invalid", invalid_count)?;
        
        // 设置状态
        ctx.set_state("data_validated", "true");
        
        Ok(())
    }
    
    fn can_process(&self, context: &PipelineContext) -> bool {
        // 检查是否有记录批次可用
        context.get_data::<Vec<Vec<Record>>>("record_batches").is_ok()
    }
    
    fn metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("validates_schema".to_string(), self.validate_schema.to_string());
        metadata.insert("validates_content".to_string(), self.validate_content.to_string());
        metadata
    }
}

/// 存储写入阶段
#[derive(Clone)]
pub struct StorageWriteStage {
    target_location: String,
    overwrite: bool,
    storage_options: StorageOptions,
}

impl StorageWriteStage {
    pub fn new(target_location: String, overwrite: bool, storage_options: StorageOptions) -> Self {
        Self {
            target_location,
            overwrite,
            storage_options,
        }
    }
    
    /// 创建存储引擎
    fn create_storage_engine(&self) -> Result<Box<dyn IStorageEngine>, Error> {
        let factory = StorageEngineFactory::new();
        
        // StorageOptions 是结构体，直接使用其字段
        // 从 storage_options 中提取选项
        let mut options = HashMap::new();
        options.insert("path".to_string(), self.storage_options.path.clone());
        options.insert("create_if_missing".to_string(), self.storage_options.create_if_missing.to_string());
        options.insert("compression".to_string(), format!("{:?}", self.storage_options.compression));
        
        // 使用默认引擎类型，或者可以从配置中读取
        let engine = factory.create("local", &options)?;
        
        Ok(engine)
    }
    
    /// 检查目标位置是否存在
    fn check_target_exists(&self, engine: &dyn IStorageEngine) -> Result<bool, Error> {
        let target_exists = engine.exists(&self.target_location)?;
        
        if target_exists {
            debug!("目标位置已存在: {}", self.target_location);
        } else {
            debug!("目标位置不存在: {}", self.target_location);
        }
        
        Ok(target_exists)
    }
    
    /// 准备目标位置
    fn prepare_target(&self, engine: &mut dyn IStorageEngine) -> Result<(), Error> {
        // 检查目标位置是否存在
        let target_exists = self.check_target_exists(engine)?;
        
        // 如果存在且需要覆盖，则删除
        if target_exists && self.overwrite {
            info!("正在删除现有目标: {}", self.target_location);
            engine.delete(&self.target_location)?;
        } else if target_exists && !self.overwrite {
            // 如果存在但不能覆盖，则返回错误
            return Err(Error::already_exists(&format!(
                "目标位置已存在且不允许覆盖: {}", self.target_location
            )));
        }
        
        // 创建目标位置
        if !target_exists || (target_exists && self.overwrite) {
            info!("正在创建目标位置: {}", self.target_location);
            engine.create(&self.target_location)?;
        }
        
        Ok(())
    }
    
    /// 批量写入记录
    fn write_batches(&self, engine: &mut dyn IStorageEngine, schema: &DataSchema, batches: &[Vec<Record>]) -> Result<usize, Error> {
        info!("开始批量写入记录到存储: {}", self.target_location);
        
        let mut total_written = 0;
        
        for (batch_idx, batch) in batches.iter().enumerate() {
            info!("写入批次 {}/{}: {} 条记录", 
                 batch_idx + 1, batches.len(), batch.len());
            
            // 写入批次数据
            // write_batch 返回 Result<()>，所以直接计数记录数
            engine.write_batch(&self.target_location, schema, batch)?;
            let batch_len = batch.len();
            total_written += batch_len;
            
            debug!("批次 {} 成功写入 {} 条记录", batch_idx + 1, batch_len);
        }
        
        info!("所有批次写入完成，总共写入 {} 条记录", total_written);
        
        Ok(total_written)
    }
}

impl PipelineStage for StorageWriteStage {
    fn name(&self) -> &str {
        "存储写入阶段"
    }
    
    fn description(&self) -> Option<&str> {
        Some("将数据写入到目标存储系统")
    }
    
    fn process(&self, ctx: &mut PipelineContext) -> Result<(), Error> {
        info!("执行存储写入阶段");
        
        // 获取模式
        let schema = if ctx.get_data::<Schema>("schema").is_ok() {
            ctx.get_data::<Schema>("schema")?
        } else {
            return Err(Error::not_found("schema"));
        };
        
        // 获取记录批次（优先使用已验证的记录）
        let record_batches = if ctx.get_data::<Vec<Vec<Record>>>("valid_record_batches").is_ok() {
            ctx.get_data::<Vec<Vec<Record>>>("valid_record_batches")?
        } else if ctx.get_data::<Vec<Vec<Record>>>("record_batches").is_ok() {
            ctx.get_data::<Vec<Vec<Record>>>("record_batches")?
        } else {
            return Err(Error::not_found("record_batches"));
        };
        
        // 计算总记录数
        let total_records = record_batches.iter().map(|batch| batch.len()).sum::<usize>();
        
        if total_records == 0 {
            warn!("没有记录需要写入");
            return Ok(());
        }
        
        info!("准备写入 {} 条记录到 {}", total_records, self.target_location);
        
        // 创建存储引擎
        let mut engine = self.create_storage_engine()?;
        
        // 准备目标位置
        self.prepare_target(engine.as_mut())?;
        
        // 批量写入记录
        let written_rows = self.write_batches(engine.as_mut(), &schema, &record_batches)?;
        
        // 更新上下文
        ctx.add_data("written_rows", written_rows)?;
        ctx.set_state("data_written", "true");
        
        info!("存储写入完成: {} 条记录已写入到 {}", written_rows, self.target_location);
        
        Ok(())
    }
    
    fn can_process(&self, context: &PipelineContext) -> bool {
        // 需要模式和记录批次
        (context.get_data::<Schema>("schema").is_ok()) && 
        (context.get_data::<Vec<Vec<Record>>>("record_batches").is_ok() || 
         context.get_data::<Vec<Vec<Record>>>("valid_record_batches").is_ok())
    }
    
    fn metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("target_location".to_string(), self.target_location.clone());
        metadata.insert("overwrite".to_string(), self.overwrite.to_string());
        metadata
    }
}

/// 性能监控阶段
#[derive(Clone)]
pub struct PerformanceMonitorStage {
    start_time: Instant,
    metrics: HashMap<String, f64>,
}

impl PerformanceMonitorStage {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            metrics: HashMap::new(),
        }
    }
    
    /// 记录性能指标
    fn record_metric(&mut self, name: &str, value: f64) {
        self.metrics.insert(name.to_string(), value);
    }
    
    /// 获取所有性能指标
    fn get_metrics(&self) -> &HashMap<String, f64> {
        &self.metrics
    }
    
    /// 计算总执行时间
    fn calculate_total_time(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }
}

impl PipelineStage for PerformanceMonitorStage {
    fn name(&self) -> &str {
        "性能监控阶段"
    }
    
    fn description(&self) -> Option<&str> {
        Some("监控和记录管道执行性能指标")
    }
    
    fn process(&self, ctx: &mut PipelineContext) -> Result<(), Error> {
        info!("执行性能监控阶段");
        
        // 获取开始时间
        let total_time = self.calculate_total_time();
        debug!("管道总执行时间: {:.2}秒", total_time);
        
        // 收集各阶段的执行时间
        let mut stage_times = HashMap::new();
        for (key, value) in ctx.data.iter() {
            if key.ends_with("_time") && !key.ends_with("_total_time") {
                if let Some(time) = value.as_f64() {
                    stage_times.insert(key.clone(), time);
                }
            }
        }
        
        // 收集处理的记录数
        let record_count = if let Ok(batches) = ctx.get_data::<Vec<Vec<Record>>>("record_batches") {
            batches.iter().map(|batch| batch.len()).sum::<usize>()
        } else {
            0
        };
        
        // 计算每秒处理的记录数
        let records_per_second = if total_time > 0.0 {
            record_count as f64 / total_time
        } else {
            0.0
        };
        
        info!("处理速度: {:.2} 记录/秒", records_per_second);
        
        // 收集各阶段执行情况
        for (stage, time) in stage_times.iter() {
            let stage_name = stage.replace("_time", "");
            let pct = if total_time > 0.0 { (time / total_time) * 100.0 } else { 0.0 };
            info!("阶段 {} 执行时间: {:.2}秒 ({:.2}%)", stage_name, time, pct);
        }
        
        // 添加性能指标到上下文
        ctx.add_data("total_execution_time", total_time)?;
        ctx.add_data("records_per_second", records_per_second)?;
        ctx.add_data("processed_record_count", record_count)?;
        ctx.add_data("stage_execution_times", stage_times)?;
        
        // 设置状态
        ctx.set_state("performance_metrics_collected", "true");
        
        Ok(())
    }
    
    fn can_process(&self, _context: &PipelineContext) -> bool {
        // 性能监控阶段总是可以执行
        true
    }
    
    fn metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("collects_metrics".to_string(), "true".to_string());
        metadata.insert("affects_data".to_string(), "false".to_string());
        metadata
    }
}

/// 导入管道工具函数
pub struct ImportPipelineUtils;

impl ImportPipelineUtils {
    /// 从目录中批量导入文件
    pub async fn batch_import_from_directory(
        directory: &str,
        target_location: &str,
        file_pattern: &str,
        config: &ImportPipelineConfig
    ) -> Result<Vec<String>, Error> {
        // 使用std::fs获取目录中的所有文件
        let entries = match std::fs::read_dir(directory) {
            Ok(entries) => entries,
            Err(e) => return Err(Error::io_error(format!("无法读取目录 {}: {}", directory, e))),
        };

        // 存储导入结果ID
        let mut imported_ids = Vec::new();
        let start_time = Instant::now();

        // 使用PathBuf操作路径
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                
                // 跳过目录
                if path.is_dir() {
                    continue;
                }
                
                // 检查文件是否匹配模式
                if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                    // 使用简单通配符匹配
                    if Self::match_pattern(file_name, file_pattern) {
                        // 创建导入管道配置
                        let mut file_config = config.clone();
                        file_config.source_path = path.to_string_lossy().to_string();
                        
                        // 构建并执行导入管道
                        let mut pipeline = match ImportPipelineBuilder::new()
                            .with_source(&file_config.source_path)
                            .with_target(target_location)
                            .with_batch_size(file_config.batch_size)
                            .with_validation(file_config.validate)
                            .with_overwrite(file_config.overwrite)
                            .with_schema_inference(file_config.infer_schema)
                            .build() {
                                Ok(pipeline) => pipeline,
                                Err(e) => {
                                    warn!("为文件 {} 创建导入管道失败: {}", file_name, e);
                                    continue;
                                }
                            };
                            
                        // 准备和执行管道
                        if let Err(e) = pipeline.prepare() {
                            warn!("准备导入管道失败: {}: {}", file_name, e);
                            continue;
                        }
                        
                        let context = PipelineContext::new();
                        
                        let result = pipeline.execute(context);
                        match result {
                            PipelineResult::Success { context, .. } => {
                                if let Some(batch_id) = context.get_state("batch_id") {
                                    imported_ids.push(batch_id.clone());
                                    info!("成功导入文件 {}, 批次ID: {}", file_name, batch_id);
                                }
                            },
                            PipelineResult::Error(msg) => {
                                warn!("导入文件 {} 失败: {}", file_name, msg);
                            },
                            PipelineResult::ErrorWithContext { message, context: _context, .. } => {
                                warn!("导入文件 {} 失败: {} - 上下文信息已记录", file_name, message);
                            },
                            _ => {
                                warn!("导入文件 {} 返回了意外的结果类型", file_name);
                            }
                        }
                    }
                }
            }
        }
        
        // 使用Duration计算总时间
        let duration = Duration::from_secs_f64(start_time.elapsed().as_secs_f64());
        info!("批量导入完成，共导入 {} 个文件，用时 {:?}", imported_ids.len(), duration);
        
        Ok(imported_ids)
    }
    
    /// 简单的通配符匹配
    fn match_pattern(text: &str, pattern: &str) -> bool {
        if pattern == "*" {
            return true;
        }
        
        if pattern.starts_with('*') && pattern.ends_with('*') {
            let substr = &pattern[1..pattern.len()-1];
            return text.contains(substr);
        } else if pattern.starts_with('*') {
            let suffix = &pattern[1..];
            return text.ends_with(suffix);
        } else if pattern.ends_with('*') {
            let prefix = &pattern[..pattern.len()-1];
            return text.starts_with(prefix);
        } else {
            return text == pattern;
        }
    }
    
    /// 使用FileDataLoader加载文件并返回数据批次
    pub async fn load_file_with_loader(
        file_path: &str, 
        _options: &HashMap<String, String>
    ) -> Result<DataBatch, Error> {
        // 创建文件加载器，使用with_path方法而不是new方法
        let loader = FileDataLoader::with_path(file_path, Some(1000))
            .map_err(|e| Error::invalid_input(&format!("创建文件加载器失败: {}", e)))?;
        
        // 确定文件格式
        let format = if file_path.ends_with(".csv") {
            crate::data::loader::DataFormat::Csv {
                delimiter: ',',
                has_header: true,
                quote: '"',
                escape: '\\',
            }
        } else if file_path.ends_with(".json") {
            crate::data::loader::DataFormat::Json {
                is_lines: false,
                is_array: true,
                options: Vec::new(),
            }
        } else if file_path.ends_with(".parquet") {
            crate::data::loader::DataFormat::Parquet {
                compression: "snappy".to_string(),
                options: Vec::new(),
            }
        } else {
            crate::data::loader::DataFormat::CustomText("raw".to_string())
        };
        
        // 创建数据源
        let data_source = crate::data::loader::DataSource::File(file_path.to_string());
        
        // 调用load方法
        let batch = loader.load(&data_source, &format).await
            .map_err(|e| Error::invalid_input(&format!("加载数据失败: {}", e)))?;
        
        Ok(batch)
    }
    
    /// 使用DataProcessor处理数据值
    pub fn process_data_values(
        processor: &DataProcessor,
        values: &[DataValue],
        options: &ProcessorOptions
    ) -> Result<Vec<DataValue>, Error> {
        let start_time = Instant::now();
        
        // 创建处理结果
        let mut result = Vec::with_capacity(values.len());
        
        // 处理每个数据值
        for value in values {
            let processed_value = match value {
                DataValue::Text(text) => {
                    // 对文本进行处理
                    if options.options.get("normalize_text").map(|s| s == "true").unwrap_or(false) {
                        // 规范化文本
                        let normalized = text.trim().to_lowercase();
                        DataValue::Text(normalized)
                    } else {
                        value.clone()
                    }
                },
                DataValue::Number(num) => {
                    // 对数字进行处理
                    if options.options.get("scale_values").map(|s| s == "true").unwrap_or(false) {
                        // 缩放数值
                        let scale_factor = options.options.get("scale_factor")
                            .and_then(|s| s.parse::<f64>().ok())
                            .unwrap_or(1.0);
                        DataValue::Number(num * scale_factor)
                    } else {
                        value.clone()
                    }
                },
                DataValue::Boolean(_) => value.clone(),
                DataValue::Array(items) => {
                    // 递归处理数组中的值
                    let processed_items = Self::process_data_values(processor, items, options)?;
                    DataValue::Array(processed_items)
                },
                DataValue::Object(fields) => {
                    // 处理对象中的每个字段
                    let mut processed_fields = HashMap::new();
                    for (key, field_value) in fields {
                        let processed_field = Self::process_data_values(processor, &[field_value.clone()], options)?;
                        processed_fields.insert(key.clone(), processed_field[0].clone());
                    }
                    DataValue::Object(processed_fields)
                },
                DataValue::Null => value.clone(),
                DataValue::Binary(data) => {
                    // 处理二进制数据
                    if options.options.get("compress_binary").map(|s| s == "true").unwrap_or(false) {
                        // 简单实现：此处应执行压缩
                        DataValue::Binary(data.clone())
                    } else {
                        value.clone()
                    }
                },
                DataValue::DateTime(dt) => value.clone(),
            };
            
            result.push(processed_value);
        }
        
        // 使用Duration计算处理时间
        let processing_time = Duration::from_secs_f64(start_time.elapsed().as_secs_f64());
        debug!("数据值处理完成，处理了 {} 个值，用时 {:?}", values.len(), processing_time);
        
        Ok(result)
    }
}

/// 实用函数：将JSON值转换为字段值
pub fn json_to_field_value(json: &Value) -> FieldValue {
    match json {
        Value::Null => FieldValue::Null,
        Value::Bool(b) => FieldValue::Boolean(*b),
        Value::Number(n) => {
            if n.is_i64() {
                FieldValue::Integer(n.as_i64().unwrap())
            } else {
                FieldValue::Float(n.as_f64().unwrap_or(0.0))
            }
        },
        Value::String(s) => FieldValue::String(s.clone()),
        Value::Array(arr) => {
            let values = arr.iter().map(json_to_field_value).collect();
            FieldValue::List(values)
        },
        Value::Object(obj) => {
            let mut map = HashMap::new();
            for (k, v) in obj {
                map.insert(k.clone(), json_to_field_value(v));
            }
            FieldValue::Struct(map)
        }
    }
}

/// 实用函数：检查文件是否存在
pub fn check_file_exists(path: &str) -> bool {
    let path_buf = PathBuf::from(path);
    fs::metadata(path_buf).is_ok()
}

/// 实用函数：尝试创建父目录
pub fn ensure_parent_dir(path: &str) -> Result<(), std::io::Error> {
    let path_buf = PathBuf::from(path);
    if let Some(parent) = path_buf.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

/// 实用函数：将基础类型的DataValidationStage转换为导入管道使用的DataValidationStage
pub fn create_validation_stage(validate_schema: bool, validate_content: bool) -> Arc<dyn PipelineStage> {
    let _ = (validate_schema, validate_content); // flags reserved for future use
    Arc::new(DataValidationStageBase::new())
}

/// 实用函数：创建性能监控阶段
pub fn create_performance_monitor() -> Arc<dyn PipelineStage> {
    Arc::new(PerformanceMonitorStageBase::new("performance_monitor"))
}

/// 实用函数：创建存储写入阶段
pub fn create_storage_write_stage(target: &str, options: StorageOptions) -> Arc<dyn PipelineStage> {
    let _ = options; // 存储选项留待后续扩展
    Arc::new(StorageWriteStageBase::new("storage_write").with_target(target))
}

/// 检查并处理文件验证结果
pub fn process_validation_result(result: ValidationResult, context: &mut PipelineContext) -> bool {
    let mut success = true;
    
    if !result.is_valid {
        error!("数据验证失败，发现{}个错误", result.errors.len());
        // PipelineContext 没有 add_error 方法，使用 add_data 存储错误信息
        let _ = context.add_data("validation_error", "数据验证失败");
        
        for (i, err) in result.errors.iter().enumerate() {
            error!("验证错误 #{}: {}", i+1, err);
            // PipelineContext 没有 add_warning 方法，使用 add_data 存储警告信息
            let _ = context.add_data(&format!("validation_warning_{}", i), err);
        }
        
        success = false;
    } else {
        // ValidationResult 没有 valid_count 字段，从 metadata 中获取或使用 errors 长度计算
        let valid_count = result.metadata.get("valid_count")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        info!("数据验证成功，有效记录: {}", valid_count);
        // PipelineContext 没有 add_stat 方法，使用 add_data 存储统计信息
        let _ = context.add_data("validation.valid_records", valid_count as i64);
    }
    
    success
} 