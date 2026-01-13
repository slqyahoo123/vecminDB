//! 数据处理管道模块
//!
//! 本模块提供了构建数据处理流水线的组件，用于数据转换、清洗和验证。

// 导入子模块
mod core;
mod stages;
mod validators;
pub mod import_pipeline;
pub mod pipeline_registry;
pub mod traits;
pub mod monitor;
#[cfg(feature = "examples")]
pub mod examples;
pub mod validation;
pub mod file_detection;
pub mod schema_inference;
pub mod storage_writer;
pub mod performance;
pub mod pipeline;
pub mod record_batch;
pub mod parallel_processor;

#[cfg(test)]
mod tests;

// 重新导出核心组件
pub use self::core::{
    DataStage, DataValidator, DataTransformer, 
    ValidationError, ValidationWarning,
    PipelineConfig, DataPipeline, DataProcessor
};
pub use crate::core::interfaces::ValidationResult;

// 重新导出并发处理器
pub use self::parallel_processor::{
    ParallelProcessor, ParallelProcessorBuilder, ParallelConfig, ParallelStrategy,
    ProcessingTask, ProcessingResult,
    ParallelProcessingMetrics,
};

// 从performance模块正确导入
pub use self::performance::PipelinePerformanceMetrics;

// 导出示例函数
#[cfg(feature = "examples")]
pub use self::examples::{
    run_monitoring_example,
    run_measure_time_example,
    run_examples,
    run_all_examples
};

// 重新导出数据处理阶段
pub use self::stages::{
    TypeConverterStage, MissingValueHandlerStage, MissingValueStrategy,
    NormalizerStage, NormalizationMethod
};

// 重新导出数据验证器
pub use self::validators::{
    TypeValidator, RangeValidator, NotNullValidator, UniquenessValidator
};

pub use pipeline_registry::{PipelineRegistry, get_registry};

// 从import_pipeline模块导出组件并创建必要的别名
pub use import_pipeline::{
    ImportPipeline, ImportPipelineConfig, ImportPipelineBuilder,
    // 导出FileDetectionStage和SchemaInferenceStage并创建对应别名
    FileDetectionStage as FileDetectionStageImpl, 
    SchemaInferenceStage as SchemaInferenceStageImpl,
    DataImportStage as DataImportStageImpl,
    DataValidationStage as DataValidationStageImpl,
    StorageWriteStage as StorageWriteStageImpl,
    PerformanceMonitorStage as PerformanceMonitorStageImpl
};

// 导出新的特征定义
pub use traits::{
    ValidationRule as TraitsValidationRule,
    ValidationErrorStrategy as TraitsValidationErrorStrategy,
    PipelineStage as TraitsPipelineStage,
    PipelineStageStatus,
    PipelineContext as TraitsPipelineContext,
    DataProcessor as TraitsDataProcessor,
    PipelineMonitor as TraitsPipelineMonitor
};

// 导出高级性能监控组件
pub use monitor::{
    AdvancedPerformanceMonitorStage,
    ResourceMetrics,
    MonitoringTools,
    measure_time,
    create_monitoring_tools,
    create_performance_monitor
};

pub use self::validation::{
    DataValidationStage as ValidationDataValidationStage, 
    ValidationRule, 
    ValidationType
};
pub use crate::core::interfaces::ValidationResult as ValidationResultType;

pub use self::file_detection::FileDetectionStage;
pub use self::schema_inference::SchemaInferenceStage;
pub use self::storage_writer::StorageWriteStage as WriterStorageWriteStage;
pub use self::performance::{
    PerformanceMonitorStage as MetricsPerformanceMonitorStage
};

pub use self::pipeline::{
    Pipeline, PipelineStage, PipelineContext, PipelineResult,
    BasicPipeline
};

pub use self::record_batch::RecordBatch;

use crate::error::Result;
use std::collections::HashMap;

/// 创建默认的数据处理管道
pub fn create_default_pipeline() -> DataPipeline {
    let config = PipelineConfig {
        id: "default_pipeline".to_string(),
        name: "default_pipeline".to_string(),
        description: Some("默认数据处理管道".to_string()),
        stages: Vec::new(),
        custom_config: {
            let mut map = HashMap::new();
            map.insert("batch_size".to_string(), "1000".to_string());
            map.insert("parallel".to_string(), "false".to_string());
            map.insert("enable_validation".to_string(), "true".to_string());
            map
        },
    };
    
    DataPipeline::new(config)
}

/// 创建类型转换处理管道
pub fn create_type_conversion_pipeline() -> DataPipeline {
    let mut pipeline = create_default_pipeline();
    
    // 添加类型验证器
    let mut type_validator = TypeValidator::new("type_validator");
    type_validator.check_all_fields(true);
    pipeline.add_validator(type_validator);
    
    // 添加类型转换阶段
    let mut type_converter = TypeConverterStage::new("type_converter");
    type_converter.with_skip_errors(true);
    pipeline.add_stage(type_converter);
    
    pipeline
}

/// 创建数据清洗处理管道
pub fn create_data_cleaning_pipeline() -> DataPipeline {
    let mut pipeline = create_default_pipeline();
    
    // 添加非空验证器
    let not_null_validator = NotNullValidator::new("not_null_validator");
    pipeline.add_validator(not_null_validator);
    
    // 添加缺失值处理阶段
    let missing_handler = MissingValueHandlerStage::new("missing_value_handler");
    pipeline.add_stage(missing_handler);
    
    pipeline
}

/// 创建数据标准化处理管道
pub fn create_normalization_pipeline() -> DataPipeline {
    let mut pipeline = create_default_pipeline();
    
    // 添加数据范围验证器
    let range_validator = RangeValidator::new("range_validator");
    pipeline.add_validator(range_validator);
    
    // 添加标准化阶段
    let normalizer = NormalizerStage::new("normalizer", NormalizationMethod::ZScore);
    pipeline.add_stage(normalizer);
    
    pipeline
}

// 注意: 下面的定义在pipeline模块中已经存在，我们使用类型别名来解决命名冲突，
// 同时保持向后兼容性，确保现有代码可以继续使用这些定义

/// 管道上下文类型别名，使用pipeline模块中定义的PipelineContext
pub type PipelineContextCompat = self::pipeline::PipelineContext;

/// 管道阶段类型别名，使用pipeline模块中定义的PipelineStage
pub type PipelineStageCompat = dyn self::pipeline::PipelineStage + Send + Sync;

/// 管道结果类型别名，使用pipeline模块中定义的PipelineResult
pub type PipelineResultCompat = self::pipeline::PipelineResult;

/// 管道接口类型别名，使用pipeline模块中定义的Pipeline
pub type PipelineCompat = dyn self::pipeline::Pipeline + Send + Sync;

