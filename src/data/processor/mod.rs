// Data Processor Module
// 数据处理器模块

// 重新导出核心模块
pub mod types_core;
pub mod utils;
pub mod schema_ops;
pub mod data_ops;
pub mod record_ops;
pub mod processor_impl;

// 保留原有模块
pub mod config;
pub mod types;
pub mod core;
pub mod processors;
pub mod traits;

// 新增的模块
pub mod transformers;

// 重新导出主要类型和结构
pub use processor_impl::{DataProcessor, ImportStats, ExportStats};
pub use types_core::{
    ProcessorState, ProcessorStatus, ProcessorType, 
    TaskHandle, ProcessorMetrics, MemoryInfo,
    Processor, StorageHealthCheck
};
pub use schema_ops::{
    extract_schema_from_metadata, infer_schema_from_data, 
    SchemaMerger, MergeStrategy, compare_schemas, generate_schema_statistics
};
pub use data_ops::{
    DataParser, DataConverter, DataValidator, DataCleaner, DataStatistics
};
pub use record_ops::{
    RecordProcessor, FeatureExtractor, ValidationRule, RecordValueType,
    RecordTransformer, Transformation
};

// 导出缺失的重要类型
pub use config::{ProcessorConfig, TransformerConfig};
pub use types::{DataType, ColumnInfo, ProcessorDataset, ProcessorBatch};

// 为向后兼容性提供类型别名
pub type ProcessedDataset = ProcessorDataset;
pub type ProcessedBatch = ProcessorBatch;

// 确保transformers模块被包含
pub use transformers::{
    NumericTransformer, CategoricalTransformer, DateTimeTransformer, TransformerType
};

// 重新导出工具函数
pub use utils::{
    file_exists, ensure_dir_exists, read_file, write_file,
    get_file_size, is_directory, is_file, get_file_extension,
    join_path, create_temp_filename, safe_delete_file, safe_delete_dir,
    copy_file, move_file, list_files, calculate_file_hash, check_disk_space
};

// 添加缺失的处理函数
pub use processors::{
    normalize::normalize,
    tokenize::tokenize,
    encode::encode,
    transform::transform,
    filter::filter,
    augment::augment
};

// 重新导出原有的核心功能（为了向后兼容）
pub use crate::data::pipeline::traits::DataProcessor as TraitsDataProcessor;

// 公共API - 主要功能的便捷接口
use crate::Result;
use crate::data::DataFormat;

/// 创建数据处理器的便捷函数
pub fn create_processor(
    id: String, 
    name: String, 
    format: DataFormat, 
    config: config::ProcessorConfig
) -> DataProcessor {
    DataProcessor::new(id, name, format, config)
}

/// 快速导入数据的便捷函数
pub async fn quick_import(
    source: &str,
    destination: &str,
    format: DataFormat,
) -> Result<ImportStats> {
    let config = config::ProcessorConfig::default();
    let processor = create_processor(
        "quick_import".to_string(),
        "Quick Import Processor".to_string(),
        format,
        config.clone()
    );
    
    processor.import(source, destination, &config).await
}

/// 快速导出数据的便捷函数
pub async fn quick_export(
    source: &str,
    destination: &str,
    format: DataFormat,
) -> Result<ExportStats> {
    let config = config::ProcessorConfig::default();
    let format_clone = format.clone();
    let processor = create_processor(
        "quick_export".to_string(),
        "Quick Export Processor".to_string(),
        format,
        config
    );
    
    processor.export(source, destination, format_clone).await
}

/// 快速解析数据的便捷函数
pub fn quick_parse(data: &[u8], format: DataFormat) -> Result<Vec<crate::data::record::Record>> {
    let parser = DataParser::new(format);
    let config = config::ProcessorConfig::default();
    parser.parse_data(data, &config)
}

/// 快速验证数据格式的便捷函数
pub fn quick_validate(data: &[u8], format: DataFormat) -> Result<Vec<u8>> {
    let validator = DataValidator::new(format);
    validator.validate_and_fix_format(data)
}

/// 快速清理数据的便捷函数
pub fn quick_clean(data: &[u8]) -> Result<Vec<u8>> {
    let cleaner = DataCleaner::new();
    cleaner.clean_data(data)
}

/// 快速统计数据的便捷函数
pub fn quick_stats(records: &[crate::data::record::Record]) -> std::collections::HashMap<String, serde_json::Value> {
    let statistics = DataStatistics::new();
    statistics.calculate_basic_stats(records)
}

/// 处理器工厂
pub struct ProcessorFactory;

impl ProcessorFactory {
    /// 创建默认的文件处理器
    pub fn create_file_processor(format: DataFormat) -> DataProcessor {
        let config = config::ProcessorConfig::default();
        DataProcessor::new(
            format!("file_processor_{:?}", format),
            format!("File Processor for {:?}", format),
            format,
            config
        )
    }
    
    /// 创建批处理器
    pub fn create_batch_processor(format: DataFormat, batch_size: usize) -> DataProcessor {
        let mut config = config::ProcessorConfig::default();
        config.batch_size = batch_size;
        
        DataProcessor::new(
            format!("batch_processor_{:?}_{}", format, batch_size),
            format!("Batch Processor for {:?} (batch_size: {})", format, batch_size),
            format,
            config
        )
    }
    
    /// 创建流处理器
    pub fn create_stream_processor(format: DataFormat) -> DataProcessor {
        let mut config = config::ProcessorConfig::default();
        // 标记流模式可用
        config.process_type = "stream".to_string();
        
        DataProcessor::new(
            format!("stream_processor_{:?}", format),
            format!("Stream Processor for {:?}", format),
            format,
            config
        )
    }
}

/// 处理器管理器
pub struct ProcessorManager {
    processors: std::collections::HashMap<String, DataProcessor>,
}

impl ProcessorManager {
    /// 创建新的处理器管理器
    pub fn new() -> Self {
        Self {
            processors: std::collections::HashMap::new(),
        }
    }
    
    /// 注册处理器
    pub fn register_processor(&mut self, processor: DataProcessor) {
        let id = processor.id().to_string();
        self.processors.insert(id, processor);
    }
    
    /// 获取处理器
    pub fn get_processor(&self, id: &str) -> Option<&DataProcessor> {
        self.processors.get(id)
    }
    
    /// 获取可变处理器引用
    pub fn get_processor_mut(&mut self, id: &str) -> Option<&mut DataProcessor> {
        self.processors.get_mut(id)
    }
    
    /// 移除处理器
    pub fn remove_processor(&mut self, id: &str) -> Option<DataProcessor> {
        self.processors.remove(id)
    }
    
    /// 列出所有处理器ID
    pub fn list_processors(&self) -> Vec<String> {
        self.processors.keys().cloned().collect()
    }
    
    /// 获取处理器数量
    pub fn processor_count(&self) -> usize {
        self.processors.len()
    }
    
    /// 清理所有处理器
    pub fn clear(&mut self) {
        self.processors.clear();
    }
    
    /// 健康检查所有处理器
    pub fn health_check_all(&self) -> Result<std::collections::HashMap<String, bool>> {
        let mut results = std::collections::HashMap::new();
        
        for (id, processor) in &self.processors {
            match processor.health_check() {
                Ok(healthy) => {
                    results.insert(id.clone(), healthy);
                },
                Err(e) => {
                    log::error!("处理器 {} 健康检查失败: {}", id, e);
                    results.insert(id.clone(), false);
                }
            }
        }
        
        Ok(results)
    }
}

/// 数据处理管道
pub struct ProcessingPipeline {
    steps: Vec<Box<dyn PipelineStep>>,
}

impl ProcessingPipeline {
    /// 创建新的处理管道
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
        }
    }
    
    /// 添加处理步骤
    pub fn add_step(&mut self, step: Box<dyn PipelineStep>) {
        self.steps.push(step);
    }
    
    /// 执行管道
    pub async fn execute(&self, mut data: ProcessingData) -> Result<ProcessingData> {
        for (i, step) in self.steps.iter().enumerate() {
            log::debug!("执行管道步骤 {}: {}", i, step.name());
            data = step.process(data).await?;
        }
        Ok(data)
    }
    
    /// 获取步骤数量
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
}

/// 管道步骤trait
#[async_trait::async_trait]
pub trait PipelineStep: Send + Sync {
    /// 步骤名称
    fn name(&self) -> &str;
    
    /// 处理数据
    async fn process(&self, data: ProcessingData) -> Result<ProcessingData>;
}

/// 处理数据容器
#[derive(Debug, Clone)]
pub struct ProcessingData {
    pub records: Vec<crate::data::record::Record>,
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
    pub format: DataFormat,
}

impl ProcessingData {
    /// 创建新的处理数据
    pub fn new(records: Vec<crate::data::record::Record>, format: DataFormat) -> Self {
        Self {
            records,
            metadata: std::collections::HashMap::new(),
            format,
        }
    }
    
    /// 添加元数据
    pub fn add_metadata(&mut self, key: String, value: serde_json::Value) {
        self.metadata.insert(key, value);
    }
    
    /// 获取元数据
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }
    
    /// 获取记录数量
    pub fn record_count(&self) -> usize {
        self.records.len()
    }
}

/// 默认实现
impl Default for ProcessorManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ProcessingPipeline {
    fn default() -> Self {
        Self::new()
    }
} 