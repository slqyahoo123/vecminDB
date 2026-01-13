// Data Processor Core Implementation
// 数据处理器核心实现 - 重新导出和兼容性接口

// trim unused pipeline trait imports; only re-exports used here
// use crate::Result; // re-export hub; no direct use here

// 导出TraitsDataProcessor供其他模块使用
pub use crate::data::pipeline::traits::DataProcessor as TraitsDataProcessor;

// 重新导出新模块中的类型，以保持向后兼容性
pub use super::types_core::{
    ProcessorState, ProcessorStatus, ProcessorType, 
    TaskHandle, ProcessorMetrics, MemoryInfo,
    StorageHealthCheck, Processor
};

pub use super::processor_impl::{DataProcessor, ImportStats, ExportStats};

pub use super::schema_ops::{
    extract_schema_from_metadata, infer_schema_from_data, 
    SchemaMerger, MergeStrategy
};

pub use super::data_ops::{
    DataParser, DataConverter, DataValidator, DataCleaner, DataStatistics
};

pub use super::record_ops::{
    RecordProcessor, FeatureExtractor, ValidationRule, RecordValueType,
    RecordTransformer, Transformation
};

pub use super::utils::{
    file_exists, ensure_dir_exists, read_file, write_file,
    get_file_size, is_directory, is_file, get_file_extension,
    join_path, create_temp_filename, safe_delete_file, safe_delete_dir,
    copy_file, move_file, list_files, calculate_file_hash, check_disk_space
};

// 便捷函数，为了保持向后兼容性
pub use super::{
    create_processor, quick_import, quick_export, quick_parse,
    quick_validate, quick_clean, quick_stats,
    ProcessorFactory, ProcessorManager, ProcessingPipeline
}; 