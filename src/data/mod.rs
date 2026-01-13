// 数据模块
// 负责数据处理、特征提取、模型选择和数据连接等功能

// 引入必要的依赖
use std::collections::HashMap;
// Arc 未直接在此模块使用，去除未使用导入
use serde::{Serialize, Deserialize};
use crate::error::Result;

// 核心模块
pub mod schema;      // 数据架构定义
pub mod loader;      // 数据加载器
pub mod processor;   // 数据处理器
pub mod pipeline;    // 数据管道
pub mod shared;      // 跨系统集成的共享核心功能
pub mod value;       // 数据值类型
pub mod transform;   // 数据转换操作
pub mod manager;     // 数据管理器

// 新增的模块化组件
pub mod types;       // 基础类型定义
pub mod dataset;     // 数据集管理
pub mod batch;       // 数据批次处理
pub mod manager_core; // 数据管理器核心
pub mod cache;       // 缓存管理
pub mod processing;  // 数据处理系统
pub mod factory;     // 工厂函数

// 数据迭代器模块
pub mod iterator;    // 数据迭代器实现

// 特征提取模块
pub mod text_features;          // 文本特征提取
pub mod multimodal;             // 多模态特征提取

// 算法选择模块
pub mod method_selector;        // 方法选择器

// 连接器模块
pub mod connector;              // 数据库连接器

// 工具模块
pub mod utils;                  // 通用工具
pub mod adaptive_weights;       // 自适应权重调整

// 统一特征提取接口模块
pub mod feature;

// 验证模块
pub mod validation;  // 数据验证

// 数据处理其他模块
pub mod record;      // 数据记录

// 数据IO相关功能
pub mod io;

// 新增的高级功能模块
pub mod streaming;      // 流式数据处理
pub mod versioning;     // 数据版本管理
pub mod consistency;    // 跨节点数据一致性
pub mod format_support; // 扩展数据格式支持

// 分片处理模块
pub mod shard;

// 图片处理模块
pub mod image_processing;

// 导出模块
pub mod exports;

// 重新导出基础类型
pub use types::{
    DataConfig, DataFormat, DatasetMetadata, DataStatus, DataSplit,
    ProcessingStep, ProcessingOptions, DataItem, DataSample, DataType
};

// 重新导出数据集相关（包括DataSource、LocalDatabaseConfig、ApiConfig）
pub use dataset::{Dataset, ExtendedDataset, ProcessedDataset};

// 重新导出批次相关
pub use batch::{DataBatch, DataIterator, DataBatchExt, DataSourceConfig, BasicDataSource};

// 重新导出管理器相关
pub use manager_core::{DataManager, RawData, ProcessedData, DataService};
pub use manager::{DataManagerConfig};

// 重新导出缓存相关
pub use cache::{AsyncCacheManager, CacheStats, WarmupStats};

// 重新导出处理相关
pub use processing::{DataPipeline as ProcessingDataPipeline, DataPipelineConfig, DataProcessingSystem};

// 重新导出工厂函数
pub use factory::{
    infer_schema_from_file, create_csv_loader, create_data_pipeline,
    create_data_processor, create_memory_loader, create_file_loader,
    create_data_validator, create_data_transformer,
    create_data_manager, create_cache_manager,
    create_quick_dataset, create_data_batch,
    create_batch_iterator, create_streaming_iterator
};

// 重新导出数据记录
pub use record::DataRecord;

// 重新导出处理器相关
pub use processor::{
    ProcessorConfig, TransformerConfig, ColumnInfo, ValidationRule,
    NumericTransformer, CategoricalTransformer, DateTimeTransformer,
    normalize, tokenize, encode, transform as proc_transform, filter, augment,
    types::{
        ProcessorBatch as ProcessedBatch, ProcessorDataset as ProcessorProcessedDataset,
        DataContext, StandardFeatureConfig,
        ProcessingError, ProcessorType, ProcessorMetadata,
        core::{DataPipeline, ProcessingStats, DataQualityMetrics, BatchProcessor, BatchConfig, AdvancedProcessor, AdvancedProcessingConfig}
    }
};

// 重新导出处理器相关的FeatureType，避免与types中的DataType冲突
pub use processor::types::DataType as FeatureType;

// 重新导出文本特征提取
pub use text_features::{
    TextFeatureExtractor, 
    TextPreprocessor as TextPreprocessorInternal, 
    TextProcessingOptions,
    default_processing_options,
    TextFeatureConfig, TextFeatureMethod, MixedFeatureConfig,
};
pub use text_features::vectorizer::TextVectorizer;
pub use text_features::encoder::SentenceEncoder;
pub use text_features::embedding::TextEmbedding;
pub use text_features::extractors::models::FeatureImportance;

// 重新导出多模态相关
pub use multimodal::{
    MultiModalExtractor, MultiModalConfig, ModalityType, FusionStrategy as MMFusionStrategy,
    AlignmentMethod, AlignmentConfig, ModalityConfig, ExtractionConfig,
    extractors::image::ImageProcessingConfig, extractors::audio::config::AudioProcessingConfig
};

// 重新导出自适应权重（这里有NormalizationMethod的导入）
pub use adaptive_weights::{
    AdaptiveWeightsConfig, AdaptiveStrategy, NormalizationMethod as AdaptiveNormalizationMethod
};

// 重新导出方法选择器
pub use method_selector::{
    MethodSelector, DataAnalyzer, MethodEvaluator,
    MethodEvaluation, MethodSelectorConfig, DataCharacteristics as MsDataCharacteristics,
    DomainRule, PerformanceDataPoint, TextFieldStats as MsTextFieldStats,
    select_best_method, select_best_method_with_config, new_selector, new_default_selector
};
pub use crate::data::text_features::NumericStats as MsNumericFieldStats;
pub use crate::data::text_features::CategoricalFieldStats as MsCategoricalFieldStats;

// 重新导出加载器
pub use loader::DataLoader;

// 重新导出架构相关
pub use schema::{
    DataSchema, DataField,
    FieldDefinition, 
    FieldConstraints,
    FieldType, FieldSource, Schema, IndexDefinition, IndexType, SchemaMetadata
};

// 重新导出连接器
pub use connector::{
    DatabaseType, DatabaseConfig as ImportedDatabaseConfig, QueryParams, QueryParam, SortDirection
};

// 重新导出数据值，确保DataValue和ProcessedBatch是public的
pub use value::{DataValue, UnifiedValue, ScalarValue, VectorValue, MatrixValue, TensorValue, DType, DataValueAdapter};

// 重新导出核心类型
pub use crate::core::common_types::UnifiedTensorData;

// 重新导出转换相关
pub use transform::{DataTransformer, RecordTransformer, Transformer};

/// 数据转换类型枚举
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataTransformationType {
    /// 字段映射：重命名字段
    FieldMapping {
        source: String,
        target: String,
    },
    /// 类型转换：转换字段类型
    TypeConversion {
        field: String,
        target_type: FieldType,
    },
    /// 过滤：基于条件过滤数据
    Filtering {
        conditions: Vec<FilterCondition>,
    },
    /// 聚合：对字段进行聚合操作
    Aggregation {
        field: String,
        operation: AggregationOperation,
    },
    /// 标准化：对字段进行标准化处理
    Normalization {
        field: String,
        method: NormalizationMethod,
    },
    /// 特征工程：生成新的特征字段
    FeatureEngineering {
        features: Vec<FeatureDefinition>,
    },
    /// 自定义转换：用户定义的转换
    Custom {
        name: String,
        config: HashMap<String, serde_json::Value>,
    },
}

/// 过滤条件结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    pub field: String,
    pub operator: FilterOperator,
    pub value: serde_json::Value,
}

/// 过滤操作符枚举
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Contains,
    NotContains,
    StartsWith,
    EndsWith,
    IsNull,
    IsNotNull,
    In,
    NotIn,
}

/// 聚合操作枚举
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationOperation {
    Sum,
    Average,
    Count,
    Min,
    Max,
    StandardDeviation,
    Variance,
    Median,
    Mode,
    Percentile(f64),
    CountDistinct,
    First,
    Last,
}

/// 标准化方法枚举
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationMethod {
    ZScore,
    MinMax,
    Robust,
    Unit,
    MaxAbs,
    Quantile,
    PowerTransform,
    StandardScaler,
}

/// 特征定义结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDefinition {
    pub name: String,
    pub data_type: FieldType,
    pub nullable: Option<bool>,
    pub expression: Option<String>,
    pub dependencies: Option<Vec<String>>,
    pub metadata: Option<HashMap<String, String>>,
}

/// 数据转换配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTransformation {
    /// 转换名称
    pub name: String,
    /// 转换类型
    pub transformation_type: DataTransformationType,
    /// 转换参数
    pub parameters: HashMap<String, serde_json::Value>,
    /// 转换描述
    pub description: Option<String>,
    /// 是否启用
    pub enabled: bool,
    /// 执行顺序
    pub order: i32,
}

impl DataTransformation {
    /// 创建新的数据转换
    pub fn new(name: &str, transformation_type: DataTransformationType) -> Self {
        Self {
            name: name.to_string(),
            transformation_type,
            parameters: HashMap::new(),
            description: None,
            enabled: true,
            order: 0,
        }
    }

    /// 设置转换参数
    pub fn with_parameter<T: Serialize>(mut self, key: &str, value: T) -> Self {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.parameters.insert(key.to_string(), json_value);
        }
        self
    }

    /// 设置转换描述
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = Some(description.to_string());
        self
    }

    /// 设置启用状态
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// 设置执行顺序
    pub fn with_order(mut self, order: i32) -> Self {
        self.order = order;
        self
    }

    /// 创建字段映射转换
    pub fn field_mapping(name: &str, source: &str, target: &str) -> Self {
        Self::new(name, DataTransformationType::FieldMapping {
            source: source.to_string(),
            target: target.to_string(),
        })
    }

    /// 创建类型转换
    pub fn type_conversion(name: &str, field: &str, target_type: FieldType) -> Self {
        Self::new(name, DataTransformationType::TypeConversion {
            field: field.to_string(),
            target_type,
        })
    }

    /// 创建过滤转换
    pub fn filtering(name: &str, conditions: Vec<FilterCondition>) -> Self {
        Self::new(name, DataTransformationType::Filtering { conditions })
    }

    /// 创建聚合转换
    pub fn aggregation(name: &str, field: &str, operation: AggregationOperation) -> Self {
        Self::new(name, DataTransformationType::Aggregation {
            field: field.to_string(),
            operation,
        })
    }

    /// 创建标准化转换
    pub fn normalization(name: &str, field: &str, method: NormalizationMethod) -> Self {
        Self::new(name, DataTransformationType::Normalization {
            field: field.to_string(),
            method,
        })
    }

    /// 创建特征工程转换
    pub fn feature_engineering(name: &str, features: Vec<FeatureDefinition>) -> Self {
        Self::new(name, DataTransformationType::FeatureEngineering { features })
    }

    /// 创建自定义转换
    pub fn custom(name: &str, config: HashMap<String, serde_json::Value>) -> Self {
        Self::new(name, DataTransformationType::Custom {
            name: name.to_string(),
            config,
        })
    }
}

/// 通用数据存储接口
pub trait DataStorage: Send + Sync {
    /// 存储数据
    fn store(&self, key: &str, data: &[u8]) -> Result<()>;
    /// 获取数据
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;
    /// 删除数据
    fn delete(&self, key: &str) -> Result<()>;
    /// 检查数据是否存在
    fn exists(&self, key: &str) -> Result<bool>;
    /// 列出所有键
    fn list_keys(&self) -> Result<Vec<String>>;
    /// 批量操作
    fn batch_store(&self, items: &[(String, Vec<u8>)]) -> Result<()>;
    /// 清理存储
    fn clear(&self) -> Result<()>;
    /// 获取存储统计信息
    fn stats(&self) -> Result<StorageStats>;
}

/// 存储统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    pub total_keys: usize,
    pub total_size_bytes: u64,
    pub last_modified: Option<std::time::SystemTime>,
}

// 重新导出流式处理相关
pub use streaming::{
    StreamingConfig, StreamingBatch, StreamingStats, StreamingDataSource,
    FileStreamingSource, StreamingProcessor, StreamingAggregator,
    create_file_streaming_source, create_streaming_processor
};

// 重新导出版本管理相关
pub use versioning::{
    VersionId, BranchName, TagName, VersionInfo, FileEntry, FileType,
    BranchInfo, TagInfo, DiffType, DiffEntry, MergeStrategy, MergeConflict,
    ConflictType, MergeResult, MergeStats, DataVersionManager, VersionConfig,
    create_version_manager
};

// 重新导出一致性相关
pub use consistency::{
    TransactionId, NodeId, Timestamp, MVCCController, MVCCConfig,
    TransactionInfo, TransactionStatus, DistributedLockManager, LockConfig,
    LockInfo, LockType, LockStatus, LockRequest, ConsistencyCoordinator,
    ConsistencyConfig, ConsistencyLevel, ReadStrategy, WriteStrategy,
    ConflictResolution, create_consistency_coordinator
};

// 重新导出扩展格式支持相关
pub use format_support::{
    ExtendedDataFormat, FormatDetector, DataFormatReader, DataFormatWriter,
    XmlReader, YamlReader, TomlReader, TsvReader, FixedWidthReader,
    FixedWidthField, FixedWidthDataType, DataFormatManager,
    create_data_format_manager, create_fixed_width_reader
};

// 重新导出分片相关
pub use shard::{DataShardManager, ShardConfig, ShardInfo, ShardStrategy, PartitionKey, ShardMetrics};

// 重新导出数据流水线类型
pub use pipeline::{
    PipelineConfig, PipelineResult,
    ParallelProcessor, ParallelConfig
};

// 从正确模块导入其他类型
pub use text_features::pipeline::ProcessingStage;
pub use adaptive_weights::WeightAdjuster; // 单独导入WeightAdjuster避免重复

// 同时导出具体的性能指标类型
pub use pipeline::ParallelProcessingMetrics; // 新增

/// 记录值类型 - 与handlers.rs兼容
pub use crate::data::value::DataValue as RecordValue;

/// 数据行结构 - 与iterator.rs兼容
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DataRow {
    /// 行数据
    pub data: HashMap<String, DataValue>,
    /// 行ID
    pub id: Option<String>,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

impl DataRow {
    /// 创建新的数据行
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            id: None,
            metadata: HashMap::new(),
        }
    }

    /// 创建带数据的数据行
    pub fn with_data(data: HashMap<String, DataValue>) -> Self {
        Self {
            data,
            id: None,
            metadata: HashMap::new(),
        }
    }

    /// 设置行ID
    pub fn with_id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }

    /// 添加字段
    pub fn add_field(&mut self, key: String, value: DataValue) {
        self.data.insert(key, value);
    }

    /// 获取字段
    pub fn get_field(&self, key: &str) -> Option<&DataValue> {
        self.data.get(key)
    }

    /// 获取所有字段
    pub fn fields(&self) -> &HashMap<String, DataValue> {
        &self.data
    }
}

impl Default for DataRow {
    fn default() -> Self {
        Self::new()
    }
}

// 导出UserDataProcessor
pub use processor::processor_impl::UserDataProcessor;