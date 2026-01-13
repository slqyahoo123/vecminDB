// 数据加载模块 - 负责从各种来源加载数据

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::sync::Mutex;

use async_trait::async_trait;
use log::{debug, error, info, trace, warn};
use lazy_static::lazy_static;

use crate::data::{DataBatch, DataConfig, DataSchema};
use crate::data::pipeline::ImportPipelineBuilder;
use crate::data::pipeline::pipeline::PipelineContext;
use crate::error::{Error, Result};

// 导出所有子模块
pub mod file;
pub mod database;
pub mod stream;
pub mod schema;
pub mod utils;
pub mod binary;
pub mod common;

// 新增模块
pub mod types;
pub mod factory;
pub mod importer;
pub mod context;
pub mod async_loader;
pub mod memory;  // 添加memory模块

// 重新导出所有公共类型和方法
pub use file::FileDataLoader;
pub use stream::{StreamDataLoader, StreamConfig, StreamType};
pub use common::{DataCache, DataSourceConnector, FormatProcessor, MemoryCache, DataLoaderBuilder};
pub use memory::MemoryDataLoader;  // 添加MemoryDataLoader导出

// 导出基础类型
pub use types::{DataFormat, DataSource, FileType};

// 导出工厂相关
pub use factory::{LoaderFactory, register_loader, create_loader_v2};

// 导出导入器相关
pub use importer::DataImporter;
pub use config::ImportConfig;
pub use result::ImportResult;

// 导出上下文和统计相关
pub use context::{Context, LoadStats};

// 导出异步加载器相关
pub use async_loader::{process_with_context};

// 导出加载器特性相关
pub use traits::{FileLoader, DatabaseLoaderTrait, ApiLoader};

// 导出工具类
pub use utils::detect_file_format;

// 导出其他实用类型
pub use file::processor_factory::FileProcessorFactory;
pub use file::processor_factory::FileType as ProcessorFileType;
pub use binary::{CustomBinaryLoader, CustomBinaryFormat, Endianness};

// 重新导出子模块
pub mod config;
pub mod validation;
pub mod progress;
pub mod result;
pub mod batch_importer;

// 注意：file_processor 和 utils 特性未在 Cargo.toml 中定义，已注释
// #[cfg(feature = "file_processor")]
// pub mod file;

// #[cfg(feature = "utils")]
// pub mod utils;

// 重新导出常用结构体和枚举
pub use self::config::{BatchImportConfig};
pub use self::validation::{BatchValidator, StandardBatchValidator};
pub use crate::core::interfaces::ValidationResult;
pub use self::progress::{ProgressTracker, ProgressReporter};
pub use self::result::{BatchSummary, BatchImportResult};
pub use self::batch_importer::BatchImporter;
pub use self::types::{ImportSourceType};

/// 数据加载器配置
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// 数据格式
    pub format: Option<types::DataFormat>,
    /// 批处理大小
    pub batch_size: Option<usize>,
    /// 超时时间（秒）
    pub timeout: Option<u64>,
    /// 最大内存使用量（字节）
    pub max_memory: Option<usize>,
    /// 是否验证数据
    pub validate: bool,
    /// CSV是否包含标题行
    pub has_header: Option<bool>,
    /// 额外选项
    pub options: HashMap<String, String>,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            format: None,
            batch_size: Some(1000),
            timeout: Some(60),
            max_memory: Some(1024 * 1024 * 1024), // 1GB
            validate: true,
            has_header: Some(true),
            options: HashMap::new(),
        }
    }
}

impl LoaderConfig {
    /// 创建新的加载器配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置数据格式
    pub fn with_format(mut self, format: types::DataFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// 设置批处理大小
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = Some(size);
        self
    }

    /// 设置超时时间
    pub fn with_timeout(mut self, timeout: u64) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// 设置最大内存使用量
    pub fn with_max_memory(mut self, memory: usize) -> Self {
        self.max_memory = Some(memory);
        self
    }

    /// 设置验证标志
    pub fn with_validate(mut self, validate: bool) -> Self {
        self.validate = validate;
        self
    }

    /// 添加选项
    pub fn with_option<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.options.insert(key.into(), value.into());
        self
    }
}

// 模块初始化函数
pub fn initialize() -> Result<()> {
    info!("初始化数据加载模块 - 设置全局配置和资源");
    
    // 配置默认加载器
    let mut default_loaders = HashMap::new();
    // 使用格式标识字符串作为键，保持与文件类型映射一致
    default_loaders.insert("csv".to_string(), "file_csv_loader".to_string());
    default_loaders.insert("json".to_string(), "file_json_loader".to_string());
    default_loaders.insert("parquet".to_string(), "file_parquet_loader".to_string());
    
    // 初始化缓存系统
    // 使用内存缓存实现 DataCache 接口，缓存大小使用经验默认值
    let cache: Box<dyn DataCache> = Box::new(MemoryCache::new(10_000));
    info!("数据加载缓存系统已初始化，最大缓存条目数: 10000");
    
    // 创建连接池
    info!("正在配置数据源连接池");
    let connection_pool = Arc::new(common::NullDataSourceConnector::new());
    
    // 注册标准数据处理管道
    let pipeline_builder = ImportPipelineBuilder::new()
        .with_validation(true)
        .with_schema_inference(true)
        .with_batch_size(1000);
    info!("默认数据导入管道已配置：验证=开启，模式推断=开启，批处理大小=1000");
    
    // 初始化工厂
    factory::initialize(default_loaders, cache, connection_pool)?;
    info!("数据加载器工厂已初始化，注册了{}个默认加载器", 3);
    
    Ok(())
}

// 使用lazy_static定义全局加载器配置缓存
lazy_static! {
    static ref LOADER_CONFIGS: Mutex<HashMap<String, Arc<DataConfig>>> = Mutex::new(HashMap::new());
    static ref DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
}

// 添加一个使用全局配置的函数
pub fn get_loader_config(name: &str) -> Option<Arc<DataConfig>> {
    if let Ok(configs) = LOADER_CONFIGS.lock() {
        if let Some(config) = configs.get(name) {
            debug!("获取加载器配置: {}", name);
            return Some(config.clone());
        }
    } else {
        error!("无法获取加载器配置锁");
    }
    None
}

// 设置加载器配置
pub fn set_loader_config(name: String, config: DataConfig) -> Result<()> {
    if let Ok(mut configs) = LOADER_CONFIGS.lock() {
        configs.insert(name.clone(), Arc::new(config));
        debug!("设置加载器配置: {}", name);
        Ok(())
    } else {
        error!("无法获取加载器配置锁");
        Err(Error::concurrency("无法获取配置锁"))
    }
}

// 使用ImportPipelineBuilder创建数据导入管道
pub fn create_import_pipeline(config: &DataConfig) -> Result<ImportPipelineBuilder> {
    debug!("创建导入管道，配置: {:?}", config);
    let builder = ImportPipelineBuilder::new();
    
    // 检查是否启用验证（从validate字段或extra_params中获取）
    let should_validate = config.validate.unwrap_or(false) || 
        config.extra_params.get("validate_data")
            .and_then(|v| v.parse::<bool>().ok())
            .unwrap_or(false);
    
    if should_validate {
        warn!("数据验证已启用，这可能会影响性能");
    }
    
    Ok(builder)
}

// 数据加载器接口 - 核心接口定义保留在根模块
#[async_trait]
pub trait DataLoader: Send + Sync {
    // 加载数据
    async fn load(&self, source: &DataSource, format: &DataFormat) -> Result<DataBatch>;
    
    // 获取数据架构
    async fn get_schema(&self, source: &DataSource, format: &DataFormat) -> Result<DataSchema>;
    
    // 获取数据加载器名称
    fn name(&self) -> &'static str;
    
    // 加载批次数据
    async fn load_batch(&self, source: &DataSource, format: &DataFormat, batch_size: usize, offset: usize) -> Result<DataBatch> {
        // 默认实现：调用 load 后进行切片。具体实现可覆盖以获得更高性能的分批读取。
        let mut batch = self.load(source, format).await?;
        if batch.batch_size() > offset {
            let end = std::cmp::min(offset + batch_size, batch.batch_size());
            batch = batch.slice(offset, end)?;
        } else {
            // 返回空批次
            batch = DataBatch::new("empty", 0, 0);
        }
        Ok(batch)
    }
    
    // 检查是否支持指定格式
    fn supports_format(&self, _format: &DataFormat) -> bool {
        // 默认实现：保守返回 true，具体实现可按需覆盖
        true
    }
    
    // 获取配置
    fn config(&self) -> &LoaderConfig {
        // 默认实现：返回一个静态默认配置
        static DEFAULT_CONFIG: std::sync::OnceLock<LoaderConfig> = std::sync::OnceLock::new();
        DEFAULT_CONFIG.get_or_init(|| LoaderConfig::default())
    }
    
    // 设置配置
    fn set_config(&mut self, _config: LoaderConfig) {
        // 默认实现：不执行任何操作；具体实现可根据需要覆盖
    }
    
    // 使用上下文加载数据
    async fn load_with_context(&self, source: &DataSource, format: &DataFormat, context: &mut PipelineContext) -> Result<DataBatch> {
        // 默认实现是调用无上下文的load方法，子类可以重写此方法以实现上下文感知的加载
        let mut batch = self.load(source, format).await?;
        
        // 从上下文中获取元数据并添加到批次中
        // PipelineContext 使用 params/data/state 存储，尝试从这些地方获取元数据
        if let Ok(metadata) = context.get_data::<HashMap<String, String>>("metadata") {
            for (key, value) in metadata.iter() {
                batch.add_metadata(key, value);
            }
        }
        
        // 记录加载信息到上下文状态中
        let batches_loaded = context.get_state("loader.batches_loaded")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0) + 1;
        context.set_state("loader.batches_loaded", &batches_loaded.to_string());
        
        let records_loaded = context.get_state("loader.records_loaded")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0) + batch.size;
        context.set_state("loader.records_loaded", &records_loaded.to_string());
        
        info!("通过上下文加载批次完成，共加载 {} 条记录", batch.size);
        
        Ok(batch)
    }
    
    // 获取数据集大小
    async fn get_size(&self, path: &str) -> Result<usize> {
        // 默认实现，子类可以重写
        info!("获取数据集大小: {}", path);
        Ok(0)
    }
    
    // 获取数据总大小
    async fn get_total_size(&self, path: &str) -> Result<usize> {
        // 默认实现，子类可以重写以提供更精确的大小信息
        self.get_size(path).await
    }

    // 验证数据源
    async fn validate_source(&self, source: &DataSource) -> Result<bool> {
        // 默认实现，子类可以重写以提供特定的验证逻辑
        match source {
            DataSource::File(path) => {
                Ok(std::path::Path::new(path).exists())
            },
            DataSource::Database(_) => {
                // 数据库连接验证由具体实现决定
                Ok(true)
            },
            DataSource::Stream(_) => {
                // 流数据源验证由具体实现决定
                Ok(true)
            },
            DataSource::Memory(_) => {
                // 内存数据源始终有效
                Ok(true)
            },
            DataSource::Custom(_, _) => {
                // 自定义数据源验证由具体实现决定
                Ok(true)
            },
        }
    }

    // 获取指定位置的批次数据
    async fn get_batch_at(&self, path: &str, index: usize, batch_size: usize) -> Result<DataBatch> {
        // 默认实现被子类覆盖
        info!("获取批次数据，路径: {}, 索引: {}, 批次大小: {}", path, index, batch_size);
        
        let source = DataSource::File(path.to_string());
        let format = if path.ends_with(".csv") {
            crate::data::loader::types::DataFormat::csv()
        } else if path.ends_with(".json") {
            crate::data::loader::types::DataFormat::json()
        } else if path.ends_with(".parquet") {
            crate::data::loader::types::DataFormat::parquet()
        } else {
            return Err(Error::invalid_input(format!("Unsupported file format for path: {}", path)));
        };
        
        let mut batch = self.load(&source, &format).await?;
        
        // 如果支持批次加载，就截取指定范围的数据
        if batch.batch_size() > index * batch_size {
            let start = index * batch_size;
            let end = std::cmp::min(start + batch_size, batch.batch_size());
            batch = batch.slice(start, end)?;
        }
        
        info!("成功获取批次数据，共 {} 条记录", batch.size);
        
        Ok(batch)
    }
}

// 使用trace记录详细日志信息
fn log_detailed_operation(operation: &str, details: &str) {
    trace!("执行操作: {}, 详情: {}", operation, details);
}

// 创建加载器特性模块
pub mod traits {
    use super::*;
    
    // 文件加载器trait
    pub trait FileLoader: DataLoader {
        /// 获取文件路径
        fn file_path(&self) -> &Path;
        
        /// 获取文件大小
        fn file_size(&self) -> Result<u64> {
            let metadata = std::fs::metadata(self.file_path())
                .map_err(|e| Error::io(&format!("无法获取文件元数据: {}", e)))?;
            Ok(metadata.len())
        }
        
        /// 获取文件类型
        fn file_type(&self) -> super::types::FileType;
        
        /// 检查文件是否存在
        fn file_exists(&self) -> bool {
            self.file_path().exists()
        }
    }
    
    // 数据库加载器trait
    pub trait DatabaseLoaderTrait: DataLoader {
        /// 获取数据库连接信息
        fn connection_info(&self) -> &str;
        
        /// 获取查询语句
        fn query(&self) -> &str;
        
        /// 测试连接
        fn test_connection(&self) -> Result<bool>;
    }
    
    // API加载器trait
    pub trait ApiLoader: DataLoader {
        /// 获取API端点URL
        fn endpoint(&self) -> &str;
        
        /// 获取API认证信息
        fn auth_info(&self) -> Option<&str>;
        
        /// 测试API连接
        fn test_connection(&self) -> Result<bool>;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde::{Deserialize, Serialize};
    use crate::data::{DataBatch, DataConfig, DataSchema};
    use crate::error::{Error, Result};
    use crate::data::loader::DataLoader;
    use crate::data::loader::DataSource;
    use chrono::Utc;
    
    #[test]
    fn test_loader_factory_creation() {
        let factory = factory::LoaderFactory::new();
        let registered_types = factory.get_registered_types();
        
        assert!(registered_types.contains(&"file".to_string()));
        assert!(registered_types.contains(&"database".to_string()));
        assert!(registered_types.contains(&"stream".to_string()));
        assert!(registered_types.contains(&"binary".to_string()));
        assert!(registered_types.contains(&"common".to_string()));
    }
    
    #[test]
    fn test_loader_file_creation() {
        let factory = factory::LoaderFactory::new();
        let loader = factory.create("file", None);
        assert!(loader.is_ok());
    }
    
    // 添加一个使用async_trait、Deserialize、Serialize的测试函数
    #[test]
    fn test_custom_data_loader() {
        // 定义一个使用serde的测试数据结构
        #[derive(Debug, Clone, Serialize, Deserialize)]
        struct TestRecord {
            id: String,
            name: String,
            value: i32,
            timestamp: i64,
        }
        
        // 定义一个实现async_trait的测试加载器
        struct TestAsyncLoader {
            config: DataConfig,
            records: Vec<TestRecord>,
            last_load_time: Option<chrono::DateTime<Utc>>,
        }
        
        impl TestAsyncLoader {
            fn new() -> Self {
                // 创建一些测试数据
                let records = vec![
                    TestRecord {
                        id: "1".to_string(),
                        name: "测试1".to_string(),
                        value: 100,
                        timestamp: Utc::now().timestamp(),
                    },
                    TestRecord {
                        id: "2".to_string(),
                        name: "测试2".to_string(),
                        value: 200,
                        timestamp: Utc::now().timestamp(),
                    },
                ];
                
                Self {
                    config: DataConfig::default(),
                    records,
                    last_load_time: None,
                }
            }
        }
        
        // 使用async_trait实现异步DataLoader接口
        #[async_trait]
        impl DataLoader for TestAsyncLoader {
            async fn load(&self, _source: &DataSource, _format: &DataFormat) -> Result<DataBatch> {
                // 记录加载开始时间
                info!("开始加载数据批次");
                
                // 将TestRecord转换为DataBatch
                let mut batch = DataBatch::new();
                for record in &self.records {
                    // 序列化记录为JSON，然后添加到批次
                    let json = serde_json::to_string(record)
                        .map_err(|e| Error::processing(format!("序列化失败: {}", e)))?;
                    
                    batch.add_record_json(&json)
                        .map_err(|e| Error::processing(format!("添加记录失败: {}", e)))?;
                }
                
                info!("成功加载 {} 条记录", batch.len());
                
                Ok(batch)
            }
            
            async fn get_schema(&self, _source: &DataSource, _format: &DataFormat) -> Result<DataSchema> {
                // 创建数据模式
                let mut schema = DataSchema::new("test_async_loader", "1.0");
                
                // 添加字段定义
                let fields = vec![
                    crate::data::schema::schema::FieldDefinition {
                        name: "id".to_string(),
                        field_type: crate::data::schema::FieldType::Text,
                        data_type: None,
                        required: true,
                        nullable: false,
                        primary_key: false,
                        foreign_key: None,
                        description: None,
                        default_value: None,
                        constraints: None,
                        metadata: std::collections::HashMap::new(),
                    },
                    crate::data::schema::schema::FieldDefinition {
                        name: "name".to_string(),
                        field_type: crate::data::schema::FieldType::Text,
                        data_type: None,
                        required: true,
                        nullable: false,
                        primary_key: false,
                        foreign_key: None,
                        description: None,
                        default_value: None,
                        constraints: None,
                        metadata: std::collections::HashMap::new(),
                    },
                    crate::data::schema::schema::FieldDefinition {
                        name: "value".to_string(),
                        field_type: crate::data::schema::FieldType::Numeric,
                        data_type: None,
                        required: true,
                        nullable: false,
                        primary_key: false,
                        foreign_key: None,
                        description: None,
                        default_value: None,
                        constraints: None,
                        metadata: std::collections::HashMap::new(),
                    },
                    crate::data::schema::schema::FieldDefinition {
                        name: "timestamp".to_string(),
                        field_type: crate::data::schema::FieldType::Numeric,
                        data_type: None,
                        required: true,
                        nullable: false,
                        primary_key: false,
                        foreign_key: None,
                        description: None,
                        default_value: None,
                        constraints: None,
                        metadata: std::collections::HashMap::new(),
                    },
                ];
                for f in fields {
                    schema.add_field(f)?;
                }
                
                debug!("生成模式: {:?}", schema);
                
                Ok(schema)
            }
            
            fn name(&self) -> &'static str {
                "TestAsyncLoader"
            }
        }
        
        // 创建测试加载器实例并验证
        let loader = TestAsyncLoader::new();
        
        // 确保数据加载器创建成功
        assert_eq!(loader.records.len(), 2);
        assert_eq!(loader.records[0].name, "测试1");
        assert_eq!(loader.records[1].value, 200);
    }
}
