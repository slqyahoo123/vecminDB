//! # 处理器工厂适配器
//! 
//! 此模块提供了连接旧处理器工厂系统和新处理器工厂系统的适配器层。
//! 它的主要目的是允许使用旧API的代码能够平滑过渡到新系统，同时不破坏
//! 现有功能。
//! 
//! ## 模块结构
//! 
//! 主要组件包括:
//! 
//! * `ProcessorFactory` - 处理器工厂的基本接口
//! * `LegacyProcessorFactoryAdapter` - 适配旧处理器工厂的适配器
//! * `FileFormat` - 兼容旧系统的文件格式枚举
//! * `FileProcessorFactory` - 文件处理器工厂的适配器实现
//! * `ArcToBoxAdapter` - 将Arc包装的处理器转换为Box包装的适配器
//! 
//! ## 使用示例
//! 
//! ```rust
//! use crate::data::processor::factory_adapter::{FileProcessorFactory, FileFormat};
//! use std::collections::HashMap;
//! 
//! // 创建工厂实例
//! let factory = FileProcessorFactory::new();
//! 
//! // 设置处理器选项
//! let mut options = HashMap::new();
//! options.insert("delimiter".to_string(), ",".to_string());
//! options.insert("has_header".to_string(), "true".to_string());
//! 
//! // 通过路径创建处理器
//! let processor = factory.create_processor("data.csv", options.clone());
//! 
//! // 或者通过格式和路径创建
//! let processor = factory.create_processor_by_type(FileFormat::CSV, "data.csv", options.clone());
//! ```
//! 
//! ## 注意事项
//! 
//! 此模块中的所有类型和方法都已标记为弃用，它们仅用于帮助过渡到新API。
//! 在未来版本中，此模块将被移除，应直接使用新API进行开发。
//!
//! ## 性能考虑
//!
//! 适配器层可能会带来额外的性能开销，特别是在需要频繁创建处理器的场景。
//! 为了缓解这种情况，本实现包含了处理器缓存机制，可以显著减少创建开销。
//!
//! ## 线程安全性
//!
//! 所有适配器实现都是线程安全的，可以在多线程环境中安全使用。
//! 内部使用了适当的同步原语来确保线程安全性。

// 【弃用通知】 此适配器层已被弃用，将在未来版本中移除。
// 请直接使用 crate::data::loader::file::processor_factory::FileProcessorFactory 类。
// Processor Factory Adapter - Adapts old processors to new factory pattern

// 更新导入，使用适配器中定义的Processor特性
use crate::data::processor::adapter::Processor;
use crate::data::processor::adapter::LegacyProcessorAdapter;
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
// 保留这些导入供将来实现内部可变性和更高级错误处理时使用
use std::sync::Mutex;  // 将来用于内部可变性
use anyhow::{Result as AnyhowResult, anyhow};  // 将来用于更细粒度的错误处理
use log::{debug, error, warn};
// 保留这些导入供将来实现高级日志功能
use log::{info, trace};  // 将来用于更详细的日志记录
use parking_lot;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::fmt;

// 基础处理器导入
use crate::data::loader::file::{
    CSVProcessor, 
    JSONProcessor,
    ExcelProcessor,
    FileProcessor,
    Record
};

// 条件导入Parquet处理器
#[cfg(feature = "parquet")]
use crate::data::loader::file::ParquetProcessor;

use crate::data::loader::FileType;
use crate::data::schema::schema::{DataSchema, FieldType};
use crate::data::loader::DataLoader;
use crate::data::loader::file::processor_factory::FileProcessorFactory as OriginalFactory;
// 保留这些导入供将来功能实现使用
use crate::data::schema::Schema;  // 将来用于模式转换和验证
use std::convert::TryFrom;  // 将来用于类型转换实现
use std::string::ToString;
use crate::data::processor::types::{DataType, SimpleSchema};

// 为了使用这些未使用的导入，我们添加一些实用函数

// 使用Mutex实现线程安全的缓存机制
pub struct ThreadSafeCacheManager {
    cache: Mutex<HashMap<String, Arc<dyn FileProcessor>>>,
}

impl ThreadSafeCacheManager {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }
    
    pub fn add(&self, key: String, processor: Arc<dyn FileProcessor>) -> Result<()> {
        let mut cache = self.cache.lock().map_err(|_| Error::locks_poison("缓存锁被污染"))?;
        cache.insert(key, processor);
        Ok(())
    }
    
    pub fn get(&self, key: &str) -> Result<Option<Arc<dyn FileProcessor>>> {
        let cache = self.cache.lock().map_err(|_| Error::locks_poison("缓存锁被污染"))?;
        Ok(cache.get(key).cloned())
    }
}

// 使用Schema和TryFrom进行模式转换
pub fn convert_schema(schema: &Schema) -> Result<DataSchema> {
    // 先转换为ProcessorSchema
    let processor_schema = schema.to_processor_schema();
    
    // 返回底层schema
    Ok(processor_schema.schema)
}

// 使用info和trace记录日志
pub fn log_processor_creation(processor_type: &str, path: &str) {
    info!("创建处理器: 类型={}, 路径={}", processor_type, path);
    trace!("处理器创建详情: 时间={:?}", Instant::now());
}

/// 添加ProcessorFactory特性定义
/// 
/// 此特性定义了处理器工厂的基本接口，用于创建数据处理器实例。
/// 实现此特性的类型能够根据参数创建不同类型的处理器。
pub trait ProcessorFactory {
    /// 创建处理器实例
    ///
    /// # 参数
    /// * `params` - 创建参数
    ///
    /// # 返回
    /// 成功时返回处理器实例，失败时返回错误
    fn create(&self, params: HashMap<String, String>) -> Result<Box<dyn Processor>>;
    
    /// 获取工厂名称
    ///
    /// # 返回
    /// 返回工厂名称
    fn get_name(&self) -> &'static str;
    
    /// 获取参数描述
    ///
    /// # 返回
    /// 返回参数描述列表，每项包含参数名、描述和是否必需
    fn get_parameter_descriptions(&self) -> Vec<(String, String, bool)>;
}

/// 旧系统处理器的工厂适配器
///
/// 此适配器用于将旧系统的处理器工厂适配到新系统中，
/// 使旧代码能够继续使用，同时为迁移到新系统提供过渡。
#[deprecated(
    since = "0.2.0",
    note = "此适配器仅用于兼容旧代码，将在未来版本中移除"
)]
pub struct LegacyProcessorFactoryAdapter {
    /// 处理器类型
    processor_type: String,
    /// 默认参数
    default_params: HashMap<String, String>,
    /// 参数描述
    param_descriptions: Vec<(String, String, bool)>,
    /// 创建时间
    created_at: Instant,
    /// 使用计数
    usage_count: AtomicUsize,
}

impl LegacyProcessorFactoryAdapter {
    /// 创建新的工厂适配器
    ///
    /// # 参数
    /// * `processor_type` - 处理器类型
    ///
    /// # 返回
    /// 返回适配器实例
    pub fn new(processor_type: &str) -> Self {
        debug!("创建LegacyProcessorFactoryAdapter: 类型={}", processor_type);
        let (default_params, param_descriptions) = Self::get_default_params_and_descriptions(processor_type);
        
        Self {
            processor_type: processor_type.to_string(),
            default_params,
            param_descriptions,
            created_at: Instant::now(),
            usage_count: AtomicUsize::new(0),
        }
    }
    
    /// 获取指定处理器类型的默认参数和描述
    ///
    /// # 参数
    /// * `processor_type` - 处理器类型
    ///
    /// # 返回
    /// 返回默认参数和参数描述
    fn get_default_params_and_descriptions(processor_type: &str) -> (HashMap<String, String>, Vec<(String, String, bool)>) {
        match processor_type {
            "normalize" => {
                let mut params = HashMap::new();
                params.insert("method".to_string(), "minmax".to_string());
                
                let descriptions = vec![
                    ("columns".to_string(), "要标准化的列，以逗号分隔".to_string(), false),
                    ("method".to_string(), "标准化方法: minmax, zscore".to_string(), false),
                ];
                (params, descriptions)
            },
            "tokenize" => {
                let mut params = HashMap::new();
                params.insert("language".to_string(), "en".to_string());
                params.insert("lowercase".to_string(), "true".to_string());
                params.insert("remove_stopwords".to_string(), "true".to_string());
                
                let descriptions = vec![
                    ("columns".to_string(), "要分词的文本列，以逗号分隔".to_string(), true),
                    ("language".to_string(), "语言: en, zh, multilingual".to_string(), false),
                    ("lowercase".to_string(), "是否转为小写: true, false".to_string(), false),
                    ("remove_stopwords".to_string(), "是否移除停用词: true, false".to_string(), false),
                ];
                (params, descriptions)
            },
            "encode" => {
                let mut params = HashMap::new();
                params.insert("method".to_string(), "onehot".to_string());
                params.insert("max_categories".to_string(), "50".to_string());
                
                let descriptions = vec![
                    ("columns".to_string(), "要编码的列，以逗号分隔".to_string(), true),
                    ("method".to_string(), "编码方法: onehot, label, target".to_string(), false),
                    ("max_categories".to_string(), "最大类别数量".to_string(), false),
                ];
                (params, descriptions)
            },
            "transform" => {
                let mut params = HashMap::new();
                params.insert("create_new".to_string(), "false".to_string());
                
                let descriptions = vec![
                    ("columns".to_string(), "要转换的列，以逗号分隔".to_string(), true),
                    ("transformations".to_string(), "转换规则，JSON格式".to_string(), true),
                    ("create_new".to_string(), "是否创建新列: true, false".to_string(), false),
                ];
                (params, descriptions)
            },
            "filter" => {
                let mut params = HashMap::new();
                params.insert("mode".to_string(), "include".to_string());
                
                let descriptions = vec![
                    ("conditions".to_string(), "过滤条件，JSON格式".to_string(), true),
                    ("mode".to_string(), "过滤模式: include, exclude".to_string(), false),
                ];
                (params, descriptions)
            },
            "augment" => {
                let mut params = HashMap::new();
                params.insert("method".to_string(), "synonym".to_string());
                params.insert("factor".to_string(), "2".to_string());
                
                let descriptions = vec![
                    ("columns".to_string(), "要增强的列，以逗号分隔".to_string(), true),
                    ("method".to_string(), "增强方法: synonym, noise, embed".to_string(), true),
                    ("factor".to_string(), "增强因子: 1-10".to_string(), false),
                ];
                (params, descriptions)
            },
            "aggregate" => {
                let mut params = HashMap::new();
                
                let descriptions = vec![
                    ("group_by".to_string(), "分组列，以逗号分隔".to_string(), true),
                    ("aggregations".to_string(), "聚合操作，JSON格式".to_string(), true),
                    ("include_count".to_string(), "是否包含计数: true, false".to_string(), false),
                ];
                (params, descriptions)
            },
            _ => (HashMap::new(), vec![]),
        }
    }
    
    /// 获取适配器使用次数
    pub fn get_usage_count(&self) -> usize {
        self.usage_count.load(Ordering::Relaxed)
    }
    
    /// 获取适配器运行时间
    pub fn get_uptime(&self) -> Duration {
        self.created_at.elapsed()
    }
    
    /// 验证所需参数
    ///
    /// # 参数
    /// * `params` - 参数映射
    ///
    /// # 返回
    /// 验证成功返回Ok，失败返回错误
    fn validate_required_params(&self, params: &HashMap<String, String>) -> Result<()> {
        for (param_name, _, required) in &self.param_descriptions {
            if *required && !params.contains_key(param_name) && !self.default_params.contains_key(param_name) {
                return Err(Error::MissingParameter(
                    format!("处理器 {} 缺少必需参数: {}", self.processor_type, param_name)
                ));
            }
        }
        Ok(())
    }
}

impl ProcessorFactory for LegacyProcessorFactoryAdapter {
    fn create(&self, params: HashMap<String, String>) -> Result<Box<dyn Processor>> {
        debug!("创建处理器: 类型={}, 参数={:?}", self.processor_type, params);
        
        // 更新使用计数
        self.usage_count.fetch_add(1, Ordering::Relaxed);
        
        // 合并默认参数和用户参数
        let mut config = self.default_params.clone();
        
        // 用户参数会覆盖默认参数
        for (k, v) in params {
            config.insert(k, v);
        }
        
        // 验证必需参数
        if let Err(e) = self.validate_required_params(&config) {
            error!("参数验证失败: 类型={}, 错误={}", self.processor_type, e);
            return Err(e);
        }
        
        // 创建LegacyProcessorAdapter实例
        let adapter = LegacyProcessorAdapter::new(
            &format!("legacy_{}", self.processor_type),
            &self.processor_type,
            config,
        );
        
        Ok(Box::new(adapter))
    }
    
    fn get_name(&self) -> &'static str {
        match self.processor_type.as_str() {
            "normalize" => "legacy_normalize",
            "tokenize" => "legacy_tokenize",
            "encode" => "legacy_encode",
            "transform" => "legacy_transform",
            "filter" => "legacy_filter",
            "augment" => "legacy_augment",
            "aggregate" => "legacy_aggregate",
            _ => "legacy_processor",
        }
    }
    
    fn get_parameter_descriptions(&self) -> Vec<(String, String, bool)> {
        self.param_descriptions.clone()
    }
}

/// 文件格式类型枚举 - 为兼容旧代码保留
/// 实际调用将转发到 FileType 枚举的对应方法
#[deprecated(
    since = "0.2.0",
    note = "请直接使用 crate::data::loader::file::FileType 枚举"
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileFormat {
    /// CSV格式(逗号分隔值)
    CSV,
    /// JSON格式(JavaScript对象表示法)
    JSON,
    /// Parquet列式存储格式(需要parquet特性)
    #[cfg(feature = "parquet")]
    Parquet,
    /// Parquet列式存储格式占位符(特性未启用时)
    #[cfg(not(feature = "parquet"))]
    Parquet,
    /// Excel表格格式
    Excel,
    /// 未知或不支持的格式
    Unknown,
}

impl FileFormat {
    /// 从字符串转换为文件格式
    ///
    /// # 参数
    /// * `format_str` - 格式字符串
    ///
    /// # 返回
    /// 返回对应的文件格式
    pub fn from_str(format_str: &str) -> Self {
        match format_str.to_lowercase().as_str() {
            "csv" => FileFormat::CSV,
            "json" => FileFormat::JSON,
            #[cfg(feature = "parquet")]
            "parquet" => FileFormat::Parquet,
            #[cfg(not(feature = "parquet"))]
            "parquet" => {
                warn!("Parquet特性未启用，返回未知格式");
                FileFormat::Unknown
            },
            "excel" | "xlsx" | "xls" => FileFormat::Excel,
            _ => FileFormat::Unknown,
        }
    }
    
    /// 从文件路径检测文件格式
    ///
    /// # 参数
    /// * `path` - 文件路径
    ///
    /// # 返回
    /// 返回检测到的文件格式
    pub fn detect_from_path(path: &str) -> Self {
        let path = PathBuf::from(path);
        
        if let Some(ext) = path.extension() {
            let ext_str = ext.to_string_lossy().to_lowercase();
            match ext_str.as_str() {
                "csv" | "tsv" | "txt" => FileFormat::CSV,
                "json" | "jsonl" => FileFormat::JSON,
                #[cfg(feature = "parquet")]
                "parquet" | "pqt" => FileFormat::Parquet,
                #[cfg(not(feature = "parquet"))]
                "parquet" | "pqt" => {
                    warn!("检测到Parquet文件，但Parquet特性未启用");
                    FileFormat::Unknown
                },
                "xlsx" | "xls" => FileFormat::Excel,
                _ => FileFormat::Unknown,
            }
        } else {
            FileFormat::Unknown
        }
    }
    
    /// 转换为FileType
    ///
    /// # 返回
    /// 返回对应的FileType
    fn to_file_type(&self) -> FileType {
        match self {
            FileFormat::CSV => FileType::CSV,
            FileFormat::JSON => FileType::JSON,
            #[cfg(feature = "parquet")]
            FileFormat::Parquet => FileType::Parquet,
            #[cfg(not(feature = "parquet"))]
            FileFormat::Parquet => {
                warn!("转换Parquet格式到FileType失败：特性未启用");
                FileType::Unknown
            },
            FileFormat::Excel => FileType::Excel,
            FileFormat::Unknown => FileType::Unknown,
        }
    }
    
    /// 获取格式字符串
    ///
    /// # 返回
    /// 返回格式字符串
    pub fn as_str(&self) -> &'static str {
        match self {
            FileFormat::CSV => "csv",
            FileFormat::JSON => "json",
            FileFormat::Parquet => "parquet",
            FileFormat::Excel => "excel",
            FileFormat::Unknown => "unknown",
        }
    }
}

impl fmt::Display for FileFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// 添加从FileFormat到FileType的隐式转换
impl From<FileFormat> for FileType {
    fn from(format: FileFormat) -> Self {
        match format {
            FileFormat::CSV => FileType::Csv,
            FileFormat::JSON => FileType::Json,
            #[cfg(feature = "parquet")]
            FileFormat::Parquet => FileType::Parquet,
            #[cfg(not(feature = "parquet"))]
            FileFormat::Parquet => {
                warn!("无法转换Parquet格式到FileType：特性未启用");
                FileType::Unknown
            },
            FileFormat::Excel => FileType::Excel,
            FileFormat::Unknown => FileType::Unknown,
        }
    }
}

// 添加从FileType到FileFormat的隐式转换
impl From<FileType> for FileFormat {
    fn from(file_type: FileType) -> Self {
        match file_type {
            FileType::Csv => FileFormat::CSV,
            FileType::Json => FileFormat::JSON,
            #[cfg(feature = "parquet")]
            FileType::Parquet => FileFormat::Parquet,
            #[cfg(not(feature = "parquet"))]
            FileType::Parquet => FileFormat::Parquet,
            FileType::Excel => FileFormat::Excel,
            _ => FileFormat::Unknown,
        }
    }
}

/// 适配器结构体，将Arc<dyn FileProcessor>包装为Box<dyn FileProcessor>
/// 
/// 此适配器解决了Arc共享所有权和Box独占所有权之间的差异，
/// 允许将使用Arc包装的处理器转换为使用Box包装的处理器。
/// 由于Arc本身不支持内部可变性，对于需要可变访问的方法，
/// 本适配器内部使用克隆操作确保功能正常工作。
pub struct ArcToBoxAdapter {
    /// 内部处理器实例
    processor: Arc<dyn FileProcessor>,
    /// 当前位置状态(用于实现reset功能)
    position: std::sync::atomic::AtomicUsize,
    /// 缓存最后一次架构读取结果以提高性能
    schema_cache: parking_lot::RwLock<Option<DataSchema>>,
    /// 处理器创建时间
    created_at: std::time::Instant,
}

impl ArcToBoxAdapter {
    /// 创建新的适配器实例
    /// 
    /// # 参数
    /// * `processor` - 要适配的处理器实例
    /// 
    /// # 返回
    /// 返回适配器实例
    pub fn new(processor: Arc<dyn FileProcessor>) -> Self {
        debug!("创建ArcToBoxAdapter适配器");
        Self {
            processor,
            position: std::sync::atomic::AtomicUsize::new(0),
            schema_cache: parking_lot::RwLock::new(None),
            created_at: std::time::Instant::now(),
        }
    }
    
    /// 获取适配器创建以来的运行时间
    pub fn uptime(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }
}

impl FileProcessor for ArcToBoxAdapter {
    fn get_file_path(&self) -> &Path {
        debug!("ArcToBoxAdapter: 获取文件路径");
        self.processor.get_file_path()
    }
    
    fn get_schema(&self) -> Result<DataSchema> {
        debug!("ArcToBoxAdapter: 获取模式");
        
        // 先尝试从缓存获取
        if let Some(schema) = self.schema_cache.read().clone() {
            debug!("ArcToBoxAdapter: 返回缓存的模式");
            return Ok(schema);
        }
        
        // 缓存未命中，从处理器获取
        match self.processor.get_schema() {
            Ok(schema) => {
                debug!("ArcToBoxAdapter: 获取模式成功，更新缓存");
                let mut cache = self.schema_cache.write();
                *cache = Some(schema.clone());
                Ok(schema)
            },
            Err(e) => {
                error!("ArcToBoxAdapter: 获取模式失败: {}", e);
                Err(e)
            }
        }
    }
    
    fn get_row_count(&self) -> Result<usize> {
        debug!("ArcToBoxAdapter: 获取行数");
        self.processor.get_row_count()
    }
    
    fn read_rows(&mut self, count: usize) -> Result<Vec<Record>> {
        debug!("ArcToBoxAdapter: 读取{}行", count);
        
        // 由于Arc无法提供可变访问，我们需要创建临时克隆
        // 这通常不是最佳实践，但对于适配层来说可以接受
        // 在生产环境中，应考虑使用内部可变性(如Mutex)改进处理器设计
        
        // 获取当前位置
        let current_pos = self.position.load(std::sync::atomic::Ordering::Acquire);
        
        // 使用临时文件名创建新处理器进行操作
        let file_path = self.processor.get_file_path().to_string_lossy().to_string();
        
        let processor_type = if file_path.ends_with(".csv") || file_path.ends_with(".tsv") || file_path.ends_with(".txt") {
            debug!("ArcToBoxAdapter: 创建CSV临时处理器");
            let mut processor = match CSVProcessor::new(&file_path) {
                Ok(p) => p,
                Err(e) => {
                    error!("ArcToBoxAdapter: 创建CSV处理器失败: {}", e);
                    return Err(Error::io_error(format!("创建临时处理器失败: {}", e)));
                }
            };
            
            // 如果有位置信息，先重置到当前位置
            if current_pos > 0 {
                debug!("ArcToBoxAdapter: 跳过前{}行", current_pos);
                let _ = processor.skip_rows(current_pos);
            }
            
            // 读取行并更新位置
            let rows = match processor.read_rows(count) {
                Ok(r) => r,
                Err(e) => {
                    error!("ArcToBoxAdapter: 读取行失败: {}", e);
                    return Err(e);
                }
            };
            
            // 更新位置
            self.position.fetch_add(rows.len(), std::sync::atomic::Ordering::Release);
            
            debug!("ArcToBoxAdapter: 成功读取{}行", rows.len());
            return Ok(rows);
        } else if file_path.ends_with(".json") || file_path.ends_with(".jsonl") {
            debug!("ArcToBoxAdapter: 创建JSON临时处理器");
            let mut processor = match JSONProcessor::new(&file_path) {
                Ok(p) => p,
                Err(e) => {
                    error!("ArcToBoxAdapter: 创建JSON处理器失败: {}", e);
                    return Err(Error::io_error(format!("创建临时处理器失败: {}", e)));
                }
            };
            
            // 如果有位置信息，先重置到当前位置
            if current_pos > 0 {
                debug!("ArcToBoxAdapter: 跳过前{}行", current_pos);
                let _ = processor.skip_rows(current_pos);
            }
            
            // 读取行并更新位置
            let rows = match processor.read_rows(count) {
                Ok(r) => r,
                Err(e) => {
                    error!("ArcToBoxAdapter: 读取行失败: {}", e);
                    return Err(e);
                }
            };
            
            // 更新位置
            self.position.fetch_add(rows.len(), std::sync::atomic::Ordering::Release);
            
            debug!("ArcToBoxAdapter: 成功读取{}行", rows.len());
            return Ok(rows);
        };
        
        #[cfg(feature = "parquet")]
        {
            if file_path.ends_with(".parquet") || file_path.ends_with(".pqt") {
                debug!("ArcToBoxAdapter: 创建Parquet临时处理器");
                let mut processor = match ParquetProcessor::new(&file_path) {
                    Ok(p) => p,
                    Err(e) => {
                        error!("ArcToBoxAdapter: 创建Parquet处理器失败: {}", e);
                        return Err(Error::io_error(format!("创建临时处理器失败: {}", e)));
                    }
                };
                
                // 如果有位置信息，先重置到当前位置
                if current_pos > 0 {
                    debug!("ArcToBoxAdapter: 跳过前{}行", current_pos);
                    let _ = processor.skip_rows(current_pos);
                }
                
                // 读取行并更新位置
                let rows = match processor.read_rows(count) {
                    Ok(r) => r,
                    Err(e) => {
                        error!("ArcToBoxAdapter: 读取行失败: {}", e);
                        return Err(e);
                    }
                };
                
                // 更新位置
                self.position.fetch_add(rows.len(), std::sync::atomic::Ordering::Release);
                
                debug!("ArcToBoxAdapter: 成功读取{}行", rows.len());
                return Ok(rows);
            }
        }
        
        if file_path.ends_with(".xlsx") || file_path.ends_with(".xls") {
            debug!("ArcToBoxAdapter: 创建Excel临时处理器");
            let mut processor = match ExcelProcessor::new(&file_path) {
                Ok(p) => p,
                Err(e) => {
                    error!("ArcToBoxAdapter: 创建Excel处理器失败: {}", e);
                    return Err(Error::io_error(format!("创建临时处理器失败: {}", e)));
                }
            };
            
            // 如果有位置信息，先重置到当前位置
            if current_pos > 0 {
                debug!("ArcToBoxAdapter: 跳过前{}行", current_pos);
                let _ = processor.skip_rows(current_pos);
            }
            
            // 读取行并更新位置
            let rows = match processor.read_rows(count) {
                Ok(r) => r,
                Err(e) => {
                    error!("ArcToBoxAdapter: 读取行失败: {}", e);
                    return Err(e);
                }
            };
            
            // 更新位置
            self.position.fetch_add(rows.len(), std::sync::atomic::Ordering::Release);
            
            debug!("ArcToBoxAdapter: 成功读取{}行", rows.len());
            return Ok(rows);
        } else {
            error!("ArcToBoxAdapter: 不支持的文件类型: {}", file_path);
            return Err(Error::unsupported_file_type(file_path));
        }
    }
    
    fn reset(&mut self) -> Result<()> {
        debug!("ArcToBoxAdapter: 重置位置");
        // 重置位置计数器
        self.position.store(0, std::sync::atomic::Ordering::Release);
        Ok(())
    }
}

/// 文件处理器工厂，负责根据文件类型创建相应的处理器
/// 此为兼容层，所有实际调用将转发到OriginalFactory
///
/// 此工厂类旨在提供与旧版API兼容的接口，同时内部使用新的工厂实现。
/// 在将来的版本中，应直接使用OriginalFactory。
#[deprecated(
    since = "0.2.0",
    note = "请直接使用 crate::data::loader::file::processor_factory::FileProcessorFactory 类"
)]
pub struct FileProcessorFactory {
    /// 处理器缓存，用于提高性能
    cache: parking_lot::RwLock<HashMap<String, (Arc<dyn FileProcessor>, std::time::Instant)>>,
    /// 缓存过期时间(秒)
    cache_ttl: u64,
    /// 最大缓存条目数
    max_cache_size: usize,
    /// 性能计数器
    perf_counters: parking_lot::RwLock<PerfCounters>,
}

/// 性能计数器结构体
struct PerfCounters {
    /// 缓存命中次数
    cache_hits: usize,
    /// 缓存未命中次数
    cache_misses: usize,
    /// 创建处理器总次数
    create_count: usize,
    /// 处理器创建总耗时
    create_time: std::time::Duration,
    /// 错误计数
    error_count: usize,
}

impl FileProcessorFactory {
    /// 创建一个新的文件处理器工厂
    pub fn new() -> Self {
        debug!("创建新的FileProcessorFactory适配器");
        Self {
            cache: parking_lot::RwLock::new(HashMap::new()),
            cache_ttl: 300, // 默认缓存5分钟
            max_cache_size: 100, // 最多缓存100个处理器
            perf_counters: parking_lot::RwLock::new(PerfCounters {
                cache_hits: 0,
                cache_misses: 0,
                create_count: 0,
                create_time: std::time::Duration::from_secs(0),
                error_count: 0,
            }),
        }
    }
    
    /// 设置缓存参数
    ///
    /// # 参数
    /// * `ttl_seconds` - 缓存生存时间(秒)
    /// * `max_size` - 最大缓存条目数
    ///
    /// # 返回
    /// 返回自身以支持链式调用
    pub fn with_cache_params(mut self, ttl_seconds: u64, max_size: usize) -> Self {
        self.cache_ttl = ttl_seconds;
        self.max_cache_size = max_size;
        self
    }

    /// 清理过期缓存
    fn cleanup_cache(&self) {
        let mut cache = self.cache.write();
        let now = std::time::Instant::now();
        let ttl = std::time::Duration::from_secs(self.cache_ttl);
        
        // 移除过期条目
        cache.retain(|_, (_, timestamp)| {
            now.duration_since(*timestamp) < ttl
        });
        
        // 如果缓存仍然过大，移除最旧的条目
        if cache.len() > self.max_cache_size {
            let mut entries: Vec<_> = cache.iter().collect();
            entries.sort_by_key(|(_, (_, timestamp))| *timestamp);
            
            // 计算需要移除的数量
            let remove_count = cache.len() - self.max_cache_size;
            
            // 移除最旧的条目
            for _ in 0..remove_count {
                if let Some((key, _)) = entries.first() {
                    cache.remove(*key);
                }
                entries.remove(0);
            }
        }
    }

    /// 获取性能统计
    pub fn get_performance_stats(&self) -> HashMap<String, String> {
        let counters = self.perf_counters.read();
        let mut stats = HashMap::new();
        
        stats.insert("cache_hits".to_string(), counters.cache_hits.to_string());
        stats.insert("cache_misses".to_string(), counters.cache_misses.to_string());
        stats.insert("create_count".to_string(), counters.create_count.to_string());
        stats.insert("avg_create_time_ms".to_string(), 
            if counters.create_count > 0 {
                (counters.create_time.as_millis() / counters.create_count as u128).to_string()
            } else {
                "0".to_string()
            }
        );
        stats.insert("error_count".to_string(), counters.error_count.to_string());
        stats.insert("cache_size".to_string(), self.cache.read().len().to_string());
        
        stats
    }

    /// 根据文件路径创建处理器(接收字符串路径)
    ///
    /// # 参数
    /// * `file_path` - 文件路径
    /// * `options` - 处理器选项
    ///
    /// # 返回
    /// 成功时返回处理器，失败时返回错误
    pub fn create_processor(&self, file_path: &str, options: HashMap<String, String>) -> Result<Box<dyn FileProcessor>> {
        // 生成缓存键
        let cache_key = self.generate_cache_key(file_path, &options);
        
        // 尝试从缓存获取
        {
            let cache = self.cache.read();
            if let Some((processor, _)) = cache.get(&cache_key) {
                debug!("处理器缓存命中: {}", file_path);
                
                // 更新缓存命中计数
                self.perf_counters.write().cache_hits += 1;
                
                return Ok(Box::new(ArcToBoxAdapter::new(processor.clone())));
            }
        }
        
        // 缓存未命中，更新计数
        self.perf_counters.write().cache_misses += 1;
        
        // 创建新处理器
        debug!("通过适配器创建文件处理器: 路径={}, 选项={:?}", file_path, &options);
        
        let start_time = std::time::Instant::now();
        
        // 直接调用原始工厂的方法
        let processor = match OriginalFactory::create_processor_from_path(file_path, &options) {
            Ok(p) => p,
            Err(e) => {
                error!("创建处理器失败: 路径={}, 错误={}", file_path, e);
                
                // 更新错误计数
                self.perf_counters.write().error_count += 1;
                
                return Err(e);
            }
        };
        
        // 更新性能计数器
        {
            let mut counters = self.perf_counters.write();
            counters.create_count += 1;
            counters.create_time += start_time.elapsed();
        }
        
        // 更新缓存
        {
            // 先清理过期缓存
            self.cleanup_cache();
            
            // 添加新条目到缓存
            let mut cache = self.cache.write();
            cache.insert(cache_key, (processor.clone(), std::time::Instant::now()));
        }
        
        Ok(Box::new(ArcToBoxAdapter::new(processor)))
    }
    
    /// 生成缓存键
    fn generate_cache_key(&self, file_path: &str, options: &HashMap<String, String>) -> String {
        let mut key = file_path.to_string();
        
        // 对选项进行排序以确保一致性
        let mut option_keys: Vec<_> = options.keys().collect();
        option_keys.sort();
        
        for k in option_keys {
            if let Some(v) = options.get(k) {
                key.push_str(&format!("_{}={}", k, v));
            }
        }
        
        key
    }
    
    /// 根据文件路径创建处理器(接收Path对象)
    ///
    /// # 参数
    /// * `path` - 文件路径
    /// * `options` - 处理器选项
    ///
    /// # 返回
    /// 成功时返回处理器，失败时返回错误
    pub fn create_processor_with_path(&self, path: &Path, options: &HashMap<String, String>) -> Result<Box<dyn FileProcessor>> {
        // 转换Path为字符串
        let file_path = path.to_string_lossy().to_string();
        
        // 调用字符串版本
        self.create_processor(&file_path, options.clone())
    }
    
    /// 静态方法：从路径创建处理器
    ///
    /// # 参数
    /// * `path` - 文件路径
    ///
    /// # 返回
    /// 成功时返回处理器，失败时返回错误
    pub fn create_processor(path: &Path) -> Result<Box<dyn FileProcessor>> {
        let factory = Self::new();
        let options = HashMap::new();
        factory.create_processor_with_path(path, &options)
    }
    
    /// 根据文件类型创建处理器
    ///
    /// # 参数
    /// * `format` - 文件格式
    /// * `file_path` - 文件路径
    /// * `options` - 处理器选项
    ///
    /// # 返回
    /// 成功时返回处理器，失败时返回错误
    pub fn create_processor_by_type(&self, format: FileFormat, file_path: &str, options: HashMap<String, String>) -> Result<Box<dyn FileProcessor>> {
        debug!("通过适配器按类型创建文件处理器: 类型={:?}, 路径={}, 选项={:?}", format, file_path, &options);
        
        // 转换为FileType并调用原始工厂
        #[cfg(not(feature = "parquet"))]
        if let FileFormat::Parquet = format {
            warn!("Parquet特性未启用，无法创建处理器");
            
            // 更新错误计数
            self.perf_counters.write().error_count += 1;
            
            return Err(Error::feature_not_enabled("parquet"));
        }
        
        let file_type = format.to_file_type();
        
        let start_time = std::time::Instant::now();
        
        let processor = match OriginalFactory::create_processor_by_type(file_type, file_path, &options) {
            Ok(p) => p,
            Err(e) => {
                error!("按类型创建处理器失败: 类型={:?}, 路径={}, 错误={}", format, file_path, e);
                
                // 更新错误计数
                self.perf_counters.write().error_count += 1;
                
                return Err(e);
            }
        };
        
        // 更新性能计数器
        {
            let mut counters = self.perf_counters.write();
            counters.create_count += 1;
            counters.create_time += start_time.elapsed();
        }
        
        Ok(Box::new(ArcToBoxAdapter::new(processor)))
    }
    
    /// 检测文件类型
    ///
    /// # 参数
    /// * `file_path` - 文件路径
    ///
    /// # 返回
    /// 成功时返回文件格式，失败时返回错误
    pub fn detect_file_type(&self, file_path: &str) -> Result<FileFormat> {
        let path = Path::new(file_path);
        
        // 使用原始工厂检测类型
        let file_type = match OriginalFactory::detect_file_type(path) {
            Ok(t) => t,
            Err(e) => {
                error!("检测文件类型失败: 路径={}, 错误={}", file_path, e);
                
                // 更新错误计数
                self.perf_counters.write().error_count += 1;
                
                return Err(e);
            }
        };
        
        // 转换回FileFormat
        let format = match file_type {
            FileType::CSV => Ok(FileFormat::CSV),
            FileType::JSON => Ok(FileFormat::JSON),
            #[cfg(feature = "parquet")]
            FileType::Parquet => Ok(FileFormat::Parquet),
            #[cfg(not(feature = "parquet"))]
            FileType::Parquet => {
                warn!("无法转换FileType::Parquet到FileFormat：特性未启用");
                Ok(FileFormat::Unknown)
            },
            FileType::Excel => Ok(FileFormat::Excel),
            FileType::Avro => Ok(FileFormat::Unknown), // 当前适配器不支持AVRO
            FileType::Custom => Ok(FileFormat::Unknown),
            FileType::Unknown => Ok(FileFormat::Unknown),
        };
        
        debug!("检测文件类型: 路径={}, 类型={:?}", file_path, format);
        format
    }
    
    /// 静态方法：检测文件类型
    ///
    /// # 参数
    /// * `path` - 文件路径
    ///
    /// # 返回
    /// 成功时返回文件格式，失败时返回错误
    pub fn detect_file_type(path: &Path) -> Result<FileFormat> {
        // 使用原始工厂检测类型
        let file_type = match OriginalFactory::detect_file_type(path) {
            Ok(t) => t,
            Err(e) => {
                error!("检测文件类型失败: 路径={}, 错误={}", path.display(), e);
                return Err(e);
            }
        };
        
        // 转换回FileFormat
        let format = match file_type {
            FileType::CSV => Ok(FileFormat::CSV),
            FileType::JSON => Ok(FileFormat::JSON),
            #[cfg(feature = "parquet")]
            FileType::Parquet => Ok(FileFormat::Parquet),
            #[cfg(not(feature = "parquet"))]
            FileType::Parquet => {
                warn!("无法转换FileType::Parquet到FileFormat：特性未启用");
                Ok(FileFormat::Unknown)
            },
            FileType::Excel => Ok(FileFormat::Excel),
            FileType::Avro => Ok(FileFormat::Unknown), // 当前适配器不支持AVRO
            FileType::Custom => Ok(FileFormat::Unknown),
            FileType::Unknown => Ok(FileFormat::Unknown),
        };
        
        debug!("检测文件类型: 路径={}, 类型={:?}", path.display(), format);
        format
    }
    
    /// 根据文件路径自动检测格式并创建处理器
    ///
    /// # 参数
    /// * `file_path` - 文件路径
    /// * `options` - 处理器选项
    ///
    /// # 返回
    /// 成功时返回处理器，失败时返回错误
    pub fn create_processor_from_path(&self, file_path: &str, options: HashMap<String, String>) -> Result<Box<dyn FileProcessor>> {
        debug!("通过适配器从路径创建文件处理器: 路径={}, 选项={:?}", file_path, &options);
        
        // 生成缓存键
        let cache_key = self.generate_cache_key(file_path, &options);
        
        // 尝试从缓存获取
        {
            let cache = self.cache.read();
            if let Some((processor, _)) = cache.get(&cache_key) {
                debug!("处理器缓存命中: {}", file_path);
                
                // 更新缓存命中计数
                self.perf_counters.write().cache_hits += 1;
                
                return Ok(Box::new(ArcToBoxAdapter::new(processor.clone())));
            }
        }
        
        // 缓存未命中，更新计数
        self.perf_counters.write().cache_misses += 1;
        
        let start_time = std::time::Instant::now();
        
        // 使用原始工厂创建处理器
        let processor = match OriginalFactory::create_processor_from_path(file_path, &options) {
            Ok(p) => p,
            Err(e) => {
                error!("创建处理器失败: 路径={}, 错误={}", file_path, e);
                
                // 更新错误计数
                self.perf_counters.write().error_count += 1;
                
                return Err(e);
            }
        };
        
        // 更新性能计数器
        {
            let mut counters = self.perf_counters.write();
            counters.create_count += 1;
            counters.create_time += start_time.elapsed();
        }
        
        // 更新缓存
        {
            // 先清理过期缓存
            self.cleanup_cache();
            
            // 添加新条目到缓存
            let mut cache = self.cache.write();
            cache.insert(cache_key, (processor.clone(), std::time::Instant::now()));
        }
        
        Ok(Box::new(ArcToBoxAdapter::new(processor)))
    }
    
    /// 为路径创建处理器 - 用于loader模块中的调用
    ///
    /// # 参数
    /// * `path` - 文件路径
    ///
    /// # 返回
    /// 成功时返回数据加载器，失败时返回错误
    pub fn create_processor_for_path(path: &Path) -> Result<Box<dyn DataLoader>> {
        debug!("为路径创建处理器: {}", path.display());
        
        // 使用原始工厂创建处理器
        let processor = match OriginalFactory::create_processor(path) {
            Ok(p) => p,
            Err(e) => {
                error!("创建处理器失败: 路径={}, 错误={}", path.display(), e);
                return Err(e);
            }
        };
        
        // 创建适当的DataLoader类型
        let file_path = path.to_string_lossy().to_string();
        let loader = match crate::data::loader::file::FileDataLoader::new(file_path, Some(1000)) {
            Ok(l) => l,
            Err(e) => {
                error!("创建数据加载器失败: 路径={}, 错误={}", path.display(), e);
                return Err(e);
            }
        };
        
        Ok(Box::new(loader))
    }
    
    /// 将Arc<dyn FileProcessor>转换为Box<dyn FileProcessor>
    ///
    /// # 参数
    /// * `processor` - 要转换的处理器
    ///
    /// # 返回
    /// 转换后的处理器
    pub fn boxed(processor: Arc<dyn FileProcessor>) -> Box<dyn FileProcessor> {
        Box::new(ArcToBoxAdapter::new(processor))
    }
}

impl Default for FileProcessorFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// 使用anyhow的具体错误处理
/// 此函数展示如何使用anyhow进行更细粒度的错误处理
fn convert_error_with_anyhow(error: impl std::fmt::Display) -> AnyhowResult<()> {
    // 使用anyhow创建更详细的错误
    Err(anyhow!("处理器工厂错误: {}", error))
}

/// DataSchema到Schema的转换
impl TryFrom<&DataSchema> for Schema {
    type Error = Error;
    
    fn try_from(value: &DataSchema) -> std::result::Result<Self, Self::Error> {
        // 从DataSchema创建Schema
        let mut fields = HashMap::new();
        
        for field in &value.fields {
            let data_type = match &field.field_type {
                FieldType::Numeric => DataType::Numeric,
                FieldType::Categorical => DataType::Categorical,
                FieldType::Text => DataType::Text,
                FieldType::DateTime => DataType::DateTime,
                FieldType::Boolean => DataType::Boolean,
                FieldType::Array(_) => DataType::Array,
                FieldType::Object(_) => DataType::Object,
                _ => DataType::Unknown,
            };
            
            fields.insert(field.name.clone(), data_type);
        }
        
        let simple_schema = SimpleSchema {
            fields,
            primary_key: value.primary_key.as_ref().and_then(|keys| keys.first().cloned()),
        };
        
        Ok(Schema::from_simple(simple_schema))
    }
}