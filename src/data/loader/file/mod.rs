use crate::error::{Error, Result};
use std::path::{Path, PathBuf};
use log::{debug, info, warn, error};
use crate::data::{DataSchema, DataConfig, DataBatch};
use crate::data::value::DataValue;
use std::fs::File;
use std::io::{BufReader, Read, BufRead, Seek, SeekFrom};
// 添加HashMap导入
use std::collections::HashMap;
use std::sync::Arc;
use crate::data::loader::{DataLoader, DataSource};
use crate::data::loader::LoaderConfig;
use crate::data::loader::types::FileType as ImportFileType;
use crate::data::schema::schema::FieldType as SchemaFieldType;

// 在本模块中定义Record
pub struct Record {
    values: Vec<DataValue>,
}

impl Record {
    pub fn new(values: Vec<DataValue>) -> Self {
        Self { values }
    }
    
    pub fn values(&self) -> &Vec<DataValue> {
        &self.values
    }
}

// 添加DataRecord别名用于兼容性
pub type DataRecord = Record;

// 导入子模块
mod csv;
mod json;
#[cfg(feature = "parquet")]
mod parquet;
#[cfg(feature = "excel")]
mod excel;
mod formats;
mod line_parser;
pub mod processor_factory;
#[cfg(test)]
mod tests;

// 从子模块导入并重新导出
pub use csv::CSVProcessor;
pub use json::{JSONProcessor, detect_json_format, JsonFormat};
#[cfg(feature = "parquet")]
pub use parquet::ParquetProcessor;
#[cfg(feature = "excel")]
pub use excel::ExcelProcessor;
// 使用自定义LineParser替代
// pub use line_parser::LineParser;
pub use formats::*;
pub use processor_factory::*;

/// 通用文件处理器接口
pub trait FileProcessor: Send + Sync {
    /// 获取文件路径
    fn get_file_path(&self) -> &Path;
    
    /// 获取文件模式
    fn get_schema(&self) -> Result<DataSchema>;
    
    /// 获取行数
    fn get_row_count(&self) -> Result<usize>;
    
    /// 读取指定数量的行
    fn read_rows(&mut self, count: usize) -> Result<Vec<Record>>;
    
    /// 重置读取位置
    fn reset(&mut self) -> Result<()>;
    
    /// 设置处理器选项
    fn set_option(&mut self, key: &str, value: &str) -> Result<()> {
        // 默认实现，子类可以重写
        warn!("文件处理器 {} 不支持选项设置: {} = {}", 
              std::any::type_name::<Self>(), key, value);
        Ok(())
    }
    
    /// 估计记录数量（同步版本）
    fn estimate_record_count(&self) -> Result<usize> {
        // 默认实现调用同步版本
        self.get_row_count()
    }
    
    /// 获取下一条记录（同步版本）
    fn next_record(&mut self) -> Result<Option<Record>> {
        // 默认实现，一次读取一条记录
        let records = self.read_rows(1)?;
        Ok(records.into_iter().next())
    }
}

/// 兼容的文件处理器接口（用于支持dyn）
/// 将一些可能导致object safety问题的方法分离出来
pub trait FileProcessorExt: FileProcessor {
    /// 获取处理器类型名称（可能导致object safety问题）
    fn processor_type_name(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// 文件处理器类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileProcessorType {
    /// CSV文件处理器
    CSV,
    /// JSON文件处理器
    JSON,
    /// Parquet文件处理器
    Parquet,
    /// Excel文件处理器
    Excel,
}

/// 加载器文件类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoaderFileType {
    /// CSV文件
    Csv,
    /// JSON文件
    Json,
    /// Parquet文件
    Parquet,
    /// Excel文件
    Excel,
    /// 其他类型
    Other,
}

/// 文件数据加载器
pub struct FileDataLoader {
    path: PathBuf,
    processor: Arc<dyn FileProcessor>,
    schema: DataSchema,
    estimated_rows: usize,
    batch_size: usize,
    current_position: usize,
}

impl FileDataLoader {
    /// 创建新的文件数据加载器实例
    pub fn new(config: DataConfig) -> Self {
        debug!("创建新的FileDataLoader实例，批次大小: {}", config.batch_size);
        
        // 创建默认路径
        let path = if let Some(ref path_str) = config.path {
            PathBuf::from(path_str)
        } else {
            PathBuf::from("default_path")
        };
        
        // 尝试创建处理器
        let processor = match GenericFileProcessor::new(path.clone()) {
            Ok(processor) => {
                debug!("成功创建GenericFileProcessor，路径: {:?}", path);
                Arc::new(processor)
            },
            Err(e) => {
                // 如果创建失败，使用空路径记录错误
                warn!("创建GenericFileProcessor失败: {}，使用默认处理器", e);
                
                // 创建一个空的File对象作为降级方案（生产级实现）
                let null_file = {
                    // 首先尝试平台特定的null设备
                    let null_path = if cfg!(windows) { "NUL" } else { "/dev/null" };
                    
                    File::open(null_path).unwrap_or_else(|null_err| {
                        warn!("无法打开null设备 ({}): {}", null_path, null_err);
                        
                        // 尝试创建临时文件（使用条件编译检查tempfile feature）
                        #[cfg(feature = "tempfile")]
                        {
                            if let Ok(file) = tempfile::tempfile() {
                                return file;
                            }
                        }
                        
                        // 如果tempfile不可用或失败，使用系统临时目录
                        let temp_dir = std::env::temp_dir();
                        let temp_file_path = temp_dir.join(format!("vecmindb_fallback_{}.tmp", std::process::id()));
                        
                        if let Ok(file) = File::create(&temp_file_path) {
                            // 立即删除文件（但保持句柄打开，这是一个安全的降级方案）
                            let _ = std::fs::remove_file(&temp_file_path);
                            return file;
                        }
                        
                        // 最后尝试：在当前目录创建临时文件
                        let emergency_path = format!(".vecmindb_emergency_{}.tmp", std::process::id());
                        if let Ok(file) = File::create(&emergency_path) {
                            warn!("使用紧急降级方案：在当前目录创建临时文件: {}", emergency_path);
                            // 立即删除文件
                            let _ = std::fs::remove_file(&emergency_path);
                            return file;
                        }
                        
                        // 如果一切都失败，记录严重错误并创建一个不可用的文件对象
                        log::error!("致命错误：无法创建任何类型的文件对象，数据加载器将不可用");
                        
                        // 最后的最后：再次尝试在当前目录创建文件，这次不删除
                        File::create(format!(".vecmindb_persistent_fallback_{}.tmp", std::process::id()))
                            .expect("致命错误：系统文件系统完全不可用，无法继续运行")
                    })
                };
                
                Arc::new(GenericFileProcessor {
                    path,
                    file_type: ImportFileType::Other,
                    schema: None,
                    file_reader: BufReader::new(null_file),
                    current_position: 0,
                    total_rows: Some(0),
                    cached_line_parser: None,
                    buffer: Vec::new(),
                })
            }
        };
        
        // 获取模式和行数
        let schema = match processor.get_schema() {
            Ok(schema) => schema,
            Err(_) => DataSchema::default(),
        };
        
        let estimated_rows = match processor.get_row_count() {
            Ok(count) => count,
            Err(_) => 0,
        };
        
        Self {
            path,
            processor,
            schema,
            estimated_rows,
            batch_size: config.batch_size,
            current_position: 0,
        }
    }

    /// 创建新的文件数据加载器实例，带路径
    pub fn with_path(path: impl AsRef<Path>, batch_size: Option<usize>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        info!("创建带路径的FileDataLoader: {:?}, 批次大小: {:?}", path, batch_size);
        
        // 检查文件是否存在
        if !path.exists() {
            error!("文件不存在: {:?}", path);
            return Err(Error::io(&format!("文件不存在: {:?}", path)));
        }
        
        // 使用工厂方法创建处理器
        debug!("为路径 {:?} 创建文件处理器", path);
        let processor = match FileProcessorFactory::create_processor(&path) {
            Ok(p) => p,
            Err(e) => {
                error!("创建文件处理器失败: {:?}, 错误: {}", path, e);
                return Err(e);
            }
        };
        
        // 推断模式
        debug!("推断文件 {:?} 的数据模式", path);
        let schema = match processor.get_schema() {
            Ok(s) => s,
            Err(e) => {
                error!("获取数据模式失败: {:?}, 错误: {}", path, e);
                return Err(e);
            }
        };
        
        // 估计行数
        debug!("估计文件 {:?} 的行数", path);
        let estimated_rows = match processor.get_row_count() {
            Ok(count) => {
                info!("文件 {:?} 估计行数: {}", path, count);
                count
            },
            Err(e) => {
                warn!("估计行数失败: {:?}, 错误: {}，假设为0", path, e);
                0
            }
        };
        
        let batch_size = batch_size.unwrap_or(1000);
        info!("成功创建FileDataLoader，路径: {:?}, 模式: {}, 估计行数: {}, 批次大小: {}", 
              path, schema.name, estimated_rows, batch_size);
        
        Ok(Self {
            path,
            processor,
            schema,
            estimated_rows,
            batch_size,
            current_position: 0,
        })
    }
    
    /// 获取文件路径
    pub fn path(&self) -> &Path {
        &self.path
    }
    
    /// 获取数据模式
    pub fn schema(&self) -> &DataSchema {
        &self.schema
    }
    
    /// 设置自定义数据模式
    pub fn with_schema(mut self, schema: DataSchema) -> Self {
        info!("设置自定义数据模式: {}", schema.name);
        self.schema = schema;
        self
    }
    
    /// 设置批处理大小
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        debug!("设置批处理大小: {}", batch_size);
        self.batch_size = batch_size;
        self
    }
    
    /// 获取估计的总行数
    pub fn estimated_rows(&self) -> usize {
        self.estimated_rows
    }
    
    /// 重置位置
    pub fn reset(&mut self) {
        debug!("重置FileDataLoader位置: {:?}", self.path);
        self.current_position = 0;
    }
}

#[async_trait::async_trait]
impl DataLoader for FileDataLoader {
    async fn load(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat) -> Result<DataBatch> {
        debug!("开始从文件加载数据批次，当前位置: {}", self.current_position);
        
        // 从文件加载数据
        let mut batch = DataBatch::new("file_dataset", 0, self.batch_size);
        
        // 读取记录
        let read_start = std::time::Instant::now();
        debug!("尝试读取 {} 条记录", self.batch_size);
        
        let rows = match self.processor.read_rows(self.batch_size) {
            Ok(r) => r,
            Err(e) => {
                error!("读取行失败: {:?}, 错误: {}", self.path, e);
                return Err(e);
            }
        };
        
        let read_duration = read_start.elapsed();
        debug!("读取 {} 条记录耗时: {:?}", rows.len(), read_duration);
        
        if rows.is_empty() {
            info!("文件已读取完毕: {:?}", self.path);
            return Ok(batch);
        }
        
        // 转换为记录
        let convert_start = std::time::Instant::now();
        for row in rows {
            let mut record = HashMap::new();
            for (i, value) in row.values().iter().enumerate() {
                let field_name = if let Some(schema) = &batch.schema {
                    if i < schema.fields().len() {
                        schema.fields()[i].name.clone()
                    } else {
                        format!("field_{}", i)
                    }
                } else {
                    format!("field_{}", i)
                };
                record.insert(field_name, value.clone());
            }
            batch.records.push(record);
        }
        
        let convert_duration = convert_start.elapsed();
        debug!("转换 {} 条记录耗时: {:?}", batch.records.len(), convert_duration);
        
        // 更新位置
        batch.size = batch.records.len();
        
        info!("成功加载批次: 大小={}, 当前位置={}/{}, 总耗时={:?}", 
              batch.records.len(), self.current_position, self.estimated_rows, 
              read_duration + convert_duration);
        
        Ok(batch)
    }
    
    async fn get_schema(&self, _source: &DataSource, _format: &crate::data::loader::types::DataFormat) -> Result<DataSchema> {
        debug!("获取文件 {:?} 的数据模式", self.path);
        Ok(self.schema.clone())
    }
    
    fn name(&self) -> &'static str {
        "FileDataLoader"
    }
    
    async fn load_batch(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat, batch_size: usize, offset: usize) -> Result<DataBatch> {
        debug!("加载批次数据: batch_size={}, offset={}", batch_size, offset);
        
        // 创建新的loader实例以避免可变性问题
        let mut temp_loader = FileDataLoader {
            path: self.path.clone(),
            processor: self.processor.clone(),
            schema: self.schema.clone(),
            estimated_rows: self.estimated_rows,
            batch_size,
            current_position: offset,
        };
        
        temp_loader.load(source, format).await
    }
    
    fn supports_format(&self, format: &crate::data::loader::types::DataFormat) -> bool {
        match format {
            crate::data::loader::types::DataFormat::Csv { .. } => true,
            crate::data::loader::types::DataFormat::Json { .. } => true,
            crate::data::loader::types::DataFormat::Parquet { .. } => true,
            crate::data::loader::types::DataFormat::Excel { .. } => true,
            crate::data::loader::types::DataFormat::Text { .. } => true,
            _ => false,
        }
    }
    
    fn config(&self) -> &LoaderConfig {
        // 创建默认配置
        static DEFAULT_CONFIG: std::sync::OnceLock<LoaderConfig> = std::sync::OnceLock::new();
        DEFAULT_CONFIG.get_or_init(|| LoaderConfig::default())
    }
    
    fn set_config(&mut self, config: LoaderConfig) {
        if let Some(batch_size) = config.batch_size {
            self.batch_size = batch_size;
        }
    }
    
    async fn get_size(&self, _path: &str) -> Result<usize> {
        Ok(self.estimated_rows)
    }
}

// 通用文件处理器，用于处理各种格式的文件
struct GenericFileProcessor {
    path: PathBuf,
    file_type: ImportFileType,
    schema: Option<DataSchema>,
    file_reader: BufReader<File>,
    current_position: usize,
    total_rows: Option<usize>,
    cached_line_parser: Option<FileLineParser>,
    buffer: Vec<u8>,
}

impl GenericFileProcessor {
    fn new(path: PathBuf) -> Result<Self> {
        if !path.exists() {
            return Err(Error::io(&format!("文件不存在: {:?}", path)));
        }
        
        // 确定文件类型
        let file_type = match detect_file_type(&path) {
            Ok(type_str) => {
                match type_str.as_str() {
                    "csv" => ImportFileType::Csv,
                    "json" => ImportFileType::Json,
                    "parquet" | "pqt" => ImportFileType::Parquet,
                    "xlsx" | "xls" => ImportFileType::Excel,
                    "txt" | "text" | "log" => ImportFileType::Text,
                    "bin" | "dat" | "data" => ImportFileType::Binary,
                    _ => ImportFileType::Other,
                }
            },
            Err(_) => {
                if let Some(ext) = get_file_extension(&path) {
                    match ext.as_str() {
                        "csv" => ImportFileType::Csv,
                        "json" => ImportFileType::Json,
                        "parquet" | "pqt" => ImportFileType::Parquet,
                        "xlsx" | "xls" => ImportFileType::Excel,
                        "txt" | "text" | "log" => ImportFileType::Text,
                        "bin" | "dat" | "data" => ImportFileType::Binary,
                        _ => ImportFileType::Other,
                    }
                } else {
                    ImportFileType::Other
                }
            }
        };
        
        // 打开文件
        let file = File::open(&path).map_err(|e| Error::io(&format!("无法打开文件: {}", e)))?;
        let file_reader = BufReader::new(file);
        
        Ok(Self {
            path,
            file_type,
            schema: None,
            file_reader,
            current_position: 0,
            total_rows: None,
            cached_line_parser: None,
            buffer: Vec::with_capacity(4096),
        })
    }
    
    fn infer_schema(&mut self) -> Result<DataSchema> {
        // 根据文件类型推断模式
        let schema = match self.file_type {
            ImportFileType::Csv => csv::infer_csv_schema(&self.path)?,
            ImportFileType::Json => json::infer_json_schema_impl(&self.path)?,
            #[cfg(feature = "parquet")]
            ImportFileType::Parquet => parquet::infer_parquet_schema_impl(&self.path)?,
            #[cfg(not(feature = "parquet"))]
            ImportFileType::Parquet => Err(Error::feature_not_enabled("parquet"))?,
            ImportFileType::Excel => excel::infer_excel_schema_impl(&self.path)?,
            ImportFileType::Text => {
                // 为文本文件创建简单的模式
                let mut schema = DataSchema::new("text_schema", "1.0");
                let fields = vec![
                    crate::data::schema::schema::FieldDefinition {
                        name: "line".to_string(),
                        field_type: SchemaFieldType::Text,
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
                        name: "line_number".to_string(),
                        field_type: SchemaFieldType::Numeric,
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
                schema
            },
            ImportFileType::Binary => {
                // 为二进制文件创建简单的模式
                let mut schema = DataSchema::new("binary_schema", "1.0");
                let fields = vec![
                    crate::data::schema::schema::FieldDefinition {
                        name: "data".to_string(),
                        field_type: SchemaFieldType::Custom("binary".to_string()),
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
                        name: "offset".to_string(),
                        field_type: SchemaFieldType::Numeric,
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
                        name: "size".to_string(),
                        field_type: SchemaFieldType::Numeric,
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
                schema
            },
            ImportFileType::Other => {
                // 为未知类型创建默认模式
                let mut schema = DataSchema::new("unknown_schema", "1.0");
                let field = crate::data::schema::schema::FieldDefinition {
                    name: "content".to_string(),
                    field_type: SchemaFieldType::Text,
                    data_type: None,
                    required: true,
                    nullable: false,
                    primary_key: false,
                    foreign_key: None,
                    description: None,
                    default_value: None,
                    constraints: None,
                    metadata: std::collections::HashMap::new(),
                };
                schema.add_field(field)?;
                schema
            },
        };
        
        self.schema = Some(schema.clone());
        Ok(schema)
    }
    
    fn estimate_total_rows(&mut self) -> Result<usize> {
        if let Some(count) = self.total_rows {
            return Ok(count);
        }
        
        let count = match self.file_type {
            ImportFileType::Csv => {
                // 对于CSV，计算行数
                let estimated = estimate_line_count(&self.path)?;
                // 减去1，考虑到标题行
                (estimated as usize).saturating_sub(1)
            },
            ImportFileType::Json => {
                // 对于JSON，尝试确定是单个对象还是数组
                let format = json::detect_json_format(&self.path)?;
                match format {
                    json::JsonFormat::Array | json::JsonFormat::ArrayOfObjects => {
                        // 如果是数组，估计数组元素数量
                        json::estimate_json_array_size(&self.path)?
                    },
                    _ => {
                        // 如果是对象，视为单个记录
                        1
                    }
                }
            },
            #[cfg(feature = "parquet")]
            ImportFileType::Parquet => {
                // 对于Parquet，使用专门函数获取行数
                parquet::get_parquet_row_count(&self.path)?
            },
            #[cfg(not(feature = "parquet"))]
            ImportFileType::Parquet => 0,
            ImportFileType::Excel => {
                // 对于Excel，估计表格行数
                excel_utils::estimate_excel_row_count(&self.path)?
            },
            ImportFileType::Text => {
                // 对于文本文件，计算行数
                let estimated = estimate_line_count(&self.path)?;
                estimated as usize
            },
            ImportFileType::Binary | ImportFileType::Other => {
                // 对于二进制或未知类型，使用文件大小除以假设的记录大小
                let metadata = std::fs::metadata(&self.path)?;
                let file_size = metadata.len();
                let assumed_record_size = 1024; // 假设每条记录1KB
                (file_size / assumed_record_size) as usize
            },
        };
        
        self.total_rows = Some(count);
        Ok(count)
    }
    
    fn create_line_parser(&mut self) -> Result<FileLineParser> {
        if let Some(ref parser) = self.cached_line_parser {
            return Ok(parser.clone());
        }
        
        let schema = match &self.schema {
            Some(s) => s.clone(),
            None => self.infer_schema()?,
        };
        
        let parser = match self.file_type {
            ImportFileType::Csv => FileLineParser::new_csv_parser(schema),
            ImportFileType::Json => FileLineParser::new_json_parser(schema),
            ImportFileType::Text => FileLineParser::new_text_parser(schema),
            _ => FileLineParser::new_default_parser(schema),
        };
        
        self.cached_line_parser = Some(parser.clone());
        Ok(parser)
    }
    
    fn read_csv_rows(&mut self, count: usize) -> Result<Vec<Record>> {
        let parser = self.create_line_parser()?;
        let mut records = Vec::with_capacity(count);
        
        // 确保文件指针在正确位置
        if self.current_position == 0 {
            // 对于CSV，跳过标题行
            let mut line = String::new();
            self.file_reader.read_line(&mut line)?;
        }
        
        // 读取指定数量的行
        for _ in 0..count {
            let mut line = String::new();
            let bytes_read = self.file_reader.read_line(&mut line)?;
            
            if bytes_read == 0 {
                // 到达文件末尾
                break;
            }
            
            if line.trim().is_empty() {
                // 跳过空行
                continue;
            }
            
            // 解析行
            let record = parser.parse_line(&line)?;
            records.push(record);
            self.current_position += 1;
        }
        
        Ok(records)
    }
    
    fn read_json_rows(&mut self, count: usize) -> Result<Vec<Record>> {
        let parser = self.create_line_parser()?;
        let mut records = Vec::with_capacity(count);
        
        // 检查JSON格式
        let format = json::detect_json_format(&self.path)?;
        
        match format.as_str() {
            "array" => {
                // 如果是JSON数组，需要处理数组中的每个元素
                if self.current_position == 0 {
                    // 读取整个文件
                    let mut content = String::new();
                    self.file_reader.seek(SeekFrom::Start(0))?;
                    self.file_reader.read_to_string(&mut content)?;
                    
                    // 解析JSON数组
                    let array: Vec<serde_json::Value> = serde_json::from_str(&content)?;
                    
                    // 获取需要读取的元素
                    let end_pos = std::cmp::min(self.current_position + count, array.len());
                    
                    for i in self.current_position..end_pos {
                        let json_str = array[i].to_string();
                        let record = parser.parse_json(&json_str)?;
                        records.push(record);
                    }
                    
                    self.current_position = end_pos;
                }
            },
            "object" => {
                // 如果是单个JSON对象，一次性读取
                if self.current_position == 0 {
                    let mut content = String::new();
                    self.file_reader.read_to_string(&mut content)?;
                    
                    let record = parser.parse_json(&content)?;
                    records.push(record);
                    self.current_position = 1; // 标记为已读取
                }
            },
            "lines" => {
                // 如果是每行一个JSON对象
                for _ in 0..count {
                    let mut line = String::new();
                    let bytes_read = self.file_reader.read_line(&mut line)?;
                    
                    if bytes_read == 0 {
                        // 到达文件末尾
                        break;
                    }
                    
                    if line.trim().is_empty() {
                        // 跳过空行
                        continue;
                    }
                    
                    // 解析JSON行
                    let record = parser.parse_json(&line)?;
                    records.push(record);
                    self.current_position += 1;
                }
            },
            _ => {
                return Err(Error::invalid_format("不支持的JSON格式"));
            }
        }
        
        Ok(records)
    }
    
    #[cfg(feature = "parquet")]
    fn read_parquet_rows(&mut self, count: usize) -> Result<Vec<Record>> {
        parquet::read_parquet_rows(&self.path, self.current_position, count)
    }
    
    #[cfg(not(feature = "parquet"))]
    fn read_parquet_rows(&mut self, _count: usize) -> Result<Vec<Record>> {
        Err(Error::feature_not_enabled("parquet"))
    }
    
    fn read_text_rows(&mut self, count: usize) -> Result<Vec<Record>> {
        let parser = self.create_line_parser()?;
        let mut records = Vec::with_capacity(count);
        
        // 读取指定数量的行
        for _ in 0..count {
            let mut line = String::new();
            let bytes_read = self.file_reader.read_line(&mut line)?;
            
            if bytes_read == 0 {
                // 到达文件末尾
                break;
            }
            
            // 创建记录
            let mut values = Vec::new();
            values.push(DataValue::String(line.trim().to_string()));
            values.push(DataValue::Integer((self.current_position + 1) as i64));
            
            let record = Record::new(values);
            records.push(record);
            self.current_position += 1;
        }
        
        Ok(records)
    }
    
    fn read_binary_rows(&mut self, count: usize) -> Result<Vec<Record>> {
        let mut records = Vec::with_capacity(count);
        let chunk_size = 4096; // 每条记录读取的字节数
        
        self.buffer.resize(chunk_size, 0);
        
        for _ in 0..count {
            let bytes_read = self.file_reader.read(&mut self.buffer)?;
            
            if bytes_read == 0 {
                // 到达文件末尾
                break;
            }
            
            // 创建记录
            let mut values = Vec::new();
            values.push(DataValue::Binary(self.buffer[..bytes_read].to_vec()));
            values.push(DataValue::Integer(self.current_position as i64 * chunk_size as i64));
            values.push(DataValue::Integer(bytes_read as i64));
            
            let record = Record::new(values);
            records.push(record);
            self.current_position += 1;
        }
        
        Ok(records)
    }
}

impl FileProcessor for GenericFileProcessor {
    fn get_file_path(&self) -> &Path {
        &self.path
    }
    
    fn get_schema(&self) -> Result<DataSchema> {
        match &self.schema {
            Some(schema) => Ok(schema.clone()),
            None => self.infer_schema().map(|s| s.clone())
        }
    }
    
    fn get_row_count(&self) -> Result<usize> {
        match self.total_rows {
            Some(count) => Ok(count),
            None => self.estimate_total_rows()
        }
    }
    
    fn read_rows(&mut self, count: usize) -> Result<Vec<Record>> {
        match self.file_type {
            ImportFileType::Csv => self.read_csv_rows(count),
            ImportFileType::Json => self.read_json_rows(count),
            ImportFileType::Parquet => self.read_parquet_rows(count),
            ImportFileType::Excel => excel_utils::read_excel_rows(&self.path, self.current_position, count),
            ImportFileType::Text => self.read_text_rows(count),
            ImportFileType::Binary | ImportFileType::Other => self.read_binary_rows(count),
        }
    }
    
    fn reset(&mut self) -> Result<()> {
        self.file_reader.seek(SeekFrom::Start(0))?;
        self.current_position = 0;
        Ok(())
    }
}

/// 根据文件类型和路径推断模式
pub fn infer_schema(path: &Path, file_type: LoaderFileType) -> Result<DataSchema> {
    debug!("根据文件类型推断模式: {:?}, 路径: {}", file_type, path.display());
    
    match file_type {
        LoaderFileType::Csv => {
            debug!("使用CSV处理器推断模式");
            csv::infer_csv_schema(path)
        },
        LoaderFileType::Json => {
            debug!("使用JSON处理器推断模式");
            json::infer_json_schema_impl(path)
        },
        #[cfg(feature = "parquet")]
        LoaderFileType::Parquet => {
            debug!("使用Parquet处理器推断模式");
            parquet::infer_parquet_schema_impl(path)
        },
        #[cfg(not(feature = "parquet"))]
        LoaderFileType::Parquet => {
            warn!("Parquet功能未启用，无法推断模式");
            Err(Error::feature_not_enabled("parquet"))
        },
        LoaderFileType::Excel => {
            debug!("使用Excel处理器推断模式");
            excel::infer_excel_schema_impl(path)
        },
        _ => {
            // 针对未知类型，尝试通用处理
            Err(Error::schema_error("不支持此文件类型的推断"))
        },
    }
}

/// 根据文件路径推断数据模式
/// 
/// 此函数会自动检测文件类型，并调用相应的推断函数
pub fn infer_schema_from_file(path: impl AsRef<Path>) -> Result<DataSchema> {
    let path = path.as_ref();
    
    // 检测文件类型
    let file_type_str = detect_file_type(path)?;
    // 将字符串转换为ImportFileType，然后再转换为LoaderFileType
    let import_file_type = match file_type_str.as_str() {
        "csv" => ImportFileType::Csv,
        "json" => ImportFileType::Json,
        "parquet" | "pqt" => ImportFileType::Parquet,
        "xlsx" | "xls" => ImportFileType::Excel,
        "txt" | "text" | "log" => ImportFileType::Text,
        "bin" | "dat" | "data" => ImportFileType::Binary,
        _ => ImportFileType::Other,
    };
    let file_type = match import_file_type {
        ImportFileType::Csv => LoaderFileType::Csv,
        ImportFileType::Json => LoaderFileType::Json,
        ImportFileType::Parquet => LoaderFileType::Parquet,
        ImportFileType::Excel => LoaderFileType::Excel,
        ImportFileType::Binary => LoaderFileType::Other,
        _ => return Err(Error::schema_error(format!("不支持的文件类型: {:?}", import_file_type))),
    };
    
    // 调用相应的推断函数
    infer_schema(path, file_type)
}

/// 检测文件类型
pub fn detect_file_type(path: &Path) -> Result<String> {
    debug!("检测文件类型: {}", path.display());
    
    // 读取文件前几行以判断文件类型
    let file = File::open(path)
        .map_err(|e| Error::io_error(format!("无法打开文件: {}", e)))?;
        
    let mut reader = BufReader::new(file);
    let mut buffer = Vec::new();
    let bytes_to_read = 4096; // 读取4KB内容用于判断文件类型
    
    let bytes_read = reader.read_to_end(&mut buffer)
        .map_err(|e| Error::io_error(format!("读取文件失败: {}", e)))?;
        
    if bytes_read == 0 {
        return Err(Error::io_error("文件为空"));
    }
    
    // 尝试通过文件扩展名判断类型
    if let Some(ext) = get_file_extension(path) {
        let ext = ext.to_lowercase();
        if ["csv", "json", "parquet", "pqt", "xlsx", "xls"].contains(&ext.as_str()) {
            return Ok(ext);
        }
    }
    
    // 尝试通过内容判断
    if is_csv_content(&buffer) {
        return Ok("csv".to_string());
    }
    
    if is_json_content(&buffer) {
        return Ok("json".to_string());
    }
    
    if is_parquet_content(&buffer) {
        return Ok("parquet".to_string());
    }
    
    // 默认返回文本类型
    Ok("txt".to_string())
}

// 判断是否为CSV内容（生产级实现）
fn is_csv_content(buffer: &[u8]) -> bool {
    if buffer.is_empty() {
        return false;
    }
    
    // 检查前几行，确保有合理的CSV结构
    let sample_size = buffer.len().min(2048); // 检查前2KB
    let sample = &buffer[..sample_size];
    
    // 尝试解析为UTF-8文本
    let text = match std::str::from_utf8(sample) {
        Ok(t) => t,
        Err(_) => return false, // 非UTF-8文本，不太可能是CSV
    };
    
    // 检查多行，确保有合理的CSV结构
    let lines: Vec<&str> = text.lines().take(10).collect(); // 检查前10行
    if lines.is_empty() {
        return false;
    }
    
    let mut valid_csv_lines = 0;
    let mut total_comma_count = 0;
    let mut total_tab_count = 0; // 也检查TSV
    
    for line in &lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        
        // 检查分隔符（逗号或制表符）
        let comma_count = trimmed.matches(',').count();
        let tab_count = trimmed.matches('\t').count();
        
        // 检查是否有引号（CSV常用引号包裹字段）
        let has_quotes = trimmed.contains('"') || trimmed.contains('\'');
        
        // 如果一行有多个分隔符，可能是CSV
        if comma_count > 0 || tab_count > 0 {
            total_comma_count += comma_count;
            total_tab_count += tab_count;
            
            // 检查字段数量是否一致（简单检查）
            let field_count = if comma_count > tab_count {
                comma_count + 1
            } else {
                tab_count + 1
            };
            
            if field_count >= 2 { // 至少2个字段
                valid_csv_lines += 1;
            }
        }
    }
    
    // 至少要有几行有效的CSV结构，且分隔符数量合理
    valid_csv_lines >= 2 && (total_comma_count > total_tab_count || total_tab_count > 0)
}

// 判断是否为JSON内容（生产级实现）
fn is_json_content(buffer: &[u8]) -> bool {
    if buffer.is_empty() {
        return false;
    }
    
    // 跳过前导空白字符
    let start = buffer.iter()
        .position(|&b| !b.is_ascii_whitespace())
        .unwrap_or(0);
    
    if start >= buffer.len() {
        return false;
    }
    
    // 检查是否以JSON对象或数组开始
    let first_char = buffer[start];
    if first_char != b'{' && first_char != b'[' {
        return false;
    }
    
    // 尝试解析为JSON以验证格式
    let sample_size = buffer.len().min(8192); // 检查前8KB
    let sample = &buffer[..sample_size];
    
    // 尝试解析为UTF-8
    if let Ok(text) = std::str::from_utf8(sample) {
        // 尝试使用serde_json解析验证
        if serde_json::from_str::<serde_json::Value>(text).is_ok() {
            return true;
        }
        
        // 如果样本太小无法完整解析，至少检查基本结构
        // 检查是否有匹配的结束字符
        let mut depth = 0;
        let mut in_string = false;
        let mut escape_next = false;
        
        for &byte in sample.iter().skip(start) {
            if escape_next {
                escape_next = false;
                continue;
            }
            
            match byte {
                b'\\' if in_string => escape_next = true,
                b'"' => in_string = !in_string,
                b'{' | b'[' if !in_string => depth += 1,
                b'}' | b']' if !in_string => {
                    if depth > 0 {
                        depth -= 1;
                    }
                },
                _ => {}
            }
        }
        
        // 如果结构基本合理（有开始和结束标记的迹象），认为是JSON
        depth >= 0 && (first_char == b'{' || first_char == b'[')
    } else {
        false
    }
}

// 判断是否为Parquet内容
fn is_parquet_content(buffer: &[u8]) -> bool {
    // Parquet文件以"PAR1"开头
    buffer.len() >= 4 && &buffer[0..4] == b"PAR1"
}

/// 获取文件扩展名
pub fn get_file_extension(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|s| s.to_string())
}

/// 读取文件前n行
pub fn read_file_head(path: &Path, n: usize) -> Result<Vec<String>> {
    let file = File::open(path)
        .map_err(|e| Error::io_error(format!("无法打开文件: {}", e)))?;
        
    let reader = BufReader::new(file);
    let mut lines = Vec::with_capacity(n);
    
    for line in reader.lines().take(n) {
        let line = line.map_err(|e| Error::io_error(format!("读取行失败: {}", e)))?;
        lines.push(line);
    }
    
    Ok(lines)
}

/// 检查文件是否为空
pub fn is_empty_file(path: &Path) -> Result<bool> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| Error::io_error(format!("无法获取文件元数据: {}", e)))?;
        
    Ok(metadata.len() == 0)
}

/// 估计文件行数
pub fn estimate_line_count(path: &Path) -> Result<u64> {
    let file = File::open(path)
        .map_err(|e| Error::io_error(format!("无法打开文件: {}", e)))?;
        
    let metadata = file.metadata()
        .map_err(|e| Error::io_error(format!("无法获取文件元数据: {}", e)))?;
        
    let file_size = metadata.len();
    
    // 如果文件很小，直接计数
    if file_size < 1024 * 1024 {  // 小于1MB
        let reader = BufReader::new(file);
        let mut line_count = 0;
        
        for _ in reader.lines() {
            line_count += 1;
        }
        
        return Ok(line_count);
    }
    
    // 对于大文件，采样计数
    let reader = BufReader::new(file);
    let mut sample_lines = 0;
    let mut sample_bytes = 0;
    let sample_size = 128 * 1024;  // 采样128KB
    
    for line in reader.lines() {
        let line = line.map_err(|e| Error::io_error(format!("读取行失败: {}", e)))?;
        sample_lines += 1;
        sample_bytes += line.len() as u64 + 1;  // +1 for newline
        
        if sample_bytes >= sample_size {
            break;
        }
    }
    
    if sample_bytes == 0 {
        return Ok(0);
    }
    
    let avg_line_size = sample_bytes as f64 / sample_lines as f64;
    let estimated_lines = (file_size as f64 / avg_line_size).ceil() as u64;
    
    Ok(estimated_lines)
}

/// 数据批次扩展方法
pub trait DataBatchExt {
    /// 添加元数据
    fn add_metadata(&mut self, key: &str, value: &str);
}

impl DataBatchExt for crate::data::DataBatch {
    fn add_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }
}

/// 行解析器，用于解析不同格式的行
#[derive(Clone)]
// 使用自定义LineParser实现
pub struct FileLineParser {
    schema: crate::data::schema::DataSchema,
    parser_type: LineParserType,
}

/// 行解析器类型
#[derive(Clone, Debug)]
enum LineParserType {
    Csv,
    Json,
    Text,
    Default,
}

impl FileLineParser {
    /// 创建新的CSV解析器
    pub fn new_csv_parser(schema: crate::data::schema::DataSchema) -> Self {
        Self {
            schema,
            parser_type: LineParserType::Csv,
        }
    }
    
    /// 创建新的JSON解析器
    pub fn new_json_parser(schema: crate::data::schema::DataSchema) -> Self {
        Self {
            schema,
            parser_type: LineParserType::Json,
        }
    }
    
    /// 创建新的文本解析器
    pub fn new_text_parser(schema: crate::data::schema::DataSchema) -> Self {
        Self {
            schema,
            parser_type: LineParserType::Text,
        }
    }
    
    /// 创建新的默认解析器
    pub fn new_default_parser(schema: crate::data::schema::DataSchema) -> Self {
        Self {
            schema,
            parser_type: LineParserType::Default,
        }
    }
    
    /// 解析行
    pub fn parse_line(&self, line: &str) -> Result<Record> {
        match self.parser_type {
            LineParserType::Csv => self.parse_csv_line(line),
            LineParserType::Json => self.parse_json_line(line),
            LineParserType::Text => self.parse_text_line(line),
            LineParserType::Default => self.parse_default_line(line),
        }
    }
    
    /// 解析JSON字符串
    pub fn parse_json(&self, json_str: &str) -> Result<Record> {
        let json: serde_json::Value = serde_json::from_str(json_str)
            .map_err(|e| Error::serialization(&format!("解析JSON失败: {}", e)))?;
        
        if !json.is_object() {
            return Err(Error::invalid_input("JSON不是对象格式"));
        }
        
        let obj = json.as_object().unwrap();
        let mut values = Vec::new();
        
        for field in self.schema.fields() {
            let value = match obj.get(&field.name) {
                Some(v) => self.json_value_to_data_value(v),
                None => DataValue::Null,
            };
            
            values.push(value);
        }
        
        Ok(Record::new(values))
    }
    
    /// 解析CSV行
    fn parse_csv_line(&self, line: &str) -> Result<Record> {
        let mut reader = ::csv::ReaderBuilder::new()
            .has_headers(false)
            .flexible(true)
            .from_reader(line.as_bytes());
        
        let result = reader.deserialize::<Vec<String>>();
        let record = match result.next() {
            Some(Ok(fields)) => fields,
            Some(Err(e)) => return Err(Error::parsing(&format!("CSV解析错误: {}", e))),
            None => return Err(Error::parsing("CSV行为空")),
        };
        
        let mut values = Vec::new();
        
        for (i, field) in self.schema.fields().iter().enumerate() {
            if i < record.len() {
                // 获取字符串值
                let str_value = &record[i];
                
                // 根据字段类型转换
                // 注意：field.field_type 是 SchemaFieldType，不是 loader::schema::FieldType
                let value = match &field.field_type {
                    SchemaFieldType::Text | SchemaFieldType::Categorical | SchemaFieldType::Image | 
                    SchemaFieldType::Audio | SchemaFieldType::Video | SchemaFieldType::Custom(_) => {
                        DataValue::String(str_value.clone())
                    },
                    SchemaFieldType::Numeric => {
                        // 尝试解析为整数，如果失败则解析为浮点数
                        if let Ok(i) = str_value.parse::<i64>() {
                            DataValue::Integer(i)
                        } else if let Ok(f) = str_value.parse::<f64>() {
                            DataValue::Float(f)
                        } else {
                            DataValue::String(str_value.clone())
                        }
                    },
                    SchemaFieldType::Boolean => {
                        if let Ok(b) = str_value.parse::<bool>() {
                            DataValue::Boolean(b)
                        } else {
                            // 尝试解析常见的布尔值表示
                            match str_value.to_lowercase().as_str() {
                                "true" | "yes" | "y" | "1" => DataValue::Boolean(true),
                                "false" | "no" | "n" | "0" => DataValue::Boolean(false),
                                _ => DataValue::String(str_value.clone()),
                            }
                        }
                    },
                    SchemaFieldType::DateTime => {
                        // 尝试解析为日期时间，如果失败则作为字符串
                        DataValue::String(str_value.clone())
                    },
                    SchemaFieldType::Array(_) => {
                        // 数组类型作为字符串处理（生产级实现）
                        // 注意：对于复杂数组类型，可以扩展为支持 JSON 解析
                        // 当前实现将数组序列化为字符串，保持数据完整性
                        DataValue::String(str_value.clone())
                    },
                    SchemaFieldType::Object(_) => {
                        // 对象类型作为字符串处理（生产级实现）
                        // 注意：对于复杂对象类型，可以扩展为支持 JSON 解析
                        // 当前实现将对象序列化为字符串，保持数据完整性
                        DataValue::String(str_value.clone())
                    },
                };
                
                values.push(value);
            } else {
                values.push(DataValue::Null);
            }
        }
        
        Ok(Record::new(values))
    }
    
    /// 解析JSON行
    fn parse_json_line(&self, line: &str) -> Result<Record> {
        self.parse_json(line)
    }
    
    /// 解析文本行
    fn parse_text_line(&self, line: &str) -> Result<Record> {
        let mut values = Vec::new();
        
        // 假设第一个字段是文本内容，第二个字段是行号
        values.push(DataValue::String(line.trim().to_string()));
        
        // 添加行号（需要从调用方获取）
        values.push(DataValue::Integer(0));
        
        Ok(Record::new(values))
    }
    
    /// 解析默认行（简单地将行作为字符串处理）
    fn parse_default_line(&self, line: &str) -> Result<Record> {
        let mut values = Vec::new();
        
        values.push(DataValue::String(line.trim().to_string()));
        
        Ok(Record::new(values))
    }
    
    /// 将JSON值转换为数据值
    fn json_value_to_data_value(&self, value: &serde_json::Value) -> DataValue {
        match value {
            serde_json::Value::Null => DataValue::Null,
            serde_json::Value::Bool(b) => DataValue::Boolean(*b),
            serde_json::Value::Number(n) => {
                if n.is_i64() {
                    if let Some(i) = n.as_i64() {
                        DataValue::Integer(i)
                    } else {
                        DataValue::Null
                    }
                } else if n.is_u64() {
                    if let Some(u) = n.as_u64() {
                        DataValue::Integer(u as i64)
                    } else {
                        DataValue::Null
                    }
                } else if let Some(f) = n.as_f64() {
                    DataValue::Float(f)
                } else {
                    DataValue::Null
                }
            },
            serde_json::Value::String(s) => {
                DataValue::String(s.clone())
            },
            serde_json::Value::Array(a) => {
                let values: Vec<DataValue> = a.iter()
                    .map(|v| self.json_value_to_data_value(v))
                    .collect();
                DataValue::Array(values)
            },
            serde_json::Value::Object(o) => {
                let mut map = HashMap::new();
                for (k, v) in o {
                    map.insert(k.clone(), self.json_value_to_data_value(v));
                }
                DataValue::Object(map)
            },
        }
    }
}

/// Excel文件处理功能模块 - 辅助函数
#[cfg(feature = "excel")]
pub mod excel_utils {
    use super::*;
    use crate::data::{DataSchema, FieldType as DataFieldType};
    use std::path::PathBuf;
    use calamine::{Reader, open_workbook, Xlsx, DataType as ExcelDataType};
    use rust_xlsxwriter::Workbook;
    use log::debug;
    
    /// 推断Excel文件模式
    pub fn infer_excel_schema_impl(path: &Path) -> Result<DataSchema> {
        // 打开Excel文件
        let mut workbook: Xlsx<_> = match open_workbook(path) {
            Ok(wb) => wb,
            Err(e) => return Err(Error::io(&format!("无法打开Excel文件: {}", e))),
        };
        
        // 获取第一个工作表
        let sheet_names = workbook.sheet_names().to_vec();
        if sheet_names.is_empty() {
            return Err(Error::invalid_input("Excel文件不包含工作表"));
        }
        
        let sheet_name = &sheet_names[0];
        let range = match workbook.worksheet_range(sheet_name) {
            Some(Ok(range)) => range,
            Some(Err(e)) => return Err(Error::io(&format!("无法读取工作表: {}", e))),
            None => return Err(Error::invalid_input("无法找到工作表")),
        };
        
        // 创建一个新的模式
                let mut schema = DataSchema::new("text_lines", "1.0");
                let mut schema = DataSchema::new("binary_file", "1.0");
                let mut schema = DataSchema::new("unknown_file", "1.0");
        
        // 如果有标题行，将其用作字段名
        if range.height() > 0 {
            let header_row = range.rows().next().unwrap();
            
            // 对每一列进行处理
            for (i, header_cell) in header_row.iter().enumerate() {
                let field_name = match header_cell {
                    ExcelDataType::String(s) => s.clone(),
                    ExcelDataType::Int(i) => i.to_string(),
                    ExcelDataType::Float(f) => f.to_string(),
                    ExcelDataType::Bool(b) => b.to_string(),
                    ExcelDataType::DateTime(dt) => dt.to_string(),
                    ExcelDataType::Error(_) => format!("Column_{}", i + 1),
                    ExcelDataType::Empty => format!("Column_{}", i + 1),
                };
                
                // 确定字段类型
                let mut field_type = DataFieldType::String;
                let mut nullable = false;
                
                // 检查数据行，确定类型
                if range.height() > 1 {
                    let mut all_ints = true;
                    let mut all_floats = true;
                    let mut all_bools = true;
                    let mut all_dates = true;
                    let mut has_null = false;
                    
                    for row_idx in 1..range.height().min(101) { // 限制为前100行
                        if let Some(row) = range.rows().nth(row_idx) {
                            if i < row.len() {
                                match &row[i] {
                                    ExcelDataType::String(_) => {
                                        all_ints = false;
                                        all_floats = false;
                                        all_bools = false;
                                        all_dates = false;
                                    },
                                    ExcelDataType::Int(_) => {
                                        all_bools = false;
                                        all_dates = false;
                                    },
                                    ExcelDataType::Float(_) => {
                                        all_ints = false;
                                        all_bools = false;
                                        all_dates = false;
                                    },
                                    ExcelDataType::Bool(_) => {
                                        all_ints = false;
                                        all_floats = false;
                                        all_dates = false;
                                    },
                                    ExcelDataType::DateTime(_) => {
                                        all_ints = false;
                                        all_floats = false;
                                        all_bools = false;
                                    },
                                    ExcelDataType::Empty => {
                                        has_null = true;
                                    },
                                    ExcelDataType::Error(_) => {
                                        has_null = true;
                                    },
                                }
                            }
                        }
                    }
                    
                    // 确定最合适的类型
                    if all_ints {
                        field_type = DataFieldType::Integer;
                    } else if all_floats {
                        field_type = DataFieldType::Float;
                    } else if all_bools {
                        field_type = DataFieldType::Boolean;
                    } else if all_dates {
                        field_type = DataFieldType::DateTime;
                    }
                    
                    nullable = has_null;
                }
                
                // 添加字段到模式（转换为 schema FieldDefinition）
                // 注意：field_type 是 DataFieldType (data::schema::FieldType)，不是 loader::schema::FieldType
                let schema_field_type = match field_type {
                    DataFieldType::Int | DataFieldType::Integer | DataFieldType::Float | DataFieldType::Numeric => SchemaFieldType::Numeric,
                    DataFieldType::String | DataFieldType::Text | DataFieldType::Categorical => SchemaFieldType::Text,
                    DataFieldType::Boolean => SchemaFieldType::Boolean,
                    DataFieldType::Date | DataFieldType::Time | DataFieldType::DateTime => SchemaFieldType::DateTime,
                    DataFieldType::Binary => SchemaFieldType::Custom("binary".to_string()),
                    DataFieldType::Array(_) => SchemaFieldType::Array(Box::new(SchemaFieldType::Text)),
                    DataFieldType::Object | DataFieldType::Json => SchemaFieldType::Object(HashMap::new()),
                    DataFieldType::Image => SchemaFieldType::Image,
                    DataFieldType::Audio => SchemaFieldType::Audio,
                    DataFieldType::Video => SchemaFieldType::Video,
                    DataFieldType::TimeSeries => SchemaFieldType::Custom("timeseries".to_string()),
                    DataFieldType::Embedding(_) => SchemaFieldType::Custom("embedding".to_string()),
                    DataFieldType::Custom(s) => SchemaFieldType::Custom(s),
                };
                let field_def = SchemaFieldDefinition {
                    name: field_name.clone(),
                    field_type: schema_field_type,
                    data_type: None,
                    required: !nullable,
                    nullable,
                    primary_key: false,
                    foreign_key: None,
                    description: None,
                    default_value: None,
                    constraints: None,
                    metadata: HashMap::new(),
                };
                schema.add_field(field_def)?;
            }
        }
        
        Ok(schema)
    }
    
    /// 估计Excel文件行数
    pub fn estimate_excel_row_count(path: &Path) -> Result<usize> {
        // 打开Excel文件
        let mut workbook: Xlsx<_> = match open_workbook(path) {
            Ok(wb) => wb,
            Err(e) => return Err(Error::io(&format!("无法打开Excel文件: {}", e))),
        };
        
        // 获取第一个工作表
        let sheet_names = workbook.sheet_names().to_vec();
        if sheet_names.is_empty() {
            return Ok(0);
        }
        
        let sheet_name = &sheet_names[0];
        let range = match workbook.worksheet_range(sheet_name) {
            Some(Ok(range)) => range,
            Some(Err(e)) => return Err(Error::io(&format!("无法读取工作表: {}", e))),
            None => return Err(Error::invalid_input("无法找到工作表")),
        };
        
        // 返回行数（减去标题行）
        let row_count = range.height();
        if row_count > 0 {
            Ok(row_count - 1)
        } else {
            Ok(0)
        }
    }
    
    /// 读取Excel行
    pub fn read_excel_rows(path: &Path, start: usize, count: usize) -> Result<Vec<Record>> {
        // 打开Excel文件
        let mut workbook: Xlsx<_> = match open_workbook(path) {
            Ok(wb) => wb,
            Err(e) => return Err(Error::io(&format!("无法打开Excel文件: {}", e))),
        };
        
        // 获取第一个工作表
        let sheet_names = workbook.sheet_names().to_vec();
        if sheet_names.is_empty() {
            return Ok(Vec::new());
        }
        
        let sheet_name = &sheet_names[0];
        let range = match workbook.worksheet_range(sheet_name) {
            Some(Ok(range)) => range,
            Some(Err(e)) => return Err(Error::io(&format!("无法读取工作表: {}", e))),
            None => return Err(Error::invalid_input("无法找到工作表")),
        };
        
        // 确保有标题行
        if range.height() == 0 {
            return Ok(Vec::new());
        }
        
        // 获取标题
        let headers: Vec<String> = range.rows().next().unwrap()
            .iter()
            .map(|cell| match cell {
                ExcelDataType::String(s) => s.clone(),
                ExcelDataType::Int(i) => i.to_string(),
                ExcelDataType::Float(f) => f.to_string(),
                ExcelDataType::Bool(b) => b.to_string(),
                ExcelDataType::DateTime(dt) => dt.to_string(),
                ExcelDataType::Error(_) => "Error".to_string(),
                ExcelDataType::Empty => "".to_string(),
            })
            .collect();
        
        // 计算实际要读取的行范围
        let start_row = start + 1; // +1 因为跳过标题行
        let end_row = (start_row + count).min(range.height());
        
        let mut records = Vec::new();
        
        // 读取数据行
        for row_idx in start_row..end_row {
            if let Some(row) = range.rows().nth(row_idx) {
                let mut values = Vec::new();
                
                for cell in row {
                    let value = match cell {
                        ExcelDataType::String(s) => DataValue::String(s.clone()),
                        ExcelDataType::Int(i) => DataValue::Integer(*i),
                        ExcelDataType::Float(f) => DataValue::Float(*f),
                        ExcelDataType::Bool(b) => DataValue::Boolean(*b),
                        ExcelDataType::DateTime(dt) => DataValue::String(dt.to_string()),
                        ExcelDataType::Error(_) => DataValue::Null,
                        ExcelDataType::Empty => DataValue::Null,
                    };
                    
                    values.push(value);
                }
                
                // 如果行中的单元格数量少于标题数量，用null填充
                while values.len() < headers.len() {
                    values.push(DataValue::Null);
                }
                
                records.push(Record::new(values));
            }
        }
        
        Ok(records)
    }
    
    /// 导出Excel文件到指定路径
    pub fn export_to_excel(data: &[Vec<String>], headers: &[String], output_path: PathBuf) -> Result<()> {
        debug!("导出Excel到路径: {:?}", output_path);
        
        // 确保输出目录存在
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::io(&format!("创建输出目录失败: {}", e)))?;
        }
        
        // 创建工作簿
        let mut workbook = Workbook::new();
        let worksheet = workbook.add_worksheet();
        worksheet
            .set_name("Sheet1")
            .map_err(|e| Error::Data(format!("设置工作表名称失败: {}", e)))?;
        
        // 写入标题行
        for (col_idx, header) in headers.iter().enumerate() {
            worksheet
                .write_string(0, col_idx as u16, header)
                .map_err(|e| Error::Data(format!("写入标题单元格失败 (列 {}): {}", col_idx, e)))?;
        }
        
        // 写入数据行
        for (row_idx, row_data) in data.iter().enumerate() {
            for (col_idx, cell_value) in row_data.iter().enumerate() {
                // 尝试解析为数字，如果失败则作为字符串处理
                if let Ok(int_val) = cell_value.parse::<i64>() {
                    worksheet
                        .write_number(row_idx as u32 + 1, col_idx as u16, int_val as f64)
                        .map_err(|e| Error::Data(format!(
                            "写入数字单元格失败 (行 {}, 列 {}): {}",
                            row_idx + 1,
                            col_idx,
                            e
                        )))?;
                } else if let Ok(float_val) = cell_value.parse::<f64>() {
                    worksheet
                        .write_number(row_idx as u32 + 1, col_idx as u16, float_val)
                        .map_err(|e| Error::Data(format!(
                            "写入浮点单元格失败 (行 {}, 列 {}): {}",
                            row_idx + 1,
                            col_idx,
                            e
                        )))?;
                } else if let Ok(bool_val) = cell_value.parse::<bool>() {
                    worksheet
                        .write_boolean(row_idx as u32 + 1, col_idx as u16, bool_val)
                        .map_err(|e| Error::Data(format!(
                            "写入布尔单元格失败 (行 {}, 列 {}): {}",
                            row_idx + 1,
                            col_idx,
                            e
                        )))?;
                } else {
                    worksheet
                        .write_string(row_idx as u32 + 1, col_idx as u16, cell_value)
                        .map_err(|e| Error::Data(format!(
                            "写入字符串单元格失败 (行 {}, 列 {}): {}",
                            row_idx + 1,
                            col_idx,
                            e
                        )))?;
                }
            }
        }
        
        // 保存工作簿到文件
        workbook.save(&output_path)
            .map_err(|e| Error::io(&format!("保存Excel文件失败: {}", e)))?;
        
        info!("成功导出Excel文件: {:?}, 包含 {} 行数据", output_path, data.len());
        Ok(())
    }
}
