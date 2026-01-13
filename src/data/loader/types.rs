// 数据加载器类型定义

use std::collections::HashMap;
use std::fmt;
use serde::{Serialize, Deserialize};
use std::path::Path;
use log::debug;

/// 数据格式枚举
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataFormat {
    /// CSV 格式
    Csv {
        /// 分隔符
        delimiter: char,
        /// 是否有标题行
        has_header: bool,
        /// 引号字符
        quote: char,
        /// 转义字符
        escape: char,
    },
    /// JSON 格式
    Json {
        /// 是否是行分隔的JSON
        is_lines: bool,
        /// 是否是数组
        is_array: bool,
        /// 额外选项
        options: Vec<(String, String)>,
    },
    /// Parquet 格式
    Parquet {
        /// 压缩类型
        compression: String,
        /// 额外选项
        options: Vec<(String, String)>,
    },
    /// Avro 格式
    Avro {
        /// 模式
        schema: String,
        /// 额外选项
        options: Vec<(String, String)>,
    },
    /// Excel 格式
    Excel {
        /// 工作表名称
        sheet_name: Option<String>,
        /// 工作表索引
        sheet_index: Option<usize>,
        /// 是否有标题行
        has_header: bool,
    },
    /// 文本格式
    Text {
        /// 换行符
        new_line: String,
        /// 编码
        encoding: String,
    },
    /// 张量格式
    Tensor {
        /// 数据类型
        dtype: String,
        /// 形状
        shape: Vec<usize>,
        /// 压缩格式
        compression: Option<String>,
        /// 字节序
        endian: String,
    },
    /// 自定义文本格式
    CustomText(String),
    /// 自定义二进制格式
    CustomBinary(String),
}

impl DataFormat {
    /// 创建新的CSV格式
    pub fn csv() -> Self {
        Self::Csv {
            delimiter: ',',
            has_header: true,
            quote: '"',
            escape: '\\',
        }
    }
    
    /// 创建新的JSON格式
    pub fn json() -> Self {
        Self::Json {
            is_lines: false,
            is_array: true,
            options: Vec::new(),
        }
    }
    
    /// 创建新的行分隔JSON格式
    pub fn jsonl() -> Self {
        Self::Json {
            is_lines: true,
            is_array: false,
            options: Vec::new(),
        }
    }
    
    /// 创建新的Parquet格式
    pub fn parquet() -> Self {
        Self::Parquet {
            compression: "snappy".to_string(),
            options: Vec::new(),
        }
    }
    
    /// 创建新的Avro格式
    pub fn avro() -> Self {
        Self::Avro {
            schema: String::new(),
            options: Vec::new(),
        }
    }
    
    /// 创建新的Excel格式
    pub fn excel() -> Self {
        Self::Excel {
            sheet_name: None,
            sheet_index: Some(0),
            has_header: true,
        }
    }
    
    /// 创建新的文本格式
    pub fn text() -> Self {
        Self::Text {
            new_line: "\n".to_string(),
            encoding: "utf-8".to_string(),
        }
    }
    
    /// 从文件类型创建数据格式
    pub fn from_file_type(file_type: FileType) -> Self {
        match file_type {
            FileType::Csv => Self::csv(),
            FileType::Json => Self::json(),
            FileType::Parquet => Self::parquet(),
            FileType::Avro => Self::avro(),
            FileType::Excel => Self::excel(),
            FileType::Text => Self::text(),
            _ => Self::json(), // 默认为JSON
        }
    }
    
    /// 获取文件扩展名
    pub fn file_extension(&self) -> &'static str {
        match self {
            Self::Csv { .. } => "csv",
            Self::Json { is_lines, .. } => if *is_lines { "jsonl" } else { "json" },
            Self::Parquet { .. } => "parquet",
            Self::Avro { .. } => "avro",
            Self::Excel { .. } => "xlsx",
            Self::Text { .. } => "txt",
            Self::CustomText(_) => "txt",
            Self::CustomBinary(_) => "bin",
            Self::Tensor { .. } => "tensor",
        }
    }
    
    /// 获取MIME类型
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Csv { .. } => "text/csv",
            Self::Json { .. } => "application/json",
            Self::Parquet { .. } => "application/vnd.apache.parquet",
            Self::Avro { .. } => "application/avro",
            Self::Excel { .. } => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            Self::Text { .. } => "text/plain",
            Self::CustomText(_) => "text/plain",
            Self::CustomBinary(_) => "application/octet-stream",
            Self::Tensor { .. } => "application/octet-stream",
        }
    }
}

impl fmt::Display for DataFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Csv { .. } => write!(f, "CSV"),
            Self::Json { is_lines, .. } => if *is_lines {
                write!(f, "JSONL")
            } else {
                write!(f, "JSON")
            },
            Self::Parquet { .. } => write!(f, "Parquet"),
            Self::Avro { .. } => write!(f, "Avro"),
            Self::Excel { .. } => write!(f, "Excel"),
            Self::Text { .. } => write!(f, "Text"),
            Self::CustomText(text) => write!(f, "CustomText({})", text),
            Self::CustomBinary(format) => write!(f, "CustomBinary({})", format),
            Self::Tensor { dtype, shape, compression, endian } => {
                write!(f, "Tensor(dtype={}, shape={:?}, endian={}", dtype, shape, endian)?;
                if let Some(compression) = compression {
                    write!(f, ", compression={}", compression)?;
                }
                write!(f, ")")
            }
        }
    }
}

/// 数据源枚举
#[derive(Debug, Clone)]
pub enum DataSource {
    /// 文件数据源，包含文件路径
    File(String),
    /// 数据库数据源，包含数据库配置
    Database(crate::data::ImportedDatabaseConfig),
    /// 流数据源，包含流标识符
    Stream(String),
    /// 自定义数据源，包含类型和参数
    Custom(String, HashMap<String, String>),
    /// 内存数据源，包含二进制数据
    Memory(Vec<u8>),
}

impl DataSource {
    /// 获取数据源类型字符串
    pub fn source_type(&self) -> &'static str {
        match self {
            DataSource::File(_) => "file",
            DataSource::Database(_) => "database",
            DataSource::Stream(_) => "stream",
            DataSource::Custom(_, _) => "custom",
            DataSource::Memory(_) => "memory",
        }
    }
    
    /// 获取数据源路径或标识符
    pub fn identifier(&self) -> String {
        match self {
            DataSource::File(path) => path.clone(),
            DataSource::Database(config) => format!("db:{}", config.connection_string),
            DataSource::Stream(id) => format!("stream:{}", id),
            DataSource::Custom(type_id, _) => format!("custom:{}", type_id),
            DataSource::Memory(_) => format!("memory:{}", uuid::Uuid::new_v4()),
        }
    }
    
    /// 检查数据源是否有效
    pub fn is_valid(&self) -> bool {
        match self {
            DataSource::File(path) => !path.is_empty() && std::path::Path::new(path).exists(),
            DataSource::Database(config) => !config.connection_string.is_empty(),
            DataSource::Stream(id) => !id.is_empty(),
            DataSource::Custom(_, params) => !params.is_empty(),
            DataSource::Memory(data) => !data.is_empty(),
        }
    }
}

/// 文件类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    /// CSV文件
    Csv,
    /// JSON文件
    Json,
    /// Parquet文件
    Parquet,
    /// Avro文件
    Avro,
    /// Excel文件
    Excel,
    /// SQLite数据库
    Sqlite,
    /// XML文件
    Xml,
    /// 文本文件
    Text,
    /// 二进制文件
    Binary,
    /// 其他类型
    Other,
    /// 未知类型
    Unknown
}

impl FileType {
    /// 从文件路径推断文件类型
    pub fn from_path<P: AsRef<Path>>(path: P) -> Self {
        let path_ref = path.as_ref();
        
        if let Some(extension) = path_ref.extension() {
            let ext_str = extension.to_string_lossy().to_lowercase();
            
            match ext_str.as_str() {
                "csv" => FileType::Csv,
                "json" | "jsonl" => FileType::Json,
                "parquet" | "pq" => FileType::Parquet,
                "avro" => FileType::Avro,
                "xlsx" | "xls" => FileType::Excel,
                "sqlite" | "db" | "sqlite3" => FileType::Sqlite,
                "xml" => FileType::Xml,
                "txt" => FileType::Text,
                "bin" => FileType::Binary,
                _ => {
                    debug!("未知文件扩展名: {}", ext_str);
                    FileType::Other
                }
            }
        } else {
            debug!("无法从路径确定文件类型: {:?}", path_ref);
            FileType::Unknown
        }
    }
    
    /// 从字符串解析文件类型
    pub fn from_string(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "csv" => Some(FileType::Csv),
            "json" | "jsonl" => Some(FileType::Json),
            "parquet" | "pq" => Some(FileType::Parquet),
            "avro" => Some(FileType::Avro),
            "excel" | "xlsx" | "xls" => Some(FileType::Excel),
            "sqlite" | "db" | "sqlite3" => Some(FileType::Sqlite),
            "xml" => Some(FileType::Xml),
            "text" | "txt" => Some(FileType::Text),
            "binary" | "bin" => Some(FileType::Binary),
            "other" => Some(FileType::Other),
            _ => None
        }
    }
    
    /// 获取文件扩展名
    pub fn extension(&self) -> &'static str {
        match self {
            FileType::Csv => "csv",
            FileType::Json => "json",
            FileType::Parquet => "parquet",
            FileType::Avro => "avro",
            FileType::Excel => "xlsx",
            FileType::Sqlite => "sqlite",
            FileType::Xml => "xml",
            FileType::Text => "txt",
            FileType::Unknown => "",
        }
    }
    
    /// 检查文件类型是否支持
    pub fn is_supported(&self) -> bool {
        match self {
            FileType::Unknown => false,
            _ => true
        }
    }
}

impl fmt::Display for FileType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FileType::Csv => write!(f, "csv"),
            FileType::Json => write!(f, "json"),
            FileType::Parquet => write!(f, "parquet"),
            FileType::Avro => write!(f, "avro"),
            FileType::Excel => write!(f, "excel"),
            FileType::Sqlite => write!(f, "sqlite"),
            FileType::Xml => write!(f, "xml"),
            FileType::Text => write!(f, "text"),
            FileType::Unknown => write!(f, "unknown"),
        }
    }
}

/// 格式转换器类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormatConverterType {
    /// CSV 到 JSON 转换器
    CsvToJson,
    /// JSON 到 CSV 转换器
    JsonToCsv,
    /// Parquet 到 CSV 转换器
    ParquetToCsv,
    /// CSV 到 Parquet 转换器
    CsvToParquet,
    /// 自定义转换器
    Custom,
}

/// 加载器分类
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoaderCategory {
    /// 文件加载器
    File,
    /// 数据库加载器
    Database,
    /// 流加载器
    Stream,
    /// 远程加载器
    Remote,
    /// 内存加载器
    Memory,
    /// 自定义加载器
    Custom,
}

impl fmt::Display for LoaderCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoaderCategory::File => write!(f, "file"),
            LoaderCategory::Database => write!(f, "database"),
            LoaderCategory::Stream => write!(f, "stream"),
            LoaderCategory::Remote => write!(f, "remote"),
            LoaderCategory::Memory => write!(f, "memory"),
            LoaderCategory::Custom => write!(f, "custom"),
        }
    }
}

/// 加载器状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoaderState {
    /// 初始化状态
    Initializing,
    /// 就绪状态
    Ready,
    /// 正在加载状态
    Loading,
    /// 已关闭状态
    Closed,
    /// 错误状态
    Error,
}

/// 加载模式枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadMode {
    /// 同步加载
    Sync,
    /// 异步加载
    Async,
    /// 流式加载
    Streaming,
    /// 批量加载
    Batch,
}

/// 加载优先级
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LoadPriority {
    /// 低优先级
    Low = 0,
    /// 正常优先级
    Normal = 1,
    /// 高优先级
    High = 2,
    /// 紧急优先级
    Critical = 3,
}

impl Default for LoadPriority {
    fn default() -> Self {
        LoadPriority::Normal
    }
}

/// 二进制加载选项
#[derive(Debug, Clone)]
pub struct BinaryLoadOptions {
    /// 二进制格式
    pub format: String,
    /// 字节序
    pub endianness: Endianness,
    /// 对齐字节数
    pub alignment: usize,
    /// 跳过字节数
    pub skip_bytes: usize,
    /// 最大加载字节数
    pub max_bytes: Option<usize>,
    /// 自定义选项
    pub custom_options: HashMap<String, String>,
}

impl Default for BinaryLoadOptions {
    fn default() -> Self {
        Self {
            format: "raw".to_string(),
            endianness: Endianness::Little,
            alignment: 1,
            skip_bytes: 0,
            max_bytes: None,
            custom_options: HashMap::new(),
        }
    }
}

/// 字节序
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    /// 小端字节序
    Little,
    /// 大端字节序
    Big,
}

/// 导入源类型
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportSourceType {
    /// 本地文件
    LocalFile,
    /// 远程HTTP/HTTPS
    Http,
    /// 数据库
    Database,
    /// 流数据
    Stream,
    /// 内存数据
    Memory,
    /// S3
    S3,
    /// Azure Blob
    AzureBlob,
}

impl ImportSourceType {
    /// 从URL或路径判断源类型
    pub fn from_path(path: &str) -> Self {
        if path.starts_with("http://") || path.starts_with("https://") {
            Self::Http
        } else if path.starts_with("s3://") {
            Self::S3
        } else if path.starts_with("azure://") || path.starts_with("azblob://") {
            Self::AzureBlob
        } else if path.starts_with("jdbc:") || path.starts_with("db:") {
            Self::Database
        } else if path.starts_with("stream:") {
            Self::Stream
        } else if path.starts_with("memory:") {
            Self::Memory
        } else {
            Self::LocalFile
        }
    }
    
    /// 是否需要网络访问
    pub fn requires_network(&self) -> bool {
        match self {
            Self::LocalFile | Self::Memory => false,
            _ => true,
        }
    }
}

impl fmt::Display for ImportSourceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LocalFile => write!(f, "local_file"),
            Self::Http => write!(f, "http"),
            Self::Database => write!(f, "database"),
            Self::Stream => write!(f, "stream"),
            Self::Memory => write!(f, "memory"),
            Self::S3 => write!(f, "s3"),
            Self::AzureBlob => write!(f, "azure_blob"),
        }
    }
}

/// 数据模式定义 - 与iterator.rs兼容
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DataSchema {
    /// 字段列表
    pub fields: Vec<SchemaField>,
    /// 主键字段
    pub primary_key: Option<String>,
    /// 外键约束
    pub foreign_keys: Vec<ForeignKeyConstraint>,
    /// 索引定义
    pub indexes: Vec<IndexDefinition>,
    /// 模式版本
    pub version: String,
    /// 元数据
    pub metadata: HashMap<String, String>,
}

/// 模式字段定义
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SchemaField {
    /// 字段名称
    pub name: String,
    /// 字段类型
    pub field_type: String,
    /// 是否可空
    pub nullable: bool,
    /// 默认值
    pub default_value: Option<String>,
    /// 字段约束
    pub constraints: Vec<FieldConstraint>,
    /// 字段描述
    pub description: Option<String>,
}

/// 字段约束
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum FieldConstraint {
    /// 非空约束
    NotNull,
    /// 唯一约束
    Unique,
    /// 检查约束
    Check(String),
    /// 长度约束
    Length { min: Option<usize>, max: Option<usize> },
    /// 范围约束
    Range { min: Option<f64>, max: Option<f64> },
    /// 格式约束（正则表达式）
    Format(String),
}

/// 外键约束
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ForeignKeyConstraint {
    /// 本地字段
    pub local_field: String,
    /// 引用表
    pub referenced_table: String,
    /// 引用字段
    pub referenced_field: String,
    /// 删除操作
    pub on_delete: ReferentialAction,
    /// 更新操作
    pub on_update: ReferentialAction,
}

/// 引用操作
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ReferentialAction {
    /// 级联操作
    Cascade,
    /// 设为NULL
    SetNull,
    /// 设为默认值
    SetDefault,
    /// 限制操作
    Restrict,
    /// 无操作
    NoAction,
}

/// 索引定义
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexDefinition {
    /// 索引名称
    pub name: String,
    /// 索引字段
    pub fields: Vec<String>,
    /// 是否唯一索引
    pub unique: bool,
    /// 索引类型
    pub index_type: IndexType,
}

/// 索引类型
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum IndexType {
    /// B-Tree索引
    BTree,
    /// 哈希索引
    Hash,
    /// 全文索引
    FullText,
    /// 空间索引
    Spatial,
}

impl DataSchema {
    /// 创建新的数据模式
    pub fn new() -> Self {
        Self {
            fields: Vec::new(),
            primary_key: None,
            foreign_keys: Vec::new(),
            indexes: Vec::new(),
            version: "1.0".to_string(),
            metadata: HashMap::new(),
        }
    }

    /// 添加字段
    pub fn add_field(&mut self, field: SchemaField) {
        self.fields.push(field);
    }
    
    /// 获取字段数量
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// 设置主键
    pub fn with_primary_key(mut self, key: String) -> Self {
        self.primary_key = Some(key);
        self
    }

    /// 添加外键约束
    pub fn add_foreign_key(&mut self, constraint: ForeignKeyConstraint) {
        self.foreign_keys.push(constraint);
    }

    /// 添加索引
    pub fn add_index(&mut self, index: IndexDefinition) {
        self.indexes.push(index);
    }

    /// 验证模式
    pub fn validate(&self) -> Result<(), String> {
        if self.fields.is_empty() {
            return Err("模式必须包含至少一个字段".to_string());
        }

        // 验证主键字段存在
        if let Some(pk) = &self.primary_key {
            if !self.fields.iter().any(|f| &f.name == pk) {
                return Err(format!("主键字段 '{}' 不存在", pk));
            }
        }

        // 验证外键字段存在
        for fk in &self.foreign_keys {
            if !self.fields.iter().any(|f| f.name == fk.local_field) {
                return Err(format!("外键字段 '{}' 不存在", fk.local_field));
            }
        }

        // 验证索引字段存在
        for idx in &self.indexes {
            for field in &idx.fields {
                if !self.fields.iter().any(|f| &f.name == field) {
                    return Err(format!("索引字段 '{}' 不存在", field));
                }
            }
        }

        Ok(())
    }

    /// 获取字段
    pub fn get_field(&self, name: &str) -> Option<&SchemaField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// 获取字段类型
    pub fn get_field_type(&self, name: &str) -> Option<&str> {
        self.get_field(name).map(|f| f.field_type.as_str())
    }
}

impl Default for DataSchema {
    fn default() -> Self {
        Self::new()
    }
}

impl SchemaField {
    /// 创建新字段
    pub fn new(name: String, field_type: String) -> Self {
        Self {
            name,
            field_type,
            nullable: true,
            default_value: None,
            constraints: Vec::new(),
            description: None,
        }
    }

    /// 设置为非空
    pub fn not_null(mut self) -> Self {
        self.nullable = false;
        self.constraints.push(FieldConstraint::NotNull);
        self
    }

    /// 设置为唯一
    pub fn unique(mut self) -> Self {
        self.constraints.push(FieldConstraint::Unique);
        self
    }

    /// 设置默认值
    pub fn with_default(mut self, value: String) -> Self {
        self.default_value = Some(value);
        self
    }

    /// 添加约束
    pub fn with_constraint(mut self, constraint: FieldConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// 设置描述
    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }
} 