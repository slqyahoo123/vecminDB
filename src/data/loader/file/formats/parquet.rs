// Parquet格式选项
// Parquet format options

use std::collections::HashMap;
use crate::data::loader::file::formats::{FormatType, FormatOptions};

/// Parquet格式选项
/// Parquet format options
#[derive(Debug, Clone, Default)]
pub struct ParquetOptions {
    /// 是否使用模式
    /// Whether to use schema
    pub use_schema: bool,
    
    /// 自定义模式路径
    /// Custom schema path
    pub schema_path: Option<String>,
    
    /// 是否压缩
    /// Whether to compress
    pub compress: bool,
    
    /// 压缩编解码器
    /// Compression codec
    pub codec: String,
    
    /// 行组大小
    /// Row group size
    pub row_group_size: usize,
    
    /// 页大小
    /// Page size
    pub page_size: usize,
    
    /// 字典页大小
    /// Dictionary page size
    pub dictionary_page_size: usize,
    
    /// 是否开启字典编码
    /// Whether to enable dictionary encoding
    pub enable_dictionary: bool,
    
    /// 是否开启统计信息
    /// Whether to enable statistics
    pub enable_statistics: bool,
}

impl ParquetOptions {
    /// 创建新的Parquet选项
    /// Create new Parquet options
    pub fn new() -> Self {
        Self {
            use_schema: true,
            schema_path: None,
            compress: true,
            codec: "snappy".to_string(),
            row_group_size: 1024 * 1024,
            page_size: 8 * 1024,
            dictionary_page_size: 1024 * 1024,
            enable_dictionary: true,
            enable_statistics: true,
        }
    }
    
    /// 设置是否使用模式
    /// Set whether to use schema
    pub fn with_use_schema(mut self, use_schema: bool) -> Self {
        self.use_schema = use_schema;
        self
    }
    
    /// 设置自定义模式路径
    /// Set custom schema path
    pub fn with_schema_path(mut self, schema_path: impl Into<String>) -> Self {
        self.schema_path = Some(schema_path.into());
        self
    }
    
    /// 设置是否压缩
    /// Set whether to compress
    pub fn with_compress(mut self, compress: bool) -> Self {
        self.compress = compress;
        self
    }
    
    /// 设置压缩编解码器
    /// Set compression codec
    pub fn with_codec(mut self, codec: impl Into<String>) -> Self {
        self.codec = codec.into();
        self
    }
    
    /// 设置行组大小
    /// Set row group size
    pub fn with_row_group_size(mut self, row_group_size: usize) -> Self {
        self.row_group_size = row_group_size;
        self
    }
    
    /// 设置页大小
    /// Set page size
    pub fn with_page_size(mut self, page_size: usize) -> Self {
        self.page_size = page_size;
        self
    }
    
    /// 设置字典页大小
    /// Set dictionary page size
    pub fn with_dictionary_page_size(mut self, dictionary_page_size: usize) -> Self {
        self.dictionary_page_size = dictionary_page_size;
        self
    }
    
    /// 设置是否开启字典编码
    /// Set whether to enable dictionary encoding
    pub fn with_enable_dictionary(mut self, enable_dictionary: bool) -> Self {
        self.enable_dictionary = enable_dictionary;
        self
    }
    
    /// 设置是否开启统计信息
    /// Set whether to enable statistics
    pub fn with_enable_statistics(mut self, enable_statistics: bool) -> Self {
        self.enable_statistics = enable_statistics;
        self
    }
}

impl FormatOptions for ParquetOptions {
    fn format_type(&self) -> FormatType {
        FormatType::Parquet
    }
    
    fn to_options_map(&self) -> HashMap<String, String> {
        let mut options = HashMap::new();
        
        options.insert("format".to_string(), "parquet".to_string());
        options.insert("use_schema".to_string(), self.use_schema.to_string());
        
        if let Some(schema_path) = &self.schema_path {
            options.insert("schema_path".to_string(), schema_path.clone());
        }
        
        options.insert("compress".to_string(), self.compress.to_string());
        options.insert("codec".to_string(), self.codec.clone());
        options.insert("row_group_size".to_string(), self.row_group_size.to_string());
        options.insert("page_size".to_string(), self.page_size.to_string());
        options.insert("dictionary_page_size".to_string(), self.dictionary_page_size.to_string());
        options.insert("enable_dictionary".to_string(), self.enable_dictionary.to_string());
        options.insert("enable_statistics".to_string(), self.enable_statistics.to_string());
        
        options
    }
    
    fn from_options_map(options: &HashMap<String, String>) -> Self {
        let mut parquet_options = Self::new();
        
        if let Some(use_schema) = options.get("use_schema") {
            if let Ok(value) = use_schema.parse::<bool>() {
                parquet_options.use_schema = value;
            }
        }
        
        if let Some(schema_path) = options.get("schema_path") {
            parquet_options.schema_path = Some(schema_path.clone());
        }
        
        if let Some(compress) = options.get("compress") {
            if let Ok(value) = compress.parse::<bool>() {
                parquet_options.compress = value;
            }
        }
        
        if let Some(codec) = options.get("codec") {
            parquet_options.codec = codec.clone();
        }
        
        if let Some(row_group_size) = options.get("row_group_size") {
            if let Ok(value) = row_group_size.parse::<usize>() {
                parquet_options.row_group_size = value;
            }
        }
        
        if let Some(page_size) = options.get("page_size") {
            if let Ok(value) = page_size.parse::<usize>() {
                parquet_options.page_size = value;
            }
        }
        
        if let Some(dictionary_page_size) = options.get("dictionary_page_size") {
            if let Ok(value) = dictionary_page_size.parse::<usize>() {
                parquet_options.dictionary_page_size = value;
            }
        }
        
        if let Some(enable_dictionary) = options.get("enable_dictionary") {
            if let Ok(value) = enable_dictionary.parse::<bool>() {
                parquet_options.enable_dictionary = value;
            }
        }
        
        if let Some(enable_statistics) = options.get("enable_statistics") {
            if let Ok(value) = enable_statistics.parse::<bool>() {
                parquet_options.enable_statistics = value;
            }
        }
        
        parquet_options
    }
} 