// Avro格式选项
// Avro format options

use std::collections::HashMap;
use crate::data::loader::file::formats::{FormatType, FormatOptions};

/// Avro格式选项
/// Avro format options
#[derive(Debug, Clone, Default)]
pub struct AvroOptions {
    /// 是否使用模式
    /// Whether to use schema
    pub use_schema: bool,
    
    /// 是否使用代码生成
    /// Whether to use code generation
    pub use_code_gen: bool,
    
    /// 自定义模式路径
    /// Custom schema path
    pub schema_path: Option<String>,
    
    /// 是否压缩
    /// Whether to compress
    pub compress: bool,
    
    /// 压缩编解码器
    /// Compression codec
    pub codec: String,
    
    /// 块大小
    /// Block size
    pub block_size: usize,
    
    /// 同步标记大小
    /// Sync marker size
    pub sync_interval: usize,
}

impl AvroOptions {
    /// 创建新的Avro选项
    /// Create new Avro options
    pub fn new() -> Self {
        Self {
            use_schema: true,
            use_code_gen: false,
            schema_path: None,
            compress: true,
            codec: "snappy".to_string(),
            block_size: 16384,
            sync_interval: 16000,
        }
    }
    
    /// 设置是否使用模式
    /// Set whether to use schema
    pub fn with_use_schema(mut self, use_schema: bool) -> Self {
        self.use_schema = use_schema;
        self
    }
    
    /// 设置是否使用代码生成
    /// Set whether to use code generation
    pub fn with_use_code_gen(mut self, use_code_gen: bool) -> Self {
        self.use_code_gen = use_code_gen;
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
    
    /// 设置块大小
    /// Set block size
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }
    
    /// 设置同步标记大小
    /// Set sync marker size
    pub fn with_sync_interval(mut self, sync_interval: usize) -> Self {
        self.sync_interval = sync_interval;
        self
    }
}

impl FormatOptions for AvroOptions {
    fn format_type(&self) -> FormatType {
        FormatType::Avro
    }
    
    fn to_options_map(&self) -> HashMap<String, String> {
        let mut options = HashMap::new();
        
        options.insert("format".to_string(), "avro".to_string());
        options.insert("use_schema".to_string(), self.use_schema.to_string());
        options.insert("use_code_gen".to_string(), self.use_code_gen.to_string());
        
        if let Some(schema_path) = &self.schema_path {
            options.insert("schema_path".to_string(), schema_path.clone());
        }
        
        options.insert("compress".to_string(), self.compress.to_string());
        options.insert("codec".to_string(), self.codec.clone());
        options.insert("block_size".to_string(), self.block_size.to_string());
        options.insert("sync_interval".to_string(), self.sync_interval.to_string());
        
        options
    }
    
    fn from_options_map(options: &HashMap<String, String>) -> Self {
        let mut avro_options = Self::new();
        
        if let Some(use_schema) = options.get("use_schema") {
            if let Ok(value) = use_schema.parse::<bool>() {
                avro_options.use_schema = value;
            }
        }
        
        if let Some(use_code_gen) = options.get("use_code_gen") {
            if let Ok(value) = use_code_gen.parse::<bool>() {
                avro_options.use_code_gen = value;
            }
        }
        
        if let Some(schema_path) = options.get("schema_path") {
            avro_options.schema_path = Some(schema_path.clone());
        }
        
        if let Some(compress) = options.get("compress") {
            if let Ok(value) = compress.parse::<bool>() {
                avro_options.compress = value;
            }
        }
        
        if let Some(codec) = options.get("codec") {
            avro_options.codec = codec.clone();
        }
        
        if let Some(block_size) = options.get("block_size") {
            if let Ok(value) = block_size.parse::<usize>() {
                avro_options.block_size = value;
            }
        }
        
        if let Some(sync_interval) = options.get("sync_interval") {
            if let Ok(value) = sync_interval.parse::<usize>() {
                avro_options.sync_interval = value;
            }
        }
        
        avro_options
    }
} 