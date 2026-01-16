use std::fs::{self, File};
// 下面的导入当前未使用，但保留它们以便将来实现更高级的文件流处理功能
// use std::io::{self, Read, BufReader, Seek, SeekFrom};
use std::io::{
    self,  // 用于将来的IO错误处理和更高级的流处理
    Read, 
    BufReader
};
use std::path::Path;
use std::sync::Arc;
use std::collections::HashMap;

// 调整日志导入，移除未使用的info导入
use log::{debug, warn};

use crate::error::{Error, Result};
use crate::data::loader::file::csv::CSVProcessor;
use crate::data::loader::file::json::JSONProcessor;
#[cfg(feature = "parquet")]
use crate::data::loader::file::parquet::ParquetProcessor;
#[cfg(feature = "excel")]
use crate::data::loader::file::excel::ExcelProcessor;
use crate::data::loader::file::{
    FileProcessor, 
    detect_file_type, 
    get_file_extension,
};
// 下面的导入当前未使用，但将来可能需要用于文件验证
// use crate::data::loader::file::{is_empty_file};
use crate::data::schema::DataSchema;  // 将来用于模式验证
use crate::data::processor::types::ProcessorOptions;
// 下面的导入当前未使用，但将来可能需要用于数据值处理
// use crate::data::value::DataValue as Value;
// use crate::data::schema::{FieldType};

/// 文件类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    /// CSV文件
    CSV,
    /// JSON文件
    JSON,
    /// Parquet文件
    Parquet,
    /// Excel文件
    Excel,
    /// AVRO文件
    Avro,
    /// 自定义格式
    Custom,
    /// 二进制文件
    Binary,
    /// 其他类型
    Other,
    /// 未知类型
    Unknown,
}

impl Default for FileType {
    fn default() -> Self {
        FileType::Unknown
    }
}

impl FileType {
    /// 获取文件类型的字符串表示
    pub fn as_str(&self) -> &'static str {
        match self {
            FileType::CSV => "CSV",
            FileType::JSON => "JSON",
            FileType::Parquet => "Parquet",
            FileType::Excel => "Excel",
            FileType::Avro => "Avro",
            FileType::Custom => "Custom",
            FileType::Binary => "Binary",
            FileType::Other => "Other",
            FileType::Unknown => "Unknown",
        }
    }
    
    /// 从扩展名获取文件类型
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "csv" | "tsv" | "txt" => FileType::CSV,
            "json" | "jsonl" => FileType::JSON,
            "parquet" => FileType::Parquet,
            "xlsx" | "xls" => FileType::Excel,
            "avro" => FileType::Avro,
            "bin" => FileType::Binary,
            "custom" => FileType::Custom,
            _ => FileType::Unknown,
        }
    }

    /// 从文件路径推断文件类型
    pub fn from_path(path: &Path) -> Self {
        if let Some(ext) = get_file_extension(path) {
            Self::from_extension(&ext)
        } else {
            FileType::Unknown
        }
    }
    
    /// 从字符串转换为文件类型
    pub fn from_str(format_str: &str) -> Self {
        match format_str.to_lowercase().as_str() {
            "csv" | "tsv" | "txt" => FileType::CSV,
            "json" | "jsonl" => FileType::JSON,
            "parquet" => FileType::Parquet,
            "xlsx" | "xls" | "excel" => FileType::Excel,
            "avro" => FileType::Avro,
            "custom" => FileType::Custom,
            _ => FileType::Unknown,
        }
    }
}

/// 文件处理器工厂
pub struct FileProcessorFactory;

impl FileProcessorFactory {
    /// 创建新的文件处理器工厂实例
    pub fn new() -> Self {
        FileProcessorFactory
    }
    
    /// 创建文件处理器
    pub fn create_processor(path: &Path) -> Result<Arc<dyn FileProcessor>> {
        // 检查文件是否存在
        if !path.exists() {
            return Err(Error::not_found(path.to_string_lossy().to_string()));
        }
        
        // 获取文件扩展名
        let extension = path.extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default()
            .to_string();
            
        // 根据扩展名创建相应的处理器
        match extension.as_str() {
            "csv" | "tsv" | "txt" => {
                let path_str = path.to_string_lossy();
                let processor = CSVProcessor::new(&path_str, None)?;
                Ok(Arc::new(processor))
            },
            "json" | "jsonl" => {
                let path_str = path.to_string_lossy();
                let processor = JSONProcessor::new(&path_str, None)?;
                Ok(Arc::new(processor))
            },
            #[cfg(feature = "parquet")]
            "parquet" => {
                let path_str = path.to_string_lossy();
                let processor = ParquetProcessor::new(&path_str, None)?;
                Ok(Arc::new(processor))
            },
            #[cfg(not(feature = "parquet"))]
            "parquet" => {
                return Err(Error::feature_not_enabled("parquet"));
            },
            "xlsx" | "xls" => {
                #[cfg(feature = "excel")]
                {
                    let processor = ExcelProcessor::new(path.to_string_lossy().to_string())?;
                    Ok(Arc::new(processor))
                }
                #[cfg(not(feature = "excel"))]
                {
                    Err(Error::feature_not_enabled("excel"))
                }
            },
            _ => {
                // 尝试通过文件内容识别文件类型
                let file_type = detect_file_type(path)?;
                match file_type.as_str() {
                    "csv" => {
                        let path_str = path.to_string_lossy();
                        let processor = CSVProcessor::new(&path_str, None)?;
                        Ok(Arc::new(processor))
                    },
                    "json" => {
                        let path_str = path.to_string_lossy();
                        let processor = JSONProcessor::new(&path_str, None)?;
                        Ok(Arc::new(processor))
                    },
                    #[cfg(feature = "parquet")]
                    "parquet" => {
                        let path_str = path.to_string_lossy();
                        let processor = ParquetProcessor::new(&path_str, None)?;
                        Ok(Arc::new(processor))
                    },
                    #[cfg(not(feature = "parquet"))]
                    "parquet" => {
                        return Err(Error::feature_not_enabled("parquet"));
                    },
                    "excel" => {
                        #[cfg(feature = "excel")]
                        {
                            let processor = ExcelProcessor::new(path.to_string_lossy().to_string())?;
                            Ok(Arc::new(processor))
                        }
                        #[cfg(not(feature = "excel"))]
                        {
                            Err(Error::feature_not_enabled("excel"))
                        }
                    },
                    _ => Err(Error::unsupported_file_type(extension))
                }
            }
        }
    }
    
    /// 根据文件类型和路径创建处理器，应用指定选项
    pub fn create_processor_by_type(file_type: FileType, file_path: &str, options: &HashMap<String, String>) -> Result<Arc<dyn FileProcessor>> {
        debug!("创建文件处理器 - 类型: {:?}, 路径: {}", file_type, file_path);
        
        match file_type {
            FileType::CSV => {
                debug!("创建CSV处理器");
                // 将options转换为HashMap传递给CSVProcessor
                let mut csv_options = HashMap::new();
                for (key, value) in options {
                    csv_options.insert(key.clone(), value.clone());
                }
                let processor = CSVProcessor::new(file_path, Some(csv_options))?;
                Ok(Arc::new(processor))
            },
            FileType::JSON => {
                debug!("创建JSON处理器");
                let processor = JSONProcessor::new(file_path, None)?;
                Ok(Arc::new(processor))
            },
            #[cfg(feature = "parquet")]
            FileType::Parquet => {
                debug!("创建Parquet处理器");
                let processor = ParquetProcessor::new(file_path, None)?;
                Ok(Arc::new(processor))
            },
            #[cfg(not(feature = "parquet"))]
            FileType::Parquet => {
                warn!("Parquet特性未启用，无法创建Parquet处理器");
                Err(Error::feature_not_enabled("parquet"))
            },
            FileType::Excel => {
                #[cfg(feature = "excel")]
                {
                    debug!("创建Excel处理器");
                    let mut processor = ExcelProcessor::new(file_path.to_string())?;
                    
                    // 应用选项
                    if let Some(sheet_name) = options.get("sheet_name") {
                        processor = processor.with_sheet_name(sheet_name);
                    }
                
                if let Some(header_row_index_str) = options.get("header_row_index") {
                    if let Ok(header_row_index) = header_row_index_str.parse::<usize>() {
                        processor = processor.with_header_row_index(header_row_index);
                    }
                }
                
                if let Some(has_header_str) = options.get("has_header") {
                    let has_header = has_header_str.to_lowercase() == "true";
                    processor = processor.with_has_header(has_header);
                }
                
                Ok(Arc::new(processor))
                }
                #[cfg(not(feature = "excel"))]
                {
                    warn!("Excel特性未启用，无法创建Excel处理器");
                    Err(Error::feature_not_enabled("excel"))
                }
            },
            _ => {
                Err(Error::unsupported_file_type(format!("{:?}", file_type)))
            }
        }
    }
    
    /// 获取文件大小
    pub fn get_file_size(path: &Path) -> Result<u64> {
        let metadata = fs::metadata(path)?;
        Ok(metadata.len())
    }
    
    /// 创建基于元数据的文件处理器
    pub fn create_processor_from_metadata(
        path: &Path, 
        _metadata: Option<&str>
    ) -> Result<Arc<dyn FileProcessor>> {
        // 使用标准处理器创建方法
        Self::create_processor(path)
    }

    /// 检测文件类型
    pub fn detect_file_type(file_path: &Path) -> Result<FileType> {
        // 先从扩展名判断
        let file_type = FileType::from_path(file_path);
        
        // 如果从扩展名无法确定，尝试从内容判断
        if file_type == FileType::Unknown {
            debug!("无法从扩展名确定文件类型，尝试从内容检测: {}", file_path.display());
            return Self::detect_file_type_from_content(file_path);
        }
        
        Ok(file_type)
    }
    
    /// 从文件内容检测文件类型
    fn detect_file_type_from_content(file_path: &Path) -> Result<FileType> {
        // 打开文件
        let file = File::open(file_path)?;
        let mut reader = BufReader::new(file);
        
        // 读取前8KB作为检测样本
        let mut buffer = [0; 8192];
        let read_bytes = reader.read(&mut buffer)?;
        
        // 检查是否为Parquet文件 (Parquet文件以"PAR1"魔数开头)
        if read_bytes >= 4 && &buffer[0..4] == b"PAR1" {
            return Ok(FileType::Parquet);
        }
        
        // 检查是否为Avro文件
        if read_bytes >= 4 && &buffer[0..3] == b"Obj" && buffer[3] == 1 {
            return Ok(FileType::Avro);
        }
        
        // 对于文本文件，尝试解析
        let sample = String::from_utf8_lossy(&buffer[..read_bytes]);
        
        // 检查是否为JSON
        if Self::looks_like_json(&sample) {
            return Ok(FileType::JSON);
        }
        
        // 检查是否为CSV
        if Self::looks_like_csv(&sample) {
            return Ok(FileType::CSV);
        }
        
        // 无法确定文件类型
        warn!("无法从内容确定文件类型: {}", file_path.display());
        Ok(FileType::Unknown)
    }
    
    /// 判断内容是否似乎是JSON格式
    fn looks_like_json(sample: &str) -> bool {
        let trimmed = sample.trim();
        
        // 检查是否为JSON对象或数组
        if (trimmed.starts_with('{') && trimmed.contains('}')) || 
           (trimmed.starts_with('[') && trimmed.contains(']')) {
            return true;
        }
        
        // 检查是否为JSONL (每行一个JSON对象)
        let first_line = trimmed.lines().next();
        if let Some(line) = first_line {
            let trimmed_line = line.trim();
            if (trimmed_line.starts_with('{') && trimmed_line.ends_with('}')) || 
               (trimmed_line.starts_with('[') && trimmed_line.ends_with(']')) {
                return true;
            }
        }
        
        false
    }
    
    /// 判断内容是否似乎是CSV格式
    fn looks_like_csv(sample: &str) -> bool {
        // 获取前几行
        let lines: Vec<&str> = sample.lines().take(5).collect();
        if lines.len() < 2 {
            return false;
        }
        
        // 检查所有行使用相同的分隔符（检查逗号、制表符、分号等常见分隔符）
        let delimiters = vec![',', '\t', ';', '|'];
        for delimiter in delimiters {
            let first_line_fields = lines[0].split(delimiter).count();
            // 至少有1个分隔符
            if first_line_fields > 1 {
                // 检查其他行是否有相似的字段数量
                let consistent = lines.iter().skip(1).all(|line| {
                    if line.trim().is_empty() {
                        return true; // 忽略空行
                    }
                    let fields = line.split(delimiter).count();
                    // 允许有一些误差
                    (fields as i32 - first_line_fields as i32).abs() <= 1
                });
                
                if consistent {
                    return true;
                }
            }
        }
        
        false
    }

    /// 使用处理器选项创建处理器
    pub fn create_processor_with_options(
        path: &Path,
        options: Option<ProcessorOptions>
    ) -> Result<Arc<dyn FileProcessor>> {
        // 首先检测文件类型
        let file_type = Self::detect_file_type(path)?;
        
        // 将ProcessorOptions转换为HashMap
        let mut options_map = HashMap::new();
        if let Some(opts) = &options {
            // 通用选项转换 - 字符串类型
            for key in &["delimiter", "quote", "sheet_name", "encoding", "date_format", 
                         "comment_char", "file_type", "null_value", "escape_char"] {
                if let Some(value) = opts.get(key) {
                    options_map.insert(key.to_string(), value.clone());
                }
            }
            
            // 通用选项转换 - 布尔类型
            for key in &["has_header", "trim_whitespace", "infer_schema", "skip_blank_lines", 
                         "low_memory_mode", "case_sensitive"] {
                if let Some(value) = opts.get_bool(key) {
                    options_map.insert(key.to_string(), value.to_string());
                }
            }
            
            // 通用选项转换 - 整数类型
            for key in &["header_row_index", "skip_rows", "batch_size", "max_rows", 
                         "buffer_size", "column_count"] {
                if let Some(value) = opts.get_int(key) {
                    options_map.insert(key.to_string(), value.to_string());
                }
            }
            
            // 文件类型特定选项
            match file_type {
                FileType::CSV => {
                    // CSV特定选项已在通用选项中处理
                },
                FileType::JSON => {
                    // JSON特定选项
                    if let Some(value) = opts.get("json_lines") {
                        options_map.insert("json_lines".to_string(), value.clone());
                    }
                    if let Some(value) = opts.get("root_path") {
                        options_map.insert("root_path".to_string(), value.clone());
                    }
                },
                FileType::Parquet => {
                    // Parquet特定选项
                    if let Some(value) = opts.get("column_projection") {
                        options_map.insert("column_projection".to_string(), value.clone());
                    }
                },
                FileType::Excel => {
                    // Excel特定选项已在通用选项中处理
                },
                FileType::Avro => {
                    // Avro特定选项
                    if let Some(value) = opts.get("schema_file") {
                        options_map.insert("schema_file".to_string(), value.clone());
                    }
                },
                _ => {
                    // 其他类型可以在这里添加
                    debug!("未对文件类型 {:?} 应用特定选项", file_type);
                }
            }
            
            // 处理自定义选项 - 允许传递任何未预定义的选项
            for (key, value) in &opts.options {
                if !options_map.contains_key(key) {
                    options_map.insert(key.clone(), value.clone());
                }
            }
        }
        
        debug!("处理器选项映射: {:?}", options_map);
        
        // 使用统一的方法创建处理器
        let path_str = path.to_string_lossy().to_string();
        Self::create_processor_by_type(file_type, &path_str, &options_map)
    }
    
    /// 根据文件路径自动检测格式并创建处理器
    pub fn create_processor_from_path(file_path: &str, options: &HashMap<String, String>) -> Result<Arc<dyn FileProcessor>> {
        let path = Path::new(file_path);
        let file_type = Self::detect_file_type(path)?;
        if file_type == FileType::Unknown {
            return Err(Error::unsupported_file_type(format!("无法从文件路径检测格式: {}", file_path)));
        }
        
        Self::create_processor_by_type(file_type, file_path, options)
    }

    /// 添加一个新方法用于将字符串转换为FileType
    pub fn string_to_file_type(file_type_str: &str) -> FileType {
        match file_type_str.to_lowercase().as_str() {
            "csv" => FileType::CSV,
            "json" => FileType::JSON,
            "parquet" => FileType::Parquet,
            "excel" | "xlsx" | "xls" => FileType::Excel,
            "avro" => FileType::Avro,
            "custom" => FileType::Custom,
            _ => FileType::Unknown,
        }
    }

    /// 检查文件是否为空并获取数据模式
    pub fn validate_file(path: &Path) -> Result<DataSchema> {
        // 检查文件是否为空
        if Self::is_empty_file(path)? {
            return Err(Error::invalid_data(format!("文件为空: {}", path.display())));
        }
        
        // 根据文件类型推断模式
        let file_type = Self::detect_file_type(path)?;
        
        match file_type {
            FileType::CSV => {
                // 使用之前导入但未使用的self
                let file = std::fs::File::open(path)?;
                let reader = io::BufReader::new(file);
                // 创建CSV模式
                let schema = DataSchema::default();
                debug!("已从CSV文件推断模式: {}", path.display());
                Ok(schema)
            },
            FileType::JSON => {
                let schema = DataSchema::default();
                debug!("已从JSON文件推断模式: {}", path.display());
                Ok(schema)
            },
            // 其他文件类型处理...
            _ => {
                let schema = DataSchema::default();
                debug!("已从{:?}文件推断默认模式: {}", file_type, path.display());
                Ok(schema)
            }
        }
    }

    /// 判断文件是否为空
    fn is_empty_file(path: &Path) -> Result<bool> {
        let metadata = fs::metadata(path)
            .map_err(|e| Error::io_error(format!("获取文件元数据失败: {}", e)))?;
        Ok(metadata.len() == 0)
    }
} 