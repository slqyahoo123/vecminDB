use super::FileProcessor;
// 下面这个导入当前未使用，但保留它是为了支持将来与Parquet格式的集成
// use parquet::record::Field as ParquetField;
use crate::data::loader::file::Record;
use ::csv::{ByteRecord, ReaderBuilder, StringRecord};
// Writer在当前实现中未使用，但将来会用于CSV导出功能
// use csv::Writer;
use log::{debug, error, info, trace, warn};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, BufRead};
use std::path::Path;
use std::sync::Arc;
use crate::data::schema::schema::DataSchema;
use crate::error::{Error, Result};
use ::csv::WriterBuilder;
use lazy_static::lazy_static;
use regex::Regex;
use crate::data::DataValue;

/// 值类型枚举
#[derive(Debug, Clone, PartialEq)]
pub enum ValueType {
    String,
    Integer,
    Float,
    Boolean,
    Date,
    DateTime,
    Time,
    Binary,
    Array,
    Object,
    Null,
}

impl ValueType {
    /// 根据数据模式 FieldType 转换为 ValueType
    pub fn from_schema_field_type(field_type: &crate::data::schema::schema::FieldType) -> Self {
        use crate::data::schema::schema::FieldType as SchemaFieldType;
        match field_type {
            SchemaFieldType::Numeric => ValueType::Float,
            SchemaFieldType::Boolean => ValueType::Boolean,
            SchemaFieldType::DateTime => ValueType::DateTime,
            SchemaFieldType::Array(_) => ValueType::Array,
            SchemaFieldType::Object(_) => ValueType::Object,
            SchemaFieldType::Categorical
            | SchemaFieldType::Text
            | SchemaFieldType::Image
            | SchemaFieldType::Audio
            | SchemaFieldType::Video
            | SchemaFieldType::Custom(_) => ValueType::String,
        }
    }

    /// 根据FieldType转换为ValueType
    pub fn from_field_type(field_type: &FieldType) -> Self {
        match field_type {
            FieldType::String => Self::String,
            FieldType::Number => Self::Float,
            FieldType::Boolean => Self::Boolean,
            FieldType::Date => Self::Date,
            FieldType::Unknown => Self::String,
        }
    }

    /// 转换为数据模式字段类型（schema::FieldType）
    pub fn to_schema_field_type(&self) -> crate::data::schema::schema::FieldType {
        use crate::data::schema::schema::FieldType as SchemaFieldType;
        match self {
            ValueType::String => SchemaFieldType::Text,
            ValueType::Integer => SchemaFieldType::Numeric,
            ValueType::Float => SchemaFieldType::Numeric,
            ValueType::Boolean => SchemaFieldType::Boolean,
            ValueType::Date | ValueType::DateTime | ValueType::Time => SchemaFieldType::DateTime,
            ValueType::Binary => SchemaFieldType::Custom("binary".to_string()),
            ValueType::Array => SchemaFieldType::Array(Box::new(SchemaFieldType::Text)),
            ValueType::Object => SchemaFieldType::Object(HashMap::new()),
            ValueType::Null => SchemaFieldType::Custom("null".to_string()),
        }
    }
}

/// 字段类型
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FieldType {
    String,
    Number,
    Integer,
    Float,
    Boolean,
    Date,
    Timestamp,
    Array,
    Null,
    Unknown,
}

impl FieldType {
    /// 转换为数据模式字段类型
    pub fn to_schema_field_type(&self) -> crate::data::schema::schema::FieldType {
        use crate::data::schema::schema::FieldType as SchemaFieldType;
        match self {
            FieldType::String => SchemaFieldType::Text,
            FieldType::Number | FieldType::Integer | FieldType::Float => SchemaFieldType::Numeric,
            FieldType::Boolean => SchemaFieldType::Boolean,
            FieldType::Date | FieldType::Timestamp => SchemaFieldType::DateTime,
            FieldType::Array => SchemaFieldType::Array(Box::new(SchemaFieldType::Text)),
            FieldType::Null | FieldType::Unknown => SchemaFieldType::Custom("unknown".to_string()),
        }
    }
    
    /// 从数据模式字段类型转换
    pub fn from_schema_field_type(schema_field_type: &crate::data::schema::schema::FieldType) -> Self {
        use crate::data::schema::schema::FieldType as SchemaFieldType;
        match schema_field_type {
            SchemaFieldType::Text | SchemaFieldType::Categorical | SchemaFieldType::Image 
            | SchemaFieldType::Audio | SchemaFieldType::Video | SchemaFieldType::Custom(_) => FieldType::String,
            SchemaFieldType::Numeric => FieldType::Number,
            SchemaFieldType::Boolean => FieldType::Boolean,
            SchemaFieldType::DateTime => FieldType::Date,
            SchemaFieldType::Array(_) => FieldType::Array,
            SchemaFieldType::Object(_) => FieldType::Unknown,
        }
    }
}

/// 字段定义
#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub field_type: FieldType,
}

impl Field {
    pub fn new(name: String, field_type: FieldType) -> Self {
        Self { name, field_type }
    }
    
    pub fn new_with_value_type(name: impl Into<String>, value_type: ValueType) -> Self {
        let field_type = match value_type {
            ValueType::String => FieldType::String,
            ValueType::Integer | ValueType::Float => FieldType::Number,
            ValueType::Boolean => FieldType::Boolean,
            ValueType::Date | ValueType::DateTime | ValueType::Time => FieldType::Date,
            _ => FieldType::Unknown,
        };
        
        Self { name: name.into(), field_type }
    }
    
    pub fn name(&self) -> &str {
        &self.name
    }
    
    pub fn value_type(&self) -> ValueType {
        ValueType::from_field_type(&self.field_type)
    }
}

/// 数据模式
#[derive(Debug, Clone)]
pub struct Schema {
    pub fields: Vec<Field>,
}

impl Schema {
    pub fn new(fields: Vec<Field>) -> Self {
        Self { fields }
    }
    
    pub fn fields(&self) -> &Vec<Field> {
        &self.fields
    }
}

/// 处理CSV文件的处理器
#[derive(Debug)]
pub struct CSVProcessor {
    path: String,
    reader: Option<csv::Reader<BufReader<File>>>,
    schema: Option<Schema>,
    total_rows: Option<usize>,
    current_position: usize,
    options: HashMap<String, String>,
    delimiter: u8,
    has_header: bool,
}

// 定义日期格式的正则表达式常量
lazy_static! {
    // ISO 日期格式: 2023-01-01
    static ref ISO_DATE_REGEX: Regex = Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();
    
    // 日期时间格式: 2023-01-01 12:34:56
    static ref DATETIME_REGEX: Regex = Regex::new(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$").unwrap();
    
    // ISO 日期时间格式: 2023-01-01T12:34:56
    static ref ISO_DATETIME_REGEX: Regex = Regex::new(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$").unwrap();
    
    // ISO 日期时间带毫秒: 2023-01-01T12:34:56.789
    static ref ISO_DATETIME_MS_REGEX: Regex = Regex::new(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{1,9}$").unwrap();
    
    // 带时区的ISO日期时间: 2023-01-01T12:34:56Z, 2023-01-01T12:34:56+08:00
    static ref ISO_DATETIME_TZ_REGEX: Regex = Regex::new(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})$").unwrap();
    
    // 美式日期格式: 01/01/2023
    static ref US_DATE_REGEX: Regex = Regex::new(r"^\d{2}/\d{2}/\d{4}$").unwrap();
    
    // 欧式日期格式: 01.01.2023
    static ref EU_DATE_REGEX: Regex = Regex::new(r"^\d{2}\.\d{2}\.\d{4}$").unwrap();
}

impl CSVProcessor {
    /// 创建新的CSV处理器
    pub fn new(path: &str, options: Option<HashMap<String, String>>) -> Result<Self> {
        let file_path = Path::new(path);
        if !file_path.exists() {
            return Err(Error::file_not_found(format!("File not found: {}", path)));
        }

        let mut processor_options = HashMap::new();
        if let Some(opts) = options {
            processor_options = opts;
        }

        // 获取分隔符配置，默认为逗号
        let delimiter = if processor_options.contains_key("delimiter") {
            processor_options.get("delimiter").unwrap().as_bytes()[0]
        } else {
            b','
        };

        // 获取是否有表头配置，默认为true
        let has_header = if processor_options.contains_key("has_header") {
            processor_options.get("has_header").unwrap() == "true"
        } else {
            true
        };

        Ok(CSVProcessor {
            path: path.to_string(),
            reader: None,
            schema: None,
            total_rows: None,
            current_position: 0,
            options: processor_options,
            delimiter,
            has_header,
        })
    }

    /// 自动检测CSV文件的分隔符
    pub fn detect_delimiter(path: &str) -> Result<u8> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut sample = String::new();
        reader.read_to_string(&mut sample)?;

        // 检查常见分隔符
        let delimiters = [b',', b';', b'\t', b'|'];
        let mut max_count = 0;
        let mut detected_delimiter = b','; // 默认为逗号

        for &delimiter in &delimiters {
            let count = sample.lines().take(10).map(|line| {
                line.chars().filter(|&c| c == delimiter as char).count()
            }).sum::<usize>();

            if count > max_count {
                max_count = count;
                detected_delimiter = delimiter;
            }
        }

        Ok(detected_delimiter)
    }

    /// 初始化CSV阅读器
    fn init_reader(&mut self) -> Result<()> {
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);

        let mut csv_reader = ReaderBuilder::new()
            .delimiter(self.delimiter)
            .has_headers(self.has_header)
            .from_reader(reader);

        self.reader = Some(csv_reader);
        Ok(())
    }

    /// 计算总行数
    fn count_rows(&mut self) -> Result<usize> {
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut csv_reader = ReaderBuilder::new()
            .delimiter(self.delimiter)
            .has_headers(self.has_header)
            .from_reader(reader);

        let mut count = 0;
        
        // 使用ByteRecord可以更高效地统计行数，不进行字符串解析
        let mut record = ByteRecord::new();
        while csv_reader.read_byte_record(&mut record)? {
            count += 1;
        }

        if self.has_header && count > 0 {
            count -= 1; // 减去表头
        }

        debug!("CSV文件 {} 包含 {} 行数据", self.path, count);
        self.total_rows = Some(count);
        Ok(count)
    }

    /// 推断CSV文件的数据模式
    fn infer_schema(&mut self) -> Result<Schema> {
        info!("开始推断CSV文件模式: {}", self.path);
        
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut csv_reader = ReaderBuilder::new()
            .delimiter(self.delimiter)
            .has_headers(self.has_header)
            .from_reader(reader);

        // 读取表头获取字段名
        let headers: Vec<String> = if self.has_header {
            let mut header = StringRecord::new();
            if csv_reader.read_record(&mut header)? {
                header.iter().map(|s| s.to_string()).collect()
            } else {
                warn!("CSV文件为空，没有表头: {}", self.path);
                return Err(Error::data("CSV文件为空，没有表头"));
            }
        } else {
            // 没有表头，使用默认列名
            let mut record = StringRecord::new();
            if csv_reader.read_record(&mut record)? {
                (0..record.len()).map(|i| format!("column_{}", i + 1)).collect()
            } else {
                warn!("CSV文件为空: {}", self.path);
                return Err(Error::data("CSV文件为空"));
            }
        };

        debug!("CSV文件表头: {:?}", headers);

        // 采样一些行来推断数据类型
        let mut fields = Vec::new();
        let mut sample_records = Vec::new();
        let mut record = StringRecord::new();
        
        // 如果有表头，需要重置文件指针
        if self.has_header {
            // 重新创建 reader 并跳过表头，避免嵌套 BufReader 导致 read_line 不可用
            let mut reader = BufReader::new(File::open(&self.path)?);
            reader.seek(SeekFrom::Start(0))?;
            let mut first_line = String::new();
            reader.read_line(&mut first_line)?;
            // 使用新的 reader 重建 csv_reader，且不再读取表头
            csv_reader = ReaderBuilder::new()
                .delimiter(self.delimiter)
                .has_headers(false)
                .from_reader(reader);
        }

        // 读取最多100行进行样本分析
        let sample_size = 100;
        let mut sample_count = 0;
        while sample_count < sample_size && csv_reader.read_record(&mut record)? {
            sample_records.push(record.clone());
            sample_count += 1;
        }

        debug!("收集了 {} 行样本进行类型推断", sample_count);

        // 对每一列进行类型推断
        for (index, header) in headers.iter().enumerate() {
            let mut string_count = 0;
            let mut number_count = 0;
            let mut boolean_count = 0;
            let mut date_count = 0;
            let mut null_count = 0;

            for sample in &sample_records {
                if index >= sample.len() {
                    continue;
                }
                
                let value = sample.get(index).unwrap_or("");
                
                // 尝试解析为JSON值以便可靠地推断类型
                match serde_json::from_str::<JsonValue>(value) {
                    Ok(json_value) => {
                        match json_value {
                            JsonValue::String(s) => {
                                // 检查是否为日期格式
                                if Self::is_date_format(&s) {
                                    date_count += 1;
                                } else {
                                    string_count += 1;
                                }
                            },
                            JsonValue::Number(_) => number_count += 1,
                            JsonValue::Bool(_) => boolean_count += 1,
                            JsonValue::Null => null_count += 1,
                            _ => string_count += 1, // 处理其他类型作为字符串
                        }
                    },
                    Err(_) => {
                        // 如果无法解析为JSON，尝试其他解析方法
                        if value.trim().is_empty() {
                            null_count += 1;
                        } else if value.parse::<f64>().is_ok() {
                            number_count += 1;
                        } else if value.to_lowercase() == "true" || value.to_lowercase() == "false" {
                            boolean_count += 1;
                        } else if Self::is_date_format(value) {
                            date_count += 1;
                        } else {
                            string_count += 1;
                        }
                    }
                }
            }

            // 根据统计结果确定字段类型
            let field_type = if string_count >= number_count && string_count >= boolean_count && string_count >= date_count {
                FieldType::String
            } else if number_count >= string_count && number_count >= boolean_count && number_count >= date_count {
                FieldType::Number
            } else if boolean_count >= string_count && boolean_count >= number_count && boolean_count >= date_count {
                FieldType::Boolean
            } else if date_count >= string_count && date_count >= number_count && date_count >= boolean_count {
                FieldType::Date
            } else {
                FieldType::Unknown
            };

            debug!("列 '{}' 类型推断: string={}, number={}, boolean={}, date={}, null={}, 最终类型={:?}", 
                   header, string_count, number_count, boolean_count, date_count, null_count, field_type);
            
            fields.push(Field::new(header.clone(), field_type));
        }

        let schema = Schema::new(fields);
        info!("完成CSV文件模式推断，共 {} 个字段", schema.fields.len());
        self.schema = Some(schema.clone());
        
        Ok(schema)
    }

    /// 判断字符串是否为日期格式
    pub fn is_date_format(s: &str) -> bool {
        trace!("检查是否为日期格式: {}", s);
        
        // 检查常见日期格式
        if ISO_DATE_REGEX.is_match(s) || 
           DATETIME_REGEX.is_match(s) || 
           ISO_DATETIME_REGEX.is_match(s) || 
           ISO_DATETIME_MS_REGEX.is_match(s) || 
           ISO_DATETIME_TZ_REGEX.is_match(s) || 
           US_DATE_REGEX.is_match(s) || 
           EU_DATE_REGEX.is_match(s) {
            return true;
        }
        
        // 尝试解析为日期时间
        if chrono::DateTime::parse_from_rfc3339(s).is_ok() ||
           chrono::DateTime::parse_from_rfc2822(s).is_ok() ||
           chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").is_ok() ||
           chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").is_ok() {
            return true;
        }
        
        false
    }

    /// 将CSV文件写入到指定路径
    pub fn write_csv(path: &str, records: &[Record], schema: &Schema, options: Option<HashMap<String, String>>) -> Result<()> {
        info!("开始将 {} 条记录写入CSV文件: {}", records.len(), path);
        
        // 创建目录（如果不存在）
        if let Some(parent) = Path::new(path).parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)?;
                debug!("创建目录: {:?}", parent);
            }
        }

        // 设置写入选项
        let delimiter = options.as_ref()
            .and_then(|o| o.get("delimiter"))
            .and_then(|d| d.as_bytes().first())
            .cloned()
            .unwrap_or(b',');

        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        
        let mut csv_writer = WriterBuilder::new()
            .delimiter(delimiter)
            .from_writer(writer);

        // 写入表头
        let headers: Vec<String> = schema.fields.iter()
            .map(|f| f.name.clone())
            .collect();
            
        debug!("写入CSV表头: {:?}", headers);
        csv_writer.write_record(&headers)?;

        // 写入数据行
        for record in records {
            let values = record.values();
            let row: Vec<String> = values.iter().map(|v| v.to_string()).collect();
            csv_writer.write_record(&row)?;
        }

        // 确保所有数据都写入磁盘
        csv_writer.flush()?;
        info!("成功写入 {} 条记录到CSV文件: {}", records.len(), path);
        
        Ok(())
    }
    
    /// 创建共享CSV处理器实例
    pub fn create_shared(path: &str, options: Option<HashMap<String, String>>) -> Result<Arc<CSVProcessor>> {
        let processor = Self::new(path, options)?;
        Ok(Arc::new(processor))
    }

    /// 处理CSV文件并提供详细日志记录
    pub fn process_with_logging(&mut self, log_level: &str) -> Result<Vec<Record>> {
        info!("开始处理CSV文件 '{}' 并记录详细日志，级别: {}", self.path, log_level);
        
        // 初始化日志记录
        let log_enabled = match log_level {
            "debug" => true,
            "info" => true,
            "trace" => true,
            _ => false,
        };
        
        // 初始化读取器
        if self.reader.is_none() {
            self.init_reader()?;
        }
        
        let reader = self.reader.as_mut().unwrap();
        let schema = self.get_schema()?;
        let mut records = Vec::new();
        
        // 记录开始处理
        if log_enabled {
            info!("开始处理CSV文件: {}", self.path);
            info!("CSV模式: {:?}", schema.fields);
        }
        
        // 处理数据行
        let mut row_idx = 0;
        for row_result in reader.records() {
            row_idx += 1;
            
            match row_result {
                Ok(csv_row) => {
                    // 记录行数据
                    if log_level == "trace" {
                        trace!("处理CSV行 {}: {:?}", row_idx, csv_row);
                    }
                    
                    let mut values = Vec::new();
                    let mut field_errors = Vec::new();
                    
                    // 处理每个字段
                    for (i, field) in schema.fields().iter().enumerate() {
                        let value_str = if i < csv_row.len() { csv_row.get(i).unwrap_or("") } else { "" };
                        
                        // 根据字段类型转换值
                        let value_type = ValueType::from_schema_field_type(&field.field_type);
                        let value = match value_type {
                            ValueType::String => DataValue::String(value_str.to_string()),
                            ValueType::Integer => {
                                if value_str.is_empty() {
                                    DataValue::Null
                                } else {
                                    match value_str.parse::<i64>() {
                                        Ok(n) => DataValue::Integer(n),
                                        Err(e) => {
                                            if log_enabled {
                                                warn!("行 {} 字段 '{}' 整数解析错误: {}", row_idx, field.name(), e);
                                            }
                                            field_errors.push(format!("字段 '{}' 整数解析错误: {}", field.name(), e));
                                            DataValue::String(value_str.to_string())
                                        }
                                    }
                                }
                            },
                            ValueType::Float => {
                                if value_str.is_empty() {
                                    DataValue::Null
                                } else {
                                    match value_str.parse::<f64>() {
                                        Ok(n) => DataValue::Number(n),
                                        Err(e) => {
                                            if log_enabled {
                                                warn!("行 {} 字段 '{}' 浮点数解析错误: {}", row_idx, field.name(), e);
                                            }
                                            field_errors.push(format!("字段 '{}' 浮点数解析错误: {}", field.name(), e));
                                            DataValue::String(value_str.to_string())
                                        }
                                    }
                                }
                            },
                            ValueType::Boolean => {
                                if value_str.is_empty() {
                                    DataValue::Null
                                } else {
                                    match value_str.to_lowercase().as_str() {
                                        "true" | "1" | "yes" | "y" => DataValue::Boolean(true),
                                        "false" | "0" | "no" | "n" => DataValue::Boolean(false),
                                        _ => {
                                            if log_enabled {
                                                warn!("行 {} 字段 '{}' 布尔值解析错误: '{}'", row_idx, field.name(), value_str);
                                            }
                                            field_errors.push(format!("字段 '{}' 布尔值解析错误: '{}'", field.name(), value_str));
                                            DataValue::String(value_str.to_string())
                                        }
                                    }
                                }
                            },
                            ValueType::Date => {
                                if value_str.is_empty() {
                                    DataValue::Null
                                } else if Self::is_date_format(value_str) {
                                    DataValue::DateTime(value_str.to_string())
                                } else {
                                    if log_enabled {
                                        warn!("行 {} 字段 '{}' 日期解析错误: '{}'", row_idx, field.name(), value_str);
                                    }
                                    field_errors.push(format!("字段 '{}' 日期解析错误: '{}'", field.name(), value_str));
                                    DataValue::String(value_str.to_string())
                                }
                            },
                            _ => DataValue::String(value_str.to_string()),
                        };
                        
                        values.push(value);
                    }
                    
                    // 记录字段处理结果
                    if !field_errors.is_empty() && log_enabled {
                        warn!("行 {} 处理过程中的字段错误: {}", row_idx, field_errors.join(", "));
                    }
                    
                    // 添加记录
                    records.push(Record::new(values));
                    
                    // 记录处理进度
                    if log_enabled && row_idx % 10000 == 0 {
                        info!("已处理 {} 行 CSV 数据", row_idx);
                    }
                },
                Err(e) => {
                    // 记录解析错误
                    if log_enabled {
                        error!("CSV行 {} 解析错误: {}", row_idx, e);
                    }
                    
                    // 根据配置决定是否忽略错误
                    let ignore_errors = self.options.get("ignore_errors")
                        .map(|v| v == "true")
                        .unwrap_or(false);
                        
                    if !ignore_errors {
                        return Err(Error::io(format!("CSV解析错误(行 {}): {}", row_idx, e)));
                    }
                }
            }
        }
        
        // 记录处理完成
        if log_enabled {
            info!("CSV文件处理完成: {}, 共处理 {} 行", self.path, row_idx);
        }
        
        Ok(records)
    }
    
    /// 验证CSV文件格式
    pub fn validate(&self) -> Result<ValidationResult> {
        info!("验证CSV文件: {}", self.path);
        
        // 打开文件
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        
        // 创建CSV读取器
        let mut csv_reader = ReaderBuilder::new()
            .delimiter(self.delimiter)
            .has_headers(self.has_header)
            .from_reader(reader);
            
        // 检查表头
        let headers = if self.has_header {
            match csv_reader.headers() {
                Ok(headers) => Some(headers.clone()),
                Err(e) => {
                    let mut vr = ValidationResult::failure(vec![format!("CSV表头解析错误: {}", e)]);
                    vr.metadata.insert("stage".into(), "headers".into());
                    return Ok(vr);
                }
            }
        } else {
            None
        };
        
        // 验证结果
        let mut result = ValidationResult::success();
        result.metadata.insert("stage".into(), "rows".into());
        
        // 检查每一行
        let mut row_count = 0;
        let mut empty_rows = 0;
        let mut error_rows = 0;
        let mut inconsistent_columns = 0;
        
        let expected_columns = headers.as_ref().map(|h| h.len()).unwrap_or(0);
        
        for (i, row_result) in csv_reader.records().enumerate() {
            row_count += 1;
            
            match row_result {
                Ok(row) => {
                    // 检查空行
                    if row.len() == 0 || (row.len() == 1 && row.get(0).unwrap_or("").is_empty()) {
                        empty_rows += 1;
                        result.warnings.push(format!("CSV行 {} 为空行", i + 1));
                        continue;
                    }
                    
                    // 检查列数一致性
                    if expected_columns > 0 && row.len() != expected_columns {
                        inconsistent_columns += 1;
                        result.warnings.push(format!(
                            "CSV行 {} 列数不一致: 期望 {}, 实际 {}", 
                            i + 1, expected_columns, row.len()
                        ));
                    }
                },
                Err(e) => {
                    error_rows += 1;
                    result.errors.push(format!("CSV行 {} 解析错误: {}", i + 1, e));
                    
                    // 如果错误太多，提前退出
                    if error_rows > 10 {
                        result.errors.push(format!("错误行数过多，验证中止"));
                        result.is_valid = false;
                        return Ok(result);
                    }
                }
            }
            
            // 限制检查的行数
            if row_count >= 1000 {
                result.warnings.push(format!("CSV文件行数超过1000，部分验证"));
                break;
            }
        }
        
        // 检查错误率
        if error_rows > 0 {
            let error_rate = error_rows as f64 / row_count as f64;
            if error_rate > 0.1 {
                result.errors.push(format!(
                    "CSV错误率过高: {:.2}% ({} / {})", 
                    error_rate * 100.0, error_rows, row_count
                ));
                result.is_valid = false;
            }
        }
        
        // 检查行为空的情况
        if empty_rows == row_count {
            result.errors.push(format!("CSV文件全为空行"));
            result.is_valid = false;
        } else if empty_rows > row_count / 2 {
            result.warnings.push(format!(
                "CSV文件超过一半为空行: {} / {}", 
                empty_rows, row_count
            ));
        }
        
        // 检查列不一致的情况
        if inconsistent_columns > 0 {
            let inconsistent_rate = inconsistent_columns as f64 / row_count as f64;
            if inconsistent_rate > 0.1 {
                result.warnings.push(format!(
                    "CSV文件列数不一致情况较多: {:.2}% ({} / {})", 
                    inconsistent_rate * 100.0, inconsistent_columns, row_count
                ));
            }
        }
        
        // 记录验证结果
        info!("CSV文件验证完成: {}", self.path);
        if result.is_valid {
            info!("验证结果: 有效，行数 {}, 警告 {}", row_count, result.warnings.len());
        } else {
            warn!("验证结果: 无效，行数 {}, 错误 {}, 警告 {}", 
                  row_count, result.errors.len(), result.warnings.len());
        }
        
        Ok(result)
    }
}

use crate::core::interfaces::ValidationResult;

impl FileProcessor for CSVProcessor {
    fn get_file_path(&self) -> &Path {
        Path::new(&self.path)
    }
    
    fn get_schema(&self) -> Result<DataSchema> {
        if self.schema.is_none() {
            // 由于这里需要推断schema，我们需要克隆并推断
            let mut temp_processor = CSVProcessor::new(&self.path, Some(self.options.clone()))?;
            let schema = temp_processor.infer_schema()?;
            
            // 转换为DataSchema
            let mut data_schema = DataSchema::new("csv_schema", "1.0");
            for field in schema.fields() {
                let field_type = field.field_type.to_schema_field_type();
                let field_def = crate::data::schema::schema::FieldDefinition {
                    name: field.name().to_string(),
                    field_type,
                    data_type: None,
                    required: false,
                    nullable: true,
                    primary_key: false,
                    foreign_key: None,
                    description: None,
                    default_value: None,
                    constraints: None,
                    metadata: HashMap::new(),
                };
                data_schema.add_field(field_def)?;
            }
            return Ok(data_schema);
        }
        
        // 如果已有schema，转换为DataSchema
        let schema = self.schema.as_ref().unwrap();
        let mut data_schema = DataSchema::new("csv_schema", "1.0");
        for field in schema.fields() {
            let field_type = field.field_type.to_schema_field_type();
            let field_def = crate::data::schema::schema::FieldDefinition {
                name: field.name().to_string(),
                field_type,
                data_type: None,
                required: false,
                nullable: true,
                primary_key: false,
                foreign_key: None,
                description: None,
                default_value: None,
                constraints: None,
                metadata: HashMap::new(),
            };
            data_schema.add_field(field_def)?;
        }
        
        Ok(data_schema)
    }
    
    fn get_row_count(&self) -> Result<usize> {
        if self.total_rows.is_none() {
            // 创建临时处理器来计算行数
            let mut temp_processor = CSVProcessor::new(&self.path, Some(self.options.clone()))?;
            let count = temp_processor.count_rows()?;
            return Ok(count);
        }
        
        Ok(self.total_rows.unwrap())
    }
    
    fn read_rows(&mut self, count: usize) -> Result<Vec<Record>> {
        if self.reader.is_none() {
            self.init_reader()?;
        }

        let reader = self.reader.as_mut().unwrap();
        let schema = {
            let mut temp_processor = CSVProcessor::new(&self.path, Some(self.options.clone()))?;
            let inferred = temp_processor.infer_schema()?;
            let mut data_schema = DataSchema::new("csv_schema", "1.0");
            for field in inferred.fields() {
                let field_type = field.field_type.to_schema_field_type();
                let field_def = crate::data::schema::schema::FieldDefinition {
                    name: field.name().to_string(),
                    field_type,
                    data_type: None,
                    required: false,
                    nullable: true,
                    primary_key: false,
                    foreign_key: None,
                    description: None,
                    default_value: None,
                    constraints: None,
                    metadata: HashMap::new(),
                };
                data_schema.add_field(field_def)?;
            }
            data_schema
        };
        let mut records = Vec::with_capacity(count);

        for record_result in reader.records().take(count) {
            match record_result {
                Ok(csv_record) => {
                    let mut values = Vec::with_capacity(schema.fields().len());

                    for (i, field) in schema.fields().iter().enumerate() {
                        let value_str = if i < csv_record.len() { csv_record.get(i).unwrap_or("") } else { "" };

                        let value = match &field.field_type {
                            crate::data::schema::schema::FieldType::Boolean => {
                                if value_str.is_empty() {
                                    DataValue::Null
                                } else {
                                    match value_str.to_lowercase().as_str() {
                                        "true" | "1" | "yes" | "y" => DataValue::Boolean(true),
                                        "false" | "0" | "no" | "n" => DataValue::Boolean(false),
                                        _ => DataValue::String(value_str.to_string()),
                                    }
                                }
                            }
                            crate::data::schema::schema::FieldType::Numeric => {
                                if value_str.is_empty() {
                                    DataValue::Null
                                } else if let Ok(i) = value_str.parse::<i64>() {
                                    DataValue::Integer(i)
                                } else if let Ok(f) = value_str.parse::<f64>() {
                                    DataValue::Float(f)
                                } else {
                                    DataValue::String(value_str.to_string())
                                }
                            }
                            _ => {
                                if value_str.is_empty() {
                                    DataValue::Null
                                } else {
                                    DataValue::String(value_str.to_string())
                                }
                            }
                        };

                        values.push(value);
                    }

                    let record = Record::new(values);
                    records.push(record);
                },
                Err(e) => {
                    return Err(Error::parse_error(format!("Error reading CSV record: {}", e)));
                }
            }
            
            self.current_position += 1;
        }
        
        Ok(records)
    }
    
    fn reset(&mut self) -> Result<()> {
        self.current_position = 0;
        if self.reader.is_some() {
            self.init_reader()?;
        }
        Ok(())
    }
}

// 下面实现的这些方法不属于FileProcessor trait接口，但保留作为CSVProcessor的实例方法

impl CSVProcessor {
    // 这些是CSVProcessor特有的方法
    fn get_current_position(&self) -> usize {
        self.current_position
    }
    
    fn process_stream<R: Read>(&mut self, stream: R) -> Result<Vec<Record>> {
        let reader = BufReader::new(stream);
        let mut csv_reader = ReaderBuilder::new()
            .delimiter(self.delimiter)
            .has_headers(self.has_header)
            .from_reader(reader);

        let schema = if self.schema.is_none() {
            let headers: Vec<String> = if self.has_header {
                match csv_reader.headers() {
                    Ok(headers) => headers.iter().map(|s| s.to_string()).collect(),
                    Err(e) => return Err(Error::data(format!("Failed to read CSV headers: {}", e))),
                }
            } else {
                // 如果没有表头，会在读取第一行时处理
                Vec::new()
            };

            // 创建临时模式
            let fields = headers.iter()
                .map(|name| Field::new(name.to_string(), FieldType::String))
                .collect::<Vec<_>>();
            
            Schema::new(fields)
        } else {
            self.schema.clone().unwrap()
        };

        let mut records = Vec::new();

        for record_result in csv_reader.records() {
            match record_result {
                Ok(csv_record) => {
                    let mut values = Vec::with_capacity(schema.fields().len());
                    
                    for (i, field) in schema.fields().iter().enumerate() {
                        let value_str = if i < csv_record.len() { csv_record.get(i).unwrap_or("") } else { "" };
                        
                        let value = match field.value_type() {
                            ValueType::Boolean => {
                                if value_str.is_empty() {
                                    DataValue::Null
                                } else {
                                    match value_str.to_lowercase().as_str() {
                                        "true" | "1" | "yes" | "y" => DataValue::Boolean(true),
                                        "false" | "0" | "no" | "n" => DataValue::Boolean(false),
                                        _ => DataValue::String(value_str.to_string()),
                                    }
                                }
                            },
                            ValueType::Integer => {
                                if value_str.is_empty() {
                                    DataValue::Null
                                } else {
                                    match value_str.parse::<i64>() {
                                        Ok(i) => DataValue::Integer(i),
                                        Err(_) => DataValue::String(value_str.to_string()),
                                    }
                                }
                            },
                            ValueType::Float => {
                                if value_str.is_empty() {
                                    DataValue::Null
                                } else {
                                    match value_str.parse::<f64>() {
                                        Ok(f) => DataValue::Float(f),
                                        Err(_) => DataValue::String(value_str.to_string()),
                                    }
                                }
                            },
                            _ => {
                                if value_str.is_empty() {
                                    DataValue::Null
                                } else {
                                    DataValue::String(value_str.to_string())
                                }
                            },
                        };
                        
                        values.push(value);
                    }
                    
                    let record = Record::new(values);
                    records.push(record);
                },
                Err(e) => {
                    warn!("Error reading CSV record from stream: {}", e);
                    continue;
                }
            }
        }
        
        Ok(records)
    }
}

/// 推断CSV文件的数据模式
pub fn infer_csv_schema(path: &Path) -> Result<DataSchema> {
    let path_str = path.to_str().ok_or_else(|| Error::data("无效的文件路径"))?;
    
    // 创建CSV处理器
    let mut processor = CSVProcessor::new(path_str, None)?;
    
    // 使用处理器获取模式
    processor.get_schema()
} 