use std::path::{Path, PathBuf};
use std::io::{Read, BufReader, Seek, SeekFrom};
use std::fs::File;
use std::collections::HashMap;
use regex::Regex;
use crate::error::{Error, Result};
use crate::data::{DataSchema, DataField, FieldType, FieldSource, SchemaMetadata};
use log::{debug, info, warn};

/// 自定义格式处理器特性
pub trait CustomFormatProcessor {
    /// 读取数据样本
    fn read_sample(&self, max_bytes: usize) -> Result<Vec<u8>>;
    
    /// 检测文件类型
    fn detect_file_type(&self) -> Result<String>;
    
    /// 获取估计的行数或记录数
    fn estimate_record_count(&self) -> Result<usize>;
    
    /// 读取数据样本记录
    fn read_sample_records(&self, max_records: usize) -> Result<Vec<HashMap<String, String>>>;
}

/// 自定义文本格式处理器
pub struct CustomTextProcessor {
    path: PathBuf,
    delimiter: Option<String>,
    record_pattern: Option<Regex>,
}

impl CustomTextProcessor {
    /// 创建新的自定义文本处理器
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            delimiter: None,
            record_pattern: None,
        }
    }
    
    /// 设置记录分隔符
    pub fn with_delimiter(mut self, delimiter: String) -> Self {
        self.delimiter = Some(delimiter);
        self
    }
    
    /// 设置记录模式
    pub fn with_record_pattern(mut self, pattern: &str) -> Result<Self> {
        let regex = Regex::new(pattern)
            .map_err(|e| Error::invalid_input(format!("无效的正则表达式: {}", e)))?;
            
        self.record_pattern = Some(regex);
        Ok(self)
    }
    
    /// 尝试自动检测记录分隔方式
    pub fn detect_record_format(&mut self) -> Result<()> {
        // 读取文件样本
        let sample = self.read_sample(10 * 1024)?; // 读取10KB样本
        let sample_text = String::from_utf8_lossy(&sample);
        
        // 尝试常见的行分隔符
        let newline_count = sample_text.matches('\n').count();
        let carriage_return_count = sample_text.matches('\r').count();
        
        // 检查是否有空行作为记录分隔符
        let empty_line_count = Regex::new(r"\n\s*\n").unwrap().find_iter(&sample_text).count();
        
        // 如果有大量空行，可能是以空行分隔的记录
        if empty_line_count > 0 && empty_line_count * 3 < newline_count {
            self.record_pattern = Some(Regex::new(r"(?m)^(.+?)(?:\n\s*\n|\z)").unwrap());
            return Ok(());
        }
        
        // 检查常见的分隔符模式
        let common_delimiters = ["|", ":::", ";;;", "###"];
        for delimiter in common_delimiters.iter() {
            if sample_text.contains(delimiter) {
                self.delimiter = Some(delimiter.to_string());
                return Ok(());
            }
        }
        
        // 如果没有检测到特殊分隔符，使用换行符作为默认分隔符
        if carriage_return_count > 0 && carriage_return_count >= newline_count / 2 {
            self.delimiter = Some("\r\n".to_string());
        } else {
            self.delimiter = Some("\n".to_string());
        }
        
        Ok(())
    }
    
    /// 解析单个记录
    fn parse_record(&self, record_text: &str) -> HashMap<String, String> {
        let mut record = HashMap::new();
        
        // 如果有分隔符，按分隔符分割
        if let Some(delimiter) = &self.delimiter {
            let parts: Vec<&str> = record_text.split(delimiter).collect();
            for (i, part) in parts.iter().enumerate() {
                record.insert(format!("field_{}", i + 1), part.trim().to_string());
            }
            return record;
        }
        
        // 如果没有分隔符但有模式，使用模式进行捕获
        if let Some(pattern) = &self.record_pattern {
            if let Some(captures) = pattern.captures(record_text) {
                for i in 1..captures.len() {
                    if let Some(value) = captures.get(i) {
                        record.insert(format!("field_{}", i), value.as_str().to_string());
                    }
                }
                return record;
            }
        }
        
        // 默认情况下，将整个记录作为一个字段
        record.insert("content".to_string(), record_text.to_string());
        record
    }
}

impl CustomFormatProcessor for CustomTextProcessor {
    fn read_sample(&self, max_bytes: usize) -> Result<Vec<u8>> {
        let mut file = File::open(&self.path)
            .map_err(|e| Error::io_error(format!("打开文件失败: {}", e)))?;
            
        let file_size = file.metadata()
            .map_err(|e| Error::io_error(format!("获取文件信息失败: {}", e)))?
            .len() as usize;
            
        let bytes_to_read = max_bytes.min(file_size);
        let mut buffer = vec![0u8; bytes_to_read];
        
        file.read_exact(&mut buffer[..bytes_to_read])
            .map_err(|e| Error::io_error(format!("读取文件样本失败: {}", e)))?;
            
        Ok(buffer)
    }
    
    fn detect_file_type(&self) -> Result<String> {
        // 读取文件前几个字节来判断是否是文本文件
        let sample = self.read_sample(1024)?;
        
        // 检查是否包含空字节（通常表示二进制文件）
        let has_null_byte = sample.iter().any(|&b| b == 0);
        if has_null_byte {
            return Ok("binary".to_string());
        }
        
        // 尝试将样本解析为UTF-8文本
        match String::from_utf8(sample.clone()) {
            Ok(_) => Ok("text".to_string()),
            Err(_) => {
                // 检查是否可能是其他编码的文本
                // 这里简化处理，只检查常见ASCII范围
                let printable_chars = sample.iter().filter(|&&b| b >= 32 && b <= 126).count();
                if printable_chars > sample.len() * 8 / 10 {
                    Ok("text".to_string())
                } else {
                    Ok("binary".to_string())
                }
            }
        }
    }
    
    fn estimate_record_count(&self) -> Result<usize> {
        let file_size = std::fs::metadata(&self.path)
            .map_err(|e| Error::io_error(format!("获取文件大小失败: {}", e)))?
            .len() as usize;
            
        // 读取样本估算平均记录大小
        let sample = self.read_sample(1024 * 100)?; // 读取100KB样本
        let sample_text = String::from_utf8_lossy(&sample);
        
        let delimiter = self.delimiter.as_deref().unwrap_or("\n");
        let records: Vec<&str> = sample_text.split(delimiter).collect();
        
        if records.is_empty() {
            return Ok(0);
        }
        
        // 计算平均记录大小
        let avg_record_size = sample.len() / records.len();
        if avg_record_size == 0 {
            return Ok(file_size); // 防止除零错误
        }
        
        // 估算总记录数
        Ok(file_size / avg_record_size)
    }
    
    fn read_sample_records(&self, max_records: usize) -> Result<Vec<HashMap<String, String>>> {
        let mut file = File::open(&self.path)
            .map_err(|e| Error::io_error(format!("打开文件失败: {}", e)))?;
            
        let mut reader = BufReader::new(file);
        let mut buffer = Vec::new();
        
        // 读取部分文件内容
        reader.read_to_end(&mut buffer)
            .map_err(|e| Error::io_error(format!("读取文件失败: {}", e)))?;
            
        let text = String::from_utf8_lossy(&buffer);
        
        // 根据分隔符或模式分割记录
        let records = if let Some(delimiter) = &self.delimiter {
            text.split(delimiter).collect::<Vec<_>>()
        } else if let Some(pattern) = &self.record_pattern {
            pattern.find_iter(&text)
                .map(|m| m.as_str())
                .collect::<Vec<_>>()
        } else {
            vec![&text]
        };
        
        // 解析记录
        let mut parsed_records = Vec::new();
        for (i, record_text) in records.iter().enumerate() {
            if i >= max_records {
                break;
            }
            
            if !record_text.trim().is_empty() {
                let parsed = self.parse_record(record_text);
                parsed_records.push(parsed);
            }
        }
        
        Ok(parsed_records)
    }
}

/// 自定义二进制格式处理器
pub struct CustomBinaryProcessor {
    path: PathBuf,
    record_size: Option<usize>,
    header_size: usize,
}

impl CustomBinaryProcessor {
    /// 创建新的自定义二进制处理器
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            record_size: None,
            header_size: 0,
        }
    }
    
    /// 设置固定记录大小
    pub fn with_record_size(mut self, size: usize) -> Self {
        self.record_size = Some(size);
        self
    }
    
    /// 设置头部大小
    pub fn with_header_size(mut self, size: usize) -> Self {
        self.header_size = size;
        self
    }
    
    /// 尝试检测记录大小
    pub fn detect_record_size(&mut self) -> Result<()> {
        // 读取文件样本
        let sample = self.read_sample(4096)?;
        
        // 尝试检测常见的二进制格式
        // 这里简化处理，仅检查几种常见的固定长度记录格式
        
        // 检查是否有固定长度的记录标记
        let potential_sizes = [8, 16, 32, 64, 128, 256, 512, 1024];
        
        for &size in &potential_sizes {
            if sample.len() >= size * 2 {
                // 检查样本中是否有规律性的模式
                let mut found = true;
                for i in 0..2 {
                    let start = i * size;
                    let end = start + 4;
                    if end < sample.len() && sample[start..end] == sample[size+start..size+end] {
                        continue;
                    }
                    found = false;
                    break;
                }
                
                if found {
                    self.record_size = Some(size);
                    return Ok(());
                }
            }
        }
        
        // 默认设置为64字节
        self.record_size = Some(64);
        Ok(())
    }
    
    /// 解析二进制记录为字符串映射
    fn parse_binary_record(&self, data: &[u8]) -> HashMap<String, String> {
        let mut record = HashMap::new();
        
        // 简单地将二进制数据按固定长度分割
        let chunk_size = 8; // 假设每8字节为一个字段
        
        for (i, chunk) in data.chunks(chunk_size).enumerate() {
            let value = if chunk.len() == 8 {
                // 尝试将8字节解析为数值
                let value = u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3],
                    chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                value.to_string()
            } else {
                // 否则使用十六进制表示
                format!("{:02X?}", chunk)
            };
            
            record.insert(format!("field_{}", i + 1), value);
        }
        
        record
    }
}

impl CustomFormatProcessor for CustomBinaryProcessor {
    fn read_sample(&self, max_bytes: usize) -> Result<Vec<u8>> {
        let mut file = File::open(&self.path)
            .map_err(|e| Error::io_error(format!("打开文件失败: {}", e)))?;
            
        let file_size = file.metadata()
            .map_err(|e| Error::io_error(format!("获取文件信息失败: {}", e)))?
            .len() as usize;
            
        let bytes_to_read = max_bytes.min(file_size);
        let mut buffer = vec![0u8; bytes_to_read];
        
        file.read_exact(&mut buffer[..bytes_to_read])
            .map_err(|e| Error::io_error(format!("读取文件样本失败: {}", e)))?;
            
        Ok(buffer)
    }
    
    fn detect_file_type(&self) -> Result<String> {
        Ok("binary".to_string())
    }
    
    fn estimate_record_count(&self) -> Result<usize> {
        let file_size = std::fs::metadata(&self.path)
            .map_err(|e| Error::io_error(format!("获取文件大小失败: {}", e)))?
            .len() as usize;
            
        let data_size = file_size.saturating_sub(self.header_size);
        
        if let Some(record_size) = self.record_size {
            if record_size == 0 {
                return Ok(0); // 防止除零错误
            }
            Ok(data_size / record_size)
        } else {
            // 默认假设每个记录64字节
            Ok(data_size / 64)
        }
    }
    
    fn read_sample_records(&self, max_records: usize) -> Result<Vec<HashMap<String, String>>> {
        let mut file = File::open(&self.path)
            .map_err(|e| Error::io_error(format!("打开文件失败: {}", e)))?;
            
        // 跳过头部
        if self.header_size > 0 {
            file.seek(SeekFrom::Start(self.header_size as u64))
                .map_err(|e| Error::io_error(format!("跳过文件头部失败: {}", e)))?;
        }
        
        let record_size = self.record_size.unwrap_or(64);
        let bytes_to_read = record_size * max_records;
        
        let mut buffer = vec![0u8; bytes_to_read];
        let bytes_read = file.read(&mut buffer)
            .map_err(|e| Error::io_error(format!("读取文件样本失败: {}", e)))?;
            
        buffer.truncate(bytes_read);
        
        // 按记录大小分割
        let mut records = Vec::new();
        for chunk in buffer.chunks(record_size) {
            if chunk.len() == record_size {
                let parsed = self.parse_binary_record(chunk);
                records.push(parsed);
            }
        }
        
        Ok(records)
    }
}

/// 推断自定义文本格式的数据模式
pub fn infer_custom_text_schema_impl(path: &Path) -> Result<DataSchema> {
    info!("从自定义文本文件推断数据模式: {}", path.display());
    
    let mut processor = CustomTextProcessor::new(path);
    
    // 自动检测记录格式
    processor.detect_record_format()?;
    
    // 读取样本记录
    let sample_records = processor.read_sample_records(100)?;
    if sample_records.is_empty() {
        return Err(Error::invalid_data("无法从文件中读取有效记录".to_string()));
    }
    
    // 从样本记录中推断字段
    let mut fields = HashMap::new();
    for record in &sample_records {
        for (key, value) in record {
            if !fields.contains_key(key) {
                // 推断字段类型
                let field_type = infer_field_type(value);
                fields.insert(key.clone(), (field_type, false));
            }
        }
    }
    
    // 构建数据字段列表
    let mut data_fields = Vec::new();
    for (name, (field_type, nullable)) in fields {
        data_fields.push(DataField {
            name,
            field_type,
            required: !nullable,
            source: FieldSource::File,
            description: None,
            default_value: None,
        });
    }
    
    // 估计行数
    let row_count = processor.estimate_record_count()?;
    
    // 创建模式元数据
    let metadata = SchemaMetadata {
        source: "file".to_string(),
        format: "custom_text".to_string(),
        path: path.to_str().unwrap_or("").to_string(),
        record_count: Some(row_count),
        created_at: chrono::Utc::now(),
        version: "1.0".to_string(),
        delimiter: processor.delimiter.clone(),
        ..Default::default()
    };
    
    Ok(DataSchema {
        fields: data_fields,
        metadata: Some(metadata),
    })
}

/// 推断自定义二进制格式的数据模式
pub fn infer_custom_binary_schema_impl(path: &Path) -> Result<DataSchema> {
    info!("从自定义二进制文件推断数据模式: {}", path.display());
    
    let mut processor = CustomBinaryProcessor::new(path);
    
    // 自动检测记录大小
    processor.detect_record_size()?;
    
    // 读取样本记录
    let sample_records = processor.read_sample_records(50)?;
    if sample_records.is_empty() {
        return Err(Error::invalid_data("无法从文件中读取有效记录".to_string()));
    }
    
    // 从样本记录中推断字段
    let mut fields = HashMap::new();
    for record in &sample_records {
        for (key, value) in record {
            if !fields.contains_key(key) {
                // 推断字段类型
                let field_type = infer_field_type(value);
                fields.insert(key.clone(), (field_type, false));
            }
        }
    }
    
    // 构建数据字段列表
    let mut data_fields = Vec::new();
    for (name, (field_type, nullable)) in fields {
        data_fields.push(DataField {
            name,
            field_type,
            required: !nullable,
            source: FieldSource::File,
            description: None,
            default_value: None,
        });
    }
    
    // 估计行数
    let row_count = processor.estimate_record_count()?;
    
    // 创建模式元数据
    let metadata = SchemaMetadata {
        source: "file".to_string(),
        format: "custom_binary".to_string(),
        path: path.to_str().unwrap_or("").to_string(),
        record_count: Some(row_count),
        created_at: chrono::Utc::now(),
        version: "1.0".to_string(),
        binary_record_size: processor.record_size,
        binary_header_size: Some(processor.header_size),
        ..Default::default()
    };
    
    Ok(DataSchema {
        fields: data_fields,
        metadata: Some(metadata),
    })
}

/// 根据字符串值推断字段类型
fn infer_field_type(value: &str) -> FieldType {
    if value.trim().is_empty() {
        return FieldType::Null;
    }
    
    // 尝试解析布尔值
    match value.to_lowercase().as_str() {
        "true" | "false" | "yes" | "no" | "y" | "n" | "1" | "0" => {
            return FieldType::Boolean;
        },
        _ => {}
    }
    
    // 尝试解析整数
    if value.parse::<i64>().is_ok() {
        return FieldType::Integer;
    }
    
    // 尝试解析浮点数
    if value.parse::<f64>().is_ok() {
        return FieldType::Float;
    }
    
    // 尝试解析日期
    let date_patterns = [
        r"^\d{4}-\d{2}-\d{2}$",                      // yyyy-MM-dd
        r"^\d{2}/\d{2}/\d{4}$",                      // MM/dd/yyyy
        r"^\d{2}\.\d{2}\.\d{4}$",                    // dd.MM.yyyy
    ];
    
    for pattern in &date_patterns {
        if Regex::new(pattern).unwrap().is_match(value) {
            return FieldType::Date;
        }
    }
    
    // 尝试解析时间戳
    let timestamp_patterns = [
        r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}",  // yyyy-MM-dd HH:mm:ss
        r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}",        // yyyy-MM-dd HH:mm
    ];
    
    for pattern in &timestamp_patterns {
        if Regex::new(pattern).unwrap().is_match(value) {
            return FieldType::Timestamp;
        }
    }
    
    // 检查是否可能是JSON数组
    if value.trim().starts_with('[') && value.trim().ends_with(']') {
        return FieldType::Array;
    }
    
    // 检查是否可能是JSON对象
    if value.trim().starts_with('{') && value.trim().ends_with('}') {
        return FieldType::Object;
    }
    
    // 默认为字符串类型
    FieldType::String
} 