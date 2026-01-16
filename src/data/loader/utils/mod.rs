use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::{self, Read, BufReader, BufWriter, Write, BufRead};
use log::{debug, error, warn};
use regex::Regex;

use crate::data::DataBatch;
use crate::data::loader::types::{FileType, DataFormat};
use crate::error::{Error, Result};
// 预留：将来用于更复杂的文件扫描过滤规则
// use std::collections::HashMap;

/// 用于检测文件类型的基本字节签名
const FILE_SIGNATURES: &[(&[u8], FileType)] = &[
    (&[0x50, 0x4B, 0x03, 0x04], FileType::Excel), // XLSX (ZIP)
    (&[0x50, 0x41, 0x52, 0x31], FileType::Parquet), // Parquet
    (&[0x4F, 0x62, 0x6A, 0x01], FileType::Avro), // Avro
    (&[0x53, 0x51, 0x4C, 0x69, 0x74, 0x65], FileType::Sqlite), // SQLite
];

/// 扫描目录，获取匹配的文件
pub fn scan_directory<P: AsRef<Path>>(
    directory: P,
    pattern: Option<&str>,
    recursive: bool,
) -> Result<Vec<PathBuf>> {
    let dir_path = directory.as_ref();
    debug!("扫描目录: {:?}, 模式: {:?}, 递归: {}", dir_path, pattern, recursive);
    
    if !dir_path.exists() || !dir_path.is_dir() {
        return Err(Error::invalid_input(format!("指定的路径不是有效目录: {:?}", dir_path)));
    }
    
    let mut result = Vec::new();
    let regex_pattern = if let Some(p) = pattern {
        Some(Regex::new(p).map_err(|e| {
            Error::invalid_input(format!("无效的正则表达式模式: {}", e))
        })?)
    } else {
        None
    };
    
    scan_directory_internal(dir_path, &regex_pattern, recursive, &mut result)?;
    
    Ok(result)
}

/// 内部目录扫描实现
fn scan_directory_internal(
    directory: &Path,
    pattern: &Option<Regex>,
    recursive: bool,
    result: &mut Vec<PathBuf>,
) -> Result<()> {
    for entry in fs::read_dir(directory)
        .map_err(|e| Error::io_error(format!("读取目录失败: {}", e)))? {
        let entry = entry.map_err(|e| Error::io_error(format!("读取目录项失败: {}", e)))?;
        let path = entry.path();
        
        if path.is_dir() && recursive {
            scan_directory_internal(&path, pattern, recursive, result)?;
        } else if path.is_file() {
            let path_str = path.to_string_lossy();
            
            let matches = if let Some(re) = pattern {
                re.is_match(&path_str)
            } else {
                true
            };
            
            if matches {
                result.push(path);
            }
        }
    }
    
    Ok(())
}

/// 检测文件格式
pub fn detect_file_format<P: AsRef<Path>>(path: P) -> Result<DataFormat> {
    let path_ref = path.as_ref();
    let file_type = FileType::from_path(path_ref);

    match file_type {
        FileType::Unknown => {
            // 尝试通过读取文件头部来检测
            let detected_type = detect_file_type_by_content(path_ref)?;
            Ok(DataFormat::from_file_type(detected_type))
        },
        _ => {
            // 基于文件扩展名创建适当的数据格式
            Ok(DataFormat::from_file_type(file_type))
        }
    }
}

/// 通过文件内容检测文件类型
fn detect_file_type_by_content<P: AsRef<Path>>(path: P) -> Result<FileType> {
    let path_ref = path.as_ref();
    let mut file = match File::open(path_ref) {
        Ok(f) => f,
        Err(e) => {
            return Err(Error::io_error(format!("无法打开文件 {:?}: {}", path_ref, e)));
        }
    };
    
    let mut buffer = [0u8; 16];
    let read_count = match file.read(&mut buffer) {
        Ok(count) => count,
        Err(e) => {
            return Err(Error::io_error(format!("读取文件 {:?} 失败: {}", path_ref, e)));
        }
    };
    
    // 如果文件太小，可能无法确定其类型
    if read_count < 4 {
        warn!("文件太小，无法确定类型: {:?}", path_ref);
        return Ok(FileType::Text); // 默认为文本文件
    }
    
    // 首先检查文件签名
    for (signature, file_type) in FILE_SIGNATURES {
        if read_count >= signature.len() && &buffer[..signature.len()] == *signature {
            return Ok(*file_type);
        }
    }
    
    // 检查是否为JSON
    if (buffer[0] == b'{' || buffer[0] == b'[') && is_valid_json(path_ref) {
        return Ok(FileType::Json);
    }
    
    // 检查是否为CSV
    if is_valid_csv(path_ref) {
        return Ok(FileType::Csv);
    }
    
    // 检查是否为XML
    if read_count >= 5 && &buffer[..5] == b"<?xml" {
        return Ok(FileType::Xml);
    }
    
    // 默认为文本文件
    Ok(FileType::Text)
}

/// 检查文件是否为有效的JSON
fn is_valid_json<P: AsRef<Path>>(path: P) -> bool {
    let file = match File::open(path.as_ref()) {
        Ok(f) => f,
        Err(_) => return false,
    };
    
    let reader = BufReader::new(file);
    match serde_json::from_reader::<_, serde_json::Value>(reader) {
        Ok(_) => true,
        Err(_) => false,
    }
}

/// 检查文件是否为有效的CSV
fn is_valid_csv<P: AsRef<Path>>(path: P) -> bool {
    let file = match File::open(path.as_ref()) {
        Ok(f) => f,
        Err(_) => return false,
    };
    
    let mut reader = csv::Reader::from_reader(file);
    match reader.headers() {
        Ok(_) => true,
        Err(_) => false,
    }
}

/// 将数据批次保存到文件
pub fn save_batch_to_file<P: AsRef<Path>>(
    batch: &DataBatch,
    output_path: P,
    format: DataFormat,
) -> Result<PathBuf> {
    let path = output_path.as_ref();
    
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)
                .map_err(|e| Error::io_error(format!("创建目录失败: {}", e)))?;
        }
    }
    
    match format {
        DataFormat::Csv { delimiter, has_header, .. } => {
            save_batch_as_csv(batch, path, delimiter, has_header)?;
        },
        DataFormat::Json { is_lines, is_array, .. } => {
            save_batch_as_json(batch, path, is_lines, is_array)?;
        },
        DataFormat::Parquet { .. } => {
            save_batch_as_parquet(batch, path)?;
        },
        _ => {
            return Err(Error::not_implemented(format!(
                "不支持保存为 {:?} 格式", format
            )));
        }
    }
    
    Ok(path.to_path_buf())
}

/// 将数据批次保存为CSV
fn save_batch_as_csv<P: AsRef<Path>>(
    batch: &DataBatch,
    path: P,
    delimiter: char,
    has_header: bool,
) -> Result<()> {
    let writer = File::create(path.as_ref())
        .map_err(|e| Error::io_error(format!("无法创建CSV文件: {}", e)))?;
    
    let mut csv_writer = csv::WriterBuilder::new()
        .delimiter(delimiter as u8)
        .has_headers(has_header)
        .from_writer(writer);
    
    // 如果有模式且需要写入标题，先写入字段名称
    if has_header && batch.schema.is_some() {
        let schema = batch.schema.as_ref().unwrap();
        let field_names: Vec<String> = schema.fields().iter()
            .map(|f| f.name.clone())
            .collect();
        
        csv_writer.write_record(&field_names)
            .map_err(|e| Error::io_error(format!("写入CSV标题失败: {}", e)))?;
    }
    
    // 写入记录
    for record in &batch.records {
        let values: Vec<String> = if let Some(schema) = &batch.schema {
            schema.fields().iter()
                .map(|field| {
                    // HashMap<String, DataValue> 使用 get 方法
                    record.get(&field.name)
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| String::new()) // 字段不存在时使用空字符串
                })
                .collect()
        } else {
            // 没有模式时，使用记录的所有字段
            record.values()
                .map(|v| v.to_string())
                .collect()
        };
        
        csv_writer.write_record(&values)
            .map_err(|e| Error::io_error(format!("写入CSV记录失败: {}", e)))?;
    }
    
    csv_writer.flush()
        .map_err(|e| Error::io_error(format!("刷新CSV写入器失败: {}", e)))?;
    
    Ok(())
}

/// 将数据批次保存为JSON
fn save_batch_as_json<P: AsRef<Path>>(
    batch: &DataBatch,
    path: P,
    is_lines: bool,
    is_array: bool,
) -> Result<()> {
    let file = File::create(path.as_ref())
        .map_err(|e| Error::io_error(format!("无法创建JSON文件: {}", e)))?;
    
    if is_lines {
        // JSONL格式（每行一个JSON对象）
        let mut writer = io::BufWriter::new(file);
        for record in &batch.records {
            let json_value = serde_json::to_value(record)
                .map_err(|e| Error::serialization(format!("序列化记录失败: {}", e)))?;
            
            let json_str = serde_json::to_string(&json_value)
                .map_err(|e| Error::serialization(format!("JSON字符串化失败: {}", e)))?;
            
            writeln!(writer, "{}", json_str)
                .map_err(|e| Error::io_error(format!("写入JSONL行失败: {}", e)))?;
        }
        
        writer.flush()
            .map_err(|e| Error::io_error(format!("刷新JSONL写入器失败: {}", e)))?;
    } else {
        // 标准JSON格式
        if is_array {
            // 记录数组
            let json = serde_json::to_value(&batch.records)
                .map_err(|e| Error::serialization(format!("序列化批次记录失败: {}", e)))?;
            
            serde_json::to_writer_pretty(file, &json)
                .map_err(|e| Error::io_error(format!("写入JSON数组失败: {}", e)))?;
        } else {
            // 批次对象
            let json = serde_json::to_value(batch)
                .map_err(|e| Error::serialization(format!("序列化批次失败: {}", e)))?;
            
            serde_json::to_writer_pretty(file, &json)
                .map_err(|e| Error::io_error(format!("写入JSON对象失败: {}", e)))?;
        }
    }
    
    Ok(())
}

/// 将数据批次保存为Parquet
#[cfg(feature = "parquet")]
fn save_batch_as_parquet<P: AsRef<Path>>(batch: &DataBatch, path: P) -> Result<()> {
    use parquet::basic::Compression;
    use parquet::column::writer::{ColumnWriter, Writer};
    use parquet::data_type::ByteArray;
    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::parser::parse_message_type;
    use std::fs::File;
    use std::sync::Arc;
    use std::collections::BTreeSet;
    
    info!("开始保存数据批次为Parquet格式: {} 条记录", batch.records.len());
    
    if batch.records.is_empty() {
        warn!("数据批次为空，无法保存为Parquet");
        return Err(Error::invalid_input("数据批次为空，无法保存为Parquet格式".to_string()));
    }
    
    // 收集所有字段名
    let mut field_names = BTreeSet::new();
    for record in &batch.records {
        for key in record.keys() {
            field_names.insert(key.clone());
        }
    }
    let fields: Vec<String> = field_names.into_iter().collect();
    
    // 推断字段类型
    #[derive(Clone, Copy)]
    enum ColType { Utf8, F64, I64, Bool }
    
    fn infer_col_type(name: &str, records: &[HashMap<String, crate::data::DataValue>]) -> ColType {
        let mut saw_str = false;
        let mut saw_num = false;
        let mut saw_int = false;
        let mut saw_bool = false;
        
        for record in records {
            if let Some(v) = record.get(name) {
                match v {
                    crate::data::DataValue::String(_) => saw_str = true,
                    crate::data::DataValue::Float(_) | crate::data::DataValue::Number(_) => saw_num = true,
                    crate::data::DataValue::Integer(_) => saw_int = true,
                    crate::data::DataValue::Boolean(_) => saw_bool = true,
                    _ => saw_str = true,
                }
            }
        }
        
        if saw_bool && !saw_str && !saw_num && !saw_int {
            ColType::Bool
        } else if saw_int && !saw_num && !saw_str {
            ColType::I64
        } else if saw_num {
            ColType::F64
        } else {
            ColType::Utf8
        }
    }
    
    let col_types: Vec<ColType> = fields.iter()
        .map(|f| infer_col_type(f, &batch.records))
        .collect();
    
    // 构建Parquet模式
    fn sanitize_field_name(name: &str) -> String {
        name.chars()
            .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
            .collect()
    }
    
    let mut schema_msg = String::from("message schema {\n");
    for (name, ct) in fields.iter().zip(col_types.iter()) {
        let sanitized = sanitize_field_name(name);
        let line = match ct {
            ColType::Utf8 => format!("  OPTIONAL BINARY {} (UTF8);\n", sanitized),
            ColType::F64 => format!("  OPTIONAL DOUBLE {};\n", sanitized),
            ColType::I64 => format!("  OPTIONAL INT64 {};\n", sanitized),
            ColType::Bool => format!("  OPTIONAL BOOLEAN {};\n", sanitized),
        };
        schema_msg.push_str(&line);
    }
    schema_msg.push_str("}\n");
    
    // 解析模式
    let schema = Arc::new(parse_message_type(&schema_msg)
        .map_err(|e| Error::serialization(format!("解析Parquet模式失败: {}", e)))?);
    
    // 设置写入属性
    let props = Arc::new(WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_write_batch_size(1024)
        .build());
    
    // 创建文件
    let file = File::create(path.as_ref())
        .map_err(|e| Error::io_error(format!("创建Parquet文件失败: {}", e)))?;
    
    let mut writer = SerializedFileWriter::new(file, schema.clone(), props.clone())
        .map_err(|e| Error::serialization(format!("创建Parquet写入器失败: {}", e)))?;
    
    // 写入数据
    let num_rows = batch.records.len();
    let mut row_group_writer = writer.next_row_group()
        .map_err(|e| Error::serialization(format!("创建行组写入器失败: {}", e)))?;
    
    for (field_name, col_type) in fields.iter().zip(col_types.iter()) {
        if let Some(mut col_writer) = row_group_writer.next_column()
            .map_err(|e| Error::serialization(format!("创建列写入器失败: {}", e)))? {
            
            let mut def_levels: Vec<i16> = Vec::with_capacity(num_rows);
            let rep_levels: Vec<i16> = vec![0; num_rows];
            
            match (col_type, &mut col_writer) {
                (ColType::Utf8, ColumnWriter::ByteArrayColumnWriter(ref mut w)) => {
                    let mut values: Vec<ByteArray> = Vec::new();
                    for record in &batch.records {
                        if let Some(v) = record.get(field_name) {
                            def_levels.push(1);
                            let str_value = match v {
                                crate::data::DataValue::String(s) => s.clone(),
                                crate::data::DataValue::Integer(i) => i.to_string(),
                                crate::data::DataValue::Float(f) => f.to_string(),
                                crate::data::DataValue::Boolean(b) => b.to_string(),
                                crate::data::DataValue::Null => String::new(),
                                _ => format!("{:?}", v),
                            };
                            values.push(ByteArray::from(str_value));
                        } else {
                            def_levels.push(0);
                        }
                    }
                    w.write_batch(&values, Some(&def_levels), Some(&rep_levels))
                        .map_err(|e| Error::serialization(format!("写入字符串列失败: {}", e)))?;
                }
                (ColType::F64, ColumnWriter::DoubleColumnWriter(ref mut w)) => {
                    let mut values: Vec<f64> = Vec::new();
                    for record in &batch.records {
                        if let Some(v) = record.get(field_name) {
                            def_levels.push(1);
                            let float_value = match v {
                                crate::data::DataValue::Float(f) => *f,
                                crate::data::DataValue::Number(n) => *n,
                                crate::data::DataValue::Integer(i) => *i as f64,
                                _ => 0.0,
                            };
                            values.push(float_value);
                        } else {
                            def_levels.push(0);
                        }
                    }
                    w.write_batch(&values, Some(&def_levels), Some(&rep_levels))
                        .map_err(|e| Error::serialization(format!("写入浮点数列失败: {}", e)))?;
                }
                (ColType::I64, ColumnWriter::Int64ColumnWriter(ref mut w)) => {
                    let mut values: Vec<i64> = Vec::new();
                    for record in &batch.records {
                        if let Some(v) = record.get(field_name) {
                            def_levels.push(1);
                            let int_value = match v {
                                crate::data::DataValue::Integer(i) => *i,
                                crate::data::DataValue::Float(f) => *f as i64,
                                crate::data::DataValue::Number(n) => *n as i64,
                                _ => 0,
                            };
                            values.push(int_value);
                        } else {
                            def_levels.push(0);
                        }
                    }
                    w.write_batch(&values, Some(&def_levels), Some(&rep_levels))
                        .map_err(|e| Error::serialization(format!("写入整数列失败: {}", e)))?;
                }
                (ColType::Bool, ColumnWriter::BoolColumnWriter(ref mut w)) => {
                    let mut values: Vec<bool> = Vec::new();
                    for record in &batch.records {
                        if let Some(v) = record.get(field_name) {
                            def_levels.push(1);
                            let bool_value = match v {
                                crate::data::DataValue::Boolean(b) => *b,
                                _ => false,
                            };
                            values.push(bool_value);
                        } else {
                            def_levels.push(0);
                        }
                    }
                    w.write_batch(&values, Some(&def_levels), Some(&rep_levels))
                        .map_err(|e| Error::serialization(format!("写入布尔列失败: {}", e)))?;
                }
                _ => {
                    // 对于不匹配的类型，写入空值
                    for _ in 0..num_rows {
                        def_levels.push(0);
                    }
                }
            }
            
            row_group_writer.close_column(col_writer)
                .map_err(|e| Error::serialization(format!("关闭列写入器失败: {}", e)))?;
        }
    }
    
    writer.close_row_group(row_group_writer)
        .map_err(|e| Error::serialization(format!("关闭行组失败: {}", e)))?;
    
    writer.close()
        .map_err(|e| Error::serialization(format!("关闭Parquet文件失败: {}", e)))?;
    
    info!("成功保存 {} 条记录到Parquet文件: {}", num_rows, path.as_ref().display());
    Ok(())
}

/// 将数据批次保存为Parquet（未启用parquet特性时）
#[cfg(not(feature = "parquet"))]
fn save_batch_as_parquet<P: AsRef<Path>>(_batch: &DataBatch, _path: P) -> Result<()> {
    Err(Error::feature_not_enabled("parquet"))
}

/// 导出批处理文件的方法
/// 此方法会根据批次大小将数据分割成多个文件
pub fn export_batch_files<P: AsRef<Path>>(
    batch: &DataBatch,
    output_dir: P,
    format: DataFormat,
    prefix: &str,
    batch_size: usize
) -> Result<Vec<PathBuf>> {
    let dir_path = output_dir.as_ref();
    
    // 确保输出目录存在
    if !dir_path.exists() {
        fs::create_dir_all(dir_path)
            .map_err(|e| {
                error!("创建导出目录失败: {}", e);
                Error::io_error(format!("创建导出目录失败: {}", e))
            })?;
    }
    
    let total_records = batch.size;
    if total_records == 0 {
        error!("批次为空，无数据可导出");
        return Err(Error::invalid_input("批次为空，无数据可导出"));
    }
    
    let mut result = Vec::new();
    let batch_count = (total_records + batch_size - 1) / batch_size;
    
    debug!("将{}条记录分割为{}个批次，每批大小: {}", total_records, batch_count, batch_size);
    
    for i in 0..batch_count {
        let start = i * batch_size;
        let end = std::cmp::min(start + batch_size, total_records);
        
        // 创建子批次
        let sub_batch = match batch.slice(start, end) {
            Ok(sb) => sb,
            Err(e) => {
                error!("创建子批次失败: {}", e);
                return Err(e);
            }
        };
        
        // 构建输出文件名
        let file_name = format!("{}_{:04}.{}", prefix, i, format.file_extension());
        let output_path = dir_path.join(file_name);
        
        // 保存批次
        match save_batch_to_file(&sub_batch, &output_path, format.clone()) {
            Ok(path) => {
                result.push(path);
                debug!("已保存子批次 {}/{} 到 {:?}", i+1, batch_count, output_path);
            },
            Err(e) => {
                error!("保存子批次 {}/{} 失败: {}", i+1, batch_count, e);
                return Err(e);
            }
        }
    }
    
    debug!("成功导出 {} 个批次文件", result.len());
    Ok(result)
}

/// 将文件分割为多个子文件
pub fn split_file<P: AsRef<Path>>(
    input_file: P,
    output_dir: P,
    chunk_size: usize,
    format: DataFormat
) -> Result<Vec<PathBuf>> {
    let input_path = input_file.as_ref();
    let dir_path = output_dir.as_ref();
    
    if !input_path.exists() {
        error!("输入文件不存在: {:?}", input_path);
        return Err(Error::not_found(format!("输入文件不存在: {:?}", input_path)));
    }
    
    // 确保输出目录存在
    if !dir_path.exists() {
        fs::create_dir_all(dir_path)
            .map_err(|e| {
                error!("创建输出目录失败: {}", e);
                Error::io_error(format!("创建输出目录失败: {}", e))
            })?;
    }
    
    // 根据文件格式选择不同的分割策略
    match format {
        DataFormat::Csv { .. } => split_csv_file(input_path, dir_path, chunk_size, format),
        DataFormat::Json { .. } => split_json_file(input_path, dir_path, chunk_size, format),
        _ => {
            error!("不支持分割 {:?} 格式的文件", format);
            Err(Error::not_implemented(format!("不支持分割 {:?} 格式的文件", format)))
        }
    }
}

/// 分割CSV文件
fn split_csv_file<P: AsRef<Path>>(
    input_file: P,
    output_dir: P,
    chunk_size: usize,
    format: DataFormat
) -> Result<Vec<PathBuf>> {
    let input_path = input_file.as_ref();
    let dir_path = output_dir.as_ref();
    
    let file = File::open(input_path)
        .map_err(|e| {
            error!("打开CSV文件失败: {}", e);
            Error::io_error(format!("打开CSV文件失败: {}", e))
        })?;
    
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(if let DataFormat::Csv { delimiter, .. } = format {
            delimiter as u8
        } else {
            b','
        })
        .has_headers(if let DataFormat::Csv { has_header, .. } = format {
            has_header
        } else {
            true
        })
        .from_reader(file);
    
    let headers = reader.headers().cloned()
        .map_err(|e| {
            error!("读取CSV头部失败: {}", e);
            Error::io_error(format!("读取CSV头部失败: {}", e))
        })?;
    
    let mut result = Vec::new();
    let mut current_chunk = 0;
    let mut current_records = 0;
    let file_stem = input_path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("split");
    
    let mut current_writer = None;
    let mut current_output_path = None;
    
    for record in reader.records() {
        let record = record.map_err(|e| {
            error!("读取CSV记录失败: {}", e);
            Error::io_error(format!("读取CSV记录失败: {}", e))
        })?;
        
        // 如果需要创建新的写入器
        if current_writer.is_none() || current_records >= chunk_size {
            if let Some(writer) = current_writer.take() {
                // 关闭当前写入器
                drop(writer);
                
                if let Some(path) = current_output_path.take() {
                    result.push(path);
                }
            }
            
            // 创建新的输出文件
            let output_file_name = format!("{}_{:04}.csv", file_stem, current_chunk);
            let output_path = dir_path.join(output_file_name);
            current_output_path = Some(output_path.clone());
            
            // 创建新的写入器
            let writer = File::create(&output_path)
                .map_err(|e| {
                    error!("创建CSV输出文件失败: {}", e);
                    Error::io_error(format!("创建CSV输出文件失败: {}", e))
                })?;
            
            let mut csv_writer = csv::WriterBuilder::new()
                .delimiter(if let DataFormat::Csv { delimiter, .. } = format {
                    delimiter as u8
                } else {
                    b','
                })
                .from_writer(writer);
            
            // 写入标题
            csv_writer.write_record(&headers)
                .map_err(|e| {
                    error!("写入CSV标题失败: {}", e);
                    Error::io_error(format!("写入CSV标题失败: {}", e))
                })?;
            
            current_writer = Some(csv_writer);
            current_chunk += 1;
            current_records = 0;
        }
        
        // 写入记录
        if let Some(writer) = &mut current_writer {
            writer.write_record(&record)
                .map_err(|e| {
                    error!("写入CSV记录失败: {}", e);
                    Error::io_error(format!("写入CSV记录失败: {}", e))
                })?;
            
            current_records += 1;
        }
    }
    
    // 关闭最后一个写入器
    if let Some(writer) = current_writer.take() {
        drop(writer);
        
        if let Some(path) = current_output_path.take() {
            result.push(path);
        }
    }
    
    Ok(result)
}

/// 分割JSON文件
fn split_json_file<P: AsRef<Path>>(
    input_file: P,
    output_dir: P,
    chunk_size: usize,
    format: DataFormat
) -> Result<Vec<PathBuf>> {
    let input_path = input_file.as_ref();
    let dir_path = output_dir.as_ref();
    
    let file = File::open(input_path)
        .map_err(|e| {
            error!("打开JSON文件失败: {}", e);
            Error::io_error(format!("打开JSON文件失败: {}", e))
        })?;
    
    let reader = BufReader::new(file);
    
    // 确定JSON格式类型
    let is_lines = if let DataFormat::Json { is_lines, .. } = format {
        is_lines
    } else {
        false
    };
    
    let is_array = if let DataFormat::Json { is_array, .. } = format {
        is_array
    } else {
        true
    };
    
    let mut result = Vec::new();
    
    if is_lines {
        // 处理JSON Lines格式
        let file_stem = input_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("split");
        
        let mut current_chunk = 0;
        let mut records = Vec::new();
        
        for line in BufReader::new(File::open(input_path)
            .map_err(|e| {
                error!("打开JSON文件失败: {}", e);
                Error::io_error(format!("打开JSON文件失败: {}", e))
            })?)
            .lines() {
            
            let line = line.map_err(|e| {
                error!("读取JSON行失败: {}", e);
                Error::io_error(format!("读取JSON行失败: {}", e))
            })?;
            
            if !line.trim().is_empty() {
                // 解析JSON确保格式正确
                match serde_json::from_str::<serde_json::Value>(&line) {
                    Ok(value) => records.push(value),
                    Err(e) => {
                        error!("解析JSON行失败: {}", e);
                        return Err(Error::invalid_input(format!("解析JSON行失败: {}", e)));
                    }
                }
            }
            
            // 如果达到块大小，写入文件
            if records.len() >= chunk_size {
                let output_path = write_json_chunk(
                    &records, 
                    dir_path, 
                    file_stem, 
                    current_chunk, 
                    is_lines,
                    false // 以数组形式输出JSONL块
                )?;
                
                result.push(output_path);
                records.clear();
                current_chunk += 1;
            }
        }
        
        // 写入最后一批数据（如果有）
        if !records.is_empty() {
            let output_path = write_json_chunk(
                &records, 
                dir_path, 
                file_stem, 
                current_chunk, 
                is_lines,
                false // 以数组形式输出JSONL块
            )?;
            
            result.push(output_path);
        }
    } else if is_array {
        // 处理JSON数组格式
        let json_value: serde_json::Value = serde_json::from_reader(reader)
            .map_err(|e| {
                error!("解析JSON文件失败: {}", e);
                Error::invalid_input(format!("解析JSON文件失败: {}", e))
            })?;
        
        // 确保是数组
        if !json_value.is_array() {
            error!("JSON文件不是数组格式");
            return Err(Error::invalid_input("JSON文件必须是数组格式"));
        }
        
        let array = json_value.as_array().unwrap();
        let total_items = array.len();
        let chunk_count = (total_items + chunk_size - 1) / chunk_size;
        
        let file_stem = input_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("split");
        
        for i in 0..chunk_count {
            let start = i * chunk_size;
            let end = std::cmp::min(start + chunk_size, total_items);
            
            // 创建子数组
            let chunk: Vec<serde_json::Value> = array[start..end].to_vec();
            
            // 写入文件
            let output_path = write_json_chunk(
                &chunk, 
                dir_path, 
                file_stem, 
                i, 
                false, // 不是JSONL
                true   // 是数组
            )?;
            
            result.push(output_path);
        }
    } else {
        // 处理单个JSON对象
        error!("不支持分割非数组非行JSON格式");
        return Err(Error::not_implemented("不支持分割非数组非行JSON格式"));
    }
    
    Ok(result)
}

/// 写入JSON数据块到文件
fn write_json_chunk<P: AsRef<Path>>(
    data: &[serde_json::Value],
    output_dir: P,
    file_stem: &str,
    chunk_index: usize,
    is_lines: bool,
    is_array: bool
) -> Result<PathBuf> {
    // 创建输出文件路径
    let output_file_name = format!("{}_{:04}.json", file_stem, chunk_index);
    let output_path = output_dir.as_ref().join(output_file_name);
    
    let file = File::create(&output_path)
        .map_err(|e| {
            error!("创建JSON输出文件失败: {}", e);
            Error::io_error(format!("创建JSON输出文件失败: {}", e))
        })?;
    
    let mut writer = BufWriter::new(file);
    
    if is_lines {
        // 写入JSON Lines格式
        for value in data {
            let line = serde_json::to_string(value)
                .map_err(|e| {
                    error!("序列化JSON对象失败: {}", e);
                    Error::serialization(format!("序列化JSON对象失败: {}", e))
                })?;
            
            writeln!(writer, "{}", line)
                .map_err(|e| {
                    error!("写入JSON行失败: {}", e);
                    Error::io_error(format!("写入JSON行失败: {}", e))
                })?;
        }
    } else if is_array {
        // 写入JSON数组格式
        serde_json::to_writer_pretty(writer, data)
            .map_err(|e| {
                error!("写入JSON数组失败: {}", e);
                Error::io_error(format!("写入JSON数组失败: {}", e))
            })?;
    } else {
        // 单个对象，不应该到达这里
        error!("不支持写入非数组非行JSON格式");
        return Err(Error::not_implemented("不支持写入非数组非行JSON格式"));
    }
    
    // 刷新缓冲区
    writer.flush()
        .map_err(|e| {
            error!("刷新JSON写入器失败: {}", e);
            Error::io_error(format!("刷新JSON写入器失败: {}", e))
        })?;
    
    Ok(output_path)
} 