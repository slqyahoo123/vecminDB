//! 数据输入输出功能
//! 提供从各种文件格式加载和保存数据的功能

use std::path::Path;
use std::fs::{File, create_dir_all};
use std::io::{BufReader, BufWriter};

use log::{debug, info, warn, error, trace};
use serde_json::{Value as JsonValue};
use csv::{ReaderBuilder, WriterBuilder};
#[cfg(feature = "parquet")]
use parquet::file::reader::{FileReader, SerializedFileReader};
#[cfg(feature = "parquet")]
use parquet::file::writer::{SerializedFileWriter};
#[cfg(feature = "parquet")]
use parquet::schema::parser::parse_message_type;
#[cfg(feature = "parquet")]
use parquet::schema::types::Type as SchemaType;
#[cfg(feature = "parquet")]
use parquet::record::{Row, RowAccessor};
#[cfg(feature = "parquet")]
use parquet::basic::{Compression};
#[cfg(feature = "parquet")]
use parquet::file::properties::WriterProperties;
#[cfg(feature = "parquet")]
use parquet::column::writer::ColumnWriter;
#[cfg(feature = "parquet")]
use parquet::file::writer::SerializedColumnWriter;
#[cfg(feature = "parquet")]
use parquet::data_type::ByteArray;

use crate::error::{Result, Error};
use crate::data::value::DataValue;

/// 从文件加载数据
///
/// # 参数
///
/// * `path` - 文件路径，支持CSV, JSON, Parquet格式
///
/// # 返回
///
/// 返回数据值的向量，每个元素代表一条记录
pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Vec<DataValue>> {
    let path = path.as_ref();
    
    // 检查文件是否存在
    if !path.exists() {
        return Err(Error::FileNotFound(path.to_string_lossy().to_string()));
    }
    
    info!("加载数据文件: {:?}", path);
    
    // 根据文件扩展名选择合适的解析器
    match path.extension().and_then(|e| e.to_str()) {
        Some("json") => {
            debug!("检测到JSON文件格式");
            load_json(path)
        },
        Some("csv") => {
            debug!("检测到CSV文件格式");
            load_csv(path)
        },
        #[cfg(feature = "parquet")]
        Some("parquet") => {
            debug!("检测到Parquet文件格式");
            load_parquet(path)
        },
        #[cfg(not(feature = "parquet"))]
        Some("parquet") => {
            error!("Parquet功能需要启用 'parquet' feature");
            Err(Error::feature_not_enabled("parquet"))
        },
        Some(ext) => {
            error!("不支持的文件格式: {}", ext);
            Err(Error::unsupported_format(format!("不支持的文件格式: {:?}", path)))
        },
        None => {
            error!("无法确定文件格式: {:?}", path);
            Err(Error::unsupported_format(format!("不支持的文件格式: {:?}", path)))
        }
    }
}

/// 保存数据到文件
///
/// # 参数
///
/// * `data` - 数据值的切片，每个元素代表一条记录
/// * `path` - 目标文件路径，支持CSV, JSON, Parquet格式
pub fn save_to_file<P: AsRef<Path>>(data: &[DataValue], path: P) -> Result<()> {
    let path = path.as_ref();
    
    // 如果目录不存在，尝试创建
    if let Some(parent) = path.parent() {
        if !parent.exists() {
            debug!("创建目录: {:?}", parent);
            create_dir_all(parent).map_err(|e| {
                error!("创建目录失败: {:?}, 错误: {}", parent, e);
                Error::IoError(format!("无法创建目录 {:?}: {}", parent, e))
            })?;
        }
    }
    
    info!("保存数据文件: {:?}, 记录数: {}", path, data.len());
    
    // 根据文件扩展名选择合适的保存格式
    match path.extension().and_then(|e| e.to_str()) {
        Some("json") => {
            debug!("使用JSON格式保存");
            save_json(data, path)
        },
        Some("csv") => {
            debug!("使用CSV格式保存");
            save_csv(data, path)
        },
        #[cfg(feature = "parquet")]
        Some("parquet") => {
            debug!("使用Parquet格式保存");
            save_parquet(data, path)
        },
        #[cfg(not(feature = "parquet"))]
        Some("parquet") => {
            error!("Parquet功能需要启用 'parquet' feature");
            Err(Error::feature_not_enabled("parquet"))
        },
        Some(ext) => {
            error!("不支持的文件格式: {}", ext);
            Err(Error::unsupported_format(format!("不支持的文件格式: {:?}", path)))
        },
        None => {
            error!("无法确定文件格式: {:?}", path);
            Err(Error::unsupported_format(format!("不支持的文件格式: {:?}", path)))
        }
    }
}

/// 加载JSON文件
fn load_json<P: AsRef<Path>>(path: P) -> Result<Vec<DataValue>> {
    let file = File::open(path).map_err(|e| {
        error!("打开JSON文件失败: {}", e);
        Error::IoError(format!("打开文件失败: {}", e))
    })?;
    
    let reader = BufReader::new(file);
    trace!("JSON文件已打开并创建缓冲读取器");
    
    let values: Vec<JsonValue> = serde_json::from_reader(reader).map_err(|e| {
        error!("解析JSON文件失败: {}", e);
        Error::invalid_input(format!("解析JSON失败: {}", e))
    })?;
    
    debug!("成功解析JSON文件，记录数: {}", values.len());
    
    // 将JSON值转换为DataValue
    let result: Vec<DataValue> = values.into_iter()
        .map(DataValue::from_json)
        .collect();
    
    info!("从JSON加载了 {} 条记录", result.len());
    Ok(result)
}

/// 保存为JSON文件
fn save_json<P: AsRef<Path>>(data: &[DataValue], path: P) -> Result<()> {
    let file = File::create(path).map_err(|e| {
        error!("创建JSON文件失败: {}", e);
        Error::IoError(format!("创建文件失败: {}", e))
    })?;
    
    let writer = BufWriter::new(file);
    trace!("JSON文件已创建并创建缓冲写入器");
    
    // 将DataValue转换为JSON值
    let json_values: Vec<JsonValue> = data.iter()
        .map(|v| v.to_json())
        .collect();
    
    // 写入JSON数据
    serde_json::to_writer_pretty(writer, &json_values).map_err(|e| {
        error!("写入JSON文件失败: {}", e);
        Error::IoError(format!("写入JSON失败: {}", e))
    })?;
    
    info!("成功保存 {} 条记录到JSON文件", data.len());
    Ok(())
}

/// 加载CSV文件
fn load_csv<P: AsRef<Path>>(path: P) -> Result<Vec<DataValue>> {
    let file = File::open(path).map_err(|e| {
        error!("打开CSV文件失败: {}", e);
        Error::IoError(format!("打开文件失败: {}", e))
    })?;
    
    let reader = BufReader::new(file);
    trace!("CSV文件已打开并创建缓冲读取器");
    
    // 创建CSV读取器并配置
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .trim(csv::Trim::All)
        .flexible(true)
        .from_reader(reader);
    
    // 读取标题行
    let headers = reader.headers().map_err(|e| {
        error!("读取CSV标题失败: {}", e);
        Error::invalid_input(format!("读取CSV标题失败: {}", e))
    })?.clone();
    
    debug!("CSV文件标题: {:?}", headers);
    
    let mut records = Vec::new();
    
    // 读取数据行
    for (row_idx, result) in reader.records().enumerate() {
        let record = result.map_err(|e| {
            error!("读取CSV行 {} 失败: {}", row_idx + 1, e);
            Error::invalid_input(format!("读取CSV行 {} 失败: {}", row_idx + 1, e))
        })?;
        
        // 创建记录数据
        let mut record_data = std::collections::HashMap::new();
        
        for (i, field) in record.iter().enumerate() {
            let field_name = if i < headers.len() {
                headers[i].to_string()
            } else {
                format!("field_{}", i)
            };
            
            // 尝试将值解析为适当的类型
            let value = parse_csv_value(field);
            record_data.insert(field_name, value);
        }
        
        records.push(DataValue::Object(record_data));
    }
    
    info!("从CSV加载了 {} 条记录", records.len());
    Ok(records)
}

/// 解析CSV单元格值为适当的数据类型
fn parse_csv_value(value: &str) -> DataValue {
    // 尝试解析为数字
    if let Ok(int_val) = value.parse::<i64>() {
        return DataValue::Integer(int_val);
    }
    
    if let Ok(float_val) = value.parse::<f64>() {
        return DataValue::Number(float_val);
    }
    
    // 检查布尔值
    match value.to_lowercase().as_str() {
        "true" | "yes" | "1" => return DataValue::Boolean(true),
        "false" | "no" | "0" => return DataValue::Boolean(false),
        "null" | "na" | "n/a" | "" => return DataValue::Null,
        _ => {}
    }
    
    // 尝试解析为日期时间
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(value) {
        return DataValue::DateTime(dt.with_timezone(&chrono::Utc).to_rfc3339());
    }
    
    // 默认为字符串
    DataValue::String(value.to_string())
}

/// 保存为CSV文件
fn save_csv<P: AsRef<Path>>(data: &[DataValue], path: P) -> Result<()> {
    if data.is_empty() {
        warn!("No data to write to CSV file");
        // 创建空文件
        File::create(path).map_err(|e| {
            error!("创建空CSV文件失败: {}", e);
            Error::IoError(format!("创建空文件失败: {}", e))
        })?;
        return Ok(());
    }
    
    let file = File::create(path).map_err(|e| {
        error!("创建CSV文件失败: {}", e);
        Error::IoError(format!("创建文件失败: {}", e))
    })?;
    
    let mut writer = WriterBuilder::new()
        .has_headers(true)
        .from_writer(BufWriter::new(file));
    
    trace!("CSV文件已创建并配置写入器");
    
    // 提取所有记录中的所有键
    let mut all_fields = std::collections::HashSet::new();
    
    for value in data {
        if let DataValue::Object(map) = value {
            for key in map.keys() {
                all_fields.insert(key.clone());
            }
        }
    }
    
    // 将字段转换为有序列表
    let headers: Vec<String> = all_fields.into_iter().collect();
    debug!("CSV标题行: {:?}", headers);
    
    // 写入标题行
    writer.write_record(&headers).map_err(|e| {
        error!("写入CSV标题失败: {}", e);
        Error::IoError(format!("写入CSV标题失败: {}", e))
    })?;
    
    // 写入数据行
    for (idx, value) in data.iter().enumerate() {
        if let DataValue::Object(map) = value {
            let row: Vec<String> = headers.iter()
                .map(|key| {
                    if let Some(field_value) = map.get(key) {
                        format_csv_value(field_value)
                    } else {
                        String::new() // 空字段
                    }
                })
                .collect();
            
            writer.write_record(&row).map_err(|e| {
                error!("写入CSV行 {} 失败: {}", idx + 1, e);
                Error::IoError(format!("写入CSV行 {} 失败: {}", idx + 1, e))
            })?;
        } else {
            warn!("Skipping non-object record (index {}): {:?}", idx, value);
        }
    }
    
    // 确保所有数据都写入磁盘
    writer.flush().map_err(|e| {
        error!("刷新CSV写入器失败: {}", e);
        Error::IoError(format!("刷新写入器失败: {}", e))
    })?;
    
    info!("成功保存 {} 条记录到CSV文件", data.len());
    Ok(())
}

/// 将DataValue格式化为CSV单元格值
fn format_csv_value(value: &DataValue) -> String {
    match value {
        DataValue::Null => String::new(),
        DataValue::Boolean(b) => b.to_string(),
        DataValue::Integer(i) => i.to_string(),
        DataValue::Number(n) => n.to_string(),
        DataValue::String(s) => s.clone(),
        DataValue::DateTime(dt) => {
            // DateTime 存储为字符串，尝试解析为 DateTime 对象
            if let Ok(parsed_dt) = chrono::DateTime::parse_from_rfc3339(&dt) {
                parsed_dt.to_rfc3339()
            } else {
                dt.clone()
            }
        },
        DataValue::Array(arr) => {
            let items: Vec<String> = arr.iter()
                .map(format_csv_value)
                .collect();
            format!("[{}]", items.join(","))
        },
        DataValue::Object(obj) => {
            let items: Vec<String> = obj.iter()
                .map(|(k, v)| format!("\"{}\":\"{}\"", k, format_csv_value(v)))
                .collect();
            format!("{{{}}}", items.join(","))
        },
        DataValue::Binary(bytes) => format!("<binary:{} bytes>", bytes.len()),
        DataValue::Tensor(tensor) => format!("<tensor:{:?}>", tensor.shape),
        _ => format!("<unsupported:{:?}>", value),
    }
}

/// 加载Parquet文件
#[cfg(feature = "parquet")]
fn load_parquet<P: AsRef<Path>>(path: P) -> Result<Vec<DataValue>> {
    let path = path.as_ref();
    
    // 打开Parquet文件
    let file = File::open(path).map_err(|e| {
        error!("打开Parquet文件失败: {}", e);
        Error::IoError(format!("打开文件失败: {}", e))
    })?;
    
    trace!("Parquet文件已打开");
    
    // 创建Parquet读取器
    let reader = SerializedFileReader::new(file).map_err(|e| {
        error!("创建Parquet读取器失败: {}", e);
        Error::invalid_input(format!("创建Parquet读取器失败: {}", e))
    })?;
    
    // 获取文件元数据
    let metadata = reader.metadata();
    let file_metadata = metadata.file_metadata();
    let schema = file_metadata.schema();
    
    debug!("Parquet文件模式: {:?}", schema);
    debug!("Parquet行组数: {}", metadata.num_row_groups());
    
    let mut records = Vec::new();
    
    // 处理每个行组
    for i in 0..metadata.num_row_groups() {
        let row_group = reader.get_row_group(i).map_err(|e| {
            error!("读取Parquet行组 {} 失败: {}", i, e);
            Error::invalid_input(format!("读取Parquet行组失败: {}", e))
        })?;
        
        let row_group_metadata = row_group.metadata();
        debug!("行组 {} 行数: {}", i, row_group_metadata.num_rows());
        
        // 获取并处理行
        let mut row_iter = row_group.get_row_iter(None).map_err(|e| {
            error!("获取Parquet行迭代器失败: {}", e);
            Error::invalid_input(format!("获取行迭代器失败: {}", e))
        })?;
        
        while let Some(row_result) = row_iter.next() {
            let row = row_result.map_err(|e| {
                error!("读取Parquet行失败: {}", e);
                Error::invalid_input(format!("读取Parquet行失败: {}", e))
            })?;
            let record = convert_parquet_row_to_data_value(&row, schema)?;
            records.push(record);
        }
    }
    
    info!("从Parquet加载了 {} 条记录", records.len());
    Ok(records)
}

/// 将Parquet行转换为DataValue
#[cfg(feature = "parquet")]
fn convert_parquet_row_to_data_value(row: &Row, schema: &SchemaType) -> Result<DataValue> {
    let mut record_data = std::collections::HashMap::new();
    
    for (i, field) in schema.get_fields().iter().enumerate() {
        let field_name = field.name();
        let value = convert_parquet_value(row, i)?;
        record_data.insert(field_name.to_string(), value);
    }
    
    Ok(DataValue::Object(record_data))
}

/// 将Parquet值转换为DataValue
#[cfg(feature = "parquet")]
fn convert_parquet_value(row: &Row, idx: usize) -> Result<DataValue> {
    // 尝试获取布尔值
    if let Ok(value) = row.get_bool(idx) {
        return Ok(DataValue::Boolean(value));
    }
    
    // 尝试获取整数
    if let Ok(value) = row.get_int(idx) {
        return Ok(DataValue::Integer(value as i64));
    }
    
    if let Ok(value) = row.get_long(idx) {
        return Ok(DataValue::Integer(value));
    }
    
    // 尝试获取浮点数
    if let Ok(value) = row.get_float(idx) {
        return Ok(DataValue::Number(value as f64));
    }
    
    if let Ok(value) = row.get_double(idx) {
        return Ok(DataValue::Number(value));
    }
    
    // 尝试获取字符串
    if let Ok(value) = row.get_string(idx) {
        return Ok(DataValue::String(value.clone()));
    }
    
    // 尝试获取字节数组
    if let Ok(value) = row.get_bytes(idx) {
        return Ok(DataValue::Binary(value.data().to_vec()));
    }
    
    // 更复杂的类型处理可以根据需要扩展
    
    // 默认返回Null
    Ok(DataValue::Null)
}

/// 保存为Parquet文件
#[cfg(feature = "parquet")]
fn save_parquet<P: AsRef<Path>>(data: &[DataValue], path: P) -> Result<()> {
    if data.is_empty() {
        warn!("No data to write to Parquet file");
        return Ok(());
    }
    
    // 推断模式
    let schema = infer_parquet_schema(data)?;
    debug!("推断的Parquet模式: {}", schema);
    
    // 解析模式字符串
    let schema_type = parse_message_type(&schema).map_err(|e| {
        error!("解析Parquet模式失败: {}", e);
        Error::invalid_input(format!("解析Parquet模式失败: {}", e))
    })?;
    
    // 设置写入属性
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    
    // 创建Parquet文件和写入器
    let file = File::create(path).map_err(|e| {
        error!("创建Parquet文件失败: {}", e);
        Error::IoError(format!("创建文件失败: {}", e))
    })?;
    
    let mut writer = SerializedFileWriter::new(file, Arc::new(schema_type), Arc::new(props)).map_err(|e| {
        error!("创建Parquet写入器失败: {}", e);
        Error::IoError(format!("创建Parquet写入器失败: {}", e))
    })?;
    
    // 创建行组写入器
    let mut row_group_writer = writer.next_row_group().map_err(|e| {
        error!("创建Parquet行组写入器失败: {}", e);
        Error::IoError(format!("创建行组写入器失败: {}", e))
    })?;
    
    // 写入数据
    let result = (|| -> Result<()> {
        // 获取模式字段
        let schema_fields = schema_type.get_fields();
        let num_rows = data.len();
        
        // 为每个字段写入数据
        for (field_idx, field) in schema_fields.iter().enumerate() {
            let field_name = field.name();
            let field_type = field.get_physical_type();
            
            // 获取列写入器
            if let Some(mut col_writer) = row_group_writer.next_column()
                .map_err(|e| {
                    error!("创建列写入器失败: {}", e);
                    Error::IoError(format!("创建列写入器失败: {}", e))
                })? {
            
                // 准备定义级别和重复级别
                let mut def_levels: Vec<i16> = Vec::with_capacity(num_rows);
                let rep_levels: Vec<i16> = vec![0; num_rows];
                
                // 根据字段类型写入数据
                // SerializedColumnWriter can be dereferenced to ColumnWriter using std::ops::DerefMut
                // We need to use a workaround since direct matching doesn't work
                // Use unsafe to access ColumnWriter from SerializedColumnWriter
                match field_type {
                    parquet::basic::Type::BOOLEAN => {
                        let mut values: Vec<bool> = Vec::new();
                        for record in data {
                            if let DataValue::Object(map) = record {
                                if let Some(val) = map.get(field_name) {
                                    def_levels.push(1);
                                    let bool_value = match val {
                                        DataValue::Boolean(b) => *b,
                                        _ => false,
                                    };
                                    values.push(bool_value);
                                } else {
                                    def_levels.push(0);
                                }
                            } else {
                                def_levels.push(0);
                            }
                        }
                        // Use unsafe to access ColumnWriter from SerializedColumnWriter
                        unsafe {
                            if let ColumnWriter::BoolColumnWriter(ref mut w) = &mut *(&mut col_writer as *mut SerializedColumnWriter as *mut ColumnWriter) {
                            w.write_batch(&values, Some(&def_levels), Some(&rep_levels))
                                .map_err(|e| {
                                    error!("写入布尔列失败: {}", e);
                                    Error::IoError(format!("写入布尔列失败: {}", e))
                                })?;
                            }
                        }
                    },
                    parquet::basic::Type::INT64 => {
                        let mut values: Vec<i64> = Vec::new();
                        for record in data {
                            if let DataValue::Object(map) = record {
                                if let Some(val) = map.get(field_name) {
                                    def_levels.push(1);
                                    let int_value = match val {
                                        DataValue::Integer(i) => *i,
                                        DataValue::Number(n) => *n as i64,
                                        DataValue::Float(f) => *f as i64,
                                        _ => 0,
                                    };
                                    values.push(int_value);
                                } else {
                                    def_levels.push(0);
                                }
                            } else {
                                def_levels.push(0);
                            }
                        }
                        unsafe {
                            if let ColumnWriter::Int64ColumnWriter(ref mut w) = &mut *(&mut col_writer as *mut SerializedColumnWriter as *mut ColumnWriter) {
                            w.write_batch(&values, Some(&def_levels), Some(&rep_levels))
                                .map_err(|e| {
                                    error!("写入整数列失败: {}", e);
                                    Error::IoError(format!("写入整数列失败: {}", e))
                                })?;
                            }
                        }
                    },
                    parquet::basic::Type::DOUBLE => {
                        let mut values: Vec<f64> = Vec::new();
                        for record in data {
                            if let DataValue::Object(map) = record {
                                if let Some(val) = map.get(field_name) {
                                    def_levels.push(1);
                                    let float_value = match val {
                                        DataValue::Float(f) => *f,
                                        DataValue::Number(n) => *n,
                                        DataValue::Integer(i) => *i as f64,
                                        _ => 0.0,
                                    };
                                    values.push(float_value);
                                } else {
                                    def_levels.push(0);
                                }
                            } else {
                                def_levels.push(0);
                            }
                        }
                        unsafe {
                            if let ColumnWriter::DoubleColumnWriter(ref mut w) = &mut *(&mut col_writer as *mut SerializedColumnWriter as *mut ColumnWriter) {
                            w.write_batch(&values, Some(&def_levels), Some(&rep_levels))
                                .map_err(|e| {
                                    error!("写入浮点数列失败: {}", e);
                                    Error::IoError(format!("写入浮点数列失败: {}", e))
                                })?;
                            }
                        }
                    },
                    parquet::basic::Type::BYTE_ARRAY => {
                        let mut values: Vec<ByteArray> = Vec::new();
                        for record in data {
                            if let DataValue::Object(map) = record {
                                if let Some(val) = map.get(field_name) {
                                    def_levels.push(1);
                                    let str_value = match val {
                                        DataValue::String(s) => s.clone(),
                                        DataValue::Integer(i) => i.to_string(),
                                        DataValue::Float(f) => f.to_string(),
                                        DataValue::Number(n) => n.to_string(),
                                        DataValue::Boolean(b) => b.to_string(),
                                        DataValue::DateTime(dt) => dt.to_string(),
                                        DataValue::Binary(bin) => {
                                            // 对于二进制数据，转换为Base64字符串
                                            use base64::{Engine as _, engine::general_purpose};
                                            general_purpose::STANDARD.encode(bin)
                                        },
                                        DataValue::Null => String::new(),
                                        _ => format!("{:?}", val),
                                    };
                                    values.push(ByteArray::from(str_value.as_str()));
                                } else {
                                    def_levels.push(0);
                                }
                            } else {
                                def_levels.push(0);
                            }
                        }
                        unsafe {
                            if let ColumnWriter::ByteArrayColumnWriter(ref mut w) = &mut *(&mut col_writer as *mut SerializedColumnWriter as *mut ColumnWriter) {
                            w.write_batch(&values, Some(&def_levels), Some(&rep_levels))
                                .map_err(|e| {
                                    error!("写入字符串列失败: {}", e);
                                    Error::IoError(format!("写入字符串列失败: {}", e))
                                })?;
                            }
                        }
                    },
                    _ => {
                        // 对于不支持的类型，写入空值
                        for _ in 0..num_rows {
                            def_levels.push(0);
                        }
                    }
                }
                
                // 关闭列写入器 - SerializedColumnWriter is automatically closed when dropped
            }
        }
        
        // 完成行组写入 - SerializedRowGroupWriter is automatically closed when dropped
        drop(row_group_writer);
        
        // 完成文件写入
        writer.close().map_err(|e| {
            error!("关闭Parquet行组写入器失败: {}", e);
            Error::IoError(format!("关闭行组写入器失败: {}", e))
        })?;
        
        
        Ok(())
    })();
    
    if result.is_ok() {
        info!("成功保存 {} 条记录到Parquet文件", data.len());
    }
    
    result
}

/// 根据数据推断Parquet模式
#[cfg(feature = "parquet")]
fn infer_parquet_schema(data: &[DataValue]) -> Result<String> {
    if data.is_empty() {
        return Err(Error::invalid_input("无法从空数据推断模式"));
    }
    
    // 提取字段和类型
    let mut fields = std::collections::HashMap::new();
    
    for value in data {
        if let DataValue::Object(map) = value {
            for (key, val) in map {
                let field_type = match val {
                    DataValue::Null => continue, // 跳过空值
                    DataValue::Boolean(_) => "BOOLEAN",
                    DataValue::Integer(_) => "INT64",
                    DataValue::Number(_) => "DOUBLE",
                    DataValue::String(_) => "UTF8",
                    DataValue::DateTime(_) => "UTF8",
                    DataValue::Binary(_) => "BYTE_ARRAY",
                    _ => "UTF8", // 默认为字符串
                };
                
                fields.insert(key.clone(), field_type);
            }
        }
    }
    
    // 构建模式字符串
    let mut schema = String::from("message schema {\n");
    
    for (name, field_type) in fields {
        schema.push_str(&format!("  OPTIONAL {} {};\n", field_type, name));
    }
    
    schema.push_str("}");
    
    Ok(schema)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_load_save_json() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.json");
        
        // 准备测试数据
        let mut record = std::collections::HashMap::new();
        record.insert("name".to_string(), DataValue::String("测试".to_string()));
        record.insert("age".to_string(), DataValue::Integer(30));
        record.insert("active".to_string(), DataValue::Boolean(true));
        
        let data = vec![DataValue::Object(record)];
        
        // 保存数据
        save_to_file(&data, &file_path).unwrap();
        
        // 加载数据
        let loaded_data = load_from_file(&file_path).unwrap();
        
        // 验证数据
        assert_eq!(data.len(), loaded_data.len());
        
        if let DataValue::Object(loaded_record) = &loaded_data[0] {
            assert_eq!(loaded_record.get("name").unwrap(), &DataValue::String("测试".to_string()));
            assert_eq!(loaded_record.get("age").unwrap(), &DataValue::Integer(30));
            assert_eq!(loaded_record.get("active").unwrap(), &DataValue::Boolean(true));
        } else {
            panic!("加载的数据不是对象类型");
        }
    }
    
    #[test]
    fn test_load_save_csv() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.csv");
        
        // 准备测试数据
        let mut record1 = std::collections::HashMap::new();
        record1.insert("name".to_string(), DataValue::String("张三".to_string()));
        record1.insert("age".to_string(), DataValue::Integer(25));
        
        let mut record2 = std::collections::HashMap::new();
        record2.insert("name".to_string(), DataValue::String("李四".to_string()));
        record2.insert("age".to_string(), DataValue::Integer(30));
        
        let data = vec![
            DataValue::Object(record1),
            DataValue::Object(record2)
        ];
        
        // 保存数据
        save_to_file(&data, &file_path).unwrap();
        
        // 加载数据
        let loaded_data = load_from_file(&file_path).unwrap();
        
        // 验证数据
        assert_eq!(data.len(), loaded_data.len());
        
        // 由于CSV加载后的顺序可能变化，这里只验证存在性
        let names = vec!["张三", "李四"];
        let ages = vec![25, 30];
        
        for record in loaded_data {
            if let DataValue::Object(map) = record {
                if let Some(DataValue::String(name)) = map.get("name") {
                    assert!(names.contains(&name.as_str()));
                    
                    if let Some(DataValue::Integer(age)) = map.get("age") {
                        if name == "张三" {
                            assert_eq!(*age, 25);
                        } else if name == "李四" {
                            assert_eq!(*age, 30);
                        }
                    } else {
                        panic!("age字段不是整数类型");
                    }
                } else {
                    panic!("name字段不是字符串类型");
                }
            } else {
                panic!("加载的数据不是对象类型");
            }
        }
    }
    
    // 更多测试可以根据需要添加
} 