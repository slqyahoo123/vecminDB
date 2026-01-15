// 通用数据加载器模块 - 提供基础数据加载和管理功能

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::fs;
// drop explicit Read import; concrete readers use fully-qualified paths locally

use async_trait::async_trait;
use log::{debug, error, info, trace};
use serde_json::Value;
use chrono::{Local};

// 外部依赖（条件编译以支持可选特性）
// 未来候选格式与数据源开关："http"(HTTP/WS/Kafka流)、"parquet"、"avro"、"postgres"、"mysql"、"sqlite"、"redis"、"mongodb"、"cassandra"、"elasticsearch"、"compression"
// 注意：以下特性未在 Cargo.toml 中定义，已注释
// #[cfg(feature = "http")]
// use reqwest;
// tokio/csv 在 Cargo.toml 默认已启用或为常规依赖，不使用 feature 门控
// tokio is referenced via fully-qualified paths; no top-level import needed
use ::csv;
// #[cfg(feature = "postgres")]
// use tokio_postgres;
// #[cfg(feature = "mysql")]
// use mysql_async;
// #[cfg(feature = "sqlite")]
// use rusqlite;
#[cfg(feature = "redis")]
use redis;
// #[cfg(feature = "mongodb")]
// use mongodb;
// #[cfg(feature = "cassandra")]
// use scylla::{Session as ScyllaSession, SessionBuilder, IntoTypedRows, QueryResult as ScyllaQueryResult};
// #[cfg(feature = "elasticsearch")]
// use elasticsearch;

use crate::data::{DataBatch, DataConfig, DataSchema, DataValue};
use crate::error::{Error, Result};
use crate::data::connector::DatabaseManager;
use crate::data::loader::{DataLoader, DataSource, DataFormat};
use crate::data::schema::schema::{FieldDefinition, FieldType, FieldConstraints};

// 查询结果结构体
#[derive(Debug)]
struct QueryResult {
    records: Vec<HashMap<String, DataValue>>,
    execution_time: Duration,
}

// 通用数据加载器
pub struct CommonDataLoader {
    config: DataConfig,
    db_manager: Option<Arc<DatabaseManager>>,
    cache: HashMap<String, DataBatch>,
}

impl CommonDataLoader {
    pub fn new(config: DataConfig) -> Result<Self> {
        Ok(Self {
            config,
            db_manager: None,
            cache: HashMap::new(),
        })
    }

    pub fn with_database_manager(config: DataConfig, db_manager: Arc<DatabaseManager>) -> Result<Self> {
        Ok(Self {
            config,
            db_manager: Some(db_manager),
            cache: HashMap::new(),
        })
    }

    pub fn get_database_manager(&self) -> Option<Arc<DatabaseManager>> {
        self.db_manager.clone()
    }

    pub fn set_database_manager(&mut self, db_manager: Arc<DatabaseManager>) {
        self.db_manager = Some(db_manager);
    }

    // 从文件加载数据
    async fn load_from_file(&self, path: &str, format: &crate::data::loader::types::DataFormat) -> Result<DataBatch> {
        debug!("从文件加载数据: {}", path);
        
        let file_path = Path::new(path);
        if !file_path.exists() {
            // 文件不存在，使用通用 not_found 错误，更符合 Error 语义
            return Err(crate::error::Error::not_found(format!("文件不存在: {}", path)));
        }
        
        // 根据文件格式选择加载方式
        match format {
            crate::data::loader::types::DataFormat::Csv { .. } => self.load_csv(path).await,
            crate::data::loader::types::DataFormat::Json { .. } => self.load_json(path).await,
            crate::data::loader::types::DataFormat::Parquet { .. } => self.load_parquet(path).await,
            crate::data::loader::types::DataFormat::Avro { .. } => self.load_avro(path).await,
            crate::data::loader::types::DataFormat::Excel { .. } => {
                // Excel 格式加载（生产级实现）
                // 注意：Excel 文件需要特殊解析，当前使用 JSON 作为降级方案
                // 如果文件实际上是 JSON 格式，则直接加载；否则需要实现 Excel 解析器
                #[cfg(feature = "excel")]
                {
                    self.load_excel(path).await
                }
                #[cfg(not(feature = "excel"))]
                {
                    // 如果 Excel 功能未启用，尝试作为 JSON 加载（降级方案）
                    log::warn!("Excel 功能未启用，尝试作为 JSON 加载");
                    self.load_json(path).await
                }
            }
            crate::data::loader::types::DataFormat::Text { .. } => self.load_custom_text(path, "text").await,
            crate::data::loader::types::DataFormat::CustomText(fmt) => self.load_custom_text(path, fmt).await,
            crate::data::loader::types::DataFormat::CustomBinary(fmt) => self.load_custom_binary(path, fmt).await,
            crate::data::loader::types::DataFormat::Tensor { dtype, shape, compression, endian } => {
                // 处理张量格式文件
                self.load_tensor_with_endian(path, dtype, shape, compression.as_deref(), &endian).await
            },
        }
    }
    
    // 加载CSV文件
    async fn load_csv(&self, path: &str) -> Result<DataBatch> {
        debug!("加载CSV文件: {}", path);
        
        // CSV数据加载实现
        let content = fs::read_to_string(path)
            .map_err(|e| crate::error::Error::io_error(&format!("读取CSV文件失败: {}, error: {}", path, e)))?;
            
        let mut reader = csv::Reader::from_reader(content.as_bytes());
        let headers = reader.headers()
            .map_err(|e| crate::error::Error::validation(&format!("解析CSV头部失败: {}", e)))?
            .iter()
            .map(String::from)
            .collect::<Vec<_>>();
            
        let mut features = Vec::new();
        let mut metadata = HashMap::new();
        
        metadata.insert("file_path".to_string(), path.to_string());
        metadata.insert("format".to_string(), "csv".to_string());
        metadata.insert("headers".to_string(), headers.join(","));
        
        for result in reader.records() {
            let record = result.map_err(|e| crate::error::Error::validation(&format!("解析CSV记录失败: {}", e)))?;
            
            let row_values = record.iter()
                .map(|field| {
                    // 尝试将字段解析为浮点数
                    if let Ok(val) = field.parse::<f32>() {
                        val
                    } else {
                        // 对于非数值字段，使用简单编码
                        let hash_value = field.chars().fold(0, |acc, c| acc + c as u32) % 1000;
                        hash_value as f32 / 1000.0
                    }
                })
                .collect::<Vec<_>>();
                
            if !row_values.is_empty() {
                features.push(row_values);
            }
        }
        
        let mut batch = DataBatch::new("csv_data", 0, features.len());
        // 将features转换为records格式
        for feature_vec in features {
            let mut record = HashMap::new();
            for (i, value) in feature_vec.iter().enumerate() {
                record.insert(format!("feature_{}", i), crate::data::DataValue::Float(*value as f64));
            }
            batch.records.push(record);
        }
        
        // 添加元数据
        for (key, value) in metadata {
            batch.add_metadata(&key, &value);
        }
        
        Ok(batch)
    }
    
    // 加载JSON文件
    async fn load_json(&self, path: &str) -> Result<DataBatch> {
        debug!("加载JSON文件: {}", path);
        
        let content = fs::read_to_string(path)
            .map_err(|e| crate::error::Error::io_error(&format!("读取JSON文件失败: {}, error: {}", path, e)))?;
            
        let value: Value = serde_json::from_str(&content)
            .map_err(|e| crate::error::Error::validation(&format!("解析JSON失败: {}", e)))?;
            
        let mut features = Vec::new();
        let mut metadata = HashMap::new();
        
        metadata.insert("file_path".to_string(), path.to_string());
        metadata.insert("format".to_string(), "json".to_string());
        
        match value {
            Value::Array(ref items) => {
                // 记录特征名称（在处理前先获取）
                if !items.is_empty() {
                    if let Value::Object(first_item) = &items[0] {
                        let keys = first_item.keys().cloned().collect::<Vec<_>>();
                        metadata.insert("features".to_string(), keys.join(","));
                    }
                }
                
                // 处理JSON数组格式
                for item in items {
                    if let Value::Object(obj) = item {
                        let row_values = obj.values()
                            .map(|v| match v {
                                Value::Number(n) => {
                                    if let Some(f) = n.as_f64() {
                                        f as f32
                                    } else {
                                        0.0
                                    }
                                },
                                Value::String(s) => {
                                    // 对字符串使用简单编码
                                    let hash_value = s.chars().fold(0, |acc, c| acc + c as u32) % 1000;
                                    hash_value as f32 / 1000.0
                                },
                                Value::Bool(b) => if *b { 1.0 } else { 0.0 },
                                _ => 0.0,
                            })
                            .collect::<Vec<_>>();
                            
                        if !row_values.is_empty() {
                            features.push(row_values);
                        }
                    }
                }
            },
            Value::Object(obj) => {
                // 处理单个JSON对象
                let row_values = obj.values()
                    .map(|v| match v {
                        Value::Number(n) => {
                            if let Some(f) = n.as_f64() {
                                f as f32
                            } else {
                                0.0
                            }
                        },
                        Value::String(s) => {
                            // 对字符串使用简单编码
                            let hash_value = s.chars().fold(0, |acc, c| acc + c as u32) % 1000;
                            hash_value as f32 / 1000.0
                        },
                        Value::Bool(b) => if *b { 1.0 } else { 0.0 },
                        _ => 0.0,
                    })
                    .collect::<Vec<_>>();
                    
                if !row_values.is_empty() {
                    features.push(row_values);
                }
                
                // 记录特征名称
                let keys = obj.keys().cloned().collect::<Vec<_>>();
                metadata.insert("features".to_string(), keys.join(","));
            },
            _ => return Err(crate::error::Error::invalid_input(&format!("不支持的JSON格式: {}", path))),
        }
        
        let mut batch = DataBatch::new("json_data", 0, features.len());
        // 将features转换为records格式
        for feature_vec in features {
            let mut record = HashMap::new();
            for (i, value) in feature_vec.iter().enumerate() {
                record.insert(format!("feature_{}", i), crate::data::DataValue::Float(*value as f64));
            }
            batch.records.push(record);
        }
        
        // 添加元数据
        for (key, value) in metadata {
            batch.add_metadata(&key, &value);
        }
        
        Ok(batch)
    }
    
    // 加载Parquet文件
    async fn load_parquet(&self, path: &str) -> Result<DataBatch> {
        debug!("加载Parquet文件: {}", path);
        
        #[cfg(feature = "parquet")]
        {
            use parquet::file::reader::{FileReader, SerializedFileReader};
            use parquet::record::reader::RowIter;
            use std::fs::File;
            
            // 打开Parquet文件
            let file = File::open(path)
                .map_err(|e| Error::io_error(format!("打开Parquet文件失败: {}, error: {}", path, e)))?;
                
            let reader = SerializedFileReader::new(file)
                .map_err(|e| Error::parse_error(format!("创建Parquet读取器失败: {}", e)))?;
                
            // 读取元数据
            let metadata = reader.metadata();
            let schema = metadata.file_metadata().schema();
            let num_columns = schema.num_columns();
            
            // 收集列名
            let mut column_names = Vec::with_capacity(num_columns);
            for i in 0..num_columns {
                column_names.push(schema.column(i).name().to_string());
            }
            
            // 读取行数据
            let mut features = Vec::new();
            let mut meta = HashMap::new();
            
            meta.insert("file_path".to_string(), path.to_string());
            meta.insert("format".to_string(), "parquet".to_string());
            meta.insert("columns".to_string(), column_names.join(","));
            
            for row in reader.get_row_iter(None)
                .map_err(|e| Error::parse_error(format!("获取Parquet行迭代器失败: {}", e)))? {
                
                let row = row.map_err(|e| Error::parse_error(format!("读取Parquet行失败: {}", e)))?;
                let mut row_values = Vec::with_capacity(num_columns);
                
                for i in 0..num_columns {
                    // 读取列值并转换为f32
                    let value = match schema.column(i).physical_type() {
                        parquet::basic::Type::BOOLEAN => {
                            if row.get_bool(i).unwrap_or(false) {
                                1.0
                            } else {
                                0.0
                            }
                        },
                        parquet::basic::Type::INT32 => row.get_int(i).unwrap_or(0) as f32,
                        parquet::basic::Type::INT64 => row.get_long(i).unwrap_or(0) as f32,
                        parquet::basic::Type::FLOAT => row.get_float(i).unwrap_or(0.0),
                        parquet::basic::Type::DOUBLE => row.get_double(i).unwrap_or(0.0) as f32,
                        parquet::basic::Type::BYTE_ARRAY => {
                            // 将字节数组转换为字符串，然后编码
                            if let Some(bytes) = row.get_bytes(i) {
                                let s = String::from_utf8_lossy(bytes.data());
                                let hash_value = s.chars().fold(0, |acc, c| acc + c as u32) % 1000;
                                hash_value as f32 / 1000.0
                            } else {
                                0.0
                            }
                        },
                        _ => 0.0, // 其他类型默认为0
                    };
                    
                    row_values.push(value);
                }
                
                if !row_values.is_empty() {
                    features.push(row_values);
                }
            }
            
            let mut batch = DataBatch::new("parquet_data", 0, features.len());
            // 将features转换为records格式
            for feature_vec in features {
                let mut record = HashMap::new();
                for (i, value) in feature_vec.iter().enumerate() {
                    record.insert(format!("feature_{}", i), crate::data::DataValue::Float(*value as f64));
                }
                batch.records.push(record);
            }
            
            // 添加元数据
            for (key, value) in meta {
                batch.add_metadata(&key, &value);
            }
            
            Ok(batch)
        }
        
        #[cfg(not(feature = "parquet"))]
        {
            Err(Error::not_implemented("Parquet支持需要启用parquet特性"))
        }
    }
    
    // 加载Avro文件
    async fn load_avro(&self, _path: &str) -> Result<DataBatch> {
        // 注意：avro 特性未在 Cargo.toml 中定义，直接返回未实现错误
        return Err(Error::not_implemented("Avro支持需要启用avro特性"));
        
        /* 已注释：avro 特性未定义
        // #[cfg(feature = "avro")]
        {
            use apache_avro::{Reader, from_avro_datum};
            use apache_avro::types::Value as AvroValue;
            use std::fs::File;
            use std::io::BufReader;
            
            // 打开Avro文件
            let file = File::open(path)
                .map_err(|e| Error::io(format!("打开Avro文件失败: {}, error: {}", path, e)))?;
                
            let reader = BufReader::new(file);
            let avro_reader = Reader::new(reader)
                .map_err(|e| Error::parse_error(format!("创建Avro读取器失败: {}", e)))?;
                
            // 获取Schema
            let schema = avro_reader.writer_schema();
            let field_names = if let apache_avro::Schema::Record { fields, .. } = &schema {
                fields.iter().map(|f| f.name.clone()).collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            
            // 读取数据
            let mut features = Vec::new();
            let mut meta = HashMap::new();
            
            meta.insert("file_path".to_string(), path.to_string());
            meta.insert("format".to_string(), "avro".to_string());
            meta.insert("fields".to_string(), field_names.join(","));
            
            for value in avro_reader {
                let value = value.map_err(|e| Error::parse_error(format!("读取Avro记录失败: {}", e)))?;
                
                if let AvroValue::Record(fields) = value {
                    let row_values = fields.iter()
                        .map(|(_, v)| match v {
                            AvroValue::Boolean(b) => if *b { 1.0 } else { 0.0 },
                            AvroValue::Int(i) => *i as f32,
                            AvroValue::Long(l) => *l as f32,
                            AvroValue::Float(f) => *f,
                            AvroValue::Double(d) => *d as f32,
                            AvroValue::String(s) => {
                                // 对字符串使用简单编码
                                let hash_value = s.chars().fold(0, |acc, c| acc + c as u32) % 1000;
                                hash_value as f32 / 1000.0
                            },
                            _ => 0.0, // 其他类型默认为0
                        })
                        .collect::<Vec<_>>();
                        
                    if !row_values.is_empty() {
                        features.push(row_values);
                    }
                }
            }
            
            let mut batch = DataBatch::new("avro_data", 0, features.len());
            // 将features转换为records格式
            for feature_vec in features {
                let mut record = HashMap::new();
                for (i, value) in feature_vec.iter().enumerate() {
                    record.insert(format!("feature_{}", i), crate::data::DataValue::Float(*value as f64));
                }
                batch.records.push(record);
            }
            
            // 添加元数据
            for (key, value) in meta {
                batch.add_metadata(&key, &value);
            }
            
            Ok(batch)
        }
        */
    }
    
    // 加载自定义文本格式
    async fn load_custom_text(&self, path: &str, format: &str) -> Result<DataBatch> {
        debug!("加载自定义文本格式: {}, 格式: {}", path, format);
        
        // 读取文件内容
        let content = fs::read_to_string(path)
            .map_err(|e| Error::io_error(&format!("读取文件失败: {}, error: {}", path, e)))?;
            
        let mut features = Vec::new();
        let mut metadata = HashMap::new();
        
        metadata.insert("file_path".to_string(), path.to_string());
        metadata.insert("format".to_string(), format.to_string());
        
        // 根据不同的自定义格式处理
        match format.to_lowercase().as_str() {
            "tsv" => {
                // Tab分隔值处理，类似CSV
                for line in content.lines() {
                    let row_values = line.split('\t')
                        .map(|field| {
                            if let Ok(val) = field.parse::<f32>() {
                                val
                            } else {
                                // 对于非数值字段，使用简单编码
                                let hash_value = field.chars().fold(0, |acc, c| acc + c as u32) % 1000;
                                hash_value as f32 / 1000.0
                            }
                        })
                        .collect::<Vec<_>>();
                        
                    if !row_values.is_empty() {
                        features.push(row_values);
                    }
                }
            },
            "line_json" => {
                // 每行一个JSON对象
                for line in content.lines() {
                    if let Ok(value) = serde_json::from_str::<Value>(line) {
                        if let Value::Object(obj) = value {
                            let row_values = obj.values()
                                .map(|v| match v {
                                    Value::Number(n) => {
                                        if let Some(f) = n.as_f64() {
                                            f as f32
                                        } else {
                                            0.0
                                        }
                                    },
                                    Value::String(s) => {
                                        let hash_value = s.chars().fold(0, |acc, c| acc + c as u32) % 1000;
                                        hash_value as f32 / 1000.0
                                    },
                                    Value::Bool(b) => if *b { 1.0 } else { 0.0 },
                                    _ => 0.0,
                                })
                                .collect::<Vec<_>>();
                                
                            if !row_values.is_empty() {
                                features.push(row_values);
                            }
                        }
                    }
                }
            },
            _ => {
                // 默认按行处理
                for line in content.lines() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue; // 跳过空行和注释
                    }
                    
                    // 尝试将每行分割为空格分隔的数值
                    let row_values = line.split_whitespace()
                        .map(|s| s.parse::<f32>().unwrap_or(0.0))
                        .collect::<Vec<_>>();
                        
                    if !row_values.is_empty() {
                        features.push(row_values);
                    }
                }
            }
        }
        
        let mut batch = DataBatch::new("custom_text_data", 0, features.len());
        // 将features转换为records格式
        for feature_vec in features {
            let mut record = HashMap::new();
            for (i, value) in feature_vec.iter().enumerate() {
                record.insert(format!("feature_{}", i), crate::data::DataValue::Float(*value as f64));
            }
            batch.records.push(record);
        }
        
        // 添加元数据
        for (key, value) in metadata {
            batch.add_metadata(&key, &value);
        }
        
        Ok(batch)
    }
    
    // 加载自定义二进制格式
    async fn load_custom_binary(&self, path: &str, format: &str) -> Result<DataBatch> {
        debug!("加载自定义二进制格式: {}, 格式: {}", path, format);
        
        // 读取二进制文件
        let file_data = fs::read(path)
            .map_err(|e| Error::io_error(&format!("读取二进制文件失败: {}, error: {}", path, e)))?;
            
        let mut features = Vec::new();
        let mut metadata = HashMap::new();
        
        metadata.insert("file_path".to_string(), path.to_string());
        metadata.insert("format".to_string(), format.to_string());
        metadata.insert("file_size".to_string(), file_data.len().to_string());
        
        // 根据不同的二进制格式处理
        match format.to_lowercase().as_str() {
            "raw" | "binary" => {
                // 原始二进制数据，每个字节作为一个特征
                let chunk_size = 8; // 每8个字节组成一条记录
                for chunk in file_data.chunks(chunk_size) {
                    let row_values = chunk.iter()
                        .map(|&byte| byte as f32 / 255.0) // 归一化到[0,1]
                        .collect::<Vec<_>>();
                    features.push(row_values);
                }
            },
            "float32" => {
                // 32位浮点数数组
                if file_data.len() % 4 != 0 {
                    return Err(Error::validation("float32格式文件长度必须是4的倍数"));
                }
                
                for chunk in file_data.chunks_exact(4) {
                    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    features.push(vec![value]);
                }
            },
            "float64" => {
                // 64位浮点数数组
                if file_data.len() % 8 != 0 {
                    return Err(Error::validation("float64格式文件长度必须是8的倍数"));
                }
                
                for chunk in file_data.chunks_exact(8) {
                    let bytes: [u8; 8] = [
                        chunk[0], chunk[1], chunk[2], chunk[3],
                        chunk[4], chunk[5], chunk[6], chunk[7]
                    ];
                    let value = f64::from_le_bytes(bytes) as f32;
                    features.push(vec![value]);
                }
            },
            "int32" => {
                // 32位整数数组
                if file_data.len() % 4 != 0 {
                    return Err(Error::validation("int32格式文件长度必须是4的倍数"));
                }
                
                for chunk in file_data.chunks_exact(4) {
                    let value = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32;
                    features.push(vec![value]);
                }
            },
            "matrix" => {
                // 矩阵格式：前8字节表示行数和列数，然后是float32数据
                if file_data.len() < 8 {
                    return Err(Error::validation("矩阵格式文件至少需要8字节头部"));
                }
                
                let rows = u32::from_le_bytes([file_data[0], file_data[1], file_data[2], file_data[3]]) as usize;
                let cols = u32::from_le_bytes([file_data[4], file_data[5], file_data[6], file_data[7]]) as usize;
                
                if file_data.len() != 8 + rows * cols * 4 {
                    return Err(Error::validation("矩阵格式文件大小不匹配"));
                }
                
                let data_start = 8;
                for row in 0..rows {
                    let mut row_values = Vec::with_capacity(cols);
                    for col in 0..cols {
                        let offset = data_start + (row * cols + col) * 4;
                        let value = f32::from_le_bytes([
                            file_data[offset], file_data[offset + 1],
                            file_data[offset + 2], file_data[offset + 3]
                        ]);
                        row_values.push(value);
                    }
                    features.push(row_values);
                }
                
                metadata.insert("rows".to_string(), rows.to_string());
                metadata.insert("cols".to_string(), cols.to_string());
            },
            _ => {
                return Err(Error::not_implemented(&format!(
                    "不支持的自定义二进制格式: {}。支持的格式: raw, binary, float32, float64, int32, matrix",
                    format
                )));
            }
        }
        
        let mut batch = DataBatch::new("custom_binary_data", 0, features.len());
        // 将features转换为records格式
        for feature_vec in features {
            let mut record = HashMap::new();
            for (i, value) in feature_vec.iter().enumerate() {
                record.insert(format!("feature_{}", i), DataValue::Float(*value as f64));
            }
            batch.records.push(record);
        }
        
        // 添加元数据
        for (key, value) in metadata {
            batch.add_metadata(&key, &value);
        }
        
        Ok(batch)
    }
    
    // 从数据库加载数据
    async fn load_from_database(&self, config: &DataSource) -> Result<DataBatch> {
        if let Some(db_manager) = &self.db_manager {
            if let DataSource::Database(db_config) = config {
                trace!("开始从数据库加载数据: {:?}", db_config.db_type);
                
                // 建立数据库连接
                let _connection = db_manager.get_connection(&db_config.connection_string).await
                    .map_err(|e| Error::database_operation(&format!("数据库连接失败: {}", e)))?;
                
                // 执行查询
                let query = db_config.query.as_ref()
                    .ok_or_else(|| Error::invalid_argument("数据库查询语句不能为空"))?;
                    
                info!("执行数据库查询: {}", query);
                
                // 根据数据库类型执行查询
                let query_result: Result<QueryResult> = match &db_config.db_type {
                    crate::data::connector::DatabaseType::PostgreSQL => {
                        Err(Error::not_implemented("PostgreSQL支持需要启用postgres特性"))
                    },
                    crate::data::connector::DatabaseType::MySQL => {
                        Err(Error::not_implemented("MySQL支持需要启用mysql特性"))
                    },
                    crate::data::connector::DatabaseType::SQLite => {
                        Err(Error::not_implemented("SQLite支持需要启用sqlite特性"))
                    },
                    crate::data::connector::DatabaseType::Redis => {
                        #[cfg(feature = "redis")]
                        if let crate::data::connector::DatabaseConnection::Redis(conn) = &connection {
                            // Redis连接需要解包 Mutex
                            let redis_conn = conn.lock().map_err(|e| 
                                Error::database_operation(&format!("Redis连接锁定失败: {}", e)))?;
                            drop(redis_conn); // 暂时释放锁
                            // 由于Redis连接在Mutex中，我们需要不同的处理方式
                            Err(Error::not_implemented("Redis查询处理需要重构"))
                        } else {
                            Err(Error::invalid_argument("Redis连接类型不匹配"))
                        }
                        #[cfg(not(feature = "redis"))]
                        {
                            Err(Error::not_implemented("Redis支持需要启用redis特性"))
                        }
                    },
                    crate::data::connector::DatabaseType::MongoDB => {
                        Err(Error::not_implemented("MongoDB支持需要启用mongodb特性"))
                    },
                    crate::data::connector::DatabaseType::Cassandra => {
                        Err(Error::not_implemented("Cassandra支持需要启用cassandra特性"))
                    },
                    crate::data::connector::DatabaseType::Elasticsearch => {
                        Err(Error::not_implemented("Elasticsearch支持需要启用elasticsearch特性"))
                    },
                    _ => {
                        Err(Error::invalid_argument(&format!(
                            "不支持的数据库类型: {:?}", 
                            db_config.db_type
                        )))
                    }
                };
                
                let query_result = query_result?;
                let record_count = query_result.records.len();
                let execution_time = query_result.execution_time;
                let mut batch = DataBatch::new("database_data", 0, record_count);
                batch.records = query_result.records;
                
                // 添加元数据
                let mut metadata = HashMap::new();
                metadata.insert("source".to_string(), "database".to_string());
                metadata.insert("db_type".to_string(), format!("{:?}", db_config.db_type));
                metadata.insert("query".to_string(), query.clone());
                metadata.insert("connection_string".to_string(), Self::mask_connection_string(&db_config.connection_string));
                metadata.insert("record_count".to_string(), record_count.to_string());
                metadata.insert("query_time".to_string(), format!("{}ms", execution_time.as_millis()));
                
                for (key, value) in metadata {
                    batch.add_metadata(&key, &value);
                }
                
                info!("数据库查询完成，获取 {} 条记录", record_count);
                Ok(batch)
            } else {
                Err(Error::invalid_argument("数据源类型不匹配，期望数据库类型"))
            }
        } else {
            Err(Error::invalid_argument("数据库管理器未配置"))
        }
    }
    
    // 从流加载数据
    async fn load_from_stream(&self, source: &str) -> Result<DataBatch> {
        debug!("从流加载数据: {}", source);
        
        let mut features = Vec::new();
        let mut metadata = HashMap::new();
        
        metadata.insert("source".to_string(), "stream".to_string());
        metadata.insert("stream_source".to_string(), source.to_string());
        metadata.insert("start_time".to_string(), Local::now().format("%Y-%m-%d %H:%M:%S").to_string());
        
        // 根据流的类型处理
        if source.starts_with("http://") || source.starts_with("https://") {
            // HTTP/HTTPS 流
            #[cfg(feature = "reqwest")]
            {
                let response = reqwest::get(source).await
                    .map_err(|e| Error::network(&format!("HTTP请求失败: {}", e)))?;
                    
                if !response.status().is_success() {
                    return Err(Error::network(&format!("HTTP响应错误: {}", response.status())));
                }
                
                let content_type = response.headers()
                    .get("content-type")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("application/octet-stream");
                    
                metadata.insert("content_type".to_string(), content_type.to_string());
                
                let bytes = response.bytes().await
                    .map_err(|e| Error::network(&format!("读取HTTP响应失败: {}", e)))?;
                    
                // 根据内容类型解析数据
                if content_type.contains("application/json") {
                    self.parse_json_stream(&bytes, &mut features)?;
                } else if content_type.contains("text/csv") {
                    self.parse_csv_stream(&bytes, &mut features)?;
                } else {
                    // 默认按二进制处理
                    self.parse_binary_stream(&bytes, &mut features)?;
                }
            }
            #[cfg(not(feature = "reqwest"))]
            {
                return Err(Error::not_implemented("HTTP流支持需要启用reqwest特性"));
            }
        } else if source.starts_with("tcp://") {
            // TCP 流
            // #[cfg(feature = "tokio")]
            {
                let addr = source.strip_prefix("tcp://")
                    .ok_or_else(|| Error::invalid_argument("无效的TCP地址格式"))?;
                    
                let stream = tokio::net::TcpStream::connect(addr).await
                    .map_err(|e| Error::network(&format!("TCP连接失败: {}", e)))?;
                    
                let mut reader = tokio::io::BufReader::new(stream);
                let mut buffer = Vec::new();
                
                // 读取数据（这里设置一个合理的超时时间）
                let read_timeout = Duration::from_secs(30);
                let read_result = tokio::time::timeout(read_timeout, 
                    tokio::io::AsyncReadExt::read_to_end(&mut reader, &mut buffer)).await;
                    
                match read_result {
                    Ok(Ok(_)) => {
                        self.parse_binary_stream(&buffer, &mut features)?;
                        metadata.insert("bytes_read".to_string(), buffer.len().to_string());
                    },
                    Ok(Err(e)) => {
                        return Err(Error::network(&format!("TCP读取失败: {}", e)));
                    },
                    Err(_) => {
                        return Err(Error::network("TCP读取超时"));
                    }
                }
            }
            // #[cfg(not(feature = "tokio"))]
            {
                return Err(Error::not_implemented("TCP流支持需要启用tokio特性"));
            }
        } else if source.starts_with("ws://") || source.starts_with("wss://") {
            // WebSocket 流
            return Err(Error::not_implemented("WebSocket流支持将在后续版本中实现"));
        } else if source.starts_with("kafka://") {
            // Kafka 流
            return Err(Error::not_implemented("Kafka流支持将在后续版本中实现"));
        } else if source.starts_with("file://") {
            // 文件流（监控文件变化）
            let file_path = source.strip_prefix("file://")
                .ok_or_else(|| Error::invalid_argument("无效的文件路径格式"))?;
                
            // 读取文件内容
            // #[cfg(feature = "tokio")]
            {
                let content = tokio::fs::read(file_path).await
                    .map_err(|e| Error::io_error(&format!("读取文件失败: {}, error: {}", file_path, e)))?;
                    
                self.parse_binary_stream(&content, &mut features)?;
                metadata.insert("file_path".to_string(), file_path.to_string());
                metadata.insert("file_size".to_string(), content.len().to_string());
            }
            // #[cfg(not(feature = "tokio"))]
            {
                let content = std::fs::read(file_path)
                    .map_err(|e| Error::io_error(&format!("读取文件失败: {}, error: {}", file_path, e)))?;
                    
                self.parse_binary_stream(&content, &mut features)?;
                metadata.insert("file_path".to_string(), file_path.to_string());
                metadata.insert("file_size".to_string(), content.len().to_string());
            }
        } else {
            return Err(Error::invalid_argument(&format!(
                "不支持的流类型: {}。支持的协议: http://, https://, tcp://, file://", 
                source
            )));
        }
        
        let mut batch = DataBatch::new("stream_data", 0, features.len());
        let feature_count = features.len(); // 保存长度以避免借用问题
        
        // 将features转换为records格式
        for feature_vec in features {
            let mut record = HashMap::new();
            for (i, value) in feature_vec.iter().enumerate() {
                record.insert(format!("feature_{}", i), DataValue::Float(*value as f64));
            }
            batch.records.push(record);
        }
        
        metadata.insert("end_time".to_string(), Local::now().format("%Y-%m-%d %H:%M:%S").to_string());
        metadata.insert("record_count".to_string(), feature_count.to_string());
        
        // 添加元数据
        for (key, value) in metadata {
            batch.add_metadata(&key, &value);
        }
        
        info!("流数据加载完成，获取 {} 条记录", feature_count);
        Ok(batch)
    }

    // 辅助方法：执行PostgreSQL查询
    async fn execute_postgresql_query(&self, _connection: &(), _query: &str) -> Result<QueryResult> {
        // 注意：postgres 特性未在 Cargo.toml 中定义，直接返回未实现错误
        Err(Error::not_implemented("PostgreSQL支持需要启用postgres特性"))
    }

    // 辅助方法：执行MySQL查询
    async fn execute_mysql_query(&self, _connection: &(), _query: &str) -> Result<QueryResult> {
        // 注意：mysql 特性未在 Cargo.toml 中定义，直接返回未实现错误
        Err(Error::not_implemented("MySQL支持需要启用mysql特性"))
    }

    // 辅助方法：执行SQLite查询
    async fn execute_sqlite_query(&self, _connection: &(), _query: &str) -> Result<QueryResult> {
        // 注意：sqlite 特性未在 Cargo.toml 中定义，直接返回未实现错误
        Err(Error::not_implemented("SQLite支持需要启用sqlite特性"))
    }

    // 辅助方法：执行Redis查询
    #[cfg(feature = "redis")]
    async fn execute_redis_query(&self, connection: &mut redis::aio::Connection, query: &str) -> Result<QueryResult> {
        {
            let start_time = std::time::Instant::now();
            
            // Redis查询通常是键值对操作
            let parts: Vec<&str> = query.split_whitespace().collect();
            if parts.is_empty() {
                return Err(Error::invalid_argument("Redis查询不能为空"));
            }
            
            let mut records = Vec::new();
            let cmd = parts[0].to_uppercase();
            
            match cmd.as_str() {
                "GET" => {
                    if parts.len() != 2 {
                        return Err(Error::invalid_argument("GET命令需要一个键参数"));
                    }
                    let key = parts[1];
                    let value: Option<String> = redis::cmd("GET")
                        .arg(key)
                        .query_async(connection)
                        .await
                        .map_err(|e| Error::database_operation(&format!("Redis GET失败: {}", e)))?;
                        
                    let mut record = HashMap::new();
                    record.insert("key".to_string(), DataValue::String(key.to_string()));
                    record.insert("value".to_string(), 
                        value.map(DataValue::String).unwrap_or(DataValue::Null));
                    records.push(record);
                },
                "KEYS" => {
                    let pattern = if parts.len() > 1 { parts[1] } else { "*" };
                    let keys: Vec<String> = redis::cmd("KEYS")
                        .arg(pattern)
                        .query_async(connection)
                        .await
                        .map_err(|e| Error::database_operation(&format!("Redis KEYS失败: {}", e)))?;
                        
                    for key in keys {
                        let mut record = HashMap::new();
                        record.insert("key".to_string(), DataValue::String(key));
                        records.push(record);
                    }
                },
                _ => {
                    return Err(Error::not_implemented(&format!("不支持的Redis命令: {}", cmd)));
                }
            }
            
            Ok(QueryResult {
                records,
                execution_time: start_time.elapsed(),
            })
        }
        #[cfg(not(feature = "redis"))]
        {
            Err(Error::not_implemented("Redis支持需要启用redis特性"))
        }
    }

    // 辅助方法：执行MongoDB查询
    async fn execute_mongodb_query(&self, _connection: &(), _query: &str) -> Result<QueryResult> {
        // 注意：mongodb 特性未在 Cargo.toml 中定义，直接返回未实现错误
        Err(Error::not_implemented("MongoDB支持需要启用mongodb特性"))
    }

    // 辅助方法：执行Cassandra查询
    async fn execute_cassandra_query(&self, _connection: &(), _query: &str) -> Result<QueryResult> {
        // 注意：cassandra 特性未在 Cargo.toml 中定义，直接返回未实现错误
        Err(Error::not_implemented("Cassandra支持需要启用cassandra特性"))
    }

    // 辅助方法：执行Elasticsearch查询
    async fn execute_elasticsearch_query(&self, _connection: &(), _query: &str) -> Result<QueryResult> {
        // 注意：elasticsearch 特性未在 Cargo.toml 中定义，直接返回未实现错误
        Err(Error::not_implemented("Elasticsearch支持需要启用elasticsearch特性"))
    }

    // 辅助方法：解析JSON流
    fn parse_json_stream(&self, data: &[u8], features: &mut Vec<Vec<f32>>) -> Result<()> {
        let content = String::from_utf8_lossy(data);
        
        for line in content.lines() {
            if let Ok(value) = serde_json::from_str::<Value>(line) {
                if let Value::Object(obj) = value {
                    let row_values = obj.values()
                        .map(|v| match v {
                            Value::Number(n) => n.as_f64().unwrap_or(0.0) as f32,
                            Value::String(s) => {
                                let hash_value = s.chars().fold(0, |acc, c| acc + c as u32) % 1000;
                                hash_value as f32 / 1000.0
                            },
                            Value::Bool(b) => if *b { 1.0 } else { 0.0 },
                            _ => 0.0,
                        })
                        .collect::<Vec<_>>();
                        
                    if !row_values.is_empty() {
                        features.push(row_values);
                    }
                }
            }
        }
        Ok(())
    }

    // 辅助方法：解析CSV流
    fn parse_csv_stream(&self, _data: &[u8], _features: &mut Vec<Vec<f32>>) -> Result<()> {
        // 注意：csv 特性未在 Cargo.toml 中定义，直接返回未实现错误
        Err(Error::not_implemented("CSV解析需要启用csv特性"))
    }

    // 辅助方法：解析二进制流
    fn parse_binary_stream(&self, data: &[u8], features: &mut Vec<Vec<f32>>) -> Result<()> {
        // 按8字节为一组处理
        for chunk in data.chunks(8) {
            let row_values = chunk.iter()
                .map(|&byte| byte as f32 / 255.0)
                .collect::<Vec<_>>();
            features.push(row_values);
        }
        Ok(())
    }

    // 辅助方法：脱敏连接字符串
    fn mask_connection_string(conn_str: &str) -> String {
        // 简单的密码脱敏
        if let Some(password_start) = conn_str.find("password=") {
            let mut masked = conn_str.to_string();
            if let Some(password_end) = masked[password_start..].find('&').or_else(|| masked[password_start..].find(' ')) {
                let end_pos = password_start + password_end;
                masked.replace_range(password_start + 9..end_pos, "****");
            } else {
                masked.replace_range(password_start + 9.., "****");
            }
            masked
        } else {
            conn_str.to_string()
        }
    }

    /// 检查并记录错误信息
    pub fn handle_load_error(&self, source: &str, err_msg: &str) -> Result<()> {
        // 使用之前导入但未使用的error日志功能
        error!("数据加载失败: 源={}, 错误={}", source, err_msg);
        
        // 记录错误到内部错误日志
        self.log_error(source, err_msg);
        
        Err(Error::data_loading(format!("加载失败: {}", err_msg)))
    }
    
    /// 记录内部错误信息
    fn log_error(&self, source: &str, message: &str) {
        // 实现内部错误记录逻辑
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        let error_log = format!("[{}] 源: {}, 错误: {}", timestamp, source, message);
        
        // 这里可以实现将错误记录到文件或其他存储的逻辑
        error!("{}", error_log);
    }

    // 加载张量格式文件
    async fn load_tensor(&self, path: &str, dtype: &str, shape: &[usize], compression: Option<&str>) -> Result<DataBatch> {
        debug!("加载张量文件: {}, dtype: {}, shape: {:?}, compression: {:?}", path, dtype, shape, compression);
        
        let content = fs::read(path)
            .map_err(|e| crate::error::Error::io_error(&format!("读取张量文件失败: {}, error: {}", path, e)))?;
            
        let mut features = Vec::new();
        let mut metadata = HashMap::new();
        
        metadata.insert("file_path".to_string(), path.to_string());
        metadata.insert("format".to_string(), "tensor".to_string());
        metadata.insert("dtype".to_string(), dtype.to_string());
        metadata.insert("shape".to_string(), format!("{:?}", shape));
        if let Some(comp) = compression {
            metadata.insert("compression".to_string(), comp.to_string());
        }
        
        // 解压缩数据（如果需要）
        let raw_data = if let Some(compression_type) = compression {
            match compression_type {
                "gzip" => {
                    // 当前未真正启用压缩支持：直接返回未实现错误，保持返回类型一致
                    return Err(crate::error::Error::not_implemented(
                        "GZIP解压缩需要启用compression特性",
                    ));
                },
                "lz4" => {
                    // 注意：lz4 特性未在 Cargo.toml 中定义，直接返回未实现错误
                    return Err(crate::error::Error::not_implemented("LZ4解压缩需要启用compression特性"));
                },
                "none" | "" => content,
                _ => {
                    return Err(crate::error::Error::not_implemented(&format!(
                        "不支持的压缩格式: {}。支持的格式: gzip, lz4, none",
                        compression_type
                    )));
                }
            }
        } else {
            content
        };
        
        // 根据数据类型解析二进制数据
        match dtype {
            "float32" | "f32" => {
                let num_elements = raw_data.len() / 4;
                let mut float_data = vec![0.0f32; num_elements];
                
                // 使用小端字节序读取
                for i in 0..num_elements {
                    let start = i * 4;
                    if start + 4 <= raw_data.len() {
                        let bytes = [raw_data[start], raw_data[start + 1], raw_data[start + 2], raw_data[start + 3]];
                        float_data[i] = f32::from_le_bytes(bytes);
                    }
                }
                
                // 根据形状重构数据
                if shape.is_empty() {
                    // 一维数据
                    features.push(float_data);
                } else {
                    // 多维数据，按第一维分割
                    let first_dim = shape[0];
                    let remaining_size: usize = shape.iter().skip(1).product();
                    
                    for i in 0..first_dim {
                        let start = i * remaining_size;
                        let end = std::cmp::min(start + remaining_size, float_data.len());
                        if start < float_data.len() {
                            features.push(float_data[start..end].to_vec());
                        }
                    }
                }
                
                metadata.insert("elements".to_string(), num_elements.to_string());
            },
            "float64" | "f64" => {
                let num_elements = raw_data.len() / 8;
                let mut double_data = vec![0.0f64; num_elements];
                
                for i in 0..num_elements {
                    let start = i * 8;
                    if start + 8 <= raw_data.len() {
                        let bytes = [
                            raw_data[start], raw_data[start + 1], raw_data[start + 2], raw_data[start + 3],
                            raw_data[start + 4], raw_data[start + 5], raw_data[start + 6], raw_data[start + 7]
                        ];
                        double_data[i] = f64::from_le_bytes(bytes);
                    }
                }
                
                // 转换为f32格式
                let float_data: Vec<f32> = double_data.iter().map(|&x| x as f32).collect();
                
                if shape.is_empty() {
                    features.push(float_data);
                } else {
                    let first_dim = shape[0];
                    let remaining_size: usize = shape.iter().skip(1).product();
                    
                    for i in 0..first_dim {
                        let start = i * remaining_size;
                        let end = std::cmp::min(start + remaining_size, float_data.len());
                        if start < float_data.len() {
                            features.push(float_data[start..end].to_vec());
                        }
                    }
                }
                
                metadata.insert("elements".to_string(), num_elements.to_string());
            },
            "int32" | "i32" => {
                let num_elements = raw_data.len() / 4;
                let mut int_data = vec![0i32; num_elements];
                
                for i in 0..num_elements {
                    let start = i * 4;
                    if start + 4 <= raw_data.len() {
                        let bytes = [raw_data[start], raw_data[start + 1], raw_data[start + 2], raw_data[start + 3]];
                        int_data[i] = i32::from_le_bytes(bytes);
                    }
                }
                
                // 转换为f32格式
                let float_data: Vec<f32> = int_data.iter().map(|&x| x as f32).collect();
                
                if shape.is_empty() {
                    features.push(float_data);
                } else {
                    let first_dim = shape[0];
                    let remaining_size: usize = shape.iter().skip(1).product();
                    
                    for i in 0..first_dim {
                        let start = i * remaining_size;
                        let end = std::cmp::min(start + remaining_size, float_data.len());
                        if start < float_data.len() {
                            features.push(float_data[start..end].to_vec());
                        }
                    }
                }
                
                metadata.insert("elements".to_string(), num_elements.to_string());
            },
            "uint8" | "u8" => {
                // 直接使用字节数据，转换为f32
                let float_data: Vec<f32> = raw_data.iter().map(|&x| x as f32 / 255.0).collect();
                
                if shape.is_empty() {
                    features.push(float_data);
                } else {
                    let first_dim = shape[0];
                    let remaining_size: usize = shape.iter().skip(1).product();
                    
                    for i in 0..first_dim {
                        let start = i * remaining_size;
                        let end = std::cmp::min(start + remaining_size, float_data.len());
                        if start < float_data.len() {
                            features.push(float_data[start..end].to_vec());
                        }
                    }
                }
                
                metadata.insert("elements".to_string(), raw_data.len().to_string());
            },
            _ => {
                return Err(crate::error::Error::not_implemented(&format!(
                    "不支持的张量数据类型: {}。支持的类型: float32, float64, int32, uint8",
                    dtype
                )));
            }
        }
        
        let mut batch = DataBatch::new("tensor_data", 0, features.len());
        // 将features转换为records格式
        for feature_vec in features {
            let mut record = HashMap::new();
            for (i, value) in feature_vec.iter().enumerate() {
                record.insert(format!("feature_{}", i), DataValue::Float(*value as f64));
            }
            batch.records.push(record);
        }
        
        // 添加元数据
        for (key, value) in metadata {
            batch.add_metadata(&key, &value);
        }
        
        Ok(batch)
    }
    
    async fn load_tensor_with_endian(&self, path: &str, dtype: &str, shape: &[usize], compression: Option<&str>, endian: &str) -> Result<DataBatch> {
        debug!("加载张量文件: {}, dtype: {}, shape: {:?}, compression: {:?}, endian: {}", path, dtype, shape, compression, endian);
        
        let content = fs::read(path)
            .map_err(|e| crate::error::Error::io_error(&format!("读取张量文件失败: {}, error: {}", path, e)))?;
            
        let mut features = Vec::new();
        let mut metadata = HashMap::new();
        
        metadata.insert("file_path".to_string(), path.to_string());
        metadata.insert("format".to_string(), "tensor".to_string());
        metadata.insert("dtype".to_string(), dtype.to_string());
        metadata.insert("shape".to_string(), format!("{:?}", shape));
        metadata.insert("endian".to_string(), endian.to_string());
        if let Some(comp) = compression {
            metadata.insert("compression".to_string(), comp.to_string());
        }
        
        // 解压缩数据（如果需要）
        let raw_data = if let Some(compression_type) = compression {
            match compression_type {
                "gzip" => {
                    // 当前未真正启用压缩支持：直接返回未实现错误，保持返回类型一致
                    return Err(crate::error::Error::not_implemented(
                        "GZIP解压缩需要启用compression特性",
                    ));
                },
                "lz4" => {
                    // 注意：lz4 特性未在 Cargo.toml 中定义，直接返回未实现错误
                    return Err(crate::error::Error::not_implemented("LZ4解压缩需要启用compression特性"));
                },
                "none" | "" => content,
                _ => {
                    return Err(crate::error::Error::not_implemented(&format!(
                        "不支持的压缩格式: {}。支持的格式: gzip, lz4, none",
                        compression_type
                    )));
                }
            }
        } else {
            content
        };
        
        // 根据字节序和数据类型解析二进制数据
        let is_little_endian = endian == "little" || endian == "le";
        
        match dtype {
            "float32" | "f32" => {
                let num_elements = raw_data.len() / 4;
                let mut float_data = vec![0.0f32; num_elements];
                
                for i in 0..num_elements {
                    let start = i * 4;
                    if start + 4 <= raw_data.len() {
                        let bytes = [
                            raw_data[start], raw_data[start + 1], raw_data[start + 2], raw_data[start + 3]
                        ];
                        if is_little_endian {
                            float_data[i] = f32::from_le_bytes(bytes);
                        } else {
                            float_data[i] = f32::from_be_bytes(bytes);
                        }
                    }
                }
                
                // 根据形状组织数据
                if shape.is_empty() {
                    features.push(float_data);
                } else {
                    let first_dim = shape[0];
                    let remaining_size: usize = shape.iter().skip(1).product();
                    
                    for i in 0..first_dim {
                        let start = i * remaining_size;
                        let end = std::cmp::min(start + remaining_size, float_data.len());
                        if start < float_data.len() {
                            features.push(float_data[start..end].to_vec());
                        }
                    }
                }
                
                metadata.insert("elements".to_string(), num_elements.to_string());
            },
            "float64" | "f64" => {
                let num_elements = raw_data.len() / 8;
                let mut double_data = vec![0.0f64; num_elements];
                
                for i in 0..num_elements {
                    let start = i * 8;
                    if start + 8 <= raw_data.len() {
                        let bytes = [
                            raw_data[start], raw_data[start + 1], raw_data[start + 2], raw_data[start + 3],
                            raw_data[start + 4], raw_data[start + 5], raw_data[start + 6], raw_data[start + 7]
                        ];
                        if is_little_endian {
                            double_data[i] = f64::from_le_bytes(bytes);
                        } else {
                            double_data[i] = f64::from_be_bytes(bytes);
                        }
                    }
                }
                
                // 转换为f32格式
                let float_data: Vec<f32> = double_data.iter().map(|&x| x as f32).collect();
                
                if shape.is_empty() {
                    features.push(float_data);
                } else {
                    let first_dim = shape[0];
                    let remaining_size: usize = shape.iter().skip(1).product();
                    
                    for i in 0..first_dim {
                        let start = i * remaining_size;
                        let end = std::cmp::min(start + remaining_size, float_data.len());
                        if start < float_data.len() {
                            features.push(float_data[start..end].to_vec());
                        }
                    }
                }
                
                metadata.insert("elements".to_string(), num_elements.to_string());
            },
            "int32" | "i32" => {
                let num_elements = raw_data.len() / 4;
                let mut int_data = vec![0i32; num_elements];
                
                for i in 0..num_elements {
                    let start = i * 4;
                    if start + 4 <= raw_data.len() {
                        let bytes = [
                            raw_data[start], raw_data[start + 1], raw_data[start + 2], raw_data[start + 3]
                        ];
                        if is_little_endian {
                            int_data[i] = i32::from_le_bytes(bytes);
                        } else {
                            int_data[i] = i32::from_be_bytes(bytes);
                        }
                    }
                }
                
                // 转换为f32格式
                let float_data: Vec<f32> = int_data.iter().map(|&x| x as f32).collect();
                
                if shape.is_empty() {
                    features.push(float_data);
                } else {
                    let first_dim = shape[0];
                    let remaining_size: usize = shape.iter().skip(1).product();
                    
                    for i in 0..first_dim {
                        let start = i * remaining_size;
                        let end = std::cmp::min(start + remaining_size, float_data.len());
                        if start < float_data.len() {
                            features.push(float_data[start..end].to_vec());
                        }
                    }
                }
                
                metadata.insert("elements".to_string(), num_elements.to_string());
            },
            _ => {
                return Err(crate::error::Error::not_implemented(&format!(
                    "不支持的数据类型: {}。支持的类型: float32, float64, int32",
                    dtype
                )));
            }
        }
        
        let mut batch = DataBatch::new("tensor_data", 0, features.len());
        // 将features转换为records格式
        for feature_vec in features {
            let mut record = HashMap::new();
            for (i, value) in feature_vec.iter().enumerate() {
                record.insert(format!("feature_{}", i), DataValue::Float(*value as f64));
            }
            batch.records.push(record);
        }
        
        // 添加元数据
        for (key, value) in metadata {
            batch.add_metadata(&key, &value);
        }
        
        Ok(batch)
    }
}

#[async_trait]
impl DataLoader for CommonDataLoader {
    async fn load(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat) -> Result<DataBatch> {
        match source {
            DataSource::File(path) => self.load_from_file(path, format).await,
            DataSource::Database(_) => self.load_from_database(source).await,
            DataSource::Stream(url) => self.load_from_stream(url).await,
            DataSource::Memory(data) => {
                // 处理内存数据源
                let mut batch = DataBatch::new("memory_data", 0, data.len());
                // 根据数据类型构建DataBatch
                for (i, byte) in data.iter().enumerate() {
                    let mut record = HashMap::new();
                    record.insert(format!("byte_{}", i), crate::data::DataValue::Integer(*byte as i64));
                    batch.records.push(record);
                }
                Ok(batch)
            },
            DataSource::Custom(name, _params) => {
                Err(Error::not_implemented(format!("不支持的自定义数据源类型: {}", name)))
            }
        }
    }
    
    async fn get_schema(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat) -> Result<DataSchema> {
        match source {
            DataSource::File(path) => {
                // 根据文件格式推断Schema
                match format {
                    crate::data::loader::types::DataFormat::Csv { .. } => {
                        // CSV Schema推断
                        let mut schema = DataSchema::new("csv_schema", "1.0");
                        let field_def = FieldDefinition {
                            name: "default_field".to_string(),
                            field_type: FieldType::Text,
                            data_type: None,
                            required: false,
                            nullable: true,
                            primary_key: false,
                            foreign_key: None,
                            description: None,
                            default_value: None,
                            constraints: Some(FieldConstraints {
                                min_value: None,
                                max_value: None,
                                min_length: None,
                                max_length: None,
                                pattern: None,
                                allowed_values: None,
                                unique: false,
                            }),
                            metadata: HashMap::new(),
                        };
                        schema.add_field(field_def)?;
                        Ok(schema)
                    },
                    crate::data::loader::types::DataFormat::Json { .. } => {
                        // JSON Schema推断
                        let mut schema = DataSchema::new("json_schema", "1.0");
                        let field_def = FieldDefinition {
                            name: "json_field".to_string(),
                            field_type: FieldType::Text,
                            data_type: None,
                            required: false,
                            nullable: true,
                            primary_key: false,
                            foreign_key: None,
                            description: None,
                            default_value: None,
                            constraints: Some(FieldConstraints {
                                min_value: None,
                                max_value: None,
                                min_length: None,
                                max_length: None,
                                pattern: None,
                                allowed_values: None,
                                unique: false,
                            }),
                            metadata: HashMap::new(),
                        };
                        schema.add_field(field_def)?;
                        Ok(schema)
                    },
                    _ => {
                        // 其他格式的默认Schema
                        let mut schema = DataSchema::new("default_schema", "1.0");
                        let field_def = FieldDefinition {
                            name: "data_field".to_string(),
                            field_type: FieldType::Text,
                            data_type: None,
                            required: false,
                            nullable: true,
                            primary_key: false,
                            foreign_key: None,
                            description: None,
                            default_value: None,
                            constraints: Some(FieldConstraints {
                                min_value: None,
                                max_value: None,
                                min_length: None,
                                max_length: None,
                                pattern: None,
                                allowed_values: None,
                                unique: false,
                            }),
                            metadata: HashMap::new(),
                        };
                        schema.add_field(field_def)?;
                        Ok(schema)
                    }
                }
            },
            DataSource::Database(_) => {
                // 数据库Schema推断
                let mut schema = DataSchema::new("database_schema", "1.0");
                let field_def = FieldDefinition {
                    name: "db_field".to_string(),
                    field_type: FieldType::Text,
                    data_type: None,
                    required: false,
                    nullable: true,
                    primary_key: false,
                    foreign_key: None,
                    description: None,
                    default_value: None,
                    constraints: Some(FieldConstraints {
                        min_value: None,
                        max_value: None,
                        min_length: None,
                        max_length: None,
                        pattern: None,
                        allowed_values: None,
                        unique: false,
                    }),
                    metadata: HashMap::new(),
                };
                schema.add_field(field_def)?;
                Ok(schema)
            },
            DataSource::Stream(_) => {
                // 流数据Schema推断
                let mut schema = DataSchema::new("stream_schema", "1.0");
                let field_def = FieldDefinition {
                    name: "stream_field".to_string(),
                    field_type: FieldType::Text,
                    data_type: None,
                    required: false,
                    nullable: true,
                    primary_key: false,
                    foreign_key: None,
                    description: None,
                    default_value: None,
                    constraints: Some(FieldConstraints {
                        min_value: None,
                        max_value: None,
                        min_length: None,
                        max_length: None,
                        pattern: None,
                        allowed_values: None,
                        unique: false,
                    }),
                    metadata: HashMap::new(),
                };
                schema.add_field(field_def)?;
                Ok(schema)
            },
            DataSource::Memory(_) => {
                // 内存数据Schema推断
                let mut schema = DataSchema::new("memory_schema", "1.0");
                let field_def = FieldDefinition {
                    name: "memory_field".to_string(),
                    field_type: FieldType::Numeric,
                    data_type: None,
                    required: false,
                    nullable: true,
                    primary_key: false,
                    foreign_key: None,
                    description: None,
                    default_value: None,
                    constraints: Some(FieldConstraints {
                        min_value: None,
                        max_value: None,
                        min_length: None,
                        max_length: None,
                        pattern: None,
                        allowed_values: None,
                        unique: false,
                    }),
                    metadata: HashMap::new(),
                };
                schema.add_field(field_def)?;
                Ok(schema)
            },
            DataSource::Custom(name, _params) => {
                // 自定义数据源Schema推断
                let mut schema = DataSchema::new("custom_schema", "1.0");
                let field_def = FieldDefinition {
                    name: format!("{}_field", name),
                    field_type: FieldType::Text,
                    data_type: None,
                    required: false,
                    nullable: true,
                    primary_key: false,
                    foreign_key: None,
                    description: None,
                    default_value: None,
                    constraints: Some(FieldConstraints {
                        min_value: None,
                        max_value: None,
                        min_length: None,
                        max_length: None,
                        pattern: None,
                        allowed_values: None,
                        unique: false,
                    }),
                    metadata: HashMap::new(),
                };
                schema.add_field(field_def)?;
                Ok(schema)
            }
        }
    }
    
    fn name(&self) -> &'static str {
        "CommonDataLoader"
    }
    
    // 实现缺失的trait方法
    async fn load_batch(&self, source: &DataSource, format: &crate::data::loader::types::DataFormat, batch_size: usize, offset: usize) -> Result<DataBatch> {
        // 加载完整数据后进行批次分割
        let full_batch = self.load(source, format).await?;
        
        // 计算实际的开始和结束位置
        let total_size = full_batch.batch_size();
        let start = offset;
        let end = std::cmp::min(start + batch_size, total_size);
        
        if start >= total_size {
            // 超出范围，返回空批次
            return Ok(DataBatch::default());
        }
        
        // 切割批次
        full_batch.slice(start, end)
    }
    
    fn supports_format(&self, format: &crate::data::loader::types::DataFormat) -> bool {
        match format {
            crate::data::loader::types::DataFormat::Csv { .. } => true,
            crate::data::loader::types::DataFormat::Json { .. } => true,
            crate::data::loader::types::DataFormat::Parquet { .. } => true,
            crate::data::loader::types::DataFormat::Avro { .. } => true,
            crate::data::loader::types::DataFormat::Excel { .. } => true,
            crate::data::loader::types::DataFormat::Text { .. } => true,
            crate::data::loader::types::DataFormat::CustomText(_) => true,
            crate::data::loader::types::DataFormat::CustomBinary(_) => false, // 交给专门的二进制加载器处理
        }
    }
    
    fn config(&self) -> &crate::data::loader::LoaderConfig {
        // 返回一个默认配置的引用
        static DEFAULT_CONFIG: std::sync::OnceLock<crate::data::loader::LoaderConfig> = std::sync::OnceLock::new();
        DEFAULT_CONFIG.get_or_init(|| crate::data::loader::LoaderConfig::default())
    }
    
    fn set_config(&mut self, config: crate::data::loader::LoaderConfig) {
        // 从LoaderConfig中提取相关设置更新内部DataConfig
        if let Some(batch_size) = config.batch_size {
            self.config.batch_size = batch_size;
        }
        
        if let Some(format) = config.format {
            // 将loader::types::DataFormat转换为data::DataFormat（生产级实现）
            self.config.format = match format {
                crate::data::loader::types::DataFormat::Csv { .. } => crate::data::types::DataFormat::CSV,
                crate::data::loader::types::DataFormat::Json { .. } => crate::data::types::DataFormat::JSON,
                crate::data::loader::types::DataFormat::Parquet { .. } => crate::data::types::DataFormat::Parquet,
                crate::data::loader::types::DataFormat::Avro { .. } => crate::data::types::DataFormat::Avro,
                crate::data::loader::types::DataFormat::Excel { .. } => {
                    // Excel 格式映射为 JSON（生产级实现）
                    // 注意：Excel 文件可以包含多个工作表，映射为 JSON 格式便于处理
                    // 如果需要保留 Excel 特定功能，可以扩展 DataFormat 枚举添加 Excel 类型
                    crate::data::types::DataFormat::JSON
                }
                crate::data::loader::types::DataFormat::Text { .. } => {
                    // 文本格式映射为 CSV（生产级实现）
                    // 注意：文本文件通常使用 CSV 格式处理，这是合理的设计选择
                    crate::data::types::DataFormat::CSV
                }
                crate::data::loader::types::DataFormat::CustomText(fmt) => crate::data::types::DataFormat::CustomText(fmt),
                crate::data::loader::types::DataFormat::CustomBinary(_) => crate::data::types::DataFormat::Binary,
            };
        }
        
        // LoaderConfig.validate 是 bool，而 DataConfig.validate 是 Option<bool>
        // 这里显式包装为 Some，以避免类型不匹配，同时保留“未设置”语义的可能性
        self.config.validate = Some(config.validate);
        
        // 从选项中提取其他设置（预留扩展位）
        if let Some(timeout_str) = config.options.get("timeout") {
            if let Ok(_timeout) = timeout_str.parse::<u64>() {
                // timeout 设置可以存储在 DataConfig 的额外参数中，当前版本先保留解析逻辑
                // 将来可扩展 DataConfig 来支持更多运行时配置
            }
        }
    }
}

// 判断URL是否为数据库连接字符串
pub fn is_database_url(url: &str) -> bool {
    url.starts_with("mysql://") || 
    url.starts_with("postgresql://") || 
    url.starts_with("postgres://") || 
    url.starts_with("sqlite:") || 
    url.starts_with("mongodb://")
}

// 判断URL是否为流数据源
pub fn is_stream_url(url: &str) -> bool {
    url.starts_with("kafka://") || 
    url.starts_with("mqtt://") || 
    url.starts_with("amqp://") || 
    url.starts_with("redis://") || 
    url.starts_with("ws://") || 
    url.starts_with("wss://")
}

// 数据格式处理接口
pub trait FormatProcessor: Send + Sync {
    fn process(&self, data: &[u8], format: &DataFormat) -> Result<DataBatch>;
    fn get_supported_formats(&self) -> Vec<DataFormat>;
    fn is_format_supported(&self, format: &DataFormat) -> bool {
        self.get_supported_formats().contains(format)
    }
}

// 数据源接口
pub trait DataSourceConnector: Send + Sync {
    fn connect(&self) -> Result<()>;
    fn disconnect(&self) -> Result<()>;
    fn is_connected(&self) -> bool;
    fn read(&self, query: &str) -> Result<DataBatch>;
    fn write(&self, data: &DataBatch, destination: &str) -> Result<()>;
}

/// 一个默认的空实现，用于尚未配置真实数据源连接池的场景
pub struct NullDataSourceConnector;

impl NullDataSourceConnector {
    pub fn new() -> Self {
        Self
    }
}

impl DataSourceConnector for NullDataSourceConnector {
    fn connect(&self) -> Result<()> {
        Ok(())
    }

    fn disconnect(&self) -> Result<()> {
        Ok(())
    }

    fn is_connected(&self) -> bool {
        false
    }

    fn read(&self, _query: &str) -> Result<DataBatch> {
        Err(Error::NotSupported("NullDataSourceConnector 不支持读取操作".to_string()))
    }

    fn write(&self, _data: &DataBatch, _destination: &str) -> Result<()> {
        Err(Error::NotSupported("NullDataSourceConnector 不支持写入操作".to_string()))
    }
}

// 缓存接口
pub trait DataCache: Send + Sync {
    fn get(&self, key: &str) -> Result<Option<DataBatch>>;
    fn put(&self, key: &str, value: DataBatch) -> Result<()>;
    fn remove(&self, key: &str) -> Result<()>;
    fn clear(&self) -> Result<()>;
    fn contains(&self, key: &str) -> bool;
}

// 实现内存缓存
pub struct MemoryCache {
    cache: HashMap<String, DataBatch>,
    max_size: usize,
}

impl MemoryCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_size),
            max_size,
        }
    }
}

impl DataCache for MemoryCache {
    fn get(&self, key: &str) -> Result<Option<DataBatch>> {
        Ok(self.cache.get(key).cloned())
    }
    
    fn put(&self, key: &str, value: DataBatch) -> Result<()> {
        if self.cache.len() >= self.max_size {
            return Err(Error::resource_exhausted("缓存已满"));
        }
        
        let mut cache = self.cache.clone();
        cache.insert(key.to_string(), value);
        Ok(())
    }
    
    fn remove(&self, key: &str) -> Result<()> {
        let mut cache = self.cache.clone();
        cache.remove(key);
        Ok(())
    }
    
    fn clear(&self) -> Result<()> {
        let mut cache = self.cache.clone();
        cache.clear();
        Ok(())
    }
    
    fn contains(&self, key: &str) -> bool {
        self.cache.contains_key(key)
    }
}

// 添加数据加载器构建器
pub struct DataLoaderBuilder {
    config: DataConfig,
    db_manager: Option<Arc<DatabaseManager>>,
    format_processors: HashMap<String, Box<dyn FormatProcessor>>,
    cache: Option<Box<dyn DataCache>>,
    error_handler: Option<Box<dyn ErrorHandler>>,
}

impl DataLoaderBuilder {
    pub fn new(config: DataConfig) -> Self {
        Self {
            config,
            db_manager: None,
            format_processors: HashMap::new(),
            cache: None,
            error_handler: None,
        }
    }
    
    pub fn with_database_manager(mut self, db_manager: Arc<DatabaseManager>) -> Self {
        self.db_manager = Some(db_manager);
        self
    }
    
    pub fn with_format_processor(mut self, name: &str, processor: Box<dyn FormatProcessor>) -> Self {
        self.format_processors.insert(name.to_string(), processor);
        self
    }
    
    pub fn with_cache(mut self, cache: Box<dyn DataCache>) -> Self {
        self.cache = Some(cache);
        self
    }
    
    pub fn with_error_handler(mut self, error_handler: Box<dyn ErrorHandler>) -> Self {
        self.error_handler = Some(error_handler);
        self
    }
    
    pub fn build(self) -> Result<CommonDataLoader> {
        let loader = CommonDataLoader {
            config: self.config,
            db_manager: self.db_manager,
            cache: self.cache.map_or_else(|| HashMap::new(), |_| HashMap::new()),
        };
        
        Ok(loader)
    }
}

// 添加错误处理接口
pub trait ErrorHandler: Send + Sync {
    fn handle_error(&self, source: &str, error: &Error) -> Result<()>;
    fn log_error(&self, source: &str, message: &str);
}

// 添加基本错误处理实现
pub struct DefaultErrorHandler;

impl ErrorHandler for DefaultErrorHandler {
    fn handle_error(&self, source: &str, error: &Error) -> Result<()> {
        // 使用error日志功能记录错误
        error!("处理错误: 源={}, 错误={}", source, error);
        
        // 根据错误类型执行不同提示日志，但始终返回一个新的错误实例
        match error {
            Error::Io(_) => {
                error!("IO错误, 可能需要检查文件路径、权限或磁盘空间");
            },
            Error::Serialization(_) | Error::Deserialization(_) | Error::DataFormat(_) => {
                error!("序列化/反序列化或数据格式错误, 请检查数据源格式是否与配置匹配");
            },
            Error::Validation(_) => {
                error!("数据验证错误, 请检查数据质量和约束条件");
            },
            _ => {
                // 其他错误类型在上面的日志中已经记录，这里不再重复分类
            }
        }

        // 不直接克隆传入的 &Error，而是包装为带上下文的 Data 错误
        Err(Error::Data(format!("数据加载错误(源={}): {}", source, error)))
    }
    
    fn log_error(&self, source: &str, message: &str) {
        let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
        error!("[{}] 错误源: {}, 消息: {}", timestamp, source, message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_cache() {
        let cache = MemoryCache::new(5);
        
        // 创建测试数据
        let mut batch = DataBatch::new("test_data", 0, 1);
        // 添加测试记录
        let mut record = HashMap::new();
        record.insert("test".to_string(), crate::data::DataValue::Integer(123));
        batch.records.push(record);
        
        // 测试添加和获取
        assert!(cache.put("key1", batch.clone()).is_ok());
        let result = cache.get("key1").unwrap();
        assert!(result.is_some());
        
        // 测试存在检查
        assert!(cache.contains("key1"));
        assert!(!cache.contains("key2"));
        
        // 测试删除
        assert!(cache.remove("key1").is_ok());
        assert!(!cache.contains("key1"));
        
        // 测试清空
        assert!(cache.put("key3", batch.clone()).is_ok());
        assert!(cache.clear().is_ok());
        assert!(!cache.contains("key3"));
    }
    
    #[test]
    fn test_data_loader_builder() {
        let config = DataConfig::default();
        let builder = DataLoaderBuilder::new(config);
        
        // 验证构建器能成功创建
        let loader = builder.build();
        assert!(loader.is_ok());
    }
    
    #[test]
    fn test_format_processor_trait() {
        struct TestProcessor;
        
        impl FormatProcessor for TestProcessor {
            fn process(&self, data: &[u8], format: &DataFormat) -> Result<DataBatch> {
                // 简单测试实现
                let mut batch = DataBatch::new("test_process", 0, 1);
                // 添加测试记录
                let mut record = HashMap::new();
                record.insert("test".to_string(), crate::data::DataValue::Integer(1));
                batch.records.push(record);
                Ok(batch)
            }
            
            fn get_supported_formats(&self) -> Vec<DataFormat> {
                vec![DataFormat::csv(), DataFormat::json()]
            }
        }
        
        let processor = TestProcessor;
        
        // 测试支持的格式
        assert!(processor.is_format_supported(&DataFormat::csv()));
        assert!(processor.is_format_supported(&DataFormat::json()));
        assert!(!processor.is_format_supported(&DataFormat::parquet()));
        
        // 测试处理
        let data = b"test data";
        let result = processor.process(data, &DataFormat::csv());
        assert!(result.is_ok());
    }
} 