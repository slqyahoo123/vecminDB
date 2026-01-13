use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::sync::Arc;

use apache_avro::{Reader, Schema as AvroSchema};
use apache_avro::types::Value as AvroValue;
use crate::error::{Error, Result};
use crate::data::{DataSchema, DataField, FieldType, FieldSource, SchemaMetadata};
use log::{debug, info, warn};

/// Avro文件处理器
pub struct AvroProcessor {
    path: PathBuf,
}

impl AvroProcessor {
    /// 创建新的Avro处理器
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
        }
    }
    
    /// 从Avro文件中读取schema
    pub fn read_schema(&self) -> Result<apache_avro::Schema> {
        let file = File::open(&self.path)
            .map_err(|e| Error::io_error(format!("无法打开Avro文件: {}", e)))?;
        
        let reader = BufReader::new(file);
        let reader = apache_avro::Reader::new(reader)
            .map_err(|e| Error::invalid_data(format!("无法读取Avro文件: {}", e)))?;
            
        Ok(reader.writer_schema().clone())
    }
    
    /// 获取Avro文件行数
    pub fn get_row_count(&self) -> Result<usize> {
        let file = File::open(&self.path)
            .map_err(|e| Error::io_error(format!("无法打开Avro文件: {}", e)))?;
            
        let reader = BufReader::new(file);
        let mut reader = apache_avro::Reader::new(reader)
            .map_err(|e| Error::invalid_data(format!("无法读取Avro文件: {}", e)))?;
            
        let mut count = 0;
        while let Some(_) = reader.next() {
            count += 1;
        }
        
        Ok(count)
    }
    
    /// 从Avro文件中读取若干行数据
    /// 
    /// 参数:
    /// - `start_idx`: 开始的行索引，0表示从第一行开始
    /// - `limit`: 可选的限制行数，None表示不限制
    pub fn read_rows(&self, start_idx: usize, limit: Option<usize>) -> Result<Vec<HashMap<String, String>>> {
        let file = File::open(&self.path)
            .map_err(|e| Error::io_error(format!("无法打开Avro文件: {}", e)))?;
            
        let reader = BufReader::new(file);
        let mut reader = apache_avro::Reader::new(reader)
            .map_err(|e| Error::invalid_data(format!("无法读取Avro文件: {}", e)))?;
            
        let schema = reader.writer_schema();
        let field_names = extract_field_names(&schema);
        
        let mut rows = Vec::new();
        let mut row_count = 0;
        
        // 跳过开始索引之前的行
        for _ in 0..start_idx {
            if reader.next().is_none() {
                break; // 没有更多行了
            }
        }
        
        // 读取所需行数
        while let Some(result) = reader.next() {
            if let Some(max) = limit {
                if row_count >= max {
                    break;
                }
            }
            
            let record = result
                .map_err(|e| Error::invalid_data(format!("无法读取Avro记录: {}", e)))?;
                
            let mut row_data = HashMap::new();
            
            if let apache_avro::types::Value::Record(fields) = record {
                for (field_name, field_value) in fields {
                    let value = avro_value_to_string(&field_value);
                    row_data.insert(field_name, value);
                }
            }
            
            rows.push(row_data);
            row_count += 1;
        }
        
        Ok(rows)
    }
    
    /// 估计Avro文件的总行数
    pub fn estimate_row_count(&self) -> Result<usize> {
        // 对于Avro文件，需要完整扫描才能获取准确的行数
        // 为了性能考虑，可以使用文件大小来估算
        
        let metadata = std::fs::metadata(&self.path)
            .map_err(|e| Error::io_error(format!("无法获取文件元数据: {}", e)))?;
            
        let file_size = metadata.len() as usize;
        
        // 尝试读取前100行来估算平均行大小
        let sample_rows = self.read_rows(0, Some(100))?;
        
        if sample_rows.is_empty() {
            return Ok(0);
        }
        
        // 估算每行大小
        let avg_row_size = file_size / sample_rows.len();
        
        // 估算总行数
        let estimated_rows = file_size / avg_row_size;
        
        Ok(estimated_rows)
    }
}

/// 从Avro模式中提取字段名称列表
fn extract_field_names(schema: &apache_avro::Schema) -> Vec<String> {
    let mut field_names = Vec::new();
    
    match schema {
        apache_avro::Schema::Record { fields, .. } => {
            for field in fields {
                field_names.push(field.name.clone());
            }
        },
        _ => {}
    }
    
    field_names
}

/// 将Avro值转换为字符串
fn avro_value_to_string(value: &apache_avro::types::Value) -> String {
    match value {
        apache_avro::types::Value::Null => "null".to_string(),
        apache_avro::types::Value::Boolean(b) => b.to_string(),
        apache_avro::types::Value::Int(i) => i.to_string(),
        apache_avro::types::Value::Long(l) => l.to_string(),
        apache_avro::types::Value::Float(f) => f.to_string(),
        apache_avro::types::Value::Double(d) => d.to_string(),
        apache_avro::types::Value::Bytes(b) => format!("{:?}", b),
        apache_avro::types::Value::String(s) => s.clone(),
        apache_avro::types::Value::Fixed(_, b) => format!("{:?}", b),
        apache_avro::types::Value::Enum(_, s) => s.clone(),
        apache_avro::types::Value::Union(box_value) => avro_value_to_string(box_value),
        apache_avro::types::Value::Array(values) => {
            let items: Vec<String> = values.iter()
                .map(|v| avro_value_to_string(v))
                .collect();
            format!("[{}]", items.join(", "))
        },
        apache_avro::types::Value::Map(map) => {
            let items: Vec<String> = map.iter()
                .map(|(k, v)| format!("\"{}\": {}", k, avro_value_to_string(v)))
                .collect();
            format!("{{{}}}", items.join(", "))
        },
        apache_avro::types::Value::Record(fields) => {
            let items: Vec<String> = fields.iter()
                .map(|(k, v)| format!("\"{}\": {}", k, avro_value_to_string(v)))
                .collect();
            format!("{{{}}}", items.join(", "))
        },
        _ => "unknown".to_string(),
    }
}

/// 将Avro字段类型转换为系统字段类型
fn convert_avro_type_to_field_type(avro_type: &apache_avro::Schema) -> FieldType {
    match avro_type {
        apache_avro::Schema::Null => FieldType::Null,
        apache_avro::Schema::Boolean => FieldType::Boolean,
        apache_avro::Schema::Int => FieldType::Integer,
        apache_avro::Schema::Long => FieldType::Integer,
        apache_avro::Schema::Float => FieldType::Float,
        apache_avro::Schema::Double => FieldType::Float,
        apache_avro::Schema::Bytes => FieldType::Binary,
        apache_avro::Schema::String => FieldType::String,
        apache_avro::Schema::Array(_) => FieldType::Array,
        apache_avro::Schema::Map(_) => FieldType::Object,
        apache_avro::Schema::Record { .. } => FieldType::Object,
        apache_avro::Schema::Enum { .. } => FieldType::String,
        apache_avro::Schema::Union(variants) => {
            // 如果Union包含Null，则取第一个非Null类型
            if variants.contains(&apache_avro::Schema::Null) {
                for variant in variants {
                    if !matches!(variant, apache_avro::Schema::Null) {
                        return convert_avro_type_to_field_type(variant);
                    }
                }
            }
            // 默认为String类型
            FieldType::String
        },
        apache_avro::Schema::Fixed { .. } => FieldType::Binary,
        _ => FieldType::String,
    }
}

/// 从Avro schema构建系统数据字段
fn build_fields_from_avro_schema(schema: &apache_avro::Schema) -> Vec<DataField> {
    let mut fields = Vec::new();
    
    if let apache_avro::Schema::Record { fields: schema_fields, .. } = schema {
        for field in schema_fields {
            let field_type = convert_avro_type_to_field_type(&field.schema);
            
            // 检查字段是否为可空类型
            let nullable = match &field.schema {
                apache_avro::Schema::Union(variants) => variants.contains(&apache_avro::Schema::Null),
                _ => false,
            };
            
            fields.push(DataField {
                name: field.name.clone(),
                field_type,
                required: !nullable,
                source: FieldSource::File,
                description: None,
                default_value: None,
            });
        }
    }
    
    fields
}

/// 推断Avro文件模式
pub fn infer_avro_schema_impl(path: &Path) -> Result<DataSchema> {
    info!("从Avro文件推断数据模式: {}", path.display());
    
    let processor = AvroProcessor::new(path);
    
    // 读取Avro文件schema
    let avro_schema = processor.read_schema()?;
    
    // 从Avro schema构建系统数据字段
    let fields = build_fields_from_avro_schema(&avro_schema);
    
    // 估计行数
    let row_count = processor.estimate_row_count()?;
    
    // 创建模式元数据
    let metadata = SchemaMetadata {
        source: "file".to_string(),
        format: "avro".to_string(),
        path: path.to_str().unwrap_or("").to_string(),
        record_count: Some(row_count),
        created_at: chrono::Utc::now(),
        version: "1.0".to_string(),
        ..Default::default()
    };
    
    Ok(DataSchema {
        fields,
        metadata: Some(metadata),
    })
} 