use crate::error::{Error, Result};
use crate::data::{DataSchema};
// 下面这个导入当前未使用，但保留它是为了将来支持数据类型转换功能
// use crate::data::{FieldType as DataFieldType};
use std::path::Path;
use std::path::PathBuf;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
// info在将来的日志功能中会使用
use log::{debug, warn};
use std::collections::HashMap;
// from_str当前未使用，但将来会用于字符串解析功能
use serde_json::{Value, from_reader};
use chrono::{NaiveDate, NaiveDateTime};
use regex::Regex;
use crate::data::loader::file::Record;
use crate::data::DataValue;
use crate::data::loader::file::FileProcessor;
// 下面这个导入在当前实现中未使用，但保留它用于将来的功能扩展
// use super::FileProcessor as SuperFileProcessor;
// 导入csv.rs中的Field和Schema
use crate::data::loader::file::csv::{Field, FieldType, Schema};
// 下面这个导入当前未使用，但将来会用于结果处理
// use anyhow::{Result as AnyhowResult};
// 下面这个导入当前未使用，但将来会用于线程安全处理
// use std::sync::Arc;
use std::io::{BufWriter};
use crate::data::record::DataField;
use crate::data::loader::file::csv::ValueType;
use crate::data::DataBatch;
use base64::{engine::general_purpose, Engine as _};
use crate::data::schema::{FieldType as OuterFieldType};
use crate::data::schema::schema::FieldType as InnerFieldType;

/// 将内部schema定义的FieldType映射为对外暴露的FieldType（兼容旧FieldType）
fn inner_to_outer_field_type(inner: &InnerFieldType) -> OuterFieldType {
    match inner {
        InnerFieldType::Numeric => OuterFieldType::Numeric,
        InnerFieldType::Categorical => OuterFieldType::Categorical,
        InnerFieldType::Text => OuterFieldType::Text,
        InnerFieldType::Image => OuterFieldType::Image,
        InnerFieldType::Audio => OuterFieldType::Audio,
        InnerFieldType::Video => OuterFieldType::Video,
        InnerFieldType::DateTime => OuterFieldType::DateTime,
        InnerFieldType::Boolean => OuterFieldType::Boolean,
        // 内部数组带元素类型，外部只保留元素类型信息
        InnerFieldType::Array(elem) => {
            OuterFieldType::Array(Box::new(inner_to_outer_field_type(elem)))
        }
        // 内部对象带字段类型映射，外部只有一个 Object 标记
        InnerFieldType::Object(_) => OuterFieldType::Object,
        // 其它一律映射为自定义类型，保持名称
        InnerFieldType::Custom(s) => OuterFieldType::Custom(s.clone()),
    }
}

/// 将对外暴露的FieldType映射为内部schema定义的FieldType
fn outer_to_inner_field_type(outer: &OuterFieldType) -> InnerFieldType {
    match outer {
        // 传统/数值型统一映射为Numeric
        OuterFieldType::Int
        | OuterFieldType::Integer
        | OuterFieldType::Float
        | OuterFieldType::Numeric => InnerFieldType::Numeric,
        // 文本/字符串
        OuterFieldType::String | OuterFieldType::Text => InnerFieldType::Text,
        // 布尔
        OuterFieldType::Boolean => InnerFieldType::Boolean,
        // 日期时间相关
        OuterFieldType::Date | OuterFieldType::Time | OuterFieldType::DateTime => {
            InnerFieldType::DateTime
        }
        // 结构化类型：外部没有字段细节，这里先映射为空对象
        OuterFieldType::Object => InnerFieldType::Object(HashMap::new()),
        OuterFieldType::Array(elem) => {
            InnerFieldType::Array(Box::new(outer_to_inner_field_type(elem)))
        }
        // 分类 / 多媒体 / 时间序列等直接按语义映射
        OuterFieldType::Categorical => InnerFieldType::Categorical,
        OuterFieldType::Image => InnerFieldType::Image,
        OuterFieldType::Audio => InnerFieldType::Audio,
        OuterFieldType::Video => InnerFieldType::Video,
        OuterFieldType::TimeSeries => InnerFieldType::Custom("time_series".to_string()),
        // 二进制/嵌入向量统一映射为自定义类型
        OuterFieldType::Binary => InnerFieldType::Custom("binary".to_string()),
        OuterFieldType::Embedding(dim) => {
            InnerFieldType::Custom(format!("embedding_{}", dim))
        }
        // 其他自定义类型保持名称
        OuterFieldType::Json => InnerFieldType::Custom("json".to_string()),
        OuterFieldType::Custom(s) => InnerFieldType::Custom(s.clone()),
    }
}

/// JSON格式类型
#[derive(Debug, PartialEq, Eq)]
pub enum JsonFormat {
    /// 单个JSON对象
    Object,
    /// JSON数组
    Array,
    /// JSON对象数组
    ArrayOfObjects,
    /// 行分隔的JSON (JSONL)
    JSONL,
    /// 未知格式
    Unknown,
}

impl JsonFormat {
    /// 返回格式的字符串表示
    pub fn as_str(&self) -> &'static str {
        match self {
            JsonFormat::Object => "Object",
            JsonFormat::Array => "Array",
            JsonFormat::ArrayOfObjects => "ArrayOfObjects",
            JsonFormat::JSONL => "JSONL",
            JsonFormat::Unknown => "Unknown",
        }
    }
}

/// JSON处理器
pub struct JSONProcessor {
    /// 文件路径
    file_path: PathBuf,
    /// 当前读取位置
    current_row: usize,
    /// 是否为数组JSON
    is_array: Option<bool>,
    /// 是否为对象数组
    is_array_of_objects: Option<bool>,
    schema: Option<Schema>,
    data: Option<Vec<Value>>,
    total_rows: Option<usize>,
    current_position: usize,
    options: HashMap<String, String>,
}

impl JSONProcessor {
    /// 创建新的JSON处理器
    pub fn new(path: &str, options: Option<HashMap<String, String>>) -> Result<Self> {
        let file_path = PathBuf::from(path);
        // 验证文件是否存在
        if !file_path.exists() {
            return Err(Error::not_found(file_path.to_string_lossy().to_string()));
        }
        
        let processor_options = if let Some(opts) = options {
            opts
        } else {
            HashMap::new()
        };

        Ok(Self {
            file_path,
            current_row: 0,
            is_array: None,
            is_array_of_objects: None,
            schema: None,
            data: None,
            total_rows: None,
            current_position: 0,
            options: processor_options,
        })
    }

    /// 检测JSON文件格式
    fn detect_json_format(&self) -> Result<(bool, bool)> {
        let file = File::open(&self.file_path)?;
        let reader = BufReader::new(file);

        // 读取JSON数据
        let value: Value = from_reader(reader)?;

        // 判断是否为数组
        let is_array = value.is_array();
        
        // 判断是否为对象数组
        let is_array_of_objects = if is_array {
            let array = value.as_array().unwrap();
            !array.is_empty() && array.iter().all(|item| item.is_object())
        } else {
            false
        };

        Ok((is_array, is_array_of_objects))
    }

    /// 读取JSON对象
    fn read_json_object(&self) -> Result<Value> {
        let file = File::open(&self.file_path)?;
        let reader = BufReader::new(file);
        let value: Value = from_reader(reader)?;
        Ok(value)
    }

    /// 读取JSON数组
    fn read_json_array(&self, start: usize, limit: Option<usize>) -> Result<Vec<Value>> {
        let file = File::open(&self.file_path)?;
        let reader = BufReader::new(file);
        let value: Value = from_reader(reader)?;
        
        if !value.is_array() {
            return Err(Error::invalid_data("JSON不是数组格式"));
        }
        
        let array = value.as_array().unwrap();
        let end = start + limit.unwrap_or(array.len());
        let end = end.min(array.len());
        
        Ok(array[start..end].to_vec())
    }

    /// 将JSON值转换为字段类型
    fn json_value_to_field_type(value: &Value) -> FieldType {
        match value {
            Value::Null => FieldType::Unknown,
            Value::Bool(_) => FieldType::Boolean,
            Value::Number(_) => FieldType::Number,
            Value::String(s) => {
                // 尝试解析为日期或时间戳
                if Self::looks_like_date(s) || Self::looks_like_timestamp(s) {
                    FieldType::Date
                } else {
                    FieldType::String
                }
            },
            Value::Array(_) => FieldType::Unknown,
            Value::Object(_) => FieldType::Unknown,
        }
    }

    /// 判断字符串是否看起来像日期
    fn looks_like_date(s: &str) -> bool {
        // 简单的日期格式检测
        let date_regex = Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();
        if date_regex.is_match(s) {
            if let Ok(_) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
                return true;
            }
        }
        false
    }

    /// 判断字符串是否看起来像时间戳
    fn looks_like_timestamp(s: &str) -> bool {
        // ISO 8601/RFC 3339格式检测
        let timestamp_regex = Regex::new(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}").unwrap();
        if timestamp_regex.is_match(s) {
            if let Ok(_) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f%:z") {
                return true;
            }
        }
        false
    }

    /// 将JSON值转换为数据值
    fn json_value_to_data_value(value: &Value) -> DataValue {
        match value {
            Value::Null => DataValue::Null,
            Value::Bool(b) => DataValue::Boolean(*b),
            Value::Number(n) => {
                if n.is_i64() {
                    if let Some(i) = n.as_i64() {
                        DataValue::Integer(i)
                    } else {
                        DataValue::Null
                    }
                } else if n.is_u64() {
                    if let Some(u) = n.as_u64() {
                        DataValue::Integer(u as i64)
                    } else {
                        DataValue::Null
                    }
                } else if let Some(f) = n.as_f64() {
                    DataValue::Float(f)
                } else {
                    DataValue::Null
                }
            },
            Value::String(s) => {
                if Self::looks_like_date(s) {
                    DataValue::String(s.clone()) // 日期作为字符串处理
                } else if Self::looks_like_timestamp(s) {
                    DataValue::String(s.clone()) // 时间戳作为字符串处理
                } else {
                    DataValue::String(s.clone())
                }
            },
            Value::Array(a) => {
                let values: Vec<DataValue> = a.iter()
                    .map(|v| Self::json_value_to_data_value(v))
                    .collect();
                DataValue::Array(values)
            },
            Value::Object(o) => {
                let mut map = HashMap::new();
                for (k, v) in o {
                    map.insert(k.clone(), Self::json_value_to_data_value(v));
                }
                DataValue::Object(map)
            },
        }
    }

    /// 加载JSON数据
    fn load_data(&mut self) -> Result<()> {
        let file = File::open(&self.file_path)?;
        let reader = BufReader::new(file);

        // 检查文件是否为JSON数组
        let data: serde_json::Value = serde_json::from_reader(reader)?;
        
        if !data.is_array() {
            // 如果不是数组，尝试将单个对象包装成数组
            if data.is_object() {
                let mut vec = Vec::new();
                vec.push(data);
                self.data = Some(vec);
            } else {
                return Err(Error::invalid_input("JSON data is not an array or object".to_string()));
            }
        } else {
            // 如果是数组，直接使用
            let array = data.as_array().unwrap();
            self.data = Some(array.clone());
        }

        Ok(())
    }

    /// 自动推断JSON数据的模式
    fn infer_schema(&mut self) -> Result<Schema> {
        if self.data.is_none() {
            self.load_data()?;
        }

        let data = self.data.as_ref().unwrap();
        if data.is_empty() {
            return Err(Error::invalid_data("JSON data is empty, cannot infer schema".to_string()));
        }

        // 使用第一个记录推断基本结构
        let first_record = &data[0];
        if !first_record.is_object() {
            return Err(Error::invalid_input("JSON data must contain objects".to_string()));
        }

        let obj = first_record.as_object().unwrap();
        let mut fields = Vec::new();

        // 首先从第一个记录提取所有字段
        for (key, value) in obj {
            let field_type = Self::json_value_to_type(value);
            fields.push(Field::new(key.clone(), field_type));
        }

        // 如果有多条记录，检查前100条或全部记录（取较小值）来完善类型推断
        let sample_size = std::cmp::min(data.len(), 100);
        
        // 对每个字段，通过检查样本中的所有值来完善类型推断
        for record_idx in 1..sample_size {
            let record = &data[record_idx];
            if !record.is_object() {
                continue;
            }
            
            let obj = record.as_object().unwrap();
            
            // 检查每个已知字段
            for field_idx in 0..fields.len() {
                let field = &fields[field_idx];
                let field_name = field.name();
                
                // 如果当前记录包含此字段
                if let Some(value) = obj.get(field_name) {
                    let current_type = &field.field_type;
                    let new_type = Self::json_value_to_type(value);
                    
                    // 如果类型不同，选择更通用的类型
                    if current_type != &new_type && new_type != FieldType::Unknown {
                        let updated_type = Self::resolve_type_conflict(current_type.clone(), new_type);
                        fields[field_idx] = Field::new(field_name.to_string(), updated_type);
                    }
                }
            }
            
            // 检查是否有新字段出现
            for (key, value) in obj {
                if !fields.iter().any(|f| f.name() == key) {
                    let field_type = Self::json_value_to_type(value);
                    fields.push(Field::new(key.clone(), field_type));
                }
            }
        }

        Ok(Schema::new(fields))
    }

    /// 将JSON值转换为类型
    fn json_value_to_type(value: &Value) -> FieldType {
        match value {
            Value::Null => FieldType::Unknown,
            Value::Bool(_) => FieldType::Boolean,
            Value::Number(_) => FieldType::Number,
            Value::String(s) => {
                if Self::looks_like_date(s) || Self::looks_like_timestamp(s) {
                    FieldType::Date
                } else {
                    FieldType::String
                }
            },
            Value::Array(_) => FieldType::Unknown,
            Value::Object(_) => FieldType::Unknown,
        }
    }

    /// 解决类型冲突
    fn resolve_type_conflict(type1: FieldType, type2: FieldType) -> FieldType {
        if type1 == type2 {
            return type1;
        }

        // 如果其中一个类型是Unknown，返回另一个类型
        if type1 == FieldType::Unknown {
            return type2;
        }
        if type2 == FieldType::Unknown {
            return type1;
        }

        // 大部分冲突情况下，降级为字符串类型
        FieldType::String
    }

    /// 将JSON值转换为内部Value类型
    fn json_to_value(json_value: &Value, field_type: ValueType) -> Value {
        match json_value {
            Value::Null => Value::Null,
            Value::Bool(b) => Value::Bool(*b),
            Value::Number(n) => {
                if field_type == ValueType::Integer {
                    if let Some(i) = n.as_i64() {
                        Value::Number(i.into())
                    } else {
                        Value::Number(serde_json::Number::from_f64(n.as_f64().unwrap_or(0.0)).unwrap_or_else(|| serde_json::Number::from(0)))
                    }
                } else {
                    let num = serde_json::Number::from_f64(n.as_f64().unwrap_or(0.0)).unwrap_or_else(|| serde_json::Number::from(0));
                    Value::Number(num)
                }
            },
            Value::String(s) => {
                // 尝试解析日期时间格式，如果失败则保持为字符串
                if field_type == ValueType::DateTime {
                    // 实现日期时间解析逻辑
                    // 日期时间解析（生产级实现）
                    // 尝试多种日期时间格式解析
                    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&s) {
                        Value::String(dt.to_rfc3339())
                    } else if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S") {
                        Value::String(dt.format("%Y-%m-%d %H:%M:%S").to_string())
                    } else if let Ok(dt) = chrono::NaiveDate::parse_from_str(&s, "%Y-%m-%d") {
                        Value::String(dt.format("%Y-%m-%d").to_string())
                    } else {
                        // 如果所有格式都失败，保持为字符串
                        Value::String(s.clone())
                    }
                } else {
                    Value::String(s.clone())
                }
            },
            Value::Array(arr) => {
                Value::Array(arr.clone())
            },
            Value::Object(obj) => {
                Value::Object(obj.clone())
            },
        }
    }

    /// 将记录转换为内部Record类型
    fn json_to_record(&self, json_obj: &serde_json::Map<String, Value>, schema: &Schema) -> Record {
        let mut values = Vec::with_capacity(schema.fields().len());
        
        for field in schema.fields() {
            let field_name = field.name();
            
            if let Some(json_value) = json_obj.get(field_name) {
                values.push(Self::json_value_to_data_value(json_value));
            } else {
                values.push(DataValue::Null);
            }
        }
        
        Record::new(values)
    }
    
    /// 写入JSON文件
    pub fn write_json(path: &str, records: &[Record], schema: &Schema, options: Option<HashMap<String, String>>) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        
        let array = records.iter().map(|record| {
            let mut obj = serde_json::Map::new();
            
            for (i, field) in schema.fields().iter().enumerate() {
                if i < record.values().len() {
                    let value = &record.values()[i];
                    let json_value = Self::data_value_to_json(value);
                    obj.insert(field.name().to_string(), json_value);
                }
            }
            
            serde_json::Value::Object(obj)
        }).collect::<Vec<_>>();
        
        serde_json::to_writer_pretty(writer, &array)?;
        Ok(())
    }
    
    /// 将内部Value类型转换为JSON值（已废弃，使用 data_value_to_json）
    fn value_to_json(value: &Value) -> Value {
        match value {
            Value::Null => Value::Null,
            Value::Bool(b) => Value::Bool(*b),
            Value::Number(n) => Value::Number(n.clone()),
            Value::String(s) => Value::String(s.clone()),
            Value::Array(arr) => Value::Array(arr.clone()),
            Value::Object(obj) => Value::Object(obj.clone()),
        }
    }
    
    /// 将DataValue转换为JSON值
    fn data_value_to_json(value: &DataValue) -> Value {
        match value {
            DataValue::Null => Value::Null,
            DataValue::Boolean(b) => Value::Bool(*b),
            DataValue::Integer(i) => Value::Number(serde_json::Number::from(*i)),
            DataValue::Float(f) => Value::Number(
                serde_json::Number::from_f64(*f)
                    .unwrap_or_else(|| serde_json::Number::from(0))
            ),
            DataValue::Number(n) => Value::Number(
                serde_json::Number::from_f64(*n)
                    .unwrap_or_else(|| serde_json::Number::from(0))
            ),
            DataValue::String(s) => Value::String(s.clone()),
            DataValue::Array(arr) => {
                Value::Array(arr.iter().map(|v| Self::data_value_to_json(v)).collect())
            },
            DataValue::Object(obj) => {
                let mut map = serde_json::Map::new();
                for (k, v) in obj {
                    map.insert(k.clone(), Self::data_value_to_json(v));
                }
                Value::Object(map)
            },
            DataValue::StringArray(arr) => {
                Value::Array(arr.iter().map(|s| Value::String(s.clone())).collect())
            },
            DataValue::Binary(b) => {
                // 将二进制数据编码为base64字符串
                Value::String(general_purpose::STANDARD.encode(b))
            },
            DataValue::DateTime(dt) => Value::String(dt.clone()),
            DataValue::Tensor(_) => {
                // 张量数据序列化为JSON对象
                Value::Object(serde_json::Map::new())
            },
        }
    }

    /// 从指定位置读取JSON数据
    fn read_from_position(&mut self, position: u64, count: usize) -> Result<Vec<Value>> {
        let file = File::open(&self.file_path)?;
        let mut reader = BufReader::new(file);
        
        // 使用SeekFrom跳转到文件的指定位置
        reader.seek(SeekFrom::Start(position))?;
        
        // 读取指定数量的JSON对象
        let mut values = Vec::with_capacity(count);
        let mut buffer = String::new();
        
        for _ in 0..count {
            buffer.clear();
            if reader.read_line(&mut buffer)? == 0 {
                break; // 已经到达文件末尾
            }
            
            if buffer.trim().is_empty() {
                continue;
            }
            
            match serde_json::from_str::<Value>(&buffer) {
                Ok(value) => values.push(value),
                Err(e) => debug!("解析JSON行失败: {}", e),
            }
        }
        
        Ok(values)
    }
}

impl FileProcessor for JSONProcessor {
    fn get_file_path(&self) -> &Path {
        &self.file_path
    }
    
    fn get_schema(&self) -> Result<DataSchema> {
        let csv_schema = if let Some(schema) = &self.schema {
            schema.clone()
        } else {
            self.infer_schema()?
        };
        
        // 将 csv::Schema 转换为 DataSchema
        let mut data_schema = DataSchema::new("json_schema", "1.0");
        for field in csv_schema.fields() {
            let field_def = crate::data::schema::schema::FieldDefinition {
                name: field.name().to_string(),
                field_type: field.field_type.to_schema_field_type(),
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
        if let Some(total) = self.total_rows {
            Ok(total)
        } else {
            // 创建临时处理器来计算行数
            let mut temp_processor = JSONProcessor::new(
                self.file_path.to_str().unwrap(),
                Some(self.options.clone())
            )?;
            temp_processor.load_data()?;
            if let Some(data) = &temp_processor.data {
                Ok(data.len())
            } else {
                Ok(0)
            }
        }
    }
    
    fn read_rows(&mut self, count: usize) -> Result<Vec<Record>> {
        // 确保数据已加载
        if self.data.is_none() {
            self.load_data()?;
        }
        
        let data = self.data.as_ref().ok_or_else(|| Error::invalid_state("未加载数据".to_string()))?;
        
        // 确保模式已推断
        let schema = if let Some(schema) = &self.schema {
            schema.clone()
        } else {
            self.infer_schema()?
        };
        
        let mut records = Vec::with_capacity(count);
        let mut i = 0;
        
        while i < count && self.current_row < data.len() {
            if let Some(obj) = data[self.current_row].as_object() {
                let record = self.json_to_record(obj, &schema);
                records.push(record);
            }
            
            self.current_row += 1;
            i += 1;
        }
        
        // 更新当前位置
        self.current_position = self.current_row;
        
        Ok(records)
    }
    
    fn reset(&mut self) -> Result<()> {
        self.current_row = 0;
        Ok(())
    }
}

// 非trait方法单独实现
impl JSONProcessor {
    // 获取当前位置
    pub fn get_current_position(&self) -> usize {
        self.current_position
    }

    // 处理流数据
    pub fn process_stream<R: Read>(&mut self, stream: R) -> Result<Vec<Record>> {
        let reader = BufReader::new(stream);
        
        // 解析JSON数据
        let json_data: serde_json::Value = serde_json::from_reader(reader)?;
        
        let json_array = if json_data.is_array() {
            json_data.as_array().unwrap().clone()
        } else if json_data.is_object() {
            // 如果是单个对象，将其包装成数组
            let mut array = Vec::new();
            array.push(json_data);
            array
        } else {
            return Err(Error::invalid_input("JSON data is not an array or object".to_string()));
        };
        
        // 推断或使用现有模式
        let schema = if let Some(schema) = &self.schema {
            schema.clone()
        } else {
            // 临时设置数据以推断模式
            self.data = Some(json_array.clone());
            let schema = self.infer_schema()?;
            self.data = None; // 重置，避免占用内存
            schema
        };
        
        let mut records = Vec::with_capacity(json_array.len());
        
        for json_value in &json_array {
            if let Some(json_obj) = json_value.as_object() {
                let record = self.json_to_record(json_obj, &schema);
                records.push(record);
            } else {
                warn!("Skipping non-object JSON value");
            }
        }
        
        Ok(records)
    }
}

/// 推断JSON文件模式
pub fn infer_json_schema_impl(path: &Path) -> Result<DataSchema> {
    debug!("推断JSON文件模式: {}", path.display());
    
    // 检查文件是否存在
    if !path.exists() {
        return Err(Error::not_found(path.to_string_lossy().to_string()));
    }
    
    // 检查是否为空文件
    if super::is_empty_file(path)? {
        return Err(Error::invalid_data("JSON文件为空".to_string()));
    }

    // 检测JSON格式
    let format = detect_json_format(path)?;
    debug!("检测到JSON格式: {:?}", format);
    
    // 根据不同格式推断模式（返回 DataField，稍后转换为 Field）
    let data_fields = match format {
        JsonFormat::Object => {
            // 读取JSON对象
            let file = File::open(path)
                .map_err(|e| Error::io_error(format!("无法打开JSON文件: {}", e)))?;
            let reader = BufReader::new(file);
            let value: Value = serde_json::from_reader(reader)
                .map_err(|e| Error::invalid_data(format!("无法解析JSON: {}", e)))?;
            
            infer_json_object_schema(&value, None)
        },
        JsonFormat::Array => {
            // 单层数组处理更复杂，需要分析数组中的元素类型
            let file = File::open(path)
                .map_err(|e| Error::io_error(format!("无法打开JSON文件: {}", e)))?;
            let reader = BufReader::new(file);
            let value: Value = serde_json::from_reader(reader)
                .map_err(|e| Error::invalid_data(format!("无法解析JSON: {}", e)))?;
            
            let array = value.as_array().unwrap();
            infer_json_array_schema(array, None)
        },
        JsonFormat::ArrayOfObjects => {
            // 对象数组处理，综合分析所有对象的结构
            let file = File::open(path)
                .map_err(|e| Error::io_error(format!("无法打开JSON文件: {}", e)))?;
            let reader = BufReader::new(file);
            let value: Value = serde_json::from_reader(reader)
                .map_err(|e| Error::invalid_data(format!("无法解析JSON: {}", e)))?;
            
            let array = value.as_array().unwrap();
            infer_json_array_of_objects_schema(array, None)
        },
        JsonFormat::JSONL => {
            // JSONL格式，按行读取
            infer_jsonl_schema(path, 100)?
        },
        JsonFormat::Unknown => {
            return Err(Error::invalid_data("无法识别的JSON格式".to_string()));
        }
    };
    
    // 将 DataField 转换为内部schema定义的 FieldDefinition
    let fields: Vec<crate::data::schema::schema::FieldDefinition> = data_fields
        .into_iter()
        .map(|df| {
            let mut metadata = HashMap::new();
            metadata.insert("source_type".to_string(), "json".to_string());
            metadata.insert("source_path".to_string(), path.to_string_lossy().to_string());
            metadata.insert("json_format".to_string(), format!("{:?}", format));
            crate::data::schema::schema::FieldDefinition {
                name: df.name,
                field_type: outer_to_inner_field_type(&df.field_type),
                data_type: None,
                required: df.required,
                nullable: !df.required,
                primary_key: false,
                foreign_key: None,
                description: None,
                default_value: None,
                constraints: None,
                metadata,
            }
        })
        .collect();
    
    // 估算记录数
    let record_count = match format {
        JsonFormat::Object => 1,
        JsonFormat::Array | JsonFormat::ArrayOfObjects => {
            // 读取数组长度
            let file = File::open(path)
                .map_err(|e| Error::io_error(format!("无法打开JSON文件: {}", e)))?;
            let reader = BufReader::new(file);
            let value: Value = serde_json::from_reader(reader)
                .map_err(|e| Error::invalid_data(format!("无法解析JSON: {}", e)))?;
            
            value.as_array().map(|a| a.len()).unwrap_or(0)
        },
        JsonFormat::JSONL => {
            // 估算行数
            estimate_jsonl_lines(path)?
        },
        JsonFormat::Unknown => 0,
    };
    
    // 构建最终模式
    let mut schema = DataSchema::new("json_schema", "1.0");
    for field_def in fields {
        let mut field_def = field_def;
        // 附带元数据信息
        let mut meta = HashMap::new();
        meta.insert("source_type".to_string(), "json".to_string());
        meta.insert("source_path".to_string(), path.to_string_lossy().to_string());
        meta.insert("json_format".to_string(), format!("{:?}", format));
        meta.insert("record_count".to_string(), record_count.to_string());
        field_def.metadata = meta;
        schema.add_field(field_def)?;
    }
    debug!("成功推断JSON模式: {}, 字段数: {}", path.display(), schema.fields().len());
    
    Ok(schema)
}

/// 从JSON对象推断模式
fn infer_json_object_schema(value: &Value, parent_path: Option<&str>) -> Vec<DataField> {
    match value {
        Value::Object(obj) => {
            let mut fields = Vec::new();
            
            for (key, val) in obj {
                let field_path = if let Some(parent) = parent_path {
                    format!("{}.{}", parent, key)
                } else {
                    key.clone()
                };
                
                // 处理嵌套对象
                if val.is_object() {
                    let nested_fields = infer_json_object_schema(val, Some(&field_path));
                    fields.extend(nested_fields);
                } 
                // 处理嵌套数组
                else if val.is_array() {
                    // 如果数组中的元素是对象，则推断它们的结构
                    let array = val.as_array().unwrap();
                    if !array.is_empty() && array[0].is_object() {
                        let nested_fields = infer_json_array_of_objects_schema(array, Some(&field_path));
                        fields.extend(nested_fields);
                    } else {
                        // 作为单个数组字段处理
                        let csv_field_type = determine_array_type(array);
                        let inner_type = csv_field_type.to_schema_field_type();
                        let outer_type = inner_to_outer_field_type(&inner_type);
                        fields.push(DataField::new(
                            field_path.clone(),
                            outer_type,
                            true,
                            None,
                        ));
                    }
                } 
                // 处理简单类型
                else {
                    let csv_field_type = JSONProcessor::json_value_to_field_type(val);
                    let inner_type = csv_field_type.to_schema_field_type();
                    let outer_type = inner_to_outer_field_type(&inner_type);
                    fields.push(DataField::new(
                        field_path.clone(),
                        outer_type,
                        true,
                        None,
                    ));
                }
            }
            
            fields
        },
        _ => Vec::new()
    }
}

/// 从JSON数组推断模式
fn infer_json_array_schema(array: &Vec<Value>, parent_path: Option<&str>) -> Vec<DataField> {
    if array.is_empty() {
        return Vec::new();
    }
    
    // 检查是否为基本类型数组
    let is_primitive = array.iter().all(|val| {
        val.is_null() || val.is_boolean() || val.is_number() || val.is_string()
    });
    
    if is_primitive {
        let field_name = parent_path.unwrap_or("items").to_string();
        let csv_field_type = determine_array_type(array);
        let inner_type = csv_field_type.to_schema_field_type();
        let outer_type = inner_to_outer_field_type(&inner_type);
        
        return vec![DataField::new(
            field_name,
            outer_type,
            true,
            None,
        )];
    }
    
    // 复杂数组类型处理
    if array[0].is_object() {
        infer_json_array_of_objects_schema(array, parent_path)
    } else if array[0].is_array() {
        // 数组的数组
        let mut fields = Vec::new();
        for (i, val) in array.iter().enumerate() {
            if val.is_array() {
                let nested_path = if let Some(parent) = parent_path {
                    format!("{}[{}]", parent, i)
                } else {
                    format!("array[{}]", i)
                };
                
                let nested_fields = infer_json_array_schema(val.as_array().unwrap(), Some(&nested_path));
                fields.extend(nested_fields);
            }
        }
        fields
    } else {
        Vec::new()
    }
}

/// 从JSON对象数组推断模式
fn infer_json_array_of_objects_schema(array: &Vec<Value>, parent_path: Option<&str>) -> Vec<DataField> {
    if array.is_empty() {
        return Vec::new();
    }
    
    // 收集所有对象中的所有字段
    let mut all_fields = HashMap::new();
    let mut field_types = HashMap::new();
    
    for value in array {
        if let Value::Object(obj) = value {
            for (key, val) in obj {
                let field_path = if let Some(parent) = parent_path {
                    format!("{}.{}", parent, key)
                } else {
                    key.clone()
                };
                
                // 记录字段存在
                all_fields.entry(field_path.clone()).or_insert(0);
                
                // 记录字段类型
                let field_type = JSONProcessor::json_value_to_field_type(val);
                field_types
                    .entry(field_path)
                    .or_insert_with(Vec::new)
                    .push(field_type);
            }
        }
    }
    
    // 构建字段列表
    let mut fields = Vec::new();
    for (field_path, _) in all_fields {
        let field_types_vec = field_types.get(&field_path).unwrap();
        let field_type = determine_field_type(field_types_vec);
        
        let csv_field_type = field_type;
        let inner_type = csv_field_type.to_schema_field_type();
        let outer_type = inner_to_outer_field_type(&inner_type);
        fields.push(DataField::new(
            field_path.clone(),
            outer_type,
            true, // 数组中的字段默认可为空
            None,
        ));
    }
    
    fields
}

/// 确定数组的类型
fn determine_array_type(array: &Vec<Value>) -> FieldType {
    if array.is_empty() {
        return FieldType::Array;
    }
    
    // 收集数组元素类型
    let mut element_types = Vec::new();
    for value in array {
        element_types.push(JSONProcessor::json_value_to_field_type(value));
    }
    
    // 确定元素类型
    let element_type = determine_field_type(&element_types);
    
    // 返回数组类型
    match element_type {
        FieldType::Integer => FieldType::Array, // 整数数组
        FieldType::Float => FieldType::Array, // 浮点数数组
        FieldType::Boolean => FieldType::Array, // 布尔数组
        FieldType::String => FieldType::Array, // 字符串数组
        FieldType::Date => FieldType::Array, // 日期数组
        FieldType::Timestamp => FieldType::Array, // 时间戳数组
        _ => FieldType::Array, // 其他类型的数组
    }
}

/// 确定字段类型
fn determine_field_type(types: &Vec<FieldType>) -> FieldType {
    if types.is_empty() {
        return FieldType::Null;
    }
    
    // 优先判断是否所有值都是同一类型
    let first_type = types[0];
    if types.iter().all(|&t| t == first_type) {
        return first_type;
    }
    
    // 计算各类型出现频率
    let mut type_counts: HashMap<FieldType, usize> = HashMap::new();
    for &t in types {
        *type_counts.entry(t).or_insert(0) += 1;
    }
    
    // 过滤掉Null类型
    type_counts.remove(&FieldType::Null);
    
    if type_counts.is_empty() {
        return FieldType::Null;
    }
    
    // 找出最常见的类型
    let (most_common_type, _) = type_counts
        .iter()
        .max_by_key(|&(_, count)| count)
        .unwrap_or((&FieldType::String, &0));
    
    // 类型升级规则
    if type_counts.len() > 1 {
        // 如果同时有整数和浮点数，选择浮点数
        if type_counts.contains_key(&FieldType::Integer) && type_counts.contains_key(&FieldType::Float) {
            return FieldType::Float;
        }
        
        // 如果同时有数字和字符串，可能是类别数据，选择字符串
        if (type_counts.contains_key(&FieldType::Integer) || type_counts.contains_key(&FieldType::Float)) 
           && type_counts.contains_key(&FieldType::String) {
            return FieldType::String;
        }
        
        // 如果同时有日期/时间和字符串，选择字符串
        if (type_counts.contains_key(&FieldType::Date) || type_counts.contains_key(&FieldType::Timestamp)) 
           && type_counts.contains_key(&FieldType::String) {
            return FieldType::String;
        }
    }
    
    *most_common_type
}

/// 检测JSON文件格式
/// 
/// 返回格式字符串: "array"、"object"或"lines"
pub fn detect_json_format(path: &Path) -> Result<JsonFormat> {
    // 打开文件
    let file = File::open(path).map_err(|e| Error::io_error(format!("无法打开JSON文件: {}", e)))?;
    let mut reader = BufReader::new(file);
    
    // 读取文件前1KB用于格式检测
    let mut buffer = [0; 1024];
    let bytes_read = reader.read(&mut buffer).map_err(|e| Error::io_error(format!("读取JSON文件失败: {}", e)))?;
    
    if bytes_read == 0 {
        return Err(Error::invalid_input("JSON文件为空"));
    }
    
    // 跳过前导空白字符
    let mut start_index = 0;
    while start_index < bytes_read && (buffer[start_index] == b' ' || buffer[start_index] == b'\t' || 
           buffer[start_index] == b'\n' || buffer[start_index] == b'\r') {
        start_index += 1;
    }
    
    if start_index >= bytes_read {
        return Err(Error::invalid_input("JSON文件只包含空白字符"));
    }
    
    // 检查第一个非空白字符
    match buffer[start_index] {
        b'[' => {
            // 可能是JSON数组，尝试验证
            reader.seek(SeekFrom::Start(0)).map_err(|e| Error::io_error(format!("重置文件指针失败: {}", e)))?;
            
            // 尝试解析为JSON数组
            match serde_json::from_reader::<_, Value>(reader) {
                Ok(value) => {
                    if value.is_array() {
                        let array = value.as_array().unwrap();
                        if !array.is_empty() && array[0].is_object() {
                            // 是对象数组
                            return Ok(JsonFormat::ArrayOfObjects);
                        } else {
                            // 是简单数组
                            return Ok(JsonFormat::Array);
                        }
                    } else {
                        return Err(Error::invalid_input("JSON文件格式解析错误"));
                    }
                },
                Err(_) => {
                    // 解析失败，可能不是有效的JSON数组
                    return Err(Error::invalid_input("JSON文件不是有效的数组格式"));
                }
            }
        },
        b'{' => {
            // 可能是单个JSON对象
            reader.seek(SeekFrom::Start(0)).map_err(|e| Error::io_error(format!("重置文件指针失败: {}", e)))?;
            
            // 检查是否为JSONL格式（每行一个JSON对象）
            let mut line = String::new();
            reader.read_line(&mut line).map_err(|e| Error::io_error(format!("读取JSON行失败: {}", e)))?;
            
            if line.trim().ends_with('}') {
                // 读取下一行，检查是否有更多对象
                let mut second_line = String::new();
                let bytes_read = reader.read_line(&mut second_line).map_err(|e| Error::io_error(format!("读取第二行失败: {}", e)))?;
                
                if bytes_read > 0 && second_line.trim().starts_with('{') {
                    // 似乎是JSONL格式
                    return Ok(JsonFormat::JSONL);
                } else {
                    // 尝试验证是否为单个有效JSON对象
                    reader.seek(SeekFrom::Start(0)).map_err(|e| Error::io_error(format!("重置文件指针失败: {}", e)))?;
                    match serde_json::from_reader::<_, Value>(reader) {
                        Ok(value) => {
                            if value.is_object() {
                                return Ok(JsonFormat::Object);
                            } else {
                                return Err(Error::invalid_input("JSON文件不是有效的对象格式"));
                            }
                        },
                        Err(_) => {
                            // 可能是JSONL格式，但不是有效的单个JSON对象
                            return Ok(JsonFormat::JSONL);
                        }
                    }
                }
            } else {
                // 可能是JSONL格式或无效JSON
                return Ok(JsonFormat::JSONL);
            }
        },
        _ => {
            // 不是标准JSON开头，检查是否为JSONL格式
            reader.seek(SeekFrom::Start(0)).map_err(|e| Error::io_error(format!("重置文件指针失败: {}", e)))?;
            
            // 读取前几行，检查是否每行都是JSON对象
            let mut valid_json_lines = 0;
            let mut line = String::new();
            
            for _ in 0..5 {
                line.clear();
                let bytes_read = reader.read_line(&mut line).map_err(|e| Error::io_error(format!("读取JSON行失败: {}", e)))?;
                if bytes_read == 0 {
                    break;
                }
                
                let trimmed = line.trim();
                if !trimmed.is_empty() {
                    if let Ok(_) = serde_json::from_str::<Value>(trimmed) {
                        valid_json_lines += 1;
                    }
                }
            }
            
            if valid_json_lines > 0 {
                return Ok(JsonFormat::JSONL);
            } else {
                return Ok(JsonFormat::Unknown);
            }
        }
    }
}

/// 估计JSON数组的大小（元素数量）
pub fn estimate_json_array_size(path: &Path) -> Result<usize> {
    // 打开文件
    let file = File::open(path).map_err(|e| Error::io_error(format!("无法打开JSON文件: {}", e)))?;
    let reader = BufReader::new(file);
    
    // 检测文件格式
    let format = detect_json_format(path)?;
    
    match format {
        JsonFormat::Array | JsonFormat::ArrayOfObjects => {
            // 对于数组格式，直接解析并计数
            let value: Value = serde_json::from_reader(reader)
                .map_err(|e| Error::serialization(&format!("解析JSON失败: {}", e)))?;
            
            if let Some(array) = value.as_array() {
                return Ok(array.len());
            } else {
                return Err(Error::invalid_input("JSON不是数组格式"));
            }
        },
        JsonFormat::Object => {
            // 对于单个对象，返回1
            return Ok(1);
        },
        JsonFormat::JSONL => {
            // 对于JSONL格式，估计行数
            return estimate_jsonl_lines(path);
        },
        JsonFormat::Unknown => {
            return Err(Error::invalid_input("不支持的JSON格式"));
        }
    }
}

/// 推断JSONL格式的模式
fn infer_jsonl_schema(path: &Path, sample_lines: usize) -> Result<Vec<DataField>> {
    let file = File::open(path)
        .map_err(|e| Error::io_error(format!("无法打开JSONL文件: {}", e)))?;
    
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    
    // 合并所有对象的字段
    let mut all_fields = HashMap::new();
    let mut field_types = HashMap::new();
    let mut line_count = 0;
    
    while let Some(Ok(line)) = lines.next() {
        if line.trim().is_empty() {
            continue;
        }
        
        // 解析JSON行
        match serde_json::from_str::<Value>(&line) {
            Ok(Value::Object(obj)) => {
                for (key, val) in obj {
                    // 记录字段存在
                    all_fields.entry(key.clone()).or_insert(0);
                    
                    // 记录字段类型
                    let field_type = JSONProcessor::json_value_to_field_type(&val);
                    field_types
                        .entry(key)
                        .or_insert_with(Vec::new)
                        .push(field_type);
                }
            },
            Err(e) => {
                warn!("解析JSONL行失败: {}", e);
                continue;
            },
            _ => {
                warn!("JSONL行不是对象格式");
                continue;
            }
        }
        
        line_count += 1;
        if line_count >= sample_lines {
            break;
        }
    }
    
    // 构建字段列表
    let mut fields = Vec::new();
    for (field_name, _) in all_fields {
        let field_types_vec = field_types.get(&field_name).unwrap();
        let field_type = determine_field_type(field_types_vec);
        
        let csv_field_type = field_type;
        let inner_type = csv_field_type.to_schema_field_type();
        let outer_type = inner_to_outer_field_type(&inner_type);
        fields.push(DataField::new(
            field_name.clone(),
            outer_type,
            true, // JSONL中的字段默认可为空
            None,
        ));
    }
    
    Ok(fields)
}

/// 估算JSONL文件的行数
fn estimate_jsonl_lines(path: &Path) -> Result<usize> {
    let file_size = std::fs::metadata(path)
        .map_err(|e| Error::io_error(format!("无法获取文件元数据: {}", e)))?
        .len();
    
    // 对于小文件，直接计数
    if file_size < 10_000_000 { // 10MB
        let file = File::open(path)
            .map_err(|e| Error::io_error(format!("无法打开JSONL文件: {}", e)))?;
        
        let reader = BufReader::new(file);
        let line_count = reader.lines().count();
        
        return Ok(line_count);
    }
    
    // 对于大文件，采样估算
    let file = File::open(path)
        .map_err(|e| Error::io_error(format!("无法打开JSONL文件: {}", e)))?;
    
    let mut reader = BufReader::new(file);
    let sample_size = 1_000_000; // 1MB样本
    let mut buffer = vec![0; sample_size];
    
    let bytes_read = reader.read(&mut buffer)
        .map_err(|e| Error::io_error(format!("读取JSONL文件失败: {}", e)))?;
    
    buffer.truncate(bytes_read);
    let sample = String::from_utf8_lossy(&buffer);
    
    // 计算样本中的行数
    let line_count = sample.lines().count();
    
    // 估算总行数
    let estimated_lines = (line_count as f64 / bytes_read as f64 * file_size as f64).round() as usize;
    
    Ok(estimated_lines)
}

/// JSON数据加载器
pub struct JsonLoader {
    /// 文件路径
    file_path: PathBuf,
    /// 加载选项
    options: HashMap<String, String>,
}

impl JsonLoader {
    /// 创建新的JSON加载器
    pub fn new(path: &str) -> Result<Self> {
        let file_path = PathBuf::from(path);
        
        // 验证文件是否存在
        if !file_path.exists() {
            return Err(Error::not_found(file_path.to_string_lossy().to_string()));
        }
        
        Ok(Self {
            file_path,
            options: HashMap::new(),
        })
    }
    
    /// 从文件加载数据
    pub fn load_from_file(&self, path: &str) -> Result<DataBatch> {
        let mut processor = JSONProcessor::new(path, Some(self.options.clone()))?;
        
        // 推断模式
        let schema = processor.get_schema()?;
        
        // 获取行数
        let row_count = processor.get_row_count()?;
        
        // 读取所有行
        let records = processor.read_rows(row_count)?;
        
        // 将Record转换为HashMap格式
        let mut hashmap_records = Vec::new();
        let field_names: Vec<String> = schema.fields().iter().map(|f| f.name().to_string()).collect();
        
        for record in records {
            let mut hashmap_record = HashMap::new();
            for (i, value) in record.values().iter().enumerate() {
                if i < field_names.len() {
                    hashmap_record.insert(field_names[i].clone(), value.clone());
                }
            }
            hashmap_records.push(hashmap_record);
        }
        
        // 创建DataBatch
        let data_batch = DataBatch::from_records(hashmap_records, Some(schema))?;
        
        Ok(data_batch)
    }
    
    /// 设置选项
    pub fn with_option<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.options.insert(key.into(), value.into());
        self
    }
    
    /// 获取文件路径
    pub fn file_path(&self) -> &Path {
        &self.file_path
    }
}
