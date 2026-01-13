// 数据记录模块
// 处理数据记录的核心结构和操作

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use crate::error::Result;
use crate::data::value::DataValue;
use std::fmt;

/// 数据记录，表示单个数据条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    /// 记录ID
    pub id: Option<String>,
    /// 字段值映射
    pub fields: HashMap<String, Value>,
    /// 记录元数据
    pub metadata: HashMap<String, String>,
}

/// 字段值类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    /// 对应DataValue
    Data(DataValue),
    /// 嵌套记录
    Record(Box<Record>),
    /// 记录引用
    Reference(String),
}

impl Record {
    /// 创建新的记录
    pub fn new() -> Self {
        Self {
            id: None,
            fields: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// 设置记录ID
    pub fn with_id(mut self, id: &str) -> Self {
        self.id = Some(id.to_string());
        self
    }

    /// 添加字段
    pub fn add_field(&mut self, name: &str, value: Value) -> &mut Self {
        self.fields.insert(name.to_string(), value);
        self
    }

    /// 获取字段
    pub fn get_field(&self, name: &str) -> Option<&Value> {
        self.fields.get(name)
    }

    /// 获取字段值
    pub fn get_value(&self, name: &str) -> Option<&Value> {
        self.fields.get(name)
    }

    /// 添加元数据
    pub fn add_metadata(&mut self, key: &str, value: &str) -> &mut Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// 获取元数据
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// 检查字段是否存在
    pub fn has_field(&self, name: &str) -> bool {
        self.fields.contains_key(name)
    }

    /// 获取所有字段名
    pub fn field_names(&self) -> Vec<String> {
        self.fields.keys().cloned().collect()
    }

    /// 字段数量
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// 转换为DataValue
    pub fn to_data_value(&self) -> Result<DataValue> {
        let mut map = HashMap::new();
        for (key, value) in &self.fields {
            map.insert(key.clone(), value.to_data_value()?);
        }
        Ok(DataValue::Object(map))
    }

    /// 从DataValue创建
    pub fn from_data_value(value: &DataValue) -> Result<Self> {
        if let DataValue::Object(map) = value {
            let mut record = Record::new();
            for (key, val) in map {
                record.add_field(key, Value::Data(val.clone()));
            }
            Ok(record)
        } else {
            Err(crate::error::Error::invalid_argument("期望对象类型DataValue"))
        }
    }
}

impl Default for Record {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Record {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Record{{")?;
        let mut first = true;
        for (key, value) in &self.fields {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{}: {}", key, value)?;
            first = false;
        }
        write!(f, "}}")
    }
}

impl Value {
    /// 转换为DataValue
    pub fn to_data_value(&self) -> Result<DataValue> {
        match self {
            Value::Data(data) => Ok(data.clone()),
            Value::Record(record) => record.to_data_value(),
            Value::Reference(ref_id) => Ok(DataValue::String(ref_id.clone())),
        }
    }

    /// 从DataValue创建
    pub fn from_data_value(value: DataValue) -> Self {
        Value::Data(value)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Data(data) => write!(f, "{}", data),
            Value::Record(record) => write!(f, "{}", record),
            Value::Reference(ref_id) => write!(f, "Ref({})", ref_id),
        }
    }
}

/// 将DataField导出，用于其他模块引用
pub struct DataField {
    pub name: String,
    pub field_type: crate::data::schema::FieldType,
    pub required: bool,
    pub description: Option<String>,
    pub metadata: HashMap<String, String>,
}

impl DataField {
    /// 创建新的数据字段
    pub fn new(
        name: String,
        field_type: crate::data::schema::FieldType,
        required: bool,
        description: Option<String>,
    ) -> Self {
        Self {
            name,
            field_type,
            required,
            description,
            metadata: HashMap::new(),
        }
    }

    /// 创建新的数据字段（简化版本）
    pub fn simple(name: String, field_type: crate::data::schema::FieldType) -> Self {
        Self::new(name, field_type, false, None)
    }

    /// 添加元数据
    pub fn add_metadata(&mut self, key: String, value: String) -> &mut Self {
        self.metadata.insert(key, value);
        self
    }

    /// 获取元数据
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// 设置为必需字段
    pub fn set_required(mut self, required: bool) -> Self {
        self.required = required;
        self
    }

    /// 设置描述
    pub fn set_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }
}

/// 记录集合，表示一组记录
#[derive(Debug, Clone, Default)]
pub struct RecordSet {
    pub records: Vec<Record>,
    pub schema_id: Option<String>,
    pub metadata: HashMap<String, String>,
}

impl RecordSet {
    /// 创建新的记录集
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            schema_id: None,
            metadata: HashMap::new(),
        }
    }

    /// 添加记录
    pub fn add_record(&mut self, record: Record) -> &mut Self {
        self.records.push(record);
        self
    }

    /// 记录数量
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// 获取记录
    pub fn get_record(&self, index: usize) -> Option<&Record> {
        self.records.get(index)
    }

    /// 获取可变记录
    pub fn get_record_mut(&mut self, index: usize) -> Option<&mut Record> {
        self.records.get_mut(index)
    }

    /// 设置模式ID
    pub fn with_schema_id(mut self, schema_id: &str) -> Self {
        self.schema_id = Some(schema_id.to_string());
        self
    }

    /// 添加元数据
    pub fn add_metadata(&mut self, key: &str, value: &str) -> &mut Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// 数据记录引用
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordReference {
    pub id: String,
    pub schema_id: Option<String>,
    pub collection: Option<String>,
}

impl RecordReference {
    /// 创建新的记录引用
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            schema_id: None,
            collection: None,
        }
    }

    /// 设置模式ID
    pub fn with_schema_id(mut self, schema_id: &str) -> Self {
        self.schema_id = Some(schema_id.to_string());
        self
    }

    /// 设置集合
    pub fn with_collection(mut self, collection: &str) -> Self {
        self.collection = Some(collection.to_string());
        self
    }
}

/// 导出DataRecord类型别名
pub type DataRecord = Record;

/// 记录值类型 - 与handlers.rs兼容
pub use crate::data::value::DataValue as RecordValue; 