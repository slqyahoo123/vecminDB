use std::collections::HashMap;
use std::fmt;
use crate::core::CoreTensorData;
use crate::error::{Error, Result};
use regex;
use serde;
use serde_json;
use serde::{Serialize, Deserialize};
use base64::{Engine as _, engine::general_purpose};
// DataType 和 DeviceType 在当前实现中已不再直接使用，保留相关逻辑由 CoreTensorData 管理

/// 数据值类型，支持多种数据格式
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataValue {
    /// 空值
    Null,
    /// 布尔值
    Boolean(bool),
    /// 整数值
    Integer(i64),
    /// 浮点数值
    Float(f64),
    /// 数字值(通用)
    Number(f64),
    /// 字符串值
    String(String),
    /// 文本值（String的别名，用于兼容性）
    Text(String),
    /// 数组值
    Array(Vec<DataValue>),
    /// 字符串数组
    StringArray(Vec<String>),
    /// 对象值
    Object(HashMap<String, DataValue>),
    /// 二进制数据
    Binary(Vec<u8>),
    /// 日期时间 (存储为ISO 8601格式字符串)
    DateTime(String),
    /// 张量数据 - 用于AI模型
    Tensor(CoreTensorData),
}

/// 统一数据值表示 - 用于模块间数据传递，减少转换开销
#[derive(Debug, Clone)]
pub enum UnifiedValue {
    /// 标量值
    Scalar(ScalarValue),
    /// 向量值
    Vector(VectorValue),
    /// 矩阵值
    Matrix(MatrixValue),
    /// 张量值
    Tensor(TensorValue),
    /// 文本值
    Text(String),
    /// 整数值
    Integer(i64),
    /// 浮点数值
    Float(f64),
    /// 二进制数据
    Binary(Vec<u8>),
    /// 复合数据
    Composite(HashMap<String, UnifiedValue>),
}

/// 标量值类型
#[derive(Debug, Clone)]
pub enum ScalarValue {
    /// 布尔值
    Bool(bool),
    /// 整数
    Int(i64),
    /// 浮点数
    Float(f64),
}

/// 向量值类型
#[derive(Debug, Clone)]
pub struct VectorValue {
    /// 向量数据
    pub data: Vec<f32>,
    /// 数据类型
    pub dtype: DType,
}

/// 矩阵值类型
#[derive(Debug, Clone)]
pub struct MatrixValue {
    /// 矩阵数据 (按行存储)
    pub data: Vec<f32>,
    /// 行数
    pub rows: usize,
    /// 列数
    pub cols: usize,
    /// 数据类型
    pub dtype: DType,
}

/// 张量值类型
#[derive(Debug, Clone)]
pub struct TensorValue {
    /// 张量数据
    pub data: Vec<f32>,
    /// 形状
    pub shape: Vec<usize>,
    /// 数据类型
    pub dtype: DType,
}

/// 数据类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// 32位浮点
    Float32,
    /// 64位浮点
    Float64,
    /// 32位整数
    Int32,
    /// 64位整数
    Int64,
}

impl DataValue {
    /// 检查值是否为空
    pub fn is_null(&self) -> bool {
        matches!(self, DataValue::Null)
    }

    /// 尝试获取布尔值
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            DataValue::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// 尝试获取整数值
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            DataValue::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// 尝试获取浮点数值
    pub fn as_number(&self) -> Option<f64> {
        match self {
            DataValue::Float(f) => Some(*f),
            DataValue::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// 尝试获取字符串引用
    pub fn as_str(&self) -> Option<&str> {
        match self {
            DataValue::String(s) | DataValue::Text(s) => Some(s),
            _ => None,
        }
    }

    /// 尝试获取数组引用
    pub fn as_array(&self) -> Option<&Vec<DataValue>> {
        match self {
            DataValue::Array(a) => Some(a),
            _ => None,
        }
    }

    /// 尝试获取字符串数组引用
    pub fn as_string_array(&self) -> Option<&Vec<String>> {
        match self {
            DataValue::StringArray(a) => Some(a),
            _ => None,
        }
    }

    /// 尝试获取对象引用
    pub fn as_object(&self) -> Option<&HashMap<String, DataValue>> {
        match self {
            DataValue::Object(o) => Some(o),
            _ => None,
        }
    }

    /// 尝试获取二进制数据引用
    pub fn as_binary(&self) -> Option<&Vec<u8>> {
        match self {
            DataValue::Binary(b) => Some(b),
            _ => None,
        }
    }

    /// 尝试获取日期时间引用
    pub fn as_datetime(&self) -> Option<&str> {
        match self {
            DataValue::DateTime(d) => Some(d),
            _ => None,
        }
    }

    /// 尝试获取张量数据引用
    pub fn as_tensor(&self) -> Option<&CoreTensorData> {
        match self {
            DataValue::Tensor(t) => Some(t),
            _ => None,
        }
    }
    
    /// 从二进制数据创建DataValue
    pub fn from_binary(bytes: Vec<u8>) -> Self {
        DataValue::Binary(bytes)
    }
    
    /// 从字符串数组创建DataValue
    pub fn from_string_array(strings: Vec<String>) -> Self {
        DataValue::StringArray(strings)
    }

    /// 从JSON值转换为DataValue
    pub fn from_json(json: serde_json::Value) -> Self {
        match json {
            serde_json::Value::Null => DataValue::Null,
            serde_json::Value::Bool(b) => DataValue::Boolean(b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    DataValue::Integer(i)
                } else if let Some(f) = n.as_f64() {
                    DataValue::Float(f)
                } else {
                    // 默认作为字符串处理
                    DataValue::String(n.to_string())
                }
            },
            serde_json::Value::String(s) => {
                // 尝试识别日期和时间戳
                if is_date_format(&s) || is_timestamp_format(&s) {
                    DataValue::DateTime(s)
                } else {
                    DataValue::String(s)
                }
            },
            serde_json::Value::Array(arr) => {
                // 检查是否全部是字符串
                let all_strings = arr.iter().all(|v| v.is_string());
                
                if all_strings {
                    // 转换为StringArray
                    let strings = arr.into_iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect();
                    DataValue::StringArray(strings)
                } else {
                    // 普通数组
                    let values: Vec<DataValue> = arr.into_iter()
                        .map(DataValue::from_json)
                        .collect();
                    DataValue::Array(values)
                }
            },
            serde_json::Value::Object(obj) => {
                let mut map = HashMap::new();
                for (key, value) in obj {
                    map.insert(key, DataValue::from_json(value));
                }
                DataValue::Object(map)
            }
        }
    }

    /// 将DataValue转换为JSON值
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            DataValue::Null => serde_json::Value::Null,
            DataValue::Boolean(b) => serde_json::Value::Bool(*b),
            DataValue::Integer(i) => serde_json::Value::Number(serde_json::Number::from(*i)),
            DataValue::Float(f) => {
                if let Some(number) = serde_json::Number::from_f64(*f) {
                    serde_json::Value::Number(number)
                } else {
                    // 如果无法表示为JSON数字，则转为字符串
                    serde_json::Value::String(f.to_string())
                }
            },
            DataValue::Number(n) => {
                if let Some(number) = serde_json::Number::from_f64(*n) {
                    serde_json::Value::Number(number)
                } else {
                    // 如果无法表示为JSON数字，则转为字符串
                    serde_json::Value::String(n.to_string())
                }
            },
            DataValue::String(s) | DataValue::Text(s) => serde_json::Value::String(s.clone()),
            DataValue::Array(arr) => {
                let values = arr.iter()
                    .map(|v| v.to_json())
                    .collect();
                serde_json::Value::Array(values)
            },
            DataValue::StringArray(arr) => {
                let values = arr.iter()
                    .map(|s| serde_json::Value::String(s.clone()))
                    .collect();
                serde_json::Value::Array(values)
            },
            DataValue::Object(obj) => {
                let mut map = serde_json::Map::new();
                for (key, value) in obj {
                    map.insert(key.clone(), value.to_json());
                }
                serde_json::Value::Object(map)
            },
            DataValue::Binary(bytes) => {
                // 转换为base64字符串
                let base64 = general_purpose::STANDARD.encode(bytes);
                let mut map = serde_json::Map::new();
                map.insert("type".to_string(), serde_json::Value::String("binary".to_string()));
                map.insert("data".to_string(), serde_json::Value::String(base64));
                serde_json::Value::Object(map)
            },
            DataValue::DateTime(dt) => serde_json::Value::String(dt.clone()),
            DataValue::Tensor(tensor) => {
                // 创建包含形状和数据的对象
                let mut map = serde_json::Map::new();
                map.insert("type".to_string(), serde_json::Value::String("tensor".to_string()));
                map.insert("shape".to_string(), serde_json::to_value(&tensor.shape).unwrap_or(serde_json::Value::Null));
                
                // 转换数据 - 这里仅转换少量数据作为预览
                let preview: Vec<f32> = tensor.data.iter().take(10).cloned().collect();
                map.insert("data_preview".to_string(), serde_json::to_value(&preview).unwrap_or(serde_json::Value::Null));
                map.insert("data_len".to_string(), serde_json::Value::Number(serde_json::Number::from(tensor.data.len())));
                
                serde_json::Value::Object(map)
            }
        }
    }
}

/// 检查字符串是否为日期格式 (YYYY-MM-DD)
fn is_date_format(s: &str) -> bool {
    let re = regex::Regex::new(r"^\d{4}-\d{2}-\d{2}$").unwrap();
    re.is_match(s)
}

/// 检查字符串是否为时间戳格式 (ISO8601)
fn is_timestamp_format(s: &str) -> bool {
    let re = regex::Regex::new(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$").unwrap();
    re.is_match(s)
}

impl Default for DataValue {
    fn default() -> Self {
        DataValue::Null
    }
}

impl fmt::Display for DataValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataValue::Null => write!(f, "null"),
            DataValue::Boolean(b) => write!(f, "{}", b),
            DataValue::Integer(i) => write!(f, "{}", i),
            DataValue::Float(val) => write!(f, "{}", val),
            DataValue::String(s) | DataValue::Text(s) => write!(f, "\"{}\"", s),
            DataValue::Array(a) => {
                write!(f, "[")?;
                for (i, item) in a.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            },
            DataValue::StringArray(a) => {
                write!(f, "[")?;
                for (i, item) in a.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "\"{}\"", item)?;
                }
                write!(f, "]")
            },
            DataValue::Object(o) => {
                write!(f, "{{")?;
                for (i, (k, v)) in o.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "\"{}\": {}", k, v)?;
                }
                write!(f, "}}")
            },
            DataValue::Binary(b) => {
                write!(f, "<binary data: {} bytes>", b.len())
            },
            DataValue::DateTime(d) => write!(f, "\"{}\"", d),
            DataValue::Tensor(t) => {
                write!(f, "<tensor: shape={:?}>", t.shape)
            }
        }
    }
}

// 实现From特性，将各种类型转换为DataValue
impl From<bool> for DataValue {
    fn from(value: bool) -> Self {
        DataValue::Boolean(value)
    }
}

impl From<i32> for DataValue {
    fn from(value: i32) -> Self {
        DataValue::Integer(value as i64)
    }
}

impl From<i64> for DataValue {
    fn from(value: i64) -> Self {
        DataValue::Integer(value)
    }
}

impl From<f32> for DataValue {
    fn from(value: f32) -> Self {
        DataValue::Float(value as f64)
    }
}

impl From<f64> for DataValue {
    fn from(value: f64) -> Self {
        DataValue::Float(value)
    }
}

impl From<String> for DataValue {
    fn from(value: String) -> Self {
        DataValue::String(value)
    }
}

impl From<&str> for DataValue {
    fn from(value: &str) -> Self {
        DataValue::String(value.to_string())
    }
}

// 注意：我们不直接实现From<Vec<u8>>以避免冲突
// 对于Vec<u8>，应该使用DataValue::from_binary方法

// 实现Vec<String>到DataValue的转换
impl From<Vec<String>> for DataValue {
    fn from(value: Vec<String>) -> Self {
        DataValue::StringArray(value)
    }
}

// 实现Vec<bool>到DataValue的转换
impl From<Vec<bool>> for DataValue {
    fn from(value: Vec<bool>) -> Self {
        DataValue::Array(value.into_iter().map(DataValue::Boolean).collect())
    }
}

// 实现Vec<i32>到DataValue的转换
impl From<Vec<i32>> for DataValue {
    fn from(value: Vec<i32>) -> Self {
        DataValue::Array(value.into_iter().map(|v| DataValue::Integer(v as i64)).collect())
    }
}

// 实现Vec<i64>到DataValue的转换
impl From<Vec<i64>> for DataValue {
    fn from(value: Vec<i64>) -> Self {
        DataValue::Array(value.into_iter().map(DataValue::Integer).collect())
    }
}

// 实现Vec<f32>到DataValue的转换
impl From<Vec<f32>> for DataValue {
    fn from(value: Vec<f32>) -> Self {
        DataValue::Array(value.into_iter().map(|v| DataValue::Float(v as f64)).collect())
    }
}

// 实现Vec<f64>到DataValue的转换
impl From<Vec<f64>> for DataValue {
    fn from(value: Vec<f64>) -> Self {
        DataValue::Array(value.into_iter().map(DataValue::Float).collect())
    }
}

// 实现从HashMap的转换
impl<T: Into<DataValue>> From<HashMap<String, T>> for DataValue {
    fn from(value: HashMap<String, T>) -> Self {
        let mut map = HashMap::new();
        for (k, v) in value {
            map.insert(k, v.into());
        }
        DataValue::Object(map)
    }
}

/// 从CoreTensorData实现转换
impl From<CoreTensorData> for DataValue {
    fn from(value: CoreTensorData) -> Self {
        DataValue::Tensor(value)
    }
}

/// 为ManagerDataset实现转换为DataValue的From trait
impl From<crate::data::manager::ManagerDataset> for DataValue {
    fn from(dataset: crate::data::manager::ManagerDataset) -> Self {
        let mut object = HashMap::new();
        object.insert("id".to_string(), DataValue::String(dataset.id));
        object.insert("name".to_string(), DataValue::String(dataset.name));
        object.insert("dataset_type".to_string(), DataValue::String(format!("{:?}", dataset.dataset_type)));
        object.insert("format".to_string(), DataValue::String(dataset.format));
        object.insert("size".to_string(), DataValue::Integer(dataset.size as i64));
        object.insert("created_at".to_string(), DataValue::Integer(dataset.created_at));
        object.insert("updated_at".to_string(), DataValue::Integer(dataset.updated_at));
        
        // 转换metadata
        for (key, value) in dataset.metadata {
            object.insert(format!("metadata_{}", key), DataValue::String(value));
        }
        
        DataValue::Object(object)
    }
}

/// 为ManagerDataset引用实现转换为DataValue的From trait
impl From<&crate::data::manager::ManagerDataset> for DataValue {
    fn from(dataset: &crate::data::manager::ManagerDataset) -> Self {
        dataset.clone().into()
    }
}

/// 数据值适配器 - 用于在不同系统的数据值类型之间进行转换
pub struct DataValueAdapter;

impl DataValueAdapter {
    /// 将其他系统的数据值转换为标准DataValue
    /// 如果转换失败，返回None
    pub fn convert_from_any<T>(value: &T) -> Option<DataValue>
    where
        T: serde::Serialize,
    {
        // 使用serde_json作为中介进行转换
        match serde_json::to_value(value) {
            Ok(json_value) => Self::from_json_value(json_value),
            Err(_) => None,
        }
    }
    
    /// 从JSON值转换为DataValue
    pub fn from_json_value(value: serde_json::Value) -> Option<DataValue> {
        match value {
            serde_json::Value::Null => Some(DataValue::Null),
            serde_json::Value::Bool(b) => Some(DataValue::Boolean(b)),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Some(DataValue::Integer(i))
                } else if let Some(f) = n.as_f64() {
                    Some(DataValue::Float(f))
                } else {
                    None
                }
            },
            serde_json::Value::String(s) => Some(DataValue::String(s)),
            serde_json::Value::Array(a) => {
                // 检查是否全部是字符串
                let all_strings = a.iter().all(|v| v.is_string());
                
                if all_strings {
                    // 转换为StringArray
                    let strings = a.into_iter()
                        .filter_map(|v| match v {
                            serde_json::Value::String(s) => Some(s),
                            _ => None,
                        })
                        .collect();
                    Some(DataValue::StringArray(strings))
                } else {
                    // 普通数组
                    let mut items = Vec::new();
                    for item in a {
                        if let Some(converted) = Self::from_json_value(item) {
                            items.push(converted);
                        } else {
                            return None;
                        }
                    }
                    Some(DataValue::Array(items))
                }
            },
            serde_json::Value::Object(o) => {
                let mut map = HashMap::new();
                for (key, value) in o {
                    if let Some(converted) = Self::from_json_value(value) {
                        map.insert(key, converted);
                    } else {
                        return None;
                    }
                }
                Some(DataValue::Object(map))
            },
        }
    }
    
    /// 将DataValue转换为JSON值
    pub fn to_json_value(value: &DataValue) -> serde_json::Value {
        value.to_json()
    }
    
    /// 将DataValue转换为任意可反序列化的类型
    pub fn convert_to<T>(value: &DataValue) -> Result<T>
    where
        T: serde::de::DeserializeOwned,
    {
        let json_value = value.to_json();
        serde_json::from_value(json_value)
            .map_err(|e| Error::serialization(format!("Failed to convert DataValue: {}", e)))
    }
}

impl VectorValue {
    /// 创建新的向量值
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            dtype: DType::Float32,
        }
    }
    
    /// 从向量创建向量值
    pub fn from_vec(data: Vec<f32>) -> Self {
        Self {
            data,
            dtype: DType::Float32,
        }
    }
    
    /// 获取向量数据
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }
    
    /// 获取向量维度
    pub fn dimension(&self) -> usize {
        self.data.len()
    }
}

impl Default for VectorValue {
    fn default() -> Self {
        Self::new()
    }
}

// 实现DataValue与UnifiedValue的转换
impl From<DataValue> for UnifiedValue {
    fn from(value: DataValue) -> Self {
        match value {
            DataValue::Null => UnifiedValue::Scalar(ScalarValue::Int(0)),
            DataValue::Boolean(b) => UnifiedValue::Scalar(ScalarValue::Bool(b)),
            DataValue::Integer(i) => UnifiedValue::Scalar(ScalarValue::Int(i)),
            DataValue::Float(f) => UnifiedValue::Scalar(ScalarValue::Float(f)),
            DataValue::Number(n) => UnifiedValue::Scalar(ScalarValue::Float(n)),
            DataValue::String(s) => UnifiedValue::Text(s),
            DataValue::StringArray(arr) => {
                // 将字符串数组转换为复合数据
                let mut map = HashMap::new();
                for (i, s) in arr.into_iter().enumerate() {
                    map.insert(i.to_string(), UnifiedValue::Text(s));
                }
                UnifiedValue::Composite(map)
            },
            DataValue::Array(arr) => {
                // 检查是否为数值数组
                let all_numbers = arr.iter().all(|v| matches!(v, DataValue::Float(_) | DataValue::Integer(_) | DataValue::Number(_)));
                
                if all_numbers {
                    // 转换为向量
                    let mut data = Vec::new();
                    for item in arr {
                        match item {
                            DataValue::Float(f) => data.push(f as f32),
                            DataValue::Integer(i) => data.push(i as f32),
                            DataValue::Number(n) => data.push(n as f32),
                            _ => {} // 应该不会发生
                        }
                    }
                    UnifiedValue::Vector(VectorValue { data, dtype: DType::Float32 })
                } else {
                    // 转换为复合数据
                    let mut map = HashMap::new();
                    for (i, v) in arr.into_iter().enumerate() {
                        map.insert(i.to_string(), v.into());
                    }
                    UnifiedValue::Composite(map)
                }
            },
            DataValue::Object(obj) => {
                let mut map = HashMap::new();
                for (k, v) in obj {
                    map.insert(k, v.into());
                }
                UnifiedValue::Composite(map)
            },
            DataValue::Binary(data) => UnifiedValue::Binary(data),
            DataValue::DateTime(s) => UnifiedValue::Text(s),
            DataValue::Tensor(tensor) => {
                UnifiedValue::Tensor(TensorValue {
                    data: tensor.data,
                    shape: tensor.shape,
                    dtype: DType::Float32,
                })
            }
        }
    }
}

// 实现UnifiedValue到DataValue的转换
// 使用自定义命名的转换trait，避免与标准库TryFrom冲突
pub trait UnifiedToData {
    type Error;
    fn unified_to_data(self) -> std::result::Result<DataValue, Self::Error>;
}

impl UnifiedToData for UnifiedValue {
    type Error = Error;
    
    fn unified_to_data(self) -> Result<DataValue> {
        match self {
            UnifiedValue::Scalar(ScalarValue::Bool(b)) => Ok(DataValue::Boolean(b)),
            UnifiedValue::Scalar(ScalarValue::Int(i)) => Ok(DataValue::Integer(i)),
            UnifiedValue::Scalar(ScalarValue::Float(f)) => Ok(DataValue::Float(f)),
            UnifiedValue::Text(s) => Ok(DataValue::String(s)),
            UnifiedValue::Vector(v) => {
                let values: Vec<DataValue> = v.data.into_iter()
                    .map(|f| DataValue::Float(f as f64))
                    .collect();
                Ok(DataValue::Array(values))
            }
            UnifiedValue::Matrix(m) => {
                let mut rows = Vec::with_capacity(m.rows);
                for i in 0..m.rows {
                    let start = i * m.cols;
                    let end = start + m.cols;
                    let row: Vec<DataValue> = m.data[start..end].iter()
                        .map(|&f| DataValue::Float(f as f64))
                        .collect();
                    rows.push(DataValue::Array(row));
                }
                Ok(DataValue::Array(rows))
            }
            UnifiedValue::Tensor(t) => {
                // 使用 CoreTensorData 的默认值，并覆盖 shape 和 data 字段
                let mut tensor = CoreTensorData::default();
                tensor.shape = t.shape;
                tensor.data = t.data;
                Ok(DataValue::Tensor(tensor))
            }
            UnifiedValue::Binary(b) => Ok(DataValue::Binary(b)),
            UnifiedValue::Composite(c) => {
                let mut obj = HashMap::new();
                for (k, v) in c {
                    match v.unified_to_data() {
                        Ok(data_value) => { obj.insert(k, data_value); }
                        Err(_) => return Err(Error::invalid_data("无法转换复合值的字段")),
                    }
                }
                Ok(DataValue::Object(obj))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_value_conversions() {
        // 测试基本类型转换
        let bool_val = DataValue::from(true);
        assert_eq!(bool_val.as_bool(), Some(true));

        let int_val = DataValue::from(42i64);
        assert_eq!(int_val.as_integer(), Some(42));

        let float_val = DataValue::from(3.14f64);
        assert_eq!(float_val.as_number(), Some(3.14));

        let string_val = DataValue::from("hello");
        assert_eq!(string_val.as_str(), Some("hello"));

        // 测试复合类型
        let array_val = DataValue::from(vec![
            DataValue::from(1),
            DataValue::from(2),
            DataValue::from(3),
        ]);
        assert_eq!(array_val.as_array().unwrap().len(), 3);

        let mut map = HashMap::new();
        map.insert("key1".to_string(), DataValue::from("value1"));
        map.insert("key2".to_string(), DataValue::from(123));
        let obj_val = DataValue::from(map);
        assert_eq!(obj_val.as_object().unwrap().len(), 2);
        
        // 测试二进制数据
        let binary_val = DataValue::from_binary(vec![1, 2, 3, 4]);
        assert_eq!(binary_val.as_binary().unwrap(), &vec![1, 2, 3, 4]);
    }
}
