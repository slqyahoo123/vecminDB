// 数据转换模块，用于处理数据记录的转换、映射和过滤操作

use crate::error::Error;
use crate::data::{DataValue, schema::Schema};
use std::collections::HashMap;
use crate::data::schema::schema::{FieldDefinition, FieldType};
use serde::{Deserialize, Serialize};

/// 数据转换错误
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformError {
    /// 字段不存在
    FieldNotFound(String),
    /// 无效的字段类型
    InvalidFieldType { field: String, expected: String, found: String },
    /// 转换失败
    TransformFailed { field: String, message: String },
    /// 验证失败
    ValidationFailed { field: String, message: String },
    /// 一般性错误
    General(String),
}

impl std::fmt::Display for TransformError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformError::FieldNotFound(field) => write!(f, "字段未找到: {}", field),
            TransformError::InvalidFieldType { field, expected, found } => {
                write!(f, "字段 '{}' 类型无效. 预期: {}, 实际: {}", field, expected, found)
            }
            TransformError::TransformFailed { field, message } => {
                write!(f, "字段 '{}' 转换失败: {}", field, message)
            }
            TransformError::ValidationFailed { field, message } => {
                write!(f, "字段 '{}' 验证失败: {}", field, message)
            }
            TransformError::General(msg) => write!(f, "转换错误: {}", msg),
        }
    }
}

impl std::error::Error for TransformError {}

impl From<TransformError> for Error {
    fn from(err: TransformError) -> Self {
        Error::invalid_data(format!("{}", err))
    }
}

/// 数据转换结果
pub type TransformResult<T> = std::result::Result<T, TransformError>;

/// 数据转换器特性
pub trait Transformer: Send + Sync {
    /// 转换单个数据值
    fn transform_value(&self, value: &DataValue) -> TransformResult<DataValue>;
    
    /// 转换字段名称
    fn transform_field_name(&self, name: &str) -> String {
        name.to_string()
    }
    
    /// 获取转换器名称
    fn name(&self) -> &str;
    
    /// 获取转换器描述
    fn description(&self) -> &str {
        ""
    }
    
    /// 判断转换器是否应用于特定字段
    fn applies_to_field(&self, _field_def: &FieldDefinition) -> bool {
        true
    }
    
    /// 克隆转换器
    fn box_clone(&self) -> Box<dyn Transformer>;
}

impl Clone for Box<dyn Transformer> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

/// 数据记录的转换
#[derive(Clone, Default)]
pub struct RecordTransformer {
    /// 字段转换器映射
    field_transformers: HashMap<String, Vec<Box<dyn Transformer>>>,
    /// 全局转换器
    global_transformers: Vec<Box<dyn Transformer>>,
    /// 输入模式
    input_schema: Option<Schema>,
    /// 输出模式
    output_schema: Option<Schema>,
}

impl RecordTransformer {
    /// 创建新的记录转换器
    pub fn new() -> Self {
        Self {
            field_transformers: HashMap::new(),
            global_transformers: Vec::new(),
            input_schema: None,
            output_schema: None,
        }
    }
    
    /// 设置输入模式
    pub fn with_input_schema(mut self, schema: Schema) -> Self {
        self.input_schema = Some(schema);
        self
    }
    
    /// 设置输出模式
    pub fn with_output_schema(mut self, schema: Schema) -> Self {
        self.output_schema = Some(schema);
        self
    }
    
    /// 添加字段转换器
    pub fn add_field_transformer(
        &mut self,
        field_name: &str,
        transformer: Box<dyn Transformer>,
    ) -> &mut Self {
        self.field_transformers
            .entry(field_name.to_string())
            .or_insert_with(Vec::new)
            .push(transformer);
        self
    }
    
    /// 添加全局转换器
    pub fn add_global_transformer(&mut self, transformer: Box<dyn Transformer>) -> &mut Self {
        self.global_transformers.push(transformer);
        self
    }
    
    /// 转换数据记录
    pub fn transform(&self, record: &HashMap<String, DataValue>) -> TransformResult<HashMap<String, DataValue>> {
        let mut transformed = HashMap::new();
        
        // 应用字段转换器
        for (field, value) in record {
            let mut current_value = value.clone();
            
            // 应用特定字段的转换器
            if let Some(transformers) = self.field_transformers.get(field) {
                for transformer in transformers {
                    current_value = transformer.transform_value(&current_value)?;
                }
            }
            
            // 应用全局转换器
            for transformer in &self.global_transformers {
                if let Some(schema) = &self.input_schema {
                    if let Some(field_def) = schema.get_field(field) {
                        if transformer.applies_to_field(field_def) {
                            current_value = transformer.transform_value(&current_value)?;
                        }
                    }
                } else {
                    current_value = transformer.transform_value(&current_value)?;
                }
            }
            
            // 转换字段名称
            let mut transformed_field_name = field.clone();
            for transformer in &self.global_transformers {
                transformed_field_name = transformer.transform_field_name(&transformed_field_name);
            }
            
            transformed.insert(transformed_field_name, current_value);
        }
        
        // 验证输出模式
        if let Some(schema) = &self.output_schema {
            self.validate_against_schema(&transformed, schema)?;
        }
        
        Ok(transformed)
    }
    
    /// 根据模式验证数据记录
    fn validate_against_schema(
        &self,
        record: &HashMap<String, DataValue>,
        schema: &Schema,
    ) -> TransformResult<()> {
        // 检查所有必填字段
        for field_def in schema.fields() {
            // 基于字段名称检查是否存在必填字段
            if field_def.required && !record.contains_key(&field_def.name) {
                return Err(TransformError::FieldNotFound(field_def.name.clone()));
            }

            if let Some(value) = record.get(&field_def.name) {
                // 根据字段类型和 DataValue 的变体进行基本类型兼容性检查
                let compatible = match field_def.field_type() {
                    crate::data::schema::schema::FieldType::Numeric => {
                        matches!(value, DataValue::Integer(_) | DataValue::Float(_))
                    }
                    crate::data::schema::schema::FieldType::Text => {
                        matches!(value, DataValue::String(_) | DataValue::Text(_))
                    }
                    crate::data::schema::schema::FieldType::Boolean => {
                        matches!(value, DataValue::Boolean(_))
                    }
                    crate::data::schema::schema::FieldType::DateTime => {
                        matches!(value, DataValue::DateTime(_))
                    }
                    // 其他类型暂时不做严格校验
                    _ => true,
                };

                if !compatible {
                    return Err(TransformError::InvalidFieldType {
                        field: field_def.name.clone(),
                        expected: format!("{:?}", field_def.field_type()),
                        found: format!("{:?}", value),
                    });
                }
            }
        }
        
        Ok(())
    }
}

/// 字符串转换器
pub struct StringTransformer {
    /// 转换器名称
    name: String,
    /// 转换器描述
    description: String,
    /// 转换函数
    transform_fn: Box<dyn Fn(&str) -> TransformResult<String> + Send + Sync>,
}

impl StringTransformer {
    /// 创建新的字符串转换器
    pub fn new<F>(name: &str, description: &str, transform_fn: F) -> Self
    where
        F: Fn(&str) -> TransformResult<String> + Send + Sync + 'static,
    {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            transform_fn: Box::new(transform_fn),
        }
    }
    
    /// 创建一个转换为大写的转换器
    pub fn to_uppercase() -> Self {
        Self::new(
            "to_uppercase",
            "将字符串转换为大写",
            |s| Ok(s.to_uppercase()),
        )
    }
    
    /// 创建一个转换为小写的转换器
    pub fn to_lowercase() -> Self {
        Self::new(
            "to_lowercase",
            "将字符串转换为小写",
            |s| Ok(s.to_lowercase()),
        )
    }
    
    /// 创建一个去除首尾空白的转换器
    pub fn trim() -> Self {
        Self::new("trim", "去除字符串首尾空白", |s| Ok(s.trim().to_string()))
    }
    
    /// 创建一个替换文本的转换器
    pub fn replace(from: &str, to: &str) -> Self {
        let from_str = from.to_string();
        let to_str = to.to_string();
        Self::new(
            &format!("replace_{}_{}", from, to),
            &format!("替换 '{}' 为 '{}'", from, to),
            move |s| Ok(s.replace(&from_str, &to_str)),
        )
    }
    
    /// 创建一个截取子字符串的转换器
    pub fn substring(start: usize, length: Option<usize>) -> Self {
        Self::new(
            &format!(
                "substring_{}_{}",
                start,
                length.map_or("end".to_string(), |l| l.to_string())
            ),
            &format!(
                "截取从 {} 开始{}的子字符串",
                start,
                length.map_or("到结尾".to_string(), |l| format!("长度为 {}", l))
            ),
            move |s| {
                let end = length.map_or(s.len(), |l| (start + l).min(s.len()));
                if start > s.len() {
                    return Err(TransformError::TransformFailed {
                        field: "".to_string(),
                        message: format!("起始位置 {} 超出字符串长度 {}", start, s.len()),
                    });
                }
                Ok(s.get(start..end)
                    .ok_or_else(|| TransformError::TransformFailed {
                        field: "".to_string(),
                        message: "子字符串范围无效".to_string(),
                    })?
                    .to_string())
            },
        )
    }
}

impl Transformer for StringTransformer {
    fn transform_value(&self, value: &DataValue) -> TransformResult<DataValue> {
        match value {
            DataValue::String(s) => {
                let transformed = (self.transform_fn)(s)?;
                Ok(DataValue::String(transformed))
            }
            _ => Err(TransformError::InvalidFieldType {
                field: "".to_string(),
                expected: "String".to_string(),
                found: format!("{:?}", value),
            }),
        }
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    fn applies_to_field(&self, field_def: &FieldDefinition) -> bool {
        matches!(field_def.field_type(), FieldType::Text)
    }
    
    fn box_clone(&self) -> Box<dyn Transformer> {
        // 由于闭包不能 clone，我们创建一个新的实例，但只能复制名称和描述
        // 这会导致功能丢失，但在某些场景下可能可以接受
        // 更好的方法是存储足够的信息来重新创建闭包
        Box::new(StringTransformer {
            name: self.name.clone(),
            description: self.description.clone(),
            transform_fn: Box::new(|_| Err(TransformError::General("Cloned transformer not functional".to_string()))),
        })
    }
}

/// 数值转换器
pub struct NumberTransformer {
    /// 转换器名称
    name: String,
    /// 转换器描述
    description: String,
    /// 转换函数
    transform_fn: Box<dyn Fn(f64) -> TransformResult<f64> + Send + Sync>,
}

impl NumberTransformer {
    /// 创建新的数值转换器
    pub fn new<F>(name: &str, description: &str, transform_fn: F) -> Self
    where
        F: Fn(f64) -> TransformResult<f64> + Send + Sync + 'static,
    {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            transform_fn: Box::new(transform_fn),
        }
    }
    
    /// 创建一个乘法转换器
    pub fn multiply(factor: f64) -> Self {
        Self::new(
            &format!("multiply_{}", factor),
            &format!("将数值乘以 {}", factor),
            move |n| Ok(n * factor),
        )
    }
    
    /// 创建一个除法转换器
    pub fn divide(divisor: f64) -> Self {
        Self::new(
            &format!("divide_{}", divisor),
            &format!("将数值除以 {}", divisor),
            move |n| {
                if divisor == 0.0 {
                    return Err(TransformError::TransformFailed {
                        field: "".to_string(),
                        message: "除数不能为零".to_string(),
                    });
                }
                Ok(n / divisor)
            },
        )
    }
    
    /// 创建一个加法转换器
    pub fn add(addend: f64) -> Self {
        Self::new(
            &format!("add_{}", addend),
            &format!("将数值加上 {}", addend),
            move |n| Ok(n + addend),
        )
    }
    
    /// 创建一个减法转换器
    pub fn subtract(subtrahend: f64) -> Self {
        Self::new(
            &format!("subtract_{}", subtrahend),
            &format!("将数值减去 {}", subtrahend),
            move |n| Ok(n - subtrahend),
        )
    }
    
    /// 创建一个幂运算转换器
    pub fn power(exponent: f64) -> Self {
        Self::new(
            &format!("power_{}", exponent),
            &format!("将数值的 {} 次幂", exponent),
            move |n| Ok(n.powf(exponent)),
        )
    }
    
    /// 创建一个取绝对值的转换器
    pub fn abs() -> Self {
        Self::new("abs", "取数值的绝对值", |n| Ok(n.abs()))
    }
    
    /// 创建一个四舍五入的转换器
    pub fn round() -> Self {
        Self::new("round", "四舍五入到最接近的整数", |n| Ok(n.round()))
    }
    
    /// 创建一个向下取整的转换器
    pub fn floor() -> Self {
        Self::new("floor", "向下取整", |n| Ok(n.floor()))
    }
    
    /// 创建一个向上取整的转换器
    pub fn ceil() -> Self {
        Self::new("ceil", "向上取整", |n| Ok(n.ceil()))
    }
}

impl Transformer for NumberTransformer {
    fn transform_value(&self, value: &DataValue) -> TransformResult<DataValue> {
        match value {
            DataValue::Number(n) => {
                let transformed = (self.transform_fn)(*n)?;
                Ok(DataValue::Number(transformed))
            }
            DataValue::Integer(i) => {
                let transformed = (self.transform_fn)(*i as f64)?;
                Ok(DataValue::Number(transformed))
            }
            _ => Err(TransformError::InvalidFieldType {
                field: "".to_string(),
                expected: "Number".to_string(),
                found: format!("{:?}", value),
            }),
        }
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    fn applies_to_field(&self, field_def: &FieldDefinition) -> bool {
        matches!(field_def.field_type(), FieldType::Numeric)
    }
    
    fn box_clone(&self) -> Box<dyn Transformer> {
        // 由于闭包不能 clone，我们创建一个新的实例，但只能复制名称和描述
        Box::new(NumberTransformer {
            name: self.name.clone(),
            description: self.description.clone(),
            transform_fn: Box::new(|_| Err(TransformError::General("Cloned transformer not functional".to_string()))),
        })
    }
}

/// 日期时间转换器
pub struct DateTimeTransformer {
    /// 转换器名称
    name: String,
    /// 转换器描述
    description: String,
    /// 转换函数
    transform_fn: Box<dyn Fn(&chrono::DateTime<chrono::Utc>) -> TransformResult<chrono::DateTime<chrono::Utc>> + Send + Sync>,
}

impl DateTimeTransformer {
    /// 创建新的日期时间转换器
    pub fn new<F>(name: &str, description: &str, transform_fn: F) -> Self
    where
        F: Fn(&chrono::DateTime<chrono::Utc>) -> TransformResult<chrono::DateTime<chrono::Utc>> + Send + Sync + 'static,
    {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            transform_fn: Box::new(transform_fn),
        }
    }
    
    /// 创建一个添加天数的转换器
    pub fn add_days(days: i64) -> Self {
        Self::new(
            &format!("add_days_{}", days),
            &format!("添加 {} 天到日期", days),
            move |dt| {
                Ok(*dt + chrono::Duration::days(days))
            },
        )
    }
    
    /// 创建一个添加小时的转换器
    pub fn add_hours(hours: i64) -> Self {
        Self::new(
            &format!("add_hours_{}", hours),
            &format!("添加 {} 小时到日期时间", hours),
            move |dt| {
                Ok(*dt + chrono::Duration::hours(hours))
            },
        )
    }
    
    /// 创建一个添加分钟的转换器
    pub fn add_minutes(minutes: i64) -> Self {
        Self::new(
            &format!("add_minutes_{}", minutes),
            &format!("添加 {} 分钟到日期时间", minutes),
            move |dt| {
                Ok(*dt + chrono::Duration::minutes(minutes))
            },
        )
    }
    
    /// 创建一个添加秒的转换器
    pub fn add_seconds(seconds: i64) -> Self {
        Self::new(
            &format!("add_seconds_{}", seconds),
            &format!("添加 {} 秒到日期时间", seconds),
            move |dt| {
                Ok(*dt + chrono::Duration::seconds(seconds))
            },
        )
    }
    
    /// 创建一个设置为当天开始的转换器
    pub fn start_of_day() -> Self {
        Self::new(
            "start_of_day",
            "设置为当天的开始时间",
            |dt| {
                let naive = dt.naive_utc().date().and_hms_opt(0, 0, 0).unwrap();
                Ok(chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(naive, chrono::Utc))
            },
        )
    }
    
    /// 创建一个设置为当天结束的转换器
    pub fn end_of_day() -> Self {
        Self::new(
            "end_of_day",
            "设置为当天的结束时间",
            |dt| {
                let naive = dt.naive_utc().date().and_hms_opt(23, 59, 59).unwrap();
                Ok(chrono::DateTime::<chrono::Utc>::from_naive_utc_and_offset(naive, chrono::Utc))
            },
        )
    }
    
    /// 创建一个格式化为字符串的转换器
    pub fn format(format_str: &str) -> StringTransformer {
        let format_string = format_str.to_string();
        StringTransformer::new(
            &format!("datetime_format_{}", format_str),
            &format!("将日期时间格式化为 '{}'", format_str),
            move |s| {
                let dt = chrono::DateTime::parse_from_rfc3339(s)
                    .map_err(|e| TransformError::TransformFailed {
                        field: "".to_string(),
                        message: format!("无法解析日期时间: {}", e),
                    })?;
                Ok(dt.format(&format_string).to_string())
            },
        )
    }
}

impl Transformer for DateTimeTransformer {
    fn transform_value(&self, value: &DataValue) -> TransformResult<DataValue> {
        match value {
            DataValue::DateTime(s) => {
                // DataValue::DateTime 内部存储为 RFC3339 字符串
                let dt = chrono::DateTime::parse_from_rfc3339(s)
                    .map_err(|e| TransformError::TransformFailed {
                        field: "".to_string(),
                        message: format!("无法解析日期时间字符串: {}", e),
                    })?;
                let utc_dt = dt.with_timezone(&chrono::Utc);
                let transformed = (self.transform_fn)(&utc_dt)?;
                Ok(DataValue::DateTime(transformed.to_rfc3339()))
            }
            DataValue::String(s) => {
                let dt = chrono::DateTime::parse_from_rfc3339(s)
                    .map_err(|e| TransformError::TransformFailed {
                        field: "".to_string(),
                        message: format!("无法解析日期时间字符串: {}", e),
                    })?;
                let utc_dt = dt.with_timezone(&chrono::Utc);
                let transformed = (self.transform_fn)(&utc_dt)?;
                Ok(DataValue::DateTime(transformed.to_rfc3339()))
            }
            _ => Err(TransformError::InvalidFieldType {
                field: "".to_string(),
                expected: "DateTime or String".to_string(),
                found: format!("{:?}", value),
            }),
        }
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn description(&self) -> &str {
        &self.description
    }
    
    fn applies_to_field(&self, field_def: &FieldDefinition) -> bool {
        matches!(field_def.field_type(), FieldType::DateTime | FieldType::Text)
    }
    
    fn box_clone(&self) -> Box<dyn Transformer> {
        // 由于闭包不能 clone，我们创建一个新的实例，但只能复制名称和描述
        Box::new(DateTimeTransformer {
            name: self.name.clone(),
            description: self.description.clone(),
            transform_fn: Box::new(|_| Err(TransformError::General("Cloned transformer not functional".to_string()))),
        })
    }
}

/// 数据转换器（RecordTransformer的别名）
pub type DataTransformer = RecordTransformer;



#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_string_transformer() {
        let transformer = StringTransformer::to_uppercase();
        let value = DataValue::String("hello".to_string());
        let result = transformer.transform_value(&value).unwrap();
        assert_eq!(result, DataValue::String("HELLO".to_string()));
        
        let transformer = StringTransformer::trim();
        let value = DataValue::String("  hello  ".to_string());
        let result = transformer.transform_value(&value).unwrap();
        assert_eq!(result, DataValue::String("hello".to_string()));
        
        let transformer = StringTransformer::replace("l", "x");
        let value = DataValue::String("hello".to_string());
        let result = transformer.transform_value(&value).unwrap();
        assert_eq!(result, DataValue::String("hexxo".to_string()));
    }
    
    #[test]
    fn test_number_transformer() {
        let transformer = NumberTransformer::multiply(2.0);
        let value = DataValue::Number(5.0);
        let result = transformer.transform_value(&value).unwrap();
        assert_eq!(result, DataValue::Number(10.0));
        
        let transformer = NumberTransformer::add(3.0);
        let value = DataValue::Number(5.0);
        let result = transformer.transform_value(&value).unwrap();
        assert_eq!(result, DataValue::Number(8.0));
        
        let transformer = NumberTransformer::round();
        let value = DataValue::Number(5.7);
        let result = transformer.transform_value(&value).unwrap();
        assert_eq!(result, DataValue::Number(6.0));
    }
    
    #[test]
    fn test_record_transformer() {
        let mut transformer = RecordTransformer::new();
        
        transformer.add_field_transformer(
            "name",
            Box::new(StringTransformer::to_uppercase()),
        );
        transformer.add_field_transformer(
            "age",
            Box::new(NumberTransformer::add(1.0)),
        );
        
        let mut record = HashMap::new();
        record.insert("name".to_string(), DataValue::String("john".to_string()));
        record.insert("age".to_string(), DataValue::Number(30.0));
        
        let result = transformer.transform(&record).unwrap();
        
        assert_eq!(
            result.get("name").unwrap(),
            &DataValue::String("JOHN".to_string())
        );
        assert_eq!(result.get("age").unwrap(), &DataValue::Number(31.0));
    }
    
    #[test]
    fn test_schema_validation() {
        let mut schema = Schema::new("test");
        
        schema.add_field(crate::data::schema::schema::FieldDefinition {
            name: "name".to_string(),
            field_type: crate::data::schema::schema::FieldType::Text,
            data_type: None,
            required: true,
            nullable: false,
            primary_key: false,
            foreign_key: None,
            description: None,
            default_value: None,
            constraints: None,
            metadata: HashMap::new(),
        }).unwrap();
        
        schema.add_field(crate::data::schema::schema::FieldDefinition {
            name: "age".to_string(),
            field_type: crate::data::schema::schema::FieldType::Numeric,
            data_type: None,
            required: true,
            nullable: false,
            primary_key: false,
            foreign_key: None,
            description: None,
            default_value: None,
            constraints: Some(crate::data::schema::schema::FieldConstraints {
                min_value: Some(0.0),
                max_value: Some(150.0),
                min_length: None,
                max_length: None,
                pattern: None,
                allowed_values: None,
                unique: false,
            }),
            metadata: HashMap::new(),
        }).unwrap();
        
        let mut transformer = RecordTransformer::new()
            .with_output_schema(schema);
        
        transformer.add_field_transformer(
            "name",
            Box::new(StringTransformer::to_uppercase()),
        );
        
        // 缺少必填字段的记录
        let mut record = HashMap::new();
        record.insert("name".to_string(), DataValue::String("john".to_string()));
        
        let result = transformer.transform(&record);
        assert!(result.is_err());
        
        // 字段类型错误的记录
        let mut record = HashMap::new();
        record.insert("name".to_string(), DataValue::String("john".to_string()));
        record.insert("age".to_string(), DataValue::String("thirty".to_string()));
        
        let result = transformer.transform(&record);
        assert!(result.is_err());
        
        // 有效记录
        let mut record = HashMap::new();
        record.insert("name".to_string(), DataValue::String("john".to_string()));
        record.insert("age".to_string(), DataValue::Number(30.0));
        
        let result = transformer.transform(&record);
        assert!(result.is_ok());
        let transformed = result.unwrap();
        assert_eq!(
            transformed.get("name").unwrap(),
            &DataValue::String("JOHN".to_string())
        );
    }
} 