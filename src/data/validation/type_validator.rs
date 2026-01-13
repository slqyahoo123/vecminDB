use crate::error::Result;
use crate::data::schema::{DataSchema, FieldType};
use crate::data::schema::schema::FieldType as SchemaFieldType;
use crate::data::record::Record;
use crate::data::value::DataValue;
use std::collections::{HashSet, HashMap};
 

use super::{DataValidator, ValidationError, ValidationWarning, ValidationType, ValidationLocation};
use crate::core::interfaces::ValidationResult;

// 辅助函数：将SchemaFieldType转换为FieldType
fn convert_schema_field_type(schema_type: &SchemaFieldType) -> FieldType {
    match schema_type {
        SchemaFieldType::Numeric => FieldType::Numeric,
        SchemaFieldType::Text => FieldType::Text,
        SchemaFieldType::Categorical => FieldType::Categorical,
        SchemaFieldType::Boolean => FieldType::Boolean,
        SchemaFieldType::DateTime => FieldType::DateTime,
        SchemaFieldType::Image => FieldType::Image,
        SchemaFieldType::Audio => FieldType::Audio,
        SchemaFieldType::Video => FieldType::Video,
        SchemaFieldType::Array(inner) => FieldType::Array(Box::new(convert_schema_field_type(inner))),
        SchemaFieldType::Object(_) => FieldType::Object,
        SchemaFieldType::Custom(name) => FieldType::Custom(name.clone()),
    }
}

// 辅助函数：添加错误到ValidationResult
fn add_error_to_result(result: &mut ValidationResult, error: ValidationError) {
    result.errors.push(error.message);
    result.is_valid = false;
}

// 辅助函数：添加警告到ValidationResult
fn add_warning_to_result(result: &mut ValidationResult, warning: ValidationWarning) {
    result.warnings.push(warning.message);
}

/// 类型验证器
pub struct TypeValidator {
    /// 验证器名称
    name: String,
    /// 数据模式
    schema: DataSchema,
    /// 需要验证的字段列表
    fields_to_validate: Option<HashSet<String>>,
    /// 是否验证所有字段
    validate_all_fields: bool,
    /// 是否严格模式（类型必须完全匹配）
    strict_mode: bool,
    /// 是否忽略未在模式中定义的字段
    ignore_undefined_fields: bool,
}

impl TypeValidator {
    /// 创建新的类型验证器
    pub fn new<S: Into<String>>(name: S, schema: DataSchema) -> Self {
        Self {
            name: name.into(),
            schema,
            fields_to_validate: None,
            validate_all_fields: true,
            strict_mode: false,
            ignore_undefined_fields: true,
        }
    }
    
    /// 添加需要验证的字段
    pub fn add_field<S: Into<String>>(mut self, field_name: S) -> Self {
        let field_name = field_name.into();
        
        let fields = self.fields_to_validate.get_or_insert_with(HashSet::new);
        fields.insert(field_name);
        
        self.validate_all_fields = false;
        self
    }
    
    /// 设置是否验证所有字段
    pub fn validate_all_fields(mut self, validate_all: bool) -> Self {
        self.validate_all_fields = validate_all;
        self
    }
    
    /// 设置是否严格模式
    pub fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }
    
    /// 设置是否忽略未定义的字段
    pub fn ignore_undefined(mut self, ignore: bool) -> Self {
        self.ignore_undefined_fields = ignore;
        self
    }
    
    /// 验证数据值是否符合预期类型
    fn validate_value(&self, value: &DataValue, expected_type: &FieldType, field_name: &str) -> Result<ValidationResult> {
        let mut result = ValidationResult::success();
        
        match expected_type {
            FieldType::Boolean => {
                if !matches!(value, DataValue::Boolean(_)) {
                    // 非严格模式下，尝试兼容处理
                    if !self.strict_mode {
                        match value {
                            DataValue::String(s) => {
                                let s_lower = s.to_lowercase();
                                if s_lower != "true" && s_lower != "false" && 
                                   s_lower != "yes" && s_lower != "no" && 
                                   s_lower != "1" && s_lower != "0" {
                                    add_error_to_result(&mut result, ValidationError::new(
                                        "type_mismatch".to_string(),
                                        format!("字段 '{}' 值类型不匹配: 期望 Boolean, 实际为 {:?}", field_name, value)
                                    ).with_field(field_name.to_string()));
                                } else {
                                    add_warning_to_result(&mut result, ValidationWarning::new(
                                        "type_conversion".to_string(),
                                        format!("字段 '{}' 值类型已自动转换: 从 String 到 Boolean", field_name)
                                    ).with_field(field_name.to_string()));
                                }
                            },
                            DataValue::Integer(n) => {
                                if *n != 0 && *n != 1 {
                                    add_error_to_result(&mut result, ValidationError::new(
                                        "type_mismatch".to_string(),
                                        format!("字段 '{}' 值类型不匹配: 期望 Boolean, 实际为 {:?}", field_name, value)
                                    ).with_field(field_name.to_string()));
                                } else {
                                    add_warning_to_result(&mut result, ValidationWarning::new(
                                        "type_conversion".to_string(),
                                        format!("字段 '{}' 值类型已自动转换: 从 Integer 到 Boolean", field_name)
                                    ).with_field(field_name.to_string()));
                                }
                            },
                            _ => {
                                add_error_to_result(&mut result, ValidationError::new(
                                    "type_mismatch".to_string(),
                                    format!("字段 '{}' 值类型不匹配: 期望 Boolean, 实际为 {:?}", field_name, value)
                                ).with_field(field_name.to_string()));
                            }
                        }
                    } else {
                        add_error_to_result(&mut result, ValidationError::new(
                            "type_mismatch".to_string(),
                            format!("字段 '{}' 值类型不匹配: 期望 Boolean, 实际为 {:?}", field_name, value)
                        ).with_field(field_name.to_string()));
                    }
                }
            },
            FieldType::Integer => {
                if !matches!(value, DataValue::Integer(_)) {
                    // 非严格模式下，尝试兼容处理
                    if !self.strict_mode {
                        match value {
                            DataValue::Float(f) => {
                                // 检查是否为整数值
                                if *f == f.trunc() {
                                    add_warning_to_result(&mut result, ValidationWarning::new(
                                        "type_conversion".to_string(),
                                        format!("字段 '{}' 值类型已自动转换: 从 Float 到 Integer", field_name)
                                    ).with_field(field_name.to_string()));
                                } else {
                                    add_error_to_result(&mut result, ValidationError::new(
                                        "type_mismatch".to_string(),
                                        format!("字段 '{}' 值类型不匹配: 期望 Integer, 实际为 Float {}", field_name, f)
                                    ).with_field(field_name.to_string()));
                                }
                            },
                            DataValue::String(s) => {
                                // 尝试解析为整数
                                if let Ok(_) = s.parse::<i64>() {
                                    add_warning_to_result(&mut result, ValidationWarning::new(
                                        "type_conversion".to_string(),
                                        format!("字段 '{}' 值类型已自动转换: 从 String 到 Integer", field_name)
                                    ).with_field(field_name.to_string()));
                                } else {
                                    add_error_to_result(&mut result, ValidationError::new(
                                        "type_mismatch".to_string(),
                                        format!("字段 '{}' 值类型不匹配: 期望 Integer, 实际为 String '{}'", field_name, s)
                                    ).with_field(field_name.to_string()));
                                }
                            },
                            _ => {
                                add_error_to_result(&mut result, ValidationError::new(
                                    "type_mismatch".to_string(),
                                    format!("字段 '{}' 值类型不匹配: 期望 Integer, 实际为 {:?}", field_name, value)
                                ).with_field(field_name.to_string()));
                            }
                        }
                    } else {
                        add_error_to_result(&mut result, ValidationError::new(
                            "type_mismatch".to_string(),
                            format!("字段 '{}' 值类型不匹配: 期望 Integer, 实际为 {:?}", field_name, value)
                        ).with_field(field_name.to_string()));
                    }
                }
            },
            FieldType::Float => {
                if !matches!(value, DataValue::Float(_)) {
                    // 非严格模式下，尝试兼容处理
                    if !self.strict_mode {
                        match value {
                            DataValue::Integer(i) => {
                                add_warning_to_result(&mut result, ValidationWarning::new(
                                    "type_conversion".to_string(),
                                    format!("字段 '{}' 值类型已自动转换: 从 Integer 到 Float", field_name)
                                ).with_field(field_name.to_string()));
                            },
                            DataValue::String(s) => {
                                // 尝试解析为浮点数
                                if let Ok(_) = s.parse::<f64>() {
                                    add_warning_to_result(&mut result, ValidationWarning::new(
                                        "type_conversion".to_string(),
                                        format!("字段 '{}' 值类型已自动转换: 从 String 到 Float", field_name)
                                    ).with_field(field_name.to_string()));
                                } else {
                                    add_error_to_result(&mut result, ValidationError::new(
                                        "type_mismatch".to_string(),
                                        format!("字段 '{}' 值类型不匹配: 期望 Float, 实际为 String '{}'", field_name, s)
                                    ).with_field(field_name.to_string()));
                                }
                            },
                            _ => {
                                add_error_to_result(&mut result, ValidationError::new(
                                    "type_mismatch".to_string(),
                                    format!("字段 '{}' 值类型不匹配: 期望 Float, 实际为 {:?}", field_name, value)
                                ).with_field(field_name.to_string()));
                            }
                        }
                    } else {
                        add_error_to_result(&mut result, ValidationError::new(
                            "type_mismatch".to_string(),
                            format!("字段 '{}' 值类型不匹配: 期望 Float, 实际为 {:?}", field_name, value)
                        ).with_field(field_name.to_string()));
                    }
                }
            },
            FieldType::String => {
                if !matches!(value, DataValue::String(_)) {
                    // 非严格模式下，几乎所有类型都可以转为字符串
                    if !self.strict_mode {
                        add_warning_to_result(&mut result, ValidationWarning::new(
                            "type_conversion".to_string(),
                            format!("字段 '{}' 值类型已自动转换: 从 {:?} 到 String", field_name, value)
                        ).with_field(field_name.to_string()));
                    } else {
                        add_error_to_result(&mut result, ValidationError::new(
                            "type_mismatch".to_string(),
                            format!("字段 '{}' 值类型不匹配: 期望 String, 实际为 {:?}", field_name, value)
                        ).with_field(field_name.to_string()));
                    }
                }
            },
            FieldType::Date => {
                // 当前 DataValue 仅提供 DateTime 字段，使用 ISO 字符串保存日期/时间
                if !matches!(value, DataValue::DateTime(_)) {
                    // 非严格模式下，尝试从字符串转换
                    if !self.strict_mode && matches!(value, DataValue::String(_)) {
                        add_warning_to_result(&mut result, ValidationWarning::new(
                            "type_conversion".to_string(),
                            format!("字段 '{}' 值类型已自动转换: 从 String 到 Date", field_name)
                        ).with_field(field_name.to_string()));
                    } else {
                        add_error_to_result(&mut result, ValidationError::new(
                            "type_mismatch".to_string(),
                            format!("字段 '{}' 值类型不匹配: 期望 Date, 实际为 {:?}", field_name, value)
                        ).with_field(field_name.to_string()));
                    }
                }
            },
            FieldType::DateTime => {
                if !matches!(value, DataValue::DateTime(_)) {
                    // 非严格模式下，尝试从字符串或整数转换
                    if !self.strict_mode && (matches!(value, DataValue::String(_)) || matches!(value, DataValue::Integer(_))) {
                        add_warning_to_result(&mut result, ValidationWarning::new(
                            "type_conversion".to_string(),
                            format!("字段 '{}' 值类型已自动转换: 从 {:?} 到 DateTime", field_name, value)
                        ).with_field(field_name.to_string()));
                    } else {
                        add_error_to_result(&mut result, ValidationError::new(
                            "type_mismatch".to_string(),
                            format!("字段 '{}' 值类型不匹配: 期望 DateTime, 实际为 {:?}", field_name, value)
                        ).with_field(field_name.to_string()));
                    }
                }
            },
            FieldType::Array(_) => {
                if !matches!(value, DataValue::Array(_)) {
                    add_error_to_result(&mut result, ValidationError::new(
                        "type_mismatch".to_string(),
                        format!("字段 '{}' 值类型不匹配: 期望 Array, 实际为 {:?}", field_name, value)
                    ).with_field(field_name.to_string()));
                }
            },
            FieldType::Object => {
                if !matches!(value, DataValue::Object(_)) {
                    add_error_to_result(&mut result, ValidationError::new(
                        "type_mismatch".to_string(),
                        format!("字段 '{}' 值类型不匹配: 期望 Object, 实际为 {:?}", field_name, value)
                    ).with_field(field_name.to_string()));
                }
            },
        }
        
        Ok(result)
    }
}

impl DataValidator for TypeValidator {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn validation_type(&self) -> ValidationType {
        ValidationType::Type
    }
    
    fn validate_record(&self, record: &Record, schema: &DataSchema, index: usize) -> Result<ValidationResult> {
        let mut result = ValidationResult::success();
        
        // 验证记录中的每个字段
        for field in schema.fields() {
            // 检查是否需要验证此字段
            if !self.validate_all_fields {
                if let Some(fields) = &self.fields_to_validate {
                    if !fields.contains(&field.name) {
                        continue;
                    }
                }
            }
            
            // 获取记录中的值
            if let Some(value) = record.get_field(&field.name) {
                // 转换为DataValue
                let data_value = value.to_data_value()?;
                // 验证类型
                let field_type = convert_schema_field_type(&field.field_type);
                let field_result = self.validate_value(&data_value, &field_type, &field.name)?;
                
                // 合并结果
                result.errors.extend(field_result.errors);
                result.warnings.extend(field_result.warnings);
                if !field_result.is_valid {
                    result.is_valid = false;
                }
            } else if !field.nullable {
                // 字段不允许为空但没有值
                add_error_to_result(&mut result, ValidationError::new(
                    "missing_required_field".to_string(),
                    format!("缺少必填字段: '{}'", field.name)
                ).with_field(field.name.clone())
                 .with_location(ValidationLocation {
                     line: Some(index),
                     column: None,
                     offset: None,
                     context: None,
                 }));
            }
        }
        
        // 检查记录中是否存在模式中未定义的字段
        if !self.ignore_undefined_fields {
            let schema_field_names: HashSet<String> = schema.fields()
                .iter()
                .map(|f| f.name.clone())
                .collect();
            
            for field_name in record.field_names() {
                if !schema_field_names.contains(&field_name) {
                    add_warning_to_result(&mut result, ValidationWarning::new(
                        "undefined_field".to_string(),
                        format!("模式中未定义的字段: '{}'", field_name)
                    ).with_field(field_name.clone())
                     .with_location(ValidationLocation {
                         line: Some(index),
                         column: None,
                         offset: None,
                         context: None,
                     }));
                }
            }
        }
        
        // 更新记录无效计数
        if !result.is_valid {
            result.metadata.insert("invalid_records".to_string(), "1".to_string());
        }
        
        Ok(result)
    }
    
    fn get_schema(&self) -> Result<DataSchema> {
        Ok(self.schema.clone())
    }
}

impl super::Validator for TypeValidator {
    fn validate(&self, data: &HashMap<String, DataValue>) -> ValidationResult {
        // 创建一个临时Record用于验证
        let mut record = Record::new();
        for (key, value) in data {
            record.add_field(key, crate::data::record::Value::from_data_value(value.clone()));
        }
        
        // 使用DataValidator的validate_record方法
        match self.validate_record(&record, &self.schema, 0) {
            Ok(result) => result,
            Err(_) => ValidationResult::failure(vec!["类型验证失败".to_string()]),
        }
    }
    
    fn validation_type(&self) -> ValidationType {
        ValidationType::Type
    }
} 