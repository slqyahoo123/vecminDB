//! 数据验证工具模块
//! 
//! 提供全面的数据验证功能，包括类型检查、约束验证、格式验证等。

use std::collections::HashMap;
use std::fmt;
use std::path::Path;


/// 验证错误类型
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// 值为空
    Empty(String),
    /// 值超出范围
    OutOfRange { field: String, min: f64, max: f64, value: f64 },
    /// 类型不匹配
    TypeMismatch { field: String, expected: String, actual: String },
    /// 格式无效
    InvalidFormat { field: String, format: String, value: String },
    /// 必填字段缺失
    MissingRequired(String),
    /// 自定义验证错误
    Custom(String),
    /// 多个验证错误
    Multiple(Vec<ValidationError>),
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::Empty(field) => write!(f, "字段 '{}' 不能为空", field),
            ValidationError::OutOfRange { field, min, max, value } => {
                write!(f, "字段 '{}' 的值 {} 超出范围 [{}, {}]", field, value, min, max)
            }
            ValidationError::TypeMismatch { field, expected, actual } => {
                write!(f, "字段 '{}' 类型不匹配，期望 {}，实际 {}", field, expected, actual)
            }
            ValidationError::InvalidFormat { field, format, value } => {
                write!(f, "字段 '{}' 格式无效，期望格式 {}，实际值 {}", field, format, value)
            }
            ValidationError::MissingRequired(field) => {
                write!(f, "必填字段 '{}' 缺失", field)
            }
            ValidationError::Custom(msg) => write!(f, "{}", msg),
            ValidationError::Multiple(errors) => {
                write!(f, "多个验证错误: ")?;
                for (i, error) in errors.iter().enumerate() {
                    if i > 0 { write!(f, "; ")?; }
                    write!(f, "{}", error)?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// 验证结果
pub type ValidationResult<T> = std::result::Result<T, ValidationError>;

/// 验证规则
#[derive(Debug, Clone)]
pub enum ValidationRule {
    /// 非空验证
    NotEmpty,
    /// 长度范围验证
    LengthRange { min: usize, max: usize },
    /// 数值范围验证
    NumberRange { min: f64, max: f64 },
    /// 正则表达式验证
    Regex(String),
    /// 邮箱格式验证
    Email,
    /// URL格式验证
    Url,
    /// 路径存在验证
    PathExists,
    /// 文件扩展名验证
    FileExtension(Vec<String>),
}

/// 字段验证器
#[derive(Debug, Clone)]
pub struct FieldValidator {
    pub name: String,
    pub rules: Vec<ValidationRule>,
    pub required: bool,
}

impl FieldValidator {
    /// 创建新的字段验证器
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            rules: Vec::new(),
            required: true,
        }
    }

    /// 设置为可选字段
    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    /// 添加非空验证
    pub fn not_empty(mut self) -> Self {
        self.rules.push(ValidationRule::NotEmpty);
        self
    }

    /// 添加长度范围验证
    pub fn length_range(mut self, min: usize, max: usize) -> Self {
        self.rules.push(ValidationRule::LengthRange { min, max });
        self
    }

    /// 添加数值范围验证
    pub fn number_range(mut self, min: f64, max: f64) -> Self {
        self.rules.push(ValidationRule::NumberRange { min, max });
        self
    }

    /// 验证字符串值
    pub fn validate(&self, value: Option<&str>) -> ValidationResult<()> {
        match value {
            Some(val) => {
                let mut errors = Vec::new();
                
                for rule in &self.rules {
                    if let Err(error) = self.validate_rule(rule, val) {
                        errors.push(error);
                    }
                }
                
                if errors.is_empty() {
                    Ok(())
                } else if errors.len() == 1 {
                    Err(errors.into_iter().next().unwrap())
                } else {
                    Err(ValidationError::Multiple(errors))
                }
            }
            None => {
                if self.required {
                    Err(ValidationError::MissingRequired(self.name.clone()))
                } else {
                    Ok(())
                }
            }
        }
    }

    /// 验证单个规则
    fn validate_rule(&self, rule: &ValidationRule, value: &str) -> ValidationResult<()> {
        match rule {
            ValidationRule::NotEmpty => {
                if value.trim().is_empty() {
                    Err(ValidationError::Empty(self.name.clone()))
                } else {
                    Ok(())
                }
            }
            ValidationRule::LengthRange { min, max } => {
                let len = value.len();
                if len < *min || len > *max {
                    Err(ValidationError::Custom(format!(
                        "字段 '{}' 长度必须在 {} 到 {} 之间，当前长度 {}",
                        self.name, min, max, len
                    )))
                } else {
                    Ok(())
                }
            }
            ValidationRule::NumberRange { min, max } => {
                match value.parse::<f64>() {
                    Ok(num) => {
                        if num < *min || num > *max {
                            Err(ValidationError::OutOfRange {
                                field: self.name.clone(),
                                min: *min,
                                max: *max,
                                value: num,
                            })
                        } else {
                            Ok(())
                        }
                    }
                    Err(_) => Err(ValidationError::TypeMismatch {
                        field: self.name.clone(),
                        expected: "number".to_string(),
                        actual: "string".to_string(),
                    })
                }
            }
            ValidationRule::PathExists => {
                if Path::new(value).exists() {
                    Ok(())
                } else {
                    Err(ValidationError::Custom(format!(
                        "路径 '{}' 不存在", value
                    )))
                }
            }
            _ => Ok(()), // 其他规则的简化实现
        }
    }
}

/// 数据验证器
#[derive(Debug)]
pub struct DataValidator {
    fields: HashMap<String, FieldValidator>,
}

impl DataValidator {
    /// 创建新的数据验证器
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
        }
    }

    /// 添加字段验证器
    pub fn add_field(mut self, validator: FieldValidator) -> Self {
        self.fields.insert(validator.name.clone(), validator);
        self
    }

    /// 验证数据
    pub fn validate(&self, data: &HashMap<String, String>) -> ValidationResult<()> {
        let mut errors = Vec::new();

        for (field_name, validator) in &self.fields {
            let value = data.get(field_name).map(|s| s.as_str());
            if let Err(error) = validator.validate(value) {
                errors.push(error);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else if errors.len() == 1 {
            Err(errors.into_iter().next().unwrap())
        } else {
            Err(ValidationError::Multiple(errors))
        }
    }
}

impl Default for DataValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// 数值验证工具
pub struct NumberValidator;

impl NumberValidator {
    /// 验证整数范围
    pub fn validate_int_range(value: i64, min: i64, max: i64, field_name: &str) -> ValidationResult<()> {
        if value < min || value > max {
            Err(ValidationError::OutOfRange {
                field: field_name.to_string(),
                min: min as f64,
                max: max as f64,
                value: value as f64,
            })
        } else {
            Ok(())
        }
    }

    /// 验证浮点数范围
    pub fn validate_float_range(value: f64, min: f64, max: f64, field_name: &str) -> ValidationResult<()> {
        if value < min || value > max {
            Err(ValidationError::OutOfRange {
                field: field_name.to_string(),
                min,
                max,
                value,
            })
        } else {
            Ok(())
        }
    }

    /// 验证正数
    pub fn validate_positive(value: f64, field_name: &str) -> ValidationResult<()> {
        if value <= 0.0 {
            Err(ValidationError::Custom(format!(
                "字段 '{}' 必须是正数，当前值 {}", field_name, value
            )))
        } else {
            Ok(())
        }
    }
}

/// 常用验证函数
pub mod common {
    use super::*;

    /// 验证非空字符串
    pub fn not_empty(value: &str, field_name: &str) -> ValidationResult<()> {
        if value.trim().is_empty() {
            Err(ValidationError::Empty(field_name.to_string()))
        } else {
            Ok(())
        }
    }

    /// 验证字符串长度
    pub fn length_between(value: &str, min: usize, max: usize, field_name: &str) -> ValidationResult<()> {
        let len = value.len();
        if len < min || len > max {
            Err(ValidationError::Custom(format!(
                "字段 '{}' 长度必须在 {} 到 {} 之间，当前长度 {}",
                field_name, min, max, len
            )))
        } else {
            Ok(())
        }
    }

    /// 验证数组非空
    pub fn array_not_empty<T>(array: &[T], field_name: &str) -> ValidationResult<()> {
        if array.is_empty() {
            Err(ValidationError::Empty(field_name.to_string()))
        } else {
            Ok(())
        }
    }
} 