// 范围验证器模块, 用于验证数字、日期和字符串值是否在指定范围内
use crate::error::Result;
use crate::data::value::DataValue;
use crate::data::record::Record;
use crate::data::validation::{ValidationError, DataValidationResult, ValidationContext};
use crate::core::interfaces::ValidationResult;
use crate::data::schema::DataSchema;
use super::{DataValidator, ValidationType, ErrorSeverity, Validator};
use std::collections::HashMap;
use std::fmt;
use serde::{Deserialize, Serialize};
 
 

// 辅助函数：根据数值级别映射严重程度并构造 ValidationError
fn make_validation_error(code: &str, message: impl Into<String>, level: i32) -> ValidationError {
    let severity = match level {
        0 => ErrorSeverity::Info,
        1 => ErrorSeverity::Error,
        2 => ErrorSeverity::Critical,
        _ => ErrorSeverity::Error,
    };
    ValidationError::new(code.to_string(), message.into()).with_severity(severity)
}

/// 范围比较操作
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeOperation {
    /// 大于最小值
    GreaterThan,
    /// 大于等于最小值
    GreaterThanOrEqual,
    /// 小于最大值
    LessThan,
    /// 小于等于最大值
    LessThanOrEqual,
    /// 在范围内 (包括边界)
    Between,
    /// 在范围外
    Outside,
}

/// 范围验证模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RangeValidationMode {
    /// 验证数值范围
    Value,
    /// 验证字符串长度范围
    Length,
    /// 验证集合大小范围
    Size,
}

impl Default for RangeValidationMode {
    fn default() -> Self {
        RangeValidationMode::Value
    }
}

/// 范围验证器配置选项
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeValidatorConfig {
    /// 最小值 (包含边界)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    
    /// 最大值 (包含边界)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
    
    /// 是否排除最小值
    #[serde(default)]
    pub exclude_min: bool,
    
    /// 是否排除最大值
    #[serde(default)]
    pub exclude_max: bool,
    
    /// 验证模式
    #[serde(default)]
    pub mode: RangeValidationMode,
    
    /// 自定义错误信息
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    
    /// 最小值 (不包含边界) - 兼容性字段
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclusive_min: Option<DataValue>,
    
    /// 最大值 (不包含边界) - 兼容性字段
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclusive_max: Option<DataValue>,
    
    /// 自定义错误信息 - 兼容性字段
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    
    /// 错误级别 (可选)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_level: Option<i32>,
    
    /// 自定义验证标识符 (可选)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub validation_id: Option<String>,
    
    /// 额外元数据
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, DataValue>>,
}

impl Default for RangeValidatorConfig {
    fn default() -> Self {
        Self {
            min: None,
            max: None,
            exclude_min: false,
            exclude_max: false,
            mode: RangeValidationMode::default(),
            message: None,
            exclusive_min: None,
            exclusive_max: None,
            error_message: None,
            error_level: None,
            validation_id: None,
            metadata: None,
        }
    }
}

/// 范围验证器，用于根据配置的范围验证数值、字符串长度或集合大小
pub struct RangeValidator {
    config: RangeValidatorConfig,
}

impl RangeValidator {
    /// 创建新的范围验证器
    pub fn new(config: RangeValidatorConfig) -> std::result::Result<Self, String> {
        // 验证配置的有效性
        Self::validate_config(&config)?;
        
        Ok(Self { config })
    }
    
    /// 验证配置的有效性
    fn validate_config(config: &RangeValidatorConfig) -> Result<(), String> {
        // 检查是否至少设置了一个约束条件
        if config.min.is_none() && config.max.is_none() {
            return Err("At least one of min or max must be set".to_string());
        }
        
        // 验证最小值小于最大值（如果两者都设置了）
        if let (Some(min), Some(max)) = (config.min, config.max) {
            if min >= max {
                return Err("Min value must be less than max value".to_string());
            }
        }
        
        Ok(())
    }
    
    /// 比较两个DataValue值的大小
    fn is_less_than(a: &DataValue, b: &DataValue) -> bool {
        match (a, b) {
            (DataValue::Integer(a_val), DataValue::Integer(b_val)) => a_val <= b_val,
            (DataValue::Integer(a_val), DataValue::Number(b_val)) => (*a_val as f64) <= *b_val,
            (DataValue::Number(a_val), DataValue::Integer(b_val)) => *a_val <= (*b_val as f64),
            (DataValue::Number(a_val), DataValue::Number(b_val)) => a_val <= b_val,
            (DataValue::DateTime(a_val), DataValue::DateTime(b_val)) => a_val <= b_val,
            _ => false,
        }
    }
    
    /// 比较两个DataValue值是否严格小于
    fn is_strictly_less_than(a: &DataValue, b: &DataValue) -> bool {
        match (a, b) {
            (DataValue::Integer(a_val), DataValue::Integer(b_val)) => a_val < b_val,
            (DataValue::Integer(a_val), DataValue::Number(b_val)) => (*a_val as f64) < *b_val,
            (DataValue::Number(a_val), DataValue::Integer(b_val)) => *a_val < (*b_val as f64),
            (DataValue::Number(a_val), DataValue::Number(b_val)) => a_val < b_val,
            (DataValue::DateTime(a_val), DataValue::DateTime(b_val)) => a_val < b_val,
            _ => false,
        }
    }
    
    /// 生成范围错误消息
    fn generate_error_message(&self, value: &DataValue) -> String {
        if let Some(error_msg) = &self.config.error_message {
            return error_msg.clone();
        }
        
        let mut bounds = Vec::new();
        
        if let Some(min) = &self.config.min {
            bounds.push(format!("≥ {}", Self::format_value(&DataValue::Number(*min))));
        }
        
        if let Some(exclusive_min) = &self.config.exclusive_min {
            bounds.push(format!("> {}", Self::format_value(exclusive_min)));
        }
        
        if let Some(max) = &self.config.max {
            bounds.push(format!("≤ {}", Self::format_value(&DataValue::Number(*max))));
        }
        
        if let Some(exclusive_max) = &self.config.exclusive_max {
            bounds.push(format!("< {}", Self::format_value(exclusive_max)));
        }
        
        let bounds_str = bounds.join(" and ");
        format!("Value {} must be {}", Self::format_value(value), bounds_str)
    }
    
    /// 格式化DataValue为字符串表示
    fn format_value(value: &DataValue) -> String {
        match value {
            DataValue::Integer(n) => n.to_string(),
            DataValue::Number(n) => n.to_string(),
            DataValue::DateTime(dt) => dt.clone(),
            _ => format!("{:?}", value),
        }
    }
    
    /// 获取验证器类型
    pub fn validator_type(&self) -> ValidationType {
        ValidationType::Range
    }
    
    /// 验证值是否在范围内
    pub fn validate(&self, value: &DataValue, context: &ValidationContext) -> Result<DataValidationResult> {
        match self.config.mode {
            RangeValidationMode::Value => self.validate_value_range(value, context),
            RangeValidationMode::Length => self.validate_length_range(value, context),
            RangeValidationMode::Size => self.validate_size_range(value, context),
        }
    }

    /// 验证数值范围
    fn validate_value_range(&self, value: &DataValue, _context: &ValidationContext) -> Result<DataValidationResult> {
        let numeric_value = match value {
            DataValue::Integer(n) => *n as f64,
            DataValue::Number(n) => *n,
            _ => {
                return Ok(DataValidationResult::failure(
                    vec![ValidationError::new(
                        "invalid_type".to_string(),
                        "Expected numeric value for range validation".to_string(),
                    )],
                    std::time::Duration::from_secs(0),
                ));
            }
        };

        if let Some(min) = self.config.min {
            if self.config.exclude_min {
                if numeric_value <= min {
                    return Ok(DataValidationResult::failure(
                        vec![ValidationError::new(
                            "range_violation".to_string(),
                            format!("Value {} must be greater than {}", numeric_value, min),
                        )],
                        std::time::Duration::from_secs(0),
                    ));
                }
            } else if numeric_value < min {
                return Ok(DataValidationResult::failure(
                    vec![ValidationError::new(
                        "range_violation".to_string(),
                        format!("Value {} must be greater than or equal to {}", numeric_value, min),
                    )],
                    std::time::Duration::from_secs(0),
                ));
            }
        }

        if let Some(max) = self.config.max {
            if self.config.exclude_max {
                if numeric_value >= max {
                    return Ok(DataValidationResult::failure(
                        vec![ValidationError::new(
                            "range_violation".to_string(),
                            format!("Value {} must be less than {}", numeric_value, max),
                        )],
                        std::time::Duration::from_secs(0),
                    ));
                }
            } else if numeric_value > max {
                return Ok(DataValidationResult::failure(
                    vec![ValidationError::new(
                        "range_violation".to_string(),
                        format!("Value {} must be less than or equal to {}", numeric_value, max),
                    )],
                    std::time::Duration::from_secs(0),
                ));
            }
        }

        Ok(DataValidationResult::success(1, std::time::Duration::from_secs(0)))
    }

    /// 验证字符串长度范围
    fn validate_length_range(&self, value: &DataValue, _context: &ValidationContext) -> Result<DataValidationResult> {
        let length = match value {
            DataValue::String(s) => s.len() as f64,
            _ => {
                return Ok(DataValidationResult::failure(
                    vec![ValidationError::new(
                        "invalid_type".to_string(),
                        "Expected string value for length validation".to_string(),
                    )],
                    std::time::Duration::from_secs(0),
                ));
            }
        };

        if let Some(min) = self.config.min {
            if length < min {
                return Ok(DataValidationResult::failure(
                    vec![ValidationError::new(
                        "length_violation".to_string(),
                        format!("String length {} is less than minimum {}", length, min),
                    )],
                    std::time::Duration::from_secs(0),
                ));
            }
        }

        if let Some(max) = self.config.max {
            if length > max {
                return Ok(DataValidationResult::failure(
                    vec![ValidationError::new(
                        "length_violation".to_string(),
                        format!("String length {} exceeds maximum {}", length, max),
                    )],
                    std::time::Duration::from_secs(0),
                ));
            }
        }

        Ok(DataValidationResult::success(1, std::time::Duration::from_secs(0)))
    }

    /// 验证集合大小范围
    fn validate_size_range(&self, value: &DataValue, _context: &ValidationContext) -> Result<DataValidationResult> {
        let size = match value {
            DataValue::Array(arr) => arr.len() as f64,
            DataValue::Object(obj) => obj.len() as f64,
            _ => {
                return Ok(DataValidationResult::failure(
                    vec![ValidationError::new(
                        "invalid_type".to_string(),
                        "Expected array or object value for size validation".to_string(),
                    )],
                    std::time::Duration::from_secs(0),
                ));
            }
        };

        if let Some(min) = self.config.min {
            if size < min {
                return Ok(DataValidationResult::failure(
                    vec![ValidationError::new(
                        "size_violation".to_string(),
                        format!("Collection size {} is less than minimum {}", size, min),
                    )],
                    std::time::Duration::from_secs(0),
                ));
            }
        }

        if let Some(max) = self.config.max {
            if size > max {
                return Ok(DataValidationResult::failure(
                    vec![ValidationError::new(
                        "size_violation".to_string(),
                        format!("Collection size {} exceeds maximum {}", size, max),
                    )],
                    std::time::Duration::from_secs(0),
                ));
            }
        }

        Ok(DataValidationResult::success(1, std::time::Duration::from_secs(0)))
    }
    
    /// 创建范围错误
    fn create_out_of_range_error(&self, value: &DataValue) -> DataValidationResult {
        let error_message = self.generate_error_message(value);
        let error_code = ValidationErrorCode::OutOfRange;
        let error_level = self.config.error_level.unwrap_or(1);
        
        let mut error = ValidationError::new("out_of_range".to_string(), error_message)
            .with_severity(match error_level { 0 => ErrorSeverity::Info, 1 => ErrorSeverity::Error, 2 => ErrorSeverity::Critical, _ => ErrorSeverity::Error });
        
        // 添加上下文信息到metadata
        error = error.with_value(format!("{:?}", value));
        
        if let Some(min) = &self.config.min {
            // 将min添加到metadata中（通过value字段或其他方式）
        }
        
        if let Some(max) = &self.config.max {
            // 将max添加到metadata中
        }
        
        if let Some(exclusive_min) = &self.config.exclusive_min {
            // 将exclusive_min添加到metadata中
        }
        
        if let Some(exclusive_max) = &self.config.exclusive_max {
            // 将exclusive_max添加到metadata中
        }
        
        if let Some(validation_id) = &self.config.validation_id {
            // 将validation_id添加到metadata中
        }
        
        if let Some(metadata) = &self.config.metadata {
            // 将metadata添加到error中（如果需要）
        }
        
        DataValidationResult::failure(vec![error], std::time::Duration::from_secs(0))
    }
    
    /// 创建整数范围验证器
    pub fn integer_range(min: Option<i64>, max: Option<i64>) -> std::result::Result<Self, String> {
        let mut config = RangeValidatorConfig::default();
        config.mode = RangeValidationMode::Value;
        
        if let Some(min_val) = min {
            config.min = Some(min_val as f64);
        }
        
        if let Some(max_val) = max {
            config.max = Some(max_val as f64);
        }
        
        Self::new(config)
    }
    
    /// 创建浮点数范围验证器
    pub fn float_range(min: Option<f64>, max: Option<f64>) -> std::result::Result<Self, String> {
        let mut config = RangeValidatorConfig::default();
        config.mode = RangeValidationMode::Value;
        config.min = min;
        config.max = max;
        
        Self::new(config)
    }
    
    /// 创建长度范围验证器
    pub fn length_range(min: Option<usize>, max: Option<usize>) -> std::result::Result<Self, String> {
        let mut config = RangeValidatorConfig::default();
        config.mode = RangeValidationMode::Length;
        
        if let Some(min_val) = min {
            config.min = Some(min_val as f64);
        }
        
        if let Some(max_val) = max {
            config.max = Some(max_val as f64);
        }
        
        Self::new(config)
    }
    
    /// 创建大小范围验证器
    pub fn size_range(min: Option<usize>, max: Option<usize>) -> std::result::Result<Self, String> {
        let mut config = RangeValidatorConfig::default();
        config.mode = RangeValidationMode::Size;
        
        if let Some(min_val) = min {
            config.min = Some(min_val as f64);
        }
        
        if let Some(max_val) = max {
            config.max = Some(max_val as f64);
        }
        
        Self::new(config)
    }
}

impl DataValidator for RangeValidator {
    fn name(&self) -> &str {
        "range"
    }

    fn validation_type(&self) -> ValidationType {
        ValidationType::Range
    }

    /// 验证多条记录
    fn validate(&self, records: &[Record]) -> Result<ValidationResult> {
        let mut result = ValidationResult::success();
        
        for (index, record) in records.iter().enumerate() {
            let record_result = self.validate_record(record, &self.get_schema()?, index)?;
            result.errors.extend(record_result.errors);
            result.warnings.extend(record_result.warnings);
            if !record_result.is_valid {
                result.is_valid = false;
            }
        }
        
        Ok(result)
    }

    /// 验证单条记录
    fn validate_record(&self, record: &Record, _schema: &DataSchema, _index: usize) -> Result<ValidationResult> {
        let mut result = ValidationResult::success();
        
        // 遍历记录中的所有字段进行范围验证
        for (field_name, value) in record.fields.iter() {
            // 转换为DataValue
            let data_value = value.to_data_value()?;
            // 创建验证上下文
            let context = ValidationContext::new(format!("record_{}", _index));
            
            // 执行范围验证
            let field_result = self.validate(&data_value, &context)?;
            
            if !field_result.is_valid {
                result.is_valid = false;
                for error in field_result.errors {
                    let validation_error = ValidationError::new(
                        format!("RANGE_VALIDATION_{}", field_name.to_uppercase()),
                        error.message
                    ).with_field(field_name.clone());
                    result.errors.push(validation_error.message);
                }
            }
        }
        
        Ok(result)
    }

    /// 获取模式
    fn get_schema(&self) -> Result<DataSchema> {
        // 创建一个基本的数据模式
        let mut schema = DataSchema::new("range_validation", "1.0");
        
        // 添加范围验证的字段定义
        use crate::data::schema::schema::FieldDefinition;
        schema.add_field(FieldDefinition {
            name: "range_field".to_string(),
            field_type: crate::data::schema::schema::FieldType::Numeric,
            data_type: Some("float64".to_string()),
            required: false,
            nullable: true,
            primary_key: false,
            foreign_key: None,
            description: Some("Range validation field".to_string()),
            default_value: None,
            constraints: None,
            metadata: std::collections::HashMap::new(),
        })?;
        
        Ok(schema)
    }
}

/// 范围限制类型，可以是包含边界或排除边界
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundType {
    /// 包含边界值
    Inclusive,
    /// 不包含边界值
    Exclusive,
}

impl fmt::Display for BoundType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoundType::Inclusive => write!(f, "包含"),
            BoundType::Exclusive => write!(f, "排除"),
        }
    }
}

/// 范围配置，定义数值验证的上下限
#[derive(Debug, Clone)]
pub struct RangeConfig<T> {
    /// 最小值（可选）
    pub min: Option<T>,
    /// 最大值（可选）
    pub max: Option<T>,
    /// 最小值边界类型
    pub min_bound_type: BoundType,
    /// 最大值边界类型
    pub max_bound_type: BoundType,
}

impl<T> RangeConfig<T> {
    /// 创建新的范围配置
    pub fn new(min: Option<T>, max: Option<T>) -> Self {
        Self {
            min,
            max,
            min_bound_type: BoundType::Inclusive,
            max_bound_type: BoundType::Inclusive,
        }
    }

    /// 设置最小值边界类型
    pub fn with_min_bound_type(mut self, bound_type: BoundType) -> Self {
        self.min_bound_type = bound_type;
        self
    }

    /// 设置最大值边界类型
    pub fn with_max_bound_type(mut self, bound_type: BoundType) -> Self {
        self.max_bound_type = bound_type;
        self
    }

    /// 设置两个边界类型
    pub fn with_bound_types(mut self, min_type: BoundType, max_type: BoundType) -> Self {
        self.min_bound_type = min_type;
        self.max_bound_type = max_type;
        self
    }
}

/// 整数范围验证器
#[derive(Debug, Clone)]
pub struct IntRangeValidator {
    /// 字段名称
    field_name: String,
    /// 范围配置
    config: RangeConfig<i64>,
    /// 验证错误消息
    error_message: Option<String>,
}

impl IntRangeValidator {
    /// 创建新的整数范围验证器
    pub fn new<S: Into<String>>(field_name: S, config: RangeConfig<i64>) -> Self {
        Self {
            field_name: field_name.into(),
            config,
            error_message: None,
        }
    }

    /// 设置自定义错误消息
    pub fn with_error_message<S: Into<String>>(mut self, message: S) -> Self {
        self.error_message = Some(message.into());
        self
    }

    /// 获取范围描述
    fn get_range_description(&self) -> String {
        let min_symbol = match self.config.min_bound_type {
            BoundType::Inclusive => "≥",
            BoundType::Exclusive => ">",
        };
        
        let max_symbol = match self.config.max_bound_type {
            BoundType::Inclusive => "≤",
            BoundType::Exclusive => "<",
        };
        
        match (self.config.min, self.config.max) {
            (Some(min), Some(max)) => format!("{} {} 且 {} {}", min_symbol, min, max_symbol, max),
            (Some(min), None) => format!("{} {}", min_symbol, min),
            (None, Some(max)) => format!("{} {}", max_symbol, max),
            (None, None) => "无限制".to_string(),
        }
    }
}

impl Validator for IntRangeValidator {
    fn validate(&self, data: &HashMap<String, DataValue>) -> ValidationResult {
        let value = match data.get(&self.field_name) {
            Some(val) => val,
            None => {
                // 字段不存在时跳过验证
                return ValidationResult::success();
            }
        };

        let int_value = match value {
            DataValue::Integer(i) => *i,
            DataValue::Number(f) => *f as i64,
            DataValue::String(s) => {
                match s.parse::<i64>() {
                    Ok(i) => i,
                    Err(_) => {
                        let mut result = DataValidationResult::failure(
                            vec![ValidationError::new(
                                "PARSE_ERROR".to_string(),
                                self.error_message.clone().unwrap_or_else(|| {
                                    format!("字段值 '{}' 不是有效的整数", s)
                                })
                            )],
                            std::time::Duration::from_millis(0)
                        );
                        // 将DataValidationResult转换为ValidationResult
                        let mut validation_result = ValidationResult::success();
                        validation_result.is_valid = false;
                        for error in result.errors {
                            validation_result.errors.push(error.message);
                        }
                        return validation_result;
                    }
                }
            }
            _ => {
                let mut result = ValidationResult::success();
                result.is_valid = false;
                result.errors.push(
                    self.error_message.clone().unwrap_or_else(|| {
                        format!("字段类型 '{:?}' 不支持整数范围验证", value)
                    })
                );
                return result;
            }
        };

        // 验证最小值
        if let Some(min) = self.config.min {
            let valid = match self.config.min_bound_type {
                BoundType::Inclusive => int_value >= min,
                BoundType::Exclusive => int_value > min,
            };

            if !valid {
                let error_msg = self.error_message.clone().unwrap_or_else(|| {
                    format!("值 {} 小于最小值 {} ({})", int_value, min, self.config.min_bound_type)
                });
                
                let mut result = ValidationResult::success();
                result.is_valid = false;
                result.errors.push(error_msg);
                return result;
            }
        }

        // 验证最大值
        if let Some(max) = self.config.max {
            let valid = match self.config.max_bound_type {
                BoundType::Inclusive => int_value <= max,
                BoundType::Exclusive => int_value < max,
            };

            if !valid {
                let error_msg = self.error_message.clone().unwrap_or_else(|| {
                    format!("值 {} 大于最大值 {} ({})", int_value, max, self.config.max_bound_type)
                });
                
                let mut result = ValidationResult::success();
                result.is_valid = false;
                result.errors.push(error_msg);
                return result;
            }
        }

        ValidationResult::success()
    }

    fn validation_type(&self) -> ValidationType {
        ValidationType::Range
    }
}

/// 浮点数范围验证器
#[derive(Debug, Clone)]
pub struct FloatRangeValidator {
    /// 字段名称
    field_name: String,
    /// 范围配置
    config: RangeConfig<f64>,
    /// 验证错误消息
    error_message: Option<String>,
}

impl FloatRangeValidator {
    /// 创建新的浮点数范围验证器
    pub fn new<S: Into<String>>(field_name: S, config: RangeConfig<f64>) -> Self {
        Self {
            field_name: field_name.into(),
            config,
            error_message: None,
        }
    }

    /// 设置自定义错误消息
    pub fn with_error_message<S: Into<String>>(mut self, message: S) -> Self {
        self.error_message = Some(message.into());
        self
    }

    /// 获取范围描述
    fn get_range_description(&self) -> String {
        let min_symbol = match self.config.min_bound_type {
            BoundType::Inclusive => "≥",
            BoundType::Exclusive => ">",
        };
        
        let max_symbol = match self.config.max_bound_type {
            BoundType::Inclusive => "≤",
            BoundType::Exclusive => "<",
        };
        
        match (self.config.min, self.config.max) {
            (Some(min), Some(max)) => format!("{} {} 且 {} {}", min_symbol, min, max_symbol, max),
            (Some(min), None) => format!("{} {}", min_symbol, min),
            (None, Some(max)) => format!("{} {}", max_symbol, max),
            (None, None) => "无限制".to_string(),
        }
    }
}

impl Validator for FloatRangeValidator {
    fn validate(&self, data: &HashMap<String, DataValue>) -> ValidationResult {
        let value = match data.get(&self.field_name) {
            Some(val) => val,
            None => {
                // 字段不存在时跳过验证
                return ValidationResult::success();
            }
        };

        let float_value = match value {
            DataValue::Number(f) => *f,
            DataValue::Integer(i) => *i as f64,
            DataValue::String(s) => {
                match s.parse::<f64>() {
                    Ok(f) => f,
                    Err(_) => {
                        let mut result = ValidationResult::success();
                        result.is_valid = false;
                        result.errors.push(
                            self.error_message.clone().unwrap_or_else(|| {
                                format!("字段值 '{}' 不是有效的浮点数", s)
                            })
                        );
                        return result;
                    }
                }
            }
            _ => {
                let mut result = ValidationResult::success();
                result.is_valid = false;
                result.errors.push(
                    self.error_message.clone().unwrap_or_else(|| {
                        format!("字段类型 '{:?}' 不支持浮点数范围验证", value)
                    })
                );
                return result;
            }
        };

        // 验证最小值
        if let Some(min) = self.config.min {
            let valid = match self.config.min_bound_type {
                BoundType::Inclusive => float_value >= min,
                BoundType::Exclusive => float_value > min,
            };

            if !valid {
                let error_msg = self.error_message.clone().unwrap_or_else(|| {
                    format!("值 {} 小于最小值 {} ({})", float_value, min, self.config.min_bound_type)
                });
                
                let mut result = ValidationResult::success();
                result.is_valid = false;
                result.errors.push(error_msg);
                return result;
            }
        }

        // 验证最大值
        if let Some(max) = self.config.max {
            let valid = match self.config.max_bound_type {
                BoundType::Inclusive => float_value <= max,
                BoundType::Exclusive => float_value < max,
            };

            if !valid {
                let error_msg = self.error_message.clone().unwrap_or_else(|| {
                    format!("值 {} 大于最大值 {} ({})", float_value, max, self.config.max_bound_type)
                });
                
                let mut result = ValidationResult::success();
                result.is_valid = false;
                result.errors.push(error_msg);
                return result;
            }
        }

        ValidationResult::success()
    }

    fn validation_type(&self) -> ValidationType {
        ValidationType::Range
    }
}

/// 字符串长度范围验证器
#[derive(Debug, Clone)]
pub struct LengthRangeValidator {
    /// 字段名称
    field_name: String,
    /// 范围配置
    config: RangeConfig<usize>,
    /// 验证错误消息
    error_message: Option<String>,
}

impl LengthRangeValidator {
    /// 创建新的字符串长度范围验证器
    pub fn new<S: Into<String>>(field_name: S, config: RangeConfig<usize>) -> Self {
        Self {
            field_name: field_name.into(),
            config,
            error_message: None,
        }
    }

    /// 设置自定义错误消息
    pub fn with_error_message<S: Into<String>>(mut self, message: S) -> Self {
        self.error_message = Some(message.into());
        self
    }

    /// 获取范围描述
    fn get_range_description(&self) -> String {
        match (self.config.min, self.config.max) {
            (Some(min), Some(max)) => {
                if min == max {
                    format!("必须为 {} 个字符", min)
                } else {
                    format!("在 {} 到 {} 个字符之间", min, max)
                }
            }
            (Some(min), None) => format!("至少 {} 个字符", min),
            (None, Some(max)) => format!("最多 {} 个字符", max),
            (None, None) => "无限制".to_string(),
        }
    }
}

impl Validator for LengthRangeValidator {
    fn validate(&self, data: &HashMap<String, DataValue>) -> ValidationResult {
        let value = match data.get(&self.field_name) {
            Some(val) => val,
            None => {
                // 字段不存在时跳过验证
                return ValidationResult::success();
            }
        };

        let length = match value {
            DataValue::String(s) => s.len(),
            DataValue::Array(arr) => arr.len(),
            _ => {
                let mut result = ValidationResult::success();
                result.is_valid = false;
                result.errors.push(
                    self.error_message.clone().unwrap_or_else(|| {
                        format!("字段类型 '{:?}' 不支持长度验证", value)
                    })
                );
                return result;
            }
        };

        // 验证最小长度
        if let Some(min) = self.config.min {
            let valid = match self.config.min_bound_type {
                BoundType::Inclusive => length >= min,
                BoundType::Exclusive => length > min,
            };

            if !valid {
                let error_msg = self.error_message.clone().unwrap_or_else(|| {
                    format!("长度 {} 小于最小长度 {} ({})", length, min, self.config.min_bound_type)
                });
                
                let mut result = ValidationResult::success();
                result.is_valid = false;
                result.errors.push(error_msg);
                return result;
            }
        }

        // 验证最大长度
        if let Some(max) = self.config.max {
            let valid = match self.config.max_bound_type {
                BoundType::Inclusive => length <= max,
                BoundType::Exclusive => length < max,
            };

            if !valid {
                let error_msg = self.error_message.clone().unwrap_or_else(|| {
                    format!("长度 {} 大于最大长度 {} ({})", length, max, self.config.max_bound_type)
                });
                
                let mut result = ValidationResult::success();
                result.is_valid = false;
                result.errors.push(error_msg);
                return result;
            }
        }

        ValidationResult::success()
    }

    fn validation_type(&self) -> ValidationType {
        ValidationType::Range
    }
}

/// 创建整数范围验证器的便捷函数
pub fn int_range<S: Into<String>>(
    field_name: S,
    min: Option<i64>,
    max: Option<i64>,
) -> IntRangeValidator {
    IntRangeValidator::new(field_name, RangeConfig::new(min, max))
}

/// 创建浮点数范围验证器的便捷函数
pub fn float_range<S: Into<String>>(
    field_name: S,
    min: Option<f64>,
    max: Option<f64>,
) -> FloatRangeValidator {
    FloatRangeValidator::new(field_name, RangeConfig::new(min, max))
}

/// 创建字符串长度范围验证器的便捷函数
pub fn length_range<S: Into<String>>(
    field_name: S,
    min: Option<usize>,
    max: Option<usize>,
) -> LengthRangeValidator {
    LengthRangeValidator::new(field_name, RangeConfig::new(min, max))
}

/// 创建固定长度验证器的便捷函数
pub fn exact_length<S: Into<String>>(field_name: S, length: usize) -> LengthRangeValidator {
    LengthRangeValidator::new(field_name, RangeConfig::new(Some(length), Some(length)))
}

/// 验证错误码
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationErrorCode {
    /// 类型不匹配
    TypeMismatch,
    /// 超出范围
    OutOfRange,
    /// 格式错误
    FormatError,
    /// 必填字段缺失
    MissingRequired,
    /// 模式不匹配
    PatternMismatch,
    /// 长度错误
    LengthError,
    /// 唯一性冲突
    UniquenessViolation,
    /// 自定义规则失败
    CustomRuleFailed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_range_validator_value_mode() {
        let config = RangeValidatorConfig {
            min: Some(0.0),
            max: Some(100.0),
            exclude_min: false,
            exclude_max: false,
            mode: RangeValidationMode::Value,
            message: None,
        };
        
        let validator = RangeValidator::new(config);
        let context = ValidationContext {
            field_name: Some("score"),
            path: None,
            schema_type: None,
        };
        
        // 有效值
        let valid_value = DataValue::Number(50.0);
        let result = validator.validate(&valid_value, &context);
        assert!(result.is_valid());
        
        // 边界值（包含）
        let min_value = DataValue::Number(0.0);
        let result = validator.validate(&min_value, &context);
        assert!(result.is_valid());
        
        let max_value = DataValue::Number(100.0);
        let result = validator.validate(&max_value, &context);
        assert!(result.is_valid());
        
        // 无效值
        let invalid_value = DataValue::Number(150.0);
        let result = validator.validate(&invalid_value, &context);
        assert!(!result.is_valid());
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].code, "range_violation");
        
        // 非数值
        let non_numeric = DataValue::String("test".to_string());
        let result = validator.validate(&non_numeric, &context);
        assert!(!result.is_valid());
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].code, "invalid_type");
    }
    
    #[test]
    fn test_range_validator_length_mode() {
        let config = RangeValidatorConfig {
            min: Some(5.0),
            max: Some(10.0),
            exclude_min: false,
            exclude_max: false,
            mode: RangeValidationMode::Length,
            message: None,
        };
        
        let validator = RangeValidator::new(config);
        let context = ValidationContext {
            field_name: Some("username"),
            path: None,
            schema_type: None,
        };
        
        // 有效长度
        let valid_length = DataValue::String("username".to_string());
        let result = validator.validate(&valid_length, &context);
        assert!(result.is_valid());
        
        // 边界值
        let min_length = DataValue::String("12345".to_string());
        let result = validator.validate(&min_length, &context);
        assert!(result.is_valid());
        
        // 无效长度（太短）
        let too_short = DataValue::String("test".to_string());
        let result = validator.validate(&too_short, &context);
        assert!(!result.is_valid());
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].code, "length_violation");
        
        // 无效长度（太长）
        let too_long = DataValue::String("verylongusername".to_string());
        let result = validator.validate(&too_long, &context);
        assert!(!result.is_valid());
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].code, "length_violation");
        
        // 非字符串
        let non_string = DataValue::Integer(42);
        let result = validator.validate(&non_string, &context);
        assert!(!result.is_valid());
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].code, "invalid_type");
    }
    
    #[test]
    fn test_range_validator_size_mode() {
        let config = RangeValidatorConfig {
            min: Some(1.0),
            max: Some(5.0),
            exclude_min: false,
            exclude_max: false,
            mode: RangeValidationMode::Size,
            message: None,
        };
        
        let validator = RangeValidator::new(config);
        let context = ValidationContext {
            field_name: Some("tags"),
            path: None,
            schema_type: None,
        };
        
        // 有效大小（数组）
        let valid_array = DataValue::Array(vec![
            DataValue::String("tag1".to_string()),
            DataValue::String("tag2".to_string()),
        ]);
        let result = validator.validate(&valid_array, &context);
        assert!(result.is_valid());
        
        // 有效大小（对象）
        let mut valid_object = std::collections::HashMap::new();
        valid_object.insert("key1".to_string(), DataValue::String("value1".to_string()));
        valid_object.insert("key2".to_string(), DataValue::String("value2".to_string()));
        let valid_object = DataValue::Object(valid_object);
        let result = validator.validate(&valid_object, &context);
        assert!(result.is_valid());
        
        // 无效大小（空数组）
        let empty_array = DataValue::Array(vec![]);
        let result = validator.validate(&empty_array, &context);
        assert!(!result.is_valid());
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].code, "size_violation");
        
        // 无效大小（太大的数组）
        let large_array = DataValue::Array(vec![
            DataValue::String("tag1".to_string()),
            DataValue::String("tag2".to_string()),
            DataValue::String("tag3".to_string()),
            DataValue::String("tag4".to_string()),
            DataValue::String("tag5".to_string()),
            DataValue::String("tag6".to_string()),
        ]);
        let result = validator.validate(&large_array, &context);
        assert!(!result.is_valid());
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].code, "size_violation");
        
        // 非集合类型
        let non_collection = DataValue::Number(42.0);
        let result = validator.validate(&non_collection, &context);
        assert!(!result.is_valid());
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].code, "invalid_type");
    }
    
    #[test]
    fn test_range_validator_with_exclusions() {
        let config = RangeValidatorConfig {
            min: Some(0.0),
            max: Some(10.0),
            exclude_min: true,  // 排除最小值
            exclude_max: true,  // 排除最大值
            mode: RangeValidationMode::Value,
            message: None,
        };
        
        let validator = RangeValidator::new(config);
        let context = ValidationContext::default();
        
        // 有效值
        let valid_value = DataValue::Number(5.0);
        let result = validator.validate(&valid_value, &context);
        assert!(result.is_valid());
        
        // 边界值（排除在外）
        let min_value = DataValue::Number(0.0);
        let result = validator.validate(&min_value, &context);
        assert!(!result.is_valid());
        
        let max_value = DataValue::Number(10.0);
        let result = validator.validate(&max_value, &context);
        assert!(!result.is_valid());
        
        // 刚好有效的边界值
        let just_above_min = DataValue::Number(0.01);
        let result = validator.validate(&just_above_min, &context);
        assert!(result.is_valid());
        
        let just_below_max = DataValue::Number(9.99);
        let result = validator.validate(&just_below_max, &context);
        assert!(result.is_valid());
    }
} 