use crate::error::Result;
use crate::data::schema::DataSchema;
use crate::data::record::Record;
use crate::data::value::DataValue;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use regex::Regex;
use log::{debug, warn};
use serde_json::Value;
use chrono::NaiveDate;
use lazy_static::lazy_static;

use super::{DataValidator, ValidationType};
use crate::core::interfaces::ValidationResult;
use super::Validator;

// 日期格式常量
const DATE_FORMAT_ISO: &str = "%Y-%m-%d";

/// 验证规则类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RuleType {
    /// 是否为空
    NotNull,
    /// 最小长度
    MinLength,
    /// 最大长度
    MaxLength,
    /// 最小值
    MinValue,
    /// 最大值
    MaxValue,
    /// 正则表达式匹配
    Regex,
    /// 枚举值检查
    Enum,
    /// 自定义表达式
    Expression,
    /// 必填项
    Required,
    /// 模式匹配
    Pattern,
    /// 范围检查
    Range,
    /// 唯一性检查
    Unique,
    /// 格式检查
    Format,
    /// 自定义验证器
    Custom
}

impl RuleType {
    /// 从字符串创建RuleType
    pub fn from_str(rule_type: &str) -> Option<RuleType> {
        match rule_type.to_lowercase().as_str() {
            "required" => Some(RuleType::Required),
            "pattern" => Some(RuleType::Pattern),
            "range" => Some(RuleType::Range),
            "unique" => Some(RuleType::Unique),
            "format" => Some(RuleType::Format),
            "custom" => Some(RuleType::Custom),
            _ => None
        }
    }
}

/// 验证规则
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// 规则类型
    pub rule_type: RuleType,
    /// 字段名
    pub field_name: String,
    /// 规则参数
    pub parameters: HashMap<String, Value>,
    /// 错误消息
    pub error_message: Option<String>,
    /// 是否为警告（而非错误）
    pub is_warning: bool,
}

impl ValidationRule {
    /// 创建新的验证规则
    pub fn new(rule_type: RuleType, field_name: &str) -> Self {
        Self {
            rule_type,
            field_name: field_name.to_string(),
            parameters: HashMap::new(),
            error_message: None,
            is_warning: false,
        }
    }
    
    /// 添加参数
    pub fn add_parameter<S: Into<String>>(mut self, key: S, value: S) -> Self {
        self.parameters.insert(key.into(), serde_json::Value::String(value.into()));
        self
    }
    
    /// 设置错误消息
    pub fn with_error_message<S: Into<String>>(mut self, message: S) -> Self {
        self.error_message = Some(message.into());
        self
    }
    
    /// 设置为警告
    pub fn as_warning(mut self) -> Self {
        self.is_warning = true;
        self
    }
    
    /// 获取错误消息
    fn get_error_message(&self) -> String {
        if let Some(message) = &self.error_message {
            message.clone()
        } else {
            match self.rule_type {
                RuleType::NotNull => format!("字段 '{}' 不能为空", self.field_name),
                RuleType::MinLength => {
                    let min = self.parameters.get("min").and_then(|v| v.as_str()).unwrap_or("0");
                    format!("字段 '{}' 长度不能小于 {}", self.field_name, min)
                },
                RuleType::MaxLength => {
                    let max = self.parameters.get("max").and_then(|v| v.as_str()).unwrap_or("0");
                    format!("字段 '{}' 长度不能大于 {}", self.field_name, max)
                },
                RuleType::MinValue => {
                    let min = self.parameters.get("min").and_then(|v| v.as_str()).unwrap_or("0");
                    format!("字段 '{}' 值不能小于 {}", self.field_name, min)
                },
                RuleType::MaxValue => {
                    let max = self.parameters.get("max").and_then(|v| v.as_str()).unwrap_or("0");
                    format!("字段 '{}' 值不能大于 {}", self.field_name, max)
                },
                RuleType::Regex => {
                    let pattern = self.parameters.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
                    format!("字段 '{}' 不符合模式 '{}'", self.field_name, pattern)
                },
                RuleType::Enum => {
                    let values = self.parameters.get("values").and_then(|v| v.as_str()).unwrap_or("");
                    format!("字段 '{}' 值不在有效范围内 [{}]", self.field_name, values)
                },
                RuleType::Expression => {
                    let expr = self.parameters.get("expr").and_then(|v| v.as_str()).unwrap_or("");
                    format!("字段 '{}' 不满足表达式 '{}'", self.field_name, expr)
                },
                RuleType::Required => format!("字段 '{}' 为必填项", self.field_name),
                RuleType::Pattern => {
                    let pattern = self.parameters.get("pattern").and_then(|v| v.as_str()).unwrap_or("");
                    format!("字段 '{}' 不匹配模式 '{}'", self.field_name, pattern)
                },
                RuleType::Range => {
                    let min_msg = self.parameters.get("min").and_then(|v| v.as_f64())
                        .map(|m| format!("字段 '{}' 值小于最小值 {}", self.field_name, m))
                        .unwrap_or_default();
                    let max_msg = self.parameters.get("max").and_then(|v| v.as_f64())
                        .map(|m| format!("字段 '{}' 值大于最大值 {}", self.field_name, m))
                        .unwrap_or_default();
                    
                    if !min_msg.is_empty() && !max_msg.is_empty() {
                        format!("字段 '{}' 值不在有效范围内: 最小值 {}, 最大值 {}", 
                            self.field_name, 
                            self.parameters.get("min").and_then(|v| v.as_f64()).unwrap_or_default(),
                            self.parameters.get("max").and_then(|v| v.as_f64()).unwrap_or_default())
                    } else if !min_msg.is_empty() {
                        min_msg
                    } else if !max_msg.is_empty() {
                        max_msg
                    } else {
                        format!("字段 '{}' 值不在有效范围内", self.field_name)
                    }
                },
                RuleType::Unique => format!("唯一性验证需要在完整数据集上进行，单条记录验证中跳过"),
                RuleType::Format => {
                    let format = self.parameters.get("format").and_then(|v| v.as_str()).unwrap_or("");
                    format!("字段 '{}' 不是有效的 {} 格式", self.field_name, format)
                },
                RuleType::Custom => format!("使用注册的自定义验证器"),
            }
        }
    }
}

/// 自定义验证器
pub struct CustomValidator {
    /// 验证器名称
    name: String,
    /// 数据模式
    schema: DataSchema,
    /// 验证规则
    rules: Vec<ValidationRule>,
    /// 验证器集合
    validators: HashMap<String, Box<dyn Validator>>,
}

impl CustomValidator {
    /// 创建新的自定义验证器
    pub fn new<S: Into<String>>(name: S, schema: DataSchema) -> Self {
        Self {
            name: name.into(),
            schema,
            rules: Vec::new(),
            validators: HashMap::new(),
        }
    }
    
    /// 从规则列表创建自定义验证器
    pub fn with_rules<S: Into<String>>(name: S, schema: DataSchema, rules: Vec<ValidationRule>) -> Self {
        Self {
            name: name.into(),
            schema,
            rules,
            validators: HashMap::new(),
        }
    }
    
    /// 添加验证规则
    pub fn add_rule(mut self, rule: ValidationRule) -> Self {
        self.rules.push(rule);
        self
    }
    
    /// 添加验证器
    pub fn add_validator(&mut self, field: String, validator: Box<dyn Validator>) {
        self.validators.insert(field, validator);
    }
    
    /// 应用规则验证值
    fn apply_rule(&self, rule: &ValidationRule, value: &DataValue, index: usize) -> Result<ValidationResult> {
        let mut result = ValidationResult::success();
        
        match rule.rule_type {
            RuleType::NotNull => {
                if matches!(value, DataValue::Null) {
                    if rule.is_warning {
                        result.warnings.push(rule.get_error_message());
                    } else {
                        result.is_valid = false;
                        result.errors.push(rule.get_error_message());
                    }
                }
            },
            RuleType::MinLength => {
                if let Some(min_str) = rule.parameters.get("min") {
                    if let Some(min_str) = min_str.as_str() {
                    if let Ok(min) = min_str.parse::<usize>() {
                        match value {
                            DataValue::String(s) => {
                                if s.len() < min {
                                    if rule.is_warning {
                                        result.warnings.push(rule.get_error_message());
                                    } else {
                                        result.is_valid = false;
                                        result.errors.push(rule.get_error_message());
                                    }
                                }
                            },
                            DataValue::Array(arr) => {
                                if arr.len() < min {
                                    if rule.is_warning {
                                        result.warnings.push(rule.get_error_message());
                                    } else {
                                        result.is_valid = false;
                                        result.errors.push(rule.get_error_message());
                                    }
                                }
                            },
                            _ => {
                                    // 不适用于其他类型
                                warn!("MinLength rule not applicable to type: {:?}", value);
                                }
                            }
                        }
                    }
                }
            },
            RuleType::MaxLength => {
                if let Some(max_str) = rule.parameters.get("max") {
                    if let Some(max_str) = max_str.as_str() {
                    if let Ok(max) = max_str.parse::<usize>() {
                        match value {
                            DataValue::String(s) => {
                                if s.len() > max {
                                    if rule.is_warning {
                                        result.warnings.push(rule.get_error_message());
                                    } else {
                                        result.is_valid = false;
                                        result.errors.push(rule.get_error_message());
                                    }
                                }
                            },
                            DataValue::Array(arr) => {
                                if arr.len() > max {
                                    if rule.is_warning {
                                        result.warnings.push(rule.get_error_message());
                                    } else {
                                        result.is_valid = false;
                                        result.errors.push(rule.get_error_message());
                                    }
                                }
                            },
                            _ => {
                                    // 不适用于其他类型
                                warn!("MaxLength rule not applicable to type: {:?}", value);
                                }
                            }
                        }
                    }
                }
            },
            RuleType::MinValue => {
                if let Some(min_str) = rule.parameters.get("min") {
                    match value {
                        DataValue::Integer(i) => {
                            if let Some(min_str) = min_str.as_str() {
                            if let Ok(min) = min_str.parse::<i64>() {
                                if *i < min {
                                    if rule.is_warning {
                                        result.warnings.push(rule.get_error_message());
                                    } else {
                                        result.is_valid = false;
                                        result.errors.push(rule.get_error_message());
                                        }
                                    }
                                }
                            }
                        },
                        DataValue::Float(f) => {
                            if let Some(min_str) = min_str.as_str() {
                            if let Ok(min) = min_str.parse::<f64>() {
                                if *f < min {
                                    if rule.is_warning {
                                        result.warnings.push(rule.get_error_message());
                                    } else {
                                        result.is_valid = false;
                                        result.errors.push(rule.get_error_message());
                                        }
                                    }
                                }
                            }
                        },
                        _ => {
                            // 不适用于其他类型
                            warn!("MinValue rule not applicable to type: {:?}", value);
                        }
                    }
                }
            },
            RuleType::MaxValue => {
                if let Some(max_str) = rule.parameters.get("max") {
                    match value {
                        DataValue::Integer(i) => {
                            if let Some(max_str) = max_str.as_str() {
                            if let Ok(max) = max_str.parse::<i64>() {
                                if *i > max {
                                    if rule.is_warning {
                                        result.warnings.push(rule.get_error_message());
                                    } else {
                                        result.is_valid = false;
                                        result.errors.push(rule.get_error_message());
                                        }
                                    }
                                }
                            }
                        },
                        DataValue::Float(f) => {
                            if let Some(max_str) = max_str.as_str() {
                            if let Ok(max) = max_str.parse::<f64>() {
                                if *f > max {
                                    if rule.is_warning {
                                        result.warnings.push(rule.get_error_message());
                                    } else {
                                        result.is_valid = false;
                                        result.errors.push(rule.get_error_message());
                                        }
                                    }
                                }
                            }
                        },
                        _ => {
                            // 不适用于其他类型
                            warn!("MaxValue rule not applicable to type: {:?}", value);
                        }
                    }
                }
            },
            RuleType::Regex => {
                if let Some(pattern) = rule.parameters.get("pattern") {
                    if let Some(pattern_str) = pattern.as_str() {
                    if let DataValue::String(s) = value {
                            match Regex::new(pattern_str) {
                            Ok(regex) => {
                                if !regex.is_match(s) {
                                    if rule.is_warning {
                                        result.warnings.push(rule.get_error_message());
                                    } else {
                                        result.is_valid = false;
                                        result.errors.push(rule.get_error_message());
                                    }
                                }
                            },
                            Err(e) => {
                                    warn!("Invalid regex pattern: {}, error: {}", pattern_str, e);
                                    result.warnings.push(format!("Invalid regex pattern: {}", pattern_str));
                                }
                            }
                        } else {
                            // 不适用于非字符串类型
                            warn!("Regex rule not applicable to type: {:?}", value);
                        }
                    }
                }
            },
            RuleType::Enum => {
                if let Some(values_str) = rule.parameters.get("values") {
                    if let Some(values_str) = values_str.as_str() {
                    let values: Vec<&str> = values_str.split(',').map(|s| s.trim()).collect();
                    
                    match value {
                        DataValue::String(s) => {
                            if !values.contains(&s.as_str()) {
                                if rule.is_warning {
                                    result.warnings.push(rule.get_error_message());
                                } else {
                                    result.is_valid = false;
                                    result.errors.push(rule.get_error_message());
                                }
                            }
                        },
                        DataValue::Integer(i) => {
                            let s = i.to_string();
                            if !values.contains(&s.as_str()) {
                                if rule.is_warning {
                                    result.warnings.push(rule.get_error_message());
                                } else {
                                    result.is_valid = false;
                                    result.errors.push(rule.get_error_message());
                                }
                            }
                        },
                        DataValue::Number(n) => {
                            let s = n.to_string();
                            if !values.contains(&s.as_str()) {
                                if rule.is_warning {
                                    result.warnings.push(rule.get_error_message());
                                } else {
                                    result.is_valid = false;
                                    result.errors.push(rule.get_error_message());
                                }
                            }
                        },
                        _ => {
                            // 不适用于其他类型
                            warn!("Enum rule not applicable to type: {:?}", value);
                        }
                        }
                    }
                }
            },
            RuleType::Expression => {
                // 表达式规则尚未实现
                warn!("Expression rule not yet implemented");
                result.warnings.push("Expression rule not yet implemented".to_string());
            },
            RuleType::Required => {
                if !matches!(value, DataValue::Null) {
                    if rule.is_warning {
                        result.warnings.push(rule.get_error_message());
                    } else {
                        result.is_valid = false;
                        result.errors.push(rule.get_error_message());
                    }
                }
            },
            RuleType::Pattern => {
                if let Some(pattern) = rule.parameters.get("pattern") {
                    if let Some(pattern_str) = pattern.as_str() {
                        if let Some(value_str) = value.as_str() {
                            match Regex::new(pattern_str) {
                                Ok(re) => {
                                    if !re.is_match(value_str) {
                                        if rule.is_warning {
                                            result.warnings.push(rule.get_error_message());
                                        } else {
                                            result.is_valid = false;
                                            result.errors.push(rule.get_error_message());
                                        }
                                    }
                                },
                                Err(e) => {
                                    warn!("Invalid regex pattern {}: {}", pattern_str, e);
                                    let msg = format!("Validation rule error: Invalid regex pattern {}", pattern_str);
                                    result.is_valid = false;
                                    result.errors.push(msg);
                                }
                            }
                        } else {
                            // 非字符串值无法进行正则匹配
                            warn!("Pattern rule not applicable to non-string type: {:?}", value);
                        }
                    }
                }
            },
            RuleType::Range => {
                let min = rule.parameters.get("min").and_then(|v| v.as_f64());
                let max = rule.parameters.get("max").and_then(|v| v.as_f64());
                
                if let Some(val_num) = value.as_number() {
                    if let Some(min_val) = min {
                        if val_num < min_val {
                            if rule.is_warning {
                                result.warnings.push(rule.get_error_message());
                            } else {
                                result.is_valid = false;
                                result.errors.push(rule.get_error_message());
                            }
                        }
                    }
                    
                    if let Some(max_val) = max {
                        if val_num > max_val {
                            if rule.is_warning {
                                result.warnings.push(rule.get_error_message());
                            } else {
                                result.is_valid = false;
                                result.errors.push(rule.get_error_message());
                            }
                        }
                    }
                } else {
                    // 非数值类型无法进行范围比较
                    warn!("Range rule not applicable to non-numeric type: {:?}", value);
                }
            },
            RuleType::Unique => {
                // 唯一性验证需要在完整数据集上进行，这里仅做记录
                debug!("Uniqueness validation requires a complete dataset, skipped for single record validation");
            },
            RuleType::Format => {
                if let Some(format) = rule.parameters.get("format").and_then(|v| v.as_str()) {
                    if let Some(val_str) = value.as_str() {
                        let is_valid = match format {
                            "email" => self.validate_email(val_str),
                            "date" => self.validate_date(val_str),
                            "url" => self.validate_url(val_str),
                            "ipv4" => self.validate_ipv4(val_str),
                            "ipv6" => self.validate_ipv6(val_str),
                            // 允许通过自定义日期格式，例如 date:%Y/%m/%d 或 date:%Y-%m-%d %H:%M:%S
                            _ if format.starts_with("date:") => {
                                let fmt = &format[5..];
                                self.validate_date_with_format(val_str, fmt)
                            }
                            _ => true
                        };
                        
                        if !is_valid {
                            if rule.is_warning {
                                result.warnings.push(rule.get_error_message());
                            } else {
                                result.is_valid = false;
                                result.errors.push(rule.get_error_message());
                            }
                        }
                    } else {
                        // 非字符串类型无法进行格式验证
                        warn!("Format rule not applicable to non-string type: {:?}", value);
                    }
                }
            },
            RuleType::Custom => {
                // 使用注册的自定义验证器
                if let Some(validator) = self.validators.get(&rule.field_name) {
                    let field_record = HashMap::from([(rule.field_name.clone(), value.clone())]);
                    let validation_result = validator.validate(&field_record);
                    
                    for error in &validation_result.errors {
                        result.is_valid = false;
                        result.errors.push(error.clone());
                    }
                }
            }
        }
        
        Ok(result)
    }
}

impl Validator for CustomValidator {
    fn validate(&self, data: &HashMap<String, DataValue>) -> ValidationResult {
        let mut result = ValidationResult::success();
        
        // 应用每条规则
        for rule in &self.rules {
            if let Some(value) = data.get(&rule.field_name) {
                match self.apply_rule(rule, value, 0) {
                    Ok(rule_result) => {
                        result.is_valid = result.is_valid && rule_result.is_valid;
                        result.errors.extend(rule_result.errors);
                        result.warnings.extend(rule_result.warnings);
                        if let Some(score) = rule_result.score {
                            result.score = result.score.map(|s| (s + score) / 2.0).or(Some(score));
                        }
                        result.metadata.extend(rule_result.metadata);
                    }
                    Err(_) => {
                        result.is_valid = false;
                        result.errors.push(format!("验证规则 '{}' 执行失败", rule.field_name));
                    }
                }
            } else if rule.rule_type == RuleType::NotNull {
                if rule.is_warning {
                    result.warnings.push(rule.get_error_message());
                } else {
                    result.is_valid = false;
                    result.errors.push(rule.get_error_message());
                }
            }
        }
        
        result
    }
    
    fn validation_type(&self) -> ValidationType {
        ValidationType::Custom
    }
}

impl DataValidator for CustomValidator {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn validation_type(&self) -> ValidationType {
        ValidationType::Custom
    }
    
    fn validate_record(&self, record: &Record, _schema: &DataSchema, index: usize) -> Result<ValidationResult> {
        let mut result = ValidationResult::success();
        
        // 应用每条规则
        for rule in &self.rules {
            if let Some(record_value) = record.get_field(&rule.field_name) {
                // 将 Record::Value 转换为 DataValue
                let data_value = record_value.to_data_value()?;
                let rule_result = self.apply_rule(rule, &data_value, index)?;
                
                // 手动合并结果
                result.is_valid = result.is_valid && rule_result.is_valid;
                result.errors.extend(rule_result.errors);
                result.warnings.extend(rule_result.warnings);
                if let Some(score) = rule_result.score {
                    result.score = result.score.map(|s| (s + score) / 2.0).or(Some(score));
                }
                result.metadata.extend(rule_result.metadata);
            } else if rule.rule_type == RuleType::NotNull {
                // 字段不存在，且规则是NotNull，添加错误
                if rule.is_warning {
                    result.warnings.push(rule.get_error_message());
                } else {
                    result.is_valid = false;
                    result.errors.push(rule.get_error_message());
                }
            }
        }
        
        Ok(result)
    }
    
    fn get_schema(&self) -> Result<DataSchema> {
        Ok(self.schema.clone())
    }
}

impl CustomValidator {
    // 辅助方法：验证电子邮件格式
    fn validate_email(&self, value: &str) -> bool {
        lazy_static! {
            static ref EMAIL_RE: Regex = Regex::new(
                r"^[a-zA-Z0-9.!#$%&*+/=?^'_{}|~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
            ).unwrap();
        }
        EMAIL_RE.is_match(value)
    }
    
    // 验证日期格式 (YYYY-MM-DD)
    fn validate_date(&self, value: &str) -> bool {
        lazy_static! {
            static ref DATE_RE: Regex = Regex::new(
                r"^\d{4}-\d{2}-\d{2}$"
            ).unwrap();
        }
        if DATE_RE.is_match(value) {
            // 简单验证日期有效性
            if let Ok(_date) = chrono::NaiveDate::parse_from_str(value, DATE_FORMAT_ISO) {
                return true;
            }
            false
        } else {
            false
        }
    }
    
    // 使用指定格式验证日期
    fn validate_date_with_format(&self, value: &str, fmt: &str) -> bool {
        NaiveDate::parse_from_str(value, fmt).is_ok()
    }
    
    // 验证URL格式
    fn validate_url(&self, value: &str) -> bool {
        url::Url::parse(value).is_ok()
    }
    
    // 验证IPv4格式
    fn validate_ipv4(&self, value: &str) -> bool {
        value.parse::<std::net::Ipv4Addr>().is_ok()
    }
    
    // 验证IPv6格式
    fn validate_ipv6(&self, value: &str) -> bool {
        value.parse::<std::net::Ipv6Addr>().is_ok()
    }
} 
