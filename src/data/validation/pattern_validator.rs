// 模式验证器模块，用于验证字符串是否匹配特定模式
use crate::error::Result;
use crate::data::value::DataValue;
 
use crate::data::validation::{ValidationError, DataValidationResult, ValidationWarning, Validator, ValidationType, ErrorSeverity};
use crate::core::interfaces::ValidationResult;
// use super::{DataValidator}; // unified Validator trait is used here
use std::collections::HashMap;
use regex::Regex;
use std::fmt;
// use log::warn; // no direct warn! used in this module
// use serde::{Deserialize, Serialize}; // serde used in other submodules; not directly here
use std::sync::Arc;

/// 模式类型枚举
#[derive(Debug, Clone)]
pub enum PatternType {
    /// 正则表达式模式
    Regex(Regex),
    /// 电子邮件模式
    Email,
    /// URL模式
    Url,
    /// 日期模式 (ISO 8601)
    Date,
    /// 时间模式 (ISO 8601)
    Time,
    /// 日期时间模式 (ISO 8601)
    DateTime,
    /// IP地址模式
    IPAddress,
    /// IPv4地址模式
    IPv4,
    /// IPv6地址模式
    IPv6,
    /// UUID模式
    UUID,
    /// 自定义字符串模式
    Custom(String),
}

impl fmt::Display for PatternType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatternType::Regex(_) => write!(f, "正则表达式"),
            PatternType::Email => write!(f, "电子邮件"),
            PatternType::Url => write!(f, "URL"),
            PatternType::Date => write!(f, "日期"),
            PatternType::Time => write!(f, "时间"),
            PatternType::DateTime => write!(f, "日期时间"),
            PatternType::IPAddress => write!(f, "IP地址"),
            PatternType::IPv4 => write!(f, "IPv4地址"),
            PatternType::IPv6 => write!(f, "IPv6地址"),
            PatternType::UUID => write!(f, "UUID"),
            PatternType::Custom(name) => write!(f, "自定义模式: {}", name),
        }
    }
}

impl PatternType {
    /// 检查字符串是否匹配此模式
    pub fn matches(&self, value: &str) -> bool {
        match self {
            PatternType::Regex(regex) => regex.is_match(value),
            PatternType::Email => {
                // 简化版电子邮件验证
                let email_regex = Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
                    .expect("无效的电子邮件正则表达式");
                email_regex.is_match(value)
            }
            PatternType::Url => {
                // 简化版URL验证
                let url_regex = Regex::new(r"^(https?|ftp)://[^\s/$.?#].[^\s]*$")
                    .expect("无效的URL正则表达式");
                url_regex.is_match(value)
            }
            PatternType::Date => {
                // ISO 8601 日期验证 (YYYY-MM-DD)
                let date_regex = Regex::new(r"^\d{4}-\d{2}-\d{2}$")
                    .expect("无效的日期正则表达式");
                date_regex.is_match(value)
            }
            PatternType::Time => {
                // ISO 8601 时间验证 (hh:mm:ss)
                let time_regex = Regex::new(r"^\d{2}:\d{2}:\d{2}(.\d+)?$")
                    .expect("无效的时间正则表达式");
                time_regex.is_match(value)
            }
            PatternType::DateTime => {
                // ISO 8601 日期时间验证
                let datetime_regex = 
                    Regex::new(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(.\d+)?(Z|[+-]\d{2}:\d{2})?$")
                    .expect("无效的日期时间正则表达式");
                datetime_regex.is_match(value)
            }
            PatternType::IPAddress => {
                // IP地址验证 (IPv4 或 IPv6)
                self.matches_ipv4(value) || self.matches_ipv6(value)
            }
            PatternType::IPv4 => self.matches_ipv4(value),
            PatternType::IPv6 => self.matches_ipv6(value),
            PatternType::UUID => {
                // UUID验证
                let uuid_regex = 
                    Regex::new(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
                    .expect("无效的UUID正则表达式");
                uuid_regex.is_match(value)
            }
            PatternType::Custom(_) => {
                // 自定义模式需要自行实现匹配逻辑
                false
            }
        }
    }

    /// 检查是否匹配IPv4地址
    fn matches_ipv4(&self, value: &str) -> bool {
        let ipv4_regex = Regex::new(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")
            .expect("无效的IPv4正则表达式");
        
        if ipv4_regex.is_match(value) {
            // 检查每个数字是否在0-255范围内
            return value
                .split('.')
                .map(|s| s.parse::<u8>())
                .all(|r| r.is_ok());
        }
        false
    }

    /// 检查是否匹配IPv6地址
    fn matches_ipv6(&self, value: &str) -> bool {
        // 简化版IPv6验证
        let ipv6_regex = Regex::new(r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$")
            .expect("无效的IPv6正则表达式");
        ipv6_regex.is_match(value)
    }

    /// 获取模式的描述
    pub fn description(&self) -> String {
        match self {
            PatternType::Regex(regex) => format!("正则表达式: {}", regex.as_str()),
            PatternType::Email => "电子邮件格式".to_string(),
            PatternType::Url => "URL格式".to_string(),
            PatternType::Date => "日期格式 (YYYY-MM-DD)".to_string(),
            PatternType::Time => "时间格式 (hh:mm:ss)".to_string(),
            PatternType::DateTime => "日期时间格式 (ISO 8601)".to_string(),
            PatternType::IPAddress => "IP地址".to_string(),
            PatternType::IPv4 => "IPv4地址".to_string(),
            PatternType::IPv6 => "IPv6地址".to_string(),
            PatternType::UUID => "UUID格式".to_string(),
            PatternType::Custom(name) => format!("自定义模式: {}", name),
        }
    }

    /// 获取模式对应的正则表达式
    pub fn get_regex(&self) -> std::result::Result<Regex, regex::Error> {
        match self {
            PatternType::Regex(regex) => Ok(regex.clone()),
            PatternType::Custom(pattern) => Regex::new(pattern),
            PatternType::Email => Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
            PatternType::Url => Regex::new(r"^(https?|ftp)://[^\s/$.?#].[^\s]*$"),
            PatternType::Date => Regex::new(r"^\d{4}-\d{2}-\d{2}$"),
            PatternType::Time => Regex::new(r"^\d{2}:\d{2}:\d{2}(.\d+)?$"),
            PatternType::DateTime => Regex::new(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(.\d+)?(Z|[+-]\d{2}:\d{2})?$"),
            PatternType::IPAddress => Regex::new(r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$|^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$"),
            PatternType::IPv4 => Regex::new(r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"),
            PatternType::IPv6 => Regex::new(r"^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$"),
            PatternType::UUID => Regex::new(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"),
        }
    }
}

/// 配置结构，定义模式验证器的参数
#[derive(Debug, Clone)]
pub struct PatternValidatorConfig {
    /// 验证模式类型
    pub pattern_type: PatternType,
    /// 自定义错误信息
    pub error_message: Option<String>,
    /// 验证错误的级别
    pub error_level: ErrorSeverity,
    /// 验证器唯一标识
    pub validation_id: Option<String>,
    /// 转义模式
    pub escape_pattern: bool,
    /// 不区分大小写
    pub case_insensitive: bool,
    /// 添加到验证结果的元数据
    pub metadata: Option<HashMap<String, String>>,
}

impl Default for PatternValidatorConfig {
    fn default() -> Self {
        Self {
            pattern_type: PatternType::Custom(String::new()),
            error_message: None,
            error_level: ErrorSeverity::Error,
            validation_id: None,
            escape_pattern: false,
            case_insensitive: false,
            metadata: None,
        }
    }
}

/// 模式验证器，用于根据正则表达式或预定义的模式验证字符串数据
#[derive(Debug, Clone)]
pub struct PatternValidator {
    /// 验证器配置
    config: PatternValidatorConfig,
    /// 缓存的正则表达式实例
    regex: Option<Arc<Regex>>,
}

impl PatternValidator {
    /// 创建新的模式验证器
    pub fn new(config: PatternValidatorConfig) -> std::result::Result<Self, regex::Error> {
        let mut regex_opts = regex::RegexBuilder::new("");
        
        if config.case_insensitive {
            regex_opts.case_insensitive(true);
        }
        
        if config.escape_pattern {
            if let PatternType::Custom(ref pattern) = config.pattern_type {
                let escaped = regex::escape(pattern);
                regex_opts = regex::RegexBuilder::new(&escaped);
            }
        }
        
        // 初始化正则表达式
        let regex = match config.pattern_type {
            PatternType::Custom(ref pattern) if pattern.is_empty() => None,
            _ => {
                let pattern = config.pattern_type.get_regex()?;
                Some(Arc::new(pattern))
            }
        };
        
        Ok(Self { config, regex })
    }
    
    /// 创建电子邮件验证器
    pub fn email(error_level: ErrorSeverity) -> std::result::Result<Self, regex::Error> {
        Self::new(PatternValidatorConfig {
            pattern_type: PatternType::Email,
            error_message: Some("无效的电子邮件地址".to_string()),
            error_level,
            ..Default::default()
        })
    }
    
    /// 创建URL验证器
    pub fn url(error_level: ErrorSeverity) -> std::result::Result<Self, regex::Error> {
        Self::new(PatternValidatorConfig {
            pattern_type: PatternType::Url,
            error_message: Some("无效的URL地址".to_string()),
            error_level,
            ..Default::default()
        })
    }
    
    /// 创建IP地址验证器
    pub fn ip_address(error_level: ErrorSeverity) -> std::result::Result<Self, regex::Error> {
        Self::new(PatternValidatorConfig {
            pattern_type: PatternType::IPAddress,
            error_message: Some("无效的IP地址".to_string()),
            error_level,
            ..Default::default()
        })
    }
    
    /// 创建日期验证器
    pub fn date(error_level: ErrorSeverity) -> std::result::Result<Self, regex::Error> {
        Self::new(PatternValidatorConfig {
            pattern_type: PatternType::Date,
            error_message: Some("无效的日期格式，应为YYYY-MM-DD".to_string()),
            error_level,
            ..Default::default()
        })
    }
    
    /// 创建日期时间验证器
    pub fn datetime(error_level: ErrorSeverity) -> std::result::Result<Self, regex::Error> {
        Self::new(PatternValidatorConfig {
            pattern_type: PatternType::DateTime,
            error_message: Some("无效的日期时间格式，应为YYYY-MM-DDThh:mm:ss".to_string()),
            error_level,
            ..Default::default()
        })
    }
    
    /// 创建UUID验证器
    pub fn uuid(error_level: ErrorSeverity) -> std::result::Result<Self, regex::Error> {
        Self::new(PatternValidatorConfig {
            pattern_type: PatternType::UUID,
            error_message: Some("无效的UUID格式".to_string()),
            error_level,
            ..Default::default()
        })
    }
    
    /// 创建信用卡号验证器
    pub fn credit_card(error_level: ErrorSeverity) -> Result<Self, regex::Error> {
        Self::new(PatternValidatorConfig {
            pattern_type: PatternType::Custom(r"^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9][0-9])[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})$".to_string()),
            error_message: Some("无效的信用卡号".to_string()),
            error_level,
            ..Default::default()
        })
    }
    
    /// 创建电话号码验证器
    pub fn phone_number(error_level: ErrorSeverity) -> Result<Self, regex::Error> {
        Self::new(PatternValidatorConfig {
            pattern_type: PatternType::Custom(r"^\+[1-9]\d{1,14}$".to_string()),
            error_message: Some("无效的电话号码格式".to_string()),
            error_level,
            ..Default::default()
        })
    }
    
    /// 创建使用自定义正则表达式的验证器
    pub fn custom(
        pattern: &str,
        error_message: Option<String>,
        error_level: ErrorSeverity,
    ) -> Result<Self, regex::Error> {
        Self::new(PatternValidatorConfig {
            pattern_type: PatternType::Custom(pattern.to_string()),
            error_message,
            error_level,
            ..Default::default()
        })
    }
    
    /// 获取验证器的配置
    pub fn config(&self) -> &PatternValidatorConfig {
        &self.config
    }
}

impl Validator for PatternValidator {
    fn validate(&self, data: &HashMap<String, DataValue>) -> ValidationResult {
        // 验证结果
        let mut result = ValidationResult::success();
        
        // 对每个字段进行验证
        for (field_name, value) in data {
            let field_result = self.validate_field(field_name, value);
            // 将DataValidationResult转换为ValidationResult
            result.errors.extend(field_result.errors.iter().map(|e| e.message.clone()));
            result.warnings.extend(field_result.warnings.iter().map(|w| w.message.clone()));
            if !field_result.is_valid {
                result.is_valid = false;
            }
        }
        
        result
    }

    fn validation_type(&self) -> ValidationType {
        ValidationType::Pattern
    }
}

impl PatternValidator {
    /// 验证单个字段
    fn validate_field(&self, field_name: &str, value: &DataValue) -> DataValidationResult {
        // 验证结果
        let mut result = DataValidationResult::default();
        
        // 提取字符串值
        let string_value = match value {
            DataValue::String(s) => s,
            _ => {
                // 非字符串类型不能应用模式验证
                let error = ValidationError {
                    code: "invalid_type".to_string(),
                    message: "模式验证只能应用于字符串类型".to_string(),
                    field: Some(field_name.to_string()),
                    value: Some(format!("{:?}", value)),
                    severity: self.config.error_level,
                    location: None,
                };
                result.add_error(error);
                return result;
            }
        };
        
        // 检查是否有正则表达式
        let regex = match &self.regex {
            Some(r) => r,
            None => {
                let warning = ValidationWarning {
                    code: "missing_pattern".to_string(),
                    message: "没有配置有效的模式".to_string(),
                    field: Some(field_name.to_string()),
                    suggestion: Some("请配置有效的模式".to_string()),
                    location: None,
                };
                result.warnings.push(warning);
                return result;
            }
        };
        
        // 执行模式匹配
        if !regex.is_match(string_value) {
            let message = self.config.error_message.clone().unwrap_or_else(|| {
                format!(
                    "值 '{}' 不匹配所需的模式: {}",
                    string_value,
                    self.config.pattern_type.description()
                )
            });
            
            let error = ValidationError {
                code: "pattern_mismatch".to_string(),
                message,
                field: Some(field_name.to_string()),
                value: Some(string_value.clone()),
                severity: self.config.error_level,
                location: None,
            };
            
            result.add_error(error);
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_email_validator() {
        let validator = PatternValidator::email(ErrorSeverity::Error).unwrap();
        
        // 有效的电子邮件
        let result = validator.validate(&DataValue::String("user@example.com".to_string()));
        assert!(result.is_valid());
        
        // 无效的电子邮件
        let result = validator.validate(&DataValue::String("invalid-email".to_string()));
        assert!(!result.is_valid());
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].level, ErrorSeverity::Error);
        
        // 非字符串类型
        let result = validator.validate(&DataValue::Integer(123));
        assert!(!result.is_valid());
    }
    
    #[test]
    fn test_url_validator() {
        let validator = PatternValidator::url(ErrorSeverity::Error).unwrap();
        
        // 有效的URL
        let result = validator.validate(&DataValue::String("https://example.com".to_string()));
        assert!(result.is_valid());
        
        // 无效的URL
        let result = validator.validate(&DataValue::String("invalid-url".to_string()));
        assert!(!result.is_valid());
    }
    
    #[test]
    fn test_date_validator() {
        let validator = PatternValidator::date(ErrorSeverity::Error).unwrap();
        
        // 有效的日期
        let result = validator.validate(&DataValue::String("2023-01-15".to_string()));
        assert!(result.is_valid());
        
        // 无效的日期
        let result = validator.validate(&DataValue::String("2023/01/15".to_string()));
        assert!(!result.is_valid());
    }
    
    #[test]
    fn test_custom_validator() {
        // 创建自定义验证器：只接受3位数字
        let validator = PatternValidator::custom(
            r"^\d{3}$",
            Some("必须是3位数字".to_string()),
            ErrorSeverity::Warning,
        )
        .unwrap();
        
        // 有效的值
        let result = validator.validate(&DataValue::String("123".to_string()));
        assert!(result.is_valid());
        
        // 无效的值
        let result = validator.validate(&DataValue::String("12".to_string()));
        assert!(!result.is_valid());
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].level, ErrorSeverity::Warning);
        assert_eq!(result.errors[0].message, "必须是3位数字");
    }
    
    #[test]
    fn test_case_insensitive() {
        // 创建不区分大小写的验证器
        let validator = PatternValidator::new(PatternValidatorConfig {
            pattern_type: PatternType::Custom("^abc$".to_string()),
            case_insensitive: true,
            ..Default::default()
        })
        .unwrap();
        
        // 测试大写
        let result = validator.validate(&DataValue::String("ABC".to_string()));
        assert!(result.is_valid());
        
        // 测试混合大小写
        let result = validator.validate(&DataValue::String("AbC".to_string()));
        assert!(result.is_valid());
    }
    
    #[test]
    fn test_escape_pattern() {
        // 创建转义模式的验证器，验证文本必须包含特殊字符 ".*+"
        let validator = PatternValidator::new(PatternValidatorConfig {
            pattern_type: PatternType::Custom(".*+".to_string()),
            escape_pattern: true,
            ..Default::default()
        })
        .unwrap();
        
        // 有效的值 - 包含 ".*+"
        let result = validator.validate(&DataValue::String("This contains .*+".to_string()));
        assert!(result.is_valid());
        
        // 无效的值 - 不包含 ".*+"
        let result = validator.validate(&DataValue::String("This does not match".to_string()));
        assert!(!result.is_valid());
    }
    
    #[test]
    fn test_uuid_validator() {
        let validator = PatternValidator::uuid(ErrorSeverity::Error).unwrap();
        
        // 有效的UUID
        let result = validator.validate(&DataValue::String("550e8400-e29b-41d4-a716-446655440000".to_string()));
        assert!(result.is_valid());
        
        // 无效的UUID
        let result = validator.validate(&DataValue::String("550e8400-e29b-XXXX-a716-446655440000".to_string()));
        assert!(!result.is_valid());
    }
    
    #[test]
    fn test_validation_result_metadata() {
        // 创建带有元数据的验证器
        let mut metadata = HashMap::new();
        metadata.insert("field".to_string(), "username".to_string());
        metadata.insert("rule".to_string(), "pattern".to_string());
        
        let validator = PatternValidator::new(PatternValidatorConfig {
            pattern_type: PatternType::Custom(r"^[a-zA-Z]+$".to_string()),
            validation_id: Some("alpha_only".to_string()),
            metadata: Some(metadata),
            ..Default::default()
        })
        .unwrap();
        
        // 无效的值 - 包含非字母字符
        let result = validator.validate(&DataValue::String("user123".to_string()));
        assert!(!result.is_valid());
        assert_eq!(result.errors.len(), 1);
        
        // 检查错误中的元数据和ID
        let error = &result.errors[0];
        assert_eq!(error.validation_id, Some("alpha_only".to_string()));
        assert!(error.metadata.is_some());
        assert_eq!(
            error.metadata.as_ref().unwrap().get("field").unwrap(),
            "username"
        );
    }
} 