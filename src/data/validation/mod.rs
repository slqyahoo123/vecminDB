// 数据验证模块，提供数据校验功能

use crate::error::Result;
use crate::data::schema::DataSchema;
// use crate::data::record::DataField; // not used in this module directly
use crate::data::record::Record;
use std::collections::HashMap;
// use crate::data::schema::FieldType; // not used here
// use log::{debug, warn, info}; // logging occurs in submodules
use crate::data::value::DataValue;
use std::time::Duration;
use serde::{Serialize, Deserialize};
use crate::core::interfaces::ValidationResult;

pub mod type_validator;
mod range_validator;
mod pattern_validator;
pub mod custom_validator;
mod validator_factory;
pub mod async_validator;

// 重新导出validator_factory中的公共函数
pub use validator_factory::{create_default_validator, create_validator, create_custom_validator};

// 从子模块重新导出类型
pub use type_validator::TypeValidator;
pub use range_validator::RangeValidator;
pub use pattern_validator::PatternValidator;
pub use custom_validator::{CustomValidator, ValidationRule, RuleType};
pub use async_validator::{AsyncValidator, CompositeAsyncValidator, CachedAsyncValidator};

/// 通用验证器特性，用于定义自定义验证逻辑
pub trait Validator: Send + Sync {
    /// 验证数据字段
    fn validate(&self, data: &HashMap<String, DataValue>) -> ValidationResult;
    
    /// 获取验证类型
    fn validation_type(&self) -> ValidationType;
}

/// 验证上下文
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationContext {
    /// 验证会话ID
    pub session_id: String,
    /// 字段路径
    pub field_path: Vec<String>,
    /// 验证配置
    pub config: ValidationConfig,
    /// 上下文元数据
    pub metadata: HashMap<String, String>,
    /// 验证开始时间
    pub start_time: std::time::SystemTime,
    /// 记录索引
    pub record_index: Option<usize>,
    /// 验证会话元数据
    pub session_metadata: HashMap<String, String>,
}

impl ValidationContext {
    /// 创建新的验证上下文
    pub fn new(session_id: String) -> Self {
        Self {
            session_id,
            field_path: Vec::new(),
            config: ValidationConfig::default(),
            metadata: HashMap::new(),
            start_time: std::time::SystemTime::now(),
            record_index: None,
            session_metadata: HashMap::new(),
        }
    }
    
    /// 添加字段路径
    pub fn push_field(&mut self, field: String) {
        self.field_path.push(field);
    }
    
    /// 移除字段路径
    pub fn pop_field(&mut self) {
        self.field_path.pop();
    }
    
    /// 获取当前字段路径
    pub fn current_path(&self) -> String {
        self.field_path.join(".")
    }
}

impl Default for ValidationContext {
    fn default() -> Self {
        Self::new("default".to_string())
    }
}

impl ValidationContext {
    /// 克隆上下文
    pub fn clone(&self) -> Self {
        Self {
            session_id: self.session_id.clone(),
            field_path: self.field_path.clone(),
            config: self.config.clone(),
            metadata: self.metadata.clone(),
            start_time: self.start_time,
            record_index: self.record_index,
            session_metadata: self.session_metadata.clone(),
        }
    }
}

/// 验证配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// 是否启用严格模式
    pub strict_mode: bool,
    /// 最大错误数
    pub max_errors: usize,
    /// 是否收集警告
    pub collect_warnings: bool,
    /// 验证超时时间（秒）
    pub timeout_seconds: u64,
    /// 最大并发验证数
    pub max_concurrent_validations: Option<usize>,
    /// 验证超时时间
    pub validation_timeout: Option<Duration>,
    /// 是否启用缓存
    pub enable_cache: bool,
    /// 缓存TTL
    pub cache_ttl: Duration,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strict_mode: false,
            max_errors: 100,
            collect_warnings: true,
            timeout_seconds: 30,
            max_concurrent_validations: Some(10),
            validation_timeout: Some(Duration::from_secs(30)),
            enable_cache: true,
            cache_ttl: Duration::from_secs(300),
        }
    }
}

/// 验证结果（数据校验域专用，避免与核心门面同名冲突）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataValidationResult {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub metadata: HashMap<String, String>,
    pub validation_time: Duration,
    pub validated_count: usize,
    pub error_count: usize,
    pub warning_count: usize,
}

impl DataValidationResult {
    /// 创建成功的验证结果
    pub fn success(validated_count: usize, validation_time: Duration) -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            metadata: HashMap::new(),
            validation_time,
            validated_count,
            error_count: 0,
            warning_count: 0,
        }
    }

    /// 创建失败的验证结果
    pub fn failure(errors: Vec<ValidationError>, validation_time: Duration) -> Self {
        let error_count = errors.len();
        Self {
            is_valid: false,
            errors,
            warnings: Vec::new(),
            metadata: HashMap::new(),
            validation_time,
            validated_count: 0,
            error_count,
            warning_count: 0,
        }
    }

    /// 添加错误
    pub fn add_error(&mut self, error: ValidationError) {
        self.errors.push(error);
        self.error_count += 1;
        self.is_valid = false;
    }

    /// 添加警告
    pub fn add_warning(&mut self, warning: ValidationWarning) {
        self.warnings.push(warning);
        self.warning_count += 1;
    }

    /// 合并验证结果
    pub fn merge(&mut self, other: DataValidationResult) {
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
        self.metadata.extend(other.metadata);
        self.validated_count += other.validated_count;
        self.error_count += other.error_count;
        self.warning_count += other.warning_count;
        self.validation_time += other.validation_time;
        
        if !other.is_valid {
            self.is_valid = false;
        }
    }

    /// 获取错误摘要
    pub fn get_error_summary(&self) -> String {
        if self.errors.is_empty() {
            "No errors".to_string()
        } else {
            format!("{} errors found", self.errors.len())
        }
    }

    /// 获取详细报告
    pub fn get_detailed_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str(&format!("Validation Result: {}\n", 
            if self.is_valid { "PASSED" } else { "FAILED" }));
        report.push_str(&format!("Validated items: {}\n", self.validated_count));
        report.push_str(&format!("Errors: {}\n", self.error_count));
        report.push_str(&format!("Warnings: {}\n", self.warning_count));
        report.push_str(&format!("Validation time: {:?}\n", self.validation_time));
        
        if !self.errors.is_empty() {
            report.push_str("\nErrors:\n");
            for (i, error) in self.errors.iter().enumerate() {
                report.push_str(&format!("  {}. {}\n", i + 1, error.message));
            }
        }
        
        if !self.warnings.is_empty() {
            report.push_str("\nWarnings:\n");
            for (i, warning) in self.warnings.iter().enumerate() {
                report.push_str(&format!("  {}. {}\n", i + 1, warning.message));
            }
        }
        
        report
    }
}

impl Default for DataValidationResult {
    fn default() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            metadata: HashMap::new(),
            validation_time: Duration::from_secs(0),
            validated_count: 0,
            error_count: 0,
            warning_count: 0,
        }
    }
}

/// 验证错误
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub code: String,
    pub message: String,
    pub field: Option<String>,
    pub value: Option<String>,
    pub severity: ErrorSeverity,
    pub location: Option<ValidationLocation>,
}

impl ValidationError {
    pub fn new(code: String, message: String) -> Self {
        Self {
            code,
            message,
            field: None,
            value: None,
            severity: ErrorSeverity::Error,
            location: None,
        }
    }

    pub fn with_field(mut self, field: String) -> Self {
        self.field = Some(field);
        self
    }

    pub fn with_value(mut self, value: String) -> Self {
        self.value = Some(value);
        self
    }

    pub fn with_severity(mut self, severity: ErrorSeverity) -> Self {
        self.severity = severity;
        self
    }

    pub fn with_location(mut self, location: ValidationLocation) -> Self {
        self.location = Some(location);
        self
    }
}

/// 验证警告
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    pub code: String,
    pub message: String,
    pub field: Option<String>,
    pub suggestion: Option<String>,
    pub location: Option<ValidationLocation>,
}

impl ValidationWarning {
    pub fn new(code: String, message: String) -> Self {
        Self {
            code,
            message,
            field: None,
            suggestion: None,
            location: None,
        }
    }

    pub fn with_field(mut self, field: String) -> Self {
        self.field = Some(field);
        self
    }

    pub fn with_suggestion(mut self, suggestion: String) -> Self {
        self.suggestion = Some(suggestion);
        self
    }

    pub fn with_location(mut self, location: ValidationLocation) -> Self {
        self.location = Some(location);
        self
    }
}

/// 错误严重程度
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// 验证位置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationLocation {
    pub line: Option<usize>,
    pub column: Option<usize>,
    pub offset: Option<usize>,
    pub context: Option<String>,
}

impl ValidationLocation {
    pub fn new() -> Self {
        Self {
            line: None,
            column: None,
            offset: None,
            context: None,
        }
    }

    pub fn with_line(mut self, line: usize) -> Self {
        self.line = Some(line);
        self
    }

    pub fn with_column(mut self, column: usize) -> Self {
        self.column = Some(column);
        self
    }

    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    pub fn with_context(mut self, context: String) -> Self {
        self.context = Some(context);
        self
    }
}

/// 验证类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationType {
    /// 类型验证
    Type,
    /// 范围验证
    Range,
    /// 模式验证
    Pattern,
    /// 非空验证
    NotNull,
    /// 唯一性验证
    Unique,
    /// 自定义验证
    Custom,
}

/// 数据验证器特性
pub trait DataValidator: Send + Sync {
    /// 获取验证器名称
    fn name(&self) -> &str;
    
    /// 获取验证类型
    fn validation_type(&self) -> ValidationType;
    
    /// 验证单条记录
    fn validate_record(&self, record: &Record, schema: &DataSchema, index: usize) -> Result<ValidationResult>;
    
    /// 验证多条记录
    fn validate(&self, records: &[Record]) -> Result<ValidationResult> {
        let schema = self.get_schema()?;
        
        let mut result = ValidationResult::success();
        result.metadata.insert("total_records".to_string(), records.len().to_string());
        
        for (i, record) in records.iter().enumerate() {
            let record_result = self.validate_record(record, &schema, i)?;
            
            if !record_result.is_valid {
                let current_count = result.metadata.get("invalid_records")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(0);
                result.metadata.insert("invalid_records".to_string(), (current_count + 1).to_string());
            }
            
            result.merge(record_result);
        }
        
        Ok(result)
    }
    
    /// 获取模式
    fn get_schema(&self) -> Result<DataSchema>;
}

// 注释掉重复定义的函数
// pub fn create_default_validator(schema: &DataSchema) -> Result<Box<dyn DataValidator>> {
//     let factory = ValidatorFactory::new();
//     factory.create_default_validator(schema)
// }

// pub fn create_validator(validator_type: &str, schema: &DataSchema, options: &HashMap<String, String>) -> Result<Box<dyn DataValidator>> {
//     let factory = ValidatorFactory::new();
//     factory.create_validator(validator_type, schema, options)
// } 