use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::data::{DataSchema, DataBatch, DataValue};
use crate::data::schema::schema::FieldType;
use log::{debug, info, warn, error};

use super::core::{DataValidator, ValidationError};
use crate::core::interfaces::ValidationResult;

/// 字段类型验证器
pub struct TypeValidator {
    /// 验证器名称
    name: String,
    
    /// 需要验证的字段
    fields: HashSet<String>,
    
    /// 是否检查所有字段
    check_all_fields: bool,
    
    /// 是否允许Null值
    allow_null: bool,
}

impl TypeValidator {
    /// 创建新的类型验证器
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            fields: HashSet::new(),
            check_all_fields: false,
            allow_null: true,
        }
    }
    
    /// 添加需要验证的字段
    pub fn add_field<S: Into<String>>(&mut self, field_name: S) -> &mut Self {
        self.fields.insert(field_name.into());
        self
    }
    
    /// 添加多个需要验证的字段
    pub fn add_fields<S: Into<String>, I: IntoIterator<Item = S>>(&mut self, field_names: I) -> &mut Self {
        for name in field_names {
            self.fields.insert(name.into());
        }
        self
    }
    
    /// 设置是否检查所有字段
    pub fn check_all_fields(&mut self, check_all: bool) -> &mut Self {
        self.check_all_fields = check_all;
        self
    }
    
    /// 设置是否允许Null值
    pub fn allow_null(&mut self, allow: bool) -> &mut Self {
        self.allow_null = allow;
        self
    }
    
    /// 检查值是否与字段类型匹配
    fn check_type_match(&self, value: &DataValue, field_type: &FieldType) -> bool {
        // 如果允许null且值为null，则直接返回true
        if self.allow_null && value.is_null() {
            return true;
        }
        
        match (value, field_type) {
            (DataValue::Null, _) => self.allow_null,
            (DataValue::Integer(_) | DataValue::Float(_), FieldType::Numeric) => true,
            (DataValue::Boolean(_), FieldType::Boolean) => true,
            (DataValue::String(_) | DataValue::Text(_), FieldType::Text) => true,
            (DataValue::DateTime(_), FieldType::DateTime) => true,
            (DataValue::Array(_), FieldType::Array(_)) => true,
            (DataValue::Object(_), FieldType::Object(_)) => true,
            (DataValue::Binary(_), FieldType::Custom(_)) => true,
            _ => false,
        }
    }
    
    /// 添加所有字段到验证器中
    pub fn with_schema_fields(&mut self, schema: &DataSchema) -> &mut Self {
        // 使用HashSet添加所有字段
        let field_names: HashSet<String> = schema.fields()
            .iter()
            .map(|field| field.name().to_string())
            .collect();
            
        self.fields.extend(field_names);
        self.check_all_fields = true;
        self
    }
    
    /// 基于字段类型添加字段
    pub fn add_fields_by_type(&mut self, schema: &DataSchema, field_type: &FieldType) -> &mut Self {
        for field in schema.fields() {
            if field.field_type() == field_type {
                self.fields.insert(field.name().to_string());
            }
        }
        self
    }
    
    /// 自动检测并排除Null值异常高的字段
    pub fn exclude_sparse_fields(&mut self, batch: &DataBatch, threshold: f64) -> &mut Self {
        let schema = match batch.schema() {
            Some(s) => s,
            None => return self,
        };
        let fields = schema.fields();
        let record_count = batch.records().len();
        
        if record_count == 0 {
            return self;
        }
        
        for field in fields {
            let field_name = field.name();
            let null_count = batch.records().iter()
                .filter(|record| {
                    record.get(field_name)
                        .map(|v| v.is_null())
                        .unwrap_or(true)
                })
                .count();
                
            let null_ratio = null_count as f64 / record_count as f64;
            
            if null_ratio > threshold {
                self.fields.remove(field_name);
            }
        }
        
        self
    }
}

impl DataValidator for TypeValidator {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn validate(&self, batch: &DataBatch, schema: &DataSchema) -> Result<ValidationResult> {
        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            score: Some(1.0),
            metadata: std::collections::HashMap::new(),
        };
        
        let fields = schema.fields();
        
        // 获取字段名和类型
        let mut field_info = HashMap::new();
        for field in fields {
            if self.check_all_fields || self.fields.contains(field.name()) {
                field_info.insert(field.name().to_string(), field.field_type().clone());
            }
        }
        
        // 验证每条记录
        for (record_idx, record) in batch.records().iter().enumerate() {
            for (field_name, field_type) in &field_info {
                if let Some(value) = record.get(field_name) {
                    if !self.check_type_match(value, field_type) {
                        result.is_valid = false;
                        result.errors.push(format!(
                            "记录 #{} 字段 '{}' 的值类型 {:?} 与期望类型 {:?} 不匹配",
                            record_idx,
                            field_name,
                            std::any::type_name_of_val(value),
                            field_type
                        ));
                    }
                }
            }
        }
        
        Ok(result)
    }
}

/// 值范围验证器
pub struct RangeValidator {
    /// 验证器名称
    name: String,
    
    /// 数值字段的最小值限制
    min_values: HashMap<String, f64>,
    
    /// 数值字段的最大值限制
    max_values: HashMap<String, f64>,
    
    /// 字符串字段的最小长度限制
    min_lengths: HashMap<String, usize>,
    
    /// 字符串字段的最大长度限制
    max_lengths: HashMap<String, usize>,
}

impl RangeValidator {
    /// 创建新的范围验证器
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            min_values: HashMap::new(),
            max_values: HashMap::new(),
            min_lengths: HashMap::new(),
            max_lengths: HashMap::new(),
        }
    }
    
    /// 设置数值字段的最小值
    pub fn set_min_value<S: Into<String>>(&mut self, field_name: S, min_value: f64) -> &mut Self {
        self.min_values.insert(field_name.into(), min_value);
        self
    }
    
    /// 设置数值字段的最大值
    pub fn set_max_value<S: Into<String>>(&mut self, field_name: S, max_value: f64) -> &mut Self {
        self.max_values.insert(field_name.into(), max_value);
        self
    }
    
    /// 设置数值字段的值范围
    pub fn set_value_range<S: Into<String>>(&mut self, field_name: S, min_value: f64, max_value: f64) -> &mut Self {
        let field = field_name.into();
        self.min_values.insert(field.clone(), min_value);
        self.max_values.insert(field, max_value);
        self
    }
    
    /// 设置字符串字段的最小长度
    pub fn set_min_length<S: Into<String>>(&mut self, field_name: S, min_length: usize) -> &mut Self {
        self.min_lengths.insert(field_name.into(), min_length);
        self
    }
    
    /// 设置字符串字段的最大长度
    pub fn set_max_length<S: Into<String>>(&mut self, field_name: S, max_length: usize) -> &mut Self {
        self.max_lengths.insert(field_name.into(), max_length);
        self
    }
    
    /// 设置字符串字段的长度范围
    pub fn set_length_range<S: Into<String>>(&mut self, field_name: S, min_length: usize, max_length: usize) -> &mut Self {
        let field = field_name.into();
        self.min_lengths.insert(field.clone(), min_length);
        self.max_lengths.insert(field, max_length);
        self
    }
}

impl DataValidator for RangeValidator {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn validate(&self, batch: &DataBatch, schema: &DataSchema) -> Result<ValidationResult> {
        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            score: Some(1.0),
            metadata: std::collections::HashMap::new(),
        };
        
        // 验证每条记录
        for (record_idx, record) in batch.records().iter().enumerate() {
            // 验证数值范围
            for (field_name, min_value) in &self.min_values {
                if let Some(value) = record.get(field_name) {
                    match value {
                        DataValue::Integer(i) => {
                            if (*i as f64) < *min_value {
                                result.is_valid = false;
                                result.errors.push(format!(
                                    "记录 #{} 字段 '{}' 的值 {} 小于最小值 {}",
                                    record_idx, field_name, i, min_value
                                ));
                            }
                        },
                        DataValue::Float(f) => {
                            if *f < *min_value {
                                result.is_valid = false;
                                result.errors.push(format!(
                                    "记录 #{} 字段 '{}' 的值 {} 小于最小值 {}",
                                    record_idx, field_name, f, min_value
                                ));
                            }
                        },
                        _ => {} // 忽略非数值类型
                    }
                }
            }
            
            for (field_name, max_value) in &self.max_values {
                if let Some(value) = record.get(field_name) {
                    match value {
                        DataValue::Integer(i) => {
                            if (*i as f64) > *max_value {
                                result.is_valid = false;
                                result.errors.push(format!(
                                    "记录 #{} 字段 '{}' 的值 {} 大于最大值 {}",
                                    record_idx, field_name, i, max_value
                                ));
                            }
                        },
                        DataValue::Float(f) => {
                            if *f > *max_value {
                                result.is_valid = false;
                                result.errors.push(format!(
                                    "记录 #{} 字段 '{}' 的值 {} 大于最大值 {}",
                                    record_idx, field_name, f, max_value
                                ));
                            }
                        },
                        _ => {} // 忽略非数值类型
                    }
                }
            }
            
            // 验证字符串长度
            for (field_name, min_length) in &self.min_lengths {
                if let Some(DataValue::String(s)) = record.get(field_name) {
                    if s.len() < *min_length {
                        result.is_valid = false;
                        result.errors.push(format!(
                            "记录 #{} 字段 '{}' 的字符串长度 {} 小于最小长度 {}",
                            record_idx, field_name, s.len(), min_length
                        ));
                    }
                }
            }
            
            for (field_name, max_length) in &self.max_lengths {
                if let Some(DataValue::String(s)) = record.get(field_name) {
                    if s.len() > *max_length {
                        result.is_valid = false;
                        result.errors.push(format!(
                            "记录 #{} 字段 '{}' 的字符串长度 {} 大于最大长度 {}",
                            record_idx, field_name, s.len(), max_length
                        ));
                    }
                }
            }
        }
        
        Ok(result)
    }
}

/// 非空验证器
pub struct NotNullValidator {
    /// 验证器名称
    name: String,
    
    /// 必须非空的字段
    required_fields: HashSet<String>,
}

impl NotNullValidator {
    /// 创建新的非空验证器
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            required_fields: HashSet::new(),
        }
    }
    
    /// 添加必须非空的字段
    pub fn add_required_field<S: Into<String>>(&mut self, field_name: S) -> &mut Self {
        self.required_fields.insert(field_name.into());
        self
    }
    
    /// 添加多个必须非空的字段
    pub fn add_required_fields<S: Into<String>, I: IntoIterator<Item = S>>(&mut self, field_names: I) -> &mut Self {
        for name in field_names {
            self.required_fields.insert(name.into());
        }
        self
    }
}

impl DataValidator for NotNullValidator {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn validate(&self, batch: &DataBatch, schema: &DataSchema) -> Result<ValidationResult> {
        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            score: Some(1.0),
            metadata: std::collections::HashMap::new(),
        };
        
        // 验证每条记录
        for (record_idx, record) in batch.records().iter().enumerate() {
            for field_name in &self.required_fields {
                if let Some(value) = record.get(field_name) {
                    if value.is_null() {
                        result.is_valid = false;
                        result.errors.push(format!(
                            "记录 #{} 字段 '{}' 不能为空",
                            record_idx, field_name
                        ));
                    }
                } else {
                    result.is_valid = false;
                    result.errors.push(format!(
                        "记录 #{} 字段 '{}' 不能为空",
                        record_idx, field_name
                    ));
                }
            }
        }
        
        Ok(result)
    }
}

/// 唯一性验证器
pub struct UniquenessValidator {
    /// 验证器名称
    name: String,
    
    /// 需要检查唯一性的字段或字段组合
    unique_fields: Vec<Vec<String>>,
}

impl UniquenessValidator {
    /// 创建新的唯一性验证器
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            unique_fields: Vec::new(),
        }
    }
    
    /// 添加需要唯一的单个字段
    pub fn add_unique_field<S: Into<String>>(&mut self, field_name: S) -> &mut Self {
        self.unique_fields.push(vec![field_name.into()]);
        self
    }
    
    /// 添加需要联合唯一的字段组合
    pub fn add_unique_combination<S: Into<String>, I: IntoIterator<Item = S>>(&mut self, field_names: I) -> &mut Self {
        let fields: Vec<String> = field_names.into_iter().map(|s| s.into()).collect();
        if !fields.is_empty() {
            self.unique_fields.push(fields);
        }
        self
    }
}

impl DataValidator for UniquenessValidator {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn validate(&self, batch: &DataBatch, schema: &DataSchema) -> Result<ValidationResult> {
        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            score: Some(1.0),
            metadata: std::collections::HashMap::new(),
        };
        
        // 对每组唯一字段进行验证
        for unique_group in &self.unique_fields {
            // 检查所有字段是否存在
            let mut missing_fields = Vec::new();
            
            for field_name in unique_group {
                // 检查字段是否存在于 schema 中
                if !schema.fields().iter().any(|f| f.name() == field_name) {
                    missing_fields.push(field_name.clone());
                }
            }
            
            if !missing_fields.is_empty() {
                result.warnings.push(format!(
                    "跳过唯一性检查，以下字段不存在: {}",
                    missing_fields.join(", ")
                ));
                continue;
            }
            
            // 检查唯一性
            let mut seen_values: HashMap<String, usize> = HashMap::new();
            
            for (record_idx, record) in batch.records().iter().enumerate() {
                // 获取这组字段的值
                let mut values = Vec::new();
                for field_name in unique_group {
                    if let Some(value) = record.get(field_name) {
                        values.push(value.clone());
                    } else {
                        values.push(DataValue::Null);
                    }
                }
                
                // 检查是否已经出现过（使用序列化后的字符串作为键）
                let key = serde_json::to_string(&values)
                    .unwrap_or_else(|_| format!("{:?}", values));
                if let Some(&prev_idx) = seen_values.get(&key) {
                    result.is_valid = false;
                    result.errors.push(format!(
                        "唯一性约束违反: 字段 [{}] 的值在记录 #{} 和 #{} 中重复",
                        unique_group.join(", "),
                        prev_idx,
                        record_idx
                    ));
                } else {
                    seen_values.insert(key, record_idx);
                }
            }
        }
        
        Ok(result)
    }
}

/// 高级验证错误处理器
pub struct ErrorHandler {
    /// 处理器名称
    name: String,
    /// 错误处理模式
    mode: ErrorHandlingMode,
    /// 记录错误日志的级别
    log_level: LogLevel,
}

/// 错误处理模式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorHandlingMode {
    /// 严格模式：任何错误都会中断处理
    Strict,
    /// 宽容模式：会记录错误但继续处理
    Lenient,
    /// 忽略模式：完全忽略错误
    Ignore,
}

/// 日志级别
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// 调试级别
    Debug,
    /// 信息级别
    Info,
    /// 警告级别
    Warn,
    /// 错误级别
    Error,
}

impl ErrorHandler {
    /// 创建新的错误处理器
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            mode: ErrorHandlingMode::Lenient,
            log_level: LogLevel::Warn,
        }
    }
    
    /// 设置错误处理模式
    pub fn with_mode(mut self, mode: ErrorHandlingMode) -> Self {
        self.mode = mode;
        self
    }
    
    /// 设置日志级别
    pub fn with_log_level(mut self, level: LogLevel) -> Self {
        self.log_level = level;
        self
    }
    
    /// 处理验证错误
    pub fn handle_validation_error(&self, error: &ValidationError) -> Result<()> {
        // 记录错误
        match self.log_level {
            LogLevel::Debug => debug!("[{}] 验证错误: {}", self.name, error.message),
            LogLevel::Info => info!("[{}] 验证错误: {}", self.name, error.message),
            LogLevel::Warn => warn!("[{}] 验证错误: {}", self.name, error.message),
            LogLevel::Error => error!("[{}] 验证错误: {}", self.name, error.message),
        }
        
        // 根据模式处理错误
        match self.mode {
            ErrorHandlingMode::Strict => {
                Err(Error::validation(&format!(
                    "验证失败: {}{}",
                    error.message,
                    error.field.as_ref().map_or(String::new(), |f| format!(" (字段: {})", f))
                )))
            },
            ErrorHandlingMode::Lenient => {
                // 记录但不中断
                debug!("[{}] 忽略验证错误并继续: {}", self.name, error.message);
                Ok(())
            },
            ErrorHandlingMode::Ignore => {
                // 完全忽略
                Ok(())
            },
        }
    }
    
    /// 处理验证结果
    pub fn handle_validation_result(&self, result: &ValidationResult) -> Result<()> {
        if !result.is_valid {
            // 记录整体验证状态
            info!("[{}] 验证失败: 有 {} 个错误, {} 个警告", 
                 self.name, result.errors.len(), result.warnings.len());
            
            // 记录每个错误详情
            for (i, error) in result.errors.iter().enumerate() {
                debug!("[{}] 错误 #{}: {}", self.name, i + 1, error);
            }
            
            // 记录每个警告详情
            for (i, warning) in result.warnings.iter().enumerate() {
                debug!("[{}] 警告 #{}: {}", self.name, i + 1, warning);
            }
            
            // 根据模式处理错误
            match self.mode {
                ErrorHandlingMode::Strict => {
                    Err(Error::validation(&format!(
                        "验证失败: {} 个错误, {} 个警告",
                        result.errors.len(), result.warnings.len()
                    )))
                },
                ErrorHandlingMode::Lenient | ErrorHandlingMode::Ignore => {
                    // 记录但不中断或完全忽略
                    Ok(())
                },
            }
        } else {
            // 验证通过
            info!("[{}] 验证通过", self.name);
            Ok(())
        }
    }
    
    /// 创建自定义验证错误
    pub fn create_error<S: Into<String>>(&self, message: S, field: Option<String>, record_index: Option<usize>) -> ValidationError {
        let error = ValidationError {
            field,
            message: message.into(),
            record_index,
        };
        
        // 记录错误
        match self.log_level {
            LogLevel::Debug => debug!("[{}] 创建验证错误: {}", self.name, error.message),
            LogLevel::Info => info!("[{}] 创建验证错误: {}", self.name, error.message),
            LogLevel::Warn => warn!("[{}] 创建验证错误: {}", self.name, error.message),
            LogLevel::Error => error!("[{}] 创建验证错误: {}", self.name, error.message),
        }
        
        error
    }
}

/// 复合验证工具
pub struct CompositeValidator {
    /// 验证器名称
    name: String,
    /// 子验证器列表
    validators: Vec<Box<dyn DataValidator>>,
    /// 错误处理器
    error_handler: ErrorHandler,
}

impl CompositeValidator {
    /// 创建新的复合验证器
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            validators: Vec::new(),
            error_handler: ErrorHandler::new("composite_validator")
                .with_mode(ErrorHandlingMode::Lenient),
        }
    }
    
    /// 添加验证器
    pub fn add_validator<V: DataValidator + 'static>(&mut self, validator: V) -> &mut Self {
        self.validators.push(Box::new(validator));
        self
    }
    
    /// 设置错误处理器
    pub fn with_error_handler(&mut self, handler: ErrorHandler) -> &mut Self {
        self.error_handler = handler;
        self
    }
    
    /// 设置错误处理模式
    pub fn with_error_mode(&mut self, mode: ErrorHandlingMode) -> &mut Self {
        self.error_handler = self.error_handler.with_mode(mode);
        self
    }
}

impl DataValidator for CompositeValidator {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn validate(&self, batch: &DataBatch, schema: &DataSchema) -> Result<ValidationResult> {
        // 合并所有验证器的结果
        let mut combined_result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            score: Some(1.0),
            metadata: std::collections::HashMap::new(),
        };
        
        debug!("[{}] 开始验证，使用 {} 个验证器", self.name, self.validators.len());
        
        // 对每个验证器执行验证
        for (i, validator) in self.validators.iter().enumerate() {
            debug!("[{}] 执行验证器 #{}: {}", self.name, i + 1, validator.name());
            
            match validator.validate(batch, schema) {
                Ok(result) => {
                    // 合并验证结果
                    combined_result.is_valid = combined_result.is_valid && result.is_valid;
                    combined_result.errors.extend(result.errors);
                    combined_result.warnings.extend(result.warnings);
                    
                    // 记录每个验证器的结果
                    if !result.is_valid {
                        info!("[{}] 验证器 '{}' 发现 {} 个错误", 
                             self.name, validator.name(), result.errors.len());
                    } else {
                        debug!("[{}] 验证器 '{}' 通过验证", self.name, validator.name());
                    }
                },
                Err(e) => {
                    // 验证器自身出错
                    error!("[{}] 验证器 '{}' 执行失败: {}", self.name, validator.name(), e);
                    
                    // 根据错误处理模式决定是否继续
                    match self.error_handler.mode {
                        ErrorHandlingMode::Strict => {
                            return Err(Error::validation(&format!(
                                "验证器 '{}' 执行失败: {}", validator.name(), e
                            )));
                        },
                        _ => {
                            // 添加为验证错误并继续
                            combined_result.is_valid = false;
                            combined_result.errors.push(format!(
                                "验证器 '{}' 执行失败: {}",
                                validator.name(), e
                            ));
                        }
                    }
                }
            }
        }
        
        // 记录整体验证结果
        if combined_result.is_valid {
            info!("[{}] 所有验证器通过验证", self.name);
        } else {
            warn!("[{}] 验证失败: {} 个错误, {} 个警告", 
                 self.name, combined_result.errors.len(), combined_result.warnings.len());
        }
        
        Ok(combined_result)
    }
}

/// 共享验证器集合，使用Arc实现引用计数共享
pub struct SharedValidators {
    /// 验证器集合
    validators: Vec<Arc<dyn DataValidator + Send + Sync>>,
    
    /// 验证器名称到索引的映射
    validator_map: HashMap<String, usize>,
}

impl SharedValidators {
    /// 创建新的共享验证器集合
    pub fn new() -> Self {
        Self {
            validators: Vec::new(),
            validator_map: HashMap::new(),
        }
    }
    
    /// 添加验证器
    pub fn add_validator<V: DataValidator + Send + Sync + 'static>(&mut self, validator: V) -> &mut Self {
        let name = validator.name().to_string();
        let validator = Arc::new(validator);
        
        self.validator_map.insert(name, self.validators.len());
        self.validators.push(validator);
        
        self
    }
    
    /// 获取验证器
    pub fn get_validator(&self, name: &str) -> Option<Arc<dyn DataValidator + Send + Sync>> {
        if let Some(&idx) = self.validator_map.get(name) {
            if idx < self.validators.len() {
                return Some(Arc::clone(&self.validators[idx]));
            }
        }
        None
    }
    
    /// 移除验证器
    pub fn remove_validator(&mut self, name: &str) -> bool {
        if let Some(&idx) = self.validator_map.get(name) {
            self.validators.remove(idx);
            self.validator_map.remove(name);
            
            // 更新映射表中的索引
            for (_, index) in self.validator_map.iter_mut() {
                if *index > idx {
                    *index -= 1;
                }
            }
            
            true
        } else {
            false
        }
    }
    
    /// 获取所有验证器名称
    pub fn get_validator_names(&self) -> Vec<String> {
        self.validator_map.keys().cloned().collect()
    }
    
    /// 验证数据批次
    pub fn validate(&self, batch: &DataBatch, schema: &DataSchema) -> Result<ValidationResult> {
        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            score: Some(1.0),
            metadata: std::collections::HashMap::new(),
        };
        
        for validator in &self.validators {
            match validator.validate(batch, schema) {
                Ok(sub_result) => {
                    if !sub_result.is_valid {
                        result.is_valid = false;
                    }
                    result.errors.extend(sub_result.errors);
                    result.warnings.extend(sub_result.warnings);
                },
                Err(e) => {
                    warn!("验证器 {} 执行失败: {}", validator.name(), e);
                    result.is_valid = false;
                    result.errors.push(format!(
                        "验证器 {} 执行失败: {}",
                        validator.name(), e
                    ));
                }
            }
        }
        
        Ok(result)
    }
} 