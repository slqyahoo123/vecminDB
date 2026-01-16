use std::collections::HashMap;
use log::{info, debug, warn};
use crate::data::schema::DataSchema;
use crate::data::schema::schema::FieldType;
use crate::data::record::{Record, Value};
use crate::data::value::DataValue;
use crate::data::pipeline::{PipelineStage, PipelineContext, Result};
use crate::Error;

/// 验证类型枚举，定义支持的验证规则类型
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationType {
    /// 非空验证
    NotNull,
    /// 类型匹配验证
    TypeMatch,
    /// 范围验证（数值）
    Range,
    /// 格式验证（字符串）
    Format,
    /// 唯一性验证
    Unique,
    /// 引用验证
    Reference,
    /// 自定义验证
    Custom,
}

impl ValidationType {
    /// 从字符串转换到验证类型
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "notnull" | "not_null" => ValidationType::NotNull,
            "typematch" | "type_match" => ValidationType::TypeMatch, 
            "range" => ValidationType::Range,
            "format" => ValidationType::Format,
            "unique" => ValidationType::Unique,
            "reference" => ValidationType::Reference,
            "custom" => ValidationType::Custom,
            _ => {
                warn!("未知的验证类型: {}, 默认使用类型匹配验证", s);
                ValidationType::TypeMatch
            }
        }
    }
}

/// 验证规则结构，定义单个字段的验证规则
#[derive(Debug, Clone)]
pub struct ValidationRule {
    /// 字段名称
    pub field_name: String,
    /// 验证类型
    pub validation_type: ValidationType,
    /// 验证参数
    pub parameters: HashMap<String, String>,
    /// 错误消息模板
    pub error_message: Option<String>,
}

impl ValidationRule {
    /// 创建新的验证规则
    pub fn new(
        field_name: String, 
        validation_type: ValidationType, 
        parameters: HashMap<String, String>,
        error_message: Option<String>
    ) -> Self {
        ValidationRule {
            field_name,
            validation_type,
            parameters,
            error_message,
        }
    }

    /// 验证单个值是否符合规则
    pub fn validate(&self, value: &Value, schema: &DataSchema) -> Result<()> {
        match self.validation_type {
            ValidationType::NotNull => self.validate_not_null(value),
            ValidationType::TypeMatch => self.validate_type_match(value, schema),
            ValidationType::Range => self.validate_range(value),
            ValidationType::Format => self.validate_format(value),
            ValidationType::Unique => Ok(()), // 唯一性验证需要在数据集级别进行，此处跳过
            ValidationType::Reference => Ok(()), // 引用验证需要额外数据，此处跳过
            ValidationType::Custom => {
                // 生产级实现：执行自定义验证规则
                self.validate_custom(value)
            }
        }
    }

    // 非空验证
    fn validate_not_null(&self, value: &Value) -> Result<()> {
        // Value 枚举没有 is_null 方法，检查是否为 Data(DataValue::Null)
        if matches!(value, Value::Data(DataValue::Null)) {
            let error_msg = self.error_message.clone()
                .unwrap_or_else(|| format!("字段 '{}' 不能为空", self.field_name));
            return Err(Error::validation(error_msg));
        }
        Ok(())
    }

    // 类型匹配验证
    fn validate_type_match(&self, value: &Value, schema: &DataSchema) -> Result<()> {
        // Value 枚举没有 is_null 方法，检查是否为 Data(DataValue::Null)
        if matches!(value, Value::Data(DataValue::Null)) {
            return Ok(()); // 空值跳过类型验证
        }

        let field = match schema.get_field(&self.field_name) {
            Some(f) => f,
            None => return Err(Error::validation(
                format!("字段 '{}' 在模式中不存在", self.field_name)
            )),
        };

        // 检查值是否与字段类型匹配
        // FieldType 在 schema.rs 中定义为 Numeric, Text, Boolean 等
        // Value 是 Value::Data(DataValue::...) 的形式
        match field.field_type {
            FieldType::Text => {
                if !matches!(value, Value::Data(DataValue::String(_)) | Value::Data(DataValue::Text(_))) {
                    return self.type_mismatch_error(&field.field_type);
                }
            },
            FieldType::Numeric => {
                if !matches!(value, Value::Data(DataValue::Integer(_)) | Value::Data(DataValue::Float(_)) | Value::Data(DataValue::Number(_))) {
                    return self.type_mismatch_error(&field.field_type);
                }
            },
            FieldType::Boolean => {
                if !matches!(value, Value::Data(DataValue::Boolean(_))) {
                    return self.type_mismatch_error(&field.field_type);
                }
            },
            FieldType::DateTime => {
                if !matches!(value, Value::Data(DataValue::DateTime(_))) {
                    return self.type_mismatch_error(&field.field_type);
                }
            },
            FieldType::Array(_) => {
                if !matches!(value, Value::Data(DataValue::Array(_))) {
                    return self.type_mismatch_error(&field.field_type);
                }
            },
            FieldType::Object(_) => {
                // 对象类型需要特殊处理
                // 暂时跳过验证
            },
            _ => {
                warn!("不支持的字段类型验证: {:?}", field.field_type);
                // 其他类型暂时跳过验证
            }
        }

        Ok(())
    }

    // 范围验证
    fn validate_range(&self, value: &Value) -> Result<()> {
        // Value 枚举没有 is_null 方法，检查是否为 Data(DataValue::Null)
        if matches!(value, Value::Data(DataValue::Null)) {
            return Ok(()); // 空值跳过范围验证
        }

        let min = self.parameters.get("min").map(|s| s.parse::<f64>());
        let max = self.parameters.get("max").map(|s| s.parse::<f64>());

        let numeric_value = match value {
            Value::Data(DataValue::Integer(i)) => Ok(*i as f64),
            Value::Data(DataValue::Float(f)) => Ok(*f),
            Value::Data(DataValue::Number(n)) => Ok(*n),
            _ => Err(Error::validation(
                format!("字段 '{}' 不是数值类型，无法进行范围验证", self.field_name)
            )),
        }?;

        // 最小值验证
        if let Some(Ok(min_val)) = min {
            if numeric_value < min_val {
                let error_msg = self.error_message.clone()
                    .unwrap_or_else(|| format!("字段 '{}' 的值 {} 小于最小值 {}", 
                                    self.field_name, numeric_value, min_val));
                return Err(Error::validation(error_msg));
            }
        }

        // 最大值验证
        if let Some(Ok(max_val)) = max {
            if numeric_value > max_val {
                let error_msg = self.error_message.clone()
                    .unwrap_or_else(|| format!("字段 '{}' 的值 {} 大于最大值 {}", 
                                    self.field_name, numeric_value, max_val));
                return Err(Error::validation(error_msg));
            }
        }

        Ok(())
    }

    // 格式验证
    fn validate_format(&self, value: &Value) -> Result<()> {
        // Value 枚举没有 is_null 方法，检查是否为 Data(DataValue::Null)
        if matches!(value, Value::Data(DataValue::Null)) {
            return Ok(()); // 空值跳过格式验证
        }

        let pattern = match self.parameters.get("pattern") {
            Some(p) => p,
            None => return Err(Error::validation(
                format!("格式验证规则必须指定 'pattern' 参数")
            )),
        };

        let value_str = match value {
            Value::Data(DataValue::String(s)) => s,
            _ => return Err(Error::validation(
                format!("字段 '{}' 不是字符串类型，无法进行格式验证", self.field_name)
            )),
        };

        // 使用正则表达式验证
        let regex = match regex::Regex::new(pattern) {
            Ok(r) => r,
            Err(e) => return Err(Error::validation(
                format!("正则表达式 '{}' 无效: {}", pattern, e)
            )),
        };

        if !regex.is_match(value_str) {
            let error_msg = self.error_message.clone()
                .unwrap_or_else(|| format!("字段 '{}' 的值 '{}' 不符合格式要求: '{}'", 
                                self.field_name, value_str, pattern));
            return Err(Error::validation(error_msg));
        }

        Ok(())
    }

    // 生成类型不匹配错误
    fn type_mismatch_error(&self, expected_type: &FieldType) -> Result<()> {
        let error_msg = self.error_message.clone()
            .unwrap_or_else(|| format!("字段 '{}' 的值类型不匹配，期望类型: {:?}", 
                            self.field_name, expected_type));
        Err(Error::validation(error_msg))
    }
    
    // 自定义验证规则 - 生产级实现
    fn validate_custom(&self, value: &Value) -> Result<()> {
        // 从参数中获取自定义验证规则类型
        let rule_type = self.parameters.get("rule_type")
            .map(|v| v.as_str())
            .unwrap_or("default");
        
        match rule_type {
            "regex" => {
                // 正则表达式验证
                let pattern = self.parameters.get("pattern")
                    .map(|v| v.as_str())
                    .ok_or_else(|| Error::validation(
                        "自定义验证规则 'regex' 需要 'pattern' 参数"
                    ))?;
                
                let value_str = match value {
                    Value::Data(DataValue::String(s)) => s,
                    Value::Data(DataValue::Text(s)) => s,
                    _ => return Err(Error::validation(
                        format!("字段 '{}' 不是字符串类型，无法进行正则表达式验证", self.field_name)
                    )),
                };
                
                let regex = regex::Regex::new(pattern)
                    .map_err(|e| Error::validation(
                        format!("正则表达式 '{}' 无效: {}", pattern, e)
                    ))?;
                
                if !regex.is_match(value_str) {
                    let error_msg = self.error_message.clone()
                        .unwrap_or_else(|| format!("字段 '{}' 的值 '{}' 不符合正则表达式: '{}'", 
                                        self.field_name, value_str, pattern));
                    return Err(Error::validation(error_msg));
                }
            },
            "length" => {
                // 长度验证
                let min_len = self.parameters.get("min_length")
                    .and_then(|v| v.parse::<usize>().ok());
                let max_len = self.parameters.get("max_length")
                    .and_then(|v| v.parse::<usize>().ok());
                
                let value_str = match value {
                    Value::Data(DataValue::String(s)) => s,
                    Value::Data(DataValue::Text(s)) => s,
                    _ => return Err(Error::validation(
                        format!("字段 '{}' 不是字符串类型，无法进行长度验证", self.field_name)
                    )),
                };
                
                let len = value_str.len();
                
                if let Some(min) = min_len {
                    if len < min {
                        let error_msg = self.error_message.clone()
                            .unwrap_or_else(|| format!("字段 '{}' 的长度 {} 小于最小值 {}", 
                                            self.field_name, len, min));
                        return Err(Error::validation(error_msg));
                    }
                }
                
                if let Some(max) = max_len {
                    if len > max {
                        let error_msg = self.error_message.clone()
                            .unwrap_or_else(|| format!("字段 '{}' 的长度 {} 大于最大值 {}", 
                                            self.field_name, len, max));
                        return Err(Error::validation(error_msg));
                    }
                }
            },
            "enum" => {
                // 枚举值验证
                // allowed_values 应该是逗号分隔的字符串，例如 "value1,value2,value3"
                let allowed_values_str = self.parameters.get("allowed_values")
                    .map(|v| v.as_str())
                    .ok_or_else(|| Error::validation(
                        "自定义验证规则 'enum' 需要 'allowed_values' 参数（逗号分隔的字符串）"
                    ))?;
                
                let allowed_values: Vec<&str> = allowed_values_str.split(',').map(|s| s.trim()).collect();
                
                let value_str = match value {
                    Value::Data(DataValue::String(s)) => s.as_str(),
                    Value::Data(DataValue::Text(s)) => s.as_str(),
                    _ => return Err(Error::validation(
                        format!("字段 '{}' 不是字符串类型，无法进行枚举验证", self.field_name)
                    )),
                };
                
                if !allowed_values.contains(&value_str) {
                    let allowed_str = allowed_values.join(", ");
                    let error_msg = self.error_message.clone()
                        .unwrap_or_else(|| format!("字段 '{}' 的值 '{}' 不在允许的枚举值中: [{}]", 
                                        self.field_name, value_str, allowed_str));
                    return Err(Error::validation(error_msg));
                }
            },
            "custom_function" => {
                // 自定义函数验证（通过参数传递验证逻辑）
                // 这里可以实现更复杂的验证逻辑，例如调用外部验证函数
                // 当前实现：检查是否有自定义验证结果参数
                if let Some(expected_result) = self.parameters.get("expected_result") {
                    let value_str = format!("{:?}", value);
                    let expected_str = expected_result.as_str();
                    
                    // 简单的相等性检查
                    if value_str != expected_str {
                        let error_msg = self.error_message.clone()
                            .unwrap_or_else(|| format!("字段 '{}' 的自定义验证失败", self.field_name));
                        return Err(Error::validation(error_msg));
                    }
                } else {
                    // 如果没有指定验证逻辑，默认通过
                    warn!("自定义验证规则 'custom_function' 未指定验证逻辑，默认通过");
                }
            },
            _ => {
                // 未知的自定义验证类型
                let error_msg = self.error_message.clone()
                    .unwrap_or_else(|| format!("未知的自定义验证规则类型: '{}'", rule_type));
                return Err(Error::validation(error_msg));
            }
        }
        
        Ok(())
    }
}

/// 验证结果项
#[derive(Debug, Clone)]
pub struct ValidationResultItem {
    /// 记录索引
    pub record_index: usize,
    /// 字段名
    pub field_name: String,
    /// 错误消息
    pub error_message: String,
    /// 验证类型
    pub validation_type: ValidationType,
    /// 字段值
    pub value: Option<Value>,
}

use crate::core::interfaces::ValidationResult;

// 后续如需该模块的明细统计，可通过单独结构承载并序列化进 `details` 或另建领域专用类型

/// 数据验证器
pub struct DataValidator {
    /// 验证规则
    rules: Vec<ValidationRule>,
    /// 是否在首次错误时停止验证
    stop_on_first_error: bool,
    /// 错误限制数量，超过此数量停止验证
    error_limit: Option<usize>,
}

impl DataValidator {
    /// 创建新的数据验证器
    pub fn new() -> Self {
        DataValidator {
            rules: Vec::new(),
            stop_on_first_error: false,
            error_limit: None,
        }
    }

    /// 添加验证规则
    pub fn add_rule(&mut self, rule: ValidationRule) -> &mut Self {
        self.rules.push(rule);
        self
    }

    /// 设置首次错误停止
    pub fn with_stop_on_first_error(&mut self, stop: bool) -> &mut Self {
        self.stop_on_first_error = stop;
        self
    }

    /// 设置错误限制
    pub fn with_error_limit(&mut self, limit: usize) -> &mut Self {
        self.error_limit = Some(limit);
        self
    }

    /// 验证单条记录
    pub fn validate_record(
        &self, 
        record: &Record, 
        record_index: usize, 
        schema: &DataSchema
    ) -> Result<ValidationResult> {
        let mut result = ValidationResult::success().with_detail("scope", "record".to_string());
        result.metadata.insert("total_records".to_string(), "1".to_string());

        for rule in &self.rules {
            if let Some(value) = record.get_field(&rule.field_name) {
                if let Err(e) = rule.validate(value, schema) {
                    // Error::Validation 是元组变体
                    if let Error::Validation(msg) = e {
                        // 写入 metadata 附加明细
                        result.is_valid = false;
                        result.errors.push(format!("[{}] {}", rule.field_name, msg));
                        result.metadata.insert("last_error_field".to_string(), rule.field_name.clone());
                        result.metadata.insert("last_error_index".to_string(), record_index.to_string());
                        
                        if self.stop_on_first_error {
                            break;
                        }
                    } else {
                        // 其他类型错误直接返回
                        return Err(e);
                    }
                }
            } else if rule.validation_type == ValidationType::NotNull {
                // 字段不存在且有非空验证规则
                let error_msg = rule.error_message.clone()
                    .unwrap_or_else(|| format!("字段 '{}' 不存在但要求非空", rule.field_name));
                
                result.is_valid = false;
                result.errors.push(error_msg);
                result.metadata.insert("last_error_field".to_string(), rule.field_name.clone());
                result.metadata.insert("last_error_index".to_string(), record_index.to_string());
                
                if self.stop_on_first_error {
                    break;
                }
            }

            // 检查是否达到错误限制
            if let Some(limit) = self.error_limit {
                if result.errors.len() >= limit {
                    break;
                }
            }
        }

        Ok(result)
    }

    /// 验证记录集
    pub fn validate_records(
        &self, 
        records: &[Record], 
        schema: &DataSchema
    ) -> Result<ValidationResult> {
        let mut overall_result = ValidationResult::success().with_detail("scope", "records".to_string());
        overall_result.metadata.insert("total_records".to_string(), records.len().to_string());

        for (i, record) in records.iter().enumerate() {
            let record_result = self.validate_record(record, i, schema)?;
            if !record_result.is_valid {
                overall_result.is_valid = false;
                overall_result.errors.extend(record_result.errors);
                
                // 检查是否达到错误限制
                if let Some(limit) = self.error_limit {
                    if overall_result.errors.len() >= limit {
                        break;
                    }
                }
            }
        }

        Ok(overall_result)
    }

    /// 根据模式创建默认验证规则
    pub fn create_default_rules_from_schema(schema: &DataSchema) -> Vec<ValidationRule> {
        let mut rules = Vec::new();
        
        for field in schema.fields() {
            // 添加类型验证规则
            rules.push(ValidationRule::new(
                field.name.clone(),
                ValidationType::TypeMatch,
                HashMap::new(),
                None
            ));
            
            // 为非空字段添加非空验证规则
            // nullable 是 bool 类型，不是 Option<bool>
            if !field.nullable {
                rules.push(ValidationRule::new(
                    field.name.clone(),
                    ValidationType::NotNull,
                    HashMap::new(),
                    None
                ));
            }
        }
        
        rules
    }
}

/// 数据验证阶段
pub struct DataValidationStage {
    schema: Option<DataSchema>,
    rules: Vec<ValidationRule>,
    stop_on_first_error: bool,
    error_limit: Option<usize>,
}

impl DataValidationStage {
    /// 创建新的数据验证阶段
    pub fn new() -> Self {
        DataValidationStage {
            schema: None,
            rules: Vec::new(),
            stop_on_first_error: false,
            error_limit: Some(100), // 默认最多显示100个错误
        }
    }
    
    /// 添加验证规则
    pub fn add_rule(&mut self, rule: ValidationRule) -> &mut Self {
        self.rules.push(rule);
        self
    }
    
    /// 设置验证规则
    pub fn with_rules(&mut self, rules: Vec<ValidationRule>) -> &mut Self {
        self.rules = rules;
        self
    }
    
    /// 设置首次错误停止
    pub fn with_stop_on_first_error(&mut self, stop: bool) -> &mut Self {
        self.stop_on_first_error = stop;
        self
    }
    
    /// 设置错误限制
    pub fn with_error_limit(&mut self, limit: Option<usize>) -> &mut Self {
        self.error_limit = limit;
        self
    }
}

impl PipelineStage for DataValidationStage {
    fn name(&self) -> &str {
        "数据验证阶段"
    }

    fn description(&self) -> Option<&str> {
        Some("验证数据记录是否符合定义规则")
    }

    fn process(&self, ctx: &mut PipelineContext) -> Result<()> {
        info!("执行数据验证阶段");
        
        // 获取模式
        let schema = ctx.get_data::<DataSchema>("schema")
            .map_err(|_| Error::not_found("schema"))?;
        
        // 如果没有规则，从模式创建默认规则
        let mut rules = if self.rules.is_empty() && self.schema.is_some() {
            DataValidator::create_default_rules_from_schema(&schema)
        } else {
            self.rules.clone()
        };
        
        // 创建验证器
        let mut validator = DataValidator::new();
        validator.with_stop_on_first_error(self.stop_on_first_error);
        if let Some(limit) = self.error_limit {
            validator.with_error_limit(limit);
        }
        
        // 添加验证规则
        for rule in &rules {
            validator.add_rule(rule.clone());
        }
        
        // 获取记录批次
        let record_batches = ctx.get_data::<Vec<Vec<Record>>>("record_batches")
            .map_err(|_| Error::not_found("record_batches"))?;
        
        // 验证记录批次
        let mut overall_result = ValidationResult::success().with_detail("scope", "batches".to_string());
        let mut total_records = 0;
        
        for (i, records) in record_batches.iter().enumerate() {
            debug!("验证第 {} 批记录，包含 {} 条记录", i+1, records.len());
            total_records += records.len();
            
            let batch_result = validator.validate_records(records, &schema)?;
            if !batch_result.is_valid {
                debug!("第 {} 批记录验证失败，发现 {} 个错误", i+1, batch_result.errors.len());
                overall_result.is_valid = false;
                overall_result.errors.extend(batch_result.errors.clone());
                
                // 检查是否达到错误限制
                if let Some(limit) = self.error_limit {
                    if overall_result.errors.len() >= limit {
                        debug!("达到错误限制 {}，停止验证", limit);
                        break;
                    }
                }
            }
        }
        
        // 设置总记录数
        overall_result.metadata.insert("total_records".to_string(), total_records.to_string());
        
        // 将验证结果添加到上下文
        ctx.add_data("validation_result", overall_result.clone())?;
        
        // 日志记录验证结果
        if overall_result.is_valid {
            info!("数据验证成功，验证了 {} 条记录", total_records);
        } else {
            warn!("数据验证失败: 发现 {} 个错误，共验证 {} 条记录", 
                 overall_result.errors.len(), total_records);
        }
        
        Ok(())
    }
    
    fn can_process(&self, context: &PipelineContext) -> bool {
        context.data.contains_key("schema") && context.data.contains_key("record_batches")
    }
    
    fn metadata(&self) -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("requires_schema".to_string(), "true".to_string());
        metadata.insert("requires_record_batches".to_string(), "true".to_string());
        metadata.insert("rule_count".to_string(), self.rules.len().to_string());
        metadata
    }
} 