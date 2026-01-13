use crate::data::DataBatch;
use crate::data::DataSchema;
use crate::error::Result;

use crate::core::interfaces::ValidationResult;

// 兼容本模块原先的语义字段，通过扩展 details/metadata 表达
trait LoaderValidationExt {
    fn with_message(self, message: impl Into<String>) -> Self;
    fn with_invalid_records(self, count: usize) -> Self;
}

impl LoaderValidationExt for ValidationResult {
    fn with_message(mut self, message: impl Into<String>) -> Self {
        self.metadata.insert("message".into(), message.into());
        self
    }

    fn with_invalid_records(mut self, count: usize) -> Self {
        self.metadata.insert("invalid_records".into(), count.to_string());
        self
    }
}

/// 提供数据批次验证功能的特质
pub trait BatchValidator: Send + Sync {
    /// 验证数据批次
    fn validate_batch(&self, batch: &DataBatch, schema: Option<&DataSchema>) -> Result<ValidationResult>;
}

/// 标准数据批次验证器
pub struct StandardBatchValidator;

impl StandardBatchValidator {
    /// 创建新的标准验证器
    pub fn new() -> Self {
        Self {}
    }
}

impl BatchValidator for StandardBatchValidator {
    fn validate_batch(&self, batch: &DataBatch, schema: Option<&DataSchema>) -> Result<ValidationResult> {
        let mut result = ValidationResult::success().with_message("验证通过");
        
        // 检查批次是否为空
        if batch.records.is_empty() {
            result.is_valid = false;
            result = result.with_message("批次不包含任何记录");
            return Ok(result);
        }
        
        // 验证架构（如果提供）
        if let Some(expected_schema) = schema {
            if let Some(batch_schema) = &batch.schema {
                // 检查字段数量
                let schema_fields = expected_schema.fields().len();
                let batch_fields = batch_schema.fields().len();
                
                if schema_fields != batch_fields {
                    result.is_valid = false;
                    result = result.with_message(format!(
                        "字段数量不匹配: 预期 {}, 实际 {}",
                        schema_fields, batch_fields
                    ));
                    result.warnings.push("字段数量不匹配".to_string());
                    return Ok(result);
                }
                
                // 检查字段名称和类型
                for field in expected_schema.fields() {
                    if let Some(batch_field) = batch_schema.get_field(&field.name) {
                        if field.field_type != batch_field.field_type {
                            result.warnings.push(format!(
                                "字段 '{}' 类型不匹配: 预期 {:?}, 实际 {:?}",
                                field.name, field.field_type, batch_field.field_type
                            ));
                        }
                    } else {
                        result.warnings.push(format!(
                            "缺少字段: '{}'", field.name
                        ));
                    }
                }
                
                // 如果有警告，设置为无效
                if !result.warnings.is_empty() {
                    result.is_valid = false;
                    result = result.with_message(format!("架构验证失败, {} 个问题", result.warnings.len()));
                }
            }
        }
        
        // 验证记录
        let mut invalid_count = 0;
        for (i, record) in batch.records.iter().enumerate() {
            // 检查记录是否有所有必需字段
            if let Some(schema) = &batch.schema {
                for field in schema.fields() {
                    if !field.nullable {
                        // HashMap<String, DataValue> 使用 get 方法
                        match record.get(&field.name) {
                            Some(value) => {
                                if value.is_null() {
                                    invalid_count += 1;
                                    if result.warnings.len() < 10 { // 限制警告数量
                                        result.warnings.push(format!(
                                            "记录 #{} 的必需字段 '{}' 为空", i, field.name
                                        ));
                                    }
                                }
                            },
                            None => {
                                invalid_count += 1;
                                if result.warnings.len() < 10 {
                                    result.warnings.push(format!(
                                        "记录 #{} 缺少必需字段 '{}'", i, field.name
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        
        if invalid_count > 0 {
            result.is_valid = false;
            result = result.with_invalid_records(invalid_count)
                           .with_message(format!("发现 {} 条无效记录", invalid_count));
        }
        
        Ok(result)
    }
}

/// 批次验证管理器，支持多种验证策略
pub struct ValidationManager {
    validators: Vec<Box<dyn BatchValidator>>,
}

impl ValidationManager {
    /// 创建新的验证管理器
    pub fn new() -> Self {
        Self {
            validators: Vec::new(),
        }
    }

    /// 添加验证器
    pub fn add_validator<V: BatchValidator + 'static>(&mut self, validator: V) -> &mut Self {
        self.validators.push(Box::new(validator));
        self
    }

    /// 执行所有验证
    pub fn validate_batch(&self, batch: &DataBatch, schema: Option<&DataSchema>) -> Result<ValidationResult> {
        let mut final_result = ValidationResult::success();
        
        for validator in &self.validators {
            let result = validator.validate_batch(batch, schema)?;
            
            // 如果任何验证器返回无效结果，最终结果也是无效的
            if !result.is_valid {
                final_result.is_valid = false;
                if let Some(msg) = result.metadata.get("message").cloned() {
                    final_result = final_result.with_message(msg);
                }
                if let Some(cnt) = result.metadata.get("invalid_records").and_then(|s| s.parse::<usize>().ok()) {
                    final_result = final_result.with_invalid_records(cnt);
                }
            }
            
            // 合并所有警告
            for warning in &result.warnings {
                final_result.warnings.push(warning.clone());
            }
        }
        
        // 如果没有验证器，使用标准验证器
        if self.validators.is_empty() {
            let standard_validator = StandardBatchValidator::new();
            return standard_validator.validate_batch(batch, schema);
        }
        
        Ok(final_result)
    }
} 