use crate::error::{Error, Result};
use crate::data::schema::DataSchema;
use std::collections::HashMap;
use crate::data::value::DataValue;
use super::Validator;
// use crate::core::interfaces::ValidationResult; // factory returns trait object; no direct use here
use super::TypeValidator;  // 直接从当前模块导入TypeValidator

use super::custom_validator::{CustomValidator, ValidationRule, RuleType};

/// 创建默认验证器
pub fn create_default_validator(schema: DataSchema) -> Result<Box<dyn Validator>> {
	// 创建类型验证器
	let validator = TypeValidator::new("default_validator", schema);
	Ok(Box::new(validator))
}

/// 根据验证器类型创建验证器
pub fn create_validator(
	validator_type: &str,
	schema: DataSchema,
	options: &HashMap<String, DataValue>,
) -> Result<Box<dyn Validator>> {
    match validator_type.to_lowercase().as_str() {
		"type" => {
			// 获取并应用配置选项
			let use_strict = get_option_bool(options, "strict", false);
			let ignore_undefined = get_option_bool(options, "ignore_undefined", true);
			let validator = TypeValidator::new("type_validator", schema)
				.with_strict_mode(use_strict)
				.ignore_undefined(ignore_undefined);
			Ok(Box::new(validator))
		},
        "custom" => {
            let name = get_option_str(options, "name", "custom_validator");
			
			let mut rules = Vec::new();
			
			if let Some(DataValue::Array(rules_array)) = options.get("rules") {
				for rule_value in rules_array {
					if let Some(rule_map) = rule_value.as_object() {
						// 提取规则类型
						let rule_type = rule_map.get("type")
							.and_then(|v| v.as_str())
							.ok_or_else(|| Error::invalid_argument("规则必须指定类型".to_string()))?;
						
						// 提取字段名
						let field_name = rule_map.get("field")
							.and_then(|v| v.as_str())
							.ok_or_else(|| Error::invalid_argument("规则必须指定字段名".to_string()))?;
						
						// 解析规则类型
						let rule_type = match rule_type {
							"not_null" => RuleType::NotNull,
							"min_length" => RuleType::MinLength,
							"max_length" => RuleType::MaxLength,
							"min_value" => RuleType::MinValue,
							"max_value" => RuleType::MaxValue,
							"regex" => RuleType::Regex,
							"enum" => RuleType::Enum,
							"expression" => RuleType::Expression,
							_ => return Err(Error::invalid_argument(format!("不支持的规则类型: {}", rule_type))),
						};
						
						// 创建规则
						let mut rule = ValidationRule::new(rule_type, field_name);
						
						// 处理参数
						if let Some(params) = rule_map.get("parameters").and_then(|v| v.as_object()) {
							for (key, value) in params {
								if let Some(value_str) = value.as_str() {
									rule = rule.add_parameter(key.clone(), value_str.to_string());
								}
							}
						}
						rules.push(rule);
					}
				}
			}
			
			let validator = CustomValidator::with_rules(name, schema, rules);
			Ok(Box::new(validator))
		},
		_ => Err(Error::invalid_argument(format!("未知的验证器类型: {}", validator_type)))
	}
}

fn get_option_bool(options: &HashMap<String, DataValue>, key: &str, default: bool) -> bool {
	options.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
} 

/// 便捷创建自定义验证器（直接传入规则）
pub fn create_custom_validator(
    name: &str,
    schema: DataSchema,
    rules: Vec<ValidationRule>,
) -> Result<Box<dyn Validator>> {
    let validator = CustomValidator::with_rules(name.to_string(), schema, rules);
    Ok(Box::new(validator))
}

fn get_option_str(options: &HashMap<String, DataValue>, key: &str, default: &str) -> String {
    options
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| default.to_string())
}

fn get_option_int(options: &HashMap<String, DataValue>, key: &str, default: i64) -> i64 {
    options
        .get(key)
        .and_then(|v| v.as_integer().or_else(|| v.as_number().map(|f| f as i64)))
        .unwrap_or(default)
}

fn get_option_float(options: &HashMap<String, DataValue>, key: &str, default: f64) -> f64 {
    options
        .get(key)
        .and_then(|v| v.as_number().or_else(|| v.as_integer().map(|i| i as f64)))
        .unwrap_or(default)
}