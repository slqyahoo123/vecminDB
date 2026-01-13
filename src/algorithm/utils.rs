use std::collections::HashMap;
use crate::error::{Error, Result};
use uuid::Uuid;
use serde::{Serialize, Deserialize};

/// 生成唯一算法ID
pub fn generate_id() -> String {
    Uuid::new_v4().to_string()
}

/// 算法验证错误
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmValidationError {
    /// 错误代码
    pub code: String,
    /// 错误消息
    pub message: String,
    /// 错误位置
    pub location: Option<String>,
    /// 错误详情
    pub details: Option<String>,
}

impl AlgorithmValidationError {
    /// 创建新的验证错误
    pub fn new(code: &str, message: &str) -> Self {
        Self {
            code: code.to_string(),
            message: message.to_string(),
            location: None,
            details: None,
        }
    }
    
    /// 设置错误位置
    pub fn with_location(mut self, location: &str) -> Self {
        self.location = Some(location.to_string());
        self
    }
    
    /// 设置错误详情
    pub fn with_details(mut self, details: &str) -> Self {
        self.details = Some(details.to_string());
        self
    }
}

/// 解析算法参数
pub fn parse_algorithm_params(
    params_str: &str,
    expected_types: &HashMap<String, String>,
) -> Result<HashMap<String, serde_json::Value>> {
    let params: HashMap<String, serde_json::Value> = match serde_json::from_str(params_str) {
        Ok(p) => p,
        Err(e) => return Err(Error::invalid_input(format!("无法解析算法参数: {}", e))),
    };
    
    let mut result = HashMap::new();
    
    // 验证参数类型
    for (name, expected_type) in expected_types {
        if let Some(value) = params.get(name) {
            let value_type = get_json_value_type(value);
            if !is_compatible_type(&value_type, expected_type) {
                return Err(Error::invalid_input(format!(
                    "参数 '{}' 类型错误: 预期 {}, 实际 {}",
                    name, expected_type, value_type
                )));
            }
            result.insert(name.clone(), value.clone());
        } else if expected_type.starts_with("required_") {
            return Err(Error::invalid_input(format!("缺少必需的参数: '{}'", name)));
        }
    }
    
    Ok(result)
}

/// 获取JSON值的类型
fn get_json_value_type(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(_) => "boolean".to_string(),
        serde_json::Value::Number(n) => {
            if n.is_i64() {
                "integer".to_string()
            } else {
                "number".to_string()
            }
        }
        serde_json::Value::String(_) => "string".to_string(),
        serde_json::Value::Array(_) => "array".to_string(),
        serde_json::Value::Object(_) => "object".to_string(),
    }
}

/// 检查类型兼容性
fn is_compatible_type(actual: &str, expected: &str) -> bool {
    if expected.starts_with("required_") {
        return is_compatible_type(actual, &expected[9..]);
    }
    
    match (actual, expected) {
        (a, e) if a == e => true,
        ("integer", "number") => true,
        ("number", "integer") => true,
        (_, "any") => true,
        (_, e) if e.contains("|") => {
            e.split("|").any(|t| is_compatible_type(actual, t.trim()))
        },
        (_, _) => false,
    }
}

/// 算法安全性检查
pub fn check_algorithm_security(code: &str) -> Result<(bool, Vec<String>)> {
    let prohibited_patterns = [
        "std::process::Command",
        "std::process::exit",
        "std::fs::remove",
        "std::fs::write",
        "std::env::set",
        "unsafe",
        "exec",
        "system(",
        "shell_exec",
        "passthru",
    ];
    
    let mut is_safe = true;
    let mut warnings = Vec::new();
    
    for pattern in &prohibited_patterns {
        if code.contains(pattern) {
            is_safe = false;
            warnings.push(format!("发现潜在危险代码: {}", pattern));
        }
    }
    
    // 检查是否使用网络API
    if code.contains("std::net") {
        warnings.push("使用网络功能可能需要额外权限".to_string());
    }
    
    // 检查文件系统操作
    if code.contains("std::fs") {
        warnings.push("使用文件系统操作可能需要额外权限".to_string());
    }
    
    Ok((is_safe, warnings))
}

/// 创建算法沙箱参数
pub fn create_sandbox_params(memory_limit: usize, cpu_limit: u64) -> HashMap<String, String> {
    let mut params = HashMap::new();
    params.insert("memory_limit".to_string(), memory_limit.to_string());
    params.insert("cpu_limit".to_string(), cpu_limit.to_string());
    params.insert("allow_network".to_string(), "false".to_string());
    params.insert("allow_filesystem".to_string(), "false".to_string());
    params
}

/// 计算算法代码哈希
pub fn calculate_code_hash(code: &str) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(code.as_bytes());
    let result = hasher.finalize();
    format!("{:x}", result)
}

/// 将JSON值转换为字符串表示
pub fn json_to_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => s.clone(),
        _ => value.to_string(),
    }
} 