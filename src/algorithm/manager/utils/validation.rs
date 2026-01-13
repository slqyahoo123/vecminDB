// 验证工具

use crate::error::{Error, Result};
use crate::algorithm::types::Algorithm;

/// 验证算法基本属性
pub fn validate_algorithm_basic(algorithm: &Algorithm) -> Result<()> {
    if algorithm.id.is_empty() {
        return Err(Error::InvalidInput("算法ID不能为空".to_string()));
    }

    if algorithm.name.is_empty() {
        return Err(Error::InvalidInput("算法名称不能为空".to_string()));
    }

    if algorithm.code.is_empty() {
        return Err(Error::InvalidInput("算法代码不能为空".to_string()));
    }

    if algorithm.language.is_empty() {
        return Err(Error::InvalidInput("编程语言不能为空".to_string()));
    }

    if algorithm.version_string.is_empty() {
        return Err(Error::InvalidInput("算法版本不能为空".to_string()));
    }

    Ok(())
}

/// 验证算法代码长度
pub fn validate_algorithm_code_length(code: &str, max_length: usize) -> Result<()> {
    if code.len() > max_length {
        return Err(Error::InvalidInput(format!(
            "算法代码过长，最大长度: {} 字符", max_length
        )));
    }
    Ok(())
}

/// 验证算法参数
pub fn validate_algorithm_parameters(parameters: &serde_json::Value) -> Result<()> {
    if let Some(obj) = parameters.as_object() {
        for (key, value) in obj {
            if key.is_empty() {
                return Err(Error::InvalidInput("参数名不能为空".to_string()));
            }
            
            if value.is_null() {
                return Err(Error::InvalidInput(format!(
                    "参数值不能为null: {}", key
                )));
            }
        }
    }
    Ok(())
}

/// 验证算法类型
pub fn validate_algorithm_type(algorithm_type: &crate::algorithm::types::AlgorithmType) -> Result<()> {
    use crate::algorithm::types::AlgorithmType;
    
    match algorithm_type {
        AlgorithmType::MachineLearning |
        AlgorithmType::DataProcessing |
        AlgorithmType::Optimization |
        AlgorithmType::Custom => Ok(()),
        _ => Err(Error::InvalidInput(format!(
            "不支持的算法类型: {:?}", algorithm_type
        ))),
    }
} 