/// 验证模块
/// 
/// 提供算法、输入输出、语法、安全性等各种验证函数

use crate::{Result, Error};
use crate::core::interfaces::ValidationResult as CoreValidationResult;
use crate::core::types::CoreAlgorithmDefinition;
use std::collections::HashMap;
use std::sync::RwLock;

// 类型别名
type AlgorithmDefinition = CoreAlgorithmDefinition;

/// 验证算法ID
pub(crate) fn validate_algorithm_id(algo_id: &str) -> Result<()> {
    if algo_id.is_empty() {
        return Err(Error::InvalidInput("算法ID不能为空".to_string()));
    }
    
    if algo_id.len() > 100 {
        return Err(Error::InvalidInput("算法ID长度不能超过100个字符".to_string()));
    }
    
    // 检查算法ID格式
    if !algo_id.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
        return Err(Error::InvalidInput("算法ID只能包含字母、数字、连字符和下划线".to_string()));
    }
    
    Ok(())
}

/// 验证输入张量
pub(crate) fn validate_input_tensors(inputs: &[crate::core::UnifiedTensorData]) -> Result<()> {
    if inputs.is_empty() {
        return Err(Error::InvalidInput("输入张量不能为空".to_string()));
    }
    
    for (i, tensor) in inputs.iter().enumerate() {
        validate_single_tensor(i, tensor)?;
    }
    
    Ok(())
}

/// 验证单个张量
pub(crate) fn validate_single_tensor(index: usize, tensor: &crate::core::UnifiedTensorData) -> Result<()> {
    // 检查张量形状
    if tensor.shape.is_empty() {
        return Err(Error::InvalidInput(
            format!("张量 {} 的形状不能为空", index)
        ));
    }
    
    // 检查张量数据
    if tensor.data.is_empty() {
        return Err(Error::InvalidInput(
            format!("张量 {} 的数据不能为空", index)
        ));
    }
    
    // 验证形状与数据大小的一致性
    let expected_size: usize = tensor.shape.iter().product();
    if tensor.data.len() != expected_size {
        return Err(Error::InvalidInput(
            format!("张量 {} 的数据大小与形状不匹配: 期望 {}, 实际 {}", 
                   index, expected_size, tensor.data.len())
        ));
    }
    
    // 检查数值的有效性
    for (i, &value) in tensor.data.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::InvalidInput(
                format!("张量 {} 的第 {} 个元素不是有限数值: {}", index, i, value)
            ));
        }
    }
    
    Ok(())
}

/// 验证算法输出
pub(crate) fn validate_algorithm_outputs(outputs: &[crate::core::UnifiedTensorData]) -> Result<()> {
    if outputs.is_empty() {
        return Err(Error::InvalidInput("算法输出不能为空".to_string()));
    }
    
    for (i, output) in outputs.iter().enumerate() {
        if output.shape.is_empty() {
            return Err(Error::InvalidInput(
                format!("输出张量 {} 的形状不能为空", i)
            ));
        }
        
        if output.data.is_empty() {
            return Err(Error::InvalidInput(
                format!("输出张量 {} 的数据不能为空", i)
            ));
        }
        
        // 验证形状与数据大小的一致性
        let expected_size: usize = output.shape.iter().product();
        if output.data.len() != expected_size {
            return Err(Error::InvalidInput(
                format!("输出张量 {} 的数据大小与形状不匹配: 期望 {}, 实际 {}", 
                       i, expected_size, output.data.len())
            ));
        }
        
        // 检查数值的有效性
        for (j, &value) in output.data.iter().enumerate() {
            if !value.is_finite() {
                return Err(Error::InvalidInput(
                    format!("输出张量 {} 的第 {} 个元素不是有限数值: {}", i, j, value)
                ));
            }
        }
    }
    
    Ok(())
}

/// 验证语法
pub(crate) async fn validate_syntax(algo_code: &str, result: &mut CoreValidationResult) -> Result<()> {
    if algo_code.is_empty() {
        result.errors.push("算法代码不能为空".to_string());
        return Ok(());
    }
    
    // 检查基本的语法结构
    if !algo_code.contains("function") && !algo_code.contains("def") {
        result.warnings.push("算法代码可能缺少函数定义".to_string());
    }
    
    // 检查是否有明显的语法错误
    if algo_code.contains("undefined") || algo_code.contains("null") {
        result.errors.push("算法代码包含未定义的变量或空值".to_string());
    }
    
    Ok(())
}

/// 检查安全性
pub(crate) async fn check_security(algo_code: &str, result: &mut CoreValidationResult) -> Result<()> {
    // 检查危险操作
    let dangerous_patterns = [
        "eval(", "exec(", "system(", "subprocess",
        "os.system", "subprocess.call", "subprocess.Popen",
        "import os", "import subprocess", "import sys"
    ];
    
    for pattern in &dangerous_patterns {
        if algo_code.contains(pattern) {
            result.errors.push(
                format!("算法代码包含危险操作: {}", pattern)
            );
        }
    }
    
    // 检查文件操作
    if algo_code.contains("open(") || algo_code.contains("file(") {
        result.warnings.push("算法代码包含文件操作，可能存在安全风险".to_string());
    }
    
    Ok(())
}

/// 分析性能
pub(crate) async fn check_performance(_algo_code: &str, result: &mut CoreValidationResult) -> Result<()> {
    // 添加性能提示
    result.warnings.push("建议使用向量化操作以提高性能".to_string());
    result.warnings.push("考虑使用缓存机制减少重复计算".to_string());
    
    Ok(())
}

/// 检查依赖
pub(crate) async fn check_dependencies(algo_code: &str, result: &mut CoreValidationResult) -> Result<()> {
    // 检查常见的依赖
    let common_deps = ["numpy", "pandas", "scikit-learn", "tensorflow", "torch"];
    
    for dep in &common_deps {
        if algo_code.contains(dep) {
            result.metadata.insert(format!("dependency_{}", dep), "required".to_string());
        }
    }
    
    Ok(())
}

/// 验证算法定义
pub(crate) fn validate_algorithm_definition(algo_def: &AlgorithmDefinition) -> Result<()> {
    if algo_def.name.is_empty() {
        return Err(Error::InvalidInput("算法名称不能为空".to_string()));
    }
    
    if algo_def.version.is_empty() {
        return Err(Error::InvalidInput("算法版本不能为空".to_string()));
    }
    
    if algo_def.source_code.is_empty() {
        return Err(Error::InvalidInput("算法代码不能为空".to_string()));
    }
    
    // 注意：algorithm_type 是枚举类型，不需要检查是否为空
    // 原代码中的检查可能有误，这里移除
    
    Ok(())
}

/// 检查算法唯一性
pub(crate) fn check_algorithm_uniqueness(
    name: &str,
    algorithm_cache: &RwLock<HashMap<String, AlgorithmDefinition>>
) -> Result<()> {
    let cache = algorithm_cache.read()
        .map_err(|_| Error::locks_poison("算法缓存读取锁获取失败：无法检查算法唯一性"))?;
    for (_, algo_def) in cache.iter() {
        if algo_def.name == name {
            return Err(Error::InvalidInput(
                format!("算法名称已存在: {}", name)
            ));
        }
    }
    Ok(())
}

/// 验证核心张量输入
pub(crate) fn validate_input_core_tensors(inputs: &[crate::core::types::CoreTensorData]) -> Result<()> {
    if inputs.is_empty() {
        return Err(Error::InvalidInput("输入张量不能为空".to_string()));
    }
    
    for (i, tensor) in inputs.iter().enumerate() {
        validate_single_core_tensor(i, tensor)?;
    }
    
    Ok(())
}

/// 验证单个核心张量
pub(crate) fn validate_single_core_tensor(index: usize, tensor: &crate::core::types::CoreTensorData) -> Result<()> {
    // 检查张量形状
    if tensor.shape.is_empty() {
        return Err(Error::InvalidInput(
            format!("张量 {} 的形状不能为空", index)
        ));
    }
    
    // 检查张量数据
    if tensor.data.is_empty() {
        return Err(Error::InvalidInput(
            format!("张量 {} 的数据不能为空", index)
        ));
    }
    
    // 验证形状与数据大小的一致性
    let expected_size: usize = tensor.shape.iter().product();
    if tensor.data.len() != expected_size {
        return Err(Error::InvalidInput(
            format!("张量 {} 的数据大小与形状不匹配: 期望 {}, 实际 {}", 
                   index, expected_size, tensor.data.len())
        ));
    }
    
    // 检查数值的有效性
    for (i, &value) in tensor.data.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::InvalidInput(
                format!("张量 {} 的第 {} 个元素不是有限数值: {}", index, i, value)
            ));
        }
    }
    
    Ok(())
}

/// 验证核心张量输出
pub(crate) fn validate_core_tensor_outputs(outputs: &[crate::core::types::CoreTensorData]) -> Result<()> {
    if outputs.is_empty() {
        return Err(Error::InvalidInput("算法输出不能为空".to_string()));
    }
    
    for (i, output) in outputs.iter().enumerate() {
        if output.shape.is_empty() {
            return Err(Error::InvalidInput(
                format!("输出张量 {} 的形状不能为空", i)
            ));
        }
        
        if output.data.is_empty() {
            return Err(Error::InvalidInput(
                format!("输出张量 {} 的数据不能为空", i)
            ));
        }
        
        // 验证形状与数据大小的一致性
        let expected_size: usize = output.shape.iter().product();
        if output.data.len() != expected_size {
            return Err(Error::InvalidInput(
                format!("输出张量 {} 的数据大小与形状不匹配: 期望 {}, 实际 {}", 
                       i, expected_size, output.data.len())
            ));
        }
        
        // 检查数值的有效性
        for (j, &value) in output.data.iter().enumerate() {
            if !value.is_finite() {
                return Err(Error::InvalidInput(
                    format!("输出张量 {} 的第 {} 个元素不是有限数值: {}", i, j, value)
                ));
            }
        }
    }
    
    Ok(())
}

