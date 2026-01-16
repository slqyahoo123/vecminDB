use crate::Error;
use crate::Result;
use serde_json::Value;
use std::collections::HashMap;

/// 验证特征是否有效
pub fn validate_features(features: &[f32]) -> Result<()> {
    if features.is_empty() {
        return Err(Error::invalid_argument("特征向量为空".to_string()));
    }
    
    // 检查是否有NaN或无限值
    for (i, &val) in features.iter().enumerate() {
        if val.is_nan() {
            return Err(Error::invalid_argument(
                format!("特征向量第{}个元素为NaN", i)
            ));
        }
        
        if val.is_infinite() {
            return Err(Error::invalid_argument(
                format!("特征向量第{}个元素为无限值", i)
            ));
        }
    }
    
    Ok(())
}

/// 验证JSON数据是否有效
pub fn validate_json_data(data: &Value) -> Result<()> {
    if !data.is_object() && !data.is_array() {
        return Err(Error::invalid_argument(
            "JSON数据必须是对象或数组".to_string()
        ));
    }
    
    if data.is_object() {
        let obj = data.as_object().unwrap();
        if obj.is_empty() {
            return Err(Error::invalid_argument("JSON对象为空".to_string()));
        }
    }
    
    if data.is_array() {
        let arr = data.as_array().unwrap();
        if arr.is_empty() {
            return Err(Error::invalid_argument("JSON数组为空".to_string()));
        }
    }
    
    Ok(())
}

/// 验证字段名是否合法
pub fn validate_field_names(field_names: &[String]) -> Result<()> {
    let mut seen = HashMap::new();
    
    for (i, name) in field_names.iter().enumerate() {
        if name.is_empty() {
            return Err(Error::invalid_argument(
                format!("字段名第{}个元素为空", i)
            ));
        }
        
        if seen.contains_key(name) {
            return Err(Error::invalid_argument(
                format!("字段名重复: {}", name)
            ));
        }
        
        seen.insert(name.clone(), true);
    }
    
    Ok(())
} 