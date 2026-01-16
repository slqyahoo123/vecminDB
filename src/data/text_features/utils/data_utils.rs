// 数据工具模块
// 提供数据转换和字段处理功能

use crate::Result;
use serde_json::Value;

/// 将字符串转换为浮点数
pub fn convert_to_float(value: &str) -> Result<f32> {
    value.parse::<f32>().map_err(|e| crate::Error::invalid_argument(format!("无法转换为浮点数: {}", e)))
}

/// 从JSON对象中获取字段值
pub fn get_field_value(data: &Value, field: &str) -> Option<String> {
    match data.get(field) {
        Some(val) => {
            if val.is_string() {
                Some(val.as_str().unwrap().to_string())
            } else {
                Some(val.to_string())
            }
        }
        None => None,
    }
}

/// 检查JSON对象中是否存在字段
pub fn field_exists(data: &Value, field: &str) -> bool {
    data.get(field).is_some()
} 