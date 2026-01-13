use std::str::FromStr;
use log::debug;

use crate::data::loader::schema::FieldType;
use crate::error::Result;

/// 行分析器模块，用于处理CSV和其他基于行的文件格式
pub struct LineParser;

impl LineParser {
    /// 分析行数据并将其转换为适当的类型
    pub fn parse_field(value: &str) -> Result<(FieldType, Option<f64>)> {
        // 去除前后空格
        let value = value.trim();
        
        // 检查是否为空值
        if value.is_empty() {
            return Ok((FieldType::Text, None));
        }
        
        // 检查是否为布尔值
        if let Ok(bool_val) = bool::from_str(value) {
            return Ok((FieldType::Boolean, Some(if bool_val { 1.0 } else { 0.0 })));
        }
        
        // 检查是否为整数
        if let Ok(int_val) = i64::from_str(value) {
            return Ok((FieldType::Integer, Some(int_val as f64)));
        }
        
        // 检查是否为浮点数
        if let Ok(float_val) = f64::from_str(value) {
            return Ok((FieldType::Float, Some(float_val)));
        }
        
        // 检查是否为日期或时间戳
        if Self::is_date_format(value) {
            return Ok((FieldType::Date, None));
        }
        
        if Self::is_timestamp_format(value) {
            return Ok((FieldType::Timestamp, None));
        }
        
        // 默认为文本类型
        Ok((FieldType::Text, None))
    }
    
    /// 判断字符串是否为日期格式
    pub fn is_date_format(value: &str) -> bool {
        use chrono::NaiveDate;
        
        let date_formats = &[
            "%Y-%m-%d",     // 2022-01-01
            "%d/%m/%Y",     // 01/01/2022
            "%m/%d/%Y",     // 01/01/2022
            "%Y/%m/%d",     // 2022/01/01
            "%d-%m-%Y",     // 01-01-2022
            "%m-%d-%Y",     // 01-01-2022
        ];
        
        for &fmt in date_formats {
            if NaiveDate::parse_from_str(value, fmt).is_ok() {
                debug!("检测到日期格式: {} ({})", value, fmt);
                return true;
            }
        }
        
        false
    }
    
    /// 判断字符串是否为时间戳格式
    pub fn is_timestamp_format(value: &str) -> bool {
        use chrono::NaiveDateTime;
        
        let timestamp_formats = &[
            "%Y-%m-%d %H:%M:%S",       // 2022-01-01 12:34:56
            "%d/%m/%Y %H:%M:%S",       // 01/01/2022 12:34:56
            "%Y-%m-%dT%H:%M:%S",       // 2022-01-01T12:34:56
            "%Y-%m-%d %H:%M:%S%.3f",   // 2022-01-01 12:34:56.123
            "%Y-%m-%dT%H:%M:%S%.3fZ",  // 2022-01-01T12:34:56.123Z
        ];
        
        for &fmt in timestamp_formats {
            if NaiveDateTime::parse_from_str(value, fmt).is_ok() {
                debug!("检测到时间戳格式: {} ({})", value, fmt);
                return true;
            }
        }
        
        // 检查ISO 8601格式
        if value.len() > 19 && value.contains('T') && 
           (value.contains('Z') || value.contains('+') || value.contains('-')) {
            if let Ok(_) = chrono::DateTime::parse_from_rfc3339(value) {
                debug!("检测到ISO 8601时间戳: {}", value);
                return true;
            }
        }
        
        false
    }
    
    /// 分析字段类型并返回最佳推测
    pub fn analyze_field_types(values: &[&str]) -> FieldType {
        if values.is_empty() {
            return FieldType::Text;
        }
        
        let mut type_counts = std::collections::HashMap::new();
        
        // 分析每个值的类型
        for &value in values {
            let (field_type, _) = Self::parse_field(value)
                                    .unwrap_or((FieldType::Text, None));
            *type_counts.entry(field_type).or_insert(0) += 1;
        }
        
        // 找出最常见的类型
        type_counts.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(field_type, _)| field_type)
            .unwrap_or(FieldType::Text)
    }
    
    /// 规范化布尔值字符串
    pub fn normalize_boolean(value: &str) -> Option<bool> {
        match value.to_lowercase().trim() {
            "true" | "yes" | "y" | "1" | "t" => Some(true),
            "false" | "no" | "n" | "0" | "f" => Some(false),
            _ => None,
        }
    }
    
    /// 计算行中的字段数
    pub fn count_fields(line: &str, delimiter: char) -> usize {
        // 简单计数，不考虑复杂的CSV转义情况
        let mut count = 1;
        let mut in_quotes = false;
        
        for c in line.chars() {
            if c == '"' {
                in_quotes = !in_quotes;
            } else if c == delimiter && !in_quotes {
                count += 1;
            }
        }
        
        count
    }
} 