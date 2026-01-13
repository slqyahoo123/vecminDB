// Filter Processor
// 数据过滤处理器

use std::collections::HashMap;
use std::path::Path;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use crate::data::processor::processor_impl::DataProcessor;
use crate::data::types::DataFormat;
use crate::data::processor::config::ProcessorConfig;
use crate::error::{Error, Result};

/// 过滤条件类型
#[derive(Debug, Clone)]
pub enum FilterCondition {
    /// 等于
    Equal(String),
    /// 不等于
    NotEqual(String),
    /// 大于
    GreaterThan(f64),
    /// 小于
    LessThan(f64),
    /// 大于等于
    GreaterThanOrEqual(f64),
    /// 小于等于
    LessThanOrEqual(f64),
    /// 包含
    Contains(String),
    /// 不包含
    NotContains(String),
    /// 以...开头
    StartsWith(String),
    /// 以...结尾
    EndsWith(String),
    /// 正则表达式
    Regex(String),
    /// 在范围内
    InRange(f64, f64),
    /// 不在范围内
    NotInRange(f64, f64),
    /// 为空
    IsNull,
    /// 不为空
    IsNotNull,
}

impl FilterCondition {
    /// 解析过滤条件字符串
    pub fn parse(condition_str: &str) -> Result<Self> {
        let parts: Vec<&str> = condition_str.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(Error::invalid_input("过滤条件格式错误，应为 'type:value'"));
        }
        
        match parts[0].to_lowercase().as_str() {
            "eq" => Ok(FilterCondition::Equal(parts[1].to_string())),
            "ne" => Ok(FilterCondition::NotEqual(parts[1].to_string())),
            "gt" => {
                let value = parts[1].parse::<f64>()
                    .map_err(|_| Error::invalid_input("数值过滤条件需要数字值"))?;
                Ok(FilterCondition::GreaterThan(value))
            },
            "lt" => {
                let value = parts[1].parse::<f64>()
                    .map_err(|_| Error::invalid_input("数值过滤条件需要数字值"))?;
                Ok(FilterCondition::LessThan(value))
            },
            "gte" => {
                let value = parts[1].parse::<f64>()
                    .map_err(|_| Error::invalid_input("数值过滤条件需要数字值"))?;
                Ok(FilterCondition::GreaterThanOrEqual(value))
            },
            "lte" => {
                let value = parts[1].parse::<f64>()
                    .map_err(|_| Error::invalid_input("数值过滤条件需要数字值"))?;
                Ok(FilterCondition::LessThanOrEqual(value))
            },
            "contains" => Ok(FilterCondition::Contains(parts[1].to_string())),
            "not_contains" => Ok(FilterCondition::NotContains(parts[1].to_string())),
            "starts_with" => Ok(FilterCondition::StartsWith(parts[1].to_string())),
            "ends_with" => Ok(FilterCondition::EndsWith(parts[1].to_string())),
            "regex" => Ok(FilterCondition::Regex(parts[1].to_string())),
            "range" => {
                let range_parts: Vec<&str> = parts[1].split(',').collect();
                if range_parts.len() != 2 {
                    return Err(Error::invalid_input("范围条件需要两个数值，用逗号分隔"));
                }
                let min = range_parts[0].parse::<f64>()
                    .map_err(|_| Error::invalid_input("范围条件需要数字值"))?;
                let max = range_parts[1].parse::<f64>()
                    .map_err(|_| Error::invalid_input("范围条件需要数字值"))?;
                Ok(FilterCondition::InRange(min, max))
            },
            "not_range" => {
                let range_parts: Vec<&str> = parts[1].split(',').collect();
                if range_parts.len() != 2 {
                    return Err(Error::invalid_input("范围条件需要两个数值，用逗号分隔"));
                }
                let min = range_parts[0].parse::<f64>()
                    .map_err(|_| Error::invalid_input("范围条件需要数字值"))?;
                let max = range_parts[1].parse::<f64>()
                    .map_err(|_| Error::invalid_input("范围条件需要数字值"))?;
                Ok(FilterCondition::NotInRange(min, max))
            },
            "is_null" => Ok(FilterCondition::IsNull),
            "is_not_null" => Ok(FilterCondition::IsNotNull),
            _ => Err(Error::invalid_input(&format!("未知的过滤条件类型: {}", parts[0]))),
        }
    }
    
    /// 判断值是否满足过滤条件
    pub fn matches(&self, value: &str) -> Result<bool> {
        match self {
            FilterCondition::Equal(target) => Ok(value == target),
            FilterCondition::NotEqual(target) => Ok(value != target),
            FilterCondition::GreaterThan(target) => {
                let val = value.parse::<f64>()
                    .map_err(|_| Error::invalid_input("无法将值解析为数字"))?;
                Ok(val > *target)
            },
            FilterCondition::LessThan(target) => {
                let val = value.parse::<f64>()
                    .map_err(|_| Error::invalid_input("无法将值解析为数字"))?;
                Ok(val < *target)
            },
            FilterCondition::GreaterThanOrEqual(target) => {
                let val = value.parse::<f64>()
                    .map_err(|_| Error::invalid_input("无法将值解析为数字"))?;
                Ok(val >= *target)
            },
            FilterCondition::LessThanOrEqual(target) => {
                let val = value.parse::<f64>()
                    .map_err(|_| Error::invalid_input("无法将值解析为数字"))?;
                Ok(val <= *target)
            },
            FilterCondition::Contains(target) => Ok(value.contains(target)),
            FilterCondition::NotContains(target) => Ok(!value.contains(target)),
            FilterCondition::StartsWith(target) => Ok(value.starts_with(target)),
            FilterCondition::EndsWith(target) => Ok(value.ends_with(target)),
            FilterCondition::Regex(pattern) => {
                let regex = regex::Regex::new(pattern)
                    .map_err(|e| Error::invalid_input(&format!("正则表达式错误: {}", e)))?;
                Ok(regex.is_match(value))
            },
            FilterCondition::InRange(min, max) => {
                let val = value.parse::<f64>()
                    .map_err(|_| Error::invalid_input("无法将值解析为数字"))?;
                Ok(val >= *min && val <= *max)
            },
            FilterCondition::NotInRange(min, max) => {
                let val = value.parse::<f64>()
                    .map_err(|_| Error::invalid_input("无法将值解析为数字"))?;
                Ok(val < *min || val > *max)
            },
            FilterCondition::IsNull => Ok(value.is_empty() || value == "null" || value == "NULL"),
            FilterCondition::IsNotNull => Ok(!value.is_empty() && value != "null" && value != "NULL"),
        }
    }
}

/// 数据过滤处理
pub async fn filter(
    source_path: &str,
    output_path: &str,
    options: &HashMap<String, String>
) -> Result<()> {
    let mut config = ProcessorConfig::default();
    
    // 解析选项
    if let Some(batch_size) = options.get("batch_size") {
        if let Ok(size) = batch_size.parse::<usize>() {
            config.batch_size = size;
        }
    }
    
    // 创建处理器实例
    let processor = DataProcessor::new(
        "filter".to_string(),
        "过滤处理器".to_string(),
        DataFormat::JSON,
        config
    );
    
    filter_with_processor(&processor, source_path, output_path, options).await
}

/// 使用处理器进行过滤
pub async fn filter_with_processor(
    _processor: &DataProcessor,
    source_path: &str,
    output_path: &str,
    options: &HashMap<String, String>
) -> Result<()> {
    // 创建输出目录
    tokio::fs::create_dir_all(output_path).await?;
    
    // 读取源文件元数据
    let metadata_file = Path::new(source_path).join("metadata.json");
    let metadata_content = tokio::fs::read_to_string(&metadata_file).await?;
    let metadata: serde_json::Value = serde_json::from_str(&metadata_content)?;
    
    // 解析过滤条件
    let mut column_filters: HashMap<String, Vec<FilterCondition>> = HashMap::new();
    
    for (key, value) in options {
        if key.starts_with("filter_") {
            let column_name = key.strip_prefix("filter_").unwrap();
            let conditions: Result<Vec<FilterCondition>> = value
                .split(';')
                .map(|cond| FilterCondition::parse(cond.trim()))
                .collect();
            
            match conditions {
                Ok(conds) => {
                    column_filters.insert(column_name.to_string(), conds);
                },
                Err(e) => {
                    return Err(Error::invalid_input(&format!("解析过滤条件失败: {}", e)));
                }
            }
        }
    }
    
    if column_filters.is_empty() {
        return Err(Error::invalid_input("没有指定过滤条件"));
    }
    
    // 读取数据文件
    let data_file = Path::new(source_path).join("data.bin");
    let mut file = tokio::fs::File::open(&data_file).await?;
    
    // 读取列数和行数
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf).await?;
    let columns_count = u32::from_le_bytes(buf) as usize;
    
    file.read_exact(&mut buf).await?;
    let rows_count = u32::from_le_bytes(buf) as usize;
    
    // 读取列名
    let mut columns = Vec::new();
    for _ in 0..columns_count {
        file.read_exact(&mut buf).await?;
        let len = u32::from_le_bytes(buf) as usize;
        let mut name_buf = vec![0u8; len];
        file.read_exact(&mut name_buf).await?;
        let column_name = String::from_utf8(name_buf)?;
        columns.push(column_name);
    }
    
    // 找到需要过滤的列的索引
    let mut filter_column_indices: HashMap<usize, Vec<FilterCondition>> = HashMap::new();
    for (col_name, conditions) in &column_filters {
        if let Some(col_idx) = columns.iter().position(|c| c == col_name) {
            filter_column_indices.insert(col_idx, conditions.clone());
        } else {
            return Err(Error::invalid_input(&format!("找不到列: {}", col_name)));
        }
    }
    
    // 读取所有数据并应用过滤条件
    let mut filtered_rows = Vec::new();
    for _ in 0..rows_count {
        let mut row = Vec::new();
        for _ in 0..columns_count {
            file.read_exact(&mut buf).await?;
            let len = u32::from_le_bytes(buf) as usize;
            let mut val_buf = vec![0u8; len];
            file.read_exact(&mut val_buf).await?;
            let value = String::from_utf8(val_buf)?;
            row.push(value);
        }
        
        // 检查是否满足所有过滤条件
        let mut matches_all = true;
        for (col_idx, conditions) in &filter_column_indices {
            let value = &row[*col_idx];
            for condition in conditions {
                if !condition.matches(value)? {
                    matches_all = false;
                    break;
                }
            }
            if !matches_all {
                break;
            }
        }
        
        if matches_all {
            filtered_rows.push(row);
        }
    }
    
    // 创建输出数据文件
    let output_data_file = Path::new(output_path).join("data.bin");
    let mut output_file = tokio::fs::File::create(&output_data_file).await?;
    
    // 保存行数用于元数据
    let filtered_rows_count = filtered_rows.len();
    
    // 写入列数和行数
    output_file.write_all(&(columns_count as u32).to_le_bytes()).await?;
    output_file.write_all(&(filtered_rows_count as u32).to_le_bytes()).await?;
    
    // 写入列名
    for column in &columns {
        let bytes = column.as_bytes();
        output_file.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
        output_file.write_all(bytes).await?;
    }
    
    // 写入过滤后的数据
    for row in filtered_rows {
        for value in row {
            let bytes = value.as_bytes();
            output_file.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
            output_file.write_all(bytes).await?;
        }
    }
    
    // 更新并保存元数据
    let filter_info: HashMap<String, Vec<String>> = column_filters.into_iter()
        .map(|(col, conditions)| {
            let condition_strings: Vec<String> = conditions.into_iter()
                .map(|c| format!("{:?}", c))
                .collect();
            (col, condition_strings)
        })
        .collect();
    
    let filtered_metadata = serde_json::json!({
        "format": metadata["format"],
        "columns": metadata["columns"],
        "rows": filtered_rows_count,
        "schema": metadata["schema"],
        "parent": source_path,
        "processor": "filter",
        "filters": filter_info,
        "original_rows": rows_count
    });
    
    let output_metadata_file = Path::new(output_path).join("metadata.json");
    tokio::fs::write(&output_metadata_file, serde_json::to_string_pretty(&filtered_metadata)?).await?;
    
    Ok(())
}

/// 异步读取文件的文件头
async fn read_file_header_async(path: &Path, num_bytes: usize) -> Result<Vec<u8>> {
    let mut file = tokio::fs::File::open(path).await?;
    let mut buffer = vec![0u8; num_bytes];
    file.read_exact(&mut buffer).await?;
    Ok(buffer)
} 