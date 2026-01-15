// Normalize Processor
// 数据标准化处理器

use std::collections::HashMap;
use std::path::Path;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use log::{info, warn};
use crate::Result;
use crate::Error;
use crate::data::processor::processor_impl::DataProcessor;
// 导入新的模块化结构
// remove unused imports from earlier refactor
use crate::data::types::DataFormat;
use crate::data::processor::config::ProcessorConfig;
// use tokio::fs; // not needed; use fully-qualified paths

/// 标准化方法
#[derive(Debug, Clone)]
pub enum NormalizationMethod {
    /// Z-Score标准化
    ZScore,
    /// Min-Max标准化
    MinMax,
    /// 单位向量标准化
    UnitVector,
    /// 百分位标准化
    Percentile,
}

impl Default for NormalizationMethod {
    fn default() -> Self {
        Self::ZScore
    }
}

/// 标准化数据处理 - 适配器版本
/// 此函数用于支持LegacyProcessorAdapter的调用方式
/// 
/// 将数值型数据标准化为均值为0、标准差为1的分布
pub async fn normalize(
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
        "normalizer".to_string(),
        "标准化处理器".to_string(),
        DataFormat::JSON,
        config
    );
    
    normalize_with_processor(&processor, source_path, output_path, options).await
}

/// 标准化数据处理 - 完整版本
/// 
/// 将数值型数据标准化为均值为0、标准差为1的分布
pub async fn normalize_with_processor(
    processor: &DataProcessor,
    source_path: &str,
    output_path: &str,
    options: &HashMap<String, String>
) -> Result<()> {
    // 使用处理器的配置进行日志记录
    info!("开始标准化处理: {} -> {} (处理器: {})", source_path, output_path, processor.name);
    
    // 创建输出目录
    tokio::fs::create_dir_all(output_path).await?;
    
    // 读取源文件元数据
    let metadata_file = Path::new(source_path).join("metadata.json");
    let metadata_content = tokio::fs::read_to_string(&metadata_file).await?;
    let metadata: serde_json::Value = serde_json::from_str(&metadata_content)?;
    
    // 获取标准化方法
    let normalization_method = if let Some(method) = options.get("method") {
        match method.to_lowercase().as_str() {
            "zscore" => NormalizationMethod::ZScore,
            "minmax" => NormalizationMethod::MinMax,
            "unitvector" => NormalizationMethod::UnitVector,
            "percentile" => NormalizationMethod::Percentile,
            _ => NormalizationMethod::ZScore,
        }
    } else {
        NormalizationMethod::ZScore
    };
    
    // 获取要标准化的列
    let normalize_columns = if let Some(cols) = options.get("columns") {
        cols.split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    
    if normalize_columns.is_empty() {
        return Err(Error::invalid_input("没有指定要标准化的列"));
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
        let column_name = String::from_utf8(name_buf)
            .map_err(|e| Error::invalid_data(format!("列名UTF-8解码失败: {}", e)))?;
        columns.push(column_name);
    }
    
    // 获取schema信息（从metadata中获取而非直接访问处理器）
    let schema_info = metadata.get("schema")
        .ok_or_else(|| Error::invalid_data("缺少schema信息".to_string()))?;
    
    // 验证schema与要标准化的列的兼容性
    if let Some(schema_columns) = schema_info.get("columns") {
        if let Some(columns_array) = schema_columns.as_array() {
            for normalize_col in &normalize_columns {
                let found = columns_array.iter().any(|col| {
                    col.get("name").and_then(|n| n.as_str()) == Some(normalize_col)
                });
                if !found {
                    warn!("标准化列 '{}' 在schema中未找到", normalize_col);
                }
            }
        }
    }
    
    // 读取所有数据并收集数值列的统计信息
    let mut rows = Vec::new();
    let mut column_stats: HashMap<usize, (f64, f64, Vec<f64>)> = HashMap::new(); // (sum, sum_squared, values)
    
    for _ in 0..rows_count {
        let mut row = Vec::new();
        for col_idx in 0..columns_count {
            file.read_exact(&mut buf).await?;
            let len = u32::from_le_bytes(buf) as usize;
            let mut val_buf = vec![0u8; len];
            file.read_exact(&mut val_buf).await?;
            let value = String::from_utf8(val_buf)
                .map_err(|e| Error::invalid_data(format!("数据值UTF-8解码失败: {}", e)))?;
            
            let col_name = &columns[col_idx];
            if normalize_columns.contains(col_name) {
                // 尝试解析为数值
                if let Ok(num_val) = value.parse::<f64>() {
                    let entry = column_stats.entry(col_idx).or_insert((0.0, 0.0, Vec::new()));
                    entry.0 += num_val; // sum
                    entry.1 += num_val * num_val; // sum of squares
                    entry.2.push(num_val); // values
                }
            }
            
            row.push(value);
        }
        rows.push(row);
    }
    
    // 计算标准化参数
    let mut normalization_params: HashMap<usize, (f64, f64)> = HashMap::new(); // (param1, param2)
    
    for (col_idx, (sum, sum_squared, values)) in &column_stats {
        let n = values.len() as f64;
        if n > 0.0 {
            match normalization_method {
                NormalizationMethod::ZScore => {
                    let mean = sum / n;
                    let variance = (sum_squared / n) - (mean * mean);
                    let std_dev = variance.sqrt();
                    normalization_params.insert(*col_idx, (mean, std_dev));
                },
                NormalizationMethod::MinMax => {
                    let min_val = values.iter().copied().fold(f64::INFINITY, f64::min);
                    let max_val = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    normalization_params.insert(*col_idx, (min_val, max_val));
                },
                NormalizationMethod::UnitVector => {
                    let magnitude = values.iter().map(|v| v * v).sum::<f64>().sqrt();
                    normalization_params.insert(*col_idx, (magnitude, 1.0));
                },
                NormalizationMethod::Percentile => {
                    let mut sorted_values = values.clone();
                    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let p25 = sorted_values[(0.25 * n) as usize];
                    let p75 = sorted_values[(0.75 * n) as usize];
                    normalization_params.insert(*col_idx, (p25, p75));
                },
            }
        }
    }
    
    // 应用标准化
    let mut normalized_rows = Vec::new();
    for row in rows {
        let mut normalized_row = Vec::new();
        for (col_idx, value) in row.iter().enumerate() {
            let col_name = &columns[col_idx];
            if normalize_columns.contains(col_name) && normalization_params.contains_key(&col_idx) {
                if let Ok(num_val) = value.parse::<f64>() {
                    let (param1, param2) = normalization_params[&col_idx];
                    let normalized_val = match normalization_method {
                        NormalizationMethod::ZScore => {
                            if param2 != 0.0 { (num_val - param1) / param2 } else { 0.0 }
                        },
                        NormalizationMethod::MinMax => {
                            if param2 != param1 { (num_val - param1) / (param2 - param1) } else { 0.0 }
                        },
                        NormalizationMethod::UnitVector => {
                            if param1 != 0.0 { num_val / param1 } else { 0.0 }
                        },
                        NormalizationMethod::Percentile => {
                            if param2 != param1 { (num_val - param1) / (param2 - param1) } else { 0.0 }
                        },
                    };
                    normalized_row.push(normalized_val.to_string());
                } else {
                    // 非数值保持不变
                    normalized_row.push(value.clone());
                }
            } else {
                // 非标准化列保持不变
                normalized_row.push(value.clone());
            }
        }
        normalized_rows.push(normalized_row);
    }
    
    // 创建输出数据文件
    let output_data_file = Path::new(output_path).join("data.bin");
    let mut output_file = tokio::fs::File::create(&output_data_file).await?;
    
    // 保存行数用于元数据
    let normalized_rows_count = normalized_rows.len();
    
    // 写入列数和行数
    output_file.write_all(&(columns_count as u32).to_le_bytes()).await?;
    output_file.write_all(&(normalized_rows_count as u32).to_le_bytes()).await?;
    
    // 写入列名
    for column in &columns {
        let bytes = column.as_bytes();
        output_file.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
        output_file.write_all(bytes).await?;
    }
    
    // 写入标准化后的数据
    for row in normalized_rows {
        for value in row {
            let bytes = value.as_bytes();
            output_file.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
            output_file.write_all(bytes).await?;
        }
    }
    
    // 保存标准化参数到元数据
    let mut params_info = serde_json::Map::new();
    for (col_idx, (param1, param2)) in normalization_params {
        let col_name = &columns[col_idx];
        params_info.insert(col_name.clone(), serde_json::json!({
            "method": format!("{:?}", normalization_method),
            "param1": param1,
            "param2": param2
        }));
    }
    
    // 更新并保存元数据
    let normalized_metadata = serde_json::json!({
        "format": metadata["format"],
        "columns": metadata["columns"],
        "rows": normalized_rows_count,
        "schema": metadata["schema"],
        "parent": source_path,
        "processor": "normalize",
        "normalization_method": format!("{:?}", normalization_method),
        "normalized_columns": normalize_columns,
        "normalization_params": params_info
    });
    
    let output_metadata_file = Path::new(output_path).join("metadata.json");
    tokio::fs::write(&output_metadata_file, serde_json::to_string_pretty(&normalized_metadata)?).await?;
    
    Ok(())
} 