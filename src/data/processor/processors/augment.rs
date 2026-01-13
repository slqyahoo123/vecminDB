// Augment Processor
// 数据增强处理器

use std::collections::HashMap;
use std::path::Path;
use tokio::io::AsyncReadExt;
use rand::{Rng, thread_rng};
use crate::Result;
use crate::Error;
use crate::data::processor::core::DataProcessor;
use crate::data::processor::config;

/// 数据增强处理 - 适配器版本
/// 此函数用于支持LegacyProcessorAdapter的调用方式
///
/// 通过简单变换增加数据量
pub async fn augment(
    source_path: &str,
    output_path: &str,
    options: &HashMap<String, String>
) -> Result<()> {
    // 创建一个临时DataProcessor对象
    let config = config::ProcessorConfig::default();
    let processor = DataProcessor::new(
        "augment".to_string(),
        "数据增强处理器".to_string(),
        crate::data::DataFormat::JSON,
        config
    );
    
    // 调用完整版augment函数
    augment_with_processor(&processor, source_path, output_path, options).await
}

/// 数据增强处理 - 完整版本
/// 
/// 通过简单变换增加数据量
pub async fn augment_with_processor(
    processor: &DataProcessor,
    source_path: &str,
    output_path: &str,
    options: &HashMap<String, String>
) -> Result<()> {
    let _processor = processor;
    
    // 读取元数据
    let metadata_file = Path::new(source_path).join("metadata.json");
    let metadata_str = tokio::fs::read_to_string(&metadata_file).await?;
    let metadata: serde_json::Value = serde_json::from_str(&metadata_str)?;
    
    // 读取数据文件
    let data_file = Path::new(source_path).join("data.bin");
    let mut file = tokio::fs::File::open(&data_file).await?;
    
    // 读取列数和行数
    let columns_count = tokio::io::AsyncReadExt::read_u32(&mut file).await? as usize;
    let rows_count = tokio::io::AsyncReadExt::read_u32(&mut file).await? as usize;
    
    // 读取列名
    let mut columns = Vec::new();
    for _ in 0..columns_count {
        let len = tokio::io::AsyncReadExt::read_u32(&mut file).await? as usize;
        let mut buf = vec![0u8; len];
        tokio::io::AsyncReadExt::read_exact(&mut file, &mut buf).await?;
        let column = String::from_utf8(buf)?;
        columns.push(column);
    }
    
    // 确定增强方法和要增强的列
    let augment_type = options.get("type").unwrap_or(&"noise".to_string()).to_lowercase();
    let augment_factor = options.get("factor").map_or(2.0, |f| f.parse::<f64>().unwrap_or(2.0));
    let columns_to_augment = if let Some(cols) = options.get("columns") {
        cols.split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>()
    } else {
        // 默认不增强任何列，需要明确指定
        Vec::new()
    };
    
    if columns_to_augment.is_empty() {
        return Err(Error::InvalidInput("没有指定要增强的列".into()));
    }
    
    // 读取所有数据
    let mut rows = Vec::new();
    for _ in 0..rows_count {
        let mut row = Vec::new();
        for _ in 0..columns_count {
            let len = tokio::io::AsyncReadExt::read_u32(&mut file).await? as usize;
            let mut buf = vec![0u8; len];
            tokio::io::AsyncReadExt::read_exact(&mut file, &mut buf).await?;
            let value = String::from_utf8(buf)?;
            row.push(value);
        }
        rows.push(row);
    }
    
    // 应用增强
    let mut augmented_rows = Vec::new();
    
    // 首先添加原始数据
    augmented_rows.extend(rows.clone());
    
    // 随机数生成器
    let mut rng = thread_rng();
    
    // 增强次数 = 原始数据量 * 增强因子
    let augment_count = (rows.len() as f64 * (augment_factor - 1.0)).round() as usize;
    
    for _ in 0..augment_count {
        // 随机选择一行作为基础
        let base_idx = rng.gen_range(0..rows.len());
        let base_row = &rows[base_idx];
        
        // 创建新行
        let mut new_row = base_row.clone();
        
        // 应用增强
        for col_name in &columns_to_augment {
            if let Some(col_idx) = columns.iter().position(|c| c == col_name) {
                match augment_type.as_str() {
                    "noise" => {
                        // 对数值列添加噪声
                        if let Ok(mut value) = new_row[col_idx].parse::<f64>() {
                            // 添加小范围噪声 (±10%)
                            let noise = (rng.gen::<f64>() - 0.5) * 0.2 * value.abs();
                            value += noise;
                            new_row[col_idx] = value.to_string();
                        }
                    },
                    "shuffle" => {
                        // 随机交换两列的值
                        if columns_to_augment.len() >= 2 {
                            // 在增强列中找到不同于当前列的其他列
                            let available_cols: Vec<_> = columns_to_augment.iter()
                                .filter(|c| *c != col_name)
                                .collect();
                                
                            if !available_cols.is_empty() {
                                let other_col_name = available_cols[rng.gen_range(0..available_cols.len())];
                                if let Some(other_idx) = columns.iter().position(|c| c == other_col_name) {
                                    // 交换值
                                    new_row.swap(col_idx, other_idx);
                                }
                            }
                        }
                    },
                    "duplicate" => {
                        // 直接使用原始行，不做修改
                    },
                    "remove" => {
                        // 将值设为空
                        new_row[col_idx] = "".to_string();
                    },
                    _ => {
                        return Err(Error::InvalidInput(format!("不支持的增强类型: {}", augment_type)));
                    }
                }
            }
        }
        
        // 添加到增强数据集
        augmented_rows.push(new_row);
    }
    
    // 创建输出数据文件
    let output_data_file = Path::new(output_path).join("data.bin");
    let mut output_file = tokio::fs::File::create(&output_data_file).await?;
    
    // 写入列数和行数
    tokio::io::AsyncWriteExt::write_u32(&mut output_file, columns_count as u32).await?;
    tokio::io::AsyncWriteExt::write_u32(&mut output_file, augmented_rows.len() as u32).await?;
    
    // 写入列名
    for column in &columns {
        let bytes = column.as_bytes();
        tokio::io::AsyncWriteExt::write_u32(&mut output_file, bytes.len() as u32).await?;
        tokio::io::AsyncWriteExt::write_all(&mut output_file, bytes).await?;
    }
    
    // 写入增强后的数据
    for row in &augmented_rows {
        for value in row {
            let bytes = value.as_bytes();
            tokio::io::AsyncWriteExt::write_u32(&mut output_file, bytes.len() as u32).await?;
            tokio::io::AsyncWriteExt::write_all(&mut output_file, bytes).await?;
        }
    }
    
    // 更新并保存元数据
    let augmented_metadata = serde_json::json!({
        "format": metadata["format"],
        "columns": columns,
        "rows": augmented_rows.len(),
        "schema": metadata["schema"],
        "parent": source_path,
        "processor": "augment",
        "augment_type": augment_type,
        "augment_factor": augment_factor,
        "augmented_columns": columns_to_augment
    });
    
    let output_metadata_file = Path::new(output_path).join("metadata.json");
    tokio::fs::write(&output_metadata_file, serde_json::to_string_pretty(&augmented_metadata)?).await?;
    
    Ok(())
}

/// 异步读取二进制数据长度
async fn read_binary_data_size_async(path: &Path) -> Result<(usize, usize)> {
    let mut file = tokio::fs::File::open(path).await?;
    
    // 读取列数和行数
    let columns_count = file.read_u32().await? as usize;
    let rows_count = file.read_u32().await? as usize;
    
    Ok((columns_count, rows_count))
} 