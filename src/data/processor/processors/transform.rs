// Transform Processor
// 数据转换处理器

use std::collections::HashMap;
use std::path::Path;
use tokio::io::AsyncReadExt;
use crate::Result;
use crate::Error;
use crate::data::processor::core::DataProcessor;
use crate::data::processor::config;
use crate::data::processor::transformers::{TransformerType, NumericTransformer, DateTimeTransformer, CategoricalTransformer};
use crate::data::processor::types::DataType;
use log::{warn, debug};

/// 获取数据类型的转换器
fn get_transformer_for_data_type(data_type: &DataType) -> Result<TransformerType> {
    match data_type {
        DataType::Numeric | DataType::Float => {
            Ok(TransformerType::Numeric(NumericTransformer::new(0.0, 1.0)))
        },
        DataType::DateTime => {
            Ok(TransformerType::DateTime(DateTimeTransformer::new("%Y-%m-%d")))
        },
        DataType::Categorical | DataType::Text => {
            Ok(TransformerType::Categorical(CategoricalTransformer::new(Vec::new())))
        },
        _ => Err(Error::InvalidArgument(format!("不支持的数据类型: {:?}", data_type)))
    }
}

/// 异步读取文件内容
async fn read_file_async(path: &Path) -> Result<Vec<u8>> {
    let mut file = tokio::fs::File::open(path).await?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).await?;
    Ok(buffer)
}

/// 数据转换处理 - 适配器版本
/// 此函数用于支持LegacyProcessorAdapter的调用方式
/// 
/// 应用各种转换器处理数据
pub async fn transform(
    source_path: &str,
    output_path: &str,
    options: &HashMap<String, String>
) -> Result<()> {
    // 创建一个临时DataProcessor对象
    let config = config::ProcessorConfig::default();
    let processor = DataProcessor::new(
        "transform".to_string(),
        "数据转换处理器".to_string(),
        crate::data::DataFormat::JSON,
        config
    );
    
    // 调用完整版transform函数
    transform_with_processor(&processor, source_path, output_path, options).await
}

/// 数据转换处理 - 完整版本
/// 
/// 应用各种转换器处理数据
pub async fn transform_with_processor(
    processor: &DataProcessor,
    source_path: &str,
    output_path: &str,
    options: &HashMap<String, String>
) -> Result<()> {
    // 使用处理器记录转换开始
    debug!("开始数据转换: {} -> {} (处理器: {})", source_path, output_path, processor.name);
    
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
    
    // 确定要转换的列和转换类型
    let transform_spec = options.get("transforms").ok_or_else(|| 
        Error::InvalidInput("没有指定要转换的列和转换类型".into())
    )?;
    
    // 解析转换规格
    // 格式: "column1:numeric,column2:datetime,column3:categorical"
    let mut transforms = HashMap::new();
    for spec in transform_spec.split(',') {
        let parts: Vec<&str> = spec.split(':').collect();
        if parts.len() == 2 {
            let column = parts[0].trim();
            let transform_type = match parts[1].trim().to_lowercase().as_str() {
                "numeric" => TransformerType::Numeric(NumericTransformer::new(0.0, 1.0)),
                "datetime" => TransformerType::DateTime(DateTimeTransformer::new("%Y-%m-%d")),
                "categorical" => TransformerType::Categorical(CategoricalTransformer::new(Vec::new())),
                _ => return Err(Error::InvalidInput(format!("不支持的转换类型: {}", parts[1])))
            };
            transforms.insert(column.to_string(), transform_type);
        } else {
            return Err(Error::InvalidInput(format!("无效的转换规格: {}", spec)));
        }
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
    
    // 为分类转换器收集类别
    for (col_name, transformer) in &mut transforms {
        if let TransformerType::Categorical(cat_transformer) = transformer {
            // 查找列索引
            if let Some(col_idx) = columns.iter().position(|c| c == col_name) {
                // 收集唯一值
                let mut categories = Vec::new();
                for row in &rows {
                    let value = &row[col_idx];
                    if !categories.contains(&value.to_string()) {
                        categories.push(value.to_string());
                    }
                }
                // 更新转换器
                *cat_transformer = CategoricalTransformer::new(categories);
            }
        }
    }
    
    // 转换数据
    let mut transformed_rows = Vec::new();
    let mut transformation_failures = 0;
    
    for (row_idx, row) in rows.iter().enumerate() {
        let mut transformed_row = Vec::new();
        for (col_idx, value) in row.iter().enumerate() {
            let col_name = &columns[col_idx];
            if let Some(transformer) = transforms.get(col_name) {
                // 转换数据
                match transformer.transform(value) {
                    Ok(transformed) => {
                        // 将f32转换为String
                        let transformed_str = transformed.to_string();
                        transformed_row.push(transformed_str);
                        debug!("成功转换行 {}, 列 '{}' 的值", row_idx, col_name);
                    },
                    Err(err) => {
                        // 改进错误处理
                        transformation_failures += 1;
                        warn!("转换失败: 行 {}, 列 '{}', 值 '{}', 错误: {:?}", 
                              row_idx, col_name, value, err);
                        
                        // 根据转换器类型确定合适的默认值
                        let default_value = match transformer {
                            TransformerType::Numeric(_) => "0.0".to_string(),
                            TransformerType::DateTime(_) => "".to_string(),
                            TransformerType::Categorical(_) => "unknown".to_string(),
                        };
                        
                        transformed_row.push(default_value);
                    }
                }
            } else {
                // 保持非转换列不变
                transformed_row.push(value.clone());
            }
        }
        transformed_rows.push(transformed_row);
    }
    
    // 创建输出数据文件
    let output_data_file = Path::new(output_path).join("data.bin");
    let mut output_file = tokio::fs::File::create(&output_data_file).await?;
    
    // 保存行数用于元数据
    let transformed_rows_count = transformed_rows.len();
    
    // 写入列数和行数
    tokio::io::AsyncWriteExt::write_u32(&mut output_file, columns_count as u32).await?;
    tokio::io::AsyncWriteExt::write_u32(&mut output_file, transformed_rows_count as u32).await?;
    
    // 写入列名
    for column in &columns {
        let bytes = column.as_bytes();
        tokio::io::AsyncWriteExt::write_u32(&mut output_file, bytes.len() as u32).await?;
        tokio::io::AsyncWriteExt::write_all(&mut output_file, bytes).await?;
    }
    
    // 写入转换后的数据
    for row in transformed_rows {
        for value in row {
            let bytes = value.as_bytes();
            tokio::io::AsyncWriteExt::write_u32(&mut output_file, bytes.len() as u32).await?;
            tokio::io::AsyncWriteExt::write_all(&mut output_file, bytes).await?;
        }
    }
    
    // 保存转换器信息
    let mut transformers_info = HashMap::new();
    for (col_name, transformer) in &transforms {
        let transformer_type = match transformer {
            TransformerType::Numeric(_) => "numeric",
            TransformerType::DateTime(_) => "datetime",
            TransformerType::Categorical(_) => "categorical",
        };
        transformers_info.insert(col_name.clone(), transformer_type.to_string());
    }
    
    // 更新并保存元数据
    let mut transformed_metadata = serde_json::json!({
        "format": metadata["format"],
        "columns": columns,
        "rows": transformed_rows_count,
        "schema": metadata["schema"],
        "parent": source_path,
        "processor": "transform",
        "transformers": transformers_info
    });
    
    // 添加转换统计信息到元数据
    if transformation_failures > 0 {
        transformed_metadata["transformation_stats"] = serde_json::json!({
            "total_rows": rows.len(),
            "failure_count": transformation_failures,
            "success_rate": (rows.len() - transformation_failures) as f64 / rows.len() as f64
        });
    }
    
    let output_metadata_file = Path::new(output_path).join("metadata.json");
    tokio::fs::write(&output_metadata_file, serde_json::to_string_pretty(&transformed_metadata)?).await?;
    
    Ok(())
} 