// Encode Processor
// 数据编码处理器

use std::collections::HashMap;
use std::path::Path;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use crate::data::processor::processor_impl::DataProcessor;
use crate::data::types::DataFormat;
use crate::data::processor::config::ProcessorConfig;
use crate::error::{Error, Result};

/// 转换器类型
#[derive(Debug, Clone)]
pub enum TransformerType {
    /// 分类编码器
    Category,
    /// 独热编码器
    OneHot,
    /// 词袋编码器
    BagOfWords,
    /// TF-IDF编码器
    TfIdf,
    /// 自定义编码器
    Custom(String),
}

impl Default for TransformerType {
    fn default() -> Self {
        Self::Category
    }
}

impl TransformerType {
    pub fn to_string(&self) -> String {
        match self {
            TransformerType::Category => "category".to_string(),
            TransformerType::OneHot => "onehot".to_string(),
            TransformerType::BagOfWords => "bagofwords".to_string(),
            TransformerType::TfIdf => "tfidf".to_string(),
            TransformerType::Custom(name) => name.clone(),
        }
    }
}

/// 数据编码处理 - 适配器版本
pub async fn encode(
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
        "encoder".to_string(),
        "编码处理器".to_string(),
        DataFormat::JSON,
        config
    );
    
    encode_with_processor(&processor, source_path, output_path, options).await
}

/// 数据编码处理 - 完整版本
pub async fn encode_with_processor(
    _processor: &DataProcessor,
    source_path: &str,
    output_path: &str,
    options: &HashMap<String, String>
) -> Result<()> {
    // 创建输出目录
    tokio::fs::create_dir_all(output_path).await?;
    
    // 读取元数据
    let metadata_file = Path::new(source_path).join("metadata.json");
    let metadata_content = tokio::fs::read_to_string(&metadata_file).await?;
    let metadata: serde_json::Value = serde_json::from_str(&metadata_content)?;
    
    // 获取编码器类型
    let encoder_type = if let Some(e_type) = options.get("encoder_type") {
        match e_type.to_lowercase().as_str() {
            "category" => TransformerType::Category,
            "onehot" => TransformerType::OneHot,
            "bagofwords" => TransformerType::BagOfWords,
            "tfidf" => TransformerType::TfIdf,
            custom => TransformerType::Custom(custom.to_string()),
        }
    } else {
        TransformerType::Category
    };
    
    // 确定要编码的列
    let encode_columns = if let Some(cols) = options.get("columns") {
        cols.split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    
    if encode_columns.is_empty() {
        return Err(Error::invalid_input("没有指定要编码的列"));
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
        let mut col_buf = vec![0u8; len];
        file.read_exact(&mut col_buf).await?;
        let column = String::from_utf8(col_buf)?;
        columns.push(column);
    }
    
    // 读取所有数据
    let mut rows = Vec::new();
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
        rows.push(row);
    }
    
    // 为每个要编码的列建立类别映射
    let mut column_encoders: HashMap<usize, HashMap<String, usize>> = HashMap::new();
    
    for col_idx in 0..columns_count {
        let col_name = &columns[col_idx];
        if encode_columns.contains(col_name) {
            let mut unique_values = std::collections::HashSet::new();
            for row in &rows {
                unique_values.insert(row[col_idx].clone());
            }
            
            let mut value_to_index = HashMap::new();
            for (idx, value) in unique_values.into_iter().enumerate() {
                value_to_index.insert(value, idx);
            }
            column_encoders.insert(col_idx, value_to_index);
        }
    }
    
    // 编码数据
    let mut encoded_rows = Vec::new();
    for row in rows {
        let mut encoded_row = Vec::new();
        for (col_idx, value) in row.iter().enumerate() {
            let col_name = &columns[col_idx];
            if encode_columns.contains(col_name) && column_encoders.contains_key(&col_idx) {
                match &encoder_type {
                    TransformerType::Category => {
                        let encoder = &column_encoders[&col_idx];
                        if let Some(&encoded_val) = encoder.get(value) {
                            encoded_row.push(encoded_val.to_string());
                        } else {
                            encoded_row.push("0".to_string()); // 未知值
                        }
                    },
                    TransformerType::OneHot => {
                        let encoder = &column_encoders[&col_idx];
                        let num_categories = encoder.len();
                        let mut onehot = vec![0; num_categories];
                        if let Some(&encoded_val) = encoder.get(value) {
                            onehot[encoded_val] = 1;
                        }
                        let onehot_str = onehot.iter().map(|&x| x.to_string()).collect::<Vec<_>>().join(",");
                        encoded_row.push(format!("[{}]", onehot_str));
                    },
                    TransformerType::BagOfWords => {
                        // 简化的词袋编码：将文本分割为单词并计数
                        let words: HashMap<String, usize> = value.split_whitespace()
                            .fold(HashMap::new(), |mut acc, word| {
                                *acc.entry(word.to_string()).or_insert(0) += 1;
                                acc
                            });
                        let bow_json = serde_json::to_string(&words)?;
                        encoded_row.push(bow_json);
                    },
                    TransformerType::TfIdf => {
                        // 简化的TF-IDF编码：这里只做基本的词频计算
                        let words: Vec<&str> = value.split_whitespace().collect();
                        let word_count = words.len();
                        let mut tf = HashMap::new();
                        for word in words {
                            *tf.entry(word.to_string()).or_insert(0) += 1;
                        }
                        // 计算TF
                        let tf_normalized: HashMap<String, f64> = tf.into_iter()
                            .map(|(word, count)| (word, count as f64 / word_count as f64))
                            .collect();
                        let tfidf_json = serde_json::to_string(&tf_normalized)?;
                        encoded_row.push(tfidf_json);
                    },
                    TransformerType::Custom(name) => {
                        // 自定义编码器的占位符实现
                        encoded_row.push(format!("{}:{}", name, value));
                    },
                }
            } else {
                // 保持非编码列不变
                encoded_row.push(value.clone());
            }
        }
        encoded_rows.push(encoded_row);
    }
    
    // 创建输出数据文件
    let output_data_file = Path::new(output_path).join("data.bin");
    let mut output_file = tokio::fs::File::create(&output_data_file).await?;
    
    // 保存行数用于元数据
    let encoded_rows_count = encoded_rows.len();
    
    // 写入列数和行数
    output_file.write_all(&(columns_count as u32).to_le_bytes()).await?;
    output_file.write_all(&(encoded_rows_count as u32).to_le_bytes()).await?;
    
    // 写入列名
    for column in &columns {
        let bytes = column.as_bytes();
        output_file.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
        output_file.write_all(bytes).await?;
    }
    
    // 写入编码后的数据
    for row in encoded_rows {
        for value in row {
            let bytes = value.as_bytes();
            output_file.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
            output_file.write_all(bytes).await?;
        }
    }
    
    // 保存编码器信息到元数据
    let mut encoder_info = serde_json::Map::new();
    for (col_idx, encoder) in column_encoders {
        let col_name = &columns[col_idx];
        encoder_info.insert(col_name.clone(), serde_json::json!({
            "type": encoder_type.to_string(),
            "mapping": encoder
        }));
    }
    
    // 更新并保存元数据
    let encoded_metadata = serde_json::json!({
        "format": metadata["format"],
        "columns": metadata["columns"],
        "rows": encoded_rows_count,
        "schema": metadata["schema"],
        "parent": source_path,
        "processor": "encode",
        "encoder_type": encoder_type.to_string(),
        "encoded_columns": encode_columns,
        "encoders": encoder_info
    });
    
    let output_metadata_file = Path::new(output_path).join("metadata.json");
    tokio::fs::write(&output_metadata_file, serde_json::to_string_pretty(&encoded_metadata)?).await?;
    
    Ok(())
} 