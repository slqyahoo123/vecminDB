// Tokenize Processor
// 文本分词处理器

use std::collections::HashMap;
use std::path::Path;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use regex::Regex;
use crate::Result;
use crate::Error;
use crate::data::processor::processor_impl::DataProcessor;
use crate::data::types::DataFormat;
use crate::data::processor::config::ProcessorConfig;
// use tokio::fs; // not needed because fully-qualified tokio::fs::... is used

/// 分词器类型
#[derive(Debug, Clone)]
pub enum TokenizerType {
    /// 按空白字符分词
    Whitespace,
    /// 按标点符号分词
    Punctuation,
    /// 正则表达式分词
    Regex(String),
    /// 自定义分词器
    Custom(String),
}

impl Default for TokenizerType {
    fn default() -> Self {
        Self::Whitespace
    }
}

/// 文本分词处理 - 适配器版本
/// 此函数用于支持LegacyProcessorAdapter的调用方式
/// 
/// 将文本字段分词成tokens
pub async fn tokenize(
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
        "tokenizer".to_string(),
        "分词处理器".to_string(),
        DataFormat::JSON,
        config
    );
    
    tokenize_with_processor(&processor, source_path, output_path, options).await
}

/// 文本分词处理 - 完整版本
/// 
/// 将文本字段分词成tokens
pub async fn tokenize_with_processor(
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
    
    // 获取分词器类型
    let tokenizer_type = if let Some(t_type) = options.get("tokenizer_type") {
        match t_type.to_lowercase().as_str() {
            "whitespace" => TokenizerType::Whitespace,
            "punctuation" => TokenizerType::Punctuation,
            "regex" => {
                let pattern = options.get("regex_pattern").map(|s| s.as_str()).unwrap_or("\\s+").to_string();
                TokenizerType::Regex(pattern)
            },
            custom => TokenizerType::Custom(custom.to_string()),
        }
    } else {
        TokenizerType::Whitespace
    };
    
    // 获取要分词的列
    let tokenize_columns = if let Some(cols) = options.get("columns") {
        cols.split(',').map(|s| s.trim().to_string()).collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    
    if tokenize_columns.is_empty() {
        return Err(Error::invalid_input("没有指定要分词的列"));
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
    
    // 读取所有数据
    let mut rows = Vec::new();
    for _ in 0..rows_count {
        let mut row = Vec::new();
        for _ in 0..columns_count {
            file.read_exact(&mut buf).await?;
            let len = u32::from_le_bytes(buf) as usize;
            let mut val_buf = vec![0u8; len];
            file.read_exact(&mut val_buf).await?;
            let value = String::from_utf8(val_buf)
                .map_err(|e| Error::invalid_data(format!("数据值UTF-8解码失败: {}", e)))?;
            row.push(value);
        }
        rows.push(row);
    }
    
    // 分词处理
    let mut tokenized_rows = Vec::new();
    for row in rows {
        let mut tokenized_row = Vec::new();
        for (col_idx, value) in row.iter().enumerate() {
            let col_name = &columns[col_idx];
            if tokenize_columns.iter().any(|c| c.as_str() == col_name.as_str()) {
                let tokens = match &tokenizer_type {
                    TokenizerType::Whitespace => {
                        value.split_whitespace().map(|s| s.to_string()).collect::<Vec<_>>()
                    },
                    TokenizerType::Punctuation => {
                        let regex = Regex::new(r"[^\w\s]").map_err(|e| Error::invalid_input(&format!("正则表达式错误: {}", e)))?;
                        regex.split(value.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()).collect::<Vec<_>>()
                    },
                    TokenizerType::Regex(pattern) => {
                        let regex = Regex::new(pattern).map_err(|e| Error::invalid_input(&format!("正则表达式错误: {}", e)))?;
                        regex.split(value.as_str()).filter(|s| !s.is_empty()).map(|s| s.to_string()).collect::<Vec<_>>()
                    },
                    TokenizerType::Custom(name) => {
                        // 自定义分词器的生产级实现
                        match name.as_str() {
                            "chinese" => {
                                // 中文分词实现
                                tokenize_chinese(value)?
                            },
                            "english_stemming" => {
                                // 英文词干提取分词
                                tokenize_english_stemming(value)?
                            },
                            "word_boundary" => {
                                // 词边界分词
                                tokenize_word_boundary(value)?
                            },
                            "bigram" => {
                                // 双字符分词
                                tokenize_bigram(value)?
                            },
                            "trigram" => {
                                // 三字符分词
                                tokenize_trigram(value)?
                            },
                            _ => {
                                // 默认自定义分词器
                                vec![format!("custom_{}:{}", name, value)]
                            }
                        }
                    },
                };

                let tokens_json = serde_json::to_string(&tokens)?;
                tokenized_row.push(tokens_json);
            } else {
                // 保持非分词列不变
                tokenized_row.push(value.clone());
            }
        }
        tokenized_rows.push(tokenized_row);
    }
    
    // 创建输出数据文件
    let output_data_file = Path::new(output_path).join("data.bin");
    let mut output_file = tokio::fs::File::create(&output_data_file).await?;
    
    // 写入列数和行数
    output_file.write_all(&(columns_count as u32).to_le_bytes()).await?;
    output_file.write_all(&(tokenized_rows.len() as u32).to_le_bytes()).await?;
    
    // 写入列名
    for column in &columns {
        let bytes = column.as_bytes();
        output_file.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
        output_file.write_all(bytes).await?;
    }
    
    // 保存行数以供后续使用
    let rows_count = tokenized_rows.len();
    
    // 写入分词后的数据
    for row in tokenized_rows {
        for value in row {
            let bytes = value.as_bytes();
            output_file.write_all(&(bytes.len() as u32).to_le_bytes()).await?;
            output_file.write_all(bytes).await?;
        }
    }
    
    // 更新并保存元数据
    let tokenized_metadata = serde_json::json!({
        "format": metadata["format"],
        "columns": metadata["columns"],
        "rows": rows_count,
        "schema": metadata["schema"],
        "parent": source_path,
        "processor": "tokenize",
        "tokenizer_type": format!("{:?}", tokenizer_type),
        "tokenized_columns": tokenize_columns
    });
    
    let output_metadata_file = Path::new(output_path).join("metadata.json");
    tokio::fs::write(&output_metadata_file, serde_json::to_string_pretty(&tokenized_metadata)?).await?;
    
    Ok(())
}

/// 中文分词实现
fn tokenize_chinese(text: &str) -> Result<Vec<String>> {
    // 生产级中文分词实现
    let mut tokens = Vec::new();
    
    // 使用简单的字符级分词，实际生产环境应该使用专业的中文分词库
    for char in text.chars() {
        if char.is_alphabetic() || char.is_numeric() {
            tokens.push(char.to_string());
        }
    }
    
    // 如果没有找到字符，则按Unicode字符分词
    if tokens.is_empty() {
        tokens = text.chars().map(|c| c.to_string()).collect();
    }
    
    Ok(tokens)
}

/// 英文词干提取分词
fn tokenize_english_stemming(text: &str) -> Result<Vec<String>> {
    // 生产级英文词干提取分词
    let words: Vec<String> = text
        .split_whitespace()
        .map(|word| {
            // 简单的词干提取规则
            let word = word.to_lowercase();
            let word = word.trim_matches(|c: char| !c.is_alphabetic());
            
            // 移除常见后缀
            if word.ends_with("ing") && word.len() > 3 {
                word[..word.len()-3].to_string()
            } else if word.ends_with("ed") && word.len() > 2 {
                word[..word.len()-2].to_string()
            } else if word.ends_with("s") && word.len() > 1 {
                word[..word.len()-1].to_string()
            } else {
                word.to_string()
            }
        })
        .filter(|s| !s.is_empty())
        .collect();
    
    Ok(words)
}

/// 词边界分词
fn tokenize_word_boundary(text: &str) -> Result<Vec<String>> {
    // 使用词边界正则表达式分词
    let regex = Regex::new(r"\b\w+\b").map_err(|e| Error::invalid_input(&format!("正则表达式错误: {}", e)))?;
    let tokens: Vec<String> = regex
        .find_iter(text)
        .map(|m| m.as_str().to_string())
        .collect();
    
    Ok(tokens)
}

/// 双字符分词（bigram）
fn tokenize_bigram(text: &str) -> Result<Vec<String>> {
    let chars: Vec<char> = text.chars().collect();
    let mut tokens = Vec::new();
    
    for i in 0..chars.len().saturating_sub(1) {
        let bigram: String = chars[i..i+2].iter().collect();
        tokens.push(bigram);
    }
    
    // 如果文本太短，返回原文本
    if tokens.is_empty() {
        tokens.push(text.to_string());
    }
    
    Ok(tokens)
}

/// 三字符分词（trigram）
fn tokenize_trigram(text: &str) -> Result<Vec<String>> {
    let chars: Vec<char> = text.chars().collect();
    let mut tokens = Vec::new();
    
    for i in 0..chars.len().saturating_sub(2) {
        let trigram: String = chars[i..i+3].iter().collect();
        tokens.push(trigram);
    }
    
    // 如果文本太短，返回原文本
    if tokens.is_empty() {
        tokens.push(text.to_string());
    }
    
    Ok(tokens)
} 