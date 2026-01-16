// 处理器核心库 - 共享核心功能，用于两个处理器系统的融合
// Processor Core Library - Shared core functionality for processor systems integration

use crate::data::value::DataValue;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// 文本处理工具函数
pub mod text {
    use crate::error::{Error, Result};
    use regex::Regex;
    
    /// 文本转换（大小写，去空格等）
    pub fn transform_text(text: &str, transform_type: &str) -> String {
        match transform_type {
            "lowercase" => text.to_lowercase(),
            "uppercase" => text.to_uppercase(),
            "trim" => text.trim().to_string(),
            "trim_start" => text.trim_start().to_string(),
            "trim_end" => text.trim_end().to_string(),
            "remove_whitespace" => text.chars().filter(|c| !c.is_whitespace()).collect(),
            _ => text.to_string(),
        }
    }
    
    /// 文本替换（支持正则表达式）
    pub fn replace_text(text: &str, pattern: &str, replacement: &str, use_regex: bool) -> Result<String> {
        if use_regex {
            match Regex::new(pattern) {
                Ok(re) => Ok(re.replace_all(text, replacement).to_string()),
                Err(e) => Err(Error::invalid_argument(format!("无效的正则表达式: {}", e))),
            }
        } else {
            Ok(text.replace(pattern, replacement))
        }
    }
    
    /// 文本分割
    pub fn split_text(text: &str, separator: &str, max_splits: Option<usize>) -> Vec<String> {
        if let Some(max) = max_splits {
            text.splitn(max + 1, separator).map(String::from).collect()
        } else {
            text.split(separator).map(String::from).collect()
        }
    }
    
    /// 文本拼接
    pub fn join_text(parts: &[String], delimiter: &str) -> String {
        parts.join(delimiter)
    }
}

/// 处理器统计信息
pub struct ProcessorStats {
    pub processor_name: String,
    pub processed_count: usize,
    pub processing_time: Duration,
    pub created_at: Instant,
    pub last_used: Instant,
}

impl ProcessorStats {
    /// 创建新的处理器统计
    pub fn new(name: &str) -> Self {
        Self {
            processor_name: name.to_string(),
            processed_count: 0,
            processing_time: Duration::from_secs(0),
            created_at: Instant::now(),
            last_used: Instant::now(),
        }
    }
    
    /// 更新统计信息
    pub fn update(&mut self, duration: Duration) {
        self.processed_count += 1;
        self.processing_time += duration;
        self.last_used = Instant::now();
    }
    
    /// 计算平均处理时间
    pub fn average_processing_time(&self) -> Duration {
        if self.processed_count == 0 {
            Duration::from_secs(0)
        } else {
            self.processing_time / self.processed_count as u32
        }
    }
    
    /// 重置统计信息
    pub fn reset(&mut self) {
        self.processed_count = 0;
        self.processing_time = Duration::from_secs(0);
        self.last_used = Instant::now();
    }
}

/// 共享处理器上下文数据结构
/// Shared processor context data structure
pub struct SharedProcessorContext {
    pub data: HashMap<String, DataValue>,
    pub metadata: HashMap<String, String>,
}

impl SharedProcessorContext {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            metadata: HashMap::new(),
        }
    }
    
    pub fn from_data(data: HashMap<String, DataValue>) -> Self {
        Self {
            data,
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }
}

/// 共享处理器结果数据结构
/// Shared processor result data structure
pub struct SharedProcessorResult {
    pub data: HashMap<String, DataValue>,
    pub metadata: HashMap<String, String>,
}

impl SharedProcessorResult {
    pub fn new(data: HashMap<String, DataValue>) -> Self {
        Self {
            data,
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }
} 