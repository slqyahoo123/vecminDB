use std::collections::HashMap;
use regex::Regex;
use serde::{Deserialize, Serialize};
use crate::data::pipeline::{ProcessorContext, ProcessorFactory, ProcessorResult, Processor, ProcessorStats};
use crate::error::{Error, Result};
use crate::data::DataValue;

/// 文本转换处理器
/// 
/// 支持将文本转换为大写、小写、去空格、大写首字母等操作
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextTransformProcessor {
    /// 字段名
    field: String,
    /// 转换类型: upper, lower, trim, capitalize
    transform_type: String,
    /// 是否创建新字段
    create_new_field: bool,
    /// 新字段名（如果create_new_field为true）
    new_field_name: Option<String>,
    /// 处理器统计信息
    stats: ProcessorStats,
}

impl TextTransformProcessor {
    pub fn new(field: String, transform_type: String, create_new_field: bool, new_field_name: Option<String>) -> Self {
        Self {
            field,
            transform_type,
            create_new_field,
            new_field_name,
            stats: ProcessorStats::new("text_transform"),
        }
    }
    
    /// 执行文本转换操作
    fn transform_text(&self, text: &str) -> String {
        match self.transform_type.as_str() {
            "upper" => text.to_uppercase(),
            "lower" => text.to_lowercase(),
            "trim" => text.trim().to_string(),
            "capitalize" => {
                let mut c = text.chars();
                match c.next() {
                    None => String::new(),
                    Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                }
            },
            _ => text.to_string(),
        }
    }
}

impl Processor for TextTransformProcessor {
    fn process(&mut self, mut context: ProcessorContext) -> Result<ProcessorResult> {
        let start = std::time::Instant::now();
        
        // 获取需处理的字段值
        let value = match context.data.get(&self.field) {
            Some(DataValue::String(s)) => s.clone(),
            Some(_) => return Err(Error::InvalidDataType(format!("字段 {} 不是字符串类型", self.field))),
            None => return Err(Error::FieldNotFound(self.field.clone())),
        };
        
        // 执行转换
        let transformed = self.transform_text(&value);
        
        // 更新数据
        if self.create_new_field {
            let new_field = self.new_field_name.clone().unwrap_or_else(|| format!("{}_transformed", self.field));
            context.data.insert(new_field, DataValue::String(transformed));
        } else {
            context.data.insert(self.field.clone(), DataValue::String(transformed));
        }
        
        // 更新统计信息
        self.stats.processed_count += 1;
        self.stats.processing_time += start.elapsed();
        
        Ok(ProcessorResult {
            data: context.data,
            metadata: context.metadata,
        })
    }
    
    fn get_stats(&self) -> &ProcessorStats {
        &self.stats
    }
    
    fn reset_stats(&mut self) {
        self.stats = ProcessorStats::new("text_transform");
    }
}

/// 文本替换处理器
/// 
/// 支持普通替换和正则表达式替换
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextReplaceProcessor {
    /// 字段名
    field: String,
    /// 要匹配的模式
    pattern: String,
    /// 替换为的内容
    replacement: String,
    /// 是否使用正则表达式
    use_regex: bool,
    /// 是否创建新字段
    create_new_field: bool,
    /// 新字段名（如果create_new_field为true）
    new_field_name: Option<String>,
    /// 处理器统计信息
    stats: ProcessorStats,
    /// 编译后的正则表达式（不序列化）
    #[serde(skip)]
    compiled_regex: Option<Regex>,
}

impl TextReplaceProcessor {
    pub fn new(
        field: String, 
        pattern: String, 
        replacement: String, 
        use_regex: bool,
        create_new_field: bool, 
        new_field_name: Option<String>
    ) -> Result<Self> {
        let compiled_regex = if use_regex {
            Some(Regex::new(&pattern).map_err(|e| Error::InvalidRegex(e.to_string()))?)
        } else {
            None
        };
        
        Ok(Self {
            field,
            pattern,
            replacement,
            use_regex,
            create_new_field,
            new_field_name,
            stats: ProcessorStats::new("text_replace"),
            compiled_regex,
        })
    }
    
    /// 执行文本替换操作
    fn replace_text(&mut self, text: &str) -> Result<String> {
        if self.use_regex {
            if self.compiled_regex.is_none() {
                self.compiled_regex = Some(Regex::new(&self.pattern).map_err(|e| Error::InvalidRegex(e.to_string()))?);
            }
            
            let regex = self.compiled_regex.as_ref().unwrap();
            Ok(regex.replace_all(text, self.replacement.as_str()).to_string())
        } else {
            Ok(text.replace(&self.pattern, &self.replacement))
        }
    }
}

impl Processor for TextReplaceProcessor {
    fn process(&mut self, mut context: ProcessorContext) -> Result<ProcessorResult> {
        let start = std::time::Instant::now();
        
        // 获取需处理的字段值
        let value = match context.data.get(&self.field) {
            Some(DataValue::String(s)) => s.clone(),
            Some(_) => return Err(Error::InvalidDataType(format!("字段 {} 不是字符串类型", self.field))),
            None => return Err(Error::FieldNotFound(self.field.clone())),
        };
        
        // 执行替换
        let replaced = self.replace_text(&value)?;
        
        // 更新数据
        if self.create_new_field {
            let new_field = self.new_field_name.clone().unwrap_or_else(|| format!("{}_replaced", self.field));
            context.data.insert(new_field, DataValue::String(replaced));
        } else {
            context.data.insert(self.field.clone(), DataValue::String(replaced));
        }
        
        // 更新统计信息
        self.stats.processed_count += 1;
        self.stats.processing_time += start.elapsed();
        
        Ok(ProcessorResult {
            data: context.data,
            metadata: context.metadata,
        })
    }
    
    fn get_stats(&self) -> &ProcessorStats {
        &self.stats
    }
    
    fn reset_stats(&mut self) {
        self.stats = ProcessorStats::new("text_replace");
    }
}

/// 文本分割处理器
/// 
/// 将文本按分隔符分割成数组
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSplitProcessor {
    /// 字段名
    field: String,
    /// 分隔符
    separator: String,
    /// 最大分割数量（可选）
    max_splits: Option<usize>,
    /// 是否保留原始字段
    keep_original: bool,
    /// 目标字段名（如不保留原始字段，则覆盖原字段）
    target_field: Option<String>,
    /// 处理器统计信息
    stats: ProcessorStats,
}

impl TextSplitProcessor {
    pub fn new(
        field: String, 
        separator: String, 
        max_splits: Option<usize>,
        keep_original: bool,
        target_field: Option<String>
    ) -> Self {
        Self {
            field,
            separator,
            max_splits,
            keep_original,
            target_field,
            stats: ProcessorStats::new("text_split"),
        }
    }
}

impl Processor for TextSplitProcessor {
    fn process(&mut self, mut context: ProcessorContext) -> Result<ProcessorResult> {
        let start = std::time::Instant::now();
        
        // 获取需处理的字段值
        let value = match context.data.get(&self.field) {
            Some(DataValue::String(s)) => s.clone(),
            Some(_) => return Err(Error::InvalidDataType(format!("字段 {} 不是字符串类型", self.field))),
            None => return Err(Error::FieldNotFound(self.field.clone())),
        };
        
        // 执行分割
        let parts: Vec<String> = if let Some(limit) = self.max_splits {
            value.splitn(limit + 1, &self.separator).map(|s| s.to_string()).collect()
        } else {
            value.split(&self.separator).map(|s| s.to_string()).collect()
        };
        
        let array_value = DataValue::Array(parts.into_iter().map(DataValue::String).collect());
        
        // 更新数据
        let target = self.target_field.clone().unwrap_or_else(|| {
            if self.keep_original {
                format!("{}_split", self.field)
            } else {
                self.field.clone()
            }
        });
        
        context.data.insert(target, array_value);
        
        // 更新统计信息
        self.stats.processed_count += 1;
        self.stats.processing_time += start.elapsed();
        
        Ok(ProcessorResult {
            data: context.data,
            metadata: context.metadata,
        })
    }
    
    fn get_stats(&self) -> &ProcessorStats {
        &self.stats
    }
    
    fn reset_stats(&mut self) {
        self.stats = ProcessorStats::new("text_split");
    }
}

/// 文本连接处理器
/// 
/// 将数组连接成文本
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextJoinProcessor {
    /// 字段名（数组字段）
    field: String,
    /// 连接符
    delimiter: String,
    /// 是否保留原始字段
    keep_original: bool,
    /// 目标字段名（如不保留原始字段，则覆盖原字段）
    target_field: Option<String>,
    /// 处理器统计信息
    stats: ProcessorStats,
}

impl TextJoinProcessor {
    pub fn new(
        field: String, 
        delimiter: String, 
        keep_original: bool,
        target_field: Option<String>
    ) -> Self {
        Self {
            field,
            delimiter,
            keep_original,
            target_field,
            stats: ProcessorStats::new("text_join"),
        }
    }
}

impl Processor for TextJoinProcessor {
    fn process(&mut self, mut context: ProcessorContext) -> Result<ProcessorResult> {
        let start = std::time::Instant::now();
        
        // 获取需处理的字段值
        let array = match context.data.get(&self.field) {
            Some(DataValue::Array(arr)) => arr.clone(),
            Some(_) => return Err(Error::InvalidDataType(format!("字段 {} 不是数组类型", self.field))),
            None => return Err(Error::FieldNotFound(self.field.clone())),
        };
        
        // 将数组元素转换为字符串
        let string_values: Vec<String> = array.iter().map(|v| v.to_string()).collect();
        
        // 执行连接
        let joined = string_values.join(&self.delimiter);
        
        // 更新数据
        let target = self.target_field.clone().unwrap_or_else(|| {
            if self.keep_original {
                format!("{}_joined", self.field)
            } else {
                self.field.clone()
            }
        });
        
        context.data.insert(target, DataValue::String(joined));
        
        // 更新统计信息
        self.stats.processed_count += 1;
        self.stats.processing_time += start.elapsed();
        
        Ok(ProcessorResult {
            data: context.data,
            metadata: context.metadata,
        })
    }
    
    fn get_stats(&self) -> &ProcessorStats {
        &self.stats
    }
    
    fn reset_stats(&mut self) {
        self.stats = ProcessorStats::new("text_join");
    }
}

// 工厂实现

/// 文本转换处理器工厂
#[derive(Debug, Clone)]
pub struct TextTransformProcessorFactory;

impl ProcessorFactory for TextTransformProcessorFactory {
    fn create(&self, params: HashMap<String, String>) -> Result<Box<dyn Processor>> {
        let field = params.get("field").ok_or_else(|| Error::MissingParameter("field".to_string()))?.clone();
        let transform_type = params.get("transform_type").ok_or_else(|| Error::MissingParameter("transform_type".to_string()))?.clone();
        
        let create_new_field = params.get("create_new_field")
            .map(|v| v.to_lowercase() == "true")
            .unwrap_or(false);
        
        let new_field_name = params.get("new_field_name").cloned();
        
        Ok(Box::new(TextTransformProcessor::new(
            field,
            transform_type,
            create_new_field,
            new_field_name,
        )))
    }
    
    fn get_name(&self) -> &'static str {
        "text_transform"
    }
    
    fn get_parameter_descriptions(&self) -> Vec<(String, String, bool)> {
        vec![
            ("field".to_string(), "要处理的字段名".to_string(), true),
            ("transform_type".to_string(), "转换类型：upper, lower, trim, capitalize".to_string(), true),
            ("create_new_field".to_string(), "是否创建新字段（true/false）".to_string(), false),
            ("new_field_name".to_string(), "新字段名称".to_string(), false),
        ]
    }
}

/// 文本替换处理器工厂
#[derive(Debug, Clone)]
pub struct TextReplaceProcessorFactory;

impl ProcessorFactory for TextReplaceProcessorFactory {
    fn create(&self, params: HashMap<String, String>) -> Result<Box<dyn Processor>> {
        let field = params.get("field").ok_or_else(|| Error::MissingParameter("field".to_string()))?.clone();
        let pattern = params.get("pattern").ok_or_else(|| Error::MissingParameter("pattern".to_string()))?.clone();
        let replacement = params.get("replacement").ok_or_else(|| Error::MissingParameter("replacement".to_string()))?.clone();
        
        let use_regex = params.get("use_regex")
            .map(|v| v.to_lowercase() == "true")
            .unwrap_or(false);
        
        let create_new_field = params.get("create_new_field")
            .map(|v| v.to_lowercase() == "true")
            .unwrap_or(false);
        
        let new_field_name = params.get("new_field_name").cloned();
        
        Ok(Box::new(TextReplaceProcessor::new(
            field,
            pattern,
            replacement,
            use_regex,
            create_new_field,
            new_field_name,
        )?))
    }
    
    fn get_name(&self) -> &'static str {
        "text_replace"
    }
    
    fn get_parameter_descriptions(&self) -> Vec<(String, String, bool)> {
        vec![
            ("field".to_string(), "要处理的字段名".to_string(), true),
            ("pattern".to_string(), "匹配模式".to_string(), true),
            ("replacement".to_string(), "替换内容".to_string(), true),
            ("use_regex".to_string(), "是否使用正则表达式（true/false）".to_string(), false),
            ("create_new_field".to_string(), "是否创建新字段（true/false）".to_string(), false),
            ("new_field_name".to_string(), "新字段名称".to_string(), false),
        ]
    }
}

/// 文本分割处理器工厂
#[derive(Debug, Clone)]
pub struct TextSplitProcessorFactory;

impl ProcessorFactory for TextSplitProcessorFactory {
    fn create(&self, params: HashMap<String, String>) -> Result<Box<dyn Processor>> {
        let field = params.get("field").ok_or_else(|| Error::MissingParameter("field".to_string()))?.clone();
        let separator = params.get("separator").ok_or_else(|| Error::MissingParameter("separator".to_string()))?.clone();
        
        let max_splits = params.get("max_splits")
            .and_then(|v| v.parse::<usize>().ok());
        
        let keep_original = params.get("keep_original")
            .map(|v| v.to_lowercase() == "true")
            .unwrap_or(true);
        
        let target_field = params.get("target_field").cloned();
        
        Ok(Box::new(TextSplitProcessor::new(
            field,
            separator,
            max_splits,
            keep_original,
            target_field,
        )))
    }
    
    fn get_name(&self) -> &'static str {
        "text_split"
    }
    
    fn get_parameter_descriptions(&self) -> Vec<(String, String, bool)> {
        vec![
            ("field".to_string(), "要分割的字段名".to_string(), true),
            ("separator".to_string(), "分隔符".to_string(), true),
            ("max_splits".to_string(), "最大分割数".to_string(), false),
            ("keep_original".to_string(), "是否保留原始字段（true/false）".to_string(), false),
            ("target_field".to_string(), "目标字段名".to_string(), false),
        ]
    }
}

/// 文本连接处理器工厂
#[derive(Debug, Clone)]
pub struct TextJoinProcessorFactory;

impl ProcessorFactory for TextJoinProcessorFactory {
    fn create(&self, params: HashMap<String, String>) -> Result<Box<dyn Processor>> {
        let field = params.get("field").ok_or_else(|| Error::MissingParameter("field".to_string()))?.clone();
        let delimiter = params.get("delimiter").ok_or_else(|| Error::MissingParameter("delimiter".to_string()))?.clone();
        
        let keep_original = params.get("keep_original")
            .map(|v| v.to_lowercase() == "true")
            .unwrap_or(true);
        
        let target_field = params.get("target_field").cloned();
        
        Ok(Box::new(TextJoinProcessor::new(
            field,
            delimiter,
            keep_original,
            target_field,
        )))
    }
    
    fn get_name(&self) -> &'static str {
        "text_join"
    }
    
    fn get_parameter_descriptions(&self) -> Vec<(String, String, bool)> {
        vec![
            ("field".to_string(), "要连接的数组字段名".to_string(), true),
            ("delimiter".to_string(), "连接符".to_string(), true),
            ("keep_original".to_string(), "是否保留原始字段（true/false）".to_string(), false),
            ("target_field".to_string(), "目标字段名".to_string(), false),
        ]
    }
}

/// 注册所有文本处理器
pub fn register_processors(registry: &mut HashMap<String, Box<dyn ProcessorFactory>>) {
    registry.insert("text_transform".to_string(), Box::new(TextTransformProcessorFactory));
    registry.insert("text_replace".to_string(), Box::new(TextReplaceProcessorFactory));
    registry.insert("text_split".to_string(), Box::new(TextSplitProcessorFactory));
    registry.insert("text_join".to_string(), Box::new(TextJoinProcessorFactory));
} 