// 自定义格式选项
// Custom format options

use std::collections::HashMap;
use crate::data::loader::file::formats::{FormatType, FormatOptions};

/// 自定义格式选项
/// Custom format options
#[derive(Debug, Clone, Default)]
pub struct CustomOptions {
    /// 处理器类名
    /// Processor class name
    pub processor_class: String,
    
    /// 处理器路径
    /// Processor path
    pub processor_path: Option<String>,
    
    /// 解析器名称
    /// Parser name
    pub parser_name: Option<String>,
    
    /// 自定义参数
    /// Custom parameters
    pub parameters: HashMap<String, String>,
    
    /// 是否缓存处理器
    /// Whether to cache processor
    pub cache_processor: bool,
}

impl CustomOptions {
    /// 创建新的自定义格式选项
    /// Create new custom format options
    pub fn new(processor_class: impl Into<String>) -> Self {
        Self {
            processor_class: processor_class.into(),
            processor_path: None,
            parser_name: None,
            parameters: HashMap::new(),
            cache_processor: true,
        }
    }
    
    /// 设置处理器类名
    /// Set processor class name
    pub fn with_processor_class(mut self, processor_class: impl Into<String>) -> Self {
        self.processor_class = processor_class.into();
        self
    }
    
    /// 设置处理器路径
    /// Set processor path
    pub fn with_processor_path(mut self, processor_path: impl Into<String>) -> Self {
        self.processor_path = Some(processor_path.into());
        self
    }
    
    /// 设置解析器名称
    /// Set parser name
    pub fn with_parser_name(mut self, parser_name: impl Into<String>) -> Self {
        self.parser_name = Some(parser_name.into());
        self
    }
    
    /// 添加自定义参数
    /// Add custom parameter
    pub fn with_parameter(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }
    
    /// 添加多个自定义参数
    /// Add multiple custom parameters
    pub fn with_parameters(mut self, parameters: HashMap<String, String>) -> Self {
        self.parameters.extend(parameters);
        self
    }
    
    /// 设置是否缓存处理器
    /// Set whether to cache processor
    pub fn with_cache_processor(mut self, cache_processor: bool) -> Self {
        self.cache_processor = cache_processor;
        self
    }
    
    /// 获取参数值
    /// Get parameter value
    pub fn get_parameter(&self, key: &str) -> Option<&String> {
        self.parameters.get(key)
    }
    
    /// 获取参数值，如果不存在则返回默认值
    /// Get parameter value, return default value if not exists
    pub fn get_parameter_or(&self, key: &str, default_value: &str) -> String {
        self.parameters.get(key).map(|v| v.clone()).unwrap_or_else(|| default_value.to_string())
    }
}

impl FormatOptions for CustomOptions {
    fn format_type(&self) -> FormatType {
        FormatType::Custom
    }
    
    fn to_options_map(&self) -> HashMap<String, String> {
        let mut options = HashMap::new();
        
        options.insert("format".to_string(), "custom".to_string());
        options.insert("processor_class".to_string(), self.processor_class.clone());
        
        if let Some(processor_path) = &self.processor_path {
            options.insert("processor_path".to_string(), processor_path.clone());
        }
        
        if let Some(parser_name) = &self.parser_name {
            options.insert("parser_name".to_string(), parser_name.clone());
        }
        
        options.insert("cache_processor".to_string(), self.cache_processor.to_string());
        
        // 添加所有自定义参数，以param_前缀区分
        for (key, value) in &self.parameters {
            options.insert(format!("param_{}", key), value.clone());
        }
        
        options
    }
    
    fn from_options_map(options: &HashMap<String, String>) -> Self {
        let processor_class = options.get("processor_class")
            .cloned()
            .unwrap_or_else(|| "DefaultCustomProcessor".to_string());
            
        let mut custom_options = Self::new(processor_class);
        
        if let Some(processor_path) = options.get("processor_path") {
            custom_options.processor_path = Some(processor_path.clone());
        }
        
        if let Some(parser_name) = options.get("parser_name") {
            custom_options.parser_name = Some(parser_name.clone());
        }
        
        if let Some(cache_processor) = options.get("cache_processor") {
            if let Ok(value) = cache_processor.parse::<bool>() {
                custom_options.cache_processor = value;
            }
        }
        
        // 提取所有以param_开头的自定义参数
        for (key, value) in options {
            if key.starts_with("param_") {
                let param_key = key.strip_prefix("param_").unwrap_or(key);
                custom_options.parameters.insert(param_key.to_string(), value.clone());
            }
        }
        
        custom_options
    }
} 