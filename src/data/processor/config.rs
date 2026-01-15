// Data Processor Configuration
// 数据处理器配置

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::data::processor::types::ColumnInfo;
use crate::data::schema::DataSchema;

/// 字段验证规则
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldRule {
    /// 字段名
    pub field_name: String,
    /// 是否必需
    pub required: bool,
    /// 数据类型验证
    pub data_type: Option<String>,
    /// 最小值/长度
    pub min_value: Option<f64>,
    /// 最大值/长度
    pub max_value: Option<f64>,
    /// 正则表达式模式
    pub pattern: Option<String>,
    /// 允许的值列表
    pub allowed_values: Option<Vec<String>>,
    /// 自定义验证函数名
    pub custom_validator: Option<String>,
}

impl Default for FieldRule {
    fn default() -> Self {
        Self {
            field_name: String::new(),
            required: false,
            data_type: None,
            min_value: None,
            max_value: None,
            pattern: None,
            allowed_values: None,
            custom_validator: None,
        }
    }
}

/// 数据转换配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTransformation {
    /// 转换类型
    pub transform_type: String,
    /// 源字段
    pub source_field: String,
    /// 目标字段
    pub target_field: String,
    /// 转换参数
    pub parameters: HashMap<String, serde_json::Value>,
    /// 是否启用
    pub enabled: bool,
}

impl Default for DataTransformation {
    fn default() -> Self {
        Self {
            transform_type: "identity".to_string(),
            source_field: String::new(),
            target_field: String::new(),
            parameters: HashMap::new(),
            enabled: true,
        }
    }
}

/// 处理器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    // 基本设置
    pub batch_size: usize,
    pub feature_columns: Vec<String>,
    pub label_column: Option<String>,
    pub skip_header: bool,
    pub delimiter: char,
    pub missing_value: String,
    
    // 数据清洗设置
    pub clean_special_chars: bool,
    pub normalize_case: bool,
    pub handle_outliers: bool,
    pub min_value: f64,
    pub max_value: f64,
    /// 最大文件大小（字节）
    pub max_file_size: Option<usize>,
    
    // 列信息
    pub column_info: Vec<ColumnInfo>,
    
    // 转换设置
    pub transformers: HashMap<String, TransformerConfig>,
    
    pub normalize: bool,
    pub remove_duplicates: bool,
    pub fill_missing: bool,
    
    // 新增字段用于API兼容性
    pub process_type: String,
    pub options: Option<HashMap<String, serde_json::Value>>,
    
    // 数据处理选项
    pub enable_compression: Option<bool>,
    pub enable_encryption: Option<bool>,
    pub enable_validation: Option<bool>,
    
    // 编码设置
    pub input_encoding: Option<String>,
    pub output_encoding: Option<String>,
    
    // 新增的缺失字段
    /// 数据集名称
    pub dataset_name: String,
    /// 是否启用验证
    pub validate: bool,
    /// 是否只有在所有数据都有效时才处理
    pub process_only_if_all_valid: bool,
    /// 字段验证规则
    pub field_rules: Vec<FieldRule>,
    /// 数据转换配置
    pub transformations: Vec<DataTransformation>,
    /// 数据模式
    pub schema: Option<DataSchema>,
    /// 处理器选项
    pub processor_options: HashMap<String, String>,
    /// API兼容性字段 - 处理缺失值
    pub handle_missing_values: bool,
    /// API兼容性字段 - 标准化数字
    pub normalize_numbers: bool,
    /// API兼容性字段 - 转换为小写
    pub convert_to_lowercase: bool,
    /// API兼容性字段 - 编码格式
    pub encoding: Option<String>,
    /// API兼容性字段 - 日期格式
    pub date_format: Option<String>,
    /// API兼容性字段 - 小数分隔符
    pub decimal_separator: Option<String>,
    /// API兼容性字段 - 千位分隔符
    pub thousands_separator: Option<String>,
    /// API兼容性字段 - 布尔真值
    pub boolean_true_values: Option<Vec<String>>,
    /// API兼容性字段 - 布尔假值
    pub boolean_false_values: Option<Vec<String>>,
    /// API兼容性字段 - 空值标识
    pub null_values: Option<Vec<String>>,
    /// API兼容性字段 - 是否修剪空白
    pub trim_whitespace: bool,
    /// API兼容性字段 - 最大字段大小
    pub max_field_size: Option<usize>,
    /// API兼容性字段 - 是否验证数据类型
    pub validate_data_types: bool,
    /// API兼容性字段 - 是否跳过空行
    pub skip_empty_lines: bool,
    /// API兼容性字段 - 注释字符
    pub comment_char: Option<String>,
    /// API兼容性字段 - 转义字符
    pub escape_char: Option<String>,
    /// API兼容性字段 - 引用字符
    pub quote_char: Option<String>,
    /// API兼容性字段 - 字段分隔符
    pub field_delimiter: Option<String>,
    /// API兼容性字段 - 记录分隔符
    pub record_delimiter: Option<String>,
    /// API兼容性字段 - 是否忽略解析错误
    pub ignore_parse_errors: bool,
    /// API兼容性字段 - 最大解析错误数
    pub max_parse_errors: Option<usize>,
    /// API兼容性字段 - 默认字符串值
    pub default_string_value: Option<String>,
    /// API兼容性字段 - 默认数字值
    pub default_number_value: Option<f64>,
    /// API兼容性字段 - 默认布尔值
    pub default_boolean_value: Option<bool>,
    /// API兼容性字段 - 时区
    pub timezone: Option<String>,
}

/// 转换器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransformerConfig {
    Numeric {
        scale: f32,
        offset: f32,
    },
    Categorical {
        categories: Vec<String>,
    },
    DateTime {
        format: String,
        reference_date: String,
    },
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            feature_columns: Vec::new(),
            label_column: None,
            skip_header: true,
            delimiter: ',',
            missing_value: "NULL".to_string(),
            clean_special_chars: true,
            normalize_case: false,
            handle_outliers: true,
            min_value: f64::MIN,
            max_value: f64::MAX,
            max_file_size: None,
            column_info: Vec::new(),
            transformers: HashMap::new(),
            normalize: true,
            remove_duplicates: true,
            fill_missing: true,
            process_type: "default".to_string(),
            options: None,
            enable_compression: None,
            enable_encryption: None,
            enable_validation: None,
            input_encoding: None,
            output_encoding: None,
            // 新增字段的默认值
            dataset_name: String::new(),
            validate: true,
            process_only_if_all_valid: false,
            field_rules: Vec::new(),
            transformations: Vec::new(),
            schema: None,
            processor_options: HashMap::new(),
            handle_missing_values: true,
            normalize_numbers: false,
            convert_to_lowercase: false,
            encoding: None,
            date_format: None,
            decimal_separator: None,
            thousands_separator: None,
            boolean_true_values: None,
            boolean_false_values: None,
            null_values: None,
            trim_whitespace: true,
            max_field_size: None,
            validate_data_types: true,
            skip_empty_lines: false,
            comment_char: None,
            escape_char: None,
            quote_char: None,
            field_delimiter: None,
            record_delimiter: None,
            ignore_parse_errors: false,
            max_parse_errors: None,
            default_string_value: None,
            default_number_value: None,
            default_boolean_value: None,
            timezone: None,
        }
    }
}

impl ProcessorConfig {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// 从HashMap创建ProcessorConfig
    pub fn from_map(map: HashMap<String, String>) -> Result<Self, crate::error::Error> {
        let mut config = Self::default();
        
        for (key, value) in map {
            match key.as_str() {
                "batch_size" => {
                    config.batch_size = value.parse().map_err(|e| 
                        crate::error::Error::invalid_argument(format!("无效的batch_size: {}", e)))?;
                },
                "skip_header" => {
                    config.skip_header = value.parse().map_err(|e| 
                        crate::error::Error::invalid_argument(format!("无效的skip_header: {}", e)))?;
                },
                "delimiter" => {
                    config.delimiter = value.chars().next().unwrap_or(',');
                },
                "missing_value" => {
                    config.missing_value = value;
                },
                "clean_special_chars" => {
                    config.clean_special_chars = value.parse().map_err(|e| 
                        crate::error::Error::invalid_argument(format!("无效的clean_special_chars: {}", e)))?;
                },
                "normalize_case" => {
                    config.normalize_case = value.parse().map_err(|e| 
                        crate::error::Error::invalid_argument(format!("无效的normalize_case: {}", e)))?;
                },
                "handle_outliers" => {
                    config.handle_outliers = value.parse().map_err(|e| 
                        crate::error::Error::invalid_argument(format!("无效的handle_outliers: {}", e)))?;
                },
                "normalize" => {
                    config.normalize = value.parse().map_err(|e| 
                        crate::error::Error::invalid_argument(format!("无效的normalize: {}", e)))?;
                },
                "remove_duplicates" => {
                    config.remove_duplicates = value.parse().map_err(|e| 
                        crate::error::Error::invalid_argument(format!("无效的remove_duplicates: {}", e)))?;
                },
                "fill_missing" => {
                    config.fill_missing = value.parse().map_err(|e| 
                        crate::error::Error::invalid_argument(format!("无效的fill_missing: {}", e)))?;
                },
                "process_type" => {
                    config.process_type = value;
                },
                "dataset_name" => {
                    config.dataset_name = value;
                },
                "validate" => {
                    config.validate = value.parse().map_err(|e| 
                        crate::error::Error::invalid_argument(format!("无效的validate: {}", e)))?;
                },
                "process_only_if_all_valid" => {
                    config.process_only_if_all_valid = value.parse().map_err(|e| 
                        crate::error::Error::invalid_argument(format!("无效的process_only_if_all_valid: {}", e)))?;
                },
                "handle_missing_values" => {
                    config.handle_missing_values = value.parse().map_err(|e| 
                        crate::error::Error::invalid_argument(format!("无效的handle_missing_values: {}", e)))?;
                },
                "normalize_numbers" => {
                    config.normalize_numbers = value.parse().map_err(|e| 
                        crate::error::Error::invalid_argument(format!("无效的normalize_numbers: {}", e)))?;
                },
                "convert_to_lowercase" => {
                    config.convert_to_lowercase = value.parse().map_err(|e| 
                        crate::error::Error::invalid_argument(format!("无效的convert_to_lowercase: {}", e)))?;
                },
                "trim_whitespace" => {
                    config.trim_whitespace = value.parse().map_err(|e| 
                        crate::error::Error::invalid_argument(format!("无效的trim_whitespace: {}", e)))?;
                },
                "validate_data_types" => {
                    config.validate_data_types = value.parse().map_err(|e| 
                        crate::error::Error::invalid_argument(format!("无效的validate_data_types: {}", e)))?;
                },
                "skip_empty_lines" => {
                    config.skip_empty_lines = value.parse().map_err(|e| 
                        crate::error::Error::invalid_argument(format!("无效的skip_empty_lines: {}", e)))?;
                },
                "ignore_parse_errors" => {
                    config.ignore_parse_errors = value.parse().map_err(|e| 
                        crate::error::Error::invalid_argument(format!("无效的ignore_parse_errors: {}", e)))?;
                },
                "encoding" => {
                    config.encoding = Some(value);
                },
                "date_format" => {
                    config.date_format = Some(value);
                },
                "decimal_separator" => {
                    config.decimal_separator = Some(value);
                },
                "thousands_separator" => {
                    config.thousands_separator = Some(value);
                },
                "comment_char" => {
                    config.comment_char = Some(value);
                },
                "escape_char" => {
                    config.escape_char = Some(value);
                },
                "quote_char" => {
                    config.quote_char = Some(value);
                },
                "field_delimiter" => {
                    config.field_delimiter = Some(value);
                },
                "record_delimiter" => {
                    config.record_delimiter = Some(value);
                },
                "default_string_value" => {
                    config.default_string_value = Some(value);
                },
                "timezone" => {
                    config.timezone = Some(value);
                },
                _ => {
                    // 对于未知的键，添加到processor_options中
                    config.processor_options.insert(key, value);
                }
            }
        }
        
        config.validate_config()?;
        Ok(config)
    }
    
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }
    
    pub fn with_feature_columns(mut self, columns: Vec<String>) -> Self {
        self.feature_columns = columns;
        self
    }
    
    pub fn with_label_column(mut self, column: Option<String>) -> Self {
        self.label_column = column;
        self
    }
    
    pub fn with_column_info(mut self, info: Vec<ColumnInfo>) -> Self {
        self.column_info = info;
        self
    }
    
    pub fn add_transformer(mut self, column: String, config: TransformerConfig) -> Self {
        self.transformers.insert(column, config);
        self
    }
    
    pub fn enable_outlier_handling(mut self, min: f64, max: f64) -> Self {
        self.handle_outliers = true;
        self.min_value = min;
        self.max_value = max;
        self
    }
    
    pub fn disable_outlier_handling(mut self) -> Self {
        self.handle_outliers = false;
        self
    }
    
    pub fn enable_special_chars_cleaning(mut self) -> Self {
        self.clean_special_chars = true;
        self
    }
    
    pub fn disable_special_chars_cleaning(mut self) -> Self {
        self.clean_special_chars = false;
        self
    }
    
    pub fn enable_case_normalization(mut self) -> Self {
        self.normalize_case = true;
        self
    }
    
    pub fn disable_case_normalization(mut self) -> Self {
        self.normalize_case = false;
        self
    }
    
    /// 添加字段验证规则
    pub fn add_field_rule(mut self, rule: FieldRule) -> Self {
        self.field_rules.push(rule);
        self
    }
    
    /// 添加数据转换
    pub fn add_transformation(mut self, transformation: DataTransformation) -> Self {
        self.transformations.push(transformation);
        self
    }
    
    /// 设置数据模式
    pub fn with_schema(mut self, schema: DataSchema) -> Self {
        self.schema = Some(schema);
        self
    }
    
    /// 启用验证
    pub fn enable_validation(mut self) -> Self {
        self.validate = true;
        self
    }
    
    /// 禁用验证
    pub fn disable_validation(mut self) -> Self {
        self.validate = false;
        self
    }
    
    /// 设置数据集名称
    pub fn with_dataset_name(mut self, name: String) -> Self {
        self.dataset_name = name;
        self
    }
    
    /// 设置数据集名称（字符串引用）
    pub fn with_dataset_name_str(mut self, name: &str) -> Self {
        self.dataset_name = name.to_string();
        self
    }
    
    /// 设置只有在所有数据都有效时才处理
    pub fn require_all_valid(mut self, require: bool) -> Self {
        self.process_only_if_all_valid = require;
        self
    }
    
    /// 添加处理器选项
    pub fn add_processor_option(mut self, key: &str, value: &str) -> Self {
        self.processor_options.insert(key.to_string(), value.to_string());
        self
    }
    
    /// 验证配置的完整性
    pub fn validate_config(&self) -> Result<(), String> {
        // 验证基本设置
        if self.batch_size == 0 {
            return Err("批次大小不能为0".to_string());
        }
        
        // 验证字段规则
        for rule in &self.field_rules {
            if rule.field_name.is_empty() {
                return Err("字段规则必须有字段名".to_string());
            }
        }
        
        // 验证转换配置
        for transformation in &self.transformations {
            if transformation.source_field.is_empty() {
                return Err("转换配置必须有源字段".to_string());
            }
            if transformation.target_field.is_empty() {
                return Err("转换配置必须有目标字段".to_string());
            }
        }
        
        Ok(())
    }
} 