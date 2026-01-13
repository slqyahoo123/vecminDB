// 格式定义模块
// Format definitions module

// 导入子模块
pub mod avro;
pub mod parquet;
pub mod custom;

// 重新导出
pub use avro::*;
pub use parquet::*;
pub use custom::*;

/// 数据格式类型
/// Data format type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatType {
    /// CSV格式
    CSV,
    /// JSON格式
    JSON,
    /// Parquet格式
    Parquet,
    /// Excel格式
    Excel,
    /// Avro格式
    Avro,
    /// 自定义格式
    Custom,
    /// 未知格式
    Unknown,
}

impl Default for FormatType {
    fn default() -> Self {
        FormatType::Unknown
    }
}

impl FormatType {
    /// 获取格式类型的字符串表示
    /// Get string representation of format type
    pub fn as_str(&self) -> &'static str {
        match self {
            FormatType::CSV => "CSV",
            FormatType::JSON => "JSON",
            FormatType::Parquet => "Parquet",
            FormatType::Excel => "Excel",
            FormatType::Avro => "Avro",
            FormatType::Custom => "Custom",
            FormatType::Unknown => "Unknown",
        }
    }
    
    /// 从字符串转换为格式类型
    /// Convert from string to format type
    pub fn from_str(format_str: &str) -> Self {
        match format_str.to_lowercase().as_str() {
            "csv" => FormatType::CSV,
            "json" => FormatType::JSON,
            "parquet" => FormatType::Parquet,
            "excel" | "xlsx" | "xls" => FormatType::Excel,
            "avro" => FormatType::Avro,
            "custom" => FormatType::Custom,
            _ => FormatType::Unknown,
        }
    }
}

/// 格式选项接口
/// Format options interface
pub trait FormatOptions {
    /// 获取格式类型
    /// Get format type
    fn format_type(&self) -> FormatType;
    
    /// 转换为选项映射
    /// Convert to options map
    fn to_options_map(&self) -> std::collections::HashMap<String, String>;
    
    /// 从选项映射创建
    /// Create from options map
    fn from_options_map(options: &std::collections::HashMap<String, String>) -> Self where Self: Sized;
} 